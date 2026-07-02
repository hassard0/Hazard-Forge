// Slice SC1 (THE REAL-SPONZA HERO BAKE) — the EXTERNAL-image-URI loader seam, unit-tested.
//
// Multi-file .gltf assets (the Khronos PBR Sponza) reference their images as file URIs relative to
// the .gltf's own directory instead of embedding them in a .glb buffer_view. The loader's DecodeImage
// resolves those URIs through two PURE helpers (hf::asset::PercentDecodeUri +
// hf::asset::ReadUriBytesRelativeTo) and feeds the SAME stbi_load_from_memory decode the embedded
// path uses — so "URI bytes == the bytes that would have been embedded" IS the equivalence proof.
//
// This test creates its own fixture at a temp dir (nothing committed): a tiny 2x2 RGBA PNG whose
// FILENAME CONTAINS A SPACE plus a minimal .gltf referencing it with a %20-escaped URI. It parses the
// .gltf with cgltf (declarations only — the IMPLEMENTATION lives in hf_core's gltf_loader.cpp), pulls
// the authored image->uri, resolves it with the REAL helper, and asserts:
//   1. the resolved bytes are byte-identical to the PNG the test wrote (== what embedding would feed),
//   2. stbi_load_from_memory over those bytes yields the exact authored 2x2 pixels (the same decoder
//      call DecodeImage makes — proving the decoded texture matches the embedded route's), and
//   3. the fallback contract holds: data:/scheme'd URIs and missing files return empty (no throw).
// Plus PercentDecodeUri unit cases. Pure CPU (hf_core), no RHI/device, ASan-eligible.

#include "asset/gltf_loader.h"
#include "cgltf/cgltf.h"       // declarations only (no CGLTF_IMPLEMENTATION — it lives in hf_core)
#include "stb/stb_image.h"     // declarations only (no STB_IMAGE_IMPLEMENTATION — same TU rule)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::asset;
namespace fs = std::filesystem;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS: %s\n", what); }
}

// A tiny VALID 2x2 RGBA8 PNG (75 bytes): pixels row0 = red, green; row1 = blue, white.
static const uint8_t kPng2x2[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x08, 0x06, 0x00, 0x00, 0x00, 0x72, 0xb6, 0x0d,
    0x24, 0x00, 0x00, 0x00, 0x12, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xf0,
    0x1f, 0x0c, 0x81, 0x34, 0x18, 0x00, 0x00, 0x49, 0xc8, 0x09, 0xf7, 0xf9, 0xab, 0xb6, 0x0d, 0x00,
    0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

int main() {
    HF_TEST_MAIN_INIT();

    // ---- PercentDecodeUri: pure unit cases ------------------------------------------------------
    check(PercentDecodeUri("plain123.jpg") == "plain123.jpg", "percent-decode: plain name unchanged");
    check(PercentDecodeUri("tex%202x2.png") == "tex 2x2.png", "percent-decode: %20 -> space");
    check(PercentDecodeUri("a%2Bb%2fc") == "a+b/c", "percent-decode: %2B/%2f (mixed case hex)");
    check(PercentDecodeUri("bad%G1%2") == "bad%G1%2", "percent-decode: malformed escapes verbatim");
    check(PercentDecodeUri("") == "", "percent-decode: empty");
    check(PercentDecodeUri(nullptr) == "", "percent-decode: null");

    // ---- Fixture: a temp dir holding "tex 2x2.png" + a minimal .gltf with a %20-escaped URI ------
    const fs::path dir = fs::temp_directory_path() / "hf_sc1_uri_fixture";
    std::error_code ec;
    fs::create_directories(dir, ec);
    check(!ec, "fixture: temp dir created");

    const fs::path pngPath = dir / "tex 2x2.png";
    {
        std::ofstream f(pngPath, std::ios::binary);
        f.write(reinterpret_cast<const char*>(kPng2x2), sizeof(kPng2x2));
        check(f.good(), "fixture: 2x2 png written");
    }
    const fs::path gltfPath = dir / "uri_image.gltf";
    {
        // Minimal, images-only glTF: enough for cgltf_parse_file to hand back the authored uri.
        const char* json =
            "{\"asset\":{\"version\":\"2.0\"},\"images\":[{\"uri\":\"tex%202x2.png\"}]}";
        std::ofstream f(gltfPath, std::ios::binary);
        f << json;
        check(f.good(), "fixture: .gltf written");
    }

    // ---- Parse with cgltf (the engine's parser) and resolve the REAL authored uri ---------------
    const std::string gltfPathStr = gltfPath.string();
    cgltf_options options{};
    cgltf_data* data = nullptr;
    cgltf_result res = cgltf_parse_file(&options, gltfPathStr.c_str(), &data);
    check(res == cgltf_result_success && data, "cgltf: fixture .gltf parses");
    if (res == cgltf_result_success && data) {
        res = cgltf_load_buffers(&options, data, gltfPathStr.c_str());
        check(res == cgltf_result_success, "cgltf: load_buffers succeeds (no buffers)");
        check(data->images_count == 1 && data->images[0].uri &&
                  std::strcmp(data->images[0].uri, "tex%202x2.png") == 0,
              "cgltf: image[0] is the escaped external uri, NOT a buffer_view");
        check(data->images_count == 1 && data->images[0].buffer_view == nullptr,
              "cgltf: image[0] has no buffer_view (the pre-SC1 fallback trigger)");

        // 1. The resolved bytes are byte-identical to the file the test wrote — i.e. exactly the
        //    bytes an embedded buffer_view of this image would have carried into the shared decode.
        std::vector<uint8_t> got = ReadUriBytesRelativeTo(gltfPathStr.c_str(), data->images[0].uri);
        check(got.size() == sizeof(kPng2x2) &&
                  std::memcmp(got.data(), kPng2x2, sizeof(kPng2x2)) == 0,
              "ReadUriBytesRelativeTo: uri bytes == the embedded-equivalent png bytes");

        // 2. The SAME decoder call DecodeImage makes yields the authored pixels.
        if (!got.empty()) {
            int w = 0, h = 0, comp = 0;
            stbi_uc* px = stbi_load_from_memory(got.data(), (int)got.size(), &w, &h, &comp, 4);
            check(px != nullptr && w == 2 && h == 2, "stbi: uri bytes decode to a 2x2 image");
            if (px && w == 2 && h == 2) {
                const uint8_t expect[16] = {255, 0, 0, 255,  0, 255, 0, 255,
                                            0, 0, 255, 255,  255, 255, 255, 255};
                check(std::memcmp(px, expect, 16) == 0,
                      "stbi: decoded pixels match the authored red/green/blue/white 2x2");
            }
            if (px) stbi_image_free(px);
        }
        cgltf_free(data);
    }

    // ---- 3. Fallback contract: non-file URIs and missing files return empty (never throw) -------
    check(ReadUriBytesRelativeTo(gltfPathStr.c_str(), "data:image/png;base64,AAAA").empty(),
          "fallback: data: uri returns empty");
    check(ReadUriBytesRelativeTo(gltfPathStr.c_str(), "https://example.com/x.png").empty(),
          "fallback: scheme'd uri returns empty");
    check(ReadUriBytesRelativeTo(gltfPathStr.c_str(), "no_such_file.png").empty(),
          "fallback: missing file returns empty");
    check(ReadUriBytesRelativeTo(nullptr, "tex.png").empty(), "fallback: null gltf path");
    check(ReadUriBytesRelativeTo(gltfPathStr.c_str(), nullptr).empty(), "fallback: null uri");

    // Cleanup (best-effort; a leftover temp dir is harmless).
    fs::remove_all(dir, ec);

    if (g_fail == 0) std::printf("gltf_uri_image_test: ALL PASS\n");
    else             std::printf("gltf_uri_image_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
