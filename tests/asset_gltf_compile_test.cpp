// Unit test for the glTF FORMAT-COVERAGE closer (engine/asset/gltf_compile.h, issue #16).
//
// Closes the deterministic content-addressed asset pipeline's last format gap: the pipeline compiled ONLY
// OBJ; CompileGltf adds glTF / glb -> the EXACT SAME canonical Q16.16 CompiledMesh blob CompileObj emits.
//
// THE HEADLINE (proven here): a self-contained single-triangle glTF (an embedded base64 data-URI buffer)
// with positions (0,0,0)/(1,0,0)/(0,1,0) + indices {0,1,2} compiles, under the SAME params, to the
// BYTE-IDENTICAL blob the S1 OBJ fixture of the SAME geometry produces — glTF and OBJ of equivalent
// geometry yield the same artifact. DETERMINISTIC: same bytes in -> byte-identical blob; a changed vertex
// changes the digest. ALL existing asset_compiler goldens (S1..S6) are re-asserted UNCHANGED.
//
// CGLTF: cgltf's CGLTF_IMPLEMENTATION lives in EXACTLY ONE TU. Under CMake this test links hf_core, whose
// gltf_loader.cpp already provides it -> the CMake build defines HF_GLTF_TEST_NO_IMPL to suppress a second
// copy (a duplicate-symbol link error otherwise). A PURE standalone clang compile (no hf_core) does NOT
// define it, so this TU emits the implementation itself:
//   clang++ -std=c++20 -I engine -I tests -I third_party tests/asset_gltf_compile_test.cpp -o agc
//
// NOTE: cgltf's implementation block lives OUTSIDE its include guard, so the header must be #included
// EXACTLY ONCE with CGLTF_IMPLEMENTATION active. We therefore do NOT #include "cgltf/cgltf.h" directly —
// gltf_compile.h is the single includer; we just set the macro before it (standalone) or leave it unset
// (CMake / hf_core build, where gltf_loader.cpp already emitted the implementation).
#ifndef HF_GLTF_TEST_NO_IMPL
#  define CGLTF_IMPLEMENTATION
#endif

#include "asset/asset_compiler.h"
#include "asset/gltf_compile.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"

using namespace hf::asset;
namespace net = hf::net;

static int g_fail = 0;
static void check(bool c, const char* what) {
    if (!c) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else    { std::printf("PASS: %s\n", what); }
}

// --- The minimal self-contained glTF fixture (FIXED forever — the golden pins its compiled digest) --------
// A single triangle: POSITION (0,0,0),(1,0,0),(0,1,0) + indices {0,1,2}, equivalent to the S1 OBJ fixture's
// geometry. The buffer is an EMBEDDED base64 data: URI (44 bytes: 3*vec3 float32 = 36, then 3*uint16 = 6,
// padded to 8) so the file is fully self-contained — NO external .bin, parses from memory with no path.
// Keep FIXED forever (the digest is pinned over its compiled blob).
static const char* kTriangleGltf =
    "{\n"
    "  \"asset\": { \"version\": \"2.0\" },\n"
    "  \"buffers\": [ { \"byteLength\": 44, \"uri\": \"data:application/octet-stream;base64,"
    "AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAABAAIAAAA=\" } ],\n"
    "  \"bufferViews\": [\n"
    "    { \"buffer\": 0, \"byteOffset\": 0,  \"byteLength\": 36, \"target\": 34962 },\n"
    "    { \"buffer\": 0, \"byteOffset\": 36, \"byteLength\": 6,  \"target\": 34963 }\n"
    "  ],\n"
    "  \"accessors\": [\n"
    "    { \"bufferView\": 0, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\","
    " \"min\": [0,0,0], \"max\": [1,1,0] },\n"
    "    { \"bufferView\": 1, \"componentType\": 5123, \"count\": 3, \"type\": \"SCALAR\" }\n"
    "  ],\n"
    "  \"meshes\": [ { \"primitives\": [ { \"attributes\": { \"POSITION\": 0 }, \"indices\": 1 } ] } ]\n"
    "}\n";

static std::size_t StrLen(const char* s) { std::size_t n = 0; while (s[n] != '\0') ++n; return n; }

int main() {
    HF_TEST_MAIN_INIT();

    const uint8_t* gbytes = (const uint8_t*)kTriangleGltf;
    const std::size_t glen = StrLen(kTriangleGltf);
    const CompileParams params = ShowcaseParams();   // scale = 2.0 in Q16.16 (the S1/S2 showcase params)

    // The compiled glTF artifact + its digest.
    const std::vector<uint8_t> gblob = CompileGltf(gbytes, glen, params);
    const CacheKey gdigest = DigestArtifact(gblob);
    std::printf("asset-gltf: compiled-artifact digest = 0x%016llx  (%zu bytes)\n",
                (unsigned long long)gdigest, gblob.size());

    // (G-1) PARSE + HEADER — the blob decodes to magic/version, vertexCount 3, indexCount 3.
    {
        CompiledMesh out;
        const bool ok = DecodeCompiledMesh(gblob, out);
        check(ok && out.magic == kCompiledMeshMagic && out.version == kCompiledMeshVersion &&
              out.vertexCount == 3 && out.indexCount == 3,
              "asset-gltf: CompileGltf(triangle) decodes — magic/version OK, vertexCount 3, indexCount 3");
    }

    // (G-2) PINNED ARTIFACT DIGEST — the cross-compiler anchor (identical MSVC + clang). It equals the S2
    // OBJ artifact digest 0xf7ee13c169dc0464 BY DESIGN: the triangle glTF and the S1 OBJ fixture decode to
    // the SAME positions + indices, and CompileGltf emits CompileObj's exact blob format -> identical bytes
    // -> identical digest. The format-parity headline holds even at the pinned-hash level.
    const CacheKey kPinnedGltf = 0xf7ee13c169dc0464ULL;
    check(gdigest == kPinnedGltf,
          "asset-gltf: CompileGltf(triangle) digest == pinned uint64 (== the S2 OBJ artifact digest — format parity)");

    // (G-3) DETERMINISTIC — recompiling the same bytes -> a byte-identical blob.
    {
        const std::vector<uint8_t> gblob2 = CompileGltf(gbytes, glen, params);
        check(gblob2 == gblob,
              "asset-gltf: same glTF bytes + params -> a BYTE-IDENTICAL blob (deterministic / content-addressed)");
    }

    // (G-4) FORMAT PARITY (THE HEADLINE) — the single-triangle glTF and the S1 OBJ fixture of the SAME
    // geometry ((0,0,0),(1,0,0),(0,1,0) + tri {0,1,2}) compile, under the same params, to the IDENTICAL blob.
    {
        const std::vector<uint8_t> oblob = CompileObj(ShowcaseRawBytes(), ShowcaseRawLen(), params);
        check(gblob == oblob,
              "asset-gltf: glTF and OBJ of EQUIVALENT geometry compile to the BYTE-IDENTICAL blob (format parity)");
    }

    // (G-5) CONTENT LOAD-BEARING — a glTF with one vertex moved (apex y: 1 -> 2) -> a DIFFERENT digest.
    // Buffer = positions (0,0,0),(1,0,0),(0,2,0) + indices {0,1,2}; only the apex y float differs.
    {
        // (0,2,0): float 2.0 = 0x40000000 -> LE base64. Reuse the same layout, swap vertex-2 y.
        static const char* kMovedGltf =
            "{\n"
            "  \"asset\": { \"version\": \"2.0\" },\n"
            "  \"buffers\": [ { \"byteLength\": 44, \"uri\": \"data:application/octet-stream;base64,"
            "AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAAEAAAAAAAAABAAIAAAA=\" } ],\n"
            "  \"bufferViews\": [\n"
            "    { \"buffer\": 0, \"byteOffset\": 0,  \"byteLength\": 36, \"target\": 34962 },\n"
            "    { \"buffer\": 0, \"byteOffset\": 36, \"byteLength\": 6,  \"target\": 34963 }\n"
            "  ],\n"
            "  \"accessors\": [\n"
            "    { \"bufferView\": 0, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\","
            " \"min\": [0,0,0], \"max\": [1,2,0] },\n"
            "    { \"bufferView\": 1, \"componentType\": 5123, \"count\": 3, \"type\": \"SCALAR\" }\n"
            "  ],\n"
            "  \"meshes\": [ { \"primitives\": [ { \"attributes\": { \"POSITION\": 0 }, \"indices\": 1 } ] } ]\n"
            "}\n";
        const CacheKey moved = DigestArtifact(CompileGltf((const uint8_t*)kMovedGltf, StrLen(kMovedGltf), params));
        check(moved != gdigest,
              "asset-gltf: a moved vertex in the glTF -> a DIFFERENT artifact digest (geometry is load-bearing)");
    }

    // (G-6) PARAM LOAD-BEARING — change `scale` -> a DIFFERENT digest (the params header is in the blob).
    {
        CompileParams p2 = params;
        p2.scale = 3 * 65536;   // 3.0 instead of 2.0
        const CacheKey a2 = DigestArtifact(CompileGltf(gbytes, glen, p2));
        check(a2 != gdigest,
              "asset-gltf: a changed compile param (scale) changes the glTF artifact digest (params are load-bearing)");
    }

    // (G-7) PIPELINE WIRING — GetOrCompileGltf is a cache MISS then HIT, the HIT blob == a cold CompileGltf.
    {
        AssetCache c;
        const CompileResult r1 = GetOrCompileGltf(c, gbytes, glen, params);
        const CompileResult r2 = GetOrCompileGltf(c, gbytes, glen, params);
        check(r1.wasHit == false && r2.wasHit == true && r1.blob == gblob && r2.blob == gblob,
              "asset-gltf: GetOrCompileGltf wires glTF into the cache — cold MISS, warm HIT, HIT blob == cold blob");
    }

    // (G-8) EXISTING GOLDENS UNCHANGED — re-assert every pinned asset_compiler digest (S1..S6) is intact.
    {
        const CacheKey s1 = MakeKey((uint32_t)AssetKind::Mesh,
                                    HashRawAsset(ShowcaseRawBytes(), ShowcaseRawLen()),
                                    HashParams(ShowcaseParams()));
        const CacheKey s2 = DigestArtifact(CompileObj(ShowcaseRawBytes(), ShowcaseRawLen(), ShowcaseParams()));
        const CacheKey s3 = DigestCache(MakeShowcaseCache());
        const CacheKey s4 = Rebuild(MakeShowcaseGraph(), std::vector<NodeId>{ 0 }).digest;
        const CacheKey s5 = ManifestDigest([]{ AssetCache c; return CompileSet(MakeShowcaseJobs(), c); }());
        check(s1 == 0x7fb6a48b4b99f1b7ULL && s2 == 0xf7ee13c169dc0464ULL &&
              s3 == 0x029174f13e64c9f1ULL && s4 == 0x0808b56e0322d8c1ULL &&
              s5 == 0x37ca56d9da205682ULL,
              "asset-gltf: ALL existing asset_compiler goldens (S1 key/S2 artifact/S3 cache/S4 rebuild/S5 manifest) UNCHANGED");
    }

    if (g_fail == 0) std::printf("asset_gltf_compile_test: ALL PASS\n");
    else             std::printf("asset_gltf_compile_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
