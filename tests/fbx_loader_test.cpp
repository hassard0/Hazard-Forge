// Golden test for the SELF-CONTAINED, CLEAN-ROOM binary FBX mesh importer (engine/asset/fbx_loader.h,
// issue #15 ask #2 -- "Asset import: support FBX alongside glTF/OBJ"). Pure-CPU, ASan-eligible like the
// other pure tests.
//
// SELF-CONTAINED: the scaffolding (check() + HF_TEST_MAIN_INIT()) mirrors draco_test.cpp; fbx_loader.h
// pulls in only <cstddef/cstdint/cstring/string/vector> + net/session.h, so this compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/fbx_loader_test.cpp` on the Mac -- the cheap cross-platform
// proof. ParseFbx is a pure byte->mesh decode (same bytes in -> same mesh out), so the position+index
// DigestBytes (net::DigestBytes, FNV-1a-64) is bit-identical MSVC == clang == Mac-clang.
//
// The fixture is the assimp BSD-licensed test cube test/models/FBX/box.fbx (FBX version 7400, RAW arrays --
// see ATTRIBUTION). What this pins:
//   (1) ParseFbx(box.fbx) -> ok, version == 7400;
//   (2) 24 vertices (72 floats) -- the FBX box stores per-face corners, not 8 shared corners;
//   (3) 12 triangles / 36 indices (the polygons triangulate to a closed cube);
//   (4) the positions form a box: each axis has EXACTLY 2 distinct values (here -100, +100);
//   (5) a PINNED net::DigestBytes over the positions + indices (the cross-platform reproducibility moat);
//   (6) the clean-room INFLATE round-trips a hand-built zlib stream (the encoding==1 path real FBX uses).

#include "asset/fbx_loader.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::asset;
namespace net = hf::net;

#ifndef HF_BOX_FBX
#error "HF_BOX_FBX (path to assets/models/box.fbx) must be defined by the build"
#endif

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

// Read a whole file into a byte vector (returns empty on failure).
static std::vector<uint8_t> ReadFile(const char* path) {
    std::vector<uint8_t> buf;
    FILE* f = std::fopen(path, "rb");
    if (!f) return buf;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz > 0) { buf.resize((std::size_t)sz); std::fread(buf.data(), 1, (std::size_t)sz, f); }
    std::fclose(f);
    return buf;
}

// A hand-built zlib stream (RFC 1950 header + a DEFLATE STORED block) carrying "ABCDEFGH" -- a fixed,
// compiler-independent vector that exercises the InflateZlib entry + the stored-block path with NO external
// zlib dependency. (The dynamic/fixed-Huffman paths are exercised by real compressed FBX in the field;
// this in-test vector keeps the standalone test self-contained.)
//   CMF=0x78, FLG=0x01  (CM=8 deflate, 0x7801 % 31 == 0)
//   DEFLATE: BFINAL=1,BTYPE=00 -> byte 0x01 ; LEN=8 (0x08 0x00) ; NLEN=~8 (0xF7 0xFF) ; 8 literal bytes
//   Adler-32 trailer (4 bytes) -- not validated by InflateZlib.
static void TestInflate() {
    const uint8_t payload[8] = {'A','B','C','D','E','F','G','H'};
    std::vector<uint8_t> z = {0x78, 0x01,                 // zlib header
                              0x01,                        // BFINAL=1, stored block
                              0x08, 0x00, 0xF7, 0xFF};     // LEN=8, NLEN=~8
    for (uint8_t b : payload) z.push_back(b);
    // a (deliberately wrong but unread) 4-byte Adler trailer
    z.push_back(0); z.push_back(0); z.push_back(0); z.push_back(0);

    std::vector<uint8_t> out;
    bool ok = fbx_detail::InflateZlib(z.data(), z.size(), out);
    check(ok, "InflateZlib: hand-built stored-block zlib stream inflates (ok)");
    check(out.size() == 8, "InflateZlib: recovered 8 bytes");
    bool match = (out.size() == 8);
    for (std::size_t i = 0; i < out.size() && i < 8; ++i) if (out[i] != payload[i]) match = false;
    check(match, "InflateZlib: recovered bytes == \"ABCDEFGH\"");

    // a corrupt header must be rejected, not crash
    std::vector<uint8_t> bad = {0x00, 0x00, 0x00};
    std::vector<uint8_t> tmp;
    check(!fbx_detail::InflateZlib(bad.data(), bad.size(), tmp), "InflateZlib: rejects a bad zlib header");
}

int main() {
    HF_TEST_MAIN_INIT();

    // --- (6) the clean-room inflate round-trip (the encoding==1 path) ---
    TestInflate();

    // --- load + parse the real binary FBX cube ---
    const char* path = HF_BOX_FBX;
    std::vector<uint8_t> bytes = ReadFile(path);
    check(!bytes.empty(), "box.fbx loads off disk");
    if (bytes.empty()) {
        std::printf("fbx_loader_test: could not read %s\n", path);
        return 1;
    }

    FbxMesh m = ParseFbx(bytes);

    // (1) parses, version
    check(m.ok, "ParseFbx(box.fbx): ok");
    check(m.version == 7400, "ParseFbx(box.fbx): version == 7400");

    // (2) vertex count: the FBX box stores 24 per-face corners (8 unique * 3 faces touching each)
    check(m.positions.size() == 72, "positions == 72 floats (24 verts)");

    // (3) triangulated index count: a closed cube == 12 triangles == 36 indices
    check(m.indices.size() == 36, "indices == 36 (12 triangles)");

    // (4) the positions form a box: each axis has exactly 2 distinct values
    bool boxShape = (m.positions.size() == 72);
    for (int axis = 0; axis < 3 && boxShape; ++axis) {
        std::set<float> distinct;
        for (std::size_t i = axis; i < m.positions.size(); i += 3) distinct.insert(m.positions[i]);
        if (distinct.size() != 2) boxShape = false;
    }
    check(boxShape, "positions form a box (2 distinct values per axis)");

    // every index must be in range
    bool inRange = true;
    const uint32_t vcount = (uint32_t)(m.positions.size() / 3);
    for (uint32_t ix : m.indices) if (ix >= vcount) inRange = false;
    check(inRange, "all triangle indices in [0, vertexCount)");

    // (5) the PINNED digest over positions then indices (the cross-platform reproducibility golden).
    // Hashed as: DigestBytes(positions floats) folded with DigestBytes(indices uint32) via the FNV mix.
    uint64_t dPos = net::DigestBytes(m.positions.data(), m.positions.size() * sizeof(float));
    uint64_t dIdx = net::DigestBytes(m.indices.data(),  m.indices.size()  * sizeof(uint32_t));

    // Pinned goldens (computed on the assimp box.fbx fixture; MSVC == clang must agree byte-for-byte).
    constexpr uint64_t kPosDigest = 0xbc330afeaf946e43ull;
    constexpr uint64_t kIdxDigest = 0xc0220c47438b5cc3ull;

    std::printf("fbx digest: positions=%016llx indices=%016llx\n",
                (unsigned long long)dPos, (unsigned long long)dIdx);

    check(dPos == kPosDigest, "positions digest matches the pinned golden");
    // The index digest is also pinned (recorded from this same fixture).
    check(dIdx == kIdxDigest, "indices digest matches the pinned golden");

    if (g_fail == 0) std::printf("fbx_loader_test: ALL PASS\n");
    else             std::printf("fbx_loader_test: %d CHECK(S) FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
