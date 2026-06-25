// Golden test for the SELF-CONTAINED, CLEAN-ROOM USDA (ASCII USD) mesh importer (engine/asset/usd_loader.h,
// issue #15 ask #3 -- "Asset import: support FBX, OBJ and USD alongside glTF"). Pure-CPU, ASan-eligible like
// the other pure tests. With OBJ + glTF + FBX + USD all importing, this CLOSES #15.
//
// SELF-CONTAINED: the scaffolding (check() + HF_TEST_MAIN_INIT()) mirrors fbx_loader_test.cpp; usd_loader.h
// pulls in only <cstddef/cstdint/cstring/string/vector> + net/session.h, so this compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/usd_loader_test.cpp` on the Mac -- the cheap cross-platform
// proof. ParseUsda is a pure text->mesh decode (same bytes in -> same mesh out), so the position+index
// DigestBytes (net::DigestBytes, FNV-1a-64) is bit-identical MSVC == clang == Mac-clang.
//
// The fixture is the authored unit cube assets/models/cube.usda (8 points, 6 quad faces). What this pins:
//   (1) ParseUsda(cube.usda) -> ok;
//   (2) 8 vertices (24 floats) -- the authored cube shares 8 corners;
//   (3) 12 triangles / 36 indices (the 6 quads fan-triangulate to a closed cube);
//   (4) the positions form a cube: each axis has EXACTLY 2 distinct values (here -0.5, +0.5);
//   (5) a PINNED net::DigestBytes over the positions + indices (the cross-platform reproducibility moat);
//   (6) a tiny EMBEDDED USDA literal (a single triangle) parses standalone with no file dependency.

#include "asset/usd_loader.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::asset;
namespace net = hf::net;

#ifndef HF_CUBE_USDA
#error "HF_CUBE_USDA (path to assets/models/cube.usda) must be defined by the build"
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

// (6) A standalone embedded USDA literal: a single triangle, no file. Proves the parser works on an
// in-memory string (whitespace/newlines/comments + the bracketed arrays) with no disk dependency.
static void TestEmbedded() {
    const char* usda =
        "#usda 1.0\n"
        "def Mesh \"tri\" {\n"
        "    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]  # three corners\n"
        "    int[] faceVertexCounts = [3]\n"
        "    int[] faceVertexIndices = [0, 1, 2]\n"
        "}\n";
    UsdMesh m = ParseUsda(usda, std::strlen(usda));
    check(m.ok, "embedded: single-triangle USDA literal parses (ok)");
    check(m.positions.size() == 9, "embedded: 9 position floats (3 verts)");
    check(m.indices.size() == 3, "embedded: 3 indices (1 triangle)");
    bool idxOk = (m.indices.size() == 3) && m.indices[0] == 0 && m.indices[1] == 1 && m.indices[2] == 2;
    check(idxOk, "embedded: indices == {0,1,2}");
    // a malformed input (no Mesh) -> ok=false, no crash
    const char* bad = "#usda 1.0\ndef Xform \"empty\" { }\n";
    UsdMesh mb = ParseUsda(bad, std::strlen(bad));
    check(!mb.ok, "embedded: a USDA with no def Mesh -> ok=false");
}

int main() {
    HF_TEST_MAIN_INIT();

    // --- (6) the embedded standalone-no-file case ---
    TestEmbedded();

    // --- load + parse the authored USDA cube ---
    const char* path = HF_CUBE_USDA;
    std::vector<uint8_t> bytes = ReadFile(path);
    check(!bytes.empty(), "cube.usda loads off disk");
    if (bytes.empty()) {
        std::printf("usd_loader_test: could not read %s\n", path);
        return 1;
    }

    UsdMesh m = ParseUsda((const char*)bytes.data(), bytes.size());

    // (1) parses
    check(m.ok, "ParseUsda(cube.usda): ok");

    // (2) vertex count: the authored cube shares 8 corners -> 24 floats
    check(m.positions.size() == 24, "positions == 24 floats (8 verts)");

    // (3) triangulated index count: 6 quads fan-triangulated == 12 triangles == 36 indices
    check(m.indices.size() == 36, "indices == 36 (12 triangles)");

    // (4) the positions form a cube: each axis has exactly 2 distinct values (-0.5, +0.5)
    bool cubeShape = (m.positions.size() == 24);
    for (int axis = 0; axis < 3 && cubeShape; ++axis) {
        std::set<float> distinct;
        for (std::size_t i = axis; i < m.positions.size(); i += 3) distinct.insert(m.positions[i]);
        if (distinct.size() != 2) cubeShape = false;
    }
    check(cubeShape, "positions form a cube (2 distinct values per axis)");

    // every index must be in range
    bool inRange = true;
    const uint32_t vcount = (uint32_t)(m.positions.size() / 3);
    for (uint32_t ix : m.indices) if (ix >= vcount) inRange = false;
    check(inRange, "all triangle indices in [0, vertexCount)");

    // (5) the PINNED digest over positions then indices (the cross-platform reproducibility golden).
    uint64_t dPos = net::DigestBytes(m.positions.data(), m.positions.size() * sizeof(float));
    uint64_t dIdx = net::DigestBytes(m.indices.data(),  m.indices.size()  * sizeof(uint32_t));

    // Pinned goldens (computed on the authored cube.usda fixture; MSVC == clang must agree byte-for-byte).
    constexpr uint64_t kPosDigest = 0x23d56aed37ec0843ull;
    constexpr uint64_t kIdxDigest = 0xbfcc4c0bd639b973ull;

    std::printf("usd digest: positions=%016llx indices=%016llx\n",
                (unsigned long long)dPos, (unsigned long long)dIdx);

    check(dPos == kPosDigest, "positions digest matches the pinned golden");
    check(dIdx == kIdxDigest, "indices digest matches the pinned golden");

    if (g_fail == 0) std::printf("usd_loader_test: ALL PASS\n");
    else             std::printf("usd_loader_test: %d CHECK(S) FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
