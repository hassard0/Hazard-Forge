// Unit test for the Wavefront OBJ geometry parser (engine/asset/obj_loader.h, issue #15 ask #1).
//
// Proves the issue's headline value-prop — "your asset pipeline is reproducible": re-importing the same OBJ
// produces a BYTE-IDENTICAL mesh, pinned via net::DigestBytes and identical across MSVC + Windows-clang +
// Mac/clang. Plus parser correctness: triangulation, corner forms (v / v/vt / v/vt/vn / v//vn), dedup,
// 1-based + negative indices. Pure hf_core (no device/RHI/float-render), standalone-clang-compilable:
//   clang++ -std=c++20 -I engine -I tests tests/obj_loader_test.cpp

#include "asset/obj_loader.h"
#include "net/session.h"   // DigestBytes

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "test_main.h"

using namespace hf::asset;

static int g_fail = 0;
static void check(bool c, const char* what) {
    if (!c) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else    { std::printf("PASS: %s\n", what); }
}

static uint64_t DigestMesh(const ObjMesh& m) {
    uint64_t h = hf::net::DigestBytes(m.vertices.data(), m.vertices.size() * sizeof(ObjVertex));
    // fold the indices in (a second DigestBytes seeded by the first via a byte view of h).
    uint64_t h2 = hf::net::DigestBytes(m.indices.data(), m.indices.size() * sizeof(uint32_t));
    unsigned char mix[16];
    for (int i = 0; i < 8; ++i) { mix[i] = (unsigned char)(h >> (i * 8)); mix[8 + i] = (unsigned char)(h2 >> (i * 8)); }
    return hf::net::DigestBytes(mix, sizeof mix);
}

int main() {
    HF_TEST_MAIN_INIT();

    // A fixed OBJ exercising: a quad (fan-triangulated -> 2 tris), shared corners (dedup), a v/vt/vn tri,
    // a v//vn tri (no uv), and a face using NEGATIVE (relative) indices.
    const std::string obj =
        "# a small test mesh\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 1 1\n"
        "vt 0 1\n"
        "vn 0 0 1\n"
        "f 1/1/1 2/2/1 3/3/1 4/4/1\n"   // a QUAD -> fan into 2 triangles, all share vn 1
        "v 2 0 0\n"
        "v 3 0 0\n"
        "v 2 1 0\n"
        "f 5//1 6//1 7//1\n"             // v//vn corner form (no uv)
        "f -3 -2 -1\n";                  // NEGATIVE indices -> the last three positions (5,6,7)

    const ObjMesh m = ParseObj(obj);

    // (1) raw counts.
    check(m.positions == 7, "obj: parsed 7 positions (v)");
    check(m.faces == 3, "obj: parsed 3 faces (f, pre-triangulation)");

    // (2) triangulation: quad -> 2 tris (6 idx), + 2 tris (6 idx) = 12 indices = 4 triangles.
    check(m.indices.size() == 12, "obj: quad fan-triangulated + 2 tris -> 12 indices (4 triangles)");

    // (3) dedup: the quad has 4 unique (v,vt,vn); the v//vn tri 3 unique; the negative-index tri reuses the
    //     SAME 3 positions but with NO uv/vn -> distinct corners (different vt/vn) -> 3 more. = 4 + 3 + 3 = 10.
    check(m.vertices.size() == 10, "obj: deduplicated to 10 unique corners");

    // (4) corner data is correct — vertex 0 = pos(0,0,0) uv(0,0) normal(0,0,1).
    {
        const ObjVertex& v0 = m.vertices[0];
        check(v0.pos[0] == 0 && v0.pos[1] == 0 && v0.pos[2] == 0, "obj: vertex 0 position");
        check(v0.uv[0] == 0 && v0.uv[1] == 0, "obj: vertex 0 uv");
        check(v0.normal[2] == 1.0f, "obj: vertex 0 normal (from vn)");
    }

    // (5) the v//vn corner has uv defaulted to (0,0) and the normal from vn.
    //     vertex 4 is the first corner of the v//vn face (position v5 = (2,0,0)).
    {
        const ObjVertex& v = m.vertices[4];
        check(v.pos[0] == 2 && v.pos[1] == 0, "obj: v//vn corner takes the position");
        check(v.uv[0] == 0 && v.uv[1] == 0, "obj: v//vn corner defaults uv to (0,0)");
        check(v.normal[2] == 1.0f, "obj: v//vn corner takes the normal");
    }

    // (6) NEGATIVE-index face resolved to the last three positions (2,0,0)/(3,0,0)/(2,1,0).
    {
        // the last 3 vertices are the negative-index corners (position-only).
        const ObjVertex& a = m.vertices[m.vertices.size() - 3];
        check(a.pos[0] == 2 && a.pos[1] == 0 && a.pos[2] == 0, "obj: negative index -3 resolves to position 5");
    }

    // (7) PINNED digest — reproducible import (the cross-platform reproducibility headline).
    const uint64_t h = DigestMesh(m);
    std::printf("obj: mesh digest = 0x%016llx\n", (unsigned long long)h);
    check(h == 0xb7c727c4ab149216ULL, "obj: mesh digest == pinned uint64 (byte-stable reproducible import)");

    // (8) re-parse is bit-identical (deterministic).
    check(DigestMesh(ParseObj(obj)) == h, "obj: re-parsing the same OBJ is bit-identical");

    // (9) robustness: unknown keywords + blank lines + comments are skipped, not fatal.
    {
        const std::string weird = "o object\ng group\nmtllib x.mtl\nusemtl red\ns 1\n\n# c\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
        const ObjMesh w = ParseObj(weird);
        check(w.positions == 3 && w.indices.size() == 3, "obj: unknown keywords/blanks/comments skipped (forward-compatible)");
    }

    // (10) an empty / geometry-less input yields an empty mesh (no crash).
    {
        const ObjMesh empty = ParseObj("# nothing here\n");
        check(empty.vertices.empty() && empty.indices.empty(), "obj: empty input -> empty mesh (no crash)");
    }

    if (g_fail == 0) std::printf("obj_loader_test: ALL PASS\n");
    else             std::printf("obj_loader_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
