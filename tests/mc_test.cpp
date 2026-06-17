// Slice MC1 — GPU Isosurface Meshing Slice 1: the per-cell MARCHING-CUBES CASE CLASSIFICATION integer
// core (render/mc.h) that the GPU shaders/mc_classify.comp.hlsl copies VERBATIM + proves bit-identical.
// Pure CPU (header-only, no device, no backend symbols). Namespace hf::render::mc.
//
// What this test PINS (the contracts the GPU mc_classify.comp + the GPU==CPU proof build on):
//   * CaseIndex truth table: all-below -> 0x00, all-above -> 0xFF, each single corner i above ->
//     (1<<i), a few mixed configs hand-verified; the "inside" convention is scalar > isovalue.
//   * kCornerOffset matches the LOCKED canonical MC numbering 0=(0,0,0)..7=(0,1,1).
//   * ClassifyCells over a known tiny field (a planar boundary -> a known per-cell case) + cellId
//     layout (cz*(ny-1)+cy)*(nx-1)+cx.
//   * MakeSphereField: an SDF sphere -> the surface-cell set is a thin SHELL (a sanity bound:
//     0 < surfaceCells < cellCount, and inside/outside cells both exist).
//   * classifyEnabled=false modeled (empty -> all-zero) + DETERMINISM (two passes bit-identical).
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/mc.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace mc = hf::render::mc;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= CaseIndex truth table =================
    {
        const int32_t iso = 0;
        int32_t below[8]; for (int i = 0; i < 8; ++i) below[i] = -1;   // all below iso
        int32_t above[8]; for (int i = 0; i < 8; ++i) above[i] = +1;   // all above iso
        check(mc::CaseIndex(below, iso) == 0x00, "all corners below iso -> 0x00 (empty)");
        check(mc::CaseIndex(above, iso) == 0xFF, "all corners above iso -> 0xFF (full)");

        // "inside" is STRICTLY greater: a corner exactly AT the isovalue is NOT inside.
        int32_t atIso[8]; for (int i = 0; i < 8; ++i) atIso[i] = 0;
        check(mc::CaseIndex(atIso, iso) == 0x00, "corners AT iso -> 0x00 (strict >, not >=)");

        // Each single corner i above -> exactly bit i set (1<<i).
        for (int i = 0; i < 8; ++i) {
            int32_t c[8]; for (int k = 0; k < 8; ++k) c[k] = -1;
            c[i] = +1;
            check(mc::CaseIndex(c, iso) == (uint8_t)(1u << i), "single corner i above -> (1<<i)");
        }
        // Hand-verified: only corner 0 above -> 0x01 (the locked convention's named case).
        {
            int32_t c[8]; for (int k = 0; k < 8; ++k) c[k] = -1; c[0] = +1;
            check(mc::CaseIndex(c, iso) == 0x01, "only corner 0 above -> 0x01");
        }
        // A few mixed configs: corners 0 and 7 above -> bit0|bit7 = 0x81; corners 1,2,3 -> 0x0E.
        {
            int32_t c[8]; for (int k = 0; k < 8; ++k) c[k] = -1; c[0] = c[7] = +1;
            check(mc::CaseIndex(c, iso) == 0x81, "corners 0+7 above -> 0x81");
        }
        {
            int32_t c[8]; for (int k = 0; k < 8; ++k) c[k] = -1; c[1] = c[2] = c[3] = +1;
            check(mc::CaseIndex(c, iso) == 0x0E, "corners 1+2+3 above -> 0x0E");
        }
        // Non-zero isovalue: the compare uses the iso, not 0.
        {
            int32_t c[8]; for (int k = 0; k < 8; ++k) c[k] = 5; c[4] = 20;
            check(mc::CaseIndex(c, 10) == (uint8_t)(1u << 4), "iso=10: only corner4(20)>10 -> 0x10");
        }
    }

    // ================= kCornerOffset matches the locked convention =================
    {
        const int expect[8][3] = {
            {0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}
        };
        bool ok = true;
        for (int i = 0; i < 8; ++i)
            for (int a = 0; a < 3; ++a)
                if (mc::kCornerOffset[i][a] != expect[i][a]) ok = false;
        check(ok, "kCornerOffset is the locked canonical MC numbering 0=(0,0,0)..7=(0,1,1)");
    }

    // ================= VoxelField geometry =================
    {
        mc::VoxelField f; f.nx = 3; f.ny = 4; f.nz = 5; f.scalar.assign(3*4*5, 0);
        check(f.cornerCount() == 60, "cornerCount == nx*ny*nz");
        check(f.cellCount() == 2*3*4, "cellCount == (nx-1)(ny-1)(nz-1)");
        mc::VoxelField tiny; tiny.nx = 1; tiny.ny = 5; tiny.nz = 5;
        check(tiny.cellCount() == 0, "a degenerate (nx<2) field has 0 cells");
    }

    // ================= ClassifyCells over a known planar-boundary field =================
    {
        // A 3x2x2-corner field (2x1x1 = 2 cells along x). The boundary is a plane at x: corners with
        // x==0 are below iso (-1), x>=1 are above (+1). iso=0.
        //   cell cx=0 owns corners x in {0,1}: corner0(x0)below, corner1(x1)above, ... the corners with
        //   local dx==1 are above. Per kCornerOffset the dx==1 corners are i=1,2,5,6 -> bits 1,2,5,6 set
        //   = 0b01100110 = 0x66.
        //   cell cx=1 owns corners x in {1,2}: ALL above -> 0xFF.
        mc::VoxelField f; f.nx = 3; f.ny = 2; f.nz = 2;
        f.scalar.assign((size_t)3*2*2, 0);
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 3; ++x)
                    f.scalar[(size_t)(z*2 + y)*3 + x] = (x == 0) ? -1 : +1;
        std::vector<uint8_t> cases;
        mc::ClassifyCells(f, 0, cases);
        check(cases.size() == 2, "planar field -> 2 cells");
        check(cases[0] == 0x66, "cell cx=0 (x-boundary) -> dx==1 corners {1,2,5,6} above -> 0x66");
        check(cases[1] == 0xFF, "cell cx=1 (all corners x>=1) -> 0xFF (full)");
    }

    // ================= ClassifyCells empty/full whole-field =================
    {
        mc::VoxelField f; f.nx = f.ny = f.nz = 4; f.scalar.assign(4*4*4, -5);
        std::vector<uint8_t> cases;
        mc::ClassifyCells(f, 0, cases);
        bool allEmpty = true; for (uint8_t c : cases) if (c != 0x00) allEmpty = false;
        check(allEmpty && cases.size() == 27, "all-below field -> every cell 0x00");
        f.scalar.assign(4*4*4, +5);
        mc::ClassifyCells(f, 0, cases);
        bool allFull = true; for (uint8_t c : cases) if (c != 0xFF) allFull = false;
        check(allFull, "all-above field -> every cell 0xFF");
    }

    // ================= MakeSphereField -> a thin surface SHELL (sanity bound) =================
    {
        const int n = 33;          // 33³ corners -> 32³ cells (the showcase config)
        const int radius = 12;
        mc::VoxelField f = mc::MakeSphereField(n, radius);
        check(f.nx == n && f.ny == n && f.nz == n, "sphere field is n³ corners");
        check((int)f.scalar.size() == n*n*n, "sphere field scalar sized n³");
        // The centre corner is deep INSIDE (scalar > 0); a corner at the field edge is OUTSIDE (< 0).
        check(mc::SampleField(f, n/2, n/2, n/2) > 0, "sphere centre is inside (scalar>0)");
        check(mc::SampleField(f, 0, 0, 0) < 0, "sphere field corner is outside (scalar<0)");

        std::vector<uint8_t> cases;
        mc::ClassifyCells(f, 0, cases);
        int total = (int)cases.size();
        int surf = 0, inside = 0, outside = 0;
        for (uint8_t c : cases) {
            if (c == 0xFF) ++inside;
            else if (c == 0x00) ++outside;
            else ++surf;
        }
        check(total == 32*32*32, "32³ cells");
        check(surf > 0 && surf < total, "surface cells are a non-trivial proper subset (a shell)");
        check(inside > 0, "some cells are fully inside the sphere (0xFF)");
        check(outside > 0, "some cells are fully outside the sphere (0x00)");
        // A shell is THIN: surface cells << the inside-volume cells (a loose but real upper bound —
        // a radius-12 sphere shell has far fewer cells than its ~ (4/3)π12³ interior).
        check(surf < inside, "the surface shell is thinner than the interior volume");
    }

    // ================= disabled-path modeled (empty/cleared) + DETERMINISM =================
    {
        // The shader's classifyEnabled=false path writes 0 everywhere; modeled on the CPU side as the
        // cleared all-zero buffer (NOT calling ClassifyCells). Pin that a zero-cleared buffer is the
        // expected disabled output and ClassifyCells of a non-trivial field is NOT all-zero.
        mc::VoxelField f = mc::MakeSphereField(17, 6);
        std::vector<uint8_t> a, b;
        mc::ClassifyCells(f, 0, a);
        mc::ClassifyCells(f, 0, b);
        check(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()) == 0,
              "determinism: two ClassifyCells passes BYTE-IDENTICAL");
        std::vector<uint8_t> cleared((size_t)f.cellCount(), 0u);
        bool aHasNonZero = false; for (uint8_t c : a) if (c != 0) aHasNonZero = true;
        check(aHasNonZero, "a real field classifies to a NON-zero set (so the disabled no-op is meaningful)");
        check(cleared.size() == a.size(), "the disabled (cleared) buffer is sized cellCount()");
    }

    // ================= MC2: kTriTable well-formedness =================
    {
        // Every row: a run of valid edge triplets (each edge in [0,11]) followed by -1 padding to the
        // end; the number of non-negative entries is a multiple of 3 (whole triangles).
        bool wellFormed = true;
        for (int c = 0; c < 256; ++c) {
            int n = 0;             // count of non-negative (edge) entries
            bool seenTerminator = false;
            for (int e = 0; e < 16; ++e) {
                int8_t v = mc::kTriTable[c][e];
                if (v == -1) { seenTerminator = true; continue; }
                // A real edge index must come BEFORE any -1 terminator (no edge after a -1).
                if (seenTerminator) wellFormed = false;
                if (v < 0 || v > 11) wellFormed = false;   // valid edge range
                ++n;
            }
            if (n % 3 != 0) wellFormed = false;            // whole triangles only
            if (n / 3 > 5) wellFormed = false;             // a cell emits at most 5 triangles
        }
        check(wellFormed, "kTriTable rows well-formed (groups of 3, edges in [0,11], <=5 tris, -1 padded)");
    }

    // ================= MC2: kTriCount == derived-from-kTriTable for ALL 256 cases =================
    {
        bool consistent = true;
        for (int c = 0; c < 256; ++c) {
            int derived = 0;
            for (int e = 0; e < 16; ++e) if (mc::kTriTable[c][e] >= 0) ++derived;
            derived /= 3;
            if (mc::kTriCount[c] != derived) consistent = false;
            if (mc::kTriCount[c] < 0 || mc::kTriCount[c] > 5) consistent = false;   // range [0,5]
        }
        check(consistent, "kTriCount[c] == (#non-neg kTriTable[c])/3 for all 256, each in [0,5]");
        check(mc::kTriCount[0x00] == 0, "kTriCount[0x00] == 0 (fully-out cell emits no triangles)");
        check(mc::kTriCount[0xFF] == 0, "kTriCount[0xFF] == 0 (fully-in cell emits no triangles)");
        // CountTriangles is the kTriCount accessor.
        check(mc::CountTriangles(0x00) == 0 && mc::CountTriangles(0xFF) == 0,
              "CountTriangles(0x00)==CountTriangles(0xFF)==0");
        // A single-corner case emits exactly 1 triangle (one corner cut off by 3 edges).
        check(mc::CountTriangles(0x01) == 1, "single-corner case 0x01 -> 1 triangle");
    }

    // ================= MC2: CountCells over a known tiny field -> known counts =================
    {
        // The same planar-boundary 3x2x2 field as the ClassifyCells test: cell0 case 0x66, cell1 0xFF.
        mc::VoxelField f; f.nx = 3; f.ny = 2; f.nz = 2;
        f.scalar.assign((size_t)3*2*2, 0);
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 3; ++x)
                    f.scalar[(size_t)(z*2 + y)*3 + x] = (x == 0) ? -1 : +1;
        std::vector<uint8_t> cases;
        mc::ClassifyCells(f, 0, cases);
        std::vector<uint32_t> counts;
        mc::CountCells(f, 0, counts);
        check(counts.size() == 2, "planar field -> 2 cell counts");
        // counts[cellId] must equal CountTriangles(case[cellId]) by construction.
        check(counts[0] == (uint32_t)mc::CountTriangles(cases[0]),
              "count[0] == CountTriangles(case[0]) (cell cx=0)");
        check(counts[1] == (uint32_t)mc::CountTriangles(cases[1]),
              "count[1] == CountTriangles(case[1]) (cell cx=1, full -> 0 tris)");
        check(counts[1] == 0u, "the FULL cell (0xFF) emits 0 triangles");

        // TotalTriangles == Σ counts.
        uint32_t total = mc::TotalTriangles(std::span<const uint32_t>(counts));
        check(total == counts[0] + counts[1], "TotalTriangles == Σ counts (tiny field)");
    }

    // ================= MC2: countEnabled-off modeled (all-zero + total 0) + determinism =========
    {
        mc::VoxelField f = mc::MakeSphereField(33, 12);
        std::vector<uint32_t> a, b;
        mc::CountCells(f, 0, a);
        mc::CountCells(f, 0, b);
        check(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(uint32_t)) == 0,
              "determinism: two CountCells passes BYTE-IDENTICAL");
        // The disabled (countEnabled=false) path is modeled as the cleared all-zero buffer + total 0.
        std::vector<uint32_t> cleared((size_t)f.cellCount(), 0u);
        check(mc::TotalTriangles(std::span<const uint32_t>(cleared)) == 0u,
              "the disabled (cleared) counts sum to total 0");
        // A real sphere field counts to a NON-zero total (so the disabled no-op is meaningful) and
        // every per-cell count is in [0,5].
        uint32_t total = mc::TotalTriangles(std::span<const uint32_t>(a));
        check(total > 0u, "a real sphere field has a NON-zero total triangle count");
        bool inRange = true;
        for (uint32_t c : a) if (c > 5u) inRange = false;
        check(inRange, "every per-cell count is in [0,5]");
        // Total consistency: Σ_cells CountTriangles(case) == TotalTriangles(counts).
        std::vector<uint8_t> cases;
        mc::ClassifyCells(f, 0, cases);
        uint32_t totalFromCases = 0u;
        for (uint8_t cs : cases) totalFromCases += (uint32_t)mc::CountTriangles(cs);
        check(totalFromCases == total, "Σ CountTriangles(case) == TotalTriangles(counts) (sphere field)");
    }

    // ================= MC3: EdgeMidpoint for known edges (the two corner coords SUMMED) ===========
    {
        // For the cell at the ORIGIN (0,0,0): edge e connects kEdgeCorner[e]=(a,b); the midpoint in
        // half-units = kCornerOffset[a] + kCornerOffset[b].
        // edge 0: corners (0,1) = (0,0,0)+(1,0,0) -> (1,0,0).
        mc::McVertex v0 = mc::EdgeMidpoint(0, 0, 0, 0);
        check(v0.x == 1 && v0.y == 0 && v0.z == 0 && v0.w == 0,
              "EdgeMidpoint cell(0,0,0) edge0 (corners 0-1) -> (1,0,0) half-units");
        // edge 10: corners (2,6) = (1,1,0)+(1,1,1) -> (2,2,1).
        mc::McVertex v10 = mc::EdgeMidpoint(0, 0, 0, 10);
        check(v10.x == 2 && v10.y == 2 && v10.z == 1 && v10.w == 0,
              "EdgeMidpoint cell(0,0,0) edge10 (corners 2-6) -> (2,2,1) half-units");
        // edge 8 (vertical 0-4): (0,0,0)+(0,0,1) -> (0,0,1).
        mc::McVertex v8 = mc::EdgeMidpoint(0, 0, 0, 8);
        check(v8.x == 0 && v8.y == 0 && v8.z == 1, "EdgeMidpoint edge8 (corners 0-4) -> (0,0,1)");
        // A SHIFTED cell (1,2,3): the same edge 0 is offset by 2*cell (both summed corners include cell).
        mc::McVertex vs = mc::EdgeMidpoint(1, 2, 3, 0);
        check(vs.x == (1 + 2) && vs.y == (2 + 2) && vs.z == (3 + 3),
              "EdgeMidpoint cell(1,2,3) edge0 = (cx+cx+1, cy+cy, cz+cz) -> (3,4,6)");
    }

    // ================= MC3: PrefixSumOffsets known counts -> known offsets + total ================
    {
        const uint32_t counts[5] = {3, 0, 2, 5, 1};
        uint32_t offsets[5] = {99, 99, 99, 99, 99};
        uint32_t total = 999u;
        mc::PrefixSumOffsets(std::span<const uint32_t>(counts, 5),
                             std::span<uint32_t>(offsets, 5), total);
        check(offsets[0] == 0 && offsets[1] == 3 && offsets[2] == 3 &&
              offsets[3] == 5 && offsets[4] == 10,
              "PrefixSumOffsets {3,0,2,5,1} -> exclusive {0,3,3,5,10}");
        check(total == 11u, "PrefixSumOffsets total == Σ counts == 11");
        // Empty input -> total 0.
        uint32_t emptyTotal = 7u;
        mc::PrefixSumOffsets(std::span<const uint32_t>(),
                             std::span<uint32_t>(), emptyTotal);
        check(emptyTotal == 0u, "PrefixSumOffsets of empty -> total 0");
    }

    // ================= MC3: EmitCell for a known case (tri count, midpoint verts, identity idx) ====
    {
        // Case 0x01 (only corner 0 above) emits 1 triangle, kTriTable[0x01] = {0,8,3,...}. At the origin
        // cell with triOffset=0, slots 0..2 are edges 0,8,3 -> midpoints, indices the identity {0,1,2}.
        const uint8_t kase = 0x01;
        const int n = mc::CountTriangles(kase);
        check(n == 1, "case 0x01 emits 1 triangle");
        std::vector<mc::McVertex> verts((size_t)n * 3, mc::McVertex{-9, -9, -9, -9});
        std::vector<uint32_t> idx((size_t)n * 3, 0xDEADu);
        mc::EmitCell(0, 0, 0, kase, 0u, std::span<mc::McVertex>(verts), std::span<uint32_t>(idx));
        // Edges for 0x01 are 0,8,3.
        mc::McVertex e0 = mc::EdgeMidpoint(0, 0, 0, 0);
        mc::McVertex e8 = mc::EdgeMidpoint(0, 0, 0, 8);
        mc::McVertex e3 = mc::EdgeMidpoint(0, 0, 0, 3);
        check(verts[0].x == e0.x && verts[0].y == e0.y && verts[0].z == e0.z,
              "EmitCell case 0x01 v0 == EdgeMidpoint(edge0)");
        check(verts[1].x == e8.x && verts[1].y == e8.y && verts[1].z == e8.z,
              "EmitCell case 0x01 v1 == EdgeMidpoint(edge8)");
        check(verts[2].x == e3.x && verts[2].y == e3.y && verts[2].z == e3.z,
              "EmitCell case 0x01 v2 == EdgeMidpoint(edge3)");
        check(idx[0] == 0 && idx[1] == 1 && idx[2] == 2, "EmitCell case 0x01 indices are identity {0,1,2}");

        // A non-zero triOffset writes into the offset slots (the disjoint-range contract).
        std::vector<mc::McVertex> verts2((size_t)6, mc::McVertex{0, 0, 0, 0});
        std::vector<uint32_t> idx2((size_t)6, 0u);
        mc::EmitCell(0, 0, 0, kase, 1u, std::span<mc::McVertex>(verts2), std::span<uint32_t>(idx2));
        check(idx2[3] == 3 && idx2[4] == 4 && idx2[5] == 5,
              "EmitCell triOffset=1 writes identity indices {3,4,5} into slots 3..5");
        bool slot0Untouched = (verts2[0].x == 0 && idx2[0] == 0);
        check(slot0Untouched, "EmitCell triOffset=1 leaves the prior cell's range untouched");

        // Empty case 0x00 emits nothing.
        std::vector<mc::McVertex> verts3;
        std::vector<uint32_t> idx3;
        mc::EmitCell(0, 0, 0, 0x00, 0u, std::span<mc::McVertex>(verts3), std::span<uint32_t>(idx3));
        check(verts3.empty() && idx3.empty(), "EmitCell case 0x00 emits nothing (empty soup)");
    }

    // ================= MC3: MarchCells over a tiny field -> known counts + sizing =================
    {
        // The planar-boundary 3x2x2 field: cell0 case 0x66 (some tris), cell1 0xFF (0 tris).
        mc::VoxelField f; f.nx = 3; f.ny = 2; f.nz = 2;
        f.scalar.assign((size_t)3*2*2, 0);
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 3; ++x)
                    f.scalar[(size_t)(z*2 + y)*3 + x] = (x == 0) ? -1 : +1;
        std::vector<uint32_t> counts;
        mc::CountCells(f, 0, counts);
        uint32_t expectTotal = mc::TotalTriangles(std::span<const uint32_t>(counts));

        std::vector<mc::McVertex> verts;
        std::vector<uint32_t> idx;
        uint32_t triCount = 999u;
        mc::MarchCells(f, 0, verts, idx, triCount);
        check(triCount == expectTotal, "MarchCells triCount == TotalTriangles(counts)");
        check(verts.size() == (size_t)triCount * 3 && idx.size() == (size_t)triCount * 3,
              "MarchCells verts.size()==idx.size()==3*triCount");
        // The index buffer is the IDENTITY [0,3T).
        bool identity = true;
        for (size_t i = 0; i < idx.size(); ++i) if (idx[i] != (uint32_t)i) identity = false;
        check(identity, "MarchCells index buffer is the identity [0,3*triCount)");
    }

    // ================= MC3: MarchCells over the sphere field -> consistency + determinism =========
    {
        mc::VoxelField f = mc::MakeSphereField(33, 12);
        std::vector<mc::McVertex> verts; std::vector<uint32_t> idx; uint32_t triCount = 0u;
        mc::MarchCells(f, 0, verts, idx, triCount);
        // Total must equal the MC2 total over the SAME field.
        std::vector<uint32_t> counts;
        mc::CountCells(f, 0, counts);
        uint32_t mc2total = mc::TotalTriangles(std::span<const uint32_t>(counts));
        check(triCount == mc2total, "MarchCells(sphere) triCount == MC2 TotalTriangles");
        check(triCount == 5240u, "MarchCells(sphere 33/r12/iso0) triCount == 5240 (the locked count)");
        check(verts.size() == 3u * 5240u && idx.size() == 3u * 5240u,
              "MarchCells(sphere) verts==idx==3*5240==15720");
        // Every index in range [0, 3*triCount).
        bool inRange = true;
        for (uint32_t v : idx) if (v >= 3u * triCount) inRange = false;
        check(inRange, "MarchCells(sphere) every index in [0,3*triCount)");

        // Determinism: a second march is byte-identical (verts + indices).
        std::vector<mc::McVertex> verts2; std::vector<uint32_t> idx2; uint32_t triCount2 = 0u;
        mc::MarchCells(f, 0, verts2, idx2, triCount2);
        bool same = (triCount == triCount2) && verts.size() == verts2.size() && idx.size() == idx2.size() &&
                    std::memcmp(verts.data(), verts2.data(), verts.size() * sizeof(mc::McVertex)) == 0 &&
                    std::memcmp(idx.data(), idx2.data(), idx.size() * sizeof(uint32_t)) == 0;
        check(same, "MarchCells determinism: two marches BYTE-IDENTICAL (verts + indices)");

        // The disabled (meshEnabled=false) path is modeled as the cleared (empty) buffers: a real field
        // marches to a NON-empty mesh (so the no-op is meaningful).
        check(!verts.empty(), "a real sphere field marches to a NON-empty mesh");
    }

    if (g_fail == 0) std::printf("mc_test: ALL PASS\n");
    else std::printf("mc_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
