// Slice NAV1 — Deterministic GPU Navmesh BEACHHEAD: the integer HEIGHTFIELD SPAN RASTERIZATION core
// (engine/nav/navmesh.h) that the GPU shaders/nav_raster_count/scan/emit.comp.hlsl copy VERBATIM +
// prove bit-identical. Pure CPU (header-only, no device, no backend symbols). Namespace hf::nav.
//
// What this test PINS (the contracts the GPU nav_raster_* shaders + the GPU==CPU proof build on):
//   * QuantizeCoord: world->voxel column index via FloorDiv, INCLUDING negative coords (correct floor,
//     not truncate-toward-zero — the straddle-the-origin case).
//   * PointInTriXZ: a known point inside a triangle's XZ projection -> covered; a known point outside ->
//     not covered; both winding orders (CW + CCW) covered.
//   * TriColumnAabb: the inclusive column range a triangle covers, clamped to the grid.
//   * RasterizeTriangleSpans: a single triangle over a known column set -> the expected raw spans
//     (count + offset + the y-span values); a full ground quad fills every column with one span.
//   * MergeColumnSpans: two overlapping spans -> one; two disjoint spans -> two (sorted by ymin);
//     touching spans -> one.
//   * empty input: zero triangles -> zero spans (the no-op the GPU disabled-path mirrors).
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests.
#include "nav/navmesh.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace nav = hf::nav;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A small flat heightfield (w x h column grid; bmin/bmax/cs/ch carried but unused by the per-voxel math).
static nav::Heightfield MakeHf(int w, int h) {
    nav::Heightfield hf;
    hf.w = w; hf.h = h;
    hf.bminX = 0; hf.bminY = 0; hf.bminZ = 0;
    hf.bmaxX = w; hf.bmaxY = 64; hf.bmaxZ = h;
    hf.cs = 1; hf.ch = 1;
    return hf;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= QuantizeCoord (incl negative coords via FloorDiv) =================
    {
        // bmin=0, cs=4: coord 0..3 -> col 0, 4..7 -> col 1.
        check(nav::QuantizeCoord(0, 0, 4) == 0, "QuantizeCoord 0/4 == 0");
        check(nav::QuantizeCoord(3, 0, 4) == 0, "QuantizeCoord 3/4 == 0");
        check(nav::QuantizeCoord(4, 0, 4) == 1, "QuantizeCoord 4/4 == 1");
        check(nav::QuantizeCoord(7, 0, 4) == 1, "QuantizeCoord 7/4 == 1");
        // NEGATIVE coords: -1 must floor to col -1 (NOT 0 as truncate-toward-zero would give).
        check(nav::QuantizeCoord(-1, 0, 4) == -1, "QuantizeCoord -1/4 == -1 (floor, not trunc)");
        check(nav::QuantizeCoord(-4, 0, 4) == -1, "QuantizeCoord -4/4 == -1");
        check(nav::QuantizeCoord(-5, 0, 4) == -2, "QuantizeCoord -5/4 == -2");
        // With a nonzero bmin: coord 10, bmin 2, cs 4 -> (10-2)/4 == 2.
        check(nav::QuantizeCoord(10, 2, 4) == 2, "QuantizeCoord (10-2)/4 == 2");
        // Straddle: bmin -8, coord -8 -> col 0, coord -9 -> col -1.
        check(nav::QuantizeCoord(-8, -8, 4) == 0, "QuantizeCoord straddle (-8-(-8))/4 == 0");
        check(nav::QuantizeCoord(-9, -8, 4) == -1, "QuantizeCoord straddle (-9-(-8))/4 == -1");
    }

    // ================= PointInTriXZ (cover test, both windings) =================
    {
        // A right triangle in XZ: (0,0)-(4,0)-(0,4) (y ignored by the XZ test).
        nav::NavTri t{nav::NavVert{0, 0, 0}, nav::NavVert{4, 0, 0}, nav::NavVert{0, 0, 4}};
        check(nav::PointInTriXZ(1, 1, t), "PointInTriXZ inside (1,1) covered");
        check(nav::PointInTriXZ(0, 0, t), "PointInTriXZ on-vertex (0,0) covered");
        check(!nav::PointInTriXZ(3, 3, t), "PointInTriXZ outside (3,3) not covered");
        check(!nav::PointInTriXZ(5, 0, t), "PointInTriXZ outside (5,0) not covered");
        // Reversed winding (CW) must give the SAME inside/outside (consistent-sign test).
        nav::NavTri tcw{nav::NavVert{0, 0, 0}, nav::NavVert{0, 0, 4}, nav::NavVert{4, 0, 0}};
        check(nav::PointInTriXZ(1, 1, tcw), "PointInTriXZ CW inside (1,1) covered");
        check(!nav::PointInTriXZ(3, 3, tcw), "PointInTriXZ CW outside (3,3) not covered");
    }

    // ================= TriColumnAabb (clamped inclusive column range) =================
    {
        nav::Heightfield hf = MakeHf(8, 8);
        nav::NavTri t{nav::NavVert{2, 0, 1}, nav::NavVert{5, 0, 1}, nav::NavVert{5, 0, 4}};
        nav::ColumnAabb ab;
        check(nav::TriColumnAabb(t, hf, ab), "TriColumnAabb non-empty");
        check(ab.x0 == 2 && ab.x1 == 5 && ab.z0 == 1 && ab.z1 == 4, "TriColumnAabb expected range");
        // A triangle fully off the grid (negative) clamps to empty.
        nav::NavTri off{nav::NavVert{-5, 0, -5}, nav::NavVert{-3, 0, -5}, nav::NavVert{-3, 0, -3}};
        nav::ColumnAabb ab2;
        check(!nav::TriColumnAabb(off, hf, ab2), "TriColumnAabb off-grid -> empty");
    }

    // ================= RasterizeTriangleSpans: single triangle -> expected span =================
    {
        nav::Heightfield hf = MakeHf(4, 4);
        // One triangle covering column (0,0) at y=5..9 (TriYSpan -> ymin=5, ymax=9). The triangle
        // (0,5,0)-(3,9,0)-(0,7,3) covers (0,0) (the (0,0) vertex column) but not, say, (3,3).
        nav::NavTri t{nav::NavVert{0, 5, 0}, nav::NavVert{3, 9, 0}, nav::NavVert{0, 7, 3}};
        std::vector<nav::NavTri> tris{t};
        std::vector<uint32_t> colCount, colOffset;
        std::vector<nav::Span> spans;
        nav::RasterizeTriangleSpans(hf, std::span<const nav::NavTri>(tris), colCount, colOffset, spans);

        check(colCount.size() == 16u, "single-tri colCount sized w*h");
        // Column (0,0) is covered (a vertex sits there) -> count 1; its emitted span is the tri y-range.
        const int col00 = hf.columnId(0, 0);
        check(colCount[(size_t)col00] == 1u, "single-tri column (0,0) covered (count 1)");
        const uint32_t base = colOffset[(size_t)col00];
        check(base < spans.size(), "single-tri (0,0) offset in range");
        const nav::Span& s = spans[(size_t)base];
        check(s.ymin == 5u && s.ymax == 9u, "single-tri span y-range 5..9 (TriYSpan)");
        check(s.area == nav::kDefaultArea, "single-tri span area == kDefaultArea");
        // The total spans == sum of colCount, and the exclusive prefix-sum is consistent.
        uint32_t sum = 0u; for (uint32_t c : colCount) sum += c;
        check(spans.size() == sum, "single-tri spans.size == sum(colCount)");
        uint32_t running = 0u;
        bool offOk = true;
        for (size_t c = 0; c < colCount.size(); ++c) { if (colOffset[c] != running) offOk = false; running += colCount[c]; }
        check(offOk, "single-tri colOffset == exclusive prefix-sum");
        check(spans.size() >= 1u, "single-tri produced >=1 span");
    }

    // ================= RasterizeTriangleSpans: a ground quad fills every column =================
    {
        nav::Heightfield hf = MakeHf(6, 6);
        // Two tris forming the full ground quad [0,0]-[5,5] at y=0.
        std::vector<nav::NavTri> tris{
            nav::NavTri{nav::NavVert{0, 0, 0}, nav::NavVert{5, 0, 0}, nav::NavVert{5, 0, 5}},
            nav::NavTri{nav::NavVert{0, 0, 0}, nav::NavVert{5, 0, 5}, nav::NavVert{0, 0, 5}},
        };
        std::vector<uint32_t> colCount, colOffset;
        std::vector<nav::Span> spans;
        nav::RasterizeTriangleSpans(hf, std::span<const nav::NavTri>(tris), colCount, colOffset, spans);
        // Every column must be covered by at least one of the two ground tris.
        bool allCovered = true;
        for (uint32_t c : colCount) if (c < 1u) allCovered = false;
        check(allCovered, "ground quad covers every column");
        // Every emitted span is the y=0 ground span.
        bool allGround = true;
        for (const nav::Span& s : spans) if (s.ymin != 0u || s.ymax != 0u) allGround = false;
        check(allGround, "ground quad spans are y=0..0");
    }

    // ================= MergeColumnSpans (deterministic merge) =================
    {
        // Two overlapping spans -> one.
        std::vector<nav::Span> raw{nav::Span{0u, 5u, 1u}, nav::Span{3u, 8u, 1u}};
        auto m = nav::MergeColumnSpans(raw);
        check(m.size() == 1u, "merge overlapping -> 1 span");
        check(m[0].ymin == 0u && m[0].ymax == 8u, "merge overlapping -> [0,8]");

        // Two DISJOINT spans (a gap) -> two, sorted by ymin.
        std::vector<nav::Span> raw2{nav::Span{10u, 12u, 1u}, nav::Span{0u, 3u, 1u}};
        auto m2 = nav::MergeColumnSpans(raw2);
        check(m2.size() == 2u, "merge disjoint -> 2 spans");
        check(m2[0].ymin == 0u && m2[0].ymax == 3u, "merge disjoint sorted: first [0,3]");
        check(m2[1].ymin == 10u && m2[1].ymax == 12u, "merge disjoint sorted: second [10,12]");

        // TOUCHING spans (ymax+1 == next ymin) -> one contiguous span.
        std::vector<nav::Span> raw3{nav::Span{0u, 4u, 1u}, nav::Span{5u, 9u, 1u}};
        auto m3 = nav::MergeColumnSpans(raw3);
        check(m3.size() == 1u, "merge touching -> 1 span");
        check(m3[0].ymin == 0u && m3[0].ymax == 9u, "merge touching -> [0,9]");

        // Empty input -> empty.
        std::vector<nav::Span> empty;
        check(nav::MergeColumnSpans(empty).empty(), "merge empty -> empty");
    }

    // ================= empty input -> empty heightfield (the no-op) =================
    {
        nav::Heightfield hf = MakeHf(5, 5);
        std::vector<nav::NavTri> tris;  // zero triangles
        std::vector<uint32_t> colCount, colOffset;
        std::vector<nav::Span> spans;
        nav::RasterizeTriangleSpans(hf, std::span<const nav::NavTri>(tris), colCount, colOffset, spans);
        check(colCount.size() == 25u, "empty: colCount still sized w*h");
        bool allZero = true; for (uint32_t c : colCount) if (c != 0u) allZero = false;
        check(allZero, "empty: every colCount == 0");
        check(spans.empty(), "empty: zero spans (no-op)");
    }

    // ================= determinism: two rasterizations byte-identical =================
    {
        nav::Heightfield hf = MakeHf(16, 16);
        std::vector<nav::NavTri> tris = nav::MakeShowcaseTriangles(hf);
        std::vector<uint32_t> c1, o1, c2, o2;
        std::vector<nav::Span> s1, s2;
        nav::RasterizeTriangleSpans(hf, std::span<const nav::NavTri>(tris), c1, o1, s1);
        nav::RasterizeTriangleSpans(hf, std::span<const nav::NavTri>(tris), c2, o2, s2);
        bool same = (c1 == c2) && (o1 == o2) && (s1.size() == s2.size());
        if (same) for (size_t i = 0; i < s1.size(); ++i)
            if (s1[i].ymin != s2[i].ymin || s1[i].ymax != s2[i].ymax || s1[i].area != s2[i].area) same = false;
        check(same, "two rasterizations BYTE-IDENTICAL (deterministic)");
        check(!s1.empty(), "showcase scene produces spans (coverage)");
    }

    // ================= NAV2: FilterWalkableSpans — flat ground -> all walkable =================
    {
        // A 6x6 flat ground (one y=0 span per column, plenty of clearance above). cfg height 2.
        nav::Heightfield hf = MakeHf(6, 6);
        nav::WalkableConfig cfg; cfg.walkableHeight = 2; cfg.walkableClimb = 1;
        std::vector<std::vector<nav::Span>> merged((size_t)hf.columnCount());
        for (auto& m : merged) m.push_back(nav::Span{0u, 0u, 1u});   // y=0..0 solid ground
        std::vector<uint32_t> walkable; std::vector<int32_t> surfaceY;
        nav::FilterWalkableSpans(hf, cfg, merged, walkable, surfaceY);
        bool allWalk = true; for (uint32_t w : walkable) if (w != 1u) allWalk = false;
        check(allWalk, "NAV2 flat ground: every column walkable");
        bool surf0 = true; for (int32_t s : surfaceY) if (s != 0) surf0 = false;
        check(surf0, "NAV2 flat ground: surfaceY == 0 everywhere");
        bool areaSet = true; for (auto& m : merged) for (auto& s : m) if (s.area != 1u) areaSet = false;
        check(areaSet, "NAV2 flat ground: span.area stamped walkable=1");
    }

    // ================= NAV2: clearance below walkableHeight -> NOT walkable =================
    {
        // A column whose only span's top has a solid ceiling right above (clearance < height).
        nav::Heightfield hf = MakeHf(3, 3);
        nav::WalkableConfig cfg; cfg.walkableHeight = 4; cfg.walkableClimb = 1;
        std::vector<std::vector<nav::Span>> merged((size_t)hf.columnCount());
        // Column 0: ground y=0..0, ceiling y=2..3 -> clearance above the ground = 2-0-1 = 1 < 4 -> NOT.
        merged[(size_t)hf.columnId(0, 0)] = {nav::Span{0u, 0u, 1u}, nav::Span{2u, 3u, 1u}};
        std::vector<uint32_t> walkable; std::vector<int32_t> surfaceY;
        nav::FilterWalkableSpans(hf, cfg, merged, walkable, surfaceY);
        // The ground span (low) is not walkable; the ceiling span's top has fieldTop-3 clearance.
        const int col00 = hf.columnId(0, 0);
        // fieldTop = bmaxY-1 = 63; ceiling top y=3 -> clearance 60 >= 4 -> the CEILING top IS walkable.
        check(walkable[(size_t)col00] == 1u, "NAV2 cramped: column still walkable on the ceiling top");
        check(surfaceY[(size_t)col00] == 3, "NAV2 cramped: surfaceY is the ceiling top (3)");
        // The low ground span got area=0 (cramped); the ceiling span area=1.
        check(merged[(size_t)col00][0].area == 0u, "NAV2 cramped: ground span area=0 (clearance<height)");
        check(merged[(size_t)col00][1].area == 1u, "NAV2 cramped: ceiling span area=1");
        // A fully-clamped lid (no clearance to fieldTop either) -> not walkable. Span y=60..63 (top==63).
        nav::Heightfield hf2 = MakeHf(1, 1);
        std::vector<std::vector<nav::Span>> m2((size_t)hf2.columnCount());
        m2[0] = {nav::Span{60u, 63u, 1u}};   // top at fieldTop -> clearance 63-63=0 < 4 -> not walkable
        std::vector<uint32_t> wk2; std::vector<int32_t> sy2;
        nav::FilterWalkableSpans(hf2, cfg, m2, wk2, sy2);
        check(wk2[0] == 0u, "NAV2 cramped: span at fieldTop -> not walkable (0 clearance)");
    }

    // ================= NAV2: BuildDistanceField — flat ground peaks in the centre =================
    {
        nav::Heightfield hf = MakeHf(9, 9);
        nav::WalkableConfig cfg; cfg.walkableHeight = 2; cfg.walkableClimb = 1;
        std::vector<std::vector<nav::Span>> merged((size_t)hf.columnCount());
        for (auto& m : merged) m.push_back(nav::Span{0u, 0u, 1u});
        std::vector<uint32_t> walkable; std::vector<int32_t> surfaceY;
        nav::FilterWalkableSpans(hf, cfg, merged, walkable, surfaceY);
        std::vector<uint32_t> dist;
        nav::BuildDistanceField(hf, cfg, walkable, surfaceY, dist);
        // The border is seeded 0; the centre cell (4,4) is the farthest from any border.
        const uint32_t centre = dist[(size_t)hf.columnId(4, 4)];
        const uint32_t border = dist[(size_t)hf.columnId(0, 4)];
        check(border == 0u, "NAV2 distfield: border cell dist == 0 (seed)");
        check(centre > 0u, "NAV2 distfield: centre dist > 0 (interior peak)");
        // Monotonicity: the centre is the global max; no interior cell exceeds it.
        uint32_t maxd = 0u; for (uint32_t d : dist) if (d != nav::kDistInf && d > maxd) maxd = d;
        check(centre == maxd, "NAV2 distfield: centre is the global peak");
        // Chamfer monotonicity: a cell one ring in from the border (1,4) is < the centre.
        check(dist[(size_t)hf.columnId(1, 4)] < centre, "NAV2 distfield: interior > near-border (gradient)");
        // No sentinel leaks into the read-back.
        bool noInf = true; for (uint32_t d : dist) if (d == nav::kDistInf) noInf = false;
        check(noInf, "NAV2 distfield: no kDistInf in the output");
    }

    // ================= NAV2: a step exceeding walkableClimb breaks connectivity =================
    {
        // A 9x1-ish grid where the left half is at y=0 and the right half is a TALL step (y=20),
        // step 20 > climb 1 -> the two regions are NOT connected -> the distance does not bleed across.
        nav::Heightfield hf = MakeHf(9, 3);
        nav::WalkableConfig cfg; cfg.walkableHeight = 2; cfg.walkableClimb = 1;
        std::vector<std::vector<nav::Span>> merged((size_t)hf.columnCount());
        for (int z = 0; z < 3; ++z)
            for (int x = 0; x < 9; ++x) {
                const uint32_t y = (x < 4) ? 0u : 20u;   // left low, right HIGH (step 20)
                merged[(size_t)hf.columnId(x, z)].push_back(nav::Span{y, y, 1u});
            }
        std::vector<uint32_t> walkable; std::vector<int32_t> surfaceY;
        nav::FilterWalkableSpans(hf, cfg, merged, walkable, surfaceY);
        check(surfaceY[(size_t)hf.columnId(3, 1)] == 0, "NAV2 step: left surface y=0");
        check(surfaceY[(size_t)hf.columnId(4, 1)] == 20, "NAV2 step: right surface y=20");
        // The columns straddling the step are NOT connected (|0-20|=20 > climb 1).
        check(!nav::IsConnected(walkable[(size_t)hf.columnId(3, 1)], surfaceY[(size_t)hf.columnId(3, 1)],
                                walkable[(size_t)hf.columnId(4, 1)], surfaceY[(size_t)hf.columnId(4, 1)],
                                cfg.walkableClimb), "NAV2 step: across-step columns NOT connected");
        std::vector<uint32_t> dist;
        nav::BuildDistanceField(hf, cfg, walkable, surfaceY, dist);
        // Because the step breaks connectivity, neither side's interior is reachable from the OTHER side;
        // each side is its own thin strip bounded by its own borders + the step boundary -> distance
        // stays small (no bleed-through making a deep gradient across the full 9-wide grid).
        uint32_t maxd = 0u; for (uint32_t d : dist) if (d != nav::kDistInf && d > maxd) maxd = d;
        // A connected 9-wide strip would reach the centre column with a larger value; the broken strips
        // are each <=4 wide so the geodesic peak is bounded well under a full-width gradient.
        check(maxd <= 6u, "NAV2 step: distance does NOT bleed across the disconnected step");
    }

    // ================= NAV2: empty / all-non-walkable -> all-zero distance =================
    {
        nav::Heightfield hf = MakeHf(5, 5);
        nav::WalkableConfig cfg; cfg.walkableHeight = 2; cfg.walkableClimb = 1;
        std::vector<std::vector<nav::Span>> merged((size_t)hf.columnCount());   // no spans anywhere
        std::vector<uint32_t> walkable; std::vector<int32_t> surfaceY;
        nav::FilterWalkableSpans(hf, cfg, merged, walkable, surfaceY);
        bool noWalk = true; for (uint32_t w : walkable) if (w != 0u) noWalk = false;
        check(noWalk, "NAV2 empty: no walkable columns");
        std::vector<uint32_t> dist;
        nav::BuildDistanceField(hf, cfg, walkable, surfaceY, dist);
        bool allZero = true; for (uint32_t d : dist) if (d != 0u) allZero = false;
        check(allZero, "NAV2 empty: all-zero distance (no-op)");
    }

    if (g_fail == 0) { std::printf("nav_test OK\n"); return 0; }
    std::printf("nav_test: %d failures\n", g_fail);
    return 1;
}
