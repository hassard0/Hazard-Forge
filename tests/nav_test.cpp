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
#include <cstring>
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

    // ================= NAV3: BuildRegions — a single open basin -> exactly 1 region =================
    {
        // A flat connected ground: every walkable cell reachable from every other -> one basin.
        nav::Heightfield hf = MakeHf(10, 10);
        nav::WalkableConfig cfg; cfg.walkableHeight = 2; cfg.walkableClimb = 1;
        std::vector<std::vector<nav::Span>> merged((size_t)hf.columnCount());
        for (auto& m : merged) m.push_back(nav::Span{0u, 0u, 1u});
        std::vector<uint32_t> walkable; std::vector<int32_t> surfaceY;
        nav::FilterWalkableSpans(hf, cfg, merged, walkable, surfaceY);
        std::vector<uint32_t> dist;
        nav::BuildDistanceField(hf, cfg, walkable, surfaceY, dist);
        std::vector<uint32_t> region;
        const uint32_t maxDist = nav::MaxDistOf(dist);
        const uint32_t rc = nav::BuildRegions(hf, cfg, walkable, surfaceY, dist, maxDist, region);
        check(rc == 1u, "NAV3 open basin: exactly 1 region");
        // Every walkable cell (dist>0 interior) is assigned region 1; non-walkable / border (dist 0)
        // -> region 0. (Border cells are walkable but seeded dist 0 -> they get NO region.)
        bool walkAssigned = true, nonZeroOk = true;
        for (size_t c = 0; c < region.size(); ++c) {
            if (dist[c] > 0u && region[c] != 1u) walkAssigned = false;
            if (dist[c] == 0u && region[c] != 0u) nonZeroOk = false;
        }
        check(walkAssigned, "NAV3 open basin: every interior walkable cell -> region 1");
        check(nonZeroOk, "NAV3 open basin: every dist-0 cell -> region 0");
    }

    // ================= NAV3: a too-tall wall splits the space -> >=2 regions =================
    {
        // Left half y=0, right half a TALL step y=20 (step 20 > climb 1 -> disconnected). The two
        // sides become DIFFERENT regions; a cell on each side has a different region id.
        nav::Heightfield hf = MakeHf(11, 7);
        nav::WalkableConfig cfg; cfg.walkableHeight = 2; cfg.walkableClimb = 1;
        std::vector<std::vector<nav::Span>> merged((size_t)hf.columnCount());
        for (int z = 0; z < hf.h; ++z)
            for (int x = 0; x < hf.w; ++x) {
                const uint32_t y = (x < hf.w / 2) ? 0u : 20u;
                merged[(size_t)hf.columnId(x, z)].push_back(nav::Span{y, y, 1u});
            }
        std::vector<uint32_t> walkable; std::vector<int32_t> surfaceY;
        nav::FilterWalkableSpans(hf, cfg, merged, walkable, surfaceY);
        std::vector<uint32_t> dist;
        nav::BuildDistanceField(hf, cfg, walkable, surfaceY, dist);
        std::vector<uint32_t> region;
        const uint32_t maxDist = nav::MaxDistOf(dist);
        const uint32_t rc = nav::BuildRegions(hf, cfg, walkable, surfaceY, dist, maxDist, region);
        check(rc >= 2u, "NAV3 wall split: >=2 regions");
        // A cell on the left interior vs a cell on the right interior have DIFFERENT region ids.
        const size_t leftCell = (size_t)hf.columnId(2, 3);
        const size_t rightCell = (size_t)hf.columnId(8, 3);
        check(region[leftCell] != 0u && region[rightCell] != 0u,
              "NAV3 wall split: both sides have interior walkable cells assigned");
        check(region[leftCell] != region[rightCell],
              "NAV3 wall split: left-side region id != right-side region id");
    }

    // ================= NAV3: determinism — two BuildRegions runs byte-identical =================
    {
        nav::Heightfield hf = MakeHf(12, 12);
        nav::WalkableConfig cfg; cfg.walkableHeight = 2; cfg.walkableClimb = 1;
        std::vector<std::vector<nav::Span>> merged((size_t)hf.columnCount());
        // A mild bump in the middle (a y=2 patch) to make a non-trivial region structure.
        for (int z = 0; z < hf.h; ++z)
            for (int x = 0; x < hf.w; ++x) {
                const bool bump = (x >= 4 && x <= 7 && z >= 4 && z <= 7);
                const uint32_t y = bump ? 2u : 0u;
                merged[(size_t)hf.columnId(x, z)].push_back(nav::Span{y, y, 1u});
            }
        std::vector<uint32_t> walkable; std::vector<int32_t> surfaceY;
        nav::FilterWalkableSpans(hf, cfg, merged, walkable, surfaceY);
        std::vector<uint32_t> dist;
        nav::BuildDistanceField(hf, cfg, walkable, surfaceY, dist);
        const uint32_t maxDist = nav::MaxDistOf(dist);
        std::vector<uint32_t> r1, r2;
        const uint32_t c1 = nav::BuildRegions(hf, cfg, walkable, surfaceY, dist, maxDist, r1);
        const uint32_t c2 = nav::BuildRegions(hf, cfg, walkable, surfaceY, dist, maxDist, r2);
        check(c1 == c2, "NAV3 determinism: region count identical across two runs");
        check(r1 == r2, "NAV3 determinism: region[] byte-identical across two runs");

        // Connectivity: each region id forms a SINGLE 4-connected component (flood from the first
        // cell of each id over connected walkable neighbours; every cell of that id must be reached).
        const int w = hf.w, h = hf.h;
        bool allConnected = true;
        for (uint32_t id = 1u; id <= c1; ++id) {
            // find the first cell with this id.
            int startX = -1, startZ = -1;
            for (int z = 0; z < h && startX < 0; ++z)
                for (int x = 0; x < w; ++x)
                    if (r1[(size_t)hf.columnId(x, z)] == id) { startX = x; startZ = z; break; }
            if (startX < 0) continue;
            std::vector<uint8_t> seen((size_t)(w * h), 0u);
            std::vector<int> stack;
            stack.push_back(hf.columnId(startX, startZ));
            seen[(size_t)hf.columnId(startX, startZ)] = 1u;
            uint32_t reached = 0u;
            while (!stack.empty()) {
                const int cc = stack.back(); stack.pop_back();
                ++reached;
                const int cx = cc % w, cz = cc / w;
                const int nbrs[4][2] = {{cx, cz - 1}, {cx, cz + 1}, {cx - 1, cz}, {cx + 1, cz}};
                for (auto& nb : nbrs) {
                    const int nx = nb[0], nz = nb[1];
                    if (nx < 0 || nz < 0 || nx >= w || nz >= h) continue;
                    const int nc = hf.columnId(nx, nz);
                    if (seen[(size_t)nc]) continue;
                    if (r1[(size_t)nc] != id) continue;
                    seen[(size_t)nc] = 1u;
                    stack.push_back(nc);
                }
            }
            uint32_t total = 0u;
            for (uint32_t v : r1) if (v == id) ++total;
            if (reached != total) allConnected = false;
        }
        check(allConnected, "NAV3 connectivity: each region id is a single 4-connected component");
    }

    // ================= NAV3: empty / all-non-walkable -> 0 regions =================
    {
        nav::Heightfield hf = MakeHf(6, 6);
        nav::WalkableConfig cfg; cfg.walkableHeight = 2; cfg.walkableClimb = 1;
        std::vector<std::vector<nav::Span>> merged((size_t)hf.columnCount());   // no spans
        std::vector<uint32_t> walkable; std::vector<int32_t> surfaceY;
        nav::FilterWalkableSpans(hf, cfg, merged, walkable, surfaceY);
        std::vector<uint32_t> dist;
        nav::BuildDistanceField(hf, cfg, walkable, surfaceY, dist);
        std::vector<uint32_t> region;
        const uint32_t maxDist = nav::MaxDistOf(dist);
        const uint32_t rc = nav::BuildRegions(hf, cfg, walkable, surfaceY, dist, maxDist, region);
        check(rc == 0u, "NAV3 empty: 0 regions");
        bool allZero = true; for (uint32_t r : region) if (r != 0u) allZero = false;
        check(allZero, "NAV3 empty: region[] all zero (no-op)");
    }

    // ================= NAV4: TraceContours — a single square region -> a 4-vertex CW contour ========
    {
        // 4x4 grid; a 2x2 square region (cells (1,1),(2,1),(1,2),(2,2)) = region 1.
        nav::Heightfield hf = MakeHf(4, 4);
        std::vector<uint32_t> region((size_t)hf.columnCount(), 0u);
        auto set = [&](int x, int z) { region[(size_t)hf.columnId(x, z)] = 1u; };
        set(1, 1); set(2, 1); set(1, 2); set(2, 2);
        std::vector<nav::Contour> contours;
        nav::TraceContours(hf, region, 1u, contours);
        check(contours.size() == 1u, "NAV4 square: exactly 1 contour");
        check(!contours.empty() && contours[0].region == 1u, "NAV4 square: contour region id == 1");
        check(!contours.empty() && contours[0].verts.size() == 4u,
              "NAV4 square: 4-vertex contour loop");
        // The four corners of the 2x2 square: (1,1),(3,1),(3,3),(1,3).
        if (!contours.empty() && contours[0].verts.size() == 4u) {
            const auto& v = contours[0].verts;
            bool corners = v[0].x == 1 && v[0].z == 1 && v[1].x == 3 && v[1].z == 1 &&
                           v[2].x == 3 && v[2].z == 3 && v[3].x == 1 && v[3].z == 3;
            check(corners, "NAV4 square: contour corners (1,1)(3,1)(3,3)(1,3)");
        }

        // ----- BuildPolyMesh on the square -> 2 triangles sharing the diagonal, mutually adjacent. ---
        std::vector<nav::ContourVertex> simp;
        nav::SimplifyContour(contours[0].verts, 0, simp);   // maxError 0 -> keep all 4 corners.
        check(simp.size() == 4u, "NAV4 square: simplify(maxError=0) keeps 4 corners");
        std::vector<nav::Contour> sc; sc.push_back(nav::Contour{1u, simp});
        std::vector<nav::Poly> polys;
        nav::BuildPolyMesh(sc, polys);
        check(polys.size() == 2u, "NAV4 square: ear-clip -> 2 triangles");
        if (polys.size() == 2u) {
            // The two triangles share one edge (the diagonal) -> each is the other's neighbour on
            // exactly one edge, and the other two edges of each are boundary (kNoNeighbour).
            int adj0 = 0, adj1 = 0;
            for (int e = 0; e < 3; ++e) {
                if (polys[0].nbr[e] == 1u) ++adj0;
                if (polys[1].nbr[e] == 0u) ++adj1;
            }
            check(adj0 == 1 && adj1 == 1, "NAV4 square: 2 triangles mutually adjacent (shared diagonal)");
            int bnd0 = 0; for (int e = 0; e < 3; ++e) if (polys[0].nbr[e] == nav::kNoNeighbour) ++bnd0;
            check(bnd0 == 2, "NAV4 square: each triangle has 2 boundary edges");
        }
    }

    // ================= NAV4: an L-shaped region -> the expected simplified contour vert count ========
    {
        // 5x5 grid; an L: vertical bar x=1 (z=1..3) + horizontal foot z=3 (x=1..3) = region 1.
        nav::Heightfield hf = MakeHf(5, 5);
        std::vector<uint32_t> region((size_t)hf.columnCount(), 0u);
        auto set = [&](int x, int z) { region[(size_t)hf.columnId(x, z)] = 1u; };
        set(1, 1); set(1, 2); set(1, 3); set(2, 3); set(3, 3);
        std::vector<nav::Contour> contours;
        nav::TraceContours(hf, region, 1u, contours);
        check(contours.size() == 1u, "NAV4 L-shape: exactly 1 contour");
        // The L outline has 6 right-angle corners; maxError 0 keeps them all.
        std::vector<nav::ContourVertex> simp;
        nav::SimplifyContour(contours[0].verts, 0, simp);
        check(simp.size() == 6u, "NAV4 L-shape: simplified contour has 6 corners");
        // Ear-clip an n=6 contour -> n-2 = 4 triangles.
        std::vector<nav::Contour> sc; sc.push_back(nav::Contour{1u, simp});
        std::vector<nav::Poly> polys;
        nav::BuildPolyMesh(sc, polys);
        check(polys.size() == 4u, "NAV4 L-shape: ear-clip -> 4 triangles (n-2)");
        // Adjacency symmetry: a is b's neighbour <=> b is a's.
        bool symmetric = true;
        for (size_t i = 0; i < polys.size(); ++i)
            for (int e = 0; e < 3; ++e) {
                uint32_t q = polys[i].nbr[e];
                if (q == nav::kNoNeighbour) continue;
                bool back = false;
                for (int f = 0; f < 3; ++f) if (polys[q].nbr[f] == (uint32_t)i) back = true;
                if (!back) symmetric = false;
            }
        check(symmetric, "NAV4 L-shape: adjacency symmetric (a~b <=> b~a)");
    }

    // ================= NAV4: ear-clip produces n-2 triangles for a convex n-gon ======================
    {
        // A convex hexagon (CCW); ear-clip -> 6-2 = 4 triangles.
        std::vector<nav::ContourVertex> hex = {{2, 0}, {4, 0}, {6, 3}, {4, 6}, {2, 6}, {0, 3}};
        std::vector<nav::Contour> sc; sc.push_back(nav::Contour{1u, hex});
        std::vector<nav::Poly> polys;
        nav::BuildPolyMesh(sc, polys);
        check(polys.size() == 4u, "NAV4 n-gon: convex hexagon -> 4 triangles (n-2)");
        // Every poly carries the source region id.
        bool regionOk = true; for (const auto& p : polys) if (p.region != 1u) regionOk = false;
        check(regionOk, "NAV4 n-gon: every poly carries its source region id");
    }

    // ================= NAV4: determinism — two full pipelines byte-identical =========================
    {
        nav::Heightfield hf = MakeHf(5, 5);
        std::vector<uint32_t> region((size_t)hf.columnCount(), 0u);
        auto set = [&](int x, int z) { region[(size_t)hf.columnId(x, z)] = 1u; };
        set(1, 1); set(1, 2); set(1, 3); set(2, 3); set(3, 3);
        auto run = [&](std::vector<nav::Poly>& outPolys, std::vector<nav::Contour>& outC) {
            nav::TraceContours(hf, region, 1u, outC);
            for (auto& c : outC) {
                std::vector<nav::ContourVertex> s;
                nav::SimplifyContour(c.verts, 0, s);
                c.verts = s;
            }
            nav::BuildPolyMesh(outC, outPolys);
        };
        std::vector<nav::Contour> c1, c2;
        std::vector<nav::Poly> p1, p2;
        run(p1, c1); run(p2, c2);
        bool same = (p1.size() == p2.size()) &&
                    (p1.empty() || std::memcmp(p1.data(), p2.data(), p1.size() * sizeof(nav::Poly)) == 0);
        check(same, "NAV4 determinism: two full pipeline runs byte-identical");
    }

    // ================= NAV4: empty -> zero contours / zero polys =====================================
    {
        nav::Heightfield hf = MakeHf(6, 6);
        std::vector<uint32_t> region((size_t)hf.columnCount(), 0u);   // no regions.
        std::vector<nav::Contour> contours;
        nav::TraceContours(hf, region, 0u, contours);
        check(contours.empty(), "NAV4 empty: 0 contours");
        std::vector<nav::Poly> polys;
        nav::BuildPolyMesh(contours, polys);
        check(polys.empty(), "NAV4 empty: 0 polys (no-op)");
    }

    if (g_fail == 0) { std::printf("nav_test OK\n"); return 0; }
    std::printf("nav_test: %d failures\n", g_fail);
    return 1;
}
