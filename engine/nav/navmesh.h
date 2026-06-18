#pragma once
// Slice NAV1 — Deterministic GPU Navmesh BEACHHEAD: INTEGER HEIGHTFIELD SPAN RASTERIZATION (the
// FIRST slice of FLAGSHIP #7: DETERMINISTIC GPU NAVMESH + PATHFINDING). Pure CPU (header-only, no
// device, no backend symbols). Namespace hf::nav.
//
// WHAT THIS IS: the integer core of a Recast-style navmesh build — input world-space triangles are
// HOST-SNAPPED to an int32 voxel grid and RASTERIZED, per voxel COLUMN (x,z), into SOLID SPANS
// {ymin, ymax, area} (a heightfield). A column the triangle covers (its cell-center inside the
// triangle's XZ projection, by an integer edge-function sign test) gets a span equal to the
// triangle's integer y-range over that column's AABB; spans in a column are sorted by ymin and
// MERGED (overlapping/touching -> one). The GPU compute passes (shaders/nav_raster_count/scan/emit)
// copy the per-column math VERBATIM and prove bit-identical to this header's RasterizeTriangleSpans
// (memcmp GPU==CPU, NO tolerance) — the mc.h count->scan->emit / fpx.h int32 broadphase pattern
// applied to span rasterization.
//
// THE CROSS-BACKEND CRUX (why GPU==CPU + Vulkan==Metal hold bit-exactly): everything is PURE INT32 —
// the host snaps the verts to int32 voxel coords ONCE (the only float, build-time), the per-column
// cover test is integer edge functions (a 2D cross product of int32 deltas, int64 ONLY as the CPU
// reference's overflow-safe intermediate — the SHADER stays int32 because the snapped voxel coords
// are small, see the documented bound below), and the emitted span is integer min/max of the verts'
// quantized y. The per-column write is order-independent (each column's spans depend only on the
// triangles that cover it, processed in a fixed triangle order then sorted+merged deterministically),
// so a GPU thread-race CANNOT change the bytes. Integer-in -> integer-out -> identical bits on every
// vendor (the strict zero-differing-pixel bar, like mc.h / fpx.h broadphase / swraster / VSM).
//
// SEAM DISCIPLINE: ZERO backend (vk*/MTL*/mtl::/Backend::) symbols. NO GPU, NO new RHI. Mentions of
// "GPU" here are doc-only. INTEGER on every path (no float on the bit-exact path; the ONE float is
// the host-snap quantize of the input world verts). The nav_raster_* shaders are int32-only -> they
// MSL-generate NATIVELY on Metal (unlike fpx_integrate's int64 fxmul). REUSE MAP (file:line): the
// count->scan->emit structure mirrors engine/render/mc.h:425-592 (CountCells/PrefixSumOffsets/
// EmitCell/MarchCells); the int32 discipline + the per-thread count/emit-into-disjoint-range mirror
// engine/sim/fpx.h:239-289 (CountPairs/BuildPairs); the voxel quantize uses engine/sim/fpx.h:177
// FloorDiv (deterministic floor division for negative coords) — READ, NOT modified (nav is additive
// + parallel; fpx.h stays byte-identical).

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "sim/fpx.h"   // FloorDiv (deterministic floor-division for negative voxel coords); read-only

namespace hf::nav {

// ----- The host-snapped input triangle (int32 voxel coords) --------------------------------------
// The host snaps each world-space vertex to int32 voxel coords ONCE at build (the ONLY float, a
// build-time constant); from here everything is integer. A triangle is its 3 voxel-space vertices.
struct NavVert {
    int32_t x, y, z;
};
struct NavTri {
    NavVert v0, v1, v2;
};

// ----- A SOLID SPAN in a voxel column --------------------------------------------------------------
// [ymin, ymax] inclusive voxel-y interval of solid space in one (x,z) column; area = a walkable-flag
// PLACEHOLDER (= 1 for NAV1; NAV2's walkable-filter sets it for real). 3 x uint32 = 12 bytes, no
// padding holes (memcmp-able, std430-compatible). Sorted by ymin, non-overlapping within a column.
struct Span {
    uint32_t ymin;
    uint32_t ymax;
    uint32_t area;
};

// ----- The heightfield --------------------------------------------------------------------------
// An integer voxel grid: bmin/bmax are the int32 voxel BOUNDS (inclusive lo, exclusive hi in voxel
// units), w/h the column-grid dims (w = x columns, h = z columns). cs/ch are the world cell-size /
// cell-height the host used to snap (carried for provenance; the per-voxel math never touches them).
// The flat column id is col = z*w + x, x in [0,w), z in [0,h). spans[] is the compacted per-column
// span list, colOffset[col] the exclusive prefix-sum write base, colCount[col] the span count.
struct Heightfield {
    int32_t bminX = 0, bminY = 0, bminZ = 0;   // inclusive voxel-space lower bound
    int32_t bmaxX = 0, bmaxY = 0, bmaxZ = 0;   // exclusive voxel-space upper bound
    int32_t cs = 1;                            // world cell size (provenance; e.g. host-snap scale)
    int32_t ch = 1;                            // world cell height (provenance)
    int32_t w = 0, h = 0;                      // column-grid dims (w = #x columns, h = #z columns)

    int columnCount() const { return w * h; }
    // col = z*w + x (the mc.h cellId linearization, 2D here).
    int columnId(int x, int z) const { return z * w + x; }
};

// ----- Host-snap quantize (the ONE float crossing, build-time) -----------------------------------
// QuantizeCoord(worldFloatScaledToInt, bmin, cs): given a world coord ALREADY scaled to int units by
// the host (so this stays pure integer), return its voxel column index = FloorDiv(coord - bmin, cs).
// FloorDiv (engine/sim/fpx.h) is the DETERMINISTIC floor division (C++/HLSL `/` truncates toward
// zero, which lands a negative coord in the WRONG cell at the boundary) so triangles straddling the
// origin quantize correctly. Pure integer; the shader's column-quantize copies THIS.
inline int32_t QuantizeCoord(int32_t coord, int32_t bmin, int32_t cs) {
    return hf::sim::fpx::FloorDiv(coord - bmin, cs);
}

// ----- The integer triangle AABB over the column grid --------------------------------------------
// TriColumnAabb(tri, hf): the inclusive [x0,x1] x [z0,z1] column range the triangle's voxel-space XZ
// AABB covers, clamped to [0,w) x [0,h). The tri verts are ALREADY voxel coords (host-snapped), so
// the column index of a vert IS its voxel x/z (bmin already subtracted at snap time). Pure integer
// min/max + clamp. Returns false if the clamped range is empty (the triangle misses the grid).
struct ColumnAabb {
    int32_t x0, x1, z0, z1;   // inclusive column ranges
};
inline bool TriColumnAabb(const NavTri& t, const Heightfield& hf, ColumnAabb& out) {
    int32_t minX = t.v0.x, maxX = t.v0.x, minZ = t.v0.z, maxZ = t.v0.z;
    if (t.v1.x < minX) minX = t.v1.x; if (t.v1.x > maxX) maxX = t.v1.x;
    if (t.v2.x < minX) minX = t.v2.x; if (t.v2.x > maxX) maxX = t.v2.x;
    if (t.v1.z < minZ) minZ = t.v1.z; if (t.v1.z > maxZ) maxZ = t.v1.z;
    if (t.v2.z < minZ) minZ = t.v2.z; if (t.v2.z > maxZ) maxZ = t.v2.z;
    // Clamp to the column grid [0,w) x [0,h).
    if (minX < 0) minX = 0;
    if (minZ < 0) minZ = 0;
    if (maxX > hf.w - 1) maxX = hf.w - 1;
    if (maxZ > hf.h - 1) maxZ = hf.h - 1;
    out.x0 = minX; out.x1 = maxX; out.z0 = minZ; out.z1 = maxZ;
    return minX <= maxX && minZ <= maxZ;
}

// ----- The integer cover test (edge-function sign test) ------------------------------------------
// PointInTriXZ(px, pz, t): is the column-center sample (px, pz) inside the triangle's XZ projection?
// A 2D edge function E(a, b, p) = (b.x-a.x)*(p.z-a.z) - (b.z-a.z)*(p.x-a.x) (the signed area / cross
// product of the edge a->b with a->p). The point is inside iff the three edge signs are all >=0 OR
// all <=0 (a consistent orientation -> covers both winding orders; the fpx/swraster integer-edge
// discipline). Pure integer; on the CPU the products use int64 to be overflow-safe (the reference),
// while the SHADER stays int32 because the host-snapped voxel coords are small (see the bound in
// nav_raster_count.comp). The shader copies THIS body's sign logic VERBATIM (int32 products).
inline bool PointInTriXZ(int32_t px, int32_t pz, const NavTri& t) {
    // Edge functions as int64 (overflow-safe reference). For the modest voxel-grid coords NAV1 uses
    // (|coord| <= a few thousand), the int32 products in the shader are identical bit-for-bit.
    const int64_t e0 = (int64_t)(t.v1.x - t.v0.x) * (int64_t)(pz - t.v0.z) -
                       (int64_t)(t.v1.z - t.v0.z) * (int64_t)(px - t.v0.x);
    const int64_t e1 = (int64_t)(t.v2.x - t.v1.x) * (int64_t)(pz - t.v1.z) -
                       (int64_t)(t.v2.z - t.v1.z) * (int64_t)(px - t.v1.x);
    const int64_t e2 = (int64_t)(t.v0.x - t.v2.x) * (int64_t)(pz - t.v2.z) -
                       (int64_t)(t.v0.z - t.v2.z) * (int64_t)(px - t.v2.x);
    const bool allNonNeg = (e0 >= 0) && (e1 >= 0) && (e2 >= 0);
    const bool allNonPos = (e0 <= 0) && (e1 <= 0) && (e2 <= 0);
    return allNonNeg || allNonPos;
}

// ----- The triangle's integer y-interval ---------------------------------------------------------
// TriYSpan(t): the conservative integer [ymin, ymax] voxel-y interval of the triangle (the min/max of
// its 3 voxel-y verts). NAV1 emits this whole interval into every column the triangle covers (a
// conservative solid span — the Recast rasterizeTri discipline kept integer; per-column slope
// interpolation is a deferred refinement). Pure integer. The shader copies THIS VERBATIM.
inline void TriYSpan(const NavTri& t, int32_t& ymin, int32_t& ymax) {
    ymin = t.v0.y; ymax = t.v0.y;
    if (t.v1.y < ymin) ymin = t.v1.y; if (t.v1.y > ymax) ymax = t.v1.y;
    if (t.v2.y < ymin) ymin = t.v2.y; if (t.v2.y > ymax) ymax = t.v2.y;
}

// The walkable-flag placeholder area NAV1 stamps on every emitted span (NAV2 sets it for real).
static constexpr uint32_t kDefaultArea = 1u;

// ----- ColumnSpanCount: spans contributed to ONE column (the count pass per-column math) ----------
// For column (cx,cz), count how many of the input triangles cover its cell-center (cx,cz) — i.e.
// PointInTriXZ(cx, cz, tri) over the triangles whose ColumnAabb includes (cx,cz). This is the RAW
// (pre-merge) span count the count pass writes. The merge happens in the emit/finalize pass; the
// count buffer is the UPPER BOUND (raw covering count) the prefix-sum reserves space for. Pure
// integer. The shader nav_raster_count.comp computes THIS per thread (one thread per column).
inline uint32_t ColumnSpanCount(int32_t cx, int32_t cz, const Heightfield& hf,
                                std::span<const NavTri> tris) {
    uint32_t c = 0;
    for (const NavTri& t : tris) {
        ColumnAabb ab;
        if (!TriColumnAabb(t, hf, ab)) continue;
        if (cx < ab.x0 || cx > ab.x1 || cz < ab.z0 || cz > ab.z1) continue;
        if (PointInTriXZ(cx, cz, t)) ++c;
    }
    return c;
}

// ----- The CPU reference: RasterizeTriangleSpans -------------------------------------------------
// The full reference the GPU memcmp's against. Two-phase (mirrors the GPU count->scan->emit):
//   1) per column, RAW count = #triangles covering its center (ColumnSpanCount) -> colCount[].
//   2) exclusive prefix-sum of colCount -> colOffset[] (the per-column write base) + total.
//   3) per column, EMIT each covering triangle's TriYSpan into spans[] at its offset, in the FIXED
//      triangle order; the spans buffer is the RAW (un-merged) emit — DETERMINISTIC by construction
//      (triangle order is fixed, each column's range is disjoint). NAV1's GPU emit matches THIS raw
//      buffer bit-for-bit. (Merging overlapping spans is a separate pure-CPU finalize step,
//      MergeColumnSpans, exposed for the nav_test / NAV2 walkable pass; it is NOT part of the GPU
//      bit-exact buffer because the merge is column-local + variable-length.)
// Fills out.colCount / out.colOffset / out.spans (spans sized to total). Pure integer.
inline void RasterizeTriangleSpans(const Heightfield& hf, std::span<const NavTri> tris,
                                   std::vector<uint32_t>& colCount,
                                   std::vector<uint32_t>& colOffset,
                                   std::vector<Span>& spans) {
    const size_t nCols = (size_t)hf.columnCount();
    colCount.assign(nCols, 0u);
    colOffset.assign(nCols, 0u);

    // 1) per-column raw count.
    for (int cz = 0; cz < hf.h; ++cz)
        for (int cx = 0; cx < hf.w; ++cx)
            colCount[(size_t)hf.columnId(cx, cz)] = ColumnSpanCount(cx, cz, hf, tris);

    // 2) exclusive prefix-sum -> per-column write base + total (the mc.h PrefixSumOffsets serial scan).
    uint32_t running = 0u;
    for (size_t c = 0; c < nCols; ++c) {
        colOffset[c] = running;
        running += colCount[c];
    }
    const uint32_t total = running;

    // 3) per-column emit each covering triangle's y-span at the column's offset (fixed tri order ->
    // the same order the GPU emit walks; each column's [offset, offset+count) range is disjoint).
    spans.assign((size_t)total, Span{0u, 0u, 0u});
    for (int cz = 0; cz < hf.h; ++cz)
        for (int cx = 0; cx < hf.w; ++cx) {
            const uint32_t base = colOffset[(size_t)hf.columnId(cx, cz)];
            uint32_t local = 0u;
            for (uint32_t ti = 0; ti < (uint32_t)tris.size(); ++ti) {
                const NavTri& t = tris[ti];
                ColumnAabb ab;
                if (!TriColumnAabb(t, hf, ab)) continue;
                if (cx < ab.x0 || cx > ab.x1 || cz < ab.z0 || cz > ab.z1) continue;
                if (!PointInTriXZ(cx, cz, t)) continue;
                int32_t ymin, ymax;
                TriYSpan(t, ymin, ymax);
                spans[(size_t)(base + local)] = Span{(uint32_t)ymin, (uint32_t)ymax, kDefaultArea};
                ++local;
            }
        }
}

// ----- MergeColumnSpans: the deterministic span merge (pure-CPU finalize) ------------------------
// Given a column's RAW spans (any order, possibly overlapping), produce the sorted-by-ymin,
// non-overlapping merged list (touching/overlapping spans -> one, area carried from the first). The
// deterministic Recast span-merge: sort by ymin (stable, tie-break ymax), then sweep merging any span
// whose ymin <= current ymax+1 (touching counts as overlapping for a contiguous solid column). Pure
// integer. Exposed for nav_test + NAV2; NOT part of the GPU bit-exact buffer (variable-length).
inline std::vector<Span> MergeColumnSpans(std::vector<Span> raw) {
    if (raw.empty()) return raw;
    std::sort(raw.begin(), raw.end(), [](const Span& a, const Span& b) {
        if (a.ymin != b.ymin) return a.ymin < b.ymin;
        return a.ymax < b.ymax;
    });
    std::vector<Span> merged;
    merged.push_back(raw[0]);
    for (size_t i = 1; i < raw.size(); ++i) {
        Span& cur = merged.back();
        const Span& s = raw[i];
        // Touching (s.ymin <= cur.ymax + 1) or overlapping -> extend the current span.
        if (s.ymin <= cur.ymax + 1u) {
            if (s.ymax > cur.ymax) cur.ymax = s.ymax;
        } else {
            merged.push_back(s);
        }
    }
    return merged;
}

// ----- A deterministic showcase scene: a ground plane + a ramp + a box step -----------------------
// MakeShowcaseTriangles(hf): a small fixed set of host-snapped voxel triangles — a full ground quad
// (2 tris covering the whole column grid), a raised box/step (a quad at an elevated y over a sub-rect
// of the grid), and a ramp (a sloped quad). All coords are int32 voxel units already (the host-snap
// is folded into these constants for determinism). Returns the triangle list the showcase rasterizes
// into a coherent heightfield (the ground fills every column; the box/ramp add stacked spans).
inline std::vector<NavTri> MakeShowcaseTriangles(const Heightfield& hf) {
    std::vector<NavTri> tris;
    const int32_t x0 = 0, x1 = hf.w - 1, z0 = 0, z1 = hf.h - 1;

    // Ground plane at y=0 covering the whole grid (two tris of the quad [x0,z0]-[x1,z1]).
    tris.push_back(NavTri{NavVert{x0, 0, z0}, NavVert{x1, 0, z0}, NavVert{x1, 0, z1}});
    tris.push_back(NavTri{NavVert{x0, 0, z0}, NavVert{x1, 0, z1}, NavVert{x0, 0, z1}});

    // A raised box-top step at y=8 over a centered sub-rectangle (stacks a second span over its cols).
    const int32_t bx0 = hf.w / 4, bx1 = hf.w / 2, bz0 = hf.h / 4, bz1 = hf.h / 2;
    const int32_t boxY = 8;
    tris.push_back(NavTri{NavVert{bx0, boxY, bz0}, NavVert{bx1, boxY, bz0}, NavVert{bx1, boxY, bz1}});
    tris.push_back(NavTri{NavVert{bx0, boxY, bz0}, NavVert{bx1, boxY, bz1}, NavVert{bx0, boxY, bz1}});

    // A ramp (sloped quad) over another sub-rectangle, y rising from 0 to 12 across z.
    const int32_t rx0 = hf.w / 2, rx1 = (3 * hf.w) / 4, rz0 = hf.h / 2, rz1 = (3 * hf.h) / 4;
    tris.push_back(NavTri{NavVert{rx0, 0, rz0}, NavVert{rx1, 0, rz0}, NavVert{rx1, 12, rz1}});
    tris.push_back(NavTri{NavVert{rx0, 0, rz0}, NavVert{rx1, 12, rz1}, NavVert{rx0, 12, rz1}});

    return tris;
}

// =================================================================================================
// Slice NAV2 — WALKABLE FILTER + INTEGER DISTANCE FIELD (additive over the NAV1 heightfield above).
// Pure integer (NO <cmath>, NO float, NO int64, NO backend symbols). The CPU reference the GPU
// nav_filter.comp / nav_distance.comp shaders copy VERBATIM + memcmp bit-identical against. Operates
// on the NAV1 MERGED per-column spans (MergeColumnSpans) so clearance-above is correct.
// =================================================================================================

// ----- WalkableConfig: the agent walkability parameters (voxel units) ----------------------------
// walkableHeight = the min clearance (in voxel-y) an agent needs to stand on a span's top (the gap to
// the next solid span above, or to the heightfield top, must be >= this). walkableClimb = the max
// step height (in voxel-y) an agent can climb between two adjacent walkable columns (the 4-neighbour
// max-step connectivity test). Both pure integer compares (the fpx.h::AabbOverlap discipline).
struct WalkableConfig {
    int walkableHeight = 1;   // min vertical clearance above a walkable surface (voxels)
    int walkableClimb = 1;    // max step between adjacent walkable surfaces (voxels)
};

// A sentinel "infinity" distance for the chamfer seed (a large int32, far inside int32 range so
// dist+3 never overflows). The grid is small (w,h <= a few thousand) and the chamfer weights are 2/3,
// so a true geodesic distance is bounded by ~3*(w+h) << kDistInf; kDistInf is purely the unreachable
// seed for walkable cells before the sweeps relax them.
static constexpr uint32_t kDistInf = 0x3FFFFFFFu;   // ~1.07e9, < INT32_MAX; dist+3 stays in int32

// ----- FilterWalkableSpans: mark walkable spans + derive the per-column walkable surface ----------
// Given the NAV1 heightfield and, PER COLUMN, that column's MERGED spans (MergeColumnSpans output,
// sorted-by-ymin non-overlapping), for each column:
//   1) Walk its merged spans from TOP to BOTTOM. A span's TOP is a WALKABLE surface iff the clearance
//      to the next solid span ABOVE it (gap = aboveSpan.ymin - thisSpan.ymax - 1), or to the
//      heightfield top (gap = (bmaxY-1) - thisSpan.ymax) if it is the topmost span, is >=
//      walkableHeight. Set span.area = 1 (walkable) else 0 (mutates mergedSpansPerColumn in place).
//   2) walkable[col] = 1 iff the column has >= 1 walkable span; surfaceY[col] = the TOP walkable
//      span's top-y (ymax) — the topmost walkable surface, the cell the distance field uses (0 if no
//      walkable span, and walkable[col]==0 marks it not a surface).
// Then a SECOND pass applies the 4-neighbour max-step CONNECTIVITY test: a walkable column is
// CONNECTED to a neighbour iff that neighbour is walkable AND abs(surfaceY[col]-surfaceY[nbr]) <=
// walkableClimb. A walkable column with NO connected walkable neighbour on a side borders the
// non-walkable region (a distance-field seed comes from the BuildDistanceField border/non-walkable
// rule below; connectivity is consumed there to keep the distance GEODESIC). Pure integer. The shader
// nav_filter.comp computes walkable[]+surfaceY[] per thread (one thread per column).
//
// mergedSpansPerColumn: a vector of size hf.columnCount(); entry [col] is that column's merged spans
// (already MergeColumnSpans'd). walkable[] / surfaceY[] are (re)sized to columnCount() and filled.
inline void FilterWalkableSpans(const Heightfield& hf, const WalkableConfig& cfg,
                                std::vector<std::vector<Span>>& mergedSpansPerColumn,
                                std::vector<uint32_t>& walkable,
                                std::vector<int32_t>& surfaceY) {
    const size_t nCols = (size_t)hf.columnCount();
    walkable.assign(nCols, 0u);
    surfaceY.assign(nCols, 0);
    const int32_t fieldTop = hf.bmaxY - 1;   // inclusive top voxel-y of the heightfield

    for (size_t c = 0; c < nCols; ++c) {
        std::vector<Span>& spans = mergedSpansPerColumn[c];
        // Merged spans are sorted ascending by ymin (NAV1::MergeColumnSpans). Walk TOP -> BOTTOM (the
        // last entry is the highest span). For span i, the span ABOVE is span i+1 (higher ymin).
        int32_t topWalkableY = 0;
        bool anyWalkable = false;
        for (size_t i = spans.size(); i-- > 0;) {
            Span& s = spans[i];
            int32_t clearance;
            if (i + 1 < spans.size()) {
                // Gap to the next solid span above (its ymin - this ymax - 1).
                clearance = (int32_t)spans[i + 1].ymin - (int32_t)s.ymax - 1;
            } else {
                // Topmost span: clearance to the heightfield top.
                clearance = fieldTop - (int32_t)s.ymax;
            }
            const bool isWalkable = clearance >= cfg.walkableHeight;
            s.area = isWalkable ? 1u : 0u;
            if (isWalkable && !anyWalkable) {
                // The first walkable surface seen walking top->bottom IS the topmost walkable surface.
                topWalkableY = (int32_t)s.ymax;
                anyWalkable = true;
            }
        }
        if (anyWalkable) {
            walkable[c] = 1u;
            surfaceY[c] = topWalkableY;
        }
    }
    // NOTE: the 4-neighbour max-step CONNECTIVITY test (abs(surfaceY[col]-surfaceY[nbr]) <=
    // walkableClimb) is applied lazily inside BuildDistanceField (a neighbour is only traversed if
    // walkable AND connected), so the distance is geodesic over the walkable surface. surfaceY[] +
    // walkable[] are the inputs that test needs; no separate connectivity buffer is materialized
    // (YAGNI — the spec's connectivity is a property of the distance sweep, not a stored array).
    (void)cfg;
}

// ----- IsConnected: the 4-neighbour max-step connectivity predicate (pure integer) ----------------
// Two walkable columns a,b (adjacent) are CONNECTED iff both walkable AND the step between their
// surfaces is within climb: abs(surfaceY[a]-surfaceY[b]) <= walkableClimb. VERBATIM the shader test.
inline bool IsConnected(uint32_t walkA, int32_t surfA, uint32_t walkB, int32_t surfB, int32_t climb) {
    if (walkA == 0u || walkB == 0u) return false;
    int32_t d = surfA - surfB;
    if (d < 0) d = -d;
    return d <= climb;
}

// ----- BuildDistanceField: the integer two-pass chamfer distance transform --------------------------
// Over the w x h walkable grid, compute dist[col] = the integer chamfer distance from each walkable
// cell to the nearest non-walkable / border / non-connected boundary. Seed: non-walkable cells (and
// the grid border) = 0; walkable cells = kDistInf. Two sweeps (the standard Recast integer chamfer,
// cardinal weight 2 / diagonal weight 3, NO sqrt, NO int64):
//   FORWARD  row-major TL->BR: relax against the already-visited up/left neighbours (W, NW, N, NE).
//   BACKWARD reverse  BR->TL: relax against the down/right neighbours (E, SE, S, SW).
// A neighbour is only traversed if it is walkable AND CONNECTED to the cell (the max-step test) — so
// the distance is GEODESIC over the walkable surface, not Euclidean-through-walls. Deterministic
// single-thread serial -> bit-exact (the nav_raster_scan single-thread mirror). Output dist[] (w*h).
inline void BuildDistanceField(const Heightfield& hf, const WalkableConfig& cfg,
                               const std::vector<uint32_t>& walkable,
                               const std::vector<int32_t>& surfaceY,
                               std::vector<uint32_t>& dist) {
    const int w = hf.w, h = hf.h;
    const size_t nCols = (size_t)hf.columnCount();
    dist.assign(nCols, 0u);
    const int32_t climb = cfg.walkableClimb;

    // Seed: walkable interior = kDistInf, everything else (non-walkable OR border) = 0.
    for (int z = 0; z < h; ++z)
        for (int x = 0; x < w; ++x) {
            const size_t c = (size_t)hf.columnId(x, z);
            const bool border = (x == 0 || z == 0 || x == w - 1 || z == h - 1);
            dist[c] = (walkable[c] != 0u && !border) ? kDistInf : 0u;
        }

    const uint32_t kCard = 2u, kDiag = 3u;
    // Relax dist[c] against neighbour (nx,nz) with weight wgt IF the neighbour is walkable+connected.
    auto relax = [&](size_t c, int cx, int cz, int nx, int nz, uint32_t wgt) {
        if (nx < 0 || nz < 0 || nx >= w || nz >= h) return;
        const size_t nc = (size_t)hf.columnId(nx, nz);
        if (!IsConnected(walkable[c], surfaceY[c], walkable[nc], surfaceY[nc], climb)) return;
        const uint32_t cand = dist[nc] + wgt;
        if (cand < dist[c]) dist[c] = cand;
    };

    // FORWARD sweep TL->BR: up/left neighbours (W, NW, N, NE) already finalized this pass.
    for (int z = 0; z < h; ++z)
        for (int x = 0; x < w; ++x) {
            const size_t c = (size_t)hf.columnId(x, z);
            if (dist[c] == 0u) continue;   // a seed (0) can only stay 0
            relax(c, x, z, x - 1, z,     kCard);   // W
            relax(c, x, z, x - 1, z - 1, kDiag);   // NW
            relax(c, x, z, x,     z - 1, kCard);   // N
            relax(c, x, z, x + 1, z - 1, kDiag);   // NE
        }
    // BACKWARD sweep BR->TL: down/right neighbours (E, SE, S, SW).
    for (int z = h - 1; z >= 0; --z)
        for (int x = w - 1; x >= 0; --x) {
            const size_t c = (size_t)hf.columnId(x, z);
            if (dist[c] == 0u) continue;
            relax(c, x, z, x + 1, z,     kCard);   // E
            relax(c, x, z, x + 1, z + 1, kDiag);   // SE
            relax(c, x, z, x,     z + 1, kCard);   // S
            relax(c, x, z, x - 1, z + 1, kDiag);   // SW
        }
    // Any walkable cell still at kDistInf (an isolated walkable island unreachable from a boundary)
    // is clamped to 0 so the read-back never carries the sentinel (deterministic, documented).
    for (size_t c = 0; c < nCols; ++c)
        if (dist[c] == kDistInf) dist[c] = 0u;
}

// =================================================================================================
// Slice NAV3 — WATERSHED REGION GENERATION (additive over the NAV2 distance field above). Pure
// integer (NO <cmath>, NO float, NO int64, NO backend symbols). The CPU reference the GPU
// nav_region.comp shader copies VERBATIM + memcmp bit-identical against. The MAKE-OR-BREAK slice:
// an integer watershed partitions the walkable distance field into REGIONS (connected basins) so
// each gets a distinct deterministic id, bit-exact CPU<->Vulkan<->Metal by a FIXED scan order.
// =================================================================================================

// ----- MaxDistOf: the peak chamfer distance over the field (the watershed's descending loop bound) --
// Returns the largest dist[] value (the highest "water level"); 0 if the field is empty/all-zero. The
// watershed descends level = maxDist..1. Pure integer; the shader recomputes the same max.
inline uint32_t MaxDistOf(const std::vector<uint32_t>& dist) {
    uint32_t m = 0u;
    for (uint32_t d : dist)
        if (d != kDistInf && d > m) m = d;
    return m;
}

// ----- BuildRegions: the LOCKED level-descending fixed-order integer watershed (the make-or-break) --
// Partitions the NAV2 walkable distance field into REGIONS: each connected walkable basin gets a
// distinct deterministic region id. region[c] = 0 for non-walkable / unassigned-isolated cells; ids
// are dense from 1. The algorithm (the spec's LOCKED pseudocode, verbatim — EVERY ordering decision
// is PINNED so the converged assignment is identical CPU<->GPU<->both backends):
//   for level = maxDist down to 1 (descending water level — ridge tops first):
//     (A) GROW: fixed-point expansion of existing regions into THIS level's unassigned cells. Repeat
//         a full ASCENDING-cellId scan until no change: an unassigned walkable cell at dist==level
//         adopts the LOWEST region id among its 4 neighbours (fixed order up,down,left,right) that are
//         assigned AND IsConnected (the NAV2 max-step predicate).
//     (B) SEED: any still-unassigned walkable cell AT this level (ASCENDING cellId) starts a NEW
//         region (nextRegion++), then that seed is GROWN across this level (same fixed-point scan,
//         restricted to dist==level cells connected to a cell already in the seed's region).
// Single-thread serial (the shader is [numthreads(1,1,1)]) so there is NO GPU race; the fixed scan
// order + lowest-id tie-break make the result order-independent regardless. Pure int32 (region ids /
// levels are small). Output region[] (one uint per column; 0 = none). regionCount = the returned
// nextRegion-1 (also derivable as the max region id).
inline uint32_t BuildRegions(const Heightfield& hf, const WalkableConfig& cfg,
                             const std::vector<uint32_t>& walkable,
                             const std::vector<int32_t>& surfaceY,
                             const std::vector<uint32_t>& dist, uint32_t maxDist,
                             std::vector<uint32_t>& region) {
    const int w = hf.w, h = hf.h;
    const size_t nCols = (size_t)hf.columnCount();
    region.assign(nCols, 0u);
    const int32_t climb = cfg.walkableClimb;
    uint32_t nextRegion = 1u;

    // connected(c, nx, nz): is in-grid neighbour (nx,nz) walkable AND within climb of cell c?
    // (the NAV2 IsConnected max-step predicate, applied to a 4-neighbour). Returns the neighbour's
    // flat id in nc (valid only when it returns true).
    auto connected = [&](size_t c, int nx, int nz, size_t& nc) -> bool {
        if (nx < 0 || nz < 0 || nx >= w || nz >= h) return false;
        nc = (size_t)hf.columnId(nx, nz);
        return IsConnected(walkable[c], surfaceY[c], walkable[nc], surfaceY[nc], climb);
    };

    // Descend level = maxDist..1. Use a signed loop counter so the >=1 guard terminates (an unsigned
    // counter would wrap below 1 and never end). maxDist is small (~28), well inside int range.
    for (int32_t level = (int32_t)maxDist; level >= 1; --level) {
        // (A) GROW: existing regions expand into this level's unassigned cells (fixed-point).
        bool changed = true;
        while (changed) {
            changed = false;
            for (int z = 0; z < h; ++z)
                for (int x = 0; x < w; ++x) {
                    const size_t c = (size_t)hf.columnId(x, z);
                    if (region[c] != 0u || walkable[c] == 0u || dist[c] != (uint32_t)level) continue;
                    // Adopt the LOWEST region id among the 4 neighbours assigned AND connected
                    // (fixed neighbour order: up (z-1), down (z+1), left (x-1), right (x+1)).
                    uint32_t best = 0u;
                    size_t nc;
                    if (connected(c, x, z - 1, nc) && region[nc] != 0u) { if (best == 0u || region[nc] < best) best = region[nc]; }
                    if (connected(c, x, z + 1, nc) && region[nc] != 0u) { if (best == 0u || region[nc] < best) best = region[nc]; }
                    if (connected(c, x - 1, z, nc) && region[nc] != 0u) { if (best == 0u || region[nc] < best) best = region[nc]; }
                    if (connected(c, x + 1, z, nc) && region[nc] != 0u) { if (best == 0u || region[nc] < best) best = region[nc]; }
                    if (best != 0u) { region[c] = best; changed = true; }
                }
        }
        // (B) SEED: any still-unassigned walkable cell AT this level starts a NEW region (ascending
        // cellId), then is grown across this level (same fixed-point, restricted to dist==level cells
        // connected to a cell already in THIS seed's region).
        for (int z = 0; z < h; ++z)
            for (int x = 0; x < w; ++x) {
                const size_t c = (size_t)hf.columnId(x, z);
                if (region[c] != 0u || walkable[c] == 0u || dist[c] != (uint32_t)level) continue;
                const uint32_t thisSeed = nextRegion;
                region[c] = thisSeed;
                ++nextRegion;
                // Grow this seed across the current level.
                bool grew = true;
                while (grew) {
                    grew = false;
                    for (int gz = 0; gz < h; ++gz)
                        for (int gx = 0; gx < w; ++gx) {
                            const size_t c2 = (size_t)hf.columnId(gx, gz);
                            if (region[c2] != 0u || walkable[c2] == 0u || dist[c2] != (uint32_t)level) continue;
                            size_t nc;
                            bool adopt =
                                (connected(c2, gx, gz - 1, nc) && region[nc] == thisSeed) ||
                                (connected(c2, gx, gz + 1, nc) && region[nc] == thisSeed) ||
                                (connected(c2, gx - 1, gz, nc) && region[nc] == thisSeed) ||
                                (connected(c2, gx + 1, gz, nc) && region[nc] == thisSeed);
                            if (adopt) { region[c2] = thisSeed; grew = true; }
                        }
                }
            }
    }
    return nextRegion - 1u;   // regionCount (0 if no walkable cells got a region)
}

}  // namespace hf::nav
