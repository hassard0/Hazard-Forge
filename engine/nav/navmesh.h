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
#include <array>
#include <cstdint>
#include <span>
#include <utility>
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

// =================================================================================================
// Slice NAV4 — CONTOUR TRACING + INTEGER POLYGONIZATION (additive over the NAV3 region partition
// above). Pure integer (NO <cmath>, NO float, NO int64 on the bit-exact path — int32 only; see the
// overflow bound below). The CPU reference the GPU nav_contour.comp / nav_polygonize.comp shaders
// copy VERBATIM + memcmp bit-identical against. Turns each NAV3 region into a closed integer CONTOUR
// (a deterministic boundary walk), SIMPLIFIES it (integer Douglas–Peucker, perpendicular-distance
// SQUARED — no sqrt), and TRIANGULATES it into convex polygons (ear-clip) + per-edge cross-poly
// ADJACENCY (the graph NAV5's A* runs over). Every ordering decision is PINNED so the result is
// identical CPU<->GPU<->both backends (the NAV3 single-thread discipline, extended to 3 sequential
// algorithms).
//
// THE INT32 OVERFLOW BOUND (why the shaders stay Metal-MSL-native, int32 only): all coordinates are
// cell-corner voxel ints in [0, max(w,h)]. The showcase grid is 32x32 -> corner coords in [0,32].
//   - Douglas–Peucker perpendicular-distance-squared: cross = (bx-ax)*(pz-az) - (bz-az)*(px-ax);
//     |delta| <= 32 -> |cross| <= 2*32*32 = 2048 -> cross*cross <= ~4.2e6; the denominator
//     dx*dx+dz*dz <= 2*32*32 = 2048; numerator = cross*cross <= 4.2e6 << INT32_MAX (~2.1e9). Safe.
//   - Ear-clip orientation / point-in-triangle: the SAME PointInTriXZ int64 reference reduces to int32
//     products (each <= 32*64 = 2048) for these coords (NAV1's documented bound). Safe.
// For a general (larger) grid where corner coords could exceed ~32767, the DP numerator (cross*cross,
// a 4th-power term) would need int64 -> that ONE shader would go Vulkan-only + CPU-ref on Metal (the
// FPX1/swraster convention). For NAV4's bounded showcase int32 is exact and the shaders are native.
// =================================================================================================

// ----- ContourVertex: an integer cell-CORNER vertex on a region boundary loop --------------------
// (x,z) are corner-lattice voxel coords: a cell (cx,cz) occupies the unit square with corners
// (cx,cz)..(cx+1,cz+1), so corner coords range [0, w] x [0, h]. A contour is a closed CCW/CW loop of
// these (the walk pins one fixed winding). 2 x int32 = 8 bytes, no padding (memcmp-able, std430).
struct ContourVertex {
    int32_t x, z;
};

// ----- Contour: one region's closed boundary loop ------------------------------------------------
// region = the NAV3 region id this loop bounds; verts = the (simplified) closed integer corner loop
// (implicitly closed: the last vertex connects back to the first; the loop is NOT duplicated).
struct Contour {
    uint32_t region = 0u;
    std::vector<ContourVertex> verts;
};

// ----- Poly: one convex polygon (a triangle for NAV4) + its per-edge neighbour ids ----------------
// idx[0..2] = indices into the OWNING contour's simplified vertex list (a triangle, CCW by the
// ear-clip winding). nbr[e] = the poly id (index into the BuildPolyMesh output) sharing edge e
// (idx[e]->idx[(e+1)%3]); kNoNeighbour if that edge is a contour boundary (no adjacent poly). region
// carries the source region id (for the showcase coloring + NAV5). 8 x uint32 = 32 bytes (std430).
static constexpr uint32_t kNoNeighbour = 0xFFFFFFFFu;
struct Poly {
    uint32_t idx[3];     // contour-local vertex indices (triangle, CCW)
    uint32_t nbr[3];     // per-edge neighbour poly id (global), kNoNeighbour if boundary
    uint32_t region;     // source NAV3 region id
    uint32_t _pad;       // -> 32 bytes, std430-clean, memcmp-able
};

// ----- Edge2 (signed area / cross product) — the integer orientation primitive --------------------
// Cross2(ax,az, bx,bz, px,pz) = (bx-ax)*(pz-az) - (bz-az)*(px-ax): twice the signed area of triangle
// (a,b,p) — the SAME 2D edge function NAV1's PointInTriXZ uses. >0 = p left of a->b (CCW), <0 = right
// (CW), 0 = collinear. Pure int32 for the bounded grid (see the overflow bound above). The shader
// copies THIS verbatim.
inline int32_t Cross2(int32_t ax, int32_t az, int32_t bx, int32_t bz, int32_t px, int32_t pz) {
    return (bx - ax) * (pz - az) - (bz - az) * (px - ax);
}

// ----- TraceContours: the deterministic integer boundary walk (per region) ------------------------
// For each region id in ASCENDING order (1..regionCount), find its LOWEST-cellId boundary cell and
// walk the region boundary in a FIXED turn order, emitting an integer corner vertex at each boundary
// CORNER (a turn) until the walk returns to the start. The walk keeps the region cells on its RIGHT
// (a clockwise loop in (x,z) screen space where z grows downward — the Recast left-wall-follow, here
// pinned as: at each cell-edge step, prefer to turn so the boundary stays on the right). Determinism:
// the start cell is the lowest cellId in the region (so its TOP edge, z-1 neighbour, is guaranteed a
// boundary because a region cell above would have a lower cellId), the start corner + heading are
// fixed, and the per-step turn order is fixed. Output: one Contour per region (region id ascending),
// regions with no cells skipped. Pure int32, single-thread serial -> bit-exact.
//
// The walk is a corner turtle on the cell-edge graph. State: a corner (px,pz) and a heading dir in
// {0:+x, 1:+z, 2:-x, 3:-z}. We walk the boundary so the IN-region cell is on the right of the heading.
// At each corner we choose the next heading by the fixed priority: try to turn LEFT (keep hugging),
// else go STRAIGHT, else turn RIGHT, else turn BACK — the standard wall-follower. A vertex is emitted
// whenever the heading CHANGES (a corner), giving the minimal integer corner loop (degenerate single
// cell -> its 4 corners).
inline void TraceContours(const Heightfield& hf, const std::vector<uint32_t>& region,
                          uint32_t regionCount, std::vector<Contour>& contours) {
    const int w = hf.w, h = hf.h;
    contours.clear();

    // inReg(R, x, z): is cell (x,z) in-grid AND region[cell]==R? (out-of-bounds = not in region).
    auto inReg = [&](uint32_t R, int x, int z) -> bool {
        if (x < 0 || z < 0 || x >= w || z >= h) return false;
        return region[(size_t)(z * w + x)] == R;
    };

    // The 4 headings as (dx,dz): 0:+x, 1:+z, 2:-x, 3:-z. For a heading d walking an edge with the
    // in-region cell on its RIGHT, the cell on the right of the heading is at the offset rightCell[d]
    // relative to the edge's "lower-left" corner convention used below.
    const int dx[4] = {1, 0, -1, 0};
    const int dz[4] = {0, 1, 0, -1};

    for (uint32_t R = 1u; R <= regionCount; ++R) {
        // Find the lowest-cellId cell in region R (ascending z, then x).
        int sx = -1, sz = -1;
        for (int z = 0; z < h && sx < 0; ++z)
            for (int x = 0; x < w; ++x)
                if (region[(size_t)(z * w + x)] == R) { sx = x; sz = z; break; }
        if (sx < 0) continue;   // region id with no cells (shouldn't happen for dense ids) -> skip.

        // Start at the TOP-LEFT corner of the start cell, heading +x along its top edge. Because the
        // start cell is the lowest cellId in R, the cell ABOVE it (z-1) is NOT in R, so this top edge
        // IS a boundary edge with the region cell below-right -> the in-region cell sits on the RIGHT
        // of the +x heading. (Corner coords: cell (cx,cz) top-left corner is (cx,cz).)
        const int startX = sx, startZ = sz;
        int curX = startX, curZ = startZ;
        int dir = 0;   // +x
        Contour c;
        c.region = R;

        // Walk until we return to (startX,startZ) heading +x again (a full loop). A guard bound caps
        // the step count (perimeter <= 2*(w+h)*#cells, generously 8*w*h) so a logic bug can't hang.
        const int maxSteps = 8 * (w * h) + 16;
        int lastDir = -1;
        for (int step = 0; step < maxSteps; ++step) {
            // Emit a vertex at the current corner whenever the heading just changed (a real corner).
            if (dir != lastDir) {
                c.verts.push_back(ContourVertex{curX, curZ});
                lastDir = dir;
            }
            // Advance one unit along the heading to the next corner.
            curX += dx[dir];
            curZ += dz[dir];
            // Terminate when we are back at the start corner about to repeat the first edge.
            if (curX == startX && curZ == startZ) break;

            // Decide the next heading. Walking heading dir with the region on the RIGHT: examine the
            // two cells incident to the corner we just reached that determine the next boundary edge.
            // We pick the next dir by the fixed wall-follow priority: LEFT, STRAIGHT, RIGHT, BACK,
            // choosing the first whose RIGHT-side cell is in-region and LEFT-side cell is out.
            // For a heading, the cell on the right of the edge ending at corner (curX,curZ) and the
            // cell on the left are computed from the heading. We test candidate headings in order
            // {turnRight, straight, turnLeft} relative to dir — this hugs the wall clockwise (region
            // on the right) and is the FIXED, deterministic turn order.
            //   turnRight = (dir+1)&3, straight = dir, turnLeft = (dir+3)&3.
            // A heading nd is VALID iff the cell to its right is in-region and the cell to its left is
            // out-of-region (a boundary edge with the wall on the right).
            auto edgeRightLeftCells = [&](int nd, int ex, int ez, int& rcx, int& rcz, int& lcx, int& lcz) {
                // The edge starts at corner (ex,ez) heading nd. The cell on the RIGHT and LEFT of that
                // directed edge (corner-lattice convention: cell (cx,cz) spans corners (cx,cz)-(cx+1,cz+1)).
                if (nd == 0) {        // +x : right cell is below the edge (z), left is above (z-1)
                    rcx = ex;     rcz = ez;     lcx = ex;     lcz = ez - 1;
                } else if (nd == 1) { // +z : right cell is left of the edge (x-1), left is right (x)
                    rcx = ex - 1; rcz = ez;     lcx = ex;     lcz = ez;
                } else if (nd == 2) { // -x : right cell is above (z-1), left is below (z)
                    rcx = ex - 1; rcz = ez - 1; lcx = ex - 1; lcz = ez;
                } else {              // -z : right cell is right (x), left is left (x-1)
                    rcx = ex;     rcz = ez - 1; lcx = ex - 1; lcz = ez - 1;
                }
            };
            const int turnRight = (dir + 1) & 3;
            const int straight  = dir;
            const int turnLeft  = (dir + 3) & 3;
            const int cand[3] = {turnRight, straight, turnLeft};
            int nextDir = (dir + 2) & 3;   // default = turn back (only if nothing else valid)
            for (int k = 0; k < 3; ++k) {
                int rcx, rcz, lcx, lcz;
                edgeRightLeftCells(cand[k], curX, curZ, rcx, rcz, lcx, lcz);
                if (inReg(R, rcx, rcz) && !inReg(R, lcx, lcz)) { nextDir = cand[k]; break; }
            }
            dir = nextDir;
        }
        contours.push_back(std::move(c));
    }
}

// ----- PerpDistSq: integer perpendicular-distance-SQUARED of point p from segment a->b -------------
// Returns (cross^2) where cross = Cross2(a,b,p) — i.e. (2*area)^2 — and the segment's squared length
// dd = (bx-ax)^2 + (bz-az)^2 via the out param. The true perpendicular distance squared is
// cross^2 / dd; Douglas–Peucker compares cross^2 vs maxError^2 * dd (cross-multiplied, so NO division,
// NO sqrt, pure int32 for the bounded grid). Degenerate a==b (dd==0) -> returns the squared point
// distance |p-a|^2 (so a zero-length "segment" still simplifies sanely). Pure int32 (overflow bound
// above). The shader copies THIS verbatim.
inline int64_t PerpDistSqNum(const ContourVertex& a, const ContourVertex& b, const ContourVertex& p,
                             int32_t& dd) {
    dd = (b.x - a.x) * (b.x - a.x) + (b.z - a.z) * (b.z - a.z);
    if (dd == 0) {
        const int32_t ex = p.x - a.x, ez = p.z - a.z;
        return (int64_t)(ex * ex + ez * ez);   // |p-a|^2 (still int32 magnitude; widened only to match)
    }
    const int32_t cr = Cross2(a.x, a.z, b.x, b.z, p.x, p.z);
    return (int64_t)cr * (int64_t)cr;          // cross^2 (the perpendicular-dist^2 numerator)
}

// ----- SimplifyContour: integer Douglas–Peucker (perpendicular-distance-squared) ------------------
// Simplify a closed integer contour loop with an integer perpendicular-distance test vs maxError
// (compared SQUARED: keep a vertex iff its perpendicular-dist^2 from the chord exceeds maxError^2, so
// NO sqrt). A closed loop is split at its two extreme anchors first (the lowest-index vertex + the
// vertex farthest from it), then each open span is Douglas–Peucker'd with an EXPLICIT fixed-order
// stack (push [hi,lo]; pop, find the max-deviation index in (lo,hi), TIE -> LOWEST index; if its
// dist^2 > maxError^2*dd keep it + recurse both halves, else drop all interior). A minimum of 3
// vertices is kept (a triangle is the smallest polygon). Deterministic by the fixed anchor choice +
// fixed stack order + lowest-index tie-break. Pure int32. The shader copies THIS verbatim.
inline void SimplifyContour(const std::vector<ContourVertex>& in, int32_t maxError,
                            std::vector<ContourVertex>& out) {
    out.clear();
    const int n = (int)in.size();
    if (n <= 3) { out = in; return; }          // already minimal (or degenerate) -> keep as-is.
    const int64_t err2 = (int64_t)maxError * (int64_t)maxError;

    // The two fixed anchors of the closed loop: index 0 (the lowest-index vertex, deterministic) and
    // the index FARTHEST (max squared distance) from vertex 0 (tie -> lowest index). These split the
    // loop into two open chains DP simplifies independently.
    int far = 0; int64_t farD = -1;
    for (int i = 1; i < n; ++i) {
        const int32_t ex = in[(size_t)i].x - in[0].x, ez = in[(size_t)i].z - in[0].z;
        const int64_t d = (int64_t)(ex * ex + ez * ez);
        if (d > farD) { farD = d; far = i; }   // strict > -> ties keep the LOWEST index.
    }

    // keep[] marks which input vertices survive. Anchors 0 and far are always kept.
    std::vector<uint8_t> keep((size_t)n, 0u);
    keep[0] = 1u; keep[(size_t)far] = 1u;

    // DP a half-open chain [lo..hi] (endpoints fixed-kept) via an explicit stack of (lo,hi) spans.
    // The chord endpoint `hi` may be `n` (the loop-close anchor) -> it maps to vertex 0 (chordVert).
    auto chordVert = [&](int idx) -> const ContourVertex& { return (idx >= n) ? in[0] : in[(size_t)idx]; };
    auto dpChain = [&](int lo, int hi) {
        // Stack of index pairs; process in a FIXED order (LIFO, push order pinned) -> deterministic.
        std::vector<std::pair<int,int>> stack;
        stack.push_back({lo, hi});
        while (!stack.empty()) {
            const auto seg = stack.back(); stack.pop_back();
            const int a = seg.first, b = seg.second;
            if (b <= a + 1) continue;          // no interior vertices.
            // Find the interior index of MAX perpendicular-dist^2 from the chord (in[a]->chordVert(b));
            // the comparison is cross-multiplied (num > err2*dd) to stay integer. TIE -> LOWEST index.
            int bestIdx = -1; int64_t bestNum = 0; int32_t bestDd = 1;
            for (int i = a + 1; i < b; ++i) {
                int32_t dd;
                const int64_t num = PerpDistSqNum(in[(size_t)a], chordVert(b), in[(size_t)i], dd);
                // Compare num/dd > bestNum/bestDd as num*bestDd > bestNum*dd (positive denominators).
                if (bestIdx < 0 ||
                    (int64_t)num * (int64_t)bestDd > (int64_t)bestNum * (int64_t)dd) {
                    bestIdx = i; bestNum = num; bestDd = dd;
                }
            }
            if (bestIdx < 0) continue;
            // Keep it iff its perpendicular-dist^2 exceeds maxError^2: num/dd > err2  <=>  num > err2*dd.
            if (bestNum > err2 * (int64_t)bestDd) {
                keep[(size_t)bestIdx] = 1u;
                // Push the two halves; pinned push order (LOWER half last -> popped FIRST) so the
                // traversal order is fixed + deterministic.
                stack.push_back({bestIdx, b});
                stack.push_back({a, bestIdx});
            }
        }
    };
    dpChain(0, far);
    dpChain(far, n);   // the second chain wraps far..n-1..back-to-0; index n maps to vertex 0 below.

    // The far..end chain's far endpoint is `far`, its other endpoint is vertex 0 (the loop close).
    // dpChain(far, n) treated index n as vertex 0 conceptually; emit kept vertices in [0,n) order.
    for (int i = 0; i < n; ++i)
        if (keep[(size_t)i]) out.push_back(in[(size_t)i]);

    // Guarantee >= 3 vertices: if simplification collapsed too far (e.g. a near-collinear loop), fall
    // back to keeping evenly-spaced anchors from the input (deterministic) until 3 remain.
    if ((int)out.size() < 3) {
        out.clear();
        keep.assign((size_t)n, 0u);
        keep[0] = 1u;
        keep[(size_t)(n / 3)] = 1u;
        keep[(size_t)((2 * n) / 3)] = 1u;
        for (int i = 0; i < n; ++i) if (keep[(size_t)i]) out.push_back(in[(size_t)i]);
    }
}

// ----- BuildPolyMesh: ear-clip triangulation + per-edge cross-poly adjacency ----------------------
// Triangulate each simplified contour into convex polygons (triangles — convex by construction) by
// ear-clipping, then build per-triangle-edge ADJACENCY (two polys sharing an undirected edge are
// neighbours; a contour-boundary edge has no neighbour). The ear-clip repeatedly clips the
// LOWEST-index valid EAR: an ear is a convex vertex (relative to the contour's winding) whose triangle
// (prev,cur,next) contains NO OTHER contour vertex (the integer Cross2 orientation + point-in-triangle
// tests — the PointInTriXZ form). Deterministic by the fixed lowest-index ear order. Pure int32.
//
// Output `polys`: all triangles across all contours, laid out CONTOUR BY CONTOUR (the count->scan->emit
// ordering: each contour contributes (vertexCount-2) triangles; offsets are implicit in the emit
// order). poly.idx[] are indices into THAT contour's simplified vertex list; poly.nbr[] are GLOBAL
// poly ids (into `polys`). The shader copies THIS verbatim.
//
// Adjacency: after triangulating a contour, every directed edge (a->b) of every triangle is recorded;
// two triangles sharing the UNDIRECTED edge {a,b} (one as a->b, the other as b->a) are neighbours.
// Adjacency is built PER CONTOUR (a contour is a single simple polygon; triangles only share edges
// within their own contour). The shared diagonal of a quad -> the two triangles are mutual neighbours.
inline void BuildPolyMesh(const std::vector<Contour>& contours, std::vector<Poly>& polys) {
    polys.clear();

    for (const Contour& c : contours) {
        const std::vector<ContourVertex>& v = c.verts;
        const int n = (int)v.size();
        if (n < 3) continue;   // not a polygon (shouldn't happen post-SimplifyContour).

        const uint32_t firstPolyId = (uint32_t)polys.size();   // base for this contour's triangles.

        // Determine the contour's winding (signed area sign) so "convex" is winding-relative. Shoelace
        // via Cross2 sum from vertex 0.
        int64_t area2 = 0;
        for (int i = 1; i + 1 < n; ++i)
            area2 += (int64_t)Cross2(v[0].x, v[0].z, v[(size_t)i].x, v[(size_t)i].z,
                                     v[(size_t)(i + 1)].x, v[(size_t)(i + 1)].z);
        // ccw = true if the loop is counter-clockwise (area2 > 0). For a convex vertex, the turn
        // prev->cur->next has the SAME sign as the loop winding.
        const int windSign = (area2 > 0) ? 1 : -1;

        // Remaining vertex indices (into v), in order; clipped as ears are removed.
        std::vector<int> rem((size_t)n);
        for (int i = 0; i < n; ++i) rem[(size_t)i] = i;

        // pointInTri(a,b,cc, p): is corner p strictly-or-on inside triangle (a,b,cc)? Uses the NAV1
        // PointInTriXZ sign discipline (all edge-cross signs consistent with the winding).
        auto pointInTri = [&](const ContourVertex& A, const ContourVertex& B, const ContourVertex& C,
                              const ContourVertex& P) -> bool {
            const int32_t d0 = Cross2(A.x, A.z, B.x, B.z, P.x, P.z);
            const int32_t d1 = Cross2(B.x, B.z, C.x, C.z, P.x, P.z);
            const int32_t d2 = Cross2(C.x, C.z, A.x, A.z, P.x, P.z);
            const bool anyNeg = (d0 < 0) || (d1 < 0) || (d2 < 0);
            const bool anyPos = (d0 > 0) || (d1 > 0) || (d2 > 0);
            return !(anyNeg && anyPos);   // inside (or on an edge) iff signs are consistent.
        };

        // Triangles produced for THIS contour: each is the triple of contour-local vertex indices.
        std::vector<std::array<int, 3>> tris;

        // Ear-clip: while >3 remain, find the LOWEST-index valid ear among rem[] and clip it.
        int guard = 0;
        const int guardMax = n * n + 16;
        while ((int)rem.size() > 3 && guard++ < guardMax) {
            const int m = (int)rem.size();
            int earAt = -1;   // position in rem[] of the lowest-index valid ear.
            for (int i = 0; i < m; ++i) {
                const int ip = rem[(size_t)((i + m - 1) % m)];
                const int ic = rem[(size_t)i];
                const int in_ = rem[(size_t)((i + 1) % m)];
                const ContourVertex& A = v[(size_t)ip];
                const ContourVertex& B = v[(size_t)ic];
                const ContourVertex& C = v[(size_t)in_];
                // Convex iff the turn A->B->C matches the loop winding (cross sign == windSign).
                const int32_t cr = Cross2(A.x, A.z, B.x, B.z, C.x, C.z);
                if (cr == 0) continue;                       // collinear -> not a valid ear.
                if ((cr > 0 ? 1 : -1) != windSign) continue; // reflex -> not an ear.
                // No OTHER remaining vertex inside triangle (A,B,C).
                bool clean = true;
                for (int j = 0; j < m && clean; ++j) {
                    const int vj = rem[(size_t)j];
                    if (vj == ip || vj == ic || vj == in_) continue;
                    if (pointInTri(A, B, C, v[(size_t)vj])) clean = false;
                }
                if (clean) { earAt = i; break; }             // LOWEST-index valid ear.
            }
            if (earAt < 0) {
                // No ear found (degenerate / collinear contour) -> fan-triangulate the remainder from
                // rem[0] (deterministic fallback) and stop.
                break;
            }
            const int ip = rem[(size_t)((earAt + m - 1) % m)];
            const int ic = rem[(size_t)earAt];
            const int in_ = rem[(size_t)((earAt + 1) % m)];
            tris.push_back({ip, ic, in_});
            rem.erase(rem.begin() + earAt);
        }
        // Emit the final triangle (or fan the leftover if the ear search bailed).
        if ((int)rem.size() == 3) {
            tris.push_back({rem[0], rem[1], rem[2]});
        } else if ((int)rem.size() > 3) {
            for (int i = 1; i + 1 < (int)rem.size(); ++i)
                tris.push_back({rem[0], rem[(size_t)i], rem[(size_t)(i + 1)]});
        }

        // Append the triangles as Polys (neighbours filled below).
        for (const auto& t : tris) {
            Poly p{};
            p.idx[0] = (uint32_t)t[0]; p.idx[1] = (uint32_t)t[1]; p.idx[2] = (uint32_t)t[2];
            p.nbr[0] = kNoNeighbour; p.nbr[1] = kNoNeighbour; p.nbr[2] = kNoNeighbour;
            p.region = c.region; p._pad = 0u;
            polys.push_back(p);
        }

        // Per-edge adjacency WITHIN this contour: for each pair of this contour's triangles, an edge
        // e of poly P (idx[e]->idx[(e+1)%3]) matches the REVERSED edge f of poly Q -> neighbours.
        const uint32_t lastPolyId = (uint32_t)polys.size();
        for (uint32_t pi = firstPolyId; pi < lastPolyId; ++pi)
            for (int e = 0; e < 3; ++e) {
                if (polys[pi].nbr[e] != kNoNeighbour) continue;
                const uint32_t a = polys[pi].idx[e];
                const uint32_t b = polys[pi].idx[(e + 1) % 3];
                for (uint32_t qi = firstPolyId; qi < lastPolyId && polys[pi].nbr[e] == kNoNeighbour; ++qi) {
                    if (qi == pi) continue;
                    for (int f = 0; f < 3; ++f) {
                        const uint32_t qa = polys[qi].idx[f];
                        const uint32_t qb = polys[qi].idx[(f + 1) % 3];
                        if (qa == b && qb == a) {            // reversed shared edge -> neighbours.
                            polys[pi].nbr[e] = qi;
                            polys[qi].nbr[f] = pi;
                            break;
                        }
                    }
                }
            }
    }
}

// =================================================================================================
// Slice NAV5 — INTEGER A* PATHFINDING over the NAV4 poly adjacency graph (the FLAGSHIP HEADLINE).
// Additive over NAV1-NAV4 (their functions stay byte-identical). Pure integer (NO <cmath>, NO float,
// NO int64 on the bit-exact path — int32 only; see the overflow bound below). The CPU reference the
// GPU shaders/nav_astar.comp.hlsl copies VERBATIM + memcmp's bit-identical against. Runs a fully
// INTEGER A* (integer cost graph + integer-keyed deterministic frontier + lowest-id tie-break) over
// the NAV4 polys (nodes) + per-edge adjacency (edges) to produce a CORRIDOR (the poly-id sequence
// start->goal) + total integer cost — bit-exact, replayable, cross-platform-identical pathfinding
// (the thing UE5's float Detour cannot guarantee bit-for-bit across machines; pairs with FPX5 lockstep).
//
// THE COST + HEURISTIC (locked, DOCUMENTED — the determinism + optimality contract):
//   * Node anchor = the poly's INTEGER CENTROID: the truncating-integer average of its 3 contour
//     vertices' (x,z) coords (ComputePolyCentroids below). The poly idx[] are CONTOUR-LOCAL, so the
//     centroid needs the poly's owning-contour vertex base (the per-poly vertex base array, the same
//     layout the GPU shader receives).
//   * Edge cost g and heuristic h are BOTH the MANHATTAN distance |dx| + |dz| between centroids.
//     Manhattan is a true metric -> h is ADMISSIBLE *and* CONSISTENT (h(n) <= cost(n,n') + h(n') and
//     h(goal)=0), so A* is OPTIMAL; and integer Manhattan needs NO sqrt and NO int64 (the showcase
//     corner coords <= 32 -> centroid coords in [0,32] -> a single edge cost <= 64, a whole-corridor
//     cost <= 64*polys << INT32_MAX), so nav_astar.comp stays PURE INT32 / Metal-native (the NAV3/NAV4
//     int32 convention, unlike fpx's int64). (Squared-Euclidean was rejected: it is NOT a metric, so
//     it would break admissibility — Manhattan is the deterministic, optimal, int32-safe choice.)
//   * Frontier: a LINEAR MIN-SCAN over the open set (the graph is small; obviously deterministic). Pop
//     the open node with the lowest f = g + h, tie-break LOWEST poly id. came_from[] + g[]; reconstruct
//     the corridor by walking came_from goal->start then reversing. Single-thread serial -> bit-exact.
// =================================================================================================

// A sentinel "infinity" cost for the A* g/f arrays (a large int32, far inside int32 range so g+h never
// overflows). The Manhattan cost of any corridor in the bounded showcase is tiny (<< kPathInf); this is
// purely the unreached-node seed. came_from uses kNoCameFrom for "no predecessor".
static constexpr int32_t kPathInf     = 0x3FFFFFFF;   // ~1.07e9, < INT32_MAX; g+h stays in int32
static constexpr uint32_t kNoCameFrom = 0xFFFFFFFFu;  // came_from sentinel (no predecessor / start)

// ----- ComputePolyCentroids: the integer per-poly centroid anchors (the A* cost anchor) -----------
// For each poly, centroid = the truncating-integer average of its 3 contour vertices' (x,z). The poly
// idx[] are CONTOUR-LOCAL indices, so polyVertBase[p] is the base (into the flat contour-vertex array)
// of poly p's owning contour — i.e. flatVerts[(polyVertBase[p] + idx[k]) * 2 + {0,1}] is vertex k's
// (x,z). (polyVertBase mirrors the GPU layout: each region's polys share that region's gVOff[] base.)
// Pure int32. The shader copies THIS verbatim. cx/cz are sized to polys.size().
inline void ComputePolyCentroids(const std::vector<Poly>& polys,
                                 const std::vector<int32_t>& flatVerts,
                                 const std::vector<uint32_t>& polyVertBase,
                                 std::vector<int32_t>& cx, std::vector<int32_t>& cz) {
    const size_t nP = polys.size();
    cx.assign(nP, 0); cz.assign(nP, 0);
    for (size_t p = 0; p < nP; ++p) {
        const uint32_t vb = polyVertBase[p];
        int32_t sx = 0, sz = 0;
        for (int k = 0; k < 3; ++k) {
            const uint32_t vi = vb + polys[p].idx[k];
            sx += flatVerts[(size_t)vi * 2u];
            sz += flatVerts[(size_t)vi * 2u + 1u];
        }
        cx[p] = sx / 3;   // truncating integer average (deterministic).
        cz[p] = sz / 3;
    }
}

// ----- ManhattanDist: the integer cost/heuristic metric (centroid->centroid) ----------------------
// |dx| + |dz| between two integer centroids. A true metric (admissible + consistent as a heuristic).
// Pure int32 (bounded coords). The shader copies THIS verbatim.
inline int32_t ManhattanDist(int32_t ax, int32_t az, int32_t bx, int32_t bz) {
    int32_t dx = ax - bx; if (dx < 0) dx = -dx;
    int32_t dz = az - bz; if (dz < 0) dz = -dz;
    return dx + dz;
}

// ----- ConnectedComponents: the deterministic flood over poly adjacency (start/goal selection input) -
// Label each poly with its connected-component id (a deterministic flood over the NAV4 per-edge
// adjacency, ascending poly-id seed order so the labels are fixed). comp[p] in [0, nComp); returns
// nComp. (NAV4 triangulates each region independently, so components are per-region — no inter-region
// portals; NAV5 paths within the LARGEST component, the spec's documented scope.) Pure integer,
// single-thread serial -> deterministic. The shader copies THIS verbatim.
inline uint32_t ConnectedComponents(const std::vector<Poly>& polys, std::vector<uint32_t>& comp) {
    const size_t nP = polys.size();
    comp.assign(nP, kNoCameFrom);   // unassigned sentinel
    uint32_t next = 0u;
    // A fixed-capacity integer stack flood (ascending seed order, fixed neighbour order 0,1,2).
    std::vector<uint32_t> stack;
    for (uint32_t s = 0; s < (uint32_t)nP; ++s) {
        if (comp[s] != kNoCameFrom) continue;
        const uint32_t c = next++;
        comp[s] = c;
        stack.clear();
        stack.push_back(s);
        while (!stack.empty()) {
            const uint32_t p = stack.back(); stack.pop_back();
            for (int e = 0; e < 3; ++e) {
                const uint32_t q = polys[p].nbr[e];
                if (q == kNoNeighbour || q >= (uint32_t)nP) continue;
                if (comp[q] != kNoCameFrom) continue;
                comp[q] = c;
                stack.push_back(q);
            }
        }
    }
    return next;
}

// ----- SelectStartGoal: the deterministic start + goal within the LARGEST component ----------------
// Pick the LARGEST connected component (tie -> the one with the LOWEST min poly id); start = the lowest
// poly id in it; goal = the poly in it with the MAXIMUM integer (Manhattan) centroid distance from start
// (tie -> lowest id). Pure deterministic. Returns false (start=goal=0) if there are no polys. cx/cz are
// the ComputePolyCentroids output. The shader copies THIS verbatim.
inline bool SelectStartGoal(const std::vector<Poly>& polys, const std::vector<int32_t>& cx,
                            const std::vector<int32_t>& cz, uint32_t& start, uint32_t& goal) {
    start = 0u; goal = 0u;
    const size_t nP = polys.size();
    if (nP == 0) return false;
    std::vector<uint32_t> comp;
    const uint32_t nComp = ConnectedComponents(polys, comp);
    if (nComp == 0u) return false;
    // Per-component size + lowest min poly id (the first poly with that label, scanning ascending).
    std::vector<uint32_t> sizeOf((size_t)nComp, 0u);
    std::vector<uint32_t> minIdOf((size_t)nComp, kNoCameFrom);
    for (uint32_t p = 0; p < (uint32_t)nP; ++p) {
        const uint32_t c = comp[p];
        ++sizeOf[c];
        if (minIdOf[c] == kNoCameFrom) minIdOf[c] = p;   // first (lowest) poly id of this component.
    }
    // Largest component; tie -> lowest min poly id.
    uint32_t bestC = 0u;
    for (uint32_t c = 1u; c < nComp; ++c) {
        if (sizeOf[c] > sizeOf[bestC] ||
            (sizeOf[c] == sizeOf[bestC] && minIdOf[c] < minIdOf[bestC]))
            bestC = c;
    }
    start = minIdOf[bestC];
    // Goal = the poly in bestC with max Manhattan centroid distance from start (tie -> lowest id).
    goal = start;
    int32_t bestD = -1;
    for (uint32_t p = 0; p < (uint32_t)nP; ++p) {
        if (comp[p] != bestC) continue;
        const int32_t d = ManhattanDist(cx[start], cz[start], cx[p], cz[p]);
        if (d > bestD) { bestD = d; goal = p; }   // strict > -> ties keep the LOWEST id (ascending scan).
    }
    return true;
}

// ----- FindPath: the deterministic INTEGER A* (the headline) --------------------------------------
// Runs A* over the poly adjacency graph from start to goal. Nodes = polys; edges = nbr[]; edge cost +
// heuristic = ManhattanDist between centroids (cx/cz). Frontier = a linear min-scan (lowest f=g+h, tie
// -> lowest poly id). Output: corridor = the poly-id sequence start->goal (empty if unreachable; a
// single {start} if start==goal); returns the total integer g-cost at goal (0 for start==goal,
// kPathInf-sentinel-free; the corridor being empty signals "no path"). Pure int32, single-thread serial
// -> bit-exact CPU<->GPU<->both backends. The shader copies THIS body VERBATIM.
inline int32_t FindPath(const std::vector<Poly>& polys, const std::vector<int32_t>& cx,
                        const std::vector<int32_t>& cz, uint32_t start, uint32_t goal,
                        std::vector<uint32_t>& corridor) {
    corridor.clear();
    const size_t nP = polys.size();
    if (nP == 0u || start >= (uint32_t)nP || goal >= (uint32_t)nP) return 0;
    if (start == goal) { corridor.push_back(start); return 0; }

    std::vector<int32_t>  g((size_t)nP, kPathInf);       // best-known cost from start.
    std::vector<uint8_t>  open((size_t)nP, 0u);          // 1 = node is in the open set.
    std::vector<uint8_t>  closed((size_t)nP, 0u);        // 1 = node already expanded.
    std::vector<uint32_t> cameFrom((size_t)nP, kNoCameFrom);

    g[start] = 0;
    open[start] = 1u;

    while (true) {
        // Pop the open node with the lowest f = g + h, tie-break LOWEST poly id (the linear min-scan).
        uint32_t cur = kNoCameFrom;
        int32_t bestF = kPathInf;
        for (uint32_t p = 0; p < (uint32_t)nP; ++p) {
            if (open[p] == 0u) continue;
            const int32_t h = ManhattanDist(cx[p], cz[p], cx[goal], cz[goal]);
            const int32_t f = g[p] + h;
            if (cur == kNoCameFrom || f < bestF) { bestF = f; cur = p; }   // ascending scan -> tie keeps lowest id.
        }
        if (cur == kNoCameFrom) break;   // open empty -> goal unreachable.
        if (cur == goal) break;          // goal popped -> done (consistent h -> optimal on pop).

        open[cur] = 0u;
        closed[cur] = 1u;

        // Relax neighbours (fixed edge order 0,1,2).
        for (int e = 0; e < 3; ++e) {
            const uint32_t nb = polys[cur].nbr[e];
            if (nb == kNoNeighbour || nb >= (uint32_t)nP) continue;
            if (closed[nb] != 0u) continue;
            const int32_t step = ManhattanDist(cx[cur], cz[cur], cx[nb], cz[nb]);
            const int32_t tentative = g[cur] + step;
            if (tentative < g[nb]) {
                g[nb] = tentative;
                cameFrom[nb] = cur;
                open[nb] = 1u;
            }
        }
    }

    // Reconstruct: if the goal was reached (g[goal] < kPathInf), walk came_from goal->start then reverse.
    if (g[goal] >= kPathInf) return 0;   // unreachable -> empty corridor.
    std::vector<uint32_t> rev;
    uint32_t node = goal;
    // Guard the walk (<= nP steps) so a malformed came_from chain can't loop forever.
    for (size_t guard = 0; guard <= nP; ++guard) {
        rev.push_back(node);
        if (node == start) break;
        node = cameFrom[node];
        if (node == kNoCameFrom) { rev.clear(); break; }   // broken chain -> no corridor.
    }
    if (rev.empty() || rev.back() != start) { corridor.clear(); return 0; }
    for (size_t i = rev.size(); i-- > 0;) corridor.push_back(rev[i]);
    return g[goal];
}

// =================================================================================================
// Slice NAV6 — LIT 3D RENDER CAPSTONE: render-only FLOAT helpers (the money-shot, COMPLETES flagship
// #7). These are the ONLY float-crossing functions in navmesh.h and are STRICTLY render-only — they
// are NOT part of the bit-exact integer build/pathfind path (NAV1-NAV5 stay byte-identical above). The
// navmesh GEOMETRY they convert is still the bit-exact integer result (the NAV4 BuildPolyMesh polys +
// the NAV5 FindPath corridor) — the provenance — so the lit render derives from the integer navmesh;
// only the final raster/shade is float (the FPX6/MC5 float visresolve-bar, NOT the NAV1-5 integer bar).
//
// THE ONE HOST FLOAT CONVERSION: world coord = voxelCoord / (float)scale (NavVertToWorld). The showcase
// uses a 32x32 corner-coord grid (corner coords in [0,32]); it picks `scale` so the navmesh frames a
// fixed camera (the showcase documents its scale, matching the showcase's voxel->world mapping). These
// helpers depend on NOTHING but the integer navmesh types above (NO scene/math/rhi include) — they emit
// a plain POD float-vertex the showcase trivially copies into scene::Vertex (the mc.h RenderVertex
// convention), keeping navmesh.h header-only + backend-free.
// =================================================================================================

// ----- NavWorldVertex: a render-ready lit vertex (world position + per-region color) ---------------
// POD float6 (no padding holes), trivially copied into scene::Vertex by the showcase (pos + color;
// the showcase fills uv/normal/tangent — a flat upward normal for the ground-hugging navmesh sheet).
struct NavWorldVertex {
    float px, py, pz;   // world position = corner-coord / scale (the ONE host float divide), raised
    float r, g, b;      // per-region color (the watershed region id -> a distinct hue)
};

// ----- NavWorldPoint: a render-ready world-space point (the corridor polyline / markers) ----------
struct NavWorldPoint {
    float x, y, z;
};

// ----- NavVertToWorld: the single host float crossing (render-only) -------------------------------
// Convert an integer corner/voxel coord to a float world coord = coord / (float)scale. The showcase
// passes the SAME scale for x and z so the navmesh keeps its aspect; the documented voxel->world
// mapping. Pure render-only float (NOT on the bit-exact path).
inline float NavVertToWorld(int32_t coord, int32_t scale) {
    return (float)coord / (float)(scale != 0 ? scale : 1);
}

// ----- NavRegionColor: a deterministic per-region hue (render-only) -------------------------------
// A small fixed palette indexed by region id so region 1 / region 2 / ... read as distinct translucent
// sheets. Deterministic (same id -> same color every run). region 0 (none) is never rendered.
inline void NavRegionColor(uint32_t region, float& r, float& g, float& b) {
    // 6 legible, well-separated hues (cycled for >6 regions). [region][0..2] = r,g,b in [0,1].
    static const float kPalette[6][3] = {
        {0.20f, 0.65f, 0.95f},   // 1: blue
        {0.95f, 0.55f, 0.20f},   // 2: orange
        {0.35f, 0.85f, 0.40f},   // 3: green
        {0.85f, 0.30f, 0.65f},   // 4: magenta
        {0.90f, 0.85f, 0.25f},   // 5: yellow
        {0.30f, 0.80f, 0.80f},   // 6: cyan
    };
    const uint32_t i = (region == 0u) ? 0u : ((region - 1u) % 6u);
    r = kPalette[i][0]; g = kPalette[i][1]; b = kPalette[i][2];
}

// ----- PolyMeshToRenderMesh: the navmesh polys -> a lit triangle mesh (render-only float) ----------
// Turn the bit-exact integer navmesh (the NAV4 polys + their per-region contour-local vertices) into a
// float triangle mesh in world space, with PER-REGION vertex colors, RAISED slightly above the ground
// so the translucent overlay sits over the lit ground plane (not z-fighting it). For each poly, emit
// its 3 vertices (3*N total, an un-indexed soup — the showcase draws it flat) with the region's color.
//
// Inputs mirror the showcase's read-back layout (the SAME flat-vertex + per-region-base addressing the
// A* uses): polys[] (NAV4 BuildPolyMesh order, contour-by-contour), flatVerts[] (interleaved (x,z) per
// contour vertex, region-by-region), polyVertBase[p] (the vertex base of poly p's owning region — the
// SAME array ComputePolyCentroids takes). `scale` is NavVertToWorld's divisor; `raiseY` lifts the sheet
// above the ground (world units). Output: outVerts (3*polys.size() NavWorldVertex). Pure float, render-
// only; the integer navmesh feeding it is UNCHANGED. outVerts.size() == 3*polys.size().
inline void PolyMeshToRenderMesh(const std::vector<Poly>& polys,
                                 const std::vector<int32_t>& flatVerts,
                                 const std::vector<uint32_t>& polyVertBase,
                                 int32_t scale, float raiseY,
                                 std::vector<NavWorldVertex>& outVerts) {
    outVerts.clear();
    outVerts.reserve(polys.size() * 3u);
    for (size_t p = 0; p < polys.size(); ++p) {
        const uint32_t vb = polyVertBase[p];
        float r, g, b;
        NavRegionColor(polys[p].region, r, g, b);
        for (int k = 0; k < 3; ++k) {
            const uint32_t vi = vb + polys[p].idx[k];
            const int32_t cx = flatVerts[(size_t)vi * 2u];
            const int32_t cz = flatVerts[(size_t)vi * 2u + 1u];
            NavWorldVertex v;
            v.px = NavVertToWorld(cx, scale);
            v.py = raiseY;
            v.pz = NavVertToWorld(cz, scale);
            v.r = r; v.g = g; v.b = b;
            outVerts.push_back(v);
        }
    }
}

// ----- PathToWorldPolyline: the A* corridor -> a float world-space line strip (render-only) --------
// Convert the NAV5 corridor (a poly-id sequence start->goal) into a world-space polyline through each
// corridor poly's INTEGER centroid (the SAME ComputePolyCentroids cx/cz the A* used — provenance), at a
// fixed world height `lineY` (raised above the navmesh sheet so the bright line reads clearly). Output:
// outPoints (one NavWorldPoint per corridor poly — corridor.size() points; the showcase draws segments
// between consecutive points). Pure float, render-only; the corridor + centroids are the bit-exact
// integer A* result. outPoints.size() == corridor.size().
inline void PathToWorldPolyline(const std::vector<uint32_t>& corridor,
                                const std::vector<int32_t>& cx, const std::vector<int32_t>& cz,
                                int32_t scale, float lineY, std::vector<NavWorldPoint>& outPoints) {
    outPoints.clear();
    outPoints.reserve(corridor.size());
    for (uint32_t pid : corridor) {
        if (pid >= (uint32_t)cx.size()) continue;   // guard a malformed corridor id (never for a valid path)
        NavWorldPoint pt;
        pt.x = NavVertToWorld(cx[pid], scale);
        pt.y = lineY;
        pt.z = NavVertToWorld(cz[pid], scale);
        outPoints.push_back(pt);
    }
}

// =================================================================================================
// Slice NAV7 — MULTI-LAYER NAVMESH: overhangs / bridges / stacked walkable surfaces (Track-R R8 of
// docs/SUPERIORITY_ROADMAP.md — closes the flagship-#7 "one-surface-per-column" caveat). Additive
// over NAV1-NAV6 (their functions stay byte-identical above). PURE CPU v1 (no GPU pass, no shader —
// the NAV5 A* was single-thread [numthreads(1,1,1)] anyway; a GPU ML pipeline is a deferred
// refinement). Pure integer on every path (NO float, NO <cmath>; int64 only inside the FNV digest
// helper, which is NOT on the nav build path).
//
// WHAT THIS CLOSES: NAV2's FilterWalkableSpans collapses each column to ONE walkable surface
// (surfaceY[col] = the TOPMOST walkable span top), so a bridge over a tunnel loses the tunnel floor —
// the columns under the deck contribute only the deck. NAV7 keeps EVERY walkable span top as a
// SURFACE (a layer): the flat surface array + per-column CSR (count/offset — the NAV1 colCount/
// colOffset convention, NO std::map) lets an agent path UNDER the bridge and OVER it as distinct,
// never-connected surfaces of the SAME (x,z) columns.
//
// THE LOCKED ML CONTRACTS (every ordering decision pinned — the NAV3/NAV5 determinism discipline):
//   * Surface order: ascending column id (z*w+x row-major), then ascending surface y within a column
//     (merged spans are ymin-ascending + non-overlapping, so a plain walk emits ascending tops).
//     surface index = colOffsetML[col] + layer, layer 0 = the LOWEST walkable surface of the column.
//   * Connectivity: surface a (column A) ~ surface b (adjacent column B) iff |a.y - b.y| <=
//     walkableClimb (IsConnectedML — the NAV2 IsConnected max-step test per-LAYER; the walkable[]
//     gates are implied: only walkable spans become surfaces). Two surfaces of the SAME column are
//     NEVER connected (no teleport through floors).
//   * Distance field: the NAV2 two-sweep integer chamfer generalized to the surface array — the
//     sweep ORDER over surfaces (ascending index forward, descending backward) IS the NAV2 row-major
//     sweep order because the surface array is column-id-ordered; a surface relaxes against EVERY
//     connected surface in the 8 neighbour columns (W/NW/N/NE forward, E/SE/S/SW backward, the NAV2
//     weights 2/3). Border-column surfaces seed 0 (the NAV2 border rule).
//   * A*: FindPathML = the NAV5 FindPath body VERBATIM with the fixed-3 poly nbr[] generalized to a
//     CSR adjacency (nbrOffset/nbrCount/nbrList); frontier = the same linear min-scan, f = g+h,
//     tie-break LOWEST node index; neighbour relax order = the CSR list order (pinned by
//     BuildSurfaceAdjacencyML: neighbour columns in the NAV3 fixed order up/down/left/right, then
//     ascending layer within a column). Cost + heuristic = ManhattanDist on the (x,z) anchors —
//     vertical distance is NOT costed (documented; matches NAV5's planar-centroid metric).
//
// THE IDENTITY-AT-CONFIG PROOF (the append-only equivalence): on a SINGLE-LAYER field (every column
// <= 1 walkable surface) FilterWalkableSpansML emits exactly the FilterWalkableSpans walkable set
// (colCountML[col] == walkable[col], surface y == surfaceY[col]) and BuildDistanceFieldML's per-
// surface distance bit-equals BuildDistanceField's per-column distance — same seeds, same sweep
// order, same relax; FindPathML on a <=3-degree CSR graph bit-equals FindPath on the same graph
// (nav_test pins all three). NOTE (honesty): the NAV1-6 SHOWCASE field is NOT strictly single-layer —
// the box-step columns already carry a walkable ground span UNDER the box top that NAV2 collapses
// away (the very caveat NAV7 closes) — so the strict bit-identity proof runs on a truly-single-layer
// field, plus a topmost-layer correspondence proof on the full showcase field.
//
// DOCUMENTED GAPS (deferred, NOT closed here): hole-carving, polygon merge (triangles-as-polys),
// inter-region portals (the rest of Track-R R8), a GPU ML pipeline, and ML watershed/contour/
// polymesh (NAV7 paths on the surface GRID graph directly, not on an ML polymesh).
// =================================================================================================

// ----- MLSurface: one walkable surface (a layer) of a column --------------------------------------
// col = the flat column id (z*w+x), y = the surface top voxel-y (the span's ymax), layer = the
// 0-based ordinal among the column's WALKABLE surfaces, bottom-up (a cramped span gets NO layer).
// 12 bytes, no padding holes (memcmp-able — the Span discipline).
struct MLSurface {
    uint32_t col;
    int32_t  y;
    uint32_t layer;
};

// ----- FilterWalkableSpansML: EVERY walkable span top becomes a surface (the ML extraction) --------
// The NAV2 FilterWalkableSpans clearance rule applied per-SPAN without the topmost collapse: for each
// column (ascending col id), walk its merged spans BOTTOM to TOP (ascending ymin); a span's top is a
// walkable SURFACE iff the clearance to the next solid span above (gap = above.ymin - this.ymax - 1),
// or to the heightfield top ((bmaxY-1) - this.ymax) for the topmost span, is >= walkableHeight —
// the IDENTICAL integer test FilterWalkableSpans applies. Emits surfaces in the LOCKED order
// (ascending col, then ascending y) into the flat array + the per-column CSR (colOffsetML = exclusive
// prefix sum, colCountML = per-column surface count — the NAV1 colOffset/colCount convention).
// Takes the merged spans CONST (unlike FilterWalkableSpans it does NOT stamp span.area — the ML
// extraction is a pure read; the single-layer path keeps sole ownership of the area mutation).
inline void FilterWalkableSpansML(const Heightfield& hf, const WalkableConfig& cfg,
                                  const std::vector<std::vector<Span>>& mergedSpansPerColumn,
                                  std::vector<MLSurface>& surfaces,
                                  std::vector<uint32_t>& colOffsetML,
                                  std::vector<uint32_t>& colCountML) {
    const size_t nCols = (size_t)hf.columnCount();
    surfaces.clear();
    colOffsetML.assign(nCols, 0u);
    colCountML.assign(nCols, 0u);
    const int32_t fieldTop = hf.bmaxY - 1;   // inclusive top voxel-y (the NAV2 rule)

    for (size_t c = 0; c < nCols; ++c) {
        colOffsetML[c] = (uint32_t)surfaces.size();
        const std::vector<Span>& spans = mergedSpansPerColumn[c];
        uint32_t layer = 0u;
        for (size_t i = 0; i < spans.size(); ++i) {
            int32_t clearance;
            if (i + 1 < spans.size()) {
                clearance = (int32_t)spans[i + 1].ymin - (int32_t)spans[i].ymax - 1;   // gap to above
            } else {
                clearance = fieldTop - (int32_t)spans[i].ymax;                          // to field top
            }
            if (clearance >= cfg.walkableHeight) {
                surfaces.push_back(MLSurface{(uint32_t)c, (int32_t)spans[i].ymax, layer});
                ++layer;
            }
        }
        colCountML[c] = layer;
    }
}

// ----- IsConnectedML: the per-layer max-step connectivity predicate --------------------------------
// Two surfaces in ADJACENT columns connect iff |ya - yb| <= walkableClimb (the NAV2 IsConnected test
// per-layer; both are walkable by construction — only walkable spans become surfaces). Same-column
// surfaces are NEVER connected (the caller never asks; the adjacency/chamfer only cross columns).
inline bool IsConnectedML(int32_t ya, int32_t yb, int32_t climb) {
    int32_t d = ya - yb;
    if (d < 0) d = -d;
    return d <= climb;
}

// ----- SurfaceAnchorsML: the per-surface integer (x,z) anchors (the A* cost anchors) ---------------
// cx[i]/cz[i] = surface i's column coords (col % w, col / w) — the FindPathML Manhattan anchors (the
// NAV5 centroid-anchor role; vertical distance is NOT costed, documented above).
inline void SurfaceAnchorsML(const Heightfield& hf, const std::vector<MLSurface>& surfaces,
                             std::vector<int32_t>& cx, std::vector<int32_t>& cz) {
    const size_t nS = surfaces.size();
    cx.assign(nS, 0); cz.assign(nS, 0);
    for (size_t i = 0; i < nS; ++i) {
        cx[i] = (int32_t)(surfaces[i].col % (uint32_t)hf.w);
        cz[i] = (int32_t)(surfaces[i].col / (uint32_t)hf.w);
    }
}

// ----- BuildSurfaceAdjacencyML: the deterministic CSR surface graph (the A* edges) -----------------
// For each surface s (ascending index), append the connected surfaces of the 4 neighbour columns in
// the NAV3 FIXED order up (z-1), down (z+1), left (x-1), right (x+1); within a neighbour column,
// ascending layer. nbrOffset = exclusive prefix sum, nbrCount = per-surface edge count (CSR — the
// count/offset convention). Deterministic by the pinned orders; pure integer.
inline void BuildSurfaceAdjacencyML(const Heightfield& hf, const WalkableConfig& cfg,
                                    const std::vector<MLSurface>& surfaces,
                                    const std::vector<uint32_t>& colOffsetML,
                                    const std::vector<uint32_t>& colCountML,
                                    std::vector<uint32_t>& nbrOffset,
                                    std::vector<uint32_t>& nbrCount,
                                    std::vector<uint32_t>& nbrList) {
    const int w = hf.w, h = hf.h;
    const size_t nS = surfaces.size();
    nbrOffset.assign(nS, 0u);
    nbrCount.assign(nS, 0u);
    nbrList.clear();
    const int32_t climb = cfg.walkableClimb;

    for (size_t s = 0; s < nS; ++s) {
        nbrOffset[s] = (uint32_t)nbrList.size();
        const int x = (int)(surfaces[s].col % (uint32_t)w);
        const int z = (int)(surfaces[s].col / (uint32_t)w);
        const int32_t y = surfaces[s].y;
        // The NAV3 fixed neighbour-column order: up (z-1), down (z+1), left (x-1), right (x+1).
        const int nbr[4][2] = {{x, z - 1}, {x, z + 1}, {x - 1, z}, {x + 1, z}};
        for (int k = 0; k < 4; ++k) {
            const int nx = nbr[k][0], nz = nbr[k][1];
            if (nx < 0 || nz < 0 || nx >= w || nz >= h) continue;
            const size_t nc = (size_t)(nz * w + nx);
            const uint32_t base = colOffsetML[nc], cnt = colCountML[nc];
            for (uint32_t l = 0; l < cnt; ++l) {                      // ascending layer (pinned)
                if (IsConnectedML(y, surfaces[(size_t)(base + l)].y, climb))
                    nbrList.push_back(base + l);
            }
        }
        nbrCount[s] = (uint32_t)nbrList.size() - nbrOffset[s];
    }
}

// ----- BuildDistanceFieldML: the NAV2 two-sweep integer chamfer over the SURFACE graph -------------
// dist[i] = the integer chamfer distance of surface i to the nearest boundary, GEODESIC over the
// per-layer connectivity. Seed: border-column surfaces = 0, interior = kDistInf (non-walkable columns
// simply have no surfaces — the NAV2 non-walkable-cell seed has no ML node to carry). Two sweeps over
// the surface ARRAY (its order IS row-major column order): FORWARD ascending index relaxing against
// the connected surfaces of the W/NW/N/NE columns, BACKWARD descending against E/SE/S/SW (the NAV2
// weights: cardinal 2, diagonal 3). Multiple connected candidates in one neighbour column all relax
// (min wins — deterministic, order-free). Residual kDistInf clamps to 0 (the NAV2 isolated-island
// rule). On a single-layer field this is BIT-IDENTICAL to BuildDistanceField per column (same seeds,
// same visitation order, same relax — nav_test pins it). Pure integer, single-thread serial.
inline void BuildDistanceFieldML(const Heightfield& hf, const WalkableConfig& cfg,
                                 const std::vector<MLSurface>& surfaces,
                                 const std::vector<uint32_t>& colOffsetML,
                                 const std::vector<uint32_t>& colCountML,
                                 std::vector<uint32_t>& dist) {
    const int w = hf.w, h = hf.h;
    const size_t nS = surfaces.size();
    dist.assign(nS, 0u);
    const int32_t climb = cfg.walkableClimb;

    // Seed: border-column surfaces = 0, interior surfaces = kDistInf (the NAV2 border rule).
    for (size_t s = 0; s < nS; ++s) {
        const int x = (int)(surfaces[s].col % (uint32_t)w);
        const int z = (int)(surfaces[s].col / (uint32_t)w);
        const bool border = (x == 0 || z == 0 || x == w - 1 || z == h - 1);
        dist[s] = border ? 0u : kDistInf;
    }

    const uint32_t kCard = 2u, kDiag = 3u;
    // Relax surface s against every CONNECTED surface of neighbour column (nx,nz) with weight wgt.
    auto relax = [&](size_t s, int nx, int nz, uint32_t wgt) {
        if (nx < 0 || nz < 0 || nx >= w || nz >= h) return;
        const size_t nc = (size_t)(nz * w + nx);
        const uint32_t base = colOffsetML[nc], cnt = colCountML[nc];
        for (uint32_t l = 0; l < cnt; ++l) {
            const size_t t = (size_t)(base + l);
            if (!IsConnectedML(surfaces[s].y, surfaces[t].y, climb)) continue;
            const uint32_t cand = dist[t] + wgt;
            if (cand < dist[s]) dist[s] = cand;
        }
    };

    // FORWARD sweep (ascending surface index == the NAV2 TL->BR row-major order): W, NW, N, NE.
    for (size_t s = 0; s < nS; ++s) {
        if (dist[s] == 0u) continue;   // a seed (0) can only stay 0 (the NAV2 rule)
        const int x = (int)(surfaces[s].col % (uint32_t)w);
        const int z = (int)(surfaces[s].col / (uint32_t)w);
        relax(s, x - 1, z,     kCard);   // W
        relax(s, x - 1, z - 1, kDiag);   // NW
        relax(s, x,     z - 1, kCard);   // N
        relax(s, x + 1, z - 1, kDiag);   // NE
    }
    // BACKWARD sweep (descending == BR->TL): E, SE, S, SW.
    for (size_t s = nS; s-- > 0;) {
        if (dist[s] == 0u) continue;
        const int x = (int)(surfaces[s].col % (uint32_t)w);
        const int z = (int)(surfaces[s].col / (uint32_t)w);
        relax(s, x + 1, z,     kCard);   // E
        relax(s, x + 1, z + 1, kDiag);   // SE
        relax(s, x,     z + 1, kCard);   // S
        relax(s, x - 1, z + 1, kDiag);   // SW
    }
    // Residual sentinel (an isolated surface island) clamps to 0 (the NAV2 rule).
    for (size_t s = 0; s < nS; ++s)
        if (dist[s] == kDistInf) dist[s] = 0u;
}

// ----- FindPathML: the NAV5 integer A* generalized to a CSR node graph (the NAV7 headline) ---------
// The FindPath body VERBATIM with the fixed-3 poly nbr[] replaced by the CSR adjacency
// (nbrOffset/nbrCount/nbrList — for surfaces, BuildSurfaceAdjacencyML's pinned-order graph; the node
// count is nbrCount.size(), and cx/cz are the per-node integer anchors, SurfaceAnchorsML for
// surfaces). Frontier = the linear min-scan (lowest f = g + h, tie-break LOWEST node index); edge
// cost + heuristic = ManhattanDist on the anchors (admissible + consistent -> optimal, the NAV5
// contract); neighbour relax order = the CSR list order (pinned). Output: corridor = the node-id
// sequence start->goal (empty if unreachable; {start} if start==goal); returns the total integer
// g-cost at goal. On a <=3-degree graph this bit-equals FindPath (nav_test pins it). Pure int32,
// single-thread serial -> bit-exact across compilers/platforms.
inline int32_t FindPathML(const std::vector<uint32_t>& nbrOffset,
                          const std::vector<uint32_t>& nbrCount,
                          const std::vector<uint32_t>& nbrList,
                          const std::vector<int32_t>& cx, const std::vector<int32_t>& cz,
                          uint32_t start, uint32_t goal, std::vector<uint32_t>& corridor) {
    corridor.clear();
    const size_t nP = nbrCount.size();
    if (nP == 0u || start >= (uint32_t)nP || goal >= (uint32_t)nP) return 0;
    if (start == goal) { corridor.push_back(start); return 0; }

    std::vector<int32_t>  g(nP, kPathInf);       // best-known cost from start.
    std::vector<uint8_t>  open(nP, 0u);          // 1 = node is in the open set.
    std::vector<uint8_t>  closed(nP, 0u);        // 1 = node already expanded.
    std::vector<uint32_t> cameFrom(nP, kNoCameFrom);

    g[start] = 0;
    open[start] = 1u;

    while (true) {
        // Pop the open node with the lowest f = g + h, tie-break LOWEST node id (the linear min-scan).
        uint32_t cur = kNoCameFrom;
        int32_t bestF = kPathInf;
        for (uint32_t p = 0; p < (uint32_t)nP; ++p) {
            if (open[p] == 0u) continue;
            const int32_t hh = ManhattanDist(cx[p], cz[p], cx[goal], cz[goal]);
            const int32_t f = g[p] + hh;
            if (cur == kNoCameFrom || f < bestF) { bestF = f; cur = p; }   // ascending scan -> tie keeps lowest id.
        }
        if (cur == kNoCameFrom) break;   // open empty -> goal unreachable.
        if (cur == goal) break;          // goal popped -> done (consistent h -> optimal on pop).

        open[cur] = 0u;
        closed[cur] = 1u;

        // Relax neighbours in the pinned CSR list order.
        const uint32_t eBase = nbrOffset[(size_t)cur], eCnt = nbrCount[(size_t)cur];
        for (uint32_t e = 0; e < eCnt; ++e) {
            const uint32_t nb = nbrList[(size_t)(eBase + e)];
            if (nb >= (uint32_t)nP) continue;
            if (closed[nb] != 0u) continue;
            const int32_t step = ManhattanDist(cx[cur], cz[cur], cx[nb], cz[nb]);
            const int32_t tentative = g[cur] + step;
            if (tentative < g[nb]) {
                g[nb] = tentative;
                cameFrom[nb] = cur;
                open[nb] = 1u;
            }
        }
    }

    // Reconstruct: if the goal was reached, walk came_from goal->start then reverse (the NAV5 walk).
    if (g[goal] >= kPathInf) return 0;   // unreachable -> empty corridor.
    std::vector<uint32_t> rev;
    uint32_t node = goal;
    for (size_t guard = 0; guard <= nP; ++guard) {
        rev.push_back(node);
        if (node == start) break;
        node = cameFrom[node];
        if (node == kNoCameFrom) { rev.clear(); break; }   // broken chain -> no corridor.
    }
    if (rev.empty() || rev.back() != start) { corridor.clear(); return 0; }
    for (size_t i = rev.size(); i-- > 0;) corridor.push_back(rev[i]);
    return g[goal];
}

// ----- The NAV7 proof scene: a BRIDGE over a TUNNEL --------------------------------------------------
// BridgeSceneLayout carries the integer layout MakeBridgeTunnelSpans builds (inclusive ranges) so the
// tests/showcase pick start/goal columns deterministically.
struct BridgeSceneLayout {
    int32_t deckY;             // the bridge-deck surface y
    int32_t bandZ0, bandZ1;    // the bridge z band (inclusive)
    int32_t rampW0, rampW1;    // west ramp x range (inclusive), heights 1..deckY
    int32_t deckX0, deckX1;    // deck x range (inclusive) — the TWO-surface (tunnel) columns
    int32_t rampE0, rampE1;    // east ramp x range (inclusive), heights deckY..1
};

// MakeBridgeTunnelSpans(hf, out): per-column MERGED spans (the MergeColumnSpans output form —
// sorted-by-ymin, non-overlapping) for the bridge-over-tunnel scene, built directly at the span level
// (a spans-level scene builder; the NAV2/NAV3 test convention — the conservative NAV1 TriYSpan cannot
// rasterize a thin elevated slab or a climbable ramp). Designed for the 32x32 showcase grid:
//   * every column: a ground span {0,0};
//   * the bridge z band (4 columns deep, centered): a west EMBANKMENT ramp (solid {0,y}, y stepping
//     1..deckY one voxel per column — climbable at walkableClimb 1), the DECK (a thin slab
//     {deckY,deckY} ABOVE the preserved ground span — the tunnel: clearance deckY-1 under it), and a
//     mirrored east ramp (deckY..1);
//   * everything integer, deterministic, order-free.
// With walkableHeight 2 / climb 1: deck columns carry TWO surfaces (tunnel floor y=0 layer 0 + deck
// y=deckY layer 1); ramp columns ONE surface (the embankment top); all other columns ONE (ground).
inline BridgeSceneLayout MakeBridgeTunnelSpans(const Heightfield& hf,
                                               std::vector<std::vector<Span>>& mergedPerColumn) {
    BridgeSceneLayout L;
    L.deckY  = 6;
    L.bandZ0 = hf.h / 2 - 2;          L.bandZ1 = hf.h / 2 + 1;
    L.rampW0 = hf.w / 8;              L.rampW1 = hf.w / 8 + L.deckY - 1;
    L.rampE1 = hf.w - hf.w / 8 - 1;   L.rampE0 = L.rampE1 - (L.deckY - 1);
    L.deckX0 = L.rampW1 + 1;          L.deckX1 = L.rampE0 - 1;

    const size_t nCols = (size_t)hf.columnCount();
    mergedPerColumn.assign(nCols, {});
    for (int z = 0; z < hf.h; ++z)
        for (int x = 0; x < hf.w; ++x) {
            std::vector<Span>& spans = mergedPerColumn[(size_t)hf.columnId(x, z)];
            const bool inBand = (z >= L.bandZ0 && z <= L.bandZ1);
            if (inBand && x >= L.rampW0 && x <= L.rampW1) {
                const uint32_t y = (uint32_t)(x - L.rampW0 + 1);            // west ramp: 1..deckY
                spans.push_back(Span{0u, y, 1u});                           // solid embankment
            } else if (inBand && x >= L.rampE0 && x <= L.rampE1) {
                const uint32_t y = (uint32_t)(L.rampE1 - x + 1);            // east ramp: deckY..1
                spans.push_back(Span{0u, y, 1u});
            } else if (inBand && x >= L.deckX0 && x <= L.deckX1) {
                spans.push_back(Span{0u, 0u, 1u});                          // tunnel floor (ground)
                spans.push_back(Span{(uint32_t)L.deckY, (uint32_t)L.deckY, 1u});   // the deck slab
            } else {
                spans.push_back(Span{0u, 0u, 1u});                          // open ground
            }
        }
    return L;
}

// ----- Fnv1a64ML: the FNV-1a 64 digest helper (the cross-compiler/cross-backend pin) ---------------
// NOT on the nav build path (proof/stat-line plumbing only — the showcase/test digest convention).
// Chainable: pass the previous digest as the seed to fold multiple buffers.
inline uint64_t Fnv1a64ML(const void* data, size_t bytes, uint64_t h = 1469598103934665603ull) {
    const unsigned char* p = (const unsigned char*)data;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// =================================================================================================
// Slice NAV8 — OBSTACLE HOLE-CARVING + CONVEX POLYGON MERGE + INTER-REGION PORTALS + FUNNEL
// STRING-PULLING (the Track-R R8 REMAINDER of docs/SUPERIORITY_ROADMAP.md — closes the three
// flagship-#7 sub-caveats NAV7 documented as deferred: "hole-carving", "triangles-as-polys",
// "inter-region portals"). Additive over NAV1-NAV7 (their functions stay byte-identical above).
// PURE CPU (no GPU pass, no shader — the NAV7 precedent; a GPU NAV8 pipeline is a deferred
// refinement), so unlike NAV4's shader-mirrored int32 discipline, int64 is permitted here as a
// plain CPU intermediate (shoelace accumulators, the integer sqrt of the length metric, scaled
// point-in-poly products). Still NO float, NO <cmath> — pure integer on every path.
//
// WHAT THIS CLOSES (the three caveats):
//   1) HOLE-CARVING: NAV4's TraceContours walks only the OUTER boundary of a region, so an obstacle
//      fully INSIDE a region (a pillar: its columns already non-walkable per NAV2) never carves a
//      hole the contour respects — the ear-clip polygonizes right across it. NAV8 adds
//      TraceContoursWithHoles8: after the outer trace, non-region POCKETS fully enclosed by the
//      region (a 4-connected flood from the grid border over non-region cells marks OUTSIDE;
//      the un-reached non-region components are the pockets — every 4-neighbour of a pocket cell
//      outside the pocket is a region cell by construction) are each traced into a HOLE contour
//      by the same NAV4 corner turtle applied to the pocket mask. WINDING/AREA-SIGN CLASSIFICATION
//      (integer shoelace): every traced loop (outer AND hole — both walked with the in-set cells on
//      the RIGHT) has POSITIVE shoelace area2 (ContourArea2_8 > 0, pinned); a hole's area is
//      SUBTRACTED in the carve identity  2*regionCellCount == outerArea2 - SUM(holeArea2)  (the
//      pinned accounting that ties the contours to the cells).
//      THE POLYGONIZER SCHEME (documented choice): a region WITH holes is polygonized with the
//      hole's cells EXCLUDED — per-cell triangulation of the region's own cells (row-major, two
//      fixed-diagonal triangles per cell), so NO polygon covers a hole cell BY CONSTRUCTION (the
//      simplest deterministic scheme; the bridge/diagonal splice into the outer loop is a deferred
//      refinement — it complicates the ear-clip with duplicated bridge vertices for no additional
//      guarantee). The merge pass below coalesces the cells back into large convex polys. A region
//      WITHOUT holes keeps the NAV4 ear-clip (BuildPolyMesh reused verbatim on its outer contour).
//   2) POLYGON MERGE: NAV4 emits triangles, never merged. MergeConvexPolys8 greedily merges
//      adjacent polys sharing an EXACT reversed vertex-pair edge into CONVEX polygons up to
//      kMaxVertsPerPoly (= 6, the Recast convention): merge iff the spliced result stays convex
//      (integer cross-product checks at EVERY vertex, NON-STRICT — collinear vertices are legal,
//      Recast-style, so two unit quads merge into a 6-vertex rectangle) and has no duplicate
//      vertex. Deterministic order: ascending pa; for each pa repeatedly merge the LOWEST-index
//      mergeable pb (the lowest (i,j) shared edge per pair is the one candidate). Same-region only
//      (region borders survive as portal geometry). Coverage is EXACT: SUM(PolyArea2_8) is
//      invariant across the merge (pinned integer identity).
//   3) INTER-REGION PORTALS + FUNNEL: NAV4 adjacency existed only within one contour's
//      triangulation. BuildPortals8 emits a PORTAL for every pair of polys (ANY regions) whose
//      boundary edges are COLLINEAR, OPPOSITELY ORIENTED, and OVERLAP over a segment of nonzero
//      length (integer parameter projection onto the edge; the overlap endpoints are exact input
//      vertices, no division): intra-region shared edges AND cross-region border segments both
//      become portals, so the A* graph spans regions. FilterPortalsByClimb8 then drops any
//      AXIS-PARALLEL portal whose flanking cell pairs are not ALL walkable + within walkableClimb
//      (the NAV2 IsConnected test per unit lattice edge — a region border across a too-tall step
//      yields NO portal; pinned). The A* is FindPathML REUSED VERBATIM (nodes = polys via
//      PortalsToCsr8, anchors = PolyCenter8 truncating-average centers, cost/heuristic = the NAV5
//      Manhattan center-to-center metric). STRING-PULLING: StringPull8 is the classic
//      simple-stupid-funnel in pure integer (Cross2 signs) over the portal chain
//      (BuildFunnelChannel8) in DOUBLED coordinates (portal corners doubled, cell centers 2c+1 —
//      everything stays integer): the path becomes a taut polyline through portal edges instead of
//      the grid staircase.
//
// THE LOCKED NAV8 CONVENTIONS (every ordering decision pinned — the NAV3/NAV5 discipline):
//   * Poly8 winding: POSITIVE shoelace area2 (Cross2 fan) — normalized at emit; the interior is on
//     the LEFT of each directed boundary edge (the Cross2 (x,z)-as-(x,y) convention).
//   * Portal orientation: the portal segment (ax,az)->(bx,bz) runs ALONG polyA's boundary winding
//     (ascending edge parameter). Traveling FROM polyA the funnel-LEFT endpoint is (bx,bz) and
//     funnel-RIGHT is (ax,az); traveling from polyB they swap. (Derivation: for positive winding
//     the from-poly interior is left of the edge; the crossing heading is the edge's right normal;
//     the agent's left hand then points along the edge direction.)
//   * Funnel tie-breaks (the Mikko Mononen classic, signs transposed to Cross2): tighten RIGHT iff
//     Cross2(apex, right, cand) >= 0, accept iff apex==right or Cross2(apex, left, cand) < 0, else
//     the LEFT point is a corner (collinear-with-left counts as crossover); mirrored for LEFT with
//     <= 0 / > 0. Consecutive duplicate points are dropped at emit. A guard caps the restart loop
//     (never trips on a sane channel; documented bound O(n^2)).
//   * Corridor->portal lookup: the LOWEST-index portal connecting the pair (a multi-segment shared
//     boundary contributes multiple portals; the A* corridor is unaffected — the cost anchor is
//     the poly center — so the funnel just needs ONE deterministic choice).
//   * Start/goal poly resolution: FindContainingPoly8 = the LOWEST-index poly containing the
//     (doubled) point, boundary-inclusive.
//   * Length metric: PolylineLenQ8_8 = SUM(floor(256 * sqrt(dx^2 + dz^2))) per segment (integer
//     binary-search sqrt, Q24.8) over DOUBLED coords — 512 units per voxel edge; the SAME metric
//     measures the funnel polyline and the grid-staircase polyline (comparable, deterministic).
//     NOTE (honesty): the poly A* corridor optimizes the NAV5 Manhattan-center metric, so the
//     funnel is taut WITHIN that corridor (global-Euclidean-optimal corridor choice is a deferred
//     refinement; on the proof scenes the corridor is the intuitive one and the funnel is strictly
//     shorter than the grid staircase — pinned).
//   * The cell-grid comparison path: BuildCellGridCsr8 (nodes = ALL columns, edges = the NAV3
//     fixed-order 4-neighbour walkable+IsConnected pairs, anchors = (x,z)) + FindPathML — the
//     "grid staircase" the funnel is measured against, via cell-center (2c+1) doubled polylines.
//
// DOCUMENTED GAPS (deferred, NOT closed here): a GPU NAV8 pipeline; bridge-spliced hole ear-clips
// (holed regions pay a cell-triangulation poly-count premium the merge only partly recovers);
// partial-edge merging (merge needs an EXACTLY-shared vertex-pair edge); portal splitting at
// climb-connectivity changes (a portal is kept only if its WHOLE width is traversable);
// multi-layer (NAV7) portals/funnel — NAV8 funnels over the single-layer poly mesh.
// =================================================================================================

// ----- kMaxVertsPerPoly: the Recast merged-poly vertex cap ---------------------------------------
static constexpr uint32_t kMaxVertsPerPoly = 6u;

// ----- Poly8: a convex polygon in GLOBAL corner-lattice coords (the NAV8 mesh unit) ---------------
// Unlike NAV4's Poly (contour-local indices), Poly8 carries its vertex coords directly — merge and
// portal overlap tests need coordinate comparisons across contours/regions, and the grids are tiny.
// nverts in [3, kMaxVertsPerPoly]; verts CCW by the positive-shoelace convention; region = the
// owning region id. 56 bytes, all 4-byte members, no padding holes (memcmp/digest-able).
struct Poly8 {
    uint32_t nverts;
    int32_t  vx[kMaxVertsPerPoly];
    int32_t  vz[kMaxVertsPerPoly];
    uint32_t region;
};

// ----- Portal8: a shared-boundary segment between two polys (the A* edge + funnel gate) -----------
// polyA < polyB (build order). (ax,az)->(bx,bz) runs along polyA's boundary winding (the LOCKED
// orientation convention above). 24 bytes, no padding (memcmp/digest-able).
struct Portal8 {
    uint32_t polyA, polyB;
    int32_t  ax, az;
    int32_t  bx, bz;
};

// ----- NavPoint8: an integer 2D point (funnel path vertices, DOUBLED coords by convention) --------
struct NavPoint8 {
    int32_t x, z;
};

// ----- RegionContours8: one region's outer loop + its hole loops ----------------------------------
struct RegionContours8 {
    uint32_t region = 0u;
    Contour outer;
    std::vector<Contour> holes;
};

// ----- ContourArea2_8: the integer shoelace (2x signed area) of a closed loop ---------------------
// Fan sum of Cross2 from vertex 0 (the BuildPolyMesh winding probe, exposed). Positive for every
// loop the NAV4/NAV8 turtle traces (in-set cells on the right — pinned by nav_test).
inline int64_t ContourArea2_8(const std::vector<ContourVertex>& v) {
    int64_t a2 = 0;
    const int n = (int)v.size();
    for (int i = 1; i + 1 < n; ++i)
        a2 += (int64_t)Cross2(v[0].x, v[0].z, v[(size_t)i].x, v[(size_t)i].z,
                              v[(size_t)(i + 1)].x, v[(size_t)(i + 1)].z);
    return a2;
}

// ----- TraceMaskBoundary8: the NAV4 corner turtle generalized to an arbitrary cell mask -----------
// VERBATIM the TraceContours walk (same start rule, same fixed turn priority, same emit-on-turn)
// with the region-id test replaced by a mask lookup — so it traces a REGION's outer loop (mask =
// the region cells) and a POCKET's boundary (mask = the pocket cells) with identical, pinned
// determinism. (startX,startZ) MUST be the lowest-cellId cell of the mask's connected component
// (the caller guarantees it — the cell above is then outside the mask, so the top edge is a
// boundary edge with the in-set cell on the right of the +x heading).
inline void TraceMaskBoundary8(int w, int h, const std::vector<uint8_t>& mask,
                               int startX, int startZ, std::vector<ContourVertex>& verts) {
    verts.clear();
    auto inSet = [&](int x, int z) -> bool {
        if (x < 0 || z < 0 || x >= w || z >= h) return false;
        return mask[(size_t)(z * w + x)] != 0u;
    };
    const int dx[4] = {1, 0, -1, 0};
    const int dz[4] = {0, 1, 0, -1};
    int curX = startX, curZ = startZ;
    int dir = 0;   // +x (the NAV4 start heading)
    int lastDir = -1;
    const int maxSteps = 8 * (w * h) + 16;
    for (int step = 0; step < maxSteps; ++step) {
        if (dir != lastDir) {
            verts.push_back(ContourVertex{curX, curZ});
            lastDir = dir;
        }
        curX += dx[dir];
        curZ += dz[dir];
        if (curX == startX && curZ == startZ) break;
        auto edgeRightLeftCells = [&](int nd, int ex, int ez, int& rcx, int& rcz, int& lcx, int& lcz) {
            if (nd == 0) {
                rcx = ex;     rcz = ez;     lcx = ex;     lcz = ez - 1;
            } else if (nd == 1) {
                rcx = ex - 1; rcz = ez;     lcx = ex;     lcz = ez;
            } else if (nd == 2) {
                rcx = ex - 1; rcz = ez - 1; lcx = ex - 1; lcz = ez;
            } else {
                rcx = ex;     rcz = ez - 1; lcx = ex - 1; lcz = ez - 1;
            }
        };
        const int cand[3] = {(dir + 1) & 3, dir, (dir + 3) & 3};
        int nextDir = (dir + 2) & 3;
        for (int k = 0; k < 3; ++k) {
            int rcx, rcz, lcx, lcz;
            edgeRightLeftCells(cand[k], curX, curZ, rcx, rcz, lcx, lcz);
            if (inSet(rcx, rcz) && !inSet(lcx, lcz)) { nextDir = cand[k]; break; }
        }
        dir = nextDir;
    }
}

// ----- TraceContoursWithHoles8: outer + interior-hole contours per region (the carve) -------------
// For each region id ascending: (1) the outer loop = TraceMaskBoundary8 on the region mask (BIT-
// IDENTICAL to NAV4's TraceContours output — pinned); (2) OUTSIDE = a 4-connected stack flood over
// non-region cells seeded from every non-region border cell (ascending seed order; the result is a
// set, order-free); (3) each remaining non-region component (a POCKET, discovered in ascending
// lowest-cellId order) is traced into a HOLE contour by the same turtle on the pocket mask. A
// pocket that touches the border is by construction OUTSIDE (a bay, not a hole). A different
// region's cells enclosed in the pocket are part of the pocket (non-R) — the hole covers them and
// the enclosed region gets its own outer loop (their coincident boundaries later yield portals).
// Pure integer, single-thread serial -> deterministic.
inline void TraceContoursWithHoles8(const Heightfield& hf, const std::vector<uint32_t>& region,
                                    uint32_t regionCount, std::vector<RegionContours8>& out) {
    const int w = hf.w, h = hf.h;
    const size_t nCols = (size_t)(w * h);
    out.clear();

    for (uint32_t R = 1u; R <= regionCount; ++R) {
        // The region mask + its lowest-cellId cell (row-major first hit).
        std::vector<uint8_t> inR(nCols, 0u);
        int sx = -1, sz = -1;
        for (int z = 0; z < h; ++z)
            for (int x = 0; x < w; ++x) {
                const size_t c = (size_t)(z * w + x);
                if (region[c] != R) continue;
                inR[c] = 1u;
                if (sx < 0) { sx = x; sz = z; }
            }
        if (sx < 0) continue;   // region id with no cells -> skip (the NAV4 rule).

        RegionContours8 rc;
        rc.region = R;
        rc.outer.region = R;
        TraceMaskBoundary8(w, h, inR, sx, sz, rc.outer.verts);

        // OUTSIDE flood: 4-connected over non-R cells, seeded from every non-R border cell.
        std::vector<uint8_t> outside(nCols, 0u);
        std::vector<int> stack;
        for (int z = 0; z < h; ++z)
            for (int x = 0; x < w; ++x) {
                if (x != 0 && z != 0 && x != w - 1 && z != h - 1) continue;   // border cells only
                const size_t c = (size_t)(z * w + x);
                if (inR[c] != 0u || outside[c] != 0u) continue;
                outside[c] = 1u;
                stack.push_back((int)c);
                while (!stack.empty()) {
                    const int cc = stack.back(); stack.pop_back();
                    const int cx = cc % w, cz = cc / w;
                    const int nb[4][2] = {{cx, cz - 1}, {cx, cz + 1}, {cx - 1, cz}, {cx + 1, cz}};
                    for (int k = 0; k < 4; ++k) {
                        const int nx = nb[k][0], nz = nb[k][1];
                        if (nx < 0 || nz < 0 || nx >= w || nz >= h) continue;
                        const size_t nc = (size_t)(nz * w + nx);
                        if (inR[nc] != 0u || outside[nc] != 0u) continue;
                        outside[nc] = 1u;
                        stack.push_back((int)nc);
                    }
                }
            }

        // POCKETS: non-R, not outside -> 4-connected components in ascending lowest-cellId order.
        std::vector<uint8_t> pocketSeen(nCols, 0u);
        for (int z = 0; z < h; ++z)
            for (int x = 0; x < w; ++x) {
                const size_t c = (size_t)(z * w + x);
                if (inR[c] != 0u || outside[c] != 0u || pocketSeen[c] != 0u) continue;
                // Flood THIS pocket into its own mask (start (x,z) is its lowest cellId).
                std::vector<uint8_t> pmask(nCols, 0u);
                pmask[c] = 1u;
                pocketSeen[c] = 1u;
                stack.clear();
                stack.push_back((int)c);
                while (!stack.empty()) {
                    const int cc = stack.back(); stack.pop_back();
                    const int cx = cc % w, cz = cc / w;
                    const int nb[4][2] = {{cx, cz - 1}, {cx, cz + 1}, {cx - 1, cz}, {cx + 1, cz}};
                    for (int k = 0; k < 4; ++k) {
                        const int nx = nb[k][0], nz = nb[k][1];
                        if (nx < 0 || nz < 0 || nx >= w || nz >= h) continue;
                        const size_t nc = (size_t)(nz * w + nx);
                        if (inR[nc] != 0u || outside[nc] != 0u || pocketSeen[nc] != 0u) continue;
                        pmask[nc] = 1u;
                        pocketSeen[nc] = 1u;
                        stack.push_back((int)nc);
                    }
                }
                Contour hole;
                hole.region = R;
                TraceMaskBoundary8(w, h, pmask, x, z, hole.verts);
                rc.holes.push_back(std::move(hole));
            }

        out.push_back(std::move(rc));
    }
}

// ----- PolyArea2_8 / IsConvexPoly8 / PolyCenter8: the Poly8 integer primitives --------------------
// PolyArea2_8 = the shoelace fan (2x signed area, int64 accumulator); positive for the pinned
// winding. IsConvexPoly8 = every consecutive turn Cross2 >= 0 (NON-STRICT: collinear verts legal,
// the Recast merged-poly convention) AND area2 > 0. PolyCenter8 = the truncating-integer vertex
// average (the NAV5 centroid convention generalized to nverts; grid coords are >= 0 here so the
// truncation is a plain floor).
inline int64_t PolyArea2_8(const Poly8& p) {
    int64_t a2 = 0;
    for (uint32_t i = 1u; i + 1u < p.nverts; ++i)
        a2 += (int64_t)Cross2(p.vx[0], p.vz[0], p.vx[i], p.vz[i], p.vx[i + 1u], p.vz[i + 1u]);
    return a2;
}
inline bool IsConvexPoly8(const Poly8& p) {
    if (p.nverts < 3u || p.nverts > kMaxVertsPerPoly) return false;
    for (uint32_t i = 0; i < p.nverts; ++i) {
        const uint32_t j = (i + 1u) % p.nverts;
        const uint32_t k = (i + 2u) % p.nverts;
        if (Cross2(p.vx[i], p.vz[i], p.vx[j], p.vz[j], p.vx[k], p.vz[k]) < 0) return false;
    }
    return PolyArea2_8(p) > 0;
}
inline void PolyCenter8(const Poly8& p, int32_t& cx, int32_t& cz) {
    int32_t sx = 0, sz = 0;
    for (uint32_t i = 0; i < p.nverts; ++i) { sx += p.vx[i]; sz += p.vz[i]; }
    cx = sx / (int32_t)p.nverts;
    cz = sz / (int32_t)p.nverts;
}

// ----- PointInConvexPoly8Scaled: boundary-inclusive point-in-convex-poly at a coord scale ---------
// True iff (px,pz) — given in SCALE-multiplied coords — is inside or on the poly whose verts are
// multiplied by the same scale (all edge Cross2 >= 0 for the positive winding; int64 products).
// scale=2 tests a DOUBLED point (cell centers 2c+1); the funnel containment sampler uses 2*K.
inline bool PointInConvexPoly8Scaled(const Poly8& p, int32_t px, int32_t pz, int32_t scale) {
    for (uint32_t i = 0; i < p.nverts; ++i) {
        const uint32_t j = (i + 1u) % p.nverts;
        const int64_t ax = (int64_t)p.vx[i] * scale, az = (int64_t)p.vz[i] * scale;
        const int64_t bx = (int64_t)p.vx[j] * scale, bz = (int64_t)p.vz[j] * scale;
        const int64_t cr = (bx - ax) * ((int64_t)pz - az) - (bz - az) * ((int64_t)px - ax);
        if (cr < 0) return false;
    }
    return true;
}

// ----- PolygonizeRegion8: one region's contours -> Poly8 triangles (holes respected) --------------
// No holes -> the NAV4 ear-clip REUSED VERBATIM (BuildPolyMesh on the outer contour; idx -> global
// coords; winding normalized positive, zero-area slivers skipped — never produced on the proof
// scenes). With holes -> the documented hole-cells-excluded scheme: row-major per-cell fixed-
// diagonal triangulation of the region's own cells (two positive-winding unit triangles per cell:
// [(x,z),(x+1,z),(x+1,z+1)] + [(x,z),(x+1,z+1),(x,z+1)]) — NO polygon covers a hole cell BY
// CONSTRUCTION. APPENDS to outPolys (the BuildPolyMesh8 driver concatenates regions).
inline void PolygonizeRegion8(const Heightfield& hf, const RegionContours8& rc,
                              const std::vector<uint32_t>& region, std::vector<Poly8>& outPolys) {
    if (rc.holes.empty()) {
        if (rc.outer.verts.size() < 3u) return;
        std::vector<Contour> one;
        one.push_back(rc.outer);
        std::vector<Poly> tris;
        BuildPolyMesh(one, tris);
        for (const Poly& t : tris) {
            Poly8 p{};
            p.nverts = 3u;
            p.region = rc.region;
            for (int k = 0; k < 3; ++k) {
                p.vx[k] = rc.outer.verts[(size_t)t.idx[k]].x;
                p.vz[k] = rc.outer.verts[(size_t)t.idx[k]].z;
            }
            const int64_t a2 = PolyArea2_8(p);
            if (a2 == 0) continue;   // degenerate sliver -> skip (documented)
            if (a2 < 0) {            // normalize to the pinned positive winding
                std::swap(p.vx[1], p.vx[2]);
                std::swap(p.vz[1], p.vz[2]);
            }
            outPolys.push_back(p);
        }
    } else {
        for (int z = 0; z < hf.h; ++z)
            for (int x = 0; x < hf.w; ++x) {
                if (region[(size_t)hf.columnId(x, z)] != rc.region) continue;
                Poly8 a{};
                a.nverts = 3u; a.region = rc.region;
                a.vx[0] = x;     a.vz[0] = z;
                a.vx[1] = x + 1; a.vz[1] = z;
                a.vx[2] = x + 1; a.vz[2] = z + 1;
                Poly8 b{};
                b.nverts = 3u; b.region = rc.region;
                b.vx[0] = x;     b.vz[0] = z;
                b.vx[1] = x + 1; b.vz[1] = z + 1;
                b.vx[2] = x;     b.vz[2] = z + 1;
                outPolys.push_back(a);
                outPolys.push_back(b);
            }
    }
}

// ----- BuildPolyMesh8: all regions' contours -> the concatenated Poly8 triangle mesh --------------
inline void BuildPolyMesh8(const Heightfield& hf, const std::vector<RegionContours8>& rcs,
                           const std::vector<uint32_t>& region, std::vector<Poly8>& polys) {
    polys.clear();
    for (const RegionContours8& rc : rcs) PolygonizeRegion8(hf, rc, region, polys);
}

// ----- MergeConvexPolys8: the greedy Recast-style convex merge (in place) -------------------------
// Ascending pa; for each pa repeatedly merge the LOWEST-index mergeable pb > pa (same region,
// sharing an EXACT reversed vertex-pair edge — the lowest (i,j) edge of the pair is the one
// candidate) whenever the splice stays within kMaxVertsPerPoly, has no duplicate vertex, and
// IsConvexPoly8 holds (non-strict turns). The splice: polyA's verts starting after the shared edge
// (na-1 of them) + polyB's likewise (nb-1) = na+nb-2. Deterministic by the pinned scan order.
// Returns the merge count (polys.size() shrinks by exactly this). Coverage-exact: the area2 sum is
// invariant (each merge unions two disjoint-interior polys along their shared edge).
inline uint32_t MergeConvexPolys8(std::vector<Poly8>& polys) {
    uint32_t merges = 0u;
    for (size_t pa = 0; pa < polys.size(); ++pa) {
        bool again = true;
        while (again) {
            again = false;
            for (size_t pb = pa + 1; pb < polys.size(); ++pb) {
                if (polys[pb].region != polys[pa].region) continue;
                const Poly8 A = polys[pa];   // copies: the erase below would invalidate references
                const Poly8 B = polys[pb];
                int ei = -1, ej = -1;
                for (uint32_t i = 0; i < A.nverts && ei < 0; ++i) {
                    const uint32_t i1 = (i + 1u) % A.nverts;
                    for (uint32_t j = 0; j < B.nverts; ++j) {
                        const uint32_t j1 = (j + 1u) % B.nverts;
                        if (A.vx[i] == B.vx[j1] && A.vz[i] == B.vz[j1] &&
                            A.vx[i1] == B.vx[j] && A.vz[i1] == B.vz[j]) {
                            ei = (int)i;
                            ej = (int)j;
                            break;
                        }
                    }
                }
                if (ei < 0) continue;
                if (A.nverts + B.nverts - 2u > kMaxVertsPerPoly) continue;
                Poly8 M{};
                M.nverts = A.nverts + B.nverts - 2u;
                M.region = A.region;
                uint32_t k = 0u;
                for (uint32_t t = 0; t + 1u < A.nverts; ++t, ++k) {
                    const uint32_t idx = ((uint32_t)ei + 1u + t) % A.nverts;
                    M.vx[k] = A.vx[idx];
                    M.vz[k] = A.vz[idx];
                }
                for (uint32_t t = 0; t + 1u < B.nverts; ++t, ++k) {
                    const uint32_t idx = ((uint32_t)ej + 1u + t) % B.nverts;
                    M.vx[k] = B.vx[idx];
                    M.vz[k] = B.vz[idx];
                }
                bool dup = false;
                for (uint32_t a = 0; a < M.nverts && !dup; ++a)
                    for (uint32_t b2 = a + 1u; b2 < M.nverts; ++b2)
                        if (M.vx[a] == M.vx[b2] && M.vz[a] == M.vz[b2]) { dup = true; break; }
                if (dup) continue;
                if (!IsConvexPoly8(M)) continue;
                polys[pa] = M;
                polys.erase(polys.begin() + (ptrdiff_t)pb);
                ++merges;
                again = true;
                break;   // rescan pb from pa+1 (pa changed)
            }
        }
    }
    return merges;
}

// ----- BuildPortals8: shared-boundary segments between poly pairs (the A* edges) ------------------
// For every pa < pb (ascending) and every edge pair (i ascending, j ascending): a portal exists iff
// the two directed edges are COLLINEAR (both endpoints of B's edge on A's edge line, Cross2 == 0),
// OPPOSITELY ORIENTED (dot < 0 — same-winding polys walk a shared boundary in opposite directions),
// and OVERLAP over a segment of NONZERO length. Overlap via integer parameter projection t = dot(q -
// P0, d) onto A's edge d (t in [0, dd]): lo = max(0, t(Q1)), hi = min(dd, t(Q0)); hi <= lo -> none
// (point-touch rejected). The overlap endpoints are EXACT input vertices (lo is P0 or Q1, hi is P1
// or Q0 — no division). Emitted oriented along polyA's edge (the LOCKED portal convention). Multiple
// disjoint shared segments -> multiple portals (ascending edge order). Sound for the cell-lattice
// meshes NAV8 builds: distinct walkable areas are >= 1 cell apart, so coincident opposite boundaries
// are exactly the true adjacencies. Pure int32 (coords bounded by the grid).
inline void BuildPortals8(const std::vector<Poly8>& polys, std::vector<Portal8>& portals) {
    portals.clear();
    const uint32_t n = (uint32_t)polys.size();
    for (uint32_t pa = 0; pa < n; ++pa)
        for (uint32_t pb = pa + 1u; pb < n; ++pb) {
            const Poly8& A = polys[pa];
            const Poly8& B = polys[pb];
            for (uint32_t i = 0; i < A.nverts; ++i) {
                const uint32_t i1 = (i + 1u) % A.nverts;
                const int32_t p0x = A.vx[i], p0z = A.vz[i];
                const int32_t p1x = A.vx[i1], p1z = A.vz[i1];
                const int32_t dx = p1x - p0x, dz = p1z - p0z;
                const int32_t dd = dx * dx + dz * dz;
                if (dd == 0) continue;
                for (uint32_t j = 0; j < B.nverts; ++j) {
                    const uint32_t j1 = (j + 1u) % B.nverts;
                    const int32_t q0x = B.vx[j], q0z = B.vz[j];
                    const int32_t q1x = B.vx[j1], q1z = B.vz[j1];
                    if (Cross2(p0x, p0z, p1x, p1z, q0x, q0z) != 0) continue;
                    if (Cross2(p0x, p0z, p1x, p1z, q1x, q1z) != 0) continue;
                    if (dx * (q1x - q0x) + dz * (q1z - q0z) >= 0) continue;   // must be OPPOSITE
                    const int32_t t0 = (q0x - p0x) * dx + (q0z - p0z) * dz;   // Q0's param (higher)
                    const int32_t t1 = (q1x - p0x) * dx + (q1z - p0z) * dz;   // Q1's param (lower)
                    const int32_t lo = t1 > 0 ? t1 : 0;
                    const int32_t hi = t0 < dd ? t0 : dd;
                    if (hi <= lo) continue;                                    // no (or point) overlap
                    Portal8 pt;
                    pt.polyA = pa;
                    pt.polyB = pb;
                    if (lo == 0) { pt.ax = p0x; pt.az = p0z; } else { pt.ax = q1x; pt.az = q1z; }
                    if (hi == dd) { pt.bx = p1x; pt.bz = p1z; } else { pt.bx = q0x; pt.bz = q0z; }
                    portals.push_back(pt);
                }
            }
        }
}

// ----- FilterPortalsByClimb8: drop portals whose width is not fully traversable -------------------
// An AXIS-PARALLEL portal separates cell pairs along its lattice line: vertical at x=k flanks
// (k-1,z)|(k,z) per unit edge, horizontal at z=k flanks (x,k-1)|(x,k). The portal is KEPT iff EVERY
// unit edge's flanking pair is walkable AND within walkableClimb (the NAV2 IsConnected test) — the
// ALL-connected convention (splitting a portal at connectivity changes is a deferred refinement).
// Non-axis-parallel portals (intra-region ear-clip diagonals) are kept as-is (intra-region
// traversability is given by the NAV3 region build). Order-preserving erase -> deterministic.
inline void FilterPortalsByClimb8(const Heightfield& hf, const WalkableConfig& cfg,
                                  const std::vector<uint32_t>& walkable,
                                  const std::vector<int32_t>& surfaceY,
                                  std::vector<Portal8>& portals) {
    const int w = hf.w, h = hf.h;
    auto cellPairOk = [&](int ax, int az, int bx, int bz) -> bool {
        if (ax < 0 || az < 0 || ax >= w || az >= h) return false;
        if (bx < 0 || bz < 0 || bx >= w || bz >= h) return false;
        const size_t ca = (size_t)(az * w + ax), cb = (size_t)(bz * w + bx);
        return IsConnected(walkable[ca], surfaceY[ca], walkable[cb], surfaceY[cb], cfg.walkableClimb);
    };
    std::vector<Portal8> kept;
    kept.reserve(portals.size());
    for (const Portal8& pt : portals) {
        const int32_t dx = pt.bx - pt.ax, dz = pt.bz - pt.az;
        bool ok = true;
        if (dx != 0 && dz != 0) {
            ok = true;   // diagonal (intra-region) -> kept, documented
        } else if (dx == 0) {
            const int32_t z0 = dz > 0 ? pt.az : pt.bz;
            const int32_t z1 = dz > 0 ? pt.bz : pt.az;
            for (int32_t z = z0; z < z1 && ok; ++z)
                if (!cellPairOk(pt.ax - 1, (int)z, pt.ax, (int)z)) ok = false;
        } else {
            const int32_t x0 = dx > 0 ? pt.ax : pt.bx;
            const int32_t x1 = dx > 0 ? pt.bx : pt.ax;
            for (int32_t x = x0; x < x1 && ok; ++x)
                if (!cellPairOk((int)x, pt.az - 1, (int)x, pt.az)) ok = false;
        }
        if (ok) kept.push_back(pt);
    }
    portals.swap(kept);
}

// ----- PortalsToCsr8: the portal graph as the FindPathML CSR (the A* reuse seam) ------------------
// For each poly ascending, append the far poly of every portal that touches it, in ascending portal
// order (duplicates from multi-segment boundaries are harmless: the A* relax is idempotent).
inline void PortalsToCsr8(uint32_t nPolys, const std::vector<Portal8>& portals,
                          std::vector<uint32_t>& nbrOffset, std::vector<uint32_t>& nbrCount,
                          std::vector<uint32_t>& nbrList) {
    nbrOffset.assign((size_t)nPolys, 0u);
    nbrCount.assign((size_t)nPolys, 0u);
    nbrList.clear();
    for (uint32_t p = 0; p < nPolys; ++p) {
        nbrOffset[p] = (uint32_t)nbrList.size();
        for (size_t k = 0; k < portals.size(); ++k) {
            if (portals[k].polyA == p) nbrList.push_back(portals[k].polyB);
            else if (portals[k].polyB == p) nbrList.push_back(portals[k].polyA);
        }
        nbrCount[p] = (uint32_t)nbrList.size() - nbrOffset[p];
    }
}

// ----- PolyCenters8: the per-poly integer anchors (the FindPathML cx/cz input) --------------------
inline void PolyCenters8(const std::vector<Poly8>& polys,
                         std::vector<int32_t>& cx, std::vector<int32_t>& cz) {
    const size_t nP = polys.size();
    cx.assign(nP, 0);
    cz.assign(nP, 0);
    for (size_t p = 0; p < nP; ++p) PolyCenter8(polys[p], cx[p], cz[p]);
}

// ----- FindContainingPoly8: the lowest-index poly containing a DOUBLED point ----------------------
inline uint32_t FindContainingPoly8(const std::vector<Poly8>& polys, int32_t pxD, int32_t pzD) {
    for (uint32_t p = 0; p < (uint32_t)polys.size(); ++p)
        if (PointInConvexPoly8Scaled(polys[p], pxD, pzD, 2)) return p;
    return kNoCameFrom;
}

// ----- FindPortalBetween8: the LOWEST-index portal connecting an unordered poly pair --------------
inline uint32_t FindPortalBetween8(const std::vector<Portal8>& portals, uint32_t a, uint32_t b) {
    for (uint32_t k = 0; k < (uint32_t)portals.size(); ++k) {
        const bool fwd = portals[k].polyA == a && portals[k].polyB == b;
        const bool rev = portals[k].polyA == b && portals[k].polyB == a;
        if (fwd || rev) return k;
    }
    return kNoCameFrom;
}

// ----- BuildFunnelChannel8: the corridor -> the funnel's (left,right) portal chain ----------------
// DOUBLED coords throughout (portal lattice corners x2; start/goal are already-doubled points, e.g.
// cell centers 2c+1). Entry 0 = the start point (left==right), entries 1..m = the corridor's portal
// crossings oriented by the LOCKED travel convention (from polyA: left=(bx,bz), right=(ax,az); from
// polyB: swapped), the last entry = the goal point (left==right). Returns false if a corridor step
// has no portal (a malformed corridor — never for a FindPathML corridor over PortalsToCsr8).
inline bool BuildFunnelChannel8(const std::vector<uint32_t>& corridor,
                                const std::vector<Portal8>& portals,
                                NavPoint8 startD, NavPoint8 goalD,
                                std::vector<NavPoint8>& left, std::vector<NavPoint8>& right) {
    left.clear();
    right.clear();
    left.push_back(startD);
    right.push_back(startD);
    for (size_t i = 0; i + 1 < corridor.size(); ++i) {
        const uint32_t k = FindPortalBetween8(portals, corridor[i], corridor[i + 1]);
        if (k == kNoCameFrom) return false;
        const Portal8& pt = portals[(size_t)k];
        if (corridor[i] == pt.polyA) {
            left.push_back(NavPoint8{2 * pt.bx, 2 * pt.bz});
            right.push_back(NavPoint8{2 * pt.ax, 2 * pt.az});
        } else {
            left.push_back(NavPoint8{2 * pt.ax, 2 * pt.az});
            right.push_back(NavPoint8{2 * pt.bx, 2 * pt.bz});
        }
    }
    left.push_back(goalD);
    right.push_back(goalD);
    return true;
}

// ----- StringPull8: the integer simple-stupid funnel (the taut path) ------------------------------
// The classic funnel over the (left,right) channel from BuildFunnelChannel8 (entry 0 = start, last
// = goal). Pure integer Cross2 signs with the LOCKED tie-breaks documented in the section header.
// Consecutive duplicate points are dropped. Output: the taut polyline start..goal in DOUBLED coords.
inline void StringPull8(const std::vector<NavPoint8>& left, const std::vector<NavPoint8>& right,
                        std::vector<NavPoint8>& outPath) {
    outPath.clear();
    const size_t n = left.size();
    if (n == 0 || right.size() != n) return;
    auto eq = [](NavPoint8 a, NavPoint8 b) { return a.x == b.x && a.z == b.z; };
    auto push = [&](NavPoint8 p) {
        if (outPath.empty() || !eq(outPath.back(), p)) outPath.push_back(p);
    };
    NavPoint8 apex = left[0], pl = left[0], pr = right[0];
    size_t apexI = 0, leftI = 0, rightI = 0;
    push(apex);
    long long guard = 0;
    const long long guardMax = 4ll * (long long)n * (long long)n + 16ll;
    for (size_t i = 1; i < n; ++i) {
        if (guard++ > guardMax) break;   // never trips on a sane channel (documented O(n^2) bound)
        const NavPoint8 l = left[i], r = right[i];
        // Tighten RIGHT: cand must not spread right of the apex->right ray.
        if (Cross2(apex.x, apex.z, pr.x, pr.z, r.x, r.z) >= 0) {
            if (eq(apex, pr) || Cross2(apex.x, apex.z, pl.x, pl.z, r.x, r.z) < 0) {
                pr = r;
                rightI = i;
            } else {
                // Right crossed over left -> the LEFT point is a path corner; restart past it.
                push(pl);
                apex = pl;
                apexI = leftI;
                pl = apex;
                pr = apex;
                rightI = apexI;
                i = apexI;
                continue;
            }
        }
        // Tighten LEFT (mirror).
        if (Cross2(apex.x, apex.z, pl.x, pl.z, l.x, l.z) <= 0) {
            if (eq(apex, pl) || Cross2(apex.x, apex.z, pr.x, pr.z, l.x, l.z) > 0) {
                pl = l;
                leftI = i;
            } else {
                push(pr);
                apex = pr;
                apexI = rightI;
                pl = apex;
                pr = apex;
                leftI = apexI;
                i = apexI;
                continue;
            }
        }
    }
    push(left[n - 1]);   // the goal (left.back() == right.back() by construction)
}

// ----- IntSqrtFloor8: deterministic floor integer square root (binary search, no <cmath>) ---------
inline int64_t IntSqrtFloor8(int64_t v) {
    if (v <= 0) return 0;
    int64_t lo = 1, hi = 3037000499ll;   // floor(sqrt(INT64_MAX))
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo + 1) / 2;
        if (mid <= v / mid) lo = mid;    // mid*mid <= v, overflow-free
        else hi = mid - 1;
    }
    return lo;
}

// ----- PolylineLenQ8_8: the Q24.8 integer Euclidean length of a polyline --------------------------
// SUM(floor(256 * sqrt(dx^2 + dz^2))) per segment. Over DOUBLED coords one voxel edge = 512 units.
// The SAME metric measures the funnel polyline and the grid-staircase polyline (comparable,
// deterministic; per-segment floors make the sum a well-defined pinned integer).
inline int64_t PolylineLenQ8_8(const std::vector<NavPoint8>& pts) {
    int64_t len = 0;
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        const int64_t dx = (int64_t)pts[i + 1].x - (int64_t)pts[i].x;
        const int64_t dz = (int64_t)pts[i + 1].z - (int64_t)pts[i].z;
        len += IntSqrtFloor8((dx * dx + dz * dz) * 65536ll);
    }
    return len;
}

// ----- BuildCellGridCsr8: the walkable cell grid as a FindPathML CSR (the staircase baseline) -----
// Nodes = ALL columns (non-walkable ones get zero edges, keeping node id == column id); edges = the
// NAV3 fixed-order 4-neighbours (up, down, left, right) where both cells are walkable AND within
// walkableClimb (IsConnected). Anchors = the column (x,z). FindPathML over this graph is the
// bit-exact grid A* whose staircase polyline (cell centers, doubled 2c+1) the funnel is measured
// against.
inline void BuildCellGridCsr8(const Heightfield& hf, const WalkableConfig& cfg,
                              const std::vector<uint32_t>& walkable,
                              const std::vector<int32_t>& surfaceY,
                              std::vector<uint32_t>& nbrOffset, std::vector<uint32_t>& nbrCount,
                              std::vector<uint32_t>& nbrList,
                              std::vector<int32_t>& cx, std::vector<int32_t>& cz) {
    const int w = hf.w, h = hf.h;
    const size_t nCols = (size_t)(w * h);
    nbrOffset.assign(nCols, 0u);
    nbrCount.assign(nCols, 0u);
    nbrList.clear();
    cx.assign(nCols, 0);
    cz.assign(nCols, 0);
    for (int z = 0; z < h; ++z)
        for (int x = 0; x < w; ++x) {
            const size_t c = (size_t)(z * w + x);
            cx[c] = x;
            cz[c] = z;
            nbrOffset[c] = (uint32_t)nbrList.size();
            if (walkable[c] != 0u) {
                const int nb[4][2] = {{x, z - 1}, {x, z + 1}, {x - 1, z}, {x + 1, z}};
                for (int k = 0; k < 4; ++k) {
                    const int nx = nb[k][0], nz = nb[k][1];
                    if (nx < 0 || nz < 0 || nx >= w || nz >= h) continue;
                    const size_t nc = (size_t)(nz * w + nx);
                    if (IsConnected(walkable[c], surfaceY[c], walkable[nc], surfaceY[nc],
                                    cfg.walkableClimb))
                        nbrList.push_back((uint32_t)nc);
                }
            }
            nbrCount[c] = (uint32_t)nbrList.size() - nbrOffset[c];
        }
}

// ----- The NAV8 proof scene (a): a room with a central PILLAR --------------------------------------
// Ground {0,0} everywhere; the pillar columns are SOLID TO THE FIELD TOP ({0, bmaxY-1}: zero
// clearance above the span top -> NAV2's FilterWalkableSpans marks them non-walkable — the premise
// "an obstacle span row already makes columns non-walkable" honored at the span level). Pillar rect
// = [w*3/8, w*5/8-1] x [h*3/8, h*5/8-1] (32x32 -> [12,19]^2, an 8x8 pillar). NOTE (documented): the
// proof pipeline assigns the room a SINGLE region (region[c] = walkable[c]) instead of running the
// NAV3 watershed — a symmetric annulus legitimately watersheds into MULTIPLE basins, none of which
// individually encloses the pillar; hole-carving consumes ANY region partition, and the
// multi-region path is proven by the L two-region scene below.
struct PillarRoomLayout8 {
    int32_t px0, px1, pz0, pz1;   // pillar cell rect (inclusive)
};
inline PillarRoomLayout8 MakePillarRoomSpans8(const Heightfield& hf,
                                              std::vector<std::vector<Span>>& mergedPerColumn) {
    PillarRoomLayout8 L;
    L.px0 = (hf.w * 3) / 8;
    L.px1 = (hf.w * 5) / 8 - 1;
    L.pz0 = (hf.h * 3) / 8;
    L.pz1 = (hf.h * 5) / 8 - 1;
    const uint32_t lid = (uint32_t)(hf.bmaxY - 1);
    mergedPerColumn.assign((size_t)hf.columnCount(), {});
    for (int z = 0; z < hf.h; ++z)
        for (int x = 0; x < hf.w; ++x) {
            std::vector<Span>& s = mergedPerColumn[(size_t)hf.columnId(x, z)];
            const bool pillar = (x >= L.px0 && x <= L.px1 && z >= L.pz0 && z <= L.pz1);
            if (pillar) s.push_back(Span{0u, lid, 1u});   // solid to the top: zero clearance
            else        s.push_back(Span{0u, 0u, 1u});    // open ground
        }
    return L;
}

// ----- The NAV8 proof scene (b): an L-shaped TWO-REGION field --------------------------------------
// Two hand-assigned regions forming an L (designed for a 24x24 grid): region 1 = the vertical arm,
// region 2 = the horizontal foot, adjacent along the lattice seam x = bx0 over the foot's z rows —
// the inter-region portal seam. All L cells walkable at surfaceY 0. start/goal = the pinned
// diagonal test cells (deep in each arm; the taut funnel cuts the inner corner (bx0, bz0)).
struct LTwoRegionLayout8 {
    int32_t ax0, ax1, az0, az1;   // region-1 vertical arm cell rect (inclusive)
    int32_t bx0, bx1, bz0, bz1;   // region-2 horizontal foot cell rect (inclusive)
    int32_t startX, startZ;       // suggested start cell (region 1)
    int32_t goalX, goalZ;         // suggested goal cell (region 2)
};
inline LTwoRegionLayout8 MakeLTwoRegionGrid8(const Heightfield& hf, std::vector<uint32_t>& region,
                                             std::vector<uint32_t>& walkable,
                                             std::vector<int32_t>& surfaceY) {
    LTwoRegionLayout8 L;
    L.ax0 = 2;
    L.ax1 = (hf.w * 5) / 12 - 1;
    L.az0 = 2;
    L.az1 = hf.h - 3;
    L.bx0 = L.ax1 + 1;
    L.bx1 = hf.w - 3;
    L.bz0 = (hf.h * 7) / 12;
    L.bz1 = hf.h - 3;
    L.startX = L.ax0 + 3;
    L.startZ = L.az0 + 2;
    L.goalX = L.bx1 - 2;
    L.goalZ = L.bz0 + 3;
    const size_t nCols = (size_t)hf.columnCount();
    region.assign(nCols, 0u);
    walkable.assign(nCols, 0u);
    surfaceY.assign(nCols, 0);
    for (int z = 0; z < hf.h; ++z)
        for (int x = 0; x < hf.w; ++x) {
            const size_t c = (size_t)hf.columnId(x, z);
            uint32_t r = 0u;
            if (x >= L.ax0 && x <= L.ax1 && z >= L.az0 && z <= L.az1) r = 1u;
            else if (x >= L.bx0 && x <= L.bx1 && z >= L.bz0 && z <= L.bz1) r = 2u;
            if (r != 0u) {
                region[c] = r;
                walkable[c] = 1u;
                surfaceY[c] = 0;
            }
        }
    return L;
}

}  // namespace hf::nav
