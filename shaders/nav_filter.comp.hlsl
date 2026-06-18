// Slice NAV2 — Deterministic GPU Navmesh: the per-COLUMN WALKABLE FILTER compute pass (the SECOND
// slice of FLAGSHIP #7). ONE thread per column (col < columnCount). The thread walks that column's
// MERGED spans (the NAV1 MergeColumnSpans output, uploaded as a flat buffer + per-column offset/count)
// from TOP to BOTTOM, marks each span's TOP walkable iff the clearance ABOVE it (the gap to the next
// solid span above, or to the heightfield top) >= walkableHeight, and writes the per-column WALKABLE
// MASK gWalkable[col] (1 iff the column has >=1 walkable span) + gSurfaceY[col] (the topmost walkable
// span's top-y). NO atomics (each thread writes its own column outputs). enabled=0 -> walkable 0.
//
// WHY BIT-IDENTICAL to the CPU navmesh.h::FilterWalkableSpans (the make-or-break): the whole predicate
// is PURE INT32 integer compares (clearance = aboveYmin - thisYmax - 1, or fieldTop - thisYmax;
// walkable iff clearance >= walkableHeight) — the fpx.h::AabbOverlap discipline, NO products, NO int64.
// So this MSL-generates NATIVELY on Metal. A divergence vs the header is exactly what the host's
// GPU==CPU memcmp (walkable[] + surfaceY[]) catches. The 4-neighbour max-step connectivity test is
// applied by nav_distance.comp (it needs walkable+surfaceY of BOTH cells), not here.
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gSpans     : the flat MERGED Span array (3 x uint per span: ymin,ymax,area), READ/WRITE (area).
//   b1 gColOffset : one uint per column (the column's first merged span index in gSpans), READ.
//   b2 gColCount  : one uint per column (the column's merged span count), READ.
//   b3 gWalkable  : one uint per column (1=has a walkable surface, 0=not), WRITE.
//   b4 gSurfaceY  : one int per column (the topmost walkable surface y; 0 if none), WRITE.
//   b5 gParams    : { w, h, walkableHeight, enabled } + { fieldTop, _, _, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations
// (same as nav_raster_count.comp), not backend CODE symbols.

#define HF_NAV_THREADS 64

// std430 Span mirror (engine/nav/navmesh.h::Span): 3 x uint32 (12 bytes), memcmp-able.
struct Span {
    uint ymin;
    uint ymax;
    uint area;
};

// Params (std430). Mirrors the C++ upload struct.
//   cfg : x=w, y=h, z=walkableHeight, w=enabled
//   ext : x=fieldTop (the inclusive top voxel-y of the heightfield, bmaxY-1)
struct NavFilterParams {
    int4 cfg;
    int4 ext;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<Span>            gSpans     : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            gColOffset : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>            gColCount  : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>            gWalkable  : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<int>             gSurfaceY  : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<NavFilterParams> gParams    : register(u5);

[numthreads(HF_NAV_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int w             = gParams[0].cfg.x;
    int h             = gParams[0].cfg.y;
    int walkableHeight = gParams[0].cfg.z;
    int enabled       = gParams[0].cfg.w;
    int fieldTop      = gParams[0].ext.x;

    uint col = gid.x;
    int columnCount = w * h;
    if ((int)col >= columnCount) return;

    // Disabled -> walkable 0 / surfaceY 0 (the byte-identical no-op).
    if (enabled == 0) { gWalkable[col] = 0u; gSurfaceY[col] = 0; return; }

    uint base  = gColOffset[col];
    uint count = gColCount[col];

    int  topWalkableY = 0;
    uint anyWalkable  = 0u;
    // Merged spans are sorted ASCENDING by ymin. Walk TOP -> BOTTOM (highest index = highest span).
    // The span ABOVE span i is span i+1 (higher ymin). VERBATIM navmesh.h::FilterWalkableSpans.
    for (int i = (int)count - 1; i >= 0; --i) {
        uint idx = base + (uint)i;
        int thisYmax = (int)gSpans[idx].ymax;
        int clearance;
        if (i + 1 < (int)count) {
            int aboveYmin = (int)gSpans[base + (uint)(i + 1)].ymin;
            clearance = aboveYmin - thisYmax - 1;
        } else {
            clearance = fieldTop - thisYmax;   // topmost span -> clearance to the heightfield top
        }
        uint isWalkable = (clearance >= walkableHeight) ? 1u : 0u;
        gSpans[idx].area = isWalkable;
        if (isWalkable != 0u && anyWalkable == 0u) {
            topWalkableY = thisYmax;   // first walkable seen top->bottom = the topmost walkable surface
            anyWalkable = 1u;
        }
    }

    gWalkable[col] = anyWalkable;
    gSurfaceY[col] = (anyWalkable != 0u) ? topWalkableY : 0;
}
