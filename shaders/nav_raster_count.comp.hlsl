// Slice NAV1 — Deterministic GPU Navmesh BEACHHEAD: the per-COLUMN span-RASTER COUNT compute pass (the
// FIRST slice of FLAGSHIP #7; the MC2 mc_count / FPX2 fpx_pair_count analog on voxel columns). ONE
// thread per column (col < columnCount). The thread decomposes col -> (cx,cz), scans every input
// triangle, and writes colCount[col] = #triangles whose XZ projection covers the column-center (cx,cz)
// AND whose voxel-column AABB includes (cx,cz) — the RAW (pre-merge) covering-span count. NO atomics
// (each thread writes its own colCount[col]); the host prefix-sums for the per-column write base.
// enabled=0 -> write 0.
//
// WHY BIT-IDENTICAL to the CPU navmesh.h::ColumnSpanCount (the make-or-break): the whole predicate is
// PURE INT32 — TriColumnAabb is integer min/max + clamp, and PointInTriXZ is three integer edge
// functions (a 2D cross product of int32 deltas) tested for a consistent sign. So this MSL-generates
// NATIVELY on Metal (no int64, no --msl-version 20200), unlike fpx_integrate's int64 fxmul. A
// divergence vs the header is exactly what the host's GPU==CPU memcmp (colCount) catches.
//
// INT32 OVERFLOW BOUND (why the shader stays int32 while the CPU reference uses int64): the edge
// function is (dx)*(dz') - (dz)*(dx') where each factor is a difference of voxel coords. For NAV1's
// showcase + tests the voxel grid is small (|coord| <= a few thousand), so |factor| <= ~few-thousand
// and the product <= ~1e7, far inside int32's 2.1e9 range. The CPU reference (navmesh.h) widens to
// int64 to be overflow-safe in general; for these bounded coords the int32 products are bit-identical.
// (If a future NAV slice needs a larger grid, the bound is re-checked or the products widened both
// sides.)
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gTris     : the int32 NavTri array (3 verts x int3, std430), READ.
//   b1 gColCount : one uint per column (the raw covering-span count), WRITE.
//   b2 gParams   : { w, h, triCount, enabled }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations
// (same as mc_count.comp / fpx_pair_count.comp), not backend CODE symbols.

#define HF_NAV_THREADS 64

// std430 NavTri mirror (engine/nav/navmesh.h::NavTri): 3 verts x (int3 + int pad) = 12 x int32 ... but
// to stay std430-tight + memcmp-able we pack 9 int32 (3 x int3). HLSL StructuredBuffer of a 9-int
// struct -> 36 bytes, no padding holes (matches the host FpxBody-style packing).
struct NavTri {
    int v0x, v0y, v0z;
    int v1x, v1y, v1z;
    int v2x, v2y, v2z;
};

// Params (std430). Mirrors the C++ upload struct.
//   cfg : x=w (x columns), y=h (z columns), z=triCount, w=enabled
struct NavParams {
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<NavTri>    gTris     : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>      gColCount : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<NavParams> gParams   : register(u2);

// PointInTriXZ(px,pz, tri): three integer edge-function signs, consistent orientation -> inside.
// VERBATIM navmesh.h::PointInTriXZ (int32 products; see the overflow bound above).
bool PointInTriXZ(int px, int pz, NavTri t) {
    int e0 = (t.v1x - t.v0x) * (pz - t.v0z) - (t.v1z - t.v0z) * (px - t.v0x);
    int e1 = (t.v2x - t.v1x) * (pz - t.v1z) - (t.v2z - t.v1z) * (px - t.v1x);
    int e2 = (t.v0x - t.v2x) * (pz - t.v2z) - (t.v0z - t.v2z) * (px - t.v2x);
    bool allNonNeg = (e0 >= 0) && (e1 >= 0) && (e2 >= 0);
    bool allNonPos = (e0 <= 0) && (e1 <= 0) && (e2 <= 0);
    return allNonNeg || allNonPos;
}

// CoversColumn(cx,cz, tri, w,h): the column-AABB clamp test (TriColumnAabb) AND the cover test
// (PointInTriXZ) — VERBATIM navmesh.h::ColumnSpanCount's per-triangle body. Pure integer.
bool CoversColumn(int cx, int cz, NavTri t, int w, int h) {
    int minX = t.v0x, maxX = t.v0x, minZ = t.v0z, maxZ = t.v0z;
    if (t.v1x < minX) minX = t.v1x; if (t.v1x > maxX) maxX = t.v1x;
    if (t.v2x < minX) minX = t.v2x; if (t.v2x > maxX) maxX = t.v2x;
    if (t.v1z < minZ) minZ = t.v1z; if (t.v1z > maxZ) maxZ = t.v1z;
    if (t.v2z < minZ) minZ = t.v2z; if (t.v2z > maxZ) maxZ = t.v2z;
    if (minX < 0) minX = 0;
    if (minZ < 0) minZ = 0;
    if (maxX > w - 1) maxX = w - 1;
    if (maxZ > h - 1) maxZ = h - 1;
    if (minX > maxX || minZ > maxZ) return false;        // tri misses the grid
    if (cx < minX || cx > maxX || cz < minZ || cz > maxZ) return false;
    return PointInTriXZ(cx, cz, t);
}

[numthreads(HF_NAV_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int w        = gParams[0].cfg.x;
    int h        = gParams[0].cfg.y;
    int triCount = gParams[0].cfg.z;
    int enabled  = gParams[0].cfg.w;

    uint col = gid.x;
    int columnCount = w * h;
    if ((int)col >= columnCount) return;

    // Disabled -> write 0, NO span counted (colCount stays the cleared all-zero upload; the
    // byte-identical no-op).
    if (enabled == 0) { gColCount[col] = 0u; return; }

    // Decompose col -> (cx,cz). col = cz*w + cx (VERBATIM navmesh.h::columnId).
    int cx = (int)col % w;
    int cz = (int)col / w;

    uint c = 0u;
    for (int ti = 0; ti < triCount; ++ti)
        if (CoversColumn(cx, cz, gTris[(uint)ti], w, h)) c += 1u;

    gColCount[col] = c;   // order-independent per-column integer write (NO atomics)
}
