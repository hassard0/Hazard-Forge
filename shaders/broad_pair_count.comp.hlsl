// Slice BP2 — Deterministic Integer Broadphase: THE CANDIDATE-PAIR GENERATOR per-body COUNT compute pass (the
// 1st of the BP2 pair count->scan->emit). ONE thread per body i (i < bodyCount). The thread scans body i's
// 3x3x3 = 27-cell stencil over the BP1 grid + CSR cell table (broad_cell_*), in the FIXED (dz,dy,dx) order,
// clamped to the bounded grid, and counts every body j in those cells with j > i (the canonical de-dup) AND
// fpx::AabbOverlap(BodyAabb(i), BodyAabb(j)) (the six-compare predicate copied VERBATIM from broad.h/fpx.h).
// perBodyCount[i] = that count. The count is per-body DISJOINT (race-free) -> ONE thread per body, no atomics.
//
// WHY BIT-IDENTICAL to the CPU broad.h::CountBroadphasePairs (the make-or-break): same FloorDiv cell math,
// same FIXED stencil order, same j>i guard, same six-compare AabbOverlap. PURE INT32 (integer divide + a
// sign-correct adjust + integer add/sub + compares; NO fxmul, NO int64, NO sqrt) -> MSL-generates NATIVELY on
// Metal (a TRUE GPU pass on BOTH backends, the broad_cell_*/grain_neighbor_* precedent).
//
// Buffers (storage, bound at compute bindings 0..3; on Metal these land at buffer(0..3)):
//   b0 gBodies      : the Q16.16 FxBody mirror array (pos.xyz + radius, 16-byte std430 ints), READ.
//   b1 cellStart    : the BP1 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellBodies   : the BP1 body indices grouped by cell (bodyCount), READ.
//   b3 perBodyCount : one uint per body (the per-body pair count), WRITE.
//   b4 gParams      : the BodyGridParams (cellSize, cellMin.xyz, gridDim.xyz, bodyCount, cellCount,enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_BROAD_THREADS 64

// std430 FxBody mirror (matches broad_cell's BroadBody): pos.xyz + radius = 4 x int32 (16 bytes). BP2 reads
// pos AND radius (the AABB half-extent).
struct BroadBody {
    int  px, py, pz;   // Q16.16 body position
    int  radius;       // Q16.16 body broadphase half-extent
};

// std430 grid params (the C++ BodyGridParams upload mirror).
//   grid : x=cellSize (Q16.16 cell edge), y=cellMinX, z=cellMinY, w=cellMinZ
//   dim  : x=gridDimX, y=gridDimY, z=gridDimZ, w=bodyCount
//   cfg  : x=cellCount, y=enabled, z=unused, w=unused
struct BodyGridParams {
    int4 grid;
    int4 dim;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<BroadBody>      gBodies      : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>           cellStart    : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>           cellBodies   : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>           perBodyCount : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<BodyGridParams> gParams      : register(u4);

// FloorDiv(n, d): deterministic FLOOR division for positive divisor d, ANY-sign n — VERBATIM
// engine/sim/fpx.h::FloorDiv. Pure int32 (no int64).
int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

// FlatCellId(cx, cy, cz): an ABSOLUTE cell coord -> flat cell id in the bounded dense grid — VERBATIM the
// CPU FlatBodyCellId (offset by cellMin, CellId into gridDim).
uint FlatCellId(int cx, int cy, int cz) {
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gx = gParams[0].dim.x, gy = gParams[0].dim.y;
    int lx = cx - cellMinX, ly = cy - cellMinY, lz = cz - cellMinZ;
    return (uint)((lz * gy + ly) * gx + lx);
}

// AabbOverlap of bodies a,b: SIX integer compares over the per-axis [pos-radius, pos+radius] AABBs —
// VERBATIM engine/sim/fpx.h::AabbOverlap over BodyAabb. NO products, NO int64.
bool BodiesOverlap(BroadBody a, BroadBody b) {
    int aloX = a.px - a.radius, ahiX = a.px + a.radius;
    int aloY = a.py - a.radius, ahiY = a.py + a.radius;
    int aloZ = a.pz - a.radius, ahiZ = a.pz + a.radius;
    int bloX = b.px - b.radius, bhiX = b.px + b.radius;
    int bloY = b.py - b.radius, bhiY = b.py + b.radius;
    int bloZ = b.pz - b.radius, bhiZ = b.pz + b.radius;
    return aloX <= bhiX && bloX <= ahiX &&
           aloY <= bhiY && bloY <= ahiY &&
           aloZ <= bhiZ && bloZ <= ahiZ;
}

[numthreads(HF_BROAD_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int bodyCount = gParams[0].dim.w;
    int enabled   = gParams[0].cfg.y;
    int cellSize  = gParams[0].grid.x;
    int cellMinX  = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gdimX = gParams[0].dim.x, gdimY = gParams[0].dim.y, gdimZ = gParams[0].dim.z;

    uint i = gid.x;
    if ((int)i >= bodyCount) return;
    if (enabled == 0) { perBodyCount[i] = 0u; return; }

    BroadBody bi = gBodies[i];
    int cix = FloorDiv(bi.px, cellSize);
    int ciy = FloorDiv(bi.py, cellSize);
    int ciz = FloorDiv(bi.pz, cellSize);

    uint c = 0u;
    // FIXED (dz,dy,dx) stencil order — VERBATIM the CPU triple loop.
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int ncx = cix + dx, ncy = ciy + dy, ncz = ciz + dz;
        if (ncx < cellMinX || ncx >= cellMinX + gdimX) continue;
        if (ncy < cellMinY || ncy >= cellMinY + gdimY) continue;
        if (ncz < cellMinZ || ncz >= cellMinZ + gdimZ) continue;
        uint cell = FlatCellId(ncx, ncy, ncz);
        for (uint s = cellStart[cell]; s < cellStart[cell + 1u]; ++s) {
            uint j = cellBodies[s];
            if (j <= i) continue;                       // canonical de-dup: emit (i,j) once
            if (BodiesOverlap(bi, gBodies[j])) ++c;
        }
    }
    perBodyCount[i] = c;
}
