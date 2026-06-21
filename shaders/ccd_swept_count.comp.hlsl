// Slice CD2 — Deterministic Integer CCD: THE SWEPT-AABB BROADPHASE per-body COUNT compute pass (the 1st of the
// CD2 swept pair count->scan->emit; the broad_pair_count twin keyed on EXPLICIT swept AABBs). ONE thread per
// body i (i < bodyCount). The thread scans body i's 3x3x3 = 27-cell stencil over the BP1 grid + CSR cell table
// (built host-side over the swept-AABB CENTRES), in the FIXED (dz,dy,dx) order, clamped to the bounded grid, and
// counts every body j in those cells with j > i (the canonical de-dup) AND AabbOverlap over the SWEPT AABBs (the
// six-compare predicate over the explicit lo/hi boxes). perBodyCount[i] = that count. Per-body DISJOINT
// (race-free) -> ONE thread per body, no atomics.
//
// WHY BIT-IDENTICAL to the CPU ccd.h::CountSweptPairs (the make-or-break): same FloorDiv cell math over the AABB
// centre ((lo+hi)/2), same FIXED stencil order, same j>i guard, same six-compare AabbOverlap over lo/hi. The
// swept-AABB PRECOMPUTE (integrate end pose + 2x BuildHullAabb + union — the int64 part) is done HOST-side; the
// shader reads the int32 lo/hi AABBs. PURE INT32 (integer divide + a sign-correct adjust + add/sub + compares;
// NO fxmul, NO int64, NO sqrt) -> MSL-generates NATIVELY on Metal (the broad_pair_* tier, a TRUE GPU pass).
//
// Buffers (storage, bound at compute bindings 0..3; on Metal these land at buffer(0..3)):
//   b0 gAabbs       : the swept-AABB array (lo.xyz + hi.xyz, 24-byte std430 ints), READ.
//   b1 cellStart    : the BP1 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellBodies   : the BP1 body indices grouped by cell (bodyCount), READ.
//   b3 perBodyCount : one uint per body (the per-body pair count), WRITE.
//   b4 gParams      : the BodyGridParams (cellSize, cellMin.xyz, gridDim.xyz, bodyCount, cellCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_CCD_SWEPT_THREADS 64

// std430 swept-AABB mirror: lo.xyz + hi.xyz = 6 x int32 (24 bytes). The CD2 ccd.h::fpx::FxAabb upload.
struct SweptAabb {
    int lox, loy, loz;   // Q16.16 swept-AABB min corner
    int hix, hiy, hiz;   // Q16.16 swept-AABB max corner
};

// std430 grid params (the C++ BodyGridParams upload mirror, == the broad_pair shaders).
//   grid : x=cellSize (Q16.16 cell edge), y=cellMinX, z=cellMinY, w=cellMinZ
//   dim  : x=gridDimX, y=gridDimY, z=gridDimZ, w=bodyCount
//   cfg  : x=cellCount, y=enabled, z=unused, w=unused
struct BodyGridParams {
    int4 grid;
    int4 dim;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<SweptAabb>       gAabbs       : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            cellStart    : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>            cellBodies   : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>            perBodyCount : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<BodyGridParams>  gParams      : register(u4);

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

// AabbOverlap of swept AABBs a,b: SIX integer compares over the explicit lo/hi boxes — VERBATIM
// engine/sim/fpx.h::AabbOverlap. NO products, NO int64.
bool AabbsOverlap(SweptAabb a, SweptAabb b) {
    return a.lox <= b.hix && b.lox <= a.hix &&
           a.loy <= b.hiy && b.loy <= a.hiy &&
           a.loz <= b.hiz && b.loz <= a.hiz;
}

[numthreads(HF_CCD_SWEPT_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int bodyCount = gParams[0].dim.w;
    int enabled   = gParams[0].cfg.y;
    int cellSize  = gParams[0].grid.x;
    int cellMinX  = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gdimX = gParams[0].dim.x, gdimY = gParams[0].dim.y, gdimZ = gParams[0].dim.z;

    uint i = gid.x;
    if ((int)i >= bodyCount) return;
    if (enabled == 0) { perBodyCount[i] = 0u; return; }

    SweptAabb bi = gAabbs[i];
    // Cell of body i = the AABB CENTRE ((lo+hi)/2 per axis), FloorDiv'd — matches ccd.h::AabbCenter + BodyCellOf.
    int cenX = (bi.lox + bi.hix) / 2, cenY = (bi.loy + bi.hiy) / 2, cenZ = (bi.loz + bi.hiz) / 2;
    int cix = FloorDiv(cenX, cellSize);
    int ciy = FloorDiv(cenY, cellSize);
    int ciz = FloorDiv(cenZ, cellSize);

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
            if (AabbsOverlap(bi, gAabbs[j])) ++c;
        }
    }
    perBodyCount[i] = c;
}
