// Slice CD2 — Deterministic Integer CCD: THE SWEPT-AABB BROADPHASE PAIR-SCATTER compute pass (the 3rd of the CD2
// swept pair count->scan->emit; the broad_pair_emit twin keyed on EXPLICIT swept AABBs). SINGLE-THREAD
// ([numthreads(1,1,1)]): one thread walks body 0..bodyCount-1 ASCENDING and, for each body i, scatters every
// accepted candidate (i, j>i) into i's DISJOINT slice at perBodyOffset[i] + (a per-body local cursor), in the
// FIXED (dz,dy,dx) stencil order — VERBATIM the CPU ccd.h::BuildSweptPairsFromAabbs emit. THIS IS WHY IT IS
// SINGLE-THREAD (the DET-CRUX): the per-i emit ORDER (stencil cells, then ascending j within a cell) must be
// reproduced bit-for-bit; a parallel atomic cursor would make the within-list order GPU-scheduling-dependent ->
// nondeterministic. (Each body i's slice IS disjoint, so the WRITES don't race; but the per-i emit ORDER is the
// contract, so we mirror the CPU's single-thread ascending walk with a per-body local cursor.)
//
// WHY BIT-IDENTICAL to the CPU ccd.h::BuildSweptPairsFromAabbs (the make-or-break): same FloorDiv cell math over
// the AABB centre, same FIXED stencil order, same j>i guard, same six-compare AabbOverlap over lo/hi, same
// per-body offset base + local cursor -> the pairsOut array is byte-for-byte the CPU's. Pure int32 -> MSL-gens
// natively on Metal.
//
// Buffers (storage, bound at compute bindings 0..5; on Metal these land at buffer(0..5)):
//   b0 gAabbs        : the swept-AABB array (lo.xyz + hi.xyz, 24-byte std430 ints), READ.
//   b1 cellStart     : the BP1 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellBodies    : the BP1 body indices grouped by cell (bodyCount), READ.
//   b3 perBodyOffset : the CD2 per-body exclusive prefix-sum (bodyCount+1), READ.
//   b4 pairsOut      : the output FxPair array {i,j} (totalPairs), WRITE (pre-cleared).
//   b5 gParams       : the BodyGridParams, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

// std430 swept-AABB mirror (24 bytes; matches ccd_swept_count).
struct SweptAabb {
    int lox, loy, loz;
    int hix, hiy, hiz;
};

// std430 FxPair mirror: 2 x uint (8 bytes) — matches fpx::FxPair {i, j}.
struct BroadPair {
    uint i, j;
};

struct BodyGridParams {
    int4 grid;   // x=cellSize, y=cellMinX, z=cellMinY, w=cellMinZ
    int4 dim;    // x=gridDimX, y=gridDimY, z=gridDimZ, w=bodyCount
    int4 cfg;    // x=cellCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<SweptAabb>      gAabbs        : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>           cellStart     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>           cellBodies    : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>           perBodyOffset : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<BroadPair>      pairsOut      : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<BodyGridParams> gParams       : register(u5);

int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

uint FlatCellId(int cx, int cy, int cz) {
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gx = gParams[0].dim.x, gy = gParams[0].dim.y;
    int lx = cx - cellMinX, ly = cy - cellMinY, lz = cz - cellMinZ;
    return (uint)((lz * gy + ly) * gx + lx);
}

bool AabbsOverlap(SweptAabb a, SweptAabb b) {
    return a.lox <= b.hix && b.lox <= a.hix &&
           a.loy <= b.hiy && b.loy <= a.hiy &&
           a.loz <= b.hiz && b.loz <= a.hiz;
}

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: ascending-body scatter (the per-i emit-order crux)

    int bodyCount = gParams[0].dim.w;
    int enabled   = gParams[0].cfg.y;
    if (enabled == 0) return;   // disabled -> emit nothing (pairsOut stays the cleared upload)
    int cellSize  = gParams[0].grid.x;
    int cellMinX  = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gdimX = gParams[0].dim.x, gdimY = gParams[0].dim.y, gdimZ = gParams[0].dim.z;

    for (int i = 0; i < bodyCount; ++i) {
        SweptAabb bi = gAabbs[i];
        int cenX = (bi.lox + bi.hix) / 2, cenY = (bi.loy + bi.hiy) / 2, cenZ = (bi.loz + bi.hiz) / 2;
        int cix = FloorDiv(cenX, cellSize);
        int ciy = FloorDiv(cenY, cellSize);
        int ciz = FloorDiv(cenZ, cellSize);
        uint base = perBodyOffset[i];
        uint local = 0u;
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
                if (j <= (uint)i) continue;
                if (AabbsOverlap(bi, gAabbs[j])) {
                    BroadPair p;
                    p.i = (uint)i;
                    p.j = j;
                    pairsOut[base + local] = p;
                    ++local;
                }
            }
        }
    }
}
