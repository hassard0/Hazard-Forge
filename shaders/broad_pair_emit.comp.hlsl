// Slice BP2 — Deterministic Integer Broadphase: THE CANDIDATE-PAIR GENERATOR PAIR-SCATTER compute pass (the
// 3rd of the BP2 pair count->scan->emit; the broad_cell_emit / grain_neighbor_emit twin). SINGLE-THREAD
// ([numthreads(1,1,1)]): one thread walks body 0..bodyCount-1 ASCENDING and, for each body i, scatters every
// accepted candidate (i, j>i) into i's DISJOINT slice at perBodyOffset[i] + (a per-body local cursor), in the
// FIXED (dz,dy,dx) stencil order — VERBATIM the CPU broad.h::BuildBroadphasePairs emit. THIS IS WHY IT IS
// SINGLE-THREAD (the DET-CRUX): the per-i emit ORDER (stencil cells, then ascending j within a cell) must be
// reproduced bit-for-bit; a parallel atomic cursor would make the within-list order GPU-scheduling-dependent
// -> nondeterministic. (Each body i's slice IS disjoint, so the WRITES don't race; but the per-i emit ORDER
// is the contract, so we mirror the CPU's single-thread ascending walk with a per-body local cursor.)
//
// WHY BIT-IDENTICAL to the CPU broad.h::BuildBroadphasePairs (the make-or-break): same FloorDiv cell math,
// same FIXED stencil order, same j>i guard, same six-compare AabbOverlap, same per-body offset base + local
// cursor -> the pairsOut array is byte-for-byte the CPU's. Pure int32 -> MSL-gens natively on Metal.
//
// Buffers (storage, bound at compute bindings 0..5; on Metal these land at buffer(0..5)):
//   b0 gBodies       : the Q16.16 FxBody mirror array (16-byte std430 ints), READ (pos + radius).
//   b1 cellStart     : the BP1 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellBodies    : the BP1 body indices grouped by cell (bodyCount), READ.
//   b3 perBodyOffset : the BP2 per-body exclusive prefix-sum (bodyCount+1), READ.
//   b4 pairsOut      : the output FxPair array {i,j} (totalPairs), WRITE (pre-cleared).
//   b5 gParams       : the BodyGridParams, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

// std430 FxBody mirror (16 bytes; matches broad_pair_count).
struct BroadBody {
    int  px, py, pz;
    int  radius;
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

[[vk::binding(0, 0)]] RWStructuredBuffer<BroadBody>      gBodies       : register(u0);
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
        BroadBody bi = gBodies[i];
        int cix = FloorDiv(bi.px, cellSize);
        int ciy = FloorDiv(bi.py, cellSize);
        int ciz = FloorDiv(bi.pz, cellSize);
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
                if (BodiesOverlap(bi, gBodies[j])) {
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
