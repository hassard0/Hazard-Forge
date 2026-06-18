// Slice FL2 — Deterministic GPU Fluid: the GRID-HASH NEIGHBOR SEARCH cell-table PARTICLE-SCATTER compute
// pass (the 3rd of the FL2 cell-table count->scan->emit). SINGLE-THREAD ([numthreads(1,1,1)]): one thread
// walks particle 0..particleCount-1 ASCENDING and scatters each particle's index into its cell's slice at
// cellStart[cell] + cursor[cell]++. The within-cell order is ASCENDING PARTICLE INDEX — VERBATIM the CPU
// fluid.h::BuildCellTable emit. THIS IS WHY IT IS SINGLE-THREAD (the determinism crux, NOT the FPX2
// per-body-disjoint pattern): multiple particles scatter into the SAME cell, so a parallel atomic-cursor
// emit would order them by GPU SCHEDULING (non-deterministic). Walking particles ascending in ONE thread
// with a per-cell cursor reproduces the CPU's ascending-index within-cell order EXACTLY -> bit-identical.
// The cell COUNT (fluid_cell_count) is an order-independent SUM so it can be parallel+atomic; the EMIT's
// within-cell ORDER is what forces single-thread (the mc_scan/cloth_solve single-thread precedent).
//
// The per-cell cursor is kept in the cellCursor buffer (pre-cleared to 0; one uint per cell), NOT a
// groupshared/local array (cellCount can exceed any fixed local size). enabled=0 -> emit nothing
// (cellParticles stays the cleared upload).
//
// WHY BIT-IDENTICAL to the CPU fluid.h::BuildCellTable (the make-or-break): same FlatCellId integer math,
// same ascending-particle walk, same per-cell cursor -> the cellParticles array is byte-for-byte the CPU's.
// Pure int32 -> MSL-gens natively on Metal.
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gParticles    : the Q16.16 FluidParticle array (44-byte std430 ints), READ (only pos).
//   b1 cellStart     : the FL2 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellCursor    : one uint per cell, the per-cell write cursor, READ+WRITE (pre-cleared to 0).
//   b3 cellParticles : the output particle-index array grouped by cell, WRITE (pre-cleared).
//   b4 gParams       : the FluidGridParams, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

// std430 FluidParticle mirror (44 bytes).
struct FluidParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

struct FluidGridParams {
    int4 grid;   // x=h, y=cellMinX, z=cellMinY, w=cellMinZ
    int4 dim;    // x=gridDimX, y=gridDimY, z=gridDimZ, w=particleCount
    int4 cfg;    // x=cellCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle>  gParticles    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>           cellStart     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>           cellCursor    : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>           cellParticles : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<FluidGridParams> gParams      : register(u4);

int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

uint FlatCellId(int px, int py, int pz) {
    int h        = gParams[0].grid.x;
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gx = gParams[0].dim.x, gy = gParams[0].dim.y;
    int cx = FloorDiv(px, h) - cellMinX;
    int cy = FloorDiv(py, h) - cellMinY;
    int cz = FloorDiv(pz, h) - cellMinZ;
    return (uint)((cz * gy + cy) * gx + cx);
}

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: ascending-particle scatter (the within-cell order crux)

    int particleCount = gParams[0].dim.w;
    int enabled       = gParams[0].cfg.y;
    if (enabled == 0) return;

    for (int i = 0; i < particleCount; ++i) {
        FluidParticle p = gParticles[i];
        uint cell = FlatCellId(p.px, p.py, p.pz);
        uint local = cellCursor[cell];
        cellParticles[cellStart[cell] + local] = (uint)i;
        cellCursor[cell] = local + 1u;
    }
}
