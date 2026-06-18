// Slice FL2 — Deterministic GPU Fluid: the GRID-HASH NEIGHBOR SEARCH per-particle NEIGHBOR-EMIT compute
// pass (the 3rd of the FL2 neighbor-list count->scan->emit; the FPX2 fpx_pair_emit / CL2 cloth_edge_emit
// analog). ONE thread per PARTICLE i (i < particleCount). The thread re-scans the SAME 27-cell stencil in
// the SAME fixed order, reads its write base neighborStart[i] (the FL2 prefix-sum), and EMITS each accepted
// j into neighbors[] at neighborStart[i] + local++. Each particle writes into its OWN DISJOINT
// [neighborStart[i], neighborStart[i]+count[i]) range -> race-free, NO atomics (the FPX2 per-body-disjoint
// pattern). The order is ascending stencil-cell (dz,dy,dx) then ascending j (cellParticles is ascending-
// index per cell from the cell-table emit) -> VERBATIM the CPU fluid.h::BuildNeighborList emit order.
// enabled=0 -> emit nothing (neighbors stays the pre-cleared upload).
//
// WHY BIT-IDENTICAL to the CPU fluid.h::BuildNeighborList (the make-or-break): every value written is PURE
// INT32 — the same per-axis NeighborAccept over the same host-snapped positions, the same stencil walk in
// the same fixed order, and each particle's range is disjoint so a thread race CANNOT change any byte. A
// divergence vs the header is exactly what the host's GPU==CPU memcmp (neighbors) catches. INT32 only ->
// MSL-gens natively on Metal.
//
// Buffers (storage, bound at compute bindings 0..5; on Metal these land at buffer(0..5)):
//   b0 gParticles     : the Q16.16 FluidParticle array (44-byte std430 ints), READ (pos).
//   b1 cellStart      : the FL2 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellParticles  : the FL2 cell-table particle indices grouped by cell, READ.
//   b3 neighborStart  : the FL2 neighbor-list prefix-sum write base (particleCount+1), READ.
//   b4 neighbors      : the output candidate-neighbor j-index array grouped by i, WRITE (pre-cleared).
//   b5 gParams        : the FluidGridParams, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_FLUID_THREADS 64

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
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>           cellParticles : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>           neighborStart : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>           neighbors     : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<FluidGridParams> gParams      : register(u5);

int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

bool NeighborAccept(int ax, int ay, int az, int bx, int by, int bz, int h) {
    int dx = ax - bx; if (dx < 0) dx = -dx;
    int dy = ay - by; if (dy < 0) dy = -dy;
    int dz = az - bz; if (dz < 0) dz = -dz;
    return dx < h && dy < h && dz < h;
}

[numthreads(HF_FLUID_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int h        = gParams[0].grid.x;
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gdx = gParams[0].dim.x, gdy = gParams[0].dim.y, gdz = gParams[0].dim.z;
    int particleCount = gParams[0].dim.w;
    int enabled       = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= particleCount) return;
    if (enabled == 0) return;

    FluidParticle pi = gParticles[i];
    int cix = FloorDiv(pi.px, h);
    int ciy = FloorDiv(pi.py, h);
    int ciz = FloorDiv(pi.pz, h);

    uint base  = neighborStart[i];   // this particle's disjoint write base (from the prefix-sum)
    uint local = 0u;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int ncx = cix + dx, ncy = ciy + dy, ncz = ciz + dz;
        if (ncx < cellMinX || ncx >= cellMinX + gdx) continue;
        if (ncy < cellMinY || ncy >= cellMinY + gdy) continue;
        if (ncz < cellMinZ || ncz >= cellMinZ + gdz) continue;
        uint cell = (uint)(((ncz - cellMinZ) * gdy + (ncy - cellMinY)) * gdx + (ncx - cellMinX));
        uint s0 = cellStart[cell], s1 = cellStart[cell + 1u];
        for (uint s = s0; s < s1; ++s) {
            uint j = cellParticles[s];
            if (j == i) continue;
            FluidParticle pj = gParticles[j];
            if (NeighborAccept(pi.px, pi.py, pi.pz, pj.px, pj.py, pj.pz, h)) {
                neighbors[base + local] = j;
                local += 1u;
            }
        }
    }
}
