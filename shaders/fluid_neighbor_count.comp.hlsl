// Slice FL2 — Deterministic GPU Fluid: the GRID-HASH NEIGHBOR SEARCH per-particle NEIGHBOR-COUNT compute
// pass (the 1st of the FL2 neighbor-list count->scan->emit; the FPX2 fpx_pair_count / CL2 cloth_edge_count
// analog). ONE thread per PARTICLE i (i < particleCount). The thread computes particle i's cell, scans the
// 27 cells of its 3x3x3 stencil (the cell + its 26 neighbors), and for each particle j != i in those cells
// (read from the FL2 cell table cellStart/cellParticles) counts it iff the PURE INT32 per-axis
// NeighborAccept (|p_i.axis - p_j.axis| < h on every axis) passes -> perParticleCount[i]. NO atomics (each
// thread writes its OWN perParticleCount[i], a per-particle independent count). enabled=0 -> write 0.
//
// WHY BIT-IDENTICAL to the CPU fluid.h::CountNeighbors (the make-or-break): NeighborAccept is a per-axis
// integer subtract + abs + compare — PURE INT32, NO products, NO int64, NO sqrt (the exact radial r<h cull
// is DEFERRED to FL3; the box-candidate set is correct since FL3's kernel is 0 beyond h). The stencil walk
// is the same fixed (dz,dy,dx) order over the same cell table. So this MSL-generates NATIVELY on Metal
// (unlike FL1's int64 fxmul integrator) -> the host's GPU==CPU memcmp (perParticleCount) catches divergence.
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gParticles        : the Q16.16 FluidParticle array (44-byte std430 ints), READ (pos).
//   b1 cellStart         : the FL2 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellParticles     : the FL2 cell-table particle indices grouped by cell, READ.
//   b3 perParticleCount  : one uint per particle (the candidate-neighbor count), WRITE.
//   b4 gParams           : the FluidGridParams (h, cellMin.xyz, gridDim.xyz, particleCount, cellCount, enabled), READ.
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

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle>  gParticles       : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>           cellStart        : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>           cellParticles    : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>           perParticleCount : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<FluidGridParams> gParams         : register(u4);

int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

// NeighborAccept(ax,ay,az, bx,by,bz, h): the PURE INT32 per-axis |dx| < h candidate test — VERBATIM
// fluid.h::NeighborAccept. NO products, NO int64, NO sqrt.
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
    if (enabled == 0) { perParticleCount[i] = 0u; return; }

    FluidParticle pi = gParticles[i];
    int cix = FloorDiv(pi.px, h);
    int ciy = FloorDiv(pi.py, h);
    int ciz = FloorDiv(pi.pz, h);

    uint c = 0u;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int ncx = cix + dx, ncy = ciy + dy, ncz = ciz + dz;
        // Skip stencil cells outside the bounded grid (clamp).
        if (ncx < cellMinX || ncx >= cellMinX + gdx) continue;
        if (ncy < cellMinY || ncy >= cellMinY + gdy) continue;
        if (ncz < cellMinZ || ncz >= cellMinZ + gdz) continue;
        uint cell = (uint)(((ncz - cellMinZ) * gdy + (ncy - cellMinY)) * gdx + (ncx - cellMinX));
        uint s0 = cellStart[cell], s1 = cellStart[cell + 1u];
        for (uint s = s0; s < s1; ++s) {
            uint j = cellParticles[s];
            if (j == i) continue;
            FluidParticle pj = gParticles[j];
            if (NeighborAccept(pi.px, pi.py, pi.pz, pj.px, pj.py, pj.pz, h)) c += 1u;
        }
    }
    perParticleCount[i] = c;   // per-particle independent count (NO atomics)
}
