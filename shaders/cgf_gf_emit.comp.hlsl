// Slice GF1 — Deterministic Grain<->Fluid Coupling: the SHARED-GRID CROSS QUERY grain->fluid per-grain
// NEIGHBOR-EMIT compute pass (the 3rd of the gf cross list count->scan->emit; the GR2 grain_neighbor_emit /
// FL2 fluid_neighbor_emit twin applied CROSS-POOL). ONE thread per GRAIN i (i < grainCount). The thread
// re-scans the SAME 27-cell stencil in the SAME fixed order over the SHARED grid + the FLUID cell table, reads
// its write base gfStart[i] (the prefix-sum), and EMITS each accepted FLUID index j into gfNeighbors[] at
// gfStart[i] + local++. Each grain writes into its OWN DISJOINT [gfStart[i], gfStart[i]+count[i]) range ->
// race-free, NO atomics. The order is ascending stencil-cell (dz,dy,dx) then ascending j (the fluid cell table
// is ascending-index per cell) -> VERBATIM the CPU couple_gf.h::BuildCGFNeighbors gf emit order. NO j==i
// self-skip (distinct pools). enabled=0 -> emit nothing (gfNeighbors stays the pre-cleared upload).
//
// WHY BIT-IDENTICAL to the CPU couple_gf.h (gf direction, the make-or-break): every value written is PURE INT32
// — the same per-axis box reject over the same host-snapped positions, the same stencil walk in the same fixed
// order, and each grain's range is disjoint so a thread race CANNOT change any byte. A divergence vs the header
// is exactly what the host's GPU==CPU memcmp (gfNeighbors) catches. INT32 only -> MSL-gens natively on Metal.
//
// Buffers (storage, bound at compute bindings 0..5; on Metal these land at buffer(0..5)):
//   b0 gFluid     : the Q16.16 FluidParticle array (44-byte std430 ints), READ (pos) — the TARGET pool.
//   b1 cellStart  : the FLUID cell table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellTargets: the FLUID cell table fluid indices grouped by cell, READ.
//   b3 gGrains    : the Q16.16 GrainParticle array (48-byte std430 ints), READ (pos) — the QUERY pool.
//   b4 gfStart    : the gf cross-list prefix-sum write base (grainCount+1), READ.
//   b5 gfNeighbors: the output fluid-index array grouped by grain, WRITE (pre-cleared).
//   b6 gParams    : the CGFGridParams, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_CGF_THREADS 64

struct FluidParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

struct GrainParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    int  radius;
    uint flags;
};

struct CGFGridParams {
    int4 grid;
    int4 dim;    // dim.w = grainCount
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle> gFluid      : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>          cellStart   : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>          cellTargets : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<GrainParticle> gGrains     : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>          gfStart     : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<uint>          gfNeighbors : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<CGFGridParams> gParams     : register(u6);

int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

bool CrossAccept(int ax, int ay, int az, int bx, int by, int bz, int h) {
    int dx = ax - bx; if (dx < 0) dx = -dx;
    int dy = ay - by; if (dy < 0) dy = -dy;
    int dz = az - bz; if (dz < 0) dz = -dz;
    return dx < h && dy < h && dz < h;
}

[numthreads(HF_CGF_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int h        = gParams[0].grid.x;
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gdx = gParams[0].dim.x, gdy = gParams[0].dim.y, gdz = gParams[0].dim.z;
    int grainCount = gParams[0].dim.w;
    int enabled    = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= grainCount) return;
    if (enabled == 0) return;

    GrainParticle pi = gGrains[i];
    int cix = FloorDiv(pi.px, h);
    int ciy = FloorDiv(pi.py, h);
    int ciz = FloorDiv(pi.pz, h);

    uint base  = gfStart[i];   // this grain's disjoint write base (from the prefix-sum)
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
            uint j = cellTargets[s];
            FluidParticle pj = gFluid[j];
            if (CrossAccept(pi.px, pi.py, pi.pz, pj.px, pj.py, pj.pz, h)) {
                gfNeighbors[base + local] = j;
                local += 1u;
            }
        }
    }
}
