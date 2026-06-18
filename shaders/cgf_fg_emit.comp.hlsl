// Slice GF1 — Deterministic Grain<->Fluid Coupling: the SHARED-GRID CROSS QUERY fluid->grain per-fluid
// NEIGHBOR-EMIT compute pass (the 3rd of the fg cross list count->scan->emit; the gf emit MIRROR). ONE thread
// per FLUID i (i < fluidCount). The thread re-scans the SAME 27-cell stencil in the SAME fixed order over the
// SHARED grid + the GRAIN cell table, reads its write base fgStart[i] (the prefix-sum), and EMITS each accepted
// GRAIN index j into fgNeighbors[] at fgStart[i] + local++. Each fluid writes into its OWN DISJOINT range ->
// race-free, NO atomics. The order is ascending stencil-cell (dz,dy,dx) then ascending j (the grain cell table
// is ascending-index per cell) -> VERBATIM the CPU couple_gf.h::BuildCGFNeighbors fg emit order. NO j==i
// self-skip (distinct pools). enabled=0 -> emit nothing (fgNeighbors stays the pre-cleared upload).
//
// WHY BIT-IDENTICAL to the CPU couple_gf.h (fg direction): every value written is PURE INT32 — the same per-axis
// box reject over the same host-snapped positions, the same stencil walk in the same fixed order, each fluid's
// range disjoint. The host's GPU==CPU memcmp (fgNeighbors) catches divergence. INT32 only -> MSL-native.
//
// Buffers (storage, bound at compute bindings 0..6; on Metal these land at buffer(0..6)):
//   b0 gGrains    : the Q16.16 GrainParticle array (48-byte std430 ints), READ (pos) — the TARGET pool.
//   b1 cellStart  : the GRAIN cell table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellTargets: the GRAIN cell table grain indices grouped by cell, READ.
//   b3 gFluid     : the Q16.16 FluidParticle array (44-byte std430 ints), READ (pos) — the QUERY pool.
//   b4 fgStart    : the fg cross-list prefix-sum write base (fluidCount+1), READ.
//   b5 fgNeighbors: the output grain-index array grouped by fluid, WRITE (pre-cleared).
//   b6 gParams    : the CGFGridParams, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_CGF_THREADS 64

struct GrainParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    int  radius;
    uint flags;
};

struct FluidParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

struct CGFGridParams {
    int4 grid;
    int4 dim;    // dim.w = fluidCount
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<GrainParticle> gGrains     : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>          cellStart   : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>          cellTargets : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<FluidParticle> gFluid      : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>          fgStart     : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<uint>          fgNeighbors : register(u5);
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
    int fluidCount = gParams[0].dim.w;
    int enabled    = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= fluidCount) return;
    if (enabled == 0) return;

    FluidParticle pi = gFluid[i];
    int cix = FloorDiv(pi.px, h);
    int ciy = FloorDiv(pi.py, h);
    int ciz = FloorDiv(pi.pz, h);

    uint base  = fgStart[i];   // this fluid's disjoint write base (from the prefix-sum)
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
            GrainParticle pj = gGrains[j];
            if (CrossAccept(pi.px, pi.py, pi.pz, pj.px, pj.py, pj.pz, h)) {
                fgNeighbors[base + local] = j;
                local += 1u;
            }
        }
    }
}
