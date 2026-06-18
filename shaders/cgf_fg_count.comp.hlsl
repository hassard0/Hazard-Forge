// Slice GF1 — Deterministic Grain<->Fluid Coupling: the SHARED-GRID CROSS QUERY fluid->grain per-fluid
// NEIGHBOR-COUNT compute pass (the 1st of the fg cross list count->scan->emit; the GR2 grain_neighbor_count /
// FL2 fluid_neighbor_count twin applied CROSS-POOL, the gf MIRROR). ONE thread per FLUID i (i < fluidCount).
// The thread computes fluid i's cell over the SHARED grid, scans the 27 cells of its 3x3x3 stencil, and for
// each GRAIN j in those cells (read from the GRAIN cell table cellStart/cellTargets) counts it iff the PURE
// INT32 per-axis box reject (|fluid_i.axis - grain_j.axis| < h on every axis) passes -> fgCount[i]. NO atomics.
// NO j==i self-skip — the two pools are DISTINCT. enabled=0 -> write 0.
//
// WHY BIT-IDENTICAL to the CPU couple_gf.h::CountCross (fg direction): the box reject is a per-axis integer
// subtract + abs + compare — PURE INT32, NO products, NO int64, NO sqrt. The stencil walk is the same fixed
// (dz,dy,dx) order over the same shared grid + the same grain cell table. MSL-native on Metal -> the host's
// GPU==CPU memcmp (fgCount) catches divergence.
//
// Buffers (storage, bound at compute bindings 0..5; on Metal these land at buffer(0..5)):
//   b0 gGrains    : the Q16.16 GrainParticle array (48-byte std430 ints), READ (pos) — the TARGET pool.
//   b1 cellStart  : the GRAIN cell table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellTargets: the GRAIN cell table grain indices grouped by cell, READ.
//   b3 gFluid     : the Q16.16 FluidParticle array (44-byte std430 ints), READ (pos) — the QUERY pool.
//   b4 fgCount    : one uint per fluid (the cross-pool grain-neighbour count), WRITE.
//   b5 gParams    : the CGFGridParams (h, cellMin.xyz, gridDim.xyz, fluidCount, cellCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_CGF_THREADS 64

struct GrainParticle {       // std430 48-byte mirror (engine/sim/grain.h::GrainParticle)
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    int  radius;
    uint flags;
};

struct FluidParticle {       // std430 44-byte mirror (engine/sim/fluid.h::FluidParticle)
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

struct CGFGridParams {
    int4 grid;   // x=h, y=cellMinX, z=cellMinY, w=cellMinZ
    int4 dim;    // x=gridDimX, y=gridDimY, z=gridDimZ, w=queryCount (fluidCount for fg)
    int4 cfg;    // x=cellCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<GrainParticle> gGrains    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>          cellStart  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>          cellTargets: register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<FluidParticle> gFluid     : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>          fgCount    : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<CGFGridParams> gParams    : register(u5);

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
    if (enabled == 0) { fgCount[i] = 0u; return; }

    FluidParticle pi = gFluid[i];
    int cix = FloorDiv(pi.px, h);
    int ciy = FloorDiv(pi.py, h);
    int ciz = FloorDiv(pi.pz, h);

    uint c = 0u;
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
            if (CrossAccept(pi.px, pi.py, pi.pz, pj.px, pj.py, pj.pz, h)) c += 1u;
        }
    }
    fgCount[i] = c;   // per-fluid independent cross count (NO atomics, NO self-skip)
}
