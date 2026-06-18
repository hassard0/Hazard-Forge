// Slice GR2 — Deterministic GPU Granular/Sand: the GRID-HASH NEIGHBOR SEARCH per-grain NEIGHBOR-COUNT compute
// pass (the 1st of the GR2 neighbor-list count->scan->emit; the FL2 fluid_neighbor_count twin). ONE thread
// per GRAIN i (i < grainCount). The thread computes grain i's cell, scans the 27 cells of its 3x3x3 stencil
// (the cell + its 26 neighbors), and for each grain j != i in those cells (read from the GR2 cell table
// cellStart/cellGrains) counts it iff the PURE INT32 per-axis GrainNeighborAccept (|pos_i.axis - pos_j.axis| <
// hSearch on every axis) passes -> perGrainCount[i]. NO atomics (each thread writes its OWN perGrainCount[i],
// a per-grain independent count). enabled=0 -> write 0.
//
// WHY BIT-IDENTICAL to the CPU grain.h::CountGrainNeighbors (the make-or-break): GrainNeighborAccept is a
// per-axis integer subtract + abs + compare — PURE INT32, NO products, NO int64, NO sqrt (the exact radial
// overlap cull is DEFERRED to GR3's contact solve; the box-candidate set is correct since GR3's projection is
// a no-op beyond overlap). The stencil walk is the same fixed (dz,dy,dx) order over the same cell table. So
// this MSL-generates NATIVELY on Metal (unlike GR1's int64 fxmul integrator) -> the host's GPU==CPU memcmp
// (perGrainCount) catches divergence.
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gGrains       : the Q16.16 GrainParticle array (48-byte std430 ints), READ (pos).
//   b1 cellStart     : the GR2 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellGrains    : the GR2 cell-table grain indices grouped by cell, READ.
//   b3 perGrainCount : one uint per grain (the candidate-neighbor count), WRITE.
//   b4 gParams       : the GrainGridParams (hSearch, cellMin.xyz, gridDim.xyz, grainCount, cellCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_GRAIN_THREADS 64

struct GrainParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    int  radius;
    uint flags;
};

struct GrainGridParams {
    int4 grid;   // x=hSearch, y=cellMinX, z=cellMinY, w=cellMinZ
    int4 dim;    // x=gridDimX, y=gridDimY, z=gridDimZ, w=grainCount
    int4 cfg;    // x=cellCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<GrainParticle>   gGrains       : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            cellStart     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>            cellGrains    : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>            perGrainCount : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<GrainGridParams> gParams       : register(u4);

int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

// GrainNeighborAccept(ax,ay,az, bx,by,bz, hSearch): the PURE INT32 per-axis |dx| < hSearch candidate test —
// VERBATIM grain.h::GrainNeighborAccept. NO products, NO int64, NO sqrt.
bool GrainNeighborAccept(int ax, int ay, int az, int bx, int by, int bz, int hSearch) {
    int dx = ax - bx; if (dx < 0) dx = -dx;
    int dy = ay - by; if (dy < 0) dy = -dy;
    int dz = az - bz; if (dz < 0) dz = -dz;
    return dx < hSearch && dy < hSearch && dz < hSearch;
}

[numthreads(HF_GRAIN_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int h        = gParams[0].grid.x;
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gdx = gParams[0].dim.x, gdy = gParams[0].dim.y, gdz = gParams[0].dim.z;
    int grainCount = gParams[0].dim.w;
    int enabled    = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= grainCount) return;
    if (enabled == 0) { perGrainCount[i] = 0u; return; }

    GrainParticle pi = gGrains[i];
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
            uint j = cellGrains[s];
            if (j == i) continue;
            GrainParticle pj = gGrains[j];
            if (GrainNeighborAccept(pi.px, pi.py, pi.pz, pj.px, pj.py, pj.pz, h)) c += 1u;
        }
    }
    perGrainCount[i] = c;   // per-grain independent count (NO atomics)
}
