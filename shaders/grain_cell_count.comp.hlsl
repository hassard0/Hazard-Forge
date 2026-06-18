// Slice GR2 — Deterministic GPU Granular/Sand: the GRID-HASH NEIGHBOR SEARCH per-cell CELL-COUNT compute
// pass (the 1st of the GR2 cell-table count->scan->emit; the FL2 fluid_cell_count / FPX2 fpx_pair_count / CL2
// cloth_edge_count twin on grains). ONE thread per GRAIN i (i < grainCount). The thread computes grain i's
// flat cell id (GrainCellOf = FloorDiv per axis at cell-size hSearch, offset into the bounded dense grid by
// cellMin, then the CellId linearization — VERBATIM engine/sim/grain.h) and ATOMICALLY increments
// cellCount[cellId] (InterlockedAdd — multiple grains land in the same cell, so the per-cell counter is a
// race; atomic add keeps it deterministic in VALUE, the order-independent sum). enabled=0 -> no increment
// (cellCount stays the cleared all-zero upload, the byte-identical no-op).
//
// WHY BIT-IDENTICAL to the CPU grain.h::BuildGrainCellTable's count (the make-or-break): the cell id is PURE
// INT32 — FloorDiv is integer divide + a sign-correct adjust, the offset is integer subtract, CellId is
// integer mul/add. NO fxmul, NO int64, NO float. So this MSL-generates NATIVELY on Metal (unlike GR1's int64
// grain_integrate which is Vulkan-only), the FL2/FPX2/CL2 precedent. The per-cell COUNT is order-independent
// (a sum), so InterlockedAdd gives the same final value on every vendor -> the host's GPU==CPU memcmp
// (cellCount) catches any divergence in the cell math.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gGrains   : the Q16.16 GrainParticle array (pos.xyz, prev.xyz, vel.xyz, invMass, radius, flags —
//                  std430 ints, 48 bytes), READ (only pos is read here).
//   b1 cellCount : one uint per cell (the per-cell grain count), READ+WRITE (InterlockedAdd).
//   b2 gParams   : the GrainGridParams (hSearch, cellMin.xyz, gridDim.xyz, grainCount, cellCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_GRAIN_THREADS 64

// std430 GrainParticle mirror (engine/sim/grain.h::GrainParticle): 12 x 4-byte = 48 bytes (memcmp-able).
struct GrainParticle {
    int  px, py, pz;     // Q16.16 current position
    int  prx, pry, prz;  // Q16.16 previous position
    int  vx, vy, vz;     // Q16.16 velocity
    int  invMass;        // Q16.16 inverse mass
    int  radius;         // Q16.16 grain radius
    uint flags;          // bit0 = STATIC
};

// std430 grid params (the C++ GrainGridParams upload mirror).
//   grid : x=hSearch (Q16.16 cell size), y=cellMinX, z=cellMinY, w=cellMinZ
//   dim  : x=gridDimX, y=gridDimY, z=gridDimZ, w=grainCount
//   cfg  : x=cellCount, y=enabled, z=unused, w=unused
struct GrainGridParams {
    int4 grid;
    int4 dim;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<GrainParticle>   gGrains   : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            cellCount : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<GrainGridParams> gParams   : register(u2);

// FloorDiv(n, d): deterministic FLOOR division for positive divisor d, ANY-sign n — VERBATIM
// engine/sim/fpx.h::FloorDiv. Pure int32 (no int64).
int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

// FlatCellId(px, py, pz): grain position -> flat cell id in the bounded dense grid — VERBATIM the CPU
// GrainCellOf + FlatGrainCellId (FloorDiv per axis at cell-size hSearch, offset by cellMin, CellId into gridDim).
uint FlatCellId(int px, int py, int pz) {
    int h        = gParams[0].grid.x;
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gx = gParams[0].dim.x, gy = gParams[0].dim.y;   // gridDim.x, gridDim.y
    int cx = FloorDiv(px, h) - cellMinX;
    int cy = FloorDiv(py, h) - cellMinY;
    int cz = FloorDiv(pz, h) - cellMinZ;
    return (uint)((cz * gy + cy) * gx + cx);
}

[numthreads(HF_GRAIN_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int grainCount = gParams[0].dim.w;
    int enabled    = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= grainCount) return;
    if (enabled == 0) return;   // disabled -> cellCount stays cleared (byte-identical no-op)

    GrainParticle p = gGrains[i];
    uint cell = FlatCellId(p.px, p.py, p.pz);
    InterlockedAdd(cellCount[cell], 1u);   // per-cell count is an order-independent sum -> deterministic
}
