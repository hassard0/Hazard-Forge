// Slice FL2 — Deterministic GPU Fluid: the GRID-HASH NEIGHBOR SEARCH per-cell CELL-COUNT compute pass
// (the 1st of the FL2 cell-table count->scan->emit; the FPX2 fpx_pair_count / CL2 cloth_edge_count /
// MC2 mc_count analog on fluid particles). ONE thread per PARTICLE i (i < particleCount). The thread
// computes particle i's flat cell id (CellOf = FloorDiv per axis at cell-size h, offset into the bounded
// dense grid by cellMin, then the CellId linearization — VERBATIM engine/sim/fluid.h) and ATOMICALLY
// increments cellCount[cellId] (InterlockedAdd — multiple particles land in the same cell, so the per-cell
// counter is a race; atomic add keeps it deterministic in VALUE, the order-independent sum). enabled=0 ->
// no increment (cellCount stays the cleared all-zero upload, the byte-identical no-op).
//
// WHY BIT-IDENTICAL to the CPU fluid.h::BuildCellTable's count (the make-or-break): the cell id is PURE
// INT32 — FloorDiv is integer divide + a sign-correct adjust, the offset is integer subtract, CellId is
// integer mul/add. NO fxmul, NO int64, NO float. So this MSL-generates NATIVELY on Metal (unlike FL1's
// int64 fxmul integrator which is Vulkan-only), the FPX2/CL2 precedent. The per-cell COUNT is
// order-independent (a sum), so InterlockedAdd gives the same final value on every vendor -> the host's
// GPU==CPU memcmp (cellCount) catches any divergence in the cell math.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gParticles : the Q16.16 FluidParticle array (pos.xyz, prev.xyz, vel.xyz, invMass, flags — std430
//                   ints, 44 bytes), READ (only pos is read here).
//   b1 cellCount  : one uint per cell (the per-cell particle count), READ+WRITE (InterlockedAdd).
//   b2 gParams    : the FluidGridParams (h, cellMin.xyz, gridDim.xyz, particleCount, cellCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_FLUID_THREADS 64

// std430 FluidParticle mirror (engine/sim/fluid.h::FluidParticle): 11 x 4-byte = 44 bytes (memcmp-able).
struct FluidParticle {
    int  px, py, pz;     // Q16.16 current position
    int  prx, pry, prz;  // Q16.16 previous position
    int  vx, vy, vz;     // Q16.16 velocity
    int  invMass;        // Q16.16 inverse mass
    uint flags;          // bit0 = STATIC
};

// std430 grid params (the C++ FluidGridParams upload mirror).
//   grid : x=h (Q16.16 cell size), y=cellMinX, z=cellMinY, w=cellMinZ
//   dim  : x=gridDimX, y=gridDimY, z=gridDimZ, w=particleCount
//   cfg  : x=cellCount, y=enabled, z=unused, w=unused
struct FluidGridParams {
    int4 grid;
    int4 dim;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle>  gParticles : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>           cellCount  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<FluidGridParams> gParams   : register(u2);

// FloorDiv(n, d): deterministic FLOOR division for positive divisor d, ANY-sign n — VERBATIM
// engine/sim/fpx.h::FloorDiv (the swraster floorDiv). Pure int32 (no int64).
int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

// FlatCellId(px, py, pz): particle position -> flat cell id in the bounded dense grid — VERBATIM the
// CPU CellOf + FlatCellId (FloorDiv per axis at cell-size h, offset by cellMin, CellId into gridDim).
uint FlatCellId(int px, int py, int pz) {
    int h        = gParams[0].grid.x;
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gx = gParams[0].dim.x, gy = gParams[0].dim.y;   // gridDim.x, gridDim.y
    int cx = FloorDiv(px, h) - cellMinX;
    int cy = FloorDiv(py, h) - cellMinY;
    int cz = FloorDiv(pz, h) - cellMinZ;
    return (uint)((cz * gy + cy) * gx + cx);
}

[numthreads(HF_FLUID_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int particleCount = gParams[0].dim.w;
    int enabled       = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= particleCount) return;
    if (enabled == 0) return;   // disabled -> cellCount stays cleared (byte-identical no-op)

    FluidParticle p = gParticles[i];
    uint cell = FlatCellId(p.px, p.py, p.pz);
    InterlockedAdd(cellCount[cell], 1u);   // per-cell count is an order-independent sum -> deterministic
}
