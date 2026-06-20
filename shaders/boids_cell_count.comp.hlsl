// Slice BD2 — Deterministic GPU Crowds: the GRID-HASH NEIGHBOR LIST per-cell CELL-COUNT compute pass (the
// 1st of the BD2 cell-table count->scan->emit; the GR2 grain_cell_count twin on agents). ONE thread per
// AGENT i (i < agentCount). The thread computes agent i's flat cell id (BoidsCellOf = FloorDiv per axis at
// cell-size radius, offset into the bounded dense grid by cellMin, then the CellId linearization — VERBATIM
// engine/sim/boids.h) and ATOMICALLY increments cellCount[cellId] (InterlockedAdd — multiple agents land in
// the same cell, so the per-cell counter is a race; atomic add keeps it deterministic in VALUE, the
// order-independent sum). enabled=0 -> no increment (cellCount stays the cleared all-zero upload, the
// byte-identical no-op).
//
// WHY BIT-IDENTICAL to the CPU boids.h::BuildBoidsCellTable's count (the make-or-break): the cell id is PURE
// INT32 — FloorDiv is integer divide + a sign-correct adjust, the offset is integer subtract, CellId is
// integer mul/add. NO fxmul, NO int64, NO float. So this MSL-generates NATIVELY on Metal (unlike BD1's int64
// boids_steer which is Vulkan-only), the GR2/FL2/FPX2 precedent — a TRUE GPU pass on BOTH backends, the
// strongest cross-vendor proof. The per-cell COUNT is order-independent (a sum), so InterlockedAdd gives the
// same final value on every vendor -> the host's GPU==CPU memcmp (cellCount) catches any cell-math divergence.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gAgents   : the Q16.16 Agent array (pos.xyz, vel.xyz — std430 ints, 24 bytes), READ (only pos is read).
//   b1 cellCount : one uint per cell (the per-cell agent count), READ+WRITE (InterlockedAdd).
//   b2 gParams   : the BoidsGridParams (radius, cellMin.xyz, gridDim.xyz, agentCount, cellCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_BOIDS_THREADS 64

// std430 Agent mirror (engine/sim/boids.h::Agent): 6 x 4-byte = 24 bytes (memcmp-able).
struct Agent {
    int px, py, pz;   // Q16.16 current position
    int vx, vy, vz;   // Q16.16 velocity
};

// std430 grid params (the C++ BoidsGridParams upload mirror).
//   grid : x=radius (Q16.16 cell size), y=cellMinX, z=cellMinY, w=cellMinZ
//   dim  : x=gridDimX, y=gridDimY, z=gridDimZ, w=agentCount
//   cfg  : x=cellCount, y=enabled, z=unused, w=unused
struct BoidsGridParams {
    int4 grid;
    int4 dim;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<Agent>           gAgents   : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            cellCount : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<BoidsGridParams> gParams   : register(u2);

// FloorDiv(n, d): deterministic FLOOR division for positive divisor d, ANY-sign n — VERBATIM
// engine/sim/fpx.h::FloorDiv. Pure int32 (no int64).
int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

// FlatCellId(px, py, pz): agent position -> flat cell id in the bounded dense grid — VERBATIM the CPU
// BoidsCellOf + FlatBoidsCellId (FloorDiv per axis at cell-size radius, offset by cellMin, CellId into gridDim).
uint FlatCellId(int px, int py, int pz) {
    int h        = gParams[0].grid.x;
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gx = gParams[0].dim.x, gy = gParams[0].dim.y;   // gridDim.x, gridDim.y
    int cx = FloorDiv(px, h) - cellMinX;
    int cy = FloorDiv(py, h) - cellMinY;
    int cz = FloorDiv(pz, h) - cellMinZ;
    return (uint)((cz * gy + cy) * gx + cx);
}

[numthreads(HF_BOIDS_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int agentCount = gParams[0].dim.w;
    int enabled    = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= agentCount) return;
    if (enabled == 0) return;   // disabled -> cellCount stays cleared (byte-identical no-op)

    Agent p = gAgents[i];
    uint cell = FlatCellId(p.px, p.py, p.pz);
    InterlockedAdd(cellCount[cell], 1u);   // per-cell count is an order-independent sum -> deterministic
}
