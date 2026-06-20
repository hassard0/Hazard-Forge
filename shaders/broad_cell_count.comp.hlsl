// Slice BP1 — Deterministic Integer Broadphase: THE BODY GRID + CELL TABLE per-cell CELL-COUNT compute pass
// (the 1st of the BP1 cell-table count->scan->emit; the grain_cell_count / boids_cell_count / fluid_cell_count
// twin, keyed on fpx::FxBody instead of grains/agents/particles). ONE thread per BODY i (i < bodyCount). The
// thread computes body i's flat cell id (BodyCellOf = FloorDiv per axis at cell-size cellSize, offset into the
// bounded dense grid by cellMin, then the CellId linearization — VERBATIM engine/sim/broad.h) and ATOMICALLY
// increments cellCount[cellId] (InterlockedAdd — multiple bodies land in the same cell, so the per-cell
// counter is a race; atomic add keeps it deterministic in VALUE, the order-independent sum). enabled=0 -> no
// increment (cellCount stays the cleared all-zero upload, the byte-identical no-op).
//
// WHY BIT-IDENTICAL to the CPU broad.h::BuildBodyCellTable's count (the make-or-break): the cell id is PURE
// INT32 — FloorDiv is integer divide + a sign-correct adjust, the offset is integer subtract, CellId is
// integer mul/add. NO fxmul, NO int64, NO float. So this MSL-generates NATIVELY on Metal (the strongest proof
// tier, the grain_cell/boids_cell precedent). The per-cell COUNT is order-independent (a sum), so
// InterlockedAdd gives the same final value on every vendor -> the host's GPU==CPU memcmp (cellCount) catches
// any divergence in the cell math.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gBodies   : the Q16.16 FxBody mirror array (pos.xyz, radius — 16-byte std430 ints), READ (only pos).
//   b1 cellCount : one uint per cell (the per-cell body count), READ+WRITE (InterlockedAdd).
//   b2 gParams   : the BodyGridParams (cellSize, cellMin.xyz, gridDim.xyz, bodyCount, cellCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_BROAD_THREADS 64

// std430 FxBody mirror (the host BroadBodyGpu upload): pos.xyz + radius = 4 x int32 (16 bytes). BP1 reads ONLY
// pos; radius is carried for parity with the cgrain/couple body uploads (BP2 reads it for the AABB span).
struct BroadBody {
    int  px, py, pz;   // Q16.16 body position
    int  radius;       // Q16.16 body broadphase half-extent (unused in BP1's centre-bucket)
};

// std430 grid params (the C++ BodyGridParams upload mirror).
//   grid : x=cellSize (Q16.16 cell edge), y=cellMinX, z=cellMinY, w=cellMinZ
//   dim  : x=gridDimX, y=gridDimY, z=gridDimZ, w=bodyCount
//   cfg  : x=cellCount, y=enabled, z=unused, w=unused
struct BodyGridParams {
    int4 grid;
    int4 dim;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<BroadBody>      gBodies   : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>           cellCount : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<BodyGridParams> gParams   : register(u2);

// FloorDiv(n, d): deterministic FLOOR division for positive divisor d, ANY-sign n — VERBATIM
// engine/sim/fpx.h::FloorDiv. Pure int32 (no int64).
int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

// FlatCellId(px, py, pz): body position -> flat cell id in the bounded dense grid — VERBATIM the CPU
// BodyCellOf + FlatBodyCellId (FloorDiv per axis at cell-size cellSize, offset by cellMin, CellId into gridDim).
uint FlatCellId(int px, int py, int pz) {
    int h        = gParams[0].grid.x;
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gx = gParams[0].dim.x, gy = gParams[0].dim.y;   // gridDim.x, gridDim.y
    int cx = FloorDiv(px, h) - cellMinX;
    int cy = FloorDiv(py, h) - cellMinY;
    int cz = FloorDiv(pz, h) - cellMinZ;
    return (uint)((cz * gy + cy) * gx + cx);
}

[numthreads(HF_BROAD_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int bodyCount = gParams[0].dim.w;
    int enabled   = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= bodyCount) return;
    if (enabled == 0) return;   // disabled -> cellCount stays cleared (byte-identical no-op)

    BroadBody b = gBodies[i];
    uint cell = FlatCellId(b.px, b.py, b.pz);
    InterlockedAdd(cellCount[cell], 1u);   // per-cell count is an order-independent sum -> deterministic
}
