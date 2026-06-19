// Slice FR2 — Deterministic Rigid-Body Fracture/Destruction: FRAGMENT EXTRACTION per-cell SAMPLE-COUNT
// compute pass (the 1st of the FR2 count->scan->emit CSR over FR1's cellId[]; the GR2 grain_cell_count /
// FL2 fluid_cell_count twin applied to the fracture lattice). ONE thread per lattice SAMPLE (sample <
// sampleCount). The thread reads its cell id = gCells[sample] (FR1's cellId[] array — NO CellOf recompute,
// the sample->cell map is DIRECT) and ATOMICALLY increments cellCount[cellId] (InterlockedAdd — multiple
// samples land in the same cell, so the per-cell counter is a race; atomic add keeps it deterministic in
// VALUE, the order-independent sum). enabled=0 -> no increment (cellCount stays the cleared all-zero
// upload, the byte-identical no-op).
//
// WHY BIT-IDENTICAL to the CPU fract.h::ExtractFragments count (the make-or-break): the per-cell COUNT is
// PURE INT32 — a direct array read + an atomic add (NO products, NO int64, NO float). So this MSL-generates
// NATIVELY on Metal (the GR2/FL2/FR1 precedent), a TRUE GPU pass on both backends. The per-cell COUNT is
// order-independent (a sum), so InterlockedAdd gives the same final value on every vendor -> the host's
// GPU==CPU memcmp (cellCount) catches any divergence.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gCells    : one uint per lattice sample = the FR1 cellId (the nearest-seed index), READ.
//   b1 cellCount : one uint per cell (== seedCount cells; the per-cell sample count), READ+WRITE (atomic).
//   b2 gParams   : the FR2 params (sampleCount, seedCount, ..., enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_FRACT_THREADS 64

// FR2 params (std430). Mirrors the C++ upload struct.
//   cfg0 : x=sampleCount, y=seedCount, z=nx, w=ny
//   cfg1 : x=nz, y=fragCount, z=enabled, w=unused
struct FractFragParams {
    int4 cfg0;
    int4 cfg1;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>            gCells    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            cellCount : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<FractFragParams> gParams   : register(u2);

[numthreads(HF_FRACT_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int sampleCount = gParams[0].cfg0.x;
    int seedCount   = gParams[0].cfg0.y;
    int enabled     = gParams[0].cfg1.z;

    uint sample = gid.x;
    if ((int)sample >= sampleCount) return;
    if (enabled == 0) return;   // disabled -> cellCount stays cleared (byte-identical no-op)

    uint cell = gCells[sample];
    if ((int)cell >= seedCount) return;            // guard (FR1 guarantees cellId in [0, seedCount))
    InterlockedAdd(cellCount[cell], 1u);           // order-independent sum -> deterministic value
}
