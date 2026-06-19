// Slice FR2 — Deterministic Rigid-Body Fracture/Destruction: FRAGMENT EXTRACTION SAMPLE-SCATTER compute
// pass (the 3rd of the FR2 count->scan->emit; the GR2 grain_cell_emit / FL2 fluid_cell_emit twin).
// SINGLE-THREAD ([numthreads(1,1,1)]): one thread walks sample 0..sampleCount-1 ASCENDING and scatters each
// sample index into its cell's slice at fragStart[cell] + cursor[cell]++. The within-cell order is
// ASCENDING SAMPLE INDEX — VERBATIM the CPU fract.h::ExtractFragments emit. THIS IS WHY IT IS SINGLE-THREAD
// (the determinism crux, the GR2/FL2 lesson): multiple samples scatter into the SAME cell, so a parallel
// atomic-cursor emit would order them by GPU SCHEDULING (non-deterministic). Walking samples ascending in
// ONE thread with a per-cell cursor reproduces the CPU's ascending-index within-cell order EXACTLY ->
// bit-identical. (The cell COUNT pass is an order-independent sum so it is parallel+atomic; only the EMIT's
// within-cell ORDER forces single-thread.)
//
// The per-cell cursor is kept in the cellCursor buffer (pre-cleared to 0; one uint per cell), NOT a
// groupshared/local array (seedCount can exceed any fixed local size). enabled=0 -> emit nothing
// (fragSamples stays the cleared upload). The sample->cell map is gCells[sample] DIRECTLY (FR1's cellId[],
// no recompute). Pure int32 -> MSL-gens natively on Metal.
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gCells     : one uint per lattice sample = the FR1 cellId, READ.
//   b1 fragStart  : the FR2 cell-table exclusive prefix-sum (seedCount+1), READ.
//   b2 cellCursor : one uint per cell, the per-cell write cursor, READ+WRITE (pre-cleared to 0).
//   b3 fragSamples: the output sample-index array grouped by cell, WRITE (pre-cleared).
//   b4 gParams    : the FR2 params (sampleCount, seedCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

struct FractFragParams {
    int4 cfg0;   // x=sampleCount, y=seedCount, z=nx, w=ny
    int4 cfg1;   // x=nz, y=fragCount, z=enabled, w=unused
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>            gCells      : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            fragStart   : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>            cellCursor  : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>            fragSamples : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<FractFragParams> gParams     : register(u4);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: ascending-sample scatter (the within-cell order crux)

    int sampleCount = gParams[0].cfg0.x;
    int seedCount   = gParams[0].cfg0.y;
    int enabled     = gParams[0].cfg1.z;
    if (enabled == 0) return;

    for (int s = 0; s < sampleCount; ++s) {
        uint cell = gCells[(uint)s];
        if ((int)cell >= seedCount) continue;
        uint local = cellCursor[cell];
        fragSamples[fragStart[cell] + local] = (uint)s;
        cellCursor[cell] = local + 1u;
    }
}
