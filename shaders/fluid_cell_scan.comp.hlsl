// Slice FL2 — Deterministic GPU Fluid: the GRID-HASH NEIGHBOR SEARCH cell-table EXCLUSIVE PREFIX-SUM of
// the per-cell particle counts (the 2nd of the FL2 cell-table count->scan->emit; the FPX2 fpx_pair_scan /
// CL2 cloth_edge_scan / MC3 mc_scan analog). A SINGLE-THREAD allocator ([numthreads(1,1,1)], one thread
// walks cell 0..cellCount-1 ASCENDING, maintaining a running sum) — the DIRECT mirror of
// fluid.h::BuildCellTable's exclusive scan. The exclusive scan is INHERENTLY sequential (cellStart[c]
// depends on the sum of all PRIOR counts), so a deterministic SERIAL scan in one thread is correct +
// simplest + GPU==CPU BIT-EXACT.
//
// For each cell c: cellStart[c] = running; running += cellParticleCount[c]. After the loop the SENTINEL
// cellStart[cellCount] = running (== particleCount, the total) so the emit can read cell c's slice as
// [cellStart[c], cellStart[c+1]). Pure integer -> bit-identical GPU==CPU + cross-backend; default MSL gen
// suffices (no atomics, no int64).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 cellParticleCount : one uint per cell (the FL2 per-cell count), READ.
//   b1 cellStart         : cellCount+1 uints (the exclusive prefix-sum + the sentinel total), WRITE.
//   b2 gParams           : the FluidGridParams (cfg.x = cellCount), READ.

struct FluidGridParams {
    int4 grid;
    int4 dim;
    int4 cfg;   // cfg.x = cellCount, cfg.y = enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>            cellParticleCount : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            cellStart         : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<FluidGridParams> gParams           : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: only thread 0 does the serial scan

    uint cellCount = (uint)gParams[0].cfg.x;
    uint running = 0u;
    for (uint c = 0u; c < cellCount; ++c) {
        cellStart[c] = running;             // exclusive: base = sum of all PRIOR cell counts
        running += cellParticleCount[c];
    }
    cellStart[cellCount] = running;         // sentinel: == particleCount (the total)
}
