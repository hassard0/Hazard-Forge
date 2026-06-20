// Slice BD2 — Deterministic GPU Crowds: the GRID-HASH NEIGHBOR LIST cell-table EXCLUSIVE PREFIX-SUM of the
// per-cell agent counts (the 2nd of the BD2 cell-table count->scan->emit; the GR2 grain_cell_scan twin). A
// SINGLE-THREAD allocator ([numthreads(1,1,1)], one thread walks cell 0..cellCount-1 ASCENDING, maintaining a
// running sum) — the DIRECT mirror of boids.h::BuildBoidsCellTable's exclusive scan. The exclusive scan is
// INHERENTLY sequential (cellStart[c] depends on the sum of all PRIOR counts), so a deterministic SERIAL scan
// in one thread is correct + simplest + GPU==CPU BIT-EXACT.
//
// For each cell c: cellStart[c] = running; running += cellAgentCount[c]. After the loop the SENTINEL
// cellStart[cellCount] = running (== agentCount, the total) so the emit can read cell c's slice as
// [cellStart[c], cellStart[c+1]). Pure integer -> bit-identical GPU==CPU + cross-backend; default MSL gen
// suffices (no atomics, no int64).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 cellAgentCount : one uint per cell (the BD2 per-cell count), READ.
//   b1 cellStart      : cellCount+1 uints (the exclusive prefix-sum + the sentinel total), WRITE.
//   b2 gParams        : the BoidsGridParams (cfg.x = cellCount), READ.

struct BoidsGridParams {
    int4 grid;
    int4 dim;
    int4 cfg;   // cfg.x = cellCount, cfg.y = enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>            cellAgentCount : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            cellStart      : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<BoidsGridParams> gParams        : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: only thread 0 does the serial scan

    uint cellCount = (uint)gParams[0].cfg.x;
    uint running = 0u;
    for (uint c = 0u; c < cellCount; ++c) {
        cellStart[c] = running;             // exclusive: base = sum of all PRIOR cell counts
        running += cellAgentCount[c];
    }
    cellStart[cellCount] = running;         // sentinel: == agentCount (the total)
}
