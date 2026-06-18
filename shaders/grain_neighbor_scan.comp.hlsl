// Slice GR2 — Deterministic GPU Granular/Sand: the GRID-HASH NEIGHBOR SEARCH neighbor-list EXCLUSIVE
// PREFIX-SUM of the per-grain neighbor counts (the 2nd of the GR2 neighbor-list count->scan->emit; the FL2
// fluid_neighbor_scan twin). A SINGLE-THREAD allocator ([numthreads(1,1,1)], one thread walks grain
// 0..grainCount-1 ASCENDING, maintaining a running sum) — the DIRECT mirror of grain.h::
// BuildGrainNeighborList's exclusive scan. The exclusive scan is INHERENTLY sequential (neighborStart[i]
// depends on the sum of all PRIOR counts), so a deterministic SERIAL scan in one thread is correct + simplest
// + GPU==CPU BIT-EXACT.
//
// For each grain i: neighborStart[i] = running; running += perGrainCount[i]. After the loop the SENTINEL
// neighborStart[grainCount] = running (== total neighbor entries) so the emit can size + the host can read
// the total. Pure integer -> bit-identical GPU==CPU + cross-backend; default MSL gen suffices.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 perGrainCount  : one uint per grain (the GR2 neighbor count), READ.
//   b1 neighborStart  : grainCount+1 uints (the exclusive prefix-sum + the sentinel total), WRITE.
//   b2 gParams        : the GrainGridParams (dim.w = grainCount), READ.

struct GrainGridParams {
    int4 grid;
    int4 dim;    // dim.w = grainCount
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>            perGrainCount : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            neighborStart : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<GrainGridParams> gParams       : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: only thread 0 does the serial scan

    uint grainCount = (uint)gParams[0].dim.w;
    uint running = 0u;
    for (uint i = 0u; i < grainCount; ++i) {
        neighborStart[i] = running;            // exclusive: base = sum of all PRIOR counts
        running += perGrainCount[i];
    }
    neighborStart[grainCount] = running;       // sentinel: == total neighbor entries
}
