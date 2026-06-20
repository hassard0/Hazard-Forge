// Slice BD2 — Deterministic GPU Crowds: the GRID-HASH NEIGHBOR LIST neighbor-list EXCLUSIVE PREFIX-SUM of the
// per-agent neighbor counts (the 2nd of the BD2 neighbor-list count->scan->emit; the GR2 grain_neighbor_scan
// twin). A SINGLE-THREAD allocator ([numthreads(1,1,1)], one thread walks agent 0..agentCount-1 ASCENDING,
// maintaining a running sum) — the DIRECT mirror of boids.h::BuildBoidsNeighborList's exclusive scan. The
// exclusive scan is INHERENTLY sequential (neighborStart[i] depends on the sum of all PRIOR counts), so a
// deterministic SERIAL scan in one thread is correct + simplest + GPU==CPU BIT-EXACT.
//
// For each agent i: neighborStart[i] = running; running += perAgentCount[i]. After the loop the SENTINEL
// neighborStart[agentCount] = running (== total neighbor entries) so the emit can size + the host can read
// the total. Pure integer -> bit-identical GPU==CPU + cross-backend; default MSL gen suffices.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 perAgentCount  : one uint per agent (the BD2 neighbor count), READ.
//   b1 neighborStart  : agentCount+1 uints (the exclusive prefix-sum + the sentinel total), WRITE.
//   b2 gParams        : the BoidsGridParams (dim.w = agentCount), READ.

struct BoidsGridParams {
    int4 grid;
    int4 dim;    // dim.w = agentCount
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>            perAgentCount : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            neighborStart : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<BoidsGridParams> gParams       : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: only thread 0 does the serial scan

    uint agentCount = (uint)gParams[0].dim.w;
    uint running = 0u;
    for (uint i = 0u; i < agentCount; ++i) {
        neighborStart[i] = running;            // exclusive: base = sum of all PRIOR counts
        running += perAgentCount[i];
    }
    neighborStart[agentCount] = running;       // sentinel: == total neighbor entries
}
