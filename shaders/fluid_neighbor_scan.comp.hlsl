// Slice FL2 — Deterministic GPU Fluid: the GRID-HASH NEIGHBOR SEARCH neighbor-list EXCLUSIVE PREFIX-SUM of
// the per-particle neighbor counts (the 2nd of the FL2 neighbor-list count->scan->emit; the FPX2
// fpx_pair_scan / CL2 cloth_edge_scan / MC3 mc_scan analog). A SINGLE-THREAD allocator ([numthreads(1,1,1)],
// one thread walks particle 0..particleCount-1 ASCENDING, maintaining a running sum) — the DIRECT mirror of
// fluid.h::BuildNeighborList's exclusive scan. The exclusive scan is INHERENTLY sequential
// (neighborStart[i] depends on the sum of all PRIOR counts), so a deterministic SERIAL scan in one thread
// is correct + simplest + GPU==CPU BIT-EXACT.
//
// For each particle i: neighborStart[i] = running; running += perParticleCount[i]. After the loop the
// SENTINEL neighborStart[particleCount] = running (== total neighbor entries) so the emit can size + the
// host can read the total. Pure integer -> bit-identical GPU==CPU + cross-backend; default MSL gen suffices.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 perParticleCount : one uint per particle (the FL2 neighbor count), READ.
//   b1 neighborStart    : particleCount+1 uints (the exclusive prefix-sum + the sentinel total), WRITE.
//   b2 gParams          : the FluidGridParams (dim.w = particleCount), READ.

struct FluidGridParams {
    int4 grid;
    int4 dim;    // dim.w = particleCount
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>            perParticleCount : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            neighborStart    : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<FluidGridParams> gParams          : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: only thread 0 does the serial scan

    uint particleCount = (uint)gParams[0].dim.w;
    uint running = 0u;
    for (uint i = 0u; i < particleCount; ++i) {
        neighborStart[i] = running;            // exclusive: base = sum of all PRIOR counts
        running += perParticleCount[i];
    }
    neighborStart[particleCount] = running;    // sentinel: == total neighbor entries
}
