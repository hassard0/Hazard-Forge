// Slice GF1 — Deterministic Grain<->Fluid Coupling: the SHARED-GRID CROSS QUERY grain->fluid EXCLUSIVE
// PREFIX-SUM of the per-grain fluid-neighbour counts (the 2nd of the gf cross list count->scan->emit; the GR2
// grain_neighbor_scan / FL2 fluid_neighbor_scan twin). A SINGLE-THREAD allocator ([numthreads(1,1,1)], one
// thread walks grain 0..grainCount-1 ASCENDING, maintaining a running sum) — the DIRECT mirror of
// couple_gf.h::BuildCGFNeighbors's gf exclusive scan. The exclusive scan is INHERENTLY sequential
// (gfStart[i] depends on the sum of all PRIOR counts), so a deterministic SERIAL scan in one thread is correct
// + simplest + GPU==CPU BIT-EXACT.
//
// For each grain i: gfStart[i] = running; running += gfCount[i]. After the loop the SENTINEL
// gfStart[grainCount] = running (== total gf cross entries) so the emit can size + the host can read the total.
// Pure integer -> bit-identical GPU==CPU + cross-backend; default MSL gen suffices.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gfCount : one uint per grain (the gf cross-neighbour count), READ.
//   b1 gfStart : grainCount+1 uints (the exclusive prefix-sum + the sentinel total), WRITE.
//   b2 gParams : the CGFGridParams (dim.w = grainCount), READ.

struct CGFGridParams {
    int4 grid;
    int4 dim;    // dim.w = grainCount
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>          gfCount : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>          gfStart : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<CGFGridParams> gParams : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: only thread 0 does the serial scan

    uint grainCount = (uint)gParams[0].dim.w;
    uint running = 0u;
    for (uint i = 0u; i < grainCount; ++i) {
        gfStart[i] = running;            // exclusive: base = sum of all PRIOR counts
        running += gfCount[i];
    }
    gfStart[grainCount] = running;       // sentinel: == total gf cross entries
}
