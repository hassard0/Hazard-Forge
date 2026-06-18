// Slice GF1 — Deterministic Grain<->Fluid Coupling: the SHARED-GRID CROSS QUERY fluid->grain EXCLUSIVE
// PREFIX-SUM of the per-fluid grain-neighbour counts (the 2nd of the fg cross list count->scan->emit; the GR2
// grain_neighbor_scan / FL2 fluid_neighbor_scan twin, the gf scan MIRROR). A SINGLE-THREAD allocator
// ([numthreads(1,1,1)], one thread walks fluid 0..fluidCount-1 ASCENDING, maintaining a running sum) — the
// DIRECT mirror of couple_gf.h::BuildCGFNeighbors's fg exclusive scan. The exclusive scan is INHERENTLY
// sequential, so a deterministic SERIAL scan in one thread is correct + simplest + GPU==CPU BIT-EXACT.
//
// For each fluid i: fgStart[i] = running; running += fgCount[i]. The SENTINEL fgStart[fluidCount] = running
// (== total fg cross entries). Pure integer -> bit-identical GPU==CPU + cross-backend; default MSL gen suffices.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 fgCount : one uint per fluid (the fg cross-neighbour count), READ.
//   b1 fgStart : fluidCount+1 uints (the exclusive prefix-sum + the sentinel total), WRITE.
//   b2 gParams : the CGFGridParams (dim.w = fluidCount), READ.

struct CGFGridParams {
    int4 grid;
    int4 dim;    // dim.w = fluidCount
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>          fgCount : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>          fgStart : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<CGFGridParams> gParams : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: only thread 0 does the serial scan

    uint fluidCount = (uint)gParams[0].dim.w;
    uint running = 0u;
    for (uint i = 0u; i < fluidCount; ++i) {
        fgStart[i] = running;            // exclusive: base = sum of all PRIOR counts
        running += fgCount[i];
    }
    fgStart[fluidCount] = running;       // sentinel: == total fg cross entries
}
