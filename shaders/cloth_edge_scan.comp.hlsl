// Slice CL2 — Deterministic GPU Cloth: the DISTANCE-CONSTRAINT GRAPH EXCLUSIVE PREFIX-SUM of the per-
// particle edge counts (the FPX2 fpx_pair_scan / MC3 mc_scan analog on lattice particles). A SINGLE-THREAD
// allocator ([numthreads(1,1,1)], one thread walks particle 0..particleCount-1 ASCENDING, maintaining a
// running sum) — the DIRECT mirror of cloth.h::BuildConstraints's exclusive scan (and fpx_pair_scan.comp /
// nav_raster_scan.comp / mc_scan.comp). The exclusive scan is INHERENTLY sequential (gEdgeOffset[p] depends
// on the sum of all PRIOR counts), so a deterministic SERIAL scan in one thread is correct + simplest +
// GPU==CPU BIT-EXACT at this scale.
//
// For each particle p: gEdgeOffset[p] = running; running += gEdgeCount[p]. The emit pass then knows each
// particle's disjoint write base into gEdges. Pure integer -> bit-identical GPU==CPU + cross-backend; the
// default MSL gen suffices (no atomics, no int64 -> NO --msl-version 20200).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gEdgeCount  : one uint per particle (the CL2 owned-edge count), READ.
//   b1 gEdgeOffset : one uint per particle (the exclusive prefix-sum write base), WRITE.
//   b2 gParams     : { particleCount, _, _, _ }, READ.

struct Params {
    uint particleCount;   // number of particles (W*H)
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>   gEdgeCount  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   gEdgeOffset : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<Params> gParams     : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    // SINGLE THREAD: only thread 0 does the serial scan (guard defensively so a larger dispatch can't
    // double-run the scan).
    if (gid.x != 0u) return;

    uint particleCount = gParams[0].particleCount;

    uint running = 0u;
    for (uint p = 0u; p < particleCount; ++p) {
        gEdgeOffset[p] = running;          // exclusive: base = sum of all PRIOR counts
        running += gEdgeCount[p];
    }
}
