// Slice MC3 — GPU Isosurface Meshing Slice 3 (part a): the EXCLUSIVE PREFIX-SUM of the per-cell
// MARCHING-CUBES triangle counts. A SINGLE-THREAD allocator ([numthreads(1,1,1)], one thread walks cell
// 0..cellCount-1 ASCENDING, maintaining a running sum) — the DIRECT mirror of render/mc.h::
// PrefixSumOffsets (and the VT2 vt_alloc.comp single-thread-allocator pattern). The exclusive scan is
// INHERENTLY sequential (gOffsets[cell] depends on the sum of all PRIOR counts), so a deterministic SERIAL
// scan in one thread is correct + simplest + GPU==CPU BIT-EXACT at this scale (32768 cells). A multi-block
// parallel scan is a deferred optimization.
//
// For each cell: gOffsets[cell] = running; running += gCounts[cell]. The host then knows each cell's write
// offset into the emit output buffers. The GPU==CPU memcmp (vs PrefixSumOffsets) proves the GPU reproduces
// the CPU offset table BIT-EXACT. Pure integer -> bit-identical GPU==CPU + cross-backend; the default MSL
// gen suffices (no atomics, no integer texture.read -> NO --msl-version 20200, the mc_classify/mc_count
// lesson).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gCounts  : one uint per cell (the MC2 triangle count), READ.
//   b1 gOffsets : one uint per cell (the exclusive prefix-sum), WRITE.
//   b2 gParams  : { cellCount, _, _, _ }, READ.

struct Params {
    uint cellCount;   // number of cells (== field.cellCount())
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>   gCounts  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   gOffsets : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<Params> gParams  : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    // SINGLE THREAD: only thread 0 does the serial scan (the dispatch is 1 group of 1 thread, but guard
    // defensively so a larger dispatch can't double-run the scan).
    if (gid.x != 0u) return;

    uint cellCount = gParams[0].cellCount;

    uint running = 0u;
    for (uint cell = 0u; cell < cellCount; ++cell) {
        gOffsets[cell] = running;          // exclusive: offset = sum of all PRIOR counts
        running += gCounts[cell];
    }
}
