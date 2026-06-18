// Slice NAV1 — Deterministic GPU Navmesh BEACHHEAD: the EXCLUSIVE PREFIX-SUM of the per-column raw
// span counts (the MC3 mc_scan / FPX2 fpx_pair_scan analog on columns). A SINGLE-THREAD allocator
// ([numthreads(1,1,1)], one thread walks col 0..columnCount-1 ASCENDING, maintaining a running sum) —
// the DIRECT mirror of navmesh.h::RasterizeTriangleSpans's exclusive scan (and mc_scan.comp /
// fpx_pair_scan.comp / the VT2 vt_alloc single-thread allocator). The exclusive scan is INHERENTLY
// sequential (colOffset[col] depends on the sum of all PRIOR counts), so a deterministic SERIAL scan in
// one thread is correct + simplest + GPU==CPU BIT-EXACT at this scale.
//
// For each column: colOffset[col] = running; running += colCount[col]. The host then knows each
// column's disjoint write base into gSpans. Pure integer -> bit-identical GPU==CPU + cross-backend; the
// default MSL gen suffices (no atomics, no int64 -> NO --msl-version 20200).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gColCount  : one uint per column (the NAV1 raw span count), READ.
//   b1 gColOffset : one uint per column (the exclusive prefix-sum write base), WRITE.
//   b2 gParams    : { columnCount, _, _, _ }, READ.

struct Params {
    uint columnCount;   // number of columns (== hf.columnCount())
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>   gColCount  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   gColOffset : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<Params> gParams    : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    // SINGLE THREAD: only thread 0 does the serial scan (guard defensively so a larger dispatch can't
    // double-run the scan).
    if (gid.x != 0u) return;

    uint columnCount = gParams[0].columnCount;

    uint running = 0u;
    for (uint col = 0u; col < columnCount; ++col) {
        gColOffset[col] = running;          // exclusive: base = sum of all PRIOR counts
        running += gColCount[col];
    }
}
