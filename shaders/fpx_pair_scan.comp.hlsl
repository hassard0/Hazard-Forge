// Slice FPX2 — Deterministic Fixed-Point Physics: the integer-AABB BROADPHASE EXCLUSIVE PREFIX-SUM of the
// per-body pair counts (the MC3 mc_scan analog on bodies). A SINGLE-THREAD allocator ([numthreads(1,1,1)],
// one thread walks body 0..bodyCount-1 ASCENDING, maintaining a running sum) — the DIRECT mirror of
// fpx.h::BuildPairs's exclusive scan (and mc_scan.comp / the VT2 single-thread allocator). The exclusive
// scan is INHERENTLY sequential (perBodyOffset[i] depends on the sum of all PRIOR counts), so a
// deterministic SERIAL scan in one thread is correct + simplest + GPU==CPU BIT-EXACT at this scale.
//
// For each body i: perBodyOffset[i] = running; running += perBodyCount[i]. The host then knows each
// body's disjoint write base into gPairs. Pure integer -> bit-identical GPU==CPU + cross-backend; the
// default MSL gen suffices (no atomics, no int64 -> NO --msl-version 20200).
//
// A dedicated fpx_pair_scan (vs reusing mc_scan): mc_scan's binding layout (gCounts/gOffsets/gParams with
// {cellCount}) would actually FIT, but a dedicated shader is clearer + names match the FPX2 buffers.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 perBodyCount  : one uint per body (the FPX2 pair count), READ.
//   b1 perBodyOffset : one uint per body (the exclusive prefix-sum write base), WRITE.
//   b2 gParams       : { bodyCount, _, _, _ }, READ.

struct Params {
    uint bodyCount;   // number of bodies
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>   perBodyCount  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   perBodyOffset : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<Params> gParams       : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    // SINGLE THREAD: only thread 0 does the serial scan (guard defensively so a larger dispatch can't
    // double-run the scan).
    if (gid.x != 0u) return;

    uint bodyCount = gParams[0].bodyCount;

    uint running = 0u;
    for (uint i = 0u; i < bodyCount; ++i) {
        perBodyOffset[i] = running;        // exclusive: base = sum of all PRIOR counts
        running += perBodyCount[i];
    }
}
