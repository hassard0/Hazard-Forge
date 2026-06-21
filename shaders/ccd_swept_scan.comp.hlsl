// Slice CD2 — Deterministic Integer CCD: THE SWEPT-AABB BROADPHASE EXCLUSIVE PREFIX-SUM of the per-body pair
// counts (the 2nd of the CD2 swept pair count->scan->emit; the broad_pair_scan twin, byte-identical logic). A
// SINGLE-THREAD allocator ([numthreads(1,1,1)], one thread walks body 0..bodyCount-1 ASCENDING, maintaining a
// running sum) — the DIRECT mirror of ccd.h::BuildSweptPairsFromAabbs' exclusive scan. The exclusive scan is
// INHERENTLY sequential (perBodyOffset[i] depends on the sum of all PRIOR counts), so a deterministic SERIAL
// scan in one thread is correct + simplest + GPU==CPU BIT-EXACT.
//
// For each body i: perBodyOffset[i] = running; running += perBodyCount[i]. After the loop a SENTINEL at
// perBodyOffset[bodyCount] = running (== total pairs) lets a reader treat body i's slice as
// [perBodyOffset[i], perBodyOffset[i+1]). Pure integer -> bit-identical GPU==CPU + cross-backend; default MSL
// gen suffices (no atomics, no int64).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 perBodyCount  : one uint per body (the CD2 per-body pair count), READ.
//   b1 perBodyOffset : bodyCount+1 uints (the exclusive prefix-sum + the sentinel total), WRITE.
//   b2 gParams       : the BodyGridParams (dim.w = bodyCount), READ.

struct BodyGridParams {
    int4 grid;
    int4 dim;   // dim.w = bodyCount
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>           perBodyCount  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>           perBodyOffset : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<BodyGridParams> gParams       : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: only thread 0 does the serial scan

    uint bodyCount = (uint)gParams[0].dim.w;
    uint running = 0u;
    for (uint i = 0u; i < bodyCount; ++i) {
        perBodyOffset[i] = running;             // exclusive: base = sum of all PRIOR per-body counts
        running += perBodyCount[i];
    }
    perBodyOffset[bodyCount] = running;         // sentinel: == total pairs
}
