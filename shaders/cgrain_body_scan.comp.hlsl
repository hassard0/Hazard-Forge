// Slice CG1 — Deterministic Rigid<->Grain Coupling: the BODY->GRAIN grid-hash QUERY EXCLUSIVE PREFIX-SUM of
// the per-body gathered counts (the 2nd of the CG1 query count->scan->emit; the CP1 couple_body_scan twin). A
// SINGLE-THREAD allocator ([numthreads(1,1,1)], one thread walks body 0..bodyCount-1 ASCENDING, maintaining a
// running sum) — the DIRECT mirror of couple_grain.h::GatherBodyGrains's exclusive scan. The exclusive scan is
// INHERENTLY sequential (bodyStart[i] depends on the sum of all PRIOR counts), so a deterministic SERIAL scan
// in one thread is correct + simplest + GPU==CPU BIT-EXACT.
//
// For each body i: bodyStart[i] = running; running += perBodyCount[i]. After the loop the SENTINEL
// bodyStart[bodyCount] = running (== total gathered entries) so the emit can size + the host can read the
// total. Pure integer -> bit-identical GPU==CPU + cross-backend; default MSL gen suffices.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 perBodyCount : one uint per body (the CG1 gathered count), READ.
//   b1 bodyStart    : bodyCount+1 uints (the exclusive prefix-sum + the sentinel total), WRITE.
//   b2 gParams      : the CGrainParams (dim.w = bodyCount), READ.

struct CGrainParams {
    int4 grid;
    int4 dim;    // dim.w = bodyCount
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>          perBodyCount : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>          bodyStart    : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<CGrainParams>  gParams      : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: only thread 0 does the serial scan

    uint bodyCount = (uint)gParams[0].dim.w;
    uint running = 0u;
    for (uint i = 0u; i < bodyCount; ++i) {
        bodyStart[i] = running;            // exclusive: base = sum of all PRIOR counts
        running += perBodyCount[i];
    }
    bodyStart[bodyCount] = running;        // sentinel: == total gathered entries
}
