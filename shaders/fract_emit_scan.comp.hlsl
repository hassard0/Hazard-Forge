// Slice FR2 — Deterministic Rigid-Body Fracture/Destruction: FRAGMENT EXTRACTION EXCLUSIVE PREFIX-SUM +
// STREAM-COMPACTION compute pass (the 2nd of the FR2 count->scan->emit; the GR2 grain_cell_scan twin, plus
// the CL2/MC3 stream-compaction the grain scan does NOT do). A SINGLE-THREAD allocator ([numthreads(1,1,1)],
// one thread walks cell 0..seedCount-1 ASCENDING, maintaining a running sum) — the DIRECT mirror of
// fract.h::ExtractFragments's exclusive scan. The exclusive scan is INHERENTLY sequential (fragStart[c]
// depends on the sum of all PRIOR counts), so a deterministic SERIAL scan in one thread is correct +
// simplest + GPU==CPU BIT-EXACT.
//
// THE COMPACTION (the FR2 twist over GR2): the SAME ascending walk assigns each NON-EMPTY cell its COMPACT
// fragment index (a second running counter fragCount) -> cellToFragment[c] = fragIdx (or the kNoFragment
// sentinel for an empty/dominated cell) AND fragmentToCell[fragIdx] = c (the inverse the reduce pass reads).
// Fragments are ordered by ASCENDING cell index (deterministic) — the CL2/MC3 compact-non-empty.
//
// For each cell c: fragStart[c] = running; if cellSampleCount[c] > 0 { cellToFragment[c] = fragCount;
// fragmentToCell[fragCount] = c; ++fragCount; } else cellToFragment[c] = kNoFragment; running +=
// cellSampleCount[c]. After the loop the SENTINEL fragStart[seedCount] = running (== sampleCount) so the
// emit can read cell c's slice as [fragStart[c], fragStart[c+1]). fragCount is written to outFragCount[0]
// (the host reads it back to size the compact fragment array). Pure integer -> bit-identical GPU==CPU +
// cross-backend; default MSL gen suffices (no atomics, no int64).
//
// Buffers (storage, bound at compute bindings 0..5; on Metal these land at buffer(0..5)):
//   b0 cellSampleCount : one uint per cell (the FR2 per-cell count), READ.
//   b1 fragStart       : seedCount+1 uints (the exclusive prefix-sum + the sentinel total), WRITE.
//   b2 cellToFragment  : seedCount uints (cell -> fragment idx or kNoFragment), WRITE.
//   b3 fragmentToCell  : seedCount uints (fragment -> source cell; first fragCount entries valid), WRITE.
//   b4 outFragCount    : one uint (the compact fragment count F), WRITE.
//   b5 gParams         : the FR2 params (cfg0.y = seedCount), READ.

static const uint kNoFragment = 0xFFFFFFFFu;   // the cellToFragment sentinel (== fract.h::kNoFragment)

struct FractFragParams {
    int4 cfg0;   // x=sampleCount, y=seedCount, z=nx, w=ny
    int4 cfg1;   // x=nz, y=fragCount, z=enabled, w=unused
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>            cellSampleCount : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            fragStart       : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>            cellToFragment  : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>            fragmentToCell  : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>            outFragCount    : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<FractFragParams> gParams         : register(u5);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: serial scan + compaction (the determinism crux)

    uint seedCount = (uint)gParams[0].cfg0.y;
    uint running   = 0u;
    uint fragCount = 0u;
    for (uint c = 0u; c < seedCount; ++c) {
        fragStart[c] = running;                       // exclusive: base = sum of all PRIOR cell counts
        uint cnt = cellSampleCount[c];
        if (cnt > 0u) {
            cellToFragment[c]            = fragCount;  // compact fragment index (ascending cell)
            fragmentToCell[fragCount]    = c;          // the inverse the reduce pass reads
            ++fragCount;
        } else {
            cellToFragment[c] = kNoFragment;           // empty/dominated cell -> sentinel, no fragment
        }
        running += cnt;
    }
    fragStart[seedCount] = running;                    // sentinel: == sampleCount (every sample owned once)
    outFragCount[0]      = fragCount;                  // F (the host reads it back)
}
