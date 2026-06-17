// Slice VT2 — Runtime Virtual Texturing Slice 2: the PHYSICAL TILE-POOL ALLOCATION + virtual->physical
// INDIRECTION pass. A SINGLE-THREAD allocator ([numthreads(1,1,1)], one thread walks pageId
// 0..pageCount-1 ASCENDING, maintaining nextTile) — the DIRECT mirror of render/vt.h::AllocatePhysicalTiles
// (and of the VSM Slice VB allocator vsm.h::AllocatePhysicalPages). Allocation is INHERENTLY sequential
// (nextTile depends on prior assignments), so unlike VT1's order-independent set-write the GPU allocation
// needs a deterministic SERIAL scan — one thread is correct + simplest at this scale (pageCount ~340).
//
// For each pageId: gIndirection[pageId] = (resident && nextTile < tileCapacity) ? nextTile++ : 0xFFFFFFFF.
// 0xFFFFFFFF is the kNoTile SSBO sentinel (cast to int32 -1 on read-back; == vt.h::kNoTileU32). The
// ascending walk = mip-major = finest-mip-first priority: the first tileCapacity resident pages (by
// pageId) win the pool; the rest overflow to kNoTile. The GPU==CPU memcmp (vs the CPU AllocatePhysicalTiles)
// proves the GPU allocator reproduces the CPU indirection table BIT-EXACT. Pure integer -> bit-identical
// GPU==CPU + cross-backend; the default MSL gen suffices (no atomics, no integer texture.read -> NO
// --msl-version 20200).
//
// allocEnabled push flag (gParams.allocEnabled): 0 -> EVERY entry written 0xFFFFFFFF (all kNoTile, the
// disabled-path no-op) — written explicitly so the read-back proves the no-op regardless of the upload.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gFeedback     : the VT1 resident page set, feedback[pageId] in {0,1}, READ.
//   b1 gIndirection  : the virtual->physical indirection table, indirection[pageId] (tile or 0xFFFFFFFF), WRITE.
//   b2 gParams       : { pageCount, tileCapacity, allocEnabled, _ }, READ.

struct Params {
    uint pageCount;      // number of virtual page-table slots (== vt.pageCount())
    uint tileCapacity;   // physical tile pool capacity (== pool.tileCapacity())
    uint allocEnabled;   // 0 -> write all kNoTile (disabled no-op); 1 -> allocate
    uint _pad;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>   gFeedback    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   gIndirection : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<Params> gParams      : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    // SINGLE THREAD: only thread 0 does the serial scan (the dispatch is 1 group of 1 thread, but guard
    // defensively so a larger dispatch can't double-run the allocation).
    if (gid.x != 0u) return;

    uint pageCount     = gParams[0].pageCount;
    uint tileCapacity  = gParams[0].tileCapacity;
    uint allocEnabled  = gParams[0].allocEnabled;

    uint nextTile = 0u;
    for (uint pageId = 0u; pageId < pageCount; ++pageId) {
        uint tile = 0xFFFFFFFFu;  // kNoTile default (non-resident, overflow, or disabled)
        if (allocEnabled != 0u && gFeedback[pageId] != 0u && nextTile < tileCapacity) {
            tile = nextTile;
            nextTile += 1u;
        }
        gIndirection[pageId] = tile;
    }
}
