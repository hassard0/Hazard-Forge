// Slice VA — Virtual Shadow Maps Slice 1: the PAGE-NEEDED MARKING compute pass. ONE thread per receiver
// world-sample. Each thread picks the sample's clipmap LEVEL (SelectClipmapLevel — the INTEGER
// threshold-ladder, NO transcendental log2) + projects it into that level's top-down clipmap ortho
// (MarkPage), then sets resident[pageId] = 1. The marked set is a pure INTEGER set (order-independent),
// so writes race-free to 1 and the result is bit-identical GPU==CPU + cross-backend.
//
// The math (SelectClipmapLevel threshold-ladder + MarkPage subtract/divide/floor) is the VERBATIM mirror
// of engine/render/vsm.h. A mismatch shows up as a wrong page marked -> the host's GPU==CPU memcmp fails
// loudly. The level thresholds (level0WorldExtent*2^L) are host-precomputed and uploaded as exact float32
// bits in gParams.thresholds — the CPU + this shader read the SAME bits, so the level selection is a pure
// compare/count with no GPU transcendental on the bit-exact path (the DETERMINISM CRUX).
//
// markingEnabled push flag: false -> every thread returns early, writing NOTHING -> the resident set stays
// the cleared all-zero upload (the byte-identical disabled-path proof).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gReceivers : the receiver world positions (float4, xyz used), READ.
//   b1 gResident  : the virtual page-table resident set, resident[pageId] in {0,1}, WRITE.
//   b2 gParams    : clipmap config (levels/vpps/level0WorldExtent/cameraPos) + the level thresholds, READ.

#define HF_VSM_MAX_LEVELS 16
#define HF_VSM_THREADS    64

// Clipmap config + the host-precomputed level threshold table (std430). Mirrors the C++ upload struct.
//   dims      : x=levels, y=virtualPagesPerSide, z=receiverCount, w=unused
//   level0    : x=level0WorldExtent, yzw unused
//   cameraPos : xyz = clipmap center, w unused
//   thresholds: thresholds[L] = level0WorldExtent * 2^L (exact float32 bits; CPU + shader read the same)
struct Params {
    uint4   dims;
    float4  level0;
    float4  cameraPos;
    float4  thresholds[HF_VSM_MAX_LEVELS / 4];   // 16 floats = HF_VSM_MAX_LEVELS thresholds
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gReceivers : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   gResident  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<Params> gParams    : register(u2);

// Flat page-table index: idx = level*(vpps*vpps) + py*vpps + px. Mirrors vsm.h::PageId.
int PageId(int level, int px, int py, int vpps) {
    return level * (vpps * vpps) + py * vpps + px;
}

// Read threshold L from the float4[] pack (thresholds[L/4][L%4]).
float ThresholdAt(int L) {
    return gParams[0].thresholds[L >> 2][L & 3];
}

// DETERMINISM CRUX — the INTEGER threshold-ladder. level = number of thresholds distToCamera exceeds,
// clamped to [0, levels-1]. Mirrors vsm.h::SelectClipmapLevel VERBATIM (no log2). thresholds[L] are the
// host-precomputed exact float32 bits.
int SelectClipmapLevel(float distToCamera, int levels) {
    int level = 0;
    [loop] for (int L = 0; L < levels; ++L)
        if (distToCamera > ThresholdAt(L)) level = L + 1;
    if (level < 0) level = 0;
    if (level > levels - 1) level = levels - 1;
    return level;
}

// Mark the virtual page a world receiver point needs. Mirrors vsm.h::MarkPage VERBATIM: level via the
// threshold-ladder, project into the level's top-down clipmap ortho (origin = cameraPos snapped to the
// level's page grid; extent = level0WorldExtent*2^level), (px,py) = floor((worldXZ - origin)/pageSize),
// clamp to [0,vpps). Pure subtract/divide/floor — integer-stable cross-backend.
int MarkPage(float3 worldPos, int levels, int vpps, float level0Extent, float3 cameraPos) {
    float3 d = worldPos - cameraPos;
    float dist = length(d);
    int level = SelectClipmapLevel(dist, levels);

    float levelExtent = level0Extent * (float)(1u << (uint)level);   // exact power-of-two scale
    float pageWorldSize = levelExtent / (float)vpps;

    float originX = floor((cameraPos.x - levelExtent * 0.5) / pageWorldSize) * pageWorldSize;
    float originZ = floor((cameraPos.z - levelExtent * 0.5) / pageWorldSize) * pageWorldSize;

    int px = (int)floor((worldPos.x - originX) / pageWorldSize);
    int py = (int)floor((worldPos.z - originZ) / pageWorldSize);

    if (px < 0) px = 0; else if (px > vpps - 1) px = vpps - 1;
    if (py < 0) py = 0; else if (py > vpps - 1) py = vpps - 1;

    return PageId(level, px, py, vpps);
}

[numthreads(HF_VSM_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int levels        = (int)gParams[0].dims.x;
    int vpps          = (int)gParams[0].dims.y;
    uint receiverCount = gParams[0].dims.z;
    uint markingEnabled = gParams[0].dims.w;   // 0 -> write nothing (disabled-path no-op)
    float level0Extent = gParams[0].level0.x;
    float3 cameraPos   = gParams[0].cameraPos.xyz;

    uint i = gid.x;
    if (i >= receiverCount) return;
    if (markingEnabled == 0u) return;   // disabled -> resident stays the cleared all-zero upload

    float3 worldPos = gReceivers[i].xyz;
    int pageId = MarkPage(worldPos, levels, vpps, level0Extent, cameraPos);

    // The set is order-independent: a plain write to 1 (multiple receivers can map to the same page;
    // every write stores the same value). No atomics needed.
    gResident[pageId] = 1u;
}
