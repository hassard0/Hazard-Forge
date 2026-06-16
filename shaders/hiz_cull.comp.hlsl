// Slice CJ — Hi-Z OCCLUSION cull compute. Extends the Slice CD fully-GPU-driven cull
// (gpudriven_cull.comp.hlsl) with a hierarchical-Z OCCLUSION test: after the frustum test, an object
// that passed frustum is ALSO tested against a Hi-Z max-depth pyramid (built CPU-side from a depth
// pre-pass + uploaded; see engine/render/hiz.h) and DROPPED from the compacted survivors iff it is
// FULLY HIDDEN behind closer geometry. The survivors are ORDER-compacted (the AR single-workgroup
// prefix-sum trick -> SOURCE-INDEX order) exactly like CD, so the result is deterministic + the image
// golden is stable.
//
// The occlusion math is the EXACT mirror of hf::render::hiz::IsOccluded (project the 8 AABB corners ->
// screen rect + nearest NDC depth; pick the coarsest mip covering the rect in <=2x2 texels; occluded
// iff the object's nearest depth is FARTHER (> in [0,1] depth, LARGER == farther) than the Hi-Z MAX
// across every covered texel). CONSERVATIVE: near-plane straddle (clip w<=0), partially-off-screen,
// or a degenerate rect -> KEEP (never false-cull). The CPU hiz reference computes the SAME occluded
// count; the showcase asserts the GPU count == the CPU count AND the occlusion-culled image is
// BYTE-IDENTICAL to a frustum-only render (the culled objects were fully hidden -> zero pixels).
//
// Buffers (storage, bound at compute bindings 0..6):
//   b0 gObjects  : source per-object records (model+material+localSphere+slice+localAabb), READ.
//   b1 gPerDraw  : compacted survivor per-draw records (model+material+texIndex), WRITE.
//   b2 gCommands : compacted survivor MDI commands (5x u32), WRITE.
//   b3 gCount    : survivor count + frustumKept + occluded (3x u32), WRITE (thread 0).
//   b4 gParams   : 6 frustum planes + counts + screen dims + mip count, READ.
//   b5 gViewProj : the column-major view-proj matrix (for the AABB projection), READ.
//   b6 gHiZ      : the flat Hi-Z pyramid (all mip texels concatenated) + a mip table header, READ.

#define HF_MAX_OBJECTS 1024
#define HF_MAX_MIPS    16

// Source per-object record (std430), matches the C++ upload struct. Adds the object's LOCAL-space AABB
// (min.xyz / max.xyz, in two float4) on top of CD's record so the occlusion test can build the world
// AABB exactly like the CPU path. mat4 (4 float4) + material + localSphere + slice(uint4) + aabbMin +
// aabbMax = 9 float4 = 144 bytes.
struct ObjectIn {
    float4 modelC0;     // column 0
    float4 modelC1;     // column 1
    float4 modelC2;     // column 2
    float4 modelC3;     // column 3
    float4 material;    // x=metallic, y=roughness-tint, ...
    float4 localSphere; // xyz = local center, w = local radius
    uint4  slice;       // x=indexCount, y=firstIndex, z=vertexOffset, w=texIndex
    float4 aabbMin;     // xyz = local AABB min (w unused)
    float4 aabbMax;     // xyz = local AABB max (w unused)
};

struct PerDrawOut {
    float4 modelC0;
    float4 modelC1;
    float4 modelC2;
    float4 modelC3;
    float4 material;
    uint   texIndex;
    uint3  pad;
};

struct CommandOut {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    uint vertexOffset;
    uint firstInstance;
};

// Cull params (std430): 6 frustum planes + counts (objectCount, screenW, screenH, mipCount) + the
// per-mip table (offset into gHiZ, width, height, _) for up to HF_MAX_MIPS levels.
struct Params {
    float4 planes[6];          // 6 frustum planes (n.xyz, d)
    uint4  counts;             // x=objectCount, y=screenW, z=screenH, w=mipCount
    uint4  mips[HF_MAX_MIPS];  // per-mip: x=offset(in floats), y=width, z=height, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<ObjectIn>   gObjects  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<PerDrawOut> gPerDraw  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<CommandOut> gCommands : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>       gCount    : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<Params>     gParams   : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<float4x4>   gViewProj : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<float>      gHiZ      : register(u6);

groupshared uint gKeep[HF_MAX_OBJECTS];

// Conservative bounding-sphere frustum cull (mirrors frustum.h::SphereOutside).
bool SphereOutside(float3 center, float radius) {
    [unroll] for (int i = 0; i < 6; ++i) {
        float4 pl = gParams[0].planes[i];
        if (dot(pl.xyz, center) + pl.w < -radius) return true;
    }
    return false;
}

// Read a Hi-Z texel (mip `level`, texel (tx,ty)) from the flat pyramid via the mip table.
float HiZAt(uint level, int tx, int ty) {
    uint4 mi = gParams[0].mips[level];
    uint w = mi.y;
    return gHiZ[mi.x + (uint)ty * w + (uint)tx];
}

// The EXACT mirror of hf::render::hiz::IsOccluded. Builds the world AABB from the LOCAL AABB + model,
// projects the 8 corners through the view-proj, forms the screen pixel rect + nearest NDC depth, picks
// the coarsest mip covering the rect in <=2x2 texels, and returns true (OCCLUDED) iff the nearest
// depth is FARTHER than the Hi-Z MAX across every covered texel. Conservative KEEP on any uncertainty.
bool IsOccluded(ObjectIn o, float4x4 vp, int screenW, int screenH, uint mipCount) {
    if (mipCount == 0u || screenW <= 0 || screenH <= 0) return false;

    float3 lmn = o.aabbMin.xyz;
    float3 lmx = o.aabbMax.xyz;

    float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f, nearestZ = 1e30f;
    [unroll] for (int c = 0; c < 8; ++c) {
        float3 corner = float3((c & 1) ? lmx.x : lmn.x,
                               (c & 2) ? lmx.y : lmn.y,
                               (c & 4) ? lmx.z : lmn.z);
        // world = model * float4(corner,1)
        float3 wp = (o.modelC0.xyz * corner.x) + (o.modelC1.xyz * corner.y) +
                    (o.modelC2.xyz * corner.z) + o.modelC3.xyz;
        float4 clip = mul(vp, float4(wp, 1.0));
        if (clip.w <= 1e-6f) return false;  // near-plane straddle -> KEEP
        float3 ndc = clip.xyz / clip.w;
        float px = (ndc.x * 0.5f + 0.5f) * (float)screenW;
        float py = (ndc.y * 0.5f + 0.5f) * (float)screenH;
        minX = min(minX, px); maxX = max(maxX, px);
        minY = min(minY, py); maxY = max(maxY, py);
        nearestZ = min(nearestZ, ndc.z);
    }

    if (minX < 0.0f || minY < 0.0f || maxX > (float)screenW || maxY > (float)screenH) return false;
    if (!(maxX > minX) || !(maxY > minY)) return false;
    if (nearestZ < 0.0f || nearestZ > 1.0f) return false;

    int x0 = (int)floor(minX), x1 = (int)floor(maxX);
    int y0 = (int)floor(minY), y1 = (int)floor(maxY);
    x0 = max(0, x0); y0 = max(0, y0);
    x1 = min(screenW - 1, x1); y1 = min(screenH - 1, y1);
    if (x1 < x0 || y1 < y0) return false;

    // Mip selection: coarsest level on which the rect spans <=2 texels per axis.
    uint level = 0u;
    uint maxLevel = mipCount - 1u;
    [loop] while (level < maxLevel) {
        int tx0 = x0 >> (level + 1u), tx1 = x1 >> (level + 1u);
        int ty0 = y0 >> (level + 1u), ty1 = y1 >> (level + 1u);
        if ((tx1 - tx0) <= 1 && (ty1 - ty0) <= 1) { level += 1u; break; }
        level += 1u;
    }

    uint4 mi = gParams[0].mips[level];
    int mw = (int)mi.y, mh = (int)mi.z;
    int mx0 = x0 >> level, mx1 = x1 >> level;
    int my0 = y0 >> level, my1 = y1 >> level;
    mx0 = max(0, mx0); my0 = max(0, my0);
    mx1 = min(mw - 1, mx1); my1 = min(mh - 1, my1);

    [loop] for (int ty = my0; ty <= my1; ++ty)
        [loop] for (int tx = mx0; tx <= mx1; ++tx) {
            float hiZMax = HiZAt(level, tx, ty);
            if (nearestZ <= hiZMax) return false;  // something at/in front here -> KEEP
        }
    return true;
}

[numthreads(HF_MAX_OBJECTS, 1, 1)]
void main(uint3 lid : SV_GroupThreadID) {
    uint i = lid.x;
    uint objectCount = gParams[0].counts.x;
    int  screenW = (int)gParams[0].counts.y;
    int  screenH = (int)gParams[0].counts.z;
    uint mipCount = gParams[0].counts.w;
    float4x4 vp = gViewProj[0];

    uint keepFrustum = 0u;  // passed frustum (for the frustumKept stat)
    uint keep = 0u;         // passed frustum AND not occluded (the survivor)
    ObjectIn o;
    if (i < objectCount) {
        o = gObjects[i];
        float3 lc = o.localSphere.xyz;
        float  lr = o.localSphere.w;
        float3 center = (o.modelC0.xyz * lc.x) + (o.modelC1.xyz * lc.y) +
                        (o.modelC2.xyz * lc.z) + o.modelC3.xyz;
        float  radius = lr * length(o.modelC0.xyz);
        if (!SphereOutside(center, radius)) {
            keepFrustum = 1u;
            keep = IsOccluded(o, vp, screenW, screenH, mipCount) ? 0u : 1u;
        }
    }
    gKeep[i] = keep;
    GroupMemoryBarrierWithGroupSync();

    // ORDERED Hillis-Steele inclusive prefix sum (same as CD) over the SURVIVOR keep flags.
    [unroll] for (uint offset = 1; offset < HF_MAX_OBJECTS; offset <<= 1) {
        uint v = gKeep[i];
        uint add = (i >= offset) ? gKeep[i - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gKeep[i] = v + add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint total = gKeep[HF_MAX_OBJECTS - 1];

    if (i < objectCount && keep == 1u) {
        uint dst = gKeep[i] - 1u;

        PerDrawOut pd;
        pd.modelC0 = o.modelC0; pd.modelC1 = o.modelC1;
        pd.modelC2 = o.modelC2; pd.modelC3 = o.modelC3;
        pd.material = o.material;
        pd.texIndex = o.slice.w;
        pd.pad = uint3(0, 0, 0);
        gPerDraw[dst] = pd;

        CommandOut c;
        c.indexCount    = o.slice.x;
        c.instanceCount = 1u;
        c.firstIndex    = o.slice.y;
        c.vertexOffset  = o.slice.z;
        c.firstInstance = 0u;
        gCommands[dst] = c;
    }

    // Thread 0 writes the survivor count. The frustumKept + occluded stats are accumulated separately:
    // we reduce keepFrustum across the workgroup via a second pass over gKeep would clobber it, so do a
    // simple atomic per surviving-frustum thread instead (counts are diagnostics, not the draw count).
    if (i == 0u) {
        gCount[0] = total;        // survivor (frustum AND visible) count == the draw count
    }
    // frustumKept (gCount[1]) and occluded (gCount[2]) via atomics over the threads that passed frustum.
    if (i < objectCount && keepFrustum == 1u) {
        uint dummy;
        InterlockedAdd(gCount[1], 1u, dummy);            // frustumKept++
        if (keep == 0u) InterlockedAdd(gCount[2], 1u, dummy);  // occluded (passed frustum, hidden)++
    }
}
