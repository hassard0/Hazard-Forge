// Slice DU — Virtual-Geometry Slice 3: GPU per-CLUSTER Hi-Z OCCLUSION cull -> indirect cluster draw. DT's
// per-CLUSTER frustum cull (shaders/cluster_cull.comp.hlsl) PLUS CJ's Hi-Z OCCLUSION test
// (shaders/hiz_cull.comp.hlsl): after the frustum test, a frustum-surviving cluster-instance is ALSO tested
// against a Hi-Z max-depth pyramid (built CPU-side from a depth pre-pass + uploaded; see engine/render/hiz.h)
// and DROPPED from the compacted survivors iff its bounding-sphere AABB is FULLY HIDDEN behind closer
// geometry. ONE workgroup of 1024 threads (the (instance x cluster) count is <=1024). One thread per
// cluster-instance; the survivors are ORDER-compacted (the AR single-workgroup prefix-sum trick -> SOURCE-
// INDEX order) exactly like DT, so the result is deterministic + the image golden is stable.
//
// The CONSERVATIVE Hi-Z (only culls clusters guaranteed fully occluded) means the occlusion-on render is
// BYTE-IDENTICAL to the frustum-only (occlusionEnabled=0) render. SphereOutside is copied VERBATIM from
// cluster_cull.comp / frustum.h; IsOccluded is copied VERBATIM from hiz_cull.comp / hiz.h (the cluster AABB
// is computed inline from worldCenter +- worldRadius — the bounding-sphere AABB the CPU mirror projects).
// The CPU mirror engine/render/cluster_cull.h::CullClusterInstancesHiZ computes the SAME survivors; the
// showcase asserts the GPU survivor count == the CPU count AND the GPU-culled image is byte-identical.
//
// Buffers (storage, bound at compute bindings 0..6):
//   b0 gClusters : source per-cluster-instance records (model + color + worldSphere + slice), READ.
//   b1 gPerDraw  : compacted survivor per-draw records (model + color), WRITE.
//   b2 gCommands : compacted survivor MDI commands (5x u32), WRITE.
//   b3 gCount    : survivor count + frustumKept + occluded (3x u32), WRITE (thread 0 / atomics).
//   b4 gParams   : 6 frustum planes + counts(clusterCount, screenW, screenH, mipCount/occlusionEnabled) +
//                  the per-mip table, READ.
//   b5 gViewProj : the column-major view-proj matrix (for the AABB projection), READ.
//   b6 gHiZ      : the flat Hi-Z pyramid (all mip texels concatenated) + a mip table header, READ.

#define HF_MAX_CLUSTERS 1024
#define HF_MAX_MIPS     16

// Source per-cluster-instance record (std430). Matches the C++ upload struct + cluster_cull.comp's
// ClusterIn EXACTLY: mat4 (4 float4) + color (float4) + worldSphere (xyz=center, w=radius) + slice (uint4)
// = 7 float4 = 112 bytes. The world sphere is precomputed CPU-side (InstanceWorldSphere) so the shader does
// the IDENTICAL frustum + AABB the CPU mirror does — bit-for-bit on the cull decision.
struct ClusterIn {
    float4 modelC0;     // column 0
    float4 modelC1;     // column 1
    float4 modelC2;     // column 2
    float4 modelC3;     // column 3
    float4 color;       // per-cluster hash color (rgb, a=1)
    float4 worldSphere; // xyz = world center, w = world radius (precomputed by InstanceWorldSphere)
    uint4  slice;       // x=firstIndex (triOffset*3), y=indexCount (triCount*3), z/w pad
};

// Compacted survivor per-draw record (matches cluster_viz.vert PerDraw: column-major mat4 + float4 color
// = 80 bytes, std430).
struct PerDrawOut {
    float4 modelC0;
    float4 modelC1;
    float4 modelC2;
    float4 modelC3;
    float4 color;
};

// Compacted survivor MDI command (5x u32 == VkDrawIndexedIndirectCommand). 20 bytes; std430 stride 20.
struct CommandOut {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    uint vertexOffset;
    uint firstInstance;
};

// Cull params (std430): 6 frustum planes (n.xyz, d) + counts(clusterCount, screenW, screenH, mipCount) +
// occlusion(occlusionEnabled, _, _, _) + the per-mip table (offset-in-floats, width, height, _).
struct Params {
    float4 planes[6];           // 6 frustum planes (n.xyz, d)
    uint4  counts;              // x=clusterCount, y=screenW, z=screenH, w=mipCount
    uint4  occlusion;           // x=occlusionEnabled (0/1), y/z/w pad
    uint4  mips[HF_MAX_MIPS];   // per-mip: x=offset(in floats), y=width, z=height, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<ClusterIn>  gClusters : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<PerDrawOut> gPerDraw  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<CommandOut> gCommands : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>       gCount    : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<Params>     gParams   : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<float4x4>   gViewProj : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<float>      gHiZ      : register(u6);

// Per-thread keep flag (0/1) + the inclusive prefix-sum scratch, shared across the workgroup.
groupshared uint gKeep[HF_MAX_CLUSTERS];

// Conservative bounding-sphere frustum cull (mirrors frustum.h::SphereOutside VERBATIM).
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

// The EXACT mirror of hf::render::hiz::IsOccluded. Builds the WORLD AABB from the cluster's bounding sphere
// (worldCenter +- worldRadius — the SAME conservative sphere-AABB the CPU mirror projects), projects the 8
// corners through the view-proj, forms the screen pixel rect + nearest NDC depth, picks the coarsest mip
// covering the rect in <=2x2 texels, and returns true (OCCLUDED) iff the nearest depth is FARTHER than the
// Hi-Z MAX across every covered texel. Conservative KEEP on any uncertainty.
bool IsOccluded(float3 worldCenter, float worldRadius, float4x4 vp, int screenW, int screenH, uint mipCount) {
    if (mipCount == 0u || screenW <= 0 || screenH <= 0) return false;

    float3 lmn = worldCenter - worldRadius;  // bounding-sphere AABB min (world)
    float3 lmx = worldCenter + worldRadius;  // bounding-sphere AABB max (world)

    float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f, nearestZ = 1e30f;
    [unroll] for (int c = 0; c < 8; ++c) {
        float3 wp = float3((c & 1) ? lmx.x : lmn.x,
                           (c & 2) ? lmx.y : lmn.y,
                           (c & 4) ? lmx.z : lmn.z);
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

[numthreads(HF_MAX_CLUSTERS, 1, 1)]
void main(uint3 lid : SV_GroupThreadID) {
    uint i = lid.x;                       // one workgroup: local index == global index
    uint clusterCount = gParams[0].counts.x;
    int  screenW = (int)gParams[0].counts.y;
    int  screenH = (int)gParams[0].counts.z;
    uint mipCount = gParams[0].counts.w;
    uint occlusionEnabled = gParams[0].occlusion.x;
    float4x4 vp = gViewProj[0];

    // 1) Frustum test the cluster's PRECOMPUTED world sphere; if it survives AND occlusion is enabled,
    //    ALSO Hi-Z-occlusion-test its bounding-sphere AABB. keep = passed frustum AND not occluded.
    uint keepFrustum = 0u;  // passed frustum (the frustumKept stat / disabled-path survivor)
    uint keep = 0u;         // passed frustum AND not occluded (the survivor)
    ClusterIn o;
    if (i < clusterCount) {
        o = gClusters[i];
        if (!SphereOutside(o.worldSphere.xyz, o.worldSphere.w)) {
            keepFrustum = 1u;
            keep = 1u;
            if (occlusionEnabled != 0u &&
                IsOccluded(o.worldSphere.xyz, o.worldSphere.w, vp, screenW, screenH, mipCount)) {
                keep = 0u;  // fully hidden -> dropped
            }
        }
    }
    gKeep[i] = keep;
    GroupMemoryBarrierWithGroupSync();

    // 2) ORDERED Hillis-Steele inclusive prefix sum over gKeep (1024 elements, power of two) -> survivors
    //    land in SOURCE-INDEX order (matches the CPU mirror byte-for-byte).
    [unroll] for (uint offset = 1; offset < HF_MAX_CLUSTERS; offset <<= 1) {
        uint v = gKeep[i];
        uint add = (i >= offset) ? gKeep[i - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gKeep[i] = v + add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint total = gKeep[HF_MAX_CLUSTERS - 1];  // inclusive scan: last element == total survivors

    // 3) Each surviving thread writes its compacted per-draw + MDI command to slot = exclusive prefix.
    if (i < clusterCount && keep == 1u) {
        uint dst = gKeep[i] - 1u;   // inclusive - 1 == number of earlier survivors (exclusive prefix)

        PerDrawOut pd;
        pd.modelC0 = o.modelC0; pd.modelC1 = o.modelC1;
        pd.modelC2 = o.modelC2; pd.modelC3 = o.modelC3;
        pd.color = o.color;
        gPerDraw[dst] = pd;

        CommandOut c;
        c.indexCount    = o.slice.y;     // triCount*3
        c.instanceCount = 1u;            // one draw per surviving cluster-instance
        c.firstIndex    = o.slice.x;     // triOffset*3
        c.vertexOffset  = 0u;
        c.firstInstance = i;             // SOURCE cluster-instance index (matches the CPU mirror)
        gCommands[dst] = c;
    }

    // 4) Thread 0 writes the GPU-decided survivor count; frustumKept (gCount[1]) + occluded (gCount[2]) are
    //    accumulated via atomics over the threads that passed frustum (diagnostics, not the draw count).
    if (i == 0u) gCount[0] = total;
    if (i < clusterCount && keepFrustum == 1u) {
        uint dummy;
        InterlockedAdd(gCount[1], 1u, dummy);                  // frustumKept++
        if (keep == 0u) InterlockedAdd(gCount[2], 1u, dummy);  // occluded (passed frustum, hidden)++
    }
}
