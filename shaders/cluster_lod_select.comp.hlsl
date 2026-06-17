// Slice DV — Virtual-Geometry Slice 4: discrete cluster-LOD selection by projected SCREEN-SPACE ERROR.
// ONE thread per INSTANCE (the instance count is <=1024 = one workgroup). Each thread:
//   * computes its instance's SQUARED view distance + runs SelectLod (the squared-distance form, copied
//     VERBATIM from engine/render/cluster_lod.h) to pick one of kNumLods=3 pre-baked LODs,
//   * runs an ORDERED single-workgroup prefix sum over the per-instance selected-LOD CLUSTER COUNTS so
//     each instance's emitted MdiCommands land in INSTANCE-major / cluster-minor SOURCE order (NOT an
//     unordered atomicAdd append — that would be nondeterministic and break the golden),
//   * emits one MdiCommand per cluster of its selected LOD { indexCount=triCount*3, instanceCount=1,
//     firstIndex=triOffset*3, vertexOffset=0, firstInstance = the INSTANCE index } and the matching
//     compacted per-draw (model + hashColor) record the cluster_viz vertex shader reads as
//     PerDraw[gl_DrawID],
//   * writes its selected-LOD INTEGER to a readback buffer (the GPU==CPU bit-exact proof) and thread 0
//     writes the total emitted command count (the host reads it back for the DrawIndexedMultiIndirect).
//
// CPU MIRROR: engine/render/cluster_lod.h (SelectLod, the squared form). tests/cluster_lod_test.cpp pins
// it; the --cluster-lod-shot showcase asserts the GPU per-instance selected-LOD ints == the CPU SelectLod
// ints (memcmp, BIT-EXACT) AND forceLod0 == the full-detail render byte-identical. NO new RHI (reuses the
// ComputePipelineDesc/BindStorageBuffer/Dispatch/ComputeToVertexBarrier/ReadBuffer/DrawIndexedMultiIndirect
// surface, exactly like cluster_cull.comp). Only [[vk::binding]] above the seam.
//
// Buffers (storage, bound at compute bindings 0..6):
//   b0 gInstances : per-instance records (model + color + worldCenter + the kNumLods cluster ranges), READ.
//   b1 gClusterSrc: the per-cluster (triOffset*3, triCount*3) slices of the COMBINED LOD mesh, READ.
//   b2 gPerDraw   : compacted emitted per-draw records (model + color), WRITE.
//   b3 gCommands  : compacted emitted MDI commands (5x u32), WRITE.
//   b4 gCount     : total emitted command count (1x u32 at [0]), WRITE (thread 0).
//   b5 gParams    : view matrix + projScale + errorThreshold + errorScale + forceLod0 + instanceCount + the
//                   kNumLods geometric errors, READ.
//   b6 gSelLod    : per-instance selected-LOD integer (the GPU==CPU bit-exact proof), WRITE.

#define HF_MAX_INSTANCES 1024
#define HF_NUM_LODS 3

// Per-instance source record (std430). Matches the C++ upload struct (the cluster-lod showcase) EXACTLY:
// mat4 (4 float4) + color (float4) + worldCenter (float4, xyz used) + lodRange[3] (uint4: x=firstCluster,
// y=clusterCount, z/w pad). 4+1+1+3 = 9 float4 = 144 bytes.
struct InstanceIn {
    float4 modelC0;     // column 0
    float4 modelC1;     // column 1
    float4 modelC2;     // column 2
    float4 modelC3;     // column 3
    float4 color;       // per-instance/per-cluster base color (rgb, a=1); per-cluster hash applied below
    float4 worldCenter; // xyz = world-space instance center (for the view distance), w pad
    uint4  lodRange[HF_NUM_LODS]; // [n] = (firstCluster, clusterCount, _, _) of LOD n in the combined set
};

// Per-cluster slice of the COMBINED LOD mesh (std430): x = firstIndex (triOffset*3), y = indexCount
// (triCount*3), z/w pad. One per cluster across ALL LODs (the BuildLodMeshes combined.meshlets order).
struct ClusterSlice {
    uint4 slice;        // x=firstIndex, y=indexCount, z=hashColorKey (cluster-in-lod index), w pad
};

// Compacted emitted per-draw record (matches cluster_viz.vert PerDraw: column-major mat4 + float4 color =
// 80 bytes, std430).
struct PerDrawOut {
    float4 modelC0;
    float4 modelC1;
    float4 modelC2;
    float4 modelC3;
    float4 color;
};

// Compacted emitted MDI command (5x u32 == VkDrawIndexedIndirectCommand). 20 bytes; std430 stride 20.
struct CommandOut {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    uint vertexOffset;
    uint firstInstance;
};

// LOD-select params (std430): the column-major view matrix (4 float4) + (projScale, errorThreshold,
// errorScale, forceLod0) + (instanceCount, _, _, _) + the kNumLods geometric errors packed into a float4
// (x=err0, y=err1, z=err2, w pad).
struct Params {
    float4 viewC0; float4 viewC1; float4 viewC2; float4 viewC3;
    float4 lodParams;     // x=projScale, y=errorThreshold, z=errorScale, w=forceLod0 (0/1)
    uint4  counts;        // x=instanceCount, y/z/w pad
    float4 geomError;     // x=err0(=0), y=err1, z=err2, w pad
};

[[vk::binding(0, 0)]] RWStructuredBuffer<InstanceIn>   gInstances  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<ClusterSlice> gClusterSrc : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<PerDrawOut>   gPerDraw    : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<CommandOut>   gCommands   : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>         gCount      : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<Params>       gParams     : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<uint>         gSelLod     : register(u6);

// Per-thread emitted-cluster count + the inclusive prefix-sum scratch, shared across the workgroup.
groupshared uint gEmit[HF_MAX_INSTANCES];

// Deterministic integer hash of a cluster index -> [0,1] RGB. Copied VERBATIM from meshlet.h::hashColor
// (the SAME multiply-xor-shift avalanche) so the per-cluster color matches the CPU bound reference exactly.
float3 HashColor(uint i) {
    uint h = i * 2654435761u + 0x9E3779B9u;
    h ^= h >> 15; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
    uint r = (h)       & 0xFFu;
    uint g = (h >> 8)  & 0xFFu;
    uint b = (h >> 16) & 0xFFu;
    return float3(r / 255.0, g / 255.0, b / 255.0);
}

// SQUARED view distance of `c` under the column-major view matrix in gParams. std::fma <-> mad: the same
// accumulation order as engine/render/cluster_lod.h::ViewDistanceSquared. Pure, no sqrt.
float ViewDistanceSquared(float3 c) {
    Params p = gParams[0];
    // column-major: view*[c,1] component r = viewCk[r] dotted with [c,1].
    float vx = mad(p.viewC0.x, c.x, mad(p.viewC1.x, c.y, mad(p.viewC2.x, c.z, p.viewC3.x)));
    float vy = mad(p.viewC0.y, c.x, mad(p.viewC1.y, c.y, mad(p.viewC2.y, c.z, p.viewC3.y)));
    float vz = mad(p.viewC0.z, c.x, mad(p.viewC1.z, c.y, mad(p.viewC2.z, c.z, p.viewC3.z)));
    return mad(vx, vx, mad(vy, vy, vz * vz));
}

// SelectLod — the squared-distance form, copied VERBATIM from engine/render/cluster_lod.h (same compare
// direction, same fma/mad accumulation, same upgrade scan). Returns the coarsest acceptable LOD in [0,2].
uint SelectLod(float3 worldCenter) {
    Params p = gParams[0];
    float projScale      = p.lodParams.x;
    float errorThreshold = p.lodParams.y;
    float errorScale     = p.lodParams.z;
    bool  forceLod0      = (p.lodParams.w != 0.0);

    if (forceLod0 || errorScale == 0.0) return 0u;  // disabled path: always LOD0

    float dist2  = ViewDistanceSquared(worldCenter);
    float allowed = errorThreshold * errorScale;       // screen-error budget (>= 0)
    float rhs     = mad(allowed * allowed, dist2, 0.0); // (allowed)^2 * dist2

    float geomErr[HF_NUM_LODS];
    geomErr[0] = p.geomError.x; geomErr[1] = p.geomError.y; geomErr[2] = p.geomError.z;

    uint lod = 0u;
    [unroll] for (uint n = 1; n < HF_NUM_LODS; ++n) {
        float ge  = geomErr[n] * projScale;
        float lhs = ge * ge;
        if (lhs <= rhs) lod = n;
    }
    if (lod >= HF_NUM_LODS) lod = HF_NUM_LODS - 1u;
    return lod;
}

[numthreads(HF_MAX_INSTANCES, 1, 1)]
void main(uint3 lid : SV_GroupThreadID) {
    uint i = lid.x;                          // one workgroup: local index == instance index
    uint instanceCount = gParams[0].counts.x;

    // 1) Select this instance's LOD + record it (the GPU==CPU bit-exact proof) + its emitted cluster count.
    uint sel = 0u;
    uint emit = 0u;
    InstanceIn inst;
    if (i < instanceCount) {
        inst = gInstances[i];
        sel  = SelectLod(inst.worldCenter.xyz);
        gSelLod[i] = sel;
        emit = inst.lodRange[sel].y;         // clusterCount of the selected LOD
    }
    gEmit[i] = emit;
    GroupMemoryBarrierWithGroupSync();

    // 2) ORDERED Hillis-Steele inclusive prefix sum over gEmit (1024 elements). After this gEmit[i] holds
    //    the count of emitted commands at instances [0..i]; the exclusive prefix is this instance's output
    //    base -> commands land in INSTANCE-major / cluster-minor SOURCE order.
    [unroll] for (uint offset = 1; offset < HF_MAX_INSTANCES; offset <<= 1) {
        uint v = gEmit[i];
        uint add = (i >= offset) ? gEmit[i - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gEmit[i] = v + add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint total = gEmit[HF_MAX_INSTANCES - 1];  // inclusive scan: last element == total emitted commands

    // 3) Each instance writes its selected LOD's clusters into the compacted buffers at base = exclusive
    //    prefix, cluster-minor order. Reassemble the model columns once.
    if (i < instanceCount && emit > 0u) {
        uint base = gEmit[i] - emit;         // inclusive - own count == exclusive prefix
        uint firstCluster = inst.lodRange[sel].x;
        for (uint k = 0u; k < emit; ++k) {
            uint dst = base + k;
            ClusterSlice cs = gClusterSrc[firstCluster + k];

            PerDrawOut pd;
            pd.modelC0 = inst.modelC0; pd.modelC1 = inst.modelC1;
            pd.modelC2 = inst.modelC2; pd.modelC3 = inst.modelC3;
            pd.color = float4(HashColor(cs.slice.z), 1.0);  // per-cluster hash (cluster-in-lod index)
            gPerDraw[dst] = pd;

            CommandOut c;
            c.indexCount    = cs.slice.y;    // triCount*3
            c.instanceCount = 1u;
            c.firstIndex    = cs.slice.x;    // triOffset*3 (into the combined LOD index buffer)
            c.vertexOffset  = 0u;
            c.firstInstance = i;             // the INSTANCE index (the per-draw fetch key)
            gCommands[dst] = c;
        }
    }

    // 4) Thread 0 writes the total emitted command count (the host reads it back for the drawCount).
    if (i == 0u) gCount[0] = total;
}
