// Slice DT — Virtual-Geometry Slice 2: GPU per-cluster frustum cull -> indirect cluster draw. A NEAR-COPY
// of shaders/gpudriven_cull.comp.hlsl with the CLUSTER-INSTANCE as the unit (a DS cluster is just an
// object with a per-cluster bounding sphere). ONE workgroup of 1024 threads (the (instance x cluster)
// count is <=1024). One thread per cluster-instance:
//   * each thread tests its cluster-instance's WORLD bounding sphere against the six frustum planes (the
//     SAME Gribb-Hartmann convention as engine/render/frustum.h — SphereOutside copied VERBATIM), then
//     the workgroup runs an ORDERED single-workgroup prefix sum so survivors are written in SOURCE-INDEX
//     order (NOT an unordered atomicAdd append — that would be nondeterministic and break the golden).
//   * the OUTPUT command is the 5x-u32 MdiCommand { indexCount = triCount*3, instanceCount = 1,
//     firstIndex = triOffset*3, vertexOffset = 0, firstInstance = SOURCE cluster-instance index }; the
//     OUTPUT per-draw record is the compacted (model + hashColor) the cluster_viz vertex shader reads as
//     PerDraw[gl_DrawID]. The compute COMPACTS the survivors into BOTH (survivor j -> perDraw[j] +
//     command[j], source order).
//   * the survivor count is written into a separate drawCount buffer (thread 0); the host reads it back
//     and issues ONE DrawIndexedMultiIndirect(drawCount) — the GPU decided AND prepared the draw.
//
// CPU MIRROR: engine/render/cluster_cull.h (BuildClusterInstances + CullClusterInstances) implements the
// identical cull+compact; tests/cluster_cull_test.cpp pins the ordered-compaction + determinism contract,
// and the showcase asserts the GPU survivor count == the CPU frustum.h reference AND the GPU-culled image
// == the CPU-culled bound image. NO new RHI (reuses the ComputePipelineDesc/BindStorageBuffer/Dispatch/
// ComputeToVertexBarrier/ReadBuffer/DrawIndexedMultiIndirect surface). Only [[vk::binding]]/HF_MSL_GEN
// above the seam.
//
// Buffers (storage, bound at compute bindings 0..3):
//   b0 gClusters : source per-cluster-instance records (model + color + worldSphere + slice), READ.
//   b1 gPerDraw  : compacted survivor per-draw records (model + color), WRITE.
//   b2 gCommands : compacted survivor MDI commands (5x u32), WRITE.
//   b3 gCount    : survivor count (1x u32 at [0]), WRITE (thread 0).
//   b4 gParams   : 6 frustum planes + cluster-instance count, READ.

#define HF_MAX_CLUSTERS 1024

// Source per-cluster-instance record (std430). Matches the C++ upload struct (the cluster-cull showcase)
// EXACTLY: mat4 (4 float4) + color (float4) + worldSphere (xyz=center, w=radius) + (firstIndex, indexCount,
// _, _) = 7 float4 = 112 bytes. The world sphere is precomputed CPU-side (InstanceWorldSphere) so the
// shader does the IDENTICAL frustum test the CPU mirror does — bit-for-bit on the cull decision.
struct ClusterIn {
    float4 modelC0;     // column 0
    float4 modelC1;     // column 1
    float4 modelC2;     // column 2
    float4 modelC3;     // column 3
    float4 color;       // per-cluster hash color (rgb, a=1)
    float4 worldSphere; // xyz = world center, w = world radius (precomputed by InstanceWorldSphere)
    uint4  slice;       // x=firstIndex (triOffset*3), y=indexCount (triCount*3), z/w pad
};

// Compacted survivor per-draw record (matches the cluster_viz.vert PerDraw byte layout: column-major mat4
// + float4 color = 80 bytes, std430).
struct PerDrawOut {
    float4 modelC0;
    float4 modelC1;
    float4 modelC2;
    float4 modelC3;
    float4 color;
};

// Compacted survivor MDI command (5x u32 == VkDrawIndexedIndirectCommand). 20 bytes; std430 packs the
// array tightly (stride 20), matching render::mdi::MdiCommand + the host's indirect buffer.
struct CommandOut {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    uint vertexOffset;
    uint firstInstance;
};

// Cull params (std430): 6 frustum planes (n.xyz, d; unit n, inside = dot(n,p)+d >= 0) + counts.
struct Params {
    float4 planes[6];   // 6 frustum planes (n.xyz, d)
    uint4  counts;      // x = clusterInstanceCount, y/z/w = pad
};

[[vk::binding(0, 0)]] RWStructuredBuffer<ClusterIn>  gClusters : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<PerDrawOut> gPerDraw  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<CommandOut> gCommands : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>       gCount    : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<Params>     gParams   : register(u4);

// Per-thread keep flag (0/1) + the inclusive prefix-sum scratch, shared across the workgroup.
groupshared uint gKeep[HF_MAX_CLUSTERS];

// Conservative bounding-sphere cull: culled IFF the sphere is fully outside at least one plane
// (signedDistance(center) < -radius). Mirrors frustum.h::SphereOutside VERBATIM.
bool SphereOutside(float3 center, float radius) {
    [unroll] for (int i = 0; i < 6; ++i) {
        float4 pl = gParams[0].planes[i];
        if (dot(pl.xyz, center) + pl.w < -radius) return true;
    }
    return false;
}

[numthreads(HF_MAX_CLUSTERS, 1, 1)]
void main(uint3 lid : SV_GroupThreadID) {
    uint i = lid.x;                       // one workgroup: local index == global index
    uint clusterCount = gParams[0].counts.x;

    // 1) Test this cluster-instance's PRECOMPUTED world sphere against the six planes. The world sphere is
    //    InstanceWorldSphere(instanceModel, boundCenter, boundRadius) computed CPU-side (the SAME values
    //    the CPU mirror tests), so the cull decision is bit-identical to render::frustum::SphereOutside.
    uint keep = 0;
    ClusterIn o;
    if (i < clusterCount) {
        o = gClusters[i];
        keep = SphereOutside(o.worldSphere.xyz, o.worldSphere.w) ? 0u : 1u;
    }
    gKeep[i] = keep;
    GroupMemoryBarrierWithGroupSync();

    // 2) ORDERED Hillis-Steele inclusive prefix sum over gKeep (1024 elements, power of two). After this
    //    gKeep[i] holds the count of survivors at indices [0..i]; the exclusive prefix (inclusive-keep) is
    //    the survivor's destination slot -> survivors land in SOURCE-INDEX order (matches the CPU mirror).
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

    // 4) Thread 0 writes the GPU-decided survivor count (the host reads it back for the drawCount).
    if (i == 0u) gCount[0] = total;
}
