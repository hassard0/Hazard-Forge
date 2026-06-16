// Slice CD — Fully-GPU-driven-CULLED pass (compute-cull -> MDI + bindless). ONE workgroup of 1024
// threads (the scene is <=1024 objects). The COMPOSITION of the four proven slices on the GPU:
//   * AR (cull.comp.hlsl): each thread tests its object's world bounding sphere against the six frustum
//     planes (the SAME Gribb-Hartmann convention as engine/render/frustum.h), then the workgroup runs an
//     ORDERED single-workgroup prefix sum so survivors are written in SOURCE-INDEX order (NOT an
//     unordered atomicAdd append — that would be nondeterministic and break the image golden).
//   * BM/CB: the OUTPUT per-draw record is GpuDrivenPerDraw (model+material+texIndex, the 96-byte std430
//     layout lit_gpudriven.{vert,frag} read as PerDraw[gl_DrawID]); the OUTPUT command is the 5x-u32
//     MdiCommand. The compute COMPACTS the survivors into BOTH (survivor j -> perDraw[j] + command[j]).
//   * The survivor count is written into a separate drawCount buffer (thread 0); the host reads it back
//     and issues ONE DrawIndexedMultiIndirect(drawCount) — the GPU decided AND (via the compacted
//     buffers) prepared the draw. (We read the count back rather than vkCmdDrawIndexedIndirectCount so
//     the existing DrawIndexedMultiIndirect path consumes it — documented in the design; the byte-
//     identical + exact-count proofs are what matter.)
//
// CPU MIRROR: engine/render/gpu_culled.h implements the identical cull+compact; tests/gpu_culled_test.cpp
// pins the ordered-compaction + per-draw-carry + determinism contract, and the showcase asserts the GPU
// survivor count == the CPU frustum.h reference AND the GPU-culled image == the CPU-culled bound image.
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gObjects  : source per-object records (model+material+localSphere+slice+texIndex), READ.
//   b1 gPerDraw  : compacted survivor per-draw records (model+material+texIndex), WRITE.
//   b2 gCommands : compacted survivor MDI commands (5x u32), WRITE.
//   b3 gCount    : survivor count (1x u32 at [0]), WRITE (thread 0).
//   b4 gParams   : 6 frustum planes + object count, READ.

#define HF_MAX_OBJECTS 1024

// Source per-object record (std430). Matches the C++ upload struct (engine/render gpu-culled showcase)
// EXACTLY: mat4 (4 float4) + material (float4) + localSphere (xyz=center, w=radius) + (indexCount,
// firstIndex, vertexOffset, texIndex) = 7 float4 = 112 bytes.
struct ObjectIn {
    float4 modelC0;     // column 0
    float4 modelC1;     // column 1
    float4 modelC2;     // column 2
    float4 modelC3;     // column 3
    float4 material;    // x=metallic, y=roughness-tint, ...
    float4 localSphere; // xyz = local center, w = local radius
    uint4  slice;       // x=indexCount, y=firstIndex, z=vertexOffset, w=texIndex
};

// Compacted survivor per-draw record (matches render::gpudriven::GpuDrivenPerDraw / lit_gpudriven.vert
// PerDraw byte layout EXACTLY: column-major mat4 + float4 material + uint texIndex + 3x uint pad = 96
// bytes, std430).
struct PerDrawOut {
    float4 modelC0;
    float4 modelC1;
    float4 modelC2;
    float4 modelC3;
    float4 material;
    uint   texIndex;
    uint3  pad;
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
    uint4  counts;      // x = objectCount, y/z/w = pad
};

[[vk::binding(0, 0)]] RWStructuredBuffer<ObjectIn>   gObjects  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<PerDrawOut> gPerDraw  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<CommandOut> gCommands : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>       gCount    : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<Params>     gParams   : register(u4);

// Per-thread keep flag (0/1) + the inclusive prefix-sum scratch, shared across the workgroup.
groupshared uint gKeep[HF_MAX_OBJECTS];

// Conservative bounding-sphere cull: culled IFF the sphere is fully outside at least one plane
// (signedDistance(center) < -radius). Mirrors frustum.h::SphereOutside exactly.
bool SphereOutside(float3 center, float radius) {
    [unroll] for (int i = 0; i < 6; ++i) {
        float4 pl = gParams[0].planes[i];
        if (dot(pl.xyz, center) + pl.w < -radius) return true;
    }
    return false;
}

[numthreads(HF_MAX_OBJECTS, 1, 1)]
void main(uint3 lid : SV_GroupThreadID) {
    uint i = lid.x;                       // one workgroup: local index == global index
    uint objectCount = gParams[0].counts.x;

    // 1) Test this object's world sphere. center = model * float4(localCenter,1); radius =
    //    localRadius * |col0| (uniform scale). The SAME math as render::gpu_cull::InstanceWorldSphere.
    uint keep = 0;
    ObjectIn o;
    if (i < objectCount) {
        o = gObjects[i];
        float3 lc = o.localSphere.xyz;
        float  lr = o.localSphere.w;
        float3 center = (o.modelC0.xyz * lc.x) + (o.modelC1.xyz * lc.y) +
                        (o.modelC2.xyz * lc.z) + o.modelC3.xyz;
        float  radius = lr * length(o.modelC0.xyz);
        keep = SphereOutside(center, radius) ? 0u : 1u;
    }
    gKeep[i] = keep;
    GroupMemoryBarrierWithGroupSync();

    // 2) ORDERED Hillis-Steele inclusive prefix sum over gKeep (1024 elements, power of two). After this
    //    gKeep[i] holds the count of survivors at indices [0..i]; the exclusive prefix (inclusive-keep)
    //    is the survivor's destination slot -> survivors land in SOURCE-INDEX order.
    [unroll] for (uint offset = 1; offset < HF_MAX_OBJECTS; offset <<= 1) {
        uint v = gKeep[i];
        uint add = (i >= offset) ? gKeep[i - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gKeep[i] = v + add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint total = gKeep[HF_MAX_OBJECTS - 1];  // inclusive scan: last element == total survivors

    // 3) Each surviving thread writes its compacted per-draw + MDI command to slot = exclusive prefix.
    if (i < objectCount && keep == 1u) {
        uint dst = gKeep[i] - 1u;   // inclusive - 1 == number of earlier survivors (exclusive prefix)

        PerDrawOut pd;
        pd.modelC0 = o.modelC0; pd.modelC1 = o.modelC1;
        pd.modelC2 = o.modelC2; pd.modelC3 = o.modelC3;
        pd.material = o.material;
        pd.texIndex = o.slice.w;
        pd.pad = uint3(0, 0, 0);
        gPerDraw[dst] = pd;

        CommandOut c;
        c.indexCount    = o.slice.x;
        c.instanceCount = 1u;            // one draw per survivor
        c.firstIndex    = o.slice.y;
        c.vertexOffset  = o.slice.z;
        c.firstInstance = 0u;            // per-draw data indexed by gl_DrawID, not firstInstance
        gCommands[dst] = c;
    }

    // 4) Thread 0 writes the GPU-decided survivor count (the host reads it back for the drawCount).
    if (i == 0u) gCount[0] = total;
}
