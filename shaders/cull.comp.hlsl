// Slice AR — GPU-driven frustum culling + indirect-draw compaction. ONE workgroup of 1024 threads
// (the grid is <=1024 instances). Each thread tests its instance's world bounding sphere against the
// six frustum planes (the SAME Gribb-Hartmann convention as engine/render/frustum.h), then the
// workgroup runs an ORDERED prefix sum so survivors are written to gSurvivors in SOURCE-INDEX order
// (NOT an unordered atomicAdd append — that would be nondeterministic and break the image golden).
// Thread 0 writes the indirect draw-args (instanceCount = total survivors). A single
// DrawIndexedIndirect then renders exactly the survivors, the count decided entirely on the GPU.
//
// CPU MIRROR: engine/render/gpu_cull.h implements the identical logic; tests/gpu_cull_test.cpp pins
// the ordered-compaction contract, and the showcases assert the GPU count == that CPU reference.
//
// Buffers (storage, bound at compute bindings 0..3; on Metal these land at buffer(0..3) via
// spirv-cross --msl-decoration-binding, matching kCsStorage + index):
//   b0 gInstances : source per-instance model matrices (float4x4 each), READ.
//   b1 gSurvivors : compacted survivor model matrices, WRITE (consumed as the per-instance stream).
//   b2 gArgs      : indirect args {indexCount, instanceCount, firstIndex, vertexOffset,
//                   firstInstance} (5x u32), WRITE (thread 0 sets instanceCount).
//   b3 gParams    : frustum planes + instance/index counts + local sphere, READ.

struct Mat4 { float4 c0; float4 c1; float4 c2; float4 c3; };  // four COLUMNS (column-major)

// Frustum params + draw constants. Six planes as float4 {n.xyz, d} (unit n, inside = dot(n,p)+d>=0).
// localCenter/localRadius is the shared unit-mesh bound; instanceCount/indexCount drive the loop +
// the indirect args. Laid out to match the CPU upload in the showcases (std430-friendly).
struct Params {
    float4 planes[6];      // 6 frustum planes (n.xyz, d)
    float4 localCenter;    // xyz = local sphere center, w = local radius
    uint4  counts;         // x = instanceCount, y = indexCount, z/w = pad
};

[[vk::binding(0, 0)]] RWStructuredBuffer<Mat4> gInstances : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<Mat4> gSurvivors : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> gArgs      : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<Params> gParams  : register(u3);

#define HF_MAX_INSTANCES 1024

// Per-thread keep flag (0/1) and the inclusive prefix-sum scratch, shared across the workgroup.
groupshared uint gKeep[HF_MAX_INSTANCES];

// Conservative bounding-sphere cull: culled IFF the sphere is fully outside at least one plane
// (signedDistance(center) < -radius). Mirrors frustum.h::SphereOutside exactly.
bool SphereOutside(float3 center, float radius) {
    [unroll] for (int i = 0; i < 6; ++i) {
        float4 pl = gParams[0].planes[i];
        float sd = dot(pl.xyz, center) + pl.w;
        if (sd < -radius) return true;
    }
    return false;
}

[numthreads(HF_MAX_INSTANCES, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID) {
    uint i = lid.x;                       // one workgroup: local index == global index
    uint instanceCount = gParams[0].counts.x;
    uint indexCount    = gParams[0].counts.y;

    // 1) Test this instance's world sphere. center = model * localCenter; radius = localRadius *
    //    |col0| (uniform scale). Same as the CPU mirror.
    uint keep = 0;
    Mat4 m;
    if (i < instanceCount) {
        m = gInstances[i];
        float3 lc = gParams[0].localCenter.xyz;
        float  lr = gParams[0].localCenter.w;
        // model * float4(lc, 1): combine the four columns.
        float3 center = (m.c0.xyz * lc.x) + (m.c1.xyz * lc.y) + (m.c2.xyz * lc.z) + m.c3.xyz;
        float  col0Len = length(m.c0.xyz);
        float  radius = lr * col0Len;
        keep = SphereOutside(center, radius) ? 0u : 1u;
    }
    gKeep[i] = keep;
    GroupMemoryBarrierWithGroupSync();

    // 2) ORDERED Hillis-Steele inclusive prefix sum over gKeep (1024 elements, power of two). After
    //    this gKeep[i] holds the count of survivors at indices [0..i]. The exclusive prefix
    //    (inclusive - keep) is the survivor's destination slot -> survivors land in source order.
    [unroll] for (uint offset = 1; offset < HF_MAX_INSTANCES; offset <<= 1) {
        uint v = gKeep[i];
        uint add = (i >= offset) ? gKeep[i - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gKeep[i] = v + add;
        GroupMemoryBarrierWithGroupSync();
    }

    uint total = gKeep[HF_MAX_INSTANCES - 1];  // inclusive scan: last element == total survivors

    // 3) Each surviving thread writes its model to the compacted slot = exclusive prefix.
    if (i < instanceCount && keep == 1u) {
        uint dst = gKeep[i] - 1u;   // inclusive - 1 == number of earlier survivors (exclusive prefix)
        gSurvivors[dst] = m;
    }

    // 4) Thread 0 writes the indirect draw-args: instanceCount = survivor total, rest constant.
    if (i == 0u) {
        gArgs[0] = indexCount;   // indexCount
        gArgs[1] = total;        // instanceCount  <-- the GPU-decided survivor count
        gArgs[2] = 0u;           // firstIndex
        gArgs[3] = 0u;           // vertexOffset
        gArgs[4] = 0u;           // firstInstance
    }
}
