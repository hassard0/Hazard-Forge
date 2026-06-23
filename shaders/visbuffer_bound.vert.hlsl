// Slice DW — visibility-buffer vertex shader, the BOUND (per-cluster push-constant) variant. This is the
// Metal path's analogue of visbuffer.vert: instead of the SPIR-V DrawIndex builtin (which glslc/spirv-
// cross cannot lower — exactly like cluster_viz.vert / meshlet_viz.vert's MDI variant), the host issues
// ONE DrawIndexed per survivor and PUSHES that survivor's clusterID (the survivor draw index) + its model
// matrix as a constant. The shader transforms the vertex and forwards the flat clusterID; the SHARED
// visbuffer.frag then packs (clusterID << 7) | (SV_PrimitiveID & 0x7F). The rasterized geometry + the
// written integer IDs are identical to the Vulkan MDI path (the survivor draw index is the same).
//
// triID = SV_PrimitiveID, in visbuffer.frag. On Metal spirv-cross lowers SV_PrimitiveID to [[primitive_id]]
// (valid in an MSL fragment function); the controller verifies the MSL on the real Mac. If a toolchain
// can't lower it, the fragment falls back to packing clusterID-only (triID deferred to the DX resolve).
struct VSInput {
    [[vk::location(0)]] float3 pos : POSITION;  // only position is needed (no shading -> no normal)
};
struct VSOutput {
    float4 clip : SV_Position;
    [[vk::location(0)]] nointerpolation uint clusterID : CLUSTERID;
};
#include "frame_data.hlsli"

// Same HF_MSL_GEN split as meshlet_viz.vert: glslang lowers [[vk::push_constant]] to a Uniform colliding
// with Frame, so for the MSL-gen path the push constant is an explicit cbuffer at distinct bindings
// (Frame=binding1, PushC=binding2) that spirv-cross maps to the engine's flat Metal buffer indices.
// Push constant: { float4x4 model; uint clusterID; } (clusterID is the survivor draw index, packed by the
// fragment). A float4 carries the uint in .x as a bit-cast so the 80-byte layout matches the bound path.
struct PushLayout { float4x4 model; uint clusterID; uint pad0; uint pad1; uint pad2; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 model; uint clusterID; uint pad0; uint pad1; uint pad2; };
#define HF_MODEL model
#define HF_CLUSTERID clusterID
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] PushLayout pc;
#define HF_MODEL pc.model
#define HF_CLUSTERID pc.clusterID
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    float4 world = mul(HF_MODEL, float4(i.pos, 1.0));
    o.clip = mul(f.viewProj, world);
    o.clusterID = HF_CLUSTERID;  // the survivor draw index = the clusterID the visibility buffer packs
    return o;
}
