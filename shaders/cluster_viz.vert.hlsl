// Slice DT — virtual-geometry CLUSTER cull MDI vertex shader. VULKAN-ONLY (uses the SPIR-V DrawIndex
// builtin). The MDI analogue of meshlet_viz.vert: instead of a per-cluster model+color PUSH constant, it
// reads the per-draw (model + hashColor) from a STORAGE buffer PerDraw[ ] indexed by gl_DrawID
// ([[vk::builtin("DrawIndex")]]) — the compacted survivor record shaders/cluster_cull.comp.hlsl wrote.
// Forwards the per-cluster flat color modulated by a FIXED-direction Lambert term (IDENTICAL to
// meshlet_viz.vert) for shape readability, so the fragment shader is the SHARED meshlet_viz.frag flat-
// color pass (no GPU-side hash — the color is decided CPU-side, bit-exact). Fully deterministic.
//
// NOT in the Metal MSL-gen list: glslc/spirv-cross cannot lower the DrawIndex builtin; the Metal golden
// renders the identical scene via its per-cluster-instance BOUND path (the meshlet_viz push-constant
// shader), so the image is backend-identical (the camera composes the Metal-clip flip).
struct VSInput {
    [[vk::location(0)]] float3 pos    : POSITION;
    [[vk::location(3)]] float3 normal : NORMAL;
};
struct VSOutput {
    float4 clip : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;  // per-cluster flat color * fixed Lambert (meshlet_viz.frag)
};
#include "frame_data.hlsli"

// Per-draw record (matches the cluster_cull.comp.hlsl PerDrawOut byte layout exactly: column-major mat4 +
// float4 color = 80 bytes, std430). HLSL reads the four COLUMNS into a column-major model so mul(model, v)
// matches the meshlet_viz push-constant per-cluster path bit-for-bit.
struct PerDraw {
    float4 modelCol0;  // column 0
    float4 modelCol1;  // column 1
    float4 modelCol2;  // column 2
    float4 modelCol3;  // column 3
    float4 color;      // per-cluster hash color (rgb, a=1)
};

// Frame UBO at set 0 (binding 0); the per-draw SSBO at set 2 (binding 0) — the dedicated VERTEX-stage
// storage set bound via BindPerDrawData (the SAME surface lit_gpudriven.vert uses).
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(0, 2)]] StructuredBuffer<PerDraw> gPerDraw;

VSOutput main(VSInput i, [[vk::builtin("DrawIndex")]] uint drawId : DRAWID) {
    VSOutput o;
    PerDraw pd = gPerDraw[drawId];
    // Reassemble the column-major model matrix from the four per-draw columns (HLSL float4x4(rows...)
    // takes ROWS, so row r is the r-th component of each column). Identical to lit_gpudriven.vert.
    float4x4 model = float4x4(
        float4(pd.modelCol0.x, pd.modelCol1.x, pd.modelCol2.x, pd.modelCol3.x),
        float4(pd.modelCol0.y, pd.modelCol1.y, pd.modelCol2.y, pd.modelCol3.y),
        float4(pd.modelCol0.z, pd.modelCol1.z, pd.modelCol2.z, pd.modelCol3.z),
        float4(pd.modelCol0.w, pd.modelCol1.w, pd.modelCol2.w, pd.modelCol3.w));

    float4 world = mul(model, float4(i.pos, 1.0));
    o.clip = mul(f.viewProj, world);
    // Fixed-direction Lambert (a constant key direction, NOT the scene light) so spheres read as 3D
    // shapes — IDENTICAL to meshlet_viz.vert (range [0.35,1]). Deterministic + backend-agnostic.
    float3 N = normalize(mul((float3x3)model, i.normal));
    float3 keyDir = normalize(float3(0.4, 0.7, 0.6));
    float lambert = saturate(dot(N, keyDir)) * 0.65 + 0.35;
    o.color = pd.color.xyz * lambert;
    return o;
}
