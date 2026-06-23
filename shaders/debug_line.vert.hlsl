// Slice W — immediate-mode debug-line vertex shader. Vertices are WORLD-space (the DebugDraw
// collector emits world positions), so there is NO model matrix: just project pos by
// FrameData.viewProj and pass the per-vertex color through. Pairs with a LINE_LIST pipeline
// (GraphicsPipelineDesc.lineList = true).
struct VSInput {
    [[vk::location(0)]] float3 pos   : POSITION;
    [[vk::location(1)]] float3 color : COLOR;
};
struct VSOutput {
    float4 pos   : SV_Position;
    float3 color : COLOR;
};
#include "frame_data.hlsli"
// See shaders/lit.vert.hlsl for the HF_MSL_GEN rationale. This shader has no push constant, so on
// the generated-MSL path the FrameData cbuffer lands at binding(1,0) (the engine's flat Metal
// frame-uniform slot); on the Vulkan/DXC path it is set 0 binding 0 (the per-frame UBO).
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    o.pos = mul(f.viewProj, float4(i.pos, 1.0));
    o.color = i.color;
    return o;
}
