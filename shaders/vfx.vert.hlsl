// Slice CC — CPU particle / VFX emitter billboard vertex shader. The billboard quads are built
// CPU-side (engine/vfx/particles.cpp BuildBillboards) directly in WORLD space as camera-facing quads
// (already spanning cameraRight/cameraUp), so there is NO model matrix here: project the world
// position by FrameData.viewProj and pass the corner UV + per-particle color through. Pairs with an
// ADDITIVE-blend, no-depth-write pipeline (the VFX pass draws over the opaque scene). DISTINCT from
// the fixed gpu-particles fountain (shaders/particle.vert.hlsl); this is the authorable emitter layer.
//
// Vertex layout (matches vfx::BillboardVertex, 36 bytes): pos RGB32F @0, uv RG32F @12, color RGBA32F @20.
struct VSInput {
    [[vk::location(0)]] float3 pos   : POSITION;
    [[vk::location(1)]] float2 uv    : TEXCOORD0;
    [[vk::location(2)]] float4 color : COLOR;
};
struct VSOutput {
    float4 pos   : SV_Position;
    [[vk::location(0)]] float2 uv    : TEXCOORD0;
    [[vk::location(1)]] float4 color : COLOR;
};
#include "frame_data.hlsli"
// See shaders/lit.vert.hlsl for the HF_MSL_GEN rationale. No push constant, so on the generated-MSL
// path the FrameData cbuffer lands at binding(1,0) (the engine's flat Metal frame-uniform slot); on
// the Vulkan/DXC path it is set 0 binding 0 (the per-frame UBO).
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    o.pos   = mul(f.viewProj, float4(i.pos, 1.0));
    o.uv    = i.uv;
    o.color = i.color;
    return o;
}
