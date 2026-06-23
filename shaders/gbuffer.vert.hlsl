// SSAO normal+depth prepass — STATIC vertex shader (Slice Y). Renders scene geometry and passes the
// VIEW-SPACE normal and VIEW-SPACE position to the fragment stage, which writes view-space normal +
// linear depth into an RGBA16F g-buffer for the SSAO pass. Clip position uses the shared
// FrameData.viewProj (so geometry rasterizes identically to the lit pass); the push constant supplies
// BOTH the model matrix AND the view matrix so view-space quantities can be computed here without
// disturbing the shared FrameData layout. Mirrors lit.vert's HF_MSL_GEN binding convention.
struct VSInput {
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 color   : COLOR;
    [[vk::location(2)]] float2 uv      : TEXCOORD0;
    [[vk::location(3)]] float3 normal  : NORMAL;
    [[vk::location(4)]] float3 tangent : TANGENT;
};
struct VSOutput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 vnormal : NORMAL;     // view-space normal
    [[vk::location(1)]] float3 vpos    : POSITION0;  // view-space position
};
#include "frame_data.hlsli"
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 model; float4x4 view; };
#define HF_MODEL model
#define HF_VIEW  view
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4x4 model; float4x4 view; } pc;
#define HF_MODEL pc.model
#define HF_VIEW  pc.view
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    float4 world = mul(HF_MODEL, float4(i.pos, 1.0));
    o.clip = mul(f.viewProj, world);
    // View-space position + normal. modelView = view * model; (float3x3) is the rotation part, valid
    // for rotation + uniform scale (true for the spheres and the uniformly-scaled ground plane).
    float4x4 modelView = mul(HF_VIEW, HF_MODEL);
    o.vpos = mul(modelView, float4(i.pos, 1.0)).xyz;
    o.vnormal = normalize(mul((float3x3)modelView, i.normal));
    return o;
}
