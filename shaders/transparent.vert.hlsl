// Transparent ("glass") vertex shader (Slice T). Shares the standard mesh vertex layout and the
// per-frame UBO, and carries a per-object { tint.rgb, baseAlpha } via the push constant alongside the
// model matrix — same 80-byte push layout as lit.vert.hlsl (float4x4 model + float4), so the
// HF_MSL_GEN binding story (Frame at buffer(1), push constant at buffer(2)) is identical. It forwards
// the tint+baseAlpha to the fragment as a flat (nointerpolation) interpolant.
struct VSInput {
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 color   : COLOR;
    [[vk::location(2)]] float2 uv      : TEXCOORD0;
    [[vk::location(3)]] float3 normal  : NORMAL;
    [[vk::location(4)]] float3 tangent : TANGENT;
};
struct VSOutput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 wnormal : NORMAL;
    [[vk::location(1)]] float3 wpos    : POSITION0;
    // Per-object glass parameters, constant across the primitive: rgb = tint, w = base alpha.
    [[vk::location(2)]] nointerpolation float4 tintAlpha : TEXCOORD0;
};
#include "frame_data.hlsli"
// Same HF_MSL_GEN handling as lit.vert.hlsl: glslang lowers [[vk::push_constant]] to a plain uniform
// at (set 0, binding 0), colliding with Frame, so for the MSL-generation path we declare the push
// block as an explicit cbuffer at distinct bindings (Frame -> buffer(1), PushC -> buffer(2) after
// --msl-decoration-binding). The Vulkan/DXC path keeps the real push constant.
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 model; float4 tintAlpha; };
#define HF_MODEL model
#define HF_TINTALPHA tintAlpha
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4x4 model; float4 tintAlpha; } pc;
#define HF_MODEL pc.model
#define HF_TINTALPHA pc.tintAlpha
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    float4 world = mul(HF_MODEL, float4(i.pos, 1.0));
    o.wpos = world.xyz;
    o.clip = mul(f.viewProj, world);
    // Rotation + uniform scale only (same caveat as lit.vert): use (float3x3)model for the normal.
    o.wnormal = normalize(mul((float3x3)HF_MODEL, i.normal));
    o.tintAlpha = HF_TINTALPHA;
    return o;
}
