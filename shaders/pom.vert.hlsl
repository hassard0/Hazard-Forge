// Slice CP — Parallax Occlusion Mapping vertex shader. A near-clone of lit.vert that ALSO forwards the
// POM parameters (heightScale, numSteps) packed in the per-draw material push constant's .zw so the
// fragment shader (pom.frag.hlsl) can run the tangent-space height march. The model transform, world
// position/normal/tangent, uv, and metallic/roughness pass-through are IDENTICAL to lit.vert, so with
// heightScale == 0 the downstream shading is byte-identical to the plain lit/normal-mapped path.
struct VSInput {
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 color   : COLOR;
    [[vk::location(2)]] float2 uv      : TEXCOORD0;
    [[vk::location(3)]] float3 normal  : NORMAL;
    [[vk::location(4)]] float3 tangent : TANGENT;
};
struct VSOutput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
    // x=metallic, y=roughness, z=heightScale, w=numSteps. nointerpolation: straight from the push const.
    [[vk::location(4)]] nointerpolation float4 material : TEXCOORD1;
    [[vk::location(5)]] float3 wtangent : TANGENT;
};
#include "frame_data.hlsli"
// Same HF_MSL_GEN seam-discipline as lit.vert: glslang lowers [[vk::push_constant]] to a plain uniform,
// so for MSL generation we declare the model+material as an explicit cbuffer at distinct bindings whose
// spirv-cross flat indices match the engine's vertex buffer1=FrameData / buffer2=model layout. The
// Vulkan/DXC path keeps the real push constant. (These #ifdef'd names are NOT backend code symbols.)
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 model; float4 material; };
#define HF_MODEL model
#define HF_MATERIAL material
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4x4 model; float4 material; } pc;
#define HF_MODEL pc.model
#define HF_MATERIAL pc.material
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    float4 world = mul(HF_MODEL, float4(i.pos, 1.0));
    o.wpos = world.xyz;
    o.clip = mul(f.viewProj, world);
    o.wnormal = normalize(mul((float3x3)HF_MODEL, i.normal));
    o.wtangent = mul((float3x3)HF_MODEL, i.tangent);
    o.color = i.color; o.uv = i.uv;
    o.material = HF_MATERIAL;  // metallic, roughness, heightScale, numSteps
    return o;
}
