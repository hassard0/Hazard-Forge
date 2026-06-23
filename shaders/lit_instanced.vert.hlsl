// Instanced variant of lit.vert.hlsl: GPU instanced rendering. The per-object model matrix arrives
// NOT from the push constant but from a SECOND, per-instance vertex stream (binding 1): four
// RGBA32_Float attributes at locations 7,8,9,10 carry the four COLUMNS of a column-major float4x4
// (matching the CPU-side math::Mat4 byte layout and the existing push-constant model matrix). The
// position/normal/tangent are transformed by this per-instance model exactly like lit.vert.hlsl, so
// one DrawIndexedInstanced call places a whole field of meshes. The fragment stage is the SHARED,
// UNCHANGED lit.frag.hlsl. A tiny push constant still carries the per-draw float4 material
// (metallic/roughness) so the shared fragment's material path is fed. Mirrors lit.vert.hlsl's
// HF_MSL_GEN binding conventions (Frame at binding(1,0), PushC at binding(2,0) for the MSL path).
struct VSInput {
    [[vk::location(0)]]  float3 pos     : POSITION;
    [[vk::location(1)]]  float3 color   : COLOR;
    [[vk::location(2)]]  float2 uv      : TEXCOORD0;
    [[vk::location(3)]]  float3 normal  : NORMAL;
    [[vk::location(4)]]  float3 tangent : TANGENT;
    // Per-instance model matrix columns (binding 1, per-instance step rate).
    [[vk::location(7)]]  float4 imodel0 : TEXCOORD2;  // column 0
    [[vk::location(8)]]  float4 imodel1 : TEXCOORD3;  // column 1
    [[vk::location(9)]]  float4 imodel2 : TEXCOORD4;  // column 2
    [[vk::location(10)]] float4 imodel3 : TEXCOORD5;  // column 3
};
struct VSOutput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
    [[vk::location(4)]] nointerpolation float2 material : TEXCOORD1; // x=metallic, y=roughness
    [[vk::location(5)]] float3 wtangent : TANGENT;
};
#include "frame_data.hlsli"
// PushC now carries ONLY the per-draw material (the model matrix comes from the instance stream).
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(2, 0)]] cbuffer PushC { float4 material; };
#define HF_MATERIAL material
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4 material; } pc;
#define HF_MATERIAL pc.material
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    // Assemble the column-major model matrix from the four per-instance column attributes. HLSL
    // float4x4(rows...) takes ROWS, so transpose the columns into the constructor (row r is the r-th
    // component of each column). Equivalent to a column-major matrix whose mul(model, v) matches the
    // push-constant path.
    float4x4 model = float4x4(
        float4(i.imodel0.x, i.imodel1.x, i.imodel2.x, i.imodel3.x),
        float4(i.imodel0.y, i.imodel1.y, i.imodel2.y, i.imodel3.y),
        float4(i.imodel0.z, i.imodel1.z, i.imodel2.z, i.imodel3.z),
        float4(i.imodel0.w, i.imodel1.w, i.imodel2.w, i.imodel3.w));

    float4 world = mul(model, float4(i.pos, 1.0));
    o.wpos = world.xyz;
    o.clip = mul(f.viewProj, world);
    o.wnormal = normalize(mul((float3x3)model, i.normal));
    o.wtangent = mul((float3x3)model, i.tangent);
    o.color = i.color; o.uv = i.uv;
    o.material = HF_MATERIAL.xy;
    return o;
}
