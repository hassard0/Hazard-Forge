// Instanced depth-only shadow pass: transform each instance's geometry into the light's clip space.
// The per-object model matrix comes from the SECOND, per-instance vertex stream (binding 1): four
// RGBA32_Float attributes at locations 7-10 carrying the columns of a column-major float4x4 (same
// convention as lit_instanced.vert.hlsl). One DrawIndexedInstanced casts the whole field's shadows.
// No push constant is used (the field's transforms are all in the instance stream).
struct VSInput {
    [[vk::location(0)]]  float3 pos    : POSITION;
    [[vk::location(1)]]  float3 color  : COLOR;
    [[vk::location(2)]]  float2 uv     : TEXCOORD0;
    [[vk::location(3)]]  float3 normal : NORMAL;
    [[vk::location(7)]]  float4 imodel0 : TEXCOORD2;  // column 0
    [[vk::location(8)]]  float4 imodel1 : TEXCOORD3;  // column 1
    [[vk::location(9)]]  float4 imodel2 : TEXCOORD4;  // column 2
    [[vk::location(10)]] float4 imodel3 : TEXCOORD5;  // column 3
};
#include "frame_data.hlsli"
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
#endif

float4 main(VSInput i) : SV_Position {
    float4x4 model = float4x4(
        float4(i.imodel0.x, i.imodel1.x, i.imodel2.x, i.imodel3.x),
        float4(i.imodel0.y, i.imodel1.y, i.imodel2.y, i.imodel3.y),
        float4(i.imodel0.z, i.imodel1.z, i.imodel2.z, i.imodel3.z),
        float4(i.imodel0.w, i.imodel1.w, i.imodel2.w, i.imodel3.w));
    float4 world = mul(model, float4(i.pos, 1.0));
    return mul(f.lightViewProj, world);
}
