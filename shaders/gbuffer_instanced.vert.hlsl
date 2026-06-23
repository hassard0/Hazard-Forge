// SSAO normal+depth prepass — INSTANCED vertex shader (Slice Y). Instanced variant of gbuffer.vert:
// the per-object model matrix arrives from the SECOND per-instance vertex stream (binding 1, columns
// at locations 7-10, exactly like lit_instanced.vert), and the push constant carries ONLY the view
// matrix. Writes the same view-space normal + view-space position interpolants as gbuffer.vert.
struct VSInput {
    [[vk::location(0)]]  float3 pos     : POSITION;
    [[vk::location(1)]]  float3 color   : COLOR;
    [[vk::location(2)]]  float2 uv      : TEXCOORD0;
    [[vk::location(3)]]  float3 normal  : NORMAL;
    [[vk::location(4)]]  float3 tangent : TANGENT;
    [[vk::location(7)]]  float4 imodel0 : TEXCOORD2;  // column 0
    [[vk::location(8)]]  float4 imodel1 : TEXCOORD3;  // column 1
    [[vk::location(9)]]  float4 imodel2 : TEXCOORD4;  // column 2
    [[vk::location(10)]] float4 imodel3 : TEXCOORD5;  // column 3
};
struct VSOutput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 vnormal : NORMAL;     // view-space normal
    [[vk::location(1)]] float3 vpos    : POSITION0;  // view-space position
};
#include "frame_data.hlsli"
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 view; };
#define HF_VIEW view
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4x4 view; } pc;
#define HF_VIEW pc.view
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    // Assemble the column-major model matrix from the four per-instance column attributes (HLSL
    // float4x4 takes ROWS, so transpose the columns in — identical to lit_instanced.vert).
    float4x4 model = float4x4(
        float4(i.imodel0.x, i.imodel1.x, i.imodel2.x, i.imodel3.x),
        float4(i.imodel0.y, i.imodel1.y, i.imodel2.y, i.imodel3.y),
        float4(i.imodel0.z, i.imodel1.z, i.imodel2.z, i.imodel3.z),
        float4(i.imodel0.w, i.imodel1.w, i.imodel2.w, i.imodel3.w));
    float4 world = mul(model, float4(i.pos, 1.0));
    o.clip = mul(f.viewProj, world);
    float4x4 modelView = mul(HF_VIEW, model);
    o.vpos = mul(modelView, float4(i.pos, 1.0)).xyz;
    o.vnormal = normalize(mul((float3x3)modelView, i.normal));
    return o;
}
