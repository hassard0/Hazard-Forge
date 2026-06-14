// Skinned depth-only shadow pass: same skinning as lit_skinned.vert.hlsl, but outputs only the
// light-space clip position (no fragment stage). Needed so the Fox casts a correctly-posed shadow
// (the static shadow.vert.hlsl ignores joints and would render the bind pose).
struct VSInput {
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 color   : COLOR;
    [[vk::location(2)]] float2 uv      : TEXCOORD0;
    [[vk::location(3)]] float3 normal  : NORMAL;
    [[vk::location(4)]] float3 tangent : TANGENT;
    [[vk::location(5)]] float4 joints  : BLENDINDICES;
    [[vk::location(6)]] float4 weights : BLENDWEIGHT;
};
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};
struct JointPalette { float4x4 joints[64]; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 model; };
[[vk::binding(3, 2)]] cbuffer Joints { JointPalette jp; };
#define HF_MODEL model
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4x4 model; } pc;
[[vk::binding(0, 2)]] cbuffer Joints { JointPalette jp; };
#define HF_MODEL pc.model
#endif

float4 main(VSInput i) : SV_Position {
    int4 idx = int4(i.joints + 0.5);
    float4x4 skin = i.weights.x * jp.joints[idx.x]
                  + i.weights.y * jp.joints[idx.y]
                  + i.weights.z * jp.joints[idx.z]
                  + i.weights.w * jp.joints[idx.w];
    float4 world = mul(HF_MODEL, mul(skin, float4(i.pos, 1.0)));
    return mul(f.lightViewProj, world);
}
