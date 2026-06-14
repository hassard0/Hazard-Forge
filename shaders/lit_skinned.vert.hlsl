// Skinned variant of lit.vert.hlsl: GPU vertex skinning. Reads the joint-index (loc 5) and
// joint-weight (loc 6) quads, builds the per-vertex skinning matrix from the JointPalette UBO
// (descriptor set 2), transforms position by skinMat then the push-constant model matrix, and
// transforms the normal/tangent by skinMat's upper 3x3 (rotation + uniform-scale only — the Fox has
// no non-uniform joint scale, so no inverse-transpose is needed). The fragment stage is the SHARED
// lit.frag.hlsl, unchanged. Mirrors lit.vert.hlsl's HF_MSL_GEN conventions exactly.
struct VSInput {
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 color   : COLOR;
    [[vk::location(2)]] float2 uv      : TEXCOORD0;
    [[vk::location(3)]] float3 normal  : NORMAL;
    [[vk::location(4)]] float3 tangent : TANGENT;
    [[vk::location(5)]] float4 joints  : BLENDINDICES;  // joint indices, stored as floats
    [[vk::location(6)]] float4 weights : BLENDWEIGHT;   // joint weights (sum to 1)
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
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};
// 64-joint palette (set 2). The CPU writes palette[j] = global[j] * inverseBind[j]; vertices index
// into it via the joint-index quad. See lit.vert.hlsl for the HF_MSL_GEN binding rationale; the
// joint palette adds set 2 -> binding(3,2) so spirv-cross --msl-decoration-binding lands it at the
// next free Metal vertex buffer slot (buffer(3)).
struct JointPalette { float4x4 joints[64]; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 model; float4 material; };
[[vk::binding(3, 2)]] cbuffer Joints { JointPalette jp; };
#define HF_MODEL model
#define HF_MATERIAL material
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4x4 model; float4 material; } pc;
[[vk::binding(0, 2)]] cbuffer Joints { JointPalette jp; };
#define HF_MODEL pc.model
#define HF_MATERIAL pc.material
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    // Per-vertex skinning matrix: weighted sum of the four influencing joints' palette matrices.
    int4 idx = int4(i.joints + 0.5);  // float-stored indices -> int (round)
    float4x4 skin = i.weights.x * jp.joints[idx.x]
                  + i.weights.y * jp.joints[idx.y]
                  + i.weights.z * jp.joints[idx.z]
                  + i.weights.w * jp.joints[idx.w];

    float4 skinned = mul(skin, float4(i.pos, 1.0));
    float4 world = mul(HF_MODEL, skinned);
    o.wpos = world.xyz;
    o.clip = mul(f.viewProj, world);

    // Normal/tangent through skin then model (rotation + uniform scale only).
    float3 sn = mul((float3x3)skin, i.normal);
    float3 st = mul((float3x3)skin, i.tangent);
    o.wnormal = normalize(mul((float3x3)HF_MODEL, sn));
    o.wtangent = mul((float3x3)HF_MODEL, st);
    o.color = i.color; o.uv = i.uv;
    o.material = HF_MATERIAL.xy;
    return o;
}
