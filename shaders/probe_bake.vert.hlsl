// Slice AK — probe REFLECTION bake vertex shader. Transforms room geometry into ONE cube face's clip
// space. The face view-proj arrives in the PUSH CONSTANT alongside the model matrix (exactly like
// shadow_csm.vert), so all 6 faces can be rendered into the 6 atlas tiles within ONE render pass via
// SetViewport WITHOUT re-uploading the per-frame UBO between faces (a single shared UBO would make
// every face see the LAST face's VP). Passes uv + world normal/pos through to the bake fragment.
//
// Push constant: { float4x4 faceViewProj; float4x4 model; } = 128 bytes.
struct VSInput {
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 color   : COLOR;
    [[vk::location(2)]] float2 uv      : TEXCOORD0;
    [[vk::location(3)]] float3 normal  : NORMAL;
    [[vk::location(4)]] float3 tangent : TANGENT;
};
struct VSOutput {
    float4 clip : SV_Position;
    [[vk::location(0)]] float2 uv      : TEXCOORD0;
    [[vk::location(1)]] float3 wnormal : NORMAL;
    [[vk::location(2)]] float3 wpos    : POSITION0;
};
#ifdef HF_MSL_GEN
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 faceViewProj; float4x4 model; };
#define HF_VP    faceViewProj
#define HF_MODEL model
#else
[[vk::push_constant]] struct { float4x4 faceViewProj; float4x4 model; } pc;
#define HF_VP    pc.faceViewProj
#define HF_MODEL pc.model
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    float4 world = mul(HF_MODEL, float4(i.pos, 1.0));
    o.wpos = world.xyz;
    o.clip = mul(HF_VP, world);
    o.wnormal = normalize(mul((float3x3)HF_MODEL, i.normal));
    o.uv = i.uv;
    return o;
}
