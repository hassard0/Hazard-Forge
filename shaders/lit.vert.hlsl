struct VSInput {
    [[vk::location(0)]] float3 pos    : POSITION;
    [[vk::location(1)]] float3 color  : COLOR;
    [[vk::location(2)]] float2 uv     : TEXCOORD0;
    [[vk::location(3)]] float3 normal : NORMAL;
};
struct VSOutput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
};
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};
// HF_MSL_GEN: the glslang HLSL frontend (glslc/glslangValidator, used on macOS where DXC is
// unavailable) does NOT honour [[vk::push_constant]] — it lowers it to a plain Uniform buffer at
// (set 0, binding 0), colliding with the Frame cbuffer. So for the MSL-generation path only we
// declare the model matrix as an explicit cbuffer with distinct bindings chosen so that
// `spirv-cross --msl-decoration-binding` emits exactly the engine's flat Metal buffer indices
// (vertex: buffer0 = vertex stream, buffer1 = FrameData, buffer2 = model). The Vulkan/DXC path is
// untouched: it keeps the real push constant and the original [[vk::binding(0,0)]] Frame.
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 model; };
#define HF_MODEL model
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4x4 model; } pc;
#define HF_MODEL pc.model
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    float4 world = mul(HF_MODEL, float4(i.pos, 1.0));
    o.wpos = world.xyz;
    o.clip = mul(f.viewProj, world);
    // (float3x3)model is correct for rotation + uniform scale only. Non-uniform scale needs the
    // inverse-transpose normal matrix (pass it separately when scaled geometry is introduced).
    o.wnormal = normalize(mul((float3x3)HF_MODEL, i.normal));
    o.color = i.color; o.uv = i.uv;
    return o;
}
