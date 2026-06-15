// Depth-only CSM shadow caster (Slice AD): transform geometry into ONE cascade's light clip space.
// Unlike shadow.vert (which reads the single lightViewProj from the frame UBO), the cascade's
// lightViewProj arrives in the PUSH CONSTANT alongside the model matrix, so the same per-frame UBO
// can drive all N cascades without re-uploading between them. The atlas-tile sub-rect is selected on
// the CPU side via SetViewport before each cascade's draws.
//
// Push constant layout: { float4x4 cascadeViewProj; float4x4 model; } = 128 bytes.
struct VSInput {
    [[vk::location(0)]] float3 pos    : POSITION;
    [[vk::location(1)]] float3 color  : COLOR;
    [[vk::location(2)]] float2 uv     : TEXCOORD0;
    [[vk::location(3)]] float3 normal : NORMAL;
};
// See shaders/lit.vert.hlsl for the HF_MSL_GEN rationale (glslang ignores [[vk::push_constant]]).
// For MSL gen the push block becomes an explicit cbuffer at a distinct binding so spirv-cross emits
// the engine's flat Metal buffer index (vertex: buffer0 = vertex stream, buffer1 = ..., buffer2 = pc).
#ifdef HF_MSL_GEN
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 cascadeViewProj; float4x4 model; };
#define HF_VP    cascadeViewProj
#define HF_MODEL model
#else
[[vk::push_constant]] struct { float4x4 cascadeViewProj; float4x4 model; } pc;
#define HF_VP    pc.cascadeViewProj
#define HF_MODEL pc.model
#endif

float4 main(VSInput i) : SV_Position {
    float4 world = mul(HF_MODEL, float4(i.pos, 1.0));
    return mul(HF_VP, world);
}
