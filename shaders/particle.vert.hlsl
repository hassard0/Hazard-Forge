// Particle point renderer. The particle storage buffer (written by particles.comp.hlsl) is bound
// as the vertex stream: attribute 0 = position (posLife.xyz), attribute 1 = velocity (velSeed.xyz).
// Output a clip-space point; color + point size derived from speed so the fountain reads as motion.
struct VSInput {
    [[vk::location(0)]] float3 pos : POSITION;
    [[vk::location(1)]] float3 vel : NORMAL;   // velocity reused as a generic vec3 attribute
};
struct VSOutput {
    float4 clip : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
    [[vk::builtin("PointSize")]] float psize : PSIZE;  // Vulkan gl_PointSize
};

#include "frame_data.hlsli"
// Match the engine binding convention used by the other shaders: real Frame cbuffer at (0,0) for
// DXC/Vulkan; HF_MSL_GEN moves it to (1,0) so spirv-cross emits Metal buffer(1) (see lit.vert.hlsl).
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    o.clip = mul(f.viewProj, float4(i.pos, 1.0));
    float speed = length(i.vel);
    // High-contrast cool palette so the fountain pops over the warm/yellow scene under additive
    // blend: cyan core -> white tips with rising speed (never warm, which would wash out).
    float t = saturate(speed / 6.0);
    o.color = lerp(float3(0.15, 0.65, 1.0), float3(0.9, 0.97, 1.0), t);
    o.psize = 2.0;
    return o;
}
