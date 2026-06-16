// ImGui UI vertex shader. Maps ImDrawVert (pos: float2, uv: float2, col: RGBA8) to clip space via
// a screen-space ortho transform passed as a vertex push constant {scale.xy, translate.xy}.
// Vertex color arrives as a normalized RGBA8_UNorm attribute (0..1) and is passed straight through.
struct PushConstants { float2 scale; float2 translate; };

// HF_MSL_GEN: glslang (glslc, used for the macOS MSL-gen path) does NOT honour [[vk::push_constant]]
// — it lowers it to a plain uniform at (set 0, binding 0), which would land on Metal vertex buffer(0)
// and collide with the stage_in vertex data. So for the MSL-gen path only we declare the ortho as an
// explicit cbuffer at [[vk::binding(2,0)]] so spirv-cross --msl-decoration-binding lands it on Metal
// vertex buffer(2) — exactly where MetalCommandBuffer::PushConstants writes it (kVbPushConst). The
// Vulkan/DXC path keeps the real push constant. ImGuiRenderer's UiPush struct matches both layouts.
#ifdef HF_MSL_GEN
[[vk::binding(2, 0)]] cbuffer PushC { PushConstants pc; };
#else
[[vk::push_constant]] PushConstants pc;
#endif

struct VSInput {
    [[vk::location(0)]] float2 pos : POSITION;
    [[vk::location(1)]] float2 uv  : TEXCOORD0;
    [[vk::location(2)]] float4 col : COLOR0;   // RGBA8_UNorm -> normalized 0..1
};
struct VSOutput {
    float4 pos : SV_Position;
    [[vk::location(0)]] float4 col : COLOR0;
    [[vk::location(1)]] float2 uv  : TEXCOORD0;
};

VSOutput main(VSInput i) {
    VSOutput o;
    o.pos = float4(i.pos * pc.scale + pc.translate, 0.0, 1.0);
    o.col = i.col;
    o.uv  = i.uv;
    return o;
}
