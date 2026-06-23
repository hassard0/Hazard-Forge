// Order-Independent Transparency — accumulate-pass VERTEX shader (Slice CO, Weighted Blended OIT).
// Rasterizes a transparent object and passes (a) the VIEW-space linear depth (= -vpos.z, the SAME
// positive linear depth gbuffer.frag stores) so the fragment can compute the McGuire oit::Weight, and
// (b) the per-draw straight color + alpha (constant across the primitive, from the push constant).
// Clip position uses the shared FrameData.viewProj (so the transparent geometry rasterizes identically
// to the lit/gbuffer passes); the push constant supplies model + color/alpha. The VIEW-space linear
// depth is read straight off the clip w: Mat4::Perspective sets m[11] = -1, so clip.w == -view.z ==
// the positive view-space linear depth (the SAME quantity gbuffer.frag stores as -vpos.z) — no view
// matrix needed in the push constant, keeping it at 80B (<= the 128B push-constant budget). Mirrors
// lit.vert's HF_MSL_GEN binding scheme.
struct VSInput {
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 color   : COLOR;
    [[vk::location(2)]] float2 uv      : TEXCOORD0;
    [[vk::location(3)]] float3 normal  : NORMAL;
    [[vk::location(4)]] float3 tangent : TANGENT;
};
struct VSOutput {
    float4 clip      : SV_Position;
    // View-space linear depth (positive, increasing with distance) — feeds oit::Weight in the frag.
    [[vk::location(0)]] float viewDepth : TEXCOORD0;
    // Per-draw straight color (rgb) + alpha (w), constant across the primitive (nointerpolation: it
    // comes straight from the push constant, no perspective division needed).
    [[vk::location(1)]] nointerpolation float4 colorAlpha : TEXCOORD1;
};
#include "frame_data.hlsli"
// HF_MSL_GEN: glslang lowers [[vk::push_constant]] to a UBO at (set0,binding0) colliding with Frame,
// so for the MSL path declare the push block as an explicit cbuffer at distinct bindings (matching the
// engine's flat Metal buffer indices: vertex buffer0 = vertex stream, buffer1 = FrameData, buffer2 =
// the push block). The Vulkan/DXC path keeps the real push constant + [[vk::binding(0,0)]] Frame.
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 model; float4 colorAlpha; };
#define HF_MODEL model
#define HF_CA    colorAlpha
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4x4 model; float4 colorAlpha; } pc;
#define HF_MODEL pc.model
#define HF_CA    pc.colorAlpha
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    float4 world = mul(HF_MODEL, float4(i.pos, 1.0));
    o.clip = mul(f.viewProj, world);
    // View-space linear depth straight off the clip w (m[11] == -1 -> clip.w == -view.z, positive in
    // front of the camera, increasing with distance) — identical to gbuffer.frag's -vpos.z.
    o.viewDepth = o.clip.w;
    o.colorAlpha = HF_CA;
    return o;
}
