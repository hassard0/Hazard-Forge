// Hand-written MSL equivalent of shaders/post.vert.hlsl + shaders/post.frag.hlsl, for the Metal
// fullscreen post-processing pass. Mirrors the Vulkan post pipeline: a fullscreen triangle
// generated from [[vertex_id]] (no vertex buffer), then a fragment that samples the offscreen
// render-target color image and applies ACES tonemap + gamma + cinematic grade + vignette.
//
// Binding convention (must match engine/rhi_metal/metal_common.h):
//   fragment: texture(0) / sampler(0) = the render-target color image (bound via BindTexture(rt))
//
// The post pipeline is created with desc.fullscreen = true, so MetalPipeline gives it no vertex
// descriptor and MetalCommandBuffer binds it with cull mode NONE.

#include <metal_stdlib>
using namespace metal;

struct PostVSOut {
    float4 clip [[position]];
    float2 uv;
};

// Fullscreen triangle from the vertex id: (0,0),(2,0),(0,2) -> covers clip space with one tri.
//   uv.x = p.x as-is.
//   uv.y is flipped (1 - p.y): the RT color texture stores row 0 = top (the lit pass already
//   applies the Vulkan->Metal NDC Y flip when rendering into it), and Metal samples uv (0,0) at
//   the texture's top-left — so without this flip the post output would be upside-down relative to
//   the no-post capture. Flipping uv.y keeps the final orientation identical to the direct path.
vertex PostVSOut post_vertex(uint vid [[vertex_id]]) {
    PostVSOut o;
    float2 p = float2((vid << 1) & 2, vid & 2);   // (0,0),(2,0),(0,2)
    o.uv = float2(p.x, 1.0 - p.y);
    o.clip = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

constant float3 kLuma = float3(0.299, 0.587, 0.114);

// ACES filmic tonemapping (Narkowicz approximation): cinematic highlight roll-off.
static float3 ACES(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// Cinematic color grade on the displayed LDR color (ported from post.frag.hlsl):
//  - split-tone: cool/teal shadows, warm highlights
//  - gentle contrast S-curve + small saturation boost.
static float3 ColorGrade(float3 c) {
    float luma = dot(c, kLuma);
    const float3 shadowTint = float3(0.90, 1.00, 1.06);  // cool / teal
    const float3 highTint   = float3(1.06, 1.00, 0.92);  // warm
    const float  gradeAmt   = 0.08;
    float3 tint = mix(shadowTint, highTint, luma);
    c = mix(c, c * tint, gradeAmt);
    c = mix(c, smoothstep(0.0, 1.0, c), 0.15);           // contrast S-curve
    float gluma = dot(c, kLuma);
    c = mix(float3(gluma), c, 1.08);                     // saturation
    return c;
}

fragment float4 post_fragment(PostVSOut in [[stage_in]],
                              texture2d<float> tex [[texture(0)]],
                              sampler smp          [[sampler(0)]]) {
    float3 c = tex.sample(smp, in.uv).rgb;   // scene color (linear-ish)

    c = ACES(c);                             // filmic tonemap
    c = pow(c, float3(1.0 / 2.2));           // gamma -> displayed LDR
    c = ColorGrade(c);                       // split-tone + contrast + saturation

    // Vignette: darken the corners.
    float2 d = in.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(d));
    return float4(saturate(c * vig), 1.0);
}
