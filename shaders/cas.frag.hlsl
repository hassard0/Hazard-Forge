// Contrast-Adaptive Sharpening (CAS; AMD FidelityFX CAS) — a NEW fullscreen post pass (Slice DF). A
// fullscreen pass (reuse post.vert) that reads the resolved SDR scene color (t0/s0 — the engine's
// standard post.frag tonemapped LDR result, rendered into an intermediate RT), gathers the cross
// neighborhood (center + up/down/left/right via the texel size from GetDimensions), and applies
// render/cas.h's CasSharpen (CasWeight + CasSharpen copied VERBATIM below) from a small `sharpness`
// push constant, outputting the sharpened color. The EXISTING tonemap/post path + its goldens stay
// BYTE-IDENTICAL (this is a NEW path behind --cas-shot; at sharpness 0 a pure pass-through).
//
// THE sharpness=0 NO-OP PROOF: CasWeight returns EXACTLY 0 at sharpness == 0 (the short-circuit), so the
// CAS sharpen degenerates to out = (center + 0)/(1 + 0) = center and the clamp to the center's own
// neighborhood [min,max] is a no-op -> CasSharpen(..., 0) == center EXACTLY (no bias, no division/clamp
// drift). The showcase renders this shader at sharpness 0 and asserts BYTE-IDENTICAL (SHA) to the
// engine's standard (unsharpened) render of the same scene — the SAME cas shader at sharpness 0 vs the
// unsharpened render, so the proof is backend-portable.
//
// SEAM DISCIPLINE: this shader is ABOVE the RHI seam; the only mentions of vk/MSL here are the
// HF_MSL_GEN generation-path guard + [[vk::binding]] decorations (same as dof.frag / gtao.frag /
// color_grade.frag), not backend CODE symbols. spirv-cross maps these SPIR-V bindings to the engine's
// flat Metal texture/sampler indices so the SAME HLSL feeds Vulkan (DXC) and Metal (glslang->spirv-cross):
// bit-identical math. Single frame, NO RNG/time -> deterministic, two runs byte-identical.
[[vk::binding(0, 0)]] Texture2D    gScene : register(t0);   // resolved SDR (tonemapped LDR) scene color
[[vk::binding(1, 0)]] SamplerState gSmp   : register(s0);

// CasParams — a single sharpness scalar (padded to a float4 for clean cbuffer alignment). 16 bytes,
// well within the 128B push-constant budget. Mirrors CasParams in the CPU-side showcase.
struct CasParams {
    float4 sharpness;   // .x = sharpness in [0,1] (0 -> the no-op pass-through); .yzw pad
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer CasPC { CasParams cp; };
#define HF_CP cp
#else
[[vk::push_constant]] struct { CasParams p; } pc;
#define HF_CP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

// render/cas.h kLuma* (Rec.601-ish luma weights for the adaptive-weight luminance envelope).
static const float3 kCasLuma = float3(0.299, 0.587, 0.114);
static const float  kPeakLo  = -0.125;  // -1/8 : gentle sharpen (render/cas.h kPeakLo)
static const float  kPeakHi  = -0.200;  // -1/5 : max CAS sharpen (render/cas.h kPeakHi)
static const float  kCasEps  = 1e-4;    // render/cas.h kEps

// ---- render/cas.h CasWeight, copied VERBATIM. amp = sqrt(clamp(min(minL,1-maxL)/max(maxL,eps),0,1));
//   w = amp*lerp(-1/8,-1/5,sharpness); EXACTLY 0 at sharpness 0 (the no-op proof's exact OFF path). ----
float CasWeight(float minL, float maxL, float sharpness) {
    if (sharpness <= 0.0) return 0.0;                  // exact OFF: w 0 -> out == center (no-op proof)
    float headroom = min(minL, 1.0 - maxL);
    float denom = max(maxL, kCasEps);
    float ratio = clamp(headroom / denom, 0.0, 1.0);
    float amp = sqrt(ratio);
    float peak = lerp(kPeakLo, kPeakHi, sharpness);    // -1/8 .. -1/5
    return amp * peak;                                 // the negative per-neighbor lobe (w <= 0)
}

// ---- render/cas.h CasSharpen, copied VERBATIM. out = (center + w*(up+down+left+right))/(1+4w),
//   clamped per-channel to the neighborhood [min,max] (the no-ringing clamp); sharpness 0 -> w 0 ->
//   out == center EXACTLY (we return center directly when w == 0). ----
float3 CasSharpen(float3 c, float3 u, float3 d, float3 l, float3 r, float sharpness) {
    float3 lo = min(c, min(min(u, d), min(l, r)));     // per-channel neighborhood min
    float3 hi = max(c, max(max(u, d), max(l, r)));     // per-channel neighborhood max
    float minL = dot(lo, kCasLuma);
    float maxL = dot(hi, kCasLuma);
    float w = CasWeight(minL, maxL, sharpness);
    if (w == 0.0) return c;                            // exact OFF (the byte-identical no-op proof)
    float denom = 1.0 + 4.0 * w;
    float3 v = (c + w * (u + d + l + r)) / denom;      // normalized CAS sharpen
    return clamp(v, lo, hi);                           // no-ringing clamp: never past the neighbor min/max
}

float4 main(PSInput i) : SV_Target {
    float texW, texH;
    gScene.GetDimensions(texW, texH);
    float2 texel = 1.0 / float2(texW, texH);

    float3 c = gScene.Sample(gSmp, i.uv).rgb;                                  // center
    float3 u = gScene.Sample(gSmp, i.uv + float2(0.0, -texel.y)).rgb;          // up
    float3 d = gScene.Sample(gSmp, i.uv + float2(0.0,  texel.y)).rgb;          // down
    float3 l = gScene.Sample(gSmp, i.uv + float2(-texel.x, 0.0)).rgb;          // left
    float3 r = gScene.Sample(gSmp, i.uv + float2( texel.x, 0.0)).rgb;          // right

    float3 sharp = CasSharpen(c, u, d, l, r, HF_CP.sharpness.x);  // sharpness 0 -> sharp == c EXACTLY
    return float4(sharp, 1.0);
}
