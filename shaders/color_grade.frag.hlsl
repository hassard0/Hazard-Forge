// Color grading — analytic lift/gamma/gain + ASC-CDL slope/offset/power + saturation, applied
// POST-tonemap (Slice DB). A fullscreen pass (reuse post.vert) that reads the resolved TONEMAPPED scene
// color (t0/s0 — the engine's standard post.frag ACES+gamma LDR result, rendered into an intermediate
// RT) and applies grade::Grade(color, params) from render/color_grade.h (LiftGammaGain + ASC_CDL +
// Saturate + Grade copied VERBATIM below), outputting the graded color. The grade controls come from a
// small fragment push constant (the GradeParams, ≤128B). The EXISTING tonemap/post path + its goldens
// stay BYTE-IDENTICAL (this is a NEW path behind --colorgrade-shot; at the IDENTITY params it is a pure
// pass-through).
//
// THE IDENTITY-GRADE NO-OP PROOF: at the IDENTITY params (lift 0, gamma 1, gain 1, slope 1, offset 0,
// power 1, saturation 1) Grade(c) == c EXACTLY (the 1/gamma & power exponents are guarded to skip the
// pow at exponent 1, saturation == 1 short-circuits, lift 0 / offset 0 add nothing, the clamp >= 0 is a
// no-op on an in-range LDR color) — so the color_grade pass at identity is a pure pass-through. The
// showcase renders this shader at the identity params and asserts BYTE-IDENTICAL (SHA) to the engine's
// standard ungraded (tonemap-only) render of the same scene — the SAME color_grade shader at identity
// vs the ungraded render, so the proof is backend-portable.
//
// SEAM DISCIPLINE: this shader is ABOVE the RHI seam; the only mentions of vk/MSL here are the
// HF_MSL_GEN generation-path guard + [[vk::binding]] decorations (same as dof.frag / gtao.frag /
// sss_blur.frag), not backend CODE symbols. spirv-cross maps these SPIR-V bindings to the engine's flat
// Metal texture/sampler indices so the SAME HLSL feeds Vulkan (DXC) and Metal (glslang->spirv-cross):
// bit-identical math. Single frame, NO RNG/time -> deterministic, two runs byte-identical.
[[vk::binding(0, 0)]] Texture2D    gScene : register(t0);   // resolved TONEMAPPED LDR scene color
[[vk::binding(1, 0)]] SamplerState gSmp   : register(s0);

// GradeParams — each Vec3 padded to a float4 for clean cbuffer alignment (6 float4 + a float4 carrying
// saturation in .x) = 112 bytes, within the 128B push-constant budget. Mirrors GradeParams in the
// CPU-side showcase / render/color_grade.h (the showcase packs the same layout).
struct GradeParams {
    float4 lift;     // .xyz = lift   (identity 0)
    float4 gamma;    // .xyz = gamma  (identity 1)
    float4 gain;     // .xyz = gain   (identity 1)
    float4 slope;    // .xyz = slope  (identity 1)
    float4 offset;   // .xyz = offset (identity 0)
    float4 power;    // .xyz = power  (identity 1)
    float4 sat;      // .x   = saturation (identity 1); .yzw pad
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer GradePC { GradeParams gp; };
#define HF_GP gp
#else
[[vk::push_constant]] struct { GradeParams p; } pc;
#define HF_GP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

static const float3 kLuma709 = float3(0.2126, 0.7152, 0.0722);  // render/color_grade.h kLuma*

// ---- render/color_grade.h PowCh, copied VERBATIM. base^exp, but EXACT base at exp == 1 (the guarded
//   identity, no library pow rounding) and a clamped-non-negative base (a fractional power of a negative
//   is NaN). ----
float PowCh(float base, float exp) {
    if (exp == 1.0) return base;
    float b = (base > 0.0) ? base : 0.0;
    return pow(b, exp);
}

// ---- render/color_grade.h LiftGammaGain, copied VERBATIM. out = (gain*(c + lift*(1-c)))^(1/gamma);
//   identity (lift 0, gamma 1, gain 1) -> c EXACTLY. ----
float3 LiftGammaGain(float3 c, float3 lift, float3 gamma, float3 gain) {
    float3 lifted = c + lift * (1.0 - c);   // raise the shadows (fades out toward white)
    float3 gained = gain * lifted;          // highlight gain
    return float3(PowCh(gained.x, 1.0 / gamma.x),
                  PowCh(gained.y, 1.0 / gamma.y),
                  PowCh(gained.z, 1.0 / gamma.z));
}

// ---- render/color_grade.h ASC_CDL, copied VERBATIM. out = max(c*slope + offset, 0)^power;
//   identity (slope 1, offset 0, power 1) -> c EXACTLY. ----
float3 ASC_CDL(float3 c, float3 slope, float3 offset, float3 power) {
    float3 sop = max(c * slope + offset, 0.0);   // slope*in + offset, clamped >= 0 before the power
    return float3(PowCh(sop.x, power.x),
                  PowCh(sop.y, power.y),
                  PowCh(sop.z, power.z));
}

// ---- render/color_grade.h Saturate, copied VERBATIM. lerp(Rec.709 luma, c, saturation);
//   identity (saturation 1) -> c EXACTLY (short-circuited). ----
float3 Saturate709(float3 c, float saturation) {
    if (saturation == 1.0) return c;
    float grey = dot(c, kLuma709);
    return grey + (c - grey) * saturation;
}

// ---- render/color_grade.h Grade, copied VERBATIM. LiftGammaGain -> ASC_CDL -> Saturate;
//   identity params -> c EXACTLY (the no-op proof's per-pixel core). ----
float3 Grade(float3 c, GradeParams p) {
    float3 r = LiftGammaGain(c, p.lift.xyz, p.gamma.xyz, p.gain.xyz);
    r = ASC_CDL(r, p.slope.xyz, p.offset.xyz, p.power.xyz);
    r = Saturate709(r, p.sat.x);
    return r;
}

float4 main(PSInput i) : SV_Target {
    float3 c = gScene.Sample(gSmp, i.uv).rgb;   // the resolved tonemapped LDR scene color
    float3 g = Grade(c, HF_GP);                 // identity params -> g == c EXACTLY (pure pass-through)
    return float4(g, 1.0);
}
