// Subsurface scattering -- screen-space SEPARABLE diffusion blur (Slice CZ; Jimenez et al. 2015). A
// fullscreen pass (reuse post.vert) run TWICE: a HORIZONTAL pass then a VERTICAL pass (a uniform `axis`
// selects which). Each pass reads the resolved HDR scene color (t0/s0) and an "SSS G-buffer" carrying
// the subsurface MASK + the view-space linear-depth (t3/s3, bound via BindTexturePair like SSR/DoF),
// then runs the render/sss.h separable diffusion gather (DiffusionWeight + DepthFalloff + BlurAxis,
// copied VERBATIM below) along the pass axis. The two 1D passes compose the 2D subsurface diffusion: the
// diffuse lit color of a flagged subsurface material is softened by a small depth-aware radius -- the
// soft translucent glow that wraps the terminator. Non-subsurface pixels (mask 0) pass through
// unchanged.
//
// THE SUBSURFACE MASK + DEPTH SOURCE (documented -- NO new RHI seam): the second texture of the
// BindTexturePair (t3/s3) is an "SSS G-buffer" RT (RGBA16F, the SAME format/role as the SSAO/SSR
// G-buffer) produced by sss_mask.frag: it stores the per-pixel subsurface MASK in .r (1 = flagged
// subsurface material, 0 = opaque -- a flagged-material channel, exactly like a material-id G-buffer
// flag, driven by a per-draw material push constant) and the VIEW-SPACE LINEAR DEPTH in .w (= -vpos.z,
// the same convention gbuffer.frag uses). So SSS needs only the EXISTING scene-color + a depth/mask
// G-buffer pair -- the SAME BindTexturePair fullscreen path SSR/DoF/SSGI-denoise use (the SSS pass does
// not need the surface normal, so the normal channels are repurposed to carry the mask). The gather
// reads centerMask + depth from this RT and multiplies each TAP by its own mask so SSS color only pools
// within the subsurface region. The intermediate (horizontal-pass output) RT carries the blurred color;
// the vertical pass re-reads the SAME mask/depth G-buffer, so the mask is consistent across both passes.
//
// THE sssStrength=0 / sssWidth=0 NO-OP PROOF: the per-tap step is sssStrength * sssWidth / centerDepth;
// with sssStrength == 0 (or sssWidth == 0) every tap lands on the center -> the normalized gather equals
// the center color EXACTLY, so the two-pass SSS output is BYTE-IDENTICAL to the non-SSS lit render. We
// also short-circuit the early-out (sssStrength<=0 || sssWidth<=0 -> center) so the identity is exact +
// branch-clean regardless of the depth/mask field -- the SAME sss shader at sssStrength=0 vs the
// non-SSS render is byte-identical on every backend (backend-portable proof).
//
// SEAM DISCIPLINE: this shader is ABOVE the RHI seam; the only mentions of vk/MSL here are the
// HF_MSL_GEN generation-path guard + [[vk::binding]] decorations (same as dof.frag / gtao.frag), not
// backend CODE symbols. spirv-cross maps these SPIR-V bindings to the engine's flat Metal
// texture/sampler indices so the SAME HLSL feeds Vulkan (DXC) and Metal (glslang->spirv-cross):
// bit-identical math. SSS is a NEW path behind --sss-shot; the existing lit/post path + its goldens
// stay BYTE-IDENTICAL. Single frame, NO RNG/time -> deterministic, two runs byte-identical.
[[vk::binding(0, 0)]] Texture2D    gScene : register(t0);   // resolved HDR scene color (this pass input)
[[vk::binding(1, 0)]] SamplerState gSmp   : register(s0);
[[vk::binding(3, 0)]] Texture2D    gSss   : register(t3);   // SSS G-buffer: mask.r + linDepth.w
[[vk::binding(4, 0)]] SamplerState gSSmp  : register(s3);

struct SssParams {
    float2 texel;        // 1/size
    float  sssWidthPx;   // subsurface diffusion profile width in pixels
    float  sssStrength;  // global SSS strength (0 -> the no-op pass-through)
    float  taps;         // gather tap count along the axis (cast to int)
    float  depthScale;   // DepthFalloff depth window (view-space linear-depth units)
    float  axis;         // 0 = horizontal pass, 1 = vertical pass
    float  pad;
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer SssPC { SssParams sp; };
#define HF_SP sp
#else
[[vk::push_constant]] struct { SssParams p; } pc;
#define HF_SP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

// ---- render/sss.h DiffusionWeight, copied VERBATIM. A single normalized Gaussian skin profile
//   w(d) = exp(-kFalloff * (d/width)^2), kFalloff=3 -> w(0)=1, positive, strictly decreasing, a wider
//   width spreads weight. ----
float DiffusionWeight(float distancePx, float sssWidthPx) {
    const float kFalloff = 3.0;
    float width = max(sssWidthPx, 1e-6);
    float x = distancePx / width;
    return exp(-kFalloff * x * x);
}

// ---- render/sss.h DepthFalloff, copied VERBATIM. A Gaussian of the depth difference normalized by the
//   depth window: f(0)=1 exactly, -> 0 across a large depth step (silhouette stop), monotone. ----
float DepthFalloff(float tapDepth, float centerDepth, float depthScale) {
    float ds = max(depthScale, 1e-6);
    float diff = (tapDepth - centerDepth) / ds;
    return exp(-(diff * diff));
}

float4 main(PSInput i) : SV_Target {
    float3 centerCol = gScene.Sample(gSmp, i.uv).rgb;

    // NO-OP early-out: zero strength or zero width -> pure pass-through (the byte-identical proof).
    if (HF_SP.sssStrength <= 0.0 || HF_SP.sssWidthPx <= 0.0) return float4(centerCol, 1.0);

    float4 sssC = gSss.Sample(gSSmp, i.uv);
    float centerMask = sssC.r;
    // A non-subsurface pixel (mask 0) passes through unchanged (keeps the lit/post path byte-identical).
    if (centerMask <= 0.0) return float4(centerCol, 1.0);

    float centerDepth = sssC.w;
    // Background / no surface (cleared G-buffer w == 0): nothing to diffuse.
    if (centerDepth <= 1e-4) return float4(centerCol, 1.0);

    int taps = (int)(HF_SP.taps + 0.5);
    if (taps < 1) taps = 1;
    int half = taps / 2;

    // The pass axis as a per-pixel UV delta (horizontal: x texel; vertical: y texel).
    float2 axisPx = (HF_SP.axis < 0.5) ? float2(HF_SP.texel.x, 0.0) : float2(0.0, HF_SP.texel.y);

    // Perspective-correct per-tap step in pixels (closer surface -> wider screen-space diffusion).
    float step = HF_SP.sssStrength * HF_SP.sssWidthPx / max(centerDepth, 1e-4);

    float3 sum = float3(0.0, 0.0, 0.0);
    float  wsum = 0.0;
    [loop] for (int t = -half; t <= half; ++t) {
        float distPx = step * (float)t;
        float2 tuv = i.uv + axisPx * distPx;
        float4 tapG = gSss.Sample(gSSmp, tuv);
        float tapDepth = tapG.w;
        float tapMask  = tapG.r;
        float w = DiffusionWeight(abs(distPx), HF_SP.sssWidthPx)
                * DepthFalloff(tapDepth, centerDepth, HF_SP.depthScale)
                * tapMask;
        sum  += w * gScene.Sample(gSmp, tuv).rgb;
        wsum += w;
    }

    // The center tap (t==0) always carries weight 1*1*centerMask>0, so wsum > 0; the guard is total.
    float3 c = (wsum > 1e-8) ? (sum / wsum) : centerCol;
    return float4(c, 1.0);
}
