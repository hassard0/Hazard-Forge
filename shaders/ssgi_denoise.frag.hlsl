// Screen-space global illumination — bilateral (edge-preserving) DENOISE pass (Slice BR). The
// single-frame SSGI gather (ssgi.frag, K=16 rays/pixel) is noisy. This fullscreen pass smooths the
// SSGI indirect buffer with a depth+normal-guided BILATERAL blur so the floor color-bleed pool reads
// smooth while the red/green bleed stays CRISP at surface boundaries (the blur does not cross a
// geometry edge). It is the SAME shape as ssao_blur.frag (a fullscreen NxN blur of a screen buffer)
// but each tap's box weight is replaced by an EDGE-STOPPING weight from the G-buffer.
//
// KERNEL CHOICE (documented): a SINGLE (2*radius+1)x(2*radius+1) box of taps (NOT separable H/V).
// Rationale: a true 2D bilateral weight does not factor across the two axes (the depth/normal
// edge-stop is per-tap, not a product of two 1D Gaussians), so a single 2D pass is the correct
// edge-preserving result and needs no intermediate target — exactly the ssao_blur single-pass shape.
// radius is small (2 -> 5x5) so the cost is bounded. The math (BilateralWeight) is mirrored VERBATIM
// from engine/render/ssgi.h so the CPU unit test exercises the identical weighting.
//
// Bindings mirror ssgi_composite/ssr: the SSGI indirect buffer at t0/s0, the g-buffer (view normal.xyz
// + linear depth.w, written by gbuffer.frag) at t3/s3 (BindTexturePair). The denoised indirect is
// written to a new SSGI RT; the EXISTING ssgi_composite then ADDS denoisedIndirect over the scene.
// ssgi.frag/ssgi_composite/ssr/ssao shaders + their goldens are UNTOUCHED — this is an inserted pass
// on the SSGI path only, behind the --ssgi-denoise-shot showcase. Single frame, NO RNG/time ->
// deterministic, two runs byte-identical.
[[vk::binding(0, 0)]] Texture2D    gSsgi  : register(t0);   // raw SSGI indirect diffuse (rgb); a = 1
[[vk::binding(1, 0)]] SamplerState gSmp   : register(s0);
[[vk::binding(3, 0)]] Texture2D    gGbuf  : register(t3);   // view normal.xyz + linDepth.w
[[vk::binding(4, 0)]] SamplerState gGSmp  : register(s3);

struct DenoiseParams {
    float2 texel;        // 1/size
    float  radius;       // kernel half-width in pixels (as a float; clamped to [0, kMaxRadius])
    float  spatialSigma; // Gaussian spatial falloff (pixels)
    float  depthSigma;   // edge-stop sigma in view-space linear-depth units
    float  normalPower;  // exponent on max(dot(nC,nT),0): higher = sharper normal edge-stop
    float2 pad;
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer DenoisePC { DenoiseParams dp; };
#define HF_DP dp
#else
[[vk::push_constant]] struct { DenoiseParams p; } pc;
#define HF_DP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

static const int kMaxRadius = 4;   // hard upper bound on the kernel half-width (loop cap)

// Edge-stopping tap weight — IDENTICAL to engine/render/ssgi.h BilateralWeight:
//   spatial = exp(-spatialDist2 / (2*spatialSigma^2))
//   depth   = exp(-(depthTap-depthCenter)^2 / (2*depthSigma^2))
//   normal  = pow(max(dot(nC,nT),0), normalPower)
//   weight  = spatial * depth * normal
// A tap across a large depth step or an opposing normal gets ~0 weight (edge preserved).
float BilateralWeight(float spatialDist2, float spatialSigma,
                      float depthCenter, float depthTap, float depthSigma,
                      float3 nCenter, float3 nTap, float normalPower) {
    float ss = max(spatialSigma, 1e-6);
    float ds = max(depthSigma, 1e-6);
    float spatial = exp(-spatialDist2 / (2.0 * ss * ss));
    float dDiff   = depthTap - depthCenter;
    float depthW  = exp(-(dDiff * dDiff) / (2.0 * ds * ds));
    float nd      = max(dot(nCenter, nTap), 0.0);
    float normalW = pow(nd, normalPower);
    return spatial * depthW * normalW;
}

float4 main(PSInput i) : SV_Target {
    float4 gC = gGbuf.Sample(gGSmp, i.uv);
    float  depthC = gC.w;
    // Background / no surface (g-buffer cleared w == 0): pass the raw indirect through unchanged so the
    // denoise is a no-op outside geometry (keeps the raw path identical for background pixels).
    if (depthC <= 0.0001) return gSsgi.Sample(gSmp, i.uv);

    float3 nC = normalize(gC.xyz);
    int    R  = (int)clamp(HF_DP.radius, 0.0, (float)kMaxRadius);

    float3 sum  = float3(0.0, 0.0, 0.0);
    float  wsum = 0.0;

    [loop] for (int dy = -kMaxRadius; dy <= kMaxRadius; ++dy) {
        if (dy < -R || dy > R) continue;
        [loop] for (int dx = -kMaxRadius; dx <= kMaxRadius; ++dx) {
            if (dx < -R || dx > R) continue;
            float2 off = float2((float)dx, (float)dy) * HF_DP.texel;
            float2 tuv = i.uv + off;

            float4 gT = gGbuf.Sample(gGSmp, tuv);
            float  depthT = gT.w;
            // A tap on background contributes nothing (no surface to share indirect with).
            if (depthT <= 0.0001) continue;
            float3 nT = normalize(gT.xyz);

            float spatialDist2 = (float)(dx * dx + dy * dy);
            float w = BilateralWeight(spatialDist2, HF_DP.spatialSigma,
                                      depthC, depthT, HF_DP.depthSigma,
                                      nC, nT, HF_DP.normalPower);
            sum  += w * gSsgi.Sample(gSmp, tuv).rgb;
            wsum += w;
        }
    }

    // Normalize by the accumulated weight (the center tap always contributes weight 1, so wsum > 0).
    float3 denoised = (wsum > 1e-6) ? (sum / wsum) : gSsgi.Sample(gSmp, i.uv).rgb;
    return float4(denoised, 1.0);
}
