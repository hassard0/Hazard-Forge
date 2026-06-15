// HDR bloom — bright-pass prefilter (Slice U). Samples the full-res HDR scene and keeps only the
// energy above a soft luma threshold (Karis/COD knee curve), writing the bright part into a
// half-res HDR target. The scene is exposure-scaled FIRST so the threshold lives in the same domain
// the composite pass tonemaps (exposure 1.7, matching post.frag), making the knee meaningful
// against the displayed highlights. Output is LINEAR HDR — no tonemap here.
//
// Params arrive as a fragment-stage push constant. The Vulkan/DXC path uses a real [[vk::push_constant]];
// the MSL-gen path (glslang ignores it) declares an equivalent cbuffer at [[vk::binding(1,0)]] so
// spirv-cross --msl-decoration-binding lands it on the engine's flat fragment buffer slot (kFbPushConst=1).
[[vk::binding(0, 0)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 0)]] SamplerState gSmp : register(s0);

struct BloomParams { float2 texel; float threshold; float knee; float strength; float intensity; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer BloomPC { BloomParams bp; };
#define HF_BP bp
#else
[[vk::push_constant]] struct { BloomParams p; } pc;
#define HF_BP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };
static const float3 kLuma = float3(0.2126, 0.7152, 0.0722);

float4 main(PSInput i) : SV_Target {
    // 4-tap bilinear box (half-texel offsets) to average the 2x2 footprint when downsampling to
    // half res — cheaper than a 13-tap here and the bright pass tolerates the softer kernel.
    float2 t = HF_BP.texel;
    float3 c = gTex.Sample(gSmp, i.uv + float2(-1, -1) * t * 0.5).rgb
             + gTex.Sample(gSmp, i.uv + float2( 1, -1) * t * 0.5).rgb
             + gTex.Sample(gSmp, i.uv + float2(-1,  1) * t * 0.5).rgb
             + gTex.Sample(gSmp, i.uv + float2( 1,  1) * t * 0.5).rgb;
    c *= 0.25;

    // Match the composite/post exposure so the threshold is meaningful against displayed highlights.
    c *= HF_BP.intensity;   // intensity = exposure (1.7)

    // Soft-knee bright pass (Karis). Keep the excess above `threshold` with a quadratic knee of
    // width `knee` so the cutoff is smooth (no hard banding ring around bright objects).
    float br = max(c.r, max(c.g, c.b));
    float kn = HF_BP.knee;
    float soft = br - HF_BP.threshold + kn;
    soft = clamp(soft, 0.0, 2.0 * kn);
    soft = soft * soft / (4.0 * kn + 1e-5);
    float contribution = max(soft, br - HF_BP.threshold) / max(br, 1e-5);
    return float4(c * contribution, 1.0);
}
