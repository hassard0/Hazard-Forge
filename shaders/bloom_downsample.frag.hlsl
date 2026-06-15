// HDR bloom — downsample (Slice U). The 13-tap "dual filter" (Jimenez / COD Advanced Warfare) box
// downsample: 13 bilinear taps arranged as a center, an inner 2x2 ring, and an outer 3x3 ring,
// weighted so the result is a smooth, firefly-resistant half-res reduction. Used to build the bloom
// mip chain (each pass halves resolution, accumulating progressively wider blur radius).
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

float3 T(float2 uv) { return gTex.Sample(gSmp, uv).rgb; }

float4 main(PSInput i) : SV_Target {
    // `texel` is the SOURCE (finer mip) texel size. Sample one source-texel out for the inner ring
    // and two out for the outer ring, exactly the COD 13-tap layout.
    float2 t = HF_BP.texel;
    float2 uv = i.uv;

    // Center.
    float3 c = T(uv);
    // Inner 2x2 ring (at +/- 1 source texel diagonally), weight 0.5 total -> 0.125 each.
    float3 inner = T(uv + float2(-1, -1) * t) + T(uv + float2( 1, -1) * t)
                 + T(uv + float2(-1,  1) * t) + T(uv + float2( 1,  1) * t);
    // Outer ring: corners (+/-2), edge-midpoints (+/-2 on one axis), and the center-cross (+/-2).
    float3 outerCorners = T(uv + float2(-2, -2) * t) + T(uv + float2( 2, -2) * t)
                        + T(uv + float2(-2,  2) * t) + T(uv + float2( 2,  2) * t);
    float3 outerCross   = T(uv + float2(-2,  0) * t) + T(uv + float2( 2,  0) * t)
                        + T(uv + float2( 0, -2) * t) + T(uv + float2( 0,  2) * t);

    // COD weights: center 0.125, inner ring 0.5 (0.125 each), outer corners 0.125 (0.03125 each),
    // outer cross 0.25 (two contributions share each edge in the original; approximated here).
    float3 result = c * 0.125
                  + inner * 0.125
                  + outerCorners * 0.03125
                  + outerCross * 0.0625;
    return float4(result, 1.0);
}
