// HDR bloom — upsample + combine (Slice U). Reads TWO inputs: the COARSER accumulated bloom mip
// (gTex, binding 0/1) and this level's DOWNSAMPLE mip (gTex2, binding 3/4). It applies a 3x3 tent
// (Hanning) blur to the coarser mip while upsampling it to this (finer) resolution, then ADDS the
// finer downsample mip, writing the sum to a fresh "up" RT. Accumulating two-input combine (rather
// than additive blend onto an existing RT) keeps each pass a clean clear-then-write — no read-modify-
// write hazard and no reliance on load-preserve. `texel` is the COARSE (gTex) mip texel size;
// `strength` scales the coarser contribution as it propagates up the chain.
[[vk::binding(0, 0)]] Texture2D    gTex  : register(t0);   // coarser accumulated bloom (to upsample)
[[vk::binding(1, 0)]] SamplerState gSmp  : register(s0);
[[vk::binding(3, 0)]] Texture2D    gTex2 : register(t3);   // this level's downsample mip (to add)
[[vk::binding(4, 0)]] SamplerState gSmp2 : register(s3);

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
    // 3x3 tent kernel (weights 1 2 1 / 2 4 2 / 1 2 1, sum 16) on the coarser mip, sampled at +/- one
    // coarse texel for a smooth upsample.
    float2 t = HF_BP.texel;
    float2 uv = i.uv;
    float3 coarse =
        1.0 * T(uv + float2(-1, -1) * t) + 2.0 * T(uv + float2(0, -1) * t) + 1.0 * T(uv + float2(1, -1) * t) +
        2.0 * T(uv + float2(-1,  0) * t) + 4.0 * T(uv + float2(0,  0) * t) + 2.0 * T(uv + float2(1,  0) * t) +
        1.0 * T(uv + float2(-1,  1) * t) + 2.0 * T(uv + float2(0,  1) * t) + 1.0 * T(uv + float2(1,  1) * t);
    coarse *= (1.0 / 16.0);

    float3 fine = gTex2.Sample(gSmp2, uv).rgb;   // this level's own downsample mip
    // Combine: this level's energy + the (strength-scaled) upsampled coarser bloom.
    return float4(fine + coarse * HF_BP.strength, 1.0);
}
