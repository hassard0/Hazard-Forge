// SSAO blur pass (Slice Y). A 4x4 box blur over the raw AO target to remove the per-pixel kernel
// noise introduced by the tiled rotation noise, yielding smooth contact AO. Fullscreen; reads the AO
// texture (R channel) and the source texel size from the push constant.
[[vk::binding(0, 0)]] Texture2D    gAO  : register(t0);
[[vk::binding(1, 0)]] SamplerState gSmp : register(s0);

struct BlurParams { float2 texel; float2 pad; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer BlurPC { BlurParams bp; };
#define HF_BP bp
#else
[[vk::push_constant]] struct { BlurParams p; } pc;
#define HF_BP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

float4 main(PSInput i) : SV_Target {
    // 4x4 box blur centered on the texel (offsets -1.5..+1.5).
    float sum = 0.0;
    [unroll] for (int y = -2; y < 2; ++y)
    [unroll] for (int x = -2; x < 2; ++x) {
        float2 off = (float2(x, y) + 0.5) * HF_BP.texel;
        sum += gAO.Sample(gSmp, i.uv + off).r;
    }
    float ao = sum / 16.0;
    return float4(ao, ao, ao, 1.0);
}
