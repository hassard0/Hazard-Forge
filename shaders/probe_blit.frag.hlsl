// Slice AK — probe reflection BLIT. Copies one reflection tile from the bake's reflection RT into the
// matching tile of the FINAL probe atlas. Run per tile via SetViewport (fullscreen triangle -> uv in
// [0,1] over the destination tile); the push constant gives the SOURCE tile's atlas-UV origin + size
// in the reflection RT so we sample exactly that tile. A plain passthrough — no tonemap, linear HDR.
[[vk::binding(0, 0)]] Texture2D    gSrc    : register(t0);
[[vk::binding(1, 0)]] SamplerState gSrcSmp : register(s0);

struct BlitParams { float4 srcRect; };  // xy = tile origin (atlas UV), zw = tile size (atlas UV)
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer BlitPC { BlitParams bp; };
#define HF_BP bp
#else
[[vk::push_constant]] struct { BlitParams p; } pc;
#define HF_BP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

float4 main(PSInput i) : SV_Target {
    float2 src = HF_BP.srcRect.xy + i.uv * HF_BP.srcRect.zw;
    return float4(gSrc.SampleLevel(gSrcSmp, src, 0.0).rgb, 1.0);
}
