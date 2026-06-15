// Slice BA — Text / HUD fragment shader. Samples the baked-font atlas (BuildFontAtlas: white RGB,
// coverage in alpha) at the material set 0 slot (binding 0 = sampled image, binding 1 = sampler —
// the same layout the UI/material BindTexture uses when the pipeline has no per-frame set), uses the
// atlas alpha as glyph COVERAGE, and outputs a CONFIGURABLE text color modulated by that coverage.
// The pipeline is alpha-blended (src_alpha / one_minus_src_alpha), so the transparent background
// (coverage 0) leaves the scene untouched and the ink (coverage 1) draws the text color over it.
//
// The text color arrives as a FRAGMENT-stage push constant. The Vulkan/DXC path uses a real
// [[vk::push_constant]]; the MSL-gen path (glslang ignores it) declares an equivalent cbuffer at
// [[vk::binding(1,0)]] so spirv-cross --msl-decoration-binding lands it on the engine's flat
// fragment push-constant buffer slot (kFbPushConst=1) — the SAME convention the bloom passes use.
[[vk::binding(0, 0)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 0)]] SamplerState gSmp : register(s0);

struct TextParams { float4 color; };  // rgb = text color, a = overall opacity multiplier
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer TextPC { TextParams tp; };
#define HF_TP tp
#else
[[vk::push_constant]] struct { TextParams p; } pc;
#define HF_TP pc.p
#endif

struct PSInput {
    float4 pos : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

float4 main(PSInput i) : SV_Target {
    float coverage = gTex.Sample(gSmp, i.uv).a;
    return float4(HF_TP.color.rgb, coverage * HF_TP.color.a);
}
