// Order-Independent Transparency — RESOLVE fragment shader (Slice CO, Weighted Blended OIT). A
// fullscreen pass (reuses post.vert) reading the two weighted-blended targets — the ACCUM target (t0:
// RGBA16F, Σ premultColor*w in rgb, Σ alpha*w in a) and the REVEALAGE target (t3: Π(1-alpha) in r,
// bound via BindTexturePair) — and producing the WBOIT resolved TRANSPARENT layer per oit::ResolveOver:
//     transparentRGB = accum.rgb / max(accum.a, 1e-4)            // weighted-average straight color
//     coverage       = 1 - revealage                            // total transparent coverage
//     out            = float4(transparentRGB, coverage)
// It writes this transparent layer into a dedicated RT (rgb = resolved transparent color, a = coverage).
// A FINAL composite pass (the SAME generic lerp+tonemap as water_composite.frag) then blends
// lerp(opaqueScene, transparentRGB, coverage) == transparentRGB*(1-revealage) + opaqueScene*revealage
// — EXACTLY oit::ResolveOver(accum, revealage, opaqueScene). Decomposing the resolve into "produce the
// transparent layer" + "lerp it over the opaque scene by coverage" keeps every pass a 2-input fullscreen
// pass (the engine's pair-bind path), mirroring the water RT -> water_composite structure.
//
// ORDER INDEPENDENCE: accum + revealage are a SUM + PRODUCT over the transparent fragments, so they are
// IDENTICAL for any draw order -> this resolve produces a BYTE-IDENTICAL transparent layer (hence a
// byte-identical final image) regardless of the order the transparent set was drawn (the showcase
// asserts permuted==canonical SHA). NEW path behind --oit-shot; existing shaders + goldens UNTOUCHED.
// Mirrors engine/render/oit.h oit::ResolveOver.
[[vk::binding(0, 0)]] Texture2D    gAccum     : register(t0);   // Σ premultColor*w (rgb), Σ alpha*w (a)
[[vk::binding(1, 0)]] SamplerState gSmp       : register(s0);
[[vk::binding(3, 0)]] Texture2D    gRevealage : register(t3);   // Π(1-alpha) in .r
[[vk::binding(4, 0)]] SamplerState gRSmp      : register(s3);

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

float4 main(PSInput i) : SV_Target {
    float4 accum     = gAccum.Sample(gSmp, i.uv);
    float  revealage = gRevealage.Sample(gRSmp, i.uv).r;

    // transparentRGB = accum.rgb / max(accum.a, eps) — the weighted-average straight color. The eps
    // guards the divide where no/near-zero-alpha fragments landed (then revealage ~ 1 so coverage ~ 0
    // and the lerp keeps the opaque scene anyway).
    float3 transparentRGB = accum.rgb / max(accum.a, 1e-4);
    float  coverage = 1.0 - revealage;             // 1 - Π(1-alpha)

    // Emit (transparentRGB, coverage); the final composite lerps it over the opaque scene by coverage
    // == oit::ResolveOver. coverage 0 (no transparent fragment) leaves the plain opaque scene.
    return float4(transparentRGB, saturate(coverage));
}
