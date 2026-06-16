// Order-Independent Transparency — REVEALAGE-pass fragment shader (Slice CO, Weighted Blended OIT).
// The companion of oit_accum.frag: this pass writes the REVEALAGE target (a single R/RGBA16F target,
// CLEARED to 1.0) under the WBOIT multiplicative blend (oitRevealageBlend == src*ZERO + dst*(1 -
// srcAlpha)), so the hardware performs
//     revealage *= (1 - alpha)
// across ALL transparent fragments at a pixel, in ANY draw order (a commutative PRODUCT). After the
// pass revealage == Π(1 - alpha_i) ∈ [0,1] — how much of the opaque background still shows through the
// stacked transparent layers. The resolve composites (1 - revealage) as the total transparent
// coverage. Reuses oit_accum.vert (so the SAME transparent geometry rasterizes identically); only the
// per-draw alpha (push-constant colorAlpha.w) is used here.
//
// The blend factors emit `1 - srcAlpha` against the destination, so the fragment's OUTPUT alpha must be
// the straight `alpha` (the blend reads srcAlpha). We write `alpha` to ALL channels so the revealage
// product accumulates identically whether the target is read as .r or .a downstream. NEW path behind
// --oit-shot; existing shaders + goldens UNTOUCHED. Mirrors engine/render/oit.h (revealage *= 1-alpha
// inside oit::Accumulate).
struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float viewDepth : TEXCOORD0;                  // (unused here; kept for the shared vert)
    [[vk::location(1)]] nointerpolation float4 colorAlpha : TEXCOORD1; // w = straight alpha
};

float4 main(PSInput i) : SV_Target {
    float alpha = saturate(i.colorAlpha.w);
    // src alpha = `alpha`; the oitRevealageBlend (dst *= 1 - srcAlpha) multiplies the revealage target
    // by (1 - alpha). The src color factor is ZERO, so the rgb output is irrelevant to the result; we
    // still emit `alpha` in every channel for a well-defined, backend-identical source.
    return float4(alpha, alpha, alpha, alpha);
}
