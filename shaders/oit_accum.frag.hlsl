// Order-Independent Transparency — ACCUM-pass fragment shader (Slice CO, Weighted Blended OIT,
// McGuire & Bavoil 2013). One of the TWO weighted-blended targets: this pass writes the ACCUM target
// (RT0, RGBA16F) under an ADDITIVE blend (ONE, ONE) so the hardware performs
//     accum += float4(premultColor * weight,  alpha * weight)
// across ALL transparent fragments at a pixel, in ANY draw order (a commutative SUM). The companion
// oit_revealage.frag writes the REVEALAGE target under a multiplicative blend. A later fullscreen
// oit_resolve.frag composites accum.rgb/max(accum.a,eps) over the opaque scene by (1-revealage).
//
// (WHY TWO PASSES not one MRT: this engine's render graph drives ONE color attachment per pass — see
// engine/render/render_graph.cpp — so WBOIT's two outputs are split into two additive passes over the
// SAME transparent geometry rather than a single dual-attachment MRT. The math is IDENTICAL: accum is
// still a SUM, revealage still a PRODUCT, so order-independence holds. The accum + revealage targets
// are both RGBA16F render targets bound like the SSAO/SSR/G-buffer float targets.)
//
// THE WEIGHT (mirrors engine/render/oit.h oit::Weight VERBATIM): w = alpha * clamp(0.03 / (1e-5 +
// (viewDepth/200)^4), 1e-2, 3e3) — the McGuire depth weight, positive + finite, down-weighting distant
// fragments so a near transparent surface dominates the unsorted average. viewDepth is the view-space
// linear depth the vertex shader passed (= -vpos.z). NEW path behind --oit-shot; existing shaders +
// goldens UNTOUCHED.
struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float viewDepth : TEXCOORD0;                  // view-space linear depth
    [[vk::location(1)]] nointerpolation float4 colorAlpha : TEXCOORD1; // rgb = straight color, w = alpha
};

// oit::Weight, copied VERBATIM from engine/render/oit.h (the CPU unit test exercises the same form).
float OitWeight(float viewDepth, float alpha) {
    float a = clamp(alpha, 0.0, 1.0);
    float zd = viewDepth / 200.0;
    float zd2 = zd * zd;
    float zd4 = zd2 * zd2;                       // (viewDepth/200)^4, grows with depth
    float depthTerm = 0.03 / (1e-5 + zd4);       // shrinks as depth grows; +1e-5 floors the denom
    depthTerm = clamp(depthTerm, 1e-2, 3e3);     // bounded, finite, positive
    return a * depthTerm;
}

float4 main(PSInput i) : SV_Target {
    float  alpha = saturate(i.colorAlpha.w);
    float3 straight = i.colorAlpha.rgb;
    float  w = OitWeight(i.viewDepth, alpha);
    // Premultiplied color * weight in rgb; alpha * weight in w. The additive (ONE,ONE) blend turns
    // these into the running sums accum.rgb += premult*w and accum.a += alpha*w (oit::Accumulate).
    float3 premult = straight * alpha;
    return float4(premult * w, alpha * w);
}
