// Slice CC — CPU particle / VFX emitter billboard fragment shader. Computes a PROCEDURAL soft round
// sprite from the corner UV (like the decal's in-shader texture — no asset, no extra texture bind),
// modulated by the per-particle color the vertex stage forwarded. The pipeline is ADDITIVE-blended
// (src*1 + dst*1, order-independent — see below) with depth-test ON / depth-write OFF, so the
// particles glow over the lit + shadowed opaque scene without needing a back-to-front sort.
//
// BLEND DECISION (documented): ADDITIVE (one/one) is chosen over sorted back-to-front over-blend
// because it is ORDER-INDEPENDENT — the CPU emitter retires/swap-removes particles every step, so the
// live-pool order is not view-sorted; additive blending makes the result independent of draw order
// (a fountain/plume of glowing embers reads correctly without per-frame depth sorting). The frag
// emits color*alpha so the soft-sprite alpha falloff still shapes each particle's contribution.
struct PSInput {
    float4 pos   : SV_Position;
    [[vk::location(0)]] float2 uv    : TEXCOORD0;
    [[vk::location(1)]] float4 color : COLOR;
};

// Soft round sprite coverage from the [0,1]^2 corner UV: 1 at the center, smoothly fading to 0 at
// radius 0.5 (the quad edge). Deterministic; no texture fetch.
float SpriteAlpha(float2 uv) {
    float2 c = uv - 0.5;                  // centered [-0.5, 0.5]
    float r = length(c) * 2.0;            // 0 at center, 1 at the inscribed circle edge
    return smoothstep(1.0, 0.0, r);       // soft round falloff
}

float4 main(PSInput i) : SV_Target {
    float a = SpriteAlpha(i.uv) * i.color.a;     // sprite falloff * per-particle alpha
    // Premultiply the color by the coverage alpha so additive (one/one) blending sums each particle's
    // shaped contribution; the alpha channel carries the same so an over-blend pipeline would also work.
    return float4(i.color.rgb * a, a);
}
