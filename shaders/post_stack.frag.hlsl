// Data-driven post-process stack — the configurable composite chain (Slice BN). The FINAL pass for the
// --poststack-shot showcase. The resolved (linear HDR) scene is in gTex; an ORDERED effect list arrives
// via the push constant as a COUNT + a flat float STREAM. Each effect is `kind` followed by that kind's
// params; the shader walks the stream with a running cursor and applies each effect IN ORDER, with the
// SAME per-pixel math as engine/render/post_stack.h (the unit test pins that CPU side). This is a
// SEPARATE pass from post.frag — the default --shot post path is byte-identical and untouched.
//
// The flat-stream encoding keeps the whole push constant within the 128-byte minimum guaranteed
// maxPushConstantsSize (a vec4-PER-EFFECT array would blow past it on a 128-byte GPU). The stream is a
// float4[7] (28 floats = 112 bytes) + an int4 count (16 bytes) = 128 bytes; using float4 (not a scalar
// float array) also keeps the same 16-byte stride under the Metal cbuffer (std140) path. Param counts
// per kind: Tonemap 1 (exposure); ColorGrade 9 (lift.xyz,gamma.xyz,gain.xyz); ChromaticAberration 1
// (strength); Vignette 2 (outer,inner); FilmGrain 1 (intensity). Showcase total = 19 floats <= 28.
//
// Effect kinds (must match hf::render::post::Kind): 0 Tonemap, 1 ColorGrade, 2 ChromaticAberration,
// 3 Vignette, 4 FilmGrain. All deterministic + clock/RNG-free; FilmGrain hashes the INTEGER pixel coord
// with a fixed uint hash (bit-identical to the C++ GrainHash01), golden-stable across runs + backends.
[[vk::binding(0, 0)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 0)]] SamplerState gSmp : register(s0);

#define HF_STREAM_VEC4 7   // 7 float4 = 28 floats = 112 bytes; + int4 count = 128 bytes total

struct StackParams {
    int4   count;                  // .x = effect count; yzw pad
    float4 stream[HF_STREAM_VEC4]; // flat [kind, params...] per effect, in order (read via Stream())
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer StackPC { StackParams sp; };
#define HF_SP sp
#else
[[vk::push_constant]] struct { StackParams p; } pc;
#define HF_SP pc.p
#endif

// Read the i-th scalar from the float4 stream (4 scalars per float4).
float Stream(int idx) {
    float4 v = HF_SP.stream[idx >> 2];
    int lane = idx & 3;
    return lane == 0 ? v.x : (lane == 1 ? v.y : (lane == 2 ? v.z : v.w));
}

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

// --- Per-effect math — IDENTICAL to engine/render/post_stack.h ---------------------------------

float3 ApplyTonemap(float3 c, float exposure) {
    c *= exposure;
    float a = 2.51, b = 0.03, c2 = 2.43, d = 0.59, e = 0.14;
    c = saturate((c * (a * c + b)) / (c * (c2 * c + d) + e));   // ACES (Narkowicz)
    return pow(c, 1.0 / 2.2);                                   // gamma -> LDR
}

float3 ApplyColorGrade(float3 c, float3 lift, float3 gamma, float3 gain) {
    float3 base = max(c + lift, 0.0);
    float3 invG = float3(gamma.x != 0.0 ? 1.0 / gamma.x : 1.0,
                         gamma.y != 0.0 ? 1.0 / gamma.y : 1.0,
                         gamma.z != 0.0 ? 1.0 / gamma.z : 1.0);
    return gain * pow(base, invG);
}

float VignetteFactor(float2 uv, float outer, float inner) {
    return smoothstep(outer, inner, length(uv - 0.5));
}

// Radial UV offset (R outward, B inward), strength in pixels. At the exact center -> zero.
float2 ChromaticOffset(float2 uv, float strength, float2 texel) {
    float2 dir = uv - 0.5;
    float len = length(dir);
    if (len <= 1e-6) return float2(0.0, 0.0);
    return (dir / len) * strength * texel;
}
float3 ApplyChromaticAberration(float2 uv, float strength, float2 texel) {
    float2 off = ChromaticOffset(uv, strength, texel);
    float r = gTex.Sample(gSmp, uv + off).r;
    float g = gTex.Sample(gSmp, uv).g;
    float b = gTex.Sample(gSmp, uv - off).b;
    return float3(r, g, b);
}

// FilmGrain: fixed uint hash of the INTEGER pixel coord -> bit-identical to the C++ GrainHash01.
float GrainHash01(uint px, uint py) {
    uint h = px * 0x9E3779B1u + py * 0x85EBCA77u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return (float)(h >> 8) * (1.0 / 16777216.0);
}
float3 ApplyFilmGrain(float3 c, uint2 pix, float intensity) {
    return c + (GrainHash01(pix.x, pix.y) - 0.5) * intensity;
}

float4 main(PSInput i) : SV_Target {
    float w, h;
    gTex.GetDimensions(w, h);
    float2 texel = 1.0 / float2(w, h);
    uint2 pix = uint2(i.pos.xy);   // integer pixel coordinate (SV_Position at pixel centers)

    float3 c = gTex.Sample(gSmp, i.uv).rgb;

    int n = HF_SP.count.x;
    int cur = 0;                                   // running stream cursor
    [loop] for (int e = 0; e < n; ++e) {
        int k = (int)Stream(cur++);
        if (k == 0) {                              // Tonemap (1 param)
            c = ApplyTonemap(c, Stream(cur)); cur += 1;
        } else if (k == 1) {                       // ColorGrade (9 params)
            float3 lift  = float3(Stream(cur + 0), Stream(cur + 1), Stream(cur + 2));
            float3 gamma = float3(Stream(cur + 3), Stream(cur + 4), Stream(cur + 5));
            float3 gain  = float3(Stream(cur + 6), Stream(cur + 7), Stream(cur + 8));
            c = ApplyColorGrade(c, lift, gamma, gain); cur += 9;
        } else if (k == 2) {                       // ChromaticAberration (1 param)
            c = ApplyChromaticAberration(i.uv, Stream(cur), texel); cur += 1;
        } else if (k == 3) {                       // Vignette (2 params)
            c *= VignetteFactor(i.uv, Stream(cur + 0), Stream(cur + 1)); cur += 2;
        } else if (k == 4) {                       // FilmGrain (1 param)
            c = ApplyFilmGrain(c, pix, Stream(cur)); cur += 1;
        }
    }

    return float4(saturate(c), 1.0);
}
