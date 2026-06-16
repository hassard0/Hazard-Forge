#pragma once
// Data-driven post-process stack — pure CPU config + orchestration + per-effect math (Slice BN).
// No device, no backend symbols. Shared by the --poststack-shot showcase AND tests/post_stack_test.cpp
// so the unit test exercises the SAME per-pixel math the in-shader stack (shaders/post_stack.frag.hlsl)
// applies, exactly as engine/render/ssr.h is shared with the SSR pass + tests/ssr_test.cpp and
// engine/render/decal.h with the decal pass.
//
// Model: a PostStack is an ORDERED list of PostEffect, each a Kind + a small param block. The renderer
// (and the CPU mirror here) applies the enabled effects IN ORDER to the resolved scene color. The
// per-effect evaluators are deterministic (clock/RNG-free): ColorGrade is a per-channel lift/gamma/gain
// curve, ChromaticAberration radially offsets the R/B sample UVs, FilmGrain adds a FIXED integer-hash of
// the INTEGER pixel coordinate (no time/frame -> golden-stable). Tonemap (ACES) + Vignette mirror the
// existing post.frag so the stack composes the engine's established look. The stack config is data
// (JSON-authorable via LoadPostStack) so an agent can compose looks.
//
// Shader parity note: FilmGrain's hash is an unsigned 32-bit integer hash with well-defined wraparound
// so the C++ and HLSL evaluations are bit-for-bit equivalent; ColorGrade/ChromaticAberration/Vignette
// use the same float math the shader uses. The CPU evaluators take a small Sampler callback for the
// UV-sampling effects (ChromaticAberration) so the math is testable without a GPU.

#include "math/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace hf::render::post {

// The effect kinds, in their canonical short-name spelling (used by the JSON loader + the showcase line).
enum class Kind {
    Tonemap,             // ACES filmic tonemap (Narkowicz) + gamma to displayed LDR.
    ColorGrade,          // per-channel lift/gamma/gain.
    ChromaticAberration, // radial RGB split, strength in pixels.
    Vignette,            // darken toward the corners.
    FilmGrain,           // fixed integer-hash luminance noise.
};

// Per-effect parameters. Only the fields meaningful for a given Kind are used; the rest stay at their
// neutral defaults (so a default-constructed PostEffect of any Kind is the identity for its unused
// fields). Defaults are the neutral/identity values.
struct PostEffect {
    Kind kind = Kind::Tonemap;

    // ColorGrade: out = gain * pow(max(in + lift, 0), 1/gamma), per channel. Identity = (0,1,1).
    math::Vec3 lift{0.0f, 0.0f, 0.0f};
    math::Vec3 gamma{1.0f, 1.0f, 1.0f};
    math::Vec3 gain{1.0f, 1.0f, 1.0f};

    // ChromaticAberration: radial RGB-split strength in PIXELS at the screen edge.
    float strength = 0.0f;

    // Vignette: smoothstep(outer, inner, radius) darkening (inner < outer). Defaults match post.frag.
    float vignetteOuter = 0.8f;
    float vignetteInner = 0.35f;

    // FilmGrain: additive luminance-noise amplitude (the +/-0.5 hash is scaled by this).
    float intensity = 0.0f;

    // Tonemap: linear exposure applied before the ACES curve (matches post.frag's kExposure).
    float exposure = 1.0f;
};

struct PostStack {
    std::vector<PostEffect> effects;   // applied IN ORDER. Empty == pass-through.
    bool empty() const { return effects.empty(); }
    size_t size() const { return effects.size(); }
};

// --- Per-effect math (shared with shaders/post_stack.frag.hlsl) ---------------------------------

static const math::Vec3 kLuma{0.299f, 0.587f, 0.114f};
inline float Luma(const math::Vec3& c) { return math::dot(c, kLuma); }
inline float Clamp01(float x) { return std::min(std::max(x, 0.0f), 1.0f); }
inline math::Vec3 Saturate(const math::Vec3& c) {
    return {Clamp01(c.x), Clamp01(c.y), Clamp01(c.z)};
}
inline float Smoothstep(float a, float b, float x) {
    float t = Clamp01((x - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
}

// ACES filmic tonemap (Narkowicz) + gamma -> displayed LDR. IDENTICAL to post.frag's ACES+gamma, with
// the linear exposure folded in first.
inline math::Vec3 ApplyTonemap(const math::Vec3& cIn, float exposure) {
    auto aces = [](float x) {
        const float a = 2.51f, b = 0.03f, c2 = 2.43f, d = 0.59f, e = 0.14f;
        return Clamp01((x * (a * x + b)) / (x * (c2 * x + d) + e));
    };
    math::Vec3 c{cIn.x * exposure, cIn.y * exposure, cIn.z * exposure};
    c = {aces(c.x), aces(c.y), aces(c.z)};
    const float invG = 1.0f / 2.2f;
    return {std::pow(c.x, invG), std::pow(c.y, invG), std::pow(c.z, invG)};  // gamma -> LDR
}

// ColorGrade: out = gain * pow(max(in + lift, 0), 1/gamma), per channel. Identity (lift0/gamma1/gain1)
// returns the input unchanged.
inline math::Vec3 ApplyColorGrade(const math::Vec3& c, const math::Vec3& lift,
                                  const math::Vec3& gamma, const math::Vec3& gain) {
    auto ch = [](float in, float lf, float gm, float gn) {
        float base = std::max(in + lf, 0.0f);
        float invG = (gm != 0.0f) ? (1.0f / gm) : 1.0f;
        return gn * std::pow(base, invG);
    };
    return {ch(c.x, lift.x, gamma.x, gain.x),
            ch(c.y, lift.y, gamma.y, gain.y),
            ch(c.z, lift.z, gamma.z, gain.z)};
}

// Vignette: multiply by smoothstep(outer, inner, |uv-0.5|) — darkens the corners, 1.0 at center.
inline float VignetteFactor(const math::Vec2& uv, float outer, float inner) {
    float dx = uv.x - 0.5f, dy = uv.y - 0.5f;
    float r = std::sqrt(dx * dx + dy * dy);
    return Smoothstep(outer, inner, r);
}
inline math::Vec3 ApplyVignette(const math::Vec3& c, const math::Vec2& uv,
                                float outer, float inner) {
    float v = VignetteFactor(uv, outer, inner);
    return {c.x * v, c.y * v, c.z * v};
}

// ChromaticAberration: the radial offset (in UV) for a given pixel — sample R outward, B inward along
// the direction from screen center, scaled by `strength` pixels. Returns the per-channel UV offset
// magnitude as a 2D vector (dir * strength * texel); the shader samples R at uv+off, B at uv-off, G at
// uv. At the exact center (uv == 0.5) the radial direction is zero -> NO aberration.
inline math::Vec2 ChromaticOffset(const math::Vec2& uv, float strength, const math::Vec2& texel) {
    math::Vec2 dir{uv.x - 0.5f, uv.y - 0.5f};
    float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
    if (len <= 1e-6f) return {0.0f, 0.0f};       // center: radial direction undefined -> no offset
    math::Vec2 n{dir.x / len, dir.y / len};
    return {n.x * strength * texel.x, n.y * strength * texel.y};
}
// CPU evaluator: given a sampler (uv -> color), produce the aberrated color. R from uv+off, G from uv,
// B from uv-off. With strength 0 (or at center) all three sample uv -> identical to no aberration.
inline math::Vec3 ApplyChromaticAberration(
    const math::Vec2& uv, float strength, const math::Vec2& texel,
    const std::function<math::Vec3(math::Vec2)>& sample) {
    math::Vec2 off = ChromaticOffset(uv, strength, texel);
    math::Vec3 r = sample({uv.x + off.x, uv.y + off.y});
    math::Vec3 g = sample(uv);
    math::Vec3 b = sample({uv.x - off.x, uv.y - off.y});
    return {r.x, g.y, b.z};
}

// FilmGrain: a FIXED unsigned-32-bit integer hash of the INTEGER pixel coordinate. No time/frame input
// -> deterministic + golden-stable, and bit-for-bit reproducible in HLSL (same uint wraparound math).
// Returns a value in [0,1).
inline float GrainHash01(uint32_t px, uint32_t py) {
    // A small integer hash (xor-shift / multiply mix). All ops are 32-bit unsigned with defined
    // wraparound, so HLSL's uint arithmetic produces the identical bit pattern.
    uint32_t h = px * 0x9E3779B1u + py * 0x85EBCA77u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    // Map the top 24 bits to [0,1) so the float is exact + identical across compilers.
    return (float)(h >> 8) * (1.0f / 16777216.0f);
}
// The signed, bounded grain value for a pixel: (hash - 0.5) * intensity, in [-0.5*intensity, 0.5*intensity].
inline float Grain(uint32_t px, uint32_t py, float intensity) {
    return (GrainHash01(px, py) - 0.5f) * intensity;
}
inline math::Vec3 ApplyFilmGrain(const math::Vec3& c, uint32_t px, uint32_t py, float intensity) {
    float g = Grain(px, py, intensity);
    return {c.x + g, c.y + g, c.z + g};
}

// --- Stack evaluation (CPU mirror) --------------------------------------------------------------
// Apply the full stack to ONE pixel, IN ORDER. `uv` is the pixel's screen UV in [0,1]; `texel` is
// 1/resolution; (px,py) the integer pixel coord; `sample` returns the scene color at an arbitrary UV
// (used by ChromaticAberration). The first sampled color is `sample(uv)`. The returned color is NOT
// auto-saturated (the caller / shader saturates for the LDR write) so the math is exact for tests.
inline math::Vec3 ApplyStack(const PostStack& stack, const math::Vec2& uv, const math::Vec2& texel,
                             uint32_t px, uint32_t py,
                             const std::function<math::Vec3(math::Vec2)>& sample) {
    math::Vec3 c = sample(uv);
    for (const PostEffect& e : stack.effects) {
        switch (e.kind) {
            case Kind::Tonemap:
                c = ApplyTonemap(c, e.exposure);
                break;
            case Kind::ColorGrade:
                c = ApplyColorGrade(c, e.lift, e.gamma, e.gain);
                break;
            case Kind::ChromaticAberration:
                c = ApplyChromaticAberration(uv, e.strength, texel, sample);
                break;
            case Kind::Vignette:
                c = ApplyVignette(c, uv, e.vignetteOuter, e.vignetteInner);
                break;
            case Kind::FilmGrain:
                c = ApplyFilmGrain(c, px, py, e.intensity);
                break;
        }
    }
    return c;
}

// --- Config: kind <-> short name + JSON loader + default stack -----------------------------------

inline const char* KindName(Kind k) {
    switch (k) {
        case Kind::Tonemap:             return "tonemap";
        case Kind::ColorGrade:          return "colorgrade";
        case Kind::ChromaticAberration: return "chromatic";
        case Kind::Vignette:            return "vignette";
        case Kind::FilmGrain:           return "grain";
    }
    return "tonemap";
}

// --- JSON loader (implemented in post_stack.cpp; uses third_party/json) -------------------------
struct LoadResult {
    bool ok = false;
    std::string error;
    PostStack stack;
    explicit operator bool() const { return ok; }
};

// Parse a PostStack from a JSON string. An absent/empty "effects" array yields a pass-through (empty)
// stack; order is preserved; an unknown effect "kind" fails the load (ok=false, error set).
LoadResult LoadPostStack(const std::string& json);

// The FIXED showcase stack: a cinematic chain over the resolved scene. Tonemap (exposure 1.7, matching
// the engine's established look) -> ColorGrade (a warm teal-orange grade: lift shadows cool, lift gains
// warm) -> ChromaticAberration (subtle) -> Vignette -> FilmGrain (subtle). Deterministic; the
// --poststack-shot showcase and the Metal visual_test --poststack render this identical config.
inline PostStack DefaultShowcaseStack() {
    PostStack s;
    PostEffect tonemap; tonemap.kind = Kind::Tonemap; tonemap.exposure = 1.7f;
    PostEffect grade;   grade.kind = Kind::ColorGrade;
    grade.lift  = {-0.02f, 0.0f, 0.03f};   // cool the shadows slightly
    grade.gamma = {1.0f, 1.0f, 1.0f};
    grade.gain  = {1.10f, 1.02f, 0.90f};   // warm/orange highlights, pull blue
    PostEffect chroma;  chroma.kind = Kind::ChromaticAberration; chroma.strength = 2.0f;
    PostEffect vig;     vig.kind = Kind::Vignette; vig.vignetteOuter = 0.8f; vig.vignetteInner = 0.35f;
    PostEffect grain;   grain.kind = Kind::FilmGrain; grain.intensity = 0.05f;
    s.effects = {tonemap, grade, chroma, vig, grain};
    return s;
}

}  // namespace hf::render::post
