#pragma once
// Volumetric cloud math (Slice CH) — pure CPU (header-only, no device, no backend symbols). Shared by
// the --clouds-shot showcase AND tests/clouds_test.cpp AND shaders/clouds.frag.hlsl, so the unit test
// exercises the SAME deterministic noise + FBM + slab-density + Beer-Lambert + Henyey-Greenstein math
// the in-shader raymarch uses. Mirrors the ssr.h/water.h shared-math pattern.
//
// A raymarched cumulus LAYER lives between two altitudes (a "slab"): slabBottom..slabTop. The density
// at a world point is a fractal-noise (FBM) field, SHAPED by:
//   * a HEIGHT GRADIENT that fades the density to 0 at the slab's bottom and top and peaks in the
//     middle (so clouds are puffy blobs floating inside the layer, not a solid wall),
//   * a COVERAGE THRESHOLD `max(fbm - coverage, 0)` that carves the continuous noise into discrete
//     clouds with clear sky between them (raising coverage => less/thinner cloud), and
//   * a fixed-time WIND advection (the sample point is offset by t*wind) so the field can be animated,
//     but the showcase + goldens sample at a single FIXED time so they are byte-stable.
//
// DETERMINISM: the noise is a pure function of position (an INTEGER-LATTICE HASH + trilinear interp,
// NO RNG, NO clock), so the same (worldPos, t) always yields the same density. Two runs are
// bit-identical, and the CPU test and the HLSL shader (which copies this math verbatim) agree.
//
// --- The hash (documented) ---
// Hash3 takes the integer lattice cell (ix,iy,iz) to a pseudo-random value in [0,1) via the classic
// sine-fract hash: n = dot(cell, (a,b,c)); h = frac(sin(n) * M). The exact constants are fixed below
// (kHashA/B/C, kHashM) so the CPU and shader produce identical bits. Noise3 then trilinearly
// interpolates the eight surrounding lattice hashes with a smoothstep (Hermite) fade, giving a smooth
// value-noise in [0,1]. (Sine-fract hashing is the standard, portable, RNG-free choice for shader/CPU
// shared procedural noise; quality is sufficient for a cumulus field and it is exactly reproducible.)

#include "math/math.h"
#include <cmath>

namespace hf::render::clouds {

inline constexpr float kPi = 3.14159265358979323846f;

// --- Slab + showcase constants (documented; shared CPU/shader so goldens are reproducible). ---------
// The cloud layer altitudes (world Y). Chosen well above the scene so the clouds read as a distant
// sky layer over the lit scene.
inline constexpr float kSlabBottom = 18.0f;
inline constexpr float kSlabTop    = 34.0f;
// The FIXED sample time the showcase + goldens use (documented constant; no clock/RNG). Non-zero so
// the wind-advected field is sampled mid-animation. Two runs at this t are byte-identical.
inline constexpr float kFixedTime  = 2.0f;
// Default coverage threshold: noise below this is clear sky; above it becomes cloud. Tuned so the slab
// has distinct puffy cumulus with open sky between (not overcast, not empty).
inline constexpr float kCoverage   = 0.42f;
// Noise feature scale: world distance is multiplied by this before sampling the lattice (smaller =>
// larger, fluffier clouds). Documented constant shared with the shader.
inline constexpr float kNoiseScale = 0.06f;
// Wind: the (x,z) offset per unit time the field is advected by (t*wind). Fixed direction/speed.
inline constexpr float kWindX = 0.8f;
inline constexpr float kWindZ = 0.25f;
// FBM parameters (fixed lacunarity/gain). Gain 0.5 with the leading 0.5 amplitude keeps the sum in
// [0,1] (geometric series 0.5 + 0.25 + ... < 1).
inline constexpr int   kFbmOctaves   = 5;
inline constexpr float kFbmLacunarity = 2.0f;
inline constexpr float kFbmGain       = 0.5f;

// --- The hash constants (fixed so CPU + shader bits agree). -----------------------------------------
inline constexpr float kHashA = 12.9898f;
inline constexpr float kHashB = 78.233f;
inline constexpr float kHashC = 37.719f;
inline constexpr float kHashM = 43758.5453f;

// frac(x) = x - floor(x) (matches HLSL frac).
inline float Frac(float x) { return x - std::floor(x); }

// Hash an INTEGER lattice cell to a pseudo-random value in [0,1). Pure function, no RNG.
inline float Hash3(float ix, float iy, float iz) {
    float n = ix * kHashA + iy * kHashB + iz * kHashC;
    return Frac(std::sin(n) * kHashM);
}

// Smoothstep (Hermite) fade 3t^2 - 2t^3 — the standard value-noise interpolant; matches HLSL.
inline float Fade(float t) { return t * t * (3.0f - 2.0f * t); }

// Deterministic value-noise in [0,1]: trilinear interpolation of the eight surrounding lattice hashes
// with a smoothstep fade. Pure function of p (no RNG), so it is bit-reproducible and matches the
// shader's Noise3. At an integer lattice point it returns that corner's hash exactly.
inline float Noise3(const math::Vec3& p) {
    float fx = std::floor(p.x), fy = std::floor(p.y), fz = std::floor(p.z);
    float tx = Fade(p.x - fx), ty = Fade(p.y - fy), tz = Fade(p.z - fz);

    // Eight corner hashes of the unit cell.
    float c000 = Hash3(fx,        fy,        fz);
    float c100 = Hash3(fx + 1.0f, fy,        fz);
    float c010 = Hash3(fx,        fy + 1.0f, fz);
    float c110 = Hash3(fx + 1.0f, fy + 1.0f, fz);
    float c001 = Hash3(fx,        fy,        fz + 1.0f);
    float c101 = Hash3(fx + 1.0f, fy,        fz + 1.0f);
    float c011 = Hash3(fx,        fy + 1.0f, fz + 1.0f);
    float c111 = Hash3(fx + 1.0f, fy + 1.0f, fz + 1.0f);

    // Trilinear blend.
    float x00 = c000 + (c100 - c000) * tx;
    float x10 = c010 + (c110 - c010) * tx;
    float x01 = c001 + (c101 - c001) * tx;
    float x11 = c011 + (c111 - c011) * tx;
    float y0 = x00 + (x10 - x00) * ty;
    float y1 = x01 + (x11 - x01) * ty;
    return y0 + (y1 - y0) * tz;
}

// Fractal sum of Noise3 (a few octaves at fixed lacunarity/gain). The amplitude starts at gain and
// halves each octave so the total stays within [0,1] (sum of 0.5+0.25+... < 1). Deterministic.
inline float Fbm(const math::Vec3& p, int octaves) {
    float sum = 0.0f;
    float amp = kFbmGain;
    float freq = 1.0f;
    math::Vec3 q = p;
    for (int o = 0; o < octaves; ++o) {
        sum += amp * Noise3(q * freq);
        freq *= kFbmLacunarity;
        amp  *= kFbmGain;
    }
    return sum;
}

// Height gradient inside the slab: 0 at slabBottom and slabTop, peaking (1) in the middle. A simple
// "round" profile h*(1-h) normalized to [0,1] (4*h*(1-h)) so cumulus are fat in the middle of the
// layer and feather out at the top/bottom. Outside [bottom,top] this is clamped to 0 by the caller.
inline float HeightGradient(float y, float slabBottom, float slabTop) {
    if (y <= slabBottom || y >= slabTop) return 0.0f;
    float h = (y - slabBottom) / (slabTop - slabBottom);   // 0..1 across the slab
    return 4.0f * h * (1.0f - h);                          // 0 at edges, 1 at the middle
}

// The cloud density at a world point + time, with an EXPLICIT coverage threshold. Returns 0 outside
// the slab. The FBM is advected by t*wind, carved by the coverage threshold (max(fbm-coverage,0)),
// and shaped by the height gradient. Deterministic.
inline float DensityCoverage(const math::Vec3& worldPos, float t, float slabBottom, float slabTop,
                             float coverage) {
    if (worldPos.y <= slabBottom || worldPos.y >= slabTop) return 0.0f;
    // Advect by the wind (offset the sample point), then scale into noise space.
    math::Vec3 sp{(worldPos.x + t * kWindX) * kNoiseScale,
                  worldPos.y * kNoiseScale,
                  (worldPos.z + t * kWindZ) * kNoiseScale};
    float fbm = Fbm(sp, kFbmOctaves);
    float carved = fbm - coverage;
    if (carved <= 0.0f) return 0.0f;             // below the coverage threshold -> clear sky
    return carved * HeightGradient(worldPos.y, slabBottom, slabTop);
}

// The cloud density using the documented DEFAULT coverage (kCoverage). Convenience overload used by
// the showcase + tests; the shader passes coverage as a push-constant.
inline float Density(const math::Vec3& worldPos, float t, float slabBottom, float slabTop) {
    return DensityCoverage(worldPos, t, slabBottom, slabTop, kCoverage);
}

// Beer-Lambert extinction: transmittance after an optical depth. Beer(0)=1, monotone decreasing to 0.
inline float Beer(float opticalDepth) { return std::exp(-opticalDepth); }

// --- Cloud shadow on the ground/scene (Slice CK) ----------------------------------------------------
// The fraction of DIRECT SUNLIGHT that reaches `worldPos` after passing through the cloud slab — i.e.
// the sun's transmittance to that surface point. 1 = full sun (no cloud blocks the ray), 0 = fully
// shadowed (the sun is behind opaque cloud). Multiply the directional-light DIRECT term by this to cast
// dappled cloud shadows on the scene; ambient/IBL/point lights are unaffected (clouds only block the
// sun). Shared CPU/shader math (lit_cloudshadow.frag mirrors it verbatim) + unit-tested.
//
// CONVENTION: `sunDir` is the DIRECTIONAL-LIGHT direction — the direction the sunlight TRAVELS (same as
// FrameData.lightDir). The direction FROM the surface TOWARD the sun is therefore `-sunDir`. We march
// from `worldPos` along `-sunDir` toward the sun, clipped to the cloud slab [slabBottom, slabTop],
// accumulating optical depth = sum(Density * stepLen) over `steps` uniform samples, and return
// Beer(opticalDepth). DETERMINISTIC: fixed `t`, fixed `steps`, the CH integer-lattice hash noise (no
// RNG/clock) -> two runs bit-identical and the CPU test + the shader agree.
//
// If the toward-sun ray never crosses the slab (e.g. pointing away/parallel, or the slab is entirely
// behind the start), there is no cloud along the ray -> optical depth 0 -> full sun (returns 1).
inline float CloudShadow(const math::Vec3& worldPos, const math::Vec3& sunDir, float t,
                         float slabBottom, float slabTop, int steps) {
    // Direction from the surface toward the sun (opposite the light's travel direction).
    math::Vec3 toSun = math::normalize(sunDir * -1.0f);

    // Intersect the toward-sun ray with the two horizontal slab planes (y = slabBottom, y = slabTop).
    // A near-horizontal ray (|toSun.y| ~ 0) never meaningfully enters/leaves the slab.
    if (std::fabs(toSun.y) <= 1e-4f) return 1.0f;
    float t0 = (slabBottom - worldPos.y) / toSun.y;
    float t1 = (slabTop    - worldPos.y) / toSun.y;
    float tEnter = std::fmin(t0, t1);
    float tExit  = std::fmax(t0, t1);
    tEnter = std::fmax(tEnter, 0.0f);          // never march behind the surface (toward the ground)
    if (tExit <= tEnter) return 1.0f;          // the slab is not in front along the toward-sun ray

    int   n = (steps > 0) ? steps : 1;
    float stepLen = (tExit - tEnter) / (float)n;
    float opticalDepth = 0.0f;
    for (int s = 0; s < n; ++s) {
        float ts = tEnter + ((float)s + 0.5f) * stepLen;
        math::Vec3 p = worldPos + toSun * ts;
        opticalDepth += Density(p, t, slabBottom, slabTop) * stepLen;
    }
    return Beer(opticalDepth);
}

// Henyey-Greenstein phase function. cosAngle is between the view ray and the light direction; g>0
// forward-scatters (peaks at cosAngle=1) so looking toward the sun through the cloud glows. g=0 is
// isotropic 1/(4*pi). Matches the volumetric fog HG (engine/render/volumetric.h) form.
inline float HenyeyGreenstein(float cosAngle, float g) {
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosAngle;
    if (denom < 1e-6f) denom = 1e-6f;
    return (1.0f - g2) / (4.0f * kPi * denom * std::sqrt(denom));
}

} // namespace hf::render::clouds
