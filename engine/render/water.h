#pragma once
// Water rendering math (Slice CF) — pure CPU (header-only, no device, no backend symbols). Shared by
// the --water-shot showcase AND tests/water_test.cpp AND shaders/water.{vert,frag}.hlsl, so the unit
// test exercises the SAME Gerstner displacement + analytic normal + fresnel + depth-tint math the
// in-shader vertex displacement and fragment shading use. Mirrors the ssr.h/ssgi.h shared-math pattern.
//
// A Gerstner (trochoidal) wave moves surface points in BOTH the horizontal (along the wave direction)
// and vertical axes, producing the sharp-crest/round-trough profile of real water (vs a plain sine
// height field). The surface position for one wave with unit direction D=(Dx,Dz), amplitude A,
// wavelength L, steepness Q and speed S is:
//
//   k     = 2*pi / L                          (spatial angular frequency)
//   w     = S * k                             (temporal angular frequency; phase speed S)
//   theta = k*(D.x*x + D.z*z) - w*t           (phase at (x,z) and time t)
//   P(x,z,t) = ( x + (Q*A) * D.x * cos(theta),     // horizontal x displacement (crest pinch)
//                A * sin(theta),                    // vertical displacement (height)
//                z + (Q*A) * D.z * cos(theta) )     // horizontal z displacement
//
// Summing over a small fixed wave set gives a rich, deterministic surface. The reference-grid point is
// (x,0,z); Displace returns the DISPLACEMENT (P - (x,0,z)) so callers add it to the base grid vertex.
//
// The analytic normal comes from the exact partial derivatives of the summed surface w.r.t. the grid
// parameters x and z (the Gerstner tangent dP/dx and bitangent dP/dz), N = normalize(cross(B, T)) —
// NOT a finite difference. For a single wave the standard closed form of the surface partials is:
//
//   dP/dx = ( 1 - Q*A*Dx*Dx*k*sin(theta),   A*Dx*k*cos(theta),   -Q*A*Dx*Dz*k*sin(theta) )
//   dP/dz = ( -Q*A*Dx*Dz*k*sin(theta),       A*Dz*k*cos(theta),    1 - Q*A*Dz*Dz*k*sin(theta) )
//
// (these are the term-by-term derivatives of P above; summed over the wave set they give the tangent
// and bitangent of the summed surface). The normal is cross(dP/dz, dP/dx) normalized so a flat,
// zero-amplitude surface yields exactly +Y. The water_test cross-checks this analytic normal against a
// finite difference of Displace to prove the derivative is correct.

#include "math/math.h"
#include <cmath>
#include <cstddef>

namespace hf::render::water {

// One Gerstner wave. `dir` is the (assumed-unit) horizontal propagation direction in the XZ plane;
// `amplitude` the wave height; `wavelength` the crest-to-crest distance; `steepness` Q in [0,1] (0 =
// rolling sine, ~1 = sharp Gerstner crest); `speed` the phase speed (so w = speed * k).
struct GerstnerWave {
    math::Vec2 dir{1.0f, 0.0f};
    float amplitude = 0.0f;
    float wavelength = 1.0f;
    float steepness = 0.0f;
    float speed = 0.0f;
};

inline constexpr float kPi = 3.14159265358979323846f;

// FIXED wave time the showcase + goldens sample at (documented constant; no clock/RNG). Two runs at
// this t are byte-identical. Chosen non-zero so the surface is mid-animation (visibly rippled).
inline constexpr float kFixedTime = 1.3f;

// The DISPLACEMENT of a single Gerstner wave at grid parameter (x,z) and time t (P - basePoint).
inline math::Vec3 DisplaceWave(float x, float z, float t, const GerstnerWave& wv) {
    float k = (wv.wavelength > 1e-6f) ? (2.0f * kPi / wv.wavelength) : 0.0f;
    float w = wv.speed * k;
    float theta = k * (wv.dir.x * x + wv.dir.y * z) - w * t;
    float c = std::cos(theta);
    float s = std::sin(theta);
    float qa = wv.steepness * wv.amplitude;
    return math::Vec3{qa * wv.dir.x * c, wv.amplitude * s, qa * wv.dir.y * c};
}

// Summed displacement over a wave set (P - basePoint). Add to (x,0,z) for the displaced surface point.
inline math::Vec3 Displace(float x, float z, float t, const GerstnerWave* waves, std::size_t n) {
    math::Vec3 d{0.0f, 0.0f, 0.0f};
    for (std::size_t i = 0; i < n; ++i) d = d + DisplaceWave(x, z, t, waves[i]);
    return d;
}

// ANALYTIC surface normal from the summed exact partial derivatives (Gerstner tangent x bitangent),
// NOT a finite difference. Flat (zero amplitude) -> +Y. Always unit length.
inline math::Vec3 Normal(float x, float z, float t, const GerstnerWave* waves, std::size_t n) {
    // Tangent T = dP/dx, Bitangent B = dP/dz of the summed surface. Start from the identity base
    // (the undisplaced grid's dP/dx = (1,0,0), dP/dz = (0,0,1)) and add each wave's derivative terms.
    math::Vec3 T{1.0f, 0.0f, 0.0f};   // dP/dx
    math::Vec3 B{0.0f, 0.0f, 1.0f};   // dP/dz
    for (std::size_t i = 0; i < n; ++i) {
        const GerstnerWave& wv = waves[i];
        float k = (wv.wavelength > 1e-6f) ? (2.0f * kPi / wv.wavelength) : 0.0f;
        float w = wv.speed * k;
        float theta = k * (wv.dir.x * x + wv.dir.y * z) - w * t;
        float c = std::cos(theta);
        float sgn = std::sin(theta);
        float A = wv.amplitude;
        float qa = wv.steepness * wv.amplitude;
        float Dx = wv.dir.x, Dz = wv.dir.y;
        // dP/dx
        T.x += -qa * Dx * Dx * k * sgn;
        T.y +=  A  * Dx * k * c;
        T.z += -qa * Dx * Dz * k * sgn;
        // dP/dz
        B.x += -qa * Dx * Dz * k * sgn;
        B.y +=  A  * Dz * k * c;
        B.z += -qa * Dz * Dz * k * sgn;
    }
    // N = cross(B, T) so a flat surface (T=(1,0,0), B=(0,0,1)) gives cross((0,0,1),(1,0,0)) = (0,1,0).
    return math::normalize(math::cross(B, T));
}

// Schlick fresnel: reflectance at a view angle whose cosine to the normal is NdotV. f0 is the
// base (head-on) reflectance (~0.02 for water). Fresnel(1,f0)=f0, Fresnel(0,f0)=1, monotone in
// (1-NdotV). NdotV is clamped to [0,1] so out-of-range inputs stay well-defined.
inline float Fresnel(float NdotV, float f0) {
    float c = NdotV < 0.0f ? 0.0f : (NdotV > 1.0f ? 1.0f : NdotV);
    float m = 1.0f - c;
    float m5 = m * m * m * m * m;
    return f0 + (1.0f - f0) * m5;
}

// Depth-based water tint (Beer-Lambert-ish): at depth 0 below the surface the color is `shallow`; as
// the optical depth grows the color attenuates toward `deep`. t = 1 - exp(-absorption*depth) is the
// fraction absorbed, so out = lerp(shallow, deep, t): depth 0 -> shallow, large depth -> ~deep,
// monotone increasing toward deep with depth. `depth` is clamped to >= 0.
inline math::Vec3 RefractTint(float depth, const math::Vec3& shallow, const math::Vec3& deep,
                              float absorption) {
    float d = depth < 0.0f ? 0.0f : depth;
    float tf = 1.0f - std::exp(-absorption * d);
    return math::Vec3{
        shallow.x + (deep.x - shallow.x) * tf,
        shallow.y + (deep.y - shallow.y) * tf,
        shallow.z + (deep.z - shallow.z) * tf,
    };
}

// The FIXED showcase wave set (3 waves, crossing directions). Documented constant; deterministic.
// Returned by value so callers/tests/showcase all use the identical parameters.
struct WaveSet {
    GerstnerWave waves[3];
    std::size_t count = 3;
};

inline WaveSet ShowcaseWaves() {
    WaveSet ws;
    // Three crossing waves: a long primary swell + two shorter chop waves at angles, decreasing
    // amplitude/wavelength so the sum has both a broad roll and fine ripple.
    // Directions are pre-normalized literals (kept as constants so CPU + shader agree bit-for-bit).
    ws.waves[0] = GerstnerWave{math::Vec2{1.0f, 0.0f},               0.22f, 4.0f, 0.55f, 1.1f};
    ws.waves[1] = GerstnerWave{math::Vec2{0.8f, 0.6f},              0.13f, 2.3f, 0.65f, 1.4f};
    ws.waves[2] = GerstnerWave{math::Vec2{-0.50702014f, 0.86193424f}, 0.07f, 1.3f, 0.75f, 1.7f};
    ws.count = 3;
    return ws;
}

} // namespace hf::render::water
