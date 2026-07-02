#pragma once
// Physical atmospheric scattering (Slice AT1, Track-S S8) — pure CPU (header-only, no device, no
// backend symbols). Shared by the --at1-sky-shot showcase AND tests/atmosphere_test.cpp AND
// shaders/atmosphere.frag.hlsl, so the unit test exercises the SAME deterministic Rayleigh + Mie
// single-scattering integral the fullscreen sky pass evaluates. Mirrors the clouds.h/ssr.h/water.h
// shared-math discipline (same constants, same loop counts, mirrored verbatim in-shader).
//
// THE MODEL — the standard real-time single-scattering formulation (Nishita-class):
//   * A spherical planet of radius kPlanetRadius under an atmosphere shell out to kAtmosphereRadius.
//   * Two exponential density profiles: Rayleigh (air molecules) exp(-h/Hr) with Hr ~ 8 km, and Mie
//     (aerosols/haze) exp(-h/Hm) with Hm ~ 1.2 km (haze hugs the ground — the horizon murk).
//   * Wavelength-dependent Rayleigh scattering coefficients at sea level,
//     betaR = (5.8e-6, 13.5e-6, 33.1e-6) per meter for (~680, ~550, ~440 nm) — blue scatters ~5.7x
//     more than red: the BLUE zenith, and the sunset REDDENING (blue is scattered OUT of the long
//     grazing sun path, leaving the transmitted disc/halo red).
//   * A grey Mie coefficient betaM = 21e-6 /m with a Henyey-Greenstein phase (g = 0.76, strongly
//     forward) — the bright halo around the sun; Mie EXTINCTION is betaM * 1.1 (absorption ~10%).
//   * Phases: Rayleigh 3/(16*pi)*(1+cos^2), Mie = HG (matches clouds.h HenyeyGreenstein's form).
//
// SkyColor(viewDir, sunDir): from a ground-level eye (kEyeHeight above the planet surface at the
// north pole of the sphere), intersect the view ray with the atmosphere shell (clamped to the ground
// where the ray hits the planet), then integrate kViewSamples segments along the view ray; at each
// sample take kSunSamples segments along the ray TOWARD the sun for the sun-path optical depth. The
// per-sample transmittance exp(-(betaR*(DrView+DrSun) + betaM*1.1*(DmView+DmSun))) weights the
// accumulated Rayleigh/Mie in-scatter; the result is HDR radiance (sun radiance kSunIntensity),
// tonemap-ready. CONVENTION: `sunDir` points TOWARD the sun (NOT the light-travel direction clouds.h
// uses — documented here because the two suites differ).
//
// DETERMINISM (the render suite's float class, like clouds/water): FIXED sample counts, fixed
// documented constants, pure functions of (viewDir, sunDir) — no clock, no RNG. Two evaluations are
// bit-identical, and the CPU test + the HLSL shader (which copies this math verbatim) agree by
// construction.
//
// CROSS-COMPILER BIT-EXACTNESS (MSVC == clang): std::exp is NOT stable across compilers here — MSVC
// /O2 auto-vectorizes exp calls to the vector CRT (__vdecl_expf4), which differs from clang's scalar
// ucrt expf by up to ~2e-5 relative over this integral (measured). So the CPU core uses DetExp — a
// deterministic exp twin (range-reduce + degree-6 polynomial + exact ldexp scale; pure float
// mul/add, no libm, no FMA on the SSE2 baseline) — and the direction helpers compute their trig in
// DOUBLE then cast to float (a double-libm last-ULP wiggle is far below float resolution). Result:
// the digest hashes RAW FLOAT BITS and is bit-identical MSVC vs clang — stronger than a quantization
// tolerance. The SHADER mirror uses hardware exp/sin/cos (the usual CPU<->GPU float-render-class
// tolerance, same as clouds/water); the mirrored structure, constants and loop counts are identical.
//
// HONEST LIMITS (single scattering, documented rather than fudged):
//   * NO multiple scattering: twilight (sun below the horizon) and shadowed sky are too dark, and
//     the sky away from the sun at low sun is dimmer than reality.
//   * NO ozone absorption: the zenith at low sun is slightly less blue than the real sky.
//   * NO ground albedo bounce: rays that hit the planet return only the in-scattered path, so the
//     below-horizon band is near-black.
//   * The HG Mie phase (per the slice spec) is a mild simplification of Cornette-Shanks.

#include "math/math.h"
#include <cmath>
#include <cstdint>
#include <cstring>

namespace hf::render::atmo {

inline constexpr float kPi = 3.14159265358979323846f;

// --- Physical constants (documented; shared CPU/shader so the goldens are reproducible). ------------
inline constexpr float kPlanetRadius     = 6360e3f;   // m — planet radius R
inline constexpr float kAtmosphereRadius = 6420e3f;   // m — top of atmosphere Ra (60 km shell)
inline constexpr float kHr = 7994.0f;                 // m — Rayleigh density scale height (~8 km)
inline constexpr float kHm = 1200.0f;                 // m — Mie density scale height (haze layer)
// Rayleigh scattering coefficients at sea level, per meter, for (red ~680nm, green ~550nm, blue ~440nm).
inline constexpr float kBetaR_R = 5.8e-6f;
inline constexpr float kBetaR_G = 13.5e-6f;
inline constexpr float kBetaR_B = 33.1e-6f;
inline constexpr float kBetaM   = 21e-6f;             // grey Mie scattering coefficient, per meter
inline constexpr float kMieExtinction = 1.1f;         // Mie extinction = kBetaM * this (absorption ~10%)
inline constexpr float kMieG    = 0.76f;              // HG asymmetry — strong forward scatter (sun halo)
inline constexpr float kSunIntensity = 20.0f;         // sun radiance scale (HDR; tonemapped by the shot)
inline constexpr float kEyeHeight = 2.0f;             // m — the ground-level eye above the surface

// FIXED integration sample counts (deterministic; the shader uses the same literals).
inline constexpr int kViewSamples = 16;               // N segments along the view ray
inline constexpr int kSunSamples  = 8;                // M segments along each toward-sun ray

// --- Showcase constants (the --at1-sky-shot 3-panel geometry; shared CPU/shader). -------------------
// Sun elevations for the three panels: noon / low / sunset. The reddening progression.
inline constexpr float kSunElevNoonRad   = 1.221730f; // 70 deg
inline constexpr float kSunElevLowRad    = 0.261799f; // 15 deg
inline constexpr float kSunElevSunsetRad = 0.034907f; //  2 deg
// Each panel is a panorama: horizontal = azimuth centered on the sun's azimuth (span 200 deg),
// vertical = view elevation 0 (horizon, bottom) .. 89 deg (near-zenith, top).
inline constexpr float kPanelAzSpanRad   = 3.490659f; // 200 deg
inline constexpr float kPanelElevMaxRad  = 1.553343f; //  89 deg
inline constexpr float kShotExposure     = 1.4f;      // tonemap exposure for the LDR shot

// Deterministic float exp — bit-identical across MSVC/clang (see the header comment). Standard
// range reduction exp(x) = 2^n * exp(r): n = round(x * log2(e)), r = x - n*ln2 (split hi/lo so r is
// accurate), exp(r) by a degree-6 Taylor polynomial (|r| <= ~0.347 -> relative error ~1e-7, at the
// libm-ULP level), then an EXACT ldexp power-of-two scale. Only float mul/add/ldexp — every step is
// IEEE-exact or identically rounded on both compilers (SSE2 baseline, no FMA contraction).
inline float DetExp(float x) {
    if (x < -87.0f) return 0.0f;        // below float range -> 0 (transmittance limit)
    if (x > 88.0f)  x = 88.0f;          // clamp (never hit by this model's inputs)
    float n = std::floor(x * 1.44269504f + 0.5f);
    const float kLn2Hi = 0.693359375f;         // ln2 split so n*kLn2Hi is exact for |n| < 2^15
    const float kLn2Lo = -2.12194440e-4f;
    float r = (x - n * kLn2Hi) - n * kLn2Lo;
    // exp(r) = 1 + r + r^2/2 + ... + r^6/720 (Horner).
    float p = 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    return std::ldexp(p, (int)n);
}

// The sun sits at azimuth 0 = the -Z direction; elevation e above the horizon. Trig in DOUBLE then
// cast to float (cross-compiler determinism — see the header comment).
inline math::Vec3 SunDirFromElevation(float elevRad) {
    return {0.0f, (float)std::sin((double)elevRad), (float)-std::cos((double)elevRad)};
}

// Panorama view direction for panel-local (u01, v01): u01 in [0,1] maps azimuth across
// kPanelAzSpanRad centered on the sun azimuth; v01 in [0,1] maps elevation 0 (horizon) ..
// kPanelElevMaxRad (near-zenith). Unit length by construction. Trig in double (see above).
inline math::Vec3 PanoramaViewDir(float u01, float v01) {
    double az = ((double)u01 - 0.5) * (double)kPanelAzSpanRad;
    double el = (double)v01 * (double)kPanelElevMaxRad;
    double ce = std::cos(el);
    return {(float)(std::sin(az) * ce), (float)std::sin(el), (float)(-std::cos(az) * ce)};
}

// --- Phase functions ---------------------------------------------------------------------------------
// Rayleigh phase 3/(16*pi) * (1 + cos^2 theta). Normalized over the sphere.
inline float RayleighPhase(float cosTheta) {
    return 3.0f / (16.0f * kPi) * (1.0f + cosTheta * cosTheta);
}

// Henyey-Greenstein Mie phase (same closed form as clouds.h HenyeyGreenstein; restated here so the
// atmosphere header is self-contained and the shader mirror is one file). g > 0 forward-scatters.
inline float MiePhase(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    if (denom < 1e-6f) denom = 1e-6f;
    return (1.0f - g2) / (4.0f * kPi * denom * std::sqrt(denom));
}

// --- Ray / sphere (center at the origin) --------------------------------------------------------------
// FAR intersection t of the unit ray (ro, rd) with the sphere |p| = radius; negative if none.
// For an eye inside the atmosphere shell this is the shell EXIT distance.
inline float RaySphereFar(const math::Vec3& ro, const math::Vec3& rd, float radius) {
    float b = math::dot(ro, rd);
    float c = math::dot(ro, ro) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return -1.0f;
    return -b + std::sqrt(disc);
}

// NEAR intersection t (the first hit along +t); negative if none or behind the origin.
// Used to clamp the view ray at the planet surface.
inline float RaySphereNear(const math::Vec3& ro, const math::Vec3& rd, float radius) {
    float b = math::dot(ro, rd);
    float c = math::dot(ro, ro) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return -1.0f;
    return -b - std::sqrt(disc);
}

// --- Optical depth + transmittance --------------------------------------------------------------------
// Integrate the Rayleigh and Mie optical depths (density * length) along the ray (ro, rd) over
// [0, tMax] with kViewSamples midpoint segments. Pure, deterministic.
inline void OpticalDepth(const math::Vec3& ro, const math::Vec3& rd, float tMax,
                         float& odR, float& odM) {
    odR = 0.0f; odM = 0.0f;
    float seg = tMax / (float)kViewSamples;
    for (int i = 0; i < kViewSamples; ++i) {
        float t = ((float)i + 0.5f) * seg;
        math::Vec3 p = ro + rd * t;
        float h = math::length(p) - kPlanetRadius;
        if (h < 0.0f) h = 0.0f;
        odR += DetExp(-h / kHr) * seg;
        odM += DetExp(-h / kHm) * seg;
    }
}

// RGB transmittance from the ground-level eye along viewDir for a path length tPath (clamped to the
// atmosphere exit): exp(-(betaR*odR + betaM*1.1*odM)). Monotonically non-increasing in tPath (the
// pinned physics invariant: a longer path extincts more).
inline math::Vec3 Transmittance(const math::Vec3& viewDirIn, float tPath) {
    math::Vec3 viewDir = math::normalize(viewDirIn);
    math::Vec3 ro{0.0f, kPlanetRadius + kEyeHeight, 0.0f};
    float tExit = RaySphereFar(ro, viewDir, kAtmosphereRadius);
    float tMax = (tPath < tExit) ? tPath : tExit;
    if (tMax <= 0.0f) return {1.0f, 1.0f, 1.0f};
    float odR, odM;
    OpticalDepth(ro, viewDir, tMax, odR, odM);
    float me = kBetaM * kMieExtinction * odM;
    return {DetExp(-(kBetaR_R * odR + me)),
            DetExp(-(kBetaR_G * odR + me)),
            DetExp(-(kBetaR_B * odR + me))};
}

// --- The single-scattering sky ------------------------------------------------------------------------
// HDR sky radiance for a view ray from the ground-level eye, sun toward `sunDir` (unit, TOWARD the
// sun). kViewSamples midpoint segments along the view ray x kSunSamples midpoint segments along each
// toward-sun ray. Samples whose sun ray passes below the surface are in the planet's shadow and
// contribute nothing (the horizon darkens opposite a setting sun). Deterministic; mirrored verbatim
// in shaders/atmosphere.frag.hlsl.
inline math::Vec3 SkyColor(const math::Vec3& viewDirIn, const math::Vec3& sunDirIn) {
    math::Vec3 viewDir = math::normalize(viewDirIn);
    math::Vec3 sunDir  = math::normalize(sunDirIn);
    math::Vec3 ro{0.0f, kPlanetRadius + kEyeHeight, 0.0f};

    float tMax = RaySphereFar(ro, viewDir, kAtmosphereRadius);
    if (tMax <= 0.0f) return {0.0f, 0.0f, 0.0f};
    // Clamp at the planet surface where the ray hits the ground (below-horizon rays).
    float tGround = RaySphereNear(ro, viewDir, kPlanetRadius);
    if (tGround > 0.0f && tGround < tMax) tMax = tGround;

    float cosTheta = math::dot(viewDir, sunDir);
    float phR = RayleighPhase(cosTheta);
    float phM = MiePhase(cosTheta, kMieG);

    float seg = tMax / (float)kViewSamples;
    float odR = 0.0f, odM = 0.0f;             // view-path optical depths, accumulated front-to-back
    float sumR_r = 0.0f, sumR_g = 0.0f, sumR_b = 0.0f;   // Rayleigh in-scatter accumulators
    float sumM_r = 0.0f, sumM_g = 0.0f, sumM_b = 0.0f;   // Mie in-scatter accumulators

    for (int i = 0; i < kViewSamples; ++i) {
        float t = ((float)i + 0.5f) * seg;
        math::Vec3 p = ro + viewDir * t;
        float h = math::length(p) - kPlanetRadius;
        if (h < 0.0f) h = 0.0f;
        float hr = DetExp(-h / kHr) * seg;   // Rayleigh mass in this segment
        float hm = DetExp(-h / kHm) * seg;   // Mie mass in this segment
        odR += hr;
        odM += hm;

        // Optical depth along the ray toward the sun (to the shell exit).
        float tSun = RaySphereFar(p, sunDir, kAtmosphereRadius);
        if (tSun <= 0.0f) continue;
        float segS = tSun / (float)kSunSamples;
        float odRs = 0.0f, odMs = 0.0f;
        bool shadowed = false;
        for (int j = 0; j < kSunSamples; ++j) {
            float ts = ((float)j + 0.5f) * segS;
            math::Vec3 ps = p + sunDir * ts;
            float hs = math::length(ps) - kPlanetRadius;
            if (hs < 0.0f) { shadowed = true; break; }   // sun ray dips below the surface
            odRs += DetExp(-hs / kHr) * segS;
            odMs += DetExp(-hs / kHm) * segS;
        }
        if (shadowed) continue;

        // Transmittance eye -> sample -> sun (Rayleigh per channel + grey Mie extinction).
        float me = kBetaM * kMieExtinction * (odM + odMs);
        float attR = DetExp(-(kBetaR_R * (odR + odRs) + me));
        float attG = DetExp(-(kBetaR_G * (odR + odRs) + me));
        float attB = DetExp(-(kBetaR_B * (odR + odRs) + me));
        sumR_r += attR * hr; sumR_g += attG * hr; sumR_b += attB * hr;
        sumM_r += attR * hm; sumM_g += attG * hm; sumM_b += attB * hm;
    }

    return {(sumR_r * kBetaR_R * phR + sumM_r * kBetaM * phM) * kSunIntensity,
            (sumR_g * kBetaR_G * phR + sumM_g * kBetaM * phM) * kSunIntensity,
            (sumR_b * kBetaR_B * phR + sumM_b * kBetaM * phM) * kSunIntensity};
}

// Exposure tonemap for the LDR shot: 1 - exp(-c * exposure), then gamma 1/2.2. Shared with the
// shader so the panels' CPU-predicted colors and the rendered pixels use the same curve.
inline math::Vec3 TonemapSky(const math::Vec3& c, float exposure) {
    float r = 1.0f - DetExp(-c.x * exposure);
    float g = 1.0f - DetExp(-c.y * exposure);
    float b = 1.0f - DetExp(-c.z * exposure);
    return {std::pow(r, 1.0f / 2.2f), std::pow(g, 1.0f / 2.2f), std::pow(b, 1.0f / 2.2f)};
}

// --- The pinned digest ---------------------------------------------------------------------------------
// FNV-1a 64 over a fixed grid of SkyColor samples: the 3 showcase sun elevations x view azimuths
// -90..90 deg (step 30) x view elevations 0..90 deg (step 10). Hashes the RAW FLOAT BITS of every
// HDR channel — EXACT, no quantization tolerance needed, because the core math is cross-compiler
// bit-identical by construction (DetExp + double-trig direction setup; see the header comment).
// Printed by the showcase stat line and pinned (MSVC == clang) by the test.
inline uint64_t SkyDigest() {
    uint64_t h = 1469598103934665603ull;               // FNV-1a 64 offset basis
    auto mixf = [&h](float f) {
        uint32_t v;
        static_assert(sizeof(v) == sizeof(f), "float bits");
        std::memcpy(&v, &f, sizeof(v));
        for (int b = 0; b < 4; ++b) {
            h ^= (uint64_t)((v >> (8 * b)) & 0xFFu);
            h *= 1099511628211ull;                     // FNV-1a 64 prime
        }
    };
    const float sunElev[3] = {kSunElevNoonRad, kSunElevLowRad, kSunElevSunsetRad};
    const double kDeg = 3.14159265358979323846 / 180.0;
    for (int s = 0; s < 3; ++s) {
        math::Vec3 sunDir = SunDirFromElevation(sunElev[s]);
        for (int az = -90; az <= 90; az += 30) {
            for (int el = 0; el <= 90; el += 10) {
                double a = (double)az * kDeg, e = (double)el * kDeg;
                double ce = std::cos(e);
                math::Vec3 v{(float)(std::sin(a) * ce), (float)std::sin(e),
                             (float)(-std::cos(a) * ce)};
                math::Vec3 c = SkyColor(v, sunDir);
                mixf(c.x); mixf(c.y); mixf(c.z);
            }
        }
    }
    return h;
}

} // namespace hf::render::atmo
