#pragma once
// Screen-space global illumination (SSGI) math — pure CPU (header-only, no device, no backend
// symbols). Shared by the --ssgi-shot showcase reasoning AND tests/ssgi_test.cpp so the unit test
// exercises the SAME hemisphere kernel + view<->screen projection the in-shader ray-march uses.
//
// SSGI is one bounce of indirect DIFFUSE lighting via screen space. For each pixel we reconstruct the
// view-space position P + normal N from the G-buffer (EXACTLY like SSR — same reconstruction), trace
// K rays in the cosine-weighted hemisphere about N, march each in view space against the G-buffer
// (the SAME march + binary-search as ssr.frag), and on a hit sample the ALREADY-LIT HDR scene color
// as the incoming radiance for that ray. The accumulated mean of the K hit radiances is the indirect
// diffuse irradiance, which the composite multiplies by albedo and ADDS to the scene.
//
// REUSE: the view<->screen projection + view-space reconstruction is SHARED with SSR — this header
// includes render/ssr.h and re-exports ssr::ReconstructViewPos / ssr::ViewToScreenUV rather than
// duplicating that math (the shader likewise copies ssr.frag's ReconstructViewPos/ProjectToUV
// verbatim). Conventions match engine/render/ssr.h + the SSAO G-buffer: RH view space (-Z forward),
// view-space LINEAR depth = -vpos.z; the yFlip sign is -1 on Vulkan, +1 on Metal.

#include "math/math.h"
#include "render/ssr.h"   // SHARE the view<->screen reconstruction (ReconstructViewPos/ViewToScreenUV)
#include <algorithm>
#include <cmath>
#include <vector>

namespace hf::render::ssgi {

// Re-export the SHARED SSR reconstruction so callers/tests can write ssgi::ReconstructViewPos and the
// reuse is explicit at the call site (these are the SAME functions ssr.frag's march uses — the SSGI
// march reconstructs P,N identically). Documented reuse, not a duplicate implementation.
using ssr::ReconstructViewPos;
using ssr::ViewToScreenUV;

// SSGI parameters (mirrors SsrParams). marchDist/thickness drive the SAME ray-march as SSR; K is the
// number of hemisphere rays; intensity scales the accumulated indirect irradiance in the composite.
struct SsgiParams {
    int   K = 16;            // hemisphere rays per pixel
    float marchDist = 6.0f;  // total view-space march length per ray
    float thickness = 0.5f;  // depth-compare band (view units)
    float intensity = 1.0f;  // indirect-diffuse gain applied in the composite
};

// --- Cosine-weighted hemisphere kernel ----------------------------------------------------------
// HemisphereDir(i,K,normal) is the i-th of K FIXED directions, cosine-weighted about `normal`.
//
// Distribution (documented, deterministic, NO RNG/time):
//   * The 2D sample (u1,u2) is the i-th point of the base-2 RADICAL-INVERSE Hammersley set:
//       u1 = (i + 0.5) / K          (stratified along the first axis)
//       u2 = RadicalInverse_2(i)    (van der Corput, low-discrepancy along the second axis)
//   * Cosine (Malley) mapping to the +Z hemisphere of LOCAL tangent space:
//       r = sqrt(u1);  phi = 2*pi*u2;  local = (r*cos phi, r*sin phi, sqrt(1-u1))
//     so the local z-component is cos(theta) and the density is cos(theta)/pi — i.e. the cosine
//     weight is BAKED INTO the sampling, which is why AccumulateIndirect is a simple mean (below).
//   * The local sample is rotated into world/view space by a TBN basis built from `normal` and a
//     STABLE tangent (the smaller-axis trick) so the frame is a continuous, deterministic function of
//     the normal (no per-pixel RNG; the shader may additionally rotate the whole set by a baked 4x4
//     dither like SSR, which does NOT change this CPU kernel's per-(i,K,normal) output).
//
// Guarantees (unit-tested): unit length, dot(dir,normal) >= 0 (in-hemisphere), deterministic, and for
// normal=+Z the local frame is identity so the returned dirs equal the raw cosine-mapped samples.

// Van der Corput radical inverse in base 2 (bit-reversal of i over 32 bits, scaled to [0,1)).
inline float RadicalInverse2(uint32_t i) {
    i = (i << 16) | (i >> 16);
    i = ((i & 0x55555555u) << 1) | ((i & 0xAAAAAAAAu) >> 1);
    i = ((i & 0x33333333u) << 2) | ((i & 0xCCCCCCCCu) >> 2);
    i = ((i & 0x0F0F0F0Fu) << 4) | ((i & 0xF0F0F0F0u) >> 4);
    i = ((i & 0x00FF00FFu) << 8) | ((i & 0xFF00FF00u) >> 8);
    return static_cast<float>(i) * 2.3283064365386963e-10f;  // / 2^32
}

// Build a stable orthonormal tangent basis (T,B) for a unit normal N. Uses the "smaller-axis"
// reference vector so the basis varies continuously and is a pure function of N (no RNG, no branch on
// floating equality that could differ across builds). Same construction the shader uses.
inline void BuildTangentBasis(const math::Vec3& N, math::Vec3& T, math::Vec3& B) {
    math::Vec3 n = math::normalize(N);
    // Pick the world axis least aligned with N as the reference up to avoid a degenerate cross.
    math::Vec3 ref = (std::fabs(n.x) <= std::fabs(n.y) && std::fabs(n.x) <= std::fabs(n.z))
                         ? math::Vec3{1.0f, 0.0f, 0.0f}
                         : (std::fabs(n.y) <= std::fabs(n.z) ? math::Vec3{0.0f, 1.0f, 0.0f}
                                                             : math::Vec3{0.0f, 0.0f, 1.0f});
    T = math::normalize(math::cross(ref, n));
    B = math::cross(n, T);
}

inline math::Vec3 HemisphereDir(int i, int K, const math::Vec3& normal) {
    if (K < 1) K = 1;
    if (i < 0) i = 0;
    if (i >= K) i = K - 1;

    // Hammersley point (u1 stratified, u2 = radical inverse base 2).
    float u1 = (static_cast<float>(i) + 0.5f) / static_cast<float>(K);
    float u2 = RadicalInverse2(static_cast<uint32_t>(i));

    // Cosine (Malley) mapping onto the +Z hemisphere of local tangent space.
    float r = std::sqrt(std::max(0.0f, u1));
    float phi = 6.2831853071795864769f * u2;
    float lx = r * std::cos(phi);
    float ly = r * std::sin(phi);
    float lz = std::sqrt(std::max(0.0f, 1.0f - u1));   // cos(theta)

    // Rotate the local sample into the hemisphere about `normal` via a stable TBN basis.
    math::Vec3 N = math::normalize(normal);
    math::Vec3 T, B;
    BuildTangentBasis(N, T, B);
    math::Vec3 dir{
        T.x * lx + B.x * ly + N.x * lz,
        T.y * lx + B.y * ly + N.y * lz,
        T.z * lx + B.z * ly + N.z * lz,
    };
    return math::normalize(dir);
}

// --- Indirect-diffuse accumulation --------------------------------------------------------------
// AccumulateIndirect = the Monte-Carlo estimator of indirect diffuse irradiance for one pixel:
//   E_indirect = (1/K) * sum_k L_hit_k
// Because the hemisphere directions are COSINE-distributed (density cos(theta)/pi), the importance
// weight and the cosine foreshortening term cancel, leaving a plain MEAN of the hit radiances as the
// unbiased estimator (the standard Malley/cosine-sampling result). A ray that MISSES contributes 0
// (its entry is the zero radiance), i.e. unlit screen-space gaps add no indirect light — the
// documented fallback (the caller may instead push an ambient constant for misses; here misses are
// simply absent/zero, matching the shader which accumulates 0 on a miss). An empty input yields 0.
inline math::Vec3 AccumulateIndirect(const std::vector<math::Vec3>& hitRadiances) {
    if (hitRadiances.empty()) return math::Vec3{0.0f, 0.0f, 0.0f};
    math::Vec3 sum{0.0f, 0.0f, 0.0f};
    for (const auto& r : hitRadiances) sum = sum + r;
    float inv = 1.0f / static_cast<float>(hitRadiances.size());
    return sum * inv;
}

} // namespace hf::render::ssgi
