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

// --- Temporal accumulation: per-frame jittered hemisphere kernel (Slice BV) ---------------------
// For a STATIC scene + STATIC camera the SSGI grain is driven down by AVERAGING N frames, each with a
// DIFFERENT (rotated) hemisphere kernel — exactly the fixed-N deterministic accumulation pattern TAA
// (Slice AP) established (render N fixed-jittered frames, average; no time/RNG, two runs bit-identical;
// the camera doesn't move so there is NO reprojection, just a running mean). HemisphereDirJittered is
// the base HemisphereDir kernel with a FIXED per-frame AZIMUTH rotation applied to phi:
//
//   rot(frame) = frame * kGoldenAngleTurns                     (turns; 1 turn = 2*pi rad)
//   phi        = 2*pi * (u2 + rot(frame))
//
// where kGoldenAngleTurns = (3 - sqrt(5)) / 2 (the golden angle expressed as a fraction of a full
// turn, ~0.381966). The golden-angle step spreads successive frames' azimuths maximally evenly around
// the circle (the same low-discrepancy spiral the Fibonacci/Vogel disk uses), so N frames sample N
// well-separated rotated copies of the base set — maximizing the new coverage each frame adds. The
// rotation is a pure function of `frame` (no RNG/time), so the per-(i,K,normal,frame) output is
// deterministic and identical on Vulkan + Metal.
//
// frame 0 => rot == 0 => the returned direction is BYTE-IDENTICAL to HemisphereDir (documented clean
// relationship): this keeps the single-frame raw --ssgi-shot path (which is "frame 0") unchanged.
//
// The in-shader counterpart is ssgi.frag.hlsl's HemisphereDir(i,K,N,rot): the showcase passes
// rot = frame * kGoldenAngleTurns as the per-frame kernel rotation (ADDED to the per-pixel dither so
// both the spatial dither and the temporal rotation compose into the single azimuth offset).
inline constexpr float kGoldenAngleTurns = 0.38196601125010515f;  // (3 - sqrt(5)) / 2

// Cosine-weighted hemisphere direction with an explicit azimuth rotation `rotTurns` (in turns) — the
// shared core HemisphereDir and HemisphereDirJittered both call. rotTurns == 0 reproduces HemisphereDir
// exactly (same arithmetic, in the same order), so frame 0 is byte-stable.
inline math::Vec3 HemisphereDirRot(int i, int K, const math::Vec3& normal, float rotTurns) {
    if (K < 1) K = 1;
    if (i < 0) i = 0;
    if (i >= K) i = K - 1;

    float u1 = (static_cast<float>(i) + 0.5f) / static_cast<float>(K);
    float u2 = RadicalInverse2(static_cast<uint32_t>(i));

    float r = std::sqrt(std::max(0.0f, u1));
    float phi = 6.2831853071795864769f * (u2 + rotTurns);
    float lx = r * std::cos(phi);
    float ly = r * std::sin(phi);
    float lz = std::sqrt(std::max(0.0f, 1.0f - u1));   // cos(theta)

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

// i-th of K cosine-weighted hemisphere dirs for accumulation `frame`, rotated by frame*goldenAngle.
// Deterministic per (i,K,normal,frame); frame 0 == HemisphereDir (byte-identical).
inline math::Vec3 HemisphereDirJittered(int i, int K, const math::Vec3& normal, int frame) {
    return HemisphereDirRot(i, K, normal, static_cast<float>(frame) * kGoldenAngleTurns);
}

// Temporal-accumulation parameters. accumFrames = the FIXED number of jittered SSGI frames averaged
// into the running mean (N=8, matching TAA's kAccumFrames). Fixed N + fixed per-frame rotation + fixed
// accumulation order (frame 0..N-1) = deterministic, two runs byte-identical.
struct SsgiTemporalParams {
    int accumFrames = 8;   // N jittered SSGI frames averaged (the showcase prints {accumFrames:8})
};

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

// --- Bilateral (edge-preserving) denoise of the SSGI indirect buffer (Slice BR) ----------------
// The single-frame SSGI gather (K=16 rays/pixel) is noisy. A bilateral blur of the indirect buffer
// smooths the grain WITHOUT crossing geometry edges, so the floor pool stays smooth while the
// red/green color bleed stays crisp at surface boundaries. It is the SAME shape as the SSAO blur
// (shaders/ssao_blur.frag.hlsl) but each tap's box weight is replaced by an EDGE-STOPPING weight
// from the G-buffer. This is a PURE function of the SSGI buffer + G-buffer (no time/RNG): two runs
// byte-identical. The SAME math is mirrored verbatim in shaders/ssgi_denoise.frag.hlsl.
//
// BilateralWeight = the weight of a neighbor TAP relative to the CENTER pixel, the product of three
// non-negative terms (so a tap that fails ANY edge test gets ~0 weight and is effectively excluded):
//
//   spatial  = exp(-spatialDist2 / (2 * spatialSigma^2))          // Gaussian falloff in pixels^2
//   depth    = exp(-(depthTap-depthCenter)^2 / (2 * depthSigma^2))// edge-stop across a depth step
//   normal   = pow(max(dot(nCenter, nTap), 0), normalPower)       // edge-stop across a normal step
//   weight   = spatial * depth * normal
//
// Properties (unit-tested): weight == 1 at the center (zero spatial dist, equal depth, equal normal);
// decays monotonically with spatial distance; -> ~0 across a large depth difference (depthDiff >>
// depthSigma); -> 0 across opposing normals (dot(nCenter,nTap) <= 0 -> the pow term is 0); monotonic
// (non-increasing) in each of the spatial-distance, |depth diff| and normal-divergence axes. The
// caller (the shader) accumulates sum += weight * indirectTap and wsum += weight over the kernel and
// outputs sum/wsum (normalized), so a denoised pixel is the edge-aware weighted mean of its window.
//
// `spatialSigma` is passed implicitly via SsgiDenoiseParams to the caller's spatialDist2 scaling; here
// it is taken from the params and threaded in by the caller. To keep BilateralWeight a single pure
// expression we pass spatialSigma as part of the precomputed spatialDist2 term? No — we keep the raw
// inputs explicit so the test can pin every term. Signature mirrors the shader's helper exactly.
inline float BilateralWeight(float spatialDist2, float spatialSigma,
                             float depthCenter, float depthTap, float depthSigma,
                             const math::Vec3& nCenter, const math::Vec3& nTap,
                             float normalPower) {
    float ss = (spatialSigma > 1e-6f) ? spatialSigma : 1e-6f;
    float ds = (depthSigma   > 1e-6f) ? depthSigma   : 1e-6f;
    float spatial = std::exp(-spatialDist2 / (2.0f * ss * ss));
    float dDiff   = depthTap - depthCenter;
    float depth   = std::exp(-(dDiff * dDiff) / (2.0f * ds * ds));
    float nd      = std::max(math::dot(nCenter, nTap), 0.0f);
    float normal  = std::pow(nd, normalPower);
    return spatial * depth * normal;
}

// SSGI denoise parameters. Defaults: a radius-2 (5x5) bilateral window, a spatial sigma of ~2 pixels
// (so the kernel edge taps still contribute meaningfully), a depth sigma in VIEW-space linear-depth
// units tuned so a same-surface depth gradient passes but a geometry step (panel<->floor) is rejected,
// and a normal power that sharply attenuates taps whose normal diverges from the center's. These are
// the values the --ssgi-denoise-shot showcase prints + the shader uses.
struct SsgiDenoiseParams {
    int   radius       = 2;      // kernel half-width in pixels (window is (2*radius+1)^2 taps)
    float spatialSigma = 2.0f;   // Gaussian spatial falloff (pixels)
    float depthSigma   = 0.50f;  // edge-stop sigma in view-space linear-depth units
    float normalPower  = 16.0f;  // exponent on max(dot(nC,nT),0): higher = sharper normal edge-stop
};

// CPU bilateral denoise of a single-channel scalar field (used by the unit test's blur-sanity model;
// the shader does the same accumulation per RGB channel). `field`/`depth` are row-major width*height;
// `normals` is the per-pixel view normal. Returns the denoised field. EDGE-PRESERVING: a tap across a
// large depth step or an opposing normal contributes ~0, so distinct surfaces don't bleed together.
inline std::vector<float> BilateralDenoiseScalar(const std::vector<float>& field,
                                                 const std::vector<float>& depth,
                                                 const std::vector<math::Vec3>& normals,
                                                 int width, int height,
                                                 const SsgiDenoiseParams& p) {
    std::vector<float> out(field.size(), 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int   ci = y * width + x;
            float dC = depth[ci];
            math::Vec3 nC = normals[ci];
            float sum = 0.0f, wsum = 0.0f;
            for (int dy = -p.radius; dy <= p.radius; ++dy) {
                int ty = y + dy;
                if (ty < 0 || ty >= height) continue;
                for (int dx = -p.radius; dx <= p.radius; ++dx) {
                    int tx = x + dx;
                    if (tx < 0 || tx >= width) continue;
                    int ti = ty * width + tx;
                    float spatialDist2 = static_cast<float>(dx * dx + dy * dy);
                    float w = BilateralWeight(spatialDist2, p.spatialSigma,
                                              dC, depth[ti], p.depthSigma,
                                              nC, normals[ti], p.normalPower);
                    sum += w * field[ti];
                    wsum += w;
                }
            }
            out[ci] = (wsum > 1e-12f) ? (sum / wsum) : field[ci];
        }
    }
    return out;
}

} // namespace hf::render::ssgi
