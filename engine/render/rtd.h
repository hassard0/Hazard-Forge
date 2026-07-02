#pragma once
// Slice RTD1 — DETERMINISTIC STOCHASTIC RAY-TRACED SOFT SHADOWS + SVGF-LITE TEMPORAL DENOISER (Track-S S9
// of docs/SUPERIORITY_ROADMAP.md, hf::render::rtd). Pure CPU (header-only, NO device, NO backend symbols).
// The RT arc (RT1-RT7) is exact/binary — hard shadows, mirror reflections, HW==CPU byte-equal. RTD1 adds
// the missing pair the roadmap flags as the documented RT gap:
//
//   (a) STOCHASTIC RT — AREA-LIGHT SOFT SHADOWS via jittered shadow rays. Per pixel per frame k, ONE
//       shadow ray toward a DETERMINISTIC point on a disc area light. The sample point comes from a FIXED
//       64-entry golden-angle (Vogel) spiral over the unit disc, indexed by (pixel hash + frame k) — NO
//       RNG, NO clock (the ssgi_temporal / TAA fixed-sequence discipline). The 1-sample result is a
//       BINARY visibility -> a NOISY penumbra.
//   (b) an SVGF-LITE DENOISER — fixed-N TEMPORAL accumulation (N = kRtdAccumFrames = 8, the
//       taa::kAccumFrames convention; a running mean of the visibility buffer, static camera so no
//       reprojection — see the reprojection note below) followed by the ssgi.h BILATERAL SPATIAL pass
//       (ssgi::BilateralDenoiseScalar called VERBATIM — the SAME edge-stopping weight --ssgi-denoise
//       ships) over the accumulated visibility, edge-guarded by the RT hit records' depth (hit.t) +
//       normal (the RT scene has no raster G-buffer; the ray tracer's own hit t/normal IS the G-buffer).
//
// A deterministic, golden-able RT denoiser is something DLSS-RR/OptiX cannot claim: the WHOLE stochastic
// pipeline is a pure function of its inputs -> two runs byte-identical, and the INTEGER half is
// HW==CPU byte-equal per the RT-arc bar.
//
// THE HONEST PROOF SPLIT (integer vs float — pinned, not hidden):
//   * INTEGER (byte-equal class): the per-pixel VISIBILITY COUNTS (0..N accumulated binary shadow-ray
//     results) + the ACCUMULATED soft-shadow image (integer Lambert scaled by vis/N — an EXACT Q16.16
//     fraction (vis<<16)/N). Pure Q16.16/uint32 -> the GPU twin shaders/rt_softshadow.comp.hlsl copies the
//     math VERBATIM and the Vulkan showcase proves memcmp HW==CPU on BOTH buffers (the RT-arc bar).
//   * FLOAT (visresolve-class exception, documented): the bilateral SPATIAL pass (ssgi.h exp/pow weights)
//     + the final denoised shade. Deterministic (no clock/RNG -> two runs byte-identical on a given
//     platform/libm) but NOT cross-vendor byte-comparable — the same split every float capstone slice
//     (FPX6/MC5/GR6...) documents.
//
// THE SAMPLING SCHEME (pinned exactly; the shader mirrors it verbatim):
//   * kRtdDiscSamples[64]: the j-th point of a Vogel/golden-angle spiral over the UNIT DISC, baked as
//     Q16.16 HOST LITERALS (the fpx host-baked-LUT precedent — NO runtime sqrt/cos/sin anywhere):
//       r_j = sqrt((j+0.5)/64),  theta_j = j * goldenAngle (pi*(3-sqrt(5)) rad = 2.39996...),
//       (x_j, y_j) = round(65536 * r_j * (cos theta_j, sin theta_j)).
//   * The per-(pixel, frame) sample index: j(p,k) = (PcgHash(kRtdPixelSeed, pixelIndex) + k*kRtdFrameStride)
//     & 63. PcgHash is engine/pcg/pcg.h's seeded integer avalanche REUSED VERBATIM (the established pcg
//     discipline — pure uint32 wrapping ops, identical on every compiler/vendor). kRtdFrameStride = 39 =
//     round(64/phi) is the GOLDEN-RATIO stride: coprime with 64, so (i) any prefix of frames covers the
//     spiral near-uniformly (consecutive frames jump ~0.61 of the table — the golden-angle discipline at
//     the FRAME level, matching ssgi::kGoldenAngleTurns at the kernel level), and (ii) 64 frames visit ALL
//     64 points exactly once -> the 64-sample GROUND TRUTH is the full-table mean, INDEPENDENT of the
//     pixel hash (unit-tested).
//   * The area light: a DISC — center + sx*axisU + sy*axisV with (sx,sy) = kRtdDiscSamples[j] in
//     [-kOne,kOne]^2 (unit disc) and axisU/axisV the radius-scaled Q16.16 half-axes. All fxmul integer.
//   * The shadow ray: origin = hit.pos + normal*rtrace::kRtShadowEps (the RT3 anti-acne offset REUSED),
//     dir = samplePoint - origin. t is in units of |dir| -> the LIGHT POINT IS AT t == kOne, so the
//     occlusion test is RANGED: occluded iff ANY primitive yields kRtShadowMinT < t < kOne (an occluder
//     BEYOND the light does NOT occlude — unlike RT3's unbounded directional test). Order-independent
//     boolean OR (early-out safe), the TraceAnyHit contract with a far bound.
//
// TEMPORAL REPROJECTION (documented future work): the camera is STATIC (the ssgi_temporal precedent), so
// the temporal mean needs no reprojection. Motion-reprojection composes with the shipped TSR (US3)
// disocclusion machinery — that is a follow-on slice, NOT claimed here.
//
// REUSE MAP: rtrace.h fx/FxVec3/fxmul/fxdiv/FxDot/RtNormalize/IntersectSphere/IntersectAabb/TraceClosest/
// PrimaryRay/AlbedoFor/PackRGBA8/kRtShadowEps/kRtShadowMinT (#included READ-ONLY, BYTE-FROZEN);
// ssgi.h BilateralWeight/BilateralDenoiseScalar/SsgiDenoiseParams (#included READ-ONLY, called VERBATIM —
// NOT mirrored); pcg.h PcgHash (#included READ-ONLY, called VERBATIM). NO existing header edited.
//
// SEAM DISCIPLINE: ZERO backend (vk*/MTL*/mtl::) symbols. NO GPU, NO new RHI here. Mentions of "GPU"/HW
// are doc-only.

#include <cstdint>
#include <span>
#include <vector>

#include "pcg/pcg.h"       // PcgHash (READ-ONLY — the seeded integer avalanche, reused verbatim)
#include "render/rtrace.h" // the frozen Q16.16 RT reference (READ-ONLY)
#include "render/ssgi.h"   // BilateralDenoiseScalar / SsgiDenoiseParams (READ-ONLY, called verbatim)

namespace hf::render::rtd {

namespace rt = hf::render::rtrace;
using rt::fx;
using rt::FxVec3;
using rt::kFrac;
using rt::kOne;

// ----- The pinned sampling constants (the shader shaders/rt_softshadow.comp.hlsl mirrors ALL of these) --
inline constexpr uint32_t kRtdDiscSampleCount = 64;  // Vogel-spiral points on the unit disc
inline constexpr uint32_t kRtdAccumFrames     = 8;   // N accumulated frames (the taa::kAccumFrames convention)
inline constexpr uint32_t kRtdTruthFrames     = 64;  // the ground-truth sample count (covers ALL 64 points)
inline constexpr uint32_t kRtdFrameStride     = 39;  // round(64/phi), coprime with 64 (golden-ratio stride)
inline constexpr uint32_t kRtdPixelSeed       = 0x52544431u;  // 'RTD1' — the PcgHash seed for the pixel hash

// The 64-entry golden-angle (Vogel) spiral over the UNIT DISC, Q16.16 host literals (generation formula in
// the header comment; max |p|^2 = 0.9922 * kOne^2 — strictly inside the disc). PINNED — the cross-backend
// byte-equal proof and the golden rest on these exact integers.
struct RtdDiscPoint { fx x, y; };
inline constexpr RtdDiscPoint kRtdDiscSamples[kRtdDiscSampleCount] = {
    {   5793,       0}, {  -7398,    6777}, {   1132,  -12903}, {   9325,   12163},
    { -17112,   -3027}, {  16210,  -10312}, {  -5422,   20170}, { -10340,  -19910},
    {  22434,    8193}, { -23339,    9634}, {  11251,  -24043}, {   8314,   26507},
    { -25059,  -14522}, {  29397,   -6463}, { -17941,   25519}, {  -4145,  -31985},
    {  25445,   21445}, { -34240,    1416}, {  24976,  -24854}, {  -1671,   36136},
    { -23764,  -28478}, {  37646,    5065}, { -31897,   22193}, {   8716,  -38744},
    {  20160,   35182}, { -39411,  -12573}, {  38282,  -17687}, { -16585,   39629},
    { -14802,  -41152}, {  39386,   20700}, { -43747,   11532}, {  24866,  -38673},
    {   7910,   46027}, { -37487,  -29032}, {  47953,   -3973}, { -33146,   35829},
    {    241,  -49492}, {  33705,   37155}, { -50613,   -4691}, {  41012,  -31126},
    {  -9331,   51292}, { -28107,  -44665}, {  51506,   14116}, { -48070,   24669},
    {  18996,  -51239}, {  20835,   51180}, { -50480,  -23923}, {  53953,  -16634},
    { -28846,   49221}, { -12101,  -56351}, {  47460,   33713}, { -58337,    7270},
    {  38473,  -45200}, {   2184,   59880}, { -42449,  -43075}, {  60949,    3116},
    { -47469,   39221}, {   8584,  -61523}, {  35534,   51606}, { -61581,  -14170},
    {  55439,  -31410}, { -19824,   61108}, { -26876,  -58923}, {  60095,   25495},
};

// ----- The disc area light (Q16.16) ---------------------------------------------------------------------
// samplePoint(j) = center + sx_j*axisU + sy_j*axisV with (sx,sy) = kRtdDiscSamples[j] (unit disc).
// axisU/axisV are the RADIUS-SCALED half-axes (they need not be unit; their length IS the disc radius
// along that axis). Pure integer.
struct RtdAreaLight {
    FxVec3 center;
    FxVec3 axisU;
    FxVec3 axisV;
};

// RtdSampleIndex — the per-(pixel, frame) spiral index. pixelIndex = py*width + px (row-major). PcgHash is
// pcg::PcgHash reused VERBATIM; the golden-ratio frame stride walks the table so any frame prefix covers
// the spiral near-uniformly and 64 frames cover it EXACTLY once (gcd(39,64) == 1).
inline uint32_t RtdSampleIndex(uint32_t pixelIndex, uint32_t frame) {
    return (hf::pcg::PcgHash(kRtdPixelSeed, pixelIndex) + frame * kRtdFrameStride) &
           (kRtdDiscSampleCount - 1u);
}

// RtdSamplePoint — the j-th deterministic point ON the disc light (world Q16.16). Pure integer
// (FxAdd/FxScale are int64-intermediate fxmuls per axis).
inline FxVec3 RtdSamplePoint(const RtdAreaLight& light, uint32_t j) {
    const RtdDiscPoint s = kRtdDiscSamples[j & (kRtdDiscSampleCount - 1u)];
    FxVec3 p = light.center;
    p = rt::FxAdd(p, rt::FxScale(light.axisU, s.x));
    p = rt::FxAdd(p, rt::FxScale(light.axisV, s.y));
    return p;
}

// ----- Ranged any-hit occlusion (the RT3 TraceAnyHit contract with a FAR BOUND) --------------------------
// occluded iff ANY primitive's fx intersection yields minT < t < maxT. For a shadow ray whose dir is the
// FULL vector to the light point, the light sits at t == kOne, so maxT = kOne rejects occluders BEYOND the
// light (a directional light has no far bound; an area light does). An order-independent boolean OR — MAY
// early-out on the first qualifying hit without breaking determinism (the HW any-hit early-out == the SW
// full scan == this, the same argument as rtrace::TraceAnyHit). rtrace.h is BYTE-FROZEN, so the ranged
// variant lives HERE (append-only discipline).
inline bool TraceAnyHitRanged(const rt::RtRay& ray, const rt::RtScene& scene, fx minT, fx maxT) {
    for (const rt::RtSphere& s : scene.spheres) {
        rt::RtHit h;
        if (rt::IntersectSphere(ray, s, h) && h.t > minT && h.t < maxT) return true;
    }
    for (const rt::RtAabb& b : scene.aabbs) {
        rt::RtHit h;
        if (rt::IntersectAabb(ray, b, h) && h.t > minT && h.t < maxT) return true;
    }
    return false;
}

// ----- The per-pixel primary surface record (the RT "G-buffer") ------------------------------------------
// The RT scene has no raster G-buffer; the primary hit's t (depth) + normal ARE the G-buffer the bilateral
// edge-stops on (documented — the ssgi.h weight takes exactly a depth + a normal per tap). ndl is the
// CLAMPED Lambert term toward the light CENTER (Q16.16), computed once (static camera + static light).
struct RtdSurface {
    rt::RtHit hit;   // primIndex == rtrace::kRtMiss on a primary miss
    fx        ndl = 0;
};

// TracePrimarySurfaces — one primary TraceClosest per pixel (the frozen rtrace math), recording the hit +
// the clamped ndl toward the light center. Returns the primary-hit count. Pure integer, deterministic.
inline uint32_t TracePrimarySurfaces(const rt::RtScene& scene, const RtdAreaLight& light,
                                     const rt::RtCamera& cam, uint32_t width, uint32_t height,
                                     std::vector<RtdSurface>& out) {
    out.assign((size_t)width * height, RtdSurface{});
    uint32_t hits = 0;
    for (uint32_t py = 0; py < height; ++py) {
        for (uint32_t px = 0; px < width; ++px) {
            rt::RtRay ray = rt::PrimaryRay(cam, px, py, width, height);
            RtdSurface s;
            s.hit = rt::TraceClosest(ray, scene);
            if (s.hit.primIndex != rt::kRtMiss) {
                ++hits;
                FxVec3 L = rt::RtNormalize(rt::FxSub(light.center, s.hit.pos));
                fx ndl = rt::FxDot(s.hit.normal, L);
                s.ndl = ndl < 0 ? 0 : ndl;
            }
            out[(size_t)py * width + px] = s;
        }
    }
    return hits;
}

// AccumulateSoftShadowVis — the STOCHASTIC visibility accumulation (the integer heart of RTD1). For every
// pixel with a primary hit, for each frame k in [frame0, frame0+frames): ONE shadow ray toward
// RtdSamplePoint(light, RtdSampleIndex(pixel, k)); vis[pixel] += (occluded ? 0 : 1). ADDS into `vis`
// (caller zero-initializes), so vis over [0,N) == the sum of per-frame accumulations (unit-tested
// additivity — the running-mean/temporal-accumulation identity). Pure integer; the GPU twin loops the SAME
// k in-shader (an order-independent integer sum). Miss pixels stay 0.
inline void AccumulateSoftShadowVis(const rt::RtScene& scene, const RtdAreaLight& light,
                                    const std::vector<RtdSurface>& surfaces, uint32_t width,
                                    uint32_t height, uint32_t frame0, uint32_t frames,
                                    std::span<uint32_t> vis) {
    for (uint32_t py = 0; py < height; ++py) {
        for (uint32_t px = 0; px < width; ++px) {
            const size_t idx = (size_t)py * width + px;
            const RtdSurface& s = surfaces[idx];
            if (s.hit.primIndex == rt::kRtMiss) continue;
            rt::RtRay shadowRay;
            shadowRay.origin = rt::FxAdd(s.hit.pos, rt::FxScale(s.hit.normal, rt::kRtShadowEps));
            for (uint32_t k = frame0; k < frame0 + frames; ++k) {
                uint32_t j = RtdSampleIndex((uint32_t)idx, k);
                FxVec3 sp = RtdSamplePoint(light, j);
                shadowRay.dir = rt::FxSub(sp, shadowRay.origin);
                if (!TraceAnyHitRanged(shadowRay, scene, rt::kRtShadowMinT, kOne)) ++vis[idx];
            }
        }
    }
}

// ----- The INTEGER accumulated shade (byte-equal class) --------------------------------------------------
// diffuse = ambient + ((kOne-ambient) * ndl) * visFrac with visFrac = (vis << kFrac) / frames — an EXACT
// Q16.16 fraction (vis <= frames <= 64 -> no overflow). The fxmul ORDER is pinned (the shader mirrors it):
// lambert = fxmul(fxmul(kOne - ambient, ndl), visFrac). vis == frames reproduces the fully-lit RT3-style
// shade; vis == 0 leaves the ambient floor. Albedo/quantize identical to rtrace::ShadeHitShadowed.
inline uint32_t ShadeSoftShadowInt(const RtdSurface& s, uint32_t vis, uint32_t frames,
                                   uint32_t background) {
    if (s.hit.primIndex == rt::kRtMiss) return background;
    const fx ambient = (fx)(kOne * 18 / 100);  // 0.18 ambient floor (== ShadeHitInt/ShadeHitShadowed)
    fx visFrac = (fx)(((int64_t)vis << kFrac) / (int64_t)frames);
    fx lambert = rt::fxmul(rt::fxmul(kOne - ambient, s.ndl), visFrac);
    fx diffuse = ambient + lambert;
    FxVec3 alb = rt::AlbedoFor(s.hit.primIndex);
    auto q = [&](fx ch) -> int32_t {
        fx lit = rt::fxmul(ch, diffuse);
        return (int32_t)(((int64_t)lit * 255) >> kFrac);
    };
    return rt::PackRGBA8(q(alb.x), q(alb.y), q(alb.z), 255);
}

// RenderSoftShadowImageInt — the full accumulated INTEGER image (the buffer the Vulkan showcase memcmp's
// HW==CPU). outRGBA8 sized width*height.
inline void RenderSoftShadowImageInt(const rt::RtScene& scene, const std::vector<RtdSurface>& surfaces,
                                     std::span<const uint32_t> vis, uint32_t frames, uint32_t width,
                                     uint32_t height, std::span<uint32_t> outRGBA8) {
    for (size_t i = 0; i < (size_t)width * height; ++i)
        outRGBA8[i] = ShadeSoftShadowInt(surfaces[i], vis[i], frames, scene.background);
}

// ----- The SVGF-LITE SPATIAL pass (FLOAT — the documented visresolve-class exception) --------------------
// RtdDenoiseParams — the PINNED RTD1 tuning of the ssgi bilateral. Identical to the ssgi defaults EXCEPT
// spatialSigma is tightened 2.0 -> 1.0: unlike SSGI's indirect field, the shadow field's penumbra RAMP is
// the SIGNAL and lives on ONE flat surface (the ground), so the depth/normal edge-stops cannot protect it
// — only the spatial falloff does. Pinned by a parameter sweep on the 320x240 showcase scene (band MAE vs
// the 64-sample truth): the ssgi default (r2/sigma2) yields maeD 0.0860 > the raw 8-sample's 0.0748 (the
// classic SVGF over-blur); r2/sigma1 yields maeD 0.0631 AND varD 0.0575 — STRICTLY better than the raw
// 8-sample on BOTH metrics. The weight MATH is still ssgi::BilateralWeight verbatim; only the sigma
// parameter differs (documented, not hidden).
inline ssgi::SsgiDenoiseParams RtdDenoiseParams() {
    ssgi::SsgiDenoiseParams p{};
    p.radius = 2;
    p.spatialSigma = 1.0f;   // tightened vs the ssgi default 2.0 (see above)
    p.depthSigma = 0.50f;    // == the ssgi default
    p.normalPower = 16.0f;   // == the ssgi default
    return p;
}

// DenoiseSoftShadowVis: vis/frames -> a float field; depth = hit.t / kOne (the primary ray's fx t — the RT
// G-buffer depth; misses get a 1e8 sentinel so the depth edge-stop excludes them); normal = hit.normal /
// kOne (misses get the zero normal -> BilateralWeight's dot term zeroes every tap -> the pixel passes
// through unchanged). The blur itself is ssgi::BilateralDenoiseScalar CALLED VERBATIM (the --ssgi-denoise
// math — spatial Gaussian * depth Gaussian * normal power, normalized) with the PINNED RtdDenoiseParams
// unless overridden. Deterministic (a pure function; no clock/RNG) — two runs byte-identical; float
// exp/pow -> NOT cross-vendor byte-comparable.
inline std::vector<float> DenoiseSoftShadowVis(const std::vector<RtdSurface>& surfaces,
                                               std::span<const uint32_t> vis, uint32_t frames,
                                               uint32_t width, uint32_t height,
                                               const ssgi::SsgiDenoiseParams& p = RtdDenoiseParams()) {
    const size_t n = (size_t)width * height;
    std::vector<float> field(n, 0.0f);
    std::vector<float> depth(n, 0.0f);
    std::vector<math::Vec3> normals(n, math::Vec3{0.0f, 0.0f, 0.0f});
    const float invFrames = 1.0f / (float)frames;
    const float invOne = 1.0f / (float)(int)kOne;
    for (size_t i = 0; i < n; ++i) {
        const RtdSurface& s = surfaces[i];
        if (s.hit.primIndex == rt::kRtMiss) {
            depth[i] = 1.0e8f;  // sentinel: the depth edge-stop excludes miss pixels from hit windows
            continue;
        }
        field[i] = (float)vis[i] * invFrames;
        depth[i] = (float)s.hit.t * invOne;
        normals[i] = math::Vec3{(float)s.hit.normal.x * invOne, (float)s.hit.normal.y * invOne,
                                (float)s.hit.normal.z * invOne};
    }
    return ssgi::BilateralDenoiseScalar(field, depth, normals, (int)width, (int)height, p);
}

// ShadeSoftShadowFloat — the float twin of ShadeSoftShadowInt over a DENOISED [0,1] visibility.
// diffuse = 0.18 + 0.82*ndl*visD; channel = albedo*diffuse; quantize by truncation (matching the integer
// path's floor). FLOAT class (deterministic per platform; not cross-vendor byte-comparable).
inline uint32_t ShadeSoftShadowFloat(const RtdSurface& s, float visD, uint32_t background) {
    if (s.hit.primIndex == rt::kRtMiss) return background;
    const float invOne = 1.0f / (float)(int)kOne;
    if (visD < 0.0f) visD = 0.0f;
    if (visD > 1.0f) visD = 1.0f;
    float ndl = (float)s.ndl * invOne;
    float diffuse = 0.18f + 0.82f * ndl * visD;
    FxVec3 alb = rt::AlbedoFor(s.hit.primIndex);
    auto q = [&](fx ch) -> int32_t {
        float lit = ((float)ch * invOne) * diffuse;
        if (lit < 0.0f) lit = 0.0f;
        if (lit > 1.0f) lit = 1.0f;
        return (int32_t)(lit * 255.0f);
    };
    return rt::PackRGBA8(q(alb.x), q(alb.y), q(alb.z), 255);
}

// RenderSoftShadowImageFloat — the DENOISED soft-shadow image (the showcase money shot; float class).
inline void RenderSoftShadowImageFloat(const rt::RtScene& scene, const std::vector<RtdSurface>& surfaces,
                                       const std::vector<float>& visD, uint32_t width, uint32_t height,
                                       std::span<uint32_t> outRGBA8) {
    for (size_t i = 0; i < (size_t)width * height; ++i)
        outRGBA8[i] = ShadeSoftShadowFloat(surfaces[i], visD[i], scene.background);
}

// ----- The quality metrics (the denoiser is LOAD-BEARING — pinned, fail-loudly) --------------------------
// The PENUMBRA BAND: primary-hit pixels whose 64-sample GROUND-TRUTH visibility is strictly fractional
// (0 < vis64 < 64) — the pixels where the area light is PARTIALLY occluded (a hard-shadow pipeline has an
// EMPTY band; a nonzero band is the soft-shadows-exist proof). Metrics over the band:
//   * var{1,8,D}: the population variance of the {raw 1-sample, raw 8-sample mean, denoised} visibility —
//     the NOISE metric (c): the denoiser must yield varD < var1 STRICTLY (and in practice < var8).
//   * mae{1,8,D}: the mean |field - truth| against the 64-sample truth — the GROUND-TRUTH error (d): the
//     denoised 8-sample result must sit closer to the truth than the raw 1-sample.
// Accumulated in double in a FIXED row-major order -> deterministic.
struct RtdMetrics {
    uint32_t bandPixels = 0;
    float var1 = 0, var8 = 0, varD = 0;
    float mae1 = 0, mae8 = 0, maeD = 0;
};

inline RtdMetrics ComputeRtdMetrics(const std::vector<RtdSurface>& surfaces,
                                    std::span<const uint32_t> vis1, std::span<const uint32_t> vis8,
                                    const std::vector<float>& visD, std::span<const uint32_t> vis64,
                                    uint32_t width, uint32_t height) {
    RtdMetrics m;
    const size_t n = (size_t)width * height;
    // Pass 1: band membership + means.
    double s1 = 0, s8 = 0, sD = 0;
    for (size_t i = 0; i < n; ++i) {
        if (surfaces[i].hit.primIndex == rt::kRtMiss) continue;
        if (vis64[i] == 0 || vis64[i] >= kRtdTruthFrames) continue;
        ++m.bandPixels;
        s1 += (double)vis1[i];
        s8 += (double)vis8[i] / (double)kRtdAccumFrames;
        sD += (double)visD[i];
    }
    if (m.bandPixels == 0) return m;
    const double inv = 1.0 / (double)m.bandPixels;
    const double m1 = s1 * inv, m8 = s8 * inv, mD = sD * inv;
    // Pass 2: variances + MAE vs the truth.
    double v1 = 0, v8 = 0, vD = 0, e1 = 0, e8 = 0, eD = 0;
    for (size_t i = 0; i < n; ++i) {
        if (surfaces[i].hit.primIndex == rt::kRtMiss) continue;
        if (vis64[i] == 0 || vis64[i] >= kRtdTruthFrames) continue;
        const double truth = (double)vis64[i] / (double)kRtdTruthFrames;
        const double f1 = (double)vis1[i];
        const double f8 = (double)vis8[i] / (double)kRtdAccumFrames;
        const double fD = (double)visD[i];
        v1 += (f1 - m1) * (f1 - m1);
        v8 += (f8 - m8) * (f8 - m8);
        vD += (fD - mD) * (fD - mD);
        e1 += (f1 > truth) ? (f1 - truth) : (truth - f1);
        e8 += (f8 > truth) ? (f8 - truth) : (truth - f8);
        eD += (fD > truth) ? (fD - truth) : (truth - fD);
    }
    m.var1 = (float)(v1 * inv); m.var8 = (float)(v8 * inv); m.varD = (float)(vD * inv);
    m.mae1 = (float)(e1 * inv); m.mae8 = (float)(e8 * inv); m.maeD = (float)(eD * inv);
    return m;
}

// ----- Fnv1a64 digest (the stat-line pin for the INTEGER buffers) ----------------------------------------
// The standard FNV-1a 64 over raw bytes (the nav::Fnv1a64ML shape — defined here so the two showcases +
// the test share one definition without touching nav headers). Integer-buffer digests are cross-platform
// exact; float-buffer digests are per-platform only (documented where printed).
inline uint64_t RtdFnv1a64(const void* data, size_t bytes, uint64_t h = 1469598103934665603ull) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

// ----- The pinned RTD1 showcase light --------------------------------------------------------------------
// A HORIZONTAL disc (a ceiling-panel light) of radius 3 world units centered at (6, 12, -4.5) — EXACTLY
// 15 * (0.4, 0.8, -0.3), i.e. sitting ON the RT3 directional light's axis so the soft image reads as "the
// RT3 scene, the light given AREA". Occluders (sphere tops y in [0.75, 1], boxes to y = 2) sit ~11 units
// under the light and ~2-3 units over the ground plane (y = -1) -> a penumbra of roughly
// radius * (occluder->ground) / (light->occluder) = 3 * 2/11 = 0.55 world units — a clearly visible
// multi-pixel band at 320x240. Pure Q16.16 literals (F is the frozen rtrace fraction helper).
inline RtdAreaLight BuildRtd1Light() {
    RtdAreaLight l;
    l.center = FxVec3{rt::F(6, 1), rt::F(12, 1), rt::F(-9, 2)};
    l.axisU  = FxVec3{rt::F(3, 1), 0, 0};
    l.axisV  = FxVec3{0, 0, rt::F(3, 1)};
    return l;
}

}  // namespace hf::render::rtd
