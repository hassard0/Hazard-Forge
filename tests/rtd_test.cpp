// Slice RTD1 — DETERMINISTIC STOCHASTIC RT SOFT SHADOWS + SVGF-LITE DENOISER (Track-S S9): the CPU-side
// invariants the HW accumulation kernel (shaders/rt_softshadow.comp) and the --rtd1-softshadow(-shot)
// showcases rest on. The HW==CPU memcmp proof lives in the Vulkan showcase (it needs a GPU); this
// pure-CPU test pins:
//
//   * THE SAMPLE SEQUENCE: the 64-entry golden-angle Vogel disc table lies STRICTLY inside the unit disc
//     and its byte digest is PINNED (the shader mirrors the exact integers); RtdSampleIndex covers ALL 64
//     spiral points EXACTLY ONCE over 64 frames for ANY pixel (gcd(39,64)==1 — the ground-truth
//     hash-independence rests on this); the first few indices of a pinned pixel are PINNED; distinct
//     pixels get distinct frame-0 indices (the hash actually decorrelates neighbors).
//   * RANGED ANY-HIT: an occluder between the surface and the light point occludes; an occluder BEYOND
//     the light point (t > kOne) does NOT (the far bound — the difference vs RT3's unbounded directional
//     test); an open ray does not; the boolean is ORDER-INDEPENDENT (shuffled primitive storage).
//   * ACCUMULATION EXACTNESS: vis over [0,N) == the SUM of the N per-frame accumulations (the
//     running-mean/temporal-accumulation identity, exact in integers); vis <= frames; the 64-frame
//     ground-truth visibility equals the direct full-table sum for EVERY pixel (hash-independent).
//   * THE INTEGER SHADE: monotonic non-decreasing in vis; vis==0 -> the ambient floor (strictly darker
//     than lit); vis==frames -> the analytic ambient + (1-ambient)*ndl shade; a miss -> background.
//   * THE PINNED INTEGER DIGESTS: FNV-1a64 of the 8-frame visibility buffer + the accumulated integer
//     image on the pinned RT2 scene + disc light at 96x72 — cross-compiler exact (MSVC == clang; these are
//     the same integers the Vulkan HW readback must reproduce).
//   * THE DENOISER IS LOAD-BEARING (float class, inequality pins — robust across libm): the penumbra band
//     (0 < 64-sample truth < 64) is NONEMPTY; denoised band variance < raw 1-sample STRICT; denoised MAE
//     vs the 64-sample truth < raw 1-sample STRICT; the float denoise is two-run byte-identical.
//
// Defines the RT2 scene locally with the frozen rtrace:: types (rtrace.h/ssgi.h/pcg.h BYTE-FROZEN,
// #included read-only via rtd.h). Pure C++ (hf_core), ASan-eligible.
#include "render/rtd.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "test_main.h"

using namespace hf;
namespace rt = hf::render::rtrace;
namespace rtd = hf::render::rtd;
using rt::fx;
using rt::kOne;
using rt::FxVec3;
using rt::F;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// The PINNED integer digests (computed once on the pinned scene below; MUST be identical MSVC + clang —
// pure integer math end-to-end).
static const uint64_t kPinnedDiscDigest  = 0x369d1d1fca87ea88ull;  // the 64-entry table bytes
static const uint64_t kPinnedVis8Digest  = 0x6cbed647eed45d4dull;  // 8-frame visibility @96x72
static const uint64_t kPinnedAccumDigest = 0x885b85ec5cbd4f3dull;  // accumulated integer image @96x72

// Build the SAME RT2 scene the showcases define (a ground AABB + a 4x4 sphere grid + two boxes).
struct Rt2Scene {
    std::vector<rt::RtSphere> spheres;
    std::vector<rt::RtAabb>   aabbs;
    rt::RtScene  scene;
    rt::RtCamera camera;
};

static Rt2Scene BuildRt2Scene() {
    Rt2Scene r;
    uint32_t nextPrim = 0;
    r.aabbs.push_back(rt::RtAabb{FxVec3{F(-20,1), F(-3,1), F(-20,1)},
                                 FxVec3{F(20,1),  F(-1,1), F(20,1)}, nextPrim++});
    for (int gz = 0; gz < 4; ++gz) {
        for (int gx = 0; gx < 4; ++gx) {
            fx cx = F(2 * gx - 3, 1);
            fx cz = F(2 + 2 * gz, 1);
            fx cy = F(0, 1);
            fx rad = (gz & 1) ? F(3, 4) : F(1, 1);
            r.spheres.push_back(rt::RtSphere{FxVec3{cx, cy, cz}, rad, nextPrim++});
        }
    }
    r.aabbs.push_back(rt::RtAabb{FxVec3{F(-9,2), F(-1,1), F(1,1)},
                                 FxVec3{F(-5,2), F(3,2),  F(5,2)}, nextPrim++});
    r.aabbs.push_back(rt::RtAabb{FxVec3{F(5,2),  F(-1,1), F(2,1)},
                                 FxVec3{F(9,2),  F(2,1),  F(7,2)}, nextPrim++});
    r.scene.spheres = std::span<const rt::RtSphere>(r.spheres);
    r.scene.aabbs   = std::span<const rt::RtAabb>(r.aabbs);
    r.scene.lightDir = rt::RtNormalize(FxVec3{F(4,10), F(8,10), F(-3,10)});
    r.scene.background = rt::PackRGBA8(34, 40, 56, 255);
    r.camera.eye     = FxVec3{F(0,1), F(2,1), F(-9,1)};
    r.camera.right   = FxVec3{kOne, 0, 0};
    r.camera.up      = FxVec3{0, kOne, 0};
    r.camera.forward = FxVec3{0, 0, kOne};
    r.camera.halfW   = F(7, 10);
    r.camera.halfH   = F(7, 10);
    return r;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- THE SAMPLE SEQUENCE ------------------------------------------------------------------------
    {
        // Every disc point strictly inside the unit disc (|p|^2 < kOne^2 in int64 — the light sample can
        // never leave the disc).
        bool inside = true;
        for (uint32_t j = 0; j < rtd::kRtdDiscSampleCount; ++j) {
            int64_t x = rtd::kRtdDiscSamples[j].x, y = rtd::kRtdDiscSamples[j].y;
            if (x * x + y * y >= (int64_t)kOne * (int64_t)kOne) inside = false;
        }
        check(inside, "all 64 Vogel disc points strictly inside the unit disc");

        // The table digest is PINNED (the shader mirrors these exact integers).
        uint64_t discDigest = rtd::RtdFnv1a64(rtd::kRtdDiscSamples, sizeof(rtd::kRtdDiscSamples));
        std::printf("disc table digest: 0x%016llx\n", (unsigned long long)discDigest);
        check(discDigest == kPinnedDiscDigest, "disc table digest matches the pin");

        // 64 frames cover ALL 64 spiral points EXACTLY once for ANY pixel (gcd(39,64)==1) — the
        // ground-truth hash-independence property.
        for (uint32_t pix : {0u, 1u, 12345u, 76799u}) {
            uint32_t seen[64] = {};
            for (uint32_t k = 0; k < 64; ++k) ++seen[rtd::RtdSampleIndex(pix, k)];
            bool all1 = true;
            for (uint32_t j = 0; j < 64; ++j) if (seen[j] != 1u) all1 = false;
            check(all1, "64 frames visit all 64 spiral points exactly once");
        }

        // The first indices of pixel 0 are PINNED: j(0,k) = (PcgHash(0x52544431,0) + 39k) & 63.
        uint32_t h0 = hf::pcg::PcgHash(rtd::kRtdPixelSeed, 0u) & 63u;
        check(rtd::RtdSampleIndex(0, 0) == h0, "sample index frame 0 == hash & 63");
        check(rtd::RtdSampleIndex(0, 1) == ((h0 + 39u) & 63u), "sample index frame 1 == (hash+39) & 63");
        check(rtd::RtdSampleIndex(0, 2) == ((h0 + 78u) & 63u), "sample index frame 2 == (hash+78) & 63");

        // Distinct pixels get decorrelated frame-0 indices (>= 2 distinct values across 16 neighbors).
        uint32_t distinct[64] = {};
        for (uint32_t pix = 0; pix < 16; ++pix) distinct[rtd::RtdSampleIndex(pix, 0)] = 1;
        uint32_t cnt = 0;
        for (uint32_t j = 0; j < 64; ++j) cnt += distinct[j];
        check(cnt >= 2, "the pixel hash decorrelates neighboring pixels' frame-0 sample");
    }

    // ---- RANGED ANY-HIT (the far bound vs RT3's unbounded directional test) --------------------------
    {
        // A unit sphere at (0,5,0); the shadow ray from the origin straight up with |dir| = 10 (the light
        // point at t == kOne is at (0,10,0), ABOVE the sphere -> occluded).
        std::vector<rt::RtSphere> s1 = {rt::RtSphere{FxVec3{0, F(5,1), 0}, kOne, 0}};
        rt::RtScene sc{};
        sc.spheres = std::span<const rt::RtSphere>(s1);
        rt::RtRay up{FxVec3{0, 0, 0}, FxVec3{0, F(10,1), 0}};
        check(rtd::TraceAnyHitRanged(up, sc, rt::kRtShadowMinT, kOne),
              "occluder between the surface and the light point -> occluded");

        // The SAME geometry but |dir| = 3 (the light point at (0,3,0), BELOW the sphere at (0,5,0) —
        // the occluder is BEYOND the light -> NOT occluded; RT3's unbounded test would say occluded).
        rt::RtRay shortUp{FxVec3{0, 0, 0}, FxVec3{0, F(3,1), 0}};
        check(!rtd::TraceAnyHitRanged(shortUp, sc, rt::kRtShadowMinT, kOne),
              "occluder BEYOND the light point (t > kOne) -> NOT occluded (the far bound)");

        // An open ray (no primitive along it) -> not occluded.
        rt::RtRay side{FxVec3{0, 0, 0}, FxVec3{F(10,1), 0, 0}};
        check(!rtd::TraceAnyHitRanged(side, sc, rt::kRtShadowMinT, kOne),
              "open ray -> NOT occluded");

        // Order-independence: shuffled primitive storage yields the IDENTICAL boolean (the HW early-out
        // == the SW full-scan guarantee).
        std::vector<rt::RtSphere> many = {
            rt::RtSphere{FxVec3{F(3,1), F(5,1), 0}, kOne, 0},   // off-axis (no hit)
            rt::RtSphere{FxVec3{0, F(5,1), 0},      kOne, 1},   // the occluder
            rt::RtSphere{FxVec3{F(-3,1), F(5,1), 0}, kOne, 2},  // off-axis (no hit)
        };
        std::vector<rt::RtSphere> shuffled = {many[2], many[0], many[1]};
        rt::RtScene scA{}, scB{};
        scA.spheres = std::span<const rt::RtSphere>(many);
        scB.spheres = std::span<const rt::RtSphere>(shuffled);
        check(rtd::TraceAnyHitRanged(up, scA, rt::kRtShadowMinT, kOne) ==
              rtd::TraceAnyHitRanged(up, scB, rt::kRtShadowMinT, kOne),
              "ranged any-hit is order-independent (shuffled storage, same boolean)");
    }

    // ---- The pinned scene pipeline @96x72 (small = fast; the SAME construction as the showcases) -----
    Rt2Scene r = BuildRt2Scene();
    const rtd::RtdAreaLight light = rtd::BuildRtd1Light();
    const uint32_t W = 96, H = 72;
    const size_t n = (size_t)W * H;
    const uint32_t kN = rtd::kRtdAccumFrames;

    std::vector<rtd::RtdSurface> surfaces;
    uint32_t hits = rtd::TracePrimarySurfaces(r.scene, light, r.camera, W, H, surfaces);
    check(hits > 0 && hits < n, "primary surfaces: 0 < hits < pixels");

    std::vector<uint32_t> vis1(n, 0), vis8(n, 0), vis64(n, 0);
    rtd::AccumulateSoftShadowVis(r.scene, light, surfaces, W, H, 0, 1, std::span<uint32_t>(vis1));
    rtd::AccumulateSoftShadowVis(r.scene, light, surfaces, W, H, 0, kN, std::span<uint32_t>(vis8));
    rtd::AccumulateSoftShadowVis(r.scene, light, surfaces, W, H, 0, rtd::kRtdTruthFrames,
                                 std::span<uint32_t>(vis64));

    // ---- ACCUMULATION EXACTNESS ----------------------------------------------------------------------
    {
        // vis over [0,N) == the SUM of the N per-frame accumulations (the temporal running-mean identity,
        // exact in integers).
        std::vector<uint32_t> visSum(n, 0);
        for (uint32_t k = 0; k < kN; ++k)
            rtd::AccumulateSoftShadowVis(r.scene, light, surfaces, W, H, k, 1, std::span<uint32_t>(visSum));
        check(std::memcmp(vis8.data(), visSum.data(), n * sizeof(uint32_t)) == 0,
              "vis[0,8) == sum of the 8 per-frame accumulations (temporal accumulation exact)");

        // vis bounded by frames.
        bool bounded = true;
        for (size_t i = 0; i < n; ++i)
            if (vis8[i] > kN || vis1[i] > 1u || vis64[i] > rtd::kRtdTruthFrames) bounded = false;
        check(bounded, "visibility counts bounded by the frame count");

        // The 64-frame ground truth is HASH-INDEPENDENT: it equals the direct full-table sum per pixel.
        bool truthOk = true;
        for (size_t i = 0; i < n && truthOk; ++i) {
            const rtd::RtdSurface& s = surfaces[i];
            if (s.hit.primIndex == rt::kRtMiss) { if (vis64[i] != 0) truthOk = false; continue; }
            rt::RtRay ray;
            ray.origin = rt::FxAdd(s.hit.pos, rt::FxScale(s.hit.normal, rt::kRtShadowEps));
            uint32_t direct = 0;
            for (uint32_t j = 0; j < rtd::kRtdDiscSampleCount; ++j) {
                ray.dir = rt::FxSub(rtd::RtdSamplePoint(light, j), ray.origin);
                if (!rtd::TraceAnyHitRanged(ray, r.scene, rt::kRtShadowMinT, kOne)) ++direct;
            }
            if (direct != vis64[i]) truthOk = false;
        }
        check(truthOk, "64-frame ground truth == the direct full-table sum (hash-independent)");
    }

    // ---- THE INTEGER SHADE -----------------------------------------------------------------------------
    {
        // Find a lit surface pixel (ndl > 0).
        size_t litIdx = n;
        for (size_t i = 0; i < n; ++i)
            if (surfaces[i].hit.primIndex != rt::kRtMiss && surfaces[i].ndl > 0) { litIdx = i; break; }
        check(litIdx < n, "a lit surface pixel exists");
        if (litIdx < n) {
            const rtd::RtdSurface& s = surfaces[litIdx];
            // Monotone non-decreasing in vis per channel; vis=0 strictly darker than vis=N.
            uint32_t prev = 0;
            bool mono = true;
            for (uint32_t v = 0; v <= kN; ++v) {
                uint32_t c = rtd::ShadeSoftShadowInt(s, v, kN, r.scene.background);
                if (v > 0) {
                    for (int ch = 0; ch < 3; ++ch)
                        if (((c >> (8 * ch)) & 0xFF) < ((prev >> (8 * ch)) & 0xFF)) mono = false;
                }
                prev = c;
            }
            check(mono, "integer shade monotone non-decreasing in vis");
            uint32_t dark = rtd::ShadeSoftShadowInt(s, 0, kN, r.scene.background);
            uint32_t lit = rtd::ShadeSoftShadowInt(s, kN, kN, r.scene.background);
            check(dark != lit, "vis=0 strictly darker than vis=N on a lit surface");

            // vis == frames reproduces the analytic ambient + (1-ambient)*ndl shade (visFrac == kOne).
            const fx ambient = (fx)(kOne * 18 / 100);
            fx lambert = rt::fxmul(rt::fxmul(kOne - ambient, s.ndl), kOne);
            fx diffuse = ambient + lambert;
            FxVec3 alb = rt::AlbedoFor(s.hit.primIndex);
            auto q = [&](fx ch) -> int32_t {
                return (int32_t)(((int64_t)rt::fxmul(ch, diffuse) * 255) >> rt::kFrac);
            };
            check(lit == rt::PackRGBA8(q(alb.x), q(alb.y), q(alb.z), 255),
                  "vis=N shade == the analytic ambient + (1-ambient)*ndl shade");

            // Miss -> background.
            rtd::RtdSurface miss{};
            check(rtd::ShadeSoftShadowInt(miss, 0, kN, r.scene.background) == r.scene.background,
                  "miss shades to background");
        }
    }

    // ---- THE PINNED INTEGER DIGESTS (cross-compiler exact — MSVC == clang) ---------------------------
    {
        std::vector<uint32_t> accum(n, 0);
        rtd::RenderSoftShadowImageInt(r.scene, surfaces, std::span<const uint32_t>(vis8), kN, W, H,
                                      std::span<uint32_t>(accum));
        uint64_t visDigest = rtd::RtdFnv1a64(vis8.data(), vis8.size() * sizeof(uint32_t));
        uint64_t accDigest = rtd::RtdFnv1a64(accum.data(), accum.size() * sizeof(uint32_t));
        std::printf("vis8 digest @96x72:  0x%016llx\n", (unsigned long long)visDigest);
        std::printf("accum digest @96x72: 0x%016llx\n", (unsigned long long)accDigest);
        check(visDigest == kPinnedVis8Digest, "8-frame visibility digest matches the pin");
        check(accDigest == kPinnedAccumDigest, "accumulated integer image digest matches the pin");
    }

    // ---- THE DENOISER IS LOAD-BEARING (float class — inequality pins + two-run identity) --------------
    {
        std::vector<float> visD = rtd::DenoiseSoftShadowVis(surfaces, std::span<const uint32_t>(vis8), kN,
                                                            W, H);
        rtd::RtdMetrics met = rtd::ComputeRtdMetrics(surfaces, std::span<const uint32_t>(vis1),
                                                     std::span<const uint32_t>(vis8), visD,
                                                     std::span<const uint32_t>(vis64), W, H);
        std::printf("metrics @96x72: band:%u var1:%.6f var8:%.6f varD:%.6f mae1:%.6f mae8:%.6f maeD:%.6f\n",
                    met.bandPixels, met.var1, met.var8, met.varD, met.mae1, met.mae8, met.maeD);
        check(met.bandPixels > 0, "penumbra band nonempty (soft shadows exist)");
        check(met.varD < met.var1, "denoised band variance < raw 1-sample (noise reduction STRICT)");
        check(met.maeD < met.mae1, "denoised MAE < raw 1-sample vs the 64-sample ground truth STRICT");

        // The float denoise is two-run byte-identical (pure function, no clock/RNG).
        std::vector<float> visD2 = rtd::DenoiseSoftShadowVis(surfaces, std::span<const uint32_t>(vis8), kN,
                                                             W, H);
        check(std::memcmp(visD.data(), visD2.data(), n * sizeof(float)) == 0,
              "float denoise two-run byte-identical");

        // The denoised image render is deterministic too.
        std::vector<uint32_t> img1(n, 0), img2(n, 0);
        rtd::RenderSoftShadowImageFloat(r.scene, surfaces, visD, W, H, std::span<uint32_t>(img1));
        rtd::RenderSoftShadowImageFloat(r.scene, surfaces, visD2, W, H, std::span<uint32_t>(img2));
        check(std::memcmp(img1.data(), img2.data(), n * sizeof(uint32_t)) == 0,
              "denoised image render two-run byte-identical");
    }

    if (g_fail == 0) {
        std::printf("rtd_test: ALL PASS\n");
        return 0;
    }
    std::printf("rtd_test: %d FAILURES\n", g_fail);
    return 1;
}
