// Slice PTR1 — A BYTE-REPRODUCIBLE MULTI-BOUNCE PATH-TRACED REFERENCE RENDER (engine/render/pathtrace.h,
// hf::render::pt). Pure CPU (header-only, no device, no backend symbols). PTR1 is the per-pixel Monte-Carlo
// PATH TRACER the shipped RT arc lacked: render/rtrace.h::rt1_trace is DIRECT-ONLY (primary + closest-hit +
// integer Lambert; RT3 one shadow ray, RT4 one mirror bounce) — NO hemisphere-sampled indirect diffuse
// bounce, NO Monte-Carlo GI. PTR1 adds NEE + cosine-weighted multi-bounce global illumination, accumulated
// in PURE Q16.16 INTEGER with a HASH sampler + a FROZEN cos/sin LUT (integer literals) -> the render is
// BYTE-IDENTICAL run-to-run AND cross-platform (MSVC == clang == Apple, verified by the harness running both
// compilers). This is the moat UE5's float GPU path tracer + temporal denoiser structurally cannot make.
//
// What this test PINS:
//   (a) SAMPLER — PtSample01 is a pure function (byte-identical, pinned values) in [0,kOne); the primary
//       sub-pixel jitter STRATIFIES an NxN grid (each of the 16 strata covered exactly once, each jitter in
//       its own cell).
//   (b) GI — a multi-bounce render has strictly MORE total energy than a direct-only (1-bounce) render (the
//       indirect light is added), pinned; and COLOR BLEEDING is present (the white floor near the RED wall
//       reads red R>G,R>B; near the GREEN wall reads green G>R,G>B) — the ground-truth GI signature.
//   (c) REPRODUCIBLE — two full renders are byte-identical; the image DIGEST + energy are PINNED (the exact
//       integers MSVC and clang must both produce — the byte-reproducibility proof; true byte-identity, not
//       a quantized-float band).
//   (d) CONVERGENCE — 4x samples-per-pixel reduces the even/odd half-split noise metric (pinned monotone).
//   (e) RADIANCE SANITY — the NEE geometry term (PtDirectIrradianceAnalytic) matches the closed-form
//       E = Le*area*cosS*cosL/d^2 at a known point-under-the-light within Q16.16 rounding tolerance.
//
// Pure C++ (hf_core), ASan-eligible. pathtrace.h #includes render/rtrace.h + pcg/pcg.h + sim/fpx.h READ-ONLY
// (all BYTE-FROZEN).
#include "render/pathtrace.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace pt = hf::render::pt;
using pt::fx;
using pt::kOne;
using pt::kFrac;
using pt::FxVec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static double fxd(fx v) { return (double)v / (double)(int)kOne; }

int main() {
    HF_TEST_MAIN_INIT();

    pt::PtScene1 S = pt::BuildPtr1Scene();

    // ================= (a) SAMPLER =================
    {
        // Purity + PINNED values (the exact integers MSVC == clang must both produce).
        check(pt::PtSample01(10, 20, 64, 0, 0, 0) == 20163, "PtSample01: pinned value #1 (10,20,64,0,0,0)==20163");
        check(pt::PtSample01(5, 7, 64, 3, 2, 1) == 37293, "PtSample01: pinned value #2 (5,7,64,3,2,1)==37293");
        // Two calls identical (pure function, no RNG state / clock).
        check(pt::PtSample01(3, 4, 64, 1, 2, 3) == pt::PtSample01(3, 4, 64, 1, 2, 3),
              "PtSample01: pure function (two calls identical)");
        // Range [0,kOne).
        bool inRange = true;
        for (uint32_t s = 0; s < 200; ++s) {
            fx v = pt::PtSample01(7, 11, 64, s, s % 4, s % 3);
            if (v < 0 || v >= kOne) inRange = false;
        }
        check(inRange, "PtSample01: every draw is in [0,kOne)");

        // Stratification: for spp=16 (n=4) the primary sub-pixel jitter covers each of the 16 grid cells
        // exactly once, each jitter landing inside its own cell [gx/n,(gx+1)/n) x [gy/n,(gy+1)/n).
        const uint32_t spp = 16, n = 4, px = 12, py = 9, w = 64;
        int cellCount[16] = {0};
        bool jitterInCell = true;
        for (uint32_t s = 0; s < spp; ++s) {
            const fx ux = pt::PtSample01(px, py, w, s, 0, 100);
            const fx uy = pt::PtSample01(px, py, w, s, 0, 101);
            const uint32_t gx = s % n, gy = s / n;
            const fx jx = (fx)((((int64_t)gx << kFrac) + ux) / (int64_t)n);
            const fx jy = (fx)((((int64_t)gy << kFrac) + uy) / (int64_t)n);
            // jx must lie in [gx/n, (gx+1)/n).
            const fx cellLoX = (fx)(((int64_t)gx << kFrac) / n), cellHiX = (fx)(((int64_t)(gx + 1) << kFrac) / n);
            const fx cellLoY = (fx)(((int64_t)gy << kFrac) / n), cellHiY = (fx)(((int64_t)(gy + 1) << kFrac) / n);
            if (!(jx >= cellLoX && jx < cellHiX && jy >= cellLoY && jy < cellHiY)) jitterInCell = false;
            cellCount[gy * n + gx]++;
        }
        bool coverOnce = true;
        for (int i = 0; i < 16; ++i) if (cellCount[i] != 1) coverOnce = false;
        check(coverOnce, "stratification: each of the 16 sub-pixel strata covered exactly once");
        check(jitterInCell, "stratification: each jitter lands inside its own stratum cell");
    }

    // ================= (c) REPRODUCIBLE (pinned digest + energy; two runs byte-identical) =================
    {
        const uint32_t w = 128, h = 96, spp = 16; const int b = 4;
        std::vector<uint32_t> img((size_t)w * h), img2((size_t)w * h);
        pt::PtStats st = pt::PtRender(S, w, h, spp, b, std::span<uint32_t>(img));
        pt::PtStats st2 = pt::PtRender(S, w, h, spp, b, std::span<uint32_t>(img2));
        check(std::memcmp(img.data(), img2.data(), img.size() * sizeof(uint32_t)) == 0,
              "reproducible: two full renders are BYTE-IDENTICAL");
        check(st.digest == st2.digest, "reproducible: two-run digests equal");
        // The PINNED digest + energy (the exact integers MSVC == clang == Apple must all reproduce — the
        // true-byte-identity proof, pure integer + frozen LUT, NO float anywhere).
        check(st.digest == 0xced1f2656e1a08efull, "reproducible: PINNED image digest @128x96 s16 b4");
        check(st.energy == 2668416ull, "reproducible: PINNED image energy @128x96 s16 b4");
        // A fresh scene build renders byte-identically (pure function of the scene).
        pt::PtScene1 S2 = pt::BuildPtr1Scene();
        std::vector<uint32_t> img3((size_t)w * h);
        pt::PtStats st3 = pt::PtRender(S2, w, h, spp, b, std::span<uint32_t>(img3));
        check(st3.digest == st.digest, "reproducible: a fresh scene build renders the identical image");
    }

    // ================= (b) GI: multibounce energy > direct-only + color bleeding =================
    {
        const uint32_t w = 96, h = 72, spp = 16;
        std::vector<uint32_t> i1((size_t)w * h), i4((size_t)w * h);
        pt::PtStats s1 = pt::PtRender(S, w, h, spp, /*bounces*/ 1, std::span<uint32_t>(i1));
        pt::PtStats s4 = pt::PtRender(S, w, h, spp, /*bounces*/ 4, std::span<uint32_t>(i4));
        check(s4.energy > s1.energy, "GI: multi-bounce (b=4) has MORE energy than direct-only (b=1)");
        // PINNED energies (the exact indirect contribution).
        check(s1.energy == 1069523ull, "GI: PINNED direct-only energy (b=1) @96x72 s16");
        check(s4.energy == 1502675ull, "GI: PINNED multi-bounce energy (b=4) @96x72 s16");
        check(s4.digest == 0x4f7077da3135ca5eull, "GI: PINNED multi-bounce digest @96x72 s16");

        // Color bleeding: average the floor's left third (near the RED wall) vs right third (near GREEN).
        auto region = [&](const std::vector<uint32_t>& im, uint32_t x0, uint32_t x1, uint32_t y0, uint32_t y1,
                          double& R, double& G, double& B) {
            double r = 0, g = 0, bl = 0; uint64_t nn = 0;
            for (uint32_t y = y0; y < y1; ++y) for (uint32_t x = x0; x < x1; ++x) {
                uint32_t c = im[(size_t)y * w + x];
                r += c & 0xFF; g += (c >> 8) & 0xFF; bl += (c >> 16) & 0xFF; ++nn;
            }
            R = r / nn; G = g / nn; B = bl / nn;
        };
        double lr, lg, lb, rr, rg, rb;
        region(i4, 0, w / 4, h * 3 / 4, h, lr, lg, lb);       // floor near the LEFT (red) wall
        region(i4, w * 3 / 4, w, h * 3 / 4, h, rr, rg, rb);   // floor near the RIGHT (green) wall
        check(lr > lg && lr > lb, "GI color bleed: white floor near the RED wall reads red (R>G, R>B)");
        check(rg > rr && rg > rb, "GI color bleed: white floor near the GREEN wall reads green (G>R, G>B)");
        // The RED tint (R-G) is stronger on the red-wall floor; the GREEN tint (G-R) stronger on the
        // green-wall floor — the differential signature (robust to the scene's geometric asymmetry).
        check((lr - lg) > (rr - rg), "GI color bleed: red tint (R-G) stronger on the red-wall floor");
        check((rg - rr) > (lg - lr), "GI color bleed: green tint (G-R) stronger on the green-wall floor");
    }

    // ================= (d) CONVERGENCE: 4x samples reduces the noise metric =================
    {
        const uint32_t w = 64, h = 48; const int b = 4;
        std::vector<uint32_t> a((size_t)w * h), c((size_t)w * h);
        pt::PtStats s16 = pt::PtRender(S, w, h, 16, b, std::span<uint32_t>(a), /*wantNoise*/ true);
        pt::PtStats s64 = pt::PtRender(S, w, h, 64, b, std::span<uint32_t>(c), /*wantNoise*/ true);
        check(s64.noiseMAE < s16.noiseMAE, "convergence: 4x spp reduces the half-split noise metric");
        check(s16.noiseMAE == 25u, "convergence: PINNED noise @spp16");
        check(s64.noiseMAE == 13u, "convergence: PINNED noise @spp64 (< spp16 -> converging)");
    }

    // ================= (e) RADIANCE SANITY: NEE geometry term vs closed form =================
    {
        // A floor point directly BELOW the light center: wi = (0, 3.99, 0), cosS = cosL = 1.
        FxVec3 point{0, 0, pt::F(2, 1)};
        FxVec3 normal{0, kOne, 0};
        FxVec3 E = pt::PtDirectIrradianceAnalytic(point, normal, S.light);
        // Closed form E = Le * area * cosS * cosL / d^2 (per channel), computed in double.
        const double d2 = std::pow(fxd(pt::F(399, 100)), 2.0);   // (3.99)^2
        const double area = fxd(S.light.area);
        const double exx = fxd(S.light.emission.x) * area * 1.0 * 1.0 / d2;
        const double got = fxd(E.x);
        check(std::fabs(got - exx) < 0.01, "radiance sanity: NEE geometry term matches closed form E=Le*A*cos*cos/d^2");
        // PINNED exact fx (the integer estimator value MSVC == clang must reproduce).
        check(E.x == 106668, "radiance sanity: PINNED analytic irradiance E.x (fx)");
    }

    if (g_fail == 0) { std::printf("pathtrace_test OK\n"); return 0; }
    std::printf("pathtrace_test: %d failures\n", g_fail);
    return 1;
}
