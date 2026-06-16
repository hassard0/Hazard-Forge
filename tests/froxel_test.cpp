// Slice CS — Froxel Volumetric Fog (sun single-scattering). Pure CPU math: the exponential Z-slice
// mapping + its exact inverse, the height-fog density, the Henyey-Greenstein phase, and the front-to-back
// single-scattering IntegrateStep. No device, ASan-eligible (links hf_core). Exercises the EXACT math the
// --froxelfog-shot showcase + the froxel_inject/integrate/apply shaders use (engine/render/froxel.h).
//
// Properties pinned (per the spec):
//   * Z-SLICE INVERSE (the apply-sampling correctness foundation): ViewZToSlice(SliceZ(k)) == k across
//     the grid; SliceZ monotone increasing; covers [zNear,zFar]; SliceZ(0)==zNear, SliceZ(dimZ)==zFar.
//   * DENSITY: height falloff (higher y -> less fog with positive falloff); baseDensity==0 -> 0
//     everywhere; never negative.
//   * PHASE: forward-peaked (Phase(1,g>0) > Phase(-1,g>0)); isotropic (Phase(cos,0) constant); finite.
//   * INTEGRATION: zero-density column -> transmittance==1 & inScatter==0 (the NO-OP proof); a constant-
//     density sigma over length L -> transmittance==exp(-sigma*L) (analytic Beer-Lambert); transmittance
//     monotone decreasing along the march; front-to-back weighting (a near scatterer contributes at full
//     transmittance, a far one is attenuated).
//   * DETERMINISM: same inputs -> bit-identical outputs (pure functions, no RNG/time).
#include "render/froxel.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>

namespace froxel = hf::render::froxel;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

int main() {
    froxel::FroxelGrid g;            // 16x9x64 over [0.5, 80] (the showcase grid)
    g.dimX = 16; g.dimY = 9; g.dimZ = 64; g.zNear = 0.5f; g.zFar = 80.0f;

    // ---- Z-SLICE INVERSE: ViewZToSlice(SliceZ(k)) == k for real-valued k across [0,dimZ]. This is the
    //      apply's depth->slice sampling correctness — a mismatch is an off-by-one that breaks the
    //      density=0 == scene proof. ----
    {
        bool inverseOk = true;
        for (int i = 0; i <= g.dimZ; ++i) {
            float k = (float)i;
            float vz = froxel::SliceZ(g, k);
            float kBack = froxel::ViewZToSlice(g, vz);
            if (!approx(kBack, k, 1e-3f)) inverseOk = false;
        }
        // Also at fractional k (the apply maps a continuous depth).
        for (float k : {0.5f, 7.25f, 31.5f, 63.9f}) {
            float vz = froxel::SliceZ(g, k);
            if (!approx(froxel::ViewZToSlice(g, vz), k, 1e-3f)) inverseOk = false;
        }
        check(inverseOk, "ViewZToSlice(SliceZ(k)) == k across the grid (exact inverse)");
    }

    // ---- SliceZ endpoints + monotonicity + coverage. ----
    {
        check(approx(froxel::SliceZ(g, 0.0f), g.zNear), "SliceZ(0) == zNear");
        check(approx(froxel::SliceZ(g, (float)g.dimZ), g.zFar, 1e-2f), "SliceZ(dimZ) == zFar");
        bool monotone = true;
        float prev = froxel::SliceZ(g, 0.0f);
        for (int i = 1; i <= g.dimZ; ++i) {
            float cur = froxel::SliceZ(g, (float)i);
            if (cur <= prev) monotone = false;
            prev = cur;
        }
        check(monotone, "SliceZ monotone increasing in k");
        // Coverage: every slice center lies strictly inside [zNear, zFar].
        bool covered = true;
        for (int fz = 0; fz < g.dimZ; ++fz) {
            float c = froxel::SliceCenterViewZ(g, fz);
            if (!(c > g.zNear - 1e-3f && c < g.zFar + 1e-3f)) covered = false;
        }
        check(covered, "all slice centers lie within [zNear, zFar]");
        // Clamping outside the range.
        check(froxel::ViewZToSlice(g, g.zNear * 0.5f) == 0.0f, "ViewZToSlice clamps below zNear -> 0");
        check(froxel::ViewZToSlice(g, g.zFar * 2.0f) == (float)g.dimZ, "ViewZToSlice clamps above zFar -> dimZ");
    }

    // ---- DENSITY: height falloff, baseDensity==0 -> 0, non-negative. ----
    {
        const float base = 0.08f, falloff = 0.15f, href = 0.0f;
        // baseDensity == 0 -> EXACTLY 0 at any height (the no-op foundation).
        for (float y : {-5.0f, 0.0f, 3.0f, 20.0f}) {
            check(froxel::Density(Vec3{0, y, 0}, 0.0f, falloff, href) == 0.0f,
                  "Density(baseDensity==0) == 0 EXACTLY");
        }
        // Positive falloff: higher y -> less fog. Strictly decreasing with altitude.
        float dLow  = froxel::Density(Vec3{0, 0.0f, 0}, base, falloff, href);
        float dMid  = froxel::Density(Vec3{0, 5.0f, 0}, base, falloff, href);
        float dHigh = froxel::Density(Vec3{0, 12.0f, 0}, base, falloff, href);
        check(dLow > dMid && dMid > dHigh, "Density decreases with altitude (positive height falloff)");
        check(approx(dLow, base), "Density at heightRef == baseDensity");
        // Non-negative everywhere (even far above where exp underflows toward 0).
        check(froxel::Density(Vec3{0, 1000.0f, 0}, base, falloff, href) >= 0.0f, "Density non-negative");
    }

    // ---- PHASE: forward-peak, isotropic, finite. ----
    {
        const float g1 = 0.6f;
        check(froxel::Phase(1.0f, g1) > froxel::Phase(-1.0f, g1),
              "Phase forward-peaked: Phase(1,g>0) > Phase(-1,g>0)");
        // Isotropic at g==0: Phase(cos,0) constant == 1/(4*pi).
        float p0 = froxel::Phase(0.0f, 0.0f);
        bool iso = true;
        for (float c : {-1.0f, -0.3f, 0.5f, 1.0f}) {
            if (!approx(froxel::Phase(c, 0.0f), p0, 1e-6f)) iso = false;
        }
        check(iso, "Phase(cos,0) isotropic (constant)");
        check(approx(p0, 1.0f / (4.0f * froxel::kPi), 1e-6f), "Phase(cos,0) == 1/(4*pi)");
        // Finite for all cos in [-1,1] at a strong g.
        bool finite = true;
        for (float c = -1.0f; c <= 1.0f; c += 0.1f) {
            float p = froxel::Phase(c, 0.9f);
            if (!std::isfinite(p) || p < 0.0f) finite = false;
        }
        check(finite, "Phase finite + non-negative for all cos in [-1,1]");
    }

    // ---- INTEGRATION NO-OP: a column of ZERO density/scatter leaves transmittance==1, inScatter==0
    //      EXACTLY (the per-step foundation of density=0 == scene). ----
    {
        float T = 1.0f;
        Vec3 L{0, 0, 0};
        for (int s = 0; s < g.dimZ; ++s) {
            float stepLen = froxel::SliceZ(g, (float)(s + 1)) - froxel::SliceZ(g, (float)s);
            froxel::IntegrateStep(Vec3{0, 0, 0}, 0.0f, stepLen, L, T);
        }
        check(T == 1.0f, "zero-density column -> transmittance == 1 EXACTLY (no-op)");
        check(L.x == 0.0f && L.y == 0.0f && L.z == 0.0f, "zero-density column -> inScatter == 0 EXACTLY (no-op)");
    }

    // ---- INTEGRATION ANALYTIC: a constant extinction sigma over total length L -> transmittance ==
    //      exp(-sigma*L) (Beer-Lambert). Uniform steps over [0,L]. ----
    {
        const float sigma = 0.3f;
        const float Ltot = 20.0f;
        const int   N = 256;
        const float stepLen = Ltot / (float)N;
        float T = 1.0f;
        Vec3 L{0, 0, 0};
        bool monotone = true;
        float prevT = T;
        for (int s = 0; s < N; ++s) {
            froxel::IntegrateStep(Vec3{0, 0, 0}, sigma, stepLen, L, T);
            if (T > prevT) monotone = false;        // transmittance never increases
            prevT = T;
        }
        check(approx(T, std::exp(-sigma * Ltot), 1e-4f),
              "constant-sigma column -> transmittance == exp(-sigma*L) (analytic Beer-Lambert)");
        check(monotone, "transmittance monotone decreasing along the march");
        check(L.x == 0.0f, "zero scatter -> inScatter stays 0 even with extinction");
    }

    // ---- FRONT-TO-BACK WEIGHTING: a near scatterer contributes at FULL transmittance; the same
    //      scatterer placed BEHIND attenuating fog contributes LESS. ----
    {
        const float sigma = 0.5f;
        const float stepLen = 1.0f;
        const Vec3 scatter{1.0f, 1.0f, 1.0f};   // unit in-scatter per unit length

        // Case A: the scatterer is in the FIRST (nearest) froxel at full transmittance, then 8 absorbing
        // froxels follow.
        {
            float T = 1.0f; Vec3 L{0, 0, 0};
            froxel::IntegrateStep(scatter, sigma, stepLen, L, T);   // near scatterer first (T==1 when added)
            for (int s = 0; s < 8; ++s) froxel::IntegrateStep(Vec3{0, 0, 0}, sigma, stepLen, L, T);
            check(approx(L.x, 1.0f * stepLen, 1e-5f),
                  "a near scatterer contributes at FULL transmittance (T==1)");
        }
        // Case B: 8 absorbing froxels FIRST, then the same scatterer behind them -> attenuated by the
        // accumulated transmittance exp(-sigma*8).
        {
            float T = 1.0f; Vec3 L{0, 0, 0};
            for (int s = 0; s < 8; ++s) froxel::IntegrateStep(Vec3{0, 0, 0}, sigma, stepLen, L, T);
            float Tbefore = T;
            froxel::IntegrateStep(scatter, sigma, stepLen, L, T);
            check(approx(L.x, Tbefore * 1.0f * stepLen, 1e-5f),
                  "a far scatterer contributes attenuated by the front transmittance");
            check(Tbefore < 1.0f, "transmittance dropped below 1 in front of the far scatterer");
        }
    }

    // ---- DETERMINISM: same inputs -> bit-identical outputs. ----
    {
        float T1 = 1.0f, T2 = 1.0f; Vec3 L1{0,0,0}, L2{0,0,0};
        for (int s = 0; s < 32; ++s) {
            Vec3 sc{0.2f, 0.3f, 0.4f};
            froxel::IntegrateStep(sc, 0.1f, 0.5f, L1, T1);
            froxel::IntegrateStep(sc, 0.1f, 0.5f, L2, T2);
        }
        check(T1 == T2 && L1.x == L2.x && L1.y == L2.y && L1.z == L2.z,
              "IntegrateStep is deterministic (bit-identical across runs)");
        check(froxel::SliceZ(g, 17.3f) == froxel::SliceZ(g, 17.3f), "SliceZ deterministic");
        check(froxel::Phase(0.3f, 0.5f) == froxel::Phase(0.3f, 0.5f), "Phase deterministic");
    }

    if (g_fail == 0) { std::printf("froxel_test: all checks passed\n"); return 0; }
    std::printf("froxel_test: %d FAILURES\n", g_fail);
    return 1;
}
