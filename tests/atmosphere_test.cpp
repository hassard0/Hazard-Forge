// Slice AT1 — physical atmospheric scattering (Rayleigh + Henyey-Greenstein Mie single scattering).
// Pure CPU: the deterministic wavelength-dependent single-scattering integral (engine/render/
// atmosphere.h — the SAME math the --at1-sky-shot showcase and shaders/atmosphere.frag.hlsl use,
// mirrored verbatim). No device, ASan-eligible (links hf_core). Pins:
//   (a) DETERMINISM: two evaluations bit-identical + the FNV-1a-64 digest over a fixed
//       (viewDir, sunDir) grid pinned over RAW FLOAT BITS — EXACT MSVC == clang (no quantization
//       tolerance needed; atmo::DetExp + double-trig make the core cross-compiler bit-identical;
//       the clang check uses -ffp-contract=off, see kPinnedDigest below).
//   (b) THE PHYSICS: (i) the noon zenith is BLUE (B > G > R); (ii) the sun-ward horizon at sunset is
//       RED-shifted (R > B); (iii) the noon horizon is brighter than the zenith (haze whitening);
//       (iv) transmittance decreases monotonically with path length; (v) the away-from-sun horizon
//       is dimmer than the sun-ward horizon.
//   (c) SANITY: no NaN / negative radiance, bounded, over a sweep grid of view x sun directions.
#include "render/atmosphere.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace atmo = hf::render::atmo;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static float Luma(const Vec3& c) { return 0.299f * c.x + 0.587f * c.y + 0.114f * c.z; }

// The pinned digest — RAW FLOAT BITS over the fixed grid (see atmo::SkyDigest), EXACT (no
// quantization tolerance): verified bit-identical between MSVC 19.44 (/O2, the project build) and
// clang 22 (-O2 -ffp-contract=off). NOTE the clang check needs -ffp-contract=off (strict C++ source
// semantics): clang otherwise forms llvm.fmuladd from `b*b - c` in the ray-sphere intersect and
// CONSTANT-FOLDS it with FUSED (infinitely precise) semantics where the eye position is compile-time
// known, diverging from MSVC (which never contracts) by a few ULP. MSVC's vector-libm exp divergence
// (__vdecl_expf4 vs scalar expf) is fixed at the SOURCE by atmo::DetExp, so no flag is needed on the
// MSVC side.
static const unsigned long long kPinnedDigest = 0x5792dedf3715ec68ull;

int main() {
    HF_TEST_MAIN_INIT();
    const float kDeg = atmo::kPi / 180.0f;

    // ---- (a) Determinism: two evaluations bit-identical; the fixed-grid digest is pinned. ----
    {
        Vec3 sun = atmo::SunDirFromElevation(atmo::kSunElevLowRad);
        Vec3 v = atmo::PanoramaViewDir(0.31f, 0.42f);
        Vec3 a = atmo::SkyColor(v, sun);
        Vec3 b = atmo::SkyColor(v, sun);
        check(a.x == b.x && a.y == b.y && a.z == b.z,
              "SkyColor is bit-deterministic for the same (viewDir, sunDir)");

        unsigned long long d1 = atmo::SkyDigest();
        unsigned long long d2 = atmo::SkyDigest();
        check(d1 == d2, "SkyDigest is deterministic across evaluations");
        std::printf("atmosphere digest: 0x%016llx\n", d1);
        check(d1 == kPinnedDigest, "SkyDigest matches the pinned value");
    }

    // ---- (b)(i) The noon zenith is BLUE: Rayleigh scatters ~440nm ~5.7x more than ~680nm. ----
    {
        Vec3 sun = atmo::SunDirFromElevation(atmo::kSunElevNoonRad);
        Vec3 zenith = atmo::SkyColor(Vec3{0.0f, 1.0f, 0.0f}, sun);
        std::printf("noon zenith RGB   = (%.6f, %.6f, %.6f)\n",
                    (double)zenith.x, (double)zenith.y, (double)zenith.z);
        check(zenith.z > zenith.x, "noon zenith: B > R (the blue sky)");
        check(zenith.z > zenith.y && zenith.y > zenith.x,
              "noon zenith: B > G > R (Rayleigh wavelength ordering)");
        check(zenith.z > 0.05f, "noon zenith blue has non-trivial radiance");
    }

    // ---- (b)(ii) The sun-ward horizon at sunset (~2 deg) is RED-shifted. ----
    {
        Vec3 sun = atmo::SunDirFromElevation(atmo::kSunElevSunsetRad);
        // Looking toward the sun azimuth, just above the horizon (1 deg): the long grazing sun path
        // extincts blue, leaving the transmitted glow red.
        Vec3 v = atmo::PanoramaViewDir(0.5f, 1.0f * kDeg / atmo::kPanelElevMaxRad);
        Vec3 c = atmo::SkyColor(v, sun);
        std::printf("sunset sun-ward horizon RGB = (%.6f, %.6f, %.6f)\n",
                    (double)c.x, (double)c.y, (double)c.z);
        check(c.x > c.z, "sunset sun-ward horizon: R > B (the reddening)");
        check(c.x > c.y, "sunset sun-ward horizon: R > G");
        check(c.x > 0.05f, "sunset glow has non-trivial radiance");

        // Progression: the sun-ward horizon red:blue ratio GROWS as the sun drops (noon -> sunset).
        Vec3 sunNoon = atmo::SunDirFromElevation(atmo::kSunElevNoonRad);
        Vec3 cNoon = atmo::SkyColor(v, sunNoon);
        float ratioNoon = cNoon.x / cNoon.z;
        float ratioSunset = c.x / c.z;
        std::printf("sun-ward horizon R/B ratio: noon %.4f -> sunset %.4f\n",
                    (double)ratioNoon, (double)ratioSunset);
        check(ratioSunset > ratioNoon, "R/B ratio increases as the sun drops (reddening progression)");
    }

    // ---- (b)(iii) The noon horizon band is brighter/whiter than the zenith (long-path whitening).
    // The band is sampled 3 deg above the horizon, 90 deg in azimuth AWAY from the sun (so neither
    // the sun halo nor a grazing path dominates) — brighter holds across ALL azimuths there.
    // DOCUMENTED single-scattering limit: at grazing elevations (<~2 deg) extinction wins without
    // multiple scattering, so the last degree above the horizon dims instead of brightening.
    {
        Vec3 sun = atmo::SunDirFromElevation(atmo::kSunElevNoonRad);
        Vec3 zenith = atmo::SkyColor(Vec3{0.0f, 1.0f, 0.0f}, sun);
        Vec3 vh = atmo::PanoramaViewDir(0.5f + (90.0f * kDeg) / atmo::kPanelAzSpanRad,
                                        3.0f * kDeg / atmo::kPanelElevMaxRad);
        Vec3 horizon = atmo::SkyColor(vh, sun);
        std::printf("noon horizon RGB  = (%.6f, %.6f, %.6f)  (zenith luma %.6f, horizon luma %.6f)\n",
                    (double)horizon.x, (double)horizon.y, (double)horizon.z,
                    (double)Luma(zenith), (double)Luma(horizon));
        check(Luma(horizon) > Luma(zenith), "noon horizon band is brighter than the zenith");
        // WHITER: the horizon's blue dominance (B/R) is weaker than the zenith's.
        check(horizon.z / horizon.x < zenith.z / zenith.x,
              "noon horizon is whiter (lower B/R) than the zenith");
        // Pinned brightness ratio: the band exceeds the zenith by a non-trivial margin.
        std::printf("noon horizon/zenith luma ratio = %.4f\n",
                    (double)(Luma(horizon) / Luma(zenith)));
        check(Luma(horizon) / Luma(zenith) > 1.15f,
              "noon horizon/zenith luma ratio is pinned above 1.15");
    }

    // ---- (b)(iv) Transmittance decreases monotonically with path length (and is 1 at length 0). ----
    {
        Vec3 v = atmo::PanoramaViewDir(0.35f, 0.1f);   // a long slanted path
        Vec3 t0 = atmo::Transmittance(v, 0.0f);
        check(t0.x == 1.0f && t0.y == 1.0f && t0.z == 1.0f, "Transmittance(0) == 1");
        float prevR = 1.0f, prevG = 1.0f, prevB = 1.0f;
        bool mono = true, positive = true;
        for (float tp = 2000.0f; tp <= 100000.0f; tp += 2000.0f) {
            Vec3 tr = atmo::Transmittance(v, tp);
            if (!(tr.x < prevR && tr.y < prevG && tr.z < prevB)) mono = false;
            if (!(tr.x > 0.0f && tr.y > 0.0f && tr.z > 0.0f)) positive = false;
            prevR = tr.x; prevG = tr.y; prevB = tr.z;
        }
        check(mono, "transmittance strictly decreases with path length (every channel)");
        check(positive, "transmittance stays positive");
        // Blue extincts fastest (largest betaR): after a long path T_B < T_G < T_R.
        Vec3 tl = atmo::Transmittance(v, 100000.0f);
        std::printf("transmittance @100km slant = (%.6f, %.6f, %.6f)\n",
                    (double)tl.x, (double)tl.y, (double)tl.z);
        check(tl.z < tl.y && tl.y < tl.x, "blue extincts fastest along a long path (T_B < T_G < T_R)");
    }

    // ---- (b)(v) The away-from-sun sky is dimmer than sun-ward (low sun, single scattering). ----
    {
        Vec3 sun = atmo::SunDirFromElevation(atmo::kSunElevLowRad);
        float elev = 5.0f * kDeg;
        Vec3 vToward = atmo::PanoramaViewDir(0.5f, elev / atmo::kPanelElevMaxRad);   // sun azimuth
        // Directly opposite the sun in azimuth, same elevation.
        float ce = std::cos(elev);
        Vec3 vAway{0.0f, std::sin(elev), ce};   // sun is toward -Z; away is +Z
        Vec3 cT = atmo::SkyColor(vToward, sun);
        Vec3 cA = atmo::SkyColor(vAway, sun);
        std::printf("sun-ward luma %.6f vs away-from-sun luma %.6f (sun elev 15 deg)\n",
                    (double)Luma(cT), (double)Luma(cA));
        check(Luma(cT) > Luma(cA), "sun-ward sky is brighter than the away-from-sun sky");
        check(Luma(cA) > 0.0f, "away-from-sun sky is still lit (not black)");
    }

    // ---- (c) Sanity sweep: no NaN, no negative, bounded, over view x sun grids. ----
    {
        bool ok = true;
        float maxSeen = 0.0f;
        const float sunElevs[5] = {-2.0f, 0.0f, 2.0f, 15.0f, 70.0f};   // incl. below-horizon sun
        for (int s = 0; s < 5 && ok; ++s) {
            Vec3 sun = atmo::SunDirFromElevation(sunElevs[s] * kDeg);
            for (int az = -180; az <= 180 && ok; az += 20) {
                for (int el = -10; el <= 90 && ok; el += 5) {          // incl. below-horizon views
                    float a = (float)az * kDeg, e = (float)el * kDeg;
                    float c2 = std::cos(e);
                    Vec3 v{std::sin(a) * c2, std::sin(e), -std::cos(a) * c2};
                    Vec3 c = atmo::SkyColor(v, sun);
                    if (std::isnan(c.x) || std::isnan(c.y) || std::isnan(c.z)) ok = false;
                    if (c.x < 0.0f || c.y < 0.0f || c.z < 0.0f) ok = false;
                    if (c.x > 1000.0f || c.y > 1000.0f || c.z > 1000.0f) ok = false;
                    float m = (c.x > c.y) ? c.x : c.y; if (c.z > m) m = c.z;
                    if (m > maxSeen) maxSeen = m;
                }
            }
        }
        std::printf("sweep max HDR channel = %.6f\n", (double)maxSeen);
        check(ok, "no NaN / negative / unbounded radiance over the sweep grid");
    }

    // ---- Phase functions: closed-form + normalization spot checks. ----
    {
        // Rayleigh: 3/(16pi)*(1+cos^2) — forward == backward, minimum at 90 deg.
        float f = atmo::RayleighPhase(1.0f), s = atmo::RayleighPhase(0.0f), b = atmo::RayleighPhase(-1.0f);
        check(std::fabs(f - b) < 1e-7f, "Rayleigh phase is symmetric (forward == backward)");
        check(f > s && b > s, "Rayleigh phase has its minimum at 90 deg");
        check(std::fabs(f - 3.0f / (16.0f * atmo::kPi) * 2.0f) < 1e-6f,
              "Rayleigh forward matches the closed form");
        // Mie HG g=0.76 peaks hard forward.
        check(atmo::MiePhase(1.0f, atmo::kMieG) > atmo::MiePhase(0.9f, atmo::kMieG) &&
              atmo::MiePhase(0.9f, atmo::kMieG) > atmo::MiePhase(0.0f, atmo::kMieG),
              "HG Mie phase peaks toward the sun (g = 0.76 forward scatter)");
    }

    if (g_fail == 0) std::printf("atmosphere_test: all checks passed\n");
    else std::printf("atmosphere_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
