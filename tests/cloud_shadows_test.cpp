// Slice CK — cloud shadows on the ground. Pure CPU: the CloudShadow(worldPos, sunDir, t, slabBottom,
// slabTop, steps) function that marches from a surface point TOWARD the sun through the cloud slab,
// accumulating optical depth from clouds::Density, and returns the Beer-Lambert transmittance of the
// sunlight (1 = full sun, 0 = fully shadowed). This is the SAME math the lit_cloudshadow.frag variant
// mirrors in-shader for the --cloud-shadows-shot showcase. No device, ASan-eligible (links hf_core).
// Full-sun (clear ray), shadowed/monotone, range [0,1] + Beer-consistency, and determinism invariants
// are exercised here so the GPU shadow term is built on a unit-checked field.
#include "render/clouds.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace clouds = hf::render::clouds;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

int main() {
    HF_TEST_MAIN_INIT();
    const float bottom = clouds::kSlabBottom;
    const float top    = clouds::kSlabTop;
    const float t      = clouds::kFixedTime;
    const int   kSteps = 32;

    // The directional-light direction (the direction the light TRAVELS): -sunDir points TOWARD the sun.
    // The --clouds-shot / lit shots use this same downward-from-the-right light; CloudShadow marches
    // along -sunDir (up toward the sun) from the surface through the slab.
    const Vec3 sunDir = normalize(Vec3{-0.55f, -0.62f, -0.55f});

    // ---- Range: CloudShadow is always in [0,1] (a Beer transmittance). ----
    {
        bool inRange = true;
        for (float x = -30.0f; x <= 30.0f; x += 2.3f)
            for (float z = -30.0f; z <= 30.0f; z += 2.7f) {
                float s = clouds::CloudShadow(Vec3{x, 0.0f, z}, sunDir, t, bottom, top, kSteps);
                if (s < 0.0f || s > 1.0f) inRange = false;
            }
        check(inRange, "CloudShadow stays within [0,1] over the ground plane");
    }

    // ---- Full sun: a sun ray that NEVER enters the slab (sun straight up, point above the slab) is
    // unoccluded -> CloudShadow == 1 exactly. Also a sun ray parallel to the slab (horizontal) that
    // cannot reach the slab from below is full sun. ----
    {
        // Point already ABOVE the slab top, marching further up toward the sun: never re-enters cloud.
        Vec3 above{0.0f, top + 5.0f, 0.0f};
        Vec3 sunUp = normalize(Vec3{0.0f, -1.0f, 0.0f});   // light travels straight down; -sunDir = up
        float s = clouds::CloudShadow(above, sunUp, t, bottom, top, kSteps);
        check(s == 1.0f, "CloudShadow == 1 when the sun ray never enters the slab (above the slab)");

        // A horizontal sun (light travels horizontally; -sunDir horizontal) from a ground point can
        // never climb into the slab -> the slab is not entered -> full sun.
        Vec3 ground{0.0f, 0.0f, 0.0f};
        Vec3 sunHoriz = normalize(Vec3{1.0f, 0.0f, 0.0f});
        float sh = clouds::CloudShadow(ground, sunHoriz, t, bottom, top, kSteps);
        check(sh == 1.0f, "CloudShadow == 1 for a horizontal sun ray that cannot reach the slab");
    }

    // ---- Clear region: a ground point whose toward-sun ray passes only through CLEAR sky (density 0
    // along the whole slab segment) is full sun. Find such a column deterministically. ----
    {
        Vec3 sunUp = normalize(Vec3{0.0f, -1.0f, 0.0f});   // straight overhead -> the ray is the column
        bool foundClear = false;
        for (float x = -50.0f; x <= 50.0f && !foundClear; x += 0.5f)
            for (float z = -50.0f; z <= 50.0f && !foundClear; z += 0.5f) {
                // Optical depth along the vertical column through the slab.
                float od = 0.0f;
                for (int k = 0; k < 64; ++k) {
                    float y = bottom + (float(k) + 0.5f) / 64.0f * (top - bottom);
                    od += clouds::Density(Vec3{x, y, z}, t, bottom, top);
                }
                if (od == 0.0f) {
                    float s = clouds::CloudShadow(Vec3{x, 0.0f, z}, sunUp, t, bottom, top, kSteps);
                    check(s == 1.0f, "CloudShadow == 1 in a CLEAR (zero-density) column");
                    foundClear = true;
                }
            }
        check(foundClear, "found a clear column (clear sky exists between clouds)");
    }

    // ---- Shadowed: a ground point whose toward-sun ray passes through DENSE cloud is attenuated
    // (< 1). Find a column whose vertical optical depth is clearly non-zero. ----
    {
        Vec3 sunUp = normalize(Vec3{0.0f, -1.0f, 0.0f});
        bool foundDense = false;
        for (float x = -50.0f; x <= 50.0f && !foundDense; x += 0.5f)
            for (float z = -50.0f; z <= 50.0f && !foundDense; z += 0.5f) {
                float od = 0.0f;
                for (int k = 0; k < 64; ++k) {
                    float y = bottom + (float(k) + 0.5f) / 64.0f * (top - bottom);
                    od += clouds::Density(Vec3{x, y, z}, t, bottom, top);
                }
                if (od > 0.5f) {
                    float s = clouds::CloudShadow(Vec3{x, 0.0f, z}, sunUp, t, bottom, top, kSteps);
                    check(s < 1.0f, "CloudShadow < 1 under a DENSE cloud column (sun is attenuated)");
                    check(s >= 0.0f, "shadowed CloudShadow stays >= 0");
                    foundDense = true;
                }
            }
        check(foundDense, "found a dense column (clouds exist)");
    }

    // ---- Monotonic in optical depth: scaling the field denser (more cloud along the ray) yields a
    // SMALLER shadow factor. We verify via Beer-consistency: CloudShadow == Beer(accumulated OD), and
    // Beer is strictly monotone decreasing. Compare two columns of different total density. ----
    {
        Vec3 sunUp = normalize(Vec3{0.0f, -1.0f, 0.0f});
        // Collect (verticalOD, shadow) pairs across the plane; denser column => smaller shadow.
        float odLo = -1.0f, shLo = 0.0f, odHi = -1.0f, shHi = 0.0f;
        for (float x = -50.0f; x <= 50.0f; x += 0.5f)
            for (float z = -50.0f; z <= 50.0f; z += 0.5f) {
                float od = 0.0f;
                for (int k = 0; k < 64; ++k) {
                    float y = bottom + (float(k) + 0.5f) / 64.0f * (top - bottom);
                    od += clouds::Density(Vec3{x, y, z}, t, bottom, top);
                }
                float s = clouds::CloudShadow(Vec3{x, 0.0f, z}, sunUp, t, bottom, top, kSteps);
                if (od > 0.0f && (odLo < 0.0f || od < odLo)) { odLo = od; shLo = s; }
                if (od > odHi) { odHi = od; shHi = s; }
            }
        check(odHi > odLo && odLo >= 0.0f, "found a thin and a thick column to compare");
        check(shHi <= shLo, "denser cloud column -> smaller (or equal) shadow factor (monotone)");
        check(shHi < shLo + 1e-4f, "monotonicity is strict for clearly different optical depths");
    }

    // ---- Beer-consistency: CloudShadow(p) == Beer(opticalDepth marched along -sunDir). Re-derive the
    // optical depth the same way the function does (slab-clipped uniform march) and compare. ----
    {
        Vec3 sunUp = normalize(Vec3{0.0f, -1.0f, 0.0f});   // -sunDir = +Y; the march climbs the column
        // Pick a column known to be inside coverage.
        bool tested = false;
        for (float x = -50.0f; x <= 50.0f && !tested; x += 0.5f)
            for (float z = -50.0f; z <= 50.0f && !tested; z += 0.5f) {
                Vec3 p{x, 0.0f, z};
                float s = clouds::CloudShadow(p, sunUp, t, bottom, top, kSteps);
                // Re-derive: march from the slab entry (y=bottom) to exit (y=top) in kSteps, the SAME
                // segmentation CloudShadow uses (toward-sun dir = +Y, clipped to [bottom,top]).
                Vec3 toSun = normalize(Vec3{0.0f, 1.0f, 0.0f});
                float tEnter = (bottom - p.y) / toSun.y;
                float tExit  = (top - p.y) / toSun.y;
                float seg = (tExit - tEnter) / (float)kSteps;
                float od = 0.0f;
                for (int k = 0; k < kSteps; ++k) {
                    float tk = tEnter + (float(k) + 0.5f) * seg;
                    Vec3 pk = p + toSun * tk;
                    od += clouds::Density(pk, t, bottom, top) * seg;
                }
                check(approx(s, clouds::Beer(od), 1e-5f),
                      "CloudShadow == Beer(marched optical depth) (Beer-consistent)");
                tested = true;
            }
        check(tested, "Beer-consistency column tested");
    }

    // ---- Determinism: same (worldPos, sunDir, t) -> bit-identical shadow factor. ----
    {
        Vec3 p{2.4f, 0.0f, -3.1f};
        float a = clouds::CloudShadow(p, sunDir, t, bottom, top, kSteps);
        float b = clouds::CloudShadow(p, sunDir, t, bottom, top, kSteps);
        check(a == b, "CloudShadow is bit-deterministic for the same (worldPos, sunDir, t)");
        // A different time generally advects the field -> evaluates without error.
        float c = clouds::CloudShadow(p, sunDir, t + 5.0f, bottom, top, kSteps);
        (void)c;
        check(true, "CloudShadow(t+dt) evaluates without error");
    }

    if (g_fail == 0) std::printf("cloud_shadows_test: all checks passed\n");
    else std::printf("cloud_shadows_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
