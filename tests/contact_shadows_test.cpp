// Slice CT — Screen-Space Contact Shadows. Pure CPU math: the screen-space depth ray-march toward the
// sun (engine/render/contact_shadows.h). No device, ASan-eligible (links hf_core). Exercises the EXACT
// march the --contactshadow-shot showcase + contact_shadows.frag.hlsl shader use.
//
// Properties pinned (per the spec):
//   * DISABLED = IDENTITY (the core proof): RayMarchShadow(..., maxDist=0) == 1 and (..., steps=0) == 1
//     for ANY input — no march -> no occluder -> fully lit.
//   * CLEAR RAY = LIT: an empty/flat depth field along the ray -> factor 1 (no occluder).
//   * OCCLUDER WITHIN THICKNESS = SHADOWED: a depth sample placed in FRONT of the ray within the
//     thickness window -> factor < 1; a CLOSER occluder (hit earlier in the march) -> <= the farther one
//     (monotone); deterministic value.
//   * OCCLUDER BEYOND THICKNESS = NO FALSE SHADOW: an occluder in front of the ray but FARTHER than
//     `thickness` -> factor 1 (no haloing — the safety case).
//   * BIAS = NO SELF-OCCLUDE: a flat lit surface (scene depth ~= ray depth along the march) -> factor 1.
//   * DETERMINISM: same inputs -> bit-identical factor.
#include "render/contact_shadows.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>

namespace contact = hf::render::contact;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    // Shared projection params (a typical 60-deg-fovY 16:9 view; Vulkan yFlip = -1).
    const float kTanHalfFovY = std::tan(0.5f * 1.04719755f);
    const float kAspect = 1280.0f / 720.0f;
    const float kYFlip = -1.0f;
    const float kThickness = 0.5f;
    const float kBias = 0.02f;

    // A camera-facing fragment a few units in front of the camera, on the optical axis (projects to the
    // screen center). The sun is up-and-to-the-side so the march walks laterally across the screen.
    const float d0 = 5.0f;
    Vec3 viewPos{0.0f, 0.0f, -d0};
    // View-space direction TO the sun: purely lateral in the view plane (+x screen-rightward, a little
    // +y), with ZERO z so the marched ray stays at the constant view-distance d0. A flat depth field at
    // d0 then yields diff == 0 along the whole march (no self-occlusion). Normalized internally.
    Vec3 toSun = hf::math::normalize(Vec3{1.0f, 0.4f, 0.0f});

    // ---- DISABLED = IDENTITY: maxDist=0 -> 1, steps=0 -> 1, for ANY (even occluding) field. ----
    {
        // A field that WOULD occlude (a wall much closer than the ray everywhere) — proves the identity
        // is unconditional (no march, regardless of the scene).
        auto wall = [&](float, float) { return d0 - 0.2f; };
        float f0 = contact::RayMarchShadow(wall, viewPos, toSun, /*maxDist*/ 0.0f, /*steps*/ 16,
                                           kThickness, kBias, kTanHalfFovY, kAspect, kYFlip);
        check(f0 == 1.0f, "RayMarchShadow(maxDist=0) == 1 EXACTLY (no march -> fully lit)");
        float fs = contact::RayMarchShadow(wall, viewPos, toSun, /*maxDist*/ 1.0f, /*steps*/ 0,
                                           kThickness, kBias, kTanHalfFovY, kAspect, kYFlip);
        check(fs == 1.0f, "RayMarchShadow(steps=0) == 1 EXACTLY (no march -> fully lit)");
        float fn = contact::RayMarchShadow(wall, viewPos, toSun, /*maxDist*/ -1.0f, /*steps*/ 16,
                                           kThickness, kBias, kTanHalfFovY, kAspect, kYFlip);
        check(fn == 1.0f, "RayMarchShadow(maxDist<0) == 1 EXACTLY (clamped no-op)");
    }

    // ---- CLEAR RAY = LIT: a flat depth field at the ray's own depth places no surface in front of the
    //      ray -> no occluder -> factor 1. (The march recedes slightly, so a constant d0 is at or behind
    //      the ray, never in front beyond the bias.) ----
    {
        auto flat = [&](float, float) { return d0; };
        float f = contact::RayMarchShadow(flat, viewPos, toSun, /*maxDist*/ 1.0f, /*steps*/ 16,
                                          kThickness, kBias, kTanHalfFovY, kAspect, kYFlip);
        check(f == 1.0f, "clear/flat depth field -> factor 1 (no occluder)");
        // An EMPTY field (cleared depth == 0, background) also yields 1 (no surface to occlude against).
        auto empty = [&](float, float) { return 0.0f; };
        float fe = contact::RayMarchShadow(empty, viewPos, toSun, /*maxDist*/ 1.0f, /*steps*/ 16,
                                           kThickness, kBias, kTanHalfFovY, kAspect, kYFlip);
        check(fe == 1.0f, "empty (background) depth field -> factor 1 (nothing to occlude)");
    }

    // ---- BIAS = NO SELF-OCCLUDE: a flat surface whose depth is exactly the ray's view-distance MINUS a
    //      hair (within the bias) must NOT self-shadow. Model the scene depth as the ray's own distance
    //      offset by less than the bias -> diff <= bias -> not occluded -> factor 1. ----
    {
        // Return the ray's view-distance at the sampled point minus a tiny epsilon (< bias). Because we
        // do not know t from (u,v) directly, approximate a flat surface a hair in front of the camera
        // plane through viewPos: a constant just-barely-closer depth never exceeds the bias for the
        // near steps, and the far steps recede so the surface falls behind the ray. The net is no hit.
        auto flatBias = [&](float, float) { return d0 - 0.01f; };  // 0.01 < kBias (0.02)
        float f = contact::RayMarchShadow(flatBias, viewPos, toSun, /*maxDist*/ 0.3f, /*steps*/ 16,
                                          kThickness, kBias, kTanHalfFovY, kAspect, kYFlip);
        check(f == 1.0f, "flat surface within the bias -> no self-occlusion (factor 1)");
    }

    // ---- OCCLUDER WITHIN THICKNESS = SHADOWED. Place a surface CLOSER than the ray (in front by more
    //      than the bias, within the thickness window) in the screen region the march projects into.
    //      The march moves toward +x in screen space, so an occluder that activates for u > center
    //      catches the ray. Expect factor < 1. ----
    {
        // The marched ray's view-distance grows only slightly (toSun.z is small), so the ray sits ~d0.
        // An occluder at d0 - 0.2 is 0.2 closer than the ray: 0.2 > bias (0.02) and 0.2 <= thickness
        // (0.5) -> a valid contact occluder. Activate it once the march has moved right of center.
        auto occ = [&](float u, float) { return (u > 0.52f) ? (d0 - 0.2f) : d0; };
        float f = contact::RayMarchShadow(occ, viewPos, toSun, /*maxDist*/ 1.5f, /*steps*/ 24,
                                          kThickness, kBias, kTanHalfFovY, kAspect, kYFlip);
        check(f >= 0.0f && f <= 1.0f, "occluded factor in [0,1]");
        check(f < 1.0f, "occluder within the thickness window -> SHADOWED (factor < 1)");

        // CLOSER occluder (hit earlier in the march) -> factor <= the farther one (monotone). An
        // occluder that activates even sooner (smaller u threshold) is hit at a smaller t -> darker.
        auto occNear = [&](float u, float) { return (u > 0.505f) ? (d0 - 0.2f) : d0; };
        auto occFar  = [&](float u, float) { return (u > 0.55f)  ? (d0 - 0.2f) : d0; };
        float fNear = contact::RayMarchShadow(occNear, viewPos, toSun, 1.5f, 24,
                                              kThickness, kBias, kTanHalfFovY, kAspect, kYFlip);
        float fFar  = contact::RayMarchShadow(occFar, viewPos, toSun, 1.5f, 24,
                                              kThickness, kBias, kTanHalfFovY, kAspect, kYFlip);
        check(fNear < 1.0f && fFar < 1.0f, "both near + far occluders shadow");
        check(fNear <= fFar + 1e-6f, "a closer/earlier occluder darkens MORE (monotone)");
    }

    // ---- OCCLUDER BEYOND THICKNESS = NO FALSE SHADOW (the safety/haloing case). A surface in front of
    //      the ray but FARTHER than `thickness` behind the ray is a different object the ray passes
    //      safely in front of -> NOT a contact occluder -> factor 1. Place the occluder so diff >
    //      thickness everywhere it is hit. ----
    {
        // d0 - 1.0 is 1.0 closer than the ray: 1.0 > thickness (0.5) -> beyond the window -> no shadow.
        auto farOcc = [&](float u, float) { return (u > 0.52f) ? (d0 - 1.0f) : d0; };
        float f = contact::RayMarchShadow(farOcc, viewPos, toSun, /*maxDist*/ 1.5f, /*steps*/ 24,
                                          kThickness, kBias, kTanHalfFovY, kAspect, kYFlip);
        check(f == 1.0f, "occluder BEYOND the thickness window -> NO false shadow (factor 1)");
    }

    // ---- DETERMINISM: same inputs -> bit-identical factor (pure function, no RNG/time). ----
    {
        auto occ = [&](float u, float) { return (u > 0.52f) ? (d0 - 0.2f) : d0; };
        float a = contact::RayMarchShadow(occ, viewPos, toSun, 1.5f, 24,
                                          kThickness, kBias, kTanHalfFovY, kAspect, kYFlip, 0.37f);
        float b = contact::RayMarchShadow(occ, viewPos, toSun, 1.5f, 24,
                                          kThickness, kBias, kTanHalfFovY, kAspect, kYFlip, 0.37f);
        check(a == b, "RayMarchShadow is deterministic (bit-identical across runs)");
        // The dither must NOT break the maxDist=0 identity (every step still 0).
        auto wall = [&](float, float) { return d0 - 0.2f; };
        float z = contact::RayMarchShadow(wall, viewPos, toSun, 0.0f, 16,
                                          kThickness, kBias, kTanHalfFovY, kAspect, kYFlip, 0.83f);
        check(z == 1.0f, "maxDist=0 identity holds for ANY dither offset");
    }

    if (g_fail == 0) { std::printf("contact_shadows_test: all checks passed\n"); return 0; }
    std::printf("contact_shadows_test: %d FAILURES\n", g_fail);
    return 1;
}
