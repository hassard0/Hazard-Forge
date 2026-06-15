// Slice AH — screen-space reflections. Pure CPU math: view<->screen projection round-trip + the
// view-space reflection ray. No device, ASan-eligible (links hf_core). Mirrors the math the
// --ssr-shot showcase and ssr.frag use (engine/render/ssr.h).
#include "render/ssr.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>

using namespace hf::math;
namespace ssr = hf::render::ssr;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-3f) { return std::fabs(a - b) < eps; }

int main() {
    const float fovY = 1.04719755f;          // 60 deg, matches the showcase
    const float tanHalf = std::tan(0.5f * fovY);
    const float aspect = 1280.0f / 720.0f;

    // Use the Vulkan sign convention (yFlip = -1) for the round-trip checks.
    const float yFlip = -1.0f;

    // ---- A view-space point on the -Z axis (straight ahead) projects to screen center. ----
    {
        Vec3 p{0.0f, 0.0f, -5.0f};           // 5 units in front of the camera, centered
        Vec3 uvd = ssr::ViewToScreenUV(p, tanHalf, aspect, yFlip);
        check(approx(uvd.x, 0.5f), "centered point projects to uv.x ~ 0.5");
        check(approx(uvd.y, 0.5f), "centered point projects to uv.y ~ 0.5");
        check(approx(uvd.z, 5.0f), "linear depth == -vp.z == 5");
    }

    // ---- ViewToScreenUV and ReconstructViewPos are mutual inverses. ----
    {
        Vec3 p{1.3f, -0.7f, -6.0f};
        Vec3 uvd = ssr::ViewToScreenUV(p, tanHalf, aspect, yFlip);
        Vec3 r = ssr::ReconstructViewPos(uvd.x, uvd.y, uvd.z, tanHalf, aspect, yFlip);
        check(approx(r.x, p.x), "round-trip recovers view x");
        check(approx(r.y, p.y), "round-trip recovers view y");
        check(approx(r.z, p.z), "round-trip recovers view z");
    }

    // ---- A point to the RIGHT of center (+x in view space) lands at uv.x > 0.5. ----
    {
        Vec3 right{2.0f, 0.0f, -5.0f};
        Vec3 uvd = ssr::ViewToScreenUV(right, tanHalf, aspect, yFlip);
        check(uvd.x > 0.5f, "+x view point lands right of center");
        check(approx(uvd.y, 0.5f), "+x view point stays vertically centered");
    }

    // ---- Reflection: a ray traveling straight DOWN reflecting off a horizontal floor goes UP. ----
    // Floor at y=0 has world-up normal; expressed in a view that looks level, view normal ~ +Y.
    {
        Vec3 incident{0.0f, -1.0f, 0.0f};    // heading down
        Vec3 floorN{0.0f, 1.0f, 0.0f};       // up-facing
        Vec3 r = ssr::ReflectView(incident, floorN);
        check(approx(r.x, 0.0f), "reflect: x unchanged");
        check(approx(r.y, 1.0f), "reflect: down ray off floor goes up");
        check(approx(r.z, 0.0f), "reflect: z unchanged");
    }

    // ---- Reflection: a 45-deg ray off a horizontal floor mirrors the vertical component. ----
    {
        Vec3 incident = normalize(Vec3{1.0f, -1.0f, 0.0f});
        Vec3 floorN{0.0f, 1.0f, 0.0f};
        Vec3 r = ssr::ReflectView(incident, floorN);
        check(approx(r.x, incident.x), "reflect 45deg: horizontal component preserved");
        check(approx(r.y, -incident.y), "reflect 45deg: vertical component flipped");
        check(approx(length(r), 1.0f), "reflect preserves length");
    }

    // ---- Reflection about an arbitrary normal still satisfies reflect identity (R . N == -I . N). ----
    {
        Vec3 incident = normalize(Vec3{0.3f, -0.8f, -0.5f});
        Vec3 n = normalize(Vec3{0.1f, 0.95f, 0.2f});
        Vec3 r = ssr::ReflectView(incident, n);
        check(approx(dot(r, n), -dot(incident, n)),
              "reflect identity: dot(R,N) == -dot(I,N)");
        check(approx(length(r), length(incident)), "reflect preserves magnitude");
    }

    // ---- Metal sign convention (yFlip = +1) also round-trips. ----
    {
        const float yf = 1.0f;
        Vec3 p{-0.9f, 1.1f, -4.0f};
        Vec3 uvd = ssr::ViewToScreenUV(p, tanHalf, aspect, yf);
        Vec3 r = ssr::ReconstructViewPos(uvd.x, uvd.y, uvd.z, tanHalf, aspect, yf);
        check(approx(r.x, p.x) && approx(r.y, p.y) && approx(r.z, p.z),
              "Metal yFlip=+1 round-trips");
        // The two backends place +Y on opposite UV halves.
        Vec3 uvVk = ssr::ViewToScreenUV(p, tanHalf, aspect, -1.0f);
        check((uvd.y - 0.5f) * (uvVk.y - 0.5f) < 0.0f,
              "Vulkan/Metal place +Y on opposite UV halves");
    }

    if (g_fail == 0) std::printf("ssr_test: all checks passed\n");
    else std::printf("ssr_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
