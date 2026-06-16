// Slice CF — water rendering math. Pure CPU: Gerstner displacement, ANALYTIC surface normal
// (cross-checked against a finite difference of Displace), Schlick fresnel, depth-based refract tint,
// and determinism. No device, ASan-eligible (links hf_core). Mirrors the math the --water-shot
// showcase and water.{vert,frag} use (engine/render/water.h).
#include "render/water.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>

using namespace hf::math;
namespace water = hf::render::water;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-3f) { return std::fabs(a - b) < eps; }

int main() {
    // ---- Single-wave Gerstner displacement at t=0: known closed form. ----
    {
        // A +X-traveling wave at the origin (x=z=0, t=0): theta = 0 -> cos=1, sin=0.
        water::GerstnerWave wv{Vec2{1.0f, 0.0f}, 0.5f, 4.0f, 0.6f, 1.0f};
        Vec3 d = water::DisplaceWave(0.0f, 0.0f, 0.0f, wv);
        // horizontal x = Q*A*Dx*cos(0) = 0.6*0.5*1*1 = 0.30; vertical = A*sin(0) = 0; z = 0.
        check(approx(d.x, 0.30f), "single wave x displacement at theta=0 == Q*A*Dx");
        check(approx(d.y, 0.0f),  "single wave vertical displacement at theta=0 == 0");
        check(approx(d.z, 0.0f),  "single wave z displacement (Dz=0) == 0");

        // A quarter wavelength along +X (x = L/4 = 1.0) gives theta = k*x = (2pi/4)*1 = pi/2:
        // cos=0 -> x horizontal = 0; sin=1 -> vertical = A = 0.5.
        Vec3 d2 = water::DisplaceWave(1.0f, 0.0f, 0.0f, wv);
        check(approx(d2.x, 0.0f), "quarter-wavelength horizontal x == 0 (cos(pi/2)=0)");
        check(approx(d2.y, 0.5f), "quarter-wavelength vertical == amplitude (sin(pi/2)=1)");
    }

    // ---- Sum of N waves == sum of the individual displacements. ----
    {
        water::WaveSet ws = water::ShowcaseWaves();
        const float x = 1.7f, z = -0.6f, t = water::kFixedTime;
        Vec3 summed = water::Displace(x, z, t, ws.waves, ws.count);
        Vec3 byHand{0, 0, 0};
        for (std::size_t i = 0; i < ws.count; ++i)
            byHand = byHand + water::DisplaceWave(x, z, t, ws.waves[i]);
        check(approx(summed.x, byHand.x) && approx(summed.y, byHand.y) && approx(summed.z, byHand.z),
              "Displace over the set == sum of per-wave DisplaceWave");
    }

    // ---- Flat (zero-amplitude) water: analytic normal is exactly +Y. ----
    {
        water::GerstnerWave flat[2] = {
            water::GerstnerWave{Vec2{1.0f, 0.0f}, 0.0f, 4.0f, 0.6f, 1.0f},
            water::GerstnerWave{Vec2{0.0f, 1.0f}, 0.0f, 2.0f, 0.6f, 1.0f},
        };
        Vec3 n = water::Normal(2.3f, -1.1f, 0.7f, flat, 2);
        check(approx(n.x, 0.0f) && approx(n.y, 1.0f) && approx(n.z, 0.0f),
              "zero-amplitude water normal == +Y");
    }

    // ---- Analytic normal is unit length on a real wave set. ----
    {
        water::WaveSet ws = water::ShowcaseWaves();
        for (float x = -3.0f; x <= 3.0f; x += 1.3f)
            for (float z = -3.0f; z <= 3.0f; z += 1.1f) {
                Vec3 n = water::Normal(x, z, water::kFixedTime, ws.waves, ws.count);
                check(approx(length(n), 1.0f), "analytic normal is unit length");
            }
    }

    // ---- A known single wave tilts the normal the expected direction. ----
    {
        // +X wave. On the up-slope (just before the crest) the surface rises toward +X, so the normal
        // tilts toward -X (and stays +Y dominant). Sample at theta = -pi/2 region.
        water::GerstnerWave wv{Vec2{1.0f, 0.0f}, 0.3f, 4.0f, 0.5f, 1.0f};
        // theta = k*x, k = 2pi/4 = pi/2. x = 1 -> theta = pi/2 (crest). Just before crest x=0.5 ->
        // theta = pi/4: rising slope. dHeight/dx = A*k*cos > 0 -> normal.x < 0.
        Vec3 n = water::Normal(0.5f, 0.0f, 0.0f, &wv, 1);
        check(n.y > 0.0f, "single-wave normal stays +Y dominant");
        check(n.x < 0.0f, "single-wave normal tilts away from the rising +X slope");
        check(approx(n.z, 0.0f), "single +X wave does not tilt the normal in z");
    }

    // ---- ANALYTIC normal == FINITE-DIFFERENCE of Displace (proves the derivative is right). ----
    {
        water::WaveSet ws = water::ShowcaseWaves();
        const float t = water::kFixedTime;
        const float h = 1e-3f;
        for (float x = -2.5f; x <= 2.5f; x += 1.7f)
            for (float z = -2.5f; z <= 2.5f; z += 1.3f) {
                // Finite-difference tangent/bitangent of the full surface point P = (x,0,z) + Displace.
                Vec3 dpx = water::Displace(x + h, z, t, ws.waves, ws.count)
                         - water::Displace(x - h, z, t, ws.waves, ws.count);
                Vec3 dpz = water::Displace(x, z + h, t, ws.waves, ws.count)
                         - water::Displace(x, z - h, t, ws.waves, ws.count);
                // P partials: add the base grid's (1,0,0)/(0,0,1); central difference / (2h).
                Vec3 T{1.0f + dpx.x / (2 * h), dpx.y / (2 * h), dpx.z / (2 * h)};
                Vec3 B{dpz.x / (2 * h), dpz.y / (2 * h), 1.0f + dpz.z / (2 * h)};
                Vec3 nFd = normalize(cross(B, T));
                Vec3 nAn = water::Normal(x, z, t, ws.waves, ws.count);
                check(approx(nAn.x, nFd.x, 5e-3f) && approx(nAn.y, nFd.y, 5e-3f) &&
                      approx(nAn.z, nFd.z, 5e-3f),
                      "analytic normal matches finite-difference of Displace");
            }
    }

    // ---- Fresnel: head-on == f0, grazing == 1, monotone in the view angle. ----
    {
        const float f0 = 0.02f;
        check(approx(water::Fresnel(1.0f, f0), f0), "Fresnel(NdotV=1) == f0 (head-on)");
        check(approx(water::Fresnel(0.0f, f0), 1.0f), "Fresnel(NdotV=0) == 1 (grazing)");
        // Monotonic: as NdotV decreases (more grazing) reflectance increases.
        float prev = water::Fresnel(1.0f, f0);
        for (float c = 0.9f; c >= 0.0f; c -= 0.1f) {
            float v = water::Fresnel(c, f0);
            check(v >= prev - 1e-6f, "Fresnel increases monotonically toward grazing");
            prev = v;
        }
    }

    // ---- RefractTint: depth 0 -> shallow, large depth -> ~deep, monotone. ----
    {
        Vec3 shallow{0.30f, 0.55f, 0.55f};
        Vec3 deep{0.02f, 0.08f, 0.12f};
        const float absorption = 0.5f;
        Vec3 d0 = water::RefractTint(0.0f, shallow, deep, absorption);
        check(approx(d0.x, shallow.x) && approx(d0.y, shallow.y) && approx(d0.z, shallow.z),
              "RefractTint(depth=0) == shallow color");
        Vec3 dBig = water::RefractTint(50.0f, shallow, deep, absorption);
        check(approx(dBig.x, deep.x, 2e-3f) && approx(dBig.y, deep.y, 2e-3f) &&
              approx(dBig.z, deep.z, 2e-3f), "RefractTint(large depth) ~ deep color");
        // Monotone: the green channel goes shallow.y -> deep.y (decreasing) with depth.
        float prev = water::RefractTint(0.0f, shallow, deep, absorption).y;
        for (float dp = 0.5f; dp <= 10.0f; dp += 0.5f) {
            float v = water::RefractTint(dp, shallow, deep, absorption).y;
            check(v <= prev + 1e-6f, "RefractTint absorbs monotonically with depth");
            prev = v;
        }
    }

    // ---- Determinism: same (x,z,t) -> identical Displace + Normal across calls. ----
    {
        water::WaveSet ws = water::ShowcaseWaves();
        const float x = 0.9f, z = 2.1f, t = water::kFixedTime;
        Vec3 d1 = water::Displace(x, z, t, ws.waves, ws.count);
        Vec3 d2 = water::Displace(x, z, t, ws.waves, ws.count);
        check(d1.x == d2.x && d1.y == d2.y && d1.z == d2.z, "Displace is bit-deterministic");
        Vec3 n1 = water::Normal(x, z, t, ws.waves, ws.count);
        Vec3 n2 = water::Normal(x, z, t, ws.waves, ws.count);
        check(n1.x == n2.x && n1.y == n2.y && n1.z == n2.z, "Normal is bit-deterministic");
    }

    if (g_fail == 0) std::printf("water_test: all checks passed\n");
    else std::printf("water_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
