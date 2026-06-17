// Unit test for the CPU particle / VFX emitter system (engine/vfx/particles.{h,cpp}, Slice CC).
// Pure CPU (hf_core), ASan-eligible like the other pure tests — NO GPU, NO RHI.
//
// The emitter is an AUTHORABLE data-driven VFX layer (spawn rate, lifetime, init velocity + spread,
// gravity, drag, size/color over life) simulated at a FIXED timestep with a FIXED-SEED integer LCG
// for the spread jitter — so a fixed seed + step sequence + config is bit-identical run-to-run and
// platform-to-platform, which is what makes the --vfx-shot golden safe. (Distinct from the fixed
// `gpu-particles` compute fountain.)
//
// What this pins:
//   * Spawn rate  — over T seconds at rate R, ~R*T particles are spawned (within accumulator
//                   rounding); the live pool is capped at maxParticles.
//   * Lifetime    — a particle retires exactly when age >= lifetime; alive count is correct after
//                   stepping past lifetimes.
//   * Integration — gravity + drag over one/two steps match a hand-computed semi-implicit Euler step;
//                   no-gravity + no-drag == linear motion.
//   * Curves      — SizeAt/ColorAt lerp the endpoints (age 0 => start, age == lifetime => end) + the
//                   midpoint.
//   * Determinism — same seed + step sequence => bit-identical particle state across two runs.
//   * Billboards  — 6 verts/particle, camera-facing (quad spans cameraRight/cameraUp), centered on
//                   the particle pos, sized by SizeAt, colored by ColorAt.

#include "vfx/particles.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
using hf::math::Vec3;
using hf::math::Vec4;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

// A bare config: no gravity, no drag, no spread, infinite-ish lifetime — used to isolate one
// behavior at a time. Caller overrides the fields under test.
static vfx::EmitterConfig BareConfig() {
    vfx::EmitterConfig c;
    c.origin = {0, 0, 0};
    c.spawnRate = 0.0f;     // no spawning by default
    c.lifetime = 1000.0f;
    c.initVel = {0, 0, 0};
    c.velSpread = 0.0f;
    c.gravity = {0, 0, 0};
    c.drag = 0.0f;
    c.startSize = 1.0f;
    c.endSize = 0.0f;
    c.startColor = {1, 0, 0, 1};
    c.endColor = {0, 0, 1, 0};
    c.seed = 12345u;
    c.maxParticles = 100000;
    return c;
}

int main() {
    HF_TEST_MAIN_INIT();
    // ---- Spawn rate -----------------------------------------------------------------------------
    {
        vfx::EmitterConfig c = BareConfig();
        c.spawnRate = 100.0f;       // 100 particles/sec
        const float dt = 1.0f / 60.0f;
        const int steps = 120;      // 2 seconds
        vfx::ParticleSystem sys;
        for (int s = 0; s < steps; ++s) sys.Step(c, dt);
        // Expected spawned ~= R * T = 100 * 2 = 200 (accumulator carries the fractional remainder,
        // so the cumulative count is exactly floor(R*T) +/- 1).
        uint64_t spawned = sys.SpawnedCount();
        check(spawned >= 199 && spawned <= 201, "spawn rate ~= R*T over T seconds");
    }
    // Cap at maxParticles: a huge rate + long lifetime saturates the pool at the cap.
    {
        vfx::EmitterConfig c = BareConfig();
        c.spawnRate = 100000.0f;
        c.lifetime = 1000.0f;       // nothing retires
        c.maxParticles = 50;
        const float dt = 1.0f / 60.0f;
        vfx::ParticleSystem sys;
        for (int s = 0; s < 30; ++s) sys.Step(c, dt);
        check(sys.AliveCount() == 50, "alive pool capped at maxParticles");
        check((int)sys.Alive().size() == 50, "Alive() span size == cap");
    }

    // ---- Lifetime / retirement ------------------------------------------------------------------
    {
        // One spawn burst then no more spawns; step until they retire. Use a controlled spawn: rate
        // chosen so exactly 1 particle spawns on the first step, then rate 0.
        vfx::EmitterConfig c = BareConfig();
        const float dt = 0.1f;
        c.lifetime = 0.5f;          // retires at age >= 0.5
        // Spawn exactly one particle on step 0: spawnRate*dt == 1 => spawnRate = 10.
        c.spawnRate = 10.0f;
        vfx::ParticleSystem sys;
        sys.Step(c, dt);            // spawn 1; after this step age == dt == 0.1
        check(sys.AliveCount() == 1, "one particle alive after first step");
        c.spawnRate = 0.0f;         // stop spawning
        // ages after subsequent steps: 0.2, 0.3, 0.4, 0.5(retire)
        sys.Step(c, dt);            // age 0.2
        sys.Step(c, dt);            // age 0.3
        sys.Step(c, dt);            // age 0.4
        check(sys.AliveCount() == 1, "particle alive just below lifetime (age 0.4 < 0.5)");
        sys.Step(c, dt);            // age 0.5 => age >= lifetime => retire
        check(sys.AliveCount() == 0, "particle retires exactly when age >= lifetime");
    }

    // ---- Integration: gravity + drag (hand-checked) ---------------------------------------------
    {
        // Single particle, known initial velocity, known gravity + drag, semi-implicit Euler:
        //   vel += gravity*dt;  vel *= (1 - drag*dt);  pos += vel*dt;  age += dt.
        vfx::EmitterConfig c = BareConfig();
        const float dt = 0.5f;
        c.lifetime = 1000.0f;
        c.spawnRate = 2.0f;          // spawnRate*dt == 1 => exactly one spawn on step 0
        c.initVel = {1.0f, 2.0f, -1.0f};
        c.velSpread = 0.0f;          // no jitter => deterministic exact vel
        c.gravity = {0.0f, -10.0f, 0.0f};
        c.drag = 0.2f;
        c.origin = {0, 0, 0};
        vfx::ParticleSystem sys;
        sys.Step(c, dt);             // spawn + integrate one step
        c.spawnRate = 0.0f;
        check(sys.AliveCount() == 1, "integration: one particle");
        // Hand-compute step 1 (spawned this step, integrated once):
        //   vel0 = (1, 2, -1)
        //   vel += gravity*dt = (1, 2 - 10*0.5, -1) = (1, -3, -1)
        //   vel *= (1 - drag*dt) = (1 - 0.2*0.5) = 0.9 => (0.9, -2.7, -0.9)
        //   pos += vel*dt => (0.9*0.5, -2.7*0.5, -0.9*0.5) = (0.45, -1.35, -0.45)
        {
            const vfx::Particle& p = sys.Alive()[0];
            check(approx(p.vel.x, 0.9f) && approx(p.vel.y, -2.7f) && approx(p.vel.z, -0.9f),
                  "integration step 1 velocity (gravity+drag)");
            check(approx(p.pos.x, 0.45f) && approx(p.pos.y, -1.35f) && approx(p.pos.z, -0.45f),
                  "integration step 1 position");
            check(approx(p.age, 0.5f), "integration step 1 age == dt");
        }
        // Step 2 (no new spawns):
        //   vel = (0.9, -2.7, -0.9)
        //   vel += gravity*dt = (0.9, -2.7 - 5, -0.9) = (0.9, -7.7, -0.9)
        //   vel *= 0.9 => (0.81, -6.93, -0.81)
        //   pos += vel*dt => (0.45 + 0.405, -1.35 - 3.465, -0.45 - 0.405) = (0.855, -4.815, -0.855)
        sys.Step(c, dt);
        {
            const vfx::Particle& p = sys.Alive()[0];
            check(approx(p.vel.x, 0.81f) && approx(p.vel.y, -6.93f) && approx(p.vel.z, -0.81f),
                  "integration step 2 velocity");
            check(approx(p.pos.x, 0.855f) && approx(p.pos.y, -4.815f) && approx(p.pos.z, -0.855f),
                  "integration step 2 position");
            check(approx(p.age, 1.0f), "integration step 2 age");
        }
    }
    // No gravity + no drag == linear motion.
    {
        vfx::EmitterConfig c = BareConfig();
        const float dt = 0.25f;
        c.spawnRate = 4.0f;          // spawnRate*dt == 1 => one spawn on step 0
        c.initVel = {2.0f, 0.0f, 3.0f};
        c.velSpread = 0.0f;
        c.gravity = {0, 0, 0};
        c.drag = 0.0f;
        vfx::ParticleSystem sys;
        sys.Step(c, dt);
        c.spawnRate = 0.0f;
        for (int s = 0; s < 3; ++s) sys.Step(c, dt);  // 4 steps total
        const vfx::Particle& p = sys.Alive()[0];
        // Linear: pos == initVel * (4*dt) = initVel * 1.0; vel unchanged.
        check(approx(p.vel.x, 2.0f) && approx(p.vel.y, 0.0f) && approx(p.vel.z, 3.0f),
              "no-gravity/no-drag velocity unchanged");
        check(approx(p.pos.x, 2.0f) && approx(p.pos.z, 3.0f) && approx(p.pos.y, 0.0f),
              "no-gravity/no-drag linear position");
    }

    // ---- Lifetime curves: SizeAt / ColorAt ------------------------------------------------------
    {
        vfx::EmitterConfig c = BareConfig();
        c.startSize = 2.0f; c.endSize = 0.5f;
        c.startColor = {1.0f, 0.0f, 0.0f, 1.0f};
        c.endColor   = {0.0f, 0.0f, 1.0f, 0.0f};
        c.lifetime = 4.0f;
        vfx::Particle p; p.lifetime = c.lifetime;
        // age 0 => start
        p.age = 0.0f;
        check(approx(vfx::SizeAt(c, p), 2.0f), "SizeAt age 0 == startSize");
        {
            Vec4 col = vfx::ColorAt(c, p);
            check(approx(col.x, 1) && approx(col.y, 0) && approx(col.z, 0) && approx(col.w, 1),
                  "ColorAt age 0 == startColor");
        }
        // age == lifetime => end
        p.age = c.lifetime;
        check(approx(vfx::SizeAt(c, p), 0.5f), "SizeAt age==lifetime == endSize");
        {
            Vec4 col = vfx::ColorAt(c, p);
            check(approx(col.x, 0) && approx(col.y, 0) && approx(col.z, 1) && approx(col.w, 0),
                  "ColorAt age==lifetime == endColor");
        }
        // midpoint
        p.age = c.lifetime * 0.5f;
        check(approx(vfx::SizeAt(c, p), 1.25f), "SizeAt midpoint == lerp midpoint");
        {
            Vec4 col = vfx::ColorAt(c, p);
            check(approx(col.x, 0.5f) && approx(col.y, 0) && approx(col.z, 0.5f) && approx(col.w, 0.5f),
                  "ColorAt midpoint == lerp midpoint");
        }
    }

    // ---- Determinism: two runs bit-identical ----------------------------------------------------
    {
        vfx::EmitterConfig c = BareConfig();
        c.spawnRate = 200.0f;
        c.lifetime = 1.5f;
        c.initVel = {0.0f, 5.0f, 0.0f};
        c.velSpread = 2.0f;          // exercise the seeded LCG jitter
        c.gravity = {0.0f, -9.8f, 0.0f};
        c.drag = 0.15f;
        c.seed = 0xC0FFEEu;
        const float dt = 1.0f / 120.0f;
        const int steps = 180;

        auto run = [&]() {
            vfx::ParticleSystem sys;
            for (int s = 0; s < steps; ++s) sys.Step(c, dt);
            std::vector<vfx::Particle> out(sys.Alive().begin(), sys.Alive().end());
            return out;
        };
        std::vector<vfx::Particle> a = run();
        std::vector<vfx::Particle> b = run();
        bool same = (a.size() == b.size()) &&
                    (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(vfx::Particle)) == 0);
        check(!a.empty(), "determinism run produced particles");
        check(same, "two runs are bit-identical (same seed + steps)");
    }

    // ---- Billboard generation -------------------------------------------------------------------
    {
        vfx::EmitterConfig c = BareConfig();
        c.startSize = 0.5f; c.endSize = 0.5f;   // constant size for an exact check
        c.startColor = {0.2f, 0.4f, 0.6f, 0.8f};
        c.endColor   = {0.2f, 0.4f, 0.6f, 0.8f};
        c.lifetime = 10.0f;
        vfx::Particle p;
        p.pos = {1.0f, 2.0f, 3.0f};
        p.age = 0.0f; p.lifetime = c.lifetime;
        const Vec3 right{1.0f, 0.0f, 0.0f};
        const Vec3 up{0.0f, 1.0f, 0.0f};
        std::vector<vfx::BillboardVertex> verts;
        std::vector<vfx::Particle> one{p};
        vfx::BuildBillboards(one, c, right, up, verts);
        check(verts.size() == 6, "6 verts per particle (2 triangles)");
        // The four expected corners (half = SizeAt = 0.5): pos +/- right*half +/- up*half.
        const float h = 0.5f;
        // Compute centroid: should be the particle pos.
        Vec3 centroid{0, 0, 0};
        for (auto& v : verts) centroid = centroid + v.pos;
        centroid = centroid * (1.0f / 6.0f);
        check(approx(centroid.x, 1.0f) && approx(centroid.y, 2.0f) && approx(centroid.z, 3.0f),
              "billboard centered on particle pos");
        // Every vertex lies in the plane spanned by right/up offset from pos by +/- half on each axis,
        // i.e. its offset from pos has |dot(right)| == h and |dot(up)| == h, and zero out-of-plane.
        bool spanOk = true, planeOk = true;
        for (auto& v : verts) {
            Vec3 d = v.pos - p.pos;
            float dr = hf::math::dot(d, right);
            float du = hf::math::dot(d, up);
            if (!(approx(std::fabs(dr), h) && approx(std::fabs(du), h))) spanOk = false;
            // out-of-plane component (along right x up = +Z here) must be 0.
            Vec3 normal = hf::math::cross(right, up);
            if (!approx(hf::math::dot(d, normal), 0.0f)) planeOk = false;
        }
        check(spanOk, "billboard quad spans +/- SizeAt along cameraRight and cameraUp");
        check(planeOk, "billboard quad is camera-facing (lies in the right/up plane)");
        // Color matches ColorAt (constant here).
        bool colorOk = true;
        for (auto& v : verts) {
            if (!(approx(v.color.x, 0.2f) && approx(v.color.y, 0.4f) &&
                  approx(v.color.z, 0.6f) && approx(v.color.w, 0.8f))) colorOk = false;
        }
        check(colorOk, "billboard vertex color == ColorAt");
        // UVs cover the [0,1]^2 corners (each of the 4 corners appears at least once).
        bool saw00 = false, saw10 = false, saw11 = false, saw01 = false;
        for (auto& v : verts) {
            if (approx(v.uv.x, 0) && approx(v.uv.y, 0)) saw00 = true;
            if (approx(v.uv.x, 1) && approx(v.uv.y, 0)) saw10 = true;
            if (approx(v.uv.x, 1) && approx(v.uv.y, 1)) saw11 = true;
            if (approx(v.uv.x, 0) && approx(v.uv.y, 1)) saw01 = true;
        }
        check(saw00 && saw10 && saw11 && saw01, "billboard UVs cover the 4 sprite corners");
        // Size scales the quad: a bigger particle => bigger half-extent.
        vfx::EmitterConfig big = c; big.startSize = 2.0f; big.endSize = 2.0f;
        std::vector<vfx::BillboardVertex> bv;
        vfx::BuildBillboards(one, big, right, up, bv);
        float maxOff = 0.0f;
        for (auto& v : bv) maxOff = std::fmax(maxOff, std::fabs(hf::math::dot(v.pos - p.pos, right)));
        check(approx(maxOff, 2.0f), "billboard sized by SizeAt");
    }

    if (g_fail == 0) std::printf("vfx_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
