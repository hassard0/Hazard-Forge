// Slice S unit test: the impulse-based rigid-body world. Pure C++ (no GPU): validates gravity +
// rest, sphere-sphere separation without explosion, static-body immovability, and determinism.
#include "physics/world.h"
#include "physics/body.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();
    const float dt = 1.0f / 120.0f;

    // --- 1. A sphere dropped from height falls under gravity, rests at y ~= radius, and STAYS. ---
    {
        physics::World w;
        physics::RigidBody s = physics::MakeDynamicSphere({0.0f, 3.0f, 0.0f}, 0.5f);
        w.bodies.push_back(s);
        for (int i = 0; i < 600; ++i) w.Step(dt);   // 5 s — plenty to settle
        float y = w.bodies[0].position.y;
        check(std::fabs(y - 0.5f) < 0.02f, "dropped sphere rests at y ~= radius");
        float v = math::length(w.bodies[0].linVel);
        check(v < 0.05f, "dropped sphere is at rest (|linVel| ~ 0)");
        // And it does NOT sink through the floor.
        check(y > 0.45f, "dropped sphere did not sink through the plane");
    }

    // --- 2. Two overlapping spheres separate and do NOT explode (kinetic energy bounded). ---
    {
        physics::World w;
        w.gravity = {0, 0, 0};   // isolate the contact response from gravity
        // Two unit-radius spheres overlapping by 0.4 along X.
        physics::RigidBody a = physics::MakeDynamicSphere({-0.3f, 0.0f, 0.0f}, 0.5f);
        physics::RigidBody b = physics::MakeDynamicSphere({ 0.3f, 0.0f, 0.0f}, 0.5f);
        w.bodies.push_back(a);
        w.bodies.push_back(b);
        for (int i = 0; i < 240; ++i) w.Step(dt);
        float dist = math::length(w.bodies[1].position - w.bodies[0].position);
        check(dist > 1.0f - 0.02f, "overlapping spheres separated to ~ rA+rB");
        // No explosion: kinetic energy stays small/bounded (no energy injected).
        check(w.KineticEnergy() < 1.0f, "no explosion (kinetic energy bounded)");
        // Symmetry-ish: they pushed apart roughly equally (centers stay centered near origin).
        float mid = 0.5f * (w.bodies[0].position.x + w.bodies[1].position.x);
        check(std::fabs(mid) < 0.05f, "separation is symmetric about the origin");
    }

    // --- 3. A static body (invMass 0) never moves, even when a dynamic sphere lands on it. ---
    {
        physics::World w;
        physics::RigidBody stat = physics::MakeStaticSphere({0.0f, 0.5f, 0.0f}, 0.5f);
        physics::RigidBody dyn  = physics::MakeDynamicSphere({0.0f, 2.0f, 0.0f}, 0.5f);
        w.bodies.push_back(stat);
        w.bodies.push_back(dyn);
        math::Vec3 p0 = w.bodies[0].position;
        for (int i = 0; i < 400; ++i) w.Step(dt);
        check(w.bodies[0].position.x == p0.x &&
              w.bodies[0].position.y == p0.y &&
              w.bodies[0].position.z == p0.z, "static body never moves");
        // The dynamic sphere should rest on top of the static one (y ~= 1.0), not pass through it.
        check(w.bodies[1].position.y > 0.95f, "dynamic sphere rests on the static body");
    }

    // --- 4. Determinism: two identical worlds stepped identically are BIT-identical. ---
    {
        auto build = [] {
            physics::World w;
            for (int i = 0; i < 5; ++i)
                w.bodies.push_back(physics::MakeDynamicSphere(
                    {(float)i * 0.6f - 1.2f, 1.5f + 0.1f * i, 0.0f}, 0.4f));
            return w;
        };
        physics::World w1 = build();
        physics::World w2 = build();
        for (int i = 0; i < 300; ++i) { w1.Step(dt); w2.Step(dt); }
        bool identical = true;
        for (size_t i = 0; i < w1.bodies.size(); ++i) {
            const auto& A = w1.bodies[i]; const auto& B = w2.bodies[i];
            if (A.position.x != B.position.x || A.position.y != B.position.y ||
                A.position.z != B.position.z ||
                A.linVel.x != B.linVel.x || A.linVel.y != B.linVel.y || A.linVel.z != B.linVel.z ||
                A.orientation.x != B.orientation.x || A.orientation.w != B.orientation.w)
                identical = false;
        }
        check(identical, "simulation is bit-deterministic across runs");
    }

    if (g_fail == 0) std::printf("physics_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
