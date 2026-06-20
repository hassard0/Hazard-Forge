// Slice BD1 — Deterministic GPU Crowds: the INTEGER STEERING PRIMITIVE integer core (engine/sim/boids.h) the
// GPU shaders/boids_steer.comp.hlsl copies VERBATIM + proves bit-identical. Pure CPU (header-only, no device,
// no backend symbols). Namespace hf::sim::boids.
//
// What this test PINS (the contracts the GPU boids_steer.comp + the GPU==CPU proof build on):
//   * SteerSeek: the force points target-ward (each component has the SAME SIGN as target - pos), is zero at
//     the target, and scales with seekGain.
//   * SteerSeparation: a too-close neighbor pushes the agent AWAY (the force points away from the neighbor);
//     a neighbor BEYOND sepRadius contributes nothing (zero force); two equidistant neighbors on opposite
//     sides cancel.
//   * StepBoids: a tight cluster seeking a target SPREADS + ADVANCES (meanToTarget drops, minSep stays above
//     a floor); a sepGain=0 control COLLAPSES (minSep -> ~0); the per-axis clamp caps force/speed; two runs
//     byte-identical.
//
// HONEST CAVEAT (the GR4-shape): boids are POINTS with steering FORCES (a soft separation push, NOT a hard
// non-penetration contact); the per-axis clamp is an axis-BOX, not a radial magnitude clamp. The headline is
// DETERMINISM + cross-platform bit-identity.
//
// Pure C++ (hf_core), ASan-eligible like the other sim-math tests.
#include "sim/boids.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace boids = hf::sim::boids;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static boids::fx fxabs(boids::fx v) { return v < 0 ? -v : v; }

// A Q16.16 scalar from an integer world unit.
static boids::fx wu(int u) { return (boids::fx)(u * (int)boids::kOne); }
// A Q16.16 scalar from a fraction n/d of a world unit.
static boids::fx frac(int n, int d) { return (boids::fx)((int64_t)n * (int64_t)boids::kOne / d); }

// An agent at integer world (x,y,z) with zero velocity.
static boids::Agent at(int x, int y, int z) {
    return boids::Agent{boids::FxVec3{wu(x), wu(y), wu(z)}, boids::FxVec3{0, 0, 0}};
}

// The shared steering tuning the StepBoids cases use (matches the showcase config).
static boids::BoidsConfig flockCfg(boids::fx sepGain) {
    boids::BoidsConfig c;
    c.seekGain  = frac(1, 4);    // 0.25 proportional seek
    c.sepGain   = sepGain;       // 0.5 (on) or 0 (control)
    c.sepRadius = wu(2);         // 2-world-unit neighbor radius
    c.maxForce  = wu(8);
    c.maxSpeed  = wu(6);
    c.target    = boids::FxVec3{wu(40), 0, 0};   // a target offset to one side (+x)
    c.gravity   = boids::FxVec3{0, 0, 0};
    return c;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ===== SteerSeek =====
    {
        boids::BoidsConfig c;
        c.seekGain = frac(1, 2);   // 0.5
        c.target   = boids::FxVec3{wu(10), wu(5), wu(-3)};
        boids::Agent a = at(0, 0, 0);
        boids::FxVec3 f = boids::SteerSeek(a, c.target, c);
        // force points target-ward: same sign as (target - pos) per axis, nonzero where target != pos.
        check(f.x > 0, "SteerSeek: +x target -> +x force");
        check(f.y > 0, "SteerSeek: +y target -> +y force");
        check(f.z < 0, "SteerSeek: -z target -> -z force");
        // scales with seekGain: 0.5 * (10,5,-3) world units.
        check(f.x == boids::fxmul(wu(10), c.seekGain), "SteerSeek: force.x == desired.x * seekGain");
        check(f.y == boids::fxmul(wu(5),  c.seekGain), "SteerSeek: force.y == desired.y * seekGain");
        // at the target -> zero force.
        boids::Agent atTarget{c.target, boids::FxVec3{0, 0, 0}};
        boids::FxVec3 f0 = boids::SteerSeek(atTarget, c.target, c);
        check(f0.x == 0 && f0.y == 0 && f0.z == 0, "SteerSeek: zero at the target");
        // larger gain -> larger force.
        boids::BoidsConfig c2 = c; c2.seekGain = boids::kOne;  // 1.0
        boids::FxVec3 f2 = boids::SteerSeek(a, c.target, c2);
        check(fxabs(f2.x) > fxabs(f.x), "SteerSeek: larger gain -> larger force");
    }

    // ===== SteerSeparation =====
    {
        boids::BoidsConfig c;
        c.sepGain   = boids::kOne;   // 1.0 so the sign survives
        c.sepRadius = wu(3);
        // a is at origin; a single close neighbor to the +x side -> push toward -x (away).
        boids::Agent a = at(0, 0, 0);
        std::vector<boids::Agent> others = {a, at(1, 0, 0)};   // index 0 = self, index 1 = +x neighbor
        boids::FxVec3 f = boids::SteerSeparation(a, std::span<const boids::Agent>(others), c, 0);
        check(f.x < 0, "SteerSeparation: a close +x neighbor pushes -x (away)");
        check(f.y == 0 && f.z == 0, "SteerSeparation: pure +x neighbor -> only x push");
        // a neighbor BEYOND sepRadius contributes nothing.
        std::vector<boids::Agent> far = {a, at(10, 0, 0)};     // 10 > sepRadius 3
        boids::FxVec3 ff = boids::SteerSeparation(a, std::span<const boids::Agent>(far), c, 0);
        check(ff.x == 0 && ff.y == 0 && ff.z == 0, "SteerSeparation: beyond sepRadius -> zero");
        // two equidistant neighbors on opposite sides cancel.
        std::vector<boids::Agent> both = {a, at(1, 0, 0), at(-1, 0, 0)};
        boids::FxVec3 fb = boids::SteerSeparation(a, std::span<const boids::Agent>(both), c, 0);
        check(fb.x == 0 && fb.y == 0 && fb.z == 0, "SteerSeparation: symmetric neighbors cancel");
        // exactly on the radius is NOT a neighbor (strict d² < r²): a neighbor at distance == sepRadius.
        std::vector<boids::Agent> onEdge = {a, at(3, 0, 0)};   // distance 3 == sepRadius 3
        boids::FxVec3 fe = boids::SteerSeparation(a, std::span<const boids::Agent>(onEdge), c, 0);
        check(fe.x == 0 && fe.y == 0 && fe.z == 0, "SteerSeparation: at exactly sepRadius -> excluded (d²<r²)");
    }

    // ===== StepBoids: a cluster seeks the target -> spreads + advances =====
    {
        boids::BoidsConfig c = flockCfg(frac(1, 2));   // sepGain 0.5 (separation ON)
        const boids::fx dt = boids::kOne / 60;
        // a tight 4x4 cluster near the origin (1-unit spacing < sepRadius 2 so they push apart).
        std::vector<boids::Agent> agents;
        for (int gx = 0; gx < 4; ++gx)
            for (int gz = 0; gz < 4; ++gz)
                agents.push_back(at(gx, 0, gz));

        boids::BoidsStats before = boids::MeasureBoids(agents, c);
        boids::StepBoidsSteps(agents, c, dt, 240);
        boids::BoidsStats after = boids::MeasureBoids(agents, c);

        // they sought the target: mean distance to target DROPPED.
        check(after.meanToTarget < before.meanToTarget, "StepBoids: meanToTarget drops (sought the target)");
        // they didn't collapse: min separation stays above a floor (separation kept them apart).
        check(after.minSep > frac(1, 4), "StepBoids: minSep above a floor (didn't collapse)");
        // they moved (nonzero mean speed).
        check(after.meanSpeed > 0, "StepBoids: nonzero mean speed (the flock is moving)");
    }

    // ===== StepBoids control: sepGain=0 collapses =====
    {
        boids::BoidsConfig on  = flockCfg(frac(1, 2));
        boids::BoidsConfig off = flockCfg(0);          // separation OFF
        const boids::fx dt = boids::kOne / 60;
        std::vector<boids::Agent> agOn, agOff;
        for (int gx = 0; gx < 4; ++gx)
            for (int gz = 0; gz < 4; ++gz) { agOn.push_back(at(gx, 0, gz)); agOff.push_back(at(gx, 0, gz)); }
        boids::StepBoidsSteps(agOn,  on,  dt, 600);
        boids::StepBoidsSteps(agOff, off, dt, 600);
        boids::BoidsStats sOn  = boids::MeasureBoids(agOn,  on);
        boids::BoidsStats sOff = boids::MeasureBoids(agOff, off);
        // the no-separation flock collapses much tighter than the separating one.
        check(sOff.minSep < sOn.minSep, "StepBoids control: sepGain=0 collapses tighter than sepGain>0");
        check(sOff.minSep <= frac(1, 8), "StepBoids control: sepGain=0 minSep -> ~0 (collapsed)");
    }

    // ===== per-axis clamp caps force + speed =====
    {
        // ClampAxis directly.
        check(boids::ClampAxis(wu(100), wu(8)) == wu(8),  "ClampAxis: positive over-limit clamped to +limit");
        check(boids::ClampAxis(wu(-100), wu(8)) == wu(-8),"ClampAxis: negative over-limit clamped to -limit");
        check(boids::ClampAxis(wu(3), wu(8)) == wu(3),    "ClampAxis: within limit unchanged");
        // a huge seek force + huge dt must NOT exceed maxSpeed per axis after one step.
        boids::BoidsConfig c;
        c.seekGain  = boids::kOne * 100;   // absurd gain -> huge raw force
        c.sepGain   = 0;
        c.sepRadius = 0;
        c.maxForce  = wu(8);
        c.maxSpeed  = wu(6);
        c.target    = boids::FxVec3{wu(1000), 0, 0};
        std::vector<boids::Agent> a = {at(0, 0, 0)};
        boids::StepBoids(a, c, boids::kOne / 60);
        check(fxabs(a[0].vel.x) <= c.maxSpeed, "StepBoids: vel.x capped to maxSpeed despite huge force");
        check(fxabs(a[0].vel.y) <= c.maxSpeed, "StepBoids: vel.y capped to maxSpeed");
    }

    // ===== two runs byte-identical (determinism) =====
    {
        boids::BoidsConfig c = flockCfg(frac(1, 2));
        const boids::fx dt = boids::kOne / 60;
        std::vector<boids::Agent> a1, a2;
        for (int gx = 0; gx < 6; ++gx)
            for (int gz = 0; gz < 6; ++gz) { a1.push_back(at(gx, 0, gz)); a2.push_back(at(gx, 0, gz)); }
        boids::StepBoidsSteps(a1, c, dt, 300);
        boids::StepBoidsSteps(a2, c, dt, 300);
        check(a1.size() == a2.size() &&
              std::memcmp(a1.data(), a2.data(), a1.size() * sizeof(boids::Agent)) == 0,
              "StepBoids: two runs BYTE-IDENTICAL");
        static_assert(sizeof(boids::Agent) == 24, "Agent std430 layout (6 x int32, 24 bytes)");
    }

    if (g_fail == 0) std::printf("boids_test: ALL PASS\n");
    else             std::printf("boids_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
