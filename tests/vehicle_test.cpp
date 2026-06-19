// Slice VH1 — Deterministic Vehicle Physics: the SUSPENSION SPRING JOINT integer core (engine/sim/vehicle.h)
// the GPU shaders/vehicle_spring_solve.comp.hlsl copies VERBATIM + proves bit-identical. Pure CPU
// (header-only, no device, no backend symbols). Namespace hf::sim::vehicle.
//
// What this test PINS (the contracts the GPU vehicle_spring_solve.comp + the GPU==CPU proof build on):
//   * SolveSpringJoint: a body STRETCHED beyond restLen is pulled IN (length decreases toward restLen); a
//     body COMPRESSED below restLen is pushed OUT (length increases toward restLen); at exactly restLen the
//     positional correction is ~0; a pinned body (invMass 0) -> only the dynamic one moves; the velocity
//     damping reduces the relative normal speed; damping=0 -> no velocity change.
//   * StepSpringWorld: a displaced body on a DAMPED spring settles at ~restLen at REST (speed below a
//     threshold); two runs byte-identical; the UNDAMPED (damping=0) control keeps OSCILLATING (does not rest).
//
// HONEST CAVEAT (cloth/JT-identical): spring stiffness ∝ iterations — a stiff spring settles within a
// deterministic-but-nonzero band of restLen, NOT exactly restLen. The headline is DETERMINISM + cross-
// platform bit-identity, NOT analytic spring mechanics.
//
// Pure C++ (hf_core), ASan-eligible like the other sim-math tests.
#include "sim/vehicle.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace vehicle = hf::sim::vehicle;
namespace fpx = hf::sim::fpx;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// abs of a Q16.16 fx.
static vehicle::fx fxabs(vehicle::fx v) { return v < 0 ? -v : v; }

// Build a dynamic FxBody at (x,y,z) in Q16.16 with unit invMass + identity orient.
static fpx::FxBody dyn(int x, int y, int z) {
    fpx::FxBody b;
    b.pos = fpx::FxVec3{(vehicle::fx)(x * (int)vehicle::kOne), (vehicle::fx)(y * (int)vehicle::kOne),
                        (vehicle::fx)(z * (int)vehicle::kOne)};
    b.vel = fpx::FxVec3{0, 0, 0};
    b.invMass = vehicle::kOne;
    b.flags = fpx::kFlagDynamic;
    b.radius = 0;
    b.orient = fpx::FxQuat{0, 0, 0, vehicle::kOne};
    b.angVel = fpx::FxVec3{0, 0, 0};
    return b;
}
// Build a pinned (static, invMass 0) FxBody at (x,y,z).
static fpx::FxBody pinned(int x, int y, int z) {
    fpx::FxBody b = dyn(x, y, z);
    b.invMass = 0;
    b.flags = 0;
    return b;
}

// A spring (centre-to-centre anchors) tying bodies a<->b at restLen R with stiffness/damping.
static vehicle::FxSpringJoint spring(uint32_t a, uint32_t b, vehicle::fx restLen,
                                     vehicle::fx stiffness, vehicle::fx damping) {
    vehicle::FxSpringJoint j;
    j.bodyA = a; j.bodyB = b;
    j.anchorA = vehicle::FxVec3{0, 0, 0};
    j.anchorB = vehicle::FxVec3{0, 0, 0};
    j.restLen = restLen;
    j.stiffness = stiffness;
    j.damping = damping;
    return j;
}

static vehicle::FxWorld zeroGravWorld() {
    vehicle::FxWorld w;
    w.gravity = vehicle::FxVec3{0, 0, 0};
    w.groundY = (vehicle::fx)(-1000 * (int)vehicle::kOne);   // far below -> isolate the spring
    return w;
}

int main() {
    HF_TEST_MAIN_INIT();

    const vehicle::fx kRest = 2 * (int)vehicle::kOne;          // rest length 2.0
    const vehicle::fx kStiff = vehicle::kOne;                  // full positional restore (kOne)
    const vehicle::fx kDamp = vehicle::kOne / 2;               // half normal-velocity damping

    // ================= SpringLength: the anchor-to-anchor distance =================
    {
        vehicle::FxWorld w = zeroGravWorld();
        w.bodies = {dyn(0, 0, 0), dyn(5, 0, 0)};
        vehicle::FxSpringJoint j = spring(0, 1, kRest, kStiff, kDamp);
        check(vehicle::SpringLength(w, j) == (vehicle::fx)(5 * (int)vehicle::kOne),
              "SpringLength: anchor-to-anchor distance (5.0)");
    }

    // ================= SolveSpringJoint: a STRETCHED spring is pulled IN toward restLen =================
    {
        // Bodies at x=0 and x=6 -> length 6 > restLen 2: the spring is STRETCHED, so it pulls the bodies
        // together (length decreases toward restLen). stiffness kOne, equal masses -> each moves half of
        // (len - restLen) = 2 toward the midpoint -> body0 to x=2, body1 to x=4, new length 2 == restLen.
        vehicle::FxWorld w = zeroGravWorld();
        w.bodies = {dyn(0, 0, 0), dyn(6, 0, 0)};
        vehicle::FxSpringJoint j = spring(0, 1, kRest, kStiff, /*damping=*/0);
        const vehicle::fx before = vehicle::SpringLength(w, j);
        vehicle::SolveSpringJoint(w, j);
        const vehicle::fx after = vehicle::SpringLength(w, j);
        check(after < before, "SolveSpringJoint stretched: length decreases toward restLen (pulled in)");
        check(fxabs(after - kRest) <= vehicle::kOne / 256,
              "SolveSpringJoint stretched: stiffness kOne restores restLen in one pass (within LSB band)");
        check(w.bodies[0].pos.x == (vehicle::fx)(2 * (int)vehicle::kOne) &&
              w.bodies[1].pos.x == (vehicle::fx)(4 * (int)vehicle::kOne),
              "SolveSpringJoint stretched: equal masses each move half the deflection");
    }

    // ================= SolveSpringJoint: a COMPRESSED spring is pushed OUT toward restLen =================
    {
        // Bodies at x=0 and x=1 -> length 1 < restLen 2: the spring is COMPRESSED, so it pushes the bodies
        // apart (length increases toward restLen). pen = len - restLen = -1; the negative pen reverses the
        // correction sign -> the bodies move apart.
        vehicle::FxWorld w = zeroGravWorld();
        w.bodies = {dyn(0, 0, 0), dyn(1, 0, 0)};
        vehicle::FxSpringJoint j = spring(0, 1, kRest, kStiff, /*damping=*/0);
        const vehicle::fx before = vehicle::SpringLength(w, j);
        vehicle::SolveSpringJoint(w, j);
        const vehicle::fx after = vehicle::SpringLength(w, j);
        check(after > before, "SolveSpringJoint compressed: length increases toward restLen (pushed out)");
        check(fxabs(after - kRest) <= vehicle::kOne / 256,
              "SolveSpringJoint compressed: stiffness kOne restores restLen (within LSB band)");
        // body0 moves -X, body1 moves +X (apart).
        check(w.bodies[0].pos.x < 0 && w.bodies[1].pos.x > (vehicle::fx)(1 * (int)vehicle::kOne),
              "SolveSpringJoint compressed: bodies move apart");
    }

    // ================= SolveSpringJoint: at exactly restLen the positional correction is ~0 =============
    {
        vehicle::FxWorld w = zeroGravWorld();
        w.bodies = {dyn(0, 0, 0), dyn(2, 0, 0)};   // length 2 == restLen
        vehicle::FxSpringJoint j = spring(0, 1, kRest, kStiff, /*damping=*/0);
        const vehicle::FxVec3 p0 = w.bodies[0].pos, p1 = w.bodies[1].pos;
        vehicle::SolveSpringJoint(w, j);
        check(w.bodies[0].pos.x == p0.x && w.bodies[1].pos.x == p1.x,
              "SolveSpringJoint at restLen: positional correction is 0 (pen == 0)");
    }

    // ================= SolveSpringJoint: a pinned body -> only the dynamic one moves =================
    {
        // Body 0 PINNED at x=0, body 1 dynamic at x=6, restLen 2: the pinned anchor holds; the dynamic body
        // takes the FULL correction (moves 4 units to x=2 -> length 2 == restLen).
        vehicle::FxWorld w = zeroGravWorld();
        w.bodies = {pinned(0, 0, 0), dyn(6, 0, 0)};
        vehicle::FxSpringJoint j = spring(0, 1, kRest, kStiff, /*damping=*/0);
        vehicle::SolveSpringJoint(w, j);
        check(w.bodies[0].pos.x == 0, "SolveSpringJoint pinned: the pinned body never moves");
        check(w.bodies[1].pos.x == (vehicle::fx)(2 * (int)vehicle::kOne),
              "SolveSpringJoint pinned: the dynamic body takes the full correction to restLen");
    }

    // ================= SolveSpringJoint: both pinned -> no-op =================
    {
        vehicle::FxWorld w = zeroGravWorld();
        w.bodies = {pinned(0, 0, 0), pinned(6, 0, 0)};
        const vehicle::FxVec3 p0 = w.bodies[0].pos, p1 = w.bodies[1].pos;
        vehicle::SolveSpringJoint(w, spring(0, 1, kRest, kStiff, kDamp));
        check(w.bodies[0].pos.x == p0.x && w.bodies[1].pos.x == p1.x,
              "SolveSpringJoint both pinned: no-op (wsum 0 skip)");
    }

    // ================= SolveSpringJoint: coincident anchors -> no-op (len 0 skip) =================
    {
        vehicle::FxWorld w = zeroGravWorld();
        w.bodies = {dyn(2, 2, 2), dyn(2, 2, 2)};
        const vehicle::FxVec3 p0 = w.bodies[0].pos, p1 = w.bodies[1].pos;
        vehicle::SolveSpringJoint(w, spring(0, 1, kRest, kStiff, kDamp));
        check(w.bodies[0].pos.x == p0.x && w.bodies[0].pos.y == p0.y && w.bodies[0].pos.z == p0.z &&
              w.bodies[1].pos.x == p1.x && w.bodies[1].pos.y == p1.y && w.bodies[1].pos.z == p1.z,
              "SolveSpringJoint coincident anchors: no-op (len 0 skip)");
    }

    // ================= SolveSpringJoint: out-of-range body index -> no-op =================
    {
        vehicle::FxWorld w = zeroGravWorld();
        w.bodies = {dyn(0, 0, 0), dyn(6, 0, 0)};
        const vehicle::FxVec3 p0 = w.bodies[0].pos, p1 = w.bodies[1].pos;
        vehicle::FxSpringJoint j = spring(0, 99, kRest, kStiff, kDamp);   // out of range
        vehicle::SolveSpringJoint(w, j);
        check(w.bodies[0].pos.x == p0.x && w.bodies[1].pos.x == p1.x,
              "SolveSpringJoint out-of-range index: no-op");
    }

    // ================= SolveSpringJoint: the velocity damping reduces the relative normal speed ==========
    {
        // The bodies are AT restLen (pen 0, no positional motion) but APPROACHING along the spring normal
        // (body1 moving -X toward body0). The damper reduces the relative normal speed. damping kOne/2 ->
        // the relative normal velocity halves toward 0.
        vehicle::FxWorld w = zeroGravWorld();
        w.bodies = {dyn(0, 0, 0), dyn(2, 0, 0)};   // at restLen
        w.bodies[1].vel.x = -(vehicle::fx)vehicle::kOne;   // body1 moving toward body0 (closing the spring)
        vehicle::FxSpringJoint j = spring(0, 1, kRest, /*stiffness=*/0, kDamp);  // damping ONLY (no position)
        // relative normal speed before: vRel = (b.vel - a.vel)·n, n = +X -> -1.0.
        const vehicle::FxVec3 nrm{vehicle::kOne, 0, 0};
        const vehicle::fx relBefore =
            fpx::fxmul(w.bodies[1].vel.x - w.bodies[0].vel.x, nrm.x);
        vehicle::SolveSpringJoint(w, j);
        const vehicle::fx relAfter =
            fpx::fxmul(w.bodies[1].vel.x - w.bodies[0].vel.x, nrm.x);
        check(fxabs(relAfter) < fxabs(relBefore),
              "SolveSpringJoint damping: the relative normal speed is reduced");
        // positions unchanged (stiffness 0 -> no positional correction).
        check(w.bodies[0].pos.x == 0 && w.bodies[1].pos.x == (vehicle::fx)(2 * (int)vehicle::kOne),
              "SolveSpringJoint damping-only: no positional change (stiffness 0)");
    }

    // ================= SolveSpringJoint: damping=0 -> no velocity change =================
    {
        vehicle::FxWorld w = zeroGravWorld();
        w.bodies = {dyn(0, 0, 0), dyn(2, 0, 0)};   // at restLen so position is a no-op too
        w.bodies[1].vel.x = -(vehicle::fx)vehicle::kOne;
        const vehicle::FxVec3 v0 = w.bodies[0].vel, v1 = w.bodies[1].vel;
        vehicle::SolveSpringJoint(w, spring(0, 1, kRest, kStiff, /*damping=*/0));
        check(w.bodies[0].vel.x == v0.x && w.bodies[1].vel.x == v1.x,
              "SolveSpringJoint damping=0: velocities unchanged");
    }

    // ================= StepSpringWorld: a displaced body on a DAMPED spring settles at ~restLen, at REST =
    {
        // A pinned anchor at (0, 10, 0) + a dynamic body hung below; restLen 2 so the rest position is
        // (0, 8, 0). Start the body DISPLACED (at y=5, stretched to length 5) so it oscillates, then the
        // damper kills the swing -> it settles at ~restLen below the anchor, at rest. gravity ON (the
        // suspension holds it up against gravity).
        const vehicle::fx gravY = (vehicle::fx)(-9.8 * (double)vehicle::kOne + (-9.8 < 0 ? -0.5 : 0.5));
        vehicle::FxWorld w;
        w.gravity = vehicle::FxVec3{0, gravY, 0};
        w.groundY = (vehicle::fx)(-1000 * (int)vehicle::kOne);   // far below -> the focus is the spring
        w.bodies = {pinned(0, 10, 0), dyn(0, 5, 0)};             // displaced (length 5 > restLen 2)
        std::vector<vehicle::FxSpringJoint> springs = {
            spring(0, 1, kRest, vehicle::kOne / 4, vehicle::kOne / 4)};   // soft + damped
        const vehicle::fx dt = vehicle::kOne / 60;
        const int kIters = 16;
        vehicle::StepSpringWorldSteps(w, springs, dt, kIters, 600);

        // The pinned anchor NEVER moved.
        check(w.bodies[0].pos.x == 0 && w.bodies[0].pos.y == (vehicle::fx)(10 * (int)vehicle::kOne),
              "StepSpringWorld: the pinned anchor holds exactly");
        // The spring restored ~restLen: |len - restLen| within a documented band (the Gauss-Seidel residual
        // is deterministic-but-nonzero; under gravity it sags slightly below restLen — within ~1/2 unit).
        const vehicle::fx len = vehicle::SpringLength(w, springs[0]);
        check(fxabs(len - kRest) < vehicle::kOne / 2,
              "StepSpringWorld: the spring length settled within a small band of restLen");
        // The body came to REST: its speed dropped below a threshold (the damper killed the oscillation).
        const vehicle::fx speed = vehicle::FxLength(w.bodies[1].vel);
        check(speed < vehicle::kOne / 8,
              "StepSpringWorld: the body came to rest (speed below threshold — the damper did the work)");
        // The body hangs BELOW the anchor (the suspension holds it ~restLen under).
        check(w.bodies[1].pos.y < w.bodies[0].pos.y, "StepSpringWorld: the body hangs below the anchor");
    }

    // ================= StepSpringWorld: two runs byte-identical (determinism) =================
    {
        const vehicle::fx gravY = (vehicle::fx)(-9.8 * (double)vehicle::kOne + (-9.8 < 0 ? -0.5 : 0.5));
        auto build = [&]() {
            vehicle::FxWorld w;
            w.gravity = vehicle::FxVec3{0, gravY, 0};
            w.groundY = (vehicle::fx)(-1000 * (int)vehicle::kOne);
            w.bodies = {pinned(0, 10, 0), dyn(0, 5, 0)};
            return w;
        };
        std::vector<vehicle::FxSpringJoint> springs = {
            spring(0, 1, kRest, vehicle::kOne / 4, vehicle::kOne / 4)};
        const vehicle::fx dt = vehicle::kOne / 60;
        vehicle::FxWorld a = build(), b = build();
        vehicle::StepSpringWorldSteps(a, springs, dt, 16, 300);
        vehicle::StepSpringWorldSteps(b, springs, dt, 16, 300);
        const bool same = a.bodies.size() == b.bodies.size() &&
                          std::memcmp(a.bodies.data(), b.bodies.data(),
                                      a.bodies.size() * sizeof(fpx::FxBody)) == 0;
        check(same, "StepSpringWorld determinism: two runs BYTE-IDENTICAL");
    }

    // ================= StepSpringWorld: the UNDAMPED control keeps OSCILLATING (does not rest) ===========
    {
        // The SAME displaced scene but damping=0 -> the spring is a pure (soft) oscillator with no energy
        // sink, so after the same settle the body is STILL MOVING (its end speed exceeds the rest band) —
        // proving the DAMPER, not just the position solve, brings the damped body to rest.
        const vehicle::fx gravY = (vehicle::fx)(-9.8 * (double)vehicle::kOne + (-9.8 < 0 ? -0.5 : 0.5));
        vehicle::FxWorld w;
        w.gravity = vehicle::FxVec3{0, gravY, 0};
        w.groundY = (vehicle::fx)(-1000 * (int)vehicle::kOne);
        w.bodies = {pinned(0, 10, 0), dyn(0, 5, 0)};
        std::vector<vehicle::FxSpringJoint> springs = {
            spring(0, 1, kRest, vehicle::kOne / 4, /*damping=*/0)};   // NO damping
        const vehicle::fx dt = vehicle::kOne / 60;
        vehicle::StepSpringWorldSteps(w, springs, dt, 16, 600);
        const vehicle::fx speed = vehicle::FxLength(w.bodies[1].vel);
        check(speed >= vehicle::kOne / 8,
              "StepSpringWorld control: damping=0 keeps oscillating (end speed exceeds the rest band)");
    }

    // ============================================================================================
    // Slice VH2 — THE VEHICLE RIG + WHEEL HINGE (VehicleFromConfig / StepVehicleRig / MeasureVehicleRig).
    // ============================================================================================

    // ================= VehicleFromConfig: 5 bodies (1 chassis + 4 wheels), 4 springs, 4 hinges =========
    {
        vehicle::VehicleConfig cfg;   // the documented defaults
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        check(veh.world.bodies.size() == 5, "VehicleFromConfig: 5 bodies (1 chassis + 4 wheels)");
        check(veh.springs.size() == 4, "VehicleFromConfig: 4 suspension springs");
        check(veh.hinges.size() == 4, "VehicleFromConfig: 4 wheel hinges");
        check(veh.chassisIndex == 0, "VehicleFromConfig: chassis is body 0");
        // The chassis is at ride height (groundY + rideHeight).
        check(veh.world.bodies[veh.chassisIndex].pos.y == cfg.groundY + cfg.rideHeight,
              "VehicleFromConfig: the chassis is at ride height");
        // Each wheel hangs ~suspensionLen below the chassis centre, at its corner.
        for (int k = 0; k < 4; ++k) {
            const fpx::FxBody& w = veh.world.bodies[veh.wheelIndex[k]];
            check(w.pos.y == veh.world.bodies[veh.chassisIndex].pos.y - cfg.suspensionLen,
                  "VehicleFromConfig: the wheel hangs suspensionLen below the chassis");
            check(w.invMass == cfg.wheelInvMass, "VehicleFromConfig: the wheel invMass is set per cfg");
            check((w.flags & fpx::kFlagDynamic) != 0u, "VehicleFromConfig: the wheel is dynamic");
        }
        check(veh.world.bodies[veh.chassisIndex].invMass == cfg.chassisInvMass,
              "VehicleFromConfig: the chassis invMass is set per cfg");
        // Each spring ties the chassis to its wheel (bodyA chassis, bodyB wheel); each hinge is a hinge.
        for (int k = 0; k < 4; ++k) {
            check(veh.springs[k].bodyA == veh.chassisIndex && veh.springs[k].bodyB == veh.wheelIndex[k],
                  "VehicleFromConfig: spring k ties chassis -> wheel k");
            check(veh.springs[k].restLen == cfg.suspensionLen,
                  "VehicleFromConfig: spring restLen == suspensionLen");
            check(veh.hinges[k].bodyA == veh.chassisIndex && veh.hinges[k].bodyB == veh.wheelIndex[k],
                  "VehicleFromConfig: hinge k ties chassis -> wheel k");
            check(veh.hinges[k].kind == vehicle::kAngularHinge,
                  "VehicleFromConfig: hinge k is kAngularHinge");
        }
    }

    // ================= StepVehicleRig: a dropped car settles at ride height, wheels on the ground =======
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        const vehicle::fx dt = vehicle::kOne / 60;
        const int kIters = 16;
        vehicle::StepVehicleRigSteps(veh, dt, kIters, 400);   // settle

        const vehicle::VehicleRigState st = vehicle::MeasureVehicleRig(veh, cfg);
        // The wheels rest ON the ground (every wheel bottom within the band of groundY; nothing buried).
        check(st.wheelsOnGround, "StepVehicleRig: the 4 wheels rest on the ground");
        check(st.minWheelBottom >= cfg.groundY - vehicle::kOne / 16,
              "StepVehicleRig: no wheel is buried below the ground");
        // The chassis floats ABOVE the wheels (the suspension holds it up).
        const vehicle::fx wheelMeanY = [&]() {
            int64_t s = 0;
            for (int k = 0; k < 4; ++k) s += (int64_t)veh.world.bodies[veh.wheelIndex[k]].pos.y;
            return (vehicle::fx)(s / 4);
        }();
        check(st.chassisY > wheelMeanY, "StepVehicleRig: the chassis floats above the wheels");
        // The chassis settled in a plausible ride-height band (above the ground, not flown away).
        check(st.chassisY > cfg.groundY && st.chassisY < cfg.groundY + 4 * (int)vehicle::kOne,
              "StepVehicleRig: the chassis settled in the ride-height band");
        // The springs are COMPRESSED (mean spring length < restLen — they hold the chassis up).
        check(st.meanSpringLen < cfg.suspensionLen,
              "StepVehicleRig: the springs are compressed (mean length < restLen — holding the chassis)");
        // The hinges keep the wheels IN-PLANE (the off-axis swing is small).
        check(st.maxHingeOffAxis < vehicle::kOne / 4,
              "StepVehicleRig: the hinges keep the wheels in their rolling plane");
    }

    // ================= StepVehicleRig: two runs byte-identical (determinism) =================
    {
        vehicle::VehicleConfig cfg;
        const vehicle::fx dt = vehicle::kOne / 60;
        vehicle::Vehicle a = vehicle::VehicleFromConfig(cfg);
        vehicle::Vehicle b = vehicle::VehicleFromConfig(cfg);
        vehicle::StepVehicleRigSteps(a, dt, 16, 200);
        vehicle::StepVehicleRigSteps(b, dt, 16, 200);
        const bool same = a.world.bodies.size() == b.world.bodies.size() &&
                          std::memcmp(a.world.bodies.data(), b.world.bodies.data(),
                                      a.world.bodies.size() * sizeof(fpx::FxBody)) == 0;
        check(same, "StepVehicleRig determinism: two runs BYTE-IDENTICAL");
    }

    if (g_fail == 0) std::printf("vehicle_test: ALL PASS\n");
    else std::printf("vehicle_test: %d FAILURE(S)\n", g_fail);
    return g_fail ? 1 : 0;
}
