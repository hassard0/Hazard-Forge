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

    // ============================================================================================
    // Slice VH3 — DRIVE + STEER COMMANDS + THE LOCKED VEHICLE TICK (ApplyVehicleCommand / StepVehicle /
    // StepVehicleSteps / MeasureVehicleDrive). The car RESPONDS TO INPUT: kCmdDriveTorque spins a wheel,
    // kCmdSteer re-aims a front hinge; StepVehicle folds the FPX2 broadphase + FPX3 contacts into the VH2
    // spring/hinge solve (the JT3 trailing-block mold). Pure CPU, ASan-eligible.
    // ============================================================================================

    // Build a steer command (front hinge `target` via FxCommand::bodyId, Q16.16 steer angle in arg.x).
    auto steerCmd = [](uint32_t tick, uint32_t target, vehicle::fx angle) {
        vehicle::FxCommand c; c.tick = tick; c.kind = vehicle::kCmdSteer; c.bodyId = target;
        c.arg = vehicle::FxVec3{angle, 0, 0}; return c;
    };
    // Build a drive-torque command (wheel `target` via FxCommand::bodyId, Q16.16 axle spin in arg.x).
    auto driveCmd = [](uint32_t tick, uint32_t target, vehicle::fx spin) {
        vehicle::FxCommand c; c.tick = tick; c.kind = vehicle::kCmdDriveTorque; c.bodyId = target;
        c.arg = vehicle::FxVec3{spin, 0, 0}; return c;
    };

    // ================= ApplyVehicleCommand: kCmdDriveTorque raises the target wheel's angVel ==============
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        const uint32_t wheel = veh.wheelIndex[3];   // a rear wheel
        const vehicle::fx spin = vehicle::kOne * 4;  // a strong axle spin (Q16.16)
        const vehicle::FxVec3 before = veh.world.bodies[wheel].angVel;
        vehicle::ApplyVehicleCommand(veh, cfg, driveCmd(0, wheel, spin));
        const vehicle::FxVec3 after = veh.world.bodies[wheel].angVel;
        // The drive adds FxScale(spinAxis=(0,0,1), spin) -> angVel.z increases by spin; x/y unchanged.
        check(after.z == before.z + spin, "ApplyVehicleCommand drive: target wheel angVel.z += spin");
        check(after.x == before.x && after.y == before.y,
              "ApplyVehicleCommand drive: only the axle (Z) component changed");
        // The OTHER wheels are unchanged.
        for (int k = 0; k < 4; ++k) {
            if (veh.wheelIndex[k] == wheel) continue;
            check(veh.world.bodies[veh.wheelIndex[k]].angVel.z == 0,
                  "ApplyVehicleCommand drive: non-target wheels unchanged");
        }
    }

    // ================= ApplyVehicleCommand: a static / out-of-range drive target is a no-op ==============
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        // Out-of-range target: no crash, no change anywhere.
        std::vector<fpx::FxBody> before = veh.world.bodies;
        vehicle::ApplyVehicleCommand(veh, cfg, driveCmd(0, 999u, vehicle::kOne));
        check(std::memcmp(veh.world.bodies.data(), before.data(),
                          before.size() * sizeof(fpx::FxBody)) == 0,
              "ApplyVehicleCommand drive: out-of-range target is a no-op");
        // A STATIC body target: pin the chassis static, drive it -> no-op.
        veh.world.bodies[veh.chassisIndex].flags = 0;        // make it static
        veh.world.bodies[veh.chassisIndex].invMass = 0;
        const vehicle::FxVec3 cBefore = veh.world.bodies[veh.chassisIndex].angVel;
        vehicle::ApplyVehicleCommand(veh, cfg, driveCmd(0, veh.chassisIndex, vehicle::kOne * 5));
        check(veh.world.bodies[veh.chassisIndex].angVel.x == cBefore.x &&
              veh.world.bodies[veh.chassisIndex].angVel.y == cBefore.y &&
              veh.world.bodies[veh.chassisIndex].angVel.z == cBefore.z,
              "ApplyVehicleCommand drive: a static target is a no-op");
    }

    // ================= ApplyVehicleCommand: kCmdSteer rotates ONLY the targeted front hinge ==============
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        // The rest hinge axis is the lateral (0,0,1) Z axle -> axis.x == 0 for all 4 hinges.
        for (int k = 0; k < 4; ++k)
            check(veh.hinges[k].axis.x == 0, "ApplyVehicleCommand steer: rest hinge axis.x == 0");
        const vehicle::fx steerAngle = vehicle::kOne / 4;   // ~0.25 rad steer (Q16.16)
        vehicle::ApplyVehicleCommand(veh, cfg, steerCmd(0, 0u, steerAngle));  // steer front-right (hinge 0)
        // Hinge 0's axis re-aimed about the up axis (Y) -> its X component is now NON-zero (rotated).
        check(veh.hinges[0].axis.x != 0,
              "ApplyVehicleCommand steer: the targeted front hinge axis was re-aimed (axis.x != 0)");
        // The OTHER hinges (1 front-left, 2/3 rear) are UNCHANGED.
        check(veh.hinges[1].axis.x == 0 && veh.hinges[2].axis.x == 0 && veh.hinges[3].axis.x == 0,
              "ApplyVehicleCommand steer: only the targeted hinge changed");
        // The re-aimed axis is still ~unit length (FxNormalize after the rotate).
        const vehicle::fx len = vehicle::FxLength(veh.hinges[0].axis);
        check(fxabs(len - vehicle::kOne) < vehicle::kOne / 64,
              "ApplyVehicleCommand steer: the re-aimed hinge axis stays unit length");
    }

    // ================= ApplyVehicleCommand: a non-front / out-of-range steer target is a no-op ===========
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        const vehicle::fx a2 = veh.hinges[2].axis.x, a3 = veh.hinges[3].axis.x;
        vehicle::ApplyVehicleCommand(veh, cfg, steerCmd(0, 2u, vehicle::kOne / 4));  // rear hinge -> no-op
        vehicle::ApplyVehicleCommand(veh, cfg, steerCmd(0, 99u, vehicle::kOne / 4)); // OOB -> no-op
        check(veh.hinges[2].axis.x == a2 && veh.hinges[3].axis.x == a3,
              "ApplyVehicleCommand steer: a non-front / out-of-range target is a no-op");
    }

    // ================= StepVehicle: a drive stream moves the chassis forward + spins the driven wheels ===
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        const vehicle::fx dt = vehicle::kOne / 60;
        const int kIters = 16, kSolveIters = 8, kTicks = 120;
        const vehicle::fx startX = veh.world.bodies[veh.chassisIndex].pos.x;
        const std::vector<uint32_t> driven = {veh.wheelIndex[2], veh.wheelIndex[3]};  // the rears

        // A scripted stream: spin BOTH rear wheels every tick + push the chassis forward via an axle-aligned
        // X impulse on the chassis (drive comes from the command/integrate path — the VH3/VH4 caveat) +
        // steer the front-right hinge once. We model "throttle" as a per-tick chassis forward velocity bump
        // through a drive command on the chassis body (kCmdDriveTorque sets angVel; to MOVE the chassis we
        // also nudge its X velocity here via the stream's fpx-style impulse is NOT available — instead we
        // give the rear wheels a forward velocity via a dedicated forward push on the chassis using a
        // direct vel set, mirroring the showcase). For the unit test we assert the SPIN took + the chassis
        // moved forward under a forward seed.
        std::vector<vehicle::FxCommand> stream;
        for (int t = 0; t < kTicks; ++t) {
            stream.push_back(driveCmd((uint32_t)t, veh.wheelIndex[2], vehicle::kOne));   // spin rear-right
            stream.push_back(driveCmd((uint32_t)t, veh.wheelIndex[3], vehicle::kOne));   // spin rear-left
        }
        stream.push_back(steerCmd(0, 0u, vehicle::kOne / 4));  // steer the front-right hinge once

        // Seed the chassis with a forward (X) velocity so the drive stream's tick visibly moves it forward
        // (the integrate carries pos += vel*dt; the VH3 caveat: contacts don't spin a body, so the forward
        // motion is the command/integrate path — here the seed stands in for VH4 traction).
        veh.world.bodies[veh.chassisIndex].vel.x = vehicle::kOne;   // +1 unit/s forward

        vehicle::StepVehicleSteps(veh, cfg, stream, dt, kTicks, kIters, kSolveIters);

        const vehicle::VehicleDriveState st = vehicle::MeasureVehicleDrive(veh, cfg, driven, startX);
        check(st.forwardDisp > 0, "StepVehicle drive: the chassis moved forward (forwardDisp > 0)");
        check(st.meanDrivenAngVel > 0, "StepVehicle drive: the driven wheels are spinning (meanAngVel > 0)");
        // The steer re-aimed the front hinge (its axle gained an X component) while the rears did not.
        check(fxabs(st.frontHingeHeadingX) > 0 && st.rearHingeHeadingX == 0,
              "StepVehicle drive: the front hinge re-aimed, the rears unchanged");
    }

    // ================= StepVehicle: a no-command run stays at the VH2 settled pose (forward ~ 0) =========
    {
        vehicle::VehicleConfig cfg;
        const vehicle::fx dt = vehicle::kOne / 60;
        const int kIters = 16, kSolveIters = 8, kTicks = 200;

        // The VH2 settled reference: StepVehicleRig K times.
        vehicle::Vehicle rigRef = vehicle::VehicleFromConfig(cfg);
        // The VH3 no-command control: StepVehicle K times with an EMPTY stream. With no commands + no
        // forward seed the chassis must NOT drive forward (forwardDisp ~ 0) — it just settles.
        vehicle::Vehicle ctrl = vehicle::VehicleFromConfig(cfg);
        const vehicle::fx startX = ctrl.world.bodies[ctrl.chassisIndex].pos.x;
        const std::vector<vehicle::FxCommand> empty;
        vehicle::StepVehicleSteps(ctrl, cfg, empty, dt, kTicks, kIters, kSolveIters);

        const vehicle::VehicleDriveState st =
            vehicle::MeasureVehicleDrive(ctrl, cfg, {ctrl.wheelIndex[2], ctrl.wheelIndex[3]}, startX);
        check(fxabs(st.forwardDisp) < vehicle::kOne / 4,
              "StepVehicle no-command: the chassis stays put (forwardDisp ~ 0)");
        // The car still SETTLED: chassis in the ride-height band, wheels at/above the ground.
        const vehicle::VehicleRigState rs = vehicle::MeasureVehicleRig(ctrl, cfg);
        check(rs.chassisY > cfg.groundY && rs.chassisY < cfg.groundY + 4 * (int)vehicle::kOne,
              "StepVehicle no-command: the chassis settled in the ride-height band");
        check(rs.minWheelBottom >= cfg.groundY - vehicle::kOne,
              "StepVehicle no-command: no wheel is buried (the contact pass held)");
    }

    // ================= StepVehicle: two runs byte-identical (determinism) =================
    {
        vehicle::VehicleConfig cfg;
        const vehicle::fx dt = vehicle::kOne / 60;
        const int kIters = 16, kSolveIters = 8, kTicks = 120;
        std::vector<vehicle::FxCommand> stream;
        for (int t = 0; t < kTicks; ++t)
            stream.push_back(driveCmd((uint32_t)t, 3u, vehicle::kOne));
        stream.push_back(steerCmd(0, 1u, vehicle::kOne / 4));

        vehicle::Vehicle a = vehicle::VehicleFromConfig(cfg);
        vehicle::Vehicle b = vehicle::VehicleFromConfig(cfg);
        vehicle::StepVehicleSteps(a, cfg, stream, dt, kTicks, kIters, kSolveIters);
        vehicle::StepVehicleSteps(b, cfg, stream, dt, kTicks, kIters, kSolveIters);
        const bool same = a.world.bodies.size() == b.world.bodies.size() &&
                          std::memcmp(a.world.bodies.data(), b.world.bodies.data(),
                                      a.world.bodies.size() * sizeof(fpx::FxBody)) == 0;
        check(same, "StepVehicle determinism: two runs BYTE-IDENTICAL");
    }

    // ================= StepVehicle: residual contact overlaps within band (non-interpenetrating) =========
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        const vehicle::fx dt = vehicle::kOne / 60;
        const int kIters = 16, kSolveIters = 8, kTicks = 200;
        const vehicle::fx startX = veh.world.bodies[veh.chassisIndex].pos.x;
        vehicle::StepVehicleSteps(veh, cfg, {}, dt, kTicks, kIters, kSolveIters);
        const vehicle::VehicleDriveState st =
            vehicle::MeasureVehicleDrive(veh, cfg, {veh.wheelIndex[2], veh.wheelIndex[3]}, startX);
        // The car is a 5-body rig with the wheels spread at the corners -> the contact pass keeps it
        // non-interpenetrating; the residual-overlap count is within a documented small band.
        check(st.residualOverlaps <= 4u,
              "StepVehicle contacts: residual overlaps within band (the car non-interpenetrating)");
    }

    // ====================================================================================================
    // Slice VH4 — WHEEL-GROUND TRACTION / FRICTION (ApplyWheelTraction / StepVehicleDriven /
    // StepVehicleDrivenSteps). The car drives from wheel SPIN via a Coulomb-cone tangential ground force —
    // NOT a velocity seed. Pure CPU; ASan-eligible.

    // ================= ApplyWheelTraction: a grounded spinning wheel accelerates the chassis along fwd ===
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        // Settle the rig so the wheels are grounded (their bottoms within kContactEps of groundY).
        const vehicle::fx dt = vehicle::kOne / 60;
        vehicle::StepVehicleRigSteps(veh, dt, 16, 200);
        // Spin a rear wheel about its axle (the +Z lateral spin axis -> a +X rolling direction).
        const fpx::FxVec3 chassisVel0 = veh.world.bodies[veh.chassisIndex].vel;
        veh.world.bodies[veh.wheelIndex[2]].angVel =
            fpx::FxVec3{0, 0, 4 * (int)vehicle::kOne};   // spin about +Z axle
        veh.world.bodies[veh.wheelIndex[3]].angVel =
            fpx::FxVec3{0, 0, 4 * (int)vehicle::kOne};
        vehicle::ApplyWheelTraction(veh, cfg);
        const fpx::FxVec3 chassisVel1 = veh.world.bodies[veh.chassisIndex].vel;
        // The chassis gained forward (X) velocity from the gripped spin (fwd = up × Z-axle = +X).
        check(chassisVel1.x > chassisVel0.x,
              "ApplyWheelTraction: a grounded spinning wheel accelerated the chassis forward (+X)");
        // The wheel spin was bled (the tyre gripped — momentum left the spin).
        check(veh.world.bodies[veh.wheelIndex[2]].angVel.z < 4 * (int)vehicle::kOne,
              "ApplyWheelTraction: the gripped wheel's spin was bled toward no-slip");
    }

    // ================= ApplyWheelTraction: a NON-grounded wheel contributes no traction =================
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        // Lift the whole car well above the ground so NO wheel is grounded.
        const vehicle::fx lift = 10 * (int)vehicle::kOne;
        for (auto& b : veh.world.bodies) b.pos.y += lift;
        // Spin all wheels.
        for (int k = 0; k < 4; ++k)
            veh.world.bodies[veh.wheelIndex[k]].angVel = fpx::FxVec3{0, 0, 4 * (int)vehicle::kOne};
        const fpx::FxVec3 chassisVel0 = veh.world.bodies[veh.chassisIndex].vel;
        vehicle::ApplyWheelTraction(veh, cfg);
        const fpx::FxVec3 chassisVel1 = veh.world.bodies[veh.chassisIndex].vel;
        check(chassisVel1.x == chassisVel0.x && chassisVel1.y == chassisVel0.y &&
              chassisVel1.z == chassisVel0.z,
              "ApplyWheelTraction: airborne wheels contribute NO traction (chassis vel unchanged)");
    }

    // ================= ApplyWheelTraction: kMuMax==0 -> zero chassis acceleration (cone saturates) =======
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        const vehicle::fx dt = vehicle::kOne / 60;
        vehicle::StepVehicleRigSteps(veh, dt, 16, 200);
        // Isolate ONE grounded driven wheel (lift the other three) so the chassis gain is bounded by a
        // SINGLE friction cone. The friction-cone proof: an over-spun wheel's per-tick chassis gain
        // saturates at kMuMax * kChassisShare, NOT the raw slip*kGripK (which would be enormous). The
        // PRIMARY no-grip proof is the showcase's kMuMax==0 path; here we assert the saturation bound.
        for (int k = 1; k < 4; ++k) veh.world.bodies[veh.wheelIndex[k]].pos.y += 10 * (int)vehicle::kOne;
        veh.world.bodies[veh.chassisIndex].vel = fpx::FxVec3{0, 0, 0};
        veh.world.bodies[veh.wheelIndex[0]].angVel = fpx::FxVec3{0, 0, 1000 * (int)vehicle::kOne};
        vehicle::ApplyWheelTraction(veh, cfg);
        const vehicle::fx gain = veh.world.bodies[veh.chassisIndex].vel.x;
        // The single grounded wheel's huge slip clamps to +kMuMax -> chassis gains kMuMax*kChassisShare.
        const vehicle::fx perWheelCap = vehicle::fxmul(vehicle::kMuMax, vehicle::kChassisShare);
        check(gain > 0 && gain <= perWheelCap + 1,
              "ApplyWheelTraction: a huge over-spin SATURATES to the friction cone (bounded chassis gain)");
    }

    // ================= ApplyWheelTraction: a steered front wheel pushes along its re-aimed heading ========
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        const vehicle::fx dt = vehicle::kOne / 60;
        vehicle::StepVehicleRigSteps(veh, dt, 16, 200);
        // Steer the front-right hinge (index 0) so its axle re-aims -> fwd gains a Z component.
        vehicle::ApplyVehicleCommand(veh, cfg, steerCmd(0, 0u, vehicle::kOne / 2));  // ~0.5 rad
        // Zero the chassis vel + spin ONLY the steered front wheel.
        veh.world.bodies[veh.chassisIndex].vel = fpx::FxVec3{0, 0, 0};
        veh.world.bodies[veh.wheelIndex[0]].angVel = fpx::FxVec3{0, 0, 4 * (int)vehicle::kOne};
        // Make ONLY wheel 0 grounded (lift the others so they don't contribute) to isolate the heading.
        for (int k = 1; k < 4; ++k) veh.world.bodies[veh.wheelIndex[k]].pos.y += 10 * (int)vehicle::kOne;
        vehicle::ApplyWheelTraction(veh, cfg);
        const fpx::FxVec3 cv = veh.world.bodies[veh.chassisIndex].vel;
        // The re-aimed axle (now with an X component) gives fwd = up × axle a LATERAL (Z) component, so the
        // chassis gains a Z component it would NOT have at the rest heading (fwd would be pure +X).
        check(fxabs(cv.z) > 0,
              "ApplyWheelTraction: a steered front wheel pushes along its re-aimed heading (lateral gain)");
    }

    // ================= StepVehicleDriven: a drive stream with NO seed moves the chassis forward ==========
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        const vehicle::fx dt = vehicle::kOne / 60;
        const int kIters = 16, kSolveIters = 8, kTicks = 240;
        const vehicle::fx startX = veh.world.bodies[veh.chassisIndex].pos.x;
        const std::vector<uint32_t> driven = {veh.wheelIndex[2], veh.wheelIndex[3]};
        // Spin BOTH rear wheels every tick — NO chassis velocity seed (the VH3 seed is GONE; forward must
        // come from TRACTION).
        std::vector<vehicle::FxCommand> stream;
        for (int t = 0; t < kTicks; ++t) {
            stream.push_back(driveCmd((uint32_t)t, veh.wheelIndex[2], vehicle::kOne));
            stream.push_back(driveCmd((uint32_t)t, veh.wheelIndex[3], vehicle::kOne));
        }
        vehicle::StepVehicleDrivenSteps(veh, cfg, stream, dt, kTicks, kIters, kSolveIters);
        const vehicle::VehicleDriveState st = vehicle::MeasureVehicleDrive(veh, cfg, driven, startX);
        check(st.forwardDisp > vehicle::kOne / 2,
              "StepVehicleDriven: the drive stream moved the chassis forward WITHOUT a seed (traction)");
        check(st.meanDrivenAngVel > 0,
              "StepVehicleDriven: the driven wheels are spinning (the throttle took)");
    }

    // ================= StepVehicleDriven: a no-grip control (no commands, no seed) stays put =============
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle ctrl = vehicle::VehicleFromConfig(cfg);
        const vehicle::fx dt = vehicle::kOne / 60;
        const int kIters = 16, kSolveIters = 8, kTicks = 240;
        const vehicle::fx startX = ctrl.world.bodies[ctrl.chassisIndex].pos.x;
        const std::vector<vehicle::FxCommand> empty;
        vehicle::StepVehicleDrivenSteps(ctrl, cfg, empty, dt, kTicks, kIters, kSolveIters);
        const vehicle::VehicleDriveState st = vehicle::MeasureVehicleDrive(
            ctrl, cfg, {ctrl.wheelIndex[2], ctrl.wheelIndex[3]}, startX);
        check(fxabs(st.forwardDisp) < vehicle::kOne / 4,
              "StepVehicleDriven control: no commands + no seed -> the chassis stays put (no traction)");
    }

    // ================= StepVehicleDriven: two runs byte-identical (determinism) =================
    {
        vehicle::VehicleConfig cfg;
        const vehicle::fx dt = vehicle::kOne / 60;
        const int kIters = 16, kSolveIters = 8, kTicks = 120;
        std::vector<vehicle::FxCommand> stream;
        for (int t = 0; t < kTicks; ++t) {
            stream.push_back(driveCmd((uint32_t)t, 3u, vehicle::kOne));
            stream.push_back(driveCmd((uint32_t)t, 4u, vehicle::kOne));
        }
        stream.push_back(steerCmd(0, 0u, vehicle::kOne / 4));
        vehicle::Vehicle a = vehicle::VehicleFromConfig(cfg);
        vehicle::Vehicle b = vehicle::VehicleFromConfig(cfg);
        vehicle::StepVehicleDrivenSteps(a, cfg, stream, dt, kTicks, kIters, kSolveIters);
        vehicle::StepVehicleDrivenSteps(b, cfg, stream, dt, kTicks, kIters, kSolveIters);
        const bool same = a.world.bodies.size() == b.world.bodies.size() &&
                          std::memcmp(a.world.bodies.data(), b.world.bodies.data(),
                                      a.world.bodies.size() * sizeof(fpx::FxBody)) == 0;
        check(same, "StepVehicleDriven determinism: two runs BYTE-IDENTICAL");
    }

    // ============================================================================================
    // Slice VH5 — LOCKSTEP + ROLLBACK (SnapshotVehicle / RestoreVehicle / SimVehicleTick /
    // RunVehicleLockstep / RunVehicleRollback). The bit-exact driven tick is true cross-platform LOCKSTEP +
    // ROLLBACK: two peers fed ONLY the command stream re-derive the car bit-for-bit; a mispredicted steer is
    // corrected by rolling back to a snapshot (INCLUDING the steered hinge axes) + re-simulating. PURE CPU.
    // ============================================================================================

    // A bodies+hinge-axes equality predicate (the lockstep proof: the snapshot captures BOTH).
    auto vehicleEqual = [](const vehicle::Vehicle& a, const vehicle::Vehicle& b) {
        if (a.world.bodies.size() != b.world.bodies.size()) return false;
        if (std::memcmp(a.world.bodies.data(), b.world.bodies.data(),
                        a.world.bodies.size() * sizeof(fpx::FxBody)) != 0) return false;
        if (a.hinges.size() != b.hinges.size()) return false;
        for (size_t i = 0; i < a.hinges.size(); ++i) {
            const vehicle::FxVec3& xa = a.hinges[i].axis;
            const vehicle::FxVec3& xb = b.hinges[i].axis;
            if (xa.x != xb.x || xa.y != xb.y || xa.z != xb.z) return false;
        }
        return true;
    };
    // A snapshot==vehicle predicate (the round-trip proof: bodies + the 4 captured hinge axes).
    auto snapEqualsVehicle = [](const vehicle::VehicleSnapshot& s, const vehicle::Vehicle& v) {
        if (s.bodies.bodies.size() != v.world.bodies.size()) return false;
        if (std::memcmp(s.bodies.bodies.data(), v.world.bodies.data(),
                        v.world.bodies.size() * sizeof(fpx::FxBody)) != 0) return false;
        for (int k = 0; k < 4; ++k) {
            const vehicle::FxVec3& sa = s.hingeAxis[k];
            const vehicle::FxVec3& va = v.hinges[(size_t)k].axis;
            if (sa.x != va.x || sa.y != va.y || sa.z != va.z) return false;
        }
        return true;
    };

    // The shared VH5 drive+steer scene (a non-trivial path: spin the rear wheels every tick + steer a front
    // hinge at a few ticks so the steered hinge axes become live replayable state the snapshot must carry).
    auto buildVh5Stream = [&](const vehicle::Vehicle& veh, int ticks) {
        std::vector<vehicle::FxCommand> stream;
        for (int t = 0; t < ticks; ++t) {
            stream.push_back(driveCmd((uint32_t)t, veh.wheelIndex[2], vehicle::kOne));
            stream.push_back(driveCmd((uint32_t)t, veh.wheelIndex[3], vehicle::kOne));
        }
        // Steer the front-right hinge (index 0) at ticks 4 and 9 -> the hinge axis is genuinely mutated.
        stream.push_back(steerCmd(4u, 0u, vehicle::kOne / 4));
        stream.push_back(steerCmd(9u, 0u, vehicle::kOne / 4));
        return stream;
    };

    const vehicle::fx kVh5Dt = vehicle::kOne / 60;
    const int kVh5Iters = 16, kVh5SolveIters = 8, kVh5Ticks = 40;
    const int kVh5DivergeTick = 12;

    // ================= SnapshotVehicle/RestoreVehicle round-trip INCLUDING the hinge axes ================
    {
        vehicle::VehicleConfig cfg;
        vehicle::Vehicle veh = vehicle::VehicleFromConfig(cfg);
        const std::vector<vehicle::FxCommand> stream = buildVh5Stream(veh, kVh5Ticks);
        // Advance a few ticks (PAST the first steer at tick 4) so the bodies moved AND a front hinge axis
        // was steered off its rest heading -> the snapshot must capture the steered state.
        for (int t = 0; t < 8; ++t)
            vehicle::SimVehicleTick(veh, cfg, stream, (uint32_t)t, kVh5Dt, kVh5Iters, kVh5SolveIters);
        // The front hinge actually steered (axis.x != 0 -> a non-vacuous hinge-axis-in-snapshot proof).
        check(veh.hinges[0].axis.x != 0,
              "VH5 snapshot scene: the front hinge axis was steered off its rest heading (axis.x != 0)");
        const vehicle::VehicleSnapshot snap = vehicle::SnapshotVehicle(veh);
        check(snapEqualsVehicle(snap, veh),
              "VH5 SnapshotVehicle: captures the bodies AND the 4 hinge axes BIT-EXACT");
        // Mutate (drive+steer a few more ticks -> the bodies AND the steered hinge axis change), then restore.
        const vehicle::FxVec3 preAxis = veh.hinges[0].axis;
        for (int t = 8; t < 12; ++t)
            vehicle::SimVehicleTick(veh, cfg, stream, (uint32_t)t, kVh5Dt, kVh5Iters, kVh5SolveIters);
        // (tick 9 steered the front hinge again -> the live axis differs from the snapshot's.)
        check(veh.hinges[0].axis.x != preAxis.x || veh.hinges[0].axis.y != preAxis.y ||
              veh.hinges[0].axis.z != preAxis.z,
              "VH5 snapshot scene: mutation changed the steered hinge axis (a real hinge divergence)");
        vehicle::RestoreVehicle(veh, snap);
        check(snapEqualsVehicle(snap, veh),
              "VH5 RestoreVehicle: restores the bodies AND the hinge axes to the snapshot BIT-EXACT");
        check(veh.hinges[0].axis.x == preAxis.x && veh.hinges[0].axis.y == preAxis.y &&
              veh.hinges[0].axis.z == preAxis.z,
              "VH5 RestoreVehicle: the restored hinge axis == the pre-mutation steered axis");
    }

    // ================= RunVehicleLockstep: authority == replica BIT-IDENTICAL (inputs-only) ==============
    {
        vehicle::VehicleConfig cfg;
        const vehicle::Vehicle init = vehicle::VehicleFromConfig(cfg);
        const std::vector<vehicle::FxCommand> stream = buildVh5Stream(init, kVh5Ticks);
        const vehicle::Vehicle authority =
            vehicle::RunVehicleLockstep(cfg, init, stream, kVh5Ticks, kVh5Dt, kVh5Iters, kVh5SolveIters);
        const vehicle::Vehicle replica =
            vehicle::RunVehicleLockstep(cfg, init, stream, kVh5Ticks, kVh5Dt, kVh5Iters, kVh5SolveIters);
        check(vehicleEqual(authority, replica),
              "VH5 lockstep: authority == replica BIT-IDENTICAL (inputs-only re-sim, bodies + hinge axes)");
        // The stream did non-trivial work: the steer re-aimed the front hinge away from its rest lateral
        // heading (axis.x != 0) -> the replay is non-vacuous.
        check(authority.hinges[0].axis.x != 0,
              "VH5 lockstep: the steer stream re-aimed the front hinge (the inputs do work)");
    }

    // ================= RunVehicleRollback: mispredict diverges, rollback corrects to authority ===========
    {
        vehicle::VehicleConfig cfg;
        const vehicle::Vehicle init = vehicle::VehicleFromConfig(cfg);
        const std::vector<vehicle::FxCommand> authStream = buildVh5Stream(init, kVh5Ticks);
        // The MISPREDICTED stream: authStream + a WRONG strong steer on the front hinge AT divergeTick (so
        // the divergence is specifically a STEERED-HINGE divergence — the VH twist the snapshot must carry).
        std::vector<vehicle::FxCommand> mispredictStream = authStream;
        mispredictStream.push_back(steerCmd((uint32_t)kVh5DivergeTick, 0u, vehicle::kOne));   // wrong big steer

        const vehicle::Vehicle authority =
            vehicle::RunVehicleLockstep(cfg, init, authStream, kVh5Ticks, kVh5Dt, kVh5Iters, kVh5SolveIters);
        const vehicle::Vehicle rolledBack =
            vehicle::RunVehicleRollback(cfg, init, authStream, mispredictStream, kVh5DivergeTick,
                                        kVh5DivergeTick, kVh5Ticks, kVh5Dt, kVh5Iters, kVh5SolveIters);
        const vehicle::Vehicle mispredicted =
            vehicle::RunVehicleLockstep(cfg, init, mispredictStream, kVh5Ticks, kVh5Dt, kVh5Iters,
                                        kVh5SolveIters);
        check(vehicleEqual(rolledBack, authority),
              "VH5 rollback: corrected == authority BIT-EXACT (bodies + hinge axes)");
        check(!vehicleEqual(mispredicted, authority),
              "VH5 rollback: the mispredicted state DIFFERED from authority (a real divergence, non-vacuous)");
        // The divergence is specifically a STEERED-HINGE divergence: the mispredicted front hinge axis differs
        // from authority's (the snapshot/restore preserving the hinge axes is what fixes it).
        check(mispredicted.hinges[0].axis.x != authority.hinges[0].axis.x ||
              mispredicted.hinges[0].axis.y != authority.hinges[0].axis.y ||
              mispredicted.hinges[0].axis.z != authority.hinges[0].axis.z,
              "VH5 rollback: the mispredicted front-hinge axis diverged (the steered-state divergence)");
        check(rolledBack.hinges[0].axis.x == authority.hinges[0].axis.x &&
              rolledBack.hinges[0].axis.y == authority.hinges[0].axis.y &&
              rolledBack.hinges[0].axis.z == authority.hinges[0].axis.z,
              "VH5 rollback: the corrected front-hinge axis == authority (the steered state was restored)");
    }

    // ================= RunVehicleLockstep: two full runs byte-identical (determinism) ====================
    {
        vehicle::VehicleConfig cfg;
        const vehicle::Vehicle init = vehicle::VehicleFromConfig(cfg);
        const std::vector<vehicle::FxCommand> stream = buildVh5Stream(init, kVh5Ticks);
        const vehicle::Vehicle a =
            vehicle::RunVehicleLockstep(cfg, init, stream, kVh5Ticks, kVh5Dt, kVh5Iters, kVh5SolveIters);
        const vehicle::Vehicle b =
            vehicle::RunVehicleLockstep(cfg, init, stream, kVh5Ticks, kVh5Dt, kVh5Iters, kVh5SolveIters);
        check(vehicleEqual(a, b), "VH5 determinism: two full lockstep runs BYTE-IDENTICAL");
    }

    if (g_fail == 0) std::printf("vehicle_test: ALL PASS\n");
    else std::printf("vehicle_test: %d FAILURE(S)\n", g_fail);
    return g_fail ? 1 : 0;
}
