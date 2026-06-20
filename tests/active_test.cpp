// Slice AC1 — Deterministic Active Ragdoll / Physical-Animation Blending: the ANGULAR POSE-DRIVE primitive
// integer core (engine/sim/active.h) the GPU shaders/active_drive_solve.comp.hlsl copies VERBATIM + proves
// bit-identical. Pure CPU (header-only, no device, no backend symbols). Namespace hf::sim::active.
//
// What this test PINS (the contracts the GPU active_drive_solve.comp + the GPU==CPU proof build on):
//   * SolveAngularDrive: a single joint driven with stiffness kOne SNAPS qrel = qA⁻¹·qB to qTarget (within
//     an LSB band); stiffness 0 leaves qrel UNCHANGED; a pinned body (invMass 0) is NOT rotated; the
//     inverse-mass split rotates the lighter (higher-invMass) body MORE; out-of-range / both-pinned skip.
//   * DriveAngleCos: kOne when qrel == qTarget; below kOne off-target.
//   * StepDriveWorld: the 3-link chain drives to + HOLDS the L-pose (DriveAngleCos near kOne); the
//     stiffness-0 control hangs straight down (no drive); two runs byte-identical.
//
// HONEST CAVEAT (the JT2/VH1 caveat shape): the drive is a stiffness-scaled NLERP toward target (a soft
// angular constraint), NOT analytic motor mechanics; the held angle is a deterministic-but-nonzero
// Gauss-Seidel residual (more iters -> tighter, NOT exact). The headline is DETERMINISM + cross-platform
// bit-identity, NOT analytic torque mechanics.
//
// Pure C++ (hf_core), ASan-eligible like the other sim-math tests.
#include "sim/active.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace active = hf::sim::active;
namespace joint = hf::sim::joint;
namespace fpx = hf::sim::fpx;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// abs of a Q16.16 fx.
static active::fx fxabs(active::fx v) { return v < 0 ? -v : v; }

// Build a dynamic FxBody at (x,y,z) in Q16.16 with invMass `im` + a given orientation.
static fpx::FxBody body(int x, int y, int z, active::fx im, const fpx::FxQuat& q) {
    fpx::FxBody b;
    b.pos = fpx::FxVec3{(active::fx)(x * (int)active::kOne), (active::fx)(y * (int)active::kOne),
                        (active::fx)(z * (int)active::kOne)};
    b.vel = fpx::FxVec3{0, 0, 0};
    b.invMass = im;
    b.flags = (im == 0) ? 0u : fpx::kFlagDynamic;
    b.radius = 0;
    b.orient = q;
    b.angVel = fpx::FxVec3{0, 0, 0};
    return b;
}
static fpx::FxQuat ident() { return fpx::FxQuat{0, 0, 0, active::kOne}; }

// A drive record (bodyA -> bodyB) toward qTarget at `stiffness`.
static active::FxAngularDrive drive(uint32_t a, uint32_t b, const fpx::FxQuat& qTarget,
                                    active::fx stiffness) {
    active::FxAngularDrive d;
    d.bodyA = a; d.bodyB = b; d.qTarget = qTarget; d.stiffness = stiffness;
    return d;
}

int main() {
    HF_TEST_MAIN_INIT();

    // The L-pose target: a 90-degree relative rotation about Z. cos(45) = sin(45) = 0.70710678 -> Q16.16
    // ~46341. The drive bends a joint to this relative orientation (qA⁻¹·qB -> qZ90).
    const active::fx kSqrtHalf = (active::fx)(46341);   // 0.70710678 * 65536 rounded
    const fpx::FxQuat qZ90{0, 0, kSqrtHalf, kSqrtHalf};

    // The std430 stride contract (the GPU mirror packs the 28-byte record into a 16-byte-aligned 32-byte
    // stride). The C++ FxAngularDrive is the 28-byte logical record (no GPU-only pad).
    check(sizeof(active::FxAngularDrive) == 28, "FxAngularDrive logical layout (7 x int32 = 28 bytes)");

    // ================= SolveAngularDrive: stiffness kOne SNAPS qrel to qTarget (within a band) ===========
    {
        // A pinned frame body (A, identity) + a dynamic body (B, identity). The drive targets qZ90 at full
        // stiffness; one pass should drive qrel = qA⁻¹·qB ~ qZ90 (the dynamic body takes the full rotation).
        active::FxWorld w;
        w.gravity = fpx::FxVec3{0, 0, 0};
        w.groundY = (active::fx)(-1000 * (int)active::kOne);
        w.bodies = {body(0, 0, 0, 0, ident()), body(0, 0, 0, active::kOne, ident())};   // A pinned, B dynamic
        std::vector<active::FxAngularDrive> drives = {drive(0, 1, qZ90, active::kOne)};
        active::SolveAngularDrive(w, drives[0]);
        // qrel = qA⁻¹·qB ; A is identity so qrel == qB. After a full-stiffness drive B.orient ~ qZ90.
        const fpx::FxQuat qrel = fpx::FxQuatMul(joint::QConj(w.bodies[0].orient), w.bodies[1].orient);
        const active::fx band = active::kOne / 64;   // ~0.0156 LSB band (the integer normalize residual)
        check(fxabs(qrel.z - qZ90.z) < band && fxabs(qrel.w - qZ90.w) < band &&
              fxabs(qrel.x) < band && fxabs(qrel.y) < band,
              "SolveAngularDrive stiffness kOne: qrel snaps to qTarget (within LSB band)");
        // DriveAngleCos ~ kOne after the snap (qrel reached the target).
        const active::fx c = active::DriveAngleCos(w, drives[0]);
        check(c > active::kOne - band, "SolveAngularDrive stiffness kOne: DriveAngleCos ~ kOne (on target)");
    }

    // ================= SolveAngularDrive: stiffness 0 leaves qrel UNCHANGED =================
    {
        active::FxWorld w;
        w.gravity = fpx::FxVec3{0, 0, 0};
        w.groundY = (active::fx)(-1000 * (int)active::kOne);
        w.bodies = {body(0, 0, 0, 0, ident()), body(0, 0, 0, active::kOne, ident())};
        const fpx::FxQuat qB0 = w.bodies[1].orient;
        active::SolveAngularDrive(w, drive(0, 1, qZ90, /*stiffness=*/0));
        check(w.bodies[1].orient.x == qB0.x && w.bodies[1].orient.y == qB0.y &&
              w.bodies[1].orient.z == qB0.z && w.bodies[1].orient.w == qB0.w,
              "SolveAngularDrive stiffness 0: qrel unchanged (no drive)");
    }

    // ================= SolveAngularDrive: a pinned body is NOT rotated =================
    {
        // B pinned (invMass 0), A dynamic. The drive must rotate ONLY A (B's share is 0).
        active::FxWorld w;
        w.gravity = fpx::FxVec3{0, 0, 0};
        w.groundY = (active::fx)(-1000 * (int)active::kOne);
        w.bodies = {body(0, 0, 0, active::kOne, ident()), body(0, 0, 0, 0, ident())};   // A dynamic, B pinned
        const fpx::FxQuat qB0 = w.bodies[1].orient;
        active::SolveAngularDrive(w, drive(0, 1, qZ90, active::kOne));
        check(w.bodies[1].orient.x == qB0.x && w.bodies[1].orient.y == qB0.y &&
              w.bodies[1].orient.z == qB0.z && w.bodies[1].orient.w == qB0.w,
              "SolveAngularDrive: the pinned body (invMass 0) is NOT rotated");
        // A moved (it took the whole correction).
        check(!(w.bodies[0].orient.x == 0 && w.bodies[0].orient.y == 0 && w.bodies[0].orient.z == 0 &&
                w.bodies[0].orient.w == active::kOne),
              "SolveAngularDrive: the dynamic body took the rotation when its partner is pinned");
    }

    // ================= SolveAngularDrive: the inverse-mass split rotates the lighter body MORE ===========
    {
        // A heavy (invMass kOne/4) + B light (invMass kOne). The lighter body (higher invMass) should rotate
        // MORE (its share wB = invMass/wsum is larger). Compare how far each moved from identity (the .z
        // magnitude after the drive toward qZ90, which is a +Z rotation).
        active::FxWorld w;
        w.gravity = fpx::FxVec3{0, 0, 0};
        w.groundY = (active::fx)(-1000 * (int)active::kOne);
        w.bodies = {body(0, 0, 0, active::kOne / 4, ident()), body(0, 0, 0, active::kOne, ident())};
        active::SolveAngularDrive(w, drive(0, 1, qZ90, active::kOne));
        // A drives toward QConj(qZ90) (qAtarget = qB·conj(qrelDriven)) -> A rotates -Z; B rotates +Z. The
        // LIGHTER body B (invMass kOne) moves MORE than the heavier A (invMass kOne/4): |B.z| > |A.z|.
        check(fxabs(w.bodies[1].orient.z) > fxabs(w.bodies[0].orient.z),
              "SolveAngularDrive inverse-mass split: the lighter body (higher invMass) rotates more");
    }

    // ================= SolveAngularDrive: out-of-range / both-pinned -> no-op =================
    {
        active::FxWorld w;
        w.gravity = fpx::FxVec3{0, 0, 0};
        w.groundY = (active::fx)(-1000 * (int)active::kOne);
        w.bodies = {body(0, 0, 0, active::kOne, ident()), body(0, 0, 0, active::kOne, ident())};
        const std::vector<fpx::FxBody> before = w.bodies;
        active::SolveAngularDrive(w, drive(0, 99u, qZ90, active::kOne));   // out of range
        check(std::memcmp(w.bodies.data(), before.data(), before.size() * sizeof(fpx::FxBody)) == 0,
              "SolveAngularDrive out-of-range index: no-op");
        // both pinned.
        active::FxWorld w2;
        w2.gravity = fpx::FxVec3{0, 0, 0};
        w2.groundY = (active::fx)(-1000 * (int)active::kOne);
        w2.bodies = {body(0, 0, 0, 0, ident()), body(0, 0, 0, 0, ident())};
        const std::vector<fpx::FxBody> before2 = w2.bodies;
        active::SolveAngularDrive(w2, drive(0, 1, qZ90, active::kOne));
        check(std::memcmp(w2.bodies.data(), before2.data(), before2.size() * sizeof(fpx::FxBody)) == 0,
              "SolveAngularDrive both pinned: no-op (wsum 0 skip)");
    }

    // ================= DriveAngleCos: kOne on-target, below kOne off-target =================
    {
        active::FxWorld w;
        w.gravity = fpx::FxVec3{0, 0, 0};
        w.groundY = (active::fx)(-1000 * (int)active::kOne);
        // qrel == identity, target == identity -> DriveAngleCos == kOne (on target).
        w.bodies = {body(0, 0, 0, 0, ident()), body(0, 0, 0, active::kOne, ident())};
        check(active::DriveAngleCos(w, drive(0, 1, ident(), active::kOne)) == active::kOne,
              "DriveAngleCos: on-target (qrel == qTarget) == kOne");
        // qrel == identity, target == qZ90 -> DriveAngleCos == cos(45) ~ kSqrtHalf < kOne (off target).
        const active::fx c = active::DriveAngleCos(w, drive(0, 1, qZ90, active::kOne));
        check(c < active::kOne - active::kOne / 64,
              "DriveAngleCos: off-target (qrel != qTarget) < kOne");
    }

    // ================= StepDriveWorld: a 3-link chain drives to + HOLDS the L-pose =================
    // The AC1 SCENE: a pinned invMass-0 root + 3 dynamic ball-jointed bodies, with an FxAngularDrive on each
    // joint whose qTarget bends the chain into an L-shape (a 90-degree relative rotation about Z on each
    // joint). Settle K StepDriveWorld steps -> the chain is DRIVEN to + HOLDS the L-pose against gravity (vs
    // a stiffness-0 control that hangs straight down).
    const active::fx kGravY = (active::fx)(-9.8 * (double)active::kOne + (-9.8 < 0 ? -0.5 : 0.5));   // round
    const active::fx kDt = active::kOne / 60;
    const int kBodyCount = 4;          // 1 pinned root + 3 dynamic links
    const int kSteps = 400;            // settle the driven pose
    const int kIters = 24;             // Gauss-Seidel passes per step
    const active::fx kAnchor = active::kOne / 2;   // 0.5 link-end anchor offset along the chain

    auto buildScene = [&](active::fx driveStiff,
                          std::vector<active::FxJoint>& joints,
                          std::vector<active::FxAngularDrive>& drives) {
        active::FxWorld world;
        world.gravity = fpx::FxVec3{0, kGravY, 0};
        world.groundY = (active::fx)(-1000 * (int)active::kOne);   // far below -> focus the driven pose
        // body 0 = pinned root at (0,6); bodies 1..3 dynamic, hung straight down (each 1.0 below the prev).
        world.bodies.push_back(body(0, 6, 0, 0, ident()));
        for (int i = 1; i < kBodyCount; ++i)
            world.bodies.push_back(body(0, 6 - i, 0, active::kOne, ident()));
        // a ball joint per edge: parent's lower end (-Y anchor) pinned to the child's upper end (+Y anchor).
        joints.clear();
        for (uint32_t k = 0; k + 1 < (uint32_t)kBodyCount; ++k) {
            active::FxJoint j;
            j.bodyA = k; j.bodyB = k + 1;
            j.anchorA = fpx::FxVec3{0, -kAnchor, 0};
            j.anchorB = fpx::FxVec3{0, kAnchor, 0};
            j.kind = joint::kJointBall;
            joints.push_back(j);
        }
        // a drive per edge: bend each joint to the L-pose (a +90 deg relative rotation about Z).
        drives.clear();
        for (uint32_t k = 0; k + 1 < (uint32_t)kBodyCount; ++k)
            drives.push_back(drive(k, k + 1, qZ90, driveStiff));
        return world;
    };

    std::vector<active::FxJoint> joints;
    std::vector<active::FxAngularDrive> drives;
    const std::vector<active::FxAngularLimit> noLimits;
    const active::fx kDriveStiff = active::kOne / 8;   // soft (0.125) per-iteration drive -> holds the pose

    {
        active::FxWorld world = buildScene(kDriveStiff, joints, drives);
        active::StepDriveWorldSteps(world, joints, noLimits, drives, kDt, kIters, kSteps);
        // The pinned root NEVER moved.
        check(world.bodies[0].pos.x == 0 && world.bodies[0].pos.y == (active::fx)(6 * (int)active::kOne),
              "StepDriveWorld: the pinned root holds exactly");
        // Each drive HELD its target: DriveAngleCos near kOne within a band (the Gauss-Seidel residual).
        const active::fx kHoldBand = active::kOne / 8;   // within ~0.125 of cos==kOne (a held drive)
        int64_t sumCos = 0;
        bool allHeld = true;
        for (const active::FxAngularDrive& d : drives) {
            const active::fx c = active::DriveAngleCos(world, d);
            sumCos += (int64_t)c;
            if (c < active::kOne - kHoldBand) allHeld = false;
        }
        check(allHeld, "StepDriveWorld: every drive HELD its target (DriveAngleCos near kOne)");
        // The chain BENT away from straight: the leaf link's X drifted from the straight-hang X (0).
        check(fxabs(world.bodies[kBodyCount - 1].pos.x) > active::kOne / 2,
              "StepDriveWorld: the chain bent to the L-pose (leaf X drifted from straight)");
        (void)sumCos;
    }

    // ================= StepDriveWorld: the stiffness-0 control hangs STRAIGHT down =================
    {
        std::vector<active::FxJoint> cJoints;
        std::vector<active::FxAngularDrive> cDrives;
        active::FxWorld ctrl = buildScene(/*driveStiff=*/0, cJoints, cDrives);   // NO drive
        active::StepDriveWorldSteps(ctrl, cJoints, noLimits, cDrives, kDt, kIters, kSteps);
        // With no drive the chain hangs straight down: the leaf link stays ~directly below the root (X ~ 0).
        check(fxabs(ctrl.bodies[kBodyCount - 1].pos.x) < active::kOne / 4,
              "StepDriveWorld control: stiffness-0 hangs straight (no drive -> leaf below root)");
        // And the leaf dropped below the root (it fell under gravity, held only by the ball joints).
        check(ctrl.bodies[kBodyCount - 1].pos.y < ctrl.bodies[0].pos.y,
              "StepDriveWorld control: the chain hangs below the root");
    }

    // ================= StepDriveWorld: two runs byte-identical (determinism) =================
    {
        std::vector<active::FxJoint> jA, jB;
        std::vector<active::FxAngularDrive> dA, dB;
        active::FxWorld a = buildScene(kDriveStiff, jA, dA);
        active::FxWorld b = buildScene(kDriveStiff, jB, dB);
        active::StepDriveWorldSteps(a, jA, noLimits, dA, kDt, kIters, 200);
        active::StepDriveWorldSteps(b, jB, noLimits, dB, kDt, kIters, 200);
        const bool same = a.bodies.size() == b.bodies.size() &&
                          std::memcmp(a.bodies.data(), b.bodies.data(),
                                      a.bodies.size() * sizeof(fpx::FxBody)) == 0;
        check(same, "StepDriveWorld determinism: two runs BYTE-IDENTICAL");
    }

    if (g_fail == 0) std::printf("active_test: ALL PASS\n");
    else std::printf("active_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
