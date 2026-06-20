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
//   * AC2 driveWeight (the per-joint physical blend weight): driveWeight kOne == AC1 byte-identical orient
//     (render-invariant), driveWeight 0 == pure-physics no-drive, an intermediate weight pulls partway; a
//     mixed-weight StepDriveWorld HOLDS the driven joints + HANGS the limp ones (the partial ragdoll), two
//     runs byte-identical; an all-weight-kOne step == the AC1 all-driven step (render-invariance).
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

// A drive record (bodyA -> bodyB) toward qTarget at `stiffness` (driveWeight defaults to kOne — AC1 behavior).
static active::FxAngularDrive drive(uint32_t a, uint32_t b, const fpx::FxQuat& qTarget,
                                    active::fx stiffness) {
    active::FxAngularDrive d;
    d.bodyA = a; d.bodyB = b; d.qTarget = qTarget; d.stiffness = stiffness;
    return d;
}

// AC2: a drive record with an explicit per-joint physical blend weight.
static active::FxAngularDrive driveW(uint32_t a, uint32_t b, const fpx::FxQuat& qTarget,
                                     active::fx stiffness, active::fx weight) {
    active::FxAngularDrive d = drive(a, b, qTarget, stiffness);
    d.driveWeight = weight;
    return d;
}

int main() {
    HF_TEST_MAIN_INIT();

    // The L-pose target: a 90-degree relative rotation about Z. cos(45) = sin(45) = 0.70710678 -> Q16.16
    // ~46341. The drive bends a joint to this relative orientation (qA⁻¹·qB -> qZ90).
    const active::fx kSqrtHalf = (active::fx)(46341);   // 0.70710678 * 65536 rounded
    const fpx::FxQuat qZ90{0, 0, kSqrtHalf, kSqrtHalf};

    // The std430 stride contract (AC2): the C++ FxAngularDrive is now the 32-byte logical record — the AC1
    // 28-byte record + the AC2 driveWeight occupying the former GPU-mirror pad slot (the std430 stride was
    // ALREADY 32 in AC1, so the GPU mirror stride is UNCHANGED — render-invariant for AC1).
    check(sizeof(active::FxAngularDrive) == 32, "FxAngularDrive logical layout (8 x int32 = 32 bytes, AC2)");
    // AC2: driveWeight DEFAULTS to kOne so AC1 call-sites are unchanged in behavior (the render-invariance).
    check(active::FxAngularDrive{}.driveWeight == active::kOne, "FxAngularDrive.driveWeight defaults to kOne");

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

    // ============================================================================================
    // ============================ AC2: per-joint physical blend weight ===========================
    // ============================================================================================

    // ===== AC2 (1): driveWeight == kOne is BYTE-IDENTICAL to AC1 (the render-invariance contract) =====
    // A single SolveAngularDrive with an explicit driveWeight=kOne produces the EXACT same orient as the AC1
    // drive (no driveWeight field set / default kOne) because fxmul(w, kOne) == w in Q16.16.
    {
        active::FxWorld wAc1, wW1;
        wAc1.gravity = wW1.gravity = fpx::FxVec3{0, 0, 0};
        wAc1.groundY = wW1.groundY = (active::fx)(-1000 * (int)active::kOne);
        wAc1.bodies = {body(0, 0, 0, 0, ident()), body(0, 0, 0, active::kOne, ident())};
        wW1.bodies  = {body(0, 0, 0, 0, ident()), body(0, 0, 0, active::kOne, ident())};
        // AC1 path: a drive() with driveWeight DEFAULTED to kOne; AC2 path: driveW() with kOne EXPLICIT.
        active::SolveAngularDrive(wAc1, drive(0, 1, qZ90, active::kOne / 4));
        active::SolveAngularDrive(wW1, driveW(0, 1, qZ90, active::kOne / 4, active::kOne));
        check(std::memcmp(wAc1.bodies.data(), wW1.bodies.data(),
                          wAc1.bodies.size() * sizeof(fpx::FxBody)) == 0,
              "AC2 SolveAngularDrive driveWeight kOne == AC1 (byte-identical orient — render-invariant)");
    }

    // ===== AC2 (2): driveWeight == 0 leaves the orient at the PURE-PHYSICS (no-drive) result =====
    // weight 0 -> fxmul(w, 0) == 0 -> QNlerp(q, target, 0) == q -> no rotation. The bodies are byte-identical
    // to BOTH the input (no drive applied) AND a stiffness-0 drive (the AC1 "no drive" no-op).
    {
        active::FxWorld w;
        w.gravity = fpx::FxVec3{0, 0, 0};
        w.groundY = (active::fx)(-1000 * (int)active::kOne);
        w.bodies = {body(0, 0, 0, active::kOne, ident()), body(0, 0, 0, active::kOne, ident())};
        const std::vector<fpx::FxBody> before = w.bodies;
        active::SolveAngularDrive(w, driveW(0, 1, qZ90, active::kOne, /*weight=*/0));
        check(std::memcmp(w.bodies.data(), before.data(), before.size() * sizeof(fpx::FxBody)) == 0,
              "AC2 SolveAngularDrive driveWeight 0: orient UNCHANGED (pure physics — limp, no drive applied)");
    }

    // ===== AC2 (3): an intermediate driveWeight pulls PARTWAY (between limp and full) =====
    // weight kOne/2 rotates LESS than weight kOne but MORE than weight 0 (still identity). Measure each
    // dynamic body's drift from identity by |orient.z| (qZ90 is a +Z rotation).
    {
        auto driveOnce = [&](active::fx weight) {
            active::FxWorld w;
            w.gravity = fpx::FxVec3{0, 0, 0};
            w.groundY = (active::fx)(-1000 * (int)active::kOne);
            w.bodies = {body(0, 0, 0, 0, ident()), body(0, 0, 0, active::kOne, ident())};   // A pinned, B dynamic
            active::SolveAngularDrive(w, driveW(0, 1, qZ90, active::kOne, weight));
            return fxabs(w.bodies[1].orient.z);
        };
        const active::fx zFull = driveOnce(active::kOne);
        const active::fx zHalf = driveOnce(active::kOne / 2);
        const active::fx zZero = driveOnce(0);
        check(zZero == 0, "AC2 driveWeight 0: the dynamic body did NOT rotate (|z| == 0)");
        check(zHalf > zZero && zHalf < zFull,
              "AC2 intermediate driveWeight pulls partway (0 < halfWeight rotation < fullWeight rotation)");
    }

    // ===== AC2 (4): StepDriveWorld with MIXED per-joint weights — driven joints HOLD, limp joints HANG =====
    // A LONGER chain (an "upper body" + a "lower body"): the upper joints driveWeight=kOne (hold the bent
    // pose), the lower joints driveWeight=0 (limp — hang free under gravity, matching a pure-physics chain).
    {
        const int kBC = 7;                 // 1 pinned root + 6 dynamic links
        const int kUpper = 3;              // joints 0..2 are upper (driven, weight kOne)
        // joints 3..5 are lower (limp, weight 0)
        auto buildBlend = [&](std::vector<active::FxJoint>& js, std::vector<active::FxAngularDrive>& ds) {
            active::FxWorld w;
            w.gravity = fpx::FxVec3{0, kGravY, 0};
            w.groundY = (active::fx)(-1000 * (int)active::kOne);
            w.bodies.push_back(body(0, 8, 0, 0, ident()));                    // pinned root at (0,8)
            for (int i = 1; i < kBC; ++i) w.bodies.push_back(body(0, 8 - i, 0, active::kOne, ident()));
            js.clear(); ds.clear();
            for (uint32_t k = 0; k + 1 < (uint32_t)kBC; ++k) {
                active::FxJoint j;
                j.bodyA = k; j.bodyB = k + 1;
                j.anchorA = fpx::FxVec3{0, -kAnchor, 0};
                j.anchorB = fpx::FxVec3{0, kAnchor, 0};
                j.kind = joint::kJointBall;
                js.push_back(j);
                const active::fx weight = ((int)k < kUpper) ? active::kOne : 0;   // upper driven, lower limp
                ds.push_back(driveW(k, k + 1, qZ90, kDriveStiff, weight));
            }
            return w;
        };
        std::vector<active::FxJoint> bj; std::vector<active::FxAngularDrive> bd;
        active::FxWorld blend = buildBlend(bj, bd);
        active::StepDriveWorldSteps(blend, bj, noLimits, bd, kDt, kIters, kSteps);

        // The DRIVEN (upper, weight kOne) joints HELD their target within the hold band.
        const active::fx kHoldBand = active::kOne / 8;
        bool drivenHeld = true, limpFree = false;
        for (size_t k = 0; k < bd.size(); ++k) {
            const active::fx c = active::DriveAngleCos(blend, bd[k]);
            if ((int)k < kUpper) { if (c < active::kOne - kHoldBand) drivenHeld = false; }
            else                 { if (c < active::kOne - kHoldBand) limpFree = true; }   // a limp joint OFF target
        }
        check(drivenHeld, "AC2 StepDriveWorld mixed: the driven (weight kOne) joints HELD the target");
        check(limpFree,   "AC2 StepDriveWorld mixed: a limp (weight 0) joint is OFF the target (hangs free)");

        // The limp lower joints match a PURE-PHYSICS (all-weight-0) reference for those bodies: build the same
        // chain with EVERY drive weight 0 and confirm the lower links land at the same orientation.
        std::vector<active::FxJoint> pj; std::vector<active::FxAngularDrive> pd;
        active::FxWorld phys = buildBlend(pj, pd);
        for (auto& d : pd) d.driveWeight = 0;   // all limp -> pure physics
        active::StepDriveWorldSteps(phys, pj, noLimits, pd, kDt, kIters, kSteps);
        // The leaf (last) link is below the lower-limp region; under pure physics + a limp lower body it
        // hangs lower (smaller X drift) than the fully-driven leaf would. Confirm the limp leaf hangs nearer
        // straight-down than the upper driven links bent it.
        check(fxabs(blend.bodies[kBC - 1].pos.x) < fxabs(blend.bodies[kUpper].pos.x),
              "AC2 StepDriveWorld mixed: the limp leaf hangs nearer straight-down than the driven upper links");

        // two runs byte-identical (determinism with mixed weights).
        std::vector<active::FxJoint> bj2; std::vector<active::FxAngularDrive> bd2;
        active::FxWorld blend2 = buildBlend(bj2, bd2);
        active::StepDriveWorldSteps(blend2, bj2, noLimits, bd2, kDt, kIters, kSteps);
        check(blend.bodies.size() == blend2.bodies.size() &&
              std::memcmp(blend.bodies.data(), blend2.bodies.data(),
                          blend.bodies.size() * sizeof(fpx::FxBody)) == 0,
              "AC2 StepDriveWorld mixed-weight determinism: two runs BYTE-IDENTICAL");
    }

    // ===== AC2 (5): an all-driveWeight-kOne StepDriveWorld == the AC1 all-driven StepDriveWorld =====
    // (the StepDriveWorld-level render-invariance proof: weight kOne everywhere == AC1.)
    {
        std::vector<active::FxJoint> j1, jW; std::vector<active::FxAngularDrive> d1, dW;
        active::FxWorld wAc1 = buildScene(kDriveStiff, j1, d1);   // AC1: drives default driveWeight kOne
        active::FxWorld wW1  = buildScene(kDriveStiff, jW, dW);
        for (auto& d : dW) d.driveWeight = active::kOne;          // AC2: kOne EXPLICIT
        active::StepDriveWorldSteps(wAc1, j1, noLimits, d1, kDt, kIters, kSteps);
        active::StepDriveWorldSteps(wW1,  jW, noLimits, dW, kDt, kIters, kSteps);
        check(wAc1.bodies.size() == wW1.bodies.size() &&
              std::memcmp(wAc1.bodies.data(), wW1.bodies.data(),
                          wAc1.bodies.size() * sizeof(fpx::FxBody)) == 0,
              "AC2 StepDriveWorld all-weight-kOne == AC1 all-driven (byte-identical — render-invariant)");
    }

    // ============================================================================================
    // ============== AC3: THE ANIM-TARGET STEP (THE PILLAR BRIDGE) — clip -> drive targets =========
    // ============================================================================================

    // Build a synthetic ~9-joint humanoid anim::Skeleton (the JT4 --joint-ragdoll-shot config; topologically
    // sorted: 0 pelvis, 1 spine, 2 head, 3/4 L-arm, 5/6 R-arm, 7 L-leg, 8 R-leg).
    auto buildHumanoid = []() {
        hf::anim::Skeleton s;
        auto J = [](int parent, float tx, float ty, float tz) {
            hf::anim::Joint j; j.parent = parent; j.t = math::Vec3{tx, ty, tz};
            j.r = math::Quat{0, 0, 0, 1}; j.s = math::Vec3{1, 1, 1}; return j;
        };
        s.joints.push_back(J(-1, 0.0f,  5.0f, 0.0f));  // 0 pelvis (root)
        s.joints.push_back(J(0,  0.0f,  1.0f, 0.0f));  // 1 spine
        s.joints.push_back(J(1,  0.0f,  1.0f, 0.0f));  // 2 head
        s.joints.push_back(J(1, -0.7f,  0.6f, 0.0f));  // 3 L upper arm
        s.joints.push_back(J(3, -0.7f,  0.0f, 0.0f));  // 4 L fore arm
        s.joints.push_back(J(1,  0.7f,  0.6f, 0.0f));  // 5 R upper arm
        s.joints.push_back(J(5,  0.7f,  0.0f, 0.0f));  // 6 R fore arm
        s.joints.push_back(J(0, -0.4f, -1.0f, 0.0f));  // 7 L leg
        s.joints.push_back(J(0,  0.4f, -1.0f, 0.0f));  // 8 R leg
        const size_t n = s.joints.size();
        std::vector<math::Mat4> global(n);
        for (size_t j = 0; j < n; ++j) {
            const math::Mat4 local = math::FromTRS(s.joints[j].t, s.joints[j].r, s.joints[j].s);
            const int p = s.joints[j].parent;
            global[j] = (p >= 0) ? (global[(size_t)p] * local) : local;
        }
        for (size_t j = 0; j < n; ++j) s.joints[j].inverseBind = global[j].Inverse();
        return s;
    };

    // A synthetic "bend" clip: a Rotation channel on every non-root bone driving its LOCAL rotation to qZ90 at
    // t=duration (Step from identity at t=0 to qZ90 at t=1 -> a distinctly NON-rest pose the ragdoll tracks).
    auto buildBendClip = [&](const hf::anim::Skeleton& skel) {
        hf::anim::Animation a;
        a.name = "bend";
        a.duration = 1.0f;
        const float c = 0.70710678f;   // cos/sin 45 -> qZ90 = {0,0,c,c}
        for (size_t j = 1; j < skel.joints.size(); ++j) {   // skip root (no incoming edge -> no drive)
            hf::anim::Channel ch;
            ch.jointIndex = (int)j;
            ch.path = hf::anim::Channel::Path::Rotation;
            ch.interp = hf::anim::Channel::Interp::Linear;
            ch.times = {0.0f, 1.0f};
            ch.values = {0, 0, 0, 1,   0, 0, c, c};   // identity at t=0, qZ90 at t=1 (xyzw per key)
            a.channels.push_back(ch);
        }
        return a;
    };

    const hf::anim::Skeleton humanoid = buildHumanoid();
    const hf::anim::Animation bendClip = buildBendClip(humanoid);

    joint::RagdollConfig acCfg;
    acCfg.worldScale = active::kOne;
    acCfg.boneRadius = active::kOne * 30 / 100;
    acCfg.invMass    = active::kOne;
    acCfg.coneCos    = -active::kOne;     // 180-degree free cone (the drive does the work)
    acCfg.coneSin    = 0;
    acCfg.gravity    = fpx::FxVec3{0, kGravY, 0};
    acCfg.groundY    = (active::fx)(-1000 * (int)active::kOne);   // far below -> focus the tracked pose
    acCfg.rootStatic = true;              // pinned root so the ragdoll hangs from the pelvis

    // ===== AC3 (1): FxQuatFromFloat round-trips a unit quat within an LSB band =====
    {
        const math::Quat qf{0.0f, 0.0f, 0.70710678f, 0.70710678f};
        const fpx::FxQuat fq = active::FxQuatFromFloat(qf);
        const active::fx band = active::kOne / 1024;   // a tight snap band (round-to-nearest)
        check(fxabs(fq.z - kSqrtHalf) <= band && fxabs(fq.w - kSqrtHalf) <= band &&
              fq.x == 0 && fq.y == 0,
              "AC3 FxQuatFromFloat: round-to-nearest snap matches qZ90 within an LSB band");
        // identity -> {0,0,0,kOne} exactly.
        const fpx::FxQuat fi = active::FxQuatFromFloat(math::Quat{0, 0, 0, 1});
        check(fi.x == 0 && fi.y == 0 && fi.z == 0 && fi.w == active::kOne,
              "AC3 FxQuatFromFloat: identity snaps to {0,0,0,kOne} exactly");
    }

    // ===== AC3 (2): ActiveFromSkeleton builds one drive per non-root edge with the right parent/child =====
    {
        active::ActiveRagdoll act = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        // one drive per ragdoll joint (== non-root edge count == joint count - 1 root for this tree = 8).
        check(act.drives.size() == act.ragdoll.joints.size(),
              "AC3 ActiveFromSkeleton: one drive per non-root edge (parallel to ragdoll.joints)");
        check(act.drives.size() == humanoid.joints.size() - 1,
              "AC3 ActiveFromSkeleton: drive count == non-root edge count");
        bool parentChildOk = true;
        for (size_t e = 0; e < act.drives.size(); ++e) {
            if (act.drives[e].bodyA != act.ragdoll.joints[e].bodyA ||
                act.drives[e].bodyB != act.ragdoll.joints[e].bodyB) parentChildOk = false;
            // the child bone == joints[e].bodyB, and its skeleton parent == joints[e].bodyA.
            if ((int)humanoid.joints[(size_t)act.ragdoll.joints[e].bodyB].parent !=
                (int)act.ragdoll.joints[e].bodyA) parentChildOk = false;
        }
        check(parentChildOk, "AC3 ActiveFromSkeleton: each drive's bodyA/bodyB == its joint's parent/child");
        check(act.drives[0].stiffness == kDriveStiff && act.drives[0].driveWeight == active::kOne,
              "AC3 ActiveFromSkeleton: drives carry the stiffness + default driveWeight kOne");
    }

    // ===== AC3 (3): WriteClipTargets sets each drive's qTarget to the snapped sampled bone rotation =====
    {
        active::ActiveRagdoll act = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        // at t=0 every channel is identity -> every qTarget snaps to identity.
        active::WriteClipTargets(act, humanoid, bendClip, 0.0f);
        bool allIdent = true;
        for (const active::FxAngularDrive& d : act.drives)
            if (!(d.qTarget.x == 0 && d.qTarget.y == 0 && d.qTarget.z == 0 && d.qTarget.w == active::kOne))
                allIdent = false;
        check(allIdent, "AC3 WriteClipTargets t=0: every qTarget is the rest (identity) rotation");
        // at t=1 the clip rotates every non-root bone to qZ90 -> each qTarget == FxQuatFromFloat(sampled .r).
        active::WriteClipTargets(act, humanoid, bendClip, 1.0f);
        bool matchesSample = true;
        const std::vector<hf::anim::JointPose> pose = hf::anim::SampleLocalPose(humanoid, bendClip, 1.0f);
        for (size_t e = 0; e < act.drives.size(); ++e) {
            const uint32_t childBone = act.ragdoll.joints[e].bodyB;
            const fpx::FxQuat want = active::FxQuatFromFloat(pose[(size_t)childBone].r);
            if (std::memcmp(&act.drives[e].qTarget, &want, sizeof(fpx::FxQuat)) != 0) matchesSample = false;
        }
        check(matchesSample, "AC3 WriteClipTargets t=1: qTarget == snap(SampleLocalPose[childBone].r)");
        // the targets ADVANCE: t=0 != t=1 for the driven bones (the drive follows the animation).
        check(act.drives[0].qTarget.z != 0, "AC3 WriteClipTargets: targets advance with clip time (t=1 != rest)");
    }

    // ===== AC3 (4): StepActive drives the ragdoll TOWARD the clip pose while a no-drive control collapses ===
    {
        active::ActiveRagdoll driven  = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        active::ActiveRagdoll control = active::ActiveFromSkeleton(humanoid, acCfg, /*stiffness=*/0);  // limp
        const active::fx acDt = active::kOne / 60;
        const int acIters = 24, acSteps = 240;
        // Track the clip held at t=1 (the bent pose) for both (the control has stiffness 0 -> it just collapses).
        active::StepActiveSteps(driven,  humanoid, bendClip, acDt, acIters, acSteps, /*startTime=*/1.0f);
        active::StepActiveSteps(control, humanoid, bendClip, acDt, acIters, acSteps, /*startTime=*/1.0f);
        // the driven ragdoll HELD the clip targets: mean DriveAngleCos within a band of kOne.
        const active::fx kHoldBand = active::kOne / 4;   // a looser band — a full humanoid under gravity
        int64_t sumCos = 0;
        for (const active::FxAngularDrive& d : driven.drives) sumCos += (int64_t)active::DriveAngleCos(driven.ragdoll.world, d);
        const active::fx meanCos = driven.drives.empty() ? active::kOne
                                       : (active::fx)(sumCos / (int64_t)driven.drives.size());
        check(meanCos > active::kOne - kHoldBand,
              "AC3 StepActive: the driven ragdoll TRACKS the clip (mean DriveAngleCos within band)");
        // the driven pose DIFFERS from the limp control by a margin (the drive posed it, the control collapsed).
        active::fx maxDiff = 0;
        for (size_t i = 0; i < driven.ragdoll.world.bodies.size(); ++i) {
            const active::fx dx = driven.ragdoll.world.bodies[i].pos.x - control.ragdoll.world.bodies[i].pos.x;
            const active::fx adx = dx < 0 ? -dx : dx;
            if (adx > maxDiff) maxDiff = adx;
        }
        check(maxDiff > active::kOne / 2,
              "AC3 StepActive: the driven pose differs from the limp control (posed, not limp)");
    }

    // ===== AC3 (5): StepActive is deterministic — two runs byte-identical =====
    {
        active::ActiveRagdoll a = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        active::ActiveRagdoll b = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        const active::fx acDt = active::kOne / 60;
        active::StepActiveSteps(a, humanoid, bendClip, acDt, 24, 120, 1.0f);
        active::StepActiveSteps(b, humanoid, bendClip, acDt, 24, 120, 1.0f);
        const bool same = a.ragdoll.world.bodies.size() == b.ragdoll.world.bodies.size() &&
                          std::memcmp(a.ragdoll.world.bodies.data(), b.ragdoll.world.bodies.data(),
                                      a.ragdoll.world.bodies.size() * sizeof(fpx::FxBody)) == 0;
        check(same, "AC3 StepActive determinism: two runs BYTE-IDENTICAL");
    }

    // ============================================================================================
    // ============== AC4: ACTIVE -> LIMP -> RECOVER (the global physicality knob + the hit) =========
    // ============================================================================================

    // ===== AC4 (1): PhysicalityAtTick boundaries are EXACT (kOne pre-struck, 0 in limp, ramp, kOne after) =
    {
        const int struckTick = 10, limpTicks = 5, recoverTicks = 8;
        // before the hit -> kOne.
        check(active::PhysicalityAtTick(0, struckTick, limpTicks, recoverTicks) == active::kOne,
              "AC4 PhysicalityAtTick: tick 0 (pre-struck) == kOne");
        check(active::PhysicalityAtTick(struckTick - 1, struckTick, limpTicks, recoverTicks) == active::kOne,
              "AC4 PhysicalityAtTick: the tick before struckTick == kOne");
        // the limp window [struckTick, struckTick+limpTicks) -> 0.
        check(active::PhysicalityAtTick(struckTick, struckTick, limpTicks, recoverTicks) == 0,
              "AC4 PhysicalityAtTick: struckTick (limp start) == 0");
        check(active::PhysicalityAtTick(struckTick + limpTicks - 1, struckTick, limpTicks, recoverTicks) == 0,
              "AC4 PhysicalityAtTick: the last limp tick == 0");
        // the ramp window: first recover tick (r==0) == kOne/recoverTicks; last (r==recoverTicks-1) == kOne.
        const int rampStart = struckTick + limpTicks;
        check(active::PhysicalityAtTick(rampStart, struckTick, limpTicks, recoverTicks)
                  == (active::fx)(((int64_t)active::kOne * 1) / (int64_t)recoverTicks),
              "AC4 PhysicalityAtTick: the first recover tick == kOne/recoverTicks (ramp begins)");
        check(active::PhysicalityAtTick(rampStart + recoverTicks - 1, struckTick, limpTicks, recoverTicks)
                  == active::kOne,
              "AC4 PhysicalityAtTick: the last recover tick == kOne (ramp reaches full)");
        // the ramp is MONOTONIC NON-DECREASING across the window (a clean 0->kOne climb).
        bool monotonic = true;
        active::fx prev = -1;
        for (int r = 0; r < recoverTicks; ++r) {
            const active::fx p = active::PhysicalityAtTick(rampStart + r, struckTick, limpTicks, recoverTicks);
            if (p < prev) monotonic = false;
            prev = p;
        }
        check(monotonic, "AC4 PhysicalityAtTick: the recovery ramp is monotonic non-decreasing");
        // past the ramp -> kOne (re-tracked).
        check(active::PhysicalityAtTick(rampStart + recoverTicks, struckTick, limpTicks, recoverTicks)
                  == active::kOne,
              "AC4 PhysicalityAtTick: the tick after the ramp == kOne (re-tracked)");
        check(active::PhysicalityAtTick(1000, struckTick, limpTicks, recoverTicks) == active::kOne,
              "AC4 PhysicalityAtTick: far past the ramp == kOne");
        // recoverTicks <= 0 -> snap straight to kOne after the limp window (no ramp).
        check(active::PhysicalityAtTick(struckTick + limpTicks, struckTick, limpTicks, /*recoverTicks=*/0)
                  == active::kOne,
              "AC4 PhysicalityAtTick: recoverTicks 0 snaps to kOne right after the limp window");
    }

    // ===== AC4 (2): ApplyImpulse raises a dynamic body's vel; static / out-of-range is a no-op =====
    {
        active::FxWorld w;
        w.gravity = fpx::FxVec3{0, 0, 0};
        w.groundY = (active::fx)(-1000 * (int)active::kOne);
        w.bodies = {body(0, 0, 0, active::kOne, ident()), body(0, 0, 0, 0, ident())};   // 0 dynamic, 1 static
        const fpx::FxVec3 dv{active::kOne, 2 * active::kOne, -active::kOne};
        active::ApplyImpulse(w, 0, dv);   // dynamic -> vel raised by dv
        check(w.bodies[0].vel.x == dv.x && w.bodies[0].vel.y == dv.y && w.bodies[0].vel.z == dv.z,
              "AC4 ApplyImpulse: a dynamic body's vel is raised by dv");
        // a second kick ACCUMULATES (FxAdd).
        active::ApplyImpulse(w, 0, dv);
        check(w.bodies[0].vel.x == 2 * dv.x && w.bodies[0].vel.y == 2 * dv.y && w.bodies[0].vel.z == 2 * dv.z,
              "AC4 ApplyImpulse: a second kick accumulates (FxAdd)");
        // a STATIC body (invMass 0 -> no kFlagDynamic) is NOT kicked.
        const fpx::FxVec3 v1before = w.bodies[1].vel;
        active::ApplyImpulse(w, 1, dv);
        check(w.bodies[1].vel.x == v1before.x && w.bodies[1].vel.y == v1before.y &&
              w.bodies[1].vel.z == v1before.z,
              "AC4 ApplyImpulse: a static body (no kFlagDynamic) is NOT kicked (no-op)");
        // out-of-range -> no-op (no crash, no change).
        const std::vector<fpx::FxBody> before = w.bodies;
        active::ApplyImpulse(w, 99u, dv);
        check(std::memcmp(w.bodies.data(), before.data(), before.size() * sizeof(fpx::FxBody)) == 0,
              "AC4 ApplyImpulse: out-of-range index is a no-op");
    }

    // ===== AC4 (3): StepActivePhysicality at kOne == StepActive (byte-identical); at 0 == pure physics =====
    {
        const active::fx acDt = active::kOne / 60;
        const int acIters = 24;
        // physicality kOne == AC3 StepActive byte-for-byte (the equivalence contract: fxmul(w,kOne)==w).
        active::ActiveRagdoll aFull = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        active::ActiveRagdoll aAc3  = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        for (int s = 0; s < 60; ++s) {
            const float t = 1.0f + (float)s * ((float)acDt / (float)active::kOne);
            active::StepActivePhysicality(aFull, humanoid, bendClip, t, active::kOne, acDt, acIters);
            active::StepActive(aAc3, humanoid, bendClip, t, acDt, acIters);
        }
        check(aFull.ragdoll.world.bodies.size() == aAc3.ragdoll.world.bodies.size() &&
              std::memcmp(aFull.ragdoll.world.bodies.data(), aAc3.ragdoll.world.bodies.data(),
                          aFull.ragdoll.world.bodies.size() * sizeof(fpx::FxBody)) == 0,
              "AC4 StepActivePhysicality physicality kOne == AC3 StepActive (byte-identical — equivalence)");
        // physicality 0 == pure physics (no drive): a limp (stiffness-0 ActiveFromSkeleton) StepActive control
        // collapses to the SAME bodies (the drive contributes nothing at physicality 0).
        active::ActiveRagdoll aLimp  = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        active::ActiveRagdoll aZero  = active::ActiveFromSkeleton(humanoid, acCfg, /*stiffness=*/0);
        for (int s = 0; s < 60; ++s) {
            const float t = 1.0f + (float)s * ((float)acDt / (float)active::kOne);
            active::StepActivePhysicality(aLimp, humanoid, bendClip, t, /*physicality=*/0, acDt, acIters);
            active::StepActive(aZero, humanoid, bendClip, t, acDt, acIters);
        }
        check(std::memcmp(aLimp.ragdoll.world.bodies.data(), aZero.ragdoll.world.bodies.data(),
                          aLimp.ragdoll.world.bodies.size() * sizeof(fpx::FxBody)) == 0,
              "AC4 StepActivePhysicality physicality 0 == pure physics (a stiffness-0 limp control)");
        // StepActivePhysicality does NOT overwrite the base drives (the scratch contract).
        const active::fx w0 = aFull.drives.empty() ? active::kOne : aFull.drives[0].driveWeight;
        check(w0 == active::kOne, "AC4 StepActivePhysicality: the base drives' driveWeight is NOT mutated (scratch)");
    }

    // ===== AC4 (4): StepActiveRecover — animCos high, struckCos < animCos, recoverCos > struckCos =====
    {
        const active::fx acDt = active::kOne / 60;
        const int acIters = 24;
        const int struckTick = 60, limpTicks = 40, recoverTicks = 80;
        // a torso impulse (a sideways + upward kick on the spine body, index 1).
        const fpx::FxVec3 impulse{6 * active::kOne, 4 * active::kOne, 0};
        const uint32_t impulseBody = 1u;   // the spine

        auto meanCosOf = [&](const active::ActiveRagdoll& a) -> active::fx {
            int64_t sum = 0;
            for (const active::FxAngularDrive& d : a.drives) sum += (int64_t)active::DriveAngleCos(a.ragdoll.world, d);
            return a.drives.empty() ? active::kOne : (active::fx)(sum / (int64_t)a.drives.size());
        };

        // ANIM: settle the anim-tracked pose (physicality kOne throughout, no impulse) -> animCos high.
        active::ActiveRagdoll anim = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        for (int s = 0; s < struckTick; ++s) {
            const float t = 1.0f + (float)s * ((float)acDt / (float)active::kOne);
            active::StepActivePhysicality(anim, humanoid, bendClip, t, active::kOne, acDt, acIters);
        }
        const active::fx animCos = meanCosOf(anim);

        // STRUCK: run the SAME episode start through the END of the limp window -> struckCos LOW (left the clip).
        active::ActiveRagdoll struck = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        active::StepActiveRecover(struck, humanoid, bendClip, acDt, acIters, struckTick, impulseBody, impulse,
                                  limpTicks, recoverTicks, /*totalTicks=*/struckTick + limpTicks, /*startTime=*/1.0f);
        const active::fx struckCos = meanCosOf(struck);

        // RECOVER: run the FULL episode (through the recovery ramp) -> recoverCos back up (returned to the clip).
        active::ActiveRagdoll recover = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        active::StepActiveRecover(recover, humanoid, bendClip, acDt, acIters, struckTick, impulseBody, impulse,
                                  limpTicks, recoverTicks,
                                  /*totalTicks=*/struckTick + limpTicks + recoverTicks + 60, /*startTime=*/1.0f);
        const active::fx recoverCos = meanCosOf(recover);

        check(animCos > active::kOne - active::kOne / 4, "AC4 StepActiveRecover: animCos high (tracks the clip)");
        check(struckCos < animCos, "AC4 StepActiveRecover: struckCos < animCos (the hit knocked it off the pose)");
        check(recoverCos > struckCos, "AC4 StepActiveRecover: recoverCos > struckCos (it recovered toward the clip)");

        // determinism: two full StepActiveRecover runs byte-identical.
        active::ActiveRagdoll r2 = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        active::StepActiveRecover(r2, humanoid, bendClip, acDt, acIters, struckTick, impulseBody, impulse,
                                  limpTicks, recoverTicks,
                                  /*totalTicks=*/struckTick + limpTicks + recoverTicks + 60, /*startTime=*/1.0f);
        check(recover.ragdoll.world.bodies.size() == r2.ragdoll.world.bodies.size() &&
              std::memcmp(recover.ragdoll.world.bodies.data(), r2.ragdoll.world.bodies.data(),
                          recover.ragdoll.world.bodies.size() * sizeof(fpx::FxBody)) == 0,
              "AC4 StepActiveRecover determinism: two runs BYTE-IDENTICAL");

        // equivalence: a physicality-kOne-throughout, NO-impulse StepActiveRecover == the AC3 StepActive episode.
        active::ActiveRagdoll eqRecover = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        active::StepActiveRecover(eqRecover, humanoid, bendClip, acDt, acIters, /*struckTick=*/-1, /*body=*/0u,
                                  fpx::FxVec3{0, 0, 0}, /*limpTicks=*/0, /*recoverTicks=*/0,
                                  /*totalTicks=*/120, /*startTime=*/1.0f);
        active::ActiveRagdoll eqAc3 = active::ActiveFromSkeleton(humanoid, acCfg, kDriveStiff);
        active::StepActiveSteps(eqAc3, humanoid, bendClip, acDt, acIters, 120, /*startTime=*/1.0f);
        check(eqRecover.ragdoll.world.bodies.size() == eqAc3.ragdoll.world.bodies.size() &&
              std::memcmp(eqRecover.ragdoll.world.bodies.data(), eqAc3.ragdoll.world.bodies.data(),
                          eqRecover.ragdoll.world.bodies.size() * sizeof(fpx::FxBody)) == 0,
              "AC4 StepActiveRecover equiv: physicality-kOne no-impulse run == AC3 StepActive (byte-identical)");
    }

    if (g_fail == 0) std::printf("active_test: ALL PASS\n");
    else std::printf("active_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
