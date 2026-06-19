// Slice JT1 — Deterministic Articulated-Body Ragdoll: the JOINT GRAPH + BALL-JOINT CONSTRAINT integer
// core (engine/sim/joint.h) the GPU shaders/joint_ball_solve.comp.hlsl copies VERBATIM + proves
// bit-identical. Pure CPU (header-only, no device, no backend symbols). Namespace hf::sim::joint.
//
// What this test PINS (the contracts the GPU joint_ball_solve.comp + the GPU==CPU proof build on):
//   * WorldAnchor: identity orient -> pos + anchor; a rotated orient rotates the local anchor by orient.
//   * SolveBallJoint: two equal-mass bodies with offset anchors -> both move HALF the gap toward
//     coincidence (one projection halves the gap by the inverse-mass split); a pinned body (invMass 0)
//     -> only the dynamic one moves; coincident anchors -> no-op; both pinned -> no-op.
//   * StepJointWorld: a 1-joint chain (pinned root + 1 dynamic) settles with the gap shrinking + the
//     dynamic body below the root; two runs byte-identical.
//
// Pure C++ (hf_core), ASan-eligible like the other sim-math tests.
#include "sim/joint.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace joint = hf::sim::joint;
namespace fpx = hf::sim::fpx;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Build a dynamic FxBody at (x,y,z) in Q16.16 with unit invMass + identity orient.
static fpx::FxBody dyn(int x, int y, int z) {
    fpx::FxBody b;
    b.pos = fpx::FxVec3{(joint::fx)(x * (int)joint::kOne), (joint::fx)(y * (int)joint::kOne),
                        (joint::fx)(z * (int)joint::kOne)};
    b.vel = fpx::FxVec3{0, 0, 0};
    b.invMass = joint::kOne;
    b.flags = fpx::kFlagDynamic;
    b.radius = 0;
    b.orient = fpx::FxQuat{0, 0, 0, joint::kOne};
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

int main() {
    HF_TEST_MAIN_INIT();

    // ================= WorldAnchor: identity orient -> pos + anchor =================
    {
        fpx::FxBody b = dyn(3, 5, 7);
        const joint::FxVec3 anchor{joint::kOne, 2 * (int)joint::kOne, 0};  // (1, 2, 0)
        const joint::FxVec3 w = joint::WorldAnchor(b, anchor);
        check(w.x == (joint::fx)(4 * (int)joint::kOne) && w.y == (joint::fx)(7 * (int)joint::kOne) &&
              w.z == (joint::fx)(7 * (int)joint::kOne),
              "WorldAnchor identity orient -> pos + anchor exactly");
        // Zero anchor -> the body centre.
        const joint::FxVec3 w0 = joint::WorldAnchor(b, joint::FxVec3{0, 0, 0});
        check(w0.x == b.pos.x && w0.y == b.pos.y && w0.z == b.pos.z,
              "WorldAnchor zero anchor -> pos exactly");
    }

    // ================= WorldAnchor: a rotated orient rotates the local anchor =================
    {
        // A 180-degree rotation about Y: quat (0, sin90, 0, cos90) = (0, 1, 0, 0) -> maps +X anchor to -X.
        fpx::FxBody b = dyn(0, 0, 0);
        b.orient = fpx::FxQuat{0, joint::kOne, 0, 0};   // (x,y,z,w) = (0,1,0,0), a 180° Y rotation
        const joint::FxVec3 anchor{joint::kOne, 0, 0};  // +X local
        const joint::FxVec3 w = joint::WorldAnchor(b, anchor);
        // FxRotate of (1,0,0) by a 180° Y rotation is (-1,0,0); pos is origin -> world anchor ~ (-1,0,0).
        check(w.x == -(joint::fx)joint::kOne && w.y == 0 && w.z == 0,
              "WorldAnchor 180-deg Y orient rotates +X anchor to -X");
        // Compare directly against FxRotate (the WorldAnchor definition is pos + FxRotate).
        const joint::FxVec3 r = fpx::FxRotate(b.orient, anchor);
        check(w.x == b.pos.x + r.x && w.y == b.pos.y + r.y && w.z == b.pos.z + r.z,
              "WorldAnchor == pos + FxRotate(orient, anchor)");
    }

    // ================= SolveBallJoint: two equal-mass bodies -> both move HALF the gap =================
    {
        // Body 0 at x=0, body 1 at x=4, anchors at each centre (anchor 0,0,0). The gap is 4 along +X; one
        // ball projection with equal inverse masses moves each body HALF the gap (2 units) toward the
        // midpoint -> both land at x=2 (coincident anchors). Hand-checked Q16.16.
        joint::FxWorld w;
        w.gravity = joint::FxVec3{0, 0, 0};
        w.groundY = (joint::fx)(-1000 * (int)joint::kOne);
        w.bodies = {dyn(0, 0, 0), dyn(4, 0, 0)};
        joint::FxJoint j;
        j.bodyA = 0; j.bodyB = 1;
        j.anchorA = joint::FxVec3{0, 0, 0};
        j.anchorB = joint::FxVec3{0, 0, 0};
        j.kind = joint::kJointBall;
        joint::SolveBallJoint(w, j);
        check(w.bodies[0].pos.x == (joint::fx)(2 * (int)joint::kOne) &&
              w.bodies[1].pos.x == (joint::fx)(2 * (int)joint::kOne),
              "SolveBallJoint equal mass: both move half the gap to coincidence (x=2)");
        check(w.bodies[0].pos.y == 0 && w.bodies[1].pos.y == 0,
              "SolveBallJoint: no off-axis motion for an on-axis gap");
    }

    // ================= SolveBallJoint: a pinned body -> only the dynamic one moves =================
    {
        // Body 0 PINNED at x=0, body 1 dynamic at x=4, anchors at centres. The pinned body holds; the
        // dynamic body takes the FULL correction -> moves all 4 units to x=0 (coincident).
        joint::FxWorld w;
        w.gravity = joint::FxVec3{0, 0, 0};
        w.groundY = (joint::fx)(-1000 * (int)joint::kOne);
        w.bodies = {pinned(0, 0, 0), dyn(4, 0, 0)};
        joint::FxJoint j;
        j.bodyA = 0; j.bodyB = 1;
        j.anchorA = joint::FxVec3{0, 0, 0};
        j.anchorB = joint::FxVec3{0, 0, 0};
        joint::SolveBallJoint(w, j);
        check(w.bodies[0].pos.x == 0, "SolveBallJoint pinned: the pinned body never moves");
        check(w.bodies[1].pos.x == 0, "SolveBallJoint pinned: the dynamic body takes the full correction");
    }

    // ================= SolveBallJoint: coincident anchors -> no-op =================
    {
        joint::FxWorld w;
        w.gravity = joint::FxVec3{0, 0, 0};
        w.groundY = (joint::fx)(-1000 * (int)joint::kOne);
        w.bodies = {dyn(2, 2, 2), dyn(2, 2, 2)};   // same position, anchors at centres -> coincident
        const joint::FxVec3 p0 = w.bodies[0].pos, p1 = w.bodies[1].pos;
        joint::FxJoint j;
        j.bodyA = 0; j.bodyB = 1;
        j.anchorA = joint::FxVec3{0, 0, 0};
        j.anchorB = joint::FxVec3{0, 0, 0};
        joint::SolveBallJoint(w, j);
        check(w.bodies[0].pos.x == p0.x && w.bodies[0].pos.y == p0.y && w.bodies[0].pos.z == p0.z &&
              w.bodies[1].pos.x == p1.x && w.bodies[1].pos.y == p1.y && w.bodies[1].pos.z == p1.z,
              "SolveBallJoint coincident anchors: no-op (len 0 skip)");
    }

    // ================= SolveBallJoint: both pinned -> no-op =================
    {
        joint::FxWorld w;
        w.gravity = joint::FxVec3{0, 0, 0};
        w.groundY = (joint::fx)(-1000 * (int)joint::kOne);
        w.bodies = {pinned(0, 0, 0), pinned(4, 0, 0)};
        const joint::FxVec3 p0 = w.bodies[0].pos, p1 = w.bodies[1].pos;
        joint::FxJoint j;
        j.bodyA = 0; j.bodyB = 1;
        j.anchorA = joint::FxVec3{0, 0, 0};
        j.anchorB = joint::FxVec3{0, 0, 0};
        joint::SolveBallJoint(w, j);
        check(w.bodies[0].pos.x == p0.x && w.bodies[1].pos.x == p1.x,
              "SolveBallJoint both pinned: no-op (wsum 0 skip)");
    }

    // ================= SolveBallJoint: out-of-range body index -> no-op =================
    {
        joint::FxWorld w;
        w.gravity = joint::FxVec3{0, 0, 0};
        w.groundY = (joint::fx)(-1000 * (int)joint::kOne);
        w.bodies = {dyn(0, 0, 0), dyn(4, 0, 0)};
        const joint::FxVec3 p0 = w.bodies[0].pos, p1 = w.bodies[1].pos;
        joint::FxJoint j;
        j.bodyA = 0; j.bodyB = 99;   // out of range
        joint::SolveBallJoint(w, j);
        check(w.bodies[0].pos.x == p0.x && w.bodies[1].pos.x == p1.x,
              "SolveBallJoint out-of-range index: no-op");
    }

    // ================= SolveBallJoint: off-centre anchors pull the anchor points together ============
    {
        // Body 0 at x=0 with a +X anchor offset of 1 (world anchor at x=1); body 1 at x=4 with a -X anchor
        // offset of 1 (world anchor at x=3). The anchor gap is 2 (from x=1 to x=3); equal masses -> each
        // body moves 1 unit toward the midpoint -> body0 to x=1, body1 to x=3, anchors meet at x=2.
        joint::FxWorld w;
        w.gravity = joint::FxVec3{0, 0, 0};
        w.groundY = (joint::fx)(-1000 * (int)joint::kOne);
        w.bodies = {dyn(0, 0, 0), dyn(4, 0, 0)};
        joint::FxJoint j;
        j.bodyA = 0; j.bodyB = 1;
        j.anchorA = joint::FxVec3{joint::kOne, 0, 0};    // +X offset on body0 -> world anchor x=1
        j.anchorB = joint::FxVec3{-(joint::fx)joint::kOne, 0, 0};  // -X offset on body1 -> world anchor x=3
        joint::SolveBallJoint(w, j);
        check(w.bodies[0].pos.x == (joint::fx)joint::kOne &&
              w.bodies[1].pos.x == (joint::fx)(3 * (int)joint::kOne),
              "SolveBallJoint off-centre anchors: centres move so the anchors meet at the midpoint");
        // After the move, the world anchors are coincident (gap 0).
        check(joint::AnchorGap(w, j) == 0, "SolveBallJoint off-centre: anchor gap closed to 0");
    }

    // ================= StepJointWorld: a 1-joint chain settles (gap shrinks, body hangs below root) ====
    {
        // A pinned root at (0, 10, 0) + a dynamic body 1 unit below at (0, 9, 0), ball-jointed centre-to-
        // centre. Under gravity the dynamic body falls; the ball joint pulls it back toward the root. The
        // anchor gap (centre-to-centre distance) settles to a small band, and the dynamic body hangs BELOW
        // the root (pos.y < root pos.y).
        joint::FxWorld w;
        const joint::fx gravY = (joint::fx)(-9.8 * (double)joint::kOne + (-9.8 < 0 ? -0.5 : 0.5));
        w.gravity = joint::FxVec3{0, gravY, 0};
        w.groundY = (joint::fx)(-1000 * (int)joint::kOne);   // far below -> the focus is the hang
        w.bodies = {pinned(0, 10, 0), dyn(0, 9, 0)};
        // Anchors at the LINK ENDS (the showcase hanging-chain semantics): the root's anchor is at its
        // LOWER end (-0.5 in y), the child's anchor at its UPPER end (+0.5 in y) -> the ball joint pins
        // those two end-points coincident, so the dynamic body's CENTRE hangs ~1.0 unit below the root's
        // centre (the link length). At rest (root y=10, child y=9) the anchors already coincide at y=9.5.
        const joint::fx kHalf = joint::kOne / 2;
        joint::FxJoint j;
        j.bodyA = 0; j.bodyB = 1;
        j.anchorA = joint::FxVec3{0, -kHalf, 0};   // root's lower end
        j.anchorB = joint::FxVec3{0,  kHalf, 0};   // child's upper end
        std::vector<joint::FxJoint> joints = {j};

        const joint::fx dt = joint::kOne / 60;
        const int kIters = 8;
        joint::StepJointWorldSteps(w, joints, dt, kIters, 200);
        const joint::fx gap1 = joint::AnchorGap(w, joints[0]);

        // The pinned root NEVER moved.
        check(w.bodies[0].pos.x == 0 && w.bodies[0].pos.y == (joint::fx)(10 * (int)joint::kOne) &&
              w.bodies[0].pos.z == 0, "StepJointWorld: the pinned root holds exactly");
        // The link stays connected: the settled world-anchor gap is SMALL (the end-anchors meet) — NOT
        // flying apart. The Gauss-Seidel residual is deterministic-but-nonzero; assert within ~1/4 unit.
        check(gap1 < joint::kOne / 4,
              "StepJointWorld: the chain stays connected (anchor gap small, not scattered)");
        // The dynamic body hangs BELOW the root (the link length keeps its centre ~1 unit under).
        check(w.bodies[1].pos.y < w.bodies[0].pos.y, "StepJointWorld: the dynamic body hangs below the root");
    }

    // ================= StepJointWorld: two runs byte-identical (determinism) =================
    {
        joint::FxWorld w0;
        const joint::fx gravY = (joint::fx)(-9.8 * (double)joint::kOne + (-9.8 < 0 ? -0.5 : 0.5));
        w0.gravity = joint::FxVec3{0, gravY, 0};
        w0.groundY = (joint::fx)(-1000 * (int)joint::kOne);
        // A 4-link chain: pinned root + 3 dynamic links hanging, ball-jointed at the link ENDS (so the
        // chain genuinely dangles, each link's centre ~1 unit below its parent — the showcase semantics).
        w0.bodies = {pinned(0, 12, 0), dyn(0, 11, 0), dyn(0, 10, 0), dyn(0, 9, 0)};
        const joint::fx kHalf2 = joint::kOne / 2;
        std::vector<joint::FxJoint> joints;
        for (uint32_t k = 0; k + 1 < (uint32_t)w0.bodies.size(); ++k) {
            joint::FxJoint j;
            j.bodyA = k; j.bodyB = k + 1;
            j.anchorA = joint::FxVec3{0, -kHalf2, 0};   // parent's lower end
            j.anchorB = joint::FxVec3{0,  kHalf2, 0};   // child's upper end
            joints.push_back(j);
        }
        const joint::fx dt = joint::kOne / 60;
        joint::FxWorld a = w0, b = w0;
        joint::StepJointWorldSteps(a, joints, dt, 6, 150);
        joint::StepJointWorldSteps(b, joints, dt, 6, 150);
        bool same = a.bodies.size() == b.bodies.size() &&
                    std::memcmp(a.bodies.data(), b.bodies.data(),
                                a.bodies.size() * sizeof(fpx::FxBody)) == 0;
        check(same, "StepJointWorld determinism: two runs BYTE-IDENTICAL");

        // The whole chain hung: the mean dynamic pos.y dropped below the root.
        int64_t sumY = 0; int dynCount = 0;
        for (size_t i = 1; i < a.bodies.size(); ++i) { sumY += a.bodies[i].pos.y; ++dynCount; }
        const joint::fx meanY = dynCount ? (joint::fx)(sumY / dynCount) : 0;
        check(meanY < a.bodies[0].pos.y, "StepJointWorld: the chain hung below the pinned root");
        // MaxAnchorGap stays small (the chain held together).
        check(joint::MaxAnchorGap(a, joints) < 2 * (int)joint::kOne,
              "StepJointWorld: max anchor gap small (chain connected, not scattered)");
    }

    if (g_fail == 0) std::printf("joint_test: ALL PASS\n");
    else std::printf("joint_test: %d FAILURE(S)\n", g_fail);
    return g_fail ? 1 : 0;
}
