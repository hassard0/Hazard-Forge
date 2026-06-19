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
#include <cmath>     // Slice JT2: test-side host construction of angle/axis quaternions (NOT the sim path)
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

    // =============================================================================================
    // ===== Slice JT2 — ANGULAR LIMITS (hinge + cone): the swing-twist + cone clamp + nlerp =========
    // =============================================================================================

    // A Q16.16 quaternion from an axis (host doubles) + angle (radians), host-snapped (test construction
    // only — the SIM path stays pure-integer). q = {axis*sin(θ/2), cos(θ/2)}.
    auto quatAxisAngle = [](double ax, double ay, double az, double ang) {
        const double s = std::sin(ang * 0.5), c = std::cos(ang * 0.5);
        const double len = std::sqrt(ax * ax + ay * ay + az * az);
        if (len > 0) { ax /= len; ay /= len; az /= len; }
        auto snap = [](double v) { return (fpx::fx)(v * (double)joint::kOne + (v < 0 ? -0.5 : 0.5)); };
        return fpx::FxQuat{snap(ax * s), snap(ay * s), snap(az * s), snap(c)};
    };
    // |quaternion| within an LSB band of kOne (the normalize tolerance).
    auto unitish = [](const fpx::FxQuat& q) {
        const int64_t m2 = (int64_t)q.x * q.x + (int64_t)q.y * q.y + (int64_t)q.z * q.z + (int64_t)q.w * q.w;
        const int64_t one2 = (int64_t)joint::kOne * (int64_t)joint::kOne;
        const int64_t band = one2 / 64;   // ~1.5% on |q|^2
        const int64_t d = m2 - one2;
        return (d < 0 ? -d : d) < band;
    };

    // ================= QConj: the conjugate negates xyz, keeps w =================
    {
        const fpx::FxQuat q{1, 2, 3, 4};
        const fpx::FxQuat c = joint::QConj(q);
        check(c.x == -1 && c.y == -2 && c.z == -3 && c.w == 4, "QConj negates xyz, keeps w");
    }

    // ================= FxDot: the Q16.16 3-vector dot =================
    {
        // (1,2,0)·(3,0,0) = 3 (in Q16.16).
        const joint::FxVec3 a{joint::kOne, 2 * (int)joint::kOne, 0};
        const joint::FxVec3 b{3 * (int)joint::kOne, 0, 0};
        check(joint::FxDot(a, b) == 3 * (int)joint::kOne, "FxDot computes the Q16.16 dot product");
    }

    // ================= QNlerp: t=0 -> p, t=kOne -> q, midpoint =================
    {
        const fpx::FxQuat p{0, 0, 0, joint::kOne};
        const fpx::FxQuat q{joint::kOne, 0, 0, 0};
        const fpx::FxQuat at0 = joint::QNlerp(p, q, 0);
        check(at0.x == p.x && at0.w == p.w, "QNlerp t=0 -> p");
        const fpx::FxQuat at1 = joint::QNlerp(p, q, joint::kOne);
        check(at1.x == q.x && at1.w == q.w, "QNlerp t=kOne -> q");
        const fpx::FxQuat mid = joint::QNlerp(p, q, joint::kOne / 2);
        check(mid.x == joint::kOne / 2 && mid.w == joint::kOne / 2, "QNlerp midpoint is the component mean");
    }

    // ================= swing-twist round-trip: swing * twist == qrel (within an LSB band) =============
    {
        // qrel = a rotation about an off-axis direction (so both swing + twist are non-trivial). Decompose
        // about Y; recompose swing*twist and compare to qrel.
        const fpx::FxQuat qrel = quatAxisAngle(0.3, 1.0, 0.2, 0.8);   // a generic rotation
        const joint::FxVec3 axis{0, joint::kOne, 0};
        const joint::FxVec3 qrelXyz{qrel.x, qrel.y, qrel.z};
        const joint::fx proj = joint::FxDot(qrelXyz, axis);
        const fpx::FxQuat twist = joint::FxQuatNormalize(
            fpx::FxQuat{joint::fxmul(axis.x, proj), joint::fxmul(axis.y, proj),
                        joint::fxmul(axis.z, proj), qrel.w});
        const fpx::FxQuat swing = joint::FxQuatMul(qrel, joint::QConj(twist));
        const fpx::FxQuat recomposed = joint::FxQuatNormalize(joint::FxQuatMul(swing, twist));
        // qrel was built unit; compare recomposed against the normalized qrel within a band.
        const fpx::FxQuat qrelN = joint::FxQuatNormalize(qrel);
        const joint::fx band = joint::kOne / 32;   // ~3% per component
        auto near = [&](joint::fx u, joint::fx v) { joint::fx d = u - v; return (d < 0 ? -d : d) < band; };
        // The decomposition is sign-ambiguous (q and -q are the same rotation); accept either.
        const bool same = near(recomposed.x, qrelN.x) && near(recomposed.y, qrelN.y) &&
                          near(recomposed.z, qrelN.z) && near(recomposed.w, qrelN.w);
        const bool neg = near(recomposed.x, -qrelN.x) && near(recomposed.y, -qrelN.y) &&
                         near(recomposed.z, -qrelN.z) && near(recomposed.w, -qrelN.w);
        check(same || neg, "swing-twist round-trip: swing*twist == qrel within an LSB band");
        check(unitish(swing) && unitish(twist), "swing-twist: both swing and twist are unit");
    }

    // Build a 2-body world (frame A pinned-or-dynamic + body B) for the angular-limit cases.
    auto twoBodyWorld = [&](bool aPinned, const fpx::FxQuat& qA, const fpx::FxQuat& qB) {
        joint::FxWorld w;
        w.gravity = joint::FxVec3{0, 0, 0};
        w.groundY = (joint::fx)(-1000 * (int)joint::kOne);
        fpx::FxBody a = aPinned ? pinned(0, 0, 0) : dyn(0, 0, 0);
        fpx::FxBody b = dyn(1, 0, 0);
        a.orient = qA; b.orient = qB;
        w.bodies = {a, b};
        return w;
    };

    // ================= SolveAngularLimit HINGE drives an off-axis body back into the hinge plane ======
    {
        // A is identity; B is rotated about X (off the Y hinge axis) by ~0.6 rad -> a pure SWING. A HINGE
        // (cosHalfLimit=kOne, sinHalfLimit=0) forces the swing toward identity, so the swing .w rises toward
        // kOne (the body is driven back into the hinge plane). Multiple passes tighten (Gauss-Seidel).
        const fpx::FxQuat qA{0, 0, 0, joint::kOne};
        const fpx::FxQuat qB = quatAxisAngle(1, 0, 0, 0.6);   // a swing about X
        joint::FxWorld w = twoBodyWorld(/*aPinned=*/false, qA, qB);
        joint::FxAngularLimit lim;
        lim.bodyA = 0; lim.bodyB = 1;
        lim.axis = joint::FxVec3{0, joint::kOne, 0};   // Y hinge
        lim.cosHalfLimit = joint::kOne; lim.sinHalfLimit = 0;
        lim.kind = joint::kAngularHinge;
        const joint::fx before = joint::SwingAngleCos(w, lim);
        for (int it = 0; it < 8; ++it) joint::SolveAngularLimit(w, lim);
        const joint::fx after = joint::SwingAngleCos(w, lim);
        check(after > before, "SolveAngularLimit HINGE: the swing .w rises toward cosHalfLimit (driven in-plane)");
        check(after > joint::kOne - joint::kOne / 16,
              "SolveAngularLimit HINGE: the body is driven (nearly) into the hinge plane");
    }

    // ================= SolveAngularLimit CONE clamps a beyond-cone orientation to the cone =============
    {
        // The cone half-angle is 30° -> cosHalfLimit = cos(15°), sinHalfLimit = sin(15°). B is swung 60°
        // about X (well beyond the cone). After the clamp the swing .w must rise to AT LEAST ~cos(15°).
        const double half = 30.0 * 3.14159265358979323846 / 180.0;   // cone half-angle 30°
        const joint::fx cosHalf = (joint::fx)(std::cos(half * 0.5) * (double)joint::kOne + 0.5);
        const joint::fx sinHalf = (joint::fx)(std::sin(half * 0.5) * (double)joint::kOne + 0.5);
        const fpx::FxQuat qA{0, 0, 0, joint::kOne};
        const fpx::FxQuat qB = quatAxisAngle(1, 0, 0, 60.0 * 3.14159265358979323846 / 180.0);  // 60° swing
        joint::FxWorld w = twoBodyWorld(/*aPinned=*/true, qA, qB);   // pin A so all correction lands on B
        joint::FxAngularLimit lim;
        lim.bodyA = 0; lim.bodyB = 1;
        lim.axis = joint::FxVec3{0, joint::kOne, 0};
        lim.cosHalfLimit = cosHalf; lim.sinHalfLimit = sinHalf;
        lim.kind = joint::kAngularCone;
        const joint::fx before = joint::SwingAngleCos(w, lim);
        for (int it = 0; it < 8; ++it) joint::SolveAngularLimit(w, lim);
        const joint::fx after = joint::SwingAngleCos(w, lim);
        check(before < cosHalf, "SolveAngularLimit CONE: the beyond-cone orientation starts outside the cone");
        check(after >= cosHalf - joint::kOne / 64,
              "SolveAngularLimit CONE: the swing is clamped to (about) the cone half-angle");
    }

    // ================= SolveAngularLimit: an in-limit orientation is a (near) no-op =================
    {
        // A 10° swing inside a 30° cone -> the clamp branch never triggers; the orientation barely changes.
        const double half = 30.0 * 3.14159265358979323846 / 180.0;
        const joint::fx cosHalf = (joint::fx)(std::cos(half * 0.5) * (double)joint::kOne + 0.5);
        const joint::fx sinHalf = (joint::fx)(std::sin(half * 0.5) * (double)joint::kOne + 0.5);
        const fpx::FxQuat qA{0, 0, 0, joint::kOne};
        const fpx::FxQuat qB = quatAxisAngle(1, 0, 0, 10.0 * 3.14159265358979323846 / 180.0);
        joint::FxWorld w = twoBodyWorld(/*aPinned=*/true, qA, qB);
        const fpx::FxQuat bBefore = w.bodies[1].orient;
        joint::FxAngularLimit lim;
        lim.bodyA = 0; lim.bodyB = 1;
        lim.axis = joint::FxVec3{0, joint::kOne, 0};
        lim.cosHalfLimit = cosHalf; lim.sinHalfLimit = sinHalf;
        lim.kind = joint::kAngularCone;
        joint::SolveAngularLimit(w, lim);
        const fpx::FxQuat bAfter = w.bodies[1].orient;
        const joint::fx band = joint::kOne / 32;
        auto near = [&](joint::fx u, joint::fx v) { joint::fx d = u - v; return (d < 0 ? -d : d) < band; };
        check(near(bAfter.x, bBefore.x) && near(bAfter.y, bBefore.y) &&
              near(bAfter.z, bBefore.z) && near(bAfter.w, bBefore.w),
              "SolveAngularLimit in-limit: a within-cone orientation is a (near) no-op");
    }

    // ================= SolveAngularLimit: a pinned body (invMass 0) is NOT rotated =================
    {
        // A is PINNED. A HINGE should rotate ONLY B (A holds its orientation exactly — wA = 0).
        const fpx::FxQuat qA{0, 0, 0, joint::kOne};
        const fpx::FxQuat qB = quatAxisAngle(1, 0, 0, 0.6);
        joint::FxWorld w = twoBodyWorld(/*aPinned=*/true, qA, qB);
        const fpx::FxQuat aBefore = w.bodies[0].orient;
        joint::FxAngularLimit lim;
        lim.bodyA = 0; lim.bodyB = 1;
        lim.axis = joint::FxVec3{0, joint::kOne, 0};
        lim.cosHalfLimit = joint::kOne; lim.sinHalfLimit = 0;
        lim.kind = joint::kAngularHinge;
        joint::SolveAngularLimit(w, lim);
        check(w.bodies[0].orient.x == aBefore.x && w.bodies[0].orient.y == aBefore.y &&
              w.bodies[0].orient.z == aBefore.z && w.bodies[0].orient.w == aBefore.w,
              "SolveAngularLimit pinned: the pinned body's orientation is NOT rotated");
    }

    // ================= SolveAngularLimit: a FREE limit (cosHalfLimit = -kOne) never clamps ============
    {
        // cosHalfLimit = -kOne is a 180° cone -> swing.w >= -kOne is ALWAYS true -> the clamp branch never
        // triggers -> the orientation is (nlerp toward itself, since qrelClamped == qrel) a near no-op.
        const fpx::FxQuat qA{0, 0, 0, joint::kOne};
        const fpx::FxQuat qB = quatAxisAngle(1, 0, 0, 1.2);   // a large swing
        joint::FxWorld w = twoBodyWorld(/*aPinned=*/true, qA, qB);
        const fpx::FxQuat bBefore = w.bodies[1].orient;
        joint::FxAngularLimit lim;
        lim.bodyA = 0; lim.bodyB = 1;
        lim.axis = joint::FxVec3{0, joint::kOne, 0};
        lim.cosHalfLimit = -(joint::fx)joint::kOne; lim.sinHalfLimit = 0;   // free
        lim.kind = joint::kAngularCone;
        joint::SolveAngularLimit(w, lim);
        const fpx::FxQuat bAfter = w.bodies[1].orient;
        const joint::fx band = joint::kOne / 16;
        auto near = [&](joint::fx u, joint::fx v) { joint::fx d = u - v; return (d < 0 ? -d : d) < band; };
        check(near(bAfter.x, bBefore.x) && near(bAfter.y, bBefore.y) &&
              near(bAfter.z, bBefore.z) && near(bAfter.w, bBefore.w),
              "SolveAngularLimit free (cosHalfLimit=-kOne): never clamps (a near no-op)");
    }

    // ================= StepArticulated: two runs byte-identical (determinism) =================
    {
        // The swinging-door scene (== the showcase): a pinned frame + a door, a ball joint at the hinge + a
        // HINGE limit about Y, the door seeded a twist + a small off-axis angVel.
        const joint::fx gravY = (joint::fx)(-9.8 * (double)joint::kOne + (-9.8 < 0 ? -0.5 : 0.5));
        const joint::fx kDoorHalf = joint::kOne;
        auto build = [&]() {
            joint::FxWorld w;
            w.gravity = joint::FxVec3{0, gravY, 0};
            w.groundY = (joint::fx)(-100 * (int)joint::kOne);
            fpx::FxBody frame = pinned(6, 12, 0);
            fpx::FxBody door = dyn(6, 12, 0);
            door.pos.x += kDoorHalf;
            door.angVel = joint::FxVec3{joint::kOne * 3 / 4, joint::kOne * 3 / 2, 0};  // off-axis + twist
            w.bodies = {frame, door};
            return w;
        };
        std::vector<joint::FxJoint> joints2;
        {
            joint::FxJoint j;
            j.bodyA = 0; j.bodyB = 1;
            j.anchorA = joint::FxVec3{0, 0, 0};
            j.anchorB = joint::FxVec3{-kDoorHalf, 0, 0};
            j.kind = joint::kJointBall;
            joints2.push_back(j);
        }
        std::vector<joint::FxAngularLimit> limits2;
        {
            joint::FxAngularLimit lim;
            lim.bodyA = 0; lim.bodyB = 1;
            lim.axis = joint::FxVec3{0, joint::kOne, 0};
            lim.cosHalfLimit = joint::kOne; lim.sinHalfLimit = 0;
            lim.kind = joint::kAngularHinge;
            limits2.push_back(lim);
        }
        const joint::fx dt = joint::kOne / 60;
        joint::FxWorld a = build(), b = build();
        joint::StepArticulatedSteps(a, joints2, limits2, dt, 16, 120);
        joint::StepArticulatedSteps(b, joints2, limits2, dt, 16, 120);
        const bool same = a.bodies.size() == b.bodies.size() &&
                          std::memcmp(a.bodies.data(), b.bodies.data(),
                                      a.bodies.size() * sizeof(fpx::FxBody)) == 0;
        check(same, "StepArticulated determinism: two runs BYTE-IDENTICAL");
        // The pinned frame never moved.
        check(a.bodies[0].pos.x == (joint::fx)(6 * (int)joint::kOne) &&
              a.bodies[0].pos.y == (joint::fx)(12 * (int)joint::kOne),
              "StepArticulated: the pinned frame holds exactly");
        // The door held its hinge: its off-axis swing stayed within the cone (in the hinge plane).
        const joint::fx swingCos = joint::SwingAngleCos(a, limits2[0]);
        check(swingCos >= joint::kOne - joint::kOne / 64,
              "StepArticulated: the door stayed in the hinge plane (swing within the cone)");
    }

    if (g_fail == 0) std::printf("joint_test: ALL PASS\n");
    else std::printf("joint_test: %d FAILURE(S)\n", g_fail);
    return g_fail ? 1 : 0;
}
