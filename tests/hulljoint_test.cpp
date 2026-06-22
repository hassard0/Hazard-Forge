// Slice HF4 — Hull Friction + Joints: HULL JOINTS COMPOSED (the one-deterministic-step headline of FLAGSHIP #30:
// HULL FRICTION + HULL JOINTS, hf::sim::hulljoint). The integer core (engine/sim/hulljoint.h, a NEW additive sibling)
// the GPU shaders/hulljoint_step.comp.hlsl copies VERBATIM + proves bit-identical. HF4 composes the body-agnostic
// joint.h solvers (ball / hinge / angular-limit) with the HF3 hull friction contacts in ONE deterministic tick.
// #includes hullfric/joint/warmhull/gjk/fpx READ-ONLY (ALL BYTE-FROZEN).
//
// What this test PINS (the contracts the GPU hulljoint_step.comp + the GPU==CPU proof build on, the spec proofs):
//   * StepJointedHullWorldN two-run determinism (a fixed world yields byte-identical final bodies + cache).
//   * a hung ball-joint CHAIN stays CONNECTED (MaxAnchorGap small after settling — the links did not fly apart).
//   * a HINGE holds its angular limit (SwingAngleCos >= cosHalfLimit — the door stayed in its plane).
//   * joints + friction COMPOSE (a jointed hull also resting on the ground keeps BOTH the joint AND the friction cone
//     satisfied in the one step).
//   * THE NO-JOINTS CONTROL: StepJointedHullWorld with EMPTY joint/limit lists == hullfric::StepWarmFrictionHullWorld
//     BYTE-FOR-BYTE (proving the joints are purely additive — the friction step is byte-frozen under composition).
//
// Pure C++ (hf_core), ASan-eligible like the other sim tests.
#include "sim/hulljoint.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace hulljoint = hf::sim::hulljoint;
namespace hullfric  = hf::sim::hullfric;
namespace joint     = hf::sim::joint;
namespace convex    = hf::sim::convex;
namespace gjk       = hf::sim::gjk;
namespace fpx       = hf::sim::fpx;
using gjk::fx;
using gjk::kOne;
using gjk::FxVec3;
using gjk::FxHull;
using fpx::FxBody;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static fx absfx(fx v) { return v < 0 ? -v : v; }
static fx F(double v) { return (fx)(v * 65536.0); }

// ----- The PINNED HF4 scene (the spec's three elements composed in ONE tick): a CHAIN of 3 dynamic hulls hung by
// ball joints from a STATIC anchor + a HINGED hull DOOR (a ball joint at the pivot + an angular-limit hinge to a
// static frame) + a hull resting on a static ground hull with friction. All boxes are 2x2x2 (half-extent 1).
//   bodies[0]   : the static CHAIN ANCHOR (invMass 0), at y = 6, x = -4.
//   bodies[1..3]: the 3 dynamic chain links, hung straight down (each kLinkGap below the previous) — the ball joints
//                 pin each link's TOP anchor to the previous body's BOTTOM anchor.
//   bodies[4]   : the static GROUND hull at y = 0, x = 2.
//   bodies[5]   : a dynamic hull RESTING on the ground at y ~ 2 (the friction-resting element — NO joints).
//   bodies[6]   : the static HINGE FRAME (invMass 0) at x = 4, y = 6 — the door's pivot anchor.
//   bodies[7]   : the dynamic hinged DOOR, hung kLinkGap below the frame, pinned at the pivot by a ball joint +
//                 limited by a hinge angular limit (swing forced into the frame's plane about world +Z).
// joints : 3 chain ball joints (anchor->link0, link0->link1, link1->link2) + 1 door pivot ball joint (frame->door).
// limits : 1 hinge limit (frame -> door) about world +Z.
static constexpr double kLinkGap = 2.2;   // link-to-link vertical spacing (slightly > 2 so they hang, not overlap)

static hulljoint::JointedHullWorld BuildSceneFull(bool withJoints) {
    hulljoint::JointedHullWorld w;
    const FxHull boxH = gjk::MakeBox(kOne, kOne, kOne);

    auto mkStatic = [&](double x, double y, double z) {
        FxBody b; b.pos = {F(x), F(y), F(z)}; b.orient = {0, 0, 0, kOne};
        b.vel = {0,0,0}; b.angVel = {0,0,0}; b.invMass = 0; b.flags = 0u; b.radius = 0;
        return b;
    };
    auto mkDyn = [&](double x, double y, double z) {
        FxBody b; b.pos = {F(x), F(y), F(z)}; b.orient = {0, 0, 0, kOne};
        b.vel = {0,0,0}; b.angVel = {0,0,0}; b.invMass = kOne; b.flags = fpx::kFlagDynamic; b.radius = 0;
        return b;
    };

    const double anchorY = 6.0;
    FxBody anchor = mkStatic(-4.0, anchorY, 0.0);                         // 0: static chain anchor
    FxBody link0  = mkDyn(-4.0, anchorY - kLinkGap, 0.0);                 // 1
    FxBody link1  = mkDyn(-4.0, anchorY - 2 * kLinkGap, 0.0);             // 2
    FxBody link2  = mkDyn(-4.0, anchorY - 3 * kLinkGap, 0.0);             // 3
    FxBody ground = mkStatic(2.0, 0.0, 0.0);                             // 4: static ground hull
    FxBody resting = mkDyn(2.0, 2.0 - 0.0625, 0.0);                       // 5: friction-resting hull (NO joints)
    FxBody frame  = mkStatic(4.0, anchorY, 0.0);                         // 6: static hinge frame (door pivot)
    FxBody door   = mkDyn(4.0, anchorY - kLinkGap, 0.0);                  // 7: dynamic hinged door (hung off the frame)

    w.hulls.bodies = {anchor, link0, link1, link2, ground, resting, frame, door};
    w.hulls.hulls  = {boxH, boxH, boxH, boxH, boxH, boxH, boxH, boxH};

    if (withJoints) {
        // The ball-joint anchors: pin the upper body's BOTTOM anchor (0,-kLinkGap/2,0) to the lower body's TOP anchor
        // (0,+kLinkGap/2,0) so the rest length of the link is ~kLinkGap (the boxes hang end-to-end, just clear of
        // overlap).
        const FxVec3 bottomA{0, F(-kLinkGap / 2.0), 0};
        const FxVec3 topA{0, F(kLinkGap / 2.0), 0};
        auto mkBall = [&](uint32_t a, uint32_t b) {
            joint::FxJoint j; j.bodyA = a; j.bodyB = b; j.anchorA = bottomA; j.anchorB = topA;
            j.kind = joint::kJointBall; j.limit = 0; return j;
        };
        // 3 chain ball joints + 1 door-pivot ball joint (the door hangs off the static frame at the pivot).
        w.joints = { mkBall(0, 1), mkBall(1, 2), mkBall(2, 3), mkBall(6, 7) };

        // 1 hinge limit: the static frame (A=6) limits the door (B=7) about the world +Z axis (a hinge keeps the door
        // swinging in the XY plane). cosHalfLimit=kOne, sinHalfLimit=0 -> the swing is forced to identity (pure hinge).
        joint::FxAngularLimit hinge;
        hinge.bodyA = 6; hinge.bodyB = 7;
        hinge.axis = {0, 0, kOne};
        hinge.cosHalfLimit = kOne; hinge.sinHalfLimit = 0;
        hinge.kind = joint::kAngularHinge;
        w.limits = { hinge };
    }
    return w;
}

static hulljoint::JointedHullStepConfig StepCfg(fx mu) {
    hulljoint::JointedHullStepConfig cfg;
    cfg.fric.mu = mu;
    cfg.fric.solveIters = 12;
    cfg.fric.posIters   = 4;
    cfg.jointIters      = 8;
    return cfg;
}

static constexpr uint32_t kTicks = 240;

int main() {
    HF_TEST_MAIN_INIT();

    // (1) two-run determinism — a fixed jointed world yields byte-identical final bodies + cache.
    {
        hulljoint::JointedHullWorld w1 = BuildSceneFull(true), w2 = BuildSceneFull(true);
        const auto cfg = StepCfg(F(0.6));
        hulljoint::StepJointedHullWorldN(w1, cfg, kTicks);
        hulljoint::StepJointedHullWorldN(w2, cfg, kTicks);
        bool bodiesEq = (w1.hulls.bodies.size() == w2.hulls.bodies.size());
        if (bodiesEq)
            bodiesEq = (std::memcmp(w1.hulls.bodies.data(), w2.hulls.bodies.data(),
                                    w1.hulls.bodies.size() * sizeof(FxBody)) == 0);
        check(bodiesEq, "two-run determinism: final bodies byte-identical");
        bool cacheEq = (w1.cache.entries.size() == w2.cache.entries.size());
        if (cacheEq && !w1.cache.entries.empty())
            cacheEq = (std::memcmp(w1.cache.entries.data(), w2.cache.entries.data(),
                                   w1.cache.entries.size() * sizeof(hullfric::CachedHullFrictionContact)) == 0);
        check(cacheEq, "two-run determinism: friction cache byte-identical");
    }

    // (2) the hung ball-joint CHAIN stays CONNECTED — MaxAnchorGap small after settling (the links did not fly apart).
    {
        hulljoint::JointedHullWorld w = BuildSceneFull(true);
        hulljoint::StepJointedHullWorldN(w, StepCfg(F(0.6)), kTicks);
        hulljoint::JointMeasure jm = hulljoint::MeasureJointedHull(w);
        std::printf("hf4-joint chain: maxAnchorGap=%d (%.4f units), swingCos=%d\n",
                    (int)jm.maxAnchorGap, (double)jm.maxAnchorGap / 65536.0, (int)jm.swingAngleCos);
        // A connected chain keeps the worst anchor gap well under half a world unit (the Gauss-Seidel residual band).
        check(jm.maxAnchorGap < (kOne / 2), "chain connected: maxAnchorGap small after settling");
    }

    // (3) the HINGE holds its angular limit — SwingAngleCos >= cosHalfLimit (the door stayed in its hinge plane).
    {
        hulljoint::JointedHullWorld w = BuildSceneFull(true);
        hulljoint::StepJointedHullWorldN(w, StepCfg(F(0.6)), kTicks);
        hulljoint::JointMeasure jm = hulljoint::MeasureJointedHull(w);
        // The hinge has cosHalfLimit == kOne; a held hinge keeps the swing cosine within an LSB band of kOne.
        check(jm.swingAngleCos >= kOne - (kOne / 64), "hinge holds plane: swingCos >= limitCos (within band)");
    }

    // (4) joints + friction COMPOSE — the friction-resting hull (body 5) keeps the cone satisfied WHILE the joints are
    //     satisfied. Re-derive the final tick's friction manifolds and check the cone on the converged accumulators.
    {
        hulljoint::JointedHullWorld w = BuildSceneFull(true);
        const auto cfg = StepCfg(F(0.6));
        hulljoint::StepJointedHullWorldN(w, cfg, kTicks);
        // The friction cone holds on every persisted cache entry (|jt| <= mu*jn, jn >= 0).
        bool coneOk = true;
        bool sawContact = !w.cache.entries.empty();
        for (const auto& ce : w.cache.entries) {
            const fx jn = ce.normalImpulse;
            const fx cone = fpx::fxmul(cfg.fric.mu, jn < 0 ? 0 : jn);
            if (jn < 0) coneOk = false;
            if (absfx(ce.tangentImpulse1) > cone + 2) coneOk = false;   // +2 LSB fixed-point slack
            if (absfx(ce.tangentImpulse2) > cone + 2) coneOk = false;
        }
        check(sawContact, "compose: a friction contact persisted (the resting hull is gripped)");
        check(coneOk, "compose: friction cone holds (|jt|<=mu*jn, jn>=0) while joints satisfied");
        // The joints are STILL satisfied at the same final state (the chain stayed connected through the friction tick).
        hulljoint::JointMeasure jm = hulljoint::MeasureJointedHull(w);
        check(jm.maxAnchorGap < (kOne / 2), "compose: the chain stayed connected through the friction step");
    }

    // (5) THE NO-JOINTS CONTROL — StepJointedHullWorld with EMPTY joint/limit lists == hullfric::StepWarmFriction-
    //     HullWorld BYTE-FOR-BYTE (the joints are purely additive; the friction step is byte-frozen under composition).
    {
        hulljoint::JointedHullWorld wj = BuildSceneFull(false);   // SAME scene, NO joints/limits
        gjk::HullWorld wf;                                        // the frozen HF3 world over the SAME bodies/hulls
        wf.bodies = wj.hulls.bodies;
        wf.hulls  = wj.hulls.hulls;
        hullfric::HullFrictionCache cf;

        const auto cfg = StepCfg(F(0.6));
        hullfric::HullFrictionStepConfig fc = cfg.fric;
        for (uint32_t t = 0; t < kTicks; ++t) {
            hulljoint::StepJointedHullWorld(wj, cfg);
            hullfric::StepWarmFrictionHullWorld(wf, cf, fc);
        }
        bool bodiesEq = (wj.hulls.bodies.size() == wf.bodies.size());
        if (bodiesEq)
            bodiesEq = (std::memcmp(wj.hulls.bodies.data(), wf.bodies.data(),
                                    wf.bodies.size() * sizeof(FxBody)) == 0);
        check(bodiesEq, "no-joints control: empty-joint step == HF3 StepWarmFrictionHullWorld (bodies byte-equal)");
        bool cacheEq = (wj.cache.entries.size() == cf.entries.size());
        if (cacheEq && !cf.entries.empty())
            cacheEq = (std::memcmp(wj.cache.entries.data(), cf.entries.data(),
                                   cf.entries.size() * sizeof(hullfric::CachedHullFrictionContact)) == 0);
        check(cacheEq, "no-joints control: empty-joint cache == HF3 cache (byte-equal)");
    }

    if (g_fail == 0) std::printf("hulljoint_test: ALL PASS\n");
    else             std::printf("hulljoint_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
