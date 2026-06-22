// Slice HF5 — Hull Friction + Joints: LOCKSTEP + ROLLBACK over the friction+joint hull world (the netcode
// headline; the FIFTH slice of FLAGSHIP #30: HULL FRICTION + HULL JOINTS, hf::sim::hulljoint). HF5 is PURE
// CPU — it appends a deterministic in-process two-peer lockstep + rollback harness over the HF4
// StepJointedHullWorld (the WH5/JT5/FR5/VD5 mold). NO GPU, NO shader, NO RHI. #includes hulljoint.h (which
// pulls in hullfric/joint/warmhull/gjk/fpx READ-ONLY/BYTE-FROZEN).
//
// What this test PINS (the spec proofs, the contracts the showcase builds on):
//   * the (bodies, friction cache) snapshot->restore round-trip is bit-exact (JointedHullStatesEqual).
//   * two-peer LOCKSTEP convergence: an authority + a replica fed the SAME inputs (fresh caches) re-derive the
//     WHOLE (bodies, cache) PAIR byte-for-byte each peer.
//   * ROLLBACK corrects EXACTLY: a mispredicted speculative state is rolled back + re-simmed to the authority.
//   * the misprediction was REAL: the speculative pre-rollback state DIVERGED from the authority across the
//     chain AND the door AND the friction-resting hull (then got corrected).
//   * THE CACHE IS NECESSARY (the warm-state-necessary proof): a restore that DROPS the HullFrictionCache
//     re-seeds friction from cold and DIVERGES; restoring the cache too converges.
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
static fx F(double v) { return (fx)(v * 65536.0); }
static fx FI(int v)   { return (fx)((int64_t)v * (int64_t)kOne); }

// ----- The PINNED HF4 scene (== the hf4-joint / --hf5-net scene): a CHAIN of 3 dynamic hulls hung by ball
// joints from a STATIC anchor + a HINGED hull DOOR (a ball joint at the pivot + an angular-limit hinge to a
// static frame) + a hull resting on a static ground hull with friction. All boxes are 2x2x2 (half-extent 1).
static constexpr double kLinkGap = 2.2;

static hulljoint::JointedHullWorld BuildScene() {
    hulljoint::JointedHullWorld w;
    const FxHull boxH = gjk::MakeBox(kOne, kOne, kOne);
    auto mkStatic = [&](double x, double y, double z) {
        FxBody b; b.pos = {F(x), F(y), F(z)}; b.orient = {0, 0, 0, kOne};
        b.vel = {0,0,0}; b.angVel = {0,0,0}; b.invMass = 0; b.flags = 0u; b.radius = 0; return b;
    };
    auto mkDyn = [&](double x, double y, double z) {
        FxBody b; b.pos = {F(x), F(y), F(z)}; b.orient = {0, 0, 0, kOne};
        b.vel = {0,0,0}; b.angVel = {0,0,0}; b.invMass = kOne; b.flags = fpx::kFlagDynamic; b.radius = 0; return b;
    };
    const double aY = 6.0;
    w.hulls.bodies = {
        mkStatic(-4.0, aY, 0.0),                 // 0 chain anchor (static)
        mkDyn(-4.0, aY - kLinkGap, 0.0),         // 1 link0
        mkDyn(-4.0, aY - 2 * kLinkGap, 0.0),     // 2 link1
        mkDyn(-4.0, aY - 3 * kLinkGap, 0.0),     // 3 link2
        mkStatic(2.0, 0.0, 0.0),                 // 4 ground (static)
        mkDyn(2.0, 2.0 - 0.0625, 0.0),           // 5 friction-resting hull
        mkStatic(4.0, aY, 0.0),                  // 6 hinge frame (static)
        mkDyn(4.0, aY - kLinkGap, 0.0),          // 7 hinged door
    };
    w.hulls.hulls = {boxH, boxH, boxH, boxH, boxH, boxH, boxH, boxH};
    const FxVec3 bottomA{0, F(-kLinkGap / 2.0), 0};
    const FxVec3 topA{0, F(kLinkGap / 2.0), 0};
    auto mkBall = [&](uint32_t a, uint32_t b) {
        joint::FxJoint j; j.bodyA = a; j.bodyB = b; j.anchorA = bottomA; j.anchorB = topA;
        j.kind = joint::kJointBall; j.limit = 0; return j;
    };
    w.joints = { mkBall(0, 1), mkBall(1, 2), mkBall(2, 3), mkBall(6, 7) };
    joint::FxAngularLimit hinge;
    hinge.bodyA = 6; hinge.bodyB = 7; hinge.axis = {0, 0, kOne};
    hinge.cosHalfLimit = kOne; hinge.sinHalfLimit = 0; hinge.kind = joint::kAngularHinge;
    w.limits = { hinge };
    return w;
}

static hulljoint::JointedHullStepConfig StepCfg() {
    hulljoint::JointedHullStepConfig cfg;
    cfg.fric.mu = (fx)((int64_t)6 * kOne / 10);   // 0.6 Coulomb cone
    cfg.fric.solveIters = 12;
    cfg.fric.posIters   = 4;
    cfg.jointIters      = 8;
    return cfg;
}

// kRollbackAt is chosen in the STABLE-RESTING window (tick ~160+) where the friction-resting hull has settled
// into a non-empty warm cache (4 ground contacts) — so the cache-necessary proof is non-vacuous (dropping the
// warm friction state at restore actually changes the re-sim).
static constexpr uint32_t kTicks      = 240u;
static constexpr uint32_t kRollbackAt = 180u;

// The authoritative command stream: external impulses driving the chain end (body 3), the door (body 7), and
// the friction-resting hull (body 5) — the heterogeneous drive. Integer + deterministic.
static std::vector<convex::ConvexCommand> AuthStream() {
    return {
        convex::ConvexCommand{20u, convex::kConvexCmdAddImpulse, 3u, convex::FxVec3{FI(3), 0, 0}},
        convex::ConvexCommand{40u, convex::kConvexCmdAddImpulse, 7u, convex::FxVec3{FI(2), 0, 0}},
        convex::ConvexCommand{60u, convex::kConvexCmdAddImpulse, 5u, convex::FxVec3{FI(2), 0, 0}},
    };
}

// The MISPREDICTED stream: the auth stream + a WRONG strong impulse AT rollbackAt on the chain end AND the
// door AND the resting hull, so the speculative state diverges across all three (the heterogeneous-divergence
// proof).
static std::vector<convex::ConvexCommand> MispredictStream() {
    std::vector<convex::ConvexCommand> s = AuthStream();
    s.push_back(convex::ConvexCommand{kRollbackAt, convex::kConvexCmdAddImpulse, 3u, convex::FxVec3{FI(12), 0, 0}});
    s.push_back(convex::ConvexCommand{kRollbackAt, convex::kConvexCmdAddImpulse, 7u, convex::FxVec3{-FI(12), 0, 0}});
    s.push_back(convex::ConvexCommand{kRollbackAt, convex::kConvexCmdAddImpulse, 5u, convex::FxVec3{FI(12), 0, 0}});
    return s;
}

int main() {
    HF_TEST_MAIN_INIT();

    const hulljoint::JointedHullWorld kScene = BuildScene();
    hulljoint::JointedHullParams authParams;
    authParams.cfg = StepCfg();
    authParams.commands = AuthStream();
    hulljoint::JointedHullParams mispParams;
    mispParams.cfg = StepCfg();
    mispParams.commands = MispredictStream();

    // (1) snapshot -> restore round-trip is bit-exact over the (bodies, cache) PAIR.
    {
        hulljoint::JointedHullWorld w = kScene;
        for (uint32_t t = 0; t < kRollbackAt; ++t) hulljoint::SimJointedHullTick(w, authParams, t);
        const hulljoint::JointedHullSnapshot snap = hulljoint::SnapshotJointedHull(w, (int32_t)kRollbackAt);
        check(!snap.cache.entries.empty(), "round-trip: a friction cache actually accumulated (non-vacuous)");
        // Advance one tick (mutates bodies + cache), then restore + compare to the snapshot.
        hulljoint::SimJointedHullTick(w, authParams, kRollbackAt);
        hulljoint::RestoreJointedHull(w, snap);
        const hulljoint::JointedHullSnapshot after = hulljoint::SnapshotJointedHull(w, (int32_t)kRollbackAt);
        check(hulljoint::JointedHullStatesEqual(snap, after),
              "snapshot->restore round-trip bit-exact (bodies + friction cache)");
    }

    // (2) two-peer LOCKSTEP convergence — authority == replica over the WHOLE PAIR, each tick by construction
    // (RunJointedHullLockstep asserts equality at the end; we ALSO walk the loop tick-by-tick here).
    {
        bool identical = false;
        const hulljoint::JointedHullSnapshot authority =
            hulljoint::RunJointedHullLockstep(kScene, authParams, kTicks, &identical);
        check(identical, "lockstep: authority == replica BIT-IDENTICAL (whole PAIR, inputs only)");

        // Walk tick-by-tick to prove the PAIR matches at EVERY tick (not just the end).
        hulljoint::JointedHullWorld a = kScene, b = kScene;
        bool everyTick = true;
        for (uint32_t t = 0; t < kTicks; ++t) {
            hulljoint::SimJointedHullTick(a, authParams, t);
            hulljoint::SimJointedHullTick(b, authParams, t);
            const auto sa = hulljoint::SnapshotJointedHull(a, (int32_t)(t + 1));
            const auto sb = hulljoint::SnapshotJointedHull(b, (int32_t)(t + 1));
            if (!hulljoint::JointedHullStatesEqual(sa, sb)) { everyTick = false; break; }
        }
        check(everyTick, "lockstep: the (bodies, cache) PAIR is byte-equal at EVERY tick");

        // determinism: a fresh full run equals the authority.
        const hulljoint::JointedHullSnapshot authority2 =
            hulljoint::RunJointedHullLockstep(kScene, authParams, kTicks);
        check(hulljoint::JointedHullStatesEqual(authority, authority2),
              "determinism: two full lockstep runs BYTE-IDENTICAL");
    }

    // (3) ROLLBACK corrects EXACTLY + (4) the misprediction was REAL (diverged then corrected).
    {
        bool corrected = false, diverged = false;
        const hulljoint::JointedHullSnapshot rolled =
            hulljoint::RunJointedHullRollback(kScene, authParams, mispParams, kTicks, kRollbackAt,
                                              &corrected, &diverged);
        const hulljoint::JointedHullSnapshot authority =
            hulljoint::RunJointedHullLockstep(kScene, authParams, kTicks);
        check(corrected && hulljoint::JointedHullStatesEqual(rolled, authority),
              "rollback: corrected == authority BIT-EXACT (whole PAIR)");
        check(diverged, "mispredict: the speculative pre-rollback state DIVERGED (real divergence corrected)");

        // The divergence is HETEROGENEOUS — confirm the speculative state differs across the chain end (3),
        // the door (7), AND the friction-resting hull (5) bodies specifically.
        hulljoint::JointedHullWorld spec = kScene, auth = kScene;
        uint32_t specTicks = 3u;   // == RunJointedHullRollback's bound
        for (uint32_t t = 0; t < kRollbackAt; ++t) {
            hulljoint::SimJointedHullTick(spec, authParams, t);
            hulljoint::SimJointedHullTick(auth, authParams, t);
        }
        for (uint32_t s = 0; s < specTicks; ++s) {
            hulljoint::SimJointedHullTick(spec, mispParams, kRollbackAt + s);
            hulljoint::SimJointedHullTick(auth, authParams, kRollbackAt + s);
        }
        auto differ = [&](uint32_t i) {
            return std::memcmp(&spec.hulls.bodies[i], &auth.hulls.bodies[i], sizeof(FxBody)) != 0;
        };
        check(differ(3) && differ(7) && differ(5),
              "mispredict diverged across the chain (3) AND the door (7) AND the friction hull (5)");
    }

    // (5) THE CACHE IS NECESSARY (the warm-state-necessary proof): a restore that DROPS the HullFrictionCache
    // re-seeds friction from cold and DIVERGES from the authority; restoring the cache too converges.
    {
        const hulljoint::JointedHullSnapshot authority =
            hulljoint::RunJointedHullLockstep(kScene, authParams, kTicks);

        // INCLUDE the cache: full snapshot/restore at rollbackAt, re-sim -> equals authority.
        {
            hulljoint::JointedHullWorld w = kScene;
            for (uint32_t t = 0; t < kRollbackAt; ++t) hulljoint::SimJointedHullTick(w, authParams, t);
            const hulljoint::JointedHullSnapshot full = hulljoint::SnapshotJointedHull(w, (int32_t)kRollbackAt);
            // perturb (mispredict a couple ticks) then restore the FULL snapshot.
            for (uint32_t s = 0; s < 3u; ++s) hulljoint::SimJointedHullTick(w, mispParams, kRollbackAt + s);
            hulljoint::RestoreJointedHull(w, full);
            for (uint32_t t = kRollbackAt; t < kTicks; ++t) hulljoint::SimJointedHullTick(w, authParams, t);
            const hulljoint::JointedHullSnapshot got = hulljoint::SnapshotJointedHull(w, (int32_t)kTicks);
            check(hulljoint::JointedHullStatesEqual(got, authority),
                  "cache necessary: INCLUDE the friction cache -> corrected == authority");
        }

        // OMIT the cache: snapshot only the bodies (drop the HullFrictionCache -> cold-start), re-sim ->
        // DIVERGES from authority (the warm-start friction state was lost).
        {
            hulljoint::JointedHullWorld w = kScene;
            for (uint32_t t = 0; t < kRollbackAt; ++t) hulljoint::SimJointedHullTick(w, authParams, t);
            hulljoint::JointedHullSnapshot bodiesOnly = hulljoint::SnapshotJointedHull(w, (int32_t)kRollbackAt);
            bodiesOnly.cache.entries.clear();   // DROP the warm friction state (the cold-restore bug)
            for (uint32_t s = 0; s < 3u; ++s) hulljoint::SimJointedHullTick(w, mispParams, kRollbackAt + s);
            hulljoint::RestoreJointedHull(w, bodiesOnly);   // restores bodies + an EMPTY cache
            for (uint32_t t = kRollbackAt; t < kTicks; ++t) hulljoint::SimJointedHullTick(w, authParams, t);
            const hulljoint::JointedHullSnapshot got = hulljoint::SnapshotJointedHull(w, (int32_t)kTicks);
            check(!hulljoint::JointedHullStatesEqual(got, authority),
                  "cache necessary: OMIT the friction cache -> diverges (warm-state necessary)");
        }
    }

    if (g_fail == 0) std::printf("hulljoint_net_test: ALL PASS (HF5 lockstep + rollback + cache-necessary)\n");
    else             std::printf("hulljoint_net_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
