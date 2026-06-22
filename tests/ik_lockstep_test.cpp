// Slice IK5 — Deterministic IK Control-Rig: LOCKSTEP + ROLLBACK over IK-driven character motion (THE MOAT
// HEADLINE, the 5th slice of FLAGSHIP #32). The pure-CPU contract test for the engine/anim/ik.h lockstep
// machinery appended in IK5 (SimIkTick / SnapshotIkRig / RestoreIkRig / IkStatesEqual / RunIkLockstep /
// RunIkRollback). Pure C++ (header-only, no device, no backend symbols). Namespace hf::anim::ik.
//
// What this test PINS (the contracts the showcase + the netcode headline build on):
//   * snapshot/restore round-trip: RestoreIkRig(state, SnapshotIkRig(state0, t)) leaves the targets + the
//     corrected pose byte-identical to state0 AND resumes at tick t (the rollback restore point is exact).
//   * lockstep two-peer convergence: authority + replica fed ONLY the SAME target-move stream are byte-equal
//     (whole rig state: targets + pose) EACH tick (RunIkLockstep sets identical==true).
//   * rollback corrected==authority: advance to a rollbackAt, snapshot, mispredict a wrong target stream,
//     roll back + re-sim the correct stream -> bit-exact == the straight authority run.
//   * the mispredict GENUINELY diverged: the speculative pre-rollback state DIFFERED from the authority (a
//     real divergence was fixed — the divergence flag is meaningful).
//   * a no-op mispredict (identical stream) does NOT diverge: the flag is falsifiable, not always-true.
//   * determinism: two full RunIkLockstep runs are byte-identical.
//
// PURE CPU: SimIkTick is the frozen IK4 SolveRigToTargets after the deterministic integer target nudge, so the
// converged state is bit-identical by construction (no GPU, no shader). The headline is DETERMINISM +
// cross-platform bit-identity + rollback-replayability — what a float engine cannot do.
#include "anim/ik.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace ik = hf::anim::ik;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static ik::fx F(double v) { return (ik::fx)std::llround(v * (double)ik::kOne); }

// The fixed hand-built TEST SKELETON: a 5-joint humanoid leg lying straight down -Y (== the IK4 config):
//   0 pelvis (root, -1) | 1 hip (0) | 2 knee (1) | 3 ankle (2) | 4 foot (3, the end-effector).
// The IK chain is hip->knee->ankle->foot (joints 1,2,3,4, count 4). The base local rotations are identity.
static const int kJointCount = 5;
static const int kParents[kJointCount] = {-1, 0, 1, 2, 3};
static const double kSegDown[kJointCount] = {0.0, 0.2, 0.5, 0.5, 0.2};

static ik::IkBasePose BuildBasePose() {
    ik::IkBasePose b{};
    b.count = kJointCount;
    for (int j = 0; j < kJointCount; ++j) {
        b.localT[j] = ik::FxVec3{0, (ik::fx)(-F(kSegDown[j])), 0};
        b.localR[j] = ik::FxQuat{0, 0, 0, ik::kOne};   // identity
    }
    return b;
}

// The initial sim state: one leg chain, the target at the rest foot (0, -1.4, 0), pose solved to the base.
static ik::IkSimState BuildInitialState() {
    ik::IkSimState s{};
    s.base = BuildBasePose();
    s.chains = 1;
    s.rig[0].count = 4;
    s.rig[0].joint[0] = 1; s.rig[0].joint[1] = 2; s.rig[0].joint[2] = 3; s.rig[0].joint[3] = 4;
    s.rig[0].target = ik::FxVec3{0, (ik::fx)(-F(1.4)), 0};   // the rest foot (straight down)
    s.rig[0].iters = 12;
    ik::SolveRigState(s, kParents);   // seed the pose from the initial target
    return s;
}

// The authority target-move stream: nudge the foot target forward + up along a planted arc over several ticks
// (the "dragged foot target"). dx/dy are world deltas in Q16.16.
static std::vector<ik::IkCommand> BuildAuthStream() {
    auto move = [](int tick, double dx, double dy) {
        return ik::IkCommand{tick, 0, ik::FxVec3{F(dx), F(dy), 0}, ik::kCmdMoveTarget};
    };
    return {
        move(2,  0.10, 0.10),   // drag forward + up
        move(4,  0.10, 0.05),
        move(6,  0.08, 0.05),
        move(8,  0.06, 0.04),
        move(10, 0.05, 0.03),
    };
}

int main() {
    HF_TEST_MAIN_INIT();
    const ik::IkSimState init = BuildInitialState();
    const std::vector<ik::IkCommand> authStream = BuildAuthStream();
    const int kTicks = 16;
    const int kRollbackAt = 6;

    // ===== snapshot/restore round-trip: bit-exact (targets + pose + tick) =====
    {
        ik::IkSimState s = init;
        // advance a few ticks so the state is non-trivial.
        for (int t = 0; t < 5; ++t) ik::SimIkTick(s, kParents, authStream, t);
        const ik::IkSimState saved = s;            // the truth to compare against
        const ik::IkSnapshot snap = ik::SnapshotIkRig(s, 5);
        // mutate the state (advance more) then restore.
        for (int t = 5; t < 10; ++t) ik::SimIkTick(s, kParents, authStream, t);
        const int resume = ik::RestoreIkRig(s, snap);
        check(resume == 5, "RestoreIkRig returns the snapshot tick");
        check(ik::IkStatesEqual(s, saved), "snapshot/restore round-trip is bit-exact (targets + pose)");
    }

    // ===== lockstep two-peer convergence: authority == replica EACH tick =====
    {
        bool ident = false;
        const ik::IkSimState authority =
            ik::RunIkLockstep(init, kParents, authStream, kTicks, &ident);
        check(ident, "RunIkLockstep: authority==replica BIT-IDENTICAL every tick (whole rig state)");
        // an independent replica from the SAME inputs -> byte-equal.
        bool ident2 = false;
        const ik::IkSimState replica =
            ik::RunIkLockstep(init, kParents, authStream, kTicks, &ident2);
        check(ik::IkStatesEqual(authority, replica),
              "two RunIkLockstep runs converge byte-identically (inputs-only)");
        check(ident2, "the second lockstep run also held the per-tick invariant");
    }

    // ===== rollback corrected==authority + the mispredict genuinely diverged =====
    {
        bool ident = false;
        const ik::IkSimState authority =
            ik::RunIkLockstep(init, kParents, authStream, kTicks, &ident);
        // a WRONG mispredict: a big target yank in the opposite direction at the rollback tick (diverges).
        const std::vector<ik::IkCommand> mispredict = {
            ik::IkCommand{kRollbackAt, 0, ik::FxVec3{F(-0.5), F(-0.4), 0}, ik::kCmdMoveTarget},
        };
        bool correctedEq = false, diverged = false;
        const ik::IkSimState corrected =
            ik::RunIkRollback(init, kParents, authStream, mispredict, kTicks, kRollbackAt,
                              &correctedEq, &diverged);
        check(correctedEq, "RunIkRollback: corrected == authority BIT-EXACT (the rollback fixed it)");
        check(ik::IkStatesEqual(corrected, authority),
              "the returned corrected state equals the straight authority run");
        check(diverged, "the pre-rollback mispredict DIFFERED from authority (a real divergence was fixed)");
    }

    // ===== a no-op mispredict (== the authority stream) does NOT diverge (the flag is meaningful) =====
    {
        bool correctedEq = false, diverged = false;
        // mispredict == the authority stream -> the speculative state CANNOT differ from authority.
        const ik::IkSimState corrected =
            ik::RunIkRollback(init, kParents, authStream, authStream, kTicks, kRollbackAt,
                              &correctedEq, &diverged);
        check(correctedEq, "no-op mispredict: corrected still == authority");
        check(!diverged, "no-op mispredict (identical stream) does NOT diverge (the flag is falsifiable)");
        (void)corrected;
    }

    if (g_fail == 0) std::printf("ik_lockstep_test: ALL PASS\n");
    else std::printf("ik_lockstep_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
