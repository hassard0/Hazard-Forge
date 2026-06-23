// Slice AI5 — Deterministic AI: LOCKSTEP + ROLLBACK over the NPC AI decisions (THE MOAT HEADLINE, the 5th
// slice of the DETERMINISTIC AI flagship #28). The pure-CPU contract test for the engine/ai/ai.h lockstep
// machinery appended in AI5 (SimAiTick / SnapshotAi / RestoreAi / AiStatesEqual / RunAiLockstep /
// RunAiRollback). Pure C++ (header-only, no device, no backend symbols). Namespace hf::ai.
//
// What this test PINS (the contracts the showcase + the netcode headline build on):
//   * snapshot/restore round-trip: RestoreAi(state, SnapshotAi(state0)) leaves the agents + player + tick
//     byte-identical to state0 (the rollback restore point is exact).
//   * lockstep two-peer convergence: authority + replica fed ONLY the SAME player-move stream are byte-equal
//     (whole AI state: agents + player + tick) EACH tick (RunAiLockstep sets identical==true).
//   * rollback corrected==authority: advance to a rollbackAt, snapshot, mispredict a WRONG player stream
//     that flips an agent's chase<->patrol decision, roll back + re-sim the correct stream -> bit-exact ==
//     the straight authority run.
//   * the mispredict GENUINELY diverged: the speculative pre-rollback state DIFFERED from the authority (a
//     real divergence — an agent's decision/dest/corridor — was fixed; the divergence flag is meaningful).
//   * a no-op mispredict (identical stream) does NOT diverge: the flag is falsifiable, not always-true.
//   * determinism: two full RunAiLockstep runs are byte-identical.
//
// PURE CPU: SimAiTick is the frozen AI4 StepAiWorld after the deterministic integer player nudge, so the
// converged state is bit-identical by construction (no GPU, no shader). The const navmesh/tree/blockers are
// re-supplied as params at restore (NOT snapshotted — the IK5/VD5 const-topology discipline). The headline
// is DETERMINISM + cross-platform bit-identity + rollback-replayability — what a float engine cannot do.
#include "ai/ai.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace ai  = hf::ai;
namespace fpx = hf::sim::fpx;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A world-unit Q16.16 player move command at a given tick (dx, dz in WORLD units).
static ai::AiCommand Move(int tick, int dx, int dz) {
    return ai::AiCommand{tick, fpx::FxVec3{(fpx::fx)(dx << fpx::kFrac), 0, (fpx::fx)(dz << fpx::kFrac)}};
}

int main() {
    HF_TEST_MAIN_INIT();

    // The canonical AI4 scene supplies the const navmesh + decision-tree table + blockers + the agents +
    // the player. The replayable state is the agents + player + tick (BuildAi5InitialState).
    const ai::Ai4Scene scene = ai::BuildAi4Scene();
    const ai::AiSimState init = ai::BuildAi5InitialState(scene);
    const int kTicks = 16;
    const int kRollbackAt = 6;

    // The authoritative player-move stream: drift the player toward the agents (so an agent CHASES), a
    // deterministic per-tick nudge. Pure integer world deltas.
    const std::vector<ai::AiCommand> authStream = {
        Move(2, -1, 0), Move(4, -1, 0), Move(6, 0, -1), Move(8, -1, 0), Move(10, 0, -1),
    };

    auto blockers = [&]() { return scene.blockers.data(); };
    const int kBlockers = (int)scene.blockers.size();

    // ===== snapshot/restore round-trip: bit-exact (agents + player + tick) =====
    {
        ai::AiSimState s = init;
        for (int t = 0; t < 5; ++t)
            ai::SimAiTick(s, scene.nav, scene.trees, blockers(), kBlockers, authStream, t);
        const ai::AiSimState saved = s;            // the truth to compare against
        const ai::AiSnapshot snap = ai::SnapshotAi(s);
        // mutate the state (advance more) then restore.
        for (int t = 5; t < 10; ++t)
            ai::SimAiTick(s, scene.nav, scene.trees, blockers(), kBlockers, authStream, t);
        ai::RestoreAi(s, snap);
        check(s.tick == 5, "RestoreAi resumes at the snapshot tick");
        check(ai::AiStatesEqual(s, saved), "snapshot/restore round-trip is bit-exact (agents+player+tick)");
    }

    // ===== lockstep two-peer convergence: authority == replica EACH tick =====
    {
        bool ident = false;
        const ai::AiSimState authority =
            ai::RunAiLockstep(init, scene.nav, scene.trees, blockers(), kBlockers, authStream, kTicks, &ident);
        check(ident, "RunAiLockstep: authority==replica BIT-IDENTICAL every tick (whole AI state)");
        // an independent replica from the SAME inputs -> byte-equal.
        bool ident2 = false;
        const ai::AiSimState replica =
            ai::RunAiLockstep(init, scene.nav, scene.trees, blockers(), kBlockers, authStream, kTicks, &ident2);
        check(ai::AiStatesEqual(authority, replica),
              "two RunAiLockstep runs converge byte-identically (inputs-only)");
        check(ident2, "the second lockstep run also held the per-tick invariant");
    }

    // ===== rollback corrected==authority + the mispredict genuinely diverged =====
    {
        bool ident = false;
        const ai::AiSimState authority =
            ai::RunAiLockstep(init, scene.nav, scene.trees, blockers(), kBlockers, authStream, kTicks, &ident);
        // a WRONG mispredict: yank the player FAR away at the rollback tick so an agent that should chase
        // (the authority moved the player close) instead reverts to PATROL -> the BT decision + dest +
        // corridor diverge.
        const std::vector<ai::AiCommand> mispredict = {
            Move(kRollbackAt, 20, 20),
        };
        bool correctedEq = false, diverged = false;
        const ai::AiSimState corrected =
            ai::RunAiRollback(init, scene.nav, scene.trees, blockers(), kBlockers, authStream, mispredict,
                              kTicks, kRollbackAt, &correctedEq, &diverged);
        check(correctedEq, "RunAiRollback: corrected == authority BIT-EXACT (the rollback fixed it)");
        check(ai::AiStatesEqual(corrected, authority),
              "the returned corrected state equals the straight authority run");
        check(diverged, "the pre-rollback mispredict DIFFERED from authority (a real divergence was fixed)");
    }

    // ===== a no-op mispredict (== the authority stream) does NOT diverge (the flag is meaningful) =====
    {
        bool correctedEq = false, diverged = false;
        // mispredict == the authority stream -> the speculative state CANNOT differ from authority.
        const ai::AiSimState corrected =
            ai::RunAiRollback(init, scene.nav, scene.trees, blockers(), kBlockers, authStream, authStream,
                              kTicks, kRollbackAt, &correctedEq, &diverged);
        check(correctedEq, "no-op mispredict: corrected still == authority");
        check(!diverged, "no-op mispredict (identical stream) does NOT diverge (the flag is falsifiable)");
        (void)corrected;
    }

    if (g_fail == 0) std::printf("ai_lockstep_test: ALL PASS\n");
    else std::printf("ai_lockstep_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
