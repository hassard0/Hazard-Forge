// Slice AI4 — Deterministic AI: THE NPC AGENT IN THE GAMEPLAY TICK (sense -> decide -> query -> path ->
// act) — the 4th slice of the DETERMINISTIC AI flagship (GitHub issue #28, hf::ai). ai.h is APPEND-ONLY
// (AI1/AI2/AI3 byte-frozen); AI4 COMPOSES the three frozen primitives + the frozen navmesh A* into one
// deterministic per-agent five-beat pass (StepAi / StepAiWorld) over a parallel std::vector<AiAgent>.
//
// What this test PINS (the determinism contract + the perception->decision switch + the path validity):
//   * StepAiWorld TWO-RUN determinism: two independent K-tick sequences over the canonical scene produce
//     the SAME DigestAgents (byte-identical agent state) — the order[]-keyed integer composition.
//   * The falsifiable perception->decision switch: with the player VISIBLE + close, agent 0 enters CHASE
//     and its corridor steers toward the player; with the player moved BEHIND a blocker (occluded), the
//     agent reverts to PATROL — deterministically (the AI3 LOS -> the AI1 tree branch).
//   * The path is a VALID nav corridor: each adjacent corridor poly is a true navmesh neighbour.
//   * The agent makes PROGRESS toward its destination over ticks (integer distance to the steer point
//     decreases, or it has arrived).
//   * Un-stepped state stays consistent (DigestAgents is a pure function of the agent array).
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "ai/ai.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace ai  = hf::ai;
namespace fpx = hf::sim::fpx;
namespace nav = hf::nav;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A Q16.16 helper: integer world units -> fx.
static fpx::fx FI(int v) { return (fpx::fx)((int64_t)v * (int64_t)fpx::kOne); }

// Is `b` a navmesh neighbour of `a` (one of a's nbr[] entries)?
static bool IsNeighbour(const std::vector<nav::Poly>& polys, uint32_t a, uint32_t b) {
    if (a >= polys.size()) return false;
    for (int e = 0; e < 3; ++e) if (polys[a].nbr[e] == b) return true;
    return false;
}

// |dx|+|dz| integer distance between an agent's Q16.16 pos and a Q16.16 steer point.
static int64_t SteerDist(const ai::AiAgent& a) {
    const int64_t dx = (int64_t)a.destWorldX - (int64_t)a.pos.x;
    const int64_t dz = (int64_t)a.destWorldZ - (int64_t)a.pos.z;
    return (dx < 0 ? -dx : dx) + (dz < 0 ? -dz : dz);
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- (0) The scene builds with a sane shape (agents in the largest component, a player, blockers) --
    ai::Ai4Scene scene = ai::BuildAi4Scene();
    check(!scene.agents.empty(), "ai4: BuildAi4Scene seeds at least one agent");
    check(!scene.nav.polys.empty(), "ai4: the navmesh has polys");
    check(!scene.trees.empty(), "ai4: the decision-tree table is non-empty");

    // ---- (1) StepAiWorld two-run determinism: same DigestAgents over K ticks --------------------------
    {
        const int K = 24;
        auto run = [&]() {
            ai::Ai4Scene s = ai::BuildAi4Scene();
            for (int t = 0; t < K; ++t)
                ai::StepAiWorld(s.agents, s.nav, s.blockers.data(), (int)s.blockers.size(),
                                s.player, s.trees);
            return ai::DigestAgents(s.agents);
        };
        const uint64_t d1 = run();
        const uint64_t d2 = run();
        check(d1 == d2, "ai4: two StepAiWorld sequences are BYTE-IDENTICAL (DigestAgents equal)");
    }

    // ---- (2) The falsifiable perception->decision switch: visible+close -> CHASE; occluded -> PATROL --
    {
        // Put the player ONE world unit from agent 0 (well within chase range) with a clear line.
        ai::Ai4Scene s = ai::BuildAi4Scene();
        ai::AiAgent& a0 = s.agents[0];
        s.player = fpx::FxVec3{a0.pos.x + FI(1), 0, a0.pos.z};   // 1 unit to the +x, no blocker between

        // No blockers -> the player is visible + close -> CHASE.
        std::vector<ai::AiBlocker> noBlock;
        ai::StepAi(s.agents, s.nav, noBlock.data(), 0, s.player, s.trees);
        check(s.agents[0].state == ai::kAgentChase,
              "ai4: player visible AND close -> agent 0 enters CHASE");
        check(s.agents[0].bb.Get(ai::kBbCanSeeTarget) == 1,
              "ai4: the perception blackboard records canSeeTarget=1 when visible");

        // Now drop a blocker squarely ON the segment agent0 -> player (occlude the view) -> PATROL.
        ai::Ai4Scene s2 = ai::BuildAi4Scene();
        ai::AiAgent& b0 = s2.agents[0];
        s2.player = fpx::FxVec3{b0.pos.x + FI(4), 0, b0.pos.z};   // 4 units away, same row
        // A box spanning the row between the agent (x = ax) and the player (x = ax+4), straddling z = az.
        const int ax = (int)(b0.pos.x >> fpx::kFrac);
        const int az = (int)(b0.pos.z >> fpx::kFrac);
        ai::AiBlocker wall;
        wall.min = fpx::FxVec3{FI(ax + 1), 0,        FI(az - 2)};
        wall.max = fpx::FxVec3{FI(ax + 3), FI(2),    FI(az + 2)};
        std::vector<ai::AiBlocker> blocked = { wall };
        ai::StepAi(s2.agents, s2.nav, blocked.data(), (int)blocked.size(), s2.player, s2.trees);
        check(s2.agents[0].bb.Get(ai::kBbCanSeeTarget) == 0,
              "ai4: a blocker on the sight-line occludes the player (canSeeTarget=0)");
        check(s2.agents[0].state == ai::kAgentPatrol,
              "ai4: player occluded by a blocker -> agent 0 reverts to PATROL (deterministic flip)");
    }

    // ---- (3) The path is a VALID nav corridor: adjacent polys are true navmesh neighbours -------------
    {
        ai::Ai4Scene s = ai::BuildAi4Scene();
        ai::StepAi(s.agents, s.nav, s.blockers.data(), (int)s.blockers.size(), s.player, s.trees);
        bool anyCorridor = false;
        for (const ai::AiAgent& a : s.agents) {
            if (a.corridor.size() >= 2) {
                anyCorridor = true;
                for (size_t k = 0; k + 1 < a.corridor.size(); ++k)
                    check(IsNeighbour(s.nav.polys, a.corridor[k], a.corridor[k + 1]),
                          "ai4: each corridor step is a true navmesh neighbour (valid corridor)");
            }
        }
        check(anyCorridor, "ai4: at least one agent has a multi-poly corridor to steer along");
    }

    // ---- (4) Progress: an agent steering toward a destination reduces its steer distance over ticks ---
    {
        // Drive agent 0 in CHASE with a clear, close, REACHABLE player and confirm it makes progress.
        ai::Ai4Scene s = ai::BuildAi4Scene();
        ai::AiAgent& a0 = s.agents[0];
        s.player = fpx::FxVec3{a0.pos.x + FI(2), 0, a0.pos.z + FI(2)};
        std::vector<ai::AiBlocker> noBlock;

        // First step: establish the steer point + record the distance.
        ai::StepAiWorld(s.agents, s.nav, noBlock.data(), 0, s.player, s.trees);
        const int64_t d0 = SteerDist(s.agents[0]);
        // Step several more ticks; the agent should move toward the steer point (distance non-increasing,
        // and strictly decreasing at least once unless it already arrived).
        int64_t dPrev = d0;
        bool progressed = (d0 == 0);
        for (int t = 0; t < 8; ++t) {
            ai::StepAiWorld(s.agents, s.nav, noBlock.data(), 0, s.player, s.trees);
            const int64_t d = SteerDist(s.agents[0]);
            if (d < dPrev) progressed = true;
            dPrev = d;
        }
        check(progressed, "ai4: a chasing agent makes progress toward its steer point over ticks");
        check(s.agents[0].state == ai::kAgentChase,
              "ai4: the close, visible, reachable player keeps the agent in CHASE");
    }

    // ---- (5) DigestAgents is a pure function (recompute -> identical; a mutation changes it) -----------
    {
        ai::Ai4Scene s = ai::BuildAi4Scene();
        ai::StepAiWorld(s.agents, s.nav, s.blockers.data(), (int)s.blockers.size(), s.player, s.trees);
        const uint64_t h1 = ai::DigestAgents(s.agents);
        const uint64_t h2 = ai::DigestAgents(s.agents);
        check(h1 == h2, "ai4: DigestAgents is a pure function (recompute byte-identical)");
        if (!s.agents.empty()) {
            ai::AiAgent saved = s.agents[0];
            s.agents[0].pos.x += 1;                       // a single-LSB change
            check(ai::DigestAgents(s.agents) != h1, "ai4: a 1-LSB agent change changes the digest");
            s.agents[0] = saved;
            check(ai::DigestAgents(s.agents) == h1, "ai4: restoring the agent restores the digest");
        }
    }

    if (g_fail == 0) std::printf("ai_agent_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
