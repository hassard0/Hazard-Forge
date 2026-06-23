// Slice AI6 — Deterministic AI: LIT 3D NPC RENDER CAPSTONE (the public AI sample, the 6th and FINAL slice
// of the DETERMINISTIC AI flagship #28). The pure-CPU contract test for the engine/ai/ai.h render bridge
// appended in AI6 (AgentToRenderInstances / AgentStateColor). Pure C++ (header-only, no device, no backend
// symbols). Namespace hf::ai.
//
// What this test PINS (the contracts the lit 3D showcase + the public-AI-sample claim build on):
//   * provenance two-calls byte-equal: two AgentToRenderInstances calls over the SAME world produce a
//     byte-identical marker soup (gjk::HullRenderMeshEqual) — a pure function, the render-provenance proof.
//   * marker count == agent count + 1: one cube per agent (in FIXED array index order) PLUS the player
//     marker -> (agents+1) marker cubes (each a 12-triangle box -> 36 verts).
//   * the integer AI world is byte-unmutated by the render: AiStatesEqual(before, after) holds — the render
//     is a PURE READ (the AI1-AI5 integer world is NOT touched by the float marker build).
//   * the CHASE/PATROL tint differs: AgentStateColor(kAgentChase) != AgentStateColor(kAgentPatrol) — the
//     state->color mapping is REAL, so a flipped BT decision visibly recolors the agent (falsifiable).
//   * zero agents -> the player marker only (no agent cubes); an EMPTY world (no agents, but the player is
//     always drawn) is a coherent minimal scene, not a crash.
//
// PURE CPU: AgentToRenderInstances delegates to the FROZEN verdict::AppendMarkerCube (a render-only float
// helper OUTSIDE the bit-exact loop); the bit-exact part is the integer AI world (AI1-AI5), the markers are
// float (render-only). The headline is the public AI sample: the bit-exact deterministic NPC world rendered
// as a coherent lit 3D scene, with provenance from the bit-exact AI5 world.
#include "ai/ai.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace ai      = hf::ai;
namespace fpx     = hf::sim::fpx;
namespace gjk     = hf::sim::gjk;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    // The canonical AI4 scene + the AI5 converged world (the SAME the showcase renders): build, run the
    // lockstep over a player-move stream, render the converged authority.
    const ai::Ai4Scene scene = ai::BuildAi4Scene();
    const ai::AiSimState init = ai::BuildAi5InitialState(scene);
    const int kTicks = 16;
    auto Move = [](int tick, int dx, int dz) {
        return ai::AiCommand{tick, fpx::FxVec3{(fpx::fx)(dx << fpx::kFrac), 0, (fpx::fx)(dz << fpx::kFrac)}};
    };
    const std::vector<ai::AiCommand> authStream = {
        Move(2, -1, 0), Move(4, -1, 0), Move(6, 0, -1), Move(8, -1, 0), Move(10, 0, -1),
    };
    bool ident = false;
    const ai::AiSimState converged =
        ai::RunAiLockstep(init, scene.nav, scene.trees, scene.blockers.data(),
                          (int)scene.blockers.size(), authStream, kTicks, &ident);
    check(ident, "RunAiLockstep converged (the AI5 world the render derives from is bit-exact)");
    const size_t kAgentCount = converged.agents.size();
    check(kAgentCount > 0, "the canonical scene has agents to render");

    // ===== provenance: two AgentToRenderInstances calls over the SAME world are BYTE-EQUAL =====
    {
        const gjk::HullRenderMesh a = ai::AgentToRenderInstances(converged);
        const gjk::HullRenderMesh b = ai::AgentToRenderInstances(converged);
        check(gjk::HullRenderMeshEqual(a, b),
              "two AgentToRenderInstances calls are byte-equal (render provenance: a pure function)");
    }

    // ===== marker count == agent count + 1 (the player); each cube is 36 verts (12 tris) =====
    {
        const gjk::HullRenderMesh soup = ai::AgentToRenderInstances(converged);
        const size_t kExpectedCubes = kAgentCount + 1u;          // agents + the player marker
        check(soup.verts.size() == kExpectedCubes * 36u,
              "marker soup has (agents+1) cubes of 36 verts each");
        check(soup.triangles == (uint32_t)(kExpectedCubes * 12u),
              "marker soup triangle count == (agents+1)*12");
    }

    // ===== the integer AI world is byte-unmutated by the render (a pure READ) =====
    {
        ai::AiSimState before = converged;
        const gjk::HullRenderMesh soup = ai::AgentToRenderInstances(before);
        (void)soup;
        check(ai::AiStatesEqual(before, converged),
              "AgentToRenderInstances does NOT mutate the integer AI world (pure read)");
    }

    // ===== the CHASE/PATROL tint differs (the state->color mapping is REAL) =====
    {
        float chase[3], patrol[3], unknown[3];
        ai::AgentStateColor(ai::kAgentChase, chase);
        ai::AgentStateColor(ai::kAgentPatrol, patrol);
        ai::AgentStateColor(0, unknown);
        const bool chaseVsPatrolDiffer =
            (chase[0] != patrol[0]) || (chase[1] != patrol[1]) || (chase[2] != patrol[2]);
        check(chaseVsPatrolDiffer, "CHASE tint != PATROL tint (state->color is real -> visible decision)");
        // CHASE reads warm (red dominant); PATROL reads cool (red NOT dominant).
        check(chase[0] > chase[2], "CHASE tint is warm (red > blue)");
        check(patrol[2] >= patrol[0], "PATROL tint is cool (blue >= red)");
        (void)unknown;
    }

    // ===== zero agents -> the player marker only (a coherent minimal scene, not empty/crash) =====
    {
        std::vector<ai::AiAgent> noAgents;
        const fpx::FxVec3 player{0, 0, 0};
        const gjk::HullRenderMesh soup = ai::AgentToRenderInstances(noAgents, player);
        check(soup.verts.size() == 36u, "zero agents -> exactly the player marker cube (36 verts)");
        check(soup.triangles == 12u, "zero agents -> the player marker is 12 triangles");
    }

    if (g_fail == 0) std::printf("ai_render_test: ALL PASS\n");
    else std::printf("ai_render_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
