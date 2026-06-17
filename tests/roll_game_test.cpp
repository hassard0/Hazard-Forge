// Slice AX unit test: the deterministic roll-a-ball gameplay loop (engine/game/roll_game.{h,cpp}).
// Pure C++ (hf_core), ASan-eligible like the other pure tests — NO GPU. Validates:
//   * Determinism:     running ScriptedTrack() twice yields a BIT-identical final GameState.
//   * Win condition:   the scripted track collects all 3 pickups -> score == 3, won == true.
//   * Collection logic: a player AT a pickup collects only it; a distant player collects none; a
//                       pickup is collected at most once.
//   * Physics integration: the player moves under input (in the input direction) and gravity keeps
//                       it on the ground (y ~= radius, no sink/explosion — bounded via KineticEnergy).
#include "game/roll_game.h"
#include "physics/world.h"
#include "physics/body.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Bit-exact equality of two GameStates (determinism oracle): score/step/won, the player body's
// position + linear velocity, and every pickup's pos/radius/collected flag.
static bool StateBitEqual(const game::GameState& a, const game::GameState& b,
                          const physics::World& wa, const physics::World& wb) {
    if (a.score != b.score || a.step != b.step || a.won != b.won) return false;
    if (a.playerBodyIndex != b.playerBodyIndex) return false;
    if (a.pickups.size() != b.pickups.size()) return false;
    const auto& pa = wa.bodies[(size_t)a.playerBodyIndex];
    const auto& pb = wb.bodies[(size_t)b.playerBodyIndex];
    if (pa.position.x != pb.position.x || pa.position.y != pb.position.y ||
        pa.position.z != pb.position.z) return false;
    if (pa.linVel.x != pb.linVel.x || pa.linVel.y != pb.linVel.y ||
        pa.linVel.z != pb.linVel.z) return false;
    for (size_t i = 0; i < a.pickups.size(); ++i) {
        const auto& x = a.pickups[i]; const auto& y = b.pickups[i];
        if (x.collected != y.collected) return false;
        if (x.pos.x != y.pos.x || x.pos.y != y.pos.y || x.pos.z != y.pos.z) return false;
        if (x.radius != y.radius) return false;
    }
    return true;
}

// Run the full scripted track to completion, leaving the world + state at the END.
static game::GameState RunFullTrack(physics::World& world) {
    game::GameState gs = game::MakeRollGame(world);
    const float dt = 1.0f / 120.0f;
    auto track = game::ScriptedTrack();
    for (const auto& in : track) game::StepGame(world, gs, in, dt);
    return gs;
}

int main() {
    HF_TEST_MAIN_INIT();
    const float dt = 1.0f / 120.0f;

    // --- 1. Determinism: two identical playthroughs are bit-identical at the end. ----------------
    {
        physics::World w1, w2;
        game::GameState g1 = RunFullTrack(w1);
        game::GameState g2 = RunFullTrack(w2);
        check(StateBitEqual(g1, g2, w1, w2),
              "ScriptedTrack runs twice -> bit-identical final GameState");
    }

    // --- 2. Win condition: the scripted track collects all 3 pickups. ----------------------------
    {
        physics::World w;
        game::GameState g = RunFullTrack(w);
        check(g.score == game::kPickupCount, "scripted track collects all 3 pickups (score == 3)");
        check(g.won, "scripted track wins (won == true)");
        bool allCollected = true;
        for (const auto& p : g.pickups) if (!p.collected) allCollected = false;
        check(allCollected, "every pickup flagged collected after a winning run");
    }

    // --- 3. Collection logic: AT a pickup collects only it; distant collects none; once only. -----
    {
        // Player exactly at pickup[0] -> collects pickup[0] only, on the first step.
        physics::World w;
        game::GameState g = game::MakeRollGame(w);
        // Teleport the player onto pickup[0] (and freeze it there); no move input.
        w.bodies[(size_t)g.playerBodyIndex].position = g.pickups[0].pos;
        w.bodies[(size_t)g.playerBodyIndex].position.y = game::kPlayerRadius;
        w.bodies[(size_t)g.playerBodyIndex].linVel = {0, 0, 0};
        int scoreBefore = g.score;
        game::StepGame(w, g, game::GameInput{}, dt);
        check(g.score == scoreBefore + 1, "player at a pickup collects exactly one");
        check(g.pickups[0].collected, "the overlapped pickup (0) is the one collected");
        check(!g.pickups[1].collected && !g.pickups[2].collected,
              "distant pickups are NOT collected");
        // Step again on the same spot -> NO double-collect.
        int scoreAfter = g.score;
        game::StepGame(w, g, game::GameInput{}, dt);
        check(g.score == scoreAfter, "a pickup is collected at most once (no double count)");
    }
    {
        // A player far from every pickup collects none over many idle steps.
        physics::World w;
        game::GameState g = game::MakeRollGame(w);
        w.bodies[(size_t)g.playerBodyIndex].position = {100.0f, game::kPlayerRadius, 100.0f};
        w.bodies[(size_t)g.playerBodyIndex].linVel = {0, 0, 0};
        for (int i = 0; i < 30; ++i) game::StepGame(w, g, game::GameInput{}, dt);
        check(g.score == 0, "a distant player collects no pickups");
        check(!g.won, "a distant player has not won");
    }

    // --- 4. Physics integration: the player moves under input, stays grounded, no explosion. ------
    {
        physics::World w;
        game::GameState g = game::MakeRollGame(w);
        math::Vec3 start = w.bodies[(size_t)g.playerBodyIndex].position;
        // Drive +X for a stretch of steps.
        game::GameInput push;
        push.moveDir = {1.0f, 0.0f, 0.0f};
        for (int i = 0; i < 60; ++i) game::StepGame(w, g, push, dt);
        math::Vec3 now = w.bodies[(size_t)g.playerBodyIndex].position;
        check(now.x > start.x + 0.2f, "player moves in the +X input direction");
        // Grounded: stays at ~ player radius (does not sink, does not fly off).
        check(std::fabs(now.y - game::kPlayerRadius) < 0.1f,
              "gravity keeps the player on the ground (y ~= radius)");
        // No explosion: total kinetic energy stays bounded.
        check(w.KineticEnergy() < 50.0f, "no explosion (kinetic energy bounded)");
    }

    if (g_fail == 0) { std::printf("roll_game_test: all checks passed\n"); return 0; }
    std::printf("roll_game_test: %d failures\n", g_fail);
    return 1;
}
