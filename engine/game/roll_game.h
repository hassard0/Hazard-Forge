#pragma once
// Slice AX — playable game sample (deterministic "roll-a-ball, collect the pickups").
//
// PURE CPU above physics + math: this module has ZERO RHI / graphics-backend symbols (no vk*/MTL*/
// mtl::/Backend::Metal). It builds on the deterministic impulse `physics::World` + `RigidBody`
// (engine/physics) and `engine/math`, and is compiled into BOTH hf_core (ASan-scoped, unit-tested)
// and hf_engine (the live --game-shot showcase). The sample is driven by a fixed SCRIPTED input
// track — no live input, no RNG, no clock — so the same track on the same build yields a
// bit-identical playthrough: identical final GameState AND identical final rendered frame.
#include <vector>

#include "math/math.h"
#include "physics/world.h"

namespace hf::game {

// One fixed-step of authored input. `moveDir` is a planar (XZ) move direction, magnitude 0..1 (it is
// projected onto the ground plane and scaled by a fixed accel inside StepGame). `jump` adds a one-
// shot +Y velocity, but ONLY when the player is grounded.
struct GameInput {
    math::Vec3 moveDir{0.0f, 0.0f, 0.0f};
    bool jump = false;
};

// A collectible. `collected` flips true (once) the step the player's sphere overlaps it.
struct Pickup {
    math::Vec3 pos{0.0f, 0.0f, 0.0f};
    float radius = 0.0f;
    bool collected = false;
};

// The whole mutable game state. `playerBodyIndex` indexes into `World::bodies` (the dynamic player
// sphere). `won` flips true once `score == pickups.size()`.
struct GameState {
    int playerBodyIndex = -1;
    std::vector<Pickup> pickups;
    int score = 0;
    int step = 0;
    bool won = false;
};

// --- Fixed simulation constants (documented; RNG/clock-free) -------------------------------------
constexpr float kPlayerRadius = 0.5f;      // player sphere collider radius
constexpr float kPickupRadius = 0.3f;      // pickup sphere collider radius (visual + collection)
constexpr int   kPickupCount  = 3;         // N pickups in the sample
constexpr float kMoveAccel    = 18.0f;     // m/s^2 planar acceleration applied per input step
constexpr float kMaxPlanarSpd = 3.0f;      // clamp on planar (XZ) speed — keeps control deterministic
constexpr float kJumpSpeed    = 4.5f;      // +Y velocity injected on a grounded jump
constexpr float kGroundedEps  = 0.05f;     // grounded if player.position.y <= radius + eps

// Build the World for the roll game: a static ground plane (y=0) + a dynamic player sphere at a
// fixed start + N=3 pickups at fixed positions resting on the ground. Returns the initial GameState.
// Deterministic: no RNG, no clock.
GameState MakeRollGame(physics::World& world);

// Advance the game ONE fixed step: apply `in` as a planar acceleration to the player's linear
// velocity (and a grounded jump), step the physics World by `dt`, then collect any uncollected
// pickup the player now overlaps (distance(player, pickup) < rPlayer + rPickup) -> score++. Sets
// won = (score == pickups.size()); step++. Deterministic.
void StepGame(physics::World& world, GameState& gs, const GameInput& in, float dt);

// The fixed, hand-authored input track that, over its full length at the engine fixed dt, rolls the
// player through all 3 pickups so the playthrough WINS (final score == 3). This is the deterministic
// scenario the golden + state assertions pin.
std::vector<GameInput> ScriptedTrack();

} // namespace hf::game
