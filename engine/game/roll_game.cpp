// Slice AX — roll-a-ball gameplay loop implementation. Pure CPU above physics + math (NO RHI /
// backend symbols). See roll_game.h for the contract; determinism is the whole point — every value
// here is fixed and the only state evolution is via the deterministic physics::World::Step.
#include "game/roll_game.h"

#include "physics/body.h"

#include <cmath>

namespace hf::game {

// --- Fixed scene layout (documented; deterministic) ---------------------------------------------
// The player starts at the origin on the ground. The three pickups sit on the ground (y = pickup
// radius) along an L-shaped route the scripted track drives through: out along +X, then turn and
// run along +Z. Positions are hand-chosen so a grounded rolling sphere passes within the collection
// radius (rPlayer + rPickup = 0.8 m) of each in turn.
namespace {
const math::Vec3 kPlayerStart{0.0f, kPlayerRadius, 0.0f};
const math::Vec3 kPickup0{ 2.5f, kPickupRadius, 0.0f};
const math::Vec3 kPickup1{ 5.0f, kPickupRadius, 0.0f};
const math::Vec3 kPickup2{ 5.0f, kPickupRadius, 2.5f};
} // namespace

GameState MakeRollGame(physics::World& world) {
    world = physics::World{};                 // fresh, deterministic world (default gravity, ground y=0)

    // Dynamic player sphere at the fixed start.
    world.bodies.push_back(physics::MakeDynamicSphere(kPlayerStart, kPlayerRadius));

    GameState gs;
    gs.playerBodyIndex = static_cast<int>(world.bodies.size()) - 1;
    gs.pickups = {
        Pickup{kPickup0, kPickupRadius, false},
        Pickup{kPickup1, kPickupRadius, false},
        Pickup{kPickup2, kPickupRadius, false},
    };
    gs.score = 0;
    gs.step = 0;
    gs.won = false;
    return gs;
}

void StepGame(physics::World& world, GameState& gs, const GameInput& in, float dt) {
    physics::RigidBody& player = world.bodies[static_cast<size_t>(gs.playerBodyIndex)];

    // Apply the planar (XZ) input as an acceleration on the player's linear velocity. moveDir's Y is
    // ignored (the move is ground-planar); magnitude is taken as authored (0..1) and scaled by the
    // fixed accel constant. This is an explicit-Euler velocity bump — deterministic.
    math::Vec3 planar{in.moveDir.x, 0.0f, in.moveDir.z};
    player.linVel = player.linVel + planar * (kMoveAccel * dt);

    // Clamp planar (XZ) speed so the controllable terminal velocity is fixed — this keeps the
    // scripted track robust + the control feel tight (standard roll-a-ball), without touching the
    // physics solver. The vertical (Y) component is left to gravity/jump.
    {
        float vx = player.linVel.x, vz = player.linVel.z;
        float planarSpeed = std::sqrt(vx * vx + vz * vz);
        if (planarSpeed > kMaxPlanarSpd) {
            float s = kMaxPlanarSpd / planarSpeed;
            player.linVel.x = vx * s;
            player.linVel.z = vz * s;
        }
    }

    // Grounded jump: only when resting on the plane (within eps of y == radius) does a jump inject a
    // fixed +Y velocity (so you cannot fly by holding jump mid-air).
    bool grounded = player.position.y <= player.radius + kGroundedEps;
    if (in.jump && grounded) {
        player.linVel.y = kJumpSpeed;
    }

    // Advance the deterministic physics one fixed step (gravity + ground + contacts).
    world.Step(dt);

    // Collection: any uncollected pickup the player now overlaps is collected (once). Sphere overlap:
    // center distance < rPlayer + rPickup.
    for (Pickup& p : gs.pickups) {
        if (p.collected) continue;
        float rSum = player.radius + p.radius;
        if (math::length(player.position - p.pos) < rSum) {
            p.collected = true;
            ++gs.score;
        }
    }

    gs.won = (gs.score == static_cast<int>(gs.pickups.size()));
    ++gs.step;
}

std::vector<GameInput> ScriptedTrack() {
    // Hand-authored fixed track. Each segment drives a planar direction for a fixed number of steps;
    // tuned (with the fixed accel + ground friction of physics::World) to roll the player through
    // pickup0 (+X), pickup1 (further +X), then turn to pickup2 (+Z), then coast. The total length is
    // fixed, so the playthrough — and the captured frame — are deterministic.
    std::vector<GameInput> track;
    auto drive = [&](math::Vec3 dir, int steps) {
        GameInput in;
        in.moveDir = dir;
        for (int i = 0; i < steps; ++i) track.push_back(in);
    };
    auto coast = [&](int steps) {
        for (int i = 0; i < steps; ++i) track.push_back(GameInput{});
    };

    drive({1.0f, 0.0f, 0.0f}, 205);  // run +X at the clamped speed through pickup0 (2.5) to pickup1 (5)
    drive({-0.45f, 0.0f, 1.0f}, 130); // turn +Z toward pickup2, biasing -X to hold x ~= 5
    coast(45);                        // settle
    return track;
}

} // namespace hf::game
