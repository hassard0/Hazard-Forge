// Slice BY — client prediction + server reconciliation implementation. See prediction.h for the
// contract + the documented decision on why the reconcile frame carries velocity. Pure CPU above
// engine/math + engine/game + engine/physics + engine/net (BQ); ZERO RHI / backend symbols.
// Deterministic: the local sim is the SAME deterministic roll-game StepGame the authority runs, so a
// rewind to an authoritative frame + a replay of the buffered inputs reproduces the authority's state
// bit-exactly. Compiled into BOTH hf_core (ASan, unit-tested) and hf_engine (the --netpredict-shot
// showcase).
#include "net/prediction.h"

#include "math/math.h"
#include "physics/body.h"

namespace hf::net {

// The engine fixed timestep the roll-game runs at (matches --game-shot / --net-shot / --netsim-shot).
namespace {
constexpr float kDt = 1.0f / 120.0f;
}  // namespace

// --- AuthState::Capture --------------------------------------------------------------------------
AuthState AuthState::Capture(int tick, const game::GameState& gs, const physics::World& world) {
    AuthState a;
    a.tick = tick;
    a.snap = Replicator::Capture((uint32_t)tick, gs, world);  // BQ snapshot (replicated view)
    if (gs.playerBodyIndex >= 0 && (size_t)gs.playerBodyIndex < world.bodies.size())
        a.playerBody = world.bodies[(size_t)gs.playerBodyIndex];  // full body incl. velocity
    a.score = gs.score;
    a.collected.reserve(gs.pickups.size());
    for (const game::Pickup& p : gs.pickups) a.collected.push_back(p.collected ? 1u : 0u);
    return a;
}

// --- PredictedClient -----------------------------------------------------------------------------
PredictedClient::PredictedClient() {
    // Start the predicted sim from the SAME initial roll-game state as the authority. With no server-
    // only effect a clean predict reproduces the authority exactly (reconcile is a no-op).
    gs_ = game::MakeRollGame(world_);
}

math::Vec3 PredictedClient::PlayerPos() const {
    if (gs_.playerBodyIndex >= 0 && (size_t)gs_.playerBodyIndex < world_.bodies.size())
        return world_.bodies[(size_t)gs_.playerBodyIndex].position;
    return math::Vec3{0.0f, 0.0f, 0.0f};
}

void PredictedClient::PredictTick(const InputCmd& cmd) {
    // Apply the local input to the predicted sim IMMEDIATELY (one deterministic step) — moving ahead of
    // the server without waiting for a round-trip — and buffer it as unacknowledged for a later replay.
    game::StepGame(world_, gs_, cmd.input, kDt);
    pending_.push_back(cmd);
    predictedTick_ = cmd.tick;
}

void PredictedClient::OnAuthoritative(const AuthState& auth) {
    // Stale/duplicate guard: an authoritative frame at or behind the last one we reconciled against
    // carries no new information (the BU channel can reorder), so ignore it.
    if (auth.tick <= lastReconciledTick_) return;

    // Snapshot the predicted player position BEFORE the rewind so we can measure the misprediction (how
    // far the correction moved it).
    const math::Vec3 before = PlayerPos();

    // 1. RESET the local sim to the acknowledged authoritative state at auth.tick. Reset the full player
    //    body (position + orientation + velocity — velocity is why the reconcile frame carries the whole
    //    RigidBody; see prediction.h) and the authoritative score / pickup-collected bits. This is the
    //    rewind: the predicted sim is now exactly the authority's TRUE state at auth.tick.
    if (gs_.playerBodyIndex >= 0 && (size_t)gs_.playerBodyIndex < world_.bodies.size())
        world_.bodies[(size_t)gs_.playerBodyIndex] = auth.playerBody;
    gs_.score = auth.score;
    // Tick convention (locked): AuthState.tick == T is the state AFTER executing T steps (gs.step == T);
    // InputCmd.tick == t drives step t (advances gs.step t -> t+1). So a cmd with tick t is ACKNOWLEDGED
    // by an authoritative frame at tick T iff t < T (the authority has already run step t).
    gs_.step  = auth.tick;
    for (size_t i = 0; i < gs_.pickups.size() && i < auth.collected.size(); ++i)
        gs_.pickups[i].collected = (auth.collected[i] != 0u);
    gs_.won = (gs_.score == (int)gs_.pickups.size());

    // 2. DROP acknowledged inputs (tick < auth.tick) — the authority has already run those steps — and
    //    keep the buffer bounded. Pending is tick-ascending, so pop from the front.
    while (!pending_.empty() && pending_.front().tick < auth.tick) pending_.pop_front();

    // 3. REPLAY the still-unacknowledged inputs (tick >= auth.tick) in order to re-derive the predicted
    //    "now" on top of the corrected authoritative base. After this the predicted state again leads the
    //    server by the buffered inputs — but now corrected for whatever the authority did (e.g. a
    //    server-only impulse the client could not have predicted).
    for (const InputCmd& cmd : pending_) game::StepGame(world_, gs_, cmd.input, kDt);

    // 4. Account the misprediction: the distance the player's predicted position moved across the rewind+
    //    replay. ~0 when the prediction was right (no server-only effect); > 0 when the authority revealed
    //    an effect the client had not predicted.
    const math::Vec3 after = PlayerPos();
    lastMisprediction_ = math::length(after - before);
    if (lastMisprediction_ > maxMisprediction_) maxMisprediction_ = lastMisprediction_;

    lastReconciledTick_ = auth.tick;
    ++reconciles_;
}

}  // namespace hf::net
