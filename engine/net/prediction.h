#pragma once
// Slice BY — Client Prediction + Server Reconciliation (Phase 4 #24). Completes the networking
// TRILOGY: replication (BQ) -> transport + interpolation (BU) -> prediction + reconciliation (this).
// Pure CPU, deterministic, NO sockets (the BU in-process SimChannel carries the authoritative
// snapshots).
//
// HARD RULE: this module is PURE CPU above engine/math + engine/game + engine/physics + engine/net
// (BQ Snapshot + BU SimChannel). It has ZERO RHI / graphics-backend symbols (no vk*/MTL*/mtl::/
// Backend::Metal). It is compiled into BOTH hf_core (ASan-scoped, unit-tested in
// tests/prediction_test.cpp) and hf_engine (the live --netpredict-shot showcase that renders the
// client's PREDICTED + RECONCILED scene through the existing lit/shadowed path).
//
// THE PROBLEM. Network latency means a client that waited for the server to acknowledge its input
// before moving would feel laggy. CLIENT PREDICTION fixes this: the client runs the SAME deterministic
// game sim locally and applies its OWN inputs IMMEDIATELY (no round-trip wait), predicting where it
// will be. The server stays AUTHORITATIVE — it runs the real sim PLUS effects the client cannot
// predict (here: scripted SERVER-ONLY impulses). When a (delayed) authoritative snapshot arrives, the
// client RECONCILES: it rewinds to the acknowledged authoritative state and REPLAYS its still-
// unacknowledged inputs, correcting any MISPREDICTION the server-only effect caused.
//
// WHY THE RECONCILE STATE CARRIES VELOCITY (documented decision). The roll-game sim is a rigid-body
// integration: a step is a pure function of the FULL body state (position, orientation, AND linear/
// angular velocity) + the input. A BQ Snapshot replicates only position + orientation (sufficient for
// the BQ/BU render-the-replica use). To rewind-and-replay BIT-EXACTLY — so the reconciled client lands
// on the authority's TRUE state, not an approximation — the reconcile payload must reset the full body
// state. So OnAuthoritative takes an AuthState that WRAPS the BQ Snapshot (tick + replicated pickups,
// the snapshot-centric design the spec calls for) PLUS the authoritative player RigidBody (full
// velocity) and the authoritative score/collected bits. This is the standard "authoritative state
// frame" of real prediction netcode; finite-differencing two snapshots for velocity would be lossy and
// would break the bit-exact convergence the test pins. The BQ wire format is UNTOUCHED (the BU channel
// still carries serialized Snapshots for the replicated-pickup view; the player reconcile frame travels
// alongside in-process).
#include <cstdint>
#include <deque>
#include <vector>

#include "game/roll_game.h"
#include "net/snapshot.h"
#include "physics/world.h"

namespace hf::net {

// One fixed-step of the LOCAL player's input, tagged with the tick it applies at. The client buffers
// these (unacknowledged) so reconciliation can REPLAY the ones the server hasn't yet confirmed.
struct InputCmd {
    int             tick = 0;
    game::GameInput input;
};

// The AUTHORITATIVE reconcile frame the client rewinds to. It carries the BQ Snapshot (tick +
// replicated entities — the player + uncollected pickups, exactly what BU delivers) PLUS the full
// authoritative player RigidBody (position + orientation + velocity) and the authoritative game score /
// pickup-collected bits, so the client can reset its LOCAL World/GameState to the authority's TRUE
// state and replay deterministically. `tick` mirrors snap.tick (the acknowledged tick).
struct AuthState {
    int                  tick = 0;
    Snapshot             snap;          // BQ snapshot at this tick (player id 0 + uncollected pickups)
    physics::RigidBody   playerBody;    // full authoritative player body (incl. velocity) for replay
    int                  score = 0;     // authoritative score at this tick
    std::vector<uint8_t> collected;     // per-pickup collected flag (authoritative), pickup-index order

    // Build an AuthState from the authority's live roll-game sim at `tick`. Reuses Replicator::Capture
    // for the snapshot so the replicated view is byte-identical to BQ/BU.
    static AuthState Capture(int tick, const game::GameState& gs, const physics::World& world);
};

// The LOCAL player's predicting client. Holds its own roll-game World/GameState (the predicted sim), a
// ring of unacknowledged InputCmds, and the last AuthState it reconciled against. PredictTick advances
// the prediction immediately under local input; OnAuthoritative rewinds + replays to reconcile.
class PredictedClient {
public:
    // Start the local predicted sim from the SAME initial roll-game state as the authority (so a clean
    // run with no server-only effect predicts exactly). The first tick the client predicts is tick 0's
    // input applied to the initial state; PredictedTick() tracks how far ahead the prediction is.
    PredictedClient();

    // Apply `cmd` to the LOCAL sim IMMEDIATELY (one StepGame) — predicting ahead of the server without
    // waiting for a snapshot. The command is buffered (unacknowledged) for a later reconcile replay. The
    // predicted state now reflects local input through cmd.tick.
    void PredictTick(const InputCmd& cmd);

    // RECONCILE against an arriving authoritative frame. Resets the LOCAL World/GameState to `auth` (the
    // acknowledged authoritative state at auth.tick), DROPS every buffered InputCmd with tick <=
    // auth.tick (acknowledged), then REPLAYS every remaining buffered InputCmd (tick >= auth.tick) in
    // order via StepGame to re-derive the predicted "now". (Tick convention: AuthState.tick == T is the
    // state after T steps; InputCmd.tick == t drives step t, so a cmd is acknowledged iff t < T.) Tracks
    // lastMisprediction (the distance the
    // player's predicted position moved between BEFORE and AFTER the rewind+replay — i.e. how wrong the
    // prediction was) and maxMisprediction (the running peak). A stale/duplicate snapshot (tick <= the
    // last reconciled tick) is ignored.
    void OnAuthoritative(const AuthState& auth);

    // --- Predicted state accessors (for rendering + the tests). ----------------------------------
    const physics::World&   World() const { return world_; }
    const game::GameState&  State() const { return gs_; }
    math::Vec3              PlayerPos() const;  // the predicted player position (for render + error)

    // The tick the prediction has advanced THROUGH (the last InputCmd's tick, -1 before any predict).
    int  PredictedTick()      const { return predictedTick_; }
    int  LastReconciledTick() const { return lastReconciledTick_; }

    // Reconciliation diagnostics (for the showcase stat line + tests).
    float    LastMisprediction() const { return lastMisprediction_; }  // most recent correction distance
    float    MaxMisprediction()  const { return maxMisprediction_; }   // running peak correction distance
    uint32_t Reconciles()        const { return reconciles_; }         // OnAuthoritative apply count
    size_t   PendingInputs()     const { return pending_.size(); }     // buffered (unacked) input count

private:
    physics::World    world_;            // the predicted local sim
    game::GameState   gs_;
    std::deque<InputCmd> pending_;       // unacknowledged inputs (tick-ascending), replayed on reconcile

    int      predictedTick_      = -1;   // last tick PredictTick advanced through
    int      lastReconciledTick_ = -1;   // last auth.tick we reconciled against (stale-snapshot guard)
    float    lastMisprediction_  = 0.0f;
    float    maxMisprediction_   = 0.0f;
    uint32_t reconciles_         = 0;
};

}  // namespace hf::net
