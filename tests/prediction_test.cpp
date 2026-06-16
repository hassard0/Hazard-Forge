// Unit test for client prediction + server reconciliation (engine/net/prediction.{h,cpp}, Slice BY).
// Pure CPU, deterministic, NO sockets / GPU. Asserts:
//   * Prediction advances: PredictTick moves the predicted player ahead under local input WITHOUT
//     waiting for any authoritative snapshot (position moves in the input direction).
//   * Reconcile with NO misprediction: when the authority ran the IDENTICAL sim (no server-only
//     effect), OnAuthoritative leaves the predicted "now" UNCHANGED (replay reproduces the same state)
//     -> lastMisprediction ~= 0.
//   * Reconcile WITH misprediction: inject a SERVER-ONLY impulse the client did NOT predict; before
//     reconcile the predicted state differs from the authority's, after OnAuthoritative + replay the
//     predicted "now" == the authority's TRUE corrected state -> lastMisprediction > 0.
//   * Input buffer: acknowledged inputs (tick < auth.tick) are dropped; unacknowledged (tick >=
//     auth.tick) are replayed in order; the buffer does NOT grow unbounded.
//   * Convergence/determinism: over a full run with a scripted server-only impulse, the reconciled
//     client converges to the authority (finalError within tolerance, maxMisprediction > 0); two runs
//     are identical.
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "net/prediction.h"

#include "game/roll_game.h"
#include "physics/world.h"
#include "physics/body.h"
#include "net/snapshot.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static constexpr float kDt = 1.0f / 120.0f;

// --- A self-contained AUTHORITY sim with a scripted SERVER-ONLY impulse. --------------------------
// The authority runs the SAME roll-game the client predicts, PLUS a fixed lateral impulse applied to
// the player at a documented tick the client cannot predict. This is what makes the test non-trivial:
// without it, prediction would always be exactly right and reconciliation a no-op.
//
// Tick convention (locked, matches prediction.cpp): "tick T" is the state AFTER executing T steps. The
// server-only impulse is applied as an instantaneous velocity bump to the player's linVel just BEFORE
// executing step `kImpulseTick` (so it influences step kImpulseTick onward). The client never applies
// it, so its prediction diverges from kImpulseTick until the authoritative frame at/after kImpulseTick
// reconciles it.
static constexpr int        kImpulseTick = 60;                  // documented server-only event tick
static const math::Vec3     kServerImpulse{0.0f, 0.0f, 2.4f};   // fixed lateral (+Z) velocity bump

struct Authority {
    physics::World  world;
    game::GameState gs;
    Authority() { gs = game::MakeRollGame(world); }

    // Advance one authoritative step: apply the SERVER-ONLY impulse if this is the impulse step, then
    // StepGame with the (client-known) scripted input. `stepIndex` is the step being executed (0-based);
    // the post-step state is "tick stepIndex+1".
    void StepAuthority(int stepIndex, const game::GameInput& in) {
        if (stepIndex == kImpulseTick) {
            physics::RigidBody& body = world.bodies[(size_t)gs.playerBodyIndex];
            body.linVel = body.linVel + kServerImpulse;
        }
        game::StepGame(world, gs, in, kDt);
    }
};

int main() {
    std::vector<game::GameInput> track = game::ScriptedTrack();
    const int N = (int)track.size();

    // --- 1. Prediction advances under local input (no snapshot needed). ---------------------------
    {
        net::PredictedClient client;
        const math::Vec3 start = client.PlayerPos();
        // Drive +X for a handful of ticks; the predicted player must move +X immediately.
        for (int t = 0; t < 20; ++t) {
            net::InputCmd cmd; cmd.tick = t; cmd.input.moveDir = {1.0f, 0.0f, 0.0f};
            client.PredictTick(cmd);
        }
        const math::Vec3 now = client.PlayerPos();
        check(now.x > start.x + 0.05f, "prediction advances: predicted player moved +X under local input");
        check(client.PredictedTick() == 19, "prediction advances: PredictedTick tracks the last input tick");
        check(client.PendingInputs() == 20, "prediction advances: all inputs buffered (unacknowledged)");
        check(client.Reconciles() == 0, "prediction advances: no reconcile happened yet");
    }

    // --- 2. Reconcile with NO misprediction leaves the predicted "now" unchanged. -----------------
    {
        // Authority runs the IDENTICAL sim (NO server-only impulse) so the client predicts exactly.
        physics::World aw; game::GameState ags = game::MakeRollGame(aw);
        net::PredictedClient client;
        const int upto = 40;
        // Predict ticks 0..upto-1 locally; step the (impulse-free) authority in lockstep.
        for (int t = 0; t < upto; ++t) {
            net::InputCmd cmd; cmd.tick = t; cmd.input = track[(size_t)t];
            client.PredictTick(cmd);
            game::StepGame(aw, ags, track[(size_t)t], kDt);
        }
        // Reconcile against an EARLIER authoritative frame (tick 30, state after 30 steps). Capture the
        // authority's state at tick 30 by re-running it from scratch to exactly 30 steps.
        physics::World aw30; game::GameState ags30 = game::MakeRollGame(aw30);
        for (int t = 0; t < 30; ++t) game::StepGame(aw30, ags30, track[(size_t)t], kDt);
        net::AuthState auth = net::AuthState::Capture(30, ags30, aw30);

        const math::Vec3 before = client.PlayerPos();
        client.OnAuthoritative(auth);
        const math::Vec3 after = client.PlayerPos();
        check(client.LastMisprediction() < 1e-4f,
              "reconcile (no misprediction): lastMisprediction ~= 0");
        check(math::length(after - before) < 1e-4f,
              "reconcile (no misprediction): predicted now UNCHANGED by replay");
        // After dropping acked inputs (tick < 30) the buffer holds exactly ticks 30..39 (10 inputs).
        check(client.PendingInputs() == 10,
              "reconcile (no misprediction): acked inputs dropped, unacked retained");
        check(client.Reconciles() == 1, "reconcile (no misprediction): one reconcile counted");
    }

    // --- 3. Reconcile WITH a server-only misprediction corrects to the authority's TRUE state. -----
    {
        Authority auth;            // runs the impulse-bearing authority
        net::PredictedClient client;
        // Predict + step the authority in lockstep through a window straddling the impulse.
        const int upto = kImpulseTick + 30;   // 90: well past the impulse so the divergence is real
        for (int t = 0; t < upto; ++t) {
            net::InputCmd cmd; cmd.tick = t; cmd.input = track[(size_t)t];
            client.PredictTick(cmd);
            auth.StepAuthority(t, track[(size_t)t]);
        }
        // The authoritative frame the client will reconcile to: state AFTER kImpulseTick+5 steps (the
        // impulse is already baked into the authoritative body's position + velocity).
        const int ackTick = kImpulseTick + 5;
        Authority ref;             // independent authority re-run to exactly ackTick steps
        for (int t = 0; t < ackTick; ++t) ref.StepAuthority(t, track[(size_t)t]);
        net::AuthState reconcileFrame = net::AuthState::Capture(ackTick, ref.gs, ref.world);

        // Independent "true" predicted-now: the authority's state after ALL `upto` steps (what the
        // client SHOULD converge to once it reconciles + replays its unacked inputs).
        Authority truth;
        for (int t = 0; t < upto; ++t) truth.StepAuthority(t, track[(size_t)t]);
        const math::Vec3 truePos =
            truth.world.bodies[(size_t)truth.gs.playerBodyIndex].position;

        const math::Vec3 mispredicted = client.PlayerPos();
        check(math::length(mispredicted - truePos) > 0.05f,
              "reconcile (misprediction): BEFORE reconcile predicted differs from authority's true state");

        client.OnAuthoritative(reconcileFrame);
        const math::Vec3 corrected = client.PlayerPos();
        check(client.LastMisprediction() > 0.0f,
              "reconcile (misprediction): lastMisprediction > 0 (a real correction occurred)");
        check(client.MaxMisprediction() > 0.0f,
              "reconcile (misprediction): maxMisprediction > 0");
        // After reconcile + replay the predicted "now" lands on the authority's TRUE state, bit-close.
        check(math::length(corrected - truePos) < 1e-3f,
              "reconcile (misprediction): AFTER reconcile predicted now == authority's TRUE corrected state");
        // Acked inputs (tick < ackTick) dropped; unacked (ackTick..upto-1) retained = upto - ackTick.
        check((int)client.PendingInputs() == upto - ackTick,
              "reconcile (misprediction): buffer holds exactly the unacknowledged inputs");
    }

    // --- 4. Input buffer: bounded growth + ordered replay over a streaming reconcile. -------------
    {
        Authority authSim;
        net::PredictedClient client;
        size_t maxPending = 0;
        // Stream: predict each tick; every 4 ticks reconcile to an authority frame a few ticks behind
        // (modeling latency). The buffer must stay BOUNDED (it never accumulates the whole run).
        const int latency = 6;
        for (int t = 0; t < 200; ++t) {
            net::InputCmd cmd; cmd.tick = t; cmd.input = track[(size_t)t];
            client.PredictTick(cmd);
            authSim.StepAuthority(t, track[(size_t)t]);
            if (t >= latency && (t % 4 == 0)) {
                const int ackTick = t - latency;  // authoritative state from `latency` ticks ago
                Authority ref;
                for (int k = 0; k < ackTick; ++k) ref.StepAuthority(k, track[(size_t)k]);
                client.OnAuthoritative(net::AuthState::Capture(ackTick, ref.gs, ref.world));
            }
            if (client.PendingInputs() > maxPending) maxPending = client.PendingInputs();
        }
        // The pending buffer never exceeds ~latency + the reconcile stride — NOT the 200-tick run.
        check(maxPending <= (size_t)(latency + 8),
              "input buffer: pending stays bounded (acked inputs dropped, no unbounded growth)");
        check(client.PendingInputs() <= (size_t)(latency + 8),
              "input buffer: final pending bounded");
    }

    // --- 5. Convergence + determinism over the FULL run with the server-only impulse. -------------
    // A full predict+reconcile run with a fixed latency. The reconciled client must converge to the
    // authority at the end (finalError small) AND a real misprediction must occur (maxMisprediction>0).
    // Two identical runs must produce byte-identical diagnostics + final positions.
    auto runFull = [&](float& finalError, float& maxMis, int& reconciles) {
        Authority authSim;
        net::PredictedClient client;
        const int latency = 5;
        for (int t = 0; t < N; ++t) {
            net::InputCmd cmd; cmd.tick = t; cmd.input = track[(size_t)t];
            client.PredictTick(cmd);
            authSim.StepAuthority(t, track[(size_t)t]);
            if (t >= latency) {
                const int ackTick = t - latency;
                Authority ref;
                for (int k = 0; k < ackTick; ++k) ref.StepAuthority(k, track[(size_t)k]);
                client.OnAuthoritative(net::AuthState::Capture(ackTick, ref.gs, ref.world));
            }
        }
        // Final convergence: the predicted player vs the authority's TRUE final state.
        const math::Vec3 cp = client.PlayerPos();
        const math::Vec3 ap = authSim.world.bodies[(size_t)authSim.gs.playerBodyIndex].position;
        finalError = math::length(cp - ap);
        maxMis = client.MaxMisprediction();
        reconciles = (int)client.Reconciles();
    };

    float fe1 = 0, mm1 = 0; int rc1 = 0;
    float fe2 = 0, mm2 = 0; int rc2 = 0;
    runFull(fe1, mm1, rc1);
    runFull(fe2, mm2, rc2);

    check(mm1 > 0.0f, "convergence: a real misprediction occurred (maxMisprediction > 0)");
    // finalError is small: the client trails the authority by `latency` unreconciled ticks at the very
    // end (the last few inputs were never acked), so it is within a few ticks of player motion.
    check(fe1 < 0.10f, "convergence: reconciled client converges to authority (finalError small)");
    check(rc1 > 0, "convergence: reconciliations happened");
    // Determinism: two runs byte-identical (raw float bits).
    auto bitsEq = [](float a, float b) {
        uint32_t ua, ub; std::memcpy(&ua, &a, 4); std::memcpy(&ub, &b, 4); return ua == ub;
    };
    check(bitsEq(fe1, fe2) && bitsEq(mm1, mm2) && rc1 == rc2,
          "determinism: two full runs byte-identical (finalError, maxMisprediction, reconciles)");

    if (g_fail == 0) std::printf("prediction_test: OK\n");
    else             std::printf("prediction_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
