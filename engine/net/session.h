// Generic deterministic SESSION core (Slice NS1, flagship #24 NETCODE, beachhead).
//
// The irreducible rollback-netcode primitive: a header-only, transport-agnostic, templated
// Session<World,Input> that drives N ticks of a caller-supplied deterministic Step from a per-tick
// input ring and exposes an FNV-1a-64 state DIGEST. This generalizes the copy-pasted FPX5-style
// RunLockstep (the per-sim lockstep harnesses) into ONE parameterized engine: two peers fed the
// same inputs re-derive a bit-identical world EVERY tick (the lockstep invariant) and a hard-pinned
// final digest. Pure-CPU INTEGER — no float, no <cmath>, no clock/RNG, no GPU.
//
// SELF-CONTAINED ON PURPOSE: this header includes ONLY <cstdint>/<vector>/<cstddef> (NO fpx / mixer
// / RHI) so it compiles STANDALONE with `clang++ -std=c++20 -I engine -I tests tests/session_test.cpp`
// on the Mac (like audio/dsp.h) — the cheap cross-platform proof. World/Input are template params;
// the deterministic Step/Digest are supplied by the caller as template callables (NO <functional>).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hf::net {

// --- DigestBytes: FNV-1a-64 over n raw bytes (the generic state-digest currency) --------------------
// Reuses the engine-wide FNV-1a-64 constants (offset basis 1469598103934665603 / prime 1099511628211 —
// the SAME FNV as audio/dsp.h::DigestBuffer / ai.h DigestBlackboard). Hashing byte-by-byte keeps the
// digest stable; two equal byte spans hash IDENTICALLY, a single changed byte changes the digest. The
// per-sim harnesses only memcmp; a PINNED uint64_t golden is NS1's new contribution.
inline uint64_t DigestBytes(const void* data, std::size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    uint64_t h = 1469598103934665603ull;           // the FNV-1a 64-bit offset basis
    for (std::size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ull;                      // the FNV-1a 64-bit prime
    }
    return h;
}

// --- InputRing: the per-tick input store, queried by tick index -------------------------------------
// A simple growable structure (std::vector<std::vector<Input>> indexed by tick). Inputs at the SAME
// tick are kept in INSERTION ORDER (the deterministic in-array application order — the SimTick
// contract). At(tick) returns an empty vector for a tick with no inputs (deterministic).
template <class Input>
struct InputRing {
    std::vector<std::vector<Input>> byTick;   // byTick[t] = inputs applied on tick t (insertion order)

    // Append an input for `tick`, growing to cover the tick. Same-tick inputs stay in insertion order.
    void AddInput(uint32_t tick, const Input& in) {
        if (static_cast<std::size_t>(tick) >= byTick.size())
            byTick.resize(static_cast<std::size_t>(tick) + 1);
        byTick[static_cast<std::size_t>(tick)].push_back(in);
    }

    // Inputs applied on `tick` — an empty vector for a tick with no inputs (deterministic).
    const std::vector<Input>& At(uint32_t tick) const {
        static const std::vector<Input> kEmpty{};
        if (static_cast<std::size_t>(tick) >= byTick.size()) return kEmpty;
        return byTick[static_cast<std::size_t>(tick)];
    }
};

// --- Session: the generic deterministic state + its input ring + the next tick ----------------------
template <class World, class Input>
struct Session {
    World            world;     // the deterministic state
    InputRing<Input> ring;      // inputs by tick
    uint32_t         tick = 0;  // the next tick to step
    uint32_t         delay = 0; // NS2: input-delay — local input applies `delay` ticks ahead (0 = NS1)
};

// Advance one tick: apply the deterministic transition step(world, inputs-this-tick, tick), then ++tick.
// `step` is the SimTick / ApplyCommand+integrate analog — pure of its inputs (no hidden global state).
template <class World, class Input, class StepFn>
void Advance(Session<World, Input>& s, StepFn step) {
    step(s.world, s.ring.At(s.tick), s.tick);
    ++s.tick;
}

// One-shot reference: run `ticks` advances from `init` over a COPY of `ring`, return the final digest.
// The generalized fpx-style RunLockstep — deterministic of (init, ring, ticks, step, digest) alone, so
// two peers / two calls produce the IDENTICAL uint64_t.
template <class World, class Input, class StepFn, class DigestFn>
uint64_t RunLockstep(World init, const InputRing<Input>& ring, uint32_t ticks,
                     StepFn step, DigestFn digest) {
    Session<World, Input> s;
    s.world = init;
    s.ring  = ring;     // a COPY — the caller's ring is untouched
    s.tick  = 0;
    for (uint32_t t = 0; t < ticks; ++t) Advance(s, step);
    return digest(s.world);
}

// ============================ NS2: INPUT-DELAY BUFFER ================================================
// The production knob NS1's raw lockstep lacks: a configurable INPUT DELAY. Local input is scheduled to
// apply `s.delay` ticks in the FUTURE instead of on the current tick. In netcode this gives the remote
// peer's (network-travelled) input time to arrive BEFORE its apply-tick, so the common case needs no
// rollback. The delay lives ENTIRELY in the submit->schedule mapping below — Advance is UNCHANGED (it
// still pulls ring.At(tick)) — which is exactly why delay is a pure SCHEDULING SHIFT: it changes WHEN
// inputs apply, never the game transition. A delay-D session whose inputs are submitted D ticks early is
// therefore byte-identical to a delay-0 session. Pure-CPU integer (no float/<cmath>/clock/RNG).

// Schedule a local input to apply `s.delay` ticks ahead of the current tick. At delay==0 this lands on
// the current tick (NS1 "apply now"); at delay==D it lands at s.tick + D.
template <class World, class Input>
void SubmitLocalInput(Session<World, Input>& s, const Input& in) {
    s.ring.AddInput(s.tick + s.delay, in);
}

// Schedule an input at an EXPLICIT apply-tick (independent of s.delay) — the reference path / tests.
template <class World, class Input>
void SubmitInputAt(Session<World, Input>& s, uint32_t applyTick, const Input& in) {
    s.ring.AddInput(applyTick, in);
}

// ============================ NS3: PREDICTION + SNAPSHOT RING + ROLLBACK =============================
// THE CRUX of the netcode flagship — a GGPO-class rollback core. NS2's input-delay hides latency when
// the remote input arrives in time; NS3 handles the case where it does NOT: PREDICT the missing remote
// input (reuse the last confirmed remote), advance speculatively, and when the truth arrives for a past
// tick that we MISPREDICTED, ROLL BACK — restore the world from a per-tick SNAPSHOT RING and re-simulate
// forward with the corrected inputs. The make-or-break invariant: a predicted+rolled-back run reaches the
// BIT-IDENTICAL digest of a no-latency authority run, AND a real misprediction actually fired
// (didRollback). This generalizes fpx.h:RunRollback to a rolling snapshot window + arbitrary
// misprediction ticks. The World is copy-restorable (the toy worlds value-copy; a real sim would plug in
// SnapshotWorld/RestoreWorld). Pure-CPU integer — no float/<cmath>/clock/RNG. Input must be
// equality-comparable (to detect a mispredict).
//
// The model: two input streams over T ticks — local[t] (known immediately) and remote[t] (the peer's
// input, known only LAG ticks late: it "arrives" at tick t+LAG). Each tick we step with local[t] + the
// remote we currently BELIEVE (confirmed if arrived, else a prediction = lastConfirmed remote).

template <class World, class Input>
struct RollbackSession {
    World world;                          // current speculative world
    uint32_t tick = 0;                    // next tick to simulate
    uint32_t confirmedThrough = 0;        // all remote inputs < this tick are confirmed
    std::vector<World> snaps;             // snaps[t] = world at the START of tick t (the snapshot ring/log)
    std::vector<Input> appliedRemote;     // the remote input actually APPLIED at each past tick (pred/real)
    std::vector<Input> appliedLocal;      // the local input applied at each past tick (for the replay)
    std::vector<Input> confirmedRemote;   // the real remote inputs received so far (by tick)
    std::vector<bool>  haveRemote;        // whether confirmedRemote[t] has arrived
    Input lastConfirmed{};                // the prediction source (last confirmed remote input)
    bool  didRollback = false;            // set true if any rollback fired (the proof flag)
};

// Grow the per-tick vectors to cover index `t` (snaps/appliedRemote/appliedLocal/confirmedRemote/haveRemote).
template <class World, class Input>
inline void RollbackGrow(RollbackSession<World, Input>& s, uint32_t t) {
    const std::size_t need = static_cast<std::size_t>(t) + 1;
    if (s.snaps.size()           < need) s.snaps.resize(need);
    if (s.appliedRemote.size()   < need) s.appliedRemote.resize(need);
    if (s.appliedLocal.size()    < need) s.appliedLocal.resize(need);
    if (s.confirmedRemote.size() < need) s.confirmedRemote.resize(need);
    if (s.haveRemote.size()      < need) s.haveRemote.resize(need, false);
}

// StepPredicted: advance ONE tick speculatively. Snapshot the world at the start of the tick, pick the
// remote input (confirmed if it has arrived for this tick, else PREDICT = lastConfirmed), record both
// applied inputs, step over BOTH inputs in a fixed order (local then remote), and ++tick.
template <class World, class Input, class StepFn>
void StepPredicted(RollbackSession<World, Input>& s, const Input& localThisTick, StepFn step) {
    const uint32_t t = s.tick;
    RollbackGrow(s, t);
    s.snaps[t] = s.world;                                   // snapshot the START-of-tick world
    Input r = s.haveRemote[t] ? s.confirmedRemote[t] : s.lastConfirmed;   // confirmed else PREDICT
    s.appliedRemote[t] = r;
    s.appliedLocal[t]  = localThisTick;
    step(s.world, { localThisTick, r }, t);                // deterministic step over BOTH inputs (fixed order)
    ++s.tick;
}

// ConfirmRemote: the real remote input for tick `at` arrives. Record it, extend the confirmed prefix
// (advancing confirmedThrough/lastConfirmed over every contiguous arrived tick). If `at` was already
// simulated AND we mispredicted it (applied != real), ROLL BACK: restore snaps[at] and re-simulate
// at..tick-1 with the now-corrected remote inputs, re-snapshotting as we go. tick is unchanged after.
template <class World, class Input, class StepFn>
void ConfirmRemote(RollbackSession<World, Input>& s, uint32_t at, const Input& real, StepFn step) {
    RollbackGrow(s, at);
    s.confirmedRemote[at] = real;
    s.haveRemote[at]      = true;

    // Detect a misprediction of an ALREADY-simulated tick BEFORE we mutate lastConfirmed below.
    const bool mispredicted =
        (at < s.tick) && (s.appliedRemote[at] != real);

    // Extend the confirmed prefix: advance confirmedThrough over every contiguous arrived tick, tracking
    // the last confirmed remote (the prediction source for future / replayed ticks).
    while (s.confirmedThrough < s.haveRemote.size() &&
           s.haveRemote[s.confirmedThrough]) {
        s.lastConfirmed = s.confirmedRemote[s.confirmedThrough];
        ++s.confirmedThrough;
    }

    if (!mispredicted) return;

    // ROLLBACK: restore the world to the start of tick `at` and re-simulate forward to the current tick.
    s.world = s.snaps[at];
    s.didRollback = true;
    for (uint32_t u = at; u < s.tick; ++u) {
        s.snaps[u] = s.world;                              // re-snapshot the corrected START-of-tick world
        Input r = s.haveRemote[u] ? s.confirmedRemote[u] : s.lastConfirmed;
        s.appliedRemote[u] = r;
        step(s.world, { s.appliedLocal[u], r }, u);        // replay with the SAME local, corrected remote
    }
    // s.tick is unchanged — we re-reached the same tick with the corrected world.
}

// ============================ NS4: TRANSPORT-AGNOSTIC INJECTED INTERFACE =============================
// NS3 proved rollback CORRECTNESS given a clean (origin+LAG) arrival pattern. NS4 makes the session
// TRANSPORT-AGNOSTIC: remote inputs arrive through an INJECTED packet interface so a TEST can script a
// fully deterministic ADVERSARIAL delivery schedule — delay, reorder, and transient loss-with-resend —
// with NO real sockets. The transport models only the DELIVERY SCHEDULE (which remote `forTick` lands at
// which `deliverTick`); the session reuses NS3's StepPredicted/ConfirmRemote UNCHANGED (ConfirmRemote
// already tolerates arbitrary `at` order + duplicate re-deliveries safely). The make-or-break invariant:
// under a scripted loss+reorder+delay schedule the session STILL converges to the SAME pinned digest as
// the clean no-latency authority run — rollback correctness under adversity. Pure-CPU integer (no
// float/<cmath>/clock/RNG); the "adversity" is a FIXED scripted permutation, never random.

// An in-flight remote-input packet: the remote input for `forTick` that the transport delivers at
// `deliverTick`. deliverTick >= forTick models LATENCY; a non-monotonic deliverTick vs forTick (a later
// forTick delivered before an earlier one) models REORDER; two packets with the same forTick model a
// RESEND (a redundant copy — harmless because ConfirmRemote of an already-confirmed tick is a no-op).
template <class Input>
struct InflightPacket {
    uint32_t deliverTick = 0;   // the tick at which this packet "arrives" at the receiver
    uint32_t forTick     = 0;   // the origin tick of the remote input it carries
    Input    input{};           // the remote input value for forTick
};

// A scripted, deterministic transport: an ordered list of in-flight packets (a fixed permutation the
// TEST authors). `cursor` is reserved for an in-order walk; the deterministic delivery path below uses a
// per-tick linear scan (safe even when deliverTicks are scheduled non-monotonically).
template <class Input>
struct ScriptedTransport {
    std::vector<InflightPacket<Input>> sched;   // the scripted delivery schedule (insertion order)
    std::size_t                        cursor = 0;
};

// Schedule a remote input `in` (origin `forTick`) to be delivered at `deliverTick`. Appends to sched, so
// the insertion order IS the per-tick delivery order (Deliver preserves it). Call it as many times as the
// schedule needs — including multiple packets for one forTick (a resend) or out-of-order deliverTicks.
template <class Input>
void Schedule(ScriptedTransport<Input>& tx, uint32_t deliverTick, uint32_t forTick, const Input& in) {
    tx.sched.push_back(InflightPacket<Input>{ deliverTick, forTick, in });
}

// Deliver every packet whose deliverTick == currentTick, returned in scheduled (insertion) order. A
// per-tick linear scan over the whole schedule is the safe deterministic form: the test may schedule
// non-monotonic deliverTicks (reorder), so a cursor walk could miss/misorder them. Deterministic of
// (tx.sched, currentTick) alone — two identical schedules deliver byte-identically.
template <class Input>
std::vector<InflightPacket<Input>> Deliver(ScriptedTransport<Input>& tx, uint32_t currentTick) {
    std::vector<InflightPacket<Input>> out;
    for (std::size_t i = 0; i < tx.sched.size(); ++i)
        if (tx.sched[i].deliverTick == currentTick) out.push_back(tx.sched[i]);
    return out;
}

// RunWithTransport: drive an NS3 RollbackSession through `totalTicks` with remote inputs arriving via the
// scripted transport instead of a fixed LAG. For each tick t: step speculatively with local[t] (predicting
// any not-yet-arrived remote), THEN deliver every packet scheduled for tick t and ConfirmRemote each one
// (which rolls back on a mispredicted past tick). `totalTicks` MUST exceed the last deliverTick so every
// remote input is confirmed before the run ends — otherwise the final world is still speculative. Pure-CPU
// integer; StepPredicted/ConfirmRemote are the NS3 primitives, reused verbatim.
template <class World, class Input, class StepFn>
void RunWithTransport(RollbackSession<World, Input>& s, const std::vector<Input>& local,
                      ScriptedTransport<Input>& tx, uint32_t totalTicks, StepFn step) {
    for (uint32_t t = 0; t < totalTicks; ++t) {
        StepPredicted(s, local[static_cast<std::size_t>(t)], step);   // speculative tick (predict missing remote)
        std::vector<InflightPacket<Input>> arrivals = Deliver(tx, t); // packets that land THIS tick (in order)
        for (std::size_t i = 0; i < arrivals.size(); ++i)
            ConfirmRemote(s, arrivals[i].forTick, arrivals[i].input, step);  // confirm (+ rollback if mispredicted)
    }
}

}  // namespace hf::net
