// Unit test for the GENERIC deterministic SESSION core (engine/net/session.h, Slice NS1, flagship #24
// NETCODE, beachhead). Pure CPU (hf_core), ASan-eligible like the other pure tests.
//
// session.h is templated on World/Input with the deterministic Step/Digest supplied by the caller, so
// it is driven here by TWO trivial SELF-CONTAINED integer TOY worlds (NOT fpx — proving the abstraction
// is generic without pulling a heavy dependency; the real-sim drivers live in a later non-standalone
// path). Everything is INTEGER, so the digests are bit-identical run-to-run AND platform-to-platform
// (MSVC vs Apple clang). The golden is a PINNED FNV-1a-64 DigestBytes value IN the test (NO image, NO
// render-bake) — the cross-platform proof is the SAME CPU test producing the IDENTICAL hash on both.
//
// What this pins:
//   * 2-PEER LOCKSTEP (make-or-break) — two independent Sessions from the same init over the same ring,
//     advanced in lockstep, have EQUAL digests at EVERY tick (not just the end) — the lockstep invariant.
//   * PINNED DIGEST — the converged final digest of ToyA (and ToyB) == a hard-pinned uint64_t.
//   * INPUT-ORDER DETERMINISM — two inputs on the same tick in insertion order give a deterministic
//     result; REVERSING the insertion order changes the digest (order is load-bearing + deterministic).
//   * REPLAY-STABLE — two RunLockstep over the same ring are bit-identical.
//   * GENERIC — both ToyA and ToyB drive the SAME Session/Advance/RunLockstep code.

#include "net/session.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::net;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// ---- ToyA: a scalar accumulator. Input=int32 folded with the tick so order/tick MATTER. -------------
struct ToyA { int64_t acc = 0; };
using InA = int32_t;
static void StepA(ToyA& w, const std::vector<InA>& inputs, uint32_t tick) {
    for (const InA in : inputs)
        w.acc += static_cast<int64_t>(in) * static_cast<int64_t>(tick + 1);
}
static uint64_t DigestA(const ToyA& w) { return DigestBytes(&w.acc, sizeof w.acc); }

// ---- ToyB: a cell array. Input={idx,delta}; each input adds delta to cells[idx % size]. -------------
struct ToyB { std::vector<int32_t> cells; };
struct InB  { int idx; int32_t delta; };
static void StepB(ToyB& w, const std::vector<InB>& inputs, uint32_t /*tick*/) {
    if (w.cells.empty()) return;
    for (const InB& in : inputs)
        w.cells[static_cast<std::size_t>(in.idx) % w.cells.size()] += in.delta;
}
static uint64_t DigestB(const ToyB& w) {
    return DigestBytes(w.cells.data(), w.cells.size() * sizeof(int32_t));
}

int main() {
    HF_TEST_MAIN_INIT();
    const uint32_t kTicks = 16;

    // A fixed deterministic ToyA input ring (several ticks, some with multiple same-tick inputs).
    auto makeRingA = []() -> InputRing<InA> {
        InputRing<InA> r;
        r.AddInput(0,  5);
        r.AddInput(1,  3); r.AddInput(1, -2);     // two inputs on tick 1 (insertion order matters)
        r.AddInput(3,  7);
        r.AddInput(7, 11); r.AddInput(7,  4); r.AddInput(7, -9);  // three on tick 7
        r.AddInput(10, 2);
        r.AddInput(15, 6);
        return r;
    };

    // ---- (1) 2-PEER LOCKSTEP (make-or-break): two independent Sessions, equal digest EVERY tick. ----
    bool everyTickEqual = true;
    {
        const InputRing<InA> ring = makeRingA();
        Session<ToyA, InA> peerA, peerB;
        peerA.world = ToyA{}; peerA.ring = ring; peerA.tick = 0;
        peerB.world = ToyA{}; peerB.ring = ring; peerB.tick = 0;
        // Digest equal at the START (both fresh).
        if (DigestA(peerA.world) != DigestA(peerB.world)) everyTickEqual = false;
        for (uint32_t t = 0; t < kTicks; ++t) {
            Advance(peerA, StepA);   // peer A and peer B run the SAME deterministic Step
            Advance(peerB, StepA);
            if (DigestA(peerA.world) != DigestA(peerB.world)) everyTickEqual = false;
            if (peerA.tick != peerB.tick) everyTickEqual = false;
        }
        check(everyTickEqual, "ns1: 2-peer lockstep — digests EQUAL at every tick");
    }

    // ---- (2) PINNED DIGEST — converged final digest of ToyA (and ToyB). -----------------------------
    const uint64_t hToyA = RunLockstep<ToyA, InA>(ToyA{}, makeRingA(), kTicks, StepA, DigestA);

    // A fixed deterministic ToyB input ring over an 8-cell world.
    auto makeRingB = []() -> InputRing<InB> {
        InputRing<InB> r;
        r.AddInput(0,  InB{0,  10});
        r.AddInput(1,  InB{3,  -4}); r.AddInput(1, InB{3, 7});   // two on tick 1, same cell
        r.AddInput(2,  InB{9,   5});                              // idx 9 % 8 == cell 1
        r.AddInput(5,  InB{2, 100}); r.AddInput(5, InB{6, -50});
        r.AddInput(11, InB{7,   3});
        r.AddInput(14, InB{1,  -8});
        return r;
    };
    const ToyB initB{std::vector<int32_t>(8, 0)};
    const uint64_t hToyB = RunLockstep<ToyB, InB>(initB, makeRingB(), kTicks, StepB, DigestB);

    // Pinned goldens (computed on first run, hardcoded — the regression anchor / cross-platform bar).
    const uint64_t kPinnedToyA = 0x6227bc7b4046d08aull; // the converged ToyA digest (regression anchor)
    const uint64_t kPinnedToyB = 0x92c15ff1bf8f538eull; // the converged ToyB digest (regression anchor)
    check(hToyA == kPinnedToyA, "ns1: pinned digest: ToyA converged digest matches golden");
    check(hToyB == kPinnedToyB, "ns1: pinned digest: ToyB converged digest matches golden");
    check(hToyA != hToyB, "ns1: ToyA and ToyB digests differ (distinct worlds)");

    // ---- (3) INPUT-ORDER DETERMINISM — same-tick insertion order is load-bearing + deterministic. ---
    uint64_t hInOrder = 0, hReversed = 0;
    {
        // Two inputs on the SAME tick; ToyB at the SAME cell with asymmetric deltas (so the intermediate
        // matters only via determinism, but reversing the order is the load-bearing proof of ordering).
        // Use ToyA where the per-input fold is sum-commutative? No — pick a world where order changes the
        // result. ToyB additive at one cell is order-INDEPENDENT, so use a fold that is order-sensitive:
        // apply each input as cells[c] = cells[c]*3 + delta (a Horner-style fold) so order matters.
        struct ToyC { std::vector<int32_t> cells; };
        using InC = int32_t;
        auto stepC = [](ToyC& w, const std::vector<InC>& ins, uint32_t) {
            for (const InC d : ins) w.cells[0] = w.cells[0] * 3 + d;   // non-commutative fold
        };
        auto digestC = [](const ToyC& w) {
            return DigestBytes(w.cells.data(), w.cells.size() * sizeof(int32_t));
        };
        const ToyC initC{std::vector<int32_t>(1, 0)};

        InputRing<InC> inOrder;
        inOrder.AddInput(0, 5); inOrder.AddInput(0, 7);   // [5, 7]
        hInOrder = RunLockstep<ToyC, InC>(initC, inOrder, 1, stepC, digestC);

        InputRing<InC> reversed;
        reversed.AddInput(0, 7); reversed.AddInput(0, 5); // [7, 5] — reversed insertion order
        hReversed = RunLockstep<ToyC, InC>(initC, reversed, 1, stepC, digestC);

        check(hInOrder != hReversed, "ns1: input-order determinism — reversing same-tick order changes digest");
        // And deterministic: re-running the same order reproduces the digest.
        const uint64_t hInOrder2 = RunLockstep<ToyC, InC>(initC, inOrder, 1, stepC, digestC);
        check(hInOrder == hInOrder2, "ns1: input-order: same order reproduces the digest (deterministic)");
    }

    // ---- (4) REPLAY-STABLE — two RunLockstep over the same ring are bit-identical. -------------------
    {
        const uint64_t a = RunLockstep<ToyA, InA>(ToyA{}, makeRingA(), kTicks, StepA, DigestA);
        const uint64_t b = RunLockstep<ToyB, InB>(initB,  makeRingB(), kTicks, StepB, DigestB);
        check(a == hToyA, "ns1: replay-stable — ToyA two RunLockstep bit-identical");
        check(b == hToyB, "ns1: replay-stable — ToyB two RunLockstep bit-identical");
    }

    // ===================================== NS2: INPUT-DELAY BUFFER ===================================
    // The NS2 contribution: a per-session input DELAY. Local input scheduled via SubmitLocalInput lands
    // `s.delay` ticks in the future. We prove delay is a PURE SCHEDULING SHIFT (it changes WHEN inputs
    // apply, not the game transition) by showing a delay-D session whose inputs are submitted D ticks
    // EARLY is byte-identical to a delay-0 reference. ToyA folds input with the tick, so a wrong apply-
    // tick would change the digest — making the equality load-bearing.
    const uint32_t kNs2Ticks = 24;
    const uint32_t kDelta    = 5;   // the input delay D

    // A fixed set of (applyTick, input) events — chosen so applyTick >= kDelta for every event (so the
    // "submitted D early" tick applyTick - kDelta is a valid non-negative tick we can advance to).
    struct Ev { uint32_t applyTick; InA in; };
    const Ev kEvents[] = {
        { 5,  9}, { 6, -4}, { 6,  2},   // two events apply on tick 6 (insertion order preserved)
        {10,  7}, {13, -3}, {18, 12},
        {20,  1}, {23,  5},
    };
    const std::size_t kNev = sizeof(kEvents) / sizeof(kEvents[0]);

    // ---- (NS2-1) DELAY IS A PURE SCHEDULING SHIFT (make-or-break) -----------------------------------
    // REFERENCE: delay-0 session, events scheduled at their explicit applyTick via SubmitInputAt.
    uint64_t hRef = 0;
    {
        Session<ToyA, InA> s;
        s.world = ToyA{}; s.tick = 0; s.delay = 0;
        for (std::size_t e = 0; e < kNev; ++e)
            SubmitInputAt(s, kEvents[e].applyTick, kEvents[e].in);
        for (uint32_t t = 0; t < kNs2Ticks; ++t) Advance(s, StepA);
        hRef = DigestA(s.world);
    }

    // DELAYED: delay-DELTA session, the SAME events submitted via SubmitLocalInput at tick
    // (applyTick - DELTA). We step the session forward and submit each event exactly when the current
    // tick reaches its (applyTick - DELTA), so SubmitLocalInput (= AddInput(tick + delay)) lands it on
    // applyTick. Events are pre-sorted by applyTick (the array already is).
    uint64_t hDelayed = 0;
    {
        Session<ToyA, InA> s;
        s.world = ToyA{}; s.tick = 0; s.delay = kDelta;
        std::size_t e = 0;
        for (uint32_t t = 0; t < kNs2Ticks; ++t) {
            // Submit every event whose submit-tick (applyTick - delta) is the current tick.
            while (e < kNev && (kEvents[e].applyTick - kDelta) == s.tick) {
                SubmitLocalInput(s, kEvents[e].in);   // lands at s.tick + kDelta == applyTick
                ++e;
            }
            Advance(s, StepA);
        }
        hDelayed = DigestA(s.world);
        check(e == kNev, "ns2: all events were submitted in the delayed run");
    }
    check(hDelayed == hRef, "ns2: delay-D (submitted D early) == delay-0 BYTE-IDENTICAL (pure shift)");

    // ---- (NS2-2) DELAY ACTUALLY DELAYS — no effect before applyTick, diverges at applyTick. ---------
    // A single input submitted on tick t0 in a delay-DELTA session must NOT change the world for ticks
    // [t0, t0+DELTA) (vs a no-input run) and MUST change it at tick t0+DELTA (the input applied).
    bool delayActuallyDelays = true;
    {
        const uint32_t t0 = 3;
        const InA      kIn = 42;
        // Baseline: a no-input delay-DELTA session — record its per-tick digest.
        // Delayed:   identical but we SubmitLocalInput(kIn) right after reaching tick t0.
        Session<ToyA, InA> base, del;
        base.world = ToyA{}; base.tick = 0; base.delay = kDelta;
        del.world  = ToyA{}; del.tick  = 0; del.delay  = kDelta;
        for (uint32_t t = 0; t < kNs2Ticks; ++t) {
            if (del.tick == t0) SubmitLocalInput(del, kIn);  // lands at t0 + kDelta
            Advance(base, StepA);
            Advance(del,  StepA);
            const uint32_t now = base.tick;  // == del.tick (both just advanced); world AFTER tick now-1
            const bool eq = (DigestA(base.world) == DigestA(del.world));
            // For every completed tick strictly before t0+kDelta, the worlds must still be EQUAL.
            if (now <= t0 + kDelta) { if (!eq) delayActuallyDelays = false; }
            // Once we've completed the apply-tick (t0+kDelta), the worlds must DIVERGE.
            if (now == t0 + kDelta + 1) { if (eq) delayActuallyDelays = false; }
        }
    }
    check(delayActuallyDelays, "ns2: delay actually delays (no effect before applyTick, diverges at it)");

    // ---- (NS2-3) PINNED DIGEST — the reference converged digest == a hard-pinned uint64_t. ----------
    const uint64_t kPinnedNs2 = 0x480eb38d762ace44ull; // the converged delay-0 reference digest (anchor)
    check(hRef == kPinnedNs2, "ns2: pinned digest: delay-0 reference converged digest matches golden");

    // ---- (NS2-4) DELAY=0 == NS1 — a delay-0 SubmitLocalInput at tick t lands on tick t. -------------
    // SubmitLocalInput on a delay-0 session must be equivalent to the NS1 direct AddInput(t, ...).
    bool delay0IsNs1 = true;
    {
        const uint32_t kSmall = 8;
        // Via NS1 direct AddInput.
        Session<ToyA, InA> a; a.world = ToyA{}; a.tick = 0; a.delay = 0;
        a.ring.AddInput(2, 11); a.ring.AddInput(2, -3); a.ring.AddInput(5, 8);
        for (uint32_t t = 0; t < kSmall; ++t) Advance(a, StepA);
        // Via delay-0 SubmitLocalInput, advancing to each tick and submitting there.
        Session<ToyA, InA> b; b.world = ToyA{}; b.tick = 0; b.delay = 0;
        for (uint32_t t = 0; t < kSmall; ++t) {
            if (b.tick == 2) { SubmitLocalInput(b, 11); SubmitLocalInput(b, -3); }
            if (b.tick == 5) { SubmitLocalInput(b, 8); }
            Advance(b, StepA);
        }
        if (DigestA(a.world) != DigestA(b.world)) delay0IsNs1 = false;
    }
    check(delay0IsNs1, "ns2: delay=0 SubmitLocalInput lands on the current tick (== NS1 AddInput)");

    // ---- Showcase / numeric proof (printed; no image golden). ---------------------------------------
    std::printf("ns1-session: generic lockstep session (2 peers, toy worlds A+B)\n");
    std::printf("ns1-session: 2-peer lockstep — digests equal every tick {ticks:%u}\n", kTicks);
    std::printf("ns1-session: converged digest pinned {toyA:0x%016llx, toyB:0x%016llx}\n",
                static_cast<unsigned long long>(hToyA), static_cast<unsigned long long>(hToyB));
    std::printf("ns1-session: input-order determinism {inOrder:0x%016llx, reversed:0x%016llx} H1 != H2\n",
                static_cast<unsigned long long>(hInOrder), static_cast<unsigned long long>(hReversed));
    std::printf("ns2-delay: input-delay buffer (local input applies `delay` ticks later)\n");
    std::printf("ns2-delay: delay-D (submitted D early) == delay-0 BYTE-IDENTICAL {hash:0x%016llx}\n",
                static_cast<unsigned long long>(hDelayed));
    std::printf("ns2-delay: delay actually delays (no effect before applyTick) {delta:%u, ok:%s}\n",
                kDelta, delayActuallyDelays ? "true" : "false");
    std::printf("ns2-delay: pinned converged digest {hash:0x%016llx}\n",
                static_cast<unsigned long long>(hRef));

    // ================================ NS3: PREDICTION + ROLLBACK =====================================
    // THE CRUX: predict the missing remote input, advance speculatively, roll back from the snapshot ring
    // when the truth arrives and differs. Make-or-break: a predicted+rolled-back run (LAG>0) reaches the
    // BYTE-IDENTICAL digest of a no-latency AUTHORITY run, AND a real misprediction actually fired.
    //
    // Toy world: an int64 accumulator folded ORDER-SENSITIVELY over BOTH inputs (local then remote), so a
    // wrong remote prediction changes the state: acc = acc*K + local*A + remote*B. The remote stream
    // VARIES, so predictions (reuse last confirmed remote) are sometimes wrong -> a real rollback fires.
    using InR = int32_t;
    struct RWorld { int64_t acc = 0; };
    // step over {local, remote} in fixed order — an order-sensitive Horner fold (K!=A!=B, all nonzero).
    auto stepR = [](RWorld& w, std::initializer_list<InR> lr, uint32_t /*tick*/) {
        const InR* p = lr.begin();
        const int64_t local  = static_cast<int64_t>(p[0]);
        const int64_t remote = static_cast<int64_t>(p[1]);
        w.acc = w.acc * 6 + local * 3 + remote * 5;    // K=6, A=3, B=5 — remote is load-bearing
    };
    auto digestR = [](const RWorld& w) { return DigestBytes(&w.acc, sizeof w.acc); };

    const uint32_t kRTicks = 24;
    // Fixed local + remote streams. remote VARIES tick-to-tick (different from the prior confirmed value
    // often enough that prediction-by-reuse mispredicts).
    std::vector<InR> local(kRTicks), remote(kRTicks);
    for (uint32_t t = 0; t < kRTicks; ++t) {
        local[t]  = static_cast<InR>(1 + (t * 7) % 11);          // deterministic, no RNG
        remote[t] = static_cast<InR>(-3 + static_cast<int>((t * 5) % 13) - static_cast<int>(t % 4));
    }

    // AUTHORITY: a no-latency lockstep applying the TRUE {local[t], remote[t]} every tick -> D_auth.
    uint64_t dAuth = 0;
    {
        RWorld w;
        for (uint32_t t = 0; t < kRTicks; ++t) stepR(w, { local[t], remote[t] }, t);
        dAuth = digestR(w);
    }

    // ---- (NS3-1) ROLLBACK CORRECTNESS (make-or-break): LAG>0 predict+rollback == authority. ----------
    const uint32_t kLag = 3;
    uint64_t dRollback = 0;
    bool firedRollback = false;
    {
        RollbackSession<RWorld, InR> s;
        for (uint32_t t = 0; t < kRTicks; ++t) {
            // Deliver any remote whose arrival tick (origin + LAG) is NOW, BEFORE stepping this tick.
            // remote[u] originates at tick u and "arrives" at tick u+kLag.
            if (t >= kLag) ConfirmRemote(s, t - kLag, remote[t - kLag], stepR);
            StepPredicted(s, local[t], stepR);
        }
        // Drain the remaining in-flight confirmations (the last kLag remotes) -> final rollbacks.
        for (uint32_t u = (kRTicks >= kLag ? kRTicks - kLag : 0); u < kRTicks; ++u)
            ConfirmRemote(s, u, remote[u], stepR);
        dRollback     = digestR(s.world);
        firedRollback = s.didRollback;
    }
    check(dRollback == dAuth, "ns3: rollback correctness — predicted+rolled-back == authority BYTE-IDENTICAL");
    check(firedRollback, "ns3: a real misprediction diverged then was corrected (didRollback)");

    // ---- (NS3-2) LAG=0 no-rollback == authority trivially (every input confirmed immediately). -------
    uint64_t dLag0 = 0;
    bool lag0NoRollback = false;
    {
        RollbackSession<RWorld, InR> s;
        for (uint32_t t = 0; t < kRTicks; ++t) {
            ConfirmRemote(s, t, remote[t], stepR);   // confirm BEFORE stepping -> never a prediction
            StepPredicted(s, local[t], stepR);
        }
        dLag0          = digestR(s.world);
        lag0NoRollback = !s.didRollback;
    }
    check(dLag0 == dAuth, "ns3: LAG=0 no-rollback == authority byte-identical");
    check(lag0NoRollback, "ns3: LAG=0 never rolled back (every input confirmed immediately)");

    // ---- (NS3-3) PINNED — D_auth == hard-pinned uint64_t (the cross-platform regression anchor). -----
    const uint64_t kPinnedNs3Auth = 0x1aa9738bcc0c7001ull; // the converged authority digest (anchor)
    check(dAuth == kPinnedNs3Auth, "ns3: pinned authority digest matches golden");

    // ---- NS3 showcase (printed; no image golden). ----------------------------------------------------
    std::printf("ns3-rollback: prediction + snapshot ring + rollback on misprediction\n");
    std::printf("ns3-rollback: rollback correctness — predicted+rolled-back == authority BYTE-IDENTICAL {hash:0x%016llx}\n",
                static_cast<unsigned long long>(dRollback));
    std::printf("ns3-rollback: a real misprediction diverged then was corrected {didRollback:%s}\n",
                firedRollback ? "true" : "false");
    std::printf("ns3-rollback: LAG=0 no-rollback == authority {ok:%s}\n",
                (dLag0 == dAuth && lag0NoRollback) ? "true" : "false");
    std::printf("ns3-rollback: pinned authority digest {hash:0x%016llx}\n",
                static_cast<unsigned long long>(dAuth));

    // ================================ NS4: TRANSPORT-AGNOSTIC + ADVERSARIAL =========================
    // NS4 makes the session transport-agnostic: remote inputs arrive through an INJECTED ScriptedTransport
    // so we can author a deterministic ADVERSARIAL delivery schedule (delay/reorder/loss-with-resend) with
    // NO sockets, reusing the NS3 toy world + local[]/remote[] streams. Make-or-break: under the adversity
    // the session STILL converges to D_auth BYTE-IDENTICAL with didRollback == true.

    // Build a deliberately adversarial schedule that delivers EVERY remote[t] at least once but with
    // delay, reorder, and resend. All deliverTicks are deterministic functions of t (no RNG). We deliver
    // each forTick from a base delay, then sprinkle reordering (some later forTicks land before earlier
    // ones) and resends (a redundant later copy of selected forTicks). The last deliverTick is tracked so
    // we can choose a totalTicks that drains the whole schedule.
    auto buildAdversarial = [&](ScriptedTransport<InR>& tx) -> uint32_t {
        uint32_t lastDeliver = 0;
        for (uint32_t t = 0; t < kRTicks; ++t) {
            // Base delay: a per-tick jitter in [1..4] so deliverTick = t + jitter (always latency).
            const uint32_t jitter = 1 + (t * 3) % 4;
            uint32_t deliver = t + jitter;
            // Reorder: for every 3rd tick, push the delivery LATER so it lands after subsequent ticks'
            // packets — producing non-monotonic deliverTick vs forTick (a later forTick arriving earlier).
            if (t % 3 == 0) deliver += 5;
            Schedule(tx, deliver, t, remote[t]);
            if (deliver > lastDeliver) lastDeliver = deliver;
            // Loss-with-resend: for every 4th tick, schedule a redundant LATER copy (the "resend"). The
            // first copy still lands; the resend must be a harmless no-op once confirmed.
            if (t % 4 == 0) {
                const uint32_t resend = deliver + 6;
                Schedule(tx, resend, t, remote[t]);
                if (resend > lastDeliver) lastDeliver = resend;
            }
        }
        return lastDeliver;
    };

    // ---- (NS4-1) CONVERGENCE UNDER ADVERSITY (make-or-break). ---------------------------------------
    uint64_t dAdversarial = 0;
    bool     ns4Rollback  = false;
    uint32_t ns4Total     = 0;
    {
        ScriptedTransport<InR> tx;
        const uint32_t lastDeliver = buildAdversarial(tx);
        ns4Total = lastDeliver + 2;   // exceed the last deliverTick so every packet drains
        // local[] must cover totalTicks; kRTicks is the original stream length. ns4Total may exceed it,
        // so extend local deterministically for the trailing ticks (no remote origin there — purely to
        // keep stepping; remote for those ticks is predicted but never confirmed beyond kRTicks).
        std::vector<InR> localExt(ns4Total);
        for (uint32_t t = 0; t < ns4Total; ++t)
            localExt[t] = (t < kRTicks) ? local[t] : static_cast<InR>(0);

        // AUTHORITY for the adversarial run must match: the authority world steps the TRUE
        // {localExt[t], remote'[t]} where remote' is the confirmed remote for t (= remote[t] for
        // t < kRTicks) and the prediction-source (lastConfirmed = remote[kRTicks-1]) for trailing ticks,
        // since RunWithTransport never confirms a forTick >= kRTicks. Recompute D_auth over ns4Total.
        uint64_t dAuthExt = 0;
        {
            RWorld w;
            InR lastRemote = InR{};   // lastConfirmed before any confirm (matches RollbackSession default)
            for (uint32_t t = 0; t < ns4Total; ++t) {
                InR rem = (t < kRTicks) ? remote[t] : lastRemote;
                stepR(w, { localExt[t], rem }, t);
                if (t < kRTicks) lastRemote = remote[t];
            }
            dAuthExt = digestR(w);
        }

        RollbackSession<RWorld, InR> s;
        RunWithTransport(s, localExt, tx, ns4Total, stepR);
        dAdversarial = digestR(s.world);
        ns4Rollback  = s.didRollback;

        check(dAdversarial == dAuthExt,
              "ns4: converges under adversity — final == authority BYTE-IDENTICAL");
        check(ns4Rollback, "ns4: adversity forced real rollbacks (didRollback)");
    }

    // ---- (NS4-2) SCHEDULE DETERMINISM — same adversarial schedule twice -> byte-identical. ----------
    bool ns4Deterministic = false;
    {
        ScriptedTransport<InR> tx1, tx2;
        const uint32_t last1 = buildAdversarial(tx1);
        const uint32_t last2 = buildAdversarial(tx2);
        const uint32_t total = (last1 > last2 ? last1 : last2) + 2;
        std::vector<InR> localExt(total);
        for (uint32_t t = 0; t < total; ++t)
            localExt[t] = (t < kRTicks) ? local[t] : static_cast<InR>(0);

        RollbackSession<RWorld, InR> a, b;
        RunWithTransport(a, localExt, tx1, total, stepR);
        RunWithTransport(b, localExt, tx2, total, stepR);
        ns4Deterministic = (digestR(a.world) == digestR(b.world));
        check(ns4Deterministic, "ns4: schedule determinism — same schedule twice byte-identical");
    }

    // ---- (NS4-3) DUPLICATE DELIVERY IS A NO-OP — an extra resend of every forTick doesn't change it. -
    bool ns4DupNoOp = false;
    {
        ScriptedTransport<InR> tx;
        uint32_t lastDeliver = buildAdversarial(tx);
        // Add an EXTRA redundant copy of EVERY forTick, delivered late — pure resends, all must no-op.
        for (uint32_t t = 0; t < kRTicks; ++t) {
            const uint32_t extra = t + kRTicks + 3;   // well after the original delivery
            Schedule(tx, extra, t, remote[t]);
            if (extra > lastDeliver) lastDeliver = extra;
        }
        const uint32_t total = lastDeliver + 2;
        std::vector<InR> localExt(total);
        for (uint32_t t = 0; t < total; ++t)
            localExt[t] = (t < kRTicks) ? local[t] : static_cast<InR>(0);

        // Authority over `total` ticks (same trailing-prediction model as NS4-1).
        uint64_t dAuthExt = 0;
        {
            RWorld w;
            InR lastRemote = InR{};
            for (uint32_t t = 0; t < total; ++t) {
                InR rem = (t < kRTicks) ? remote[t] : lastRemote;
                stepR(w, { localExt[t], rem }, t);
                if (t < kRTicks) lastRemote = remote[t];
            }
            dAuthExt = digestR(w);
        }

        RollbackSession<RWorld, InR> s;
        RunWithTransport(s, localExt, tx, total, stepR);
        ns4DupNoOp = (digestR(s.world) == dAuthExt);
        check(ns4DupNoOp, "ns4: duplicate delivery (resend) is a no-op — still == authority");
    }

    // ---- (NS4-4) PINNED — D_auth (the NS3 authority over the true streams) == hard-pinned uint64_t. --
    // Reuse the NS3 D_auth (kRTicks-length, true {local,remote}) as the pinned authority anchor; the
    // adversarial run's own authority (dAuthExt above) was verified byte-identical to its transported run.
    const uint64_t kPinnedNs4Auth = kPinnedNs3Auth;   // identical streams -> identical authority digest
    check(dAuth == kPinnedNs4Auth, "ns4: pinned authority digest matches golden");

    // ---- NS4 showcase (printed; no image golden). ---------------------------------------------------
    std::printf("ns4-transport: injected transport + adversarial schedule (delay/reorder/resend)\n");
    std::printf("ns4-transport: converges under adversity — final == authority BYTE-IDENTICAL {hash:0x%016llx}\n",
                static_cast<unsigned long long>(dAdversarial));
    std::printf("ns4-transport: adversity forced rollbacks {didRollback:%s}\n",
                ns4Rollback ? "true" : "false");
    std::printf("ns4-transport: schedule determinism — same schedule twice identical {ok:%s}\n",
                ns4Deterministic ? "true" : "false");
    std::printf("ns4-transport: pinned authority digest {hash:0x%016llx}\n",
                static_cast<unsigned long long>(dAuth));

    // ================================ NS5: DESYNC DETECTOR via DIGEST EXCHANGE =======================
    // The safety net: peers exchange per-tick state digests and a mismatch is caught at the EXACT tick.
    // We reuse ToyA (folds input with the tick, so a wrong input at tick K diverges the trace at K and —
    // because the accumulator carries forward — STAYS diverged). Make-or-break: (1) a CLEAN session (both
    // peers same inputs) -> ZERO desync; (2) a CORRUPTED peer A (one extra input at tick K) is detected at
    // the EXACT first diverging tick K with diverging digests; (3) localization is exact (identical < K,
    // differ at K); (4) the clean trace's final digest is pinned.
    const uint32_t kNs5Ticks = 16;
    const uint32_t kK        = 7;     // the chosen corruption tick

    // Peer B's (and the clean peer A's) shared input ring.
    auto makeRingNs5 = []() -> InputRing<InA> {
        InputRing<InA> r;
        r.AddInput(0,  4);
        r.AddInput(2,  9); r.AddInput(2, -1);   // two on tick 2
        r.AddInput(5, 13);
        r.AddInput(7,  6);                       // tick K=7 has a legitimate input in BOTH peers
        r.AddInput(9, -8);
        r.AddInput(12, 3);
        r.AddInput(15, 7);
        return r;
    };

    // Peer A's CORRUPTED ring: identical to B's PLUS one extra input at tick K (so A diverges at K).
    auto makeRingNs5Corrupt = [&]() -> InputRing<InA> {
        InputRing<InA> r = makeRingNs5();
        r.AddInput(kK, 999);   // the desync seed — an extra input only peer A applies at tick K
        return r;
    };

    // Build the two per-tick digest traces (peer B clean, peer A clean for case 1; peer A corrupt case 2).
    const std::vector<uint64_t> traceB     = DigestTrace<ToyA, InA>(ToyA{}, makeRingNs5(),        kNs5Ticks, StepA, DigestA);
    const std::vector<uint64_t> traceAClean= DigestTrace<ToyA, InA>(ToyA{}, makeRingNs5(),        kNs5Ticks, StepA, DigestA);
    const std::vector<uint64_t> traceACorr = DigestTrace<ToyA, InA>(ToyA{}, makeRingNs5Corrupt(), kNs5Ticks, StepA, DigestA);

    // ---- (NS5-1) CLEAN SESSION -> ZERO DESYNC (make-or-break). ---------------------------------------
    bool ns5CleanZeroDesync = false;
    {
        DesyncDetector d;
        for (uint32_t t = 0; t < kNs5Ticks; ++t) RecordLocal(d, t, traceB[t]);          // peer B records ITS trace
        for (uint32_t t = 0; t < kNs5Ticks; ++t)                                         // ingest peer A's (clean) checksums
            IngestRemote(d, ChecksumPacket{ t, traceAClean[t] });
        ns5CleanZeroDesync = (d.desynced == false);
        check(ns5CleanZeroDesync, "ns5: clean session — identical peers report ZERO desync");
    }

    // ---- (NS5-2) CORRUPTED PEER DETECTED AT THE EXACT TICK K. ----------------------------------------
    DesyncDetector dCorr;
    {
        for (uint32_t t = 0; t < kNs5Ticks; ++t) RecordLocal(dCorr, t, traceB[t]);       // peer B records ITS trace
        for (uint32_t t = 0; t < kNs5Ticks; ++t)                                         // ingest peer A's CORRUPT checksums (ascending)
            IngestRemote(dCorr, ChecksumPacket{ t, traceACorr[t] });
        check(dCorr.desynced == true,             "ns5: corrupted peer is detected (desynced)");
        check(dCorr.desyncTick == kK,             "ns5: desync located at the EXACT first diverging tick K");
        check(dCorr.localDigest != dCorr.remoteDigest, "ns5: the two diverging digests differ");
        check(dCorr.localDigest == traceB[kK],    "ns5: latched local digest == peer B's digest at K");
        check(dCorr.remoteDigest == traceACorr[kK],"ns5: latched remote digest == peer A's digest at K");
    }

    // ---- (NS5-3) LOCALIZATION EXACT — identical for every tick < K, differ at K. ---------------------
    bool ns5LocalizationExact = true;
    {
        for (uint32_t t = 0; t < kK; ++t)
            if (traceB[t] != traceACorr[t]) ns5LocalizationExact = false;   // must be IDENTICAL before K
        if (traceB[kK] == traceACorr[kK]) ns5LocalizationExact = false;     // must DIFFER at K
        check(ns5LocalizationExact, "ns5: localization exact — identical before K, differ at K");
    }

    // ---- (NS5-4) PINNED — the clean trace's final digest == a hard-pinned uint64_t. ------------------
    const uint64_t ns5CleanFinal = traceB[kNs5Ticks - 1];
    const uint64_t kPinnedNs5Final = 0x49aa655446b5c3a2ull; // the clean trace's final digest (anchor)
    check(ns5CleanFinal == kPinnedNs5Final, "ns5: pinned clean final digest matches golden");
    // Also confirm clean traces agree at EVERY tick (the lockstep invariant the detector relies on).
    {
        bool cleanEqualEvery = true;
        for (uint32_t t = 0; t < kNs5Ticks; ++t)
            if (traceB[t] != traceAClean[t]) cleanEqualEvery = false;
        check(cleanEqualEvery, "ns5: clean traces equal at EVERY tick (lockstep invariant)");
    }

    // ---- NS5 showcase (printed; no image golden). ---------------------------------------------------
    std::printf("ns5-desync: per-tick digest exchange desync detector\n");
    std::printf("ns5-desync: clean session — zero desync {desynced:%s}\n",
                ns5CleanZeroDesync ? "false" : "true");   // ns5CleanZeroDesync == (desynced==false)
    std::printf("ns5-desync: corrupted peer detected at exact tick {desyncTick:%u, local:0x%016llx, remote:0x%016llx}\n",
                dCorr.desyncTick,
                static_cast<unsigned long long>(dCorr.localDigest),
                static_cast<unsigned long long>(dCorr.remoteDigest));
    std::printf("ns5-desync: localization exact — identical before K, differ at K {ok:%s}\n",
                ns5LocalizationExact ? "true" : "false");
    std::printf("ns5-desync: pinned clean final digest {hash:0x%016llx}\n",
                static_cast<unsigned long long>(ns5CleanFinal));

    if (g_fail == 0) { std::printf("session_test: ALL CHECKS PASSED\n"); return 0; }
    std::printf("session_test: %d failures\n", g_fail);
    return 1;
}
