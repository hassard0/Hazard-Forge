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

    // ---- Showcase / numeric proof (printed; no image golden). ---------------------------------------
    std::printf("ns1-session: generic lockstep session (2 peers, toy worlds A+B)\n");
    std::printf("ns1-session: 2-peer lockstep — digests equal every tick {ticks:%u}\n", kTicks);
    std::printf("ns1-session: converged digest pinned {toyA:0x%016llx, toyB:0x%016llx}\n",
                static_cast<unsigned long long>(hToyA), static_cast<unsigned long long>(hToyB));
    std::printf("ns1-session: input-order determinism {inOrder:0x%016llx, reversed:0x%016llx} H1 != H2\n",
                static_cast<unsigned long long>(hInOrder), static_cast<unsigned long long>(hReversed));

    if (g_fail == 0) { std::printf("session_test: ALL CHECKS PASSED\n"); return 0; }
    std::printf("session_test: %d failures\n", g_fail);
    return 1;
}
