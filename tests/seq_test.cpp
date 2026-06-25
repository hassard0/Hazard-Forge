// Unit test for the deterministic Q16.16 scalar keyframe track (engine/seq/seq.h, Slice SEQ-S1,
// issue #25 — the DETERMINISTIC CINEMATIC SEQUENCER beachhead). Pure CPU (hf_core), ASan-eligible
// like the other pure tests.
//
// SELF-CONTAINED: the test scaffolding (check() + HF_TEST_MAIN_INIT()) is copied from flow_test.cpp /
// econ_test.cpp (NOT included) so this compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/seq_test.cpp` on the Mac — the cheap cross-platform
// proof. Everything is INTEGER keyframe interpolation in Q16.16 (fxmul/fxdiv), so the sampled sweep —
// and hence seq::DigestTrack (FNV-1a-64) over it — is bit-identical run-to-run AND platform-to-platform
// (MSVC vs Apple clang). The golden is a PINNED FNV-1a-64 DigestTrack value IN the test (NO image, NO
// render-bake — UE5's float Sequencer cannot pin this cross-platform).
//
// What this pins (the six SEQ-S1 assertions):
//   (a) DigestTrack(SampleSweep(showcase, kOne/30, 90)) == a hard-pinned uint64 (the cross-platform proof);
//   (b) re-sweeping the same track is bit-identical (deterministic / replay-stable);
//   (c) cloning the showcase + nudging one keyframe value changes the digest (keys are load-bearing);
//   (d) a linear-segment midpoint is exact — SampleScalar at t=0.5s of a 0->kOne key == kOne/2;
//   (e) clamp-low — sampling before the first key holds values.front();
//   (f) clamp-high — sampling after the last key holds values.back().

#include "seq/seq.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::seq;
namespace flow = hf::flow;  // S3: the deterministic-VM the event track composes with (EventRecord/Graph/StepGraph)

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- Build the showcase track + sweep it (3 seconds at 30 Hz = 90 samples). ---------------------
    const ScalarTrack showcase = MakeShowcaseTrack();
    const std::vector<fx> sweep = SampleSweep(showcase, kOne / 30, 90);
    const uint64_t digest = DigestTrack(sweep);

    std::printf("seq-s1: showcase sweep digest = 0x%016llx\n",
                static_cast<unsigned long long>(digest));

    // The pinned golden (computed on first run, hardcoded — the regression anchor / cross-platform bar).
    const uint64_t kPinnedDigest = 0xd314f17ebe3d480bull;  // PINNED on first run (MSVC == clang)

    // ---- (a) PINNED DIGEST — the cross-platform make-or-break (identical on MSVC + clang). ----------
    check(digest == kPinnedDigest,
          "seq-s1: DigestTrack(SampleSweep(showcase, kOne/30, 90)) == pinned uint64 (the cross-platform proof)");

    // ---- (b) REPLAY-STABLE — re-sampling the same track reproduces the digest. ----------------------
    {
        const std::vector<fx> sweep2 = SampleSweep(showcase, kOne / 30, 90);
        check(DigestTrack(sweep2) == digest,
              "seq-s1: re-sampling the same track is bit-identical (deterministic)");
    }

    // ---- (c) LOAD-BEARING — clone the showcase, nudge one keyframe value -> differs. -----------------
    // A full 1.0 Q16.16 unit (kOne) on key 1's value: a single-LSB (+1) nudge is below the fixed-point
    // truncation floor at these 30Hz sample points (the lerp's >>16 swallows it), so we nudge by a
    // visible amount — the point is that a keyframe VALUE is load-bearing for the sampled output.
    {
        ScalarTrack mutated = MakeShowcaseTrack();
        mutated.values[1] += kOne;   // nudge key 1's value by 1.0 (Q16.16)
        const std::vector<fx> mutSweep = SampleSweep(mutated, kOne / 30, 90);
        check(DigestTrack(mutSweep) != digest,
              "seq-s1: nudging one keyframe value changes the digest (keys are load-bearing)");
    }

    // ---- (d) LINEAR MIDPOINT — on the segment times[0..1] (values 0 -> kOne), t=0.5s -> kOne/2. ------
    // t01 = fxdiv(kOne/2, kOne) = kOne/2; value = 0 + fxmul(kOne/2, kOne) = kOne/2 (0.5 is exact in Q16.16).
    {
        check(SampleScalar(showcase, kOne / 2) == kOne / 2,
              "seq-s1: a linear-segment midpoint is exact — SampleScalar at t=0.5s of a 0->kOne key == kOne/2");
    }

    // ---- (e) CLAMP LOW — sampling before the first key holds values.front(). ------------------------
    {
        check(SampleScalar(showcase, -kOne) == showcase.values.front(),
              "seq-s1: clamp-low — sampling before the first key holds values.front()");
    }

    // ---- (f) CLAMP HIGH — sampling after the last key holds values.back(). ---------------------------
    {
        check(SampleScalar(showcase, 100 * kOne) == showcase.values.back(),
              "seq-s1: clamp-high — sampling after the last key holds values.back()");
    }

    // =============================== SEQ-S2 — easing-curve LUT + multi-track ======================
    // The pinned table digests (FNV-1a-64 over each LUT's raw int32 bytes) + the multi-track sweep
    // digest. PINNED on first run (MSVC == clang), the cross-platform bar.
    const uint64_t kPinnedSineTable = 0x8f13b44545cc3c97ull;  // PINNED on first run (MSVC == clang)
    const uint64_t kPinnedQuadIn    = 0x7ebbb0956a7f50a2ull;  // PINNED on first run (MSVC == clang)
    const uint64_t kPinnedQuadOut   = 0x5289c36d8551004aull;  // PINNED on first run (MSVC == clang)
    const uint64_t kPinnedSeqSweep  = 0xee44096d40ab3946ull;  // PINNED on first run (MSVC == clang)

    const uint64_t sineDig = DigestTrack(SineEaseTable());
    const uint64_t qInDig  = DigestTrack(QuadInTable());
    const uint64_t qOutDig = DigestTrack(QuadOutTable());
    std::printf("seq-s2: sine-ease-table digest = 0x%016llx\n", (unsigned long long)sineDig);
    std::printf("seq-s2: quad-in-table digest = 0x%016llx   quad-out-table digest = 0x%016llx\n",
                (unsigned long long)qInDig, (unsigned long long)qOutDig);

    // ---- (1) S1 INVARIANT — re-run S1 assertion 1; the Linear path MUST stay bit-identical. -------
    check(DigestTrack(SampleSweep(MakeShowcaseTrack(), kOne / 30, 90)) == 0xd314f17ebe3d480bull,
          "seq-s2: S1 invariant — showcase Linear sweep digest STILL 0xd314f17ebe3d480b (Ease no-op)");

    // ---- (2) TABLE DIGESTS PINNED — the host-baked LUTs are byte-stable MSVC + clang. -------------
    check(sineDig == kPinnedSineTable && qInDig == kPinnedQuadIn && qOutDig == kPinnedQuadOut,
          "seq-s2: easing-table digests == pinned (the host-baked LUTs are byte-stable cross-platform)");

    // ---- (3) ENDPOINTS EXACT — for every Easing, Ease(e,0)==0 and Ease(e,kOne)==kOne. -------------
    {
        bool ok = true;
        const Easing all[] = {Easing::Step, Easing::Linear, Easing::EaseInOutSine,
                              Easing::EaseInQuad, Easing::EaseOutQuad};
        for (Easing e : all) { if (Ease(e, 0) != 0 || Ease(e, kOne) != kOne) ok = false; }
        check(ok, "seq-s2: Ease(*, 0) == 0 and Ease(*, kOne) == kOne exactly (every easing fixes the endpoints)");
    }

    // ---- (4) LINEAR IDENTITY — for a sweep of t01 in [0,kOne], Ease(Linear,t01)==t01. -------------
    {
        bool ok = true;
        for (int s = 0; s <= 64; ++s) {
            const fx t01 = (fx)((int64_t)s * (int64_t)kOne / 64);
            if (Ease(Easing::Linear, t01) != t01) ok = false;
        }
        check(ok, "seq-s2: Ease(Linear, t) == t for a sweep of t (the identity easing is a no-op)");
    }

    // ---- (5) SINE SYMMETRY — Ease(EaseInOutSine, kOne/2) == kOne/2 (+-1 LSB). ----------------------
    {
        const fx mid = Ease(Easing::EaseInOutSine, kOne / 2);
        const fx d   = mid - kOne / 2;
        check((d <= 1 && d >= -1),
              "seq-s2: EaseInOutSine is symmetric — Ease(s, kOne/2) == kOne/2 (the S-curve midpoint, +-1 LSB)");
    }

    // ---- (6) CURVE BENDS — at t=kOne/4 the sine ease differs from Linear (not a hidden identity). --
    {
        check(Ease(Easing::EaseInOutSine, kOne / 4) != Ease(Easing::Linear, kOne / 4),
              "seq-s2: EaseInOutSine(t) != Linear(t) at t=kOne/4 (the curve actually bends — not a sneaky identity)");
    }

    // ---- (7) MULTI-TRACK BUS — SampleSequence at t=0.5s; channel 0 == the S1 track verbatim. ------
    {
        const Sequence seq = MakeShowcaseSequence();
        const std::vector<fx> bus = SampleSequence(seq, kOne / 2);
        check(bus.size() == 3 && bus[0] == SampleScalar(MakeShowcaseTrack(), kOne / 2),
              "seq-s2: SampleSequence at t=0.5s returns the per-channel eased bus (multi-track sampling)");
    }

    // ---- (8) SEQUENCE SWEEP PINNED — the whole multi-track timeline is byte-stable. ---------------
    {
        const Sequence seq = MakeShowcaseSequence();
        const std::vector<fx> sweep2 = SampleSequenceSweep(seq, kOne / 30, 90);
        const uint64_t seqDig = DigestTrack(sweep2);   // FNV-1a-64 over the contiguous int32 bytes
        std::printf("seq-s2: sequence-bus sweep digest = 0x%016llx\n", (unsigned long long)seqDig);
        check(seqDig == kPinnedSeqSweep,
              "seq-s2: SampleSequenceSweep digest == pinned uint64 (the multi-track timeline is byte-stable)");
    }

    // =============================== SEQ-S3 — event track (composes flow.h) ======================
    // Discrete timeline events firing at integer tick boundaries, hashed via flow::DigestEvents (flow's
    // hand-LE event digest) — the fired-event SET is byte-stable MSVC + clang AND it composes with the
    // deterministic flow VM. PINNED on first run (MSVC == clang), the cross-platform bar.
    const uint64_t kPinnedEventSweep = 0x1035f49824b6ac7aull;  // PINNED on first run (MSVC == clang)

    const EventTrack evShowcase = MakeShowcaseEvents();
    // Sweep 3s @ 30Hz (dt=kOne/30, 90 ticks) — covers all 5 events (last at 2.5s < 3.0s).
    const std::vector<flow::EventRecord> evSweep = SampleEventSweep(evShowcase, kOne / 30, 90);
    const uint64_t evDig = flow::DigestEvents(evSweep);
    std::printf("seq-s3: event-sweep trace digest = 0x%016llx  (%zu events)\n",
                (unsigned long long)evDig, evSweep.size());

    // ---- (1) PRIOR INVARIANT — re-assert S1 + all 4 S2 digests, all UNCHANGED (S3 is additive). ----
    {
        const bool s1ok = DigestTrack(SampleSweep(MakeShowcaseTrack(), kOne / 30, 90)) == 0xd314f17ebe3d480bull;
        const bool s2tbl = DigestTrack(SineEaseTable()) == 0x8f13b44545cc3c97ull
                        && DigestTrack(QuadInTable())   == 0x7ebbb0956a7f50a2ull
                        && DigestTrack(QuadOutTable())  == 0x5289c36d8551004aull;
        const std::vector<fx> s2sweep = SampleSequenceSweep(MakeShowcaseSequence(), kOne / 30, 90);
        const bool s2seq = DigestTrack(s2sweep) == 0xee44096d40ab3946ull;
        check(s1ok && s2tbl && s2seq,
              "seq-s3: prior invariant — S1 0xd314f17ebe3d480b + all 4 S2 digests UNCHANGED (S3 is purely additive)");
    }

    // ---- (2) FIRES-ONCE COUNT — every event fires exactly once across the sweep (count == 5). ------
    check(evSweep.size() == evShowcase.times.size() && evSweep.size() == 5,
          "seq-s3: SampleEventSweep fires every event exactly once across the sweep (count == track size)");

    // ---- (3) PINNED TRACE DIGEST — the fired stream is byte-stable MSVC + clang. --------------------
    check(evDig == kPinnedEventSweep,
          "seq-s3: the fired-event trace digest == pinned uint64 (flow::DigestEvents, byte-stable cross-platform)");

    // ---- (4) HALF-OPEN NO-DOUBLE-FIRE — an event exactly on a window boundary fires once, not twice.
    // dt = kOne/2 makes the event at kOne/2 (id 10) land EXACTLY on a window boundary (t=0.5s is the start
    // of window i=1: [kOne/2, kOne)). Sweep 6 windows (0..3s) and count id 10 in the trace == 1.
    {
        const std::vector<flow::EventRecord> bnd = SampleEventSweep(evShowcase, kOne / 2, 6);
        int count10 = 0;
        for (const flow::EventRecord& e : bnd) if (e.eventId == 10u) ++count10;
        check(count10 == 1,
              "seq-s3: half-open window — an event exactly on a tick boundary fires once, not twice (no double-fire)");
    }

    // ---- (5) EMPTY / NEGATIVE WINDOW — [t,t) and a negative window (t<tPrev) fire nothing. ----------
    {
        const std::vector<flow::EventRecord> empt = SampleEvents(evShowcase, 2 * kOne, 2 * kOne);  // [t,t)
        const std::vector<flow::EventRecord> neg  = SampleEvents(evShowcase, 2 * kOne, kOne);      // t<tPrev
        check(empt.empty() && neg.empty(),
              "seq-s3: empty/negative window [t,t) fires nothing");
    }

    // ---- (6) LOAD-BEARING TIME — shift an event out of the swept window -> a DIFFERENT digest. -------
    // The fired-trace digest hashes (eventId, payload) per record in emission order, so a nudge is only
    // load-bearing if it changes WHICH events fall in the swept range [0, 3s) or their order. We shift the
    // LAST event (times[4], 2.5s) PAST the sweep end (to 12.5s, beyond 90 ticks * kOne/30 = 3.0s) — which
    // keeps the STRICTLY-ASCENDING invariant intact (it is already the last key) yet drops it from the
    // fired trace (5 -> 4 events) -> a different digest: the event TIME is load-bearing.
    {
        EventTrack mutated = MakeShowcaseEvents();
        mutated.times[4] += 10 * kOne;   // shift the last event out of the swept [0, 3s) window -> it no longer fires
        const std::vector<flow::EventRecord> mutSweep = SampleEventSweep(mutated, kOne / 30, 90);
        check(mutSweep.size() == 4 && flow::DigestEvents(mutSweep) != evDig,
              "seq-s3: shifting an event out of the swept window changes the trace digest (event times are load-bearing)");
    }

    // ---- (7) FLOW COMPOSITION — feed a fired event's payload into a flow kInput channel; assert the
    // flow output equals the hand-computed expected (the sequence drives the deterministic VM). Build a
    // tiny 2-node graph: n0 = kInput[0] (the external per-tick input), n1 = kAdd(n0, n0) = 2*input.
    // Take event index 3's payload (2*kOne), feed it as inputs[0], StepGraph once, assert n1 == 4*kOne.
    {
        flow::Graph g;
        g.nodes.resize(2);
        g.nodes[0] = flow::Node{ flow::kInput, /*a=*/0, /*b=*/0, /*c=*/0, /*const=*/0 };  // input index 0
        g.nodes[1] = flow::Node{ flow::kAdd,   /*a=*/0, /*b=*/0, /*c=*/1, /*const=*/0 };  // n0 + n0 = 2*input

        const flow::EventRecord fired = evSweep[3];               // the event with payload 2*kOne
        std::vector<flow::Reg> inputs = { fired.payload };        // drive the kInput channel from the event
        flow::GraphState state = flow::MakeState(g);
        const std::vector<flow::Reg> regs = flow::StepGraph(g, state, inputs, /*tick=*/0);

        const flow::Reg expected = (flow::Reg)(2 * kOne);        // payload 2*kOne, doubled = 4*kOne
        check(regs.size() == 2 && regs[0] == fired.payload && regs[1] == 2 * fired.payload
                  && regs[1] == 2 * expected,
              "seq-s3: composition — a fired payload fed into a flow kInput channel yields the expected flow trace");
    }

    if (g_fail == 0) { std::printf("seq_test: ALL PASS\n"); return 0; }
    std::printf("seq_test: %d FAIL\n", g_fail);
    return 1;
}
