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

    // =============================== SEQ-S4 — transform / rotation track (THE Q16.16 CRUX) ========
    // The headline: a transform track (translation + rotation + scale) sampled into an integer
    // FxTransform, where the rotation is the float FQuat::Slerp that breaks UE5 Sequencer, rebuilt as a
    // DETERMINISTIC Q16.16 integer NLERP (zero transcendentals). The transform-sweep digest is byte-stable
    // MSVC + clang; the ~unit drift band + the shortest-arc flip are the rotation proofs. PINNED on first
    // run (MSVC == clang), the cross-platform bar.
    const uint64_t kPinnedTransformSweep = 0x59e3f94ce2da437dull;  // PINNED on first run (MSVC == clang)
    // ~unit drift band: measured worst |len - kOne| == 2 LSB over the [0,2s] rotation sweep (nlerp +
    // FxQuatNormalize integer normalize). Pinned at 4 LSB (2x headroom, the FPX4 ~unit discipline) — a
    // ~6.1e-5 relative drift off unit, deterministic MSVC == clang.
    const fx       kUnitBand             = 4;         // ~unit drift band in LSB (measured worst = 2)

    const TransformTrack xfShowcase = MakeShowcaseTransform();
    const std::vector<fx> xfSweep = SampleTransformSweep(xfShowcase, kOne / 30, 90);  // 3s @ 30Hz, 10 fx/tick
    const uint64_t xfDig = DigestTrack(xfSweep);
    std::printf("seq-s4: transform-sweep digest = 0x%016llx  (%zu fx)\n",
                (unsigned long long)xfDig, xfSweep.size());

    // ---- (1) PRIOR INVARIANT — re-assert S1 + all 4 S2 digests + S3 event-sweep, all UNCHANGED. -----
    {
        const bool s1ok = DigestTrack(SampleSweep(MakeShowcaseTrack(), kOne / 30, 90)) == 0xd314f17ebe3d480bull;
        const bool s2tbl = DigestTrack(SineEaseTable()) == 0x8f13b44545cc3c97ull
                        && DigestTrack(QuadInTable())   == 0x7ebbb0956a7f50a2ull
                        && DigestTrack(QuadOutTable())  == 0x5289c36d8551004aull;
        const bool s2seq = DigestTrack(SampleSequenceSweep(MakeShowcaseSequence(), kOne / 30, 90)) == 0xee44096d40ab3946ull;
        const bool s3ev  = flow::DigestEvents(SampleEventSweep(MakeShowcaseEvents(), kOne / 30, 90)) == 0x1035f49824b6ac7aull;
        check(s1ok && s2tbl && s2seq && s3ev,
              "seq-s4: prior invariant — S1 + all 4 S2 + S3 event-sweep digests UNCHANGED (S4 is purely additive)");
    }

    // ---- (2) PINNED TRANSFORM SWEEP — the whole transform timeline is byte-stable MSVC + clang. ------
    check(xfDig == kPinnedTransformSweep,
          "seq-s4: SampleTransformSweep digest == pinned uint64 (the transform timeline is byte-stable)");

    // ---- (3) REPLAY-STABLE — a second sweep reproduces the digest. -----------------------------------
    {
        const std::vector<fx> xfSweep2 = SampleTransformSweep(xfShowcase, kOne / 30, 90);
        check(DigestTrack(xfSweep2) == xfDig,
              "seq-s4: re-sampling the transform track is bit-identical (deterministic)");
    }

    // ---- (4) ~UNIT ROTATION — for several sampled t, |SampleRotation| is within the drift band of kOne.
    // nlerp+normalize keeps the quat ≈unit; len = FxISqrt(int64 sum-of-squares) (Q32.32 -> Q16.16). Assert
    // abs(len - kOne) <= kUnitBand (a small measured/pinned constant — the FPX4 ~unit discipline).
    {
        bool ok = true;
        fx worst = 0;
        for (int s = 0; s <= 60; ++s) {
            const fx t = (fx)((int64_t)s * (int64_t)(2 * kOne) / 60);   // sweep [0, 2s]
            const FxQuat q = SampleRotation(xfShowcase.rotation, t);
            const int64_t ss = (int64_t)q.x * q.x + (int64_t)q.y * q.y
                             + (int64_t)q.z * q.z + (int64_t)q.w * q.w;
            const fx len = (fx)FxISqrt(ss);
            fx d = len - kOne; if (d < 0) d = -d;
            if (d > worst) worst = d;
            if (d > kUnitBand) ok = false;
        }
        std::printf("seq-s4: ~unit rotation worst |len - kOne| = %d LSB (band = %d)\n", (int)worst, (int)kUnitBand);
        check(ok, "seq-s4: a sampled rotation stays ~unit — |q| within the documented Q16.16 drift band of kOne");
    }

    // ---- (5) SHORTEST-ARC — prove the dot<0 flip fires and CHANGES the result. -----------------------
    // Build a 2-key track {q(0°)=identity, q(180° about Y)={0,kOne,0,0}}. The dot of identity.w(=kOne) and
    // q180.w(=0) plus identity.y(=0)*q180.y(=kOne) is 0 — NOT negative — so to force the flip we use the
    // ANTIPODE of q180: q(-180°)={0,-kOne,0,0} (the SAME rotation, double-cover). Now dot(identity, -q180)
    // computes against w=0,y=-kOne -> still 0. The decisive pair is q(90°)->q(-270°-ish). Concretely we
    // take qa=q(90°)={0,46341,0,46341}, qb=antipode-of-q(90°)={0,-46341,0,-46341} (the SAME orientation).
    // Without the flip, nlerp(qa,qb,0.5) = {0,0,0,0} -> normalize -> identity {0,0,0,kOne} (a WRONG,
    // collapsed result). WITH the flip (dot = 46341*-46341*2 < 0 -> negate qb back to qa), nlerp midpoint
    // == qa itself (a valid ≈90° quat). So: midpoint.w must be the ~46341 of qa-normalized, NOT kOne.
    {
        RotationTrack flipTr;
        flipTr.times = {0, kOne};
        flipTr.keys  = { FxQuat{0,  46341, 0,  46341},     // q(90° about Y)
                         FxQuat{0, -46341, 0, -46341} };    // antipode (SAME rotation, opposite hemisphere)
        const FxQuat mid = SampleRotation(flipTr, kOne / 2);  // midpoint of the segment

        // WITHOUT the flip the midpoint would collapse to identity (w == kOne, y == 0). WITH the flip it
        // equals normalized q(90°) (w ≈ 46341, y ≈ 46341). Prove the flip fired + changed the result.
        const FxQuat qaNorm = FxQuatNormalize(FxQuat{0, 46341, 0, 46341});
        fx dw = mid.w - qaNorm.w; if (dw < 0) dw = -dw;
        fx dy = mid.y - qaNorm.y; if (dy < 0) dy = -dy;
        const bool flippedResult = (dw <= kUnitBand && dy <= kUnitBand);   // == normalized q(90°)
        const bool notCollapsed  = (mid.w < kOne - 8 * kUnitBand) && (mid.y > 8 * kUnitBand);  // NOT identity
        std::printf("seq-s4: shortest-arc midpoint = {%d,%d,%d,%d}  (qaNorm.w=%d qaNorm.y=%d)\n",
                    (int)mid.x, (int)mid.y, (int)mid.z, (int)mid.w, (int)qaNorm.w, (int)qaNorm.y);
        check(flippedResult && notCollapsed,
              "seq-s4: shortest-arc — interpolating across the antipode takes the short path (flip fires, no collapse)");
    }

    // ---- (6) ENDPOINT EXACT — SampleRotation at a key time == that key, normalized. ------------------
    {
        const RotationTrack rt = MakeShowcaseRotation();
        bool ok = true;
        for (std::size_t k = 0; k < rt.times.size(); ++k) {
            const FxQuat got = SampleRotation(rt, rt.times[k]);
            const FxQuat exp = FxQuatNormalize(rt.keys[k]);
            // The last key returns keys.back() un-normalized (the hold path); the showcase 180° key
            // {0,kOne,0,0} is already exactly unit so == its own normalize. Interior keys hit t01==0 ->
            // nlerp returns qa, then normalize. Compare normalized both ways for robustness.
            const FxQuat gotN = FxQuatNormalize(got);
            if (gotN.x != exp.x || gotN.y != exp.y || gotN.z != exp.z || gotN.w != exp.w) ok = false;
        }
        check(ok, "seq-s4: rotation endpoints are exact — SampleRotation at a key time == that key (normalized)");
    }

    // ---- (7) LOAD-BEARING ROTATION KEY — perturb one rotation key -> a DIFFERENT transform-sweep digest.
    {
        TransformTrack mutated = MakeShowcaseTransform();
        mutated.rotation.keys[1].y += kOne / 4;   // nudge the 90° key's y component (a visible amount)
        const std::vector<fx> mutSweep = SampleTransformSweep(mutated, kOne / 30, 90);
        check(DigestTrack(mutSweep) != xfDig,
              "seq-s4: nudging one rotation key changes the transform-sweep digest (rotation keys are load-bearing)");
    }

    if (g_fail == 0) { std::printf("seq_test: ALL PASS\n"); return 0; }
    std::printf("seq_test: %d FAIL\n", g_fail);
    return 1;
}
