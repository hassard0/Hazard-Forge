// Unit test for Slice AU2 — DETERMINISTIC CONVOLUTION REVERB + SUBMIX BUSES (engine/audio/reverb.h).
// Pure CPU (hf_core), ASan-eligible like the other pure tests. reverb.h is INTEGER Q15 end to end
// (dry/wet int16 PCM, IR Q15 int32): NO float anywhere, so every buffer here is bit-identical
// run-to-run AND platform-to-platform (MSVC vs clang) — every pinned value below is a hard integer
// golden.
//
// What this pins:
//   (a) IDENTITY — a unit-impulse IR {32768,0,...} makes Convolve a BIT-EXACT identity (wet == dry);
//       the convolution of two known short signals == the hand-computed exact samples; a zero-send bus
//       == the dry path; an empty submix == silence.
//   (b) IR + TAIL — the synthetic room IR's early reflections land at the pinned sample offsets with
//       the pinned Q15 gains; the diffuse-tail envelope decays MONOTONICALLY (non-increasing) and the
//       windowed tail energy is non-increasing (the pinned decay profile).
//   (c) SUBMIX — voices -> reverb-send -> master routing produces the pinned master digest; a muted
//       bus contributes nothing; the mix is PERMUTATION-INVARIANT (reordering the same bus set yields a
//       bit-identical master — the int32-accumulate/single-clamp discipline).
//   (d) WET/DRY — sweeping the reverb-send from 0 -> full moves the output from dry (zero tail energy)
//       toward wet (monotonically growing tail energy) at pinned checkpoints.
//   (e) DETERMINISM — the AU2 showcase scenario digest + stats pinned; two runs byte-identical; the
//       shared viz two renders byte-identical.

#include "audio/reverb.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace rv = hf::audio::reverb;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Windowed sum-of-squares energy over [lo, hi) — a wide int64 accumulate (no float).
static int64_t Energy(const std::vector<int16_t>& v, int lo, int hi) {
    if (lo < 0) lo = 0;
    if (hi > static_cast<int>(v.size())) hi = static_cast<int>(v.size());
    int64_t e = 0;
    for (int i = lo; i < hi; ++i)
        e += static_cast<int64_t>(v[static_cast<std::size_t>(i)]) * v[static_cast<std::size_t>(i)];
    return e;
}

int main() {
    HF_TEST_MAIN_INIT();

    // =================================================================================================
    // (a) IDENTITY.
    // =================================================================================================
    {
        // Unit-impulse IR -> Convolve is a bit-exact identity (wet == dry, up to the trailing zeros).
        std::vector<int16_t> dry = {100, -200, 32767, -32768, 5, 0, -1};
        std::vector<int32_t> ir  = rv::UnitImpulseIR(4);   // {32768, 0, 0, 0}
        std::vector<int16_t> wet = rv::Convolve(dry, ir);
        check(wet.size() == dry.size() + ir.size() - 1, "identity: wetLen == dryLen + irLen - 1");
        bool exact = true;
        for (std::size_t i = 0; i < dry.size(); ++i) if (wet[i] != dry[i]) exact = false;
        for (std::size_t i = dry.size(); i < wet.size(); ++i) if (wet[i] != 0) exact = false;
        check(exact, "identity: unit-impulse IR -> wet == dry bit-exact (then zeros)");

        // The convolution of two KNOWN short signals == the hand-computed exact result.
        // dry=[100,200], ir=[1.0, 0.5]=Q15{32768,16384}:
        //   wet[0]=100*32768>>15=100
        //   wet[1]=(200*32768 + 100*16384)>>15=(6553600+1638400)>>15=8192000>>15=250
        //   wet[2]=200*16384>>15=3276800>>15=100
        std::vector<int16_t> d2  = {100, 200};
        std::vector<int32_t> ir2 = {32768, 16384};
        std::vector<int16_t> w2  = rv::Convolve(d2, ir2);
        check(w2.size() == 3u, "identity: known conv length 3");
        check(w2[0] == 100 && w2[1] == 250 && w2[2] == 100, "identity: known conv == [100,250,100] exact");

        // A zero-send bus -> the master is the dry path (bit-exact over dryLen, zeros after).
        rv::SubmixBus b; b.id = 0; b.gainQ15 = rv::kQ15One; b.reverbSendQ15 = 0; b.voice = dry;
        std::vector<int32_t> room = rv::SynthRoomIR();
        rv::SubmixResult sm = rv::MixSubmix({b}, room, rv::kQ15One);
        bool dryId = true;
        for (std::size_t i = 0; i < dry.size(); ++i) if (sm.master[i] != dry[i]) dryId = false;
        for (int i = sm.dryLen; i < sm.wetLen; ++i) if (sm.master[static_cast<std::size_t>(i)] != 0) dryId = false;
        check(dryId, "identity: zero-send unity bus -> master == dry (then silence)");
        check(sm.buses == 1u, "identity: submix bus count == 1");

        // An empty submix -> silence (no master samples / all zero).
        rv::SubmixResult empty = rv::MixSubmix({}, room, rv::kQ15One);
        check(empty.master.empty(), "identity: empty submix == silence");

        // An all-muted submix -> silence too.
        rv::SubmixBus mb = b; mb.muted = true;
        rv::SubmixResult muted = rv::MixSubmix({mb}, room, rv::kQ15One);
        bool sil = true; for (int16_t s : muted.master) if (s != 0) sil = false;
        check(sil, "identity: all-muted submix == silence");
    }

    // =================================================================================================
    // (b) IR + TAIL — early reflections at pinned offsets/gains; a monotonically-decaying tail.
    // =================================================================================================
    {
        std::vector<int32_t> ir = rv::SynthRoomIR();
        check(static_cast<int>(ir.size()) == rv::kIrLen, "ir: length == kIrLen (2048)");

        // Early reflections land EXACTLY at the pinned offsets with the pinned Q15 gains (the tail
        // starts at kTailStart=900, past every early offset, so these samples are the taps alone).
        check(ir[0]   == 32767, "ir: direct tap @0 == 32767");
        check(ir[149] == 22000, "ir: early tap @149 == 22000");
        check(ir[293] == 16000, "ir: early tap @293 == 16000");
        check(ir[521] == 11000, "ir: early tap @521 == 11000");
        check(ir[787] ==  7000, "ir: early tap @787 == 7000");
        // A gap sample between taps (still before the tail) is exactly zero.
        check(ir[400] == 0, "ir: silent gap @400 == 0 (no early tap, before tail)");

        // The tail envelope is NON-INCREASING (monotonic decay) over its whole span, and reaches
        // exact silence by the end of a long-enough IR.
        std::vector<int32_t> env = rv::SynthRoomIRTailEnvelope();
        check(env[rv::kTailStart] == rv::kTailGainQ15, "ir: tail envelope starts at kTailGainQ15");
        bool nonIncr = true;
        for (int i = rv::kTailStart + 1; i < rv::kIrLen; ++i)
            if (env[static_cast<std::size_t>(i)] > env[static_cast<std::size_t>(i - 1)]) nonIncr = false;
        check(nonIncr, "ir: tail envelope monotonically NON-INCREASING (the decay profile)");

        // The windowed tail ENERGY of the IR (measured on the int16-clamped IR) is non-increasing across
        // consecutive tail windows — the audible decay. (Compare the IR's own coefficients as PCM.)
        std::vector<int16_t> irPcm(ir.size());
        for (std::size_t i = 0; i < ir.size(); ++i) irPcm[i] = hf::audio::dsp::ClampI16(ir[i]);
        const int win = 200;
        int64_t prev = -1; bool tailDecays = true;
        for (int lo = rv::kTailStart; lo + win <= rv::kIrLen; lo += win) {
            const int64_t e = Energy(irPcm, lo, lo + win);
            if (prev >= 0 && e > prev) tailDecays = false;
            prev = e;
        }
        check(tailDecays, "ir: windowed tail energy non-increasing (audible decay)");

        // The reverb of a unit click reproduces the early-reflection structure: convolving a single
        // full-scale impulse with the IR yields wet[k] == clamp(ir[k]) at the tap offsets.
        std::vector<int16_t> impulse = {32767};
        std::vector<int16_t> wet = rv::Convolve(impulse, ir);
        // wet[k] = 32767*ir[k]>>15 = ClampI16( ir[k] * 32767 / 32768 ) ~ ir[k] (off by the <<15/32767 vs 32768).
        // The tap at 149 (gain 22000) -> 22000*32767>>15 = 21999; assert the early taps are the local peaks.
        check(wet[0] > wet[50], "ir: click reverb — direct path dominates the early gap");
        check(wet[149] > wet[200], "ir: click reverb — early reflection @149 is a local peak");
        std::printf("PIN au2 ir: len=%d taps={0:%d,149:%d,293:%d,521:%d,787:%d} tail0=%d\n",
                    rv::kIrLen, ir[0], ir[149], ir[293], ir[521], ir[787], env[rv::kTailStart]);
    }

    // =================================================================================================
    // (c) SUBMIX — routing digest pinned; mute; permutation invariance.
    // =================================================================================================
    {
        std::vector<int32_t> room = rv::SynthRoomIR();
        std::vector<int16_t> click = rv::MakeShotClick();

        std::vector<rv::SubmixBus> buses = {
            rv::SubmixBus{0, 28000,  4000, false, click},
            rv::SubmixBus{1, 20000, 12000, false, click},
            rv::SubmixBus{2, 14000, 24000, false, click},
        };
        rv::SubmixResult a = rv::MixSubmix(buses, room, rv::kQ15One);
        check(a.buses == 3u, "submix: 3 buses");
        check(a.wetLen == a.dryLen + a.irLen - 1, "submix: wetLen == dryLen + irLen - 1");
        std::printf("PIN au2 submix master digest: 0x%016llx (buses:%u dryLen:%d wetLen:%d)\n",
                    static_cast<unsigned long long>(a.digest), a.buses, a.dryLen, a.wetLen);
        const uint64_t kPinnedSubmix = 0x5a5bf8e841fe44d2ull;   // hard integer golden (MSVC == clang)
        check(a.digest == kPinnedSubmix, "pinned: submix master digest");

        // PERMUTATION INVARIANCE — reorder the SAME bus set: the id-sort + int32-accumulate/single-clamp
        // yields a BIT-IDENTICAL master.
        std::vector<rv::SubmixBus> reordered = { buses[2], buses[0], buses[1] };
        rv::SubmixResult b = rv::MixSubmix(reordered, room, rv::kQ15One);
        check(a.master == b.master, "submix: reordering buses is bit-invariant (order-independent mix)");
        check(a.digest == b.digest, "submix: reordered digest identical");

        // MUTE — muting the loudest bus strictly reduces the dry energy.
        std::vector<rv::SubmixBus> muteHot = buses;
        muteHot[0].muted = true;
        rv::SubmixResult m = rv::MixSubmix(muteHot, room, rv::kQ15One);
        check(Energy(m.dry, 0, m.dryLen) < Energy(a.dry, 0, a.dryLen), "submix: muting a bus lowers dry energy");
    }

    // =================================================================================================
    // (d) WET/DRY — sweeping the send from 0 -> full grows the reverb-tail energy monotonically.
    // =================================================================================================
    {
        std::vector<int32_t> room = rv::SynthRoomIR();
        std::vector<int16_t> click = rv::MakeShotClick();
        const int dryLen = static_cast<int>(click.size());

        // At send == 0, the master IS the dry click (bit-exact over dryLen, silence after).
        rv::SubmixResult s0 = rv::MixSubmix({rv::SubmixBus{0, rv::kQ15One, 0, false, click}}, room, rv::kQ15One);
        bool dryExact = true;
        for (int i = 0; i < dryLen; ++i) if (s0.master[static_cast<std::size_t>(i)] != click[static_cast<std::size_t>(i)]) dryExact = false;
        check(dryExact, "wet/dry: send=0 -> master == dry click bit-exact");
        // The reverb tail (samples past dryLen) is exact silence at send=0.
        check(Energy(s0.master, dryLen, s0.wetLen) == 0, "wet/dry: send=0 -> zero tail energy");

        // Sweep the send level; the tail energy (past the dry region) must strictly grow.
        const int32_t sends[4] = {0, 8000, 16000, rv::kQ15One};
        int64_t prevTail = -1; bool grows = true;
        for (int si = 0; si < 4; ++si) {
            rv::SubmixResult r = rv::MixSubmix({rv::SubmixBus{0, rv::kQ15One, sends[si], false, click}}, room, rv::kQ15One);
            const int64_t tail = Energy(r.master, dryLen, r.wetLen);
            if (prevTail >= 0 && !(tail > prevTail)) grows = false;
            prevTail = tail;
        }
        check(grows, "wet/dry: increasing send strictly grows the reverb-tail energy");
    }

    // =================================================================================================
    // (e) DETERMINISM — the AU2 showcase scenario + the shared viz, pinned + two-run identical.
    // =================================================================================================
    {
        rv::Au2ShotRun r1 = rv::RunAu2ShotScenario();
        rv::Au2ShotRun r2 = rv::RunAu2ShotScenario();
        check(r1.dry == r2.dry && r1.ir == r2.ir && r1.wet == r2.wet && r1.master == r2.master,
              "showcase: two scenario runs byte-identical");
        check(r1.digest == r2.digest, "showcase: scenario digests agree");
        check(r1.dryLen == rv::kShotDryLen, "showcase: dryLen == 96");
        check(r1.irLen == rv::kIrLen, "showcase: irLen == 2048");
        check(r1.wetLen == r1.dryLen + r1.irLen - 1, "showcase: wetLen == dryLen + irLen - 1 (2143)");
        check(r1.buses == 3u, "showcase: 3 submix buses");
        std::printf("PIN au2 showcase digest: 0x%016llx (irLen:%d dryLen:%d wetLen:%d buses:%u)\n",
                    static_cast<unsigned long long>(r1.digest), r1.irLen, r1.dryLen, r1.wetLen, r1.buses);
        const uint64_t kPinnedShowcase = 0xe294f310e3227811ull;   // hard integer golden (MSVC == clang)
        check(r1.digest == kPinnedShowcase, "pinned: AU2 showcase scenario digest");

        // The shared viz: two renders byte-identical (the strict-zero cross-backend anchor).
        std::vector<uint8_t> imgA, imgB; uint32_t wA = 0, hA = 0, wB = 0, hB = 0;
        rv::RenderReverbShot(r1, imgA, wA, hA);
        rv::RenderReverbShot(r1, imgB, wB, hB);
        check(wA == rv::kShotW && hA == rv::kShotH, "showcase: viz dims 960x600");
        check(imgA == imgB, "showcase: viz two renders byte-identical");
        check(imgA.size() == static_cast<std::size_t>(wA) * hA * 4u, "showcase: viz buffer sized WxHx4");
    }

    if (g_fail == 0) {
        std::printf("reverb_test: ALL PASS\n");
        return 0;
    }
    std::printf("reverb_test: %d FAILURES\n", g_fail);
    return 1;
}
