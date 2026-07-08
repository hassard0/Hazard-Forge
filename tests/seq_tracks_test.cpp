// Unit test for the deterministic SEMANTIC sequencer track types (engine/seq/seq_tracks.h, Slice SQ2 —
// the DETERMINISTIC CINEMATIC SEQUENCER, next-tier parity gap #6). Pure CPU (hf_core), ASan-eligible.
//
// SELF-CONTAINED scaffolding (check() copied from seq_test.cpp, NOT included) so this compiles STANDALONE
// with `clang++ -std=c++20 -I engine -I tests tests/seq_tracks_test.cpp` on the Mac — the cheap
// cross-platform proof. seq_tracks.h is pure INTEGER Q16.16 (composes seq.h read-only), so every digest
// (net::DigestBytes / FNV-1a-64) is bit-identical run-to-run AND MSVC-vs-Apple-clang. Goldens are PINNED
// uint64 values IN the test (NO image / render-bake — UE5's float Sequencer cannot pin these).
//
// What this pins:
//   (a) CAMERA-CUT — piecewise-constant ActiveCamera; before-first-cut default; a cut exactly on a tick.
//   (b) AUDIO — cues fire in the exact crossed half-open windows, once each, ascending; boundary; the
//       union-of-windows == one-window composition (the scrub lemma).
//   (c) MATERIAL — param AT a keyframe == that key; eased between; multiple tracks independent.
//   (d) SUB-SEQUENCE — sub at startTick == sub t=0; playRate=2 double-speed; clamp past the end; depth-2
//       nesting; the composite eval digest pinned.
//   (e) SCRUB — SeekCine(N) == PlayCine to N (THE load-bearing determinism claim); reverse seek.
//   (f) AU2 composition — a fired cue drives an audio_graph envelope retrigger, byte-identical replay.
//   (g) the shared showcase-viz stats (pixel digest + eval digest) pinned (== the --sq2-cinematic shot).

#include "seq/seq_tracks.h"
#include "audio/audio_graph.h"   // AU2: demonstrate a fired cue driving a real audio graph (composition)

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::seq;
namespace ag = hf::audio::graph;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

int main() {
    HF_TEST_MAIN_INIT();

    const Timeline cine = MakeSq2Cinematic();

    // ================================ (a) CAMERA-CUT ==============================================
    {
        // A separate track whose first cut is at 1s -> before it, the caller's default shows.
        CameraCutTrack tr;
        tr.cuts = { CameraCut{kOne, 7}, CameraCut{2 * kOne, 9} };
        check(ActiveCamera(tr, 0, 3) == 3,        "sq2-cam: before first cut -> default camera");
        check(ActiveCamera(tr, kOne / 2, 3) == 3, "sq2-cam: still before first cut -> default");
        check(ActiveCamera(tr, kOne, 3) == 7,     "sq2-cam: a cut exactly ON its tick is active (<= convention)");
        check(ActiveCamera(tr, kOne + 1, 3) == 7, "sq2-cam: piecewise-constant hold after a cut");
        check(ActiveCamera(tr, 2 * kOne, 3) == 9, "sq2-cam: second cut active at its tick");
        check(ActiveCamera(tr, 100 * kOne, 3) == 9,"sq2-cam: past the last cut holds the last camera");
        // Single-cut track: default for t<cut, that camera for all t>=cut.
        CameraCutTrack one; one.cuts = { CameraCut{kOne, 5} };
        check(ActiveCamera(one, 0, 0) == 0 && ActiveCamera(one, kOne, 0) == 5 &&
              ActiveCamera(one, 50 * kOne, 0) == 5, "sq2-cam: single cut piecewise-constant");
        // The cinematic: cam 0 in shot A, cam 1 after the cut at 2s.
        check(ActiveCamera(cine.camera, kOne, cine.defaultCamera) == 0 &&
              ActiveCamera(cine.camera, 2 * kOne, cine.defaultCamera) == 1 &&
              ActiveCamera(cine.camera, 3 * kOne, cine.defaultCamera) == 1,
              "sq2-cam: cinematic shot A cam0 / shot B cam1 across the cut");
    }

    // ================================ (b) AUDIO ===================================================
    {
        const AudioTrack& a = cine.audio;  // cues at 0.5s (id 100) and 2.0s (id 200)
        // A window that straddles the intro cue fires it exactly once.
        std::vector<AudioCue> w = SampleAudioCues(a, kOne / 4, kOne);
        check(w.size() == 1 && w[0].soundId == 100, "sq2-audio: intro cue fires once in its window");
        // HALF-OPEN: a cue exactly at the window START fires; at the window END does NOT (fires next).
        check(SampleAudioCues(a, kOne / 2, kOne).size() == 1,   "sq2-audio: cue at window start [t..) fires");
        check(SampleAudioCues(a, 0, kOne / 2).size() == 0,      "sq2-audio: cue at window end [..t) does NOT fire");
        check(SampleAudioCues(a, kOne / 2, 2 * kOne).size() == 1,"sq2-audio: [0.5,2) captures only the intro");
        // Empty/negative window fires nothing.
        check(SampleAudioCues(a, kOne, kOne).empty() && SampleAudioCues(a, 2 * kOne, kOne).empty(),
              "sq2-audio: empty/negative window fires nothing");
        // COMPOSITION LEMMA: the union of consecutive per-tick windows == one big window (same set/order).
        std::vector<AudioCue> perTick;
        for (uint32_t i = 0; i < kSq2Ticks; ++i) {
            const fx t0 = (fx)((int64_t)i * (int64_t)kSq2Dt);
            const fx t1 = (fx)((int64_t)(i + 1) * (int64_t)kSq2Dt);
            for (const AudioCue& q : SampleAudioCues(a, t0, t1)) perTick.push_back(q);
        }
        const fx tEnd = (fx)((int64_t)kSq2Ticks * (int64_t)kSq2Dt);
        std::vector<AudioCue> oneWin = SampleAudioCues(a, 0, tEnd);
        bool sameCues = (perTick.size() == oneWin.size());
        for (std::size_t i = 0; sameCues && i < perTick.size(); ++i)
            sameCues = perTick[i].tick == oneWin[i].tick && perTick[i].soundId == oneWin[i].soundId &&
                       perTick[i].gain == oneWin[i].gain;
        check(sameCues && oneWin.size() == 2, "sq2-audio: union of per-tick windows == one big window (the scrub lemma)");
    }

    // ================================ (c) MATERIAL ================================================
    {
        const MaterialParamTrack& m = cine.materials[0];  // emissive 0 ->(2s) kOne ->(4s) 0, EaseInOutSine
        check(SampleMaterialParam(m, 0) == 0,          "sq2-mat: param at keyframe 0 == 0");
        check(SampleMaterialParam(m, 2 * kOne) == kOne,"sq2-mat: param at the peak keyframe == kOne");
        check(SampleMaterialParam(m, 4 * kOne) == 0,   "sq2-mat: param at the last keyframe == 0");
        // Between keys: eased value strictly inside (0, kOne) and == the raw SampleScalar of its curve.
        const fx mid = SampleMaterialParam(m, kOne);
        check(mid > 0 && mid < kOne && mid == SampleScalar(m.curve, kOne),
              "sq2-mat: eased value between keys == SampleScalar(curve) and in (0,kOne)");
        // Two independent tracks: driving different params does not cross-talk.
        MaterialParamTrack m2;
        m2.targetMaterialId = 9; m2.paramId = 2;
        m2.curve.times = {0, kOne}; m2.curve.values = {kOne, 0}; m2.curve.easing = Easing::Linear;
        check(SampleMaterialParam(m2, kOne / 2) == kOne / 2 && SampleMaterialParam(m, kOne / 2) != SampleMaterialParam(m2, kOne / 2),
              "sq2-mat: independent param tracks");
    }

    // ================================ (d) SUB-SEQUENCE ============================================
    {
        // The cinematic's sub: ambient starts at 0.5s, playRate 2x, clamp [0, kOne].
        const SubSequence& s = cine.subseq.subs[0];
        check(SubLocalTime(s, s.startTick) == 0, "sq2-sub: at startTick -> sub local t == 0");
        // playRate=2x: 0.25s of parent time past the start -> 0.5s of sub-local time.
        check(SubLocalTime(s, s.startTick + kOne / 4) == kOne / 2, "sq2-sub: playRate=2 runs at double speed");
        // Clamp past the sub's end: a parent time that would overrun -> clampEnd (hold last frame).
        check(SubLocalTime(s, s.startTick + kOne) == kOne, "sq2-sub: local time clamps to clampEnd past the sub end");
        // Before startTick: negative raw -> clampStart (hold first frame).
        check(SubLocalTime(s, 0) == s.clampStart, "sq2-sub: before startTick -> clampStart (hold first frame)");
        // DEPTH-2 nesting: the ambient sub itself holds a flicker sub.
        check(SubDepth(cine) == 2, "sq2-sub: cinematic nests to depth 2");
        // The recursive eval carries the nested structure: subs[0].subs[0] exists (depth-2 in the eval tree).
        const fx scrubT = (fx)((int64_t)kSq2ScrubTick * (int64_t)kSq2Dt);
        const TimelineEval e = EvalTimeline(cine, scrubT);
        check(e.subs.size() == 1 && e.subs[0].subs.size() == 1,
              "sq2-sub: recursive eval reaches depth-2");
    }

    // ================================ (e) SCRUB — SeekCine == PlayCine (THE MOAT) =================
    // The load-bearing determinism claim: seeking to tick N is BIT-IDENTICAL to playing forward to N,
    // for BOTH the instantaneous state AND the fired-cue trace. Checked at several N (must be EQUAL, not
    // just pinned) — the property UE5's float playback timing structurally cannot hold.
    {
        bool allEqual = true;
        for (uint32_t N : {0u, 15u, 60u, 75u, 90u, 120u}) {
            const uint64_t play = DigestCine(PlayCine(cine, kSq2Dt, N));
            const uint64_t seek = DigestCine(SeekCine(cine, kSq2Dt, N));
            if (play != seek) { allEqual = false; std::printf("  scrub mismatch at N=%u: play=0x%016llx seek=0x%016llx\n",
                                                              N, (unsigned long long)play, (unsigned long long)seek); }
        }
        check(allEqual, "sq2-scrub: SeekCine(N) == PlayCine to N (the deterministic scrub — load-bearing)");
        // Reverse seek: seek forward to 120, then back to 30 -> identical to a fresh play-to-30 (stateless).
        const uint64_t fwd = DigestCine(SeekCine(cine, kSq2Dt, 120));
        (void)fwd;
        const uint64_t back = DigestCine(SeekCine(cine, kSq2Dt, 30));
        const uint64_t fresh = DigestCine(PlayCine(cine, kSq2Dt, 30));
        check(back == fresh, "sq2-scrub: reverse seek (120 -> 30) reproduces play-to-30 exactly");
    }

    // ================================ (f) AU2 COMPOSITION =========================================
    // A fired cue routes to the AU2 audio graph: we retrigger a node's envelope (envT=0) at each fired
    // cue's tick — the RenderAu1Showcase host-schedule convention (`st.st[n].envT = 0`). Applying the
    // SAME fired-cue schedule twice yields BYTE-IDENTICAL audio (deterministic composition).
    {
        auto renderWithCues = [&](std::vector<int16_t>& out) {
            ag::Graph g = ag::MakeShowcaseGraph();          // osc->adsr->gain->delay->pan->out (node 1 = ADSR)
            ag::GraphState st = ag::MakeGraphState(g);
            const int kSR = 48000, kBlock = 480, kBlocks = 20;
            out.clear();
            std::vector<AudioCue> cues = SampleAudioCues(cine.audio, 0, (fx)((int64_t)kBlocks * kBlock * kOne / kSR));
            std::size_t ci = 0;
            for (int b = 0; b < kBlocks; ++b) {
                const int sample = b * kBlock;
                const fx tNow = (fx)((int64_t)sample * kOne / kSR);
                while (ci < cues.size() && cues[ci].tick <= tNow) { st.st[1].envT = 0; ++ci; }  // cue -> retrigger ADSR
                ag::RenderBlock(g, st, kSR, kBlock, out);
            }
        };
        std::vector<int16_t> a1, a2;
        renderWithCues(a1);
        renderWithCues(a2);
        check(a1.size() == a2.size() && ag::DigestAudioBlock(a1) == ag::DigestAudioBlock(a2),
              "sq2-au2: a fired cue drives an audio-graph envelope retrigger, byte-identical replay");
    }

    // ================================ (g) SHOWCASE-VIZ STATS ======================================
    {
        std::vector<uint8_t> img1, img2;
        Sq2VizStats s1{}, s2{};
        RenderSq2CinematicViz(img1, s1);
        RenderSq2CinematicViz(img2, s2);
        check(img1.size() == img2.size() && s1.pixDigest == s2.pixDigest,
              "sq2-viz: two renders byte-identical (deterministic strict-zero)");
        check(s1.tracks == 4 && s1.cuts == 2 && s1.cues == 2 && s1.subDepth == 2 && s1.ticks == kSq2Ticks,
              "sq2-viz: stat line tracks=4 cuts=2 cues=2 subDepth=2 ticks=120");
        std::printf("sq2-viz: %ux%u pixDigest=0x%016llx evalDigest=0x%016llx\n",
                    s1.width, s1.height, (unsigned long long)s1.pixDigest, (unsigned long long)s1.evalDigest);
    }

    // ================================ PINNED CROSS-PLATFORM DIGESTS ===============================
    // These are the make-or-break cross-platform anchors (identical MSVC + Apple clang). PINNED on first
    // run below; a change to any semantic-track evaluation moves them.
    {
        const fx scrubT = (fx)((int64_t)kSq2ScrubTick * (int64_t)kSq2Dt);
        const uint64_t evalDigest = DigestEval(EvalTimeline(cine, scrubT));
        const uint64_t cineDigest = DigestCine(SeekCine(cine, kSq2Dt, kSq2ScrubTick));
        std::vector<uint8_t> img; Sq2VizStats vs{};
        RenderSq2CinematicViz(img, vs);

        std::printf("sq2: evalDigest=0x%016llx cineDigest=0x%016llx pixDigest=0x%016llx\n",
                    (unsigned long long)evalDigest, (unsigned long long)cineDigest,
                    (unsigned long long)vs.pixDigest);

        const uint64_t kPinEval = 0x798c4d941e24aa46ull;   // PINNED on first run (MSVC == clang)
        const uint64_t kPinCine = 0xdad69500234307a3ull;   // PINNED on first run (MSVC == clang)
        const uint64_t kPinPix  = 0x10a63b075559de01ull;   // PINNED on first run (MSVC == clang)
        check(evalDigest == kPinEval,  "sq2: EvalTimeline digest == pinned uint64 (cross-platform)");
        check(cineDigest == kPinCine,  "sq2: SeekCine digest == pinned uint64 (cross-platform)");
        check(vs.pixDigest == kPinPix, "sq2: showcase-viz pixel digest == pinned uint64 (cross-platform)");
    }

    if (g_fail == 0) std::printf("ALL SQ2 TESTS PASSED\n");
    else             std::printf("%d SQ2 TEST(S) FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
