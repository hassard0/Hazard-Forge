// Unit test for the cinematic-sequencer TIMELINE VIEW + EDIT data models (engine/editor/seq_editor_data.h
// + engine/editor/seq_edit_ops.h, issue #25 — the GUI half of the Cinematic Sequencer / timeline). Pure CPU
// (hf_core), ASan-eligible like the other pure tests, NO image / NO render-bake.
//
// SELF-CONTAINED: the scaffolding (check() + HF_TEST_MAIN_INIT) mirrors flow_editor_test.cpp /
// profiler_view_test.cpp / seq_test.cpp so this compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/seq_editor_test.cpp` on the Mac — the cheap cross-platform
// proof. The WHOLE layout is INTEGER (track lanes at a fixed vertical stride, keyframe markers at
// time->X / value->Y via integer proportions of the Q16.16 range, the SampleScalar interpolation polyline
// mapped the same way) so the view — and hence DigestSeqTimelineView (FNV-1a-64) over it — is bit-identical
// run-to-run AND platform-to-platform (MSVC == Apple clang). The golden is a PINNED FNV-1a-64 value IN the
// test.
//
// The sequence is hf::seq::MakeShowcaseSequence(): a FIXED 3-channel timeline (ch0 Linear, ch1 EaseInOutSine,
// ch2 EaseInQuad), each a 4-key track on times {0,1,2,3}s. So: 3 lanes, 12 keyframe markers (4 per track),
// 3*(curveSteps+1) curve vertices.
//
// What this pins:
//   (a) the lane count == the track count (3) + a hand-checked lane stride/position;
//   (b) the keyframe-marker count == sum of per-track key counts (12) + markers carry the right Q16.16
//       (time,value) + the first/last keyframe of a track sit at the time-axis ends;
//   (c) the curve-point count == tracks*(curveSteps+1) + the polyline endpoints sit at the axis ends;
//   (d) DigestSeqTimelineView(view) == a hard-pinned uint64 (the cross-platform proof);
//   (e) re-building the view is bit-identical (deterministic / replay-stable);
//   (f) an AddKeyframe edit changes the view digest DETERMINISTICALLY (a new key -> a new marker), and the
//       edit is itself replay-stable (re-applying to a fresh sequence reproduces the new digest);
//   (g) AddKeyframe keeps `times` strictly ascending (the ScalarTrack invariant) incl. the overwrite-on-
//       exact-time path; DeleteKeyframe / MoveKeyframe round-trip.

#include "editor/seq_editor_data.h"
#include "editor/seq_edit_ops.h"
#include "seq/seq.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::editor;
using hf::seq::fx;
using hf::seq::kOne;
using hf::seq::Sequence;
using hf::seq::ScalarTrack;
using hf::seq::MakeShowcaseSequence;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

int main() {
    HF_TEST_MAIN_INIT();

    const Sequence seq = MakeShowcaseSequence();
    const SeqLayout L;                       // default fixed grid
    const fx playhead = (fx)(3 * kOne / 2);  // 1.5s — a non-endpoint playhead inside [0,3]s

    const SeqTimelineView view = BuildSeqTimelineView(seq, playhead, L);

    std::printf("seq-editor: tracks=%zu lanes=%zu keys=%zu curve=%zu playheadX=%d timeAxisW=%d\n",
                seq.tracks.size(), view.lanes.size(), view.keys.size(), view.curve.size(),
                view.playheadX, view.timeAxisW);
    std::printf("seq-editor: tMin=%d tMax=%d (Q16.16)\n", (int)view.tMinFx, (int)view.tMaxFx);

    // ---- (a) lane count == track count + a hand-checked lane stride. ----------------------------------
    check(view.lanes.size() == seq.tracks.size() && view.lanes.size() == 3,
          "seq-editor: one lane per track (3 lanes)");
    if (view.lanes.size() == 3) {
        check(view.lanes[0].x == L.originX && view.lanes[0].y == L.originY &&
              view.lanes[0].w == L.timeAxisW && view.lanes[0].h == L.laneH,
              "seq-editor: lane 0 at (originX, originY) spanning timeAxisW x laneH");
        check(view.lanes[1].y == L.originY + 1 * (L.laneH + L.laneGap),
              "seq-editor: lane 1 one (laneH+laneGap) below");
        check(view.lanes[2].y == L.originY + 2 * (L.laneH + L.laneGap),
              "seq-editor: lane 2 two strides below");
        check(view.lanes[0].trackIndex == 0 && view.lanes[2].trackIndex == 2,
              "seq-editor: lanes carry their track index in order");
    }

    // ---- (b) keyframe-marker count == sum of per-track key counts (12). -------------------------------
    std::size_t expectKeys = 0;
    for (const ScalarTrack& tr : seq.tracks) expectKeys += tr.times.size();
    check(view.keys.size() == expectKeys && view.keys.size() == 12,
          "seq-editor: one marker per keyframe (12 = 4 per track x 3)");
    if (view.keys.size() == 12) {
        // The shared time axis spans [tMin, tMax]; a track's FIRST key (t==tMin) sits at the left axis edge
        // (originX) and its LAST key (t==tMax) at the right edge (originX+timeAxisW).
        const SeqKeyMarker& k0 = view.keys[0];   // track 0, key 0 (t==0==tMin)
        const SeqKeyMarker& k3 = view.keys[3];   // track 0, key 3 (t==3s==tMax)
        check(k0.trackIndex == 0 && k0.keyIndex == 0 && k0.timeFx == 0,
              "seq-editor: first marker is track0/key0 at t==0");
        check(k0.x == L.originX, "seq-editor: t==tMin maps to the left axis edge (originX)");
        check(k3.keyIndex == 3 && k3.timeFx == 3 * kOne && k3.x == L.originX + L.timeAxisW,
              "seq-editor: t==tMax maps to the right axis edge (originX+timeAxisW)");
        // Markers carry the raw Q16.16 (time,value) for labeling.
        check(view.keys[0].valueFx == seq.tracks[0].values[0] &&
              view.keys[3].valueFx == seq.tracks[0].values[3],
              "seq-editor: markers carry the keyframe Q16.16 values");
        // The marker at t==1.5s playhead does NOT exist (keys are AT keyframes 0,1,2,3); the midpoint key
        // of track 0 (key 1, t==1s) maps to a quarter of the axis.
        check(view.keys[1].timeFx == kOne && view.keys[1].x == L.originX + L.timeAxisW / 3,
              "seq-editor: key at t==1s maps to 1/3 of a [0,3]s axis");
    }

    // ---- (c) curve-point count == tracks*(curveSteps+1) + polyline endpoints at the axis ends. --------
    const std::size_t expectCurve = seq.tracks.size() * (std::size_t)(L.curveSteps + 1u);
    check(view.curve.size() == expectCurve,
          "seq-editor: curveSteps+1 polyline vertices per track");
    if (!view.curve.empty()) {
        const SeqCurvePoint& c0 = view.curve.front();              // track 0, step 0 (t==tMin)
        const SeqCurvePoint& cLast = view.curve[L.curveSteps];     // track 0, step curveSteps (t==tMax)
        check(c0.trackIndex == 0 && c0.x == L.originX,
              "seq-editor: track-0 curve starts at the left axis edge");
        check(cLast.trackIndex == 0 && cLast.x == L.originX + L.timeAxisW,
              "seq-editor: track-0 curve ends at the right axis edge");
    }

    // ---- (d) PINNED DIGEST — the cross-platform make-or-break (identical on MSVC + clang). ------------
    const uint64_t digest = DigestSeqTimelineView(view);
    std::printf("seq-editor: view digest = 0x%016llx\n", (unsigned long long)digest);
    const uint64_t kPinnedDigest = 0xf7a649c83eba8e7full;  // PINNED on first run (MSVC == clang)
    check(digest == kPinnedDigest,
          "seq-editor: DigestSeqTimelineView(view) == pinned uint64 (the cross-platform proof)");

    // ---- (e) REPLAY-STABLE — re-building the same view reproduces the digest. -------------------------
    check(DigestSeqTimelineView(BuildSeqTimelineView(seq, playhead, L)) == digest,
          "seq-editor: re-building the view is bit-identical (deterministic)");

    // ---- (f) an AddKeyframe edit changes the digest deterministically. --------------------------------
    // Insert a new keyframe into track 1 at t==2.5s (between its keys at 2s and 3s), value 0.75. This adds a
    // marker + reshapes the curve -> the view digest MUST change, reproducibly.
    {
        Sequence edited = MakeShowcaseSequence();
        const std::size_t at = AddKeyframe(edited.tracks[1], (fx)(5 * kOne / 2), (fx)(3 * kOne / 4));
        check(edited.tracks[1].times.size() == 5 && at == 3,
              "seq-editor: AddKeyframe inserts at the sorted slot (track1 now 5 keys, new at index 3)");
        // Strictly-ascending invariant preserved.
        bool ascending = true;
        for (std::size_t i = 1; i < edited.tracks[1].times.size(); ++i)
            if (!(edited.tracks[1].times[i] > edited.tracks[1].times[i - 1])) ascending = false;
        check(ascending, "seq-editor: AddKeyframe keeps times strictly ascending");

        const SeqTimelineView ev = BuildSeqTimelineView(edited, playhead, L);
        const uint64_t editedDigest = DigestSeqTimelineView(ev);
        std::printf("seq-editor: edited view digest = 0x%016llx (keys=%zu)\n",
                    (unsigned long long)editedDigest, ev.keys.size());
        check(ev.keys.size() == 13, "seq-editor: the edit adds exactly one keyframe marker (12 -> 13)");
        check(editedDigest != digest, "seq-editor: AddKeyframe changes the view digest");
        // Replay-stable: re-applying the SAME edit to a fresh sequence reproduces the edited digest.
        Sequence edited2 = MakeShowcaseSequence();
        AddKeyframe(edited2.tracks[1], (fx)(5 * kOne / 2), (fx)(3 * kOne / 4));
        check(DigestSeqTimelineView(BuildSeqTimelineView(edited2, playhead, L)) == editedDigest,
              "seq-editor: the AddKeyframe edit is replay-stable (same digest)");
    }

    // ---- (g) edit-op invariants: overwrite-on-exact-time, Delete/Move round-trip. ---------------------
    {
        ScalarTrack tr = seq.tracks[0];          // times {0,1,2,3}, values {0,1,-0.5,2}
        const std::size_t n0 = tr.times.size();
        // Overwrite: AddKeyframe at an EXISTING time (t==1s) replaces the value, does NOT add a key.
        const std::size_t hit = AddKeyframe(tr, kOne, (fx)(7 * kOne));
        check(tr.times.size() == n0 && hit == 1 && tr.values[1] == (fx)(7 * kOne),
              "seq-editor: AddKeyframe at an existing time overwrites (no duplicate time)");
        // DeleteKeyframe removes one pair; out-of-range is a no-op false.
        check(DeleteKeyframe(tr, 1) && tr.times.size() == n0 - 1,
              "seq-editor: DeleteKeyframe erases the (time,value) pair");
        check(!DeleteKeyframe(tr, 999), "seq-editor: DeleteKeyframe out-of-range returns false");
        // MoveKeyframe re-times+re-sorts: move key 0 (t==0) to t==2.5s -> it relocates after the others.
        ScalarTrack tr2 = seq.tracks[0];
        check(MoveKeyframe(tr2, 0, (fx)(5 * kOne / 2), (fx)(9 * kOne)),
              "seq-editor: MoveKeyframe succeeds in range");
        bool asc2 = true;
        for (std::size_t i = 1; i < tr2.times.size(); ++i)
            if (!(tr2.times[i] > tr2.times[i - 1])) asc2 = false;
        check(asc2 && tr2.times.size() == seq.tracks[0].times.size(),
              "seq-editor: MoveKeyframe re-sorts + preserves the strictly-ascending invariant");
    }

    if (g_fail == 0) std::printf("ALL PASS\n");
    else             std::printf("%d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
