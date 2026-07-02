#pragma once
// Hazard Forge — cinematic-sequencer timeline editor panels (Dear ImGui rendering of the SeqTimelineView).
//
// Issue #25 (Cinematic Sequencer / timeline), the editor's ImGui layer: turns the deterministic, ImGui-free
// SeqTimelineView (seq_editor_data.h) into Dear ImGui widgets — horizontal track lanes with alternating
// backgrounds, keyframe diamonds, the per-track sampled interpolation polyline, a playhead line, and a time
// ruler. This is the sequencer twin of flow_editor_panels.cpp / profiler_view_panels.cpp: the DATA (the view
// + the layout) is computed in hf_core and unit-tested headlessly; THIS module only issues ImGui draw calls
// over it. Depends on ImGui + seq_editor_data.h only (no rhi/backend symbols).

#include <cstdint>
#include <vector>

#include "editor/editor_panels.h"     // UiRect (the ED1 probe-rect currency for synthetic input)
#include "editor/seq_editor_data.h"   // SeqTimelineView / SeqLayout (the pure deterministic view)
#include "seq/seq.h"                  // hf::seq::Sequence (for the track-kind / easing labels)

namespace hf::editor {

// --- Slice ED2: the timeline editor's INTERACTIVE state + widget-rect probe. ----------------------
// SeqEditorState is the panel's cross-frame UI state: the selected keyframe (clicked diamond). The
// selection is the MOVE target (Left/Right arrow nudges its time via MoveKeyframe) and the DELETE
// target (the Delete key calls DeleteKeyframe). Plain ints, no ImGui types (imgui.h-free header).
struct SeqEditorState {
    int selectedTrack = -1;   // Sequence track index of the selected key (-1 = none)
    int selectedKey   = -1;   // key index within that track (-1 = none)
};

// SeqEditorUIProbe (the ED1 EditorUIProbe twin): screen rects of the clickable timeline geometry so the
// --ed2-dry-run can aim REAL synthetic io events at real widget geometry. `canvasOrigin` records the
// Timeline window's content origin `o` (x0/y0; the P() offset) so a harness can convert a screen click X
// into the view-local pixel the pinned SeqMapXToTime convention consumes.
struct SeqEditorUIProbe {
    UiRect canvasOrigin;          // x0/y0 = the Timeline canvas content origin (x1/y1 duplicated)
    std::vector<UiRect> lanes;    // one per view lane, lane order (the click-to-add hit boxes)
    std::vector<UiRect> keys;     // one per view key marker, view.keys order (+/-6 px diamond hit boxes)
};

// ---- SeqMapXToTime (Slice ED2): the PINNED inverse of seq_editor_data.h's MapTimeToX. --------------
// Convention (LOCKED; the dry-run's twin recomputes it): the canvas-local pixel x is clamped into
// [originX, originX + timeAxisW], then t = tMin + ((x - originX) * (tMax - tMin) + timeAxisW/2) /
// timeAxisW — int64 round-to-nearest, the exact mirror of MapTimeToX's (+span/2)/span rounding. Pure
// integer, deterministic; a degenerate axis maps every x to tMin. NOTE: MapTimeToX(SeqMapXToTime(x)) may
// differ from x by the rounding quantum — the CONVENTION, not a bug (each map rounds to ITS OWN grid).
inline fx SeqMapXToTime(int x, fx tMin, fx tMax, int originX, int timeAxisW) {
    if (timeAxisW <= 0 || tMax <= tMin) return tMin;        // degenerate axis -> tMin
    int lx = x - originX;
    if (lx < 0) lx = 0;
    if (lx > timeAxisW) lx = timeAxisW;
    const int64_t span = static_cast<int64_t>(tMax) - static_cast<int64_t>(tMin);   // > 0
    const int64_t t = static_cast<int64_t>(tMin) +
                      (static_cast<int64_t>(lx) * span + static_cast<int64_t>(timeAxisW) / 2) /
                          static_cast<int64_t>(timeAxisW);
    return static_cast<fx>(t);
}

// Build the docked cinematic-sequencer timeline editor for this frame from a pre-laid-out SeqTimelineView.
// Call between ImGui::NewFrame() and ImGui::Render(). `fbWidth`/`fbHeight` are the framebuffer size; the
// canvas panel fills the frame and draws the timeline (lanes + keyframe diamonds + interpolation curves +
// playhead + time ruler) via the ImGui draw list, with a left strip listing the track palette + stats.
// `seq` is shown for read-back (track / keyframe counts + per-track easing); the view is the authority on
// geometry. Deterministic given the same view (ImGui geometry is CPU-built).
//
// Slice ED2 (interactive authoring): when `editSeq` + `editState` are non-null the panel EDITS — a click
// inside a track lane (not on a diamond) adds a key at the PINNED SeqMapXToTime x->time (value =
// SampleScalar at that time, i.e. ON the current curve) via AddKeyframe; a click within +/-6 px of a
// keyframe diamond selects it; Left/Right arrows nudge the selected key's time by -/+ kOne/8 (0.125 s)
// via MoveKeyframe (the selection follows the re-sorted key); the Delete key calls DeleteKeyframe. The
// caller owns the post-edit view rebuild (BuildSeqTimelineView re-lays-out next frame). With the defaults
// (nullptr) the panel renders EXACTLY as before ED2 — the selection ring is the only new visual and it
// draws only in edit mode with a live selection, so the static --seq-editor-shot golden is byte-identical
// (existing 5-arg callers compile + render unchanged). `probe`, when non-null, receives this frame's
// widget rects (headless input synthesis).
void BuildSeqEditorUI(const Sequence& seq, const SeqTimelineView& view,
                      uint32_t fbWidth, uint32_t fbHeight, const SeqLayout& layout = SeqLayout{},
                      Sequence* editSeq = nullptr, SeqEditorState* editState = nullptr,
                      SeqEditorUIProbe* probe = nullptr);

}  // namespace hf::editor
