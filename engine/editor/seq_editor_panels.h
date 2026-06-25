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

#include "editor/seq_editor_data.h"   // SeqTimelineView / SeqLayout (the pure deterministic view)
#include "seq/seq.h"                  // hf::seq::Sequence (for the track-kind / easing labels)

namespace hf::editor {

// Build the docked cinematic-sequencer timeline editor for this frame from a pre-laid-out SeqTimelineView.
// Call between ImGui::NewFrame() and ImGui::Render(). `fbWidth`/`fbHeight` are the framebuffer size; the
// canvas panel fills the frame and draws the timeline (lanes + keyframe diamonds + interpolation curves +
// playhead + time ruler) via the ImGui draw list, with a left strip listing the track palette + stats.
// `seq` is shown for read-back (track / keyframe counts + per-track easing); the view is the authority on
// geometry. Deterministic given the same view (ImGui geometry is CPU-built).
void BuildSeqEditorUI(const Sequence& seq, const SeqTimelineView& view,
                      uint32_t fbWidth, uint32_t fbHeight, const SeqLayout& layout = SeqLayout{});

}  // namespace hf::editor
