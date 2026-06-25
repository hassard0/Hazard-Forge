#pragma once
// Hazard Forge — profiler timeline VIEW panels (Dear ImGui rendering of the ProfilerView).
//
// Issue #31 (Profiler + GPU debugger integration), the GUI's ImGui layer: turns the deterministic, ImGui-free
// ProfilerView (profiler_view_data.h) into Dear ImGui widgets — an Insights-class profiler window: a TIMELINE
// row of frame cells (colored bars labeled with frame number + cpu time), the SCOPE TREE (indented, cpu-time-
// proportional labeled bars from BuildScopeTree), and a DRAW-CALL inspection list (pass name + draw count).
// This is the profiler twin of editor_panels.cpp / flow_editor_panels.cpp: the DATA (the view + the layout) is
// computed in hf_core and unit-tested headlessly; THIS module only issues ImGui draw calls over it. Depends on
// ImGui + profiler_view_data.h only (no rhi/backend symbols).

#include <cstdint>

#include "editor/profiler_view_data.h"  // ProfilerView / ProfLayout (the pure deterministic view)
#include "profile/profile.h"            // profile::Capture (for the NameTable label resolution)

namespace hf::editor {

// Build the docked profiler view for this frame from a pre-laid-out ProfilerView. Call between
// ImGui::NewFrame() and ImGui::Render(). `fbWidth`/`fbHeight` are the framebuffer size; the window fills the
// frame and draws the frame-cell timeline, the indented scope-tree bars, and the draw-call list via the ImGui
// draw list. `capture` is read ONLY to resolve interned name ids -> label byte-strings (the view is the
// authority on geometry). Deterministic given the same view (ImGui geometry is CPU-built).
void BuildProfilerViewUI(const profile::Capture& capture, const ProfilerView& view,
                         uint32_t fbWidth, uint32_t fbHeight, const ProfLayout& layout = ProfLayout{});

}  // namespace hf::editor
