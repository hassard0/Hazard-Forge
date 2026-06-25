#pragma once
// Hazard Forge — UMG-Designer widget-tree editor panels (Dear ImGui rendering of the WidgetEditorView).
//
// Issue #30 (UMG-class UI framework — widget hierarchy + data binding + animations), the designer's ImGui
// layer: turns the deterministic, ImGui-free WidgetEditorView (widget_editor_data.h) into Dear ImGui widgets
// — a HIERARCHY column (indented, kind-labeled rows with a selection highlight), a LAYOUT PREVIEW pane
// (nested boxes = the scaled SolveLayout rects, the selected one highlighted), and a PROPERTY INSPECTOR (the
// selected widget's Style fields as label : value rows). This is the UMG twin of seq_editor_panels.cpp /
// flow_editor_panels.cpp / profiler_view_panels.cpp: the DATA (the view + the layout) is computed in hf_core
// and unit-tested headlessly; THIS module only issues ImGui draw calls over it. Depends on ImGui +
// widget_editor_data.h only (no rhi/backend symbols).

#include <cstdint>

#include "editor/widget_editor_data.h"   // WidgetEditorView / WidgetEditorLayout (the pure deterministic view)
#include "ui/widget.h"                   // hf::ui::Tree (for kind labels / widget read-back)

namespace hf::editor {

// Build the docked UMG-Designer widget-tree editor for this frame from a pre-laid-out WidgetEditorView.
// Call between ImGui::NewFrame() and ImGui::Render(). `fbWidth`/`fbHeight` are the framebuffer size; a left
// panel draws the hierarchy + inspector, the main canvas draws the layout preview (nested boxes) via the
// ImGui draw list. `tree` is shown for read-back (kind names / widget count); the view is the authority on
// geometry. Deterministic given the same view (ImGui geometry is CPU-built).
void BuildWidgetEditorUI(const Tree& tree, const WidgetEditorView& view,
                         uint32_t fbWidth, uint32_t fbHeight,
                         const WidgetEditorLayout& layout = WidgetEditorLayout{});

}  // namespace hf::editor
