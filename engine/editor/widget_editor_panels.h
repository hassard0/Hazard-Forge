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
#include <vector>

#include "editor/editor_panels.h"        // UiRect (the ED1 probe-rect currency for synthetic input)
#include "editor/widget_editor_data.h"   // WidgetEditorView / WidgetEditorLayout (the pure deterministic view)
#include "ui/widget.h"                   // hf::ui::Tree (for kind labels / widget read-back)

namespace hf::editor {

// --- Slice ED2: the widget designer's INTERACTIVE state + widget-rect probe. ----------------------
// WidgetEditorState is the panel's cross-frame UI state: the hierarchy-selected widget. The CALLER
// rebuilds the WidgetEditorView from it each frame (the view carries selection into the inspector +
// preview highlight). Plain ints, no ImGui types (imgui.h-free header).
struct WidgetEditorState {
    uint32_t selected = hf::ui::kNoWidget;   // the hierarchy-selected WidgetId (kNoWidget = none)
};

// WidgetEditorUIProbe (the ED1 EditorUIProbe twin): screen rects of the designer's clickable widgets so
// the --ed2-dry-run can aim REAL synthetic io events at real widget geometry. The edit-mode-only
// affordances (add-child buttons / delete button / inspector drag fields) probe invalid rects when the
// panel is built read-only.
struct WidgetEditorUIProbe {
    std::vector<UiRect> hierarchyRows;    // one per view row, view.rows order (the Selectables)
    std::vector<UiRect> addButtons;       // one per view row: the [+] add-child button (edit mode only)
    UiRect deleteButton;                  // the "Delete selected" button (edit mode only)
    std::vector<UiRect> inspectorFields;  // one per inspector row: the DragInt rect (editable rows only)
};

// Build the docked UMG-Designer widget-tree editor for this frame from a pre-laid-out WidgetEditorView.
// Call between ImGui::NewFrame() and ImGui::Render(). `fbWidth`/`fbHeight` are the framebuffer size; a left
// panel draws the hierarchy + inspector, the main canvas draws the layout preview (nested boxes) via the
// ImGui draw list. `tree` is shown for read-back (kind names / widget count); the view is the authority on
// geometry. Deterministic given the same view (ImGui geometry is CPU-built).
//
// Slice ED2 (interactive authoring): when `editTree` + `editState` are non-null the panel EDITS —
// clicking a hierarchy row selects that widget (editState->selected; the caller rebuilds the view from
// it); each row grows an EDIT-MODE-ONLY [+] button that appends a child under that row's widget via
// AddChildWidget (PINNED payload: a default-constructed Style + kind 0/Panel); a "Delete selected"
// button (edit mode only, root-guarded) removes the selected subtree via DeleteWidget; and the
// inspector's editable Style rows become ED1-style Ctrl+click-typeable DragInts that write through
// SetWidgetStyleProp (InspectorProp codes 0..11 == WidgetStyleProp codes 0..11 by construction; the
// kind row stays read-only). Because every new visual affordance draws ONLY in edit mode, the static
// --widget-editor-shot golden (editTree == nullptr) is byte-identical (existing 5-arg callers compile +
// render unchanged). `probe`, when non-null, receives this frame's widget rects.
void BuildWidgetEditorUI(const Tree& tree, const WidgetEditorView& view,
                         uint32_t fbWidth, uint32_t fbHeight,
                         const WidgetEditorLayout& layout = WidgetEditorLayout{},
                         Tree* editTree = nullptr, WidgetEditorState* editState = nullptr,
                         WidgetEditorUIProbe* probe = nullptr);

}  // namespace hf::editor
