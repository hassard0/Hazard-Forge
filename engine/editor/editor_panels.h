#pragma once
// Hazard Forge — editor panels built from live ECS scene data (Dear ImGui rendering).
//
// Builds the DOCKED Dear ImGui editor (Scene Hierarchy / Inspector / Stats panels tiled around a
// central scene Viewport) each frame from the engine's ECS registry + the scene's named resources.
// No user input is required to populate them — they render the live TransformC/MaterialC/MeshC of the
// scene's entities, so a headless --editor-shot capture shows real scene state in a docked layout.
//
// Slice ED1 (interactive inspector editing): the Inspector's Transform/Material readouts are LIVE
// WIDGETS (DragFloat per component, DragFloat for metallic/roughness, a texture-name combo for the
// base color). A change to any widget calls the EXISTING pure-CPU edit ops (edit_ops.h
// ApplyTransformEdit/ApplyMaterialEdit) with an ABSOLUTE SET of the edited field, so the ECS mutates
// the same deterministic way the programmatic --editor-edit path does and the edit persists through
// Ctrl+S / DumpScene. Edit semantics (documented, LOCKED): apply-on-value-change per frame — a mouse
// drag streams absolute sets (live feedback: the panel re-reads the ECS each frame, so the widget and
// the scene stay in lockstep); a Ctrl+click typed entry commits exactly ONE set on Enter/deactivate
// (the deterministic route the --ed1-dry-run proof drives).
//
// The panel DATA (what each panel would display) is factored into editor_panel_data.{h,cpp} (pure
// CPU, ImGui-free, unit-tested + backend-agnostic, lives in hf_core); THIS module turns that data
// into Dear ImGui widgets in a deterministic tiled-docked layout. Depends on ecs/scene + ImGui only
// (no rhi/backend symbols).
#include <cstdint>
#include <vector>

#include "ecs/ecs.h"
#include "editor/edit_history.h"       // Slice ED5: the deterministic undo/redo command stack.
#include "editor/editor_panel_data.h"  // EditorState + the panel-data model (pure).
#include "scene/scene_io.h"

namespace hf::editor {

// --- Slice ED1: the OPTIONAL widget-rect probe for headless synthetic input. ----------------------
// The --ed1-dry-run interactivity proof synthesizes real ImGui mouse/keyboard events; to aim them it
// needs the SCREEN RECTS of the clickable widgets. When BuildEditorUI is given a non-null probe it
// records, for THAT frame, the rect of every hierarchy row plus each editable inspector field. Plain
// floats (no ImGui types) so this header stays imgui.h-free. Rects are in ImGui screen coordinates
// (the same space io.AddMousePosEvent expects).
struct UiRect {
    float x0 = 0.0f, y0 = 0.0f, x1 = 0.0f, y1 = 0.0f;
    bool valid = false;
    float cx() const { return 0.5f * (x0 + x1); }  // center x — the synthetic click target.
    float cy() const { return 0.5f * (y0 + y1); }  // center y.
};
struct EditorUIProbe {
    std::vector<UiRect> hierarchyRows;  // one rect per Scene Hierarchy Selectable, view order.
    UiRect posX, posY, posZ;            // the three Position drag fields.
    UiRect eulerX, eulerY, eulerZ;      // the three Euler drag fields (radians).
    UiRect scaleX, scaleY, scaleZ;      // the three Scale drag fields.
    UiRect metallic, roughness;         // the Material scalar drag fields.
    UiRect baseColorCombo;              // the base-color texture-name combo.
};

// Build the docked editor panels for this frame from the live registry. Call between
// ImGui::NewFrame() and ImGui::Render(). `fbWidth`/`fbHeight` are the framebuffer size; the panels
// tile the frame deterministically (fixed split ratios from DefaultDockLayout) around a central
// viewport region the rendered scene shows through. EditorState (the selected entity) is defined in
// editor_panel_data.h. Inspector widget edits mutate `registry` through edit_ops (see the ED1 note
// above). `probe`, when non-null, receives this frame's widget rects (headless input synthesis).
//
// Slice ED5 (deterministic undo/redo): when `history` is non-null every inspector edit is applied
// through the RECORDED wrappers (edit_history.h — same ops, plus a before/after command on the
// stack) and the panel handles Ctrl+Z (Undo) / Ctrl+Y (Redo) while no text field is being edited.
// With the default (nullptr) the panel behaves EXACTLY as before ED5 — raw ops, no key handling —
// so existing callers compile + render byte-identically (the static --editor-shot golden).
void BuildEditorUI(ecs::Registry& registry, const scene::SceneResources& resources,
                   EditorState& state, uint32_t fbWidth, uint32_t fbHeight,
                   EditorUIProbe* probe = nullptr, EditHistory* history = nullptr);

}  // namespace hf::editor
