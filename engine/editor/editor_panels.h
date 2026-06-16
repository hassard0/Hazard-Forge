#pragma once
// Hazard Forge — editor panels built from live ECS scene data (Dear ImGui rendering).
//
// Builds the DOCKED Dear ImGui editor (Scene Hierarchy / Inspector / Stats panels tiled around a
// central scene Viewport) each frame from the engine's ECS registry + the scene's named resources.
// No user input is required to populate them — they render the live TransformC/MaterialC/MeshC of the
// scene's entities, so a headless --editor-shot capture shows real scene state in a docked layout.
//
// The panel DATA (what each panel would display) is factored into editor_panel_data.{h,cpp} (pure
// CPU, ImGui-free, unit-tested + backend-agnostic, lives in hf_core); THIS module turns that data
// into Dear ImGui widgets in a deterministic tiled-docked layout. Depends on ecs/scene + ImGui only
// (no rhi/backend symbols).
#include <cstdint>

#include "ecs/ecs.h"
#include "editor/editor_panel_data.h"  // EditorState + the panel-data model (pure).
#include "scene/scene_io.h"

namespace hf::editor {

// Build the docked editor panels for this frame from the live registry. Call between
// ImGui::NewFrame() and ImGui::Render(). `fbWidth`/`fbHeight` are the framebuffer size; the panels
// tile the frame deterministically (fixed split ratios from DefaultDockLayout) around a central
// viewport region the rendered scene shows through. EditorState (the selected entity) is defined in
// editor_panel_data.h.
void BuildEditorUI(ecs::Registry& registry, const scene::SceneResources& resources,
                   EditorState& state, uint32_t fbWidth, uint32_t fbHeight);

}  // namespace hf::editor
