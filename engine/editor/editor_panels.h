#pragma once
// Hazard Forge — editor panels built from live ECS scene data.
//
// Builds the Dear ImGui panels (Scene Hierarchy, Inspector, Stats) each frame from the engine's ECS
// registry + the scene's named resources. No user input is required to populate them — they render
// the live TransformC/MaterialC/MeshC of the scene's entities, so a headless --shot capture shows
// real scene state. Depends only on ecs/scene + ImGui (no rhi/backend symbols).
#include <cstdint>

#include "ecs/ecs.h"
#include "scene/scene_io.h"

namespace hf::editor {

// Per-editor UI state that persists across frames (the selected entity in the hierarchy).
struct EditorState {
    int selectedEntity = 0;  // index into the scene's drawable-entity list; -1 = none
};

// Build the editor panels for this frame from the live registry. Call between ImGui::NewFrame() and
// ImGui::Render(). `fbWidth`/`fbHeight` are the framebuffer size (for the stats overlay + dockspace).
void BuildEditorUI(ecs::Registry& registry, const scene::SceneResources& resources,
                   EditorState& state, uint32_t fbWidth, uint32_t fbHeight);

}  // namespace hf::editor
