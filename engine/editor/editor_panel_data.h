#pragma once
// Hazard Forge — editor PANEL DATA model (pure CPU, ImGui-free, backend-free).
//
// Slice BT (docked editor): the deterministic SEAM between "what each editor panel would display"
// and the Dear ImGui calls that draw it. BuildEditorUI (editor_panels.cpp) issues ImGui widgets;
// this file computes the DATA those widgets show, with ZERO ImGui / rhi / backend symbols, so it can
// be unit-tested headlessly (assert the data, not the pixels) and lives in hf_core alongside
// introspect.{h,cpp}. The docked layout is described here as fixed split ratios + fixed dock-node ids
// so the layout is itself a deterministic, testable VALUE (the golden's layout is code-driven, never
// loaded from a machine-dependent imgui.ini).
//
// Touches ONLY the ECS Registry, SceneResources (opaque named pointers, never dereferenced — same
// contract as introspect/scene_io), and math. NO vk*/Metal/rhi rendering symbols, NO imgui.h.

#include <cstdint>
#include <string>
#include <vector>

#include "ecs/ecs.h"
#include "math/math.h"
#include "scene/scene_io.h"

namespace hf::editor {

// Per-editor UI state that persists across frames (the selected entity in the hierarchy). This is the
// ONLY mutable editor state; everything else in PanelData is derived from the live registry each call.
struct EditorState {
    int selectedEntity = 0;  // index into the scene's drawable-entity list; -1 = none.
};

// One row in the Scene Hierarchy panel: a drawable entity + its display label. The label is the
// entity's mesh resource name suffixed with its view-order index (e.g. "duck #3"), else "Entity N".
struct HierarchyRow {
    ecs::Entity entity;     // the ECS handle (id + generation).
    std::string label;      // human-readable hierarchy label.
};

// The Inspector panel's view of the selected entity (TransformC / MaterialC / MeshC). `valid` is false
// when nothing is selected (selectedEntity < 0 or no drawable entities), in which case the Inspector
// shows "No entity selected." and the other fields are defaults.
struct InspectorData {
    bool valid = false;
    int  index = -1;            // the selected view-order index this reflects.
    std::string label;          // the selected entity's hierarchy label.
    std::string meshName;       // mesh resource name ("" if none).
    std::string baseColorName;  // base-color texture name ("" if none).
    math::Vec3 position{0, 0, 0};
    math::Vec3 eulerRadians{0, 0, 0};
    math::Vec3 scale{1, 1, 1};
    float metallic = 0.0f;
    float roughness = 0.5f;
};

// The Stats panel's counts (derived from the live registry).
struct StatsData {
    int    entityCount = 0;   // drawable entities (TransformC + MeshC + MaterialC).
    int    meshCount = 0;     // entities carrying a MeshC.
    size_t aliveCount = 0;    // total alive ECS entities.
};

// The complete, deterministic data behind one editor frame's panels. BuildPanelData clamps the
// selection into range (or to "none") and fills the three panel views; BuildEditorUI then renders
// them. Two calls on an identical registry/state yield byte-identical PanelData (determinism).
struct PanelData {
    std::vector<HierarchyRow> hierarchy;
    InspectorData inspector;
    StatsData stats;
};

// Compute the panel data for one frame from the live registry. CLAMPS state.selectedEntity into
// [0, count) when at least one entity exists (a >= count selection clamps to the last; a < 0
// selection with entities present snaps to 0); leaves it < 0 / inspector.valid=false only when the
// scene has no drawable entities. The clamp is WRITTEN BACK into `state` so the persistent selection
// stays in range across frames (matching the live --fly behaviour). Pure CPU; no ImGui/rhi/backend.
PanelData BuildPanelData(ecs::Registry& registry, const scene::SceneResources& resources,
                         EditorState& state);

// --- Docked layout description (fixed, code-driven — the golden's layout is a deterministic VALUE) --
//
// The DockSpace is split deterministically every run (no imgui.ini): the central node holds the
// Viewport (the rendered scene), a left column holds the Scene Hierarchy, a right column holds the
// Inspector, and a bottom strip of the left column holds Stats. The split RATIOS are fixed constants
// so the layout is byte-stable run-to-run and identical across backends (ImGui geometry is CPU-built).
struct DockLayout {
    // Fraction of the full width peeled off the LEFT for the Hierarchy/Stats column.
    float leftRatio = 0.22f;
    // Fraction of the REMAINING width peeled off the RIGHT for the Inspector column.
    float rightRatio = 0.28f;
    // Fraction of the LEFT column's height peeled off the BOTTOM for the Stats panel.
    float leftBottomRatio = 0.45f;
    // Window titles docked into each region (must match the ImGui::Begin titles in BuildEditorUI).
    const char* viewportTitle  = "Viewport";
    const char* hierarchyTitle = "Scene Hierarchy";
    const char* inspectorTitle = "Inspector";
    const char* statsTitle      = "Stats";
};

// The single shared layout instance (fixed constants). Both the layout builder and the test read it.
DockLayout DefaultDockLayout();

}  // namespace hf::editor
