// Hazard Forge — editor PANEL DATA model (pure CPU). See editor_panel_data.h.
#include "editor/editor_panel_data.h"

#include "scene/components.h"

#include <string>

namespace hf::editor {

PanelData BuildPanelData(ecs::Registry& registry, const scene::SceneResources& resources,
                         EditorState& state) {
    using scene::MaterialC;
    using scene::MeshC;
    using scene::TransformC;

    PanelData out;

    // --- Hierarchy: every drawable entity (view order) + a display label. ---
    int i = 0;
    for (auto [e, tc, mc, mat] : registry.view<TransformC, MeshC, MaterialC>()) {
        (void)tc; (void)mat;
        std::string meshName = mc.mesh ? resources.NameOfMesh(mc.mesh) : std::string();
        std::string label = meshName.empty() ? ("Entity " + std::to_string(i))
                                             : (meshName + " #" + std::to_string(i));
        out.hierarchy.push_back({e, std::move(label)});
        ++i;
    }
    const int count = static_cast<int>(out.hierarchy.size());

    // --- Selection clamp (written back so the persistent selection stays valid across frames). ---
    if (count == 0) {
        state.selectedEntity = -1;
    } else {
        if (state.selectedEntity < 0) state.selectedEntity = 0;
        if (state.selectedEntity >= count) state.selectedEntity = count - 1;
    }

    // --- Inspector: the selected entity's transform + material + mesh. ---
    if (state.selectedEntity >= 0 && state.selectedEntity < count) {
        const HierarchyRow& row = out.hierarchy[state.selectedEntity];
        const auto& tc = registry.get<TransformC>(row.entity);
        const auto& mc = registry.get<MeshC>(row.entity);
        const auto& mat = registry.get<MaterialC>(row.entity);

        out.inspector.valid = true;
        out.inspector.index = state.selectedEntity;
        out.inspector.label = row.label;
        out.inspector.meshName = mc.mesh ? resources.NameOfMesh(mc.mesh) : std::string();
        out.inspector.baseColorName = mat.base ? resources.NameOfTexture(mat.base) : std::string();
        out.inspector.position = tc.t.position;
        out.inspector.eulerRadians = tc.t.eulerRadians;
        out.inspector.scale = tc.t.scale;
        out.inspector.metallic = mat.metallic;
        out.inspector.roughness = mat.roughness;
    }

    // --- Stats: per-component counts. ---
    out.stats.entityCount = count;
    int meshCount = 0;
    for (auto [e, c] : registry.view<MeshC>()) { (void)e; (void)c; ++meshCount; }
    out.stats.meshCount = meshCount;
    out.stats.aliveCount = registry.aliveCount();

    return out;
}

DockLayout DefaultDockLayout() { return DockLayout{}; }

}  // namespace hf::editor
