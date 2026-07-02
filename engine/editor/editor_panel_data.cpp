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

    // --- Slice ED4: sanitize the multi-select set against the current entity count (written back,
    // like the primary clamp): drop out-of-range members, collapse a set shrunk below 2 back to
    // single-select, and re-anchor the primary into the set if the clamp moved it outside. Keeps
    // the edit_ops3.h invariant (empty, or size >= 2 sorted ascending containing the primary)
    // under entity-count changes. A no-op for the (default) empty set. ---
    {
        std::vector<int>& ms = state.multiSelection;
        ms.erase(std::remove_if(ms.begin(), ms.end(),
                                [count](int v) { return v < 0 || v >= count; }),
                 ms.end());
        if (ms.size() < 2) {
            if (ms.size() == 1) state.selectedEntity = ms[0];
            ms.clear();
        } else if (!SelectionContains(ms, state.selectedEntity)) {
            state.selectedEntity = ms.back();
        }
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
