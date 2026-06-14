#include "editor/editor_panels.h"

#include "imgui.h"

#include "scene/components.h"

#include <cstdio>
#include <string>
#include <vector>

namespace hf::editor {

namespace {

// Collect the scene's drawable entities (creation/view order) plus a display name derived from the
// entity's mesh resource name (e.g. "cube", "duck") suffixed with its index, else "Entity N".
struct EntityRow {
    ecs::Entity entity;
    std::string name;
};

std::vector<EntityRow> CollectEntities(ecs::Registry& reg,
                                       const scene::SceneResources& resources) {
    std::vector<EntityRow> rows;
    int i = 0;
    for (auto [e, tc, mc, mat] : reg.view<scene::TransformC, scene::MeshC, scene::MaterialC>()) {
        (void)tc; (void)mat;
        std::string meshName = mc.mesh ? resources.NameOfMesh(mc.mesh) : std::string();
        std::string label;
        if (!meshName.empty()) {
            label = meshName + " #" + std::to_string(i);
        } else {
            label = "Entity " + std::to_string(i);
        }
        rows.push_back({e, std::move(label)});
        ++i;
    }
    return rows;
}

}  // namespace

void BuildEditorUI(ecs::Registry& registry, const scene::SceneResources& resources,
                   EditorState& state, uint32_t fbWidth, uint32_t fbHeight) {
    std::vector<EntityRow> rows = CollectEntities(registry, resources);
    const int count = static_cast<int>(rows.size());
    if (state.selectedEntity < 0 && count > 0) state.selectedEntity = 0;
    if (state.selectedEntity >= count) state.selectedEntity = count - 1;

    // --- Menu bar (docking-branch nicety; renders fine without input). ---
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Hazard Forge")) {
            ImGui::MenuItem("Editor", nullptr, true);
            ImGui::EndMenu();
        }
        ImGui::TextDisabled("  |  live ECS scene");
        ImGui::EndMainMenuBar();
    }

    // --- Scene Hierarchy: list every drawable ECS entity; click selects (default = first). ---
    ImGui::SetNextWindowPos(ImVec2(16, 36), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(260, 320), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Scene Hierarchy")) {
        ImGui::Text("%d entities", count);
        ImGui::Separator();
        for (int i = 0; i < count; ++i) {
            const bool selected = (i == state.selectedEntity);
            if (ImGui::Selectable(rows[i].name.c_str(), selected)) {
                state.selectedEntity = i;
            }
        }
    }
    ImGui::End();

    // --- Inspector: the selected entity's TransformC + MaterialC, editable via Drag widgets. ---
    ImGui::SetNextWindowPos(ImVec2(290, 36), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 320), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Inspector")) {
        if (state.selectedEntity >= 0 && state.selectedEntity < count) {
            const EntityRow& row = rows[state.selectedEntity];
            ImGui::Text("Selected: %s", row.name.c_str());
            ImGui::Separator();

            auto& tc = registry.get<scene::TransformC>(row.entity);
            float pos[3]   = {tc.t.position.x, tc.t.position.y, tc.t.position.z};
            float euler[3] = {tc.t.eulerRadians.x, tc.t.eulerRadians.y, tc.t.eulerRadians.z};
            float scale[3] = {tc.t.scale.x, tc.t.scale.y, tc.t.scale.z};
            ImGui::TextUnformatted("Transform");
            if (ImGui::DragFloat3("Position", pos, 0.05f)) {
                tc.t.position = {pos[0], pos[1], pos[2]};
            }
            if (ImGui::DragFloat3("Euler (rad)", euler, 0.01f)) {
                tc.t.eulerRadians = {euler[0], euler[1], euler[2]};
            }
            if (ImGui::DragFloat3("Scale", scale, 0.05f)) {
                tc.t.scale = {scale[0], scale[1], scale[2]};
            }

            ImGui::Separator();
            ImGui::TextUnformatted("Material");
            auto& mat = registry.get<scene::MaterialC>(row.entity);
            ImGui::DragFloat("Metallic", &mat.metallic, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat("Roughness", &mat.roughness, 0.01f, 0.0f, 1.0f);
            const std::string base = resources.NameOfTexture(mat.base);
            ImGui::Text("Base color: %s", base.empty() ? "(none)" : base.c_str());
        } else {
            ImGui::TextDisabled("No entity selected.");
        }
    }
    ImGui::End();

    // --- Stats overlay: entity count, frame size, title. ---
    ImGui::SetNextWindowPos(ImVec2((float)fbWidth - 236.0f, 36.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(220, 110), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Stats")) {
        ImGui::TextUnformatted("Hazard Forge Editor");
        ImGui::Separator();
        ImGui::Text("Entities: %d", count);
        ImGui::Text("Frame: %u x %u", fbWidth, fbHeight);
        ImGui::Text("Alive: %zu", registry.aliveCount());
    }
    ImGui::End();
}

}  // namespace hf::editor
