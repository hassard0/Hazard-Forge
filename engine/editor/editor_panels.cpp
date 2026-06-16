#include "editor/editor_panels.h"

#include "imgui.h"

#include "editor/editor_panel_data.h"

#include <cstdio>
#include <string>
#include <vector>

namespace hf::editor {

// --- Docked layout, built PROGRAMMATICALLY (deterministic; no imgui.ini). -------------------------
//
// The vendored Dear ImGui (1.91.8, master branch) does NOT include the docking feature / DockBuilder
// API, so the docked layout is realized by TILING fixed-position, fixed-size panels around a central
// viewport — every Begin uses ImGuiCond_Always with positions/sizes derived from the framebuffer size
// and the FIXED DockLayout split ratios (DefaultDockLayout). The windows are NoMove|NoResize|
// NoCollapse so neither cursor input nor a persisted .ini can perturb the layout: the result is a
// proper docked editor frame (Scene Hierarchy left, Inspector right, Stats bottom-left, Viewport
// center) that is byte-stable run-to-run and identical across backends (ImGui geometry is CPU-built).
//
// Layout (fb = fbWidth x fbHeight, below the menu bar of height `menuH`):
//   left column    = [0, leftW)                 width  = leftRatio * fb.w
//   right column   = [fb.w - rightW, fb.w)      width  = rightRatio * (fb.w - leftW)
//   center viewport= [leftW, fb.w - rightW)     the rendered scene shows through here
//   Hierarchy      = left column, top      (height = (1-leftBottomRatio) * column height)
//   Stats          = left column, bottom   (height = leftBottomRatio    * column height)
//   Inspector      = right column, full height
// The split ratios live in editor_panel_data.h (DefaultDockLayout) so the layout is a deterministic,
// unit-testable VALUE shared with the panel-data test.

namespace {

// A docked panel window: fixed rect, no move/resize/collapse, no title-bar bring-to-front shuffling.
bool BeginDocked(const char* title, ImVec2 pos, ImVec2 size) {
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoCollapse |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus;
    return ImGui::Begin(title, nullptr, flags);
}

}  // namespace

void BuildEditorUI(ecs::Registry& registry, const scene::SceneResources& resources,
                   EditorState& state, uint32_t fbWidth, uint32_t fbHeight) {
    // --- Panel DATA (pure, ImGui-free, unit-tested): hierarchy rows + inspector + stats. ---
    const PanelData data = BuildPanelData(registry, resources, state);
    const DockLayout layout = DefaultDockLayout();
    const int count = static_cast<int>(data.hierarchy.size());

    // --- Menu bar (fixed; renders fine without input). Reserve its height for the tiling below. ---
    float menuH = 0.0f;
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Hazard Forge")) {
            ImGui::MenuItem("Editor", nullptr, true);
            ImGui::EndMenu();
        }
        ImGui::TextDisabled("  |  docked editor  |  live ECS scene");
        menuH = ImGui::GetWindowSize().y;
        ImGui::EndMainMenuBar();
    }

    // --- Compute the docked tile rects from the FIXED split ratios. ---
    const float fbW = static_cast<float>(fbWidth);
    const float fbH = static_cast<float>(fbHeight);
    const float top = menuH;
    const float bodyH = fbH - top;

    const float leftW  = fbW * layout.leftRatio;
    const float rightW = (fbW - leftW) * layout.rightRatio;
    const float centerX = leftW;
    const float centerW = fbW - leftW - rightW;
    const float rightX  = fbW - rightW;

    const float leftBottomH = bodyH * layout.leftBottomRatio;
    const float leftTopH    = bodyH - leftBottomH;

    // --- Central Viewport panel: frames the region the rendered scene shows through. The scene was
    // drawn first (the editor chrome is composited over it), so this panel intentionally has NO
    // background — only a labeled border framing the live scene viewport. ---
    {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));  // transparent: scene shows through.
        ImGui::SetNextWindowBgAlpha(0.0f);
        if (BeginDocked(layout.viewportTitle, ImVec2(centerX, top), ImVec2(centerW, bodyH))) {
            ImGui::TextDisabled("Scene viewport (%d entities)", count);
        }
        ImGui::End();
        ImGui::PopStyleColor();
    }

    // --- Scene Hierarchy: every drawable ECS entity; the selected row is highlighted. ---
    if (BeginDocked(layout.hierarchyTitle, ImVec2(0.0f, top), ImVec2(leftW, leftTopH))) {
        ImGui::Text("%d entities", count);
        ImGui::Separator();
        for (int i = 0; i < count; ++i) {
            const bool selected = (i == state.selectedEntity);
            if (ImGui::Selectable(data.hierarchy[i].label.c_str(), selected)) {
                state.selectedEntity = i;
            }
        }
    }
    ImGui::End();

    // --- Stats (bottom of the left column): entity / mesh / alive counts + frame size. ---
    if (BeginDocked(layout.statsTitle, ImVec2(0.0f, top + leftTopH), ImVec2(leftW, leftBottomH))) {
        ImGui::TextUnformatted("Hazard Forge Editor");
        ImGui::Separator();
        ImGui::Text("Entities: %d", data.stats.entityCount);
        ImGui::Text("Meshes: %d", data.stats.meshCount);
        ImGui::Text("Alive: %zu", data.stats.aliveCount);
        ImGui::Text("Frame: %u x %u", fbWidth, fbHeight);
    }
    ImGui::End();

    // --- Inspector (right column): the selected entity's Transform + Material + Mesh (read-only). ---
    if (BeginDocked(layout.inspectorTitle, ImVec2(rightX, top), ImVec2(rightW, bodyH))) {
        if (data.inspector.valid) {
            const InspectorData& in = data.inspector;
            ImGui::Text("Selected: %s", in.label.c_str());
            ImGui::Separator();
            ImGui::TextUnformatted("Transform");
            ImGui::Text("Position: %.2f, %.2f, %.2f", in.position.x, in.position.y, in.position.z);
            ImGui::Text("Euler:    %.2f, %.2f, %.2f", in.eulerRadians.x, in.eulerRadians.y,
                        in.eulerRadians.z);
            ImGui::Text("Scale:    %.2f, %.2f, %.2f", in.scale.x, in.scale.y, in.scale.z);
            ImGui::Separator();
            ImGui::TextUnformatted("Mesh");
            ImGui::Text("%s", in.meshName.empty() ? "(none)" : in.meshName.c_str());
            ImGui::Separator();
            ImGui::TextUnformatted("Material");
            ImGui::Text("Metallic:  %.2f", in.metallic);
            ImGui::Text("Roughness: %.2f", in.roughness);
            ImGui::Text("Base color: %s",
                        in.baseColorName.empty() ? "(none)" : in.baseColorName.c_str());
        } else {
            ImGui::TextDisabled("No entity selected.");
        }
    }
    ImGui::End();
}

}  // namespace hf::editor
