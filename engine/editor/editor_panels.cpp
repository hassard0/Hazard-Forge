#include "editor/editor_panels.h"

#include "imgui.h"

#include "editor/edit_ops.h"  // Slice ED1: the pure-CPU write path the inspector widgets drive.
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

// Record the LAST-submitted item's screen rect into a probe slot (no-op with a null slot).
void CaptureItemRect(UiRect* slot) {
    if (!slot) return;
    const ImVec2 mn = ImGui::GetItemRectMin();
    const ImVec2 mx = ImGui::GetItemRectMax();
    *slot = UiRect{mn.x, mn.y, mx.x, mx.y, true};
}

// Slice ED1 — one labelled 3-component drag row (the DragFloat3 look, but split into three explicit
// DragFloats so the probe can record each component's exact rect for synthetic input). Returns true
// if ANY component changed this frame (mouse drag or a Ctrl+click typed commit); `v` then holds the
// full new vector. Width math mirrors DragFloat3 (CalcItemWidth split three ways, inner spacing
// between components, the label on the same line).
bool DragRow3(const char* label, float v[3], float speed, UiRect* r0, UiRect* r1, UiRect* r2) {
    const ImGuiStyle& style = ImGui::GetStyle();
    const float fullW = ImGui::CalcItemWidth();
    const float compW = (fullW - 2.0f * style.ItemInnerSpacing.x) / 3.0f;
    UiRect* slots[3] = {r0, r1, r2};
    bool changed = false;
    ImGui::PushID(label);
    for (int c = 0; c < 3; ++c) {
        if (c > 0) ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
        ImGui::PushID(c);
        ImGui::SetNextItemWidth(compW);
        changed |= ImGui::DragFloat("", &v[c], speed);
        CaptureItemRect(slots[c]);
        ImGui::PopID();
    }
    ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
    ImGui::TextUnformatted(label);
    ImGui::PopID();
    return changed;
}

}  // namespace

void BuildEditorUI(ecs::Registry& registry, const scene::SceneResources& resources,
                   EditorState& state, uint32_t fbWidth, uint32_t fbHeight,
                   EditorUIProbe* probe, EditHistory* history) {
    // --- Slice ED5: the edit write paths, RECORDED when a history is wired (the recorded wrappers
    // capture before/after around the SAME raw op; with a null history the raw op runs directly, so
    // pre-ED5 callers behave byte-identically). All inspector edits below route through these. ---
    auto applyTransform = [&](int entity, const TransformEdit& e) {
        if (history) RecordedApplyTransformEdit(*history, registry, entity, e);
        else ApplyTransformEdit(registry, entity, e);
    };
    auto applyMaterial = [&](int entity, const MaterialEdit& e) {
        if (history) RecordedApplyMaterialEdit(*history, registry, entity, e);
        else ApplyMaterialEdit(registry, entity, e);
    };
    // --- Slice ED5: Ctrl+Z undo / Ctrl+Y redo (history mode only; suppressed while a text field is
    // active so ImGui's own InputText Ctrl+Z stays local to the field). Key handling only — zero
    // visual chrome, so the static shots are untouched. ---
    if (history) {
        const ImGuiIO& io = ImGui::GetIO();
        if (io.KeyCtrl && !io.WantTextInput) {
            const EditTargets targets{&registry, nullptr};
            if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) Undo(*history, targets);
            if (ImGui::IsKeyPressed(ImGuiKey_Y, false)) Redo(*history, targets);
        }
    }

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
        if (probe) probe->hierarchyRows.assign(static_cast<size_t>(count), UiRect{});
        for (int i = 0; i < count; ++i) {
            const bool selected = (i == state.selectedEntity);
            if (ImGui::Selectable(data.hierarchy[i].label.c_str(), selected)) {
                state.selectedEntity = i;
            }
            if (probe) CaptureItemRect(&probe->hierarchyRows[static_cast<size_t>(i)]);
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

    // --- Inspector (right column): the selected entity's Transform + Material + Mesh — Slice ED1:
    // LIVE EDITABLE. Each widget shows the panel-data value (re-read from the ECS every frame) and,
    // on a value change (drag or Ctrl+click typed commit), applies an ABSOLUTE SET of that field
    // through the existing pure-CPU edit ops — the same deterministic write path the programmatic
    // --editor-edit showcase uses, so widget edits persist through Ctrl+S / DumpScene. Applying per
    // value-change keeps mouse drags live (the ECS updates, the panel re-reads it: no snap-back) and
    // is a single set for a typed entry (the --ed1-dry-run route); the edit ops are absolute sets,
    // so per-frame application is idempotent with respect to the final value. ---
    if (BeginDocked(layout.inspectorTitle, ImVec2(rightX, top), ImVec2(rightW, bodyH))) {
        if (data.inspector.valid) {
            const InspectorData& in = data.inspector;
            ImGui::Text("Selected: %s", in.label.c_str());
            ImGui::Separator();
            ImGui::TextUnformatted("Transform");
            {
                float pos[3] = {in.position.x, in.position.y, in.position.z};
                if (DragRow3("Position", pos, 0.01f, probe ? &probe->posX : nullptr,
                             probe ? &probe->posY : nullptr, probe ? &probe->posZ : nullptr)) {
                    TransformEdit e;
                    e.setPosition = true;
                    e.position = {pos[0], pos[1], pos[2]};
                    applyTransform(in.index, e);
                }
                float eul[3] = {in.eulerRadians.x, in.eulerRadians.y, in.eulerRadians.z};
                if (DragRow3("Euler", eul, 0.01f, probe ? &probe->eulerX : nullptr,
                             probe ? &probe->eulerY : nullptr, probe ? &probe->eulerZ : nullptr)) {
                    TransformEdit e;
                    e.setEuler = true;
                    e.euler = {eul[0], eul[1], eul[2]};
                    applyTransform(in.index, e);
                }
                float scl[3] = {in.scale.x, in.scale.y, in.scale.z};
                if (DragRow3("Scale", scl, 0.01f, probe ? &probe->scaleX : nullptr,
                             probe ? &probe->scaleY : nullptr, probe ? &probe->scaleZ : nullptr)) {
                    TransformEdit e;
                    e.setScale = true;
                    e.scale = {scl[0], scl[1], scl[2]};
                    applyTransform(in.index, e);
                }
            }
            ImGui::Separator();
            ImGui::TextUnformatted("Mesh");
            ImGui::Text("%s", in.meshName.empty() ? "(none)" : in.meshName.c_str());
            ImGui::Separator();
            ImGui::TextUnformatted("Material");
            {
                const ImGuiStyle& style = ImGui::GetStyle();
                float metallic = in.metallic;
                ImGui::SetNextItemWidth(ImGui::CalcItemWidth());
                if (ImGui::DragFloat("##Metallic", &metallic, 0.005f, 0.0f, 1.0f)) {
                    MaterialEdit e;
                    e.setMetallic = true;
                    e.metallic = metallic;
                    applyMaterial(in.index, e);
                }
                if (probe) CaptureItemRect(&probe->metallic);
                ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
                ImGui::TextUnformatted("Metallic");

                float roughness = in.roughness;
                ImGui::SetNextItemWidth(ImGui::CalcItemWidth());
                if (ImGui::DragFloat("##Roughness", &roughness, 0.005f, 0.0f, 1.0f)) {
                    MaterialEdit e;
                    e.setRoughness = true;
                    e.roughness = roughness;
                    applyMaterial(in.index, e);
                }
                if (probe) CaptureItemRect(&probe->roughness);
                ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
                ImGui::TextUnformatted("Roughness");

                // Base color is a NAMED TEXTURE in this engine (MaterialC.base is an opaque
                // rhi::ITexture*, resolved by name from SceneResources — not an RGB factor), so the
                // edit widget is a texture-name combo, not a color picker. std::map iteration gives a
                // deterministic name order. Selecting a name applies the resolved pointer through
                // ApplyMaterialEdit (edit_ops never dereferences it).
                const char* preview = in.baseColorName.empty() ? "(none)" : in.baseColorName.c_str();
                ImGui::SetNextItemWidth(ImGui::CalcItemWidth());
                if (ImGui::BeginCombo("##BaseColor", preview)) {
                    for (const auto& [name, tex] : resources.textures) {
                        const bool selectedTex = (name == in.baseColorName);
                        if (ImGui::Selectable(name.c_str(), selectedTex)) {
                            MaterialEdit e;
                            e.setBaseColor = true;
                            e.baseColor = tex;
                            applyMaterial(in.index, e);
                        }
                    }
                    ImGui::EndCombo();
                }
                if (probe) CaptureItemRect(&probe->baseColorCombo);
                ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
                ImGui::TextUnformatted("Base color");
            }
        } else {
            ImGui::TextDisabled("No entity selected.");
        }
    }
    ImGui::End();
}

}  // namespace hf::editor
