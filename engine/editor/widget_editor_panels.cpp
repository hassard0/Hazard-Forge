#include "editor/widget_editor_panels.h"

#include "imgui.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace hf::editor {

// --- Docked layout, built PROGRAMMATICALLY (deterministic; no imgui.ini). Mirrors seq_editor_panels.cpp's
// BeginDocked: fixed rect, no move/resize/collapse, no bring-to-front shuffle, so neither cursor input nor a
// persisted .ini can perturb the frame — the widget designer is byte-stable run-to-run + cross-backend.
namespace {

bool BeginDocked(const char* title, ImVec2 pos, ImVec2 size, ImGuiWindowFlags extra = 0) {
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoCollapse |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus | extra;
    return ImGui::Begin(title, nullptr, flags);
}

// A per-depth accent color for the hierarchy + preview, cycled by tree depth (deterministic — no input/time
// dependence). Cool-to-warm so nesting levels read distinctly.
ImU32 DepthColor(int depth) {
    switch (((depth % 5) + 5) % 5) {
        case 0:  return IM_COL32(120, 175, 235, 255);   // blue   (root)
        case 1:  return IM_COL32(110, 200, 130, 255);   // green
        case 2:  return IM_COL32(230, 190, 100, 255);   // amber
        case 3:  return IM_COL32(210, 130, 200, 255);   // magenta
        default: return IM_COL32(120, 210, 205, 255);   // teal
    }
}

// A short human label for a widget kind (caller-defined; the showcase uses 0=Panel, 1=Text). Unknown kinds
// fall back to a numbered "Kind N" so the designer never shows a blank row.
const char* KindLabel(uint32_t kind, char* scratch, std::size_t cap) {
    switch (kind) {
        case 0u: return "Panel";
        case 1u: return "Text";
        case 2u: return "Image";
        case 3u: return "Button";
        default: std::snprintf(scratch, cap, "Kind %u", kind); return scratch;
    }
}

}  // namespace

void BuildWidgetEditorUI(const Tree& tree, const WidgetEditorView& view,
                         uint32_t fbWidth, uint32_t fbHeight, const WidgetEditorLayout& layout) {
    (void)layout;
    const float fbW = static_cast<float>(fbWidth);
    const float fbH = static_cast<float>(fbHeight);

    // --- Menu bar (fixed; renders fine without input). Reserve its height for the tiling below. ---
    float menuH = 0.0f;
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Hazard Forge")) {
            ImGui::MenuItem("UMG Widget Designer", nullptr, true);
            ImGui::EndMenu();
        }
        ImGui::TextDisabled("  |  widget designer  |  deterministic integer box-model");
        menuH = ImGui::GetWindowSize().y;
        ImGui::EndMainMenuBar();
    }

    const float top = menuH;
    const float bodyH = fbH - top;
    const float leftW = fbW * 0.27f;        // left column: hierarchy (top) + inspector (bottom)
    const float canvasX = leftW;
    const float canvasW = fbW - leftW;
    const float hierH = bodyH * 0.55f;      // hierarchy panel height inside the left column
    const float inspH = bodyH - hierH;      // inspector panel height

    // --- HIERARCHY panel (top-left): the widget tree as indented, kind-labeled rows with selection. ---
    if (BeginDocked("Hierarchy", ImVec2(0.0f, top), ImVec2(leftW, hierH))) {
        ImGui::TextUnformatted("Widget Tree");
        ImGui::SameLine();
        ImGui::TextDisabled("(%d widgets)", view.widgetCount);
        ImGui::Separator();
        for (const WidgetTreeRow& r : view.rows) {
            char scratch[32];
            const uint32_t kind = r.id < tree.widgets.size() ? tree.widgets[r.id].kind : r.kind;
            const char* kindName = KindLabel(kind, scratch, sizeof(scratch));
            // Indent by depth; color by depth; mark the selection with a leading caret + highlight.
            ImGui::Indent(static_cast<float>(r.depth) * 16.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, DepthColor(r.depth));
            char rowlbl[80];
            std::snprintf(rowlbl, sizeof(rowlbl), "%s%s  #%u",
                          r.selected ? "> " : "  ", kindName, r.id);
            if (r.selected) {
                // Selection highlight band behind the row.
                const ImVec2 p = ImGui::GetCursorScreenPos();
                ImDrawList* dl = ImGui::GetWindowDrawList();
                dl->AddRectFilled(ImVec2(p.x - 4.0f, p.y - 1.0f),
                                  ImVec2(p.x + leftW, p.y + ImGui::GetTextLineHeight() + 1.0f),
                                  IM_COL32(60, 90, 130, 160), 3.0f);
            }
            ImGui::Selectable(rowlbl, r.selected);
            ImGui::PopStyleColor();
            ImGui::Unindent(static_cast<float>(r.depth) * 16.0f);
        }
    }
    ImGui::End();

    // --- INSPECTOR panel (bottom-left): the selected widget's Style fields as name : value rows. ---
    if (BeginDocked("Inspector", ImVec2(0.0f, top + hierH), ImVec2(leftW, inspH))) {
        if (view.selected == hf::ui::kNoWidget || view.inspector.empty()) {
            ImGui::TextDisabled("No widget selected");
        } else {
            ImGui::Text("Selected: widget #%u", view.selected);
            ImGui::Separator();
            ImGui::Columns(2, "inspcols", false);
            ImGui::SetColumnWidth(0, leftW * 0.5f);
            for (const InspectorField& f : view.inspector) {
                ImGui::TextDisabled("%s", f.name);
                ImGui::NextColumn();
                ImGui::Text("%d", f.value);
                ImGui::NextColumn();
            }
            ImGui::Columns(1);
        }
    }
    ImGui::End();

    // --- LAYOUT PREVIEW canvas (right): the scaled SolveLayout rects as nested boxes; selection on top. ---
    if (BeginDocked("Layout Preview", ImVec2(canvasX, top), ImVec2(canvasW, bodyH),
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImGui::TextDisabled("Computed layout (SolveLayout, integer pixels, %d boxes)",
                            static_cast<int>(view.boxes.size()));
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 o = ImGui::GetCursorScreenPos();
        auto P = [&](int x, int y) { return ImVec2(o.x + static_cast<float>(x),
                                                   o.y + static_cast<float>(y)); };

        // Boxes are emitted in widget-index order (parent before child), so drawing in order nests
        // correctly (children paint over their parents). The selected box is drawn LAST so it's on top.
        const WidgetLayoutBox* sel = nullptr;
        for (const WidgetLayoutBox& b : view.boxes) {
            if (b.selected) { sel = &b; continue; }
            // depth-tint the fill by id parity for visual separation; the rect outline carries the box.
            const ImVec2 tl = P(b.x, b.y);
            const ImVec2 br = P(b.x + b.w, b.y + b.h);
            const ImU32 fill = (b.id & 1u) ? IM_COL32(40, 46, 58, 90) : IM_COL32(32, 37, 48, 90);
            dl->AddRectFilled(tl, br, fill, 3.0f);
            dl->AddRect(tl, br, IM_COL32(120, 130, 150, 220), 3.0f, 0, 1.5f);
            // id label tucked into the top-left of each box.
            char idlbl[16];
            std::snprintf(idlbl, sizeof(idlbl), "#%u", b.id);
            dl->AddText(ImVec2(tl.x + 3.0f, tl.y + 2.0f), IM_COL32(170, 180, 200, 230), idlbl);
        }
        if (sel) {
            const ImVec2 tl = P(sel->x, sel->y);
            const ImVec2 br = P(sel->x + sel->w, sel->y + sel->h);
            dl->AddRectFilled(tl, br, IM_COL32(70, 110, 160, 120), 3.0f);
            dl->AddRect(tl, br, IM_COL32(120, 200, 255, 255), 3.0f, 0, 2.5f);
            char idlbl[24];
            std::snprintf(idlbl, sizeof(idlbl), "#%u (selected)", sel->id);
            dl->AddText(ImVec2(tl.x + 3.0f, tl.y + 2.0f), IM_COL32(220, 240, 255, 255), idlbl);
        }
    }
    ImGui::End();
}

}  // namespace hf::editor
