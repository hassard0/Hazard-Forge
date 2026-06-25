#include "editor/flow_editor_panels.h"

#include "imgui.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace hf::editor {

// --- Docked layout, built PROGRAMMATICALLY (deterministic; no imgui.ini). Mirrors editor_panels.cpp's
// BeginDocked: fixed rect, no move/resize/collapse, no bring-to-front shuffle, so neither cursor input nor
// a persisted .ini can perturb the frame — the node-graph editor is byte-stable run-to-run + cross-backend.
namespace {

bool BeginDocked(const char* title, ImVec2 pos, ImVec2 size, ImGuiWindowFlags extra = 0) {
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoCollapse |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus | extra;
    return ImGui::Begin(title, nullptr, flags);
}

// A node box's fill color by kind family: consts/inputs (sources) warm, arithmetic cool, stateful (Delay/
// Counter/Latch) purple, control (Select) green. Deterministic palette (no input/time dependence).
ImU32 KindColor(uint32_t kind) {
    switch (kind) {
        case flow::kConst:   return IM_COL32(110, 90, 55, 235);   // amber source
        case flow::kInput:   return IM_COL32(70, 110, 60, 235);   // green source
        case flow::kCounter: return IM_COL32(95, 70, 120, 235);   // purple stateful
        case flow::kDelay:   return IM_COL32(80, 70, 120, 235);   // purple stateful
        case flow::kLatch:   return IM_COL32(105, 65, 110, 235);  // purple stateful
        case flow::kSelect:  return IM_COL32(55, 100, 110, 235);  // teal control
        default:             return IM_COL32(55, 70, 100, 235);   // blue arithmetic (Add/Sub/Mul/Min/Max)
    }
}

}  // namespace

void BuildFlowEditorUI(const flow::Graph& graph, const FlowGraphView& view,
                       uint32_t fbWidth, uint32_t fbHeight, const FlowLayout& layout) {
    (void)layout;
    const float fbW = static_cast<float>(fbWidth);
    const float fbH = static_cast<float>(fbHeight);

    // --- Menu bar (fixed; renders fine without input). Reserve its height for the tiling below. ---
    float menuH = 0.0f;
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Hazard Forge")) {
            ImGui::MenuItem("Flow Graph Editor", nullptr, true);
            ImGui::EndMenu();
        }
        ImGui::TextDisabled("  |  visual scripting  |  deterministic flow VM");
        menuH = ImGui::GetWindowSize().y;
        ImGui::EndMainMenuBar();
    }

    const float top = menuH;
    const float bodyH = fbH - top;
    const float paletteW = fbW * 0.18f;     // left palette strip
    const float canvasX = paletteW;
    const float canvasW = fbW - paletteW;

    // --- Node PALETTE (left strip): the addable node kinds. Clicking a row is the editor's "add node"
    // affordance (flow_edit_ops::AddFlowNode); headless capture just shows the menu. ---
    if (BeginDocked("Node Palette", ImVec2(0.0f, top), ImVec2(paletteW, bodyH))) {
        ImGui::TextUnformatted("Add Node");
        ImGui::Separator();
        const uint32_t kinds[] = { flow::kConst, flow::kAdd, flow::kSub, flow::kMul, flow::kMin,
                                   flow::kMax, flow::kSelect, flow::kInput, flow::kCounter,
                                   flow::kDelay, flow::kLatch };
        for (uint32_t k : kinds) {
            ImGui::PushStyleColor(ImGuiCol_Header, KindColor(k));
            // Selectable as a button-like palette entry (selected=false; the headless shot just lists them).
            ImGui::Selectable(KindLabel(k), false);
            ImGui::PopStyleColor();
        }
        ImGui::Separator();
        ImGui::TextDisabled("Nodes: %zu", graph.nodes.size());
        ImGui::TextDisabled("Wires: %zu", view.wires.size());
        ImGui::TextDisabled("Grid: %d x %d", view.gridCols, view.gridRows);
    }
    ImGui::End();

    // --- CANVAS: the node graph. Draw wires first (behind), then node boxes (front), via the window's
    // draw list so geometry comes straight from the deterministic FlowGraphView (CPU-built). ---
    if (BeginDocked("Flow Graph", ImVec2(canvasX, top), ImVec2(canvasW, bodyH),
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        // Canvas-space origin: the view's integer coords are offset by the window's content top-left so the
        // graph sits inside the canvas panel.
        const ImVec2 o = ImGui::GetCursorScreenPos();
        auto P = [&](int x, int y) { return ImVec2(o.x + static_cast<float>(x),
                                                   o.y + static_cast<float>(y)); };

        // Wires: a cubic bezier from the parent output anchor to the child input slot (horizontal tangents
        // so the curve reads as a Blueprint wire). Color-tinted, drawn behind the boxes.
        for (const FlowWireView& w : view.wires) {
            const ImVec2 a = P(w.fromX, w.fromY);
            const ImVec2 b = P(w.toX, w.toY);
            const float dx = (b.x - a.x) * 0.5f + 24.0f;
            dl->AddBezierCubic(a, ImVec2(a.x + dx, a.y), ImVec2(b.x - dx, b.y), b,
                               IM_COL32(180, 190, 210, 220), 2.5f);
            // Endpoint pins.
            dl->AddCircleFilled(a, 3.5f, IM_COL32(210, 210, 160, 255));
            dl->AddCircleFilled(b, 3.5f, IM_COL32(160, 210, 210, 255));
        }

        // Node boxes: a filled rounded rect + border, the kind label, and the constArg for source kinds.
        for (const FlowNodeView& nv : view.nodes) {
            const ImVec2 tl = P(nv.x, nv.y);
            const ImVec2 br = P(nv.x + layout.boxW, nv.y + layout.boxH);
            dl->AddRectFilled(tl, br, KindColor(nv.kind), 6.0f);
            dl->AddRect(tl, br, IM_COL32(225, 230, 240, 255), 6.0f, 0, 1.5f);
            // Title.
            char title[64];
            std::snprintf(title, sizeof(title), "%s  #%u", nv.label.c_str(), nv.id);
            dl->AddText(ImVec2(tl.x + 8.0f, tl.y + 6.0f), IM_COL32(245, 245, 245, 255), title);
            // constArg for source kinds.
            if (ShowsConstArg(nv.kind)) {
                char sub[48];
                std::snprintf(sub, sizeof(sub), "= %d", static_cast<int>(nv.constArg));
                dl->AddText(ImVec2(tl.x + 8.0f, tl.y + 28.0f), IM_COL32(230, 220, 180, 255), sub);
            }
            // Column/row tag (bottom-left) so the deterministic layout is visible in the shot.
            char tag[32];
            std::snprintf(tag, sizeof(tag), "c%d r%d", nv.col, nv.row);
            dl->AddText(ImVec2(tl.x + 8.0f, br.y - 18.0f), IM_COL32(200, 205, 215, 200), tag);
        }
    }
    ImGui::End();
}

}  // namespace hf::editor
