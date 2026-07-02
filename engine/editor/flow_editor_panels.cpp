#include "editor/flow_editor_panels.h"

#include "imgui.h"

#include "editor/flow_edit_ops.h"   // Slice ED2: the pure-CPU write path the palette/canvas clicks drive.

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

// Record the LAST-submitted item's screen rect into a probe slot (no-op with a null slot; the ED1
// CaptureItemRect twin — editor_panels.cpp keeps its own static, so each panel TU carries a copy).
void CaptureItemRect(UiRect* slot) {
    if (!slot) return;
    const ImVec2 mn = ImGui::GetItemRectMin();
    const ImVec2 mx = ImGui::GetItemRectMax();
    *slot = UiRect{mn.x, mn.y, mx.x, mx.y, true};
}

}  // namespace

void BuildFlowEditorUI(const flow::Graph& graph, const FlowGraphView& view,
                       uint32_t fbWidth, uint32_t fbHeight, const FlowLayout& layout,
                       flow::Graph* editGraph, FlowEditorState* editState,
                       FlowEditorUIProbe* probe) {
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

    // --- Node PALETTE (left strip): the addable node kinds. Clicking a row IS the editor's "add node"
    // affordance (Slice ED2): the SAME Selectable the static shot draws now reports its click, and in
    // edit mode that click calls flow_edit_ops::AddFlowNode(kind) — the node lands at the deterministic
    // layout position when the caller re-runs BuildFlowGraphView. No new visual element -> the static
    // golden is byte-identical (a Selectable's pixels don't change from having its return value read). ---
    if (probe) probe->paletteEntries.clear();
    if (BeginDocked("Node Palette", ImVec2(0.0f, top), ImVec2(paletteW, bodyH))) {
        ImGui::TextUnformatted("Add Node");
        ImGui::Separator();
        const uint32_t kinds[] = { flow::kConst, flow::kAdd, flow::kSub, flow::kMul, flow::kMin,
                                   flow::kMax, flow::kSelect, flow::kInput, flow::kCounter,
                                   flow::kDelay, flow::kLatch };
        for (uint32_t k : kinds) {
            ImGui::PushStyleColor(ImGuiCol_Header, KindColor(k));
            // Selectable as a button-like palette entry; fires on click release (edit mode adds a node).
            if (ImGui::Selectable(KindLabel(k), false)) {
                if (editGraph) AddFlowNode(*editGraph, k, /*constArg=*/0);
            }
            if (probe) {
                UiRect r;
                CaptureItemRect(&r);
                probe->paletteEntries.push_back(r);
            }
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
        if (probe) {
            probe->nodeBoxes.assign(view.nodes.size(), UiRect{});
            probe->inputSlots.assign(view.nodes.size() * 3u, UiRect{});
        }
        for (const FlowNodeView& nv : view.nodes) {
            const ImVec2 tl = P(nv.x, nv.y);
            const ImVec2 br = P(nv.x + layout.boxW, nv.y + layout.boxH);
            dl->AddRectFilled(tl, br, KindColor(nv.kind), 6.0f);
            dl->AddRect(tl, br, IM_COL32(225, 230, 240, 255), 6.0f, 0, 1.5f);
            // Slice ED2: the canvas-selection highlight — drawn ONLY in edit mode with a live selection,
            // so the static --flow-editor-shot (editState == nullptr) stays byte-identical.
            if (editState && editState->selectedNode == static_cast<int>(nv.id)) {
                dl->AddRect(ImVec2(tl.x - 2.0f, tl.y - 2.0f), ImVec2(br.x + 2.0f, br.y + 2.0f),
                            IM_COL32(255, 210, 80, 255), 7.0f, 0, 2.5f);
            }
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
            // Probe rects: the node box + its input-slot anchors (the SAME slot geometry the wire
            // endpoints use: left edge, y = top + (slot+1)*boxH/4 — the data builder's inAnchor).
            if (probe) {
                const std::size_t ni = static_cast<std::size_t>(nv.id);
                if (ni < probe->nodeBoxes.size())
                    probe->nodeBoxes[ni] = UiRect{tl.x, tl.y, br.x, br.y, true};
                const uint32_t mask = flow::EdgeMask(nv.kind);
                for (uint32_t slot = 0; slot < 3u; ++slot) {
                    if (!(mask & (1u << slot))) continue;
                    const ImVec2 c = P(nv.x, nv.y + static_cast<int>(slot + 1u) * layout.boxH / 4);
                    const std::size_t si = ni * 3u + slot;
                    if (si < probe->inputSlots.size())
                        probe->inputSlots[si] = UiRect{c.x - 8.0f, c.y - 8.0f,
                                                       c.x + 8.0f, c.y + 8.0f, true};
                }
            }
        }

        // --- Slice ED2: canvas INTERACTION (edit mode only; the static shot passes nullptrs and skips
        // this entirely). One left click is hit-tested against the SAME deterministic geometry that was
        // just drawn, input slots first (they sit on a box's left edge):
        //   * a click within +/-8 px of a REAL input-slot anchor of a node OTHER than the selection,
        //     with a selection live, wires selected -> that slot via ConnectFlow (source stays selected);
        //   * otherwise a click inside a node box selects that node (lowest NodeId wins on overlap —
        //     boxes never overlap in the grid layout, this is just a deterministic tie-break);
        //   * a click on empty canvas clears the selection.
        // The Delete key deletes the selected node via DeleteFlowNode (ids shift -> selection clears).
        // The caller re-runs BuildFlowGraphView, so the next frame draws the post-edit layout. ---
        if (editGraph && editState) {
            if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(0)) {
                const ImVec2 m = ImGui::GetMousePos();
                bool consumed = false;
                // 1) Input slots (only meaningful with a different node selected as the wire SOURCE).
                if (editState->selectedNode >= 0) {
                    for (const FlowNodeView& nv : view.nodes) {
                        if (static_cast<int>(nv.id) == editState->selectedNode) continue;
                        const uint32_t mask = flow::EdgeMask(nv.kind);
                        for (uint32_t slot = 0; slot < 3u && !consumed; ++slot) {
                            if (!(mask & (1u << slot))) continue;
                            const ImVec2 c = P(nv.x, nv.y + static_cast<int>(slot + 1u) * layout.boxH / 4);
                            if (m.x >= c.x - 8.0f && m.x < c.x + 8.0f &&
                                m.y >= c.y - 8.0f && m.y < c.y + 8.0f) {
                                ConnectFlow(*editGraph,
                                            static_cast<flow::NodeId>(editState->selectedNode),
                                            nv.id, slot);
                                consumed = true;
                            }
                        }
                        if (consumed) break;
                    }
                }
                // 2) Node body -> select (or empty canvas -> clear).
                if (!consumed) {
                    int hit = -1;
                    for (const FlowNodeView& nv : view.nodes) {
                        const ImVec2 tl = P(nv.x, nv.y);
                        const ImVec2 br = P(nv.x + layout.boxW, nv.y + layout.boxH);
                        if (m.x >= tl.x && m.x < br.x && m.y >= tl.y && m.y < br.y) {
                            hit = static_cast<int>(nv.id);
                            break;
                        }
                    }
                    editState->selectedNode = hit;
                }
            }
            // Delete affordance: the Delete key on the canvas selection (no visual chrome -> the static
            // golden is untouched; the dry-run synthesizes the key event).
            if (editState->selectedNode >= 0 && ImGui::IsKeyPressed(ImGuiKey_Delete, false)) {
                DeleteFlowNode(*editGraph, static_cast<flow::NodeId>(editState->selectedNode));
                editState->selectedNode = -1;   // ids shifted -> the selection is void
            }
        }
    }
    ImGui::End();
}

// ====================================================================================================
// Issue #24 — LIVE EXECUTION FEEDBACK editor. Draws the SAME editor as BuildFlowEditorUI (boxes + wires +
// palette + the AUTHORED constArg badge) PLUS, on each node box, the node's LIVE evaluated value
// (values[node.id], the flow VM output from FlowLiveValues) as a distinct "= <value>" badge in a
// DIFFERENT color (cyan) than the authored constArg (amber) — so a node whose live result differs from
// its authored payload is visible at a glance: the author -> execute -> visualize Blueprint loop.
// Self-contained (the draw body mirrors BuildFlowEditorUI + the one extra live badge); ImGui geometry is
// CPU-built so it is byte-stable run-to-run + cross-backend for a fixed (view, values, w, h).
// ====================================================================================================
void BuildFlowEditorLiveUI(const flow::Graph& graph, const FlowGraphView& view,
                           const std::vector<flow::Reg>& values, int w, int h,
                           const FlowLayout& layout) {
    const float fbW = static_cast<float>(w);
    const float fbH = static_cast<float>(h);

    // --- Menu bar: same as BuildFlowEditorUI but flagged LIVE so the shot reads as the execution view. ---
    float menuH = 0.0f;
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Hazard Forge")) {
            ImGui::MenuItem("Flow Graph Editor (Live)", nullptr, true);
            ImGui::EndMenu();
        }
        ImGui::TextDisabled("  |  visual scripting  |  LIVE execution values");
        menuH = ImGui::GetWindowSize().y;
        ImGui::EndMainMenuBar();
    }

    const float top = menuH;
    const float bodyH = fbH - top;
    const float paletteW = fbW * 0.18f;     // left palette strip
    const float canvasX = paletteW;
    const float canvasW = fbW - paletteW;

    // --- Node PALETTE (left strip): identical to the static editor + a live read-back stat. ---
    if (BeginDocked("Node Palette", ImVec2(0.0f, top), ImVec2(paletteW, bodyH))) {
        ImGui::TextUnformatted("Add Node");
        ImGui::Separator();
        const uint32_t kinds[] = { flow::kConst, flow::kAdd, flow::kSub, flow::kMul, flow::kMin,
                                   flow::kMax, flow::kSelect, flow::kInput, flow::kCounter,
                                   flow::kDelay, flow::kLatch };
        for (uint32_t k : kinds) {
            ImGui::PushStyleColor(ImGuiCol_Header, KindColor(k));
            ImGui::Selectable(KindLabel(k), false);
            ImGui::PopStyleColor();
        }
        ImGui::Separator();
        ImGui::TextDisabled("Nodes: %zu", graph.nodes.size());
        ImGui::TextDisabled("Wires: %zu", view.wires.size());
        ImGui::TextDisabled("Grid: %d x %d", view.gridCols, view.gridRows);
        ImGui::Separator();
        ImGui::TextDisabled("Live values: %zu", values.size());
    }
    ImGui::End();

    // --- CANVAS: wires (behind) then node boxes (front), via the window draw list. ---
    if (BeginDocked("Flow Graph (Live)", ImVec2(canvasX, top), ImVec2(canvasW, bodyH),
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 o = ImGui::GetCursorScreenPos();
        auto P = [&](int x, int y) { return ImVec2(o.x + static_cast<float>(x),
                                                   o.y + static_cast<float>(y)); };

        // Wires (identical to the static editor).
        for (const FlowWireView& wv : view.wires) {
            const ImVec2 a = P(wv.fromX, wv.fromY);
            const ImVec2 b = P(wv.toX, wv.toY);
            const float dx = (b.x - a.x) * 0.5f + 24.0f;
            dl->AddBezierCubic(a, ImVec2(a.x + dx, a.y), ImVec2(b.x - dx, b.y), b,
                               IM_COL32(180, 190, 210, 220), 2.5f);
            dl->AddCircleFilled(a, 3.5f, IM_COL32(210, 210, 160, 255));
            dl->AddCircleFilled(b, 3.5f, IM_COL32(160, 210, 210, 255));
        }

        // Node boxes: same as the static editor PLUS the LIVE value badge.
        for (const FlowNodeView& nv : view.nodes) {
            const ImVec2 tl = P(nv.x, nv.y);
            const ImVec2 br = P(nv.x + layout.boxW, nv.y + layout.boxH);
            dl->AddRectFilled(tl, br, KindColor(nv.kind), 6.0f);
            dl->AddRect(tl, br, IM_COL32(225, 230, 240, 255), 6.0f, 0, 1.5f);
            // Title.
            char title[64];
            std::snprintf(title, sizeof(title), "%s  #%u", nv.label.c_str(), nv.id);
            dl->AddText(ImVec2(tl.x + 8.0f, tl.y + 6.0f), IM_COL32(245, 245, 245, 255), title);
            // AUTHORED constArg (amber) for source kinds — same as the static editor.
            if (ShowsConstArg(nv.kind)) {
                char sub[48];
                std::snprintf(sub, sizeof(sub), "= %d", static_cast<int>(nv.constArg));
                dl->AddText(ImVec2(tl.x + 8.0f, tl.y + 28.0f), IM_COL32(230, 220, 180, 255), sub);
            }
            // LIVE evaluated value (cyan) for EVERY node — the VM output flowing OUT of this node. A
            // different color than the amber authored badge so live != authored is visible at a glance.
            // Drawn on the box's right side so it never overlaps the authored badge.
            {
                const std::size_t idx = static_cast<std::size_t>(nv.id);
                char live[48];
                if (idx < values.size())
                    std::snprintf(live, sizeof(live), "= %d", static_cast<int>(values[idx]));
                else
                    std::snprintf(live, sizeof(live), "= ?");
                const ImVec2 sz = ImGui::CalcTextSize(live);
                dl->AddText(ImVec2(br.x - sz.x - 8.0f, tl.y + 28.0f),
                            IM_COL32(120, 235, 235, 255), live);   // cyan = LIVE (distinct from amber)
            }
            // Column/row tag (bottom-left), same as the static editor.
            char tag[32];
            std::snprintf(tag, sizeof(tag), "c%d r%d", nv.col, nv.row);
            dl->AddText(ImVec2(tl.x + 8.0f, br.y - 18.0f), IM_COL32(200, 205, 215, 200), tag);
        }
    }
    ImGui::End();
}

}  // namespace hf::editor
