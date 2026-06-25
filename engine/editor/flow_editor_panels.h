#pragma once
// Hazard Forge — flow-graph node editor panels (Dear ImGui rendering of the FlowGraphView).
//
// Issue #24 (Blueprint-class visual scripting), the editor's ImGui layer: turns the deterministic,
// ImGui-free FlowGraphView (flow_editor_data.h) into Dear ImGui widgets — labeled node boxes positioned by
// the topological-rank layout, wires drawn between input/output slots, and a node palette. This is the
// flow-VM twin of editor_panels.cpp (BuildEditorUI): the DATA (the view + the layout) is computed in
// hf_core and unit-tested headlessly; THIS module only issues ImGui draw calls over it. Depends on ImGui +
// flow_editor_data.h only (no rhi/backend symbols).

#include <cstdint>
#include <vector>

#include "editor/flow_editor_data.h"   // FlowGraphView / FlowLayout (the pure deterministic view)
#include "flow/flow.h"                 // flow::Graph / flow::Reg (palette add semantics + live values)

namespace hf::editor {

// Build the docked flow-graph editor for this frame from a pre-laid-out FlowGraphView. Call between
// ImGui::NewFrame() and ImGui::Render(). `fbWidth`/`fbHeight` are the framebuffer size; the canvas panel
// fills the frame and draws the node graph (boxes + wires) via the ImGui draw list, with a left palette
// strip listing the addable node kinds and a stats line. `graph` is shown for read-back (node/kind counts);
// the view is the authority on geometry. Deterministic given the same view (ImGui geometry is CPU-built).
void BuildFlowEditorUI(const flow::Graph& graph, const FlowGraphView& view,
                       uint32_t fbWidth, uint32_t fbHeight, const FlowLayout& layout = FlowLayout{});

// Issue #24 — LIVE EXECUTION FEEDBACK editor: the SAME node-graph editor as BuildFlowEditorUI PLUS each
// node's LIVE evaluated value (the flow VM output, values[node.id] from FlowLiveValues) drawn as a distinct
// "= <value>" badge in a DIFFERENT color than the authored constArg, so a node whose live result differs
// from its authored payload reads at a glance (the author -> execute -> visualize Blueprint loop). `values`
// is NodeId-indexed (values[i] is node i's computed output); a short `values` shows "= ?" for missing slots
// (never UB). Deterministic given the same view + values (ImGui geometry is CPU-built). Headless-renderable.
void BuildFlowEditorLiveUI(const flow::Graph& graph, const FlowGraphView& view,
                           const std::vector<flow::Reg>& values, int w, int h,
                           const FlowLayout& layout = FlowLayout{});

}  // namespace hf::editor
