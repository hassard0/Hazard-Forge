#pragma once
// Hazard Forge — flow-graph EDITOR EDIT OPERATIONS (pure CPU, ImGui-free, backend-free).
//
// Issue #24 (Blueprint-class visual scripting), the editor WRITE path: deterministic, programmatic
// graph-mutation ops the node-graph editor's palette / connect / delete actions call. Mirrors
// edit_ops.{h,cpp} (the ECS editor's write path) but for flow::Graph instead of the ECS registry. Each op
// MUTATES the Graph deterministically and KEEPS NodeId stability where it matters: AddFlowNode appends
// (ids never recycled, matching flow.h's "NodeId == index, monotonic" contract); ConnectFlow rewires an
// input slot; DeleteFlowNode removes a node AND re-maps every surviving reference (so the graph stays
// well-formed and TopoOrder-valid — a dangling/self reference becomes the "no edge" sentinel).
//
// Header-only + SELF-CONTAINED (only flow/flow.h, which itself pulls <cstddef>/<cstdint>/<vector> +
// net/session.h) so it compiles STANDALONE with clang on the Mac exactly like flow.h / flow_editor_data.h.
// NO ImGui / rhi / ECS / float / hash containers. The view-digest after an edit is deterministic + CHANGES
// (proven by the golden test).

#include "flow/flow.h"

namespace hf::editor {

// AddFlowNode: append a node of `kind` carrying `constArg`, with all inputs set to the "no edge" sentinel
// (the node's OWN new id), and return its NodeId (== its new index). Append-only -> existing ids are
// untouched (the flow.h monotonic-id contract). The new node is a topo ROOT until ConnectFlow wires it.
inline flow::NodeId AddFlowNode(flow::Graph& g, uint32_t kind, flow::Reg constArg = 0) {
    const flow::NodeId id = static_cast<flow::NodeId>(g.nodes.size());
    flow::Node nd;
    nd.kind = kind;
    nd.a = nd.b = nd.c = id;      // unused inputs -> own id = the "no edge" sentinel (flow.h convention)
    nd.constArg = constArg;
    g.nodes.push_back(nd);
    return id;
}

// ConnectFlow: wire parent `from`'s output into node `to`'s input `slot` (0=a, 1=b, 2=c). A no-op (returns
// false) if `to` is out of range or `slot` > 2; otherwise sets the slot and returns true. Setting a slot
// that is not an EdgeMask edge for `to`'s kind is allowed but inert (TopoOrder/Evaluate ignore it) —
// matching the flow VM's "only EdgeMask bits are real edges" contract. Does NOT validate acyclicity here
// (TopoOrder deterministically rejects a cycle at evaluate time); the editor may form-then-fix.
inline bool ConnectFlow(flow::Graph& g, flow::NodeId from, flow::NodeId to, uint32_t slot) {
    if (static_cast<std::size_t>(to) >= g.nodes.size() || slot > 2u) return false;
    flow::Node& nd = g.nodes[static_cast<std::size_t>(to)];
    if (slot == 0u) nd.a = from;
    else if (slot == 1u) nd.b = from;
    else nd.c = from;
    return true;
}

// DisconnectFlow: reset node `to`'s input `slot` back to the "no edge" sentinel (its own id). Returns
// false on an out-of-range node or slot. The inverse of ConnectFlow for a single slot.
inline bool DisconnectFlow(flow::Graph& g, flow::NodeId to, uint32_t slot) {
    if (static_cast<std::size_t>(to) >= g.nodes.size() || slot > 2u) return false;
    if (slot == 0u) g.nodes[static_cast<std::size_t>(to)].a = to;
    else if (slot == 1u) g.nodes[static_cast<std::size_t>(to)].b = to;
    else g.nodes[static_cast<std::size_t>(to)].c = to;
    return true;
}

// DeleteFlowNode: remove node `victim` and RE-MAP every surviving node's input references so the graph
// stays well-formed (NodeId == index after the erase). Returns false on an out-of-range victim. Re-map
// rule for a reference `r` on a surviving node `self` (whose new index is `ns`):
//   - r == victim            -> ns (the node's OWN new id = "no edge"; the edge to the deleted node is cut)
//   - r >  victim            -> r - 1 (indices above the victim shift down by one)
//   - r <  victim            -> r unchanged
//   - r out of original range -> clamped to ns ("no edge") so it never dangles past the shrunk array
// This keeps TopoOrder valid (no reference points at a non-existent or shifted-wrong index) and is fully
// deterministic. Self-references (the "no edge" sentinel) are preserved as self-references under the remap.
inline bool DeleteFlowNode(flow::Graph& g, flow::NodeId victim) {
    const std::size_t n = g.nodes.size();
    if (static_cast<std::size_t>(victim) >= n) return false;

    // Build the surviving node list with re-mapped references in one pass.
    flow::Graph out;
    out.nodes.reserve(n - 1);
    for (std::size_t i = 0; i < n; ++i) {
        if (static_cast<flow::NodeId>(i) == victim) continue;     // drop the victim
        // New index of node i after the erase: shift down by one if it was above the victim.
        const flow::NodeId ns =
            (i > static_cast<std::size_t>(victim)) ? static_cast<flow::NodeId>(i - 1)
                                                   : static_cast<flow::NodeId>(i);
        flow::Node nd = g.nodes[i];
        auto remap = [&](flow::NodeId r) -> flow::NodeId {
            if (static_cast<std::size_t>(r) >= n) return ns;       // was out of range -> "no edge"
            if (r == victim) return ns;                            // edge to the deleted node -> "no edge"
            return (r > victim) ? static_cast<flow::NodeId>(r - 1) // shift indices above the victim down
                                : r;                               // indices below the victim unchanged
        };
        nd.a = remap(nd.a);
        nd.b = remap(nd.b);
        nd.c = remap(nd.c);
        out.nodes.push_back(nd);
    }
    g = out;
    return true;
}

}  // namespace hf::editor
