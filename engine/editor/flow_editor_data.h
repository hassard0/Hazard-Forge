#pragma once
// Hazard Forge — flow-graph EDITOR DATA model (pure CPU, ImGui-free, backend-free).
//
// Issue #24 (Blueprint-class visual scripting), the EDITOR half: the deterministic SEAM between "how a
// flow::Graph should be DRAWN as a node graph" and the Dear ImGui calls that draw it. This is the EXACT
// editor_panel_data.{h,cpp} discipline applied to the flow VM (engine/flow/flow.h): BuildFlowGraphView
// computes a byte-identical VIEW (node boxes positioned on a fixed integer grid by topological rank +
// wires per EdgeMask) from a Graph, with ZERO ImGui / rhi / backend symbols, so it is unit-tested
// headlessly (assert the layout + a pinned FNV-1a-64 digest, NOT pixels) and lives in hf_core.
//
// The flow VM already pins ONE canonical topological order (flow::TopoOrder — Kahn's algorithm,
// lowest-NodeId-first tie-break). The LAYOUT reuses that determinism: a node's COLUMN is its longest-path
// rank from the roots (computed over the same EdgeMask edges), its ROW is its appearance order within that
// rank in the canonical TopoOrder. Same graph -> byte-identical view -> a deterministic golden. UE5
// Blueprint node layout is editor-state / mouse-driven (no two layouts agree); this is a code-driven VALUE.
//
// Touches ONLY flow/flow.h (the Graph + TopoOrder + EdgeMask) and net/session.h (DigestBytes). NO
// vk*/Metal/rhi rendering symbols, NO imgui.h, NO <cmath>/<algorithm>/float/clock/RNG/hash containers.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "flow/flow.h"        // hf::flow::Graph / Node / NodeId / Kind / EdgeMask / TopoOrder / IsRealEdge
#include "net/session.h"      // hf::net::DigestBytes — the pinned-golden FNV-1a-64 currency

namespace hf::editor {

// ---- Layout parameters (fixed integer grid; the layout is a deterministic VALUE) -------------------
// A node occupies a boxW x boxH cell; columns are colStride apart on X, rows rowStride apart on Y, with a
// fixed (originX, originY) top-left margin. All integers -> byte-stable cross-platform (no float layout).
struct FlowLayout {
    int originX   = 40;    // left margin of column 0's boxes
    int originY   = 40;    // top margin of row 0's boxes
    int colStride = 200;   // horizontal distance between column origins (column = topo rank)
    int rowStride = 110;   // vertical distance between row origins (row = order within a rank)
    int boxW      = 150;   // node box width
    int boxH      = 64;    // node box height
};

// One drawn node: its id, its top-left box position (x,y) on the integer grid, its grid cell (col,row),
// its flow Kind, the kind's human label, and (for kConst/kInput/kCounter) the constArg shown in the box.
struct FlowNodeView {
    flow::NodeId id   = 0;
    int          x    = 0;     // box top-left X (== originX + col*colStride)
    int          y    = 0;     // box top-left Y (== originY + row*rowStride)
    int          col  = 0;     // topological rank (longest-path from roots)
    int          row  = 0;     // order within the rank (canonical TopoOrder appearance order)
    uint32_t     kind = flow::kConst;
    std::string  label;        // kind name (e.g. "Add", "Const", "Counter")
    flow::Reg    constArg = 0; // the kConst/kInput/kCounter payload (shown in the box; 0 otherwise)
};

// One drawn wire: a real input edge (per EdgeMask) from parent `from`'s output to this node `to`'s input
// `slot` (0=a, 1=b, 2=c). The endpoints are pixel coordinates: (fromX,fromY) the parent box's output
// anchor (right-center), (toX,toY) this node's input-slot anchor (left edge, vertically spread by slot).
struct FlowWireView {
    int          fromX = 0, fromY = 0;   // parent output anchor (right-center of the parent box)
    int          toX   = 0, toY   = 0;   // child input-slot anchor (left edge of the child box)
    flow::NodeId from  = 0, to    = 0;   // the parent / child NodeIds
    uint32_t     slot  = 0;              // which input slot of `to` this edge feeds (0=a,1=b,2=c)
};

// The complete, deterministic view of a flow::Graph laid out for the editor. Two calls on an identical
// graph yield BYTE-IDENTICAL views (determinism); the digest below pins it as the golden.
struct FlowGraphView {
    std::vector<FlowNodeView> nodes;   // one per Graph node, in NodeId order (nodes[i].id == i)
    std::vector<FlowWireView> wires;   // one per real input edge, in a fixed (child asc, slot asc) order
    int gridCols = 0;                  // number of columns used (max col + 1, 0 for empty)
    int gridRows = 0;                  // tallest column's row count (max row + 1, 0 for empty)
};

// ---- KindLabel: the fixed display name for a flow Kind (the box label). Stable forever (part of the
// view + the golden). An unknown kind -> "Node" (deterministic fallback).
inline const char* KindLabel(uint32_t kind) {
    switch (kind) {
        case flow::kConst:   return "Const";
        case flow::kAdd:     return "Add";
        case flow::kSub:     return "Sub";
        case flow::kMul:     return "Mul";
        case flow::kMin:     return "Min";
        case flow::kMax:     return "Max";
        case flow::kSelect:  return "Select";
        case flow::kInput:   return "Input";
        case flow::kCounter: return "Counter";
        case flow::kDelay:   return "Delay";
        case flow::kLatch:   return "Latch";
        default:             return "Node";
    }
}

// ShowsConstArg: whether this kind's box should display its constArg (Const/Input/Counter carry a
// meaningful payload; the others ignore constArg so the box hides it). Deterministic, pure.
inline bool ShowsConstArg(uint32_t kind) {
    return kind == flow::kConst || kind == flow::kInput || kind == flow::kCounter;
}

// ---- BuildFlowGraphView: lay a flow::Graph out as a node graph (THE pure deterministic builder) ------
//
// COLUMN (topological rank) = longest-path distance from a root over the SAME EdgeMask edges TopoOrder
// uses (a root has no real input edges -> rank 0). Computed by relaxing ranks in the CANONICAL TopoOrder
// (a node is processed only after all its real-edge parents, so rank[child] = max(rank[parent])+1 settles
// in one pass). A cyclic graph (TopoOrder false) lays every node out at column 0 (a deterministic
// degenerate fallback, never UB) — the editor still renders the boxes, just unranked.
//
// ROW (order within a column) = the running count of nodes already assigned to that column, walked in the
// canonical TopoOrder (so rows are stable + gap-free per column). This makes the WHOLE layout a function
// of (graph, layout params) alone -> byte-identical run-to-run AND cross-platform.
//
// WIRES = for each node, each input field whose EdgeMask bit is set AND IsRealEdge (in-range, non-self)
// emits one FlowWireView from the parent's output anchor to this node's slot anchor, in (child NodeId
// ascending, slot ascending a->b->c) order — a fixed, hash-free emission order.
inline FlowGraphView BuildFlowGraphView(const flow::Graph& g, const FlowLayout& L = FlowLayout{}) {
    FlowGraphView view;
    const std::size_t n = g.nodes.size();
    if (n == 0) return view;

    // Canonical order (the determinism anchor). On a cycle, fall back to NodeId order so layout still works.
    std::vector<flow::NodeId> order;
    const bool topoOk = flow::TopoOrder(g, order);
    if (!topoOk) {
        order.clear();
        order.reserve(n);
        for (std::size_t i = 0; i < n; ++i) order.push_back(static_cast<flow::NodeId>(i));
    }

    // --- Pass 1: column (longest-path rank) + row (per-column running index), walked in canonical order. -
    std::vector<int> col(n, 0);
    std::vector<int> row(n, 0);
    std::vector<int> colFill;   // colFill[c] = how many nodes already placed in column c (the next row)
    for (const flow::NodeId id : order) {
        const std::size_t si = static_cast<std::size_t>(id);
        const flow::Node& nd = g.nodes[si];
        const uint32_t mask = flow::EdgeMask(nd.kind);
        // rank = max(parent rank)+1 over real edges (parents already processed in canonical order). Roots
        // (no real edge) stay rank 0. On the cyclic fallback every node stays column 0 (parents may not be
        // ordered-before, so no relaxation fires) — the deterministic degenerate layout.
        int r = 0;
        if (topoOk) {
            auto relax = [&](flow::NodeId in) {
                if (flow::IsRealEdge(g, id, in)) {
                    const int cand = col[static_cast<std::size_t>(in)] + 1;
                    if (cand > r) r = cand;
                }
            };
            if (mask & 0b001u) relax(nd.a);
            if (mask & 0b010u) relax(nd.b);
            if (mask & 0b100u) relax(nd.c);
        }
        col[si] = r;
        if (static_cast<std::size_t>(r) >= colFill.size()) colFill.resize(static_cast<std::size_t>(r) + 1, 0);
        row[si] = colFill[static_cast<std::size_t>(r)];
        ++colFill[static_cast<std::size_t>(r)];
    }

    // --- Pass 2: emit one FlowNodeView per node (in NodeId order so nodes[i].id == i). -------------------
    int maxCol = 0, maxRow = 0;
    view.nodes.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const flow::Node& nd = g.nodes[i];
        FlowNodeView nv;
        nv.id   = static_cast<flow::NodeId>(i);
        nv.col  = col[i];
        nv.row  = row[i];
        nv.x    = L.originX + col[i] * L.colStride;
        nv.y    = L.originY + row[i] * L.rowStride;
        nv.kind = nd.kind;
        nv.label = KindLabel(nd.kind);
        nv.constArg = ShowsConstArg(nd.kind) ? nd.constArg : flow::Reg{0};
        view.nodes[i] = std::move(nv);
        if (col[i] > maxCol) maxCol = col[i];
        if (row[i] > maxRow) maxRow = row[i];
    }
    view.gridCols = maxCol + 1;
    view.gridRows = maxRow + 1;

    // --- Pass 3: emit wires per EdgeMask in (child asc, slot asc) order. ---------------------------------
    // Slot anchor on the child's LEFT edge, spread vertically by slot (slot 0 high, 1 mid, 2 low) so a
    // multi-input node's wires fan out readably. Parent anchor on its RIGHT edge, vertically centered.
    auto outAnchor = [&](flow::NodeId p) -> std::pair<int,int> {
        const FlowNodeView& pv = view.nodes[static_cast<std::size_t>(p)];
        return { pv.x + L.boxW, pv.y + L.boxH / 2 };   // right-center of the parent box
    };
    auto inAnchor = [&](flow::NodeId child, uint32_t slot) -> std::pair<int,int> {
        const FlowNodeView& cv = view.nodes[static_cast<std::size_t>(child)];
        // 3 evenly spread slots down the left edge: y = top + (slot+1)/4 * boxH.
        const int sy = cv.y + static_cast<int>((slot + 1u)) * L.boxH / 4;
        return { cv.x, sy };
    };
    for (std::size_t i = 0; i < n; ++i) {
        const flow::NodeId child = static_cast<flow::NodeId>(i);
        const flow::Node& nd = g.nodes[i];
        const uint32_t mask = flow::EdgeMask(nd.kind);
        const flow::NodeId ins[3] = { nd.a, nd.b, nd.c };
        for (uint32_t slot = 0; slot < 3; ++slot) {     // slot order a(0) -> b(1) -> c(2)
            if (!(mask & (1u << slot))) continue;        // not an edge slot for this kind
            const flow::NodeId parent = ins[slot];
            if (!flow::IsRealEdge(g, child, parent)) continue;  // out-of-range / self -> no wire
            FlowWireView w;
            w.from = parent;
            w.to   = child;
            w.slot = slot;
            auto [ox, oy] = outAnchor(parent);
            auto [ix, iy] = inAnchor(child, slot);
            w.fromX = ox; w.fromY = oy;
            w.toX   = ix; w.toY   = iy;
            view.wires.push_back(w);
        }
    }

    return view;
}

// ---- DigestFlowGraphView: FNV-1a-64 over a HAND little-endian serialization of the view (the golden) --
// HAND-LE field by field (NEVER memcpy the structs — padding/endianness-unsafe; the flow.h DigestEvents /
// replay.h discipline) so the digest is byte-stable cross-platform. Encodes the node grid (id,col,row,
// x,y,kind,constArg) then every wire (from,to,slot,fromX,fromY,toX,toY). int values are encoded as their
// two's-complement uint32 LE bits. The label string is NOT hashed (it is derived purely from kind).
inline uint64_t DigestFlowGraphView(const FlowGraphView& v) {
    std::vector<unsigned char> buf;
    buf.reserve(v.nodes.size() * 28u + v.wires.size() * 28u + 16u);
    auto putU32 = [&](uint32_t x) {
        buf.push_back(static_cast<unsigned char>( x        & 0xFFu));
        buf.push_back(static_cast<unsigned char>((x >> 8)  & 0xFFu));
        buf.push_back(static_cast<unsigned char>((x >> 16) & 0xFFu));
        buf.push_back(static_cast<unsigned char>((x >> 24) & 0xFFu));
    };
    auto putI32 = [&](int x) { putU32(static_cast<uint32_t>(x)); };

    putU32(static_cast<uint32_t>(v.nodes.size()));
    putI32(v.gridCols);
    putI32(v.gridRows);
    for (const FlowNodeView& nv : v.nodes) {
        putU32(nv.id);
        putI32(nv.col);
        putI32(nv.row);
        putI32(nv.x);
        putI32(nv.y);
        putU32(nv.kind);
        putI32(static_cast<int>(nv.constArg));
    }
    putU32(static_cast<uint32_t>(v.wires.size()));
    for (const FlowWireView& w : v.wires) {
        putU32(w.from);
        putU32(w.to);
        putU32(w.slot);
        putI32(w.fromX);
        putI32(w.fromY);
        putI32(w.toX);
        putI32(w.toY);
    }
    return hf::net::DigestBytes(buf.data(), buf.size());
}

// ====================================================================================================
// Issue #24 — LIVE EXECUTION FEEDBACK (the Blueprint loop's third beat: author -> EXECUTE -> VISUALIZE).
// APPEND-ONLY below everything above. Does NOT touch BuildFlowGraphView / FlowNodeView / KindLabel /
// ShowsConstArg / DigestFlowGraphView — the static-view golden 0xaaf9beb70640a9b7 is UNCHANGED.
//
// The static view (above) surfaces each node's AUTHORED constArg. THIS surfaces each node's LIVE
// EVALUATED value — the flow VM's computed output for that node (flow::Evaluate, the canonical-topo
// register file). The editor can now show what UE5 Blueprint shows: the value that actually flows out of
// each node when the graph runs. The annotated state is itself a deterministic VALUE -> a NEW pinned
// digest below proves the value-annotated editor frame is byte-stable run-to-run AND cross-platform.
// Pure CPU, ImGui-free, backend-free (same discipline as the static view).
// ====================================================================================================

// FlowLiveValues: the NodeId-indexed LIVE register file for graph `g` — element i is node i's COMPUTED
// output (flow::Evaluate's canonical-topo evaluation; out-of-range/self inputs read 0; on a cycle an
// all-zero file). This is exactly the VM output the editor annotates each node box with. Deterministic of
// `g` alone (Evaluate topo-sorts first), so two calls are byte-identical.
inline std::vector<flow::Reg> FlowLiveValues(const flow::Graph& g) {
    return flow::Evaluate(g);
}

// DigestFlowLiveView: a NEW pinned FNV-1a-64 over the VALUE-ANNOTATED editor state = the existing
// static-view bytes (DigestFlowGraphView's serialization) THEN each node's LIVE value folded in (int32
// two's-complement LE, in NodeId order). Reuses DigestFlowGraphView for the static half (so the static
// golden is literally a prefix of this), then mixes the live register file in NodeId order. Proving THIS
// digest is stable proves the whole author->execute->visualize frame is deterministic + cross-platform.
// `values` is FlowLiveValues(g) (NodeId-indexed); a short/long `values` is tolerated (missing slots fold
// 0, extra slots ignored) so a value file sized to the graph always digests well-defined bytes.
inline uint64_t DigestFlowLiveView(const FlowGraphView& view, const std::vector<flow::Reg>& values) {
    // Fold the static-view digest (a uint64) in LE first, then each node's live value (int32 LE, NodeId
    // order). Hand little-endian field by field — NEVER memcpy a host struct (the DigestFlowGraphView /
    // replay.h discipline) -> byte-stable cross-platform.
    std::vector<unsigned char> buf;
    buf.reserve(8u + view.nodes.size() * 4u);
    const uint64_t staticDigest = DigestFlowGraphView(view);
    auto putU32 = [&](uint32_t x) {
        buf.push_back(static_cast<unsigned char>( x        & 0xFFu));
        buf.push_back(static_cast<unsigned char>((x >> 8)  & 0xFFu));
        buf.push_back(static_cast<unsigned char>((x >> 16) & 0xFFu));
        buf.push_back(static_cast<unsigned char>((x >> 24) & 0xFFu));
    };
    putU32(static_cast<uint32_t>( staticDigest        & 0xFFFFFFFFull));   // static digest low 32
    putU32(static_cast<uint32_t>((staticDigest >> 32) & 0xFFFFFFFFull));   // static digest high 32
    // Live value per node, in NodeId order (nodes[i].id == i). Missing slots fold 0 (deterministic).
    for (const FlowNodeView& nv : view.nodes) {
        const std::size_t idx = static_cast<std::size_t>(nv.id);
        const flow::Reg v = (idx < values.size()) ? values[idx] : flow::Reg{0};
        putU32(static_cast<uint32_t>(v));   // int32 bits as uint32 (two's-complement LE)
    }
    return hf::net::DigestBytes(buf.data(), buf.size());
}

}  // namespace hf::editor
