// Deterministic node-graph EXECUTION VM (Slice FLOW-S1, flagship #24 visual-scripting beachhead).
//
// The deterministic core a Blueprint-class VISUAL EDITOR would drive (the editor is OUT OF SCOPE). UE5
// Blueprints are *the* canonical non-deterministic UE5 subsystem: event order depends on actor
// registration / tick groups, float math, TMap iteration, per-frame timing -> two machines running the
// same graph on the same inputs routinely diverge (UE5's own deterministic-rollback path EXCLUDES
// Blueprints). A node graph whose evaluation is BIT-IDENTICAL across MSVC/clang/Vulkan/Metal and
// lockstep/rollback/replay-able is a capability UE5 structurally lacks.
//
// S1 establishes the two pieces every later slice builds on: the graph data model + the CANONICAL
// topological scheduler (the central determinism risk — a DAG has many valid topo orders; we pin ONE
// via Kahn's algorithm with a LOWEST-NodeId-first tie-break, the wfc::Propagate "scan a flat ready[]
// vector ascending" idiom — NEVER insertion/hash order). Pure-CPU INTEGER.
//
// SELF-CONTAINED ON PURPOSE: this header includes ONLY <cstddef>/<cstdint>/<vector> plus
// "net/session.h" (for hf::net::DigestBytes) — NO fpx / RHI / GPU / shader / <cmath> / float / clock /
// RNG / <random> / <unordered_*> / <map> / <functional> / std::hash / <algorithm> — so it compiles
// STANDALONE with `clang++ -std=c++20 -I engine -I tests tests/flow_test.cpp` on the Mac (like
// econ.h / wfc.h) — the cheapest cross-platform proof (NO render-bake, NO image). Integer min/max are
// done via ternary (no <algorithm>); all iteration is in a FIXED order (no hash-ordered containers).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "net/session.h"  // hf::net::DigestBytes — the pinned-golden FNV-1a-64 currency

namespace hf::flow {

using NodeId = uint32_t;   // a node's id == its index into Graph::nodes (pinned, monotonic, never recycled)
using Reg    = int32_t;    // a node's output value — INTEGER ONLY (raw int or Q16.16; no float ever)

// FIXED enum numbering = the wire contract (serialized later; never renumber).
enum Kind : uint32_t {
    kConst  = 0,   // 0 inputs: output = constArg
    kAdd    = 1,   // 2 inputs: regs[a] + regs[b]
    kSub    = 2,   // 2 inputs: regs[a] - regs[b]
    kMul    = 3,   // 2 inputs: regs[a] * regs[b]  (int32 wrap is deterministic; v1 assumes bounded values)
    kMin    = 4,   // 2 inputs: integer min(regs[a], regs[b])  (ternary; NO <algorithm>)
    kMax    = 5,   // 2 inputs: integer max(regs[a], regs[b])  (ternary; NO <algorithm>)
    kSelect = 6,   // 3 inputs: regs[c] != 0 ? regs[a] : regs[b]  (c is the predicate)
};

struct Node {
    uint32_t kind = kConst;          // Kind
    NodeId   a = 0, b = 0, c = 0;    // input node ids; kSelect uses c as the predicate (c!=0 -> a else b)
    Reg      constArg = 0;           // kConst payload (ignored by other kinds)
};

struct Graph {
    std::vector<Node> nodes;         // nodes[i] has NodeId i
};

// ---- "Unused input" / "no edge" convention (pinned, used consistently in TopoOrder + Evaluate + the
// showcase + the test) --------------------------------------------------------------------------------
//   An input id `in` referenced by a node `self` is a REAL dependency edge IFF it is BOTH in range
//   (`in < nodes.size()`) AND not self (`in != self`). Otherwise — an out-of-range id OR a self-reference
//   — it is "no edge": it contributes NOTHING to the in-degree for topo-sort and reads 0 in Evaluate. So a
//   kConst (0 inputs) sets a=b=c to its OWN id (sentinel "no edge"); a 2-input kind sets c to its own id;
//   etc. This makes a self-reference / out-of-range id BY CONSTRUCTION never a dependency or a cycle.
inline bool IsRealEdge(const Graph& g, NodeId self, NodeId in) {
    return in != self && static_cast<std::size_t>(in) < g.nodes.size();
}

// ---- TopoOrder: Kahn's algorithm producing the ONE CANONICAL order ----------------------------------
// indeg[i] = count of node i's REAL input edges (in-range, non-self). Then repeatedly pick the LOWEST
// not-yet-emitted NodeId with indeg==0 via an ASCENDING SCAN over a flat bool emitted[] vector (the
// wfc::Propagate "pop the lowest pending id" idiom — NEVER insertion/hash order), emit it, and decrement
// the in-degree of every node that REALLY depends on it. Returns true + the full order on success;
// returns false on a CYCLE (some node never reaches indeg 0) — a DETERMINISTIC rejection (outOrder
// cleared, NEVER UB, never a hang: the outer loop is bounded by nodes.size()).
inline bool TopoOrder(const Graph& g, std::vector<NodeId>& outOrder) {
    outOrder.clear();
    const std::size_t n = g.nodes.size();
    if (n == 0) return true;  // empty graph trivially ordered

    // in-degree from REAL edges only (an out-of-range / self id is not an edge).
    std::vector<uint32_t> indeg(n, 0u);
    for (std::size_t i = 0; i < n; ++i) {
        const Node& nd = g.nodes[i];
        const NodeId self = static_cast<NodeId>(i);
        if (IsRealEdge(g, self, nd.a)) ++indeg[i];
        if (IsRealEdge(g, self, nd.b)) ++indeg[i];
        if (IsRealEdge(g, self, nd.c)) ++indeg[i];
    }

    std::vector<uint8_t> emitted(n, 0u);
    outOrder.reserve(n);

    // Emit exactly n nodes, picking the LOWEST ready (indeg==0, not emitted) NodeId each round.
    for (std::size_t round = 0; round < n; ++round) {
        NodeId pick = 0;
        bool found = false;
        for (std::size_t i = 0; i < n; ++i) {  // ASCENDING scan -> lowest-id-first tie-break (the canon)
            if (!emitted[i] && indeg[i] == 0u) { pick = static_cast<NodeId>(i); found = true; break; }
        }
        if (!found) { outOrder.clear(); return false; }  // CYCLE: no ready node remains -> deterministic reject

        emitted[pick] = 1u;
        outOrder.push_back(pick);

        // Decrement the in-degree of every node that REALLY depends on `pick` (scan ascending, fixed order).
        for (std::size_t j = 0; j < n; ++j) {
            if (emitted[j]) continue;
            const Node& dep = g.nodes[j];
            const NodeId self = static_cast<NodeId>(j);
            if (IsRealEdge(g, self, dep.a) && dep.a == pick) --indeg[j];
            if (IsRealEdge(g, self, dep.b) && dep.b == pick) --indeg[j];
            if (IsRealEdge(g, self, dep.c) && dep.c == pick) --indeg[j];
        }
    }
    return true;
}

// ---- Evaluate: TopoOrder, then walk the order computing each node's Reg from already-evaluated inputs --
// regs is one Reg per node, indexed by NodeId. A "no edge" input (out-of-range/self per the convention)
// reads 0 (the deterministic-no-op gate). Per kind: kConst->constArg; kAdd/kSub/kMul -> the int op;
// kMin/kMax -> integer min/max via ternary; kSelect -> regs[c]!=0 ? regs[a] : regs[b]. On a CYCLE
// (TopoOrder false) return an all-zero register file of size nodes.size() (deterministic, no UB). Pure int.
inline std::vector<Reg> Evaluate(const Graph& g) {
    const std::size_t n = g.nodes.size();
    std::vector<Reg> regs(n, 0);

    std::vector<NodeId> order;
    if (!TopoOrder(g, order)) return regs;  // cycle -> all-zero register file (deterministic)

    // Read an input register honoring the "no edge" convention: out-of-range/self -> 0.
    auto readIn = [&](NodeId self, NodeId in) -> Reg {
        return IsRealEdge(g, self, in) ? regs[static_cast<std::size_t>(in)] : Reg{0};
    };

    for (const NodeId id : order) {
        const Node& nd = g.nodes[static_cast<std::size_t>(id)];
        const Reg av = readIn(id, nd.a);
        const Reg bv = readIn(id, nd.b);
        const Reg cv = readIn(id, nd.c);
        Reg out = 0;
        switch (nd.kind) {
            case kConst:  out = nd.constArg;                  break;
            case kAdd:    out = av + bv;                      break;
            case kSub:    out = av - bv;                      break;
            case kMul:    out = av * bv;                      break;
            case kMin:    out = (av < bv) ? av : bv;          break;  // integer min (ternary, no <algorithm>)
            case kMax:    out = (av > bv) ? av : bv;          break;  // integer max (ternary, no <algorithm>)
            case kSelect: out = (cv != 0) ? av : bv;          break;  // c predicate routes a vs b
            default:      out = 0;                            break;  // unknown kind -> deterministic 0
        }
        regs[static_cast<std::size_t>(id)] = out;
    }
    return regs;
}

// ---- DigestGraph: FNV-1a-64 over the evaluated register file (the pinned-golden currency) ------------
// regs is the NodeId-indexed register file from Evaluate (a contiguous int32_t span -> byte-stable);
// reuses net::DigestBytes (the engine-wide FNV-1a-64). The golden pins the showcase's register file.
inline uint64_t DigestGraph(const std::vector<Reg>& regs) {
    return hf::net::DigestBytes(regs.data(), regs.size() * sizeof(Reg));
}

// ====================================================================================================
// Fixtures (deterministic integer literals — the MakeShowcase* precedent). FIXED forever (the golden
// pins MakeShowcaseGraph()'s evaluation).
// ====================================================================================================

// MakeShowcaseGraph: a FIXED 10-node arithmetic+select DAG. CRUCIAL: the array order is NOT a topo order
// (node 0 depends on nodes 4 & 5, node 9 depends on 0 & 8) — so Evaluate MUST topo-sort first; a naive
// array-order walk would read un-evaluated registers and produce the WRONG digest. Unused input fields are
// set to the node's OWN id (the "no edge" sentinel). The logical graph:
//   n1=Const 7, n2=Const 3, n3=Const 10, n4=Const 5, n5=Const 2          (the consts; deps of the rest)
//   n6 = Mul(n1, n2)        = 7 * 3            = 21
//   n7 = Sub(n3, n5)        = 10 - 2           = 8
//   n0 = Add(n4, n5)        = 5 + 2            = 7
//   n8 = Select(c=n2, a=n6, b=n7) = (3!=0) ? 21 : 8 = 21   (predicate routes between the two subtrees)
//   n9 = Max(n0, n8)        = max(7, 21)       = 21
// Hand-checked: regs[8] (the kSelect) == 21 (chooses the a=n6 branch because regs[2]==3 != 0).
inline Graph MakeShowcaseGraph() {
    Graph g;
    g.nodes.resize(10);

    // node 0: Add(n4, n5) = 5 + 2 = 7  (depends on LATER nodes -> array order != topo order)
    g.nodes[0] = Node{ kAdd, /*a=*/4, /*b=*/5, /*c=*/0, /*const=*/0 };
    // nodes 1..5: the consts. Unused a/b/c = own id (no edge).
    g.nodes[1] = Node{ kConst, 1, 1, 1, 7 };
    g.nodes[2] = Node{ kConst, 2, 2, 2, 3 };
    g.nodes[3] = Node{ kConst, 3, 3, 3, 10 };
    g.nodes[4] = Node{ kConst, 4, 4, 4, 5 };
    g.nodes[5] = Node{ kConst, 5, 5, 5, 2 };
    // node 6: Mul(n1, n2) = 7 * 3 = 21.  c unused -> own id.
    g.nodes[6] = Node{ kMul, /*a=*/1, /*b=*/2, /*c=*/6, 0 };
    // node 7: Sub(n3, n5) = 10 - 2 = 8.  c unused -> own id.
    g.nodes[7] = Node{ kSub, /*a=*/3, /*b=*/5, /*c=*/7, 0 };
    // node 8: Select(c=n2, a=n6, b=n7) = (3 != 0) ? 21 : 8 = 21.
    g.nodes[8] = Node{ kSelect, /*a=*/6, /*b=*/7, /*c=*/2, 0 };
    // node 9: Max(n0, n8) = max(7, 21) = 21.  c unused -> own id.
    g.nodes[9] = Node{ kMax, /*a=*/0, /*b=*/8, /*c=*/9, 0 };

    return g;
}

// MakeCyclicGraph: a tiny graph with a REAL cycle (node 0 -> node 1 -> node 0) for the rejection test.
// node 0 reads node 1 and node 1 reads node 0 -> neither ever reaches indeg 0 -> TopoOrder returns false.
inline Graph MakeCyclicGraph() {
    Graph g;
    g.nodes.resize(2);
    g.nodes[0] = Node{ kAdd, /*a=*/1, /*b=*/0, /*c=*/0, 0 };  // reads node 1 (real edge), b/c=self (no edge)
    g.nodes[1] = Node{ kAdd, /*a=*/0, /*b=*/1, /*c=*/1, 0 };  // reads node 0 (real edge), b/c=self (no edge)
    return g;
}

// PermIndex: the fixed permutation Permuted applies — old NodeId i moves to array slot (n-1)-i (reverse).
// Exposed so the canonical-order proof can map a permuted graph's NodeId-indexed register file BACK to the
// original logical labeling (regs_orig[i] must equal regs_perm[PermIndex(n,i)]). A pure index bijection.
inline NodeId PermIndex(std::size_t n, NodeId i) {
    return (static_cast<std::size_t>(i) < n)
               ? static_cast<NodeId>((n - 1) - static_cast<std::size_t>(i))
               : i;
}

// CANONICAL-ORDER PROOF (how assertion (e) uses this): the topo order is canonical GIVEN a fixed NodeId
// labeling — but a permutation RELABELS the nodes, so a permuted graph's canonical order is a different
// NodeId sequence. The layout-INVARIANT fact is the per-node LOGICAL result: node i of g and its image
// (slot PermIndex(n,i)) of Permuted(g) must compute the IDENTICAL value. The test maps the permuted
// register file back through PermIndex and digests it — equal to g's NodeId-indexed digest IFF the
// scheduler computed the same logical DAG regardless of array layout. (UnpermuteRegs does the mapping.)

// UnpermuteRegs: given the register file of Permuted(g) (indexed by the PERMUTED NodeId), return it
// re-indexed by the ORIGINAL NodeId (orig[i] = permRegs[PermIndex(n,i)]). After this, the original graph
// and the permuted graph have BYTE-IDENTICAL register files IFF they are the same logical computation.
inline std::vector<Reg> UnpermuteRegs(const std::vector<Reg>& permRegs) {
    const std::size_t n = permRegs.size();
    std::vector<Reg> orig(n, 0);
    for (std::size_t i = 0; i < n; ++i)
        orig[i] = permRegs[static_cast<std::size_t>(PermIndex(n, static_cast<NodeId>(i)))];
    return orig;
}

// Permuted: g with its node array REVERSED and all NodeId references remapped consistently — the SAME
// logical graph in a DIFFERENT array order (for the canonical-order proof). The remap is
// new_index(old) = (n-1) - old (== PermIndex); references are remapped the same way. References that were
// the node's own id stay self-references under the bijection, so the "no edge" sentinels carry over.
inline Graph Permuted(const Graph& g) {
    const std::size_t n = g.nodes.size();
    Graph out;
    out.nodes.resize(n);
    if (n == 0) return out;

    auto remap = [&](NodeId old) -> NodeId {
        // Remap only in-range ids through the reversal bijection; leave out-of-range ids as-is (still
        // out-of-range under any layout -> still "no edge").
        return (static_cast<std::size_t>(old) < n)
                   ? static_cast<NodeId>((n - 1) - static_cast<std::size_t>(old))
                   : old;
    };

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t newIdx = (n - 1) - i;       // node i moves to slot (n-1)-i
        Node nd = g.nodes[i];
        nd.a = remap(nd.a);
        nd.b = remap(nd.b);
        nd.c = remap(nd.c);
        out.nodes[newIdx] = nd;
    }
    return out;
}

}  // namespace hf::flow
