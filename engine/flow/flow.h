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
    // ---- S2 stateful kinds (APPEND-ONLY; never renumber). See the S2 block below for semantics. -----
    kInput   = 7,   // output = inputs[constArg] (the external per-tick input; 0 if oob) — NO current-tick edge
    kCounter = 8,   // output = state.prev[self] + constArg (accumulate by constArg) — self via prev, NO edge
    kDelay   = 9,   // output = state.prev[a] (a's PREVIOUS-tick value, a 1-tick lag) — a is a STATE read, NO edge
    kLatch   = 10,  // output = (regs[c]!=0) ? regs[a] : state.prev[self] (hold until c re-fires) — a,c are edges
};

// ---- EdgeMask: which of {a,b,c} are CURRENT-TICK topo edges for a given kind (THE determinism crux) --
// bit0=a, bit1=b, bit2=c. A bit set => that input is a real same-tick data dependency that TopoOrder must
// order before this node. State reads (kDelay's a, kCounter/kLatch self) are NOT edges — they read
// state.prev (last tick's register file), so a feedback loop `acc=Add(input, Delay(acc))` is NOT a topo
// cycle. The S1 stateless kinds list exactly their used inputs as edges (kAdd..kMax -> a,b; kSelect ->
// a,b,c; kConst -> none), so gating S1's TopoOrder by EdgeMask is BYTE-IDENTICAL for an S1-only graph
// (the S1-invariant). Pure integer, no <algorithm>.
inline uint32_t EdgeMask(uint32_t kind) {
    switch (kind) {
        case kAdd: case kSub: case kMul: case kMin: case kMax: return 0b011u;  // a,b
        case kSelect:                                          return 0b111u;  // a,b,c
        case kLatch:                                           return 0b101u;  // a,c (b unused; self via prev)
        // kConst, kInput, kCounter, kDelay: no current-tick edges (kDelay's a is a PREV-state read).
        default:                                               return 0b000u;
    }
}

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

    // in-degree from REAL CURRENT-TICK edges only. An input field counts as an edge IFF its EdgeMask bit
    // is set (it is a same-tick data edge for this kind, NOT a state read) AND it IsRealEdge (in-range,
    // non-self). For the S1 stateless kinds EdgeMask lists exactly their used inputs, so this is
    // byte-identical to S1's "count every IsRealEdge input" (the verified S1-invariant). S2 state reads
    // (kDelay's a, kCounter/kLatch self) have their bit clear -> NOT counted -> feedback isn't a cycle.
    std::vector<uint32_t> indeg(n, 0u);
    for (std::size_t i = 0; i < n; ++i) {
        const Node& nd = g.nodes[i];
        const NodeId self = static_cast<NodeId>(i);
        const uint32_t mask = EdgeMask(nd.kind);
        if ((mask & 0b001u) && IsRealEdge(g, self, nd.a)) ++indeg[i];
        if ((mask & 0b010u) && IsRealEdge(g, self, nd.b)) ++indeg[i];
        if ((mask & 0b100u) && IsRealEdge(g, self, nd.c)) ++indeg[i];
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

        // Decrement the in-degree of every node that REALLY depends on `pick` via a current-tick edge
        // (same EdgeMask + IsRealEdge gate as the in-degree build above; scan ascending, fixed order).
        for (std::size_t j = 0; j < n; ++j) {
            if (emitted[j]) continue;
            const Node& dep = g.nodes[j];
            const NodeId self = static_cast<NodeId>(j);
            const uint32_t mask = EdgeMask(dep.kind);
            if ((mask & 0b001u) && IsRealEdge(g, self, dep.a) && dep.a == pick) --indeg[j];
            if ((mask & 0b010u) && IsRealEdge(g, self, dep.b) && dep.b == pick) --indeg[j];
            if ((mask & 0b100u) && IsRealEdge(g, self, dep.c) && dep.c == pick) --indeg[j];
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

// ====================================================================================================
// Slice FLOW-S2 — STATEFUL nodes + the per-tick step (issue #24, the StepFn shape). APPEND-ONLY below S1.
//
// S1 evaluated a stateless DAG ONCE. S2 makes the VM TICK: persistent per-node state across ticks
// (GraphState), external per-tick inputs (kInput), and stateful nodes (kCounter/kDelay/kLatch) — driven
// by StepGraph(graph, state, inputs, tick), whose signature is exactly what net::Session::Advance's
// step(world, inputs, tick) templates over (so S4 wraps it in the rollback engine with zero new netcode).
//
// THE CRUX: stateful kinds read state.prev (the PREVIOUS tick's register file) instead of current-tick
// edges. EdgeMask (above) clears those inputs' edge bits, so a FEEDBACK loop like
// `acc = Add(input, Delay(acc))` is NOT a topological cycle — yet evaluation stays bit-identical because
// state is just a snapshot-able register file. EDGEMASK / TOPOORDER REFINEMENT: option (a) — S1's
// TopoOrder already counted only IsRealEdge inputs; we additionally gate each input on its EdgeMask bit.
// Since the S1 stateless kinds' EdgeMask lists exactly their used inputs, this is byte-identical for an
// S1-only graph -> S1's pinned digest 0x0e5b8ec26f0d8730 is UNCHANGED (asserted by the test). We did NOT
// add a separate TopoOrderTick (option b); the single gated scheduler serves both S1 and S2.
// Pure-CPU INTEGER; header stays self-contained (only <cstddef>/<cstdint>/<vector> + net/session.h).
// ====================================================================================================

// GraphState: the persistent register file — one Reg per node = its value at the END of the last tick
// (the snapshot-able state the stateful kinds read via `prev`). Sized to the graph's node count.
struct GraphState { std::vector<Reg> prev; };

// MakeState: a fresh zero-initialized state for graph g (prev[NodeId] all 0 -> tick-0 reads see 0).
inline GraphState MakeState(const Graph& g) {
    GraphState s;
    s.prev.assign(g.nodes.size(), Reg{0});
    return s;
}

// StepGraph: ONE deterministic tick. TopoOrder (EdgeMask-gated, so feedback loops aren't cycles), then
// walk the order computing each node's Reg into `regs`:
//   - S1 kinds: exactly as Evaluate (kConst/kAdd/.../kSelect over regs[edge], "no edge" reads 0).
//   - kInput  : inputs[constArg] if constArg in [0, inputs.size()), else 0 (the external per-tick input).
//   - kCounter: state.prev[self] + constArg (accumulate by constArg each tick; self read via prev).
//   - kDelay  : state.prev[a] if a in-range (a's PREVIOUS-tick value, a 1-tick lag), else 0. a is a STATE
//               read (EdgeMask 0), so kDelay is a topo-root — evaluated early — yet a feedback loop
//               a -> Add -> Delay -> (feeds a) is NOT a topo cycle.
//   - kLatch  : (regs[c] != 0) ? regs[a] : state.prev[self] (capture a when predicate c fires, else hold).
// Then commit `state.prev = regs;` and return regs (== state.prev after the call). On a CYCLE (TopoOrder
// false) returns an all-zero register file of size nodes.size() and commits it (deterministic, no UB). The
// `tick` param is kept to match net::Session::Advance's step(world, inputs, tick); v1 does not read it.
inline std::vector<Reg> StepGraph(const Graph& g, GraphState& state,
                                  const std::vector<Reg>& inputs, uint32_t tick) {
    (void)tick;  // reserved for future tick-aware nodes (keeps the Advance-compatible signature)
    const std::size_t n = g.nodes.size();
    std::vector<Reg> regs(n, 0);

    // Defensive: a state whose size drifted from the graph would UB the prev reads; size it to the graph.
    if (state.prev.size() != n) state.prev.assign(n, Reg{0});

    std::vector<NodeId> order;
    if (!TopoOrder(g, order)) { state.prev = regs; return regs; }  // cycle -> all-zero (deterministic)

    // Read a CURRENT-TICK input register honoring the "no edge" convention (out-of-range/self -> 0).
    auto readIn = [&](NodeId self, NodeId in) -> Reg {
        return IsRealEdge(g, self, in) ? regs[static_cast<std::size_t>(in)] : Reg{0};
    };
    // Read a PREVIOUS-tick state register (out-of-range -> 0). Used by kDelay (a) and self-holds.
    auto readPrev = [&](NodeId in) -> Reg {
        return (static_cast<std::size_t>(in) < n) ? state.prev[static_cast<std::size_t>(in)] : Reg{0};
    };

    for (const NodeId id : order) {
        const std::size_t si = static_cast<std::size_t>(id);
        const Node& nd = g.nodes[si];
        const Reg av = readIn(id, nd.a);
        const Reg bv = readIn(id, nd.b);
        const Reg cv = readIn(id, nd.c);
        Reg out = 0;
        switch (nd.kind) {
            case kConst:  out = nd.constArg;                  break;
            case kAdd:    out = av + bv;                      break;
            case kSub:    out = av - bv;                      break;
            case kMul:    out = av * bv;                      break;
            case kMin:    out = (av < bv) ? av : bv;          break;
            case kMax:    out = (av > bv) ? av : bv;          break;
            case kSelect: out = (cv != 0) ? av : bv;          break;
            // ---- S2 stateful kinds -------------------------------------------------------------------
            case kInput:
                out = (nd.constArg >= 0 &&
                       static_cast<std::size_t>(nd.constArg) < inputs.size())
                          ? inputs[static_cast<std::size_t>(nd.constArg)] : Reg{0};
                break;
            case kCounter: out = readPrev(id) + nd.constArg;  break;  // prev[self] + constArg
            case kDelay:   out = readPrev(nd.a);              break;  // a's PREVIOUS-tick value (1-tick lag)
            case kLatch:   out = (cv != 0) ? av : readPrev(id); break; // capture a on c, else hold prev[self]
            default:       out = 0;                           break;
        }
        regs[si] = out;
    }

    state.prev = regs;   // commit this tick's register file as the new persistent state
    return regs;
}

// DigestState: FNV-1a-64 over the persistent register file (the per-tick checksum currency; == the
// DigestGraph of state.prev). Reuses net::DigestBytes — the engine-wide pinned-golden hash.
inline uint64_t DigestState(const GraphState& s) {
    return hf::net::DigestBytes(s.prev.data(), s.prev.size() * sizeof(Reg));
}

// RunGraphTrace: run `ticks` StepGraphs from a FRESH state over the per-tick `inputStream`, recording
// DigestState AFTER each tick -> a per-tick digest trace (the net::DigestTrace shape, proving tick-by-tick
// determinism). inputStream[t] is the inputs for tick t; a missing/short entry -> an empty (all-zero)
// input vector. Deterministic of (g, inputStream, ticks) alone, so two runs emit the IDENTICAL trace.
inline std::vector<uint64_t> RunGraphTrace(const Graph& g,
                                           const std::vector<std::vector<Reg>>& inputStream,
                                           uint32_t ticks) {
    GraphState state = MakeState(g);
    std::vector<uint64_t> trace;
    trace.reserve(static_cast<std::size_t>(ticks));
    static const std::vector<Reg> kEmpty{};
    for (uint32_t t = 0; t < ticks; ++t) {
        const std::vector<Reg>& in =
            (static_cast<std::size_t>(t) < inputStream.size()) ? inputStream[static_cast<std::size_t>(t)]
                                                               : kEmpty;
        StepGraph(g, state, in, t);
        trace.push_back(DigestState(state));   // record the digest AFTER the tick
    }
    return trace;
}

// ====================================================================================================
// S2 fixtures (FIXED forever — the golden pins MakeShowcaseStateGraph()'s per-tick trace).
// ====================================================================================================

// MakeShowcaseStateGraph: a FIXED 8-node graph exercising ALL FOUR stateful kinds. The logical graph
// (NodeId-indexed; unused input fields = the node's OWN id, the "no edge" sentinel):
//   n0 = kInput[0]                            -- the external per-tick input at index 0
//   n1 = kDelay(a=n2)                         -- last tick's accumulator value (the FEEDBACK lag; a is a
//                                                state read, NOT an edge -> n2->n1->n2 is NOT a cycle)
//   n2 = Add(n0, n1)  = input + prevAcc       -- the running accumulator (acc += input via feedback)
//   n3 = kCounter(constArg=3)                 -- prev[self]+3 each tick -> 3,6,9,... (== t*3 after tick t)
//   n4 = kConst 2                             -- a constant used to build the latch predicate
//   n5 = Sub(n3, n4)? no -- predicate: n5 = Min(n3, n4)?  We use a kSelect-free predicate:
//        n5 = kConst 0/1 pattern is not tick-derived; instead derive the predicate from the counter parity
//        indirectly. SIMPLEST deterministic tick-derived predicate that's nonzero on SOME ticks: use the
//        delayed input as the gate. n5 = n1 (the delayed accumulator) -> nonzero once the accumulator > 0.
//   n5 = kDelay(a=n2) is the same as n1; to keep n5 distinct & meaningful we make n5 a Min that is 0 on
//        tick 0 then nonzero: n5 = Min(n0, n3). On tick 0: Min(input0, 3). With input0 != 0 -> nonzero.
//   n6 = kLatch(a=n2, c=n5) -- capture the accumulator n2 whenever predicate n5 != 0, else HOLD prev[self]
//   n7 = Max(n2, n6)        -- a pure-S1 sink combining the live accumulator and the latched value
//
// This gives a NON-TRIVIAL per-tick trace: kInput varies, the feedback accumulator grows, the counter
// climbs by 3, the latch captures/holds, and a Max sink combines them. (See the test for the hand-checks.)
inline Graph MakeShowcaseStateGraph() {
    Graph g;
    g.nodes.resize(8);

    // n0: external input index 0.  a/b/c unused -> own id; constArg = the input index (0).
    g.nodes[0] = Node{ kInput, /*a=*/0, /*b=*/0, /*c=*/0, /*const=*/0 };
    // n1: Delay(a=n2) -> n2's PREVIOUS-tick value (the feedback lag). b/c unused -> own id.
    g.nodes[1] = Node{ kDelay, /*a=*/2, /*b=*/1, /*c=*/1, 0 };
    // n2: Add(n0, n1) = input + prevAcc -> the running accumulator (feedback via n1=Delay(n2)).
    g.nodes[2] = Node{ kAdd, /*a=*/0, /*b=*/1, /*c=*/2, 0 };
    // n3: Counter(+3) -> prev[self]+3 each tick = 3,6,9,12,...  a/b/c unused -> own id.
    g.nodes[3] = Node{ kCounter, /*a=*/3, /*b=*/3, /*c=*/3, /*const=*/3 };
    // n4: Const 2 (a building block for predicates / hand-checks).  a/b/c = own id.
    g.nodes[4] = Node{ kConst, /*a=*/4, /*b=*/4, /*c=*/4, /*const=*/2 };
    // n5: Min(n0, n3) -> the latch predicate (nonzero whenever both the input and the counter are nonzero,
    //     i.e. every tick the input is nonzero, since n3 >= 3 > 0 always).  c unused -> own id.
    g.nodes[5] = Node{ kMin, /*a=*/0, /*b=*/3, /*c=*/5, 0 };
    // n6: Latch(a=n2, c=n5) -> capture the accumulator n2 when predicate n5 != 0, else hold prev[self].
    g.nodes[6] = Node{ kLatch, /*a=*/2, /*b=*/6, /*c=*/5, 0 };
    // n7: Max(n2, n6) -> a pure-S1 sink over the live accumulator and the latched value.
    g.nodes[7] = Node{ kMax, /*a=*/2, /*b=*/6, /*c=*/7, 0 };

    return g;
}

// MakeShowcaseInputStream: a FIXED 8-tick stream of varied single-channel integer inputs (index 0 only).
// Chosen so the accumulator (n2) grows non-monotonically and the latch predicate (Min(input, counter))
// dips to 0 on tick 3 (input 0) -> the latch HOLDS that tick (the hand-checked hold proof).
inline std::vector<std::vector<Reg>> MakeShowcaseInputStream() {
    return {
        { 5 },   // tick 0
        { 2 },   // tick 1
        { 7 },   // tick 2
        { 0 },   // tick 3  (input 0 -> predicate Min(0, counter)=0 -> latch HOLDS)
        { 3 },   // tick 4
        { 9 },   // tick 5
        { 1 },   // tick 6
        { 4 },   // tick 7
    };
}

}  // namespace hf::flow
