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

// ====================================================================================================
// Slice FLOW-S3 — CONTROL FLOW + EVENTS: the deterministic Blueprint EXECUTION wire (issue #24).
// APPEND-ONLY below S2 (S1/S2 above stay byte-identical: S1 0x0e5b8ec26f0d8730, S2 trace final
// 0x670cf80b235bdafd are UNCHANGED — S3 adds NEW types/functions only, touches nothing above).
//
// S1/S2 are the DATA wire (a DAG of integer registers, ticking). Blueprint has a SECOND wire — the white
// EXECUTION pin that decides WHICH nodes fire and in WHAT ORDER. S3 adds that exec layer as a SEPARATE
// graph (ExecGraph) of Sequence/Branch/Gate/Event nodes, traversed from an entry in a PINNED order,
// reading predicates from the S2 data register file. In UE5 exec/event order depends on tick groups +
// actor registration + float timing (two machines fire events in different orders); here two peers fed
// the same inputs fire the SAME events on the SAME tick in the SAME order — the literal Blueprint exec
// model made bit-identical. Pure-CPU INTEGER; header stays self-contained (only <cstddef>/<cstdint>/
// <vector> + net/session.h).
// ====================================================================================================

// ExecKind: FIXED enum numbering = the wire contract (never renumber). Exec nodes are SEPARATE from data
// nodes (data = values; exec = control flow).
enum ExecKind : uint32_t {
    eSeq    = 0,   // fire ALL of `next` in order (a Sequence node)
    eBranch = 1,   // read regs[pred]: !=0 -> fire next[0] (TRUE), else next[1] (FALSE) — exactly ONE
    eGate   = 2,   // fire next[0] ONLY if regs[pred] != 0 (open), else nothing (a Gate)
    eEvent  = 3,   // emit EventRecord{eventId, payload=regs[pred]} then fire next[0]
};

// ExecNode: one exec-graph node. `pred` is a DATA NodeId (the Branch/Gate predicate, or an Event's payload
// source); `eventId` is the fired tag for eEvent; `next` is the ORDERED exec-successor indices (into
// ExecGraph::nodes) — the pinned traversal order. A "no successor" slot is simply a short `next`.
struct ExecNode {
    uint32_t              kind    = eSeq;   // ExecKind
    NodeId                pred    = 0;      // a DATA node id (Branch/Gate predicate or Event payload source)
    uint32_t              eventId = 0;      // eEvent: the fired event's id (a fixed tag)
    std::vector<uint32_t> next;             // ordered exec-successor indices into ExecGraph::nodes
};

// ExecGraph: the exec wire — a flat node array + the entry index traversal starts from.
struct ExecGraph {
    std::vector<ExecNode> nodes;
    uint32_t              entry = 0;
};

// EventRecord: one fired event (the exec-trace currency). payload is the Reg value of pred at fire time.
struct EventRecord { uint32_t eventId = 0; Reg payload = 0; };

// ---- RunExec: traverse the exec graph from `entry` in a PINNED order, reading predicates from the data
// register file `regs` (the S2 StepGraph output), returning fired events in TRAVERSAL order.
//
// TRAVERSAL RULE (the determinism crux):
//   - A STACK (a std::vector used LIFO) seeded with `entry`. Each pop dispatches by kind and pushes its
//     chosen successors onto the stack in REVERSED next-list order, so they POP in forward next-list
//     order (a stack reverses, so we reverse-on-push to cancel it) -> exec successors are processed in
//     ascending `next`-index position, NEVER hash/insertion-of-a-set order. NO unordered_* anywhere.
//   - eSeq   : push ALL of next (reversed) -> all fire, in next order.
//   - eBranch: read regs[pred] (out-of-range -> 0); push next[0] if regs[pred]!=0 (TRUE) else next[1]
//              (FALSE) — exactly ONE successor (a missing pin -> no successor pushed).
//   - eGate  : push next[0] only if regs[pred]!=0 (open), else nothing.
//   - eEvent : emit EventRecord{eventId, payload=regs[pred] (oob->0)}, then push next[0] (if present).
//   - An out-of-range successor index is SKIPPED (deterministic no-op), never UB.
//
// VISITED / RE-ENTRY RULE: there is NO per-node visited guard — Blueprint allows a node to be re-entered
// (a Sequence fanning into a shared sub-tree fires it twice), so we keep that semantics. Re-entry / an
// accidental exec LOOP is bounded purely by the STEP CAP below.
//
// STEP CAP (the bound): at most kStepCap = nodes.size()*8 + 64 pops. A graph that would loop forever
// (a successor reaching back) simply stops at the cap — a DETERMINISTIC give-up, never a hang, never UB.
// (v1 showcase exec graphs are acyclic and finish far under the cap; the cap is the safety net + the
// bounded-traversal proof in the test.)
inline std::vector<EventRecord> RunExec(const ExecGraph& eg, const std::vector<Reg>& regs) {
    std::vector<EventRecord> events;
    const std::size_t n = eg.nodes.size();
    if (n == 0) return events;

    // Read a data predicate register with the "no edge / out-of-range -> 0" discipline.
    auto readReg = [&](NodeId id) -> Reg {
        return (static_cast<std::size_t>(id) < regs.size()) ? regs[static_cast<std::size_t>(id)] : Reg{0};
    };
    // Push a successor index onto the stack iff it is in range (out-of-range -> deterministic skip).
    auto pushIf = [&](std::vector<uint32_t>& stk, uint32_t idx) {
        if (static_cast<std::size_t>(idx) < n) stk.push_back(idx);
    };

    const std::size_t kStepCap = n * 8u + 64u;   // the bound: deterministic give-up, never a hang

    std::vector<uint32_t> stack;
    pushIf(stack, eg.entry);

    std::size_t steps = 0;
    while (!stack.empty() && steps < kStepCap) {
        ++steps;
        const uint32_t cur = stack.back();
        stack.pop_back();
        const ExecNode& en = eg.nodes[static_cast<std::size_t>(cur)];

        switch (en.kind) {
            case eSeq: {
                // Fire ALL successors in next-list order: push REVERSED so they pop forward.
                for (std::size_t k = en.next.size(); k-- > 0; ) pushIf(stack, en.next[k]);
                break;
            }
            case eBranch: {
                const bool taken = (readReg(en.pred) != 0);
                if (taken) { if (en.next.size() > 0) pushIf(stack, en.next[0]); }   // TRUE pin
                else       { if (en.next.size() > 1) pushIf(stack, en.next[1]); }   // FALSE pin
                break;
            }
            case eGate: {
                if (readReg(en.pred) != 0 && en.next.size() > 0) pushIf(stack, en.next[0]);  // open
                break;
            }
            case eEvent: {
                events.push_back(EventRecord{ en.eventId, readReg(en.pred) });
                if (en.next.size() > 0) pushIf(stack, en.next[0]);
                break;
            }
            default: break;   // unknown exec kind -> deterministic no-op
        }
    }
    return events;
}

// ---- StepFlow: the FULL per-tick step = S2 data + S3 control -> events. This is the StepFn shape S4
// wraps in net::Session (StepGraph already matches Advance's step(world,inputs,tick); RunExec layers on
// the exec trace). Runs StepGraph (updates `state`, returns regs) then RunExec over those regs.
inline std::vector<EventRecord> StepFlow(const Graph& dataG, const ExecGraph& execG,
                                         GraphState& state, const std::vector<Reg>& inputs, uint32_t tick) {
    const std::vector<Reg> regs = StepGraph(dataG, state, inputs, tick);
    return RunExec(execG, regs);
}

// ---- DigestEvents: FNV-1a-64 over the fired-event trace. HAND-SERIALIZED little-endian (eventId as 4 LE
// bytes, then payload's int32 bits as 4 LE bytes, per record) into a byte buffer, then net::DigestBytes —
// NOT a struct memcpy (EventRecord may carry padding; hand-LE is padding-safe + endianness-stable, the
// replay.h discipline). An empty trace digests the empty buffer (a fixed value) -> ticks with no events
// still have a well-defined deterministic digest.
inline uint64_t DigestEvents(const std::vector<EventRecord>& ev) {
    std::vector<unsigned char> buf;
    buf.reserve(ev.size() * 8u);
    auto putU32 = [&](uint32_t v) {
        buf.push_back(static_cast<unsigned char>( v        & 0xFFu));
        buf.push_back(static_cast<unsigned char>((v >> 8)  & 0xFFu));
        buf.push_back(static_cast<unsigned char>((v >> 16) & 0xFFu));
        buf.push_back(static_cast<unsigned char>((v >> 24) & 0xFFu));
    };
    for (const EventRecord& e : ev) {
        putU32(e.eventId);
        putU32(static_cast<uint32_t>(e.payload));   // int32 bits as a uint32 (two's-complement LE)
    }
    return hf::net::DigestBytes(buf.data(), buf.size());
}

// ---- RunFlowTrace: run `ticks` StepFlows from a FRESH state over the per-tick `inputStream`, recording
// DigestEvents(events) AFTER each tick -> the per-tick EVENT-trace digest stream (the S3 golden currency,
// the net::DigestTrace shape for the exec wire). Deterministic of (dataG, execG, inputStream, ticks)
// alone -> two runs emit the IDENTICAL trace. A missing/short input entry -> an empty (all-zero) input.
inline std::vector<uint64_t> RunFlowTrace(const Graph& dataG, const ExecGraph& execG,
                                          const std::vector<std::vector<Reg>>& inputStream,
                                          uint32_t ticks) {
    GraphState state = MakeState(dataG);
    std::vector<uint64_t> trace;
    trace.reserve(static_cast<std::size_t>(ticks));
    static const std::vector<Reg> kEmpty{};
    for (uint32_t t = 0; t < ticks; ++t) {
        const std::vector<Reg>& in =
            (static_cast<std::size_t>(t) < inputStream.size()) ? inputStream[static_cast<std::size_t>(t)]
                                                               : kEmpty;
        const std::vector<EventRecord> ev = StepFlow(dataG, execG, state, in, t);
        trace.push_back(DigestEvents(ev));   // record the event-trace digest AFTER the tick
    }
    return trace;
}

// ====================================================================================================
// S3 fixtures (FIXED forever — the golden pins the per-tick event trace of MakeShowcaseExecGraph() over
// MakeShowcaseControlData() + MakeControlInputStream()).
// ====================================================================================================

// MakeShowcaseControlData: a FIXED 6-node DATA graph providing the exec PREDICATES. The headline is a
// self-deriving TICK-PARITY toggle built from a kDelay feedback (NO mod node needed, NO external parity
// channel): parity = Sub(one, Delay(parity)). At tick 0 prev[parity]=0 -> parity=1-0=1; tick 1
// prev=1 -> 0; tick 2 -> 1; ... so parity = 1,0,1,0,... (NONZERO on EVEN ticks 0,2,4,6; ZERO on ODD
// ticks 1,3,5,7). Delay's `a` is a STATE read (EdgeMask 0), so parity->Delay->parity is NOT a topo cycle.
//   n0 = kInput[0]              -- the external per-tick input (also the Event payload source + a gate cond)
//   n1 = kConst 1              -- the "one" used to flip the toggle
//   n2 = kDelay(a=n4)          -- the PREVIOUS-tick parity (the feedback lag)
//   n3 = kConst 0              -- (a spare const; keeps the layout fixed / a known-zero predicate source)
//   n4 = Sub(n1, n2) = 1 - prevParity  -- THE PARITY TOGGLE (1 on even ticks, 0 on odd ticks)
//   n5 = n0 via Max(n0,n3)=Max(input,0) -- the GATE condition = the input (>0 when input>0, 0 when input 0)
// Predicates used by the exec graph: parity = n4 (Branch), gateCond = n5 (Gate), payloadSrc = n0 (Event).
inline Graph MakeShowcaseControlData() {
    Graph g;
    g.nodes.resize(6);
    g.nodes[0] = Node{ kInput, /*a=*/0, /*b=*/0, /*c=*/0, /*const=*/0 };  // input index 0
    g.nodes[1] = Node{ kConst, /*a=*/1, /*b=*/1, /*c=*/1, /*const=*/1 };  // one
    g.nodes[2] = Node{ kDelay, /*a=*/4, /*b=*/2, /*c=*/2, /*const=*/0 };  // prev parity (Delay of n4)
    g.nodes[3] = Node{ kConst, /*a=*/3, /*b=*/3, /*c=*/3, /*const=*/0 };  // zero
    g.nodes[4] = Node{ kSub,   /*a=*/1, /*b=*/2, /*c=*/4, /*const=*/0 };  // 1 - prevParity = THE TOGGLE
    g.nodes[5] = Node{ kMax,   /*a=*/0, /*b=*/3, /*c=*/5, /*const=*/0 };  // Max(input,0) = gate condition
    return g;
}

// Event id tags (FIXED forever — part of the wire contract / the pinned trace).
enum FlowEventId : uint32_t {
    kEvTrue  = 100,   // the eBranch TRUE pin's event (fires on EVEN ticks, parity n4 != 0)
    kEvFalse = 101,   // the eBranch FALSE pin's event (fires on ODD ticks, parity n4 == 0)
    kEvGate  = 200,   // the eGate's event (fires ONLY when the gate cond n5 != 0, i.e. input > 0)
    kEvSeqA  = 300,   // the trailing eSeq's FIRST event (fixed-order proof: A before B)
    kEvSeqB  = 301,   // the trailing eSeq's SECOND event
};

// MakeShowcaseExecGraph: a FIXED exec graph over MakeShowcaseControlData() exercising EVERY ExecKind and
// producing a non-trivial per-tick event trace. The shape (indices into nodes[]):
//
//   [0] entry eSeq  -> next {1, 4}      : fire the Branch sub-tree (1), THEN the Gate (4), in THIS order
//   [1] eBranch(pred=n4 parity) -> next {2, 3}  : TRUE pin [2] on even ticks, FALSE pin [3] on odd ticks
//   [2] eEvent(kEvTrue,  payload=n0)    : the TRUE event   (no successor)
//   [3] eEvent(kEvFalse, payload=n0)    : the FALSE event  (no successor)
//   [4] eGate(pred=n5 gateCond) -> next {5}     : OPEN only when input>0 (n5 != 0)
//   [5] eSeq -> next {6, 7}             : fire TWO events in FIXED order A-then-B (the order proof)
//   [6] eEvent(kEvGate, payload=n0)     : behind the gate -> the gate's event (== kEvSeqA position? no:
//                                          [6] is kEvGate, the first of the eSeq pair)
//   [7] eEvent(kEvSeqB, payload=n0)     : the second of the eSeq pair
//
// Per-tick trace logic (input from MakeControlInputStream):
//   - EVEN tick (parity n4==1): Branch fires [2]=kEvTrue.  ODD tick (parity 0): Branch fires [3]=kEvFalse.
//   - Gate [4] opens iff input>0 -> then eSeq [5] fires [6]=kEvGate THEN [7]=kEvSeqB (A-before-B order).
//     If input==0 the gate is CLOSED -> NEITHER kEvGate nor kEvSeqB fires that tick (the gate proof).
// So a typical even+input>0 tick trace = [kEvTrue(payload=input), kEvGate(input), kEvSeqB(input)] in THAT
// exec order; an odd+input>0 tick = [kEvFalse, kEvGate, kEvSeqB]; an input==0 tick drops the last two.
inline ExecGraph MakeShowcaseExecGraph() {
    ExecGraph eg;
    eg.nodes.resize(8);
    eg.entry = 0;

    eg.nodes[0] = ExecNode{ eSeq,    /*pred=*/0, /*eventId=*/0,        /*next=*/{1u, 4u} };  // entry
    eg.nodes[1] = ExecNode{ eBranch, /*pred=*/4, /*eventId=*/0,        /*next=*/{2u, 3u} };  // on parity n4
    eg.nodes[2] = ExecNode{ eEvent,  /*pred=*/0, /*eventId=*/kEvTrue,  /*next=*/{} };        // TRUE event
    eg.nodes[3] = ExecNode{ eEvent,  /*pred=*/0, /*eventId=*/kEvFalse, /*next=*/{} };        // FALSE event
    eg.nodes[4] = ExecNode{ eGate,   /*pred=*/5, /*eventId=*/0,        /*next=*/{5u} };       // on gateCond n5
    eg.nodes[5] = ExecNode{ eSeq,    /*pred=*/0, /*eventId=*/0,        /*next=*/{6u, 7u} };  // fixed-order pair
    eg.nodes[6] = ExecNode{ eEvent,  /*pred=*/0, /*eventId=*/kEvGate,  /*next=*/{} };        // pair A (gated)
    eg.nodes[7] = ExecNode{ eEvent,  /*pred=*/0, /*eventId=*/kEvSeqB,  /*next=*/{} };        // pair B (gated)

    return eg;
}

// MakeControlInputStream: a FIXED 8-tick single-channel input stream (index 0). The input doubles as the
// Event PAYLOAD source (n0) AND the gate condition (n5=Max(input,0)). Tick 3 and tick 6 carry input 0 ->
// the gate CLOSES those ticks (the gate-blocks proof); the other ticks carry input>0 -> the gate opens.
inline std::vector<std::vector<Reg>> MakeControlInputStream() {
    return {
        { 5 },   // tick 0  even, input 5  -> Branch TRUE,  gate OPEN  (payload 5)
        { 2 },   // tick 1  odd,  input 2  -> Branch FALSE, gate OPEN
        { 7 },   // tick 2  even, input 7  -> Branch TRUE,  gate OPEN
        { 0 },   // tick 3  odd,  input 0  -> Branch FALSE, gate CLOSED (gate proof)
        { 3 },   // tick 4  even, input 3  -> Branch TRUE,  gate OPEN
        { 9 },   // tick 5  odd,  input 9  -> Branch FALSE, gate OPEN
        { 0 },   // tick 6  even, input 0  -> Branch TRUE,  gate CLOSED (gate proof)
        { 4 },   // tick 7  odd,  input 4  -> Branch FALSE, gate OPEN
    };
}

// ====================================================================================================
// Slice FLOW-S4 — LOCKSTEP / REPLAY COMPOSITION: the moat payoff (issue #24). APPEND-ONLY below S3
// (S1/S2/S3 above stay byte-identical: S1 0x0e5b8ec26f0d8730, S2 trace final 0x670cf80b235bdafd, S3
// event-trace final 0xd5735423148033cc are UNCHANGED — S4 adds ONE convenience function only, touches
// nothing above).
//
// THE HEADLINE: a flow::Graph is a pure function (state, inputs) -> state' — EXACTLY the StepFn shape
// net::Session<World,Input>::Advance / RunLockstep / DigestTrace / DesyncDetector template over. So a
// visual script runs ON the existing deterministic rollback-netcode engine with ZERO new netcode: World =
// flow::GraphState (a plain std::vector<int32_t> -> value-copy snapshot works), Input = flow::Reg (one per
// kInput channel), step = a lambda capturing the static Graph that calls StepGraph, digest = DigestState.
// Two peers fed only the input stream re-derive a BIT-IDENTICAL graph state at every tick (lockstep), the
// run is replay-stable, and a one-tick input divergence is LOCATED at the exact tick (the NS5 detector over
// a graph). UE5 Blueprints structurally cannot do this (non-deterministic event order). The substance of
// S4 lives in the TEST (it reuses the net:: templates verbatim); the header adds only ONE bridge.
// Pure-CPU INTEGER; header stays self-contained (only <cstddef>/<cstdint>/<vector> + net/session.h).
// ====================================================================================================

// BuildInputRing: bridge a flow per-tick input stream (stream[t] = the inputs feeding the kInput channels
// on tick t) into a net::InputRing<Reg>. For each tick t, AddInput(t, v) for each v in stream[t] IN ORDER,
// so the kInput channel index (a node's constArg) == the insertion index == InputRing::At(t)'s vector
// index. This makes the net::Session-driven eval (StepGraph over ring.At(t)) byte-identical to the direct
// S2 RunGraphTrace (StepGraph over stream[t]) — the COMPOSITION proof. Pure integer, no new include.
inline hf::net::InputRing<Reg> BuildInputRing(const std::vector<std::vector<Reg>>& stream) {
    hf::net::InputRing<Reg> ring;
    for (uint32_t t = 0; t < stream.size(); ++t)
        for (const Reg v : stream[static_cast<std::size_t>(t)]) ring.AddInput(t, v);
    return ring;
}

// ====================================================================================================
// Slice FLOW-S5 — ROLLBACK + SERIALIZATION: the netcode-grade capstone (issue #24). APPEND-ONLY below
// S4 (S1/S2/S3/S4 above stay byte-identical: S1 0x0e5b8ec26f0d8730, S2 trace final 0x670cf80b235bdafd,
// S3 event-trace final 0xd5735423148033cc, S4 lockstep final 0x670cf80b235bdafd are UNCHANGED — S5 adds
// NEW functions only, touches nothing above).
//
// S4 proved a flow::Graph composes with net::Session for lockstep/replay/desync. S5 completes the
// netcode-grade runtime in two halves:
//   (1) ROLLBACK — a flow::GraphState is a copy-restorable World, so net::RollbackSession<GraphState,Reg>
//       drives the GGPO-class predict->snapshot->rollback loop over a graph with ZERO new netcode. A
//       mispredicted REMOTE input rolls the graph state back to the bit-identical authority. The proof is
//       in the TEST (it reuses RollbackSession/StepPredicted/ConfirmRemote/ScriptedTransport/
//       RunWithTransport verbatim); the header adds the explicit Snapshot/Restore bridge below.
//   (2) SERIALIZATION — a Graph (the static visual script) round-trips to bytes byte-identically: a
//       visual script is a savable, shippable artifact (a save game / a multiplayer-sync delta). Hand
//       little-endian, field-by-field — NEVER memcpy a host struct (padding/endianness-unsafe; the
//       replay.h/wav.cpp discipline). UE5 Blueprints cannot roll back deterministically (their
//       non-deterministic event order breaks re-simulation) — exactly why UE5's rollback path excludes
//       them. Pure-CPU INTEGER; header stays self-contained (only <cstddef>/<cstdint>/<vector> +
//       net/session.h).
// ====================================================================================================

// SnapshotState / RestoreState: explicit, bit-exact state snapshot + restore. GraphState is ONE
// contiguous integer vector (prev), so EVERY stateful node's slot is in it by construction -> a value
// copy is a COMPLETE snapshot (the net::Session value-copy works implicitly; these make it explicit and
// let the test PROVE completeness the verdict.h way — an incomplete restore MUST diverge).
inline GraphState SnapshotState(const GraphState& s) { return s; }                  // deep copy (vector)
inline void       RestoreState(GraphState& s, const GraphState& snap) { s = snap; }  // bit-exact restore

// PutU32 / GetU32: hand little-endian 32-bit codec (the replay.h discipline — NEVER memcpy a host
// struct). S3's DigestEvents used a LOCAL lambda, not a namespace-level PutU32, so we define both here.
inline void PutU32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(static_cast<uint8_t>( v        & 0xFFu));
    b.push_back(static_cast<uint8_t>((v >> 8)  & 0xFFu));
    b.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
    b.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
}
inline uint32_t GetU32(const uint8_t* p) {
    return  static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}

// SerializeGraph: encode a Graph to bytes. Layout (all u32 LE): nodeCount, then per node
// kind, a, b, c, constArg (the int32 constArg's two's-complement bits as a uint32). Hand-LE field by
// field — the on-disk format is byte-stable cross-platform (MSVC == clang). A 5*4 = 20-byte fixed record
// per node plus a 4-byte header.
inline std::vector<uint8_t> SerializeGraph(const Graph& g) {
    std::vector<uint8_t> bytes;
    bytes.reserve(4u + g.nodes.size() * 20u);
    PutU32(bytes, static_cast<uint32_t>(g.nodes.size()));
    for (const Node& nd : g.nodes) {
        PutU32(bytes, nd.kind);
        PutU32(bytes, nd.a);
        PutU32(bytes, nd.b);
        PutU32(bytes, nd.c);
        PutU32(bytes, static_cast<uint32_t>(nd.constArg));  // int32 bits as uint32 (two's-complement LE)
    }
    return bytes;
}

// DeserializeGraph: decode the SerializeGraph byte layout back into `out` (the inverse). Defensive length
// checks at EVERY read -> returns false (and leaves out cleared) on truncation, never UB. On success
// out.nodes equals the original field-for-field and SerializeGraph(out) == the input bytes.
inline bool DeserializeGraph(const std::vector<uint8_t>& bytes, Graph& out) {
    out.nodes.clear();
    if (bytes.size() < 4u) return false;                    // need at least the node-count header
    std::size_t off = 0;
    const uint32_t count = GetU32(bytes.data() + off);
    off += 4u;
    // Truncation guard: every node is a fixed 20-byte record; the total must fit exactly-or-more.
    if (bytes.size() < off + static_cast<std::size_t>(count) * 20u) return false;
    out.nodes.resize(static_cast<std::size_t>(count));
    for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i) {
        Node nd;
        nd.kind     =                   GetU32(bytes.data() + off); off += 4u;
        nd.a        =                   GetU32(bytes.data() + off); off += 4u;
        nd.b        =                   GetU32(bytes.data() + off); off += 4u;
        nd.c        =                   GetU32(bytes.data() + off); off += 4u;
        nd.constArg = static_cast<Reg>( GetU32(bytes.data() + off)); off += 4u;  // uint32 bits -> int32
        out.nodes[i] = nd;
    }
    return true;
}

}  // namespace hf::flow
