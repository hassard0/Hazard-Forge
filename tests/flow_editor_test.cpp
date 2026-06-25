// Unit test for the flow-graph EDITOR DATA model + edit ops (engine/editor/flow_editor_data.h +
// engine/editor/flow_edit_ops.h, issue #24 — the EDITOR half of Blueprint-class visual scripting). Pure
// CPU (hf_core), ASan-eligible like the other pure tests.
//
// SELF-CONTAINED: the scaffolding (check() + HF_TEST_MAIN_INIT) mirrors flow_test.cpp so this compiles
// STANDALONE with `clang++ -std=c++20 -I engine -I tests tests/flow_editor_test.cpp` on the Mac — the
// cheap cross-platform proof. The whole layout is INTEGER (column = topo rank, row = within-rank order on a
// fixed grid) over the flow VM's CANONICAL TopoOrder, so the view — and hence DigestFlowGraphView
// (FNV-1a-64) over it — is bit-identical run-to-run AND platform-to-platform (MSVC == Apple clang). The
// golden is a PINNED FNV-1a-64 value IN the test (NO image, NO bake).
//
// What this pins:
//   (a) the view node count == the showcase graph's node count;
//   (b) the deterministic LAYOUT — a few node (col,row,x,y) positions are hand-checked;
//   (c) DigestFlowGraphView(view) == a hard-pinned uint64 (the cross-platform proof);
//   (d) re-building the view is bit-identical (deterministic / replay-stable);
//   (e) an edit op (AddFlowNode + ConnectFlow) changes the digest deterministically + repeatably;
//   (f) DeleteFlowNode re-maps references + keeps the graph TopoOrder-valid (still evaluable);
//   (g) wires match EdgeMask — every real input edge of the showcase produces exactly one wire.

#include "editor/flow_editor_data.h"
#include "editor/flow_edit_ops.h"
#include "flow/flow.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::editor;
using namespace hf::flow;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

// The showcase the editor renders: the flow VM's FIXED 10-node arithmetic+select DAG (MakeShowcaseGraph).
// Its array order is deliberately NOT a topo order (node 0 depends on 4&5; node 9 on 0&8), so the layout's
// topological-rank columns are a real test of the canonical-order layout (not array order).
int main() {
    HF_TEST_MAIN_INIT();

    const Graph showcase = MakeShowcaseGraph();
    const FlowLayout L;  // default fixed grid
    const FlowGraphView view = BuildFlowGraphView(showcase, L);

    std::printf("flow-editor: nodes=%zu wires=%zu gridCols=%d gridRows=%d\n",
                view.nodes.size(), view.wires.size(), view.gridCols, view.gridRows);

    // ---- (a) node count matches the graph. ----------------------------------------------------------
    check(view.nodes.size() == showcase.nodes.size(),
          "flow-editor: view node count == graph node count");

    // ---- (b) deterministic LAYOUT (hand-checked columns/rows for a few nodes). -----------------------
    // The logical graph (see MakeShowcaseGraph): the five consts n1..n5 and the const-less roots are rank 0;
    // n6=Mul(n1,n2), n7=Sub(n3,n5), n0=Add(n4,n5) are rank 1; n8=Select(n6,n7,n2) is rank 2; n9=Max(n0,n8)
    // is rank 3 (its deepest parent n8 is rank 2). Rows within a rank follow canonical TopoOrder appearance.
    auto& nv = view.nodes;
    // The consts are roots -> column 0.
    check(nv[1].col == 0 && nv[2].col == 0 && nv[3].col == 0 && nv[4].col == 0 && nv[5].col == 0,
          "flow-editor: the five const nodes are all column 0 (roots)");
    // n6/n7/n0 are one edge deep -> column 1.
    check(nv[6].col == 1 && nv[7].col == 1 && nv[0].col == 1,
          "flow-editor: Mul/Sub/Add (one edge deep) are column 1");
    // n8 (Select over n6,n7) -> column 2; n9 (Max over n0,n8) -> column 3 (deepest path).
    check(nv[8].col == 2, "flow-editor: Select is column 2");
    check(nv[9].col == 3, "flow-editor: Max is column 3 (longest path)");
    // Box X is a pure function of the column on the fixed grid.
    check(nv[9].x == L.originX + 3 * L.colStride && nv[9].y == L.originY + nv[9].row * L.rowStride,
          "flow-editor: node box (x,y) == origin + (col*colStride, row*rowStride)");
    // Print the layout we produced (for the report).
    for (std::size_t i = 0; i < nv.size(); ++i) {
        std::printf("  n%zu %-7s col=%d row=%d  box=(%d,%d)\n",
                    i, nv[i].label.c_str(), nv[i].col, nv[i].row, nv[i].x, nv[i].y);
    }

    // ---- (g) wires match EdgeMask — exactly one wire per real input edge of the showcase. ------------
    std::size_t expectedWires = 0;
    for (std::size_t i = 0; i < showcase.nodes.size(); ++i) {
        const Node& ndi = showcase.nodes[i];
        const uint32_t mask = EdgeMask(ndi.kind);
        const NodeId ins[3] = { ndi.a, ndi.b, ndi.c };
        for (uint32_t s = 0; s < 3; ++s)
            if ((mask & (1u << s)) && IsRealEdge(showcase, static_cast<NodeId>(i), ins[s]))
                ++expectedWires;
    }
    check(view.wires.size() == expectedWires,
          "flow-editor: wire count == number of real EdgeMask input edges");

    // ---- (c) PINNED DIGEST — the cross-platform make-or-break (identical on MSVC + clang). -----------
    const uint64_t digest = DigestFlowGraphView(view);
    std::printf("flow-editor: view digest = 0x%016llx\n",
                static_cast<unsigned long long>(digest));
    const uint64_t kPinnedDigest = 0xaaf9beb70640a9b7ull;  // PINNED on first run (MSVC == clang)
    check(digest == kPinnedDigest,
          "flow-editor: DigestFlowGraphView(view) == pinned uint64 (the cross-platform proof)");

    // ---- (d) REPLAY-STABLE — re-building the same view reproduces the digest. ------------------------
    check(DigestFlowGraphView(BuildFlowGraphView(showcase, L)) == digest,
          "flow-editor: re-building the view is bit-identical (deterministic)");

    // ---- (e) EDIT OP changes the digest deterministically + repeatably. ------------------------------
    // Add a kMul node, wire the existing Max (n9) and a const (n4) into it -> a new deepest sink. The view
    // digest must CHANGE, and the same edit sequence on a fresh clone must reproduce that new digest.
    auto applyEdit = [](Graph g) -> Graph {
        const NodeId nn = AddFlowNode(g, kMul, /*constArg=*/0);
        ConnectFlow(g, /*from=*/9, /*to=*/nn, /*slot=*/0);  // a = old Max
        ConnectFlow(g, /*from=*/4, /*to=*/nn, /*slot=*/1);  // b = const n4
        return g;
    };
    const Graph edited1 = applyEdit(MakeShowcaseGraph());
    const Graph edited2 = applyEdit(MakeShowcaseGraph());
    const uint64_t editedDigest = DigestFlowGraphView(BuildFlowGraphView(edited1, L));
    std::printf("flow-editor: edited view digest = 0x%016llx\n",
                static_cast<unsigned long long>(editedDigest));
    check(editedDigest != digest,
          "flow-editor: an AddFlowNode+ConnectFlow edit changes the view digest (edits are load-bearing)");
    check(DigestFlowGraphView(BuildFlowGraphView(edited2, L)) == editedDigest,
          "flow-editor: the same edit sequence reproduces the same view digest (deterministic edit)");
    // The new node should be the deepest column (a sink over the old Max).
    {
        const FlowGraphView ev = BuildFlowGraphView(edited1, L);
        check(ev.nodes.back().col == view.nodes[9].col + 1,
              "flow-editor: the added Mul sink is one column deeper than the old Max");
    }

    // ---- (f) DeleteFlowNode re-maps references + keeps the graph evaluable (TopoOrder valid). ---------
    {
        Graph g = MakeShowcaseGraph();
        // Delete const n4 (feeds n0=Add(n4,n5)). After the erase, every id>4 shifts down by one and the
        // edge n0->n4 becomes "no edge". The graph must still TopoOrder (no cycle / dangling index).
        const bool ok = DeleteFlowNode(g, /*victim=*/4);
        std::vector<NodeId> order;
        const bool topoOk = TopoOrder(g, order);
        check(ok && g.nodes.size() == showcase.nodes.size() - 1 && topoOk,
              "flow-editor: DeleteFlowNode removes the node, re-maps refs, keeps TopoOrder valid");
        // Evaluate still works (no UB) and produces a full register file.
        const std::vector<Reg> regs = Evaluate(g);
        check(regs.size() == g.nodes.size(),
              "flow-editor: the post-delete graph still evaluates (register file sized to nodes)");
        // The view of the shrunk graph is well-formed (no wire points past the array).
        const FlowGraphView dv = BuildFlowGraphView(g, L);
        bool wiresInRange = true;
        for (const FlowWireView& w : dv.wires)
            if (static_cast<std::size_t>(w.from) >= g.nodes.size() ||
                static_cast<std::size_t>(w.to)   >= g.nodes.size()) wiresInRange = false;
        check(wiresInRange, "flow-editor: post-delete view wires all reference in-range nodes");
    }

    // ---- (h) LIVE EXECUTION FEEDBACK (issue #24, the author->execute->visualize loop). ---------------
    // FlowLiveValues == flow::Evaluate (the canonical-topo VM output), the evaluated values are the
    // hand-checked ints, the NEW live digest is pinned, and re-eval is bit-identical (deterministic).
    {
        // (h.a) the STATIC golden STILL holds (the live path did NOT perturb BuildFlowGraphView/the digest).
        check(DigestFlowGraphView(view) == kPinnedDigest,
              "flow-live: static DigestFlowGraphView == 0xaaf9beb70640a9b7 UNCHANGED (append-only proof)");

        // (h.b) FlowLiveValues(showcase) == flow::Evaluate(showcase), and the evaluated values are the
        // expected ints. MakeShowcaseGraph hand-checks: n0=Add(5,2)=7, n6=Mul(7,3)=21, n7=Sub(10,2)=8,
        // n8=Select(c=3?,21,8)=21, n9=Max(7,21)=21; the consts n1..n5 = 7,3,10,5,2.
        const std::vector<Reg> live = FlowLiveValues(showcase);
        const std::vector<Reg> eval = Evaluate(showcase);
        check(live.size() == showcase.nodes.size() && live == eval,
              "flow-live: FlowLiveValues == flow::Evaluate (the VM output, NodeId-indexed)");
        std::printf("flow-live: evaluated values =");
        for (std::size_t i = 0; i < live.size(); ++i)
            std::printf(" n%zu=%d", i, static_cast<int>(live[i]));
        std::printf("\n");
        const bool valuesOk =
            live.size() == 10 &&
            live[0] == 7  && live[1] == 7  && live[2] == 3  && live[3] == 10 && live[4] == 5 &&
            live[5] == 2  && live[6] == 21 && live[7] == 8  && live[8] == 21 && live[9] == 21;
        check(valuesOk,
              "flow-live: the evaluated showcase values are the hand-checked ints "
              "(n6=21,n7=8,n8=21,n9=21,...)");

        // (h.c) the NEW pinned live digest + re-eval is identical (deterministic / replay-stable).
        const uint64_t liveDigest = DigestFlowLiveView(view, live);
        std::printf("flow-live: live digest = 0x%016llx\n",
                    static_cast<unsigned long long>(liveDigest));
        const uint64_t kPinnedLiveDigest = 0x0f43e3ba0b638c14ull;  // PINNED on first run (MSVC == clang)
        check(liveDigest == kPinnedLiveDigest,
              "flow-live: DigestFlowLiveView == pinned uint64 (the value-annotated cross-platform proof)");
        // Re-build the view + re-evaluate -> the SAME live digest (deterministic).
        check(DigestFlowLiveView(BuildFlowGraphView(showcase, L), FlowLiveValues(showcase)) == liveDigest,
              "flow-live: re-evaluating the live view is bit-identical (deterministic)");
        // The live digest folds the static one in -> it MUST differ from the bare static digest.
        check(liveDigest != kPinnedDigest,
              "flow-live: the live digest differs from the static digest (live values are load-bearing)");
    }

    if (g_fail == 0) std::printf("ALL PASS\n");
    else             std::printf("%d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
