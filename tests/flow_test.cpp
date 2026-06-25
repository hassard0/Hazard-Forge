// Unit test for the deterministic node-graph EXECUTION VM (engine/flow/flow.h, Slice FLOW-S1, flagship
// #24 visual-scripting beachhead). Pure CPU (hf_core), ASan-eligible like the other pure tests.
//
// SELF-CONTAINED: the test scaffolding (check() + HF_TEST_MAIN_INIT()) is copied from econ_test.cpp /
// wfc_test.cpp (NOT included) so this compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/flow_test.cpp` on the Mac — the cheap cross-platform proof.
// Everything is INTEGER evaluation over a CANONICAL topological order, so the evaluated register file —
// and hence flow::DigestGraph (FNV-1a-64) over it — is bit-identical run-to-run AND platform-to-platform
// (MSVC vs Apple clang). The golden is a PINNED FNV-1a-64 DigestGraph value IN the test (NO image, NO bake).
//
// What this pins (the six FLOW-S1 assertions):
//   (a) DigestGraph(Evaluate(showcase)) == a hard-pinned uint64 (the cross-platform proof);
//   (b) re-evaluating the same graph is bit-identical (deterministic / replay-stable);
//   (c) flipping one node's constArg in a clone changes the digest (inputs are load-bearing);
//   (d) a cyclic graph is a deterministic rejection (TopoOrder false, Evaluate all-zero, no UB/hang);
//   (e) CANONICAL: a permuted-but-equivalent graph evaluates to the SAME digest (the central proof);
//   (f) a hand-checked kSelect node evaluates to the expected branch value.

#include "flow/flow.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::flow;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- Build the showcase graph + evaluate it. ----------------------------------------------------
    // regs is the NodeId-indexed register file (Evaluate topo-sorts first, then walks the canonical order).
    const Graph showcase = MakeShowcaseGraph();
    const std::vector<Reg> regs = Evaluate(showcase);
    const uint64_t digest = DigestGraph(regs);

    std::printf("flow-s1: showcase eval digest = 0x%016llx\n",
                static_cast<unsigned long long>(digest));

    // The pinned golden (computed on first run, hardcoded — the regression anchor / cross-platform bar).
    const uint64_t kPinnedDigest = 0x0e5b8ec26f0d8730ull;  // PINNED on first run (MSVC == clang)

    // ---- (a) PINNED DIGEST — the cross-platform make-or-break (identical on MSVC + clang). ----------
    check(digest == kPinnedDigest,
          "flow-s1: DigestGraph(Evaluate(showcase)) == pinned uint64 (the cross-platform proof)");

    // ---- (b) REPLAY-STABLE — re-evaluating the same graph reproduces the digest. --------------------
    {
        check(DigestGraph(Evaluate(showcase)) == digest,
              "flow-s1: re-evaluating the same graph is bit-identical (deterministic)");
    }

    // ---- (c) LOAD-BEARING — clone the showcase, flip one node's constArg, re-evaluate -> differs. ----
    {
        Graph mutated = MakeShowcaseGraph();
        // node 1 is `kConst 7` feeding the Mul (n6) -> the Select (n8) -> the Max (n9): a load-bearing const.
        mutated.nodes[1].constArg += 1;  // 7 -> 8
        check(DigestGraph(Evaluate(mutated)) != digest,
              "flow-s1: changing one node's constArg changes the digest (inputs are load-bearing)");
    }

    // ---- (d) CYCLE REJECTION — TopoOrder false; Evaluate all-zero, size == node count; no hang/UB. ---
    {
        const Graph cyclic = MakeCyclicGraph();
        std::vector<NodeId> order;
        const bool topoOk = TopoOrder(cyclic, order);
        const std::vector<Reg> cregs = Evaluate(cyclic);
        bool allZero = (cregs.size() == cyclic.nodes.size());
        for (const Reg r : cregs) if (r != 0) allZero = false;
        check(!topoOk && allZero,
              "flow-s1: a cyclic graph is a deterministic rejection (TopoOrder false, Evaluate all-zero, no UB/hang)");
    }

    // ---- (e) CANONICAL ORDER — a permuted-but-equivalent graph evaluates to the SAME logical result. -
    // The permuted graph RELABELS the nodes (so its canonical topo order is a different NodeId sequence),
    // but every node's LOGICAL value is preserved: mapping the permuted register file back through the
    // permutation (UnpermuteRegs) yields a BYTE-IDENTICAL register file -> the SAME digest. This proves the
    // scheduler computes the same DAG regardless of array layout (the central determinism proof).
    {
        const Graph perm = Permuted(showcase);
        const std::vector<Reg> permRegs   = Evaluate(perm);
        const std::vector<Reg> aligned    = UnpermuteRegs(permRegs);  // back to original NodeId labeling
        check(DigestGraph(aligned) == digest,
              "flow-s1: topo order is CANONICAL — a permuted-but-equivalent graph evaluates to the SAME digest");
    }

    // ---- (f) SELECT CORRECTNESS — the hand-checked kSelect node (node 8) routes to its a-branch. -----
    {
        // node 8 = Select(c=n2, a=n6, b=n7): regs[2]==3 (!=0) -> picks a=n6 = 7*3 = 21.
        check(regs[8] == 21,
              "flow-s1: kSelect routes on its predicate (a hand-checked node value is correct)");
    }

    if (g_fail == 0) { std::printf("flow_test: ALL PASS\n"); return 0; }
    std::printf("flow_test: %d FAIL\n", g_fail);
    return 1;
}
