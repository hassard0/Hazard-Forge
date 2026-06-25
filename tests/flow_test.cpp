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

    // ==================================================================================================
    // Slice FLOW-S2 — STATEFUL nodes + the per-tick StepGraph. Append-only (S1 above stays green).
    // The golden = a PINNED per-tick DigestState trace (8 ticks) over MakeShowcaseStateGraph() +
    // MakeShowcaseInputStream(), bit-identical Windows/MSVC vs Mac/clang via the standalone compile.
    // ==================================================================================================

    const Graph stateGraph = MakeShowcaseStateGraph();
    const std::vector<std::vector<Reg>> inputStream = MakeShowcaseInputStream();
    const std::vector<uint64_t> trace = RunGraphTrace(stateGraph, inputStream, 8);

    std::printf("flow-s2: per-tick trace (8 ticks):\n");
    for (std::size_t t = 0; t < trace.size(); ++t)
        std::printf("  tick %zu digest = 0x%016llx\n", t,
                    static_cast<unsigned long long>(trace[t]));
    std::printf("flow-s2: per-tick trace (8 ticks) final digest = 0x%016llx\n",
                static_cast<unsigned long long>(trace.empty() ? 0ull : trace.back()));

    // ---- (1) PINNED TRACE — the cross-platform make-or-break (identical on MSVC + clang). ------------
    // PINNED on first run (MSVC == clang). The full per-tick DigestState trace of the showcase state-graph.
    const uint64_t kPinnedTrace[8] = {
        0x326199e9ea68bd71ull,  // tick 0
        0xe4d4018ceff11885ull,  // tick 1
        0x1a597abc7d1e83f1ull,  // tick 2
        0xfe34dea0d3e2304dull,  // tick 3
        0x4fd9f7635c8ca621ull,  // tick 4
        0xfb4733c14fd84378ull,  // tick 5
        0xbc99994fb746cde5ull,  // tick 6
        0x670cf80b235bdafdull,  // tick 7
    };
    {
        bool match = (trace.size() == 8);
        for (std::size_t t = 0; match && t < 8; ++t)
            if (trace[t] != kPinnedTrace[t]) match = false;
        check(match,
              "flow-s2: RunGraphTrace digest trace == pinned uint64[] (every tick, deterministic)");
    }

    // ---- (2) REPLAY-STABLE — a second run reproduces the IDENTICAL trace. ---------------------------
    {
        const std::vector<uint64_t> trace2 = RunGraphTrace(stateGraph, inputStream, 8);
        check(trace2 == trace,
              "flow-s2: re-running the same graph+inputs is bit-identical");
    }

    // ---- (3) FEEDBACK NOT A CYCLE — TopoOrder of the feedback graph succeeds; the accumulator (n2)
    //          produces the expected running sum (acc += input each tick). n2 = Add(n0=Input, n1=Delay(n2)).
    {
        std::vector<NodeId> order;
        const bool topoOk = TopoOrder(stateGraph, order);  // must be TRUE despite n2<->n1 feedback
        // Hand-step the accumulator (n2) and check the running sum for the first few ticks.
        GraphState st = MakeState(stateGraph);
        std::vector<Reg> r0 = StepGraph(stateGraph, st, inputStream[0], 0);  // input 5 -> acc 5
        std::vector<Reg> r1 = StepGraph(stateGraph, st, inputStream[1], 1);  // input 2 -> acc 7
        std::vector<Reg> r2 = StepGraph(stateGraph, st, inputStream[2], 2);  // input 7 -> acc 14
        std::vector<Reg> r3 = StepGraph(stateGraph, st, inputStream[3], 3);  // input 0 -> acc 14
        const bool accOk = (r0[2] == 5) && (r1[2] == 7) && (r2[2] == 14) && (r3[2] == 14);
        check(topoOk && accOk,
              "flow-s2: the kDelay feedback loop is NOT a topo cycle (TopoOrder succeeds) and accumulates correctly");
    }

    // ---- (4) COUNTER — n3 = kCounter(+3): its value after tick t is (t+1)*3 (3,6,9,12,...). ----------
    {
        GraphState st = MakeState(stateGraph);
        bool counterOk = true;
        for (uint32_t t = 0; t < 8; ++t) {
            std::vector<Reg> r = StepGraph(stateGraph, st, inputStream[t], t);
            if (r[3] != static_cast<Reg>((t + 1) * 3)) counterOk = false;  // accumulates by constArg=3
        }
        check(counterOk,
              "flow-s2: kCounter increments by constArg each tick (hand-checked sequence)");
    }

    // ---- (5) DELAY LAG — n1 = kDelay(a=n2): n1 at tick t == n2's value at tick t-1 (0 at tick 0). ----
    {
        GraphState st = MakeState(stateGraph);
        Reg prevN2 = 0;          // n2 had no previous value before tick 0 -> the delay reads 0 at tick 0
        bool delayOk = true;
        for (uint32_t t = 0; t < 8; ++t) {
            std::vector<Reg> r = StepGraph(stateGraph, st, inputStream[t], t);
            if (r[1] != prevN2) delayOk = false;   // n1(this tick) must equal n2(previous tick)
            prevN2 = r[2];                          // remember n2 for the next tick's expected delay
        }
        check(delayOk,
              "flow-s2: kDelay outputs the previous tick's input value (1-tick lag, hand-checked)");
    }

    // ---- (6) LATCH HOLD — n6 = kLatch(a=n2, c=n5): holds across predicate-0 ticks, updates when nonzero.
    // Predicate n5 = Min(input, counter); on tick 3 the input is 0 -> n5==0 -> n6 HOLDS tick 2's capture.
    {
        GraphState st = MakeState(stateGraph);
        std::vector<Reg> r0 = StepGraph(stateGraph, st, inputStream[0], 0);  // pred Min(5,3)=3 !=0 -> capture n2=5
        std::vector<Reg> r1 = StepGraph(stateGraph, st, inputStream[1], 1);  // pred !=0 -> capture n2=7
        std::vector<Reg> r2 = StepGraph(stateGraph, st, inputStream[2], 2);  // pred !=0 -> capture n2=14
        std::vector<Reg> r3 = StepGraph(stateGraph, st, inputStream[3], 3);  // input 0 -> pred 0 -> HOLD 14
        std::vector<Reg> r4 = StepGraph(stateGraph, st, inputStream[4], 4);  // input 3 -> pred !=0 -> capture n2=17
        const bool latchOk =
            (r0[6] == 5) && (r1[6] == 7) && (r2[6] == 14) &&
            (r3[6] == 14) &&                       // HELD across the predicate-0 tick (== tick 2's value)
            (r3[5] == 0)  &&                        // confirm the predicate really was 0 that tick
            (r4[6] == 17);                          // updated when the predicate re-fired
        check(latchOk,
              "flow-s2: kLatch holds its value until the predicate re-fires (hand-checked)");
    }

    // ---- (7) S1 INVARIANT — the EdgeMask refinement is byte-identical for S1 graphs. -----------------
    {
        check(DigestGraph(Evaluate(MakeShowcaseGraph())) == kPinnedDigest,
              "flow-s2: S1's pinned digest 0x0e5b8ec26f0d8730 is UNCHANGED (the EdgeMask refinement is byte-identical for S1 graphs)");
    }

    if (g_fail == 0) { std::printf("flow_test: ALL PASS\n"); return 0; }
    std::printf("flow_test: %d FAIL\n", g_fail);
    return 1;
}
