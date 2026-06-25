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

    // ==================================================================================================
    // Slice FLOW-S3 — CONTROL FLOW + EVENTS: the deterministic Blueprint EXECUTION wire. Append-only
    // (S1+S2 above stay green). The golden = a PINNED per-tick EVENT-trace digest stream (8 ticks) over
    // MakeShowcaseControlData() + MakeShowcaseExecGraph() + MakeControlInputStream(), bit-identical
    // Windows/MSVC vs Mac/clang via the standalone compile.
    // ==================================================================================================

    const Graph     ctrlData = MakeShowcaseControlData();
    const ExecGraph execG    = MakeShowcaseExecGraph();
    const std::vector<std::vector<Reg>> ctrlInputs = MakeControlInputStream();
    const std::vector<uint64_t> evTrace = RunFlowTrace(ctrlData, execG, ctrlInputs, 8);

    std::printf("flow-s3: per-tick event trace (8 ticks):\n");
    for (std::size_t t = 0; t < evTrace.size(); ++t)
        std::printf("  tick %zu event-digest = 0x%016llx\n", t,
                    static_cast<unsigned long long>(evTrace[t]));
    std::printf("flow-s3: per-tick event trace (8 ticks) final digest = 0x%016llx\n",
                static_cast<unsigned long long>(evTrace.empty() ? 0ull : evTrace.back()));

    // ---- (1) PINNED EVENT TRACE — the cross-platform make-or-break (identical on MSVC + clang). -------
    // PINNED on first run (MSVC == clang). The full per-tick DigestEvents trace of the showcase exec graph.
    const uint64_t kPinnedEvTrace[8] = {
        0x7cad1c04312095a8ull,  // tick 0  even, input 5 -> [kEvTrue, kEvGate, kEvSeqB] payload 5
        0xf045888c785b1a6aull,  // tick 1  odd,  input 2 -> [kEvFalse, kEvGate, kEvSeqB] payload 2
        0xe2d0039142b98e8aull,  // tick 2  even, input 7 -> [kEvTrue, kEvGate, kEvSeqB] payload 7
        0x42f905fdea17fca6ull,  // tick 3  odd,  input 0 -> [kEvFalse] only (gate CLOSED)
        0x938040d670d9e2ceull,  // tick 4  even, input 3 -> [kEvTrue, kEvGate, kEvSeqB] payload 3
        0xa18386dc7b61c9f1ull,  // tick 5  odd,  input 9 -> [kEvFalse, kEvGate, kEvSeqB] payload 9
        0x61f3cd06f50746c7ull,  // tick 6  even, input 0 -> [kEvTrue] only (gate CLOSED)
        0xd5735423148033ccull,  // tick 7  odd,  input 4 -> [kEvFalse, kEvGate, kEvSeqB] payload 4
    };
    {
        bool match = (evTrace.size() == 8);
        for (std::size_t t = 0; match && t < 8; ++t)
            if (evTrace[t] != kPinnedEvTrace[t]) match = false;
        check(match,
              "flow-s3: RunFlowTrace event-trace digest stream == pinned uint64[] (deterministic per-tick events)");
    }

    // ---- (2) REPLAY-STABLE — a second run reproduces the IDENTICAL event trace. ----------------------
    {
        const std::vector<uint64_t> evTrace2 = RunFlowTrace(ctrlData, execG, ctrlInputs, 8);
        check(evTrace2 == evTrace,
              "flow-s3: re-running is bit-identical");
    }

    // ---- (3) BRANCH — only the taken pin's event fires. Hand-check two ticks: tick 0 (even, parity n4!=0)
    //          fires kEvTrue and NOT kEvFalse; tick 1 (odd, parity n4==0) fires kEvFalse and NOT kEvTrue.
    {
        GraphState st0 = MakeState(ctrlData);
        // Advance to tick 0 fresh.
        std::vector<EventRecord> e0 = StepFlow(ctrlData, execG, st0, ctrlInputs[0], 0);  // even
        std::vector<EventRecord> e1 = StepFlow(ctrlData, execG, st0, ctrlInputs[1], 1);  // odd
        bool t0True = false, t0False = false, t1True = false, t1False = false;
        for (const EventRecord& e : e0) { if (e.eventId == kEvTrue) t0True = true; if (e.eventId == kEvFalse) t0False = true; }
        for (const EventRecord& e : e1) { if (e.eventId == kEvTrue) t1True = true; if (e.eventId == kEvFalse) t1False = true; }
        check(t0True && !t0False && t1False && !t1True,
              "flow-s3: eBranch fires ONLY the taken pin's event (true on even ticks, false on odd — hand-checked)");
    }

    // ---- (4) SEQUENCE ORDER — the eSeq pair [6]=kEvGate then [7]=kEvSeqB fires in next-list order (A
    //          before B), not reversed/hash. Hand-check on tick 0 (gate OPEN): kEvGate precedes kEvSeqB.
    {
        GraphState st = MakeState(ctrlData);
        std::vector<EventRecord> e0 = StepFlow(ctrlData, execG, st, ctrlInputs[0], 0);  // input 5 -> gate open
        // Find the positions of kEvGate and kEvSeqB in the fired-event order.
        int posGate = -1, posSeqB = -1;
        for (std::size_t i = 0; i < e0.size(); ++i) {
            if (e0[i].eventId == kEvGate && posGate < 0) posGate = static_cast<int>(i);
            if (e0[i].eventId == kEvSeqB && posSeqB < 0) posSeqB = static_cast<int>(i);
        }
        check(posGate >= 0 && posSeqB >= 0 && posGate < posSeqB,
              "flow-s3: eSeq fires its successor events in the FIXED order (hand-checked event sequence)");
    }

    // ---- (5) GATE — the gated events (kEvGate, kEvSeqB) are ABSENT on a closed-gate tick (input 0) and
    //          PRESENT on an open-gate tick. Tick 0 (input 5) open; tick 3 (input 0) closed.
    {
        GraphState st = MakeState(ctrlData);
        std::vector<EventRecord> e0 = StepFlow(ctrlData, execG, st, ctrlInputs[0], 0);  // input 5 -> OPEN
        std::vector<EventRecord> e1 = StepFlow(ctrlData, execG, st, ctrlInputs[1], 1);  // input 2 -> OPEN
        std::vector<EventRecord> e2 = StepFlow(ctrlData, execG, st, ctrlInputs[2], 2);  // input 7 -> OPEN
        std::vector<EventRecord> e3 = StepFlow(ctrlData, execG, st, ctrlInputs[3], 3);  // input 0 -> CLOSED
        bool openHasGate = false, closedHasGate = false;
        for (const EventRecord& e : e0) if (e.eventId == kEvGate) openHasGate = true;
        for (const EventRecord& e : e3) if (e.eventId == kEvGate) closedHasGate = true;
        check(openHasGate && !closedHasGate,
              "flow-s3: eGate blocks the event when its predicate is 0, passes when nonzero (hand-checked)");
    }

    // ---- (6) EVENT PAYLOAD — an eEvent's payload == regs[pred] at that tick. The Events read n0 (the
    //          input). On tick 2 input==7 -> every fired event's payload == 7.
    {
        GraphState st = MakeState(ctrlData);
        StepFlow(ctrlData, execG, st, ctrlInputs[0], 0);
        StepFlow(ctrlData, execG, st, ctrlInputs[1], 1);
        std::vector<EventRecord> e2 = StepFlow(ctrlData, execG, st, ctrlInputs[2], 2);  // input 7
        bool payloadOk = !e2.empty();
        for (const EventRecord& e : e2) if (e.payload != 7) payloadOk = false;
        check(payloadOk,
              "flow-s3: eEvent's payload == regs[pred] at that tick (hand-checked)");
    }

    // ---- (7) BOUNDED — a contrived exec graph with a LOOP (a successor reaching back) terminates within
    //          the step cap, no hang. Node 0 eSeq -> {0} loops onto itself forever absent the cap.
    {
        ExecGraph loopEg;
        loopEg.nodes.resize(1);
        loopEg.entry = 0;
        loopEg.nodes[0] = ExecNode{ eSeq, /*pred=*/0, /*eventId=*/0, /*next=*/{0u} };  // self-loop
        std::vector<Reg> dummyRegs(4, 0);
        std::vector<EventRecord> evLoop = RunExec(loopEg, dummyRegs);  // must RETURN (bounded), not hang
        check(evLoop.empty(),  // no eEvent in the loop -> no events; the point is it TERMINATES
              "flow-s3: the traversal is bounded (a contrived exec loop terminates deterministically, no hang)");
    }

    // ==================================================================================================
    // Slice FLOW-S4 — LOCKSTEP / REPLAY COMPOSITION: the moat payoff. Append-only (S1+S2+S3 above stay
    // green). The flow::Graph IS a net::Session StepFn: World = GraphState, Input = Reg, step = a lambda
    // capturing the static data graph that calls StepGraph, digest = DigestState. We reuse the net::
    // RunLockstep / DigestTrace / DesyncDetector templates VERBATIM (from the already-included net/session.h)
    // — ZERO new netcode. The data graph + input stream are the S2 showcase fixtures. T = the S2 tick count.
    // ==================================================================================================

    const Graph& dataG = stateGraph;                          // World source = the S2 showcase state graph
    const std::vector<std::vector<Reg>>& s4Stream = inputStream;  // the S2 8-tick input stream
    const uint32_t T = static_cast<uint32_t>(s4Stream.size()); // T = 8 (matches MakeShowcaseInputStream)

    // The StepFn shape net::Session::Advance templates over: step(world, inputs-this-tick, tick). The lambda
    // captures the static data graph (config, not state) and forwards to StepGraph (the S2 per-tick step).
    auto step   = [&](GraphState& w, const std::vector<Reg>& inputs, uint32_t tick) {
        StepGraph(dataG, w, inputs, tick);
    };
    auto s4Digest = [&](const GraphState& w) { return DigestState(w); };

    // Build the net::InputRing<Reg> from the flow input stream (the kInput channel index == insertion index).
    hf::net::InputRing<Reg> ring = BuildInputRing(s4Stream);

    // ---- PART A — lockstep over net::Session --------------------------------------------------------
    const uint64_t s4Final = hf::net::RunLockstep<GraphState, Reg>(
        MakeState(dataG), ring, T, step, s4Digest);
    std::printf("flow-s4: lockstep final state digest = 0x%016llx  (T=%u ticks)\n",
                static_cast<unsigned long long>(s4Final), T);

    // (1) PINNED LOCKSTEP FINAL — net::RunLockstep over the graph == a hard-pinned uint64 (a peer re-derives
    //     the bit-identical graph state from the input stream alone). PINNED on first run (MSVC == clang).
    const uint64_t kPinnedS4Final = 0x670cf80b235bdafdull;  // PINNED on first run
    check(s4Final == kPinnedS4Final,
          "flow-s4: net::RunLockstep over the graph == pinned uint64 (a peer re-derives the bit-identical graph state)");

    // (2) LOCKSTEP INVARIANT — two independent net::DigestTrace calls -> IDENTICAL per-tick traces.
    const std::vector<uint64_t> traceA = hf::net::DigestTrace<GraphState, Reg>(
        MakeState(dataG), ring, T, step, s4Digest);
    {
        const std::vector<uint64_t> traceA2 = hf::net::DigestTrace<GraphState, Reg>(
            MakeState(dataG), ring, T, step, s4Digest);
        check(traceA == traceA2,
              "flow-s4: two peers from the same input ring have EQUAL net::DigestTrace at EVERY tick (lockstep invariant)");
    }

    // (3) COMPOSITION (the make-or-break) — the net::Session-driven DigestTrace == the DIRECT S2
    //     RunGraphTrace tick-for-tick (proving the graph eval through the netcode engine is the SAME
    //     computation as the direct eval -> the graph is a valid deterministic StepFn).
    {
        const std::vector<uint64_t> directTrace = RunGraphTrace(dataG, s4Stream, T);
        check(traceA == directTrace,
              "flow-s4: the net::Session-driven DigestTrace == the direct S2 RunGraphTrace (the graph IS a valid StepFn)");
    }

    // (4) REPLAY-STABLE — a second RunLockstep over the same ring -> identical final digest.
    {
        const uint64_t s4Replay = hf::net::RunLockstep<GraphState, Reg>(
            MakeState(dataG), ring, T, step, s4Digest);
        check(s4Replay == s4Final,
              "flow-s4: a replay (second RunLockstep over the same ring) reproduces the identical final digest");
    }

    // ---- PART B — desync localization (the NS5 detector over a graph) -------------------------------
    // Two input streams identical except one tick K's input differs -> traces match for t<K, diverge at K;
    // net::DesyncDetector latches the exact tick.
    const uint32_t K = 3;  // PINNED: the divergence tick (input 0 -> 99 on tick 3)
    std::vector<std::vector<Reg>> streamB = s4Stream;
    streamB[static_cast<std::size_t>(K)] = { 99 };  // change tick K's single input (0 -> 99)
    hf::net::InputRing<Reg> ringB = BuildInputRing(streamB);
    const std::vector<uint64_t> traceB = hf::net::DigestTrace<GraphState, Reg>(
        MakeState(dataG), ringB, T, step, s4Digest);

    // (5) CLEAN — DesyncDetector over (traceA vs traceA) -> no desync.
    {
        hf::net::DesyncDetector dClean;
        for (uint32_t t = 0; t < T; ++t) hf::net::RecordLocal(dClean, t, traceA[t]);
        for (uint32_t t = 0; t < T; ++t)
            hf::net::IngestRemote(dClean, hf::net::ChecksumPacket{ t, traceA[t] });
        check(!dClean.desynced,
              "flow-s4: identical input streams report NO desync (clean)");
    }

    // (6) LOCATED — traceA (local) vs traceB (remote, tick K's input changed) -> d.desynced &&
    //     d.desyncTick == K, traces equal for t < K. Pin K.
    {
        hf::net::DesyncDetector dDiv;
        for (uint32_t t = 0; t < T; ++t) hf::net::RecordLocal(dDiv, t, traceA[t]);
        for (uint32_t t = 0; t < T; ++t)
            hf::net::IngestRemote(dDiv, hf::net::ChecksumPacket{ t, traceB[t] });
        bool matchBeforeK = true;
        for (uint32_t t = 0; t < K; ++t) if (traceA[t] != traceB[t]) matchBeforeK = false;
        std::printf("flow-s4: desync injected at tick K=%u, detector latched tick=%u\n",
                    K, dDiv.desyncTick);
        check(dDiv.desynced && dDiv.desyncTick == K && matchBeforeK,
              "flow-s4: a one-tick input divergence is LOCATED at the exact tick K (net::DesyncDetector), traces match for t<K");
    }

    // ==================================================================================================
    // Slice FLOW-S5 — ROLLBACK + SERIALIZATION: the netcode-grade capstone. Append-only (S1-S4 above stay
    // green). PART A: a 2-input rollback graph driven through net::RollbackSession via RunWithTransport
    // with a ScriptedTransport that DELAYS + MUTATES one remote -> a real mispredict + rollback that
    // converges to the bit-identical net::RunLockstep authority. PART B: snapshot completeness (the
    // verdict.h lesson). PART C: a Graph round-trips byte-identically (a savable visual script).
    // ==================================================================================================

    // ---- PART A — rollback correctness (the GGPO proof for visual scripting) -------------------------
    // The 2-channel rollback graph: net::RollbackSession steps step(world,{local,remote},tick), a 2-elem
    // input vector. So kInput[0]=LOCAL, kInput[1]=REMOTE feed a running accumulator via the S2 feedback
    // pattern:  acc = Add( Add(kInput[0], kInput[1]), Delay(acc) ).  A mispredicted remote genuinely
    // changes acc -> a wrong prediction diverges and the rollback visibly corrects it.
    //   n0 = kInput[0]  (LOCAL)
    //   n1 = kInput[1]  (REMOTE)
    //   n2 = Add(n0, n1)            = local + remote (this tick's contribution)
    //   n3 = kDelay(a=n4)           = the PREVIOUS-tick accumulator (the feedback lag; a is a state read)
    //   n4 = Add(n2, n3)            = (local+remote) + prevAcc -> THE RUNNING ACCUMULATOR
    Graph rbGraph;
    rbGraph.nodes.resize(5);
    rbGraph.nodes[0] = Node{ kInput, /*a=*/0, /*b=*/0, /*c=*/0, /*const=*/0 };  // LOCAL  (channel 0)
    rbGraph.nodes[1] = Node{ kInput, /*a=*/1, /*b=*/1, /*c=*/1, /*const=*/1 };  // REMOTE (channel 1)
    rbGraph.nodes[2] = Node{ kAdd,   /*a=*/0, /*b=*/1, /*c=*/2, 0 };            // local + remote
    rbGraph.nodes[3] = Node{ kDelay, /*a=*/4, /*b=*/3, /*c=*/3, 0 };            // prev accumulator (feedback)
    rbGraph.nodes[4] = Node{ kAdd,   /*a=*/2, /*b=*/3, /*c=*/4, 0 };            // running accumulator

    // The two input streams over RB_T ticks. local[t] is known immediately; remote[t] is the peer's input.
    const uint32_t RB_T = 8;
    const std::vector<Reg> rbLocal  = { 5, 2, 7, 1, 3, 9, 6, 4 };
    const std::vector<Reg> rbRemote = { 1, 4, 2, 8, 3, 7, 5, 9 };  // remote[3]=8 (the mispredict tick K=3)

    // The rollback step: a 2-channel StepGraph (ins = {local, remote}). Matches RollbackSession's
    // step(world, {localThisTick, remoteThisTick}, tick).
    auto rbStep = [&](GraphState& w, const std::vector<Reg>& ins, uint32_t tick) {
        StepGraph(rbGraph, w, ins, tick);
    };

    // Authority = net::RunLockstep over the TRUE combined per-tick {local[t], remote[t]} inputs. Build an
    // InputRing whose At(t) == {local[t], remote[t]} (channel 0 then 1 — the kInput constArg order).
    hf::net::InputRing<Reg> rbAuthorityRing;
    for (uint32_t t = 0; t < RB_T; ++t) {
        rbAuthorityRing.AddInput(t, rbLocal[t]);   // channel 0 = LOCAL
        rbAuthorityRing.AddInput(t, rbRemote[t]);  // channel 1 = REMOTE
    }
    const uint64_t rbAuthority = hf::net::RunLockstep<GraphState, Reg>(
        MakeState(rbGraph), rbAuthorityRing, RB_T,
        rbStep, [&](const GraphState& w) { return DigestState(w); });

    // The mispredict tick: remote[K] is DELAYED past its origin tick AND differs from the prediction
    // (the prediction reuses the last confirmed remote = remote[K-1]=2; the real remote[K]=8 != 2 -> a
    // genuine misprediction fires when remote[K] finally arrives). K=3.
    const uint32_t RB_K = 3;

    // PRIMARY schedule: deliver every remote on-time EXCEPT remote[K], delayed to deliver at tick K+2.
    hf::net::ScriptedTransport<Reg> tx;
    for (uint32_t t = 0; t < RB_T; ++t) {
        if (t == RB_K) hf::net::Schedule(tx, /*deliverTick=*/RB_K + 2, /*forTick=*/t, rbRemote[t]);
        else           hf::net::Schedule(tx, /*deliverTick=*/t,        /*forTick=*/t, rbRemote[t]);
    }
    hf::net::RollbackSession<GraphState, Reg> rbS;
    rbS.world = MakeState(rbGraph);
    hf::net::RunWithTransport<GraphState, Reg>(rbS, rbLocal, tx, RB_T, rbStep);
    const uint64_t rbFinal = DigestState(rbS.world);

    std::printf("flow-s5: rollback authority digest = 0x%016llx, didRollback = %s\n",
                static_cast<unsigned long long>(rbAuthority), rbS.didRollback ? "true" : "false");

    // PINNED on first run (MSVC == clang): the clean-authority rollback final digest.
    const uint64_t kPinnedRbAuthority = 0x88b1cdd54ce2e9b0ull;

    // (1) ROLLBACK == AUTHORITY — the rolled-back final state == the clean net::RunLockstep authority == pinned.
    check(rbFinal == rbAuthority && rbAuthority == kPinnedRbAuthority,
          "flow-s5: a mispredicted input rolls the graph state back to the BIT-IDENTICAL authority (== pinned uint64)");

    // (2) DIDROLLBACK — a real misprediction fired (remote[K]=8 != predicted lastConfirmed remote[K-1]=2).
    check(rbS.didRollback,
          "flow-s5: rollback actually fired (didRollback == true)");

    // (3) ADVERSARIAL — a heavier delay/reorder schedule STILL converges to the same authority digest.
    // remote[K] delayed even further (deliver at the LAST tick), remote[5] reordered to deliver at tick 1
    // (before its origin? no — at tick 1 < origin 5 means it arrives early as a future-confirm; harmless),
    // and a RESEND of remote[2] (a duplicate, a no-op). All confirmed before the run ends.
    hf::net::ScriptedTransport<Reg> txAdv;
    for (uint32_t t = 0; t < RB_T; ++t) {
        if (t == RB_K)      hf::net::Schedule(txAdv, /*deliver=*/RB_T - 1, /*forTick=*/t, rbRemote[t]);
        else if (t == 5)    hf::net::Schedule(txAdv, /*deliver=*/6,        /*forTick=*/t, rbRemote[t]);
        else if (t == 2)    hf::net::Schedule(txAdv, /*deliver=*/4,        /*forTick=*/t, rbRemote[t]);
        else                hf::net::Schedule(txAdv, /*deliver=*/t,        /*forTick=*/t, rbRemote[t]);
    }
    hf::net::Schedule(txAdv, /*deliver=*/7, /*forTick=*/2, rbRemote[2]);  // a RESEND of remote[2] (no-op)
    hf::net::RollbackSession<GraphState, Reg> rbAdv;
    rbAdv.world = MakeState(rbGraph);
    hf::net::RunWithTransport<GraphState, Reg>(rbAdv, rbLocal, txAdv, RB_T, rbStep);
    check(DigestState(rbAdv.world) == rbAuthority,
          "flow-s5: the adversarial (delayed/reordered/resent) schedule converges to the SAME pinned authority digest");

    // ---- PART B — snapshot completeness (the verdict.h lesson) ---------------------------------------
    // The combined-input stream for a straight per-tick advance (channel 0=local, 1=remote).
    auto rbAdvanceK = [&](GraphState& w, uint32_t fromTick, uint32_t k) {
        for (uint32_t i = 0; i < k; ++i) {
            const uint32_t t = fromTick + i;
            StepGraph(rbGraph, w, { rbLocal[t], rbRemote[t] }, t);
        }
    };
    const uint32_t RB_SNAP_AT = 2;   // snapshot after advancing 2 ticks
    const uint32_t RB_RE_K    = 3;   // re-advance 3 ticks (ticks 2,3,4)

    // (4) COMPLETE — from a state advanced RB_SNAP_AT ticks: SnapshotState; advance RB_RE_K (diverge with
    //     WRONG inputs); RestoreState; re-advance the SAME RB_RE_K correct ticks -> byte-identical to a
    //     straight advance of RB_RE_K from the snapshot.
    {
        GraphState base = MakeState(rbGraph);
        rbAdvanceK(base, 0, RB_SNAP_AT);                         // advance to the snapshot point
        const GraphState snap = SnapshotState(base);            // explicit snapshot

        // Diverge: advance with deliberately WRONG inputs to scramble the live state.
        GraphState live = base;
        for (uint32_t i = 0; i < RB_RE_K; ++i)
            StepGraph(rbGraph, live, { 999, 777 }, RB_SNAP_AT + i);

        // Restore + re-advance the SAME RB_RE_K correct ticks.
        RestoreState(live, snap);
        rbAdvanceK(live, RB_SNAP_AT, RB_RE_K);

        // The straight reference: a separate advance of RB_RE_K from the snapshot.
        GraphState straight = snap;
        rbAdvanceK(straight, RB_SNAP_AT, RB_RE_K);

        check(DigestState(live) == DigestState(straight),
              "flow-s5: advance->snapshot->diverge->restore->re-advance == straight-advance (snapshot is complete)");
    }

    // (5) INCOMPLETE DIVERGES — restore a snapshot with ONE register slot zeroed -> re-advance differs
    //     (proves no stateful slot escapes the snapshot — the feedback accumulator n4 is load-bearing).
    {
        GraphState base = MakeState(rbGraph);
        rbAdvanceK(base, 0, RB_SNAP_AT);
        const GraphState snap = SnapshotState(base);

        GraphState whole = snap;       rbAdvanceK(whole, RB_SNAP_AT, RB_RE_K);

        GraphState broken = snap;
        broken.prev[4] = 0;            // zero the accumulator slot n4 (an INCOMPLETE restore)
        rbAdvanceK(broken, RB_SNAP_AT, RB_RE_K);

        check(DigestState(whole) != DigestState(broken),
              "flow-s5: a deliberately INCOMPLETE restore (zeroing one register) DIVERGES (the snapshot must be whole)");
    }

    // ---- PART C — serialization round-trip (a savable visual script) --------------------------------
    {
        const Graph g = MakeShowcaseGraph();
        const std::vector<uint8_t> ser = SerializeGraph(g);
        Graph out;
        const bool ok = DeserializeGraph(ser, out);
        const std::vector<uint8_t> reSer = SerializeGraph(out);

        const uint64_t serDigest = hf::net::DigestBytes(ser.data(), ser.size());
        std::printf("flow-s5: SerializeGraph(showcase) digest = 0x%016llx  (%zu bytes)\n",
                    static_cast<unsigned long long>(serDigest), ser.size());

        // (6) ROUND-TRIP — deserialize succeeds, re-serialize is byte-exact, out.nodes == g.nodes field-for-field.
        bool fieldsEqual = (out.nodes.size() == g.nodes.size());
        for (std::size_t i = 0; fieldsEqual && i < g.nodes.size(); ++i)
            fieldsEqual = (out.nodes[i].kind     == g.nodes[i].kind &&
                           out.nodes[i].a        == g.nodes[i].a &&
                           out.nodes[i].b        == g.nodes[i].b &&
                           out.nodes[i].c        == g.nodes[i].c &&
                           out.nodes[i].constArg == g.nodes[i].constArg);
        check(ok && reSer == ser && fieldsEqual,
              "flow-s5: Deserialize(Serialize(graph)) round-trips byte-exact (a save-game / sync artifact)");

        // (7) PINNED SERIALIZED HASH — the on-disk format is byte-stable cross-platform (MSVC == clang).
        const uint64_t kPinnedSerDigest = 0xd26266a3a25e5145ull;  // PINNED on first run
        check(serDigest == kPinnedSerDigest,
              "flow-s5: SerializeGraph digest == pinned uint64 (stable on-disk format, identical MSVC + Mac/clang)");

        // (8) LOAD EVALUATES SAME — a loaded graph evaluates to the SAME result as the original.
        check(DigestGraph(Evaluate(out)) == DigestGraph(Evaluate(g)),
              "flow-s5: a loaded graph evaluates to the SAME result as the original (the script survives save/load)");

        // Truncation defense: a short buffer is rejected deterministically (no UB).
        std::vector<uint8_t> trunc = ser;
        if (trunc.size() > 4) trunc.resize(trunc.size() - 1);   // drop a byte -> not a whole record
        Graph dummy;
        check(!DeserializeGraph(trunc, dummy),
              "flow-s5: DeserializeGraph rejects a truncated buffer deterministically (defensive length checks)");
    }

    if (g_fail == 0) { std::printf("flow_test: ALL PASS\n"); return 0; }
    std::printf("flow_test: %d FAIL\n", g_fail);
    return 1;
}
