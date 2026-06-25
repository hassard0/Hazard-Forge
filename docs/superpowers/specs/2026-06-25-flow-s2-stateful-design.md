# Slice FLOW-S2 — Stateful nodes + the per-tick step (Issue #24, the StepFn shape)

S1 built the stateless graph: a DAG of integer nodes evaluated in a canonical topological order. S2 makes
the VM TICK: persistent per-node state across ticks, external per-tick inputs, and stateful nodes
(Counter, Delay, Latch) — driven by `StepGraph(graph, state, inputs, tick)`, the exact signature
`net::Session<World,Input>::Advance`'s `step(world, inputs, tick)` templates over (so S4 wraps it in the
rollback engine with zero new netcode). The crux: stateful nodes read PREVIOUS state instead of
current-tick edges, which lets graphs have **feedback loops** (the Blueprint/dataflow staple) WITHOUT a
topological cycle — and stays bit-identical because state is just a register file snapshot.

Pure-CPU INTEGER, append-only to `engine/flow/flow.h` (do NOT modify S1 — ADD below). NO render. Golden =
a pinned per-tick `DigestBytes` trace, identical Windows/MSVC + Mac/clang via the standalone clang compile.

## Append to engine/flow/flow.h (below S1, in hf::flow)

1. **New node kinds** (extend the `Kind` enum — APPEND new values, do NOT renumber S1's kConst..kSelect):
   ```cpp
   // ... kConst=0 ... kSelect=6 (S1, unchanged) ...
   kInput   = 7,   // output = inputs[constArg]  (the external per-tick input at index constArg; 0 if oob)
   kCounter = 8,   // output = prevState[self] + constArg   (accumulates by constArg each tick)
   kDelay   = 9,   // output = prevState[a]   (the input node a's value from the PREVIOUS tick — a 1-tick lag)
   kLatch   = 10,  // output = (regs[c] != 0) ? regs[a] : prevState[self]  (hold until the predicate re-fires)
   ```
2. **`GraphState`** — the persistent register file (one `Reg` per node = its value at the END of the last
   tick; the snapshot-able state):
   ```cpp
   struct GraphState { std::vector<Reg> prev; };   // prev[NodeId]; sized to graph.nodes.size()
   inline GraphState MakeState(const Graph& g) { GraphState s; s.prev.assign(g.nodes.size(), 0); return s; }
   ```
3. **`EdgeMask`** — the per-kind CURRENT-TICK topo-edge semantics (THE determinism-critical bit). Returns
   which of `{a, b, c}` are real current-tick data edges for `TopoOrder`/`Evaluate` (the others are either
   unused or STATE reads from `prev`, which are NOT topo edges):
   ```cpp
   // bit0=a, bit1=b, bit2=c. State reads (kDelay's a, kCounter/kLatch self) are NOT edges -> feedback OK.
   inline uint32_t EdgeMask(uint32_t kind) {
       switch (kind) {
           case kAdd: case kSub: case kMul: case kMin: case kMax: return 0b011;  // a,b
           case kSelect:                                            return 0b111;  // a,b,c
           case kLatch:                                             return 0b101;  // a,c (b unused; self via prev)
           // kConst, kInput, kCounter, kDelay: no current-tick edges (kDelay's a is a PREV-state read).
           default:                                                 return 0b000;
       }
   }
   ```
   **`TopoOrder` and the S1 in-degree must use `EdgeMask`** — count an input field as an edge only if its
   bit is set in `EdgeMask(kind)` AND it `IsRealEdge` (in-range, non-self). This is the one change to the
   S1 scheduler — make it ADDITIVE: either (a) S1's `TopoOrder` already only counts `IsRealEdge` inputs and
   you refine it to also gate on `EdgeMask` (a small edit — acceptable since S1's stateless kinds all have
   their inputs as edges, so gating by `EdgeMask` is byte-identical for an S1-only graph → S1's pinned
   digest UNCHANGED), or (b) add a new `TopoOrderTick` used by `StepGraph`. **Prefer (a) but VERIFY S1's
   pinned digest `0x0e5b8ec26f0d8730` is unchanged** (the S1 kinds' EdgeMask matches "all listed inputs are
   edges", so it must be). Document which you did.
4. **`StepGraph`** — one deterministic tick:
   ```cpp
   // Evaluate the graph for this tick into a fresh register file, then commit it as the new prevState.
   // Stateful kinds read `state.prev` for their feedback/held values; kInput reads `inputs`; everything
   // else is the S1 Evaluate. Returns the new register file (== state.prev after the call).
   inline std::vector<Reg> StepGraph(const Graph& g, GraphState& state,
                                     const std::vector<Reg>& inputs, uint32_t tick);
   ```
   Algorithm: `TopoOrder` (now EdgeMask-gated); walk the order computing each node's `Reg` into `regs`:
   - S1 kinds: exactly as `Evaluate` (kConst/kAdd/.../kSelect over `regs[edge]`).
   - kInput: `(constArg >= 0 && (size_t)constArg < inputs.size()) ? inputs[constArg] : 0`.
   - kCounter: `state.prev[self] + constArg`.
   - kDelay: `state.prev[a]` (a's PREVIOUS-tick value; if `a` out-of-range read 0). NOTE a is a state read,
     so kDelay is a topo-root (EdgeMask 0) — it's evaluated early, but it reads `prev`, so a feedback loop
     `a -> Add -> Delay -> (feeds a)` is NOT a topo cycle.
   - kLatch: `(regs[c] != 0) ? regs[a] : state.prev[self]`.
   Then `state.prev = regs;` and `return regs;`. (The `tick` param is available for future tick-aware nodes;
   v1 may not read it — that's fine, keep it in the signature to match `Advance`.)
5. **`uint64_t DigestState(const GraphState& s)`** = `net::DigestBytes(s.prev.data(), s.prev.size()*sizeof(Reg))`
   (== `DigestGraph(s.prev)`; add it for clarity).
6. **`std::vector<uint64_t> RunGraphTrace(const Graph& g, const std::vector<std::vector<Reg>>& inputStream,
   uint32_t ticks)`** — run `ticks` `StepGraph`s from a fresh state over the per-tick `inputStream`,
   recording `DigestState` AFTER each tick → a per-tick digest trace (the `net::DigestTrace` shape, proving
   tick-by-tick determinism). `inputStream[t]` is the inputs for tick t (empty/short → zeros).

### Fixtures
- `Graph MakeShowcaseStateGraph()` — a fixed graph exercising all four stateful kinds: a kInput, a kCounter,
  a kDelay forming a FEEDBACK loop (e.g. `acc = Add(input, Delay(acc))` — a running accumulator via
  feedback, the canonical proof feedback works without a cycle), and a kLatch gated by a tick-derived
  predicate. Built so the per-tick trace is non-trivial.
- `std::vector<std::vector<Reg>> MakeShowcaseInputStream()` — a fixed per-tick input stream (e.g. 8 ticks of
  varied integer inputs).

## The golden (PINNED, cross-platform) — append to tests/flow_test.cpp
```
flow-s2: per-tick trace (8 ticks) final digest = 0x<...>
PASS flow-s2: RunGraphTrace digest trace == pinned uint64[] (every tick, deterministic)
PASS flow-s2: re-running the same graph+inputs is bit-identical
PASS flow-s2: the kDelay feedback loop is NOT a topo cycle (TopoOrder succeeds) and accumulates correctly
PASS flow-s2: kCounter increments by constArg each tick (hand-checked sequence)
PASS flow-s2: kDelay outputs the previous tick's input value (1-tick lag, hand-checked)
PASS flow-s2: kLatch holds its value until the predicate re-fires (hand-checked)
PASS flow-s2: S1's pinned digest 0x0e5b8ec26f0d8730 is UNCHANGED (the EdgeMask refinement is byte-identical for S1 graphs)
```
Assertions:
1. **PINNED TRACE** — `RunGraphTrace(MakeShowcaseStateGraph(), MakeShowcaseInputStream(), 8)` == a pinned
   `std::vector<uint64_t>` (or pin the FNV digest of the trace + the final-tick digest; run once, pin).
2. **REPLAY-STABLE** — a second run → identical trace.
3. **FEEDBACK NOT A CYCLE** — `TopoOrder` of the feedback graph returns `true` (kDelay's `a` is a state
   read, not an edge); the accumulator produces the expected running sum (hand-checked first few ticks).
4. **COUNTER** — a kCounter's value over ticks is `t * constArg` (hand-checked).
5. **DELAY LAG** — a kDelay's output at tick `t` equals its input's value at tick `t-1` (0 at tick 0).
6. **LATCH HOLD** — a kLatch holds its captured value across ticks where the predicate is 0, updates when
   it's nonzero (hand-checked).
7. **S1 INVARIANT** — `DigestGraph(Evaluate(MakeShowcaseGraph()))` still == `0x0e5b8ec26f0d8730` (the
   EdgeMask change did not perturb S1).

Keep S1's assertions green (append-only — S1's pinned digest unchanged).

## Cross-platform proof (cheap loop — NO render-bake)
Controller `scp`s `engine/flow/flow.h` + `engine/net/session.h` + `tests/flow_test.cpp` (+ `tests/test_main.h`
+ `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/flow_test.cpp -o /tmp/flow && /tmp/flow`, confirming ALL assertions PASS with the IDENTICAL pinned
trace. (Local Windows clang is the fast pre-check.) NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/flow/flow.h` (add the new kinds + `GraphState`/`MakeState`/`EdgeMask`/`StepGraph`/
  `DigestState`/`RunGraphTrace` + fixtures; the one allowed S1 edit is gating `TopoOrder`'s edge count by
  `EdgeMask` — which MUST leave S1's pinned digest byte-identical; verify it). Header stays SELF-CONTAINED:
  only `<cstddef>/<cstdint>/<vector>` + `net/session.h`. Do NOT include any other header. Do NOT modify
  `net/session.h`.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<random>`/`<unordered_*>`/`<map>`/`<functional>`/
  `std::hash`/`<algorithm>` (ternary min/max). Topo tie-break stays lowest-id ascending-scan.
- `tests/flow_test.cpp` stays self-contained; APPEND the S2 assertions + fixtures. Keep S1's green.
- Branch `fix-flow-s2`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target flow_test'`
  then run `flow_test` and confirm ALL assertions (S1 + S2) PASS, exit 0. ALSO compile standalone with the
  local clang (`C:\Program Files\LLVM\bin\clang++.exe`) and confirm IDENTICAL digests. First run: pin.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `flow_test` builds + PASSES on Windows with
  every assertion green (esp. the feedback-not-a-cycle proof + the S1-invariant), and the local clang
  standalone passes with identical digests. Report: commit hash, full test output (printed trace/digests +
  PASS lines), the pinned values, confirmation S1's `0x0e5b8ec26f0d8730` is unchanged, confirmation the
  header is still self-contained (list `#include`s), the showcase state-graph + input stream you built, how
  you handled the `EdgeMask`/`TopoOrder` refinement (option a vs b), and the local-clang result. Commit
  message via temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical trace, ff-merges to
  master + pushes + deletes the branch + advances to S3 — control flow + events.)
