# Slice FLOW-S3 — Control flow + events: the deterministic Blueprint exec wire (Issue #24)

S1/S2 built the DATA graph (a DAG of integer nodes, stateful, ticking). Blueprint has a SECOND wire kind —
the white **execution pin** that decides WHICH nodes run and in what ORDER. S3 adds that exec layer: an
exec graph of Branch / Sequence / Gate / Event nodes, traversed from an entry in a PINNED order, reading
predicates from the data register file. The whole point is determinism: in UE5 the exec/event order
depends on tick groups + actor registration + float timing (two machines fire events in different orders);
here two peers fed the same inputs fire the **same events on the same tick, in the same order** — the
literal Blueprint execution model made bit-identical.

Pure-CPU INTEGER, append-only to `engine/flow/flow.h` (do NOT modify S1/S2 — ADD below). NO render. Golden
= a pinned per-tick EVENT-trace digest, identical Windows/MSVC + Mac/clang via the standalone clang compile.

## Append to engine/flow/flow.h (below S2, in hf::flow)

1. **The exec layer types** — exec nodes are SEPARATE from data nodes (data = values, exec = control flow):
   ```cpp
   enum ExecKind : uint32_t { eSeq = 0, eBranch = 1, eGate = 2, eEvent = 3 };

   struct ExecNode {
       uint32_t              kind = eSeq;   // ExecKind
       NodeId                pred = 0;      // a DATA node id: the Branch/Gate predicate, or an Event's payload source
       uint32_t              eventId = 0;   // eEvent: the fired event's id (a fixed tag)
       std::vector<uint32_t> next;          // ordered exec-successor indices (into ExecGraph::nodes)
   };
   struct ExecGraph {
       std::vector<ExecNode> nodes;
       uint32_t              entry = 0;      // the exec node traversal starts from
   };

   struct EventRecord { uint32_t eventId = 0; Reg payload = 0; };   // one fired event
   ```
2. **`RunExec`** — traverse the exec graph from `entry` in a PINNED order, reading predicates from the data
   register file `regs` (the S2 `StepGraph` output), returning the fired events in traversal order:
   ```cpp
   inline std::vector<EventRecord> RunExec(const ExecGraph& eg, const std::vector<Reg>& regs);
   ```
   Traversal (DETERMINISTIC + BOUNDED — the determinism crux):
   - A worklist/stack seeded with `entry`. Pop, dispatch by kind, push successors **in pinned order** so
     they're processed in `next`-list order (e.g. push `next` reversed onto a stack so it pops forward, OR
     a recursive DFS over `next` in order — pick one, document it; both are deterministic given fixed
     `next` vectors). NO `unordered_*`, no hash order.
   - `eSeq`: fire ALL of `next` in order (push them all).
   - `eBranch`: read `regs[pred]` (out-of-range → 0); fire `next[0]` if `regs[pred] != 0` (the TRUE pin),
     else `next[1]` (the FALSE pin) — exactly ONE successor. (Missing pin → no successor.)
   - `eGate`: fire `next[0]` only if `regs[pred] != 0` (open), else nothing — the gate.
   - `eEvent`: emit `EventRecord{ eventId, payload = regs[pred] (oob→0) }`, then fire `next[0]`.
   - **Bound the traversal** by a step cap (e.g. `eg.nodes.size() * K` or a fixed cap) so an exec loop can't
     hang — a deterministic give-up, never UB. (v1 exec graphs are bounded; the cap is the safety net.) A
     per-node visited guard is OPTIONAL (Blueprint allows re-entry; v1 may cap re-entry via the step bound).
     Document the chosen rule.
3. **`StepFlow`** — the FULL per-tick step combining S2 data + S3 control → events (this is the `StepFn` S4
   wraps in `net::Session`):
   ```cpp
   inline std::vector<EventRecord> StepFlow(const Graph& dataG, const ExecGraph& execG,
                                            GraphState& state, const std::vector<Reg>& inputs, uint32_t tick);
   ```
   = `StepGraph(dataG, state, inputs, tick)` (updates `state`, returns `regs`) then `RunExec(execG, regs)`.
4. **`uint64_t DigestEvents(const std::vector<EventRecord>& ev)`** — hand-LE the events (PutU32 eventId +
   PutI32 payload per record into a byte buffer, then `net::DigestBytes`) — the event-trace golden currency.
   (Hand-serialize the fields, NOT a struct memcpy, to be padding-safe cross-platform — the replay.h
   discipline.)
5. **`std::vector<uint64_t> RunFlowTrace(const Graph& dataG, const ExecGraph& execG,
   const std::vector<std::vector<Reg>>& inputStream, uint32_t ticks)`** — run `ticks` `StepFlow`s from a
   fresh state, recording `DigestEvents(events)` AFTER each tick → the per-tick event-trace digest stream.

### Fixtures
- `ExecGraph MakeShowcaseExecGraph()` — a fixed exec graph over the S2 data graph (or a small dedicated data
  graph): an entry `eSeq` → a `eBranch` on a tick-parity predicate (a data node that is nonzero on
  odd/even ticks — e.g. derive parity from a kCounter or kInput) routing to a TRUE `eEvent` vs a FALSE
  `eEvent`; a `eGate` that blocks an event when its predicate is 0; an `eSeq` firing two events in a fixed
  order (to prove order). Built so the per-tick event trace is non-trivial + exercises every ExecKind.
- A `Graph MakeShowcaseControlData()` (or reuse the S2 graph) providing the predicates (a parity node, a
  gate-condition node).
- `std::vector<std::vector<Reg>> MakeControlInputStream()` — a fixed per-tick input stream.

## The golden (PINNED, cross-platform) — append to tests/flow_test.cpp
```
flow-s3: per-tick event trace (N ticks) final digest = 0x<...>
PASS flow-s3: RunFlowTrace event-trace digest stream == pinned uint64[] (deterministic per-tick events)
PASS flow-s3: re-running is bit-identical
PASS flow-s3: eBranch fires ONLY the taken pin's event (true on odd ticks, false on even — hand-checked)
PASS flow-s3: eSeq fires its successor events in the FIXED order (hand-checked event sequence)
PASS flow-s3: eGate blocks the event when its predicate is 0, passes when nonzero (hand-checked)
PASS flow-s3: eEvent's payload == regs[pred] at that tick (hand-checked)
PASS flow-s3: the traversal is bounded (a contrived exec loop terminates deterministically, no hang)
```
Assertions:
1. **PINNED EVENT TRACE** — `RunFlowTrace(...)` event-trace digest stream == a pinned `std::vector<uint64_t>`
   (run once, pin).
2. **REPLAY-STABLE** — second run → identical.
3. **BRANCH** — on a tick where the predicate is nonzero, the TRUE event fires (and NOT the false); on a
   zero tick, the FALSE event fires — hand-check the EventRecords for two specific ticks.
4. **SEQUENCE ORDER** — an `eSeq` with two `eEvent` successors fires them in `next`-list order (the event
   trace has them in that order, not reversed/hash order).
5. **GATE** — an `eGate` event is absent on ticks where its predicate is 0, present when nonzero.
6. **EVENT PAYLOAD** — an `eEvent`'s `payload` equals `regs[pred]` at that tick (hand-check).
7. **BOUNDED** — a contrived exec graph with a loop (a node whose successor reaches back to it) terminates
   within the step cap (no hang), deterministically.

Keep S1+S2 assertions green (append-only — S1 `0x0e5b8ec26f0d8730`, S2 trace `0x670cf80b235bdafd` unchanged).

## Cross-platform proof (cheap loop — NO render-bake)
Controller `scp`s `engine/flow/flow.h` + `engine/net/session.h` + `tests/flow_test.cpp` (+ `tests/test_main.h`
+ `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/flow_test.cpp -o /tmp/flow && /tmp/flow`, confirming ALL assertions PASS with the IDENTICAL pinned
event trace. (Local Windows clang is the fast pre-check.) NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/flow/flow.h` (add the exec types + `RunExec`/`StepFlow`/`DigestEvents`/
  `RunFlowTrace` + fixtures below S2; do NOT modify S1/S2). Header stays SELF-CONTAINED: only `<cstddef>/
  <cstdint>/<vector>` + `net/session.h`. NO new includes. Do NOT modify `net/session.h`.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<random>`/`<unordered_*>`/`<map>`/`<functional>`/
  `std::hash`/`<algorithm>`. The exec traversal order is PINNED (fixed `next`-vector order, never hash/
  insertion-of-a-set order).
- `tests/flow_test.cpp` stays self-contained; APPEND the S3 assertions + fixtures. Keep S1+S2 green.
- Branch `fix-flow-s3`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target flow_test'`
  then run `flow_test` and confirm ALL assertions (S1+S2+S3) PASS, exit 0. ALSO local clang standalone,
  identical digests. First run: pin.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `flow_test` builds + PASSES on Windows with
  every assertion green (esp. branch-only-taken-pin + sequence-order + bounded-traversal), and the local
  clang standalone passes with identical digests. Report: commit hash, full test output (printed event
  trace/digests + PASS lines), the pinned values, confirmation S1/S2 digests unchanged, confirmation the
  header is still self-contained (list `#include`s), the exec graph you built + the traversal rule (stack/
  recursion + the step cap + visited rule), and the local-clang result. Commit message via temp file +
  `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical event trace, ff-merges
  to master + pushes + deletes the branch + advances to S4 — lockstep/replay composition via net::Session.)
