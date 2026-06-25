# Slice FLOW-S4 — Lockstep / replay composition: the moat payoff (Issue #24)

This is the flagship's headline. S1–S3 built a deterministic node-graph VM (data + state + control). S4
proves the moat: a `flow::Graph` is a pure function `(state, inputs) -> state'` — EXACTLY the `StepFn`
`net::Session<World,Input>::Advance` / `RunLockstep` / `DigestTrace` / `RollbackSession` template over — so
a visual script runs ON the existing rollback-netcode engine with ZERO new netcode. Two peers fed only the
input stream re-derive a **bit-identical** graph state at every tick (lockstep), the run is **replay-stable**,
and a divergence is **located** at the exact tick (desync detector). UE5 Blueprints structurally cannot do
this — their event order is non-deterministic, which is why UE5's deterministic-rollback path excludes
Blueprints. A deterministic, lockstep/replay-able visual script is the capability UE5 lacks.

Pure-CPU INTEGER. The lockstep/replay/desync proof reuses `net::Session`/`RunLockstep`/`DigestTrace`/
`DesyncDetector`/`ChecksumPacket`/`RecordLocal`/`IngestRemote`/`InputRing` VERBATIM in the TEST — NO new
netcode. Golden = pinned `uint64`s, identical Windows/MSVC + Mac/clang via the standalone clang compile.

## The mapping (World = GraphState, Input = Reg)
- **World** = `flow::GraphState` (the persistent register file — a plain `std::vector<int32_t>`, so
  `net::Session`'s value-copy snapshot/restore works out of the box, like the toy worlds).
- **Input** = `flow::Reg` — one per `kInput` channel. `net::InputRing<Reg>::At(t)` returns the vector of
  inputs for tick t (insertion order); a `kInput` node reads `inputs[constArg]`, so the ring's per-tick
  inputs map directly to the channels.
- **Step** = a lambda capturing the static `Graph` (the data graph is config, not state):
  `[&](GraphState& w, const std::vector<Reg>& inputs, uint32_t tick){ StepGraph(dataG, w, inputs, tick); }`.
- **Digest** = `[&](const GraphState& w){ return DigestState(w); }`.

## Append to engine/flow/flow.h (below S3, in hf::flow) — minimal
Only a tiny convenience (everything else is the netcode reused in the test):
```cpp
// Build a net::InputRing<Reg> from a per-tick input stream: for each tick t, AddInput(t, v) for each v in
// stream[t] (so the kInput channel index == the insertion index). The bridge from a flow input stream to
// the net::Session input model.
inline hf::net::InputRing<Reg> BuildInputRing(const std::vector<std::vector<Reg>>& stream) {
    hf::net::InputRing<Reg> ring;
    for (uint32_t t = 0; t < stream.size(); ++t)
        for (const Reg v : stream[t]) ring.AddInput(t, v);
    return ring;
}
```
(That's it for the header — S4's substance is proving composition in the test. Optionally add a thin
`uint64_t GenerateFlow(const Graph& dataG, const std::vector<std::vector<Reg>>& stream, uint32_t ticks)` =
`net::RunLockstep<GraphState,Reg>(MakeState(dataG), BuildInputRing(stream), ticks, step, DigestState)` if a
named entry point reads cleaner — implementer's call.)

## The goldens (PINNED, cross-platform) — append to tests/flow_test.cpp
Use the S2 `MakeShowcaseStateGraph()` + `MakeShowcaseInputStream()` (the data graph is the World; its
per-tick state digest is the lockstep currency). Build the step + digest lambdas + the input ring.

### Part A — lockstep over net::Session
```
flow-s4: lockstep final state digest = 0x<...>  (T ticks)
PASS flow-s4: net::RunLockstep over the graph == pinned uint64 (a peer re-derives the bit-identical graph state)
PASS flow-s4: two peers from the same input ring have EQUAL net::DigestTrace at EVERY tick (lockstep invariant)
PASS flow-s4: the net::Session-driven DigestTrace == the direct S2 RunGraphTrace (the graph IS a valid StepFn)
PASS flow-s4: a replay (second RunLockstep over the same ring) reproduces the identical final digest
```
1. **PINNED LOCKSTEP FINAL** — `net::RunLockstep<GraphState, Reg>(MakeState(dataG), BuildInputRing(stream),
   T, step, DigestState)` == a hard-pinned `uint64_t` (run once, pin; identical MSVC + clang). (T = the S2
   tick count, e.g. 8.)
2. **LOCKSTEP INVARIANT** — two independent `net::DigestTrace<GraphState, Reg>(...)` calls → IDENTICAL
   per-tick traces (every tick equal — the lockstep invariant, the `session_test` shape).
3. **COMPOSITION (the make-or-break)** — the `net::Session`-driven `DigestTrace` equals the DIRECT S2
   `RunGraphTrace(dataG, stream, T)` tick-for-tick — proving the graph eval through the netcode engine is
   the SAME computation as the direct eval (the graph is a valid deterministic `StepFn`).
4. **REPLAY-STABLE** — a second `RunLockstep` over the same ring → identical final digest.

### Part B — desync localization (the NS5 detector over a graph)
Two input streams identical except one tick `K`'s input differs → traces match for `t < K`, diverge at `K`;
`net::DesyncDetector` latches the exact tick.
```
flow-s4: desync injected at tick K=<k>, detector latched tick=<k>
PASS flow-s4: identical input streams report NO desync (clean)
PASS flow-s4: a one-tick input divergence is LOCATED at the exact tick K (net::DesyncDetector), traces match for t<K
```
5. **CLEAN** — `DesyncDetector` over (traceA vs traceA) → no desync.
6. **LOCATED** — traceA vs traceB (tick K's input changed) → `d.desynced && d.desyncTick == K`, traces equal
   for `t < K`. Pin K.

Keep S1+S2+S3 assertions green (append-only — S1 `0x0e5b8ec26f0d8730`, S2 `0x670cf80b235bdafd`, S3
`0xd5735423148033cc` unchanged).

## Cross-platform proof (cheap loop — NO render-bake)
Controller `scp`s `engine/flow/flow.h` + `engine/net/session.h` + `tests/flow_test.cpp` (+ `tests/test_main.h`
+ `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/flow_test.cpp -o /tmp/flow && /tmp/flow`, confirming ALL assertions PASS with IDENTICAL pinned digests
+ latched tick. (Local Windows clang is the fast pre-check.) NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/flow/flow.h` (add `BuildInputRing` [+ optional `GenerateFlow`] below S3; do NOT
  modify S1/S2/S3). Header stays SELF-CONTAINED: only `<cstddef>/<cstdint>/<vector>` + `net/session.h`
  (which already provides `InputRing`/`Session`/`RunLockstep`/`DigestTrace`/`DesyncDetector`). NO new
  includes. Do NOT modify `net/session.h`.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<random>`/`<unordered_*>`/`<map>`/`<functional>`/
  `std::hash`/`<algorithm>`.
- `tests/flow_test.cpp` stays self-contained; APPEND the S4 assertions (it may use the `net::` templates from
  the already-included `net/session.h`). Keep S1–S3 green.
- Branch `fix-flow-s4`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target flow_test'`
  then run `flow_test` and confirm ALL assertions (S1–S4) PASS, exit 0. ALSO local clang standalone,
  identical digests + latched tick. First run: pin.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `flow_test` builds + PASSES on Windows with
  every assertion green (esp. the lockstep invariant + the composition equality + the desync-located-at-K),
  and the local clang standalone passes with identical digests. Report: commit hash, full test output
  (printed digests + latched tick + PASS lines), the pinned digests + T + K, confirmation S1–S3 digests
  unchanged, confirmation the header is still self-contained (list `#include`s), and the local-clang result.
  Commit message via temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests + latched
  tick, ff-merges to master + pushes + deletes the branch + advances to S5 — rollback + serialization.)
