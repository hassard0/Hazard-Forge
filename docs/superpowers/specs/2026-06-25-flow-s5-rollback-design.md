# Slice FLOW-S5 — Rollback + serialization: the netcode-grade capstone (Issue #24)

S4 proved a `flow::Graph` composes with `net::Session` for lockstep/replay/desync. S5 completes the
netcode-grade runtime: **rollback** (a mispredicted input rolls the graph state back to the bit-identical
authority via `net::RollbackSession`) and **serialization** (a graph round-trips byte-identically — a
visual script is a savable, shippable artifact: a save game / a multiplayer-sync delta). This is GGPO-class
rollback for visual scripting — UE5 Blueprints cannot roll back deterministically (their non-deterministic
event order breaks re-simulation), which is exactly why UE5's rollback path excludes them.

Pure-CPU INTEGER, append-only to `engine/flow/flow.h` (do NOT modify S1–S4 — ADD below). The rollback proof
reuses `net::RollbackSession`/`StepPredicted`/`ConfirmRemote`/`ScriptedTransport`/`RunWithTransport`
VERBATIM in the TEST. Golden = pinned `uint64`s, identical Windows/MSVC + Mac/clang.

## The rollback input model (2 channels: local + remote)
`net::RollbackSession` steps `step(world, {localThisTick, remoteThisTick}, tick)` — a 2-element input
vector per tick (the local peer's input + the predicted/confirmed remote input). For flow that maps to a
graph with TWO `kInput` channels: `kInput[0]` reads the LOCAL input, `kInput[1]` reads the REMOTE input. So
S5 uses a small **2-input rollback graph** where a mispredicted remote genuinely changes the state (e.g. a
running accumulator `acc += local + remote` via the S2 feedback pattern) so a wrong prediction diverges and
the rollback visibly corrects it.

## Append to engine/flow/flow.h (below S4, in hf::flow)

1. **`SnapshotState` / `RestoreState`** — explicit state snapshot/restore (the `net::Session` value-copy
   works implicitly, but make it explicit + prove COMPLETENESS the verdict.h way):
   ```cpp
   inline GraphState SnapshotState(const GraphState& s) { return s; }                 // deep copy (vector)
   inline void       RestoreState(GraphState& s, const GraphState& snap) { s = snap; } // bit-exact restore
   ```
   (Trivial because `GraphState{prev}` is one contiguous integer vector — EVERY stateful node's slot is in
   it by construction. The proof, not the code, is the slice's value: an incomplete restore must diverge.)
2. **Serialization (hand little-endian, the replay.h/wav.cpp discipline — NEVER memcpy a host struct)** —
   serialize a `Graph` (the static visual script) to bytes + decode back:
   ```cpp
   inline void PutU32(std::vector<uint8_t>& b, uint32_t v);   // 4 bytes LE
   inline uint32_t GetU32(const uint8_t* p);                  // read 4 bytes LE
   // Layout: nodeCount(u32), then per node: kind(u32), a(u32), b(u32), c(u32), constArg(u32 of its bits).
   inline std::vector<uint8_t> SerializeGraph(const Graph& g);
   inline bool DeserializeGraph(const std::vector<uint8_t>& bytes, Graph& out);   // false on truncation
   ```
   (Reg `constArg` is `int32_t` — serialize as `PutU32((uint32_t)constArg)`, read back `(Reg)GetU32(...)`.
   Self-contained: keep `PutU32`/`GetU32` inline in flow.h. If the S3 `DigestEvents` already added a `PutU32`
   in `hf::flow`, REUSE it — do NOT redefine; otherwise add it here.)
   - Optionally also `SerializeState`/`DeserializeState` for the runtime `GraphState` (a contiguous int
     vector — trivial LE) if the rollback/save story wants the live state too. Graph (the script) is the
     required one; state is optional.

## The goldens (PINNED, cross-platform) — append to tests/flow_test.cpp

### Part A — rollback correctness (the GGPO proof for visual scripting)
Build a 2-input rollback graph + local/remote input streams; drive `net::RollbackSession<GraphState, Reg>`
via `net::RunWithTransport` with a `ScriptedTransport` that DELAYS at least one remote input past its origin
tick AND makes the delayed remote differ from the prediction (so a real mispredict + rollback fires).
Authority = `net::RunLockstep` over the TRUE combined `{local[t], remote[t]}` per-tick inputs.
```
flow-s5: rollback authority digest = 0x<...>, didRollback = <b>
PASS flow-s5: a mispredicted input rolls the graph state back to the BIT-IDENTICAL authority
PASS flow-s5: rollback actually fired (didRollback == true)
PASS flow-s5: the adversarial (delayed/reordered) schedule converges to the SAME pinned authority digest
```
1. **ROLLBACK == AUTHORITY** — `RunWithTransport` final `DigestState(s.world)` == the clean-authority digest
   (a `net::RunLockstep` over the true `{local, remote}` inputs) == a pinned `uint64_t`.
2. **DIDROLLBACK** — `s.didRollback == true` (a real misprediction fired — the delayed remote differs from
   the predicted lastConfirmed).
3. **ADVERSARIAL CONVERGENCE** — same under a delay/reorder schedule.

### Part B — snapshot completeness (the verdict.h lesson)
```
PASS flow-s5: advance->snapshot->diverge->restore->re-advance == straight-advance (snapshot is complete)
PASS flow-s5: a deliberately INCOMPLETE restore (zeroing one register) DIVERGES (the snapshot must be whole)
```
4. **COMPLETE** — from a state, `SnapshotState`; advance K ticks (diverge); `RestoreState`; re-advance the
   SAME K ticks → byte-identical to a straight advance of the same K ticks from the snapshot. (Proves the
   whole `prev` vector is the state.)
5. **INCOMPLETE DIVERGES** — restore a snapshot with ONE register slot zeroed, re-advance → a DIFFERENT
   digest (proves no stateful slot escapes the snapshot — the verdict.h completeness discipline).

### Part C — serialization round-trip (a savable visual script)
```
flow-s5: SerializeGraph(showcase) digest = 0x<...>  (<N> bytes)
PASS flow-s5: Deserialize(Serialize(graph)) round-trips byte-exact (a save-game / sync artifact)
PASS flow-s5: SerializeGraph digest == pinned uint64 (stable on-disk format, identical MSVC + Mac/clang)
PASS flow-s5: a loaded graph evaluates to the SAME result as the original (the script survives save/load)
```
6. **ROUND-TRIP** — `DeserializeGraph(SerializeGraph(g), out)` succeeds and `SerializeGraph(out) ==
   SerializeGraph(g)` byte-exact; the decoded `out.nodes` equals `g.nodes` field-for-field.
7. **PINNED SERIALIZED HASH** — `net::DigestBytes(SerializeGraph(MakeShowcaseGraph()))` == a pinned
   `uint64_t` (the on-disk format is byte-stable cross-platform).
8. **LOAD EVALUATES SAME** — `DigestGraph(Evaluate(out))` == `DigestGraph(Evaluate(g))` (a save/loaded
   script computes the same thing).

Keep S1–S4 assertions green (append-only — S1 `0x0e5b8ec26f0d8730`, S2 `0x670cf80b235bdafd`, S3
`0xd5735423148033cc`, S4 `0x670cf80b235bdafd` unchanged).

## Cross-platform proof (cheap loop — NO render-bake)
Controller `scp`s `engine/flow/flow.h` + `engine/net/session.h` + `tests/flow_test.cpp` (+ `tests/test_main.h`
+ `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/flow_test.cpp -o /tmp/flow && /tmp/flow`, confirming ALL assertions PASS with IDENTICAL pinned digests.
(Local Windows clang is the fast pre-check.) NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/flow/flow.h` (add `SnapshotState`/`RestoreState`/`PutU32`/`GetU32`[reuse if S3
  added it]/`SerializeGraph`/`DeserializeGraph` [+ optional state serialize] below S4; do NOT modify S1–S4).
  Header stays SELF-CONTAINED: only `<cstddef>/<cstdint>/<vector>` + `net/session.h`. NO new includes. Do
  NOT modify `net/session.h` (reuse `RollbackSession`/`RunWithTransport`/`ScriptedTransport` read-only).
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<random>`/`<unordered_*>`/`<map>`/`<functional>`/
  `std::hash`/`<algorithm>`. Serialization is hand-LE field-by-field — NEVER memcpy a host struct.
- `tests/flow_test.cpp` stays self-contained; APPEND the S5 assertions + the rollback graph fixture. Keep
  S1–S4 green.
- Branch `fix-flow-s5`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target flow_test'`
  then run `flow_test` and confirm ALL assertions (S1–S5) PASS, exit 0. ALSO local clang standalone,
  identical digests. First run: pin.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `flow_test` builds + PASSES on Windows with
  every assertion green (esp. rollback==authority + didRollback + incomplete-restore-diverges + the
  serialized round-trip), and the local clang standalone passes with identical digests. Report: commit hash,
  full test output (printed digests + PASS lines), the pinned digests, confirmation S1–S4 digests unchanged,
  confirmation the header is still self-contained (list `#include`s), the rollback graph + how you forced
  the mispredict, and the local-clang result. Commit message via temp file + `git commit -F` (Bash heredoc).
  Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S6 — the OPTIONAL render capstone, then closes out
  flagship / issue #24 with the ARCHITECTURE.md section.)
