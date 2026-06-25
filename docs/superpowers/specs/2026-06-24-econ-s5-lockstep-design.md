# Slice ECON-S5 — Quest state machine + lockstep/rollback/desync capstone (Flagship #30 ECON, 5th/6 — THE HEADLINE)

This is the flagship's headline. S1–S4 built a complete deterministic economy (ledger, crafting, economy
tick, market). S5 adds the **quest/objective state machine** (integer-condition objectives advanced by the
economy state) and then PROVES the moat: wrap the ENTIRE economy + quest state in `net::Session` and show
two peers fed only the command stream re-derive a **bit-identical** economy + quest state (`RunLockstep`),
a mispredicted command **rolls back** to the bit-identical authority state (`RollbackSession`), and a
divergence is **located** at the exact tick (`net::DesyncDetector`). This is the capability UE5's gameplay
layer (Blueprint logic, replicated actor state, float timers, GAS) structurally cannot offer — its
gameplay/economy/quest state is the canonical non-deterministic glue that cannot lockstep, cannot
deterministically rollback, cannot bit-exactly replay. Still the cheapest proof — pinned `uint64` hashes,
identical MSVC + clang, NO render-bake.

Pure-CPU INTEGER, append-only to `engine/econ/econ.h` (do NOT modify S1–S4 — ADD below). The lockstep/
rollback/desync proof reuses the netcode machinery (`net::Session`/`RunLockstep`/`DigestTrace`/
`RollbackSession`/`StepPredicted`/`ConfirmRemote`/`ScriptedTransport`/`RunWithTransport`/`DesyncDetector`)
VERBATIM in the TEST — NO duplication.

## Append to engine/econ/econ.h (below S4, in hf::econ)

### Part 1 — the quest state machine (flat-index, integer conditions)
```cpp
struct Objective {
    uint32_t entity;     // condition: ledger.At(entity, item) >= threshold
    uint32_t item;
    Qty      threshold;
    int32_t  prereq;     // -1 = no prerequisite; else this objective only ACTIVATES once objective[prereq] is complete
};
struct QuestGraph { std::vector<Objective> objectives; };   // STATIC config (not snapshotted)

enum ObjStatus : uint8_t { kLocked = 0, kActive = 1, kComplete = 2 };
struct QuestState { std::vector<uint8_t> status; };          // parallel to objectives; THIS is snapshotted state
```
- `QuestState MakeQuestState(const QuestGraph& g)` — all objectives `kLocked` except those with `prereq == -1`
  which start `kActive`.
- `void AdvanceQuests(const World& ledger, const QuestGraph& g, QuestState& q)` — ONE pass in FIXED objective
  order: a `kLocked` objective whose `prereq` is `kComplete` (or has no prereq) becomes `kActive`; a `kActive`
  objective whose integer condition holds (`ledger.At(entity,item) >= threshold`, bounds-checked) becomes
  `kComplete`. (Single fixed-order pass per call — a multi-step chain completes across successive ticks, NOT
  within one call; this keeps it deterministic + lockstep-aligned. Document this.) Pure integer, no float.

### Part 2 — the unified snapshot-able state + tagged command
```cpp
struct EconState {            // the net::Session World — ALL mutable state, value-copy snapshot/restore-able
    World      ledger;
    Market     market;
    QuestState quests;
};

enum CmdTag : uint32_t { kTxnCmd = 0, kCraftCmd = 1, kTradeCmd = 2, kTickCmd = 3, kNopCmd = 4 };
struct EconCommand {          // the net::Session Input (a tagged union of the S1-S4 operations)
    uint32_t   tag;
    Command    txn{};         // kTxnCmd  (S1 Add/Remove/Transfer)
    CraftOrder craft{};       // kCraftCmd (S2 — one ApplyRecipe `count` times via DrainCraftQueue of one order)
    TradeOrder trade{};       // kTradeCmd (S4 ExecuteTrade)
    // kTickCmd: run one EconTick(ledger, rules);  kNopCmd: no-op
};
bool operator==(const EconCommand& a, const EconCommand& b);   // REQUIRED — RollbackSession compares Inputs to detect a mispredict
```
(`operator==` must compare all relevant fields by tag — implement a full field compare, or memcmp-free
field equality. It is load-bearing: `ConfirmRemote` uses `appliedRemote[at] != real` to detect a
misprediction.)

### Part 3 — the deterministic step (the net::Session contract)
```cpp
// Apply ONE command to the state (dispatch on tag), using the STATIC config (rules/recipes/graph).
void ApplyEconCommand(EconState& s, const EconCommand& c,
                      const EconRules& rules, const RecipeSet& recipes);
// The net::Session step: apply EVERY command for this tick in ARRAY ORDER, then AdvanceQuests ONCE.
// (Provided as a helper OR built as a capturing lambda in the test — see below. The quest advance runs
// once per tick AFTER the tick's commands, so quests stay lockstep-aligned.)
```
The step the test passes to `net::RunLockstep`/`RollbackSession` is a lambda capturing `rules`/`recipes`/
`graph`:
```cpp
auto step = [&](EconState& w, const std::vector<EconCommand>& cmds, uint32_t /*tick*/) {
    for (const EconCommand& c : cmds) ApplyEconCommand(w, c, rules, recipes);
    AdvanceQuests(w.ledger, graph, w.quests);
};
```
(`ApplyEconCommand` reuses S1–S4 verbatim: `kTxnCmd`→`ApplyCommand`, `kCraftCmd`→`DrainCraftQueue` of one
order, `kTradeCmd`→`ExecuteTrade`, `kTickCmd`→`EconTick`. All deterministic, all integer.)

### Part 4 — the combined digest
```cpp
uint64_t DigestEconState(const EconState& s);   // combine DigestState(ledger,market) + DigestBytes(quests.status), fixed order
```

### Fixtures
`QuestGraph MakeShowcaseQuests()` — a fixed chained quest graph (e.g. obj0: entity0 has ≥1 item2 [craft an
ingot]; obj1 prereq 0: entity0 has ≥1 item3 [craft a tool]; obj2 prereq 1: entity0 has ≥5 item3) so the
chain completes across multiple ticks. `std::vector<EconCommand> MakeShowcaseCommandStream()` — a fixed
mixed stream (txns, crafts, trades, ticks) over the showcase world/recipes/market that drives the economy
AND completes the quest chain. `EconState MakeShowcaseState()` — bundles `MakeShowcaseWorld` + coin seed +
`MakeShowcaseMarket` + `MakeQuestState`.

## The goldens (PINNED, cross-platform) — append to tests/econ_test.cpp
Use `hf::net::` directly (the test already has `net/session.h` transitively; include it if needed). Build
the showcase state/graph/recipes/rules/market and a fixed `InputRing<EconCommand>` from the command stream
(one or more commands per tick).

### Part A — lockstep over net::Session (the determinism invariant)
```
econ-s5: lockstep final econ-state digest = 0x<...>  (T ticks)
PASS econ-s5: two peers from the same command stream have EQUAL DigestEconState at EVERY tick (lockstep invariant)
PASS econ-s5: net::RunLockstep final digest == pinned uint64 (two peers re-derive the bit-identical economy+quests)
PASS econ-s5: the quest chain COMPLETES deterministically (final QuestState == all kComplete, pinned)
```
1. **LOCKSTEP INVARIANT** — two independent `net::Session<EconState,EconCommand>` from the same init + ring,
   advanced in lockstep, have EQUAL `DigestEconState` at EVERY tick (use `net::DigestTrace` twice → identical
   traces; or two `Session`s advanced tick-by-tick). 2. **PINNED FINAL** — `net::RunLockstep<EconState,
   EconCommand>(MakeShowcaseState(), ring, T, step, DigestEconState)` == a hard-pinned `uint64_t`. 3. **QUEST
   COMPLETION** — the final `QuestState.status` is all `kComplete` (the chain finished), pinned.

### Part B — rollback correctness (the make-or-break, the GGPO proof for gameplay state)
Model two peers: `local[t]` = this peer's command per tick, `remote[t]` = the other peer's; drive a
`net::RollbackSession<EconState,EconCommand>` via `net::RunWithTransport` with a `ScriptedTransport` that
DELAYS at least one remote command past its origin tick (so the session predicts wrong and must roll back).
```
econ-s5: rollback authority digest = 0x<...>, didRollback = <b>
PASS econ-s5: a mispredicted command rolls back to the BIT-IDENTICAL authority economy+quest state
PASS econ-s5: rollback actually fired (didRollback == true)
PASS econ-s5: the adversarial (delayed/reordered) schedule converges to the SAME pinned digest as the clean authority run
```
4. **ROLLBACK == AUTHORITY** — `RunWithTransport` final `DigestEconState(s.world)` == the clean authority
   digest (a `net::RunLockstep` over the TRUE combined per-tick inputs `{local[t], remote[t]}`) == a pinned
   `uint64_t`. 5. **DIDROLLBACK** — `s.didRollback == true` (a real misprediction fired — pick a remote
   command stream where the delayed command differs from the prediction so the rollback is genuine). 6.
   **ADVERSARIAL CONVERGENCE** — same as (4) under the scripted delay/reorder (the netcode NS4 proof applied
   to gameplay state).

### Part C — desync localization (the NS5 detector over gameplay state)
Two command streams identical except one tick `K` differs → traces match for `t < K`, diverge at `K`;
`net::DesyncDetector` latches the exact tick.
```
econ-s5: desync injected at tick K=<k>, detector latched tick=<k>
PASS econ-s5: identical command streams report NO desync (clean)
PASS econ-s5: a one-command divergence is LOCATED at the exact tick K (net::DesyncDetector), traces match for t<K
```
7. **CLEAN** — `DesyncDetector` over (traceA vs traceA) reports no desync. 8. **LOCATED** — traceA vs traceB
   (tick K's command changed) → `d.desynced && d.desyncTick == K`, traces equal for `t < K`. Pin K.

Keep S1–S4 assertions green (append-only — S1 `0xaa712207f7663e03`, S2 `0x95147ff9dabbfd13`, S3
`0xca63394d5a6a9a2b`, S4 `0x087296381931e8b0`/`0x57a1cd074f19817a` unchanged).

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/econ/econ.h` + `engine/net/session.h` + `tests/econ_test.cpp` (+ `tests/test_main.h`
+ `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/econ_test.cpp -o /tmp/econ && /tmp/econ`, confirming ALL assertions PASS with IDENTICAL pinned digests
+ latched tick. (Local Windows clang is the fast pre-check.) NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/econ/econ.h` (add `Objective`/`QuestGraph`/`ObjStatus`/`QuestState`/`MakeQuestState`/
  `AdvanceQuests`/`EconState`/`CmdTag`/`EconCommand`/`operator==`/`ApplyEconCommand`/`DigestEconState` +
  fixtures below S4; do NOT modify S1–S4 — reuse `ApplyCommand`/`DrainCraftQueue`/`ExecuteTrade`/`EconTick`/
  `DigestState`/`DigestWorld`). Header stays SELF-CONTAINED: only `<cstdint>/<cstddef>/<vector>` +
  `net/session.h`. Do NOT include `pcg.h`/`fpx.h`/`<algorithm>`/any other header. Do NOT modify
  `net/session.h` (reuse `Session`/`RollbackSession`/`DigestTrace`/`DesyncDetector`/`ScriptedTransport`/
  `RunWithTransport` read-only) or any existing header.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<random>`/`<unordered_*>`/`<map>`/`<functional>`/`std::hash`.
- `tests/econ_test.cpp` stays self-contained; APPEND the S5 assertions + fixtures (it may use the `net::`
  templates from the already-included `net/session.h`). Keep S1–S4 green.
- Branch `fix-econ-s5`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target econ_test'`
  then run `econ_test` and confirm ALL assertions (S1–S5) PASS, exit 0. ALSO compile standalone with the
  local clang (`C:\Program Files\LLVM\bin\clang++.exe`) and confirm IDENTICAL pinned digests + latched tick.
  First run: pin the digests + K.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `econ_test` builds + PASSES on Windows with
  every assertion green (esp. the lockstep invariant + rollback==authority + didRollback + desync-located),
  and the local clang standalone passes with identical digests. Report: commit hash, full test output
  (printed digests + latched tick + PASS lines), the pinned digests + T + K, confirmation S1–S4 digests
  unchanged, confirmation the header is still self-contained (list `#include`s — the same 4), the quest
  graph + command stream fixtures, and the local-clang result. Commit message via temp file + `git commit
  -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests + latched
  tick, ff-merges to master + pushes + deletes the branch + advances to S6 — the optional render capstone.)
