# Slice WFC-S3 — Full Solve with deterministic BACKTRACKING (Flagship #29 WFC, 3rd/6 — THE MAKE-OR-BREAK)

S1 built `Propagate`; S2 built `SelectCell`/`Collapse`/`ObserveStep` (observe-without-backtrack). S3 is
the heart of the flagship: a **complete WFC solver** — observe → propagate, and on a CONTRADICTION
(empty domain), **backtrack** to the last decision, remove the just-tried tile, and retry — until the
grid is fully collapsed or proven unsolvable. This is exactly where real WFC implementations go
non-deterministic (ordered-set iteration, RNG re-seeding, float entropy on retry); S3 makes the entire
backtracking search **bit-identical CPU/Vulkan/Metal** by pinning every ordering and snapshotting domains
by byte-exact value-copy. The golden is the digest of a **fully-collapsed, contradiction-free tilemap** —
a complete deterministic level — plus a proof that backtracking actually fired.

Pure-CPU INTEGER, append-only to `engine/wfc/wfc.h` (do NOT modify S1/S2 — ADD below). NO render. Golden
= pinned `net::DigestBytes`, identical Windows/MSVC + Mac/clang via the standalone clang compile.

## Append to engine/wfc/wfc.h (below S2, in hf::wfc)

The decision-stack backtracking solver. The pattern mirrors `fpx::SnapshotWorld`/`RestoreWorld` +
`net::RollbackSession::snaps`/`didRollback` (value-copy snapshot, restore, a fired-flag), applied to the
WFC domain array.

1. **`Decision`** — one entry on the backtracking stack:
   ```cpp
   struct Decision {
       int32_t              cellId;        // the cell this decision collapsed
       Domain               triedMask;     // tiles ALREADY tried at this cell (bit set => tried+failed/active)
       std::vector<Domain>  snapshot;      // a byte-exact value-copy of g.cell AS OF just BEFORE this collapse
   };
   ```
   (The snapshot is the whole `g.cell` vector copied by value — the `RestoreWorld` discipline. It restores
   bit-exact, which is the determinism guarantee; a partial/delta snapshot would be an optimization for
   later, not S3.)

2. **`SolveResult`** — the outcome + provenance:
   ```cpp
   struct SolveResult {
       bool     solved       = false;  // true iff fully collapsed with no contradiction
       bool     didBacktrack = false;  // true iff >=1 backtrack fired (the RollbackSession::didRollback twin)
       uint32_t steps        = 0;      // observe+backtrack iterations consumed (the maxSteps guard counter)
       uint32_t backtracks   = 0;      // how many times we popped/retried
   };
   ```

3. **`Solve`** — the full loop:
   ```cpp
   inline SolveResult Solve(const TileSet& ts, Grid& g, uint32_t seed, uint32_t maxSteps);
   ```
   Algorithm (every ordering FIXED):
   - Loop up to `maxSteps` (the `nav` bounded-iteration guard — a deterministic ceiling so a pathological
     tileset returns `solved=false` instead of hanging):
     - `c = SelectCell(ts, g)`. If `c == -1` → fully collapsed → `solved=true`, return.
     - **Decide:** push a `Decision{ cellId=c, triedMask=0, snapshot=g.cell }` (value-copy the WHOLE grid
       BEFORE collapsing). Then pick a tile to try via the **S2 weighted draw restricted to not-yet-tried
       tiles** — see the trial rule below — set `g.cell[c] = (1<<tile)`, mark `triedMask |= (1<<tile)`,
       seed the worklist with `c`, and `Propagate`.
     - **On contradiction** (`Propagate` returned false, OR `SelectCell` later finds a 0-domain — but
       `Propagate`'s bool is the signal): **backtrack**. `++backtracks; didBacktrack=true`. Pop the
       behavior: RESTORE `g.cell = top.snapshot` (bit-exact), then try the NEXT untried tile at
       `top.cellId`: the candidate set is `originalDomainAtDecision & ~top.triedMask` (the snapshot still
       holds that cell's pre-collapse domain → `top.snapshot[top.cellId]` is the original domain). If that
       candidate set is non-empty, pick the next tile from it (same weighted/ascending rule), set it, mark
       `triedMask`, propagate again (staying at this decision — do NOT push a new Decision; we're retrying
       the same cell). If the candidate set is EMPTY (all tiles at this cell exhausted), **pop this
       Decision entirely** and backtrack to the PARENT decision (restore its snapshot and try ITS next
       tile). If the stack becomes empty with no candidates → `solved=false` (unsolvable), return.
   - If the loop hits `maxSteps` without fully collapsing → `solved=false` (deterministic give-up).
   - `steps` = iterations consumed.

   **The tile-trial rule (load-bearing for determinism):** at a decision, the candidate domain is
   `cand = snapshotDomainOfCell & ~triedMask` (untried surviving tiles). Pick ONE deterministically — use
   the S2 weighted draw over `cand`: `total = SumWeight(ts, cand); r = WfcHash(seed ^ kCollapseSalt,
   (uint32_t)cellId * someStride + triedCount) % total; walk cand's set bits ASCENDING, choose first where
   running weight > r`. Vary the WfcHash index by the number of tiles already tried at this cell (e.g.
   `cellId * 64u + PopCount(triedMask)`) so successive retries at the same cell draw DIFFERENT tiles
   deterministically (not the same one forever). Keep this index formula FIXED and pinned. Decided.

   **Determinism rules (the make-or-break — pin ALL of these):**
   - Cell selection = `SelectCell` (S2: min popcount, tie min SumWeight, tie lowest id) — already pinned.
   - Tile trial = the weighted draw over untried tiles, indexed by `(cellId, triedCount)` — pinned.
   - Backtrack = LIFO over a `std::vector<Decision>` (push_back/pop_back) — pinned order.
   - Snapshot = whole-`g.cell` value-copy; restore = assign back — byte-exact round-trip (the
     `RestoreWorld` guarantee).
   - NO `std::unordered_*`, NO `std::hash`, NO pointer-keyed/float anything in the solve path.
   - `maxSteps` is an explicit parameter (deterministic ceiling), NOT a wall-clock.

   (Helper allowed: factor a small `inline int PickTileFromDomain(ts, Domain cand, seed, idx)` used by both
   S2's `Collapse` path conceptually and S3 — but do NOT modify S2's `Collapse`; add a NEW helper
   append-only. Or inline it in `Solve`. Implementer's choice; keep it pinned.)

## The golden (PINNED, cross-platform) — append to tests/wfc_test.cpp
### Part A — a full solve on the permissive showcase tileset
Build `MakeShowcaseTileSet()` + a fresh 16×16 grid, `Solve(ts, g, seed=0x1234ABCDu, maxSteps=100000)`.
```
wfc-s3: solve -> solved=<b> didBacktrack=<b> steps=<n> backtracks=<n>, digest=0x<...>
PASS wfc-s3: Solve fully collapsed the grid (solved==true, every cell PopCount==1)
PASS wfc-s3: fully-collapsed grid digest == pinned uint64 (the cross-platform proof)
PASS wfc-s3: the collapsed assignment is GLOBALLY consistent (every adjacent pair satisfies the rules)
PASS wfc-s3: re-running the same seed is bit-identical (solved/steps/backtracks/digest all identical)
PASS wfc-s3: a different seed yields a different (but still valid+consistent) full collapse
```
Assertions:
1. **SOLVED** — `solved == true`; every cell has `PopCount == 1`.
2. **PINNED DIGEST** — `DigestGrid(g)` == a hard-pinned `uint64_t` (run once, pin; identical MSVC + clang).
3. **GLOBAL CONSISTENCY** — for every cell and every in-bounds neighbor, the neighbor's single tile is in
   `AllowedMask(ts, thisTile, dir)` (a full validity sweep — the solved tilemap obeys EVERY adjacency rule,
   not just locally). This is the real proof the solver produced a correct level.
4. **REPLAY-STABLE** — a second `Solve` from the same seed reproduces identical `solved`/`steps`/
   `backtracks`/digest (the WHOLE search is deterministic, including the backtracking path).
5. **SEED-DRIVEN** — a different seed solves to a DIFFERENT digest that is ALSO fully-collapsed + globally
   consistent (different valid level, same correctness).

### Part B — backtracking actually fires (the RollbackSession::didRollback twin)
Craft (or find a seed for) a scenario that FORCES at least one contradiction+backtrack, and assert it.
Two acceptable ways — pick whichever is cleaner and pin it:
- (a) A **constrained tileset** — a deliberately tight rule-set (e.g. a "stripes/pipes" set where greedy
  collapse frequently paints itself into a corner) on a small grid, where `Solve` reports
  `didBacktrack==true` (`backtracks >= 1`) yet still `solved==true` with a globally-consistent result; OR
- (b) **pre-constrain** the showcase grid (pin a few opposite-corner cells to incompatible tiles via the
  gradient — e.g. force water at one corner and rock at the adjacent cell, which the gradient forbids
  directly) so propagation hits a contradiction and the solver must back out.
```
PASS wfc-s3: backtracking FIRES on the constrained scenario (didBacktrack==true, backtracks>=1) and still solves consistently
PASS wfc-s3: the constrained solve is also bit-identical on re-run (deterministic backtracking path) + pinned digest
PASS wfc-s3: an UNSOLVABLE scenario returns solved==false deterministically (e.g. a cell pre-pinned to 0 domain or contradictory pins), no hang
```
Assertions:
6. **BACKTRACK FIRED** — the constrained scenario yields `didBacktrack==true && backtracks>=1 && solved==
   true`, globally consistent, with a pinned digest; re-run bit-identical.
7. **UNSOLVABLE IS DETERMINISTIC** — a genuinely unsolvable setup (e.g. two adjacent cells pre-pinned to
   tiles that the rules forbid as neighbors, with no alternative) returns `solved==false` within `maxSteps`
   (no hang), reproducibly.

Keep S1+S2 assertions green (append-only — S1 `0xaaa67b9af6f293c8`, S2 `0x4c9e67d356f4b920` unchanged).

## Cross-platform proof (cheap loop — NO render-bake)
Controller `scp`s `engine/wfc/wfc.h` + `engine/net/session.h` + `tests/wfc_test.cpp` (+ `tests/test_main.h`
+ `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/wfc_test.cpp -o /tmp/wfc && /tmp/wfc`, confirming ALL assertions PASS with IDENTICAL pinned digests.
NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/wfc/wfc.h` (add `Decision`/`SolveResult`/`Solve` [+ optional `PickTileFromDomain`]
  below S2; do NOT modify S1/S2 types/functions). Header stays SELF-CONTAINED: only `<bit>/<cstdint>/
  <cstddef>/<vector>` + `net/session.h`. Do NOT include `pcg.h`/`fpx.h`/any other header. Do NOT modify
  `net/session.h` or any existing header.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<random>`/`<unordered_*>`/`<functional>`/`std::hash`. No
  hash-ordered containers in the solve path.
- `tests/wfc_test.cpp` stays self-contained; APPEND the S3 assertions. Keep S1+S2 green.
- Branch `fix-wfc-s3`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target wfc_test'`
  then run `wfc_test` and confirm ALL assertions (S1+S2+S3) PASS, exit 0. First run: pin the digests.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `wfc_test` builds + PASSES on Windows
  with every assertion green (esp. the global-consistency sweep + the backtrack-fired proof + the
  unsolvable-deterministic proof), and (if a local clang exists) the standalone clang compile passes.
  Report: commit hash, full test output (printed solve stats + digests + PASS lines), the pinned digests +
  the maxSteps/seed used, confirmation S1/S2 digests are unchanged, confirmation the header is still
  self-contained (list `#include`s), and the local-clang result (or none → controller runs Mac). Commit
  message via temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S4 — Wang/overlapping model + region constraints.)
