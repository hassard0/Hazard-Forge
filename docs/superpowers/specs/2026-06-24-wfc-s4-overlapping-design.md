# Slice WFC-S4 — Overlapping/Wang model (learn rules from a sample) + region constraints (Flagship #29 WFC, 4th/6)

S1–S3 take a HAND-AUTHORED adjacency rule-set and solve it. S4 makes the rule-set itself **learned from a
sample tilemap** (the "overlapping/Wang" model that gives WFC its reputation — feed it a small example, it
generates more that looks like it) and adds **region pre-constraints** (the caller pins certain cells
before the solve — entrance/exit/border tiles, the authoring knob). Both stay strict-integer and pinned.
The headline stays intact: rules derived deterministically from a sample → the same generated levels
bit-identically on every machine, which a float-hash WFC cannot guarantee.

Pure-CPU INTEGER, append-only to `engine/wfc/wfc.h` (do NOT modify S1–S3 — ADD below). NO render. Golden
= pinned `net::DigestBytes`, identical Windows/MSVC + Mac/clang via the standalone clang compile.

## Append to engine/wfc/wfc.h (below S3, in hf::wfc)

### Part 1 — Adjacency LEARNED from a sample tilemap (the Wang/overlapping model)
Given a small sample tilemap (a `w*h` array of tile ids), DERIVE the `TileSet` (adjacency masks + weights)
by scanning every adjacent pair: if tile `a` sits immediately on the `dir` side of tile `b` ANYWHERE in
the sample, then `a` is permitted on the `dir` side of `b`. Weight[t] = the integer COUNT of tile `t`'s
occurrences in the sample (frequency-as-weight — common tiles collapse more often). This is pure integer
counting; no float, no hashing of the grid.

```cpp
struct SampleMap {
    int32_t              w = 0, h = 0;
    std::vector<int32_t> tile;        // [w*h]: tile[z*w+x] = the tile id at that sample cell (0..tileCount-1)
};

// Learn a TileSet from a sample: tileCount = max tile id + 1; weight[t] = count of t in the sample;
// allowed[t*4+dir] starts 0 and gets bit `u` set whenever tile u is found on the `dir` side of a cell
// holding tile t (scan every in-bounds adjacent pair, both directions, fixed cell order ascending).
inline TileSet LearnTileSet(const SampleMap& s);
```
Determinism: scan sample cells in ascending id; for each, for each in-bounds neighbor in fixed dir order
`{kRight,kUp,kLeft,kDown}`, OR the neighbor's tile bit into `allowed[thisTile*4+dir]`. The learned rule is
automatically adjacency-symmetric (every pair is observed from both sides), so `IsSymmetric(LearnTileSet(s))`
must hold — assert it. NO `std::hash`, NO `std::unordered_*`, NO float anywhere.

(If a stricter overlapping-N×N pattern model is trivial to add on top, the implementer MAY add an
`LearnPatternTileSet(sample, N)` that treats each distinct N×N pattern as a "super-tile" and derives
pattern-adjacency — BUT keep tileCount ≤ 64, and dedup/order patterns by FIRST-OCCURRENCE INDEX using a
sorted/indexed `std::vector` (NEVER a hash-set walk, whose iteration order is non-deterministic). This is
OPTIONAL; the simple per-tile Wang model above is the required deliverable. If pattern count would exceed
64, fall back to or pin the simple model. Do NOT use `std::hash` for pattern identity — compare pattern
bytes directly, or hash with `WfcHash` over the pattern's tile bytes for a fast pre-check then confirm by
byte compare.)

### Part 2 — Region pre-constraints (authoring)
Let the caller pin/restrict cells BEFORE solving, then propagate the consequences:
```cpp
// Restrict a cell's domain to exactly one tile (a hard pin). Does NOT propagate by itself.
inline void PinCell(Grid& g, int32_t x, int32_t z, uint32_t tile);
// Restrict a cell's domain to a caller-supplied mask (a soft region constraint — e.g. "only floor tiles here").
inline void ConstrainCell(Grid& g, int32_t x, int32_t z, Domain allowedHere);
// Apply all pins/constraints already written into g, then Propagate from every constrained cell.
// Returns false if the constraints are immediately contradictory (some domain became 0). Seeds the worklist
// with every cell whose domain != all-tiles.
inline bool ApplyConstraints(const TileSet& ts, Grid& g);
```
`Solve` already starts from whatever `g` contains, so the flow is: `MakeShowcaseGrid` → `PinCell`/
`ConstrainCell` some cells → `ApplyConstraints` (propagate the pins) → `Solve` (fills the rest honoring
them). The solved grid must (a) keep every pinned cell at its pinned tile and (b) be globally consistent.
(`Solve` needs no change — pins are just a pre-narrowed starting grid; its snapshots/backtracking already
respect them because a pinned single-bit domain has no other tile to try.)

## The goldens (PINNED, cross-platform) — append to tests/wfc_test.cpp
### Part A — learned rules reproduce + solve
Build a small fixed `SampleMap` (integer literals — e.g. a 6×6 hand-typed terrain patch using the S1 tile
ids 0=water,1=sand,2=grass,3=rock arranged as a believable gradient island), `LearnTileSet` it, and solve.
```
wfc-s4: learned tileset: tiles=<n>, weights=[...], adjacency digest=0x<...>
PASS wfc-s4: LearnTileSet is adjacency-symmetric (learned rules are AC-3 sound)
PASS wfc-s4: learned adjacency-mask digest == pinned uint64 (rules derived deterministically from the sample)
PASS wfc-s4: learned weights == per-tile occurrence counts in the sample
PASS wfc-s4: Solve(learnedTileSet, 16x16, seed) fully collapses + globally consistent, digest == pinned
PASS wfc-s4: every tile in the solved output also appears in the sample (no tile invented out of nothing)
```
Assertions:
1. **SYMMETRIC** — `IsSymmetric(LearnTileSet(sample))`.
2. **ADJACENCY DIGEST PINNED** — `net::DigestBytes` over the learned `allowed` vector == a pinned `uint64_t`
   (the rules are a deterministic function of the sample; identical MSVC + clang).
3. **WEIGHTS = COUNTS** — `weight[t]` equals the literal occurrence count of tile `t` in the sample (compute
   the expected counts in the test).
4. **SOLVE PINNED + CONSISTENT** — `Solve(learned, 16×16, seed)` fully collapses, is globally consistent
   (the S3 validity sweep), and its grid digest == a pinned `uint64_t`.
5. **NO INVENTED TILES** — every tile id appearing in the solved grid also appears in the sample (the
   generator only uses observed tiles).

### Part B — region pre-constraints honored
Take the learned (or S1 showcase) tileset + a fresh 16×16 grid; `PinCell` a few cells (e.g. water at one
corner, rock at the opposite corner — both consistent with the gradient), `ConstrainCell` a border row to
a sand-only mask, `ApplyConstraints`, then `Solve`.
```
wfc-s4: constrained solve -> solved=<b>, pinned cells honored=<b>, digest=0x<...>
PASS wfc-s4: ApplyConstraints propagated the pins without contradiction
PASS wfc-s4: Solve with pins fully collapses + globally consistent, digest == pinned
PASS wfc-s4: every pinned cell holds EXACTLY its pinned tile in the solved output
PASS wfc-s4: the sand-constrained border row holds only sand in the solved output
PASS wfc-s4: re-running the constrained solve is bit-identical (deterministic)
PASS wfc-s4: contradictory pins (two adjacent cells pinned to forbidden neighbors) -> ApplyConstraints returns false (or Solve solved==false), deterministically
```
Assertions:
6. **PINS PROPAGATE** — `ApplyConstraints` returns true (no immediate contradiction) on the consistent pins.
7. **SOLVE PINNED + CONSISTENT** — `Solve` fully collapses, globally consistent, grid digest == a pinned
   `uint64_t`.
8. **PINS HONORED** — every pinned cell holds exactly its pinned tile in the output; the sand-constrained
   border holds only sand.
9. **REPLAY-STABLE** — re-running the constrained solve from the same seed is bit-identical.
10. **CONTRADICTORY PINS DETERMINISTIC** — pinning two adjacent cells to a forbidden pair makes
    `ApplyConstraints` return false (or `Solve` return `solved==false`) reproducibly, no hang.

Keep S1+S2+S3 assertions green (append-only — S1 `0xaaa67b9af6f293c8`, S2 `0x4c9e67d356f4b920`, S3
`0x0fffd74f7e8419ac` + `0x8adb136f5b4c690a` unchanged).

## Cross-platform proof (cheap loop — NO render-bake)
Controller `scp`s `engine/wfc/wfc.h` + `engine/net/session.h` + `tests/wfc_test.cpp` (+ `tests/test_main.h`
+ `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/wfc_test.cpp -o /tmp/wfc && /tmp/wfc`, confirming ALL assertions PASS with IDENTICAL pinned digests.
NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/wfc/wfc.h` (add `SampleMap`/`LearnTileSet` [+ optional `LearnPatternTileSet`] +
  `PinCell`/`ConstrainCell`/`ApplyConstraints` below S3; do NOT modify S1–S3 types/functions). Header stays
  SELF-CONTAINED: only `<bit>/<cstdint>/<cstddef>/<vector>` + `net/session.h`. Do NOT include `pcg.h`/
  `fpx.h`/any other header. Do NOT modify `net/session.h` or any existing header.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<random>`/`<unordered_*>`/`<functional>`/`std::hash`. No
  hash-ordered containers in any logic path (sample scan + pattern dedup ascending/first-occurrence only).
- `tests/wfc_test.cpp` stays self-contained; APPEND the S4 assertions. Keep S1–S3 green.
- Branch `fix-wfc-s4`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target wfc_test'`
  then run `wfc_test` and confirm ALL assertions (S1–S4) PASS, exit 0. First run: pin the digests.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `wfc_test` builds + PASSES on Windows
  with every assertion green (esp. the learned-adjacency digest + the pins-honored checks), and (if a local
  clang exists) the standalone clang compile passes. Report: commit hash, full test output (printed
  learned-tileset stats + digests + PASS lines), the pinned digests + the sample/seed used, confirmation
  S1–S3 digests are unchanged, confirmation the header is still self-contained (list `#include`s), and the
  local-clang result (or none → controller runs Mac). Commit message via temp file + `git commit -F` (Bash
  heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S5 — the seed-deterministic lockstep headline.)
