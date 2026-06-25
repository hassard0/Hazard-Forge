# Slice WFC-S2 — Min-entropy cell selection + weighted collapse (Flagship #29 WFC, 2nd/6)

S1 built the domain grid + adjacency rule-set + integer AC-3 `Propagate`. S2 adds the OBSERVE half of
wave-function-collapse: pick the next cell to decide by **integer min-entropy**, **collapse** it to a
single tile by a **seeded weighted draw**, then propagate. This is where real WFC classically goes
non-deterministic — float Shannon entropy + RNG + hash-ordered cell iteration. S2 neutralizes all three
with a pure-integer popcount-entropy surrogate, fixed tie-breaking, and the engine's proven integer hash.

Pure-CPU INTEGER, append-only to `engine/wfc/wfc.h` (do NOT modify S1's code — ADD below it). NO render.
Golden = pinned `net::DigestBytes` over the grid after a fixed number of collapses, identical
Windows/MSVC + Mac/clang via the standalone clang compile.

## Keep wfc.h SELF-CONTAINED (do NOT include pcg.h)
`pcg.h`'s `PcgRand01`/`PcgRandRange` return `fx` (Q16.16) and `pcg.h` transitively pulls in
`particles.h` + the fixed-point math chain — including it would break wfc.h's self-containment and the
cheap standalone-clang proof. The codebase's established pattern is to COPY the proven pure-`uint32` hash
ops verbatim (pcg.h itself says it copies `particles.h::ParticleHash` "verbatim ops"). So S2 copies the
pure-`uint32` `PcgHash` into wfc.h — same constants, same ops → the same stream — citing
`engine/pcg/pcg.h:42-48` as canonical. No `fx`, no new include.

## Append to engine/wfc/wfc.h (below S1's code, in hf::wfc)

1. **`WfcHash`** — the integer PRNG, copied VERBATIM from `engine/pcg/pcg.h:42-48` (pure `uint32`, no
   float/clock/RNG). Keep the exact constants:
   ```cpp
   // Integer hash — copied verbatim from engine/pcg/pcg.h::PcgHash (which copies particles.h::ParticleHash):
   // the SAME pure-uint32 ops/constants, so the stream is identical. Kept inline here to preserve wfc.h's
   // self-containment (pcg.h pulls in the fx/particles chain). Deterministic + cross-vendor identical.
   inline uint32_t WfcHash(uint32_t seed, uint32_t index) {
       uint32_t h = seed * 2654435761u;
       h ^= (index + 0x9E3779B9u + (h << 6) + (h >> 2));
       h += index * 0x85EBCA6Bu;
       h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
       return h;
   }
   inline constexpr uint32_t kCollapseSalt = 0x57464301u;  // 'WFC\1' — the collapse stream's salt (distinct layer)
   ```

2. **`PopCount`** — integer set-bit count of a `Domain` (the entropy surrogate). Use C++20 `std::popcount`
   (add `#include <bit>` — it is std, deterministic, cross-vendor) OR a manual loop. Either is bit-exact;
   prefer `std::popcount` for clarity:
   ```cpp
   inline int PopCount(Domain d);   // # tiles still allowed
   ```

3. **`int SumWeight(const TileSet& ts, Domain d)`** — sum of `weight[t]` over the set bits of `d` (ascending
   t). Integer; the entropy tie-breaker.

4. **`SelectCell`** — the min-entropy observer. Among all UNDECIDED cells (`PopCount(domain) > 1`), return
   the cell id of **minimum integer entropy**, defined and tie-broken in this FIXED order:
   - primary: minimum `PopCount(domain)` (fewest remaining tiles);
   - tie → minimum `SumWeight(ts, domain)`;
   - tie → lowest cell id.
   Skip decided cells (`PopCount == 1`) and contradictions (`PopCount == 0`). Return `-1` if no cell is
   undecided (fully collapsed). NO float Shannon entropy, NO hash-set iteration — scan cell ids ascending.
   ```cpp
   inline int32_t SelectCell(const TileSet& ts, const Grid& g);   // -1 if fully collapsed
   ```

5. **`Collapse`** — decide one cell by a seeded WEIGHTED integer draw. Among the tiles set in
   `g.cell[cellId]`, pick one with probability proportional to `weight[t]`, deterministically from the
   seed: `uint32_t total = SumWeight(ts, domain); uint32_t r = WfcHash(seed ^ kCollapseSalt, (uint32_t)cellId)
   % total;` then walk the set tiles ASCENDING accumulating `weight[t]`, choosing the first tile where the
   running sum `> r`. Set `g.cell[cellId] = Domain{1} << chosenTile` (a single-bit domain = decided).
   (Modulo over the integer weight total is fully deterministic + cross-platform; distribution bias is
   irrelevant to determinism. All weights are positive — guaranteed by the tileset.)
   ```cpp
   inline uint32_t Collapse(const TileSet& ts, Grid& g, int32_t cellId, uint32_t seed);  // returns chosen tile
   ```

6. **`ObserveStep`** — one full observe iteration: `SelectCell` → if `-1` return a "done" sentinel; else
   `Collapse(seed)` that cell, seed the worklist with it, and `Propagate`. Returns a small status enum or
   bool pair indicating {done, contradiction, progressed}. (S2 does NOT backtrack on contradiction — that
   is S3; S2's showcase is tuned so the first K observes don't contradict, and the test asserts no
   contradiction occurred.)
   ```cpp
   enum class StepResult { kProgressed, kDone, kContradiction };
   inline StepResult ObserveStep(const TileSet& ts, Grid& g, uint32_t seed);
   ```
   (Use a per-step index for the collapse seed if you want distinct draws per step — but cellId already
   differentiates the WfcHash index, so `WfcHash(seed ^ salt, cellId)` is sufficient and simplest. Keep it
   cellId-indexed so a re-run is bit-identical.)

## The golden (PINNED, cross-platform) — append to tests/wfc_test.cpp
Build the S1 showcase tileset + a fresh 16×16 grid, pick a fixed `seed` (e.g. `0x1234ABCDu`), and run
`ObserveStep` for a FIXED count K (choose K so no contradiction occurs on the permissive terrain-gradient
tileset — e.g. K=12; the implementer confirms by running, reduces K if a contradiction appears, and pins
whatever K is contradiction-free). Print the live digest, then assert:
```
wfc-s2: after K=<K> collapses, grid digest = 0x<...>
PASS wfc-s2: K observe steps all progressed (no contradiction on the showcase)
PASS wfc-s2: collapsed grid digest == pinned uint64 (the cross-platform proof)
PASS wfc-s2: re-running the same seed is bit-identical (deterministic)
PASS wfc-s2: a DIFFERENT seed produces a DIFFERENT digest (the seed drives the result)
PASS wfc-s2: every collapsed cell has PopCount==1 (decided) and its tile is allowed by its neighbors
PASS wfc-s2: SelectCell returns -1 only when no cell has PopCount>1 (observer terminates correctly)
```
Assertions:
1. **NO CONTRADICTION** — all K `ObserveStep`s returned `kProgressed` (none `kContradiction`).
2. **PINNED DIGEST** — `DigestGrid(g)` after K collapses == a hard-pinned `uint64_t` (run once, pin the
   printed value; identical MSVC + clang).
3. **REPLAY-STABLE** — a second independent run from the same seed yields the identical digest.
4. **SEED-DRIVEN** — a different seed (e.g. `0x1234ABCD ^ 0xFFFF`) yields a DIFFERENT digest (the draw is
   genuinely seeded, not constant).
5. **DECIDED + CONSISTENT** — every cell collapsed by the run has `PopCount == 1`, and for each such cell
   its chosen tile is in the allowed mask of each in-bounds neighbor's domain (the collapse + propagate
   left a locally-consistent partial assignment). (Check against neighbors' current domains via
   `AllowedMask`.)
6. **OBSERVER TERMINATION** — a small sanity check that `SelectCell` returns `-1` exactly when all cells
   have `PopCount <= 1` (construct a tiny fully-decided grid and assert `-1`).

Keep S1's assertions green (append-only — S1's pinned digest `0xaaa67b9af6f293c8` unchanged).

## Cross-platform proof (cheap loop — NO render-bake)
Controller `scp`s `engine/wfc/wfc.h` + `engine/net/session.h` + `tests/wfc_test.cpp` (+ `tests/test_main.h`
+ `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/wfc_test.cpp -o /tmp/wfc && /tmp/wfc`, confirming ALL assertions PASS with the IDENTICAL pinned
digests. NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/wfc/wfc.h` (add `WfcHash`/`kCollapseSalt`/`PopCount`/`SumWeight`/`SelectCell`/
  `Collapse`/`ObserveStep`/`StepResult` below S1; do NOT modify S1's types/functions). Header stays
  SELF-CONTAINED: only `<cstdint>/<cstddef>/<vector>` (+ optionally `<bit>` for `std::popcount`) +
  `net/session.h`. Do NOT include `pcg.h`/`fpx.h`/any other header. Do NOT modify `net/session.h` or any
  existing header.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<random>`/`<unordered_*>`/`<functional>`/`std::hash`.
  No hash-ordered containers in any logic path (scan cell ids ascending).
- `tests/wfc_test.cpp` stays self-contained; APPEND the S2 assertions. Keep S1's green.
- Branch `fix-wfc-s2`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target wfc_test'`
  then run `wfc_test` and confirm ALL assertions (S1 + S2) PASS, exit 0. First run: pin K + the digest.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `wfc_test` builds + PASSES on Windows
  with every assertion green, and (if a local clang exists) the standalone clang compile passes. Report:
  commit hash, full test output (printed digests + PASS lines), the pinned K + digest, confirmation S1's
  digest `0xaaa67b9af6f293c8` is unchanged, confirmation the header is still self-contained (list
  `#include`s), and the local-clang result (or none → controller runs Mac). Commit message via temp file
  + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S3 — the full Solve with backtracking.)
