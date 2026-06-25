# Slice WFC-S5 — Seed-deterministic lockstep + desync localization (Flagship #29 WFC, 5th/6 — THE BANNER SLICE)

This is the flagship's headline. S1–S4 built a complete, deterministic WFC solver (propagate → observe →
backtrack → learned rules + constraints). S5 PROVES the moat claim: a WFC level is a **pure function of
the seed**, so two peers fed only the seed re-derive the **byte-identical** level — and a STREAM of
seed-driven generations runs on the existing `net::Session` engine, is **lockstep-identical at every step**,
**replay-able**, and a divergence is **located** at the exact step (the NS5 desync detector). This is the
capability UE5's PCG framework and every float-entropy WFC plugin cannot offer: their `mt19937` +
float-Shannon + hash-set-ordered generation differs across compilers/platforms and cannot be lockstepped
or replayed. Still the cheapest proof — pinned `uint64` hashes, identical MSVC + clang, NO render-bake.

Pure-CPU INTEGER, append-only to `engine/wfc/wfc.h` (do NOT modify S1–S4 — ADD below). The lockstep/
desync proof reuses the netcode machinery (`net::Session`/`Advance`/`RunLockstep`/`DigestTrace`/
`DesyncDetector`/`ChecksumPacket`/`RecordLocal`/`IngestRemote`) VERBATIM in the TEST — NO solver
duplication.

## Append to engine/wfc/wfc.h (below S4, in hf::wfc)
1. **`MakeFullGrid`** — the generic all-superposed grid for an arbitrary tile count (the generic twin of
   S1's hardcoded-4-tile `MakeShowcaseGrid`; append-only, do NOT modify `MakeShowcaseGrid`):
   ```cpp
   inline Grid MakeFullGrid(int32_t w, int32_t h, uint32_t tileCount) {
       Grid g; g.w = w; g.h = h;
       const Domain all = (tileCount < 64u) ? ((Domain{1} << tileCount) - 1) : ~Domain{0};
       g.cell.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), all);
       return g;
   }
   ```
2. **`Generate`** — the pure seed→level function (the headline API). Builds a full grid for `ts.tileCount`,
   solves it, returns the solved grid. Deterministic of `(ts, seed, w, h, maxSteps)` alone — no hidden
   state, no clock, no global RNG:
   ```cpp
   inline Grid Generate(const TileSet& ts, uint32_t seed, int32_t w, int32_t h, uint32_t maxSteps) {
       Grid g = MakeFullGrid(w, h, ts.tileCount);
       Solve(ts, g, seed, maxSteps);     // S3 — fills g (or leaves a partial grid if unsolvable)
       return g;
   }
   ```
   (Generate returns the grid whether or not Solve fully collapsed; callers that need the status can call
   `Solve` directly. Optionally also add `inline uint64_t GenerateDigest(const TileSet& ts, uint32_t seed,
   int32_t w, int32_t h, uint32_t maxSteps) { return DigestGrid(Generate(ts, seed, w, h, maxSteps)); }`.)

That is the ENTIRE header change — the rest of S5 is the proof in the test, composing `Generate` with the
netcode session engine.

## The goldens (PINNED, cross-platform) — append to tests/wfc_test.cpp
The test already (transitively) has `net/session.h`; use `hf::net` directly. Pick a fixed tileset (the S1
showcase OR the S4 learned one — pick one, keep it fixed), grid size (e.g. 12×12 to keep the stream cheap),
`maxSteps`, and a base seed.

### Part A — pure-seed determinism (the core headline)
```
wfc-s5: Generate(seed) digest = 0x<...>
PASS wfc-s5: two independent Generate(ts, seed) calls are byte-identical (pure function of the seed)
PASS wfc-s5: Generate(ts, seed) digest == pinned uint64 (cross-platform: a peer re-derives the level)
PASS wfc-s5: a different seed generates a different (still fully-collapsed, globally-consistent) level
PASS wfc-s5: a one-bit tileset rule change generates a DIFFERENT level (rules are load-bearing at generation)
```
1. `DigestGrid(Generate(ts, seed, ...))` == a hard-pinned `uint64_t` (run once, pin; identical MSVC+clang).
2. Two independent `Generate(ts, seed, ...)` calls → identical digest (pure function).
3. A different seed → a different digest that is ALSO fully collapsed + globally consistent (reuse the S3
   validity sweep).
4. A one-adjacency-bit-flipped tileset → a different digest (the learned/authored rules drive output).

### Part B — lockstep over the netcode session engine (THE COMPOSITION)
Model a STREAM of seed-driven generations as a `net::Session<Grid, uint32_t>`: the per-tick input is one
seed, the step regenerates the world's grid from it, the digest is `DigestGrid`. This runs the WFC
generator ON the existing rollback-netcode engine (genuine reuse, no new machinery).
```cpp
// In the test (lambdas capture ts, W, H, MAXSTEPS — net::Session takes template callables, no <functional>):
auto step = [&](Grid& world, const std::vector<uint32_t>& seeds, uint32_t /*tick*/) {
    if (!seeds.empty()) world = hf::wfc::Generate(ts, seeds.back(), W, H, MAXSTEPS);
};
auto digest = [&](const Grid& world) { return hf::wfc::DigestGrid(world); };
// Build an InputRing<uint32_t> with one seed per tick: ring.AddInput(t, baseSeed + t) for t in [0, T).
```
```
wfc-s5: lockstep stream final digest = 0x<...> (T levels)
PASS wfc-s5: net::RunLockstep over a T-seed generation stream == pinned uint64
PASS wfc-s5: two peers running the same seed stream produce IDENTICAL net::DigestTrace at EVERY tick (lockstep invariant)
PASS wfc-s5: a replay of the same stream reproduces the identical final digest (replay-able)
```
5. `net::RunLockstep<Grid, uint32_t>(Grid{}, ring, T, step, digest)` == a pinned `uint64_t` (the stream's
   final digest).
6. Two independent `net::DigestTrace(Grid{}, ring, T, step, digest)` calls → IDENTICAL traces (every tick's
   per-level digest matches — the lockstep invariant, `session_test` style).
7. A second `RunLockstep` over the same ring → identical final digest (replay-stable).

### Part C — desync localization (the NS5 detector applied to WFC generation)
Two peers run the SAME seed stream EXCEPT one tick `K` differs (a tampered/desynced seed) — the traces
match for ticks `< K` and diverge at `K`; `net::DesyncDetector` latches the EXACT tick. (The RP6/NS5
corruption-localization pattern, now over WFC levels.)
```cpp
// ringA: baseSeed + t.   ringB: identical EXCEPT ringB tick K uses (baseSeed + K) ^ 0x9999.
// traceA = DigestTrace(... ringA ...);  traceB = DigestTrace(... ringB ...);
// DesyncDetector d; for t in [0,T): RecordLocal(d, t, traceA[t]); for t: IngestRemote(d, {t, traceB[t]});
```
```
wfc-s5: desync injected at tick K=<k>; detector latched tick=<k>
PASS wfc-s5: identical seed streams report NO desync (clean)
PASS wfc-s5: a one-tick seed divergence is LOCATED at the exact tick K (net::DesyncDetector), traces match for t<K
PASS wfc-s5: the located divergence is deterministic on re-run (same tick + digests)
```
8. A clean run (ringA vs ringA) → `DesyncDetector` reports no desync.
9. ringA vs ringB (tick K tampered) → `d.desynced == true && d.desyncTick == K`, and `traceA[t] ==
   traceB[t]` for all `t < K` (localization, not just detection). Pin K.
10. Re-running the desync check reproduces the identical latched tick + digests.

Keep S1–S4 assertions green (append-only — all prior pinned digests unchanged: S1 `0xaaa67b9af6f293c8`,
S2 `0x4c9e67d356f4b920`, S3 `0x0fffd74f7e8419ac`/`0x8adb136f5b4c690a`, S4 `0xb3b956701ab39c83`/
`0x2e31ea681c0bd906`/`0xdeaac98f4242eeca`).

## Cross-platform proof (cheap loop — NO render-bake)
Controller `scp`s `engine/wfc/wfc.h` + `engine/net/session.h` + `tests/wfc_test.cpp` (+ `tests/test_main.h`
+ `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/wfc_test.cpp -o /tmp/wfc && /tmp/wfc`, confirming ALL assertions PASS with IDENTICAL pinned digests +
the IDENTICAL latched desync tick. NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/wfc/wfc.h` (add `MakeFullGrid`/`Generate` [+ optional `GenerateDigest`] below S4;
  do NOT modify S1–S4 types/functions). Header stays SELF-CONTAINED: only `<bit>/<cstdint>/<cstddef>/
  <vector>` + `net/session.h`. Do NOT include `pcg.h`/`fpx.h`/any other header. Do NOT modify
  `net/session.h` or any existing header (the lockstep/desync reuse is read-only, in the test).
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<random>`/`<unordered_*>`/`<functional>`/`std::hash`.
- `tests/wfc_test.cpp` stays self-contained; APPEND the S5 assertions (it may use `hf::net::Session`/
  `RunLockstep`/`DigestTrace`/`DesyncDetector`/`ChecksumPacket`/`RecordLocal`/`IngestRemote` from the
  already-included `net/session.h`). Keep S1–S4 green.
- Branch `fix-wfc-s5`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target wfc_test'`
  then run `wfc_test` and confirm ALL assertions (S1–S5) PASS, exit 0. First run: pin the digests + K.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `wfc_test` builds + PASSES on Windows
  with every assertion green (esp. the lockstep-trace-identical + the desync-located-at-K checks), and (if
  a local clang exists) the standalone clang compile passes. Report: commit hash, full test output (printed
  digests + latched tick + PASS lines), the pinned digests + K + the tileset/grid/seed/T/maxSteps used,
  confirmation S1–S4 digests are unchanged, confirmation the header is still self-contained (list
  `#include`s), and the local-clang result (or none → controller runs Mac). Commit message via temp file +
  `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for identical digests + latched tick,
  ff-merges to master + pushes + deletes the branch + advances to S6 — the lit 3D render capstone.)
