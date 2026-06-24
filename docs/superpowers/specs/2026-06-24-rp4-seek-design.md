# Slice RP4 — SEEK to an arbitrary tick (Flagship #28 REPLAY/DEMO, 4th/6)

RP3 captured keyframes (full world snapshots at coarse intervals) and proved each keyframe-at-tick-T is
a BIT-IDENTICAL mid-session restore point. RP4 turns that into the headline timeline capability:
**SEEK(N)** — jump to ANY tick N by restoring the nearest keyframe at-or-before N and replaying forward
over the input tail `[keyframeTick, N)`. This is `net::CatchUp` verbatim (restore a confirmed snapshot,
replay the tail) — the same primitive that proves late-join == bit-identical at NS6. Seeking is exact
re-derivation, NOT interpolation: `Seek(demo, N)` is byte-identical to having run the live session for N
ticks. The win over replaying-from-0 is cost: from the nearest keyframe you replay at most
`keyframeInterval-1` ticks instead of N.

Pure-CPU INTEGER, header-only `engine/replay/replay.h`, append-only (do NOT modify RP1/RP2/RP3 code —
ADD `Seek` below). NO render-bake. Goldens are pinned `uint64_t`s, identical Windows/MSVC + Mac/clang.

## Append to engine/replay/replay.h (below RP3's code, in hf::replay)

**Seek** — restore the nearest keyframe at-or-before `toTick`, then Advance forward to `toTick` over the
demo's input ring, returning the sought world AND its digest:
```cpp
template <class World, class Input>
struct SeekResult {
    World    world{};            // the world AS OF toTick (bit-identical to live-at-toTick)
    uint64_t digest = 0;         // digest(world)
    uint32_t keyframeTick = 0;   // the keyframe we restored from (the nearest <= toTick; 0 if none)
    uint32_t replayedTicks = 0;  // how many ticks we replayed forward (toTick - keyframeTick) — the seek cost
};

template <class World, class Input, class StepFn, class DigestFn>
SeekResult<World,Input> Seek(const Demo<World,Input>& demo, uint32_t toTick, StepFn step, DigestFn digest);
```
Implementation:
1. **Pick the nearest keyframe at-or-before `toTick`.** Scan `demo.keyframes` (decoded by RP3's
   DecodeDemo — sorted ascending by tick, ticks 0, K, 2K, ...) for the LAST one with `tick <= toTick`.
   The keyframe at tick 0 always exists when `keyframeInterval > 0`, so there is always a base for
   `toTick >= 0`. If `demo.keyframes` is empty (no keyframes — keyframeInterval 0) OR no keyframe has
   `tick <= toTick`, fall back to the initial world at tick 0 (`base = demo.initial`, `keyframeTick = 0`)
   — seeking still works from tick 0, just at full replay cost. (Clamp `toTick` to `demo.header.tickCount`
   — seeking past the end returns the final world; document it.)
2. **Restore + replay forward** = `net::CatchUp`: build a `net::JoinSnapshot<World>{ .tick = keyframeTick,
   .world = base }` and call `net::CatchUp(snap, toTick, demo.ring, step)`. (CatchUp restores the snapshot
   and steps every tick in `[keyframeTick, toTick)` over `demo.ring.At(t)` — the NS1 Advance body seeded
   from the keyframe instead of tick 0.) Set `replayedTicks = toTick - keyframeTick`.
3. `world` = the caught-up world; `digest = digest(world)`.

(Reuse `net::CatchUp` + `net::JoinSnapshot` directly — do NOT hand-roll the replay loop; the whole point
is that Seek IS CatchUp. If a tiny helper to find the nearest keyframe reads cleaner, add it append-only.)

## The goldens (PINNED, cross-platform) — append to tests/replay_test.cpp
Decode `demoKf4` (interval 4, keyframes at ticks 0,4,8,12 — from RP3) and Seek to several N, asserting
each equals the live reference (BIT-IDENTICAL, not interpolation). The live reference at tick N is
`net::RunLockstep<ToyA,InA>(ToyA{}, ring, N, StepA, DigestA)`.
```
rp4-seek: seek N=10 -> keyframeTick=8 replayed=2 digest=0x.... (== live RunLockstep(10))
PASS rp4-seek: Seek(demoKf4, N) digest == live RunLockstep(N) for N in {0,4,7,8,12,15,16} (bit-identical)
PASS rp4-seek: Seek restores the NEAREST keyframe <= N (keyframeTick/replayedTicks correct per N)
PASS rp4-seek: on-keyframe seek (N in {0,4,8,12}) replays ZERO ticks (keyframeTick==N, replayedTicks==0)
PASS rp4-seek: just-before-next-keyframe seek (N in {3,7,11}) replays the MAX tail (replayedTicks==N-floor(N/4)*4)
PASS rp4-seek: keyframeless demo (interval 0) seeks from tick 0 (full replay) and still == live RunLockstep(N)
PASS rp4-seek: Seek(demoKf4, N).digest == Seek(demoKf8, N).digest == live (keyframe interval is invisible to the result)
```
Assertions:
1. **SEEK == LIVE (make-or-break)** — for N in {0, 4, 7, 8, 12, 15, 16}: `Seek(demoKf4, N, StepA,
   DigestA).digest` == `net::RunLockstep<ToyA,InA>(ToyA{}, ring, N, StepA, DigestA)`. (N=0 → DigestA of
   `ToyA{}`; N=16 → the pinned final `0x6227bc7b4046d08a`.) Bit-identical re-derivation.
2. **NEAREST KEYFRAME** — the `keyframeTick`/`replayedTicks` returned match expectation: for N=10,
   keyframeTick=8, replayedTicks=2; for N=7, keyframeTick=4, replayedTicks=3; for N=4, keyframeTick=4,
   replayedTicks=0; etc.
3. **ON-KEYFRAME ZERO REPLAY** — N in {0,4,8,12} → replayedTicks==0 (restored exactly, no forward steps).
4. **MAX-TAIL** — N in {3,7,11} (just before the next keyframe) → replayedTicks == N - (N/4)*4 (i.e.
   3 each), the worst-case seek cost bounded by keyframeInterval-1.
5. **KEYFRAMELESS FALLBACK** — Seek on the no-keyframe demo (keyframeInterval 0) for the same N still ==
   live RunLockstep(N), replaying from tick 0 (keyframeTick==0, replayedTicks==N).
6. **INTERVAL-INVISIBLE** — `Seek(demoKf4, N).digest == Seek(demoKf8, N).digest == live` for the same N
   (the result is independent of how many keyframes the demo stored — interval is purely a cost knob).

Keep ALL RP1+RP2+RP3 assertions green (append-only — no format change, pinned hashes unchanged:
no-keyframe `0x92e4d491013137c4`, Kf4 `0xf2ccf29305652a39`, Kf8 `0xe7fb940bd0c3cc41`, ToyA final
`0x6227bc7b4046d08a`).

## Cross-platform proof (cheap loop — NO render-bake)
Controller `scp`s `engine/replay/replay.h` + `engine/net/session.h` + `tests/replay_test.cpp`
(+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs
`clang++ -std=c++20 -I engine -I tests tests/replay_test.cpp -o /tmp/rp4 && /tmp/rp4`, confirming ALL
assertions PASS identically. NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/replay/replay.h` (add `SeekResult`/`Seek` below RP3's code; do NOT modify
  RP1/RP2/RP3 format code — no format change this slice). Header stays SELF-CONTAINED (only
  `<cstdint>/<cstddef>/<vector>` + `net/session.h`; NO new includes). Do NOT modify `net/session.h`
  (reuse `net::CatchUp`/`net::JoinSnapshot` read-only) or any other existing header.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<functional>`/`<fstream>`.
- `tests/replay_test.cpp` stays self-contained; APPEND the RP4 assertions. Keep all prior assertions
  green (pinned hashes unchanged).
- Branch `fix-replay-rp4`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target replay_test'`
  then run `replay_test` and confirm ALL assertions (RP1+RP2+RP3+RP4) PASS, exit 0.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `replay_test` builds + PASSES on Windows
  with every assertion green (incl. all the Seek==live checks + the nearest-keyframe/zero-replay/max-tail
  bookkeeping), and (if a local clang exists) the standalone clang compile passes. Report: commit hash,
  full test output (all PASS lines + the printed seek digests/keyframeTick/replayedTicks), confirmation
  the four pinned hashes are unchanged, confirmation the header is still self-contained (list #includes),
  and the local-clang result (or none → controller runs Mac). Commit message via temp file + `git commit
  -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for identical hashes, ff-merges to
  master + pushes + deletes the branch + advances to RP5 SCRUB.)
