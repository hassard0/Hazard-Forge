# Slice RP5 — SCRUB + variable-speed playback (Flagship #28 REPLAY/DEMO, 5th/6)

RP4 gave us `Seek(N)` — jump to any tick, bit-identical to live. RP5 builds the user-facing TIMELINE on
top of it: a **Player** with a current position you can SCRUB forward and backward at VARIABLE SPEED
(1x, 2x, 0.5x-by-holding, reverse). The key invariants are determinism-preserving: speed changes only
WHEN you observe the world, never WHAT the world computes; and because the sim has NO inverse, backward
scrubbing is implemented as a re-Seek to the earlier tick (restore nearest keyframe + replay forward) —
so the path you took to reach tick N never affects the world AT tick N (path-independence / no drift).

Pure-CPU INTEGER, header-only `engine/replay/replay.h`, append-only (ADD `Player`/`MakePlayer`/
`ScrubTo`/`ScrubBy` below RP4; do NOT modify prior code). NO render-bake. Goldens pinned `uint64_t`,
identical Windows/MSVC + Mac/clang.

## Append to engine/replay/replay.h (below RP4's Seek, in hf::replay)

**Player<World,Input>** — a timeline cursor over a decoded demo. Holds the current tick + the current
world (a cache so forward scrubbing by a small delta can step from here instead of re-seeking):
```cpp
template <class World, class Input>
struct Player {
    const Demo<World,Input>* demo = nullptr;  // the decoded demo (non-owning; demo outlives the player)
    uint32_t currentTick = 0;                 // the cursor position (0 .. demo->header.tickCount)
    World    world{};                          // the world AS OF currentTick (cached)
    uint64_t digest = 0;                       // digest(world) at currentTick
};
```

**MakePlayer** — construct a Player positioned at tick 0 (the initial world):
```cpp
template <class World, class Input, class DigestFn>
Player<World,Input> MakePlayer(const Demo<World,Input>& demo, DigestFn digest);
// sets demo=&demo, currentTick=0, world=demo.initial, digest=digest(world).
```

**ScrubTo** — move the cursor to an ABSOLUTE target tick, forward or backward, and update the cached
world/digest. The determinism rule:
- **Forward** (`target >= currentTick`): step from the CACHED `world` over `[currentTick, target)` using
  `net::CatchUp`-style Advance (we already have the world at currentTick — no need to re-seek). This is
  the cheap path and proves the cache is consistent with a fresh seek.
- **Backward** (`target < currentTick`): the sim has no inverse, so RE-SEEK — call `Seek(*demo, target,
  step, digest)` (restore nearest keyframe ≤ target + replay forward). The result is bit-identical to
  forward-arriving at target.
```cpp
template <class World, class Input, class StepFn, class DigestFn>
void ScrubTo(Player<World,Input>& p, uint32_t target, StepFn step, DigestFn digest);
// clamps target to [0, demo->header.tickCount]; updates p.currentTick/world/digest.
```
(Implementation note: the FORWARD branch should produce a world bit-identical to `Seek(*demo, target)` —
that equality is the make-or-break assertion. Reuse `net::CatchUp` with a `JoinSnapshot{currentTick,
p.world}` for the forward steps so forward-scrub and seek share the exact replay body.)

**ScrubBy** — relative scrub by a signed delta (the variable-speed primitive: a 2x-forward player calls
`ScrubBy(+2)` per UI frame, a 0.5x player calls `ScrubBy(+1)` every other frame, reverse calls a
negative delta). Implement as `ScrubTo(p, p.currentTick + delta)` with saturating clamp at 0 and
`tickCount` (delta is `int64_t` to allow negative; clamp before casting to uint32):
```cpp
template <class World, class Input, class StepFn, class DigestFn>
void ScrubBy(Player<World,Input>& p, int64_t delta, StepFn step, DigestFn digest);
```

## The goldens (PINNED, cross-platform) — append to tests/replay_test.cpp
Use `demoKf4` (interval 4). Live reference at tick N = `net::RunLockstep<ToyA,InA>(ToyA{}, ring, N,
StepA, DigestA)`.
```
PASS rp5-scrub: MakePlayer at tick 0 -> world == initial, digest == DigestA(ToyA{})
PASS rp5-scrub: ScrubTo(N) (forward from 0) digest == live RunLockstep(N) == Seek(N) for N in {3,7,10,16}
PASS rp5-scrub: ScrubTo backward (16 -> 5 -> 11 -> 2) each lands == live RunLockstep(that tick) (re-seek, no drift)
PASS rp5-scrub: PATH-INDEPENDENCE — scrub 0->12->4->9 leaves the SAME world as a direct Seek(9)
PASS rp5-scrub: variable-speed 2x (ScrubBy(+2) from 0 to end) final digest == pinned 0x6227bc7b4046d08a
PASS rp5-scrub: variable-speed 1x and 2x and 4x all reach the IDENTICAL final digest (speed changes WHEN not WHAT)
PASS rp5-scrub: reverse playback (ScrubBy(-1) from 16 down to 0) each step == live RunLockstep(tick), ends at initial
PASS rp5-scrub: ScrubBy clamps at 0 and tickCount (over-scroll is a no-op at the ends)
```
Assertions:
1. **INIT** — `MakePlayer` → currentTick 0, world == `ToyA{}` (acc 0), digest == DigestA(ToyA{}).
2. **FORWARD == SEEK == LIVE** — from a fresh player, `ScrubTo(N)` then `p.digest` == `Seek(demoKf4,
   N).digest` == `net::RunLockstep(...,N,...)` for N in {3,7,10,16}.
3. **BACKWARD RE-SEEK** — scrub 16→5→11→2 (alternating back/forward); after each `ScrubTo`, `p.digest`
   == live RunLockstep(that target). (Proves backward via re-seek lands bit-identical.)
4. **PATH-INDEPENDENCE (make-or-break)** — a player scrubbed 0→12→4→9 has `p.world`/`p.digest` IDENTICAL
   to `Seek(demoKf4, 9)` (and to live RunLockstep(9)). The route taken does not perturb the destination
   — no accumulated drift.
5. **VARIABLE-SPEED REACHES PINNED END** — a 2x player (`ScrubBy(+2)` repeatedly until clamped at
   tickCount) ends with `p.digest == 0x6227bc7b4046d08a`. (16 is even so 2x lands exactly; if a stride
   overshoots, the clamp lands on tickCount — assert the final is the pinned end.)
6. **SPEED-INVARIANT** — players run at 1x (`ScrubBy(+1)`), 2x (`+2`), 4x (`+4`) from 0 to end all reach
   the IDENTICAL final digest `0x6227bc7b4046d08a` (speed changes the sampling cadence, never the result).
7. **REVERSE** — `ScrubBy(-1)` from tick 16 down to 0: at each step `p.digest` == live RunLockstep(that
   tick), ending at the initial world (DigestA(ToyA{})).
8. **CLAMP** — `ScrubBy(+100)` from near the end stops at tickCount (no OOB); `ScrubBy(-100)` from near
   the start stops at 0. Over-scroll at an end is a no-op.

Keep ALL RP1+RP2+RP3+RP4 assertions green (append-only — pinned hashes unchanged: no-keyframe
`0x92e4d491013137c4`, Kf4 `0xf2ccf29305652a39`, Kf8 `0xe7fb940bd0c3cc41`, ToyA final
`0x6227bc7b4046d08a`).

## Cross-platform proof (cheap loop — NO render-bake)
Controller `scp`s `engine/replay/replay.h` + `engine/net/session.h` + `tests/replay_test.cpp`
(+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs
`clang++ -std=c++20 -I engine -I tests tests/replay_test.cpp -o /tmp/rp5 && /tmp/rp5`, confirming ALL
assertions PASS identically. NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/replay/replay.h` (add `Player`/`MakePlayer`/`ScrubTo`/`ScrubBy` below RP4's
  code; do NOT modify RP1-RP4 — no format change). Header stays SELF-CONTAINED (only
  `<cstdint>/<cstddef>/<vector>` + `net/session.h`; NO new includes — `int64_t` is in `<cstdint>`). Do
  NOT modify `net/session.h` (reuse `net::CatchUp`/`net::JoinSnapshot` read-only) or any other header.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<functional>`/`<fstream>`. (Variable "speed" is integer
  tick STRIDE — there is no float framerate; 0.5x is "advance every other UI frame", modeled as the
  caller choosing when to call ScrubBy. RP5 proves the tick math, not a wall-clock.)
- `tests/replay_test.cpp` stays self-contained; APPEND the RP5 assertions. Keep all prior green.
- Branch `fix-replay-rp5`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target replay_test'`
  then run `replay_test` and confirm ALL assertions (RP1-RP5) PASS, exit 0.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `replay_test` builds + PASSES on Windows
  with every assertion green (esp. path-independence + speed-invariance), and (if a local clang exists)
  the standalone clang compile passes. Report: commit hash, full test output (all PASS lines), confirmation
  the four pinned hashes are unchanged, confirmation the header is still self-contained (list #includes),
  and the local-clang result (or none → controller runs Mac). Commit message via temp file + `git commit
  -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for identical hashes, ff-merges to
  master + pushes + deletes the branch + advances to RP6 — the capstone + corruption detection.)
