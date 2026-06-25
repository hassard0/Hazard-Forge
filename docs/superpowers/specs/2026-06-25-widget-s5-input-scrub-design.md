# Slice WIDGET-S5 — Input/event routing + SCRUB via net::Session (Issue #30)

S1–S4 built the tree, layout, binding, and animation. S5 is THE HEADLINE: deterministic **pointer hit-testing
+ event routing**, and wrapping the whole UI as a `net::Session` `World` so a UI **interaction sequence is
lockstep-replayable and SCRUB-able** — seek to interaction-tick S then replay forward is bit-identical to
replaying from tick 0 (the seq SCRUB==SEEK property via `net::CatchUp`). A UI session — clicks, animation,
data changes over time — replays bit-for-bit on any machine.

Pure-integer, append-only to `engine/ui/widget.h` (below S4; do NOT modify S1–S4 — `0x53da0581a48f615e` /
`0x95da64c52733eb16` / `0xbb31678bf35c1a37` / `0xc93bdcf2c0b473a3` / `0x0fb90c2346c92ca4` /
`0x494e7ff0ecd098de` stay pinned). NO new include (`net/session.h` + `seq/seq.h` already present — S5 USES
`DigestTrace`/`RunLockstep`/`JoinSnapshot`/`CatchUp`).

## Append to engine/ui/widget.h (below S4, in hf::ui)

### 1. Pointer events + deterministic hit-testing
```cpp
enum PointerType : uint32_t { kPointerNone = 0, kPointerDown = 1 };  // FROZEN
struct UiInput { int32_t x = 0, y = 0; uint32_t type = kPointerNone; };  // a net::Session Input (value-copyable)

// HitTest: the FRONT-MOST widget whose rect contains (x,y). Scan rects in REVERSE index order (later widgets
// = drawn on top = hit first), return the first containing rect, else kNoWidget. Integer compares only.
inline WidgetId HitTest(const std::vector<Rect>& rects, int32_t x, int32_t y);
```
(Contains = `x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h`. Reverse order is the topmost-on-top
convention — deterministic, no z-buffer.)

### 2. The UI world (a flat, value-copyable net::Session World)
```cpp
// The complete interactive UI state. Flat + value-copyable so net::Session's value-copy snapshot is COMPLETE
// by construction (the seq SeqPlayhead discipline). `tree` carries the styles (mutated by binding/animation),
// `model` the data, `time` the animation clock, `focus` the last-hit widget.
struct UiWorld {
    Tree                 tree;
    std::vector<int32_t> model;
    seq::fx              time  = 0;
    uint32_t             focus = kNoWidget;
};
// DigestUi: hand-LE the WHOLE world — EncodeTree(tree) bytes, then model ints, then time, then focus -> DigestBytes.
inline uint64_t DigestUi(const UiWorld& w);
```

### 3. `StepUi` — the deterministic UI transition (the net::Session StepFn)
```cpp
// One tick: advance the clock, route this tick's pointer events, then re-propagate bindings + re-apply
// animations so the model/time changes flow into the tree. Static config (anims/bindings/viewport/dt) is
// passed in; the test wraps it in a lambda capturing them (the StepPlayhead pattern). Signature maps to
// net::Session's step(world, inputs, tick) once the config is bound.
inline void StepUi(const std::vector<PropAnim>& anims, const std::vector<Binding>& bindings,
                   Rect viewport, seq::fx dtPerTick,
                   UiWorld& w, const std::vector<UiInput>& inputs, uint32_t /*tick*/);
```
- `w.time += dtPerTick`.
- For each `in` in `inputs` with `in.type == kPointerDown` (in order): `auto rects = SolveLayout(w.tree,
  viewport); WidgetId hit = HitTest(rects, in.x, in.y); w.focus = hit;` and if `hit != kNoWidget` and
  `!w.model.empty()`, **mutate the model deterministically**: `w.model[hit % w.model.size()] += 1` (a
  click bumps the model slot tied to the clicked widget — a real interactive feedback loop that drives
  bindings).
- After all events: `Propagate(w.tree, w.model, bindings); ApplyAnims(w.tree, anims, w.time);` (the model +
  the new clock both flow into the tree — reusing S3/S4 verbatim).

### 4. Fixtures (FIXED forever)
- `UiWorld MakeShowcaseUiWorld()` — `{ MakeLayoutShowcase(), MakeShowcaseModel(), 0, kNoWidget }` (the S2/S3
  fixtures). Keep FIXED.
- `net::InputRing<UiInput> MakeShowcaseInputRing()` — a fixed schedule, e.g. a `kPointerDown` at `{640, 360}`
  on tick 2 (a click in `body`/`right`), another at `{40, 30}` on tick 5 (a click in `header`), empty
  otherwise. Keep FIXED.
- Fixed `viewport = {0,0,1280,720}`, `dtPerTick = seq::kOne/10` (0.1s/tick), `N = 10` ticks.

## The golden (PINNED, cross-platform) — append to tests/widget_test.cpp
The test builds the step/digest lambdas (capturing `MakeShowcaseAnims()`/`MakeShowcaseBindings()`/viewport/dt):
```cpp
auto step = [&](ui::UiWorld& w, const std::vector<ui::UiInput>& in, uint32_t t){
    ui::StepUi(anims, bindings, viewport, dt, w, in, t); };
auto digest = [](const ui::UiWorld& w){ return ui::DigestUi(w); };
```
```
widget-s5: ui final digest = 0x<...>   ui trace digest = 0x<...>  (<N> ticks)
PASS widget-s1..s4: ... (all prior assertions STILL green — every prior digest UNCHANGED)
PASS widget-s5: HitTest is front-most — a point in an overlapping child returns the topmost widget, a miss returns kNoWidget
PASS widget-s5: RunLockstep ui final digest == pinned uint64 (a UI interaction sequence is deterministic)
PASS widget-s5: the per-tick DigestTrace digest == pinned uint64 (the interaction trace is replayable)
PASS widget-s5: SCRUB==SEEK — CatchUp(snapshot@S, N) world == from-0 playback world at N (bit-identical), several (S,N)
PASS widget-s5: a click is load-bearing — a run WITH the click differs from a run without (interaction changes state)
PASS widget-s5: two RunLockstep runs over the same ring are identical (deterministic)
```
Assertions:
1. **PRIOR INVARIANT** — re-assert S1 `0x53da0581a48f615e`, S2 `0x95da64c52733eb16`, S3 `0xbb31678bf35c1a37`
   + `0xc93bdcf2c0b473a3`, S4 `0x0fb90c2346c92ca4` + `0x494e7ff0ecd098de`, all UNCHANGED.
2. **HITTEST** — solve the layout showcase; a point inside `right`'s rect returns `right` (or its topmost
   child if one overlaps — assert the expected front-most id); a point outside all rects returns `kNoWidget`.
3. **PINNED FINAL** — `net::RunLockstep(MakeShowcaseUiWorld(), MakeShowcaseInputRing(), 10, step, digest)`
   == a hard-pinned `uint64_t` (run once, pin THAT; identical MSVC + clang).
4. **PINNED TRACE** — `net::DigestBytes` of `net::DigestTrace(...)` (the per-tick `uint64` vector) == a
   hard-pinned `uint64_t` (length 10; the interaction trace is byte-stable).
5. **SCRUB == SEEK (THE HEADLINE)** — for several `(S, N)` pairs (e.g. (2,10),(5,10),(7,7)): compute the
   from-0 world at N and the world at S (by stepping a `net::Session`), then `net::CatchUp(JoinSnapshot@S, N,
   ring, step)` and assert `DigestUi(caughtUp) == DigestUi(fromZeroAtN)` (full-world bit-identical — seek ==
   play; the seq S5 / profiler S5 precedent).
6. **CLICK LOAD-BEARING** — a `RunLockstep` over the showcase ring (with the clicks) yields a DIFFERENT final
   digest than over an EMPTY ring (no clicks) — the interaction genuinely changed the state (the model bumps
   propagated through bindings).
7. **DETERMINISTIC** — two `RunLockstep` calls over the same ring → identical final digest.

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/ui/widget.h` + `engine/seq/seq.h` + `engine/sim/fpx.h` + `engine/math/math.h` +
`engine/flow/flow.h` + `engine/net/session.h` + `tests/widget_test.cpp` (+ `tests/test_main.h` +
`engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/widget_test.cpp -o /tmp/widget && /tmp/widget`, confirming ALL assertions PASS with the IDENTICAL
pinned digests (esp. the final + trace + the SCRUB==SEEK equality). Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/ui/widget.h` (add `PointerType`/`UiInput`/`HitTest`/`UiWorld`/`DigestUi`/`StepUi`/
  `MakeShowcaseUiWorld`/`MakeShowcaseInputRing` below S4). Do NOT modify S1–S4 — all prior digests stay
  pinned.
- NO new include (`net/session.h` + `seq/seq.h` present). Header stays standalone-clang-compilable. STILL NO
  `<cmath>`/float-on-bit-exact-path/RNG/clock/`<unordered_*>`/`<map>`/`std::hash`/`<algorithm>`/`<string>`,
  NO recursion. Do NOT modify `net/session.h` (reuse `RunLockstep`/`DigestTrace`/`JoinSnapshot`/`CatchUp`
  read-only). `StepUi` reuses `SolveLayout`/`Propagate`/`ApplyAnims`/`HitTest` verbatim.
- The model mutation on click is EXACTLY `model[hit % model.size()] += 1` (documented, deterministic).
  HitTest is reverse-index front-most.
- `tests/widget_test.cpp` stays self-contained; APPEND the S5 assertions + the fixtures + the lambdas/ring.
  (Add a `namespace net = hf::net;` alias if convenient.) Keep ALL S1–S4 assertions green.
- Branch `fix-widget-s5`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target widget_test'`
  (if the `cmd /c '...&&...'` handoff swallows output, run the build via the PowerShell tool natively). Run
  `widget_test`, confirm ALL assertions (S1–S5) PASS, exit 0. ALSO compile standalone with the local clang
  `C:\Program Files\LLVM\bin\clang++.exe` and confirm the IDENTICAL digests. First run: pin the final + trace
  digests, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `widget_test` builds + PASSES on Windows
  with every assertion green (esp. prior invariants + the pinned final/trace + the SCRUB==SEEK equality +
  click-load-bearing), and the local clang standalone passes with identical digests. Report: commit hash,
  full test output (printed digests + PASS lines), the pinned final + trace `uint64`s, confirmation S1–S4
  digests unchanged, confirmation includes unchanged, how you wired SCRUB==SEEK via net::CatchUp (the S,N
  pairs), and the local-clang result. Flag any deviation. Commit message via a temp file + `git commit -F`
  (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S6 — the render capstone, the one float crossing in
  widget_render.h.)
