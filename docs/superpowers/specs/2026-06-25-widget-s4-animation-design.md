# Slice WIDGET-S4 — Widget animation via seq.h (Issue #30)

S1–S3 built the tree, layout solver, and data binding. S4 adds **animation** by composing the just-completed
SEQ flagship: a `seq::ScalarTrack` (Q16.16 keyframe curve + easing, already proven bit-exact) is bound to a
widget property; sampled at a tick and **quantized to integer pixels**, it drives the property, so the
animated tree + layout are byte-identical cross-platform AND (in S5) lockstep/scrub-able. This is "animate a
widget property over time," made deterministic — UMG's float animation curves rebuilt on the seq Q16.16
sampler.

**THE Q16.16 → PIXEL BOUNDARY (the only place fixed-point touches the integer box model):** `seq::SampleScalar`
returns a Q16.16 value; the box model is integer pixels. The conversion is a single documented round-nearest
quantize: `px = (sampled + 0x8000) >> 16` (arithmetic right shift, well-defined in C++20). Track values are
therefore "pixel values in Q16.16" (e.g. a width of 100px is `100 * kOne`). This is the seq-Q16.16 ↔
widget-pixel seam.

Pure-integer (the one float-free fixed-point crossing), append-only to `engine/ui/widget.h` (below S3; do NOT
modify S1–S3 — `0x53da0581a48f615e` / `0x95da64c52733eb16` / `0xbb31678bf35c1a37` / `0xc93bdcf2c0b473a3`
stay pinned). ONE new include: `#include "seq/seq.h"` (header-only self-contained — pulls `sim/fpx.h` +
`net/session.h` + `flow/flow.h`, all already standalone-clang-compilable; the standalone widget_test compile
stays valid with `-I engine`).

## Append to engine/ui/widget.h (below S3, in hf::ui)

### 1. The property animation
```cpp
// Bind a seq Q16.16 ScalarTrack to a widget property (reuses S3's WidgetProp ids — kPropWidth/kPropHeight/
// kPropFlexWeight). The track's values are Q16.16 PIXEL values (100px == 100*seq::kOne).
struct PropAnim {
    WidgetId         widget = 0;
    uint32_t         prop   = 0;   // a WidgetProp value
    seq::ScalarTrack track;        // the Q16.16 keyframe curve (seq.h)
};
```

### 2. The Q16.16 → pixel quantizer + `ApplyAnims`
```cpp
// Round-nearest Q16.16 -> integer pixels. The ONE documented quantization (arithmetic >> 16 is C++20-defined
// for signed). Used everywhere a sampled animation value writes the integer box model.
inline int32_t QuantizePx(seq::fx v) { return (int32_t)((v + (seq::fx)0x8000) >> 16); }

// Sample each animation at time `tSeconds` (Q16.16 seconds) and write the quantized pixel value into its
// widget property via S3's SetProp. Applied in ASCENDING anim-index order (the Propagate discipline —
// last-write-wins for two anims on the same (widget,prop)). Pure of side effects beyond the tree.
inline void ApplyAnims(Tree& t, const std::vector<PropAnim>& anims, seq::fx tSeconds);
```
(`ApplyAnims`: for each anim in order, `seq::fx v = seq::SampleScalar(anim.track, tSeconds);
SetProp(t, anim.widget, anim.prop, QuantizePx(v));` — reuses the proven seq sampler + S3's writer verbatim.)

### 3. Fixture (FIXED forever)
- `std::vector<PropAnim> MakeShowcaseAnims()` — a fixed set over the `MakeLayoutShowcase` widgets:
  - `header.kPropHeight` ← a track `times {0, kOne}`, `values {64*kOne, 128*kOne}`, Linear (so at `t = kOne/2`
    the height samples to `96*kOne` → quantizes to `96`).
  - `left.kPropFlexWeight` ← a track `times {0, kOne}`, `values {1*kOne, 5*kOne}`, Linear (at `t = kOne/2` →
    `3*kOne` → `3`).
  - `right.kPropFlexWeight` ← a track `times {0, kOne}`, `values {2*kOne, 2*kOne}`, Step/Linear (constant 2 —
    a control that does not change).
  (Keep FIXED forever — the golden pins the animated state at a fixed tick. Use the seq `Easing` already in
  `ScalarTrack`.)
- Fixed sample tick: `seq::kOne / 2` (0.5 seconds) for the golden.

## The golden (PINNED, cross-platform) — append to tests/widget_test.cpp
```
widget-s4: animated tree digest = 0x<...>   animated rects digest = 0x<...>  (t = 0.5s)
PASS widget-s1/s2/s3: ... (all prior assertions STILL green — every prior digest UNCHANGED)
PASS widget-s4: the quantizer is round-nearest — QuantizePx(96*kOne) == 96, QuantizePx(kOne/2) == 1, QuantizePx(0) == 0
PASS widget-s4: at t=0.5s the header height animates to 96 (the 64->128 linear track sampled + quantized)
PASS widget-s4: DigestTree after ApplyAnims(t=0.5s) == pinned uint64 (animated tree, byte-stable cross-platform)
PASS widget-s4: DigestRects(SolveLayout(animated tree)) == pinned uint64 (animation flowed through to layout)
PASS widget-s4: a different tick (t=0.25s) yields a DIFFERENT digest (animation is load-bearing on time)
PASS widget-s4: ApplyAnims is deterministic — re-applying at the same tick is bit-identical
```
Assertions:
1. **PRIOR INVARIANT** — re-assert S1 `0x53da0581a48f615e`, S2 `0x95da64c52733eb16`, S3 bound-tree
   `0xbb31678bf35c1a37` + bound-rects `0xc93bdcf2c0b473a3`, all UNCHANGED.
2. **QUANTIZER** — `QuantizePx(96 * seq::kOne) == 96`; `QuantizePx(seq::kOne / 2) == 1` (round-nearest of
   0.5 → 1); `QuantizePx(0) == 0`; `QuantizePx(seq::kOne) == 1`.
3. **ANIMATED VALUE** — after `ApplyAnims(MakeLayoutShowcase(), MakeShowcaseAnims(), seq::kOne/2)`,
   `GetProp(header, kPropHeight) == 96` (the 64→128 linear track at t=0.5s) and `GetProp(left,
   kPropFlexWeight) == 3`.
4. **PINNED ANIMATED TREE** — `DigestTree(animatedTree)` == a hard-pinned `uint64_t` (run once, pin THAT;
   identical MSVC + clang). DIFFERS from the unbound `0x53da0581a48f615e`.
5. **PINNED ANIMATED RECTS** — `DigestRects(SolveLayout(animatedTree, {0,0,1280,720}))` == a hard-pinned
   `uint64_t` (the animated header height + flex weights changed the layout).
6. **TIME LOAD-BEARING** — `ApplyAnims` at `t = kOne/4` (a fresh tree) → a DIFFERENT animated tree digest
   than at `t = kOne/2` (the animation depends on time).
7. **DETERMINISTIC** — `ApplyAnims` twice at the same tick (fresh trees) → identical digest.

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/ui/widget.h` + `engine/seq/seq.h` + `engine/sim/fpx.h` + `engine/math/math.h` +
`engine/flow/flow.h` + `engine/net/session.h` + `tests/widget_test.cpp` (+ `tests/test_main.h` +
`engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/widget_test.cpp -o /tmp/widget && /tmp/widget`, confirming ALL assertions PASS with the IDENTICAL
pinned digests. Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/ui/widget.h` (add `PropAnim`/`QuantizePx`/`ApplyAnims`/`MakeShowcaseAnims` below S3).
  Do NOT modify S1–S3 — all prior digests stay pinned.
- The ONLY new include is `#include "seq/seq.h"` (place it next to `net/session.h`). Header stays
  standalone-clang-compilable. STILL NO `<cmath>` directly (seq.h is `<cmath>`-free), NO float on the bit-exact
  path, NO RNG/clock/`<unordered_*>`/`<map>`/`std::hash`/`<algorithm>`/`<string>`, NO recursion. The
  quantize is `(v + 0x8000) >> 16`.
- Reuse `seq::SampleScalar` + S3's `SetProp` verbatim (zero new animation math; zero new property-write code).
- The track values are Q16.16 PIXEL values; the quantize is the documented round-nearest.
- `tests/widget_test.cpp` stays self-contained; APPEND the S4 assertions + `MakeShowcaseAnims`. Keep ALL
  S1–S3 assertions green.
- Branch `fix-widget-s4`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target widget_test'`
  (if the `cmd /c '...&&...'` handoff swallows output, run the build via the PowerShell tool natively). Run
  `widget_test`, confirm ALL assertions (S1–S4) PASS, exit 0. ALSO compile standalone with the local clang
  `C:\Program Files\LLVM\bin\clang++.exe` and confirm the IDENTICAL digests. First run: pin the animated
  tree + animated rects digests, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `widget_test` builds + PASSES on Windows
  with every assertion green (esp. prior invariants + the quantizer + the animated value == 96 + the pinned
  animated tree/rects + time-load-bearing), and the local clang standalone passes with identical digests.
  Report: commit hash, full test output (printed digests + PASS lines), the pinned animated tree + animated
  rects `uint64`s, confirmation S1–S3 digests unchanged, confirmation the header includes (S1's 4 +
  `seq/seq.h`), the anims you fixed + the sampled values at t=0.5s, and the local-clang result. Flag any
  deviation. Commit message via a temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S5 — input/event routing + the SCRUB via net::Session.)
