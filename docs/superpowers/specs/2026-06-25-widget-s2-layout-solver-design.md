# Slice WIDGET-S2 — Integer layout solver (Issue #30)

S1 shipped the widget tree + integer box model (tree digest `0x53da0581a48f615e`). S2 is THE CRUX: a
**deterministic integer layout solver** — a single top-down pass that computes each widget's pixel `Rect`
from the box model (stack direction + margins/padding + fixed sizes + flex-grow distribution), bit-identical
on every compiler. The make-or-break detail is the **integer flex distribution with a precise remainder
rule**: leftover pixels that don't divide evenly across flex weights are assigned to the LOWEST-INDEX flex
children one each, so the container is always exactly filled AND the result is deterministic.

Pure-integer, append-only to `engine/ui/widget.h` (below S1; do NOT modify S1 — `0x53da0581a48f615e` stays
pinned). NO new include.

## The layout-direction decision (append-only — a new flag bit, NOT a Style field)
S1 froze `Style` without a direction field. S2 adds direction as a NEW `StyleFlags` bit (appending an enum
value does NOT change the `Style` struct layout or S1's `EncodeTree`, so S1's digest is unaffected):
```cpp
// APPEND to StyleFlags (S1's kFixedW/kFixedH/kFlexGrow unchanged). A container lays its children along the
// MAIN axis: VERTICAL by default; HORIZONTAL when kStackH is set. (The cross axis is the other one.)
kStackH = 1u << 3,   // container stacks its children horizontally (default = vertical)
```
(No 4-mode enum — a single direction bit + the existing `kFlexGrow` covers stack + flex-grow. Fixed sizes
come from `kFixedW`/`kFixedH` + `Style::width`/`height`; flex from `kFlexGrow` + `Style::flexWeight`.)

## Append to engine/ui/widget.h (below S1, in hf::ui)

### 1. The computed rect
```cpp
struct Rect { int32_t x = 0, y = 0, w = 0, h = 0; };   // integer pixels
```

### 2. `SolveLayout` — one deterministic top-down pass (index order, NO recursion)
```cpp
inline std::vector<Rect> SolveLayout(const Tree& t, Rect viewport);
```
- Allocate `rects` sized `t.widgets.size()`, all zero. Set `rects[t.root] = viewport`.
- **Process widgets in INDEX order** (0..N-1). Because `AddWidget` always appends a child AFTER its parent,
  a parent's index is always < its children's indices, so a parent's rect is finalized before its children
  are reached — a parent lays out its OWN children when it is visited. (No recursion, no work-stack — a flat
  ascending scan, the determinism discipline.) For each widget `p`:
  - If `p` has no children (`firstChild == kNoWidget`), skip (its rect was set by its parent).
  - Compute `p`'s **content box**: `cx = rects[p].x + padL`, `cy = rects[p].y + padT`,
    `cw = rects[p].w - padL - padR`, `ch = rects[p].h - padT - padB` (clamp negatives to 0).
  - `horizontal = (p.style.flags & kStackH) != 0`. Main size = `horizontal ? cw : ch`. (Main axis = x/w when
    horizontal, y/h when vertical; cross axis is the other.)
  - **Pass A over `p`'s children** (walk the firstChild→nextSibling chain): accumulate
    `usedMain = Σ (child main margin both sides) + Σ (fixed child main size)` and `totalWeight = Σ flexWeight
    of kFlexGrow children`. (A child's "fixed main size" = `style.width` if horizontal+kFixedW, `style.height`
    if vertical+kFixedH; a non-fixed non-flex child has main size 0 in v1 — documented.)
  - `leftover = max(0, mainSize - usedMain)`. **Flex distribution:** for each flex child its
    `share = (int32_t)(((int64_t)leftover * flexWeight) / totalWeight)` (int64 intermediate; totalWeight>0
    guarded). `remainder = leftover - Σ shares`. Distribute the `remainder` pixels by giving **+1 to each
    flex child in ASCENDING child order until the remainder is exhausted** (the lowest-index flex children
    get the extra pixels — the deterministic, hash-free remainder rule; THE spec to pin precisely).
  - **Pass B — place children** along the main axis: `cursor = content main-start`. For each child in chain
    order: `cursor += child leading main margin`; the child's main size = (fixed ? fixed size : flex ? its
    distributed share : 0); main pos = `cursor`; the CROSS size = `content cross size - child cross margins`
    (stretch) unless the child is fixed on the cross axis (`kFixedW`/`kFixedH` for that axis → use
    `style.width`/`height`); cross pos = `content cross-start + child leading cross margin`. Write
    `rects[child] = {x,y,w,h}` (map main/cross back to x/y/w/h per `horizontal`). Then `cursor += child main
    size + child trailing main margin`.
- Return `rects`.

### 3. `DigestRects`
```cpp
inline uint64_t DigestRects(const std::vector<Rect>& rects);  // PutI32(x),PutI32(y),PutI32(w),PutI32(h) per rect -> DigestBytes
```

### 4. Fixture (FIXED forever)
- `Tree MakeLayoutShowcase()` — start from `MakeShowcaseTree()` (S1) and set the `body` widget's
  `style.flags |= kStackH` (so its `left`/`right` children lay out side-by-side, exercising horizontal flex;
  the root stays vertical). Keep FIXED. (Return the modified tree. This leaves S1's `MakeShowcaseTree`
  untouched/frozen.)

## The golden (PINNED, cross-platform) — append to tests/widget_test.cpp
```
widget-s2: layout rects digest = 0x<...>  (<N> rects)
PASS widget-s1: DigestTree(MakeShowcaseTree()) == 0x53da0581a48f615e UNCHANGED (append-only)
PASS widget-s2: DigestRects(SolveLayout(MakeLayoutShowcase(), {0,0,1280,720})) == pinned uint64 (cross-platform)
PASS widget-s2: the viewport is exactly filled — the vertical stack's children heights + margins sum to 720
PASS widget-s2: fixed-size widgets keep their size (header height == 64, footer height == 32)
PASS widget-s2: flex distribution is weighted — body's right child (weight 2) is ~2x the left child (weight 1)
PASS widget-s2: the remainder rule — leftover pixels go to the LOWEST-INDEX flex child (exact-fill, deterministic)
PASS widget-s2: re-solving is bit-identical (deterministic)
```
Assertions:
1. **S1 INVARIANT** — re-assert `DigestTree(MakeShowcaseTree()) == 0x53da0581a48f615e` UNCHANGED.
2. **PINNED RECTS** — `DigestRects(SolveLayout(MakeLayoutShowcase(), {0,0,1280,720}))` == a hard-pinned
   `uint64_t` (run once, pin THAT; identical MSVC + clang).
3. **EXACT FILL** — the root is a vertical stack of header(64) + body(flex) + footer(32); assert
   `header.h + body.h + footer.h + (their vertical margins)` exactly equals the content height (1280×720
   viewport minus root padding) — the flex child consumed all leftover, no lost pixels.
4. **FIXED PRESERVED** — `rects[header].h == 64` and `rects[footer].h == 32` (fixed sizes honored).
5. **WEIGHTED FLEX** — in `body` (horizontal stack), `rects[right].w` ≈ `2 * rects[left].w` (within the ±1px
   remainder; assert `right.w == 2*left.w` OR `right.w == 2*left.w ± 1` given the remainder rule) — weight 2
   vs weight 1.
6. **REMAINDER RULE** — construct a small direct case: a horizontal container of width W with two flex
   children weight 1 each where `W` is ODD (so leftover is odd) → assert the LOWER-index child gets the extra
   pixel (`left.w == right.w + 1`), and `left.w + right.w == W` (exact fill). Pin this explicitly — it is the
   determinism crux.
7. **DETERMINISTIC** — a second `SolveLayout` → identical `DigestRects`.

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/ui/widget.h` + `engine/net/session.h` + `tests/widget_test.cpp` (+
`tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine
-I tests tests/widget_test.cpp -o /tmp/widget && /tmp/widget`, confirming ALL assertions PASS with the
IDENTICAL pinned digests. Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/ui/widget.h` (add the `kStackH` flag value to `StyleFlags`, `Rect`, `SolveLayout`,
  `DigestRects`, `MakeLayoutShowcase` below S1). Adding `kStackH` to the enum does NOT change `Style`/
  `EncodeTree` — S1's digest `0x53da0581a48f615e` MUST stay pinned (re-assert it). Do NOT modify S1's
  `MakeShowcaseTree`/`EncodeTree`/`Style`/`Widget`.
- NO new include. Self-contained (4 includes: `<cstddef>/<cstdint>/<vector>` + `net/session.h`). STILL NO
  `<cmath>`/float/clock/RNG/`<unordered_*>`/`<map>`/`std::hash`/`<algorithm>`/`<string>`, NO recursion (the
  solve is a flat ascending index scan + per-container child-chain walks). The flex share uses ONE int64
  intermediate; everything else is int32.
- The remainder rule is EXACTLY "lowest-index flex children get +1 each until exhausted" — pin it (assertion
  6). The viewport must be exactly consumed by flex.
- `tests/widget_test.cpp` stays self-contained; APPEND the S2 assertions + `MakeLayoutShowcase`. Keep ALL 6
  S1 assertions green.
- Branch `fix-widget-s2`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target widget_test'`
  (if the `cmd /c '...&&...'` handoff swallows output, run the build via the PowerShell tool natively). Run
  `widget_test`, confirm ALL assertions (S1 + S2) PASS, exit 0. ALSO compile standalone with the local clang
  `C:\Program Files\LLVM\bin\clang++.exe` and confirm the IDENTICAL digests. First run: pin the layout-rects
  digest, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `widget_test` builds + PASSES on Windows
  with every assertion green (esp. S1 digest unchanged + the pinned rects digest + exact-fill + the remainder
  rule), and the local clang standalone passes with identical digests. Report: commit hash, full test output
  (printed digest + PASS lines), the pinned rects `uint64`, confirmation the S1 digest is unchanged,
  confirmation the header is self-contained (4 includes, no recursion), the layout you computed (a few rects),
  and the local-clang result. Flag any deviation. Commit message via a temp file + `git commit -F` (Bash
  heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S3 — data binding.)
