# Slice WIDGET-S3 — Data binding (Issue #30)

S1 gave the widget tree, S2 the layout solver. S3 adds **data binding**: a model (a vector of integer
values) drives widget properties through a set of bindings, applied in a **deterministic order**, so the
post-binding tree + layout are byte-identical cross-platform. This is the UMG "bind a property to a data
source" feature, made deterministic.

Bindings target **existing integer `Style` fields** (`width`, `height`, `flexWeight`) — all already in S1's
`EncodeTree` and all affecting layout, so a binding visibly changes both the tree digest and the computed
rects. (No new `Style` field is added — S1's struct + digest stay frozen. Opacity/visibility are render-side
concerns deferred to S6.)

Pure-integer, append-only to `engine/ui/widget.h` (below S2; do NOT modify S1/S2 — `0x53da0581a48f615e` and
`0x95da64c52733eb16` stay pinned). NO new include (S3 uses the SIMPLE ascending-index propagation; flow.h
composition is a documented later option, not needed for v1).

## Append to engine/ui/widget.h (below S2, in hf::ui)

### 1. Bindable properties + the binding
```cpp
// Selects a writable integer Style field. FROZEN values (the wire contract; S4 animation reuses them).
enum WidgetProp : uint32_t {
    kPropWidth      = 0,   // -> Style::width   (a fixed-size widget's pixel width)
    kPropHeight     = 1,   // -> Style::height  (a fixed-size widget's pixel height)
    kPropFlexWeight = 2,   // -> Style::flexWeight (a flex child's grow weight)
};
struct Binding {
    uint32_t srcModelIdx = 0;   // index into the model vector (the source value)
    WidgetId dstWidget   = 0;   // the widget whose property is written
    uint32_t dstProp     = 0;   // a WidgetProp value (the destination field)
};
```

### 2. `SetProp` / `GetProp` — write/read a widget property by id
```cpp
// Write `value` into the WidgetProp-selected Style field of widget `w` (out-of-range widget/prop = no-op,
// deterministic). flexWeight is uint32 — clamp negative `value` to 0 before the cast.
inline void SetProp(Tree& t, WidgetId w, uint32_t prop, int32_t value);
inline int32_t GetProp(const Tree& t, WidgetId w, uint32_t prop);   // for the tests
```

### 3. `Propagate` — apply the bindings in deterministic order
```cpp
// Apply each binding IN ASCENDING BINDING-INDEX ORDER (a flat ordered scan — the InputRing insertion-order
// discipline; NO hash-map). For each binding: if srcModelIdx and dstWidget are in range,
// SetProp(t, b.dstWidget, b.dstProp, model[b.srcModelIdx]). Two bindings to the same (widget,prop) →
// LAST-WRITE-WINS (the higher binding index wins — deterministic). Pure integer.
inline void Propagate(Tree& t, const std::vector<int32_t>& model, const std::vector<Binding>& bindings);
```

### 4. Fixtures (FIXED forever)
- `std::vector<int32_t> MakeShowcaseModel()` — a fixed model, e.g. `{ 80, 3, 48, 2, 99 }` (index 0 = a header
  height, 1 = left flex weight, 2 = footer height, 3 = right flex weight, 4 = an unused-to-show-some-model-
  values-unbound). Keep FIXED.
- `std::vector<Binding> MakeShowcaseBindings()` — a fixed set binding model values to the layout showcase's
  widgets, e.g.: `{model[0] → header.kPropHeight}`, `{model[1] → left.kPropFlexWeight}`, `{model[3] →
  right.kPropFlexWeight}`, `{model[2] → footer.kPropHeight}`. (Use the widget ids from `MakeLayoutShowcase`.)
  Include ONE pair of bindings targeting the SAME (widget,prop) to exercise last-write-wins (e.g. two
  bindings to `header.kPropHeight`, the second from a different model index — the second must win). Keep
  FIXED.

## The golden (PINNED, cross-platform) — append to tests/widget_test.cpp
```
widget-s3: bound tree digest = 0x<...>   bound rects digest = 0x<...>
PASS widget-s1/s2: ... (all prior assertions STILL green — 0x53da0581a48f615e + 0x95da64c52733eb16 UNCHANGED)
PASS widget-s3: after Propagate, the bound widget properties hold the model values (header.height == model[...])
PASS widget-s3: DigestTree after Propagate == pinned uint64 (binding changed the tree, byte-stable)
PASS widget-s3: DigestRects(SolveLayout(bound tree)) == pinned uint64 (the binding flowed through to layout)
PASS widget-s3: last-write-wins — two bindings to the same (widget,prop) resolve to the HIGHER-index binding
PASS widget-s3: binding is load-bearing — changing a model value changes the bound digest
PASS widget-s3: Propagate is deterministic — re-applying yields the identical digest
```
Assertions:
1. **PRIOR INVARIANT** — re-assert S1 `0x53da0581a48f615e` and S2 `0x95da64c52733eb16`, both UNCHANGED.
2. **BOUND PROPERTIES** — after `Propagate(tree, MakeShowcaseModel(), MakeShowcaseBindings())`,
   `GetProp(tree, left, kPropFlexWeight) == model[1]` (and the other bindings) — the model values landed in
   the widget properties.
3. **PINNED BOUND TREE** — `DigestTree(tree)` AFTER Propagate == a hard-pinned `uint64_t` (run once, pin
   THAT; identical MSVC + clang). It DIFFERS from the unbound `0x53da0581a48f615e` (binding changed the
   styles).
4. **PINNED BOUND RECTS** — `DigestRects(SolveLayout(boundTree, {0,0,1280,720}))` == a hard-pinned `uint64_t`
   (the binding flowed through to the computed layout — e.g. the header's bound height changes its rect; the
   flex-weight bindings change left/right widths).
5. **LAST-WRITE-WINS** — for the two bindings targeting the same `header.kPropHeight`, after Propagate
   `GetProp(header, kPropHeight)` == the model value of the HIGHER-index binding (not the lower). (Prove the
   deterministic conflict resolution.)
6. **LOAD-BEARING** — clone the model, change one bound value, re-Propagate (fresh tree) → a DIFFERENT bound
   tree digest.
7. **DETERMINISTIC** — Propagate twice (fresh trees) → identical bound digest.

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/ui/widget.h` + `engine/net/session.h` + `tests/widget_test.cpp` (+
`tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine
-I tests tests/widget_test.cpp -o /tmp/widget && /tmp/widget`, confirming ALL assertions PASS with the
IDENTICAL pinned digests. Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/ui/widget.h` (add `WidgetProp`/`Binding`/`SetProp`/`GetProp`/`Propagate`/
  `MakeShowcaseModel`/`MakeShowcaseBindings` below S2). Do NOT modify S1/S2 — `0x53da0581a48f615e` and
  `0x95da64c52733eb16` stay pinned. Do NOT add a field to `Style` (bind existing fields only).
- NO new include. Self-contained (4 includes). STILL NO `<cmath>`/float/clock/RNG/`<unordered_*>`/`<map>`/
  `std::hash`/`<algorithm>`/`<string>`, NO recursion. Bindings applied in ASCENDING index order (no
  hash-map). last-write-wins by index.
- `tests/widget_test.cpp` stays self-contained; APPEND the S3 assertions + the fixtures. Keep ALL S1+S2
  assertions green.
- Branch `fix-widget-s3`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target widget_test'`
  (if the `cmd /c '...&&...'` handoff swallows output, run the build via the PowerShell tool natively). Run
  `widget_test`, confirm ALL assertions (S1–S3) PASS, exit 0. ALSO compile standalone with the local clang
  `C:\Program Files\LLVM\bin\clang++.exe` and confirm the IDENTICAL digests. First run: pin the bound tree +
  bound rects digests, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `widget_test` builds + PASSES on Windows
  with every assertion green (esp. prior invariants + the pinned bound tree/rects + last-write-wins + the
  load-bearing/deterministic checks), and the local clang standalone passes with identical digests. Report:
  commit hash, full test output (printed digests + PASS lines), the pinned bound tree + bound rects `uint64`s,
  confirmation S1/S2 digests unchanged, confirmation includes unchanged (4, no recursion), the bindings you
  fixed, and the local-clang result. Flag any deviation. Commit message via a temp file + `git commit -F`
  (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S4 — widget animation via seq.h.)
