# Slice WIDGET-S1 — Widget tree + integer box model (Issue #30, beachhead)

The beachhead of the DETERMINISTIC UMG-CLASS UI FRAMEWORK (issue #30) — a deterministic UI core whose
layout/binding/animation/interaction state is byte-identical cross-platform + replayable (the float
text-shaping + pixel raster isolated later in `widget_render.h`). S1 establishes the substrate: a **widget
tree** (a hierarchy of widgets with an integer box-model style — size, margins, padding in INTEGER PIXELS) +
a deterministic tree digest.

**THE INTEGER-PIXEL DECISION (banner it):** the box model is **integer pixels**, NOT Q16.16 — UI layout is
pixel-quantized and integer constraint math is trivially deterministic (no `fxmul` rounding). Q16.16 enters
ONLY in S4 (animation), quantized back to pixels on apply. So S1's style fields are plain `int32_t` pixels.

The golden is a hard-pinned `net::DigestBytes` over the tree's hand-LE serialization, proven identical
Windows/MSVC + Mac/clang via a standalone clang compile — NO render-bake.

## NEW file: engine/ui/widget.h (namespace hf::ui)
Header-only and **SELF-CONTAINED**: include ONLY `<cstddef>`, `<cstdint>`, `<vector>`, plus
`#include "net/session.h"` (for `hf::net::DigestBytes`). NO `<cmath>` / float / clock / RNG / `<random>` /
`<unordered_*>` / `<map>` / `<functional>` / `std::hash` / `<algorithm>` / `<string>`. Do NOT include
`replay.h` / `seq.h` / `flow.h` yet (S3/S4 add `seq.h`/`flow.h`) — inline the LE appenders here. It MUST
compile standalone: `clang++ -std=c++20 -I engine -I tests tests/widget_test.cpp`. This is ONE growing header
— every later slice (S2–S5) APPENDS a section below S1; do NOT modify S1's symbols once pinned. (Existing dir
`engine/ui/`.)

### Inline little-endian appenders (self-contained — mirror flow.h:805)
```cpp
inline void PutU32(std::vector<uint8_t>& b, uint32_t v) { /* 4 bytes LE, byte-by-byte */ }
inline void PutI32(std::vector<uint8_t>& b, int32_t v) { PutU32(b, (uint32_t)v); }  // int32 bit pattern, LE-stable
```
(S2+ may add `GetU32`/`PutU64` as needed — inline.)

### Types (all in hf::ui)
```cpp
using WidgetId = uint32_t;                  // a widget's id == its index into Tree::widgets (monotonic, never recycled — the flow.h NodeId discipline)
constexpr WidgetId kNoWidget = 0xFFFFFFFFu; // the "no widget" sentinel (firstChild/nextSibling terminator)

// Box-model style flags — FROZEN bit values (the wire contract; S2 reads them). Do NOT renumber.
enum StyleFlags : uint32_t {
    kFixedW   = 1u << 0,   // width is a fixed pixel size (else derived by layout)
    kFixedH   = 1u << 1,   // height is a fixed pixel size
    kFlexGrow = 1u << 2,   // this widget grows to fill leftover main-axis space (S2)
};

struct Style {                              // ALL integer pixels
    int32_t  width = 0,  height = 0;        // fixed pixel size when kFixedW/kFixedH set
    int32_t  marginL = 0, marginT = 0, marginR = 0, marginB = 0;
    int32_t  padL = 0,    padT = 0,    padR = 0,    padB = 0;
    uint32_t flexWeight = 1;                // relative grow weight when kFlexGrow set (S2)
    uint32_t flags      = 0;                // StyleFlags bitmask
};

struct Widget {
    WidgetId parent      = kNoWidget;       // index of parent (kNoWidget for the root)
    uint32_t firstChild  = kNoWidget;       // index of the first child (kNoWidget if none)
    uint32_t nextSibling = kNoWidget;       // index of the next sibling (kNoWidget if last)
    Style    style;
    uint32_t kind        = 0;               // widget kind id (e.g. 0=panel, 1=text — caller-defined; S6 uses it)
};

struct Tree {
    std::vector<Widget> widgets;            // widgets[0] is the ROOT by convention
    WidgetId            root = 0;
};
```

### Functions (pure, deterministic)
1. **`WidgetId AddWidget(Tree& t, WidgetId parent, const Style& s, uint32_t kind)`** — append a new widget
   (id = `widgets.size()`), set its `parent`/`style`/`kind`, and **link it as the LAST child** of `parent`
   (walk `parent.firstChild`'s `nextSibling` chain to the end, set the last's `nextSibling = newId`, or set
   `parent.firstChild = newId` if the parent has no children) — children in insertion order, the ordered
   first-child/next-sibling chain (NEVER a set/map). Returns the new id. (For the root, pass `parent =
   kNoWidget`; `AddWidget` with `kNoWidget` parent just appends without linking — used once for the root.)
2. **`uint32_t ChildCount(const Tree& t, WidgetId w)`** — walk the sibling chain, count (a small helper for
   the digest + tests).
3. **`std::vector<uint8_t> EncodeTree(const Tree& t)`** — hand-LE, the GOLDEN bytes. FIXED order:
   `PutU32(widgetCount)`, `PutU32(root)`; then for each widget in index order: `PutU32(parent)`,
   `PutU32(firstChild)`, `PutU32(nextSibling)`, then the Style — `PutI32(width)`, `PutI32(height)`,
   `PutI32(marginL/T/R/B)` (4), `PutI32(padL/T/R/B)` (4), `PutU32(flexWeight)`, `PutU32(flags)` — then
   `PutU32(kind)`. (NEVER `DigestBytes` the `Widget`/`Style` struct — padding; field-by-field only.)
4. **`uint64_t DigestTree(const Tree& t)`** → `net::DigestBytes(EncodeTree(t).data(), EncodeTree(t).size())`.

### Fixture (deterministic, FIXED forever)
- `Tree MakeShowcaseTree()` — a fixed ~8-widget hierarchy: root (vertical stack, kind 0) → `header` (kFixedH,
  height 64) + `body` (kFlexGrow, weight 1) + `footer` (kFixedH, height 32); `body` → `left` (kFlexGrow,
  weight 1) + `right` (kFlexGrow, weight 2); `header` → `title` (kind 1 = text). Give each a distinct style
  (margins/padding) so the digest exercises the fields. (Keep FIXED forever — the golden pins it.)

## The golden (PINNED, cross-platform) — tests/widget_test.cpp
Self-contained test in the `seq_test.cpp` shape (copy the `check()` helper + `HF_TEST_MAIN_INIT()` from
`tests/test_main.h`). Register `hf_add_pure_test(widget_test)` in `tests/CMakeLists.txt` next to `seq_test`.
```
widget-s1: showcase tree digest = 0x<...>  (<N> widgets)
PASS widget-s1: DigestTree(MakeShowcaseTree()) == pinned uint64 (cross-platform tree anchor)
PASS widget-s1: re-encoding the same tree is byte-identical (deterministic)
PASS widget-s1: the tree shape is correct (root has 3 children; body has 2 children; header has 1)
PASS widget-s1: child order is insertion order (header before body before footer; left before right)
PASS widget-s1: a changed style field (a margin) changes the digest (style is load-bearing)
PASS widget-s1: a different tree shape (an extra child) changes the digest (hierarchy is load-bearing)
```
Assertions:
1. **PINNED TREE DIGEST** — `DigestTree(MakeShowcaseTree())` == a hard-pinned `uint64_t` (run once, pin THAT;
   identical MSVC + clang — the cross-platform anchor).
2. **DETERMINISTIC** — a second `EncodeTree` of the same tree → byte-identical (and digest equal).
3. **SHAPE** — `ChildCount(root) == 3`; `ChildCount(body) == 2`; `ChildCount(header) == 1`; walking
   `root.firstChild`→siblings yields header, body, footer in that order.
4. **CHILD ORDER** — the first child of `body` is `left`, its `nextSibling` is `right` (insertion order, not
   reversed).
5. **STYLE LOAD-BEARING** — clone the showcase, change one widget's `marginL`, re-digest → DIFFERENT.
6. **SHAPE LOAD-BEARING** — add one extra child to a widget → DIFFERENT digest.

## Cross-platform proof (the cheap loop — NO render-bake)
The CONTROLLER `scp`s `engine/ui/widget.h` + `engine/net/session.h` + `tests/widget_test.cpp` (+
`tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine
-I tests tests/widget_test.cpp -o /tmp/widget && /tmp/widget`, confirming the test PASSES with the IDENTICAL
pinned digest. (A local Windows clang at `C:\Program Files\LLVM\bin\clang++.exe` is the fast pre-check.) NO
Metal, NO `tests/golden/*`.

## Constraints (HARD)
- NEW header `engine/ui/widget.h` (existing dir `engine/ui/`); compiles STANDALONE under clang with
  `-I engine -I tests` (self-contained: only `<cstddef>/<cstdint>/<vector>` + `net/session.h`). Do NOT modify
  `net/session.h` / `ui/text.h` / any existing header. Do NOT include `replay.h`/`seq.h`/`flow.h` (inline the
  LE appenders). Do NOT add it to any RHI/GPU target.
- Pure-CPU INTEGER on the bit-exact path: NO float / `<cmath>` / clock / RNG / `<random>` / `<unordered_*>` /
  `<map>` / `<functional>` / `std::hash` / `<algorithm>` / `<string>`. The box model is `int32_t` pixels.
- Children are an ORDERED first-child/next-sibling chain (insertion order) — NEVER a set/map.
- `tests/widget_test.cpp` is SELF-CONTAINED (copy the scaffolding). Register `hf_add_pure_test(widget_test)`
  in `tests/CMakeLists.txt`. Use `test_main.h` `HF_TEST_MAIN_INIT()`.
- Branch `fix-widget-s1`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target widget_test'`
  (if the `cmd /c '...&&...'` handoff swallows output, run the build via the PowerShell tool natively). Run
  the test exe, confirm it PRINTS the digest and PASSES. First run: pick the pinned digest from the printed
  value, pin it, rebuild, confirm green. ALSO compile standalone with the local clang and confirm the
  IDENTICAL digest (MSVC==clang).
- COMPLETION CRITERIA — do NOT commit until: the header compiles, `widget_test` builds + PASSES on Windows
  with all assertions green, and the local clang standalone passes with the identical digest. Report: the
  commit hash, the full test output (printed digest + PASS lines), the exact pinned `uint64_t`, confirmation
  the header is self-contained (list its `#include`s — exactly 4: `<cstddef>/<cstdint>/<vector>` +
  `net/session.h`), the showcase tree you built, and the local-clang result. Commit message via a temp file +
  `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits the diff, runs the Mac/clang standalone to confirm the IDENTICAL digest, then
  ff-merges to master + pushes + deletes the branch + advances to S2 — the integer layout solver.)
