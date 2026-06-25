# Slice WIDGET-S6 — Render capstone: the labeled-layout money-shot (Issue #30)

S1–S5 built + PROVED the deterministic UI core (tree, layout, binding, animation, input/scrub — all
bit-identical + replayable). S6 is the visual capstone: render the computed layout as a lit 2D overlay — each
widget drawn as a **text label at its bit-exact computed pixel rect** — through the EXISTING `ui/text.h`
screen-space text pipeline (the `RunHudShowcase` path), no new shader, no new RHI. The float crossing (pixel
rect → NDC) is isolated in a separate `engine/ui/widget_render.h` (the `seq_render.h`/`econ_render.h`
precedent); the bit-exact integer layout (S2) is the real golden, the rendered pixels the documented
float visresolve-bar.

Append-only — `widget.h` is UNTOUCHED (S6 adds a NEW `widget_render.h` + a CPU provenance test + the
`--ui-shot` showcases). All S1–S5 pinned digests stay.

## STEP 0 (DO FIRST, REPORT IT): study the text-overlay wiring
Read how the HUD text overlay is wired, so `--ui-shot` replicates it:
- `metal_headless/visual_test.mm` `RunHudShowcase` (~line 6834) + the `--hud` dispatch (~74182): how it calls
  `ui::LayoutText` (~6993) and draws the text quads through the overlay pipeline + writes the PNG.
- The Vulkan side: grep `hello_triangle/main.cpp` (or wherever) for the `--hud`/text-overlay equivalent — the
  file that hosts the Vulkan `--*-shot`s (the seq/asset render-shots used `hello_triangle`).
- `engine/ui/text.h`: `LayoutText(s, originX, originY, pxScale, screenW, screenH, verts)` → `TextVertex`
  (NDC, +Y-down) + `BuildFontAtlas`.
Report the exact files/functions + the overlay draw path (atlas texture + the alpha-blend text pipeline).

## 1. NEW file: engine/ui/widget_render.h (namespace hf::ui) — the float bridge
`#include "ui/widget.h"` + `#include "ui/text.h"` + `#include "math/math.h"` (read-only). Header-only. NOT on
the bit-exact standalone path (it does float NDC + reuses the float `LayoutText`).
```cpp
// A render label: a widget's name placed at its computed rect (pixel-space; LayoutText does the NDC convert).
struct WidgetLabel { uint32_t widgetId; int32_t pxX, pxY; int32_t pxScale; /* the text is supplied by the caller's name table */ };

// Build one label per widget at its rect's top-left pixel origin (+ a small inset/padding). PURE function of
// (tree, rects) — deterministic, no RNG/clock; the label pixel positions ARE the bit-exact Rect origins (the
// provenance). Empty tree -> empty. (The caller maps widgetId -> a display string; this header stays
// string-free / clang-light by emitting positions + ids, not text.)
inline std::vector<WidgetLabel> WidgetTreeToLabels(const Tree& t, const std::vector<Rect>& rects, int32_t pxScale);

// DigestLabels: hand-LE (widgetId, pxX, pxY, pxScale) per label -> net::DigestBytes (the provenance digest —
// the label placements derive byte-for-byte from the bit-exact rects). (Reuse widget.h's PutU32/PutI32.)
inline uint64_t DigestLabels(const std::vector<WidgetLabel>& labels);
```
(The actual NDC conversion + glyph quads happen in the `--ui-shot` showcase via `ui::LayoutText(name, label.pxX,
label.pxY, label.pxScale, screenW, screenH, verts)` — the existing float text path. The header's job is the
deterministic placement; the showcase does the float draw.)

## 2. CPU provenance test (the cheap golden) — tests/widget_render_test.cpp
Self-contained, register `hf_add_pure_test(widget_render_test)`. It includes `ui/widget_render.h`. This is the
deterministic proof (no Metal):
```
widget-render: labels digest = 0x<...>  (<N> labels)
PASS widget-render: WidgetTreeToLabels(MakeLayoutShowcase, SolveLayout) has one label per widget (provenance count)
PASS widget-render: each label's pixel position == the bit-exact Rect origin (provenance — labels derive from layout)
PASS widget-render: DigestLabels == pinned uint64 (the placement is byte-stable cross-platform)
PASS widget-render: the layout is still bit-exact — DigestRects(SolveLayout(showcase)) == 0x95da64c52733eb16 (S2 unchanged)
PASS widget-render: two builds are byte-identical (deterministic)
```
Assertions:
1. **COUNT** — `WidgetTreeToLabels(...)` yields exactly `tree.widgets.size()` labels (one per widget).
2. **PROVENANCE** — for each label, `pxX == rects[widgetId].x + inset` and `pxY == rects[widgetId].y + inset`
   (the placement IS the bit-exact rect origin; assert against the S2 `SolveLayout` rects).
3. **PINNED LABELS DIGEST** — `DigestLabels(WidgetTreeToLabels(MakeLayoutShowcase(), SolveLayout(...), pxScale))`
   == a hard-pinned `uint64_t` (identical MSVC + clang).
4. **S2 STILL EXACT** — `DigestRects(SolveLayout(MakeLayoutShowcase(), {0,0,1280,720})) == 0x95da64c52733eb16`
   (the render bridge did NOT perturb the integer layout).
5. **DETERMINISTIC** — two `WidgetTreeToLabels` → identical `DigestLabels`.

## 3. The --ui-shot showcase (BOTH backends — the metal-showcase gate)
- **Vulkan `--ui-shot`** (in the file hosting the other `--*-shot`s): build `MakeLayoutShowcase()`, `SolveLayout`,
  `WidgetTreeToLabels`; for each label draw its widget name (a fixed name per widget id — "root"/"header"/
  "body"/"footer"/"left"/"right"/"title") via `ui::LayoutText` at the label's pixel position, through the
  EXISTING text-overlay pipeline (the `RunHudShowcase` path) over a cleared background; write the PNG. Print
  the FOUR proofs.
- **Metal `--ui-shot`** in `visual_test.mm` (next to `--hud`): the SAME scene (same `MakeLayoutShowcase` +
  `SolveLayout` + `WidgetTreeToLabels` + the SAME names + `LayoutText`), baking to the golden path. ⚠️ DO NOT
  OMIT — the controller greps `visual_test.mm` for `"--ui-shot"` before the Mac bake.
- **introspect.cpp**: a `{"--ui-shot", "Deterministic UMG-class UI LIT 2D RENDER CAPSTONE …"}` help entry.

### The FOUR proofs (exact lines)
1. `ui-render: {widgets:<N>, labels:<N>} from bit-exact layout` (provenance: labels == widgets).
2. `ui-render determinism: two builds BYTE-IDENTICAL` (two `WidgetTreeToLabels` byte-equal; Metal two-run
   render DIFF 0.0000).
3. `ui-render provenance: DigestRects == 0x95da64c52733eb16` (the rendered layout IS the bit-exact S2 layout).
4. `ui-render empty: base only (no-op)` (empty tree → no labels → cleared base).

## Cross-platform proof (the render-bake — CONTROLLER does this on the Mac)
The implementer's pre-bake proof: `widget_render_test` green on MSVC + local clang (the pinned labels digest +
provenance), AND the Vulkan `--ui-shot` runs + prints the four proofs. The CONTROLLER then on the Mac: greps
`visual_test.mm` for `"--ui-shot"`, builds `visual_test` (`cmake -S metal_headless -B build-metal -G Ninja &&
cmake --build build-metal --target visual_test`), runs `--ui-shot` TWICE (asserts the two PNGs byte-identical
= determinism), bakes `tests/golden/metal/ui_render.png`, and records the Vulkan-vs-Metal cross-vendor mean
(the documented float baseline — text rasterization differs sub-pixel cross-vendor). NO strict pixel-hash
golden (float slice).

## Constraints (HARD)
- NEW `engine/ui/widget_render.h` (render-only float bridge). Do NOT modify `engine/ui/widget.h` (S1–S5
  untouched — all pinned digests stay). Do NOT modify `ui/text.h`.
- NO new shader, NO new RHI — reuse the EXISTING text-overlay pipeline VERBATIM (the `--hud` path).
- `WidgetTreeToLabels`/`DigestLabels` are pure deterministic integer (the placements are integer pixels); the
  float crossing is ONLY the `LayoutText` NDC convert in the showcase. The layout digest (S2
  `0x95da64c52733eb16`) is the real golden; the rendered pixels are the documented visresolve-bar.
- WIRE BOTH `--ui-shot` showcases (Vulkan + Metal). Do NOT omit the Metal one.
- `tests/widget_render_test.cpp` self-contained; register `hf_add_pure_test(widget_render_test)`. Keep
  `widget_test` (S1–S5) green. Do NOT commit any `tests/golden/*` (the CONTROLLER bakes the Metal golden).
- Branch `fix-widget-s6`, commit there, do NOT merge.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target widget_render_test'`
  (and the visual-test target hosting `--ui-shot` — report its name; if the `cmd /c '...&&...'` handoff
  swallows output, use the PowerShell tool natively). Run `widget_render_test` + the Vulkan `--ui-shot`,
  confirm the four proofs + the pure test passes. ALSO local clang standalone for `widget_render_test`
  (identical). Do NOT attempt a Metal build on Windows.
- COMPLETION CRITERIA — do NOT commit until: `widget_render.h` compiles, `widget_render_test` builds + PASSES
  on Windows (+ local clang, identical labels digest), the Vulkan `--ui-shot` runs + prints the four proofs,
  and `widget_test` (S1–S5) is still green. Report: the STEP-0 wiring findings (the Vulkan + Metal showcase
  files/functions + the text-overlay path), commit hash, the `widget_render_test` output, the Vulkan
  `--ui-shot` four-proof output, confirmation the Metal `--ui-shot` IS wired in visual_test.mm (quote the
  `strcmp` line), confirmation `widget.h` is untouched + all S1–S5 digests still green, confirmation NO
  `tests/golden/*` committed + the visual-test target name, the local-clang result, and any deviation flagged.
  Commit message via a temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER then: greps visual_test.mm for `--ui-shot`, audits the bridge, runs the Mac render bake
  [build visual_test, run `--ui-shot` twice for determinism, bake the golden, record the cross-vendor mean],
  commits the golden, ff-merges to master + pushes + deletes the branch, then writes the ARCHITECTURE.md UI
  section + comments issue #30 — COMPLETING flagship #30.)
