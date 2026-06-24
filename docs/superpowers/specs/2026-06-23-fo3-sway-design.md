# Slice FO3 — Per-instance wind sway (Issue #21, flagship #25 FOLIAGE, 3rd slice — THE DETERMINISM HEADLINE)

The bridge that ties the deterministic wind (FO1) to the placed foliage (FO2): `ApplyWind` evaluates the integer
wind field at each plant's position and stores a per-instance Q16.16 **bend** — so the whole meadow sways as a
pure function of (seed, frame), bit-identical cross-platform. This is the headline: the swaying *data* is a strict
integer golden (no float anywhere), the thing UE5's float wind cannot make deterministic. Builds on FO1+FO2.
**STRICT-integer zero-diff cross-backend golden** (a pure-integer bent-stalk plot).

## The addition — `engine/foliage/foliage.h` (APPEND-ONLY after the FO2 block)
Add to `hf::foliage` (do NOT modify FO1/FO2; append a field to `FoliageInstance` AFTER `base`):
- Extend `struct FoliageInstance { hf::pcg::PcgInstance base; fx bend = 0; };` — the per-instance Q16.16 bend
  angle (0 = upright). (Append `bend` after `base`; FO2's `PlaceFoliage` leaves it default-0, so FO2 stays
  byte-unchanged in behaviour.)
- `inline void ApplyWind(std::vector<FoliageInstance>& instances, const WindField& wind, uint32_t frame)`: for each
  plant in FIXED order, `inst.bend = WindBend(wind, inst.base.pos, frame)`. Mutates the bend in place; the base
  placement is untouched (wind only annotates a sway, never moves the plant). Pure integer (reuses FO1 `WindBend`).
  NO `<cmath>`, NO float.

## Showcase — `--fo3-sway-shot` (Vulkan, main.cpp) + `--fo3-sway` (Metal, visual_test.mm)
A **2D top-down strict-integer plot of bent stalks**: take the FO2 meadow (`PlaceFoliage` with the same fixed
field/seed), `ApplyWind(instances, wind, frame)` with the FO1 wind, then draw each plant as a **bent stalk** — a
pure-integer line from the plant's pixel base to a top that LEANS by the bend. Small-angle pure-integer lean (NO
trig): `leanPx = (int)(((int64_t)inst.bend * kStalkLen) >> 16)` (Q16.16 bend → pixel lean; `kStalkLen` a fixed
pixel stalk length), top pixel = `(baseX + leanPx, baseY - kStalkLen)`; draw an integer line (Bresenham) base→top +
a dot at the top. Plants lean by their LOCAL wind bend → the meadow shows a coherent sway pattern matching the FO1
wind field. Reuse the FO2/`pcg4-rules`/`pcg5-graph` pure-integer marker/line code; NO float pixel math. SAME
field/seed/wind/frame/image-size IN BOTH renderers → byte-identical cross-backend by construction.

## Proof (STRICT integer — zero-diff cross-backend)
```
fo3-sway: per-instance wind sway (plants=<K>, frame=<F>)
fo3-sway: two-run BYTE-IDENTICAL
fo3-sway: frame N vs N+1 -> swept differently {frameA:0x<Ha>, frameB:0x<Hb>} Ha != Hb
fo3-sway: zero-wind -> all bends 0 (upright, no-op) {maxBend:0, hash:0x<H0>}
fo3-sway: provenance {plants:<K>, frame:<F>}
```
Assertions: (1) two runs byte-identical; (2) frame F vs F+1 → different image checksum (the meadow sways frame to
frame); (3) zero-wind (`master=0` or all `amp=0`) → every `inst.bend == 0` (all stalks vertical — the no-op
control; assert maxBend 0 AND the zero-wind image is the deterministic upright field); (4) provenance + coherent
(non-zero bends present under real wind). Register `fo3_sway` in verify.ps1 $Goldens (Flag `--fo3-sway`) +
`--fo3-sway-shot` in $vkShots, mirroring `fo2_place`.

## Constraints (HARD)
- APPEND-ONLY to engine/foliage/foliage.h (append `bend` to FoliageInstance after `base`; add `ApplyWind`; do NOT
  modify FO1/FO2 logic) + the showcase blocks + verify.ps1 registration. Reuse FO1 `WindBend` + pcg/fpx READ-ONLY.
  Do NOT modify pcg.h/mixer/any existing shader/golden. NO new RHI, NO new shader. Pure-CPU integer.
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (the controller bakes Metal and
  requires the Metal image == the Vulkan image byte-for-byte). The 2D plot MUST be pure integer (the small-angle
  lean is `bend * len >> 16`, integer Bresenham — NO float, NO trig). Do NOT route through a GPU float raster.
- Branch `fix-issue-21-fo3`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  (+ foliage_test if you extend it with an ApplyWind determinism / zero-wind check). Run `hello_triangle.exe
  --fo3-sway-shot <out.bmp>`, confirm exit 0 + proof lines.
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--fo3-sway-shot` exits 0, the proof lines print,
  two-run byte-identical, the frame-sensitivity holds, the zero-wind no-op holds (maxBend 0, all upright), the plot
  is pure-integer. Commit message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the
  branch and STOP. Report: commit hash, the proof output, image path, confirmation both renderers use identical
  field/seed/wind/frame/size, and that the plot is pure-integer. (The CONTROLLER bakes the Metal golden, confirms
  zero-diff cross-backend, eyeballs the swept meadow.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
