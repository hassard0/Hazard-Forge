# Slice PT3 — Integer thermal erosion / slope-slump (Flagship #26 TERRAIN, 3rd slice)

The complement to hydraulic carving (PT2): **thermal erosion** — where a slope exceeds the **talus angle** (angle
of repose), material slumps downslope until the slope settles to the talus, producing scree/talus slopes. Like PT2
it is a mass-conserving, contractive integer flux (the same crux discipline), but it only acts where the slope is
TOO STEEP and brings it toward the talus threshold. Composes with PT2. Strict **integer zero-diff cross-backend
golden** (a slope-slump heatmap).

## The model (excess-above-talus slump, mass-conserving, settles to the angle of repose)
Per iteration, in PINNED row-major (Gauss-Seidel) order, for each cell: find its **lowest 4-neighbour**, `dh =
h[cell] - h[low]`. If `dh > talus` (the slope exceeds the angle of repose), move the EXCESS toward the neighbour:
`slump = fxmul(dh - talus, kSlumpRate)` from the cell to that neighbour (`h[cell] -= slump; h[low] += slump;`).
Two guarantees (the PT2 crux discipline):
- **EXACT mass conservation** — the SAME `slump` is subtracted from one cell and added to another → the grid sum is
  preserved bit-for-bit (assert delta 0).
- **Settles to talus, no inversion** — with `kSlumpRate = kOne/2`, the new gap is `dh - 2·slump = dh - (dh - talus)
  = talus` → the pair is brought to EXACTLY the talus angle in one step (and `talus >= 0` so no inversion). Use
  `kSlumpRate = kOne/2`. Iterated a FIXED count, the whole field settles so NO slope exceeds talus (the
  angle-of-repose guarantee). CPU-host (no shader; the FO1–FO4/FPX1 lesson). Truncation is deterministic → bit-identical.

## The addition — `engine/terrain/erosion.h` (APPEND-ONLY after the PT2 block)
Add to `hf::terrain` (do NOT modify PT2's `ErodeHydraulic`/`kErodeRate`):
- `inline constexpr fx kSlumpRate = kOne / 2;`
- `inline void ErodeThermal(std::vector<fx>& grid, int n, int iterations, fx talus)`: the model above — `iterations`
  PINNED row-major sweeps; per cell find the lowest 4-neighbour, and if `dh = h[cell] - h[low] > talus`, move
  `fxmul(dh - talus, kSlumpRate)` from the cell to that neighbour (in place, Gauss-Seidel). `iterations <= 0` → no
  change (no-op). Pure integer, NO `<cmath>`, NO float, NO clock/RNG.
- (Optional) `inline fx MaxSlope(const std::vector<fx>& grid, int n)` — the max over all cells of `(h[cell] -
  lowest 4-neighbour)` (a Q16.16 slope proxy), for the angle-of-repose proof.

## CPU test — extend `tests/procterrain_test.cpp` (add a PT3 section, keep PT1/PT2)
Assertions: (1) **EXACT mass conservation** — int64 `sum(grid)` after `ErodeThermal` == before (delta 0); (2)
**angle-of-repose (make-or-break)** — `MaxSlope` AFTER a settled `ErodeThermal(grid, n, iters, talus)` is `<=
talus` (no slope steeper than the talus angle remains — pick `iters` large enough that the settle completes; if a
single strict `<=` is borderline, assert `MaxSlope_after <= talus` AND `MaxSlope_after < MaxSlope_before`, i.e. the
field was brought to the repose angle); (3) **zero-iters no-op** — `iterations <= 0` → unchanged; (4)
**composes with hydraulic** — `ErodeHydraulic` then `ErodeThermal` differs from `ErodeHydraulic` alone (thermal did
work on the hydraulic-eroded field); (5) **replay-stable** — two runs byte-equal. Print + `procterrain_test: ALL
CHECKS PASSED`.

## Showcase — `--pt3-thermal-shot` (Vulkan, main.cpp) + `--pt3-thermal` (Metal, visual_test.mm)
A **2D slope-slump heatmap**: `GenHeightField` (PT1, same fixed seed/n/octaves) → copy → `ErodeThermal(copy, n,
iters, talus)` → per pixel the SIGNED delta `d = slumped[i] - base[i]` colored by PURE INTEGER (positive/deposited
→ a warm ramp, negative/slumped → a cool ramp, magnitude `clamp(|d| >> shift, 0, 255)`; NO float pixel math) — the
slump pattern concentrates on the steep slopes. Fixed seed/n/octaves/iters/talus. SAME params IN BOTH renderers →
byte-identical cross-backend by construction.

## Proof (STRICT integer — zero-diff cross-backend)
```
pt3-thermal: integer thermal slump (talus=<T>, iters=<K>, n=256)
pt3-thermal: two-run BYTE-IDENTICAL
pt3-thermal: mass conserved EXACT {delta:0}
pt3-thermal: angle of repose -> max slope <= talus {before:<Sb>, after:<Sa>, talus:<T>, ok:true}
pt3-thermal: zero-iters -> unchanged (no-op) {changed:0}
pt3-thermal: composes with hydraulic (combined != hydraulic-alone) {ok:true}
```
Assertions: (1) two runs byte-identical; (2) mass conserved EXACTLY (delta 0); (3) after settle, max slope `<=
talus` (the angle-of-repose guarantee — slopes capped); (4) zero-iters → unchanged; (5) thermal-after-hydraulic
differs from hydraulic alone. Register `pt3_thermal` in verify.ps1 $Goldens (Flag `--pt3-thermal`) +
`--pt3-thermal-shot` in $vkShots, mirroring `pt2_hydraulic`.

## Constraints (HARD)
- APPEND-ONLY to engine/terrain/erosion.h (do NOT modify PT2) + extend tests/procterrain_test.cpp + the showcase
  blocks + verify.ps1 registration. Reuse fpx.h/procterrain.h/PT2 READ-ONLY. Do NOT modify heightmap.{h,cpp}/
  procterrain.h/any existing shader/golden. NO new RHI, NO new shader. Pure-CPU integer (CPU-host erosion).
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (the controller bakes Metal and
  requires the Metal image == the Vulkan image byte-for-byte). The heatmap MUST be pure integer (no float pixel
  math). Do NOT route through a GPU float raster.
- Branch `fix-terrain-pt3`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle procterrain_test'`
  Run procterrain_test.exe + hello_triangle.exe --pt3-thermal-shot <out.bmp> (exit 0 + proof lines).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, procterrain_test passes, `--pt3-thermal-shot`
  exits 0, the proof lines print, two-run byte-identical, mass conserved EXACTLY (delta 0), the angle-of-repose
  holds (max slope <= talus after settle), the zero-iters no-op holds, composes-with-hydraulic holds, the plot is
  pure-integer. Commit message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the branch
  and STOP. Report: commit hash, the proof output, image path, confirmation both renderers use identical params,
  and that the plot is pure-integer. (The CONTROLLER bakes the Metal golden, confirms zero-diff cross-backend,
  eyeballs the slump heatmap.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
