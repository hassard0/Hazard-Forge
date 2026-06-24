# Slice PT2 — Integer hydraulic erosion (Flagship #26 TERRAIN, 2nd slice — THE HEADLINE + CRUX)

The thing no offline tool ships: a **deterministic, bit-identical RUNTIME integer hydraulic erosion** — a
sediment-transport flux that moves material downhill, iterated, carving drainage valleys into the PT1 heightfield.
The moat: a seed + iteration count → the EXACT same eroded terrain on every machine (UE5 landscape erosion is an
offline non-deterministic bake). Strict **integer zero-diff cross-backend golden** (an erosion-delta heatmap).
THE CRUX is fixed-point stability — the design below guarantees no oscillation and EXACT mass conservation.

## The model (a mass-conserving, contractive integer flux)
Per iteration, in PINNED row-major (Gauss-Seidel) cell order — the `fpx::SolveContacts` order-determinism
discipline — for each cell: find its **lowest 4-neighbour** (within the grid), `dh = h[cell] - h[low]`. If `dh > 0`,
move `erode = fxmul(dh, kErodeRate)` of height **from the cell to that neighbour**: `h[cell] -= erode; h[low] +=
erode;`. Two guarantees that make this the crux-safe choice:
- **EXACT mass conservation** — the SAME `erode` value is subtracted from one cell and added to another, so the
  grid sum is preserved bit-for-bit (no within-tolerance fudge — assert `sum(eroded) == sum(base)` EXACTLY).
- **No oscillation / no inversion** — with `kErodeRate <= kOne/4`, after the move the gap is `dh - 2·erode = dh -
  2·(dh·rate)`; at `rate = kOne/8` the gap is `dh·(1 - 1/4) = 3·dh/4 > 0`, so the cell stays ABOVE its neighbour —
  the gradient strictly shrinks, never flips. Use `kErodeRate = kOne/8` (conservative, provably stable). Iterate a
  FIXED golden-stable count (e.g. 60). CPU-host (the int64 `fxmul` is Vulkan-DXC-only in shaders — keep erosion
  host-evaluated, golden-baked on Mac, the FO1–FO4 / FPX1 lesson). Truncation in `fxmul` is deterministic (every
  backend truncates `>>` identically) → bit-identical.

## The header — `engine/terrain/erosion.h` (NEW, header-only, namespace `hf::terrain`)
May `#include "sim/fpx.h"` (+ `"terrain/procterrain.h"` if convenient). Provide:
- `inline constexpr fx kErodeRate = kOne / 8;` (the provably-stable rate).
- `inline void ErodeHydraulic(std::vector<fx>& grid, int n, int iterations)`: the model above — `iterations`
  passes, each a PINNED row-major sweep; for each cell scan its 4 in-grid neighbours for the lowest, and if the
  cell is higher, move `fxmul(dh, kErodeRate)` from the cell to that neighbour (in place, so later cells in the
  sweep see the updated values — Gauss-Seidel). `iterations <= 0` → no change (the no-op). Pure integer, NO
  `<cmath>`, NO float, NO clock/RNG.
- (Optional) `inline fx GridSum(const std::vector<fx>& grid)` (an int64 accumulate, for the mass-conservation
  proof) and `inline int CountChanged(const std::vector<fx>& a, const std::vector<fx>& b)`.

## CPU test — extend `tests/procterrain_test.cpp` (add a PT2 section, keep PT1 checks)
Assertions: (1) **EXACT mass conservation (make-or-break)** — `sum(grid)` after `ErodeHydraulic` equals
`sum(grid)` before, bit-for-bit (the flux moves material, never creates/destroys it — use an int64 sum); (2)
**no inversion / stable** — after erosion, no cell is driven BELOW a neighbour it eroded toward beyond the gap
bound (or more simply: assert the per-iteration max-gradient is non-increasing, OR that the grid stays bounded
well inside ±32768·kOne — the contraction holds); (3) **zero-iters no-op** — `iterations <= 0` → the grid is
unchanged; (4) **carves** — after a real erosion the grid DIFFERS from the base (a non-trivial changed-cell count),
and the high cells lowered / low cells raised (peaks erode, valleys fill — sample a known peak and assert it
dropped); (5) **replay-stable** — two `ErodeHydraulic` runs from the same base are byte-equal. Print +
`procterrain_test: ALL CHECKS PASSED`.

## Showcase — `--pt2-hydraulic-shot` (Vulkan, main.cpp) + `--pt2-hydraulic` (Metal, visual_test.mm)
A **2D erosion-delta heatmap**: `GenHeightField` (PT1, same fixed seed/n/octaves) → copy → `ErodeHydraulic(copy,
n, iters)` → per pixel the SIGNED delta `d = eroded[i] - base[i]` colored by PURE INTEGER (positive/deposited →
green ramp, negative/eroded → blue ramp, magnitude `clamp(|d| >> shift, 0, 255)`; NO float pixel math). Fixed
seed/n/octaves/iters. SAME params IN BOTH renderers → byte-identical cross-backend by construction. (The heatmap
shows material moved from ridges to valleys — the carving.)

## Proof (STRICT integer — zero-diff cross-backend)
```
pt2-hydraulic: integer hydraulic erosion (iters=<K>, n=256, rate=kOne/8)
pt2-hydraulic: two-run BYTE-IDENTICAL
pt2-hydraulic: mass conserved EXACT {baseSum:<S>, erodedSum:<S>, delta:0}
pt2-hydraulic: zero-iters -> unchanged (no-op) {changed:0}
pt2-hydraulic: carves -> material moved {changedCells:<C>, peakDropped:true}
```
Assertions: (1) two runs byte-identical; (2) mass conserved EXACTLY (delta 0); (3) zero-iters → unchanged; (4) a
real erosion changed a non-trivial number of cells AND a sampled peak dropped (the carving is real). Register
`pt2_hydraulic` in verify.ps1 $Goldens (Flag `--pt2-hydraulic`) + `--pt2-hydraulic-shot` in $vkShots, mirroring
`pt1_height`.

## Constraints (HARD)
- NEW engine/terrain/erosion.h + extend tests/procterrain_test.cpp + the showcase blocks + verify.ps1
  registration. Reuse fpx.h/procterrain.h READ-ONLY. Do NOT modify heightmap.{h,cpp}/procterrain.h(PT1 logic)/any
  existing shader/golden. NO new RHI, NO new shader. Pure-CPU integer.
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (the controller bakes Metal and
  requires the Metal image == the Vulkan image byte-for-byte). The heatmap MUST be pure integer (no float pixel
  math). Do NOT route through a GPU float raster. Keep erosion CPU-host (no shader).
- Branch `fix-terrain-pt2`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle procterrain_test'`
  Run procterrain_test.exe (ALL CHECKS PASSED) + hello_triangle.exe --pt2-hydraulic-shot <out.bmp> (exit 0 + proof lines).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, procterrain_test passes, `--pt2-hydraulic-shot`
  exits 0, the proof lines print, two-run byte-identical, mass conserved EXACTLY (delta 0), the zero-iters no-op
  holds, the carving is real (changed cells + peak dropped), the plot is pure-integer. Commit message via a temp
  file + `git commit -F` (use the Bash tool heredoc). Commit to the branch and STOP. Report: commit hash, the proof
  output, image path, confirmation both renderers use identical params, and that the plot is pure-integer. (The
  CONTROLLER bakes the Metal golden, confirms zero-diff cross-backend, eyeballs the erosion-delta heatmap.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
