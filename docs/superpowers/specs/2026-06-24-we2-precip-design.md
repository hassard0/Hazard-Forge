# Slice WE2 — Deterministic integer precipitation field (Flagship #27 WEATHER, 2nd slice)

Rain/snow as a **deterministic integer streak field**: each drop's position is a pure function of (seed, frame) —
the drop falls and wraps with NO accumulator state, so two peers see the byte-identical rain. Builds on WE1.
**STRICT-integer zero-diff cross-backend golden** (a CPU-rendered integer precip plot). The existing float
`particles.h` is LCG/float-dt and non-portable — WE2 uses the deterministic-scatter precedent instead.

## The addition — `engine/weather/weather.h` (APPEND-ONLY after the WE1 block)
Add to `hf::weather` (do NOT modify WE1; `#include "pcg/pcg.h"` for `PcgRand01`):
- `struct PrecipField { uint32_t seed = 0; int count = 0; fx areaW = kOne; fx areaD = kOne; fx columnH = kOne; fx fallSpeed = kOne; };`
  (a drop count, the XZ scatter extent, the vertical column height, and the per-frame fall speed in Q16.16).
- `inline FxVec3 PrecipDrop(const PrecipField& p, uint32_t i, uint32_t frame)`: drop `i`'s world position at
  `frame`. `x = PcgRandRange(p.seed, i*3+0, 0, p.areaW); z = PcgRandRange(p.seed, i*3+2, 0, p.areaD);` (a fixed
  scatter); the START phase `y0 = PcgRandRange(p.seed, i*3+1, 0, p.columnH);` (each drop starts at a deterministic
  height). The Y at `frame`: `int64_t yRaw = (int64_t)y0 - (int64_t)p.fallSpeed * frame;` then wrap into
  `[0, columnH)` with a POSITIVE integer modulo: `fx y = (fx)(((yRaw % p.columnH) + p.columnH) % p.columnH);` (the
  drop falls by `fallSpeed` per frame and wraps to the top — pure integer, frame the ONLY time input, NO
  accumulator). Return `{x, y, z}`. Pure integer, NO `<cmath>`, NO float, NO clock/RNG.
- `inline std::vector<FxVec3> GenPrecip(const PrecipField& p, uint32_t frame)`: `PrecipDrop(p, i, frame)` for
  `i ∈ [0, p.count)` in order. Return the drops. `count <= 0` → empty (the no-op). Pure integer.

## CPU test — extend `tests/weather_test.cpp` (add a WE2 section, keep WE1 checks)
Assertions: (1) **replay-stable** — `GenPrecip` identical across calls for the same args; (2) **rain falls** —
`GenPrecip(p, F)` != `GenPrecip(p, F+1)` (every drop descended by `fallSpeed`, modulo wrap — assert each drop's Y
at F+1 equals the wrapped `Y(F) - fallSpeed`); (3) **bounds** — every drop's `x ∈ [0, areaW)`, `z ∈ [0, areaD)`,
`y ∈ [0, columnH)` (the wrap keeps drops in the column); (4) **no-op** — `count <= 0` → 0 drops; (5) **wrap
continuity** — a drop near the bottom wraps to near the top across the frame boundary (no negative Y, no escape).
Print + `weather_test: ALL CHECKS PASSED`.

## Showcase — `--we2-precip-shot` (Vulkan, main.cpp) + `--we2-precip` (Metal, visual_test.mm)
A **2D SIDE-VIEW integer plot of the rain** (X horizontal, Y vertical — so the falling reads): `GenPrecip(field,
frame)`, map each drop's `(x, y)` to an INTEGER pixel coord (X over `[0,areaW)`, Y over `[0,columnH)`, the column
shown vertically) and draw a short **vertical streak** (a few pixels tall — the rain motion blur) per drop via pure
integer line/marker code (reuse the FO/PCG pure-integer marker code). 256×256. Fixed seed/count/area/columnH/
fallSpeed/frame. SAME params IN BOTH renderers → byte-identical cross-backend by construction. (The plot shows
streaks descending the column; at the next frame they've moved down.)

## Proof (STRICT integer — zero-diff cross-backend)
```
we2-precip: integer precipitation field (drops=<K>, fallSpeed=<S>, frame=<F>)
we2-precip: two-run BYTE-IDENTICAL
we2-precip: rain falls -> frame F vs F+1 descended {frameA:0x<Ha>, frameB:0x<Hb>} Ha != Hb
we2-precip: zero-count -> clear (no-op) {drops:0}
we2-precip: provenance {drops:<K>, fallSpeed:<S>, frame:<F>}
```
Assertions: (1) two runs byte-identical; (2) frame F vs F+1 → a different image checksum (the rain DESCENDED); (3)
`count <= 0` → 0 drops (clear, the no-op); (4) provenance + coherent (drops present under a real count). Register
`we2_precip` in verify.ps1 $Goldens (Flag `--we2-precip`) + `--we2-precip-shot` in $vkShots, mirroring `we1_clouddensity`.

## Constraints (HARD)
- APPEND-ONLY to engine/weather/weather.h (do NOT modify WE1) + extend tests/weather_test.cpp + the showcase blocks
  + verify.ps1 registration. Reuse fpx.h/pcg.h READ-ONLY. Do NOT modify particles.h/clouds.h/procterrain.h/any
  existing shader/golden. NO new RHI, NO new shader. Pure-CPU integer.
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (the controller bakes Metal and
  requires the Metal image == the Vulkan image byte-for-byte). The plot MUST be pure integer (no float pixel math).
  Do NOT route through a GPU float raster.
- Branch `fix-weather-we2`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows + the test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle weather_test'`
  Run weather_test.exe (ALL CHECKS PASSED) + hello_triangle.exe --we2-precip-shot <out.bmp> (exit 0 + proof lines).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, weather_test passes, `--we2-precip-shot` exits 0,
  the proof lines print, two-run byte-identical, the rain-falls (frame-sensitivity) holds, the zero-count no-op
  holds, the plot is pure-integer. Commit message via a temp file + `git commit -F` (use the Bash tool heredoc).
  Commit to the branch and STOP. Report: commit hash, the proof output, image path, confirmation both renderers use
  identical params, and that the plot is pure-integer. (The CONTROLLER bakes the Metal golden, confirms zero-diff
  cross-backend, eyeballs the falling rain.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
