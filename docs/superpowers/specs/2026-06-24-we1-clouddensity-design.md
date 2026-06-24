# Slice WE1 — Deterministic integer drifting cloud-density field (Flagship #27 WEATHER, beachhead)

The irreducible primitive: a **deterministic integer cloud-density field** that DRIFTS with the frame — `IntCloudDensity(x,z,frame,...)`
returns a Q16.16 density in `[0,kOne]` from integer fBm value-noise, advected by an integer wind offset that grows
with the frame counter, bit-identical CPU↔Vulkan↔Metal BY CONSTRUCTION (NO runtime `sin`/`frac(sin())`/float —
unlike the existing float `clouds.h`). The moat: clouds drift as a pure function of (seed, frame), so two peers see
the byte-identical cloudscape (UE5 volumetric clouds are float/clock-driven). Establishes the NEW `engine/weather/weather.h`
+ `hf::weather`. **STRICT-integer zero-diff cross-backend golden** (a CPU-rendered integer cloud-density slab cut,
the FO1/PT1 precedent). NO GPU/shader (host-evaluated; the float `clouds.frag` raymarch comes at WE4).

## Do NOT use clouds.h as the data source
`engine/render/clouds.h` is a FLOAT cpu-mirror-of-shader cumulus lib (`Hash3 = frac(sin(...))`, `std::sin`/`exp` on
the hot path) designed to be copied into `clouds.frag.hlsl` — it is the **float RENDER bridge** this flagship reuses
at WE4, NOT the strict-integer data source. WE1 builds the deterministic integer DATA in a NEW header `weather.h`.

## The header — `engine/weather/weather.h` (NEW, header-only, namespace `hf::weather`)
This is a render-flagship data slice (the golden is a Mac-BAKED image), so weather.h MAY `#include "sim/fpx.h"` +
`"terrain/procterrain.h"` (reuse the integer noise basis read-only). Provide:
- `inline constexpr fx kCloudDriftRate = <a small Q16.16 per-frame X offset, e.g. kOne/4>;` (the deterministic
  drift speed — clouds translate `kCloudDriftRate` world units per frame in +X).
- `inline fx IntCloudDensity(fx x, fx z, uint32_t frame, uint32_t seed, fx coverage, int octaves)`: compute a
  drifted fBm value-noise then carve it by coverage into a density:
  - `driftX = (fx)((int64_t)kCloudDriftRate * frame);` (a Q16.16 offset growing with `frame` — pure integer, frame
    is the ONLY time input, NO clock). Sample the noise at `(x + driftX, z)`.
  - `nrm` = the fBm at `(x+driftX, z)` NORMALIZED to `[0, kOne)` (reuse `terrain::IntHeight(x+driftX, z, octaves,
    seed)` and divide by its octave-sum bound, OR sum `terrain::IntValueNoise` octaves yourself with halving
    amplitude and normalize — the result must be a Q16.16 in `[0,kOne)`).
  - **Coverage carve:** `threshold = kOne - coverage;` `fx d = nrm - threshold;` `if (d < 0) d = 0;` then rescale to
    `[0,kOne]` (e.g. `d = (coverage > 0) ? clamp(fxdiv(d, coverage), 0, kOne) : 0`). So `coverage = kOne` → density
    `= nrm` (full); `coverage = 0` → `threshold = kOne` → `d = nrm - kOne <= 0` → density `0` EVERYWHERE (the
    no-op). Return the Q16.16 density in `[0,kOne]`. Pure integer, NO `<cmath>`, NO float, NO clock/RNG.
- `inline std::vector<fx> GenCloudSlice(uint32_t seed, int n, fx worldSize, uint32_t frame, fx coverage, int octaves)`:
  sample `IntCloudDensity` over an `n×n` horizontal slab cut over `[0, worldSize)` (the `GenHeightField` shape).
  Return the `n*n` grid. Pure integer.

## CPU test — `tests/weather_test.cpp` (NEW, register `hf_add_pure_test(weather_test)` in tests/CMakeLists.txt)
Assertions: (1) **replay-stable** — `IntCloudDensity`/`GenCloudSlice` identical across calls for the same args;
(2) **drift** — `GenCloudSlice(...,frame=F)` != `GenCloudSlice(...,frame=F+1)` (the field DRIFTED — the clouds
move); (3) **bounds** — every density `∈ [0, kOne]`; (4) **zero-coverage no-op** — `coverage = 0` → every density
`0` (maxDensity 0); (5) **coverage monotone** — higher `coverage` → more/denser cloud (the total density sum is
non-decreasing in coverage). Print + `weather_test: ALL CHECKS PASSED`.

## Showcase — `--we1-clouddensity-shot` (Vulkan, main.cpp) + `--we1-clouddensity` (Metal, visual_test.mm)
A **2D grayscale cloud-density slab cut** (n=256): `GenCloudSlice(seed, 256, worldSize, frame, coverage, octaves)`,
per pixel map the Q16.16 density `[0,kOne]` to `[0,255]` by a PURE INTEGER scale (`g = clamp(density >> shift, 0,
255)`; NO float pixel math). Fixed seed/n/worldSize/frame/coverage/octaves. Reuse the FO1/PT1 pure-integer BMP-write
+ checksum code. SAME params IN BOTH renderers → byte-identical cross-backend by construction.

## Proof (STRICT integer — zero-diff cross-backend)
```
we1-clouddensity: drifting integer cloud-density (frame=<F>, coverage=<C>, octaves=<K>, n=256)
we1-clouddensity: two-run BYTE-IDENTICAL
we1-clouddensity: clouds drift -> frame F vs F+1 different {frameA:0x<Ha>, frameB:0x<Hb>} Ha != Hb
we1-clouddensity: zero-coverage -> clear sky (no-op) {maxDensity:0}
we1-clouddensity: provenance {frame:<F>, coverage:<C>, octaves:<K>}
```
Assertions: (1) two runs byte-identical; (2) frame F vs F+1 → a different image checksum (the clouds DRIFTED); (3)
zero-coverage → all-zero density (clear sky, the no-op); (4) provenance + coherent (non-flat cloud field under real
coverage). Register `we1_clouddensity` in verify.ps1 $Goldens (Flag `--we1-clouddensity`) + `--we1-clouddensity-shot`
in $vkShots, mirroring `fo1_wind` / `pt1_height`.

## Constraints (HARD)
- NEW engine/weather/weather.h + tests/weather_test.cpp + its CMake registration + the showcase blocks + verify.ps1
  registration. Reuse fpx.h/procterrain.h READ-ONLY. Do NOT modify clouds.h/procterrain.h/foliage.h/any existing
  shader/golden. NO new RHI, NO new shader. Pure-CPU integer (the cloud field + the 2D plot).
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (the controller bakes Metal and
  requires the Metal image == the Vulkan image byte-for-byte). The grayscale plot MUST be pure integer (no float
  pixel math). Do NOT route through a GPU float raster.
- Branch `fix-weather-we1`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows + the test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle weather_test'`
  Run weather_test.exe (ALL CHECKS PASSED) + hello_triangle.exe --we1-clouddensity-shot <out.bmp> (exit 0 + proof lines).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, weather_test passes, `--we1-clouddensity-shot`
  exits 0, the proof lines print, two-run byte-identical, the drift (frame-sensitivity) holds, the zero-coverage
  no-op holds, the plot is pure-integer. Commit message via a temp file + `git commit -F` (use the Bash tool
  heredoc). Commit to the branch and STOP. Report: commit hash, the proof output, image path, confirmation both
  renderers use identical params, and that the plot is pure-integer. (The CONTROLLER bakes the Metal golden,
  confirms zero-diff cross-backend, eyeballs the drifting cloud field.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
