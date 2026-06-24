# Slice PT1 — Integer fBm heightfield generation (Flagship #26 PROCEDURAL TERRAIN, beachhead)

The irreducible primitive: a **deterministic integer heightfield** — `IntHeight(x,z)` returns a Q16.16 height from
a fractal sum (fBm) of integer value-noise octaves, bit-identical CPU↔Vulkan↔Metal BY CONSTRUCTION (NO runtime
`sin`/`sqrt`/`floor` — the strict-integer twin of the existing float `terrain::Height`). Establishes the NEW
`engine/terrain/procterrain.h` + the strict-integer heightfield the erosion (PT2/PT3) and render (PT4–PT6) build
on. **STRICT-integer zero-diff cross-backend golden** (a CPU-rendered integer grayscale heightmap, the FO1/PCG-plot
precedent). NO GPU/shader (host-evaluated).

## Do NOT touch the existing heightmap.h
`engine/terrain/heightmap.{h,cpp}` (`terrain::Height`/`BuildTerrain`) is FLOAT and FROZEN by goldens (`terrain`,
`terrain_stream`) + `terrain_test.cpp`. PT1 adds a NEW additive sibling `procterrain.h` in the same `hf::terrain`
namespace (mirroring how `foliage.h` is a new header beside `pcg.h`). Do NOT modify heightmap.{h,cpp}.

## The header — `engine/terrain/procterrain.h` (NEW, header-only, namespace `hf::terrain`)
This is a render-flagship data slice (the golden is a Mac-BAKED image, not a clang-standalone hash), so procterrain.h
MAY freely `#include "sim/fpx.h"` (+ `"pcg/pcg.h"` if you use PcgHash). Reuse `fpx.h` read-only for `fx`/`kOne`/
`kFrac`/`fxmul`. Provide (mirror the float `HashLattice`/`ValueNoise`/`Height` at heightmap.cpp:26-61 in INTEGER):
- `inline fx IntHashLattice(int32_t ix, int32_t iz, uint32_t seed)`: a Q16.16 corner value in `[0, kOne)` — the
  same integer hash shape as heightmap.cpp's `HashLattice` (`uint32 h = ix*374761393u + iz*668265263u; h = (h ^
  (h>>13)) * 1274126177u; h ^= (h>>16);`) folding `seed` in (e.g. `+ seed*<a prime>` before mixing), then return
  `(fx)(h >> 16)` (top 16 bits as the Q16.16 fraction → strictly `< kOne`). NO float.
- `inline fx IntValueNoise(fx x, fx z, uint32_t seed)`: smoothstep-faded bilinear value noise over the integer
  lattice, `x`/`z` in Q16.16. `ix = (int32_t)(x >> kFrac)` (arithmetic shift = floor, C++20), `tx = x - (ix <<
  kFrac)` (fractional in `[0,kOne)`); smoothstep `sx = fxmul(fxmul(tx,tx), 3*kOne - 2*tx)` (the `t²(3-2t)` fade in
  Q16.16); same for z; the 4 corners `IntHashLattice(ix{,+1}, iz{,+1}, seed)`; bilinear blend `a = c00 +
  fxmul(c10-c00, sx); b = c01 + fxmul(c11-c01, sx); return a + fxmul(b-a, sz);`. Pure integer.
- `inline fx IntHeight(fx x, fx z, int octaves, uint32_t seed)`: fBm — `fx h = 0, amp = <base, e.g. kOne>, fx fx_
  = <base freq>; for o in [0,octaves): h += fxmul(amp, IntValueNoise(fxmul(x, freq), fxmul(z, freq), seed + o));
  amp >>= 1; freq <<= 1 (double freq);` — sum octaves with halving amplitude / doubling frequency. `octaves <= 0`
  → 0 (flat, the no-op). Pure integer.
- `inline std::vector<fx> GenHeightField(uint32_t seed, int n, fx worldSize, int octaves)`: sample `IntHeight` at
  an `n×n` grid over `[0, worldSize)` (cell `(gx,gz)` → `x = FloorDiv(gx*worldSize, n)` or `(fx)((int64)gx*worldSize/n)`,
  similarly z; fold the seed via an offset added to x/z OR through `IntHashLattice`'s seed). Return the `n*n` grid
  (row-major). Pure integer, NO `<cmath>`, NO float.

## CPU test — `tests/procterrain_test.cpp` (NEW, register `hf_add_pure_test(procterrain_test)` in tests/CMakeLists.txt)
Assertions: (1) **replay-stable** — `IntHeight`/`GenHeightField` identical across calls for the same args; (2)
**seed-sensitive** — a different `seed` → a different field (the grids differ); (3) **bounds** — `IntValueNoise ∈
[0, kOne)`; the fBm height is bounded (well inside the ±32768 Q16.16 world bound for the chosen base amp/octaves —
assert a sane max); (4) **zero-octaves flat no-op** — `octaves <= 0` → every cell 0; (5) **smooth** — adjacent
cells differ by a bounded amount (no creased seams — a loose per-step bound, proving the smoothstep fade works).
Print + `procterrain_test: ALL CHECKS PASSED`.

## Showcase — `--pt1-height-shot` (Vulkan, main.cpp) + `--pt1-height` (Metal, visual_test.mm)
A **2D grayscale heightmap** of the n×n field (n=256), pure-integer: `GenHeightField(seed, 256, worldSize,
octaves)`, then per pixel map the Q16.16 height to `[0,255]` by a PURE INTEGER scale/shift (normalize against the
known height range — e.g. `g = clamp((h + bias) >> shift, 0, 255)`; NO float pixel math). Write the grayscale BMP.
Reuse the FO1/PCG pure-integer BMP-write + checksum code. Fixed seed/n/worldSize/octaves. SAME params IN BOTH
renderers → byte-identical cross-backend by construction.

## Proof (STRICT integer — zero-diff cross-backend)
```
pt1-height: integer fBm heightfield (octaves=<K>, n=256, seed=<S>)
pt1-height: two-run BYTE-IDENTICAL
pt1-height: different seed -> different field {seedA:0x<Ha>, seedB:0x<Hb>} Ha != Hb
pt1-height: zero-octaves -> flat (no-op) {maxH:0}
pt1-height: provenance {octaves:<K>, n:256}
```
Assertions: (1) two runs byte-identical; (2) a different seed → a different image checksum; (3) `octaves<=0` → a
flat field (maxH 0, the no-op); (4) provenance + coherent (non-flat under real octaves). Register `pt1_height` in
verify.ps1 $Goldens (Flag `--pt1-height`) + `--pt1-height-shot` in $vkShots, mirroring `fo1_wind`.

## Constraints (HARD)
- NEW engine/terrain/procterrain.h + tests/procterrain_test.cpp + its CMake registration + the showcase blocks +
  verify.ps1 registration. Reuse fpx.h/pcg.h READ-ONLY. Do NOT modify heightmap.{h,cpp}/terrain_stream/any
  existing shader/golden. NO new RHI, NO new shader. Pure-CPU integer (heightfield + the 2D plot).
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (the controller bakes Metal and
  requires the Metal image == the Vulkan image byte-for-byte). The grayscale plot MUST be pure integer (no float
  pixel math). Do NOT route through a GPU float raster.
- Branch `fix-terrain-pt1`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle procterrain_test'`
  Run `procterrain_test.exe` (ALL CHECKS PASSED) + `hello_triangle.exe --pt1-height-shot <out.bmp>` (exit 0 + proof lines).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, procterrain_test passes, `--pt1-height-shot` exits
  0, the proof lines print, two-run byte-identical, different-seed-different, the zero-octaves flat no-op holds, the
  plot is pure-integer. Commit message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the
  branch and STOP. Report: commit hash, the proof output, image path, confirmation both renderers use identical
  params, and that the plot is pure-integer. (The CONTROLLER bakes the Metal golden, confirms zero-diff
  cross-backend, eyeballs the heightmap.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
