# Slice PCG2 — Jittered-grid point scatter (Issue #22, flagship #22 DETERMINISTIC PCG, 2nd slice)

The first real generation primitive: **deterministic jittered-grid scatter** — one point per grid cell, each
offset within its cell by a seeded PcgRand01 jitter, so the lattice is broken without any float/trig. This is the
declarative replacement for the engine's hand-coded `for(gx)for(gz)` instance grids. Pure int32 (the jitter is
one `fxmul`), provable with a **STRICT zero-differing-pixel cross-backend golden** (like PCG1, NOT a float bar).
Builds directly on PCG1's hash-PRNG. NO GPU shader (CPU-side host generator).

## The addition — `engine/pcg/pcg.h` (APPEND-ONLY to the existing header)
Add to `hf::pcg` (do NOT modify the PCG1 primitives — append after them):
- A small `struct PcgArea { FxVec3 min; FxVec3 max; }` (the XZ scatter region in Q16.16 world units; Y is the
  ground plane — use `min.y` for the scatter height, the area is a flat patch). Keep it a plain struct.
- `std::vector<FxVec3> ScatterGrid(const PcgStream& stream, const PcgArea& area, int cellsX, int cellsZ)`:
  partition the area's XZ extent into `cellsX × cellsZ` equal cells; for each cell `(cx, cz)` in **fixed
  ascending order** (`cz` outer, `cx` inner — pin the order so the output vector is deterministic), emit ONE
  point at the cell's min corner plus a per-axis in-cell jitter `fxmul(PcgRand01(stream, idx*2+0), cellW)` in X
  and `fxmul(PcgRand01(stream, idx*2+1), cellW)` in Z (where `idx` is the linear cell index `cz*cellsX+cx`, and
  `cellW`/`cellD` are the integer cell extents = `(max.x-min.x)/cellsX` etc. via integer divide). Y = `area.min.y`.
  Because `PcgRand01 ∈ [0,kOne)` and the jitter scales it by the cell extent, every point stays strictly inside
  its own cell (`cellMin ≤ p < cellMin + cellExtent`). Return empty for `cellsX<=0 || cellsZ<=0` or a degenerate
  area (the no-op control). Keep it int32 — the only multiply is `fxmul` (int64-intermediate, CPU-side, fine).
Use `idx*2+0` / `idx*2+1` as the two jitter indices so X and Z draw independent PcgRand01 samples (distinct
stream indices). Reuse `fpx.h` integer divide / `FloorDiv` as needed; do not introduce float.

## CPU test — extend `tests/pcg_test.cpp` (add a PCG2 section, keep the PCG1 checks)
Assertions: (1) **count** — `ScatterGrid` returns exactly `cellsX*cellsZ` points; (2) **in-cell containment** —
every emitted point lies within its own cell's integer AABB (recompute the cell bounds and assert
`cellMin.x ≤ p.x < cellMin.x+cellW` etc. — pure integer compares, the property that makes it a valid scatter);
(3) **replay-stable** — two calls with the same args are byte-equal (compare the vectors element-by-element);
(4) **seed-sensitive** — a different `stream.seed` gives a DIFFERENT layout but the SAME count (the
"different seed → different but valid" control); (5) **no-op** — `cellsX<=0` or a degenerate area → empty vector.
Follow the existing PCG1 test structure; print `pcg_test: ALL CHECKS PASSED` on success.

## Showcase — `--pcg2-scatter-shot` (Vulkan, main.cpp) + `--pcg2-scatter` (Metal, visual_test.mm)
A **2D top-down strict-integer plot of the scattered grid**. Pick a fixed area (e.g. a square XZ patch),
`cellsX=cellsZ` (e.g. 48×48 = 2304 points), a fixed seed/salt. Map each scattered point's (x,z) to an INTEGER
pixel coord in a fixed-size image (e.g. 256×256) by pure integer math (scale the Q16.16 world coord into pixel
range via integer ops — NO float pixel rounding) and write a marker pixel. The result is a jittered point field
(visibly grid-structured but with the lattice broken by the jitter — contrast the perfectly-regular grid). SAME
area/cells/seed/salt/image-size IN BOTH renderers (main.cpp and visual_test.mm) so the image is byte-identical
cross-backend by construction. Mirror PCG1's `--pcg1-hash-shot` plot/save/checksum code (reuse its pure-integer
pixel approach verbatim — DO NOT use any float pixel/color math, which would risk MSVC↔clang divergence).

## Proof (STRICT integer — zero-diff cross-backend)
```
pcg2-scatter: jittered-grid scatter (cellsX=48 cellsZ=48 -> 2304 points, seed=<S>)
pcg2-scatter: two-run BYTE-IDENTICAL
pcg2-scatter: count == cellsX*cellsZ AND all points in-cell {points:2304, inCell:2304}
pcg2-scatter: different seed -> different field {seedA_hash:<hA>, seedB_hash:<hB>} hA != hB (same 2304 count)
pcg2-scatter: provenance {cells:2304, jittered:true, empty-area:0}
```
Assertions: (1) two runs byte-identical; (2) point count == cellsX*cellsZ and ALL points pass the in-cell
containment check; (3) a different seed → a different image checksum but the same count; (4) the empty-area
control yields 0 points; provenance coherent. Register `pcg2_scatter` in verify.ps1 $Goldens (Flag
`--pcg2-scatter`) + `--pcg2-scatter-shot` in $vkShots, mirroring how `pcg1_hash` is registered.

## Constraints (HARD)
- APPEND-ONLY to engine/pcg/pcg.h (do NOT modify the PCG1 primitives) + extend tests/pcg_test.cpp + the showcase
  blocks in main.cpp/visual_test.mm + the verify.ps1 registration. Reuse fpx.h/particles.h/PCG1 READ-ONLY. Do NOT
  modify any other header/shader/golden. NO new RHI, NO new shader. Int32 (the only mul is `fxmul`).
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (the controller bakes Metal and
  requires the Metal image == the Vulkan image byte-for-byte). So the 2D plot MUST be pure integer (no float
  pixel math). Do NOT route it through a GPU float raster.
- Branch `fix-issue-22-pcg2`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows + the pcg_test target via the PowerShell tool (NOT bash — vcvars no-ops under Git Bash),
  single-quoting the cmd arg so PowerShell doesn't choke on the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle pcg_test'`
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--pcg2-scatter-shot` exits 0, the proof lines
  print, two-run byte-identical, count+in-cell verified, different-seed-different verified, AND pcg_test passes.
  Commit message via a temp file + `git commit -F` (PowerShell here-strings break on `--flag` dashes; use the
  Bash tool heredoc). Commit to the branch and STOP. Report: commit hash, proof output, image path, confirmation
  both renderers use identical area/cells/seed/size, that pcg_test passes, and that the plot is pure-integer.
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
