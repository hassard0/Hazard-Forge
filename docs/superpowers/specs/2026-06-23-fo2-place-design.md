# Slice FO2 — PCG-driven foliage placement (Issue #21, flagship #25 FOLIAGE, 2nd slice — pure reuse)

Place the foliage by REUSING the just-completed deterministic PCG: a `FoliageField` holds a `PcgGraph` and
`PlaceFoliage` calls `pcg::Generate` verbatim, wrapping each scattered `PcgInstance` as a plant. So the whole
meadow is a pure function of the seed (scatter → mask → transform → prune, all already proven bit-exact). Strict
**integer zero-diff cross-backend golden** (a pure-integer 2D plant plot, the `pcg2_scatter` precedent). Builds on
FO1; the foliage layer adds NO new placement math — it composes PCG.

## The addition — `engine/foliage/foliage.h` (APPEND-ONLY after the FO1 block)
Add to `hf::foliage` (do NOT modify the FO1 wind code; foliage.h already `#include`s `sim/fpx.h` — also
`#include "pcg/pcg.h"`):
- `struct FoliageInstance { hf::pcg::PcgInstance base; };` (a plant = a PCG instance; later slices append a `bend`
  param (FO3) and a `lod` field (FO4) — keep the struct extensible, base first).
- `struct FoliageField { hf::pcg::PcgGraph graph; };` (the declarative meadow recipe — the PcgGraph that scatters
  the plants; later slices may add a plant-type id, fine).
- `inline std::vector<FoliageInstance> PlaceFoliage(const FoliageField& field, const hf::pcg::PcgStream& stream)`:
  `auto pcgInstances = hf::pcg::Generate(field.graph, stream);` then wrap each into a `FoliageInstance{inst}` in the
  SAME order; return the vector. (Pure reuse — `Generate` already does scatter/mask/transform/prune deterministically.)
  No new math, no float.

## Showcase — `--fo2-place-shot` (Vulkan, main.cpp) + `--fo2-place` (Metal, visual_test.mm)
A **2D top-down strict-integer plot of the placed plants**: build a fixed `FoliageField` (a `PcgGraph` over a
fixed area / cells / radial mask / transform — pick a meadow recipe like the PCG5/PCG6 graph), a fixed seed,
`PlaceFoliage` it, and plot each plant's `base.pos` (x,z) as a marker at its INTEGER pixel coord (reuse the
`--pcg2-scatter-shot` / `--fo1-wind-shot` pure-integer pixel-map + marker + checksum code VERBATIM — NO float pixel
math). 256×256 image. SAME field/seed/image-size IN BOTH renderers so the image is byte-identical cross-backend by
construction. (Optionally tint each marker by its `base.scale` bucket for a richer plot — pure-integer.)

## Proof (STRICT integer — zero-diff cross-backend)
```
fo2-place: PCG-driven foliage placement (plants=<K>, seed=<S>)
fo2-place: two-run BYTE-IDENTICAL
fo2-place: provenance plants == Generate(graph,stream) count {plants:<K>, generated:<K>}
fo2-place: different seed -> different meadow {seedA:0x<Ha>, seedB:0x<Hb>} Ha != Hb (both valid)
fo2-place: empty graph -> 0 plants (no-op) {emptyPlants:0}
```
Assertions: (1) two runs byte-identical; (2) provenance — `PlaceFoliage(field,stream).size() == Generate(field.graph,
stream).size()` (the placement IS the PCG output, count matches); (3) a different seed → a different image checksum
(a different meadow) with a healthy plant count; (4) an empty graph (`cellsX<=0`) → 0 plants (no-op). Register
`fo2_place` in verify.ps1 $Goldens (Flag `--fo2-place`) + `--fo2-place-shot` in $vkShots, mirroring `fo1_wind`/
`pcg2_scatter`.

## Constraints (HARD)
- APPEND-ONLY to engine/foliage/foliage.h (do NOT modify the FO1 code) + the showcase blocks + verify.ps1
  registration. Reuse pcg.h/fpx.h READ-ONLY (call `pcg::Generate`). Do NOT modify pcg.h/mixer/any existing shader/
  golden. NO new RHI, NO new shader. Pure-CPU integer.
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (the controller bakes Metal and
  requires the Metal image == the Vulkan image byte-for-byte). The 2D plot MUST be pure integer (no float pixel
  math). Do NOT route through a GPU float raster.
- Branch `fix-issue-21-fo2`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  (+ foliage_test if you extend it). Run `hello_triangle.exe --fo2-place-shot <out.bmp>`, confirm exit 0 + proof lines.
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--fo2-place-shot` exits 0, the proof lines print,
  two-run byte-identical, the provenance count matches, different-seed-different verified, the empty-graph no-op
  holds, the plot is pure-integer. Commit message via a temp file + `git commit -F` (use the Bash tool heredoc).
  Commit to the branch and STOP. Report: commit hash, the proof output, image path, confirmation both renderers use
  identical field/seed/size, and that the plot is pure-integer. (The CONTROLLER bakes the Metal golden, confirms
  zero-diff cross-backend, eyeballs the meadow plot.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
