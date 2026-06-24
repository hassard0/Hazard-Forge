# Slice FO4 — Integer distance-LOD pick (Issue #21, flagship #25 FOLIAGE, 4th slice)

A deterministic **integer distance-LOD** pick: each plant's LOD bucket is chosen by its integer XZ distance to the
camera against integer thresholds — so the LOD assignment is bit-identical cross-backend (NO float thresholds).
This is the scale primitive: near plants render full, far plants degrade to billboards, beyond-far are culled.
Builds on FO2/FO3. **STRICT-integer zero-diff cross-backend golden** (a pure-integer LOD-tinted plot). v1 is a
simple deterministic distance bucket (the existing `cluster-lod`/`gpu-cull` machinery is GPU HiZ/meshlet float
raster, heavier than v1 needs — honest scope).

## The addition — `engine/foliage/foliage.h` (APPEND-ONLY after the FO3 block)
Add to `hf::foliage` (do NOT modify FO1/FO2/FO3; you MAY append a `lod` field to `FoliageInstance` after `bend`):
- `inline uint32_t FoliageLod(const FxVec3& instPos, const FxVec3& camPos, fx nearR, fx farR)`: compute the XZ
  distance `fx d = FxLength({instPos.x - camPos.x, 0, instPos.z - camPos.z})` (Y zeroed; `FxLength` is the int64
  path, CPU-side → bit-identical). Bucket against integer thresholds (`midR = nearR + (farR - nearR) / 2`):
  `d < nearR -> 0` (near, full plant); `d < midR -> 1` (mid); `d < farR -> 2` (far billboard); else `-> 3`
  (culled). Pure integer compares. Return the bucket `[0,3]`.
- (Optional) extend `struct FoliageInstance { ... fx bend = 0; uint32_t lod = 0; };` (append `lod` after `bend`)
  and a `inline void AssignLods(std::vector<FoliageInstance>& v, const FxVec3& camPos, fx nearR, fx farR)` that
  sets `inst.lod = FoliageLod(inst.base.pos, camPos, nearR, farR)` per plant — convenient for FO5.
  Pure integer, NO `<cmath>`, NO float.

## Showcase — `--fo4-lod-shot` (Vulkan, main.cpp) + `--fo4-lod` (Metal, visual_test.mm)
A **2D top-down strict-integer plot tinting each plant by its LOD bucket**: take the FO2 meadow (`PlaceFoliage`
same fixed field/seed), pick a fixed `camPos` (e.g. a corner or the meadow centre) + fixed `nearR`/`farR`, compute
each plant's `FoliageLod`, and draw each plant's marker with a per-LOD integer color (e.g. LOD0 bright green, LOD1
yellow-green, LOD2 dim teal, LOD3 culled → dark/omitted). The result shows concentric LOD rings around the camera.
Reuse the FO2/FO3 pure-integer marker/pixel-map/checksum code; NO float pixel math. SAME field/seed/camPos/nearR/
farR/image-size IN BOTH renderers → byte-identical cross-backend by construction.

## Proof (STRICT integer — zero-diff cross-backend)
```
fo4-lod: distance LOD pick (near=<R0>, far=<R1>, plants=<K>)
fo4-lod: two-run BYTE-IDENTICAL
fo4-lod: monotone — farther plant never picks a nearer LOD {ok:true}
fo4-lod: per-bucket counts {lod0:<a>, lod1:<b>, lod2:<c>, culled:<d>}
fo4-lod: all-near (huge farR) -> every plant LOD 0 (no-op) {allLod0:true}
```
Assertions: (1) two runs byte-identical; (2) **monotone** — sort the plants by XZ distance to the camera and
assert the LOD bucket is non-decreasing (a farther plant NEVER gets a lower LOD — the pick is a correct distance
bucketing); (3) per-bucket counts sum to the plant count and are a non-trivial spread (the LOD does real work —
not all in one bucket for the chosen radii); (4) **no-op control** — a huge `farR` (and `nearR` ≥ the field
extent) → every plant LOD 0. Register `fo4_lod` in verify.ps1 $Goldens (Flag `--fo4-lod`) + `--fo4-lod-shot` in
$vkShots, mirroring `fo3_sway`.

## Constraints (HARD)
- APPEND-ONLY to engine/foliage/foliage.h (do NOT modify FO1/FO2/FO3 logic; append `lod` to FoliageInstance after
  `bend` + add `FoliageLod`/`AssignLods`) + the showcase blocks + verify.ps1 registration. Reuse FxLength + the
  FO2/FO3 field READ-ONLY. Do NOT modify pcg.h/mixer/any existing shader/golden. NO new RHI, NO new shader.
  Pure-CPU integer.
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (the controller bakes Metal and
  requires the Metal image == the Vulkan image byte-for-byte). The 2D plot MUST be pure integer (FoliageLod is
  integer; per-LOD colors are fixed integer literals). Do NOT route through a GPU float raster.
- Branch `fix-issue-21-fo4`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  (+ foliage_test if you extend it with a FoliageLod monotone/no-op check). Run `hello_triangle.exe --fo4-lod-shot
  <out.bmp>`, confirm exit 0 + proof lines.
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--fo4-lod-shot` exits 0, the proof lines print,
  two-run byte-identical, the monotone property holds, the per-bucket counts are a non-trivial spread, the
  all-near no-op holds, the plot is pure-integer. Commit message via a temp file + `git commit -F` (use the Bash
  tool heredoc). Commit to the branch and STOP. Report: commit hash, the proof output, image path, confirmation
  both renderers use identical field/seed/cam/radii/size, and that the plot is pure-integer. (The CONTROLLER bakes
  the Metal golden, confirms zero-diff cross-backend, eyeballs the LOD rings.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
