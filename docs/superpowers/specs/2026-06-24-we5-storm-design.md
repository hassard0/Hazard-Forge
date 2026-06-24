# Slice WE5 — Weather composited over the terrain+foliage hero (Flagship #27 WEATHER, 5th slice — composes #25+#26+WE1-4)

The full living-world scene: the eroded-valley meadow (PT5: terrain #26 + foliage #25) UNDER drifting clouds (WE4)
with rain (WE2) at a time-of-day (WE3) — every layer a deterministic function of (seed, frame). FLOAT
visresolve-bar (all the DATA is bit-exact integer; the composite render is float). NO new shader, NO new RHI
(reuse the PT5 scene + the WE4 cloud pass + the instanced-lit pipeline for rain + the WE3 sun).

## The addition — `engine/weather/weather.h` (APPEND-ONLY; a rain render bridge, the float crossing)
Add to `hf::weather` (do NOT modify WE1–WE4 logic; `#include "math/math.h"` like the foliage/pcg render bridges):
- `inline std::vector<math::Mat4> RainToRenderInstances(const std::vector<FxVec3>& drops, float dropScale, float worldMinX, float worldMinZ, float worldSpan, float baseY, float columnWorldH)`:
  the foliage `FoliageToRenderInstances` twin for rain — each drop (from `GenPrecip`, a Q16.16 position in the precip
  field's local `[0,areaW)×[0,columnH)×[0,areaD)` space) → a `math::Mat4` instance positioned in WORLD space over the
  terrain: map the drop's local XZ to `worldMinX + (x/areaW)*worldSpan` etc., and the drop's local Y to a WORLD Y
  `baseY + (y/columnH)*columnWorldH` (the rain column sits ABOVE the terrain). `FromTRS(worldPos, identity, Vec3{dropScale,
  dropScale*streakLen, dropScale})` — a thin tall streak (taller in Y for the rain look). Return the Mat4 array (the
  `scene::InstanceData` packing the instanced-lit pipeline consumes, matching `FoliageToRenderInstances`). The ONLY
  float code added; comment as the render bridge. (Pass the precip field's `areaW/columnH/areaD` so the mapping is a
  pure function — the rain positions derive from the bit-exact `GenPrecip`.)

## Driver — `--we5-storm-shot` (Vulkan, main.cpp) + `--we5-storm` (Metal, visual_test.mm)
Clone the `--pt5-meadow-shot` block (the terrain mesh + seated foliage scene) and ADD the three weather layers:
1. **Drifting clouds (WE4):** insert the clouds pass (the `--clouds-shot`/`--we4-drift` clouds.frag + composite
   wiring) into the render graph over the meadow scene, with `cprm.time = weather::CloudDriftTime(kFrame)` (the
   deterministic drift) — so clouds drift overhead between the scene and post passes.
2. **Rain (WE2):** `auto drops = weather::GenPrecip(precipField, kFrame);` →
   `weather::RainToRenderInstances(drops, dropScale, terrainMinX, terrainMinZ, terrainSpan, terrainTopY, columnWorldH)`
   → an instanced-lit draw (reuse the meadow's `lit_instanced` instanced-draw wiring) — falling streaks over the
   terrain.
3. **Time-of-day sun (WE3):** `auto ss = weather::SunSky(kFrame);` set `fd.lightDir = normalize(FxToFloat(ss.sunDir))`,
   `fd.lightColor`/sky from `ss.skyColor` — the whole scene lit + tinted by the time-of-day.
SAME seed/erosion/field/wind/frame/camera + the WE precip/cloud/sun params IN BOTH renderers (the terrain mesh +
seated foliage + rain instances + cloud drift + sun are byte-identical by construction; only the GPU float raster
diverges). Match the showcase's own scene.

## Proof (FLOAT visresolve-bar)
```
we5-storm: weather over eroded-valley meadow (plants:<K>, verts:<V>, drops:<D>, frame:<F>)
we5-storm: two renders BYTE-IDENTICAL
we5-storm: provenance plants seated + drift==recomputed + precip==GenPrecip + sun==SunSky {plants:<K>, seated:<K>, drops:<D>, ok:true}
we5-storm: empty -> bare meadow no weather (no-op) {emptyDrops:0}
```
Assertions: (1) two renders byte-identical; (2) provenance — every plant still seated (`plant.y == SampleHeight`),
the cloud `time == CloudDriftTime(kFrame)`, the rain instance count `== GenPrecip(...).size()`, the sun `== SunSky(kFrame)`;
`shaded > 0`; (3) a no-op control (count=0 rain + zero cloud coverage) → the bare meadow renders (the weather is
additive). Register `we5_storm` in verify.ps1 $Goldens (Flag `--we5-storm`) + `--we5-storm-shot` in $vkShots,
mirroring `pt5_meadow` (a FLOAT golden).

## YOU MUST EYEBALL YOUR OWN RESULT before committing
Convert the BMP to PNG (Add-Type System.Drawing) and READ the png. Confirm it shows the **eroded-valley meadow under
drifting clouds with rain at a time-of-day** — terrain + foliage + clouds overhead + rain streaks + the sun tint, a
coherent composited weather scene (NOT a black/broken render, NOT clouds-only or meadow-only). If a layer is
missing/broken, fix it and re-eyeball.

## Constraints (HARD)
- APPEND-ONLY to engine/weather/weather.h (do NOT modify WE1–WE4 logic; add `RainToRenderInstances`) + the showcase
  blocks + verify.ps1 registration. Reuse the PT5 scene + WE4 clouds pass + the instanced-lit pipeline + WE2/WE3 +
  math.h READ-ONLY. Do NOT modify any shader/existing header/existing golden. NO new shader, NO new RHI. Float ONLY
  in the render bridges + render path.
- FLOAT slice: do NOT assert strict zero-diff. Proof = two-run byte-identical + provenance + coherent.
- Branch `fix-weather-we5`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  Run `hello_triangle.exe --we5-storm-shot <out.bmp>`, confirm exit 0 + proof lines + provenance + shaded>0, THEN eyeball.
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--we5-storm-shot` exits 0, the proof lines print,
  two renders byte-identical, the provenance holds (plants seated + drift/precip/sun recompute), the no-op holds,
  shaded>0, AND you eyeballed the PNG and it shows the composited weather scene. Commit message via a temp file +
  `git commit -F` (use the Bash tool heredoc). Commit to the branch and STOP. Report: commit hash, the proof output,
  the PNG path + eyeball verdict, confirmation both renderers use identical params, and the plant/vert/drop counts.
  (The CONTROLLER bakes the Metal golden, confirms two-run 0.0000 + cross-vendor mean, eyeballs the storm scene. If
  a layer is missing/broken, that's a real failure — report it.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
