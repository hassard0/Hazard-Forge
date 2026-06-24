# Slice WE4 — Lit 3D drifting-cloud render (Flagship #27 WEATHER, 4th slice — first FLOAT slice)

Render the deterministic clouds as a lit 3D volumetric layer through the EXISTING `clouds.frag` raymarch — but
driven by the WE1 **integer drift** + the WE3 **integer sun**, so the cloudscape drifts and is lit bit-identically
by (seed, frame). The arc's first **float** slice: the cloud-density DATA + drift + sun are bit-exact integer; only
the raymarch raster is float (the visresolve-bar — Metal two-run DIFF 0.0000 + cross-vendor mean + provenance + eyeball).
NO new shader, NO new RHI (reuse the `--clouds-shot` clouds pass verbatim; only the advection + sun INPUTS change).

## The addition — `engine/weather/weather.h` (APPEND-ONLY; tiny render-bridge helpers, the float crossing)
Add to `hf::weather` (do NOT modify WE1–WE3):
- `inline float CloudDriftTime(uint32_t frame)`: the float advection time the cloud shader consumes, derived from
  the WE1 integer drift — `return hf::sim::fpx::FxToFloat((fx)((int64_t)kCloudDriftRate * frame));` (the bit-exact
  integer drift offset converted to float once — both backends compute the SAME integer then the SAME `FxToFloat`,
  so the cloud DRIFT is deterministic cross-platform; the FO5/PT4 single-float-crossing precedent).
- (Optional) small `FxToFloat3(FxVec3)` helpers to feed `SunSky(frame).sunDir`/`.skyColor` into the render as
  floats. These are the ONLY float code added; clearly comment them as the render bridge.

## Driver — `--we4-drift-shot` (Vulkan, main.cpp) + `--we4-drift` (Metal, visual_test.mm)
Clone the `--clouds-shot` block (main.cpp ~85201-85410) VERBATIM, with two INPUT changes (the data driving the
existing float clouds pass):
1. **Deterministic drift:** `cprm.time = weather::CloudDriftTime(kFrame);` (replace the frozen `clouds::kFixedTime`)
   — so the clouds advect by the bit-exact integer drift at `kFrame` (NOT a clock).
2. **Deterministic sun from WE3:** `auto ss = weather::SunSky(kFrame);` set the scene/cloud light direction
   `fd.lightDir = normalize(FxToFloat(ss.sunDir))` and the cloud `sunColor`/`skyTop`/`skyBottom` from `ss.skyColor`
   (so the clouds are lit + tinted by the time-of-day) — fold WE3's sun/sky into the existing CloudParams fields.
   (Keep the rest of the clouds scene/coverage/steps as-is.)
SAME `kFrame`/seed/coverage/camera IN BOTH renderers (the drift + sun INPUTS are bit-exact integer, so the cloud
data is byte-identical by construction; only the GPU float raymarch raster diverges = the visresolve bar). Match
the showcase's own scene (avoid a default-scene size mismatch).

## Proof (FLOAT visresolve-bar)
```
we4-drift: lit drifting clouds (frame=<F>, drift-time=<t>, sun-elev=<e>)
we4-drift: two renders BYTE-IDENTICAL
we4-drift: provenance drift == recomputed {cloudTime:<t>, expected:<t>} AND sun from SunSky {ok:true}
we4-drift: clouds shaded {shaded:<P>}
```
Assertions: (1) two renders byte-identical (deterministic drift + sun + render); (2) provenance — `cprm.time ==
weather::CloudDriftTime(kFrame)` (the float advection IS the bit-exact integer drift) AND the light dir/colors ==
`SunSky(kFrame)` (the render derives from the bit-exact WE1/WE3 data); (3) `shaded > 0` (a coherent lit cloud
layer — not empty). Register `we4_drift` in verify.ps1 $Goldens (Flag `--we4-drift`) + `--we4-drift-shot` in
$vkShots, mirroring `pt4_mesh` / the `clouds` golden (a FLOAT golden).

## Constraints (HARD)
- APPEND-ONLY tiny render-bridge helpers to engine/weather/weather.h (do NOT modify WE1–WE3 logic) + the showcase
  blocks + verify.ps1 registration. Reuse the EXISTING clouds pass (`clouds.frag`/`clouds_composite.frag`) + WE1/WE3
  READ-ONLY. Do NOT modify clouds.h/clouds.frag/any shader/existing header/existing golden. NO new shader, NO new
  RHI. Float is allowed ONLY in the render-bridge helpers + the render path (NOT in WE1–WE3 data).
- FLOAT slice: do NOT assert strict zero-diff. Proof = two-run byte-identical + provenance + coherent; the
  controller handles Metal two-run 0.0000 + cross-vendor mean + eyeball.
- Branch `fix-weather-we4`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  Run `hello_triangle.exe --we4-drift-shot <out.bmp>`, confirm exit 0 + proof lines + the provenance + shaded>0 (a
  non-black lit cloud render).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--we4-drift-shot` exits 0, the proof lines print,
  two renders byte-identical, the provenance holds (cprm.time == CloudDriftTime, sun from SunSky), shaded>0. Commit
  message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the branch and STOP. Report:
  commit hash, the proof output, image path, confirmation both renderers use identical frame/seed/camera, and the
  shaded count. (The CONTROLLER bakes the Metal golden, confirms two-run 0.0000 + cross-vendor mean, eyeballs a
  coherent lit drifting-cloud sky. If the render is black/empty, that's a real failure — report it.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
