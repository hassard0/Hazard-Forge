# Slice WE6 — The storm hero money-shot (Flagship #27 WEATHER, 6th/FINAL slice — capstone)

The capstone that COMPLETES flagship #27 and the entire deterministic living-world arc: a cinematic **eroded-valley
foliage meadow under a deterministic drifting overcast at a golden-hour sun with rain streaking through** — every
layer (terrain #26, foliage #25, clouds WE1/WE4, rain WE2, time-of-day WE3) a pure function of (seed, frame), so the
whole storm is byte-identical on every machine and netcode-replayable. FLOAT visresolve-bar (Metal two-run DIFF
0.0000 + cross-vendor mean + EYEBALL + provenance). NO new shader, NO new RHI (driver tuning over WE5).

## What to do (driver tuning — composes WE5)
Clone the `--we5-storm-shot` block → `--we6-hero-shot`, and TUNE it into a dramatic storm hero (WE5 is a distant
view where the rain/clouds are subtle; WE6 must read as a storm you're standing in). Apply to BOTH renderers
identically:
1. **Cinematic low/near camera** — bring the camera DOWN and IN to a low 3/4 / near-ground angle across the eroded
   valley meadow, so the relief + the leaning wind-swept foliage + the rain streaking past are the visible subject,
   receding to a stormy horizon (the FO6/PT6 hero framing).
2. **Dramatic overcast** — raise the cloud `coverage`/`densityMul`/composite intensity so the sky reads as a
   defined drifting overcast (heavier than WE5's mid-coverage), framed so the cloud band fills the upper scene.
3. **Visible rain** — raise the precip `count` and/or `dropScale`/`streakLen` and bring the rain column so the
   streaks clearly fall through the foreground (a readable downpour, not distant specks). Keep it deterministic
   (the WE2 `GenPrecip` at `kFrame`).
4. **Golden-hour sun** — pick a `kFrame` whose `SunSky` puts the sun low/warm (dawn or dusk) for a dramatic warm
   key light + a moody sky tint (the WE3 ramp). Tune the FO wind amplitude so the meadow visibly ripples in the
   storm.
5. **Honest caveats inherited** — the sky-IBL cool tint (FO6/PT6) + the cloud lighting is the existing float
   `clouds.frag` — framing/drift/rain/sun carry the hero; document it.
SAME seed/erosion/field/wind/frame/camera/weather params IN BOTH renderers so every layer is byte-identical (only
the GPU float raster diverges = the visresolve bar). Match the showcase's own scene.

## Proof (FLOAT visresolve-bar)
```
we6-hero: storm-valley meadow hero (plants:<K>, verts:<V>, drops:<D>, frame:<F>)
we6-hero: two renders BYTE-IDENTICAL
we6-hero: provenance plants seated + drift==recomputed + precip==GenPrecip + sun==SunSky {plants:<K>, seated:<K>, drops:<D>, ok:true}
we6-hero: empty -> bare valley (no-op) {emptyDrops:0}
```
Assertions: (1) two renders byte-identical; (2) provenance — every plant seated, cloud drift == `CloudDriftTime`,
rain count == `GenPrecip` size, sun == `SunSky`; `shaded > 0`; (3) the empty no-op (no rain + zero cloud coverage)
→ the bare valley. Register `we6_hero` in verify.ps1 $Goldens (Flag `--we6-hero`) + `--we6-hero-shot` in $vkShots,
mirroring `we5_storm` / `pt6_hero` (a FLOAT golden).

## YOU MUST EYEBALL YOUR OWN RESULT before committing
This is the capstone money-shot — convert the BMP to PNG (Add-Type System.Drawing) and READ the png. Confirm it
reads as a **stormy eroded-valley meadow**: a drifting overcast sky, rain visibly streaking through the foreground,
the wind-swept foliage leaning over the carved relief, a golden/moody time-of-day light, a believable storm hero —
NOT a clear-sky meadow with faint specks, NOT a broken/black render. If the rain/clouds are too subtle / the
framing is distant / a layer is missing, ITERATE (closer camera, heavier coverage, denser/bigger rain, a more
dramatic golden-hour frame) until it's a genuine storm hero. Only commit once it genuinely reads as a storm.

## Constraints (HARD)
- Driver tuning only (the FO6/PT6 precedent) — you likely do NOT need to touch weather.h/foliage.h/procterrain.h
  (if you add a tuned variant, append-only a NEW function; do NOT modify WE1–WE5 logic) + the showcase blocks +
  verify.ps1 registration. Do NOT modify any shader/existing header/existing golden. NO new shader, NO new RHI.
- FLOAT slice: do NOT assert strict zero-diff. Proof = two-run byte-identical + provenance + coherent.
- Branch `fix-weather-we6`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  Run `hello_triangle.exe --we6-hero-shot <out.bmp>`, confirm exit 0 + proof lines + provenance + shaded>0, THEN eyeball.
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--we6-hero-shot` exits 0, the proof lines print,
  two renders byte-identical, the provenance holds, the no-op holds, shaded>0, AND you eyeballed the PNG and it
  genuinely reads as a stormy eroded-valley meadow. Commit message via a temp file + `git commit -F` (use the Bash
  tool heredoc). Commit to the branch and STOP. Report: commit hash, the proof output, the PNG path + eyeball
  verdict (describe what it shows), confirmation both renderers use identical params, and the plant/vert/drop
  counts. (The CONTROLLER bakes the Metal golden, confirms two-run 0.0000 + cross-vendor mean, eyeballs the storm
  hero, and COMPLETES flagship #27. If after iterating you cannot make it a compelling storm, STOP and report.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
