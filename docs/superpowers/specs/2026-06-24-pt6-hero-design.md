# Slice PT6 — The hero money-shot (Flagship #26 TERRAIN, 6th/FINAL slice — capstone)

The capstone that COMPLETES flagship #26 and the whole deterministic procedural-visual stack: a polished, cinematic
**eroded valley carpeted with the wind-swept foliage meadow** — the deterministic integer heightfield (PT1) +
hydraulic+thermal erosion (PT2/PT3) + the lit terrain mesh (PT4) + the seated, wind-swept foliage (PT5) tuned into a
believable hero. A seed → this exact byte-identical landscape on every machine. FLOAT visresolve-bar (Metal two-run
DIFF 0.0000 + cross-vendor mean + EYEBALL + provenance). NO new shader, NO new RHI (driver tuning over PT4+PT5).

## What to do (driver tuning — composes PT5)
Clone the `--pt5-meadow-shot` block → `--pt6-hero-shot`, and TUNE it into a hero (PT5 is a distant top-down-ish view;
PT6 must read as a landscape you're standing in). Apply to BOTH renderers identically:
1. **Cinematic low/near camera** — bring the camera DOWN and IN to a low 3/4 / near-ground angle looking ACROSS the
   eroded valley toward a horizon, so the carved relief + the meadow leaning in the wind are the visible subject, and
   the terrain recedes to a horizon (the FO6 hero framing applied to the terrain+meadow scene).
2. **Read the eroded relief** — tune the erosion iters / relief amplitude / `heightScale` so the valley has visible
   CARVED CHANNELS and ridgelines (not a flat plain), framed so the carving reads.
3. **The meadow ripples + recedes with LOD** — the wind-swept foliage visibly leaning (the FO3 sway), denser near,
   thinning to the horizon with LOD (FO4). Tune `baseScale`/`leanGain`/`heightMul`/the wind amplitude so the meadow
   carpets the valley and visibly ripples (the FO6 hero tuning, now over the terrain).
4. **Honest caveats inherited** — the sky-IBL ambient cool tint (FO6/PT4) — size/framing/relief/wind carry the hero,
   document it.
SAME seed/erosion/field/wind/frame/camera/light IN BOTH renderers so the mesh + seated instance set are byte-identical
(only the GPU float raster diverges = the visresolve bar). Match the showcase's own scene (avoid a default-scene
size mismatch).

## Proof (FLOAT visresolve-bar)
```
pt6-hero: eroded valley meadow hero (plants:<K>, verts:<V>, frame:<F>)
pt6-hero: two renders BYTE-IDENTICAL
pt6-hero: provenance plants seated on eroded terrain + instances == FoliageToRenderInstances {plants:<K>, seated:<K>, shaded:<P>}
pt6-hero: empty graph -> bare eroded valley (no-op) {emptyPlants:0}
```
Assertions: (1) two renders byte-identical; (2) provenance — every plant seated (`plant.y == SampleHeight`), the
rendered instance count matches the recomputed `FoliageToRenderInstances*` size, `shaded > 0` (coherent lit scene);
(3) an empty graph → 0 plants → the bare eroded valley (the no-op). Register `pt6_hero` in verify.ps1 $Goldens (Flag
`--pt6-hero`) + `--pt6-hero-shot` in $vkShots, mirroring `pt5_meadow` / `fo6_hero` (a FLOAT golden).

## YOU MUST EYEBALL YOUR OWN RESULT before committing
This is the capstone money-shot — convert the BMP to PNG (Add-Type System.Drawing) and READ the png image to LOOK
at it. Confirm it reads as a **wind-swept foliage meadow carpeting an eroded valley**: visible carved relief
(channels/ridges), distinct plants leaning in a coherent wind, the field receding to the horizon with LOD thinning,
a believable hero — NOT a distant flat green slab. If the relief isn't visible / the plants are too small / the lean
isn't visible / it looks broken, ITERATE (closer/lower camera, more erosion/relief, bigger baseScale, larger wind
amplitude) until it's a genuine hero. Only commit once it genuinely looks like a wind-swept meadow on an eroded
valley.

## Constraints (HARD)
- Driver tuning only (the FO6 precedent) — you likely do NOT need to touch procterrain.h/erosion.h/foliage.h (if you
  add a lean-amplified variant, add a NEW function append-only; do NOT modify PT1–PT5 logic) + the showcase blocks +
  verify.ps1 registration. Do NOT modify any shader/existing header/existing golden. NO new shader, NO new RHI.
- FLOAT slice: do NOT assert strict zero-diff. Proof = two-run byte-identical + provenance + coherent.
- Branch `fix-terrain-pt6`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  Run `hello_triangle.exe --pt6-hero-shot <out.bmp>`, confirm exit 0 + proof lines + healthy counts + shaded>0, THEN
  eyeball the PNG.
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--pt6-hero-shot` exits 0, the proof lines print,
  two renders byte-identical, the provenance holds (plants seated + instance count matches), the empty no-op holds,
  shaded>0, AND you have eyeballed the PNG and it genuinely reads as a wind-swept meadow on an eroded valley. Commit
  message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the branch and STOP. Report:
  commit hash, the proof output, the PNG path you inspected + your eyeball verdict (describe what it shows),
  confirmation both renderers use identical scene params, and the plant/vert counts. (The CONTROLLER bakes the Metal
  golden, confirms two-run 0.0000 + cross-vendor mean, eyeballs the hero, and COMPLETES flagship #26. If after
  iterating you cannot make it a compelling hero, STOP and report what you tried rather than committing a weak shot.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
