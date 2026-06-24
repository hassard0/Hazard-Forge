# Slice FO6 — The hero money-shot (Issue #21, flagship #25 FOLIAGE, 6th/FINAL slice — capstone)

The capstone that COMPLETES flagship #25: a polished, cinematic **wind-swept foliage meadow** — the deterministic
FO1–FO5 stack (integer wind field → PCG placement → per-instance sway → LOD → instanced-lit render) tuned into a
believable hero shot where the plants VISIBLY LEAN in the wind and the field recedes with LOD to the horizon.
FLOAT visresolve-bar (Metal two-run DIFF 0.0000 + documented cross-vendor mean + EYEBALL + provenance back to the
bit-exact data). NO new shader, NO new RHI (reuse the FO5 render path; FO6 is scene tuning + the capstone proof).

## What to do (driver tuning — composes FO5)
Clone the `--fo5-scale-shot` block → `--fo6-hero-shot`, and TUNE it into a hero (the FO5 scale shot is a distant
tiny-clump field; FO6 must read as a meadow you're standing in). Apply these to BOTH renderers identically:
1. **Closer, lower, cinematic camera** — bring the camera DOWN and IN so individual plants and their wind lean are
   the visible subject (a low 3/4 / near-ground angle looking across the meadow toward the horizon), not a distant
   top-down clump. The field should fill the lower frame and recede to a horizon.
2. **Bigger, taller plants so the wind lean READS** — raise `baseScale` (and/or make the plants taller than wide if
   the mesh allows) so each plant is a visible blade/clump whose **bend** is clearly leaning. The FO3 wind bend is
   small in angle — at hero scale the lean must be perceptible (consider a larger bend amplitude in the FO6
   `WindField` and/or amplifying the lean in `FoliageToRenderInstances`'s lean-from-bend at the hero baseScale —
   tune so the meadow visibly ripples WITHOUT looking broken).
3. **A meadow that recedes with LOD** — a field large/deep enough that near plants are full (LOD0), far plants
   thin/shrink (LOD1/2) and the far edge culls (LOD3) — the LOD thinning visible toward the horizon. Pick a plant
   count that's dense-but-readable (hundreds-to-thousands).
4. **Honest color caveat** — the sky-IBL ambient tints matte surfaces cool (the PCG6/FO5 documented limit, can't be
   overridden without touching shaders). Size, framing, the wind lean, and the LOD recession carry the shot;
   document the tint in the showcase comment + the ARCHITECTURE entry.
SAME field/seed/wind/frame/camera/light IN BOTH renderers so the instance set is byte-identical (only the GPU float
raster diverges = the visresolve bar). Match the showcase's own scene (avoid a default-scene size mismatch).

## Proof (FLOAT visresolve-bar)
```
fo6-hero: wind-swept foliage meadow hero (instances:<K>, frame:<F>)
fo6-hero: two renders BYTE-IDENTICAL
fo6-hero: provenance instances == FoliageToRenderInstances(ApplyWind(PlaceFoliage(seed))) {instances:<K>, shaded:<P>}
fo6-hero: empty graph -> base scene (no-op) {emptyInstances:0}
```
Assertions: (1) two renders byte-identical; (2) provenance — the rendered instance count equals the recomputed
`FoliageToRenderInstances(ApplyWind(PlaceFoliage(seed)))` size (LOD-culled excluded), `shaded > 0` (coherent lit
scene); (3) an empty graph → 0 instances → base scene (no-op). Register `fo6_hero` in verify.ps1 $Goldens (Flag
`--fo6-hero`) + `--fo6-hero-shot` in $vkShots, mirroring `fo5_scale` / `pcg6_field` (a FLOAT golden).

## YOU MUST EYEBALL YOUR OWN RESULT before committing
This is the capstone money-shot — convert the BMP to PNG (Add-Type System.Drawing) and READ the png image to LOOK
at it. Confirm with your own eyes that it reads as a **wind-swept foliage meadow**: distinct plants VISIBLY LEANING
in a coherent wind direction, a field receding to the horizon with LOD thinning, a believable hero — NOT a distant
flat clump of dots. If the plants are too small / the lean isn't visible / it looks broken, ITERATE (closer/lower
camera, bigger baseScale, larger wind amplitude) until it's a genuine hero. Only commit once it genuinely looks
like a wind-swept meadow.

## Constraints (HARD)
- APPEND-ONLY to engine/foliage/foliage.h IF you add anything (likely just driver tuning — you may not need to
  touch foliage.h at all; if you amplify the lean, do it in the FO6 showcase params or a small new render-bridge
  variant, do NOT modify FO1–FO5 logic) + the showcase blocks + verify.ps1 registration. Do NOT modify any shader/
  existing header/existing golden. NO new shader, NO new RHI.
- FLOAT slice: do NOT assert strict zero-diff. Proof = two-run byte-identical + provenance + coherent; the
  controller handles Metal two-run 0.0000 + cross-vendor mean + eyeball.
- Branch `fix-issue-21-fo6`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  Run `hello_triangle.exe --fo6-hero-shot <out.bmp>`, confirm exit 0 + proof lines + a healthy instance count +
  shaded>0, THEN eyeball the PNG.
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--fo6-hero-shot` exits 0, the proof lines print,
  two renders byte-identical, the provenance count matches, the empty no-op holds, shaded>0, AND you have eyeballed
  the PNG and it genuinely reads as a wind-swept meadow hero. Commit message via a temp file + `git commit -F` (use
  the Bash tool heredoc). Commit to the branch and STOP. Report: commit hash, the proof output, the PNG path you
  inspected + your eyeball verdict (describe what it shows), confirmation both renderers use identical scene params,
  and the instance count. (The CONTROLLER bakes the Metal golden, confirms two-run 0.0000 + cross-vendor mean,
  eyeballs the hero, and COMPLETES flagship #25. If after iterating you cannot make it a compelling hero, STOP and
  report what you tried rather than committing a weak money-shot.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
