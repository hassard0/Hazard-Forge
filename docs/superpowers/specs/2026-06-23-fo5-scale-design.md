# Slice FO5 — The scale render (Issue #21, flagship #25 FOLIAGE, 5th slice — first FLOAT slice)

Render the deterministic foliage field as lit 3D instanced geometry at SCALE (thousands of instances) through the
EXISTING instanced-lit pipeline — the bit-exact integer foliage data (FO2 placement + FO3 wind sway + FO4 LOD)
crosses to float ONCE at the render bridge. This is the arc's first **float** slice (visresolve-bar): the gate is
Metal two-run DIFF 0.0000 + provenance (instances derive from the bit-exact data) + a documented cross-vendor
mean — NOT strict zero-diff. NO new shader, NO new RHI (reuse the instanced-lit pipeline verbatim, the PCG6/PT6
precedent).

## The render bridge — `engine/foliage/foliage.h` (APPEND-ONLY after the FO4 block; render-only, the float crossing)
Add to `hf::foliage` (do NOT modify FO1–FO4; `#include "math/math.h"` like pcg.h does):
- `inline std::vector<math::Mat4> FoliageToRenderInstances(const std::vector<FoliageInstance>& instances, float baseScale)`:
  the `pcg::PcgToRenderInstances` twin, but per instance it ALSO bakes the per-instance **bend** (a float lean) and
  HONORS the **LOD** (cull `lod == 3`; optionally shrink far LODs). For each instance with `lod != 3`:
  `t = {FxToFloat(base.pos.x), FxToFloat(base.pos.y), FxToFloat(base.pos.z)}`;
  `yaw = normalize(Quat{FxToFloat(base.orient.x..w)})` (the placement yaw);
  `lean = QuatFromAxisAngle(horizontalAxis, FxToFloat(bend))` — a float lean about a fixed horizontal axis (e.g.
  +X) by the bend angle (this is the FLOAT visresolve-bar code — NOT asserted bit-exact; use the engine's
  `math::` quat-from-axis-angle, or build the quat directly with `std::cos`/`std::sin` — float is allowed HERE in
  the render bridge only);
  `scale = FxToFloat(base.scale) * baseScale * lodScale(lod)` (a per-LOD scale, e.g. 1.0 / 1.0 / 0.7 for LOD0/1/2 —
  far plants a touch smaller as a billboard stand-in; honest v1);
  `out.push_back(FromTRS(t, lean * yaw, Vec3{scale,scale,scale}))`. Skip `lod == 3`. The `scene::InstanceData`
  packing (16 floats column-major) the instanced-lit pipeline consumes — match `PcgToRenderInstances` EXACTLY.
  Empty input → empty output (the no-op). The ONLY float code in foliage.h (clearly commented as the render bridge).

## Driver — `--fo5-scale-shot` (Vulkan, main.cpp) + `--fo5-scale` (Metal, visual_test.mm)
Clone the `--pcg6-field-shot` block (the instanced-lit-sphere render + sky + shadow + post + fixed 3/4 camera +
directional light) VERBATIM — swap the rock field for the foliage field:
- Build a DENSE `FoliageField` so `PlaceFoliage` yields THOUSANDS of plants (e.g. a larger area + more cells than
  FO2's 48×48 — scale the `PcgGraph` cells up to get a few thousand instances; the field on a ground patch centred
  at the origin so it sits under the existing camera).
- `auto plants = PlaceFoliage(field, stream); ApplyWind(plants, wind, frame); AssignLods(plants, camPos, nearR,
  farR);` then `FoliageToRenderInstances(plants, baseScale)` → the instanced-lit draw. Use a `baseScale` that reads
  as small plants. Draw the ground plane (reuse the pcg6/grain ground).
- SAME field/seed/wind/frame/camera/light/LOD IN BOTH renderers (the instance set byte-identical by construction;
  only the GPU float raster diverges = the visresolve bar). Match the showcase's own scene (avoid a default-scene
  size mismatch).

## Proof (FLOAT visresolve-bar — the PCG6 precedent)
```
fo5-scale: foliage scale render (instances:<K>, lit 3D instanced, frame:<F>)
fo5-scale: two renders BYTE-IDENTICAL
fo5-scale: provenance instances == FoliageToRenderInstances(ApplyWind(PlaceFoliage(seed))) {instances:<K>, shaded:<P>}
fo5-scale: empty graph -> base scene (no-op) {emptyInstances:0}
```
Assertions: (1) two renders byte-identical (deterministic data + render); (2) provenance — the rendered instance
count equals the recomputed `FoliageToRenderInstances(...)` size (the render derives from the bit-exact FO2/FO3/FO4
data; LOD-culled excluded), `shaded > 0` (coherent lit image); (3) an empty graph → 0 instances → the base scene
(no-op). The instance count should be a HEALTHY THOUSANDS (the scale story — print it). Register `fo5_scale` in
verify.ps1 $Goldens (Flag `--fo5-scale`) + `--fo5-scale-shot` in $vkShots, mirroring `pcg6_field` (a FLOAT golden).

## Constraints (HARD)
- APPEND-ONLY to engine/foliage/foliage.h (do NOT modify FO1–FO4) + the showcase blocks + verify.ps1 registration.
  Reuse the instanced-lit pipeline + pcg/grain render helpers + math.h READ-ONLY. Do NOT modify any shader/existing
  header/existing golden. NO new shader, NO new RHI. Float is allowed ONLY in the `FoliageToRenderInstances` render
  bridge + the render path (NOT in the FO1–FO4 data).
- FLOAT slice: do NOT assert strict zero-diff cross-backend. The proof is two-run byte-identical + provenance +
  coherent; the controller handles the Metal two-run 0.0000 + cross-vendor mean + eyeball.
- Branch `fix-issue-21-fo5`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  Run `hello_triangle.exe --fo5-scale-shot <out.bmp>`, confirm exit 0 + the proof lines + a thousands instance
  count + shaded>0 (a non-black, non-trivial lit field).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--fo5-scale-shot` exits 0, the proof lines print,
  two renders byte-identical, the provenance instance count matches (a healthy thousands), the empty-graph no-op
  holds, shaded>0. Commit message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the
  branch and STOP. Report: commit hash, the proof output (instance count), image path, confirmation both renderers
  use identical field/seed/wind/frame/camera, and the rendered instance count. (The CONTROLLER bakes the Metal
  golden, confirms Metal two-run 0.0000 + cross-vendor mean, eyeballs a coherent dense lit foliage field. If the
  render is black/empty or the count isn't thousands, that's a real failure — report it.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
