# Slice PCG6 — Lit 3D render capstone (Issue #22, flagship #22, 6th/FINAL slice — the money-shot)

The capstone that COMPLETES flagship #22: render a **procedurally scattered rock/debris field** — a full
`PcgGraph` (scatter → radial mask → random yaw/scale → overlap-prune) — as lit 3D instanced spheres through the
EXISTING instanced-lit pipeline. The bit-exact integer PCG generation (PCG1–PCG5) crosses to float ONCE, at the
render bridge. This is the arc's ONLY **float** slice (visresolve-bar): the gate is Metal two-run DIFF 0.0000 +
provenance (the instances derive from the bit-exact `Generate`) + a documented cross-vendor mean — NOT strict
zero-diff (float raster diverges cross-vendor). NO new shader, NO new RHI (reuse the instanced-lit-sphere pipeline
verbatim, the PT6/GR6 precedent).

## The render bridge — `engine/pcg/pcg.h` (APPEND-ONLY after the PCG5 block; render-only, the ONE float crossing)
Add to `hf::pcg` (do NOT modify PCG1–PCG5):
- `std::vector<math::Mat4> PcgToRenderInstances(const std::vector<PcgInstance>& instances, float baseRadius)`:
  the `grain::GrainToRenderInstances` / `particles::ParticleToRenderInstances` / `fpx::FxBodyTransform` twin, but
  per-instance it uses BOTH the orient AND the scale (grain/particle were position+uniform-radius only). For each
  instance build `math::FromTRS(t, q, s)` where `t = {FxToFloat(pos.x), FxToFloat(pos.y), FxToFloat(pos.z)}`,
  `q = normalize(math::Quat{FxToFloat(orient.x), FxToFloat(orient.y), FxToFloat(orient.z), FxToFloat(orient.w)})`,
  and `s = math::Vec3{r, r, r}` with `r = FxToFloat(inst.scale) * baseRadius`. Return the column-major `Mat4`
  array (the `scene::InstanceData` / `InstanceTransformLayout` packing — read how `GrainToRenderInstances` packs
  it and match EXACTLY). This is render-only (NO sim mutation, NOT used by PCG1–PCG5). Include `math/math.h` +
  `<vector>` (already present). It is the ONLY float code in pcg.h — keep it clearly separated/commented as the
  render bridge (the integer generator stays the cross-platform-exact core).

## Driver — `--pcg6-field-shot` (Vulkan, main.cpp) + `--pcg6-field` (Metal, visual_test.mm)
Clone `RunPt6RenderShowcase` (metal_headless/visual_test.mm ~line 56217) and the Vulkan `--pt6-render-shot` block
(the instanced-lit-sphere pipeline + sky + shadow + post + the fixed 3/4 camera + directional light) VERBATIM —
just swap the particle pool for the PCG instance set:
- Build a `PcgGraph` for a **rock field on a ground patch centred at the origin** so it sits in front of the
  existing camera: `area` = an XZ patch centred on origin (e.g. min `{-6, 0, -6}`, max `{6, 0, 6}` in Q16.16 world
  units — pick what frames well under the pt6/grain camera), `cellsX=cellsZ` (e.g. 24×24), `useMask=true` (a
  radial mask centred in the patch so the field is denser in the middle), `transform` = random yaw + scale
  `[0.6, 1.4]`, `prune=true` with a `pruneRadius` so the rocks don't interpenetrate. `Generate(graph, stream)` →
  instances → `PcgToRenderInstances(instances, baseRadius)` (a `baseRadius` so the spheres read as rocks, e.g.
  `0.35f`) → the instanced-lit draw. Optionally also draw a ground plane (reuse the pt6/grain ground if present).
- SAME graph/seed/baseRadius/camera/light IN BOTH renderers (main.cpp and visual_test.mm) so the instance set is
  byte-identical by construction (the float divergence is only the GPU raster, the documented visresolve bar).
- A reflective/varied look isn't required — matte lit spheres of varied size + orientation reading as scattered
  rocks/debris is the money-shot. Navigate any default-scene size-mismatch the way pt6/grain do (match the
  showcase's own scene, not the default).

## Proof (FLOAT visresolve-bar — the PT6/GR6 precedent)
```
pcg6-field: procedural rock field (instances:<K>, lit 3D instanced spheres, seed=<S>)
pcg6-field: two renders BYTE-IDENTICAL
pcg6-field: provenance instances == PcgToRenderInstances(Generate(seed,graph)) {instances:<K>, shaded:<P>}
pcg6-field: empty graph -> base scene (no-op) {emptyInstances:0}
```
Assertions: (1) two renders byte-identical (deterministic generation + render); (2) provenance — the rendered
instance count equals `PcgToRenderInstances(Generate(seed, graph)).size()` (the render derives from the bit-exact
PCG pipeline), `shaded > 0` (coherent lit image); (3) an empty graph (`cellsX<=0`) → 0 instances → the base
scene (no-op control). Register `pcg6_field` in verify.ps1 $Goldens (Flag `--pcg6-field`) + `--pcg6-field-shot`
in $vkShots, mirroring `pt6_render` / `grain_render` (a FLOAT golden — the controller bakes Metal, the bar is
Metal two-run DIFF 0.0000 + a documented cross-vendor mean ~30–40, NOT strict zero-diff).

## Constraints (HARD)
- APPEND-ONLY to engine/pcg/pcg.h (do NOT modify PCG1–5) + the showcase blocks in main.cpp/visual_test.mm + the
  verify.ps1 registration (+ extend tests/pcg_test.cpp ONLY if you add a PcgToRenderInstances count/provenance
  check — optional). Reuse the instanced-lit pipeline + grain/pt6 render helpers + math.h READ-ONLY. Do NOT modify
  any shader / existing header / existing golden. NO new shader, NO new RHI.
- This is a FLOAT slice: do NOT assert strict zero-diff cross-backend. The proof is two-run byte-identical +
  provenance + coherent; the controller handles the Metal two-run 0.0000 + cross-vendor mean + eyeball.
- Branch `fix-issue-22-pcg6`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle pcg_test'`
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--pcg6-field-shot` exits 0, the proof lines
  print, two renders byte-identical, the provenance instance-count matches `PcgToRenderInstances(Generate(...))`,
  the empty-graph no-op holds, AND (if you touched it) pcg_test passes. Commit message via a temp file +
  `git commit -F` (use the Bash tool heredoc). Commit to the branch and STOP. Report: commit hash, proof output,
  image path, confirmation both renderers use identical graph/seed/baseRadius/camera, and the rendered instance
  count. (The CONTROLLER bakes the Metal golden, confirms Metal two-run 0.0000 + cross-vendor mean, eyeballs the
  PNG for a coherent lit scattered-rock field. This COMPLETES flagship #22.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
