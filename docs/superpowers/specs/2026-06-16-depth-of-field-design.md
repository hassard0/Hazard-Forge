# Slice CG — Depth of Field (Phase 4 #32) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A distinct post
> effect the engine lacks; reuses the G-buffer depth + the post pass. Deterministic.

**Goal:** A depth-of-field post pass — a focal plane stays sharp while nearer/farther geometry blurs by a
circle-of-confusion (CoC) derived from the scene depth and a thin-lens model. A `--dof-shot` showcase
shows a scene with a sharp focal subject and blurred fore/background; golden-verified on both backends.

## Design decisions (locked)

1. **CoC math (engine/render/dof.h, header-only pure CPU, no backend symbols).** Namespace `hf::render::dof`.
   Mirrors `ssr.h`/`water.h` (shared with the shader + `tests/dof_test.cpp`).
   - `float CircleOfConfusion(float depth, float focalDist, float aperture, float focalLength,
     float maxCoCpx)` — the thin-lens CoC in pixels: `coc = aperture * |focalLength * (depth - focalDist)|
     / (depth * (focalDist - focalLength))`, clamped to `[0, maxCoCpx]` (use the standard thin-lens
     formula; document + simplify to a robust form — the key properties: CoC == 0 at `depth == focalDist`,
     grows with `|depth - focalDist|`, separately for near (`depth < focalDist`) and far). Document the
     exact formula + units (depth in view-linear units, CoC in pixels).
   - `float BlurWeight(float tapCoCpx, float tapDistPx)` — a gather weight: a tap contributes if its CoC
     covers the center (`tapDistPx <= tapCoCpx`), normalized — document the gather model (scatter-as-gather:
     a neighbor with a large CoC "spreads" onto the center). Shared with the shader.

2. **DoF shader `shaders/dof.frag.hlsl`.** A fullscreen pass (reuse `post.vert`) reading the resolved scene
   color + the G-buffer depth (reconstruct view-linear depth like SSR/SSAO at t3/s3). Per pixel: compute
   the center CoC; gather a disk of neighbor taps, each weighted by `BlurWeight(neighborCoC, dist)` so
   in-focus pixels stay sharp and out-of-focus regions blur by their CoC; depth-aware so the sharp focal
   subject doesn't bleed onto the blurred background (and a near blurred object correctly spreads over the
   focal subject). Output the DoF-composited color. Bindings mirror SSR/SSAO (scene t0/s0, G-buffer t3/s3
   via `BindTexturePair`) + a small uniform/push constant for `focalDist/aperture/focalLength/maxCoC`.
   EXISTING shaders + their goldens UNTOUCHED (DoF is a NEW path behind the showcase flag; the default post
   path is unchanged → existing goldens byte-identical). HLSL→SPIR-V→MSL via the existing toolchain.

3. **Showcase `--dof-shot <out>` (Vulkan) / `--dof` (Metal).** A scene with clear depth separation — e.g.
   a row of objects receding from camera (the duck/spheres/cubes at varying distances) — with the focal
   distance set on a MIDDLE object so the FOREGROUND and BACKGROUND objects are visibly blurred while the
   focal object is crisp. Fixed camera, fixed lens params. Print `dof: {focalDist:F, aperture:A, maxCoC:M}`
   (deterministic). New golden `tests/golden/metal/dof.png` (Metal two runs DIFF 0.0000). Existing 53 image
   goldens UNTOUCHED.

4. **Determinism.** Fixed lens params, fixed scene/camera, fixed gather kernel (a fixed disk of taps, no
   time/RNG; a baked dither rotation per pixel like SSR is allowed — no RNG). Two runs byte-identical.

5. **Tests `tests/dof_test.cpp` (pure CPU, no GPU):**
   - **CoC zero at focus:** `CircleOfConfusion(focalDist, focalDist, ...) == 0` (sharp at the focal plane).
   - **CoC grows with defocus:** `|depth - focalDist|` larger → larger CoC; monotonic on each side
     (near + far); clamped at `maxCoCpx`.
   - **Near vs far:** a near object (`depth < focalDist`) and a far object (`depth > focalDist`) both have
     CoC > 0; the sign/branch is handled (no NaN at `depth → 0` or `depth → focalLength`).
   - **BlurWeight gather:** a tap inside its CoC contributes; outside contributes ~0; a focal (CoC≈0)
     neighbor doesn't blur the center.
   - **Determinism:** same inputs → same CoC.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `depth-of-field` (features) + `--dof-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses the SSR/SSAO G-buffer + scene-color RT + fullscreen-composite path. New files
  (`engine/render/dof.h`, `shaders/dof.frag.hlsl`, `tests/dof_test.cpp`) add ZERO backend code symbols.
  Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Bokeh-shaped (hexagonal/textured) aperture, separable/two-pass DoF, near-field dilation/tile-based
acceleration, auto-focus, focus pulling/animation, chromatic bokeh fringing, foreground-coverage alpha
handling beyond the basic depth-aware gather. One thin-lens CoC depth-gather DoF pass, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 53) + new `dof_test` (CoC at focus, defocus growth,
   near/far, blur-weight gather, determinism). Clean under `windows-msvc-asan`.
2. `--dof-shot` on Windows/Vulkan: controller visual review — the focal object is crisp while the
   foreground + background objects are visibly blurred (a clear DoF look), coherent; the `dof: {...}` line
   is deterministic (two runs → byte-identical capture). Run under the AT Vulkan-validation gate → ZERO
   errors.
3. Metal: `visual_test --dof` → new golden `tests/golden/metal/dof.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `dof.png` added; the
   other 53 byte-identical.
5. Introspect JSON rebaked exactly `+depth-of-field` + `--dof-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `dof` image
   golden in the Mac round-trip loop.
