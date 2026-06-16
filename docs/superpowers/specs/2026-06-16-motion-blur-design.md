# Slice CN — Per-Object + Camera Motion Blur (Phase 4 #38) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A genuinely-absent AAA
> post effect; reuses the TAA reprojection (prev/cur view-proj) + the G-buffer depth + the post pass.
> Deterministic at a FIXED frame, with a clean zero-velocity equivalence proof.

**Goal:** A velocity-buffer motion-blur post pass — moving geometry (and a moving camera) streak along their
screen-space motion vector, while static regions stay sharp. A `--motionblur-shot` showcase shows a scene
with a FIXED nonzero per-frame motion producing visible directional blur; golden-verified on both backends.
The pass carries a built-in correctness proof: with ZERO motion (prev view-proj == cur view-proj, no object
motion) the blurred output is BYTE-IDENTICAL to the un-blurred scene (the gather collapses to the center
tap) — asserted internally, fail loudly on any diff.

## Why this is render-safe (the zero-velocity equivalence proof)

The blur gathers `N` taps along each pixel's screen-space velocity vector `v`. When `v == 0` everywhere
(static scene + static camera), every tap lands on the center pixel and the normalized weighted sum equals
the center color exactly → the output is byte-identical to the input. So the showcase INTERNALLY renders the
SAME scene with zero motion and asserts `motionblur(zeroVel) == sceneColor` (SHA) — proving the pass is a
pure pass-through when nothing moves (no energy added/lost, no off-by-one) — then renders the FIXED-motion
version as the golden. (Same internal-assert discipline as CL clustered==brute-force and CJ Hi-Z==frustum.)

## Design decisions (locked)

1. **Velocity + blur math (engine/render/motion_blur.h, header-only pure CPU, no backend symbols).** Namespace
   `hf::render::motionblur`. Mirrors `ssr.h`/`dof.h`/`cluster.h` (shared with the shader + the unit test).
   - `math::Vec2 ScreenVelocity(const math::Vec3& worldPos, const math::Mat4& prevViewProj,
     const math::Mat4& curViewProj, int screenW, int screenH)` — project `worldPos` by both matrices to NDC,
     take the screen-space delta (cur − prev) in PIXELS. Document the NDC→pixel convention. Zero when the two
     projections + the point are identical.
   - `math::Vec2 ClampVelocity(const math::Vec2& vPx, float maxBlurPx)` — clamp the velocity vector's length
     to `maxBlurPx` (preserve direction). Caps the blur extent (perf + artifact bound).
   - `float TapWeight(float tapDepth, float centerDepth, float tapDistPx, float velLenPx)` — the gather weight
     for a tap along the velocity: a tap contributes if it lies within the blur extent (`tapDistPx <=
     velLenPx`), with a soft depth-aware term so a nearer moving foreground correctly streaks OVER a static
     background and a static foreground is NOT smeared by a moving background (document the depth comparison —
     the standard McGuire-style fore/background classification, simplified). Normalized so zero velocity →
     center weight 1, all others 0.
   - Document the gather model + the exact formulas. Pure functions, no RNG, no time.

2. **Motion-blur shader `shaders/motion_blur.frag.hlsl` (NEW fullscreen pass).** Reuse `post.vert`. Per
   pixel: reconstruct the world/view position from the G-buffer depth (like SSR/SSAO/DoF at t3/s3), compute
   the screen velocity via `motionblur::ScreenVelocity` (prev/cur view-proj from a small uniform/push block,
   ≤128B), clamp to `maxBlurPx`; gather `N` fixed taps stepping along the velocity vector, each weighted by
   `TapWeight` (depth-aware); output the normalized sum. The DEFAULT post path + its goldens stay
   BYTE-IDENTICAL (this is a NEW path behind the showcase flag). Bindings mirror SSR/DoF (scene color t0/s0,
   G-buffer depth t3/s3 via `BindTexturePair`) + the prev/cur view-proj + `maxBlurPx`/`taps` uniform. No new
   RHI seam. HLSL→SPIR-V→MSL via the existing toolchain.

3. **Showcase `--motionblur-shot <out>` (Vulkan) / `--motionblur` (Metal).** A scene (ground + objects)
   where the CAMERA pans at a FIXED rate (prev view-proj is the camera one frame earlier at a fixed dt) — OR
   one object translates at a fixed velocity — producing a clear directional blur streak on the moving
   content while static content stays sharp. Fixed camera path, fixed dt, fixed taps. Print
   `motion-blur: {maxBlurPx:M, taps:N, velScale:S}` (deterministic). INTERNALLY also render the zero-motion
   version (prevViewProj := curViewProj) and assert it is BYTE-IDENTICAL (SHA) to the plain scene color
   (no-blur) — fail loudly on any diff. New golden `tests/golden/metal/motion_blur.png` (Metal two runs DIFF
   0.0000). Existing 58 image goldens UNTOUCHED.

4. **Determinism.** Fixed camera path + dt, fixed tap count, fixed `maxBlurPx`, no RNG (a baked per-pixel
   dither offset like SSR is allowed — no RNG). Two runs byte-identical.

5. **Tests `tests/motion_blur_test.cpp` (pure CPU, no GPU):**
   - **Zero velocity:** `ScreenVelocity` with `prevViewProj == curViewProj` → `(0,0)` for any point; and the
     gather (center-only) returns the center color (weight 1 at center, 0 elsewhere).
   - **Nonzero velocity direction:** a camera/point translation produces a velocity whose direction matches
     the motion and whose magnitude grows with the motion / dt; clamps at `maxBlurPx`.
   - **Clamp:** a huge motion → `ClampVelocity` caps the length at `maxBlurPx`, preserves direction.
   - **TapWeight gather:** a tap within the blur extent contributes; beyond it ~0; the depth-aware
     fore/background rule (a nearer moving tap streaks over a farther static center; a farther moving tap does
     NOT smear a nearer static center).
   - **Determinism:** same inputs → same velocity + weights.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `motion-blur` (features) + `--motionblur-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses the SSR/DoF G-buffer + scene-color RT + fullscreen-composite path; the prev/cur view-proj
  come from the existing TAA reprojection plumbing (FrameData / a small uniform). New files
  (`engine/render/motion_blur.h`, `shaders/motion_blur.frag.hlsl`, `tests/motion_blur_test.cpp`) add ZERO
  backend code symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Tile-based velocity dilation / max-velocity tiles (use the simple per-pixel gather — conservative + still
golden-stable), per-object velocity from skinned/animated geometry (camera + rigid-object velocity only),
multi-sample reconstruction filters, shutter-curve weighting, separable/two-pass, more than a single fixed
shutter. One fixed-frame velocity-gather motion-blur pass with a zero-velocity pass-through proof,
golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 58) + new `motion_blur_test` (zero velocity, nonzero
   direction/magnitude, clamp, tap-weight depth-aware gather, determinism). Clean under `windows-msvc-asan`.
2. **Zero-velocity equivalence proof + visual:** `--motionblur-shot` on Vulkan: the moving content shows a
   clear directional blur while static content stays sharp; the INTERNAL zero-motion render is BYTE-IDENTICAL
   (SHA) to the plain scene color; the `motion-blur: {...}` line is deterministic (two runs → byte-identical
   capture). Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --motionblur` → new golden `tests/golden/metal/motion_blur.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `motion_blur.png` added;
   the other 58 byte-identical.
5. Introspect JSON rebaked exactly `+motion-blur` + `--motionblur-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `motion_blur` image
   golden in the Mac round-trip loop.
