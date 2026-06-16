# Slice CF — Water Rendering (Gerstner waves + fresnel reflect/refract) — Phase 4 #31 — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A flagship visual the
> engine lacks; reuses SSR (AH) + the procedural sky + the scene-color RT. Deterministic at a FIXED time.

**Goal:** A planar water surface with animated Gerstner waves, a fresnel blend between a sky reflection
and a refracted/absorbed view of the scene below, and a sun specular glint. Rendered at a FIXED wave time
(no clock) so it is golden-stable. A `--water-shot` showcase shows objects half-submerged in rippling
water reflecting the sky; golden-verified on both backends.

## Why fixed-time (determinism)

Water waves animate with time → nondeterministic. Sample the wave field at a FIXED time `t` (a documented
constant). Same `t` + same wave params ⇒ identical surface + normals ⇒ identical pixels ⇒ goldens match.
Two runs bit-identical. (Live animated water in an interactive loop is trivial later; this verifies the
shading at a fixed instant.)

## Design decisions (locked)

1. **Wave + water math (engine/render/water.h, header-only pure CPU, no backend symbols).** Namespace
   `hf::render::water`. Mirrors `ssr.h`/`ssgi.h` (shared with the shader + `tests/water_test.cpp`).
   - `struct GerstnerWave { math::Vec2 dir; float amplitude, wavelength, steepness, speed; };` and a fixed
     small set (3-4 waves) summed. `math::Vec3 Displace(float x, float z, float t, span<const GerstnerWave>)`
     — Gerstner horizontal+vertical displacement summed over the waves. `math::Vec3 Normal(float x, float z,
     float t, span<...>)` — the analytic surface normal from the summed wave partial derivatives (NOT a
     finite difference — use the Gerstner tangent/bitangent cross product). Document the formulas.
   - `float Fresnel(float NdotV, float f0)` — Schlick fresnel `f0 + (1-f0)*(1-NdotV)^5` (f0 ~0.02 for
     water). Shared with the shader.
   - `math::Vec3 RefractTint(float depth, const math::Vec3& shallowColor, const math::Vec3& deepColor,
     float absorption)` — depth-based water color (Beer-Lambert-ish lerp/exp). Document.

2. **Water mesh + shader.** A water grid mesh (NxN, e.g. 128x128 over the play area) displaced in the
   VERTEX shader by `water::Displace(x,z,t,...)`; the FRAGMENT shader computes the wave normal
   (`water::Normal`), then: reflect the view ray about the normal → sample the procedural SKY for the
   reflection color (reuse the existing sky/`sky.frag` ray→sky function); compute a refraction color by
   sampling the SCENE-COLOR RT at a normal-perturbed UV (the scene rendered first into the HDR RT, like
   SSR/SSGI use it) tinted by `RefractTint` using the scene depth below the water (depth-based absorption);
   `fresnel = Fresnel(NdotV)`; `out = lerp(refractColor, reflectColor, fresnel) + sunSpecular`. Sun glint =
   a sharp specular from the directional light about the wave normal. New `shaders/water.{vert,frag}.hlsl`;
   HLSL→SPIR-V→MSL via the existing toolchain. Bindings reuse the scene-color RT + G-buffer/depth (t0/s0,
   t3/s3 like SSR) + the FrameData (sky params, light, camera). EXISTING shaders + their goldens UNTOUCHED.

3. **Showcase `--water-shot <out>` (Vulkan) / `--water` (Metal).** A scene: a few objects (cubes/spheres,
   the duck) partially submerged at the water level, the water plane covering the area, the procedural sky
   above, directional light for the glint. Render the opaque scene into the HDR RT first, then the water
   surface on top (reflecting the sky, refracting/absorbing the submerged objects), fixed camera + FIXED
   wave time. Print `water: {waves:N, time:T, gridN:128}` (deterministic). New golden
   `tests/golden/metal/water.png` (Metal two runs DIFF 0.0000). Existing 52 image goldens UNTOUCHED.

4. **Determinism.** Fixed wave set, FIXED time `t`, fixed camera/light, no clock/RNG. Two runs
   byte-identical.

5. **Tests `tests/water_test.cpp` (pure CPU, no GPU):**
   - **Gerstner displacement:** at `t=0` a known single wave displaces a sample point by the expected
     amount; summing N waves is the sum of individual displacements; deterministic at fixed t.
   - **Analytic normal:** the `water::Normal` is unit length; on a FLAT water (zero amplitude) it is
     `+Y`; on a known wave the normal tilts the expected direction; cross-check the analytic normal against
     a finite-difference of `Displace` at a few points (within tolerance — proves the analytic derivative
     is right).
   - **Fresnel:** `Fresnel(1, f0) == f0` (head-on = base reflectance), `Fresnel(0, f0) == 1` (grazing =
     full reflection), monotonic.
   - **RefractTint:** depth 0 → shallowColor, large depth → ~deepColor; monotonic absorption.
   - **Determinism:** same (x,z,t) → same Displace/Normal across runs.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `water-rendering` (features) + `--water-shot` (showcases).

## RHI seam additions (summary)
- **None.** The water mesh draws through the existing pipeline path; the shader reuses the scene-color RT +
  G-buffer + FrameData bindings (like SSR/SSGI). New files (`engine/render/water.h`,
  `shaders/water.*.hlsl`, `tests/water_test.cpp`) add ZERO backend code symbols. Seam grep stays at
  baseline (2).

## Out of scope (YAGNI)
Animated/interactive water (fixed-time only), planar reflection rendering (a second scene pass — use the
sky reflection + SSR-style scene refraction instead of a full mirror pass; full planar reflection is a
future slice), caustics, foam/shoreline, buoyancy physics coupling, tessellation/FFT ocean, underwater
post (fog/distortion when the camera is below water), wet-surface darkening. One fixed-time Gerstner water
plane with fresnel reflect(sky)/refract(scene) + sun glint, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 52) + new `water_test` (Gerstner displacement,
   analytic-vs-finite-diff normal, fresnel, refract tint, determinism). Clean under `windows-msvc-asan`.
2. `--water-shot` on Windows/Vulkan: controller visual review — a recognizable rippling water surface
   (wave normals visible as varying reflection, sky reflected, submerged objects refracted/tinted, a sun
   glint), coherent; the `water: {...}` line is deterministic (two runs → byte-identical capture). Run
   under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --water` → new golden `tests/golden/metal/water.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `water.png` added; the
   other 52 byte-identical.
5. Introspect JSON rebaked exactly `+water-rendering` + `--water-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `water` image
   golden in the Mac round-trip loop.
