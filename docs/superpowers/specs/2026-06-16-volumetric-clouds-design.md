# Slice CH — Volumetric Clouds (Phase 4 #33) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A flagship visual that
> rounds out the sky/atmosphere suite (procedural sky + water[CF] + clouds). DISTINCT from the existing
> ground-level volumetric light-shaft fog (that's an in-scene fog pass; this is a raymarched cloud LAYER in
> the sky dome). Deterministic at a FIXED time/noise.

**Goal:** A raymarched volumetric cloudscape — a cumulus density layer between two altitudes, lit by the
sun (Beer-Lambert extinction + a forward-scattering phase), composited as the sky background behind the
scene. A `--clouds-shot` showcase shows the scene under a sky full of volumetric clouds; golden-verified
on both backends. Sampled at a FIXED time so it is golden-stable.

## Why fixed-time + baked noise (determinism)

Clouds animate (wind) and use noise that, if RNG-seeded at runtime, would be nondeterministic. Sample the
cloud density at a FIXED time `t` and use a DETERMINISTIC noise (a fixed-hash value/Worley noise — pure
function of position, no RNG). Same `t` + same noise ⇒ identical clouds ⇒ goldens match. Two runs
bit-identical.

## Design decisions (locked)

1. **Cloud math (engine/render/clouds.h, header-only pure CPU, no backend symbols).** Namespace
   `hf::render::clouds`. Mirrors `ssr.h`/`water.h` (shared with the shader + `tests/clouds_test.cpp`).
   - `float Noise3(math::Vec3 p)` — a deterministic value-noise (integer-lattice hash + trilinear
     interpolation; NO RNG). `float Fbm(math::Vec3 p, int octaves)` — fractal sum of `Noise3` (a few
     octaves, fixed lacunarity/gain). Document the hash + params.
   - `float Density(math::Vec3 worldPos, float t, float slabBottom, float slabTop)` — the cloud density at
     a world point + time: an FBM shaped by a height-gradient (0 at the slab edges, max in the middle) and
     a coverage threshold (`max(fbm - coverage, 0)`), advected by `t` (offset the sample by `t*wind`).
     Returns 0 outside the slab. Deterministic.
   - `float Beer(float opticalDepth)` = `exp(-opticalDepth)` (extinction). `float HenyeyGreenstein(float
     cosAngle, float g)` — the forward-scatter phase function. Shared with the shader.

2. **Cloud shader `shaders/clouds.frag.hlsl`.** A fullscreen pass (reuse `post.vert` / the sky pass
   scaffolding). For each pixel, reconstruct the view ray from the camera (FrameData camera basis, like the
   procedural sky does). RAY-MARCH the ray through the cloud SLAB (intersect the ray with the two altitude
   planes; march N fixed steps between): at each step sample `clouds::Density`, accumulate transmittance
   (`Beer`) + in-scattered sunlight (a short secondary march toward the sun for the light's optical depth,
   or a cheap density-gradient approximation — document; `HenyeyGreenstein` phase toward the sun). Output
   the cloud color + alpha (coverage). Composite: where there is NO scene geometry (the sky background —
   check depth/the G-buffer like SSR, or render clouds as the sky in the existing sky pass), blend the
   clouds over the procedural sky; the lit scene in front is unaffected (clouds are distant background).
   EXISTING sky/scene shaders + their goldens UNTOUCHED (clouds are a NEW path behind the showcase flag).
   HLSL→SPIR-V→MSL via the existing toolchain.

3. **Showcase `--clouds-shot <out>` (Vulkan) / `--clouds` (Metal).** The standard lit+shadowed scene with
   the sky REPLACED/augmented by the volumetric cloudscape (a layer of cumulus clouds lit by the sun),
   fixed camera + FIXED time `t`. Print `clouds: {steps:N, coverage:C, time:T}` (deterministic). New golden
   `tests/golden/metal/clouds.png` (Metal two runs DIFF 0.0000). Existing 54 image goldens UNTOUCHED.

4. **Determinism.** Fixed `t`, deterministic noise (no RNG), fixed march step count, fixed camera/sun. Two
   runs byte-identical.

5. **Tests `tests/clouds_test.cpp` (pure CPU, no GPU):**
   - **Noise determinism + range:** `Noise3(p)` is deterministic (same p → same value) and in `[0,1]` (or
     the documented range); `Fbm` is deterministic + bounded.
   - **Density slab:** `Density` is 0 below `slabBottom` and above `slabTop`; non-zero (for a covered
     region) inside; the height gradient makes mid-slab denser than the edges.
   - **Coverage:** raising the coverage threshold reduces total density (fewer/thinner clouds);
     deterministic.
   - **Beer + phase:** `Beer(0) == 1`, `Beer(large) → 0`, monotonic; `HenyeyGreenstein` peaks forward
     (cosAngle=1) for g>0 and integrates sanely (hand-check a couple of values).
   - **Determinism:** same (p, t) → same `Density` across runs.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `volumetric-clouds` (features) + `--clouds-shot` (showcases).

## RHI seam additions (summary)
- **None.** The cloud pass reuses the sky/fullscreen pass + the FrameData (camera/sun) bindings. New files
  (`engine/render/clouds.h`, `shaders/clouds.frag.hlsl`, `tests/clouds_test.cpp`) add ZERO backend code
  symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Animated/wind-driven clouds (fixed-time only), multiple cloud layers, weather/coverage maps from a texture,
cloud shadows on the ground, atmospheric scattering / aerial perspective integration, temporal reprojection
of the clouds, half-res + upsample, a full physically-based sky model. One fixed-time raymarched cumulus
slab with Beer-Lambert + a phase function, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 54) + new `clouds_test` (noise determinism/range,
   density slab + gradient, coverage, Beer + phase, determinism). Clean under `windows-msvc-asan`.
2. `--clouds-shot` on Windows/Vulkan: controller visual review — a recognizable volumetric cloudscape
   (puffy sunlit cumulus in the sky over the scene, depth/shading from the raymarch), coherent; the
   `clouds: {...}` line is deterministic (two runs → byte-identical capture). Run under the AT Vulkan-
   validation gate → ZERO errors.
3. Metal: `visual_test --clouds` → new golden `tests/golden/metal/clouds.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `clouds.png` added; the
   other 54 byte-identical (incl. the existing volumetric fog golden + sky-based goldens — clouds are a new
   path).
5. Introspect JSON rebaked exactly `+volumetric-clouds` + `--clouds-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `clouds` image
   golden in the Mac round-trip loop.
