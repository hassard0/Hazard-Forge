# Slice BP — Screen-Space Global Illumination (SSGI) — Phase 4 #16 (depth pivot) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A "Lumen-lite"
> indirect-diffuse bounce; the highest-impact step toward the rendering north star. Builds directly on
> SSR (Slice AH) + the shared G-buffer (SSAO/SSR/decal).

**Goal:** Add one bounce of indirect diffuse lighting via screen space. For each pixel, trace K rays in
the hemisphere about the surface normal, march them in screen space against the G-buffer (exactly like
SSR), and on a hit sample the ALREADY-LIT HDR scene color as incoming radiance; accumulate cosine-weighted
indirect diffuse, multiply by albedo, and ADD it to the scene. A `--ssgi-shot` color-bleed scene (a
strongly-colored panel bleeding onto a neutral neighbor) is golden-verified on both backends. Single-frame
and DETERMINISTIC (fixed kernel + baked dither, NO temporal accumulation).

## Why this reuses SSR (de-risked)

`shaders/ssr.frag.hlsl` already: reconstructs view-space P + N from the G-buffer (`view normal.xyz +
linDepth.w` at t3/s3), ray-marches in view space, projects each step to a screen UV (`ProjectToUV`),
depth-compares against the G-buffer, binary-searches the hit, and samples the HDR scene color (t0/s0,
via `BindTexturePair`) — masked + edge-faded, baked-dither (no RNG). SSGI is the SAME march, but:
(a) instead of ONE mirror ray `reflect(V,N)`, trace K rays in the cosine-weighted hemisphere about N;
(b) on hit, the sampled scene color is treated as incoming radiance (diffuse), not a mirror reflection;
(c) accumulate `sum(hitRadiance) / K` → indirect diffuse irradiance. Reuse the SSR reconstruction +
march helpers verbatim where possible.

## Design decisions (locked)

1. **SSGI math (engine/render/ssgi.h, header-only pure CPU, no backend symbols).** Namespace
   `hf::render::ssgi`. Mirrors `ssr.h` (shared with the shader + `tests/ssgi_test.cpp`).
   - `math::Vec3 HemisphereDir(int i, int K, const math::Vec3& normal)` — the i-th of K FIXED
     cosine-weighted hemisphere directions oriented to `normal` (build a TBN from `normal` + a stable
     tangent; distribute samples via a fixed low-discrepancy set — e.g. a baked Hammersley/golden-angle
     spiral mapped to a cosine hemisphere). Deterministic; document the distribution.
   - Reuse `ssr::ReconstructViewPos` / the view↔screen projection (don't duplicate — include/share the
     SSR header or factor a common reconstruction). Document the reuse.
   - `math::Vec3 AccumulateIndirect(const std::vector<math::Vec3>& hitRadiances)` — average over K
     (cosine weight is baked into the cosine-distributed sampling, so a simple mean is the Monte-Carlo
     estimator; document the estimator + that misses contribute 0 / an optional ambient fallback).
   - A small `SsgiParams` struct (K, marchDist, thickness, intensity, ...) mirroring `SsrParams`.

2. **SSGI shader `shaders/ssgi.frag.hlsl` + composite.** A fullscreen pass (reuse `post.vert` /the SSR
   pass scaffolding). Per pixel: reconstruct P,N from the G-buffer (skybox/background pixels excluded like
   SSR); for K hemisphere dirs, march in view space (reuse the SSR march + binary-search), on hit sample
   `gScene` as incoming radiance; accumulate; output `indirect = (sum/K) * intensity`. A composite step
   ADDS `indirect * albedo` to the scene HDR color (the SSGI pass can output indirect radiance and the
   existing composite pattern — like `ssr_composite` — adds it; or fold into one pass; pick + document).
   Bindings mirror SSR (scene HDR t0/s0, G-buffer t3/s3 via `BindTexturePair`); HLSL→SPIR-V→MSL via the
   existing toolchain. EXISTING gbuffer/lit/ssao/ssr/bloom shaders + their goldens UNTOUCHED.

3. **Showcase `--ssgi-shot <out>` (Vulkan) / `--ssgi` (Metal)** — a deterministic COLOR-BLEED scene: a
   strongly saturated panel/wall (e.g. a red and a green vertical quad) flanking a neutral white
   floor/box, lit so the colored panels bleed red/green indirect light onto the neutral surfaces (the
   classic Cornell-style GI tell). Fixed camera/lights. Render WITH SSGI; print `ssgi: {rays:K, marchDist:
   ..., intensity:...}`. New golden `tests/golden/metal/ssgi.png` (Metal two runs DIFF 0.0000). Existing
   40 image goldens UNTOUCHED. (Optionally also an `--ssgi-shot-off` toggle for an A/B controller review,
   like `--ssao-shot-off` — nice-to-have, document if added.)

4. **Determinism.** Fixed K-ray kernel (pure function of pixel/sample index, NO time/RNG; a baked dither
   may rotate the kernel per-pixel by a FIXED 4x4 pattern like SSR — no RNG). Single frame, NO temporal
   reprojection/accumulation (that's a future TAA-style enhancement). Two runs byte-identical.

5. **Tests `tests/ssgi_test.cpp` (pure CPU, no GPU):**
   - **Hemisphere kernel:** all K `HemisphereDir` are unit length, lie in the hemisphere of `normal`
     (`dot(dir,normal) >= 0`), and are deterministic (same i,K,normal → same dir); for `normal=+Z` the
     set matches hand-checked expectations; the average direction is ~`normal` (no bias).
   - **Reconstruction reuse:** `ssr::ReconstructViewPos`↔`ViewToScreenUV` round-trip still holds (guards
     the shared reconstruction).
   - **Accumulate:** mean of K hit radiances; all-miss → 0 (or the documented ambient).
   - **Color-bleed sanity (CPU mini-model):** given a tiny synthetic "scene" (a red emitter sample in
     some hemisphere dirs, neutral elsewhere), `AccumulateIndirect` yields a reddish indirect — i.e.
     color bleeds. Document the mini-model.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `screen-space-global-illumination` (features) + `--ssgi-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses the SSR/SSAO G-buffer + HDR-scene `BindTexturePair` + the fullscreen-composite path.
  New files (`engine/render/ssgi.h`, `shaders/ssgi.frag.hlsl`, `tests/ssgi_test.cpp`) add ZERO backend
  code symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Temporal accumulation / reprojection (single-frame only), denoising, multi-bounce, ray-traced/world-space
GI, horizon-based GI specifics, specular GI (diffuse only), per-object GI probes (we have reflection
probes already), importance sampling beyond the fixed kernel, half-res + upsample. One single-frame
fixed-kernel screen-space diffuse bounce, color-bleed golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 40) + new `ssgi_test` (kernel, reconstruction
   reuse, accumulate, color-bleed mini-model). Clean under `windows-msvc-asan`.
2. `--ssgi-shot` on Windows/Vulkan: controller visual review — clear COLOR BLEED (the colored panels tint
   the neutral neighbor red/green) vs a flat-ambient look, lit + coherent; the `ssgi: {...}` line is
   deterministic (two runs → byte-identical capture). Run under the AT Vulkan-validation gate → ZERO
   errors. (If `--ssgi-shot-off` added, the A/B difference is the bleed.)
3. Metal: `visual_test --ssgi` → new golden `tests/golden/metal/ssgi.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `ssgi.png` added; the
   other 40 byte-identical (existing SSR/SSAO/etc untouched).
5. Introspect JSON rebaked exactly `+screen-space-global-illumination` + `--ssgi-shot`; introspect test
   updated; no other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `ssgi` image
   golden in the Mac round-trip loop.
