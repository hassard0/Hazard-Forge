# Slice CZ — Subsurface Scattering (screen-space separable SSS) (Phase 4 #47) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A genuinely-absent AAA
> material feature (skin/wax/marble/foliage): screen-space separable subsurface scattering (Jimenez 2015) —
> light diffuses under the surface, softening shading on flagged subsurface materials. Reuses the G-buffer +
> scene color. Deterministic, with a clean sssStrength=0 no-op proof.

**Goal:** A screen-space separable SSS post pass — for pixels of a flagged subsurface material, blur the
diffuse lit color along a depth-aware diffusion kernel (a 1D horizontal pass + a 1D vertical pass), scaled by
the surface's subsurface width and the view depth, so light bleeds under the surface (the soft translucent
glow of skin/wax). A `--sss-shot` showcase shows a subsurface sphere with soft under-surface light bleed (vs
hard direct shading); golden-verified on both backends. The existing lit/post path + its goldens are
UNTOUCHED (SSS is a NEW path behind the showcase flag). The pass carries its proof: with `sssStrength == 0`
(or `sssWidth == 0`) the diffusion kernel collapses to the center tap → the SSS-applied image is
BYTE-IDENTICAL to the non-SSS lit render — asserted internally, fail loudly on any diff.

## Why this is render-safe (the sssStrength=0 / zero-width no-op proof)

The separable SSS gathers `N` taps along the blur axis weighted by the diffusion profile, with the tap step
scaled by `sssStrength * sssWidth / depth`. With `sssStrength == 0` (or `sssWidth == 0`) the step is 0 → every
tap lands on the center pixel → the normalized weighted sum equals the center color exactly → the output
equals the non-SSS render. So the showcase INTERNALLY renders with `sssStrength = 0` and asserts BYTE-IDENTICAL
(SHA) to the engine's standard lit render of the same scene — proving the two-pass blur is a pure pass-through
at zero strength (no constant bias, no normalization error, no off-by-one) — then renders the real
`sssStrength > 0` version as the golden. (Same internal-assert discipline as CN/CO/CP/CR/CS/CT/CW/CX.) The unit
test additionally proves the diffusion-weight profile is positive, decreasing with distance, normalized, and
the depth-aware falloff cuts a tap across a depth discontinuity.

## Design decisions (locked)

1. **SSS math (engine/render/sss.h, header-only pure CPU, no backend symbols).** Namespace `hf::render::sss`.
   Mirrors `dof.h`/`gtao.h`/`auto_exposure.h` (shared with the shader + the unit test).
   - `float DiffusionWeight(float distancePx, float sssWidthPx)` — the subsurface diffusion profile weight at a
     tap `distancePx` from the center, for a profile of width `sssWidthPx` (a sum-of-Gaussians skin profile OR
     a single normalized Gaussian `exp(-(d/w)^2 * k)` — document the profile + constants). Positive, decreasing
     in `distancePx`, peaks at 0.
   - `float DepthFalloff(float tapDepth, float centerDepth, float depthScale)` — a depth-aware weight that
     cuts a tap whose view depth differs from the center by more than ~the SSS width (so SSS does NOT bleed
     across a silhouette/depth discontinuity); 1 at equal depth, → 0 for large depth difference. Document.
   - `math::Vec3 BlurAxis(<color-sampler>, <depth-sampler>, <mask-sampler>, math::Vec2 uv, math::Vec2 axisPx,
     float sssWidthPx, float sssStrength, int taps, float depthScale)` — one separable pass: gather `taps`
     samples stepping along `axisPx` (horizontal or vertical), each weighted by `DiffusionWeight × DepthFalloff
     × subsurfaceMask`, normalized; the step length scales with `sssStrength * sssWidthPx / centerDepth`. At
     `sssStrength == 0` or `sssWidthPx == 0` → returns the center color EXACTLY (document the early-out). Shared
     with the shader.
   - Pure, deterministic, no RNG/time. Document the two-pass (H then V) separable approximation.

2. **SSS shader `shaders/sss_blur.frag.hlsl` (NEW fullscreen pass, run TWICE — horizontal then vertical).**
   Reuse `post.vert`. Reads the resolved lit scene color + the G-buffer depth + a subsurface MASK (a flagged
   material channel — e.g. a material-id/mask in the G-buffer, or render the subsurface objects' mask; document
   how the mask is sourced). Runs `sss::BlurAxis` along the pass axis (a uniform selects H or V). Output the
   blurred color; non-subsurface pixels pass through unchanged (mask 0 → center only). The two passes compose
   the separable 2D diffusion. The EXISTING lit/post path + its goldens stay BYTE-IDENTICAL (SSS is a NEW path
   behind the showcase flag). Bindings mirror DoF/SSR (scene color t0/s0, G-buffer depth t3/s3 via
   `BindTexturePair`) + a small uniform for `sssWidth/sssStrength/taps/depthScale/axis`. No new RHI seam.
   HLSL→SPIR-V→MSL via the existing toolchain.

3. **Showcase `--sss-shot <out>` (Vulkan) / `--sss` (Metal).** A scene with a clearly SUBSURFACE object (a
   sphere/bust flagged subsurface, lit so one side is in shadow) showing the soft under-surface light bleed —
   light wrapping around the terminator + a warm translucent glow — distinct from the hard direct-lit
   non-subsurface objects beside it. Fixed camera, fixed lights, fixed SSS params. Print
   `sss: {width:W, strength:S, taps:N}` (deterministic). INTERNALLY render with `sssStrength = 0` and assert
   BYTE-IDENTICAL (SHA) to the non-SSS lit render — fail loudly on any diff. New golden
   `tests/golden/metal/sss.png` (Metal two runs DIFF 0.0000). Existing 67 image goldens UNTOUCHED.

4. **Determinism.** Fixed lights/camera/SSS params, fixed tap pattern (no RNG/time; a baked dither allowed —
   no RNG). Two runs byte-identical.

5. **Tests `tests/sss_test.cpp` (pure CPU, no GPU):**
   - **Zero strength/width = identity:** `BlurAxis(..., sssStrength=0, ...)` and `(..., sssWidthPx=0, ...)`
     return the center color EXACTLY for any input.
   - **Diffusion profile:** `DiffusionWeight` positive, peaks at distance 0, strictly decreasing in distance;
     a wider `sssWidthPx` spreads the weight (a far tap weighted more than for a narrow width).
   - **Depth-aware falloff:** `DepthFalloff` == 1 at equal depth, → ~0 for a tap far in depth from the center
     (no bleed across a silhouette); monotone.
   - **Blur behavior:** a flat subsurface region with a bright center → the center spreads to neighbors
     (energy conserved within the mask, normalized); a non-subsurface (mask 0) pixel → unchanged.
   - **Determinism:** same inputs → same result.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `subsurface-scattering` (features) + `--sss-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses the SSR/DoF G-buffer (depth) + scene-color RT + the fullscreen two-pass path (like the SSGI
  denoise / bloom which already do separable/multi-pass fullscreen blurs). The subsurface mask reuses an
  existing G-buffer channel / material flag (document). New files (`engine/render/sss.h`,
  `shaders/sss_blur.frag.hlsl`, `tests/sss_test.cpp`) add ZERO backend code symbols. Seam grep stays at
  baseline (2).

## Out of scope (YAGNI)
Physically-based multi-layer skin (single diffusion profile), transmission/back-lighting through thin parts
(thickness map — note future), pre-integrated SSS on the BRDF, separable SSS in linear-depth-correct world
units beyond the screen-space approximation, per-material diffusion-profile authoring, SSS in the shadow pass,
texture-space SSS. One screen-space two-pass separable diffusion SSS with a zero-strength byte-identical proof
+ an analytic unit test, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 67) + new `sss_test` (zero-strength/width identity,
   diffusion profile properties, depth-aware falloff, blur energy/mask behavior, determinism). Clean under
   `windows-msvc-asan`.
2. **Zero-strength no-op proof + visual:** `--sss-shot` on Vulkan: the subsurface object shows soft
   under-surface light bleed (a recognizable SSS glow, light wrapping the terminator), coherent vs the hard
   non-subsurface objects; the INTERNAL sssStrength=0 render is BYTE-IDENTICAL (SHA) to the non-SSS lit render;
   the `sss: {...}` line is deterministic (two runs → byte-identical capture). Run under the AT Vulkan-
   validation gate → ZERO errors.
3. Metal: `visual_test --sss` → new golden `tests/golden/metal/sss.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `sss.png` added; the other
   67 byte-identical.
5. Introspect JSON rebaked exactly `+subsurface-scattering` + `--sss-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `sss` image golden in
   the Mac round-trip loop. (NOTE: `sss_blur.frag.hlsl` is a NEW shader not shared by any existing golden — no
   re-bake of other Metal goldens needed; gate the new golden's Metal compare on the EXIT CODE.)
