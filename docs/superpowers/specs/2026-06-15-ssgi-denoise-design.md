# Slice BR — SSGI Spatial Denoise (Phase 4 #18) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. Quality follow-on to
> BP (SSGI): an edge-preserving bilateral denoise that smooths the single-frame GI noise. Reuses the
> SSAO-blur (depth-aware blur) pattern.

**Goal:** The single-frame SSGI (Slice BP) is noisy (K=16 rays/pixel). Add a depth+normal-guided
BILATERAL blur of the SSGI indirect buffer to smooth the noise while preserving edges (so the red/green
color bleed stays crisp at surface boundaries but the floor pool is smooth). A NEW `--ssgi-denoise-shot`
showcase shows the denoised result; the raw `ssgi.png` (BP) stays frozen for A/B. Golden-verified on both
backends.

## Why bilateral + reuse SSAO blur

`shaders/ssao_blur.frag.hlsl` already does a depth-aware blur of the AO buffer. The SSGI denoise is the
same shape: a separable (or NxN) bilateral blur where each tap's weight = spatial Gaussian × an
EDGE-STOPPING term from the G-buffer (depth difference + normal difference), so the blur does not cross
geometry edges. Reuse the SSAO-blur structure + the G-buffer bindings (depth/normal at t3/s3). The blur
runs on the SSGI indirect buffer BEFORE the composite adds `indirect*albedo` to the scene.

## Design decisions (locked)

1. **Denoise math (extend engine/render/ssgi.h, header-only pure CPU, no backend symbols).** Add to
   `hf::render::ssgi`:
   - `float BilateralWeight(float spatialDist2, float depthCenter, float depthTap, float depthSigma,
     const math::Vec3& nCenter, const math::Vec3& nTap, float normalPower)` — the edge-stopping tap
     weight: `exp(-spatialDist2 / (2*spatialSigma^2)) * exp(-(depthDiff)^2 / (2*depthSigma^2)) *
     pow(max(dot(nCenter,nTap),0), normalPower)`. Document the formula + that taps across a depth/normal
     edge get ~0 weight. Pure, unit-testable.
   - A small `SsgiDenoiseParams` (kernel radius, spatialSigma, depthSigma, normalPower). Document defaults.

2. **Denoise shader `shaders/ssgi_denoise.frag.hlsl`.** A fullscreen pass (reuse `post.vert`) reading the
   SSGI indirect buffer + the G-buffer (depth/normal at t3/s3). For each pixel, accumulate neighbor taps
   in a `radius`×`radius` window (or separable H then V — pick + document), weighting each by
   `BilateralWeight`; normalize by the weight sum. Output the denoised indirect. Then the existing SSGI
   composite adds `denoisedIndirect * albedo` to the scene. EXISTING SSGI/SSR/SSAO shaders + their goldens
   UNTOUCHED (the denoise is an inserted pass on the SSGI path only, behind the new showcase flag — the
   raw `--ssgi-shot` path stays byte-identical so `ssgi.png` is unchanged). HLSL→SPIR-V→MSL via the
   existing toolchain.

3. **Showcase `--ssgi-denoise-shot <out>` (Vulkan) / `--ssgi-denoise` (Metal).** The SAME Cornell-style
   color-bleed scene as `--ssgi-shot`, but with the denoise pass inserted before composite → smoother
   floor pools, crisp edges. Print `ssgi-denoise: {radius:R, spatialSigma:..., depthSigma:...,
   normalPower:...}`. New golden `tests/golden/metal/ssgi_denoise.png` (Metal two runs DIFF 0.0000).
   Existing 42 image goldens UNTOUCHED (including `ssgi.png` — the raw path is unchanged; verify it stays
   byte-identical, that's the A/B baseline).

4. **Determinism.** The bilateral blur is a pure function of the SSGI buffer + G-buffer (no time/RNG). Two
   runs byte-identical.

5. **Tests `tests/ssgi_denoise_test.cpp` (pure CPU, no GPU):**
   - **BilateralWeight:** weight is 1 at the center (zero spatial dist, equal depth, equal normal); decays
     with spatial distance; → ~0 across a large depth difference; → ~0 across opposing normals
     (`dot<=0`); monotonic in each term.
   - **Blur sanity (CPU mini-model):** a small noisy 1D/2D signal with a uniform depth/normal field →
     bilateral blur reduces variance (smoother) and preserves the mean; with a depth EDGE in the middle,
     the blur does NOT mix across the edge (the two sides keep their distinct values).
   - **Determinism:** same input → same output.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `--ssgi-denoise-shot` (showcases). NO new feature string (the
   `screen-space-global-illumination` feature already covers SSGI; denoise is a quality variant) — OR add
   `ssgi-denoise` feature if you judge it distinct; pick + document. Default: showcase-only (one-line
   introspect delta).

## RHI seam additions (summary)
- **None.** The denoise reuses the SSGI/SSAO G-buffer + fullscreen-pass infra. New files
  (`shaders/ssgi_denoise.frag.hlsl`, `tests/ssgi_denoise_test.cpp`, the `ssgi.h` additions) add ZERO
  backend code symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Temporal accumulation/reprojection, à-trous multi-pass wavelet denoise (one bilateral pass — or a
separable H/V — is enough), variance-guided adaptive radius, firefly clamping beyond the bilateral,
denoising other buffers (SSR/SSAO already have their own). One depth+normal bilateral pass on SSGI,
golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 42) + new `ssgi_denoise_test` (BilateralWeight,
   blur sanity + edge preservation, determinism). Clean under `windows-msvc-asan`.
2. `--ssgi-denoise-shot` on Windows/Vulkan: controller visual review — the color bleed is visibly SMOOTHER
   than `--ssgi-shot` (less grainy floor pools) while edges stay crisp, coherent; the `ssgi-denoise: {...}`
   line is deterministic (two runs → byte-identical capture). Run under the AT Vulkan-validation gate →
   ZERO errors.
3. Metal: `visual_test --ssgi-denoise` → new golden `tests/golden/metal/ssgi_denoise.png`; two runs DIFF
   0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `ssgi_denoise.png`
   added; the other 42 byte-identical — CRITICALLY `ssgi.png` (the raw SSGI A/B baseline) is unchanged
   (proves the denoise is an additive new path, not a change to the raw SSGI).
5. Introspect JSON rebaked exactly `+--ssgi-denoise-shot` (+ `ssgi-denoise` feature only if you chose to
   add it); introspect test updated; no other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `ssgi_denoise`
   image golden in the Mac round-trip loop.
