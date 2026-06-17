# Slice DF — Contrast-Adaptive Sharpening (CAS) (Phase 4 #52) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A clean, contained,
> genuinely-absent post effect (AMD FidelityFX CAS): an adaptive 3x3 sharpen that crisps edges without ringing
> — pairs naturally with TAA (which softens). Pure fullscreen, no new RHI. Deterministic, with a clean
> sharpness=0 no-op proof. (Pre-consolidation slot: chosen for reliability — the bolder glossy-probe-mip /
> GI work comes after consolidation #20.)

**Goal:** A Contrast-Adaptive Sharpening post pass — per pixel, the 3x3 neighborhood's local min/max drives an
adaptive sharpening amount (more sharpening in low-contrast regions, less near edges to avoid ringing), scaled
by a `sharpness` control. A `--cas-shot` showcase shows the scene visibly crisper than the unsharpened render;
golden-verified on both backends. The existing post path + its goldens are UNTOUCHED (CAS is a NEW path behind
the showcase flag). The pass carries its proof: with `sharpness == 0` the sharpen weight is 0 → the output is
BYTE-IDENTICAL to the input (unsharpened) render — asserted internally, fail loudly on any diff.

## Why this is render-safe (the sharpness=0 no-op proof)

CAS computes `out = center + sharpenAmount(neighborhood, sharpness)` where the sharpen contribution scales
linearly with `sharpness`. With `sharpness == 0` the contribution is 0 → `out = center` exactly → the
CAS-applied image equals the unsharpened render. So the showcase INTERNALLY renders with `sharpness = 0` and
asserts BYTE-IDENTICAL (SHA) to the engine's standard (un-sharpened) render of the same scene — proving the
sharpen is a pure pass-through at zero (no constant bias, no clamp drift) — then renders the real
`sharpness > 0` version as the golden. (Same internal-assert discipline as the other slices.) The unit test
additionally proves the CAS kernel: identity at sharpness 0, the adaptive weight, and the no-overshoot clamp.

## Design decisions (locked)

1. **CAS math (engine/render/cas.h, header-only pure CPU, no backend symbols).** Namespace `hf::render::cas`.
   Mirrors `dof.h`/`gtao.h`/`color_grade.h` (shared with the shader + the unit test).
   - `float CasWeight(float minL, float maxL, float sharpness)` — the FidelityFX CAS adaptive weight from the
     neighborhood luminance min/max: `amp = sqrt(clamp(min(minL, 1-maxL) / max(maxL, eps), 0, 1))`; the
     per-pixel sharpening strength `w = amp * lerp(-0.125, -0.2, sharpness)` (the standard CAS peak weight
     range; document the exact constants). At `sharpness == 0` the effective contribution is 0 (document the
     exact-zero path). Returns the negative-lobe weight for the 4 (or 8) neighbors.
   - `math::Vec3 CasSharpen(const math::Vec3& center, const math::Vec3& up, const math::Vec3& down, const
     math::Vec3& left, const math::Vec3& right, float sharpness)` — the CAS 3x3 (cross/box) sharpen:
     `out = (center + w*(up+down+left+right)) / (1 + 4*w)`, clamped to the neighborhood `[min, max]` per channel
     (the no-ringing clamp). At `sharpness == 0` → `w == 0` → `out == center` EXACTLY (document). Uses
     `CasWeight` from the per-channel or luma min/max of the neighborhood. Shared with the shader.
   - Document the exact FidelityFX CAS formulation used (the normalized weighted sum + the min/max clamp).
     Pure, deterministic, no RNG/time.

2. **CAS shader `shaders/cas.frag.hlsl` (NEW fullscreen pass).** Reuse `post.vert`. Reads the resolved scene
   color (the final post output), gathers the 3x3 (or cross) neighborhood, computes `cas::CasSharpen`, outputs
   the sharpened color. The EXISTING post path + its goldens stay BYTE-IDENTICAL (CAS is a NEW path behind the
   showcase flag; at sharpness 0 a pure pass-through). Bindings mirror the existing post passes (scene color
   t0/s0) + a small uniform for `sharpness`. No new RHI seam. HLSL→SPIR-V→MSL via the existing toolchain.

3. **Showcase `--cas-shot <out>` (Vulkan) / `--cas` (Metal).** The standard lit scene with CAS applied
   (visibly crisper edges/detail vs the unsharpened render), coherent (no ringing halos thanks to the clamp).
   Fixed camera, fixed sharpness. Print `cas: {sharpness:S}` (deterministic). INTERNALLY render with
   `sharpness = 0` and assert BYTE-IDENTICAL (SHA) to the unsharpened render — fail loudly on any diff. New
   golden `tests/golden/metal/cas.png` (Metal two runs DIFF 0.0000, gate on the compare.sh EXIT CODE).
   Existing 72 image goldens UNTOUCHED.

4. **Determinism.** Fixed scene/camera/sharpness. Two runs byte-identical.

5. **Tests `tests/cas_test.cpp` (pure CPU, no GPU):**
   - **Sharpness=0 identity:** `CasSharpen(c, ...,sharpness=0) == c` EXACTLY for any neighborhood;
     `CasWeight(...,0)` → 0 (no sharpening).
   - **Sharpening behavior:** a center brighter than its neighbors → CAS makes it relatively brighter
     (increases local contrast) for `sharpness > 0`; a flat neighborhood (all equal) → unchanged (no contrast
     to sharpen).
   - **No overshoot (the clamp):** the output is clamped to the neighborhood `[min, max]` — a sharpen never
     pushes a channel beyond the brightest/darkest neighbor (no ringing); verify on a high-contrast edge.
   - **Adaptive weight:** `CasWeight` is larger in low-contrast regions than near a hard edge (the adaptive
     property); monotone in `sharpness`.
   - **Determinism:** same inputs → same output.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `contrast-adaptive-sharpening` (features) + `--cas-shot` (showcases).

## RHI seam additions (summary)
- **None.** A fullscreen post pass reading the scene color + a small uniform (like the existing tonemap/
  vignette/color-grade passes). New files (`engine/render/cas.h`, `shaders/cas.frag.hlsl`,
  `tests/cas_test.cpp`) add ZERO backend code symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Upscaling CAS (RCAS / the FSR spatial-upscale path — sharpen-only here, no resolution change), separable/
multi-pass sharpen, per-region sharpening masks, edge-aware bilateral sharpen beyond the CAS clamp, sharpening
before tonemap (CAS runs on the final SDR result — document). One contrast-adaptive 3x3 sharpen pass with a
sharpness=0 byte-identical proof + an analytic unit test, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 72) + new `cas_test` (sharpness=0 identity, sharpening
   behavior, no-overshoot clamp, adaptive weight, determinism). Clean under `windows-msvc-asan`.
2. **Sharpness=0 no-op proof + visual:** `--cas-shot` on Vulkan: the scene is visibly crisper (sharper edges/
   detail) with no ringing halos, coherent; the INTERNAL sharpness=0 render is BYTE-IDENTICAL (SHA) to the
   unsharpened render; the `cas: {...}` line is deterministic (two runs → byte-identical capture). Run under
   the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --cas` → new golden `tests/golden/metal/cas.png`; two runs DIFF 0.0000 (gate on the
   compare.sh EXIT CODE).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `cas.png` added; the other
   72 byte-identical.
5. Introspect JSON rebaked exactly `+contrast-adaptive-sharpening` + `--cas-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `cas` image golden in
   the Mac round-trip loop. (`cas.frag.hlsl` is a NEW shader not shared by any existing golden — no re-bake of
   other Metal goldens; gate the new golden on the compare.sh EXIT CODE.)
