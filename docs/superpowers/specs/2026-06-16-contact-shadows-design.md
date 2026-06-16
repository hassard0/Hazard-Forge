# Slice CT — Screen-Space Contact Shadows (Phase 4 #43) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A genuinely-absent AAA
> light feature (UE5 "Contact Shadows"): a short screen-space depth ray-march toward the sun that adds
> fine-scale contact occlusion the CSM shadow map is too coarse to capture. Reuses the G-buffer depth + the
> sun. Deterministic, with a clean maxDist=0 no-op proof.

**Goal:** Screen-space contact shadows — per pixel, march a short ray through the depth buffer toward the sun;
if it passes behind closer geometry (within a thickness window), the pixel is in a fine contact shadow that
darkens the DIRECT sun contribution. A `--contactshadow-shot` showcase shows tight contact shadowing in
small gaps/creases where the shadow map misses it; golden-verified on both backends. The existing shadow-map
(CSM) path + its golden are UNTOUCHED (contact shadows are a NEW additive term behind the showcase flag). The
pass carries its proof: with `maxDist == 0` (or `steps == 0`) the march takes no steps → no occluder is found
→ the contact-shadow factor is 1 everywhere → the image is BYTE-IDENTICAL to the no-contact-shadow render —
asserted internally, fail loudly on any diff.

## Why this is render-safe (the maxDist=0 / zero-steps no-op proof)

The contact-shadow factor multiplies the direct sun term. With `maxDist == 0` the screen-space ray never
leaves the origin pixel → no depth sample is ever in front of the ray → the factor is 1 for every pixel →
multiplying the sun term by 1 changes nothing → the output equals the no-contact-shadow render exactly. So the
showcase INTERNALLY renders with `maxDist = 0` and asserts it is BYTE-IDENTICAL (SHA) to the no-contact-shadow
render — proving the march + apply is a pure pass-through when disabled (no constant bias, no self-shadow
acne leaking through, no off-by-one) — then renders the real `maxDist > 0` version as the golden. (Same
internal-assert discipline as CN/CO/CP/CR/CS.) The unit test additionally proves: a clear ray → factor 1
exactly; a ray that passes behind a known occluder within the thickness window → factor < 1 (occluded); an
occluder BEYOND the thickness window → factor 1 (no false shadow / no haloing).

## Design decisions (locked)

1. **Contact-shadow math (engine/render/contact_shadows.h, header-only pure CPU, no backend symbols).**
   Namespace `hf::render::contact`. Mirrors `ssr.h`/`gtao.h`/`froxel.h` (shared with the shader + the unit
   test).
   - `float RayMarchShadow(<depth-sampler>, math::Vec3 viewPos, math::Vec3 lightDirView, float maxDist,
     int steps, float thickness, float bias, const math::Mat4& proj, int screenW, int screenH)` — march from
     `viewPos` toward the light (`lightDirView` = the direction TO the sun) in `steps` increments up to
     `maxDist` in view space; at each step project the marched view point to screen, sample the scene depth
     there, and compare: if the scene surface is CLOSER than the marched ray by more than `bias` AND the
     difference is within `thickness` (so a distant background does NOT cast a false contact shadow), the ray
     is occluded → return a shadow factor < 1 (hard 0, or a soft falloff by march distance — document). A
     clear march → 1. `maxDist == 0` or `steps == 0` → 1 exactly. Document the bias (avoid self-occlusion
     acne) + the thickness window + the convention. Pure, deterministic, no RNG/time.

2. **Contact-shadow shader `shaders/contact_shadows.frag.hlsl` (NEW fullscreen pass).** Reuse `post.vert`.
   Reconstruct the view position from the G-buffer depth (like SSR/SSAO/GTAO at t3/s3); run
   `contact::RayMarchShadow` toward the FrameData sun (a baked per-pixel start offset/dither is allowed — no
   RNG); output the contact-shadow factor (single channel). The apply MULTIPLIES the scene's DIRECT sun
   contribution by the factor (the ambient/IBL term is unaffected — contact shadows only occlude the sun, like
   the CSM shadow term). The EXISTING CSM shadow shader + its golden + the default lit path stay
   BYTE-IDENTICAL (contact shadows are a NEW path behind the showcase flag). Bindings mirror SSAO/GTAO
   (G-buffer t3/s3 via `BindTexturePair`) + a small uniform for `maxDist/steps/thickness/bias`. No new RHI
   seam. HLSL→SPIR-V→MSL via the existing toolchain.

3. **Showcase `--contactshadow-shot <out>` (Vulkan) / `--contactshadow` (Metal).** A scene with small objects
   resting on / near surfaces (spheres + small boxes in tight contact, a thin object near the ground) where
   the broad CSM shadow leaves a gap that screen-space contact shadows fill — tight dark contact lines at the
   bases/creases. Fixed camera, fixed sun, fixed march params. Print
   `contact-shadows: {steps:N, maxDist:D, thickness:T}` (deterministic). INTERNALLY render with `maxDist = 0`
   and assert BYTE-IDENTICAL (SHA) to the no-contact-shadow render — fail loudly on any diff. New golden
   `tests/golden/metal/contact_shadows.png` (Metal two runs DIFF 0.0000). Existing 63 image goldens UNTOUCHED.

4. **Determinism.** Fixed camera/sun/march params, baked dither (no RNG/time). Two runs byte-identical.

5. **Tests `tests/contact_shadows_test.cpp` (pure CPU, no GPU):**
   - **Disabled = identity:** `RayMarchShadow(..., maxDist=0, ...) == 1` and `steps=0 → 1` for any input.
   - **Clear ray = lit:** a flat/empty depth field along the ray → factor 1 (no occluder).
   - **Occluder within thickness = shadowed:** a depth sample placed in front of the ray within `thickness` →
     factor < 1; a CLOSER occluder → ≤ the farther one (monotone / at least occluded); deterministic value.
   - **Occluder beyond thickness = no false shadow:** an occluder in front of the ray but FARTHER than
     `thickness` behind → factor 1 (no haloing — the safety case).
   - **Bias:** the surface's own depth at the origin does NOT self-occlude (factor 1 on a flat lit surface).
   - **Determinism:** same inputs → same factor.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `contact-shadows` (features) + `--contactshadow-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses the SSAO/SSR/GTAO G-buffer (depth) + the fullscreen-pass + direct-sun-apply path. New
  files (`engine/render/contact_shadows.h`, `shaders/contact_shadows.frag.hlsl`,
  `tests/contact_shadows_test.cpp`) add ZERO backend code symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Contact shadows for point/spot lights (sun/directional only — the math generalizes later), temporal
accumulation of the contact term, denoising/blur of the contact shadow, per-light contact-shadow length,
thickness-from-a-thickness-buffer, replacing the CSM (contact shadows AUGMENT the shadow map, both coexist),
ray-traced contact shadows. One single-frame screen-space depth-march contact-shadow pass with a maxDist=0
no-op proof + an analytic unit test, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 63) + new `contact_shadows_test` (disabled=1, clear=1,
   occluder-within-thickness<1 + monotone, occluder-beyond-thickness=1 safety, bias no-self-occlude,
   determinism). Clean under `windows-msvc-asan`.
2. **maxDist=0 no-op proof + visual:** `--contactshadow-shot` on Vulkan: tight contact shadowing in the small
   gaps/creases (a recognizable contact-shadow look the broad CSM misses), coherent; the INTERNAL maxDist=0
   render is BYTE-IDENTICAL (SHA) to the no-contact-shadow render; the `contact-shadows: {...}` line is
   deterministic (two runs → byte-identical capture). Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --contactshadow` → new golden `tests/golden/metal/contact_shadows.png`; two runs DIFF
   0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `contact_shadows.png`
   added; the other 63 byte-identical (incl. the CSM shadow goldens).
5. Introspect JSON rebaked exactly `+contact-shadows` + `--contactshadow-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `contact_shadows`
   image golden in the Mac round-trip loop.
