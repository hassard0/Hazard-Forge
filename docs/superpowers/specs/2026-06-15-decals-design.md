# Slice BH — Screen-Space Projected Decals (Phase 4 #10) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A distinct
> UE5-parity feature; reuses the existing G-buffer + screen-space-composite infrastructure (SSR/SSAO).

**Goal:** Project a decal (a texture, e.g. a crack/logo) onto scene geometry within an oriented box
volume. A screen-space pass reconstructs each pixel's world position from the G-buffer, transforms it
into the decal box's local space, and — if inside the unit box — samples the decal texture by the
projected coordinates and alpha-blends it over the scene color. A `--decal-shot` showcase shows a decal
on the ground; golden-verified on both backends.

## Why screen-space (reuse existing infra)

The engine already renders a G-buffer (`gbuffer.{vert,frag}.hlsl`) and runs screen-space composite passes
(SSR `ssr.frag.hlsl` + `ssr_composite`, SSAO) that reconstruct view/world position from it. A decal is the
same shape of pass: read the G-buffer position + scene color, project into decal space, blend. So this
slice adds a decal composite pass modeled exactly on the SSR/SSAO composite, with NO new RHI seam beyond
what those passes already use (G-buffer bind + a texture + a small uniform). The projection math lives in
a pure-CPU header shared with the shader (mirrors how `engine/render/ssr.h` is shared with the SSR pass +
`tests/ssr_test.cpp`).

## Design decisions (locked)

1. **Decal projection math (engine/render/decal.h, header-only, pure CPU, no backend symbols).**
   Namespace `hf::render::decal`. Mirrors the `ssr.h` style.
   - `math::Mat4 BuildDecalTransform(const math::Vec3& center, const math::Vec3& halfExtents,
     const math::Vec3& eulerRot)` — decal-box-local → world (TRS). Its inverse (world → decal-local) is
     uploaded to the shader.
   - `math::Vec3 WorldToDecalLocal(const math::Vec3& worldPos, const math::Mat4& worldToDecal)` —
     transform a world position into decal-local space (box is the unit cube `[-0.5,0.5]^3` in local).
   - `bool InsideUnitBox(const math::Vec3& local)` — `|x|<=0.5 && |y|<=0.5 && |z|<=0.5`.
   - `math::Vec2 DecalUV(const math::Vec3& local)` — the decal projects along local -Y (top-down onto the
     ground): `uv = local.xz + 0.5` (so `[-0.5,0.5]^2 → [0,1]^2`). Document the projection axis.
   - Optional `float EdgeFade(const math::Vec3& local, float fade)` — smooth alpha falloff near the box
     faces (so the decal edge isn't a hard cut); document the function. Keep it simple/deterministic.
   These are mutual-consistent with the shader; `tests/decal_test.cpp` pins them.

2. **Decal composite shader `shaders/decal.frag.hlsl`.** A fullscreen pass (reuse `post.vert`). Inputs:
   the scene color (the lit result so far), the G-buffer (the same world/view-position + depth the SSR
   pass reads — reconstruct world pos exactly as SSR/SSAO do, using the SAME `ReconstructViewPos`/yFlip
   convention; transform view→world with the camera, OR if the G-buffer stores world pos directly use
   that — match what SSR uses and document it), the decal texture + sampler, and a small uniform/push
   constant holding `worldToDecal` (mat4), the decal albedo tint, and the edge-fade. Per pixel: world pos
   → decal-local; if `InsideUnitBox`, sample the decal texture at `DecalUV`, compute alpha = decalTex.a *
   EdgeFade, `outColor = lerp(sceneColor, decalAlbedo*decalTex.rgb, alpha)`; else passthrough scene
   color. HLSL→SPIR-V→MSL via the existing toolchain (HF_MSL_GEN). Skybox/background pixels (no geometry
   in the G-buffer) must be excluded (check the depth/position validity like SSR does).

3. **Render path.** Reuse the G-buffer scene path the SSR/SSAO showcase uses. After the lit scene is
   resolved, run the decal composite pass (scene color + G-buffer + decal) → composited color → then the
   normal tonemap/post + capture. One decal for the MVP (a box on the ground). The decal texture: reuse an
   existing engine texture that reads clearly as a decal (e.g. the `normalmap`/`checker` as an albedo, or
   a small procedural cross/crack baked like the BA font — pick one that's visibly a decal and document
   it).

4. **Showcase `--decal-shot <out>` (Vulkan) / `--decal` (Metal).** The standard lit+shadowed scene
   (rendered through the G-buffer path) + ONE decal projected top-down onto the ground plane in a fixed
   box (fixed center/extents/orientation), composited, captured from a fixed camera that frames the decal
   clearly. Print `decal: {decals:1, box:[cx,cy,cz], ...}` (deterministic). New golden
   `tests/golden/metal/decal.png` (Metal two runs DIFF 0.0000). Existing 35 image goldens UNTOUCHED.

5. **Tests `tests/decal_test.cpp` (pure CPU, no GPU):**
   - `BuildDecalTransform` then `WorldToDecalLocal` round-trips: the box center maps to local origin; a
     world point at `center + R*(halfExtents)` maps to local `(0.5,0.5,0.5)` (corner).
   - `InsideUnitBox`: center→true, a point just outside a face→false, corner→true (boundary inclusive).
   - `DecalUV`: local `(-0.5,*,-0.5)`→`(0,0)`, `(0.5,*,0.5)`→`(1,1)`, origin→`(0.5,0.5)`.
   - `EdgeFade` (if included): 1 at center, →0 approaching a face; monotonic.
   - A rotated decal: a world point on the rotated box face maps inside; a point outside maps outside.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `decals` (features) + `--decal-shot` (showcases). One-pattern rebake.

## RHI seam additions (summary)
- **None beyond the existing G-buffer + composite-pass infra** (the decal pass binds the same G-buffer the
  SSR/SSAO composite binds, plus a texture + a small uniform). If a tiny additive pure-interface flag is
  truly needed, add it in `rhi.h` with backend impls inside the backend dirs. New files
  (`engine/render/decal.h`, `shaders/decal.frag.hlsl`, `tests/decal_test.cpp`) add ZERO backend code
  symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Multiple decals / a decal buffer + culling, deferred-decal G-buffer modification (normal/roughness
decals — this is an ALBEDO-only forward composite), decal sorting/priority, animated/scrolling decals,
mesh decals, decal angle-fade by surface normal (could add later), runtime decal placement UI. One static
albedo decal box, screen-space composite, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 34) + new `decal_test` (projection round-trip,
   inside-box, UV, edge-fade, rotated case). Clean under `windows-msvc-asan`.
2. `--decal-shot` on Windows/Vulkan: controller visual review — the decal is projected onto the ground
   within its box, blended over the scene, oriented correctly, not bleeding onto the sky/background;
   coherent. Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --decal` → new golden `tests/golden/metal/decal.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `decal.png` added; the
   other 35 byte-identical.
5. Introspect JSON rebaked exactly `+decals` + `--decal-shot`; introspect test updated; no other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `decal` image
   golden in the Mac round-trip loop (and OPTIONALLY fix the cosmetic hardcoded "30 goldens" summary
   string to the correct count while editing it — pure-ASCII).
