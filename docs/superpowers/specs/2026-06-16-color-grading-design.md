# Slice DB — Color Grading (lift/gamma/gain + ASC-CDL + saturation) (Phase 4 #49) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A genuinely-absent core
> cinematic feature: analytic color grading (lift/gamma/gain + ASC-CDL slope/offset/power + saturation)
> applied after tonemapping, for a graded "look". Fully analytic (no 3D-LUT texture, no new RHI seam).
> Deterministic, with a clean identity-grade no-op proof.

**Goal:** A color-grading post pass — after the tonemap, apply an analytic grade (lift/gamma/gain, the
ASC-CDL slope/offset/power, and a saturation control) to give the scene a stylized cinematic look (e.g. a
teal-shadows / warm-highlights grade). A `--colorgrade-shot` showcase shows the scene with a distinct graded
look vs the ungraded tonemap; golden-verified on both backends. The existing tonemap/post path + its goldens
are UNTOUCHED (grading is a NEW path behind the showcase flag). The pass carries its proof: with the IDENTITY
grade (lift 0, gamma 1, gain 1, slope 1, offset 0, power 1, saturation 1) the grade is a no-op → the graded
render is BYTE-IDENTICAL to the ungraded render — asserted internally, fail loudly on any diff.

## Why this is render-safe (the identity-grade no-op proof)

The grade is a per-pixel function `graded = Grade(color, params)` that, at the IDENTITY params, returns
`color` unchanged (lift+0, ×gain 1, ^(1/gamma) with gamma 1, slope 1, offset 0, power 1, saturation lerp at 1
= the original). So the showcase INTERNALLY renders with the identity grade and asserts BYTE-IDENTICAL (SHA)
to the engine's standard (ungraded) render of the same scene — proving the grade pass is a pure pass-through
at identity (no rounding bias, no clamp drift) — then renders the real stylized-grade version as the golden.
(Same internal-assert discipline as CN/CO/CP/CR/CS/CT/CW/CX/CZ/DA.) The unit test additionally proves each
grade term analytically (identity, and the documented monotone effect of each control).

## Design decisions (locked)

1. **Grade math (engine/render/color_grade.h, header-only pure CPU, no backend symbols).** Namespace
   `hf::render::grade`. Mirrors `dof.h`/`gtao.h`/`sss.h` (shared with the shader + the unit test).
   - `struct GradeParams { math::Vec3 lift; math::Vec3 gamma; math::Vec3 gain; math::Vec3 slope; math::Vec3
     offset; math::Vec3 power; float saturation; };` — the grade controls (per-channel where noted). Document
     the IDENTITY values (lift 0, gamma 1, gain 1, slope 1, offset 0, power 1, saturation 1).
   - `math::Vec3 LiftGammaGain(const math::Vec3& c, const math::Vec3& lift, const math::Vec3& gamma, const
     math::Vec3& gain)` — the standard `out = (gain * (c + lift*(1-c)))^(1/gamma)` (document the exact form;
     at identity → c). `math::Vec3 ASC_CDL(const math::Vec3& c, const math::Vec3& slope, const math::Vec3&
     offset, const math::Vec3& power)` — `out = (c*slope + offset)^power` (clamped ≥0; at identity → c).
     `math::Vec3 Saturate(const math::Vec3& c, float saturation)` — `lerp(luma(c), c, saturation)` with
     Rec.709 luma (at saturation 1 → c).
   - `math::Vec3 Grade(const math::Vec3& c, const GradeParams& p)` — compose LiftGammaGain → ASC_CDL →
     Saturate (document the order). At the identity params → returns `c` EXACTLY (document the exact-identity
     guarantee — the formulas must be algebraically exact at identity, not approximately). Shared with the
     shader.
   - Pure, deterministic, no RNG/time.

2. **Grade shader `shaders/color_grade.frag.hlsl` (NEW fullscreen pass).** Reuse `post.vert`. Reads the
   resolved tonemapped scene color, applies `grade::Grade(color, params)` (params from a small uniform), outputs
   the graded color. The EXISTING tonemap/post path + its goldens stay BYTE-IDENTICAL (grading is a NEW path
   behind the showcase flag; at identity it is a pure pass-through). Bindings mirror the existing post passes
   (scene color t0/s0) + a small uniform for the GradeParams (≤128B — it's a handful of vec3s + a float;
   pack carefully or use a UBO). No new RHI seam. HLSL→SPIR-V→MSL via the existing toolchain.

3. **Showcase `--colorgrade-shot <out>` (Vulkan) / `--colorgrade` (Metal).** The standard lit scene with a
   distinct CINEMATIC grade — e.g. teal-shadows / warm-highlights (lift toward teal, gain toward warm) +
   slightly lifted black + a saturation bump — clearly different from the ungraded tonemap but still coherent.
   Fixed camera, fixed grade params. Print `color-grade: {sat:S, gamma:G}` (or a compact deterministic
   summary of the params). INTERNALLY render with the IDENTITY grade and assert BYTE-IDENTICAL (SHA) to the
   ungraded render — fail loudly on any diff. New golden `tests/golden/metal/color_grade.png` (Metal two runs
   DIFF 0.0000, gate on the compare.sh EXIT CODE). Existing 69 image goldens UNTOUCHED.

4. **Determinism.** Fixed grade params + scene/camera. Two runs byte-identical.

5. **Tests `tests/color_grade_test.cpp` (pure CPU, no GPU):**
   - **Identity = no-op:** `Grade(c, identity) == c` EXACTLY for a set of colors (incl. 0, 1, mid-grey, a
     saturated color); each sub-op (`LiftGammaGain`, `ASC_CDL`, `Saturate`) at its identity → `c`.
   - **Monotone controls:** increasing `gain` brightens; `lift` raises the blacks (more effect on dark than
     bright); `gamma` > 1 brightens mids; `slope`/`offset`/`power` match the ASC-CDL definition (hand-checked
     values); `saturation` 0 → greyscale (== luma), saturation > 1 → more saturated.
   - **Range/robustness:** no NaN at c=0 with the power/gamma terms; output clamped ≥ 0 where documented.
   - **Determinism:** same inputs → same graded color.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `color-grading` (features) + `--colorgrade-shot` (showcases).

## RHI seam additions (summary)
- **None.** A fullscreen post pass reading the scene color + a small uniform (like the existing tonemap/post-
  stack passes); fully analytic (no 3D-LUT texture). New files (`engine/render/color_grade.h`,
  `shaders/color_grade.frag.hlsl`, `tests/color_grade_test.cpp`) add ZERO backend code symbols. Seam grep
  stays at baseline (2).

## Out of scope (YAGNI)
3D-LUT (.cube) grading (analytic controls only — a LUT needs a 3D texture; note future), per-region/secondary
grading (qualifiers/masks), curves/tone-curve editing, white-balance temperature/tint as a separate control
(fold into lift/gain), film-stock emulation, HDR-display grading, grading before tonemap (this grades the
tonemapped SDR result — document). One analytic lift/gamma/gain + ASC-CDL + saturation grade with an identity
byte-identical proof + an analytic unit test, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 69) + new `color_grade_test` (identity no-op per-op +
   composed, monotone controls, range/no-NaN, determinism). Clean under `windows-msvc-asan`.
2. **Identity no-op proof + visual:** `--colorgrade-shot` on Vulkan: the scene shows a distinct cinematic
   grade (e.g. teal-orange), coherent; the INTERNAL identity-grade render is BYTE-IDENTICAL (SHA) to the
   ungraded render; the `color-grade: {...}` line is deterministic (two runs → byte-identical capture). Run
   under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --colorgrade` → new golden `tests/golden/metal/color_grade.png`; two runs DIFF 0.0000
   (gate on the compare.sh EXIT CODE).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `color_grade.png` added;
   the other 69 byte-identical (incl. the tonemap goldens).
5. Introspect JSON rebaked exactly `+color-grading` + `--colorgrade-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `color_grade` image
   golden in the Mac round-trip loop. (`color_grade.frag.hlsl` is a NEW shader not shared by any existing
   golden — no re-bake of other Metal goldens needed.)
