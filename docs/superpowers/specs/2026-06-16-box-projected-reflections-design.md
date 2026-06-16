# Slice DA — Box-Projected Cubemap Reflections (local reflection probe) (Phase 4 #48) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A genuinely-absent AAA
> rendering feature: local reflection probes via BOX-PROJECTED cubemap sampling — the environment cubemap
> reflection is parallax-corrected to a local box (room) volume so reflections align with the geometry
> instead of appearing infinitely distant. Reuses the existing environment/reflection cubemap (no 6-face
> capture, no new cubemap RT). Deterministic, with a clean infinite-box no-op proof.

**Goal:** Box-projected cubemap reflections — for a reflective surface, the reflection direction is
intersected with a local box volume (the "reflection probe" bounds) centered on a probe origin, and the
cubemap is sampled along the corrected direction toward the box-hit point. This makes a reflected wall/floor
line up with the actual room geometry (the hallmark of a local reflection probe) instead of the
infinitely-distant look of a plain cubemap. A `--reflprobe-shot` showcase shows a reflective floor/sphere in a
box "room" with the reflection correctly aligned to the walls; golden-verified on both backends. The existing
IBL/reflection path + its goldens are UNTOUCHED (box-projection is a NEW path behind the showcase flag). The
pass carries its proof: with an INFINITE box (or `parallaxStrength == 0`) the box intersection returns the
original reflection direction unchanged → the box-projected reflection is BYTE-IDENTICAL to the standard
(infinite-cubemap) reflection — asserted internally, fail loudly on any diff.

## Why this is render-safe (the infinite-box / zero-parallax no-op proof)

Box projection replaces the reflection direction `R` with the direction from the probe center to where `R`
(from the shaded world point `P`) exits the box AABB. As the box grows to infinity (or `parallaxStrength →
0`), the exit point → infinitely far along `R`, so the corrected direction → `R` exactly → the cubemap is
sampled along the original `R` → identical to a plain (infinite) cubemap reflection. So the showcase
INTERNALLY renders with the box set to infinite (or `parallaxStrength = 0`) and asserts BYTE-IDENTICAL (SHA)
to the engine's standard cubemap-reflection render of the same scene — proving the box-projection is a pure
identity at the infinite limit (no constant bias, no direction drift) — then renders the real finite-box
version as the golden. (Same internal-assert discipline as CN/CO/CP/CR/CS/CT/CW/CX/CZ.) The unit test
additionally proves the ray-box intersection + the correction analytically.

## Design decisions (locked)

1. **Box-projection math (engine/render/reflection_probe.h, header-only pure CPU, no backend symbols).**
   Namespace `hf::render::reflprobe`. Mirrors `ssr.h`/`gtao.h`/`sss.h` (shared with the shader + the unit test).
   - `struct ProbeBox { math::Vec3 center; math::Vec3 boxMin; math::Vec3 boxMax; };` — the reflection probe's
     origin + world-space box bounds (the room volume the cubemap represents).
   - `math::Vec3 BoxProject(const math::Vec3& reflDir, const math::Vec3& worldPos, const ProbeBox& box,
     float parallaxStrength)` — intersect the ray `worldPos + t*reflDir` (t>0) with the box AABB to get the
     exit point `hit`; the corrected sample direction is `normalize(lerp(reflDir, hit - box.center,
     parallaxStrength))`. With `parallaxStrength == 0` → returns `normalize(reflDir)` exactly (document); with
     an effectively-infinite box the `hit` is far along `reflDir` so `hit - center ≈ reflDir`. Document the
     slab-method ray-box intersection (positive-t exit) + the degenerate guards (ray parallel to a slab,
     origin outside the box → clamp/keep `reflDir`). Pure, deterministic.
   - `float RayBoxExitT(const math::Vec3& origin, const math::Vec3& dir, const math::Vec3& boxMin, const
     math::Vec3& boxMax)` — the slab-method exit-t helper (the smallest positive t at which the ray leaves the
     box); used by `BoxProject` + unit-tested directly.
   - Document the convention. No RNG/time.

2. **Reflection shader `shaders/reflprobe.frag.hlsl` (NEW reflective-surface variant).** For the reflective
   surface: compute the world reflection vector `R = reflect(-viewDir, N)`, box-project it via
   `reflprobe::BoxProject(R, worldPos, probeBox, parallaxStrength)`, sample the EXISTING environment/reflection
   cubemap (the `BindReflectionProbe` env-slot the engine already binds for IBL — reuse it; if only a 2D env
   exists, sample it via the documented mapping) along the corrected direction, and blend the reflection into
   the surface by its reflectivity. With `parallaxStrength=0` the corrected direction == `R` → identical to the
   plain cubemap reflection. The EXISTING IBL/lit path + its goldens stay BYTE-IDENTICAL (this is a NEW path
   behind the showcase flag). Reuse the existing env-cubemap binding (`BindReflectionProbe`) + a small uniform
   for the probe box + `parallaxStrength`. PREFER no new RHI seam (reuse the env-cubemap sampling already used
   by IBL); if a sharp specular env sample genuinely doesn't exist, add the minimal sampling in this shader
   only (no backend code above the seam). HLSL→SPIR-V→MSL via the existing toolchain.

3. **Showcase `--reflprobe-shot <out>` (Vulkan) / `--reflprobe` (Metal).** A reflective floor (and/or a
   glossy sphere) inside a box "room" (the probe box) with distinct colored walls in the environment cubemap,
   so the box-projected reflection visibly aligns the reflected walls/floor to the room geometry (vs the
   infinite-cubemap smear). Fixed camera, fixed probe box, fixed env. Print
   `refl-probe: {boxSize:S, parallax:1}` (deterministic). INTERNALLY render with `parallaxStrength = 0` (or
   infinite box) and assert BYTE-IDENTICAL (SHA) to the standard infinite-cubemap reflection render — fail
   loudly on any diff. New golden `tests/golden/metal/refl_probe.png` (Metal two runs DIFF 0.0000, gate on the
   compare.sh EXIT CODE). Existing 68 image goldens UNTOUCHED.

4. **Determinism.** Fixed probe box / env / camera / reflectivity. Two runs byte-identical.

5. **Tests `tests/reflection_probe_test.cpp` (pure CPU, no GPU):**
   - **Ray-box exit:** `RayBoxExitT` for known origin/dir/box → the analytic exit t (a ray from the center
     toward +X exits at the +X face; an axis-aligned + a diagonal case); positive t.
   - **Zero parallax = identity:** `BoxProject(R, P, box, parallaxStrength=0) == normalize(R)` for any inputs.
   - **Infinite box ≈ identity:** with a huge box, `BoxProject` returns ≈ `normalize(R)` (the corrected dir
     converges to R as the box grows); quantify the convergence.
   - **Finite-box correction:** for a finite box, the corrected direction points toward the box-exit relative
     to the probe center (differs from R when P ≠ center); a known geometry → the hand-computed corrected dir.
   - **Determinism:** same inputs → same direction.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `reflection-probe` (features) + `--reflprobe-shot` (showcases).

## RHI seam additions (summary)
- **Prefer none.** Reuses the existing environment/reflection cubemap binding (`BindReflectionProbe`, already
  used for IBL) + the lit/fullscreen path. The box-projection is pure shader math + a uniform. New files
  (`engine/render/reflection_probe.h`, `shaders/reflprobe.frag.hlsl`, `tests/reflection_probe_test.cpp`) add
  ZERO above-seam backend code symbols. IF a sharp specular env-cubemap sample is genuinely missing and needs
  a backend addition, add it as an ADDITIVE pure-interface in `rhi.h` with backend-dir impls (document +
  report) — but PREFER reusing the existing env sampling. Seam grep stays at baseline (2). Report the seam
  result.

## Out of scope (YAGNI)
Runtime 6-face cubemap CAPTURE of the scene (this reuses the EXISTING env cubemap + adds box-projection;
dynamic scene-capture probes are a future slice), roughness prefiltering / mip convolution of the probe
(sharp/single-mip reflection only — note future), multiple blended probes / probe selection, parallax for
diffuse IBL (specular reflection only), probe-relighting, planar reflections. One box-projected cubemap
reflection with an infinite-box byte-identical proof + an analytic ray-box unit test, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 68) + new `reflection_probe_test` (ray-box exit,
   zero-parallax identity, infinite-box convergence, finite-box correction, determinism). Clean under
   `windows-msvc-asan`.
2. **Infinite-box no-op proof + visual:** `--reflprobe-shot` on Vulkan: the reflective surface shows the
   box-aligned reflection (reflected walls line up with the room), coherent; the INTERNAL parallaxStrength=0
   render is BYTE-IDENTICAL (SHA) to the standard infinite-cubemap reflection; the `refl-probe: {...}` line is
   deterministic (two runs → byte-identical capture). Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --reflprobe` → new golden `tests/golden/metal/refl_probe.png`; two runs DIFF 0.0000
   (gate on the compare.sh EXIT CODE, not the printed rounded DIFF).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `refl_probe.png` added; the
   other 68 byte-identical (incl. any existing IBL/reflection golden).
5. Introspect JSON rebaked exactly `+reflection-probe` + `--reflprobe-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (no new above-seam code symbols; report any rhi.h additive interface). `scripts/verify.ps1`
   updated to include the new `refl_probe` image golden in the Mac round-trip loop. (`reflprobe.frag.hlsl` is a
   NEW shader not shared by any existing golden — no re-bake of other Metal goldens needed.)
