# Slice CR — GTAO (Ground-Truth Ambient Occlusion) (Phase 4 #41) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A recognized AAA quality
> upgrade over the engine's existing SSAO: horizon-based ground-truth AO with an analytic unit-tested
> visibility integral + a clean radius=0 equivalence proof. Reuses the G-buffer. Deterministic.

**Goal:** A GTAO (Jimenez et al. 2016, "Practical Realtime Strategies for Accurate Indirect Occlusion") pass
— per pixel, search the horizon angles of the surrounding depth field along several slice directions and
integrate the cosine-weighted visibility to produce physically-grounded ambient occlusion (darkening in
concavities/contacts). A `--gtao-shot` showcase shows a scene with clear contact/crease AO; golden-verified
on both backends. The existing SSAO path + its golden are UNTOUCHED (GTAO is a NEW, distinct, higher-quality
AO behind its own flag). The pass carries its proof: with `radius == 0` the horizon search finds no occluders
and the visibility integral evaluates to 1 everywhere → the GTAO-applied image is BYTE-IDENTICAL to the
no-AO scene — asserted internally, fail loudly on any diff.

## Why this is render-safe (the radius=0 / flat-field equivalence proof)

GTAO's AO factor is `1` when nothing occludes the hemisphere (both horizon angles reach ±90° = the full
cosine-weighted hemisphere integrates to 1). With `radius == 0` the search samples coincide with the center →
no horizon is raised → visibility = 1 for every pixel → multiplying the ambient term by 1 changes nothing →
the output equals the no-AO scene exactly. So the showcase INTERNALLY renders the scene with `radius = 0` and
asserts it is BYTE-IDENTICAL (SHA) to the no-AO render — proving the integration normalizes to a true
identity at zero occlusion (no constant bias, no off-by-one) — then renders the real `radius > 0` version as
the golden. (Same internal-assert discipline as CN/CO/CP.) The unit test additionally proves AO == 1 EXACTLY
on a flat depth field and matches an analytic value for a known occluder.

## Design decisions (locked)

1. **GTAO math (engine/render/gtao.h, header-only pure CPU, no backend symbols).** Namespace `hf::render::gtao`.
   Mirrors `ssr.h`/`dof.h`/`pom.h` (shared with the shader + the unit test).
   - `float IntegrateArc(float h1, float h2, float n)` — the GTAO inner integral for one slice: given the two
     horizon angles `h1,h2` (relative to the view dir) and the projected-normal angle `n`, return the
     cosine-weighted visibility for that slice using the published closed form
     `0.25 * ((-cos(2*h1 - n) + cos(n) + 2*h1*sin(n)) + (-cos(2*h2 - n) + cos(n) + 2*h2*sin(n)))`. Document.
     For an UNOCCLUDED slice (`h1 = -π/2 + n`, `h2 = π/2 + n` clamped to the hemisphere) this evaluates to 1.
   - `float HorizonAngle(...)` — given the center view position + a marched sample's view position, the
     horizon elevation angle of that sample (the per-step max-horizon update). Document the convention.
   - `float Visibility(<depth-sampler>, math::Vec3 viewPos, math::Vec3 viewNormal, float radius, int slices,
     int stepsPerSlice, int screenW, int screenH)` — the full GTAO estimate: for each of `slices` rotated
     screen directions, march `stepsPerSlice` samples each way within `radius`, track the max horizon each
     side, project the normal onto the slice plane, `IntegrateArc`, average over slices. Returns AO ∈ [0,1]
     (1 = unoccluded). At `radius == 0` → 1 exactly. Deterministic (fixed sample pattern, no RNG). Shared with
     the shader.

2. **GTAO shader `shaders/gtao.frag.hlsl` (NEW fullscreen pass).** Reuse `post.vert`. Reconstruct view
   position + normal from the G-buffer depth/normal (like SSAO/SSR at t3/s3); run the `gtao::Visibility`
   estimate (same math) with a baked per-pixel rotation (no RNG); output the AO factor (single channel). A
   resolve/apply step multiplies the scene's AMBIENT/IBL term by the AO (the direct sun is unaffected, like
   the existing SSAO apply). The EXISTING SSAO shader + its golden + the default lit path stay BYTE-IDENTICAL
   (GTAO is a NEW path behind the showcase flag). Bindings mirror SSAO (G-buffer t3/s3 via `BindTexturePair`)
   + a small uniform for `radius/slices/steps`. No new RHI seam. HLSL→SPIR-V→MSL via the existing toolchain.

3. **Showcase `--gtao-shot <out>` (Vulkan) / `--gtao` (Metal).** A scene with clear AO cues (objects resting
   on the ground, boxes forming concave corners/creases, the duck/spheres) so the horizon-based AO darkening
   in contacts + concavities is obvious. Fixed camera, fixed `radius/slices/steps`. Print
   `gtao: {slices:N, steps:M, radius:R}` (deterministic). INTERNALLY render the SAME scene with `radius = 0`
   and assert it is BYTE-IDENTICAL (SHA) to the no-AO render — fail loudly on any diff. New golden
   `tests/golden/metal/gtao.png` (Metal two runs DIFF 0.0000). Existing 61 image goldens UNTOUCHED (incl.
   the SSAO golden — GTAO is a separate path).

4. **Determinism.** Fixed camera/radius/slices/steps, fixed baked rotation pattern (no RNG/time). Two runs
   byte-identical.

5. **Tests `tests/gtao_test.cpp` (pure CPU, no GPU):**
   - **Unoccluded = 1:** `IntegrateArc` for the full unoccluded hemisphere → 1; `Visibility` over a FLAT depth
     field → 1 exactly (no occlusion); `radius == 0` → 1 exactly.
   - **Known occluder analytic match:** a single planar occluder at a known angle raises one horizon to a
     known value; `IntegrateArc` returns the analytic visibility for that geometry (hand-computed); a closer/
     taller occluder → smaller visibility (more AO); monotone.
   - **Range + symmetry:** `Visibility ∈ [0,1]` for all inputs; a symmetric occluder configuration yields the
     symmetric result; `IntegrateArc(h1,h2,n) == IntegrateArc(h2,h1,n)` (slice symmetry).
   - **Determinism:** same inputs → same AO.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `gtao` (features) + `--gtao-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses the SSAO/SSR G-buffer (depth + normal) + the fullscreen-pass + ambient-apply path. New
  files (`engine/render/gtao.h`, `shaders/gtao.frag.hlsl`, `tests/gtao_test.cpp`) add ZERO backend code
  symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Multi-bounce GTAO (the `1 - (1-ao)*albedo` re-bounce approximation — note it, future), bent-normals output
+ specular occlusion, temporal accumulation of the AO (a single-frame estimate; the existing SSAO temporal/
denoise infra is not reused here — keep it standalone), thickness/heuristic for thin occluders, half-res +
bilateral upsample, replacing SSAO (both coexist). One single-frame horizon-search GTAO pass with an analytic
unit test + a radius=0 byte-identical proof, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 61) + new `gtao_test` (unoccluded=1, flat=1, radius=0=1,
   known-occluder analytic match + monotone, range/symmetry, determinism). Clean under `windows-msvc-asan`.
2. **radius=0 equivalence proof + visual:** `--gtao-shot` on Vulkan: clear horizon-based AO darkening in the
   contacts/concavities (a recognizable ground-truth-AO look), coherent; the INTERNAL radius=0 render is
   BYTE-IDENTICAL (SHA) to the no-AO render; the `gtao: {...}` line is deterministic (two runs → byte-identical
   capture). Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --gtao` → new golden `tests/golden/metal/gtao.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `gtao.png` added; the
   other 61 byte-identical (incl. the SSAO golden).
5. Introspect JSON rebaked exactly `+gtao` + `--gtao-shot`; introspect test updated; no other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `gtao` image golden
   in the Mac round-trip loop.
