# Slice CK — Cloud Shadows on the Ground (Phase 4 #36) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A clean visual
> extension of CH (volumetric clouds): the cloudscape casts dappled shadows on the scene. Deterministic.

**Goal:** The volumetric clouds (CH) cast shadows on the ground/scene — each shaded point samples the
cloud density along the ray toward the sun and attenuates the direct sun contribution accordingly,
producing dappled cloud shadows that match the cloudscape overhead. A `--cloud-shadows-shot` showcase
shows the scene under cloud shadows; golden-verified on both backends. A NEW path (the default lit pass +
goldens are unchanged).

## Design decisions (locked)

1. **Cloud-shadow math (extend engine/render/clouds.h, pure CPU, no backend symbols).** Add to
   `hf::render::clouds`:
   - `float CloudShadow(const math::Vec3& worldPos, const math::Vec3& sunDir, float t, float slabBottom,
     float slabTop, int steps)` — march from `worldPos` toward the sun (`-sunDir` is the direction TO the
     sun; document the convention) through the cloud slab, accumulating optical depth from `Density`, and
     return the transmittance `Beer(opticalDepth)` ∈ `[0,1]` (1 = full sun, 0 = fully shadowed). Reuses
     `Density` + `Beer` (CH). Deterministic (fixed `t`, fixed steps, the CH deterministic noise). Shared
     with the shader + unit-tested.

2. **Apply in a NEW cloud-shadowed lit path (do NOT modify the default lit pass).** The
   `--cloud-shadows-shot` path multiplies the DIRECT sun (directional-light) contribution of the scene's
   surfaces by `CloudShadow(surfaceWorldPos, sunDir, t, ...)` — the ambient/IBL/point-light terms are
   unaffected (clouds only block the sun). Implement as either: (a) a small lit-shader VARIANT
   (`lit_cloudshadow.frag.hlsl`) that samples the cloud shadow and is used ONLY by `--cloud-shadows-shot`;
   or (b) a screen-space pass that, using the G-buffer world position + the known sun, computes the cloud
   shadow factor and modulates the already-lit scene's direct component (harder to separate direct from
   total — prefer the shader variant). Pick + document. The existing `lit*.frag` + their goldens stay
   BYTE-IDENTICAL. HLSL→SPIR-V→MSL via the existing toolchain. (The clouds themselves overhead are
   OPTIONAL in this shot — the focus is the ground shadows; you may render the CH cloudscape above too for
   a coherent look, or just the shadows — document.)

3. **Showcase `--cloud-shadows-shot <out>` (Vulkan) / `--cloud-shadows` (Metal).** The standard lit +
   shadowed scene (a large ground plane + objects) under cloud shadows: the ground shows moving... no —
   FIXED dappled cloud-shadow patterns (light/dark patches) from the same cloud field + time `t` as CH,
   the sun direction casting the cloud shadows across the ground. Fixed camera + FIXED time. Print
   `cloud-shadows: {steps:N, time:T}` (deterministic). New golden `tests/golden/metal/cloud_shadows.png`
   (Metal two runs DIFF 0.0000). Existing 56 image goldens UNTOUCHED.

4. **Determinism.** Fixed `t`, deterministic cloud noise (CH), fixed shadow-march steps, fixed
   camera/sun. Two runs byte-identical.

5. **Tests `tests/cloud_shadows_test.cpp` (pure CPU, no GPU) — or extend `clouds_test.cpp`:**
   - **Full sun (no cloud):** at a point whose sun ray misses the cloud slab (or in a clear region),
     `CloudShadow == 1` (no attenuation).
   - **Shadowed:** at a point whose sun ray passes through dense cloud, `CloudShadow < 1` (attenuated),
     and denser cloud → smaller value (more shadow); monotonic in optical depth.
   - **Range:** `CloudShadow ∈ [0,1]` always; `Beer`-consistent (matches `Beer(accumulatedOpticalDepth)`).
   - **Determinism:** same (worldPos, sunDir, t) → same shadow factor.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `cloud-shadows` (features) + `--cloud-shadows-shot` (showcases).

## RHI seam additions (summary)
- **None.** The cloud-shadow factor is sampled in a lit-shader variant (reuses FrameData sun + the cloud
  math); no new RHI. New files (`engine/render/clouds.h` additions, `shaders/lit_cloudshadow.frag.hlsl`,
  the test) add ZERO backend code symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Animated/wind cloud shadows (fixed-time only), self-shadowing of the clouds (CH already does in-cloud
extinction), cloud shadows on the volumetric fog / in the sky, a cloud-shadow MAP texture (sample the
density directly per-pixel — a precomputed shadow map is a perf optimization, future), colored/scattered
cloud-shadow tint, integrating cloud shadows into the DEFAULT lit pass (keep it a separate showcase to
preserve goldens). One fixed-time cloud-shadow attenuation of the sun in a new lit variant,
golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 56) + new `cloud_shadows_test` (full-sun,
   shadowed/monotonic, range, determinism). Clean under `windows-msvc-asan`.
2. `--cloud-shadows-shot` on Windows/Vulkan: controller visual review — the ground shows clear dappled
   cloud shadows (light/dark patches), the objects shadowed where clouds block the sun, coherent with the
   scene lighting; the `cloud-shadows: {...}` line is deterministic (two runs → byte-identical capture).
   Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --cloud-shadows` → new golden `tests/golden/metal/cloud_shadows.png`; two runs DIFF
   0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `cloud_shadows.png`
   added; the other 56 byte-identical (incl. `clouds.png` + the default lit goldens — this is a new path).
5. Introspect JSON rebaked exactly `+cloud-shadows` + `--cloud-shadows-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `cloud_shadows`
   image golden in the Mac round-trip loop.
