# Slice DE — Planar Reflections (mirror-plane scene reflection) (Phase 4 #51) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A genuinely-absent AAA
> feature distinct from SSR (screen-space) and the cubemap probes (DA/DD): a flat MIRROR plane reflects the
> scene by rendering it through a camera mirrored across the plane into a reflection texture. Reuses DD's
> scene-render-to-RT path. Deterministic, with a clean reflection-off no-op proof.

**Goal:** Planar reflections — for a flat reflective surface (a mirror floor / still water plane), render the
scene through a camera REFLECTED across the mirror plane (with an oblique near-plane clip so geometry behind
the mirror is excluded) into a reflection render target, then the mirror surface samples that texture
projected by its screen position. A `--planar-shot` showcase shows a mirror floor reflecting the objects
standing on it; golden-verified on both backends. The existing lit/floor path + its goldens are UNTOUCHED
(planar reflection is a NEW path behind the showcase flag). The pass carries its proof: with the reflection
contribution disabled (`reflectivity == 0`) the surface is BYTE-IDENTICAL to the matte (non-reflective)
render — asserted internally, fail loudly on any diff.

## Why this is render-safe (the reflectivity=0 no-op proof)

The mirror surface blends `lerp(matteColor, reflectionSample, reflectivity)`. With `reflectivity == 0` →
`lerp(...,0) == matteColor` exactly → the planar-reflection render equals the matte render (the reflection
texture is never read). So the showcase INTERNALLY renders with `reflectivity = 0` and asserts BYTE-IDENTICAL
(SHA) to the engine's standard matte-surface render of the same scene — proving the reflection pass is a pure
pass-through when off (no blend bias, no projection drift) — then renders the real `reflectivity > 0` version
as the golden. (Same internal-assert discipline as the other slices.) The unit test additionally proves the
reflection matrix is correct (reflects a point across the plane) + INVOLUTORY (`R·R == I`) + the oblique clip.

## Reuse map (builds on proven pieces)

- **DD (`engine/rhi` cubemap/RT path + `ReadRenderTarget`)** — the scene-render-into-an-RT-with-a-custom-
  camera path (DD does 6 of these into cube faces; planar does ONE into a 2D RT with the reflected camera).
- **The scene render path** — the reflection pass renders the SAME scene (objects + lighting) the main pass
  renders, through the mirrored camera. Reuse the lit pass.

## Design decisions (locked)

1. **Mirror math (engine/render/planar_reflection.h, header-only pure CPU, no backend symbols).** Namespace
   `hf::render::planar`. Mirrors `reflection_probe.h`/`cubemap.h`/`gtao.h` (shared with the render code + the
   unit test).
   - `math::Mat4 ReflectionMatrix(const math::Vec3& planeNormal, float planeD)` — the householder reflection
     matrix across the plane `dot(N, p) + planeD = 0` (the standard `I - 2*outer(N,N)` extended with the
     `-2*planeD*N` translation). Reflects a point to its mirror image; INVOLUTORY (`R·R == I`). Document the
     plane convention.
   - `math::Mat4 ObliqueNearClip(const math::Mat4& proj, const math::Vec4& clipPlaneView)` — modify the
     projection so its near plane == the mirror plane (in view space), clipping geometry behind the mirror
     (the standard Lengyel oblique-frustum technique). Document; at a degenerate (plane at infinity) → `proj`
     unchanged.
   - `math::Vec4 PlaneToView(const math::Vec3& planeNormal, float planeD, const math::Mat4& reflectedView)` —
     transform the world mirror plane into the reflected camera's view space (for the oblique clip). Document.
   - Pure, deterministic. The reflected camera = `ReflectionMatrix * mainView` (+ a winding flip handled in
     the render). Document the full pipeline.

2. **Reflection render (reuse DD's scene-render-into-RT + a 2D reflection RT).** Render the scene through the
   reflected camera (`ReflectionMatrix(plane) * mainViewProj`, with the oblique near-clip at the mirror plane
   + front-face winding flipped because reflection mirrors handedness) into a 2D color reflection RT. Then the
   mirror surface's shader samples that RT at its own screen-space position (the reflection is projected back
   via the mirror pixel's screen UV) and blends by `reflectivity`. REUSE the existing RT-create +
   scene-render path (DD's infra); the reflection RT is a standard 2D color target (no new RHI seam expected —
   if the winding-flip / a 2D capture RT needs a tiny additive interface, add it pure-interface with backend-
   dir impls + report). EXISTING lit/floor path + goldens stay BYTE-IDENTICAL. HLSL→SPIR-V→MSL via the
   existing toolchain.

3. **Showcase `--planar-shot <out>` (Vulkan) / `--planar` (Metal).** A mirror floor (the reflective plane)
   with distinct colored objects standing on it; the floor reflects the objects (upside-down) coherently.
   Fixed camera, fixed plane, fixed reflectivity. Print `planar: {reflectivity:R}` (deterministic). INTERNALLY
   render with `reflectivity = 0` and assert BYTE-IDENTICAL (SHA) to the matte-floor render — fail loudly on
   any diff. New golden `tests/golden/metal/planar_reflection.png` (Metal two runs DIFF 0.0000, gate on the
   compare.sh EXIT CODE). Existing 71 image goldens UNTOUCHED.

4. **Determinism.** Fixed scene/plane/camera/reflectivity. Two runs byte-identical.

5. **Tests `tests/planar_reflection_test.cpp` (pure CPU, no GPU):**
   - **Reflection matrix:** `ReflectionMatrix(N, d)` reflects a known point to its mirror image across the
     plane (a point at height +h above a y=0 plane → -h); a point ON the plane is unchanged.
   - **Involutory:** `R·R == I` (reflecting twice returns the original) for several planes.
   - **Oblique clip:** `ObliqueNearClip` sets the near plane to the mirror plane (a point just behind the
     mirror is clipped, a point just in front is kept — check the clip-space w/z sign); degenerate → proj
     unchanged.
   - **Plane-to-view:** `PlaneToView` transforms a world plane into the reflected view space correctly
     (hand-checked for an axis-aligned plane).
   - **Determinism:** same inputs → same matrices.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `planar-reflections` (features) + `--planar-shot` (showcases).

## RHI seam additions (summary)
- **Prefer none.** Reuses DD's scene-render-into-an-RT path + a standard 2D color reflection RT + the lit/
  fullscreen path. The reflected camera + oblique clip + winding flip are pure math + existing pipeline state
  (a front-face-winding toggle, if not already exposed, is an ADDITIVE pure-interface field in the pipeline
  desc with backend-dir impls — document + report; PREFER an existing winding/cull knob). New non-backend
  files (`engine/render/planar_reflection.h`, the mirror shader, `tests/planar_reflection_test.cpp`) add ZERO
  above-seam backend code symbols. Seam grep stays at baseline (2). Report the seam result.

## Out of scope (YAGNI)
Curved/arbitrary reflectors (planar = a single flat plane; curved → cubemap probes [DA/DD] or SSR), rough/
glossy planar reflections (sharp mirror only — blur is a future slice), multiple mirror planes, recursive
reflections (the reflected scene excludes the mirror itself; no mirror-in-mirror), reflection of the
transparent/water passes, stencil-masked planar reflections. One flat mirror-plane scene reflection with a
reflectivity=0 no-op proof + an involutory-matrix unit test, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 71) + new `planar_reflection_test` (reflection matrix,
   involutory, oblique clip, plane-to-view, determinism). Clean under `windows-msvc-asan`.
2. **Reflectivity=0 no-op proof + visual:** `--planar-shot` on Vulkan: the mirror floor reflects the objects
   (a recognizable upside-down reflection), coherent; the INTERNAL reflectivity=0 render is BYTE-IDENTICAL
   (SHA) to the matte-floor render; the `planar: {...}` line is deterministic (two runs → byte-identical
   capture). Run under the AT Vulkan-validation gate → ZERO errors (the reflection render pass + the
   reflection-write→sample barrier must be SYNC-HAZARD-free).
3. Metal: `visual_test --planar` → new golden `tests/golden/metal/planar_reflection.png`; two runs DIFF 0.0000
   (gate on the compare.sh EXIT CODE).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `planar_reflection.png`
   added; the other 71 byte-identical. (If a winding-flip RHI field was added, verify a broad set of existing
   proofs + goldens stay byte-identical = additive field inert.)
5. Introspect JSON rebaked exactly `+planar-reflections` + `--planar-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (no new above-seam code symbols; report any rhi.h additive field). `scripts/verify.ps1`
   updated to include the new `planar_reflection` image golden in the Mac round-trip loop. (The mirror shader
   + planar path are NEW — no re-bake of other Metal goldens; gate the new golden on the compare.sh EXIT CODE.)
