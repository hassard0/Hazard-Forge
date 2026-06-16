# Slice CX — Volumetric Shadows (sun light shafts through the fog) (Phase 4 #46) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. Completes the volumetric
> trilogy (CS sun froxel fog → CV clustered-light injection → CX VOLUMETRIC SHADOWS): the froxel inject
> samples the sun's CSM shadow map per froxel so the fog has shadowed gaps + visible sun light shafts (god
> rays) streaming through geometry — the iconic volumetric look. Reuses the existing CSM shadow map + the
> froxel inject. Deterministic, with a clean shadows-off no-op proof.

**Goal:** Per froxel, attenuate the sun in-scatter by the sun's shadow-map visibility at the froxel's world
position: a froxel occluded from the sun (behind geometry) gets NO sun scatter → a dark volume there → the lit
froxels between occluders read as bright shafts of foggy light. A `--volshadows-shot` showcase shows the
froxel fog with the sun casting volumetric shadows (light shafts through gaps between objects);
golden-verified on both backends. Built on CV (sun + clustered-light fog) — the volumetric shadow only
gates the SUN scatter (the clustered point-light scatter is unaffected this slice). The pass carries its
proof: with `volumetricShadows == false` (sun visibility forced to 1 everywhere) the fog is BYTE-IDENTICAL to
the CV `--froxellights-shot` render — asserted internally, fail loudly on any diff. (And density=0 == no-fog
still holds.)

## Why this is render-safe (the shadows-off no-op proof)

The sun in-scatter per froxel becomes `sunScatter * sunVisibility(froxelWorldPos)`. With `volumetricShadows ==
false`, `sunVisibility := 1` everywhere → `sunScatter * 1 = sunScatter` exactly → the whole inject→integrate→
apply chain produces the IDENTICAL volume + image as CV. So the showcase INTERNALLY renders with
`volumetricShadows = false` and asserts BYTE-IDENTICAL (SHA) to the CV `--froxellights-shot` (same scene/fog/
lights) — proving the shadow term is a clean multiplicative gate that is a true identity when off (no bias, no
shadow-UV drift) — then renders the real `volumetricShadows = true` version as the golden. Both proofs
(shadows-off==CV, density=0==no-fog) are internal asserts, fail loudly on any diff. (Same discipline as
CS/CV/CR/CT/CW.)

## Reuse map (builds on proven pieces)

- **CV/CS (`engine/render/froxel.h`, `shaders/froxel_inject.comp.hlsl`, integrate, apply)** — the froxel
  volume + sun/clustered-light inject + the `ComputeToCompute`/`ComputeToFragment` barriers. The inject gains
  the per-froxel sun-shadow sample.
- **The CSM shadow map (the lit pass already renders + samples it)** — the same depth shadow map + the
  cascade light-space view-proj matrices, now ALSO sampled in the froxel inject compute. Bind the shadow map
  (a sampled depth texture) + the cascade matrices to the inject compute via the existing texture/UBO bind
  paths (PREFER no new seam; if a compute-stage shadow-map bind is genuinely missing, an additive pure-
  interface in `rhi.h` with backend-dir impls — document + report).

## Design decisions (locked)

1. **Sun-visibility math (extend engine/render/froxel.h, pure CPU, no backend symbols).** Add to
   `hf::render::froxel`:
   - `float SunVisibility(const math::Vec3& worldPos, const math::Mat4& sunViewProj, <shadow-sampler>, float
     bias)` — project `worldPos` into the sun's light space, sample the shadow map, return 1 if the froxel is
     CLOSER to the sun than the stored occluder depth (lit) else 0 (shadowed); with a depth `bias` to avoid
     self-shadow acne. For a multi-cascade CSM, the caller selects the cascade by the froxel's view depth
     (document the cascade selection; the shader mirrors the lit pass's CSM cascade pick). Document the light-
     space + depth convention (matches the lit pass's shadow sampling EXACTLY so the volumetric shadow lines
     up with the surface shadows). At the degenerate (no shadow / forced visibility) → returns 1. Shared with
     the shader + unit-tested (the projection + the depth compare + the bias, against a procedural shadow
     field).

2. **Inject shader gains the sun-shadow gate (extend `shaders/froxel_inject.comp.hlsl`).** After computing the
   sun in-scatter (CS), if `volumetricShadows` is set: compute `froxel::SunVisibility(froxelWorldPos, the CSM
   cascade matrices, the shadow map, bias)` and MULTIPLY the sun in-scatter by it (the clustered-light
   in-scatter from CV is unaffected). PREFER a `volumetricShadows` uniform flag so `volumetricShadows=false`
   runs the EXACT CV code (→ the byte-identical shadows-off proof); document. Bind the CSM shadow map +
   cascade matrices to the inject compute (existing bind paths; report any seam addition). HLSL→SPIR-V→MSL via
   the existing toolchain.

3. **Showcase `--volshadows-shot <out>` (Vulkan) / `--volshadows` (Metal).** The CV fog scene (ground +
   objects + the 96 lights + fog) with the SUN now casting volumetric shadows: visible foggy light shafts
   between/around the objects where the sun reaches, darker fog in the objects' shadow volumes. Fixed camera/
   sun/lights/fog. Print `vol-shadows: {froxels:16x9x64, cascades:N, density:0.06}` (deterministic). INTERNALLY
   render (a) with `volumetricShadows=false` and assert BYTE-IDENTICAL (SHA) to the CV `--froxellights-shot`
   render of the SAME scene, AND (b) with `density=0` and assert BYTE-IDENTICAL to the no-fog scene — fail
   loudly on any diff. New golden `tests/golden/metal/vol_shadows.png` (Metal two runs DIFF 0.0000). Existing
   66 image goldens UNTOUCHED.

4. **Determinism.** Fixed sun/CSM/scene/fog, deterministic shadow sample (fixed bias, no RNG; a baked PCF
   pattern allowed — no RNG). Two runs byte-identical.

5. **Tests (extend `tests/froxel_test.cpp` OR new `tests/vol_shadows_test.cpp`, pure CPU, no GPU):**
   - **Lit froxel = visibility 1:** a froxel CLOSER to the sun than the shadow-map occluder → `SunVisibility ==
     1`; forced/degenerate → 1.
   - **Shadowed froxel = visibility 0:** a froxel BEHIND a closer occluder in the shadow map → `SunVisibility
     == 0` (within bias).
   - **Bias = no self-shadow:** a froxel AT the occluder depth (within bias) → 1 (no acne).
   - **Cascade selection:** a froxel at a known view depth selects the expected cascade; the light-space
     projection round-trips.
   - **Determinism:** same inputs → same visibility.
   - The existing CS/CV froxel cases still pass. Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `volumetric-shadows` (features) + `--volshadows-shot` (showcases).

## RHI seam additions (summary)
- **Prefer none.** The CSM shadow map is an existing sampled depth texture + the cascade matrices an existing
  UBO; bind them to the froxel inject COMPUTE via the existing texture/UBO bind paths (the same data the lit
  pass uses). IF a compute-stage shadow-map bind is genuinely missing, add an ADDITIVE pure-interface in
  `rhi.h` with impls INSIDE the backend dirs (document + report) — never a backend code symbol above the seam.
  New non-backend code (`froxel.h` additions, the inject-shader shadow gate, the test) adds ZERO above-seam
  backend code symbols. Seam grep stays at baseline (2). Report the seam result.

## Out of scope (YAGNI)
Volumetric shadows from the clustered POINT lights (sun-only this slice — point-light volumetric shadows are a
future slice), volumetric-shadow PCF softening beyond a basic sample, ray-marched/epipolar light shafts (this
is the froxel single-scatter approach), temporal/trilinear froxel upsampling (separate quality slice),
contact/screen-space volumetric shadows. One sun-shadow-gated froxel sun in-scatter with a shadows-off==CV
proof + a density=0==no-fog proof, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 66) + the new sun-visibility test cases (lit=1,
   shadowed=0, bias=no-acne, cascade selection, determinism). Clean under `windows-msvc-asan`.
2. **Two proofs + visual:** `--volshadows-shot` on Vulkan: visible foggy sun light shafts between the objects +
   darker shadow volumes in the fog, coherent; the INTERNAL `volumetricShadows=false` render is BYTE-IDENTICAL
   (SHA) to the CV `--froxellights-shot`, AND the `density=0` render is BYTE-IDENTICAL to the no-fog scene; the
   `vol-shadows: {...}` line is deterministic (two runs → byte-identical capture). Run under the AT Vulkan-
   validation gate → ZERO errors (the inject barriers + the shadow-map read stay SYNC-HAZARD-free).
3. Metal: `visual_test --volshadows` → new golden `tests/golden/metal/vol_shadows.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `vol_shadows.png` added;
   the other 66 byte-identical (incl. `froxel_fog.png`, `froxel_lights.png`, the CSM shadow goldens — the
   volumetricShadows=false path keeps `--froxellights-shot` + `--froxelfog-shot` byte-identical; verify).
5. Introspect JSON rebaked exactly `+volumetric-shadows` + `--volshadows-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (no new above-seam code symbols; report any rhi.h additive interface). `scripts/verify.ps1`
   updated to include the new `vol_shadows` image golden in the Mac round-trip loop.
