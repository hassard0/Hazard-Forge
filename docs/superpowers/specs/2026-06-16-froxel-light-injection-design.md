# Slice CV — Per-Froxel Clustered-Light Injection (Phase 4 #44) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The marquee payoff fusing
> CS (froxel volumetric fog) + CL (clustered Forward+ lighting): the clustered point lights now scatter
> through the fog, casting colored volumetric light shafts/glow — the UE5/Frostbite volumetric-fog endgame.
> Deterministic at a FIXED light set + fog params, with TWO byte-identical proofs.

**Goal:** Extend the froxel-fog inject pass (CS) to ALSO accumulate in-scattered light from the clustered
point lights (CL): for each froxel, find its cluster, iterate that cluster's assigned lights, and add each
light's `color * windowedAttenuation(dist) * phase * fogDensity` to the froxel's scattering. The integrated
volume then carries colored point-light glow, so the `--froxellights-shot` showcase shows the 96-light
clustered scene wrapped in fog with each colored light casting a volumetric shaft. Golden-verified on both
backends. The pass carries TWO proofs: (a) with the point-light injection DISABLED the fog is BYTE-IDENTICAL
to the CS sun-only froxel fog (the point-light term is purely additive); (b) with fog density 0 the result is
BYTE-IDENTICAL to the no-fog scene (the CS no-op, still holds).

## Why this is render-safe (the two equivalence proofs)

1. **Lights-off == CS sun-only fog.** The point-light in-scatter is ADDED to the sun in-scatter in the inject
   pass: `scatter = sunScatter + Σ lightScatter`. With the point-light loop disabled (a `injectLights` flag
   = false, or all lights' intensity 0), `Σ lightScatter = 0` → `scatter = sunScatter` exactly → the whole
   inject→integrate→apply chain produces the IDENTICAL volume + image as CS. The showcase INTERNALLY renders
   with `injectLights = false` and asserts BYTE-IDENTICAL (SHA) to the CS sun-only froxel fog of the same
   scene — proving the light injection is a clean additive term (no perturbation of the sun path).
2. **Density=0 == no-fog.** Unchanged from CS: density 0 → extinction 0 → transmittance 1, and ALL scattering
   (sun + lights) is `× density = 0` → in-scatter 0 → out = scene exactly. The showcase also asserts this.

Both are internal asserts, fail loudly on any diff (same discipline as CS/CL/CR/CT).

## Reuse map (builds on proven pieces)

- **CL (`engine/render/cluster.h`, `shaders/cluster_assign.comp.hlsl`)** — the cluster grid + `AssignLights`
  CPU reference + the per-cluster light index SSBO + the `PointLight` struct + `SphereAABBIntersect` +
  windowed attenuation. The froxel inject reads the SAME cluster light grid the clustered-lights pass builds.
- **CS (`engine/render/froxel.h`, `shaders/froxel_inject.comp.hlsl`, `froxel_integrate.comp.hlsl`,
  `froxel_apply.frag.hlsl`)** — the froxel grid + density + phase + `IntegrateStep` + the SSBO volume + the
  `ComputeToCompute`/`ComputeToFragment` barriers. The inject pass gains the light loop.
- **The froxel→cluster Z map** — both use exponential Z slicing; map a froxel's view-Z to a cluster slice via
  `cluster::ViewZToSlice` (the froxel and cluster XY tiles align — both 16×9; only Z resolution differs:
  froxel dimZ=64, cluster dimZ=24).

## Design decisions (locked)

1. **Light-injection math (extend engine/render/froxel.h, pure CPU, no backend symbols).** Add to
   `hf::render::froxel`:
   - `math::Vec3 InjectClusteredLights(const math::Vec3& froxelViewPos, float density, const math::Vec3&
     viewDir, float g, span<const cluster::PointLight> lights, span<const uint32_t> clusterLightIndices,
     uint32_t lightOffset, uint32_t lightCount, const math::Mat4& view)` — for each of the cluster's
     `lightCount` lights (indexed via `clusterLightIndices[lightOffset + i]`), compute the light's view-space
     position (`view * light.posWorld`), the distance to the froxel, the WINDOWED attenuation (IDENTICAL to
     CL's `atten=(1/(d²+ε))*clamp(1-(d/radius)⁴,0,1)²` — EXACTLY 0 beyond radius), the HG `Phase(cos(viewDir,
     dirToLight), g)`, and accumulate `light.color * light.intensity * atten * phase * density`. Returns the
     added scatter RGB. With `lightCount == 0` → returns 0. Shared with the shader + unit-tested.
   - Document that this is ADDED to the existing sun-scatter in the inject pass (purely additive → the
     lights-off proof).

2. **Inject shader gains the light loop (`shaders/froxel_inject.comp.hlsl`, extend the CS shader OR a new
   `froxel_inject_lights.comp.hlsl`).** After computing the sun in-scatter + extinction (CS), if `injectLights`
   is set: map the froxel center to its cluster (XY tile from the froxel XY; Z via `cluster::ViewZToSlice` of
   the froxel view-Z), read `clusterLightGrid[clusterIdx] = (offset, count)`, iterate the cluster's lights
   (same SSBO the clustered-lights pass fills), and ADD `InjectClusteredLights(...)` to the froxel's scatter.
   The cluster light grid + light SSBO bind via the existing `BindLightClusters` storage path (the SAME path
   CL uses — no new RHI seam). Prefer EXTENDING `froxel_inject.comp.hlsl` with an `injectLights` uniform flag
   so the `injectLights=false` path is the exact CS code (→ the byte-identical lights-off proof); document.
   HLSL→SPIR-V→MSL via the existing toolchain.

3. **Showcase `--froxellights-shot <out>` (Vulkan) / `--froxellights` (Metal).** The CL 96-colored-point-light
   scene (ground + objects + the 96 lights, same fixed lattice as CL) wrapped in the CS froxel fog: each
   colored light casts a volumetric glow/shaft through the fog; the lit scene shows underneath. Pipeline:
   cluster-assign (CL) → froxel inject (sun + clustered lights) → integrate → apply. Fixed camera/sun/lights/
   fog. Print `froxel-lights: {lights:96, froxels:16x9x64, density:0.06, g:0.76}` (deterministic). INTERNALLY
   render (a) with `injectLights=false` and assert BYTE-IDENTICAL (SHA) to the CS sun-only `--froxelfog-shot`
   render of the SAME fog/scene, AND (b) with `density=0` and assert BYTE-IDENTICAL to the no-fog scene — fail
   loudly on any diff. New golden `tests/golden/metal/froxel_lights.png` (Metal two runs DIFF 0.0000).
   Existing 64 image goldens UNTOUCHED.

4. **Determinism.** Fixed lights (the CL deterministic lattice, no RNG), fixed fog params, fixed grid, fixed
   Z-march. Two runs byte-identical.

5. **Tests (extend `tests/froxel_test.cpp` OR new `tests/froxel_lights_test.cpp`, pure CPU, no GPU):**
   - **No lights = no added scatter:** `InjectClusteredLights(..., lightCount=0, ...)` → `(0,0,0)` exactly
     (the lights-off additive identity).
   - **A reaching light scatters:** a single point light within `radius` of the froxel → positive added
     scatter; the magnitude scales with `density` (×0 density → 0) and falls with distance (windowed); a light
     BEYOND `radius` → 0 added (the windowed hard cutoff, matching CL).
   - **Color + phase:** a colored light adds its color tint; the HG phase peaks when the light is along the
     view ray (forward-scatter) for g>0.
   - **Froxel→cluster Z map:** a froxel at a known view-Z maps to the correct cluster slice
     (`cluster::ViewZToSlice` round-trip); the froxel/cluster XY tiles align.
   - **Determinism:** same inputs → same scatter.
   - Plus the existing CS froxel_test cases still pass. Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `froxel-light-injection` (features) + `--froxellights-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses CL's cluster light grid + light SSBO via the existing `BindLightClusters` path + CS's
  froxel volume + compute barriers (`ComputeToCompute`/`ComputeToFragment`, already in `rhi.h` from CS). New
  non-backend code (`froxel.h` additions, the inject-shader light loop, the test) adds ZERO above-seam backend
  code symbols. Seam grep stays at baseline (2). Report the seam result.

## Out of scope (YAGNI)
Per-froxel light shadows (volumetric shadows from the shadow map / per-light shadow sampling — note future),
spot/area lights (point lights only, as CL), temporal/trilinear froxel upsampling (still nearest — separate
quality slice), more than CL's light count, light flicker/animation (fixed set), multiple-scattering. One
additive clustered-point-light in-scatter term in the froxel inject pass, with a lights-off==CS-fog proof + a
density=0==no-fog proof, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 64) + the new light-injection test cases (no-lights=0,
   reaching-light scatter + density/distance scaling + windowed cutoff, color+phase, froxel→cluster Z map,
   determinism). Clean under `windows-msvc-asan`.
2. **Two equivalence proofs + visual:** `--froxellights-shot` on Vulkan: the 96 colored point lights cast
   visible volumetric glow/shafts through the fog over the lit scene, coherent; the INTERNAL `injectLights=false`
   render is BYTE-IDENTICAL (SHA) to the CS `--froxelfog-shot` sun-only fog, AND the `density=0` render is
   BYTE-IDENTICAL to the no-fog scene; the `froxel-lights: {...}` line is deterministic (two runs →
   byte-identical capture). Run under the AT Vulkan-validation gate → ZERO errors (the inject→integrate→apply
   barriers stay SYNC-HAZARD-free).
3. Metal: `visual_test --froxellights` → new golden `tests/golden/metal/froxel_lights.png`; two runs DIFF
   0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `froxel_lights.png` added;
   the other 64 byte-identical (incl. `froxel_fog.png` + `clustered_lights.png`).
5. Introspect JSON rebaked exactly `+froxel-light-injection` + `--froxellights-shot`; introspect test updated;
   no other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `froxel_lights` image
   golden in the Mac round-trip loop.
