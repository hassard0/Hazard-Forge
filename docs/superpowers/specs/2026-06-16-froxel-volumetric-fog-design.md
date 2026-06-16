# Slice CS — Froxel Volumetric Fog (sun single-scattering) (Phase 4 #42) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A flagship atmosphere
> feature: a true 3D view-space FROXEL volume with physically-based front-to-back single-scattering and
> correct per-pixel-depth transmittance — the modern UE5/Frostbite volumetric-fog architecture. DISTINCT from
> the existing 2D screen-space light-shaft god-ray (that's a radial post blur; this integrates a 3D volume and
> composites by each pixel's depth). Sun-scattering this slice; per-froxel light injection is a future slice.
> Deterministic at a FIXED sun + fog params, with a clean density=0 no-op proof.

**Goal:** A froxel (frustum-voxel) volumetric fog: the view frustum is partitioned into a 3D froxel grid; a
compute pass writes per-froxel scattering+extinction (fog density × in-scattered sunlight via a phase
function); a second pass integrates front-to-back along Z into per-froxel accumulated (in-scatter,
transmittance); a fullscreen apply samples the integrated volume at each pixel's view depth and composites
`scene*transmittance + inScatter`. A `--froxelfog-shot` showcase shows the scene wrapped in volumetric haze
with sunlit in-scattering (glow toward the sun, distance haze); golden-verified on both backends. The pass
carries its proof: with fog density 0 the transmittance is 1 and in-scatter is 0 everywhere → the
fog-composited image is BYTE-IDENTICAL to the no-fog scene — asserted internally, fail loudly on any diff.

## Why this is render-safe (the zero-density no-op proof)

Single-scattering compositing is `out = scene * T + L` where `T = Π exp(-σ·Δz)` (transmittance) and `L` is the
accumulated in-scatter. With fog density 0 → extinction σ = 0 → `T = 1` exactly, and scattering = 0 → `L = 0`
→ `out = scene` exactly. So the showcase INTERNALLY renders with `baseDensity = 0` and asserts it is
BYTE-IDENTICAL (SHA) to the engine's no-fog render of the same scene — proving the froxel build + integration
+ apply is a pure pass-through at zero density (no constant bias, no sampling offset, no off-by-one in the
Z-slice mapping) — then renders the real `baseDensity > 0` version as the golden. (Same internal-assert
discipline as CN/CO/CP/CR.)

## Design decisions (locked)

1. **Froxel + fog math (engine/render/froxel.h, header-only pure CPU, no backend symbols).** Namespace
   `hf::render::froxel`. Mirrors `cluster.h`/`clouds.h`/`gtao.h` (shared with the shaders + the unit test).
   - `struct FroxelGrid { int dimX, dimY, dimZ; float zNear, zFar; };` — view-space froxel grid (e.g.
     16×9×64), exponential Z slicing `SliceZ(k) = zNear*(zFar/zNear)^(k/dimZ)` (same distribution as
     `cluster.h`; document). `float SliceZ(const FroxelGrid&, float k)` + `float ViewZToSlice(const
     FroxelGrid&, float viewZ)` — exact inverses (unit-tested: `ViewZToSlice(SliceZ(k)) == k`), used by the
     apply to sample the volume at a pixel's depth.
   - `float Density(const math::Vec3& worldPos, float baseDensity, float heightFalloff, float heightRef)` —
     height-based exponential fog density `baseDensity * exp(-heightFalloff * (worldPos.y - heightRef))`,
     clamped ≥0; returns 0 when `baseDensity == 0`. Document units.
   - `float Phase(float cosTheta, float g)` — Henyey-Greenstein forward-scatter phase (the fog scatters sun
     toward the camera). `Phase(cos, 0)` isotropic; forward-peaked for g>0. (May reuse the clouds.h HG;
     document.)
   - `void IntegrateStep(const math::Vec3& stepScatter, float stepExtinction, float stepLen, math::Vec3&
     accumInScatter, float& transmittance)` — one front-to-back single-scattering step:
     `accumInScatter += transmittance * stepScatter * stepLen; transmittance *= exp(-stepExtinction*stepLen)`.
     At `stepExtinction==0 && stepScatter==0` → no change (transmittance stays 1, inScatter stays 0).
     Document. The shader's per-column Z-march uses this; the test verifies a known density profile yields the
     analytic transmittance `exp(-∫σ)`.

2. **Froxel shaders.** `shaders/froxel_inject.comp.hlsl` — one thread per froxel: reconstruct the froxel's
   view (and world) center (from the grid + invProj), compute `Density` × sun in-scatter
   (`sunColor * Phase(cos(viewDir,sunDir), g) * density`) and extinction (`density`), write
   `(scatterRGB, extinction)` to a flat SSBO froxel volume `[dimX*dimY*dimZ]`. `shaders/froxel_integrate.comp.hlsl`
   — one thread per (x,y) column: march Z 0→dimZ front-to-back via the `IntegrateStep` math, writing the
   accumulated `(inScatterRGB, transmittance)` per froxel into the volume. `shaders/froxel_apply.frag.hlsl`
   — a fullscreen pass: per pixel reconstruct view depth (G-buffer), `ViewZToSlice` → sample the integrated
   volume (nearest or trilinear — document; nearest is simpler + deterministic), composite
   `scene*transmittance + inScatter`. EXISTING scene/sky shaders + their goldens UNTOUCHED (froxel fog is a
   NEW path behind the showcase flag). Reuses the existing SSBO + compute-dispatch + fullscreen-composite path
   (like cluster_assign / gpu-cull / the water/SSGI composites). NO new RHI seam. HLSL→SPIR-V→MSL via the
   existing toolchain.

3. **Showcase `--froxelfog-shot <out>` (Vulkan) / `--froxelfog` (Metal).** The standard lit+shadowed scene
   (ground + objects) wrapped in volumetric fog: distance haze + sunlit in-scattering (a soft glow toward the
   sun direction), objects correctly fogged by their depth (nearer = less fog). Fixed camera, fixed sun, fixed
   fog params. Print `froxel-fog: {froxels:DIMX*DIMY*DIMZ, density:D, g:G}` (deterministic). INTERNALLY render
   with `baseDensity = 0` and assert BYTE-IDENTICAL (SHA) to the no-fog render — fail loudly on any diff. New
   golden `tests/golden/metal/froxel_fog.png` (Metal two runs DIFF 0.0000). Existing 62 image goldens
   UNTOUCHED (incl. the existing 2D light-shaft `volumetric.png` — this is a separate path).

4. **Determinism.** Fixed grid/sun/fog params, deterministic density (no RNG), fixed Z-march. Two runs
   byte-identical.

5. **Tests `tests/froxel_test.cpp` (pure CPU, no GPU):**
   - **Z-slice inverse:** `ViewZToSlice(SliceZ(k)) == k` (within tolerance) across the grid; `SliceZ`
     monotone increasing; covers `[zNear, zFar]`; slice 0 = zNear, slice dimZ = zFar.
   - **Density:** `Density` height falloff (higher y → less fog with positive falloff); `baseDensity == 0` →
     0 everywhere; non-negative.
   - **Phase:** `Phase(1, g>0)` > `Phase(-1, g>0)` (forward-peaked); `Phase(cos, 0)` constant (isotropic);
     finite for all cos∈[-1,1].
   - **Integration no-op + analytic:** a column of zero density → `transmittance == 1`, `inScatter == 0`
     (the no-op); a column of constant density σ over length L → `transmittance == exp(-σ*L)` (within
     tolerance, the analytic Beer-Lambert); transmittance monotone decreasing along the march; in-scatter
     accumulates with correct front-to-back weighting (a near scatterer contributes at full transmittance).
   - **Determinism:** same inputs → same result.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `froxel-volumetric-fog` (features) + `--froxelfog-shot` (showcases).

## RHI seam additions (summary)
- **None expected.** The froxel volume is a FLAT SSBO (NOT a 3D texture) written/read via the existing
  storage-buffer + compute-dispatch path (as cluster_assign / gpu-cull do); the apply reuses the fullscreen-
  composite path (as water/SSGI composites do). New non-backend files (`engine/render/froxel.h`, the 3
  shaders, `tests/froxel_test.cpp`) add ZERO above-seam backend code symbols. If a genuinely-missing
  capability forces an additive interface, it goes in `rhi.h` as a pure interface with backend-dir impls
  (document it) — but PREFER reusing the existing SSBO/compute plumbing (no new seam). Report the seam result.

## Out of scope (YAGNI)
Per-froxel light injection from the clustered point lights (CL) — THIS slice is sun-scattering only; clustered
light injection is the explicit follow-up slice (note it). Temporal reprojection of the froxel volume,
trilinear/temporal jitter, volumetric shadows (sun-visibility per froxel via the shadow map — note future),
multiple-scattering, height-fog layering / wind animation, 3D-texture volume (use the flat SSBO),
participating-media albedo color maps. One fixed-param sun-scattering froxel volume (inject → integrate →
depth-composited apply) with a zero-density no-op proof, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 62) + new `froxel_test` (Z-slice inverse, density
   height falloff + zero, phase forward-peak/isotropic, integration no-op + analytic Beer-Lambert + front-to-
   back, determinism). Clean under `windows-msvc-asan`.
2. **Zero-density no-op proof + visual:** `--froxelfog-shot` on Vulkan: a recognizable volumetric fog (distance
   haze + sunlit in-scatter glow, objects fogged by depth), coherent; the INTERNAL baseDensity=0 render is
   BYTE-IDENTICAL (SHA) to the no-fog render; the `froxel-fog: {...}` line is deterministic (two runs →
   byte-identical capture). Run under the AT Vulkan-validation gate → ZERO errors (watch the compute/SSBO
   sync — barriers between inject→integrate→apply).
3. Metal: `visual_test --froxelfog` → new golden `tests/golden/metal/froxel_fog.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `froxel_fog.png` added;
   the other 62 byte-identical (incl. the existing 2D `volumetric.png` light-shaft golden).
5. Introspect JSON rebaked exactly `+froxel-volumetric-fog` + `--froxelfog-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (no new above-seam code symbols; report any rhi.h additive interface). `scripts/verify.ps1`
   updated to include the new `froxel_fog` image golden in the Mac round-trip loop.
