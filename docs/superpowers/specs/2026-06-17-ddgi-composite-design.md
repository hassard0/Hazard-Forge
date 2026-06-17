# Slice DN — DDGI Slice 5: GI Composite (the visible payoff) (Phase 5 #5) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. **The CLIMAX of the DDGI
> global-illumination arc** — where the probe SH irradiance becomes VISIBLE indirect light in the lit pass.
> A surface samples the 8 nearest probes' trilinearly-blended SH toward its normal and adds the indirect
> diffuse bounce → colored GI (red wall bleeds reddish onto the floor). ≈ UE4-parity local GI. Reuses
> DH/DI/DJ/DK (ray-trace → capture → SH-encode → interpolate) end to end. NO new RHI. Deterministic, with a
> giStrength=0 byte-identical no-op proof that keeps ALL 77 existing goldens untouched.

**Goal:** Add the DDGI indirect-diffuse term to a lit-pass variant, behind a `--ddgi-shot` flag, so the Cornell
box shows the indirect color bounce. Make-safe by construction: a SIBLING shader (`lit_ddgi.frag.hlsl`) used
ONLY by `--ddgi-shot` (the existing lit pass + its goldens are byte-for-byte untouched), and a `giStrength=0`
→ `rgb + 0.0` exact disabled path == the no-GI render byte-identical. The golden shows the visible color bleed.

## Reuse map (the whole arc, end to end)
- **DK `engine/render/probe_gi.h` + `probe_sh.h`** — `NearestProbes` + `InterpolateSH`/`InterpolateIrradiance`
  (the trilinear-blended SH at a world position, fma, GPU==CPU-bit-exact-proven).
- **DJ `probe_sh.h`** — `SHEvaluate` + `ProbeSH`.
- **DI/DJ showcase chain** — capture (DI) → SH-encode (DJ compute) → the per-probe `ProbeSH[]` SSBO.
- **The existing fragment-storage seam** — `usesLightClusters` + `BindLightClusters` (set 3, fragment-stage
  storage buffers), already reused for a single SSBO via dummies by froxel-apply + autoexposure-tonemap. NO
  new RHI.
- **The lit-variant isolation pattern** — `lit_clustered.frag.hlsl` / `lit_csm.frag.hlsl` /
  `lit_contactshadow.frag.hlsl` are sibling copies of `lit.frag.hlsl`; DN adds `lit_ddgi.frag.hlsl` the same
  way so the existing goldens are structurally untouched.

## Design decisions (locked)

1. **NEW sibling shader `shaders/lit_ddgi.frag.hlsl` (do NOT edit `lit.frag.hlsl`).** A copy of
   `shaders/lit.frag.hlsl` with the DDGI indirect term added as the LAST contribution, after the existing
   direct + env/ambient accumulation. Per pixel (world pos `wpos`, world normal `N`, `albedo`, `metallic`
   already available in the lit pass):
   ```hlsl
   // DDGI indirect diffuse (NEW). Trilinearly-blended probe SH irradiance toward N, modulated by albedo.
   // giStrength==0 -> + 0.0 EXACTLY (literal-zero multiply) -> byte-identical to the no-GI render.
   float3 indirect = InterpolateIrradianceSH(wpos, N) * (1.0 - metallic);
   rgb += indirect * albedo * gGi.giStrength;
   ```
   `InterpolateIrradianceSH(wpos, N)` is the VERBATIM in-shader copy of `probegi::NearestProbes` +
   `probesh::InterpolateSH` + `probesh::SHEvaluate` (the shared-math-copied-into-shader rule, with `fma`/`mad`
   per the DH cross-backend discipline), reading the `ProbeSH[]` from the SSBO + the `ProbeGrid` params +
   `giStrength` from a small uniform block (`gGi`). The existing `lit.frag.hlsl` + ALL its goldens stay
   BYTE-IDENTICAL (DN never touches it). HLSL→SPIR-V→MSL via the existing toolchain.

2. **Bind the ProbeSH SSBO to the fragment stage via the EXISTING path (no new RHI).** The `--ddgi-shot`
   pipeline sets `litDesc.usesLightClusters = true` (declares set 3 = three fragment-stage storage buffers,
   as `lit_clustered` does); DN needs ONE storage buffer (the `ProbeSH[]`), so it binds the single SSBO via
   dummies: `cmd.BindLightClusters(*probeShBuf, *dummyBuf, *dummyBuf)` (the froxel-apply / autoexposure
   single-SSBO-via-dummies idiom). Shader: `[[vk::binding(13,3)]] StructuredBuffer<ProbeSH> gProbeSH`. The
   `ProbeGrid` params (origin/dims/spacing) + `giStrength` ride in the `--ddgi-shot` showcase's OWN FrameData
   UBO (the `lit_clustered.frag` precedent — its own FrameData layout with spare fields). NO new RHI seam, NO
   descriptor-set change, NO new virtual.

3. **Showcase `--ddgi-shot <out>` (Vulkan) / `--ddgi` (Metal).** The DJ Cornell box (red −X wall, green +X
   wall, neutral floor/ceiling/back) + a probe grid inside the room (e.g. 3×3×3 or 4×3×4 — enough for smooth
   bleed; keep render+descriptor budgets in mind per the DK 75-swatch lesson). Pipeline chain: capture (DI) →
   SH-encode (DJ compute) → interpolate-in-shader (DK) → `lit_ddgi` render (ProbeSH SSBO bound) → post.
   - Render **frame A** with `giStrength = 0` → the flat-ambient lit room (the no-GI baseline).
   - Render **frame B** with `giStrength > 0` → the indirect bounce: the red wall bleeds reddish onto the
     adjacent floor, the green wall bleeds green onto the opposite side — the classic Cornell color bleed, the
     VISIBLE DDGI payoff.
   - **INTERNAL PROOF (make-or-break):** frame A (`giStrength=0`) is BYTE-IDENTICAL (SHA) to a no-GI reference
     render of the same scene through the same `lit_ddgi` pipeline (the indirect term is `+0.0` exactly). ALSO
     render the `probeCount=0` (dimX=0 → `InterpolateSH` zero fallback) variant and assert it == frame A. Print
     `ddgi giStrength=0 == no-GI: BYTE-IDENTICAL`.
   - **Golden** = frame B → `tests/golden/metal/ddgi.png` (the color bleed); Metal two runs DIFF 0.0000, gate
     on compare.sh EXIT CODE. Print `ddgi: {probes:N, bands:3, giStrength:S}`. Existing 77 image goldens
     UNTOUCHED. **GOLDEN DISCIPLINE: commit ONLY `tests/golden/metal/ddgi.png` — do NOT commit a loose
     `tests/golden/ddgi.png` (the DK stray trap).**

4. **Determinism.** Fixed scene/grid/giStrength, the whole chain deterministic (DH/DI/DJ/DK proven). Two runs
   byte-identical. The `lit_ddgi` shading is screen-space output BUT the GI term is world-anchored probe data;
   the golden is the rendered image (verify cross-backend honestly — if the screen render is cross-backend-
   stable like the other lit goldens, great; the giStrength=0==no-GI proof is per-backend regardless).

5. **Tests `tests/ddgi_test.cpp` (pure CPU, no GPU) — OR extend probe_interp/probe_sh tests:**
   - **giStrength=0 identity:** the composite term `indirect * albedo * 0 == 0` (trivial but assert the
     contribution-at-zero is exactly zero in the CPU mirror).
   - **Indirect from a colored probe:** a `ProbeSH` encoding red-dominant irradiance → `InterpolateIrradiance`
     toward a normal facing it → a red-dominant indirect color (the color-bleed direction is correct).
   - **probeCount=0 → zero indirect** (the DK fallback, re-asserted in the composite context).
   - **Determinism.**
   - Clean under `windows-msvc-asan`. (Most of the math is already covered by probe_interp_test/probe_sh_test;
     add only the composite-specific cases.)

6. **Introspect.** Add exactly `ddgi-global-illumination` (features) + `--ddgi-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses `usesLightClusters` + `BindLightClusters` (existing fragment-storage path) for the ProbeSH
  SSBO; the GI params ride in the showcase FrameData UBO; the indirect math is in `lit_ddgi.frag.hlsl` (a new
  above-seam shader, zero backend symbols). New non-backend code (`lit_ddgi.frag.hlsl`, the composite test,
  the showcase block) adds ZERO above-seam backend symbols. rhi.h + rhi_factory + the backend dirs UNCHANGED.
  rhi_factory dispatch baseline 2. Report the seam.

## Out of scope (YAGNI)
Multi-bounce / infinite-bounce GI (single bounce from the captured probes), per-frame dynamic re-capture,
probe visibility/occlusion (chebyshev/variance) weighting, specular GI / glossy from probes (diffuse
irradiance only), probe relocation, the DDGI in the DEFAULT lit pass (keep it the `--ddgi-shot` variant so the
77 goldens are untouched). One single-bounce diffuse DDGI composite with a giStrength=0 byte-identical no-op
proof + the visible Cornell color-bleed golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 77) + the new composite test cases. Clean under
   `windows-msvc-asan`.
2. **giStrength=0 no-op proof + the VISIBLE GI:** `--ddgi-shot` on Vulkan: frame B shows the indirect color
   bleed (red wall → reddish floor, green wall → green floor) — coherent, recognizable Cornell GI; the
   INTERNAL `giStrength=0` render is BYTE-IDENTICAL (SHA) to the no-GI reference, AND the `probeCount=0`
   variant == frame A; the `ddgi: {...}` line is deterministic (two runs → byte-identical capture). Run under
   the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --ddgi` → new golden `tests/golden/metal/ddgi.png`; two runs DIFF 0.0000 (gate on the
   compare.sh EXIT CODE). The giStrength=0==no-GI proof also passes on Metal.
4. **Render-invariance (CRITICAL):** `git diff master --stat -- tests/golden/metal` shows ONLY `ddgi.png`
   added; the other 77 byte-identical (the sibling-shader isolation keeps the existing lit goldens untouched).
   `git diff master --stat -- tests/golden` shows ONLY `ddgi.png` (metal) + the 2-line introspect json — NO
   loose `tests/golden/ddgi.png`, NO other golden, NO existing `.frag`/pipeline/bind-call changed.
5. Introspect JSON rebaked exactly `+ddgi-global-illumination` + `--ddgi-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI; the existing `lit.frag.hlsl` + all existing pipelines
   UNCHANGED). `scripts/verify.ps1` updated to include the new `ddgi` image golden in the Mac round-trip loop
   (gate on compare.sh EXIT CODE).
