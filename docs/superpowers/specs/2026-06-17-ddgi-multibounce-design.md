# Slice DR — DDGI Multi-Bounce (2nd light bounce) (Phase 5 #8) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. **The next GI-quality leap**:
> single-bounce DDGI (the DH–DN arc) captures only DIRECT light into the probes, so indirect light bounces exactly
> once. Multi-bounce feeds the 1st-bounce GI BACK into the probe capture, so a second capture sees the indirect
> light on surfaces and the composite gets a 2nd bounce (brighter, more realistic GI — the step from UE4-style
> DDGI toward Lumen's infinite-bounce). Done as a FIXED 2-iteration pass in one frame (deterministic, golden-able),
> not a temporal feedback loop. Builds on DN (single-bounce-no-occlusion; orthogonal to DP occlusion). NO new RHI.
> A `bounceCount=1` disabled path is byte-identical to the single-bounce DN render. Branch: slice-dr-multibounce.

**Goal:** Add a `--ddgimb-shot` showcase that renders the DN Cornell box GI with a 2nd light bounce, and a
`bounceCount` knob. Bounce 0 = the existing DI direct-light capture → SH₀. Bounce 1 (only when `bounceCount≥2`) =
RE-capture each probe with the DN indirect term (sampling SH₀) added to the captured surfaces → SH-encode → SH₁.
The composite uses SH₍bounceCount-1₎. Make-safe: a NEW capture-with-GI shader + a NEW showcase + a NEW golden;
`bounceCount=1` skips the bounce-1 pass → the composite uses SH₀ → BYTE-IDENTICAL to the single-bounce DN render.
Existing 80 goldens untouched.

## Reuse map (the whole arc, end to end)
- **DI `engine/render/probe_capture.h`** — the per-probe cube-capture loop (single DD cube RT, BeginCubemapFace×6,
  ReadCubemapFace into the radiance store). Bounce 1 reuses this loop verbatim; only the capture FRAGMENT shader
  differs (adds the indirect term).
- **DJ `shaders/probe_sh_encode.comp.hlsl` + `probe_sh.h`** — SH-encode of a captured cube. Bounce 1 re-runs the
  SAME encode on the bounce-1 radiance → SH₁ (in a 2nd `ProbeSH[]` buffer).
- **DN `shaders/lit_ddgi.frag.hlsl` + the indirect term** (`InterpolateIrradianceSH(wpos,N)` trilinear-SH
  irradiance). The capture-with-GI shader adds the SAME indirect term to the captured surface radiance; the final
  composite IS `lit_ddgi` unchanged.
- **The fragment-storage seam** — `usesLightClusters` + `BindLightClusters` binds the ProbeSH SSBO to the fragment
  stage. The capture-with-GI pipeline binds SH₀ the same way (the capture pipeline sets `usesLightClusters=true`,
  binds `BindLightClusters(*sh0Buf,*dummy,*dummy)`). NO new RHI.

## Design decisions (locked)

1. **NEW capture-with-GI shader `shaders/probe_bake_gi.frag.hlsl` (do NOT edit DD's `probe_bake.frag`).** A copy
   of the DI capture fragment shader (which outputs the directly-lit scene radiance from the probe's view) with
   the DN indirect term added as the LAST contribution, per captured fragment:
   ```hlsl
   // 2nd-bounce feed: add the 1st-bounce GI (SH0 trilinear irradiance) to the captured surface radiance.
   float3 indirect = InterpolateIrradianceSH(wpos, N) * (1.0 - metallic);   // SH0 from gProbeSH
   radiance += indirect * albedo * gGi.giStrength;
   ```
   `InterpolateIrradianceSH` is the VERBATIM in-shader copy from DN (`NearestProbes`+`InterpolateSH`+`SHEvaluate`,
   `mad`/the DH FP discipline), reading SH₀ from the bound `ProbeSH[]` SSBO + the ProbeGrid params from the
   capture UBO. This shader is used ONLY by the bounce-1 capture in `--ddgimb-shot`; DD's `probe_bake.frag` + all
   DI/DN goldens stay byte-untouched.

2. **Two SH buffers (SH₀, SH₁); the composite picks by `bounceCount` (no new RHI).** SH₀ = DI-direct-capture →
   DJ-encode (the existing path). If `bounceCount≥2`: bounce-1 capture (probe_bake_gi, SH₀ bound to the capture
   fragment stage) → DJ-encode into SH₁. The final `lit_ddgi` composite binds `gProbeSH = (bounceCount≥2 ? SH₁ :
   SH₀)` via the existing `BindLightClusters` path. Both SH buffers are plain `ProbeSH[]` SSBOs (DJ's type); the
   capture pipeline + the encode compute are unchanged. NO new RHI, NO descriptor-set change.

3. **Showcase `--ddgimb-shot <out>` (Vulkan) / `--ddgimb` (Metal).** The DN 2×2×2=8-probe Cornell box (red/green
   walls). Pipeline: DI direct capture → DJ encode → SH₀; [if bc≥2: probe_bake_gi capture (SH₀) → DJ encode → SH₁];
   `lit_ddgi` composite (SH₍bc-1₎).
   - **Frame A** `bounceCount = 1` → composite uses SH₀ → the single-bounce GI.
   - **Frame B** `bounceCount = 2` → composite uses SH₁ → the 2nd-bounce GI: the indirect light is brighter and
     the color bleed is stronger (the walls' bounced light now itself bounces). The classic multi-bounce
     brightening — visibly more filled-in than frame A.
   - **INTERNAL PROOF (make-or-break):** frame A (`bounceCount=1`) is BYTE-IDENTICAL (SHA) to the single-bounce DN
     render of the same scene (the same `lit_ddgi` composite over SH₀ — bounce-1 is skipped entirely, so it IS the
     DN path). Print `ddgi-mb bounceCount=1 == DN single-bounce: BYTE-IDENTICAL`. ALSO assert frame B DIFFERS from
     A and is BRIGHTER in the indirect region (`ddgi-mb B brighter than A: 2nd bounce active` — e.g. mean luminance
     of frame B > frame A by a measurable margin).
   - **GPU==CPU SH₁ proof:** the bounce-1 SH₁ SSBO is BIT-EXACT to a CPU reference that SH-encodes the same
     bounce-1 read-back radiance (the DJ GPU==CPU proof, re-applied to the 2nd-bounce capture — the radiance feeding
     the encode is read back, so the encode is bit-exact regardless of the capture's sqrt/lighting). Print
     `ddgi-mb SH1 GPU==CPU: BIT-EXACT`.
   - **probeCount=0 / bounceCount=0:** probeCount=0 → indirect 0 (as DN); bounceCount clamped ≥1.
   - **Golden** = frame B → `tests/golden/metal/ddgi_mb.png` (the 2nd-bounce GI); Metal two runs DIFF 0.0000, gate
     on the compare.sh EXIT CODE. Print `ddgi-mb: {probes:8, bounces:2, giStrength:S}`. Existing 80 image goldens
     UNTOUCHED. **GOLDEN DISCIPLINE: commit ONLY `tests/golden/metal/ddgi_mb.png` — NO loose
     `tests/golden/ddgi_mb.png` (the DK stray trap).**

4. **Determinism / cross-backend.** Fixed scene/grid/bounceCount/giStrength; the whole 2-iteration chain is
   deterministic (each capture+encode is the DI/DJ proven path). Two runs byte-identical. SH₀/SH₁ are world/probe
   data; the screen render is verified cross-backend honestly (the bounceCount=1==DN byte-identity is per-backend
   regardless; for frame B the golden is baked + 2-run-compared per backend). The capture's `length`/lighting sqrt
   cross-backend caveat is sidestepped for the SH₁ GPU==CPU proof by reading back the bounce-1 RADIANCE and
   encoding from those exact bytes (the DJ discipline), and for the bounceCount=1 no-op by skipping bounce-1
   entirely.

5. **Tests `tests/ddgi_multibounce_test.cpp` (pure CPU) — OR extend ddgi_test:**
   - **bounceCount=1 selects SH₀:** the buffer-selection logic (`bc≥2 ? SH1 : SH0`) returns SH₀ at bc=1 (a tiny
     pure-logic helper `SelectBounceSH` in a header, unit-tested) → guarantees the no-op path uses SH₀.
   - **2nd-bounce monotonicity (CPU mirror):** a CPU mirror of "capture-with-GI adds a non-negative indirect to a
     non-negative radiance" → SH₁ irradiance ≥ SH₀ irradiance for a positive-albedo scene (the bounce only adds
     light) — assert the 2nd bounce is brighter, never darker.
   - **probeCount=0 → zero indirect; determinism.** Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `ddgi-multi-bounce` (features) + `--ddgimb-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses the DI cubemap capture RT, the DJ SH-encode compute, and the `usesLightClusters`/
  `BindLightClusters` fragment-storage path (for both the capture-with-GI SH₀ bind and the composite SH bind).
  Two `ProbeSH[]` SSBOs are just two buffers of the existing DJ type. New non-backend code (`probe_bake_gi.frag`,
  the bounce helper header, the test, the showcase) adds ZERO above-seam backend symbols. rhi.h + rhi_factory
  (dispatch baseline 2) + the backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI)
N>2 bounces / true infinite-bounce (exactly 2 captures — the 2nd bounce is the demonstrable leap; N-bounce is a
trivial loop extension later), temporal/cross-frame feedback, combining with DP occlusion (DR builds on DN; the
occluded multi-bounce is a future compose), specular multi-bounce. ONE fixed-2-bounce DDGI with a bounceCount=1
byte-identical-to-DN proof + the SH₁ GPU==CPU proof + the visibly-brighter 2nd-bounce golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 80) + the new multibounce test. Clean under
   `windows-msvc-asan`.
2. **no-op proof + VISIBLE 2nd bounce:** `--ddgimb-shot` on Vulkan: frame B is coherent and visibly brighter /
   more filled-in than frame A (the 2nd bounce); the INTERNAL `bounceCount=1` render is BYTE-IDENTICAL (SHA) to
   the DN single-bounce render; `ddgi-mb SH1 GPU==CPU: BIT-EXACT`; `ddgi-mb B brighter than A`; the `ddgi-mb: {...}`
   line deterministic (two runs byte-identical). Run under the AT Vulkan-validation gate → ZERO errors (the
   bounce-1 capture→encode→composite barriers SYNC-HAZARD-free; the showcase declares the empty shadow pass per the
   DQ lesson so it is validation-clean).
3. Metal: `visual_test --ddgimb` → new golden `tests/golden/metal/ddgi_mb.png`; two runs DIFF 0.0000 (gate on the
   compare.sh EXIT CODE). The bounceCount=1==DN + SH₁ GPU==CPU proofs also pass on Metal.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `ddgi_mb.png` added; the other
   80 byte-identical (incl `ddgi`/`ddgi_occ` — the new capture shader + showcase touch nothing existing).
   `git diff master --stat -- tests/golden` = ONLY `ddgi_mb.png` (metal) + the 2-line introspect json — NO loose
   `tests/golden/ddgi_mb.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+ddgi-multi-bounce` + `--ddgimb-shot`; introspect test updated; no other drift.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI; `probe_bake.frag` + `lit_ddgi.frag.hlsl` + `lit.frag.hlsl`
   UNCHANGED). `scripts/verify.ps1` updated to include the new `ddgi_mb` image golden in the Mac round-trip loop
   AND `--ddgimb-shot` in the `$vkShots` validation gate (the DQ hardening — every new GI showcase joins the gate).
