# Slice DP — DDGI Visibility Slice 2: Chebyshev Occlusion Weighting (Phase 5 #7) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. **The VISIBLE payoff of the
> DDGI visibility sub-arc** — consumes DO's per-probe distance-moment cube to compute a per-probe Chebyshev
> (variance-shadow-style) visibility weight, so a surface NO LONGER receives indirect light from probes that are
> geometrically occluded from it (a probe behind a wall stops bleeding light through the wall). This is the #1
> quality gap of plain irradiance-probe GI and the real DDGI (Majercik et al. 2019) fix. NO new RHI. A
> `occlusionStrength=0` byte-identical-to-DN no-op proof keeps it make-safe. Branch: slice-dp-occlusion.

**Goal:** Add a sibling lit-pass variant `lit_ddgi_occ.frag.hlsl` (copy of DN's `lit_ddgi.frag.hlsl` + per-probe
Chebyshev visibility weighting that reads DO's distance-moment SSBO), behind a `--ddgiocc-shot` flag, on a Cornell
box that contains an OCCLUDER panel so the leak-fix is visible. Make-safe by construction: a SIBLING shader (the
existing `lit_ddgi.frag.hlsl` + the `ddgi` golden + all 79 goldens are byte-for-byte untouched) and an
`occlusionStrength=0` → visibility==1 → byte-identical-to-DN-math disabled path.

## Reuse map (the whole visibility sub-arc, end to end)
- **DN `shaders/lit_ddgi.frag.hlsl`** — the GI composite this copies + extends (the `NearestProbes` +
  `InterpolateSH` + `SHEvaluate` in-shader trilinear-SH-irradiance term). DP weights each of the 8 probes by
  visibility BEFORE the SH blend.
- **DO `engine/render/probe_dist.h`** — `ProbeDistMoments{m[2]}` (meanDist, meanDist²), the flat store layout
  `p*384 + f*64 + v*8 + u`, `DistTexelCount`. DP samples this store per corner-probe in the probe→shading-point
  direction.
- **DD `engine/render/cubemap.h`** — the direction↔cube-face↔uv math. DP maps the world-space probe→point
  direction to a `(face, u, v)` to index the DO moment store (the SAME face convention DO captured with —
  `FaceView`/`FaceProj` axis order; match it EXACTLY or the query reads the wrong texel). Copy the dir→face/uv
  math verbatim into the shader (shared-math-copied-into-shader rule) + add a CPU mirror in `probe_dist.h`
  (`DistDirToFaceUV` / `SampleProbeMoments`) so the visibility math gets a GPU==CPU / CPU-unit proof.
- **DN fragment-storage seam** — `usesLightClusters` + `BindLightClusters` binds THREE fragment-stage storage
  buffers; DN used slot 0 (ProbeSH) + dummies. DP binds slot 0 = ProbeSH, slot 1 = ProbeDistMoments, slot 2 =
  dummy: `cmd.BindLightClusters(*probeShBuf, *probeDistBuf, *dummyBuf)`. **NO new RHI** (the 2nd slot already
  exists). Shader: `[[vk::binding(13,3)]] StructuredBuffer<ProbeSH> gProbeSH` + `[[vk::binding(14,3)]]
  StructuredBuffer<ProbeDistMoments> gProbeDist` (confirm DN's binding indices + the set-3 layout; use the next
  free slot the existing 3-buffer cluster set declares).

## Design decisions (locked)

1. **NEW sibling shader `shaders/lit_ddgi_occ.frag.hlsl` (do NOT edit `lit_ddgi.frag.hlsl`).** A copy of
   `lit_ddgi.frag.hlsl` where the indirect term's 8-probe blend is weighted by per-probe visibility. Replace the
   plain `InterpolateIrradianceSH(wpos, N)` (which internally does `NearestProbes` → 8 weights → `InterpolateSH`)
   with a visibility-weighted version:
   ```hlsl
   // For each of the 8 corner probes c: trilinear weight w[c] (from NearestProbes), then a Chebyshev
   // visibility weight vis[c] from the DO distance moments, then renormalize and blend SH.
   ProbeTrilinear tri = NearestProbes(wpos, grid);     // idx[8], w[8], valid
   float wsum = 0; float vw[8];
   for (c in 0..7) {
     float3 ppos = probePos(grid, tri.idx[c]);
     float3 dir  = wpos - ppos;  float dist = length(dir);  dir = normalize(dir);
     float2 mom  = SampleProbeMoments(gProbeDist, tri.idx[c], dir);   // {meanDist, meanDist^2}
     float mean = mom.x;  float var = max(0.0, mom.y - mean*mean);
     float dd = max(0.0, dist - mean);
     float cheb = (dist <= mean) ? 1.0 : var / (var + dd*dd);          // Chebyshev upper bound, [0,1]
     float vis  = lerp(1.0, cheb, gGi.occlusionStrength);              // occlusionStrength=0 -> vis=1 EXACTLY
     vw[c] = tri.w[c] * vis;  wsum += vw[c];
   }
   // renormalize (fall back to unweighted if all occluded -> wsum<=eps): keep DN behavior when vis==1 for all.
   ProbeSH blended = (zeroed);
   if (wsum > 1e-6) for (c in 0..7) for (i in 0..8) for (ch in 0..2)
     blended.coeff[i][ch] = mad(vw[c]/wsum, gProbeSH[tri.idx[c]].coeff[i][ch], blended.coeff[i][ch]);
   float3 indirect = SHEvaluate(blended, N) * (1.0 - metallic);
   rgb += indirect * albedo * gGi.giStrength;
   ```
   **CRITICAL no-op identity:** when `occlusionStrength == 0`, every `vis==1` so `vw[c]==tri.w[c]`,
   `wsum==Σtri.w[c]==1` (partition of unity), and `vw[c]/wsum == tri.w[c]` — so the blend is BIT-IDENTICAL to
   DN's `InterpolateSH` **provided the renormalized blend reproduces DN's exact fma order**. To guarantee
   byte-identical-to-DN at occlusionStrength=0, the `vw[c]/wsum` path must reduce to DN's exact arithmetic. SAFEST:
   special-case `occlusionStrength==0` (or `wsum` within ULPs of 1 with all vis==1) to call the VERBATIM DN
   `InterpolateSH` path, so frame A is **literally the DN code path** → guaranteed byte-identical. Document this
   branch. (The division `vw/wsum` otherwise introduces a rounding DN doesn't have → would break the byte-identity;
   the explicit occlusionStrength==0 → DN-path branch is the make-or-break guarantee.) Use `fma`/`mad` per the DH
   cross-backend discipline throughout the weighted path.

2. **Bind ProbeSH + ProbeDistMoments to the fragment stage via the EXISTING path (no new RHI).** As DN, set
   `litDesc.usesLightClusters = true`; bind `BindLightClusters(*probeShBuf, *probeDistBuf, *dummyBuf)`. The
   `ProbeGrid` params + `giStrength` + the NEW `occlusionStrength` ride the showcase's own FrameData UBO (DN
   precedent). NO new RHI seam, NO descriptor-set change, NO new virtual. rhi.h + rhi_factory + backend dirs
   UNCHANGED.

3. **`SampleProbeMoments` (extend `engine/render/probe_dist.h`, pure CPU + copied verbatim into the shader).**
   `float2 SampleProbeMoments(const ProbeDistMoments* store, int probeIdx, Vec3 dir)`: map `dir` to `(face,u,v)`
   via `DistDirToFaceUV(dir)` (the DD cube-face convention — dominant axis selects the face, the other two →
   `[0,distFace)` texel coords; match DO's `FaceView`/capture orientation EXACTLY), then read
   `store[probeIdx*384 + face*64 + v*8 + u].m`. Start with NEAREST-texel sampling (deterministic, bit-exact, no
   filter ambiguity); document that bilinear is a future refinement. Add `DistDirToFaceUV` (pure, unit-tested:
   the 6 axis directions map to the 6 face centres; round-trips DO's capture face math). This is the SAME math the
   shader runs (copied verbatim, `mad` where multiply-adds occur).

4. **Showcase `--ddgiocc-shot <out>` (Vulkan) / `--ddgiocc` (Metal).** A Cornell box WITH AN OCCLUDER: add an
   interior panel/wall that separates a lit (e.g. coloured) region from a shaded region so a probe on the bright
   side is OCCLUDED from surfaces on the far side. Pipeline: DO distance capture (per-probe distance cube) → DN SH
   capture+encode (per-probe ProbeSH) → `lit_ddgi_occ` render (both SSBOs bound). 
   - Render **frame A** with `occlusionStrength = 0` → the DN-equivalent (leaky) GI.
   - Render **frame B** with `occlusionStrength = 1` → the Chebyshev-corrected GI: the indirect bleed that
     previously leaked through the occluder onto the far-side surface is GONE / strongly attenuated (the visible
     payoff). 
   - **INTERNAL PROOF (make-or-break):** frame A (`occlusionStrength=0`) is BYTE-IDENTICAL (SHA) to the SAME scene
     rendered through the VERBATIM DN `lit_ddgi` path (occlusion off == pure DN math). Print
     `ddgi-occ occlusionStrength=0 == DN: BYTE-IDENTICAL`. Also assert frame B DIFFERS from frame A (the occlusion
     actually changed pixels — `ddgi-occ B != A: occlusion active`).
   - **VISIBILITY MATH PROOF:** a CPU mirror — for a fixed probe + a set of directions/distances, the in-shader
     Chebyshev `vis` equals the CPU `SampleProbeMoments`+Chebyshev reference (GPU==CPU bit-exact OR a CPU-unit
     assertion of the closed form); at least pin the closed-form values in the unit test (occluded dir → vis≈0,
     unoccluded → vis==1).
   - **probeCount=0 / occlusion degenerate:** probeCount=0 → no probes → indirect 0 (as DN); all-occluded
     (`wsum<=eps`) → indirect 0 (documented fallback, no NaN from /wsum).
   - **Golden** = frame B → `tests/golden/metal/ddgi_occ.png` (the leak-fixed render); Metal two runs DIFF 0.0000,
     gate on compare.sh EXIT CODE. Print `ddgi-occ: {probes:N, occlusion:1.0}`. Existing 79 image goldens UNTOUCHED
     (incl `ddgi`). **GOLDEN DISCIPLINE: commit ONLY `tests/golden/metal/ddgi_occ.png` — NO loose
     `tests/golden/ddgi_occ.png` (the DK stray trap).**

5. **Determinism / cross-backend.** Fixed scene/grid/strengths; whole chain deterministic (DO/DN proven). Two runs
   byte-identical. The Chebyshev weight is world-anchored probe data (the moment store is cross-backend-identical
   per DO); the screen render is verified cross-backend honestly (if the leak-fixed render is cross-backend-stable
   like the DN golden, great; the occlusionStrength=0==DN byte-identity is per-backend regardless). The `length`/
   `normalize` (sqrt) cross-backend caveat: it appears in the VISIBILITY WEIGHT, which modulates the SH blend —
   the occlusionStrength=0 byte-identity SIDESTEPS it (vis is forced to 1, the sqrt result is unused on the no-op
   path via the DN-path branch). For frame B, the golden is the rendered image baked per-backend; verify two-run
   determinism + cross-backend coherence and report honestly (a sub-LSB cross-backend difference in the leaked
   region, if any, is a display artifact of the sqrt-derived weight, NOT a correctness break — document, don't
   mask; if it breaks DIFF 0.0000 cross-backend, fall back to the DH FP discipline: host-precompute nothing here
   since dir is per-pixel, so instead verify the golden is baked + compared per-backend and the DIFF 0.0000 is the
   two-run-same-backend determinism, which is the committed contract).

6. **Tests `tests/probe_occlusion_test.cpp` (pure CPU, no GPU) — OR extend `probe_dist_test`:**
   - `DistDirToFaceUV`: the 6 axis dirs → the 6 face centres; a dir round-trips to an in-range `(face,u,v)`.
   - `SampleProbeMoments`: a store with a known moment at a texel → the matching dir reads it back.
   - Chebyshev closed form: `dist <= mean` → vis==1; `dist >> mean` with small var → vis≈0; var==0 → hard step.
   - `occlusionStrength=0` identity: the weighted blend with all vis==1 == the unweighted `InterpolateSH` (assert
     the CPU mirror of the no-op path equals DN's `InterpolateSH` byte-identical).
   - all-occluded → zero indirect (no NaN). Determinism. Clean under `windows-msvc-asan`.

7. **Introspect.** Add exactly `ddgi-probe-occlusion` (features) + `--ddgiocc-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses `usesLightClusters` + `BindLightClusters` (binds ProbeSH slot 0 + ProbeDistMoments slot 1, the
  existing 3-buffer fragment-storage set); the GI/occlusion params ride the showcase FrameData UBO; the visibility
  + composite math is in `lit_ddgi_occ.frag.hlsl` + `probe_dist.h` (above-seam, zero backend symbols). rhi.h +
  rhi_factory (dispatch baseline 2) + the backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI)
Bilinear/blurred moment sampling (nearest first), probe relocation, multi-bounce, specular-from-probes, the
occlusion in the DEFAULT --ddgi-shot (keep it the --ddgiocc-shot variant so the `ddgi` golden + all 79 are
untouched), a self-shadowing bias beyond the `dist<=mean` early-out. ONE Chebyshev occlusion-weighted DDGI
composite with an occlusionStrength=0 byte-identical-to-DN proof + the visible leak-fixed golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 79) + the new occlusion test. Clean under
   `windows-msvc-asan`.
2. **no-op proof + VISIBLE leak-fix:** `--ddgiocc-shot` on Vulkan: frame B shows the leak through the occluder
   GONE/attenuated vs frame A (coherent, recognizable); the INTERNAL `occlusionStrength=0` render is
   BYTE-IDENTICAL (SHA) to the DN `lit_ddgi` path, AND frame B != frame A; the visibility-math CPU mirror matches;
   the `ddgi-occ: {...}` line deterministic (two runs byte-identical). Run under the AT Vulkan-validation gate →
   ZERO errors.
3. Metal: `visual_test --ddgiocc` → new golden `tests/golden/metal/ddgi_occ.png`; two runs DIFF 0.0000 (gate on
   the compare.sh EXIT CODE). The occlusionStrength=0==DN proof also passes on Metal.
4. **Render-invariance (CRITICAL):** `git diff master --stat -- tests/golden/metal` shows ONLY `ddgi_occ.png`
   added; the other 79 byte-identical (incl `ddgi` — the sibling-shader isolation keeps DN untouched). `git diff
   master --stat -- tests/golden` shows ONLY `ddgi_occ.png` (metal) + the 2-line introspect json — NO loose
   `tests/golden/ddgi_occ.png`, NO other golden, NO existing `.frag`/pipeline/bind-call changed.
5. Introspect JSON rebaked exactly `+ddgi-probe-occlusion` + `--ddgiocc-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI; the existing `lit_ddgi.frag.hlsl` + `lit.frag.hlsl` + all
   pipelines UNCHANGED). `scripts/verify.ps1` updated to include the new `ddgi_occ` image golden in the Mac
   round-trip loop (gate on compare.sh EXIT CODE).
