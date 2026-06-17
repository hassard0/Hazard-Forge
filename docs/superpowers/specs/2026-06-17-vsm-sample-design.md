# Slice VC — Virtual Shadow Maps Slice 3: Lit-Pass VSM Indirection Sample (Phase 7 #3) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 3rd VSM slice (after VA
> page-marking + VB physical-page depth render): the LIT PASS samples the VSM — per receiver pixel, compute its
> clipmap level + virtual page, look up the physical tile in the indirection table (VB), and sample that tile's
> depth with tile-clamped PCF → a SHADOWED scene rendered through the virtual shadow map. NO new RHI (the
> indirection SSBO rides the existing fragment-storage path; the VSM atlas binds via the existing shadow/texture
> path). A `vsmEnabled=0` byte-identical-to-unshadowed no-op keeps it make-safe. Branch: slice-vc-vsmsample.
> See [[hazard-forge-vsm-roadmap]].

**Goal:** Add `shaders/lit_vsm.frag.hlsl` (a sibling lit-pass that shadows via the VSM indirection table) + a
`--vsm-sample-shot` showcase: render the VB VSM depth atlas, then render the scene lit + VSM-shadowed (receivers
darkened where occluded). Make-safe by construction: a NEW sibling shader + NEW showcase + NEW golden (the existing
`lit.frag`/`lit_csm.frag` + their goldens untouched), and a `vsmEnabled=0` → shadow factor 1.0 EXACTLY → render
BYTE-IDENTICAL to the unshadowed lit (the make-or-break no-op).

## Reuse map (file:line)
- **VA/VB `engine/render/vsm.h`** — `VsmClipmap`, `SelectClipmapLevel` (the integer threshold-ladder), `MarkPage`
  (world→page), `VsmAtlas`/`PhysicalTileOrigin`, the indirection table (virtual pageId → physical tile from
  `AllocatePhysicalPages`), `PageWorldOrtho` (the per-page light-space ortho the depth was rendered with).
- **VB `--vsm-render-shot`** — produces the VSM depth atlas (depth-as-grayscale color RT). VC renders it first, then
  samples it.
- **`shaders/lit_csm.frag.hlsl:97-113` `SampleCascadePCF`** — remap a `[0,1]` UV into an atlas TILE (`tileOrigin +
  uv*tileScale`) + 3×3 PCF clamped to the tile bounds (no neighbor bleed) + the Metal V-flip behind `HF_MSL_GEN`.
  VC's VSM sample IS this, with the tile origin fetched from the indirection table (instead of a cascade index).
- **The fragment-storage seam** — `usesLightClusters` + `BindLightClusters` (3 fragment-stage SSBOs; DX/DN used it).
  VC binds the indirection table SSBO via this path (slot 0 + dummies). NO new RHI. The VSM atlas binds as a sampled
  texture via the existing shadow-map bind (`SetShadowMap`) OR the existing material/texture bind (whichever accepts
  the VB color atlas — pick the one that's validation-clean, document; prefer reusing `SetShadowMap` if it accepts
  the color RT, else the texture path — NO new RHI either way; if a dedicated `BindShadowPageTable` is genuinely
  needed it is ONE small additive defaulted-no-op like `BindShadowMapCompute`, documented — but try the existing
  path first).
- **DH FP discipline** — the per-pixel level/page selection reuses the VA integer threshold-ladder (bit-exact); the
  light-space transform + PCF uses `fma`/`mad`.

## Design decisions (locked)

1. **NEW `shaders/lit_vsm.frag.hlsl` (sibling of `lit_csm.frag.hlsl`; do NOT edit lit.frag/lit_csm.frag).** Per
   receiver pixel (world pos `wpos`, normal `N`, the direct lighting as in `lit.frag`): compute the VSM shadow
   factor —
   ```
   level = SelectClipmapLevel(length(wpos - gVsm.cameraPos), gVsm);   // VA threshold-ladder, integer
   (px,py) = projectToClipmapPage(wpos, level, gVsm);                 // VA MarkPage projection
   pageId  = PageId(level,px,py);
   tile    = gIndirection[pageId];                                    // VB indirection SSBO
   if (tile == kNoTile) shadow = 1.0;                                 // page not resident -> lit (documented)
   else {
     lightUV, receiverDepth = projectIntoPageLightSpace(wpos, level, gVsm);  // PageWorldOrtho's view/proj
     shadow = SampleVsmTilePCF(gVsmAtlas, tile, gVsmAtlas params, lightUV, receiverDepth);  // tile-clamped 3x3 PCF
   }
   shadow = lerp(1.0, shadow, gVsm.vsmEnabled);   // vsmEnabled=0 -> shadow==1.0 EXACTLY -> unshadowed
   rgb = directLighting * shadow + ambient;       // the shadow modulates the direct term
   ```
   `SelectClipmapLevel`/`PageId`/`projectToClipmapPage`/`PageWorldOrtho`-light-space are copied VERBATIM from
   `vsm.h` (the shared-math rule). `SampleVsmTilePCF` is `SampleCascadePCF` adapted: the tile origin = `Physical
   TileOrigin(tile)` in atlas-pixel space → tile UV, 3×3 PCF clamped to the tile, the depth stored as VB's
   grayscale (reconstruct: `storedDepth = 1 - z`, so compare `receiverDepth` vs the decoded tile depth; document the
   exact encode/decode so the comparison is correct). `[[vk::binding(13,3)]] StructuredBuffer<uint> gIndirection` +
   the VSM params in the showcase FrameData UBO + `gVsmAtlas` as the bound shadow/texture.

2. **NO new RHI.** Bind the indirection SSBO via `usesLightClusters`/`BindLightClusters(*indirBuf,*dummy,*dummy)`
   (the DX/DN single-SSBO-via-dummies idiom). Bind the VSM atlas via `SetShadowMap` (or the existing texture path).
   The VSM params (clipmap levels/vpps/extent/cameraPos + atlas tilesPerSide/tileSize + `vsmEnabled`) ride the
   showcase FrameData UBO. rhi.h + rhi_factory (baseline 2) + backend dirs UNCHANGED (or the ONE documented additive
   defaulted-no-op bind if unavoidable — report). Declare the empty/real shadow pass appropriately (the VSM atlas IS
   the shadow source) so it is validation-clean.

3. **Showcase `--vsm-sample-shot <out>` (Vulkan, main.cpp) AND `--vsm-sample` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm in the diff + `#include render/vsm.h`).** The VB caster scene (boxes/spheres) + a receiver
   ground plane. Pipeline: VB physical-page depth render → VSM depth atlas → bind (atlas + indirection) → render the
   scene with `lit_vsm.frag` → the lit + VSM-shadowed image. PROOFS (fail loudly):
   - **(1) make-or-break no-op:** `vsmEnabled=0` → every shadow factor `lerp(1, ..., 0) == 1.0` EXACTLY → the render
     is BYTE-IDENTICAL (SHA) to the same scene through `lit_vsm.frag` with shadowing off (== the unshadowed lit).
     Print `vsm-sample vsmEnabled=0 == unshadowed: BYTE-IDENTICAL`.
   - **(2) shadows present (frame B):** `vsmEnabled=1` → the receivers show shadows where casters occlude them; the
     render DIFFERS from frame A (a measurable darkened-shadow region). Print `vsm-sample B != A: shadows active`.
   - **(3) page-lookup GPU==CPU bit-exact (interior pixels):** at deterministically-chosen receiver pixels, the
     `(level, pageId, tile)` the shader resolves == the CPU `vsm.h` reference (integer equality, bit-exact — the
     VA/DW interior-pixel discipline). Print `vsm-sample page-lookup GPU==CPU: <K>/<K> EXACT`.
   - **(4) two-run determinism** byte-identical.
   - **Golden** = the VSM-shadowed scene (frame B) → `tests/golden/metal/vsm_shadow.png`. Metal two runs DIFF
     0.0000, gate on compare.sh EXIT CODE. Print `vsm-sample: {residentPages:N, shadowedReceivers:...}`. Existing 89
     image goldens UNTOUCHED. **GOLDEN DISCIPLINE: ONLY `tests/golden/metal/vsm_shadow.png`; do NOT commit it — the
     CONTROLLER bakes it on the Mac (never an agent-committed placeholder).**
   - **(5) [SMOKE, not byte-exact] VSM≈CSM equivalence:** with the clipmap forced to a single level covering the
     camera frustum, the VSM-shadowed image is VISUALLY close to the existing CSM shadow (a near-identical smoke,
     `maxDiff < eps`, explicitly NOT DIFF-0.0000 — the atlas layouts/orthos differ, so byte-identity is not claimed,
     per the scout's honest cross-vendor/equivalence framing). Print `vsm-sample vs CSM (1-level): <maxDiff> < eps`.

4. **Determinism / cross-backend.** The level/page/tile selection is integer (bit-exact GPU==CPU + cross-backend);
   the PCF + light-space transform reuse `lit_csm.frag`'s cross-backend-stable math (the existing csm golden proves
   it) + the Metal V-flip. The shadowed image is baked per-backend (the Metal golden two-run DIFF 0.0000); the
   `vsmEnabled=0` byte-identity + the GPU==CPU page-lookup are the per-backend rigorous proofs. Apply `fma`/`mad`.

5. **Tests `tests/vsm_test.cpp` additions (pure CPU):**
   - The page-lookup chain `wpos → level → pageId → tile` over a known indirection table → expected tiles
     (a CPU mirror of the shader lookup; ties VC to VA/VB).
   - The `vsmEnabled=0` shadow-factor identity (the CPU mirror of `lerp(1,x,0)==1`).
   - A resident page's light-space projection round-trips a known caster/receiver. Determinism. ASan-clean.

6. **Introspect.** Add exactly `virtual-shadow-maps-sample` (features) + `--vsm-sample-shot` (showcases).

## RHI seam additions (summary)
- **None** (target). Reuses `usesLightClusters`/`BindLightClusters` (indirection SSBO), `SetShadowMap`/the texture
  path (VSM atlas), the FrameData UBO (params), and the existing lit-pass pipeline. New non-backend code
  (`lit_vsm.frag.hlsl`, `vsm.h` light-space helper if any, the showcase, the test) adds ZERO above-seam backend
  symbols. rhi.h + rhi_factory (baseline 2) + backend dirs UNCHANGED — or the ONE documented additive defaulted-
  no-op `BindShadowPageTable` if the existing fragment-SSBO path genuinely can't carry it (report which). Report the
  seam.

## Out of scope (YAGNI — VD and beyond)
Per-page caching (VD), depth-buffer receiver reconstruction for the marking (VC's receiver is the rendered scene's
geometry), sparse eviction / Nanite-cluster per-page culling, soft-shadow / contact-hardening filtering beyond the
3×3 PCF. ONE lit-pass VSM indirection sample with a vsmEnabled=0 byte-identical no-op + a GPU==CPU page-lookup proof
+ the shadowed-scene golden + a VSM≈CSM smoke.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 89) + the new `vsm_test` lookup cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--vsm-sample-shot` on Vulkan: the scene renders lit + VSM-shadowed (receivers shadowed
   under casters — coherent, recognizable shadows); `vsm-sample vsmEnabled=0 == unshadowed: BYTE-IDENTICAL` + `B !=
   A` + `page-lookup GPU==CPU EXACT` + two-run byte-identical + the CSM smoke; the `vsm-sample: {...}` line
   deterministic. Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --vsm-sample` → new golden `tests/golden/metal/vsm_shadow.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). The vsmEnabled=0 + GPU==CPU proofs also pass. **Confirm visual_test.mm in the diff;
   confirm the Metal build compiles + lit_vsm.frag MSL-generates; the controller bakes the golden on the Mac.**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `vsm_shadow.png` added; the other
   89 byte-identical (incl `csm`/`scene_shadow` — the sibling shader keeps the existing lit/shadow goldens
   untouched). `git diff master --stat -- tests/golden` = ONLY `vsm_shadow.png` (metal) + the 2-line introspect json
   — NO loose `tests/golden/vsm_shadow.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+virtual-shadow-maps-sample` + `--vsm-sample-shot`; introspect test updated.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI, or the one documented additive bind). `scripts/verify.ps1` updated
   to include the new `vsm_shadow` image golden in the Mac round-trip loop AND `--vsm-sample-shot` in the `$vkShots`
   validation gate.
