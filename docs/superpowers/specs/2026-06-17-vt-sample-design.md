# Slice VT4 — Runtime Virtual Texturing: MATERIAL-PASS SAMPLE through the INDIRECTION (Phase 9 #4) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 4th RVT slice (after VT1
> feedback + VT2 allocation + VT3 page-gen): SAMPLE the virtual texture through the virtual→physical INDIRECTION —
> the round-trip that makes RVT a usable texture. A compute pass reconstructs the virtual texture's mip-0 image in
> VIRTUAL UV space by, per virtual texel, computing its pageId → looking up `indirection[pageId]` → offsetting into
> the physical atlas → reading the texel (NEAREST). DESIGN CALL: kept INTEGER nearest-sample (NOT bilinear), so the
> sample path stays bit-exact and the golden is cross-backend BIT-IDENTICAL (the zero-differing-pixel posture, like
> VT1–VT3) — bilinear filtering + perspective-mapped geometry are an explicit DEFERRED float refinement (VT5+). This
> proves the indirection round-trip is correct before adding the float filter. NO new RHI. Namespace `hf::render::vt`.
> Branch: `slice-vt4-vtsample`. See [[hazard-forge-vt-roadmap]].

**Goal:** Extend `engine/render/vt.h` with the sample-through-indirection round-trip (`kVtMiss`,
`SampleVirtualTexel(virtualTexelX, virtualTexelY, mip, indirection, atlas, vt, pool, atlasDims)` → a packed RGBA8
uint, `ReconstructVirtualImage` CPU ref) + `shaders/vt_sample.comp.hlsl` (one thread per virtual texel, the same
lookup) + a `--vt-sample-shot` (Vulkan) / `--vt-sample` (Metal) showcase that reconstructs the virtual texture's mip-0
image from VT3's atlas via VT2's indirection, proves it BIT-EXACT vs the CPU reference, and bakes an integer golden.
Make-safe: header additions + a NEW shader + NEW showcase + NEW golden; VT1/VT2/VT3 goldens + everything else
UNCHANGED. The sample bit-identity is guaranteed by the pure-integer page-lookup + nearest texel read.

## Why integer nearest (the design call, stated honestly)
A virtual-texturing system's payoff is sampling a textured surface through the page indirection. The two ways to do it:
(a) **NEAREST** integer texel read — the sample value IS an atlas texel, no filtering → pure integer → cross-backend
BIT-IDENTICAL; (b) **BILINEAR** 4-texel lerp — float weights → cross-vendor ±1 (the visresolve-bar). VT4 chooses (a)
to PROVE the indirection round-trip (virtual UV → page → physical tile → texel) with a strict zero-differing-pixel
golden, preserving the RVT arc's bit-exact signature. Bilinear filtering, perspective-correct UV on real geometry, and
trilinear mip blending are the DEFERRED float refinement (a later visresolve-bar slice) — they change the FILTER, not
the round-trip VT4 proves. This is the same discipline VSM used (the byte-exact integer core first, the float sample
as its own slice). The golden reconstructs the virtual texture as the user ADDRESSES it (pages in virtual UV layout),
visually DISTINCT from VT3's physical-atlas golden (pages in tile-allocation order) — so VT4 is a genuinely new,
indirection-exercising slice, not a re-skin of VT3.

## The integer core (extends vt.h)
- **`kVtMiss`** = the sampled value for a virtual texel whose page is non-resident (`indirection[pageId]==kNoTile`):
  a distinct MISS color (e.g. `0xFF000000` opaque black, or a magenta `0xFFFF00FF` debug tint — pick one that reads
  clearly against the generated pages and is DISTINCT from `kAtlasClear`; document). (A coarser-mip fallback is the
  deferred refinement; VT4 uses a flat miss color.)
- **`uint32_t SampleVirtualTexel(int vx, int vy, int mip, std::span<const int32_t> indirection,
  std::span<const uint32_t> atlas, const VtTexture& vt, const VtTilePool& pool, const VtAtlasDims& atlasDims)`** —
  the round-trip: `px = vx / pageSize; py = vy / pageSize` (the virtual page at this mip); `pageId = PageId(mip,px,py)`;
  `tile = indirection[pageId]`; if `tile == kNoTile` → return `kVtMiss`; else `PhysicalTileOrigin(tile)` → `(ox,oy)`,
  local `(lx,ly) = (vx % pageSize, vy % pageSize)`, return `atlas[(oy+ly)*atlasW + (ox+lx)]` (NEAREST — a direct
  integer texel read). Pure integer; the SAME atlas VT3 generated, the SAME indirection VT2 built → the read-back
  texel equals `PageTexel(pageId,mip,lx,ly)` (a self-consistency the proof can also assert).
- **`ReconstructVirtualImage(int mip, indirection, atlas, vt, pool, atlasDims, std::span<uint32_t> imageOut)`** — CPU
  ref: for each virtual texel `(vx,vy)` in `[0, pagesPerSide(mip)*pageSize)²`, `imageOut[vy*W + vx] =
  SampleVirtualTexel(...)`. The reconstructed virtual-texture image at `mip` (resident pages show their content,
  non-resident pages show `kVtMiss`).

## Reuse map (file:line)
- **VT1/VT2/VT3 (the inputs):** `engine/render/vt.h` — `PageId`/`UnpackPageId`/`pageCount` (VT1),
  `VtTilePool`/`PhysicalTileOrigin`/`kNoTile`/`AllocatePhysicalTiles` (VT2), `VtAtlasDims`/`PageTexel`/`kAtlasClear`/
  `GeneratePhysicalAtlas` (VT3). VT4 consumes VT2's indirection + VT3's atlas.
- **The compute + SSBO + readback surface (NO new RHI):** `BufferUsage::Storage` + compute + `ReadBuffer` — the
  VT1/VT2/VT3 precedent. (The atlas + indirection ride storage SSBOs; the reconstructed image is a storage SSBO read
  back + CPU-decoded, exactly like VT3.)
- **Showcase + registration:** the VT3 `--vt-pagegen-shot`/`--vt-pagegen` template; `scripts/verify.ps1`
  `$Goldens`/`$vkShots`, `engine/editor/introspect.cpp`, `tests/introspect_test.cpp`.

## Design decisions (locked)

1. **The reconstructed image is a flat `RWStructuredBuffer<uint>` (RGBA8), produced by a compute pass.** ONE thread
   per virtual texel of the mip being reconstructed (race-free parallel map). Inputs: `gIndirection` (int per pageId),
   `gAtlas` (the VT3 atlas SSBO), `gParams{mip, virtualW, virtualH, pageSize, tilesPerSide, atlasW + the per-mip
   pagesPerSide/mipOffset tables for PageId}`. Output `gImage` (`RWStructuredBuffer<uint>`). Each thread runs
   `SampleVirtualTexel` (copied VERBATIM from vt.h). `sampleEnabled=0` → write all `kVtMiss` (disabled no-op). NO new
   RHI, NO atomics. **Reconstruct mip 0** (the finest, largest, most legible): with VT3's config (vpps0=16,
   pageSize=64) mip 0 is 16 pages/side → a 1024×1024 virtual image — the resident mip-0 pages show their VT3 content
   at their virtual-UV location, non-resident mip-0 pages show `kVtMiss`.
2. **`vt_sample.comp.hlsl` (NEW).** The per-virtual-texel sampler above; `kNoTile`/`kVtMiss` as the SSBO sentinels.
   Plain integer → NO `--msl-version 20200`. Register in BOTH compile lists.
3. **Showcase `--vt-sample-shot <out>` (Vulkan, main.cpp) AND `--vt-sample` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm + `#include "render/vt.h"`).** Reuse VT3's full pipeline: VtTexture + 576 requests →
   feedback → `AllocatePhysicalTiles` (the SAME VtTilePool/VtAtlasDims as VT3) → `GeneratePhysicalAtlas` (atlas SSBO)
   → upload atlas + indirection → dispatch `vt_sample.comp` reconstructing mip 0 → `ReadBuffer` gImage. Golden = the
   reconstructed virtual image decoded from the RGBA8 uints → `tests/golden/metal/vt_sample.png` (the virtual
   texture's mip-0 in UV layout: resident pages textured, non-resident pages the miss color — INTEGER → identical
   both backends by construction).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU sampled image BIT-EXACT:** `memcmp(gpuImage, cpuImage) == 0` over the virtual-image texels
     (integer, NO FP tol). Print `vt-sample GPU==CPU image: <W>x<H> BIT-EXACT`.
   - **(2) round-trip self-consistency:** every resident virtual texel's sampled value equals
     `PageTexel(pageId,mip,lx,ly)` (the sample read back exactly what VT3 wrote — the indirection is a correct
     virtual→physical map); every non-resident virtual texel is `kVtMiss`. Print `vt-sample: {mip:0, <W>x<H>,
     resident-pages:<R>, miss-pages:<M>, textured-texels:<N>}`.
   - **(3) GPU==CPU page-lookup @interior:** at a few deterministically-chosen interior virtual texels, the
     `(pageId, tile)` the shader resolves equals the CPU `SampleVirtualTexel` lookup (integer, bit-exact) — the
     indirection-sample correctness oracle. Print `vt-sample page-lookup @interior: <k>/<k> EXACT`.
   - **(4) disabled-path no-op:** `sampleEnabled=false` → image all-`kVtMiss`. Print `vt-sample disabled: all miss
     (no-op)`.
   - **(5) determinism:** two dispatches byte-identical.
   - **Golden discipline: ONLY `tests/golden/metal/vt_sample.png`; do NOT commit it — the CONTROLLER bakes on the
     Mac.** Existing 97 image goldens UNTOUCHED.
5. **Tests `tests/vt_test.cpp` additions (pure CPU):** `SampleVirtualTexel` — resident page → the exact atlas texel
   (== `PageTexel`); non-resident page → `kVtMiss`; the (vx,vy)→(page,local) decomposition correct at page boundaries;
   `ReconstructVirtualImage` known indirection+atlas → known image; `sampleEnabled` off modeled → all miss;
   determinism. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `runtime-virtual-texturing-sample` (features) + `--vt-sample-shot` (showcases).

## RHI seam additions (summary)
- **None.** `BufferUsage::Storage` + compute + `ReadBuffer` — the VT1/VT2/VT3 precedent. ZERO above-seam backend
  symbols. `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — VT5+ / the deferred float refinement)
**Bilinear/trilinear texel filtering** (the float refinement — VT4 is NEAREST; a later visresolve-bar slice adds the
4-texel lerp + cross-vendor smoke proof), **perspective-correct UV on real 3D geometry** (VT4 reconstructs the virtual
texture in flat UV space; a textured-mesh render is the deferred money-shot slice), per-page caching (VT5), a
coarser-mip fallback for non-resident pages (flat miss color first), real streaming. ONE integer nearest sample-
through-indirection that reconstructs the virtual texture + the GPU==CPU bit-exact proof + round-trip self-consistency
+ page-lookup oracle + disabled no-op and the integer reconstructed-image golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 90) + the new `vt_test` sample cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--vt-sample-shot` on Vulkan: a coherent reconstructed virtual texture (resident pages
   textured at their UV location, non-resident pages the miss color); `vt-sample GPU==CPU image: <W>x<H> BIT-EXACT` +
   the `{...}` line + `page-lookup @interior` + `disabled: all miss` + determinism. Run under the Vulkan-validation
   gate → ZERO errors.
3. Metal: `visual_test --vt-sample` → new golden `tests/golden/metal/vt_sample.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). Integer golden → a strict cross-backend pixel compare must show ZERO differing pixels.
   **Confirm visual_test.mm in the diff; confirm vt_sample.comp MSL-generates (plain integer, no MSL-2.2).**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `vt_sample.png` added; the other 97
   byte-identical. `git diff master --stat -- tests/golden` = ONLY `vt_sample.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+runtime-virtual-texturing-sample` + `--vt-sample-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `scripts/verify.ps1` updated: `vt_sample` golden in the Mac loop
   + `--vt-sample-shot` in `$vkShots`.
