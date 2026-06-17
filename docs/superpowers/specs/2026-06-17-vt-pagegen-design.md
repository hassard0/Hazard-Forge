# Slice VT3 — Runtime Virtual Texturing: PROCEDURAL PAGE GENERATION into the PHYSICAL ATLAS (Phase 9 #3) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 3rd RVT slice (after VT1
> feedback + VT2 allocation): GENERATE each allocated virtual page's deterministic procedural CONTENT into its
> physical tile in the atlas — the analog of VSM Slice VB's physical-page render, but writing procedural texture
> color instead of caster depth. Kept INTEGER bit-exact: a compute pass fills the physical atlas (a flat
> `RWStructuredBuffer<uint>` of RGBA8-packed texels) with a per-page pattern keyed by `(pageId, mip, localXY)` via
> pure integer math, proven GPU==CPU BIT-EXACT, with an integer atlas golden that is cross-backend BIT-IDENTICAL.
> NO new RHI. Namespace `hf::render::vt`. Branch: `slice-vt3-vtatlas`. See [[hazard-forge-vt-roadmap]].

**Goal:** Extend `engine/render/vt.h` with the physical-atlas dimensions + the deterministic per-page content
generator (`VtAtlasDims`, `PageTexel(pageId, mip, lx, ly)` → a packed RGBA8 uint, `GeneratePhysicalAtlas` CPU ref) +
`shaders/vt_pagegen.comp.hlsl` (one thread per atlas texel, or one group per allocated tile, generating the same
pattern) + a `--vt-pagegen-shot` (Vulkan) / `--vt-pagegen` (Metal) showcase that takes VT2's indirection table,
generates each allocated page's content into the atlas SSBO, proves it BIT-EXACT vs the CPU reference, and bakes an
integer atlas golden. Make-safe: header additions + a NEW shader + NEW showcase + NEW golden; VT1/VT2 goldens +
everything else UNCHANGED. The atlas bit-identity is guaranteed by the pure-integer per-texel pattern.

## Why integer + why an SSBO atlas (the design crux)
VSM-VB rendered caster DEPTH into a `CreateShadowMap` atlas (a real render target, because it rasterizes geometry).
VT3 GENERATES procedural texture content — there is no geometry to rasterize, so a COMPUTE pass is the natural fit and
keeps the whole slice PURE INTEGER (the RVT arc's signature: cross-backend bit-identical, the strict zero-differing-
pixel posture). The physical atlas is represented as a flat `RWStructuredBuffer<uint>` of `atlasW*atlasH` RGBA8-packed
texels (NOT a sampled texture yet — that is VT4's concern: VT4 will either sample the SSBO-backed atlas by integer
index or blit it to an RGBA8 RT, the SW4 blit precedent). Each texel's color is computed by pure integer ops from
`(pageId, mip, localX, localY)` — NO float, NO lighting — so the atlas is bit-identical CPU↔Vulkan↔Metal. This is the
VB analog (fill the allocated physical tiles, leave the rest cleared) reduced to its byte-exact-provable core.

## The integer core (extends vt.h)
- **`struct VtAtlasDims { int tilesPerSide; int pageSize; }`** (or derive from `VtTilePool` + `VtTexture.pageSize`) →
  `atlasW = atlasH = tilesPerSide * pageSize`, `atlasTexels()`. (e.g. tilesPerSide=12, pageSize=128 → a 1536×1536
  atlas = 2.36M texels = a ~9.4 MB uint SSBO.)
- **`kAtlasClear`** = the cleared-texel sentinel for non-allocated atlas regions (e.g. `0xFF101010` — opaque dark, so
  the golden shows the unused pool as dark; pick a value distinct from any generated texel, document it).
- **`uint32_t PageTexel(int pageId, int mip, int lx, int ly)`** — THE deterministic per-texel generator (pure
  integer): a per-page base color from a hash of `(pageId, mip)` (reuse the engine's `hashColor` / an FNV mix —
  `meshlet.h:79` style, but returning a packed RGBA8) MODULATED by an integer pattern over the page-local `(lx, ly)`
  in `[0, pageSize)` — e.g. a checkerboard (`((lx>>3) ^ (ly>>3)) & 1`) toggling brightness, or a coarse gradient
  (`lx`/`ly` quantized). Pure shifts/xor/add/mul → bit-identical. NO float. The pattern must make adjacent pages and
  adjacent mips visually distinct (so the golden is a legible "each page has its own tinted patterned tile").
- **`GeneratePhysicalAtlas(std::span<const int32_t> indirection, const VtTexture& vt, const VtTilePool& pool,
  std::span<uint32_t> atlasOut)`** — the CPU reference: fill `atlasOut` (sized `atlasTexels()`) with `kAtlasClear`;
  for each pageId with `indirection[pageId] != kNoTile`, compute its physical tile origin (`PhysicalTileOrigin`),
  unpack the pageId → `(mip, px, py)`, and for each local `(lx, ly)` in `[0, pageSize)²` write
  `atlasOut[(oy+ly)*atlasW + (ox+lx)] = PageTexel(pageId, mip, lx, ly)`. This is the EXACT atlas the GPU
  `vt_pagegen.comp` produces.

## Reuse map (file:line)
- **VT1/VT2 (the inputs):** `engine/render/vt.h` — `VtTexture`/`PageId`/`UnpackPageId`/`pageCount` (VT1),
  `VtTilePool`/`PhysicalTileOrigin`/`kNoTile`/`AllocatePhysicalTiles` (VT2). VT3 consumes VT2's indirection table.
- **VSM-VB physical-render precedent:** `engine/render/vsm.h` (`AllocatePhysicalPages` → `PhysicalTileOrigin` → the
  per-tile fill loop) — VT3's `GeneratePhysicalAtlas` mirrors the "for each allocated tile, fill its physical region"
  shape, writing color instead of depth.
- **The compute + atlas-SSBO + readback surface (NO new RHI):** `BufferUsage::Storage`,
  `BindComputePipeline`/`BindStorageBuffer`/`DispatchCompute`, `ReadBuffer` — the VT1/VT2/`vsm_mark` precedent.
- **Color packing / hash:** `meshlet.h:79` `hashColor` (adapt to a packed RGBA8 uint); the swraster/vsm CPU-color
  golden discipline (CPU-color the read-back integer atlas → identical both backends).
- **Showcase + registration:** the VT2 `--vt-alloc-shot`/`--vt-alloc` template; `scripts/verify.ps1`
  `$Goldens`/`$vkShots`, `engine/editor/introspect.cpp`, `tests/introspect_test.cpp`.

## Design decisions (locked)

1. **The atlas is a flat `RWStructuredBuffer<uint>` (RGBA8-packed), generated by a compute pass.** ONE thread per
   atlas texel (`atlasW*atlasH` threads, the simplest race-free mapping): each thread computes its `(ox,oy)` tile +
   local `(lx,ly)`, finds which pageId owns that tile (a reverse map tile→pageId built host-side from the indirection
   and uploaded, OR each thread reads the indirection... — SIMPLEST + race-free: upload a `tilePageId[tileIndex]`
   reverse table (host-built: for each allocated pageId, `tilePageId[indirection[pageId]] = pageId`), so a texel at
   tile `t` writes `PageTexel(tilePageId[t], ...)`; unallocated/out-of-range → `kAtlasClear`). This makes the GPU pass
   a pure parallel map (no sequential dependency, unlike VT2's allocator) → trivially deterministic. ALTERNATIVE: one
   group per allocated tile filling its `pageSize²` texels — also fine; pick the per-texel map for simplicity. NO new
   RHI, NO atomics.
2. **`vt_pagegen.comp.hlsl` (NEW).** Inputs: `gTilePageId` (reverse table, int per tile, `-1`/sentinel for
   unallocated), `gParams{atlasW, atlasH, tilesPerSide, pageSize, mipLevels + the per-mip pagesPerSide/mipOffset
   tables for UnpackPageId}`. Output: `gAtlas` (`RWStructuredBuffer<uint>`). Each thread: its texel `(x,y)` → tile
   `(x/pageSize, y/pageSize)` → `tileIndex` → `pageId = gTilePageId[tileIndex]`; if valid, `UnpackPageId(pageId)` →
   `(mip,px,py)`, local `(lx,ly) = (x%pageSize, y%pageSize)`, write `PageTexel(pageId,mip,lx,ly)` (the verbatim
   integer generator copied from vt.h); else write `kAtlasClear`. `genEnabled=0` → write all `kAtlasClear`. Plain
   integer → NO `--msl-version 20200`. Register in BOTH compile lists.
3. **Showcase `--vt-pagegen-shot <out>` (Vulkan, main.cpp) AND `--vt-pagegen` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm + `#include "render/vt.h"`).** Reuse VT2's full pipeline: VtTexture(4mip,128,16vpps0) + 576
   requests → feedback → `AllocatePhysicalTiles` (VtTilePool tilesPerSide=12 → 144 tiles, 144 allocated / 68 overflow)
   → build the `tilePageId` reverse table (host) → dispatch `vt_pagegen.comp` over the 1536×1536 atlas → `ReadBuffer`
   gAtlas. Golden = the atlas (downscaled if needed for a reasonable golden size, OR the full 1536×1536) directly
   decoded from the RGBA8-packed uints → `tests/golden/metal/vt_atlas.png` (each allocated tile a tinted patterned
   page, unallocated dark — INTEGER → identical both backends by construction). **If 1536² is too large for a golden,
   render the atlas at tilesPerSide=8/pageSize=64 → 512×512, or downscale by nearest-integer decimation in the CPU
   coloring; document the choice. PREFER a 1024×1024 or 512×512 atlas for a manageable golden.**
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU atlas BIT-EXACT:** `memcmp(gpuAtlas, cpuAtlas) == 0` over `atlasTexels()` (integer RGBA8, NO FP
     tol). Print `vt-pagegen GPU==CPU atlas: <atlasW>x<atlasH> BIT-EXACT`.
   - **(2) coverage / correctness:** exactly the `allocated` tiles are non-clear; every allocated tile's region is
     fully written (no `kAtlasClear` hole inside it); unallocated atlas regions are all `kAtlasClear`. Print
     `vt-pagegen: {allocated:<A> tiles, atlas:<W>x<H>, generated:<N> texels}`.
   - **(3) disabled-path no-op:** `genEnabled=false` → atlas all-`kAtlasClear`. Print `vt-pagegen disabled: all clear
     (no-op)`.
   - **(4) determinism:** two dispatches byte-identical.
   - **Golden discipline: ONLY `tests/golden/metal/vt_atlas.png`; do NOT commit it — the CONTROLLER bakes on the
     Mac.** Existing 96 image goldens UNTOUCHED.
5. **Tests `tests/vt_test.cpp` additions (pure CPU):** `PageTexel` determinism + adjacent-page/adjacent-mip produce
   DIFFERENT base colors (distinctness) + the checkerboard/gradient pattern is correct at known `(lx,ly)`;
   `GeneratePhysicalAtlas` — known indirection → known atlas (allocated tiles filled with the right page's pattern,
   unallocated `kAtlasClear`); the reverse `tilePageId` table is a correct inverse of the indirection; `genEnabled`
   off → all clear; determinism. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `runtime-virtual-texturing-pagegen` (features) + `--vt-pagegen-shot` (showcases).

## RHI seam additions (summary)
- **None.** `BufferUsage::Storage` + compute + `ReadBuffer` — the VT1/VT2 precedent. ZERO above-seam backend symbols.
  `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — VT4+)
The material-pass SAMPLE of the atlas through the indirection (VT4 — the first float texel read; this slice only
GENERATES the atlas, no sampling), per-page caching (VT5), real texture assets / streaming / transcoding (procedural
only), mip generation (each mip's pages are independently procedurally generated, no downsampling), an RGBA8 sampled-
texture atlas RT (the SSBO form suffices for VT3's proof; the RT/blit is VT4's concern). ONE compute generator filling
the allocated physical tiles with a deterministic integer pattern + the GPU==CPU bit-exact proof + coverage +
disabled no-op and the integer atlas golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 90) + the new `vt_test` pagegen cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--vt-pagegen-shot` on Vulkan: a coherent physical atlas (each allocated tile a distinct
   tinted patterned page, unallocated dark); `vt-pagegen GPU==CPU atlas: <W>x<H> BIT-EXACT` + the
   `{allocated/atlas/generated}` line + `disabled: all clear` + determinism. Run under the Vulkan-validation gate →
   ZERO errors.
3. Metal: `visual_test --vt-pagegen` → new golden `tests/golden/metal/vt_atlas.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). Integer golden → a strict cross-backend pixel compare must show ZERO differing pixels.
   **Confirm visual_test.mm in the diff; confirm vt_pagegen.comp MSL-generates (plain integer, no MSL-2.2).**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `vt_atlas.png` added; the other 96
   byte-identical. `git diff master --stat -- tests/golden` = ONLY `vt_atlas.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+runtime-virtual-texturing-pagegen` + `--vt-pagegen-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `scripts/verify.ps1` updated: `vt_atlas` golden in the Mac loop
   + `--vt-pagegen-shot` in `$vkShots`.
