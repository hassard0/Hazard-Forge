# Slice VT5 — Runtime Virtual Texturing: PER-PAGE CACHING across frames (Phase 9 #5) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 5th RVT slice (after VT1
> feedback + VT2 allocation + VT3 page-gen + VT4 sample): skip regenerating physical tiles whose page CONTENT is
> unchanged across frames — the key perf property of a runtime virtual texture (generate/stream a page once, reuse it
> until its source changes). A deterministic FNV content-key per resident page + a per-tile cache; a tile regenerates
> iff its content key changed. Proven byte-identical cached==fresh — the direct analog of VSM Slice VD's
> `PageContentKey`/`VsmPageCache`. Integer bit-exact, cross-backend BIT-IDENTICAL. NO new RHI. Namespace
> `hf::render::vt`. Branch: `slice-vt5-vtcache`. This COMPLETES a 5-slice RVT core. See [[hazard-forge-vt-roadmap]].

**Goal:** Extend `engine/render/vt.h` with the per-page content cache (`PageTexelV` — a content-version-aware
generator, `PageContentKey`, `VtPageCache`, `PageCacheHit`, `PageCacheUpdate`) + a `--vt-cache-shot` (Vulkan) /
`--vt-cache` (Metal) showcase that: Pass1 populates (all allocated tiles generated), Pass2 re-runs the SAME scene (all
cache HITS, 0 regenerated, atlas byte-identical to Pass1), Pass3 bumps ONE page's content version (only that tile
regenerates), proving cached==fresh byte-identical and the invalidation is minimal. Bakes an integer cache-status
golden. Make-safe: header additions + a NEW shader + NEW showcase + NEW golden; VT1–VT4 goldens + everything else
UNCHANGED — in particular VT3's `PageTexel(pageId,mip,lx,ly)` must stay byte-identical (it becomes
`PageTexelV(...,/*contentVersion=*/0)`).

## The integer core (extends vt.h)
- **`uint32_t PageTexelV(int pageId, int mip, int lx, int ly, uint32_t contentVersion)`** — the content-version-aware
  generator: VT3's `PageTexel` math with `contentVersion` folded into the per-page hash seed (so a bumped version
  visibly changes the page's color/pattern). **`PageTexel(pageId,mip,lx,ly)` is redefined to call
  `PageTexelV(pageId,mip,lx,ly,0u)` and MUST produce byte-identical output to the current VT3 `PageTexel` (version 0
  reproduces the existing FNV mix EXACTLY — unit-test + the unchanged VT3/VT4 goldens are the guard).** Pure integer.
- **`uint32_t PageContentKey(int pageId, int mip, uint32_t contentVersion)`** — a deterministic FNV/avalanche hash of
  `(pageId, mip, contentVersion)` (the `vsm.h` `PageContentKey` lineage). Identical inputs → identical key; a bumped
  `contentVersion` → a different key. (The page's content is fully determined by `(pageId, mip, contentVersion)` since
  `PageTexelV` is deterministic — so the key correctly captures "has this page's content changed".)
- **`struct VtPageCache { std::vector<uint32_t> key; std::vector<uint8_t> valid; }`** sized `tileCapacity()` (per
  physical tile). `PageCacheHit(cache, tileIndex, key)` = `valid[tileIndex] && cache.key[tileIndex] == key`;
  `PageCacheUpdate(cache, tileIndex, key)` sets `key[tileIndex]=key; valid[tileIndex]=1`. (Mirror `vsm.h`
  `VsmPageCache`/`PageCacheHit`/`PageCacheUpdate`.)
- **The cached generation rule:** for each allocated tile (pageId→tile from the indirection), `key =
  PageContentKey(pageId, mip, contentVersion[pageId])`; if `PageCacheHit(cache, tile, key)` → SKIP (the tile keeps its
  prior atlas content, byte-identical); else generate `PageTexelV(pageId,mip,lx,ly,contentVersion[pageId])` over its
  region + `PageCacheUpdate`. A skipped tile is the perf win; the atlas SSBO PERSISTS across passes (not re-cleared),
  so a cached tile's texels carry over untouched.

## Reuse map (file:line)
- **VSM-VD cache precedent (copy + rename):** `engine/render/vsm.h` — `PageContentKey` (FNV of page identity + the
  content-determining inputs), `VsmPageCache`, `PageCacheHit`, `PageCacheUpdate`, the FNV helpers (`FnvMix` etc.).
  VT5's cache mirrors these (texture pages; the content-determining input is `contentVersion` instead of casters).
- **VT3 generator:** `engine/render/vt.h` `PageTexel`/`GeneratePhysicalAtlas` — VT5 generalizes `PageTexel` to
  `PageTexelV` (version 0 == VT3) and makes generation cache-aware.
- **VT1/VT2 inputs:** `PageId`/`UnpackPageId`, `AllocatePhysicalTiles`/indirection/`PhysicalTileOrigin`, `BuildTilePageId`.
- **The SHA-equality cache proof discipline:** the VSM-VD `vsm_cache` proof (`all-cached==fresh BYTE-IDENTICAL` via
  atlas SHA, the froxel-density=0 SHA-equality lineage). Compute / readback / golden surface = VT3's (NO new RHI).
- **Showcase + registration:** the VT3 `--vt-pagegen-shot`/`--vt-pagegen` template.

## Design decisions (locked)

1. **The cache decision is host-side; the GPU pagegen skips cached tiles via a per-tile `needsGen` flag.** The host
   compares content keys (`PageCacheHit`) and builds a `needsGen[tileIndex]` flag array (1 = regenerate, 0 = cached).
   `shaders/vt_cachegen.comp.hlsl` (NEW, the VT3 pagegen + a cache guard): one thread per atlas texel — find its tile;
   if `gNeedsGen[tileIndex]==0` (cached, or unallocated) → RETURN without writing (the texel keeps its prior value);
   else write `PageTexelV(pageId,mip,lx,ly,contentVersion)` (the version from `gParams`/a per-tile version table).
   **The atlas SSBO is NOT cleared between passes** (cached tiles persist). NO new RHI, NO atomics. (A pass that
   regenerates everything = all `needsGen=1`; the disabled/empty cache = first pass, all allocated tiles
   `needsGen=1`.)
2. **`vt_cachegen.comp.hlsl` (NEW).** Inputs: `gTilePageId` (reverse table), `gNeedsGen` (per-tile flag),
   `gTileVersion` (per-tile contentVersion), `gParams{atlasW,tilesPerSide,pageSize,mipLevels + the UnpackPageId
   tables}`. Output `gAtlas` (persistent). Plain integer → NO `--msl-version 20200`. Register in BOTH compile lists.
   (VT3's `vt_pagegen.comp` is UNCHANGED; VT5 adds this cache-aware sibling so VT3's golden is untouched.)
3. **Showcase `--vt-cache-shot <out>` (Vulkan, main.cpp) AND `--vt-cache` (Metal, visual_test.mm — WIRE BOTH; confirm
   visual_test.mm + `#include "render/vt.h"`).** Reuse VT3/VT4's scene (SAME VtTexture/requests/VtTilePool/VtAtlasDims).
   Build feedback → `AllocatePhysicalTiles` → `BuildTilePageId`. All pages start `contentVersion=0`. Three passes over
   the PERSISTENT atlas SSBO + a `VtPageCache`:
   - **Pass1 (populate):** empty cache → every allocated tile `needsGen=1` → dispatch → all generated; record
     `atlasSHA1`, `misses1 = allocated`.
   - **Pass2 (all cached):** same versions → every allocated tile is a `PageCacheHit` → `needsGen=0` → dispatch (writes
     nothing) → `ReadBuffer` → `atlasSHA2`. ASSERT `atlasSHA2 == atlasSHA1` (cached==fresh BYTE-IDENTICAL) and
     `misses2 == 0`.
   - **Pass3 (invalidate one):** bump ONE allocated page's `contentVersion` (0→1) → its key changes → that one tile
     `needsGen=1`, the rest cached → dispatch → `ReadBuffer` → `atlasSHA3`. Independently, a FULL-uncached regen of the
     SAME version-state (all `needsGen=1` with the bumped version) → `atlasSHAfull`. ASSERT `atlasSHA3 == atlasSHAfull`
     (the cache is transparent — a partial regen equals a full regen) and `misses3 == 1` (minimal).
   - Golden = a cache-status viz of the final (Pass3) state: each allocated tile shows its content, tinted by cache
     status — **GREEN** wash for cached tiles, **RED** for the 1 regenerated tile, dark for unallocated (the `vsm_cache`
     viz lineage) → `tests/golden/metal/vt_cache.png` (INTEGER → identical both backends by construction).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) all-cached == fresh:** `atlasSHA2 == atlasSHA1` with `misses2==0`. Print `vt-cache all-cached == fresh:
     BYTE-IDENTICAL (<A>/<A> hits)`.
   - **(2) invalidation minimal + transparent:** `misses3==1` AND `atlasSHA3 == atlasSHAfull`. Print `vt-cache
     invalidation: 1 page regenerated, cached==full BYTE-IDENTICAL`.
   - **(3) GPU==CPU:** the Pass3 GPU atlas equals the CPU `GeneratePhysicalAtlas`-with-versions reference
     (`ReadBuffer` memcmp). Print `vt-cache GPU==CPU atlas: <W>x<H> BIT-EXACT`.
   - **(4) determinism:** the 3-pass sequence run twice → identical final atlas. Print `vt-cache determinism: two
     runs BYTE-IDENTICAL`.
   - **Golden discipline: ONLY `tests/golden/metal/vt_cache.png`; do NOT commit it — the CONTROLLER bakes on the
     Mac.** Existing 98 image goldens UNTOUCHED.
5. **Tests `tests/vt_test.cpp` additions (pure CPU):** `PageTexelV(...,0) == PageTexel(...)` byte-identical (the VT3
   guard); `PageContentKey` identity + a bumped version changes it + different pages/mips differ; `PageCacheHit`/
   `PageCacheUpdate` lifecycle (miss → update → hit → version-bump → miss); the cached-generation skip produces the
   same atlas as a full regen (cache transparency, CPU); `needsGen` for a one-page bump has exactly one set bit among
   allocated tiles; determinism. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `runtime-virtual-texturing-cache` (features) + `--vt-cache-shot` (showcases).

## RHI seam additions (summary)
- **None.** Pure host-side cache + the existing Storage-SSBO compute + `ReadBuffer` (the "skip a tile" = a `needsGen`
  guard returning early, leaving the persistent atlas texel untouched). ZERO above-seam backend symbols. `rhi.h` +
  `rhi_factory` (baseline 2) + backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — beyond the RVT core)
LRU eviction (ascending-priority allocation only), a real async upload/streaming budget (the deferred VT6 stretch),
the bilinear/3D-textured float refinement (the deferred float slice), real source-texture editing (the
`contentVersion` bump models it), partial-tile invalidation (whole-page granularity). ONE per-page content cache with
the cached==fresh byte-identical proof + minimal-invalidation transparency + GPU==CPU + determinism and the integer
cache-status golden — completing the 5-slice RVT core.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 90) + the new `vt_test` cache cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--vt-cache-shot` on Vulkan: a coherent cache-status atlas (mostly green cached tiles + 1 red
   regenerated + dark unallocated); `vt-cache all-cached == fresh: BYTE-IDENTICAL` + `invalidation: 1 page
   regenerated, cached==full BYTE-IDENTICAL` + `GPU==CPU atlas BIT-EXACT` + determinism. Run under the
   Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --vt-cache` → new golden `tests/golden/metal/vt_cache.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). Integer golden → a strict cross-backend pixel compare must show ZERO differing pixels.
   **Confirm visual_test.mm in the diff; confirm vt_cachegen.comp MSL-generates (plain integer, no MSL-2.2).**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `vt_cache.png` added; the other 98
   byte-identical (CRITICAL: VT3 `vt_atlas` + VT4 `vt_sample` UNCHANGED — `PageTexelV(...,0)==PageTexel` guarantees the
   VT3/VT4 generators are byte-stable). `git diff master --stat -- tests/golden` = ONLY `vt_cache.png` (metal) + the
   introspect json.
5. Introspect JSON rebaked exactly `+runtime-virtual-texturing-cache` + `--vt-cache-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `scripts/verify.ps1` updated: `vt_cache` golden in the Mac loop
   + `--vt-cache-shot` in `$vkShots`.
