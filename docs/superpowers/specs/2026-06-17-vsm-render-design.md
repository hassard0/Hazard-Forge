# Slice VB — Virtual Shadow Maps Slice 2: Physical-Page Depth Rendering (Phase 7 #2) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 2nd VSM slice (after VA
> page-marking): allocate PHYSICAL atlas tiles to the resident virtual pages (an integer virtual→physical
> indirection table, bit-exact) and RENDER each resident page's shadow-casters' depth into its tile via the
> existing CSM `SetViewport` atlas loop, each with that page's clipmap-level ortho sub-frustum. NO new RHI (reuses
> `CreateShadowMap` + `SetViewport` + `BeginShadowPass`). The golden is the colorized depth atlas. Branch:
> slice-vb-vsmrender. See [[hazard-forge-vsm-roadmap]].

**Goal:** Add the physical-page allocator (`vsm.h`) + a `--vsm-render-shot` showcase that, from VA's resident page
set, assigns each resident page a physical atlas tile (deterministic integer indirection table) and renders that
page's casters' depth into its tile, then colorizes the depth atlas as the golden. Make-safe: a NEW header
addition + NEW showcase + NEW golden, reusing the existing shadow-atlas RHI unchanged; NO existing shadow
path/golden touched.

## Reuse map (file:line — from the scout)
- **VA `engine/render/vsm.h`** — `VsmClipmap`, `PageId`/`UnpackPageId`, `SelectClipmapLevel` (threshold-ladder),
  `MarkResidentPages` (the resident set VB allocates from).
- **The CSM atlas + `SetViewport` page-render loop** — `CreateShadowMap(size)` (rhi.h:506), `SetViewport(x,y,w,h)`
  (rhi.h:392), `BeginShadowPass`/`EndShadowPass` (rhi.h:552-555). CSM renders each cascade into a tile via
  `SetViewport(col*kTile, row*kTile, kTile, kTile)` after a single full-atlas clear (main.cpp:1677-1699). VB's
  per-physical-page render IS this loop, one resident page per tile.
- **CSM ortho fit** (`engine/render/csm.h`) — the directional ortho-from-frustum math; VB computes each page's
  clipmap-level ortho (the page's world sub-region at its level, light-space).
- **VA's `ReadBuffer` bit-exact + `hashColor` viz golden discipline** + the DH FP discipline.

## Design decisions (locked)

1. **`engine/render/vsm.h` additions (pure CPU, `hf::render::vsm`).**
   - `struct VsmAtlas { int tilesPerSide; int tileSize; };` — the physical atlas is a `CreateShadowMap(tilesPerSide
     * tileSize)` carved into `tilesPerSide²` tiles. `PhysicalTileOrigin(tileIndex, atlas) -> (x,y)` pixel origin.
   - `int AllocatePhysicalPages(std::span<const uint32_t> resident, const VsmAtlas&, std::span<uint32_t>
     indirectionOut)` — the deterministic allocator: walk `resident[]` in ascending `pageId` order; each `resident
     [pageId]==1` gets the next sequential physical tile index; write `indirectionOut[pageId] = tileIndex` (and a
     sentinel `kNoTile = 0xFFFFFFFF` for non-resident). Returns the allocated count (clamped to `tilesPerSide²` —
     if more resident pages than physical tiles, the overflow pages get `kNoTile`; document the cap + the
     deterministic priority = ascending pageId, i.e. finer/nearer levels first if pageId ordering puts them first
     — confirm the ordering gives a sensible priority, else order by level then pageId; document). Integer,
     deterministic, bit-exact.
   - `PageWorldOrtho(pageId, const VsmClipmap&) -> { math::Vec3 center; float halfExtent; }` — the page's world
     sub-region: level `L`'s page `(px,py)` covers a `level0WorldExtent*2^L / vpps` square centered at the clipmap
     origin offset by the page coords. The light-space ortho for rendering that page's depth.

2. **`--vsm-render-shot <out>` (Vulkan, main.cpp) AND `--vsm-render` (Metal, visual_test.mm — WIRE BOTH; confirm
   visual_test.mm in the diff + `#include render/vsm.h`).** A fixed scene (shadow casters — e.g. the existing
   instance grid / a few boxes) + VA's clipmap + receiver set → `MarkResidentPages` → `AllocatePhysicalPages` → the
   physical atlas (`CreateShadowMap(tilesPerSide*tileSize)`). Render loop (the CSM `SetViewport` pattern): clear the
   atlas once; for each resident page, `SetViewport(physical tile)` + `BeginShadowPass` + render the casters' depth
   with `PageWorldOrtho(pageId)`'s light-space view/proj. PROOFS (fail loudly):
   - **(1) allocation GPU==CPU/deterministic bit-exact:** the indirection table is computed deterministically; CPU
     `AllocatePhysicalPages` over the (read-back) resident set → the indirection table is two-run byte-identical AND
     matches the CPU reference (memcmp). Print `vsm-render allocation: <N> pages -> <N> tiles, indirection
     BIT-EXACT`.
   - **(2) resident==allocated count:** every resident page got a physical tile (or the documented overflow is
     reported). Print `vsm-render: {residentPages:N, physicalTiles:N, atlas:WxH}`.
   - **(3) two-run determinism:** the colorized depth atlas is two-run byte-identical.
   - **(4) disabled-path:** markingEnabled=false → 0 resident → 0 allocated → the atlas is the cleared/empty depth
     (a uniform colorized clear) — byte-identical to the no-page render.
   - **Golden** = the colorized depth atlas: read back the depth atlas (`ReadRenderTarget` the depth, or sample it),
     map each tile's depth → grayscale (near=bright/far=dark within the tile), non-resident/empty tiles a fixed
     clear color → `tests/golden/metal/vsm_atlas.png`. The resident tiles show per-page depth gradients (the casters'
     silhouettes from each page's light view). Metal two runs DIFF 0.0000, gate on compare.sh EXIT CODE. Existing
     88 image goldens UNTOUCHED. **GOLDEN DISCIPLINE: ONLY `tests/golden/metal/vsm_atlas.png`; do NOT commit it —
     the CONTROLLER bakes it on the Mac (NEVER an agent-committed placeholder).**

3. **Determinism / cross-backend.** The allocation is integer/deterministic (bit-exact GPU==CPU + two-run). The
   depth render is a normal depth pass (like CSM/spot — already cross-backend-golden-stable). The colorized atlas
   is baked per-backend (the Metal golden two-run DIFF 0.0000); the Vulkan side is verified via the allocation
   bit-exact proof + the disabled-path no-op + a coherent visual. (Raw depth VALUES may differ sub-LSB
   cross-vendor — so the golden is the Metal bake, NOT a cross-backend depth memcmp; document, like the lit
   goldens.) Apply the DH discipline to the ortho/projection math so the page world-region mapping is stable.

4. **No new RHI.** `CreateShadowMap` (a larger atlas) + `SetViewport` + `BeginShadowPass`/`EndShadowPass` +
   `ReadRenderTarget` (depth readback for the viz) are all existing. The indirection table is an integer SSBO
   (Storage) if computed/consumed on GPU, or plain CPU (read back the resident set, allocate on CPU, upload the
   indirection for the render loop). PREFER CPU allocation (simplest, deterministic, the render loop drives from
   it) — no compute pass needed for VB. rhi.h + rhi_factory (baseline 2) + backend dirs UNCHANGED.

5. **Tests `tests/vsm_test.cpp` additions (pure CPU):**
   - `AllocatePhysicalPages`: a resident set → indirection table with sequential tile indices in pageId order;
     non-resident → `kNoTile`; the count == resident count (under the tile cap); overflow (more resident than
     tiles) → the documented cap + `kNoTile` for the overflow.
   - `PhysicalTileOrigin` round-trips tile index ↔ pixel origin.
   - `PageWorldOrtho`: a page's world center/extent matches the clipmap level math (level 0 small central, higher
     levels larger).
   - Determinism. Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `virtual-shadow-maps-render` (features) + `--vsm-render-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses the shadow-atlas RHI (`CreateShadowMap`/`SetViewport`/`BeginShadowPass`) + `ReadRenderTarget` +
  (optional) Storage SSBO. New non-backend code (`vsm.h` additions, the showcase, the test) adds ZERO above-seam
  backend symbols. rhi.h + rhi_factory (baseline 2) + backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — VC and beyond)
The lit-pass VSM indirection SAMPLE (VC — the shadowed scene render), per-page caching (VD), depth-buffer receiver
reconstruction, sparse eviction / Nanite-cluster per-page culling (the deferred stretch). VB only ALLOCATES +
RENDERS the physical pages + colorizes the atlas; sampling it for shadows is VC. ONE physical-page allocator +
depth render with a bit-exact allocation proof + a markingEnabled=false no-op + the colorized depth-atlas golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 88) + the new `vsm_test` allocation cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--vsm-render-shot` on Vulkan: the colorized depth atlas shows the resident tiles with
   per-page caster-depth gradients (coherent); `vsm-render allocation: ... BIT-EXACT` + `markingEnabled=false →
   empty atlas` + two-run byte-identical; the `vsm-render: {...}` line deterministic. Run under the AT Vulkan-
   validation gate → ZERO errors (the multi-viewport shadow-atlas passes SYNC-HAZARD-free, as CSM).
3. Metal: `visual_test --vsm-render` → new golden `tests/golden/metal/vsm_atlas.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). The allocation bit-exact proof also passes on Metal. **Confirm visual_test.mm in the
   diff; the controller bakes the golden on the Mac (no placeholder).**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `vsm_atlas.png` added; the other
   88 byte-identical. `git diff master --stat -- tests/golden` = ONLY `vsm_atlas.png` (metal) + the 2-line
   introspect json — NO loose `tests/golden/vsm_atlas.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+virtual-shadow-maps-render` + `--vsm-render-shot`; introspect test updated.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new `vsm_atlas`
   image golden in the Mac round-trip loop AND `--vsm-render-shot` in the `$vkShots` validation gate.
