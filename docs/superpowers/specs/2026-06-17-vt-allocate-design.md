# Slice VT2 — Runtime Virtual Texturing: PHYSICAL TILE-POOL ALLOCATION + VIRTUAL→PHYSICAL INDIRECTION TABLE (Phase 9 #2) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 2nd RVT slice (after VT1's
> page-needed feedback): given the resident page SET (VT1's feedback), ALLOCATE physical tiles in a finite tile pool
> and build the VIRTUAL→PHYSICAL indirection table the future sampling pass reads. A deterministic integer allocator
> (ascending-priority, the direct analog of VSM Slice VB's `AllocatePhysicalPages`); the indirection is proven
> GPU==CPU BIT-EXACT and the golden is an INTEGER tile-assignment debug-viz → cross-backend BIT-IDENTICAL. NO new RHI.
> Namespace `hf::render::vt`. Branch: `slice-vt2-vtalloc`. See [[hazard-forge-vt-roadmap]].

**Goal:** Extend `engine/render/vt.h` with the physical tile pool + the deterministic allocator
(`VtTilePool`, `kNoTile`, `PhysicalTileOrigin`, `AllocatePhysicalTiles`, `BuildIndirection`) + a `--vt-alloc-shot`
(Vulkan) / `--vt-alloc` (Metal) showcase that takes VT1's resident feedback set, allocates physical tiles, builds the
virtual→physical indirection table, proves it BIT-EXACT vs the CPU reference, and bakes an integer indirection
debug-viz golden. Make-safe: header additions + a NEW showcase + NEW golden; VT1's `vt_feedback` golden + everything
else UNCHANGED. The indirection bit-identity is guaranteed by the pure-integer ascending-priority allocator.

## Why this is the VSM-VB analog
VSM Slice VB (`AllocatePhysicalPages`, the shipped `vsm_atlas.png` golden) walks the resident page set in ascending
pageId order and assigns each resident page the next sequential physical tile (non-resident/overflow → `kNoTile`),
producing a virtual→physical indirection table proven byte-identical vs its CPU reference. VT2 is the SAME allocator
applied to texture pages: the resident set comes from VT1's feedback instead of VSM's receiver-marking, and the
physical pool is a VT tile pool instead of the shadow atlas. The allocator's ascending-priority determinism (the
make-or-break for GPU==CPU + cross-backend) carries over unchanged. VT2 is STRICTLY an integer slice — no rendering
yet (procedural page CONTENT is VT3); this slice produces only the tile assignment + indirection table.

## The integer core (extends vt.h)
- **`struct VtTilePool { int tilesPerSide; }`** — a finite physical tile pool of `tilesPerSide²` tiles (each holds one
  `pageSize²` page). `tileCapacity() = tilesPerSide²`. (e.g. tilesPerSide=12 → 144 physical tiles for VT1's 340
  virtual pages — deliberately SMALLER than pageCount so the overflow/`kNoTile` path is exercised.)
- **`kNoTile = -1`** (or `0xFFFFFFFF` as the SSBO sentinel) — a virtual page with no physical tile (non-resident OR
  overflow past the pool capacity). Mirror VSM's `kNoTile` (`vsm.h`).
- **`PhysicalTileOrigin(tileIndex, pool, pageSize) -> (ox, oy)`** — the physical-atlas pixel origin of a tile:
  `ox = (tileIndex % tilesPerSide) * pageSize`, `oy = (tileIndex / tilesPerSide) * pageSize`. (Copy the VSM
  `PhysicalTileOrigin` shape.)
- **`AllocatePhysicalTiles(std::span<const uint32_t> feedback, const VtTilePool& pool) -> std::vector<int32_t>
  indirection`** — the deterministic ascending-priority allocator (the `vsm.h:210 AllocatePhysicalPages` analog):
  indirection sized `feedback.size()` (== `pageCount()`), init all `kNoTile`; walk pageId 0..pageCount()-1 ascending,
  for each resident (`feedback[pageId]==1`) page assign `indirection[pageId] = nextTile++` while `nextTile <
  tileCapacity()`, else leave `kNoTile` (overflow). Ascending pageId == mip-major == finest-mip-first priority
  (nearest/most-detailed pages win the pool when it overflows — the correct VT priority, same as VSM's ascending =
  finest-clipmap-first). Returns the indirection table. **Order-independent? NO — allocation is INHERENTLY sequential
  (nextTile depends on prior assignments), so unlike VT1's set-write the GPU allocation needs a deterministic serial
  scan. DECISION: the allocator is computed ON THE HOST (CPU) and the indirection table is UPLOADED for the showcase's
  GPU consumers; the GPU "proof" is that a GPU pass READING the indirection (or a GPU prefix-scan, see below)
  reproduces the CPU table. SIMPLEST + bit-exact: compute on host, and the GPU-side proof is a trivial GPU pass that
  copies/validates the indirection (or VT3+ consumes it). PREFER: a single-thread GPU compute scan that reproduces the
  ascending allocation into an SSBO, memcmp'd vs the CPU `AllocatePhysicalTiles` — proving the GPU allocator ==
  CPU. Choose the single-thread-GPU-scan to keep a genuine GPU==CPU proof; document if instead host-computed.**
- **`BuildIndirection`** convenience = `AllocatePhysicalTiles` (the table IS the indirection).

## Reuse map (file:line)
- **VSM-VB allocator (copy + rename):** `engine/render/vsm.h` — `AllocatePhysicalPages` (the ascending-priority
  sequential allocator), `kNoTile`, `PhysicalTileOrigin`, `VsmAtlas`. VT2's `AllocatePhysicalTiles`/`VtTilePool`/
  `PhysicalTileOrigin` mirror these exactly (texture pages, not shadow pages).
- **VT1 (the input):** `engine/render/vt.h` — `VtTexture`, `PageId`/`UnpackPageId`, `pageCount()`, the `gFeedback`
  resident set. VT2 consumes the feedback set VT1 produces.
- **The single-thread GPU scan precedent (if used):** a 1-thread compute dispatch over the feedback SSBO writing the
  indirection SSBO — the same compute+SSBO+ReadBuffer surface as `vt_feedback.comp` (NO new RHI). Or the
  prefix-sum/scan shape if a parallel allocator is wanted (DEFER parallel; single-thread is fine at this scale).
- **Showcase + golden + registration:** the VT1 `--vt-feedback-shot`/`--vt-feedback` showcase (main.cpp +
  visual_test.mm) is the template; the VSM `--vsm-render-shot` (`vsm_atlas.png`) is the indirection-viz precedent.
  `meshlet.h:79` `hashColor`. `scripts/verify.ps1` `$Goldens`/`$vkShots`, `engine/editor/introspect.cpp`,
  `tests/introspect_test.cpp`.

## Design decisions (locked)

1. **The allocator is a deterministic ascending-priority sequential scan.** Computed by a single-thread GPU compute
   pass (`shaders/vt_alloc.comp.hlsl`, `[numthreads(1,1,1)]`, one thread walks pageId 0..pageCount()-1, maintains
   `nextTile`, writes `indirection[pageId]`) AND mirrored by the CPU `AllocatePhysicalTiles`. The GPU==CPU memcmp is
   the proof the GPU allocator reproduces the CPU table. (A single thread is correct at this scale — pageCount ~340;
   a parallel prefix-scan allocator is a DEFERRED optimization, not needed.) `feedback` (b0) → `indirection` (b1) →
   `gParams{pageCount, tileCapacity, allocEnabled}` (b2). NO new RHI, NO atomics. `allocEnabled=0` → indirection
   stays all-`kNoTile` (disabled no-op).
2. **`vt_alloc.comp.hlsl` (NEW).** The single-thread allocator above; `kNoTile = 0xFFFFFFFF` as the SSBO sentinel
   (cast to int32 `-1` on read-back). Plain integer → default MSL gen (no `--msl-version 20200`). Register in BOTH the
   Vulkan compile list + the Metal `hf_gen_msl` list.
3. **Showcase `--vt-alloc-shot <out>` (Vulkan, main.cpp) AND `--vt-alloc` (Metal, visual_test.mm — WIRE BOTH; confirm
   visual_test.mm in the diff + `#include "render/vt.h"`).** Reuse VT1's scene: VtTexture(4mip,128,16vpps0) + the SAME
   576 fixed requests → mark feedback (CPU or reuse the VT1 marking) → dispatch `vt_alloc.comp` over the feedback set
   with `VtTilePool{tilesPerSide=12}` (144 tiles < 212 resident → overflow exercised) → `ReadBuffer` indirection.
   Golden = the indirection table CPU-colored as a per-mip page-grid debug-viz: each page colored by its assigned
   physical-tile index (`hashColor(tileIndex)` for allocated, a distinct DIM color for `kNoTile`-resident-overflow,
   dark for non-resident) → `tests/golden/metal/vt_alloc.png` (INTEGER → identical both backends by construction).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU indirection BIT-EXACT:** `memcmp(gpuIndirection, cpuIndirection) == 0` over `pageCount()` entries
     (integer, NO FP tol). Print `vt-alloc GPU==CPU indirection: <pageCount> entries BIT-EXACT`.
   - **(2) allocation correctness:** `allocated == min(resident, tileCapacity)`; every allocated tile index is unique
     and in `[0, tileCapacity)`; every resident page within capacity got a tile, every overflow resident page is
     `kNoTile`; ascending-priority (the first `tileCapacity` resident pages by pageId are the allocated ones). Print
     `vt-alloc: {resident:<R>, capacity:<C>, allocated:<A>, overflow:<O>}` with `A=min(R,C)`, `O=max(0,R-C)`.
   - **(3) disabled-path no-op:** `allocEnabled=false` → indirection all-`kNoTile`. Print `vt-alloc disabled: all
     kNoTile (no-op)`.
   - **(4) determinism:** two dispatches byte-identical.
   - **Golden discipline: ONLY `tests/golden/metal/vt_alloc.png`; do NOT commit it — the CONTROLLER bakes on the
     Mac.** Existing 95 image goldens UNTOUCHED.
5. **Tests `tests/vt_test.cpp` additions (pure CPU):** `AllocatePhysicalTiles` — known feedback set → known
   indirection (hand-verify the first few ascending assignments); overflow (capacity < resident → exactly `capacity`
   allocated, the rest `kNoTile`, ascending wins); capacity ≥ resident → all resident allocated; `allocEnabled`-off →
   all `kNoTile`; tile indices unique + in range; `PhysicalTileOrigin` corners; determinism. Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add exactly `runtime-virtual-texturing-allocate` (features) + `--vt-alloc-shot` (showcases).

## RHI seam additions (summary)
- **None.** `BufferUsage::Storage` + `BindComputePipeline`/`BindStorageBuffer`/`DispatchCompute` + `ReadBuffer` — the
  VT1/vsm_mark precedent. New non-backend code adds ZERO above-seam backend symbols. `rhi.h` + `rhi_factory`
  (baseline 2) + backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — VT3+)
Procedural page CONTENT generation into the atlas (VT3 — this slice produces only the tile ASSIGNMENT + indirection,
no pixels), the material-pass sample (VT4), per-page caching (VT5), LRU eviction (ascending-priority only — no
temporal eviction), a parallel prefix-scan allocator (single-thread is fine at this scale), real streaming. ONE
deterministic allocator + indirection table with the GPU==CPU bit-exact proof + overflow correctness + disabled no-op
and the integer indirection golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 90) + the new `vt_test` allocator cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--vt-alloc-shot` on Vulkan: a coherent per-mip indirection debug-viz (allocated pages
   tile-colored, overflow-resident a distinct dim color, non-resident dark); `vt-alloc GPU==CPU indirection: <N>
   entries BIT-EXACT` + the `{resident/capacity/allocated/overflow}` line + `disabled: all kNoTile` + determinism. Run
   under the Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --vt-alloc` → new golden `tests/golden/metal/vt_alloc.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). Integer golden → a strict cross-backend pixel compare must show ZERO differing pixels.
   **Confirm visual_test.mm in the diff; confirm vt_alloc.comp MSL-generates (plain integer, no MSL-2.2).**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `vt_alloc.png` added; the other 95
   byte-identical. `git diff master --stat -- tests/golden` = ONLY `vt_alloc.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+runtime-virtual-texturing-allocate` + `--vt-alloc-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `scripts/verify.ps1` updated: `vt_alloc` golden in the Mac loop
   + `--vt-alloc-shot` in `$vkShots`.
