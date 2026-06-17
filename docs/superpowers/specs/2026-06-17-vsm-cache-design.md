# Slice VD — Virtual Shadow Maps Slice 4: Per-Page Caching (Phase 7 #4) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The FINAL VSM slice (after VA
> marking, VB physical-page render, VC lit sample): per-page CACHING — skip re-rendering physical pages whose
> content (clipmap level + the casters affecting them) is unchanged, the key performance property of UE5's VSM. A
> pure optimization, proven SAFE by a **cached-atlas == uncached-atlas BYTE-IDENTICAL** SHA-equality (an unchanged
> page's tile is never touched, so it stays byte-identical), plus a cache-hit stat and an invalidation case. NO new
> RHI. Completes the VSM 4-slice arc. Branch: slice-vd-vsmcache. See [[hazard-forge-vsm-roadmap]].

**Goal:** Add a per-page content key + cache to `vsm.h` and a `--vsm-cache-shot` showcase that renders the VSM
physical atlas twice (populate → re-render with caching), proves the cached atlas is byte-identical to the uncached
atlas, reports the cache-hit count, and demonstrates an invalidation (move one caster → only its page re-renders).
Make-safe by construction: a NEW header addition + NEW showcase + NEW golden; the cache only SKIPS work that would
produce identical bytes, so it cannot change any pixel.

## Reuse map (file:line)
- **VA/VB/VC `engine/render/vsm.h`** — `VsmClipmap`, `MarkResidentPages`, `AllocatePhysicalPages` + the indirection
  table, `PageWorldOrtho` (the per-page caster render), `PhysicalTileOrigin`. VD adds a content key + cache over
  these.
- **The VB `--vsm-render-shot` physical-page render loop** (main.cpp + Metal) — VD wraps each page's
  `SetViewport`+depth-render in a cache check (skip if the page's content key is unchanged).
- **The froxel `density=0` SHA-equality proof pattern** (froxel.h:24-31) — the "an optimization that must be
  byte-identical to the unoptimized path" discipline; VD's `cached == uncached` is the same shape.
- **VA/VB integer-deterministic + `ReadBuffer`/SHA bit-exact discipline.**

## Design decisions (locked)

1. **`engine/render/vsm.h` additions (pure CPU, `hf::render::vsm`).**
   - `uint64_t PageContentKey(int pageId, const VsmClipmap&, std::span<const CasterRef> casters)` — a deterministic
     hash of everything that determines a page's rendered depth: its clipmap level + world region (`PageWorldOrtho`)
     + the set of casters whose bounds overlap that page's region (each caster's transform + mesh id). Integer/
     bit-exact (a documented fixed hash — e.g. FNV over the quantized caster transforms + the page coords); two
     identical scenes → identical keys. (`CasterRef` = a small POD {meshId, modelMatrix or its quantized hash,
     boundsCenter/Radius} — define it minimally; reuse the showcase's caster list.)
   - `struct VsmPageCache { std::vector<uint64_t> key; std::vector<uint8_t> valid; };` — per virtual page (sized
     `clipmap.pageCount()`), the last content key + a valid bit. `bool PageCacheHit(int pageId, uint64_t newKey,
     const VsmPageCache&)` = `valid[pageId] && key[pageId] == newKey`. `void PageCacheUpdate(int pageId, uint64_t
     newKey, VsmPageCache&)`.
   - Document: a page is re-rendered iff `!PageCacheHit` (new/changed content); a hit SKIPS the render and keeps the
     existing tile depth (which is already correct → byte-identical).

2. **`--vsm-cache-shot <out>` (Vulkan, main.cpp) AND `--vsm-cache` (Metal, visual_test.mm — WIRE BOTH; confirm
   visual_test.mm in the diff + `#include render/vsm.h`).** Reuse the VB caster scene + clipmap. The flow:
   - **Pass 1 (populate):** clear the atlas, render ALL resident pages (cache empty → all misses), populate the
     cache. Read back the atlas → `atlasA`.
   - **Pass 2 (cached, same scene):** render with the cache: every page is a HIT → 0 re-renders → the atlas is
     untouched. Read back → `atlasB`. **PROOF (make-or-break):** `atlasB == atlasA` BYTE-IDENTICAL (the cache
     changed nothing), AND `cacheHits == residentPages` (every page cached). Print `vsm-cache all-cached == fresh:
     BYTE-IDENTICAL (<N>/<N> hits)`.
   - **Pass 3 (invalidation):** move ONE caster (deterministically) → its overlapping page(s) get a new content key
     → only those pages re-render (cache misses), the rest stay cached. Read back → `atlasC`. ALSO render the SAME
     moved scene fully-uncached (force all re-render) → `atlasC_full`. **PROOF:** `atlasC == atlasC_full`
     BYTE-IDENTICAL (the partial cached re-render produces the same atlas as the full re-render — the cache is
     transparent), AND `cacheMisses == <the expected small number of affected pages>` (a measurable, not-all
     invalidation). Print `vsm-cache invalidation: <missed> pages re-rendered, cached==full BYTE-IDENTICAL`.
   - **Determinism:** two runs byte-identical.
   - **Golden** = a cache-status viz: the depth atlas after Pass 3, with resident tiles tinted by cache state
     (CACHED = a green tint over the depth, RE-RENDERED/missed = a red tint), CPU-colored from the integer cache
     state → identical both backends by construction → `tests/golden/metal/vsm_cache.png`. Metal two runs DIFF
     0.0000, gate on compare.sh EXIT CODE. Print `vsm-cache: {residentPages:N, hits:H, misses:M}`. Existing 90
     image goldens UNTOUCHED. **GOLDEN DISCIPLINE: ONLY `tests/golden/metal/vsm_cache.png`; do NOT commit it — the
     CONTROLLER bakes it on the Mac.**

3. **NO new RHI.** The cache is pure host-side logic; "skip a page render" = not issuing that page's
   `SetViewport`+draw. Everything else (atlas, readback) is VB's existing surface. rhi.h + rhi_factory (baseline 2)
   + backend dirs UNCHANGED. ZERO above-seam backend symbols.

4. **Determinism / cross-backend.** The content key + cache validity are integer/deterministic → bit-exact + cross-
   backend. The byte-identical proofs (`cached==fresh`, `cached==full`) are per-backend SHA equalities — they hold
   on each backend because the cache only skips identical work. The viz is CPU-colored from the integer cache state.

5. **Tests `tests/vsm_test.cpp` additions (pure CPU):**
   - `PageContentKey`: identical scene → identical key; a moved caster → a changed key for the overlapping page(s)
     only (unaffected pages keep their key).
   - `PageCacheHit`/`Update`: empty cache → miss; after update → hit for the same key; changed key → miss.
   - The set of pages a caster move invalidates == the pages whose region the caster's bounds overlap (the
     invalidation is correct + minimal).
   - Determinism. Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `virtual-shadow-maps-cache` (features) + `--vsm-cache-shot` (showcases).

## RHI seam additions (summary)
- **None.** Pure host-side cache over VB's existing physical-page render + readback. New non-backend code (`vsm.h`
  additions, the showcase, the test) adds ZERO above-seam backend symbols. rhi.h + rhi_factory (baseline 2) +
  backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI)
True multi-frame temporal caching across an animation timeline (VD proves the cache within one deterministic
showcase: fresh → all-cached → invalidate), GPU-side cache management / dynamic eviction under memory pressure,
Nanite-cluster-driven per-page cluster culling, the sparse-virtual-texture page residency manager. ONE per-page
content-key cache with a cached==uncached byte-identical proof + a cache-hit stat + a minimal-invalidation proof +
the cache-status viz golden. **This completes the VSM 4-slice arc.**

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 88) + the new `vsm_test` cache cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--vsm-cache-shot` on Vulkan: `vsm-cache all-cached == fresh: BYTE-IDENTICAL (N/N hits)` +
   `vsm-cache invalidation: <M> pages re-rendered, cached==full BYTE-IDENTICAL` + two-run determinism; the cache-
   status viz shows mostly-green (cached) tiles with the invalidated page(s) red — coherent; the `vsm-cache: {...}`
   line deterministic. Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --vsm-cache` → new golden `tests/golden/metal/vsm_cache.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). The byte-identical cache proofs also pass on Metal. **Confirm visual_test.mm in the diff;
   the controller bakes the golden on the Mac.**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `vsm_cache.png` added; the other
   90 byte-identical. `git diff master --stat -- tests/golden` = ONLY `vsm_cache.png` (metal) + the 2-line
   introspect json — NO loose `tests/golden/vsm_cache.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+virtual-shadow-maps-cache` + `--vsm-cache-shot`; introspect test updated.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new `vsm_cache` image
   golden in the Mac round-trip loop AND `--vsm-cache-shot` in the `$vkShots` validation gate.
