# Slice WH2 — Warm-Started Hull Contacts: THE KEYED MANIFOLD + PERSISTENT CACHE (the structural store) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of
> FLAGSHIP #26 (WARM-STARTED HULL CONTACTS + ROBUST DETERMINISTIC STACKING, `hf::sim::warmhull`). WH1 built the
> deterministic hull contact feature ID (the int32 MSL-native key that names a contact across frames). WH2 builds
> the PERSISTENT STORE that key unlocks: a per-frame cache that matches THIS tick's manifold points to LAST tick's
> ACCUMULATED contact impulses by key — so next tick's solver (WH3) can warm-start each point from the impulse it
> carried last frame instead of re-deriving a fresh, slightly-inconsistent impulse. This is the structural
> backbone of warm-starting (no new physics yet — the PS2 step of the persist.h template). APPEND to
> `engine/sim/warmhull.h` (WH1 + manifold/gjk/persist/convex/fric/broad/ccd/fpx BYTE-FROZEN). Branch: `slice-wh2`.
> See [[hazard-forge-warmhull-roadmap]], [[hazard-forge-manifold-roadmap]], [[hazard-forge-docs-style]],
> [[hazard-forge-metal-showcase-gate]], [[hazard-forge-gpu-tdr-chunking]].

**Goal:** Extend `engine/sim/warmhull.h` (additive — WH1 byte-unchanged) with `KeyedHullManifold` (a
`convex::ContactManifold` + parallel `HullContactKey keys[4]` + `fx normalImpulse[4]`), `BuildKeyedHullManifold`
(run `manifold::HullContactMulti`, tag each of its 1-4 points with its WH1 key), `HullCache` /
`MatchHullCache` (match this tick's keyed points to last tick's accumulated impulses by key → inherit if present,
cold-start at 0 if absent) / `UpdateHullCache` (store this tick's impulses, evict keys not seen), and
`HullCacheMeasure` / `MeasureHullCache`. Add `shaders/warmhull_cache.comp.hlsl` (int64 — caches the int64 manifold).
Add the showcase `--wh2-cache-shot <out>` (Vulkan: GPU `warmhull_cache.comp` → memcmp vs CPU) / `--wh2-cache`
(Metal: CPU reference — byte-identical). Bake the integer golden `wh2_cache`. **NO new render RHI**; ONE new compute
shader (int64 Vulkan-only).

## Design call: the persist.h PS2 cache, generalized from box `FrictionPoint` to the keyed hull manifold

persist.h's `PersistentCache` (persist.h:146-159) matches a box friction manifold's points to last tick's
accumulated `fric::FrictionPoint` impulses by `ContactKey`. WH2 is the hull analog — the cache stores
`(HullContactKey → accumulated normalImpulse)` and matches against this tick's `KeyedHullManifold`. The control flow
mirrors `persist::MatchCache`/`UpdateCache` (persist.h:221-248) but over the WH1 hull key + the multi-point hull
manifold (persist's `BuildKeyedManifold` is box-only — WH2 writes the hull version new, reusing
`manifold::HullContactMulti` for the manifold and WH1's `BuildHullContactKeys` for the keys).
- **`KeyedHullManifold`** = the frozen `convex::ContactManifold` (positions/depths/normal/count, manifold.h:292) +
  `HullContactKey keys[count]` (WH1) + `fx normalImpulse[count]` (the per-point accumulated normal impulse, zeroed
  on cold-start). std430-packable (the manifold POD + 4×(4-uint key) + 4×fx).
- **`BuildKeyedHullManifold(bodyA, hullA, bodyB, hullB)`** = `manifold::HullContactMulti` (manifold.h:588) → tag
  each point with `MakeHullContactKey` via the WH1 tagged clip (`BuildHullContactKeys`). normalImpulse starts 0.
- **`HullCache`** = a fixed-size table of `(HullContactKey key, fx normalImpulse)` entries (the persist.h cache
  shape, persist.h:167-178). **`MatchHullCache(cache, keyedManifold)`** → for each manifold point, look up its key
  in the cache; if found, seed `normalImpulse` from the cached value (the WARM inherit); if absent, leave 0 (cold).
  **`UpdateHullCache(cache, keyedManifold)`** → write this tick's `normalImpulse` back under each key; evict cache
  entries whose key was NOT in this tick's manifold (the deterministic store, persist.h:221-248). FIXED order.
- The `matched + coldStarted == count` invariant (every point is either warm-matched or cold-started) is the
  cache's correctness proof, exactly as persist.h asserts for boxes.

> NOTE: WH2 is the STRUCTURAL store — it does NOT yet warm-start a solve (that is WH3). The impulse values it
> caches are still produced by a plain (non-accumulated) solve or seeded as a test fixture; WH2 proves the cache
> MATCHES + INHERITS + EVICTS deterministically. No new physics, like PS2.

## Reuse map (file:line)
- **WH1 `engine/sim/warmhull.h` (APPEND after WH1):** `HullContactKey`, `MakeHullContactKey`,
  `HullContactKeysEqual`, `HullContactKeyHash`, `ClipFaceAgainstFaceTagged`, `BuildHullContactKeys`. WH1 frozen.
- **manifold.h (read-only):** `manifold::HullContactMulti` (manifold.h:588 — the manifold), `convex::ContactManifold`
  (manifold.h:292 — the POD reused verbatim).
- **persist.h (read-only — the TEMPLATE pattern, NOT linked):** `persist::PersistentCache`/`CachedContact`
  (persist.h:146-178 — the cache shape), `persist::MatchCache`/`UpdateCache` (persist.h:221-248 — the match/inherit/
  evict control flow to mirror), `persist::ContactKeysEqual`. (persist's `BuildKeyedManifold` is box-only — WH2's
  hull version is new.)
- **The int64-cache GPU==CPU precedent:** persist.h PS2's `persist_cache.comp` (the box cache shader) — mirror for
  `warmhull_cache.comp`. The int64 Vulkan-only / Metal-CPU-ref split (the WH-line / hull-shader convention).
- **Registration:** `samples/hello_triangle/CMakeLists.txt` (`shaders/warmhull_cache.comp.hlsl:cs` — Vulkan only,
  int64 → NOT in `hf_gen_msl`), `scripts/verify.ps1` (`wh2_cache` + `--wh2-cache-shot` to `$vkShots`),
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), append to
  `tests/warmhull_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/warmhull.h`** (WH1 byte-frozen): `KeyedHullManifold`, `BuildKeyedHullManifold`,
   `HullCache`, `MatchHullCache`, `UpdateHullCache`, `HullCacheMeasure`, `MeasureHullCache`. **ONE new shader
   `shaders/warmhull_cache.comp.hlsl`** (int64, Vulkan-only — NOT in `hf_gen_msl`; Metal runs the CPU reference).
   **NO new render RHI.** manifold.h/persist.h/ALL other sim headers + ALL OTHER shaders BYTE-UNCHANGED.
2. **Showcase `--wh2-cache-shot <out>` (Vulkan) AND `--wh2-cache` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--wh2-cache` BEFORE reporting DONE).** Build a deterministic TWO-FRAME scenario: frame t
   builds + caches a keyed manifold over the MF2 battery with seeded impulses; frame t+1 (a tiny relative nudge)
   re-builds + matches → the persistent points inherit, a departed contact is evicted, a new contact cold-starts.
   Vulkan: dispatch `warmhull_cache.comp`, read back the matched `KeyedHullManifold`+cache, memcmp vs CPU. Render a
   pure-integer cache-state visualization (the matched/cold/evicted contacts). Golden =
   `tests/golden/metal/wh2_cache.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1) match/inherit:** `wh2-cache: {points:<N>, matched:<M>, cold:<C>} matched+cold==points` — every point is
     warm-matched or cold-started; matched points inherit the prior impulse (assert a matched point's seeded
     `normalImpulse` == the cached value).
   - **(2) evict:** `wh2-cache: departed key evicted (cacheSize:<S>)` — a contact present at frame t but absent at
     t+1 is evicted (its key no longer in the cache).
   - **(3) GPU==CPU:** `wh2-cache: GPU == CPU BIT-EXACT` — the GPU `warmhull_cache.comp` matched manifold+cache
     memcmp-equals the CPU reference (Vulkan).
   - **(4) determinism:** `wh2-cache determinism: two runs BYTE-IDENTICAL`.
   - Golden discipline: ONLY `tests/golden/metal/wh2_cache.png`; do NOT commit it. Existing 228 goldens UNTOUCHED.
4. **Cross-backend bar (int64 → strict on the integer data):** Vulkan GPU==CPU bit-exact; Metal CPU-ref
   byte-identical by construction. The cache/manifold data is strict-zero cross-vendor; the pure-integer render is
   strict-zero (or, if a float render is chosen, in-band — prefer the pure-integer cache-state view).
5. **Tests — APPEND to `tests/warmhull_test.cpp` (pure CPU):** `BuildKeyedHullManifold` tags counts {4,3,2};
   `MatchHullCache` inherits a matched point's impulse + cold-starts an absent one; `matched + cold == count`;
   `UpdateHullCache` evicts a departed key + keeps a persistent one across a sub-LSB nudge (the WH1 key stability
   in action); two-run byte-equal; a sliding contact (WH1 key-change boundary) cold-starts (does NOT inherit a
   wrong impulse — the honest warm-miss). Clean under `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `warmhull-cache` (features) + `--wh2-cache-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **ONE new compute shader, NO new render RHI.** `shaders/warmhull_cache.comp.hlsl` (int64, Vulkan-only — NOT in
  `hf_gen_msl`; Metal CPU-ref). Reuses the existing compute-dispatch + SSBO seam. `engine/sim/warmhull.h`
  APPEND-only (WH1 frozen); manifold.h/persist.h/gjk.h/convex.h + ALL other sim headers + ALL OTHER shaders
  UNCHANGED. Report the seam: warmhull.h APPEND-only + ONE new Vulkan-only shader; NO rhi.h change, NO MSL entry,
  NO frozen-file edit.

## Out of scope (YAGNI — later slices)
The accumulated warm-started SOLVER (WH3 — WH2 only STORES + MATCHES impulses; it does not yet solve with them).
Sleeping islands + the stable stack (WH4). Lockstep (WH5), render capstone (WH6). Hull friction (a separate future
flagship — WH2 caches the NORMAL impulse only). WH2 claims ONLY: a deterministic persistent cache that matches the
keyed hull manifold's points to last tick's accumulated impulses (inherit / cold-start / evict), the same on
CPU/Vulkan/Metal, with the integer golden + the four proofs.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "warmhull|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--wh2-cache-shot` on Vulkan: the 4 proof lines (incl. matched+cold==points + GPU==CPU) +
   exit 0 under the conan validation layer → ZERO VUID. VERIFY the cache-state visualization is coherent.
3. Metal: `visual_test --wh2-cache` → `tests/golden/metal/wh2_cache.png`; two runs DIFF 0.0000. **Confirm
   `--wh2-cache` wired in `visual_test.mm` (grep it) BEFORE the Mac bake; confirm NO `hf_gen_msl` entry (int64 →
   Metal CPU-ref).** Cross-vendor STRICT ZERO on the integer cache data + pure-integer render.
4. **Render-invariance:** ONLY `wh2_cache.png` added; the other 228 byte-identical (+ controller introspect rebake).
5. Introspect: exactly `+warmhull-cache` + `--wh2-cache-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + WH1 warmhull.h code + manifold.h/persist.h/gjk.h/convex.h + ALL other sim headers +
   ALL OTHER shaders byte-unchanged; warmhull.h APPEND-only; exactly ONE new shader `warmhull_cache.comp.hlsl`,
   Vulkan-only). `wh2_cache` in the Mac loop + `--wh2-cache-shot` in `$vkShots`.
