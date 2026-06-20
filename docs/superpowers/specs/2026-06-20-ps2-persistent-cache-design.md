# Slice PS2 — Deterministic Persistent Contacts: THE PERSISTENT MANIFOLD CACHE — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #21
> (DETERMINISTIC WARM-STARTED CONTACT CACHING + SLEEPING ISLANDS, `hf::sim::persist`). PS1 built the deterministic
> contact feature ID (`ContactKey`). PS2 builds the **persistent cache**: a store that matches THIS tick's friction
> manifold points to LAST tick's accumulated contact impulses by `ContactKey`, so a matched point INHERITS its prior
> normal+tangent impulses (an unmatched point starts at zero, a stale key is evicted) — the structural store the PS3
> warm-started solver reads. INTEGER-bit-exact. int64 (the manifold it caches is built by the int64
> `BuildFrictionPoints`) → the `persist_cache.comp` shader is Vulkan-only + a Metal CPU reference. PS1's `persist.h`
> code + CX/FC's `convex.h`/`fric.h` are BYTE-FROZEN (PS2 is additive). Branch: `slice-ps2`. See
> [[hazard-forge-persist-roadmap]].

**Goal:** Extend `engine/sim/persist.h` (additive — PS1 + fric.h byte-unchanged) with the cache: `CachedContact` (a
key + its persisted impulses) + `PersistentCache` (the store) + `KeyedFrictionManifold` (a `fric::FrictionManifold`
+ a `ContactKey` per point) + `BuildKeyedManifold` (build the manifold + its keys) + `MatchCache` (inherit prior
impulses by key) + `UpdateCache` (store this tick's impulses, evict stale). Add the new int64 shader
`shaders/persist_cache.comp.hlsl` + `--persist-cache-shot` (Vulkan) / `--persist-cache` (Metal). Bake the integer
golden `persist_cache`. **NO new RHI.**

## Design call: match this tick's manifold to last tick's impulses by `ContactKey`

A warm-started solver needs each contact point to carry last tick's accumulated impulses. The cache provides them:
- **`CachedContact { ContactKey key; fx normalImpulse; fx tangentImpulse1; fx tangentImpulse2; }`** — one cached
  contact's key + its three persisted accumulated impulses (the `fric::FrictionPoint` accumulator fields).
- **`PersistentCache { std::vector<CachedContact> entries; }`** — the store: LAST tick's accumulated impulses keyed
  by `ContactKey`. (A flat vector + a fixed linear scan — the contact sets are tiny; PS1's `ContactKeyHash` is
  available for a bucket optimization, but the deterministic baseline is the fixed-order scan.)
- **`KeyedFrictionManifold { fric::FrictionManifold fm; ContactKey keys[4]; }`** — a manifold + the `ContactKey` for
  each of its `fm.count` points. (`fric::FrictionPoint` is byte-frozen and has no key field, so PS2 stores the keys
  in a PARALLEL array — `keys[i]` ↔ `fm.pts[i]`.)
- **`BuildKeyedManifold(bodyAIdx, bodyBIdx, bodyA, boxA, bodyB, boxB)` → KeyedFrictionManifold** — run
  `fric::BuildFrictionPoints(bodyA, boxA, bodyB, boxB)` (FC2, frozen) AND the underlying `convex::BoxSatStable` to
  get the `SatResult`; for each of the `fm.count` points compute `keys[i] = MakeContactKey(bodyAIdx, bodyBIdx, sat,
  i)` (PS1). The fm's accumulators are zero at build (the FC2 contract).
- **`MatchCache(cache, keyed)` → mutates `keyed.fm`** — for each manifold point `i` in FIXED order, scan the cache
  for an entry whose `key` equals `keyed.keys[i]` (`ContactKeysEqual`); if found, copy the cached
  `normalImpulse`/`tangentImpulse1`/`tangentImpulse2` into `keyed.fm.pts[i]` (the warm-start inheritance); if not
  found, leave them at zero (a fresh contact cold-starts). FIXED scan order → deterministic.
- **`UpdateCache(cache, keyed)` → rewrites `cache`** — after a (future PS3) solve, replace the cache with THIS tick's
  contacts: for each point `i`, append `{keyed.keys[i], keyed.fm.pts[i].normalImpulse, ...}`. Keys not present this
  tick are thereby EVICTED (the new cache is exactly this tick's set). FIXED order → deterministic. (PS2 proves the
  match + the evict; the impulses it stores are whatever the manifold carries — in PS2's showcase, synthesized
  values; PS3 wires the real solved impulses.)

**THE int64 REALITY (the FC2/CX1 lesson):** the manifold build (`BoxSatStable`/`BuildManifold`/the basis) is int64;
the key compares are int32. So `persist_cache.comp` is **VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl)** — it recomputes
the keyed manifold (int64) + matches the cache (passed as an SSBO) per pair; the Metal `--persist-cache` runs the CPU
`BuildKeyedManifold`+`MatchCache` — byte-identical to the Vulkan GPU result BY CONSTRUCTION (the convex_manifold.comp
convention), while the Vulkan side carries the GPU==CPU memcmp proof.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **PS1 `engine/sim/persist.h` (read it; APPEND only after `MeasureKeys`):** `ContactKey`, `MakeContactKey`,
  `ContactKeysEqual`, `ContactKeyHash`. PS1 lines byte-frozen.
- **fric.h (read-only — do NOT edit):** `fric::FrictionManifold`/`FrictionPoint` (`fric.h:160-176`, the
  `normalImpulse`/`tangentImpulse1`/`tangentImpulse2` accumulators), `fric::BuildFrictionPoints` (`fric.h:190`).
- **convex.h (read-only):** `convex::BoxSatStable` (`convex.h:807`, the `SatResult` PS2's key needs),
  `convex::ContactManifold`, `FxBox`, `SatPair`, `FxBody`. **DO NOT modify fric.h/convex.h/fpx.h.**
- **The shader + showcase precedent:** CX2's `shaders/convex_manifold.comp.hlsl` (the int64 Vulkan-only manifold-in-
  a-shader pattern) + FC2's `shaders/fric_points.comp.hlsl` (the FrictionManifold pack) — `persist_cache.comp`
  recomputes the keyed manifold the same way + adds the cache match (a small SSBO scan). The `--fric-points-shot` /
  `--convex-manifold-shot` Vulkan showcases + the Metal blocks. Mirror. Confirm `persist_cache` NOT in hf_gen_msl.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/persist_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/persist.h`** (PS1 byte-frozen): `CachedContact`, `PersistentCache`,
   `KeyedFrictionManifold`, `BuildKeyedManifold`, `MatchCache`, `UpdateCache`, a `MeasureCache` (matched/unmatched/
   evicted counts). Pure integer, FIXED scan + store order. **NEW shader** `persist_cache.comp.hlsl` (int64,
   Vulkan-only, one thread per pair — recomputes the keyed manifold + matches the cache VERBATIM). NOT in hf_gen_msl;
   Metal runs the CPU path.
2. **Showcase `--persist-cache-shot <out>` (Vulkan) AND `--persist-cache` (Metal) — WIRE BOTH** (standalone
   arg-parse). The SCENE — a synthesized TWO-TICK match over the CX2/FC2 box-pair array: TICK 1 builds the keyed
   manifolds and seeds the cache with deterministic synthesized per-contact impulses (e.g. a fixed function of the
   key hash) via `UpdateCache`; TICK 2 rebuilds the keyed manifolds (the SAME scene → the SAME keys) and
   `MatchCache` → every point INHERITS its tick-1 impulse; PLUS a CHANGED pair (one box nudged so its SAT axis /
   clip corner differs → a different key) whose points COLD-START at zero (the cache miss); PLUS a REMOVED pair
   (separated tick 2) whose tick-1 keys are EVICTED. Vulkan: the GPU `persist_cache.comp` matches → **memcmp the GPU
   inherited-impulse manifold vs the CPU `MatchCache`** (NO tolerance). Metal: the CPU reference. Render a
   PURE-INTEGER 2D top-down: the contact points colored by their inherited normal impulse (matched = warm color,
   cold-start = black/grey), the box footprints. Golden = `tests/golden/metal/persist_cache.png` (Mac-baked by the
   CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU matched-manifold impulses == the CPU `MatchCache` byte-for-byte. Print
     `persist-cache: {pairs:<N>, contacts:<P>, matched:<M>, coldStart:<C>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `persist-cache determinism: two runs BYTE-IDENTICAL`.
   - **(3) cache correct:** every UNCHANGED contact (same key tick 1→2) inherited its EXACT tick-1 impulse; every
     CHANGED/new contact cold-started at zero; the tick-1 keys absent in tick 2 were EVICTED (the new cache size ==
     tick-2 contact count). Print `persist-cache correct: {inheritedExact:true, freshColdStart:true,
     staleEvicted:true}`; assert all.
   - **(4) round-trip:** `UpdateCache` then `MatchCache` on the SAME manifold returns the stored impulses exactly
     (the cache is a faithful store). Print `persist-cache roundtrip: {storeThenMatch==stored:true}`; assert.
   - **Golden discipline: ONLY `tests/golden/metal/persist_cache.png`; do NOT commit it.** Existing 197 image
     goldens UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests — APPEND to `tests/persist_test.cpp` (pure CPU):** `BuildKeyedManifold` gives keys parallel to the
   manifold points; `MatchCache` inherits a cached impulse for a matching key + leaves a non-matching key at zero;
   `UpdateCache` evicts a stale key (the new cache is exactly this tick's set); store-then-match round-trips; two
   runs byte-identical. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-persist-cache` (features) + `--persist-cache-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/convex.h` + `fric.h` + `fpx.h` + **PS1's persist.h code + persist_key.comp** + all other sim headers +
  `engine/nav/` + `engine/anim/` + `engine/physics/` + all EXISTING shaders UNCHANGED. The ONLY new shader is
  `persist_cache.comp.hlsl` (int64, Vulkan-only, NOT in hf_gen_msl). `persist.h` APPEND-only. Report the seam empty.

## Out of scope (YAGNI — later PS slices)
The warm-started solver that re-applies + accumulates the inherited impulses (PS3 — PS2 only MATCHES + INHERITS, it
does NOT solve; the impulses it stores in the showcase are synthesized, not solved), sleeping islands (PS4 — the new
physics), lockstep (PS5), the lit 3D render (PS6). PS2 claims ONLY: a deterministic integer persistent cache that
matches this tick's manifold to last tick's impulses by `ContactKey` (inherit / cold-start / evict), bit-identical
CPU↔Vulkan↔Metal, with the integer golden + the four proofs. NOTE: boxes only; the linear cache scan (small scene);
the feature-ID match legitimately misses at a changed contact feature (the documented sliding-contact caveat).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 107 incl. PS1's `persist_test` + the appended PS2 cases).
   Clean under `windows-msvc-asan` (build+run `persist_test` + `introspect_test`).
2. **proofs + visual:** `--persist-cache-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID. **VERIFY the image shows the contacts colored by inherited impulse (matched warm, cold-start dark — a
   coherent cache diagnostic).**
3. Metal: `visual_test --persist-cache` → new golden `tests/golden/metal/persist_cache.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `persist_cache.comp` NOT in `hf_gen_msl`.** Cross-vendor STRICT
   ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `persist_cache.png` added; the other
   197 byte-identical. `git diff master --stat -- tests/golden` = ONLY `persist_cache.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-persist-cache` + `--persist-cache-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/convex.h`/`fric.h`/`fpx.h` + **PS1's persist.h code +
   persist_key.comp** + ALL other sim headers + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL existing
   shaders byte-unchanged). `scripts/verify.ps1` updated: `persist_cache` golden in the Mac loop +
   `--persist-cache-shot` in `$vkShots`. **The ONLY new shader is `persist_cache.comp.hlsl` (int64, NOT in
   `hf_gen_msl`).**
