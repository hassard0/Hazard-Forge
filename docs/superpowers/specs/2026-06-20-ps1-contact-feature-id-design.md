# Slice PS1 — Deterministic Persistent Contacts: THE CONTACT FEATURE ID (the integer beachhead) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice (the BEACHHEAD) of FLAGSHIP #21
> (DETERMINISTIC WARM-STARTED CONTACT CACHING + SLEEPING ISLANDS, `hf::sim::persist`). Warm-starting (re-applying
> last tick's accumulated contact impulses) and sleeping (resting low-energy bodies) are the stabilization
> machinery every shipping physics engine relies on — but float caches + thresholds diverge machine-to-machine,
> which is why their networked physics is never bit-deterministic. This flagship makes that machinery DETERMINISTIC.
> PS1 builds the primitive every later slice needs: a deterministic integer **contact feature ID** that uniquely +
> reproducibly names a contact point across ticks, so the cache can match this tick's manifold points to last
> tick's accumulated impulses. PURE INT32 (no products) → a TRUE GPU pass on BOTH backends, strict zero-differing-
> pixel (the strongest proof tier). CX1-CX6's `convex.h` + FC1-FC6's `fric.h` are BYTE-FROZEN (persist.h is a NEW
> additive sibling). Branch: `slice-ps1`. See [[hazard-forge-persist-roadmap]].

**Goal:** Create `engine/sim/persist.h` (header-only, namespace `hf::sim::persist`, `#include "sim/fric.h"` read-only
ONLY — which transitively gives convex + fpx) with the contact-feature-ID primitive: `ContactKey` +
`MakeContactKey(bodyA, bodyB, sat, pointIndex)` (the deterministic ordered integer key) + `ContactKeysEqual` +
`ContactKeyHash` + a small measure. Add the new int32 (MSL-native) shader `shaders/persist_key.comp.hlsl` +
`--persist-key-shot` (Vulkan) / `--persist-key` (Metal). Bake the integer golden `persist_key`. **NO new RHI.**

## Design call: a deterministic, order-independent integer contact identity

A contact point is identified across ticks by the FEATURES that produced it — the two bodies, the separating axis,
and which clipped corner it is. PS1 packs that into a fixed integer key, computed identically on CPU and GPU:
- **`ContactKey { uint32_t bodyA; uint32_t bodyB; uint32_t axisIndex; uint32_t featureIndex; }`** — `bodyA < bodyB`
  ALWAYS (the pair is order-normalized: if the caller passes `bodyA > bodyB`, swap them — so the same pair yields
  the same key regardless of iteration order); `axisIndex` = the `convex::SatResult::axisIndex` (0..14, the SAT
  min-pen axis that generated the manifold); `featureIndex` = the manifold point's clip-order index (0..3, its slot
  in the `convex::ContactManifold::points[]` / `fric::FrictionManifold::pts[]`). Together `(bodyA, bodyB,
  axisIndex, featureIndex)` is a deterministic per-contact-point identity, stable tick-to-tick while the same SAT
  axis + clip corner persists.
- **`MakeContactKey(bodyAIdx, bodyBIdx, sat, pointIndex)` → ContactKey** — order-normalize the body indices (swap
  to `bodyA < bodyB`), take `sat.axisIndex`, `featureIndex = pointIndex`. **NOTE the swap subtlety:** when the
  bodies are swapped, the SAT axis was computed A→B for the *original* order; the key only needs to be a stable
  identity, so the swap is purely on the stored `bodyA/bodyB` fields (the `axisIndex` is the SAT result's, stored
  as-is — document that the key is an identity, not a geometric frame). Pure integer.
- **`ContactKeysEqual(a, b)` → bool** — field-by-field equality (the cache match predicate).
- **`ContactKeyHash(k)` → uint32_t** — a deterministic integer hash of the four fields (a fixed FNV-1a-style 32-bit
  hash over the four uint32s, or a fixed bit-packing if the fields fit — `bodyA/bodyB` < a few thousand, `axisIndex`
  < 16, `featureIndex` < 4, so a packed `(bodyA<<20)^(bodyB<<8)^(axisIndex<<4)^featureIndex`-style mix is fine).
  Used by PS2's cache bucket lookup. Pure int32, no products beyond shifts/xors → MSL-native.
- **`MeasureKeys(...)`** → the deterministic summary: total keys, distinct keys, max hash collision count — pure
  integer, the showcase prints + asserts (distinct contacts → distinct keys; matching contacts → equal keys).

**THE PURE-INT32 WIN (the strongest proof tier):** unlike the int64 contact MATH (SAT/manifold/impulse), the contact
KEY is pure int32 compares + shifts/xors — no Q16.16 products. So `persist_key.comp` is **MSL-NATIVE** (it goes IN
the `hf_gen_msl` list) — a TRUE GPU pass on BOTH backends, with strict zero-differing-pixel cross-vendor (the FR1/MC1/
BD2 tier, stronger than the int64-Vulkan-only/Metal-CPU-ref tier of the contact math). The keys are derived from
`convex::SatResult` + the manifold point indices (computed host-side or in a prior pass and fed as int32 inputs).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **convex.h (read-only — do NOT edit):** `convex::SatResult` (`convex.h:123`, the `.axisIndex` 0..14),
  `convex::BoxSat`/`BoxSatStable`, `convex::BuildManifold`/`ContactManifold` (`convex.h:372`, the `.count`/
  `.points[]`), `convex::SatPair`, `FxBody`, `FxBox`. PS1 derives keys from these (read-only).
- **fric.h (read-only — do NOT edit):** `fric::FrictionManifold`/`FrictionPoint` (`fric.h:160-176`),
  `fric::BuildFrictionPoints`. persist.h `#include`s fric.h read-only (transitively convex + fpx).
- **The new-shader showcase precedent (MSL-NATIVE int32):** study an INTEGER MSL-native slice's shader + showcase
  — `shaders/fract_emit_count.comp.hlsl` / the BD2 `boids_cell_*.comp` (pure int32, IN hf_gen_msl, a true GPU pass
  both backends, the strict-zero memcmp). Contrast with CX1's `convex_sat.comp` (int64, Vulkan-only) — PS1 is the
  int32/MSL-native kind. Mirror the int32 pattern. Confirm `persist_key` IS in hf_gen_msl (the strict-zero tier).
- **The Vulkan + Metal showcase wiring:** CX1's `--convex-sat-shot`/`--convex-sat` (the per-element compute +
  GPU==CPU memcmp + the standalone arg-parse + the 2D render). Mirror.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), a NEW `tests/persist_test.cpp` (+ CMake wiring, the convex_test/fric_test
  pattern).

## Design decisions (locked)
1. **NEW `engine/sim/persist.h`** (header-only, namespace `hf::sim::persist`, `#include "sim/fric.h"` read-only):
   `ContactKey`, `MakeContactKey`, `ContactKeysEqual`, `ContactKeyHash`, `MeasureKeys`. Pure int32, order-normalized
   (`bodyA < bodyB`), FIXED hash mix. **NEW shader** `persist_key.comp.hlsl` (**int32, MSL-NATIVE → IN hf_gen_msl**,
   one thread per contact — copies `MakeContactKey`+`ContactKeyHash` VERBATIM, writes the `ContactKey`+hash per
   contact). A TRUE GPU pass on both backends.
2. **Showcase `--persist-key-shot <out>` (Vulkan) AND `--persist-key` (Metal) — WIRE BOTH** (standalone arg-parse).
   The SCENE: the CX2/FC2 fixed deterministic box-pair array → run `BoxSatStable`+`BuildManifold` per pair → for
   every contact point emit a `(bodyA, bodyB, sat, pointIndex)` → `MakeContactKey`. Vulkan: the GPU
   `persist_key.comp` → **memcmp the GPU `ContactKey[]`+hash[] vs the CPU `MakeContactKey`/`ContactKeyHash`** (NO
   tolerance). Metal: the GPU MSL-native pass (NOT a CPU ref — this slice is MSL-native, so Metal runs the SHADER).
   Render a PURE-INTEGER 2D top-down view: the box footprints + each contact point colored by `ContactKeyHash` (so
   distinct keys are distinct colors; a contact that persists across two synthesized ticks keeps its color). Golden
   = `tests/golden/metal/persist_key.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU `ContactKey[]`+hash[] == the CPU byte-for-byte. Print `persist-key:
     {pairs:<N>, contacts:<P>, distinctKeys:<D>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `persist-key determinism: two runs BYTE-IDENTICAL`.
   - **(3) identity correct:** distinct contact points (different pair/axis/feature) get DISTINCT keys; the SAME
     contact re-derived from a synthesized "next tick" (same pair + same SAT axis + same clip corner) gets an EQUAL
     key (`ContactKeysEqual` true); the body order-normalization holds (`MakeContactKey(i,j,...)` ==
     `MakeContactKey(j,i,...)` for the key identity). Print `persist-key identity: {distinctDistinct:true,
     matchMatches:true, orderNormalized:true}`; assert all.
   - **(4) MSL-native (strict-zero tier):** the Metal `--persist-key` runs the GPU shader (NOT a CPU reference) and
     matches Vulkan/the golden at ZERO differing pixels. (Covered by the cross-vendor strict-zero in the gate.)
   - **Golden discipline: ONLY `tests/golden/metal/persist_key.png`; do NOT commit it.** Existing 196 image goldens
     UNTOUCHED.
4. **Cross-backend bar (INTEGER MSL-NATIVE, the STRONGEST tier):** Vulkan GPU == Metal GPU == golden, ZERO differing
   pixels (a true GPU pass on both backends, not a CPU reference).
5. **Tests `tests/persist_test.cpp` (NEW, pure CPU):** `MakeContactKey` order-normalizes (`(i,j)` and `(j,i)` give
   the same key); distinct (pair/axis/feature) → distinct keys; identical features → equal keys + equal hash;
   `ContactKeyHash` is deterministic + collision-light over the showcase contacts; two runs byte-identical. Clean
   under `windows-msvc-asan`. Wire the new test into CMake.
6. **Introspect.** Add exactly `deterministic-persist-key` (features) + `--persist-key-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/convex.h` + `fric.h` + `fpx.h` + ALL other sim headers + `engine/nav/` + `engine/anim/` +
  `engine/physics/` + all EXISTING shaders UNCHANGED. The ONLY new shader is `persist_key.comp.hlsl` (int32,
  **MSL-native, IN hf_gen_msl** — a TRUE GPU pass both backends). `persist.h` is a NEW additive sibling. Report the
  seam empty (only `persist.h` + the new shader + the showcase/test/introspect are new/changed).

## Out of scope (YAGNI — later PS slices)
The persistent manifold cache (PS2 — PS1 only MAKES the key, it does not store/match across real ticks), the
warm-started solver (PS3), sleeping islands (PS4 — the new physics), lockstep (PS5), the lit 3D render (PS6 — PS1's
render is the 2D key-color diagnostic). PS1 claims ONLY: a deterministic integer contact feature ID (+ hash) that
uniquely + reproducibly names a contact point, bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four
proofs. NOTE: boxes only (inherits the box-box manifold); the key is an IDENTITY (stable while the SAT axis + clip
corner persist — it legitimately changes when the contact feature changes, e.g. a face sliding to a new corner set,
which is the documented warm-start-misses-at-sliding-contacts caveat).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 106 + the NEW `persist_test`). Clean under `windows-msvc-asan`
   (build+run `persist_test` + `introspect_test`).
2. **proofs + visual:** `--persist-key-shot` on Vulkan: the proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID. **VERIFY the image shows the box pairs with per-contact-point key-colored markers (a coherent
   diagnostic, distinct keys distinct colors).**
3. Metal: `visual_test --persist-key` → new golden `tests/golden/metal/persist_key.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `persist_key.comp` IS in `hf_gen_msl` (MSL-native — Metal runs
   the GPU shader, the strict-zero tier).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `persist_key.png` added; the other
   196 byte-identical. `git diff master --stat -- tests/golden` = ONLY `persist_key.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-persist-key` + `--persist-key-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/convex.h`/`fric.h`/`fpx.h`/ALL sim headers + `engine/nav/` +
   `engine/anim/` + `engine/physics/` + ALL existing shaders byte-unchanged). `scripts/verify.ps1` updated:
   `persist_key` golden in the Mac loop + `--persist-key-shot` in `$vkShots`. **The ONLY new shader is
   `persist_key.comp.hlsl` (int32, IN `hf_gen_msl`).**
