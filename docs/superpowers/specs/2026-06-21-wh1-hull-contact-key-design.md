# Slice WH1 — Warm-Started Hull Contacts: THE HULL CONTACT FEATURE ID (the int32 MSL-native beachhead) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice of
> FLAGSHIP #26 (WARM-STARTED HULL CONTACTS + ROBUST DETERMINISTIC STACKING, `hf::sim::warmhull`). Flagship #25
> hardened the hull narrowphase (multi-point manifolds + full inertia) so a SINGLE hull settles on a support, but
> documented that TALL stacks destabilize: the non-accumulated Gauss-Seidel + the fixed-point 4-point manifold
> leave a per-tick RESIDUAL TORQUE that integrates into a spurious spin and topples the tower (convex.h:758-763),
> currently masked only by a global angular drag. Flagship #26 removes the SOURCE of that torque by generalizing
> the box-only warm-start + sleeping-islands machinery (`persist.h`, #21) to the hardened hull multi-point
> manifolds — an **accumulated, warm-started** solver that converges to a consistent island equilibrium instead of
> re-deriving a slightly-inconsistent impulse each tick. WH1 is the BEACHHEAD: the deterministic **hull contact
> feature ID** that names a contact across frames so next tick's manifold points can inherit last tick's
> accumulated impulses. THE CRUX: a hull manifold has no discrete SAT axis (EPA gives a continuous normal + a
> Sutherland–Hodgman clip), and the clip-ORDER slot renumbers under sub-LSB motion — so the key must be a
> GEOMETRIC PROVENANCE `(reference face, incident source feature)`, not an array slot. Because the key is pure
> int32 (small face/vertex/edge indices, shifts + xors — NO Q16.16 products, NO int64), the GPU shader is **int32
> MSL-native** (the strongest proof tier — true GPU on BOTH backends, strict-zero cross-vendor), mirroring
> persist.h's PS1 split. NEW header `engine/sim/warmhull.h` (`#include "sim/manifold.h"` — manifold/gjk/broad/ccd/
> convex/fric/persist/fpx all BYTE-FROZEN). Branch: `slice-wh1`. See [[hazard-forge-warmhull-roadmap]],
> [[hazard-forge-manifold-roadmap]], [[hazard-forge-docs-style]], [[hazard-forge-metal-showcase-gate]].

**Goal:** Create `engine/sim/warmhull.h` (namespace `hf::sim::warmhull`) with the hull contact feature ID:
`HullContactKey` (`uint32 bodyA, bodyB, refFaceId, incVertId`, order-normalized so `bodyA < bodyB`) +
`MakeHullContactKey` / `HullContactKeysEqual` / `HullContactKeyHash` (pure int32, the persist.h avalanche idiom) +
`ClipFaceAgainstFaceTagged` (a NEW provenance-carrying clip that mirrors the frozen `manifold::ClipFaceAgainstFace`
byte-for-byte and additionally emits a per-output-vertex provenance tag) + `BuildHullContactKeys` (tag each
`HullContactMulti` manifold point with its key via the tagged clip) + `HullKeyMeasure` / `MeasureHullKeys`. Add the
showcase `--wh1-keys-shot <out>` (Vulkan: GPU `warmhull_key.comp` → memcmp vs CPU) / `--wh1-keys` (Metal: GPU
`warmhull_key.comp` — strict-zero cross-vendor). Bake the integer golden `wh1_keys`. **NO new render RHI**; ONE new
compute shader (int32 MSL-native).

## Design call: a geometric-provenance feature ID, computed inside a byte-identical tagged clip

persist.h's box `ContactKey` (persist.h:47-68) is `(bodyA, bodyB, axisIndex, featureIndex)` — `axisIndex` is the
discrete box-SAT min-pen axis. The hull manifold has no such axis: `manifold::HullManifoldFromEpa` (manifold.h:467)
takes a continuous EPA normal and clips (manifold.h:372-451), and the MF2 reduction keeps "the deepest + first 3 in
clip order" (manifold.h:541-553) — so the clip-order slot **renumbers** when the bodies shift by 1 LSB. Keying on
the slot would mis-match every tick and inherit the WRONG corner's impulse (injecting energy). The fix is a
**geometric provenance** key:
- **`refFaceId`** = the reference face index `manifold::SupportFace` chose (manifold.h:184) — already deterministic
  (FIXED face order, strict-greater lowest-index tie-break). Plus a `refIsA` bit (which hull owns the reference
  face, manifold.h:502) folded into the order-normalized key.
- **`incVertId`** = the *provenance* of the clipped point: for a clip output that is an ORIGINAL incident-polygon
  vertex (it survived the clip), the incident hull's LOCAL vertex index (`incFaces.vertIdx[incFace][k]` — a small
  uint invariant under rigid motion); for a clip INTERSECTION point (an edge crossing), a packed
  `(refEdgeIndex, incEdgeIndex)` (both FIXED-order small integers from the SH loop, with a high tag-bit marking
  "intersection" vs "vertex" so the two encodings never collide).
- **Order normalization** (persist.h:60-68): store `bodyA < bodyB`; the per-key `refIsA`/role bits make the key an
  identity independent of which body is iterated first.

**`ClipFaceAgainstFaceTagged` (the determinism contract).** `manifold::ClipFaceAgainstFace` (manifold.h:372) is
FROZEN and returns only positions. WH1 reimplements a provenance-carrying parallel clip in `warmhull.h` that
mirrors the frozen clip's EXACT control flow — the SAME FIXED edge order (`k = 0..refVc-1`), the SAME strict-integer
inside test (`FxDot(sideN, p - a) >= 0`, on-plane = inside, NO tolerance band), the SAME `fxdiv` crossing formula in
the SAME pinned iteration order (manifold.h:405-447) — additionally carrying, per polygon vertex, a provenance tag
through the clip (an original incident vertex keeps its local-index tag; a crossing emits the `(refEdge, incEdge)`
tag). **The contract:** the tagged clip's OUTPUT POSITIONS are provably byte-equal to the frozen
`ClipFaceAgainstFace`'s (same integer math, same order) — assert this in the test (run both, memcmp the positions) —
and the tags are a deterministic function of the same integer signs. This is the 1-LSB-flip discipline
(manifold.h:346-348) applied to the tag.

> NOTE: the KEY is pure int32 (face/vertex/edge indices + shift/xor/avalanche hashing — NO Q16.16 products, NO
> int64), so `warmhull_key.comp.hlsl` is **MSL-native** (true GPU on both backends, strict-zero cross-vendor), even
> though the manifold that produced the indices is int64. This is exactly persist.h's PS1 (int32 MSL-native key) /
> PS2+ (int64 manifold, Vulkan-only) split, and the CD2 swept-broadphase int32-MSL-native precedent. The key shader
> is LIGHT (per-pair index packing — no GJK/EPA, no heavy loop) → no TDR risk.

## Reuse map (file:line)
- **manifold.h (read-only — REUSE, do NOT edit):** `manifold::HullContactMulti` (manifold.h:588 — produces the
  manifold WH1 tags), `manifold::HullManifoldFromEpa`/`ClipFaceAgainstFace` (manifold.h:467/372 — the clip
  `ClipFaceAgainstFaceTagged` mirrors byte-for-byte), `manifold::SupportFace`/`IncidentFace` (manifold.h:184/204 —
  the face IDs), `manifold::BuildCanonicalFaces`/`FxHullFaces` (manifold.h faces), `manifold::FaceNormalWorld`.
- **persist.h (read-only — the TEMPLATE pattern, not linked):** `persist::ContactKey` (persist.h:47-68 — the
  order-normalized key shape), `persist::ContactKeyHash` (persist.h:84-94 — the int32 shift/xor avalanche to copy),
  `persist::ContactKeysEqual`. (persist.h's `BuildKeyedManifold` is box-only — WH1 writes the hull analog new.)
- **gjk.h / convex.h (read-only):** `gjk::Gjk`/`gjk::Epa` (the narrowphase the showcase runs before tagging),
  `gjk::HullWorld`/`FxHull`/canonical builders, `convex::FxDot`/`FxCross`, `convex::ContactManifold`.
- **The int32-MSL-native compute precedent:** `shaders/ccd_swept_count.comp.hlsl` / `broad_cell_count.comp.hlsl`
  (int32 compute registered in BOTH the Vulkan list AND `hf_gen_msl`) + the PS1 `persist` key shader. Mirror the
  registration for `warmhull_key.comp.hlsl`.
- **Registration:** `samples/hello_triangle/CMakeLists.txt` (`shaders/warmhull_key.comp.hlsl:cs` — Vulkan),
  `metal_headless/CMakeLists.txt` (`hf_gen_msl(warmhull_key ...)` — Metal MSL, since int32), `scripts/verify.ps1`
  (`wh1_keys` + `--wh1-keys-shot` to `$vkShots`), `engine/editor/introspect.cpp` + `tests/introspect_test.cpp`
  (**controller rebakes the JSON golden**), a NEW `tests/warmhull_test.cpp` (registered in `tests/CMakeLists.txt`).

## Design decisions (locked)
1. **NEW header `engine/sim/warmhull.h`** (`namespace hf::sim::warmhull`, `#include "sim/manifold.h"`):
   `HullContactKey`, `MakeHullContactKey`, `HullContactKeysEqual`, `HullContactKeyHash`, `ClipFaceAgainstFaceTagged`,
   `BuildHullContactKeys`, `HullKeyMeasure`, `MeasureHullKeys`. **manifold.h and ALL other sim headers
   BYTE-UNCHANGED.** `HullContactKey` is std430-packable (4×uint32) so WH2's cache can hand it to a shader.
2. **ONE new shader `shaders/warmhull_key.comp.hlsl` — int32, MSL-NATIVE** (registered in BOTH the Vulkan compute
   list AND `hf_gen_msl`). It copies `BuildHullContactKeys` VERBATIM (pure int32). **NO new render RHI.**
3. **Showcase `--wh1-keys-shot <out>` (Vulkan) AND `--wh1-keys` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--wh1-keys` BEFORE reporting DONE).** Build a deterministic scene of overlapping hull
   pairs (a box-on-box flat contact, a tetra-on-face, an edge contact — the MF2 battery), run `HullContactMulti` +
   `BuildHullContactKeys`, DISPATCH `warmhull_key.comp` (one thread per pair), and render a simple integer
   visualization (the keyed contacts — a 2D side-view or the contact-point markers; pure-integer → strict-zero
   cross-vendor). Golden = `tests/golden/metal/wh1_keys.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact stdout lines):**
   - **(1) tagged-clip fidelity:** `wh1-keys: tagged clip == frozen clip (points BYTE-EQUAL)` — `ClipFaceAgainstFaceTagged`'s
     output positions memcmp-equal the frozen `manifold::ClipFaceAgainstFace`'s over the battery (the determinism
     contract).
   - **(2) key discrimination:** `wh1-keys: {pairs:<P>, keys:<K>} distinct contacts -> distinct keys` — distinct
     clipped contacts hash/compare distinct; and a contact under a sub-LSB relative nudge keeps the SAME key
     (same ref face + same incident source vertex) → `matchedUnderNudge:true`.
   - **(3) GPU==CPU:** `wh1-keys: GPU == CPU BIT-EXACT` — the GPU `warmhull_key.comp` key+hash array memcmp-equals
     the CPU `BuildHullContactKeys` array (Vulkan).
   - **(4) MSL-native cross-vendor:** the Metal GPU `warmhull_key.comp` output is strict-zero cross-vendor (the
     controller's Metal bake DIFF 0.0000 + the same proof lines) — the int32-MSL-native guarantee.
   - Golden discipline: ONLY `tests/golden/metal/wh1_keys.png`; do NOT commit it. Existing 227 goldens UNTOUCHED.
5. **Cross-backend bar (int32 MSL-native → strongest):** Vulkan GPU==CPU bit-exact; Metal GPU==CPU AND the same
   shader runs natively on Metal (not a CPU ref) → strict-zero cross-vendor on the key data. The render is the
   pure-integer side-view (strict-zero), or if a float lit render is chosen, in-band visresolve — pick the
   pure-integer side-view for the strongest bar.
6. **Tests — NEW `tests/warmhull_test.cpp` (pure CPU):** the tagged-clip == frozen-clip position memcmp; distinct
   contacts → distinct keys; sub-LSB nudge → same key; a ref/inc-flip or sliding contact → key CHANGES (the
   documented warm-start-miss boundary — assert it changes, the honest caveat); `HullContactKeyHash` distributes
   (no trivial collisions over the battery); order-normalization (swapping bodyA/bodyB yields the same key). Clean
   under `windows-msvc-asan`.
7. **Introspect.** Add EXACTLY `warmhull-contact-key` (features) + `--wh1-keys-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **ONE new compute shader, NO new render RHI.** `shaders/warmhull_key.comp.hlsl` (int32, MSL-native — Vulkan DXC
  SPIR-V AND Metal via `hf_gen_msl`). Reuses the existing compute-dispatch + SSBO seam. `engine/sim/warmhull.h` is
  a brand-new APPEND-only sibling; manifold.h/gjk.h/persist.h/convex.h/fric.h + ALL other sim headers + ALL OTHER
  shaders UNCHANGED. Report the seam: NEW header + ONE new MSL-native shader; NO rhi.h change, NO frozen-file edit.

## Out of scope (YAGNI — later slices)
The keyed manifold + persistent cache (WH2). The accumulated warm-started solver (WH3). Sleeping islands + the
stable stack (WH4). Lockstep (WH5), the render capstone (WH6). WH1 is ONLY the feature-ID primitive — it tags
contacts, it does NOT yet cache or warm-start anything. The warm-start-miss at sliding/flipping contacts is a
documented, in-scope BOUNDARY (the key legitimately changes → a safe cold-start), not a bug. A general quickhull
provenance (canonical hulls only, via `BuildCanonicalFaces`). WH1 claims ONLY: a deterministic, geometric-provenance
hull contact feature ID, computed by a tagged clip byte-identical to the frozen clip, the same on CPU/Vulkan/Metal
(int32 MSL-native), with the integer golden + the four proofs.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "warmhull|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--wh1-keys-shot` on Vulkan: the 4 proof lines (incl. tagged-clip==frozen + GPU==CPU) +
   exit 0 under the conan validation layer → ZERO VUID. VERIFY the keyed-contact visualization is coherent.
3. Metal: `visual_test --wh1-keys` → `tests/golden/metal/wh1_keys.png`; two runs DIFF 0.0000. **Confirm `--wh1-keys`
   wired in `visual_test.mm` (grep it) BEFORE the Mac bake; confirm `hf_gen_msl(warmhull_key ...)` IS present (int32
   MSL-native — it MUST be in the Metal MSL list, unlike the int64 hull shaders).** Cross-vendor STRICT ZERO (the
   key data + the pure-integer render).
4. **Render-invariance:** ONLY `wh1_keys.png` added; the other 227 byte-identical (+ controller introspect rebake).
5. Introspect: exactly `+warmhull-contact-key` + `--wh1-keys-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + manifold.h/gjk.h/persist.h/convex.h/fric.h + ALL other sim headers + ALL OTHER
   shaders byte-unchanged; `warmhull.h` NEW `#include "sim/manifold.h"`; exactly ONE new shader
   `warmhull_key.comp.hlsl`, in BOTH the Vulkan list AND `hf_gen_msl`). `wh1_keys` in the Mac loop +
   `--wh1-keys-shot` in `$vkShots`.
