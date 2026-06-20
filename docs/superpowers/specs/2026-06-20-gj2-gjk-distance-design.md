# Slice GJ2 — General Convex-Hull Contacts: GJK OVERLAP + CLOSEST DISTANCE (the simplex) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #22 (DETERMINISTIC
> GENERAL CONVEX-HULL CONTACTS via integer GJK + EPA, `hf::sim::gjk`). GJ1 built the support function + the
> Minkowski-difference support. GJ2 builds the **GJK algorithm itself**: the Gilbert-Johnson-Keerthi simplex
> evolution that, fed only `SupportMinkowski` queries, decides whether two convex hulls OVERLAP and (when
> separated) returns their closest-point distance + the witness points on each hull. This is the deterministic
> integer core that mainstream float engines run in floating point — whose simplex-barycentric sub-distance and
> search-direction accumulations are FPU-order/vendor-dependent, so the simplex they converge to diverges
> machine-to-machine. GJ2 does the same evolution entirely in Q16.16 with a fixed iteration bound and integer
> tie-breaks. APPEND to `engine/sim/gjk.h` (GJ1's code + convex.h/fric.h/persist.h/fpx.h BYTE-FROZEN). Branch:
> `slice-gj2`. See [[hazard-forge-gjk-roadmap]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/gjk.h` (additive — GJ1 byte-unchanged) with `Simplex` (up to 4 CSO points + their
witness points on A and B) + `GjkResult` (overlap bool, separation vector, closest points on A and B, a witness
feature) + `Gjk(hullA, bodyA, hullB, bodyB)` (the GJK main loop: seed a support, evolve the simplex via the
Johnson sub-distance toward the origin, fixed bound `kGjkMaxIter`, integer origin-containment) + a `GjkMeasure`
summary. Add the int64 GPU shader `shaders/gjk_distance.comp.hlsl` (Vulkan-only, copies `Gjk` verbatim) + the
showcase `--gjk-distance-shot` (Vulkan) / `--gjk-distance` (Metal). Bake the integer golden `gjk_distance`.

## Design call: the Johnson sub-distance GJK in Q16.16 — fixed bound, integer tie-breaks

GJK maintains a simplex (1–4 points) of the Minkowski difference A⊖B and walks it toward the origin: each
iteration finds the closest feature of the current simplex to the origin (the **sub-distance** step — reduce the
simplex to that feature: vertex/edge/triangle/tetra), computes a new search direction `d` (origin minus the
closest point, i.e. toward the origin), and adds `SupportMinkowski(d)`. It terminates when (a) the new support
makes no progress along `d` (the hulls are **separated** — the closest feature is the answer), or (b) the simplex
encloses the origin (the hulls **overlap**), or (c) the fixed iteration bound is hit.
- **The barycentric witness recovery:** each simplex point stores not just the CSO point `p = Support_A(d) −
  Support_B(−d)` but also `wA = Support_A(d)` and `wB = Support_B(−d)`. When GJK reduces to the closest feature,
  the same barycentric weights that express the closest CSO point combine `wA`/`wB` into `closestA`/`closestB`
  (the witness points on each hull) — the closest-point answer.
- **DETERMINISM CRUX (the make-or-break, spell it out for the implementer):**
  - **Fixed iteration bound** `kGjkMaxIter` (e.g. 32) — never an unbounded `while`.
  - **Integer sub-distance with fixed tie-breaks:** the Johnson sub-distance picks among vertex/edge/face Voronoi
    regions using `FxDot`/`FxCross` sign tests; ALL comparisons are integer, and any tie (a point exactly on a
    Voronoi boundary) resolves by a FIXED rule (lowest-index feature, the convex.h:28 idiom). Barycentric weights
    use `fxdiv` — pin the division order.
  - **Integer origin-containment:** the tetra-encloses-origin test is four `FxDot(faceNormal, −facePoint)` sign
    checks in a FIXED face order with a FIXED winding; a degenerate (near-zero) simplex face is handled
    deterministically (the `kEdgeEps` precedent, convex.h:130).
  - **Fixed initial search direction** (e.g. `bodyB.pos − bodyA.pos`, or a fixed axis if that is zero) so two
    runs from the same inputs evolve the identical simplex.
  - **Duplicate-support guard:** if `SupportMinkowski(d)` returns a CSO point already in the simplex (within an
    integer epsilon), terminate as separated (no progress) — deterministic, prevents cycling.
- **`Gjk` returns** `{overlap, separation (the closest CSO point = origin→closest, its FxLength = the distance),
  closestA, closestB, simplex (the terminal simplex — GJ3 EPA seeds from it), witnessFeature (a deterministic
  feature id: the terminal simplex's witness vertex indices on A and B, packed — the persist ContactKey raw
  material for later slices), iterations}`. For the overlap case, `separation` is zero / `closestA==closestB` is
  not meaningful (EPA in GJ3 computes the penetration); GJ2 only needs the overlap boolean + the terminal simplex
  there.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **GJ1 `engine/sim/gjk.h` (read it; APPEND only after `MakeWedge`, before the namespace close):** `FxHull`,
  `Support`, `SupportMinkowski` (the ONLY geometry call GJK makes), `FxNeg`, the canonical hull builders,
  `convex::fx`/`FxVec3`/`FxDot`, `fpx::FxBody`. GJ1 byte-frozen.
- **convex.h (read-only — REUSE, do NOT redefine):** `convex::FxCross` (for the face normals / Voronoi tests),
  `fpx::FxLength`/`FxISqrt` (the separation distance), `convex::ClosestPointsOnSegments` (convex.h:301 — a
  reference for the edge sub-distance closest-point math + the integer-segment idiom; reuse its approach, it is
  the closest existing integer closest-point routine), the `kEdgeEps` degeneracy precedent (convex.h:130).
- **The proof-tier convention (convex.h:16-23):** int64 (the sub-distance barycentrics + `FxDot`/`FxCross` +
  `fxdiv`) → `shaders/gjk_distance.comp.hlsl` is **VULKAN-ONLY** (NOT in `hf_gen_msl`); Metal `--gjk-distance`
  runs the CPU `Gjk` → byte-identical by construction; the Vulkan side carries the GPU==CPU memcmp. The shader
  copies `Gjk`'s body VERBATIM; one GPU thread per hull-pair.
- **The showcase + shader precedent:** GJ1's `--gjk-support-shot` (the int64 compute proof + GPU==CPU memcmp +
  diagnostic render) and `--convex-sat-shot`. Mirror for `--gjk-distance`.
- **Registration:** `scripts/verify.ps1` (append `gjk_distance` to the Mac table + `--gjk-distance-shot` to
  `$vkShots`), `metal_headless/CMakeLists.txt` (`hf_gen_msl` — do NOT add `gjk_distance.comp`; Vulkan-only),
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON golden — do NOT**),
  append to `tests/gjk_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/gjk.h`** (GJ1 byte-frozen): `kGjkMaxIter`, `Simplex { FxVec3 pts[4]; FxVec3 csoA[4];
   FxVec3 csoB[4]; uint32_t count; }`, `GjkResult`, `Gjk(...)`, `GjkMeasure` + `MeasureGjk`. Pure integer, FIXED
   iteration + tie-break order. NO new RHI; one new Vulkan-only shader.
2. **New shader `shaders/gjk_distance.comp.hlsl` (int64, VULKAN-ONLY)** — copies `Gjk` verbatim; one thread per
   hull-pair; writes the `GjkResult` (overlap + separation + closest points + witness) to an SSBO. NOT in
   `hf_gen_msl`.
3. **Showcase `--gjk-distance-shot <out>` (Vulkan) AND `--gjk-distance` (Metal) — WIRE BOTH.** A fixed pair array
   spanning the three regimes: clearly SEPARATED, nearly TOUCHING (closest distance ~0+), and SHALLOW OVERLAP
   (origin just inside). A couple of pairs use rotated bodies (exercise the GJ1 `Support` rotation). Vulkan
   dispatches `gjk_distance.comp` + memcmps GPU vs CPU `Gjk`; Metal runs CPU `Gjk`. BOTH render an integer
   diagnostic (the hull pairs + a marker on each closest-point witness / an overlap flag color). Golden =
   `tests/golden/metal/gjk_distance.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU:** `gjk-distance: {pairs:<P>, separated:<S>, overlapping:<O>} GPU==CPU BIT-EXACT` — the GPU
     `GjkResult` buffer == the CPU one byte-for-byte; assert.
   - **(2) determinism:** `gjk-distance determinism: two runs BYTE-IDENTICAL`.
   - **(3) correctness:** the overlap boolean agrees with an INDEPENDENT CPU reference for every pair — for
     separated pairs, a brute-force min-vertex-distance / SAT-style separation check confirms non-overlap AND the
     GJK closest distance matches that reference within the documented fixed-point band; for overlapping pairs,
     an independent containment check confirms overlap. Print `gjk-distance correct: {overlapAgrees:true,
     distInBand:true}`; assert. (The closest-distance band tolerance is documented, the GJ-honesty precedent —
     GJK closest-point is exact up to fixed-point truncation, NOT byte-identical to a different reference
     algorithm; the GPU==CPU of `Gjk` itself IS byte-identical.)
   - **Golden discipline: ONLY `tests/golden/metal/gjk_distance.png`; do NOT commit it.** Existing 204 image
     goldens UNTOUCHED.
5. **Cross-backend bar (int64 → strict):** Vulkan GPU==CPU bit-exact; Metal CPU-ref byte-identical by
   construction; cross-vendor ZERO differing pixels.
6. **Tests — APPEND to `tests/gjk_test.cpp`:** `Gjk` overlap boolean matches an independent reference over the
   separated/touching/overlapping pair set; the closest distance for separated pairs is within the band of a
   brute-force reference; `Gjk` is deterministic (two calls byte-equal); the terminal simplex is valid (count
   1–4, points are genuine `SupportMinkowski` results); a known analytic case (two unit boxes offset along X by
   3 units, half-extent 1 → gap 1.0) returns the expected distance within band. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-gjk-distance` (features) + `--gjk-distance-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** The shot rides the existing compute-dispatch + SSBO path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/gjk.h` is APPEND-only (GJ1 frozen); convex.h/fric.h/persist.h/fpx.h + ALL other sim headers + ALL
  existing shaders + `engine/physics/`/`nav/`/`anim/` UNCHANGED. NEW files: `shaders/gjk_distance.comp.hlsl`
  only. Report the seam: one new Vulkan-only shader, no RHI change, no frozen-file edit, gjk.h append-only.

## Out of scope (YAGNI — later slices)
EPA penetration depth/normal (GJ3 — when GJK reports overlap, GJ3's EPA seeded from GJ2's terminal simplex
computes the depth+normal), the hull world step + manifold (GJ4), lockstep (GJ5), lit render (GJ6). GJ2 claims
ONLY: a deterministic, bit-exact (CPU↔Vulkan↔Metal) GJK overlap test + separated-case closest distance/witness
points, with the integer golden + the three proofs. CAVEATS: convex polyhedra only; the closest distance is exact
up to Q16.16 truncation (within-band vs an independent reference, NOT cross-algorithm byte-identical — though
`Gjk` itself is byte-identical CPU/GPU); the overlap-case closest points are not meaningful (EPA's job).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 108 incl. the appended GJ2 `gjk_test` cases). Clean under
   `windows-msvc-asan` (build+run `gjk_test` + `introspect_test`).
2. **proofs + visual:** `--gjk-distance-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate
   → ZERO VUID. **VERIFY the diagnostic image shows the hull pairs with their closest-point witnesses / overlap
   flags coherently (no garbage/NaN).**
3. Metal: `visual_test --gjk-distance` → new golden `tests/golden/metal/gjk_distance.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `gjk_distance.comp` is NOT in `hf_gen_msl`.** Cross-vendor
   STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `gjk_distance.png` added; the
   other 204 byte-identical. `git diff master --stat -- tests/golden` = ONLY `gjk_distance.png` (metal) + the
   introspect json (controller rebake).
5. Introspect: exactly `+deterministic-gjk-distance` + `--gjk-distance-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + GJ1's gjk.h code + convex.h/fric.h/persist.h/fpx.h + ALL other sim headers + ALL
   existing shaders byte-unchanged; gjk.h APPEND-only; one new Vulkan-only shader, no RHI change).
   `scripts/verify.ps1` updated; `gjk_distance.comp` NOT in `hf_gen_msl`.
