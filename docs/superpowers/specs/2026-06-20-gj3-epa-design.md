# Slice GJ3 — General Convex-Hull Contacts: EPA PENETRATION DEPTH + NORMAL (the crux) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice (THE CRUX) of FLAGSHIP #22
> (DETERMINISTIC GENERAL CONVEX-HULL CONTACTS via integer GJK + EPA, `hf::sim::gjk`). GJ1 built the support
> function; GJ2 built GJK (overlap + closest distance) returning, for an overlapping pair, a terminal simplex
> that encloses the origin. GJ3 builds **EPA** (the Expanding Polytope Algorithm): seeded from GJK's terminal
> overlap simplex, it expands a polytope of the Minkowski difference outward to the boundary to find the
> **penetration depth** (the minimum translation to separate the hulls) + the **contact normal** + the contact
> points on each hull. This is THE crux of the whole flagship: penetration depth+normal is what the contact
> solver consumes, and EPA's iterative polytope expansion is the single most determinism-sensitive piece —
> mainstream float engines' EPA face-distance/expansion accumulations are FPU-order/vendor-dependent. GJ3 does it
> entirely in Q16.16 with a fixed iteration bound, pinned face/horizon ordering, and integer tie-breaks. APPEND
> to `engine/sim/gjk.h` (GJ1+GJ2 + convex.h/fric.h/persist.h/fpx.h BYTE-FROZEN). Branch: `slice-gj3`. See
> [[hazard-forge-gjk-roadmap]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/gjk.h` (additive — GJ1+GJ2 byte-unchanged) with `kEpaMaxIter`, `kMaxPolyVerts`,
`kMaxPolyFaces`, `PolyFace` (a triangle + outward normal + origin-distance), `Polytope` (the expanding polytope:
CSO verts + their witness verts on A/B + faces), `EpaResult` (depth, normal A→B, contact points on A and B, a
deterministic feature id, iterations, valid flag) + `Epa(hullA, bodyA, hullB, bodyB, terminalSimplex)` (seed the
polytope from GJK's terminal overlap simplex → iteratively expand the closest face toward the boundary → return
the penetration depth+normal+contacts). Add the int64 GPU shader `shaders/gjk_epa.comp.hlsl` (Vulkan-only, copies
`Epa` verbatim) + the showcase `--gjk-epa-shot` (Vulkan) / `--gjk-epa` (Metal). Bake the integer golden `gjk_epa`.

## Design call: EPA in Q16.16 — fixed bound, pinned face/horizon order, integer tie-breaks

EPA takes the GJK terminal simplex (which, for an overlapping pair, encloses the origin) and grows it into a
polytope whose faces approach the Minkowski-difference boundary. Each iteration: (1) find the polytope **face
closest to the origin** (min `face.dist`, the perpendicular origin-distance); (2) query `SupportMinkowski` along
that face's outward normal; (3) if the new support is NOT measurably farther out than the face (the support's
projection onto the normal ≤ `face.dist + kEpaTol`), the face IS the boundary → **converged**: `depth = face.dist`,
`normal = face.normal`, contacts via the barycentric projection of the origin onto the face; (4) else add the
support vertex, **remove every face the vertex can see** (faces with `dot(face.normal, newVert − face.vertex) >
0`), collect the **horizon** (the boundary edges of the removed region), and re-triangulate by connecting the new
vertex to each horizon edge. Repeat to `kEpaMaxIter`.
- **Seeding (handle the non-tetra terminal simplex).** GJK's terminal overlap simplex may be a tetra (4 pts) or,
  in degenerate-but-deterministic cases, fewer. EPA needs a tetra enclosing the origin. If `terminalSimplex.count
  < 4`, expand it deterministically: for a triangle, add `±SupportMinkowski(faceNormal)` (the side that, with the
  triangle, encloses the origin); for an edge/point, the standard blow-up adding supports along fixed axes. Pin
  every choice (fixed axis order, fixed sign rule). Document the seeding path.
- **DETERMINISM CRUX (the make-or-break — spell it out for the implementer):**
  - **Fixed iteration bound** `kEpaMaxIter` (e.g. 48) — never an unbounded loop; on hitting the bound, return the
    current closest face (deterministic, within-band — NOT necessarily the analytic optimum; this is the honest
    EPA caveat).
  - **Closest-face selection by min `face.dist` with a FIXED tie-break** (lowest face index — the convex.h:28 /
    GJ2 idiom). `face.dist` is the integer perpendicular origin-distance `dot(face.normal, face.vertex)` with
    `face.normal` a UNIT (`FxNormalize`) outward normal.
  - **Consistent winding:** every face's normal points AWAY from the polytope interior (the origin is interior,
    so `dot(normal, faceVertex) ≥ 0`); flip the winding deterministically at face creation if not.
  - **Pinned horizon construction:** when removing visible faces, collect horizon edges in a FIXED traversal
    order; an edge shared by two removed faces is interior (cancel it) — use a deterministic edge-cancel rule
    (the std EPA "if the reversed edge is already in the list, remove it, else add it"), and walk the resulting
    horizon in a FIXED order so the re-triangulation is identical CPU/GPU.
  - **Degenerate-face guard:** a near-zero face normal (`FxLength` of the raw cross `< kEdgeEps`, the convex.h:130
    precedent) is handled deterministically (skip / do not select).
  - **Duplicate-vertex guard:** a new support already in the polytope (exact integer equality) → converged (no
    progress).
  - **Fixed-size buffers:** `kMaxPolyVerts` / `kMaxPolyFaces` (e.g. 64 / 128) — the GPU has no dynamic allocation;
    if expansion would exceed the cap, return the current closest face (deterministic, documented).
- **`Epa` returns** `{depth (the penetration depth, ≥ 0), normal (the unit contact normal A→B — the direction to
  push B off A), contactA, contactB (the witness contact points on each hull, via the barycentric projection of
  the origin onto the closest face applied to vertsA/vertsB), featureFaceId (a deterministic id: the closest
  face's vertex indices packed, the ContactKey raw material), iterations, valid}`.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **GJ2 `engine/sim/gjk.h` (read it; APPEND only after `Gjk`, before the namespace close):** `Simplex` (the EPA
  seed — its `pts`/`csoA`/`csoB`), `GjkResult` (the `overlap` gate + `simplex`), `SupportMinkowski` (the ONLY
  geometry call), `FxVec3Eq` (the duplicate guard), `kGjkMaxIter`. GJ1+GJ2 byte-frozen.
- **convex.h (read-only):** `convex::FxCross` (face normals), `fpx::FxNormalize` (unit normals), `fpx::FxLength`
  (the degenerate-face guard), `convex::FxDot`, `convex::kEdgeEps` (convex.h:130 — the degeneracy epsilon
  precedent). Define `kEpaTol` as a documented small Q16.16 convergence epsilon (the `kEdgeEps`/`kFacePrefEps`
  honesty lineage).
- **The proof-tier convention (convex.h:16-23):** int64 (face normals/distances + the barycentrics) →
  `shaders/gjk_epa.comp.hlsl` is **VULKAN-ONLY** (NOT in `hf_gen_msl`); Metal `--gjk-epa` runs the CPU `Epa` →
  byte-identical by construction; the Vulkan side carries the GPU==CPU memcmp. The shader copies `Epa` verbatim;
  one GPU thread per overlapping hull-pair. (NOTE: EPA's fixed-size polytope buffers + bounded loop fit a single
  thread — the convex_step.comp single-thread convention.)
- **The showcase + shader precedent:** GJ2's `--gjk-distance-shot` (int64 compute proof + GPU==CPU memcmp +
  diagnostic). Mirror for `--gjk-epa`.
- **Registration:** `scripts/verify.ps1` (append `gjk_epa` + `--gjk-epa-shot` to `$vkShots`),
  `metal_headless/CMakeLists.txt` (`hf_gen_msl` — do NOT add `gjk_epa.comp`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden — do NOT**), append to `tests/gjk_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/gjk.h`** (GJ1+GJ2 byte-frozen): `kEpaMaxIter`, `kMaxPolyVerts`, `kMaxPolyFaces`,
   `kEpaTol`, `PolyFace`, `Polytope`, `EpaResult`, `Epa(...)` + an `EpaMeasure`/`MeasureEpa` summary. Pure
   integer, FIXED face/horizon/iteration order. NO new RHI; one new Vulkan-only shader.
2. **New shader `shaders/gjk_epa.comp.hlsl` (int64, VULKAN-ONLY)** — copies `Epa` verbatim; one thread per
   overlapping hull-pair; writes the `EpaResult` to an SSBO. NOT in `hf_gen_msl`.
3. **Showcase `--gjk-epa-shot <out>` (Vulkan) AND `--gjk-epa` (Metal) — WIRE BOTH.** A fixed set of OVERLAPPING
   pairs with known/analytic penetration: a box pushed into a box along a face (known depth + axis normal), a
   box into a box on a corner, a tetra into a box, a couple rotated. For each, run `Gjk` (GJ2) to get the
   terminal simplex, then `Epa`. Vulkan dispatches `gjk_epa.comp` + memcmps GPU vs CPU `Epa`; Metal runs CPU
   `Epa`. BOTH render an integer diagnostic (the overlapping pairs + a marker/arrow per contact normal + depth).
   Golden = `tests/golden/metal/gjk_epa.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU:** `gjk-epa: {pairs:<P>, converged:<C>, maxIter:<M>} GPU==CPU BIT-EXACT` — the GPU
     `EpaResult` buffer == the CPU one byte-for-byte; assert.
   - **(2) determinism:** `gjk-epa determinism: two runs BYTE-IDENTICAL`.
   - **(3) correctness:** for the analytic cases, the EPA depth + normal match the known answer within the
     documented fixed-point band, and the normal is unit + points A→B (separating B from A by `depth` along
     `normal` removes the overlap — verify by a re-query: translating B by `depth·normal` makes the pair
     non-overlapping or touching per `Gjk`). Print `gjk-epa correct: {depthInBand:true, normalSeparates:true}`;
     assert. (EPA depth is within-band, NOT cross-algorithm byte-identical — the honest caveat; `Epa` itself IS
     byte-identical CPU/GPU.)
   - **Golden discipline: ONLY `tests/golden/metal/gjk_epa.png`; do NOT commit it.** Existing 205 image goldens
     UNTOUCHED.
5. **Cross-backend bar (int64 → strict):** Vulkan GPU==CPU bit-exact; Metal CPU-ref byte-identical by
   construction; cross-vendor ZERO differing pixels.
6. **Tests — APPEND to `tests/gjk_test.cpp`:** the analytic box-into-box face case returns the known depth+normal
   within band; the normal separates (a re-query after translating B by `depth·normal` is non-overlapping/touching);
   `Epa` is deterministic (two calls byte-equal); the polytope is valid (faces wound outward, the closest face's
   distance ≥ 0); a `kEpaMaxIter`/buffer-cap path returns `valid` with the current closest face (no crash, no
   UB). Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-gjk-epa` (features) + `--gjk-epa-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** The shot rides the existing compute-dispatch + SSBO path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/gjk.h` is APPEND-only (GJ1+GJ2 frozen); convex.h/fric.h/persist.h/fpx.h + ALL other sim headers +
  ALL existing shaders + `engine/physics/`/`nav/`/`anim/` UNCHANGED. NEW file: `shaders/gjk_epa.comp.hlsl` only.
  Report the seam: one new Vulkan-only shader, no RHI change, no frozen-file edit, gjk.h append-only.

## Out of scope (YAGNI — later slices)
The hull world step + manifold generation (GJ4 — EPA gives ONE contact point+normal+depth; GJ4 builds the
keyable manifold + swaps the box narrowphase), lockstep (GJ5), lit render (GJ6). GJ3 claims ONLY: a deterministic,
bit-exact (CPU↔Vulkan↔Metal) EPA penetration depth + contact normal + contact points for an overlapping convex
pair, with the integer golden + the three proofs. CAVEATS: convex polyhedra only; EPA termination is within-band
(fixed bound + integer tie-break → a deterministic depth/normal, NOT guaranteed analytically optimal — the honest
EPA caveat); fixed `kMaxPolyVerts`/`kMaxPolyFaces` cap hull/penetration complexity; `Epa` itself is byte-identical
CPU/GPU (the memcmp), the depth/normal are within-band vs an independent reference.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 108 incl. the appended GJ3 `gjk_test` cases). Clean under
   `windows-msvc-asan` (build+run `gjk_test` + `introspect_test`).
2. **proofs + visual:** `--gjk-epa-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID. **VERIFY the diagnostic shows the overlapping pairs with coherent contact normals/depths (no
   garbage/NaN, normals point sensibly).**
3. Metal: `visual_test --gjk-epa` → new golden `tests/golden/metal/gjk_epa.png`; two runs DIFF 0.0000. **Confirm
   `visual_test.mm` in the diff; confirm `gjk_epa.comp` is NOT in `hf_gen_msl`.** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `gjk_epa.png` added; the other 205
   byte-identical. `git diff master --stat -- tests/golden` = ONLY `gjk_epa.png` (metal) + the introspect json
   (controller rebake).
5. Introspect: exactly `+deterministic-gjk-epa` + `--gjk-epa-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + GJ1/GJ2 gjk.h code + convex.h/fric.h/persist.h/fpx.h + ALL other sim headers + ALL
   existing shaders byte-unchanged; gjk.h APPEND-only; one new Vulkan-only shader, no RHI change).
   `scripts/verify.ps1` updated; `gjk_epa.comp` NOT in `hf_gen_msl`.
