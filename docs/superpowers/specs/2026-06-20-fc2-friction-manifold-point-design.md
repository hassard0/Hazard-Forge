# Slice FC2 — Deterministic Contact Friction: THE FRICTION-AUGMENTED MANIFOLD POINT — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #20
> (DETERMINISTIC TANGENTIAL CONTACT FRICTION, `hf::sim::fric`). FC1 built the deterministic tangent basis at a
> contact normal. FC2 builds the **per-contact solver state**: for a box-box pair, take the frozen CX2 contact
> manifold and pair every contact point with its FC1 tangent basis + zeroed (warm-start-ready) impulse
> accumulators, into a `FrictionPoint[]` array — the structure FC3's cone-clamped tangent-impulse solver consumes.
> INTEGER-bit-exact. int64 → the `fric_points.comp` shader is Vulkan-only + a Metal CPU reference (the FC1 split).
> FC1's `fric.h` code + CX1-CX6's `convex.h` are BYTE-FROZEN (FC2 is additive). Branch: `slice-fc2`. See
> [[hazard-forge-fric-roadmap]].

**Goal:** Extend `engine/sim/fric.h` (additive — FC1 + convex.h byte-unchanged) with `FrictionPoint` (a contact
point + normal + tangent basis + impulse accumulators) + `BuildFrictionPoints(bodyA, boxA, bodyB, boxB)` (run the
frozen `convex::BoxSat`/`BoxSatStable` + `convex::BuildManifold`, then attach the FC1 tangent basis + zeroed
accumulators per point) + a measure. Add the new int64 shader `shaders/fric_points.comp.hlsl` +
`--fric-points-shot` (Vulkan) / `--fric-points` (Metal). Bake the integer golden `fric_points`. **NO new RHI.**

## Design call: the per-contact friction solver state over the frozen manifold

CX2's `ContactManifold` already gives 1-4 contact points + the contact normal (sign-corrected A→B in the solver).
FC2 turns each into a `FrictionPoint` — the state FC3 will iterate:
- **`FrictionPoint { FxVec3 point; FxVec3 normal; FxVec3 t1; FxVec3 t2; fx normalImpulse; fx tangentImpulse1;
  fx tangentImpulse2; }`** — the contact point + the A→B contact normal + the FC1 orthonormal tangent basis
  (`MakeTangentBasis(normal)`) + three accumulators (the normal + two tangent impulses, ALL ZEROED at build — the
  warm-start hooks FC3 fills; FC2 establishes the structure, the baseline re-solves from zero each tick).
- **`BuildFrictionPoints(bodyA, boxA, bodyB, boxB)` → `FrictionManifold`** (a fixed-capacity `count` +
  `FrictionPoint pts[4]`): run `convex::BoxSatStable(bodyA, boxA, bodyB, boxB)` (the CX4 face-preference SAT, frozen)
  → if no overlap return `{count = 0}` → `convex::BuildManifold(...)` → for each manifold point `i`:
  `pts[i].point = manifold.points[i]`; `pts[i].normal = nAB` (the manifold normal sign-corrected to point A→B ONCE,
  exactly the `SolveManifoldImpulse` rule — flip if `FxDot(manifold.normal, FxSub(bodyB.pos, bodyA.pos)) < 0`);
  `(t1, t2) = MakeTangentBasis(nAB)`; the three impulse accumulators = 0. `count = manifold.count`. PURE INTEGER,
  FIXED order → bit-identical CPU↔Vulkan↔Metal.
- **`MeasureFrictionPoints(pairs)`** → the deterministic summary: total pairs, pairs with contact, total points, and
  the max basis orthogonality residual (max `|n·t1|`/`|n·t2|`/`|t1·t2|`) over all points — pure integer, the
  showcase prints + asserts.

**THE int64 REALITY (the FC1/CX1 lesson):** the manifold clip + the tangent-basis `FxNormalize`/`FxDot`/`FxCross`
products are int64. DXC compiles int64 (Vulkan); glslc cannot. So `fric_points.comp` is **VULKAN-SPIR-V-ONLY (NOT
in hf_gen_msl)**; the Metal `--fric-points` runs the CPU `BuildFrictionPoints` — byte-identical to the Vulkan GPU
result BY CONSTRUCTION (the FC1 convention), while the Vulkan side carries the GPU==CPU memcmp proof.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **FC1 `engine/sim/fric.h` (read it; APPEND only after `MeasureBasis`):** `TangentBasis`, `MakeTangentBasis(n)`,
  `MeasureBasis`, and the pulled `fx`/`FxVec3`/`FxDot`/`FxCross`. FC2 calls `MakeTangentBasis`. FC1's lines
  byte-frozen.
- **convex.h (read-only — do NOT edit):** `convex::BoxSatStable` (the CX4 face-preference SAT, `convex.h:807`),
  `convex::BoxSat`/`SatResult` (`convex.h:169`), `convex::BuildManifold`/`ContactManifold` (`convex.h:372`, the
  `.normal`/`.points[]`/`.count`), `convex::FxBox`, `FxBody`. The A→B sign rule is `SolveManifoldImpulse`'s
  (`convex.h:654-656`). FC2 reuses these read-only.
- **fpx.h (read-only):** `FxSub`/`FxDot`, `FxBody`, `FxVec3`. **DO NOT modify fpx.h or convex.h.**
- **The shader + showcase precedent:** FC1's `shaders/fric_basis.comp.hlsl` (the int64 Vulkan-only one-thread-per-
  element pattern + the `*Gpu` pack + the VERBATIM CPU copy) and CX2's `shaders/convex_manifold.comp.hlsl` (the
  manifold-build-in-a-shader pattern — `fric_points.comp` runs `BoxSatStable`+`BuildManifold`+`MakeTangentBasis`
  per pair VERBATIM). The `--fric-basis-shot` / CX2 `--convex-manifold-shot` Vulkan showcases + the Metal
  `--fric-basis`/`--convex-manifold` blocks. Mirror these. Confirm `fric_points` NOT in hf_gen_msl.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/fric_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/fric.h`** (FC1 byte-frozen): `FrictionPoint`, `FrictionManifold` (count + `pts[4]`),
   `BuildFrictionPoints(bodyA, boxA, bodyB, boxB)`, `MeasureFrictionPoints(pairs)`. Pure integer, FIXED order, the
   A→B normal sign-correction done ONCE per pair. **NEW shader** `fric_points.comp.hlsl` (int64, Vulkan-only, one
   thread per box pair — runs `BoxSatStable`+`BuildManifold`+`MakeTangentBasis` VERBATIM, writes the
   `FrictionPoint[]` per pair). NOT in hf_gen_msl; Metal runs the CPU path.
2. **Showcase `--fric-points-shot <out>` (Vulkan) AND `--fric-points` (Metal) — WIRE BOTH** (standalone arg-parse).
   The SCENE: REUSE the SAME fixed deterministic box-pair array as CX2's `--convex-manifold` (separated / deep
   face-face / edge-edge / touching). Vulkan: the GPU `fric_points.comp` → **memcmp the GPU `FrictionPoint[]` vs the
   CPU `BuildFrictionPoints`** (NO tolerance). Metal: the CPU reference. Render a PURE-INTEGER 2D top-down view: the
   box footprints (as CX2) PLUS each contact point as a small marker with its `t1`/`t2` tangents drawn as short
   segments (the contact tangent frame). Golden = `tests/golden/metal/fric_points.png` (Mac-baked by the CONTROLLER
   — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU `FrictionPoint[]` == the CPU `BuildFrictionPoints` byte-for-byte. Print
     `fric-points: {pairs:<N>, withContact:<M>, points:<P>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `fric-points determinism: two runs BYTE-IDENTICAL`.
   - **(3) basis correct:** every friction point's `(t1, t2)` is orthonormal to its `normal` within the integer
     epsilon (max `|n·t1|`/`|n·t2|`/`|t1·t2|` small; `FxLength(t1)`/`FxLength(t2)` ~ kOne); every point count equals
     the CX2 manifold count for that pair; all accumulators are zero at build. Print `fric-points correct:
     {basisOrthonormal:true, countsMatchManifold:true, accumulatorsZero:true}`; assert all.
   - **(4) normal A→B:** every friction point's `normal` points from A toward B (`FxDot(normal, B.pos − A.pos) ≥ 0`)
     for the overlapping pairs. Print `fric-points normal: {allPointAtoB:true}`; assert.
   - **Golden discipline: ONLY `tests/golden/metal/fric_points.png`; do NOT commit it.** Existing 190 image goldens
     UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests — APPEND to `tests/fric_test.cpp` (pure CPU):** a deep face-face pair → a `FrictionManifold` whose count
   matches `BuildManifold`, every point's basis orthonormal to the A→B normal, all accumulators zero; a separated
   pair → count 0; an edge-edge pair → count 1 with a valid basis; the normal points A→B; two runs byte-identical.
   Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-friction-points` (features) + `--fric-points-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/convex.h` + `fpx.h` + **FC1's fric.h code + fric_basis.comp** + all other sim headers + `engine/nav/`
  + `engine/anim/` + `engine/physics/` + all EXISTING shaders UNCHANGED. The ONLY new shader is
  `fric_points.comp.hlsl` (int64, Vulkan-only, NOT in hf_gen_msl). `fric.h` APPEND-only. Report the seam empty.

## Out of scope (YAGNI — later FC slices)
The cone-clamped tangent-impulse solve (FC3 — FC2 only BUILDS the per-contact state; the accumulators stay zero, no
impulse is applied), the friction-locked world step (FC4), lockstep (FC5), the lit 3D render (FC6). Cross-tick
warm-starting (FC2 leaves the accumulator fields as hooks but the baseline zeroes them each build — a documented
refinement). FC2 claims ONLY: a deterministic integer `FrictionPoint[]` (contact point + A→B normal + orthonormal
tangent basis + zeroed accumulators) per box-box pair, bit-identical CPU↔Vulkan↔Metal, with the integer golden +
the four proofs. NOTE: boxes only (inherits the box-box manifold); int64 → Vulkan-GPU + Metal-CPU-ref (the FC1/CX1
proof strength).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 106 incl. FC1's `fric_test` + the appended FC2 cases). Clean
   under `windows-msvc-asan` (build+run `fric_test` + `introspect_test`).
2. **proofs + visual:** `--fric-points-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the box pairs WITH a contact tangent frame (t1/t2 segments) at each contact point on the overlapping pairs.**
3. Metal: `visual_test --fric-points` → new golden `tests/golden/metal/fric_points.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `fric_points.comp` NOT in `hf_gen_msl`.** Cross-vendor STRICT
   ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fric_points.png` added; the other
   190 byte-identical. `git diff master --stat -- tests/golden` = ONLY `fric_points.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-friction-points` + `--fric-points-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/convex.h`/`fpx.h` + **FC1's fric.h code + fric_basis.comp** + ALL
   other sim headers + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL existing shaders byte-unchanged).
   `scripts/verify.ps1` updated: `fric_points` golden in the Mac loop + `--fric-points-shot` in `$vkShots`. **The
   ONLY new shader is `fric_points.comp.hlsl` (int64, NOT in `hf_gen_msl`).**
