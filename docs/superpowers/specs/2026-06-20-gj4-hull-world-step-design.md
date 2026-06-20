# Slice GJ4 — General Convex-Hull Contacts: THE HULL WORLD STEP (the new-physics beat) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice (THE NEW-PHYSICS BEAT) of FLAGSHIP
> #22 (DETERMINISTIC GENERAL CONVEX-HULL CONTACTS via integer GJK + EPA, `hf::sim::gjk`). GJ1-GJ3 built the
> narrowphase trio (support → GJK overlap/distance → EPA depth/normal), each bit-exact CPU↔Vulkan↔Metal. GJ4 puts
> it to work: it builds a **general-hull rigid-body world step** that swaps the box-box SAT narrowphase for the
> GJK/EPA hull narrowphase inside the EXISTING `convex::StepConvexWorld` 5-pass shell — so arbitrary convex
> polyhedra (tetrahedra, octahedra, wedges) integrate, collide, and SETTLE, deterministically and bit-identically
> across backends. This is the payoff: physics the box-only SAT solver (`convex.h::BoxSat`) literally cannot
> represent — a **tetrahedron resting on its triangular face**, a **wedge interlocked against a box**. APPEND to
> `engine/sim/gjk.h` (GJ1-GJ3 + convex.h/fric.h/persist.h/fpx.h BYTE-FROZEN). Branch: `slice-gj4`. See
> [[hazard-forge-gjk-roadmap]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/gjk.h` (additive — GJ1-GJ3 byte-unchanged) with `FxHullInvInertiaBody` (the hull's
body-space inverse inertia — new integer tetra-decomposition math) + `HullContact` (Gjk→Epa→a
`convex::ContactManifold` with the hull feature id) + `HullWorld { bodies; hulls; }` + `StepHullWorld` /
`StepHullWorldN` (the `StepConvexWorld` shell with the ONLY swap being `BoxSatStable`→`HullContact`) +
`HullStackMeasure` / `MeasureHullStack`. Add the int64 GPU shader `shaders/hull_step.comp.hlsl` (Vulkan-only,
runs `StepHullWorldN` verbatim) + the showcase `--gjk-settle-shot` (Vulkan) / `--gjk-settle` (Metal). Bake the
integer golden `gjk_settle`.

## Design call: swap the narrowphase, reuse the solver — the hull step IS the convex step with one substitution

`convex::StepConvexWorld` (convex.h) is a deterministic 5-pass tick: (1) predict-integrate dynamic bodies (gravity
+ `IntegrateBodyFull`); (2) all-pairs narrowphase (`BoxSatStable`→`BuildManifold`); (3) world Gauss-Seidel impulse
(`WorldInvInertia` + `SolveManifoldImpulse`); (4) position de-penetration (push pairs apart along the contact
normal by `(pen−slop)·beta`, `posIters` sweeps); (5) orientation re-normalize + per-tick damping. GJ4 reproduces
this shell EXACTLY with ONE swap: the box narrowphase (steps 2 + the de-pen's pen/normal in 4) becomes the GJK/EPA
hull narrowphase. Steps 1, 3, 5 are reused VERBATIM (the integrator, the impulse solver, the orientation) — they
operate on `FxBody`, which a hull body has identically.
- **`HullWorld { std::vector<fpx::FxBody> bodies; std::vector<FxHull> hulls; };`** — mirrors `convex::ConvexWorld
  { bodies; boxes; }`; `hulls[i]` is body `i`'s collision hull (local-space verts, immutable/shared like a box's
  half-extents).
- **`HullContact(const fpx::FxBody& bodyA, const FxHull& hullA, const fpx::FxBody& bodyB, const FxHull& hullB) →
  convex::ContactManifold`** — run `Gjk(hullA, bodyA, hullB, bodyB)`; if NOT overlap → return an empty manifold
  (count 0). If overlap → `Epa(...)` for the depth+normal+contact, and build a `convex::ContactManifold`: normal =
  the EPA normal, one contact point = the EPA contact (midpoint of `contactA`/`contactB`, or `contactA` — pick
  and document), penetration = the EPA depth, **count = 1** (the scout's option (a): the single GJK/EPA point —
  simplest + deterministic; a face-resting tetra may ROCK on one point, the documented stability limit; GJ-future
  could add incident-face clipping for a multi-point manifold). Map the EPA `featureFaceId` into the manifold's
  `axisIndex`/`featureIndex` slots (the `convex::SatResult` feature analog — keeps the manifold ContactKey-able
  for a future warm-start hull path). **Reuse `convex::ContactManifold` verbatim** — do NOT define a new manifold
  type; this keeps the downstream solver/de-pen bit-compatible.
- **`FxHullInvInertiaBody(const FxHull& hull, fx invMass) → FxVec3`** — the hull's DIAGONAL body-space inverse
  inertia (the `FxBoxInvInertiaBody` analog, convex.h:606). Compute the diagonal inertia `(Ixx, Iyy, Izz)` about
  the body origin via a **tetrahedral decomposition** of the hull (or, simpler and exact for the symmetric
  canonical hulls, from the vertex second-moments) in Q16.16 integer math, then `invIbody = (3·invMass-form
  reciprocals)` — document the exact formula. **DESIGN DECISION (locked):** use the DIAGONAL inertia only
  (drop the products of inertia) and reuse `convex::WorldInvInertia(body, invIbody)` (which builds `R·diag·Rᵀ`).
  This is EXACT for the canonical test hulls (tetra/octa/box/wedge built symmetric about the body origin → the
  body inertia tensor is diagonal by symmetry) and a documented approximation for a general asymmetric hull (the
  off-diagonal-dropped caveat — a full symmetric-3×3 integer inertia + inverse is a future refinement; the
  diagonal keeps GJ4 reusing `WorldInvInertia` verbatim and avoids a fixed-point 3×3 matrix inverse). A STATIC
  body (invMass==0) → (0,0,0).
- **`StepHullWorld(HullWorld& world, const convex::ConvexStepConfig& cfg)`** — the `StepConvexWorld` 5-pass shell
  with `BoxSatStable`→`HullContact`: (1) predict-integrate (reuse the `IntegrateBodyFull` + gravity path); (2)
  all-pairs (i<j fixed order) `HullContact` → the manifolds; (3) `FxHullInvInertiaBody`+`WorldInvInertia`+
  `convex::SolveManifoldImpulse` (verbatim); (4) position de-pen along the manifold normal by `(pen−slop)·beta`,
  `posIters` sweeps, fixed i<j order; (5) orientation re-normalize + `linDamp`/`angDamp`. Pure integer, FIXED
  order. **`angDamp` is the stability knob** (as in the box step, convex.h:756) — the single-point manifold leaves
  a residual torque; a mild angular drag bleeds the spurious spin so the hull RESTS (document the value used).
  `StepHullWorldN(world, cfg, ticks)` runs `ticks` steps.
- **`HullStackMeasure { fx maxSpeed; fx maxPenetration; uint32_t dynamicCount; }`** + `MeasureHullStack` — the
  `convex::StackMeasure` analog (max dynamic speed = the rest test; max penetration over all pairs via
  `HullContact` = the held test). Pure integer.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **GJ1-GJ3 `engine/sim/gjk.h` (read it; APPEND only after `Epa`/`MeasureEpa`, before the namespace close):**
  `FxHull`, `Gjk`/`GjkResult`, `Epa`/`EpaResult`, `Support`, the `using` aliases. GJ1-GJ3 byte-frozen.
- **convex.h (read-only — REUSE verbatim, do NOT redefine):** `convex::ConvexStepConfig` (convex.h:749 — the
  step params), `convex::ContactManifold` (convex.h:292 — what HullContact produces), `convex::SolveManifoldImpulse`
  (convex.h:651), `convex::WorldInvInertia` (convex.h:622 — takes a diagonal `invIbody`), `convex::FxMat3`,
  `convex::StackMeasure`/`IsDynamic` (convex.h:778/786), `fpx::IntegrateBodyFull` (the integrator — confirm its
  exact signature + the gravity/predict path `StepConvexWorld` uses; mirror it). `FxBoxInvInertiaBody`
  (convex.h:606) is the formula template for `FxHullInvInertiaBody`.
- **Read `StepConvexWorld` in full** (convex.h, the 5-pass body) — `StepHullWorld` mirrors it line-for-line with
  the narrowphase swap. The de-pen pen/normal come from `HullContact`'s manifold instead of `BoxSat`.
- **The proof-tier convention (convex.h:16-23):** int64 → `shaders/hull_step.comp.hlsl` is **VULKAN-ONLY** (NOT in
  `hf_gen_msl`); Metal `--gjk-settle` runs the CPU `StepHullWorldN` → byte-identical by construction; the Vulkan
  side carries the GPU==CPU memcmp. The shader runs `StepHullWorldN` single-thread (the `convex_step.comp`
  convention — one thread does the whole world step; a settling scene of a few bodies fits).
- **The showcase + shader precedent:** `convex::StepConvexWorld`'s `--convex-stack-shot` (the settling-stack
  compute proof + GPU==CPU memcmp + the `StackMeasure` rest/pen assertions + the side-view render). Mirror for
  `--gjk-settle`.
- **Registration:** `scripts/verify.ps1` (append `gjk_settle` + `--gjk-settle-shot` to `$vkShots`),
  `metal_headless/CMakeLists.txt` (`hf_gen_msl` — do NOT add `hull_step.comp`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden — do NOT**), append to `tests/gjk_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/gjk.h`** (GJ1-GJ3 byte-frozen): `FxHullInvInertiaBody`, `HullContact`, `HullWorld`,
   `StepHullWorld`, `StepHullWorldN`, `HullStackMeasure`, `MeasureHullStack`. Pure integer, FIXED order. NO new
   RHI; one new Vulkan-only shader. **Diagonal hull inertia reusing `WorldInvInertia`** (no 3×3 integer inverse).
   **Single-point manifold (count 1)** from EPA (the documented rock-on-one-point limit).
2. **New shader `shaders/hull_step.comp.hlsl` (int64, VULKAN-ONLY)** — runs `StepHullWorldN` verbatim, single
   thread; reads the body+hull world from SSBOs, writes the stepped bodies back. NOT in `hf_gen_msl`.
3. **Showcase `--gjk-settle-shot <out>` (Vulkan) AND `--gjk-settle` (Metal) — WIRE BOTH.** The headline scene: a
   static floor (a large box-hull) + a **tetrahedron** + an **octahedron** dropped to rest ON A FACE, and a
   **wedge** interlocked against an upright box, settled over N ticks (e.g. 240). Vulkan dispatches
   `hull_step.comp` + memcmps the GPU stepped world vs the CPU `StepHullWorldN`; Metal runs the CPU path. BOTH
   render the settled world (a side-view — reuse the convex stack render style; the hulls drawn as their
   projected footprints / a simple wire is fine for this integer golden, GJ6 is the lit 3D capstone). Golden =
   `tests/golden/metal/gjk_settle.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU:** `gjk-settle: {bodies:<B>, ticks:<K>} GPU==CPU BIT-EXACT` — the GPU final body world == the
     CPU `StepHullWorldN` byte-for-byte; assert.
   - **(2) determinism:** `gjk-settle determinism: two runs BYTE-IDENTICAL`.
   - **(3) the new physics — rest + held:** `gjk-settle rest: {maxSpeed:<v>, maxPen:<p>, restedOnFace:true}` —
     the dynamic hulls came to REST (`maxSpeed` below a documented rest band) and are HELD (`maxPenetration`
     within `slop` + a documented band, NOT sunk through the floor / each other) after N ticks; assert. The scene
     exercises a tetra/octa resting on a FACE — physics the box-only SAT cannot represent.
   - **Golden discipline: ONLY `tests/golden/metal/gjk_settle.png`; do NOT commit it.** Existing 206 image
     goldens UNTOUCHED.
5. **Cross-backend bar (int64 → strict):** Vulkan GPU==CPU bit-exact; Metal CPU-ref byte-identical; cross-vendor
   ZERO differing pixels.
6. **Tests — APPEND to `tests/gjk_test.cpp`:** `FxHullInvInertiaBody` matches `FxBoxInvInertiaBody` for a box
   hull (the cube-hull diagonal inertia equals the analytic box inertia — the cross-check) and is (0,0,0) for a
   static; `HullContact` returns a count-1 manifold with the EPA normal/penetration for an overlapping pair, an
   empty manifold for a separated pair; `StepHullWorldN` brings a dropped tetra to rest on the floor (maxSpeed
   below band, maxPen within band) over N ticks; the step is deterministic (two runs byte-equal). Clean under
   `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-hull-step` (features) + `--gjk-settle-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** The shot rides the existing compute-dispatch + SSBO path (the `convex_step` shot's seam). `rhi.h` +
  backend dirs UNCHANGED. `engine/sim/gjk.h` is APPEND-only (GJ1-GJ3 frozen); convex.h/fric.h/persist.h/fpx.h +
  ALL other sim headers + ALL existing shaders + `engine/physics/`/`nav/`/`anim/` UNCHANGED. NEW file:
  `shaders/hull_step.comp.hlsl` only. Report the seam: one new Vulkan-only shader, no RHI change, no frozen-file
  edit, gjk.h append-only.

## Out of scope (YAGNI — later slices)
Lockstep + rollback (GJ5 — the HullWorld is pure data, the CX5 harness retargets to it for free), the lit 3D
render capstone (GJ6 — a pile of mixed polyhedra, lit). Friction + warm-start on hull contacts (the manifold is
ContactKey-able via the feature id, but wiring `fric`/`persist` over `HullContact` is a future composition slice —
GJ4 is the frictionless deterministic settle, the `convex::StepConvexWorld` analog). The OPTIONAL int32-MSL-native
`HullFeatureKey` rider (a pure-int32 pack of the EPA feature id into a `persist::ContactKey`) MAY be added here if
cheap, but is not required. GJ4 claims ONLY: a deterministic, bit-exact (CPU↔Vulkan↔Metal) general-hull rigid-body
settle, with the integer golden + the three proofs. CAVEATS: convex polyhedra only; single-point manifold (a
face-resting hull may rock within the angular-damp band — the documented stability limit); diagonal hull inertia
(off-diagonal dropped — exact for the symmetric canonical hulls); the EPA within-band caveats inherited.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 108 incl. the appended GJ4 `gjk_test` cases). Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--gjk-settle-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID. **VERIFY the image shows a coherent settled scene — a tetra/octa at rest on the floor, a wedge
   interlocked, no sinking / explosion / NaN.**
3. Metal: `visual_test --gjk-settle` → new golden `tests/golden/metal/gjk_settle.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `hull_step.comp` is NOT in `hf_gen_msl`.** Cross-vendor STRICT
   ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `gjk_settle.png` added; the other
   206 byte-identical. `git diff master --stat -- tests/golden` = ONLY `gjk_settle.png` (metal) + the introspect
   json (controller rebake).
5. Introspect: exactly `+deterministic-hull-step` + `--gjk-settle-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + GJ1-GJ3 gjk.h code + convex.h/fric.h/persist.h/fpx.h + ALL other sim headers + ALL
   existing shaders byte-unchanged; gjk.h APPEND-only; one new Vulkan-only shader, no RHI change).
   `scripts/verify.ps1` updated; `hull_step.comp` NOT in `hf_gen_msl`.
