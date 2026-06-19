# Slice FR4 — Deterministic Fracture/Destruction: THE FRACTURE STEP (released fragments fall) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #14 (DETERMINISTIC
> RIGID-BODY FRACTURE / DESTRUCTION, `hf::sim::fract`). FR1-FR3 made the cells, the fragments, and the break;
> FR4 turns the break into MOVING RUBBLE — the dislodged pieces become independent `fpx::FxBody` rigid bodies
> that fall under gravity, collide, and settle into a coherent pile, while the largest (anchor) piece holds. The
> object SHATTERS and the chunks fall — bit-identical CPU↔Vulkan↔Metal. **NO new shader** — a host-driven
> multi-pass driver over the EXISTING `fpx.h` rigid solver (the CP4/CG4/GF4 mold). Branch: `slice-fr4`. See
> [[hazard-forge-fract-roadmap]].

**Goal:** Extend `engine/sim/fract.h` (additive — FR1/FR2/FR3 byte-unchanged) with the world-unit spawn
(`SpawnFractWorld`) + the fracture step (`StepFracture`/`StepFractureSteps` + a `MeasureFractRubble` honest-metrics
helper). The step composes FR3's break with the existing `fpx::FxWorld` solver (`IntegrateBodyFull` + the FPX2
broadphase + FPX3 `SolveContacts` + `ResolveGround`). Add `--fract-step-shot` (Vulkan) / `--fract-step` (Metal).
Bake the integer golden `fract_step`. **NO new shader, NO new RHI.**

## Design call: spawn one FxBody per fragment (world-unit), anchor the largest piece, step via fpx VERBATIM
FR4 is the GR/CP/CG/GF-arc's coupled-step beat applied to fracture: a host-driven driver that calls the
ALREADY-bit-exact `fpx.h` rigid solver. The fragments become rigid bodies; the break decides which fall.

**(1) World-unit spawn (`SpawnFractWorld`).** From the FR2 fragments + FR3 break, build an `fpx::FxWorld`: one
`FxBody` per fragment, with `pos = fragment.centroid · kWorldCellSize` (the lattice→world Q16.16 scale — the
ONLY float-free integer up-conversion, host-fixed), `radius = fragment.boundRadius · kWorldCellSize`, `invMass`
from `fragment.invMass` (FR2), and the **anchor rule**: the LARGEST piece (most fragments; lowest cluster-id
tie-break — deterministic) is STATIC (`invMass = 0`, NOT `kFlagDynamic` — the object's intact base); every OTHER
piece's fragments are DYNAMIC (`kFlagDynamic`, the dislodged chunks). Seed the impacted fragment's body with an
initial impact velocity (`vel = impactDir · impactSpeed`, host-fixed). Pure integer (the scale is `fxmul` by a
constant). This is where the FR1-FR3 integer-lattice world finally becomes a Q16.16 world-unit body set.

**(2) The fracture step (`StepFracture`, no new shader — the CP4/CG4/GF4 driver).** One deterministic tick over
the spawned world: (a) `IntegrateBodyFull(each dynamic body)` — the FPX4 6-DOF integrator (gravity + linear +
ORIENTATION/tumble), reused VERBATIM; (b) `CountPairs`/`BuildPairs` — the FPX2 integer broadphase (rebuilt each
tick); (c) `SolveContacts(world, pairs, solveIters)` — the FPX3 sphere-sphere non-penetration solve, reused
VERBATIM; (d) `ResolveGround(each dynamic body, groundY)` — the floor clamp. The dynamic chunks fall, collide as
spheres, and settle on/around the static anchor + the ground. Over K steps → a coherent rubble pile. Pure
integer (fpx is bit-exact), fixed op order → two-run bit-identical AND bit-exact GPU==CPU. The GPU showcase
drives the EXISTING fpx int64 step shaders over the fragment bodies (the `--fpx-pile`/`--fpx`-step machinery,
NO new shader) and memcmp's the final body set vs the CPU `StepFracture`.

**(3) HONEST REUSE (the documented simplification).** FR3's `SolveContacts` is **sphere-sphere** (no convex
manifold), so each fragment collides as its **bounding sphere** (`fragment.boundRadius`) — the rubble is a pile
of rounded chunks, NOT interlocking shards. This is the `couple.h` body-as-sphere precedent, exactly the scout's
documented first cut. True convex-convex manifold contact (SAT/clipping + an inertia tensor + torque-from-contact)
does not exist in `engine/sim` and is the deferred refinement — out of scope. The anchor-largest-piece rule is a
simplification too (a real engine picks the ground-attached piece); document both.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The fpx rigid solver to REUSE VERBATIM (`engine/sim/fpx.h`):** `FxBody{pos,vel,invMass,flags,radius,orient,
  angVel}` (fpx.h:116), `FxWorld{gravity,groundY,bodies}` (fpx.h:135), `kFlagDynamic` (133), `IntegrateBodyFull`
  (fpx.h:479 — the FPX4 6-DOF integrate), `CountPairs`/`BuildPairs` (fpx.h:239/264 — the FPX2 broadphase),
  `SolveContacts` (fpx.h:360 — the FPX3 sphere contact solve), `StepWorld` (fpx.h:372 — read it for the exact
  per-tick composition to mirror, but FR4 uses `IntegrateBodyFull` for the 6-DOF tumble instead of
  `IntegrateStep`), `ResolveGround` (fpx.h:329), `FxBodyTransform` (fpx.h:627 — carried for FR6). `fxmul`/`fx`/
  `kOne`/`kFrac`. **DO NOT modify fpx.h** — FR4 only CALLS it.
- **The CP4/CG4/GF4 coupled-step driver to MIRROR (`engine/sim/couple_grain.h::StepCGrain` / `couple_gf.h::
  StepCGF`):** the host-driven "predict → broadphase once → solve → integrate → ground" tick + the
  `StepXxxSteps(world, …, steps)` loop + the `MeasureXxxState` honest-metrics helper. FR4's `StepFracture` is the
  same shape over the spawned `fpx::FxWorld`.
- **FR1/FR2/FR3 (this branch's `fract.h`, read-only — build on, DON'T modify):** `FractFragments`
  (`fragments[]` with `cx/cy/cz` centroid, `boundRadius`, `volume`, `invMass`), `BuildFractBonds`/`FractBonds`,
  `ApplyImpactBreak`/`BreakImpact`, `CountFractPieces` (→ the per-fragment `clusterId` + piece count for the
  anchor rule).
- **The host-driven GPU step proof (the CG4/GF4 `--cgrain-step`/`--cgf-step` showcases):** how the GPU side
  drives the existing int64 shaders + memcmp's vs the CPU reference. FR4 reuses the fpx GPU step path.
- **Showcase + registration:** FR1-FR3's `--fract-*-shot` plumbing — **standalone arg-parse loop** (the FR1
  C1061 lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the
  introspect JSON golden**), `tests/fract_test.cpp`.

## Design decisions (locked)
1. **`SpawnFractWorld(fragments, bonds, severed, pieceClusters, impact, cfg) → fpx::FxWorld`** — the world-unit
   body spawn + the anchor rule (largest piece static, others dynamic) + the impact velocity seed. `cfg` carries
   `kWorldCellSize`, `gravity`, `groundY`, `impactDir`/`impactSpeed`. Pure integer.
2. **`StepFracture(world, dt, solveIters)`** — the per-tick driver (IntegrateBodyFull → broadphase → SolveContacts
   → ResolveGround), `StepFractureSteps(world, dt, solveIters, steps)` the K-step loop. NO new shader; reuses fpx
   VERBATIM. `MeasureFractRubble(world, anchor)` → the rest line of the dynamic chunks (mean pos.y), the anchor
   pos.y, the settled/airborne counts — deterministic Q16.16 stats.
3. **Showcase `--fract-step-shot <out>` (Vulkan) AND `--fract-step` (Metal) — WIRE BOTH** (standalone arg-parse).
   The FR1-FR3 scene → break → spawn → run K `StepFracture` ticks → render the settled bodies (a 2D side view,
   each body a disc at `pos`, coloured by piece id; the static anchor distinct). Vulkan: the GPU fpx step over
   the fragment bodies → **memcmp vs the CPU `StepFracture`**. Metal: the CPU reference. Golden =
   `tests/golden/metal/fract_step.png` (Mac-baked by the CONTROLLER — DO NOT commit). SMALL/FAST scene (the
   fragment count is modest — 16 fragments — so K can be generous, e.g. 120 steps).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU body world after K steps == the CPU `StepFracture` reference
     byte-for-byte. Print `fract-step: {fragments:<F>, dynamic:<D>, anchor:<A>, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `fract-step determinism: two runs BYTE-IDENTICAL`.
   - **(3) break-and-fall vs intact control:** a HARD impact → `D > 0` dynamic chunks that FELL (their mean
     pos.y dropped from the spawn line) AND settled at/above ground; the anchor (static) is UNCHANGED. A SOFT
     impact (0 severed → 1 piece → all anchor) → `D == 0`, the object is a STATIC no-op (nothing falls). Print
     `fract-step rubble: hard={dynamic:<D>, fell:<true>, settled:<true>} soft={dynamic:0, static-noop}`; assert
     accordingly.
   - **(4) no body buried:** every dynamic body rests with `pos.y >= groundY` (the ground clamp held). Print
     `fract-step ground: all <D> chunks rest on/above the floor`.
   - **Golden discipline: ONLY `tests/golden/metal/fract_step.png`; do NOT commit it.** Existing 156 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels (fpx is
   already integer-bit-exact cross-backend; FR4 only sequences it).
6. **Tests `tests/fract_test.cpp` additions (pure CPU):** `SpawnFractWorld` — the anchor is the largest piece
   (static, invMass 0); other pieces dynamic; world positions = centroid·kWorldCellSize; the impacted body seeded.
   `StepFracture` — a dynamic body above the ground falls (pos.y decreases then clamps at groundY); two dynamic
   bodies dropped together collide (end ≥ their radii apart); a static anchor never moves; a soft-impact world
   (all static) is a no-op; two runs byte-identical. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-fract-step` (features) + `--fract-step-shot` (showcases). **REBAKE
   `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the CG4/GF4/fpx surface). `rhi.h` +
  backend dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple.h` + `couple_grain.h` +
  `couple_gf.h` + `engine/physics/` + all existing shaders UNCHANGED. FR1/FR2/FR3 `fract.h` code + the FR1-FR3
  shaders UNCHANGED (FR4 additive — only the spawn + the step + the showcase). **NO new shader.** Report the seam
  empty.

## Out of scope (YAGNI — later FR slices)
Lockstep/rollback (FR5 — FR4 is the forward step only), the lit 3D render (FR6 — FR4's golden is a 2D integer
state viz, FR6 is the lit money-shot). Convex-manifold contact / inertia tensor / torque-from-contact (the
sphere-bound simplification is documented). Re-fracture on landing, plastic deformation, multi-impact. The
anchor = ground-attached piece (FR4 uses largest-piece). FR4 claims ONLY: a deterministic forward fracture step
that spawns the broken fragments as fpx bodies and settles the dislodged chunks into a rubble pile, bit-identical
CPU↔Vulkan↔Metal, with the integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 100) + the new `fract_test` step cases. Clean under
   `windows-msvc-asan` (build+run `fract_test` + `introspect_test`).
2. **proofs + visual:** `--fract-step-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image
   shows the dislodged chunks fallen into a rubble pile around/on the anchor (pixel-check; the FR1-FR3 lesson).**
3. Metal: `visual_test --fract-step` → new golden `tests/golden/metal/fract_step.png`; two runs DIFF 0.0000 (gate
   on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (FR4 is pure
   orchestration — `hf_gen_msl` UNCHANGED; the FR1/FR2 int32 + FR3 int64 shaders unchanged).** Cross-vendor STRICT
   ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fract_step.png` added; the other
   156 byte-identical. `git diff master --stat -- tests/golden` = ONLY `fract_step.png` (metal) + the introspect
   json.
5. Introspect JSON rebaked exactly `+deterministic-fract-step` + `--fract-step-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h` +
   `engine/physics/` + FR1/FR2/FR3 `fract.h`/shaders byte-unchanged). `scripts/verify.ps1` updated: `fract_step`
   golden in the Mac loop + `--fract-step-shot` in `$vkShots`. **NO new entry in `hf_gen_msl`.**
