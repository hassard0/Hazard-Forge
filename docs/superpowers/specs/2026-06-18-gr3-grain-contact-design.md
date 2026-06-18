# Slice GR3 — Deterministic GPU Granular/Sand: FRICTIONLESS CONTACT PROJECTION (the FL4 Jacobi-solve twin) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #10
> (DETERMINISTIC GPU GRANULAR / SAND, `hf::sim::grain`). Resolves grain-grain OVERLAP (non-penetration) +
> ground/`fpx::FxBody` colliders by an inverse-mass-weighted NORMAL push-apart over the GR2 candidate list —
> the FL4 **JACOBI multi-thread** position-based solve, so there is NO single-thread `[numthreads(1,1,1)]`
> dispatch and NO GPU-watchdog/TDR particle-count ceiling (the cloth CL3 limit does NOT apply — the decisive
> reuse of FL4's design win). This is the grain pile's "incompressibility": dropped grains stop interpenetrating
> and settle into a heap. NO friction yet (GR4 — without it the pile spreads FLAT; friction makes it a cone).
> Branch: `slice-gr3`. See [[hazard-forge-grain-roadmap]].

**Goal:** Extend `engine/sim/grain.h` (additive — GR1/GR2 byte-unchanged) with `SolveGrainContact` (the
Jacobi per-grain Δp accumulate over the GR2 neighbor list), the radius-aware grain colliders
(`CollideGrainPlane` / `CollideGrainSphere` / `CollideGrainSpheres`, the CL4/FL4 mold + a
`GrainSphereFromBody(fpx::FxBody)` bridge), and `StepGrainContact` / `StepGrainContactSteps` (predict →
neighbors → K Jacobi contact iterations → velocity update → collide). Add `grain_contact_dp.comp.hlsl` +
`grain_contact_apply.comp.hlsl` + `grain_collide.comp.hlsl` (int64 → **Vulkan-only** + Metal CPU reference).
Add `--grain-contact-shot` (Vulkan) / `--grain-contact` (Metal). Bake the integer golden `grain_contact`.
NO new RHI.

## Design call: the JACOBI multi-thread solve (the FL4 win — NO TDR) + the INTEGER bar (strict zero-diff)
The contact correction is per-grain INDEPENDENT in the Jacobi formulation: each grain i reads the
iteration-START positions (read-only) and accumulates its OWN Δp_i into a SEPARATE `dp[]` double-buffer,
then ALL grains apply `pos_i += dp_i` (the FL4 `SolveDensityConstraint` + apply pattern). So the GPU solve
is `[numthreads(64,1,1)]` MULTI-THREAD — NO single-thread serial dispatch, NO TDR ceiling (the CL3 limit
does NOT apply; the FL4 lesson). The math is int64 (`FxLength`/`FxNormalize` via `FxISqrt`, `fxmul`,
`fxdiv`) → `grain_contact_dp.comp` + `grain_collide.comp` are **Vulkan-SPIR-V-ONLY** (glslc can't parse
int64), NOT in `hf_gen_msl`; the Metal `--grain-contact` showcase runs the CPU `StepGrainContact` (the SAME
bit-exact reference — byte-identical by construction, the FL4/GR1 convention). Bar: strict INTEGER (Vulkan
GPU == Metal CPU-ref == golden, ZERO differing pixels). The GR2 neighbor passes (rebuilt each step from the
predicted positions) stay int32 MSL-native.

## The contact constraint (the one new bit of math — the cloth-edge / FL4-density hybrid)
For an overlapping grain PAIR (i, j) with centre distance `d = |p_i − p_j|` and `pen = (r_i + r_j) − d > 0`:
push them apart along `n_ij = unit(p_i − p_j)` by their inverse-mass-weighted share. In the per-grain Jacobi
accumulate over the SYMMETRIC GR2 neighbor list (j is in i's list AND i is in j's list), each grain i
handles ITS OWN half independently:
```
Δp_i += ( w_i / (w_i + w_j) ) · pen · n_ij          // w = invMass; w_i+w_j==0 (both static) -> skip
```
(grain j independently accumulates its `Δp_j += (w_j/(w_i+w_j))·pen·n_ji` from the same pair — no
double-apply, the FL4 "each i sums over its neighbours" structure). This is the cloth `SolveDistanceConstraint`
per-pair projection (a `pen·n` push split by inverse mass) summed over the contact neighbours like FL4's
density. `d == 0` (coincident) → the `FxNormalize` +Y fallback (deterministic). Only pairs with `pen > 0`
contribute (the EXACT radial overlap cull GR2 deferred — non-overlapping candidates are a no-op). Static
grains (flags bit0 / invMass 0) → Δp = 0 (the cloth pinned / FL4 static case). int64 throughout.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The FL4 Jacobi solve to MIRROR (`engine/sim/fluid.h`):** `SolveDensityConstraint` (the Jacobi dp[]
  double-buffer accumulate, `fluid.h:660-687`) — GR3's `SolveGrainContact` is the SAME shape (per-grain
  accumulate over the neighbor list into a separate dp[], static→0) with the contact `pen·n` term instead of
  the `(λ_i+λ_j)∇W` term; `CollidePlane`/`CollideParticleSphere`/`CollideSpheres` (`fluid.h:700-736`) — the
  collider mold (make GR3's radius-aware, see decision #3); `StepFluid` (`fluid.h:754-787`) — the
  predict→neighbors→K-Jacobi-iters→velocity→collide driver GR3's `StepGrainContact` mirrors;
  `StepFluidSteps` (`fluid.h:793`) — the K-step driver; `kFluidCollideEps` (`fluid.h:696`) — the surface-snap
  tolerance. The FL4 residual caveat (`fluid.h` FL4 header / ARCHITECTURE.md:339) — Jacobi is iterative so
  the contact residual is deterministic-but-nonzero.
- **The cloth CL4 collider twins (`engine/sim/cloth.h`):** `SphereCollider`, `SphereFromBody(fpx::FxBody)`,
  `CollidePlane`/`CollideSpheres` — GR3 mirrors these to a `GrainSphereFromBody`. The grain/rigid share the
  same Q16.16 world (the CL4 deformable-meets-rigid precedent).
- **The GR1/GR2 substrate (read this branch's `engine/sim/grain.h`):** `GrainParticle` (pos/prev/vel/
  invMass/radius/flags), `IntegrateGrains` (the GR1 predict — reuse for StepGrainContact step 1),
  `MakeGrainGrid`/`BuildGrainCellTable`/`BuildGrainNeighborList` (the GR2 neighbor list — rebuild each step
  from the predicted positions). DO NOT modify the GR1/GR2 functions; GR3 is additive.
- **The Q16.16 toolbox (read-only, `engine/sim/fpx.h`):** `FxLength` (`fpx.h:319`), `FxNormalize`
  (`fpx.h:96`), `fxmul`/`fxdiv`, `FxISqrt`, `FxAdd`/`FxSub`/`FxScale`, plus `fpx::FxBody`. DO NOT modify
  fpx.h/cloth.h/fluid.h/engine/physics/.
- **The int64 solve SHADERS to mirror (`shaders/fluid_dp.comp.hlsl` + `fluid_apply.comp.hlsl` +
  `fluid_collide.comp.hlsl`):** the FL4 Jacobi multi-thread shaders — `grain_contact_dp`/`grain_contact_apply`/
  `grain_collide` are the SAME `[numthreads(64,1,1)]` structure, int64, Vulkan-only (NOT in `hf_gen_msl`).
- **Showcase + registration:** FL4's `--fluid-solve-shot` / `--fluid-solve` is the template (the multi-pass
  host driver: predict → GR2 neighbor passes → K×{dp→apply} → velocity → collide, a ComputeToComputeBarrier
  between sub-passes). GR1/GR2 grain showcases are the immediate plumbing template. `scripts/verify.ps1`,
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden** — the
  GR2 lesson: adding `--grain-contact-shot` changes the byte-exact `default_scene.json`), `tests/grain_test.cpp`.

## Design decisions (locked)
1. **`SolveGrainContact` (the Jacobi dp[] accumulate, the FL4 structure).** For each grain i (skip static):
   accumulate `Δp_i` over its GR2 neighbours j where `pen = (r_i+r_j) − |p_i−p_j| > 0`, by `Δp_i += (w_i/
   (w_i+w_j))·pen·unit(p_i−p_j)` (int64 `FxLength`/`FxNormalize`/`fxmul`/`fxdiv`), into a SEPARATE `dp[]`
   buffer (reads iteration-start positions — Jacobi). Apply `pos_i += dp_i` for ALL after. `grain_contact_dp.comp`
   copies the accumulate VERBATIM; `grain_contact_apply.comp` the apply. Deterministic (the fixed GR2
   neighbour order).
2. **`StepGrainContact` (the FL4 driver).** (1) `IntegrateGrains` predict (GR1 — `prev=pos` snapshot);
   (2) rebuild the GR2 neighbor list from the PREDICTED positions (once per step — the cell-size = the GR2
   `hSearch`, which must be ≥ the max contact diameter); (3) K JACOBI contact iterations, each
   `SolveGrainContact` → apply (separate dp[]); (4) derive velocity `vel = (pos − prev)/dt` (the FL4 PBF
   velocity update, fxdiv per axis); (5) `CollideGrainPlane` + `CollideGrainSpheres` (project out of the
   ground + the static `fpx::FxBody` spheres, AFTER the solve). Returns the contact count (a coverage stat).
   Pure integer, fixed op order → two-run bit-identical + bit-exact GPU==CPU.
3. **Radius-aware colliders (the GR1 ground-rest discipline, into the colliders).** `CollideGrainPlane`:
   clamp `pos.y ≥ groundY + radius` (the grain SURFACE on the floor — GR1's rest). `CollideGrainSphere`:
   project the grain CENTRE out to `sphereRadius + grainRadius` (the surfaces touch, not the centres) — an
   int32 AABB reject (against `sphereRadius + grainRadius`) then the int64 `FxLength`/`FxNormalize` snap; the
   `kGrainCollideEps` surface-snap tolerance (the `kFluidCollideEps` twin). Static grains: plane-clamped, not
   sphere-projected. `GrainSphereFromBody(fpx::FxBody)` bridges a rigid body to a collider. int64 →
   `grain_collide.comp` Vulkan-only + Metal CPU.
4. **3 int64 shaders, Vulkan-only (the GR1/FL4 split — NOT in `hf_gen_msl`).** `grain_contact_dp` +
   `grain_contact_apply` + `grain_collide`, `[numthreads(64,1,1)]` JACOBI multi-thread (NO TDR ceiling).
   Report they are NOT in `hf_gen_msl` (int64), while the GR2 neighbor passes (rebuilt each step) ARE. The
   `grain_contact_apply` MAY be int32 (a plain `pos += dp` add) — if so it CAN be MSL-native; the implementer
   reports which. (Prefer the simplest correct split.)
5. **Showcase `--grain-contact-shot <out>` (Vulkan) AND `--grain-contact` (Metal) — WIRE BOTH.** Drop the
   GR1 1000-grain block, run `StepGrainContactSteps` enough steps that the grains stop interpenetrating and
   settle into a LOOSE frictionless HEAP (no longer the GR1 degenerate flat-collapse — contact gives the
   pile real volume; WITHOUT friction it spreads WIDE/FLAT, the GR4 contrast). Vulkan: the multi-pass GPU
   driver → **memcmp vs the CPU `StepGrainContactSteps` reference**. Metal: runs the CPU reference. Color the
   settled grains to a BGRA8 side view. Golden = `tests/golden/metal/grain_contact.png` (Mac-baked by the
   CONTROLLER — DO NOT commit).
6. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU grain array after K steps == the CPU `StepGrainContactSteps`
     reference byte-for-byte. Print `grain-contact: {particles:<N>, steps:<K>, iters:<I>, contacts:<C>}
     GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `grain-contact determinism: two runs BYTE-IDENTICAL`.
   - **(3) non-penetration / coverage:** the solve relieves overlap — print `grain-contact coverage:
     {penBefore:<Pb>, penAfter:<Pa>} (overlap relieved)` where `Pa < Pb` (the max or summed pair penetration
     dropped). **(The FL4 honesty discipline:** Jacobi is iterative so the residual penetration is
     deterministic-but-nonzero; report peak/summed penetration RELIEVED, not zero — a deterministic
     non-penetration metric, NOT an analytic guarantee.) Also assert no grain ends below `groundY + radius`
     and none inside a collider sphere (within `kGrainCollideEps`).
   - **(4) no-op:** a single grain / grains spread > contact range apart, no colliders → unchanged by the
     solve (free-fall + ground rest only). Print `grain-contact no-op: no overlap (solve idle)`.
   - **Golden discipline: ONLY `tests/golden/metal/grain_contact.png`; do NOT commit it.** Existing 131
     image goldens (GR1 `grain_integrate` + GR2 `grain_neighbors` + all) UNTOUCHED.
7. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
8. **Tests `tests/grain_test.cpp` additions (pure CPU):** `SolveGrainContact` — a hand-laid overlapping pair
   → the exact inverse-mass-weighted Q16.16 push (equal masses split 50/50; a static+dynamic pair pushes
   only the dynamic one); a non-overlapping pair → Δp 0; `CollideGrainPlane`/`CollideGrainSphere` — a grain
   below the floor / inside a sphere → snapped to `groundY+radius` / `sphereR+grainR`; `StepGrainContact` —
   a tiny overlapping cluster settles to reduced penetration, deterministic, GPU-order-independent (Jacobi).
   Clean under `windows-msvc-asan`.
9. **Introspect.** Add exactly `deterministic-grain-contact` (features) + `--grain-contact-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (the GR2
   controller-caught lesson — do NOT skip the golden rebake).

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + barrier + read-back path (the FL4/GR2 surface:
  `BufferUsage::Storage` + `DispatchCompute` + `ComputeToComputeBarrier` + `ReadBuffer`). `rhi.h` +
  `rhi_factory` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `cloth.h` + `fluid.h` + `engine/physics/`
  UNCHANGED. GR1/GR2 grain code + shaders UNCHANGED (GR3 additive). Report the seam is empty.

## Out of scope (YAGNI — later GR slices)
Coulomb friction / angle-of-repose (GR4 — GR3 is FRICTIONLESS, the pile spreads flat; GR4 adds the
tangential clamp that makes it cone), lockstep (GR5), the lit render (GR6). Rolling resistance, cohesion,
two-way grain↔rigid coupling, restitution/bounce, variable per-pair friction. GR3 claims ONLY: deterministic
frictionless non-penetration + colliders, bit-identical CPU↔Vulkan↔Metal, with the loose-pile integer golden
+ the four proofs (the honest "overlap relieved", not "zero penetration").

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 96) + the new `grain_test` contact cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--grain-contact-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows a coherent settled LOOSE pile (pixel-check the grain region) — do NOT trust the proof alone
   (the NAV6/CL6 lesson).**
3. Metal: `visual_test --grain-contact` → new golden `tests/golden/metal/grain_contact.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm the 3 int64
   contact shaders are correctly NOT MSL-generated (the GR2 neighbor passes still ARE).** Cross-vendor delta
   STRICT ZERO (integer bar).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `grain_contact.png` added;
   the other 131 byte-identical (GR1/GR2 + all untouched; re-run `--grain-integrate-shot` +
   `--grain-neighbors-shot` → still bit-exact). `git diff master --stat -- tests/golden` = ONLY
   `grain_contact.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-grain-contact` + `--grain-contact-shot`; introspect test
   updated. (The GR2 lesson: `git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; `engine/sim/fpx.h` + `cloth.h` + `fluid.h` +
   `engine/physics/` + GR1/GR2 grain code/shaders byte-unchanged). `scripts/verify.ps1` updated:
   `grain_contact` golden in the Mac loop + `--grain-contact-shot` in `$vkShots`. The 3 contact shaders NOT
   in `hf_gen_msl` (int64); the GR2 neighbor passes still ARE.
