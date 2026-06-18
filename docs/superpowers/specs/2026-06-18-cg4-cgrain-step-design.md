# Slice CG4 — Deterministic Rigid↔Grain Coupling: THE COUPLED STEP (the body sinking into sand) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #12 (DETERMINISTIC
> TWO-WAY RIGID↔GRAIN COUPLING, `hf::sim::cgrain`). The INTEGRATED solver: one deterministic tick that runs
> the grain's own frictional pile dynamics (GR3 non-penetration + GR4 Coulomb friction) AND both exchange
> directions (CG2 grain→body support, CG3 body→grain displacement) AND the rigid integrate — a dynamic sand
> bed AND a dynamic body in one bidirectional loop. The result: a body **sinks** into the bed under gravity,
> the sand piles around it (holding its angle of repose) and **supports** it, and it settles half-buried — no
> script. The composition of the existing bit-exact pieces; the host-driven multi-pass driver (the CP4/GR4
> shape). JACOBI throughout → NO TDR. The CP4 twin with the grain sim. Branch: `slice-cg4`. See
> [[hazard-forge-couple-grain-roadmap]].

**Goal:** Extend `engine/sim/couple_grain.h` (additive — CG1/CG2/CG3 byte-unchanged) with `StepCGrain(world,
dt, iters)` / `StepCGrainSteps` (the integrated tick composing the GR3/GR4 grain sub-passes + CG2 + CG3 + the
rigid integrate) + a `MeasureCGrainState` helper (body sink/rest line + grain repose + body motion). The GPU
showcase is a host-driven multi-pass driver over the EXISTING shaders (GR3 `grain_contact_*` + GR4
`grain_friction` + CG2 `cgrain_support` + CG3 `cgrain_displace`) — **NO new shader, NO new RHI**. Add
`--cgrain-step-shot` (Vulkan) / `--cgrain-step` (Metal). Bake the integer golden `cgrain_step`. The bit-exact
grains+body state both dynamic.

## Design call: compose the GR3/GR4 sub-passes (NOT StepGrainFriction wholesale) + the INTEGER bar
CG4 does NOT call `grain::StepGrainFriction` wholesale (it would re-predict + re-build the neighbour list and
skip the coupling). Instead it composes the GR3/GR4 SUB-functions (`IntegrateGrains`, `MakeGrainGrid`/
`BuildGrainCellTable`/`BuildGrainNeighborList`, `SolveGrainContact`, `SolveGrainFriction`, `CollideGrainPlane`)
interleaved with CG2 `AccumBodyGrainForces` + CG3 `ApplyBodyToGrains` + the rigid `IntegrateBody`/`ResolveGround`,
in the locked order below. Every pass is the EXISTING bit-exact code (GR3/GR4/CG2/CG3 int64 Vulkan-only +
Metal CPU; the CG1 query int32 MSL-native). The host driver re-runs them per step with a
`ComputeToComputeBarrier` between sub-passes (the CP4/GR4 driver). NO new shader. Bar: strict INTEGER (Vulkan
GPU == Metal CPU-ref == golden, ZERO differing pixels). (The whole step is int64-touching → the GPU showcase
runs the grain+coupling as the established Vulkan-only int64 passes + Metal runs the CPU `StepCGrain`
byte-identical; the CG1 query passes within the step stay MSL-native.)

## The coupled tick (the locked order — the CP4 twin with the grain sim)
`StepCGrain(world, dt, iters)`:
```
(1) PREDICT:  grains: IntegrateGrains(grains, gravity, dt, groundY)     // GR1: prev=pos, predict pos
              bodies: for each dynamic body, vel += gravity·dt           // velocity only; pos integrated at (5)
(2) BUILD:    grid = MakeGrainGrid(grains, hSearch); table = BuildGrainCellTable; nbr = BuildGrainNeighborList
              query = GatherBodyGrains(world)                            // CG1 (from predicted grains + body pos)
(3) ITERATE (K JACOBI iters), each:
      (3a) GR3 normal:   SolveGrainContact → apply pos+=Δp     (grains — grain-grain non-penetration)
      (3b) GR4 friction: SolveGrainFriction → apply pos+=Δp    (grains — the angle-of-repose)
      (3c) CG3 body→grain: ApplyBodyToGrains (grains pushed out of the bodies + drag reaction)   (grains)
      (3d) CG2 grain→body: AccumBodyGrainForces (support + drag → body.vel delta)                 (bodies)
(4) GRAIN VELOCITY: vel = (pos − prev)/dt   (the GR-step grain velocity update)
(5) BODY INTEGRATE: for each dynamic body, pos += vel·dt; ResolveGround(body, groundY)   (the bed/floor clamp)
              grains: CollideGrainPlane(grains, groundY)   (grains rest on the ground floor)
```
The body accumulates gravity (1) + support/drag over the K iters (3d), THEN its position integrates (5); the
sand self-piles (3a/3b) and parts around the body (3c). Over many steps the body sinks in, the sand supports
it, and it settles half-buried — emergent, no script. The neighbour list + CG1 query are built ONCE per step
(fixed across the K iters — the standard PBF choice). Pure integer, fixed op order → two-run bit-identical +
bit-exact GPU==CPU. **(The CP4 finding applies: CG2 `AccumBodyGrainForces` runs in EACH of the K iters (3d), so
its impulse is applied K× per step; balance it against once-per-step gravity with the body `invMass =
kOne/iters` compensation (mathematically exact: K·F·(1/K) = F) — the same clean compensation CP4 used.)**
Bodies do not collide with each other in CG4 (single body / non-overlapping).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The CP4 coupled-step to MIRROR (`engine/sim/couple.h`):** `StepCouple` (the (1)-(5) composed tick, the
  host multi-pass driver, the `invMass = kOne/iters` compensation) — CG4's `StepCGrain` is the SAME shape with
  the grain sim (GR3/GR4) instead of the fluid sim (FL4). `MeasureCoupleState` → `MeasureCGrainState`.
- **The GR3/GR4 grain sub-passes to COMPOSE (`engine/sim/grain.h`, read-only):** `IntegrateGrains`,
  `MakeGrainGrid`/`BuildGrainCellTable`/`BuildGrainNeighborList`, `SolveGrainContact` (GR3), `SolveGrainFriction`
  (GR4), `CollideGrainPlane`. Compose these — do NOT call `StepGrainFriction` wholesale (read its body to get
  the exact call sequence). DO NOT modify grain.h.
- **The CG2/CG3 exchange + the rigid integrate (this branch's `couple_grain.h` + `fpx.h`, read-only):**
  `AccumBodyGrainForces` (CG2), `ApplyBodyToGrains` (CG3), `GatherBodyGrains` (CG1); `fpx::IntegrateBody`
  (split: vel-only predict at (1), pos integrate at (5)), `ResolveGround`. DO NOT modify fpx.h or
  CG1/CG2/CG3 functions — CG4 ORCHESTRATES them.
- **The host multi-pass driver mold (`--couple-step-shot` CP4 + `--cgrain-support-shot` CG2 / `--fluid-solve`):**
  the per-step dispatch sequence with `ComputeToComputeBarrier`. CG4's GPU driver is the SAME, in the (1)-(5)
  order, re-running the GR2 nbr + CG1 query + GR3/GR4 + CG3/CG2 passes per step. NO new shader.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp`
  (**REBAKE the introspect JSON golden** — the GR2/CP2 lesson), `tests/cgrain_test.cpp`.

## Design decisions (locked)
1. **`StepCGrain` (the composed tick, the (1)-(5) order above).** Pure integer, fixed op order. The body's
   velocity carries gravity + the accumulated support/drag; the position integrates once per step after the
   iters. `invMass = kOne/iters` (the CP4 compensation). Returns nothing. `StepCGrainSteps(world, dt, iters,
   steps)` runs K ticks (the GPU K-step driver mirror).
2. **`MeasureCGrainState(world)` (the honest metrics helper).** Returns `{ restY (the body's pos.y), repose
   (the GR4 grain slope), sink (the body's drop from its start) }` — deterministic Q16.16 stats for the proofs.
3. **Showcase `--cgrain-step-shot <out>` (Vulkan) AND `--cgrain-step` (Metal) — WIRE BOTH.** A grain bed (a
   dense GR4-settle-able block) + a body dropped above it. Run `StepCGrainSteps` enough steps that the body
   sinks in, the sand piles around it, and it settles half-buried while the bed stays coherent. Vulkan: the
   multi-pass GPU driver → **memcmp vs the CPU `StepCGrainSteps` reference**. Metal: the CPU reference. Color
   the grains + body to a side view (the body buried in the bed, both dynamic). Golden =
   `tests/golden/metal/cgrain_step.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU world state (grains + bodies) after K steps == the CPU reference
     byte-for-byte. Print `cgrain-step: {bodies:<B>, grains:<N>, steps:<K>, iters:<I>, restY:<Y>} GPU==CPU
     BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `cgrain-step determinism: two runs BYTE-IDENTICAL`.
   - **(3) coupled (the headline):** the body settled SUPPORTED in the bed (`restY > groundY + radius` by a
     margin, bounded above — it did NOT crash through, did NOT fly out) AND it SANK from its start (the drop is
     non-trivial) AND the bed stayed coherent (the grain repose held — bounded, no explosion). Print `cgrain-
     step coupled: restY <Y>, sank <D>, bed coherent (body settled in the pile)`. (The HONEST framing: the rest
     line + sink are emergent/within-band, deterministic — the GR4/CP2 caveat shape.)
   - **(4) control:** a support=0 body sinks straight through to the bed floor while the bed still piles
     (proving the coupling does work). Print `cgrain-step control: support=0 sinks through, bed piles (coupling
     does work)`.
   - **Golden discipline: ONLY `tests/golden/metal/cgrain_step.png`; do NOT commit it.** Existing 144 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
6. **Tests `tests/cgrain_test.cpp` additions (pure CPU):** `StepCGrain` — a body over a small grain bed sinks
   to a rest line above the bed floor AND the bed stays bounded, the support=0 control sinks; deterministic;
   the composed order is fixed. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-cgrain-step` (features) + `--cgrain-step-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (the GR2/CP2
   lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** CG4 ORCHESTRATES the existing GR3 `grain_contact_*` + GR4 `grain_friction` + CG2 `cgrain_support`
  + CG3 `cgrain_displace` shaders (NO new shader). Reuse the existing compute + SSBO + dispatch + barrier +
  read-back path. `rhi.h` + `rhi_factory` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h`
  + `cloth.h` + `couple.h` + `engine/physics/` UNCHANGED. CG1/CG2/CG3 cgrain code + shaders UNCHANGED (CG4
  additive — only `StepCGrain`/`StepCGrainSteps`/`MeasureCGrainState` + the showcase). Report the seam is empty
  incl. shaders (no new shader).

## Out of scope (YAGNI — later CG slices)
Lockstep/rollback (CG5), the lit render (CG6), body-body contacts within the coupled step, support torque /
angular coupling (a sphere body has none; an asymmetric-bed refinement), multi-body. CG4 claims ONLY: the
integrated bidirectional coupled step — a body sinks into and is supported by a dynamic self-piling sand bed,
no script — bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs (the honest emergent
rest line + sink, the GR4 bed coherence, the support=0-sinks control).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 98) + the new `cgrain_test` step cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--cgrain-step-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows the body settled/buried in a coherent (non-exploded) grain bed — both dynamic (pixel-check; the
   NAV6/CL6 lesson).**
3. Metal: `visual_test --cgrain-step` → new golden `tests/golden/metal/cgrain_step.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (CG4
   orchestrates existing passes); the CG1 query passes still MSL-generate.** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cgrain_step.png` added; the
   other 144 byte-identical (re-run `--cgrain-query/support/displace-shot` → still bit-exact). `git diff master
   --stat -- tests/golden` = ONLY `cgrain_step.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cgrain-step` + `--cgrain-step-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` + ALL shaders UNCHANGED — CG4 adds NO new shader; `engine/sim/fpx.h` + `grain.h` +
   `fluid.h` + `cloth.h` + `couple.h` + `engine/physics/` + CG1/CG2/CG3 cgrain code/shaders byte-unchanged).
   `scripts/verify.ps1` updated: `cgrain_step` golden in the Mac loop + `--cgrain-step-shot` in `$vkShots`.
