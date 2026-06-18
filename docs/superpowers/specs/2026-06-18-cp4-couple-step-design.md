# Slice CP4 — Deterministic Rigid↔Fluid Coupling: THE COUPLED STEP (the bobbing barrel) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #11 (DETERMINISTIC
> TWO-WAY RIGID↔FLUID COUPLING, `hf::sim::couple`). The INTEGRATED solver: one deterministic tick that runs
> the fluid's own incompressibility (FL4) AND both exchange directions (CP2 fluid→body, CP3 body→fluid) AND
> the rigid integrate — a dynamic fluid AND a dynamic body in one bidirectional loop. The result: a barrel
> **BOBS** under emergent buoyancy + an incompressible fluid (no script). The composition of the existing
> bit-exact pieces; the host-driven multi-pass driver (the FL4/GR3 shape). JACOBI throughout → NO TDR. Branch:
> `slice-cp4`. See [[hazard-forge-couple-roadmap]].

**Goal:** Extend `engine/sim/couple.h` (additive — CP1/CP2/CP3 byte-unchanged) with `StepCouple(world, dt,
iters)` / `StepCoupleSteps` (the integrated tick composing the FL4 fluid sub-passes + CP2 + CP3 + the rigid
integrate) + a `MeasureCoupleState` helper (body float line + fluid density residual + body bob amplitude).
The GPU showcase is a host-driven multi-pass driver over the EXISTING shaders (FL4 `fluid_*` + CP2
`couple_buoyancy` + CP3 `couple_displace`) — **NO new shader, NO new RHI**. Add `--couple-step-shot` (Vulkan)
/ `--couple-step` (Metal). Bake the integer golden `couple_step`. The bit-exact fluid+body state both dynamic.

## Design call: compose the FL4 sub-passes (NOT StepFluid wholesale) + the INTEGER bar (strict zero-diff)
CP4 does NOT call `fluid::StepFluid` wholesale (it would re-predict + re-build the neighbour list and skip the
coupling). Instead it composes the FL4 SUB-functions (`IntegrateFluid`, `MakeGrid`/`BuildCellTable`/
`BuildNeighborList`, `ComputeDensity`/`ComputeLambda`/`SolveDensityConstraint`, `CollidePlane`) interleaved
with CP2 `AccumBodyForces` + CP3 `ApplyBodyToFluid` + the rigid `IntegrateBody`/`ResolveGround`, in the locked
order below. Every pass is the EXISTING bit-exact code (FL4 int64 Vulkan-only + Metal CPU; CP2 per-body; CP3
per-particle; the CP1 query int32 MSL-native). The host driver re-runs them per step with a
`ComputeToComputeBarrier` between sub-passes (the FL4/GR3 driver). NO new shader. Bar: strict INTEGER (Vulkan
GPU == Metal CPU-ref == golden, ZERO differing pixels). (The whole step is int64-touching → the GPU showcase
runs the fluid+coupling as the established Vulkan-only int64 passes + Metal runs the CPU `StepCouple`
byte-identical; the CP1 query passes within the step stay MSL-native.)

## The coupled tick (the locked order — the make-or-break)
`StepCouple(world, dt, iters)`:
```
(1) PREDICT:  fluid: IntegrateFluid(particles, gravity, dt, groundY)   // FL1: prev=pos, predict pos
              bodies: for each dynamic body, vel += gravity·dt          // velocity only; pos integrated at (5)
(2) BUILD:    grid = MakeGrid(particles, kernel.h); table = BuildCellTable; nbr = BuildNeighborList   // FL2
              query = GatherBodyParticles(world)                        // CP1 (from predicted fluid + body pos)
(3) ITERATE (K JACOBI iters), each:
      (3a) FL4 density: ComputeDensity → ComputeLambda → SolveDensityConstraint → apply pos+=Δp  (fluid)
      (3b) CP3 body→fluid: ApplyBodyToFluid (displace fluid out of bodies + drag reaction)        (fluid)
      (3c) CP2 fluid→body: AccumBodyForces (buoyancy + drag → body.vel delta)                     (bodies)
(4) FLUID VELOCITY: vel = (pos − prev)/dt   (the FL4 PBF velocity update)
(5) BODY INTEGRATE: for each dynamic body, pos += vel·dt; ResolveGround(body, groundY)   (the bed clamp)
```
The body accumulates gravity (1) + buoyancy/drag over the K iters (3c), THEN its position integrates (5);
over many steps it falls in, buoyancy builds, and it BOBS around the float line (damped by drag) — emergent,
no script. The neighbour list + CP1 query are built ONCE per step (fixed across the K iters — the standard
PBF choice). Pure integer, fixed op order → two-run bit-identical + bit-exact GPU==CPU. (Bodies do not
collide with each other in CP4 — single body / non-overlapping; body-body contacts are out of scope.)

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The FL4 fluid sub-passes to COMPOSE (`engine/sim/fluid.h`, read-only):** `IntegrateFluid` (`fluid.h:171`),
  `MakeGrid`/`BuildCellTable`/`BuildNeighborList` (`fluid.h:255-415`), `ComputeDensity`/`ComputeLambda`/
  `SolveDensityConstraint` (the FL3/FL4 density solve), `CollidePlane` (`fluid.h:700`), `FluidKernel`. Compose
  these — do NOT call `StepFluid` wholesale. DO NOT modify fluid.h.
- **The CP2/CP3 exchange + the rigid integrate (this branch's `couple.h` + `fpx.h`, read-only):**
  `AccumBodyForces` (CP2), `ApplyBodyToFluid` (CP3), `GatherBodyParticles` (CP1); `fpx::IntegrateBody`
  (`fpx.h:149`, reuse for the body predict/integrate — but CP4 splits it: vel-only predict at (1), pos
  integrate at (5)), `ResolveGround` (`fpx.h:329`). DO NOT modify fpx.h or CP1/CP2/CP3 functions — CP4 is
  additive (it ORCHESTRATES them).
- **The host multi-pass driver mold (`samples/hello_triangle/main.cpp` `--fluid-solve-shot` / CP2/CP3
  showcases):** the per-step sequence of dispatches with `ComputeToComputeBarrier` between sub-passes,
  re-running the FL2 query + FL4 density + CP3 displace + CP2 buoyancy passes per step. CP4's GPU driver is
  the SAME, in the (1)-(5) order. NO new shader.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp`
  (**REBAKE the introspect JSON golden** — the GR2 lesson), `tests/couple_test.cpp`.

## Design decisions (locked)
1. **`StepCouple` (the composed tick, the (1)-(5) order above).** Pure integer, fixed op order. The bodies'
   velocity carries gravity + the accumulated buoyancy/drag; the position integrates once per step after the
   iters. Returns nothing (the world carries the state). `StepCoupleSteps(world, dt, iters, steps)` runs K
   ticks (the GPU K-step driver mirror).
2. **`MeasureCoupleState(world)` (the honest metrics helper).** Returns `{ floatY (the body's pos.y), densityResidual
   (the FL4 summed |ρ−ρ0|), ... }` — deterministic Q16.16 stats for the proofs. The showcase also tracks the
   body's min/max y over the run (the bob amplitude).
3. **Showcase `--couple-step-shot <out>` (Vulkan) AND `--couple-step` (Metal) — WIRE BOTH.** A fluid pool
   (the FL4 dam-break or a settled block) + a body dropped above it. Run `StepCoupleSteps` enough steps that
   the body falls in, bobs, and settles near a float line while the fluid stays incompressible. Vulkan: the
   multi-pass GPU driver → **memcmp vs the CPU `StepCoupleSteps` reference**. Metal: the CPU reference. Color
   the fluid + body to a side view (the body at the float line + the fluid pool, both dynamic). Golden =
   `tests/golden/metal/couple_step.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU world state (fluid + bodies) after K steps == the CPU reference
     byte-for-byte. Print `couple-step: {bodies:<B>, particles:<N>, steps:<K>, iters:<I>, floatY:<Y>} GPU==CPU
     BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `couple-step determinism: two runs BYTE-IDENTICAL`.
   - **(3) coupled (the headline):** the body floats (`floatY > groundY + radius` by a margin, bounded above)
     AND it BOBBED (peak-to-trough y motion over the run `> threshold` — it oscillated, not monotonic) AND the
     fluid stayed coherent (the density residual is bounded — no explosion). Print `couple-step coupled:
     floatY <Y>, bob <amp>, fluid coherent (emergent bobbing)`. (The HONEST framing: the float line + the bob
     are emergent/within-band, deterministic — the GR4/FL4 caveat shape; the fluid density solve is the
     FL4 peak-relieved metric.)
   - **(4) control:** a buoy=0 body sinks to the bed while the fluid still settles (proving the coupling does
     work). Print `couple-step control: buoy=0 sinks, fluid settles (coupling does work)`.
   - **Golden discipline: ONLY `tests/golden/metal/couple_step.png`; do NOT commit it.** Existing 138 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
6. **Tests `tests/couple_test.cpp` additions (pure CPU):** `StepCouple` — a body over a small pool settles to
   a float line above the bed AND bobs (peak-to-trough > 0), the fluid density residual is bounded, the buoy=0
   control sinks; deterministic; the composed order is fixed. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-couple-step` (features) + `--couple-step-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (the GR2
   lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** CP4 ORCHESTRATES the existing FL4 `fluid_*` + CP2 `couple_buoyancy` + CP3 `couple_displace`
  shaders (NO new shader). Reuse the existing compute + SSBO + dispatch + barrier + read-back path. `rhi.h` +
  `rhi_factory` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `fluid.h` + `cloth.h` + `grain.h` +
  `engine/physics/` UNCHANGED. CP1/CP2/CP3 couple code + shaders UNCHANGED (CP4 additive — only `StepCouple`/
  `StepCoupleSteps`/`MeasureCoupleState` + the showcase). Report the seam is empty incl. shaders (no new shader).

## Out of scope (YAGNI — later CP slices)
Lockstep/rollback (CP5), the lit render (CP6), body-body contacts within the coupled step, buoyancy torque /
angular coupling (a sphere body has none; a future refinement for asymmetric bodies), multi-fluid, splash/foam.
CP4 claims ONLY: the integrated bidirectional coupled step — a body bobs in an incompressible fluid, no script
— bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs (the honest emergent float
line + bob, the FL4 fluid coherence, the buoy=0-sinks control).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 97) + the new `couple_test` step cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--couple-step-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows the body at a float line in a coherent (non-exploded) fluid pool — both dynamic (pixel-check;
   the NAV6/CL6 lesson).**
3. Metal: `visual_test --couple-step` → new golden `tests/golden/metal/couple_step.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (CP4
   orchestrates existing passes); the CP1 query passes still MSL-generate.** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `couple_step.png` added; the
   other 138 byte-identical (re-run `--couple-query/buoyancy/displace-shot` → still bit-exact). `git diff
   master --stat -- tests/golden` = ONLY `couple_step.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-couple-step` + `--couple-step-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` + ALL shaders UNCHANGED — CP4 adds NO new shader; `engine/sim/fpx.h` + `fluid.h`
   + `cloth.h` + `grain.h` + `engine/physics/` + CP1/CP2/CP3 couple code/shaders byte-unchanged).
   `scripts/verify.ps1` updated: `couple_step` golden in the Mac loop + `--couple-step-shot` in `$vkShots`.
