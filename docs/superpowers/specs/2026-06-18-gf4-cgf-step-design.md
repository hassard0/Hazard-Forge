# Slice GF4 — Deterministic Grain↔Fluid Coupling: THE COUPLED STEP (StepCGF) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #13 (DETERMINISTIC
> TWO-WAY GRAIN↔FLUID COUPLING, `hf::sim::cgf`). The CONVERGENCE slice: GF1 linked the two pools, GF2 added
> fluid→grain buoyancy, GF3 added grain→fluid displacement — GF4 runs them TOGETHER as ONE settling tick. A
> frictional sand bed + an incompressible PBF fluid co-settle in ONE Q16.16 world: WET SAND / MUD / SLURRY —
> the fluid pools on/seeps around the bed, submerged grains lighten, the whole thing re-simulates bit-for-bit.
> NO new shader, NO new RHI — `StepCGF` is a HOST-DRIVEN multi-pass driver over the EXISTING FL4/GR3/GR4/GF2/GF3
> passes (the CG4 `StepCGrain` precedent). **THE HEAVIEST step in the engine** (two dynamic pools each solved per
> iter) → SMALL/FAST scenes. Branch: `slice-gf4`. See [[hazard-forge-couple-gf-roadmap]].

**Goal:** Extend `engine/sim/couple_gf.h` (additive — GF1/GF2/GF3 byte-unchanged) with `StepCGF(world, dt,
iters)` (one coupled grain+fluid settling tick), `StepCGFSteps`, and a `MeasureCGFState` honest-metrics helper.
Add `--cgf-step-shot` (Vulkan) / `--cgf-step` (Metal) — a host-driven multi-pass driver over the EXISTING
shaders. Bake the integer golden `cgf_step`. **NO new shader, NO new RHI.**

## Design call: a host-driven multi-pass driver over the EXISTING passes (the CG4 mold), VELOCITY couplings POST
GF4 is the GF-arc's `StepCGrain` (CG4): the coupled step is NOT a new kernel — it is a fixed-order CPU driver
that calls the ALREADY-VERIFIED passes (FL4 density, GR3 normal, GR4 friction, GF2 buoyancy, GF3 displacement),
each of which already has its Vulkan-only int64 shader + Metal-CPU-reference. The GPU showcase dispatches those
existing shaders in the locked order; the Metal showcase runs the CPU `StepCGF`. So **GF4 adds ZERO shaders and
ZERO RHI** — the strongest possible "no new surface" claim. Bar: strict INTEGER (Vulkan GPU == Metal CPU-ref ==
golden, ZERO differing pixels) — every constituent pass is already integer-bit-exact cross-backend, and the
driver only sequences them.

### THE MAKE-OR-BREAK DECISION — velocity couplings run ONCE, AFTER the PBF velocity update (NOT inside the iters)
Both pools are PBF/PBD: the final velocity is DERIVED as `(pos − prev)/dt` after the position solve. Therefore
**any velocity impulse applied INSIDE the K position iterations is CLOBBERED** by that derivation. The roadmap
sketch listed `{FL4 | GR3 | GR4 | GF3 | GF2}` all inside the K iters — taken literally that is a bug (GF2's
grain-velocity buoyancy and GF3's fluid-velocity drag would be overwritten by the pos−prev update). **GF4
resolves it (a documented, deliberate refinement of the roadmap sketch):**

- **The K Jacobi iters hold ONLY each pool's POSITIONAL self-constraints** (their effect is carried by the
  pos−prev velocity derivation): FL4 density (fluid), GR3 normal + GR4 friction (grain).
- **The cross-pool VELOCITY couplings run ONCE per step, in a POST pass AFTER the velocity update** (so they
  survive, exactly as PBF XSPH-viscosity / vorticity are applied post-velocity-update): GF3 `ApplyGrainsToFluid`
  (fluid snapped out of the grains + the grain→fluid drag) then GF2 `AccumGrainBuoyancy` (grain buoyancy+drag).

This choice has THREE wins: (1) **correctness** — nothing is clobbered; (2) **maximal reuse** — GF2 AND GF3 are
called **VERBATIM** (no splitting, no new coupling math, no re-tuning — GF1/GF2/GF3 stay byte-frozen); (3) **no
K-compensation** — because GF2/GF3 run ONCE (not K×), the CG4 `invMass = kOne/iters` trick is UNNEEDED; grains
and fluid keep their natural mass. The grain→fluid displacement happening once-per-step (not re-projected each
iter) is exactly how GF3's `StepCGFDisplace` already works — the fluid's own K-iter density solve keeps it
incompressible, and the post pass parts it out of the sand each step.

## THE COUPLED TICK (`StepCGF`, the locked make-or-break order)
```
(1) PREDICT both pools (each: vel += g·dt; prev = pos; pos += vel·dt; floor clamp):
      fluid::IntegrateFluid(world.fluid, world.gravity, dt, world.groundY)      // FL1 predict
      grain::IntegrateGrains(world.grains, world.gravity, dt, world.groundY)    // GR1 predict
(2) BUILD ONCE (from the PREDICTED positions; fixed across the K iters — the standard PBF choice):
      fluid neighbours  : fGrid/fTable/fList = MakeGrid + BuildCellTable + BuildNeighborList(world.fluid)  // FL2
      grain neighbours  : gGrid/gTable/gList = MakeGrainGrid + BuildGrainCellTable + BuildGrainNeighborList // GR2
      cross-pool lists  : nbr = BuildCGFNeighbors(world)   // GF1 shared-grid gfNeighbors + fgNeighbors
      (the FluidKernel LUT `kernel` is built once from world.h/restDensity — reuse the FL scene's kernel build)
(3) K JACOBI ITERS — each pool's POSITIONAL self-constraints ONLY, in this fixed sub-order:
      (3a) FL4 density : ComputeDensity(fluid,fList,kernel) → ComputeLambda(...) → SolveDensityConstraint(... dp)
                          → fluid[i].pos += dp[i]              (incompressible pool)
      (3b) GR3 normal  : SolveGrainContact(grains,gList,dp) → grains[i].pos += dp[i]   (grain non-penetration)
      (3c) GR4 friction: SolveGrainFriction(grains,gList,kGrainMu,dp) → grains[i].pos += dp[i]  (angle of repose)
(4) VELOCITY UPDATE both pools (PBF, derives the constraint-corrected velocity):
      fluid[i].vel = (fluid[i].pos − fluid[i].prev)/dt ;  grains[i].vel = (grains[i].pos − grains[i].prev)/dt
      (static / kFlagStatic skipped in every sub-pass, as the constituent functions already do)
(5) CROSS-POOL VELOCITY COUPLING — ONCE, AFTER (4) so it survives (reuse the step-(2) cross lists, fixed):
      (5a) GF3 grain→fluid : ApplyGrainsToFluid(world, nbr)      // fluid snapped out of grains + drag reaction
      (5b) GF2 fluid→grain : AccumGrainBuoyancy(world, nbr, kBuoyPerFluid)   // submerged grains lighten + drag
(6) GROUND CLAMPS both pools: fluid::CollidePlane(world.fluid, groundY) ; grain::CollideGrainPlane(grains, groundY)
```
Pure integer, fixed op order → two runs bit-identical AND bit-exact GPU==CPU (every constituent pass is already
proven so). Over many steps: the fluid settles incompressible and pools on the bed, the grains hold their GR4
repose, submerged grains lighten toward slurry, and the fluid parts out of the grain volumes — emergent WET
SAND, no script.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The CG4 coupled driver to MIRROR (`engine/sim/couple_grain.h`):** `StepCGrain(world, dt, iters)` (predict →
  build-once → K iters {positional} → velocity update → cross-pool → ground) + `StepCGrainSteps` +
  `MeasureCGrainState`/`CGrainState`. GF4's `StepCGF` is the SAME shape with TWO particle pools (fluid+grain)
  instead of grains+rigid-bodies, and the cross-pool couplings in the POST pass (5) instead of in the iters.
- **The fluid PBF step to REUSE (`engine/sim/fluid.h`):** `StepFluid` (the predict → FL2 build → K {ComputeDensity
  → ComputeLambda → SolveDensityConstraint → apply dp} → vel=(pos−prev)/dt → collide reference) — read it to
  copy the EXACT FL4 sub-pass order + the kernel/neighbour build. `IntegrateFluid` (FL1 predict), `MakeGrid`/
  `BuildCellTable`/`BuildNeighborList` (FL2), `ComputeDensity`/`ComputeLambda`/`SolveDensityConstraint` (FL3/FL4),
  `CollidePlane` (FL ground), `FluidKernel` + its LUT build (reuse the FL showcase scene's kernel construction).
- **The grain step pieces to REUSE (`engine/sim/grain.h`):** `IntegrateGrains` (GR1 predict), `MakeGrainGrid`/
  `BuildGrainCellTable`/`BuildGrainNeighborList` (GR2), `SolveGrainContact` (GR3 normal Δp), `SolveGrainFriction`
  + `kGrainMu` (GR4 repose Δp), `CollideGrainPlane` (GR ground), `MeasureGrainRepose` (the repose stat).
- **The GF1/GF2/GF3 world + couplings (this branch's `couple_gf.h`, read-only — call VERBATIM):** `CGFWorld`
  (carries `grains`,`fluid`,`gravity`,`dt`,`groundY`,`h`), `BuildCGFNeighbors` → `CGFNeighbors`,
  `AccumGrainBuoyancy(world, nbr, kBuoyPerFluid)` (GF2, vel-only — reuse VERBATIM in (5b)), `ApplyGrainsToFluid(
  world, nbr)` (GF3, pos snap + vel drag — reuse VERBATIM in (5a)), `kBuoyPerFluid`/`kDrag`/`kDragReaction`,
  `MeasureWetDry`, `MeasureFluidGrainPenetration`. **DO NOT modify GF1/GF2/GF3 code or grain.h/fluid.h/fpx.h/
  cloth.h/couple.h/couple_grain.h** — GF4 is the additive sibling.
- **The host-driven multi-pass GPU showcase mold (the CG4 `--cgrain-shot` driver, NO new shader):** the showcase
  dispatches the EXISTING shaders in the (1)-(6) order — `fluid_dp.comp` (FL4, Vulkan-only int64) + the FL
  density/lambda passes + `grain_contact.comp` (GR3) + `grain_friction.comp` (GR4) + `cgf_displace.comp` (GF3,
  Vulkan-only int64) + `cgf_buoyancy.comp` (GF2, Vulkan-only int64) + the int32 GF1 cross-query + the int32
  fluid/grain cell passes — then memcmp's the final fluid+grain state vs the CPU `StepCGF`. The GF1 cross-query
  + cell passes stay int32 MSL-native; everything else is the existing Vulkan-only int64 set. **NO `cgf_step`
  shader exists** (GF4 is pure orchestration).
- **Showcase + registration:** GF1/GF2/GF3's `--cgf-*-shot` plumbing; `scripts/verify.ps1`, `engine/editor/
  introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden** — the GR2/CP2/GF3 lesson),
  `tests/cgf_test.cpp`.

## Design decisions (locked)
1. **`StepCGF(world, dt, iters)` — the (1)-(6) locked-order host driver above.** Reuses FL4/GR3/GR4 in the K
   iters and GF2/GF3 VERBATIM in the post pass. NO new coupling math, NO new shader, NO `invMass` compensation
   (GF2/GF3 run once). `StepCGFSteps(world, dt, iters, steps)` runs K ticks (the reference the GPU driver
   memcmp's against). The neighbour lists + kernel are built ONCE per step (the PBF fixed-neighbour choice).
2. **Showcase `--cgf-step-shot <out>` (Vulkan) AND `--cgf-step` (Metal) — WIRE BOTH, host-driven multi-pass.**
   The SMALL/FAST settling scene below. Vulkan: the multi-pass GPU driver (existing shaders, locked order) →
   **memcmp vs the CPU `StepCGF` reference** (final fluid AND grain arrays). Metal: the CPU reference. Color the
   grains (sand/tan) + fluid (cyan) to a side view (fluid pooled on/around the settled wet bed). Golden =
   `tests/golden/metal/cgf_step.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **SMALL/FAST scene (THE HEAVIEST step — perf is the top risk).** A SMALL grain bed + a SMALL fluid block
   resting on it: e.g. **`8×3×6 = 144` grain bed** (a few GR4 pre-settle steps so it starts at repose) + **`4×4×6
   = 96` fluid block** seeded just above the bed's centre, `h = 1.5`, **`iters = 3`** (low K), **`~120` steps**
   (modest). Keep BOTH pools ≤ ~150 particles and K low — the CG4 ~79s/2925-grains×300-steps note COMPOUNDS with
   two dynamic pools. The verify gate needs DIFF 0.0000, NOT a large scene. If a build/bake is slow, HALVE the
   step count before the particle count (settling is visible by ~80 steps).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU fluid AND grain arrays after `steps` coupled ticks == the CPU `StepCGF`
     reference byte-for-byte. Print `cgf-step: {grains:<G>, fluid:<F>, iters:<K>, steps:<S>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `cgf-step determinism: two runs BYTE-IDENTICAL`.
   - **(3) co-settling (the emergent metric, honest/within-band):** print `cgf-step settled: {bedY:<By>,
     fluidY:<Fy>, repose:<R>, wetY:<Wy>, dryY:<Dy>}` from `MeasureCGFState` — assert `fluidY > bedY` (the fluid
     POOLS ABOVE the bed by a margin), `repose > 0` (the GR4 bed still holds an angle of repose), and `wetY >
     dryY` (submerged grains sit higher than dry — the GF2 lift survives the coupled step). The WET/DRY +
     pool-above margins are EMERGENT/within-band (the CP2/GR4/GF2 caveat), NOT exact depths.
   - **(4) no-penetration:** `MeasureFluidGrainPenetration` after the run is RELIEVED vs a free-fall control —
     print `cgf-step no-penetration: {pen:<P>} (fluid rests on the bed)`; assert the fluid is not buried inside
     the bed (pen below a documented bound; Jacobi single-projection residual deterministic-but-nonzero — the
     FL4/GR3/GF3 caveat shape).
   - **Golden discipline: ONLY `tests/golden/metal/cgf_step.png`; do NOT commit it.** Existing 150 image goldens
     UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels. Every
   constituent pass is already integer-bit-exact cross-backend; the driver only sequences them.
6. **Tests `tests/cgf_test.cpp` additions (pure CPU):** `StepCGF` on a tiny scene (a few grains + a few fluid
   particles, 2-3 steps) — two runs byte-identical; the fluid ends ABOVE the grains (co-settle); a static grain
   AND a static fluid particle are untouched (every sub-pass skips static); `iters=0` → predict+velocity+couple
   only (no position solve), still deterministic; `MeasureCGFState` on a known scene. Clean under
   `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-cgf-step` (features) + `--cgf-step-shot` (showcases). **REBAKE
   `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (the GR2/CP2/GF3 lesson —
   `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + barrier + read-back path (the CG4/GF3 surface).
  `rhi.h` + `rhi_factory` + backend dirs UNCHANGED. `engine/sim/grain.h` + `fluid.h` + `fpx.h` + `cloth.h` +
  `couple.h` + `couple_grain.h` + `engine/physics/` UNCHANGED. GF1/GF2/GF3 cgf code + shaders UNCHANGED (GF4
  additive — only NEW `StepCGF`/`StepCGFSteps`/`MeasureCGFState` + the showcase). **NO new shader.** Report the
  seam empty.

## Out of scope (YAGNI — later GF slices)
Lockstep/rollback (GF5 — GF4 is the forward coupled step only), the lit 3D render (GF6). Surface tension,
capillarity, viscosity/XSPH, multi-grain-overlap full relief (single-projection residual documented), dynamic
basins/walls (the ground plane only). GF4 claims ONLY: a deterministic forward grain+fluid coupled settling
step composing FL4/GR3/GR4 + GF2/GF3, bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 99) + the new `cgf_test` step cases. Clean under
   `windows-msvc-asan` (build+run `cgf_test` + `introspect_test`).
2. **proofs + visual:** `--cgf-step-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image
   shows the fluid POOLED ON/AROUND the settled wet sand bed (cyan above the tan bed, the bed holding repose —
   pixel-check; the NAV6/CL6/GF3 lesson).**
3. Metal: `visual_test --cgf-step` → new golden `tests/golden/metal/cgf_step.png`; two runs DIFF 0.0000 (gate on
   `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader was added (GF4 is pure
   orchestration — `hf_gen_msl` UNCHANGED; the existing GF1 cross-query/cell passes still int32 MSL-native, the
   FL4/GR3/GR4/GF2/GF3 int64 passes still Vulkan-only).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cgf_step.png` added; the other
   150 byte-identical (re-run `--cgf-query/buoyancy/displace-shot` → still bit-exact). `git diff master --stat --
   tests/golden` = ONLY `cgf_step.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cgf-step` + `--cgf-step-shot`; introspect test updated.
   (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/grain.h` + `fluid.h` + `fpx.h` + `cloth.h` + `couple.h` +
   `couple_grain.h` + `engine/physics/` + GF1/GF2/GF3 cgf code/shaders byte-unchanged). `scripts/verify.ps1`
   updated: `cgf_step` golden in the Mac loop + `--cgf-step-shot` in `$vkShots`. **NO new entry in `hf_gen_msl`
   (GF4 adds no shader).**
