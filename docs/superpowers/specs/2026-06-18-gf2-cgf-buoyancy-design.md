# Slice GF2 — Deterministic Grain↔Fluid Coupling: BUOYANCY / SEEPAGE (fluid→grain, the crux) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #13 (DETERMINISTIC
> TWO-WAY GRAIN↔FLUID COUPLING, `hf::sim::cgf`). The FIRST momentum exchange: each grain sums, over its GF1
> fluid-neighbour list, a **buoyant** impulse (the surrounding fluid lifting it, opposing gravity) + a **drag**
> impulse (toward the local fluid velocity), so SUBMERGED grains LIGHTEN / FLUIDIZE (a wet/dry contrast) — the
> sand under the fluid loosens toward slurry. ONE-WAY for now (fluid → grain; the grain→fluid reaction is GF3).
> THE CRUX of the flagship: the per-grain force reduction over its fluid neighbours — but unlike CG2's
> one-thread-per-BODY reduction, this is **per-PARTICLE** (one thread per grain over its OWN short fluid list)
> → per-grain-disjoint, race-free, JACOBI multi-thread, NO single-thread TDR ceiling (the cleaner FL4/GR3
> shape). Branch: `slice-gf2`. See [[hazard-forge-couple-gf-roadmap]].

**Goal:** Extend `engine/sim/couple_gf.h` (additive — GF1 byte-unchanged) with `AccumGrainBuoyancy(world,
neighbors)` (the per-grain buoyancy+drag impulse accumulate over the GF1 `gfNeighbors` fluid list) +
`StepCGFBuoyancy` / `StepCGFBuoyancySteps` (re-query → accumulate → integrate the grains; the fluid held
STATIC) + a `MeasureWetDry` helper. Add `shaders/cgf_buoyancy.comp.hlsl` (int64 → **Vulkan-only** + Metal CPU
reference). Add `--cgf-buoyancy-shot` (Vulkan) / `--cgf-buoyancy` (Metal). Bake the integer golden
`cgf_buoyancy`. NO new RHI.

## Design call: int64 per-GRAIN Jacobi (no TDR — cleaner than CG2) + the INTEGER bar (strict zero-diff)
The buoyancy/drag math is int64 (`fxmul`/`fxdiv`/`FxScale`/`FxNormalize`) → `cgf_buoyancy.comp` is
**Vulkan-only** + the Metal `--cgf-buoyancy` showcase runs the CPU `StepCGFBuoyancy` (byte-identical by
construction, the FL4/GR3/CG2 split). The GF1 cross-query (re-run each step) stays int32 MSL-native. Bar:
strict INTEGER (Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels).

**THE CRUX — but EASIER than CG2:** CG2's body-force reduction was per-BODY (one thread per body, a tiny set),
which needed the "tiny body count" caveat. GF2's reduction is per-GRAIN: one thread per grain, each grain
sums over its OWN short `gfNeighbors` fluid list (the GF1 over-inclusive box list, typically a few dozen),
writing ONLY its own velocity. So it is `[numthreads(64,1,1)]` MULTI-THREAD over grains, per-grain-disjoint,
race-free, NO atomics, NO single-thread TDR — the FL4/GR3 Jacobi pattern. (The fixed `gfNeighbors` order keeps
the int64 sum bit-identical CPU↔shader.)

## The buoyancy + drag model (the new physics — a deterministic two-phase proxy)
For grain i, over its GF1 fluid neighbours j (`gfNeighbors[gfStart[i] .. gfStart[i+1])`, ascending):
```
const uint cnt = gfStart[i+1] − gfStart[i];               // # fluid particles near grain i (submersion proxy)
if (cnt == 0) continue;                                    // dry grain -> no buoyancy/drag (free GR sim)
// BUOYANCY — ∝ the submerging fluid count, opposing gravity (up = −normalize(gravity)):
F_buoy = fxmul(kBuoyPerFluid, cnt<<kFrac) · up             // more surrounding fluid -> more lift (lightens)
// DRAG — toward the local fluid velocity (GF2: the fluid is static, so this DAMPS the grain):
vFluidAvg = (Σ_j fluid[j].vel) / cnt                       // fixed-order int64 sum / the integer count
F_drag = fxmul(kDrag, vFluidAvg − grain.vel)              // per axis
grain.vel += FxScale(F_buoy + F_drag, grain.invMass) · dt  // (linear; grains are points, no torque)
```
`kBuoyPerFluid`/`kDrag` are host-snapped Q16.16 constants tuned so SUBMERGED grains LIGHTEN (rise/loosen, the
sand fluidizes under the fluid) while DRY grains pack normally — an emergent WET/DRY contrast, NOT an exact
buoyancy depth (the CP2/GR4 within-band caveat shape). This is the CG2 contact-support reduction's twin, with
a count-based buoyancy (the fluid-submersion proxy) instead of a penetration-based support, and per-particle
instead of per-body. The `count<<kFrac` Q16.16 promotion is the CG2 precedent.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The CG2 force reduction to MIRROR (`engine/sim/couple_grain.h`):** `AccumBodyGrainForces` (the fixed-order
  reduction over a gathered list, the `count<<kFrac` buoyancy + the int64 fixed-order drag-avg, the `vel`
  delta) — GF2's `AccumGrainBuoyancy` is the SAME shape with the GF1 `gfNeighbors` fluid list, per-GRAIN (one
  thread per grain), `[numthreads(64,1,1)]` multi-thread (NOT one-thread-per-body). `StepCGrainSupport` (the
  re-query → accumulate → integrate driver), `MeasureRestLine`.
- **The GF1 cross-query to BUILD ON (this branch's `engine/sim/couple_gf.h`):** `CGFWorld`, `BuildCGFNeighbors`
  → `CGFNeighbors{gfStart, gfNeighbors, fgStart, fgNeighbors}` (GF2 sums over `gfNeighbors` = each grain's
  fluid neighbours), `MakeCGFGrid`. DO NOT modify GF1's functions — GF2 is additive.
- **The grain integrate to REUSE (`engine/sim/grain.h`, read-only):** `IntegrateGrains` (the grain integrate +
  radius-aware ground rest — reuse verbatim after the buoyancy accumulate), `GrainParticle`. `fluid::FluidParticle`
  (pos, vel). `fpx::FxNormalize`/`FxScale`/`fxmul`/`fxdiv`. DO NOT modify grain.h/fluid.h/fpx.h/couple.h/couple_grain.h.
- **The int64 per-particle SHADER mold (`shaders/cgrain_support.comp.hlsl` CG2 / `shaders/grain_friction.comp.hlsl`
  GR4):** the int64 Jacobi accumulate — `cgf_buoyancy.comp` is the SAME `[numthreads(64,1,1)]` structure, one
  thread per GRAIN, int64, Vulkan-only (NOT in `hf_gen_msl`). The GF1 cross-query passes (re-run each step)
  stay int32 MSL-native.
- **Showcase + registration:** GF1's `--cgf-query-shot` is the immediate plumbing template; CG2's
  `--cgrain-support-shot` is the multi-pass driver (re-query each step → accumulate → integrate).
  `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect
  JSON golden** — the GR2/CP2 lesson), `tests/cgf_test.cpp`.

## Design decisions (locked)
1. **`AccumGrainBuoyancy(world, neighbors)` (the per-grain Jacobi reduction).** For each grain (skip static),
   sum the buoyancy (∝ its fluid-neighbour count, up) + drag (∝ vFluidAvg − vel) over its `gfNeighbors` list
   IN ASCENDING ORDER; apply the impulse as a `vel` delta (`+= FxScale(F, invMass)·dt`). int64.
   `cgf_buoyancy.comp` copies this body VERBATIM (one thread per grain). Deterministic.
2. **`StepCGFBuoyancy(world, dt)` (the driver — fluid STATIC).** Each step: `BuildCGFNeighbors` (GF1 re-query
   from the grains' current positions) → `AccumGrainBuoyancy` (apply the vel delta to the grains) →
   `IntegrateGrains` (grains move under gravity + buoyancy + the radius-aware ground rest). The fluid is NOT
   moved (the reaction is GF3). Run K steps → submerged grains lighten/rise, dry grains pack. Returns nothing.
3. **`MeasureWetDry(world)` (the honest metric helper).** The mean `pos.y` of submerged grains (cnt>0) vs dry
   grains (cnt==0) — a deterministic Q16.16 stat for the wet/dry-contrast proof.
4. **1 new int64 shader `cgf_buoyancy.comp`, Vulkan-only (NOT in `hf_gen_msl`); the GF1 cross-query passes stay
   MSL-native.** Report it. The host driver re-runs the GF1 cross-query passes + the buoyancy pass + the grain
   integrate per step (a `ComputeToComputeBarrier` between sub-passes).
5. **Showcase `--cgf-buoyancy-shot <out>` (Vulkan) AND `--cgf-buoyancy` (Metal) — WIRE BOTH.** The GF1 scene (a
   grain bed + a fluid block over part of it) → run `StepCGFBuoyancySteps` so the SUBMERGED grains LIGHTEN
   (rise/loosen) while the DRY grains pack. Vulkan: the multi-pass GPU driver → **memcmp vs the CPU
   `StepCGFBuoyancySteps` reference**. Metal: runs the CPU reference. Color the grains (wet vs dry) + fluid to
   a side view (the wet grains visibly higher/looser). Golden = `tests/golden/metal/cgf_buoyancy.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
6. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU grain state after K steps == the CPU reference byte-for-byte. Print
     `cgf-buoyancy: {grains:<G>, fluid:<F>, steps:<K>, wetY:<Wy>, dryY:<Dy>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `cgf-buoyancy determinism: two runs BYTE-IDENTICAL`.
   - **(3) WET LIGHTENS (the headline + the HONEST metric):** the submerged grains end HIGHER / less packed than
     the dry grains — `wetY > dryY` by a clear margin, deterministic + two-run byte-identical. Print
     `cgf-buoyancy lightens: wetY <Wy> > dryY <Dy> (submerged grains buoyed)`. **(The HONEST framing, the
     CP2/GR4 caveat shape:** the wet/dry contrast is EMERGENT + within-band — the proof asserts wet > dry by a
     margin + deterministic, NOT an exact buoyancy depth.)
   - **(4) buoy=0 control packs dry:** `kBuoyPerFluid = 0` → the wet and dry grains pack the SAME (`wetY ≈
     dryY`, within an LSB band) — proving the buoyancy does the work. Print `cgf-buoyancy control: buoy=0 packs
     dry (buoyancy does work)`.
   - **Golden discipline: ONLY `tests/golden/metal/cgf_buoyancy.png`; do NOT commit it.** Existing 148 image
     goldens UNTOUCHED.
7. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
8. **Tests `tests/cgf_test.cpp` additions (pure CPU):** `AccumGrainBuoyancy` — a grain over a hand-laid fluid
   list → the exact Q16.16 vel delta (buoyancy up ∝ count; drag damps toward the fluid avg; the fixed-order
   sum); a static grain → untouched; a dry grain (no fluid neighbours) → free-fall; `StepCGFBuoyancy` — the
   submerged grains end higher than the dry ones (and the buoy=0 control packs the same), deterministic,
   grain-order-independent. Clean under `windows-msvc-asan`.
9. **Introspect.** Add exactly `deterministic-cgf-buoyancy` (features) + `--cgf-buoyancy-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (the GR2/CP2
   lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + barrier + read-back path (the GF1/CG2 surface).
  `rhi.h` + `rhi_factory` + backend dirs UNCHANGED. `engine/sim/grain.h` + `fluid.h` + `fpx.h` + `cloth.h` +
  `couple.h` + `couple_grain.h` + `engine/physics/` UNCHANGED. GF1 cgf code + shaders UNCHANGED (GF2 additive).
  Report the seam is empty.

## Out of scope (YAGNI — later GF slices)
The grain→fluid contact reaction (GF3 — GF2 is ONE-WAY, the fluid is static), the full coupled step (GF4),
lockstep (GF5), the lit render (GF6), grain TORQUE / angular (grains are points). GF2 claims ONLY: a
deterministic fluid→grain buoyancy + drag that lightens submerged grains (a wet/dry contrast),
bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs (the honest within-band wet>dry + the
buoy=0-packs-dry control).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 99) + the new `cgf_test` buoyancy cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--cgf-buoyancy-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows the submerged grains LIGHTENED (higher/looser) vs the dry grains (pixel-check the wet vs dry
   regions; the NAV6/CL6 lesson).**
3. Metal: `visual_test --cgf-buoyancy` → new golden `tests/golden/metal/cgf_buoyancy.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm `cgf_buoyancy.comp` is
   correctly NOT MSL-generated (int64); the GF1 cross-query passes still ARE.** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cgf_buoyancy.png` added; the
   other 148 byte-identical (re-run `--cgf-query-shot` → still bit-exact). `git diff master --stat --
   tests/golden` = ONLY `cgf_buoyancy.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cgf-buoyancy` + `--cgf-buoyancy-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/grain.h` + `fluid.h` + `fpx.h` + `cloth.h` + `couple.h` +
   `couple_grain.h` + `engine/physics/` + GF1 cgf code/shaders byte-unchanged). `scripts/verify.ps1` updated:
   `cgf_buoyancy` golden in the Mac loop + `--cgf-buoyancy-shot` in `$vkShots`. `cgf_buoyancy.comp` NOT in
   `hf_gen_msl`.
