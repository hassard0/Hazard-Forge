# Slice CP2 — Deterministic Rigid↔Fluid Coupling: BUOYANCY + DRAG (fluid→body, the crux) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #11 (DETERMINISTIC
> TWO-WAY RIGID↔FLUID COUPLING, `hf::sim::couple`). The FIRST momentum exchange: each `fpx::FxBody` sums,
> over its CP1 gathered fluid-particle list, a **buoyant** impulse (the displaced-fluid restoring force, up) +
> a **drag** impulse (toward the local fluid velocity), and the body floats/sinks/damps. ONE-WAY for now
> (fluid → body; the body↔fluid reaction is CP3). THE CRUX of the flagship: the body-force reduction over N
> particles is order-sensitive, so it is summed **in the fixed CP1 emit order** with a **tiny body count** —
> multi-thread OVER bodies, serial over each body's short list → NO single-thread `[numthreads(1,1,1)]`
> TDR/watchdog ceiling (the CL3/FPX3 limit does not apply). Branch: `slice-cp2`. See [[hazard-forge-couple-roadmap]].

**Goal:** Extend `engine/sim/couple.h` (additive — CP1 byte-unchanged) with `AccumBodyForces(world, query)`
(the per-body buoyancy+drag impulse accumulate over the CP1 gathered list) + `StepCoupleBuoyancy` /
`StepCoupleBuoyancySteps` (re-query → accumulate → integrate the bodies; the fluid held STATIC) + a
`MeasureFloatLine` helper. Add `shaders/couple_buoyancy.comp.hlsl` (int64 → **Vulkan-only** + Metal CPU
reference). Add `--couple-buoyancy-shot` (Vulkan) / `--couple-buoyancy` (Metal). Bake the integer golden
`couple_buoyancy`. NO new RHI.

## Design call: int64 Jacobi-over-bodies (the crux) + the INTEGER bar (strict zero-diff)
The buoyancy/drag math is int64 (`fxmul`/`fxdiv`/`FxScale`/`FxLength`) → `couple_buoyancy.comp` is
**Vulkan-only** + the Metal `--couple-buoyancy` showcase runs the CPU `StepCoupleBuoyancy` (byte-identical by
construction, the FL4/GR3 split). The CP1 grid-hash query (re-run each step) stays int32 MSL-native. Bar:
strict INTEGER (Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels).

**THE CRUX — the order-sensitive body-force reduction (read carefully):** summing N fluid-particle impulses
onto one body is a reduction; the integer zero-diff bar demands a PINNED summation order. CP2 sums each
body's contributions **in the fixed CP1 gathered order** (`bodyParticles[bodyStart[i] .. bodyStart[i+1])`,
ascending — the deterministic CP1 emit order). The GPU dispatch is **one thread per body** (a serial inner
loop over that body's short gathered list) → **multi-thread OVER bodies**, NOT over particles. With a TINY
body count (1–few, ≪ the ~2s watchdog), no body's inner loop approaches the single-thread TDR ceiling that
sank CL3/FPX3 Gauss-Seidel. (Integer addition is exact regardless of order, but the fixed order keeps it
provably bit-identical to the CPU reference and future-proof.) **If body counts ever scale up, a
deterministic integer atomic-add reduction would be needed — explicitly OUT of scope for CP1–CP6 (flag
loudly, like the swraster 64-bit-atomics caveat).**

## The buoyancy + drag model (the new physics — a deterministic Archimedes proxy)
For body i, over its CP1 gathered particles j (each a unit of displaced fluid):
```
// BUOYANCY — the displaced-volume restoring force, opposing gravity (up = −normalize(gravity)):
F_buoy = fxmul(kBuoyPerParticle, gatheredCount[i]) · up        // ∝ displaced volume (the gathered count)
// DRAG — toward the local fluid velocity (CP2: the fluid is static, so this DAMPS the body):
vFluidAvg = (Σ_j particle[j].vel) / gatheredCount[i]           // fixed-order int sum / count
F_drag = fxmul(kDrag, (vFluidAvg − body.vel))                  // per axis
// apply as an impulse → velocity delta (impulse/mass; fold dt into the coeffs):
body.vel += FxScale(F_buoy + F_drag, body.invMass) · dt        // (CP2: linear only — a sphere body has no
                                                               //  buoyancy torque; angular coupling is CP4+)
```
`kBuoyPerParticle`/`kDrag` are host-snapped Q16.16 constants tuned so the body settles at a **float line**
(buoyancy ≈ gravity) inside the pool — NOT sunk to the bed, NOT flying out. A body that gathers MORE
particles (deeper) feels MORE upward force → a stable equilibrium depth (the Archimedes proxy: displaced
volume ∝ gathered count). The float line is **emergent/iterative**, NOT an exact Archimedes depth (the
GR4-repose / FL4-residual caveat shape — see the proof framing). CP2 is LINEAR (vel only); a sphere body's
buoyancy acts through its centre (no torque), so `angVel` is untouched until CP4's non-uniform coupling.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The CP1 query to BUILD ON (this branch's `engine/sim/couple.h`):** `CoupleWorld`, `GatherBodyParticles`
  → `CoupleQuery{bodyStart, bodyParticles}` (the per-body gathered list CP2 sums over), `BodyParticleAccept`.
  DO NOT modify CP1's functions — CP2 is additive.
- **The rigid body integrate to REUSE (`engine/sim/fpx.h`, read-only):** `IntegrateBody(FxBody&, gravity,
  groundY, dt)` (`fpx.h:149` — vel += g·dt, pos += vel·dt, ground clamp at `groundY + radius`), `ResolveGround`
  (`fpx.h:329`, the float-line floor). The body-integrate after the force accumulate is `IntegrateBody`
  VERBATIM (the buoyancy/drag is a `vel` delta applied BEFORE it). `FxScale`/`FxSub`/`FxLength`/`FxNormalize`/
  `fxmul`/`fxdiv` (`fpx.h:46-323`). DO NOT modify fpx.h/fluid.h/cloth.h/grain.h.
- **The int64 solve SHADER mold (`shaders/grain_contact_dp.comp.hlsl` / `couple` is per-body):**
  `couple_buoyancy.comp` is a `[numthreads(64,1,1)]` dispatch ONE THREAD PER BODY (NOT per particle), each
  thread serially summing its body's gathered list then writing the body's new `vel`. int64 → Vulkan-only
  (NOT in `hf_gen_msl`). The CP1 query passes (re-run each step) stay int32 MSL-native.
- **Showcase + registration:** CP1's `--couple-query-shot` is the immediate plumbing template; FL4's
  `--fluid-solve-shot` is the multi-pass host driver template (re-query each step → accumulate → integrate).
  `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect
  JSON golden** — the GR2 lesson), `tests/couple_test.cpp`.

## Design decisions (locked)
1. **`AccumBodyForces(world, query)` (the fixed-order reduction).** For each body (skip static / non-dynamic),
   sum the buoyancy (∝ gathered count, up) + drag (∝ vFluidAvg − vel) over its CP1 gathered list IN ASCENDING
   ORDER; apply the impulse as a `vel` delta (`+= FxScale(F, invMass)·dt`). int64. `couple_buoyancy.comp`
   copies this body VERBATIM (one thread per body). Deterministic (fixed gathered order, fixed op order).
2. **`StepCoupleBuoyancy(world, dt)` (the driver — fluid STATIC).** Each step: `GatherBodyParticles` (CP1
   re-query from the bodies' current positions) → `AccumBodyForces` (apply the vel delta) → `IntegrateBody`
   for each body (gravity + the buoyancy-adjusted vel, ground clamp). The fluid particles are NOT moved (the
   reaction is CP3). Run K steps → the body falls, enters the pool, buoyancy builds as it submerges, settles
   to a float line. Returns nothing (the bodies carry the state).
3. **`MeasureFloatLine(world)` (the honest metric helper).** The settled body's `pos.y` (or the mean over
   bodies) — a deterministic Q16.16 stat for the float-line proof.
4. **1 new int64 shader `couple_buoyancy.comp`, Vulkan-only (NOT in `hf_gen_msl`); the CP1 query passes stay
   MSL-native.** Report it. The host driver re-runs the CP1 query passes + the buoyancy pass + the integrate
   per step (a `ComputeToComputeBarrier` between sub-passes).
5. **Showcase `--couple-buoyancy-shot <out>` (Vulkan) AND `--couple-buoyancy` (Metal) — WIRE BOTH.** The CP1
   scene (a settled fluid pool) + 1–few `FxBody` spheres dropped ABOVE the pool. Run `StepCoupleBuoyancySteps`
   enough steps that the body falls in and SETTLES at a float line (damped by drag). Vulkan: the multi-pass
   GPU driver → **memcmp vs the CPU `StepCoupleBuoyancySteps` reference**. Metal: runs the CPU reference.
   Color the bodies + pool to a side view (the body visible at its float line in the pool). Golden =
   `tests/golden/metal/couple_buoyancy.png` (Mac-baked by the CONTROLLER — DO NOT commit).
6. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU body state after K steps == the CPU reference byte-for-byte. Print
     `couple-buoyancy: {bodies:<B>, particles:<N>, steps:<K>, floatY:<Y>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `couple-buoyancy determinism: two runs BYTE-IDENTICAL`.
   - **(3) FLOATS (the headline + the HONEST metric):** the body settles at a float line INSIDE the pool —
     `floatY > groundY + radius` by a clear margin (it did NOT sink to the bed) AND `floatY` is bounded above
     (it did NOT fly out), and `floatY` is deterministic + two-run byte-identical. Print `couple-buoyancy
     floats: floatY <Y> (above bed groundY+r=<R>, settled in pool)`. **(The HONEST framing, the GR4/FL4
     caveat shape:** the float line is EMERGENT + iterative — the proof asserts it floats by a margin +
     deterministic + within a buoyancy-coeff-implied band, NOT an exact Archimedes depth.)
   - **(4) zero-buoyancy control SINKS:** `kBuoyPerParticle = 0` → the body sinks to the bed (`floatY ==
     groundY + radius`) — proving buoyancy does the work. Print `couple-buoyancy control: buoy=0 sinks to bed
     (buoyancy does work)`.
   - **Golden discipline: ONLY `tests/golden/metal/couple_buoyancy.png`; do NOT commit it.** Existing 136
     image goldens UNTOUCHED.
7. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
8. **Tests `tests/couple_test.cpp` additions (pure CPU):** `AccumBodyForces` — a body over a hand-laid
   gathered list → the exact Q16.16 vel delta (buoyancy up ∝ count; drag damps toward the fluid avg; the
   fixed-order sum); a static body → untouched; `StepCoupleBuoyancy` — a body settles to a float line above
   the bed (and the buoy=0 control sinks), deterministic, body-order-independent. Clean under
   `windows-msvc-asan`.
9. **Introspect.** Add exactly `deterministic-couple-buoyancy` (features) + `--couple-buoyancy-shot`
   (showcases). **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`
   (the GR2 lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + barrier + read-back path (the CP1/FL4 surface).
  `rhi.h` + `rhi_factory` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `fluid.h` + `cloth.h` + `grain.h` +
  `engine/physics/` UNCHANGED. CP1 couple code + shaders UNCHANGED (CP2 additive). Report the seam is empty.

## Out of scope (YAGNI — later CP slices)
The fluid reaction / displacement body→fluid (CP3 — CP2 is ONE-WAY, the fluid is static), the full coupled
step (CP4), lockstep (CP5), the lit render (CP6), buoyancy TORQUE / angular coupling (CP4+, a sphere body has
none), variable per-particle displaced volume, surface-tension. CP2 claims ONLY: a deterministic
fluid→body buoyancy + drag that floats a body at an emergent float line, bit-identical CPU↔Vulkan↔Metal, with
the integer golden + the four proofs (the honest within-band float line + the buoy=0-sinks control).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 97) + the new `couple_test` buoyancy cases. Clean
   under `windows-msvc-asan`.
2. **proofs + visual:** `--couple-buoyancy-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows the body settled at a float line INSIDE the pool (pixel-check — the body is between the bed and
   the surface, not sunk, not flown out; the NAV6/CL6 lesson).**
3. Metal: `visual_test --couple-buoyancy` → new golden `tests/golden/metal/couple_buoyancy.png`; two runs
   DIFF 0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm
   `couple_buoyancy.comp` is correctly NOT MSL-generated (int64); the CP1 query passes still ARE.**
   Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `couple_buoyancy.png` added;
   the other 136 byte-identical (re-run `--couple-query-shot` → still bit-exact). `git diff master --stat --
   tests/golden` = ONLY `couple_buoyancy.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-couple-buoyancy` + `--couple-buoyancy-shot`; introspect
   test updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h` + `fluid.h` + `cloth.h` + `grain.h` +
   `engine/physics/` + CP1 couple code/shaders byte-unchanged). `scripts/verify.ps1` updated:
   `couple_buoyancy` golden in the Mac loop + `--couple-buoyancy-shot` in `$vkShots`. `couple_buoyancy.comp`
   NOT in `hf_gen_msl`.
