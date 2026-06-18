# Slice CG2 — Deterministic Rigid↔Grain Coupling: SUPPORT + DRAG (grain→body, the crux) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #12 (DETERMINISTIC
> TWO-WAY RIGID↔GRAIN COUPLING, `hf::sim::cgrain`). The FIRST momentum exchange: each `fpx::FxBody` sums, over
> its CG1 gathered grain list, a **contact-support** impulse (the grain bed pushing the body out/up along the
> contact normals) + a **drag** impulse (toward the local grain velocity), and the body decelerates and RESTS
> on the bed. ONE-WAY for now (grain → body; the body↔grain reaction is CG3). THE CRUX of the flagship: the
> body-force reduction over N grains is order-sensitive, so it is summed **in the fixed CG1 order** with a
> **tiny body count** — one thread per body, multi-thread OVER bodies → NO single-thread TDR ceiling. The CP2
> twin, with contact-support physics instead of buoyancy. Branch: `slice-cg2`. See [[hazard-forge-couple-grain-roadmap]].

**Goal:** Extend `engine/sim/couple_grain.h` (additive — CG1 byte-unchanged) with `AccumBodyGrainForces(world,
query)` (the per-body contact-support+drag impulse accumulate over the CG1 gathered list) + `StepCGrainSupport`
/ `StepCGrainSupportSteps` (re-query → accumulate → integrate the bodies; the grains held STATIC) + a
`MeasureRestLine` helper. Add `shaders/cgrain_support.comp.hlsl` (int64 → **Vulkan-only** + Metal CPU
reference). Add `--cgrain-support-shot` (Vulkan) / `--cgrain-support` (Metal). Bake the integer golden
`cgrain_support`. NO new RHI.

## Design call: int64 one-thread-per-body (the crux) + the INTEGER bar (strict zero-diff)
The support/drag math is int64 (`FxLength`/`FxNormalize`/`fxmul`/`fxdiv`) → `cgrain_support.comp` is
**Vulkan-only** + the Metal `--cgrain-support` showcase runs the CPU `StepCGrainSupport` (byte-identical by
construction, the CP2/FL4 split). The CG1 grid-hash query (re-run each step) stays int32 MSL-native. Bar:
strict INTEGER (Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels).

**THE CRUX — the order-sensitive body-force reduction (the CP2 lesson):** summing N grain impulses onto one
body is a reduction; the integer zero-diff bar demands a PINNED summation order. CG2 sums each body's
contributions **in the fixed CG1 gathered order** (`bodyGrains[bodyStart[i] .. bodyStart[i+1])`, ascending —
the deterministic CG1 emit order). The GPU dispatch is **one thread per body** (a serial inner loop over that
body's gathered list) → **multi-thread OVER bodies**, NOT over grains. With a TINY body count (1–few, ≪ the
~2s watchdog), no body's inner loop approaches the single-thread TDR ceiling. **CAVEAT (the new wrinkle vs
CP2): a body resting ON a sand bed gathers MORE grains than a body floating in fluid (a bed supports through
many simultaneous contacts), so the per-body inner loop is LONGER than CP2's (~30) case — still bounded and
far under the ~2s watchdog for a tiny body count, but flag it. If body counts ever scale up, a deterministic
integer atomic-add reduction is needed — explicitly OUT of scope for CG1–CG6 (the swraster 64-bit-atomics
caveat).**

## The support + drag model (the new physics — contact-support from a grain bed)
For body i, over its CG1 gathered grains j (each in/near the body's sphere):
```
// SUPPORT — the contact push from each overlapping grain (the body rests ON the bed it overlaps):
d = body.pos − grain[j].pos ; dist = FxLength(d) ; pen = (body.radius + grain[j].radius) − dist
if (pen > 0) {                                          // the grain is inside the body -> a contact
    n = FxNormalize(d)                                  // contact normal, body pushed AWAY from the grain
    F_support += FxScale(n, fxmul(kSupport, pen))       // ∝ penetration along the contact normal (summed)
}
// DRAG — toward the local grain velocity (CG2: the grains are static, so this DAMPS the body):
vGrainAvg = (Σ_j grain[j].vel) / gatheredCount          // fixed-order int sum / count
F_drag = fxmul(kDrag, (vGrainAvg − body.vel))           // per axis
// apply as an impulse -> velocity delta:
body.vel += FxScale(F_support + F_drag, body.invMass) · dt     // (CG2: linear only — sphere body, no torque)
```
`kSupport`/`kDrag` are host-snapped Q16.16 constants tuned so the body **rests on the bed** (the net upward
contact support balances gravity) — NOT crashing through, NOT bouncing out. The body settles where enough
grains overlap it (from below) that `F_support ≈ gravity` — an emergent **rest line**, NOT an exact sink
depth (the GR4-repose / CP2-float-line caveat shape). CG2 is LINEAR (vel only); a sphere body's contact
support acts through its centre (no net torque), so `angVel` is untouched until a future asymmetric-bed
refinement.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The CP2 force reduction to MIRROR (`engine/sim/couple.h`):** `AccumBodyForces` (the per-body fixed-order
  reduction, one-thread-per-body, the dp into body.vel — `couple.h:261-300`), `StepCoupleBuoyancy` (the
  re-query → accumulate → integrate driver, `couple.h`), `MeasureFloatLine`. CG2's `AccumBodyGrainForces` is
  the SAME shape with the contact-support term (Σ penetration·normal) instead of buoyancy (∝ count).
- **The CG1 query to BUILD ON (this branch's `engine/sim/couple_grain.h`):** `CGrainWorld`, `GatherBodyGrains`
  → `CGrainQuery{bodyStart, bodyGrains}` (the per-body gathered list CG2 sums over), `BodyGrainAccept`. DO NOT
  modify CG1's functions — CG2 is additive.
- **The grain + body to REUSE (`engine/sim/grain.h` + `fpx.h`, read-only):** `grain::GrainParticle` (pos, vel,
  radius, flags), `fpx::FxBody`, `fpx::IntegrateBody` (`fpx.h:149`, the body integrate — reuse verbatim after
  the force accumulate), `fpx::ResolveGround` (`fpx.h:329`, the bed-floor clamp), `FxLength`/`FxNormalize`/
  `FxScale`/`fxmul`/`fxdiv`. DO NOT modify fpx.h/grain.h/fluid.h/cloth.h/couple.h.
- **The int64 solve SHADER mold (`shaders/couple_buoyancy.comp.hlsl` — CP2):** the one-thread-per-body int64
  reduction shader — `cgrain_support.comp` is the SAME `[numthreads(64,1,1)]` one-thread-per-body structure,
  int64, Vulkan-only (NOT in `hf_gen_msl`). The CG1 query passes (re-run each step) stay int32 MSL-native.
- **Showcase + registration:** CG1's `--cgrain-query-shot` is the immediate plumbing template; CP2's
  `--couple-buoyancy-shot` is the multi-pass driver (re-query each step → accumulate → integrate).
  `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect
  JSON golden** — the GR2/CP2 lesson), `tests/cgrain_test.cpp`.

## Design decisions (locked)
1. **`AccumBodyGrainForces(world, query)` (the fixed-order reduction).** For each body (skip static / non-
   dynamic), sum the contact-support (Σ pen·n over overlapping gathered grains) + drag (∝ vGrainAvg − vel)
   over its CG1 gathered list IN ASCENDING ORDER; apply the impulse as a `vel` delta (`+= FxScale(F,
   invMass)·dt`). int64. `cgrain_support.comp` copies this body VERBATIM (one thread per body). Deterministic.
2. **`StepCGrainSupport(world, dt)` (the driver — grains STATIC).** Each step: `GatherBodyGrains` (CG1
   re-query from the bodies' current positions) → `AccumBodyGrainForces` (apply the vel delta) → `IntegrateBody`
   + `ResolveGround` for each body. The grains are NOT moved (the reaction is CG3). Run K steps → the body
   falls, contacts the bed, rests at a rest line. Returns nothing.
3. **`MeasureRestLine(world)` (the honest metric helper).** The settled body's `pos.y` (or the mean over
   bodies) — a deterministic Q16.16 stat for the rest-line proof.
4. **1 new int64 shader `cgrain_support.comp`, Vulkan-only (NOT in `hf_gen_msl`); the CG1 query passes stay
   MSL-native.** Report it. The host driver re-runs the CG1 query passes + the support pass + the integrate
   per step (a `ComputeToComputeBarrier` between sub-passes).
5. **Showcase `--cgrain-support-shot <out>` (Vulkan) AND `--cgrain-support` (Metal) — WIRE BOTH.** The CG1
   scene (a settled grain bed) + 1–few `FxBody` spheres dropped ABOVE the bed. Run `StepCGrainSupportSteps`
   enough steps that the body falls, contacts the bed, and RESTS on it (damped by drag). Vulkan: the multi-pass
   GPU driver → **memcmp vs the CPU `StepCGrainSupportSteps` reference**. Metal: runs the CPU reference. Color
   the bodies + bed to a side view (the body resting on the bed). Golden = `tests/golden/metal/cgrain_support.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
6. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU body state after K steps == the CPU reference byte-for-byte. Print
     `cgrain-support: {bodies:<B>, grains:<N>, steps:<K>, restY:<Y>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `cgrain-support determinism: two runs BYTE-IDENTICAL`.
   - **(3) RESTS (the headline + the HONEST metric):** the body settles at a rest line ON the bed — `restY >
     groundY + radius` by a clear margin (it did NOT crash through to the floor) AND `restY` is bounded above
     (it did NOT bounce out), and `restY` is deterministic + two-run byte-identical. Print `cgrain-support
     rests: restY <Y> (above bed floor groundY+r=<R>, supported on the bed)`. **(The HONEST framing, the GR4/CP2
     caveat shape:** the rest line is EMERGENT + iterative — the proof asserts it rests by a margin +
     deterministic + within a support-coeff-implied band, NOT an exact sink depth.)
   - **(4) zero-support control SINKS:** `kSupport = 0` → the body sinks through to the bed floor (`restY ==
     groundY + radius`) — proving the contact support does the work. Print `cgrain-support control: support=0
     sinks through (support does work)`.
   - **Golden discipline: ONLY `tests/golden/metal/cgrain_support.png`; do NOT commit it.** Existing 142 image
     goldens UNTOUCHED.
7. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
8. **Tests `tests/cgrain_test.cpp` additions (pure CPU):** `AccumBodyGrainForces` — a body over a hand-laid
   gathered list → the exact Q16.16 vel delta (support Σ pen·n; drag damps toward the grain avg; the
   fixed-order sum); a static body → untouched; an empty gather → free-fall; `StepCGrainSupport` — a body
   settles to a rest line above the bed floor (and the support=0 control sinks), deterministic,
   body-order-independent. Clean under `windows-msvc-asan`.
9. **Introspect.** Add exactly `deterministic-cgrain-support` (features) + `--cgrain-support-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (the GR2/CP2
   lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + barrier + read-back path (the CG1/CP2 surface).
  `rhi.h` + `rhi_factory` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` +
  `couple.h` + `engine/physics/` UNCHANGED. CG1 cgrain code + shaders UNCHANGED (CG2 additive). Report the
  seam is empty.

## Out of scope (YAGNI — later CG slices)
The grain reaction / displacement body→grain (CG3 — CG2 is ONE-WAY, the grains are static), the full coupled
step (CG4), lockstep (CG5), the lit render (CG6), support TORQUE / angular coupling (CG4+, a sphere body has
none), rolling resistance, cohesion. CG2 claims ONLY: a deterministic grain→body contact support + drag that
rests a body on a grain bed at an emergent rest line, bit-identical CPU↔Vulkan↔Metal, with the integer golden
+ the four proofs (the honest within-band rest line + the support=0-sinks control).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 98) + the new `cgrain_test` support cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--cgrain-support-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows the body resting ON the bed (pixel-check — the body is above the bed floor, supported, not
   crashed through, not bounced out; the NAV6/CL6 lesson).**
3. Metal: `visual_test --cgrain-support` → new golden `tests/golden/metal/cgrain_support.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm `cgrain_support.comp`
   is correctly NOT MSL-generated (int64); the CG1 query passes still ARE.** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cgrain_support.png` added; the
   other 142 byte-identical (re-run `--cgrain-query-shot` → still bit-exact). `git diff master --stat --
   tests/golden` = ONLY `cgrain_support.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cgrain-support` + `--cgrain-support-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple.h` +
   `engine/physics/` + CG1 cgrain code/shaders byte-unchanged). `scripts/verify.ps1` updated: `cgrain_support`
   golden in the Mac loop + `--cgrain-support-shot` in `$vkShots`. `cgrain_support.comp` NOT in `hf_gen_msl`.
