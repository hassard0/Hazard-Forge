# Slice GR4 — Deterministic GPU Granular/Sand: TANGENTIAL COULOMB FRICTION (THE NEW PHYSICS — the angle-of-repose money-shot) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #10
> (DETERMINISTIC GPU GRANULAR / SAND, `hf::sim::grain`) — and the flagship's SIGNATURE slice. Adds the ONE
> physics the deterministic-sim trilogy (rigid→cloth→fluid) never modeled: **dry Coulomb friction / shear**.
> After the GR3 normal (non-penetration) correction, project out the inverse-mass-weighted TANGENTIAL
> relative displacement, clamped to `μ · pen` (the Coulomb cone) — so a poured pile HOLDS A SLOPE (a
> self-supporting cone at its angle of repose, with NO container), visibly ≠ GR3's frictionless flat spread.
> The standard Unified-Particle friction, in Q16.16. JACOBI multi-thread (the FL4/GR3 win — NO TDR ceiling).
> Branch: `slice-gr4`. See [[hazard-forge-grain-roadmap]].

**Goal:** Extend `engine/sim/grain.h` (additive — GR1/GR2/GR3 byte-unchanged) with `SolveGrainFriction`
(the per-grain Jacobi tangential-friction Δp over the GR2 neighbor list), fold it into a new
`StepGrainFriction` / `StepGrainFrictionSteps` (the GR3 driver + a friction sub-pass per iteration), and a
`MeasureGrainRepose` slope helper. Add `shaders/grain_friction.comp.hlsl` (int64 → **Vulkan-only** + Metal
CPU reference; reuse the GR3 `grain_contact_apply` + `grain_collide`). Add `--grain-friction-shot` (Vulkan) /
`--grain-friction` (Metal). Bake the integer golden `grain_friction` (the angle-of-repose cone). NO new RHI.

## Design call: the JACOBI tangential friction (the new physics) + the INTEGER bar (strict zero-diff)
Friction is per-grain INDEPENDENT in the Jacobi formulation (each grain accumulates its OWN tangential Δp
from the iteration-start positions into a separate `dp[]`, then ALL apply) — so the GPU pass is
`[numthreads(64,1,1)]` MULTI-THREAD, NO single-thread/TDR (the GR3/FL4 win). int64
(`FxLength`/`FxNormalize`/`fxmul`/`fxdiv`) → `grain_friction.comp` is **Vulkan-only** + the Metal
`--grain-friction` showcase runs the CPU `StepGrainFriction` (byte-identical by construction). Bar: strict
INTEGER (Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels). The GR2 neighbor passes (rebuilt
each step) stay int32 MSL-native.

## The friction constraint (the new math — standard Unified-Particle / PBD friction)
For an overlapping pair (i, j) with `n = unit(p_i − p_j)` and `pen = (r_i+r_j) − |p_i−p_j| > 0`, compute the
TANGENTIAL part of the RELATIVE displacement this step (positions vs the step-start `prev`, snapshotted by
`IntegrateGrains`):
```
Δx_rel = (p_i − prev_i) − (p_j − prev_j)             // relative displacement this step
Δx_t   = Δx_rel − (Δx_rel · n)·n                      // tangential component (subtract the normal part)
t      = |Δx_t|
fmax   = fxmul(μ, pen)                                 // the Coulomb cone (pen = the normal-penetration proxy)
if t <= eps:        no friction
else if t <= fmax:  corr = Δx_t                        // STATIC friction: cancel ALL tangential slip
else:               corr = FxScale(Δx_t, fxdiv(fmax, t))   // KINETIC: clamp to the cone
Δp_i += −fxmul( w_i/(w_i+w_j), 1 )·corr  (i.e. −share·corr)   // inverse-mass-weighted, grain i's half
```
(grain j independently accumulates `Δp_j += +share_j·corr` from the same pair — the symmetric Jacobi
structure; no double-apply.) `μ` is a host-snapped Q16.16 friction coefficient. STATIC grains → Δp 0. This
is the standard frictional contact: static friction (t ≤ μ·pen) fully arrests tangential slip so a grain on
a slope STAYS, giving the pile its angle of repose; kinetic friction (t > μ·pen) clamps the slip to the cone
so steeper-than-repose slopes flow until they stabilize. int64 throughout.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The GR3 contact solve to BUILD ON (this branch's `engine/sim/grain.h`):** `SolveGrainContact` (the
  Jacobi normal-push accumulate — GR4's `SolveGrainFriction` is the SAME per-grain-over-neighbours shape with
  the tangential-clamp term instead of the normal push); `StepGrainContact` (the
  predict→neighbors→K-iters→velocity→collide driver — GR4's `StepGrainFriction` mirrors it, adding the
  friction sub-pass); `CollideGrainPlane`/`CollideGrainSpheres`/`GrainSphereFromBody`; `MeasureGrainPenetration`;
  `kGrainCollideEps`. DO NOT modify the GR1/GR2/GR3 functions — GR4 is additive.
- **The Q16.16 toolbox (read-only, `engine/sim/fpx.h`):** `FxLength` (`fpx.h:319`), `FxNormalize`
  (`fpx.h:96`), `fxmul`/`fxdiv`, `FxISqrt`, `FxAdd`/`FxSub`/`FxScale`, `FxDot` (if present; else compute
  `Δx_rel·n` componentwise via `fxmul`+add). DO NOT modify fpx.h/cloth.h/fluid.h/engine/physics/.
- **The int64 solve SHADERS to mirror (`shaders/grain_contact_dp.comp.hlsl`):** GR3's Jacobi accumulate
  shader — `grain_friction.comp` is the SAME `[numthreads(64,1,1)]` structure, int64, Vulkan-only (NOT in
  `hf_gen_msl`). REUSE `grain_contact_apply.comp` (the pos+=dp apply) + `grain_collide.comp` verbatim.
- **Showcase + registration:** GR3's `--grain-contact-shot` is the immediate template (the multi-pass host
  driver: predict → GR2 neighbor passes → K×{normal-dp→apply→friction-dp→apply} → velocity → collide).
  `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect
  JSON golden** — the GR2 lesson, the GR3 implementer did it correctly), `tests/grain_test.cpp`.

## Design decisions (locked)
1. **`SolveGrainFriction` (the new Jacobi tangential pass, the GR3 `SolveGrainContact` structure).** Per
   grain i (skip static), over its GR2 neighbours j with `pen > 0`: compute the tangential relative
   displacement, clamp to `fxmul(μ, pen)` (static cancels all, kinetic scales by `fxdiv(fmax, t)`), accumulate
   `Δp_i += −share·corr` (share = `fxdiv(w_i, w_i+w_j)`) into a SEPARATE `dp[]` (Jacobi). int64. `grain_friction.comp`
   copies it VERBATIM. Deterministic (the fixed GR2 neighbour order, fixed op order).
2. **`StepGrainFriction` (the GR3 driver + the friction sub-pass).** (1) `IntegrateGrains` predict
   (`prev=pos` snapshot — the friction's step-start anchor); (2) rebuild the GR2 neighbor list from the
   predicted positions; (3) K JACOBI iterations, EACH: `SolveGrainContact`→apply (the GR3 normal push,
   UNCHANGED) THEN `SolveGrainFriction`→apply (the new tangential, reads the post-normal positions); (4)
   velocity `(pos−prev)/dt`; (5) `CollideGrainPlane` + `CollideGrainSpheres`. Two Jacobi sub-passes per
   iteration (normal then friction), each per-grain independent → bit-exact + multi-thread, NO TDR. Returns
   the contact count.
3. **`μ` (the friction coefficient) is a host-snapped Q16.16 constant; the scene is a self-supporting heap
   on FLAT ground (NO container — the headline).** Drop a grain block onto the ground (the GR1 block; the
   GR3 collider sphere is OPTIONAL — prefer NO sphere so the heap is held up by FRICTION ALONE, the
   beyond-UE5 "no container" claim), run enough steps that the grains settle into a CONE with sloped sides.
   `μ` is tuned (a documented constant, e.g. μ≈0.6–1.0) so the repose slope is clearly NON-FLAT and
   recognizable (the GR4/GR3 visual contrast). NO RNG/clock — a locked deterministic config (the
   `fpx::BuildPileWorld` discipline).
4. **1 new int64 shader, Vulkan-only (`grain_friction.comp`); reuse `grain_contact_apply` + `grain_collide`.**
   Report `grain_friction` is NOT in `hf_gen_msl` (int64), the GR2 neighbor passes still ARE. The host driver
   adds a friction dp-pass + apply per iteration (a `ComputeToComputeBarrier` between sub-passes).
5. **Showcase `--grain-friction-shot <out>` (Vulkan) AND `--grain-friction` (Metal) — WIRE BOTH.** Vulkan:
   the multi-pass GPU driver → **memcmp vs the CPU `StepGrainFrictionSteps` reference**. Metal: runs the CPU
   reference. Color the settled cone to a BGRA8 side view. Golden = `tests/golden/metal/grain_friction.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
6. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU grain array after K steps == the CPU `StepGrainFrictionSteps`
     reference byte-for-byte. Print `grain-friction: {particles:<N>, steps:<K>, iters:<I>, mu:<μ>} GPU==CPU
     BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `grain-friction determinism: two runs BYTE-IDENTICAL`.
   - **(3) angle-of-repose / SLOPE STABILITY (the headline + the HONEST metric):** the pile HOLDS A SLOPE —
     `MeasureGrainRepose` returns `{height, baseRadius, slope}` (slope = height/baseRadius, a deterministic
     Q16.16 ratio). Print `grain-friction repose: {height:<H>, baseRadius:<R>, slope:<S>} (holds a slope, mu=<μ>)`.
     **The HONEST framing (the FL4/FPX3 caveat shape):** the angle of repose is EMERGENT and iterative — the
     proof asserts (a) `slope > 0` by a clear margin (a real self-supporting heap, NOT GR3's ~flat spread),
     (b) `slope` is deterministic + two-run byte-identical, (c) `slope` is within a μ-implied tolerance BAND
     — NOT an exact degree value. (A frictionless control run at the same scene gives a much smaller slope —
     the contrast proves friction does work.)
   - **(4) frictionless control / no-op:** `μ = 0` → the result equals the GR3 frictionless `StepGrainContact`
     (friction idle — the pile spreads flat). Print `grain-friction frictionless: mu=0 == GR3 contact (no friction)`.
   - **Golden discipline: ONLY `tests/golden/metal/grain_friction.png`; do NOT commit it.** Existing 132
     image goldens UNTOUCHED.
7. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
8. **Tests `tests/grain_test.cpp` additions (pure CPU):** `SolveGrainFriction` — a hand-laid pair with a
   known tangential slip < μ·pen → fully cancelled (static); slip > μ·pen → clamped to the cone (kinetic);
   μ=0 → Δp 0; a static+dynamic pair pushes only the dynamic; `StepGrainFriction` — a sloped cluster holds
   (a settled heap with slope > the μ=0 control), deterministic, Jacobi GPU-order-independent;
   `MeasureGrainRepose` on a known cone. Clean under `windows-msvc-asan`.
9. **Introspect.** Add exactly `deterministic-grain-friction` (features) + `--grain-friction-shot`
   (showcases). **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`
   (the GR2 lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + barrier + read-back path (the GR3 surface).
  `rhi.h` + `rhi_factory` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `cloth.h` + `fluid.h` +
  `engine/physics/` UNCHANGED. GR1/GR2/GR3 grain code + shaders UNCHANGED (GR4 additive — one new shader +
  reuse `grain_contact_apply`/`grain_collide`). Report the seam is empty.

## Out of scope (YAGNI — later GR slices)
Lockstep/rollback (GR5), the lit render (GR6). Rolling resistance, cohesion, per-pair variable μ,
anisotropic/static-vs-kinetic split coefficients, two-way grain↔rigid friction coupling, restitution. GR4
claims ONLY: deterministic dry Coulomb friction that gives a poured pile a self-supporting angle of repose,
bit-identical CPU↔Vulkan↔Metal, with the cone integer golden + the four proofs (the HONEST slope-stability
metric — an emergent, deterministic, within-band repose slope, NOT exact degrees).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 96) + the new `grain_test` friction cases. Clean
   under `windows-msvc-asan`.
2. **proofs + visual:** `--grain-friction-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows a coherent self-supporting CONE with sloped sides — clearly TALLER/NARROWER than GR3's flat
   spread (pixel-check; the NAV6/CL6 lesson — this is THE money-shot, scrutinize it).**
3. Metal: `visual_test --grain-friction` → new golden `tests/golden/metal/grain_friction.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm `grain_friction.comp`
   is correctly NOT MSL-generated (int64); the GR2 neighbor passes still ARE.** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `grain_friction.png` added;
   the other 132 byte-identical (re-run `--grain-integrate/neighbors/contact-shot` → still bit-exact).
   `git diff master --stat -- tests/golden` = ONLY `grain_friction.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-grain-friction` + `--grain-friction-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h` + `cloth.h` + `fluid.h` + `engine/physics/` +
   GR1/GR2/GR3 grain code/shaders byte-unchanged). `scripts/verify.ps1` updated: `grain_friction` golden in
   the Mac loop + `--grain-friction-shot` in `$vkShots`. `grain_friction.comp` NOT in `hf_gen_msl`.
