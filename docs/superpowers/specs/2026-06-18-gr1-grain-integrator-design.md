# Slice GR1 — Deterministic GPU Granular/Sand: Q16.16 GRAIN INTEGRATOR (the BEACHHEAD of FLAGSHIP #10) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice of FLAGSHIP #10
> (DETERMINISTIC GPU GRANULAR / SAND via Position-Based granular dynamics, `hf::sim::grain`) — the
> **4th member of the deterministic-sim family** (rigid `fpx` → cloth → fluid → **grain**), adding the one
> physics the trilogy never modeled: **dry friction / shear (angle-of-repose)**. GR1 is ONLY the integer
> grain POOL + gravity integrate + radius-aware ground rest — the integrator beachhead. NO neighbors (GR2),
> NO frictionless contact solve (GR3), NO Coulomb friction (GR4 — the new physics), NO lockstep (GR5),
> NO float render (GR6). Branch: `slice-gr1`. See [[hazard-forge-grain-roadmap]] (to be created) and the
> flagship-#10 scout recommendation. The STRUCTURAL TWIN of the FL1/CL1 integer beachhead: a pure-integer
> per-particle update proven GPU==CPU BIT-EXACT with a cross-backend BIT-IDENTICAL integer golden.

**Goal:** Create `engine/sim/grain.h` (header-only, namespace `hf::sim::grain`, pure integer, `#include
"sim/fpx.h"` read-only) with a `GrainParticle` pool, a deterministic `InitGrainBlock` (a dropped block of
grains), the per-particle semi-implicit-Euler integrate (`IntegrateGrainParticle` / `IntegrateGrains` /
`IntegrateGrainSteps`) under gravity with a **radius-aware** ground rest, and a `grain_integrate.comp.hlsl`
shader that runs the EXACT per-particle math. Add `--grain-integrate-shot` (Vulkan) / `--grain-integrate`
(Metal). Bake the integer golden `grain_integrate`. Make-safe: a NEW header + NEW showcase + NEW golden;
reuse the `fpx.h` Q16.16 toolbox read-only — **NO new RHI, NO new sibling-header edits**.

## Design call: the INTEGER bar (strict zero-diff), the FL1/CL1 discipline
GR1 is pure fixed-point — the GPU consumes the host-snapped Q16.16 `GrainParticle` array and does ZERO
floating point: every step is `(int64)a*b >> kFrac` (arithmetic right shift on int64 — deterministic +
identical on every compiler/vendor) + integer add + integer compare. In GR1 each grain is INDEPENDENT (no
inter-grain coupling until GR2–GR4), so the GPU per-thread write is order-independent / race-free with NO
atomics, and two runs are byte-identical. **The bar is therefore STRICT: Vulkan == Metal ZERO differing
pixels** (the FL1/FL2/CL1 integer-golden bar, NOT the float visresolve bar — that arrives only at GR6).

## The int32-vs-int64 decision (the FL1/CL1 lesson, documented — copy it verbatim)
The integrate is `vel += gravity*dt; pos += vel*dt`, both component-wise `fxmul` — the SAME form as
`fluid_integrate.comp` / `cloth_integrate.comp` / `fpx_integrate.comp`, which needed **int64** because the
`(int64)a*b` product before `>>kFrac` exceeds int32 for Q16.16 `gravity*dt` (gravity ≈ −9.8·65536 = −642253;
products of two Q16.16 world-scale values blow past 2^31). To stay bit-exact to this int64-intermediate
reference WITHOUT overflow fragility, `shaders/grain_integrate.comp.hlsl` uses **int64** and is therefore
**VULKAN-SPIR-V-ONLY** (glslc — the Metal HLSL→SPIR-V→MSL frontend — cannot parse `int64_t` in HLSL), NOT
in the Metal `hf_gen_msl` list; the Metal `--grain-integrate` showcase runs the CPU `grain::IntegrateGrains`
(the SAME bit-exact reference the Vulkan GPU==CPU memcmp compares against) → byte-identical to the Vulkan GPU
result BY CONSTRUCTION. Same established convention as `fluid_integrate.comp` (`fluid.h:28-37`).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The Q16.16 toolbox (read-only, `engine/sim/fpx.h`):** `fx` (int32 Q16.16, `fpx.h:46`), `fxmul`
  (`fpx.h:54`, the int64-intermediate multiply), `FxVec3` + `FxAdd`/`FxSub`/`FxScale` (`fpx.h:59-72`),
  `kOne`/`kFrac` (`fpx.h:47-48`). DO NOT modify `fpx.h` — grain is the additive sibling that `#include`s it
  read-only, exactly as `fluid.h:52` / `cloth.h` do.
- **The integrator math to MIRROR (the closest twin — `engine/sim/fluid.h`):** `FluidParticle`
  (`fluid.h:82-88`, the 44-byte std430 packing), `InitBlock` (`fluid.h:116-135`), `IntegrateFluidParticle`
  (`fluid.h:148-165`, the semi-implicit Euler + ground clamp), `IntegrateFluid`/`IntegrateFluidSteps`
  (`fluid.h:171-184`). GR1 is the SAME shape, with two deliberate deltas (below): a first-class `radius`
  field and a radius-aware ground rest.
- **The integer-golden discipline (the bar to copy):** FL1/FL2/CL1 — per-backend Metal golden baked by the
  CONTROLLER, strict zero-differing-pixel cross-vendor, GPU==CPU bit-exact memcmp, two-run determinism.
- **Showcase + registration:** the FL1 `--fluid-integrate-shot` template in `samples/hello_triangle/main.cpp`
  (the Vulkan compute showcase: dispatch `grain_integrate`, read back the SSBO, memcmp vs CPU, color-code to
  a BGRA8 image) + the Metal `--fluid-integrate` branch in `metal_headless/visual_test.mm` (runs the CPU
  reference, color-codes the SAME way). `scripts/verify.ps1`, `engine/editor/introspect.cpp`,
  `tests/introspect_test.cpp`. main.cpp has `/bigobj` (CL4).

## Design decisions (locked)
1. **`GrainParticle` carries `radius` from the beachhead (the one packing delta vs `FluidParticle`).** The
   std430 GPU mirror is plain int32s — `pos`(3) + `prev`(3) + `vel`(3) + `invMass`(1) + `radius`(1) +
   `flags`(1) = **12 × 4 = 48 bytes, NO padding holes** (memcmp-able; the FL/cloth packing discipline,
   treating `FxVec3` as 3 plain int32s, NOT a vec3, so no 16-byte vec3 alignment — array stride 48 is a
   multiple of the 4-byte scalar alignment). `radius` is the Q16.16 grain radius (carried for GR3 contact /
   GR4 friction); GR1 uses it ONLY for the ground rest. `invMass` = `kOne` dynamic (0 ⇒ STATIC, never
   integrates). `flags` bit0 = STATIC (reserved). The buffer layout is FINAL from the beachhead so GR2–GR6
   add NO struct churn (the FL1 rationale, `fluid.h:78`).
2. **Radius-aware ground rest (the second delta — physically correct, still pure integer).** GR1's ground
   clamp rests the grain's SURFACE on the floor, not its center: `if (pos.y < groundY + radius) { pos.y =
   groundY + radius; if (vel.y < 0) vel.y = 0; }` (`groundY + radius` is a trivial Q16.16 integer add). This
   uses `radius` meaningfully in GR1 and sets up GR3's collider projection. Everything else is the
   `IntegrateFluidParticle` body verbatim: `vel += gravity*dt` (component fxmul), `prev = pos`, `pos +=
   vel*dt` (component fxmul). Static grains (flags bit0 OR invMass 0) are UNTOUCHED. Pure integer, fixed op
   order, no RNG, no clock, each grain INDEPENDENT → order-independent, race-free, two-run bit-identical.
3. **`InitGrainBlock`: a deterministic W×H×D dropped block of grains.** The FL1 `InitBlock` twin — particle
   (ix,iy,iz) at `origin + (ix*spacing, iy*spacing, iz*spacing)`, all at rest (vel 0, prev == pos), dynamic
   (invMass `kOne`, uniform `radius`, flags 0), the block ABOVE the ground so it FALLS and piles. 10×10×10 =
   1000 grains (the FL1 count, well under any budget — GR3/GR4 are Jacobi multi-thread so there is NO
   single-thread TDR ceiling, but keep the SAME scene so it survives every later slice). `spacing` ≥ 2·radius
   so the initial block is non-overlapping (clean hand-off to GR3). Host-snapped Q16.16 constants, NO float
   in the per-step path. **(FL1 LESSON — pick `origin.y` high enough / enough steps that the block actually
   reaches the ground; FL1 had to raise 60→120 steps. The implementer may raise the step count to make the
   golden a recognizable settled pile, an accepted physical fix.)**
4. **Shader `grain_integrate.comp.hlsl` = the EXACT per-particle body, int64, `[numthreads(64,1,1)]`
   multi-thread.** One thread per grain (independent → multi-thread, no atomics, no TDR concern). Copies
   `IntegrateGrainParticle`'s math VERBATIM (the int64 `fxmul` + integrate + prev-snap + radius-aware
   floor-clamp). int64 → Vulkan-SPIR-V-only, NOT in `hf_gen_msl`; the Metal showcase runs the CPU
   `IntegrateGrains`. (Use the K-step loop per thread, the FL1 `IntegrateFluidSteps` driver shape.)
5. **Showcase `--grain-integrate-shot <out>` (Vulkan, main.cpp) AND `--grain-integrate` (Metal,
   visual_test.mm — WIRE BOTH; confirm `visual_test.mm` `#include "sim/grain.h"`).** Vulkan: upload the
   `InitGrainBlock` pool → dispatch `grain_integrate` K steps → read back → **memcmp vs the CPU
   `IntegrateGrainSteps` reference** → color-code the settled grains to a BGRA8 image. Metal: run the CPU
   reference, color-code the SAME way. Golden = `tests/golden/metal/grain_integrate.png` (baked on the Mac
   by the CONTROLLER — DO NOT commit).
6. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU read-back == the CPU `IntegrateGrainSteps` reference byte-for-byte.
     Print `grain-integrate: {particles:<N>, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two dispatches → identical read-back. Print `grain-integrate determinism: two runs
     BYTE-IDENTICAL`.
   - **(3) coverage:** the grains MOVED and piled — print `grain-integrate coverage: <moved> moved, <rest>
     resting at ground` (e.g. all 1000 moved, the bottom layers resting at `groundY + radius`).
   - **(4) static no-op:** a pool of STATIC grains (invMass 0 / flags bit0) → unchanged after K steps. Print
     `grain-integrate static: no-op (static grains unmoved)`.
   - **Golden discipline: ONLY `tests/golden/metal/grain_integrate.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 129 image goldens UNTOUCHED.
7. **Cross-backend bar (INTEGER, strict):** Vulkan read-back == Metal CPU-reference == the committed Metal
   golden, **ZERO differing pixels** (the FL1/FL2 strict integer bar — measured by the controller; ANY
   nonzero diff is a bug, unlike the GR6 float capstone).
8. **Tests `tests/grain_test.cpp` (pure CPU):** `InitGrainBlock` (N grains at the right host-snapped
   positions, uniform radius, dynamic); `IntegrateGrainParticle` (a known grain → the expected post-step
   pos/vel: a free-fall step and a radius-aware ground-rest step hand-checked in Q16.16); `IntegrateGrains`
   order-independence (shuffled vs in-order → identical, since grains are independent); static no-op. Clean
   under `windows-msvc-asan`.
9. **Introspect.** Add exactly `deterministic-grain-integrate` (features) + `--grain-integrate-shot`
   (showcases). Rebake the introspect JSON golden + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the existing compute path (the SSBO + compute pipeline + dispatch + read-back the FL1/FPX1
  set already uses) — all pre-existing. `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED.
  `engine/sim/fpx.h` + `engine/sim/cloth.h` + `engine/sim/fluid.h` + `engine/physics/` UNCHANGED (grain is a
  NEW additive sibling header; `fpx.h` is `#include`d read-only). Report the seam is empty.

## Out of scope (YAGNI — later GR slices)
Neighbor search (GR2), grain-grain contact projection (GR3), Coulomb friction / angle-of-repose (GR4 — the
new physics), lockstep/rollback (GR5), the lit 3D render (GR6). Per-grain variable radius / polydisperse
piles, cohesion, rolling resistance, two-way rigid coupling — all future. GR1 claims ONLY: a deterministic
fixed-point grain POOL that falls under gravity and rests on the ground, bit-identical CPU↔Vulkan↔Metal,
with the integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 95) + the new `grain_test` cases. Clean under
   `windows-msvc-asan` (build + run `grain_test` + `introspect_test`).
2. **proofs + visual:** `--grain-integrate-shot` on Vulkan: the 4 proofs (GPU==CPU bit-exact + determinism +
   coverage + static no-op) + exit 0. Run under the Vulkan-validation gate → ZERO VUID in the OUTPUT (set
   BOTH `VK_LAYER_PATH` to the conan `...\.conan2\p\...\layers` dir AND
   `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer LOADED = zero "not found" lines).
   **VERIFY the rendered image actually shows the settled grain block (pixel-check the colored grain region)
   — do NOT trust the proof's claim alone (the NAV6/CL6 lesson).**
3. Metal: `visual_test --grain-integrate` → new golden `tests/golden/metal/grain_integrate.png`; two runs
   DIFF 0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm
   `grain_integrate.comp` is correctly NOT MSL-generated (int64 → Vulkan-only).** The cross-vendor
   Vulkan-vs-Metal delta is **STRICT ZERO** (integer golden — unlike the future GR6 render).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `grain_integrate.png` added;
   the other 129 byte-identical. `git diff master --stat -- tests/golden` = ONLY `grain_integrate.png`
   (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-grain-integrate` + `--grain-integrate-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; `engine/sim/fpx.h` + `cloth.h` + `fluid.h` +
   `engine/physics/` byte-unchanged). `scripts/verify.ps1` updated: `grain_integrate` golden in the Mac loop
   + `--grain-integrate-shot` in `$vkShots`.
