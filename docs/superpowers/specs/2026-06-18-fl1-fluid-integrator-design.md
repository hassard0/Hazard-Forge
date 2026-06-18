# Slice FL1 — Deterministic GPU Fluid: Q16.16 PARTICLE INTEGRATOR (BEACHHEAD) (Phase 14 #1) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice of FLAGSHIP #9 —
> DETERMINISTIC GPU FLUID via Position-Based Fluids (`hf::sim::fluid`, header `engine/sim/fluid.h`). This
> BEACHHEAD establishes the core integer primitive: a Q16.16 fixed-point FLUID PARTICLE POOL that
> integrates under gravity, bit-exact CPU↔Vulkan↔Metal. PBF is the DENSITY-CONSTRAINT TWIN of the shipped
> cloth PBD solver — the whole flagship reuses the proven `engine/sim/fpx.h` Q16.16 toolbox + the cloth
> CL1–CL6 mold — and completes the deterministic-sim trilogy (rigid FPX → cloth → fluid), beyond UE5's
> float/non-deterministic Niagara. ZERO new RHI. Branch: `slice-fl1`. See [[hazard-forge-fluid-roadmap]].

**Goal:** Add `engine/sim/fluid.h` (header-only, pure-integer, namespace `hf::sim::fluid`, NO backend
symbols, NO `<cmath>` on the bit-exact path — the `fpx.h`/`cloth.h` discipline; reads `fpx.h` read-only for
the Q16.16 types) with a fluid particle pool: `FluidParticle{FxVec3 pos, prev, vel; fx invMass; uint32
flags}` (std430, the `ClothParticle` packing), `InitBlock` (a deterministic dam-break particle block —
a W×H×D lattice of particles in a corner), and `IntegrateFluid` (semi-implicit Euler under gravity, copied
verbatim from `fpx::IntegrateStep`/`cloth::IntegrateParticles`). Add `shaders/fluid_integrate.comp.hlsl`
(one thread per particle — independent, multi-thread, NO single-thread/TDR exposure; int64 fxmul →
Vulkan-only + Metal CPU path), a `--fluid-integrate-shot` (Vulkan) / `--fluid-integrate` (Metal) showcase,
the `fluid_integrate` integer golden (the free-falling particle block, CPU-colored from the integer
read-back → strict ZERO-DIFFERING-PIXEL cross-backend), and `tests/fluid_test.cpp` pinning CPU==shader
integrate math.

## Design call: INTEGER bit-exact beachhead (the CL1/FPX1 argument, reused)
FL1 is pure integer by construction (the fpx.h/cloth.h discipline): the particle pool is host-snapped to
Q16.16 integers, the shader copies the CPU integrate math VERBATIM and does ZERO float (`vel += gravity·dt;
pos += vel·dt`, all `fxmul`; floor-clamp at the ground), and the particle buffer is `memcmp`'d GPU==CPU
with ZERO tolerance. The golden is the CPU-colored integer particle read-back → literally Vulkan==Metal
BIT-IDENTICAL (the strict zero-differing-pixel bar). Like `cloth_integrate`/`fpx_integrate`, the int64
`fxmul` (gravity·dt over Q16.16 overflows int32) makes `fluid_integrate.comp` Vulkan-only (DXC compiles
int64; glslc cannot) + the Metal showcase runs the CPU `IntegrateFluid` (byte-identical by construction —
the CL1 convention). Particles are INDEPENDENT in FL1 (no constraints/neighbors yet — those are FL2–FL4),
so one thread per particle → order-independent → race-free → NO single-thread/TDR exposure (unlike the
later FL4 solve).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The Q16.16 toolbox to REUSE (read-only):** `engine/sim/fpx.h` — `fx` (int32 Q16.16), `fxmul`
  (int64 intermediate), `FxVec3`, `FxAdd`/`FxSub`/`FxScale`, `kOne`/`kFrac`, `IntegrateStep` (the
  semi-implicit Euler + ground floor-clamp). `#include "sim/fpx.h"` read-only; fpx.h stays byte-unchanged.
- **The beachhead structural template — CL1 (the cloth integrator, the EXACT twin):** `engine/sim/cloth.h`
  `ClothParticle` (the std430 packing) + `IntegrateParticles`/`IntegrateParticle` + `InitGrid`;
  `shaders/cloth_integrate.comp.hlsl` (one thread per particle, int64, Vulkan-only, NOT in `hf_gen_msl`) +
  `RunClothIntegrateShowcase` in `metal_headless/visual_test.mm` (Metal runs the CPU integrate) + the
  `--cloth-integrate-shot` branch in `samples/hello_triangle/main.cpp`. FL1 is the same shape over a 3D
  particle BLOCK instead of a 2D sheet. Mirror CL1 EXACTLY.
- **The int64-Vulkan-only + Metal-CPU convention:** `cloth_integrate.comp` / `fpx_integrate.comp` are NOT
  in `metal_headless/CMakeLists.txt`'s `hf_gen_msl` list (int64 → Vulkan-only); the Metal showcase runs
  the CPU reference. `fluid_integrate.comp` follows this.
- **The integer-golden showcase discipline:** CL1's `--cloth-integrate-shot` (ReadBuffer the integer
  particles, memcmp GPU==CPU, CPU-color a debug image, strict zero-diff). FPX1's `--fpx-shot`.
- **RHI compute envelope (must fit, ZERO additions):** `engine/rhi/rhi.h` — `BindComputePipeline`,
  `BindStorageBuffer`, `ComputePushConstants`, `DispatchCompute`, `ComputeToComputeBarrier`, `ReadBuffer`
  (the CL1/FPX1 set). `rhi.h` stays byte-unchanged.
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--fluid-integrate-shot` standalone arg
  branch — a standalone `if` at the TOP of the arg loop, the C1061 avoidance; main.cpp has `/bigobj` +
  `/STACK` — + DXC list), `metal_headless/{visual_test.mm,CMakeLists.txt}` (`RunFluidIntegrateShowcase` +
  `--fluid-integrate` — int64 Vulkan-only so the Metal showcase runs the CPU `IntegrateFluid`, NO
  `hf_gen_msl` entry), `engine/editor/introspect.cpp` (+`deterministic-fluid-integrate` feature +
  `--fluid-integrate-shot` showcase) + `tests/introspect_test.cpp` + the JSON golden, `tests/CMakeLists.txt`
  (register `fluid_test`), `scripts/verify.ps1` (`fluid_integrate` golden in the Mac loop +
  `--fluid-integrate-shot` in `$vkShots`).

## Design decisions (locked)
1. **The particle pool model.** `FluidParticle{ FxVec3 pos, prev, vel; fx invMass; uint32 flags; }` — all
   int32, std430-packable (the `ClothParticle` discipline, no padding holes). `flags` reserved (e.g. bit0
   = a future static/boundary particle). A deterministic `InitBlock(config)`: a W×H×D block of fluid
   particles at a spacing, in a corner above the ground (the classic dam-break initial condition). All
   host-snapped Q16.16; the float layout constants computed once at build + snapped (NOT in the per-step
   integer sim). Pick a modest count for FL1 (e.g. 8×8×8 = 512 or 10×10×10 = 1000 — well under the FL4 TDR
   budget so the same scene survives the later slices).
2. **The integrate (the bit-exact core).** `IntegrateFluid(particles, gravity, dt, groundY)`: per particle
   `vel = FxAdd(vel, FxScale(gravity, dt))`; `prev = pos`; `pos = FxAdd(pos, FxScale(vel, dt))`; floor-clamp
   `pos.y >= groundY` (the `IntegrateStep` ground clamp). Pure integer, copied verbatim CPU↔shader. (FL1 has
   NO neighbors/density/constraints — that is FL2–FL4; FL1 is free-fall + ground, the integrator beachhead
   — the dam-break block falls and piles at the ground.)
3. **GPU pipeline (one thread per particle).** `fluid_integrate.comp`: one thread per particle runs K
   integrate steps (or 1 step × K dispatches — pick one, document; K steps in one thread is simplest since
   particles are independent in FL1), reads/writes `gParticles`. NO atomics (independent). int64 →
   Vulkan-only + Metal CPU path (the CL1 convention). Host-snapped integers in → integers out → GPU==CPU
   bit-exact.
4. **Showcase `--fluid-integrate-shot <out>` (Vulkan, main.cpp) AND `--fluid-integrate` (Metal,
   visual_test.mm — WIRE BOTH; confirm visual_test.mm + `#include "sim/fluid.h"`).** A deterministic
   dam-break block (e.g. 10×10×10 = 1000 particles) in a corner, K integrate steps (e.g. 60) under gravity
   → the block falls and piles at the ground (a recognizable falling/settling particle mass). ReadBuffer
   the integer `gParticles`; **memcmp GPU == the CPU `IntegrateFluid` reference (the make-or-break)**;
   CPU-color a side-view debug image of the particle positions (each particle a dot at its integer pos) →
   `tests/golden/metal/fluid_integrate.png` (baked on the Mac by the CONTROLLER — DO NOT commit).
5. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance / GPU==CPU (make-or-break):** `gParticles` equals the CPU `IntegrateFluid` reference
     byte-for-byte after K steps. Print `fluid-integrate: {particles:<N>, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism (same backend):** two runs → BYTE-IDENTICAL. Print `fluid-integrate determinism:
     two runs BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the particles fell (moved down) and piled at the ground (`moved > 0`,
     `restingAtGround > 0` — a coherent falling/settling block). Print `fluid-integrate coverage: <M>
     moved, <R> at ground (coherent dam-break fall)`.
   - **(4) empty / no-op:** zero gravity (or zero particles) → no movement / cleared. Print
     `fluid-integrate static: no gravity (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/fluid_integrate.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 123 image goldens UNTOUCHED.
6. **Cross-backend bar (INTEGER, strict).** GPU==CPU-by-construction (Metal runs the CPU IntegrateFluid;
   Vulkan the int64 shader == that CPU) → Vulkan==Metal BIT-IDENTICAL: the golden is the CPU-colored
   integer read-back; the controller's cross-backend check is the STRICT ZERO-DIFFERING-PIXEL compare. Any
   nonzero cross-backend pixel diff is a real bug.
7. **Tests `tests/fluid_test.cpp` additions (pure CPU):** a single free particle → the expected Q16.16 pos
   after N steps (closed-form, hand-checked like CL1/FPX1); the ground floor-clamp; two runs identical;
   zero-gravity → unchanged; `InitBlock` produces the expected particle count + positions. Clean under
   `windows-msvc-asan`. (The integrate is also golden-verified.)
8. **Introspect.** Add exactly `deterministic-fluid-integrate` (features) + `--fluid-integrate-shot`
   (showcases). Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None expected.** Pure SSBO compute: `BindStorageBuffer` (gParticles), `DispatchCompute`,
  `ComputePushConstants` (count + gravity + dt + steps), `ReadBuffer` — all pre-existing (the CL1/FPX1 set).
  `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. `engine/sim/fpx.h` + `engine/sim/cloth.h`
  + `engine/physics/` UNCHANGED (fluid is the additive sibling; fpx.h read-only). If a genuinely-new RHI
  need surfaces, STOP and report it. Report the seam.

## Out of scope (YAGNI — later FL slices)
The grid-hash neighbor search (FL2), the density + λ kernel (FL3), the PBF density-constraint solve (FL4 —
the make-or-break), lockstep/rollback (FL5), the float lit-3D render (FL6). FL1 is ONLY the integer
particle pool + gravity integrate + ground + its bit-exact golden. No neighbors, no density, no
constraints, no float.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 94) + the new `fluid_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--fluid-integrate-shot` on Vulkan: GPU==CPU memcmp BIT-EXACT + determinism +
   coverage + static no-op; a coherent falling/settling dam-break block image. Run under the
   Vulkan-validation gate → ZERO VUID in the OUTPUT (set BOTH `VK_LAYER_PATH` to the conan
   `...\.conan2\p\...\layers` dir AND `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer
   LOADED = zero "not found" lines).
3. Metal: `visual_test --fluid-integrate` → new golden `tests/golden/metal/fluid_integrate.png`; two runs
   DIFF 0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm
   fluid_integrate is correctly EXCLUDED from `hf_gen_msl` (int64 Vulkan-only) and the Metal showcase runs
   the CPU IntegrateFluid.** Cross-backend = STRICT ZERO-DIFFERING-PIXEL (controller-measured), NOT the
   float baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fluid_integrate.png`
   added; the other 123 byte-identical. `git diff master --stat -- tests/golden` = ONLY
   `fluid_integrate.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fluid-integrate` + `--fluid-integrate-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report fluid_integrate int64 Vulkan-only + Metal CPU
   path). New `engine/sim/fluid.h`; `engine/sim/fpx.h` + `engine/sim/cloth.h` + `engine/physics/`
   UNTOUCHED. `scripts/verify.ps1` updated: `fluid_integrate` golden in the Mac loop +
   `--fluid-integrate-shot` in `$vkShots`.
