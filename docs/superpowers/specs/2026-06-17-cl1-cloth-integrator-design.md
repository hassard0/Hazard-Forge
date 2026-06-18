# Slice CL1 — Deterministic GPU Cloth: Q16.16 INTEGRATOR + GRID BUILD (BEACHHEAD) (Phase 13 #1) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice of FLAGSHIP #8 —
> DETERMINISTIC GPU CLOTH (`hf::sim::cloth`, header `engine/sim/cloth.h`). This BEACHHEAD establishes
> the core integer primitive: a Q16.16 fixed-point cloth PARTICLE LATTICE that integrates under gravity,
> bit-exact CPU↔Vulkan↔Metal. Cloth is a near-verbatim FPX3 PBD twin — the whole flagship reuses the
> proven `engine/sim/fpx.h` Q16.16 toolbox — and extends the FPX5 lockstep/rollback determinism headline
> to DEFORMABLE bodies (UE5's Chaos Cloth/NvCloth is float/non-deterministic). ZERO new RHI. Branch:
> `slice-cl1`. See [[hazard-forge-cloth-roadmap]].

**Goal:** Add `engine/sim/cloth.h` (header-only, pure-integer, namespace `hf::sim::cloth`, NO backend
symbols, NO `<cmath>` on the bit-exact path — the `fpx.h` discipline; reads `fpx.h` read-only for the
Q16.16 types) with a cloth particle lattice: `ClothParticle{FxVec3 pos, prev, vel; fx invMass; uint32
flags}`, `ClothGrid` (a fixed W×H row-major lattice with deterministic pinned-corner flags), and
`IntegrateParticles` (semi-implicit Euler under gravity, copied verbatim from `fpx::IntegrateStep`). Add
`shaders/cloth_integrate.comp.hlsl` (one thread per particle, int32-only → Metal-MSL-native), a
`--cloth-integrate-shot` (Vulkan) / `--cloth-integrate` (Metal) showcase, the `cloth_integrate` integer
golden (the free-falling lattice, CPU-colored from the integer read-back → strict ZERO-DIFFERING-PIXEL
cross-backend), and `tests/cloth_test.cpp` pinning CPU==shader integrate math.

## Design call: INTEGER bit-exact beachhead (the FPX1 argument, reused)
CL1 is pure integer by construction (the fpx.h discipline): the lattice is host-snapped to Q16.16
integers, the shader copies the CPU integrate math VERBATIM and does ZERO float (`vel += gravity·dt; pos
+= vel·dt`, all `fxmul`; floor-clamp at the ground), and the particle buffer is `memcmp`'d GPU==CPU with
ZERO tolerance. The golden is the CPU-colored integer particle read-back → literally Vulkan==Metal
BIT-IDENTICAL (the strict zero-differing-pixel bar). CL1 is INT32-ONLY (`vel += gravity·dt` — gravity·dt
fits int32 for the lattice's modest values; if a product needs int64 like FPX1's `fpx_integrate`, isolate
it, but PREFER int32 so `cloth_integrate.comp` MSL-generates natively and runs as a true GPU pass on BOTH
backends). One thread per particle (independent — no constraints yet in CL1) → order-independent →
race-free → bit-exact regardless of GPU scheduling.

> NOTE on int64 (the FPX1 lesson): `fpx_integrate.comp` needed int64 because `gravity·dt` over Q16.16
> overflowed int32 (gravity ≈ -9.8·65536, ·dt). If CL1's integrate has the same overflow, follow the FPX1
> convention: `cloth_integrate.comp` is Vulkan-only (int64) + the Metal showcase runs the CPU
> `IntegrateParticles` (byte-identical by construction). PREFER int32 if the value range allows (e.g. a
> smaller dt or scaled gravity) — but do NOT sacrifice correctness for it; document the choice. Either way
> the golden is strict zero-diff (CPU-computed).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The Q16.16 toolbox to REUSE (read-only):** `engine/sim/fpx.h` — `fx` (int32 Q16.16), `fxmul`
  (`fpx.h:50`, int64 intermediate), `FxVec3`, `FxAdd`/`FxSub`/`FxScale` (the vec ops), `kOne`/`kSub`,
  `IntegrateStep` (`fpx.h:166` — the semi-implicit Euler + ground floor-clamp `IntegrateParticles` mirrors
  per-particle), the `FxBody` std430 packing discipline (`fpx.h:~110`) for `ClothParticle`. `#include
  "sim/fpx.h"` read-only; fpx.h stays byte-unchanged.
- **The beachhead structural template:** FPX1 — `engine/sim/fpx.h::IntegrateStep` + `shaders/
  fpx_integrate.comp.hlsl` (one thread per body, host-snapped Q16.16, GPU==CPU memcmp) +
  `RunFpxShowcase` in `metal_headless/visual_test.mm` + the `--fpx-shot` branch in
  `samples/hello_triangle/main.cpp` (the integer golden discipline: dispatch → ReadBuffer the integer
  particle array → memcmp vs the CPU reference → CPU-color a debug image → proof prints). NAV1's
  `--nav-raster-shot` is the other integer-golden template.
- **The int32-native-vs-int64-Vulkan-only shader split:** `metal_headless/CMakeLists.txt` `hf_gen_msl`
  allowlist (fpx_pair_* int32 ARE listed/MSL-native; fpx_integrate int64 is NOT — Vulkan-only). Wire
  `cloth_integrate.comp` per its int32/int64 choice. `samples/hello_triangle/CMakeLists.txt` the DXC
  compile list.
- **RHI compute envelope (must fit, ZERO additions):** `engine/rhi/rhi.h` — `BindComputePipeline`,
  `BindStorageBuffer`, `ComputePushConstants`, `DispatchCompute`, `ComputeToComputeBarrier`, `ReadBuffer`
  (the FPX1/MC1/NAV1 set). `rhi.h` stays byte-unchanged.
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--cloth-integrate-shot` standalone arg
  branch — a standalone `if` at the TOP of the arg loop, the C1061 avoidance — + DXC list),
  `metal_headless/{visual_test.mm,CMakeLists.txt}` (`RunClothIntegrateShowcase` + `--cloth-integrate` +
  the shader in `hf_gen_msl` if int32), `engine/editor/introspect.cpp` (+`deterministic-cloth-integrate`
  feature + `--cloth-integrate-shot` showcase) + `tests/introspect_test.cpp` + the JSON golden,
  `tests/CMakeLists.txt` (register `cloth_test`), `scripts/verify.ps1` (`cloth_integrate` golden in the
  Mac loop + `--cloth-integrate-shot` in `$vkShots`).

## Design decisions (locked)
1. **The lattice model.** `ClothGrid{ int W, H; fx spacing; FxVec3 origin; }` + a row-major
   `std::vector<ClothParticle>` (index `r*W + c`). `ClothParticle{ FxVec3 pos, prev, vel; fx invMass;
   uint32 flags; }` — all int32, std430-packable (the `FxBody` discipline, no padding holes). `flags`
   bit0 = PINNED (invMass = 0, never integrates). A deterministic init: a flat W×H sheet at `origin`,
   spacing `spacing`, the TWO top corners (or top row) PINNED. Host-snapped Q16.16; the float layout
   constants computed once at build + snapped (NOT in the per-step integer sim).
2. **The integrate (the bit-exact core).** `IntegrateParticles(grid, particles, gravity, dt, groundY)`:
   per particle, if not PINNED: `vel = FxAdd(vel, FxScale(gravity, dt))` (or `fxmul` componentwise);
   `prev = pos`; `pos = FxAdd(pos, FxScale(vel, dt))`; floor-clamp `pos.y >= groundY` (the
   `IntegrateStep` ground clamp). Pinned particles untouched. Pure integer, copied verbatim CPU↔shader.
   (CL1 has NO constraints — that is CL3; CL1 is free-fall + pin + ground, the integrator beachhead.)
3. **GPU pipeline (one thread per particle).** `cloth_integrate.comp`: one thread per particle runs K
   integrate steps (or 1 step per dispatch × K dispatches — pick one, document; K steps in one thread is
   simplest since particles are independent in CL1), reads/writes `gParticles`. NO atomics (independent).
   INT32 if the value range allows → in `hf_gen_msl` (Metal-native); else int64 → Vulkan-only + Metal CPU
   path (the FPX1 convention). Host-snapped integers in → integers out → GPU==CPU bit-exact.
4. **Showcase `--cloth-integrate-shot <out>` (Vulkan, main.cpp) AND `--cloth-integrate` (Metal,
   visual_test.mm — WIRE BOTH; confirm visual_test.mm + `#include "sim/cloth.h"`).** A deterministic W×H
   sheet (e.g. 24×24) with the top corners pinned, K integrate steps (e.g. 60) under gravity → the
   non-pinned particles fall, the pinned corners hold (so it's a recognizable hanging/falling shape).
   ReadBuffer the integer `gParticles`; **memcmp GPU == the CPU `IntegrateParticles` reference (the
   make-or-break)**; CPU-color a side-view debug image of the particle positions (e.g. each particle a
   dot at its integer pos, pinned corners marked) → `tests/golden/metal/cloth_integrate.png` (baked on the
   Mac by the CONTROLLER — DO NOT commit).
5. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance / GPU==CPU (make-or-break):** `gParticles` equals the CPU `IntegrateParticles`
     reference byte-for-byte after K steps. Print `cloth-integrate: {particles:<N>, pinned:<P>, steps:<K>}
     GPU==CPU BIT-EXACT`.
   - **(2) determinism (same backend):** two runs → BYTE-IDENTICAL. Print `cloth-integrate determinism:
     two runs BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the non-pinned particles fell (moved down) and the pinned ones held —
     a coherent draping/falling lattice (`moved > 0`, pinned fixed). Print `cloth-integrate coverage:
     <M> moved, <P> pinned held (coherent lattice)`.
   - **(4) empty / no-op:** an all-pinned grid (or zero gravity) → no movement / cleared. Print
     `cloth-integrate static: all pinned (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/cloth_integrate.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 117 image goldens UNTOUCHED.
6. **Cross-backend bar (INTEGER, strict).** The particle buffer is host-snapped-integer → ZERO float on
   the GPU (if int32; or GPU==CPU-by-construction if the int64 Vulkan-only path) → Vulkan==Metal
   BIT-IDENTICAL: the golden is the CPU-colored integer read-back; the controller's cross-backend check is
   the STRICT ZERO-DIFFERING-PIXEL compare. Any nonzero cross-backend pixel diff is a real bug.
7. **Tests `tests/cloth_test.cpp` additions (pure CPU):** a single free particle → the expected Q16.16
   pos after N steps (closed-form, hand-checked like FPX1); a pinned particle → never moves; the ground
   floor-clamp; two runs identical; an all-pinned grid → unchanged. Clean under `windows-msvc-asan`. (The
   integrate is also golden-verified.)
8. **Introspect.** Add exactly `deterministic-cloth-integrate` (features) + `--cloth-integrate-shot`
   (showcases). Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None expected.** Pure SSBO compute: `BindStorageBuffer` (gParticles), `DispatchCompute`,
  `ComputePushConstants` (grid dims + gravity + dt + steps), `ReadBuffer` — all pre-existing (the FPX1 set).
  `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. `engine/sim/fpx.h` + `engine/physics/`
  UNCHANGED (cloth is the additive sibling; fpx.h read-only). If a genuinely-new RHI need surfaces, STOP
  and report it. Report the seam.

## Out of scope (YAGNI — later CL slices)
The distance-constraint graph (CL2), the PBD constraint solver (CL3 — the make-or-break), collision
(CL4), lockstep/rollback (CL5), the float lit-3D render (CL6). CL1 is ONLY the integer particle lattice +
gravity integrate + pin + ground + its bit-exact golden. No constraints, no collision, no float.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 93) + the new `cloth_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--cloth-integrate-shot` on Vulkan: GPU==CPU memcmp BIT-EXACT + determinism +
   coverage + static no-op; a coherent falling/draping lattice image. Run under the Vulkan-validation gate
   → ZERO VUID in the OUTPUT (set BOTH `VK_LAYER_PATH` to the conan `...\.conan2\p\...\layers` dir AND
   `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer LOADED = zero "not found" lines).
3. Metal: `visual_test --cloth-integrate` → new golden `tests/golden/metal/cloth_integrate.png`; two runs
   DIFF 0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm
   cloth_integrate MSL-generates (int32 → in `hf_gen_msl`) OR is excluded + CPU-ref'd if int64.**
   Cross-backend = STRICT ZERO-DIFFERING-PIXEL (controller-measured), NOT the float baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cloth_integrate.png`
   added; the other 117 byte-identical. `git diff master --stat -- tests/golden` = ONLY
   `cloth_integrate.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cloth-integrate` + `--cloth-integrate-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report int32-native vs int64-Vulkan-only). New
   `engine/sim/cloth.h`; `engine/sim/fpx.h` + `engine/physics/` UNTOUCHED. `scripts/verify.ps1` updated:
   `cloth_integrate` golden in the Mac loop + `--cloth-integrate-shot` in `$vkShots`.
