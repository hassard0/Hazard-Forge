# Slice CL3 — Deterministic GPU Cloth: PBD CONSTRAINT SOLVER (the make-or-break) (Phase 13 #3) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #8
> (DETERMINISTIC GPU CLOTH, `hf::sim::cloth`, header `engine/sim/cloth.h`). The MAKE-OR-BREAK slice: a
> Position-Based-Dynamics constraint solver that projects the CL2 distance constraints so the cloth holds
> its shape and DRAPES — the verbatim generalization of the proven `fpx::SolveContact` from a sphere-pair
> to a constraint-graph edge. Single-thread serial (Gauss-Seidel is order-dependent) → bit-exact
> CPU↔Vulkan↔Metal (the integer bar). int64 → Vulkan-only SPIR-V + Metal runs the byte-identical CPU
> reference (the FPX3 convention). ZERO new RHI. Branch: `slice-cl3`. See [[hazard-forge-cloth-roadmap]].

**Goal:** Extend `engine/sim/cloth.h` with `SolveDistanceConstraint` (project a single distance
constraint: `d = pos[j]-pos[i]; pen = FxLength(d) - restLen; n = FxNormalize(d); wi = fxdiv(invMass_i,
invMass_i+invMass_j); pos[i] += FxScale(n, fxmul(pen, wi)); pos[j] -= FxScale(n, fxmul(pen, wj))` — the
verbatim PBD positional projection, the `fpx::SolveContact` generalization) and `StepCloth`
(`IntegrateParticles` then K Gauss-Seidel constraint passes over the CL2 edges in a FIXED order, pinned
particles `invMass=0` never move). Add `shaders/cloth_solve.comp.hlsl` (`[numthreads(1,1,1)]` single-thread,
int64 → Vulkan-only + Metal CPU path), the `cloth_solve` integer golden (the settled/draped lattice,
CPU-colored from the integer read-back → strict ZERO-DIFFERING-PIXEL cross-backend), `--cloth-solve-shot`
(Vulkan) / `--cloth-solve` (Metal), and `tests/cloth_test.cpp` additions. Reuse CL1+CL2 verbatim
(ClothParticle / ClothGrid / IntegrateParticles / BuildConstraints) — CL3 is additive (CL1+CL2 pipelines +
goldens stay byte-identical).

## Design call: INTEGER bit-exact, single-thread serial (the FPX3 discipline, reused)
The PBD solve is Gauss-Seidel: each constraint reads the CURRENT (already-updated by earlier constraints
this pass) particle positions, so it is ORDER-DEPENDENT — exactly like `fpx::SolveContacts`. Therefore the
GPU pass is the proven single-thread `[numthreads(1,1,1)]` mirror (one thread applies all constraints in a
FIXED edge order, K passes), so the GPU is bit-exact to the CPU `StepCloth` reference by construction. The
math uses int64 (`fxmul`, `fxdiv`, `FxISqrt` inside `FxLength`/`FxNormalize`) → `cloth_solve.comp` is
Vulkan-only (DXC compiles int64; glslc cannot) + the Metal `RunClothSolveShowcase` runs the CPU `StepCloth`
(byte-identical by construction — the `fpx_solve.comp` convention, CL1's `cloth_integrate` precedent). The
golden is the CPU-colored integer particle read-back → Vulkan==Metal BIT-IDENTICAL (strict zero-diff).
**HONEST CAVEAT (FPX3-identical): PBD is iterative — after K passes the residual constraint error is
DETERMINISTIC but NOT zero (stiffness ∝ iterations); `fxdiv`/`FxISqrt` truncation means the solver is
bit-REPRODUCIBLE, not analytically exact. Claim DETERMINISM + cross-platform bit-identity (the
differentiator), NOT "more physically correct than Chaos Cloth".**

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The PBD primitive to GENERALIZE (read it carefully):** `engine/sim/fpx.h::SolveContact` (~`fpx.h:347`)
  — the sphere-pair positional projection: penetration along the contact normal, split by inverse-mass
  shares via `fxdiv`, applied to both bodies. CL3's `SolveDistanceConstraint` is the SAME projection over
  a constraint EDGE (rest length instead of sum-of-radii). Also `fpx::SolveContacts`/`StepWorld`
  (`fpx.h:~360/372`) for the K-iteration Gauss-Seidel loop in fixed order. `fxdiv` (`fpx.h:311`, int64),
  `FxNormalize` (`fpx.h:319`, via `FxISqrt` — NO std::sqrt), `FxLength`, `fxmul` (`fpx.h:50`).
- **The single-thread serial GPU mirror:** `shaders/fpx_solve.comp.hlsl` (FPX3's `[numthreads(1,1,1)]`
  Gauss-Seidel, int64, Vulkan-only) — copy its structure: one thread, K passes, fixed constraint order.
  CL1's `cloth_integrate.comp` (the int64-Vulkan-only + Metal-CPU pattern) + `RunFpxSolveShowcase` /
  `RunClothIntegrateShowcase` (the Metal-runs-CPU-reference wiring).
- **The CL1/CL2 inputs:** `engine/sim/cloth.h` — `ClothParticle` (invMass, flags PINNED→invMass 0),
  `IntegrateParticles` (CL1), `Constraint`/`BuildConstraints` (CL2 — the edge list `StepCloth` iterates).
- **The integer-golden showcase discipline:** CL1's `--cloth-integrate-shot` + FPX3's `--fpx-solve-shot`
  (ReadBuffer integer particles, memcmp GPU==CPU, CPU-color, strict zero-diff).
- **RHI compute envelope (must fit, ZERO additions):** `engine/rhi/rhi.h` — the CL1/CL2/FPX set.
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--cloth-solve-shot` standalone arg
  branch + DXC list), `metal_headless/{visual_test.mm,CMakeLists.txt}` (`RunClothSolveShowcase` +
  `--cloth-solve` — NOT in `hf_gen_msl` since int64 Vulkan-only; the Metal showcase runs the CPU
  `StepCloth`), `engine/editor/introspect.cpp` (+`deterministic-cloth-solve` feature + `--cloth-solve-shot`
  showcase) + `tests/introspect_test.cpp` + the JSON golden, `scripts/verify.ps1` (`cloth_solve` golden in
  the Mac loop + `--cloth-solve-shot` in `$vkShots`).

## Design decisions (locked)
1. **The constraint projection (the bit-exact core).** `SolveDistanceConstraint(particles, c)`: if both
   endpoints pinned (`invMass==0`) skip; `d = FxSub(pos[j], pos[i])`; `len = FxLength(d)`; if `len==0`
   skip (degenerate); `pen = len - c.restLen`; `n = FxNormalize(d)` (= `FxScale(d, fxdiv(kOne, len))` or
   the `fpx::FxNormalize`); `wsum = invMass_i + invMass_j`; `wi = fxdiv(invMass_i, wsum)`; `wj = fxdiv(
   invMass_j, wsum)`; `pos[i] = FxAdd(pos[i], FxScale(n, fxmul(pen, wi)))`; `pos[j] = FxSub(pos[j],
   FxScale(n, fxmul(pen, wj)))`. Pinned (invMass 0) → its share is 0 → never moves. All int64-backed Q16.16,
   copied verbatim CPU↔shader. (This is `SolveContact` with rest length — confirm against `fpx.h:347`.)
2. **StepCloth (the K-pass Gauss-Seidel).** `StepCloth(grid, particles, constraints, gravity, dt, groundY,
   iters)`: `IntegrateParticles` (CL1, one step) then `iters` passes, each iterating ALL constraints in
   the FIXED CL2 emit order applying `SolveDistanceConstraint`; after the passes, floor-clamp `pos.y >=
   groundY`. Sequential (each constraint sees prior updates) → single-thread on GPU. Pinned corners hold,
   the sheet drapes.
3. **GPU pipeline (single-thread serial).** `cloth_solve.comp`: `[numthreads(1,1,1)]`, one thread runs K
   steps × (integrate + iters constraint passes) over `gParticles` + `gConstraints` (+ the CL2
   gEdgeOffset if needed). int64 → Vulkan-only (NOT in `hf_gen_msl`); the Metal showcase runs the CPU
   `StepCloth`. Host-snapped integers in → integers out → GPU==CPU bit-exact (the FPX3 argument).
4. **Showcase `--cloth-solve-shot <out>` (Vulkan) AND `--cloth-solve` (Metal — WIRE BOTH; confirm
   visual_test.mm + `#include "sim/cloth.h"`).** The CL1 24×24 sheet with the top corners (or top row)
   pinned, CL2 constraints, `StepCloth` K steps (e.g. 60 steps × 8 iters) under gravity → the sheet DRAPES
   into a hanging cloth (the pinned corners hold, the constraints keep the lattice cohesive — unlike CL1's
   free-fall). ReadBuffer the integer `gParticles`; **memcmp GPU == the CPU `StepCloth` reference (the
   make-or-break)**; CPU-color a side/3-4 view of the draped lattice (each particle a dot, edges optional,
   pinned marked) → `tests/golden/metal/cloth_solve.png` (baked on the Mac by the CONTROLLER — DO NOT
   commit).
5. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance / GPU==CPU (make-or-break):** `gParticles` equals the CPU `StepCloth` reference
     byte-for-byte after K steps. Print `cloth-solve: {particles:<N>, constraints:<E>, pinned:<P>,
     steps:<K>, iters:<I>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs BYTE-IDENTICAL. Print `cloth-solve determinism: two runs BYTE-IDENTICAL`.
   - **(3) cohesion / coherence:** the sheet drapes COHESIVELY (the pinned particles held; the mean
     edge-length error is small + deterministic — the cloth kept its structure, NOT free-fall scatter).
     Print `cloth-solve coverage: pinned <P> held, residual <R> (coherent drape)` (R = a deterministic
     integer residual metric, e.g. summed |edge len - restLen|).
   - **(4) hand-check / no-op:** a single constraint between one free + one pinned particle → the free one
     moves to exactly restLen by the exact inverse-mass share (hand-checked); all-pinned → no movement.
     Print `cloth-solve static: all pinned (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/cloth_solve.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 119 image goldens UNTOUCHED.
6. **Cross-backend bar (INTEGER, strict).** GPU==CPU-by-construction (Metal runs the CPU StepCloth; Vulkan
   the int64 shader == that CPU) → Vulkan==Metal BIT-IDENTICAL: the golden is the CPU-colored integer
   read-back; the controller's cross-backend check is the STRICT ZERO-DIFFERING-PIXEL compare. Any nonzero
   cross-backend pixel diff is a real bug.
7. **Tests `tests/cloth_test.cpp` additions (pure CPU):** a 2-particle 1-constraint case (one pinned, one
   free) → the free particle ends at exactly restLen, the exact inverse-mass split (hand-checked Q16.16);
   a row of particles pinned at one end → drapes under gravity, deterministic; the residual is
   deterministic across two runs; pinned particles never move; `iters=0` == pure integrate (CL1). Clean
   under `windows-msvc-asan`.
8. **Introspect.** Add exactly `deterministic-cloth-solve` (features) + `--cloth-solve-shot` (showcases).
   Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None expected.** Pure SSBO compute (the CL1/CL2/FPX set). `rhi.h` + `rhi_factory` (baseline 2) +
  backend dirs UNCHANGED. CL1+CL2 shaders + `engine/sim/fpx.h` + `engine/physics/` UNCHANGED. Report the
  seam.

## Out of scope (YAGNI — later CL slices)
Collision vs FPX colliders (CL4), lockstep/rollback (CL5), the float render (CL6). Self-collision
(cloth-vs-cloth — a documented future slice). CL3 is ONLY the PBD distance-constraint solver + drape + its
bit-exact golden. No collision, no float. Bending stays a distance constraint (not a dihedral model — the
documented simplification). The PBD residual is deterministic-but-nonzero (the documented FPX3 caveat).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 94) + the new `cloth_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--cloth-solve-shot` on Vulkan: GPU==CPU memcmp BIT-EXACT + determinism + cohesion
   + hand-check/no-op; a coherent DRAPED cloth (the pinned corners held, the sheet cohesive — NOT CL1's
   free-fall scatter). Run under the Vulkan-validation gate → ZERO VUID in the OUTPUT (set BOTH
   `VK_LAYER_PATH` to the conan `...\.conan2\p\...\layers` dir AND
   `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer LOADED = zero "not found").
3. Metal: `visual_test --cloth-solve` → new golden `tests/golden/metal/cloth_solve.png`; two runs DIFF
   0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm cloth_solve is
   correctly EXCLUDED from `hf_gen_msl` (int64 Vulkan-only) and the Metal showcase runs the CPU StepCloth.**
   Cross-backend = STRICT ZERO-DIFFERING-PIXEL (controller-measured), NOT the float baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cloth_solve.png` added;
   the other 119 byte-identical (CL1 cloth_integrate + CL2 cloth_edges + all existing untouched). `git diff
   master --stat -- tests/golden` = ONLY `cloth_solve.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cloth-solve` + `--cloth-solve-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report cloth_solve int64 Vulkan-only + Metal CPU
   path). `scripts/verify.ps1` updated: `cloth_solve` golden in the Mac loop + `--cloth-solve-shot` in
   `$vkShots`. CL1+CL2 + `engine/sim/fpx.h` + `engine/physics/` UNTOUCHED.
