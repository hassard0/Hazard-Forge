# Slice FL4 — Deterministic GPU Fluid: PBF DENSITY-CONSTRAINT SOLVE (the incompressibility solver) (Phase 14 #4) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #9
> (DETERMINISTIC GPU FLUID via Position-Based Fluids, `hf::sim::fluid`, header `engine/sim/fluid.h`). The
> incompressibility solver: project the FL3 density constraints so the fluid holds its rest density (a
> dam-break settles into an incompressible pool) — the density-constraint twin of the cloth PBD solver.
> **DESIGN REFINEMENT vs the scout: standard Macklin–Müller PBF is JACOBI (all density/λ/Δp from an
> iteration-start snapshot, applied together) → per-particle INDEPENDENT → MULTI-THREAD, bit-exact by
> construction, and it REMOVES the single-thread/TDR ceiling the scout assumed** (the cloth's
> Gauss-Seidel was the order-dependent one; PBF density is solved Jacobi). int64 (the density/λ/Δp math)
> → Vulkan-only SPIR-V + Metal runs the byte-identical CPU reference (the FL3/CL3 convention). Strict
> zero-diff cross-backend (the integer bar). ZERO new RHI. Branch: `slice-fl4`. See
> [[hazard-forge-fluid-roadmap]].

**Goal:** Extend `engine/sim/fluid.h` with `SolveDensityConstraint` (the PBF position correction `Δp_i =
(1/ρ0) Σ_j (λ_i + λ_j) ∇W(p_i − p_j)` over the FL2 neighbours — the cloth `SolveDistanceConstraint`
generalized from a distance edge to a density constraint) and `StepFluid` (predict via `IntegrateFluid` →
`BuildNeighborList` → K JACOBI iterations of {`ComputeDensity` → `ComputeLambda` → `SolveDensityConstraint`
→ apply Δp} → derive velocity from the position change → `CollidePlane`/`CollideSpheres` reusing the CL4
colliders). Add `shaders/fluid_dp.comp.hlsl` (the per-particle Δp pass) + the StepFluid driver (reuse the
FL3 `fluid_density`/`fluid_lambda` passes per iteration + a `fluid_apply`/`fluid_collide` pass), int64 →
Vulkan-only + Metal CPU. The `fluid_solve` integer golden (the settled incompressible pool + a
deterministic integer incompressibility residual, CPU-colored from the integer read-back → strict
ZERO-DIFFERING-PIXEL cross-backend), `--fluid-solve-shot` (Vulkan) / `--fluid-solve` (Metal), and
`tests/fluid_test.cpp` additions. Reuse FL1–FL3 verbatim (IntegrateFluid / BuildNeighborList /
ComputeDensity / ComputeLambda) + the CL4 colliders (read-only) — FL4 is additive (FL1–FL3 pipelines +
goldens stay byte-identical).

## Design call: JACOBI PBF, MULTI-THREAD, bit-exact, NO TDR ceiling
PBF (Macklin & Müller 2013) is JACOBI within each solver iteration: (a) compute `ρ_i` for ALL particles
(from the iteration-start positions), (b) compute `λ_i` for ALL, (c) compute `Δp_i` for ALL (from the
iteration-start positions + the just-computed λ), (d) apply `p_i += Δp_i` for ALL. Each sub-pass is
PER-PARTICLE INDEPENDENT — every particle reads its neighbours' iteration-start state (read-only) and
writes only its own `Δp_i`/`p_i`, so there is NO race and the result is DETERMINISTIC regardless of thread
order. Therefore the GPU runs MULTI-THREAD (one thread per particle per sub-pass, a barrier between
sub-passes/iterations) and is bit-exact to the sequential CPU `StepFluid` by construction — and there is
**NO single-thread `[numthreads(1,1,1)]` dispatch and NO TDR particle-count ceiling** (the cloth/CL3
limit does NOT apply to PBF; this is the key win of the density-constraint formulation over the cloth's
order-dependent Gauss-Seidel). The math is int64 (`fxmul` for `r²`/`∇W`, `fxdiv` for `λ`), so the passes
are Vulkan-only + the Metal showcase runs the CPU `StepFluid` (byte-identical by construction). **HONEST
CAVEAT (FL3/CL3-identical): K Jacobi iterations leave a deterministic-but-nonzero incompressibility
residual (stiffness ∝ iterations); the binned kernel LUT + `fxdiv` truncation make it bit-REPRODUCIBLE,
not analytically incompressible. Claim DETERMINISM + cross-platform bit-identity, NOT "more physically
correct than Niagara".**

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The FL1–FL3 inputs to drive:** `engine/sim/fluid.h` — `IntegrateFluid` (FL1), `BuildNeighborList`
  (FL2), `BuildKernelTable`/`ComputeDensity`/`ComputeLambda` (FL3). `StepFluid` chains these per the PBF
  loop. The CL4 colliders to REUSE (from `cloth.h`, read-only — `#include "sim/cloth.h"` OR factor the
  shared `SphereCollider`/`CollidePlane`/`CollideSpheres` math): the cloth and fluid are the SAME Q16.16
  world units, so the fluid pours over the same `fpx::FxBody` sphere.
- **The PBF correction = the cloth projection generalized:** `engine/sim/cloth.h::SolveDistanceConstraint`
  (the inverse-mass-weighted positional projection along a normal) — `SolveDensityConstraint` is the same
  shape but the "constraint" is the density (the correction is `Σ_j (λ_i+λ_j) ∇W` instead of a single
  edge's `pen·n`). `fpx::FxNormalize`/`fxmul`/`fxdiv` (int64).
- **The multi-thread per-particle GPU pattern:** FL3's `fluid_density.comp`/`fluid_lambda.comp` (one
  thread per particle, int64, Vulkan-only, NOT in `hf_gen_msl`, Metal runs CPU) — `fluid_dp.comp` +
  `fluid_apply`/`fluid_collide` are the SAME pattern. The K-iteration driver issues
  density→λ→dp→apply per iteration with `ComputeToComputeBarrier` between. NOT the CL3 single-thread.
- **The integer-golden showcase discipline:** FL3's `--fluid-density-shot` + CL3's `--cloth-solve-shot`
  (ReadBuffer integer particles, memcmp GPU==CPU, CPU-color, strict zero-diff) + the `EdgeResidual` twin
  (a deterministic integer residual metric).
- **RHI compute envelope (must fit, ZERO additions):** `engine/rhi/rhi.h` — the FL1–FL3 set.
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--fluid-solve-shot` standalone arg
  branch + DXC list), `metal_headless/{visual_test.mm,CMakeLists.txt}` (`RunFluidSolveShowcase` +
  `--fluid-solve` — int64 → Vulkan-only so Metal runs the CPU StepFluid, NO `hf_gen_msl` for the int64
  passes), `engine/editor/introspect.cpp` (+`deterministic-fluid-solve` feature + `--fluid-solve-shot`
  showcase) + `tests/introspect_test.cpp` + the JSON golden, `scripts/verify.ps1` (`fluid_solve` golden
  in the Mac loop + `--fluid-solve-shot` in `$vkShots`).

## Design decisions (locked)
1. **The position correction (the bit-exact core).** `SolveDensityConstraint(particles, neighborList,
   table, λ, ρ0, h, dp)`: per particle `i`, `Δp_i = FxScale( Σ_{j∈neighbors(i)} fxmul(λ_i + λ_j, gradW
   contribution along (p_i − p_j)), 1/ρ0 )` — the spiky-gradient-weighted, λ-scaled sum over neighbours,
   plus the standard PBF tensile-instability term `s_corr` OPTIONAL (document; a fixed small negative
   pressure to prevent clustering — keep it integer or omit for FL4). Writes `dp[i]` (a separate buffer —
   Jacobi double-buffering). Pinned/boundary particles (invMass 0) → `dp = 0`. All int64-backed Q16.16,
   copied verbatim CPU↔shader.
2. **StepFluid (the JACOBI PBF loop).** `StepFluid(particles, grid, kernel, colliders, gravity, dt,
   groundY, iters)`: `IntegrateFluid` (predict) → `BuildNeighborList` (from predicted positions) → for
   `iters`: `ComputeDensity` (all) → `ComputeLambda` (all) → `SolveDensityConstraint` → `p_i += dp_i`
   (all) [each a full per-particle pass over the iteration-start snapshot] → after the iterations, `vel =
   (pos − prev)/dt` (derive velocity from the net position change) → `CollidePlane` + `CollideSpheres`
   (project out of the ground + the FxBody spheres). Deterministic (Jacobi, fixed neighbor order).
3. **GPU pipeline (multi-thread, per-particle, NO single-thread).** Each PBF sub-pass is one thread per
   particle (density, λ, dp, apply, collide), with `ComputeToComputeBarrier` between sub-passes and
   iterations. int64 → Vulkan-only (NOT in `hf_gen_msl`); the Metal showcase runs the CPU `StepFluid`.
   Host-snapped integers in → integers out → GPU==CPU bit-exact. **NO `[numthreads(1,1,1)]`, NO TDR
   ceiling** (Jacobi parallelizes — the FL4 design win).
4. **Showcase `--fluid-solve-shot <out>` (Vulkan) AND `--fluid-solve` (Metal — WIRE BOTH; confirm
   visual_test.mm + `#include "sim/fluid.h"`).** A dam-break: the FL1 block dropped into a box / onto the
   ground (+ optionally an FxBody sphere it pours over), `StepFluid` K steps × `iters` density iterations
   under gravity → the fluid SETTLES into an incompressible pool (the constraints hold ρ≈ρ0 — unlike FL1's
   free-fall pile). ReadBuffer the integer `gParticles`; **memcmp GPU == the CPU `StepFluid` reference
   (the make-or-break)**; CPU-color a side/3-4 view of the settled fluid (each particle colored by density
   or height) → `tests/golden/metal/fluid_solve.png` (baked on the Mac by the CONTROLLER — DO NOT commit).
5. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance / GPU==CPU (make-or-break):** `gParticles` equals the CPU `StepFluid` reference
     byte-for-byte after K steps. Print `fluid-solve: {particles:<N>, steps:<K>, iters:<I>, residual:<R>}
     GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs BYTE-IDENTICAL. Print `fluid-solve determinism: two runs
     BYTE-IDENTICAL`.
   - **(3) incompressibility / coherence:** the fluid settles cohesively — the mean density-error residual
     `R` (summed `|ρ_i − ρ0|`) is SMALL + deterministic (the constraints reduced compression vs the
     unsolved FL1 free-fall); the pool covers a coherent region. Print `fluid-solve coverage: settled,
     residual <R> (incompressible pool, density held)`.
   - **(4) no-solve / no-op:** `iters=0` → pure FL1 integrate (byte-identical to the free-fall). Print
     `fluid-solve no-solve: == integrate (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/fluid_solve.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 126 image goldens UNTOUCHED.
6. **Cross-backend bar (INTEGER, strict).** GPU==CPU-by-construction (Metal runs the CPU StepFluid; Vulkan
   the int64 multi-thread passes == that CPU, no race) → Vulkan==Metal BIT-IDENTICAL: the golden is the
   CPU-colored integer read-back; the controller's cross-backend check is the STRICT ZERO-DIFFERING-PIXEL
   compare. Any nonzero cross-backend pixel diff is a real bug.
7. **Tests `tests/fluid_test.cpp` additions (pure CPU):** a compressed pair (closer than rest) → the solve
   pushes them apart (density error reduced); a settled block → `R` decreases monotonically with `iters`
   (deterministic); `iters=0` == pure integrate; particles never tunnel through the ground / sphere after
   the solve+collide; determinism (two StepFluid runs identical). Clean under `windows-msvc-asan`.
8. **Introspect.** Add exactly `deterministic-fluid-solve` (features) + `--fluid-solve-shot` (showcases).
   Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None expected.** Pure SSBO compute (the FL1–FL3/CL4 set; the dp buffer is double-buffering for Jacobi).
  `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. FL1–FL3 shaders + `engine/sim/fpx.h` +
  `engine/sim/cloth.h` (read-only for the colliders) + `engine/physics/` UNCHANGED. Report the seam.

## Out of scope (YAGNI — later FL slices)
Lockstep/rollback (FL5), the float render (FL6). Vorticity confinement + XSPH viscosity (the PBF "nice
look" extras — a future refinement; FL4 is the incompressibility solve). The tensile `s_corr` is optional.
Surface tension, multiphase, two-way rigid coupling (the colliders are static `FxBody` reads, like CL4).
FL4 is ONLY the Jacobi PBF density-constraint solve + settle + its bit-exact golden. No float. The
residual is deterministic-but-nonzero (the documented caveat).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 95) + the new `fluid_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--fluid-solve-shot` on Vulkan: GPU==CPU memcmp BIT-EXACT + determinism +
   incompressibility (residual small + deterministic) + no-solve no-op; a coherent SETTLED fluid pool (the
   density held — NOT FL1's free-fall scatter). Run under the Vulkan-validation gate → ZERO VUID (set BOTH
   `VK_LAYER_PATH` to the conan `...\.conan2\p\...\layers` dir AND
   `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer LOADED = zero "not found"). (Jacobi
   multi-thread → no TDR; but if the per-step cost is high, keep steps×iters reasonable.)
3. Metal: `visual_test --fluid-solve` → new golden `tests/golden/metal/fluid_solve.png`; two runs DIFF
   0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm the int64 solve
   passes are correctly EXCLUDED from `hf_gen_msl` (Vulkan-only) and the Metal showcase runs the CPU
   StepFluid.** Cross-backend = STRICT ZERO-DIFFERING-PIXEL, NOT the float baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fluid_solve.png` added;
   the other 126 byte-identical (FL1–FL3 + all existing untouched). `git diff master --stat --
   tests/golden` = ONLY `fluid_solve.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fluid-solve` + `--fluid-solve-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report the int64 Vulkan-only + Metal CPU path; the
   FL2 int32 cell/neighbor shaders reused unchanged). `scripts/verify.ps1` updated: `fluid_solve` golden
   in the Mac loop + `--fluid-solve-shot` in `$vkShots`. FL1–FL3 + `engine/sim/fpx.h` + `engine/sim/cloth.h`
   + `engine/physics/` UNTOUCHED.
