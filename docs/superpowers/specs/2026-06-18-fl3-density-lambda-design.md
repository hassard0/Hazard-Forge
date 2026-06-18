# Slice FL3 — Deterministic GPU Fluid: DENSITY + λ (host-snapped kernel LUT, the make-or-break) (Phase 14 #3) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #9
> (DETERMINISTIC GPU FLUID via Position-Based Fluids, `hf::sim::fluid`, header `engine/sim/fluid.h`). The
> MAKE-OR-BREAK fluid-specific math: per particle, gather the SPH DENSITY ρ_i over the FL2 neighbours via
> a HOST-SNAPPED Q16.16 kernel LUT, and compute the PBF scaling factor λ_i. This is the only genuinely
> fluid-specific slice; the kernel is tabulated (not evaluated symbolically) so it is monotone/non-negative
> by construction and dodges the in-shader-divide hazard. Per-particle INDEPENDENT (a gather over the
> fixed neighbor list) → multi-thread, NO single-thread/TDR exposure. int64 (squared distance + the λ
> divide) → Vulkan-only SPIR-V + Metal runs the byte-identical CPU reference (the FL1/CL3 convention).
> Strict zero-diff cross-backend (the integer bar). ZERO new RHI. Branch: `slice-fl3`. See
> [[hazard-forge-fluid-roadmap]].

**Goal:** Extend `engine/sim/fluid.h` with `BuildKernelTable` (a host-snapped Q16.16 LUT of the PBF poly6
density kernel `W(r,h)` AND the spiky gradient kernel `∇W(r,h)` over `r²` bins — computed HOST-SIDE once at
build, the ONLY `FxISqrt`/sqrt use), `ComputeDensity` (per particle, gather `ρ_i = Σ_j W(r_ij)` over the
FL2 neighbours via the LUT), and `ComputeLambda` (`λ_i = −C_i / (Σ_k |∇_k C_i|² + ε)` with `C_i = ρ_i/ρ0 −
1`, the unilateral density constraint, `fxdiv`). Add `shaders/fluid_density.comp.hlsl` +
`shaders/fluid_lambda.comp.hlsl` (per-particle, int64 → Vulkan-only + Metal CPU path), the `fluid_density`
integer golden (the Q16.16 `ρ_i`/`λ_i` arrays + a density-field heat viz, CPU-colored from the integer
read-back → strict ZERO-DIFFERING-PIXEL cross-backend), `--fluid-density-shot` (Vulkan) / `--fluid-density`
(Metal), and `tests/fluid_test.cpp` additions. Reuse FL1+FL2 verbatim (FluidParticle / BuildNeighborList)
— FL3 is additive (FL1+FL2 pipelines + goldens stay byte-identical).

## Design call: the HOST-SNAPPED KERNEL LUT (the make-or-break move)
Raw SPH would evaluate poly6 `W(r,h)=(315/64πh⁹)(h²−r²)³` and spiky `∇W` symbolically — divide-heavy and
hard to keep monotone/non-negative in Q16.16. PBF + a TABULATED kernel dodges this entirely: precompute
`W` and `|∇W|` as host-snapped Q16.16 values over `B` bins of `r²ʼ ∈ [0, h²]` (the MC case-table / NAV
`TriYSpan` / cloth rest-data precedent — host float math snapped to integers ONCE at build, identical
across backends because `BuildKernelTable` is shared C++ run on both). Then the per-particle density
gather is: for each neighbour `j`, `r² = |p_i − p_j|²` (the squared Q16.16 distance — int64 `fxmul`,
unavoidable: bounded positions still overflow int32 for `dx²`), bin it (`r² ≥ h²` → skip, the deferred
FL2 radial cull lands here naturally), fetch `W[bin]`, accumulate. **The kernel is monotone/non-negative
BY CONSTRUCTION (the table) and there is no in-shader divide in the density**; the ONE divide is `λ_i`'s
`fxdiv` (the same int64 truncating divide the cloth/fpx solver already ships). HONEST: the binned LUT is a
FIDELITY SIMPLIFICATION (deterministic, not the analytic continuous kernel — the FPX3/CL3 caveat shape);
claim DETERMINISM + cross-platform bit-identity, NOT "more physically correct than Niagara". int64 (the
`r²` `fxmul` + the `λ` `fxdiv`) → `fluid_density.comp`/`fluid_lambda.comp` are Vulkan-only + the Metal
showcase runs the CPU reference; per-particle independent → multi-thread, NO single-thread/TDR (that lands
in FL4's Gauss-Seidel solve).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The FL1/FL2 inputs:** `engine/sim/fluid.h` — `FluidParticle`, `BuildNeighborList` (CSR neighbor list:
  `neighborStart[]` + `neighbors[]`), the `FluidGrid`. FL3 adds `BuildKernelTable`/`ComputeDensity`/
  `ComputeLambda` + the `ρ`/`λ` buffers; FL1+FL2 functions UNCHANGED.
- **The Q16.16 + LUT discipline to REUSE (read-only):** `engine/sim/fpx.h` — `fxmul` (int64, for `r²` +
  the kernel accumulate), `fxdiv` (`fpx.h:311`, int64, for `λ`), `FxISqrt` (build-time, in
  `BuildKernelTable` only — NO in-shader sqrt), `FxSub`/`FxVec3`. The host-snapped-table precedent:
  `engine/render/mc.h`'s `kTriTable` case-table + `cloth.h`'s host-snapped rest-lengths (CL2).
- **The int64-Vulkan-only + Metal-CPU convention:** FL1's `fluid_integrate.comp` / CL3's `cloth_solve.comp`
  (Vulkan-only int64 shader, NOT in `hf_gen_msl`; the Metal showcase runs the CPU reference). The
  per-particle multi-thread dispatch (no atomics, no order dependence) is the FL1 pattern, not the CL3
  single-thread.
- **The integer-golden showcase discipline:** FL2's `--fluid-neighbors-shot` (ReadBuffer the integer
  result, memcmp GPU==CPU, CPU-color, strict zero-diff). The density-field heat viz (per-particle ρ as a
  color) is the FL2 neighbor-count-viz twin.
- **RHI compute envelope (must fit, ZERO additions):** `engine/rhi/rhi.h` — the FL1/FL2 set.
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--fluid-density-shot` standalone arg
  branch + DXC list), `metal_headless/{visual_test.mm,CMakeLists.txt}` (`RunFluidDensityShowcase` +
  `--fluid-density` — int64 → Vulkan-only so the Metal showcase runs the CPU density/λ, NO `hf_gen_msl`
  entry for the int64 shaders), `engine/editor/introspect.cpp` (+`deterministic-fluid-density` feature +
  `--fluid-density-shot` showcase) + `tests/introspect_test.cpp` + the JSON golden, `scripts/verify.ps1`
  (`fluid_density` golden in the Mac loop + `--fluid-density-shot` in `$vkShots`).

## Design decisions (locked)
1. **The kernel LUT (host-snapped, build-time, the only sqrt).** `BuildKernelTable(h, ρ0, B)`: over `B`
   bins of `r² ∈ [0, h²]`, precompute `W[bin]` (poly6, host float `(h²−r²)³` × the normalization, snapped
   to Q16.16) and `gradW[bin]` (spiky `|∇W|`, host float, snapped). Pure host float → host-snapped int32,
   computed ONCE, shared C++ → identical both backends. `r ≥ h` (`bin ≥ B`) → `W = 0` (the FL2 box-candidate
   over-inclusion is culled here). Document the bin count + the normalization constants.
2. **Density (per particle, the bit-exact core).** `ComputeDensity(particles, neighborList, table, h, ρ)`:
   per particle `i`, `ρ_i = W[0]` (self) `+ Σ_{j∈neighbors(i)} W[bin(r_ij²)]`, where `r_ij² = |FxSub(p_i,
   p_j)|²` (int64 `fxmul` per axis, summed). Pure integer (int64-backed), copied verbatim CPU↔shader. The
   self-term + the gather are deterministic (fixed neighbor order from FL2).
3. **λ (per particle).** `ComputeLambda(particles, neighborList, table, h, ρ0, ρ, λ)`: `C_i = fxdiv(ρ_i,
   ρ0) − kOne`; `Σgrad² = Σ_k |∇_k C_i|²` (the gradient-of-constraint sum, using `gradW` + the direction
   `FxNormalize(p_i−p_j)` — int64); `λ_i = fxdiv(−C_i, Σgrad² + ε)`. The unilateral clamp (`C_i < 0` →
   `λ_i = 0`, fluid doesn't pull together) is a fixed integer compare. int64 (`fxdiv` + the grad²).
4. **GPU pipeline (per particle, multi-thread).** `fluid_density.comp` (one thread per particle, gather)
   then `fluid_lambda.comp` (one thread per particle). int64 → Vulkan-only (NOT in `hf_gen_msl`); the Metal
   showcase runs the CPU `ComputeDensity`+`ComputeLambda`. Per-particle independent → NO atomics, NO
   single-thread, NO TDR. Host-snapped integers in → integers out → GPU==CPU bit-exact.
5. **Showcase `--fluid-density-shot <out>` (Vulkan) AND `--fluid-density` (Metal — WIRE BOTH; confirm
   visual_test.mm + `#include "sim/fluid.h"`).** The FL1 dam-break block (a settled or mid-fall state) →
   BuildNeighborList (FL2) → BuildKernelTable → ComputeDensity → ComputeLambda. ReadBuffer the integer
   `ρ` + `λ`; **memcmp GPU == the CPU reference (the make-or-break)**; CPU-color a per-particle density
   heat viz (ρ_i → color: dense interior hot, sparse surface cold) → `tests/golden/metal/fluid_density.png`
   (baked on the Mac by the CONTROLLER — DO NOT commit).
6. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance / GPU==CPU (make-or-break):** `ρ` + `λ` equal the CPU reference byte-for-byte. Print
     `fluid-density: {particles:<N>, restDensity:<ρ0>, meanDensity:<ρ̄>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs BYTE-IDENTICAL. Print `fluid-density determinism: two runs
     BYTE-IDENTICAL`.
   - **(3) coherence:** the density is coherent — interior particles denser than surface, `ρ̄ ≈ ρ0` for a
     settled block (within a deterministic band), all `ρ_i > 0`. Print `fluid-density coverage: <D> dense
     particles, mean ρ <ρ̄> (coherent density field)`.
   - **(4) sparse / no-op:** a single isolated particle → `ρ = W[0]` (self only), `λ = 0` (or the clamp).
     Print `fluid-density sparse: self-density only (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/fluid_density.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 125 image goldens UNTOUCHED.
7. **Cross-backend bar (INTEGER, strict).** GPU==CPU-by-construction (Metal runs the CPU density/λ; Vulkan
   the int64 shader == that CPU) → Vulkan==Metal BIT-IDENTICAL: the golden is the CPU-colored integer
   read-back; the controller's cross-backend check is the STRICT ZERO-DIFFERING-PIXEL compare. Any nonzero
   cross-backend pixel diff is a real bug.
8. **Tests `tests/fluid_test.cpp` additions (pure CPU):** `BuildKernelTable` monotone non-increasing W +
   `W[≥B]=0`; an isolated particle → `ρ = W[0]`; two particles at a known `r` → `ρ = W[0]+W[bin(r²)]`
   (hand-checked); the `C_i`/`λ_i` for a known density; the unilateral clamp (`C<0`→`λ=0`); a dense block →
   `ρ̄` in the expected band; determinism. Clean under `windows-msvc-asan`.
9. **Introspect.** Add exactly `deterministic-fluid-density` (features) + `--fluid-density-shot`
   (showcases). Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None expected.** Pure SSBO compute (the FL1/FL2 set; the kernel LUT is a small storage/uniform buffer).
  `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. FL1+FL2 shaders + `engine/sim/fpx.h` +
  `engine/sim/cloth.h` + `engine/physics/` UNCHANGED. Report the seam.

## Out of scope (YAGNI — later FL slices)
The PBF position correction / Gauss-Seidel solve (FL4 — uses ρ/λ; that's where single-thread + the TDR
ceiling land), lockstep (FL5), the float render (FL6). FL3 is ONLY the per-particle density + λ + the
kernel LUT + their bit-exact golden. No position update, no float, no single-thread. The binned LUT is a
documented fidelity simplification.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 95) + the new `fluid_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--fluid-density-shot` on Vulkan: GPU==CPU memcmp BIT-EXACT + determinism +
   coherence + sparse no-op; a coherent density heat image (dense interior). Run under the
   Vulkan-validation gate → ZERO VUID (set BOTH `VK_LAYER_PATH` to the conan `...\.conan2\p\...\layers`
   dir AND `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer LOADED = zero "not found").
3. Metal: `visual_test --fluid-density` → new golden `tests/golden/metal/fluid_density.png`; two runs DIFF
   0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm the int64 density/λ
   shaders are correctly EXCLUDED from `hf_gen_msl` (Vulkan-only) and the Metal showcase runs the CPU
   density/λ.** Cross-backend = STRICT ZERO-DIFFERING-PIXEL, NOT the float baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fluid_density.png` added;
   the other 125 byte-identical (FL1+FL2 + all existing untouched). `git diff master --stat -- tests/golden`
   = ONLY `fluid_density.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fluid-density` + `--fluid-density-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report the int64 Vulkan-only + Metal CPU path; the FL2
   int32 cell/neighbor shaders may be reused unchanged). `scripts/verify.ps1` updated: `fluid_density`
   golden in the Mac loop + `--fluid-density-shot` in `$vkShots`. FL1+FL2 + `engine/sim/fpx.h` +
   `engine/sim/cloth.h` + `engine/physics/` UNTOUCHED.
