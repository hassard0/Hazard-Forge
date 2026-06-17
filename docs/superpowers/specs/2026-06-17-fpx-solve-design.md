# Slice FPX3 — Deterministic Fixed-Point Physics: COLLISION RESPONSE (PBD positional solver) (Phase 11 #3) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 3rd FPX slice (after FPX1
> integrator + FPX2 broadphase) and the MAKE-OR-BREAK of the flagship: resolve collisions in FIXED-POINT so overlapping
> bodies push apart + bodies rest on the ground, producing a stable settled PILE — proven GPU==CPU BIT-EXACT, frame-
> deterministic, cross-backend. DESIGN: a POSITION-BASED-DYNAMICS (PBD) POSITIONAL solver (move bodies apart along the
> contact normal — NO velocity impulses, NO restitution tuning, simpler + more robustly deterministic than sequential-
> impulse), run on a SINGLE THREAD (sequential resolution is inherently order-dependent, so single-thread = bit-exact,
> the VT2/mc_scan pattern). NO new RHI. Namespace `hf::sim::fpx`. Branch: `slice-fpx-solve`. See [[hazard-forge-fpx-roadmap]].

**Goal:** Extend `engine/sim/fpx.h` with the fixed-point PBD contact solver (`fxdiv`, `FxNormalize`, `SolveContacts`
= ground + sphere-sphere positional resolution over the FPX2 pairs, K iterations; `StepWorld` = `IntegrateStep` +
`SolveContacts`) + `shaders/fpx_solve.comp.hlsl` (a SINGLE-THREAD serial solver, the math copied VERBATIM) + a
`--fpx-solve-shot` (Vulkan) / `--fpx-solve` (Metal) showcase that drops a cluster, steps integrate+solve K times into a
settled pile, reads back the bodies, proves them BIT-EXACT vs the CPU reference, and bakes the settled-pile golden.
Make-safe: header additions + a NEW shader + NEW showcase + NEW golden; FPX1/FPX2 + the float `engine/physics/`
UNCHANGED.

## The fixed-point PBD solver (the make-or-break math)
PBD positional resolution: for each CONTACT (a penetrating pair, or a body-vs-ground), move the two bodies apart along
the contact normal by their inverse-mass-weighted share of the penetration depth — purely POSITIONAL (velocities
optionally corrected after, but the beachhead is positions only → simplest deterministic core). Over the FPX2
candidate pairs + the ground, iterate K times in a FIXED order.
- **`fx fxdiv(fx a, fx b)`** = `(fx)(((int64_t)a << kFrac) / b)` — a Q16.16 divide via an int64 shift + TRUNCATING
  integer divide (truncation toward zero, IDENTICAL C++/HLSL/MSL). The single most overflow-sensitive op; `b` is a
  non-zero denominator (mass sum / distance) — guard `b==0`. (int64 → see the Metal note below.)
- **`FxVec3 FxNormalize(const FxVec3& v)`** — `len = FxLength(v)` (FxISqrt of the int64 sum-of-squares); if `len==0`
  return a fixed fallback (e.g. `(0,kOne,0)`); else `{ fxdiv(v.x, len), fxdiv(v.y, len), fxdiv(v.z, len) }`. Integer
  normalize — NO `std::sqrt`, NO `<cmath>` (the FxISqrt + fxdiv discipline).
- **Ground contact (NO normalize — axis-aligned):** a body with `pos.y - radius < groundY` penetrates by `pen =
  groundY + radius - pos.y`; resolve `pos.y += pen` (the ground is static/infinite mass → the body takes the full
  correction). Pure integer, no divide, no sqrt — the safe core.
- **Sphere-sphere contact (over the FPX2 pairs):** for pair `(i,j)`, `d = pos_j - pos_i`, `dist = FxLength(d)`, `pen
  = (radius_i + radius_j) - dist`; if `pen > 0`: `n = FxNormalize(d)` (or `d / dist`), the inverse-mass shares
  `wi = fxdiv(invMass_i, invMass_i + invMass_j)`, `wj = kOne - wi`; `pos_i -= n * fxmul(pen, wi)`; `pos_j += n *
  fxmul(pen, wj)`. (Static bodies `invMass==0` take no correction.) int64 in `dist`/`fxdiv`/`fxmul`.
- **`void SolveContacts(FxWorld&, std::span<const FxPair> pairs, int iterations)`** — the CPU reference: K iterations,
  each iteration resolves ALL ground contacts then ALL pair contacts in the FIXED FPX2 pair order (ascending) — a
  deterministic Gauss-Seidel sweep. Sequential (each contact reads the latest positions) → order-dependent → the GPU
  mirror MUST be single-thread.
- **`void StepWorld(FxWorld&, std::span<const FxPair> pairs, fx dt, int solveIters)`** = `IntegrateStep(dt)` +
  `SolveContacts(pairs, solveIters)`. The full deterministic fixed-point physics step.

## Reuse map (file:line)
- **FPX1/FPX2 (the inputs):** `engine/sim/fpx.h` — `fx`/`fxmul`/`FxVec3`/`FxISqrt`/`FxLength`/`FxBody`/`IntegrateStep`
  (FPX1); `FxAabb`/`AabbOverlap`/`FxPair`/`BuildPairs` (FPX2 — the pair list feeding the solver).
- **The single-thread serial GPU pass (copy the shape):** `shaders/mc_scan.comp.hlsl` / `shaders/vt_alloc.comp.hlsl`
  (`[numthreads(1,1,1)]`, `gid.x!=0` guard, a serial loop — the inherently-sequential pattern). The solver is one
  thread doing K Gauss-Seidel iterations over the contacts.
- **The int64 / fixed-point + ISqrt discipline:** `engine/render/mc.h:461-472` (ISqrt — already copied as FxISqrt in
  FPX1), `mc.h:618` (truncating integer-divide pattern → fxdiv), `swraster.h:151` (int64 intermediates).
- **THE int64/glslc Metal LESSON (FPX1):** `fxmul`/`fxdiv`/`FxISqrt` use int64 → DXC compiles, glslc (Metal MSL-gen)
  CANNOT parse int64 → `fpx_solve.comp` is likely **Vulkan-SPIR-V-only** + the Metal `--fpx-solve` showcase runs the
  CPU `StepWorld` (byte-identical by construction — the swraster.comp / fpx_integrate.comp convention). The bake
  reveals it; apply the convention if glslc fails. (FPX2's broadphase was int32 + Metal-native; FPX3's solver is int64
  + Vulkan-only + CPU-Metal — both bit-identical.)
- **Compute + readback surface (NO new RHI):** `BufferUsage::Storage` (`rhi.h:166`), compute (`rhi.h:412-426`),
  `ReadBuffer` (`rhi.h:616`).
- **Golden + registration:** the FPX1 side-view viz (now a settled PILE — richer); `meshlet.h:79` `hashColor`;
  `verify.ps1`/`introspect.cpp`/`introspect_test.cpp`.

## Design decisions (locked)

1. **`fpx_solve.comp.hlsl` (NEW, SINGLE-THREAD `[numthreads(1,1,1)]`).** One thread: read `gBodies` + `gPairs` +
   `gParams{bodyCount, pairCount, gravity, dt, groundY, steps, solveIters, enabled}`; run `steps` of `StepWorld`
   (= `IntegrateStep` then `SolveContacts` K=`solveIters` iterations), the `fxdiv`/`FxNormalize`/ground+sphere-sphere
   resolution copied VERBATIM from fpx.h; write `gBodies` back. `enabled=0` → write input back. Sequential → one thread
   → bit-exact (no race). The pairs are STATIC for the showcase (built once from the initial cluster via FPX2's
   `BuildPairs`, uploaded) OR rebuilt per step on the host — DECISION: build the pair list ONCE on the host from the
   initial config + upload it (the beachhead resolves a FIXED candidate set; per-step rebroadphase is a later refine).
   NO atomics. **int64 (fxdiv/FxISqrt) → if glslc fails MSL-gen, remove from the Metal hf_gen_msl list + run the CPU
   `StepWorld` in the Metal showcase (the FPX1 convention); report which path.**
2. **Showcase `--fpx-solve-shot <out>` (Vulkan, main.cpp) AND `--fpx-solve` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm + `#include "sim/fpx.h"`).** A deterministic CLUSTER of bodies (e.g. a loose stack / grid
   above the ground that falls + collides), radii so they collide, gravity, dt=kOne/60, steps K (enough to settle,
   e.g. 180), solveIters (e.g. 8). Build the FPX2 pair list once from the initial config (host), upload bodies+pairs,
   dispatch `fpx_solve` (Vulkan; CPU `StepWorld` on Metal if int64), `ReadBuffer` `gBodies`. CPU-run `StepWorld` K
   times. Golden = the settled-PILE side-view (`pos >> kFrac` → pixel, `hashColor(body)` dot, groundY line) →
   `tests/golden/metal/fpx_solve.png` — a recognizable stacked/separated pile (richer than FPX1's single row;
   CPU-colored from the integer read-back → cross-backend bit-identical).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact (make-or-break):** `memcmp(gpuBodies, cpuBodies) == 0` after K integrate+solve steps
     (full pos+vel, NO tolerance). Print `fpx-solve GPU==CPU bodies: <N> bodies BIT-EXACT (<K> steps, <I> iters)`.
   - **(2) no-penetration invariant (deterministic, not physical-perfection):** after the K steps, count residual
     overlaps — assert it is DETERMINISTIC (the same on CPU/GPU) and report it; the bodies are above/at the ground
     (`pos.y - radius >= groundY` for all). Print `fpx-solve: {bodies:<N>, contacts:<C>, residual-overlap:<r>,
     above-ground:<N>/<N>}` (residual-overlap is the deterministic count after K iterations — bit-exact, not
     necessarily 0).
   - **(3) hand-checked contact:** two bodies overlapping by a known `pen` → after 1 solve iteration they are pushed
     apart by the exact Q16.16 inverse-mass-weighted amounts (hand-computed). Print `fpx-solve hand-check: pen<pen>
     → split <a>/<b> OK`.
   - **(4) disabled-path no-op:** `enabled=false` → bodies unchanged. Print `fpx-solve disabled: bodies UNCHANGED
     (no-op)`.
   - **(5) determinism:** two full runs → byte-identical bodies. Print `fpx-solve determinism: two runs
     BYTE-IDENTICAL`.
   - **Golden discipline: ONLY `tests/golden/metal/fpx_solve.png`; do NOT commit it — the CONTROLLER bakes on the
     Mac.** Existing 107 image goldens UNTOUCHED.
4. **Determinism / cross-backend.** The solver is fixed-point on host-snapped integers, single-thread serial (fixed
   Gauss-Seidel order) → bit-exact; the int64 fxdiv/FxISqrt are pinned (truncating divide, integer sqrt) identical
   CPU↔GPU; Vulkan runs the GPU shader (DXC int64), Metal runs the CPU `StepWorld` if glslc rejects int64 → byte-
   identical by construction; the golden is CPU-colored from the integer read-back → strict zero-diff cross-backend.
5. **Tests `tests/fpx_test.cpp` additions (pure CPU):** `fxdiv` known quotients incl negatives + the int64 shift;
   `FxNormalize` unit-length (within fp tol) + a known direction; ground resolution (penetrating body → pos.y =
   groundY+radius); sphere-sphere resolution (two overlapping → pushed apart by the inverse-mass shares, the exact
   Q16.16 values; static body `invMass=0` unmoved); `SolveContacts` K-iteration determinism + a known small scene;
   `StepWorld` integrate+solve; `enabled`-off → unchanged; the residual-overlap count is deterministic. Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-fixedpoint-physics-solve` (features) + `--fpx-solve-shot` (showcases).

## RHI seam additions (summary)
- **None.** `BufferUsage::Storage` + compute (single-thread) + `ReadBuffer` — the FPX1/FPX2/VT2 precedent. ZERO
  above-seam backend symbols. `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. `engine/physics/`
  UNTOUCHED. Report the seam.

## Out of scope (YAGNI — FPX4+)
Velocity impulses + restitution/bounce (the beachhead is PBD positional only — a velocity pass is a refine),
per-step re-broadphase (the pair list is built once from the initial config), friction (FPX4+), orientation/box
colliders (FPX4), the lockstep proof (FPX5), the float render (FPX6). Physical-accuracy claims — claim DETERMINISM +
cross-platform BIT-IDENTITY (the differentiator), NOT "more correct than Chaos"; residual overlap after K iterations
is fine as long as it's bit-exact. ONE fixed-point PBD positional solver (ground + sphere-sphere over the FPX2 pairs)
with the GPU==CPU bit-exact proof + the deterministic no-penetration report + the hand-checked contact + disabled
no-op + determinism and the settled-pile golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 92) + the new `fpx_test` solver cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--fpx-solve-shot` on Vulkan: a coherent settled pile (bodies resting on the ground + pushed
   apart, no gross interpenetration); all 5 proof lines. Run under the Vulkan-validation gate → ZERO VUID in the
   OUTPUT.
3. Metal: `visual_test --fpx-solve` → new golden `tests/golden/metal/fpx_solve.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). **The int64 solver: if `fpx_solve.comp` MSL-gens (glslc accepts the int64 — unlikely), Metal
   runs the GPU pass; if glslc FAILS, it is Vulkan-only + the Metal showcase runs the CPU `StepWorld` (the FPX1
   convention) → byte-identical. Confirm which + that visual_test.mm is wired.** Integer golden → a strict cross-
   backend pixel compare must show ZERO differing pixels.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fpx_solve.png` added; the other 107
   byte-identical (FPX1 `fpx.png` + FPX2 `fpx_pairs.png` untouched). `git diff master --stat -- tests/golden` = ONLY
   `fpx_solve.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fixedpoint-physics-solve` + `--fpx-solve-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `scripts/verify.ps1` updated: `fpx_solve` golden in the Mac loop
   + `--fpx-solve-shot` in `$vkShots`.
