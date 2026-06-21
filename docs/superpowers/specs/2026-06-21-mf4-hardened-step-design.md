# Slice MF4 — Hull Narrowphase Hardening: FULL INERTIA + THE RESTACKED-STABILITY STEP (the new-physics money beat) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice (THE
> HEADLINE BEAT) of FLAGSHIP #25 (DETERMINISTIC HULL NARROWPHASE HARDENING, `hf::sim::manifold`). MF1 built the
> face topology; MF2 the multi-point manifold; MF3 lifted it onto the GPU (bit-identical). MF4 is the PAYOFF: it
> wires the hardened manifold + a FULL convex inertia tensor into a stepped world, so a hull dropped FLAT on
> another **SETTLES TO REST** where the frozen single-point step leaves it TEETERING/ROCKING. This is the
> visible, falsifiable "the float engines stack stably only by being non-deterministic — we do it bit-identically
> across Vulkan and Metal" beat: stable resting polyhedra + correct rotational dynamics, deterministic and
> lockstep-ready. MF4 closes BOTH documented gjk limits at once: it swaps `gjk::HullContact` (single-point) →
> `HullContactMulti` (MF3) and `gjk::FxHullInvInertiaBody` (AABB-diagonal) → a FULL signed-tetra inertia, inside
> a sibling step `StepHullWorldHardened` — the frozen `gjk::StepHullWorld` and its 224 goldens are UNTOUCHED
> (additivity by construction). int64 GPU==CPU (the hardened step's GPU mirror), chunked 1 tick/dispatch
> (TDR-safe). APPEND to `engine/sim/manifold.h` (MF1-MF3 + gjk/broad/ccd/convex/fpx/etc BYTE-FROZEN). Branch:
> `slice-mf4`. See [[hazard-forge-manifold-roadmap]], [[hazard-forge-gjk-roadmap]],
> [[hazard-forge-gpu-tdr-chunking]], [[hazard-forge-metal-showcase-gate]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/manifold.h` (additive — MF1-MF3 byte-unchanged) with: `FxHullInertiaBodyFull(hull,
faces, invMass)` → the FULL symmetric body-space INVERSE inertia as a `convex::FxMat3` (signed-tetrahedron
decomposition of the canonical hull, body-LOCAL, int64-accumulated, then `FxMat3SymInverse`); `FxMat3SymInverse(M)`
→ the deterministic integer symmetric-3×3 inverse (adjugate/determinant, fixed order); `WorldInvInertiaFull(body,
invIbodyFull)` = R·invIbodyFull·Rᵀ (generalizing `convex::WorldInvInertia`'s outer-product to a full body tensor);
and `StepHullWorldHardened(world, cfg)` / `StepHullWorldHardenedN` — the `gjk::StepHullWorld` 5-pass shell
(gjk.h:1188) with EXACTLY TWO callee swaps (`HullContact`→`HullContactMulti`, `FxHullInvInertiaBody`+`WorldInvInertia`
→`FxHullInertiaBodyFull`+`WorldInvInertiaFull`). Add `shaders/hull_step_hardened.comp.hlsl` (int64 Vulkan-only — the
GPU mirror of `StepHullWorldHardened`, chunked 1 tick/dispatch). Add the showcase `--mf4-stack-shot <out>` (Vulkan:
GPU-step the stack via the shader, memcmp vs CPU `StepHullWorldHardenedN`; + the frozen-step teeter control) /
`--mf4-stack` (Metal: CPU `StepHullWorldHardenedN`). Bake the integer-render golden `mf4_stack`. **NO new RHI**;
ONE new compute shader (int64 Vulkan-only).

## Design call: the hardened step = the frozen shell + two swaps; the headline is "rest, not rock"

`gjk::StepHullWorld` (gjk.h:1188) is the 5-pass shell: (1) integrate+damp; (2) world inverse inertias once/tick
(`FxHullInvInertiaBody`→`WorldInvInertia`, gjk.h:1205-1206); (3) Gauss-Seidel impulse over all-pairs
(`HullContact` per pair, gjk.h:1216); (4) position de-pen (`HullContact`, uses `m.depths[0]`+`m.normal`,
gjk.h:1236). `StepHullWorldHardened` is the SAME shell with two swaps and NOTHING else:
- **inertia (step 2):** `FxHullInertiaBodyFull(world.hulls[i], faces[i], invMass)` → `WorldInvInertiaFull` (a full
  `FxMat3`), replacing the diagonal `FxHullInvInertiaBody`→`WorldInvInertia`. The frozen `convex::SolveManifoldImpulse`
  (convex.h:651) ALREADY takes `FxMat3` `invIaW`/`invIbW` → it consumes the full tensor with ZERO change.
- **contact (steps 3+4):** `HullContactMulti` (MF3) replacing `HullContact`. The de-pen reads `m.depths[0]` — MF2
  guarantees `points[0]`/`depths[0]` is the DEEPEST point, so the de-pen is correct; the impulse solve loops
  `0..count` over the 4 face points → the stabilizing multi-point contact (the teeter fix in motion).
- **FULL inertia (the new physics):** `FxHullInertiaBodyFull` computes the canonical hull's full symmetric inertia
  tensor I by signed-tetrahedron decomposition (a tetra fan from the hull centroid over each face triangle,
  accumulating the standard covariance integrals), in BODY-LOCAL space; scale by mass (`1/invMass`); then
  `FxMat3SymInverse(I)` → the body-space INVERSE inertia `FxMat3`. Static (invMass==0) → the zero matrix (takes no
  angular impulse). `WorldInvInertiaFull` rotates it to world (R·M·Rᵀ, R = `convex::BoxAxes`).

**THE HEADLINE:** a box dropped FLAT onto a static box, stepped by `StepHullWorldHardenedN`, settles to a stable
rest (`maxSpeed` and `maxAngVel` → a small fixed band) — while the SAME scene under the frozen
`gjk::StepHullWorldN` (the single-point manifold) leaves a residual teeter (the contact can't resist the tipping
moment). The showcase asserts both: hardened `settled:true`, frozen-control `teeters:true`. **The CUBE CROSS-CHECK
(the inertia validation):** `FxMat3SymInverse(FxHullInertiaBodyFull(MakeBox))` ≈ `convex::WorldInvInertia`'s
diagonal from `gjk::FxHullInvInertiaBody(MakeBox)` — the full tensor REDUCES to the known AABB-diagonal answer for
a cube (proving the new path is a strict, correct generalization, not a different model).

> THE CRUX (the secondary determinism risk): the signed-tetra covariance integrals accumulate products of vertex
> coordinates → potential int32/int64 overflow at world scale. MITIGATION: compute in BODY-LOCAL space (verts are
> body-relative, bounded by the hull's local extents — a few `kOne`), accumulate in explicit `int64`, defer the
> Q16.16 normalization to the end, and ASSERT the cube cross-check (which catches any scaling/overflow error
> against the known-correct `FxBoxInvInertiaBody`). `FxMat3SymInverse` divides by the determinant (int64 fxdiv,
> fixed order) — a degenerate/zero determinant falls back to the diagonal `WorldInvInertia` (deterministic floor).
> TDR: the hardened step is chunked **1 tick / dispatch** (the documented Windows ~2s watchdog rule — heavier than
> the single-point step, so 1-tick chunking makes TDR impossible by construction; verify ~3 runs GPU==CPU, NOT 30x).

## Reuse map (file:line)
- **MF1-MF3 `engine/sim/manifold.h` (APPEND after `HullContactMulti`):** `HullContactMulti`, `BuildCanonicalFaces`,
  `FaceNormalWorld`/`FaceCentroidWorld`, `FxHullFaces`. MF1-MF3 byte-frozen.
- **convex.h (read-only — REUSE):** `convex::FxMat3` (the 9-element matrix), `convex::WorldInvInertia`
  (convex.h:622 — the R·diag·Rᵀ outer-product idiom `WorldInvInertiaFull` generalizes), `convex::FxBoxInvInertiaBody`
  (convex.h:606 — the cube cross-check reference), `convex::BoxAxes`, `convex::SolveManifoldImpulse` (convex.h:651
  — the CONSUMER, takes `FxMat3`, UNCHANGED), `convex::ConvexStepConfig`, `convex::IsDynamic`, `convex::FxScale`.
- **gjk.h (read-only — the shell to MIRROR, do NOT edit):** `gjk::StepHullWorld`/`StepHullWorldN` (gjk.h:1188 — the
  5-pass shell), `gjk::FxHullInvInertiaBody` (gjk.h:1127 — the diagonal version being replaced + the cube-check
  reference), `gjk::HullWorld`/`FxHull`/canonical builders, `gjk::HullToRenderInstances`, `gjk::MeasureHullStack`.
- **shaders (read-only precedents):** `shaders/hull_step.comp.hlsl` (THE template — the int64 Vulkan-only GPU
  mirror of `gjk::StepHullWorld`, chunked 1 tick/dispatch; `hull_step_hardened.comp.hlsl` is it with the two
  swaps), `shaders/hull_manifold.comp.hlsl` (MF3 — the multi-point manifold HLSL to reuse for the contact swap).
- **The GPU==CPU memcmp + teeter-control showcase precedent:** GJ4 hull-step / CD4 `--ccd-bullet` (the chunked GPU
  step + GPU==CPU memcmp + the discrete/frozen CONTROL that fails the new test). Mirror for `--mf4-stack`.
- **Registration:** `samples/hello_triangle/CMakeLists.txt` (`shaders/hull_step_hardened.comp.hlsl:cs` — Vulkan),
  `scripts/verify.ps1` (`mf4_stack` + `--mf4-stack-shot` to `$vkShots`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), append to `tests/manifold_test.cpp`. **NO
  `hf_gen_msl` entry** (int64 → Metal CPU-ref).

## Design decisions (locked)
1. **APPEND to `engine/sim/manifold.h`** (MF1-MF3 byte-frozen): `FxHullInertiaBodyFull`, `FxMat3SymInverse`,
   `WorldInvInertiaFull`, `StepHullWorldHardened`, `StepHullWorldHardenedN`. **ONE new shader
   `shaders/hull_step_hardened.comp.hlsl`** (int64, Vulkan-only — DXC SPIR-V, NOT MSL). **NO new RHI** (reuse the
   compute-dispatch + the instanced-lit render). gjk.h/convex.h/ALL other sim headers + ALL OTHER shaders
   BYTE-UNCHANGED.
2. **Showcase `--mf4-stack-shot <out>` (Vulkan) AND `--mf4-stack` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--mf4-stack` BEFORE reporting DONE).** BOTH build a deterministic FLAT-rest stack (a box
   resting flat on a static box, optionally a small mixed stack), step it with `StepHullWorldHardenedN` (Vulkan:
   the GPU via the reused chunked 1-tick/dispatch shader, memcmp vs CPU; Metal: the CPU) until settled, AND step
   the IDENTICAL scene with the frozen `gjk::StepHullWorldN` (the teeter control). BOTH render the SETTLED hardened
   stack LIT 3D (the resting polyhedra). Golden = `tests/golden/metal/mf4_stack.png` (Mac-baked by the CONTROLLER
   — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1) GPU==CPU:** `mf4-stack: {bodies:<N>, ticks:<K>} GPU==CPU BIT-EXACT` — the GPU hardened-step final world
     `memcmp`-equals the CPU `StepHullWorldHardenedN`, byte-for-byte; assert.
   - **(2) THE HEADLINE — rest vs rock:** `mf4-stack: {hardenedSettled:true, frozenTeeters:true}` — under
     `StepHullWorldHardenedN` the dropped box settles (`maxSpeed`/`maxAngVel` below a fixed band); the SAME scene
     under the frozen `gjk::StepHullWorldN` does NOT (residual angular motion above the band). Assert BOTH. **The
     stable stack the single-point manifold can't hold.**
   - **(3) the cube cross-check:** `mf4-stack inertia: cube full == diagonal (maxErr:<v> <= tol)` —
     `FxHullInertiaBodyFull(MakeBox)` inverted matches `gjk::FxHullInvInertiaBody(MakeBox)`'s diagonal within a
     small fixed tolerance (the full tensor reduces to the AABB answer for a cube). Assert.
   - **(4) determinism:** `mf4-stack determinism: two runs BYTE-IDENTICAL` (~3 runs — efficient, NOT 30x).
   - **Golden discipline: ONLY `tests/golden/metal/mf4_stack.png`; do NOT commit it.** Existing 224 goldens
     UNTOUCHED.
4. **Cross-backend bar (int64 → strict on the integer proof).** Vulkan GPU==CPU bit-exact (~3× clean — chunked
   1-tick/dispatch, TDR-safe). Metal CPU-ref byte-identical. The integer settled world is strict-zero cross-vendor;
   the golden IMAGE is the float render (in-band visresolve ~20-55).
5. **Tests — APPEND to `tests/manifold_test.cpp` (CPU):** the cube cross-check (`FxHullInertiaBodyFull(MakeBox)`
   inv == `gjk::FxHullInvInertiaBody(MakeBox)` within tol); `FxMat3SymInverse` round-trips a known symmetric matrix
   (M·M⁻¹ ≈ I within tol); the flat box settles under `StepHullWorldHardenedN` (final `maxAngVel` below band) while
   the frozen step does not; a tumbling tetra's trajectory is two-run byte-equal; the hardened step is determinstic
   (two runs byte-equal). Clean under `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `manifold-hardened-step` (features) + `--mf4-stack-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **ONE new compute shader, NO new RHI.** `shaders/hull_step_hardened.comp.hlsl` (int64, Vulkan-only — DXC SPIR-V;
  NOT in `hf_gen_msl`; the `hull_step.comp.hlsl`/`hull_manifold.comp.hlsl` precedent). It reuses the EXISTING
  compute-dispatch + SSBO seam (chunked 1 tick/dispatch). The render reuses the instanced-lit pipeline.
  `engine/sim/manifold.h` APPEND-only (MF1-MF3 frozen); gjk.h/convex.h + ALL other sim headers + ALL OTHER shaders
  UNCHANGED. Report the seam: manifold.h APPEND-only + ONE new Vulkan-only shader; NO rhi.h change, NO MSL entry,
  NO frozen-file edit.

## Out of scope (YAGNI — later slices)
Lockstep (MF5 — MF4 proves the deterministic hardened STEP; MF5 wraps it in the command/snapshot harness). The
render capstone polish (MF6). The area-maximizing 4-point reduction (inherited MF2 deferral). A general quickhull
inertia (canonical hulls only — the tetra decomposition over `BuildCanonicalFaces`). Rolling/spinning friction at
the multi-point contact (the impulse solve is frictionless/normal-only, the `SolveManifoldImpulse` model). MF4
claims ONLY: a deterministic, bit-exact (CPU↔Vulkan↔Metal) hardened step where the multi-point manifold + full
inertia make a flat-dropped hull SETTLE TO REST the frozen single-point step leaves teetering, with the cube
cross-check + the integer golden + the four proofs. CAVEAT: "settled" = within-band residual (the Gauss-Seidel +
linear de-pen leave a deterministic nonzero residual — the CX/JT lineage), NOT analytic zero.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "manifold|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--mf4-stack-shot` on Vulkan: the 4 proof lines (GPU==CPU + the rest-vs-rock headline +
   the cube cross-check) + exit 0 under the conan validation layer → ZERO VUID. **~3 runs all GPU==CPU (chunked
   1-tick/dispatch — TDR-safe). VERIFY the image shows the box RESTING FLAT (settled, not tipped/teetering)** on
   the support box, no garbage/NaN/iridescence.
3. Metal: `visual_test --mf4-stack` → `tests/golden/metal/mf4_stack.png`; two runs DIFF 0.0000. **Confirm
   `--mf4-stack` is wired in `visual_test.mm` (grep it) BEFORE the Mac bake; confirm NO `hf_gen_msl` entry.**
   Cross-vendor = FLOAT visresolve in-band on the render; STRICT ZERO on the integer settled world.
4. **Render-invariance:** ONLY `mf4_stack.png` added; the other 224 byte-identical (+ controller introspect
   rebake).
5. Introspect: exactly `+manifold-hardened-step` + `--mf4-stack-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + MF1-MF3 manifold.h code + gjk.h/convex.h + ALL other sim headers + ALL OTHER shaders
   byte-unchanged; manifold.h APPEND-only; exactly ONE new shader `hull_step_hardened.comp.hlsl`, Vulkan-only, NO
   `hf_gen_msl`). `mf4_stack` in the Mac loop + `--mf4-stack-shot` in `$vkShots`.
