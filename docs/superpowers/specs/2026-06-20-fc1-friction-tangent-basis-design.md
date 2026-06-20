# Slice FC1 — Deterministic Contact Friction: THE TANGENT BASIS (the integer beachhead) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice (the BEACHHEAD) of FLAGSHIP #20
> (DETERMINISTIC TANGENTIAL CONTACT FRICTION — the Coulomb friction cone on the convex box-box angular-impulse
> contacts, `hf::sim::fric`). Friction is the most numerically-sensitive part of a contact solver and the corner
> where mainstream float solvers diverge first machine-to-machine; locking it deterministically completes the
> convex arc. FC1 builds the primitive every later friction slice needs: a deterministic, bit-exact **orthonormal
> tangent basis** `(t1, t2)` at a contact, derived from the manifold normal `n`. INTEGER-bit-exact. int64 → the
> `fric_basis.comp` shader is Vulkan-only + a Metal CPU reference (the convex_solve / fpx_solve split). CX1-CX6's
> `convex.h` code is BYTE-FROZEN (fric.h is a NEW additive sibling). Branch: `slice-fc1`. See
> [[hazard-forge-fric-roadmap]].

**Goal:** Create `engine/sim/fric.h` (header-only, namespace `hf::sim::fric`, `#include "sim/convex.h"` read-only
ONLY — which transitively gives fpx) with the tangent-basis primitive: `TangentBasis` + `MakeTangentBasis(n)` (the
fixed integer Gram-Schmidt) + a small measure. Add the new int64 shader `shaders/fric_basis.comp.hlsl` +
`--fric-basis-shot` (Vulkan) / `--fric-basis` (Metal). Bake the integer golden `fric_basis`. **NO new RHI.**

## Design call: a fixed, degenerate-safe integer Gram-Schmidt tangent basis in Q16.16

A friction impulse acts in the contact tangent plane — the plane perpendicular to the contact normal `n`. FC1
produces two orthonormal tangents spanning that plane, deterministically, for any unit `n`:
- **Pick the reference cardinal axis (degeneracy guard).** Of the three cardinal axes `e0=(1,0,0)`, `e1=(0,1,0)`,
  `e2=(0,0,1)` (each `kOne` on one component), choose the one **least aligned** with `n` — the smallest
  `|FxDot(n, e_i)|` — so the projection below never collapses. FIXED tie-break: lowest index `i` wins.
- **t1 = the chosen axis projected off `n`, normalized.** `r = e_min − FxDot(e_min, n)·n` (remove the `n`
  component), then `t1 = FxNormalize(r)` (the int64 `FxISqrt`). Because `e_min` is the least-aligned cardinal axis,
  `r` is comfortably non-zero, so the normalize is well-conditioned.
- **t2 = FxCross(n, t1).** With `n` and `t1` orthonormal (up to fixed-point drift), `n × t1` is already unit, so no
  second normalize is needed (matching `n`'s own near-unit length). `(n, t1, t2)` is a right-handed frame.
- All steps are `FxDot` / `FxNormalize` / `FxCross` (the frozen toolbox: `convex.h:54` `FxDot`, `convex.h:62`
  `FxCross`, `fpx.h:319` `FxNormalize`). FIXED order, integer → bit-identical CPU↔Vulkan↔Metal.

**THE int64 REALITY (the FPX3/CX1 lesson, the honest proof-strength call):** `FxNormalize`/`FxISqrt` + the
`FxDot`/`FxCross` Q16.16 products are int64 (world-scale products overflow int32). DXC compiles int64 (the Vulkan
path); glslc cannot parse `int64_t` in HLSL. So `fric_basis.comp` is **VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl)**;
the Metal `--fric-basis` runs the CPU `MakeTangentBasis` — byte-identical to the Vulkan GPU result BY CONSTRUCTION
(the convex_sat.comp / fpx_solve.comp convention), while the Vulkan side carries the GPU==CPU memcmp proof.

`TangentBasis { FxVec3 t1; FxVec3 t2; }` (the two tangents; `n` is the caller's input, not stored).
`MakeTangentBasis(n)` → `TangentBasis` (the Gram-Schmidt above). `MeasureBasis(normals)` → the deterministic
orthonormality summary (max `|n·t1|`, max `|n·t2|`, max `|t1·t2|`, the min/max `FxLength(t1)`) over a set of
normals — pure integer, the showcase prints + asserts it.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **convex.h (read it; do NOT edit — fric.h is a NEW sibling that `#include`s it read-only):** `FxDot`
  (convex.h:54), `FxCross` (convex.h:62), `FxVec3`/`kOne`/`fxmul`/`fxdiv` (pulled from fpx via convex.h). The
  manifold normal `n` that FC2+ will feed `MakeTangentBasis` comes from `convex.h:296 ContactManifold.normal`
  (sign-corrected A→B in the solver) — but FC1 only needs the basis primitive; it operates on any unit `FxVec3 n`.
- **fpx.h (read-only):** `FxNormalize` (fpx.h:319, int64 `FxISqrt`/`fxdiv`), `FxLength`/`FxISqrt` (fpx.h:79),
  `FxVec3`/`FxSub`/`FxScale` (fpx.h:70), `kOne`/`kFrac`. **CONFIRM the exact signatures. DO NOT modify fpx.h or
  convex.h.**
- **The new-shader showcase precedent:** CX1's `--convex-sat-shot` in `samples/hello_triangle/main.cpp` (the
  per-element compute + the GPU==CPU memcmp + the standalone arg-parse + the `*Gpu` int32 pack) and CX1's
  `shaders/convex_sat.comp.hlsl` (the int64 Vulkan-only new-shader pattern, copying the CPU body VERBATIM). Mirror
  these. Metal: mirror CX1's `--convex-sat` block in `metal_headless/visual_test.mm`.
- **The int64-Vulkan-only wiring:** how `convex_sat.comp` is registered for DXC SPIR-V (in
  `samples/hello_triangle/CMakeLists.txt`) but EXCLUDED from `hf_gen_msl` (`metal_headless/CMakeLists.txt`).
  `fric_basis.comp.hlsl` follows the SAME wiring (confirm `fric` NOT in hf_gen_msl, grep 0).
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), a NEW `tests/fric_test.cpp` (+ CMake wiring, the convex_test pattern).

## Design decisions (locked)
1. **NEW `engine/sim/fric.h`** (header-only, namespace `hf::sim::fric`, `#include "sim/convex.h"` read-only):
   `TangentBasis`, `MakeTangentBasis(n)` (the fixed integer Gram-Schmidt: least-aligned cardinal axis → project off
   `n` → normalize → cross), `MeasureBasis`. Pure integer, FIXED axis-choice + tie-break. **NEW shader**
   `fric_basis.comp.hlsl` (int64, Vulkan-only, one thread per input normal — copies `MakeTangentBasis`'s body
   VERBATIM). NOT in hf_gen_msl; Metal runs the CPU `MakeTangentBasis`.
2. **Showcase `--fric-basis-shot <out>` (Vulkan) AND `--fric-basis` (Metal) — WIRE BOTH** (standalone arg-parse).
   The SCENE: a fixed deterministic array of unit normals (~12-16) spanning the cases — axis-aligned (`±x/±y/±z`),
   the near-cardinal directions (where the least-aligned-axis choice matters), and several oblique tilted normals
   (e.g. the `FxNormalize` of small integer vectors). Vulkan: the GPU `fric_basis.comp` → **memcmp the GPU
   `TangentBasis[]` vs the CPU `MakeTangentBasis`** (NO tolerance — the make-or-break). Metal: the CPU reference.
   Render a PURE-INTEGER 2D diagnostic (each normal drawn as a point with its two tangents as short segments, or a
   small per-normal frame). Golden = `tests/golden/metal/fric_basis.png` (Mac-baked by the CONTROLLER — DO NOT
   commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU `TangentBasis[]` == the CPU `MakeTangentBasis` byte-for-byte. Print
     `fric-basis: {normals:<N>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `fric-basis determinism: two runs BYTE-IDENTICAL`.
   - **(3) orthonormal:** for every normal, `|FxDot(n,t1)|`, `|FxDot(n,t2)|`, `|FxDot(t1,t2)|` are all within a
     small integer epsilon of 0, AND `FxLength(t1)`, `FxLength(t2)` are within an integer epsilon of `kOne` (the
     basis is orthonormal up to fixed-point drift). Print `fric-basis orthonormal: {maxDotErr:<e>, lenErr:<e>}`;
     assert both below the epsilon.
   - **(4) degeneracy-safe:** the axis-aligned normals (`±x/±y/±z`, where a naive fixed-axis projection would
     collapse) still produce a valid orthonormal basis (covered by proof 3 over those entries). Print
     `fric-basis degeneracy: {axisAlignedOk:true}`; assert.
   - **Golden discipline: ONLY `tests/golden/metal/fric_basis.png`; do NOT commit it.** Existing 189 image goldens
     UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests `tests/fric_test.cpp` (NEW, pure CPU):** `MakeTangentBasis` of `+z` gives two unit tangents in the xy
   plane, mutually orthogonal and orthogonal to `z`; an oblique normal gives an orthonormal basis (the three dot
   products ~0, the two lengths ~kOne, within epsilon); the least-aligned-axis choice is the fixed argmin
   (e.g. for `n≈+x` the chosen reference axis is NOT `x`); two runs byte-identical. Clean under `windows-msvc-asan`.
   Wire the new test into CMake.
6. **Introspect.** Add exactly `deterministic-friction-basis` (features) + `--fric-basis-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the convex/fpx surface). `rhi.h` +
  backend dirs UNCHANGED. `engine/sim/convex.h` + `fpx.h` + ALL other sim headers + `engine/nav/` + `engine/anim/`
  + `engine/physics/` + all EXISTING shaders UNCHANGED. The ONLY new shader is `fric_basis.comp.hlsl` (int64,
  Vulkan-only, NOT in hf_gen_msl). `fric.h` is a NEW additive sibling (`#include`s convex.h read-only). Report the
  seam empty (only `fric.h` + the new shader + the showcase/test/introspect are new/changed).

## Out of scope (YAGNI — later FC slices)
The friction-augmented manifold point + accumulators (FC2), the cone-clamped tangent-impulse solver (FC3 — FC1 only
builds the basis, NOT the impulse), the friction-locked world step / ramp grip+slide (FC4), lockstep (FC5), the lit
3D render (FC6 — FC1's render is the 2D basis diagnostic). FC1 claims ONLY: a deterministic integer orthonormal
tangent basis at a contact normal, bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs.
NOTE (honest): int64 → Vulkan-GPU + Metal-CPU-ref (the FPX3/CX1 proof strength, not MSL-native strict-zero — the
`FxNormalize`/`FxCross` products are irreducibly int64).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 105 + the NEW `fric_test`). Clean under `windows-msvc-asan`
   (build+run `fric_test` + `introspect_test`).
2. **proofs + visual:** `--fric-basis-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   a coherent per-normal tangent-frame diagnostic (not scrambled).**
3. Metal: `visual_test --fric-basis` → new golden `tests/golden/metal/fric_basis.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `fric_basis.comp` NOT in `hf_gen_msl`.** Cross-vendor STRICT
   ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fric_basis.png` added; the other
   189 byte-identical. `git diff master --stat -- tests/golden` = ONLY `fric_basis.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-friction-basis` + `--fric-basis-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/convex.h`/`fpx.h`/ALL sim headers + `engine/nav/` +
   `engine/anim/` + `engine/physics/` + ALL existing shaders byte-unchanged). `scripts/verify.ps1` updated:
   `fric_basis` golden in the Mac loop + `--fric-basis-shot` in `$vkShots`. **The ONLY new shader is
   `fric_basis.comp.hlsl` (int64, NOT in `hf_gen_msl`).**
