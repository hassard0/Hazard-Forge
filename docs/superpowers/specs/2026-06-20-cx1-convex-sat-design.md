# Slice CX1 — Deterministic Convex Contacts: THE BOX-BOX SAT OVERLAP TEST — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice (the BEACHHEAD) of FLAGSHIP #19
> (DETERMINISTIC CONVEX RIGID-BODY CONTACTS — SAT manifold + inertia tensor + angular impulse, `hf::sim::convex`).
> The whole sim suite has the SAME caveat in SEVEN headers: contacts are sphere-sphere with NO inertia tensor / no
> torque-from-contact. This flagship lifts that ceiling. CX1 is the narrowphase BEACHHEAD: a deterministic integer
> **box-box Separating-Axis Test (SAT)** that, for a pair of ORIENTED boxes, computes the minimum-penetration axis
> + depth (or reports separation) bit-exact. Adds the new fixed-point primitives `FxMat3` (the 3×3, for CX3's
> inertia tensor) + `FxCross` (promoted) + an `FxBox` collider. INTEGER-bit-exact. int64 → the `convex_sat.comp`
> shader is Vulkan-only + a Metal CPU reference (the fpx_solve.comp split). Branch: `slice-cx1`. See
> [[hazard-forge-convex-roadmap]].

**Goal:** Create `engine/sim/convex.h` (header-only, namespace `hf::sim::convex`, `#include "sim/fpx.h"` read-only
ONLY) with `FxMat3` + `FxCross` + `FxBox` + `SatResult` + `BoxSat` (the 15-axis box-box SAT) + a small SAT-pair
measure. Add the new int64 shader `shaders/convex_sat.comp.hlsl` + `--convex-sat-shot` (Vulkan) / `--convex-sat`
(Metal). Bake the integer golden `convex_sat`. **NO new RHI.**

## Design call: the 15-axis box-box SAT in Q16.16, min-penetration axis + depth
Two oriented boxes overlap iff they overlap on ALL 15 candidate separating axes; they are SEPARATED iff ANY axis
separates them. For boxes with `FxBody{pos, orient}` + `FxBox{halfExtents}`:
- **The 15 axes:** the 3 face normals of A (A's local X/Y/Z rotated to world by `fpx::FxRotate(a.orient, e_i)`), the
  3 face normals of B, and the 9 edge-edge crosses `FxCross(faceAxisA_i, faceAxisB_j)` (skip a degenerate
  near-zero cross — `FxLength < eps` → that axis can't separate, skip it deterministically). All in Q16.16.
- **The projection / overlap test per axis `L`:** the projected half-width (radius) of box A onto `L` is
  `rA = |FxDot(L, axA_x)|·hA.x + |FxDot(L, axA_y)|·hA.y + |FxDot(L, axA_z)|·hA.z` (axA_i = A's world face axes,
  hA = A's half-extents) and similarly `rB`; the center separation is `s = |FxDot(L, FxSub(b.pos, a.pos))|`; the
  axis SEPARATES iff `s > rA + rB`; otherwise the penetration on `L` is `pen = (rA + rB) − s`. (Normalize `L` so
  the penetrations are comparable across axes: `L = FxNormalize(L)` before the test — the int64 `FxISqrt`; the
  edge-edge crosses are NOT unit. Face axes are already unit from the unit orient.)
- **The result:** if ANY axis separates → `{overlap=false}`. Else → `{overlap=true, axisIndex, penetration =
  min pen over all axes, axis = the min-pen L (signed toward B: flip if `FxDot(L, b.pos−a.pos) < 0`)}`. The min is a
  deterministic strict-`<` scan (ties keep the LOWEST axis index — bit-reproducible).

**THE int64 REALITY (the honest proof-strength call, the FPX3 lesson):** `FxNormalize`/`FxISqrt` + the `FxDot`/
`FxCross` Q16.16 products are int64 (world-scale products overflow int32). DXC compiles int64 (the Vulkan path);
glslc cannot parse `int64_t` in HLSL. So `convex_sat.comp` is **VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl)**; the Metal
`--convex-sat` runs the CPU `BoxSat` — byte-identical to the Vulkan GPU result BY CONSTRUCTION (the fpx_solve.comp /
boids_steer.comp convention), while the Vulkan side carries the GPU==CPU memcmp proof. Document this.

`FxMat3 { fx m[9]; }` (row-major 3×3 — the inertia tensor type, introduced here, fully used in CX3) + a few helpers
(identity, diagonal, `FxMat3MulVec`, `FxMat3Mul`, transpose — only what CX1 needs + the CX3 stubs; keep minimal).
`FxCross(a, b)` = the Q16.16 cross product (the `fpx::FxRotate` internal cross promoted to a public convex.h helper
— do NOT modify fpx.h). `FxBox { FxVec3 halfExtents; }`. `SatResult { bool overlap; uint32 axisIndex; fx
penetration; FxVec3 axis; }`. `BoxSat(world, boxes, i, j)` (or `BoxSat(bodyA, boxA, bodyB, boxB)`) → `SatResult`.
`MeasureSat(...)` → the count of overlapping pairs + the mean/min penetration — deterministic.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **fpx Q16.16 + body toolbox (`engine/sim/fpx.h`, read-only):** `FxVec3`/`FxQuat`/`FxBody{pos, orient, ...}`/
  `FxWorld`, `FxRotate` (rotate a local axis to world by the orient — fpx.h:~440), `FxDot`, `FxSub`, `FxNormalize`
  (~319, int64), `FxLength`/`FxISqrt` (~79, int64), `fxmul`/`fxdiv`, `kOne`/`kFrac`. **CONFIRM the exact
  signatures.** The internal cross product inside `FxRotate` (fpx.h:~441) is the body to copy into `FxCross`. **DO
  NOT modify fpx.h.**
- **The new-shader showcase precedent (`samples/hello_triangle/main.cpp`):** study `--fpx-shot` / `--fpx-pairs-shot`
  (the per-pair compute + the GPU==CPU memcmp + the standalone arg-parse) and BD1's `--boids-steer-shot` (the int64
  Vulkan-only new-shader GPU driver). Mirror these.
- **The int64-Vulkan-only shader wiring:** how `fpx_solve.comp` (int64, NOT in hf_gen_msl) is registered for DXC
  SPIR-V but excluded from `hf_gen_msl`. `convex_sat.comp.hlsl` follows the SAME wiring (confirm `convex` NOT in
  hf_gen_msl, grep 0).
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), a NEW `tests/convex_test.cpp` (+ CMake wiring, the fpx_test/boids_test
  pattern).

## Design decisions (locked)
1. **NEW `engine/sim/convex.h`** (header-only, namespace `hf::sim::convex`, `#include "sim/fpx.h"` read-only):
   `FxMat3` + the minimal mat helpers, `FxCross`, `FxBox`, `SatResult`, `BoxSat` (the 15-axis SAT above),
   `MeasureSat`. Pure integer, FIXED axis order (3 A-faces, 3 B-faces, 9 edge-crosses in a fixed (i,j) order), the
   min-pen strict-`<` lowest-index tie-break. **NEW shader** `convex_sat.comp.hlsl` (int64, Vulkan-only, one thread
   per box pair — copies `BoxSat`'s body VERBATIM). NOT in hf_gen_msl; Metal runs the CPU `BoxSat`.
2. **Showcase `--convex-sat-shot <out>` (Vulkan) AND `--convex-sat` (Metal) — WIRE BOTH** (standalone arg-parse).
   The SCENE: a fixed deterministic array of ORIENTED box pairs (~8-16 pairs) spanning the cases — clearly
   separated, deep face-face overlap, edge-edge overlap, a touching pair — each `FxBody` + `FxBox`. Vulkan: the GPU
   `convex_sat.comp` → **memcmp the GPU `SatResult[]` vs the CPU `BoxSat`** (NO tolerance — the make-or-break).
   Metal: the CPU reference. Render a PURE-INTEGER 2D top-down view (each box drawn as an oriented rectangle at its
   `pos>>kFrac`; overlapping pairs tinted hot / separated cold; optionally the min-pen axis as a short segment).
   Golden = `tests/golden/metal/convex_sat.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU `SatResult[]` == the CPU `BoxSat` byte-for-byte. Print `convex-sat:
     {pairs:<N>, overlapping:<O>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `convex-sat determinism: two runs BYTE-IDENTICAL`.
   - **(3) SAT correct:** the overlapping pairs match a brute-force/known-truth classification (the deep-overlap
     pairs report overlap=true with penetration>0 on the expected axis; the separated pairs report overlap=false)
     AND penetration is non-negative for every overlapping pair. Print `convex-sat correct: {matchesTruth:true,
     penNonNeg:true}`; assert both.
   - **(4) separation control:** a pair pulled far apart reports overlap=false (the axis test rejects). Print
     `convex-sat control: {farApart:separated}`.
   - **Golden discipline: ONLY `tests/golden/metal/convex_sat.png`; do NOT commit it.** Existing 183 image goldens
     UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests `tests/convex_test.cpp` (NEW, pure CPU):** `FxCross` matches the hand-computed cross of two axes;
   `FxMat3` identity/diagonal/MulVec correct; `BoxSat` — two clearly-separated boxes report overlap=false; a box
   deeply inside another reports overlap=true with the right min-pen axis + penetration>0; a face-face touching
   pair is the boundary case (overlap with ~0 penetration); an edge-edge overlap picks an edge-cross axis; the
   min-pen tie-break is the lowest index; two runs byte-identical. Clean under `windows-msvc-asan`. Wire the new
   test into CMake.
6. **Introspect.** Add exactly `deterministic-convex-sat` (features) + `--convex-sat-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the fpx/grain surface). `rhi.h` + backend
  dirs UNCHANGED. `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` +
  `vehicle.h` + `active.h` + `boids.h` + `engine/nav/` + `engine/anim/` + `engine/physics/` + all EXISTING shaders
  UNCHANGED. The ONLY new shader is `convex_sat.comp.hlsl` (int64, Vulkan-only, NOT in hf_gen_msl). `convex.h` is a
  NEW additive sibling (`#include`s fpx.h read-only). Report the seam empty (only `convex.h` + the new shader + the
  showcase/test/introspect are new/changed).

## Out of scope (YAGNI — later CX slices)
The contact manifold (CX2 — CX1 only computes the SAT min-pen axis + depth, NOT the contact points), the angular
impulse / inertia tensor solve (CX3 — `FxMat3` is introduced but only USED in CX3), the full convex step (CX4),
lockstep (CX5), the lit 3D render (CX6 — CX1's render is the 2D SAT diagnostic). Arbitrary convex hulls / GJK / EPA
(CX is BOXES only). CX1 claims ONLY: a deterministic integer box-box SAT (the min-penetration separating axis +
depth, or separation), bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs. NOTE (honest):
boxes only (15-axis box-box SAT, not general convex); the near-zero edge-cross skip is a deterministic degenerate
guard; int64 → Vulkan-GPU + Metal-CPU-ref (the FPX3 proof strength, not MSL-native strict-zero).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 104 + the NEW `convex_test`). Clean under `windows-msvc-asan`
   (build+run `convex_test` + `introspect_test`).
2. **proofs + visual:** `--convex-sat-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the box pairs (overlapping tinted hot, separated cold — a coherent diagnostic, not scrambled).**
3. Metal: `visual_test --convex-sat` → new golden `tests/golden/metal/convex_sat.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `convex_sat.comp` NOT in `hf_gen_msl` (int64, Metal runs the CPU
   BoxSat).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `convex_sat.png` added; the other
   183 byte-identical. `git diff master --stat -- tests/golden` = ONLY `convex_sat.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-convex-sat` + `--convex-sat-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`joint.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/
   `fract.h`/`vehicle.h`/`active.h`/`boids.h` + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL existing
   shaders byte-unchanged). `scripts/verify.ps1` updated: `convex_sat` golden in the Mac loop + `--convex-sat-shot`
   in `$vkShots`. **The ONLY new shader is `convex_sat.comp.hlsl` (int64, NOT in `hf_gen_msl`).**
