# Slice FPX4 — Deterministic Fixed-Point Physics: INTEGER QUATERNION ORIENTATION (Phase 11 #4) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 4th FPX slice (after FPX1–FPX3
> gave a translational rigid-body sim): add ORIENTATION — integrate each body's rotation from its angular velocity as
> a FIXED-POINT quaternion, normalized via integer sqrt — so the physics state is full 6-DOF (position + orientation)
> and still BIT-IDENTICAL CPU↔Vulkan↔Metal AND frame-deterministic. The prerequisite for box colliders (later) and a
> richer lockstep state (FPX5). Proven GPU==CPU BIT-EXACT, integer, cross-backend BIT-IDENTICAL. NO new RHI. The
> quaternion math is int64 (fxmul products) → Vulkan-SPIR-V-only + the Metal showcase runs the CPU path (the FPX3/
> swraster convention). Namespace `hf::sim::fpx`. Branch: `slice-fpx-orient`. See [[hazard-forge-fpx-roadmap]].

**Goal:** Extend `engine/sim/fpx.h` with a fixed-point quaternion + orientation integrator (`FxQuat`, `FxQuatMul`,
`FxQuatNormalize`, `FxRotate`, orientation fields on `FxBody`, `IntegrateOrientation`, `IntegrateBodyFull`) +
`shaders/fpx_orient.comp.hlsl` (one thread per body, integrate translation + orientation K steps) + a
`--fpx-orient-shot` (Vulkan) / `--fpx-orient` (Metal) showcase that spins a grid of bodies with staggered angular
velocities, integrates K steps, reads back the bodies, proves the orientations BIT-EXACT vs the CPU reference, and
bakes an orientation-gizmo golden. Make-safe: header additions (FPX1–FPX3's `IntegrateStep`/`StepWorld` and their
goldens UNCHANGED — orientation is DEFAULTED on `FxBody` so the existing translational path/packs are byte-identical)
+ a NEW shader + NEW showcase + NEW golden; the float `engine/physics/` UNCHANGED.

## The fixed-point quaternion core (extends fpx.h)
- **`struct FxQuat { fx x, y, z, w; };`** — a Q16.16 quaternion (identity = `{0, 0, 0, kOne}`).
- **`FxBody` gains `FxQuat orient` (DEFAULT identity) + `FxVec3 angVel` (DEFAULT {0,0,0})** — defaulted so FPX1–FPX3
  bodies/tests/showcases + their GPU packs (which only pack the FPX1–FPX3 fields) are unchanged. Document.
- **`FxQuat FxQuatMul(const FxQuat& a, const FxQuat& b)`** — the Hamilton product via `fxmul` (int64 intermediates):
  `w = aw*bw - ax*bx - ay*by - az*bz`, `x = aw*bx + ax*bw + ay*bz - az*by`, etc. (the standard quaternion product,
  each term an `fxmul`).
- **`FxQuat FxQuatNormalize(const FxQuat& q)`** — `len = FxISqrt(x²+y²+z²+w² in int64 Q-format)`; if `len==0` return
  identity; else `{ fxdiv(x,len), fxdiv(y,len), fxdiv(z,len), fxdiv(w,len) }`. Integer sqrt + truncating divide — NO
  `std::sqrt`. (Fixed-point normalization is not perfect — `|q|` drifts slightly but DETERMINISTICALLY; that's fine,
  the bit-exactness + a `|q|≈kOne within tol` check are the proofs.)
- **`FxVec3 FxRotate(const FxQuat& q, const FxVec3& v)`** — rotate a vector by a quaternion (`q * (v,0) * conj(q)`, or
  the optimized `v + 2*cross(q.xyz, cross(q.xyz, v) + q.w*v)` form — pick one, all `fxmul`). For the gizmo viz.
- **`void IntegrateOrientation(FxBody& b, fx dt)`** — the deterministic quaternion angular integrator: `dq =
  FxQuatMul({angVel.x, angVel.y, angVel.z, 0}, orient)` (the pure-quaternion ω⊗q); `orient = { orient.x +
  fxmul(fxmul(dq.x, kHalf), dt), ... }` (orient += 0.5·dq·dt, component-wise); `orient = FxQuatNormalize(orient)`.
  (`kHalf = kOne/2`.) Fixed order, per-body independent.
- **`void IntegrateBodyFull(FxBody& b, const FxVec3& gravity, fx dt)`** = the FPX1 translational `IntegrateStep` body
  + `IntegrateOrientation`. (FPX4's showcase uses this; FPX1–FPX3 keep calling the original translational
  `IntegrateStep` — UNCHANGED.)

## Reuse map (file:line)
- **FPX1–FPX3 (the inputs + the int64 discipline):** `engine/sim/fpx.h` — `fx`/`fxmul`/`FxVec3`/`FxISqrt` (FPX1),
  `fxdiv`/`FxNormalize` (FPX3 — `FxQuatNormalize` mirrors it for 4 components). `IntegrateStep`/`StepWorld` STAY
  UNCHANGED.
- **The single-thread / per-body GPU compute + the int64/Vulkan-only convention (copy FPX3):** `shaders/
  fpx_solve.comp.hlsl` (the per-thread serial structure); `fpx_solve.comp` is Vulkan-only + the Metal `--fpx-solve`
  showcase runs the CPU path — `fpx_orient.comp` (int64 quat mul) does the SAME (Vulkan compile list only; Metal
  `--fpx-orient` runs the CPU `IntegrateBodyFull`). (FPX4's orientation integrator is per-body INDEPENDENT — unlike
  FPX3's serial solver — so `fpx_orient.comp` can be one-thread-PER-BODY [64 threads/group], not single-thread; only
  the int64 forces Vulkan-only.)
- **The integer-from-readback debug-viz golden + integer line-draw (for the gizmo):** the swraster host-snap
  `pos>>kFrac → pixel`; if a gizmo needs lines, an integer Bresenham (or reuse any existing integer line helper —
  grep `debug_line`/`DrawLine`; else a simple integer DDA in the showcase). `meshlet.h:79` `hashColor`.
- **Showcase + registration:** the FPX1/FPX3 showcase shapes; `verify.ps1`/`introspect.cpp`/`introspect_test.cpp`.

## Design decisions (locked)

1. **`fpx_orient.comp.hlsl` (NEW).** ONE thread per body. Reads `gBodies` (the FPX4 pack: pos/vel/invMass/flags +
   orient.xyzw + angVel.xyz — std430, document the layout + static_assert) + `gParams{bodyCount, gravity, dt, steps,
   enabled}`. Runs `steps` of `IntegrateBodyFull` (the `FxQuatMul`/`FxQuatNormalize`/`FxISqrt`/`fxdiv` copied VERBATIM
   from fpx.h), writes `gBodies` back. `enabled=0` → write input back. Per-body independent → NO atomics, 64
   threads/group. **int64 (fxmul/fxdiv in quat math) → Vulkan-SPIR-V-only: in the Vulkan compile list, NOT the Metal
   hf_gen_msl list; the Metal `--fpx-orient` showcase runs the CPU `IntegrateBodyFull` over the same bodies →
   byte-identical by construction (the FPX1/FPX3 convention). Add the explanatory comment in metal_headless/
   CMakeLists.txt.**
2. **Showcase `--fpx-orient-shot <out>` (Vulkan, main.cpp) AND `--fpx-orient` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm + `#include "sim/fpx.h"`).** A deterministic grid of bodies (e.g. 6×6), each with a distinct
   DETERMINISTIC initial `angVel` (e.g. derived from the grid index → varied spin axes/rates), identity initial
   orientation, NO gravity (free rotation, so the bodies stay put + just spin → the viz is purely orientation),
   dt=kOne/60, K=120 steps. Upload bodies, dispatch `fpx_orient` (Vulkan; CPU `IntegrateBodyFull` on Metal),
   `ReadBuffer` `gBodies`. CPU-run `IntegrateBodyFull` K times. Golden = a grid of orientation GIZMOS: for each body,
   `FxRotate(orient, axis)` for the 3 local axes (x→red, y→green, z→blue), project each rotated axis to 2D and draw a
   short colored segment from the body's grid-cell center (integer line-draw), so each cell shows the body's final
   orientation as a 3-axis gizmo → `tests/golden/metal/fpx_orient.png` (a grid of differently-oriented gizmos;
   CPU-rendered from the integer quaternions → cross-backend bit-identical). (If a 3-axis gizmo is too much, a SINGLE
   rotated-forward-axis colored dot per cell is an acceptable simpler viz — document the choice; PREFER the gizmo for
   richness.)
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact (make-or-break):** `memcmp(gpuBodies, cpuBodies) == 0` after K steps (full state incl
     the quaternions, NO tolerance). Print `fpx-orient GPU==CPU bodies: <N> bodies BIT-EXACT (<K> steps)`.
   - **(2) quaternions stay normalized (deterministic):** for every body, `||orient|² - kOne²|` is within a documented
     fixed-point tolerance (the fixed-point normalize keeps `|q|≈1`); report the max drift. Print `fpx-orient
     normalized: max |q|-1 drift <d> (<=tol) OK`.
   - **(3) hand-checked rotation:** one body with a known `angVel` about an axis for K steps → assert its quaternion
     ≈ the closed-form rotation (within fp tol), OR exactly matches a hand-precomputed Q16.16 value. Print `fpx-orient
     hand-check: body0 quat = <q16.16 xyzw> OK`.
   - **(4) disabled-path no-op:** `enabled=false` → bodies unchanged. Print `fpx-orient disabled: bodies UNCHANGED
     (no-op)`.
   - **(5) determinism:** two runs → byte-identical. Print `fpx-orient determinism: two runs BYTE-IDENTICAL`.
   - **Golden discipline: ONLY `tests/golden/metal/fpx_orient.png`; do NOT commit it — the CONTROLLER bakes on the
     Mac.** Existing 108 image goldens UNTOUCHED.
4. **Determinism / cross-backend.** The quaternion integrator is fixed-point on host-snapped integers, per-body
   independent; the int64 fxmul/fxdiv are pinned; Vulkan runs the GPU shader (DXC int64), Metal runs the CPU
   `IntegrateBodyFull` → byte-identical; the golden is CPU-rendered from the integer quaternions → strict zero-diff
   cross-backend. Run under the Vulkan sync-validation gate → SYNC-HAZARD-free.
5. **Tests `tests/fpx_test.cpp` additions (pure CPU):** `FxQuatMul` (identity·q==q, known products, the int64
   intermediate); `FxQuatNormalize` (`|q|≈kOne`, a known un-normalized quat → known normalized); `FxRotate` (identity
   rotates v→v; a 90° rotation about an axis → the known rotated vector within tol); `IntegrateOrientation` one/K-step
   (a known angVel → known orientation drift; angVel=0 → orient unchanged); `IntegrateBodyFull` = translation + orient;
   `enabled`-off → unchanged; determinism; the `|q|` drift is deterministic. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-fixedpoint-physics-orient` (features) + `--fpx-orient-shot` (showcases).

## RHI seam additions (summary)
- **None.** `BufferUsage::Storage` + compute + `ReadBuffer` — the FPX1–FPX3 precedent. ZERO above-seam backend
  symbols. `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. `engine/physics/` + FPX1–FPX3 shaders/goldens
  UNTOUCHED. Report the seam.

## Out of scope (YAGNI — FPX5+)
The box collider via fixed-point SAT (a later slice — FPX4 is orientation INTEGRATION only; oriented-box collisions
build on it), angular response/torque from contacts (positional only so far), the lockstep replica==authority proof
(FPX5 — now with full 6-DOF state), the float render (FPX6). Physical-accuracy claims — claim DETERMINISM +
cross-platform BIT-IDENTITY. ONE fixed-point quaternion orientation integrator with the GPU==CPU bit-exact proof +
the normalized-quaternion check + the hand-checked rotation + disabled no-op + determinism and the orientation-gizmo
golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 92) + the new `fpx_test` quaternion cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--fpx-orient-shot` on Vulkan: a coherent grid of orientation gizmos (each body's rotated
   axes); all 5 proof lines. Run under the Vulkan-validation gate → ZERO VUID in the OUTPUT.
3. Metal: `visual_test --fpx-orient` → new golden `tests/golden/metal/fpx_orient.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). **fpx_orient.comp is int64 → Vulkan-only; the Metal showcase runs the CPU `IntegrateBodyFull`
   → byte-identical. Confirm visual_test.mm wired + fpx_orient.comp in the Vulkan compile list + NOT the Metal
   hf_gen_msl list.** Integer golden → a strict cross-backend pixel compare must show ZERO differing pixels.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fpx_orient.png` added; the other 108
   byte-identical (fpx/fpx_pairs/fpx_solve untouched). `git diff master --stat -- tests/golden` = ONLY
   `fpx_orient.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fixedpoint-physics-orient` + `--fpx-orient-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `scripts/verify.ps1` updated: `fpx_orient` golden in the Mac
   loop + `--fpx-orient-shot` in `$vkShots`.
