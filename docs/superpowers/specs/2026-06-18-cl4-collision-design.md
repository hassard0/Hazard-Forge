# Slice CL4 — Deterministic GPU Cloth: INTEGER COLLISION (cloth-vs-FPX rigid) (Phase 13 #4) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #8
> (DETERMINISTIC GPU CLOTH, `hf::sim::cloth`, header `engine/sim/cloth.h`). Adds INTEGER COLLISION:
> project the CL3-solved cloth particles out of a set of `fpx::FxBody` SPHERE colliders + the ground
> plane — so the cloth DRAPES over a rigid sphere. The cloth and the rigid sphere are the SAME Q16.16
> world units, so this is the FIRST deformable-meets-rigid INTEGER interaction (cloth composing with the
> FPX flagship). int64 normalize → Vulkan-only SPIR-V + Metal runs the byte-identical CPU reference (the
> CL3/FPX3 convention). Strict zero-diff cross-backend (the integer bar). ZERO new RHI. Branch:
> `slice-cl4`. See [[hazard-forge-cloth-roadmap]].

**Goal:** Extend `engine/sim/cloth.h` with `CollideSpheres` (project each particle out of any overlapping
`fpx::FxBody` sphere: if `FxLength(p - center) < radius`, move `p` to the surface along the outward
normal) and `CollidePlane` (clamp `p.y >= groundY`, already in CL3 — extend to an arbitrary plane if
convenient), and fold them into `StepCloth` (apply collision AFTER the constraint passes each step). Add a
`SphereCollider` (reuse `fpx::FxBody`'s pos+radius) set to the showcase. Add `shaders/cloth_collide.comp.hlsl`
(per-particle, int32 AABB reject + int64 normalize on overlaps → Vulkan-only + Metal CPU path) — OR fold
collision into the existing `cloth_solve.comp` step loop (pick one, document). The `cloth_collide` integer
golden (the cloth draped over a sphere, CPU-colored from the integer read-back → strict ZERO-DIFFERING-
PIXEL cross-backend), `--cloth-collide-shot` (Vulkan) / `--cloth-collide` (Metal), and `tests/cloth_test.cpp`
additions. Reuse CL1–CL3 + `fpx::FxBody` verbatim — CL4 is additive (CL1–CL3 pipelines + goldens stay
byte-identical; `fpx.h` read-only).

## Design call: INTEGER bit-exact (the CL3 discipline, extended to collision)
Collision is a positional projection (push the particle to the sphere surface along the normal) — the same
class as the PBD constraint solve. Per particle it is INDEPENDENT (each particle vs the static, read-only
collider set), so it is NOT order-dependent → it can be one-thread-per-particle (unlike the Gauss-Seidel
solve). BUT the normalize uses int64 (`FxNormalize` via `FxISqrt`), so the shader is Vulkan-only + Metal
runs the CPU reference (the CL3 convention). The golden is the CPU-colored integer particle read-back →
Vulkan==Metal BIT-IDENTICAL (strict zero-diff). The collision is applied each step AFTER the constraints
(so the cloth both holds together AND stays out of the sphere); a small static collider set keeps it cheap
(no broadphase needed — direct per-particle test, the int32 AABB reject before the int64 normalize keeps
the common no-overlap case fast). **HONEST CAVEAT (CL3-identical): a single positional projection per step
is not a full contact solve — interpenetration can transiently occur and is resolved over steps
(deterministic). Claim DETERMINISM + cross-platform bit-identity + a visible drape, NOT a physically
perfect contact model. Self-collision (cloth-vs-cloth) is OUT of scope (a future O(n²)/grid-hash slice).**

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The FPX rigid collider to compose with (read-only):** `engine/sim/fpx.h::FxBody` (pos + radius) — the
  SAME Q16.16 world units as `ClothParticle.pos`. The showcase places one or two static `FxBody` spheres;
  the cloth drapes over them. `fpx::FxLength`/`FxNormalize`/`FxSub`/`FxScale` (the projection math, int64).
- **The CL3 solver to fold collision into:** `engine/sim/cloth.h::StepCloth` (the integrate + K
  constraint passes + floor-clamp) — add the collision projection after the constraint passes. CL3's
  `cloth_solve.comp` (`[numthreads(1,1,1)]` single-thread, int64, Vulkan-only) — if folding collision into
  the same step loop, keep the single-thread dispatch (the constraint solve is still order-dependent); the
  per-particle collision within it is order-independent but runs in the same thread. (If a SEPARATE
  per-particle collision dispatch is cleaner, `cloth_collide.comp` can be multi-thread — but the int64
  normalize keeps it Vulkan-only either way; mind the CL3 TDR ceiling on total single-thread duration.)
- **The single-thread/int64 Vulkan-only + Metal-CPU convention:** CL3's `cloth_solve` + `fpx_solve`
  (Vulkan-only shader, Metal runs CPU `StepCloth`). `RunClothSolveShowcase` is the structural template.
- **The integer-golden showcase discipline:** CL3's `--cloth-solve-shot` (ReadBuffer integer particles,
  memcmp GPU==CPU, CPU-color the draped lattice, strict zero-diff).
- **RHI compute envelope (must fit, ZERO additions):** `engine/rhi/rhi.h` — the CL1–CL3/FPX set.
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--cloth-collide-shot` standalone arg
  branch + DXC list), `metal_headless/{visual_test.mm,CMakeLists.txt}` (`RunClothCollideShowcase` +
  `--cloth-collide` — Vulkan-only shader NOT in `hf_gen_msl`, Metal runs CPU `StepCloth`+collision),
  `engine/editor/introspect.cpp` (+`deterministic-cloth-collide` feature + `--cloth-collide-shot`
  showcase) + `tests/introspect_test.cpp` + the JSON golden, `scripts/verify.ps1` (`cloth_collide` golden
  in the Mac loop + `--cloth-collide-shot` in `$vkShots`).

## Design decisions (locked)
1. **The collision projection (the bit-exact core).** `CollideSpheres(particles, spheres)`: per particle
   (if not pinned), for each sphere `s`: `d = FxSub(p, s.center); dist = FxLength(d);` if `dist <
   s.radius` (the int32 AABB reject first: skip if `|p-center|` per-axis > radius), `n = FxNormalize(d)`
   (if `dist==0`, a fixed default normal e.g. +Y); `p = FxAdd(s.center, FxScale(n, s.radius))` (snap to the
   surface). `CollidePlane`: `p.y = max(p.y, groundY)` (CL3's clamp). Applied each step AFTER the
   constraint passes. Pure int64-backed Q16.16, copied verbatim CPU↔shader. Deterministic (fixed sphere
   order, fixed per-particle order).
2. **Fold into StepCloth.** `StepCloth(..., spheres)`: integrate → K constraint passes → `CollideSpheres`
   + `CollidePlane`. The collider set is a small static `std::vector<SphereCollider>` (or reuse
   `fpx::FxBody`). Pinned particles still never move. The cloth drapes over the sphere(s).
3. **GPU pipeline.** Either (a) fold collision into the CL3 `cloth_solve.comp` single-thread step loop
   (simplest — one shader, the existing dispatch; mind the TDR ceiling — keep steps×iters under it as CL3
   does), OR (b) a separate per-particle `cloth_collide.comp`. Pick one, document. int64 → Vulkan-only;
   Metal runs the CPU `StepCloth`+collision. Host-snapped integers in → integers out → GPU==CPU bit-exact.
4. **Showcase `--cloth-collide-shot <out>` (Vulkan) AND `--cloth-collide` (Metal — WIRE BOTH; confirm
   visual_test.mm + `#include "sim/cloth.h"`).** A cloth sheet (e.g. 24×24, top edge or 4 corners pinned,
   OR unpinned and dropped) positioned ABOVE a static `FxBody` sphere, `StepCloth` K steps under gravity →
   the cloth FALLS and DRAPES over the sphere (the collision holds it on the surface). ReadBuffer the
   integer `gParticles`; **memcmp GPU == the CPU `StepCloth`(+collision) reference (the make-or-break)**;
   CPU-color a 3/4 or side view of the draped cloth + the sphere outline → `tests/golden/metal/
   cloth_collide.png` (baked on the Mac by the CONTROLLER — DO NOT commit).
5. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance / GPU==CPU (make-or-break):** `gParticles` equals the CPU `StepCloth`(+collision)
     reference byte-for-byte. Print `cloth-collide: {particles:<N>, spheres:<S>, contacts:<C>, steps:<K>}
     GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs BYTE-IDENTICAL. Print `cloth-collide determinism: two runs
     BYTE-IDENTICAL`.
   - **(3) collision / coherence:** NO particle ends up inside a sphere (every particle's `dist >=
     radius - epsilon` deterministically), and `contacts > 0` (the cloth touches the sphere — a real
     drape). Print `cloth-collide coverage: <C> contacts, 0 penetrating (cloth drapes over sphere)`.
   - **(4) no-collider / no-op:** zero spheres + the CL3 scene → byte-identical to `cloth_solve` (collision
     is a no-op). Print `cloth-collide no-collider: == solve (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/cloth_collide.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 120 image goldens UNTOUCHED.
6. **Cross-backend bar (INTEGER, strict).** GPU==CPU-by-construction (Metal runs the CPU StepCloth+collision;
   Vulkan the int64 shader == that CPU) → Vulkan==Metal BIT-IDENTICAL: the golden is the CPU-colored
   integer read-back; the controller's cross-backend check is the STRICT ZERO-DIFFERING-PIXEL compare. Any
   nonzero cross-backend pixel diff is a real bug.
7. **Tests `tests/cloth_test.cpp` additions (pure CPU):** a single particle inside a sphere → projected to
   exactly the surface (`FxLength(p-center) == radius` within fp tol); a particle outside → untouched; the
   ground clamp; a small sheet over a sphere → no particle penetrates after K steps (deterministic);
   zero-sphere == pure CL3 solve; determinism. Clean under `windows-msvc-asan`.
8. **Introspect.** Add exactly `deterministic-cloth-collide` (features) + `--cloth-collide-shot`
   (showcases). Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None expected.** Pure SSBO compute (the CL1–CL3/FPX set; the collider set is a small storage buffer or
  push-constant). `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. CL1–CL3 shaders +
  `engine/sim/fpx.h` + `engine/physics/` UNCHANGED. Report the seam.

## Out of scope (YAGNI — later CL slices)
Lockstep/rollback (CL5), the float render (CL6). SELF-collision (cloth-vs-cloth — a documented future
slice, O(n²)/grid-hash). DYNAMIC colliders moving with FPX (CL4's colliders are a small STATIC set —
documented). Friction (the projection is frictionless). CL4 is ONLY the cloth-vs-static-sphere/plane
collision + drape + its bit-exact golden. No float. The single-projection-per-step contact is
deterministic-but-not-physically-perfect (the documented caveat).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 94) + the new `cloth_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--cloth-collide-shot` on Vulkan: GPU==CPU memcmp BIT-EXACT + determinism +
   collision (0 penetrating, contacts>0) + no-collider no-op; a coherent cloth DRAPED OVER A SPHERE. Run
   under the Vulkan-validation gate → ZERO VUID in the OUTPUT (set BOTH `VK_LAYER_PATH` to the conan
   `...\.conan2\p\...\layers` dir AND `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer
   LOADED = zero "not found"). Keep the steps×iters under the CL3 TDR ceiling.
3. Metal: `visual_test --cloth-collide` → new golden `tests/golden/metal/cloth_collide.png`; two runs DIFF
   0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm the int64 shader
   is correctly EXCLUDED from `hf_gen_msl` (Vulkan-only) and the Metal showcase runs the CPU
   StepCloth+collision.** Cross-backend = STRICT ZERO-DIFFERING-PIXEL, NOT the float baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cloth_collide.png` added;
   the other 120 byte-identical (CL1–CL3 + all existing untouched). `git diff master --stat --
   tests/golden` = ONLY `cloth_collide.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cloth-collide` + `--cloth-collide-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report the int64 Vulkan-only + Metal CPU path).
   `scripts/verify.ps1` updated: `cloth_collide` golden in the Mac loop + `--cloth-collide-shot` in
   `$vkShots`. CL1–CL3 + `engine/sim/fpx.h` + `engine/physics/` UNTOUCHED.
