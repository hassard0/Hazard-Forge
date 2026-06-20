# Slice AC2 — Deterministic Active Ragdoll: PER-JOINT BLEND WEIGHT (THE PARTIAL-RAGDOLL AXIS) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #17 (DETERMINISTIC
> ACTIVE RAGDOLL / PHYSICAL-ANIMATION BLENDING, `hf::sim::active`). AC1 added the angular drive-to-target
> primitive; AC2 adds the **per-joint physical blend weight** — a per-drive `driveWeight ∈ [0, kOne]` that blends
> each joint between *fully anim-driven* (weight kOne — hold the target) and *fully limp physics* (weight 0 — no
> drive, the joint hangs free). This is UE5's "physical blend weight" / partial-ragdoll (the classic "shot in the
> legs, arms still bracing"): some bones track the pose, others go ragdoll. INTEGER-bit-exact. **NO new shader**
> (the weight is a per-record field read by AC1's `active_drive_solve.comp`, which gets the one new field). Branch:
> `slice-ac2`. See [[hazard-forge-active-roadmap]].

**Goal:** Extend `engine/sim/active.h` with a `driveWeight` field on `FxAngularDrive` (filling the AC1-reserved
32-byte GPU-mirror pad slot — **the std430 stride is UNCHANGED**) and scale the drive's applied correction by it in
`SolveAngularDrive`. Add `--active-blend-shot` (Vulkan) / `--active-blend` (Metal). Bake the integer golden
`active_blend`. **NO new shader, NO new RHI. AC1's `active_drive` golden stays BYTE-IDENTICAL** (driveWeight
defaults to kOne → exactly AC1 behavior — the render-invariance contract).

## Design call: driveWeight blends the APPLIED correction (kOne = AC1 drive, 0 = pure physics)
The AC1 `FxAngularDrive` GPU mirror already reserved a trailing pad int (the 28-byte logical record padded to a
32-byte std430 stride). AC2 repurposes that slot as `driveWeight` — **no stride change, no new shader file**, the
existing `active_drive_solve.comp` gains one field read. This is the intended in-flagship extension AC1 reserved
the pad for.

`driveWeight` is NOT the same as `stiffness`: `stiffness` is the per-iteration nlerp *rate* (a low stiffness still
converges to the target over K iterations — just slower); `driveWeight` is the *physical blend alpha* — how much
of the driven correction is applied at all. The correct model: weight **scales the magnitude of the inverse-mass
correction applied to the bodies**, so:
- `driveWeight = kOne` → the full AC1 correction → the joint tracks the target (active).
- `driveWeight = 0` → ZERO correction applied → the joint is pure physics (limp ragdoll), the drive is a no-op.
- intermediate → a partial pull toward the target competing with gravity/contacts (a "soft" physical blend).

**The implementation (extend AC1's `SolveAngularDrive`, render-invariant at weight kOne):** after computing
`qrelDriven` (the AC1 drive target) and the per-body correction targets `qBtarget`/`qAtarget`, scale the nlerp
share by the weight:
```
fx wA = fxdiv(a.invMass, wsum);  fx wB = fxdiv(b.invMass, wsum);
fx sA = fxmul(wA, drv.driveWeight);   // the weighted apply share
fx sB = fxmul(wB, drv.driveWeight);
b.orient = FxQuatNormalize(QNlerp(qB, qBtarget, sB));
a.orient = FxQuatNormalize(QNlerp(qA, qAtarget, sA));
```
At `driveWeight = kOne`: `fxmul(w, kOne) == w` exactly (Q16.16 multiply by 1.0 is identity) → the AC1 apply
byte-for-byte → **AC1's `active_drive` golden is unchanged** (the showcase/test build AC1 drives with
`driveWeight = kOne`). At `driveWeight = 0`: `fxmul(w, 0) == 0` → `QNlerp(q, target, 0) == q` → no rotation → pure
physics. `active_drive_solve.comp` mirrors the same two-line scale on its (formerly-pad) `driveWeight` field.

`FxAngularDrive` gains `fx driveWeight = kOne;` (defaulting to kOne so existing AC1 call-sites are unchanged in
behavior — they may omit it). The C++ logical record is now 32 bytes (the GPU mirror was already 32); the
static_asserts update to pin `driveWeight` at the former pad offset.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **AC1 (this branch's `active.h`, EXTEND — the one allowed in-flagship edit, render-invariant):** `FxAngularDrive`
  (add the `driveWeight` field at the reserved pad offset), `SolveAngularDrive` (add the two-line weight scale on
  the apply shares — keep everything else byte-identical so weight kOne == AC1), `DriveAngleCos`/`StepDriveWorld`/
  `StepDriveWorldSteps` (UNCHANGED — they pass the drives through). `FxDot4` UNCHANGED.
- **The AC1 shader (`shaders/active_drive_solve.comp.hlsl`, EXTEND — the SAME two-line scale on the formerly-pad
  field):** read `driveWeight` from the (already-present) pad slot of the std430 `FxAngularDrive` mirror; scale the
  apply shares. The stride + binding + dispatch are UNCHANGED. Still int64 Vulkan-only, NOT in hf_gen_msl.
- **The AC1 showcase precedent (`samples/hello_triangle/main.cpp` `--active-drive-shot`, `metal_headless/
  visual_test.mm` `--active-drive`):** copy the plumbing for `--active-blend-shot` / `--active-blend`; the GPU
  driver + memcmp + the 2D render are AC1 verbatim with the new per-joint weights.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), `tests/active_test.cpp` (add AC2 cases).

## Design decisions (locked)
1. **`FxAngularDrive` gains `fx driveWeight = kOne`** at the reserved 32-byte-mirror pad offset (std430 stride
   UNCHANGED; update the static_asserts). **`SolveAngularDrive`** scales the apply shares by `driveWeight` (the two
   lines above) — byte-identical to AC1 at weight kOne (the render-invariance contract). The shader mirrors it.
2. **Showcase `--active-blend-shot <out>` (Vulkan) AND `--active-blend` (Metal) — WIRE BOTH** (standalone
   arg-parse). The SCENE: a longer chain (≥6 links — an "upper body" + a "lower body") ball-jointed from a pinned
   root, with an `FxAngularDrive` per joint sharing one bent target pose, but **per-joint driveWeight**: the upper
   joints `driveWeight = kOne` (track the target — they hold the bracing pose), the lower joints `driveWeight = 0`
   (limp — they hang under gravity). Settle K `StepDriveWorld` steps → the upper chain HOLDS the pose while the
   lower chain HANGS free (the partial ragdoll). Vulkan: GPU drive solve → **memcmp vs the CPU StepDriveWorld**.
   Metal: the CPU reference. Render the AC1 2D side-view (discs + segments + distinct root; optionally tint
   driven-vs-limp joints). Golden = `tests/golden/metal/active_blend.png` (Mac-baked by the CONTROLLER — DO NOT
   commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU body world after K steps == the CPU `StepDriveWorld` byte-for-byte. Print
     `active-blend: {bodies:<N>, drives:<D>, drivenJoints:<W1>, limpJoints:<W0>, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `active-blend determinism: two runs BYTE-IDENTICAL`.
   - **(3) partial ragdoll:** the driven (weight kOne) joints hold the target (their `DriveAngleCos >= a band`)
     WHILE the limp (weight 0) joints DON'T (their orientation matches a pure-physics-no-drive reference for those
     joints, AND they hang lower / differ from the driven joints). Print `active-blend partial: {drivenHeld:true,
     limpFree:true}`; assert both.
   - **(4) weight-kOne equivalence (the render-invariance proof):** an all-`driveWeight = kOne` blend run produces
     the EXACT same body world as the AC1 all-driven `StepDriveWorld` (weight kOne == AC1). Print `active-blend
     equiv: {allWeightOne==AC1:true}`.
   - **Golden discipline: ONLY `tests/golden/metal/active_blend.png`; do NOT commit it. AC1's `active_drive.png`
     MUST stay byte-identical (driveWeight kOne == AC1).** Existing 172 image goldens UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests `tests/active_test.cpp` additions (pure CPU):** `SolveAngularDrive` with `driveWeight = kOne` ==
   AC1 (byte-identical orient); `driveWeight = 0` leaves orient at the pure-physics result (no drive applied);
   intermediate weight pulls partway. `StepDriveWorld` with mixed per-joint weights: the driven joints hold the
   pose, the limp joints hang free; two runs byte-identical. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-active-blend` (features) + `--active-blend-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` + `vehicle.h` +
  `engine/anim/` + `engine/physics/` UNCHANGED. **NO new shader** — `active_drive_solve.comp` gains the
  `driveWeight` field read (the stride/binding/dispatch UNCHANGED; still NOT in hf_gen_msl). `active.h`'s AC1
  functions are extended render-invariantly (weight kOne == AC1). Report the seam empty except active.h +
  active_drive_solve.comp (extended) + the showcase/test/introspect.

## Out of scope (YAGNI — later AC slices)
The anim-clip target step (AC3 — AC2's target is still a host-fixed pose, NOT sampled from a clip), the
active→limp→recover blend factor + impact (AC4), lockstep/rollback (AC5), the lit skinned render (AC6). A
mass-correct PD controller. AC2 claims ONLY: a deterministic per-joint physical blend weight that holds some
joints to the target pose while others go limp, bit-identical CPU↔Vulkan↔Metal, render-invariant for AC1, with the
integer golden + the four proofs. NOTE (honest, the AC1/JT2 caveat): the weight scales a soft nlerp correction —
the partial blend is a deterministic proxy (weight kOne holds within the Gauss-Seidel residual, weight 0 is exact
pure-physics), NOT a physically-blended torque.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 103 + the new `active_test` blend cases). Clean under
   `windows-msvc-asan` (build+run `active_test` + `introspect_test`).
2. **proofs + visual:** `--active-blend-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the partial ragdoll — the upper chain holding the bent pose, the lower chain hanging free (pixel-check).**
   **CRITICAL: re-run `--active-drive-shot` → STILL bit-exact AND its golden byte-identical (the AC1
   render-invariance — weight kOne == AC1).**
3. Metal: `visual_test --active-blend` → new golden `tests/golden/metal/active_blend.png`; two runs DIFF 0.0000.
   **Confirm `active_drive_solve.comp` still NOT in `hf_gen_msl`; confirm `tests/golden/metal/active_drive.png`
   UNCHANGED in the diff.** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `active_blend.png` added (NOT
   `active_drive.png`); the other 172 byte-identical. `git diff master --stat -- tests/golden` = ONLY
   `active_blend.png` (metal) + the introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-active-blend` + `--active-blend-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + the frozen sim/anim/physics headers + ALL OTHER shaders byte-unchanged; ONLY
   `active.h` + `active_drive_solve.comp` extended). `scripts/verify.ps1` updated: `active_blend` golden +
   `--active-blend-shot` in `$vkShots`. **NO new shader file; `active` still NOT in `hf_gen_msl`.**
