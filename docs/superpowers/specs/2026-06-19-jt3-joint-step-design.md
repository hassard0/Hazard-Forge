# Slice JT3 — Deterministic Articulated-Body Ragdoll: ARTICULATED MULTI-BODY STEP — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #15 (DETERMINISTIC
> ARTICULATED-BODY RAGDOLL, `hf::sim::joint`). JT1 pinned anchors (ball), JT2 limited orientations (hinge/cone);
> JT3 makes a real MECHANISM — one deterministic tick that integrates, then interleaves the joint constraints
> with rigid-body CONTACTS (the fpx broadphase + sphere-sphere solve), so a jointed chain swings, **self-collides
> and collides with the ground**, and settles as a coherent articulated structure. **NO new shader** — a host-
> driven multi-pass driver over the EXISTING JT1/JT2 + fpx int64 shaders (the FR4 `StepFracture` mold). Branch:
> `slice-jt3`. See [[hazard-forge-joint-roadmap]].

**Goal:** Extend `engine/sim/joint.h` (additive — JT1/JT2 byte-unchanged) with `StepArticulatedContacts` (the
full tick: integrate → broadphase → K Gauss-Seidel iters {ball | angular | contacts} → ground) +
`StepArticulatedContactsSteps` + a `MeasureArticulated` honest-metrics helper. Add `--joint-step-shot` (Vulkan) /
`--joint-step` (Metal) — a host-driven multi-pass driver over the EXISTING shaders. Bake the integer golden
`joint_step`. **NO new shader, NO new RHI.**

## Design call: a host-driven driver over JT1/JT2 + fpx contacts (the FR4 mold), NO new shader
JT2's `StepArticulated` already does integrate → K {ball | angular} → ground. JT3 adds the missing piece — rigid-
body CONTACTS — so the links don't pass through each other or the ground. The new step `StepArticulatedContacts`
is the FR4 `StepFracture` shape (a host-driven multi-pass driver over existing int64 shaders) with the two joint
passes added to each Gauss-Seidel iteration:
```
StepArticulatedContacts(world, joints, angularLimits, dt, iters, solveIters):
  (1) PREDICT:  IntegrateBodyFull(each body)            // FPX4 6-DOF integrate, VERBATIM
  (2) BUILD:    BuildPairs(world, offsets, pairs)       // FPX2 integer broadphase, ONCE per tick (the PBF choice)
  (3) ITERATE (K Gauss-Seidel iters), EACH in this FIXED locked order:
        (3a) all SolveBallJoint(world, joint[e])         in joint order      // JT1 ball (anchors coincide)
        (3b) all SolveAngularLimit(world, lim[e])        in limit order      // JT2 angular (hinge/cone)
        (3c) SolveContacts(world, pairs, solveIters)                          // FPX3 sphere non-penetration
  (4) GROUND:   the floor clamp (every DYNAMIC body pos.y >= groundY)
```
The links swing under gravity (held together by the ball joints, oriented by the angular limits) AND collide as
spheres (the FPX3 `SolveContacts`), so a chain self-collides + rests on the ground as a coherent mechanism. Pure
integer, fixed op order → two-run bit-identical AND bit-exact GPU==CPU. The GPU showcase drives the EXISTING
`joint_ball_solve.comp` + `joint_angular_solve.comp` + `fpx_solve.comp` (all int64, Vulkan-only) in the locked
order, host-rebuilding the pair list each tick, and memcmp's the final body set vs the CPU
`StepArticulatedContacts`. Metal runs the CPU reference. **NO new shader, NO new RHI.**

## Reuse map (file:line — the implementer MUST ground these before coding)
- **JT1/JT2 (this branch's `joint.h`, read-only — build on, DON'T modify):** `SolveBallJoint`, `WorldAnchor`,
  `AnchorGap`, `FxJoint`; `SolveAngularLimit`, `FxAngularLimit`, `SwingAngleCos`; `StepArticulated` (JT2 — JT3's
  `StepArticulatedContacts` is this + the contact pass). DO NOT modify JT1/JT2 code.
- **The fpx contact substrate to REUSE VERBATIM (`engine/sim/fpx.h`):** `IntegrateBodyFull` (:479), `CountPairs`/
  `BuildPairs` (:239/:264, the FPX2 broadphase), `SolveContacts` (:360, the FPX3 sphere-sphere non-penetration),
  `ResolveGround` (:329) / the floor clamp, `FxPair`, `FxBody`/`FxWorld`/`kFlagDynamic`. **DO NOT modify fpx.h.**
- **The FR4 host-driven driver to MIRROR (`engine/sim/fract.h::StepFracture` / its showcase):** the exact
  integrate → BuildPairs once → {solve passes} → ground tick + the GPU showcase that drives the existing int64
  shaders in the locked order and memcmp's vs the CPU reference. JT3 is the SAME with the two joint passes added.
  Also `couple_gf.h::StepCGF` / `couple_grain.h::StepCGrain` (the multi-pass coupled-step mold) + their
  `MeasureXxxState` helpers (JT3's `MeasureArticulated`).
- **Showcase + registration:** JT1/JT2's `--joint-*-shot` plumbing — **standalone arg-parse loop** (the FR1 C1061
  lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the
  introspect JSON golden**), `tests/joint_test.cpp`.

## Design decisions (locked)
1. **`StepArticulatedContacts(world, joints, angularLimits, dt, iters, solveIters)`** — the (1)-(4) locked tick
   above (integrate → broadphase once → K iters {ball | angular | contacts} → ground). `StepArticulatedContactsSteps`
   the K-step loop. `MeasureArticulated(world, joints)` → the max anchor gap (joints held), the mean dynamic
   pos.y (settled), the body rest line + residual-overlap count (a coherence stat). NO new shader; reuses JT1/JT2
   + fpx VERBATIM.
2. **Showcase `--joint-step-shot <out>` (Vulkan) AND `--joint-step` (Metal) — WIRE BOTH** (standalone arg-parse),
   host-driven multi-pass. A swinging, self-colliding CHAIN (or a small articulated structure): a pinned root +
   ~6-8 dynamic links with ball joints + hinge/cone limits + a real sphere radius, dropped/swung so it falls,
   the links collide with each other + the ground, and it settles into a coherent draped/piled mechanism. Vulkan:
   the GPU multi-pass driver (existing shaders, locked order) → **memcmp vs the CPU `StepArticulatedContacts`**.
   Metal: the CPU reference. Render a 2D side view (bodies as discs at `pos`, joints as segments). Golden =
   `tests/golden/metal/joint_step.png` (Mac-baked by the CONTROLLER — DO NOT commit). SMALL/FAST (≤~12 bodies).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU body world after K steps == the CPU `StepArticulatedContacts` reference
     byte-for-byte. Print `joint-step: {bodies:<N>, joints:<J>, limits:<L>, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `joint-step determinism: two runs BYTE-IDENTICAL`.
   - **(3) the mechanism settled, joints held:** after settling, every joint's anchor gap is within the band AND
     the chain came to rest at/above ground (mean pos.y >= groundY, dynamic bodies stopped descending). Print
     `joint-step settled: {maxAnchorGap:<G> within band, rested:true}`.
   - **(4) contacts did work (no interpenetration):** the FPX3 residual-overlap count is below a documented bound
     (the links + ground are non-penetrating — the contact pass held), vs a no-contact control where links
     overlap. Print `joint-step contacts: {residualOverlaps:<O>} (links non-penetrating)`; assert O below the
     bound, and that the no-contact control has more overlap (contacts do the work).
   - **Golden discipline: ONLY `tests/golden/metal/joint_step.png`; do NOT commit it.** Existing 161 image
     goldens UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels (every
   constituent pass is already integer-bit-exact cross-backend; the driver only sequences them).
5. **Tests `tests/joint_test.cpp` additions (pure CPU):** `StepArticulatedContacts` — a chain with sphere radii
   that would overlap is separated by the contact pass (residual overlap below the bound); the joints still hold
   (anchor gap small); a chain above the ground falls and rests (pos.y clamped); two runs byte-identical; a
   no-contact step (skip 3c) leaves more overlap (the contact pass does the work). Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-joint-step` (features) + `--joint-step-shot` (showcases). **REBAKE
   `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the FR4/JT1/JT2 surface). `rhi.h` +
  backend dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` +
  `engine/physics/` + all existing shaders UNCHANGED. JT1/JT2 `joint.h` code + the JT1/JT2 shaders UNCHANGED
  (JT3 additive — only the step + the showcase). **NO new shader.** Report the seam empty.

## Out of scope (YAGNI — later JT slices)
The skeleton→ragdoll bind (JT4 — JT3 is the generic articulated step over an arbitrary chain), lockstep (JT5),
the lit render (JT6). Convex-manifold contact / inertia tensor / torque-from-contact (the fpx sphere-sphere +
no-inertia caveat — links collide as spheres, contacts don't spin bodies). Soft/motor drives. JT3 claims ONLY:
a deterministic articulated step composing the joints with rigid-body contacts so a chain self-collides and
settles, bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 101) + the new `joint_test` step cases. Clean under
   `windows-msvc-asan` (build+run `joint_test` + `introspect_test`).
2. **proofs + visual:** `--joint-step-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the jointed chain settled — links connected, self-collided/non-overlapping, resting (pixel-check; the JT1/JT2
   lesson).** ALSO re-run `--joint-ball-shot` + `--joint-hinge-shot` → still bit-exact (JT1/JT2 render-invariance).
3. Metal: `visual_test --joint-step` → new golden `tests/golden/metal/joint_step.png`; two runs DIFF 0.0000 (gate
   on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (JT3 is pure
   orchestration — `hf_gen_msl` UNCHANGED; the JT1/JT2 Vulkan-only shaders unchanged).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `joint_step.png` added; the other
   161 byte-identical (re-run `--joint-ball/hinge-shot` → still bit-exact). `git diff master --stat --
   tests/golden` = ONLY `joint_step.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-joint-step` + `--joint-step-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/`fract.h` +
   `engine/physics/` + JT1/JT2 `joint.h`/shaders byte-unchanged). `scripts/verify.ps1` updated: `joint_step`
   golden in the Mac loop + `--joint-step-shot` in `$vkShots`. **NO new entry in `hf_gen_msl`.**
