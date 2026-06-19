# Slice JT4 — Deterministic Articulated-Body Ragdoll: SKELETON→RAGDOLL BIND (the pillar-bridge) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #15 (DETERMINISTIC
> ARTICULATED-BODY RAGDOLL, `hf::sim::joint`) — **THE PILLAR-BRIDGE + the headline setup**. JT1-JT3 built a
> generic jointed mechanism; JT4 maps an **`anim::Skeleton`** onto it: each bone becomes an `fpx::FxBody`, each
> parent-child edge a `FxJoint` (ball) + `FxAngularLimit` (cone), so the float animation skeleton becomes a
> bit-exact RAGDOLL that collapses under gravity into a settled pose — and the pose reads back as a joint palette
> for the (later JT6) skinning path. This UNITES the physics moat with the anim pillar. **NO new shader** — the
> bind (float→Q16.16) + read-back (Q16.16→float palette) are host conversions; the collapse SIM is the bit-exact
> JT3 `StepArticulatedContacts`. Branch: `slice-jt4`. See [[hazard-forge-joint-roadmap]].

**Goal:** Extend `engine/sim/joint.h` (additive — JT1/JT2/JT3 byte-unchanged) with `RagdollFromSkeleton`
(`anim::Skeleton` → `{fpx::FxWorld + std::vector<FxJoint> + std::vector<FxAngularLimit>}`) + `PoseToPalette`
(settled `FxWorld` → `std::vector<math::Mat4>` joint palette) + a `MeasureRagdoll` honest-metrics helper. Add
`--joint-ragdoll-shot` (Vulkan) / `--joint-ragdoll` (Metal). Bake the integer golden `joint_ragdoll`. **NO new
shader, NO new RHI.**

## Design call: bind the skeleton to a ragdoll (host float→Q16.16), collapse it bit-exact, read the pose back
The float `anim::Skeleton` (`anim/skeleton.h`: `Joint{int parent; math::Mat4 inverseBind; Vec3 t; Quat r; Vec3
s;}`, topologically sorted) becomes a fixed-point ragdoll once, at bind time:

**(1) `RagdollFromSkeleton(skeleton, cfg)` — the float→Q16.16 bind (host, outside the bit-exact loop).**
- **Bind-pose globals (float):** forward-accumulate each joint's model-space bind transform over the topological
  order — `global[j] = (parent<0 ? I : global[parent[j]]) · TRS(t,r,s)` (the SAME single-forward-pass the anim
  `SampleAnimation` uses; the topological sort guarantees the parent is ready). Read it to mirror exactly.
- **Body per joint:** an `fpx::FxBody` with `pos = global[j].translation · kWorldScale` (host float → Q16.16
  snap), `orient = global[j].rotation` (→ Q16.16 `FxQuat`), `radius = boneRadius` (a fraction of the distance to
  the parent's body — the capsule-as-sphere proxy; a documented simplification), `invMass` from `cfg`. The ROOT
  joint's body is `cfg.rootStatic ? pinned (invMass 0) : dynamic` (a free ragdoll collapses fully; a pinned-root
  ragdoll dangles).
- **Joint + limit per edge:** for each non-root joint `j` with parent `p`, a `FxJoint{bodyA=p, bodyB=j, anchorA,
  anchorB, kind=ball}` with `anchorA`/`anchorB` the joint's bind position expressed in each body's LOCAL frame
  (`FxRotate(QConj(orient), worldJointPos − bodyPos)`), so the ball pins the bone's head to the parent; AND a
  `FxAngularLimit{bodyA=p, bodyB=j, axis (the bone's bind direction in p's local frame), cosHalfLimit,
  sinHalfLimit, kind=cone}` from `cfg.coneLimit` (host cos/sin). The whole map is near-mechanical — the skeleton
  tree IS the articulation graph.

**(2) The collapse (bit-exact integer SIM).** `StepArticulatedContactsSteps(world, joints, angularLimits, …)`
(JT3, VERBATIM) — the ragdoll falls under gravity, the bones stay connected (ball joints), the cones limit the
swing, the bodies self-collide + hit the ground, and it settles into a slumped pose. **This is the bit-exact
claim: GPU==CPU + cross-vendor zero on the integer ragdoll state.** The golden renders this collapsed integer
state.

**(3) `PoseToPalette(skeleton, world) → std::vector<math::Mat4>` — the Q16.16→float read-back.** For each joint
`j`, the body's current world transform `fpx::FxBodyTransform(world.bodies[j])` (the FPX6 float bridge) is the
joint's animated global transform; the skinning palette entry is `palette[j] = bodyGlobal[j] · skeleton.joints[j]
.inverseBind` (the standard `global · inverseBind` skinning matrix). This is the joint palette the GPU skinning
pipeline consumes (wired to a skinned character in JT6). Pure deterministic host float.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The skeleton + its global-transform forward pass (`engine/anim/skeleton.h` + `engine/anim/animation.cpp`):**
  `Skeleton{joints}`, `Joint{parent, inverseBind, t, r, s}` (topologically sorted). Read `animation.cpp`'s
  `SampleAnimation` (the `global[j] = global[parent]·local[j]` single forward pass + the `global·inverseBind`
  palette build) — `RagdollFromSkeleton`'s bind globals + `PoseToPalette` mirror it. DO NOT modify anim.
- **JT1/JT2/JT3 (this branch's `joint.h`, read-only — build on, DON'T modify):** `FxJoint`/`SolveBallJoint`/
  `WorldAnchor`, `FxAngularLimit`/`SolveAngularLimit` (+ `kAngularCone`), `StepArticulatedContacts`/`Steps`/
  `MeasureArticulated`, `QConj`/`QNlerp`. The collapse uses `StepArticulatedContacts` VERBATIM.
- **The fpx body + render bridge to REUSE VERBATIM (`engine/sim/fpx.h`):** `FxBody`/`FxWorld`/`kFlagDynamic`,
  `FxRotate` (:440, world↔local anchor/axis), `FxQuatNormalize`, `FxBodyTransform` (:627, the Q16.16→float
  transform `PoseToPalette` reads back), `FxToFloat`. The math float→fixed snaps use the standard `(fx)(v·kOne)`.
  DO NOT modify fpx.h.
- **The math types (`engine/math/math.h`):** `Mat4` (translation/rotation extract for the bind globals + the
  palette product), `Quat`, `Vec3`, `Mat4::operator*`. DO NOT modify.
- **A synthetic test skeleton:** the showcase BUILDS a fixed, host-coded `anim::Skeleton` (e.g. a ~9-joint
  humanoid-ish tree — a spine of 3 + 2 arms + 2 legs, or a simple branching chain) with hand-set TRS +
  inverseBind, so the golden is deterministic + self-contained (NO glTF load). If `tests/` already has a
  reusable test skeleton, prefer it; else construct one in the showcase.
- **Showcase + registration:** JT1-JT3's `--joint-*-shot` plumbing — **standalone arg-parse loop** (the FR1 C1061
  lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the
  introspect JSON golden**), `tests/joint_test.cpp`.

## Design decisions (locked)
1. **`RagdollFromSkeleton(skeleton, cfg) → struct Ragdoll{fpx::FxWorld world; std::vector<FxJoint> joints;
   std::vector<FxAngularLimit> limits;}`** (the bind above) + `RagdollConfig{fx worldScale, boneRadius, invMass,
   cone cos/sin, gravity, groundY; bool rootStatic;}`. The bind globals + the float→Q16.16 snaps are host (the
   documented single float crossing at bind — render-only-adjacent, outside the bit-exact loop).
2. **`PoseToPalette(skeleton, world) → std::vector<math::Mat4>`** (the read-back above; `bodyGlobal·inverseBind`).
   `MeasureRagdoll(skeleton, ragdoll)` → the max anchor gap (bones connected), the mean body pos.y (collapsed),
   the body rest line — deterministic stats.
3. **Showcase `--joint-ragdoll-shot <out>` (Vulkan) AND `--joint-ragdoll` (Metal) — WIRE BOTH** (standalone
   arg-parse). Build the synthetic skeleton → `RagdollFromSkeleton` (free or pinned root) → drop it →
   `StepArticulatedContactsSteps` K steps → the ragdoll collapses into a slumped pose on the ground. Vulkan: the
   GPU multi-pass collapse (the JT3 driver) → **memcmp vs the CPU `StepArticulatedContactsSteps`**. Metal: the CPU
   reference. Render the collapsed integer state (2D side view: bodies as discs, the skeleton edges as segments).
   Golden = `tests/golden/metal/joint_ragdoll.png` (Mac-baked by the CONTROLLER — DO NOT commit). SMALL/FAST.
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU ragdoll body world after K steps == the CPU reference byte-for-byte.
     Print `joint-ragdoll: {bones:<N>, joints:<J>, limits:<L>, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `joint-ragdoll determinism: two runs BYTE-IDENTICAL`.
   - **(3) the ragdoll collapsed, bones held:** after settling, the bones are connected (max anchor gap within
     band) AND the ragdoll collapsed (its mean body pos.y dropped from the bind pose, came to rest at/above
     ground). Print `joint-ragdoll collapse: {maxAnchorGap:<G> within band, slumped:true, rested:true}`.
   - **(4) palette provenance:** `PoseToPalette` rebuilt from the settled world == the showcase's palette
     byte-for-byte (the float palette is a pure function of the bit-exact body state), and the count == the joint
     count. Print `joint-ragdoll palette: {entries:<N>} == rebuild from bit-exact pose`.
   - **Golden discipline: ONLY `tests/golden/metal/joint_ragdoll.png`; do NOT commit it.** Existing 162 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** the collapsed ragdoll render is Vulkan == Metal CPU-ref == golden,
   ZERO differing pixels (the SIM is integer-bit-exact; the bind/palette host floats are deterministic + don't
   feed the golden's integer body positions).
6. **Tests `tests/joint_test.cpp` additions (pure CPU):** `RagdollFromSkeleton` — the body count == joint count;
   each non-root edge gets a ball joint + a cone limit (counts == joint count − roots); the root's invMass
   matches `rootStatic`; a body's bind pos == the skeleton's bind global translation·worldScale (spot-check a
   known joint). `PoseToPalette` — count == joints; for an UNCOLLAPSED ragdoll at the bind pose, the palette ≈ the
   skeleton's bind palette (`bindGlobal·inverseBind`, within the float→fixed→float snap tolerance). The collapse
   is deterministic (two runs byte-identical). Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-joint-ragdoll` (features) + `--joint-ragdoll-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the JT3 surface). `rhi.h` + backend
  dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` + `engine/
  anim/` + `engine/physics/` + all existing shaders UNCHANGED. JT1/JT2/JT3 `joint.h` code + shaders UNCHANGED
  (JT4 additive — only the bind + the read-back + the showcase). **NO new shader.** Report the seam empty.
  (NOTE: `joint.h` will `#include "anim/skeleton.h"` + `"math/math.h"` read-only — confirm no circular include.)

## Out of scope (YAGNI — later JT slices)
Lockstep over the ragdoll (JT5), the lit SKINNED render driving an actual character mesh (JT6 — JT4 renders the
collapsed body discs + emits the palette; JT6 feeds the palette to the skinning pipeline). Convex/capsule contact
(sphere proxy), inertia tensor/torque (the fpx caveat). Active/physical-animation blending (pose-matching
springs). glTF skeleton loading (a synthetic skeleton). JT4 claims ONLY: a deterministic skeleton→ragdoll bind
whose collapse is bit-identical CPU↔Vulkan↔Metal + a provenance-checked pose palette, with the integer golden +
the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 101) + the new `joint_test` ragdoll cases. Clean under
   `windows-msvc-asan` (build+run `joint_test` + `introspect_test`).
2. **proofs + visual:** `--joint-ragdoll-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate
   → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image
   shows the ragdoll collapsed into a coherent slumped pose — bones connected, resting (pixel-check; the JT3
   lesson).** ALSO re-run `--joint-ball/hinge/step-shot` → still bit-exact (JT1-JT3 render-invariance).
3. Metal: `visual_test --joint-ragdoll` → new golden `tests/golden/metal/joint_ragdoll.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (JT4 reuses the
   JT3 driver — `hf_gen_msl` UNCHANGED).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `joint_ragdoll.png` added; the
   other 162 byte-identical. `git diff master --stat -- tests/golden` = ONLY `joint_ragdoll.png` (metal) + the
   introspect json.
5. Introspect JSON rebaked exactly `+deterministic-joint-ragdoll` + `--joint-ragdoll-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/`fract.h` +
   `engine/anim/` + `engine/physics/` + JT1/JT2/JT3 `joint.h`/shaders byte-unchanged). `scripts/verify.ps1`
   updated: `joint_ragdoll` golden in the Mac loop + `--joint-ragdoll-shot` in `$vkShots`. **NO new entry in
   `hf_gen_msl`.**
