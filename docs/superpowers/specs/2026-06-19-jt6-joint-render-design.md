# Slice JT6 — Deterministic Articulated-Body Ragdoll: LIT 3D SKINNED RENDER CAPSTONE (the money-shot) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice of FLAGSHIP #15
> (DETERMINISTIC ARTICULATED-BODY RAGDOLL, `hf::sim::joint`) — **COMPLETES THE FLAGSHIP (the 15th)**. THE
> PILLAR-BRIDGE MONEY-SHOT: the bit-exact ragdoll pose drives a SKINNED, lit CHARACTER through the EXISTING GPU
> skinning path — the physics moat poses the animation mesh. Ragdoll-ize the Fox skeleton, collapse it, feed the
> `PoseToPalette` output to the SAME `lit_skinned` pipeline the anim FSM uses, and the Fox flops as a ragdoll.
> The JT1-JT5 sim stays strict-integer/bit-exact; here — and ONLY here — the pose crosses to FLOAT (the
> documented FLOAT visresolve-bar, the FPX6/CG6/FR6 precedent). NO new shader, NO new RHI — reuse the existing
> skinned + instanced-lit pipelines VERBATIM. Branch: `slice-jt6`. See [[hazard-forge-joint-roadmap]].

**Goal:** The showcase wires `--joint-render-shot` (Vulkan) / `--joint-render` (Metal): load the Fox skeleton →
`RagdollFromSkeleton` → collapse → `PoseToPalette` → render the Fox SKINNED + lit + shadowed via the EXISTING
`lit_skinned` path (`SetJointPalette`), with the bone bodies optionally overlaid as instanced-lit spheres
(`FxBodyTransform`). `joint.h` gains at most a thin `RagdollToPalette` convenience (or reuses `PoseToPalette`
directly). Bake the FLOAT golden `joint_render`. **NO new shader, NO new RHI.**

## Design call: the ragdoll pose drives the EXISTING Fox skinned render, FLOAT visresolve-bar
The existing `--skinning`/`--anim-fsm` showcase renders the Fox (`fox.skeleton`, a glTF skinned character)
through the `lit_skinned`/`shadow_skinned` pipelines, feeding a joint palette via `dev.SetJointPalette(...)`
(computed by `anim::SampleAnimation`/`BlendAnimations`/`sm.Evaluate`). JT6 swaps the palette SOURCE: instead of
an animation clip, the palette comes from the **collapsed RAGDOLL** — `RagdollFromSkeleton(fox.skeleton, cfg)` →
`StepArticulatedContactsSteps` (the bit-exact JT3 collapse) → `PoseToPalette(fox.skeleton, ragdoll.world)` → the
SAME `SetJointPalette` → the SAME `lit_skinned` draw. The Fox is now POSED BY PHYSICS — it flops/slumps as a
ragdoll. Everything else (camera, lights, shadow, sky, post) is the existing skinned-showcase setup reused
VERBATIM. The ONLY float crossing is `PoseToPalette` (the FPX6 `FxBodyTransform` per bone → `global·inverseBind`,
JT4's read-back). The JT1-JT5 sim is untouched + bit-exact. **NO new shader, NO new RHI.**

**THE COHERENCE REQUIREMENT (heed the GF6/FR6 lesson):** the collapsed Fox must read as a RECOGNIZABLE,
coherent ragdoll-posed fox — slumped/draped, NOT exploded or mangled. Tune for this: a SHORT settle (the fox
sags from its bind pose, doesn't shatter apart), tight-ish cone limits (joints don't hyperextend), and a
PINNED or lightly-anchored root (the fox drapes from a hip/spine anchor rather than scattering). If a free-root
full collapse mangles the mesh, prefer a pinned-root short settle that keeps the fox legible. **VIEW IT and
iterate** until it's a coherent posed character.

**THE FALLBACK (documented, only if the skinned drive is unworkable):** if the Fox skeleton's structure makes a
coherent skinned ragdoll infeasible within the slice, fall back to rendering the ragdoll BONES as instanced-lit
spheres (`FractToRenderInstances`-style via `fpx::FxBodyTransform`, the FPX6/FR6 path VERBATIM) — a lit 3D bone
view — and keep the palette-provenance proof to demonstrate the bridge. The skinned Fox is STRONGLY preferred
(it IS the flagship's unique payoff); the bone-sphere render is the safety net. Document whichever ships.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The existing skinned-character render to REUSE VERBATIM (`metal_headless/visual_test.mm:239-490` the
  `--skinning`/`--anim-fsm` showcase + the matching Vulkan `--skinning-shot`/`--anim-fsm-shot` in
  `samples/hello_triangle/main.cpp`):** the Fox model load (`fox.skeleton` + the skinned mesh), the `lit_skinned`
  + `shadow_skinned` pipelines, `dev.SetJointPalette(paletteData.data(), …)`, the camera/light/shadow/sky/post
  setup. JT6 copies this path and swaps the palette to the ragdoll's. Read how `paletteData` is laid out (the
  float matrix array `SetJointPalette` expects) so `PoseToPalette`'s `std::vector<math::Mat4>` flattens to the
  SAME layout.
- **The ragdoll (this branch's `joint.h`, read-only — DON'T modify):** `RagdollFromSkeleton`/`Ragdoll`/
  `RagdollConfig`, `StepArticulatedContactsSteps`, `PoseToPalette(skeleton, world)` (the palette read-back — the
  ONE float crossing), `MeasureRagdoll`. `fpx::FxBodyTransform` (for the optional bone-sphere overlay).
- **The instanced-lit bone render (fallback) — `engine/sim/fract.h::FractToRenderInstances` + the `--fract-render`
  showcase:** the `FxBodyTransform`-per-body → instanced-lit pipeline (the FPX6/FR6/CG6 path). JT6's bone overlay
  / fallback reuses this VERBATIM.
- **The anim skeleton + palette math (`engine/anim/skeleton.h` + `animation.cpp`):** `Skeleton`/`Joint`,
  `SampleAnimation` (the `global·inverseBind` palette the skinned pipeline consumes — `PoseToPalette` already
  matches this). DO NOT modify anim.
- **Showcase + registration:** JT1-JT5's `--joint-*-shot` plumbing — **standalone arg-parse loop** (the FR1 C1061
  lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the
  introspect JSON golden**), `tests/joint_test.cpp`.

## Design decisions (locked)
1. **The showcase `--joint-render-shot <out>` (Vulkan) AND `--joint-render` (Metal) — WIRE BOTH** (standalone
   arg-parse). Load the Fox skeleton → `RagdollFromSkeleton` → `StepArticulatedContactsSteps` K steps (short,
   coherent collapse) → `PoseToPalette` → `SetJointPalette` → render the Fox skinned + lit + shadowed (the
   EXISTING `lit_skinned` path). Golden = `tests/golden/metal/joint_render.png` (Mac-baked by the CONTROLLER — DO
   NOT commit). `joint.h` adds at most a `RagdollToPalette` alias (or none — `PoseToPalette` suffices); keep the
   header change minimal/additive.
2. **PROOFS (fail loudly; exact lines):**
   - **(1) palette provenance:** the palette fed to the skinning pipeline == `PoseToPalette` rebuilt from the
     settled ragdoll world (the float palette is a pure function of the bit-exact body state), count == the Fox
     joint count. Print `joint-render: {bones:<N>, joints:<J>, palette:<P>} from bit-exact ragdoll pose`.
   - **(2) determinism:** two renders → byte-identical (Metal). Print `joint-render determinism: two runs
     BYTE-IDENTICAL`.
   - **(3) the pose IS the ragdoll (not the bind/anim):** the rendered palette DIFFERS from the bind-pose palette
     (the ragdoll actually posed the fox — it collapsed). Print `joint-render posed: ragdoll palette != bind
     palette (physics posed the mesh)`.
   - **Golden discipline: ONLY `tests/golden/metal/joint_render.png`; do NOT commit it.** Existing 164 image
     goldens UNTOUCHED.
3. **Cross-backend bar (FLOAT visresolve-bar, NOT integer):** Metal two-run DIFF 0.0000 (gate on `compare.sh`) +
   provenance (the palette derives from the bit-exact JT1-JT5 ragdoll state) + visual parity + cross-vendor mean
   ~the engine float-render baseline (document the actual mean; the skinned pipeline's existing `skinning` golden
   is the reference — NOT held to the integer zero-diff bar — the FPX6/GR6/GF6/FR6 precedent).
4. **Tests `tests/joint_test.cpp` additions (pure CPU):** `PoseToPalette` on a collapsed ragdoll != the
   bind-pose palette (the collapse changed the pose); the palette count == joint count; the palette is
   deterministic (two collapses → identical palette). (The render itself is GPU/visual — proven by the golden +
   provenance, not a unit test.) Clean under `windows-msvc-asan`.
5. **Introspect.** Add exactly `deterministic-joint-render` (features) + `--joint-render-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the EXISTING `lit_skinned`/`shadow_skinned` + instanced-lit pipelines + `SetJointPalette` +
  the offscreen render path (the `--skinning`/`--anim-fsm`/`--fract-render` surface). `rhi.h` + backend dirs
  UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` + `engine/anim/` +
  `engine/physics/` + all existing shaders UNCHANGED. JT1-JT5 `joint.h` code + shaders UNCHANGED (JT6 additive —
  only the showcase + at most a thin palette alias). **NO new shader.** Report the seam empty.

## Out of scope (YAGNI)
Active/physical-animation blending (pose-matching springs that blend ragdoll toward an anim clip), capsule
collision, inertia tensor (the fpx caveat). Authoring a NEW skinned mesh (reuse the Fox). JT6 claims ONLY: a
deterministic lit 3D render of a SKINNED character posed by the bit-exact ragdoll (or, fallback, the ragdoll
bones as lit spheres), Metal-deterministic + provenance + cross-vendor visual parity, with the float golden +
the three proofs. **JT6 COMPLETES FLAGSHIP #15 — the FIFTEENTH flagship.**

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 101) + the new `joint_test` palette cases. Clean under
   `windows-msvc-asan` (build+run `joint_test` + `introspect_test`).
2. **proofs + visual:** `--joint-render-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate
   → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image is
   a coherent lit 3D character POSED AS A RAGDOLL — a recognizable slumped/draped fox (or, fallback, coherent lit
   bone spheres), lit/shaded, NOT mangled/exploded (pixel-check; the GF6/FR6 lesson — iterate the collapse tuning
   until coherent).** ALSO re-run `--joint-ball/hinge/step/ragdoll/lockstep-shot` → still bit-exact (JT1-JT5
   render-invariance).
3. Metal: `visual_test --joint-render` → new golden `tests/golden/metal/joint_render.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (JT6 reuses the
   skinned/instanced pipelines — `hf_gen_msl` UNCHANGED).** Cross-vendor = the FLOAT visresolve-bar (document the
   mean; NOT integer zero).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `joint_render.png` added; the
   other 164 byte-identical. `git diff master --stat -- tests/golden` = ONLY `joint_render.png` (metal) + the
   introspect json.
5. Introspect JSON rebaked exactly `+deterministic-joint-render` + `--joint-render-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/`fract.h` +
   `engine/anim/` + `engine/physics/` + JT1-JT5 `joint.h`/shaders byte-unchanged). `scripts/verify.ps1` updated:
   `joint_render` golden in the Mac loop + `--joint-render-shot` in `$vkShots`. **NO new entry in `hf_gen_msl`.**
