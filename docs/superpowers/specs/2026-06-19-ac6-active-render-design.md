# Slice AC6 — Deterministic Active Ragdoll: LIT 3D SKINNED RENDER CAPSTONE (THE MONEY-SHOT) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH + FINAL slice of FLAGSHIP #17
> (DETERMINISTIC ACTIVE RAGDOLL / PHYSICAL-ANIMATION BLENDING, `hf::sim::active`) — the LIT 3D SKINNED RENDER
> CAPSTONE, the money-shot, **JT6 VERBATIM** with the palette source swapped. AC1-AC5 built + proved a
> deterministic, blendable, hit-reacting, lockstep-replayable active ragdoll. AC6 RENDERS it: the bit-exact
> active-ragdoll pose (a character TRACKING an animation clip via physics torques) drives the EXISTING Fox skinned
> lit render — the physics moat poses the animation mesh, and the Fox is held in an animated pose *by physics*
> (not collapsed limp like JT6, not a pure anim clip like `--skinning-shot`). **FLOAT render-only** (the sim stays
> bit-exact integer; only the palette read-back crosses to float). **NO new shader, NO new RHI.** Branch:
> `slice-ac6`. See [[hazard-forge-active-roadmap]].

**Goal:** Extend `engine/sim/active.h` (additive — AC1-AC5 byte-frozen) with a thin `ActiveToPalette(skeleton,
active)` alias (= `joint::PoseToPalette(skeleton, active.ragdoll.world)` — the JT6 `RagdollToPalette` precedent). Add
`--active-render-shot` (Vulkan) / `--active-render` (Metal) that bind the Fox to an `ActiveRagdoll`, drive it to
track a clip, and render the Fox skinned + lit through the EXISTING `lit_skinned`/`shadow_skinned` pipeline. Bake
the FLOAT golden `active_render`. **NO new shader, NO new RHI.**

## Design call: the active pose drives the EXISTING Fox skinned render (JT6 verbatim, palette source swapped)
JT6 (`--joint-render-shot`, main.cpp:2439) already proved the pattern: load the Fox skeleton → bind a ragdoll →
step it → `joint::PoseToPalette` → `dev.SetJointPalette` → render the Fox skinned + lit + shadowed through the
EXISTING `lit_skinned.vert` + `lit.frag` + `shadow_skinned.vert` pipelines (the `--skinning-shot` camera/light/
shadow/sky/post REUSED VERBATIM). AC6 is JT6 with ONE swap: the palette comes from an **ActiveRagdoll tracking a
clip** instead of a collapsing JT4 ragdoll:

1. Load the Fox skeleton + skinned mesh (the JT6/`--skinning-shot` setup, REUSED VERBATIM).
2. `active::ActiveFromSkeleton(foxSkeleton, ragdollCfg, stiffness)` (AC3 — bind the Fox to an ActiveRagdoll: the
   JT4 bodies/joints/limits + one drive per edge). Use a PINNED root + a firm stiffness so the Fox holds a
   coherent animated pose (the JT6 coherence discipline — NOT a limp collapse, NOT a shattered pose).
3. `active::StepActiveSteps(active, foxSkeleton, clip, dt, iters, steps, startTime)` (AC3 — drive the Fox to TRACK
   a clip; use one of the Fox's anims — Walk/Run/Survey — or a synthetic pose clip; settle to a held animated
   pose). The Fox is posed BY PHYSICS tracking the animation.
4. `active::ActiveToPalette(foxSkeleton, active)` → `dev.SetJointPalette` → render the Fox skinned + lit (JT6 path
   verbatim). The ONLY float crossing is the palette read-back (`PoseToPalette`, Q16.16→float, render-only,
   OUTSIDE the bit-exact loop — the JT4/JT6 precedent).

**FLOAT visresolve-bar (the JT6/FPX6/VH6 precedent):** the SIM (AC1-AC5) is bit-exact, the final raster/shade is
float → the golden is Metal-baked, the gate is Metal-determinism (two renders BYTE-IDENTICAL) + provenance (the
palette IS a pure function of the bit-exact active state) + visual parity (a coherent posed Fox); the
Vulkan-vs-Metal cross-vendor delta is the documented float skinned-render baseline (~24/channel, the JT6 number).
**NO new shader/RHI** — reuses `lit_skinned`/`shadow_skinned` + the offscreen render path. MATTE/lit as JT6 (the
existing Fox material; no iridescence risk — the Fox skinned render is already proven coherent by JT6/`--skinning`).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **AC1-AC5 (this branch's `active.h`, read-only — build on, DON'T modify):** `ActiveRagdoll`, `ActiveFromSkeleton`
  (AC3 bind), `StepActiveSteps` (AC3 clip-track). DO NOT modify the AC1-AC5 functions. AC6 APPENDS only the thin
  `ActiveToPalette` alias.
- **JT4/JT6 palette read-back (`engine/sim/joint.h`):** `PoseToPalette(skeleton, world)` (joint.h:630 — the
  Q16.16→float palette; `ActiveToPalette` = `PoseToPalette(skeleton, active.ragdoll.world)`), `RagdollToPalette`
  (the JT6 thin-alias precedent). **DO NOT modify joint.h.**
- **THE JT6 RENDER SHOWCASE (`samples/hello_triangle/main.cpp` `--joint-render-shot`, main.cpp:2439 + the render
  body):** the Fox load, the `lit_skinned`/`shadow_skinned` pipeline setup, `dev.SetJointPalette`, the camera/
  light/shadow/sky/post — COPY VERBATIM, swap ONLY the palette source (the ActiveRagdoll pose). The
  `--skinning-shot` / `--anim-fsm-shot` Fox setup (main.cpp ~8425-8915) is the underlying render path. The Metal
  `--joint-render` in `metal_headless/visual_test.mm` is the Metal precedent for `--active-render`.
- **The anim clip for the Fox (`engine/anim/animation.h` + the glTF load):** how `--skinning-shot`/`--anim-fsm-shot`
  load the Fox's `anim::Animation` (Walk/Run/Survey) via `LoadGltfScene`/`SampleLocalPose`. Reuse to get the clip
  AC6 tracks. (If a real Fox clip is heavy to wire, a synthetic pose clip on the Fox skeleton is acceptable —
  choose + document; the headline is "physics-posed skinned character", the clip identity is secondary.)
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), `tests/active_test.cpp`. Standalone arg-parse (the FR1 C1061 lesson).

## Design decisions (locked)
1. **`ActiveToPalette(const anim::Skeleton& skeleton, const ActiveRagdoll& active) -> std::vector<math::Mat4>`** =
   `joint::PoseToPalette(skeleton, active.ragdoll.world)` (the thin JT6 `RagdollToPalette` alias; the ONE float
   crossing, render-only). A pure function of the bit-exact active state.
2. **Showcase `--active-render-shot <out>` (Vulkan) AND `--active-render` (Metal) — WIRE BOTH** (standalone
   arg-parse). Load the Fox, `ActiveFromSkeleton` (pinned root, firm stiffness), `StepActiveSteps` to a held
   animated pose tracking a clip (the SAME scene both backends build → byte-identical integer state), `ActiveToPalette`
   → `SetJointPalette` → render the Fox skinned + lit + shadowed (JT6 path verbatim) from the JT6/skinning camera.
   Golden = `tests/golden/metal/active_render.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) palette from bit-exact active pose:** print `active-render: {bones:<N>, joints:<J>, palette:<P>} from
     bit-exact active pose` (P == the Fox joint count; the palette IS `ActiveToPalette` of the settled active world).
   - **(2) determinism:** two renders → BYTE-IDENTICAL (the Metal-determinism gate). Print `active-render
     determinism: two runs BYTE-IDENTICAL`.
   - **(3) physics posed the mesh:** the active palette != the bind-pose palette (the physics-tracked pose actually
     moved the mesh off bind). Print `active-render posed: active palette != bind palette (physics posed the mesh)`.
   - **Golden discipline: ONLY `tests/golden/metal/active_render.png`; do NOT commit it.** Existing 176 image
     goldens UNTOUCHED (incl AC1-AC5).
4. **Cross-backend bar (FLOAT, the visresolve precedent):** the SIM is bit-exact but the final raster/shade is
   float → the golden is Metal-baked; the gate is Metal-determinism (two renders BYTE-IDENTICAL) + provenance +
   visual parity (a coherent posed Fox). The Vulkan-vs-Metal cross-vendor delta is the documented float
   skinned-render baseline (~24/channel, the JT6 number) — the CONTROLLER confirms it is in-band, NOT strict zero.
5. **Tests `tests/active_test.cpp` additions (pure CPU):** `ActiveToPalette` returns one Mat4 per Fox joint; the
   palette == `joint::PoseToPalette(skeleton, active.ragdoll.world)` (the alias is exact); a settled active pose's
   palette != the bind-pose palette. (Use the Fox skeleton or a small synthetic skeleton for the test.) Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-active-render` (features) + `--active-render-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the EXISTING `lit_skinned`/`shadow_skinned` pipeline + the offscreen render path + `SetJointPalette`
  VERBATIM. `rhi.h` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` +
  `couple*.h` + `fract.h` + `vehicle.h` + `engine/anim/` + all existing shaders + `hf_gen_msl` UNCHANGED (AC6 reuses
  `lit_skinned.vert`/`lit.frag`/`shadow_skinned.vert` — NO new shader). AC1-AC5 `active.h` functions UNCHANGED (AC6
  additive — only the `ActiveToPalette` alias + the showcase). Report the seam empty.

## Out of scope (YAGNI — flagship #17 is COMPLETE after AC6)
A new character mesh / a real game-anim integration / the AC4 limp-recover episode in the render (AC6 renders a
held clip-tracked pose — the simplest coherent money-shot; the active→limp→recover dynamics are AC4's). Root motion,
cloth on the Fox, facial anim. AC6 claims ONLY: a lit 3D render of the Fox skinned + posed by the bit-exact active
ragdoll (physics tracking an animation), through the existing pipeline, Metal-deterministic + provenance-checked +
visually coherent, the cross-vendor float baseline in-band. **Completes the 6-slice deterministic-active-ragdoll /
physical-animation-blending flagship — the SEVENTEENTH flagship.**

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 103 + the new `active_test` palette cases). Clean under
   `windows-msvc-asan` (build+run `active_test` + `introspect_test`).
2. **proofs + visual:** `--active-render-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED — the skinned pipeline
   binds the per-frame shadow map). **VERIFY the image shows a coherent lit 3D Fox posed by physics — a recognizable
   skinned character in an animated pose, NOT collapsed/limp, NOT scrambled, NOT black (the HARD visual gate; the
   JT6 coherence lesson).** Re-run `--active-lockstep-shot` + `--active-recover-shot` + `--active-step-shot` +
   `--active-blend-shot` + `--active-drive-shot` → still bit-exact (AC1-AC5 render-invariance).
3. Metal: `visual_test --active-render` → new golden `tests/golden/metal/active_render.png`; **two runs DIFF
   0.0000** (the FLOAT-capstone gate is Metal-DETERMINISM). **Confirm `visual_test.mm` in the diff; confirm NO new
   shader (`hf_gen_msl` UNCHANGED).** The Vulkan-vs-Metal cross-vendor compare is the FLOAT skinned baseline
   (~24/channel) — the controller confirms it is in-band (NOT strict zero).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `active_render.png` added; the
   other 176 byte-identical. `git diff master --stat -- tests/golden` = ONLY `active_render.png` (metal) (the
   introspect json rebake = controller, post-gate).
5. Introspect: exactly `+deterministic-active-render` + `--active-render-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + the frozen sim/anim/physics headers + ALL existing shaders byte-unchanged; ONLY
   `active.h` extended additively + the showcase/test/introspect). `scripts/verify.ps1` updated: `active_render`
   golden + `--active-render-shot` in `$vkShots`. **NO new shader; `active` still NOT in `hf_gen_msl`.**
