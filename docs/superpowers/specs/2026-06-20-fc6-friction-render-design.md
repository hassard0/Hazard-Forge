# Slice FC6 — Deterministic Contact Friction: THE LIT 3D RENDER CAPSTONE (the money-shot) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH + FINAL slice of FLAGSHIP #20
> (DETERMINISTIC TANGENTIAL CONTACT FRICTION, `hf::sim::fric`) — THE MONEY-SHOT that COMPLETES the flagship. FC1-FC5
> built + proved the deterministic, lockstep-replayable friction-locked contact sim. FC6 RENDERS it as lit,
> shadowed 3D oriented boxes — the visual payoff — with the distinctive friction money-shot: **a box gripped at
> rest on a tilted ramp** (impossible in the frictionless solver, which would slide it off). The CX6 render-capstone
> twin: a render-only FLOAT helper through the EXISTING cube-mesh instanced-lit pipeline, ZERO new render shader,
> ZERO new RHI. FC1-FC5's `fric.h` code + CX1-CX6's `convex.h` are BYTE-FROZEN (FC6 is additive, render-only).
> Branch: `slice-fc6`. See [[hazard-forge-fric-roadmap]].

**Goal:** Extend `engine/sim/fric.h` (additive — FC1-FC5 + convex.h byte-unchanged) with a thin render bridge
`FrictionToRenderInstances(world)` (delegates to the frozen `convex::ConvexToRenderInstances` — namespace symmetry +
the provenance hook) and a friction render scene. Add `--fric-render-shot` (Vulkan) / `--fric-render` (Metal). Bake
the FLOAT golden `fric_render`. **NO new render shader, NO new RHI.**

## Design call: the friction-settled scene as lit, shadowed, oriented 3D cubes (the CX6 path, reused)

The friction world is a `convex::ConvexWorld`, and CX6's `convex::ConvexToRenderInstances(world)` (frozen,
`convex.h:1189`) already turns a `ConvexWorld` into the split FLOAT instance set (static floor cubes + dynamic box
cubes, each `translate(FxToFloat(pos))·rotate(orient)·scale(2·halfExtents)`) for the cube-mesh instanced-lit
pipeline. So FC6 RENDERS the friction-settled world by REUSING that helper verbatim:
- **`fric::FrictionToRenderInstances(world)`** — a one-line delegate to `convex::ConvexToRenderInstances(world)`
  (the same `ConvexRenderInstances` split: floor vs dynamic boxes). Pure function of the bit-exact friction world
  (two calls byte-equal — the provenance contract). Provided in `fric.h` for namespace symmetry + the test/showcase
  to call from the `fric::` namespace; the actual matrices come from the frozen CX6 helper.
- **The scene:** settle the FC4 friction RAMP scene (a tilted static ramp box + a dynamic box) AND/OR the friction
  STACK scene via `StepFrictionWorldN` ~K ticks, then render. The DISTINCTIVE money-shot is the **box gripped at
  rest on the tilted ramp** — a lit 3D box sitting on an incline, held by Coulomb friction (the frictionless solver
  would have slid it off). Optionally compose the standing tower too. Render via the EXISTING cube mesh +
  `lit_instanced.vert` + `lit.frag` + `shadow_instanced.vert` REUSED VERBATIM (the CX6 / `--convex-render` path).
  **NO new shader.** MATTE (metallic 0, roughness 1, warm-amber dynamic boxes on a cool-grey ramp/floor) to DODGE
  THE GF6/BD6/CX6 IRIDESCENCE TRAP.

**FLOAT capstone (the render-bar, NOT integer-strict):** the model matrices cross `FxToFloat`, so the bar is the
CX6/FPX6 render-capstone bar:
- Metal TWO-RUN determinism: `--fric-render` twice → DIFF 0.0000.
- PROVENANCE: the instance count == the body count; the instances are a pure rebuild of the bit-exact
  friction-settled world.
- VISUAL PARITY + IN-BAND cross-vendor: the controller's Windows-Vulkan vs Mac-Metal compare is a low in-band mean
  (~20-50/channel, the documented FLOAT-render band — NOT zero).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **CX6 `convex::ConvexToRenderInstances` (`convex.h:1189`, frozen — REUSE, do NOT edit):** the `ConvexRenderInstances`
  split + the `ConvexBoxShapeTransform` (`convex.h:1172`, `FxBodyTransform`·scale per body). FC6's
  `FrictionToRenderInstances` delegates to it.
- **FC4 `engine/sim/fric.h` (read it; APPEND only after `RunFricRollback`):** `StepFrictionWorld`/
  `StepFrictionWorldN`, `FrictionStepConfig`, `ConvexWorld`. FC1-FC5 lines byte-frozen.
- **The render-capstone showcase precedent:** CX6's `--convex-render-shot` (Vulkan) / `--convex-render` (Metal) in
  `samples/hello_triangle/main.cpp` + `metal_headless/visual_test.mm` — the cube-mesh instance buffers + the lit +
  shadow draws + the matte material + the provenance/determinism/shaded proofs. Mirror it for `--fric-render`,
  feeding the friction-settled world.
- **The byte-frozen render shaders:** `shaders/lit_instanced.vert` + `shaders/lit.frag` + `shaders/
  shadow_instanced.vert` — REUSED VERBATIM (confirm UNCHANGED in the seam). NO new render shader.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/fric_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/fric.h`** (FC1-FC5 byte-frozen): `FrictionToRenderInstances(world)` (the thin delegate to
   `convex::ConvexToRenderInstances`). FLOAT, render-only. **NO new shader, NO new RHI** (the cube mesh +
   lit_instanced/shadow_instanced reused).
2. **Showcase `--fric-render-shot <out>` (Vulkan) AND `--fric-render` (Metal) — WIRE BOTH** (standalone arg-parse).
   The SCENE: the FC4 friction ramp scene (a box gripped on a tilted ramp) settled K ticks → `FrictionToRenderInstances`
   → the cube-mesh instanced lit + shadow draws (the CX6 path), MATTE warm-amber boxes on a cool-grey ramp/floor +
   a ground shadow. Golden = `tests/golden/metal/fric_render.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance:** the instance count == the body count AND the instances == a recomputed
     `FrictionToRenderInstances(world)`. Print `fric-render: {bodies:<N>, instances:<N>, dynamic:<D>} from bit-exact
     friction-settled world`; assert.
   - **(2) determinism:** two runs → identical. Print `fric-render determinism: two runs BYTE-IDENTICAL`.
   - **(3) shaded:** a non-trivial non-black pixel count (the boxes actually rendered). Print `fric-render shaded:
     {nonBlackPixels:<P>}`; assert P above a floor.
   - **Golden discipline: ONLY `tests/golden/metal/fric_render.png`; do NOT commit it.** Existing 195 image goldens
     UNTOUCHED. (FLOAT capstone — the cross-vendor compare is the controller's in-band visual-parity check.)
4. **Cross-backend bar (FLOAT, the render-capstone bar):** Metal two-run DIFF 0.0000 + provenance + visual parity +
   in-band cross-vendor mean (~20-50, NOT zero). The controller does the cross-vendor compare.
5. **Tests — APPEND to `tests/fric_test.cpp` (pure CPU):** `FrictionToRenderInstances` of a settled friction world →
   the instance count == body count; each instance's translation == `convex::ConvexBoxShapeTransform`/
   `FxBodyTransform` translation (provenance); the dynamic/static split is correct; two calls byte-identical. Clean
   under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-friction-render` (features) + `--fric-render-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None — and NO new shader.** The cube mesh + `lit_instanced.vert` + `lit.frag` + `shadow_instanced.vert` are
  REUSED VERBATIM (the CX6 cube path). `rhi.h` + backend dirs UNCHANGED. `engine/sim/convex.h` + `fpx.h` +
  **FC1-FC5's fric.h code + ALL fric shaders** + all other sim headers + `engine/nav/` + `engine/anim/` +
  `engine/physics/` + ALL existing shaders UNCHANGED. `fric.h` APPEND-only (render-only float helper). Report the
  seam empty (only the convex... fric.h APPEND + the showcase/test/introspect are new/changed; NO shaders/ change).

## Out of scope (YAGNI)
This is the flagship FINALE — after FC6 the flagship is COMPLETE. No further friction slices. All the documented
FC1-FC5 caveats remain (boxes only, isotropic single-μ, coupled-iteration μ approx, within-a-band stability). FC6
claims ONLY: the bit-exact friction-settled scene rendered as lit, shadowed, oriented 3D boxes through the existing
instanced pipeline (ZERO new render shader/RHI), with the FLOAT render-capstone proofs. NOTE: FLOAT render → the
in-band cross-vendor bar, NOT the integer zero-diff bar.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 106 incl. FC1-FC5's `fric_test` + the appended FC6 cases).
   Clean under `windows-msvc-asan` (build+run `fric_test` + `introspect_test`).
2. **proofs + visual:** `--fric-render-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID. **VERIFY the image shows a coherent LIT, SHADOWED 3D MATTE box GRIPPED on a tilted ramp (the friction
   money-shot — NOT iridescent blue, NOT a blank/scrambled frame, NOT slid off).**
3. Metal: `visual_test --fric-render` → new golden `tests/golden/metal/fric_render.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm NO new shader added.** Controller does the in-band cross-vendor
   compare (~20-50 mean).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fric_render.png` added; the other
   195 byte-identical. `git diff master --stat -- tests/golden` = ONLY `fric_render.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-friction-render` + `--fric-render-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/convex.h`/`fpx.h` + **FC1-FC5's fric.h code + ALL fric shaders**
   + `lit_instanced.vert` + `lit.frag` + `shadow_instanced.vert` + ALL other sim headers + `engine/nav/` +
   `engine/anim/` + `engine/physics/` + ALL existing shaders byte-unchanged; **NO shaders/ change at all**).
   `scripts/verify.ps1` updated: `fric_render` golden in the Mac loop + `--fric-render-shot` in `$vkShots`.
