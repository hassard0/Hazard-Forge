# Slice CX6 — Deterministic Convex Contacts: THE LIT 3D INSTANCED RENDER CAPSTONE (the money-shot) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH + FINAL slice of FLAGSHIP #19
> (DETERMINISTIC CONVEX RIGID-BODY CONTACTS, `hf::sim::convex`) — THE MONEY-SHOT that COMPLETES the flagship. CX1-CX5
> built + proved the deterministic, lockstep-replayable box-box contact sim (SAT → manifold → angular impulse →
> settling stack → lockstep/rollback). CX6 RENDERS the settled stack as lit, shadowed 3D ORIENTED BOXES — the
> visual payoff. The FR6/CP6/CG6/GF6/BD6/VH6 render-capstone twin: a render-only FLOAT helper through the EXISTING
> instanced-lit pipeline, ZERO new render shader, ZERO new RHI. CX1-CX5's `convex.h` code is BYTE-FROZEN (CX6 is
> additive, render-only). Branch: `slice-cx6`. See [[hazard-forge-convex-roadmap]].

**Goal:** Extend `engine/sim/convex.h` (additive — CX1-CX5 byte-unchanged) with a render-ONLY FLOAT helper
`ConvexToRenderInstances(world)` → a set of per-body `math::Mat4` model matrices (one oriented CUBE per box), drawn
through the EXISTING cube-mesh + `lit_instanced.vert` + `lit.frag` + `shadow_instanced.vert` (the VH6 cube path,
byte-frozen). Add `--convex-render-shot` (Vulkan) / `--convex-render` (Metal). Bake the FLOAT golden `convex_render`.
**NO new render shader, NO new RHI.**

## Design call: the settled box stack as lit, shadowed, oriented 3D cubes (the VH6 cube-mesh path)

VH6 already renders a CUBE mesh through the instanced-lit pipeline (the chassis: `VehicleToRenderInstances` → a
`math::Mat4` per body → the cube mesh + `lit_instanced.vert`/`lit.frag` + `shadow_instanced.vert`). CX6 reuses that
EXACT path for the convex stack:
- **`ConvexToRenderInstances(world)`** — a PURE FUNCTION of the (bit-exact CX1-CX5) `ConvexWorld`, run OUTSIDE the
  integer loop (FLOAT, render-only — the provenance discipline). For each body `i`: the column-major
  `math::Mat4 model = translate(FxToFloat(body.pos)) · rotate(quatToMat(body.orient)) · scale(2·FxToFloat(
  box.halfExtents))` — i.e. `fpx::FxBodyTransform(body)` for the translate+rotate (the VH6/FR6 provenance call) times
  the per-box non-uniform scale (full extents = 2·halfExtents) so a unit cube becomes the oriented box. The
  per-instance ORIENT makes the box ROTATION visible (the convex money-shot — boxes you can SEE are tilted, unlike
  the prior sphere capstones). Split the output: the STATIC floor (a distinct cool-grey, large) vs the DYNAMIC boxes
  (warm matte amber) — two instance sets OR one set + a per-instance colour/material index, the VH6 split-set shape.
- **Render** through the EXISTING cube mesh + `lit_instanced.vert` + `lit.frag` + `shadow_instanced.vert` REUSED
  VERBATIM (the `--vehicle-render`/`--fpx-render` cube path). **NO new shader.** MATTE (metallic 0, roughness 1,
  warm-amber dynamic boxes on a cool-grey floor) to **DODGE THE GF6/BD6/FR6/JT6 IRIDESCENCE TRAP** (a metallic/low-
  roughness material reads as blue iridescence — the documented hard lesson; stay matte).
- The SCENE: the CX4 settling-stack world (static floor + 3 dynamic boxes), run `StepConvexWorldN` ~K ticks to
  SETTLE (the CX4 resting tower), then `ConvexToRenderInstances` → render. A lit, shadowed 3D box tower.

**FLOAT capstone (the render-bar, NOT integer-strict):** the render is FLOAT (the model matrices cross FxToFloat),
so the bar is the FR6/CP6/BD6/VH6 render-capstone bar, NOT the integer zero-diff bar:
- Metal TWO-RUN determinism: `--convex-render` twice → DIFF 0.0000 (the Metal pipeline is itself deterministic).
- PROVENANCE: the instance count == the body count; the instances are a pure rebuild of the bit-exact world
  (instances == `ConvexToRenderInstances(world)` recomputed == identical).
- VISUAL PARITY + IN-BAND cross-vendor: the controller's Windows-Vulkan vs Mac-Metal compare is a low in-band mean
  (~22-55/channel, the documented FLOAT-render band — NOT zero; float rasterization differs sub-pixel cross-vendor).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The cube-mesh render-capstone precedent (`engine/sim/vehicle.h` VH6, ~line 982-1036):** `VehicleRenderInstances`
  + `VehicleToRenderInstances` — the split FLOAT instance set (chassis = cube mesh, warm matte), the
  `fpx::FxBodyTransform` provenance call, the OUTSIDE-the-bit-loop discipline, the matte-to-dodge-iridescence note.
  CX6's `ConvexToRenderInstances` is the DIRECT twin (all bodies = cubes). Also `engine/sim/fract.h` FR6 +
  `engine/sim/boids.h` BD6 (the render-helper shape + the FLOAT visresolve bar).
- **`fpx::FxBodyTransform` (engine/sim/fpx.h):** the FxBody → float model matrix (translate + rotate by the Q16.16
  FxQuat). CONFIRM its exact signature + that it gives the translate+rotate (CX6 multiplies the per-box scale).
- **`math::Mat4` + `scene::InstanceData` (the column-major model matrix the instanced pipeline consumes):** confirm
  the layout from VH6's usage. The cube mesh + the instanced draw wiring: study the `--vehicle-render-shot` block in
  `samples/hello_triangle/main.cpp` (the cube-mesh instance buffers + the lit + shadow draws) + `visual_test.mm`.
- **The byte-frozen render shaders:** `shaders/lit_instanced.vert` + `shaders/lit.frag` + `shaders/
  shadow_instanced.vert` — REUSED VERBATIM (confirm UNCHANGED in the seam). NO new render shader.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/convex_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/convex.h`** (CX1-CX5 byte-frozen): `ConvexRenderInstances` (the split FLOAT instance set —
   static floor vs dynamic boxes, the VH6 shape) + `ConvexToRenderInstances(world)` (the pure-function render-only
   builder — `FxBodyTransform` · per-box scale per body, OUTSIDE the integer loop). FLOAT, render-only. **NO new
   shader, NO new RHI** (the seam incl. shaders — EMPTY; the cube mesh + lit_instanced/shadow_instanced are reused).
2. **Showcase `--convex-render-shot <out>` (Vulkan) AND `--convex-render` (Metal) — WIRE BOTH** (standalone
   arg-parse). The SCENE: the CX4 settling stack settled K ticks → `ConvexToRenderInstances` → the cube-mesh
   instanced lit + shadow draws (the VH6 path). MATTE warm-amber dynamic boxes on a cool-grey floor + a ground
   shadow. Golden = `tests/golden/metal/convex_render.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance:** the instance count == the body count AND the instances == a recomputed
     `ConvexToRenderInstances(world)` (the render is a pure function of the bit-exact world). Print `convex-render:
     {bodies:<N>, instances:<N>, dynamic:<D>} from bit-exact settled stack`; assert count + rebuild equality.
   - **(2) determinism:** two runs → identical (the same instance set + the same image bytes on a given backend).
     Print `convex-render determinism: two runs BYTE-IDENTICAL`.
   - **(3) shaded:** the render touched a non-trivial pixel count (the boxes actually rendered, not a blank frame).
     Print `convex-render shaded: {nonBlackPixels:<P>}`; assert P above a floor.
   - **Golden discipline: ONLY `tests/golden/metal/convex_render.png`; do NOT commit it.** Existing 188 image
     goldens UNTOUCHED. (FLOAT capstone — the cross-vendor compare is the controller's in-band visual-parity check,
     NOT a committed strict-zero.)
4. **Cross-backend bar (FLOAT, the render-capstone bar):** Metal two-run DIFF 0.0000 + provenance + visual parity +
   in-band cross-vendor mean (~22-55, NOT zero). The controller does the cross-vendor compare.
5. **Tests — APPEND to `tests/convex_test.cpp` (pure CPU):** `ConvexToRenderInstances` of a known settled world →
   the instance count == body count; each instance's translation == `FxBodyTransform(body)` translation (the
   provenance); the dynamic/static split is correct; two calls byte-identical (pure function). Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-convex-render` (features) + `--convex-render-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None — and NO new shader.** The cube mesh + `lit_instanced.vert` + `lit.frag` + `shadow_instanced.vert` are
  REUSED VERBATIM (the VH6 cube path). `rhi.h` + backend dirs UNCHANGED. `engine/sim/fpx.h` + all sibling sim
  headers + **CX1-CX5's convex.h code + ALL convex shaders (convex_sat/manifold/solve/step.comp)** + `engine/nav/`
  + `engine/anim/` + `engine/physics/` + ALL existing shaders UNCHANGED. `convex.h` APPEND-only (render-only float
  helper). Report the seam empty (only the convex.h APPEND + the showcase/test/introspect are new/changed; NO
  shaders/ change at all).

## Out of scope (YAGNI)
This is the flagship FINALE — after CX6 the flagship is COMPLETE. No further convex slices. Friction, general convex
hulls, the area-maximizing manifold reduction, the FPX2 broadphase, multi-thread solve — all the documented CX1-CX5
caveats remain (deferred refinements, NOT CX6's concern). CX6 claims ONLY: the bit-exact settled convex stack
rendered as lit, shadowed, oriented 3D boxes through the existing instanced pipeline (ZERO new render shader/RHI),
with the FLOAT render-capstone proofs (provenance + determinism + visual parity + in-band cross-vendor). NOTE: FLOAT
render (the model matrices cross FxToFloat) → the in-band cross-vendor bar, NOT the integer zero-diff bar (the
documented render-capstone proof strength).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 105 incl. CX1-CX5's `convex_test` + the appended CX6 cases).
   Clean under `windows-msvc-asan` (build+run `convex_test` + `introspect_test`).
2. **proofs + visual:** `--convex-render-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows a
   coherent LIT, SHADOWED 3D BOX STACK (a recognizable tower of oriented matte boxes on the floor, NOT iridescent
   blue, NOT a blank/scrambled frame).**
3. Metal: `visual_test --convex-render` → new golden `tests/golden/metal/convex_render.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm NO new shader added.** Controller does the in-band cross-vendor
   compare (~22-55 mean).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `convex_render.png` added; the other
   188 byte-identical. `git diff master --stat -- tests/golden` = ONLY `convex_render.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-convex-render` + `--convex-render-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h` + all sibling sim headers + **CX1-CX5's convex.h code +
   ALL convex shaders** + `lit_instanced.vert` + `lit.frag` + `shadow_instanced.vert` + `engine/nav/` +
   `engine/anim/` + `engine/physics/` + ALL existing shaders byte-unchanged; **NO shaders/ change at all**).
   `scripts/verify.ps1` updated: `convex_render` golden in the Mac loop + `--convex-render-shot` in `$vkShots`.
