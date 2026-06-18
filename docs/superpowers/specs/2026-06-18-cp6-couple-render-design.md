# Slice CP6 — Deterministic Rigid↔Fluid Coupling: LIT 3D RENDER CAPSTONE (the money-shot, COMPLETES flagship #11) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice of FLAGSHIP #11
> (DETERMINISTIC TWO-WAY RIGID↔FLUID COUPLING, `hf::sim::couple`). RENDERS the bit-exact coupled state as a
> lit 3D scene — the **rigid body floating among the fluid** as lit 3D INSTANCED SPHERES (the barrel + the
> water droplets) — the money-shot showing the deterministic fixed-point COUPLED sim as a real lit 3D scene.
> The instance transforms come from the BIT-EXACT integer state (`fpx::FxBodyTransform` for the body +
> `fluid::FluidToRenderInstances` for the droplets; the sim stays integer). The raster/shade is float → the
> FLOAT visresolve-bar (like FPX6/FL6/GR6): Metal-baked golden, Metal-determinism DIFF 0.0000 + provenance +
> visual parity, cross-vendor ~the float baseline. Reuses the EXISTING instanced lit-sphere pipeline (FPX6/FL6
> path) — NO new RHI, NO new shader. After this slice flagship #11 is COMPLETE (6/6) — the ELEVENTH flagship.
> Branch: `slice-cp6`. See [[hazard-forge-couple-roadmap]].

**Goal:** Add render-only float helpers to `engine/sim/couple.h` (`CoupleToRenderInstances` → the combined
per-instance transform set: one big sphere per `fpx::FxBody` (the barrel, scaled by body radius, via
`FxBodyTransform`) + one small sphere per `fluid::FluidParticle` (a droplet at `pos/kOne`, scaled by the
droplet radius, via `FluidToRenderInstances`)), run the deterministic CP1–CP5 coupled sim (`StepCouple`) to a
settled/bobbing state, and render the body + fluid as lit 3D INSTANCED SPHERES through the EXISTING instanced
lit pipeline (the FPX6 `RunFpxRenderShowcase` / FL6 path) from a fixed 3/4 camera + directional light. Add
`--couple-render-shot` (Vulkan) / `--couple-render` (Metal). Bake the lit golden `couple_render`. Make-safe:
small host float helpers (CP1–CP5 sim UNCHANGED) + a NEW showcase + NEW golden; reuse the existing pipeline —
NO new shader/RHI.

## Design call: FLOAT visresolve-bar (the coupling arc's only float slice)
CP1–CP5 are strict integer/bit-exact. CP6 RASTERIZES the state with a perspective camera + lighting →
cross-vendor float (the FPX6/FL6 caveat). CP6's golden is the FLOAT bar, identical to FL6/GR6: the committed
golden is **baked on Metal**; the gate is **Metal-render == Metal-golden DIFF 0.0000** (deterministic,
two-run) + **provenance** (the rendered per-instance transforms are built from the bit-exact integer state —
the body via `FxBodyTransform`, the droplets via `pos/kOne`) + visual parity, with the Vulkan-vs-Metal delta
the documented cross-vendor smoke (~30–60 mean, the engine's float-render baseline — NOT zero). The fixed-
point SIM feeding the render is still bit-exact (CP1–CP5); only the final raster/shade is float.

## THE SCENE — a cleaner "barrel in water" money-shot (address the CP4 modest-viz + wall-leak notes)
CP4's coupled-step integer debug-viz was modest (a thin pool spread across the basin) with a few particles
tunnelling the static-particle walls (documented). CP6 is the capstone money-shot — **use a cleaner, denser
confined scene** so the lit 3D render reads as a barrel floating in water: a denser fluid pool (more particles
/ a tighter basin) settled to a coherent pool, the body floating at the waterline, run enough `StepCouple`
steps that it settles to a recognizable state (a barrel bobbing at the surface among the droplets). Optionally
drop the static wall particles from the RENDER (render only the dynamic fluid + the body) so the money-shot
isn't cluttered by wall markers — the wall particles are a sim containment detail, not part of the headline
(document the choice). If a perfectly clean pool proves finicky, a coherent settled pool + the floating body
is acceptable — the render is golden-verified for determinism+provenance regardless; but PREFER the clean
"barrel in water" read.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The coupled state (the input):** `engine/sim/couple.h` — `CoupleWorld` (bodies + particles),
  `StepCouple`/`StepCoupleSteps` (CP4 — run the coupled sim to a settled/bobbing state). The body world
  transform = `fpx::FxBodyTransform(body)` (`fpx.h:627`, the FPX6 render bridge — translate by `pos/kOne` +
  the orientation quaternion + scale by radius). The droplet transforms = `fluid::FluidToRenderInstances`
  (FL6 — one mat4 per particle at `pos/kOne`, scaled by the droplet radius).
- **The existing instanced lit-sphere pipeline to REUSE (find + mirror — do NOT invent):** FPX6's
  `RunFpxRenderShowcase` + the `--fpx-render-shot` / `--fluid-render-shot` / `--grain-render-shot` branches —
  the **instanced lit pipeline** (`lit_instanced.vert` + `lit.frag` + `scene::InstanceTransformLayout` per-
  instance mat4 + the FrameData camera/light UBO + sky/shadow/post, REUSED VERBATIM): one lit sphere instance
  per body. CP6 is the SAME — a COMBINED instance set (the body spheres + the fluid droplet spheres) drawn in
  ONE instanced draw (or two draws of the same pipeline). All instances share the pipeline/material; the body
  is distinguished by being a LARGE sphere among small droplets (NO per-instance color → no new shader). Report
  which pipeline/shader was reused. **(NAV6/CL6/FL6 lesson: VERIFY the rendered output actually shows the lit
  body + droplets — pixel-check the shaded region, do NOT trust the proof's "coherent" claim alone.)**
- **The float-golden discipline (the bar to copy):** FPX6 / FL6 / GR6 — per-backend Metal golden, determinism,
  provenance, the cross-vendor ~30–60 baseline (NOT zero-diff).
- **Showcase + registration:** the FL6/GR6 `--*-render-shot` template; `scripts/verify.ps1`,
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden** — the
  GR2 lesson). main.cpp has `/bigobj` (CL4).

## Design decisions (locked)
1. **Run the coupled sim to a compelling state, then render.** Run the deterministic CP1–CP5 pipeline (a
   denser confined pool + a body, `StepCoupleSteps` enough steps that the body floats at the waterline among
   the settled droplets). The body's + particles' final integer state is the bit-exact sim output. Build a
   per-instance `math::Mat4` per body (`FxBodyTransform`, scaled by body radius) + per fluid particle
   (`pos/kOne`, scaled by the droplet radius). Pure deterministic host float (render-only).
2. **Render through the EXISTING instanced lit-sphere pipeline — NO new RHI, NO new shader.** Reuse the
   instanced lit-sphere `GraphicsPipelineDesc` (FPX6/FL6's `lit_instanced.vert`+`lit.frag`) + the per-instance
   transform buffer + the frame/camera UBO; a unit `SphereGeometry`, the combined instance set (bodies +
   droplets), a fixed 3/4 camera + directional light + the ground. Report exactly which pipeline/shader was
   reused; if a genuinely-new shader is needed, flag it loudly (prefer reuse).
3. **Showcase `--couple-render-shot <out>` (Vulkan, main.cpp) AND `--couple-render` (Metal, visual_test.mm —
   WIRE BOTH; confirm `#include "sim/couple.h"`).** Run the deterministic coupled sim → per-body + per-particle
   instance transforms → instanced lit render → the BGRA8 image. Golden =
   `tests/golden/metal/couple_render.png` (baked on the Mac by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance (the coupled state IS bit-exact):** the per-instance transforms are built from the
     `CoupleWorld` state that equals the CPU `StepCoupleSteps` reference (the bit-exact CP1–CP5 sim — print the
     body + particle counts + that the transforms derive from `FxBodyTransform`/`pos/kOne`). Print `couple-
     render: {bodies:<B>, particles:<N>, instances:<I>, shaded:<S>} (fixed-point coupling -> lit 3D render)`.
   - **(2) determinism (same backend):** two renders → DIFF 0.0000. Print `couple-render determinism: two
     renders BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the lit body + droplets cover a coherent region (`shaded > 0`, a recognizable
     barrel-in-water). Print `couple-render coverage: <S> shaded (coherent lit barrel + fluid)`.
   - **(4) empty no-op:** zero bodies + zero particles → the cleared background / ground only. Print `couple-
     render empty: base only (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/couple_render.png`; do NOT commit it — the CONTROLLER bakes
     on the Mac.** Existing 140 image goldens UNTOUCHED.
5. **Cross-backend bar (FLOAT, per the FPX6/FL6 finding):** Metal-output == Metal-golden DIFF 0.0000
   (determinism) + provenance (the bit-exact coupled state) + VISUAL parity; the Vulkan-vs-Metal cross-vendor
   delta is the documented float baseline (~30–60 mean — like FPX6/FL6/GR6, NOT zero), measured by the
   controller (a LARGER/structural delta or a visual mismatch is a bug).
6. **Tests `tests/couple_test.cpp` additions (pure CPU):** `CoupleToRenderInstances` — a known coupled state →
   the expected combined instance transforms (the body's `FxBodyTransform` + the droplets' translate+scale,
   the right instance count B+N); the sim feeding it is the existing bit-exact `StepCouple` (already tested).
   Clean under `windows-msvc-asan`. (The render is golden-verified, not unit-tested.)
7. **Introspect.** Add exactly `deterministic-couple-render` (features) + `--couple-render-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (the GR2
   lesson).

## RHI seam additions (summary)
- **None expected.** Reuse the existing instanced lit-sphere pipeline (the per-instance transform buffer +
  `DrawInstanced`/MDI + the lit pipeline + the frame UBO) — all pre-existing (the FPX6/FL6/GR6 set). If a
  genuinely-new RHI need surfaces, STOP and report it. `rhi.h` + `rhi_factory` (baseline 2) + backend dirs
  UNCHANGED. `engine/sim/couple.h` CP1–CP5 + `engine/sim/fpx.h` + `fluid.h` + `cloth.h` + `grain.h` +
  `engine/physics/` UNCHANGED (only the CP6 render helpers appended). Report the seam.

## Out of scope (YAGNI — flagship done after this)
Per-instance body-vs-droplet color (a pipeline/shader change — out; the body is a large sphere among small
droplets), refractive/PBR water materials, metaball/screen-space fluid surfacing, foam/spray, pour animation.
Claim the deterministic fixed-point COUPLED sim (rigid + fluid, two-way, lockstep-replayable) + a lit render
of it. ONE lit 3D render of the barrel-in-water with the provenance + determinism + coverage + empty no-op
proofs and the lit golden — COMPLETING the 6-slice deterministic-coupling flagship (the ELEVENTH flagship).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 97) + the new `couple_test` render-helper cases. Clean
   under `windows-msvc-asan`.
2. **proofs + visual:** `--couple-render-shot` on Vulkan: a coherent LIT 3D barrel-in-water (the deterministic
   coupled sim, rendered); provenance + determinism + coverage + empty no-op. Run under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   rendered image actually shows the lit body floating among the lit fluid droplets (pixel-check the shaded
   region) — do NOT trust the proof's "coherent" claim alone (the NAV6/CL6/FL6 lesson). This is the flagship
   capstone — scrutinize it.**
3. Metal: `visual_test --couple-render` → new golden `tests/golden/metal/couple_render.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm the reused
   instanced-lit pipeline + any shader MSL-generate.** The cross-vendor Vulkan-vs-Metal delta is the documented
   float smoke (~30–60 mean), measured by the controller — NOT strict zero-diff (unlike CP1–CP5).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `couple_render.png` added; the
   other 140 byte-identical (CP1–CP5 + all existing untouched; re-run `--couple-query/buoyancy/displace/step/
   lockstep-shot` → still bit-exact). `git diff master --stat -- tests/golden` = ONLY `couple_render.png`
   (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-couple-render` + `--couple-render-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report the reused pipeline/shader). `scripts/verify.ps1`
   updated: `couple_render` golden in the Mac loop + `--couple-render-shot` in `$vkShots`. CP1–CP5 +
   `engine/sim/fpx.h` + `fluid.h` + `cloth.h` + `grain.h` + `engine/physics/` UNTOUCHED.
