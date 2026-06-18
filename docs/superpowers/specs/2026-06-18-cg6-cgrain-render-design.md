# Slice CG6 — Deterministic Rigid↔Grain Coupling: LIT 3D RENDER CAPSTONE (the money-shot, COMPLETES flagship #12) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice of FLAGSHIP #12
> (DETERMINISTIC TWO-WAY RIGID↔GRAIN COUPLING, `hf::sim::cgrain`). RENDERS the bit-exact coupled state as a lit
> 3D scene — the **rigid body half-buried in the sand** as lit 3D INSTANCED SPHERES (a large body sphere among
> small grain spheres) — the money-shot showing the deterministic fixed-point COUPLED rigid+granular sim as a
> real lit 3D scene. The instance transforms come from the BIT-EXACT integer state (`fpx::FxBodyTransform` for
> the body + `grain::GrainToRenderInstances` for the grains; the sim stays integer). The raster/shade is float
> → the FLOAT visresolve-bar (like FPX6/FL6/GR6/CP6): Metal-baked golden, Metal-determinism DIFF 0.0000 +
> provenance + visual parity, cross-vendor ~the float baseline. Reuses the EXISTING instanced lit-sphere
> pipeline (FPX6/FL6/GR6/CP6 path) — NO new RHI, NO new shader. After this slice flagship #12 is COMPLETE (6/6)
> — the TWELFTH flagship. Branch: `slice-cg6`. See [[hazard-forge-couple-grain-roadmap]].

**Goal:** Add render-only float helpers to `engine/sim/couple_grain.h` (`CGrainToRenderInstances` → the combined
per-instance transform set: one big sphere per `fpx::FxBody` (via `FxBodyTransform`) + one small sphere per
`grain::GrainParticle` (via `GrainToRenderInstances`/`pos/kOne`)), run the deterministic CG1–CG5 coupled sim
(`StepCGrain`) to a settled body-in-sand state, and render the body + grains as lit 3D INSTANCED SPHERES
through the EXISTING instanced lit pipeline from a fixed 3/4 camera + directional light. Add `--cgrain-render-shot`
(Vulkan) / `--cgrain-render` (Metal). Bake the lit golden `cgrain_render`. Make-safe: small host float helpers
(CG1–CG5 sim UNCHANGED) + a NEW showcase + NEW golden; reuse the existing pipeline — NO new shader/RHI.

## Design call: FLOAT visresolve-bar (the cgrain arc's only float slice) + KEEP THE SETTLE FAST
CG1–CG5 are strict integer/bit-exact. CG6 RASTERIZES the state with a perspective camera + lighting →
cross-vendor float (the FPX6/CP6 caveat). CG6's golden is the FLOAT bar, identical to CP6/GR6: the committed
golden is **baked on Metal**; the gate is **Metal-render == Metal-golden DIFF 0.0000** (deterministic,
two-run) + **provenance** (the rendered per-instance transforms are built from the bit-exact integer state —
the body via `FxBodyTransform`, the grains via `pos/kOne`) + visual parity, with the Vulkan-vs-Metal delta the
documented cross-vendor smoke (~30–60 mean, NOT zero). The fixed-point SIM feeding the render is still
bit-exact (CG1–CG5); only the final raster/shade is float. **PERFORMANCE NOTE: `StepCGrain` is HEAVY (~79s for
2925 grains × 300 steps). Use a MODEST scene (a few hundred to ~1k grains) settled in a MODEST step count so
the showcase setup is a minute or two, not an hour — the render's quality is the body-in-sand read, not a long
settle.**

## THE SCENE — a clean "body in sand" money-shot
Run the deterministic CG1–CG5 coupled sim (a body dropped onto/into a confined grain bed — the CG4
static-basin recipe) for enough steps that the body settles **half-buried** in a recognizable sand pile, then
render. PREFER rendering only the DYNAMIC grains + the body (drop the static basin-wall grains from the RENDER
— they're a containment detail, not the headline; document the choice, the CP6 precedent). If a perfectly
clean pile proves finicky, a coherent settled bed + the half-buried body is acceptable — the render is
golden-verified for determinism+provenance regardless; but PREFER the clean "body in sand" read. NO RNG/clock.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The coupled state (the input):** `engine/sim/couple_grain.h` — `CGrainWorld` (bodies + grains),
  `StepCGrain`/`StepCGrainSteps` (CG4 — run to a settled body-in-sand state). The body world transform =
  `fpx::FxBodyTransform(body)`; the grain transforms = `grain::GrainToRenderInstances` (GR6 — one mat4 per
  grain at `pos/kOne`, scaled by the grain radius).
- **The CP6 combined-instance render to MIRROR (`engine/sim/couple.h` `CoupleToRenderInstances` + the
  `--couple-render-shot` branch):** the combined instance set (the body spheres + the grain spheres) drawn in
  ONE instanced draw of the **instanced lit pipeline reused VERBATIM** (`lit_instanced.vert`+`lit.frag` +
  `scene::InstanceTransformLayout` per-instance mat4 + the FrameData camera/light UBO + sky/shadow/post). All
  instances share the pipeline/material; the body is distinguished by being a LARGE sphere among small grains
  (NO per-instance colour → no new shader). Report which pipeline/shader was reused. **(NAV6/CL6/FL6/CP6
  lesson: VERIFY the rendered output actually shows the lit body + grains — pixel-check the shaded region, do
  NOT trust the proof's "coherent" claim alone.)**
- **The float-golden discipline (the bar to copy):** FPX6 / FL6 / GR6 / CP6 — per-backend Metal golden,
  determinism, provenance, the cross-vendor ~30–60 baseline (NOT zero-diff).
- **Showcase + registration:** the CP6/GR6 `--*-render-shot` template; `scripts/verify.ps1`,
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden** — the
  GR2/CP2 lesson). main.cpp has `/bigobj` (CL4).

## Design decisions (locked)
1. **Run the coupled sim to a body-in-sand state, then render.** Run the deterministic CG1–CG5 pipeline (a
   body settling half-buried in a confined grain bed; modest grain count + step count per the perf note). The
   body's + grains' final integer state is the bit-exact sim output. Build a per-instance `math::Mat4` per body
   (`FxBodyTransform`) + per grain (`pos/kOne`, scaled by the grain radius). Pure deterministic host float
   (render-only).
2. **Render through the EXISTING instanced lit-sphere pipeline — NO new RHI, NO new shader.** Reuse the
   instanced lit-sphere `GraphicsPipelineDesc` (FPX6/CP6's `lit_instanced.vert`+`lit.frag`) + the per-instance
   transform buffer + the frame/camera UBO; a unit `SphereGeometry`, the combined instance set (bodies +
   grains), a fixed 3/4 camera + directional light + the ground. Report exactly which pipeline/shader was
   reused; if a genuinely-new shader is needed, flag it loudly (prefer reuse).
3. **Showcase `--cgrain-render-shot <out>` (Vulkan, main.cpp) AND `--cgrain-render` (Metal, visual_test.mm —
   WIRE BOTH; confirm `#include "sim/couple_grain.h"`).** Run the deterministic coupled sim → per-body +
   per-grain instance transforms → instanced lit render → the BGRA8 image. Golden =
   `tests/golden/metal/cgrain_render.png` (baked on the Mac by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance (the coupled state IS bit-exact):** the per-instance transforms are built from the
     `CGrainWorld` state that equals the CPU `StepCGrainSteps` reference (the bit-exact CG1–CG5 sim — print the
     body + grain counts + that the transforms derive from `FxBodyTransform`/`pos/kOne`). Print `cgrain-render:
     {bodies:<B>, grains:<N>, instances:<I>, shaded:<S>} (fixed-point coupling -> lit 3D render)`.
   - **(2) determinism (same backend):** two renders → DIFF 0.0000. Print `cgrain-render determinism: two
     renders BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the lit body + grains cover a coherent region (`shaded > 0`, a recognizable
     body-in-sand). Print `cgrain-render coverage: <S> shaded (coherent lit body + sand)`.
   - **(4) empty no-op:** zero bodies + zero grains → the cleared background / ground only. Print `cgrain-render
     empty: base only (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/cgrain_render.png`; do NOT commit it — the CONTROLLER bakes
     on the Mac.** Existing 146 image goldens UNTOUCHED.
5. **Cross-backend bar (FLOAT, per the FPX6/CP6 finding):** Metal-output == Metal-golden DIFF 0.0000
   (determinism) + provenance (the bit-exact coupled state) + VISUAL parity; the Vulkan-vs-Metal cross-vendor
   delta is the documented float baseline (~30–60 mean — like FPX6/CP6/GR6, NOT zero), measured by the
   controller (a LARGER/structural delta or a visual mismatch is a bug).
6. **Tests `tests/cgrain_test.cpp` additions (pure CPU):** `CGrainToRenderInstances` — a known coupled state →
   the expected combined instance transforms (the body's `FxBodyTransform` + the grains' translate+scale, the
   right instance count B+N); the sim feeding it is the existing bit-exact `StepCGrain` (already tested). Clean
   under `windows-msvc-asan`. (The render is golden-verified, not unit-tested.)
7. **Introspect.** Add exactly `deterministic-cgrain-render` (features) + `--cgrain-render-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (the GR2/CP2
   lesson).

## RHI seam additions (summary)
- **None expected.** Reuse the existing instanced lit-sphere pipeline (the per-instance transform buffer +
  `DrawInstanced`/MDI + the lit pipeline + the frame UBO) — all pre-existing (the FPX6/CP6 set). If a
  genuinely-new RHI need surfaces, STOP and report it. `rhi.h` + `rhi_factory` (baseline 2) + backend dirs
  UNCHANGED. `engine/sim/couple_grain.h` CG1–CG5 + `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` +
  `couple.h` + `engine/physics/` UNCHANGED (only the CG6 render helpers appended). Report the seam.

## Out of scope (YAGNI — flagship done after this)
Per-instance body-vs-grain colour (a pipeline/shader change — out; the body is a large sphere among small
grains), PBR/sand materials, metaball/surfacing, foam/spray, partial-burial cutaway. Claim the deterministic
fixed-point COUPLED rigid+granular sim (two-way, lockstep-replayable) + a lit render of it. ONE lit 3D render
of the body-in-sand with the provenance + determinism + coverage + empty no-op proofs and the lit golden —
COMPLETING the 6-slice deterministic-rigid↔grain-coupling flagship (the TWELFTH flagship).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 98) + the new `cgrain_test` render-helper cases. Clean
   under `windows-msvc-asan`.
2. **proofs + visual:** `--cgrain-render-shot` on Vulkan: a coherent LIT 3D body-in-sand (the deterministic
   coupled sim, rendered); provenance + determinism + coverage + empty no-op. Run under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   rendered image actually shows the lit body half-buried among the lit grains (pixel-check the shaded region)
   — do NOT trust the proof's "coherent" claim alone (the NAV6/CL6/FL6/CP6 lesson). This is the flagship
   capstone — scrutinize it.**
3. Metal: `visual_test --cgrain-render` → new golden `tests/golden/metal/cgrain_render.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm the reused
   instanced-lit pipeline + any shader MSL-generate.** The cross-vendor Vulkan-vs-Metal delta is the documented
   float smoke (~30–60 mean), measured by the controller — NOT strict zero-diff (unlike CG1–CG5).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cgrain_render.png` added; the
   other 146 byte-identical (CG1–CG5 + all existing untouched; re-run `--cgrain-query/support/displace/step/
   lockstep-shot` → still bit-exact). `git diff master --stat -- tests/golden` = ONLY `cgrain_render.png`
   (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cgrain-render` + `--cgrain-render-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report the reused pipeline/shader). `scripts/verify.ps1`
   updated: `cgrain_render` golden in the Mac loop + `--cgrain-render-shot` in `$vkShots`. CG1–CG5 +
   `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple.h` + `engine/physics/` UNTOUCHED.
