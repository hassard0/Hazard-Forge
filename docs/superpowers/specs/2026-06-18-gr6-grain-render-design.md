# Slice GR6 — Deterministic GPU Granular/Sand: LIT 3D RENDER CAPSTONE (the money-shot, COMPLETES flagship #10) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice of FLAGSHIP #10
> (DETERMINISTIC GPU GRANULAR / SAND, `hf::sim::grain`). RENDERS the bit-exact grain pile as a lit 3D scene —
> the settled sand as lit 3D INSTANCED SPHERES (a heap of lit grains) — the money-shot showing the
> deterministic fixed-point granular sim as a real lit 3D scene. The instance transforms come from the
> BIT-EXACT integer grain state (`GrainParticle.pos → pos/kOne` host-side, render-only; the sim stays
> integer). The raster/shade is float → the FLOAT visresolve-bar (like FPX6/FL6/CL6/MC6/NAV6): Metal-baked
> golden, Metal-determinism DIFF 0.0000 + provenance + visual parity, cross-vendor ~the float baseline.
> Reuses the EXISTING instanced lit-sphere pipeline (FPX6/FL6's path) — NO new RHI, NO new shader. After this
> slice flagship #10 is COMPLETE (6/6) — the TENTH flagship. Branch: `slice-gr6`. See [[hazard-forge-grain-roadmap]].

**Goal:** Add render-only float helpers to `engine/sim/grain.h` (`GrainVertToWorld` = the host `pos/kOne`
conversion; `GrainToRenderInstances` → one per-instance model matrix per grain: a sphere at the grain's
world position, scaled by the grain radius), run the deterministic GR1–GR5 granular sim (WITH friction) to a
compelling settled state, and render the grains as lit 3D INSTANCED SPHERES through the EXISTING instanced
lit pipeline (the FPX6 `RunFpxRenderShowcase` / FL6 path) from a fixed 3/4 camera + directional light (+ the
ground). Add `--grain-render-shot` (Vulkan) / `--grain-render` (Metal). Bake the lit golden `grain_render`.
Make-safe: small host float helpers (GR1–GR5 sim UNCHANGED — only a float position/instance accessor for
rendering) + a NEW showcase + NEW golden; reuse the existing instanced lit-sphere pipeline — NO new shader/RHI.

## Design call: FLOAT visresolve-bar (the grain arc's only float slice)
GR1–GR5 are strict integer/bit-exact (the granular sim is fixed-point). GR6 RASTERIZES the state with a
perspective camera + lighting → cross-vendor float (the FPX6/FL6 caveat). GR6's golden is the FLOAT bar,
identical to FL6: the committed golden is **baked on Metal**; the gate is **Metal-render == Metal-golden DIFF
0.0000** (deterministic, two-run) + **provenance** (the rendered per-instance transforms are built from the
bit-exact integer `GrainParticle.pos` — `pos/kOne`, deterministic host code) + visual parity, with the
Vulkan-vs-Metal delta the documented cross-vendor smoke (~30–60 mean, the engine's float-render baseline —
NOT zero). The fixed-point SIM feeding the render is still bit-exact (GR1–GR5); only the final raster/shade
is float.

## THE SCENE — make the capstone a DRAMATIC angle-of-repose CONE (also addresses the GR4 modest-visual note)
GR4's friction was bit-exact but its showcase was a near-cubic block that froze into a modest slump, not a
dramatic poured cone (documented). GR6 is the capstone money-shot — **use a scene that produces a
recognizable angle-of-repose CONE in lit 3D**, so the render BOTH lands the money-shot AND visually proves
friction does dramatic work. Run the deterministic GR1–GR5 friction sim (a TALLER / NARROWER starting column,
or grains released ABOVE the repose angle, with the GR4 stagger so friction engages — see the GR4 finding)
for enough steps that the sides slump down and the pile settles into a CONE with clearly sloped sides on FLAT
ground (NO container — friction-alone, the beyond-UE5 claim). The render is float, but the SIM is the
bit-exact GR1–GR5; `MeasureGrainRepose` on the final state should show a clear non-flat slope. (If a dramatic
cone proves finicky to tune deterministically, a coherent settled sand HEAP is acceptable — the render is
golden-verified for determinism+provenance regardless; but PREFER the cone, and report the repose slope.)

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The grain state (the input):** `engine/sim/grain.h` — `GrainParticle` (pos), `StepGrainFriction` (GR4 —
  run the friction sim to a settled cone/heap). World position = `pos.xyz / (float)kOne` (the ONE host float
  conversion). `MeasureGrainRepose` for the slope stat.
- **The existing instanced lit-sphere pipeline to REUSE (find + mirror — do NOT invent):** FL6's
  `FluidToRenderInstances` + the `--fluid-render-shot` branch in `samples/hello_triangle/main.cpp` + the
  `--fluid-render` branch in `metal_headless/visual_test.mm` — the **instanced lit pipeline**
  (`lit_instanced.vert` + `lit.frag` + `scene::InstanceTransformLayout` per-instance mat4 + the FrameData
  camera/light UBO + sky/shadow/post, REUSED VERBATIM from FPX6's `--fpx-render-shot`): one lit sphere
  instance per particle, the per-instance model matrix in an instance buffer. GR6 is the SAME — one sphere
  instance per GRAIN (`GrainToRenderInstances` builds the per-instance transforms from the bit-exact `pos`).
  Report which pipeline/shader was reused. **(NAV6/CL6/FL6 lesson: VERIFY the rendered output actually shows
  the lit grain spheres — pixel-check the shaded heap region, do NOT trust the proof's "coherent" claim
  alone.)**
- **The float-golden discipline (the bar to copy):** FPX6 / FL6 / NAV6 — per-backend Metal golden,
  determinism, provenance, the cross-vendor ~30–60 baseline (NOT zero-diff).
- **Showcase + registration:** the FL6 `--fluid-render-shot` template; `scripts/verify.ps1`,
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden** — the
  GR2 lesson). main.cpp has `/bigobj` (CL4).

## Design decisions (locked)
1. **Run the friction sim to a compelling cone, then render.** Run the deterministic GR1–GR5 pipeline (the
   GR4 friction sim — a column/heap that slumps to a repose cone on flat ground, the GR4 stagger so friction
   engages) to its settled state. The grains' final `GrainParticle.pos` is the bit-exact sim output. Build a
   per-instance `math::Mat4` per grain: translate by `pos/kOne`, scale by the grain radius. Pure deterministic
   host float (render-only). Optionally tint the spheres a sand color (a host constant — render-only).
2. **Render through the EXISTING instanced lit-sphere pipeline — NO new RHI, NO new shader.** Reuse the
   instanced lit-sphere `GraphicsPipelineDesc` (FPX6/FL6's `lit_instanced.vert`+`lit.frag`) + the per-instance
   transform buffer + the frame/camera UBO; a unit `SphereGeometry`, N instances (one per grain), a fixed 3/4
   camera + directional light + the ground. Report exactly which pipeline/shader was reused; if a
   genuinely-new shader is needed, flag it loudly (prefer reuse).
3. **Showcase `--grain-render-shot <out>` (Vulkan, main.cpp) AND `--grain-render` (Metal, visual_test.mm —
   WIRE BOTH; confirm `#include "sim/grain.h"`).** Run the deterministic friction sim → per-grain instance
   transforms → instanced lit render → the BGRA8 image. Golden = `tests/golden/metal/grain_render.png` (baked
   on the Mac by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance (the grain pile IS bit-exact):** the per-instance transforms are built from the
     `GrainParticle` state that equals the CPU `StepGrainFriction` reference (the bit-exact GR1–GR5 sim — print
     the grain count + that the transforms derive from `pos/kOne`). Print `grain-render: {grains:<N>,
     instances:<N>, shaded:<S>} (fixed-point sand -> lit 3D render)`.
   - **(2) determinism (same backend):** two renders → DIFF 0.0000. Print `grain-render determinism: two
     renders BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the lit grains cover a coherent region (`shaded > 0`, a recognizable sand
     heap/cone). Print `grain-render coverage: <S> shaded (coherent lit sand pile)` + optionally the repose
     slope from `MeasureGrainRepose`.
   - **(4) empty no-op:** zero grains → the cleared background / ground only. Print `grain-render empty: base
     only (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/grain_render.png`; do NOT commit it — the CONTROLLER bakes
     on the Mac.** Existing 134 image goldens UNTOUCHED.
5. **Cross-backend bar (FLOAT, per the FPX6/FL6 finding):** Metal-output == Metal-golden DIFF 0.0000
   (determinism) + provenance (the bit-exact grain state) + VISUAL parity; the Vulkan-vs-Metal cross-vendor
   delta is the documented float baseline (~30–60 mean — like FPX6/FL6/scene_shadow, NOT zero), measured by
   the controller (a LARGER/structural delta or a visual mismatch is a bug).
6. **Tests `tests/grain_test.cpp` additions (pure CPU):** `GrainVertToWorld` — a known integer pos → the
   expected float world position (scale correct); `GrainToRenderInstances` — N grains → N instance transforms
   with the right translate+scale; the sim feeding it is the existing bit-exact `StepGrainFriction` (already
   tested). Clean under `windows-msvc-asan`. (The render is golden-verified, not unit-tested.)
7. **Introspect.** Add exactly `deterministic-grain-render` (features) + `--grain-render-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (the GR2
   lesson).

## RHI seam additions (summary)
- **None expected.** Reuse the existing instanced lit-sphere pipeline (the per-instance transform buffer +
  `DrawInstanced`/MDI + the lit pipeline + the frame UBO) — all pre-existing (the FPX6/FL6 set). If a
  genuinely-new RHI need surfaces, STOP and report it. `rhi.h` + `rhi_factory` (baseline 2) + backend dirs
  UNCHANGED. `engine/sim/grain.h` GR1–GR5 + `engine/sim/fpx.h` + `cloth.h` + `fluid.h` + `engine/physics/`
  UNCHANGED (only the GR6 render helpers appended). Report the seam.

## Out of scope (YAGNI — flagship done after this)
Per-grain rolling/tumbling orientation render, PBR/SSS sand materials, ambient occlusion between grains,
metaball/surfacing of the pile (instanced spheres reuse FPX6/FL6's pipeline), pour animation/video. Claim the
deterministic fixed-point GRANULAR sim (the substance, with friction + lockstep) + a lit render of it. ONE
lit 3D render of the sand pile with the provenance + determinism + coverage + empty no-op proofs and the lit
golden — COMPLETING the 6-slice deterministic-granular flagship (the TENTH flagship).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 96) + the new `grain_test` render-helper cases. Clean
   under `windows-msvc-asan`.
2. **proofs + visual:** `--grain-render-shot` on Vulkan: a coherent LIT 3D sand pile/CONE (the deterministic
   grain sim, rendered); provenance + determinism + coverage + empty no-op. Run under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   rendered image actually shows the lit sand grains in a recognizable heap/cone (pixel-check the shaded
   region) — do NOT trust the proof's "coherent" claim alone (the NAV6/CL6/FL6 lesson). This is the flagship
   capstone — scrutinize it.**
3. Metal: `visual_test --grain-render` → new golden `tests/golden/metal/grain_render.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm the reused
   instanced-lit pipeline + any shader MSL-generate.** The cross-vendor Vulkan-vs-Metal delta is the documented
   float smoke (~30–60 mean), measured by the controller — NOT strict zero-diff (unlike GR1–GR5).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `grain_render.png` added; the
   other 134 byte-identical (GR1–GR5 + all existing untouched; re-run `--grain-integrate/neighbors/contact/
   friction/lockstep-shot` → still bit-exact). `git diff master --stat -- tests/golden` = ONLY `grain_render.png`
   (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-grain-render` + `--grain-render-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report the reused pipeline/shader). `scripts/verify.ps1`
   updated: `grain_render` golden in the Mac loop + `--grain-render-shot` in `$vkShots`. GR1–GR5 +
   `engine/sim/fpx.h` + `cloth.h` + `fluid.h` + `engine/physics/` UNTOUCHED.
