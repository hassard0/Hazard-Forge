# Slice BD6 — Deterministic GPU Crowds: LIT 3D INSTANCED RENDER CAPSTONE (THE MONEY-SHOT) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH + FINAL slice of FLAGSHIP #18
> (DETERMINISTIC GPU CROWDS, `hf::sim::boids`) — the LIT 3D INSTANCED RENDER CAPSTONE, the money-shot, the FPX6/
> GR6/VH6 precedent. BD1-BD5 built + proved a deterministic, path-following, lockstep-replayable crowd. BD6
> RENDERS it: a flock streaming along the A* corridor as a lit 3D scene — a swarm of matte agents on the ground
> under a directional light, the path traced beneath them. **FLOAT render-only** (the sim stays bit-exact integer;
> only the per-agent transform crosses to float — the ONE float crossing, render-only, OUT of the bit-exact sim
> path). **NO new shader, NO new RHI** — reuses the EXISTING `lit_instanced` pipeline VERBATIM (the
> `--grain-render-shot`/`--fpx-render-shot` path). Branch: `slice-bd6`. See [[hazard-forge-boids-roadmap]].

**Goal:** Extend `engine/sim/boids.h` (additive — BD1-BD5 byte-frozen) with `BoidsToRenderInstances` (the bit-exact
agent set → a FLOAT instance set via the agent positions, render-only — the `grain::GrainToRenderInstances` twin).
Add `--boids-render-shot` (Vulkan) / `--boids-render` (Metal) that run the bit-exact path-following flock, build the
instances, and draw a lit 3D crowd through the EXISTING `lit_instanced` pipeline. Bake the FLOAT golden
`boids_render`. **NO new shader, NO new RHI.**

## Design call: the bit-exact flock → float instances → the existing lit-instanced pipeline (matte)
The deterministic-sim flagships all close with a lit 3D render capstone on a FLOAT visresolve-bar: the integer sim
is bit-exact, ONLY the final raster/shade is float → the golden is Metal-baked, the gate is Metal-determinism (two
renders BYTE-IDENTICAL) + provenance (every instance transform IS a pure function of the bit-exact agent state) +
visual parity; the Vulkan-vs-Metal cross-vendor delta is the documented float baseline (~22-55 mean, NOT integer
zero). BD6 is that capstone for the crowd.

**`BoidsToRenderInstances(agents, cfg)`** runs OUTSIDE the bit-exact loop (render-only): the flock has streamed to
a pose along the corridor (K `StepFlockPath` steps — the SAME scene both the Vulkan `--boids-render-shot` AND the
Metal `--boids-render` build, so the integer agent state is byte-identical cross-backend by construction). For each
agent it builds a FLOAT model matrix: `translate(FxToFloat(agent.pos)) × a small uniform AGENT scale` (a matte
sphere/disc, the `GrainToRenderInstances` twin); OPTIONALLY oriented by the agent's velocity (a `lookAt`-style
heading rotation so the agents point where they're going — a render-only float orientation, document if included).
The ONE float crossing is `FxToFloat(pos)` + the transform compose; the sim path stays pure integer.

Drawn through the EXISTING lit-instanced pipeline VERBATIM (`lit_instanced.vert` + `lit.frag` +
`scene::InstanceTransformLayout`, the FrameData camera/light UBO, sky + instanced/static shadow + post — the SAME
path `--grain-render-shot`/`--fpx-render-shot` use) over the ground from a fixed 3/4 camera + a directional light.
**MATTE agents (roughness 1.0)** so they do NOT mirror the sky IBL into iridescence (the GF6/FR6/JT6/VH6 hard
lesson — high roughness + a warm/cool albedo, NOT low-roughness metal). OPTIONALLY render the A* corridor polyline
+ the navmesh faintly underneath (a second draw / lines — `nav::PathToWorldPolyline`, the NAV5 viz; document if
included). NO new shader/RHI — only a new showcase + the render-only helper.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **BD1-BD5 (this branch's `boids.h`, read-only — build on, DON'T modify):** `Agent`, `FlockConfig`, `BoidsPath`,
  `StepFlockPath`/`StepFlockPathSteps` (run the flock to its streaming pose). DO NOT modify the BD1-BD5 functions.
  BD6 APPENDS only `BoidsToRenderInstances`.
- **The render-only instance helper to TWIN (`engine/sim/grain.h::GrainToRenderInstances`, grain.h:~1015):** the
  host float `std::vector<scene::InstanceData>` (or the actual instance struct) builder from bit-exact particle
  state. `BoidsToRenderInstances` is the SAME shape with `Agent.pos`. Also `fpx::FxToFloat` (the Q16.16→float, the
  ONE crossing). **DO NOT modify grain.h/fpx.h.**
- **THE RENDER CAPSTONE SHOWCASE (`samples/hello_triangle/main.cpp`):** study `--grain-render-shot` /
  `--fpx-render-shot` (the canonical FxToFloat-per-particle lit instanced render: the instance build → the
  `lit_instanced` pipeline → camera/light/sky/shadow/post; MATTE roughness to dodge iridescence). Copy their
  plumbing VERBATIM, swap the instance source to the crowd. The Metal `--grain-render`/`--fpx-render` in
  `metal_headless/visual_test.mm` is the Metal precedent for `--boids-render`. The BD4 navmesh+corridor build (for
  the optional path underneath) + `nav::PathToWorldPolyline` (navmesh.h:1262, read-only).
- **The scene instance struct + the lit-instanced pipeline (`engine/scene/vertex.h`):** `InstanceData{float
  model[16]}` (vertex.h:50) + `InstanceTransformLayout` (vertex.h:51), `shaders/lit_instanced.vert.hlsl` +
  `shaders/lit.frag.hlsl` — REUSED VERBATIM, NO new shader.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), `tests/boids_test.cpp`. Standalone arg-parse (the FR1 C1061 lesson).

## Design decisions (locked)
1. **`BoidsToRenderInstances(const std::vector<Agent>& agents, const FlockConfig& cfg) -> std::vector<scene::
   InstanceData>`** (or the actual instance type) — one float model matrix per agent (FxToFloat(pos) × the agent
   scale, optionally velocity-oriented), FLOAT render-only. A PURE FUNCTION of the bit-exact agent state (two calls
   byte-equal — the provenance contract).
2. **Showcase `--boids-render-shot <out>` (Vulkan) AND `--boids-render` (Metal) — WIRE BOTH** (standalone
   arg-parse). Build the BD4 navmesh+corridor + a flock at the start, run K `StepFlockPath` steps to a streaming
   pose (the SAME scene both backends build → byte-identical integer state), build the instances, draw the matte
   lit 3D crowd through the EXISTING lit-instanced pipeline (camera/light/sky/shadow/post REUSED) over the ground
   (optionally the corridor polyline underneath). Golden = `tests/golden/metal/boids_render.png` (Mac-baked by the
   CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) instances from bit-exact state:** print `boids-render: {agents:<N>, instances:<M>} from bit-exact flock
     state` (M == N; the instance transforms ARE the bit-exact agent poses).
   - **(2) determinism:** two renders → BYTE-IDENTICAL (the Metal-determinism gate). Print `boids-render
     determinism: two runs BYTE-IDENTICAL`.
   - **(3) provenance:** the built instance set == a rebuild from the bit-exact agent state (`BoidsToRenderInstances`
     is a pure function — call it twice, assert byte-equal). Print `boids-render provenance: instances == rebuild`.
   - **Golden discipline: ONLY `tests/golden/metal/boids_render.png`; do NOT commit it.** Existing 182 image
     goldens UNTOUCHED (incl BD1-BD5).
4. **Cross-backend bar (FLOAT, the visresolve precedent):** the SIM is bit-exact but the final raster/shade is
   float → the golden is Metal-baked; the gate is Metal-determinism (two renders BYTE-IDENTICAL) + provenance +
   visual parity (a coherent lit 3D crowd). The Vulkan-vs-Metal cross-vendor delta is the documented float baseline
   (~22-55 mean, the GR6/FPX6 number) — the CONTROLLER confirms it is in-band, NOT a strict-zero compare.
5. **Tests `tests/boids_test.cpp` additions (pure CPU):** `BoidsToRenderInstances` — produces N instances (one per
   agent); each instance's translation == its agent's `FxToFloat(pos)`; the helper is a PURE FUNCTION (two calls
   byte-equal — the provenance contract). Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-boids-render` (features) + `--boids-render-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the EXISTING lit-instanced pipeline + the offscreen render path VERBATIM. `rhi.h` + backend dirs
  UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `joint.h` + `couple*.h` + `fract.h` +
  `vehicle.h` + `active.h` + `engine/nav/` (reused read-only) + `engine/anim/` + `engine/physics/` + all existing
  shaders (incl the boids compute shaders) + `hf_gen_msl` UNCHANGED (BD6 reuses `lit_instanced.vert` + `lit.frag` —
  NO new shader). BD1-BD5 `boids.h` code UNCHANGED (BD6 additive — only the render-only helper + the showcase).
  Report the seam empty.

## Out of scope (flagship #18 is COMPLETE after BD6)
A skinned/animated agent mesh (the agents are matte instanced primitives — the deterministic-sim-capstone
convention: lit primitives, not art assets), LODs, per-agent color-by-state beyond the basic, motion blur. BD6
claims ONLY: a lit 3D render of the bit-exact path-following flock (matte instanced agents streaming along the
corridor) through the existing pipeline, Metal-deterministic + provenance-checked + visually coherent, the
cross-vendor float baseline in-band. **Completes the 6-slice deterministic-GPU-crowds flagship — the EIGHTEENTH
flagship.**

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 104 + the new `boids_test` render-instance cases). Clean
   under `windows-msvc-asan` (build+run `boids_test` + `introspect_test`).
2. **proofs + visual:** `--boids-render-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED — the lit-instanced
   pipeline binds the per-frame shadow map). **VERIFY the image shows a coherent lit 3D crowd — a swarm of matte
   agents on the ground (streaming along the corridor), NOT iridescent, NOT black/scrambled, NOT collapsed (the
   HARD visual gate; the GF6/FR6/VH6 lesson).** Re-run `--boids-lockstep-shot` + `--boids-path-shot` +
   `--boids-flock-shot` + `--boids-neighbors-shot` + `--boids-steer-shot` → still bit-exact (BD1-BD5
   render-invariance).
3. Metal: `visual_test --boids-render` → new golden `tests/golden/metal/boids_render.png`; **two runs DIFF 0.0000**
   (the FLOAT-capstone gate is Metal-DETERMINISM). **Confirm `visual_test.mm` in the diff; confirm NO new shader
   (`hf_gen_msl` UNCHANGED).** The Vulkan-vs-Metal cross-vendor compare is the FLOAT baseline (~22-55 mean) — the
   controller confirms it is in-band (NOT strict zero).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `boids_render.png` added; the other
   182 byte-identical. `git diff master --stat -- tests/golden` = ONLY `boids_render.png` (metal) (the introspect
   json rebake = controller, post-gate).
5. Introspect: exactly `+deterministic-boids-render` + `--boids-render-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + the frozen sim/nav/anim/physics headers + ALL existing shaders byte-unchanged; ONLY
   `boids.h` extended additively + the showcase/test/introspect). `scripts/verify.ps1` updated: `boids_render`
   golden + `--boids-render-shot` in `$vkShots`. **NO new shader; `hf_gen_msl` UNCHANGED.**
