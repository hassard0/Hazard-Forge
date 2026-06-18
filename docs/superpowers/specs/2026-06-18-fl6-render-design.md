# Slice FL6 — Deterministic GPU Fluid: LIT 3D RENDER CAPSTONE (the money-shot, COMPLETES flagship #9) (Phase 14 #6) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice of FLAGSHIP #9
> (DETERMINISTIC GPU FLUID via Position-Based Fluids, `hf::sim::fluid`). RENDERS the bit-exact fluid as a
> lit 3D scene — the settled/dynamic fluid particles as lit 3D INSTANCED SPHERES (a pool/splash of lit
> droplets) — the money-shot showing the deterministic fixed-point fluid as a real lit 3D scene. The
> instance transforms come from the BIT-EXACT integer fluid state (`FluidParticle.pos` → a float position
> host-side, for rendering ONLY — the sim stays integer); the raster/shade is float, so this is the FLOAT
> visresolve-bar (like FPX6/CL6/MC6/NAV6): Metal-baked golden, Metal-determinism DIFF 0.0000 + provenance
> + visual parity, cross-vendor ~the float baseline. Reuses the EXISTING instanced lit-sphere pipeline
> (FPX6's path) — NO new RHI, NO new shader. After this slice flagship #9 is COMPLETE (6/6). Branch:
> `slice-fl6`. See [[hazard-forge-fluid-roadmap]].

**Goal:** Add render-only float helpers to `engine/sim/fluid.h` (`FluidVertToWorld` = the single host
`pos/kOne` conversion; `FluidToRenderInstances` → one per-instance model matrix per fluid particle: a
small sphere at the particle's world position, scaled by the droplet radius), run the deterministic
FL1–FL5 fluid sim to a settled/dynamic state, and render the particles as lit 3D INSTANCED SPHERES through
the EXISTING instanced lit pipeline (the FPX6 `RunFpxRenderShowcase` path) from a fixed 3/4 camera +
directional light (+ the ground + optionally the FxBody collider sphere). Add `--fluid-render-shot`
(Vulkan) / `--fluid-render` (Metal). Bake the lit golden `fluid_render`. Make-safe: small host float
helpers (FL1–FL5 sim UNCHANGED — only a float position/instance accessor for rendering) + a NEW showcase +
NEW golden; reuse the existing instanced lit-sphere pipeline — NO new shader/RHI.

## Design call: FLOAT visresolve-bar (the fluid arc's only float slice)
FL1–FL5 are strict integer/bit-exact (the fluid sim is fixed-point). FL6 RASTERIZES the state with a
perspective camera + lighting → cross-vendor float (the FPX6/CL6/visresolve caveat). So FL6's golden is the
FLOAT bar, identical to FPX6: the committed golden is **baked on Metal**; the gate is **Metal-render ==
Metal-golden DIFF 0.0000** (deterministic, two-run) + **provenance** (the rendered per-instance transforms
are built from the bit-exact integer `FluidParticle.pos` — `pos/kOne`, deterministic host code) + visual
parity, with the Vulkan-vs-Metal delta the documented cross-vendor smoke (~30–60 mean, the engine's
float-render baseline — NOT zero). The fixed-point SIM feeding the render is still bit-exact (FL1–FL5); only
the final raster/shade is float. **Use a COMPELLING fluid state (the FL4/FL5 lesson — more sim steps / a
dynamic pour over the sphere) so the money-shot reads as a recognizable fluid, not a static block.**

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The fluid state (the input):** `engine/sim/fluid.h` — `FluidParticle` (pos), `StepFluid` (FL4 — run
  the sim to a settled/dynamic state, e.g. a dam-break pouring over an FxBody sphere). World position =
  `pos.xyz / (float)kOne` (the ONE host float conversion).
- **The existing instanced lit-sphere pipeline to REUSE (find + mirror — do NOT invent):** FPX6's
  `RunFpxRenderShowcase` (`metal_headless/visual_test.mm`) + the `--fpx-render-shot` branch in
  `samples/hello_triangle/main.cpp` — the **instanced lit pipeline** (`lit_instanced.vert` + `lit.frag` +
  `scene::InstanceTransformLayout` per-instance mat4 + the FrameData camera/light UBO + sky/shadow/post,
  REUSED VERBATIM from `--instanced-shot`): one lit sphere instance per body, the per-instance model
  matrix in an instance buffer. FL6 is the SAME — one sphere instance per FLUID PARTICLE (`FluidToRender‐
  Instances` builds the per-instance transforms from the bit-exact `pos`). Report which pipeline/shader was
  reused. **(NAV6/CL6 lesson: VERIFY the rendered output actually shows the lit fluid spheres — pixel-check
  the shaded droplet region, do NOT trust the proof's "coherent" claim alone.)**
- **The float-golden discipline (the bar to copy):** FPX6 / CL6 / NAV6 — per-backend Metal golden,
  determinism, provenance, the cross-vendor ~30–60 baseline (NOT zero-diff).
- **Showcase + registration:** the FPX6 `--fpx-render-shot` template; `verify.ps1`/`introspect.cpp`/
  `introspect_test.cpp`. main.cpp has `/bigobj` (CL4).

## Design decisions (locked)
1. **Run the sim to a compelling state, then render.** Run the deterministic FL1–FL5 pipeline (a dam-break
   block dropped onto the ground + an FxBody sphere, `StepFluid` enough steps that the fluid pours/settles
   into a recognizable pool/splash over the sphere — address the FL4 sparse-viz note with MORE steps or a
   taller column). The bodies' final `FluidParticle.pos` is the bit-exact sim output. Build a per-instance
   `math::Mat4` per particle: translate by `pos/kOne`, scale by the droplet radius. Pure deterministic host
   float (render-only).
2. **Render through the EXISTING instanced lit-sphere pipeline — NO new RHI, NO new shader.** Reuse the
   instanced lit-sphere `GraphicsPipelineDesc` (FPX6's `lit_instanced.vert`+`lit.frag`) + the per-instance
   transform buffer + the frame/camera UBO; a unit `SphereGeometry`, N instances (one per particle), a
   fixed 3/4 camera + directional light + the ground (+ optionally the collider sphere as a second lit
   mesh). Report exactly which pipeline/shader was reused; if a genuinely-new shader is needed, flag it
   loudly (prefer reuse).
3. **Showcase `--fluid-render-shot <out>` (Vulkan, main.cpp) AND `--fluid-render` (Metal, visual_test.mm —
   WIRE BOTH; confirm visual_test.mm + `#include "sim/fluid.h"`).** Run the deterministic fluid sim → per-
   particle instance transforms → instanced lit render → the BGRA8 image. Golden = the lit 3D fluid pool →
   `tests/golden/metal/fluid_render.png` (baked on the Mac by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance (the fluid IS bit-exact):** the per-instance transforms are built from the
     `FluidParticle` state that equals the CPU `StepFluid` reference (the same bit-exact FL1–FL5 sim — print
     the particle count + that the transforms derive from `pos/kOne`). Print `fluid-render: {particles:<N>,
     instances:<N>, shaded:<S>} (fixed-point fluid -> lit 3D render)`.
   - **(2) determinism (same backend):** two renders → DIFF 0.0000. Print `fluid-render determinism: two
     renders BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the lit droplets cover a coherent region (`shaded > 0`, not uniform — a
     recognizable fluid pool/splash). Print `fluid-render coverage: <S> shaded (coherent lit fluid)`.
   - **(4) empty no-op:** zero particles → the cleared background / ground only. Print `fluid-render empty:
     base only (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/fluid_render.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 128 image goldens UNTOUCHED.
5. **Cross-backend bar (FLOAT, per the FPX6/CL6 finding):** Metal-output == Metal-golden DIFF 0.0000
   (determinism) + provenance (the bit-exact fluid state) + VISUAL parity; the Vulkan-vs-Metal cross-vendor
   delta is the documented float baseline (~30–60 mean — like FPX6/CL6/scene_shadow, NOT zero), measured by
   the controller (a LARGER/structural delta or a visual mismatch is a bug).
6. **Tests `tests/fluid_test.cpp` additions (pure CPU):** `FluidVertToWorld` — a known integer pos → the
   expected float world position (scale correct); `FluidToRenderInstances` — N particles → N instance
   transforms with the right translate+scale; the sim feeding it is the existing bit-exact `StepFluid`
   (already tested). Clean under `windows-msvc-asan`. (The render is golden-verified, not unit-tested.)
7. **Introspect.** Add exactly `deterministic-fluid-render` (features) + `--fluid-render-shot` (showcases).

## RHI seam additions (summary)
- **None expected.** Reuse the existing instanced lit-sphere pipeline (the per-instance transform buffer +
  `DrawInstanced`/MDI + the lit pipeline + the frame UBO) — all pre-existing (the FPX6 set). If a
  genuinely-new RHI need surfaces, STOP and report it. `rhi.h` + `rhi_factory` (baseline 2) + backend dirs
  UNCHANGED. `engine/sim/fluid.h` FL1–FL5 + `engine/sim/fpx.h` + `engine/sim/cloth.h` + `engine/physics/`
  UNCHANGED. Report the seam.

## Out of scope (YAGNI — flagship done after this)
Metaball/marching-cubes fluid surfacing (a new shader — out; instanced spheres reuse FPX6's pipeline),
screen-space-fluid rendering, PBR/refractive water materials (flat/simple-lit droplets first), foam/spray.
Claim the deterministic fixed-point FLUID (the substance) + a lit render of it. ONE lit 3D render of the
fluid pool with the provenance + determinism + coverage + empty no-op proofs and the lit golden —
COMPLETING the 6-slice deterministic-fluid flagship.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 95) + the new `fluid_test` render-helper cases.
   Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--fluid-render-shot` on Vulkan: a coherent LIT 3D fluid pool (the deterministic
   fluid, rendered); provenance + determinism + coverage + empty no-op. Run under the Vulkan-validation
   gate → ZERO VUID in the OUTPUT (set BOTH `VK_LAYER_PATH` to the conan `...\.conan2\p\...\layers` dir AND
   `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer LOADED = zero "not found" lines).
   **VERIFY the rendered image actually shows the lit fluid droplets (pixel-check the shaded region) — do
   NOT trust the proof's "coherent" claim alone (the NAV6/CL6 lesson).**
3. Metal: `visual_test --fluid-render` → new golden `tests/golden/metal/fluid_render.png`; two runs DIFF
   0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm the reused
   instanced-lit pipeline + any shader MSL-generate.** The cross-vendor Vulkan-vs-Metal delta is the
   documented float smoke (~30–60 mean), measured by the controller — NOT strict zero-diff (unlike FL1–FL5).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fluid_render.png` added;
   the other 128 byte-identical (FL1–FL5 + all existing untouched). `git diff master --stat -- tests/golden`
   = ONLY `fluid_render.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fluid-render` + `--fluid-render-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report the reused pipeline/shader). `scripts/verify.ps1`
   updated: `fluid_render` golden in the Mac loop + `--fluid-render-shot` in `$vkShots`. FL1–FL5 +
   `engine/sim/fpx.h` + `engine/sim/cloth.h` + `engine/physics/` UNTOUCHED.
