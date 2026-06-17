# Slice FPX6 — Deterministic Fixed-Point Physics: LIT 3D RENDER (the money-shot) (Phase 11 #6) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The 6th and FINAL FPX slice (completing flagship #6):
> RENDER the settled fixed-point physics state as lit 3D spheres — the money-shot showing the deterministic
> bit-exact physics as a real lit 3D scene. The mesh transforms come from the BIT-EXACT fpx state (`FxBody` pos +
> orient → a float mat4 host-side, for rendering ONLY — the sim stays integer); the raster/shade is float, so this is
> the FLOAT visresolve-bar (like MC5/FPX has had no render yet): Metal-baked golden, Metal-determinism DIFF 0.0000 +
> provenance (the rendered transforms ARE the fpx bit-exact state) + visual parity, cross-vendor ~the float baseline.
> Reuses the EXISTING instanced/lit mesh pipeline — NO new RHI, NO new shader. Namespace `hf::sim::fpx`. Branch:
> `slice-fpx-render`. After this slice flagship #6 (Deterministic Fixed-Point Physics) is COMPLETE (6/6). See
> [[hazard-forge-fpx-roadmap]].

**Goal:** Build per-body float render transforms from the bit-exact fpx state (`FxBodyTransform` → a `math::Mat4`
from `pos/kSub` + the quaternion→matrix of `orient`), run a deterministic physics sim to a settled pile, render the
bodies as lit 3D instanced spheres through the EXISTING instanced/lit pipeline from a 3/4 camera, and bake the lit
golden. Add a `--fpx-render-shot` (Vulkan) / `--fpx-render` (Metal) showcase. Make-safe: a small host transform
helper (FPX1–FPX5 sim UNCHANGED — only a float-mat4 accessor for rendering) + a NEW showcase + NEW golden; ideally NO
new shader (reuse the existing instanced lit mesh shader).

## Design call: float visresolve-bar (the FPX arc's only float slice)
FPX1–FPX5 are strict integer/bit-exact (the sim is fixed-point). FPX6 RASTERIZES the state with a perspective camera +
lighting → cross-vendor float (the visbuffer/visresolve/MC5 caveat). So FPX6's golden is the FLOAT bar, identical to
MC5: the committed golden is **baked on Metal**; the gate is **Metal-render == Metal-golden DIFF 0.0000**
(deterministic, two-run) + **provenance** (the rendered per-instance transforms are built from the fpx bit-exact
`FxBody` state — `pos/kSub` + the quaternion-to-matrix of `orient`, deterministic host code) + visual parity, with
the Vulkan-vs-Metal delta the documented cross-vendor smoke (~55–60 mean, the engine's float-render baseline — NOT
zero, NOT held to the integer bar). The fixed-point SIM feeding the render is still bit-exact (FPX1–FPX5); only the
final raster/shade is float.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The fpx state (the input):** `engine/sim/fpx.h` — `FxBody` (pos/vel/orient/angVel), `StepWorld`/`IntegrateBodyFull`/
  `SimTick` (run the sim to a settled state), `kSub`/`kFrac`. World position = `pos.xyz / (float)kSub` (the ONE host
  float conversion); orientation = `orient.xyzw / (float)kSub` → a normalized float quaternion → a rotation `math::Mat4`.
- **The existing instanced/lit mesh pipeline to REUSE (find + mirror — do NOT invent):** locate the engine's instanced
  lit-mesh render — the **instanced-rendering** showcase (`2026-06-14-instanced-rendering-design.md`, the `instanced`
  golden) and/or the GPU-driven `--mdi`/`--gpudriven` path render many transformed mesh instances through a lit shader
  (per-instance model matrix in an instance buffer / SSBO). Study its EXACT wiring (the sphere/cube `MeshGeometry`,
  the per-instance transform buffer, the lit pipeline desc, the frame/camera UBO, `DrawInstanced`/MDI) in
  `samples/hello_triangle/main.cpp` + `metal_headless/visual_test.mm`, and reuse it: a unit `SphereGeometry`, one
  instance per `FxBody` with its `FxBodyTransform` (scaled by the body's `radius/kSub`), a fixed 3/4 camera + a
  directional light. Report which pipeline/shader was reused. (MC5 reused the terrain static-lit mesh pipeline the
  same way — `RunMcRenderShowcase` is a structural template for a single-mesh lit render; here it's INSTANCED.)
- **Quaternion→matrix:** build the rotation `Mat4` from the float quaternion (the standard `q → R` formula) host-side
  in the showcase (or add `FxQuatToMat`/`FxBodyTransform` to a render-only helper — keep it OUT of the bit-exact sim
  path; it's float, for rendering only).
- **The float-golden discipline (the bar to copy):** the MC5 / `visresolve` spec — per-backend Metal golden,
  determinism, provenance, the cross-vendor ~55–60 baseline (NOT zero-diff).
- **Showcase + registration:** the MC5 `--mc-render-shot` template; `verify.ps1`/`introspect.cpp`/`introspect_test.cpp`.

## Design decisions (locked)

1. **Run the sim to a settled pile (deterministic), then render.** Use a scene that produces a VISIBLE pile (address
   the sparse-viz note): e.g. a tight cluster of N spheres (e.g. 32–64) dropped onto the ground, `SimTick`/`StepWorld`
   K steps with enough solve iterations so they settle into a packed mound; OR a confined drop. The bodies' final
   `FxBody` state is the bit-exact sim output. Build a `math::Mat4` per body: translate by `pos/kSub`, rotate by the
   quaternion-to-matrix of `orient`, scale by `radius/kSub`. Pure deterministic host code (float, render-only).
2. **Render through the EXISTING instanced/lit pipeline — NO new RHI, ideally NO new shader.** Reuse the instanced
   lit-mesh `GraphicsPipelineDesc` + the per-instance transform buffer + the frame/camera UBO; a unit `SphereGeometry`,
   N instances, a fixed 3/4 camera + directional light + the ground plane (reuse the existing ground/sky if convenient).
   Report exactly which pipeline/shader was reused; if a genuinely-new shader is needed, flag it loudly (prefer reuse).
3. **Showcase `--fpx-render-shot <out>` (Vulkan, main.cpp) AND `--fpx-render` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm + `#include "sim/fpx.h"`).** Run the deterministic pile sim → per-body transforms → instanced
   lit render → the BGRA8 image. Golden = the lit 3D pile of spheres → `tests/golden/metal/fpx_render.png` (baked on
   the Mac by the CONTROLLER — DO NOT commit). 
4. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance (the fpx state IS bit-exact):** the per-instance transforms are built from the `FxBody` state
     that equals a CPU `StepWorld`/`SimTick` reference (or simply: the sim is the same bit-exact FPX1–FPX5 sim — print
     the settled body count + that the transforms derive from `pos/kSub`+`orient`). Print `fpx-render: {bodies:<N>,
     settled:<m>, shaded:<P>} (fixed-point sim -> lit 3D render)`.
   - **(2) determinism (same backend):** two renders → DIFF 0.0000. Print `fpx-render determinism: two renders
     BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the lit spheres cover a coherent region (`shaded > 0`, not uniform — a recognizable
     pile). Print `fpx-render coverage: <P> shaded (coherent lit pile)`.
   - **(4) empty no-op:** zero bodies → the cleared background. Print `fpx-render empty: background (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/fpx_render.png`; do NOT commit it — the CONTROLLER bakes on the
     Mac.** Existing 110 image goldens UNTOUCHED.
5. **Cross-backend bar (FLOAT, per the MC5 finding):** Metal-output == Metal-golden DIFF 0.0000 (determinism) +
   provenance (the bit-exact fpx state) + VISUAL parity; the Vulkan-vs-Metal cross-vendor delta is the documented
   float baseline (~55–60 mean — like MC5/scene_shadow, NOT zero), measured by the controller (a LARGER/structural
   delta or a visual mismatch is a bug).
6. **Tests `tests/fpx_test.cpp` additions (pure CPU):** `FxBodyTransform` — identity orientation → a pure
   translate+scale; a known orientation → the expected rotation matrix (within fp tol); `pos/kSub` correct; the sim
   feeding it is the existing bit-exact `StepWorld` (already tested). Clean under `windows-msvc-asan`. (The render is
   golden-verified, not unit-tested.)
7. **Introspect.** Add exactly `deterministic-fixedpoint-physics-render` (features) + `--fpx-render-shot` (showcases).

## RHI seam additions (summary)
- **None expected.** Reuse the existing instanced/lit mesh pipeline (the per-instance transform buffer +
  `DrawInstanced`/MDI + the lit pipeline + the frame UBO) — all pre-existing. If a genuinely-new RHI need surfaces,
  STOP and report it. `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. `engine/physics/` + the FPX1–FPX5
  sim UNCHANGED. Report the seam.

## Out of scope (YAGNI — flagship done after this)
PBR/textured materials (flat/simple-lit first), a box-SAT collider, a real network transport, shadows/post on the
physics scene beyond what the reused pipeline already does. Claim DETERMINISM (the sim) + a lit render of the
bit-exact state. ONE lit 3D render of the settled fixed-point pile with the provenance + determinism + coverage +
empty no-op proofs and the lit-pile golden — COMPLETING the 6-slice deterministic-fixed-point-physics flagship.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 92) + the new `fpx_test` transform cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--fpx-render-shot` on Vulkan: a coherent LIT 3D pile of spheres (the settled fixed-point
   physics state, rendered); provenance + determinism + coverage + empty no-op. Run under the Vulkan-validation gate →
   ZERO VUID in the OUTPUT.
3. Metal: `visual_test --fpx-render` → new golden `tests/golden/metal/fpx_render.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm the reused instanced/lit pipeline + any shader
   MSL-generate.** The cross-vendor Vulkan-vs-Metal delta is the documented float smoke (~55–60 mean), measured by the
   controller — NOT strict zero-diff (unlike FPX1–FPX5).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fpx_render.png` added; the other 110
   byte-identical (FPX1–FPX5 + all existing untouched). `git diff master --stat -- tests/golden` = ONLY
   `fpx_render.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fixedpoint-physics-render` + `--fpx-render-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report the reused pipeline/shader). `scripts/verify.ps1` updated:
   `fpx_render` golden in the Mac loop + `--fpx-render-shot` in `$vkShots`.
