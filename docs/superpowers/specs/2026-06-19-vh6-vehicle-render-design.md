# Slice VH6 — Deterministic Vehicle Physics: LIT 3D INSTANCED RENDER CAPSTONE (THE MONEY-SHOT) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH + FINAL slice of FLAGSHIP #16
> (DETERMINISTIC VEHICLE PHYSICS, `hf::sim::vehicle`) — the LIT 3D RENDER CAPSTONE, the money-shot, the FPX6/FL6/
> GR6/CP6/CG6/GF6/FR6/JT6 precedent. VH1-VH5 built + proved a deterministic drivable, lockstep-replayable car.
> VH6 RENDERS the bit-exact driven car as a lit 3D scene: a matte car (a box chassis + four round wheels) driving
> on the ground under a directional light. **FLOAT render-only** — the SIM stays bit-exact integer; only the final
> per-body transform crosses to float (the ONE float crossing, render-only, OUT of the bit-exact sim path).
> **NO new shader, NO new RHI** — reuses the EXISTING lit-instanced pipeline VERBATIM. Branch: `slice-vh6`. See
> [[hazard-forge-vehicle-roadmap]].

**Goal:** Extend `engine/sim/vehicle.h` (additive — VH1-VH5 byte-unchanged) with `VehicleToRenderInstances` (the
bit-exact `Vehicle` → a FLOAT instance set via `fpx::FxBodyTransform`, render-only). Add `--vehicle-render-shot`
(Vulkan) / `--vehicle-render` (Metal) that run the bit-exact driven sim, build the instances, and draw a lit 3D
car through the EXISTING lit-instanced pipeline. Bake the FLOAT golden `vehicle_render`. **NO new shader, NO new
RHI.**

## Design call: the bit-exact driven car → float instances → the existing lit-instanced pipeline (matte)
The deterministic-sim flagships all close with a lit 3D render capstone on a FLOAT visresolve-bar: the integer sim
is bit-exact, and ONLY the final raster/shade is float, so the golden is Metal-baked and the gate is
Metal-determinism (two renders BYTE-IDENTICAL) + provenance (every instance transform IS a pure function of the
bit-exact body state) + visual parity; the Vulkan-vs-Metal cross-vendor delta is the documented float baseline
(~24-55 mean, NOT integer zero). VH6 is that capstone for the vehicle.

**`VehicleToRenderInstances(v, cfg)`** runs OUTSIDE the bit-exact loop (render-only): the car has driven to a pose
(K `StepVehicleDrivenSteps`, the SAME scene the Vulkan `--vehicle-render-shot` AND the Metal `--vehicle-render`
build, so the integer state is byte-identical cross-backend by construction). For each body it builds a FLOAT
model matrix via `fpx::FxBodyTransform` (REUSED VERBATIM — the FPX6/FR6/JT6 precedent) composed with a per-body
SHAPE scale:
- **The chassis** (1 body): `FxBodyTransform(chassis)` × a non-uniform scale to the chassis half-extents
  (`chassisHalfX/Y/Z`) → a BOX. Drawn with the cube mesh.
- **The 4 wheels** (4 bodies): `FxBodyTransform(wheel)` × a uniform `wheelRadius` scale (optionally flattened in
  the lateral axis to read as a wheel, not a ball) → ROUND wheels. Drawn with the sphere mesh.

Split into two instance sets (chassis vs wheels) so the showcase draws TWO colored instanced draws (the GF6/FR6
two-draw pattern): the chassis in a warm matte CAR-PAINT colour, the wheels in a dark matte TYRE colour. **MATTE
(roughness 1.0)** so the bodies do NOT mirror the sky IBL into iridescence (the GF6/FR6/JT6 hard lesson — the
implementer MUST set high roughness + a warm/dark albedo, NOT a low-roughness metal). The ONE float crossing is
`FxBodyTransform` + the scale compose; the sim path stays pure integer.

**The render is the EXISTING lit-instanced pipeline REUSED VERBATIM:** `lit_instanced.vert` + `lit.frag` +
`scene::InstanceTransformLayout`, the FrameData camera/light UBO, sky + instanced/static shadow + post — the SAME
path `--fpx-render-shot`/`--fract-render-shot`/`--joint-render-shot` use. Two instanced draws (the cube-mesh
chassis, then the sphere-mesh wheels) over the ground from a fixed 3/4 camera + a directional light. NO new
shader, NO new RHI, NO new pipeline — only a new showcase + the render-only helper.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **VH1-VH5 (this branch's `vehicle.h`, read-only — build on, DON'T modify):** `struct Vehicle`/`VehicleConfig`/
  `VehicleFromConfig`, `StepVehicleDriven`/`StepVehicleDrivenSteps` (run the car to its driven pose), the VH3
  command kinds + `ApplyVehicleCommand`. DO NOT modify any of these. VH6 APPENDS after the VH5 block.
- **fpx render bridge to REUSE VERBATIM (`engine/sim/fpx.h`, read-only):** `fpx::FxBodyTransform(const FxBody&)`
  (the Q16.16 pos/orient → float model matrix, render-only — the FPX6 helper every capstone uses). **DO NOT
  modify fpx.h.**
- **The render capstone precedent to MIRROR (`samples/hello_triangle/main.cpp`):** study `--fpx-render-shot`
  (FPX6 — the canonical body-pile lit render: FxBodyTransform per body → instanced spheres), `--fract-render-shot`
  (FR6 — the TWO-draw colour split: static anchor vs dynamic rubble, MATTE roughness to dodge iridescence), and
  `--joint-render-shot` (JT6 — the provenance proof shape). Copy their camera/light/sky/shadow/post setup + the
  instanced-draw plumbing VERBATIM. Find the cube mesh + the sphere mesh the instanced pipeline binds (the
  pipeline is mesh-agnostic — it binds a vertex+index buffer + the per-instance transform SSBO; the chassis draw
  binds the cube mesh, the wheels draw the sphere mesh). If a cube instanced mesh is not readily available,
  fall back to a non-uniformly-scaled sphere for the chassis and FLAG it (but prefer the box — it reads as a car).
- **The render-only instance helper shape (`grain.h::GrainToRenderInstances` / `fract.h::FractToRenderInstances` /
  `couple_gf.h::CGFToRenderInstances`):** the host float `std::vector<scene::InstanceTransform>` (or the actual
  instance struct) builder from bit-exact body state. `VehicleToRenderInstances` is the same shape with the
  chassis-box + wheel-sphere split.
- **Showcase + registration:** the `--fpx-render-shot` plumbing — **standalone arg-parse loop** (the FR1 C1061
  lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), `tests/vehicle_test.cpp`.

## Design decisions (locked)
1. **`VehicleToRenderInstances(const Vehicle& v, const VehicleConfig& cfg, <out chassis instances>, <out wheel
   instances>)`** (or returns a struct with both) — the chassis-box transform + the 4 wheel transforms, each
   `FxBodyTransform` × the shape scale, FLOAT, render-only. A `RebuildVehicleInstances`-equivalent is just calling
   it again (for the provenance proof). Pure function of the bit-exact `Vehicle` state.
2. **Showcase `--vehicle-render-shot <out>` (Vulkan) AND `--vehicle-render` (Metal) — WIRE BOTH** (standalone
   arg-parse). Build a car (`VehicleFromConfig`), feed a drive+steer stream, run K `StepVehicleDrivenSteps` to a
   driven pose (the SAME scene both backends build → byte-identical integer state), build the instances, draw the
   TWO colored instanced draws (matte warm chassis cube, matte dark wheel spheres) through the EXISTING
   lit-instanced pipeline (camera/light/sky/shadow/post REUSED) over the ground. Golden =
   `tests/golden/metal/vehicle_render.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) instances from bit-exact state:** print `vehicle-render: {bodies:<N>, instances:<M>} from bit-exact
     driven state` (M == 1 chassis + 4 wheels == 5; the instance transforms ARE the bit-exact body poses).
   - **(2) determinism:** two renders → BYTE-IDENTICAL (the Metal-determinism gate). Print `vehicle-render
     determinism: two runs BYTE-IDENTICAL`.
   - **(3) provenance:** the built instance set == a rebuild from the bit-exact `Vehicle` state (`VehicleToRender
     Instances` is a pure function — call it twice, assert byte-equal). Print `vehicle-render provenance: instances
     == rebuild`.
   - **Golden discipline: ONLY `tests/golden/metal/vehicle_render.png`; do NOT commit it.** Existing 170 image
     goldens UNTOUCHED.
4. **Cross-backend bar (FLOAT, the visresolve precedent):** the SIM is bit-exact but the final raster/shade is
   float → the golden is Metal-baked; the gate is Metal-determinism (two renders BYTE-IDENTICAL) + provenance +
   visual parity (a coherent lit 3D car). The Vulkan-vs-Metal cross-vendor delta is the documented float baseline
   (~24-55 mean, NOT integer zero) — the CONTROLLER confirms it is in-band, NOT a strict-zero compare.
5. **Tests `tests/vehicle_test.cpp` additions (pure CPU):** `VehicleToRenderInstances` — produces 5 instances (1
   chassis + 4 wheels); the chassis instance's translation == the chassis body's `FxBodyTransform` translation;
   each wheel instance's translation == its wheel body's; the helper is a PURE FUNCTION (two calls byte-equal —
   the provenance contract). Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-vehicle-render` (features) + `--vehicle-render-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the EXISTING lit-instanced pipeline + the offscreen render path VERBATIM. `rhi.h` + backend dirs
  UNCHANGED. `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` +
  `engine/anim/` + `engine/physics/` + all existing shaders + `hf_gen_msl` UNCHANGED (VH6 reuses
  `lit_instanced.vert` + `lit.frag` — NO new shader). VH1-VH5 `vehicle.h` code + the VH2-VH5 showcases UNCHANGED
  (VH6 additive — only the render-only helper + the showcase). Report the seam empty.

## Out of scope (YAGNI — flagship #16 is COMPLETE after VH6)
Cylindrical wheel geometry / a real car mesh / a glTF model (the wheels are spheres, the chassis a box — the
deterministic-sim-capstone convention: lit primitives, not art assets), tyre tread, brake lights, a driver,
motion blur, a skid trail. VH6 claims ONLY: a lit 3D render of the bit-exact driven car (box chassis + 4 wheels,
matte) through the existing pipeline, Metal-deterministic + provenance-checked + visually coherent, the
cross-vendor float baseline in-band. **Completes the 6-slice deterministic-vehicle-physics flagship — the
SIXTEENTH flagship.**

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 102) + the new `vehicle_test` render-instance cases. Clean
   under `windows-msvc-asan` (build+run `vehicle_test` + `introspect_test`).
2. **proofs + visual:** `--vehicle-render-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate
   → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED — the lit-instanced
   pipeline binds the per-frame shadow map, the GI-path VUID lesson). **VERIFY the image shows a coherent lit 3D
   car — a matte box chassis on four round wheels on the ground, NOT iridescent, NOT black/scrambled (the HARD
   visual gate; the GF6/FR6 iridescence + the FPX6 coherence lesson).** ALSO re-run `--vehicle-lockstep-shot` +
   `--vehicle-traction-shot` + `--vehicle-step-shot` + `--vehicle-rig-shot` + `--vehicle-spring-shot` → still
   bit-exact (VH1-VH5 render-invariance).
3. Metal: `visual_test --vehicle-render` → new golden `tests/golden/metal/vehicle_render.png`; **two runs DIFF
   0.0000** (the FLOAT-capstone gate is Metal-DETERMINISM, gate on `compare.sh` EXIT CODE). **Confirm
   `visual_test.mm` in the diff; confirm NO new shader (`hf_gen_msl` UNCHANGED).** The Vulkan-vs-Metal cross-vendor
   compare is the FLOAT baseline (~24-55 mean) — the controller confirms it is in-band (NOT strict zero).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `vehicle_render.png` added; the
   other 170 byte-identical. `git diff master --stat -- tests/golden` = ONLY `vehicle_render.png` (metal) (the
   introspect json rebake = controller, post-gate).
5. Introspect: exactly `+deterministic-vehicle-render` + `--vehicle-render-shot` added in introspect.cpp;
   introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`joint.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/
   `fract.h` + `engine/anim/` + `engine/physics/` + VH1-VH5 `vehicle.h` (the pre-VH6 functions) byte-unchanged +
   ALL existing shaders byte-unchanged). `scripts/verify.ps1` updated: `vehicle_render` golden in the Mac loop +
   `--vehicle-render-shot` in `$vkShots`. **NO new entry in `hf_gen_msl`.**
