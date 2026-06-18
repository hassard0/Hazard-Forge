# Slice CL6 — Deterministic GPU Cloth: LIT 3D RENDER CAPSTONE (the money-shot, COMPLETES flagship #8) (Phase 13 #6) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice of FLAGSHIP #8
> (DETERMINISTIC GPU CLOTH, `hf::sim::cloth`). RENDERS the bit-exact draped cloth as a lit 3D surface —
> the money-shot showing the deterministic fixed-point cloth draped over a sphere as a real lit 3D scene.
> The mesh comes from the BIT-EXACT integer cloth state (CL1–CL4 particle lattice → a float triangle
> mesh host-side, for rendering ONLY — the sim stays integer); the raster/shade is float, so this is the
> FLOAT visresolve-bar (like MC6/FPX6/NAV6): Metal-baked golden, Metal-determinism DIFF 0.0000 +
> provenance + visual parity, cross-vendor ~the float baseline. Reuses the EXISTING lit-mesh pipeline —
> NO new RHI, NO new shader. After this slice flagship #8 is COMPLETE (6/6). Branch: `slice-cl6`. See
> [[hazard-forge-cloth-roadmap]].

**Goal:** Add render-only float helpers to `engine/sim/cloth.h` (`ClothVertToWorld` = the single host
`pos/kOne` conversion; `ClothToRenderMesh` → the W×H particle lattice as a lit triangle mesh — two
triangles per cell quad, per-vertex normals averaged from the adjacent face normals — in the engine's
vertex format), run the deterministic CL1–CL4 cloth sim to a draped state (over a sphere), and render the
cloth as a lit 3D surface through the EXISTING lit-mesh pipeline (the FPX6/MC6 path) from a fixed 3/4
camera + directional light (+ the sphere, optionally, as a second lit mesh). Add `--cloth-render-shot`
(Vulkan) / `--cloth-render` (Metal). Bake the lit golden `cloth_render`. Make-safe: small host float
helpers (CL1–CL5 sim UNCHANGED — only float accessors for rendering) + a NEW showcase + NEW golden; reuse
the existing lit-mesh pipeline — NO new shader/RHI.

## Design call: FLOAT visresolve-bar (the cloth arc's only float slice)
CL1–CL5 are strict integer/bit-exact (the cloth sim is fixed-point). CL6 RASTERIZES the draped state with
a perspective camera + lighting → cross-vendor float (the MC6/FPX6/NAV6/visresolve caveat). So CL6's
golden is the FLOAT bar, identical to FPX6/MC6: the committed golden is **baked on Metal**; the gate is
**Metal-render == Metal-golden DIFF 0.0000** (deterministic, two-run) + **provenance** (the rendered mesh
is built from the bit-exact cloth `ClothParticle` lattice — `pos/kOne` + the per-vertex normals,
deterministic host code) + visual parity, with the Vulkan-vs-Metal delta the documented cross-vendor smoke
(~45–60 mean, the engine's float-render baseline — NOT zero). The fixed-point SIM feeding the render is
still bit-exact (CL1–CL5); only the final raster/shade is float.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The cloth state (the input):** `engine/sim/cloth.h` — `ClothParticle`/`ClothGrid` (CL1),
  `StepClothCollide` (CL4 — run the sim to a draped state over a sphere). World position = `pos.xyz /
  (float)kOne` (the ONE host float conversion). Per-vertex normal = the normalized average of the adjacent
  cell-quad face normals (float, render-only).
- **The existing lit-mesh pipeline to REUSE (find + mirror — do NOT invent):** FPX6's `RunFpxRenderShowcase`
  (the instanced/lit pipeline) and especially **MC6/MC5's `RunMcRenderShowcase`** — a single LIT MESH
  render (`lit.vert` + `lit.frag`, `scene::MeshVertexLayout`, the FrameData camera/light UBO, sky +
  static-shadow + post — REUSED VERBATIM from the terrain/MC path). The cloth is ONE lit mesh (a W×H grid
  of quads with per-vertex normals) → MC6's single-mesh lit render is the EXACT structural template.
  Report which pipeline/shader was reused. **(NOTE the NAV6 lesson: this is a LIT MESH, not debug lines —
  the lit cloth surface will fill many pixels, so the visual is robust; still, VERIFY the rendered output
  actually shows a lit cloth surface, not just the ground — pixel-check the shaded region.)**
- **The float-golden discipline (the bar to copy):** MC6 / FPX6 / NAV6 — per-backend Metal golden,
  determinism, provenance, the cross-vendor ~45–60 baseline (NOT zero-diff).
- **The CL4 draped scene:** the `--cloth-collide-shot` scene (cloth over a sphere) — reuse it as the
  render subject (a draped cloth is a more compelling money-shot than a flat sheet).
- **Showcase + registration:** the FPX6/MC6 `--*-render-shot` template; `verify.ps1`/`introspect.cpp`/
  `introspect_test.cpp`. main.cpp has `/bigobj` (CL4).

## Design decisions (locked)
1. **Run the sim to a draped state, then render.** Reuse the CL4 scene (a W×H cloth dropped onto a static
   `FxBody` sphere, top corners or top edge pinned), `StepClothCollide` K steps (under the TDR-safe budget
   if dispatched on GPU — but CL6 can run the SIM on the CPU host-side since the render only needs the
   final float mesh; the sim is bit-exact either way) → the draped cloth. Build `ClothToRenderMesh`: a
   vertex per particle (`pos/kOne`, per-vertex normal), two triangles per cell quad (W-1)×(H-1). Pure
   deterministic host float (render-only, the sim stays integer).
2. **Render through the EXISTING lit-mesh pipeline — NO new RHI, NO new shader.** Reuse the lit-mesh
   `GraphicsPipelineDesc` (`lit.vert`+`lit.frag`, `MeshVertexLayout`, the frame/camera UBO, sky + shadow +
   post — the MC6 path), a fixed 3/4 camera + directional light. Optionally render the collider sphere as
   a second lit mesh (a unit sphere scaled to the collider) so the drape reads. Double-sided / two-sided
   if the cloth back-faces show (document). Report exactly which pipeline/shader was reused; if a
   genuinely-new shader is needed, flag it loudly (prefer reuse).
3. **Showcase `--cloth-render-shot <out>` (Vulkan, main.cpp) AND `--cloth-render` (Metal, visual_test.mm —
   WIRE BOTH; confirm visual_test.mm + `#include "sim/cloth.h"`).** Run the deterministic draped-cloth sim
   → float mesh → lit render → the BGRA8 image. Golden = the lit 3D draped cloth → `tests/golden/metal/
   cloth_render.png` (baked on the Mac by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance (the cloth IS bit-exact):** the rendered mesh's vertices are built from the
     `ClothParticle` lattice that equals the CPU `StepClothCollide` reference (the same bit-exact CL1–CL4
     sim — print the particle/triangle counts + that the mesh derives from `pos/kOne` + per-vertex
     normals). Print `cloth-render: {particles:<N>, tris:<T>, shaded:<S>} (fixed-point cloth -> lit 3D
     render)`.
   - **(2) determinism (same backend):** two renders → DIFF 0.0000. Print `cloth-render determinism: two
     renders BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the lit cloth covers a coherent region (`shaded > 0`, not uniform — a
     recognizable draped surface). Print `cloth-render coverage: <S> shaded (coherent lit drape)`.
   - **(4) empty no-op:** a degenerate/empty cloth → the cleared background / ground only. Print
     `cloth-render empty: base only (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/cloth_render.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 122 image goldens UNTOUCHED.
5. **Cross-backend bar (FLOAT, per the MC6/FPX6 finding):** Metal-output == Metal-golden DIFF 0.0000
   (determinism) + provenance (the bit-exact cloth state) + VISUAL parity; the Vulkan-vs-Metal cross-vendor
   delta is the documented float baseline (~45–60 mean — like FPX6/NAV6/scene_shadow, NOT zero), measured
   by the controller (a LARGER/structural delta or a visual mismatch is a bug).
6. **Tests `tests/cloth_test.cpp` additions (pure CPU):** `ClothVertToWorld` — a known integer pos → the
   expected float world position (scale correct); `ClothToRenderMesh` — a W×H lattice → (W-1)(H-1)·2
   triangles + W·H vertices with plausible normals (a flat sheet → all normals ≈ the sheet normal); the
   sim feeding it is the existing bit-exact `StepClothCollide` (already tested). Clean under
   `windows-msvc-asan`. (The render is golden-verified, not unit-tested.)
7. **Introspect.** Add exactly `deterministic-cloth-render` (features) + `--cloth-render-shot` (showcases).

## RHI seam additions (summary)
- **None expected.** Reuse the existing lit-mesh pipeline (the MC6 path) + the frame UBO — all pre-existing.
  If a genuinely-new RHI need surfaces, STOP and report it. `rhi.h` + `rhi_factory` (baseline 2) + backend
  dirs UNCHANGED. `engine/sim/cloth.h` CL1–CL5 + `engine/sim/fpx.h` + `engine/physics/` UNCHANGED. Report
  the seam.

## Out of scope (YAGNI — flagship done after this)
PBR/textured cloth materials (flat/simple-lit first), cloth wind animation in the render, self-collision /
dynamic colliders (the documented sim gaps). Claim the deterministic fixed-point CLOTH (the substance) + a
lit render of it. ONE lit 3D render of the draped cloth with the provenance + determinism + coverage +
empty no-op proofs and the lit golden — COMPLETING the 6-slice deterministic-cloth flagship.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 94) + the new `cloth_test` render-helper cases.
   Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--cloth-render-shot` on Vulkan: a coherent LIT 3D draped cloth (the deterministic
   fixed-point cloth, rendered); provenance + determinism + coverage + empty no-op. Run under the
   Vulkan-validation gate → ZERO VUID in the OUTPUT (set BOTH `VK_LAYER_PATH` to the conan
   `...\.conan2\p\...\layers` dir AND `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer
   LOADED = zero "not found" lines). **VERIFY the rendered image actually shows the lit cloth surface
   (pixel-check the shaded region / visual review) — do NOT trust the proof's "coherent" claim alone (the
   NAV6 lesson).**
3. Metal: `visual_test --cloth-render` → new golden `tests/golden/metal/cloth_render.png`; two runs DIFF
   0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm the reused lit-mesh
   pipeline + any shader MSL-generate.** The cross-vendor Vulkan-vs-Metal delta is the documented float
   smoke (~45–60 mean), measured by the controller — NOT strict zero-diff (unlike CL1–CL5).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cloth_render.png` added;
   the other 122 byte-identical (CL1–CL5 + all existing untouched). `git diff master --stat -- tests/golden`
   = ONLY `cloth_render.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cloth-render` + `--cloth-render-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report the reused pipeline/shader). `scripts/verify.ps1`
   updated: `cloth_render` golden in the Mac loop + `--cloth-render-shot` in `$vkShots`. CL1–CL5 +
   `engine/sim/fpx.h` + `engine/physics/` UNTOUCHED.
