# Slice MC5 — GPU Isosurface Meshing: RENDER the generated mesh (lit 3D surface) (Phase 10 #5) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The 5th MC slice (after MC1–MC4 produced a bit-exact
> triangle mesh): RENDER the extracted isosurface as a lit 3D surface through the engine's existing mesh/lit pipeline
> — the capstone that demonstrates the whole flagship end-to-end (voxel/SDF field → GPU-meshed → rendered geometry).
> This is the MC arc's FIRST FLOAT slice: a perspective-projected, lit raster render is cross-vendor float (the
> visresolve/SW4 bar — NOT the strict zero-diff integer bar of MC1–MC4), so the verification posture is the documented
> float bar: a per-backend Metal-baked golden (Metal-output == Metal-golden DIFF 0.0000, deterministic) + a GPU==CPU
> interior/provenance proof + a cross-vendor ≤eps smoke (explicitly not 0.0000). NO new RHI (reuse the existing lit
> mesh pipeline). Namespace `hf::render::mc`. Branch: `slice-mc5-render`. See [[hazard-forge-mc-roadmap]].

**Goal:** Build a renderable `MeshGeometry` from MC4's bit-exact interpolated mesh (fixed-point verts → float world
positions + host-computed flat face normals), render it lit from a fixed camera through the EXISTING lit mesh pipeline,
and bake a golden of the lit extracted sphere. Add a `--mc-render-shot` (Vulkan) / `--mc-render` (Metal) showcase.
Make-safe: NEW host wiring + a NEW showcase + NEW golden; MC1–MC4 + everything else UNCHANGED; ideally NO new shader
(reuse the existing lit/flat mesh shader).

## Design call: float visresolve-bar (stated honestly)
MC1–MC4 are strict integer bit-identical because their outputs are integer buffers (case indices, counts, vertex
lattice coords). MC5 RASTERIZES the mesh with a perspective camera + lighting → the rendered image is float and
cross-vendor-divergent (the hardware rasterizer's edge/fill rules + float shading differ by ≤1 LSB across vendors —
the exact caveat `visbuffer`/`visresolve`/`SW4` documented). So MC5's golden is held to the FLOAT bar, identical to
those slices: the committed golden is **baked on Metal**; the gate is **Metal-output == Metal-golden DIFF 0.0000**
(deterministic, two-run) + a per-backend **GPU==CPU or provenance** proof + a measured **cross-vendor smoke** (Vulkan
capture vs Metal golden ≤ a small pixel-fraction, documented, NOT 0). The MESH feeding the render is still the MC4
bit-exact mesh (provenance is exact); only the final raster/shade is float.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **MC4 mesh (the input):** `engine/render/mc.h` — `MarchCellsInterp` → `std::vector<McVertex>` (fixed-point 1/kSub)
  + index buffer. World position = `vert.xyz / (float)kSub` (the ONE host float conversion). `kSub`, `MarchCells*`.
- **The existing lit MESH pipeline to REUSE (find + mirror — do NOT invent a new one):** locate an existing showcase
  that uploads a CPU `MeshGeometry`/`TerrainMesh` (positions + normals + indices) and renders it lit — the scout
  cited `scene/mesh.h` (`MeshGeometry{verts,indices}`, `CubeGeometry`/`SphereGeometry`) and `terrain/heightmap.h:34-48`
  (`TerrainMesh{verts,indices}` → "the existing lit/PBR upload, no new shader required"). Study the terrain or a basic
  lit-mesh showcase's EXACT wiring (vertex layout, the lit pipeline desc, the camera/frame UBO, `BindVertexBuffer`/
  `BindIndexBuffer`/`DrawIndexed`) in `samples/hello_triangle/main.cpp` + `metal_headless/visual_test.mm`, and reuse
  it verbatim for the MC mesh. **If the existing lit vertex format needs normals/UVs, compute flat per-face normals
  host-side** (each triangle's 3 verts get `normalize(cross(p1-p0, p2-p0))`; a triangle soup makes this trivial — no
  shared-vertex averaging) and a constant/dummy UV. NO new shader if the existing lit/flat-shaded mesh shader fits.
- **FP discipline for any GPU==CPU interior proof:** `visresolve.h:82-92` (`std::fma` + host-precomputed) if a CPU
  shade mirror is used for the interior proof.
- **The float-golden discipline (the bar to copy):** the `visresolve`/`swraster_resolve` specs — per-backend Metal
  golden, determinism, interior GPU==CPU, cross-vendor smoke (NOT 0.0000).
- **Showcase + registration:** the MC4 `--mc-interp-shot` template (scene build + dispatch + write); `scripts/verify.ps1`
  `$Goldens`/`$vkShots`, `engine/editor/introspect.cpp`, `tests/introspect_test.cpp`.

## Design decisions (locked)

1. **Build the render mesh host-side (deterministic).** `MarchCellsInterp` → fixed-point verts + indices; convert to
   the existing lit pipeline's vertex format: `position = vert.xyz / (float)kSub` (world units), `normal` = the flat
   per-face normal (host `normalize(cross(...))`, `std::fma` for the interior proof if needed), a constant UV/color.
   Center + scale the mesh into the camera frame. This host build is pure + deterministic (same mesh every run).
2. **Render through the EXISTING lit mesh pipeline — NO new RHI, ideally NO new shader.** Reuse the lit/flat mesh
   `GraphicsPipelineDesc` + the frame/camera UBO the terrain/mesh showcase uses; `BindVertexBuffer` + `BindIndexBuffer`
   + `DrawIndexed` the MC mesh; a fixed camera + a fixed directional light. If the existing lit shader requires a
   feature the MC mesh lacks, add the SMALLEST reuse (e.g. a flat-normal flag) — but PREFER an existing shader. Report
   exactly which pipeline/shader was reused.
3. **Showcase `--mc-render-shot <out>` (Vulkan, main.cpp) AND `--mc-render` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm + `#include "render/mc.h"`).** SAME MC1–MC4 sphere field → `MarchCellsInterp` → the render
   mesh → render lit → the BGRA8 image. Golden = the lit extracted sphere → `tests/golden/metal/mc_render.png`
   (baked on the Mac). **GOLDEN DISCIPLINE: ONLY `tests/golden/metal/mc_render.png`; do NOT commit it — the CONTROLLER
   bakes it on the Mac.**
4. **PROOFS (fail loudly; exact lines):**
   - **(1) mesh provenance (exact):** the rendered mesh's vertex+index buffers equal `MarchCellsInterp` (the MC4
     bit-exact mesh) — `verts:15720 indices:15720` for the sphere; the render consumes the proven geometry. Print
     `mc-render mesh: {tris:5240, verts:15720} (MC4 bit-exact mesh)`.
   - **(2) determinism (same backend):** two renders → DIFF 0.0000 (deterministic raster of a fixed mesh+camera).
     Print `mc-render determinism: two renders BYTE-IDENTICAL`.
   - **(3) GPU==CPU interior OR coverage:** at deterministically-chosen interior pixels covered by the mesh, the
     rendered shade matches a CPU flat-Lambert mirror (the `visresolve` interior bit-exact discipline, `std::fma`), OR
     (if a full CPU raster mirror is too heavy) a coverage/provenance check (the lit sphere covers a coherent disk,
     `shaded > 0`, not uniform). Print `mc-render GPU==CPU @interior: <k>/<k> EXACT` (or the coverage line).
   - **(4) disabled / empty no-op:** an empty field (`iso` above the field max → 0 tris) → the cleared background.
     Print `mc-render empty-field: background (no-op)`.
   - **(5) {stats}:** `mc-render: {tris:5240, shaded:<N>} (voxel field -> GPU-meshed -> lit render)`.
   - **Golden** = the lit BGRA8 image. Existing 103 image goldens UNTOUCHED.
5. **Cross-backend bar (HONEST, controller will measure at bake):** the committed golden is the Mac/Metal render;
   the gate is Metal-output == Metal-golden DIFF 0.0000 (compare.sh EXIT CODE) + the determinism + interior proofs.
   The Vulkan-vs-Metal pixel delta is a documented cross-vendor smoke (≤ a small fraction, max ~1 LSB/channel — the
   visresolve/SW4 float bar), NOT 0.0000. The controller measures it at bake and records it.
6. **Tests `tests/mc_test.cpp` additions (pure CPU):** the host render-mesh build — flat face normals are unit-length
   + correct orientation (outward for the sphere); `position = vert/kSub`; the mesh vert/index counts == MC4;
   determinism. (The render itself is golden-verified, not unit-tested.) Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `gpu-isosurface-meshing-render` (features) + `--mc-render-shot` (showcases).

## RHI seam additions (summary)
- **None expected.** Reuse the existing lit mesh pipeline (`BindVertexBuffer`/`BindIndexBuffer`/`DrawIndexed` +
  the lit `GraphicsPipelineDesc` + the frame UBO) — all pre-existing. If a genuinely-new RHI need surfaces, STOP and
  report it (it should not). `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — MC6+)
Smooth field-gradient (per-vertex) normals (MC6 refinement — flat face normals first), cross-cell vertex dedup,
PBR/textured materials (flat-lit first), real-time editing/CSG/LOD, feeding `BuildMeshlets` → the Nanite cluster
pipeline (a separate compose slice). ONE lit render of the extracted MC mesh through the existing pipeline with the
provenance + determinism + interior/coverage + empty no-op proofs and the lit-sphere golden (float visresolve-bar).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 91) + the new `mc_test` render-mesh-build cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--mc-render-shot` on Vulkan: a coherent LIT 3D extracted sphere (faceted, flat-shaded);
   provenance + determinism + interior/coverage + empty no-op. Run under the Vulkan-validation gate → ZERO VUID in the
   OUTPUT (note: the layer may stack-overflow on teardown on this box per the MC ops lesson — the validation result is
   the GREP of the output for VUID/SYNC-HAZARD, NOT the exit code).
3. Metal: `visual_test --mc-render` → new golden `tests/golden/metal/mc_render.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm the reused lit pipeline + any shader
   MSL-generate.** The cross-vendor Vulkan-vs-Metal delta is the documented float smoke (≤ ~1 LSB), measured by the
   controller — NOT a strict zero-diff (unlike MC1–MC4).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `mc_render.png` added; the other 103
   byte-identical (MC1–MC4 + all existing goldens untouched). `git diff master --stat -- tests/golden` = ONLY
   `mc_render.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+gpu-isosurface-meshing-render` + `--mc-render-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report the reused pipeline/shader). `scripts/verify.ps1` updated:
   `mc_render` golden in the Mac loop + `--mc-render-shot` in `$vkShots`.
