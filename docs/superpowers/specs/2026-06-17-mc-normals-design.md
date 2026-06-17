# Slice MC6 — GPU Isosurface Meshing: SMOOTH FIELD-GRADIENT NORMALS (Phase 10 #6) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The 6th and FINAL MC slice (after MC5 rendered the mesh
> flat-shaded): shade the extracted surface SMOOTHLY using per-vertex normals from the SDF FIELD GRADIENT (central
> differences) instead of MC5's flat per-face normals — the standard Marching-Cubes smooth-normal technique, the
> finishing quality step that turns the faceted sphere into a smooth one. A float visresolve-bar render slice (same
> bar as MC5: Metal-baked golden, Metal-determinism DIFF 0.0000, provenance, visual parity; cross-vendor ~the float
> baseline). Reuses MC5's render path + the existing lit pipeline — NO new RHI, NO new shader. Completes the 6-slice
> GPU Marching-Cubes flagship. Namespace `hf::render::mc`. Branch: `slice-mc6-normals`. See [[hazard-forge-mc-roadmap]].

**Goal:** Add `GradientNormal` (the central-difference SDF gradient at a world position) + `BuildSmoothRenderMesh`
(MC5's `BuildRenderMesh` with per-vertex gradient normals instead of flat face normals) to `engine/render/mc.h`; add a
`--mc-normals-shot` (Vulkan) / `--mc-normals` (Metal) showcase that renders the extracted sphere smooth-shaded through
the EXISTING lit pipeline, and bake the golden. Make-safe: header additions + a NEW showcase + NEW golden; MC1–MC5 +
everything else UNCHANGED (positions/indices are still the MC4 bit-exact mesh — only the per-vertex NORMAL changes).

## The smooth normal (field gradient)
For a scalar field `f`, the isosurface normal at a point is `normalize(∇f)` (pointing along increasing `f`; for an
SDF with "inside = scalar > iso", the OUTWARD normal is `-∇f` or `+∇f` depending on the sign convention — pick so the
sphere's outer face is lit, the MC5 outward convention). Compute `∇f` by **central differences** on the quantized
field at the vertex's grid neighborhood:
- `GradientNormal(worldPos, field)` — sample `f` at the 6 axis neighbors of the vertex's grid cell (`f(x±1,y,z)`,
  `f(x,y±1,z)`, `f(x,y,z±1)` via `SampleField`, clamped at borders), `grad = (fx+ - fx-, fy+ - fy-, fz+ - fz-)`,
  return `normalize((float3)grad)` oriented outward (negate to match MC5's outward convention; document the sign).
  The vertex's grid coords come from its fixed-point position (`vert.xyz / kSub`, floored) — deterministic. This is a
  host float computation (like MC5's flat normal); the render is float (visresolve-bar) either way.
- Smooth vs flat: MC5 gave every vertex of a triangle the FACE normal (faceted). MC6 gives each vertex the FIELD
  GRADIENT normal at its own position → adjacent triangles share consistent vertex normals → smooth (Gouraud)
  shading. (The triangle soup has duplicate vertices at shared edges, but each gets the same gradient normal at the
  same position, so the shared-edge seams are smooth.)

## Reuse map (file:line)
- **MC5 (the structure to extend):** `engine/render/mc.h` — `RenderVertex`, `BuildRenderMesh` (MC6 adds
  `BuildSmoothRenderMesh` = the same but `normal = GradientNormal(...)`); the reused lit pipeline + showcase wiring in
  `samples/hello_triangle/main.cpp` (`--mc-render-shot`) + `metal_headless/visual_test.mm` (`RunMcRenderShowcase`) —
  `--mc-normals-shot` mirrors it with `BuildSmoothRenderMesh`.
- **MC1–MC4:** `SampleField`, `kSub`, `MarchCellsInterp`, `kCornerOffset`.
- **FP discipline:** `visresolve.h:82-92` (`std::fma` / host-precompute) for any deterministic float helper.
- **The lit pipeline (reused, NO new shader):** the same `lit.vert`/`lit.frag` + `scene::MeshVertexLayout` MC5 reused
  (the smooth normal goes in the SAME normal slot — the lit shader already does per-vertex-normal Lambert/Blinn, so
  smooth normals "just work").
- **Float-golden discipline + registration:** the MC5 spec (Metal-baked golden, Metal-determinism, provenance, the
  cross-vendor float baseline ~55-60 mean — NOT zero-diff); `scripts/verify.ps1` `$Goldens`/`$vkShots`,
  `engine/editor/introspect.cpp`, `tests/introspect_test.cpp`.

## Design decisions (locked)

1. **`BuildSmoothRenderMesh` (host, deterministic).** = `BuildRenderMesh` but per-vertex `normal =
   GradientNormal(vertexWorldPos, field)`. Positions + indices IDENTICAL to MC5 / MC4 (the bit-exact mesh —
   provenance unchanged). Pure host float (same float class as MC5's flat normal; the render is float regardless).
2. **Render through the EXISTING lit pipeline — NO new RHI, NO new shader.** Identical to MC5; only the vertex normal
   buffer content differs. A fixed camera + directional light (same as MC5, so the smooth-vs-flat difference is purely
   the shading).
3. **Showcase `--mc-normals-shot <out>` (Vulkan, main.cpp) AND `--mc-normals` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm + `#include "render/mc.h"`).** SAME field/camera/light as MC5 → `BuildSmoothRenderMesh` →
   lit render → golden `tests/golden/metal/mc_normals.png` (the SMOOTH-shaded extracted sphere — no facets; Metal-baked
   by the controller, do NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance (exact):** the rendered mesh's positions+indices == `MarchCellsInterp` (MC4 bit-exact);
     `verts:15720`. Print `mc-normals mesh: {tris:5240, verts:15720} (MC4 bit-exact mesh, gradient normals)`.
   - **(2) determinism (same backend):** two renders DIFF 0.0000. Print `mc-normals determinism: two renders
     BYTE-IDENTICAL`.
   - **(3) gradient-normal correctness:** every gradient normal is unit-length (within fp tol) and outward-oriented
     (`dot(N, vertexPos - sphereCenter) > 0` for the sphere field, >X% of non-degenerate verts); the smooth render
     differs from the MC5 FLAT render (the refinement took effect — `mc_normals != mc_render` at a sampled set of
     surface pixels). Print `mc-normals: {unit:<k>/<V>, outward:<k>/<V>, smoother-than-flat: yes}`.
   - **(4) empty no-op:** empty field → cleared background. Print `mc-normals empty-field: background (no-op)`.
   - **(5) {stats}:** `mc-normals: {tris:5240, shaded:<N>} (smooth field-gradient normals)`.
   - **Golden** = the smooth-shaded lit image (Metal-baked). Existing 104 image goldens UNTOUCHED.
5. **Cross-backend bar (float visresolve-bar, per the MC5 finding):** Metal-output == Metal-golden DIFF 0.0000
   (determinism) + provenance + VISUAL parity; the Vulkan-vs-Metal cross-vendor delta is the documented float baseline
   (~55-60 mean — NOT zero, the same as MC5/scene_shadow), the controller measures + confirms it's consistent (a
   LARGER/structural delta or a visual mismatch is a bug).
6. **Tests `tests/mc_test.cpp` additions (pure CPU):** `GradientNormal` unit-length + outward for the sphere field
   (>X% of surface verts); a flat-vs-smooth normal set DIFFERS (the refinement is real); positions/indices ==
   `BuildRenderMesh` (only normals differ); determinism. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `gpu-isosurface-meshing-normals` (features) + `--mc-normals-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuse the MC5 lit-render path (the existing lit pipeline). ZERO above-seam backend symbols, NO new shader.
  `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — beyond the flagship)
Cross-cell vertex dedup, PBR/textured materials, real-time editing/CSG/LOD, chunked crack-free seams, feeding
`BuildMeshlets` → the Nanite cluster pipeline (a separate compose slice), trilinear-interpolated gradient (the
6-neighbor central difference suffices). ONE smooth-gradient-normal render of the extracted MC mesh with the
provenance + determinism + gradient-normal-correctness + smoother-than-flat + empty no-op proofs and the smooth-sphere
golden — COMPLETING the 6-slice GPU Marching-Cubes flagship.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 91) + the new `mc_test` gradient-normal cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--mc-normals-shot` on Vulkan: a coherent SMOOTH-shaded extracted sphere (no facets, unlike
   `mc_render`); provenance + determinism + gradient-normal correctness + smoother-than-flat + empty no-op. Run under
   the Vulkan-validation gate → ZERO VUID in the OUTPUT (the layer may stack-overflow on teardown — gate on the output
   grep, not the exit code, per the MC ops lesson).
3. Metal: `visual_test --mc-normals` → new golden `tests/golden/metal/mc_normals.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). **Confirm visual_test.mm in the diff.** Cross-vendor Vulkan-vs-Metal = the documented float
   baseline (~55-60 mean), measured by the controller — NOT zero-diff (a float slice).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `mc_normals.png` added; the other 104
   byte-identical (MC1–MC5 + all existing untouched). `git diff master --stat -- tests/golden` = ONLY `mc_normals.png`
   (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+gpu-isosurface-meshing-normals` + `--mc-normals-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI, no new shader). `scripts/verify.ps1` updated: `mc_normals` golden
   in the Mac loop + `--mc-normals-shot` in `$vkShots`.
