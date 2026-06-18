# Slice NAV6 — Deterministic GPU Navmesh: LIT 3D RENDER CAPSTONE (the money-shot, COMPLETES flagship #7) (Phase 12 #6) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice of FLAGSHIP #7
> (DETERMINISTIC GPU NAVMESH + PATHFINDING, `hf::nav`). RENDERS the bit-exact integer navmesh as a lit
> 3D scene — the walkable polygons as a translucent colored overlay + the A* corridor as a bright line
> over a lit ground — the money-shot proving the deterministic navmesh + path as a real 3D scene. The
> geometry comes from the BIT-EXACT integer navmesh (NAV1–NAV5; poly verts + path → float positions
> host-side, for rendering ONLY — the build/pathfind stays integer); the raster/shade is float, so this
> is the FLOAT visresolve-bar (like FPX6/MC5): Metal-baked golden, Metal-determinism DIFF 0.0000 +
> provenance + visual parity, cross-vendor ~the float baseline. Reuses the EXISTING lit-mesh +
> transparent/OIT + debug-line passes — NO new RHI, NO new shader. After this slice flagship #7 is
> COMPLETE (6/6). Branch: `slice-nav6`. See [[hazard-forge-nav-roadmap]].

**Goal:** Add render-only float helpers to `engine/nav/navmesh.h` (`NavVertToWorld` = the single host
`coord/(float)scale` conversion; `PolyMeshToRenderMesh` → the navmesh polys as a `math`/`scene` triangle
mesh with per-region vertex colors; `PathToWorldPolyline` → the A* corridor poly centroids as a float line
strip), run the full deterministic NAV1–NAV5 pipeline to a navmesh + path, and render: a lit ground/scene
(the showcase heightfield as a base) + the walkable navmesh polygons as a TRANSLUCENT per-region-colored
overlay (the existing transparent/OIT pass) + the A* corridor as a bright debug LINE (the existing
debug-line pass), from a fixed 3/4 camera + directional light. Add `--nav-render-shot` (Vulkan) /
`--nav-render` (Metal). Bake the lit golden `nav_render`. Make-safe: small host float helpers (NAV1–NAV5
build/pathfind UNCHANGED — only float accessors for rendering) + a NEW showcase + NEW golden; reuse the
existing lit-mesh + transparent + debug-line pipelines — NO new shader/RHI.

## Design call: FLOAT visresolve-bar (the NAV arc's only float slice)
NAV1–NAV5 are strict integer/bit-exact (the navmesh build + A* are integer). NAV6 RASTERIZES the result
with a perspective camera + lighting + translucency → cross-vendor float (the FPX6/MC5/visresolve caveat).
So NAV6's golden is the FLOAT bar, identical to FPX6: the committed golden is **baked on Metal**; the gate
is **Metal-render == Metal-golden DIFF 0.0000** (deterministic, two-run) + **provenance** (the rendered
mesh + line are built from the bit-exact integer navmesh — `PolyMeshToRenderMesh`/`PathToWorldPolyline`
over the NAV4 polys + NAV5 corridor, deterministic host code) + visual parity, with the Vulkan-vs-Metal
delta the documented cross-vendor smoke (~55–60 mean, the engine's float-render baseline — NOT zero, NOT
the integer bar). The integer NAVMESH feeding the render is still bit-exact (NAV1–NAV5); only the final
raster/shade is float.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The navmesh (the input):** `engine/nav/navmesh.h` — the NAV4 `BuildPolyMesh` output (poly verts +
  indices + per-region grouping) and the NAV5 `FindPath` corridor. World position = `voxelCoord /
  (float)scale` (the ONE host float conversion — match the showcase's voxel→world scale; document it).
- **The existing lit-mesh pipeline to REUSE (find + mirror — do NOT invent):** the engine's lit mesh
  render — study FPX6's `RunFpxRenderShowcase` (the instanced lit pipeline) and MC5's `RunMcRenderShowcase`
  (the single static-lit-mesh render, `lit.vert`+`lit.frag`, `scene::MeshVertexLayout`, the FrameData
  camera/light UBO, sky + shadow + post — a structural template for a single lit mesh). Reuse one of these
  for the lit ground/base scene.
- **The transparent / OIT pass (for the translucent navmesh overlay):** the engine's order-independent-
  transparency / blended pass (the `oit_test` / the transparent showcase — grep for OIT / blend / the
  transparent pipeline desc). Render the navmesh polys (per-region colored, alpha ~0.4) through it so the
  overlay reads as a translucent colored sheet over the ground. If OIT is heavyweight, a simple
  alpha-blended forward pass over the lit ground is acceptable (document the choice).
- **The debug-line pass (for the A* corridor):** the `--cull-shot` white-wireframe / `fpx_orient` gizmo /
  meshlet-viz LINE-LIST path (`engine/render/debug_draw.*`). Draw the corridor as a bright thick polyline
  (raised slightly above the navmesh so it's visible) + start/goal markers.
- **The float-golden discipline (the bar to copy):** FPX6 / MC5 — per-backend Metal golden, determinism,
  provenance, the cross-vendor ~55–60 baseline (NOT zero-diff).
- **Showcase + registration:** the FPX6 `--fpx-render-shot` template; `verify.ps1`/`introspect.cpp`/
  `introspect_test.cpp`.

## Design decisions (locked)
1. **Render the full navmesh result as a lit 3D scene.** Run the deterministic NAV1–NAV5 pipeline (the
   showcase ground + obstacles → heightfield → … → polymesh → A* path). Build: (a) a lit base — the
   ground plane (+ the obstacle boxes if convenient) as a lit mesh; (b) the navmesh polygons as a
   TRANSLUCENT mesh, per-region color (region 1 / region 2 distinct, the obstacle holes left open),
   slightly above the ground; (c) the A* corridor as a bright debug line through the poly centroids, with
   start/goal markers. Fixed 3/4 camera + directional light. Pure deterministic host float (the
   `coord/(float)scale` conversion), render-only — NAV1–NAV5 integer state UNCHANGED.
2. **Render through the EXISTING lit-mesh + transparent + debug-line pipelines — NO new RHI, ideally NO
   new shader.** Reuse the lit-mesh `GraphicsPipelineDesc` (ground), the transparent/blend pipeline
   (navmesh overlay), the debug-line pipeline (corridor), the frame/camera UBO. Report exactly which
   pipelines/shaders were reused; if a genuinely-new shader is needed, flag it loudly (prefer reuse).
3. **Showcase `--nav-render-shot <out>` (Vulkan, main.cpp) AND `--nav-render` (Metal, visual_test.mm —
   WIRE BOTH; confirm visual_test.mm + `#include "nav/navmesh.h"`).** Run the deterministic pipeline →
   float render mesh + corridor line → lit render → the BGRA8 image. Golden = the lit 3D navmesh scene →
   `tests/golden/metal/nav_render.png` (baked on the Mac by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance (the navmesh IS bit-exact):** the rendered mesh + line are built from the navmesh
     that equals the CPU `BuildPolyMesh`+`FindPath` reference (the same bit-exact NAV1–NAV5 result — print
     the region/poly/corridor counts + that the geometry derives from the integer navmesh). Print
     `nav-render: {regions:<R>, polys:<P>, corridor:<L>, shaded:<S>} (integer navmesh -> lit 3D render)`.
   - **(2) determinism (same backend):** two renders → DIFF 0.0000. Print `nav-render determinism: two
     renders BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the lit navmesh overlay + corridor cover a coherent region (`shaded >
     0`, not uniform — a recognizable navmesh + path). Print `nav-render coverage: <S> shaded (coherent
     navmesh + path)`.
   - **(4) empty no-op:** an empty navmesh (zero polys) → the lit ground only / cleared overlay. Print
     `nav-render empty: base only (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/nav_render.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 116 image goldens UNTOUCHED.
5. **Cross-backend bar (FLOAT, per the FPX6/MC5 finding):** Metal-output == Metal-golden DIFF 0.0000
   (determinism) + provenance (the bit-exact integer navmesh) + VISUAL parity; the Vulkan-vs-Metal
   cross-vendor delta is the documented float baseline (~55–60 mean — like FPX6/scene_shadow, NOT zero),
   measured by the controller (a LARGER/structural delta or a visual mismatch is a bug).
6. **Tests `tests/nav_test.cpp` additions (pure CPU):** `NavVertToWorld` — a known integer coord → the
   expected float world position (scale correct); `PolyMeshToRenderMesh` — N polys → 3N vertices with the
   right per-region colors; `PathToWorldPolyline` — an L-poly corridor → L centroid points; the navmesh
   feeding it is the existing bit-exact `BuildPolyMesh`/`FindPath` (already tested). Clean under
   `windows-msvc-asan`. (The render is golden-verified, not unit-tested.)
7. **Introspect.** Add exactly `deterministic-navmesh-render` (features) + `--nav-render-shot` (showcases).

## RHI seam additions (summary)
- **None expected.** Reuse the existing lit-mesh + transparent/blend + debug-line pipelines + the frame
  UBO — all pre-existing. If a genuinely-new RHI need surfaces, STOP and report it. `rhi.h` + `rhi_factory`
  (baseline 2) + backend dirs UNCHANGED. `engine/nav/` NAV1–NAV5 + `engine/sim/fpx.h` + `engine/physics/`
  UNCHANGED. Report the seam.

## Out of scope (YAGNI — flagship done after this)
PBR/textured navmesh materials (flat per-region color first), animated agents walking the path, inter-
region portals / hole-carving (the documented NAV-fidelity gaps — NOT rendered as fixed), funnel/string-
pull smoothing. Claim the deterministic integer NAVMESH + PATH (the substance) + a lit render of it. ONE
lit 3D render of the navmesh + A* corridor with the provenance + determinism + coverage + empty no-op
proofs and the lit golden — COMPLETING the 6-slice deterministic-navmesh-and-pathfinding flagship.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 93) + the new `nav_test` render-helper cases.
   Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--nav-render-shot` on Vulkan: a coherent LIT 3D navmesh + corridor (the
   deterministic navmesh, rendered); provenance + determinism + coverage + empty no-op. Run under the
   Vulkan-validation gate → ZERO VUID in the OUTPUT (set BOTH `VK_LAYER_PATH` to the conan
   `...\.conan2\p\...\layers` dir AND `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer
   LOADED = zero "not found" lines).
3. Metal: `visual_test --nav-render` → new golden `tests/golden/metal/nav_render.png`; two runs DIFF
   0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm the reused
   lit/transparent/debug-line pipelines + any shader MSL-generate.** The cross-vendor Vulkan-vs-Metal
   delta is the documented float smoke (~55–60 mean), measured by the controller — NOT strict zero-diff
   (unlike NAV1–NAV5).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `nav_render.png` added; the
   other 116 byte-identical (NAV1–NAV5 + all existing untouched). `git diff master --stat -- tests/golden`
   = ONLY `nav_render.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-navmesh-render` + `--nav-render-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report the reused pipelines/shaders).
   `scripts/verify.ps1` updated: `nav_render` golden in the Mac loop + `--nav-render-shot` in `$vkShots`.
   NAV1–NAV5 + `engine/sim/fpx.h` + `engine/physics/` UNTOUCHED.
