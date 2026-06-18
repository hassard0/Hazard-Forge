# Slice NAV4 — Deterministic GPU Navmesh: CONTOUR TRACING + INTEGER POLYGONIZATION (Phase 12 #4) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #7
> (DETERMINISTIC GPU NAVMESH + PATHFINDING, `hf::nav`, header `engine/nav/navmesh.h`). Turns the NAV3
> region partition into a POLYGON MESH: trace each region's boundary into an integer contour, simplify
> it, and triangulate it into convex polygons with cross-poly adjacency — the walkable graph NAV5's A*
> runs over. All integer / host-snapped → strict zero-differing-pixel cross-backend (the NAV1–NAV3
> integer bar). ZERO new RHI. Branch: `slice-nav4`. See [[hazard-forge-nav-roadmap]].

**Goal:** Extend `engine/nav/navmesh.h` with `TraceContours` (a deterministic integer boundary walk of
each NAV3 region → a closed loop of integer contour vertices), `SimplifyContour` (integer Douglas–Peucker
— a perpendicular-distance-squared test, no float), and `BuildPolyMesh` (ear-clip triangulation with an
integer cross-product orientation test → convex polygons [triangles] + per-edge cross-poly adjacency). Add
`shaders/nav_contour.comp.hlsl` + `shaders/nav_polygonize.comp.hlsl` (`[numthreads(1,1,1)]` single-thread
serial — the nav_region/nav_distance mirror; the per-region trace/triangulate is inherently sequential),
the `nav_polymesh` integer golden (the contours + triangulated polys, CPU-colored from the integer
read-back → strict zero-diff cross-backend), `--nav-polymesh-shot` (Vulkan) / `--nav-polymesh` (Metal), and
`tests/nav_test.cpp` additions. Reuse NAV1–NAV3 verbatim — NAV4 is additive (their pipelines + goldens
stay byte-identical).

## Design call: INTEGER bit-exact, determinism by pinned order (the NAV3 discipline, extended)
Contour tracing, simplification, and triangulation are all sequential and order-dependent → run them
SINGLE-THREAD (`[numthreads(1,1,1)]`, the nav_region/fpx_solve mirror) with EVERY ordering decision pinned:
the contour walk starts at each region's LOWEST-cellId boundary cell and turns in a FIXED order (e.g.
clockwise, the Recast "walk the left-hand wall" rule); Douglas–Peucker recursion is an explicit
fixed-order stack (split at the max-deviation vertex, ties → lowest index); ear-clipping always clips the
LOWEST-index valid ear. So the GPU is bit-exact to the CPU reference by construction, and Vulkan==Metal is
byte-identical (the integer bar, NOT the float visresolve bar). Coordinates are bounded small voxel ints
(the showcase grid is 32×32, coords ≤31), so the cross-product / perpendicular-distance-squared **fit
int32** and the shaders stay **Metal-MSL-native** (like NAV1's edge functions). If a stage's products
could exceed int32 for a general (larger) grid, that ONE shader uses int64 → Vulkan-only + CPU-reference
Metal (the FPX1/`swraster` convention); prefer int32 and document the bound. The variable-length per-region
output (contour vertex counts, poly counts) uses the proven count→scan→emit compaction (mc.h/NAV1).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The NAV3 input:** `engine/nav/navmesh.h` — `BuildRegions` (region[] per column), `IsConnected`,
  `MergeColumnSpans`, the Heightfield. NAV4 adds `TraceContours`/`SimplifyContour`/`BuildPolyMesh` to this
  same header; NAV1–NAV3 functions UNCHANGED. A contour boundary is between a region cell and a different-
  region / region-0 / out-of-bounds neighbour (the standard Recast boundary test).
- **The single-thread serial GPU mirror:** `shaders/nav_region.comp.hlsl` (NAV3's `[numthreads(1,1,1)]`
  watershed) + `shaders/nav_distance.comp.hlsl` — copy the one-thread-walks-everything-in-fixed-order
  structure for the trace + triangulate. The count→scan→emit layout for variable output:
  `shaders/nav_raster_{count,scan,emit}.comp.hlsl` (NAV1) + `engine/render/mc.h:425/443`.
- **Integer orientation / edge math:** NAV1's `PointInTriXZ` 2D edge-function sign test
  (`engine/nav/navmesh.h`) — the same integer cross product is the ear-clip orientation + the
  point-in-triangle (no ear contains another vertex) test. `engine/sim/fpx.h::FloorDiv`/`FxISqrt` (the
  latter only if a true length is ever needed — Douglas–Peucker uses squared distance, NO sqrt).
- **The integer-golden showcase discipline:** NAV3's `--nav-region-shot` / `RunNavRegionShowcase`
  (ReadBuffer the integer result, memcmp GPU==CPU, color the integer buffer, strict zero-diff) + the
  debug-line drawing the engine already has (the `--cull-shot` / meshlet-viz line-list path) for drawing
  contour edges + poly edges.
- **RHI compute envelope (must fit, ZERO additions):** `engine/rhi/rhi.h` — the NAV1–NAV3 set
  (BindStorageBuffer, DispatchCompute, ComputeToComputeBarrier, ComputePushConstants, ReadBuffer).
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--nav-polymesh-shot` standalone arg
  branch + DXC list), `metal_headless/{visual_test.mm,CMakeLists.txt}` (`RunNavPolymeshShowcase` +
  `--nav-polymesh` + the new shaders in `hf_gen_msl` — int32 → Metal-native, or excluded + CPU-ref if a
  stage is forced int64), `engine/editor/introspect.cpp` (+`deterministic-navmesh-polymesh` feature +
  `--nav-polymesh-shot` showcase) + `tests/introspect_test.cpp` + the JSON golden, `scripts/verify.ps1`
  (`nav_polymesh` golden in the Mac loop + `--nav-polymesh-shot` in `$vkShots`).

## Design decisions (locked)
1. **Contour tracing (deterministic boundary walk).** For each region (ascending region id), find its
   LOWEST-cellId boundary cell, then walk the region boundary in a FIXED turn order (clockwise, the
   left-wall-follow rule), emitting an integer contour vertex at each boundary corner, until the walk
   returns to the start. Output a closed integer vertex loop per region. Deterministic by the fixed start
   + fixed turn order. Pure int32. (Degenerate single-cell regions → a 4-vertex square loop.)
2. **Simplification (integer Douglas–Peucker).** Simplify each contour loop with an integer
   perpendicular-distance-squared test vs a config threshold `maxError²` (so no sqrt): keep a vertex iff
   its squared deviation from the simplified edge exceeds the threshold; recursion is an explicit
   fixed-order stack (split at the max-deviation index, tie → lowest index). Pure integer (int32 for the
   bounded grid; int64 only if the squared products could overflow — that one shader Vulkan-only +
   CPU-Metal, documented). A minimum of 3 vertices is kept per contour.
3. **Polygonization (ear-clip triangulation + adjacency).** Triangulate each simplified contour into
   convex polygons (triangles for NAV4 — convex by construction) by ear-clipping: repeatedly clip the
   LOWEST-index valid ear (an ear = a convex vertex whose triangle contains no other contour vertex — the
   integer cross-product orientation + point-in-triangle tests). Build per-triangle-edge ADJACENCY (two
   polys sharing an edge are neighbours; a boundary edge has no neighbour) — the graph NAV5's A* walks.
   Output: poly vertex indices + per-poly neighbour ids, laid out by count→scan→emit. Pure integer,
   deterministic (fixed ear order). (Convex-poly merging beyond triangles is a deferred refinement.)
4. **Showcase `--nav-polymesh-shot <out>` (Vulkan) AND `--nav-polymesh` (Metal — WIRE BOTH; confirm
   visual_test.mm + `#include "nav/navmesh.h"`).** Reuse the NAV1 scene → rasterize→merge→filter→distance→
   regions→contours→simplify→polymesh. ReadBuffer the poly mesh (vertices + indices + adjacency);
   **memcmp GPU == the CPU `TraceContours`+`SimplifyContour`+`BuildPolyMesh` reference (the make-or-break)**;
   CPU-color a debug image: the region fills (faint) + the simplified contour edges (bright lines) + the
   triangulated poly edges → `tests/golden/metal/nav_polymesh.png` (baked on the Mac by the CONTROLLER —
   DO NOT commit).
5. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance / GPU==CPU (make-or-break):** the contour verts + poly indices + adjacency equal the
     CPU reference byte-for-byte. Print `nav-polymesh: {regions:<R>, contourVerts:<V>, polys:<P>}
     GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs BYTE-IDENTICAL. Print `nav-polymesh determinism: two runs BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** every region yields a closed contour (≥3 verts) and ≥1 poly; the
     adjacency graph is consistent (each shared edge referenced by exactly its two polys). Print
     `nav-polymesh coverage: <P> polys over <R> regions (closed contours, consistent adjacency)`.
   - **(4) empty no-op:** zero regions → 0 contours / 0 polys / cleared background. Print
     `nav-polymesh empty: 0 polys (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/nav_polymesh.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 114 image goldens UNTOUCHED.
6. **Cross-backend bar (INTEGER, strict).** All outputs integer, host-snapped, ZERO GPU float →
   Vulkan==Metal BIT-IDENTICAL: the golden is the CPU-colored integer read-back; the controller's
   cross-backend check is the STRICT ZERO-DIFFERING-PIXEL compare. Any nonzero cross-backend diff is a bug.
   (If any stage is forced int64 → Vulkan-only + the Metal showcase runs the CPU reference, byte-identical
   by construction — the FPX1/swraster convention; the golden is still strict zero-diff.)
7. **Tests `tests/nav_test.cpp` additions (pure CPU):** a single square region → a 4-vertex contour → 2
   triangles sharing the diagonal (adjacency = each other); a known L-shaped region → the expected contour
   vertex count after simplification; ear-clip produces n-2 triangles for an n-gon; adjacency symmetry
   (a is b's neighbour ⟺ b is a's); determinism (twice → identical); empty → zero. Clean under
   `windows-msvc-asan`.
8. **Introspect.** Add exactly `deterministic-navmesh-polymesh` (features) + `--nav-polymesh-shot`
   (showcases). Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None expected.** Pure SSBO compute (the NAV1–NAV3 set). `rhi.h` + `rhi_factory` (baseline 2) + backend
  dirs UNCHANGED. NAV1–NAV3 shaders + `engine/sim/fpx.h` + `engine/physics/` UNCHANGED. Report the seam.

## Out of scope (YAGNI — later NAV slices)
Convex-polygon MERGING beyond triangles (Recast merges triangles into larger convex polys — a deferred
refinement; NAV4 ships the triangulated poly mesh + adjacency, which is a complete A* graph). Detail-mesh
height refinement. A* pathfinding (NAV5 — runs over NAV4's poly adjacency). The float render (NAV6). NAV4
is ONLY the integer contour + simplify + triangulate + adjacency + its bit-exact golden. No float.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 93) + the new `nav_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--nav-polymesh-shot` on Vulkan: GPU==CPU memcmp BIT-EXACT + determinism +
   coverage (closed contours, consistent adjacency) + empty no-op; a coherent contour+poly image. Run
   under the Vulkan-validation gate → ZERO VUID (set BOTH `VK_LAYER_PATH` to the conan `...\.conan2\p\
   ...\layers` dir AND `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer LOADED = zero
   "not found").
3. Metal: `visual_test --nav-polymesh` → new golden `tests/golden/metal/nav_polymesh.png`; two runs DIFF
   0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm the new shaders
   MSL-generate (int32 → in `hf_gen_msl`) OR are correctly excluded + CPU-ref'd if int64.** Cross-backend
   = STRICT ZERO-DIFFERING-PIXEL, NOT the float baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `nav_polymesh.png` added;
   the other 114 byte-identical (NAV1–NAV3 + all existing untouched). `git diff master --stat --
   tests/golden` = ONLY `nav_polymesh.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-navmesh-polymesh` + `--nav-polymesh-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report int32-native vs any int64 Vulkan-only stage).
   `scripts/verify.ps1` updated: `nav_polymesh` golden in the Mac loop + `--nav-polymesh-shot` in
   `$vkShots`. NAV1–NAV3 + `engine/sim/fpx.h` + `engine/physics/` UNTOUCHED.
