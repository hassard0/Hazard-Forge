# Slice NAV2 — Deterministic GPU Navmesh: WALKABLE FILTER + INTEGER DISTANCE FIELD (Phase 12 #2) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #7
> (DETERMINISTIC GPU NAVMESH + PATHFINDING, `hf::nav`, header `engine/nav/navmesh.h`). Builds on the
> NAV1 integer heightfield: (1) FILTER which spans are WALKABLE (integer clearance/step compares) →
> a per-column walkable surface, and (2) build an INTEGER DISTANCE FIELD over the walkable cells (an
> integer chamfer transform — NO sqrt) that NAV3's watershed regions consume. Both stages are pure
> int32 → strict zero-differing-pixel cross-backend goldens (the NAV1 integer bar, NOT the float
> visresolve bar). ZERO new RHI. Branch: `slice-nav2`. See [[hazard-forge-nav-roadmap]].

**Goal:** Extend `engine/nav/navmesh.h` with `FilterWalkableSpans` (mark each span's `area` walkable/not
by integer clearance-above ≥ agent-height + an integer max-step/climb test vs neighbour columns) producing
a per-column WALKABLE SURFACE (the top walkable span height + a walkable mask), and `BuildDistanceField`
(an integer two-pass chamfer distance transform from each walkable cell to the nearest non-walkable/border
cell — integer chamfer weights, NO `std::sqrt`). Add `shaders/nav_filter.comp.hlsl` (one thread per
column, int32) + `shaders/nav_distance.comp.hlsl` (`[numthreads(1,1,1)]` serial two-pass chamfer, the
mc_scan/fpx_solve single-thread mirror), the `nav_distfield` integer golden (CPU-colored distance read-back
→ strict zero-diff cross-backend), `--nav-distance-shot` (Vulkan) / `--nav-distance` (Metal), and
`tests/nav_test.cpp` additions. Reuse NAV1's Heightfield/Span/RasterizeTriangleSpans/MergeColumnSpans
verbatim — NAV2 is additive (NAV1's nav_raster pipeline + its golden stay byte-identical).

## Design call: INTEGER bit-exact (the NAV1 bar, extended)
Walkability is integer compares (clearance, step height, slope-as-rise/run all in voxel units → exact
in/out, the `fpx.h::AabbOverlap` six-compare discipline). The distance field is an INTEGER chamfer (the
Recast `calculateDistanceField` kept integer: cardinal weight 2, diagonal weight 3, the standard integer
chamfer — NO Euclidean sqrt) → int32-native on Metal. So both NAV2 goldens are the STRICT
zero-differing-pixel cross-backend bar (like NAV1/mc/fpx-broadphase), NOT the float ~55–60 baseline. NO
int64 anywhere in NAV2 (the chamfer is int32; a true-Euclidean distance is explicitly OUT of scope —
that's the only place sqrt/int64 would appear, deferred). The chamfer's two sweeps are inherently
SEQUENTIAL (each cell reads its already-updated neighbours), so the GPU pass is the proven single-thread
`[numthreads(1,1,1)]` serial mirror (deterministic → bit-exact); the walkable filter is per-column
independent → one thread per column, race-free.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The NAV1 heightfield (the input):** `engine/nav/navmesh.h` — `Heightfield`, `Span{ymin,ymax,area}`,
  `RasterizeTriangleSpans` (count→scan→emit), `MergeColumnSpans` (the deterministic per-column merge —
  NAV2's filter operates on MERGED spans so clearance-above is correct). NAV2 adds functions to this same
  header; NAV1's are UNCHANGED. The `area` field NAV1 stamped as a placeholder (=1) is what NAV2 sets for
  real (walkable=1 / not-walkable=0).
- **The single-thread serial GPU mirror (for the chamfer):** `shaders/nav_raster_scan.comp.hlsl`
  (NAV1's `[numthreads(1,1,1)]` serial prefix-sum) + `shaders/fpx_solve.comp.hlsl` /
  `shaders/mc_scan.comp.hlsl` — copy the single-thread serial-sweep pattern (one thread walks all cells in
  a FIXED order maintaining the running transform). The chamfer is forward sweep (row-major TL→BR) then
  backward sweep (BR→TL), each cell = min(self, neighbour+weight) — deterministic, order-fixed.
- **One-thread-per-column compute (for the filter):** `shaders/nav_raster_count.comp.hlsl` (NAV1, one
  thread per column, no atomics) — mirror its column-indexing + the per-column triangle re-scan if the
  filter re-derives merged spans per column (self-contained per thread; the bounded per-column span merge
  is a small fixed insertion — deterministic).
- **Integer compare discipline:** `engine/sim/fpx.h::AabbOverlap` (six int32 compares). Walkability =
  `clearanceAbove ≥ walkableHeight` (voxel units) AND, vs each of the 4 neighbour columns' walkable
  surface, `abs(myTop - nbrTop) ≤ walkableClimb` (the max-step test, integer). All int32.
- **The integer-golden showcase discipline (the bar to copy):** NAV1's `--nav-raster-shot` /
  `RunNavRasterShowcase` (ReadBuffer the integer result, memcmp GPU==CPU, CPU-color the integer buffer
  into a debug image, strict zero-diff cross-backend) — and `--mc-classify-shot` / `--fpx-pairs-shot`.
- **RHI compute envelope (must fit, ZERO additions):** `engine/rhi/rhi.h` — BindStorageBuffer,
  DispatchCompute, ComputeToComputeBarrier, ComputePushConstants, ReadBuffer (the NAV1/MC1/FPX1 set).
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--nav-distance-shot` standalone arg
  branch + DXC list), `metal_headless/{visual_test.mm,CMakeLists.txt}` (`RunNavDistanceShowcase` +
  `--nav-distance` + the 2 new shaders in `hf_gen_msl` — int32 → Metal-native),
  `engine/editor/introspect.cpp` (+`deterministic-navmesh-distancefield` feature + `--nav-distance-shot`
  showcase) + `tests/introspect_test.cpp` + the JSON golden, `scripts/verify.ps1` (`nav_distfield` golden
  in the Mac loop + `--nav-distance-shot` in `$vkShots`).

## Design decisions (locked)
1. **The walkable surface model.** `WalkableConfig{ int walkableHeight; int walkableClimb; }` (voxel
   units). `FilterWalkableSpans(hf, cfg, mergedSpansPerColumn)`: for each column, over its MERGED spans
   (top to bottom), a span's TOP is a walkable surface iff the clearance to the next solid span above
   (or to the heightfield top) ≥ `walkableHeight`; set `area=1` on walkable spans else 0. Then derive a
   per-column WALKABLE MASK `walkable[col] ∈ {0,1}` = the column has ≥1 walkable surface, and a per-column
   `surfaceY[col]` = the top walkable span's top-y (the cell the distance field/regions use). The 4-neighbour
   max-step test (`abs(surfaceY[col]-surfaceY[nbr]) ≤ walkableClimb`) marks a walkable cell as CONNECTED;
   a cell with no walkable connected neighbour on a side is a BORDER (distance-field seed). Pure integer.
2. **The integer distance field.** `BuildDistanceField(hf, walkable[], surfaceY[], dist[])`: a 2-pass
   integer chamfer over the w×h walkable grid. Seed: non-walkable cells (and the grid border) = distance 0;
   walkable cells = "infinity" (a large int). Forward sweep row-major: `dist[c] = min(dist[c],
   dist[cardinalNbr]+2, dist[diagNbr]+3)` over the already-visited (up/left) neighbours; backward sweep
   reverse over the down/right neighbours. Chamfer weights 2 (cardinal) / 3 (diagonal) — the standard
   integer Recast chamfer (NO sqrt, NO int64). A neighbour is only traversed if it is walkable AND
   connected (the max-step test from decision 1) — so the distance is geodesic over the walkable surface,
   not Euclidean-through-walls. Deterministic single-thread serial → bit-exact. Output `dist[]` (w×h uint).
3. **Showcase `--nav-distance-shot <out>` (Vulkan) AND `--nav-distance` (Metal — WIRE BOTH; confirm
   visual_test.mm + `#include "nav/navmesh.h"`).** Reuse NAV1's MakeShowcaseTriangles scene (ground +
   box-step + ramp) — rasterize (NAV1) → merge → filter (NAV2) → distance field. ReadBuffer the integer
   `dist[]` (and the walkable mask); **memcmp GPU == the CPU FilterWalkableSpans+BuildDistanceField
   reference (the make-or-break)**; CPU-color the distance field (a heat ramp over the walkable region,
   non-walkable = background) → `tests/golden/metal/nav_distfield.png` (baked on the Mac by the CONTROLLER
   — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance / GPU==CPU (make-or-break):** the walkable mask + `dist[]` equal the CPU reference
     byte-for-byte. Print `nav-distance: {columns:<C>, walkable:<W>, maxdist:<D>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs BYTE-IDENTICAL. Print `nav-distance determinism: two runs BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the distance field is a coherent gradient peaking in the walkable
     interior (`maxdist > 0`, the interior is farther than the border). Print
     `nav-distance coverage: <W> walkable cells, peak dist <D> (coherent gradient)`.
   - **(4) empty no-op:** an empty heightfield (or all-non-walkable) → all-zero distance / cleared
     background. Print `nav-distance empty: 0 walkable (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/nav_distfield.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 112 image goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict).** Both passes int32, host-snapped → ZERO GPU float →
   Vulkan==Metal BIT-IDENTICAL: the golden is the CPU-colored integer distance read-back; the controller's
   cross-backend check is the STRICT ZERO-DIFFERING-PIXEL compare (NOT the float baseline). Any nonzero
   cross-backend pixel diff is a real bug.
6. **Tests `tests/nav_test.cpp` additions (pure CPU):** a flat ground → all columns walkable, distance
   peaks in the centre; a tall wall/step exceeding walkableClimb → the cells across it are NOT connected
   (distance does not bleed across); clearance below walkableHeight → not walkable; chamfer monotonicity
   (interior ≥ border); empty → zero. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-navmesh-distancefield` (features) + `--nav-distance-shot`
   (showcases). Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None expected.** Pure SSBO compute (BindStorageBuffer, DispatchCompute, ComputeToComputeBarrier,
  ComputePushConstants, ReadBuffer — the NAV1 set). `rhi.h` + `rhi_factory` (baseline 2) + backend dirs
  UNCHANGED. NAV1's nav_raster shaders + `engine/sim/fpx.h` + `engine/physics/` UNCHANGED. Report the seam.

## Out of scope (YAGNI — later NAV slices)
True Euclidean distance (the only place sqrt/int64 would appear — deferred; the integer chamfer is the
NAV3 watershed input either way). Watershed REGION generation (NAV3). Contour/polygonization (NAV4). A*
(NAV5). The float lit-3D render (NAV6). NAV2 is ONLY the walkable filter + the integer chamfer distance
field + their bit-exact goldens. No agent-radius erosion yet (a possible NAV3 refinement). No float.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 93) + the new `nav_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--nav-distance-shot` on Vulkan: GPU==CPU memcmp BIT-EXACT + determinism +
   coverage + empty no-op; a coherent distance-gradient image. Run under the Vulkan-validation gate → ZERO
   VUID in the OUTPUT (set BOTH `VK_LAYER_PATH` to the conan `...\.conan2\p\...\layers` dir AND
   `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer LOADED = zero "not found" lines).
3. Metal: `visual_test --nav-distance` → new golden `tests/golden/metal/nav_distfield.png`; two runs DIFF
   0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm nav_filter +
   nav_distance MSL-generate (int32 → in `hf_gen_msl`).** Cross-backend = STRICT ZERO-DIFFERING-PIXEL
   (controller-measured), NOT the float baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `nav_distfield.png` added;
   the other 112 byte-identical (NAV1's nav_spans + all existing untouched). `git diff master --stat --
   tests/golden` = ONLY `nav_distfield.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-navmesh-distancefield` + `--nav-distance-shot`;
   introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; shaders int32-only, no int64_t). `scripts/verify.ps1`
   updated: `nav_distfield` golden in the Mac loop + `--nav-distance-shot` in `$vkShots`. NAV1 +
   `engine/sim/fpx.h` + `engine/physics/` UNTOUCHED.
