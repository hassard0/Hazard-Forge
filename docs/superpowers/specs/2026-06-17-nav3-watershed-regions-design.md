# Slice NAV3 — Deterministic GPU Navmesh: WATERSHED REGION GENERATION (the make-or-break) (Phase 12 #3) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #7
> (DETERMINISTIC GPU NAVMESH + PATHFINDING, `hf::nav`, header `engine/nav/navmesh.h`). Partitions the
> NAV2 walkable distance field into REGIONS (connected walkable basins) via an INTEGER watershed — the
> make-or-break slice (an irregular flood wavefront whose determinism is the crux). The whole pass is a
> SINGLE-THREAD serial mirror with a FIXED scan order + a LOWEST-cellId tie-break, so it is bit-exact
> CPU↔Vulkan↔Metal (the integer bar, NOT the float visresolve bar). ZERO new RHI. Branch: `slice-nav3`.
> See [[hazard-forge-nav-roadmap]].

**Goal:** Extend `engine/nav/navmesh.h` with `BuildRegions` (an integer watershed over the NAV2 distance
field: assign each walkable cell a REGION id so that connected basins get distinct, deterministic ids).
Add `shaders/nav_region.comp.hlsl` (`[numthreads(1,1,1)]` serial — the nav_distance/nav_raster_scan
mirror), the `nav_regions` integer golden (region ids hash-colored from the read-back → strict zero-diff
cross-backend), `--nav-region-shot` (Vulkan) / `--nav-region` (Metal), and `tests/nav_test.cpp` additions.
Reuse NAV1+NAV2 verbatim (Heightfield/Span/RasterizeTriangleSpans/MergeColumnSpans/FilterWalkableSpans/
IsConnected/BuildDistanceField) — NAV3 is additive (NAV1+NAV2 pipelines + their goldens stay byte-identical).

## Design call: DETERMINISM by a level-descending fixed-order flood (the make-or-break, locked)
A naive watershed ("expand each region until no change") has an irregular wavefront whose result depends on
visitation order → NOT obviously bit-exact. NAV3 LOCKS a deterministic variant: a **level-descending,
fixed-scan-order, lowest-cellId tie-break** flood, run SINGLE-THREAD (`[numthreads(1,1,1)]`) so the GPU is
bit-exact to the CPU reference by construction (the fpx_solve / nav_distance single-thread discipline). The
distance values are small bounded integers (0..maxDist, ~28 in the showcase), so a fixed `maxDist` passes
(one per descending level) is bounded and cheap. Integer-only → int32-native on Metal. This is the slice's
central risk and its central discipline: EVERY ordering decision is pinned (scan cells in ascending cellId;
break region ties by lowest region-id; seed new regions in ascending-cellId order), and the single-thread
serial execution removes any GPU race. A LARGER/structural cross-vendor delta or any nonzero diff is a bug.

## The locked watershed algorithm (CPU reference == shader, verbatim)
Input: `walkable[]`, `surfaceY[]` (NAV2), `dist[]` (NAV2 distance field), the Heightfield dims w×h.
Output: `region[]` (one uint per column; 0 = no region / non-walkable; ids start at 1).
```
region[c] = 0 for all c
nextRegion = 1
for level = maxDist down to 1:                     // descending water level (ridge tops first)
    // (A) GROW: fixed-point expansion of existing regions into this level's unassigned cells.
    repeat:
        changed = false
        for c in ascending cellId:                 // FIXED scan order
            if region[c] != 0 or !walkable[c] or dist[c] != level: continue
            // adopt the LOWEST region id among 4-neighbours that are assigned AND IsConnected
            best = 0
            for nb in {up,down,left,right} (fixed order):
                if region[nb] != 0 and IsConnected(c, nb):       // NAV2 max-step predicate
                    if best == 0 or region[nb] < best: best = region[nb]
            if best != 0: region[c] = best; changed = true
        if not changed: break
    // (B) SEED: any still-unassigned walkable cell AT this level starts a NEW region (ascending cellId).
    for c in ascending cellId:
        if region[c] == 0 and walkable[c] and dist[c] == level:
            region[c] = nextRegion; nextRegion += 1
            // immediately grow the new seed across this level (same fixed-point GROW restricted to == level)
            repeat: changed=false; for c2 ascending: if region[c2]==0 && walkable && dist==level &&
                    (a connected neighbour has region==thisSeed): region[c2]=thisSeed; changed=true; until !changed
```
Notes that make it bit-exact: the GROW fixed-point always scans ascending cellId and takes the LOWEST
neighbour region id, so the converged assignment is order-independent of thread scheduling (single-thread
anyway); SEED ids are handed out in ascending cellId; `dist == level` gates each pass so the flood strictly
descends. (Cells at `dist == 0` — non-walkable, borders, isolated islands NAV2 reset — never get a region;
region 0 is "none".) `maxDist` is read back from NAV2 / recomputed; the loop is bounded by it.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The NAV2 inputs:** `engine/nav/navmesh.h` — `BuildDistanceField` (dist[]), `FilterWalkableSpans`
  (walkable[]/surfaceY[]), `IsConnected` (the 4-neighbour max-step predicate — REUSE verbatim for the
  region grow), `kDistInf`. NAV3 adds `BuildRegions` to this same header; NAV1+NAV2 functions UNCHANGED.
- **The single-thread serial GPU mirror:** `shaders/nav_distance.comp.hlsl` (NAV2's `[numthreads(1,1,1)]`
  two-pass chamfer) + `shaders/nav_raster_scan.comp.hlsl` — copy the one-thread-walks-all-cells-in-fixed-
  order structure. `nav_region.comp` is the same: one thread runs the whole level-descending flood.
- **The integer-golden showcase discipline:** NAV2's `--nav-distance-shot` / `RunNavDistanceShowcase`
  (ReadBuffer the integer result, memcmp GPU==CPU, hash/heat-color the integer buffer, strict zero-diff)
  and NAV1's `--nav-raster-shot`. For region ids, hash-color each id (the `visbuffer` `hashColor(id)`
  discipline) so distinct regions get distinct stable colors → identical both backends by construction.
- **RHI compute envelope (must fit, ZERO additions):** `engine/rhi/rhi.h` — BindStorageBuffer,
  DispatchCompute, ComputeToComputeBarrier, ComputePushConstants, ReadBuffer (the NAV1/NAV2 set).
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--nav-region-shot` standalone arg
  branch + DXC list), `metal_headless/{visual_test.mm,CMakeLists.txt}` (`RunNavRegionShowcase` +
  `--nav-region` + `nav_region` in `hf_gen_msl` — int32 → Metal-native),
  `engine/editor/introspect.cpp` (+`deterministic-navmesh-regions` feature + `--nav-region-shot`
  showcase) + `tests/introspect_test.cpp` + the JSON golden, `scripts/verify.ps1` (`nav_regions` golden
  in the Mac loop + `--nav-region-shot` in `$vkShots`).

## Design decisions (locked)
1. **The watershed is the locked level-descending fixed-order flood above** — single-thread serial,
   ascending-cellId scan, lowest-region-id grow tie-break, ascending-cellId seed order, `IsConnected`-gated
   neighbours (geodesic, never floods across a too-tall step). Pure int32. Bounded by `maxDist` passes.
2. **Region ids are deterministic + dense from 1.** Region 0 = none (non-walkable / dist 0). The id a cell
   gets depends ONLY on the fixed algorithm, so it is identical CPU↔GPU↔both backends. `regionCount` =
   nextRegion-1.
3. **Showcase `--nav-region-shot <out>` (Vulkan) AND `--nav-region` (Metal — WIRE BOTH; confirm
   visual_test.mm + `#include "nav/navmesh.h"`).** Reuse the NAV1 scene (ground + box-step + ramp);
   rasterize→merge→filter→distance→regions. The box-step/ramp obstacles (NAV2 distance seeds, not connected
   to the ground surface) MUST partition into distinct regions from the surrounding ground — a visible,
   meaningful partition (address the "is this just one big region?" risk: the scene is chosen so the
   obstacles split the walkable space into ≥2 regions). ReadBuffer `region[]`; **memcmp GPU == the CPU
   `BuildRegions` reference (the make-or-break)**; hash-color the region ids → `tests/golden/metal/
   nav_regions.png` (baked on the Mac by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance / GPU==CPU (make-or-break):** `region[]` equals the CPU `BuildRegions` reference
     byte-for-byte. Print `nav-region: {columns:<C>, walkable:<W>, regions:<R>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs BYTE-IDENTICAL. Print `nav-region determinism: two runs BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** ≥2 regions, every walkable cell assigned a nonzero region, each region
     is a connected component (no region id appears in two disconnected blobs — a coherent partition).
     Print `nav-region coverage: <R> regions partition <W> walkable cells (each connected)`.
   - **(4) empty no-op:** all-non-walkable → 0 regions / cleared background. Print
     `nav-region empty: 0 regions (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/nav_regions.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 113 image goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict).** region[] is integer, host-snapped inputs, ZERO GPU float →
   Vulkan==Metal BIT-IDENTICAL: the golden is the hash-colored integer region read-back; the controller's
   cross-backend check is the STRICT ZERO-DIFFERING-PIXEL compare. Any nonzero cross-backend diff is a bug.
6. **Tests `tests/nav_test.cpp` additions (pure CPU):** a single open basin → exactly 1 region covering
   all walkable cells; a scene split by a too-tall wall (step > climb) → ≥2 regions, and a cell on each
   side has a different region id; determinism (BuildRegions twice → identical); every walkable cell gets
   region ≥1, every non-walkable gets 0; region connectivity (flood each region id → single component).
   Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-navmesh-regions` (features) + `--nav-region-shot`
   (showcases). Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None expected.** Pure SSBO compute (the NAV1/NAV2 set). `rhi.h` + `rhi_factory` (baseline 2) + backend
  dirs UNCHANGED. NAV1+NAV2 shaders + `engine/sim/fpx.h` + `engine/physics/` UNCHANGED. Report the seam.

## Out of scope (YAGNI — later NAV slices)
Region MERGING/filtering by min-area (small-region cull — a possible NAV4 pre-step; NAV3 produces the raw
watershed partition), monotone-region splitting, contour tracing (NAV4), polygonization (NAV4), A* (NAV5),
the float render (NAV6). NAV3 is ONLY the integer watershed region partition + its bit-exact golden. No
float. No Euclidean distance/sqrt (NAV3 consumes NAV2's integer chamfer field as-is).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 93) + the new `nav_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--nav-region-shot` on Vulkan: GPU==CPU memcmp BIT-EXACT + determinism + coverage
   (≥2 connected regions) + empty no-op; a coherent hash-colored region partition. Run under the
   Vulkan-validation gate → ZERO VUID (set BOTH `VK_LAYER_PATH` to the conan `...\.conan2\p\...\layers`
   dir AND `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer LOADED = zero "not found").
3. Metal: `visual_test --nav-region` → new golden `tests/golden/metal/nav_regions.png`; two runs DIFF
   0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm nav_region
   MSL-generates (int32 → in `hf_gen_msl`).** Cross-backend = STRICT ZERO-DIFFERING-PIXEL, NOT the float
   baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `nav_regions.png` added;
   the other 113 byte-identical (NAV1 nav_spans + NAV2 nav_distfield + all existing untouched). `git diff
   master --stat -- tests/golden` = ONLY `nav_regions.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-navmesh-regions` + `--nav-region-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; shaders int32-only, no int64_t). `scripts/verify.ps1`
   updated: `nav_regions` golden in the Mac loop + `--nav-region-shot` in `$vkShots`. NAV1+NAV2 +
   `engine/sim/fpx.h` + `engine/physics/` UNTOUCHED.
