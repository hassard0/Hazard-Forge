# Slice NAV1 — Deterministic GPU Navmesh: INTEGER HEIGHTFIELD SPAN RASTERIZATION (BEACHHEAD) (Phase 12 #1) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice of FLAGSHIP #7 —
> DETERMINISTIC GPU NAVMESH + PATHFINDING (`hf::nav`, header `engine/nav/navmesh.h`). This BEACHHEAD
> establishes the core integer primitive: rasterize input world triangles into an INTEGER voxel
> HEIGHTFIELD of solid spans, bit-exact CPU↔Vulkan↔Metal. A Recast-style walkable-space pipeline
> (rasterize → walkable-filter → watershed regions → contour → polygonize → integer A*) is the AI
> pillar the engine is missing; UE5 ships Recast/Detour as a headline and Hazard Forge has ZERO nav
> code today (total greenfield). The whole flagship reuses the proven mc.h/fpx.h machinery
> (count→scan→emit compaction, host-snapped integers, single-thread serial mirrors) and adds ZERO new
> RHI. Branch: `slice-nav1`. See [[hazard-forge-nav-roadmap]].

**Goal:** Add `engine/nav/navmesh.h` (header-only, pure-integer, namespace `hf::nav`, NO backend symbols,
NO `<cmath>` on the bit-exact path) with an integer heightfield: quantize input world-space triangles to
an int32 voxel grid and rasterize each triangle into per-column SOLID SPANS `{ymin, ymax, area}`. Build
the variable-length per-column span list with the count→scan→emit GPU compaction (copied verbatim from
`mc.h`/`fpx.h`). Add `shaders/nav_raster.comp.hlsl` (int32-only → Metal-MSL-native), a
`--nav-raster-shot` (Vulkan) / `--nav-raster` (Metal) showcase, the `nav_spans` integer golden (CPU-colored
from the integer span read-back → strict ZERO-DIFFERING-PIXEL cross-backend), and `tests/nav_test.cpp`
pinning CPU==shader span math.

## Design call: INTEGER bit-exact beachhead (the strongest cross-backend proof)
NAV1 is pure integer by construction (the mc.h/fpx.h discipline): the input triangles are HOST-SNAPPED to
an int32 voxel grid, the shader copies the CPU rasterization math VERBATIM and does ZERO float, and the
span buffer is `memcmp`'d GPU==CPU with ZERO tolerance. The golden is the CPU-colored integer span buffer
→ literally Vulkan==Metal BIT-IDENTICAL (the strict zero-differing-pixel bar, NOT the float visresolve
bar). int32-only → `nav_raster.comp` MSL-generates natively and runs as a true GPU pass on BOTH backends
(no int64, so no Vulkan-only/CPU-Metal split — unlike fpx_integrate/fpx_solve).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The structural template (count→scan→emit variable-output compaction):** `engine/render/mc.h` —
  `mc_count`/`mc_scan`/`mc_emit` (the three-pass per-cell count → single-thread exclusive prefix-sum →
  per-cell emit at offset). `shaders/mc_count.comp.hlsl` + `mc_scan.comp.hlsl` (`[numthreads(1,1,1)]`
  serial scan) + `mc_emit.comp.hlsl`. Mirror this EXACTLY for spans: nav_raster_count (spans per column)
  → nav_raster_scan (exclusive prefix-sum of per-column counts) → nav_raster_emit (write spans at offset).
  Also `fpx_pair_count/scan/emit` (`engine/sim/fpx.h` + `shaders/fpx_pair_*.comp.hlsl`) is the int32-only
  twin of the same pattern — copy its int32 discipline (no int64) so nav_raster stays Metal-native.
- **Host-snap to integer grid:** `mc.h` `kFixed` host-snap (world float → int via `round(v*scale)`); the
  `fpx.h::snap`/`kOne` discipline. NAV1 picks a voxel cell size `cs` + cell height `ch`; cell coords are
  `floor((p - bmin) / cs)`. **Use `fpx.h::FloorDiv` (engine/sim/fpx.h) for negative coords** (deterministic
  floor division, not C truncation) so triangles straddling the origin quantize correctly.
- **Integer compare discipline:** `fpx.h::AabbOverlap` (six int32 compares) for the triangle-AABB ∩
  column test (no products, no float).
- **Showcase + integer-golden discipline (the bar to copy):** an INTEGER golden slice — e.g.
  `--mc-classify-shot`/`--fpx-pairs-shot`: run the pass, ReadBuffer the integer result, CPU-color it into
  a debug image (so the golden is bit-identical cross-backend), confirm GPU==CPU memcmp + two-run
  determinism. NOT the float visresolve bar.
- **RHI compute envelope (must fit, ZERO additions):** `engine/rhi/rhi.h` — `BindComputePipeline`,
  `BindStorageBuffer`, `ComputePushConstants`, `DispatchCompute`, `ComputeToComputeBarrier`, `ReadBuffer`
  (lines ~412–440, ~610). Reuse the VT1/MC1/FPX1 pattern verbatim.
- **Shader build wiring:** `samples/hello_triangle/CMakeLists.txt` (the per-slice shader compile list +
  the `/STACK:16777216` link option already present) and `metal_headless/CMakeLists.txt` `hf_gen_msl`
  allowlist (ADD `nav_raster_count/scan/emit` — they are int32 → Metal-native, unlike the int64
  fpx_integrate which is intentionally absent). `metal_headless/visual_test.mm` (`RunNav*Showcase` +
  `#include "nav/navmesh.h"`). `engine/editor/introspect.cpp` (+`deterministic-navmesh-rasterization`
  feature + `--nav-raster-shot` showcase) + `tests/introspect_test.cpp` + the introspect JSON golden.
  `scripts/verify.ps1` (`nav_spans` in the Mac `$Goldens` loop + `--nav-raster-shot` in `$vkShots`).

## Design decisions (locked)
1. **The heightfield model.** `Heightfield{ bmin, bmax (int3 voxel bounds), cs, ch, w, h (column grid
   dims) }`; a `Span{ uint ymin, ymax; uint area; }` (area = walkable-flag placeholder = 1 for NAV1, the
   NAV2 walkable-filter sets it for real). Per column `(x,z)` a list of solid spans sorted by `ymin`,
   non-overlapping (merge touching/overlapping spans — deterministic integer merge). All coords int32
   voxel units. World→voxel is host-snapped ONCE at build (the float bmin/cs are build-time constants,
   NOT in the per-voxel sim).
2. **Rasterization (the bit-exact core).** For each input triangle: host-snap its 3 verts to int32 voxel
   coords; compute its int32 voxel AABB; for each covered column `(x,z)` in the AABB, compute the
   triangle's integer y-interval over that column's cell center (a conservative integer span — the Recast
   `rasterizeTri` discipline, but integer: clamp the triangle's y-range over the cell to `[ymin,ymax]`).
   Emit/merge that span into the column. **Pure int32, copied verbatim CPU↔shader.** Conservative (a
   column the triangle's AABB covers but the triangle misses emits no span — exact integer in/out by the
   edge-function sign test, `fpx`/`swraster` integer-edge discipline).
3. **GPU pipeline (count→scan→emit, the mc.h mirror).** Pass 1 `nav_raster_count`: one thread per triangle
   (or per triangle×column-row), count spans contributed per column → `gColCount[col]` (InterlockedAdd —
   the existing storage-buffer atomic, NO new RHI). Pass 2 `nav_raster_scan`: `[numthreads(1,1,1)]`
   exclusive prefix-sum of `gColCount` → `gColOffset` (the mc_scan/fpx_pair_scan single-thread mirror →
   bit-exact). Pass 3 `nav_raster_emit`: write each column's spans into the compacted `gSpans` buffer at
   its offset, in a DETERMINISTIC order (sort by ymin; tie-break lowest — the fpx source-order discipline).
   **All int32 → MSL-native on Metal (in `hf_gen_msl`).** If the per-column emit needs a stable sort,
   keep it a small fixed insertion sort in integer key order (deterministic, race-free).
4. **Showcase `--nav-raster-shot <out>` (Vulkan, main.cpp) AND `--nav-raster` (Metal, visual_test.mm —
   WIRE BOTH; confirm visual_test.mm + `#include "nav/navmesh.h"`).** Scene: a deterministic test mesh — a
   ground plane + a few ramps/steps/boxes (host-snapped) — rasterized into the heightfield. ReadBuffer the
   integer `gSpans` + `gColOffset`; **memcmp GPU == the CPU `RasterizeTriangleSpans` reference (the
   make-or-break proof)**; CPU-color a top-down or side debug image of the spans (e.g. per-column span
   count / span heights as colored cells) → `tests/golden/metal/nav_spans.png` (baked on the Mac by the
   CONTROLLER — DO NOT commit). The Metal showcase runs the IDENTICAL GPU passes (int32 → native), so the
   span buffer is byte-identical cross-backend by construction.
5. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance / GPU==CPU (the make-or-break):** `gSpans` + `gColOffset` equal the CPU
     `RasterizeTriangleSpans` reference byte-for-byte (full-buffer memcmp). Print
     `nav-raster: {tris:<T>, columns:<C>, spans:<S>} GPU==CPU BIT-EXACT`.
   - **(2) determinism (same backend):** two runs → the span buffer BYTE-IDENTICAL. Print
     `nav-raster determinism: two runs BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the spans cover a coherent walkable footprint (`spans > 0`, the ground
     plane fills its columns; not empty, not uniform garbage). Print
     `nav-raster coverage: <S> spans across <C> columns (coherent heightfield)`.
   - **(4) empty no-op:** zero input triangles → zero spans → cleared background. Print
     `nav-raster empty: 0 spans (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/nav_spans.png`; do NOT commit it — the CONTROLLER bakes
     on the Mac.** Existing 111 image goldens UNTOUCHED.
6. **Cross-backend bar (INTEGER, strict).** The span buffer is host-snapped-integer → ZERO float on the
   GPU → Vulkan==Metal BIT-IDENTICAL: the golden is the CPU-colored integer read-back and the controller's
   cross-backend check is the STRICT ZERO-DIFFERING-PIXEL compare (NOT the float ~55–60 baseline). Any
   nonzero cross-backend pixel diff is a real bug.
7. **Tests `tests/nav_test.cpp` additions (pure CPU):** host-snap quantize (incl. negative coords via
   FloorDiv); a single triangle over a known column → the expected integer span; span merge (two
   overlapping spans → one); the AABB column-cover set; empty input → empty heightfield. Clean under
   `windows-msvc-asan`. (The rasterization is also golden-verified.)
8. **Introspect.** Add exactly `deterministic-navmesh-rasterization` (features) + `--nav-raster-shot`
   (showcases). Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None expected.** Pure SSBO compute: `BindStorageBuffer` (input tris + gColCount/gColOffset/gSpans),
  `DispatchCompute` + `ComputeToComputeBarrier` between the count/scan/emit passes, `ComputePushConstants`
  (heightfield dims + tri count), `ReadBuffer` — all pre-existing (the MC1/FPX1 set). `rhi.h` +
  `rhi_factory` (baseline 2) + backend dirs UNCHANGED. If a genuinely-new RHI need surfaces, STOP and
  report it. Report the seam.

## Out of scope (YAGNI — later NAV slices)
Walkable filtering (slope/step/height — NAV2), distance field + watershed regions (NAV2/NAV3), contour
tracing + polygonization (NAV4), A* pathfinding (NAV5), the float lit-3D render capstone (NAV6). NAV1 is
ONLY the integer solid-span heightfield + its bit-exact golden. No float anywhere (the render capstone is
NAV6). No agent radius/erosion yet (that's NAV2's walkable pass).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 92) + the new `nav_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--nav-raster-shot` on Vulkan: GPU==CPU memcmp BIT-EXACT + determinism + coverage
   + empty no-op; a coherent heightfield debug image. Run under the Vulkan-validation gate → ZERO VUID in
   the OUTPUT (set BOTH `VK_LAYER_PATH` to the conan `...\.conan2\p\...\layers` dir AND
   `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer LOADED = zero "not found" lines).
3. Metal: `visual_test --nav-raster` → new golden `tests/golden/metal/nav_spans.png`; two runs DIFF 0.0000
   (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm nav_raster_count/scan/emit
   MSL-generate (int32 → in `hf_gen_msl`).** Because the spans are integer, the cross-backend compare is
   the STRICT ZERO-DIFFERING-PIXEL bar (controller-measured), NOT the float ~55–60 baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `nav_spans.png` added; the
   other 111 byte-identical. `git diff master --stat -- tests/golden` = ONLY `nav_spans.png` (metal) + the
   introspect json.
5. Introspect JSON rebaked exactly `+deterministic-navmesh-rasterization` + `--nav-raster-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report the reused compute pattern). `scripts/verify.ps1`
   updated: `nav_spans` golden in the Mac loop + `--nav-raster-shot` in `$vkShots`. New `engine/nav/`
   directory; `engine/physics/` + `engine/sim/fpx.h` UNTOUCHED (nav is additive + parallel).
