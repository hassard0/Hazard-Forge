# Slice MC3 — GPU Isosurface Meshing: PREFIX-SUM COMPACTION + TRIANGLE EMISSION (midpoint vertices) (Phase 10 #3) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 3rd MC slice (after MC1
> classify + MC2 count): turn the per-cell triangle counts into an actual MESH — an exclusive prefix-sum of the counts
> gives each cell its write offset, then each cell EMITS its triangles as cell-edge MIDPOINT vertices (integer grid
> coordinates, half-unit fixed-point) + an index buffer, into fixed-capacity output SSBOs. This is the geometry-
> generating heart of Marching Cubes. Kept INTEGER (midpoint = sum of two integer corner coords in half-units → no
> float), so the vertex+index buffers are GPU==CPU BIT-EXACT and the golden is cross-backend BIT-IDENTICAL. NO new RHI.
> Namespace `hf::render::mc`. Branch: `slice-mc3-emit`. See [[hazard-forge-mc-roadmap]].

**Goal:** Extend `engine/render/mc.h` with the midpoint edge-vertex placement (`EdgeMidpoint` → an `int3` in half-grid
units), the CPU reference mesher (`PrefixSumOffsets`, `EmitCell`, `MarchCells` → vertex+index buffers) + add
`shaders/mc_scan.comp.hlsl` (single-thread exclusive prefix-sum of the per-cell counts → cellTriOffset) +
`shaders/mc_emit.comp.hlsl` (one thread per cell, emits its triangles' midpoint vertices + indices at its offset) + a
`--mc-emit-shot` (Vulkan) / `--mc-emit` (Metal) showcase that meshes the MC1/MC2 sphere field, reads back the vertex +
index buffers, proves them BIT-EXACT vs the CPU mesher, and bakes an integer mesh-projection golden. Make-safe: header
additions + 2 NEW shaders + NEW showcase + NEW golden; MC1/MC2 goldens + everything else UNCHANGED.

## The integer core (extends mc.h)
- **Vertex format:** `struct McVertex { int32_t x, y, z, w; }` — the edge-MIDPOINT position in HALF-grid units
  (`w=0` pad / future use), so a midpoint is a pure integer: for an edge between corners `a,b` at integer grid coords
  `Ca, Cb`, `EdgeMidpoint = Ca + Cb` (each axis), which equals `2 * actual_midpoint` — kept as integer half-units to
  avoid any division/float. (MC4 replaces this with the fixed-point INTERPOLATED position on the same half-unit
  lattice, staying integer.) The cell at `(cx,cy,cz)` has corner `i` at grid coord `(cx,cy,cz) + kCornerOffset[i]`.
- **`McVertex EdgeMidpoint(int cx, int cy, int cz, int edge)`** — `kEdgeCorner[edge]` gives the two corner indices
  `(a,b)`; `Ca = cell + kCornerOffset[a]`, `Cb = cell + kCornerOffset[b]`; return `{Ca.x+Cb.x, Ca.y+Cb.y, Ca.z+Cb.z,
  0}`. Pure integer.
- **`void PrefixSumOffsets(std::span<const uint32_t> counts, std::span<uint32_t> offsetsOut, uint32_t& totalOut)`** —
  the CPU reference exclusive prefix-sum: `offsetsOut[i] = Σ_{j<i} counts[j]`, `totalOut = Σ counts` (== MC2's total).
- **`void EmitCell(int cx, int cy, int cz, uint8_t caseIndex, uint32_t triOffset, std::span<McVertex> vertsOut,
  std::span<uint32_t> idxOut)`** — walk `kTriTable[caseIndex]` in groups of 3 (until -1); for triangle `t` (global
  index `triOffset + t`), for its 3 edges write `vertsOut[3*(triOffset+t)+k] = EdgeMidpoint(cell, edge_k)` and
  `idxOut[3*(triOffset+t)+k] = 3*(triOffset+t)+k` (a triangle SOUP — no cross-cell vertex dedup; dedup is a deferred
  optimization). Winding per the canonical table.
- **`void MarchCells(const VoxelField&, int32_t iso, std::vector<McVertex>& verts, std::vector<uint32_t>& idx,
  uint32_t& triCount)`** — the full CPU mesher (classify → count → `PrefixSumOffsets` → `EmitCell` per cell). The
  reference the GPU memcmp's against; `verts.size() == idx.size() == 3*triCount`.

## Reuse map (file:line)
- **MC1/MC2 (the inputs):** `engine/render/mc.h` — `CaseIndex`/`ClassifyCells` (MC1), `kTriTable`/`kTriCount`/
  `CountCells`/`TotalTriangles` (MC2), `kCornerOffset`/`kEdgeCorner`.
- **The prefix-sum / compaction precedent:** `shaders/gpudriven_cull.comp.hlsl:112-144` (the Hillis-Steele scan +
  ordered compaction `slot = prefix`); the CPU mirror `engine/render/gpu_culled.h:78,107` (`CullAndCompact`, source-
  order deterministic). NOTE: that scan is single-workgroup (≤ group size); for 32768 cells use a **single-thread
  serial exclusive scan** compute pass (the VT2 `vt_alloc.comp` single-thread allocator pattern — inherently
  sequential, `[numthreads(1,1,1)]`, GPU==CPU bit-exact) → `cellTriOffset[cell]`. (A multi-block parallel scan is a
  deferred optimization; single-thread is correct + bit-exact at this scale.)
- **The host-snapped-integer / GPU-zero-float / fixed-point-lattice discipline:** `swraster.h:85-96` (`kSub` half/
  fixed-point lattice — the midpoint half-unit trick mirrors this); `swraster.comp.hlsl:1-33` (GPU math copied
  VERBATIM from the CPU header, memcmp GPU==CPU).
- **Variable-count → host readback:** `gpudriven_cull.comp.hlsl:10-14` + `gpu_culled.h:64-70` (compute writes a count;
  host `ReadBuffer`s it and sizes/draws). Here the host already knows `totalTris` from MC2, so it preallocates the
  output SSBOs to `3*totalTris` before the emit dispatch.
- **Compute + readback surface (NO new RHI):** `BufferUsage::Storage` (`rhi.h:166`), compute (`rhi.h:412-426`),
  compute→compute barrier (`rhi.h:434-437`, between scan→emit), `ReadBuffer` (`rhi.h:616`).
- **Golden + registration:** `meshlet.h:79` `hashColor`; the MC1/MC2 showcase + Z-slice viz template; `scripts/verify.ps1`
  `$Goldens`/`$vkShots`, `engine/editor/introspect.cpp`, `tests/introspect_test.cpp`.

## Design decisions (locked)

1. **Two compute passes, both bit-exact.** (a) `mc_scan.comp.hlsl` — `[numthreads(1,1,1)]` SINGLE-THREAD exclusive
   prefix-sum: one thread walks cells 0..cellCount-1, `gOffsets[cell] = running; running += gCounts[cell]` (the VT2
   single-thread-allocator pattern; serial dependency → single thread; `gid.x!=0` guard). (b) `mc_emit.comp.hlsl` —
   one thread PER CELL (parallel): read `caseIndex` (reclassify from `gField` via `CaseIndex` VERBATIM, OR read a
   persisted case buffer — reclassify is self-contained, pick that), read `triOffset = gOffsets[cell]`, and `EmitCell`
   (the `EdgeMidpoint` + `kTriTable` walk copied VERBATIM from mc.h) writing `gVerts` + `gIdx` at `3*(triOffset+t)+k`.
   Independent per-cell writes to disjoint ranges → race-free, NO atomics. A `meshEnabled=0` push flag → emit nothing
   (the disabled no-op → empty mesh). `gTriTable` (the full 256×16 table) uploaded as an SSBO for the emit. Barriers:
   upload→scan (compute), scan→emit (compute→compute), emit→readback. NO new RHI. Plain integer → NO `--msl-version
   20200`.
2. **Output capacity.** The host runs MC2's count+total first (or computes `TotalTriangles` host-side via `CountCells`
   — simplest + deterministic), preallocates `gVerts` + `gIdx` to exactly `3*totalTris` entries (`totalTris=5240` for
   the sphere → 15720 verts/indices), clears them, then scan+emit. (Worst-case `5*cells*3` preallocation is also fine
   but exact is cleaner.)
3. **Showcase `--mc-emit-shot <out>` (Vulkan, main.cpp) AND `--mc-emit` (Metal, visual_test.mm — WIRE BOTH; confirm
   visual_test.mm + `#include "render/mc.h"`).** The SAME MC1/MC2 sphere field. Upload `gField` + `gCounts`
   (host `CountCells`) + `gTriTable`, preallocate `gVerts`/`gIdx`/`gOffsets`, dispatch scan → emit → `ReadBuffer`
   `gVerts` + `gIdx`. CPU-run `MarchCells` over the SAME field → the reference. Golden = a deterministic 2D
   ORTHOGRAPHIC projection of the generated mesh: for each vertex, project its integer half-unit `(x,y)` to a pixel
   (integer divide to the image grid), splat a point (`hashColor` by cell or a flat color); the result is the sphere's
   surface as a recognizable filled-disk point cloud / silhouette → `tests/golden/metal/mc_emit.png`. CPU-rendered
   from the read-back integer vertices → identical both backends by construction → DIFF 0.0000 / strict zero diff.
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU prefix-offsets bit-exact:** `memcmp(gpuOffsets, cpuOffsets) == 0` over cellCount. Print
     `mc-emit GPU==CPU prefix-sum: <C> cells BIT-EXACT`.
   - **(2) GPU==CPU mesh bit-exact (make-or-break):** `memcmp(gpuVerts, cpuVerts)==0` AND `memcmp(gpuIdx, cpuIdx)==0`
     over `3*totalTris`. Print `mc-emit GPU==CPU mesh: <V> verts + <I> indices BIT-EXACT`.
   - **(3) count consistency:** `totalTris == MC2 total (5240)`, `verts.size()==idx.size()==3*totalTris`, every index
     in range. Print `mc-emit: {tris:<T>, verts:<V>, indices:<I>}`.
   - **(4) known-cell emission:** a hand-checked cell (a known case index) emits the expected triangle count + the
     expected first edge-midpoint vertex. Print `mc-emit known cell: case=0x<XX> tris=<n> v0=(<x>,<y>,<z>) OK`.
   - **(5) disabled-path no-op:** `meshEnabled=false` → `gVerts`/`gIdx` stay cleared. Print `mc-emit disabled: empty
     mesh (no-op)`.
   - **(6) determinism:** two full (scan+emit) runs → byte-identical verts+indices. Print `mc-emit determinism: two
     runs BYTE-IDENTICAL`.
   - **Golden discipline: ONLY `tests/golden/metal/mc_emit.png`; do NOT commit it — the CONTROLLER bakes on the Mac.**
     Existing 101 image goldens UNTOUCHED.
5. **Tests `tests/mc_test.cpp` additions (pure CPU):** `EdgeMidpoint` for known edges (the two corner coords summed);
   `PrefixSumOffsets` (known counts → known offsets + total); `EmitCell` for a known case (right tri count, right
   midpoint verts, indices identity); `MarchCells` over a tiny field → known vertex/index counts + `verts.size()==
   3*triCount`; the index buffer is the identity `[0,3T)`; determinism; `meshEnabled` off modeled → empty. Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add exactly `gpu-isosurface-meshing-emit` (features) + `--mc-emit-shot` (showcases).

## RHI seam additions (summary)
- **None.** `BufferUsage::Storage` + compute (single-thread scan + parallel emit) + compute→compute barrier +
  `ReadBuffer` — the VT2/gpudriven-cull/MC1-MC2 precedent. ZERO above-seam backend symbols. `rhi.h` + `rhi_factory`
  (baseline 2) + backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — MC4+)
Fixed-point INTERPOLATED vertex placement (MC4 — midpoint first, the swraster simplest-deterministic choice),
cross-cell vertex DEDUP / indexed sharing (a triangle soup first; dedup is a deferred optimization), rendering the
mesh (MC5), normals/shading (MC6), a multi-block parallel prefix-sum (single-thread is bit-exact at this scale). ONE
prefix-sum + emit producing a midpoint-vertex triangle-soup mesh with GPU==CPU bit-exact vertex+index buffers + the
prefix-sum proof + count consistency + known-cell + disabled no-op + determinism and the integer mesh-projection
golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 91) + the new `mc_test` emit cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--mc-emit-shot` on Vulkan: a coherent 2D mesh projection (the sphere surface as a filled-disk
   point cloud); all 6 proof lines. Run under the Vulkan-validation gate → ZERO errors (the scan→emit→readback
   SYNC-HAZARD-free).
3. Metal: `visual_test --mc-emit` → new golden `tests/golden/metal/mc_emit.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). The GPU==CPU prefix-sum + mesh memcmp + determinism proofs also pass on Metal (integer math).
   **Confirm visual_test.mm in the diff; confirm mc_scan.comp + mc_emit.comp MSL-generate (plain integer, no MSL-2.2).**
   Integer golden → a strict cross-backend pixel compare must show ZERO differing pixels.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `mc_emit.png` added; the other 101
   byte-identical (MC1/MC2 untouched). `git diff master --stat -- tests/golden` = ONLY `mc_emit.png` (metal) + the
   introspect json.
5. Introspect JSON rebaked exactly `+gpu-isosurface-meshing-emit` + `--mc-emit-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `scripts/verify.ps1` updated: `mc_emit` golden in the Mac loop +
   `--mc-emit-shot` in `$vkShots`.
