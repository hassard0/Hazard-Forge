# Slice MC1 — GPU Isosurface Meshing: PER-CELL MARCHING-CUBES CASE CLASSIFICATION (Beachhead, Phase 10 #1) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The BEACHHEAD of FLAGSHIP #5:
> GPU Isosurface Meshing via Marching Cubes — extract a triangle mesh from a scalar (voxel/SDF) field, the foundation
> of real-time procedural & destructible geometry. This first slice builds the integer core: a compute pass that
> classifies each voxel CELL into its 8-bit Marching-Cubes case index (the 8 corner-sign comparisons), proven GPU==CPU
> BIT-EXACT, with an integer case-index debug-viz golden that is cross-backend BIT-IDENTICAL (the strict zero-
> differing-pixel bar, like VT1 / swraster / VSM — NOT the float visresolve-bar). Pure integer — NO rendering, NO
> triangle emission, NO new RHI. The structural twin of VT1 (page-needed marking) and SW2 (host-snapped integer →
> GPU bit-exact replay), applied to voxel-cell classification. Namespace `hf::render::mc`. Branch:
> `slice-mc1-classify`. Grounded by a read-only Plan scout @ master `5eb3df2` (2026-06-17). See [[hazard-forge-project]].

**Goal:** Add `engine/render/mc.h` (the pure-CPU integer MC core: `VoxelField`, the locked corner/edge numbering,
`CaseIndex`, `SampleField`, `ClassifyCells` CPU reference) + `shaders/mc_classify.comp.hlsl` (one thread per cell, the
classifier copied VERBATIM, writes `gCases[cell]`) + a `--mc-classify-shot` (Vulkan) / `--mc-classify` (Metal)
showcase that classifies a fixed procedural SDF field, reads back the case-index buffer, proves it BIT-EXACT vs the
CPU reference, and bakes an integer case-index Z-slice golden. Make-safe: a NEW header + NEW shader + NEW showcase +
NEW golden; nothing existing changes. The cross-backend bit-identity is guaranteed by the pure-integer corner-sign
compares (no transcendental, no float) + the order-independent per-cell write.

## Why this is the right beachhead (the VT1 / SW2 analog)
Marching Cubes decomposes into exactly the integer primitives the engine has golden-proven cross-backend: cell
classification = 8 integer compares → an 8-bit case index; topology = integer table lookups (MC2+); midpoint vertex
placement = integer grid coords (MC3); even interpolated placement stays fixed-point (MC4, the swraster `kSub` trick).
This slice is the classification — the direct analog of VT1's `MarkFeedbackPages` (an order-independent integer set
write proven GPU==CPU bit-exact) and SW2's host-snapped-integer/GPU-zero-float replay. It is arguably a CLEANER
integer fit than VT1 (no `floor(u*pps)` float boundary at all — pure compares). Per-cell writes are independent →
race-free → no atomics.

## The integer core (the cross-backend crux)
- **`struct VoxelField { int nx, ny, nz; std::vector<int32_t> scalar; }`** — the host-filled INTEGER scalar field
  (a quantized SDF/density), `scalar[(z*ny + y)*nx + x]`, flat row-major (the swraster vis-buffer layout). `nx/ny/nz`
  = the CORNER counts per axis; cells span `[0, nx-1) x [0, ny-1) x [0, nz-1)`, so `cellCount = (nx-1)(ny-1)(nz-1)`.
- **Locked MC corner numbering** (the canonical 0–7 cube-corner convention) + edge numbering (0–11), `static_assert`/
  comment-pinned like swraster's depth-bit budget. Corner `i` local offset `(dx,dy,dz)` is the canonical MC ordering;
  this slice only needs the corner offsets (edges/tables are MC2). Document the exact convention chosen so MC2's
  tables match.
- **`uint8_t CaseIndex(const int32_t corner[8], int32_t isovalue)`** — `idx = Σ_{i<8} ((corner[i] > isovalue) ? 1 : 0)
  << i`. Pure integer compare. THE cross-backend crux: identical on CPU/Vulkan/Metal by construction (no float, no
  transcendental). (Convention: a corner is "inside" the surface when `scalar > isovalue` — pin it; MC2's tables must
  agree.)
- **`int32_t SampleField(const VoxelField&, int x, int y, int z)`** — the host integer field accessor (bounds-checked
  / clamped), used by the host SDF generator + `ClassifyCells`. The procedural field itself is generated host-side by
  a deterministic pure function (the `heightmap.h::Height` discipline — NO RNG/clock/disk).
- **`void ClassifyCells(const VoxelField&, int32_t isovalue, std::vector<uint8_t>& caseOut)`** — the CPU reference:
  `caseOut` sized `cellCount`; for each cell `(cx,cy,cz)`, gather its 8 corner scalars (`SampleField` at the 8 corner
  offsets), `caseOut[cellId] = CaseIndex(corners, isovalue)`. Order-independent → a GPU thread-race CANNOT change it
  (the VT1 set-write argument). `cellId = (cz*(ny-1) + cy)*(nx-1) + cx`.

## Reuse map (file:line)
- **The host-snap → GPU-zero-float / bit-exact-replay discipline (the SW2 pattern to clone):** `shaders/swraster.comp.hlsl:1-33`
  (GPU consumes host integers, math copied VERBATIM, `memcmp` GPU==CPU); `engine/render/swraster.h:85-96` (the
  fixed-point `kSub` lattice, for MC3/MC4 later).
- **The compute marking template:** `shaders/vsm_mark.comp.hlsl` (one thread per item, the classifier copied verbatim,
  3 SSBOs, an `enabled` flag, an order-independent integer write — `mc_classify.comp` mirrors this exactly).
- **The compute + readback surface (NO new RHI):** `BufferUsage::Storage` (`rhi.h:166`),
  `BindComputePipeline`/`BindStorageBuffer`/`DispatchCompute` (`rhi.h:412-426`),
  `ComputePipelineDesc{storageBufferCount, pushConstantSize, threadsPerGroupX}` (`rhi.h:168-185`), `ReadBuffer`
  (`rhi.h:616`).
- **The procedural host-field determinism precedent:** `engine/terrain/heightmap.h:7-9,29` (`Height()` — a deterministic
  pure function, bit-identical cross-target). The MC SDF generator copies this (e.g. a signed-distance sphere/torus
  quantized to `int32` fixed-point).
- **The integer-set debug-viz golden:** the `vt_feedback.png` / `vsm_pages.png` template — CPU-color the read-back
  integer field, `meshlet.h:79` `hashColor`.
- **Showcase + registration patterns:** the VT1 `--vt-feedback-shot`/`--vt-feedback` showcase (main.cpp +
  visual_test.mm), introspect feature+showcase (`engine/editor/introspect.cpp`), `scripts/verify.ps1`
  `$Goldens`/`$vkShots`, `tests/introspect_test.cpp`.

## Design decisions (locked)

1. **`engine/render/mc.h` (NEW, namespace `hf::render::mc`, pure CPU, header-only, 0 above-seam backend symbols,
   mirrors `swraster.h`).** `VoxelField`, the locked corner numbering + offsets, `CaseIndex`, `SampleField`,
   `ClassifyCells`. Stub-declare (NOT implement) the 256-case triangle-count/table type for MC2 so the header shape is
   forward-compatible (a comment + a `// MC2:` marker). NO float on any path.

2. **`shaders/mc_classify.comp.hlsl` (NEW).** ONE thread per cell (`cell < cellCount`). Decompose `cell` →
   `(cx,cy,cz)`, gather the 8 corner scalars from `gField` (b0) at the corner offsets, run `CaseIndex` copied VERBATIM
   from `mc.h`, write `gCases[cell] = caseIdx` (b1, a `RWStructuredBuffer<uint>` — pack the `uint8` case into a `uint`
   slot, or a `uint` per cell for simplicity). A `classifyEnabled=0` push-constant flag → write 0 (the disabled path).
   SSBOs `gField`(b0)/`gCases`(b1)/`gParams`(b2:{nx,ny,nz,isovalue,...}) per the `vsm_mark.comp.hlsl` layout.
   `ComputePipelineDesc{ storageBufferCount=3, threadsPerGroupX=64 }`. NO atomics (independent per-cell writes). Only
   `[[vk::binding]]` + `HF_MSL_GEN` above-seam. Plain integer SSBO write → default MSL gen (NO `--msl-version 20200`;
   add ONLY if the Mac requires it — the DW/VT1 lesson).

3. **Showcase `--mc-classify-shot <out>` (Vulkan, main.cpp) AND `--mc-classify` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm in the diff + `#include "render/mc.h"`).** A fixed `VoxelField` (e.g. nx=ny=nz=33 corners →
   32×32×32 cells) filled host-side with a deterministic SDF (e.g. a centered sphere of radius ~12 cells, the signed
   distance quantized to `int32` fixed-point, isovalue 0) — chosen so a non-trivial spherical-shell subset of cells
   has non-empty cases (the surface passes through them). Upload `gField`, dispatch `mc_classify.comp`, `ReadBuffer`
   `gCases`. CPU-run `ClassifyCells` over the SAME field → the reference. Golden = the case-index field CPU-colored as
   a grid of Z-slices (each slice an `(nx-1)×(ny-1)` image, `hashColor(caseIndex)` per cell, empty cases `0x00`/`0xFF`
   dark; the 32 slices tiled in a grid, e.g. 8×4) → `tests/golden/metal/mc_classify.png`. CPU-colored from the
   read-back integer field → **identical both backends by construction** → DIFF 0.0000 / strict zero differing pixels
   (gate on compare.sh EXIT CODE). The surface cells form concentric rings across slices (a recognizable sphere
   cross-section).

4. **PROOFS (fail loudly; exact print lines):**
   - **(1) GPU==CPU bit-exact (make-or-break):** `memcmp(gpuCases, cpuCases) == 0` over all cells, NO tolerance. Print
     `mc-classify GPU==CPU case set: <N> cells BIT-EXACT`.
   - **(2) disabled-path no-op:** `classifyEnabled=false` → `gCases` stays all-zero (byte-identical to the cleared
     upload). Print `mc-classify disabled: zero cases (no-op)`.
   - **(3) determinism:** two dispatches → `ReadBuffer` → byte-identical. Print `mc-classify determinism: two
     dispatches BYTE-IDENTICAL`.
   - **(4) hand-checked known cells:** assert known configurations — all 8 corners below iso → `0x00`; all 8 above →
     `0xFF`; exactly corner 0 above → `0x01` (per the locked convention). Print `mc-classify known cells: empty=0x00
     full=0xFF corner0=0x01 OK`.
   - **(5) {stats}:** `mc-classify: {field:<nx>x<ny>x<nz>, iso:<v>, cells:<C>, surface-cells:<k>/<C>}` (surface-cells
     = cells with a case index that is neither all-in nor all-out, i.e. the isosurface crosses them).
   - **Golden discipline: ONLY `tests/golden/metal/mc_classify.png`; do NOT commit it — the CONTROLLER bakes it on the
     Mac. No loose `tests/golden/mc_classify.png`.** Existing 99 image goldens UNTOUCHED.

5. **Determinism / cross-backend.** The classification is pure integer compares of host-quantized scalars; the write
   is order-independent; the golden is CPU-colored from the integer read-back → bit-identical across Vulkan/Metal AND
   the GPU==CPU memcmp holds full-field (not interior-only — there is no edge fuzz, unlike a hardware rasterizer). Run
   under the Vulkan sync-validation gate → the upload→dispatch→readback barriers SYNC-HAZARD-free.

6. **Tests `tests/mc_test.cpp` (pure CPU, NEW):** `CaseIndex` truth table (empty/full/each-single-corner/a few mixed
   configs hand-verified); the corner-numbering offsets match the locked convention; `ClassifyCells` over a known
   tiny field (e.g. a 2×2×2-cell field with a planar boundary → known case indices); a sphere SDF → the surface-cell
   count is the shell (sanity bound); `classifyEnabled=false` modeled → all-zero; determinism. Clean under
   `windows-msvc-asan`.

7. **Introspect.** Add exactly `gpu-isosurface-meshing-classify` (features) + `--mc-classify-shot` (showcases).

## RHI seam additions (summary)
- **None.** `BufferUsage::Storage` + `BindComputePipeline`/`BindStorageBuffer`/`DispatchCompute`/`ReadBuffer` — the
  VT1/vsm_mark/froxel-inject precedent. New non-backend code (`mc.h`, `mc_classify.comp.hlsl`, the showcase, the test)
  adds ZERO above-seam backend symbols. `engine/rhi/rhi.h` + `rhi_factory` (dispatch baseline 2) + the backend dirs
  UNCHANGED. Report the seam.

## Out of scope (YAGNI — MC2 and beyond)
Per-cell triangle COUNT via the 256-case table + `InterlockedAdd` (MC2), prefix-sum compaction + vertex/index emission
(MC3, midpoint integer vertices), fixed-point interpolated vertex placement (MC4), rendering the generated mesh /
feeding `BuildMeshlets` → the Nanite cluster pipeline (MC5), field-gradient normals + lit shading (MC6, the first
float / visresolve-bar slice). Real-time editing/CSG, chunked crack-free seams, LOD, Dual-Contouring's float QEF
solve — ALL deferred (the SDF field is host-generated procedurally in-memory, never streamed). ONE compute pass that
classifies cells into MC case indices with a GPU==CPU bit-exact proof + disabled no-op + determinism + hand-checked
known cells and the integer case-index Z-slice golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 90) + the new `mc_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--mc-classify-shot` on Vulkan: a coherent case-index Z-slice viz (the sphere's surface cells
   as concentric rings across slices, hash-colored); `mc-classify GPU==CPU case set: <N> cells BIT-EXACT` + `disabled:
   zero cases` + `determinism: two dispatches BYTE-IDENTICAL` + `known cells: ... OK` + the `{...}` line. Run under the
   Vulkan-validation gate → ZERO errors (the upload→dispatch→readback SYNC-HAZARD-free).
3. Metal: `visual_test --mc-classify` → new golden `tests/golden/metal/mc_classify.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). The GPU==CPU + determinism proofs also pass on Metal (integer math). **Confirm visual_test.mm
   in the diff; confirm mc_classify.comp MSL-generates + the integer SSBO write lowers on Metal (no MSL-2.2; if it
   fails, report).** This is an INTEGER golden → a strict cross-backend pixel compare must show ZERO differing pixels.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `mc_classify.png` added; the other
   99 byte-identical. `git diff master --stat -- tests/golden` = ONLY `mc_classify.png` (metal) + the 2-line introspect
   json — NO loose `tests/golden/mc_classify.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+gpu-isosurface-meshing-classify` + `--mc-classify-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new `mc_classify`
   image golden in the Mac round-trip loop AND `--mc-classify-shot` in the `$vkShots` validation gate.
