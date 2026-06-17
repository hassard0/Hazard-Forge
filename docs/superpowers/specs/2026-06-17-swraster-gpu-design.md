# Slice SW2 — Nanite Software-Raster Slice 2: GPU Compute Rasterizer (atomic-min into the vis-buffer SSBO) (Phase 8 #2) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 2nd software-raster slice
> (after SW1's CPU reference): a COMPUTE shader scan-converts cluster triangles into a `depth|id` visibility buffer
> SSBO via `InterlockedMin`, proven BIT-IDENTICAL to SW1's `swraster.h` CPU reference at EVERY pixel. The
> make-or-break: the GPU runs the SAME integer edge math over the SAME host-snapped integer `ScreenVert`s (zero GPU
> FP) + a commutative `InterlockedMin` → the GPU vis-buffer is provably the CPU reference, on any vendor. NO new RHI
> (a `RWStructuredBuffer<uint>` + `InterlockedMin`, the existing `InterlockedAdd` family). Namespace
> `hf::render::vg`. Branch: slice-sw2-swrastergpu. See [[hazard-forge-swraster-roadmap]].

**Goal:** Add `shaders/swraster.comp.hlsl` (the GPU compute rasterizer, the SW1 integer edge math copied verbatim)
+ a `--swraster-gpu-shot` showcase that dispatches it over a clustered mesh's triangles, reads back the vis-buffer
SSBO, and proves it byte-identical to `swraster.h::RasterClusters` over the SAME inputs. Make-safe: a NEW shader +
NEW showcase + NEW golden; nothing existing changes. The GPU==CPU bit-identity is guaranteed by the integer math +
the host-snapped verts + the commutative atomic-min.

## Reuse map (file:line)
- **SW1 `engine/render/swraster.h`** — `SwVisBuffer`, `kSwClear`, `PackSw`/`UnpackSw`, `ScreenVert`,
  `ProjectToScreenVert`, the int64 edge functions + top-left fill rule + `RasterTriangle`/`RasterClusters`. The GPU
  shader copies `RasterTriangle`'s integer edge/fill/min math VERBATIM; the showcase's CPU reference IS
  `RasterClusters`.
- **The compute + atomic surface** — `BufferUsage::Storage` (rhi.h:166), `BindComputePipeline`/`BindStorageBuffer`/
  `ComputePushConstants`/`DispatchCompute` (rhi.h:412-426), `ComputePipelineDesc{storageBufferCount,
  pushConstantSize, threadsPerGroupX}` (rhi.h:168-185), `ComputeToFragmentBarrier`/`ReadBuffer` (rhi.h:435,616).
  `InterlockedAdd` precedent: `shaders/autoexposure_histogram.comp.hlsl:81` (`InterlockedMin` is the same intrinsic
  family — lowered by the existing toolchain, NO MSL-2.2 needed since it's an atomic on a device `uint` buffer).
- **DW `--visbuffer-shot` Proof + the ReadBuffer/CPU-color discipline** + `meshlet.h:79` `hashColor`.
- **`meshlet.h` / `cluster_cull.h`** — a clustered mesh's triangles (e.g. `SphereGeometry` → `BuildMeshlets`) as
  the rasterizer input.

## Design decisions (locked)

1. **The vis-buffer is a `RWStructuredBuffer<uint>` SSBO of `w*h` entries** (NOT an R32_Uint RT — the SW path writes
   via atomic-min, and the engine has no storage-image path; SSBO is the validation-clean fit). Cleared to
   `kSwClear` (a tiny clear-compute dispatch `gVis[i] = kSwClear`, or a host upload of a kSwClear-filled buffer).

2. **`shaders/swraster.comp.hlsl` (NEW).** ONE thread per cluster-triangle. Inputs (SSBOs): `gScreenVerts`
   (`StructuredBuffer<ScreenVert>` — the host-snapped integer screen verts, so the GPU does ZERO FP — pass `int x,
   int y, uint z` per vert, std430-packed), `gTriangles` (per-triangle: the 3 vertex indices + the packed `visId =
   PackVisId(clusterID, triLocal)`), the `gVis` output (`RWStructuredBuffer<uint>`). Push/UBO: `w, h, triCount`.
   Each thread `t < triCount`: read its 3 `ScreenVert`s + `visId`, run the VERBATIM SW1 `RasterTriangle` integer
   edge/fill-rule/bbox loop, and per covered pixel `InterlockedMin(gVis[y*w + x], PackSw(depthQ, visId))`. Copy the
   int64 edge functions + the top-left fill rule + the flat min-depth VERBATIM from `swraster.h` (the shared-math
   rule). `ComputePipelineDesc{ storageBufferCount = 3-4, threadsPerGroupX = 64 }`. NO new RHI. Only `[[vk::binding]]`
   + `HF_MSL_GEN` above-seam. (NO `--msl-version 20200` needed — atomic-min on a device uint buffer is plain MSL;
   confirm on the Mac.)
   - **int64 in the shader:** HLSL SM6 supports `int64_t`; if DXC/spirv-cross/MSL int64 support is shaky on a
     backend, the edge products fit in int64 because the snapped coords are bounded (`w,h <= 4096`, `kSub=16` → coords
     `< 2^16`, products `< 2^32`, the subtraction `< 2^33` → fits int64; assess whether a careful 32-bit/`int` path
     with bounded inputs suffices — but PREFER int64 to match the CPU exactly; if int64 isn't available cross-vendor,
     bound the inputs so the edge function fits int32 and use int32 BOTH in `swraster.h` AND the shader so they still
     match — document the chosen width + the bound. The make-or-break is CPU and GPU using the IDENTICAL integer
     width + ops.)

3. **Showcase `--swraster-gpu-shot <out>` (Vulkan, main.cpp) AND `--swraster-gpu` (Metal, visual_test.mm — WIRE
   BOTH; confirm visual_test.mm in the diff + `#include render/swraster.h`).** A clustered mesh (e.g.
   `SphereGeometry` → `BuildMeshlets` → several hundred triangles) projected to a vis-buffer (e.g. 512×512) via
   `ProjectToScreenVert` (host) → upload `gScreenVerts` + `gTriangles` → clear `gVis` to kSwClear → dispatch
   `swraster.comp` → `ReadBuffer` `gVis`. PROOFS (fail loudly):
   - **(1) GPU==CPU bit-identical (make-or-break):** CPU-run `swraster.h::RasterClusters` over the SAME
     `gScreenVerts` + triangles → a CPU `SwVisBuffer`; `memcmp(gpuVis, cpuVis.packed) == 0` at EVERY pixel (NOT
     interior-only — the integer math makes the full frame bit-exact). Print `swraster-gpu GPU==CPU: BIT-IDENTICAL
     (<W>x<H>, <covered> covered)`.
   - **(2) determinism / atomic-min order-independence:** two GPU dispatches → `ReadBuffer` → `memcmp` byte-identical
     (the `InterlockedMin` is commutative → thread-race CANNOT change the result). Print `swraster-gpu determinism:
     two dispatches BYTE-IDENTICAL`.
   - **(3) sub-pixel coverage:** include a sub-pixel triangle the HW raster would miss; assert the GPU vis-buffer
     covers it (the SW raster's raison d'être). Print `swraster-gpu sub-pixel: COVERED`.
   - **(4) disabled-path no-op:** `triCount=0` / a `swRasterEnabled=false` dispatch → `gVis` stays all-`kSwClear`
     (byte-identical to the cleared buffer).
   - **Golden** = the GPU vis-buffer CPU-colored (`hashColor(visId >> kTriIdBits)`, background → clear) →
     `tests/golden/metal/swraster_gpu.png` (the clustered mesh software-rasterized — per-cluster hash-colored regions
     with correct depth occlusion). CPU-colored from the read-back integer SSBO → **identical both backends by
     construction** → DIFF 0.0000 (gate on compare.sh EXIT CODE). Print `swraster-gpu: {tris:T, covered:N}`. Existing
     92 image goldens UNTOUCHED. **GOLDEN DISCIPLINE: ONLY `tests/golden/metal/swraster_gpu.png`; do NOT commit it —
     the CONTROLLER bakes it on the Mac.**

4. **Determinism / cross-backend.** The GPU consumes the host-snapped integer `ScreenVert`s (zero GPU FP), runs the
   integer edge math (vendor-independent coverage), and resolves depth via the commutative `InterlockedMin` → the
   GPU vis-buffer is bit-identical to the CPU `swraster.h` reference AND across Vulkan/Metal. (No interior-only
   caveat — the integer coverage is exact at edges too, unlike a hardware rasterizer.) Run under the Vulkan sync-
   validation gate → the clear→raster→readback barriers SYNC-HAZARD-free.

5. **Tests `tests/swraster_test.cpp` additions (pure CPU):** the GPU shader's edge/fill/min math is the SW1
   `RasterTriangle` (already unit-tested); add a test that `RasterClusters` over a clustered `SphereGeometry`
   (the showcase scene) is deterministic + watertight at cluster seams (no gap/double-cover) — the CPU oracle the
   GPU memcmp's against. Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `nanite-software-raster-gpu` (features) + `--swraster-gpu-shot` (showcases).

## RHI seam additions (summary)
- **None.** `RWStructuredBuffer<uint>` (`BufferUsage::Storage`) + `InterlockedMin` (the existing `InterlockedAdd`
  family) + `BindStorageBuffer`/`DispatchCompute`/`ReadBuffer`/`ComputeToFragmentBarrier`. New non-backend code
  (`swraster.comp.hlsl`, the showcase, the test additions) adds ZERO above-seam backend symbols. rhi.h + rhi_factory
  (baseline 2) + the backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — SW3 and beyond)
The HW-large/SW-small hybrid dispatch (SW3), feeding the SW vis-buffer into the DX resolve (SW4), perspective-
correct interpolated depth (flat min-depth first), persistent-thread cluster binning / work distribution (one
thread per triangle first — fine at this scale), 64-bit visibility, any SW==HW byte-exact claim. ONE GPU compute
software rasterizer with a full-frame GPU==CPU bit-identical proof + atomic-min determinism + sub-pixel coverage +
disabled-path no-op and the software-rasterized clustered-mesh golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 89) + the new `swraster_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--swraster-gpu-shot` on Vulkan: the clustered mesh renders as software-rasterized
   hash-colored cluster regions with correct depth occlusion (coherent); `swraster-gpu GPU==CPU: BIT-IDENTICAL` +
   `determinism: two dispatches BYTE-IDENTICAL` + `sub-pixel: COVERED` + the disabled no-op; the `swraster-gpu:
   {...}` line deterministic. Run under the AT Vulkan-validation gate → ZERO errors (the compute clear→raster→
   readback SYNC-HAZARD-free).
3. Metal: `visual_test --swraster-gpu` → new golden `tests/golden/metal/swraster_gpu.png`; two runs DIFF 0.0000
   (gate on compare.sh EXIT CODE). The GPU==CPU + determinism proofs also pass on Metal (the integer math + atomic
   match the CPU). **Confirm visual_test.mm in the diff; confirm swraster.comp MSL-generates + compiles + the
   InterlockedMin lowers on Metal (atomic on a uint buffer — should need no MSL-2.2; if it fails, report).**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `swraster_gpu.png` added; the
   other 92 byte-identical. `git diff master --stat -- tests/golden` = ONLY `swraster_gpu.png` (metal) + the 2-line
   introspect json — NO loose `tests/golden/swraster_gpu.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+nanite-software-raster-gpu` + `--swraster-gpu-shot`; introspect test updated.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new `swraster_gpu`
   image golden in the Mac round-trip loop AND `--swraster-gpu-shot` in the `$vkShots` validation gate.
