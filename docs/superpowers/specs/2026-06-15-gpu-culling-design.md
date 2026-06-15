# Slice AR — GPU-Driven Culling + Indirect Draw (Phase 3 perf #2) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. Builds directly on Slice AQ's frustum math.

**Goal:** Move frustum culling of a large INSTANCED set onto the GPU. A compute shader reads a buffer of
per-instance transforms + the camera frustum planes, tests each instance's bounding sphere, and
*compacts* the survivors into an output instance buffer while atomically counting them and writing a GPU
**indirect draw-args** buffer. A single `DrawIndexedIndirect` then renders exactly the surviving
instances — the draw count is decided on the GPU, never round-tripped to the CPU. This is the canonical
"GPU-driven renderer" capability and a real step beyond a CPU cull loop.

## Why this is verifiable (two independent proofs)

1. **Exact-count proof.** The compute-written surviving-instance count is read back ONCE and asserted to
   equal the **CPU frustum reference** (`engine/render/frustum.h` from AQ run over the same instances).
   Integer equality — a GPU/CPU cull-logic mismatch fails loudly.
2. **Render-equivalence golden.** The `--gpu-cull-shot` image (instanced grid, GPU-culled, indirect
   draw) is captured. On Metal two runs DIFF 0.0000 → new golden `tests/golden/metal/gpu_cull.png`. The
   existing 24 image goldens stay BYTE-IDENTICAL (this slice adds a new showcase, touches no existing
   scene). Additionally the implementer confirms (controller visual review) the GPU-culled grid shows the
   in-frustum instances and omits the out-of-frustum ones.

Determinism: fixed camera/grid/frustum, no time/RNG; the compute compaction uses a stable atomic-append
whose ORDER can vary — so the shader writes each survivor to a DETERMINISTIC slot
(`outIndex = prefix over a fixed scan` OR write at the instance's own index into a fixed-size buffer and
let culled instances collapse via a stable compaction). To keep determinism simple and backend-identical:
**compact by stable scan** — each thread writes survivors in source-index order using a single-workgroup
prefix sum (the grid is small, ≤1024 instances, one workgroup), NOT an unordered `atomicAdd` append
(whose result order is nondeterministic and would break the golden). Document this clearly: atomic-append
is faster but nondeterministic; we use an ordered single-pass compaction for golden-stability. (A future
slice can do multi-workgroup prefix sum; YAGNI now.)

## Design decisions (locked)

1. **Reuse existing RHI compute + storage + instancing.** The engine ALREADY has `IComputePipeline`,
   `BindComputePipeline`, `BindStorageBuffer`, `ComputePushConstants`, `DispatchCompute`,
   `CreateComputePipeline`, `BufferUsage::Storage` (read-write SSBO, explicitly also bindable as a vertex
   stream), instancing (`instanceLayout`, `BindInstanceBuffer`, `DrawIndexedInstanced`). Inspect how
   clustered lighting (Slice AG) drives compute + storage buffers and FOLLOW that exact pattern.

2. **ONE new RHI seam addition: indirect indexed draw.** Add
   `IRHICommandBuffer::DrawIndexedIndirect(IBuffer& argsBuffer, size_t offset = 0)` (default no-op in the
   base like the other draw verbs; implemented in BOTH backends). The args buffer holds the standard
   5×u32 layout `{indexCount, instanceCount, firstIndex, vertexOffset, firstInstance}` — identical
   between `VkDrawIndexedIndirectCommand` and Metal's `MTLDrawIndexedPrimitivesIndirectArguments` (same
   field order/size). The compute shader writes this struct (setting `instanceCount` = survivor count,
   the rest constant for the mesh). Buffer creation must request indirect capability: add
   `BufferUsage::Indirect` to the enum (Vulkan → `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT`; Metal indirect
   buffers are just `MTLBuffer`s, no special usage) OR allow a Storage buffer to also be indirect-usable
   — pick the minimal additive change and document it. Vulkan impl: `vkCmdDrawIndexedIndirect(cb, buf,
   offset, 1, 0)`. Metal impl: `drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:
   indirectBuffer:indirectBufferOffset:` (single indirect draw, no indirect-command-buffer machinery).
   These `vk*`/`MTL*` calls live ONLY inside the backend dirs — ZERO seam leakage.

3. **Compute cull shader `shaders/cull.comp.hlsl`.** Inputs (storage/uniform): the source per-instance
   transforms (mat4 each) + per-instance bounding-sphere (or a shared local sphere + per-instance world
   center derived from the transform), the 6 frustum planes (push constant or small UBO), instance count.
   Outputs (storage, read-write): the compacted survivor instance buffer (mat4 stream, consumed as the
   per-instance vertex stream by the draw) + the indirect args buffer (writes `instanceCount`). Single
   workgroup, ordered prefix-sum compaction (see determinism note). The sphere/plane test math must
   MATCH `engine/render/frustum.h` exactly (share the convention; the CPU reference uses the same header,
   so a divergence shows up in the exact-count proof). HLSL→SPIR-V→MSL via the existing toolchain;
   inspect the clustered-lighting compute shader for the storage-binding/MSL conventions.

4. **Showcase `--gpu-cull-shot` (Vulkan) / `--gpu-cull` (Metal).** Build a grid of N=1024 instanced unit
   cubes (deterministic positions) on the ground; place the camera so a known subset is in-frustum.
   Per frame: upload source instances + frustum → dispatch `cull.comp` → barrier →
   `DrawIndexedIndirect(argsBuffer)` binding the compacted survivor buffer as the per-instance stream.
   Print `gpu-cull: {drawn: <gpu>, cpuRef: <cpu>, total: 1024}` and assert `gpu == cpuRef`. Capture one
   frame. Reuse the existing instanced lit pipeline (the per-instance mat4 stream at locations 7-10,
   stride 64) for the draw.

5. **Goldens.** New `tests/golden/metal/gpu_cull.png` (Metal two-run DIFF 0.0000). Existing 24 image
   goldens UNTOUCHED. Introspect JSON intentionally rebaked with exactly `gpu-driven-culling` (features) +
   `--gpu-cull-shot` (showcases) — same pattern as AQ; no other JSON drift.

6. **Unit test `tests/gpu_cull_test.cpp`** — CPU-side test of the SAME ordered-compaction + frustum logic
   the shader implements (a CPU mirror function in a shared header or in the test): given a set of
   instance centers + a frustum, the compacted survivor list (and count) matches a brute-force reference,
   IN ORDER. This pins the determinism contract without needing the GPU. Clean under `windows-msvc-asan`.

## RHI seam additions (summary)
- `BufferUsage::Indirect` (or indirect-capable flag) — additive enum.
- `IRHICommandBuffer::DrawIndexedIndirect(IBuffer&, size_t offset)` — additive, default no-op, both
  backends implement. All `vk*`/`MTL*` indirect-draw calls stay INSIDE backend dirs.
- Compute/storage/instancing: NO new seam (already present). New non-backend files
  (`shaders/cull.comp.hlsl`, `tests/gpu_cull_test.cpp`, any shared CPU-mirror header) add ZERO backend
  symbols. Seam grep stays at the benign baseline (3).

## Out of scope (YAGNI)
Multi-workgroup prefix sum / >1 workgroup compaction, GPU occlusion (Hi-Z) culling, per-instance LOD
selection, indirect-command-buffer batching of many draws, GPU-driven culling of the non-instanced scene
graph, atomic-append unordered compaction. One instanced grid, single-workgroup ordered GPU cull, single
indirect indexed draw, exact-count + render-equivalence verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 24) + new `gpu_cull_test` (ordered compaction +
   frustum vs brute-force reference). Clean under `windows-msvc-asan`.
2. **Exact-count proof:** `--gpu-cull-shot` prints `gpu == cpuRef` (the run asserts/loudly fails on
   mismatch); the GPU survivor count equals the CPU `frustum.h` reference over the identical instances.
3. `--gpu-cull-shot` on Windows/Vulkan: controller visual review — the in-frustum grid instances render,
   out-of-frustum ones absent; image coherent.
4. Metal: `visual_test --gpu-cull` → new golden `tests/golden/metal/gpu_cull.png`; two runs DIFF 0.0000;
   the GPU count matches the CPU reference on Metal too.
5. **Render-invariance of existing scenes:** `git diff master --stat -- tests/golden/metal` shows ONLY
   `gpu_cull.png` added; all 24 existing image goldens byte-identical.
6. Introspect JSON rebaked with exactly `gpu-driven-culling` + `--gpu-cull-shot`; introspect test updated;
   no other JSON drift.
7. Seam grep clean (3 benign; the new `DrawIndexedIndirect`/`BufferUsage::Indirect` are pure-interface in
   `engine/rhi/` with backend impls inside the backend dirs — no `vk*`/`MTL*` above the seam).
   `scripts/verify.ps1` updated to include the new `gpu_cull` image golden in the Mac round-trip loop.
