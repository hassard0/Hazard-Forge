# Slice CD — Fully-GPU-Driven-Culled Pass (compute-cull → MDI + bindless) — Phase 4 #29 — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. THE end-state modern
> GPU-driven renderer: the GPU DECIDES what to draw (compute frustum-cull → compacted draw list + count)
> AND draws it in one pass (MDI + bindless). Combines AR (GPU cull) + BM (MDI) + BZ (bindless) + CB
> (combined pass). Render-invariant by construction.

**Goal:** A compute shader frustum-culls the per-draw instances, COMPACTS the survivors into the per-draw
storage buffer (model + material + texIndex), and writes the multi-draw-indirect command COUNT
(survivor count). Then ONE `vkCmdDrawIndexedIndirect` (drawCount = the GPU-written count) + ONE bindless
texture bind renders exactly the survivors. The GPU decides AND draws in one pass. Proven byte-identical
to a CPU-frustum-culled reference + the GPU survivor-count equals the CPU `frustum.h` reference.

## Reuse map (every piece is already proven)

- **AR (`shaders/cull.comp.hlsl`, `engine/render/frustum.h`, `gpu_cull.h`)** — compute frustum-cull +
  ordered single-workgroup prefix-sum compaction + writing the indirect-args `instanceCount`; the CPU
  `frustum.h` reference for the count proof. (AR culled an instanced grid; here it compacts the per-draw
  list and writes the MDI `drawCount`.)
- **BM (`engine/render/mdi.h`, `lit_mdi.vert`, `DrawIndexedMultiIndirect`, per-draw SSBO + gl_DrawID)** —
  the MDI command buffer + per-draw data + the byte-identical-reference pattern.
- **BZ (`engine/render/bindless.h`, `vulkan_bindless.*`, `lit_bindless.frag`, bindless array)** — the
  texture array + `NonUniformResourceIndex(texIndex)`.
- **CB (`engine/render/gpu_driven.h`, `lit_gpudriven.*`)** — the combined per-draw (model+material+texIndex)
  + the 1-draw-1-bind pass.

This slice is the COMPOSITION: the compute pass writes the CB per-draw SSBO (survivors only, compacted) +
the MDI command's `drawCount`; the CB GPU-driven pass then renders from that GPU-produced buffer.

## Design decisions (locked)

1. **Compute cull + compact (extend `shaders/cull.comp.hlsl` or a new `gpudriven_cull.comp.hlsl`).** Inputs:
   the FULL per-draw list (N objects: model + world bounding sphere + material + texIndex) + the 6 frustum
   planes (from the unjittered view-proj, like AR). Per object: test the world bounding sphere vs the
   frustum (`frustum.h`/AR sphere test); SURVIVORS are compacted (ordered single-workgroup prefix-sum, the
   AR determinism trick — survivors in source-index order, NOT unordered atomicAdd) into the OUTPUT per-draw
   SSBO; the survivor count is written into the MDI command's `drawCount` field (and the command's
   `instanceCount`/indexCount set for the mesh). Deterministic + matches a CPU reference. The CPU-side
   builder (`engine/render/gpu_culled.h`, pure CPU) mirrors the cull+compact for the unit test + the count
   reference.

2. **Render path.** Upload the full per-draw list + frustum → dispatch the cull/compact compute →
   barrier → `BindBindlessTextures` once + bind the compacted per-draw SSBO + the indirect-command buffer
   → ONE `DrawIndexedIndirect`/`DrawIndexedMultiIndirect` with the GPU-written drawCount → renders only the
   survivors with their bindless textures. Reuse CB's `lit_gpudriven.{vert,frag}` verbatim (they index
   PerDraw[gl_DrawID] — now the COMPACTED survivor buffer). Read back the survivor count for the proof.

3. **Showcase `--gpucull-draw-shot <out>` (Vulkan) / `--gpucull-draw` (Metal).** A scene of N=100+ objects
   where the camera sees a subset (some off-screen, like AR's overview). Render via the
   compute-cull→GPU-driven pass. Print `gpucull-draw: {total:N, drawn:K, cpuRef:K, drawCalls:1,
   textureBinds:1}` where the GPU `drawn` == the CPU `frustum.h` reference `cpuRef` (assert; fail on
   mismatch) AND INTERNALLY render a CPU-frustum-culled per-object BOUND reference (draw only the same K
   survivors per-object) → assert `gpuCulledImage == refImage` byte-identical. New golden
   `tests/golden/metal/gpucull_draw.png` (Metal two runs DIFF 0.0000). Existing 51 image goldens UNTOUCHED.

4. **Determinism.** Ordered compaction (no unordered atomics), fixed scene/camera/frustum, no time/RNG.
   Two runs byte-identical. The survivor set + order are a pure function of the scene + frustum.

5. **Tests `tests/gpu_culled_test.cpp` (pure CPU, no GPU):**
   - **Cull+compact reference:** for a known object set + a known frustum, the CPU cull+compact yields
     exactly the in-frustum survivors, compacted in source-index order; the count matches a brute-force
     `frustum.h` reference; 0 false-culls (conservative sphere test).
   - **Per-draw carry:** each survivor's compacted PerDraw entry carries the right model + material +
     texIndex (the compaction preserves the per-draw data).
   - **Determinism:** two builds → bit-identical compacted buffer + count.
   - Clean under `windows-msvc-asan`.

## RHI seam additions (summary)
- **None new** — reuses AR's compute + storage + `BufferUsage::Indirect`, BM's `DrawIndexedMultiIndirect`/
  `BindPerDrawData`, BZ's bindless bind. The compute writing the indirect `drawCount` uses the existing
  storage-buffer + indirect path. New non-backend files (`engine/render/gpu_culled.h`,
  `shaders/gpudriven_cull.comp.hlsl`, `tests/gpu_culled_test.cpp`) add ZERO above-seam backend code symbols.
  Seam grep stays at baseline (2).

## Out of scope (YAGNI)
`vkCmdDrawIndexedIndirectCount` if the engine instead reads back the count to set drawCount (use whichever
the existing indirect path supports — document; the GPU-written-count-via-indirect-count is the ideal but a
readback is acceptable for the MVP), Hi-Z occlusion culling, two-phase culling, per-cascade shadow culling,
LOD selection in the compute pass, Metal ICB. One compute frustum-cull + compact feeding one MDI+bindless
draw, byte-identical to the CPU-culled reference, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 51) + new `gpu_culled_test` (cull+compact vs
   brute-force frustum reference, per-draw carry, determinism). Clean under `windows-msvc-asan`.
2. **GPU-culled==CPU-culled proof + count proof:** `--gpucull-draw-shot` on Vulkan culls+draws via compute
   → MDI+bindless; the captured image is BYTE-IDENTICAL (SHA) to the internal CPU-frustum-culled per-object
   BOUND reference; the GPU `drawn` count == the CPU `frustum.h` `cpuRef`; prints `gpucull-draw: {total:N,
   drawn:K, cpuRef:K, drawCalls:1, textureBinds:1}`. Two runs identical.
3. `--gpucull-draw-shot` visual review (controller): only the in-frustum objects render (off-screen ones
   absent), lit + shadowed, coherent. Run under the AT Vulkan-validation gate → ZERO errors (compute + MDI
   + descriptor-indexing + indirect all validate together).
4. Metal: `visual_test --gpucull-draw` → new golden `tests/golden/metal/gpucull_draw.png`; two runs DIFF
   0.0000.
5. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `gpucull_draw.png`
   added; the other 51 byte-identical.
6. Introspect JSON rebaked exactly `+gpu-driven-culling-draw` + `--gpucull-draw-shot`; introspect test
   updated; no other drift.
7. Seam grep clean (no new above-seam code symbols). `scripts/verify.ps1` updated to include the new
   `gpucull_draw` image golden in the Mac round-trip loop.
