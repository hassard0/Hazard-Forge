# Slice BM — GPU Multi-Draw-Indirect Batching (Phase 4 #14) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. GPU-driven rendering;
> builds on AR (single indirect draw). Render-invariant by construction (MDI image == per-draw image).

**Goal:** Render a many-distinct-object scene with ONE multi-draw-indirect call instead of N per-object
draws. On Vulkan: pack N `VkDrawIndexedIndirectCommand` into one buffer and issue ONE
`vkCmdDrawIndexedIndirect(drawCount=N)`; per-draw data (model matrix, material/mesh offsets) lives in a
storage buffer indexed by `gl_DrawID` (SPIR-V `DrawIndex`, `VK_KHR_shader_draw_parameters`). The result
must be BYTE-IDENTICAL to the same scene drawn per-object — that equality (plus the draw-call reduction
N→1) is the proof.

## Cross-backend framing (de-risked, AW-style asymmetry)

True MDI + `gl_DrawID`-indexed per-draw data is the Vulkan demonstration. Metal's equivalent is an
`MTLIndirectCommandBuffer` (ICB) — heavier to wire headless. To keep the slice tractable and the golden
robust, mirror the AW pattern:
- **Vulkan** does TRUE MDI and PROVES it: `--mdi-shot` (MDI path) is byte-identical to the same scene via
  the existing per-object path (`--mdi-shot-ref`, an internal reference) — SHA match. Report the draw-call
  count (1 MDI vs N).
- **Metal** renders the IDENTICAL scene for the golden via its working per-object path (the image is the
  same geometry/material, so `mdi.png` matches across backends). Optionally implement the Metal ICB path
  if straightforward, but it is NOT required — document the choice honestly (MDI is the Vulkan
  GPU-driven demonstration; the image is backend-identical).

This keeps the render-invariant contract airtight (the MDI image equals the per-draw image) without a
risky headless Metal-ICB dependency.

## Design decisions (locked)

1. **Per-draw data + MDI command build (engine side; pure-CPU builder + reuse AR's indirect buffer).**
   - A scene of N distinct objects (e.g. a deterministic grid of mixed cubes/spheres with per-object
     model matrices + material params — reuse the MT 144-draw scene or a similar fixed set).
   - Build a storage buffer `PerDraw { float4x4 model; float4 material; uint meshIndex; ... }[N]` (one
     entry per object) and an indirect buffer of N `VkDrawIndexedIndirectCommand`
     `{indexCount, instanceCount=1, firstIndex, vertexOffset, firstInstance=i}` (use `firstInstance`/
     `gl_DrawID` to index PerDraw — pick `gl_DrawID` (cleaner); document). A small pure-CPU builder
     (`engine/render/mdi.h`?) lays out the per-draw + command arrays deterministically; unit-test the
     layout (counts, per-command index/offset values, per-draw matrix packing).
   - RHI: reuse AR's `BufferUsage::Indirect` + storage buffers. Add `IRHICommandBuffer::
     DrawIndexedMultiIndirect(IBuffer& cmds, uint32_t drawCount, uint32_t stride)` (additive, default
     no-op; Vulkan → `vkCmdDrawIndexedIndirect(cmds, 0, drawCount, stride)`; Metal impl may no-op/fallback
     — document, since the Metal golden uses the per-object path). Enable
     `VK_KHR_shader_draw_parameters` (it's core in Vulkan 1.1+/1.3 via `shaderDrawParameters` feature —
     confirm it's enabled on the device; the engine targets 1.3).

2. **Shader.** A lit variant (`lit_mdi.vert.hlsl` or extend the instanced vert) that reads
   `PerDraw[gl_DrawID]` for the model matrix + material instead of a push constant. `[[vk::builtin(
   "DrawIndex")]] uint drawId` (HLSL→SPIR-V via DXC `-fvk-use-dx-layout`/the right builtin; the toolchain
   already does SPIR-V). The fragment stage is the existing lit/PBR. The per-object path uses the SAME
   PerDraw values via push constants so the two paths are mathematically identical → byte-identical image.

3. **Showcase `--mdi-shot <out>` (Vulkan) / `--mdi` (Metal).** Render the N-object scene via MDI (Vulkan)
   / per-object (Metal), lit + shadowed, fixed camera. Print `mdi: {objects:N, drawCalls:1, refDrawCalls:N}`
   (Vulkan) — and INTERNALLY render the per-object reference + assert `mdiImage == refImage` byte-identical
   (fail loudly on mismatch). New golden `tests/golden/metal/mdi.png` (Metal two runs DIFF 0.0000).
   Existing 38 image goldens UNTOUCHED.

4. **Tests `tests/mdi_test.cpp` (pure CPU, no GPU):**
   - **Command layout:** for N objects, the builder emits N commands with the expected
     `indexCount`/`firstIndex`/`vertexOffset`/`firstInstance` per object; `drawCount == N`.
   - **Per-draw packing:** `PerDraw[i]` holds object i's model matrix + material (round-trip a known
     object).
   - **Determinism:** two builds → bit-identical command + per-draw buffers.
   - Clean under `windows-msvc-asan`.

5. **Introspect.** Add exactly `gpu-multi-draw-indirect` (features) + `--mdi-shot` (showcases).

## RHI seam additions (summary)
- `IRHICommandBuffer::DrawIndexedMultiIndirect(IBuffer&, drawCount, stride)` — additive pure-interface,
  Vulkan impl inside `rhi_vulkan/` (`vkCmdDrawIndexedIndirect`), Metal impl no-op/fallback inside
  `rhi_metal/`. Reuses AR's `BufferUsage::Indirect` + storage buffers. New non-backend files
  (`engine/render/mdi.h`, `tests/mdi_test.cpp`, `shaders/lit_mdi.vert.hlsl`) add ZERO backend code symbols.
  Seam grep stays at baseline (the `vkCmd*`/builtin live inside the backend dir; comments tolerated).

## Out of scope (YAGNI)
GPU culling feeding the MDI count (AR's compute-cull could later write the draw buffer — a future combo),
`vkCmdDrawIndexedIndirectCount` (GPU-decided count), bindless textures, Metal ICB (optional, not required),
per-draw LOD selection, draw compaction. One fixed N-object scene, one MDI call, byte-identical to
per-draw, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 38) + new `mdi_test` (command layout, per-draw
   packing, determinism). Clean under `windows-msvc-asan`.
2. **MDI==per-draw proof:** `--mdi-shot` on Vulkan renders via ONE `vkCmdDrawIndexedIndirect(drawCount=N)`
   and the captured image is BYTE-IDENTICAL (SHA) to the internal per-object reference render of the same
   scene; the run asserts this + prints `mdi: {objects:N, drawCalls:1, refDrawCalls:N}`. Two runs
   identical.
3. `--mdi-shot` visual review (controller): the N-object scene renders correctly, lit + shadowed,
   coherent. Run under the AT Vulkan-validation gate → ZERO errors (MDI + shader-draw-parameters must be
   validation-clean).
4. Metal: `visual_test --mdi` → new golden `tests/golden/metal/mdi.png`; two runs DIFF 0.0000. (Document
   whether Metal used ICB or the per-object path; the image is backend-identical either way.)
5. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `mdi.png` added; the
   other 38 byte-identical.
6. Introspect JSON rebaked exactly `+gpu-multi-draw-indirect` + `--mdi-shot`; introspect test updated; no
   other drift.
7. Seam grep clean (no new above-seam code symbols). `scripts/verify.ps1` updated to include the new `mdi`
   image golden in the Mac round-trip loop.
