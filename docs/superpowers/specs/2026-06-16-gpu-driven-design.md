# Slice CB — Fully-GPU-Driven Pass (MDI + Bindless capstone) — Phase 4 #27 — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. Composes BM
> (multi-draw-indirect) + BZ (bindless textures) into THE modern GPU-driven renderer: an entire
> multi-material scene in ONE draw call + ONE texture bind. Render-invariant by construction.

**Goal:** Render N distinct-material objects with ONE `vkCmdDrawIndexedIndirect(drawCount=N)` AND ONE
bindless texture-array bind. The per-draw storage buffer (from BM) carries each object's model matrix +
material params + a TEXTURE INDEX (from BZ); the vertex shader reads `PerDraw[gl_DrawID].model`, the
fragment shader samples `gTextures[NonUniformResourceIndex(PerDraw[gl_DrawID].texIndex)]`. The result must
be BYTE-IDENTICAL to the same scene drawn per-object with per-material bound textures — that equality
(plus "1 draw call + 1 texture bind for the whole scene") is the capstone proof.

## Cross-backend framing (BM/BZ-style asymmetry, already proven)

- **Vulkan** does the TRUE fully-GPU-driven pass (MDI + descriptor-indexing bindless) and PROVES it:
  `--gpudriven-shot` (GPU-driven path) is byte-identical to the same scene via the existing per-object
  per-material BOUND path (`--gpudriven-shot-ref`, internal) — SHA match. Reports `drawCalls:1,
  textureBinds:1` vs `refDrawCalls:N, refTextureBinds:N+1`.
- **Metal** renders the IDENTICAL scene for the golden via its working per-object bound path (image
  backend-identical → `gpudriven.png` matches). Metal ICB + argument-buffer is OPTIONAL, NOT required —
  document honestly. (Both BM and BZ already established this asymmetry; this slice reuses it.)

## Design decisions (locked)

1. **Combined per-draw data (reuse BM `engine/render/mdi.h` + BZ `engine/render/bindless.h`).** Extend the
   BM `PerDraw` struct to also carry the BZ texture index: `PerDraw { float4x4 model; float4 material;
   uint texIndex; ...pad... }`. A pure-CPU builder lays out the N commands (BM) + the N PerDraw entries
   (model + material + bindless texIndex via `bindless::Intern`) deterministically. Unit-test the combined
   layout (the per-draw texIndex matches the bindless table's index for each object's texture; command
   count == N; deterministic).

2. **Shader `shaders/lit_gpudriven.{vert,frag}.hlsl`.** Vertex: reuse `lit_mdi.vert` (reads
   `PerDraw[gl_DrawID].model` via `[[vk::builtin("DrawIndex")]]`). Fragment: reuse the lit/PBR core but
   sample the base color from `gTextures[NonUniformResourceIndex(PerDraw[gl_DrawID].texIndex)]` (the BZ
   bindless array) instead of a bound material texture. The PER-OBJECT BOUND reference uses the EXISTING
   lit frag with the SAME texture bound + the same model push-constant → identical texels + transforms →
   byte-identical image. DXC must emit BOTH `DrawParameters` (gl_DrawID) AND `SPV_EXT_descriptor_indexing`
   (NonUniform) — confirm in the SPIR-V.

3. **Render path.** Bind the bindless array ONCE (`BindBindlessTextures`), bind the per-draw SSBO + the
   indirect-command buffer, issue ONE `DrawIndexedMultiIndirect(cmds, N)`. The whole N-object multi-material
   scene draws in one call. Reuse the BM combined cube+sphere geometry buffer + the BZ texture set. Fixed
   deterministic camera + the BZ multi-material grid (100 objects, 5 textures).

4. **Showcase `--gpudriven-shot <out>` (Vulkan) / `--gpudriven` (Metal).** Render the multi-material grid
   via the fully-GPU-driven pass (Vulkan) / per-object bound (Metal). Print `gpudriven: {objects:100,
   drawCalls:1, textureBinds:1, refDrawCalls:100, refTextureBinds:101}`. INTERNALLY render the per-object
   per-material BOUND reference + assert `gpuDrivenImage == refImage` byte-identical (fail loudly). New
   golden `tests/golden/metal/gpudriven.png` (Metal two runs DIFF 0.0000). Existing 49 image goldens
   UNTOUCHED.

5. **Tests `tests/gpu_driven_test.cpp` (pure CPU, no GPU):**
   - **Combined layout:** for N objects, N commands (BM layout) + N PerDraw entries each with the right
     model + material + `texIndex` (== `bindless::Intern` of that object's texture); `drawCount == N`.
   - **Index consistency:** distinct textures → distinct texIndex; same texture → same texIndex (dedup);
     every object's texIndex is in range of the bindless table.
   - **Determinism:** two builds → bit-identical command + per-draw + bindless table.
   - Clean under `windows-msvc-asan`.

## RHI seam additions (summary)
- **None new** — reuses BM's `DrawIndexedMultiIndirect` + `BindPerDrawData` + `BufferUsage::Indirect` and
  BZ's `CreateBindlessTextureSet` + `BindBindlessTextures`. The combined shader is new but pure HLSL. New
  non-backend files (`shaders/lit_gpudriven.*`, `tests/gpu_driven_test.cpp`, any builder additions to
  `mdi.h`) add ZERO above-seam backend code symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
GPU culling feeding the MDI count (AR's compute-cull writing the indirect buffer is the natural next combo
— note it), `vkCmdDrawIndexedIndirectCount` (GPU-decided count), per-draw LOD, Metal ICB+argument-buffer
(optional), bindless buffers, mesh shaders. One fully-GPU-driven multi-material pass (MDI + bindless),
byte-identical to per-object bound, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 49) + new `gpu_driven_test` (combined layout, index
   consistency, determinism). Clean under `windows-msvc-asan`.
2. **GPU-driven==bound proof:** `--gpudriven-shot` on Vulkan renders the N-object multi-material scene via
   ONE `DrawIndexedMultiIndirect` + ONE bindless bind and the captured image is BYTE-IDENTICAL (SHA) to the
   internal per-object per-material BOUND reference; the run asserts this + prints `gpudriven: {objects:100,
   drawCalls:1, textureBinds:1, refDrawCalls:100, refTextureBinds:101}`. Two runs identical.
3. `--gpudriven-shot` visual review (controller): the multi-material grid renders correctly, lit +
   shadowed, coherent. Run under the AT Vulkan-validation gate → ZERO errors (MDI + descriptor-indexing +
   gl_DrawID + NonUniform all validation-clean together).
4. Metal: `visual_test --gpudriven` → new golden `tests/golden/metal/gpudriven.png`; two runs DIFF 0.0000.
5. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `gpudriven.png` added;
   the other 49 byte-identical.
6. Introspect JSON rebaked exactly `+gpu-driven-rendering` + `--gpudriven-shot`; introspect test updated; no
   other drift.
7. Seam grep clean (no new above-seam code symbols). `scripts/verify.ps1` updated to include the new
   `gpudriven` image golden in the Mac round-trip loop.
