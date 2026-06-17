# Slice DW — Visibility-Buffer Slice 1: Vis-Buffer Render + Bit-Exact ID Readback (BEACHHEAD) (Phase 6 #5) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. **The visibility-buffer
> beachhead** — Nanite's rendering architecture: rasterize `(clusterID, triangleID)` into a screen-size INTEGER
> buffer (decoupling geometry from shading; the deferred material resolve is the next slice DX). This is the FIRST
> additive RHI of the virtual-geometry flagship — a single `Format::R32_Uint` enum + its 2 backend mappings —
> strictly additive + defaulted-no-op so every existing pipeline/RT/golden stays byte-for-byte unchanged.
> Branch: slice-dw-visbuffer. See [[hazard-forge-visbuffer-roadmap]].

**Goal:** Add `Format::R32_Uint`, rasterize `(clusterID<<7 | triID)` into an R32_Uint render target via the existing
DS–DV cluster MDI/bound draw, read it back, and prove the IDs are bit-exact + self-consistent + cover exactly the
CPU cull's survivor set. The image golden is a CPU-colored `hashColor(clusterID)` viz (identical both backends by
construction). NO deferred resolve, NO lighting. Make-safe: the new format is selected by NO existing call site →
all 85 existing goldens unchanged; the only real backend edit is teaching Metal's readback bpp about R32_Uint.

## RHI delta (the ONLY rhi.h change of the arc — additive, defaulted-no-op)
- **`engine/rhi/rhi.h` (~:17-26):** add `R32_Uint,` to the `Format` enum (after `D32_Float`). Purely additive —
  every existing `switch (Format)` has a `default`, so no existing consumer changes behavior; existing
  pipelines/RTs/goldens are BYTE-UNCHANGED. (Mirrors the prior `RGBA16_Float` / `Indirect` additions.)
- **`engine/rhi_vulkan/vk_common.h` `ToVk` (~:16):** `case Format::R32_Uint: return VK_FORMAT_R32_UINT;`
- **`engine/rhi_metal/metal_common.h` `ToMetalPixelFormat` (~:83):** `case Format::R32_Uint: return
  MTLPixelFormatR32Uint;`
- **`engine/rhi_vulkan/vulkan_device.cpp` `BytesPerPixel` (~:1280):** explicit `R32_Uint → 4` (default already
  returns 4; add for clarity).
- **`engine/rhi_metal/metal_device.mm` (~:174/:187-194):** the ONE real edit — `ReadRenderTarget` hardcodes
  `RGBA16F ? 8 : BGRA8`; extend so `pixelFormat == MTLPixelFormatR32Uint → bpp 4`.
- **SEAM:** dispatch baseline stays **2** (rhi_factory untouched); ZERO `vk*`/`MTL*`/`mtl::` symbols cross the seam
  (the new format strings live only inside the backend `*_common.h`, exactly like every existing format). The new
  RHI is a documented additive defaulted-no-op. Report this as the justified exception to "rhi.h unchanged".
- **SEAM SUBTLETY (handle + document):** `VulkanRenderTarget`'s ctor (vulkan_render_target.cpp:84-139)
  unconditionally writes a float `SAMPLED_IMAGE`+`defaultSampler` descriptor for the color image. For the R32_Uint
  vis-buffer this is harmless IFF the set is never bound+sampled-as-float — and DW only `ReadRenderTarget`s the
  vis-buffer (never samples it). VERIFY validation is clean when the R32_Uint RT is CREATED; if it warns AT
  ALLOCATION, gate the descriptor write on a non-integer format (a 1-line guard, still additive). Document the
  result.

## Reuse map (file:line)
- **The cluster MDI draw (DT/DV)** — `samples/hello_triangle/main.cpp:12169` (`--cluster-cull-shot`): cull/compact
  → `DrawIndexedMultiIndirect` over `cluster_viz.vert.hlsl` reading `PerDraw[gl_DrawID]` (`:42`). DW renders the
  SAME survivor draw into the R32_Uint RT instead of the BGRA8 scene RT.
- **`engine/render/cluster_cull.h`** — `ClusterInstance` / `CullClusterInstances` (`:87`) → the CPU survivor set
  the coverage proof ties to; `MdiCommand.firstInstance` = the source cluster-instance ID (`:99`).
- **`engine/render/meshlet.h`** — `kMaxTrisPerCluster = 128` (`:37` → 7 bits triID); `hashColor` (`:79`).
- **`ReadRenderTarget`** (Vulkan vulkan_device.cpp:1296 / Metal metal_device.mm:187) — bit-preserving raw
  `vkCmdCopyImageToBuffer` / `copyFromTexture:toBuffer:` (added in DD). `CaptureNextFrame`/`GetCapturedPixels`
  (rhi.h:602-605) for the BGRA8 golden image.

## Design decisions (locked)

1. **`engine/render/visbuffer.h` (NEW, pure CPU, `hf::render::vg`; shared by the test + both showcases; 0
   above-seam backend symbols).**
   - `static constexpr uint32_t kTriIdBits = 7;` (`ceil(log2(kMaxTrisPerCluster))`; static_assert `(1<<kTriIdBits)
     >= kMaxTrisPerCluster`). `static constexpr uint32_t kVisBackground = 0xFFFFFFFFu;` (the clear/sentinel — must
     not collide with any valid packed ID; document).
   - `uint32_t PackVisId(uint32_t clusterID, uint32_t triID)` = `(clusterID << kTriIdBits) | (triID & ((1<<kTriIdBits)-1))`.
   - `void UnpackVisId(uint32_t v, uint32_t& clusterID, uint32_t& triID)`.
   - A CPU coverage reference helper (which survivor covers a given INTERIOR sample) used by Proof B3 — keep it
     simple (project the cluster centroid / a known interior point to screen; the proof samples those pixels).

2. **`shaders/visbuffer.frag.hlsl` (NEW) + `shaders/visbuffer.vert.hlsl` (NEW, thin `cluster_viz.vert` variant).**
   Vert: drop the Lambert/hash-color outputs; forward the flat `clusterID` (from `gl_DrawID`/`PerDraw`) as
   `nointerpolation uint`. Frag: output a single `uint` to `SV_Target0` = `(clusterID << 7) | (SV_PrimitiveID &
   0x7F)`. NO lighting/color. Pipeline: `colorFormat = Format::R32_Uint`, `depthTest = true`, `depthWrite = true`
   (so the front-most cluster/tri wins per pixel — the rasterizer's depth resolve), NO blend flags.
   - **triID = `SV_PrimitiveID`** — Vulkan DXC supports it (→ SPIR-V `PrimitiveId`, no toolchain change). It is the
     FIRST `SV_PrimitiveID` use in the repo, so on the Mac rig VERIFY the MSL generates (`[[primitive_id]]`). IF
     glslc/spirv-cross stumble: FALLBACK to the established DT/DV per-cluster BOUND path on Metal (host one
     `DrawIndexed` per cluster, push the clusterID; derive triID from `SV_PrimitiveID` if available, else pack
     clusterID-only in DW and defer triID to DX). Document which path each backend uses.

3. **Showcase `--visbuffer-shot <out>` (Vulkan, main.cpp) AND `--visbuffer` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm in the diff, the DS lesson).** Reuse a DT/DV-style clustered-sphere scene. Cull/compact
   the survivors → render them into the R32_Uint RT through `visbuffer.{vert,frag}` → `ReadRenderTarget` the
   integer RT into `std::vector<uint8_t>` (reinterpret as `uint32_t[w*h]`). Declare the empty shadow pass per the
   DQ lesson if any lit pass runs (the vis-buffer pass itself is unlit — likely none needed; confirm validation
   clean regardless). PROOFS (Proof B — fully bit-exact + vendor-robust; print, fail loudly):
   - **(B1) self-consistency + coverage completeness:** for every non-background texel, assert `clusterID <
     survivorCount` AND `triID < survivorCluster[clusterID].triCount` (valid + in-range pairing); AND the set of
     distinct clusterIDs present == EXACTLY the CPU `CullClusterInstances` on-screen survivor set (every survivor
     writes ≥1 texel, no culled cluster writes any). Print `visbuffer self-consistent + coverage == CPU survivors:
     <N> clusters EXACT`.
   - **(B2) determinism:** two GPU vis-buffer renders → `ReadRenderTarget` → `memcmp` BIT-IDENTICAL. Print
     `visbuffer determinism: two renders BYTE-IDENTICAL`.
   - **(B3) GPU==CPU coverage at INTERIOR pixels:** for a handful of deterministically-chosen pixels at cluster
     INTERIORS (project a known interior point per visible cluster, nudge inward), assert the GPU vis-buffer's
     `clusterID` there == the CPU-predicted covering cluster. (Interior-only — avoids the edge/depth-tie
     cross-vendor fragility while proving the ID CONTENT, not just self-consistency, is correct. A full-frame
     CPU-vs-GPU memcmp is NOT cross-vendor-feasible — do NOT attempt it.) Print `visbuffer GPU==CPU interior
     coverage: <K>/<K> pixels EXACT`.
   - **Image golden (CPU-colored):** color each read-back texel — background → the clear color, else
     `hashColor(clusterID)` — into a BGRA8 buffer, write `tests/golden/metal/visbuffer.png`. **Identical on both
     backends BY CONSTRUCTION** (same integer bits in → same RGB out → trivially DIFF 0.0000). Print `visbuffer:
     {clusterInstances:M, survivors:N, written:<texels>}`. Existing 85 image goldens UNTOUCHED. **GOLDEN
     DISCIPLINE: ONLY `tests/golden/metal/visbuffer.png` — no loose `tests/golden/visbuffer.png`.**

4. **Disabled-path / no-op.** The new `Format::R32_Uint` is selected by NO existing call site → every existing
   pipeline/RT/golden is byte-for-byte unchanged (the additive-enum guarantee) — VERIFY by the full existing golden
   suite staying unchanged (render-invariance gate). Optionally a `visbufferEnabled` showcase toggle proving the
   integer pass is purely additive within the showcase.

5. **Determinism / cross-backend.** The written value is a FLAT integer (no interpolation/FP/transcendental/blend
   in the value) → inherently bit-exact cross-vendor; the cull is ordered/deterministic → the IDs are too; the
   image golden is CPU-colored from read-back bits → identical bytes both backends. (Proof B never depends on
   matching the hardware rasterizer's edge rules — only on raw integer-readback fidelity + the deterministic ID
   packing + the already-golden CPU cull set.)

6. **Tests `tests/visbuffer_test.cpp` (pure CPU; `hf_add_pure_test`):**
   - `PackVisId`/`UnpackVisId` round-trip over the full valid range; `kTriIdBits >= 7`; packing injective over
     `[0,survivorCount) × [0,128)` (collision-free); `kVisBackground` never equals any valid packed ID.
   - The CPU coverage reference's survivor membership == `CullClusterInstances` (ties DW to the DT/DV contract).
   - Determinism. Clean under `windows-msvc-asan`.

7. **Introspect.** Add exactly `virtual-geometry-visbuffer` (features) + `--visbuffer-shot` (showcases).

## RHI seam additions (summary)
- **ADDITIVE: one `Format::R32_Uint` enum + its 2 backend pixel-format mappings + 2 readback-bpp cases** (the
  Metal `ReadRenderTarget` bpp is the only real backend edit). Defaulted-no-op — selected by no existing call site,
  so all existing goldens are byte-identical. NO new interface method/descriptor/Create*/flag. dispatch baseline
  stays 2; ZERO above-seam backend symbols (the format strings stay inside the backend `*_common.h`). Document the
  additive exception in the seam report.

## Out of scope (YAGNI — DX and beyond)
The deferred MATERIAL RESOLVE pass (DX — texel-fetch the vis-buffer, reconstruct + shade; with ID-provenance +
GPU==CPU resolve-math bit-exact proofs + an image golden, NOT a byte-identical-to-forward claim, which the scout
established is NOT cross-vendor-feasible). Software raster of sub-pixel clusters (DY), streaming (DZ), RG32_Uint /
64-bit visibility packing (R32_Uint's `clusterID<<7|triID` is ample at this scale). ONE vis-buffer render with a
bit-exact self-consistency + coverage + determinism + interior-pixel-GPU==CPU proof set and a CPU-colored golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 85) + new `visbuffer_test`. Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--visbuffer-shot` on Vulkan: the CPU-colored viz shows the survivor clusters as
   hash-colored patches (== the cluster-cull viz, since it's the same survivors hash-colored); `visbuffer
   self-consistent + coverage == CPU survivors: <N> EXACT` + `determinism BYTE-IDENTICAL` + `GPU==CPU interior
   coverage EXACT`; the `visbuffer: {...}` line deterministic. Run under the AT Vulkan-validation gate → ZERO
   errors (verify the R32_Uint RT creation is validation-clean per the seam subtlety).
3. Metal: `visual_test --visbuffer` → new golden `tests/golden/metal/visbuffer.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). Proofs B1-B3 also pass on Metal. **Confirm visual_test.mm in the diff. Document whether
   Metal used SV_PrimitiveID or the per-cluster-bound fallback.**
4. **Render-invariance (CRITICAL — the additive-RHI make-safe gate):** `git diff master --stat -- tests/golden/metal`
   shows ONLY `visbuffer.png` added; the other 85 byte-identical (the R32_Uint addition perturbs NO existing RT).
   `git diff master --stat -- tests/golden` = ONLY `visbuffer.png` (metal) + the 2-line introspect json — NO loose
   `tests/golden/visbuffer.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+virtual-geometry-visbuffer` + `--visbuffer-shot`; introspect test updated.
6. **Seam grep:** rhi.h has ONLY the additive `R32_Uint` enum value (no new method/symbol); rhi_factory dispatch
   baseline 2; ZERO above-seam `vk*`/`MTL*`/`mtl::` code symbols; the backend mappings live only in the backend
   `*_common.h` + the Metal bpp edit. `scripts/verify.ps1` updated to include the new `visbuffer` image golden in
   the Mac round-trip loop AND `--visbuffer-shot` in the `$vkShots` validation gate.
