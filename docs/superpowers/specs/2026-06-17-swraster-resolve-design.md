# Slice SW4 — Software-Rasterized vis-buffer → Deferred-Material RESOLVE (Phase 8 #3) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The back-half of the Nanite
> software-raster arc: drive the EXISTING DX deferred-material resolve (`visresolve.frag`, DX) from the SOFTWARE
> visibility buffer (SW2's GPU compute SSBO / SW1's CPU `swraster.h`), completing a full
> software-raster → deferred-shade pipeline. A tiny fullscreen "unpack" blit makes the SW `depth|id` SSBO conform to
> the R32_Uint visId texture the resolve already reads → `visresolve.frag` is reused BYTE-FOR-BYTE. NO new RHI.
> Namespace `hf::render::vg`. Branch: `slice-sw4-swresolve`. See [[hazard-forge-swraster-roadmap]].
> Grounded by a read-only Plan scout @ master `32bff72` (2026-06-17).

**Goal:** Add `shaders/swraster_resolve_blit.frag.hlsl` (a fullscreen pass that unpacks the SW `depth|id` SSBO →
the R32_Uint visId render target the resolve consumes) + a `--swraster-resolve-shot` (Vulkan) / `--swraster-resolve`
(Metal) showcase that software-rasterizes a clustered mesh, blits its vis-buffer to R32_Uint, runs the UNCHANGED
`visresolve.frag` deferred resolve, and proves the output is a coherent Lambert-lit shaded image bit-exact between
GPU and CPU at interior pixels + deterministic + disabled-path all-sky. Make-safe: a NEW shader + NEW showcase + NEW
golden; `visresolve.frag` and everything else existing UNCHANGED.

## The crux — packing reconciliation (SW vis-buffer → resolve input)

**What the SW vis-buffer holds** (`engine/render/swraster.h:74` `PackSw`):
```
key   = (depthQ << 16) | (visId & 0xFFFF)            // depthQ HIGH 16, visId LOW 16
visId = PackVisId(clusterId, tri) = (clusterId<<7)|(tri & 0x7F)   // visbuffer.h:52, kTriIdBits=7
kSwClear = 0xFFFFFFFF                                 // swraster.h:65
```
**What `visresolve.frag` reads** (`shaders/visresolve.frag.hlsl:97-103`) — an R32_Uint texel, full 32 bits:
```
uint v = gVisBuffer.Load(...)
uint cid = v >> 7;  uint tid = v & 0x7F;              // (clusterID<<7)|triID, NO depth field
if (v == 0xFFFFFFFF || cid >= drawnClusterCount) -> sky
```
**The blit transform** (plain uint32, NO int64 — strips depth, keeps visId):
```
uint packed = gSwVis[y*W + x];                        // SSBO read (set 3, binding 13, via BindLightClusters)
uint out = (packed == 0xFFFFFFFFu) ? 0xFFFFFFFFu      // kSwClear  -> kVisBackground (identical sentinel)
                                   : (packed & 0xFFFFu); // discard depth, keep visId
return out;                                            // SV_Target0, R32_Uint
```
Correct because: (1) `kSwClear == kVisBackground == 0xFFFFFFFF` (`swraster.h:65`, `visbuffer.h:45`) → sentinel passes
through and the resolve's sky branch fires identically; (2) the SW `visId` low-16 bits ARE exactly
`(clusterID<<7)|triID` — both sides use the SAME `PackVisId` with the SAME `kTriIdBits=7` — and the 16-bit value
zero-extends into the resolve's 25-bit cluster field with no loss; (3) **depth is discardable** — the resolve has NO
depth sampler and runs `depthTest=false` (`main.cpp:15131`); occlusion was already resolved by the SW
`InterlockedMin`/serial-`min` over the `depth|id` key, so each blit texel already holds the winning fragment.
**Bit-budget:** the SW2 scene `SphereGeometry(24,16)` → ~769 tris → `ceil(769/128)` ≈ 6 clusters; the SW id field
covers clusters `[0, 1<<(16-7)) = [0,512)` — 16 bits comfortably suffice, visId never overflows into depth.

## Reuse map (file:line)
- **SW raster (input):** `engine/render/swraster.h` — `SwVisBuffer`/`Init` (:101), `RasterClusters` (:242),
  `ProjectToScreenVert` (:120), `PackSw`/`UnpackSw` (:74/:80), `kSwClear` (:65). GPU SSBO: `shaders/swraster.comp.hlsl`
  (`gVis` :65, `InterlockedMin(...key)` :152); host wiring `samples/hello_triangle/main.cpp:12980-13158`
  (`GpuTri`/`GpuScreenVert`/`gVis` upload + `runRaster`).
- **The resolve (reused UNCHANGED — the input contract the blit satisfies):** `shaders/visresolve.frag.hlsl`;
  CPU mirror `engine/render/visresolve.h` (`ResolvePixel` :146, `DefaultResolveMaterial` :59, `ResolveSkyColor` :73,
  `EncodeBGRA8` :167). Packing identity: `engine/render/visbuffer.h` `PackVisId`/`UnpackVisId` (:52/:58),
  `kTriIdBits=7` (:35), `kVisBackground` (:45).
- **Fullscreen-pass + R32_Uint RT + fragment-SSBO precedent:** vrPipeline `GraphicsPipelineDesc{fullscreen;
  usesFrameUniforms; usesTexture; usesLightClusters}` `main.cpp:15121-15136`; `fullscreen` flag `rhi.h:89`;
  `CreateRenderTarget(w,h,Format::R32_Uint)` `rhi.h:501` + live `main.cpp:15253`; `Format::R32_Uint` `rhi.h:26`,
  vk map `vk_common.h:24`, integer-color RT `vulkan_render_target.cpp:96`. **Fragment-stage SSBO is bound via
  `BindLightClusters`** (set 3, bindings 13/14/15) `rhi.h:148`(`usesLightClusters`)+`rhi.h:326`; froxel-apply
  precedent `main.cpp:22638` (`cmd.BindLightClusters(*volBuf,*dummyBuf,*dummyBuf)`); `ComputeToFragmentBarrier`
  `rhi.h:440` used `main.cpp:13146`; `ReadBuffer`/`ReadRenderTarget` `rhi.h:616`, `BindTexture` `rhi.h:274`.
- **Showcase patterns:** `--visresolve-shot` block `main.cpp:14971` + host wiring `:15200-15452` (ClusterMeta/index/
  vertex SSBOs, ResolveUbo, proofs, readback); `--swraster-gpu-shot` block `main.cpp:12980`; flag decl `:430-432`,
  parse `:1249`/`:1273`. Metal twins `metal_headless/visual_test.mm`: `--visresolve` :24005 + `renderResolve` :16700;
  `--swraster-gpu` :24031 + the SW2 CPU-path note :15463-15504.
- **Scene:** `scene::SphereGeometry`, `vg::BuildMeshlets` (`meshlet.h:109`/`:163`), `ms.indices`/`ms.meshlets`.
- **Registration:** `scripts/verify.ps1` `$Goldens` (:64), `$vkShots` (:315); introspect features
  `engine/editor/introspect.cpp:252-261` + showcase descriptions `:150-158`; `tests/introspect_test.cpp:407,632,705`.
  MSL gen: `CMakeLists.txt` `hf_gen_msl` def :48, `visresolve.frag ... --msl-version 20200` :182,
  `swraster.comp` (NO MSL, int64) :191-200.

## Design decisions (locked)

1. **Integration path = SSBO → R32_Uint blit, then reuse `visresolve.frag` UNCHANGED.** A fullscreen "unpack"
   fragment shader (the blit) converts the SW `depth|id` SSBO into the R32_Uint visId texture the resolve already
   reads. Rejected alternative: an SSBO-reading resolve variant — it forks a golden-locked shader and is more churn;
   the blit reuses `visresolve.frag` byte-for-byte. **RHI delta: ZERO.**

2. **`shaders/swraster_resolve_blit.frag.hlsl` (NEW, int32 only, ~30 lines).** Fullscreen pass (`fullscreen=true`,
   `post.vert`). Reads the SW vis SSBO via the `usesLightClusters` path at `[[vk::binding(13,3)]]
   RWStructuredBuffer<uint> gSwVis` (mirror the froxel-apply fragment-SSBO bind) + a tiny UBO `{uint width, height}`
   (the frame-uniform or a push/UBO consistent with the vrPipeline). Per pixel: compute `idx = y*width + x`, read
   `gSwVis[idx]`, apply transform B (`packed==kSwClear ? kVisBackground : packed & 0xFFFF`), output `SV_Target0` as
   the R32_Uint visId. **NO `--msl-version 20200`** (no int64, no integer `texture.read` — it WRITES an integer
   SV_Target, which is plain MSL). Add ONE `hf_gen_msl(... swraster_resolve_blit.frag ... frag
   swraster_resolve_blit_fragment)` line in `CMakeLists.txt`. `visresolve.frag` keeps its existing `20200` untouched.

3. **Showcase `--swraster-resolve-shot <out>` (Vulkan, main.cpp) AND `--swraster-resolve` (Metal, visual_test.mm —
   WIRE BOTH; confirm visual_test.mm in the diff + `#include render/swraster.h` + `render/visresolve.h`).** Scene =
   the SW2 scene: `SphereGeometry(24,16)` → `BuildMeshlets` → ~769 tris, a 512×512 buffer. **Vulkan path** (the blit +
   resolve are int32 → both MSL-gen): SW2 raster wiring (`main.cpp:12980-13158`) fills the SW vis SSBO via
   `swraster.comp` → `ComputeToFragmentBarrier` → (A) blit fullscreen into `CreateRenderTarget(w,h,R32_Uint)` →
   (B) `BindTexture(blitRT)` + `BindLightClusters(meta,idx,vtx)` + the resolve UBO → `visresolve.frag` → a `BGRA8`
   RT. **`ClusterMeta` from the SW meshlets:** clusterID IS the meshlet index `cl`, so
   `gClusterMeta[cl].triOffset = meshlets[cl].triOffset`, model = scene/identity model (simpler than DX's
   survivor-draw remap because SW packs the meshlet index directly, `swraster.h:255`). **Metal path** (SW convention,
   `swraster.comp` has no MSL): build the CPU `swraster.h::RasterClusters` vis-buffer over the SAME scene → upload it
   as the SSBO (or directly write the unpacked R32_Uint) → blit → the SAME `visresolve.frag` → identical image by
   construction.

4. **PROOFS (fail loudly; exact print lines):**
   - **(1) blit fidelity GPU==CPU, bit-exact full-frame:** read back the blit's R32_Uint RT; for every texel assert
     `blitRT[p] == ((swVis[p]==kSwClear) ? kVisBackground : (swVis[p] & 0xFFFF))`. Plain uint32 → DIFF 0 everywhere.
     Print `swraster-resolve blit GPU==CPU unpack: BIT-IDENTICAL (<W>x<H>)`.
   - **(2) SW-resolve == HW-resolve @interior (or principled diff):** at clean interior near-pole pixels (the
     `visbuffer.h:88` `InstanceInteriorSamples` oracle) the SW-min winner == the HW depth-test winner == same cluster,
     so the resolved RGB matches a HW `--visresolve-shot` render. Print `swraster-resolve SW==HW @interior: <k>/<n>
     EXACT`. Silhouette/edge pixels are a DOCUMENTED principled difference (SW integer top-left coverage vs HW
     rasterizer fill — `visresolve.h:23-27` caveat) and are excluded from the bit-exact set.
   - **(3) GPU==CPU resolve-math @interior, bit-exact:** reuse `vg::ResolvePixel` + `EncodeBGRA8` over the read-back
     visIds at interior samples; memcmp vs the GPU resolved BGRA (the `main.cpp:15378-15397` proof verbatim). Print
     `swraster-resolve GPU==CPU shade @interior: <k>/<n> EXACT`.
   - **(4) determinism, two full renders byte-identical** (visIds AND lit image), mirror `main.cpp:15399-15413`.
     Print `swraster-resolve determinism: two renders BYTE-IDENTICAL`.
   - **(5) disabled-path no-op:** dispatch 0 raster groups → SSBO stays `kSwClear` → blit writes all `kVisBackground`
     → resolve outputs uniform sky; assert every texel == `ResolveSkyColor()`-encoded. Print `swraster-resolve
     disabled-path: all-sky (no-op)`.
   - **(6) golden is a coherent SHADED mesh** (Lambert-lit faceted sphere via `visresolve.frag` to a `BGRA8_UNorm`
     RT — NOT a vis-id false-color hashColor): assert `shaded texels > 0` and not uniform. Print `swraster-resolve:
     {tris:<T>, survivors:<S>, shaded:<N>} (software-rasterized -> deferred-resolved lit image)`. On Metal append the
     CPU-swraster.h-reference note (like `visual_test.mm:15496`).
   - **Golden** = the resolved `BGRA8` RT → `tests/golden/metal/swraster_resolve.png`. CPU-path on Metal → identical
     both backends by construction → DIFF 0.0000 (gate on compare.sh EXIT CODE). Existing 93 image goldens UNTOUCHED.
     **GOLDEN DISCIPLINE: ONLY `tests/golden/metal/swraster_resolve.png`; do NOT commit it — the CONTROLLER bakes it
     on the Mac. No loose `tests/golden/swraster_resolve.png`.**

5. **Determinism / cross-backend.** Vulkan runs the GPU `swraster.comp` SSBO; Metal runs the CPU `swraster.h`
   reference (int64 → no MSL) — both feed the SAME int32 blit + the SAME `visresolve.frag`, so the resolved image is
   bit-identical across backends by construction. Run under the Vulkan sync-validation gate → the
   raster(compute)→barrier→blit(graphics)→barrier→resolve(graphics) chain SYNC-HAZARD-free.

6. **Introspect.** Add exactly `nanite-software-raster-resolve` (features) + `--swraster-resolve-shot` (showcases).

## RHI seam additions (summary)
- **None.** `Format::R32_Uint` (additive, pre-existing from DW), `CreateRenderTarget`, the `fullscreen`
  `GraphicsPipelineDesc`, fragment-stage SSBO via `BindLightClusters` (the froxel-apply precedent),
  `ComputeToFragmentBarrier`, `BindTexture`, `ReadRenderTarget`/`ReadBuffer` — all pre-existing. New non-backend code
  (`swraster_resolve_blit.frag.hlsl`, the showcase, registration) adds ZERO above-seam backend symbols.
  `engine/rhi/rhi.h` + `rhi_factory` (dispatch baseline 2) + the backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — SW3 and beyond)
The HW-large/SW-small hybrid dispatch (SW3), a combined raster+resolve compute kernel, ANY modification to
`visresolve.frag` (the blit makes the input conform), perspective-correct interpolated depth, 64-bit visibility,
persistent-thread binning, any SW==HW byte-exact claim at silhouettes. ONE blit shader + ONE showcase joining the
existing SW raster and the existing DX resolve via existing barriers/binds, with the 6 proofs + the shaded golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 89) + clean under `windows-msvc-asan`. (No new unit test
   strictly required — the blit transform is trivial uint32 covered by proof 1; OPTIONAL: a `swraster_test.cpp` case
   asserting `UnpackSw(PackSw(d,id)).visId == id` already exists. If a CPU blit helper is added to a header, unit-test
   it.)
2. **proofs + visual:** `--swraster-resolve-shot` on Vulkan: a coherent Lambert-lit software-rasterized sphere
   (deferred-resolved, not a false-color); proofs 1-6 all print PASS; run under the Vulkan-validation gate → ZERO
   errors (the compute→blit→resolve chain SYNC-HAZARD-free).
3. Metal: `visual_test --swraster-resolve` → new golden `tests/golden/metal/swraster_resolve.png`; two runs DIFF
   0.0000 (gate on compare.sh EXIT CODE). The GPU==CPU + determinism proofs also pass on Metal. **Confirm
   visual_test.mm in the diff; confirm `swraster_resolve_blit.frag` MSL-generates + compiles + the SSBO read +
   integer SV_Target lower on Metal (no MSL-2.2 — should be plain; if it fails, report).**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `swraster_resolve.png` added; the
   other 93 byte-identical. `git diff master --stat -- tests/golden` = ONLY `swraster_resolve.png` (metal) + the
   2-line introspect json — NO loose `tests/golden/swraster_resolve.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+nanite-software-raster-resolve` + `--swraster-resolve-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new
   `swraster_resolve` image golden in the Mac round-trip loop AND `--swraster-resolve-shot` in the `$vkShots`
   validation gate.
