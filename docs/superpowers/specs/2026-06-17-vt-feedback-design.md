# Slice VT1 — Runtime Virtual Texturing: PAGE-NEEDED FEEDBACK / MARKING (Beachhead, Phase 9 #1) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The BEACHHEAD of FLAGSHIP #4:
> Runtime Virtual Texturing (UE5's literal RVT) — a virtual texture far larger than VRAM, sampled through a
> virtual→physical page indirection driven by per-pixel FEEDBACK. This first slice builds the integer core: a
> UV→page-id quantizer + a compute pass that marks which VT pages a set of sample-requests need, into a feedback
> SSBO, proven GPU==CPU BIT-EXACT, with an integer page-id debug-viz golden that is cross-backend BIT-IDENTICAL.
> Pure integer — NO rendering, NO new RHI. The structural twin of VSM Slice VA (page-needed marking), applied to
> texture pages instead of shadow pages. Namespace `hf::render::vt`. Branch: `slice-vt1-vtfeedback`.
> Grounded by a read-only Plan scout @ master `ab53434` (2026-06-17). See [[hazard-forge-project]].

**Goal:** Add `engine/render/vt.h` (the pure-CPU integer VT page-table core: `VtTexture`, the `PageId`/`UnpackPageId`
bijection, `SelectMipLevel` integer threshold-ladder, the `VtPageId` UV→page quantizer, `MarkFeedbackPages` CPU
reference) + `shaders/vt_feedback.comp.hlsl` (one thread per sample-request, the quantizer copied VERBATIM, writes
`gFeedback[pageId]=1`) + a `--vt-feedback-shot` (Vulkan) / `--vt-feedback` (Metal) showcase that dispatches it over a
fixed set of (virtual UV, mip) requests, reads back the feedback set, proves it BIT-EXACT vs the CPU reference, and
bakes an integer page-grid debug-viz golden. Make-safe: a NEW header + NEW shader + NEW showcase + NEW golden;
nothing existing changes. The cross-backend bit-identity is guaranteed by the pure-integer page math (no GPU
transcendental) + the order-independent set write.

## Why this is the right beachhead (the VSM-VA analog)
VT is the SAME machinery as the shipped, golden-verified VSM arc (`engine/render/vsm.h`) applied to texture pages:
a virtual page table → a feedback/marking pass → physical tile allocation + indirection → a sampled lookup → per-page
caching. This slice is the marking pass — the direct analog of VSM Slice VA (`vsm.h:156` `MarkResidentPages` +
`shaders/vsm_mark.comp.hlsl`), which landed clean and is `vsm_pages.png` (cross-backend DIFF 0.0000). VT marking is
STRICTLY SIMPLER than VSM marking: UV-space quantization (`floor(uv * pagesPerSide(mip))`) has no clipmap snap / no
world-space projection. The page-id / feedback-set / threshold-ladder math is pure integer → cross-backend
bit-identical by construction, the strongest verification posture the engine has.

## The integer core (the cross-backend crux)
- **`PageId(mip, px, py)`** = a flat MIP-MAJOR index: `mipPageOffset(mip) + py * pagesPerSide(mip) + px`, where
  `pagesPerSide(m) = max(1, virtualPagesPerSideMip0 >> m)` and `mipPageOffset` is the host-precomputed prefix-sum of
  `pagesPerSide(k)²` for `k < mip`. Bijective over `[0, pageCount())`; `UnpackPageId` inverts it. (Copy the bijection
  shape from `vsm.h:68`/`:74`, "mip" replacing "clipmap level"; mip levels have DIFFERENT pagesPerSide so the offset
  is a prefix-sum, not `level*vpps²` — static_assert / runtime-assert the round-trip.)
- **`SelectMipLevel`** = the integer THRESHOLD-LADDER (copy `vsm.h:101` `SelectClipmapLevel`): host-precompute mip
  thresholds (uploaded as exact float32 bits), level = count of thresholds a texel-density value exceeds, clamped to
  `[0, mipLevels)`. NO `log2`, NO transcendental on the bit-exact path (the determinism crux, like DV squared-distance
  / VA threshold-ladder). For THIS beachhead the showcase supplies `mip` per request directly (the simplest
  deterministic form); `SelectMipLevel` is exercised + unit-tested as the general path so slice 4 can use it.
- **`VtPageId(float u, float v, int mip)`** = the UV→page quantizer: `int pps = pagesPerSide(mip);
  px = clamp((int)floor(u * pps), 0, pps-1); py = clamp((int)floor(v * pps), 0, pps-1); return PageId(mip,px,py)`.
  Pure subtract/multiply/floor — deterministic. (The `u*pps` is a host-or-GPU float multiply, but the result is
  immediately floored to an integer; with the request UVs host-supplied as exact float32 and `pps` a small power of
  two, `floor(u*pps)` is bit-identical CPU↔Vulkan↔Metal. Use `std::fma`-free plain multiply on BOTH sides — pin the
  identical op. Document this as the one float→int boundary, exactly like `ProjectToScreenVert`'s host-only round in
  swraster.h. RECOMMEND for total safety: the showcase host-snaps each request to its integer `(mip,px,py)` and the
  GPU consumes integer page coords directly — zero GPU float — mirroring the swraster.h "host-snapped ScreenVerts"
  pattern; pick this if any cross-backend `floor(u*pps)` drift appears.)
- **`MarkFeedbackPages(requests, vt, feedbackOut)`** = the CPU reference (copy `vsm.h:156`): zero `feedbackOut`
  (length `pageCount()`), for each request `feedbackOut[VtPageId(req)] = 1`. Order-independent integer set —
  commutative, so GPU thread-race CANNOT change the result. `feedbackEnabled=false` → empty set.

## Reuse map (file:line)
- **VSM structural twin (copy + rename shadow→texture):** `engine/render/vsm.h` — `PageId`/`UnpackPageId` (:68/:74),
  `BuildLevelThresholds`/`SelectClipmapLevel` (:88/:101), `MarkPage`/`MarkResidentPages` (:120/:156). `vsm.h` is
  pure-CPU header-only, 0 backend symbols — `vt.h` mirrors that shape exactly.
- **The compute shader template:** `shaders/vsm_mark.comp.hlsl` — one thread per receiver, the quantizer copied
  VERBATIM from the header, 3 SSBOs (`gRequests`/`gFeedback`/`gParams`, the `:36-38` binding layout),
  `gResident[pageId]=1u` order-independent set write (:104), `markingEnabled` push flag. `vt_feedback.comp.hlsl` is
  this with UV-quantization instead of world-projection. Plain integer SSBO write → default MSL gen (no MSL-2.2
  unless the Mac says otherwise — the DW lesson).
- **The compute + readback surface (NO new RHI):** `BufferUsage::Storage` (`rhi.h:166`),
  `BindComputePipeline`/`BindStorageBuffer`/`DispatchCompute` (`rhi.h:412-426`),
  `ComputePipelineDesc{storageBufferCount, pushConstantSize, threadsPerGroupX}` (`rhi.h:168-185`), `ReadBuffer`
  (`rhi.h:616`). The order-independent set write needs NO atomics (same as `vsm_mark.comp.hlsl:104`).
- **The integer-set debug-viz golden:** the `vsm_pages.png` template (`2026-06-17-vsm-marking-design.md:79`) —
  CPU-color the read-back integer set, `hashColor(pageId)` for resident, dark for non-resident, laid out as the page
  grid. `meshlet.h:79` `hashColor`.
- **Showcase + registration patterns:** `--vsm-mark-shot` (Vulkan, main.cpp) + `--vsm-mark` (Metal, visual_test.mm)
  + introspect `virtual-shadow-maps-marking`/`--vsm-mark-shot` (`introspect.cpp`) + `verify.ps1` `$Goldens`/`$vkShots`
  — copy each registration shape for `vt_feedback`/`--vt-feedback`/`--vt-feedback-shot`.
- **Procedural / no-disk discipline:** `engine/render/streaming.h:15-17` ("load = synchronous procedural
  construction, NOT async disk I/O; async file streaming is a future slice") — VT pages are procedurally generated
  in-memory (later slices), NEVER streamed from disk; this beachhead has no page content at all (marking only).

## Design decisions (locked)

1. **`engine/render/vt.h` (NEW, namespace `hf::render::vt`, pure CPU, header-only, 0 above-seam backend symbols,
   mirrors `vsm.h`).** `struct VtTexture { int mipLevels; int pageSize; int virtualPagesPerSideMip0; }` +
   `pagesPerSide(mip)`, `mipPageOffset(mip)` (host prefix-sum), `pageCount()`. `PageId`/`UnpackPageId` (bijection),
   `BuildMipThresholds`/`SelectMipLevel` (integer ladder), `VtPageId` (UV→page), `struct SampleRequest { float u,v;
   int mip; }` (or the host-snapped integer `(mip,px,py)` form — see the crux note), `MarkFeedbackPages` (CPU ref).
   static_assert/assert the page budget fits the feedback SSBO + the page-id field. NO float on the bit-exact output
   path beyond the single documented `floor(u*pps)` (or eliminate it via host-snapping).

2. **`shaders/vt_feedback.comp.hlsl` (NEW).** ONE thread per sample-request (`t < requestCount`). Reads its
   `SampleRequest` (UV+mip, or the host-snapped integer page coords), runs `VtPageId` (or uses the snapped coords)
   copied VERBATIM from `vt.h`, writes `gFeedback[pageId] = 1u` (order-independent set; NO atomics needed). A
   `feedbackEnabled=0` push-constant flag → write nothing (the disabled-path). SSBOs `gRequests`(b0)/`gFeedback`(b1)/
   `gParams`(b2) per the `vsm_mark.comp.hlsl:36-38` layout. `ComputePipelineDesc{ storageBufferCount=3,
   threadsPerGroupX=64 }`. NO new RHI. Only `[[vk::binding]]` + `HF_MSL_GEN` above-seam. Add the `hf_gen_msl` line for
   Metal + the Vulkan compile entry; add `--msl-version 20200` ONLY if the Mac MSL-gen requires it (default: no).

3. **Showcase `--vt-feedback-shot <out>` (Vulkan, main.cpp) AND `--vt-feedback` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm in the diff + `#include "render/vt.h"`).** A fixed `VtTexture` (e.g. mipLevels=4,
   pageSize=128, virtualPagesPerSideMip0=16 → a 2048²-mip0 virtual texture, ~341 pages over 4 mips) + a fixed,
   deterministic set of (UV, mip) sample-requests (e.g. a grid of UVs sampled at a couple of mips — chosen so a known
   subset of pages is marked, NON-trivial coverage). Upload `gRequests`, clear `gFeedback` to 0, dispatch
   `vt_feedback.comp`, `ReadBuffer` `gFeedback`. CPU-run `MarkFeedbackPages` over the SAME requests → the reference
   set. Golden = the integer feedback set CPU-colored as a per-mip page-grid debug-viz (resident → `hashColor(pageId)`,
   non-resident → dark; mips laid out side-by-side or stacked, like the VSM concentric-clipmap viz) →
   `tests/golden/metal/vt_feedback.png`. CPU-colored from the read-back integer set → **identical both backends by
   construction** → DIFF 0.0000 (gate on compare.sh EXIT CODE).

4. **PROOFS (fail loudly; exact print lines):**
   - **(1) GPU==CPU bit-exact (make-or-break):** `memcmp(gpuFeedback, cpuFeedback) == 0` (integer set, NO FP
     tolerance). Print `vt-feedback GPU==CPU page set: <N> pages BIT-EXACT`.
   - **(2) disabled-path no-op:** `feedbackEnabled=false` (or requestCount=0) → `gFeedback` stays all-zero
     (byte-identical to the cleared upload). Print `vt-feedback disabled: empty set (no-op)`.
   - **(3) determinism:** two dispatches → `ReadBuffer` → byte-identical. Print `vt-feedback determinism: two
     dispatches BYTE-IDENTICAL`.
   - **(4) {pages}:** print `vt-feedback: {vt:4mip/16vpps0, requests:<R>, resident:<N>/<pageCount>}`.
   - **Golden discipline: ONLY `tests/golden/metal/vt_feedback.png`; do NOT commit it — the CONTROLLER bakes it on
     the Mac. No loose `tests/golden/vt_feedback.png`.** Existing 94 image goldens UNTOUCHED.

5. **Determinism / cross-backend.** The page math is pure integer (the one `floor(u*pps)` boundary pinned identically
   both sides, or eliminated by host-snapping); the set write is order-independent; the golden is CPU-colored from the
   integer read-back → bit-identical across Vulkan/Metal AND the GPU==CPU memcmp holds full-set. Run under the Vulkan
   sync-validation gate → the clear→dispatch→readback barriers SYNC-HAZARD-free.

6. **Tests `tests/vt_test.cpp` (pure CPU, NEW):** `PageId`/`UnpackPageId` bijection over `[0, pageCount())`;
   `pagesPerSide`/`mipPageOffset` monotonic + prefix-sum correct; `SelectMipLevel` boundary cases (below/above every
   threshold, clamp); `VtPageId` corners (u,v at 0, just-below-1, mid → expected page; clamp at the boundary);
   `MarkFeedbackPages` known request-set → known page set + `feedbackEnabled=false` → empty; determinism. Clean under
   `windows-msvc-asan`.

7. **Introspect.** Add exactly `runtime-virtual-texturing-feedback` (features) + `--vt-feedback-shot` (showcases).

## RHI seam additions (summary)
- **None.** `BufferUsage::Storage` + `BindComputePipeline`/`BindStorageBuffer`/`DispatchCompute`/`ReadBuffer` — all
  pre-existing (the VSM-marking / froxel-inject precedent). New non-backend code (`vt.h`, `vt_feedback.comp.hlsl`, the
  showcase, the test) adds ZERO above-seam backend symbols. `engine/rhi/rhi.h` + `rhi_factory` (dispatch baseline 2) +
  the backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — VT2 and beyond)
Physical tile-pool allocation + indirection table (VT2 = the `AllocatePhysicalPages` analog), procedural page
generation into the atlas (VT3), the material-pass VT indirection SAMPLE (VT4 = the first float texel read), per-page
caching (VT5 = the `VsmPageCache` analog), feedback-driven streaming budget / mip refinement (VT6). Real disk
streaming, async upload, HW sparse residency, texture compression/transcoding, runtime mip generation — ALL deferred
(pages are procedurally generated in-memory in later slices, never streamed). ONE compute marking pass with a GPU==CPU
bit-exact feedback set + disabled no-op + determinism and the integer page-grid golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 89) + the new `vt_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--vt-feedback-shot` on Vulkan: a coherent integer page-grid debug-viz (resident pages
   hash-colored across the mip pyramid); `vt-feedback GPU==CPU page set: <N> pages BIT-EXACT` + `disabled: empty set`
   + `determinism: two dispatches BYTE-IDENTICAL` + the `{...}` line. Run under the Vulkan-validation gate → ZERO
   errors (the clear→dispatch→readback SYNC-HAZARD-free).
3. Metal: `visual_test --vt-feedback` → new golden `tests/golden/metal/vt_feedback.png`; two runs DIFF 0.0000 (gate
   on compare.sh EXIT CODE). The GPU==CPU + determinism proofs also pass on Metal (integer math). **Confirm
   visual_test.mm in the diff; confirm vt_feedback.comp MSL-generates + the integer SSBO write lowers on Metal (should
   need no MSL-2.2; if it fails, report).** This is an INTEGER golden → a strict cross-backend pixel compare must show
   ZERO differing pixels (unlike the float-resolve goldens).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `vt_feedback.png` added; the other
   94 byte-identical. `git diff master --stat -- tests/golden` = ONLY `vt_feedback.png` (metal) + the 2-line
   introspect json — NO loose `tests/golden/vt_feedback.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+runtime-virtual-texturing-feedback` + `--vt-feedback-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new `vt_feedback`
   image golden in the Mac round-trip loop AND `--vt-feedback-shot` in the `$vkShots` validation gate.
