# Slice DH — DDGI Beachhead: Probe-Grid Ray-Trace Compute (Phase 5 #1) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. **The first slice of the
> GLOBAL ILLUMINATION flagship arc** (DDGI — dynamic diffuse GI via irradiance probes; the path to literal-UE5
> parity, see the gi-roadmap). A grid of irradiance probes each trace N rays against the scene depth — pure
> compute, NO new RHI, deterministic, with a bit-exact GPU==CPU data proof + a probeCount=0 byte-identical
> no-op proof.

**Goal:** Establish the probe grid + the per-probe ray-trace compute pass — the data foundation the rest of
the DDGI arc (capture → SH-encode → update → composite) builds on. From a world-space lattice of probes, each
probe traces `kRaysPerProbe` deterministic Fibonacci-sphere rays against the scene's view-space depth field
and records the hit position + distance per ray into a flat SSBO. A `--probegi-shot` showcase reads the
ray-hit SSBO back and proves it BIT-EXACT against a CPU reference (the same depth field, the same shared
march), proves the `probeCount=0` dispatch-0 no-op, and writes a deterministic debug-viz golden. NO visible GI
yet (that arrives in the composite slice) — this slice is the verified ray-trace data layer.

## Why NO new RHI (the key architectural decision)

The probe rays march the scene depth, but the compute shader does NOT sample a depth texture (the engine's
only sampled-texture-in-compute path is the special-cased `BindShadowMapCompute`). Instead — mirroring
`froxel_inject.comp.hlsl`, which reconstructs its data analytically and never samples a depth texture in
compute — the depth field is supplied as a **flat `float` SSBO** (`float[w*h]` of view-linear depth) that the
showcase uploads from the rendered G-buffer (read back via the existing `ReadRenderTarget`). The probe trace
indexes that SSBO exactly as the CPU reference march indexes its procedural field. This keeps the slice a pure
compute + storage-buffer pass with **ZERO new RHI** (reuses `ComputePipelineDesc` + `BindStorageBuffer` +
`DispatchCompute` + `ReadBuffer` — the froxel_inject / cluster_assign / gpu-cull surface), AND makes the GPU
pass and the CPU reference march byte-identical data → the bit-exact data proof. Do NOT add a
`BindGBufferDepthCompute`; if a later DDGI slice must sample live scene/depth as a texture in compute, that
additive seam (mirroring `BindShadowMapCompute`) belongs there, not here.

## Design decisions (locked)

1. **Probe GI math (engine/render/probe_gi.h, header-only pure CPU, no backend symbols).** Namespace
   `hf::render::probegi`. Model on `froxel.h` (grid + flat std430 SSBO record) + `ssr.h`/`ssgi.h` (view-pos
   reconstruction + depth march) + `gtao.h` (templated `DepthFn` + disabled-path identity). `#include
   "render/ssr.h"` and reuse `ssr::ReconstructViewPos` / `ssr::ViewToScreenUV` (as `ssgi.h` does) so the probe
   march and the shader share the SAME projection functions.
   - `struct ProbeGrid { math::Vec3 origin; int dimX=8, dimY=4, dimZ=8; float spacing=1.0f; }` (default 256
     probes, WORLD-space lattice). `int probeCount() const { return dimX*dimY*dimZ; }`,
     `int flatIndex(int px,int py,int pz) const { return px + py*dimX + pz*dimX*dimY; }` (cx-major, same as
     froxel/cluster), `math::Vec3 probePos(int px,int py,int pz) const` = `origin + (px,py,pz)*spacing`.
   - `inline constexpr int kRaysPerProbe = 16;` (matches `SsgiParams::K`). `inline constexpr float kGoldenAngle
     = 2.39996322972865332f;` `math::Vec3 FibonacciSphere(int i, int N)` — `z = 1-(2i+1)/N`, `r =
     sqrt(max(0,1-z*z))`, `phi = i*kGoldenAngle`, `dir = (r*cos phi, r*sin phi, z)`. Deterministic, unit-length,
     FULL sphere; `FibonacciSphere(0,1) == (0,0,1)`. Clamp i/N to valid range.
   - `inline constexpr float kRayMiss = 1e30f;` `struct ProbeRayHit { float hitPosDist[4]; }` (xyz = world hit
     position, w = hit distance or `kRayMiss`); `sizeof == 16` (std430). `struct ProbeRayHits { ProbeRayHit
     rays[kRaysPerProbe]; }` (`sizeof == 256`). The SSBO is a flat `ProbeRayHit[probeCount*kRaysPerProbe]`,
     probe p ray r at `p*kRaysPerProbe + r` (froxel block layout). `static_assert` both sizes.
   - `int GetProbeGridIndex(const math::Vec3& worldPos, const ProbeGrid& grid)` — nearest cell per axis via
     `lround((v-origin)/spacing)` clamped to `[0,dim-1]` → `flatIndex`; round-trips `probePos` at lattice
     points (unit-tested). Guards `spacing<=0`.
   - `template <typename DepthFn> bool TraceRayToDepth(const math::Vec3& originWorld, const math::Vec3&
     dirWorld, float maxDist, int steps, float thickness, const math::Mat4& view, float tanHalfFovY, float
     aspect, float yFlip, DepthFn sampleDepth, ProbeRayHit& outHit)` — march `t = (maxDist/steps)*k` for
     `k=1..steps`: `pWorld = originWorld + dirWorld*t`; `pView = view*pWorld`; `uvd = ssr::ViewToScreenUV(pView,
     ...)`; if `uvd.xy` outside `[0,1]` → continue (off-screen, no occluder info); `surf = sampleDepth(uvd.x,
     uvd.y)`; HIT iff `uvd.z` is within `[surf-thickness, surf+thickness]` (the front-to-back crossing, same
     thickness band as SSR) → `outHit = {pWorld, t}`, return true. No hit after the loop → `outHit.w = kRayMiss`,
     return false. Fixed steps, no RNG, bit-identical cross-backend. Document the convention. The CPU test feeds
     a procedural `DepthFn`; the shader feeds the `gDepth` SSBO sampler — the SAME march.
   - **Disabled path:** `inline constexpr int kProbeThreads = 64;` `int ProbeDispatchGroups(const ProbeGrid&
     grid)` = `probeCount<=0 ? 0 : (probeCount + 63)/64`. `dimX/dimY/dimZ == 0` → `probeCount == 0` → 0 groups
     → `DispatchCompute(0)` → the ray-hit SSBO is untouched (== the cleared upload). This is the byte-identical
     no-op the proof rests on.

2. **Compute shader `shaders/probe_raytrace.comp.hlsl`.** Model on `cluster_assign.comp.hlsl` (one thread per
   grid cell, flat-index decode, grid guard) + the SSR/SSGI view-space march. `[numthreads(64,1,1)]`, one
   thread per probe. Bindings (NO sampled textures — same style as `froxel_inject.comp.hlsl`): `[[vk::binding
   (0,0)]] RWStructuredBuffer<ProbeRayHit> gRayHits : register(u0)` (write), `[[vk::binding(1,0)]]
   RWStructuredBuffer<ProbeParams> gParams : register(u1)` (grid origin/dims/spacing + camera view/tanHalfFovY/
   aspect/yFlip + march maxDist/steps/thickness + depthW/H), `[[vk::binding(2,0)]] RWStructuredBuffer<float>
   gDepth : register(u2)` (flat view-linear depth `float[depthW*depthH]`, indexed `gDepth[(uint)(v*H)*W +
   (uint)(u*W)]` nearest). `ComputePipelineDesc{ storageBufferCount = 3, threadsPerGroupX = 64 }`. `main`:
   guard `p >= probeCount → return`; decode `(px,py,pz)` from `p`; `probeWorld = origin + (px,py,pz)*spacing`;
   `for r in 0..15`: `dir = FibonacciSphere(r,16)`; `TraceRayToDepth(probeWorld, dir, ...)` → `gRayHits[p*16+r]`.
   Copy `FibonacciSphere` + `ReconstructViewPos`/`ViewToScreenUV` + `TraceRayToDepth` VERBATIM from the headers.
   Dispatch `ProbeDispatchGroups(grid)` groups. Only `[[vk::binding]]` + `HF_MSL_GEN` above-seam. NO new RHI.

3. **Showcase `--probegi-shot <out>` (Vulkan) / `--probegi` (Metal).** Model on the `--froxelfog-shot` block
   + the gpu-cull `ReadBuffer` proof. Pipeline: render the deterministic scene → HDR `rt` + G-buffer `gbuf`
   (the existing gbuffer pass); `ReadRenderTarget(*gbuf)` → extract the `.w` view-linear-depth channel into
   `std::vector<float> depthField[w*h]` → upload as the `gDepth` SSBO; run the probe-raytrace compute;
   `ReadBuffer(*rayHitsBuf)` the `ProbeRayHit[probeCount*16]` SSBO back. THREE PROOFS:
   - **(1) GPU==CPU bit-exact (the correctness gate):** on the CPU, for every probe p + ray r, run
     `probegi::TraceRayToDepth` with a `DepthFn` sampling the SAME `depthField` (same nearest index math);
     assert `memcmp(gpuSSBO, cpuResult) == 0` — BIT-EXACT (the probe analog of gpu-cull `gpu==cpuRef`). Print
     `probe-gi GPU==CPU ray-hits: BIT-EXACT`.
   - **(2) probeCount=0 no-op:** re-run with `grid.dimX = 0` → `ProbeDispatchGroups == 0` → `DispatchCompute(0)`;
     read the SSBO; assert byte-identical to the cleared upload (all `kRayMiss`). Print `probe-gi probeCount=0:
     SSBO UNTOUCHED == cleared`.
   - **(3) determinism:** run the GPU trace twice; assert the two SSBO readbacks are byte-identical.
   - Print `probe-gi: {probes:256, rays:16, hits:H}` (H = total ray-hits with `w != kRayMiss`, deterministic).
   - **Golden:** a thin debug-viz — a fullscreen/point pass rendering the ray-hits as colored points (hit
     world-pos → color, misses skipped) composited over the lit scene, reusing the existing `pointList`
     pipeline flag (no new RHI) → deterministic `tests/golden/metal/probegi.png` (Metal two runs DIFF 0.0000,
     gate on compare.sh EXIT CODE). The image is the cross-platform witness; correctness rests on Proof 1.
   Existing 73 image goldens UNTOUCHED.

4. **Determinism.** Fixed scene/grid/camera/march params, deterministic Fibonacci dirs (no RNG/time), fixed
   steps. Two runs byte-identical; GPU==CPU bit-exact; cross-backend DIFF 0.0000.

5. **Tests `tests/probe_gi_test.cpp` (pure CPU, no GPU):**
   - **FibonacciSphere:** unit-length for all i; deterministic; `(0,1)→(0,0,1)`; even distribution (`|mean over
     i| → 0`, z's monotone + evenly spaced ≈ 2/N, no duplicate dirs).
   - **TraceRayToDepth — flat field:** a constant-depth field → a ray aimed away MISSES (`w==kRayMiss`); a ray
     aimed at the plane HITS at the analytic `hitDist`.
   - **TraceRayToDepth — synthetic occluder:** a near-occluder block in the depth field → a ray through the
     occluder column hits at the OCCLUDER's distance (not the far plane).
   - **GetProbeGridIndex:** round-trips `probePos(px,py,pz) → flatIndex(px,py,pz)`; far-outside positions clamp
     to the boundary probe.
   - **Disabled path:** `dimX=0` / `dimY=0` / `dimZ=0` each → `probeCount()==0` → `ProbeDispatchGroups==0`.
   - **Determinism:** the full per-probe ray-set over a fixed field is bit-identical across two runs.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `ddgi-probe-raytrace` (features) + `--probegi-shot` (showcases).

## RHI seam additions (summary)
- **NONE (verified).** Reuses `ComputePipelineDesc` + `BindComputePipeline`/`BindStorageBuffer`×3/
  `DispatchCompute` + `ReadBuffer` + `ReadRenderTarget` (the froxel_inject / cluster_assign / gpu-cull
  surface). The depth field is a flat SSBO (NOT a texture-in-compute) → no `BindGBufferDepthCompute`, no
  `sampledShadowMap`. New non-backend files (`engine/render/probe_gi.h`, `shaders/probe_raytrace.comp.hlsl`,
  `tests/probe_gi_test.cpp`) add ZERO above-seam backend code symbols. Seam grep stays at baseline (2). Report
  the seam result + confirm no rhi.h change.

## Out of scope (YAGNI)
Probe radiance capture / SH encoding / probe relighting / the GI composite (those are DDGI slices 2–5 — this
beachhead is ONLY the ray-trace data layer + its proofs). Visible indirect lighting (none yet). Per-frame
re-trace, probe relocation, ray-guiding, irradiance storage. One deterministic probe-grid ray-trace compute
pass with a bit-exact GPU==CPU data proof + a probeCount=0 dispatch-0 no-op proof + a debug-viz golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 73) + new `probe_gi_test` (Fibonacci determinism/unit/
   distribution, TraceRayToDepth flat + occluder, GetProbeGridIndex round-trip, disabled dimX/Y/Z=0,
   determinism). Clean under `windows-msvc-asan`.
2. **GPU==CPU + no-op proofs + visual:** `--probegi-shot` on Vulkan: `probe-gi GPU==CPU ray-hits: BIT-EXACT`
   + `probe-gi probeCount=0: SSBO UNTOUCHED == cleared` + two-run byte-identical; the `probe-gi: {probes:256,
   rays:16, hits:H}` line is deterministic; the debug-viz shows ray-hit points over the scene (coherent). Run
   under the AT Vulkan-validation gate → ZERO errors (the compute dispatch + SSBO read/write barriers
   SYNC-HAZARD-free).
3. Metal: `visual_test --probegi` → new golden `tests/golden/metal/probegi.png`; two runs DIFF 0.0000 (gate on
   the compare.sh EXIT CODE). The GPU==CPU + no-op proofs also pass on Metal.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `probegi.png` added; the
   other 73 byte-identical.
5. Introspect JSON rebaked exactly `+ddgi-probe-raytrace` + `--probegi-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (no new above-seam code symbols; confirm rhi.h unchanged — NO new RHI). `scripts/verify.ps1`
   updated to include the new `probegi` image golden in the Mac round-trip loop (gate on compare.sh EXIT CODE).
