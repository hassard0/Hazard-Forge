# Slice SW1 — Nanite Software-Raster Slice 1: CPU-Reference Rasterizer + Integer Edge Math (BEACHHEAD) (Phase 8 #1) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The BEACHHEAD of the NANITE
> SOFTWARE-RASTERIZATION flagship — a deterministic integer/fixed-point scan-converter that rasterizes cluster
> triangles into a packed `depth|id` visibility buffer (the same `(clusterID<<7|triID)` packing as DW's hardware
> vis-buffer), resolving the front-most surface per pixel by a `min` over a 32-bit `depth|id` key. This slice is
> PURE CPU (the reference rasterizer + the exact integer edge/fill-rule math), with a small CPU-colored golden;
> the GPU compute version (SW2) copies this math verbatim and proves bit-identical. NO GPU, NO new RHI — the
> tractable first slice. Namespace `hf::render::vg`. Branch: slice-sw1-swraster. See [[hazard-forge-swraster-roadmap]].

**Goal:** Add `engine/render/swraster.h` (the `SwVisBuffer` + the deterministic integer edge-function rasterizer
with top-left fill rule + 32-bit `depth|id` pack/min) + a `--swraster-shot` showcase that rasterizes a fixed set of
cluster triangles into a small vis-buffer and CPU-colors it. Make-safe by construction: a NEW header + NEW showcase
+ NEW golden; nothing existing changes. The rasterizer is integer + fixed-point → bit-identical on every platform.

## Reuse map (file:line — from the scout)
- **DW `engine/render/visbuffer.h`** — `kTriIdBits=7`, `PackVisId`/`UnpackVisId`, `kVisBackground` sentinel pattern;
  the flat-integer bit-exact discipline (visbuffer.h:9-12). The SW vis-buffer packs the SAME `visId =
  PackVisId(clusterID,triID)` in the low bits.
- **`engine/render/meshlet.h`** — `Meshlet`, `kMaxTrisPerCluster=128`, `hashColor` (meshlet.h:79) for the viz.
- **`engine/render/visresolve.h:120-139`** — how a cluster's triangle is reconstructed (clusterID→triOffset→the 3
  world verts via the index/vertex buffers); the SW raster needs the INVERSE: world triangle → screen coverage.
- **`engine/math` (`math.h:185-200` `MulPointDivide`, `:61` `Perspective`)** — the projection (Vulkan Y-flip baked
  in); the vis-buffer pixel convention (visbuffer.h:113-115).
- **DW `--visbuffer-shot` Proof + CPU-coloring golden** (main.cpp:12840) — the showcase + golden template.

## Design decisions (locked)

1. **`engine/render/swraster.h` (NEW, pure CPU, `hf::render::vg`; 0 above-seam backend symbols; mirrors visbuffer.h).**
   - `struct SwVisBuffer { uint32_t w, h; std::vector<uint32_t> packed; };` — `packed[y*w + x]`, initialized to
     `kSwClear`. `static constexpr uint32_t kSwClear = 0xFFFFFFFFu;` (max depth, no id → background).
   - **32-bit `depth|id` packing.** `static constexpr uint32_t kSwIdBits = 16;` `static constexpr uint32_t
     kSwDepthBits = 16;` (static_assert `kSwIdBits + kSwDepthBits == 32` and `kSwIdBits >= ` the bits needed for the
     max `visId` at this scale — survivor count × 128, with `kTriIdBits=7` from DW; document the budget like
     visbuffer.h:36). `PackSw(uint32_t depthQ, uint32_t visId) = (depthQ << kSwIdBits) | (visId & ((1<<kSwIdBits)-1))`
     — depth in the HIGH bits so a SMALLER `depthQ` (nearer) makes the whole key smaller → `min` selects the
     front-most. `UnpackSw(uint32_t v, uint32_t& depthQ, uint32_t& visId)`. `kSwClear` (all 1s) = max depth + max id
     → never beaten by a real fragment unless that fragment is nearer (correct background semantics). static_assert
     `kSwClear` doesn't collide with a valid `PackSw` over the realized range (visbuffer.h:41-45 discipline).
   - `struct ScreenVert { int32_t x, y; uint32_t z; };` — fixed-point screen position (`x,y` in `1/kSub`-pixel
     units, `kSub = 16` → 4 sub-pixel bits) + a quantized depth `z` in `[0, (1<<kSwDepthBits)-1]`. **Decision (the
     cross-backend crux): the rasterizer consumes ALREADY-SNAPPED integer `ScreenVert`s** (the projection +
     fixed-point snap is done by the caller, host-side; SW2's GPU does ZERO FP — it gets the same integer
     `ScreenVert`s via SSBO → the coverage is trivially identical cross-vendor). Provide a helper
     `ProjectToScreenVert(worldPos, viewProj, w, h, kSub) -> ScreenVert` (the ONE FP step: `MulPointDivide` →
     NDC → pixel with the Y-flip → `round(px * kSub)`; document it's host-only / the snap rule).
   - `void RasterTriangle(SwVisBuffer&, const ScreenVert& a, const ScreenVert& b, const ScreenVert& c, uint32_t
     visId)` — the deterministic integer edge-function scan-converter:
     - Edge function `E(p, q, r) = int64((p.x - q.x)) * int64((r.y - q.y)) - int64((p.y - q.y)) * int64((r.x -
       q.x))` (int64 intermediates — the fixed-point products can exceed int32). For the 3 edges, evaluate at each
       candidate pixel CENTER (`(x*kSub + kSub/2, y*kSub + kSub/2)` in fixed-point).
     - **Top-left fill rule (vendor-independent, pure integer):** a pixel is covered iff for all 3 edges `E > 0`,
       OR (`E == 0` AND the edge is a top-or-left edge). A top-or-left edge test: an edge `(va→vb)` is a "left" edge
       if `dy = vb.y - va.y < 0`, or a "top" edge if `dy == 0 && dx < 0` (consistent winding; document the winding +
       the exact rule). This makes a shared edge between two triangles cover each boundary pixel EXACTLY once (no
       double-cover, no gap) — unit-tested.
     - Iterate the integer bounding box `[min..max]` of the 3 verts (in pixel units, clamped to `[0,w)×[0,h)`).
       Skip a degenerate (zero-area, `2*area == E(a,b,c) == 0`) triangle.
     - Per covered pixel: **flat per-triangle depth = `min(a.z, b.z, c.z)`** (the SIMPLEST deterministic beachhead
       — conservative; perspective-correct interpolated depth is a deferred refinement, documented). `packed[y*w+x]
       = std::min(packed[y*w+x], PackSw(depthQ, visId))` (the serial `min` — the CPU mirror of SW2's
       `InterlockedMin`; because depth is the high bits, `min` keeps the nearest, ties → lower id deterministically).
   - `void RasterClusters(SwVisBuffer&, span<const ScreenVert> verts, span<const uint32_t> indices, span<const
     Meshlet> meshlets, ...)` — the convenience that rasterizes all clusters' triangles (each `(clusterInstance,
     tri)` → `visId = PackVisId(clusterId, triLocal)`), in a deterministic order; the `min` makes the order
     irrelevant to the result (proven).

2. **Showcase `--swraster-shot <out>` (Vulkan, main.cpp) AND `--swraster` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm in the diff + `#include render/vsm... no: #include render/swraster.h`).** A small fixed
   scene: a few cluster triangles (e.g. 2 overlapping clustered quads/spheres at known depths) projected to a small
   vis-buffer (e.g. 256×256) via `ProjectToScreenVert` → `RasterClusters` → `SwVisBuffer`. CPU-color `packed[]`
   (background → clear color, else `hashColor(visId >> kTriIdBits)`) → write the BMP. PROOFS (fail loudly):
   - **(1) bit-exact reference:** the rasterized `packed[]` matches a pinned reference for the fixed scene (the test
     pins exact bytes; the showcase prints a hash). Print `swraster: {w,h, coveredPixels:N, tris:T}`.
   - **(2) determinism / min order-independence:** rasterize the SAME triangles in a SHUFFLED order → `packed[]`
     byte-identical. Print `swraster determinism (shuffled order): BYTE-IDENTICAL`.
   - **(3) sub-pixel coverage:** include a triangle SMALLER than a pixel positioned so a hardware sample-at-center
     would miss it, assert the SW buffer HAS it covered (the raison d'être — the SW raster catches sub-pixel
     geometry). Print `swraster sub-pixel triangle: COVERED (HW would miss)`.
   - **(4) disabled-path no-op:** an empty triangle set (or `swRasterEnabled=false`) → `packed[]` all-`kSwClear`
     (byte-identical to the cleared init). Print `swraster empty == cleared: BYTE-IDENTICAL`.
   - **Golden** = the CPU-colored vis-buffer → `tests/golden/metal/swraster.png` (the rasterized triangles as
     per-cluster hash-colored regions with correct depth occlusion). CPU-colored from the integer `packed[]` →
     **identical both backends by construction** → trivially DIFF 0.0000 (gate on compare.sh EXIT CODE). Existing 91
     image goldens UNTOUCHED. **GOLDEN DISCIPLINE: ONLY `tests/golden/metal/swraster.png`; do NOT commit it — the
     CONTROLLER bakes it on the Mac.**

3. **Determinism / cross-backend.** The rasterizer is integer edge functions + fixed-point screen verts (the
   projection/snap is host-only; SW2's GPU consumes the same integer `ScreenVert`s → zero GPU FP) + `min` (the
   atomic-min mirror, commutative). The output is a pure integer buffer → bit-identical every platform/run. The viz
   is CPU-colored from it → identical bytes both backends. (This is the SW1 self-contained guarantee; SW2 proves the
   GPU compute matches this CPU reference bit-for-bit.)

4. **Tests `tests/swraster_test.cpp` (pure CPU; `hf_add_pure_test`):**
   - `PackSw`/`UnpackSw` round-trip over `(depthQ, visId)`; `kSwClear` non-collision; a nearer (smaller depthQ)
     fragment's key < a farther one's (the `min` selects nearer).
   - **Top-left fill rule:** two triangles sharing an edge cover each boundary pixel EXACTLY once (no double-cover,
     no gap) over a small buffer — the watertight-raster contract.
   - **Sub-pixel coverage:** a triangle smaller than a pixel at a known position → exactly the expected covered
     pixel(s).
   - **Min order-independence:** rasterize a set + a shuffled set → identical `packed[]`.
   - **Degenerate:** a zero-area triangle → no coverage. **Depth occlusion:** a nearer triangle over a farther one
     → the nearer `visId` wins on the overlap.
   - Determinism. Clean under `windows-msvc-asan`.

5. **Introspect.** Add exactly `nanite-software-raster` (features) + `--swraster-shot` (showcases).

## RHI seam additions (summary)
- **None.** Pure-CPU header + a CPU-colored golden — NO GPU, NO RHI. New non-backend code (`swraster.h`,
  `swraster_test.cpp`, the showcase) adds ZERO above-seam backend symbols. rhi.h + rhi_factory (baseline 2) + the
  backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — SW2 and beyond)
The GPU COMPUTE rasterizer (SW2 — `swraster.comp.hlsl` with `InterlockedMin` into a vis-buffer SSBO, proven
bit-identical to this CPU reference), the HW-large/SW-small hybrid (SW3), feeding the SW vis-buffer into the DX
resolve (SW4), perspective-correct interpolated depth (flat min-depth first), 64-bit visibility, persistent-thread
cluster binning, any SW==HW byte-exact claim (cross-vendor-infeasible). ONE deterministic integer CPU software
rasterizer with a bit-exact reference + top-left-fill + sub-pixel-coverage + min-order-independence + disabled-path
proof set and a CPU-colored vis-buffer golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 88) + new `swraster_test`. Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--swraster-shot` on Vulkan: the CPU-colored vis-buffer shows the rasterized cluster
   triangles as hash-colored regions with correct depth occlusion (the nearer triangle wins on overlap); `swraster
   determinism (shuffled order): BYTE-IDENTICAL` + `sub-pixel triangle: COVERED` + `empty == cleared: BYTE-IDENTICAL`;
   the `swraster: {...}` line deterministic. Run under the AT Vulkan-validation gate → ZERO errors (the showcase is
   CPU-only + a present; trivially clean).
3. Metal: `visual_test --swraster` → new golden `tests/golden/metal/swraster.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). **Confirm visual_test.mm in the diff + `#include render/swraster.h`; the controller bakes
   the golden on the Mac.**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `swraster.png` added; the other
   91 byte-identical. `git diff master --stat -- tests/golden` = ONLY `swraster.png` (metal) + the 2-line introspect
   json — NO loose `tests/golden/swraster.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+nanite-software-raster` + `--swraster-shot`; introspect test updated.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new `swraster` image
   golden in the Mac round-trip loop AND `--swraster-shot` in the `$vkShots` validation gate.
