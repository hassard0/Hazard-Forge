# Slice VA — Virtual Shadow Maps Slice 1: Page Table + Page-Needed Marking (BEACHHEAD) (Phase 7 #1) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The BEACHHEAD of the VIRTUAL
> SHADOW MAPS flagship (the 3rd UE5-parity flagship — VSM = Nanite-scale shadowing: huge virtual resolution,
> sparsely backed by physical pages). VA builds the virtual PAGE TABLE + the page-needed MARKING (which virtual
> shadow pages are needed by the visible receivers), as a pure integer compute pass with a GPU==CPU bit-exact
> proof + a resident-page debug-viz golden. NO rendering, NO new RHI — the tractable first slice the rest of the
> arc (VB physical-page render → VC lit indirection sample → VD caching) builds on. Namespace `hf::render::vsm`.
> Branch: slice-va-vsmmark. See [[hazard-forge-vsm-roadmap]].

**Goal:** Add `engine/render/vsm.h` (clipmap math + page-table data structure + page-needed marking) + a
`vsm_mark.comp.hlsl` compute mirror, and a `--vsm-mark-shot` showcase that marks which virtual pages the visible
receivers need and writes a top-down clipmap debug-viz (resident pages colored). Make-safe by construction: a NEW
header + NEW shader + NEW showcase + NEW golden; NO existing shadow path/pipeline/golden touched. The marking output
is a pure INTEGER set → inherently bit-exact + cross-backend; proven GPU==CPU via `ReadBuffer` memcmp.

## Reuse map (file:line — from the scout)
- **`engine/render/froxel.h` / `cluster.h`** — the flat-SSBO + mirrored-CPU-header + no-op-flag TEMPLATE: `FroxelGrid`
  (froxel.h:62), `ClusterGrid` flat `idx = cx + cy*dimX + cz*dimX*dimY` (cluster.h:57). `vsm.h` copies this shape.
- **The compute + readback surface** — `BufferUsage::Storage` (rhi.h:166), `BindComputePipeline`/`BindStorageBuffer`/
  `ComputePushConstants`/`DispatchCompute` (rhi.h:412-426), `ReadBuffer` (rhi.h:610). The exact GPU==CPU bit-exact
  proof surface (froxel/cluster/probe-SH). **NO new RHI.**
- **DW `engine/render/visbuffer.h` + the `--visbuffer-shot` proof/viz** (main.cpp:1229-1240) — the `PackVisId`/
  `UnpackVisId` bijection + the `ReadBuffer` bit-exact proof + the `hashColor`-CPU-colored debug-viz golden
  TEMPLATE. VA mirrors this exactly (pageId pack/unpack + resident-set memcmp + hashColor(pageId) viz).
- **The DH cross-backend FP discipline** (probe_gi.h) — host-precompute the `log2`/projection transcendentals + use
  `fma`/`mad` so the clipmap LEVEL selection is integer-stable + bit-identical CPU↔Vulkan↔Metal.
- **DW MSL-2.2 lesson** (commit f693c63) — the integer marking shader's compute (and the viz if it texel-fetches)
  may need spirv-cross `--msl-version 20200`; add the ISOLATED per-shader flag (the `hf_gen_msl(... --msl-version
  20200)` mechanism) so every other shader's MSL-gen is byte-identical.

## Design decisions (locked)

1. **`engine/render/vsm.h` (NEW, pure CPU, `hf::render::vsm`; 0 above-seam backend symbols; mirrors froxel.h).**
   - `struct VsmClipmap { int levels; int pageSize; int virtualPagesPerSide; float level0WorldExtent; math::Vec3
     cameraPos; };` — clipmap level `L` covers `level0WorldExtent * 2^L` world units across `virtualPagesPerSide²`
     pages, centered on `cameraPos` (the standard directional clipmap). `pageCount() = levels * vpps * vpps`.
   - Flat page-table index: `PageId(level, px, py) = level*(vpps*vpps) + py*vpps + px` (the cluster.h:57 / froxel.h:67
     flat-index discipline); `UnpackPageId(id) -> (level, px, py)`. A bijection over `[0, pageCount())` (unit-tested
     like `PackVisId`/`UnpackVisId`).
   - `int SelectClipmapLevel(float distToCamera, const VsmClipmap&)` = `clamp(floor(log2(max(distToCamera, eps) /
     level0WorldExtent)), 0, levels-1)`. **DETERMINISM CRUX:** `log2` is a transcendental → to keep the integer LEVEL
     bit-identical CPU↔GPU, AVOID a raw GPU `log2`: instead select the level by an INTEGER comparison ladder against
     host-precomputed `level0WorldExtent * 2^L` thresholds (a small `levels`-entry table uploaded as exact float32
     bits both CPU + shader read) — `level` = the number of thresholds `distToCamera` exceeds, clamped. Pure
     compare/count → no transcendental on the bit-exact path → integer-stable cross-backend. Document this (it is
     the make-or-break for GPU==CPU, exactly like the DV squared-distance / DW flat-integer discipline).
   - `bool MarkPage(const math::Vec3& worldPos, const VsmClipmap&, int& outPageId)` — `level =
     SelectClipmapLevel(length(worldPos - cameraPos), ...)`; project `worldPos` into that level's clipmap ortho
     (origin = `cameraPos` snapped to the level's page grid; extent = `level0WorldExtent * 2^level`) → `(px,py)` in
     `[0, vpps)`; `outPageId = PageId(level,px,py)`; return in-range. (The XY projection is the directional-light
     clipmap's top-down axis-aligned map — plain subtract/divide/floor, integer-stable.)
   - `MarkResidentPages(std::span<const math::Vec3> receiverSamples, const VsmClipmap&, std::span<uint32_t>
     residentOut)` — the CPU reference: zero `residentOut`, for each receiver sample `MarkPage` → `residentOut
     [pageId] = 1`. This is what the GPU `vsm_mark.comp` matches byte-for-byte.

2. **`shaders/vsm_mark.comp.hlsl` (NEW).** One thread per receiver sample (an SSBO of `float3` receiver world
   positions — for a deterministic beachhead, a fixed host-generated receiver point-set, e.g. a ground-plane grid +
   the scene geometry's sample points; NO depth-buffer reconstruction yet — that's a later refinement). Copy
   `SelectClipmapLevel` (the threshold-ladder form) + `MarkPage` VERBATIM from `vsm.h`. Output: the `resident[]`
   SSBO — `resident[pageId] = 1` (use an atomic-OR or a write; the SET is order-independent so writes race-free to
   `1`). `markingEnabled` push flag: false → write nothing (empty set). `ComputePipelineDesc{ storageBufferCount,
   threadsPerGroupX = 64 }`. NO new RHI. Only `[[vk::binding]]` + `HF_MSL_GEN` above-seam. Add the ISOLATED
   `--msl-version 20200` for this shader's MSL-gen (the DW lesson) — needed if the integer SSBO write / atomics
   require it on Metal (verify on the Mac).

3. **Showcase `--vsm-mark-shot <out>` (Vulkan, main.cpp) AND `--vsm-mark` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm in the diff, the DS lesson).** A fixed clipmap (e.g. `levels=4, vpps=8, pageSize=128,
   level0WorldExtent=16, cameraPos` at a deterministic spot) + a fixed receiver point-set (a ground grid spanning
   several clipmap levels). Dispatch `vsm_mark.comp` → `ReadBuffer` the `resident[]`. PROOFS (fail loudly):
   - **(1) GPU==CPU bit-exact:** CPU `MarkResidentPages` over the SAME receiver set → `memcmp(gpuResident,
     cpuResident) == 0` (integer set, NO FP tolerance). Print `vsm-mark GPU==CPU resident set: <N> pages BIT-EXACT`.
   - **(2) pageId pack/unpack bijection** (asserted in the test; the showcase prints the resident count per level).
   - **(3) markingEnabled=false → empty set** (all-zero `resident[]`, byte-identical to the cleared upload).
   - **(4) two-run determinism** byte-identical.
   - **Golden** = a top-down clipmap debug-viz: color each resident page by `hashColor(pageId)` over its clipmap
     level (concentric clipmap rings — level 0 the small central region, each level 2× larger; resident pages lit,
     non-resident dark) → `tests/golden/metal/vsm_pages.png`. CPU-colored from the read-back integer set →
     **identical both backends by construction** → trivially DIFF 0.0000 (gate on compare.sh EXIT CODE). Print
     `vsm-mark: {levels:4, vpps:8, residentPages:N}`. Existing 87 image goldens UNTOUCHED. **GOLDEN DISCIPLINE: ONLY
     `tests/golden/metal/vsm_pages.png`.**

4. **Determinism / cross-backend.** The level selection is an integer threshold-ladder (host-precomputed thresholds,
   no GPU transcendental); the projection is subtract/divide/floor; the output is a pure integer set → bit-exact
   GPU==CPU + cross-backend + two-run identical. The viz is CPU-colored from the read-back set → identical bytes
   both backends.

5. **Tests `tests/vsm_test.cpp` (pure CPU; `hf_add_pure_test`):**
   - `PageId`/`UnpackPageId` bijection over `[0, pageCount())`.
   - `SelectClipmapLevel`: a receiver AT the camera → level 0; a far receiver → the top level; the `2^L` boundary
     distances select the expected level (threshold-ladder edges).
   - `MarkResidentPages` over a known point-set → the expected resident set; a single point → exactly 1 resident
     page at the expected `(level,px,py)`.
   - `markingEnabled`-off semantics (the CPU helper for the disabled path) → empty set.
   - Determinism. Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `virtual-shadow-maps-marking` (features) + `--vsm-mark-shot` (showcases).

## RHI seam additions (summary)
- **None.** Pure compute over an integer SSBO + `ReadBuffer` (the froxel/cluster surface). New non-backend code
  (`vsm.h`, `vsm_mark.comp.hlsl`, `tests/vsm_test.cpp`, the showcase) adds ZERO above-seam backend symbols. rhi.h +
  rhi_factory (dispatch baseline 2) + the backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — VB and beyond)
Physical-page depth rendering (VB), the lit-pass VSM indirection sample (VC), per-page caching (VD), depth-buffer
receiver reconstruction (VA uses a fixed host receiver point-set — depth reconstruction is a later refinement),
sparse virtual texturing / dynamic eviction / Nanite-cluster-driven per-page culling (the deferred stretch). ONE
clipmap page table + page-needed marking with a GPU==CPU bit-exact resident-set proof + pack/unpack bijection +
markingEnabled=false no-op + the resident-page debug-viz golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 87) + new `vsm_test`. Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--vsm-mark-shot` on Vulkan: the top-down clipmap viz shows coherent concentric clipmap
   rings with the resident pages lit (hashColor); `vsm-mark GPU==CPU resident set: <N> pages BIT-EXACT` +
   `markingEnabled=false → empty` + two-run byte-identical; the `vsm-mark: {...}` line deterministic. Run under the
   AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --vsm-mark` → new golden `tests/golden/metal/vsm_pages.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). The GPU==CPU proof also passes on Metal. **Confirm visual_test.mm in the diff; confirm
   vsm_mark.comp MSL generates (the isolated --msl-version 20200 if needed). NEVER trust an agent-committed Metal
   golden — the controller bakes it on the Mac.**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `vsm_pages.png` added; the other
   87 byte-identical. `git diff master --stat -- tests/golden` = ONLY `vsm_pages.png` (metal) + the 2-line
   introspect json — NO loose `tests/golden/vsm_pages.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+virtual-shadow-maps-marking` + `--vsm-mark-shot`; introspect test updated.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new `vsm_pages`
   image golden in the Mac round-trip loop AND `--vsm-mark-shot` in the `$vkShots` validation gate.
