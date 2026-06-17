# Slice MC4 — GPU Isosurface Meshing: FIXED-POINT INTERPOLATED vertex placement (Phase 10 #4) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 4th MC slice (after MC3's
> midpoint emission): place each edge vertex at the ACTUAL isosurface crossing (linear interpolation along the edge by
> the corner scalar values) instead of the edge midpoint — the quality step that makes the extracted surface smooth
> and watertight-on-the-field. DESIGN: the interpolation parameter `t = (iso - s0)/(s1 - s0)` is computed in
> FIXED-POINT integer arithmetic on a `1/kSub` edge lattice (the swraster `kSub` trick) — NO float escape — so the
> interpolated vertex buffer stays GPU==CPU BIT-EXACT and the golden is cross-backend BIT-IDENTICAL. MC3's midpoint
> emission is left UNCHANGED (its `mc_emit` golden is the t=0.5 reference); MC4 adds a sibling interpolated emit. NO
> new RHI. Namespace `hf::render::mc`. Branch: `slice-mc4-interp`. See [[hazard-forge-mc-roadmap]].

**Goal:** Extend `engine/render/mc.h` with the fixed-point interpolated edge-vertex placement (`kSub`, `EdgeInterp` →
an `McVertex` in `1/kSub` units, `EmitCellInterp`, `MarchCellsInterp`) + add `shaders/mc_interp.comp.hlsl` (the MC3
emit with `EdgeInterp` instead of `EdgeMidpoint`) + a `--mc-interp-shot` (Vulkan) / `--mc-interp` (Metal) showcase that
meshes the MC1–MC3 sphere field with interpolated vertices, reads back the vertex+index buffers, proves them BIT-EXACT
vs the CPU mesher, and bakes an integer mesh-projection golden. Make-safe: header additions + a NEW shader + NEW
showcase + NEW golden; MC1–MC3 goldens + everything else UNCHANGED (in particular MC3's `mc_emit` + `mc_scan.comp` +
`mc_emit.comp` are untouched — MC4 adds a sibling interpolated emit; the prefix-sum/scan is reused as-is).

## The fixed-point interpolation (the cross-backend crux)
- **`kSub = 256`** — the fixed-point edge lattice (finer than swraster's `kSub=16`, for a smooth surface). A vertex
  coordinate is stored in `1/kSub` grid units, so a midpoint is `kSub/2` and the cube spans `[0, dim*kSub)`.
- **`McVertex EdgeInterp(int cx, int cy, int cz, int edge, const VoxelField&, int32_t iso)`** — the interpolated
  crossing on `kEdgeCorner[edge] = (a,b)`. `Ca = cell + kCornerOffset[a]`, `Cb = cell + kCornerOffset[b]` (the two
  differ by exactly +1 in ONE axis — MC edges are axis-aligned). `s0 = SampleField(Ca)`, `s1 = SampleField(Cb)`.
  Compute `t` in `[0, kSub]` fixed-point:
  - if `s1 == s0` (degenerate, no gradient): `t = kSub / 2` (the midpoint — matches MC3's choice, deterministic).
  - else: `t = ((int64_t)(iso - s0) * kSub) / (s1 - s0)` (integer division — truncation toward zero, IDENTICAL in
    C++ and HLSL, so bit-exact CPU↔GPU; use int64 intermediates to avoid overflow); then clamp `t` to `[0, kSub]`.
  - `vertexPos.axis = Ca.axis * kSub + (Cb.axis - Ca.axis) * t` per axis (the spanning axis moves by `±t`, the other
    two stay at `Ca.axis * kSub`). Return `{x, y, z, 0}` in `1/kSub` units. **Pure integer; the ONE division is the
    same truncating integer divide on both sides → bit-identical.** (Document: truncation-toward-zero is the C++/HLSL
    `/` default and is consistent across backends; the `t` clamp guards out-of-range from a non-bracketing edge.)
- **`EmitCellInterp` / `MarchCellsInterp`** — the MC3 `EmitCell`/`MarchCells` with `EdgeInterp` substituted for
  `EdgeMidpoint` (same triangle-soup layout, same identity indices, same `kTriTable` walk). The CPU reference the GPU
  memcmp's against.

## Reuse map (file:line)
- **MC3 (the structure to mirror):** `engine/render/mc.h` — `McVertex`, `EdgeMidpoint`, `EmitCell`, `MarchCells`,
  `PrefixSumOffsets`; `shaders/mc_scan.comp.hlsl` (the prefix-sum, REUSED UNCHANGED), `shaders/mc_emit.comp.hlsl` (the
  emit template — `mc_interp.comp` is this with `EdgeInterp`). MC1/MC2: `CaseIndex`, `kTriTable`, `SampleField`,
  `kCornerOffset`, `kEdgeCorner`, `CountCells`/`TotalTriangles`.
- **The fixed-point integer-lattice discipline:** `swraster.h:85-96` (`kSub` sub-unit lattice; the host-only FP →
  integer-on-GPU pattern) + `swraster.h` `FloorDiv` (the deterministic integer-division helper, if a floor form is
  preferred over truncation — pick ONE and copy it verbatim CPU↔GPU). int64 edge math precedent:
  `swraster.comp.hlsl` (int64 in HLSL via DXC — note the Vulkan-only/Metal-CPU-path lesson if int64 trips glslc; the
  interp `t` numerator `(iso-s0)*kSub` fits int32 for this field [|scalar| ~ field*256, ×256 → ~2^24], so **prefer
  int32** to avoid the int64/glslc issue — confirm the bound + document, else int64 + the Vulkan-only convention).
- **Compute + readback surface (NO new RHI):** the MC3 surface — `BufferUsage::Storage` + compute + compute→compute
  barrier + `ReadBuffer`.
- **Golden + registration:** the MC3 `--mc-emit-shot` 2D-projection-viz template; `meshlet.h:79` `hashColor`;
  `scripts/verify.ps1` `$Goldens`/`$vkShots`, `engine/editor/introspect.cpp`, `tests/introspect_test.cpp`.

## Design decisions (locked)

1. **`mc_interp.comp.hlsl` (NEW).** A copy of `mc_emit.comp` with `EdgeMidpoint` replaced by `EdgeInterp` (the fixed-
   point `t` + clamp copied VERBATIM from `mc.h`); it additionally reads the two edge-corner scalars from `gField`
   (already bound for reclassification). One thread per cell, disjoint output ranges, no atomics, `meshEnabled=0` →
   empty. Reuses `mc_scan.comp` UNCHANGED for the prefix-sum. **Prefer int32 for `t`** (the numerator bound fits;
   document); if int64 is required, apply the swraster Vulkan-SPIR-V-only + Metal-CPU-path convention. Plain integer
   (modulo the int64 caveat) → NO `--msl-version 20200`. Register in BOTH compile lists.
2. **Showcase `--mc-interp-shot <out>` (Vulkan, main.cpp) AND `--mc-interp` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm + `#include "render/mc.h"`).** The SAME MC1–MC3 sphere field. Host `CountCells` →
   `TotalTriangles=5240`, preallocate `gVerts`/`gIdx` to `3*5240`, clear, upload `gField`+`gCounts`+`gTriTable`,
   dispatch `mc_scan` → (barrier) → `mc_interp` → `ReadBuffer`. CPU-run `MarchCellsInterp`. Golden = the same 2D
   orthographic projection as MC3 (vertex `1/kSub` coords → pixel via integer divide, point-splat `hashColor` by cell)
   → `tests/golden/metal/mc_interp.png` — a SMOOTHER filled-disk sphere than `mc_emit` (vertices on the true crossing,
   not the cell-edge midpoint); INTEGER → cross-backend bit-identical.
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU mesh bit-exact (make-or-break):** `memcmp(gpuVerts, cpuVerts)==0` AND `memcmp(gpuIdx, cpuIdx)==0`
     over `3*totalTris`. Print `mc-interp GPU==CPU mesh: <V> verts + <I> indices BIT-EXACT`.
   - **(2) interpolation correctness:** for a symmetric edge (`s0`, `s1` equidistant from `iso`) `EdgeInterp == the
     midpoint` (`t==kSub/2`); for a known asymmetric edge the `t` matches the hand-computed fixed-point value; every
     interpolated vertex lies WITHIN the edge bounds (each axis in `[Ca*kSub, Cb*kSub]`). Print `mc-interp: {tris:<T>,
     verts:<V>, kSub:256, on-edge:<k>/<V> OK}`.
   - **(3) on-surface property:** at the interpolated `t`, the linearly-reconstructed edge scalar `s0 + (s1-s0)*t/kSub`
     equals `iso` within ±1 fixed-point unit (the vertex sits on the isosurface to lattice precision). Print
     `mc-interp on-surface: max |field-iso| = <e> (<=1 fixed-pt) OK`.
   - **(4) known-cell vertex:** a hand-checked cell's first interpolated vertex `v0`. Print `mc-interp known cell:
     case=0x<XX> v0=(<x>,<y>,<z>) OK`.
   - **(5) disabled-path no-op:** `meshEnabled=false` → empty. Print `mc-interp disabled: empty mesh (no-op)`.
   - **(6) determinism:** two runs byte-identical. Print `mc-interp determinism: two runs BYTE-IDENTICAL`.
   - **Golden discipline: ONLY `tests/golden/metal/mc_interp.png`; do NOT commit it — the CONTROLLER bakes on the
     Mac.** Existing 102 image goldens UNTOUCHED.
4. **Tests `tests/mc_test.cpp` additions (pure CPU):** `EdgeInterp` symmetric→midpoint, asymmetric→known fixed-point
   `t`, degenerate `s0==s1`→midpoint, clamp on a non-bracketing edge, every axis in `[Ca*kSub, Cb*kSub]`;
   `MarchCellsInterp` over a tiny field → known verts; on-surface property (reconstructed scalar ≈ iso); determinism;
   `meshEnabled` off → empty. Clean under `windows-msvc-asan`.
5. **Introspect.** Add exactly `gpu-isosurface-meshing-interp` (features) + `--mc-interp-shot` (showcases).

## RHI seam additions (summary)
- **None.** The MC3 surface (Storage SSBO + compute + barrier + `ReadBuffer`). ZERO above-seam backend symbols.
  `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — MC5+)
Cross-cell vertex DEDUP (still a soup), rendering the mesh (MC5), field-gradient normals + lit shading (MC6, the first
float / visresolve-bar slice), perspective-correct anything. ONE interpolated-vertex emit (fixed-point, bit-exact) +
the GPU==CPU mesh proof + interpolation/on-surface correctness + known-cell + disabled no-op + determinism and the
integer interpolated-mesh-projection golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 91) + the new `mc_test` interp cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--mc-interp-shot` on Vulkan: a coherent (smoother than `mc_emit`) 2D mesh projection; all 6
   proof lines. Run under the Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --mc-interp` → new golden `tests/golden/metal/mc_interp.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). The GPU==CPU + determinism proofs also pass on Metal. **Confirm visual_test.mm in the diff;
   confirm mc_interp.comp MSL-generates (int32 path → no MSL-2.2; if int64 was needed, confirm the Vulkan-only/
   Metal-CPU-path handling).** Integer golden → a strict cross-backend pixel compare must show ZERO differing pixels.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `mc_interp.png` added; the other 102
   byte-identical (MC1–MC3 untouched, incl `mc_emit`). `git diff master --stat -- tests/golden` = ONLY `mc_interp.png`
   (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+gpu-isosurface-meshing-interp` + `--mc-interp-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `mc_emit.comp` + `mc_scan.comp` UNCHANGED. `scripts/verify.ps1`
   updated: `mc_interp` golden in the Mac loop + `--mc-interp-shot` in `$vkShots`.
