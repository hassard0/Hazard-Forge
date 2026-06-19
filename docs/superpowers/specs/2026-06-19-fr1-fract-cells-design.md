# Slice FR1 — Deterministic Fracture/Destruction: CELL PRE-FRACTURE / VORONOI DECOMPOSITION — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice (the BEACHHEAD) of FLAGSHIP #14
> (DETERMINISTIC RIGID-BODY FRACTURE / DESTRUCTION, `hf::sim::fract`). UE5's most-marketed feature (Chaos
> destruction) is float + non-deterministic; this flagship makes a break BIT-EXACT cross-platform and
> lockstep-replayable. FR1 is the deterministic pre-fracture: partition a source volume into N fragment cells
> (a Voronoi decomposition) against a host-supplied seed set — the LINK / decomposition, no dynamics yet (FR2
> extracts fragments, FR3 the bond/break, FR4 the step, FR5 lockstep, FR6 render). PURE INT32 → MSL-native (a
> true GPU pass on BOTH backends, the strongest cross-vendor proof). The `mc.h` `ClassifyCells` mold with
> "nearest seed" instead of "case index". Branch: `slice-fr1`. See [[hazard-forge-fract-roadmap]].

**Goal:** Create `engine/sim/fract.h` (header-only, namespace `hf::sim::fract`, `#include "sim/fpx.h"` read-only)
with the source-volume lattice + the seed set + `ClassifyFractCells` (per lattice sample → its nearest seed by
SQUARED INT32 distance, lowest-seed-index tie-break → a cell-id buffer). Add `shaders/fract_classify.comp.hlsl`
(pure int32, MSL-native). Add `--fract-cells-shot` (Vulkan) / `--fract-cells` (Metal). Bake the integer golden
`fract_cells`. **NO new RHI.**

## Design call: an integer-lattice nearest-seed classify (the MC1 mold), PURE INT32 → MSL-native
FR1 is the GR2/FL2/MC1/GF1 integer-decomposition beachhead. The source volume is a bounded integer **lattice**
(`nx·ny·nz` sample points, the `mc.h` `VoxelField` shape); the seed set is `M` points in the SAME integer lattice
coordinates (host-supplied, deterministic — a fixed seed array in the showcase, NOT generated on GPU). Each
lattice sample is assigned to its **nearest seed** by **squared integer distance** (`dx²+dy²+dz²`, pure int —
NO sqrt, NO float, the `cluster_lod.h` DV squared-distance trick), with the **lowest seed index winning ties**
(the `meshlet.h` DS total-order tie-break — guarantees a single deterministic answer). The result is a `cellId`
per sample = a deterministic Voronoi partition of the lattice. **The crux decision — keep distances INT32:** with
the lattice on small integer coords (each axis `< ~1024`), the max squared distance `3·1023² ≈ 3.1M` fits in
int32 with huge headroom → the classify is **pure int32 → MSL-native** (in `hf_gen_msl`, a TRUE GPU pass on both
Vulkan AND Metal → strict zero-differing-pixel, the strongest cross-vendor proof, exactly like MC1/GF1). NO int64,
NO Vulkan-only split at FR1. (World-unit Q16.16 fragment positions arrive at FR2 for mass properties; FR1's
classification is on the integer lattice.)

## The nearest-seed classify (the bit-exact core — order-independent, race-free)
For each lattice sample at integer coord `(x,y,z)`, over the `M` seeds `s[k]` (integer coords), in ascending `k`:
```
bestId = 0; bestD2 = INT32_MAX
for k in 0..M-1:
    dx = x - s[k].x; dy = y - s[k].y; dz = z - s[k].z       // int32
    d2 = dx*dx + dy*dy + dz*dz                               // int32 squared distance (no sqrt)
    if d2 < bestD2:  { bestD2 = d2; bestId = k }             // STRICTLY less -> lowest-index tie-break
cellId[sampleIndex] = bestId                                 // sampleIndex = (z*ny + y)*nx + x
```
`STRICTLY less` (`<`, not `<=`) is the tie-break: the first (lowest-index) seed at the minimum distance wins, so
the answer is a single deterministic value independent of evaluation order. Each sample writes ONLY its own
`cellId[sampleIndex]` → **order-independent, a GPU thread-race cannot change the result** (the MC1/VT1 set-write
argument) → race-free, `[numthreads(64,1,1)]` (or the MC1 dispatch shape). Pure int32. `fract_classify.comp`
copies this body VERBATIM (one thread per lattice sample). The CPU `ClassifyFractCells` is the byte-for-byte
reference the GPU memcmp's against.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The MC1 classify to MIRROR (`engine/render/mc.h`):** `VoxelField` (the `nx/ny/nz` + `scalar[]` lattice +
  `cellCount()`/`SampleField`), `ClassifyCells` (`mc.h:103` — the per-cell lattice gather → per-cell byte output,
  order-independent, the GPU `mc_classify.comp` reference). FR1's `FractField`/`ClassifyFractCells` is the SAME
  shape: a lattice, a per-sample loop, a pure-int classification written per-sample. Read it for the lattice
  struct + the `sampleIndex = (z*ny+y)*nx+x` linearization + the order-independence argument.
- **The squared-distance + tie-break discipline:** `engine/render/cluster_lod.h` (the DV slice — nearest-by-
  squared-distance, avoids sqrt) + `engine/render/meshlet.h` (the DS slice — Morton sort + total-order tie-break
  for a single deterministic decomposition). Read both to copy the exact "`<` strictly, lowest index wins" form.
- **The fixed-point types (`engine/sim/fpx.h`, read-only):** `fx`, `FxVec3` (carried for FR2's world-unit
  fragment positions; FR1's lattice coords are plain `int32`). **DO NOT modify fpx.h** — `fract.h` is the new
  additive sibling, `#include "sim/fpx.h"` read-only (and nothing else from `engine/sim`).
- **The hashColor viz + showcase mold (`samples/hello_triangle/main.cpp` + `metal_headless/visual_test.mm`):**
  the `--mc-classify-shot` / `--meshlet-viz` showcases that color a lattice/decomposition by `hashColor(id)` —
  FR1's golden colors each lattice sample by `hashColor(cellId)` (the cell mosaic). Reuse the existing 2D
  lattice-slice render path (a Z-slice or an orthographic scatter — match `--mc-classify`'s posture).
- **The int32 MSL-native shader precedent (`shaders/mc_classify.comp.hlsl` / `shaders/cgf_gf_count.comp.hlsl`):**
  pure-int32 compute that IS in `hf_gen_msl` (a true GPU pass on both backends). `fract_classify.comp` is the
  same — ADD it to `hf_gen_msl`. Read one for the SSBO/params binding shape.
- **Showcase + registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp`
  (**REBAKE the introspect JSON golden** — the GR2/CP2/GF lesson), a NEW `tests/fract_test.cpp` (+ wire it into
  the CMake test list like `cgf_test`).

## Design decisions (locked)
1. **`fract.h` NEW header (namespace `hf::sim::fract`):** `FractField{int nx,ny,nz; …}` (the lattice extent) +
   `FractSeed{int x,y,z}` (integer lattice coord; an `id`/payload may be added later) + a `FractCells` result
   (`std::vector<uint32_t> cellId`, one per lattice sample) + `ClassifyFractCells(field, seeds, out)` (the CPU
   reference above) + small helpers (`SampleCount`, `SampleIndex(x,y,z)`, a per-cell sample-count stat). Pure
   int32, header-only, NO device/backend symbols, NO `<cmath>`.
2. **`shaders/fract_classify.comp.hlsl`** — one thread per lattice sample, the nearest-seed loop VERBATIM, pure
   int32, **ADDED to `hf_gen_msl`** (MSL-native, a true GPU pass both backends). Seeds + field dims in the params
   SSBO/UBO; `cellId[]` the output SSBO.
3. **Showcase `--fract-cells-shot <out>` (Vulkan) AND `--fract-cells` (Metal) — WIRE BOTH.** A fixed source
   lattice (e.g. a `32×32×16` block) + a fixed deterministic seed set (e.g. `M ≈ 12-24` host-listed seeds).
   Vulkan: the GPU classify → **memcmp vs the CPU `ClassifyFractCells` reference**. Metal: the GPU classify (also
   memcmp vs CPU). Color a lattice slice by `hashColor(cellId)` → the Voronoi cell mosaic. Golden =
   `tests/golden/metal/fract_cells.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU `cellId` buffer == the CPU reference byte-for-byte. Print
     `fract-cells: {lattice:<nx>x<ny>x<nz>, seeds:<M>, cells:<distinctCellCount>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `fract-cells determinism: two runs BYTE-IDENTICAL`.
   - **(3) partition completeness:** every lattice sample is assigned exactly one cell, and every `cellId` is in
     `[0, M)` (no sample unassigned, no out-of-range id). Print `fract-cells partition: {samples:<N>, assigned:<N>}
     complete` with `assigned == samples`; assert each used `cellId < M`. (Optionally: every seed owns ≥1 sample,
     OR document that a seed may own 0 if fully dominated — pick one and assert it.)
   - **Golden discipline: ONLY `tests/golden/metal/fract_cells.png`; do NOT commit it.** Existing 153 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal GPU == golden, ZERO differing pixels (pure int32
   → MSL-native on both, the MC1/GF1 bar).
6. **Tests `tests/fract_test.cpp` (NEW, pure CPU):** `ClassifyFractCells` — a 1-seed field → all samples cellId 0;
   a 2-seed field → the split plane is the integer bisector, ties (equidistant samples) go to the LOWER seed
   index; a sample exactly on a seed → that seed; partition completeness (all assigned, all `< M`); `SampleIndex`
   round-trip. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-fract-cells` (features) + `--fract-cells-shot` (showcases). **REBAKE
   `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (`git diff master --
   tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the MC1/GF1 surface). `rhi.h` +
  `rhi_factory` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple.h` +
  `couple_grain.h` + `couple_gf.h` + `engine/physics/` + all existing render headers UNCHANGED (`fract.h` is the
  new additive sibling, `#include "sim/fpx.h"` read-only). The ONLY new shader is `fract_classify.comp` (int32,
  in `hf_gen_msl`). Report the seam empty except the one new MSL-native classify shader.

## Out of scope (YAGNI — later FR slices)
Fragment extraction / mass properties (FR2), the bond graph + break model (FR3), the fracture step through `fpx.h`
(FR4), lockstep/rollback (FR5), the lit 3D render (FR6). GPU seed GENERATION (seeds are host-supplied/deterministic
— Poisson-disk or physics-driven seeding is a later refinement). Non-convex source volumes, anisotropic/weighted
Voronoi. FR1 claims ONLY: a deterministic nearest-seed cell partition of an integer lattice, bit-identical
CPU↔Vulkan↔Metal, with the integer golden + the three proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 99) + the NEW `fract_test`. Clean under `windows-msvc-asan`
   (build+run `fract_test` + `introspect_test`).
2. **proofs + visual:** `--fract-cells-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   a coherent Voronoi cell mosaic (distinct colored cells tiling the lattice — pixel-check; the MC1/GF1 lesson).**
3. Metal: `visual_test --fract-cells` → new golden `tests/golden/metal/fract_cells.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm `fract_classify.comp` IS
   MSL-generated (int32, a true GPU pass on both).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fract_cells.png` added; the other
   153 byte-identical. `git diff master --stat -- tests/golden` = ONLY `fract_cells.png` (metal) + the introspect
   json.
5. Introspect JSON rebaked exactly `+deterministic-fract-cells` + `--fract-cells-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple.h`/`couple_grain.h`/
   `couple_gf.h` + `engine/physics/` byte-unchanged; the only new shader is `fract_classify.comp`). `scripts/
   verify.ps1` updated: `fract_cells` golden in the Mac loop + `--fract-cells-shot` in `$vkShots`. `fract_classify.
   comp` IS in `hf_gen_msl`. New `fract_test` wired into the CMake test list.
