# Slice FR3 — Deterministic Fracture/Destruction: BONDED-CLUSTER BREAK MODEL (the new physics) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #14 (DETERMINISTIC
> RIGID-BODY FRACTURE / DESTRUCTION, `hf::sim::fract`) — **THE NEW PHYSICS** (the GR4-friction-equivalent beat of
> the arc). FR1 made the cells, FR2 the fragments; FR3 BONDS adjacent fragments into a welded aggregate and
> BREAKS the bonds an impact over-stresses — emergent crack propagation along the pre-fractured cell boundaries.
> The severed-bond SET is bit-exact deterministic (the headline); the crack PATTERN is emergent/within-band (the
> GR4-repose caveat). int64 → Vulkan-only shader + Metal CPU reference (the FPX1/CL3 split). JACOBI load
> diffusion → multi-thread, NO TDR. Branch: `slice-fr3`. See [[hazard-forge-fract-roadmap]].

**Goal:** Extend `engine/sim/fract.h` (additive — FR1/FR2 byte-unchanged) with the bond graph
(`FractBond`/`FractBonds` + `BuildFractBonds`) + the impact-break model
(`ApplyImpactBreak`/`StepFractBreak` + `BreakImpact` + a connected-component piece count). Add
`shaders/fract_break.comp.hlsl` (int64 → Vulkan-only + Metal CPU reference). Add `--fract-break-shot` (Vulkan) /
`--fract-break` (Metal). Bake the integer golden `fract_break`. **NO new RHI.**

## Design call: a shared-face bond graph + Jacobi impact-load diffusion + threshold sever — deterministic SET
**(A) The bond graph (adjacency) — pure int32, deterministic.** Two fragments are BONDED iff their cells share a
lattice face: scan every sample; for its **+x/+y/+z** face-neighbours only (3 directions → each face counted
once), if the neighbour sample is in a DIFFERENT cell, that is a face-crossing between `cellId[sample]` and
`cellId[neighbour]`. Accumulate per **canonical ordered pair `(a<b)`** a `faceArea` (the face-crossing count =
the bond's contact strength) into a dense `M×M` upper-triangle (M = `seedCount`, small — e.g. 16; the dense
matrix is tiny and lets the build be a fixed-order scan, no hashing). The bond list = the non-zero upper-triangle
entries in ascending `(a,b)` order; each `FractBond` = `{fragA, fragB, faceArea, FxVec3 midpoint (the integer
mean of the crossing sample-pair midpoints, OR centroid-pair midpoint), int64 loadAccum}`. Map cells→fragments
via FR2's `cellToFragment` (skip pairs touching an empty cell — they have no fragment). Pure int32 +
deterministic.

**(B) The break model — int64, Jacobi, Vulkan-only + Metal CPU.** A host-supplied **impact** = `{fragment index,
Q16.16 impulse magnitude}` injects load at one fragment. Over **K Jacobi iterations** the load DIFFUSES through
the **intact** bonds (each intact bond transmits a strength-weighted share of the load differential between its
two fragments into the next-iteration buffer — read iteration-start load, write a SEPARATE buffer, the FL4/GR3
Jacobi discipline → multi-thread, NO TDR), accumulating per-bond load `loadAccum`. After the K iters, a bond
**SEVERS** iff its accumulated load exceeds its break threshold `fxmul(kBreakThreshold, faceArea<<kFrac)` (a
strong/large-face bond resists more — the strength-scaled threshold). The output is a per-bond `severed` flag
(0/1) + the count. int64 (`fxmul`/`fxdiv` of Q16.16 loads). `fract_break.comp` is **Vulkan-only** (DXC compiles
int64; glslc cannot) + the Metal `--fract-break` showcase runs the CPU `ApplyImpactBreak` (byte-identical by
construction — the FPX1/GR2/CL3 split). The bond-graph BUILD (int32) may be a separate MSL-native pass OR
host-built (implementer's call; the GPU-proven pass is the int64 break).

**(C) The pieces (connected components) — pure int32, a stat + the viz.** Over the SURVIVING bonds, label-
propagate fragments into connected clusters (a deterministic iterate-to-fixpoint min-label propagation, OR a
union-find with a fixed merge order — pure int32). The cluster count = the number of rigid PIECES the impact
produced (the "broke into P pieces" proof). The viz colours fragments by cluster id.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **FR1/FR2 (this branch's `fract.h`, read-only — build on, DON'T modify):** `FractField`/`SampleCoord`/
  `SampleIndex` (the lattice + the flat-id↔coord), `FractCells{cellId[]}` (the per-sample cell for the adjacency
  scan), `FractFragments` (`fragments[]` with `cx/cy/cz` centroids, `cellToFragment[]`/`fragmentToCell[]`, the
  `kNoFragment` sentinel), `FractFragment` (centroid + `volume`), `ISqrt32`. The FR3 bond endpoints are FRAGMENT
  indices (via `cellToFragment`).
- **The shared-face adjacency precedent (`engine/nav/navmesh.h`):** the cross-poly/region adjacency build (the
  nav_polygonize / region-merge neighbour detection — adjacent cells sharing an edge/face). FR3's face-crossing
  scan is the 3D voxel-face twin. Read it for the canonical-pair dedup + ascending-order discipline.
- **The int64 Jacobi diffusion mold (`engine/sim/fluid.h` FL4 `SolveDensityConstraint` / `engine/sim/grain.h`
  GR3 `SolveGrainContact`):** the read-iteration-start-state → write-separate-buffer → apply pattern that makes a
  multi-thread relaxation race-free (NO TDR). FR3's load diffusion is the SAME shape over the bond graph. The
  int64 fxmul/fxdiv come from `engine/sim/fpx.h` (`fxmul`, `fxdiv`, `FxVec3`, `kOne`, `kFrac` — read-only).
- **The int64 Vulkan-only + Metal-CPU split precedent (`shaders/grain_contact.comp.hlsl` GR3 / `cgf_buoyancy.comp`
  GF2):** an int64 compute shader NOT in `hf_gen_msl` (Vulkan-only via DXC) + the Metal showcase runs the CPU
  reference. `fract_break.comp` is the same. (Contrast FR1/FR2's int32 shaders which ARE in `hf_gen_msl`.)
- **Showcase + registration:** FR1/FR2's `--fract-*-shot` plumbing — **use the STANDALONE arg-parse loop pattern**
  (NOT main.cpp's `else if` ladder — the FR1 C1061 lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp`
  + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden**), `tests/fract_test.cpp`.

## Design decisions (locked)
1. **The bond graph (`BuildFractBonds(field, cells, fragments, out)`):** the +x/+y/+z face-crossing scan → the
   dense `M×M` upper-triangle `faceArea` accumulate → the ascending `(a,b)` `FractBond` list (fragment endpoints
   via `cellToFragment`, skip `kNoFragment`). Pure int32, deterministic. `FractBond{uint32 fragA, fragB; int32
   faceArea; FxVec3 midpoint; int64 loadAccum;}`.
2. **The break (`ApplyImpactBreak(bonds, fragments, impact, K)` → severed flags + count):** K Jacobi load-
   diffusion iters (read-start/write-separate, NO TDR) → per-bond `loadAccum` → sever iff `loadAccum >
   fxmul(kBreakThreshold, faceArea<<kFrac)`. int64. `fract_break.comp` copies this body VERBATIM. `kBreakThreshold`
   a host-snapped Q16.16 constant (documented). `BreakImpact{uint32 fragment; fx impulse;}`. A no-impact / tiny
   impulse → 0 severed (the welded body is intact).
3. **The pieces (`CountFractPieces(fragments, bonds)`):** label-propagation over the surviving bonds → the cluster
   count. Pure int32, deterministic. 1 piece = intact; >1 = broken.
4. **Showcase `--fract-break-shot <out>` (Vulkan) AND `--fract-break` (Metal) — WIRE BOTH** (standalone arg-parse
   loop). The FR1/FR2 lattice + seeds scene; build the bonds; apply a host-fixed impact (a corner/edge fragment +
   a hard impulse) → sever; render the fragments coloured by **cluster id** (the broken pieces) with the SEVERED
   bonds drawn (e.g. red segments between severed-bond centroids) over the intact bonds. Vulkan: the int64 GPU
   break → **memcmp vs the CPU `ApplyImpactBreak`** (the severed flags + per-bond loadAccum). Metal: the CPU
   reference. Golden = `tests/golden/metal/fract_break.png` (Mac-baked by the CONTROLLER — DO NOT commit).
5. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU per-bond `{loadAccum, severed}` == the CPU reference byte-for-byte. Print
     `fract-break: {fragments:<F>, bonds:<B>, severed:<S>, pieces:<P>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `fract-break determinism: two runs BYTE-IDENTICAL`.
   - **(3) the break is real + threshold-gated:** a HARD impact severs `S > 0` bonds and yields `P > 1` pieces;
     a ZERO/sub-threshold impact severs `0` bonds and yields `P == 1` (intact) — the control. Print
     `fract-break threshold: hard={severed:<S>,pieces:<P>} soft={severed:0,pieces:1}`; assert `S>0 && P>1` for
     hard, `0 && 1` for soft.
   - **(4) crack follows cell boundaries:** every severed bond connects two ADJACENT fragments (a real bond, not
     a fabricated pair) — trivially true by construction but assert `severed ⊆ bonds`. Print `fract-break cracks:
     all <S> severed bonds are cell-boundary adjacencies`.
   - **Golden discipline: ONLY `tests/golden/metal/fract_break.png`; do NOT commit it.** Existing 155 image
     goldens UNTOUCHED.
6. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels (int64
   GPU vs Metal CPU reference — the GR3/GF2 bar).
7. **Tests `tests/fract_test.cpp` additions (pure CPU):** `BuildFractBonds` — a 2-cell field → exactly 1 bond
   with `faceArea` == the shared-face sample count, ascending `(a,b)`; a 3-cell line → 2 bonds; non-adjacent cells
   → no bond. `ApplyImpactBreak` — a hard impact severs ≥1 bond, a zero impact severs none, the threshold scales
   with `faceArea` (a stronger bond survives the same load); the Jacobi diffusion is deterministic (two runs
   identical). `CountFractPieces` — intact graph → 1, one severed bridge → 2. Clean under `windows-msvc-asan`.
8. **Introspect.** Add exactly `deterministic-fract-break` (features) + `--fract-break-shot` (showcases). **REBAKE
   `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the FR2/GR3 surface). `rhi.h` + backend
  dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple.h` + `couple_grain.h` +
  `couple_gf.h` + `engine/physics/` + all existing shaders UNCHANGED. FR1/FR2 `fract.h` code + the FR1/FR2 shaders
  UNCHANGED (FR3 additive). The only new shader is `fract_break.comp` (int64, Vulkan-only — NOT in `hf_gen_msl`).
  Report the seam empty except the new Vulkan-only break shader.

## Out of scope (YAGNI — later FR slices)
The fracture STEP (FR4 — releasing the broken clusters as `fpx::FxBody` and integrating them; FR3 computes the
severed SET, FR4 turns it into moving rubble), lockstep (FR5), the lit render (FR6). True stress tensors / FEM
(FR3's Jacobi load-diffusion is a deterministic PROXY, NOT a validated fracture-mechanics model — the GR4/CP2
within-band caveat). Multiple simultaneous impacts (one impact per evaluation; the stream is FR5's job).
Re-bonding / plastic deformation. FR3 claims ONLY: a deterministic bond graph + an impact-driven threshold break
producing a bit-exact severed-bond SET + piece count, bit-identical CPU↔Vulkan↔Metal, with the integer golden.

## Honest caveats (state them in the header + the proofs)
- The severed-bond **SET + the piece count are exact-deterministic + bit-identical** cross-backend (the headline).
  The crack **PATTERN** (which specific bonds break) is **emergent/within-band** — it depends on the proxy load-
  diffusion model + the tuned `kBreakThreshold`, NOT an analytic fracture solution (the GR4-angle-of-repose
  caveat shape). The claim is determinism + replayability, not physical-fracture accuracy.
- Jacobi single-relaxation residual: the load field after K iters is a deterministic approximation (more iters →
  more diffusion); K is host-fixed for a bounded, deterministic result.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 100) + the new `fract_test` break cases. Clean under
   `windows-msvc-asan` (build+run `fract_test` + `introspect_test`).
2. **proofs + visual:** `--fract-break-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the fragments coloured by cluster/piece with severed bonds marked — a coherent "shattered into pieces" viz
   (pixel-check; the FR1/FR2 lesson).**
3. Metal: `visual_test --fract-break` → new golden `tests/golden/metal/fract_break.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm `fract_break.comp` is
   correctly NOT MSL-generated (int64, Vulkan-only); the FR1/FR2 int32 shaders still ARE.** Cross-vendor STRICT
   ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fract_break.png` added; the other
   155 byte-identical (re-run `--fract-cells/fragments-shot` → still bit-exact). `git diff master --stat --
   tests/golden` = ONLY `fract_break.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fract-break` + `--fract-break-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h` +
   `engine/physics/` + FR1/FR2 `fract.h`/shaders byte-unchanged). `scripts/verify.ps1` updated: `fract_break`
   golden in the Mac loop + `--fract-break-shot` in `$vkShots`. `fract_break.comp` NOT in `hf_gen_msl`.
