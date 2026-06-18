# Slice CL2 — Deterministic GPU Cloth: DISTANCE-CONSTRAINT GRAPH BUILD (Phase 13 #2) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #8
> (DETERMINISTIC GPU CLOTH, `hf::sim::cloth`, header `engine/sim/cloth.h`). Builds the DISTANCE-CONSTRAINT
> GRAPH over the CL1 particle lattice — structural + shear + bend edges with their rest lengths — the
> graph the CL3 PBD solver projects. Built by the proven count→scan→emit compaction, pure INT32 →
> Metal-MSL-native (a true GPU pass on both backends, unlike CL1's int64 integrate). Strict zero-diff
> cross-backend (the integer bar). ZERO new RHI. Branch: `slice-cl2`. See [[hazard-forge-cloth-roadmap]].

**Goal:** Extend `engine/sim/cloth.h` with `BuildConstraints` (enumerate the cloth's distance constraints
over the W×H lattice: STRUCTURAL edges — each particle to its right + down neighbour; SHEAR — to its two
diagonal neighbours; BEND — to its 2-away right + down neighbour; each a `Constraint{uint32 i, j; fx
restLen; uint32 kind}` with `restLen = FxLength(pos[j] - pos[i])` at build). Lay the variable-count output
out with the count→scan→emit GPU compaction. Add `shaders/cloth_edge_{count,scan,emit}.comp.hlsl` (int32
→ Metal-native), the `cloth_edges` integer golden (the constraint-adjacency viz, edge kinds color-coded
from the integer read-back → strict zero-diff cross-backend), `--cloth-edges-shot` (Vulkan) /
`--cloth-edges` (Metal), and `tests/cloth_test.cpp` additions. Reuse CL1 verbatim (ClothParticle /
ClothGrid / InitGrid) — CL2 is additive (CL1's integrate pipeline + golden stay byte-identical).

## Design call: INTEGER bit-exact, int32-NATIVE (the FPX2 twin)
Edge enumeration is pure integer index arithmetic over the lattice; `restLen` is the integer `FxLength`
(`fpx.h`, which uses `FxISqrt` — note `FxLength` internally is int64, but it is computed HOST-SIDE at
build for the CPU reference AND the result is a host-snapped int32 stored in the edge; the SHADER's edge
build can recompute `FxLength` OR consume host-snapped restLens — see decision 2). The count→scan→emit
itself (per-particle edge count → prefix-sum → emit at offset) is pure int32 index/compare math → the
shaders MSL-generate NATIVELY and run as true GPU passes on both backends (the FPX2 `fpx_pair_*` precedent
— int32, MSL-native, unlike the int64 cloth_integrate/fpx_integrate). The golden is the CPU-colored
integer edge buffer → Vulkan==Metal BIT-IDENTICAL (strict zero-diff). NO new RHI.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The CL1 lattice (the input):** `engine/sim/cloth.h` — `ClothParticle`, `ClothGrid{W,H,spacing,origin}`,
  `InitGrid`. The constraint `restLen` = `FxLength(pos[j]-pos[i])` (`fpx.h::FxLength`). CL2 adds
  `BuildConstraints` + the `Constraint` struct to this same header; CL1 functions UNCHANGED.
- **The count→scan→emit structural template (int32, MSL-native):** `engine/sim/fpx.h::CountPairs`/
  `BuildPairs` + `shaders/fpx_pair_count.comp.hlsl` + `fpx_pair_scan.comp.hlsl` (`[numthreads(1,1,1)]`
  serial exclusive prefix-sum) + `fpx_pair_emit.comp.hlsl` — the FPX2 int32 twin. Also `engine/render/
  mc.h` `mc_count`/`mc_scan`/`mc_emit` + NAV1's `nav_raster_{count,scan,emit}`. Mirror EXACTLY: per
  particle count its OWNED edges (to avoid double-counting, each edge is owned by its lower-index
  endpoint), single-thread prefix-sum, per particle emit at offset.
- **Integer length:** `engine/sim/fpx.h::FxLength`/`FxISqrt`. For the rest length, compute it host-side at
  build (the lattice is host-snapped) and store the int32 restLen in the edge — so the edge buffer is pure
  int32 and the count→scan→emit shaders never need int64 (the `FxLength` is a build-time constant per
  edge, like NAV1's `TriYSpan` / FPX2's rest data). Document this (keeps CL2 fully int32/MSL-native).
- **The integer-golden showcase discipline:** CL1's `--cloth-integrate-shot` / `RunClothIntegrateShowcase`
  + FPX2's `--fpx-pairs-shot` / `RunFpxPairsShowcase` (ReadBuffer the integer result, memcmp GPU==CPU,
  CPU-color the integer buffer, strict zero-diff). For edges, draw the lattice graph with each edge kind
  (structural/shear/bend) a distinct color (the `fpx_pairs` banded-matrix discipline, or a node-link viz).
- **RHI compute envelope (must fit, ZERO additions):** `engine/rhi/rhi.h` — BindStorageBuffer,
  DispatchCompute, ComputeToComputeBarrier, ComputePushConstants, ReadBuffer, InterlockedAdd (the
  CL1/FPX2 set).
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--cloth-edges-shot` standalone arg
  branch + DXC list), `metal_headless/{visual_test.mm,CMakeLists.txt}` (`RunClothEdgesShowcase` +
  `--cloth-edges` + the 3 shaders in `hf_gen_msl` — int32 → Metal-native),
  `engine/editor/introspect.cpp` (+`deterministic-cloth-constraints` feature + `--cloth-edges-shot`
  showcase) + `tests/introspect_test.cpp` + the JSON golden, `scripts/verify.ps1` (`cloth_edges` golden
  in the Mac loop + `--cloth-edges-shot` in `$vkShots`).

## Design decisions (locked)
1. **The constraint set.** `Constraint{ uint32 i, j; fx restLen; uint32 kind; }` (kind: 0=STRUCTURAL,
   1=SHEAR, 2=BEND). Over the W×H lattice (index `r*W+c`): STRUCTURAL = (c,c+1) horizontal + (r,r+1)
   vertical; SHEAR = the two diagonals of each cell ((r,c)-(r+1,c+1) and (r,c+1)-(r+1,c)); BEND = (c,c+2)
   + (r,r+2). Each edge owned by its lower linear index (no double-count). `restLen = FxLength(pos[j] -
   pos[i])` host-snapped int32 at build (the initial flat-sheet rest geometry). Deterministic enumeration
   order: ascending owner index, then a fixed per-particle edge order (right, down, diag1, diag2,
   bend-right, bend-down). Pure int32.
2. **GPU pipeline (count→scan→emit, the FPX2 mirror).** Pass 1 `cloth_edge_count`: one thread per particle
   counts its OWNED edges (the neighbours that exist within the grid bounds) → `gEdgeCount[p]`. Pass 2
   `cloth_edge_scan`: `[numthreads(1,1,1)]` exclusive prefix-sum → `gEdgeOffset[p]`. Pass 3
   `cloth_edge_emit`: one thread per particle writes its edges into `gEdges` at its offset, in the FIXED
   per-particle order, each with its `restLen` (host-snapped int32, recomputed in-shader from the
   host-snapped lattice OR read from a host buffer — keep it int32). **All int32 → in `hf_gen_msl`,
   Metal-native.** GPU `{gEdgeCount, gEdgeOffset, gEdges}` == the CPU `BuildConstraints` reference
   byte-for-byte.
3. **Showcase `--cloth-edges-shot <out>` (Vulkan) AND `--cloth-edges` (Metal — WIRE BOTH; confirm
   visual_test.mm + `#include "sim/cloth.h"`).** The CL1 24×24 sheet (rest geometry) → BuildConstraints.
   ReadBuffer `gEdges` + `gEdgeOffset`; **memcmp GPU == the CPU `BuildConstraints` reference (the
   make-or-break)**; CPU-color a graph viz (the lattice nodes + edges, structural/shear/bend color-coded
   by kind) → `tests/golden/metal/cloth_edges.png` (baked on the Mac by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance / GPU==CPU (make-or-break):** `gEdges` + `gEdgeOffset` equal the CPU reference
     byte-for-byte. Print `cloth-edges: {particles:<N>, edges:<E>, struct:<S>/shear:<H>/bend:<B>}
     GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs BYTE-IDENTICAL. Print `cloth-edges determinism: two runs BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the edge counts match the expected lattice formula (e.g. structural =
     W(H-1)+H(W-1)), every edge's endpoints are in-bounds + i<j, restLen>0. Print `cloth-edges coverage:
     <E> edges over <N> particles (structural/shear/bend, all in-bounds)`.
   - **(4) empty / degenerate:** a 1×1 grid → 0 edges → cleared background. Print `cloth-edges empty:
     0 edges (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/cloth_edges.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 118 image goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict).** The edge buffer is pure int32, host-snapped → ZERO GPU float →
   Vulkan==Metal BIT-IDENTICAL: the golden is the CPU-colored integer read-back; the controller's
   cross-backend check is the STRICT ZERO-DIFFERING-PIXEL compare. Any nonzero cross-backend diff is a bug.
6. **Tests `tests/cloth_test.cpp` additions (pure CPU):** a 2×2 grid → the hand-enumerated edge set
   (4 structural, 2 shear, 0 bend) with correct restLens; the edge-count formula for a W×H grid; every
   edge i<j + in-bounds; no duplicate edges; restLen of a flat sheet == spacing (structural) / spacing·√2
   (shear, within fp tol); determinism; 1×1 → empty. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-cloth-constraints` (features) + `--cloth-edges-shot`
   (showcases). Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None expected.** Pure SSBO compute (BindStorageBuffer, DispatchCompute, ComputeToComputeBarrier,
  ComputePushConstants, ReadBuffer, InterlockedAdd — the CL1/FPX2 set). `rhi.h` + `rhi_factory` (baseline
  2) + backend dirs UNCHANGED. CL1's `cloth_integrate.comp` + `engine/sim/fpx.h` + `engine/physics/`
  UNCHANGED. Report the seam.

## Out of scope (YAGNI — later CL slices)
The PBD constraint SOLVER (CL3 — uses these edges), collision (CL4), lockstep (CL5), the float render
(CL6). CL2 is ONLY the constraint graph build + its bit-exact golden. No solving, no float. (Bending as a
distance constraint, not a dihedral model — the documented simplification.)

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 94) + the new `cloth_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--cloth-edges-shot` on Vulkan: GPU==CPU memcmp BIT-EXACT + determinism +
   coverage + empty no-op; a coherent constraint-graph image. Run under the Vulkan-validation gate → ZERO
   VUID (set BOTH `VK_LAYER_PATH` to the conan `...\.conan2\p\...\layers` dir AND
   `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer LOADED = zero "not found").
3. Metal: `visual_test --cloth-edges` → new golden `tests/golden/metal/cloth_edges.png`; two runs DIFF
   0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm
   cloth_edge_{count,scan,emit} MSL-generate (int32 → in `hf_gen_msl`).** Cross-backend = STRICT
   ZERO-DIFFERING-PIXEL, NOT the float baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cloth_edges.png` added;
   the other 118 byte-identical (CL1 cloth_integrate + all existing untouched). `git diff master --stat --
   tests/golden` = ONLY `cloth_edges.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cloth-constraints` + `--cloth-edges-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; shaders int32-only, no int64_t). `scripts/verify.ps1`
   updated: `cloth_edges` golden in the Mac loop + `--cloth-edges-shot` in `$vkShots`. CL1 +
   `engine/sim/fpx.h` + `engine/physics/` UNTOUCHED.
