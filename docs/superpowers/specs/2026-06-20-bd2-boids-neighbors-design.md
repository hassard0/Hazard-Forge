# Slice BD2 — Deterministic GPU Crowds: THE GRID-HASH NEIGHBOR LIST (THE REUSE SPINE) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #18 (DETERMINISTIC GPU
> CROWDS, `hf::sim::boids`). BD1 built the steering primitive with brute-force O(N²) separation. BD2 adds the
> SCALABLE neighbor engine — a uniform spatial-hash grid + a `count→scan→emit` CSR cell table + a per-agent
> neighbor list — so a flock of thousands queries its neighbors in O(N). This is `grain.h`'s grid engine applied
> to agents, **byte-for-byte cloned**. INTEGER-bit-exact. **MSL-NATIVE int32** (pure index arithmetic — the
> FPX2/GR2/NAV2 property: a TRUE GPU pass on BOTH backends = the strongest cross-vendor proof, NOT the int64
> Vulkan-only split). **NO new RHI.** Branch: `slice-bd2`. See [[hazard-forge-boids-roadmap]].

**Goal:** Extend `engine/sim/boids.h` (additive — BD1 byte-frozen) with `BoidsGrid` + `BoidsCellOf`/`FlatBoidsCellId`/
`BoidsCellCount`/`MakeBoidsGrid` + `BoidsCellTable` + `BuildBoidsCellTable` (count→scan→emit, the grain.h twin) +
`BoidsNeighborList` + `BuildBoidsNeighborList` (per-agent neighbors within the perception radius via the grid).
Add the new int32 shaders `boids_cell_count/scan/emit.comp.hlsl` + `boids_neighbor_count/scan/emit.comp.hlsl` +
`--boids-neighbors-shot` (Vulkan) / `--boids-neighbors` (Metal). Bake the integer golden `boids_neighbors`. **NO
new RHI.**

## Design call: clone grain.h's grid engine for agents — count→scan→emit, MSL-native int32
`engine/sim/grain.h` (grep-confirmed) already has the exact engine: `GrainGrid`/`GrainCellOf` (FloorDiv per axis,
int32, grain.h:238/245) / `FlatGrainCellId` (:252) / `GrainCellCount` (:258) / `MakeGrid` (:264, the AABB→grid) /
`GrainCellTable{cellStart (CSR prefix-sum), cellGrains}` (:296) / `BuildGrainCellTable` (count→scan→emit, :301).
The shaders `grain_cell_count/scan/emit.comp` + `grain_neighbor_count/scan/emit.comp` are **int32, IN `hf_gen_msl`**
(a true GPU pass both backends — the strict zero-differing-pixel bar). BD2 mirrors ALL of this for `boids::Agent`:

- **`BoidsGrid` + `MakeBoidsGrid(agents, cellSize)`** (the `GrainGrid`/`MakeGrid` twin): the agent-AABB → a bounded
  dense grid, `cellSize` = the perception radius (BD3's flock radius / BD1's `sepRadius`). `BoidsCellOf(pos,
  cellSize)` = `FloorDiv` per axis (int32). `FlatBoidsCellId`/`BoidsCellCount`.
- **`BoidsCellTable{cellStart (cellCount+1 CSR prefix-sum), cellAgents (agent indices grouped by cell, ASCENDING
  index within a cell)}`** + **`BuildBoidsCellTable(agents, grid)`** = count→scan→emit (the `BuildGrainCellTable`
  twin, deterministic, int32).
- **`BoidsNeighborList{neighborStart (CSR), neighbors (agent indices)}`** + **`BuildBoidsNeighborList(agents, grid,
  cellTable, radius)`** = for each agent, gather candidates from the 3×3×3 cell stencil around its cell, accept
  those within `radius` (the `dist² < radius²` reject; the squared compare is int64 but it's a COMPARE only — the
  emitted list is int32 indices, and the cell/scan/emit passes are pure int32 → MSL-native). This is the
  `BuildNeighborList` (grain.h:225, the fluid.h twin) applied to agents.

**MSL-NATIVE (the strongest proof):** the cell-table + neighbor-list passes are pure index arithmetic + `FloorDiv`
+ `InterlockedAdd` + an int64 squared-distance COMPARE (no int64 RESULT stored — like GR2/NAV2). So
`boids_cell_*.comp` + `boids_neighbor_*.comp` MSL-generate NATIVELY (**IN `hf_gen_msl`**) — a TRUE GPU pass on
BOTH backends, the strongest cross-vendor proof (unlike BD1's int64-Vulkan-only integrate). **The implementer MUST
confirm the squared-distance compare stays a compare (no stored int64) and keep the shaders int32 / in
hf_gen_msl** (the GR2/NAV2 precedent — if the radius compare needs an int64 type that glslc rejects, scope the
positions so dx² fits int32, or follow grain_neighbor_*'s exact approach which is already MSL-native).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **BD1 (this branch's `boids.h`, read-only — build on, DON'T modify):** `Agent`, `BoidsConfig`, `FxVec3` usings.
  DO NOT modify the BD1 functions. BD2 APPENDS.
- **grain.h grid engine to CLONE (`engine/sim/grain.h`, read-only — mirror, do NOT modify):** `GrainGrid` (:238),
  `GrainCellOf` (:245), `FlatGrainCellId` (:252), `GrainCellCount` (:258), `MakeGrid` (:264), `GrainCellTable`
  (:296), `BuildGrainCellTable` (:301, the count→scan→emit), and the neighbor-list builder (`BuildNeighborList`,
  grain.h:225 / `engine/sim/fluid.h`). `BoidsGrid`/`BuildBoidsCellTable`/`BuildBoidsNeighborList` are these with
  `Agent`+`Agent.pos`. **DO NOT modify grain.h/fluid.h.**
- **fpx integer cell math (`engine/sim/fpx.h`, read-only):** `FloorDiv` (:177), `FxCell` (:183), `CellId` (:196).
  **DO NOT modify fpx.h.**
- **The int32-MSL-native shaders to CLONE (`shaders/`):** `grain_cell_count/scan/emit.comp.hlsl` →
  `boids_cell_*.comp.hlsl`; `grain_neighbor_count/scan/emit.comp.hlsl` → `boids_neighbor_*.comp.hlsl`. These are
  IN `hf_gen_msl` (int32, MSL-native) — `boids_*` follow the SAME registration (DXC SPIR-V + hf_gen_msl). Confirm
  `boids_cell`/`boids_neighbor` ARE added to hf_gen_msl (the opposite of BD1's boids_steer).
- **The showcase precedent:** `--grain-neighbors-shot` (the GPU cell-table + neighbor-list build + the GPU==CPU
  memcmp + the neighbor-heat 2D render) in `samples/hello_triangle/main.cpp` — mirror for `--boids-neighbors-shot`.
  Standalone arg-parse (the FR1 C1061 lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden — do NOT**), `tests/boids_test.cpp`.

## Design decisions (locked)
1. **boids.h additions** (above): `BoidsGrid`, `MakeBoidsGrid`, `BoidsCellOf`/`FlatBoidsCellId`/`BoidsCellCount`,
   `BoidsCellTable`, `BuildBoidsCellTable`, `BoidsNeighborList`, `BuildBoidsNeighborList`. Pure integer, fixed
   order (ascending index within a cell — the determinism crux). **NEW int32 shaders** `boids_cell_count/scan/
   emit.comp.hlsl` + `boids_neighbor_count/scan/emit.comp.hlsl` (the grain_* clones, IN hf_gen_msl).
2. **Showcase `--boids-neighbors-shot <out>` (Vulkan) AND `--boids-neighbors` (Metal) — WIRE BOTH** (standalone
   arg-parse). The SCENE: a larger flock (~256-1024 agents) spread over an area; build the grid + cell table +
   neighbor list. Vulkan: the GPU `boids_cell_*` + `boids_neighbor_*` dispatches → **memcmp the GPU cellStart +
   cellAgents + neighborStart + neighbors vs the CPU `BuildBoidsCellTable` + `BuildBoidsNeighborList`** (NO
   tolerance). Metal: the GPU passes (MSL-native — a true GPU pass). Render a PURE-INTEGER 2D top-down neighbor-
   heat view (each agent a dot colored by its neighbor count — dense cluster hot, sparse edge cold). Golden =
   `tests/golden/metal/boids_neighbors.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU cell table + neighbor list == the CPU reference byte-for-byte. Print
     `boids-neighbors: {agents:<N>, cells:<C>, neighbors:<M>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `boids-neighbors determinism: two runs BYTE-IDENTICAL`.
   - **(3) every neighbor within radius + complete:** every emitted neighbor pair is within `radius` (the grid
     didn't miss/over-include) AND the count matches a brute-force O(N²) reference (the grid found ALL true
     neighbors). Print `boids-neighbors correct: {withinRadius:true, matchesBruteForce:true}`; assert both.
   - **(4) partition complete:** the cell table buckets every agent exactly once (`cellAgents.size() == N`,
     `cellStart` monotone, last == N). Print `boids-neighbors partition: {complete:true}`.
   - **Golden discipline: ONLY `tests/golden/metal/boids_neighbors.png`; do NOT commit it. BD1 `boids_steer.png`
     byte-identical.** Existing 178 image goldens UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict, MSL-NATIVE):** Vulkan GPU == Metal GPU == golden, ZERO differing pixels
   (a TRUE GPU pass on both backends — the strongest proof).
5. **Tests `tests/boids_test.cpp` additions (pure CPU):** `BuildBoidsCellTable` partitions every agent once
   (CSR monotone, last == N); `BuildBoidsNeighborList` — every emitted neighbor within radius AND matches a
   brute-force O(N²) reference (no misses, no extras); two runs byte-identical. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-boids-neighbors` (features) + `--boids-neighbors-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the grain neighbor surface — Storage
  buffers + InterlockedAdd + ComputeToCompute barriers + ReadBuffer). `rhi.h` + backend dirs UNCHANGED. `engine/
  sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` + `vehicle.h` + `active.h` +
  `engine/nav/` + `engine/anim/` + `engine/physics/` + all EXISTING shaders (incl `boids_steer.comp`) UNCHANGED.
  The new shaders are `boids_cell_*.comp` + `boids_neighbor_*.comp` (int32, IN hf_gen_msl). BD1 `boids.h` code
  UNCHANGED (BD2 additive). Report the seam empty (only `boids.h` extended + the 6 new shaders + showcase/test/
  introspect).

## Out of scope (YAGNI — later BD slices)
The full flock step using the neighbor list (BD3 — BD2 only BUILDS the list, proves it correct; BD3 consumes it
for sep+align+cohesion), path-following (BD4), lockstep (BD5), the lit 3D render (BD6 — BD2's render is the 2D
neighbor-heat diagnostic). Dynamic grid resize per tick / multi-level grids. BD2 claims ONLY: a deterministic
grid-hash neighbor engine (a count→scan→emit CSR cell table + a per-agent radius neighbor list) that finds every
true neighbor with no extras, bit-identical CPU↔Vulkan↔Metal (MSL-native, a true GPU pass both backends), with the
integer golden + the four proofs. NOTE (honest): the grid is a uniform fixed-cell hash sized to the perception
radius (the GR2/FL2 precedent), not an adaptive/hierarchical structure.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 104 + the new `boids_test` neighbor cases). Clean under
   `windows-msvc-asan` (build+run `boids_test` + `introspect_test`).
2. **proofs + visual:** `--boids-neighbors-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate
   → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image
   shows a coherent neighbor-heat field (dense clusters hot, sparse edges cold — not uniform, not scrambled).**
   Re-run `--boids-steer-shot` → still bit-exact (BD1 render-invariance).
3. Metal: `visual_test --boids-neighbors` → new golden `tests/golden/metal/boids_neighbors.png`; two runs DIFF
   0.0000. **Confirm `visual_test.mm` in the diff; confirm `boids_cell_*`/`boids_neighbor_*` ARE in `hf_gen_msl`
   (int32, a TRUE Metal GPU pass — the strongest proof, unlike BD1).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `boids_neighbors.png` added (NOT
   `boids_steer.png`); the other 178 byte-identical. `git diff master --stat -- tests/golden` = ONLY
   `boids_neighbors.png` (metal) + the introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-boids-neighbors` + `--boids-neighbors-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + the frozen sim/nav/anim/physics headers + ALL existing shaders incl `boids_steer.
   comp` byte-unchanged; ONLY `boids.h` extended additively + the 6 new shaders + showcase/test/introspect).
   `scripts/verify.ps1` updated: `boids_neighbors` golden + `--boids-neighbors-shot` in `$vkShots`. **The new
   shaders ARE in `hf_gen_msl` (int32, MSL-native).**
