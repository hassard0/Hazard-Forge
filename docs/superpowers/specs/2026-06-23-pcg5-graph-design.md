# Slice PCG5 — Declarative layered pipeline + overlap-prune (Issue #22, flagship #22, 5th slice — THE DETERMINISM HEADLINE)

The strongest proof slice. Composes PCG2–PCG4 into one declarative `Generate(graph, seed)` call (the
integer-deterministic analog of a PCG graph), and adds the capstone **`PruneOverlaps`** stage: reject instances
whose footprint spheres interpenetrate, processed in a **fixed canonical order** (first-placed-wins, the
`fpx::SolveContacts` Gauss-Seidel order-determinism discipline) so a designer gets a Poisson-like minimum-spacing
guarantee deterministically. The headline: `Generate` is a **pure function of the seed**, so two netcode "peers"
produce a byte-identical world from the seed alone — replayable like every Hazard Forge sim. Strict int32 except
the prune's `FxLength` (int64, CPU-both → still byte-identical). **STRICT zero-diff cross-backend golden.**

## The addition — `engine/pcg/pcg.h` (APPEND-ONLY after the PCG4 block)
Add to `hf::pcg` (do NOT modify PCG1–PCG4):
- `struct PcgGraph {` — the declarative recipe:
  ```
  PcgArea      area;
  int          cellsX = 1, cellsZ = 1;     // ScatterStage
  bool         useMask = false;            // MaskStage (optional)
  PcgMask      mask;
  fx           density = kOne;
  PcgTransform transform;                  // TransformStage
  bool         prune = false;              // PruneStage (optional)
  fx           pruneRadius = 0;            // footprint sphere radius (scaled per-instance by instance.scale)
  ```
- `std::vector<PcgInstance> PruneOverlaps(const std::vector<PcgInstance>& in, fx baseRadius)`:
  - **Establish a CANONICAL order independent of input order** (the load-bearing bit): build an index array
    `0..in.size()-1` and **stable-sort it by a position-derived key** — primary `pos.z`, tie-break `pos.x`,
    final tie-break the original index (positions are unique per scatter cell, so the key is effectively total).
    Do NOT rely on the incoming vector order. This is what makes "shuffle the input → SAME survivors" hold.
  - Greedily walk the canonical order; KEEP instance `i` iff it does NOT overlap any ALREADY-KEPT instance `j`:
    overlap ⇔ `FxLength((pos_i - pos_j) with Y zeroed) < r_i + r_j`, where `r_i = fxmul(baseRadius, inst_i.scale)`
    (the per-instance footprint; `FxLength` is the int64 path, reuse `fpx.h`). First-placed in canonical order
    wins. Return the survivors **in canonical order** (deterministic output order).
  - `baseRadius <= 0` → no pair can overlap → all kept (a clean no-op).
- `std::vector<PcgInstance> Generate(const PcgGraph& g, const PcgStream& stream)`: compose the stages —
  (1) points = `g.useMask ? ScatterMasked(stream, g.area, g.cellsX, g.cellsZ, g.mask, g.density) : ScatterGrid(stream, g.area, g.cellsX, g.cellsZ)`;
  (2) instances = `BuildInstances(points, stream, g.transform)`;
  (3) `if (g.prune) instances = PruneOverlaps(instances, g.pruneRadius)`;
  return instances. `cellsX<=0 || cellsZ<=0` → empty (the empty-graph no-op). Pure function of `(g, stream)`.

## CPU test — extend `tests/pcg_test.cpp` (add a PCG5 section, keep PCG1–4 checks)
Assertions: (1) **no overlaps** — after a pruning `Generate`, EVERY surviving pair satisfies
`FxLength(pos_i - pos_j) >= r_i + r_j` (O(K²), K small — the make-or-break: prune actually removes overlaps);
(2) **prune removes some** — for a `pruneRadius` large enough that the dense scatter overlaps, `survivors <
instances_before_prune` (it did real work); (3) **shuffle-invariance** — take the pre-prune instances, **shuffle
them** (a fixed deterministic permutation, e.g. reverse or a fixed swap pattern — NO rng), run `PruneOverlaps` on
both the original and the shuffled vector, and assert the survivor SETS are byte-identical (same positions, same
canonical output order) — proving order-canonicality; (4) **lockstep / purity** — two independent `Generate(g,
stream)` calls are byte-equal (the "two peers from seed alone" headline); a different seed → different but valid
(same no-overlap property); (5) **no-op** — `g.prune=false` → `Generate` == `BuildInstances` over the same
scatter (PCG4), and `pruneRadius<=0` → all kept; (6) **empty** — `cellsX<=0` → empty. Print `pcg_test: ALL CHECKS PASSED`.

## Showcase — `--pcg5-graph-shot` (Vulkan, main.cpp) + `--pcg5-graph` (Metal, visual_test.mm)
A **2D top-down strict-integer plot of the full pipeline output**: a `PcgGraph` over the fixed area / 48×48 cells
/ seed, `useMask=true` (the PCG3 radial mask), `transform` = PCG4's random yaw + scale [0.5,1.5], `prune=true`
with a `pruneRadius` chosen so a visible fraction is removed (well-spaced, NONE overlapping). Draw each surviving
instance as PCG4's oriented+scaled cross-marker (reuse PCG4's `FxRotate` integer-marker code verbatim). The result
is a clean, **non-overlapping** scattered field — the money-shot of "deterministic declarative authoring". All
pixel math pure-integer. SAME graph/seed/image-size IN BOTH renderers → byte-identical cross-backend by construction.

## Proof (STRICT integer — zero-diff cross-backend)
```
pcg5-graph: layered pipeline scatter->mask->transform->prune (seed=<S>)
pcg5-graph: two-run BYTE-IDENTICAL
pcg5-graph: no surviving pair interpenetrates {survivors:<K>, overlaps:0}
pcg5-graph: prune is order-canonical: shuffled input -> SAME survivors {shuffledMatch:true}
pcg5-graph: lockstep: peerA == peerB from seed alone (byte-identical) {instances:<K>}
pcg5-graph: provenance {stages:4, beforePrune:<B>, survivors:<K>}
```
Assertions: (1) two runs byte-identical; (2) NO surviving pair interpenetrates (overlaps==0); (3) shuffled-input
prune gives the SAME survivors (order-canonical); (4) lockstep — two `Generate` calls byte-identical; (5)
provenance coherent (survivors < beforePrune, i.e. prune did work). Register `pcg5_graph` in verify.ps1 $Goldens
(Flag `--pcg5-graph`) + `--pcg5-graph-shot` in $vkShots, mirroring `pcg4_rules`.

## Constraints (HARD)
- APPEND-ONLY to engine/pcg/pcg.h (do NOT modify PCG1–4) + extend tests/pcg_test.cpp + the showcase blocks +
  verify.ps1 registration. Reuse fpx.h (incl. `FxLength`/`fxmul`)/particles.h/PCG1–4 READ-ONLY. Do NOT modify any
  other header/shader/golden. NO new RHI, NO new shader. Strict int32 except the prune's `FxLength` (int64,
  CPU-both → byte-identical). The canonical sort must be a STABLE integer sort by (pos.z, pos.x, origIndex) — no
  float comparisons.
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (controller bakes Metal, requires
  Metal image == Vulkan image byte-for-byte). The 2D plot MUST be pure integer (reuse PCG4's `FxRotate` marker +
  integer floor). Do NOT route through a GPU float raster.
- Branch `fix-issue-22-pcg5`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows + pcg_test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle pcg_test'`
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--pcg5-graph-shot` exits 0, the proof lines
  print, two-run byte-identical, overlaps==0, the shuffle-invariance holds, the lockstep equality holds, AND
  pcg_test passes. Commit message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the
  branch and STOP. Report: commit hash, proof output, image path, confirmation both renderers use identical graph/
  seed/size, that pcg_test passes, and that the plot is pure-integer.
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
