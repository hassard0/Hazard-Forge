# Slice NAV5 — Deterministic GPU Navmesh: INTEGER A* PATHFINDING (the headline) (Phase 12 #5) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #7
> (DETERMINISTIC GPU NAVMESH + PATHFINDING, `hf::nav`, header `engine/nav/navmesh.h`). Runs INTEGER A*
> over the NAV4 polygon adjacency graph to produce a path — and this is the flagship's HEADLINE: a
> fully integer cost graph + integer priority frontier → bit-exact, replayable, cross-platform-IDENTICAL
> pathfinding, which pairs with FPX5 lockstep and which UE5's float Detour cannot guarantee bit-for-bit
> across machines. Single-thread serial → strict zero-differing-pixel cross-backend (the integer bar).
> ZERO new RHI. Branch: `slice-nav5`. See [[hazard-forge-nav-roadmap]].

**Goal:** Extend `engine/nav/navmesh.h` with `FindPath` (integer A* over the NAV4 poly adjacency graph:
nodes = polys, edges = the per-edge adjacency, integer edge cost = squared/Manhattan distance between poly
centroids, an integer-keyed deterministic priority frontier, tie-break lowest poly id) → a poly CORRIDOR
(the sequence of poly ids start→goal). Add `shaders/nav_astar.comp.hlsl` (`[numthreads(1,1,1)]` serial —
the nav_region mirror; A* frontier relaxation is sequential), the `nav_path` integer golden (the corridor
drawn over the navmesh → strict zero-diff cross-backend), `--nav-path-shot` (Vulkan) / `--nav-path`
(Metal), and `tests/nav_test.cpp` additions. Reuse NAV1–NAV4 verbatim — NAV5 is additive (their pipelines
+ goldens stay byte-identical).

## Design call: INTEGER bit-exact A* (the headline — the strongest possible pathfinder proof)
A* is normally float (Euclidean cost + heuristic) → non-deterministic across machines. NAV5 is fully
INTEGER: edge cost = integer squared or Manhattan centroid distance (int32 for the bounded grid; int64
only if a squared distance could overflow — that one stage Vulkan-only + CPU-Metal, documented), the
heuristic is the same integer metric (admissible with Manhattan / consistent), and the priority frontier
is an INTEGER-keyed selection (a small deterministic binary heap OR a linear min-scan over the open set —
either is fine; the scan is simplest + obviously deterministic). EVERY tie is broken by LOWEST poly id
(the fpx/NAV3 source-order discipline) so the chosen path is unique. Run SINGLE-THREAD
(`[numthreads(1,1,1)]`, the nav_region mirror) → bit-exact CPU↔GPU and Vulkan==Metal by construction (the
integer bar, NOT the float visresolve bar). A LARGER/structural cross-vendor delta or any nonzero diff is
a bug. **Scope note (honest, grounded in NAV4):** NAV4 triangulates each region independently, so the
poly graph's connected components are per-region (no inter-region portals yet — a documented NAV-fidelity
gap). NAV5 picks its start+goal within the LARGEST connected component so A* produces a real multi-hop
corridor; cross-region portals are a future refinement (noted, not built here). The headline — a
deterministic, bit-identical, replayable integer path — holds fully on a single component.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The NAV4 input:** `engine/nav/navmesh.h` — `BuildPolyMesh` (the poly vertices + indices + per-edge
  ADJACENCY — the A* graph). NAV5 adds `FindPath` to this same header; NAV1–NAV4 functions UNCHANGED.
  Poly centroids are the integer average of the poly's contour-vertex coords (the cost-metric anchor).
- **The single-thread serial GPU mirror:** `shaders/nav_region.comp.hlsl` (NAV3's `[numthreads(1,1,1)]`
  one-thread serial) + `shaders/nav_polygonize.comp.hlsl` (NAV4) — copy the one-thread-runs-everything
  structure for the A* relaxation loop (init costs to INF, push start, repeat: pop the min-key open node,
  relax its neighbours, until the goal is popped or open is empty; reconstruct via the came-from array).
- **Integer determinism discipline:** the lowest-id tie-break + fixed scan order from NAV3 `BuildRegions`
  / `engine/sim/fpx.h` source-order. The squared-distance integer math from NAV4 `BuildPolyMesh` / NAV1
  `PointInTriXZ` (Cross2). `FxISqrt` only if a true Euclidean cost is ever needed (prefer squared/Manhattan
  — no sqrt).
- **The integer-golden showcase discipline:** NAV4's `--nav-polymesh-shot` / `RunNavPolymeshShowcase`
  (ReadBuffer the integer result, memcmp GPU==CPU, draw the navmesh + overlay, strict zero-diff) + the
  Bresenham `drawLine` used there for the corridor polyline.
- **RHI compute envelope (must fit, ZERO additions):** `engine/rhi/rhi.h` — the NAV1–NAV4 set.
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--nav-path-shot` standalone arg branch
  + DXC list), `metal_headless/{visual_test.mm,CMakeLists.txt}` (`RunNavPathShowcase` + `--nav-path` +
  `nav_astar` in `hf_gen_msl` — int32 → Metal-native, or excluded + CPU-ref if int64),
  `engine/editor/introspect.cpp` (+`deterministic-navmesh-pathfinding` feature + `--nav-path-shot`
  showcase) + `tests/introspect_test.cpp` + the JSON golden, `scripts/verify.ps1` (`nav_path` golden in
  the Mac loop + `--nav-path-shot` in `$vkShots`).

## Design decisions (locked)
1. **The A* graph.** Nodes = NAV4 polys; edges = NAV4 per-edge adjacency (a poly's neighbour list). Edge
   cost = the integer squared distance (or Manhattan — pick one, document) between the two polys' integer
   centroids. The heuristic h(poly) = the same integer metric from the poly's centroid to the goal's
   centroid (Manhattan → admissible+consistent; squared-Euclidean is NOT a metric for h, so if cost is
   squared, use Manhattan for h OR use Manhattan for both — keep it admissible so A* is optimal AND
   deterministic). Document the exact cost/heuristic choice.
2. **The frontier + determinism.** An integer open set with a deterministic min-selection: pop the node
   with the lowest f = g + h, tie-break lowest poly id. A linear min-scan over the open set is acceptable
   (the graph is small) and obviously deterministic; a binary heap is fine if the pop order is pinned to
   (f, polyId). `came_from[]` + `g[]` arrays; reconstruct the corridor by walking came_from from goal to
   start, then reverse. Pure integer, single-thread → bit-exact.
3. **Deterministic start + goal selection (so the golden is fixed).** Build the poly graph's connected
   components (by a deterministic flood over adjacency); pick the LARGEST component (tie → lowest min poly
   id); within it, start = the lowest poly id, goal = the poly with the maximum integer centroid distance
   from start (tie → lowest id). Pure deterministic. (This guarantees a real multi-hop corridor within
   NAV4's per-region components.)
4. **Showcase `--nav-path-shot <out>` (Vulkan) AND `--nav-path` (Metal — WIRE BOTH; confirm
   visual_test.mm + `#include "nav/navmesh.h"`).** Reuse the NAV1 scene → full pipeline → polymesh →
   FindPath. ReadBuffer the corridor (poly id sequence) + g-cost; **memcmp GPU == the CPU `FindPath`
   reference (the make-or-break)**; draw the navmesh (faint polys) + the corridor (the start→goal poly
   sequence as a bright polyline through poly centroids, start/goal marked) → `tests/golden/metal/
   nav_path.png` (baked on the Mac by the CONTROLLER — DO NOT commit).
5. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance / GPU==CPU (make-or-break):** the corridor poly-id sequence + total cost equal the
     CPU `FindPath` reference byte-for-byte. Print `nav-path: {polys:<P>, corridor:<L>, cost:<C>}
     GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs BYTE-IDENTICAL. Print `nav-path determinism: two runs BYTE-IDENTICAL`.
   - **(3) validity / coherence:** the corridor is a connected path (each consecutive pair is adjacent in
     the poly graph), starts at `start` and ends at `goal`, length ≥2. Print `nav-path coverage: corridor
     of <L> polys start->goal (each step adjacent)`.
   - **(4) no-path / empty:** a graph with the goal unreachable (or zero polys) → empty corridor / cleared
     background. Print `nav-path empty: no corridor (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/nav_path.png`; do NOT commit it — the CONTROLLER bakes
     on the Mac.** Existing 115 image goldens UNTOUCHED.
6. **Cross-backend bar (INTEGER, strict).** All A* state integer, ZERO GPU float → Vulkan==Metal
   BIT-IDENTICAL: the golden is the CPU-colored integer corridor read-back; the controller's cross-backend
   check is the STRICT ZERO-DIFFERING-PIXEL compare. Any nonzero cross-backend diff is a bug. (If a stage
   is int64 → Vulkan-only + Metal CPU-ref, byte-identical by construction.)
7. **Tests `tests/nav_test.cpp` additions (pure CPU):** a hand-built tiny poly graph (e.g. a 4-node line
   A-B-C-D) → A* returns A,B,C,D with the right cost; a graph with two routes → the LOWER-cost one (and
   tie → lowest-id); start==goal → single-node corridor; unreachable goal → empty; determinism (twice →
   identical); the corridor is adjacency-valid. Clean under `windows-msvc-asan`.
8. **Introspect.** Add exactly `deterministic-navmesh-pathfinding` (features) + `--nav-path-shot`
   (showcases). Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None expected.** Pure SSBO compute (the NAV1–NAV4 set). `rhi.h` + `rhi_factory` (baseline 2) + backend
  dirs UNCHANGED. NAV1–NAV4 shaders + `engine/sim/fpx.h` + `engine/physics/` UNCHANGED. Report the seam.

## Out of scope (YAGNI — later/future)
Inter-region PORTALS (NAV4 triangulates regions independently → the graph is per-region; portals are a
documented future fidelity refinement — NAV5 paths within the largest component, which fully demonstrates
the deterministic-pathfinding headline). String-pulling / funnel path smoothing (the corridor is the poly
sequence, not a smoothed line — a future refinement). Dynamic obstacle avoidance. The float render (NAV6).
NAV5 is ONLY the integer A* corridor + its bit-exact golden. No float (the corridor polyline viz is
integer-rasterized like NAV4's edges).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 93) + the new `nav_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--nav-path-shot` on Vulkan: GPU==CPU memcmp BIT-EXACT + determinism + validity
   (adjacency-connected corridor start→goal) + no-path no-op; a coherent navmesh+corridor image. Run under
   the Vulkan-validation gate → ZERO VUID (set BOTH `VK_LAYER_PATH` to the conan `...\.conan2\p\...\layers`
   dir AND `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer LOADED = zero "not found").
3. Metal: `visual_test --nav-path` → new golden `tests/golden/metal/nav_path.png`; two runs DIFF 0.0000
   (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm nav_astar MSL-generates
   (int32 → in `hf_gen_msl`) OR is excluded + CPU-ref'd if int64.** Cross-backend = STRICT
   ZERO-DIFFERING-PIXEL, NOT the float baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `nav_path.png` added; the
   other 115 byte-identical (NAV1–NAV4 + all existing untouched). `git diff master --stat -- tests/golden`
   = ONLY `nav_path.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-navmesh-pathfinding` + `--nav-path-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; report int32-native vs any int64 Vulkan-only stage).
   `scripts/verify.ps1` updated: `nav_path` golden in the Mac loop + `--nav-path-shot` in `$vkShots`.
   NAV1–NAV4 + `engine/sim/fpx.h` + `engine/physics/` UNTOUCHED.
