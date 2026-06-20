# Slice BD4 — Deterministic GPU Crowds: PATH-FOLLOWING THE A* CORRIDOR (THE NAV BRIDGE) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #18 (DETERMINISTIC GPU
> CROWDS, `hf::sim::boids`). BD1-BD3 built a free-flocking crowd (steering → neighbors → the 3 Reynolds rules).
> BD4 is the COMPOSITION HEADLINE — "crowds, not just boids": the flock **follows the bit-exact A\* corridor** from
> the NAV navmesh flagship (#7). Each agent steers along the corridor toward the goal (a SteerArrive-to-waypoint
> term) BLENDED with the BD3 flock, so the swarm streams along the navmesh path while still cohering/aligning/
> spacing. This pairs the engine's two integer pillars — nav A* + the crowd sim — both deterministic. INTEGER-bit-
> exact. The path term extends the BD3 `boids_flock.comp` RENDER-INVARIANTLY (the AC2 discipline); NAV is reused
> read-only (byte-frozen). **NO new RHI.** Branch: `slice-bd4`. See [[hazard-forge-boids-roadmap]].

**Goal:** Extend `engine/sim/boids.h` (additive — BD1-BD3 byte-frozen) with `BoidsPath` (the integer corridor
waypoints) + `SteerPath` (the per-agent corridor-follow steering: find the nearest-ahead waypoint, arrive toward
it) + `StepFlockPath` (= BD3 `StepFlock` + the path term). Extend `shaders/boids_flock.comp.hlsl` with the
path-seek term (gated by a `pathCount` uniform — `pathCount==0` → byte-identical to BD3, the render-invariance
contract). Add `--boids-path-shot` (Vulkan) / `--boids-path` (Metal). Bake the integer golden `boids_path`. **NO
new RHI, NO new shader file. BD3 `boids_flock.png` stays byte-identical** (path off → exactly BD3).

## Design call: agents arrive-follow the integer A* corridor, blended with the BD3 flock
The NAV flagship's `nav::FindPath` (navmesh.h:1101) is a deterministic INTEGER A\* that returns a CORRIDOR (a
poly-id sequence start→goal). Its waypoints are the integer poly centroids (the `cx`/`cy` arrays the navmesh
stores) → a `BoidsPath` of Q16.16 world waypoints. The crowd follows it:

- **`BoidsPath { std::vector<FxVec3> waypoints; }`** — the corridor as Q16.16 world points (built HOST-side from
  `nav::FindPath`'s corridor + the poly centroids; the nav A* is bit-exact, the corridor is fixed for the run; the
  waypoint Q16.16 conversion is a host integer snap — the corridor poly centers are already integers).
- **`SteerPath(agent, path, cfg)`** — the corridor-follow steering (deterministic, stateless — recomputed each
  tick from the agent's position): find the index `k` of the NEAREST waypoint (an L1/squared integer distance
  loop over the corridor — the corridor is short), `target = waypoints[min(k+1, last)]` (the next waypoint ahead —
  corridor progress), `pathForce = FxScale(FxSub(target, agent.pos), pathGain)` (the un-normalized arrive/seek, the
  BD1 SteerSeek shape; near the FINAL waypoint it naturally eases in). Returns the path force.
- **`StepFlockPath(agents, cfg, path, dt)`** = the BD3 `StepFlock` body with the path term added to the per-agent
  force: `force = AccumFlock(i) + SteerPath(i) + optional seek + gravity` → clamp → integrate (JACOBI, frozen
  snapshot). When `path.waypoints` is empty, `SteerPath` returns 0 → exactly BD3 `StepFlock` (the equivalence/
  render-invariance contract).

**The shader (render-invariant extension, the AC2 discipline):** `boids_flock.comp.hlsl` gains a `pathCount`
uniform + a `gWaypoints` storage buffer; when `pathCount > 0` it runs the `SteerPath` body (the nearest-waypoint
loop + the arrive force) and adds it to the flock force; when `pathCount == 0` it skips it entirely → the BD3 ops
byte-for-byte (BD3's `boids_flock` golden stays byte-identical). The path force is int64 (the `fxmul` arrive) —
`boids_flock.comp` is ALREADY int64-Vulkan-only, so no MSL-native change; Metal runs the CPU `StepFlockPath`.
**The implementer MUST keep the `pathCount==0` path byte-identical to BD3** (the controller re-verifies BD3's
golden).

`MeasureFlockPath(agents, cfg, path)` → the mean L1 distance from the flock centroid to the FINAL waypoint (the
"they reached the goal" stat — drops as the crowd streams along the corridor) + the BD3 flock stats — deterministic.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **BD1-BD3 (this branch's `boids.h`, read-only — build on, DON'T modify):** `Agent`, `FlockConfig`, `AccumFlock`,
  `StepFlock` (the body BD4 mirrors with the path term), `ClampAxisVec`, `BoidsNeighborList` + the BD2 engine. DO
  NOT modify the BD1-BD3 functions. BD4 APPENDS `BoidsPath`/`SteerPath`/`StepFlockPath` + extends `boids_flock.
  comp` render-invariantly.
- **The NAV A* corridor (`engine/nav/navmesh.h`, read-only — reuse, byte-FROZEN):** `FindPath` (navmesh.h:1101 —
  the integer A*, returns the corridor poly-id sequence), the poly centroid arrays (`cx`/`cy`, the waypoint
  coords), `PathToWorldPolyline` (navmesh.h:1262 — the corridor→world polyline, for the BD4 render of the path
  underneath), the navmesh build (`Poly`/the span-rasterize→region→polygonize from the `--nav-*-shot` showcase).
  **DO NOT modify navmesh.h** — BD4 builds a navmesh + `FindPath` HOST-side, converts the corridor to `BoidsPath`.
- **The NAV showcase (`samples/hello_triangle/main.cpp` `--nav-path-shot`):** how a navmesh is built + `FindPath`
  is called + the corridor rendered. REUSE the navmesh-build + FindPath for the BD4 scene. The BD3 `--boids-flock-
  shot` for the flock + the GPU driver. Standalone arg-parse (the FR1 C1061 lesson).
- **fpx integer ops (`engine/sim/fpx.h`, read-only):** `FxVec3`/`FxAdd`/`FxSub`/`FxScale`/`fxmul`, the integer
  distance compare. **DO NOT modify fpx.h.**
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), `tests/boids_test.cpp`.

## Design decisions (locked)
1. **boids.h additions:** `BoidsPath`, `SteerPath` (nearest-waypoint arrive, stateless), `StepFlockPath` (BD3
   StepFlock + the path term, JACOBI), `StepFlockPathSteps`, `MeasureFlockPath`. Pure integer, fixed order.
   **EXTEND `boids_flock.comp.hlsl`** with the `pathCount` uniform + `gWaypoints` buffer + the SteerPath term
   (pathCount==0 → BD3 byte-identical). Still int64-Vulkan-only.
2. **Showcase `--boids-path-shot <out>` (Vulkan) AND `--boids-path` (Metal) — WIRE BOTH** (standalone arg-parse).
   The SCENE: build a navmesh (the `--nav-path-shot` navmesh REUSED or a simple one) + `nav::FindPath` a corridor
   start→goal → `BoidsPath`; a flock (~256 agents) spawned at the start; settle K `StepFlockPath` steps → the
   crowd STREAMS along the corridor to the goal while flocking (cohering/aligning/spacing). Vulkan: per tick the
   GPU `boids_cell_*`/`boids_neighbor_*` rebuild + the extended `boids_flock.comp` (path on) step → **memcmp the GPU
   agent array vs the CPU `StepFlockPath`** (NO tolerance). Metal: the CPU reference. Render a PURE-INTEGER 2D
   top-down view (the navmesh + the A* corridor polyline faint underneath [PathToWorldPolyline], the agents as dots
   streaming along it). Golden = `tests/golden/metal/boids_path.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU agent array after K steps == the CPU `StepFlockPath` byte-for-byte. Print
     `boids-path: {agents:<N>, waypoints:<W>, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `boids-path determinism: two runs BYTE-IDENTICAL`.
   - **(3) reached the goal + flocked:** the flock centroid's distance to the FINAL waypoint DROPPED from the start
     (the crowd followed the corridor to the goal) AND it stayed a flock (min-separation above a floor). Print
     `boids-path followed: {startToGoal:<D0>, endToGoal:<D1>, reached:true, flocked:true}` with `D1 < D0` asserted.
   - **(4) BD3 equivalence (the render-invariance proof):** an EMPTY-path `StepFlockPath` run == the BD3 `StepFlock`
     byte-for-byte (path off → exactly BD3). Print `boids-path equiv: {emptyPath==BD3:true}`.
   - **Golden discipline: ONLY `tests/golden/metal/boids_path.png`; do NOT commit it. BD3 `boids_flock.png` (+ BD1/
     BD2) byte-identical.** Existing 180 image goldens UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests `tests/boids_test.cpp` additions (pure CPU):** `SteerPath` steers toward the next corridor waypoint
   (the force points along the corridor), empty path → 0; `StepFlockPath` with a corridor moves the flock centroid
   toward the goal while keeping min-separation; an EMPTY-path `StepFlockPath` == BD3 `StepFlock` byte-identical;
   two runs byte-identical. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-boids-path` (features) + `--boids-path-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the BD3 compute + SSBO + dispatch + read-back path + a `gWaypoints` storage buffer (the existing
  Storage-buffer surface). `rhi.h` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h`
  + `joint.h` + `couple*.h` + `fract.h` + `vehicle.h` + `active.h` + **`engine/nav/` (reused read-only)** + `engine/
  anim/` + `engine/physics/` + the BD2 grid shaders UNCHANGED. The ONLY shader CHANGE is the render-invariant
  extension of `boids_flock.comp.hlsl` (pathCount==0 → BD3 byte-identical; NO new shader file). BD1-BD3 `boids.h`
  code UNCHANGED (BD4 additive — only `BoidsPath`/`SteerPath`/`StepFlockPath`). Report the seam empty (only `boids.h`
  extended + `boids_flock.comp` render-invariantly extended + showcase/test/introspect).

## Out of scope (YAGNI — BD5/BD6 only remain)
Lockstep (BD5), the lit 3D render (BD6 — BD4's render is the 2D top-down diagnostic). Per-agent INDEPENDENT paths
(BD4 follows ONE shared corridor — a crowd to a common goal), dynamic re-pathing, RVO/ORCA, funnel/string-pulling
the corridor (the agents arrive-follow the poly-centroid waypoints — a deterministic proxy, not a smoothed path).
BD4 claims ONLY: a deterministic crowd that follows the bit-exact A* corridor to a goal while flocking, bit-
identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs. NOTE (honest): the corridor is the
poly-centroid waypoint sequence (the NAV A* output), arrive-followed un-normalized (the BD1 seek shape); the
"nearest-ahead waypoint" is a stateless deterministic proxy for corridor progress (no per-agent path-state).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 104 + the new `boids_test` path cases). Clean under
   `windows-msvc-asan` (build+run `boids_test` + `introspect_test`).
2. **proofs + visual:** `--boids-path-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the flock streaming along the A* corridor toward the goal (the agents follow the path polyline, a coherent
   crowd — not scattered, not stuck).** **CRITICAL: re-run `--boids-flock-shot` → STILL bit-exact AND its golden
   byte-identical (BD3 render-invariance — pathCount==0 == BD3).** Re-run `--boids-steer/neighbors-shot` → bit-exact.
3. Metal: `visual_test --boids-path` → new golden `tests/golden/metal/boids_path.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `boids_flock.comp` still NOT in hf_gen_msl + the BD2 grid passes
   STILL in; confirm `tests/golden/metal/boids_flock.png` UNCHANGED in the diff.** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `boids_path.png` added (NOT
   `boids_flock.png`/`boids_neighbors.png`/`boids_steer.png`); the other 180 byte-identical. `git diff master
   --stat -- tests/golden` = ONLY `boids_path.png` (metal) + the introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-boids-path` + `--boids-path-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + the frozen sim/nav/anim/physics headers + the BD2 grid shaders byte-unchanged; ONLY
   `boids.h` extended additively + `boids_flock.comp` render-invariantly extended + showcase/test/introspect).
   `scripts/verify.ps1` updated: `boids_path` golden + `--boids-path-shot` in `$vkShots`. **NO new shader file;
   `boids_flock.comp` stays NOT in `hf_gen_msl`; the BD3 `boids_flock` golden byte-identical.**
