# Slice BD — Scene / Asset Streaming (Phase 4 #7) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A distinct
> open-world engine system: distance-based residency streaming with a per-frame load budget.

**Goal:** Stream scene content in/out by camera distance. A world larger than what's resident is divided
into cells; a streaming manager keeps cells within a load radius resident, unloads cells beyond a larger
unload radius (hysteresis), and processes loads/unloads under a per-frame BUDGET so content streams in
OVER frames (not instantly). Driven by a SCRIPTED camera path for determinism, a `--stream-shot`
showcase renders the resident set at a fixed capture frame and asserts the resident cell set — proving the
streaming policy is deterministic and only resident cells render.

## Why scripted camera + budget (determinism)

The resident set is a pure function of (camera position, load/unload radii, budget, prior state). With a
fixed SCRIPTED camera path (a `std::vector<Vec3>` of per-frame positions, no live input/RNG/clock) and a
fixed budget, the load/unload event sequence and the resident set at every frame are deterministic ⇒ the
captured frame and the asserted resident set are bit-stable. Same path + same build ⇒ identical.

## Design decisions (locked)

1. **Streaming manager (engine/scene/streaming.{h,cpp}, pure CPU, no RHI/backend symbols).** Namespace
   `hf::scene`. Reuses `engine/math` + the scene renderable types.
   - `struct Cell { int id; math::Vec3 center; /* content descriptor: a deterministic procedural set of
     renderables generated from id+center — e.g. a small cluster of cubes/spheres at fixed offsets */ };`
   - `enum class CellState { Unloaded, Loading, Resident, Unloading };`
   - `struct StreamConfig { float loadRadius; float unloadRadius /* > loadRadius, hysteresis */;
     int loadBudgetPerFrame; int unloadBudgetPerFrame; };`
   - `class StreamingWorld` — owns the full cell grid (a fixed deterministic NxN grid of cell centers)
     and each cell's state. `Update(const math::Vec3& cameraPos)`:
     1. compute desired residency: cells with `dist(center, cam) <= loadRadius` should be resident; cells
        with `dist > unloadRadius` should be unloaded (cells in the band keep their current state —
        hysteresis).
     2. enqueue to-load (Unloaded→Loading) and to-unload (Resident→Unloading) by ASCENDING distance
        (deterministic order; nearest loads first).
     3. process up to `loadBudgetPerFrame` loads (Loading→Resident, actually building the cell's
        renderables) and `unloadBudgetPerFrame` unloads (Unloading→Unloaded, freeing them) this frame.
     - Accessors: `ResidentCellIds()` (sorted), `ResidentRenderables()` (for the renderer),
       `Stats{resident, loading, unloading, totalCells}`.
   - All pure CPU + deterministic; the "load" is the synchronous construction of the cell's procedural
     renderables (the MVP models the BUDGET/over-frames behavior, not async disk I/O — async file
     streaming is a future slice; document this).

2. **Showcase `--stream-shot <out>` (Vulkan) / `--stream` (Metal).** A fixed NxN cell grid (e.g. 8x8 =
   64 cells, each a small procedural cluster). A SCRIPTED camera path flies across the grid over a fixed
   frame count, calling `Update(camPos)` each frame (so cells stream in/out under the budget). At a FIXED
   capture frame (documented — chosen so a clear subset is resident, with some still Loading and some
   already Unloaded behind the camera), render the RESIDENT cells (+ground + sky + standard light), and
   print `stream: {frame:F, resident:R, loading:L, unloading:U, total:64, residentIds:[...]}`. The
   resident-id list at the capture frame is deterministic and is the assertable state. New golden
   `tests/golden/metal/stream.png` (Metal two runs DIFF 0.0000).

3. **Render reuses the scene path.** Resident cells' renderables go through the normal lit+shadowed scene
   render (one draw per renderable, or reuse instancing if a cell is many identical meshes). No new RHI.
   The camera at the capture frame is the scripted position; deterministic.

4. **Tests `tests/streaming_test.cpp` (pure CPU, no GPU):**
   - **Radius residency:** with budget = infinity, after one Update a cell inside loadRadius is Resident,
     one beyond unloadRadius is Unloaded, one in the hysteresis band keeps its prior state (test both:
     was-resident stays resident, was-unloaded stays unloaded).
   - **Budget throttle:** with loadBudgetPerFrame=K and M>K cells newly in range, exactly K load per
     frame; after ceil(M/K) frames all are resident; loads happen NEAREST-first (assert order).
   - **Hysteresis:** a camera oscillating within the band does NOT thrash (no repeated load/unload of a
     band cell).
   - **Determinism:** running the scripted path twice yields identical per-frame ResidentCellIds.
   - Clean under `windows-msvc-asan`.

5. **Introspect.** Add exactly `scene-streaming` (features) + `--stream-shot` (showcases). One-pattern
   rebake, no other drift.

## RHI seam additions (summary)
- **None.** Streaming is pure CPU above the scene; rendering reuses the existing lit/shadowed path. New
  files (`engine/scene/streaming.{h,cpp}`, `tests/streaming_test.cpp`) add ZERO backend symbols. Seam
  grep stays at baseline (2).

## Out of scope (YAGNI)
Async/threaded disk I/O + real asset files (the loads are synchronous procedural construction this slice;
async streaming is a future slice), LOD/mesh-decimation by distance, occlusion-driven streaming, a
quadtree/octree spatial index (a flat grid + distance test is enough at this scale), persistent
cell serialization, texture/material streaming, seamless world origin rebasing. One flat cell grid,
distance residency + hysteresis + per-frame budget, scripted-camera golden + state-asserted.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 32) + new `streaming_test` (residency, budget
   throttle + nearest-first, hysteresis no-thrash, determinism). Clean under `windows-msvc-asan`.
2. `--stream-shot` on Windows/Vulkan: controller visual review — the resident subset of the cell grid
   renders (a partial world, near-camera cells present, far ones absent), lit + shadowed, coherent; the
   `stream: {...}` state line is deterministic. Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --stream` → new golden `tests/golden/metal/stream.png`; two runs DIFF 0.0000; the
   stream state line matches Vulkan.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `stream.png` added;
   the other 32 byte-identical.
5. Introspect JSON rebaked exactly `+scene-streaming` + `--stream-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `stream` image
   golden in the Mac round-trip loop.
