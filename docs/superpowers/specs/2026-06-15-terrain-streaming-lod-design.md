# Slice BJ — Terrain-Streaming LOD Integration (Phase 4 #12) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The open-world
> capstone: ties BD (streaming) + BF (terrain) into streamed, LOD-selected terrain tiles.

**Goal:** A large terrain made of tiles streamed in/out by camera distance, each meshed at a level of
detail chosen by distance band. Reuses the BD streaming residency policy and the BF heightmap mesh gen;
adjacent tiles sample the SAME global `Height(x,z)` so they are seamless. A scripted-camera
`--terrain-stream-shot` renders the resident tiles at their LODs at a fixed frame and asserts the resident
tile set + their LOD levels — proving deterministic streamed-LOD terrain.

## Why this is verifiable (deterministic streamed LOD)

The resident tile set + each tile's LOD are pure functions of (camera position, load/unload radii,
budget, distance→LOD bands). With a fixed SCRIPTED camera path (no input/RNG/clock) and fixed bands, the
per-frame resident set AND LOD assignment are deterministic ⇒ the captured frame and the asserted state
are bit-stable. Tiles share the global height function ⇒ no per-tile randomness; LOD is a discrete band
function ⇒ no float-threshold flicker (use `<=` bands with hysteresis on the band like BD's residency
hysteresis to avoid LOD thrash).

## Design decisions (locked)

1. **Terrain tiles via the BD streaming world + BF heightmap (engine/terrain/terrain_stream.{h,cpp}, pure
   CPU, no backend symbols).** Namespace `hf::terrain`. Reuses `engine/scene/streaming` (residency policy)
   + `engine/terrain/heightmap` (`Height`, `BuildTerrain`).
   - A `TileGrid`: a fixed `T x T` grid of terrain tiles over the world; tile `(i,j)` covers a
     `tileSize x tileSize` XZ region at a fixed world offset. Each tile's mesh is `BuildTerrain` over THAT
     tile's region using the GLOBAL `Height(x,z)` (so tile edges match neighbors exactly — seamless;
     document that BuildTerrain must accept a world-space origin/extent, OR add an overload
     `BuildTerrainРegionTile(worldMinX, worldMinZ, tileSize, n)` that samples global Height — add the
     overload if BuildTerrain is centered-only).
   - **LOD by distance band:** `int LodFor(float dist)` → e.g. `dist<bandNear → LOD0 (n=96)`,
     `<bandMid → LOD1 (n=48)`, else `LOD2 (n=24)`. The tile mesh resolution `n` is the LOD's. Document the
     bands + the per-tile mesh vertex/index counts per LOD. Hysteresis on the LOD band (don't downgrade
     until a bit past the band edge) to avoid thrash — deterministic with the scripted camera.
   - `TerrainStreamWorld`: owns the tile grid + a BD-style residency manager (load/unload by radius +
     per-frame budget). `Update(cameraPos)`: residency (which tiles resident) + LOD selection (each
     resident tile's LOD from its center distance). Accessors: `ResidentTiles()` → list of
     `{tileId(i,j), lod, mesh}` sorted; `Stats{resident, loading, byLod[3], total}`.

2. **Render reuses the lit/shadowed scene path.** Each resident tile's mesh (at its LOD) is drawn lit +
   shadowed with the BF height-tint material (or the terrain material). NO new RHI, NO new shader. Fixed
   deterministic camera.

3. **Showcase `--terrain-stream-shot <out>` (Vulkan) / `--terrain-stream` (Metal).** A fixed `T x T`
   (e.g. 6x6=36) tile grid. A SCRIPTED camera path flies across it; at a FIXED capture frame (documented —
   chosen so a clear mix of LODs is visible: nearby high-LOD tiles + distant low-LOD + some unloaded),
   render the resident tiles at their LODs. Print `terrain-stream: {frame:F, resident:R, lod0:a, lod1:b,
   lod2:c, total:36, tiles:[(i,j):lod ...]}` — the resident-tile-with-LOD list is the deterministic
   assertable state. New golden `tests/golden/metal/terrain_stream.png` (Metal two runs DIFF 0.0000).
   Existing 36 image goldens UNTOUCHED.

4. **Seamlessness.** Because every tile samples the SAME global `Height(x,z)` at its shared edge
   coordinates, adjacent tiles meet exactly at LOD0. At an LOD SEAM (a high-LOD tile next to a low-LOD
   tile) there can be a crack — for the MVP, accept minor LOD-seam cracks (document; skirts/stitching is a
   future slice) OR pick capture framing where the visible seam is minimal. The golden just needs to be
   deterministic + the relief coherent; do NOT over-engineer seam stitching.

5. **Tests `tests/terrain_stream_test.cpp` (pure CPU, no GPU):**
   - **Tile coverage:** tile `(i,j)` covers the expected world region; a vertex on the shared edge of two
     adjacent tiles has the SAME world position + height (seamlessness at equal LOD).
   - **LOD selection:** `LodFor` returns the right band for sample distances; hysteresis prevents
     downgrade within the band after an upgrade.
   - **Residency + LOD determinism:** running the scripted path twice → identical per-frame
     `{ResidentTiles + their LODs}`.
   - **Mesh counts per LOD:** each LOD's tile mesh has the documented vertex/index counts.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `terrain-streaming-lod` (features) + `--terrain-stream-shot` (showcases).
   One-pattern rebake.

## RHI seam additions (summary)
- **None.** Pure CPU (residency + tile mesh gen) above the scene; rendering reuses the lit/shadowed path.
  New files add ZERO backend symbols. Seam grep stays at baseline (2). (If `BuildTerrain` needs a
  world-origin overload, add it in `engine/terrain/heightmap.h` — still pure CPU, no backend symbols, and
  verify the BF `--terrain-shot` golden stays byte-identical since the existing entry point is unchanged.)

## Out of scope (YAGNI)
LOD-seam skirts/stitching/geomorphing, GPU tessellation, quadtree/clipmap LOD, frustum culling of tiles
(BD residency is distance-only), async tile I/O, terrain materials beyond the BF tint, collision,
vegetation. One T×T tile grid, distance-banded discrete LOD, distance residency + budget, scripted-camera
golden + state-asserted.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 36) + new `terrain_stream_test` (tile coverage +
   seam, LOD selection + hysteresis, residency+LOD determinism, per-LOD mesh counts). Clean under
   `windows-msvc-asan`.
2. **BF terrain golden unchanged:** if you added a `BuildTerrain` world-origin overload, the existing
   `--terrain-shot` path is byte-identical → `terrain.png` stays byte-identical (verify on the M4).
3. `--terrain-stream-shot` on Windows/Vulkan: controller visual review — a streamed terrain field with
   visibly varying tile density (near tiles denser, far coarser), lit + shadowed, coherent; the
   `terrain-stream: {...}` state line is deterministic (two runs → byte-identical capture). Run under the
   AT Vulkan-validation gate → ZERO errors.
4. Metal: `visual_test --terrain-stream` → new golden `tests/golden/metal/terrain_stream.png`; two runs
   DIFF 0.0000; the state line matches Vulkan.
5. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `terrain_stream.png`
   added; the other 36 byte-identical (incl. `terrain.png`).
6. Introspect JSON rebaked exactly `+terrain-streaming-lod` + `--terrain-stream-shot`; introspect test
   updated; no other drift.
7. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new
   `terrain_stream` image golden in the Mac round-trip loop.
