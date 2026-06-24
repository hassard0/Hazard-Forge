# Slice PT5 — Foliage-on-terrain composition (Flagship #26 TERRAIN, 5th slice — composes flagship #25)

The composition that ties the procedural stack together: the deterministic PCG-scattered, wind-swept FOLIAGE meadow
(flagship #25) is **seated on the eroded terrain surface** by sampling the integer eroded heightfield at each
plant's XZ. So a seed → a byte-identical eroded valley WITH a byte-identical wind-swept meadow growing on it. FLOAT
visresolve-bar (the seating math is integer + provenance-checked; the render is float). NO new shader, NO new RHI
(reuse PT4's terrain mesh render + the FO5/FO6 instanced-foliage render in one scene).

## The addition — `engine/terrain/procterrain.h` (APPEND-ONLY after the PT4 block)
Add to `hf::terrain` (do NOT modify PT1–PT4 logic):
- `inline fx SampleHeight(const std::vector<fx>& grid, int n, fx worldSize, fx x, fx z)`: a **bilinear integer
  sample** of the eroded grid at world XZ. Map `x`/`z` (in the SAME Q16.16 world units the grid was generated over —
  `[0, worldSize)`) to fractional grid coordinates `gfx = (fx)((int64)x * (n-1) / worldSize)` (Q16.16), `gfz`
  likewise; the integer cell `gx = gfx >> kFrac` / `gz = gfz >> kFrac` (clamped to `[0, n-2]`); the fractional `tx =
  gfx - (gx << kFrac)` / `tz`; bilinear-blend the 4 grid corners `grid[gz*n+gx{,+1}]`, `grid[(gz+1)*n+gx{,+1}]`
  with `fxmul` (the PT1 blend shape, linear is fine for height). Return the interpolated Q16.16 height. Pure
  integer (a pure function of the grid → the seating is provenance-checkable + deterministic). NO float.

## Driver — `--pt5-meadow-shot` (Vulkan, main.cpp) + `--pt5-meadow` (Metal, visual_test.mm)
Compose PT4's terrain mesh + the foliage meadow in ONE scene (clone the `--pt4-mesh-shot` block + add the foliage,
or clone `--fo6-hero-shot` + add the terrain mesh — whichever reuses more):
1. Build the eroded grid: `GenHeightField(seed, n, worldSize, octaves)` → `ErodeHydraulic` → `ErodeThermal` (PT1→3).
2. Terrain mesh: `BuildIntTerrainMesh(grid, n, worldSizeF, heightScale)` (PT4). **Use a heightScale (and grid
   amplitude) consistent with the foliage seating so plants sit ON the surface** — the cleanest is `heightScale =
   1.0f` so the mesh vertex Y is exactly `FxToFloat(gridHeight)`, matching the foliage Y below (tune the grid
   amplitude so the relief reads). The foliage area must span the SAME world XZ extent as the terrain mesh.
3. Foliage: `auto plants = PlaceFoliage(field, stream);` then **seat each plant**: `plant.base.pos.y =
   SampleHeight(grid, n, worldSize, plant.base.pos.x, plant.base.pos.z);` (lift to the terrain surface — the
   integer seating); then `ApplyWind(plants, wind, frame); AssignLods(plants, camPos, nearR, farR);` and
   `FoliageToRenderInstancesHero(plants, baseScale, leanGain, heightMul)` → the instanced-lit draw. The plant's
   render Y `FxToFloat(plant.base.pos.y)` then lands on the mesh surface (both use `FxToFloat(height)` with the same
   `heightScale=1`). Render the terrain mesh + the instanced foliage together (lit + shadow + sky + post).
4. SAME seed/n/octaves/erosion/heightScale/field/wind/frame/camera/light IN BOTH renderers (the mesh + the seated
   instance set are byte-identical by construction; only the GPU float raster diverges = the visresolve bar).

## Proof (FLOAT visresolve-bar)
```
pt5-meadow: foliage seated on eroded terrain (plants:<K>, verts:<V>, frame:<F>)
pt5-meadow: two renders BYTE-IDENTICAL
pt5-meadow: provenance every plant.y == SampleHeight(grid, plant.xz) {plants:<K>, seated:<K>}
pt5-meadow: empty graph -> bare terrain (no-op) {emptyPlants:0}
```
Assertions: (1) two renders byte-identical; (2) **the seating is a pure function** — for EVERY plant, `plant.base.pos.y
== SampleHeight(grid, n, worldSize, plant.base.pos.x, plant.base.pos.z)` (recompute and compare — the meadow sits
EXACTLY on the eroded surface, deterministically); (3) an empty graph → 0 plants → the bare terrain renders (the
no-op); `shaded > 0`. Register `pt5_meadow` in verify.ps1 $Goldens (Flag `--pt5-meadow`) + `--pt5-meadow-shot` in
$vkShots, mirroring `pt4_mesh`.

## Constraints (HARD)
- APPEND-ONLY to engine/terrain/procterrain.h (do NOT modify PT1–PT4 logic) + the showcase blocks + verify.ps1
  registration. Reuse PT4's BuildIntTerrainMesh + erosion.h + foliage.h (PlaceFoliage/ApplyWind/AssignLods/
  FoliageToRenderInstancesHero) + the lit render path READ-ONLY. Do NOT modify heightmap.{h,cpp}/foliage.h/erosion.h/
  any shader/existing golden. NO new shader, NO new RHI. The seating (SampleHeight) is pure-integer; float only in
  the render bridges (PT4 mesh + FO foliage) + the render path.
- FLOAT slice: do NOT assert strict zero-diff. Proof = two-run byte-identical + provenance + coherent.
- Branch `fix-terrain-pt5`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  Run `hello_triangle.exe --pt5-meadow-shot <out.bmp>`, confirm exit 0 + proof lines + a healthy plant count + the
  seating-provenance + shaded>0, THEN eyeball the PNG (the meadow should sit ON the terrain relief, not floating /
  not buried).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--pt5-meadow-shot` exits 0, the proof lines print,
  two renders byte-identical, the seating provenance holds (every plant.y == SampleHeight), the empty-graph no-op
  holds, shaded>0, AND you eyeballed the PNG and the meadow sits ON the terrain surface (not floating above / buried
  below). Commit message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the branch and
  STOP. Report: commit hash, the proof output, the PNG path + your eyeball verdict (does the foliage sit on the
  relief?), confirmation both renderers use identical params, and the plant/vert counts. (The CONTROLLER bakes the
  Metal golden, confirms two-run 0.0000 + cross-vendor mean, eyeballs the seated meadow. If plants float/bury,
  that's a real seating bug — report it.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
