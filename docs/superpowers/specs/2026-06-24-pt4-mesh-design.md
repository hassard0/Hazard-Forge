# Slice PT4 — Eroded terrain mesh lit 3D render (Flagship #26 TERRAIN, 4th slice — first FLOAT slice)

Render the bit-exact integer eroded heightfield (PT1 gen + PT2 hydraulic + PT3 thermal) as a lit 3D terrain MESH
through the EXISTING lit pipeline — the integer grid crosses to float ONCE at the mesh build (per-vertex height +
a deterministic host-float normal, the FO5/PCG6 render-bridge precedent). The arc's first **float** slice
(visresolve-bar): Metal two-run DIFF 0.0000 + provenance + cross-vendor mean + eyeball. NO new shader, NO new RHI
(reuse the terrain mesh render path verbatim).

## The render bridge — `engine/terrain/procterrain.h` (APPEND-ONLY after PT1; render-only float)
Add to `hf::terrain` (do NOT modify PT1/erosion logic). Mirror the FROZEN float `BuildTerrain` (heightmap.cpp:66-130)
but drive Y from the INTEGER eroded grid:
- `inline scene::TerrainMesh BuildIntTerrainMesh(const std::vector<fx>& grid, int n, float worldSize, float heightScale)`
  (or return the engine's `TerrainMesh` type — read heightmap.h:34 `struct TerrainMesh { std::vector<scene::Vertex>
  verts; std::vector<uint32_t> indices; }`). For each grid vertex `(ix,iz)`: world `x = -half + step*ix`, `z = -half
  + step*iz` (`half = worldSize/2`, `step = worldSize/(n-1)`); `y = FxToFloat(grid[iz*n+ix]) * heightScale` (the ONE
  float crossing — render-only). The **central-difference normal** from the INTEGER grid neighbours (sample
  `grid[iz*n+ix±1]` / `grid[(iz±1)*n+ix]`, `FxToFloat`, finite-difference → `nx=-dHx, ny=1, nz=-dHz`, normalize in
  host float — deterministic, NOT bit-exact, the float side of the bar). UVs `(ix/(n-1), iz/(n-1))`. A height-tint
  vertex color ramp (mirror heightmap.cpp:102-120: low→grass, mid→rock, high→snow, normalized against the grid's
  height band). Build the `(n-1)*(n-1)*6` index grid (two tris per quad — same winding as `BuildTerrain`). Empty/
  flat grid → a flat plane (all y=0, normals up — the no-op). `#include "math/math.h"` + the scene vertex header as
  `BuildTerrain` does. This is the ONLY float code added (clearly commented as the render bridge).

## Driver — `--pt4-mesh-shot` (Vulkan, main.cpp) + `--pt4-mesh` (Metal, visual_test.mm)
Clone the EXISTING terrain-mesh render showcase (grep `RunTerrainShowcase` in visual_test.mm + the Vulkan terrain
showcase in main.cpp — the lit mesh upload via `scene::MeshVertexLayout()` + sky + shadow + post + a 3/4 camera +
directional light) VERBATIM — swap `BuildTerrain(...)` for `BuildIntTerrainMesh(erodedGrid, n, worldSize,
heightScale)` where `erodedGrid = GenHeightField(seed, n, worldSize, octaves)` then `ErodeHydraulic(grid, n, hIters)`
then `ErodeThermal(grid, n, tIters, talus)` (the full PT1→PT2→PT3 pipeline). Pick an `n` that renders well (the
existing terrain showcase's `n`, e.g. 128–256) + a `heightScale` so the relief reads. SAME seed/n/octaves/erosion/
heightScale/camera/light IN BOTH renderers (the mesh is byte-identical by construction; only the GPU float raster
diverges = the visresolve bar). Match the showcase's own scene (avoid a default-scene size mismatch).

## Proof (FLOAT visresolve-bar)
```
pt4-mesh: eroded terrain mesh (verts:<V>, tris:<T>, n:<n>)
pt4-mesh: two renders BYTE-IDENTICAL
pt4-mesh: provenance verts == grid n*n {verts:<V>, shaded:<P>}
pt4-mesh: flat grid -> flat plane (no-op) {flatVerts:<V>}
```
Assertions: (1) two renders byte-identical (deterministic grid + mesh + render); (2) provenance — `verts.size() ==
n*n`, `indices.size() == (n-1)*(n-1)*6`, `shaded > 0` (a coherent lit terrain); (3) a flat (zero) grid → a flat
plane (all verts y=0 — the no-op). Register `pt4_mesh` in verify.ps1 $Goldens (Flag `--pt4-mesh`) + `--pt4-mesh-shot`
in $vkShots, mirroring `pcg6_field` / `fo5_scale` (a FLOAT golden).

## Constraints (HARD)
- APPEND-ONLY to engine/terrain/procterrain.h (do NOT modify PT1/PT2/PT3 logic) + the showcase blocks + verify.ps1
  registration. Reuse the lit mesh render path + scene::Vertex/TerrainMesh + erosion.h/procterrain.h READ-ONLY. Do
  NOT modify heightmap.{h,cpp}/erosion.h/any shader/existing golden. NO new shader, NO new RHI. Float is allowed
  ONLY in `BuildIntTerrainMesh` (the render bridge) + the render path (NOT in PT1–PT3 data).
- FLOAT slice: do NOT assert strict zero-diff. Proof = two-run byte-identical + provenance + coherent; the
  controller handles Metal two-run 0.0000 + cross-vendor mean + eyeball.
- Branch `fix-terrain-pt4`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  Run `hello_triangle.exe --pt4-mesh-shot <out.bmp>`, confirm exit 0 + proof lines + verts==n*n + shaded>0 (a
  non-black lit terrain).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--pt4-mesh-shot` exits 0, the proof lines print,
  two renders byte-identical, the provenance (verts==n*n, indices==(n-1)²*6) holds, the flat-grid no-op holds,
  shaded>0. Commit message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the branch and
  STOP. Report: commit hash, the proof output (verts/tris/n), image path, confirmation both renderers use identical
  params, and the vert count. (The CONTROLLER bakes the Metal golden, confirms two-run 0.0000 + cross-vendor mean,
  eyeballs a coherent eroded-terrain relief. If the render is black/flat-when-it-should-have-relief, that's a real
  failure — report it.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
