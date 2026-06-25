# Slice WFC-S6 — Lit 3D render capstone: the generated-tilemap money-shot (Flagship #29 WFC, 6th/FINAL)

The capstone that COMPLETES flagship #29. S1–S5 made WFC a complete, bit-identical, lockstep-replayable
constraint solver (all proven by cheap pinned-hash clang goldens). S6 is the ONE float crossing: turn a
**collapsed integer tilemap** into a **lit 3D render** — a procedurally-generated terrain/level of tile
instances at per-tile heights, rendered through the EXISTING instanced-lit pipeline. The generation stays
bit-exact integer (the moat); only the final raster/shade is float (the FLOAT visresolve-bar). The
money-shot: a WFC-generated 3D tilemap, byte-identical generation on every machine, rendered on Vulkan
(Windows) and Metal (Mac).

This is the FIRST WFC slice that needs a Mac RENDER-bake (not the clang-hash loop), a Vulkan + Metal
showcase block, and `verify.ps1` registration. It mirrors the **PCG6-field** slice EXACTLY (the closest
twin). NO new shader, NO new RHI — reuse `lit_instanced.vert.hlsl` + `lit.frag.hlsl` + the instanced-lit
pipeline + `scene::InstanceData`/`InstanceTransformLayout`.

## Keep wfc.h self-contained — the render bridge goes in a NEW separate header
`wfc.h` (S1–S5) is self-contained (only `<bit>/<cstddef>/<cstdint>/<vector>` + `net/session.h`) so its
clang-hash proof stays cheap. The render bridge needs `math::Mat4`/`FromTRS`/`Vec3` + `fpx::FxToFloat`,
which would break that. So **DO NOT add the bridge to wfc.h** — create a NEW header:

### NEW file: engine/wfc/wfc_render.h (namespace hf::wfc)
Includes `"wfc/wfc.h"` + the engine math + `"sim/fpx.h"` (for `FxToFloat`) — same includes the PCG/foliage
render bridges use (read `engine/pcg/pcg.h`'s includes + `PcgToRenderInstances` at pcg.h:375-389 and mirror
them). This header is NOT standalone-clang (it's the render layer, only used by the showcase) — that's fine;
`wfc.h`'s S1–S5 proof is unaffected.

Define:
1. **A per-tile height/visual table** — a fixed integer mapping from tile kind → render height (and
   optionally a per-kind material). For the S1 showcase gradient (0=water,1=sand,2=grass,3=rock), a
   believable relief, e.g. heights `{water:0, sand:1, grass:2, rock:4}` in some integer unit (the implementer
   tunes for a good-looking relief). Keep it a FIXED table so the render is deterministic.
   ```cpp
   struct WfcRenderStyle {
       float tileSize   = 1.0f;   // world spacing between cell centers
       float heightUnit = 0.25f;  // world height per integer height step
       std::vector<int32_t> tileHeight;   // [tileCount] integer height per tile kind
       // optional: per-kind material (metallic/roughness) — see the showcase
   };
   ```
2. **`WfcToRenderInstances`** — the render bridge (the `PcgToRenderInstances` twin). For each DECIDED cell
   (PopCount(domain)==1), emit one instance `Mat4` placing a unit tile mesh at the cell's world position:
   `t = { x*tileSize, height*heightUnit, z*tileSize }` (centered so the map straddles the origin — subtract
   half the extent), identity orientation, scale = tileSize (so tiles tile seamlessly), via
   `math::FromTRS(t, identityQuat, Vec3{tileSize*0.5f, ...})`. The **single float crossing** is here (the
   grid cells + style table are integer; positions/heights become float once). Undecided cells (PopCount!=1)
   are skipped (a partial grid renders only its decided cells).
   ```cpp
   inline int TileOf(Domain d);   // the single set-bit index of a decided cell (-1 if not decided)
   inline std::vector<math::Mat4> WfcToRenderInstances(const Grid& g, const WfcRenderStyle& style);
   ```
   (If you want per-tile-kind COLOR distinction and the lit pipeline supports it via the per-draw material
   push-constant, you MAY return instances GROUPED by tile kind — e.g. return `std::vector<std::vector<Mat4>>`
   indexed by tile, or a flat list + a parallel `std::vector<uint32_t> tileKind` — so the showcase can issue
   one draw per kind with a per-kind material. If the pipeline has no albedo channel, rely on HEIGHT relief +
   lighting for distinction and return a flat `std::vector<Mat4>`. Pick what makes the best money-shot within
   "no new shader"; document the choice.)

## The showcase + golden (FLOAT visresolve-bar) — mirror PCG6-field EXACTLY
### Vulkan block in samples/hello_triangle/main.cpp (clone `--pcg6-field-shot`, ~main.cpp:72260-72627)
Add `--wfc6-render-shot <out.bmp>` (its OWN arg-parse line; if main.cpp's arg parse risks MSVC C1061, give
the flag its own parse loop — see memory). In the block:
1. **Bit-exact integer generation:** build the FIXED tileset (S1 `MakeShowcaseTileSet()` OR an S4 learned
   one — pick one, keep it fixed), a fixed grid size (e.g. 28×28 or 32×32 — big enough to read as a
   landscape), a fixed `kSeed`, `kMaxSteps`. `Grid g = hf::wfc::Generate(ts, kSeed, W, H, kMaxSteps);`
   Confirm `solved` (a fully-collapsed map — if the showcase gradient occasionally fails at a size, pick a
   seed/size that solves; assert it). Build `WfcRenderStyle` + `mats = WfcToRenderInstances(g, style)`.
2. **Reuse the instanced-lit pipeline VERBATIM** (lit_instanced.vert.hlsl.spv + lit.frag.hlsl.spv +
   `scene::MeshVertexLayout()`/`InstanceTransformLayout()`/`InstanceData`, the pushConstantSize=16 material).
   Prefer a CUBE/BOX mesh for the tilemap look if a box mesh helper is readily available in the showcase mesh
   utilities; otherwise the existing instanced mesh (e.g. the sphere PCG6 uses) is acceptable — eyeball it.
3. **Camera:** an elevated 3/4 hero view looking down over the tilemap (the PT6/FO6 hero framing) so the
   relief + the generated layout read clearly, receding to a horizon. Directional key light (the PCG6 light).
4. **Proof/provenance lines** (the FLOAT visresolve-bar — mirror PCG6-field's four lines):
```
wfc6-render: WFC-generated tilemap (tiles:<W*H>, instances:<K decided>, lit 3D instanced, seed:<S>)
wfc6-render: two renders BYTE-IDENTICAL
wfc6-render: provenance instances == WfcToRenderInstances(Generate(ts,seed)) {instances:<K>, shaded:<P>}
wfc6-render: empty -> base scene (no-op) {emptyInstances:0}
```
Assertions in the block (exit 1 on failure): (1) two renders byte-identical (re-render, memcmp); (2)
provenance — `WfcToRenderInstances(Generate(ts, kSeed, ...))` recomputed == `mats` (memcmp the Mat4 bytes),
and instance count == decided-cell count; (3) `shaded > 0` (a populated lit frame); (4) the empty case (a
trivially empty/uncollapsed grid → zero instances → base scene no-op). Also assert the generated map is
fully solved + globally consistent (the S3 sweep) so the render reflects a VALID tilemap.

### Metal mirror in metal_headless/visual_test.mm (clone the PCG6-field `RunPcg6FieldShowcase`, ~visual_test.mm:60634-60932)
Add the `--wfc6-render` Metal path: IDENTICAL tileset/seed/grid/style/camera/light (only the GPU float
raster + FlipProjY differ). SAME proof lines. The controller bakes this golden on the Mac. **The
showcase MUST wire BOTH `--wfc6-render-shot` (Vulkan) and `--wfc6-render` (Metal)** with the SAME scene
(memory: grep visual_test.mm for the flag before the Mac bake; a size/scene mismatch is the tell).

### verify.ps1 registration (scripts/verify.ps1)
- Add to `$Goldens` (near pcg6_field line ~83): `@{ Name = 'wfc6_render'; Flag = '--wfc6-render' }` (a FLOAT
  golden — the Metal bake is the reference).
- Add `'--wfc6-render-shot'` to the `$vkShots` array (~line 730), mirroring `'--pcg6-field-shot'`.

## Proof tier (FLOAT visresolve-bar — NOT strict zero-diff)
This is the ONE float slice of the flagship. Do NOT assert cross-backend zero-diff pixels. The bar (PCG6/
FO6/PT6 convention):
- **Integer generation bit-exact** (the WFC grid is identical Vulkan/Metal — same `Generate(ts,seed)`).
- **Two renders BYTE-IDENTICAL per vendor** (frame-frame determinism gate).
- **Provenance** (instances == recomputed `WfcToRenderInstances(Generate(...))`).
- **Metal golden** baked by the controller (Metal render == Metal golden DIFF 0.0000) + a documented
  **cross-vendor mean** (Vulkan vs Metal, in-band like PCG6/FO6 ~3–45) + **EYEBALL**.

## YOU MUST EYEBALL YOUR OWN RESULT before committing
Convert the Vulkan `--wfc6-render-shot` BMP to PNG (Add-Type System.Drawing) and READ it. Confirm it reads
as a **coherent WFC-generated 3D tilemap** — a believable generated terrain/level: tiles at varied heights
forming the gradient/biome layout the rules imply (water low, rock high, etc.), lit and legible from the
hero camera — NOT a flat featureless grid, NOT a black/broken render, NOT random noise. If it's flat,
illegible, or the camera is wrong, ITERATE (tune heights/heightUnit, camera angle, grid size, per-kind
material) until it genuinely reads as a generated tilemap. Only commit once it reads as a money-shot.

## Constraints (HARD)
- NEW `engine/wfc/wfc_render.h` (the render bridge — includes wfc.h + math + fpx, NOT standalone, that's
  fine). Do NOT add render code to `wfc.h` (keep its S1–S5 self-containment + clang-hash proof intact). Do
  NOT modify S1–S5 code in `wfc.h`. NO new shader, NO new RHI — reuse the instanced-lit pipeline verbatim.
- The generation path stays PURE INTEGER (Generate is S1–S5); only `wfc_render.h` + the showcase touch
  float. NO runtime trig in generation.
- Branch `fix-wfc-s6`, commit there, do NOT merge. Do NOT commit `tests/golden/metal/*` (the CONTROLLER
  bakes the Metal golden). Commit `engine/wfc/wfc_render.h` + the main.cpp + visual_test.mm blocks +
  verify.ps1 registration.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  Run `hello_triangle.exe --wfc6-render-shot <out.bmp>`, confirm exit 0 + the four proof lines + provenance
  + shaded>0 + two-run byte-identical, THEN eyeball the PNG.
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--wfc6-render-shot` exits 0, the proof
  lines print, two renders byte-identical, provenance holds, the no-op holds, shaded>0, the generated map is
  solved+globally-consistent, AND you eyeballed the PNG and it genuinely reads as a WFC-generated 3D
  tilemap. Confirm BOTH `--wfc6-render-shot` (Vulkan) AND `--wfc6-render` (Metal) are wired with the SAME
  scene (grep visual_test.mm for `wfc6`). Report: commit hash, the proof output, the PNG path + eyeball
  verdict (describe what it shows), the tile/instance/shaded counts, the tileset/seed/grid/style used, and
  confirmation both renderers use identical params. Commit message via temp file + `git commit -F` (Bash
  heredoc). Commit to the branch and STOP.
  (The CONTROLLER greps visual_test.mm for the flag, bakes the Metal golden on the Mac, confirms Metal
  two-run DIFF 0.0000 + cross-vendor mean + eyeballs the money-shot, commits the golden, ff-merges to
  master + pushes + deletes the branch, then writes the ARCHITECTURE.md WFC section + closes out flagship
  #29. If after iterating you cannot make it a compelling tilemap, STOP and report.)
