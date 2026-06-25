# Slice ECON-S6 — Lit 3D render capstone: the economy skyline money-shot (Flagship #30 ECON, 6th/FINAL)

The capstone that COMPLETES flagship #30. S1–S5 proved the deterministic gameplay-systems moat (ledger,
crafting, economy tick, market, and the lockstep/rollback/desync headline) — all by cheap pinned-hash
clang goldens. S6 is the ONE float crossing: visualize the deterministic economy as a **lit 3D skyline** —
instanced bars per `(entity, item)` slot, each bar's HEIGHT driven by the bit-exact integer stock, colored
by item kind, rendered through the EXISTING instanced-lit pipeline. The economy state stays bit-exact
integer (the moat); only the final raster/shade is float (the FLOAT visresolve-bar). This is OPTIONAL
eye-candy — the moat is fully proven without it — but it completes the 6-slice arc and gives the flagship a
money-shot.

This mirrors the **WFC-S6** slice EXACTLY (which itself cloned PCG6-field): a render bridge in a SEPARATE
header (so `econ.h`'s S1–S5 clang-hash proof stays intact), a Vulkan + Metal showcase block, `verify.ps1`
registration, a Mac render-bake. NO new shader, NO new RHI — reuse `lit_instanced.vert.hlsl` +
`lit.frag.hlsl` + the instanced-lit pipeline + `scene::InstanceData`/`InstanceTransformLayout`.

## Keep econ.h self-contained — the render bridge goes in a NEW separate header
`econ.h` (S1–S5) is self-contained (only `<cstdint>/<cstddef>/<vector>` + `net/session.h`). The render
bridge needs `math::Mat4`/`FromTRS`/`Vec3` + `fpx::FxToFloat`, which would break that. So **DO NOT add the
bridge to econ.h** — create a NEW header, exactly as WFC-S6 created `engine/wfc/wfc_render.h`.

### NEW file: engine/econ/econ_render.h (namespace hf::econ)
Includes `"econ/econ.h"` + the engine math + `"sim/fpx.h"` (for `FxToFloat`) — mirror
`engine/wfc/wfc_render.h`'s includes EXACTLY (read it; it is the known-good twin). NOT standalone-clang —
that's fine; `econ.h`'s S1–S5 proof is unaffected. Define:
1. **`EconRenderStyle`** — fixed layout/visual constants:
   ```cpp
   struct EconRenderStyle {
       float barSize    = 1.0f;    // world spacing between slot columns (X = entity, Z = item)
       float heightUnit = 0.05f;   // world height per unit of integer stock (tune for a readable skyline)
       float maxHeight  = 12.0f;   // clamp very tall bars so one rich slot doesn't dwarf the scene (visual only)
   };
   ```
2. **`EconToRenderInstances`** — the render bridge (the `WfcToRenderInstances` twin). For each `(entity,
   item)` slot with `stock > 0`, emit one bar instance: position `t = { (entity - (entityCount-1)*0.5f) *
   barSize, height*0.5f, (item - (itemCount-1)*0.5f) * barSize }` (centered grid), scale `Vec3{barSize*0.45f,
   height, barSize*0.45f}` where `height = min(maxHeight, FxToFloat-of-stock * heightUnit)` — a unit cube
   stretched in Y into a bar (so the bar sits on the ground plane, base at y=0, top at `height`). Identity
   orientation, via `math::FromTRS(t, identityQuat, scale)`. The **single float crossing** is here (the
   ledger is integer; heights become float once). Skip empty slots (`stock <= 0`).
   ```cpp
   inline std::vector<math::Mat4> EconToRenderInstances(const EconState& s, const EconRenderStyle& style);
   ```
   For per-item COLOR (like WFC-S6's per-kind biome albedo), return instances GROUPED by item (a parallel
   `std::vector<uint32_t> itemKind`, or `std::vector<std::vector<Mat4>>` indexed by item) so the showcase
   issues one draw per item with a per-item albedo (coin gold, ore grey, ingot blue, tool red — pick a
   readable palette). Mirror WFC-S6's `WfcTileKinds` grouping. (A flat list + height-relief alone is the
   fallback if the pipeline has no albedo; prefer per-item color for legibility.)
3. (Optional) a small quest-marker row: for each `kComplete` objective, a distinct marker instance (e.g. a
   taller thin pillar) at a reserved row — eye-candy showing the quest chain finished. Keep it simple; the
   bars are the main subject.

## The showcase + golden (FLOAT visresolve-bar) — mirror WFC-S6 / PCG6-field
Build the showcase economy and RUN it to an interesting state before rendering: `EconState s =
MakeShowcaseState();` then drive the S5 `MakeShowcaseCommandStream()` (or a longer fixed script + a few
`EconTick`s) through the step so the skyline has varied bar heights AND the quest chain is complete — a
FIXED deterministic state. `mats = EconToRenderInstances(s, style)`.

### Vulkan block in samples/hello_triangle/main.cpp (clone `--wfc6-render-shot`)
Add `--econ6-render-shot <out.bmp>` (its OWN arg-parse line; own parse loop if MSVC C1061 risk). Reuse the
instanced-lit pipeline VERBATIM (the wfc6 block is the template — same pipeline, same `scene::InstanceData`,
the per-kind-draw material loop). Prefer a CUBE/BOX mesh (the wfc6 mesh). Elevated 3/4 hero camera over the
skyline. Directional key light. Proof lines (mirror wfc6's four):
```
econ6-render: economy skyline (entities:<E>, items:<I>, bars:<K populated slots>, lit 3D instanced)
econ6-render: two renders BYTE-IDENTICAL
econ6-render: provenance instances == EconToRenderInstances(state after fixed script) {bars:<K>, shaded:<P>}
econ6-render: empty -> base scene (no-op) {emptyBars:0}
```
Assertions (exit 1 on fail): (1) two renders byte-identical (re-render + memcmp); (2) provenance —
recompute the deterministic state (run the fixed script again) + `EconToRenderInstances` == `mats` (memcmp
the Mat4 bytes), bar count == populated-slot count; (3) `shaded > 0`; (4) the empty case (a zeroed ledger →
zero bars → base scene no-op). Also assert the rendered state is the EXPECTED deterministic state (its
`DigestEconState` == a value you can pin or cross-check against the S5 pinned digest if the same script).

### Metal mirror in metal_headless/visual_test.mm (clone the wfc6 `RunWfc6RenderShowcase`)
Add the `--econ6-render` Metal path: IDENTICAL state/script/style/camera/light (only the GPU float raster +
FlipProjY differ). SAME proof lines. The controller bakes this golden. **Wire BOTH `--econ6-render-shot`
(Vulkan) AND `--econ6-render` (Metal) with the SAME scene** (memory: grep visual_test.mm for `econ6` before
the Mac bake; a scene mismatch is the tell).

### verify.ps1 registration
- `$Goldens`: `@{ Name = 'econ6_render'; Flag = '--econ6-render' }` (near wfc6_render).
- `$vkShots`: add `'--econ6-render-shot'` (near `'--wfc6-render-shot'`).

## Proof tier (FLOAT visresolve-bar — NOT strict zero-diff)
The ONE float slice of the flagship. Integer state bit-exact (identical Vulkan/Metal — same deterministic
script). Two renders BYTE-IDENTICAL per vendor. Provenance (bars == recomputed `EconToRenderInstances`).
Metal golden baked by the controller (Metal render == Metal golden DIFF 0.0000) + documented cross-vendor
mean (in-band like wfc6 ~40) + EYEBALL.

## YOU MUST EYEBALL YOUR OWN RESULT before committing
Convert the Vulkan `--econ6-render-shot` BMP to PNG (Add-Type System.Drawing) and READ it. Confirm it reads
as a **coherent 3D economy skyline** — a grid of colored bars at varied heights (the entities × items
ledger as a cityscape), legible from the hero camera, NOT flat/featureless, NOT black/broken, NOT one giant
bar dwarfing everything (tune `maxHeight`/`heightUnit`). If illegible, ITERATE (heights, camera, palette,
maxHeight clamp) until it reads as an economy skyline. Only commit once it's a money-shot.

## Constraints (HARD)
- NEW `engine/econ/econ_render.h` (the render bridge — includes econ.h + math + fpx, NOT standalone; mirror
  `wfc_render.h`). Do NOT add render code to `econ.h` (keep S1–S5 self-containment + clang-hash proof
  intact). Do NOT modify S1–S5 code in `econ.h`. NO new shader, NO new RHI — reuse the instanced-lit
  pipeline verbatim.
- The economy/state path stays PURE INTEGER (S1–S5); only `econ_render.h` + the showcase touch float.
- Branch `fix-econ-s6`, commit there, do NOT merge. Do NOT commit `tests/golden/metal/*` (the CONTROLLER
  bakes the Metal golden). Commit `engine/econ/econ_render.h` + main.cpp + visual_test.mm + verify.ps1.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  Run `hello_triangle.exe --econ6-render-shot <out.bmp>`, confirm exit 0 + the four proof lines + provenance
  + shaded>0 + two-run byte-identical, THEN eyeball the PNG.
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--econ6-render-shot` exits 0, the proof
  lines print, two renders byte-identical, provenance holds, the no-op holds, shaded>0, AND you eyeballed
  the PNG and it genuinely reads as a 3D economy skyline. Confirm BOTH `--econ6-render-shot` (Vulkan) AND
  `--econ6-render` (Metal) are wired with the SAME scene (grep visual_test.mm for `econ6`). Report: commit
  hash, the proof output, the PNG path + eyeball verdict (describe what it shows), the entity/item/bar/shaded
  counts, the script/style used, and confirmation both renderers use identical params. Commit message via
  temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER greps visual_test.mm for the flag, bakes the Metal golden on the Mac [push branch first;
  Mac Metal build = `cmake -S metal_headless`; compare.sh wants PNG], confirms Metal two-run DIFF 0.0000 +
  cross-vendor mean + eyeballs the money-shot, commits the golden, ff-merges to master + pushes + deletes
  the branch, then writes the ARCHITECTURE.md econ section + closes out flagship #30, then PIVOTS THE LOOP
  TO THE GITHUB ISSUES BACKLOG per the standing user directive. If the skyline can't be made compelling
  after iterating, STOP and report — S6 is optional, the moat is already proven.)
