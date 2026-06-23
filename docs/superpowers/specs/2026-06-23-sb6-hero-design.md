# Slice SB6 — Substrate hero money-shot (Issue #11, flagship #11, 6th/final slice)

The capstone. NO new shader math — `lit_substrate.frag.hlsl` already has all 5 lobes (clearcoat/sheen/
iridescence/anisotropy/SSS). SB6 is just a new showcase that enables ALL FIVE at once on the hero object —
the "Substrate graph" money-shot — plus its golden. COMPLETES flagship #11. FLOAT visresolve-bar slice.

## What to do
- Clone the SB5 `--sb5-sss-shot` showcase in `samples/hello_triangle/main.cpp` → `--sb6-substrate-shot`, and
  `RunSb5SssShowcase`/`--sb5-sss` in `metal_headless/visual_test.mm` → `RunSb6SubstrateShowcase`/`--sb6-substrate`.
  Same scene/camera/light/env. Set ALL the layer params (balanced so the combined material reads rich but not
  blown out — tune if needed):
  ```
  fd.substrateParams[0]  = 0.6f;   // clearcoat
  fd.substrateParams[1]  = 0.10f;  // clearcoat roughness
  fd.substrateParams[2]  = 0.4f;   // sheen
  fd.substrateParams[3]  = 0.6f;   // iridescence
  fd.substrateParams2[0] = 0.5f;   // anisotropy
  fd.substrateParams2[1] = 0.5f;   // SSS
  ```
  SCENE/PARAMS IDENTICAL in both renderers (the PT2 lesson — the controller pixel-compares Metal two-run +
  documents cross-vendor).
- Register `sb6_substrate` in `scripts/verify.ps1` `$Goldens` (Flag `--sb6-substrate`) + `--sb6-substrate-shot`
  in `$vkShots`.

EXACT proof lines (NO additivity control — all params >0 IS the point of SB6):
```
sb6-substrate: two-run BYTE-IDENTICAL
sb6-substrate: {nonBlackPixels:N, meanLuma:L} all 5 Substrate lobes active (clearcoat+sheen+iridescence+aniso+sss)
sb6-substrate: differs from base lit_pbr_ibl (the layered material is distinct)
```
Assertions: (1) two renders byte-identical; (2) shaded>0, coherent; (3) the SB6 render differs from a base
lit_pbr_ibl render of the identical scene (nonBlack>0 + a measurable difference — proves the layers are active).

## Constraints (HARD)
- NO shader change (lit_substrate already has all lobes). Do NOT modify any shader/frame_data/lit.vert. Only the
  two showcases + verify.ps1.
- Branch `fix-issue-11-sb6`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`).
- COMPLETION CRITERIA — do NOT commit until `--sb6-substrate-shot` runs exit 0 + the 3 proof lines print + the
  render differs from base PBR. (Controller bakes the Metal golden, Metal two-run 0.0000, cross-vendor doc,
  eyeballs the combined hero look, re-bakes sb1-sb5 + ibl_helmet → all 0.0000.) Then close issue #11.
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
