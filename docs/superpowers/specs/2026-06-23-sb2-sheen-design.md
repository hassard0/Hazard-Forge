# Slice SB2 — Sheen (Issue #11, flagship #11, 2nd slice)

APPEND a SHEEN lobe to `shaders/lit_substrate.frag.hlsl` (SB1 clearcoat is BYTE-FROZEN — add after the
clearcoat block, before emissive). Sheen is a retroreflective fabric/velvet lobe (glTF KHR_materials_sheen:
Charlie distribution + Ashikhmin visibility). Gated by `f.substrateParams.z` (sheen strength) — sheen=0 is an
EXACT identity over the base. FLOAT visresolve-bar slice (NOT strict-zero). NO FrameData layout change (the
substrateParams channel already exists from SB1).

## The sheen math (append to lit_substrate.frag main(), after the clearcoat block, before `rgb += emis`)
Read `float sheen = f.substrateParams.z;`. Use a fixed sheen roughness (velvet rim): `float sheenRough =
0.3;` (or reuse `roughness` — pick one and keep it; a medium value reads as fabric). Sheen color = white
(`float3 sheenColor = float3(1,1,1);`). Add a retroreflective lobe on the normal-mapped N for the DIRECTIONAL
light:
```hlsl
// SB2 sheen — Charlie distribution + Ashikhmin visibility (glTF KHR_materials_sheen). sheen=0 -> +0 -> identity.
{
    float sheen = f.substrateParams.z;
    float3 Ls = normalize(-f.lightDir.xyz);
    float3 Hs = normalize(Ls + V);
    float NoLs = max(dot(N, Ls), 0.0);
    float NoVs = max(dot(N, V), 1e-4);
    float NoHs = max(dot(N, Hs), 0.0);
    float sr = clamp(0.3, 0.07, 1.0);                  // sheen roughness (fixed velvet)
    float invR = 1.0 / sr;
    float cos2h = NoHs * NoHs;
    float sin2h = max(1.0 - cos2h, 0.0078125);
    float Dch = (2.0 + invR) * pow(sin2h, invR * 0.5) * (1.0 / (2.0 * 3.14159265));   // D_charlie
    float Vash = 1.0 / max(4.0 * (NoLs + NoVs - NoLs * NoVs), 1e-4);                   // V_ashikhmin
    float3 sheenLobe = sheenColor * (Dch * Vash * NoLs);
    rgb += sheen * sheenLobe * (f.lightColor.rgb * shadow) * ao;
}
```
sheen=0 → `rgb += 0` → byte-identical to the base. (Confirm `N`, `V`, `shadow`, `ao`, `f.lightColor`,
`f.lightDir` are the exact names used in lit_substrate's base — copy them verbatim, like SB1 did.)

## Showcase `--sb2-sheen-shot` (Vulkan) / `--sb2-sheen` (Metal)
Clone the SB1 `--sb1-clearcoat-shot` block (same scene/camera/light/env, lit_substrate pipeline). Set ONLY
`fd.substrateParams[2]=0.8f` (strong sheen; leave [0],[1] = 0 so clearcoat is off — SB2 shows sheen alone).
A fabric/velvet rim glow on the helmet's silhouette. SCENE/PARAMS IDENTICAL in both renderers.

EXACT proof lines (fail loudly):
```
sb2-sheen: two-run BYTE-IDENTICAL
sb2-sheen: sheen=0 == lit_pbr_ibl BYTE-IDENTICAL (additivity control)
sb2-sheen: {nonBlackPixels:N, meanLuma:L} sheen>0 adds a retroreflective rim
```
Assertions: (1) two renders byte-identical; (2) THE ADDITIVITY CONTROL — render the same scene through
lit_substrate with ALL substrateParams=0 AND through the frozen lit_pbr_ibl, assert byte-identical (proves
sheen-at-zero identity → existing goldens safe); (3) sheen>0 differs from sheen=0 (a real added rim).
Register `sb2_sheen` in verify.ps1 $Goldens (Flag `--sb2-sheen`) + `--sb2-sheen-shot` in $vkShots.

## Constraints (HARD)
- APPEND to lit_substrate.frag (SB1 clearcoat block byte-frozen). Do NOT modify lit_pbr.frag/lit_pbr_ibl.frag/
  existing pbr_core functions/lit.vert/the material varying/frame_data.hlsli (substrateParams already exists).
- Branch `fix-issue-11-sb2`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`; under
  Git Bash this can silently no-op — use the PowerShell tool with the absolute vcvars path).
- COMPLETION CRITERIA — do NOT commit until: `--sb2-sheen-shot` runs exit 0, all 3 proof lines print, AND the
  additivity control HOLDS (sheen=0 == lit_pbr_ibl byte-identical). (The CONTROLLER bakes the Metal golden,
  checks Metal two-run 0.0000, documents the cross-vendor mean, eyeballs the PNG for a visible velvet rim, and
  re-bakes sb1_clearcoat + ibl_helmet to confirm 0.0000 invariance — SB2 must not perturb SB1 or base IBL.)
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
