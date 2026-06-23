# Slice SB3 — Iridescence / thin-film (Issue #11, flagship #11, 3rd slice — the dramatic one)

APPEND iridescence to `shaders/lit_substrate.frag.hlsl` (SB1 clearcoat + SB2 sheen BYTE-FROZEN). Unlike
SB1/SB2 (which ADD a lobe), SB3 LERPS the base specular `F0` toward a view-angle-dependent spectral tint
(oil-slick / soap-bubble rainbow), gated by `f.substrateParams.w` — iridescence=0 leaves F0 unchanged → EXACT
identity. FLOAT visresolve-bar slice. NO FrameData layout change (channel exists).

## The math
Add a helper (to pbr_core.hlsli as an ADDITIVE new function, OR inline in lit_substrate). A thin-film
approximation — NOT physically exact (the engine's approximate-but-golden-locked altitude), a view-angle
spectral interference: optical-path-difference grows toward grazing → 3 phase-shifted cosines (RGB 120° apart)
→ a rainbow that sweeps across the surface:
```hlsl
float3 hfIridescence(float cosTheta, float thickness) {
    float opd = thickness * (1.0 - cosTheta);          // crude OPD, grows toward grazing angles
    const float TAU = 6.2831853;
    float3 phase = float3(opd * TAU, opd * TAU + 2.0944, opd * TAU + 4.1888);  // R, G, B 120° apart
    return saturate(0.5 + 0.5 * cos(phase));
}
```
In main(), AFTER the base `F0` is computed (find the `F0 = lerp(0.04..., albedo, metallic)` line) and BEFORE
F0 is used by `hfCookTorrance` and the IBL specular Fresnel, insert:
```hlsl
float iridescence = f.substrateParams.w;
float NoVi = max(dot(N, V), 1e-4);
F0 = lerp(F0, hfIridescence(NoVi, 3.0), iridescence);   // iridescence=0 -> F0 unchanged -> identity
```
Because F0 is lerped IN PLACE before BOTH the direct and IBL specular consume it, the iridescent tint shows in
both the lit highlight and the env reflection. `iridescence=0` → `lerp(F0, _, 0) == F0` exactly → byte-
identical to the base lit_pbr_ibl render. (Confirm the exact F0 variable + the N/V names by reading the
shader; place the lerp so NOTHING reads F0 before it except the lerp itself.) Film thickness 3.0 gives a few
rainbow bands across the angle range — tune so the effect reads clearly on the helmet.

## Showcase `--sb3-iridescence-shot` (Vulkan) / `--sb3-iridescence` (Metal)
Clone the SB2 `--sb2-sheen-shot` block. Set ONLY `fd.substrateParams[3]=1.0f` (full iridescence; clearcoat
[0]/[1]=0, sheen [2]=0 — SB3 shows iridescence alone). The helmet's metal reflects an oil-slick rainbow.
SCENE/PARAMS IDENTICAL in both renderers.

EXACT proof lines (fail loudly):
```
sb3-iridescence: two-run BYTE-IDENTICAL
sb3-iridescence: iridescence=0 == lit_pbr_ibl BYTE-IDENTICAL (additivity control)
sb3-iridescence: {nonBlackPixels:N, hueSpread:H} iridescence>0 shifts specular to a spectral tint
```
Assertions: (1) two renders byte-identical; (2) THE ADDITIVITY CONTROL — render the SAME scene through
lit_substrate with ALL substrateParams=0 AND through the frozen lit_pbr_ibl, assert byte-identical (proves
iridescence-at-zero identity → existing goldens safe); (3) iridescence>0 differs from iridescence=0 (a real
spectral shift — nonBlack>0 + a measurable color/hue change vs the iri=0 render). Register `sb3_iridescence`
in verify.ps1 $Goldens (Flag `--sb3-iridescence`) + `--sb3-iridescence-shot` in $vkShots.

## Constraints (HARD)
- APPEND to lit_substrate.frag (SB1/SB2 blocks byte-frozen) + optionally an ADDITIVE helper in pbr_core.hlsli
  (do NOT modify existing pbr_core functions). Do NOT modify lit_pbr.frag/lit_pbr_ibl.frag/lit.vert/the
  material varying/frame_data.hlsli.
- Branch `fix-issue-11-sb3`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`; under
  Git Bash this can silently no-op — use the PowerShell tool with the absolute vcvars path + `cd /d <repo>`).
- COMPLETION CRITERIA — do NOT commit until: `--sb3-iridescence-shot` runs exit 0, all 3 proof lines print, AND
  the additivity control HOLDS (iridescence=0 == lit_pbr_ibl byte-identical — the make-or-break for F0 lerp
  placement). (The CONTROLLER bakes the Metal golden, checks Metal two-run 0.0000, documents cross-vendor,
  eyeballs the PNG for a visible oil-slick rainbow, and re-bakes sb1/sb2 + ibl_helmet → all must stay 0.0000.)
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
