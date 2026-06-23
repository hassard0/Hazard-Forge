# Slice SB4 — Anisotropy (Issue #11, flagship #11, 4th slice)

APPEND anisotropy to `shaders/lit_substrate.frag.hlsl` (SB1/SB2/SB3 blocks BYTE-FROZEN — add after the SB3
iridescence F0 lerp / with the other lobe blocks, before emissive). Anisotropic GGX → stretched specular
highlights along the surface tangent (brushed metal, hair, satin). Gated by `f.substrateParams2.x` (anisotropy
in [-1,1], + along tangent / - along bitangent). FLOAT visresolve-bar slice. NO FrameData layout change.

## The math — an additive DIFFERENCE term (identity-at-zero by the gate)
Rather than modifying the frozen `hfCookTorrance` base specular, compute BOTH the isotropic and anisotropic
directional-light specular in lit_substrate and add their DIFFERENCE scaled by `aniso` — so aniso=0 → +0 →
EXACT identity (the gate, not formula equality, guarantees it). Append after the base rgb accumulation:
```hlsl
{
    float aniso = f.substrateParams2.x;
    // Tangent frame (Gram-Schmidt the interpolated world tangent against the shading normal N).
    float3 Ta = normalize(i.wtangent - N * dot(N, i.wtangent));
    float3 Ba = cross(N, Ta);
    float3 La = normalize(-f.lightDir.xyz);
    float3 Ha = normalize(La + V);
    float NoLa = max(dot(N, La), 0.0);
    float NoVa = max(dot(N, V), 1e-4);
    float NoHa = max(dot(N, Ha), 0.0);
    float VoHa = max(dot(V, Ha), 0.0);
    float alpha = roughness * roughness;
    float at = max(alpha * (1.0 + aniso), 1e-4);
    float ab = max(alpha * (1.0 - aniso), 1e-4);
    float ToH = dot(Ta, Ha), BoH = dot(Ba, Ha);
    float denom = (ToH*ToH)/(at*at) + (BoH*BoH)/(ab*ab) + NoHa*NoHa;
    float Daniso = 1.0 / (3.14159265 * at * ab * denom * denom);          // anisotropic GGX
    float Diso   = hfDistributionGGX(NoHa, alpha);                         // the frozen isotropic D (match its param convention!)
    float Gv = hfGeometrySmith(NoVa, NoLa, roughness);
    float3 Fv = hfFresnelSchlick(VoHa, F0);                                // F0 = the SB3-iridescent F0 (already lerped) — correct
    float invDen = 1.0 / max(4.0 * NoVa * NoLa, 1e-4);
    float3 anisoSpec = (Daniso * Gv) * Fv * (invDen * NoLa);
    float3 isoSpec   = (Diso   * Gv) * Fv * (invDen * NoLa);
    rgb += aniso * (anisoSpec - isoSpec) * (f.lightColor.rgb * shadow);    // aniso=0 -> +0 -> identity
}
```
Confirm `hfDistributionGGX`'s param convention (it may take `alpha` or `roughness` — read pbr_core/the shader
and match it EXACTLY so isoSpec mirrors the base highlight). Confirm `N`, `V`, `shadow`, `roughness`,
`i.wtangent`, `f.lightDir`, `f.lightColor`, `F0` are the shader's exact names (copy verbatim). aniso=0 → the
whole `+=` is `0` → byte-identical to the base.

## Showcase `--sb4-aniso-shot` (Vulkan) / `--sb4-aniso` (Metal)
Clone the SB3 `--sb3-iridescence-shot` block. Set ONLY `fd.substrateParams2[0]=0.8f` (strong anisotropy;
clearcoat/sheen/iridescence all 0). A stretched, smeared highlight along the helmet's tangent (brushed-metal
streak). SCENE/PARAMS IDENTICAL in both renderers.

EXACT proof lines (fail loudly):
```
sb4-aniso: two-run BYTE-IDENTICAL
sb4-aniso: anisotropy=0 == lit_pbr_ibl BYTE-IDENTICAL (additivity control)
sb4-aniso: {nonBlackPixels:N, specStretch:S} anisotropy>0 stretches the highlight along the tangent
```
Assertions: (1) two renders byte-identical; (2) THE ADDITIVITY CONTROL — render the same scene through
lit_substrate with ALL substrateParams=0 AND substrateParams2=0 AND through the frozen lit_pbr_ibl, assert
byte-identical (proves anisotropy-at-zero identity → existing goldens safe); (3) anisotropy>0 differs from
anisotropy=0 (a real stretched highlight — nonBlack>0 + a measurable change vs the aniso=0 render). Register
`sb4_aniso` in verify.ps1 $Goldens (Flag `--sb4-aniso`) + `--sb4-aniso-shot` in $vkShots.

## Constraints (HARD)
- APPEND to lit_substrate.frag (SB1/SB2/SB3 blocks byte-frozen). Do NOT modify lit_pbr.frag/lit_pbr_ibl.frag/
  existing pbr_core functions/lit.vert/the material varying/frame_data.hlsli.
- Branch `fix-issue-11-sb4`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`; under
  Git Bash this can silently no-op — use the PowerShell tool with the absolute vcvars path).
- COMPLETION CRITERIA — do NOT commit until: `--sb4-aniso-shot` runs exit 0, all 3 proof lines print, AND the
  additivity control HOLDS (anisotropy=0 == lit_pbr_ibl byte-identical). (The CONTROLLER bakes the Metal golden,
  checks Metal two-run 0.0000, documents cross-vendor, eyeballs the PNG for a stretched tangent highlight, and
  re-bakes sb1/sb2/sb3 + ibl_helmet → all must stay 0.0000.)
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
