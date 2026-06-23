# Slice SB5 — Subsurface scattering / wrap-diffuse (Issue #11, flagship #11, 5th slice)

APPEND SSS to `shaders/lit_substrate.frag.hlsl` (SB1-SB4 blocks BYTE-FROZEN — add with the other lobe blocks,
before emissive). A cheap wrap-diffuse subsurface approximation (skin/wax/marble/jade): the light wraps past
the terminator so the day/night line softens and glows into shadow. Gated by `f.substrateParams2.y` — SSS=0 is
an EXACT identity. FLOAT visresolve-bar slice. NO FrameData layout change.

## The math — an additive wrap-diffuse term (identity-at-zero by the gate)
The base Lambert diffuse is inside the frozen `hfCookTorrance` (uses `max(dot(N,L),0)`). SB5 ADDS the EXTRA
scattered light from wrapping (the difference between wrapped and clamped NoL), gated by `sss` → SSS=0 → +0 →
identity. Append after the base rgb accumulation:
```hlsl
{
    float sss = f.substrateParams2.y;
    const float wrap = 0.5;                              // wrap width (how far light bleeds past the terminator)
    float3 Ls = normalize(-f.lightDir.xyz);
    float rawNoL = dot(N, Ls);
    float NoLc   = max(rawNoL, 0.0);                     // the base Lambert cosine
    float wrappedNoL = saturate((rawNoL + wrap) / (1.0 + wrap));
    float extra = max(wrappedNoL - NoLc, 0.0);           // >=0: the extra light scattered past/at the terminator
    // Scatter the EXTRA diffuse through the (non-metallic) albedo, shadowed + AO'd like the base diffuse.
    rgb += sss * (1.0 - metallic) * albedo * (extra * (1.0 / 3.14159265)) * (f.lightColor.rgb * shadow) * ao;
}
```
sss=0 → `rgb += 0` → byte-identical to the base. The `1/π` keeps it consistent with the Lambert normalization
(`albedo/π` in hfCookTorrance). The effect: a soft waxy terminator that glows into shadow (the SSS look). Use
the shader's EXACT names (`N`, `albedo`, `metallic`, `ao`, `f.lightDir`, `f.lightColor`, `shadow`) — copy
verbatim. (Confirm the `shadow` variable name + that `albedo`/`metallic`/`ao` are in scope at the append point.)

## Showcase `--sb5-sss-shot` (Vulkan) / `--sb5-sss` (Metal)
Clone the SB4 `--sb4-aniso-shot` block. Set ONLY `fd.substrateParams2[1]=0.9f` (strong SSS; all other
substrateParams + substrateParams2 = 0). A soft, waxy/translucent terminator on the helmet (the shadow side
glows). SCENE/PARAMS IDENTICAL in both renderers.

EXACT proof lines (fail loudly):
```
sb5-sss: two-run BYTE-IDENTICAL
sb5-sss: sss=0 == lit_pbr_ibl BYTE-IDENTICAL (additivity control)
sb5-sss: {nonBlackPixels:N, terminatorSoftness:T} sss>0 softens + glows the terminator
```
Assertions: (1) two renders byte-identical; (2) THE ADDITIVITY CONTROL — render the same scene through
lit_substrate with ALL substrateParams=0 AND substrateParams2=0 AND through the frozen lit_pbr_ibl, assert
byte-identical (proves SSS-at-zero identity → existing goldens safe); (3) sss>0 differs from sss=0 (a real
softened/glowing terminator — nonBlack>0 + a measurable change, e.g. more lit pixels near the terminator).
Register `sb5_sss` in verify.ps1 $Goldens (Flag `--sb5-sss`) + `--sb5-sss-shot` in $vkShots.

## Constraints (HARD)
- APPEND to lit_substrate.frag (SB1-SB4 blocks byte-frozen). Do NOT modify lit_pbr.frag/lit_pbr_ibl.frag/
  existing pbr_core functions/lit.vert/the material varying/frame_data.hlsli.
- Branch `fix-issue-11-sb5`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`; under
  Git Bash this can silently no-op — use the PowerShell tool with the absolute vcvars path).
- COMPLETION CRITERIA — do NOT commit until: `--sb5-sss-shot` runs exit 0, all 3 proof lines print, AND the
  additivity control HOLDS (sss=0 == lit_pbr_ibl byte-identical). (The CONTROLLER bakes the Metal golden, checks
  Metal two-run 0.0000, documents cross-vendor, eyeballs the PNG for a soft glowing terminator, and re-bakes
  sb1-sb4 + ibl_helmet → all must stay 0.0000.)
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
