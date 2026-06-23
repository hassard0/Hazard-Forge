# Slice SB1 — Clearcoat (Issue #11, flagship #11 beachhead, Substrate-lite layered materials)

The beachhead of a 6-slice layered-material flagship. NEW `shaders/lit_substrate.frag.hlsl` = a byte-copy of
the frozen `shaders/lit_pbr_ibl.frag.hlsl` main()+bindings, PLUS additive BRDF lobes gated by new FrameData
scalars. SB1 adds the CLEARCOAT lobe (a 2nd GGX specular over the base, like glTF KHR_materials_clearcoat).
This is a FLOAT visresolve-bar slice (Metal two-run 0.0000 + provenance + documented cross-vendor mean, NOT
the strict-zero deterministic-sim bar).

## Param channel (the load-bearing decision — do EXACTLY this)
Add to `shaders/frame_data.hlsli` AFTER `iblParams` (do NOT touch lit.vert / the push constant / the
`material` varying — it's shared by 28 shaders):
```hlsl
float4 substrateParams;   // x=clearcoat strength, y=clearcoatRoughness, z=sheen, w=iridescence
float4 substrateParams2;  // x=anisotropy, y=SSS, z=tangentRotate, w=reserved
```
Mirror BYTE-FOR-BYTE in the 3 CPU FrameData structs: `samples/hello_triangle/main.cpp` (the canonical struct
+ its `static_assert(sizeof(FrameData)==592...)` → bump to 624), `metal_headless/visual_test.mm`,
`mac_window/main.mm`. Append at the TAIL (after iblParams) so every existing field's byte offset is unchanged
→ existing goldens stay byte-identical (the #33 lesson; do NOT reorder). 624 ≤ kFrameUboSize 1024. Confirm the
existing FrameData tail order before editing (it may have prevViewProj + iblParams — append after the LAST field).

## Golden-safety guarantee (default param=0 == base PBR identity)
Every existing showcase leaves substrateParams=0 and their shaders don't reference the new fields → byte-
untouched. lit_substrate.frag with clearcoat=0 must be EXACTLY lit_pbr_ibl.frag: `rgb *= 1.0` and both
clearcoat `+=` terms are `+0.0`. THIS IS THE MAKE-OR-BREAK — proven by proof line (2) below.

## The clearcoat math (lit_substrate.frag.hlsl, after the base direct+IBL rgb accumulation, before emissive)
Read at top of main(): `float clearcoat = f.substrateParams.x; float ccRough = clamp(f.substrateParams.y, 0.05, 1.0);`
Clearcoat is a smooth dielectric film → use the GEOMETRIC normal Ng (the interpolated world normal, NOT the
normal-mapped N), fixed F0=0.04, its own Fresnel + roughness. Reuse the pbr_core helpers
(hfDistributionGGX/hfGeometrySmith — confirm their exact signatures by reading pbr_core.hlsli):
```hlsl
float3 Ng = normalize(i.wnormal);                    // geometric normal (pre-normal-map)
float3 Lc = normalize(-f.lightDir.xyz);
float3 Hc = normalize(Lc + V);
float NoLc = max(dot(Ng, Lc), 0.0);
float NoHc = max(dot(Ng, Hc), 0.0);
float NoVc = max(dot(Ng, V), 1e-4);
float VoHc = max(dot(V, Hc), 0.0);
float ccAlpha = ccRough * ccRough;
float Dc = hfDistributionGGX(NoHc, ccAlpha);          // match the helper's actual param convention
float Gc = hfGeometrySmith(NoVc, NoLc, ccRough);
float Fc = 0.04 + 0.96 * pow(saturate(1.0 - VoHc), 5.0);
float ccSpec = (Dc * Gc * Fc) / max(4.0 * NoVc * NoLc, 1e-4) * NoLc;
rgb *= (1.0 - clearcoat * Fc);                        // attenuate what's under the film
rgb += clearcoat * ccSpec * (f.lightColor.rgb * shadow);   // direct clearcoat lobe (use the scene's shadow term)
float3 Rc = reflect(-V, Ng);
float3 ccEnv = gEnv.SampleLevel(gEnvSmp, EquirectUV(Rc), ccRough * f.iblParams.x).rgb;  // sharp dielectric env reflection
rgb += clearcoat * Fc * ccEnv * f.iblParams.y;
```
Confirm the exact base variable names (rgb / V / shadow / i.wnormal / gEnv / EquirectUV / f.iblParams) by
reading lit_pbr_ibl.frag — copy them verbatim. clearcoat=0 → rgb*=1, both += are +0 → identity.

## Showcase `--sb1-clearcoat-shot` (Vulkan) / `--sb1-clearcoat` (Metal)
Copy the `--ibl-shot` showcase block (the HDR-IBL helmet, `iblShotPath`, in samples/hello_triangle/main.cpp)
and its Metal mirror `RunIblShowcase`/`--ibl` (visual_test.mm). Swap the helmet's IBL fragment pipeline to
`lit_substrate.frag.hlsl.spv` (Vulkan) / the generated MSL (Metal). Set `fd.substrateParams[0]=0.9f;
fd.substrateParams[1]=0.08f;` (a glossy clear lacquer). Same scene/camera/light/env as --ibl. Register
`lit_substrate.frag.hlsl:ps` in `samples/hello_triangle/CMakeLists.txt` (auto SPIR-V + MSL). SCENE/PARAMS
IDENTICAL in both renderers.

EXACT proof lines (fail loudly):
```
sb1-clearcoat: two-run BYTE-IDENTICAL
sb1-clearcoat: clearcoat=0 == lit_pbr_ibl BYTE-IDENTICAL (additivity control)
sb1-clearcoat: {nonBlackPixels:N, specPeak:L} clearcoat>0 adds a second highlight
```
Assertions: (1) two renders of the showcase byte-identical (determinism); (2) THE ADDITIVITY CONTROL — render
the SAME scene through lit_substrate with substrateParams[0]=0 AND through the frozen lit_pbr_ibl, assert the
two images are byte-identical (proves identity-at-zero → existing goldens safe); (3) the clearcoat>0 render
differs from the clearcoat=0 render (a real added highlight: nonBlack>0, a brighter spec peak). Register
`sb1_clearcoat` in verify.ps1 $Goldens (Flag `--sb1-clearcoat`) + `--sb1-clearcoat-shot` in $vkShots.

## Constraints (HARD)
- NEW lit_substrate.frag.hlsl (additive sibling). DO NOT modify lit_pbr.frag/lit_pbr_ibl.frag/pbr_core.hlsli's
  EXISTING functions (you MAY add new hfSubstrate*() helpers to pbr_core.hlsli — additive only). DO NOT modify
  lit.vert or the material push constant or the material varying. frame_data.hlsli change is additive at the tail.
- The FrameData CPU mirrors MUST match frame_data.hlsli byte-for-byte (the 3 structs + the static_assert).
- Branch `fix-issue-11-sb1`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
  Register the golden in verify.ps1.
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`;
  under Git Bash this can silently no-op — use the PowerShell tool with the absolute vcvars path).
- COMPLETION CRITERIA — do NOT commit until: `--sb1-clearcoat-shot` runs exit 0, all 3 proof lines print, AND
  the additivity control (clearcoat=0 == lit_pbr_ibl byte-identical) HOLDS (this is the make-or-break: if it
  fails, an existing golden would be at risk — fix before committing). ALSO sanity: render an existing IBL
  showcase (`--ibl-shot`) after your FrameData change and confirm it still works (the FrameData layout change
  is the #33-style risk — the controller will Mac-bake + confirm ibl_helmet stays 0.0000). 
  (The CONTROLLER bakes the Metal golden, checks Metal two-run 0.0000, documents the cross-vendor mean,
  eyeballs the PNG for a visible clear-lacquer highlight, AND re-bakes ibl_helmet to confirm 0.0000 invariance.)
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
