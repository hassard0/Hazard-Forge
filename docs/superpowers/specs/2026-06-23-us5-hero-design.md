# Slice US5 — The hero money-shot (Issue #20, flagship #20, 5th/final slice)

The capstone. Composes the full TSR chain — US3's moving-camera reprojection accumulation + US4's CAS sharpen
— on a deterministic ORBITING camera at a true 2x upscale, the money-shot showing the quality gain vs naive
US1 bilinear. COMPLETES flagship #20. NO new shader (reuses tsr_resolve_reproject.frag + tsr_pack_histdepth.frag
+ cas.frag + post). FLOAT visresolve-bar slice.

## What to do (driver only — composes US3 + US4)
Clone the US3 `--us3-disocclusion-shot` block (which already has the orbiting camera + reproject accumulation +
the tsr_pack_histdepth pass) → `--us5-upscale-hero-shot`, and ADD the US4 CAS sharpen pass at the end. Resolution
= a true 2x HERO: internal `960x540`, display/capture `1920x1080` (1080p-from-540p). **If 1920x1080 headless
capture is too heavy/slow on the Mac bake, fall back to internal 640x360 → display 1280x720 (the proven 2x ratio)
— the RATIO is the hero story, document whichever you use.**
Pipeline per the N=8 orbit: `scene(internal, jittered) + depth → tsr_pack_histdepth → tsr_resolve_reproject
(display-res HDR history, reproject) → ping-pong ×8 → post/tonemap (display-res LDR) → cas (display-res,
sharpness 0.6) → capture (display-res, camera at the final orbit pose)`. SCENE/ORBIT/JITTER/N/sharpness
IDENTICAL in both renderers.

## Proof (the full upscaler beats naive bilinear under motion — the make-or-break)
Also render a NATIVE display-res reference (no jitter, single frame, final pose) and the US1 naive-bilinear
image (internal → bilinear, final pose); compute mean-abs-RGB-diff of each to native:
```
us5-upscale-hero: internal 960x540 -> display 1920x1080 (2x), orbiting camera, full TSR chain (reproject+accumulate+CAS)
us5-upscale-hero: two-run BYTE-IDENTICAL
us5-upscale-hero: TSR beats naive {naiveDiff:Dn, tsrDiff:Dt} Dt < Dn (full upscaler closer to native than bilinear)
us5-upscale-hero: provenance {frames:8, reproject:true, sharpen:true, ratio:4.00}
```
Assertions: (1) two renders byte-identical (deterministic orbit + Halton jitter, no rng); (2) `Dt < Dn` —
the full-chain TSR upscale is closer to the native display-res render than US1's naive bilinear (combines
temporal supersampling + reprojection + sharpen). This is the make-or-break flagship result. (3) provenance +
coherent. Register `us5_upscale_hero` in verify.ps1 $Goldens (Flag `--us5-upscale-hero`) + `--us5-upscale-hero-shot`
in $vkShots.

HONEST CAVEAT (document in the showcase comment + the commit): the US3 reprojection is far more effective on
Vulkan than Metal (the thin-Metal-margin cross-vendor difference). So on Metal the `Dt < Dn` win comes mostly
from the temporal supersampling + CAS sharpen rather than the reprojection; on Vulkan all three contribute. The
`Dt < Dn` assertion must still hold on the BAKING backend (Metal) — if it doesn't (the moving-camera reprojection
weakness drags Metal below naive), reduce the orbit angle so the supersampling+sharpen win dominates, OR note it.

## Constraints (HARD)
- NO new shader (reuse tsr_resolve_reproject.frag / tsr_pack_histdepth.frag / cas.frag / post VERBATIM). Do NOT
  modify any existing shader/taa.h. NO RHI change. Do NOT touch any existing golden.
- Branch `fix-issue-20-us5`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
  Register the golden in verify.ps1.
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`; under
  Git Bash this can silently no-op — use the PowerShell tool with the absolute vcvars path).
- COMPLETION CRITERIA — do NOT commit until: `--us5-upscale-hero-shot` runs exit 0, the proof lines print,
  `Dt < Dn` (the full upscaler beats naive bilinear), two-run byte-identical. (The CONTROLLER bakes the Metal
  golden, Metal two-run 0.0000, cross-vendor doc, confirms `Dt < Dn` holds on Metal too, eyeballs the PNG for a
  crisp 2x-upscaled hero. If the Metal Dt>=Dn, tune the orbit smaller so supersampling+sharpen dominate.) This
  slice COMPLETES flagship #20 — after merge, the controller closes issue #20.
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
