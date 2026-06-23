# Slice US5 — The hero money-shot (Issue #20, flagship #20, 5th/final slice)

The capstone. The STATIC-camera 2x hero: temporal supersampling (N=8 Halton-jittered accumulation via US2's
NON-reprojecting `tsr_resolve.frag`) + US4's CAS sharpen, at a true 2x upscale — the money-shot showing the
quality gain vs naive US1 bilinear. COMPLETES flagship #20. NO new shader (reuses tsr_resolve.frag + cas.frag
+ post). FLOAT visresolve-bar slice.

## DESIGN PIVOT — why static, not moving-camera reproject
The first US5 implementation composed US3's moving-camera reprojection accumulation at 2x. On Vulkan it passed
(Dt 1.12 < Dn 2.19), but on the Metal bake it FAILED: Dt 3.24 >= Dn 2.77 — the reprojection accumulation was
WORSE than a single naive-bilinear frame on Metal. Cutting the orbit from 0.040 to 0.015 rad/frame moved the
Metal Dt by only ~0.01, proving the Metal error is NOT motion-proportional ghosting but a SYSTEMATIC sub-pixel
reprojection misalignment: the documented US3 packed-depth-precision residual (linear depth packed into an
RGBA16F alpha channel loses precision; the no-RHI-change 2-sampled-texture cap is exactly why depth is packed
there). It cannot be fixed without an RHI/format change, which is forbidden.

So US5 is the PROVEN-CLEAN static path: the US4 recipe (static camera + `tsr_resolve.frag` + CAS) at US5's
true 2x hero resolution. US2 and US4 both passed cleanly cross-backend (Metal two-run 0.0000 + Dt<Dn on both
backends), so the static path is guaranteed to work on Metal. The moving-camera reprojection remains proven
separately in US3's own merged golden — US5 does not re-prove it and must not ship a Metal-failing assertion.

## What to do (driver only — US4 at the 2x hero resolution)
Make US5 the US4 `--us4-tsr-sharpen-shot` recipe (static camera + N=8 jittered accumulation via `tsr_resolve.frag`
+ CAS sharpen) but at internal `960x540` → display/capture `1920x1080` (US4 is 640x360 → 1280x720). DROP the
moving-camera plumbing (orbit, per-frame prevClip, `tsr_pack_histdepth` pass, depth RT, `tsr_resolve_reproject`).
The whole chain runs in render targets and the final 1920x1080 BGRA RT is read back via ReadRenderTarget (display
res is swapchain-independent). Camera is STATIC at the base eye pose (baseEye/center, frame 0 unjittered for the
native + naive references — exactly like US4 at its base pose).
Pipeline per the N=8 static accumulation: `scene(internal, jittered) → tsr_resolve (display-res HDR history,
identity UV — pure temporal supersample) → ping-pong ×8 → post/tonemap (display-res LDR) → cas (display-res,
sharpness 0.6) → read back the display-res CAS RT`. SCENE/JITTER/N/sharpness/resolutions IDENTICAL in both
renderers.

## Proof (the static upscaler beats naive bilinear)
Also render a NATIVE display-res reference (no jitter, single frame, base pose) and the US1 naive-bilinear
image (internal → bilinear, base pose); compute mean-abs-RGB-diff of each to native:
```
us5-upscale-hero: internal 960x540 -> display 1920x1080 (2x), STATIC camera, temporal supersample (tsr_resolve) + CAS sharpen
us5-upscale-hero: two-run BYTE-IDENTICAL
us5-upscale-hero: TSR beats naive {naiveDiff:Dn, tsrDiff:Dt} Dt < Dn (static upscaler closer to native than bilinear)
us5-upscale-hero: provenance {frames:8, reproject:false, sharpen:true, ratio:4.00}
```
Assertions: (1) two renders byte-identical (deterministic Halton jitter, no rng); (2) `Dt < Dn` — the static
TSR upscale (supersample + CAS) is closer to the native display-res render than US1's naive bilinear. US2/US4
proved static accumulation beats naive bilinear ~2.5x on BOTH backends, so this holds on Metal too. (3)
provenance (`reproject:false` — honest) + coherent. Register `us5_upscale_hero` in verify.ps1 $Goldens (Flag
`--us5-upscale-hero`) + `--us5-upscale-hero-shot` in $vkShots.

Vulkan result (this machine): `naiveDiff:2.1858, tsrDiff:0.8456` → Dt < Dn (~2.6x closer), two-run byte-identical.

## Constraints (HARD)
- NO new shader (reuse tsr_resolve.frag / cas.frag / post VERBATIM). Do NOT modify any existing shader/taa.h.
  NO RHI change. Do NOT touch any existing golden. Do NOT modify US1/US2/US3/US4 or any other showcase.
- Branch `fix-issue-20-us5`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
  Register the golden in verify.ps1.
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`; under
  Git Bash this can silently no-op — use the PowerShell tool with the absolute vcvars path).
- COMPLETION CRITERIA — do NOT commit until: `--us5-upscale-hero-shot` runs exit 0, the proof lines print with
  `reproject:false`, `Dt < Dn`, two-run byte-identical. (The CONTROLLER bakes the Metal golden, confirms Metal
  two-run 0.0000 + `Dt < Dn` on Metal, eyeballs the PNG.) This slice COMPLETES flagship #20.
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
