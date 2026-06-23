# Slice US4 — RCAS sharpening on the accumulated output (Issue #20, flagship #20, 4th slice)

Recovers the slight softness temporal accumulation introduces, via the EXISTING `shaders/cas.frag.hlsl`
(RCAS/CAS contrast-adaptive sharpen) applied as a fullscreen pass on the US2-accumulated + tonemapped output.
NO new shader (cas.frag reused VERBATIM). FLOAT visresolve-bar slice. Builds on the US2 STATIC-camera
accumulation (the clean path) — NOT US3's reprojection.

## What to do (driver only — no new shader)
Clone the US2 `--us2-tsr-shot` block → `--us4-tsr-sharpen-shot`. Same N=8 static-camera jittered half-res
accumulation into the full-res history (US2's tsr_resolve), then `post`/tonemap to a full-res LDR RT, then a
fullscreen `cas` pass (the existing cas.frag, sharpness>0, e.g. 0.6) into the swapchain → capture. So:
`scene(640x360, jittered)×8 → tsrResolve(1280x720 HDR history) → post(1280x720 LDR) → cas(1280x720, sharpness) → capture`.
- Bind the tonemapped LDR RT as cas.frag's `gScene` (t0/s0); push `CasParams.sharpness.x = 0.6`. (Read the
  existing `--cas-shot` showcase for the exact cas pipeline + push-constant wiring + how it composes with post —
  cas.frag expects the RESOLVED SDR/tonemapped LDR, so it runs AFTER tonemap.)
- SCENE/JITTER/N/sharpness IDENTICAL in both renderers.

## Proof (the free no-op proof + the sharpen-is-real proof)
```
us4-tsr-sharpen: N=8 accumulate + CAS sharpen (sharpness 0.6)
us4-tsr-sharpen: two-run BYTE-IDENTICAL
us4-tsr-sharpen: sharpness=0 == unsharpened BYTE-IDENTICAL (CAS no-op pass-through)
us4-tsr-sharpen: sharpness>0 differs from sharpness=0 {edgeGain:E} (sharpen recovers accumulation softness)
```
Assertions: (1) two renders byte-identical (determinism); (2) THE FREE NO-OP PROOF — render the SAME pipeline
with cas sharpness=0 AND assert it's byte-identical to the unsharpened US2 accumulation (cas at 0 = exact
pass-through, already true for cas.frag); (3) sharpness=0.6 differs from sharpness=0 (a real edge gain — the
sharpen is doing something, nonBlack/edge-energy higher). Register `us4_tsr_sharpen` in verify.ps1 $Goldens
(Flag `--us4-tsr-sharpen`) + `--us4-tsr-sharpen-shot` in $vkShots.

## Constraints (HARD)
- NO new shader (reuse cas.frag + tsr_resolve.frag + post VERBATIM). Do NOT modify cas.frag/tsr_resolve.frag/
  taa_resolve.frag/taa.h. NO RHI change. STATIC camera (US2 path — no reprojection). Do NOT touch any existing golden.
- Branch `fix-issue-20-us4`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
  Register the golden in verify.ps1.
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`; under
  Git Bash this can silently no-op — use the PowerShell tool with the absolute vcvars path).
- COMPLETION CRITERIA — do NOT commit until: `--us4-tsr-sharpen-shot` runs exit 0, the proof lines print, the
  sharpness=0 no-op control is byte-identical to the unsharpened accumulation, sharpness>0 differs, two-run
  byte-identical. (The CONTROLLER bakes the Metal golden, Metal two-run 0.0000, cross-vendor doc, eyeballs the
  PNG for a crisper image than us2_tsr.png.)
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
