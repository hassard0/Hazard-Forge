# Slice US3 — Moving-camera history reprojection + disocclusion (Issue #20, flagship #20, 3rd slice)

The FSR2 hard part. US2 accumulated with a STATIC camera (identity reprojection). US3 makes the camera MOVE
(a deterministic orbit) and reprojects the full-res history through the camera motion so it doesn't ghost,
with a disocclusion fallback. Reuses `motion_blur.frag`'s analytic camera-motion reprojection verbatim. FLOAT
visresolve-bar slice.

## The shader — `shaders/tsr_resolve_reproject.frag.hlsl` (US2's tsr_resolve + the motion_blur reprojection)
Copy `tsr_resolve.frag.hlsl` (US2), then replace the identity history sample with a REPROJECTED one, copying
the `motion_blur.frag` reprojection EXACTLY:
- New inputs: `gDepth` (the G-buffer LINEAR depth RT, the same one SSR/SSAO/DoF/motion_blur sample) + the
  params `tanHalfFovY`, `aspect`, and `prevClip0..3` (`prevViewProj * inverse(curView)`, a column-major mat4 as
  4 float4s — IDENTICAL to motion_blur's MbParams). Keep the US2 `curTexel`/`histTexel`/`alpha`/`firstFrame`.
- The `HF_YS` Y-flip constant (1.0 Vulkan / -1.0 under HF_MSL_GEN) — copy motion_blur's exact #ifdef.
- Reproject (copy motion_blur lines ~62-71, ~129-131):
```hlsl
float linDepth = gDepth.Sample(gSmp, i.uv).r;          // current pixel linear depth
float2 ndc = i.uv * 2.0 - 1.0;
float3 vpos = float3(ndc.x * aspect * tanHalfFovY * linDepth, HF_YS * ndc.y * tanHalfFovY * linDepth, linDepth);
float4 prevClip = prevClip0*vpos.x + prevClip1*vpos.y + prevClip2*vpos.z + prevClip3;   // (vpos.w==1 folded into col3)
float2 prevUV = i.uv;                                   // default: identity (safe)
bool disoccluded = true;
if (prevClip.w > 1e-6) {
    float2 prevNdc = prevClip.xy / prevClip.w;
    prevUV = prevNdc * 0.5 + 0.5;  prevUV.y = (HF_YS < 0.0) ? 1.0 - prevUV.y : prevUV.y;   // match motion_blur's convention
    disoccluded = any(prevUV < 0.0) || any(prevUV > 1.0);   // history reprojected off-screen -> disocclusion
}
float3 history = gHistory.Sample(gHSmp, prevUV).rgb;
```
- **Disocclusion fallback:** if `disoccluded` (history off-screen) OR `firstFrame`, use the current sample
  unblended (`return float4(current, 1.0)` for firstFrame; else set the effective alpha to 1.0 locally so the
  reprojected-off-screen pixel takes the fresh current instead of a stale/garbage history). The neighborhood
  clamp (inherited from US2) suppresses the residual ghosting on reprojected-but-drifted history.
- Confirm the EXACT prevClip construction + the prevUV.y flip against motion_blur.frag — copy verbatim, do not
  improvise the Y convention (the Vulkan-Y-down / Metal-Y-up seam is the classic trap). Register the shader for
  SPIR-V + MSL (it's float/int32 — MSL-native, like tsr_resolve/motion_blur).

## Driver (`--us3-disocclusion-shot` Vulkan / `--us3-disocclusion` Metal) — US2 + an orbiting camera + depth + prevClip
Clone the US2 `--us2-tsr-shot` block. Changes:
- **Orbiting camera:** the camera position orbits the scene over the N=8 frames by a fixed per-frame angle
  (deterministic — `angle(f) = baseAngle + f * deltaAngle`, a pure function of frame index, NO time). A SMALL
  orbit (a few degrees total) so the scene stays framed + the history mostly stays on-screen.
- **Per frame:** render the scene at half-res (jittered) into `sceneLow` AND the linear depth into a half-res (or
  full-res — match what the reproject samples) depth RT. Compute `curView`/`curViewProj` for `angle(f)`;
  `prevView`/`prevViewProj` for `angle(f-1)`; `prevClip = prevViewProj * inverse(curView)` (copy the host-side
  computation from the motion_blur `--motion-blur-shot` showcase — find how it fills MbParams.prevClip0..3).
  Run `tsr_resolve_reproject` (bind sceneLow + history + gDepth, push the params incl. prevClip + tanHalfFov +
  aspect) into the write history RT; ping-pong. After 8 frames → post → capture (1280x720, the camera at angle(7)).
- SCENE/ORBIT/JITTER/N IDENTICAL in both renderers.

## Proof (reprojection beats no-reproject under motion — the make-or-break)
In the same showcase, also accumulate the SAME orbiting sequence WITHOUT reprojection (identity history UV — the
US2 path) and a native full-res render of the FINAL camera pose. Compare both accumulations to native:
```
us3-disocclusion: orbiting camera (8 frames), reprojection + disocclusion live
us3-disocclusion: two-run BYTE-IDENTICAL
us3-disocclusion: reproject beats no-reproject {reprojDiff:Dr, identityDiff:Di} Dr < Di (reprojection cuts ghosting under motion)
us3-disocclusion: provenance {frames:8, disocclusionFallback:true}
```
Assertions: (1) two runs byte-identical (deterministic orbit + Halton jitter, no rng/time); (2) `Dr < Di` where
`*Diff` = mean abs RGB diff to the native final-pose render — reprojection accumulation is CLOSER to native than
identity accumulation (which ghosts under camera motion). This is the make-or-break: it proves the reprojection
is correct + reduces ghosting. (3) coherent (shaded>0). Register `us3_disocclusion` in verify.ps1 $Goldens (Flag
`--us3-disocclusion`) + `--us3-disocclusion-shot` in $vkShots.

## Constraints (HARD)
- NEW tsr_resolve_reproject.frag.hlsl (fork of tsr_resolve + the motion_blur reprojection copied verbatim). Do
  NOT modify tsr_resolve.frag/taa_resolve.frag/taa.h/motion_blur.frag. NO RHI change. Do NOT touch any existing golden.
- Copy the motion_blur reprojection math + the HF_YS / prevUV.y convention VERBATIM — do NOT improvise the Y-flip.
- Branch `fix-issue-20-us3`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
  Register the golden in verify.ps1.
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`; under
  Git Bash this can silently no-op — use the PowerShell tool with the absolute vcvars path).
- COMPLETION CRITERIA — do NOT commit until: `--us3-disocclusion-shot` runs exit 0, the proof lines print,
  `Dr < Di` (reprojection beats no-reproject under motion — the make-or-break), two-run byte-identical. If
  Dr >= Di, the reprojection or the Y-flip is wrong (the reprojected history is worse than identity) — DEBUG the
  prevClip/prevUV.y convention against motion_blur before committing. (The CONTROLLER bakes the Metal golden,
  Metal two-run 0.0000, cross-vendor doc, eyeballs the PNG for a sharp non-ghosted moving-camera result.)
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
