# Slice US2 — Temporal accumulation (history reprojection core) (Issue #20, flagship #20, 2nd slice)

The core temporal-supersampling step. NEW `shaders/tsr_resolve.frag.hlsl` = `taa_resolve.frag` generalized to a
RESOLUTION MISMATCH: accumulate N=8 jittered HALF-res frames into a FULL-res history → the upscaled image
sharpens frame-over-frame as the jitter sequence fills the high-res grid. FLOAT visresolve-bar slice. **Scope:
STATIC camera** (the jitter alone drives the supersampling; reprojection is identity here, exactly as the
existing TAA static shot). Moving-camera depth-reprojection + disocclusion is US3 (it needs the G-buffer depth).

## The new shader `shaders/tsr_resolve.frag.hlsl` (fork of taa_resolve.frag)
Copy `taa_resolve.frag.hlsl` verbatim, then change ONLY the texel handling so the current (half-res) and
history (full-res) are sampled with DIFFERENT texel sizes. The output runs at FULL res (bound to the full-res
history RT). Add a second texel to the params:
```hlsl
struct TsrParams { float2 curTexel; float2 histTexel; float alpha; float firstFrame; };  // curTexel=1/halfRes, histTexel=1/fullRes
```
- `gCurrent` = the half-res jittered HDR scene; sample at `i.uv` (the linear sampler bilinear-upscales it). Build
  the 3x3 neighborhood AABB using `curTexel` (the clamp source is the LOW-res image — FSR2 convention).
- `gHistory` = the FULL-res accumulated HDR; sample at `i.uv` (identity reprojection — static camera; the
  reprojection plumbing stays present for US3, like taa_resolve's comment).
- First frame (`firstFrame!=0`): return current unblended (defined accumulation start).
- Else: `clampedHistory = clamp(history, boxMin, boxMax); resolved = lerp(clampedHistory, current, alpha);`
  (the taa_resolve math verbatim). alpha = 1/(frameIndex+1) accumulation-style OR the taa steady ~0.1 — use a
  TRUE running-average `alpha = 1.0/(n+1)` so N jittered samples average into a clean supersample (document the
  choice). Register `tsr_resolve.frag.hlsl:ps` in the shader manifest (SPIR-V + MSL — it's int32/float, MSL-native).

## Driver (`--us2-tsr-shot` Vulkan / `--us2-tsr` Metal) — clone the US1 block + add the accumulation loop
- RTs: `sceneLow` (640x360 HDR) + TWO full-res history RTs `histA`/`histB` (1280x720 HDR, ping-pong).
- **Accumulation loop, N=8 frames, STATIC camera:** for `f` in 0..7:
  1. `scene` pass → `sceneLow` (640x360) with `taa::Jitter(f, 640, 360)` injected into the projection (a DIFFERENT
     sub-pixel offset each frame — this is what fills the high-res grid).
  2. `tsrResolve` pass → the write history RT (full-res): bind `sceneLow` (current) + the read history RT;
     push `{curTexel=1/(640,360), histTexel=1/(1280,720), alpha=1/(f+1), firstFrame=(f==0)}`; fullscreen draw.
  3. ping-pong (swap read/write history).
- After 8 frames: `post` pass samples the final history (full-res) → tonemap → capture (1280x720).
- SCENE/CAMERA/JITTER/N IDENTICAL in both renderers.

## Proof (the strong one: closer to native than naive bilinear)
Also render a NATIVE full-res reference (the scene at 1280x720, NO jitter, single frame, through post) and the
US1 naive-bilinear image (half-res → bilinear), in the same showcase, and compare both to native:
```
us2-tsr: internal 640x360, N=8 accumulate -> display 1280x720
us2-tsr: two-run BYTE-IDENTICAL
us2-tsr: closer to native than naive {naiveDiff:Dn, tsrDiff:Dt} Dt < Dn (temporal supersampling sharper)
us2-tsr: provenance {frames:8, histPingPong:true}
```
Assertions: (1) two full runs byte-identical (determinism — Halton jitter by frame#, no rng); (2) `tsrDiff < naiveDiff`
where `*Diff` = mean abs RGB diff to the native full-res reference (the TSR accumulation is measurably closer to
native than US1's bilinear — the make-or-break quality proof); (3) the accumulated image is coherent (shaded>0).
Register `us2_tsr` in verify.ps1 $Goldens (Flag `--us2-tsr`) + `--us2-tsr-shot` in $vkShots.

## Constraints (HARD)
- NEW tsr_resolve.frag.hlsl (fork of taa_resolve — do NOT modify taa_resolve.frag/taa.h; reuse taa::Jitter verbatim).
  NO RHI change. Do NOT touch any existing golden or shader other than adding the new one.
- STATIC camera only (US2 = jitter-driven supersampling). NO depth-based reprojection (US3). Reprojection UV is
  identity (sample history at i.uv).
- Branch `fix-issue-20-us2`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
  Register the golden in verify.ps1.
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`; under
  Git Bash this can silently no-op — use the PowerShell tool with the absolute vcvars path).
- COMPLETION CRITERIA — do NOT commit until: `--us2-tsr-shot` runs exit 0, the proof lines print, `tsrDiff < naiveDiff`
  (US2 is closer to native than naive bilinear — the quality gain is real), two-run byte-identical. (The CONTROLLER
  bakes the Metal golden, checks Metal two-run 0.0000, documents cross-vendor, eyeballs the PNG for a sharper image
  than US1's us1_upscale.png.) If tsrDiff >= naiveDiff, the accumulation/jitter is wrong — fix before committing.
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
