# Slice US1 — Jittered low-res render + naive bilinear upscale (Issue #20, flagship #20 beachhead)

The beachhead of a 5-slice temporal-upscaling (TSR/FSR2-class) flagship. US1 establishes the low-res→high-res
PATH: render the scene at HALF resolution with the existing Halton sub-pixel jitter, bilinear-upscale to FULL
resolution (the free linear-sampler upscale), tonemap, capture. NO new shader — 100% existing pipeline. This is
deliberately the NAIVE upscale (no temporal accumulation yet — that's US2) so US2's quality gain is measurable
against it. FLOAT visresolve-bar slice.

## Why no RHI/shader change is needed (verified by scout)
- `BeginRenderPass` auto-sizes the viewport/scissor from the BOUND target's extent (vulkan_command_buffer.cpp:80,102)
  → a pass bound to a half-res RT rasterizes at half-res; a fullscreen pass bound to the full-res swapchain runs
  full-res.
- The default sampler is `VK_FILTER_LINEAR` (vulkan_device.cpp:339) → sampling the half-res RT from the full-res
  `post` pass is FREE bilinear upscaling.
- The existing TAA (`engine/render/taa.h::Jitter`, Halton(2,3), no rng/time) provides the deterministic sub-pixel jitter.

## Implementation (driver code only — clone the TAA showcase)
Clone the `if (taaShotPath) { … }` block at `samples/hello_triangle/main.cpp:~87414` into a new
`if (us1UpscaleShotPath)` block; mirror `RunTaaShowcase` (visual_test.mm:~7195) as `RunUs1UpscaleShowcase`.
- **Resolutions:** full/display `W=1280, H=720` (capture); half/internal `hw=640, hh=360`.
- **RTs:** `auto sceneLow = device->CreateRenderTarget(hw, hh, kHdr);` (half-res jittered HDR) + the shadow map (2048).
- **Passes (one jittered frame, frame index 0):**
  1. `shadow` — depth-only into the shadow map (existing recordShadow).
  2. `scene` — into `sceneLow` (HDR); viewport AUTO-sizes to 640×360 from sceneLow's extent. FrameData `vp` = the
     jittered projection built EXACTLY as TAA does, but jitter computed against the HALF-res dims:
     `taa::Jitter(0, (int)hw, (int)hh)` injected into `baseProj.m[2*4+0]+=j.x; baseProj.m[2*4+1]+=j.y;`
     (the TAA convention at main.cpp:~87639), `baseProj = Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f)`,
     `aspect = hw/hh` (== W/H). Body = existing recordScene (the settled-sphere-pyramid scene the TAA shot uses).
  3. `post` — fullscreen into the swapchain (1280×720): `cmd.BindTexture(*sceneLow); cmd.Draw(3);` — the existing
     `post.frag` samples sceneLow via the LINEAR sampler (bilinear-upscales 640×360→1280×720) → ACES/grade/grain/
     vignette tonemap. `device->CaptureNextFrame()` before execute. Capture is full-res.
- **Arg parse:** `--us1-upscale-shot <out.bmp>` next to `--taa-shot` (main.cpp:~2187; own parse loop if MSVC C1061);
  Metal `--us1-upscale <out.png>` in visual_test.mm main(). SCENE/RESOLUTIONS IDENTICAL in both renderers.

## Showcase proof lines (printf; matching the repo provenance style)
```
us1-upscale: internal 640x360 -> display 1280x720 (2x bilinear), jitter Halton(2,3) idx 0
us1-upscale: two-run BYTE-IDENTICAL
us1-upscale: {internalPx:230400, displayPx:921600, ratio:4.00} provenance (resolution mismatch is real)
us1-upscale: wrote <path> (1280x720) — naive upscale, <N> bodies
```
Assertions: (1) two renders byte-identical (determinism); (2) the internal/display pixel counts + the 4.00 area
ratio (proves the half-res render → full-res output is real, not a full-res render); (3) shaded/coherent (the
scene rendered, non-black). Register `us1_upscale` in verify.ps1 $Goldens (Flag `--us1-upscale`) +
`--us1-upscale-shot` in $vkShots.

## Constraints (HARD)
- NO new shader, NO RHI change (US1 is driver code only — reuse lit/instanced/shadow/sky/post verbatim). Do NOT
  modify taa_resolve.frag/taa.h (US2 forks those). Do NOT touch any existing golden.
- US1 must NOT do accumulation/history/reprojection (those are US2). US1 = the single-frame naive bilinear baseline.
- Branch `fix-issue-20-us1`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
  Register the golden in verify.ps1.
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`; under
  Git Bash this can silently no-op — use the PowerShell tool with the absolute vcvars path).
- COMPLETION CRITERIA — do NOT commit until: `--us1-upscale-shot` runs exit 0, the proof lines print, the
  resolution-mismatch ratio is 4.00, two-run byte-identical. (The CONTROLLER bakes the Metal golden, checks Metal
  two-run 0.0000, documents the cross-vendor mean, eyeballs the PNG for a coherent — intentionally soft — upscaled
  scene.) NO additivity control (US1 isn't a layered-on-base feature) but the scene must render coherently.
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
