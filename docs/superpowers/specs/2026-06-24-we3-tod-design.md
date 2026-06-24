# Slice WE3 — Deterministic time-of-day sun + sky ramp (Flagship #27 WEATHER, 3rd slice)

The day cycle: a **deterministic integer time-of-day** — `SunSky(frame)` returns a Q16.16 **sun direction** (the sun
sweeps an arc as the frame advances) + a **sky-color ramp** (midnight → dawn → noon → dusk RGB), bit-identical
cross-platform (NO runtime trig — the baked-sine-LUT discipline). The sun/sky are a pure function of the frame, so
two peers see the byte-identical golden hour. Builds on WE1/WE2. **STRICT-integer zero-diff cross-backend golden** (a
CPU-rendered integer day-cycle strip plot). Feeds WE4/WE5's `fd.lightDir`/sky in the render slices.

## The addition — `engine/weather/weather.h` (APPEND-ONLY after the WE2 block)
Add to `hf::weather` (do NOT modify WE1/WE2; `#include "foliage/foliage.h"` to reuse the committed `kFoliageWind16`
256-entry full-wave sine LUT read-only — `kFoliageWind16[i] == round(32767·sin(2πi/256))`):
- `inline constexpr uint32_t kDayFrames = 1440;` (frames per full day cycle).
- A committed **sky-color ramp** `static const FxVec3 kSkyRamp[4]` (Q16.16 RGB in `[0,kOne]`) — the day keys at
  phase 0/¼/½/¾: **midnight** (dark blue, e.g. `{kOne/16, kOne/12, kOne/6}`), **dawn** (warm orange, e.g.
  `{kOne*8/10, kOne*4/10, kOne*3/10}`), **noon** (sky blue, e.g. `{kOne*4/10, kOne*6/10, kOne}`), **dusk** (warm
  orange like dawn). (Pick clean Q16.16 literals; the exact hues are art, the determinism is the point.)
- `struct SunSkyState { FxVec3 sunDir; FxVec3 skyColor; fx sunElev; };`
- `inline SunSkyState SunSky(uint32_t frame)`:
  - `uint32_t phase = frame % kDayFrames;` `uint32_t tabIdx = (uint32_t)((uint64_t)phase * 256 / kDayFrames) & 255;`
    (map the day phase to a 0..255 sine-table index).
  - **Sun direction** from the LUT: `fx sinE = kFoliageWind16[tabIdx];` (the elevation sine over the day — `+` near
    noon, `−` at night) and `fx cosE = kFoliageWind16[(tabIdx + 64) & 255];` (the cosine, the east→west sweep —
    quarter-phase shift). `sunDir = { cosE, sinE, 0 }` (a Q16.16 direction arcing across the sky; a small fixed `z`
    tilt is fine). `sunElev = sinE`.
  - **Sky color** by lerping `kSkyRamp` across the day: `seg = phase * 4 / kDayFrames` (0..3), `t` = the Q16.16
    fractional position within the segment; `skyColor = lerp(kSkyRamp[seg], kSkyRamp[(seg+1)&3], t)` per-channel
    with `fxmul` (wrapping back to midnight after dusk). All Q16.16.
  - Pure integer, NO `<cmath>`, NO float, NO clock/RNG.

## CPU test — extend `tests/weather_test.cpp` (add a WE3 section, keep WE1/WE2)
Assertions: (1) **replay-stable** — `SunSky(frame)` identical across calls; (2) **sun moves** — `SunSky(F).sunDir !=
SunSky(F + kDayFrames/8).sunDir` (the sun arcs as the day advances); (3) **midnight vs noon distinct** —
`SunSky(0)` (midnight: low/negative `sunElev`, dark sky) differs from `SunSky(kDayFrames/2)` (noon: high `sunElev`,
bright sky) — assert `sunElev` is higher at noon than midnight AND the sky colors differ; (4) **bounds** — every
`skyColor` channel `∈ [0, kOne]`; (5) **periodic** — `SunSky(frame) == SunSky(frame + kDayFrames)` (the day loops).
Print + `weather_test: ALL CHECKS PASSED`.

## Showcase — `--we3-tod-shot` (Vulkan, main.cpp) + `--we3-tod` (Metal, visual_test.mm)
A **2D day-cycle strip plot** (256×256, pure integer): the X axis is time-of-day across one full day
(`frame = x * kDayFrames / 256`); for each column, draw the **sky color** as a vertical gradient band (`SkyColor`
filling the column, or a top-sky/bottom-horizon split) AND overlay the **sun elevation** as a curve (a bright pixel
at `y = horizon - sunElev·scale` per column — the sun arcing up over noon and below the horizon at night). All
pixel math pure integer (map `FxVec3` Q16.16 channels to `[0,255]` by a pure shift; the sun-curve Y by integer
scale). SAME params IN BOTH renderers → byte-identical cross-backend by construction.

## Proof (STRICT integer — zero-diff cross-backend)
```
we3-tod: integer time-of-day sun + sky (dayFrames=1440)
we3-tod: two-run BYTE-IDENTICAL
we3-tod: sun arcs -> midnight vs noon distinct {midElev:<m>, noonElev:<n>} n > m
we3-tod: sky ramp -> dawn/noon/dusk distinct {ok:true}
we3-tod: periodic -> SunSky(f) == SunSky(f+dayFrames) {ok:true}
```
Assertions: (1) two runs byte-identical; (2) noon `sunElev` > midnight `sunElev` (the sun arcs up); (3) the
dawn/noon/dusk sky colors are distinct; (4) the day is periodic. Register `we3_tod` in verify.ps1 $Goldens (Flag
`--we3-tod`) + `--we3-tod-shot` in $vkShots, mirroring `we2_precip`.

## Constraints (HARD)
- APPEND-ONLY to engine/weather/weather.h (do NOT modify WE1/WE2) + extend tests/weather_test.cpp + the showcase
  blocks + verify.ps1 registration. Reuse fpx.h/foliage.h (`kFoliageWind16`) READ-ONLY. Do NOT modify foliage.h/
  clouds.h/procterrain.h/any existing shader/golden. NO new RHI, NO new shader. Pure-CPU integer.
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (the controller bakes Metal and
  requires the Metal image == the Vulkan image byte-for-byte). The plot MUST be pure integer (no float pixel math).
  Do NOT route through a GPU float raster.
- Branch `fix-weather-we3`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows + the test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle weather_test'`
  Run weather_test.exe (ALL CHECKS PASSED) + hello_triangle.exe --we3-tod-shot <out.bmp> (exit 0 + proof lines).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, weather_test passes, `--we3-tod-shot` exits 0, the
  proof lines print, two-run byte-identical, the sun-arcs (midnight<noon elev), the sky-ramp distinct, the
  periodicity holds, the plot is pure-integer. Commit message via a temp file + `git commit -F` (use the Bash tool
  heredoc). Commit to the branch and STOP. Report: commit hash, the proof output, image path, confirmation both
  renderers use identical params, and that the plot is pure-integer. (The CONTROLLER bakes the Metal golden,
  confirms zero-diff cross-backend, eyeballs the day-cycle strip.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
