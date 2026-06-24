# Slice FO1 — Deterministic integer wind field (Issue #21, flagship #25 FOLIAGE, beachhead)

The irreducible primitive: a **deterministic integer WIND FIELD** — `WindBend(wind, pos, frame)` returns a Q16.16
bend angle from a sum of host-baked sine "gust" waves over `(position, frame#)`, bit-identical CPU↔Vulkan↔Metal
BY CONSTRUCTION (NO runtime `sin`/`cos`/`<cmath>` — a committed `int16` LUT indexed by an integer phase, the audio
`kSineTable` discipline). This is the moat: UE5/SpeedTree wind is float/non-deterministic; this wind is a pure
function of position + frame, so two peers grow the byte-identical swaying meadow. Establishes
`engine/foliage/foliage.h` + `hf::foliage`. **STRICT-integer zero-diff cross-backend golden** (a CPU-rendered 2D
integer heatmap, like the PCG plots — NOT a float visresolve slice). NO GPU/shader (the wind is host-evaluated —
the int64/glslc/MSL shader boundary is sidestepped entirely; FO5's render reuses `lit_instanced.vert` verbatim).

## The header — `engine/foliage/foliage.h` (NEW, header-only, namespace `hf::foliage`)
Unlike the audio/net flagships, foliage is a render flagship with Mac-BAKED goldens, so foliage.h MAY freely
`#include "sim/fpx.h"` / `"pcg/pcg.h"` (no clang-standalone constraint). Reuse `fpx.h` read-only for `fx`/`kOne`/
`kFrac`/`fxmul`/`FxVec3`. Provide:
- A committed `static const int16_t kFoliageWind16[256]` — the SAME values as `engine/audio/mixer.cpp`'s
  `kSineTable` (`kFoliageWind16[i] == round(32767 * sin(2*pi*i/256))`). COPY the 256 int16 literals verbatim from
  `mixer.cpp` (a comment notes the generation formula; the data is committed integer literals — NO runtime `sin`).
- `struct Gust { int32_t kx; int32_t kz; uint32_t speed; fx amp; };` — a wind component: spatial frequencies
  `kx`/`kz` (how fast the phase advances across X/Z), a temporal `speed` (phase advance per frame), and a Q16.16
  amplitude `amp` (the bend this gust contributes).
- `struct WindField { Gust gusts[4]; int gustCount = 0; fx master = kOne; };` (a few gusts + a master amplitude;
  `gustCount <= 4`).
- `inline fx WindBend(const WindField& w, const FxVec3& pos, uint32_t frame)`: for each of `w.gustCount` gusts,
  compute a **uint32 phase** `phase = (uint32_t)((uint32_t)g.kx * (uint32_t)pos.x + (uint32_t)g.kz * (uint32_t)pos.z
  + g.speed * frame)` (pure uint32 wrapping arithmetic — a phase accumulator; only the top bits matter), index the
  LUT `int32_t s = kFoliageWind16[(phase >> 24) & 255]` (an int16 in `[-32767, 32767]`, ~Q15), and accumulate the
  gust contribution `bend += fxmul(g.amp, (fx)s) >> 15`-style — i.e. scale the Q15 LUT value by the Q16.16 `amp`
  into a Q16.16 angle (pick a consistent fixed-point convention: `bend += (fx)(((int64_t)g.amp * s) >> 15)`).
  Finally scale by `w.master`: `return fxmul(bend, w.master)`. The result is a small Q16.16 bend angle (radians-ish,
  bounded by the summed amplitudes). Pure integer — `fxmul`'s int64 intermediate is CPU-side (fine). NO `<cmath>`,
  NO float, NO clock/RNG.

## CPU test (optional) + the showcase
This is a render-flagship slice (the golden is a Mac-baked image, not a pinned hash), but a small
`tests/foliage_test.cpp` (register `hf_add_pure_test(foliage_test)`) asserting WindBend determinism + the
zero-amplitude no-op + frame-sensitivity is welcome (mirror pcg_test). The PRIMARY artifact is the showcase image.

## Showcase — `--fo1-wind-shot` (Vulkan, main.cpp) + `--fo1-wind` (Metal, visual_test.mm)
A **2D top-down strict-integer heatmap of the bend field** over a fixed XZ patch at a fixed frame. Map a 256×256
grid of XZ world positions (a fixed patch, e.g. `[0, 64)`² in Q16.16) → for each pixel compute `bend =
WindBend(wind, {x, 0, z}, frame)` and color it by the SIGNED bend via PURE INTEGER math (e.g. positive bend → a
green ramp, negative → a blue ramp, magnitude = `clamp(|bend| >> shift, 0, 255)`; NO float pixel math). Use a fixed
`WindField` (2–3 gusts) and a fixed `frame`. The result is a smoothly-banded wind heatmap. Reuse the PCG plot's
pure-integer BMP-write code (`--pcg1-hash-shot` / `--pcg2-scatter-shot`, main.cpp ~line 3406+). SAME wind/frame/
patch/image-size IN BOTH renderers so the image is byte-identical cross-backend by construction.

## Proof (STRICT integer — zero-diff cross-backend)
```
fo1-wind: deterministic integer wind field (gusts=<G>, frame=<F>, patch 256x256)
fo1-wind: two-run BYTE-IDENTICAL
fo1-wind: different frame -> different field {frameA:0x<Ha>, frameB:0x<Hb>} Ha != Hb
fo1-wind: zero-amplitude -> zero bend everywhere (no-op) {maxBend:0}
fo1-wind: provenance {gusts:<G>, frame:<F>}
```
Assertions: (1) two runs byte-identical; (2) a different frame → a different field (image checksum differs) — the
wind animates; (3) the zero-amplitude (`master=0` or all `amp=0`) control → `WindBend == 0` for every sampled
position (the no-op); (4) provenance + coherent (non-uniform field). Register `fo1_wind` in verify.ps1 $Goldens
(Flag `--fo1-wind`) + `--fo1-wind-shot` in $vkShots, mirroring `pcg1_hash`/`pcg2_scatter`.

## Constraints (HARD)
- NEW engine/foliage/foliage.h (+ optional tests/foliage_test.cpp + its CMake registration) + the showcase blocks
  in main.cpp/visual_test.mm + the verify.ps1 registration. Reuse fpx.h READ-ONLY; COPY the kSineTable values
  verbatim into kFoliageWind16. Do NOT modify mixer.{h,cpp}/pcg.h/any existing shader/golden. NO new RHI, NO new
  shader (the wind is CPU host-evaluated). Pure-CPU integer for the wind + the 2D plot.
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (the controller bakes Metal and
  requires the Metal image == the Vulkan image byte-for-byte). The 2D heatmap MUST be pure integer (no float pixel
  math). Do NOT route through a GPU float raster.
- Branch `fix-issue-21-fo1`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  (+ foliage_test if you add it). Run `hello_triangle.exe --fo1-wind-shot <out.bmp>`, confirm exit 0 + the proof lines.
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--fo1-wind-shot` exits 0, the proof lines print,
  two-run byte-identical, the frame-sensitivity holds, the zero-amplitude no-op holds (maxBend 0), the plot is
  pure-integer. Commit message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the branch
  and STOP. Report: commit hash, the proof output, the image path, confirmation both renderers use identical
  wind/frame/patch/size, and that the plot is pure-integer (so it'll be zero-diff cross-backend). (The CONTROLLER
  bakes the Metal golden, confirms zero-diff cross-backend, eyeballs the heatmap.)
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
