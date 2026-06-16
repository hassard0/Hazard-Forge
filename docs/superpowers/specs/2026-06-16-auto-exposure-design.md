# Slice CW — Auto-Exposure (Histogram Eye Adaptation) (Phase 4 #45) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A genuinely-absent
> camera feature (the tonemap currently uses a FIXED exposure): a luminance-histogram auto-exposure that
> adapts the exposure to the scene's brightness, like the eye. Reuses the HDR color + tonemap + the CS
> compute barriers. Deterministic (integer histogram), with a clean adaptation-off no-op proof.

**Goal:** Auto-exposure — a compute pass builds a luminance histogram of the HDR scene color, a reduce pass
computes the average (log-)luminance and the target exposure via the standard key-value formula, and the
tonemap applies that exposure so bright scenes darken and dark scenes brighten (the camera/eye adapting). A
`--autoexposure-shot` showcase shows a high-dynamic-range scene auto-exposed to a balanced result;
golden-verified on both backends. The existing fixed-exposure tonemap path + its golden are UNTOUCHED
(auto-exposure is a NEW path behind the showcase flag). The pass carries its proof: with `adaptationEnabled =
false` the exposure is the fixed reference `E0` (the engine's existing default), so the auto-exposed render is
BYTE-IDENTICAL to the existing fixed-exposure tonemap — asserted internally, fail loudly on any diff.

## Why this is render-safe (the adaptation-off no-op proof)

The tonemap applies `exposure` to the HDR color before the curve. The auto-exposure path computes `exposure`
from the histogram; with `adaptationEnabled == false` it instead uses the FIXED `E0` that the default tonemap
already uses → the auto-exposed output equals the default-tonemap output exactly. So the showcase INTERNALLY
renders with `adaptationEnabled = false` and asserts it is BYTE-IDENTICAL (SHA) to the engine's standard
(fixed-exposure) render of the same scene — proving the histogram/reduce/apply plumbing is a pure pass-through
when adaptation is off (no constant bias, no exposure drift) — then renders the real `adaptationEnabled = true`
version as the golden. (Same internal-assert discipline as CN/CO/CP/CR/CS/CT/CV.) The unit test additionally
proves the histogram + average + exposure math analytically (a uniform-luminance image → exact average → exact
EV).

## Why deterministic (integer histogram, no float atomics)

The histogram bins are INTEGER counts: each pixel does an `atomicAdd(bin, 1)`. Integer addition is commutative
+ associative, so the final per-bin count is the number of pixels in that luminance range — INDEPENDENT of the
atomic execution order → bit-deterministic. The reduce + exposure are pure functions of the (deterministic)
counts. No float atomics (which would be order-nondeterministic). Single-frame INSTANT adaptation (no temporal
history) so the golden is stable. Two runs byte-identical.

## Design decisions (locked)

1. **Auto-exposure math (engine/render/auto_exposure.h, header-only pure CPU, no backend symbols).** Namespace
   `hf::render::autoexp`. Mirrors `dof.h`/`gtao.h`/`froxel.h` (shared with the shaders + the unit test).
   - `float Luminance(const math::Vec3& rgb)` — Rec.709 luma `0.2126 r + 0.7152 g + 0.0722 b`.
   - `int LumToBin(float lum, float minLogLum, float logLumRange, int bins)` — map a luminance to a histogram
     bin via `log2(lum)` normalized to `[minLogLum, minLogLum+logLumRange]` → `[0, bins-1]`; lum ≤ 0 (or below
     the floor) → bin 0 (the standard "black" bin). `float BinToLum(int bin, float minLogLum, float
     logLumRange, int bins)` — the inverse (bin center → luminance). Document the exact mapping.
   - `float AverageLuminance(const uint32_t* histogram, int bins, float minLogLum, float logLumRange,
     uint32_t totalPixels)` — the weighted average luminance: `Σ binLum * count` over the non-zero bins
     (excluding bin 0 = black, the standard) divided by the contributing pixel count; document the
     black-bin exclusion. Returns a sensible floor if all-black.
   - `float ExposureFromAverage(float avgLum, float keyValue)` — the target exposure `keyValue / max(avgLum,
     eps)` (the standard key-value / "middle grey" auto-exposure; document keyValue ~0.18). Higher avg → lower
     exposure (bright scene darkens); lower avg → higher exposure (dark scene brightens).
   - Pure, deterministic, no RNG/time. Document the full pipeline.

2. **Auto-exposure shaders.** `shaders/autoexposure_histogram.comp.hlsl` — one thread per pixel (or tile):
   read the HDR scene color, `Luminance`, `LumToBin`, `InterlockedAdd(histogram[bin], 1)` into a `bins`-entry
   INTEGER SSBO (cleared to 0 first). `shaders/autoexposure_reduce.comp.hlsl` — reduce the histogram →
   `AverageLuminance` → `ExposureFromAverage` → write the single `exposure` float to a 1-entry SSBO (gated by
   `adaptationEnabled`; when false write `E0`). The tonemap/apply reads `exposure` from that SSBO and applies
   it (a NEW tonemap variant `tonemap_autoexp.frag.hlsl`, or the existing tonemap reading the exposure SSBO —
   prefer a variant so the default path is untouched). The EXISTING tonemap + its golden stay BYTE-IDENTICAL.
   The histogram→reduce→tonemap chain uses the CS `ComputeToCompute` (clear→histogram→reduce) +
   `ComputeToFragment` (reduce→tonemap) barriers (already in `rhi.h`). HLSL→SPIR-V→MSL via the existing
   toolchain.

3. **Showcase `--autoexposure-shot <out>` (Vulkan) / `--autoexposure` (Metal).** A high-dynamic-range scene
   (a bright sky/emissive region + a dark shadowed region) where a fixed exposure would clip the bright or
   crush the dark, and auto-exposure balances it. Fixed camera, fixed scene. Print
   `auto-exposure: {bins:N, EV:X, keyValue:0.18}` (the deterministic computed exposure). INTERNALLY render
   with `adaptationEnabled = false` and assert BYTE-IDENTICAL (SHA) to the standard fixed-exposure render —
   fail loudly on any diff. New golden `tests/golden/metal/auto_exposure.png` (Metal two runs DIFF 0.0000).
   Existing 65 image goldens UNTOUCHED.

4. **Determinism.** Integer histogram (order-independent), fixed scene/camera/keyValue, single-frame instant
   adaptation (no temporal history). Two runs byte-identical.

5. **Tests `tests/auto_exposure_test.cpp` (pure CPU, no GPU):**
   - **Luminance:** `Luminance` matches Rec.709 for known colors (white→1, black→0, pure green→0.7152).
   - **Bin mapping inverse:** `BinToLum(LumToBin(lum)) ≈ lum` (within one bin width) across the range; lum≤0 →
     bin 0; monotone.
   - **Average analytic:** a histogram with all pixels in ONE bin → `AverageLuminance` == that bin's
     luminance; a known two-bin split → the correct weighted average; the black bin (bin 0) is excluded.
   - **Exposure:** `ExposureFromAverage` = keyValue/avg; a brighter avg → smaller exposure (monotone
     decreasing); a uniform-luminance image → exact EV (hand-computed).
   - **Determinism:** same histogram → same exposure.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `auto-exposure` (features) + `--autoexposure-shot` (showcases).

## RHI seam additions (summary)
- **None.** The histogram + exposure are INTEGER/float SSBOs via the existing storage-buffer + compute-dispatch
  path (like cluster_assign / the froxel volume); the tonemap variant reuses the fullscreen path; the barriers
  (`ComputeToCompute`/`ComputeToFragment`) already exist in `rhi.h` (CS). `InterlockedAdd` on a storage buffer
  is standard HLSL. New non-backend files (`engine/render/auto_exposure.h`, the 3 shaders,
  `tests/auto_exposure_test.cpp`) add ZERO above-seam backend code symbols. Seam grep stays at baseline (2).
  Report the seam result.

## Out of scope (YAGNI)
Temporal/smoothed eye adaptation (exposure lerping toward the target over time — needs a previous-frame
exposure; this is single-frame INSTANT, note temporal as future), exposure compensation curves / metering
modes (spot/center-weighted), bloom-threshold coupling, lens vignette exposure, manual exposure UI, physically-
based camera (aperture/ISO/shutter), auto-exposure of the bloom pass. One single-frame integer-histogram
key-value auto-exposure with an adaptation-off byte-identical proof + an analytic unit test, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 65) + new `auto_exposure_test` (luminance, bin inverse,
   average analytic + black-bin exclusion, exposure monotone, determinism). Clean under `windows-msvc-asan`.
2. **Adaptation-off no-op proof + visual:** `--autoexposure-shot` on Vulkan: the HDR scene auto-exposed to a
   balanced result (bright not clipped, dark not crushed), coherent; the INTERNAL adaptationEnabled=false
   render is BYTE-IDENTICAL (SHA) to the standard fixed-exposure render; the `auto-exposure: {...}` line is
   deterministic (two runs → byte-identical capture). Run under the AT Vulkan-validation gate → ZERO errors
   (the histogram→reduce→tonemap barriers stay SYNC-HAZARD-free).
3. Metal: `visual_test --autoexposure` → new golden `tests/golden/metal/auto_exposure.png`; two runs DIFF
   0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `auto_exposure.png` added;
   the other 65 byte-identical (incl. the existing tonemap goldens).
5. Introspect JSON rebaked exactly `+auto-exposure` + `--autoexposure-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `auto_exposure` image
   golden in the Mac round-trip loop.
