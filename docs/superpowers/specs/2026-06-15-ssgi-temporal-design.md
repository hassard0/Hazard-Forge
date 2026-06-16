# Slice BV — Temporal SSGI Accumulation (Phase 4 #21) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. Completes the GI
> quality trilogy: raw SSGI (BP) → spatial denoise (BR) → TEMPORAL accumulation (this). Reuses the
> proven TAA (Slice AP) fixed-N-accumulation determinism pattern.

**Goal:** Drive the SSGI noise down further by accumulating multiple jittered SSGI frames over time. For a
STATIC scene + STATIC camera (which the SSGI color-bleed showcase uses), temporal accumulation is simply
averaging N frames each with a DIFFERENT jittered hemisphere kernel — no motion-vector reprojection
needed (the camera doesn't move). This converges to a much cleaner indirect-diffuse result than a single
frame. A NEW `--ssgi-temporal-shot` shows the accumulated result; the raw `ssgi.png` (BP) and
`ssgi_denoise.png` (BR) stay frozen for A/B/C comparison. Golden-verified on both backends.

## Why fixed-N accumulation (deterministic, no reprojection)

TAA (Slice AP) already established the pattern: render N FIXED jittered frames over a static scene and
accumulate — deterministic (fixed N, fixed per-frame jitter, no time/RNG), two runs bit-identical. Apply
the same here: each of N=8 accumulation frames rotates the SSGI hemisphere kernel by a FIXED per-frame
offset (e.g. a golden-angle rotation indexed by frame, or a different Hammersley sub-sequence per frame),
and the SSGI indirect buffers are averaged. Since the camera is STATIC, there is NO reprojection — frame i
and frame j sample the same surface, so a straight running mean is correct. (Temporal reprojection for a
MOVING camera — history reproject + neighborhood clamp like TAA's resolve — is a future enhancement;
single-static-camera accumulation is the deterministic, golden-stable scope here.)

## Design decisions (locked)

1. **Per-frame kernel rotation (extend engine/render/ssgi.h, pure CPU, no backend symbols).** Add to
   `hf::render::ssgi`:
   - `math::Vec3 HemisphereDirJittered(int i, int K, const math::Vec3& normal, int frame)` — the i-th of
     K hemisphere dirs for accumulation `frame`, where `frame` applies a FIXED rotation/offset to the base
     kernel (e.g. add `frame * goldenAngle` to the spiral azimuth, or use Hammersley with a frame-offset
     index). Deterministic per (i, K, normal, frame). Document the rotation. Falls back to `HemisphereDir`
     for frame 0 (so frame 0 == the BP raw kernel — document, keeps a clean relationship).
   - `SsgiTemporalParams { int accumFrames; ... }` (N=8 default).

2. **Accumulation in the showcase / shader.** The `--ssgi-temporal-shot` path renders the SSGI indirect
   for each of N frames (each with the jittered kernel for that frame) and ACCUMULATES into a running mean
   (a float HDR accumulation RT, `RGBA16_Float` or `RGBA32_Float` — reuse the existing HDR-RT/accumulation
   pattern from TAA), then composites `meanIndirect * albedo` into the scene exactly like BP. The per-frame
   SSGI pass is the EXISTING `ssgi.frag.hlsl` parameterized by the frame index (pass `frame` so the shader
   selects the jittered kernel — add a `frame` field to the SSGI params push constant/UBO). EXISTING raw
   SSGI (`--ssgi-shot`) + denoise (`--ssgi-denoise-shot`) paths stay BYTE-IDENTICAL (the temporal path is a
   new flag; frame-0 single-shot raw is unchanged) → `ssgi.png` + `ssgi_denoise.png` UNCHANGED.

3. **Showcase `--ssgi-temporal-shot <out>` (Vulkan) / `--ssgi-temporal` (Metal).** The SAME Cornell
   color-bleed scene; accumulate N=8 jittered SSGI frames → composite → capture. Print `ssgi-temporal:
   {accumFrames:8, rays:16, ...}`. New golden `tests/golden/metal/ssgi_temporal.png` (Metal two runs DIFF
   0.0000). Existing 45 image goldens UNTOUCHED (incl. ssgi.png + ssgi_denoise.png — verify both stay
   byte-identical, the A/B/C baselines).

4. **Determinism.** Fixed N, fixed per-frame kernel rotation, no time/RNG. Two runs byte-identical. The
   accumulation order (frame 0..N-1) is fixed.

5. **Tests `tests/ssgi_temporal_test.cpp` (pure CPU, no GPU):**
   - **Jittered kernel:** `HemisphereDirJittered(*, *, n, 0)` == `HemisphereDir` (frame 0 = base); for
     `frame>0` the dirs are unit-length + still in the hemisphere of `n` + DIFFER from frame 0 (so
     accumulation actually adds new samples); deterministic per (i,K,n,frame).
   - **Accumulation reduces variance (CPU mini-model):** averaging the indirect over N frames of a
     synthetic noisy-but-unbiased SSGI estimator yields a result with LOWER variance than a single frame
     and the SAME mean (converges toward ground truth). Document the mini-model.
   - **Determinism:** the per-frame kernels + the accumulation are stable across runs.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `--ssgi-temporal-shot` (showcases). NO new feature string (the
   `screen-space-global-illumination` feature covers it — temporal is a quality variant) — OR add
   `ssgi-temporal`; pick + document. Default: showcase-only (one-line introspect delta).

## RHI seam additions (summary)
- **None.** Reuses the SSGI/SSAO G-buffer + the TAA HDR-accumulation-RT pattern + composite. New/changed
  files (`engine/render/ssgi.h` additions, `shaders/ssgi.frag.hlsl` `frame` param, optional accumulation
  shader, `tests/ssgi_temporal_test.cpp`) add ZERO backend code symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Temporal REPROJECTION for a moving camera (history reproject + neighborhood clamp — needs motion vectors;
future), variance-guided adaptive N, exponential-moving-average with a history texture (use a straight
fixed-N mean — simpler + deterministic), denoising the accumulated result (BR's spatial denoise could be
combined later), multi-bounce. One static-camera fixed-N jittered accumulation, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 45) + new `ssgi_temporal_test` (jittered kernel,
   variance-reduction mini-model, determinism). Clean under `windows-msvc-asan`.
2. `--ssgi-temporal-shot` on Windows/Vulkan: controller visual review — the color bleed is CLEANER/less
   noisy than both `--ssgi-shot` (raw) and visibly converged, edges intact, coherent; the `ssgi-temporal:
   {...}` line is deterministic (two runs → byte-identical capture). Run under the AT Vulkan-validation
   gate → ZERO errors.
3. Metal: `visual_test --ssgi-temporal` → new golden `tests/golden/metal/ssgi_temporal.png`; two runs DIFF
   0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `ssgi_temporal.png`
   added; the other 45 byte-identical — CRITICALLY `ssgi.png` AND `ssgi_denoise.png` unchanged (proves the
   temporal path is additive, the raw + spatial paths untouched).
5. Introspect JSON rebaked exactly `+--ssgi-temporal-shot` (+ `ssgi-temporal` feature only if chosen);
   introspect test updated; no other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `ssgi_temporal`
   image golden in the Mac round-trip loop.
