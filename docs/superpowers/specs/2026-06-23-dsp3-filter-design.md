# Slice DSP3 — Fixed-point biquad filter node (Issue #26, flagship #23 DETERMINISTIC AUDIO, 3rd slice — THE CRUX)

A **fixed-point biquad FILTER node** (Direct Form I, lowpass/highpass) with COMMITTED integer coefficient presets
(host-precomputed offline — NO runtime `cos`/`tan`), int64-intermediate accumulation, and a delay line carried
across block boundaries. This is the crux slice: fixed-point filter stability + cross-block state. Pure-CPU
integer, strict **bit-exact buffer-hash golden** (no image, no Mac render-bake — same `dsp_test` hash on MSVC and
clang). Builds on DSP1/DSP2.

## Coefficient format + the committed presets (host-baked — the LUT discipline)
Use **Q14** (`16384 == 1.0`) int32 coefficients (a1 can reach ~±2.0, which overflows Q15 — Q14 has the headroom;
int32 holds it; the accumulate is int64). Direct Form I, `a0` normalized to 1:
```
y[n] = (b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]) >> 14   (then ClampI16)
```
where `x[n-1]/x[n-2]` are past INT16 inputs and `y[n-1]/y[n-2]` are past INT16 outputs (post-clamp — the feedback
uses the actual emitted samples, which keeps the fixed-point loop bounded).

**Commit a small set of presets as `static const` Q14 int32 literals.** Compute them OFFLINE with the RBJ
audio-EQ cookbook formulas (a THROWAWAY python/host calc — do NOT compute coefficients at runtime), quantize to
Q14 (round-to-nearest), and hardcode. Required presets:
- **Passthrough** (the additive-identity no-op — EXACT): `b0 = 16384` (1.0), `b1 = b2 = a1 = a2 = 0`. Then
  `y[n] = (16384*x[n]) >> 14 = x[n]` exactly → output == input BYTE-IDENTICAL. This is the make-or-break no-op.
- **Lowpass** (≥1 cutoff, e.g. fc = 2 kHz @ 48 kHz, Q = 0.707). For reference, the RBJ LPF at fc=2k/fs=48k/Q=0.707
  quantizes to approximately `{b0:236, b1:472, b2:236, a1:-26755, a2:11314}` (Q14) — **recompute these exactly in
  your throwaway calc and use YOUR rounded values** (this reference is a sanity check, not authoritative). Verify
  STABILITY: `|a2| < 16384` and `|a1| < 16384 + a2` (poles inside the unit circle) — assert this in a comment.
- **Highpass** (≥1 cutoff, e.g. fc = 2 kHz @ 48 kHz, Q = 0.707) via the RBJ HPF formulas, same Q14 quantization +
  stability check.

RBJ formulas (for the offline calc): `w0 = 2π·fc/fs`, `α = sin(w0)/(2Q)`; LPF: `b0=(1-cos w0)/2, b1=1-cos w0,
b2=(1-cos w0)/2, a0=1+α, a1=-2cos w0, a2=1-α`; HPF: `b0=(1+cos w0)/2, b1=-(1+cos w0), b2=(1+cos w0)/2,
a0=1+α, a1=-2cos w0, a2=1-α`; then divide all by `a0` and ×16384 round-to-nearest.

## The addition — `engine/audio/dsp.h` (APPEND-ONLY after the DSP2 block; keep self-contained, no mixer/fpx include)
Add to `hf::audio::dsp` (do NOT modify DSP1/DSP2; keep dsp.h compilable STANDALONE — only `<cstdint>`/`<vector>`):
- `struct BiquadCoeffs { int32_t b0, b1, b2, a1, a2; };` + the committed `static const BiquadCoeffs` presets
  (`kBiquadPassthrough`, `kBiquadLowpass2k`, `kBiquadHighpass2k`, …) as Q14 literals.
- `struct BiquadNode { BiquadCoeffs c; int16_t x1=0, x2=0, y1=0, y2=0; };` (the delay line — the carried state).
- `void FilterBlock(BiquadNode& bq, const std::vector<int16_t>& src, std::vector<int16_t>& outAppend)`: for each
  sample `x` in `src`, compute `int64_t acc = (int64_t)bq.c.b0*x + (int64_t)bq.c.b1*bq.x1 + (int64_t)bq.c.b2*bq.x2
  - (int64_t)bq.c.a1*bq.y1 - (int64_t)bq.c.a2*bq.y2;` then `int16_t y = ClampI16((int32_t)(acc >> 14));` (use an
  arithmetic right shift on the signed int64), append `y`, and shift the delay line (`x2=x1; x1=x; y2=y1; y1=y;`).
  The state persists in `bq` across calls. NO `<cmath>`, NO float.

## CPU test — extend `tests/dsp_test.cpp` (add a DSP3 section, keep DSP1/DSP2 checks)
Assertions: (1) **passthrough no-op (make-or-break)** — `FilterBlock` with `kBiquadPassthrough` over a source
returns the source BYTE-IDENTICAL (`DigestBuffer` equal); (2) **block-boundary determinism** — a saw source of
`N*256` frames filtered (lowpass) as ONE `FilterBlock` vs N separate 256-frame `FilterBlock` calls on ONE
persistent `BiquadNode` → byte-identical (the delay line carries); (3) **stability / bounded output** — filter a
LONG full-scale input (e.g. a 1 kHz square or a step) through the lowpass for many blocks and assert the output
never overflows (always within int16 — guaranteed by ClampI16, but ALSO assert it doesn't railed-clip
indefinitely / settles: the steady-state of a step through a unity-DC-gain lowpass is bounded near the input
level, not oscillating to the rails) — a deterministic bounded-output check proving the fixed-point loop is
stable; (4) **lowpass attenuates highs** — feed a high-frequency tone (e.g. 12 kHz) and a low-frequency tone (e.g.
500 Hz) at equal amplitude through the lowpass and assert the high-freq output RMS/peak is materially LOWER than
the low-freq output (the filter does its job — a coarse integer energy compare, deterministic); (5) **pinned
hash** — `saw → lowpass` digest == a hard-pinned `uint64_t`. Print the hashes + `dsp_test: ALL CHECKS PASSED`.
Report lines:
```
dsp3-filter: biquad filter node (Q14 Direct-Form-I, presets: passthrough/lowpass2k/highpass2k)
dsp3-filter: passthrough preset == source BYTE-IDENTICAL (additive-identity no-op)
dsp3-filter: block-boundary determinism: one-buffer == N-blocks {hash:0x<H>}
dsp3-filter: stable bounded output over <M> blocks {maxAbs:<v>, railed:false}
dsp3-filter: lowpass attenuates highs {lowEnergy:<L>, highEnergy:<Hh>} Hh < L
dsp3-filter: saw->lowpass pinned {hash:0x<Hf>}
```

## Proof (STRICT bit-exact buffer hash — cross-platform bar)
The golden is the pinned `DigestBuffer` in `dsp_test`; the cross-platform proof = the controller compiles+runs
`dsp_test` on the Mac (clang) and gets the IDENTICAL hashes. Make-or-break: the passthrough byte-identical no-op +
the saw→lowpass one-buffer == N-blocks (delay-line carry).

## Constraints (HARD)
- APPEND-ONLY to engine/audio/dsp.h (do NOT modify DSP1/DSP2) + extend tests/dsp_test.cpp. Keep dsp.h
  self-contained (only `<cstdint>`/`<vector>`, NO mixer/fpx include — the Mac clang single-file compile must keep
  working). Do NOT modify mixer.{h,cpp}, scene.wav, or any other file. NO RHI/shader/GPU. Pure-CPU integer. NO
  `<cmath>`, NO float at RUNTIME (the coefficient calc is OFFLINE/throwaway — only the rounded Q14 int literals
  ship).
- Coefficients are COMMITTED int Q14 literals (host-baked, the kSineTable discipline) — NO runtime `cos`/`tan`/
  coefficient computation. Verify each preset is stable (poles in the unit circle) and note it in a comment.
- Branch `fix-issue-26-dsp3`, commit there, do NOT merge. NO golden file (the pinned hash lives in the test).
- Build Windows + run dsp_test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target dsp_test'`
  then run dsp_test.exe (under build\windows-msvc-release).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, dsp_test prints ALL CHECKS PASSED, the
  passthrough no-op is byte-identical, the one-buffer == N-blocks equality holds, the stability/bounded-output
  check passes, the lowpass attenuates highs, the pinned hash matches, AND DSP1/DSP2 checks still pass. Commit
  message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the branch and STOP. Report:
  commit hash, the printed hashes, the committed Q14 coefficients you baked + their stability check, confirmation
  dsp_test passes (DSP1+DSP2+DSP3), and confirmation dsp.h stays self-contained. (The CONTROLLER runs dsp_test on
  the Mac to confirm identical hashes, then merges.)
