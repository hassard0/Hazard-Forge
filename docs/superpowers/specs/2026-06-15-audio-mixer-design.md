# Slice BB — Deterministic Audio Mixer (Phase 4 #6) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly: render audio to a WAV buffer (the audio analogue of render-to-texture) and
> byte-match a golden. Fills a real engine-completeness gap — a "more powerful than UE5" engine needs
> sound. Pure CPU, no RHI/graphics, no platform audio device.

**Goal:** A deterministic software audio mixer. Procedural voices (oscillators + noise, shaped by an
ADSR envelope, panned) are mixed into a 16-bit stereo PCM buffer over a fixed duration and written as a
WAV. `--audio-render <out.wav>` produces a byte-exact deterministic WAV golden. No speaker needed — we
render audio to a buffer exactly like we render frames to a texture, and verify it headlessly.

## Why fixed-point (cross-platform bit-exactness)

Floating-point mixing can differ in the last bit across compilers (MSVC vs Apple clang), which would
break a byte-exact cross-platform WAV golden. So the mix path is **integer / fixed-point**: Q15 gains,
int32 accumulation, hard-clamp to int16. Oscillator phase is an integer accumulator; sine uses a FIXED
integer wavetable (a `static const int16_t kSine[N]` quarter/full-wave table, linear-or-nearest lookup —
document which); noise uses a FIXED-seed integer LCG. No `float`/`double` in the sample path ⇒ the WAV is
bit-identical on every platform and every run. (A float path is a possible future enhancement; not here.)

## Design decisions (locked)

1. **Mixer (engine/audio/mixer.{h,cpp}, pure CPU, no RHI/backend symbols).** Namespace `hf::audio`.
   - `enum class Wave { Sine, Square, Noise };`
   - `struct Adsr { int attack, decay, sustainLevel /*Q15*/, release; };` (times in samples).
   - `struct Voice { Wave wave; int freqHz; int startSample; int durSample; int16_t gain /*Q15*/;
     int16_t pan /*Q15, -1..+1 mapped*/; Adsr env; uint32_t noiseSeed; };`
   - `struct MixConfig { int sampleRate; int channels /*=2*/; int totalSamples; };`
   - `void Render(const MixConfig&, const std::vector<Voice>&, std::vector<int16_t>& outInterleaved)` —
     for each output sample, sum each active voice's (oscillator * envelope * gain), apply constant-power
     OR linear pan (document which; keep it integer), accumulate in int32, clamp to int16, write L/R
     interleaved. Deterministic, integer-only in the sample loop.
   - Oscillators: `Sine` = integer phase (Q-something) → fixed wavetable lookup; `Square` = sign of the
     phase half; `Noise` = per-voice LCG (`seed = seed*1664525 + 1013904223`, take high bits). Envelope:
     piecewise-linear ADSR in Q15 evaluated from the sample offset within the voice.

2. **WAV writer (engine/audio/wav.{h,cpp} or in mixer).** `WriteWav(path, sampleRate, channels,
   const std::vector<int16_t>& interleaved)` — standard 44-byte PCM WAV header (RIFF/WAVE/fmt /data,
   PCM=1, 16-bit) + the sample bytes, little-endian. Deterministic header (no timestamps).

3. **Showcase `--audio-render <out.wav>`.** Build a FIXED "audio scene" — a short deterministic sequence,
   e.g. a 2-second 44.1kHz stereo piece: a few sine notes (an arpeggio) with ADSR at scheduled
   startSamples, a square bass note, a short noise "hit" panned — enough that the mix exercises summing,
   enveloping, panning, and clamping. Render → WriteWav. Print `audio: {sampleRate:44100, channels:2,
   samples:N, voices:V, peak:P}` (P = peak abs sample, deterministic). New golden
   `tests/golden/audio/scene.wav` (byte-exact). This is a NEW golden CATEGORY (audio), checked on Windows
   (the producing build); since the path is integer-only it is bit-identical on Mac too — OPTIONALLY
   verify on the M4 for parity (build the audio module standalone — it's pure C++, no conan/SDL needed)
   and confirm the WAV byte-matches, but the Windows byte-golden is the gate.

4. **Tests `tests/audio_test.cpp` (pure CPU):**
   - **Determinism:** `Render` twice with the same scene → bit-identical buffers.
   - **Mix math:** two voices at known constant levels sum to the expected int16 (incl. hard-clamp at
     +-32767 when they'd overflow); a single full-gain voice hits the expected peak.
   - **Pan law:** pan = full-left puts energy in L and ~0 in R (and the documented center/right cases);
     integer pan is exact at the endpoints.
   - **Envelope:** ADSR at sample offsets 0 (silent/attack start), end-of-attack (peak), sustain region
     (sustainLevel), post-release (silent) matches expected Q15 values.
   - **WAV header:** the 44-byte header fields (chunk sizes, sampleRate, byteRate, blockAlign, bits) are
     correct for a known buffer; data chunk size == samples*channels*2.
   - Clean under `windows-msvc-asan`.

5. **Introspect.** Add exactly `audio-mixer` (features) + `--audio-render` (showcases). One-pattern
   rebake, no other drift.

## RHI seam additions (summary)
- **None.** Audio is pure CPU, entirely independent of the RHI/graphics. New files (`engine/audio/*`,
  `tests/audio_test.cpp`) add ZERO backend symbols. Seam grep stays at baseline (2). The audio module
  compiles into `hf_core` (ASan-scoped, unit-tested) like physics/math.

## Out of scope (YAGNI)
Real-time playback through a platform audio device (WASAPI/CoreAudio) — that's the audio analogue of the
windowed-Metal path, a future slice; floating-point/HRTF/3D-spatialization mixing; loading WAV/OGG/MP3
asset files; streaming; reverb/filters/DSP effects chain; a music sequencer/tracker; sample-rate
conversion; >2 channels. One integer mixer, procedural voices, fixed scene, WAV golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 31) + new `audio_test` (determinism, mix math,
   pan, envelope, WAV header). Clean under `windows-msvc-asan`.
2. `--audio-render out.wav` produces the deterministic WAV; the `audio: {...}` stat line is stable across
   runs (two runs → byte-identical WAV; SHA match).
3. New golden `tests/golden/audio/scene.wav` byte-exact vs a fresh `--audio-render` on Windows (the gate).
   OPTIONAL Mac parity: build the pure-C++ audio module on the M4, render, confirm byte-match (document
   the result; integer path ⇒ expected identical).
4. **Render-invariance:** no image golden touched — `git diff master --stat -- tests/golden/metal` EMPTY;
   the only new golden is `tests/golden/audio/scene.wav`. (New category, isolated.)
5. Introspect JSON rebaked exactly `+audio-mixer` + `--audio-render`; introspect test updated; no other
   drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to byte-check the new
   `scene.wav` golden on Windows (and the Mac parity check if done). Pure-ASCII.
