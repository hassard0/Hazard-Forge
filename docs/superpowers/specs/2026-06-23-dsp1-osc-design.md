# Slice DSP1 — Block buffer + stateful wavetable oscillator node (Issue #26, flagship #23 DETERMINISTIC AUDIO, beachhead)

The irreducible primitive every later DSP slice consumes: a fixed-size audio **block** type + a **stateful
wavetable oscillator NODE** whose 32-bit phase accumulator carries across block boundaries, so rendering one big
buffer is byte-for-byte identical to rendering it as N separate blocks. Pure-CPU integer (Q15 / int16 PCM), strict
**bit-exact buffer-hash golden** (NO image, NO Mac render-bake — the cross-platform proof is the same CPU test
producing the identical hash on Windows/MSVC and Mac/clang). Establishes `engine/audio/dsp.h` + `hf::audio::dsp`.

## Why a stateful node (the determinism crux this slice pins)
The existing `mixer.cpp::Render` resets `phase = 0` per voice and renders the whole voice in one loop. A graph
node must instead **persist its phase as member state** and advance it across `RenderBlock` calls, so that a patch
evaluated block-by-block (the real runtime path) is bit-identical to a single big render. This block-boundary
continuity is THE load-bearing invariant the whole flagship rides on — DSP1 proves it and every later slice
re-asserts it.

## The header — `engine/audio/dsp.h` (NEW, header-only, namespace `hf::audio::dsp`)
Reuse READ-ONLY from the existing audio module + fpx (do NOT modify `mixer.{h,cpp}` / `mixer`'s `kSineTable` —
either `#include` the mixer header if `kSineTable`/`MulQ15`/`ClampI16` are exposed, OR copy the SAME 256-entry
sine table + the `MulQ15`/`ClampI16` helpers verbatim into `dsp.h` as committed constants — match the mixer's
exact integer values so the determinism story is identical; the mixer's table is in `mixer.cpp` `kSineTable`,
`kSineN=256`, `kSineShift=24`). Provide:
- `inline uint64_t DigestBuffer(const std::vector<int16_t>& buf)` — the engine FNV-1a-64 over the buffer bytes
  (offset basis `1469598103934665603ULL`, prime `1099511628211ULL`; hash each int16 as its two little-endian
  bytes). Returns the 64-bit digest (the per-slice golden value). (This is the same FNV used by `ai.h` /
  `verdict` `DigestSnapshot` — reuse the exact constants.)
- `enum class Wave { Sine, Saw, Square };`
- `struct OscNode { Wave wave = Wave::Sine; uint32_t freqHz = 440; uint32_t phase = 0; };` with
  `void RenderBlock(OscNode& osc, int sampleRate, int frames, std::vector<int16_t>& outAppend)`: for each of
  `frames` samples, emit `int16` `kSineTable[(osc.phase >> kSineShift) & (kSineN-1)]` (Sine), or `Saw` = the top
  16 bits of `osc.phase` mapped to a signed int16 ramp (`(int16_t)(osc.phase >> 16) ` minus the midpoint — pick a
  pure-integer mapping that ramps -32767..+32767 over a cycle), or `Square` = `(osc.phase & 0x80000000u) ? -32767
  : 32767` (the mixer convention), then `osc.phase += inc` where `inc = (uint32_t)(((uint64_t)freqHz << 32) /
  (uint32_t)sampleRate)` (the mixer's exact increment). APPEND to `outAppend` (so N calls build one stream). The
  phase is read FROM and written BACK TO `osc` so it carries across calls — that is the whole point.
- A convenience `std::vector<int16_t> RenderOsc(Wave, uint32_t freqHz, int sampleRate, int totalFrames)` that
  makes a fresh `OscNode` and renders `totalFrames` in ONE `RenderBlock` (the "one big buffer" reference).
All int64-intermediate where needed (CPU-side, fine). NO `<cmath>`, NO float, NO clock/RNG.

## CPU test — `tests/dsp_test.cpp` (NEW, register `hf_add_pure_test(dsp_test)` in tests/CMakeLists.txt by audio_test)
Assertions: (1) **block-boundary determinism (make-or-break)** — render a 440 Hz sine of `N*256` frames (e.g.
N=16) as ONE buffer (`RenderOsc`) AND as N separate 256-frame `RenderBlock` calls on a single persistent
`OscNode`, and assert the two buffers are **byte-identical** (`DigestBuffer(a) == DigestBuffer(b)`); (2) **pinned
hash** — assert `DigestBuffer(one-buffer)` equals a HARD-PINNED expected `uint64_t` (compute it on the first run,
hardcode it — the regression anchor / the golden); (3) **replay-stable** — two `RenderOsc` calls byte-identical;
(4) **per-wave** — Sine/Saw/Square each produce a deterministic pinned hash, and a Square at 440 Hz is a hard
±32767 wave (sample values are only ±32767); (5) **freq-sensitive** — 220 Hz vs 440 Hz differ. Print
`dsp_test: ALL CHECKS PASSED`. Print the computed hashes (so the controller + the Mac run can compare them).

## Showcase / introspection — `--dsp1-osc` (a numeric proof, NOT an image)
This flagship's goldens are buffer hashes, not images. Add a small `--dsp1-osc` path (wherever the engine's
audio/dsp introspection or the test harness prints — mirror how `audio_test` / the existing `--*-shot` numeric
showcases report). It should render the 440 Hz sine and print:
```
dsp1-osc: wavetable oscillator (440Hz sine, sampleRate 48000, frames 4096)
dsp1-osc: block-boundary determinism: one-buffer == N-blocks {hash:0x<H>}
dsp1-osc: per-wave hashes {sine:0x<Hs>, saw:0x<Hsaw>, square:0x<Hsq>}
dsp1-osc: provenance {frames:4096, sampleRate:48000, blockSize:256}
```
(If a `--dsp1-osc` CLI flag doesn't fit the existing harness cleanly, putting these prints in `dsp_test` is
acceptable — the golden is the pinned hash in the test, NOT an image in verify.ps1. Do register the test so it
runs in the suite.)

## Proof (STRICT bit-exact buffer hash — the cross-platform bar)
The golden is the pinned `DigestBuffer` value in `dsp_test`. The cross-platform proof = the controller runs the
SAME `dsp_test` on the Mac (via the rig) and gets the IDENTICAL pinned hash (NO render-bake, NO image). The
make-or-break is one-buffer == N-blocks (block-boundary phase carry).

## Constraints (HARD)
- NEW engine/audio/dsp.h + tests/dsp_test.cpp + the tests/CMakeLists.txt registration ONLY. Reuse mixer's
  table/helpers + fpx READ-ONLY. Do NOT modify mixer.{h,cpp}, the existing scene.wav golden, or any other file.
  NO new RHI, NO shader, NO GPU. Pure-CPU integer (int16 PCM / Q15 / uint32 phase). NO `<cmath>`, NO float.
- The sine table + phase increment + square convention MUST match the mixer's exact integer values (so the audio
  story is identical and a future unification is trivial).
- Branch `fix-issue-26-dsp1`, commit there, do NOT merge. (NO tests/golden/metal — this flagship has no image
  goldens. The DSP6 capstone will add a tests/golden/audio/song.wav later; DSP1 has none.)
- Build Windows + run the test via the PowerShell tool (NOT bash — vcvars no-ops under Git Bash), single-quoting
  the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target dsp_test'`
  then run the dsp_test.exe (find it under build\windows-msvc-release) and confirm ALL CHECKS PASSED + capture the
  printed hashes.
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, dsp_test passes (ALL CHECKS PASSED), the
  one-buffer == N-blocks equality holds, the pinned hash matches, two-run identical. Commit message via a temp
  file + `git commit -F` (use the Bash tool heredoc). Commit to the branch and STOP. Report: commit hash, the
  printed hashes (sine/saw/square + the block-boundary hash), confirmation dsp_test passes, and confirmation the
  sine table/inc/square convention match the mixer's values. (The CONTROLLER builds + runs dsp_test on the Mac to
  confirm the IDENTICAL hashes = the cross-platform proof, then merges.)
