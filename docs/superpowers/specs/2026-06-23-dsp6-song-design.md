# Slice DSP6 — Procedural-phrase capstone + lockstep/rollback (Issue #26, flagship #23, 6th/FINAL slice)

The capstone that COMPLETES flagship #23: synthesize a recognizable procedural musical phrase through the full
DSP1–DSP5 graph to a stereo PCM buffer, emit it as a committed WAV golden, and prove the audio is netcode-
replayable (two peers synthesize the byte-identical buffer from the same note-event stream; rollback corrects a
mispredicted stream to the authority). Pure-CPU integer. Two-tier proof: (a) the pinned RenderSong buffer Digest
in `dsp_test` (the cross-platform proof — identical on MSVC and Mac/clang), and (b) a committed
`tests/golden/audio/song.wav` byte-golden in verify.ps1 (the productized artifact, the `scene.wav` mechanism).

## The addition — `engine/audio/dsp.h` (APPEND-ONLY after the DSP5 block; keep self-contained — NO wav.h include)
Add to `hf::audio::dsp` (do NOT modify DSP1–5; keep dsp.h STANDALONE — only `<cstdint>`/`<vector>`; the WAV write
lives in the SHOWCASE, not here, so dsp.h stays header-only/self-contained):
- `struct AudioCommand { uint32_t freqHz; int startSample; int durSample; int32_t panX; };` (a note event: pitch,
  onset, length, and a lateral pan position `panX` in `[-32768,32768]` driving the constant-power pan — the
  arpeggio moves across the stereo field).
- `std::vector<int16_t> RenderSong(const std::vector<AudioCommand>& notes, int sampleRate, int totalFrames)`:
  build a STEREO interleaved buffer of `totalFrames` frames by accumulating each note's contribution in fixed
  command order: for each note render its `Osc → Env` mono signal (a fresh `OscNode` at `freqHz`, a fresh
  `EnvNode` with a fixed ADSR over `durSample`, starting at `startSample`), pan it via the `kPanLut` (panIdx from
  `panX`, `gainL/gainR`), and accumulate into int32 L/R accumulators at the right sample offset; after all notes,
  optionally run the mix through ONE `BiquadNode` lowpass; `ClampI16` each accumulator once → interleaved stereo.
  Pure function of `(notes, sampleRate, totalFrames)`. Use a recognizable default phrase in the showcase (e.g. a
  4-note major arpeggio 262/330/392/523 Hz, staggered, each panned across L→R). Deterministic, integer-only.
- Lockstep/rollback (the FPX5 shape, pure-CPU):
  `std::vector<int16_t> RunAudioLockstep(const std::vector<AudioCommand>& authority, int sr, int frames)` — returns
  `RenderSong(authority, …)` (the "authority" buffer); a "peer" fed the same `authority` stream calls the same
  function → byte-identical (purity = lockstep determinism).
  `bool RunAudioRollback(const std::vector<AudioCommand>& mispredicted, const std::vector<AudioCommand>& authority,
  int sr, int frames, std::vector<int16_t>& correctedOut)` — render `mispredicted` (a peer's wrong guess, e.g. one
  note's freq off), then RE-render from the `authority` stream into `correctedOut`; return `true` iff
  `correctedOut == RenderSong(authority)` (the corrected re-sim matches the authority) AND the mispredicted buffer
  DIFFERED from the authority (a real divergence was corrected). NO `<cmath>`, NO float, NO wav.h.

## CPU test — extend `tests/dsp_test.cpp` (add a DSP6 section, keep DSP1–5 checks)
Assertions: (1) **pinned hash** — `DigestBuffer(RenderSong(theArpeggio))` == a hard-pinned `uint64_t` (the
cross-platform anchor); (2) **two-render determinism** — two `RenderSong` calls byte-identical; (3) **lockstep** —
`RunAudioLockstep` (authority) == a peer's independent `RenderSong(authority)` byte-identical; (4) **rollback** —
`RunAudioRollback(mispredicted, authority, …)` returns true: corrected == authority AND mispredicted != authority
(assert both — a real mispredict was corrected); (5) **coherent** — the song buffer is non-silent (some |sample| >
0) and stereo (L and R not all-equal given the moving pan). Print the hashes + `dsp_test: ALL CHECKS PASSED`.
Report lines:
```
dsp6-song: procedural phrase (4-note arpeggio Osc->Env->pan->mix->lowpass, stereo)
dsp6-song: two renders BYTE-IDENTICAL {hash:0x<H>}
dsp6-song: lockstep: peer == authority from the note stream alone (byte-identical) {frames:<F>}
dsp6-song: rollback: corrected == authority AND mispredicted differed {ok:true}
dsp6-song: provenance {notes:4, sampleRate:48000, stereo:true}
```

## Showcase — `--dsp-song <out.wav>` (Vulkan-side main.cpp) — the productized WAV artifact
Clone the `--audio-render` block (main.cpp ~4044-4123: headless, no window/GPU). Build the SAME default arpeggio
`AudioCommand` stream, call `dsp::RenderSong(notes, kSR, kFrames)`, and `audio::WriteWav(outPath, kSR, 2, buffer)`
(reuse `engine/audio/wav.h::WriteWav` — this is the ONLY place wav.h is touched; dsp.h stays clean). Print a short
provenance line. The committed golden `tests/golden/audio/song.wav` is the byte-reference (the CONTROLLER generates
it on Windows + commits it). (Metal needs no equivalent CLI — the cross-platform guarantee is the dsp_test buffer
Digest matching on Mac/clang; the WAV is that exact buffer + a deterministic header.)

## verify.ps1 — register the WAV byte-golden (mirror the `scene.wav` block)
Add a block mirroring the existing audio-WAV golden (the `--audio-render` / `scene.wav` block near verify.ps1:615):
run `--dsp-song` to a temp WAV, byte-compare against `tests/golden/audio/song.wav`, exit non-zero on mismatch.

## Proof / cross-platform
(a) The pinned `RenderSong` Digest in `dsp_test` is the cross-platform proof — the controller compiles+runs
`dsp_test` on the Mac (clang, self-contained) → IDENTICAL hashes. (b) The `song.wav` byte-golden is the Windows
gate (re-derived identically on Mac by the same integer code, exactly as `scene.wav` is treated today — no Mac
render-bake needed). This COMPLETES flagship #23.

## Constraints (HARD)
- APPEND-ONLY to engine/audio/dsp.h (keep it self-contained — NO wav.h/mixer/fpx include) + extend
  tests/dsp_test.cpp + the `--dsp-song` showcase in main.cpp + the verify.ps1 golden block. The ONLY wav.h use is
  in the main.cpp showcase. Do NOT modify mixer.{h,cpp}, scene.wav, DSP1–5, or any other file. NO RHI/shader/GPU.
  Pure-CPU integer. NO `<cmath>`, NO float.
- The committed `tests/golden/audio/song.wav` is generated by the CONTROLLER (do NOT commit a WAV you generate —
  the controller bakes it on Windows after reviewing). You may write a temp WAV to verify `--dsp-song` runs.
- Branch `fix-issue-26-dsp6`, commit there (dsp.h + dsp_test.cpp + main.cpp + verify.ps1), do NOT merge, do NOT
  commit `tests/golden/audio/song.wav` (controller generates + commits it).
- Build Windows (hello_triangle + dsp_test) via the PowerShell tool (NOT bash), single-quoting the cmd arg for the
  (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle dsp_test'`
  then run dsp_test.exe (ALL CHECKS PASSED) and `hello_triangle.exe --dsp-song <tempwav>` (exit 0, writes a WAV).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, dsp_test prints ALL CHECKS PASSED (incl. the
  lockstep + rollback), `--dsp-song` writes a non-trivial stereo WAV (exit 0), the pinned hash matches, AND DSP1–5
  checks still pass. Commit message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the
  branch and STOP. Report: commit hash, the pinned song hash + lockstep/rollback results, the temp WAV path +
  size, confirmation dsp_test passes (DSP1–6) + dsp.h stays self-contained, and confirmation you did NOT commit a
  song.wav. (The CONTROLLER runs dsp_test on the Mac for the identical hash, generates + commits song.wav, runs
  verify.ps1's WAV golden, then merges + closes issue #26.)
