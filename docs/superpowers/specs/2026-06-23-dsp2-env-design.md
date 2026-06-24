# Slice DSP2 — Integer ADSR envelope node (Issue #26, flagship #23 DETERMINISTIC AUDIO, 2nd slice)

A streaming **ADSR envelope NODE** that shapes a source block sample-by-sample, with the envelope's elapsed-sample
state carried across block boundaries (like DSP1's phase). Pure-CPU integer (Q15), strict **bit-exact buffer-hash
golden** (no image, no Mac render-bake — same `dsp_test` hash on MSVC and clang). Builds on DSP1. Reuses the
mixer's exact linear-ADSR math (`EnvelopeAt`, `mixer.cpp:95-122`).

## The addition — `engine/audio/dsp.h` (APPEND-ONLY after the DSP1 block; keep it self-contained — header-only)
Add to `hf::audio::dsp` (do NOT modify the DSP1 osc code; keep dsp.h compilable STANDALONE — only `<cstdint>`/
`<vector>`, NO mixer include, so the Mac clang one-file compile keeps working):
- `struct Adsr { int attack = 0; int decay = 0; int sustainLevel = 32767; int release = 0; };` (attack/decay/
  release in SAMPLES; `sustainLevel` Q15) — mirror `mixer.h`'s `Adsr` fields exactly.
- `inline int EnvelopeAt(const Adsr& env, int t, int durSample)` — **copy `mixer.cpp:95-122` VERBATIM** (the
  release-takes-precedence linear ADSR: release ramp over the last `release` samples, attack 0→32767 over
  `[0,attack)`, decay 32767→sustain over `[attack,attack+decay)`, else hold sustain; returns 0 outside
  `[0,durSample]`). Match the integer math (the `int64`-intermediate ramps) exactly so it is bit-identical to the
  mixer.
- If `MulQ15`/`ClampI16` are NOT already in dsp.h from DSP1, add them verbatim from `mixer.cpp:62-70`.
- `struct EnvNode { Adsr env; int durSample = 0; uint32_t t = 0; bool enabled = true; };`
- `void ApplyEnvBlock(EnvNode& node, const std::vector<int16_t>& src, std::vector<int16_t>& outAppend)`: for each
  sample `s` in `src`, if `node.enabled` append `ClampI16(MulQ15(s, EnvelopeAt(node.env, (int)node.t, node.durSample)))`
  else append `s` verbatim (the **bypass no-op**); increment `node.t` once per sample so the envelope stage carries
  across `ApplyEnvBlock` calls. (`MulQ15(s, level)` with `level` in Q15 [0,32767] scales the sample by the
  envelope.) NO `<cmath>`, NO float.

## CPU test — extend `tests/dsp_test.cpp` (add a DSP2 section, keep the DSP1 checks)
Assertions: (1) **block-boundary determinism (make-or-break)** — build a 440 Hz sine source of `N*256` frames,
then apply the envelope (a real ADSR, e.g. attack 1000 / decay 2000 / sustain 24000 / release 1500,
`durSample = N*256`) as ONE `ApplyEnvBlock` over the whole source AND as N separate 256-frame `ApplyEnvBlock`
calls on ONE persistent `EnvNode` — assert the two outputs are byte-identical (`DigestBuffer` equal); (2)
**gate-off tail is exact silence** — for `t > durSample` `EnvelopeAt` returns 0, so samples past `durSample` are
exactly 0 (render a few blocks past the end and assert the tail is all-zero); (3) **bypass no-op** — `enabled =
false` → the output equals the source BYTE-IDENTICAL (the clean no-op control — Q15 scaling by 32767 is NOT
exactly identity, so the byte-identical no-op is the explicit bypass, not full-sustain); (4) **attack ramps from
0** — sample 0 of an envelope with `attack > 0` is 0 (silent onset), and the level is monotone non-decreasing
across the attack region (hand-check a couple of points against `EnvelopeAt`); (5) **pinned hash** — the
`osc→env` chain digest equals a hard-pinned `uint64_t`. Print the hashes + `dsp_test: ALL CHECKS PASSED`. Add the
report line:
```
dsp2-env: ADSR envelope node (a/d/s/r = 1000/2000/24000/1500, durSample 4096)
dsp2-env: block-boundary determinism: one-buffer == N-blocks {hash:0x<H>}
dsp2-env: gate-off tail is exact silence {tailZeros:true}
dsp2-env: bypass no-op (enabled=false) == source BYTE-IDENTICAL
dsp2-env: osc->env pinned {hash:0x<He>}
```

## Proof (STRICT bit-exact buffer hash — cross-platform bar)
The golden is the pinned `DigestBuffer` in `dsp_test`; the cross-platform proof = the controller compiles+runs
`dsp_test` on the Mac (clang) and gets the IDENTICAL hashes. Make-or-break: the osc→env chain rendered one-buffer
== N-blocks (both `osc.phase` AND `env.t` carry).

## Constraints (HARD)
- APPEND-ONLY to engine/audio/dsp.h (do NOT modify the DSP1 osc code) + extend tests/dsp_test.cpp. Reuse the
  mixer's EnvelopeAt/MulQ15/ClampI16 values (COPY verbatim — keep dsp.h self-contained, NO mixer include). Do NOT
  modify mixer.{h,cpp}, scene.wav, or any other file. NO RHI/shader/GPU. Pure-CPU integer. NO `<cmath>`, NO float.
- Branch `fix-issue-26-dsp2`, commit there, do NOT merge. NO golden file (the pinned hash lives in the test).
- Build Windows + run dsp_test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target dsp_test'`
  then run dsp_test.exe (under build\windows-msvc-release).
- VERIFY dsp.h STAYS STANDALONE-COMPILABLE: the controller will compile `dsp_test.cpp` with `clang++ -std=c++20
  -I engine -I tests` on the Mac — so do NOT add any include that breaks a single-translation-unit build.
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, dsp_test prints ALL CHECKS PASSED, the
  one-buffer == N-blocks equality holds, the gate-off tail is exact silence, the bypass no-op is byte-identical,
  the pinned hash matches. Commit message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to
  the branch and STOP. Report: commit hash, the printed hashes, confirmation dsp_test passes + the DSP1 checks
  still pass, and confirmation EnvelopeAt was copied verbatim from the mixer. (The CONTROLLER runs dsp_test on the
  Mac to confirm identical hashes, then merges.)
