# Slice DSP5 — Deterministic 3D spatialization node (Issue #26, flagship #23 DETERMINISTIC AUDIO, 5th slice)

A **spatialization NODE**: a mono source → stereo, positioned by listener-relative source position via integer
**distance attenuation** + **constant-power pan** (ILD) + a small **integer inter-aural delay (ITD)**. All derived
from the normalized lateral component (NO `atan2`, NO trig, NO float). Pure-CPU integer, strict **bit-exact
buffer-hash golden** (no image, no Mac render-bake). Builds on DSP1–DSP4. HONEST: this is a pan/ILD/ITD model, NOT
true HRTF convolution (the one MetaSounds-comparison aside).

## Avoiding trig (the determinism discipline)
Listener faces +Z. The horizontal relative vector is `(dx, dz) = source - listener` (ignore Y for the azimuth).
`dist = isqrt(dx*dx + dz*dz)`. The **lateral component** `normX = clamp(dx * 32768 / max(dist,1), -32768, 32768)`
(Q15, in `[-1,1]`) is the pan/ITD driver — full-left `normX=-32768`, centre `0`, full-right `+32768`. NO `atan2`:
pan position `p = (normX + 32768) >> shift` indexes a committed constant-power LUT; `itd = (kMaxItd * |normX|) >>
15` and the sign of `normX` picks the near ear. Distance gain is one integer divide per block.

## The committed pan LUT (host-baked — the kSineTable discipline)
A `static const` constant-power table, 65 entries (pan position 0..64 → `[0,1]`): bake OFFLINE (throwaway calc)
`kPanLut[i] = { gainL: round(cos(i/64 * π/2) * 32767), gainR: round(sin(i/64 * π/2) * 32767) }`, `i = 0..64`.
Properties the bake MUST satisfy (assert in a comment): `kPanLut[0] = {32767, 0}` (full-left → R silent),
`kPanLut[64] = {0, 32767}` (full-right → L silent), `kPanLut[32] = {gL, gL}` with `gainL == gainR` exactly (centre
→ equal, ≈23170) — the centre symmetry is what makes the dead-centre L==R proof exact.

## The addition — `engine/audio/dsp.h` (APPEND-ONLY after the DSP4 block; keep self-contained)
Add to `hf::audio::dsp` (do NOT modify DSP1–4; keep dsp.h STANDALONE — only `<cstdint>`/`<vector>`, NO mixer/fpx
include):
- `inline int32_t IsqrtU(int64_t v)` — a deterministic integer sqrt (copy the `fpx.h::FxISqrt` Newton/bit
  algorithm shape, but LOCAL to dsp.h so the Mac single-file compile stays clean; pure integer, no `<cmath>`).
- `struct Vec3i { int32_t x = 0, y = 0, z = 0; };` (listener/source positions, integer world units).
- `struct PanGain { int32_t gainL, gainR; };` + the committed `static const PanGain kPanLut[65]`.
- `constexpr int kMaxItd = 32;` (max inter-aural delay in samples — ~0.67 ms @ 48 kHz).
- `struct SpatialNode { Vec3i listener; Vec3i source; int32_t refDist = 256; int16_t ringL[kMaxItd] = {}; int16_t ringR[kMaxItd] = {}; int ringPos = 0; };`
  (`refDist` = the unity-gain near distance; the ITD ring carries delay-line state across blocks.)
- `void SpatializeBlock(SpatialNode& sp, const std::vector<int16_t>& monoSrc, std::vector<int16_t>& outStereoInterleaved)`:
  compute `dx = source.x - listener.x`, `dz = source.z - listener.z`, `dist = IsqrtU((int64)dx*dx + (int64)dz*dz)`;
  `normX = clamp(dx * 32768 / max(dist,1), -32768, 32768)`; `panIdx = (int)((normX + 32768) >> 10)` clamped to
  `[0,64]` (1024 = 65536/64 → 65 buckets); `PanGain pg = kPanLut[panIdx]`; `gainDist = (int32_t)clamp((int64)
  sp.refDist * 32768 / max(dist, sp.refDist), 0, 32768)` (Q15, unity within `refDist`, inverse-ish beyond);
  `itd = (kMaxItd * (normX<0 ? -normX : normX)) >> 15` (0..kMaxItd); the near ear is the side `normX` points to
  (`normX<0` → left ear near → delay the RIGHT channel by `itd`; `normX>0` → delay the LEFT). For each mono sample
  `s`: `int16_t base = ClampI16(MulQ15(s, gainDist));` `int16_t l = ClampI16(MulQ15(base, pg.gainL));` `int16_t r =
  ClampI16(MulQ15(base, pg.gainR));` then apply the integer ITD via the ring buffers (push `l`/`r`, read the
  delayed sample for the far channel `itd` samples back, the near channel reads `itd=0`); push the resulting
  `(L,R)` interleaved to `outStereoInterleaved`. The ring state persists in `sp` across calls. NO `<cmath>`, NO float.
  (Keep the ITD implementation simple + obviously deterministic — a fixed-size `kMaxItd` ring per channel,
  `ringPos` carried; at `itd==0` the output equals the un-delayed sample so centre stays exact.)

## CPU test — extend `tests/dsp_test.cpp` (add a DSP5 section, keep DSP1–4 checks)
Assertions: (1) **dead-centre L==R (make-or-break)** — source directly in front (`dx=0`, `dz=refDist`): `normX=0`,
`itd=0`, `pg.gainL==pg.gainR` → the L and R channels are BYTE-IDENTICAL (deinterleave and compare); (2)
**full-left → R silent** — source far to the left (`dx` large negative): `pg.gainR==0` → every R sample is 0;
symmetric full-right → L silent; (3) **distance attenuation monotone** — a source at 2×`refDist` produces a lower
peak/energy than at `refDist`, and beyond grows quieter (deterministic monotone); (4) **block-boundary
determinism** — a moving-but-deterministic source (or a fixed off-centre source) spatialized as ONE `N*256` mono
buffer vs N separate 256-frame `SpatializeBlock` calls on ONE persistent `SpatialNode` → byte-identical (the ITD
ring carries); (5) **pinned hash** — `osc → spatial(off-centre)` stereo digest == a hard-pinned `uint64_t`; moving
the source changes the hash. Print the hashes + `dsp_test: ALL CHECKS PASSED`. Report lines:
```
dsp5-spatial: spatialization node (distance atten + constant-power pan + integer ITD, kMaxItd 32)
dsp5-spatial: dead-centre L==R BYTE-IDENTICAL {center:true}
dsp5-spatial: full-left -> R silent / full-right -> L silent {panEdges:ok}
dsp5-spatial: distance attenuation monotone {near:<vN>, far:<vF>} vF < vN
dsp5-spatial: block-boundary determinism: one-buffer == N-blocks {hash:0x<H>}
dsp5-spatial: osc->spatial pinned {hash:0x<Hs>}
```

## Proof (STRICT bit-exact buffer hash — cross-platform bar)
The golden is the pinned `DigestBuffer` in `dsp_test`; the cross-platform proof = the controller compiles+runs
`dsp_test` on the Mac (clang) → IDENTICAL hashes. Make-or-break: dead-centre L==R + the off-centre one-buffer ==
N-blocks (ITD ring carries).

## Constraints (HARD)
- APPEND-ONLY to engine/audio/dsp.h (do NOT modify DSP1–4) + extend tests/dsp_test.cpp. Keep dsp.h self-contained
  (only `<cstdint>`/`<vector>`, NO mixer/fpx include — copy a LOCAL `IsqrtU` rather than including fpx; the Mac
  clang single-file compile must keep working). Do NOT modify mixer.{h,cpp}, scene.wav, or any other file. NO
  RHI/shader/GPU. Pure-CPU integer. NO `<cmath>`, NO `atan2`, NO float at runtime (the pan LUT is host-baked
  offline — only the rounded Q15 int literals ship).
- Branch `fix-issue-26-dsp5`, commit there, do NOT merge. NO golden file (the pinned hash lives in the test).
- Build Windows + run dsp_test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target dsp_test'`
  then run dsp_test.exe (under build\windows-msvc-release).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, dsp_test prints ALL CHECKS PASSED, dead-centre
  L==R byte-identical, the pan edges silent the off-side, distance attenuation is monotone, one-buffer == N-blocks
  holds, the pinned hash matches, AND DSP1–4 checks still pass. Commit message via a temp file + `git commit -F`
  (use the Bash tool heredoc). Commit to the branch and STOP. Report: commit hash, the printed hashes, the baked
  kPanLut endpoints + centre (assert the centre symmetry), confirmation dsp_test passes (DSP1–5), and confirmation
  dsp.h stays self-contained. (The CONTROLLER runs dsp_test on the Mac to confirm identical hashes, then merges.)
