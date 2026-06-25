# Slice PROFILE-S5 — The SCRUB: serializable `.capture` + seek-to-frame via net::CatchUp (Issue #31)

S1–S4 built the capture event stream, scope tree, timeline, and render-structure inspection. S5 is THE
HEADLINE: a **serializable `.capture` artifact** whose structure is byte-stable, and a **scrub** — seek to
frame N reproduces the BIT-IDENTICAL structural state a from-frame-0 playback reaches at N — built on
`net::CatchUp` (the same primitive the seq SCRUB==SEEK used). And it makes the moat **testable**: the
`.capture` file has the structural bytes FIRST and the timing overlay LAST in a separate length-prefixed
section, so corrupting a timing byte leaves the structural digest unchanged while corrupting a structural
byte diverges at the exact frame.

Pure-integer, append-only to `engine/profile/profile.h` (below S4; do NOT modify S1–S4 — `0xedc7791443141dfd`
/ `0xb41eb67a1d13443e` / `0xc68ff46e1ab25f37` / `0x9b75187d6a4c3bf1` stay pinned). NO new include
(`net/session.h` already present — S5 USES `CatchUp`/`JoinSnapshot`/`RunLockstep`/`DigestTrace`/
`DesyncDetector`).

## The net::Session contract (confirmed — match exactly)
- `JoinSnapshot<World>{ uint32_t tick; World world; }`; `CatchUp(snap, toTick, tail, step) -> World` restores
  `snap.world` (as of `snap.tick`) and replays `tail.At(t)` for `t in [snap.tick, toTick)`.
- `RunLockstep(World init, ring, ticks, step, digest) -> uint64_t`; `DigestTrace(...) -> vector<uint64_t>`.
- `StepFn` is `step(World&, const std::vector<Input>&, uint32_t tick)`; `DigestFn` is `digest(const World&)`.
- `DesyncDetector` + `RecordLocal(d, tick, digest)` + `IngestRemote(d, ChecksumPacket{tick, digest})`.

## Append to engine/profile/profile.h (below S4, in hf::profile)

### 1. The `.capture` file format (two sections — structural FIRST, timing LAST)
```cpp
struct CaptureHeader {
    // magic "HFCAPF1\0" (8) — the FILE magic (distinct from S1's structural-section magic).
    uint32_t version          = 1;
    uint32_t frameCount       = 0;   // BuildFrameIndex(c).size()
    uint32_t nameCount        = 0;
    uint32_t eventCount       = 0;
    uint32_t structuralByteLen = 0;  // == EncodeStructural(c).size()  (the structural section length)
    uint32_t timingByteLen    = 0;   // == eventCount * 16  (cpuNanos u64 + gpuNanos u64 per event)
    uint32_t keyframeInterval = 0;   // frames between seek keyframes (>=1)
};
constexpr std::size_t kCaptureHeaderLen = 8 /*magic*/ + 7 * 4 /*u32 fields*/;  // = 36
```
```cpp
// EncodeCapture: [magic+header][structuralSection = EncodeStructural(c) VERBATIM][timingSection].
// The structural section IS S1's EncodeStructural output, so its digest == StructuralDigest(c). The timing
// section is per event: PutU64(cpuNanos), PutU64(gpuNanos) — it starts at offset kCaptureHeaderLen +
// structuralByteLen, PROVABLY outside the structural digest's byte range.
inline std::vector<uint8_t> EncodeCapture(const Capture& c, uint32_t keyframeInterval = 1);
// DecodeCapture: inverse — parse header, parse the structural section back to names+events (the inverse of
// EncodeStructural), parse the timing section into timings. Returns false on truncation / bad magic.
inline bool DecodeCapture(const std::vector<uint8_t>& bytes, Capture& out);
// CaptureStructuralDigest: DigestBytes over ONLY the structural section bytes [kCaptureHeaderLen ..
// kCaptureHeaderLen+structuralByteLen). Equals StructuralDigest(c) by construction. Timing is excluded by
// byte-offset construction (it lives at >= kCaptureHeaderLen+structuralByteLen).
inline uint64_t CaptureStructuralDigest(const std::vector<uint8_t>& bytes);
```
(Reuse S1's `EncodeStructural` for the structural section + add an inverse parser `DecodeStructural(bytes,
off, len, Capture&)`. Reuse the inline `GetU32`/`GetU64` — add them if S2/S3 didn't.)

### 2. The scrub playback world + net::Session wiring
```cpp
// The playback world: the current frame + a running fold of every frame's structural digest seen so far.
// Flat + value-copyable so net::Session's snapshot is complete by construction (the seq.h discipline).
struct CaptureWorld { uint32_t currentFrame = 0; uint64_t acc = 0; };
// DigestCaptureWorld: hand-LE (currentFrame, acc) -> DigestBytes (the per-frame replay checksum).
inline uint64_t DigestCaptureWorld(const CaptureWorld& w);
```
The TEST builds the `step`/`digest` lambdas capturing `const std::vector<FrameIndex>& frames`:
`step(w, inputs, t) { w.currentFrame = t; if (t < frames.size()) w.acc = Mix(w.acc,
frames[t].structuralDigest); }` (a deterministic fold — `Mix` is an inline `(acc*1099511628211u) ^ digest`
or FNV step; provide an inline `Mix(uint64,uint64)->uint64` in the header). `digest = [](const CaptureWorld&
w){ return DigestCaptureWorld(w); }`.
```cpp
// SeekToFrame: restore the nearest keyframe <= N and replay forward to N via net::CatchUp — the structural
// state at N is BIT-IDENTICAL to a from-0 playback. A convenience over net::CatchUp for the capture.
// (Provide it as a small wrapper the test can call; the test ALSO drives net::CatchUp directly to prove the
// composition.)
template <class StepFn>
inline CaptureWorld SeekToFrame(const std::vector<FrameIndex>& frames, uint32_t toFrame,
                                const CaptureWorld& keyframeWorld, uint32_t keyframeFrame,
                                const hf::net::InputRing<uint32_t>& tail, StepFn step);
```
(`SeekToFrame` is literally `net::CatchUp(JoinSnapshot<CaptureWorld>{keyframeFrame, keyframeWorld}, toFrame,
tail, step)` — keep the wrapper thin; the point is the composition.)

### 3. `VerifyCapture` — structural integrity via net::DesyncDetector
```cpp
// Replay the from-0 per-frame world digests as the "local" trace; compare against an expected per-frame
// digest vector (the authority). Returns {bool ok, uint32_t firstBadFrame}. A structural corruption diverges
// at the exact frame; a timing corruption does NOT (the per-frame digest is over structural events only).
struct VerifyResult { bool ok = true; uint32_t firstBadFrame = 0; };
inline VerifyResult VerifyCapture(const Capture& decoded, const std::vector<uint64_t>& expectedFrameDigests);
```
(Build via `net::DesyncDetector` + `RecordLocal` over the decoded capture's `BuildFrameIndex` per-frame
digests, `IngestRemote` the expected — the replay.h `VerifyReplay` pattern.)

### 4. Fixture
- Reuse `MakeTimelineCapture()` (S3's 4-frame capture) but POPULATE its `timings` with FIXED nonzero values
  (e.g. `timings[i] = { (i+1)*1000, (i+1)*7 }`) via a `MakeTimelineCaptureTimed()` helper, so the timing
  section is non-empty + corruptible. Keep FIXED.

## The golden (PINNED, cross-platform) — append to tests/profile_test.cpp
```
profile-s5: capture file bytes = <N>, structural-section digest = 0x<...>
PASS profile-s1..s4: ... (all prior assertions STILL green — every prior digest UNCHANGED)
PASS profile-s5: DecodeCapture(EncodeCapture(c)) round-trips — StructuralDigest + timings recovered
PASS profile-s5: the .capture structural-section digest == StructuralDigest(c) (the section IS S1's encoding)
PASS profile-s5: SCRUB==SEEK — CatchUp(keyframe@K, N) world == from-0 playback world at N (bit-identical), several (K,N)
PASS profile-s5: corrupting a TIMING byte leaves the structural digest UNCHANGED (the moat, made testable)
PASS profile-s5: corrupting a STRUCTURAL byte changes the digest AND VerifyCapture diverges at the exact frame
PASS profile-s5: EncodeCapture is deterministic — two encodes are byte-identical
```
Assertions:
1. **PRIOR INVARIANT** — re-assert S1 `0xedc7791443141dfd`, S2 `0xb41eb67a1d13443e`, S3 `0xc68ff46e1ab25f37`,
   S4 `0x9b75187d6a4c3bf1`, all UNCHANGED.
2. **ROUND-TRIP** — `DecodeCapture(EncodeCapture(c), out)` succeeds; `StructuralDigest(out) ==
   StructuralDigest(c)`; `out.timings == c.timings` (the overlay round-trips too).
3. **STRUCTURAL SECTION == S1** — `CaptureStructuralDigest(EncodeCapture(MakeTimelineCaptureTimed()))` ==
   `StructuralDigest(MakeTimelineCaptureTimed())` == a hard-pinned `uint64_t` (the .capture's structural bytes
   ARE the S1 encoding; identical MSVC + clang).
4. **SCRUB == SEEK (THE HEADLINE)** — for several `(K, N)` pairs (e.g. K=0→N=3, K=1→N=3, K=2→N=2): build
   `frames = BuildFrameIndex(c)`; compute the from-0 world at N via a `RunLockstep`-style loop (or
   `DigestTrace`); compute the keyframe world at K the same way; then `SeekToFrame(frames, N, worldAtK, K,
   tail, step)` (== `net::CatchUp`) → assert `DigestCaptureWorld(seeked) == DigestCaptureWorld(fromZeroAtN)`
   (full-digest equality — seek == play). Assert `seeked.currentFrame == N`.
5. **MOAT — TIMING CORRUPTION IS HARMLESS** — `auto bytes = EncodeCapture(c)`; flip a byte at offset
   `kCaptureHeaderLen + structuralByteLen` (the first timing byte) → `CaptureStructuralDigest(bytes)` ==
   pinned (UNCHANGED); `DecodeCapture` still yields the same structural content (`StructuralDigest` unchanged),
   only `timings` differ; `VerifyCapture(decoded, expected)` reports `ok == true`.
6. **STRUCTURAL CORRUPTION DIVERGES** — flip a byte INSIDE the structural section (offset `kCaptureHeaderLen
   + k` for a small k past the magic) → `CaptureStructuralDigest` DIFFERS from pinned; and `VerifyCapture`
   on the corrupted decode reports `ok == false` with `firstBadFrame` == the affected frame.
7. **DETERMINISTIC** — two `EncodeCapture(c)` calls → byte-identical.

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/profile/profile.h` + `engine/net/session.h` + `tests/profile_test.cpp` (+
`tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine
-I tests tests/profile_test.cpp -o /tmp/profile && /tmp/profile`, confirming ALL assertions PASS with the
IDENTICAL pinned digests (esp. the structural-section digest + the SCRUB==SEEK equality). Local Windows clang
is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/profile/profile.h` (add `CaptureHeader`/`kCaptureHeaderLen`/`EncodeCapture`/
  `DecodeCapture`/`DecodeStructural`/`CaptureStructuralDigest`/`CaptureWorld`/`DigestCaptureWorld`/`Mix`/
  `SeekToFrame`/`VerifyResult`/`VerifyCapture`/`MakeTimelineCaptureTimed` + inline `GetU32`/`GetU64` if
  missing, below S4). Do NOT modify S1–S4 — all prior digests stay pinned.
- NO new include (`net/session.h` present). Header stays self-contained + standalone-clang-compilable. STILL
  NO `<string>`/`<cmath>`/clock/RNG/`<unordered_*>`/`<map>`/`std::hash`/`<algorithm>`, NO recursion. Do NOT
  modify `net/session.h` (reuse `CatchUp`/`JoinSnapshot`/`DesyncDetector` read-only).
- The structural digest covers ONLY the structural section bytes; timing lives at a byte offset PAST it
  (provably excluded). mtime/clock NEVER enters the structural digest.
- `tests/profile_test.cpp` stays self-contained; APPEND the S5 assertions + the fixtures + the lambdas/ring.
  (Add a `namespace net = hf::net;` alias if convenient — does not change S1–S4.) Keep ALL S1–S4 green.
- Branch `fix-profile-s5`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target profile_test'`
  (if the `cmd /c '...&&...'` handoff swallows output, run the build via the PowerShell tool natively). Run
  `profile_test`, confirm ALL assertions (S1–S5) PASS, exit 0. ALSO compile standalone with the local clang
  `C:\Program Files\LLVM\bin\clang++.exe` and confirm IDENTICAL digests. First run: pin the structural-section
  digest, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `profile_test` builds + PASSES on Windows
  with every assertion green (esp. prior invariants + round-trip + the SCRUB==SEEK equality + the
  timing-corruption-harmless + structural-corruption-diverges moat tests), and the local clang standalone
  passes with identical digests. Report: commit hash, full test output (printed digest + PASS lines), the
  pinned structural-section `uint64`, confirmation S1–S4 digests unchanged, confirmation includes unchanged,
  how you wired the SCRUB==SEEK via net::CatchUp (the K,N pairs), and the local-clang result. Flag any
  deviation. Commit message via a temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S6 — the live ScopedZone capstone + optional
  --profile-shot.)
