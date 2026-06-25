# Slice PROFILE-S3 — Frame boundaries + multi-frame timeline (Issue #31)

S1 gave the capture event stream + structural digest (`0xedc7791443141dfd`); S2 the scope tree
(`0xb41eb67a1d13443e`). S3 adds **frame boundaries** — `FrameBegin`/`FrameEnd` markers bracket each frame's
events — and a **timeline**: a per-frame index where every frame carries its OWN structural sub-digest (the
digest of just that frame's events). This is the Insights timeline view: a row of frames, each a
byte-reproducible cell. Crucially, **a frame's structural digest depends on only that frame's events** (no
cross-frame state), which is what makes S5's seek-to-frame exact.

Pure-integer, append-only to `engine/profile/profile.h` (below S2; do NOT modify S1/S2 — `0xedc7791443141dfd`
and `0xb41eb67a1d13443e` stay pinned). NO new include.

## Append to engine/profile/profile.h (below S2, in hf::profile)

### 1. Frame-marker emitters
```cpp
// FrameBegin carries the frame number in `a`; FrameEnd is a bare marker. (EvKind::FrameBegin/FrameEnd are
// already defined in S1.) Each pushes the event + a zero TimingSample (the S1 emitter discipline — keep
// events.size() == timings.size()).
inline void EmitFrameBegin(Capture& c, uint32_t frameNumber);   // {FrameBegin, 0, frameNumber, 0}
inline void EmitFrameEnd  (Capture& c);                          // {FrameEnd,   0, 0, 0}
```

### 2. Per-frame structural sub-digest (events-only, position-independent)
```cpp
// Encode a SLICE of the event stream [first, first+count) as hand-LE records (NO names, NO absolute
// position): PutU32(count), then per event PutU32((uint32_t)kind), PutU32(nameId), PutU32(a), PutU32(b).
inline std::vector<uint8_t> EncodeFrameEvents(const Capture& c, uint32_t first, uint32_t count);
inline uint64_t FrameStructuralDigest(const Capture& c, uint32_t first, uint32_t count) {
    auto b = EncodeFrameEvents(c, first, count); return net::DigestBytes(b.data(), b.size());
}
```
(The per-frame digest is over the frame's event RECORDS only — so two frames with the identical event
sequence in the same capture get the IDENTICAL digest, regardless of frame number or absolute offset. The
`nameId`s are stable interned ids shared across frames, so same-name scopes compare equal. This
position-independence is the seek-exactness property.)

### 3. The frame index (the timeline)
```cpp
struct FrameIndex {
    uint32_t frameNumber     = 0;   // the FrameBegin's `a`
    uint32_t firstEvent      = 0;   // index of this frame's FrameBegin in c.events
    uint32_t eventCount      = 0;   // number of events in [FrameBegin .. FrameEnd] inclusive
    uint64_t structuralDigest = 0;  // FrameStructuralDigest over this frame's events (the timeline cell)
};
// Split the event stream on FrameBegin/FrameEnd markers; one FrameIndex per complete frame. Events outside
// any frame (before the first FrameBegin / after the last FrameEnd / an unclosed trailing frame) are handled
// deterministically: an unclosed final frame runs to the end of the stream. Integer walk, no recursion.
inline std::vector<FrameIndex> BuildFrameIndex(const Capture& c);
```
- Walk `c.events`: on `FrameBegin` record `firstEvent = i`, `frameNumber = ev.a`, open a frame; on `FrameEnd`
  close it (`eventCount = i - firstEvent + 1`, compute `structuralDigest = FrameStructuralDigest(c,
  firstEvent, eventCount)`), append the `FrameIndex`. If the stream ends with a frame still open, close it at
  the last event (`eventCount = events.size() - firstEvent`) — deterministic, never UB.

### 4. `DigestTimeline` — the whole-timeline digest
```cpp
// Hand-LE over the frame index: PutU32(frameCount), then per frame PutU32(frameNumber), PutU32(eventCount),
// PutU64(structuralDigest). Then net::DigestBytes.
inline uint64_t DigestTimeline(const std::vector<FrameIndex>& frames);
```

### 5. Fixture (FIXED forever) — a 4-frame capture with two identical frames
- `Capture MakeTimelineCapture()` — intern `"Frame"`, `"Shadow"`, `"Lit"`, `"Post"`. Build **4 frames**:
  - frame 0: `FrameBegin(0); Enter Shadow; Draw(Shadow,2); Exit Shadow; FrameEnd`
  - frame 1: `FrameBegin(1); Enter Shadow; Draw(Shadow,2); Exit Shadow; FrameEnd` — **the identical workload
    to frame 0** (different frame number, same events) → its `structuralDigest` MUST equal frame 0's.
  - frame 2: `FrameBegin(2); Enter Lit; Draw(Lit,5); Exit Lit; FrameEnd` — a different workload.
  - frame 3: `FrameBegin(3); Enter Post; Draw(Post,1); Exit Post; FrameEnd` — another.
  (Keep FIXED forever — the golden pins its timeline.)

## The golden (PINNED, cross-platform) — append to tests/profile_test.cpp
```
profile-s3: timeline digest = 0x<...>  (<F> frames)
PASS profile-s1/s2: ... (all prior assertions STILL green — 0xedc7791443141dfd + 0xb41eb67a1d13443e UNCHANGED)
PASS profile-s3: BuildFrameIndex(timeline) has 4 frames, each bracketed FrameBegin..FrameEnd
PASS profile-s3: DigestTimeline == pinned uint64 (the multi-frame timeline is byte-stable cross-platform)
PASS profile-s3: per-frame reproducibility — frame 0 and frame 1 (identical workload) have the SAME structuralDigest
PASS profile-s3: a different-workload frame (frame 2) has a DIFFERENT structuralDigest (frames distinguish workloads)
PASS profile-s3: a frame's structuralDigest depends only on its OWN events (position-independent — the seek property)
PASS profile-s3: TIMING STILL EXCLUDED — filling timings nonzero leaves the timeline digest unchanged
```
Assertions:
1. **PRIOR INVARIANT** — re-assert S1 `0xedc7791443141dfd` and S2 `0xb41eb67a1d13443e`, both UNCHANGED.
2. **FRAME COUNT** — `BuildFrameIndex(MakeTimelineCapture())` has exactly 4 `FrameIndex` entries with
   `frameNumber` 0,1,2,3 and each `eventCount == 5` (FrameBegin + 3 scope/draw + FrameEnd).
3. **PINNED TIMELINE** — `DigestTimeline(BuildFrameIndex(MakeTimelineCapture()))` == a hard-pinned `uint64_t`
   (run once, pin THAT; identical MSVC + clang).
4. **PER-FRAME REPRODUCIBILITY** — `frames[0].structuralDigest == frames[1].structuralDigest` (identical
   workload, different frame number → identical structural cell — the byte-reproducible-frame property).
5. **FRAMES DISTINGUISH WORKLOADS** — `frames[2].structuralDigest != frames[0].structuralDigest` (a Lit-draw
   frame differs from a Shadow-draw frame).
6. **POSITION-INDEPENDENT** — construct a SECOND capture that is frame 0's workload alone (one frame), and
   assert its single frame's `structuralDigest == frames[0].structuralDigest` from the 4-frame capture (the
   digest depends on the events, not the absolute offset — the seek-exactness property S5 relies on).
7. **TIMING EXCLUDED** — fill `timings` nonzero; assert `DigestTimeline` (and each per-frame digest) is
   UNCHANGED.

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/profile/profile.h` + `engine/net/session.h` + `tests/profile_test.cpp` (+
`tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine
-I tests tests/profile_test.cpp -o /tmp/profile && /tmp/profile`, confirming ALL assertions PASS with the
IDENTICAL pinned digests. Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/profile/profile.h` (add `EmitFrameBegin`/`EmitFrameEnd`/`EncodeFrameEvents`/
  `FrameStructuralDigest`/`FrameIndex`/`BuildFrameIndex`/`DigestTimeline`/`MakeTimelineCapture` below S2). Do
  NOT modify S1/S2 — `0xedc7791443141dfd` and `0xb41eb67a1d13443e` stay pinned.
- NO new include. Self-contained (4 includes). STILL NO `<string>`/`<cmath>`/clock/RNG/`<unordered_*>`/
  `<map>`/`std::hash`/`<algorithm>`, NO recursion (integer walk).
- A frame's structural digest is over its OWN events ONLY (no cross-frame state, no `timings`).
- `tests/profile_test.cpp` stays self-contained; APPEND the S3 assertions + `MakeTimelineCapture`. Keep ALL
  S1+S2 assertions green.
- Branch `fix-profile-s3`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target profile_test'`
  (if the `cmd /c '...&&...'` handoff swallows output, run the build via the PowerShell tool natively). Run
  `profile_test`, confirm ALL assertions (S1–S3) PASS, exit 0. ALSO compile standalone with the local clang
  `C:\Program Files\LLVM\bin\clang++.exe` and confirm the IDENTICAL digests. First run: pin the timeline
  digest, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `profile_test` builds + PASSES on Windows
  with every assertion green (esp. prior invariants + the pinned timeline + per-frame reproducibility +
  position-independence + timing-excluded), and the local clang standalone passes with identical digests.
  Report: commit hash, full test output (printed digest + PASS lines), the pinned timeline `uint64`,
  confirmation S1/S2 digests unchanged, confirmation includes unchanged (4, no recursion), the per-frame
  digests you observed, and the local-clang result. Flag any deviation. Commit message via a temp file +
  `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S4 — draw-call / GPU-pass inspection via injected render
  structure.)
