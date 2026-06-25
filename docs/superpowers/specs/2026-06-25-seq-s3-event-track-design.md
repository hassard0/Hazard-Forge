# Slice SEQ-S3 — Event track (deterministic timeline events, composes flow.h) (Issue #25)

S1 shipped the scalar keyframe track; S2 added easing + multi-track `Sequence`. S3 adds the other half of a
real timeline: an **event track** — discrete events that FIRE at exact tick boundaries as the playhead
sweeps a time interval (a cutscene triggers a sound / a Blueprint pulse / a gameplay flag on frame N). The
moat: the fired event SET is bit-identical cross-platform AND it composes with the just-shipped `flow.h`
deterministic-VM (issue #24) — a sequence event feeds a `flow::Graph` input channel and the flow trace is
itself deterministic. UE5 Sequencer event tracks fire on float playback timing → two machines can fire on
different frames; ours fires on integer ticks, identically, replayably.

Pure-CPU INTEGER, append-only to `engine/seq/seq.h`. S1+S2 stay UNTOUCHED + golden-invariant (re-assert
their digests). This slice ADDS one include: `#include "flow/flow.h"` (for `flow::EventRecord` +
`flow::DigestEvents` — the event-trace currency; flow.h is itself self-contained: `<cstddef>/<cstdint>/
<vector>` + `net/session.h`, so seq.h stays standalone-clang-compilable).

## Append to engine/seq/seq.h (below S2, in hf::seq)

### 1. The event track
```cpp
struct EventTrack {
    std::vector<fx>       times;     // Q16.16 seconds, STRICTLY ASCENDING (the invariant — sorted, no dupes)
    std::vector<uint32_t> eventIds;  // the event id fired at times[i]; eventIds.size()==times.size()
    std::vector<fx>       payloads;  // OPTIONAL Q16.16 payload per event; payloads.size()==times.size() OR 0
};
```
(If `payloads` is empty, events carry payload 0. Keep the parallel-SoA discipline of `ScalarTrack`.)

### 2. `SampleEvents` — fire every event in the half-open interval `[tPrev, t)`
```cpp
inline std::vector<flow::EventRecord> SampleEvents(const EventTrack& tr, fx tPrev, fx t);
```
- Fire every event whose `times[i]` satisfies `tPrev <= times[i] < t` (HALF-OPEN — `[tPrev, t)` — so a tick
  boundary fires exactly once across consecutive sweeps; the standard sampler-window convention). Ascending
  order (the track is sorted; emit in index order). Use a hand integer scan or the S1 `FindSegment`-style
  bound — NO `<algorithm>`.
- Map each fired event to a `flow::EventRecord`. **READ flow.h's actual `EventRecord` definition** and
  populate its fields from `eventIds[i]` (the event id / node-id slot) and `payloads[i]` (the value slot) —
  match flow's field names exactly. If `EventRecord` has a tick/order field, set it from `i` or the integer
  tick `times[i]` consistently (document the choice). The point: a `std::vector<flow::EventRecord>` that
  `flow::DigestEvents` can hash directly.
- Guard: `t <= tPrev` → empty (no negative/empty window fires).

### 3. `SampleEventSweep` — drive the track across N fixed ticks → one fired trace
```cpp
inline std::vector<flow::EventRecord> SampleEventSweep(const EventTrack& tr, fx dt, uint32_t n);
```
- For `i` in `[0, n)`: window `[i*dt, (i+1)*dt)` (int64 for the products), accumulate all fired
  `EventRecord`s into one contiguous trace (the digest input). Every event in `[0, n*dt)` fires exactly once,
  in ascending time order — the deterministic event stream.

### 4. Fixture (FIXED forever)
- `EventTrack MakeShowcaseEvents()` — events at fixed ticks with distinct ids + payloads, e.g.
  `times = {kOne/2, kOne, 3*kOne/2, 2*kOne, 5*kOne/2}` (0.5s..2.5s), `eventIds = {10, 20, 30, 20, 40}`
  (note the repeated id 20 — proves id is not a key), `payloads = {kOne, -kOne, kOne/4, 2*kOne, 0}`.
  (Keep FIXED — the golden pins the fired trace.)

## The goldens (PINNED, cross-platform) — append to tests/seq_test.cpp
```
seq-s3: event-sweep trace digest = 0x<...>  (<N> events)
PASS seq-s1/s2: ... (all prior assertions STILL green — S1 0xd314f17ebe3d480b + the S2 digests UNCHANGED)
PASS seq-s3: SampleEventSweep fires every event exactly once across the sweep (count == track size)
PASS seq-s3: the fired-event trace digest == pinned uint64 (flow::DigestEvents, byte-stable cross-platform)
PASS seq-s3: half-open window — an event exactly on a tick boundary fires once, not twice (no double-fire)
PASS seq-s3: empty/negative window [t,t) fires nothing
PASS seq-s3: nudging one event time re-buckets it — the trace digest changes (times are load-bearing)
PASS seq-s3: composition — a fired payload fed into a flow kInput channel yields the expected flow trace
```
Assertions:
1. **PRIOR INVARIANT** — re-assert S1 `0xd314f17ebe3d480b` and the three S2 table digests + the S2
   sequence-sweep digest, all UNCHANGED (S3 is purely additive).
2. **FIRES-ONCE COUNT** — `SampleEventSweep(MakeShowcaseEvents(), kOne/30, 90)` (3s @ 30Hz, covers all 5
   events) yields exactly `times.size()` (5) `EventRecord`s.
3. **PINNED TRACE DIGEST** — `flow::DigestEvents(SampleEventSweep(MakeShowcaseEvents(), kOne/30, 90))` == a
   pinned `uint64` (the fired stream is byte-stable MSVC + clang).
4. **HALF-OPEN NO-DOUBLE-FIRE** — sweep with `dt` chosen so an event time lands EXACTLY on a window
   boundary (e.g. `dt = kOne/2` makes event at `kOne/2` a boundary); assert that event fires exactly once
   total across the sweep (count its id in the trace == 1).
5. **EMPTY WINDOW** — `SampleEvents(track, 2*kOne, 2*kOne)` (and a negative window `t<tPrev`) returns empty.
6. **LOAD-BEARING TIME** — clone the showcase, shift `times[2]` into a different tick bucket (e.g.
   `+= kOne/30`), re-sweep → a DIFFERENT trace digest (event times are load-bearing).
7. **FLOW COMPOSITION** — build a tiny `flow::Graph` with a `kInput` channel; take a fired event's payload,
   feed it as that graph's input for one `StepGraph`/`Evaluate`, and assert the flow output equals the
   hand-computed expected (the sequence drives the deterministic VM — the headline composition). Use flow.h's
   actual `Graph`/`kInput`/`Evaluate`/`StepGraph` API (read it first; the flow_test.cpp fixtures are the
   reference for building a minimal graph).

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/seq/seq.h` + `engine/sim/fpx.h` + `engine/math/math.h` + `engine/flow/flow.h` +
`engine/net/session.h` + `tests/seq_test.cpp` (+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`)
to the Mac and runs `clang++ -std=c++20 -I engine -I tests tests/seq_test.cpp -o /tmp/seq && /tmp/seq`,
confirming ALL assertions PASS with IDENTICAL pinned digests. Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/seq/seq.h` (add `EventTrack`/`SampleEvents`/`SampleEventSweep`/`MakeShowcaseEvents`
  below S2). Do NOT modify any S1/S2 type or function semantics. S1 digest stays `0xd314f17ebe3d480b`; the
  S2 digests stay as pinned.
- The ONLY new include is `#include "flow/flow.h"` (place it next to `sim/fpx.h`/`net/session.h`). Header
  stays self-contained + standalone-clang-compilable (`-I engine -I tests`). Still NO `<cmath>`,
  `<algorithm>`, float, `<random>`, clock, `std::hash`. Do NOT modify `flow/flow.h` (read-only reuse of
  `EventRecord`/`DigestEvents`/`Graph`/`Evaluate`/`StepGraph`/`kInput`).
- Pure-CPU INTEGER. Event windowing is integer compares; the trace is hashed via `flow::DigestEvents`
  (flow's existing hand-LE event digest — do NOT roll your own).
- `tests/seq_test.cpp` stays self-contained; APPEND the S3 assertions + `MakeShowcaseEvents` + the minimal
  flow graph fixture. Keep ALL S1+S2 assertions green.
- Branch `fix-seq-s3`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target seq_test'`
  then run `seq_test`, confirm ALL assertions (S1+S2+S3) PASS, exit 0. ALSO compile standalone with the local
  clang `C:\Program Files\LLVM\bin\clang++.exe` and confirm the IDENTICAL digests. First run: pin the S3
  digest from the printed value, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `seq_test` builds + PASSES on Windows with
  every assertion green (esp. the prior-invariant digests + the half-open no-double-fire + the flow
  composition), and the local clang standalone passes with identical digests. Report: commit hash, full test
  output (printed digest + PASS lines), the pinned S3 `uint64`, confirmation S1+S2 digests are unchanged,
  confirmation the header is self-contained (list `#include`s — exactly the 6: `<cstddef>/<cstdint>/<vector>`
  + `sim/fpx.h` + `net/session.h` + `flow/flow.h`), the exact `flow::EventRecord` fields you populated and
  from what, and the local-clang result. Flag any deviation. Commit message via a temp file + `git commit -F`
  (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only + self-containment, runs the Mac/clang standalone for the identical
  digests, ff-merges to master + pushes + deletes the branch + advances to S4 — the transform/rotation track,
  the Q16.16 integer-nlerp rotation crux.)
