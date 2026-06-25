# Slice PROFILE-S6 — Live ScopedZone capstone (Issue #31)

S1–S5 built + PROVED the deterministic, scrub-seekable capture with the structure-vs-timing split (all on
synthetic/scripted captures). S6 is the **live capstone**: a `ScopedZone` RAII helper that, in a REAL frame,
emits the structural enter/exit events AND measures wall-clock time into the timing overlay — proving the
split holds with **real measured timing**, not just synthetic. The structural digest of a live-captured frame
is byte-identical to the scripted expectation (structure golden, deterministic), while the timing slots carry
genuine, non-deterministic nanosecond measurements.

**THE ISOLATION (the seq_render.h precedent):** the ONE non-deterministic crossing — reading the clock — is
isolated in a SEPARATE header `engine/profile/profile_live.h` (which may include `<chrono>`), so the bit-exact
`engine/profile/profile.h` stays `<chrono>`-free and standalone-clang-pure. `ScopedZone` writes ONLY into the
timing overlay (`timings[]`); it NEVER touches the structural column (`events[]` is built by the same S1
`EmitEnter`/`EmitExit` used by the scripted path). The structural digest path never reads `timings[]` (the S1
invariant). This is the honest proof: real timing in, structure still golden out.

Append-only — `profile.h` is UNTOUCHED (S6 adds a NEW sibling header `profile_live.h` + a new test). All S1–S5
pinned digests stay (`0xedc7791443141dfd` / `0xb41eb67a1d13443e` / `0xc68ff46e1ab25f37` / `0x9b75187d6a4c3bf1`
/ `0x9830afc651699a70`).

## NEW file: engine/profile/profile_live.h (namespace hf::profile)
Includes `#include "profile/profile.h"` + `#include <chrono>` (the ONLY place `<chrono>` is allowed in the
profiler). Header-only. This header is NOT on the bit-exact standalone-clang path — it is a live helper
(timing is non-deterministic by nature). It compiles on MSVC + clang (chrono is portable).

### 1. `ScopedZone` — RAII scope timing (the one clock crossing)
```cpp
// On construction: EmitEnter(c, nameId) and record (a) the enter event's index, (b) the start time.
// On destruction: EmitExit(c, nameId), then write the measured duration into the ENTER event's timing slot:
//   c.timings[enterIdx].cpuNanos = (now - start) in nanoseconds.
// Structure (events) is built by the SAME S1 emitters as the scripted path → identical structural digest.
// Timing is written ONLY to the overlay → never affects the structural digest.
class ScopedZone {
public:
    ScopedZone(Capture& c, uint32_t nameId)
        : c_(&c), enterIdx_((uint32_t)c.events.size()), start_(std::chrono::steady_clock::now()) {
        EmitEnter(c, nameId);
    }
    ~ScopedZone() {
        EmitExit(*c_, c_->events[enterIdx_].nameId);
        const auto end = std::chrono::steady_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
        c_->timings[enterIdx_].cpuNanos = (uint64_t)(ns < 0 ? 0 : ns);  // clamp; monotonic clock so >= 0
    }
    ScopedZone(const ScopedZone&) = delete; ScopedZone& operator=(const ScopedZone&) = delete;
private:
    Capture* c_; uint32_t enterIdx_; std::chrono::steady_clock::time_point start_;
};
```
(Note `enterIdx_` is captured BEFORE `EmitEnter` so it points at the enter event. The dtor uses the recorded
`nameId` for the exit. The dtor writes the duration into the enter slot — the standard "scope duration lives
on the enter marker" convention.)

### 2. `ScopedFrame` — RAII frame bracket
```cpp
// ctor: EmitFrameBegin(c, frameNumber); dtor: EmitFrameEnd(c). (No timing on frame markers in v1 — the frame
// duration is the sum of its zones, which the overlay already carries; keep v1 simple.)
class ScopedFrame { /* same RAII shape over EmitFrameBegin/EmitFrameEnd */ };
```

### 3. A live capture helper (builds the showcase via RAII, with optional measurable work)
```cpp
// Build a capture whose STRUCTURE matches MakeShowcaseCapture() exactly, but via ScopedZone RAII + a busy
// loop inside each zone so the measured nanos are real and > 0. `work` controls the loop size (the test
// passes different `work` to two runs to show timing varies while structure stays identical). Returns the
// Capture (events match the scripted showcase; timings carry real measurements).
inline Capture BuildLiveShowcase(const NameTable& seedNames, uint64_t work);
```
(Intern the SAME names in the SAME order as `MakeShowcaseCapture` — "Frame"/"Shadow"/"Lit" → ids 0/1/2 — so
the structural digest matches. Inside each `ScopedZone`, run a `volatile`-accumulating busy loop of `work`
iterations so `steady_clock` measures a real, nonzero, work-dependent duration. Emit the same draws
(`EmitDraw(c, shadow, 2)` etc.) as the scripted showcase. The structure is identical to
`MakeShowcaseCapture()`; only the timing slots differ.)

## RenderGraph integration seam (documented, not a new golden)
The live engine wires real draw-call inspection by calling S4's `IngestRenderStructure(RenderStructInput)`
with data built from `render::RenderGraph::LastOrder()` + per-pass draw enumeration + MDI `drawCount` — the
POD boundary already exists (S4). This is mechanical glue with no new determinism content, so S6 does NOT add
a render golden for it; the S4 digest already proves the structure path. (A future `--profile-shot` timeline
visualization via `engine/debug/debug_draw.*` is OPTIONAL and SKIPPED here — the flagship's value is the
deterministic capture, fully proven in S1–S5.)

## The golden / liveness test — tests/profile_live_test.cpp (a LIVE test, NOT standalone-clang-pinned timing)
Self-contained test (copy the scaffold). Register `hf_add_pure_test(profile_live_test)` in
`tests/CMakeLists.txt`. It includes `profile/profile_live.h`. (It is "pure" in the build sense — links
`hf_core` — but exercises real timing; the STRUCTURAL assertions are the deterministic goldens, the timing
assertions are liveness checks.)
```
profile-s6: live structural digest = 0x<...>  (run A nanos = <..>, run B nanos = <..>)
PASS profile-s6: a live ScopedZone capture's StructuralDigest == the scripted MakeShowcaseCapture digest (0xedc7791443141dfd)
PASS profile-s6: ScopedZone built the same structural events as the scripted EmitEnter/EmitExit (RAII provenance)
PASS profile-s6: real timing is populated — a measured zone's cpuNanos > 0 after actual work
PASS profile-s6: STRUCTURE IS GOLDEN UNDER REAL TIMING — two runs with DIFFERENT work yield the IDENTICAL structural digest
PASS profile-s6: ...and the two runs' measured timings are recorded (the non-deterministic overlay, printed)
```
Assertions:
1. **LIVE STRUCTURE == SCRIPTED** — `StructuralDigest(BuildLiveShowcase(seed, work1))` ==
   `StructuralDigest(MakeShowcaseCapture())` == `0xedc7791443141dfd` (the RAII path produces the byte-identical
   structure; deterministic golden).
2. **RAII PROVENANCE** — `BuildLiveShowcase(...).events` equals `MakeShowcaseCapture().events`
   field-for-field (same kinds/nameIds/a/b in the same order — the RAII helper is a faithful structural twin).
3. **TIMING POPULATED** — after a substantial busy loop, at least one zone's `cpuNanos > 0` (the clock crossing
   works; use a loop large enough that `steady_clock` reliably measures > 0 — e.g. ≥ 1e7 iterations
   accumulating into a `volatile`).
4. **STRUCTURE GOLDEN UNDER REAL TIMING (the honest capstone)** — `StructuralDigest(BuildLiveShowcase(seed,
   work1)) == StructuralDigest(BuildLiveShowcase(seed, work2))` for `work1 != work2` (different real
   durations, IDENTICAL structure — the split holds with measured timing, the whole flagship thesis on live
   data). Both also == `0xedc7791443141dfd`.
5. **TIMINGS ARE REAL (informational, non-flaky)** — record/print the two runs' measured nanos for the "Lit"
   zone; assert each `> 0` (do NOT assert they DIFFER — that would be flaky; the point is they are real
   measurements that do not affect the structural digest). (Optionally assert `runB nanos >= runA nanos` only
   if `work2 > work1` by a large margin AND guard it as a soft check — prefer NOT asserting ordering to avoid
   flakiness; printing is enough.)

## Cross-platform proof
The CONTROLLER `scp`s `engine/profile/profile.h` + `engine/profile/profile_live.h` + `engine/net/session.h` +
`tests/profile_live_test.cpp` (+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs
`clang++ -std=c++20 -I engine -I tests tests/profile_live_test.cpp -o /tmp/plive && /tmp/plive`, confirming
the STRUCTURAL assertions PASS with the IDENTICAL pinned structural digest `0xedc7791443141dfd` (the timing
values will differ Mac vs Windows — that is the POINT; only the structure is golden). Local Windows clang is
the fast pre-check.

## Constraints (HARD)
- NEW `engine/profile/profile_live.h` (includes `profile/profile.h` + `<chrono>`). Do NOT modify
  `engine/profile/profile.h` (S1–S5 untouched — all pinned digests stay). `<chrono>` is allowed ONLY in
  `profile_live.h`, NOWHERE in `profile.h`.
- `ScopedZone` writes ONLY to `timings[]` (the overlay); it builds structure via the S1 `EmitEnter`/`EmitExit`
  ONLY. The structural digest is NEVER affected by a measured duration (re-proven by assertion 4).
- `tests/profile_live_test.cpp` is self-contained; register `hf_add_pure_test(profile_live_test)`. The
  STRUCTURAL assertions are deterministic goldens; timing assertions are `> 0` liveness checks (do NOT assert
  timing values or that two timings differ — non-flaky discipline). Keep the existing `profile_test` (S1–S5)
  green (do not touch it).
- Branch `fix-profile-s6`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`. (The optional
  `--profile-shot` is SKIPPED — report it was intentionally skipped.)
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target profile_live_test'`
  (if the `cmd /c '...&&...'` handoff swallows output, run the build via the PowerShell tool natively). Run
  `profile_live_test`, confirm ALL assertions PASS, exit 0. ALSO run the existing `profile_test` to confirm
  S1–S5 still green. ALSO compile `profile_live_test` standalone with the local clang `C:\Program Files\LLVM\
  bin\clang++.exe` and confirm the IDENTICAL structural digest.
- COMPLETION CRITERIA — do NOT commit until `profile_live.h` compiles, `profile_live_test` builds + PASSES on
  Windows (structural digest == `0xedc7791443141dfd`, timing > 0), `profile_test` (S1–S5) still green, and the
  local clang standalone passes with the identical structural digest. Report: commit hash, full test output
  (printed structural digest + the two runs' nanos + PASS lines), confirmation S1–S5 digests unchanged
  (profile.h untouched), confirmation `<chrono>` is ONLY in profile_live.h, that `--profile-shot` was skipped,
  and the local-clang result. Flag any deviation. Commit message via a temp file + `git commit -F` (Bash
  heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits, runs the Mac/clang standalone for the identical structural digest, ff-merges to
  master + pushes + deletes the branch, then writes the ARCHITECTURE.md profiler section + comments issue #31
  — COMPLETING flagship #31.)
