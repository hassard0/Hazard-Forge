# Slice SEQ-S5 — Lockstep / replay / SCRUB via net::Session: the moat headline (Issue #25)

S1–S4 built the deterministic Q16.16 timeline (scalar/easing/event/transform tracks). S5 is THE HEADLINE:
wrap timeline evaluation as a `net::Session` `StepFn` so the whole sequence becomes **lockstep-replayable,
desync-detectable, and SCRUB-able** — and prove the moat property UE5 Sequencer's float playback timing
CANNOT do: **seeking to tick S then playing forward is BIT-IDENTICAL to playing from tick 0** (a deterministic
timeline scrub). Two machines scrubbing the same cutscene land on the identical frame, every bit.

Pure-CPU INTEGER, append-only to `engine/seq/seq.h`. S1–S4 stay UNTOUCHED + golden-invariant. ONE new
include: `#include "net/session.h"` is ALREADY pulled in (S1) — so NO new include; S5 just USES more of it
(`RunLockstep`/`DigestTrace`/`RollbackSession`/`RunWithTransport`/`ScriptedTransport`/`JoinSnapshot`/
`CatchUp`/`DesyncDetector`). The 6 includes stay exactly as S4 left them. Do NOT modify `net/session.h`.

## The net::Session contract (confirmed from the header — match it EXACTLY)
- `StepFn` is called `step(World& world, const std::vector<Input>& inputsThisTick, uint32_t tick)`.
- `DigestFn` is `digest(const World&) -> uint64_t`.
- `RunLockstep(World init, const InputRing<Input>& ring, uint32_t ticks, step, digest) -> uint64_t` (final).
- `DigestTrace(init, ring, ticks, step, digest) -> std::vector<uint64_t>` (digest AFTER each tick).
- Rollback path hands the step a TWO-element input vector `{ localThisTick, remoteThisTick }` (fixed order)
  — see `StepPredicted`/`ConfirmRemote`. So the playhead step must accept a vector and SUM its elements
  (one element in plain lockstep, two in rollback).
- `JoinSnapshot<World>{ uint32_t tick; World world; }` + `CatchUp(snap, toTick, tail, step) -> World`:
  restore `snap.world` (the world AS OF `snap.tick`) and replay `tail.At(t)` for `t in [snap.tick, toTick)`.
  THIS is the scrub/seek primitive.

## Append to engine/seq/seq.h (below S4, in hf::seq)

### 1. The playhead World — a FLAT, value-copyable struct (snapshot-complete by construction)
```cpp
struct SeqPlayhead {
    fx              time = 0;   // current Q16.16 timeline position
    std::vector<fx> bus;        // the value-bus sampled at `time` (the readable outputs)
};
```
(Flat + value-copyable → `net::Session`'s value-copy snapshot captures the WHOLE state by construction —
the verdict.h completeness argument. `bus` is derived from `time` each step, so the TRUE state is `time`;
the completeness test proves a restore that drops `time` diverges.)

### 2. The deterministic transition + digest
```cpp
// Advance the playhead by the SUM of this tick's inputs (delta in Q16.16 seconds), then resample the bus.
// `inputs` is net::Session's per-tick input vector: {delta} in lockstep, {local,remote} in rollback.
inline void StepPlayhead(const Sequence& seq, SeqPlayhead& w, const std::vector<fx>& inputs, uint32_t /*tick*/) {
    fx delta = 0;
    for (fx d : inputs) delta += d;          // sum (1 elem lockstep, 2 elems rollback) — fixed order
    w.time += delta;
    w.bus = SampleSequence(seq, w.time);     // resample the multi-track value-bus at the new time
}

// Hand-LE digest of the playhead: time (1 fx) then the bus (N fx) into a contiguous fx buffer -> DigestTrack
// (S1's net::DigestBytes). Padding-safe + byte-stable (NEVER DigestBytes the struct — the vector has a heap
// pointer + size; serialize the VALUES).
inline uint64_t DigestPlayhead(const SeqPlayhead& w) {
    std::vector<fx> buf; buf.reserve(w.bus.size() + 1u);
    buf.push_back(w.time);
    for (fx v : w.bus) buf.push_back(v);
    return DigestTrack(buf);                 // == net::DigestBytes(buf.data(), buf.size()*sizeof(fx))
}
```

### 3. Fixture (FIXED forever)
- `Sequence MakeShowcasePlayheadSeq()` — reuse `MakeShowcaseSequence()` (the S2 3-channel sequence) so the
  bus is non-trivial (3 channels, mixed easings). Keep FIXED.
- A fixed per-tick delta of `kOne/30` (30 Hz playback) and a fixed tick count (e.g. 90 = 3 s) for the
  goldens. The seek tick `S` (e.g. 45 = 1.5 s) and `toTick` (90) are FIXED.

## The goldens (PINNED, cross-platform) — append to tests/seq_test.cpp
The test builds the `step`/`digest` lambdas (capturing the fixed `Sequence`) and an `InputRing<fx>` of
per-tick deltas:
```cpp
const seq::Sequence S = seq::MakeShowcasePlayheadSeq();
auto step   = [&S](seq::SeqPlayhead& w, const std::vector<seq::fx>& in, uint32_t t){ seq::StepPlayhead(S,w,in,t); };
auto digest = [](const seq::SeqPlayhead& w){ return seq::DigestPlayhead(w); };
net::InputRing<seq::fx> ring; for (uint32_t t=0;t<90;++t) ring.AddInput(t, kOne/30);   // 30 Hz advance
```
```
seq-s5: lockstep final digest = 0x<...>
seq-s5: trace-of-trace digest = 0x<...>   (90 ticks)
seq-s5: rollback authority digest = 0x<...>, didRollback = <b>
PASS seq-s1..s4: ... (ALL prior assertions STILL green — every prior digest UNCHANGED)
PASS seq-s5: RunLockstep is deterministic — two runs over the same ring yield the IDENTICAL final digest
PASS seq-s5: lockstep final digest == pinned uint64 (the timeline is bit-identical cross-platform)
PASS seq-s5: DigestTrace length == ticks and its digest == pinned uint64 (per-tick checksum stream stable)
PASS seq-s5: SCRUB=SEEK — CatchUp(snapshot@S, toTick) == the from-0 world at toTick (BIT-IDENTICAL) [THE MOAT]
PASS seq-s5: scrub reads the same frame — DigestPlayhead(seek to S) == the from-0 trace digest at tick S
PASS seq-s5: rollback — a mispredicted remote delta rolls back to the BIT-IDENTICAL authority + didRollback
PASS seq-s5: snapshot completeness — snapshot/advance-K/restore/re-advance-K == straight advance (complete)
PASS seq-s5: an incomplete restore (zeroing time) DIVERGES (no state escapes the value-copy snapshot)
```
Assertions:
1. **PRIOR INVARIANT** — re-assert S1 `0xd314f17ebe3d480b`, S2 (sine `0x8f13b44545cc3c97`, quad-in
   `0x7ebbb0956a7f50a2`, quad-out `0x5289c36d8551004a`, seq-sweep `0xee44096d40ab3946`), S3 event-sweep
   `0x1035f49824b6ac7a`, S4 transform-sweep `0x59e3f94ce2da437d` — ALL UNCHANGED.
2. **DETERMINISTIC** — two `RunLockstep` calls over the same ring → identical final digest.
3. **PINNED FINAL** — `RunLockstep(SeqPlayhead{}, ring, 90, step, digest)` == a pinned `uint64`.
4. **TRACE PINNED** — `DigestTrace(...)` has length 90 AND `net::DigestBytes` of the trace
   (`trace.data(), trace.size()*sizeof(uint64_t)`) == a pinned `uint64` (the per-tick checksum stream is
   byte-stable; note `uint64_t` byte order is the same LE on both targets).
5. **SCRUB = SEEK (THE MOAT)** — run from-0 to `toTick=90`; ALSO capture the world AS OF tick `S=45` (run a
   separate `RunLockstep`-style loop to `S`, or step a `Session` to `S` and read `s.world`) into a
   `net::JoinSnapshot<SeqPlayhead>{S, worldAtS}`. Then `CatchUp(snap, 90, ring, step)` (the ring IS the
   tail — it carries every tick's delta) and assert `DigestPlayhead(caughtUp)` ==
   `DigestPlayhead(from-0 world at 90)` (a `RunLockstep` that returns the world, or re-derive). BIT-IDENTICAL
   — seek-then-play == play-from-0. This is the scrub-determinism headline.
6. **SCRUB READS THE FRAME** — `DigestPlayhead(worldAtS)` == `DigestTrace(...)[S-1]` (the from-0 trace's
   digest after tick S; index S-1 since the trace records AFTER each tick). Seeking to S shows the exact
   frame the from-0 playback shows at S.
7. **ROLLBACK == AUTHORITY** — build a `RollbackSession<SeqPlayhead, fx>`; drive it via `RunWithTransport`
   with a `ScriptedTransport<fx>` that DELAYS a remote delta past its origin tick AND makes the delayed
   remote differ from the prediction (lastConfirmed) so a real mispredict fires. Authority =
   `RunLockstep(SeqPlayhead{}, authRing, T, step, digest)` where `authRing.At(t) = { local[t], remote[t] }`
   (two inputs per tick, summed by the step). Assert the rollback session's final
   `DigestPlayhead(s.world)` == the authority digest == a pinned `uint64` AND `s.didRollback == true`. (Use
   small fixed local/remote delta streams; the remote stream is what's predicted/corrected.)
8. **COMPLETENESS** — from some advanced playhead, value-copy a snapshot; advance K ticks (diverge); restore
   the snapshot; re-advance the SAME K ticks → byte-identical (`DigestPlayhead`) to a straight K-advance from
   the snapshot. Proves the value-copy is the whole state.
9. **INCOMPLETE DIVERGES** — restore a snapshot with `time` zeroed (drop the real state), re-advance K → a
   DIFFERENT digest (no stateful field escapes the snapshot — the verdict.h discipline).

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/seq/seq.h` + `engine/sim/fpx.h` + `engine/math/math.h` + `engine/flow/flow.h` +
`engine/net/session.h` + `tests/seq_test.cpp` (+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`)
to the Mac and runs `clang++ -std=c++20 -I engine -I tests tests/seq_test.cpp -o /tmp/seq && /tmp/seq`,
confirming ALL assertions PASS with IDENTICAL pinned digests (esp. lockstep final + trace + scrub + rollback).
Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/seq/seq.h` (add `SeqPlayhead`/`StepPlayhead`/`DigestPlayhead`/
  `MakeShowcasePlayheadSeq` below S4). Do NOT modify any S1–S4 type or function semantics. ALL prior digests
  stay pinned.
- NO new include (`net/session.h` already present). The 6 includes stay
  `<cstddef>/<cstdint>/<vector>` + `sim/fpx.h` + `net/session.h` + `flow/flow.h`. STILL NO `<cmath>`,
  `<algorithm>`, float, `<random>`, clock, `std::hash`. Do NOT modify `net/session.h` (read-only reuse of
  `RunLockstep`/`DigestTrace`/`RollbackSession`/`RunWithTransport`/`ScriptedTransport`/`JoinSnapshot`/
  `CatchUp`).
- Pure-CPU INTEGER. The step is integer add + `SampleSequence`; the digest is hand-LE `fx` (NEVER
  `DigestBytes` a `SeqPlayhead` — the `bus` vector has a heap pointer).
- `tests/seq_test.cpp` stays self-contained; APPEND the S5 assertions + the playhead fixture + the
  lambdas/rings. Keep ALL S1–S4 assertions green.
- Branch `fix-seq-s5`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target seq_test'`
  then run `seq_test`, confirm ALL assertions (S1–S5) PASS, exit 0. ALSO compile standalone with the local
  clang `C:\Program Files\LLVM\bin\clang++.exe` and confirm IDENTICAL digests. First run: pin the S5
  digests (lockstep final, trace-of-trace, rollback authority) from the printed values, rebuild, green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `seq_test` builds + PASSES on Windows with
  every assertion green (esp. prior invariants + lockstep + trace + the SCRUB=SEEK moat + rollback==authority
  + completeness/incomplete-diverges), and the local clang standalone passes with identical digests. Report:
  commit hash, full test output (printed digests + PASS lines), the pinned S5 `uint64`s, confirmation ALL
  prior digests are unchanged, confirmation the header is self-contained (list the 6 `#include`s, NO new
  one) and `<cmath>`-free, how you built the rollback mispredict (the local/remote streams + the transport
  schedule), and the local-clang result. Flag any deviation. Commit message via a temp file + `git commit -F`
  (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only + self-containment, runs the Mac/clang standalone for the identical
  digests, ff-merges to master + pushes + deletes the branch + advances to S6 — the OPTIONAL float render
  capstone — then writes the ARCHITECTURE.md seq section + comments issue #25, leaving it open for the GUI
  editor.)
