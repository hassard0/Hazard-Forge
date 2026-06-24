# Slice NS2 — Input-delay buffer (Issue #27, flagship #24 NETCODE, 2nd slice)

The production knob the per-sim lockstep primitive lacks: a configurable **input delay** — local input is scheduled
to apply `D` ticks in the FUTURE instead of immediately. Why it matters for netcode: by delaying when local input
takes effect, the remote peer's input (which travels over the network) usually arrives BEFORE its apply-tick, so
the common case needs no rollback. NS2 proves delay is a pure *scheduling shift* — it changes WHEN inputs apply,
never the game semantics — by showing a delay-`D` session whose inputs are submitted `D` ticks early is byte-
identical to a delay-0 session. Pure-CPU integer, strict bit-exact digest golden (no image, no Mac render-bake).
Builds on NS1.

## The addition — `engine/net/session.h` (APPEND-ONLY after the NS1 block; keep self-contained)
Add to `hf::net` (do NOT modify the NS1 `DigestBytes`/`InputRing`/`Session`/`Advance`/`RunLockstep`; keep session.h
STANDALONE — only `<cstddef>`/`<cstdint>`/`<vector>`):
- Add a delay field to the session WITHOUT changing the NS1 struct's existing fields/offsets — append it:
  `Session` gains `uint32_t delay = 0;` (default 0 → NS1 behaviour unchanged; place it AFTER the existing
  `tick` member so NS1's tests are unaffected).
- `template <class World, class Input> void SubmitLocalInput(Session<World,Input>& s, const Input& in)`:
  schedules `in` to apply `delay` ticks ahead — `s.ring.AddInput(s.tick + s.delay, in)`. (At `delay == 0` this is
  "apply on the current tick"; at `delay == D` the input lands at `s.tick + D`.) This is the **only** behavioural
  addition — `Advance` is unchanged (it still pulls `ring.At(tick)`); the delay lives entirely in the
  submit→schedule mapping, which is exactly why it's a pure scheduling shift.
- (Optional convenience) `template <...> void SubmitInputAt(Session<...>& s, uint32_t applyTick, const Input& in)`
  → `s.ring.AddInput(applyTick, in)` (explicit apply-tick, for the reference path / tests).
  All pure-CPU integer. NO `<cmath>`, NO float, NO clock/RNG.

## CPU test — extend `tests/session_test.cpp` (add an NS2 section, keep the NS1 checks)
Using ToyA (or ToyB) from NS1:
Assertions: (1) **delay is a pure scheduling shift (make-or-break)** — REFERENCE: a delay-0 session where a fixed
set of `(applyTick, input)` events are scheduled via `SubmitInputAt`, run T ticks → digest `D0`. DELAYED: a
delay-`DELTA` session where the SAME events are submitted via `SubmitLocalInput` at the tick `applyTick - DELTA`
(i.e. advance to `applyTick-DELTA`, submit, the input lands at `applyTick`), run the same total T ticks → assert
its digest == `D0` BYTE-IDENTICAL (delay shifts submit-time, not the game). (2) **delay actually delays** — in a
delay-`DELTA` session, submit an input at tick `t0`; assert the world digest at every tick in `[t0, t0+DELTA)` is
UNCHANGED from a no-input run (the input has not applied yet), and at `t0+DELTA` it diverges (the input applied) —
proving the delay is real. (3) **pinned digest** — `D0` == a hard-pinned `uint64_t`. (4) **delay=0 == NS1** — a
delay-0 `SubmitLocalInput` at tick `t` lands on tick `t` (equivalent to the NS1 direct `AddInput(t, …)`). Print
the digests + `session_test: ALL CHECKS PASSED`. Report lines:
```
ns2-delay: input-delay buffer (local input applies `delay` ticks later)
ns2-delay: delay-D (submitted D early) == delay-0 BYTE-IDENTICAL {hash:0x<H>}
ns2-delay: delay actually delays (no effect before applyTick) {delta:<DELTA>, ok:true}
ns2-delay: pinned converged digest {hash:0x<H>}
```

## Proof (STRICT bit-exact digest — cross-platform bar)
The golden is the pinned `DigestBytes` in `session_test`; the cross-platform proof = the controller compiles+runs
`session_test` on the Mac (clang, standalone) → IDENTICAL hashes. Make-or-break: the delay-D == delay-0
byte-identical equality (delay is a pure scheduling shift) + the pinned digest.

## Constraints (HARD)
- APPEND-ONLY to engine/net/session.h (do NOT modify the NS1 code; append `delay` after `tick`, add
  `SubmitLocalInput`/`SubmitInputAt`) + extend tests/session_test.cpp. Keep session.h self-contained (only
  `<cstddef>`/`<cstdint>`/`<vector>` — the Mac clang single-file compile must keep working). Do NOT modify any
  other file. NO RHI/shader/GPU. Pure-CPU integer. NO `<cmath>`, NO float, NO clock/RNG.
- Branch `fix-issue-27-ns2`, commit there, do NOT merge. NO golden file (the pinned hash lives in the test).
- Build Windows + run session_test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86)
  parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target session_test'`
  then run session_test.exe (under build\windows-msvc-release).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, session_test prints ALL CHECKS PASSED, the
  delay-D == delay-0 byte-identical equality holds, the delay-actually-delays check holds, the pinned digest
  matches, AND the NS1 checks still pass. Commit message via a temp file + `git commit -F` (use the Bash tool
  heredoc). Commit to the branch and STOP. Report: commit hash, the printed digests, confirmation session_test
  passes (NS1+NS2), and confirmation session.h stays self-contained. (The CONTROLLER runs session_test on the Mac
  to confirm identical hashes, then merges.)
