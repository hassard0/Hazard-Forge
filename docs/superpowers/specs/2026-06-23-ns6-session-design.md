# Slice NS6 — Full 2-peer adversarial session + late-join (Issue #27, flagship #24, 6th/FINAL slice — capstone)

The capstone that COMPLETES flagship #24: tie NS1–NS5 into a complete TWO-PEER session — each peer predicts the
other's inputs, rolls back over an adversarial transport, and exchanges desync checksums — and add **late-join**:
a peer that joins mid-session receives a confirmed snapshot + the input tail, replays it, and reaches the
bit-identical world as the peers that ran from tick 0. Pure-CPU integer, strict bit-exact digest golden (no image,
no Mac render-bake). Builds on NS1–NS5 (mostly composition; the one genuinely new primitive is `CatchUp`).

## The addition — `engine/net/session.h` (APPEND-ONLY after the NS5 block; keep self-contained)
Add to `hf::net` (do NOT modify NS1–NS5; keep session.h STANDALONE — only `<cstddef>`/`<cstdint>`/`<vector>`):
- `template <class World> struct JoinSnapshot { uint32_t tick = 0; World world{}; };` (the confirmed-state handoff
  a late-joiner receives: the world AS OF `tick`, where all inputs `< tick` are confirmed).
- `template <class World, class Input, class StepFn> World CatchUp(const JoinSnapshot<World>& snap, uint32_t toTick, const InputRing<Input>& tail, StepFn step)`:
  restore `World w = snap.world`; for `t` in `[snap.tick, toTick)` call `step(w, tail.At(t), t)`; return `w`. The
  late-joiner restores the confirmed snapshot and replays the input tail to catch up to `toTick`. (The `tail` ring
  carries the inputs for `[snap.tick, toTick)`.)
  All pure-CPU integer. NO `<cmath>`, NO float, NO clock/RNG.

## CPU test — extend `tests/session_test.cpp` (add an NS6 section, keep NS1–NS5)
Use the order-sensitive toy world (folds local+remote) + fixed `aInputs[0..T)` (peer A's inputs) and `bInputs[0..T)`
(peer B's). The deterministic per-tick step folds BOTH (peer-A-input, peer-B-input) in a fixed order. The AUTHORITY
is the lockstep over the true `{aInputs[t], bInputs[t]}` at each tick → `D_auth[t]` (and `D_auth = D_auth[T-1]`).
Assertions:
1. **Full 2-peer adversarial session converges (make-or-break).** Run TWO `RollbackSession`s over `ScriptedTransport`s:
   - peer A: local = `aInputs`, remote = `bInputs` delivered adversarially (delay/reorder/resend) → `RunWithTransport`.
   - peer B: local = `bInputs`, remote = `aInputs` delivered adversarially (a different adversarial schedule).
   (The per-tick step must apply the inputs in the SAME canonical (A,B) order on BOTH peers regardless of which is
   "local" — so peer B's step orders (A-remote, B-local) the same as peer A's (A-local, B-remote). Be explicit
   about this in the step so both peers fold identically.) Assert peer A's final digest == peer B's final digest ==
   `D_auth` BYTE-IDENTICAL, and both `didRollback == true`.
2. **Desync-clean.** Build `DesyncDetector`s and exchange the per-tick CONFIRMED digests (NS5) between A and B;
   assert neither detects a desync (a correct session never desyncs).
3. **Late-join (the new capability).** Pick a join tick `S` (where all inputs `< S` are confirmed); form a
   `JoinSnapshot{S, authorityWorldAtS}` (the authority world after tick `S-1`) + a `tail` `InputRing` holding the
   `{aInputs[t], bInputs[t]}` for `t` in `[S, T)`; `CatchUp(snap, T, tail, step)` and assert the late-joiner's
   digest == `D_auth` BYTE-IDENTICAL (it reaches the exact world the from-tick-0 peers reached). Also assert a join
   at `S=0` (snapshot = initial world, full tail) equals the from-scratch authority (the trivial-join sanity).
4. **Pinned digest.** `D_auth` == a hard-pinned `uint64_t`. Print the digests + `session_test: ALL CHECKS PASSED`.
Report lines:
```
ns6-session: full 2-peer adversarial session + late-join
ns6-session: 2 peers converge under adversity — A==B==authority BYTE-IDENTICAL {hash:0x<H>}
ns6-session: desync-clean — neither peer detects a desync {ok:true}
ns6-session: late-join at tick S catches up to bit-identical world {S:<S>, hash:0x<H>}
ns6-session: pinned authority digest {hash:0x<H>}
```

## Proof / cross-platform
The golden is the pinned `DigestBytes` in `session_test`; the cross-platform proof = the controller compiles+runs
`session_test` on the Mac (clang) → IDENTICAL hashes. Make-or-break: the 2 peers converge to the authority under
adversity (both rolled back), desync-clean, AND the late-joiner reaches the bit-identical world. This COMPLETES
flagship #24.

## Constraints (HARD)
- APPEND-ONLY to engine/net/session.h (do NOT modify NS1–NS5) + extend tests/session_test.cpp. Keep session.h
  self-contained (only `<cstddef>`/`<cstdint>`/`<vector>` — the Mac clang single-file compile must keep working).
  Do NOT modify any other file. NO RHI/shader/GPU. Pure-CPU integer. NO `<cmath>`, NO float, NO clock/RNG (the
  adversarial schedules are FIXED scripted permutations).
- Branch `fix-issue-27-ns6`, commit there, do NOT merge. NO golden file (the pinned hash lives in the test).
- Build Windows + run session_test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86)
  parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target session_test'`
  then run session_test.exe (under build\windows-msvc-release).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, session_test prints ALL CHECKS PASSED, the 2
  peers converge to the authority byte-identical (both didRollback), desync-clean holds, the late-join reaches the
  bit-identical world, the pinned digest matches, AND NS1–NS5 checks still pass. Commit message via a temp file +
  `git commit -F` (use the Bash tool heredoc). Commit to the branch and STOP. Report: commit hash, the printed
  digests, confirmation session_test passes (NS1–NS6), the late-join tick S + its digest match, and confirmation
  session.h stays self-contained. (The CONTROLLER runs session_test on the Mac to confirm identical hashes, then
  merges + closes issue #27.)
