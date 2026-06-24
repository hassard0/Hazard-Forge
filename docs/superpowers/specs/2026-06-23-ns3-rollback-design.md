# Slice NS3 — Prediction + snapshot ring + rollback (Issue #27, flagship #24 NETCODE, 3rd slice — THE CRUX)

The GGPO core: when a remote peer's input for the current tick hasn't arrived, **predict** it (reuse the last
confirmed remote input) and advance speculatively; when the real input arrives for a past tick, if it differs from
the prediction, **roll back** — restore the world from a **snapshot ring** to that tick and re-simulate forward
with the corrected inputs. The make-or-break proof: the predicted+rolled-back run reaches the **bit-identical
digest** of a no-misprediction (authority) run, AND a real misprediction diverged before being corrected. Pure-CPU
integer, strict bit-exact digest golden (no image, no Mac render-bake). Builds on NS1/NS2. (This generalizes
`fpx.h:583 RunRollback` to a rolling snapshot window + arbitrary misprediction ticks.)

## The model (a peer with a remote-input LATENCY)
Two input streams over `T` ticks: `local[t]` (known immediately) and `remote[t]` (the other peer's input, known
only `LAG` ticks late — it "arrives" at tick `t+LAG`). The peer steps each tick using `local[t]` + the remote
input it currently believes: the **confirmed** `remote[t]` if it has arrived, else a **prediction** (the last
confirmed remote input). Snapshots let it undo a wrong prediction when the truth arrives.

## The addition — `engine/net/session.h` (APPEND-ONLY after the NS2 block; keep self-contained)
Add to `hf::net` (do NOT modify NS1/NS2; keep session.h STANDALONE — only `<cstddef>`/`<cstdint>`/`<vector>`). The
toy worlds are value-copyable, so the snapshot is a `World` copy (the session contract: `World` is copy-restorable;
real sims would plug in `SnapshotWorld`/`RestoreWorld`, noted but not needed here). Provide a rollback session that
captures the predict/snapshot/rollback machinery generically:
- `template <class World, class Input> struct RollbackSession {`
    `World world;`                         // current speculative world
    `uint32_t tick = 0;`                   // next tick to simulate
    `uint32_t confirmedThrough = 0;`       // all remote inputs < this tick are confirmed
    `std::vector<World> snaps;`            // snaps[t] = world at the START of tick t (the snapshot ring/log)
    `std::vector<Input> appliedRemote;`    // the remote input actually APPLIED at each past tick (predicted or real)
    `std::vector<Input> confirmedRemote;`  // the real remote inputs received so far (by tick)
    `std::vector<bool>  haveRemote;`       // whether confirmedRemote[t] has arrived
    `Input lastConfirmed{};`               // the prediction source (last confirmed remote input)
    `bool  didRollback = false;`           // set true if any rollback fired (proof flag)
  `};`
- `template <class World, class Input, class StepFn> void StepPredicted(RollbackSession<World,Input>& s, const Input& localThisTick, StepFn step)`:
  snapshot `snaps[s.tick] = s.world`; choose `Input r = s.haveRemote[s.tick] ? s.confirmedRemote[s.tick] :
  s.lastConfirmed` (predict by reusing the last confirmed remote); record `s.appliedRemote[s.tick] = r`; call
  `step(s.world, { localThisTick, r }, s.tick)` (the deterministic per-tick step over BOTH inputs in a fixed
  order — local then remote); `++s.tick`. (Grow the vectors as needed; `snaps`/`appliedRemote` indexed by tick.)
- `template <class World, class Input, class StepFn> void ConfirmRemote(RollbackSession<World,Input>& s, uint32_t at, const Input& real, StepFn step)`:
  record `s.confirmedRemote[at] = real; s.haveRemote[at] = true;` and update `s.lastConfirmed`/`confirmedThrough`
  as the confirmed prefix extends. If `at < s.tick` AND `s.appliedRemote[at] != real` (a MISPREDICTION of an
  already-simulated tick): **rollback** — `s.world = s.snaps[at]; s.didRollback = true;` then **re-simulate**
  `at .. s.tick-1`: for each tick `u` in that range, re-snapshot `snaps[u]=world`, pick the remote input (confirmed
  if `haveRemote[u]` else `lastConfirmed`), re-apply `appliedRemote[u]`, `step(world, {local[u], thatRemote}, u)`
  — so you need the local inputs for the replayed range (store them in the session or pass a `localAt(u)` accessor;
  simplest: also keep `std::vector<Input> appliedLocal;` filled by `StepPredicted`). After the replay, `s.tick` is
  unchanged (we re-reached the same tick) and the world reflects the corrected input.
  Comparison `appliedRemote[at] != real` requires `Input` to be equality-comparable (the toy `Input`s are).
  All pure-CPU integer. NO `<cmath>`, NO float, NO clock/RNG.
(Keep the structure clear and obviously deterministic; the exact field set may vary, but it MUST: predict by
reusing the last confirmed remote, snapshot every tick, and on a late-arriving differing input restore + replay.)

## CPU test — extend `tests/session_test.cpp` (add an NS3 section, keep NS1/NS2)
Use ToyA (or a toy world whose step folds BOTH inputs order-sensitively, so a wrong remote prediction actually
changes the state). Build fixed `local[0..T)` and `remote[0..T)` streams.
Assertions: (1) **rollback correctness (make-or-break)** — AUTHORITY: a no-latency lockstep run where every tick
applies the TRUE `{local[t], remote[t]}` → digest `D_auth`. ROLLBACK: drive a `RollbackSession` with `LAG > 0` —
each tick `t` call `StepPredicted(s, local[t], step)`, and "deliver" `remote[t]` via `ConfirmRemote(s, t,
remote[t], step)` at tick `t+LAG`; after `T` ticks, drain the remaining in-flight confirmations (deliver the last
`LAG` remotes, triggering final rollbacks). Assert the rollback session's final digest == `D_auth`
BYTE-IDENTICAL. (2) **a real misprediction happened** — assert `s.didRollback == true` (the remote stream must
actually vary so predictions are sometimes wrong — choose `remote[t]` that changes); i.e. the rollback was not a
no-op. (3) **no-latency == authority trivially** — with `LAG = 0` every input is confirmed immediately, no
rollback (`didRollback == false`), digest == `D_auth`. (4) **pinned digest** — `D_auth` == a hard-pinned
`uint64_t`. Print the digests + `session_test: ALL CHECKS PASSED`. Report lines:
```
ns3-rollback: prediction + snapshot ring + rollback on misprediction
ns3-rollback: rollback correctness — predicted+rolled-back == authority BYTE-IDENTICAL {hash:0x<H>}
ns3-rollback: a real misprediction diverged then was corrected {didRollback:true}
ns3-rollback: LAG=0 no-rollback == authority {ok:true}
ns3-rollback: pinned authority digest {hash:0x<H>}
```

## Proof (STRICT bit-exact digest — cross-platform bar)
The golden is the pinned `DigestBytes` in `session_test`; the cross-platform proof = the controller compiles+runs
`session_test` on the Mac (clang) → IDENTICAL hashes. Make-or-break: rollback-correctness (predicted+rolled-back
== authority byte-identical) WITH `didRollback == true` (a genuine mispredict was corrected).

## Constraints (HARD)
- APPEND-ONLY to engine/net/session.h (do NOT modify NS1/NS2) + extend tests/session_test.cpp. Keep session.h
  self-contained (only `<cstddef>`/`<cstdint>`/`<vector>` — the Mac clang single-file compile must keep working).
  Do NOT modify any other file. NO RHI/shader/GPU. Pure-CPU integer. NO `<cmath>`, NO float, NO clock/RNG.
- Branch `fix-issue-27-ns3`, commit there, do NOT merge. NO golden file (the pinned hash lives in the test).
- Build Windows + run session_test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86)
  parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target session_test'`
  then run session_test.exe (under build\windows-msvc-release).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, session_test prints ALL CHECKS PASSED, the
  rollback-correctness equality holds (rolled-back == authority byte-identical) WITH `didRollback == true`, the
  LAG=0 no-rollback path matches, the pinned digest matches, AND NS1/NS2 checks still pass. Commit message via a
  temp file + `git commit -F` (use the Bash tool heredoc). Commit to the branch and STOP. Report: commit hash, the
  printed digests, confirmation session_test passes (NS1+NS2+NS3), the rollback fired (didRollback), and
  confirmation session.h stays self-contained. (The CONTROLLER runs session_test on the Mac to confirm identical
  hashes, then merges.)
