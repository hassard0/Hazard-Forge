# Slice NS4 — Transport-agnostic injected interface + adversarial schedule (Issue #27, flagship #24 NETCODE, 4th slice)

The session becomes transport-agnostic: remote inputs arrive through an **injected** packet interface, so a TEST
can script a **deterministic adversarial schedule** — delay, reorder, and transient loss-with-resend — with NO
real sockets. Missing inputs trigger the NS3 prediction/rollback machinery. The make-or-break proof: under a
scripted loss+reorder+delay schedule, the session STILL converges to the SAME pinned digest as the clean run —
rollback correctness under adversity. Pure-CPU integer, strict bit-exact digest golden (no image, no Mac
render-bake). Builds on NS3 (StepPredicted/ConfirmRemote reused unchanged).

## The model (every input EVENTUALLY arrives, in any order, possibly late)
Real rollback netcode never truly loses an input — inputs are resent redundantly, so each `remote[t]` EVENTUALLY
arrives (possibly out of order, delayed, after a dropped first copy that's resent). The transport models the
DELIVERY SCHEDULE; the invariant it must preserve is that every `forTick` is delivered at least once within a
bounded window. The session `ConfirmRemote`s each input as it arrives (NS3 already handles arbitrary `at` order +
duplicates safely — a re-delivery of an already-confirmed input is a no-op).

## The addition — `engine/net/session.h` (APPEND-ONLY after the NS3 block; keep self-contained)
Add to `hf::net` (do NOT modify NS1/NS2/NS3; keep session.h STANDALONE — only `<cstddef>`/`<cstdint>`/`<vector>`):
- `template <class Input> struct InflightPacket { uint32_t deliverTick = 0; uint32_t forTick = 0; Input input{}; };`
  (a remote input `forTick` that the transport delivers at `deliverTick` — `deliverTick >= forTick` models latency,
  non-monotonic `deliverTick` vs `forTick` models reorder, two packets with the same `forTick` model resend).
- `template <class Input> struct ScriptedTransport { std::vector<InflightPacket<Input>> sched; std::size_t cursor = 0; };`
  + `template <class Input> void Schedule(ScriptedTransport<Input>& tx, uint32_t deliverTick, uint32_t forTick, const Input& in)`
  (append to `sched`) and `template <class Input> std::vector<InflightPacket<Input>> Deliver(ScriptedTransport<Input>& tx, uint32_t currentTick)`
  returning all `sched` entries whose `deliverTick == currentTick` **in scheduled (insertion) order** (deterministic
  delivery order). (A simple linear scan returning the matches, or a cursor walk if `sched` is delivered in tick
  order — keep it deterministic; the test may schedule with non-monotonic `deliverTick`, so a per-tick scan is the
  safe form.)
- A driver tying the transport to the NS3 rollback session:
  `template <class World, class Input, class StepFn> void RunWithTransport(RollbackSession<World,Input>& s, const std::vector<Input>& local, ScriptedTransport<Input>& tx, uint32_t totalTicks, StepFn step)`:
  for `t` in `[0, totalTicks)`: `StepPredicted(s, local[t], step)`; then for each `pkt` in `Deliver(tx, t)` call
  `ConfirmRemote(s, pkt.forTick, pkt.input, step)`. (`totalTicks` must exceed the last `deliverTick` so every
  packet is drained / every input confirmed before the end.)
  All pure-CPU integer. NO `<cmath>`, NO float, NO clock/RNG.

## CPU test — extend `tests/session_test.cpp` (add an NS4 section, keep NS1/NS2/NS3)
Reuse the NS3 order-sensitive toy world + fixed `local[0..T)` / `remote[0..T)` (remote VARYING).
Assertions: (1) **convergence under adversity (make-or-break)** — AUTHORITY digest `D_auth` (the NS3 no-latency
lockstep over the true `{local[t], remote[t]}`). ADVERSARIAL: build a `ScriptedTransport` that delivers every
`remote[t]` but with a DELIBERATELY adversarial schedule — **delay** (`deliverTick = t + jitter`), **reorder**
(some later ticks delivered before earlier ones — non-monotonic `deliverTick`), and **loss-with-resend** (schedule
a first copy that is "dropped" by NOT actually being the one that lands first AND a redundant later copy — i.e.
ensure each `forTick` has a delivery, but vary which/when). Run `RunWithTransport` for enough `totalTicks` to drain
all packets, then assert the session's final digest == `D_auth` BYTE-IDENTICAL, with `s.didRollback == true` (the
adversity forced real rollbacks). (2) **schedule determinism** — running the SAME adversarial schedule twice
yields byte-identical digests (the transport + session are fully deterministic). (3) **duplicate delivery is a
no-op** — delivering a `forTick` twice (a resend) does not change the converged digest. (4) **pinned digest** —
`D_auth` == the NS3 pinned value (or a freshly pinned `uint64_t`). Print the digests + `session_test: ALL CHECKS
PASSED`. Report lines:
```
ns4-transport: injected transport + adversarial schedule (delay/reorder/resend)
ns4-transport: converges under adversity — final == authority BYTE-IDENTICAL {hash:0x<H>}
ns4-transport: adversity forced rollbacks {didRollback:true}
ns4-transport: schedule determinism — same schedule twice identical {ok:true}
ns4-transport: pinned authority digest {hash:0x<H>}
```

## Proof (STRICT bit-exact digest — cross-platform bar)
The golden is the pinned `DigestBytes` in `session_test`; the cross-platform proof = the controller compiles+runs
`session_test` on the Mac (clang) → IDENTICAL hashes. Make-or-break: convergence to the authority digest under the
adversarial schedule WITH `didRollback == true`.

## Constraints (HARD)
- APPEND-ONLY to engine/net/session.h (do NOT modify NS1/NS2/NS3) + extend tests/session_test.cpp. Keep session.h
  self-contained (only `<cstddef>`/`<cstdint>`/`<vector>` — the Mac clang single-file compile must keep working).
  Do NOT modify any other file. NO RHI/shader/GPU. Pure-CPU integer. NO `<cmath>`, NO float, NO clock/RNG (the
  "adversarial schedule" is a fixed scripted permutation, NOT random).
- Branch `fix-issue-27-ns4`, commit there, do NOT merge. NO golden file (the pinned hash lives in the test).
- Build Windows + run session_test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86)
  parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target session_test'`
  then run session_test.exe (under build\windows-msvc-release).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, session_test prints ALL CHECKS PASSED, the
  adversarial session converges to the authority digest BYTE-IDENTICAL with `didRollback == true`, the schedule
  determinism + duplicate-no-op hold, the pinned digest matches, AND NS1/NS2/NS3 checks still pass. Commit message
  via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the branch and STOP. Report: commit
  hash, the printed digests, didRollback status, confirmation session_test passes (NS1–NS4), and confirmation
  session.h stays self-contained. (The CONTROLLER runs session_test on the Mac to confirm identical hashes, then
  merges.)
