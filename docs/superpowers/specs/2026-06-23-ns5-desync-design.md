# Slice NS5 — Desync detector via per-tick digest exchange (Issue #27, flagship #24 NETCODE, 5th slice)

The safety net: peers exchange their per-tick **state digest**; a mismatch turns silent divergence into a hard,
LOCATED error (the exact tick + the two diverging digests). In a correct deterministic session two peers with the
same confirmed inputs MUST have identical per-tick digests, so any mismatch means a real bug (a non-deterministic
step, a missed input, a corrupted state). NS5 builds the detector and proves it (a) reports zero desync on a clean
session and (b) flags the EXACT tick a deliberately-corrupted peer diverges. Pure-CPU integer, strict bit-exact
digest golden (no image, no Mac render-bake). Builds on NS1.

## The addition — `engine/net/session.h` (APPEND-ONLY after the NS4 block; keep self-contained)
Add to `hf::net` (do NOT modify NS1–NS4; keep session.h STANDALONE — only `<cstddef>`/`<cstdint>`/`<vector>`):
- `struct ChecksumPacket { uint32_t tick = 0; uint64_t digest = 0; };` (the per-tick digest a peer broadcasts —
  what rides the transport in a real session).
- `template <class World, class Input, class StepFn, class DigestFn> std::vector<uint64_t> DigestTrace(World init, const InputRing<Input>& ring, uint32_t ticks, StepFn step, DigestFn digest)`:
  run lockstep (NS1 `Advance`) from `init` over `ring` and record `digest(world)` AFTER each tick → a per-tick
  digest trace of length `ticks` (the confirmed-state checksum stream a peer would emit).
- `struct DesyncDetector { std::vector<uint64_t> localByTick; bool desynced = false; uint32_t desyncTick = 0; uint64_t localDigest = 0; uint64_t remoteDigest = 0; };`
- `inline void RecordLocal(DesyncDetector& d, uint32_t tick, uint64_t digest)`: grow + store
  `d.localByTick[tick] = digest`.
- `inline void IngestRemote(DesyncDetector& d, const ChecksumPacket& pkt)`: if `pkt.tick < d.localByTick.size()`
  and `!d.desynced` and `d.localByTick[pkt.tick] != pkt.digest`, set `d.desynced = true; d.desyncTick = pkt.tick;
  d.localDigest = d.localByTick[pkt.tick]; d.remoteDigest = pkt.digest;` (latch the FIRST mismatch — the located
  desync). (Ingesting a tick we haven't recorded yet is a deterministic no-op; the detector latches only the
  earliest divergence.)
  All pure-CPU integer. NO `<cmath>`, NO float, NO clock/RNG.

## CPU test — extend `tests/session_test.cpp` (add an NS5 section, keep NS1–NS4)
Use a toy world + a fixed input ring.
Assertions: (1) **clean session → zero desync (make-or-break)** — two peers run the SAME inputs; build peer B's
`DesyncDetector` by `RecordLocal`-ing B's `DigestTrace`, then `IngestRemote` every one of peer A's
`ChecksumPacket{tick, A_trace[tick]}`; assert `d.desynced == false` (identical deterministic peers never desync).
(2) **corrupted peer detected at the EXACT tick** — make peer A apply ONE wrong input at a chosen tick `K` (e.g.
add an extra input to A's ring at tick K so A's trace diverges from B's at K and stays diverged); ingest A's
checksums into B's detector; assert `d.desynced == true`, `d.desyncTick == K` (the FIRST diverging tick, not
before, not after), and `d.localDigest != d.remoteDigest`. (3) **localization is exact** — assert the traces are
IDENTICAL for every tick `< K` and DIFFER at `K` (the detector flags the true first divergence). (4) **pinned
digest** — the clean trace's final digest == a hard-pinned `uint64_t`. Print the digests + `session_test: ALL
CHECKS PASSED`. Report lines:
```
ns5-desync: per-tick digest exchange desync detector
ns5-desync: clean session — zero desync {desynced:false}
ns5-desync: corrupted peer detected at exact tick {desyncTick:<K>, local:0x<HL>, remote:0x<HR>}
ns5-desync: localization exact — identical before K, differ at K {ok:true}
ns5-desync: pinned clean final digest {hash:0x<H>}
```

## Proof (STRICT bit-exact digest — cross-platform bar)
The golden is the pinned `DigestBytes` in `session_test`; the cross-platform proof = the controller compiles+runs
`session_test` on the Mac (clang) → IDENTICAL hashes (incl. the same `desyncTick` and diverging digests). Make-or-
break: clean → zero desync AND corrupted → detected at the exact tick.

## Constraints (HARD)
- APPEND-ONLY to engine/net/session.h (do NOT modify NS1–NS4) + extend tests/session_test.cpp. Keep session.h
  self-contained (only `<cstddef>`/`<cstdint>`/`<vector>` — the Mac clang single-file compile must keep working).
  Do NOT modify any other file. NO RHI/shader/GPU. Pure-CPU integer. NO `<cmath>`, NO float, NO clock/RNG.
- Branch `fix-issue-27-ns5`, commit there, do NOT merge. NO golden file (the pinned hash lives in the test).
- Build Windows + run session_test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86)
  parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target session_test'`
  then run session_test.exe (under build\windows-msvc-release).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, session_test prints ALL CHECKS PASSED, the clean
  session reports zero desync, the corrupted peer is detected at the EXACT tick K with diverging digests, the
  localization (identical before K, differ at K) holds, the pinned digest matches, AND NS1–NS4 checks still pass.
  Commit message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the branch and STOP.
  Report: commit hash, the printed digests + desyncTick, confirmation session_test passes (NS1–NS5), and
  confirmation session.h stays self-contained. (The CONTROLLER runs session_test on the Mac to confirm identical
  hashes, then merges.)
