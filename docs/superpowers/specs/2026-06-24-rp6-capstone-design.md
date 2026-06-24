# Slice RP6 — Capstone: end-to-end pipeline + corruption detection (Flagship #28 REPLAY/DEMO, 6th/FINAL)

The capstone that COMPLETES flagship #28. RP1-RP5 built the pieces: record → encode → decode → replay →
keyframe → seek → scrub. RP6 (1) runs the WHOLE pipeline end-to-end on a fixed session and reproduces the
pinned final digest, and (2) adds the headline that makes determinism a SECURITY/INTEGRITY property, not
just a reproducibility one: **a demo's per-tick digest stream is a built-in tamper detector.** Flip a
single byte in a demo's input region and the replay's re-derived per-tick digest DIVERGES from the
recorded trace at the EXACT tick the corruption takes effect — located, not silently tolerated. This is
the `net::DesyncDetector` (NS5) applied to recorded-trace-vs-replayed-trace.

Pure-CPU INTEGER, header-only `engine/replay/replay.h`, append-only (ADD `VerifyReplay` + the corruption
helper below RP5; do NOT modify prior code, no format change). NO render-bake. Goldens are pinned
`uint64_t`/`uint32_t`, identical Windows/MSVC + Mac/clang.

## The corruption-detection design (NO format change)
The per-tick digest trace is NOT serialized into the demo file (RP3 deliberately left it in-memory). So
RP6 detects corruption by comparing two DECODED demos' replayed traces — a clean reference vs a corrupted
copy — which needs NO file-format change (no re-pin):
1. The **clean** reference trace = `Replay(DecodeDemo(cleanBytes)).trace` (RP2 proved this == the recorded
   trace == a fresh `net::DigestTrace`). This is the authoritative per-tick checksum stream.
2. Corrupt a **COPY** of the bytes: `corruptBytes = cleanBytes;` then flip one bit/byte in the INPUT
   region (offset `36 + worldByteLen` onward — past the header + initial-world snapshot), specifically a
   byte that is part of an input VALUE (not a length/count field, so the ring structure stays parseable
   and exactly one input changes). Decode + replay the corrupted copy → `corruptTrace`.
3. The corrupted input changes `StepA` from its tick onward, so `corruptTrace` diverges from `cleanTrace`
   starting at the corrupted input's tick. Feed `net::DesyncDetector`: `RecordLocal(d, t, cleanTrace[t])`
   for every tick (the authoritative record), then `IngestRemote(d, ChecksumPacket{t, corruptTrace[t]})`
   for every tick (the replayed-from-corrupted checksums). The detector latches `d.desyncTick` = the FIRST
   diverging tick, with `d.localDigest` (clean) and `d.remoteDigest` (corrupt).

## Append to engine/replay/replay.h (below RP5, in hf::replay)

**VerifyReplay** — the end-to-end integrity check a player/loader runs: decode a demo, replay it, and
confirm its re-derived per-tick trace matches an expected reference trace; return the first divergence (if
any) using the same latch semantics as `net::DesyncDetector`:
```cpp
template <class World, class Input>
struct VerifyResult {
    bool     ok = true;            // true iff every replayed per-tick digest matched the expected trace
    uint32_t divergeTick = 0;      // the FIRST tick where they differ (valid iff !ok)
    uint64_t expectedDigest = 0;   // the reference digest at divergeTick
    uint64_t actualDigest = 0;     // the replayed digest at divergeTick
};

// Replay `demo` and compare its per-tick digest trace against `expectedTrace` (the authoritative recorded
// trace). Latches the FIRST mismatch (NS5 DesyncDetector semantics). `expectedTrace.size()` should equal
// demo.header.tickCount; compare min(len) ticks defensively.
template <class World, class Input, class StepFn, class DigestFn>
VerifyResult<World,Input> VerifyReplay(const Demo<World,Input>& demo,
                                       const std::vector<uint64_t>& expectedTrace,
                                       StepFn step, DigestFn digest);
```
Implementation: `Replay(demo, step, digest)` to get the actual trace, then walk ticks comparing to
`expectedTrace`; on the first mismatch set `ok=false`, `divergeTick=t`, `expectedDigest=expectedTrace[t]`,
`actualDigest=actual[t]` and stop. (Equivalently drive `net::DesyncDetector` via `RecordLocal` +
`IngestRemote` and read its latch — either is fine; reusing DesyncDetector ties it explicitly to NS5.
Prefer driving DesyncDetector so the corruption story IS the netcode desync machinery.)

(No new corruption helper is strictly required in the header — the test flips a byte in a local copy. If a
small `inline` helper `CorruptByteAt(std::vector<uint8_t>& bytes, std::size_t off)` reads cleaner, add it
append-only. Keep it trivial.)

## The goldens (PINNED, cross-platform) — append to tests/replay_test.cpp
### Part A — end-to-end pipeline (reproduces the pinned final via EVERY capability)
Record the fixed ToyA session with keyframeInterval=4 → encode (`demoKf4`) → decode → and exercise the
full stack, all landing on the pinned final `0x6227bc7b4046d08a`:
```
PASS rp6-e2e: record->encode->decode round-trips (header + initial + ring + keyframes intact)
PASS rp6-e2e: Replay(loaded) final == Seek(loaded,16) == ScrubTo(player,16) == net::RunLockstep(16) == 0x6227bc7b4046d08a
PASS rp6-e2e: VerifyReplay(loaded, cleanTrace) ok==true (no divergence on a clean demo)
```
(cleanTrace = `Replay(decode(demoKf4)).trace`; assert VerifyReplay against it is `ok==true`.)

### Part B — corruption detection (the headline)
Corrupt a copy of the demo bytes at a byte inside a chosen input value, decode+replay, and localize:
```
rp6-corrupt: corrupted byte at offset <O> (input tick <T>); clean hash=0x.. corrupt hash=0x..
rp6-corrupt: divergence first at tick <T>  clean=0x..  corrupt=0x..
PASS rp6-corrupt: clean file hash != corrupt file hash (the tamper changed the bytes)
PASS rp6-corrupt: VerifyReplay(corrupt, cleanTrace) ok==false, divergeTick == <pinned T> (located, not silent)
PASS rp6-corrupt: net::DesyncDetector latches the SAME tick <T> (clean vs corrupt per-tick digest exchange)
PASS rp6-corrupt: corrupt replay final digest != 0x6227bc7b4046d08a (the world really diverged)
PASS rp6-corrupt: a clean (uncorrupted) copy still VerifyReplay ok==true AND replays to 0x6227bc7b4046d08a
```
Assertions:
1. **HASH CONTRAST** — `DigestBytes(cleanBytes) != DigestBytes(corruptBytes)` (one flipped byte changes
   the file hash).
2. **LOCALIZED DIVERGENCE (make-or-break)** — `VerifyReplay(decode(corruptBytes), cleanTrace).ok ==
   false` and `.divergeTick == T_pinned`, where `T_pinned` is the tick of the corrupted input (the
   implementer picks the corruption offset deterministically, runs once, reads the actual divergence tick,
   and PINS it; choose an input at a MID-stream tick — e.g. one of tick 7's inputs — so the demo shows
   mid-timeline localization, not just tick 0). Pin the clean + corrupt digests at that tick too.
3. **DESYNC DETECTOR AGREEMENT** — driving `net::DesyncDetector` (RecordLocal over cleanTrace +
   IngestRemote over corruptTrace) latches `d.desynced==true` and `d.desyncTick == T_pinned` with
   `d.localDigest`==clean, `d.remoteDigest`==corrupt (the NS5 machinery localizes the tamper identically).
4. **WORLD REALLY DIVERGED** — `Replay(decode(corruptBytes)).finalDigest != 0x6227bc7b4046d08a`.
5. **CLEAN STILL GOOD** — re-decoding the untouched `cleanBytes` still `VerifyReplay ok==true` and replays
   to `0x6227bc7b4046d08a` (the corruption was isolated to the copy; the detector has no false positives).

Keep ALL RP1-RP5 assertions green (append-only — pinned hashes unchanged: no-keyframe
`0x92e4d491013137c4`, Kf4 `0xf2ccf29305652a39`, Kf8 `0xe7fb940bd0c3cc41`, ToyA final
`0x6227bc7b4046d08a`).

### Part C (OPTIONAL, only if clean to do) — genericity over a 2nd world
If trivial, run the full record→encode→decode→Replay pipeline on a SECOND self-contained toy world
(`ToyB`/`InB` from `session_test.cpp` — a cell-array world; copy it in, no fpx, keeps the test
standalone-clean) and assert `Replay == net::RunLockstep` for it too — proving the replay system is
generic over World/Input. Do NOT pull in fpx (it would break the standalone clang proof). Skip this if it
risks the build; the ToyA proof is the requirement.

## Cross-platform proof (cheap loop — NO render-bake)
Controller `scp`s `engine/replay/replay.h` + `engine/net/session.h` + `tests/replay_test.cpp`
(+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs
`clang++ -std=c++20 -I engine -I tests tests/replay_test.cpp -o /tmp/rp6 && /tmp/rp6`, confirming ALL
assertions PASS with the IDENTICAL pinned divergence tick + digests. NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/replay/replay.h` (add `VerifyResult`/`VerifyReplay` [+ optional `CorruptByteAt`]
  below RP5; do NOT modify RP1-RP5 — no format change). Header stays SELF-CONTAINED (only
  `<cstdint>/<cstddef>/<vector>` + `net/session.h`; NO new includes). Reuse `net::DesyncDetector` /
  `net::ChecksumPacket` / `net::RecordLocal` / `net::IngestRemote` read-only. Do NOT modify
  `net/session.h` or any other existing header.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<functional>`/`<fstream>`. (The demo "file" is the byte
  vector; actual disk persistence is a trivial out-of-scope caller concern.)
- `tests/replay_test.cpp` stays self-contained; APPEND the RP6 assertions. Keep all prior green.
- Branch `fix-replay-rp6`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target replay_test'`
  then run `replay_test` and confirm ALL assertions (RP1-RP6) PASS, exit 0. First run: read the actual
  divergence tick + clean/corrupt digests from output, PIN them, rebuild green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `replay_test` builds + PASSES on Windows
  with every assertion green (esp. the localized-divergence tick + DesyncDetector agreement), and (if a
  local clang exists) the standalone clang compile passes. Report: commit hash, full test output (all PASS
  lines + the corruption offset/tick/digests), the pinned divergence tick, confirmation the four prior
  pinned hashes are unchanged, confirmation the header is still self-contained (list #includes), and the
  local-clang result (or none → controller runs Mac). Commit message via temp file + `git commit -F`
  (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for identical results, ff-merges to
  master + pushes + deletes the branch, then writes the ARCHITECTURE.md replay section + closes out
  flagship #28.)
