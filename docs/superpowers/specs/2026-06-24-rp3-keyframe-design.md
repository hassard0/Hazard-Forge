# Slice RP3 — KEYFRAME snapshots (Flagship #28 REPLAY/DEMO, 3rd/6)

RP1 recorded the causal seed; RP2 replays it bit-identically from tick 0. RP3 adds **KEYFRAMES**: at
record time, every `keyframeInterval` ticks, capture a FULL world Snapshot into the demo's keyframe
table. This is the seekability substrate — RP4 restores the nearest keyframe and replays forward
(instead of re-running from tick 0). A keyframe is exactly a `net::RollbackSession::snaps` entry taken at
a COARSE interval; the interval is a HEADER FIELD (bounds file-size vs seek-cost — RP3 pins distinct
file hashes at two different intervals to prove the tradeoff is real and deterministic).

Pure-CPU INTEGER, header-only `engine/replay/replay.h`, NO render-bake. Goldens are pinned `uint64_t`s,
identical Windows/MSVC + Mac/clang.

## This slice MATURES the demo-file format (deliberate, verified re-pin)
RP1/RP2 froze a 32-byte header (magic(8) + 6 u32 fields). RP3 adds the keyframe table, which requires
**one new header field** — so the format matures and the RP1/RP2 demo-file hashes CHANGE. This is an
intentional in-flagship evolution, NOT golden churn: the implementer RE-PINS the RP1 and RP2 file-hash
constants to their new values and the controller RE-VERIFIES every assertion on Windows + Mac/clang.
**Invariant that must NOT change:** the ToyA final digest `0x6227bc7b4046d08a` (the world is unchanged;
only the file's byte layout grows). Keep version at `kDemoVersion` (nothing external consumed v1).

### New uniform layout (ALL fields LE, exact order)
```
magic(8) | version(u32) | seed(u32) | tickCount(u32) | keyframeInterval(u32) |
worldByteLen(u32) | inputByteLen(u32) | keyframeByteLen(u32) |       <-- header now 8 + 7*4 = 36 bytes
<worldBytes> | <inputBytes> | <keyframeSection (keyframeByteLen bytes)>
```
- `keyframeByteLen` is the 7th u32 field, symmetric with `worldByteLen`/`inputByteLen` (the length of
  the keyframe section that follows the inputs). **The world blob now starts at offset 36** (was 32).
- **Keyframe section** (only present when `keyframeByteLen > 0`): `keyframeCount(u32)`, then per keyframe:
  `tick(u32)`, `kfWorldByteLen(u32)`, `<kfWorldBytes>` (the driver-serialized world snapshot AS OF
  `tick`, via the SAME `serWorld` used for the initial snapshot). `keyframeByteLen` = the total size of
  this whole section (the count prefix + every entry). When `keyframeInterval == 0` → no keyframes →
  `keyframeByteLen = 0`, no section written (RP1/RP2's degenerate case, now with a 36-byte header).

## Edits to engine/replay/replay.h (deliberate format maturation — NOT append-only this slice)
This is the ONE slice that edits the RP1/RP2 format code. Do it surgically:
1. **DemoHeader** — add `uint32_t keyframeByteLen;` after `inputByteLen`.
2. **Recorder<World,Input>** — add `uint32_t keyframeInterval = 0;` and a keyframe table:
   ```cpp
   struct Keyframe { uint32_t tick; std::vector<uint8_t> worldBytes; };
   std::vector<Keyframe> keyframes;   // full snapshots at ticks 0, K, 2K, ... (driver-serialized)
   ```
   (Store the serialized bytes directly so EncodeDemo just concatenates — the Recorder needs `serWorld`
   at record time. Pass `serWorld` into `RecordSession` so it can snapshot; see step 3.)
3. **RecordSession** — add a `keyframeInterval` parameter AND a `serWorld` callable so it can capture
   keyframes while stepping. New signature:
   ```cpp
   template <class World, class Input, class StepFn, class DigestFn, class SerWorldFn>
   Recorder<World,Input> RecordSession(uint32_t seed, World initial, const net::InputRing<Input>& ring,
                                       uint32_t ticks, uint32_t keyframeInterval,
                                       StepFn step, DigestFn digest, SerWorldFn serWorld);
   ```
   Behavior: set `seed/initial/ring/tickCount/keyframeInterval`; build `digestTrace` as before; AND walk
   the session tick-by-tick (a `net::Session` advanced one tick at a time — do NOT use the one-shot
   DigestTrace for the keyframe walk; you need the intermediate worlds) capturing a keyframe whenever the
   tick index is a multiple of `keyframeInterval`. **Capture the keyframe AT tick T = the world state
   BEFORE stepping tick T** (i.e. the world as of having applied ticks `[0,T)`), so keyframe at tick 0 ==
   the initial world, keyframe at tick K == the world after K ticks, etc. (This matches `RollbackSession`
   /`CatchUp`'s "world AS OF tick S" convention — RP4 will `CatchUp` from `snap.tick=K` over the tail
   `[K, N)`.) When `keyframeInterval == 0`, capture NO keyframes (degenerate). To keep RP1/RP2 call sites
   working, keep the OLD 6-arg `RecordSession` as a thin overload that forwards `keyframeInterval=0` and a
   no-op/never-called `serWorld` — OR update the RP1/RP2 test call sites to the new signature passing
   `0` + `serToyA` (cleaner; the test owns those call sites). Pick the cleaner one and note it.
   NOTE the determinism subtlety: `net::DigestTrace` records digest AFTER each tick; the keyframe at tick
   T is BEFORE tick T. Walk one consistent loop: `for T in [0,ticks): if T % interval == 0 capture(world
   before stepping); Advance; record digest after`. Keyframe at tick 0 captured before any Advance ==
   initial world. (If `ticks` is a multiple of interval there is NO keyframe at tick==ticks; that's fine
   — RP4 seeks to a keyframe at-or-before the target.)
4. **EncodeDemo** — write the new 36-byte header (add `PutU32(out, keyframeByteLen)`), then world, then
   inputs, then the keyframe section. Compute `keyframeByteLen` by serializing the section first:
   `PutU32(count)`, then per keyframe `PutU32(tick) + PutU32(kfBytes.size()) + PutBytes(kfBytes)`;
   `keyframeByteLen = section.size()` (0 if no keyframes). EncodeDemo no longer hardcodes
   `keyframeInterval=0` — it writes `rec.keyframeInterval`.
5. **DecodeDemo** — read the 7th field `keyframeByteLen` (offset 28), world blob now at **offset 36**,
   input blob at `36 + worldByteLen`, keyframe section at `36 + worldByteLen + inputByteLen` (parse it if
   `keyframeByteLen > 0`: read `keyframeCount`, then each `(tick, kfWorldByteLen, kfWorldBytes)` →
   `deserWorld(kfWorldBytes)`). Add the decoded keyframes to `Demo<World,Input>`:
   ```cpp
   struct DecodedKeyframe { uint32_t tick; World world; };
   std::vector<DecodedKeyframe> keyframes;   // add to Demo<World,Input>
   ```
   Recompute ALL offsets from the actual field count — do NOT trust comment arithmetic; the header is
   8 (magic) + 7*4 = 36 bytes.

## The goldens (PINNED, cross-platform) — tests/replay_test.cpp
1. **RE-PIN RP1/RP2 file hash** — the existing `demoFileBytes` is now recorded with the new 6-arg... new
   call (keyframeInterval=0, serToyA). Its hash changes from `0x2add2e0b07ffcce4`; run once, read the new
   printed hash, re-pin it. The ToyA final digest assertion stays `0x6227bc7b4046d08a` (UNCHANGED). All
   existing RP1+RP2 assertions stay green against the re-pinned hash.
2. **NEW: keyframed demo at interval K1=4.** Record the SAME ToyA session with `keyframeInterval=4`,
   encode → `demoKf4`. Assert:
   ```
   rp3-keyframe: demoKf4 size = <N> bytes, keyframes = <C>
   rp3-keyframe: DigestBytes(demoKf4) = 0x.... (pinned, distinct from the no-keyframe hash)
   PASS rp3-keyframe: keyframeInterval==4 header field round-trips; keyframeCount == ceil(16/4)==4 (ticks 0,4,8,12)
   PASS rp3-keyframe: each keyframe-at-tick-T world digest == live world digest at tick T (net::DigestTrace boundary / RunLockstep-to-T)
   PASS rp3-keyframe: decode(demoKf4) keyframes round-trip (tick + world) byte/-state exact
   PASS rp3-keyframe: Replay(demoKf4) final digest still == 0x6227bc7b4046d08a (keyframes don't change playback)
   ```
   The keyframe-at-T check is the make-or-break: `DigestA(demo.keyframes[i].world)` must equal the live
   world digest after `T` ticks — compute the live reference as
   `net::RunLockstep<ToyA,InA>(ToyA{}, ring, T, StepA, DigestA)` for `T` in {0,4,8,12} (T=0 → DigestA of
   `ToyA{}`). This proves a keyframe is a BIT-IDENTICAL mid-session restore point, the seek foundation.
3. **NEW: keyframed demo at interval K2=8** (the tradeoff) — record with `keyframeInterval=8` → `demoKf8`,
   pin its (distinct, smaller) file hash, assert `keyframeCount == 2` (ticks 0,8) and `DigestBytes(demoKf8)
   != DigestBytes(demoKf4)` and both `!= ` the no-keyframe hash. (Proves the interval is a real,
   deterministic knob: smaller interval → more keyframes → larger file.)
4. Keep RP1+RP2 assertions green (re-pinned file hash; ToyA digest unchanged).

## Cross-platform proof (cheap loop — NO render-bake)
Controller `scp`s `engine/replay/replay.h` + `engine/net/session.h` + `tests/replay_test.cpp`
(+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs
`clang++ -std=c++20 -I engine -I tests tests/replay_test.cpp -o /tmp/rp3 && /tmp/rp3`, confirming ALL
assertions PASS with the IDENTICAL re-pinned + keyframe hashes. NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- `engine/replay/replay.h` stays SELF-CONTAINED (only `<cstdint>/<cstddef>/<vector>` + `net/session.h`;
  NO new includes). Do NOT modify `net/session.h`, `audio/wav.cpp`, or any other existing header. This is
  the ONE slice that edits RP1/RP2's format code (DemoHeader/Recorder/RecordSession/EncodeDemo/DecodeDemo)
  — do it surgically and keep every prior assertion meaning intact (only the file-HASH constants change;
  the ToyA digest, round-trip, integrity, and replay-equals-live semantics stay).
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<functional>`/`<fstream>`.
- `tests/replay_test.cpp` stays self-contained; re-pin the two RP1/RP2... (the one no-keyframe) file hash,
  add the RP3 keyframe assertions. Keep ToyA final digest `0x6227bc7b4046d08a` pinned + green.
- Branch `fix-replay-rp3`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target replay_test'`
  then run `replay_test` and confirm ALL assertions (RP1 re-pinned + RP2 + RP3) PASS, exit 0.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `replay_test` builds + PASSES on Windows
  with every assertion green (incl. the keyframe-at-T == live-at-T check at both intervals), and (if a
  local clang exists) the standalone clang compile passes. Report: commit hash, full test output (all
  PASS lines + the printed hashes + keyframe counts), the new pinned file hashes (no-keyframe, Kf4, Kf8),
  explicit confirmation the ToyA final digest is STILL `0x6227bc7b4046d08a`, confirmation the header is
  still self-contained (list #includes), and the local-clang result (or none → controller runs Mac).
  Commit message via temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits the diff, runs the Mac/clang standalone for identical hashes, ff-merges to
  master + pushes + deletes the branch + advances to RP4 SEEK.)
