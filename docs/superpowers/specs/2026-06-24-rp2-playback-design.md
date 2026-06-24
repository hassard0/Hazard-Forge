# Slice RP2 — PLAYBACK: DecodeDemo + Replay (Flagship #28 REPLAY/DEMO, 2nd/6)

RP1 gave us the demo-file FORMAT + the RECORDER (a fixed ToyA session → byte-exact demo bytes, pinned
hash identical Windows/MSVC + Mac/clang). RP2 adds the other half: **PLAYBACK** — decode a demo back
into its causal seed (header + initial-world snapshot + input ring) and **REPLAY** it by re-running the
EXISTING deterministic Step, proving the replayed world is BIT-IDENTICAL to the live session (not
interpolated — re-derived). This is the moat made concrete: a demo recorded on one machine replays to
the same bytes on any machine.

Pure-CPU INTEGER, header-only, append-only to `engine/replay/replay.h` (do NOT modify RP1's existing
code — ADD `DecodeDemo`/`Demo`/`Replay` below it). NO render-bake, NO GPU. Goldens are pinned
`uint64_t`s, proven identical Windows/MSVC + Mac/clang by compiling the SAME `replay_test.cpp`.

## Format stays FROZEN (clean slice boundary)
RP2 does NOT change the demo-file layout — the RP1 pinned file hash `0x2add2e0b07ffcce4` MUST stay
valid (the existing RP1 assertions are untouched). The format's first extension is RP3 (keyframes); the
per-tick digest trace lives in the Recorder in-memory for now and is NOT yet a file field. RP2's
integrity proof therefore compares the replayed trace against the in-memory recorded trace AND a fresh
live `net::DigestTrace` (all three agree) — no file-format change needed.

## Append to engine/replay/replay.h (below the RP1 code, all in hf::replay)

1. **Demo<World,Input>** — the decoded demo (the inverse of EncodeDemo's product):
   ```cpp
   template <class World, class Input>
   struct Demo {
       DemoHeader            header{};   // fields read back out of the file (magic/version/seed/tickCount/...)
       World                 initial{};  // the deserialized initial-world snapshot
       net::InputRing<Input> ring;       // the deserialized per-tick input stream
   };
   ```

2. **DecodeDemo** — parse demo bytes back into a `Demo<World,Input>`, hand-LE (GetU32 + driver
   deserializers, NO struct memcpy). Driver supplies the inverse serializers (2 free template callables):
   ```cpp
   // deserWorld(const uint8_t* p, uint32_t len)            -> World
   // deserInputRing(const uint8_t* p, uint32_t len)        -> net::InputRing<Input>
   template <class World, class Input, class DeserWorldFn, class DeserInputFn>
   Demo<World,Input> DecodeDemo(const std::vector<uint8_t>& bytes,
                                DeserWorldFn deserWorld, DeserInputFn deserInputRing);
   ```
   Steps: read `magic[8]` (PutBytes-order, i.e. `bytes[0..7]`), `version`/`seed`/`tickCount`/
   `keyframeInterval`/`worldByteLen`/`inputByteLen` via `GetU32` at the fixed offsets (8, 12, 16, 20, 24,
   28 — magic is 8 bytes then seven u32s = header is 8 + 7*4 = 36 bytes; the world blob starts at offset
   36). Validate the magic equals `kDemoMagic` and `version == kDemoVersion` (return a default/empty Demo
   or set a header field on mismatch — keep it simple: the test only feeds valid demos in RP2; RP6 does
   the corruption path). Then `deserWorld(&bytes[36], worldByteLen)` and
   `deserInputRing(&bytes[36 + worldByteLen], inputByteLen)`. (Confirm the header size: 8 + 4*7 = 36. The
   implementer MUST recompute this against EncodeDemo's actual layout — magic(8) + version,seed,tickCount,
   keyframeInterval,worldByteLen,inputByteLen = 6 u32 fields = 8 + 24 = 32, world blob starts at 32. RE-DERIVE
   the exact offset from EncodeDemo and the DemoHeader field count; do NOT trust this comment's arithmetic —
   count the actual PutU32 calls in EncodeDemo: there are 6 of them after the 8-byte magic, so the blobs
   start at offset 8 + 6*4 = 32.)

3. **Replay** — restore the initial snapshot and Advance for `tickCount` ticks over the decoded ring,
   returning the final world digest AND the re-derived per-tick trace (for the integrity check). Reuse
   `net::Advance` (the NS1 transition) — playback IS the `CatchUp`/`RunLockstep` body seeded from the
   initial world:
   ```cpp
   template <class World, class Input>
   struct ReplayResult { uint64_t finalDigest = 0; std::vector<uint64_t> trace; };

   template <class World, class Input, class StepFn, class DigestFn>
   ReplayResult<World,Input> Replay(const Demo<World,Input>& demo, StepFn step, DigestFn digest);
   ```
   Implement: `net::Session<World,Input> s; s.world = demo.initial; s.ring = demo.ring; s.tick = 0;`
   then loop `demo.header.tickCount` times calling `net::Advance(s, step)` and pushing `digest(s.world)`
   into `trace`; `finalDigest = digest(s.world)`. (Equivalently call `net::DigestTrace` + take its back()
   — but doing the Advance loop here makes the playback explicit. Either is fine; prefer the explicit loop
   so RP4/RP5 can reuse a `ReplayFrom(world, ring, from, to, ...)` shape — if trivial, factor that helper
   now, append-only.)

## The goldens (PINNED, cross-platform) — extend tests/replay_test.cpp (append below RP1's assertions)
Decode the SAME `demoFileBytes` from RP1 (record → encode → decode) with the inverse ToyA deserializers
(`deserToyA`: GetU64 → `int64_t acc`; `deserRingA`: GetU32 tickCount, then per tick GetU32 count + each
input GetU32 → `int32_t`), then `Replay` it. Print live values, then assert:
```
rp2-playback: replay final digest = 0x.... (== live RunLockstep == pinned ToyA final)
PASS rp2-playback: DecodeDemo round-trips header field-exact (magic/version/seed/tickCount/world+inputByteLen)
PASS rp2-playback: decoded initial world == original initial (ToyA acc == 0) AND decoded ring re-encodes byte-identical
PASS rp2-playback: Replay final digest == live net::RunLockstep == pinned 0x6227bc7b4046d08a
PASS rp2-playback: replay per-tick trace == recorded digestTrace == fresh net::DigestTrace (every tick, integrity)
PASS rp2-playback: Decode(Encode(rec)) re-encoded bytes == demoFileBytes (full round-trip, byte-exact)
```
Assertions:
1. **HEADER ROUND-TRIP** — `DecodeDemo`'s `Demo.header` fields equal what was recorded (magic ==
   "HFDEMO\0\0", version == 1, seed == 0x5EED1234, tickCount == 16, worldByteLen/inputByteLen == the RP1
   blob lengths).
2. **WORLD + RING ROUND-TRIP** — `demo.initial.acc == 0` (the recorded `ToyA{}`); and re-serializing
   `demo.ring` with `serRingA` yields bytes identical to re-serializing the original `ring` (the ring
   decoded byte-exact).
3. **REPLAY == LIVE == PINNED** (the make-or-break) — `Replay(demo, StepA, DigestA).finalDigest` ==
   `net::RunLockstep<ToyA,InA>(ToyA{}, ring, 16, StepA, DigestA)` == the pinned `0x6227bc7b4046d08a`.
   Bit-identical re-derivation, NOT interpolation.
4. **PER-TICK INTEGRITY** — `Replay(...).trace` (length 16) equals `rec.digestTrace` (RP1's in-memory
   recorded trace) at EVERY tick, AND equals a fresh `net::DigestTrace<ToyA,InA>(ToyA{}, ring, 16, StepA,
   DigestA)` at every tick. (This is the corruption detector's foundation: replay reproduces every
   recorded checksum.)
5. **FULL BYTE ROUND-TRIP** — `EncodeDemo(decodedRec...)` reproduces `demoFileBytes` exactly. To do this,
   rebuild a Recorder from the decoded Demo (seed/initial/ring/tickCount + recompute digestTrace) and
   re-encode; assert byte-equal to the original `demoFileBytes`. (Or directly: a freshly built
   `Recorder` from the decoded fields re-encodes to the identical 144 bytes.)

Keep RP1's existing four assertions UNCHANGED and green (the file hash `0x2add2e0b07ffcce4` is still
valid — format frozen).

## Cross-platform proof (the cheap loop — NO render-bake)
Same as RP1: the CONTROLLER `scp`s `engine/replay/replay.h` + `engine/net/session.h` +
`tests/replay_test.cpp` (+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs
`clang++ -std=c++20 -I engine -I tests tests/replay_test.cpp -o /tmp/rp2 && /tmp/rp2`, confirming the
test PASSES (all RP1 + RP2 assertions, identical pinned hashes). NO Metal, NO `tests/golden/*`, NO `--*-shot`.

## Constraints (HARD)
- APPEND-ONLY to `engine/replay/replay.h` (add `Demo`/`DecodeDemo`/`ReplayResult`/`Replay` below the RP1
  code; do NOT modify RP1's `PutU*`/`GetU*`/`DemoHeader`/`Recorder`/`RecordSession`/`EncodeDemo`). Header
  stays SELF-CONTAINED (only `<cstdint>/<cstddef>/<vector>` + `net/session.h`; NO new includes). Do NOT
  modify `net/session.h`, `audio/wav.cpp`, or any other existing header.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<functional>`/`<fstream>`.
- `tests/replay_test.cpp` stays self-contained; APPEND the RP2 assertions + the `deserToyA`/`deserRingA`
  inverse serializers (no new includes). Keep RP1's assertions intact.
- Branch `fix-replay-rp2`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target replay_test'`
  then run the built `replay_test` exe and confirm ALL assertions (RP1 + RP2) PASS, exit 0.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `replay_test` builds + PASSES on
  Windows with every RP1 + RP2 assertion green, and (if a local clang exists) the standalone clang
  compile passes too. Report: commit hash, full test output (all PASS lines + the printed digests),
  confirmation RP1's hash `0x2add2e0b07ffcce4` and ToyA digest `0x6227bc7b4046d08a` are unchanged,
  confirmation the header is still self-contained (list its #includes), and the local-clang result (or
  that none exists — the controller runs the Mac proof). Commit message via temp file + `git commit -F`
  (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical hashes, ff-merges
  to master + pushes + deletes the branch + advances to RP3 keyframes.)
