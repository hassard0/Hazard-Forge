# Slice RP1 — Demo-file format + RECORD (Flagship #28 REPLAY/DEMO, 1st/6 — beachhead)

The beachhead of the deterministic REPLAY/DEMO flagship — the capstone that PRODUCTIZES the
determinism moat. A demo file does NOT store state-over-time; it stores the **causal seed of a
session** — (seed, the initial-world snapshot, the per-tick INPUT stream, the per-tick digest trace)
— so that later slices re-derive the byte-identical world by re-running the EXISTING deterministic
Step. RP1 establishes the on-disk format and the RECORDER, and PINS the demo file's hash so the byte
layout is proven identical on Windows/MSVC and Mac/clang.

This is a PURE-CPU INTEGER, header-only slice in the netcode/audio mold (NS1 `session.h`, dsp.h) —
NO render-bake, NO GPU, NO image golden. The golden is a hard-pinned `uint64_t` =
`net::DigestBytes(demoFileBytes)`, proven identical on both platforms by compiling the SAME test.

## NEW file: engine/replay/replay.h (namespace hf::replay)
Header-only and **SELF-CONTAINED**: include ONLY `<cstdint>`, `<cstddef>`, `<vector>` and
`"net/session.h"` (which is itself self-contained — only those three std headers). NO fpx / RHI / GPU
/ `<functional>` / `<cmath>` / `<fstream>`. This is mandatory so `replay_test.cpp` compiles standalone
with `clang++ -std=c++20 -I engine -I tests tests/replay_test.cpp` on the Mac — the cheap
cross-platform proof (exactly like `session_test.cpp`).

Define, all in `hf::replay`:

1. **Little-endian byte appenders** (mirror `audio/wav.cpp:11-26` VERBATIM in discipline — hand-serialize
   EVERY multi-byte field; NEVER `memcpy` a host struct, which would embed host endianness/padding and
   break the Mac hash):
   ```cpp
   inline void PutU32(std::vector<uint8_t>& b, uint32_t v);   // 4 bytes LE
   inline void PutU64(std::vector<uint8_t>& b, uint64_t v);   // 8 bytes LE
   inline void PutBytes(std::vector<uint8_t>& b, const void* p, std::size_t n);  // raw byte copy, in order
   ```
   (`PutU64` = the 8-byte extension of `wav.cpp`'s `PutU32`. `PutBytes` appends `n` bytes from `p` in
   address order — used for the magic tag, the initial-snapshot bytes, and the input bytes, each of
   which the CALLER has already laid out LE.) Add matching readers for the header round-trip test:
   `GetU32(const uint8_t* p)` / `GetU64(const uint8_t* p)` returning the value (LE), pure of side effects.

2. **DemoHeader** — the fixed-layout file header (a plain struct in memory; serialized field-by-field LE,
   NEVER memcpy'd):
   ```cpp
   struct DemoHeader {
       char     magic[8];          // "HFDEMO\0\0" — 8 bytes, exact
       uint32_t version;           // = kDemoVersion (start at 1)
       uint32_t seed;              // the session seed (RP1: a fixed constant for the ToyA demo)
       uint32_t tickCount;         // number of recorded ticks
       uint32_t keyframeInterval;  // RP1: 0 (no keyframes yet — RP3 introduces them). A HEADER FIELD on purpose.
       uint32_t worldByteLen;      // length of the initial-snapshot blob that follows
       uint32_t inputByteLen;      // length of the serialized input stream that follows
   };
   inline constexpr uint32_t kDemoVersion = 1;
   inline constexpr char     kDemoMagic[8] = {'H','F','D','E','M','O','\0','\0'};
   ```

3. **Recorder<World, Input>** — captures one deterministic session's causal seed. RP1 keeps it minimal
   and DRIVER-GENERIC (the driver supplies how to serialize a World snapshot and an Input — 2 free
   template callables, NO `<functional>`):
   ```cpp
   template <class World, class Input>
   struct Recorder {
       uint32_t                        seed = 0;
       World                           initial{};        // the world snapshot AS OF tick 0
       net::InputRing<Input>           ring;             // the per-tick input stream (reuse net::InputRing)
       std::vector<uint64_t>           digestTrace;      // per-tick digest trace (reuse net::DigestTrace)
       uint32_t                        tickCount = 0;
   };
   ```
   Provide a free function that builds a Recorder from a session description:
   ```cpp
   // Record `ticks` of a session: capture seed, the initial world, the input ring, and the per-tick
   // digest trace (via net::DigestTrace — the SAME confirmed-state checksum stream a peer emits).
   template <class World, class Input, class StepFn, class DigestFn>
   Recorder<World,Input> RecordSession(uint32_t seed, World initial, const net::InputRing<Input>& ring,
                                       uint32_t ticks, StepFn step, DigestFn digest);
   ```
   It sets `seed/initial/ring/tickCount` and `digestTrace = net::DigestTrace(initial, ring, ticks, step, digest)`.
   (RP1 records the digest trace into the Recorder but does NOT yet need to serialize it into the file —
   keep the file = header + initial-snapshot + input stream for RP1; the trace serialization can land in
   RP2 where playback VERIFIES it. If trivial to include now, append it LE AFTER the input stream and add
   a `traceByteLen` header field — implementer's call, but pin whichever layout you choose.)

4. **EncodeDemo** — serialize a Recorder to a `std::vector<uint8_t>`, hand-LE, deterministic, byte-exact:
   ```cpp
   template <class World, class Input, class SerWorldFn, class SerInputFn>
   std::vector<uint8_t> EncodeDemo(const Recorder<World,Input>& rec,
                                   SerWorldFn serWorld,    // (const World&)  -> std::vector<uint8_t> (LE)
                                   SerInputFn serInputRing); // (const net::InputRing<Input>&) -> std::vector<uint8_t> (LE)
   ```
   Layout (ALL fields LE, in this exact order):
   `magic(8) | version(u32) | seed(u32) | tickCount(u32) | keyframeInterval(u32) | worldByteLen(u32) |
   inputByteLen(u32) | <worldBytes> | <inputBytes>`. Compute `worldByteLen`/`inputByteLen` from the
   serialized blobs. The driver's `serWorld`/`serInputRing` MUST themselves be hand-LE (field-by-field,
   no struct memcpy) — for ToyA that means writing `acc` as PutU64 of its bit pattern (it's an
   `int64_t` — serialize the two's-complement bits as `uint64_t` LE), and the input ring as
   `tickCount(u32)` then for each tick `count(u32)` then each input (an `int32_t` → PutU32 of its bits).

## The golden (PINNED, cross-platform)
In `tests/replay_test.cpp` (registered `hf_add_pure_test(replay_test)` in `tests/CMakeLists.txt`,
alongside `session_test`), record THE fixed ToyA session — identical to `session_test.cpp`:
- `ToyA{ int64_t acc }`, `using InA = int32_t`, `StepA`/`DigestA` COPIED verbatim from
  `session_test.cpp:34-41` (the test is self-contained; do NOT include session_test).
- The fixed input ring = `makeRingA()` from `session_test.cpp:60-69` (the exact same AddInput calls:
  (0,5),(1,3),(1,-2),(3,7),(7,11),(7,4),(7,-9),(10,2),(15,6)), `kTicks = 16`, a fixed `seed` constant.
- `RecordSession(seed, ToyA{}, ring, 16, StepA, DigestA)` → `EncodeDemo(rec, serToyA, serRingA)` →
  `demoFileBytes`.

Pin and assert:
```
rp1-record: demo file bytes -> DigestBytes == <PINNED uint64>   (the cross-platform byte-layout proof)
rp1-record: header round-trips field-exact (magic/version/seed/tickCount/keyframeInterval/worldByteLen/inputByteLen)
rp1-record: digestTrace.size() == 16 AND digestTrace.back() == <PINNED ToyA final digest>   (== session_test's hToyA)
rp1-record: re-encoding the same Recorder is byte-identical (deterministic, no timestamps/padding)
```
Assertions:
1. **PINNED FILE HASH** — `net::DigestBytes(demoFileBytes.data(), demoFileBytes.size())` == a hard-pinned
   `uint64_t` literal in the test. THIS is the make-or-break cross-platform proof: the SAME value must
   print on Windows/MSVC and Mac/clang (run the test once on each; the pinned value is whatever both
   agree on — the implementer pins the Windows value, the CONTROLLER confirms Mac/clang prints the
   identical hash).
2. **HEADER ROUND-TRIP** — decode the header fields back out of `demoFileBytes` via `GetU32`/`PutBytes`
   reads and check each equals what went in (magic == "HFDEMO\0\0", version == 1, tickCount == 16,
   keyframeInterval == 0, worldByteLen/inputByteLen == the blob lengths).
3. **DIGEST-TRACE PROVENANCE** — `rec.digestTrace` has length 16 and its LAST entry equals the pinned
   ToyA final digest (the SAME number `session_test.cpp` pins as `hToyA` — re-pin it here so RP1 is
   self-checking; the two tests independently agreeing is the cross-check).
4. **RE-ENCODE DETERMINISM** — `EncodeDemo` of the same Recorder twice is byte-identical (no clock, no
   RNG, no uninitialized padding).

## Cross-platform proof (the cheap loop — NO render-bake)
This is a numeric flagship: the golden is a pinned `uint64_t`, NOT an image. The CONTROLLER proves
cross-platform by `scp`-ing `tests/replay_test.cpp` + `engine/replay/replay.h` + `engine/net/session.h`
to the Mac and running `clang++ -std=c++20 -I engine -I tests tests/replay_test.cpp -o /tmp/rp1 && /tmp/rp1`,
confirming the test PASSES (the pinned hash is reproduced) — exactly the `session_test` / `dsp_test`
standalone-clang precedent. NO Metal, NO `tests/golden/metal/*`, NO `--*-shot`.

## Constraints (HARD)
- NEW header `engine/replay/replay.h` only; it must compile STANDALONE under clang with just
  `-I engine -I tests` (self-contained: only `<cstdint>/<cstddef>/<vector>` + `net/session.h`). Do NOT
  add it to any RHI/GPU target. Do NOT modify `net/session.h`, `audio/wav.cpp`, or any existing header
  (read-only reuse only — copy the LE discipline, don't refactor wav.cpp).
- Pure-CPU INTEGER: NO float, NO `<cmath>`, NO clock/RNG, NO `<functional>`, NO `<fstream>` in the
  header (EncodeDemo returns bytes; writing to disk, if ever needed, is a later slice's caller concern).
- `tests/replay_test.cpp` is SELF-CONTAINED (copy ToyA/StepA/DigestA/makeRingA in; do NOT include
  session_test.cpp). Register `hf_add_pure_test(replay_test)` in `tests/CMakeLists.txt` next to
  `session_test`. Use `test_main.h` `HF_TEST_MAIN_INIT()` like the other pure tests.
- Branch `fix-replay-rp1`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target replay_test'`
  then run the test exe (under `build\windows-msvc-release`) and confirm it PRINTS the pinned hash and
  PASSES. If the test fails on the pinned value the FIRST time (you have not yet picked it), run once,
  read the printed `DigestBytes`, pin THAT, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until: the header compiles, `replay_test` builds + PASSES on
  Windows with all four assertions green, AND you have ALSO locally sanity-compiled the test standalone
  with clang IF a local clang exists (if not, note it — the controller does the Mac clang run). Report:
  the commit hash, the full test output (the printed pinned hash + PASS lines), the exact pinned
  `uint64_t` values (file hash + ToyA final digest), and confirmation the header is self-contained
  (the clang standalone command you expect the controller to run). Commit message via a temp file +
  `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits the diff append-only, runs the Mac/clang standalone to confirm the IDENTICAL
  hash, then ff-merges to master + pushes + deletes the branch + advances to RP2.)
