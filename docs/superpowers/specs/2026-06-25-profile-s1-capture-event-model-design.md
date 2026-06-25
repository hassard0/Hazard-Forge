# Slice PROFILE-S1 — Capture event model + interned name table (Issue #31, beachhead)

The beachhead of the DETERMINISTIC SCRUB-FRIENDLY PROFILER CAPTURE (issue #31). A profiler measures TIME
(non-deterministic), so the moat is NOT "a deterministic profiler" — it is a **capture whose STRUCTURE is
deterministic + replayable + scrub-seekable, with TIMING as a non-golden overlay**. S1 establishes that
split at the foundation: a capture event stream + an interned name table, with a `StructuralDigest` that is a
pure function of the **logical content** (event kinds, interned names, structural payloads) and **provably
EXCLUDES the timing overlay**.

**THE LOAD-BEARING INVARIANT (banner it):** the structural digest covers the event stream's *logical content*
only. The parallel `timings` overlay (cpu/gpu nanoseconds) is stored but is **NEVER fed to `StructuralDigest`**
— so the same workload produces the byte-identical structural digest on a fast machine and a slow machine.
The S1 golden PROVES this: populating `timings` with arbitrary nonzero values leaves `StructuralDigest`
unchanged.

The golden is a hard-pinned `net::DigestBytes` over the structural encoding, proven identical Windows/MSVC +
Mac/clang via a standalone clang compile — NO render-bake.

## NEW file: engine/profile/profile.h (namespace hf::profile)
Header-only and **SELF-CONTAINED**: include ONLY `<cstddef>`, `<cstdint>`, `<vector>`, plus
`#include "net/session.h"` (for `hf::net::DigestBytes`). **NO `<string>`** (names are byte-strings —
`std::vector<uint8_t>`), NO `<cmath>` / float / clock / RNG / `<random>` / `<unordered_*>` / `<map>` /
`<functional>` / `std::hash` / `<algorithm>`. Do NOT `#include "replay/replay.h"` or `render_graph.h` —
inline the LE appenders here (the `flow.h`/`asset_compiler.h` self-contained discipline). It MUST compile
standalone: `clang++ -std=c++20 -I engine -I tests tests/profile_test.cpp`. This is ONE growing header —
every later slice (S2–S5) APPENDS a section below S1; do NOT modify S1's symbols once pinned. (New dir
`engine/profile/`.)

### Inline little-endian appenders (self-contained — mirror replay.h:29-49)
```cpp
inline void PutU32(std::vector<uint8_t>& b, uint32_t v) { /* 4 bytes LE, byte-by-byte */ }
inline void PutU64(std::vector<uint8_t>& b, uint64_t v) { /* 8 bytes LE, byte-by-byte */ }
inline void PutBytes(std::vector<uint8_t>& b, const void* p, std::size_t n) { /* append n bytes */ }
```
(Exact bodies from replay.h:29-49 — LE, never a struct memcpy. S2+ reuse these + add `GetU32`/`GetU64`.)

### Types (all in hf::profile)
```cpp
enum class EvKind : uint32_t {              // FROZEN values — S2+ uses them; never renumber
    ScopeEnter = 0, ScopeExit = 1, DrawCall = 2, FrameBegin = 3, FrameEnd = 4,
};

// Interned name table: names stored as raw byte-strings in FIRST-SEEN order (no <string>, no pointers in
// the digest — the chunk_diff.h content-addressing move).
struct NameTable {
    std::vector<std::vector<uint8_t>> names;   // names[id] = the interned bytes for nameId `id`
};
// Intern `[p, p+n)`: linear-scan for an existing identical byte-string; return its id, else append + return
// the new id. Deterministic (first-seen order). (Linear scan is fine — name counts are small; NO unordered.)
inline uint32_t Intern(NameTable& t, const void* p, std::size_t n);

struct CaptureEvent {                       // ONE structural event — NO timing here
    EvKind   kind   = EvKind::ScopeEnter;
    uint32_t nameId = 0;                     // index into NameTable (ScopeEnter/Draw); 0 for frame markers
    uint32_t a      = 0;                     // kind-specific structural payload (e.g. DrawCall drawCount)
    uint32_t b      = 0;                     // kind-specific structural payload (reserved; e.g. depth)
};

struct TimingSample { uint64_t cpuNanos = 0; uint64_t gpuNanos = 0; };  // the NON-golden overlay

struct Capture {
    NameTable                  names;
    std::vector<CaptureEvent>  events;       // the structural column
    std::vector<TimingSample>  timings;      // the overlay column — parallel to events (same index)
};
```

### Functions (pure, deterministic on the structural path)
1. **Emitters** — append a structural event AND a parallel (zero) timing sample, keeping the two columns the
   same length:
   ```cpp
   inline void EmitEnter(Capture& c, uint32_t nameId);            // {ScopeEnter, nameId, 0, 0}
   inline void EmitExit (Capture& c, uint32_t nameId);            // {ScopeExit,  nameId, 0, 0}
   inline void EmitDraw (Capture& c, uint32_t nameId, uint32_t drawCount);  // {DrawCall, nameId, drawCount, 0}
   ```
   Each pushes the event AND a `TimingSample{}` (zero) so `events.size() == timings.size()` always. (S6's
   live `ScopedZone` overwrites a timing slot; the structural path never reads it.)
2. **`EncodeStructural(const Capture& c) -> std::vector<uint8_t>`** — hand-LE, the GOLDEN bytes. FIXED order:
   - `PutBytes(magic = "HFCAP1\0\0", 8)`, `PutU32(version = 1)`.
   - `PutU32(nameCount)`; then for each name in order: `PutU32(len)`, `PutBytes(nameBytes, len)`.
   - `PutU32(eventCount)`; then for each event: `PutU32((uint32_t)kind)`, `PutU32(nameId)`, `PutU32(a)`,
     `PutU32(b)`.
   - **The `timings` vector is NOT serialized here** — that is the whole point (it lives in a separate
     section in S5's full `EncodeCapture`).
3. **`StructuralDigest(const Capture& c) -> uint64_t`** → `net::DigestBytes(EncodeStructural(c).data(),
   EncodeStructural(c).size())` (compute the encoding once into a local).

### Fixture (deterministic, FIXED forever)
- `Capture MakeShowcaseCapture()` — a fixed scripted stream: intern names `"Frame"`, `"Shadow"`, `"Lit"`;
  emit `Enter Frame; Enter Shadow; Draw(Shadow, 2); Exit Shadow; Enter Lit; Draw(Lit, 5); Exit Lit; Exit
  Frame`. (Keep FIXED forever — the golden pins its structural digest.) Leave `timings` all-zero.

## The golden (PINNED, cross-platform) — tests/profile_test.cpp
Self-contained test in the `obj_loader_test.cpp` / `flow_test.cpp` shape (copy the `check()` helper +
`HF_TEST_MAIN_INIT()` from `tests/test_main.h`). Register `hf_add_pure_test(profile_test)` in
`tests/CMakeLists.txt` next to `seq_test`.
```
profile-s1: showcase structural digest = 0x<...>
PASS profile-s1: StructuralDigest(MakeShowcaseCapture()) == pinned uint64 (cross-platform structural anchor)
PASS profile-s1: re-encoding the same capture is byte-identical (deterministic)
PASS profile-s1: TIMING IS EXCLUDED — filling timings with arbitrary nonzero leaves StructuralDigest UNCHANGED
PASS profile-s1: interning is first-seen-stable — the same name returns the same id; a new name a new id
PASS profile-s1: a changed structural field (a draw count) changes the digest (structure is load-bearing)
PASS profile-s1: a different scope NAME changes the digest (interned names are load-bearing)
```
Assertions:
1. **PINNED STRUCTURAL DIGEST** — `StructuralDigest(MakeShowcaseCapture())` == a hard-pinned `uint64_t` (run
   once, pin THAT; identical MSVC + clang — the cross-platform anchor).
2. **DETERMINISTIC** — a second `EncodeStructural` of the same capture → byte-identical (and digest equal).
3. **TIMING EXCLUDED (the load-bearing moat proof)** — clone the showcase, set EVERY `timings[i]` to
   arbitrary nonzero values (e.g. `{i*1000+7, i*9}`), and assert `StructuralDigest` == the pinned value
   UNCHANGED. (Proves timing is not in the structural digest — the foundation of the whole flagship.)
4. **INTERN STABLE** — `Intern(t, "Lit")` twice returns the same id; `Intern(t, "New")` returns a fresh id;
   the showcase's `"Frame"`/`"Shadow"`/`"Lit"` got ids 0/1/2 in first-seen order.
5. **STRUCTURE LOAD-BEARING** — clone the showcase, change one `DrawCall`'s `a` (the draw count), re-digest →
   DIFFERENT.
6. **NAME LOAD-BEARING** — build a capture identical except one scope interned under a different name byte →
   DIFFERENT digest.

## Cross-platform proof (the cheap loop — NO render-bake)
The CONTROLLER `scp`s `engine/profile/profile.h` + `engine/net/session.h` + `tests/profile_test.cpp` (+
`tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine
-I tests tests/profile_test.cpp -o /tmp/profile && /tmp/profile`, confirming the test PASSES with the
IDENTICAL pinned digest. (A local Windows clang at `C:\Program Files\LLVM\bin\clang++.exe` is the fast
pre-check.) NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- NEW header `engine/profile/profile.h` (new dir `engine/profile/`); compiles STANDALONE under clang with
  `-I engine -I tests` (self-contained: only `<cstddef>/<cstdint>/<vector>` + `net/session.h`). Do NOT modify
  `net/session.h` / any existing header. Do NOT `#include "replay/replay.h"` or `render_graph.h` (inline the
  LE appenders). Do NOT add it to any RHI/GPU target.
- Pure-CPU INTEGER on the bit-exact path: NO float / `<cmath>` / clock / RNG / `<random>` / `<unordered_*>` /
  `<map>` / `<functional>` / `std::hash` / `<algorithm>` / `<string>`. Names are `std::vector<uint8_t>`
  byte-strings interned by raw bytes in first-seen order.
- **Timing NEVER enters `StructuralDigest`** (the load-bearing invariant — banner it; the golden proves it).
- `tests/profile_test.cpp` is SELF-CONTAINED (copy the scaffolding). Register `hf_add_pure_test(profile_test)`
  in `tests/CMakeLists.txt`. Use `test_main.h` `HF_TEST_MAIN_INIT()`.
- Branch `fix-profile-s1`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target profile_test'`
  (NOTE: the `cmd /c '...&&...'` handoff can swallow output on this machine — if so, run the build via the
  PowerShell tool's native invocation instead.) Run the test exe, confirm it PRINTS the digest and PASSES.
  First run: pick the pinned digest from the printed value, pin it, rebuild, confirm green. ALSO compile
  standalone with the local clang and confirm the IDENTICAL digest (MSVC==clang).
- COMPLETION CRITERIA — do NOT commit until: the header compiles, `profile_test` builds + PASSES on Windows
  with all assertions green, and the local clang standalone passes with the identical digest. Report: the
  commit hash, the full test output (printed digest + PASS lines), the exact pinned `uint64_t`, confirmation
  the header is self-contained (list its `#include`s — exactly 4: `<cstddef>/<cstdint>/<vector>` +
  `net/session.h`), the showcase event stream you fixed, and the local-clang result. Commit message via a
  temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits the diff, runs the Mac/clang standalone to confirm the IDENTICAL digest, then
  ff-merges to master + pushes + deletes the branch + advances to S2 — the hierarchical scope/zone tree.)
