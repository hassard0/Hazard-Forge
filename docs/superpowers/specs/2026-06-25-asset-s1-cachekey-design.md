# Slice ASSET-S1 — Content-addressed CacheKey (Issue #16, beachhead)

The beachhead of the DETERMINISTIC CONTENT-ADDRESSED ASSET PIPELINE (issue #16) — the layer above the
existing loaders (`obj_loader.h`, `gltf_loader`) that turns raw assets into byte-identical compiled
artifacts with a content-hash cache + incremental rebuild. S1 establishes the cache **key**: a stable
`CacheKey` that is a pure function of `(raw content bytes, integer compile params, asset kind)` — identical
across MSVC/Windows-clang/Mac-clang, independent of any pointer, clock, or file mtime.

**THE LOAD-BEARING INVARIANT (state it in the header banner):** a cache key is **content + params only —
`last_write_time`/mtime NEVER enters a key**. mtime is the *trigger* for a recompile (slice S6's watch), it
is NEVER the *identity* of an artifact. If two machines (or two builds) see the same bytes + same params,
they MUST derive the same key, regardless of when the file was touched. This is the invariant the whole
flagship rests on.

The golden is a hard-pinned `net::DigestBytes` over the key derivation, proven identical Windows/MSVC +
Mac/clang via a standalone clang compile — NO render-bake, the cheapest proof.

## NEW file: engine/asset/asset_compiler.h (namespace hf::asset)
Header-only and **SELF-CONTAINED**: include ONLY `<cstdint>`, `<cstddef>`, `<vector>`, plus
`#include "net/session.h"` (for `hf::net::DigestBytes`). NO `<cmath>` / float / clock / RNG / `<random>` /
`<unordered_*>` / `<map>` / `<functional>` / `std::hash` / `<algorithm>` / `<string>` on the bit-exact path.
Do NOT `#include "replay/replay.h"` — inline the tiny LE appenders here (keep the include set minimal, the
`flow.h` self-contained discipline). It MUST compile standalone:
`clang++ -std=c++20 -I engine -I tests tests/asset_compiler_test.cpp`. This is ONE growing header — every
later slice (S2–S5) APPENDS a section below S1; do NOT modify S1's symbols once pinned.

### Inline little-endian appenders (self-contained — mirror replay.h:29-61 but local to this header)
```cpp
inline void PutU32(std::vector<uint8_t>& b, uint32_t v) {            // 4 bytes LE
    b.push_back((uint8_t)(v & 0xFF)); b.push_back((uint8_t)((v>>8)&0xFF));
    b.push_back((uint8_t)((v>>16)&0xFF)); b.push_back((uint8_t)((v>>24)&0xFF));
}
inline void PutU64(std::vector<uint8_t>& b, uint64_t v) {            // 8 bytes LE
    for (int i = 0; i < 8; ++i) b.push_back((uint8_t)((v >> (8*i)) & 0xFF));
}
```
(These exact bodies — LE, byte-by-byte, never a struct memcpy — are the cross-platform-stable serialization.
S2+ will reuse them and add `PutBytes`/`GetU32`/`GetU64` as needed; S1 needs only `PutU32`/`PutU64`.)

### Types (all in hf::asset)
```cpp
using CacheKey = uint64_t;                       // an FNV-1a-64 content-address

enum class AssetKind : uint32_t {                // S2+ append kinds; do NOT renumber
    Mesh = 0, Texture = 1, Audio = 2, Scene = 3,
};

// Integer/enum/Q16.16 compile options ONLY — NO float (float in the key would break cross-compiler
// determinism). `scale` is Q16.16 fixed-point (int32, the engine's fpx convention: 65536 == 1.0).
struct CompileParams {
    uint32_t recomputeNormals = 0;   // bool as 0/1
    int32_t  scale            = 65536;  // Q16.16 (kOne) — uniform import scale, integer-exact
    uint32_t tangentMode      = 0;   // enum: 0 none / 1 mikktspace-style / ... (S2 may use)
    uint32_t flags            = 0;   // reserved bitfield (future options; serialized so it's load-bearing)
};

struct AssetId {                     // the content-address of one asset
    uint32_t kind         = 0;       // an AssetKind value
    CacheKey contentHash  = 0;       // HashRawAsset over the raw bytes
    CacheKey paramHash    = 0;       // HashParams over the compile options
};
```

### Functions (pure, deterministic)
1. **`CacheKey HashRawAsset(const void* bytes, std::size_t n)`** — `return net::DigestBytes(bytes, n);`
   (the content hash — a pure FNV over the raw input; empty input → the FNV offset basis, a fixed value).
2. **`CacheKey HashParams(const CompileParams& p)`** — hand-LE-serialize EVERY field in a FIXED order
   (`PutU32(recomputeNormals)`, `PutU32((uint32_t)scale)`, `PutU32(tangentMode)`, `PutU32(flags)`) into a
   `std::vector<uint8_t>`, then `net::DigestBytes(buf.data(), buf.size())`. (Serialize `scale` as
   `(uint32_t)scale` — the int32 bit pattern; LE-stable. NEVER memcpy the struct — padding.)
3. **`CacheKey MakeKey(uint32_t kind, CacheKey contentHash, CacheKey paramHash)`** — hand-LE-serialize the
   triple in a FIXED order (`PutU32(kind)`, `PutU64(contentHash)`, `PutU64(paramHash)`) then
   `net::DigestBytes(...)`. The final content-address: `key = f(kind, content, params)`, order-fixed.
4. **`AssetId MakeAssetId(uint32_t kind, const void* bytes, std::size_t n, const CompileParams& p)`** — the
   convenience composer: `{ kind, HashRawAsset(bytes,n), HashParams(p) }`. (The `key` of an AssetId is
   `MakeKey(id.kind, id.contentHash, id.paramHash)`.)

### Fixture (deterministic, FIXED forever)
- `const char* ShowcaseRawBytes()` + its length — a fixed small raw blob (e.g. a tiny inline OBJ-ish text or
  just a fixed byte string like `"hf-asset-fixture-v1\n..."`), used by the golden. Keep FIXED forever — the
  golden pins its key.
- `CompileParams ShowcaseParams()` — a fixed non-default params (e.g. `recomputeNormals=1, scale=2*65536,
  tangentMode=1, flags=0`). Keep FIXED.

## The golden (PINNED, cross-platform) — tests/asset_compiler_test.cpp
Self-contained test in the `obj_loader_test.cpp` / `flow_test.cpp` shape (copy the `check()` helper +
`HF_TEST_MAIN_INIT()` from `tests/test_main.h`). Register `hf_add_pure_test(asset_compiler_test)` in
`tests/CMakeLists.txt` next to `obj_loader_test`.
```
asset-s1: showcase key = 0x<...>
PASS asset-s1: MakeKey(Mesh, HashRawAsset(fixture), HashParams(showcase)) == pinned uint64 (cross-platform)
PASS asset-s1: same (bytes, params, kind) -> the SAME key (reproducible / content-addressed)
PASS asset-s1: a single flipped content byte -> a DIFFERENT key (content is load-bearing)
PASS asset-s1: a changed compile param -> a DIFFERENT key (params are load-bearing)
PASS asset-s1: the key is independent of the buffer's address (two copies of the bytes -> the same key)
PASS asset-s1: a different AssetKind -> a DIFFERENT key (kind is load-bearing)
```
Assertions:
1. **PINNED KEY** — `MakeKey((uint32_t)AssetKind::Mesh, HashRawAsset(ShowcaseRawBytes(), len),
   HashParams(ShowcaseParams()))` == a hard-pinned `uint64_t` (run once, read the printed value, pin THAT;
   identical MSVC + clang — the make-or-break cross-platform anchor).
2. **REPRODUCIBLE** — recomputing from the same inputs → identical key.
3. **CONTENT LOAD-BEARING** — copy the fixture bytes, flip one byte, re-key → a DIFFERENT key.
4. **PARAMS LOAD-BEARING** — clone `ShowcaseParams()`, change one field (e.g. `scale += 1`), re-key →
   DIFFERENT.
5. **POINTER-INDEPENDENT** — build a SECOND buffer holding the SAME bytes at a different address; its key ==
   the fixture's key (proves no pointer/address enters the key — the content-addressed property).
6. **KIND LOAD-BEARING** — `MakeKey(Texture, ...)` with the same content/params != the `Mesh` key.

## Cross-platform proof (the cheap loop — NO render-bake)
The CONTROLLER `scp`s `engine/asset/asset_compiler.h` + `engine/net/session.h` + `tests/asset_compiler_test.cpp`
(+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine
-I tests tests/asset_compiler_test.cpp -o /tmp/asset && /tmp/asset`, confirming the test PASSES with the
IDENTICAL pinned key. (A local Windows clang at `C:\Program Files\LLVM\bin\clang++.exe` is the fast
pre-check.) NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- NEW header `engine/asset/asset_compiler.h` (in the existing `engine/asset/` dir); compiles STANDALONE under
  clang with `-I engine -I tests` (self-contained: only `<cstdint>/<cstddef>/<vector>` + `net/session.h`).
  Do NOT modify `net/session.h` / `obj_loader.h` / any existing header. Do NOT `#include "replay/replay.h"`
  (inline the LE appenders). Do NOT add it to any RHI/GPU target.
- Pure-CPU INTEGER on the bit-exact path: NO float / `<cmath>` / clock / mtime / RNG / `<random>` /
  `<unordered_*>` / `<map>` / `<functional>` / `std::hash` / `<algorithm>` / `<string>`. The key is FNV over
  hand-LE bytes; `scale` is Q16.16 int32.
- **mtime NEVER enters a key** (the load-bearing invariant — banner it).
- `tests/asset_compiler_test.cpp` is SELF-CONTAINED (copy the scaffolding; do NOT include other tests).
  Register `hf_add_pure_test(asset_compiler_test)` in `tests/CMakeLists.txt`. Use `test_main.h`
  `HF_TEST_MAIN_INIT()`.
- Branch `fix-asset-s1`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target asset_compiler_test'`
  Run the test exe, confirm it PRINTS the key and PASSES. First run: pick the pinned key from the printed
  value, pin it, rebuild, confirm green. ALSO compile standalone with the local clang and confirm the
  IDENTICAL key (MSVC==clang).
- COMPLETION CRITERIA — do NOT commit until: the header compiles, `asset_compiler_test` builds + PASSES on
  Windows with all assertions green, and the local clang standalone passes with the identical key. Report:
  the commit hash, the full test output (printed key + PASS lines), the exact pinned `uint64_t`, confirmation
  the header is self-contained (list its `#include`s — exactly 4: `<cstdint>/<cstddef>/<vector>` +
  `net/session.h`), the fixture bytes + params you fixed, and the local-clang result. Commit message via a
  temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits the diff, runs the Mac/clang standalone to confirm the IDENTICAL key, then ff-merges
  to master + pushes + deletes the branch + advances to S2 — the deterministic compiled-artifact format.)
