# Slice ASSET-S2 — Deterministic compiled-artifact format (Issue #16)

S1 shipped the content-addressed `CacheKey` (key `0x7fb6a48b4b99f1b7`). S2 turns a raw asset into a
**byte-identical compiled artifact**: a raw OBJ → `ParseObj` → a canonical, versioned, hand-LE binary blob
whose geometry is **quantized to Q16.16 integers** (NOT raw float bits) so the artifact is bit-identical
across MSVC/Windows-clang/Mac-clang. The golden is the artifact's pinned `net::DigestBytes`.

**THE CROSS-COMPILER TRICK:** the loader (`ParseObj`) produces floats, but the artifact stores **Q16.16
integers**. The float→Q16.16 conversion `(int32_t)(f * 65536.0f)` is **exact and deterministic** because
65536 is a power of two — the multiply only shifts the exponent (no mantissa rounding, no FMA/x87 ambiguity),
and the truncation of an exact value is identical on every compiler (for in-range `|f| < 32768`). So the one
float operation in the compile step yields a cross-compiler-stable integer; the artifact + its digest are
pure integer. (This is the controlled float→fixed boundary; everything downstream — S3 cache, S4 graph, S5
manifest — is pure integer over these blobs.)

Pure-integer artifact, append-only to `engine/asset/asset_compiler.h` (below S1; do NOT modify S1's symbols
— the key `0x7fb6a48b4b99f1b7` stays pinned). This slice adds ONE include: `#include "asset/obj_loader.h"`
(for `ParseObj`/`ObjMesh`/`ObjVertex` — itself a header-only self-contained loader). The standalone clang
compile stays valid (`-I engine -I tests`).

## Append to engine/asset/asset_compiler.h (below S1, in hf::asset)

### 1. The Q16.16 quantizer + integer multiply (inline, self-contained)
```cpp
// Exact float->Q16.16: 65536 is a power of two, so f*65536.0f shifts the exponent with NO rounding (exact
// for |f| < 32768), and the truncation is identical on every compiler. The ONE float op in the compile step;
// the artifact it produces is pure integer.
inline int32_t FxQuantize(float f) { return (int32_t)(f * 65536.0f); }
// Q16.16 fixed-point multiply (integer): (a*b) >> 16 via int64 intermediate (the fpx convention).
inline int32_t FxMul(int32_t a, int32_t b) { return (int32_t)(((int64_t)a * (int64_t)b) >> 16); }
```
(Reuse S1's `PutU32`/`PutU64`; ADD `PutBytes`/`GetU32`/`GetU64` if needed for the round-trip — inline,
mirroring replay.h:46-60, still NOT including replay.h.)

### 2. The compiled-mesh artifact (canonical, versioned, hand-LE)
```cpp
constexpr uint32_t kCompiledMeshMagic   = 0x484D4D31; // 'HMM1' (Hazard Mesh, v-tagged by `version`)
constexpr uint32_t kCompiledMeshVersion = 1;          // bump if the Q16.16 grid / layout changes

struct CompiledMesh {                     // the DECODED view (S2 round-trip target)
    uint32_t magic = 0, version = 0;
    uint32_t vertexCount = 0, indexCount = 0;
    CompileParams params;                 // the options baked in (echoed in the header so they're load-bearing)
    std::vector<int32_t>  verts;          // vertexCount * 8 Q16.16 ints: pos.xyz, uv.xy, normal.xyz (fixed stride)
    std::vector<uint32_t> indices;        // indexCount triangle-list indices
};
```

### 3. `CompileObj` — raw OBJ text → the blob
```cpp
inline std::vector<uint8_t> CompileObj(const char* text, std::size_t n, const CompileParams& p);
```
- `ObjMesh m = ParseObj(text, n);`
- Build the blob with the inline LE appenders, FIXED field order:
  `PutU32(magic)`, `PutU32(version)`, `PutU32(vertexCount = m.vertices.size())`,
  `PutU32(indexCount = m.indices.size())`, then the **params header** (`PutU32(recomputeNormals)`,
  `PutU32((uint32_t)scale)`, `PutU32(tangentMode)`, `PutU32(flags)` — so any param change re-digests the
  artifact), then **per vertex** in order: the 8 components quantized —
  `pos.x/y/z` each `FxMul(FxQuantize(v.pos[i]), p.scale)` (the import `scale` applied as an exact integer
  Q16.16 multiply — a real geometric transform, the load-bearing param), then `uv.x/y` =
  `FxQuantize(v.uv[i])`, then `normal.x/y/z` = `FxQuantize(v.normal[i])` — each `PutU32((uint32_t)q)` LE;
  then `PutU32(idx)` for every index.
- Return the `std::vector<uint8_t>`. (Pure integer after `FxQuantize`; the blob carries no float bits.)
  (v1 applies `scale` geometrically; `recomputeNormals`/`tangentMode` are recorded in the header — load-bearing
  on the digest — but their geometry effect is a documented v1-deferred refinement, NOT yet recomputed.)

### 4. `DecodeCompiledMesh` + `DigestArtifact`
```cpp
inline bool      DecodeCompiledMesh(const std::vector<uint8_t>& blob, CompiledMesh& out); // false on truncation/bad magic
inline CacheKey  DigestArtifact(const std::vector<uint8_t>& blob) { return net::DigestBytes(blob.data(), blob.size()); }
```
`DecodeCompiledMesh` reads the fixed-order fields back (`GetU32`/`GetU64`), validates `magic ==
kCompiledMeshMagic` and that the byte length matches `header + 8*vertexCount*4 + indexCount*4`, fills
`out.verts`/`out.indices`/`out.params`, returns false on any mismatch. (Hand-LE field-by-field — never a
struct memcpy.)

## The golden (PINNED, cross-platform) — append to tests/asset_compiler_test.cpp
```
asset-s2: compiled-artifact digest = 0x<...>  (<N> bytes)
PASS asset-s1: ... (all 6 S1 assertions STILL green — key 0x7fb6a48b4b99f1b7 UNCHANGED)
PASS asset-s2: CompileObj(fixture, showcase) magic/version == kCompiledMeshMagic / 1
PASS asset-s2: the compiled-artifact digest == pinned uint64 (Q16.16 blob, byte-stable cross-platform)
PASS asset-s2: DecodeCompiledMesh(blob) round-trips — vertexCount/indexCount/verts/indices recovered
PASS asset-s2: a changed compile param (scale) changes the artifact digest (params are load-bearing)
PASS asset-s2: different raw bytes -> a different artifact digest (content is load-bearing)
PASS asset-s2: the artifact is pure-integer Q16.16 — the showcase positions decode to exact integer values
```
Assertions:
1. **S1 INVARIANT** — re-run S1 assertion 1: the showcase key == `0x7fb6a48b4b99f1b7` UNCHANGED (S2 is
   additive).
2. **HEADER** — `CompileObj(ShowcaseRawBytes(), ShowcaseRawLen(), ShowcaseParams())` decodes to `magic ==
   kCompiledMeshMagic`, `version == 1`.
3. **PINNED ARTIFACT DIGEST** — `DigestArtifact(CompileObj(fixture, ShowcaseParams()))` == a hard-pinned
   `uint64_t` (run once, pin THAT; identical MSVC + clang — the cross-compiler anchor).
4. **ROUND-TRIP** — `DecodeCompiledMesh(blob, out)` succeeds; `out.vertexCount == 3`, `out.indexCount == 3`
   (the fixture triangle), and re-encoding `out` (or comparing `out.verts`/`out.indices`) matches.
5. **PARAM LOAD-BEARING** — compile with `scale` changed (e.g. `3*65536`) → a DIFFERENT artifact digest.
6. **CONTENT LOAD-BEARING** — compile a different OBJ text → a DIFFERENT artifact digest.
7. **PURE-INTEGER EXACTNESS** — the fixture's position `(1,0,0)` with `scale=2.0` decodes to the exact
   Q16.16 integer `131072` (= 2.0) — proving the quantization is exact, not float-fuzzy. (Pick a vertex whose
   decoded `pos.x` you can assert `==` an exact int.)

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/asset/asset_compiler.h` + `engine/asset/obj_loader.h` + `engine/net/session.h` +
`tests/asset_compiler_test.cpp` (+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and
runs `clang++ -std=c++20 -I engine -I tests tests/asset_compiler_test.cpp -o /tmp/asset && /tmp/asset`,
confirming ALL assertions PASS with the IDENTICAL pinned digests (esp. the artifact digest — the
quantization must be cross-compiler bit-exact). Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/asset/asset_compiler.h` (add `FxQuantize`/`FxMul`/`CompiledMesh`/`CompileObj`/
  `DecodeCompiledMesh`/`DigestArtifact` + any `PutBytes`/`GetU32`/`GetU64` below S1). Do NOT modify S1's
  types/functions — the key `0x7fb6a48b4b99f1b7` stays pinned.
- The ONLY new include is `#include "asset/obj_loader.h"` (next to `net/session.h`). Header stays
  standalone-clang-compilable. STILL NO `<cmath>` (the quantize is a bare `*`/cast, no `<cmath>`), NO RNG,
  NO clock/mtime, NO `<unordered_*>`/`<map>`/`std::hash`/`<algorithm>`. `FxQuantize` is the ONLY float op
  (exact, power-of-2); the artifact + digest are pure integer.
- mtime NEVER enters the artifact or its digest.
- `tests/asset_compiler_test.cpp` stays self-contained; APPEND the S2 assertions. Keep ALL 6 S1 assertions
  green.
- Branch `fix-asset-s2`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target asset_compiler_test'`
  then run `asset_compiler_test`, confirm ALL assertions (S1 + S2) PASS, exit 0. ALSO compile standalone with
  the local clang `C:\Program Files\LLVM\bin\clang++.exe` and confirm the IDENTICAL digests (esp. the
  artifact digest — if it differs MSVC vs clang, the quantization is not exact; investigate before pinning).
  First run: pick the pinned artifact digest from the printed value, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `asset_compiler_test` builds + PASSES on
  Windows with every assertion green (esp. the S1 key unchanged + the pinned artifact digest + the round-trip
  + the pure-integer exactness), and the local clang standalone passes with identical digests. Report: commit
  hash, full test output (printed digest + PASS lines), the pinned artifact `uint64` + byte length,
  confirmation the S1 key is unchanged, confirmation the header includes (S1's 4 + `asset/obj_loader.h`),
  confirmation the artifact digest is IDENTICAL MSVC vs clang (the quantization proof), and the local-clang
  result. Flag any deviation. Commit message via a temp file + `git commit -F` (Bash heredoc). Commit to the
  branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S3 — the key→artifact cache.)
