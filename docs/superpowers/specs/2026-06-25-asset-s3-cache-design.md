# Slice ASSET-S3 — The content-addressed cache (Issue #16)

S1 shipped the `CacheKey`; S2 the deterministic compiled artifact. S3 is the **cache**: a key→artifact store
with hit/miss lookup, `GetOrCompile` (compile-on-miss), hand-LE serialize/deserialize round-trip, and a
**content-addressed store digest that is independent of insertion order** — the property that makes the
cache a true content address (two machines that compiled the same assets in any order have the byte-identical
cache). Modeled structurally on `engine/world/chunk_diff.h`'s `ChunkDiffStore` (sorted vector + binary-search
insert + hand-LE serialize + `DigestBytes`).

Pure-integer, append-only to `engine/asset/asset_compiler.h` (below S2; do NOT modify S1/S2 — the key
`0x7fb6a48b4b99f1b7` and the artifact digest `0xf7ee13c169dc0464` stay pinned). NO new include (S1's 4 +
`asset/obj_loader.h` from S2 suffice). The store itself is pure integer; only `GetOrCompile`'s
compile-on-miss path reaches S2's `CompileObj` (the one float boundary, already proven exact).

## Append to engine/asset/asset_compiler.h (below S2, in hf::asset)

### 1. Inline LE helpers needed (add if S2 didn't already)
`PutBytes(buf, p, n)` (append raw bytes), `GetU32(p)`, `GetU64(p)` — inline, byte-by-byte LE (mirror
replay.h:46-60; still do NOT include replay.h). If S2 already added any of these, REUSE them — do not
redefine.

### 2. The cache (sorted vector, the ChunkDiffStore mold)
```cpp
struct CacheEntry { CacheKey key = 0; std::vector<uint8_t> blob; };

struct AssetCache {
    std::vector<CacheEntry> entries;   // SORTED-UNIQUE by key (binary-search insert/lookup) — order-stable
};

// Binary-search for `key`; return its blob or nullptr (a pure const lookup, no mutation).
inline const std::vector<uint8_t>* Lookup(const AssetCache& c, CacheKey key);

// Insert (or replace) the blob for `key`, keeping `entries` sorted-unique by key (the InsertSortedUnique
// binary-search pattern from chunk_diff.h:67). A re-insert of an existing key overwrites its blob.
inline void Insert(AssetCache& c, CacheKey key, const std::vector<uint8_t>& blob);
```

### 3. `GetOrCompile` — the cache's reason to exist
```cpp
struct CompileResult { std::vector<uint8_t> blob; bool wasHit = false; };

// Compute the artifact key (kind=Mesh: MakeKey(Mesh, HashRawAsset(bytes,n), HashParams(p))); on a cache HIT
// return the stored blob + wasHit=true; on a MISS, CompileObj(bytes,n,p) -> Insert -> return blob +
// wasHit=false. The hit blob MUST be byte-identical to a cold CompileObj of the same inputs.
inline CompileResult GetOrCompile(AssetCache& c, const void* bytes, std::size_t n, const CompileParams& p);
```
(`wasHit` is a RETURN flag only — it is NEVER serialized and NEVER affects the blob bytes. The blob a hit
returns is bit-equal to the blob a cold compile produces; the test asserts this.)

### 4. Serialize / Deserialize / Digest (hand-LE, order-stable)
```cpp
// Layout: PutU32(entryCount), then per entry IN SORTED-KEY ORDER: PutU64(key), PutU32(blobLen),
// PutBytes(blob). (entries is kept sorted by key, so the byte stream is identical regardless of the order
// the assets were inserted — the content-addressed property.)
inline std::vector<uint8_t> SerializeCache(const AssetCache& c);
inline bool DeserializeCache(const std::vector<uint8_t>& bytes, AssetCache& out);   // false on truncation
inline CacheKey DigestCache(const AssetCache& c) { auto b = SerializeCache(c); return net::DigestBytes(b.data(), b.size()); }
```

### 5. Fixtures (FIXED forever)
- Add 2 MORE fixed raw assets beyond S1's `ShowcaseRawBytes()`, e.g. `ShowcaseRawBytesB()` (a different tiny
  OBJ — a different triangle / quad) and `ShowcaseRawBytesC()` (another). Keep FIXED. (The golden pins a
  cache built from all three.)
- `AssetCache MakeShowcaseCache()` — `GetOrCompile` each of the three fixtures (with `ShowcaseParams()`) into
  a fresh cache, return it. (FIXED — the golden pins its `DigestCache`.)

## The golden (PINNED, cross-platform) — append to tests/asset_compiler_test.cpp
```
asset-s3: cache digest = 0x<...>  (<N> entries)
PASS asset-s1/s2: ... (all prior assertions STILL green — key 0x7fb6a48b4b99f1b7 + artifact 0xf7ee13c169dc0464 UNCHANGED)
PASS asset-s3: DigestCache(MakeShowcaseCache()) == pinned uint64 (content-addressed store, byte-stable)
PASS asset-s3: cold GetOrCompile is a MISS, a second GetOrCompile of the same asset is a HIT
PASS asset-s3: the HIT blob is byte-identical to a cold CompileObj of the same inputs (no wasHit leak)
PASS asset-s3: DeserializeCache(SerializeCache(c)) round-trips — same DigestCache
PASS asset-s3: the cache digest is INDEPENDENT of insertion order (insert A,B,C vs C,B,A -> same digest)
PASS asset-s3: Lookup of an absent key returns nullptr (clean miss)
```
Assertions:
1. **PRIOR INVARIANT** — re-assert S1 key `0x7fb6a48b4b99f1b7` and S2 artifact digest `0xf7ee13c169dc0464`,
   both UNCHANGED.
2. **PINNED CACHE DIGEST** — `DigestCache(MakeShowcaseCache())` == a hard-pinned `uint64_t` (run once, pin
   THAT; identical MSVC + clang).
3. **MISS THEN HIT** — a fresh `AssetCache`; `GetOrCompile(fixtureA)` → `wasHit == false`; a second
   `GetOrCompile(fixtureA)` → `wasHit == true`.
4. **NO WASHIT LEAK** — the HIT's `blob` == `CompileObj(fixtureA, params)` byte-for-byte (the hit returns the
   real artifact; `wasHit` is metadata only).
5. **ROUND-TRIP** — `DeserializeCache(SerializeCache(c), out)` succeeds and `DigestCache(out) ==
   DigestCache(c)`; `out.entries` equals `c.entries` (key + blob per entry).
6. **ORDER-INDEPENDENT** — build cache X by inserting A,B,C and cache Y by inserting C,B,A (same three
   artifacts); `DigestCache(X) == DigestCache(Y)` (the content-addressed property — sorted-by-key store).
7. **CLEAN MISS** — `Lookup(c, someAbsentKey)` returns `nullptr`.

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/asset/asset_compiler.h` + `engine/asset/obj_loader.h` + `engine/net/session.h` +
`tests/asset_compiler_test.cpp` (+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and
runs `clang++ -std=c++20 -I engine -I tests tests/asset_compiler_test.cpp -o /tmp/asset && /tmp/asset`,
confirming ALL assertions PASS with the IDENTICAL pinned digests. Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/asset/asset_compiler.h` (add `CacheEntry`/`AssetCache`/`Lookup`/`Insert`/
  `CompileResult`/`GetOrCompile`/`SerializeCache`/`DeserializeCache`/`DigestCache`/the 2 fixtures +
  `PutBytes`/`GetU32`/`GetU64` if missing). Do NOT modify S1/S2 — the key `0x7fb6a48b4b99f1b7` and artifact
  `0xf7ee13c169dc0464` stay pinned.
- NO new include (S1's 4 + `asset/obj_loader.h`). Header stays standalone-clang-compilable. STILL NO
  `<cmath>` (S2's `FxQuantize` is the only float op), NO RNG, NO clock/mtime, NO `<unordered_*>`/`<map>`/
  `std::hash`/`<algorithm>`. The store is sorted vectors + binary search (hand-written, no `lower_bound`).
- mtime NEVER enters the cache or its digest. `wasHit` NEVER serialized.
- `tests/asset_compiler_test.cpp` stays self-contained; APPEND the S3 assertions + the 2 fixtures. Keep ALL
  S1+S2 assertions green.
- Branch `fix-asset-s3`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target asset_compiler_test'`
  then run `asset_compiler_test`, confirm ALL assertions (S1–S3) PASS, exit 0. ALSO compile standalone with
  the local clang `C:\Program Files\LLVM\bin\clang++.exe` and confirm IDENTICAL digests. First run: pin the
  cache digest, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `asset_compiler_test` builds + PASSES on
  Windows with every assertion green (esp. prior invariants + the pinned cache digest + miss-then-hit +
  no-wasHit-leak + order-independence + round-trip), and the local clang standalone passes with identical
  digests. Report: commit hash, full test output (printed digest + PASS lines), the pinned cache `uint64` +
  entry count, confirmation S1 key + S2 artifact digest unchanged, confirmation the header includes are
  unchanged (S1's 4 + obj_loader.h), how you built the two extra fixtures, and the local-clang result. Flag
  any deviation. Commit message via a temp file + `git commit -F` (Bash heredoc). Commit to the branch and
  STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S4 — the dependency graph + incremental rebuild.)
