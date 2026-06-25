# Slice ASSET-S5 — Manifest / batch compile, the build as a lockstep replay (Issue #16)

S1–S4 gave the key, artifact, cache, and incremental rebuild. S5 ties them into a **batch compile** that
produces a deterministic **manifest** (the build output: every asset's NodeId → its artifact digest) AND
makes the build **literally a `net::Session` lockstep replay** — the same job stream produces the same
per-tick digest trace and the same final manifest, on any machine, in any submit order. "The build is
reproducible" becomes a pinned golden, the same way `replay.h` made playback a replay.

Pure-integer, append-only to `engine/asset/asset_compiler.h` (below S4; do NOT modify S1–S4 — the pinned
key `0x7fb6a48b4b99f1b7`, artifact `0xf7ee13c169dc0464`, cache `0x029174f13e64c9f1`, rebuild
`0x0808b56e0322d8c1` stay). NO new include (`net/session.h` already present — S5 USES more of it:
`InputRing`/`RunLockstep`/`DigestTrace`).

## Append to engine/asset/asset_compiler.h (below S4, in hf::asset)

### 1. The manifest (the build output — sorted, order-stable)
```cpp
struct ManifestEntry { NodeId node = 0; CacheKey artifactDigest = 0; };
struct Manifest { std::vector<ManifestEntry> entries; };   // SORTED-UNIQUE by node (the order-stable store)

inline void UpsertManifest(Manifest& m, NodeId node, CacheKey artifactDigest);  // sorted-unique insert/replace
inline std::vector<uint8_t> SerializeManifest(const Manifest& m);  // PutU32(count), per entry sorted: PutU32(node) PutU64(digest)
inline CacheKey ManifestDigest(const Manifest& m) { auto b = SerializeManifest(m); return net::DigestBytes(b.data(), b.size()); }
```
(Sorted-by-node + hand-LE → the manifest byte stream is identical regardless of compile order — the
content-addressed property at the build level.)

### 2. The compile job (a value-copyable net::Session Input)
```cpp
// One asset to compile: a stable NodeId + a pointer to its raw bytes (static fixtures in the test) + params.
// Value-copyable (shallow ptr) so net::Session can ring/snapshot it; the bytes outlive the run.
struct CompileJob { NodeId node = 0; const char* bytes = nullptr; std::size_t n = 0; CompileParams params; };
```

### 3. `CompileSet` — the plain batch (order-independent)
```cpp
// Compile every job into `cache` (GetOrCompile) and record node -> DigestArtifact(blob) in a manifest.
// Order-independent: the manifest is sorted by node, so any job ordering yields the same ManifestDigest.
inline Manifest CompileSet(const std::vector<CompileJob>& jobs, AssetCache& cache);
```

### 4. The build as a net::Session lockstep replay (THE HEADLINE)
The build step IS a `net::Session` StepFn — `World = Manifest`, `Input = CompileJob`:
```cpp
// The deterministic build transition: compile each job this tick into `cache`, upsert its artifact digest
// into the manifest. (A free function the test wraps in a lambda capturing the cache — like seq's
// StepPlayhead. Signature matches net::Session: step(World&, const std::vector<Input>&, uint32_t tick).)
inline void StepBuild(AssetCache& cache, Manifest& w, const std::vector<CompileJob>& jobs, uint32_t /*tick*/) {
    for (const CompileJob& j : jobs) {
        CompileResult r = GetOrCompile(cache, j.bytes, j.n, j.params);
        UpsertManifest(w, j.node, DigestArtifact(r.blob));
    }
}
```
The TEST builds the lambdas + an `InputRing<CompileJob>` (jobs spread across ticks) and drives
`net::RunLockstep(Manifest{}, ring, ticks, step, digest)` (final ManifestDigest) and
`net::DigestTrace(...)` (per-tick trace) — `digest = [](const Manifest& m){ return ManifestDigest(m); }`.

### 5. Fixtures (FIXED forever)
- `std::vector<CompileJob> MakeShowcaseJobs()` — 3 jobs: node 0 = `ShowcaseRawBytes()`, node 1 =
  `ShowcaseRawBytesB()`, node 2 = `ShowcaseRawBytesC()` (S3's fixtures), all with `ShowcaseParams()`. Keep
  FIXED. A second helper `MakeShowcaseJobsReordered()` returns the SAME three in a different order (for the
  order-independence golden).
- A fixed tick layout for the lockstep golden: e.g. job 0 on tick 0, job 1 on tick 1, job 2 on tick 2 (3
  ticks) — so the `DigestTrace` shows the manifest growing one entry per tick.

## The golden (PINNED, cross-platform) — append to tests/asset_compiler_test.cpp
```
asset-s5: manifest digest = 0x<...>  (<N> entries)
asset-s5: build trace digest = 0x<...>  (<T> ticks)
PASS asset-s1..s4: ... (all prior assertions STILL green — every prior digest UNCHANGED)
PASS asset-s5: ManifestDigest(CompileSet(showcase)) == pinned uint64 (the build output is byte-stable)
PASS asset-s5: the build is a lockstep replay — RunLockstep final digest == the CompileSet ManifestDigest
PASS asset-s5: DigestTrace digest == pinned uint64 (the per-tick build trace is replayable)
PASS asset-s5: order-independent — CompileSet(jobs) and CompileSet(reordered jobs) -> the SAME manifest digest
PASS asset-s5: two RunLockstep runs over the same ring yield the IDENTICAL final digest (deterministic build)
```
Assertions:
1. **PRIOR INVARIANT** — re-assert S1 `0x7fb6a48b4b99f1b7`, S2 `0xf7ee13c169dc0464`, S3 `0x029174f13e64c9f1`,
   S4 `0x0808b56e0322d8c1`, all UNCHANGED.
2. **PINNED MANIFEST** — `ManifestDigest(CompileSet(MakeShowcaseJobs(), freshCache))` == a hard-pinned
   `uint64_t` (run once, pin THAT; identical MSVC + clang).
3. **BUILD == LOCKSTEP** — `net::RunLockstep(Manifest{}, ring, 3, step, digest)` (the ring from the fixed tick
   layout) == the `CompileSet` ManifestDigest of assertion 2 (the lockstep build reaches the same manifest as
   the plain batch).
4. **PINNED TRACE** — `net::DigestBytes` of `net::DigestTrace(...)` (the `uint64` per-tick vector) == a
   hard-pinned `uint64_t` (the build trace is byte-stable; length == 3 ticks).
5. **ORDER-INDEPENDENT** — `ManifestDigest(CompileSet(MakeShowcaseJobs(), c1))` ==
   `ManifestDigest(CompileSet(MakeShowcaseJobsReordered(), c2))` (different submit order → identical manifest;
   the content-addressed build property).
6. **DETERMINISTIC** — two `RunLockstep` calls over the same ring → identical final digest.

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/asset/asset_compiler.h` + `engine/asset/obj_loader.h` + `engine/net/session.h` +
`tests/asset_compiler_test.cpp` (+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and
runs `clang++ -std=c++20 -I engine -I tests tests/asset_compiler_test.cpp -o /tmp/asset && /tmp/asset`,
confirming ALL assertions PASS with the IDENTICAL pinned digests (manifest + trace). Local Windows clang is
the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/asset/asset_compiler.h` (add `ManifestEntry`/`Manifest`/`UpsertManifest`/
  `SerializeManifest`/`ManifestDigest`/`CompileJob`/`CompileSet`/`StepBuild`/`MakeShowcaseJobs`/
  `MakeShowcaseJobsReordered` below S4). Do NOT modify S1–S4 — all prior digests stay pinned.
- NO new include (`net/session.h` already present). Header stays standalone-clang-compilable. STILL NO
  `<cmath>` (S2's `FxQuantize` is the only float op), NO RNG, NO clock/mtime, NO `<unordered_*>`/`<map>`/
  `std::hash`/`<algorithm>` (sorted vectors + hand binary search). Do NOT modify `net/session.h`.
- mtime NEVER enters the manifest or any digest.
- If the `net::Session` composition proves awkward (e.g. the StepFn capture), FALL BACK to a plain
  deterministic fold producing the same manifest + a hand-built digest trace, and keep assertions 2/4/5/6 —
  but PREFER the real `net::Session` wrap (it is the headline; seq S5 did exactly this).
- `tests/asset_compiler_test.cpp` stays self-contained; APPEND the S5 assertions + the fixtures + the
  lambdas/ring. (Add a `namespace net = hf::net;` alias if convenient, like seq_test — does not change S1–S4.)
  Keep ALL S1–S4 assertions green.
- Branch `fix-asset-s5`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target asset_compiler_test'`
  then run `asset_compiler_test`, confirm ALL assertions (S1–S5) PASS, exit 0. ALSO compile standalone with
  the local clang `C:\Program Files\LLVM\bin\clang++.exe` and confirm IDENTICAL digests. First run: pin the
  manifest + trace digests, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `asset_compiler_test` builds + PASSES on
  Windows with every assertion green (esp. prior invariants + the pinned manifest + build==lockstep + the
  pinned trace + order-independence), and the local clang standalone passes with identical digests. Report:
  commit hash, full test output (printed digests + PASS lines), the pinned manifest + trace `uint64`s,
  confirmation S1–S4 digests unchanged, confirmation includes unchanged, whether you used the real
  `net::Session` wrap or the fallback (and why), and the local-clang result. Flag any deviation. Commit
  message via a temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S6 — the hot-reload watch capstone, extending
  hot_reload.h.)
