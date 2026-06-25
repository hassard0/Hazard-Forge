# Slice ASSET-S6 — Hot-reload watch + incremental recompile capstone (Issue #16)

S1–S5 built the key, artifact, cache, dependency graph, and the lockstep-replayable batch build. S6 is the
**live capstone**: a watcher detects a changed input and triggers a **deterministic incremental recompile** —
only the changed asset + its transitive dependents recompile (the S4 invalidation), the rest stay cache hits
(S3), and the reloaded artifact is byte-identical to a cold compile. This is the "watch for changes, no
rebuild required" half of the issue, made deterministic.

**DESIGN CHOICE (decoupled + pure):** `engine/runtime/hot_reload.h`'s `FileWatcher` is the live editor's
real-filesystem watcher, but its `.cpp` couples to `ecs.h`/`scene_io.h` (the `ApplyReload` scene path) —
irrelevant to the asset pipeline and breaking the cheap standalone-clang proof. So S6 adds a **self-contained
NodeId-keyed `AssetWatcher`** to `asset_compiler.h` — the SAME mtime-increase logic as `runtime::FileWatcher`
(an injected mtime per asset; a change is a mtime INCREASE) but keyed by the stable `NodeId` (integer, not a
path string), so the whole flagship stays in the one pure header (no `<string>`/`<functional>`, no
ecs/scene coupling, the standalone clang loop intact). **mtime is the TRIGGER, never the key** (the S1
invariant restated): the watcher uses mtime only to decide *what* to recompile; the artifact's identity is
still its content hash.

Pure-integer, append-only to `engine/asset/asset_compiler.h` (below S5; do NOT modify S1–S5 — pinned
`0x7fb6a48b4b99f1b7` / `0xf7ee13c169dc0464` / `0x029174f13e64c9f1` / `0x0808b56e0322d8c1` /
`0x37ca56d9da205682` / `0xcfbe58567fb8fa07` stay). NO new include. (The optional live `--asset-shot` render
is SKIPPED — the asset pipeline's value is determinism, fully proven in the pure test; there is no compelling
visual money-shot. Note this in the report.)

## Append to engine/asset/asset_compiler.h (below S5, in hf::asset)

### 1. The NodeId-keyed watcher (self-contained, the FileWatcher mtime-increase logic)
```cpp
struct AssetWatcher {
    std::vector<std::pair<NodeId, int64_t>> seen;   // last-observed mtime per NodeId, SORTED by NodeId
};
// Record the baseline mtime for `node` (so the next PollChanged does NOT report it as changed). Sorted-unique
// upsert. mtime is an opaque monotonic integer (the injected "filesystem" clock — never the artifact key).
inline void WatchAsset(AssetWatcher& w, NodeId node, int64_t mtime);

// Return the NodeIds whose CURRENT mtime is GREATER than the last seen (or newly present) — i.e. edited —
// SORTED ASCENDING, and UPDATE `seen` to the current values. `current` is the injected mtime table (the test's
// in-memory "filesystem"; sorted or not). A change is a mtime INCREASE (the runtime::FileWatcher semantics).
inline std::vector<NodeId> PollChanged(AssetWatcher& w, const std::vector<std::pair<NodeId, int64_t>>& current);
```

### 2. The hot-reload step (watch → invalidate → recompile dirty → manifest)
```cpp
struct ReloadBatch {
    std::vector<NodeId> recompiled;       // the dirty set actually recompiled, in topo (RebuildOrder) order
    CacheKey            manifestDigest = 0; // the post-reload manifest digest (the whole build's new state)
    bool                ok = true;        // false on a dependency cycle (RebuildOrder failure)
};

// One reload pass: poll the watcher for edits; for the changed NodeIds, compute the S4 InvalidationSet
// (changed + transitive dependents), RebuildOrder it, GetOrCompile each dirty node from its (possibly-edited)
// CompileJob source into `cache` (the dirty ones miss+recompile; everything else stays a hit), then build the
// FULL manifest over ALL sources' current artifact digests. `sources` carries each node's CURRENT bytes
// (the test swaps a source's bytes to simulate an edit). Deterministic of (watcher state, cache, graph,
// sources, current mtimes) alone.
inline ReloadBatch HotReload(AssetWatcher& w, AssetCache& cache, const DepGraph& g,
                             const std::vector<CompileJob>& sources,
                             const std::vector<std::pair<NodeId, int64_t>>& currentMtimes);
```
(Helper: `const CompileJob* SourceForNode(const std::vector<CompileJob>&, NodeId)` — find a source by node;
recompile uses its current `bytes`/`n`/`params`. The full manifest = `UpsertManifest(node,
DigestArtifact(GetOrCompile(cache, src).blob))` over every source, so a cache hit for unchanged nodes still
contributes its (unchanged) digest.)

### 3. Fixtures (FIXED forever)
- `const char* ShowcaseRawBytesEdited()` + length — node 0's EDITED content (a different OBJ from
  `ShowcaseRawBytes()`), used to simulate an edit. Keep FIXED.
- `DepGraph MakeHotReloadGraph()` — a fixed small graph over the 3 source nodes: `0` and `1` meshes, `2`
  depends on `{0, 1}` (so editing mesh 0 invalidates `{0, 2}`). Keep FIXED.
- A fixed baseline mtime layout (e.g. all nodes at mtime 100) and an edited layout (node 0 → 101). Keep FIXED.

## The golden (PINNED, cross-platform) — append to tests/asset_compiler_test.cpp
```
asset-s6: post-reload manifest digest = 0x<...>
PASS asset-s1..s5: ... (all prior assertions STILL green — every prior digest UNCHANGED)
PASS asset-s6: a no-edit PollChanged after WatchAsset reports nothing (no spurious reload)
PASS asset-s6: editing mesh 0 (bump mtime + swap bytes) recompiles exactly {0, 2} in topo order
PASS asset-s6: reload == cold compile — node 0's reloaded artifact digest == DigestArtifact(CompileObj(edited))
PASS asset-s6: the unchanged mesh 1 is NOT recompiled (it stays a cache hit)
PASS asset-s6: the post-reload manifest digest == pinned uint64 (and differs from the pre-edit manifest)
PASS asset-s6: HotReload is deterministic — two reloads from the same edited state are identical
```
Assertions:
1. **PRIOR INVARIANT** — re-assert all of S1 `0x7fb6a48b4b99f1b7`, S2 `0xf7ee13c169dc0464`, S3
   `0x029174f13e64c9f1`, S4 `0x0808b56e0322d8c1`, S5 manifest `0x37ca56d9da205682` + trace
   `0xcfbe58567fb8fa07`, all UNCHANGED.
2. **NO SPURIOUS RELOAD** — `WatchAsset` all 3 nodes at the baseline mtime; `PollChanged` with the SAME
   mtimes → empty (a watcher just-baselined reports no change).
3. **DIRTY SET** — swap node 0's source bytes to `ShowcaseRawBytesEdited()` and bump its mtime;
   `HotReload(...)` → `recompiled == {0, 2}` in topo order (mesh 0 before its dependent 2); `ok == true`.
4. **RELOAD == COLD** — after the reload, the cache's artifact for node 0 (its `DigestArtifact`) ==
   `DigestArtifact(CompileObj(ShowcaseRawBytesEdited(), len, params))` (a from-scratch compile of the edited
   bytes — the reload produced the byte-identical artifact a clean build would).
5. **UNCHANGED HIT** — node 1 is NOT in `recompiled` (its bytes/mtime did not change → cache hit).
6. **PINNED POST-RELOAD MANIFEST** — `HotReload(...).manifestDigest` == a hard-pinned `uint64_t`, and it
   DIFFERS from the pre-edit manifest digest (the edit changed the build output).
7. **DETERMINISTIC** — running `HotReload` twice from the same edited state (fresh watcher/cache each) →
   identical `manifestDigest` + `recompiled`.

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/asset/asset_compiler.h` + `engine/asset/obj_loader.h` + `engine/net/session.h` +
`tests/asset_compiler_test.cpp` (+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and
runs `clang++ -std=c++20 -I engine -I tests tests/asset_compiler_test.cpp -o /tmp/asset && /tmp/asset`,
confirming ALL assertions PASS with the IDENTICAL pinned digests. Local Windows clang is the fast pre-check.
(No `hot_reload.cpp`, no ecs/scene — the watcher is self-contained in `asset_compiler.h`.)

## Constraints (HARD)
- APPEND-ONLY to `engine/asset/asset_compiler.h` (add `AssetWatcher`/`WatchAsset`/`PollChanged`/`ReloadBatch`/
  `SourceForNode`/`HotReload`/`ShowcaseRawBytesEdited`/`MakeHotReloadGraph` below S5). Do NOT modify S1–S5 —
  all prior digests stay pinned.
- NO new include. Header stays standalone-clang-compilable (NO `runtime/hot_reload.h`, NO `<string>`,
  NO `<functional>`). STILL NO `<cmath>` (S2's `FxQuantize` is the only float op), NO RNG, NO clock, NO
  `<unordered_*>`/`<map>`/`std::hash`/`<algorithm>` (sorted vectors + hand binary search). The watcher is
  NodeId-keyed with integer mtimes.
- **mtime is the TRIGGER only — it NEVER enters a key, an artifact, or any pinned digest.** The pinned
  manifest digest is over `{NodeId, artifactDigest}` (no mtime). The `currentMtimes` table is injected (the
  in-memory "filesystem") — NO real `std::filesystem`, NO clock.
- `tests/asset_compiler_test.cpp` stays self-contained; APPEND the S6 assertions + the fixtures. Keep ALL
  S1–S5 assertions green.
- Branch `fix-asset-s6`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`. (The optional
  `--asset-shot` render is SKIPPED — report that it was intentionally skipped.)
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target asset_compiler_test'`
  then run `asset_compiler_test`, confirm ALL assertions (S1–S6) PASS, exit 0. ALSO compile standalone with
  the local clang `C:\Program Files\LLVM\bin\clang++.exe` and confirm IDENTICAL digests. First run: pin the
  post-reload manifest digest, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `asset_compiler_test` builds + PASSES on
  Windows with every assertion green (esp. prior invariants + the dirty set {0,2} + reload==cold + the pinned
  post-reload manifest), and the local clang standalone passes with identical digests. Report: commit hash,
  full test output (printed digest + PASS lines), the pinned post-reload manifest `uint64`, confirmation
  S1–S5 digests unchanged, confirmation includes unchanged (no `<string>`/`<functional>`/`runtime/hot_reload.h`),
  the dirty set observed, that the optional `--asset-shot` was skipped, and the local-clang result. Flag any
  deviation. Commit message via a temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch, then writes the ARCHITECTURE.md asset-pipeline section + comments
  issue #16 — COMPLETING flagship #16.)
