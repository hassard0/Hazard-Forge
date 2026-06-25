# Slice ASSET-S4 — Dependency graph + deterministic incremental rebuild (Issue #16)

S1–S3 gave the key, the compiled artifact, and the content-addressed cache. S4 adds the **incremental
rebuild**: a dependency graph (a scene depends on meshes; a material depends on a texture) and a
**deterministic invalidation** — when one input changes, recompile ONLY that node + its transitive
dependents, in a deterministic topological order, leaving everything else a cache hit. This is the "no
rebuild required" half of the issue: change one mesh, and only the assets that actually depend on it recompile
— identically on every machine.

Pure-integer, append-only to `engine/asset/asset_compiler.h` (below S3; do NOT modify S1–S3 — the pinned
key `0x7fb6a48b4b99f1b7`, artifact `0xf7ee13c169dc0464`, cache `0x029174f13e64c9f1` stay). NO new include.

## The node-identity distinction (important)
S1's `AssetId{kind, contentHash, paramHash}` is the **content-address** — it CHANGES when the asset's bytes
change. A dependency graph needs a **stable logical identity** that PERSISTS across edits (when you edit
"mesh #3", its content hash changes but its graph node — and its set of dependents — does not). So S4
introduces `using NodeId = uint32_t;` — a stable logical asset id (the graph node). The content-address is
what the *cache* keys on (S3); the `NodeId` is what the *graph* keys on (S4).

## Append to engine/asset/asset_compiler.h (below S3, in hf::asset)

### 1. The dependency graph (sorted, order-stable — the ChunkDiffStore discipline)
```cpp
using NodeId = uint32_t;                          // a STABLE logical asset id (persists across edits)

struct DepGraph {
    // nodes[i] = (node, its sorted-unique list of DEPENDENCIES — the nodes it needs). Sorted by node.
    // Edge semantics: `node` DEPENDS ON each id in its dependency list (so a change to a dependency must
    // recompile `node`). Reverse edges (dependents) are derived on demand in InvalidationSet.
    std::vector<std::pair<NodeId, std::vector<NodeId>>> nodes;
};
inline void AddNode(DepGraph& g, NodeId n);                          // sorted-unique insert (no deps yet)
inline void AddDep(DepGraph& g, NodeId from, NodeId to);            // `from` depends on `to`; ensures both nodes exist; sorted-unique
inline const std::vector<NodeId>* Dependencies(const DepGraph& g, NodeId n);  // binary-search; nullptr if absent
```
(All sorted-unique vectors + hand binary search — NO `<unordered_*>`, NO `std::lower_bound`. AddDep adds the
edge AND both endpoints as nodes.)

### 2. InvalidationSet — who must recompile when `changed` changes (reverse reachability)
```cpp
// The set of nodes that must recompile when `changed` changes = `changed` itself + every node that
// transitively DEPENDS ON it (reverse reachability over the dependency edges). Returned SORTED ASCENDING by
// NodeId (deterministic membership). Empty if `changed` is not in the graph.
inline std::vector<NodeId> InvalidationSet(const DepGraph& g, NodeId changed);
```
(Compute by iterating to a fixed point: a node is in the set if it equals `changed` or it depends on any node
already in the set. A bounded loop over `nodes.size()` passes (or a worklist) — deterministic, no recursion
needed; cycles are naturally handled because membership only grows and is bounded by the node count.)

### 3. RebuildOrder — the deterministic topological recompile sequence (the flow.h TopoOrder mold)
```cpp
struct OrderResult { std::vector<NodeId> order; bool ok = true; };  // ok=false iff a cycle blocks a total order
// Kahn topological sort over the SUBGRAPH induced by `subset` (only edges between subset members count):
// repeatedly emit the LOWEST-id node whose in-subset dependencies are all already emitted (the flow.h
// TopoOrder lowest-id ascending-scan). Dependencies come BEFORE dependents (correct recompile order). If no
// node is emittable but the subset is non-empty -> a cycle -> ok=false (NO infinite loop).
inline OrderResult RebuildOrder(const DepGraph& g, const std::vector<NodeId>& subset);
```

### 4. Rebuild — the incremental result
```cpp
struct RebuildResult { std::vector<NodeId> recompiled; CacheKey digest = 0; bool ok = true; };
// Union the InvalidationSet of every changed node, topo-order it (RebuildOrder), and return the ordered
// recompile list + a digest over it (PutU32 each NodeId in order -> DigestBytes). ok=false on a cycle.
// (The COUNT proves incrementality: only changed + dependents are in `recompiled`, NOT the whole graph.)
inline RebuildResult Rebuild(const DepGraph& g, const std::vector<NodeId>& changedNodes);
```
(S4's `Rebuild` is the deterministic *plan* — which nodes, in what order. Wiring it to actually re-run
`GetOrCompile` per node + swap blobs is S6's live path; S4 proves the invalidation logic is bit-deterministic.)

### 5. Fixture (FIXED forever)
- `DepGraph MakeShowcaseGraph()` — a fixed 6-node graph:
  `0` mesh, `1` mesh, `2` scene depends on {0,1}; `3` texture, `4` material depends on {3}, `5` scene2
  depends on {4, 1}. (So node 1 is shared by scene 2 and scene2 5; node 3 only affects 4 and 5.) Keep FIXED —
  the golden pins its invalidation sets + rebuild.

## The golden (PINNED, cross-platform) — append to tests/asset_compiler_test.cpp
```
asset-s4: rebuild(change mesh 0) digest = 0x<...>  recompiled = [...]
PASS asset-s1/s2/s3: ... (all prior assertions STILL green — every prior digest UNCHANGED)
PASS asset-s4: InvalidationSet(change mesh 0) == {0, 2} exactly (mesh + the scene depending on it)
PASS asset-s4: InvalidationSet(change mesh 1) == {1, 2, 5} exactly (shared mesh hits both scenes)
PASS asset-s4: incrementality — recompiling for one mesh change touches 2 nodes, NOT all 6
PASS asset-s4: RebuildOrder is topological — every dependency precedes its dependent (and is deterministic)
PASS asset-s4: Rebuild(change mesh 0) digest == pinned uint64 (the recompile plan is byte-stable)
PASS asset-s4: a cyclic graph -> ok == false (deterministic error, no hang)
PASS asset-s4: changing a leaf texture (3) invalidates {3,4,5}, an unrelated mesh stays a cache hit
```
Assertions:
1. **PRIOR INVARIANT** — re-assert S1 `0x7fb6a48b4b99f1b7`, S2 `0xf7ee13c169dc0464`, S3 `0x029174f13e64c9f1`,
   all UNCHANGED.
2. **INVALIDATION (mesh 0)** — `InvalidationSet(MakeShowcaseGraph(), 0)` == `{0, 2}` exactly.
3. **INVALIDATION (shared mesh 1)** — `InvalidationSet(g, 1)` == `{1, 2, 5}` exactly (the shared dependency
   hits both scenes).
4. **INCREMENTALITY** — `Rebuild(g, {0}).recompiled.size() == 2` (NOT 6 — the whole point: unrelated nodes
   3,4,5 and node 1 are untouched).
5. **TOPOLOGICAL** — in `RebuildOrder(g, InvalidationSet(g, 1))`, every node appears AFTER all its in-subset
   dependencies (e.g. mesh 1 before scene 2 and scene2 5); `ok == true`.
6. **PINNED REBUILD DIGEST** — `Rebuild(g, {0}).digest` == a hard-pinned `uint64_t`.
7. **CYCLE** — a hand-built graph with a cycle (e.g. `AddDep(a,b); AddDep(b,a)`) → `RebuildOrder` (or
   `Rebuild`) returns `ok == false` (a deterministic error, no infinite loop / hang).
8. **LEAF TEXTURE** — `InvalidationSet(g, 3)` == `{3, 4, 5}` (texture → material → scene2); assert an
   unrelated node (mesh 0) is NOT in that set (it stays a cache hit).

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/asset/asset_compiler.h` + `engine/asset/obj_loader.h` + `engine/net/session.h` +
`tests/asset_compiler_test.cpp` (+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and
runs `clang++ -std=c++20 -I engine -I tests tests/asset_compiler_test.cpp -o /tmp/asset && /tmp/asset`,
confirming ALL assertions PASS with the IDENTICAL pinned digests. Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/asset/asset_compiler.h` (add `NodeId`/`DepGraph`/`AddNode`/`AddDep`/`Dependencies`/
  `InvalidationSet`/`OrderResult`/`RebuildOrder`/`RebuildResult`/`Rebuild`/`MakeShowcaseGraph` below S3). Do
  NOT modify S1–S3 — all prior digests stay pinned.
- NO new include. Header stays standalone-clang-compilable. STILL NO `<cmath>` (S2's `FxQuantize` is the only
  float op), NO RNG, NO clock/mtime, NO `<unordered_*>`/`<map>`/`std::hash`/`<algorithm>` (sorted vectors +
  HAND binary search; the topo sort is a hand Kahn lowest-id scan — NO `std::sort`/`lower_bound`).
- mtime NEVER enters the graph or any digest. The `NodeId` is a stable LOGICAL id (not a content hash).
- `tests/asset_compiler_test.cpp` stays self-contained; APPEND the S4 assertions + `MakeShowcaseGraph`. Keep
  ALL S1–S3 assertions green.
- Branch `fix-asset-s4`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target asset_compiler_test'`
  then run `asset_compiler_test`, confirm ALL assertions (S1–S4) PASS, exit 0. ALSO compile standalone with
  the local clang `C:\Program Files\LLVM\bin\clang++.exe` and confirm IDENTICAL digests. First run: pin the
  rebuild digest, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `asset_compiler_test` builds + PASSES on
  Windows with every assertion green (esp. prior invariants + the exact invalidation sets + incrementality
  count + topological order + cycle→ok=false + the pinned rebuild digest), and the local clang standalone
  passes with identical digests. Report: commit hash, full test output (printed digest + PASS lines), the
  pinned rebuild `uint64`, confirmation S1–S3 digests unchanged, confirmation includes unchanged, the
  invalidation sets you observed, and the local-clang result. Flag any deviation. Commit message via a temp
  file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S5 — the manifest / batch compile composing net::Session.)
