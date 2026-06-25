# Slice PROFILE-S2 — Hierarchical scope/zone tree (Issue #31)

S1 shipped the capture event stream + interned names + the structural digest that excludes timing (digest
`0xedc7791443141dfd`). S2 reconstructs the **scope/zone TREE** from the flat Enter/Exit event stream — the
hierarchical "track" view a profiler UI shows (a frame contains a Shadow pass containing draws, etc.) — and
aggregates draw counts bottom-up. The tree is a pure-integer deterministic structure with its own pinned
digest; an unbalanced Enter/Exit stream resolves to a deterministic canonical tree (never UB).

Pure-integer, append-only to `engine/profile/profile.h` (below S1; do NOT modify S1 — the digest
`0xedc7791443141dfd` stays pinned). NO new include.

## Append to engine/profile/profile.h (below S1, in hf::profile)

### 1. The scope tree (indices, never pointers — the chunk_diff.h discipline)
```cpp
struct ScopeNode {
    uint32_t nameId         = 0;   // the interned scope name (the synthetic ROOT uses a reserved sentinel)
    uint32_t parent         = 0;   // index into nodes[]; the root is its own parent (index 0)
    uint32_t firstChild     = 0;   // index of the first child, or kNoNode
    uint32_t nextSibling    = 0;   // index of the next sibling, or kNoNode
    uint32_t selfDrawCount  = 0;   // draws emitted DIRECTLY in this scope (not in children)
    uint32_t subtreeDrawCount = 0; // selfDrawCount + sum of all descendants' subtreeDrawCount
};
constexpr uint32_t kNoNode = 0xFFFFFFFFu;     // the "no node" sentinel for firstChild/nextSibling
constexpr uint32_t kRootName = 0xFFFFFFFFu;   // the synthetic root's reserved nameId

struct ScopeTree {
    std::vector<ScopeNode> nodes;   // nodes[0] is the synthetic ROOT (parent of all top-level scopes)
    bool                   balanced = true;  // false if the event stream had unbalanced Enter/Exit
};
```

### 2. `BuildScopeTree` — one integer-depth-stack pass (NO recursion, NO `<stack>`)
```cpp
inline ScopeTree BuildScopeTree(const Capture& c);
```
- Start with `nodes = { ScopeNode{ kRootName, 0, kNoNode, kNoNode, 0, 0 } }` (the root at index 0; its
  parent is itself). Maintain an integer **open-scope stack** as a `std::vector<uint32_t>` of node indices,
  initialized with `{0}` (the root is always open).
- Walk `c.events` in order:
  - **`ScopeEnter`**: create a new `ScopeNode{ ev.nameId, parent = stack.back(), firstChild = kNoNode,
    nextSibling = kNoNode, 0, 0 }`, append it; **link it as a child of `parent`** — push it to the FRONT or
    BACK of the parent's child list deterministically. **Use BACK (append to the sibling chain)** so children
    appear in emission order: walk `parent.firstChild`'s sibling chain to the end and set the last's
    `nextSibling = newIndex`, or set `parent.firstChild = newIndex` if empty. Then `stack.push_back(newIndex)`.
  - **`ScopeExit`**: if `stack.size() > 1` pop it (close the current scope); if `stack.size() == 1` (an exit
    with no matching enter) → set `balanced = false` and IGNORE the stray exit (deterministic).
  - **`DrawCall`**: `nodes[stack.back()].selfDrawCount += ev.a` (the draw count goes to the currently-open
    scope; draws at the root level land on the root).
  - (`FrameBegin`/`FrameEnd` are ignored by S2 — S3 handles frames.)
- **End-of-stream auto-close:** if `stack.size() > 1` after the walk (open scopes never exited) → set
  `balanced = false` (the open scopes simply remain as built — they are valid nodes; the canonical tree just
  reflects the truncation). This makes an unbalanced stream deterministic, never UB.
- **Aggregate `subtreeDrawCount` bottom-up:** because a child always has a HIGHER index than its parent (a
  child is created after its parent is pushed), iterate `nodes` from the LAST index down to 0:
  `nodes[i].subtreeDrawCount += nodes[i].selfDrawCount;` then `if (i != 0) nodes[parent].subtreeDrawCount +=
  nodes[i].subtreeDrawCount;`. (A single reverse pass — integer-exact, deterministic, no recursion.)

### 3. `DigestTree` — hand-LE over the nodes in pre-order (deterministic traversal)
```cpp
inline uint64_t DigestTree(const ScopeTree& t);
```
- Produce a deterministic **pre-order** node sequence (root, then each child subtree in `firstChild`→
  `nextSibling` order) using an explicit integer work-stack (NO recursion). For each visited node, append
  hand-LE: `PutU32(nameId)`, `PutU32(selfDrawCount)`, `PutU32(subtreeDrawCount)`, `PutU32(childCount)` (count
  the sibling chain) — a shape+counts encoding. Also `PutU32((uint32_t)balanced)` once at the front. Then
  `net::DigestBytes(buf...)`. (Pre-order + childCount makes the tree shape unambiguous and digest-stable.)
  (Do NOT serialize parent/firstChild/nextSibling indices directly — pre-order + childCount captures the
  shape without embedding array indices, so the digest is layout-stable.)

### 4. Fixture (reuse S1's) + a nested showcase
- Reuse `MakeShowcaseCapture()` from S1 (Frame{ Shadow{Draw 2} Lit{Draw 5} }). Its tree: root → Frame →
  {Shadow(self 2), Lit(self 5)}; `Frame.subtreeDrawCount == 7`, root subtree == 7.
- (Optional) a deeper `MakeNestedCapture()` if a richer tree helps the golden — but the S1 showcase already
  nests, so reusing it keeps the fixtures minimal. Use it.

## The golden (PINNED, cross-platform) — append to tests/profile_test.cpp
```
profile-s2: scope-tree digest = 0x<...>  (<N> nodes)
PASS profile-s1: ... (all 6 S1 assertions STILL green — structural digest 0xedc7791443141dfd UNCHANGED)
PASS profile-s2: BuildScopeTree(showcase) digest == pinned uint64 (the zone tree is byte-stable cross-platform)
PASS profile-s2: the tree is balanced and has the expected shape (root -> Frame -> {Shadow, Lit})
PASS profile-s2: subtree draw aggregation is exact — root subtreeDrawCount == total draws (7)
PASS profile-s2: a deeper-nested scope changes the tree digest (hierarchy is load-bearing)
PASS profile-s2: an unbalanced stream (a missing Exit) -> balanced==false + a deterministic canonical tree
PASS profile-s2: TIMING STILL EXCLUDED — filling timings nonzero leaves BOTH the structural AND tree digest unchanged
```
Assertions:
1. **S1 INVARIANT** — re-assert `StructuralDigest(MakeShowcaseCapture()) == 0xedc7791443141dfd` UNCHANGED.
2. **PINNED TREE DIGEST** — `DigestTree(BuildScopeTree(MakeShowcaseCapture()))` == a hard-pinned `uint64_t`
   (run once, pin THAT; identical MSVC + clang).
3. **SHAPE** — the built tree is `balanced == true`; the root has exactly one child (Frame); Frame has
   exactly two children (Shadow, Lit) in emission order; Shadow.selfDrawCount == 2, Lit.selfDrawCount == 5.
4. **AGGREGATION EXACT** — `nodes[root].subtreeDrawCount == 7` and `Frame.subtreeDrawCount == 7` (2 + 5).
5. **HIERARCHY LOAD-BEARING** — build a capture where a draw is nested one level DEEPER (e.g.
   `Lit{ Cull{ Draw 5 } }` instead of `Lit{ Draw 5 }`) → a DIFFERENT tree digest (the shape matters, even
   though the total draw count is identical).
6. **UNBALANCED DETERMINISTIC** — a stream with a missing Exit (e.g. `Enter A; Draw 1;` with no `Exit A`) →
   `balanced == false` AND `DigestTree` returns a stable pinned value (deterministic canonical tree, no hang
   / no UB). Assert the digest equals a second build of the same unbalanced stream.
7. **TIMING STILL EXCLUDED** — fill `timings` with arbitrary nonzero; assert BOTH `StructuralDigest` AND
   `DigestTree` are UNCHANGED (the tree is built from `events`, never `timings`).

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/profile/profile.h` + `engine/net/session.h` + `tests/profile_test.cpp` (+
`tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine
-I tests tests/profile_test.cpp -o /tmp/profile && /tmp/profile`, confirming ALL assertions PASS with the
IDENTICAL pinned digests. Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/profile/profile.h` (add `ScopeNode`/`kNoNode`/`kRootName`/`ScopeTree`/
  `BuildScopeTree`/`DigestTree` below S1). Do NOT modify S1 — `0xedc7791443141dfd` stays pinned.
- NO new include. Header stays self-contained (4 includes: `<cstddef>/<cstdint>/<vector>` + `net/session.h`).
  STILL NO `<string>`, NO `<cmath>`, NO clock/RNG, NO `<unordered_*>`/`<map>`/`std::hash`/`<algorithm>`,
  NO recursion (use an explicit integer work-stack `std::vector<uint32_t>`).
- The tree is built from `events` ONLY — `timings` is NEVER read (re-assert: filling timings nonzero leaves
  both digests unchanged).
- `tests/profile_test.cpp` stays self-contained; APPEND the S2 assertions. Keep ALL 6 S1 assertions green.
- Branch `fix-profile-s2`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target profile_test'`
  (if the `cmd /c '...&&...'` handoff swallows output, run the build via the PowerShell tool natively). Run
  `profile_test`, confirm ALL assertions (S1 + S2) PASS, exit 0. ALSO compile standalone with the local clang
  `C:\Program Files\LLVM\bin\clang++.exe` and confirm the IDENTICAL digests. First run: pin the tree digest
  (+ the unbalanced-tree digest), rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `profile_test` builds + PASSES on Windows
  with every assertion green (esp. S1 digest unchanged + the pinned tree digest + aggregation == 7 + the
  unbalanced-deterministic case + timing-still-excluded), and the local clang standalone passes with
  identical digests. Report: commit hash, full test output (printed digests + PASS lines), the pinned tree
  `uint64` (+ the unbalanced digest), confirmation the S1 digest is unchanged, confirmation the header is
  self-contained (4 includes, no recursion), the tree shape you built, and the local-clang result. Flag any
  deviation. Commit message via a temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S3 — frame boundaries + the multi-frame timeline.)
