# Slice FLOW-S1 — Graph data model + canonical topological order + integer eval (Issue #24, beachhead)

The beachhead of the deterministic VISUAL-SCRIPTING RUNTIME (issue #24, Blueprint-class). The VISUAL
EDITOR is OUT OF SCOPE — we build the deterministic node-graph EXECUTION VM the editor would drive. The
moat: UE5 Blueprints are *the* canonical non-deterministic UE5 subsystem (event order depends on actor
registration / tick groups, float math, `TMap` iteration, per-frame timing → two machines running the
same graph on the same inputs routinely diverge; UE5's own deterministic-rollback path *excludes*
Blueprints). A node graph whose evaluation is **bit-identical across MSVC/clang/Vulkan/Metal and
lockstep/rollback/replay-able** is a capability UE5 structurally lacks.

S1 establishes the two pieces every later slice builds on: the graph data model + the **canonical
topological scheduler** (the central determinism risk — a DAG has many valid topo orders; we pin ONE).
Pure-CPU INTEGER. The golden is a hard-pinned `net::DigestBytes` over the evaluated register file, proven
identical Windows/MSVC + Mac/clang via a standalone clang compile — NO render-bake, the cheapest proof.

## NEW file: engine/flow/flow.h (namespace hf::flow)
Header-only and **SELF-CONTAINED**: include ONLY `<cstddef>`, `<cstdint>`, `<vector>`, plus
`#include "net/session.h"` (for `hf::net::DigestBytes`). NO fpx / RHI / GPU / shader / `<cmath>` / float /
clock / RNG / `<random>` / `<unordered_*>` / `<map>` / `<functional>` / `std::hash`. It MUST compile
standalone: `clang++ -std=c++20 -I engine -I tests tests/flow_test.cpp` (like `econ_test.cpp`/`wfc_test.cpp`).
Determinism = FIXED iteration order (the wfc.h/econ.h discipline); NO map/hash-ordered iteration anywhere.

### Types (all in hf::flow)
```cpp
using NodeId = uint32_t;   // a node's id == its index into Graph::nodes (pinned, monotonic, never recycled)
using Reg    = int32_t;    // a node's output value — INTEGER ONLY (raw int or Q16.16; no float ever)

// FIXED enum numbering = the wire contract (serialized later; never renumber).
enum Kind : uint32_t { kConst = 0, kAdd = 1, kSub = 2, kMul = 3, kMin = 4, kMax = 5, kSelect = 6 };

struct Node {
    uint32_t kind = kConst;          // Kind
    NodeId   a = 0, b = 0, c = 0;    // input node ids; kSelect uses c as the predicate (c!=0 -> a else b).
    Reg      constArg = 0;           // kConst payload (ignored by other kinds)
};

struct Graph {
    std::vector<Node> nodes;         // nodes[i] has NodeId i. node 0 is a valid node; "no input" is encoded
                                     // by an input id that refers to a node NOT in this node's dependency set
                                     // -> reads 0 (see Evaluate). (Document the exact "unused input" convention.)
};
```
(Convention note: a node lists `a`/`b`/`c` as the NodeIds it reads. For kinds with fewer inputs (kConst = 0
inputs, kAdd/kSub/kMul/kMin/kMax = 2, kSelect = 3) the unused fields should reference the node's OWN id or a
sentinel that `Evaluate` treats as "no edge" — pick a clear convention: e.g. an input id `>= nodes.size()`
OR equal to the node's own id means "no edge / reads 0". Pin whichever; the showcase + test use it
consistently. The key invariant: a self-reference or an out-of-range id is NOT a real dependency edge for
topo-sort, and reads 0 in Evaluate.)

### Functions (pure integer, FIXED order — the determinism discipline)
1. **`bool TopoOrder(const Graph& g, std::vector<NodeId>& outOrder)`** — Kahn's algorithm producing the ONE
   CANONICAL order:
   - Compute `indeg[]` (in-degree = count of this node's REAL input edges — only edges to in-range,
     non-self NodeIds count). Build the adjacency the deterministic way (a flat `std::vector`, NO
     `unordered_*`).
   - Repeatedly: among all nodes with `indeg == 0` that are not yet emitted, pick the **LOWEST NodeId** via
     an ASCENDING SCAN over a flat `bool ready[]`/`emitted[]` vector (the `wfc::Propagate` "pop the lowest
     pending id" idiom — NEVER insertion/hash order). Emit it; decrement the in-degree of its dependents.
   - Returns `true` with `outOrder` = the full topo order on success; returns `false` on a CYCLE (some node
     never reaches indeg 0) — a DETERMINISTIC rejection (`outOrder` cleared/partial, NEVER UB, never a hang
     — bound the loop by `nodes.size()`).
   This lowest-id-first tie-break + pinned monotonic NodeIds = one canonical order on every platform.
2. **`std::vector<Reg> Evaluate(const Graph& g)`** — `TopoOrder`, then walk the order computing each node's
   `Reg` from its already-evaluated inputs into a register file `regs` (one `Reg` per node, indexed by
   NodeId). An input id that is "no edge" (out-of-range / self per the convention) or somehow not-yet-
   evaluated reads `0` (the econ `ApplyCommand` deterministic-no-op gate). Per kind: kConst→`constArg`;
   kAdd→`regs[a]+regs[b]`; kSub→`regs[a]-regs[b]`; kMul→`regs[a]*regs[b]` (int32 wrap is deterministic — v1
   assumes bounded values; document); kMin/kMax→integer min/max via ternary (NO `<algorithm>`);
   kSelect→`regs[c] != 0 ? regs[a] : regs[b]`. If `TopoOrder` returns false (cycle), return an all-zero
   register file of size `nodes.size()` (deterministic, no UB). Pure integer.
3. **`uint64_t DigestGraph(const std::vector<Reg>& regs)`** → `return hf::net::DigestBytes(regs.data(),
   regs.size() * sizeof(Reg));` — the pinned-golden currency (regs is a contiguous `int32_t` span, byte-stable).

### Fixtures (deterministic, integer literals — the MakeShowcase* precedent)
- `Graph MakeShowcaseGraph()` — a fixed ~10-node arithmetic+select DAG (consts feeding add/sub/mul/min/max
  and at least one kSelect whose predicate routes between two subtrees) — built so the topo order is
  NON-trivial (the nodes are NOT already in topo order in the array, to exercise the scheduler). Keep it
  FIXED forever (the golden pins its evaluation).
- `Graph MakeCyclicGraph()` — a tiny graph with a real cycle (a→b→a) for the cycle-rejection test.
- `Graph Permuted(const Graph& g)` — returns `g` with its node array PERMUTED and all NodeId references
  remapped consistently (a different array order encoding the SAME logical graph). Used to prove the topo
  order is canonical (the permuted graph evaluates to the SAME digest). (A simple fixed permutation, e.g.
  reverse the array + remap, is enough.)

## The golden (PINNED, cross-platform) — tests/flow_test.cpp
Self-contained test in the `econ_test.cpp` shape (copy the `check()` helper + `HF_TEST_MAIN_INIT()` from
`tests/test_main.h`). Register `hf_add_pure_test(flow_test)` in `tests/CMakeLists.txt` next to `econ_test`.
```
flow-s1: showcase eval digest = 0x<...>
PASS flow-s1: DigestGraph(Evaluate(showcase)) == pinned uint64 (the cross-platform proof)
PASS flow-s1: re-evaluating the same graph is bit-identical (deterministic)
PASS flow-s1: changing one node's constArg changes the digest (inputs are load-bearing)
PASS flow-s1: a cyclic graph is a deterministic rejection (TopoOrder false, Evaluate all-zero, no UB/hang)
PASS flow-s1: topo order is CANONICAL — a permuted-but-equivalent graph evaluates to the SAME digest
PASS flow-s1: kSelect routes on its predicate (a hand-checked node value is correct)
```
Assertions:
1. **PINNED DIGEST** — `DigestGraph(Evaluate(MakeShowcaseGraph()))` == a hard-pinned `uint64_t` (run once,
   read the printed value, pin THAT; identical MSVC + clang — the make-or-break cross-platform anchor).
2. **REPLAY-STABLE** — a second `Evaluate` of the same graph → identical digest.
3. **LOAD-BEARING** — clone the showcase, change one node's `constArg`, re-evaluate → a DIFFERENT digest.
4. **CYCLE REJECTION** — `TopoOrder(MakeCyclicGraph(), order)` returns `false`; `Evaluate` of it returns an
   all-zero register file (size == node count); no hang/UB.
5. **CANONICAL ORDER** — `DigestGraph(Evaluate(Permuted(g)))` == `DigestGraph(Evaluate(g))` (the topo order
   is canonical, independent of array layout — the central determinism proof).
6. **SELECT CORRECTNESS** — a hand-checked `kSelect` node in the showcase evaluates to the expected branch
   (verify a specific `regs[i]` value).

## Cross-platform proof (the cheap loop — NO render-bake)
The CONTROLLER `scp`s `engine/flow/flow.h` + `engine/net/session.h` + `tests/flow_test.cpp`
(+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20
-I engine -I tests tests/flow_test.cpp -o /tmp/flow && /tmp/flow`, confirming the test PASSES with the
IDENTICAL pinned digest. (A local Windows clang at `C:\Program Files\LLVM\bin\clang++.exe` is the fast
pre-check.) NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- NEW header `engine/flow/flow.h` (new dir `engine/flow/`); compiles STANDALONE under clang with just
  `-I engine -I tests` (self-contained: only `<cstddef>/<cstdint>/<vector>` + `net/session.h`). Do NOT
  modify `net/session.h` or any existing header. Do NOT add it to any RHI/GPU target.
- Pure-CPU INTEGER: NO float / `<cmath>` / clock / RNG / `<random>` / `<unordered_*>` / `<map>` /
  `<functional>` / `std::hash` / `<algorithm>` (integer min/max via ternary). No map/hash-ordered
  containers in any logic path. The topo tie-break is lowest-id ascending-scan — NEVER insertion/hash order.
- `tests/flow_test.cpp` is SELF-CONTAINED (copy the scaffolding; do NOT include other tests). Register
  `hf_add_pure_test(flow_test)` in `tests/CMakeLists.txt`. Use `test_main.h` `HF_TEST_MAIN_INIT()`.
- Branch `fix-flow-s1`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target flow_test'`
  Run the test exe, confirm it PRINTS the digest and PASSES. First run: pick the pinned digest from the
  printed value, pin it, rebuild, confirm green. ALSO compile standalone with the local clang and confirm
  the IDENTICAL digest (MSVC==clang).
- COMPLETION CRITERIA — do NOT commit until: the header compiles, `flow_test` builds + PASSES on Windows
  with all assertions green (esp. the canonical-order proof + the cycle rejection), and the local clang
  standalone passes with the identical digest. Report: the commit hash, the full test output (printed
  digest + PASS lines), the exact pinned `uint64_t`, confirmation the header is self-contained (list its
  `#include`s), the showcase graph you built + the "unused input" convention you chose, and the local-clang
  result. Commit message via a temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits the diff, runs the Mac/clang standalone to confirm the IDENTICAL digest, then
  ff-merges to master + pushes + deletes the branch + advances to S2 — stateful nodes + the per-tick step.)
