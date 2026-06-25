# Slice WFC-S1 — Domain grid + adjacency rule-set + integer AC-3 propagation (Flagship #29 WFC, 1st/6 — beachhead)

The beachhead of flagship #29: **DETERMINISTIC WAVE-FUNCTION-COLLAPSE / constraint-based procedural
generation** — the first CONSTRAINT-SOLVER paradigm in the engine. (Existing `pcg.h` is purely
generative-feed-forward: seeded hash → jittered grid → mask-reject → annotate. There is no constraint
satisfaction, no adjacency rules, no tilemap anywhere.) WFC collapses a grid of superposed tiles to a
globally-consistent layout given an adjacency rule-set. The moat: real WFC is infamous for
non-reproducibility (mt19937 + float Shannon entropy + hash-set iteration → different output per
compiler/platform, un-replayable); HF makes it **bit-identical CPU/Vulkan/Metal + lockstep-replayable
from the seed alone**, which UE5's PCG framework and every WFC plugin cannot do.

S1 establishes the data model + the integer **AC-3 constraint propagation** core (no randomness, no
collapse yet — those are S2/S3). Pure-CPU INTEGER set-logic over `uint64` domain bitmasks. The golden is
a hard-pinned `net::DigestBytes` (FNV-1a-64) over the propagated grid, proven identical on Windows/MSVC +
Mac/clang via a standalone clang compile — NO render-bake, NO image, the cheapest proof loop in the engine.

## NEW file: engine/wfc/wfc.h (namespace hf::wfc)
Header-only and **SELF-CONTAINED**: include ONLY `<cstdint>`, `<cstddef>`, `<vector>` (and `<algorithm>`
if you genuinely need it — match `nav`), plus `#include "net/session.h"` (for `hf::net::DigestBytes`,
which is itself self-contained). NO fpx / RHI / GPU / shader / `<cmath>` / float / clock / RNG / `<random>`
/ `<unordered_*>` symbols. It MUST compile standalone: `clang++ -std=c++20 -I engine -I tests
tests/wfc_test.cpp` on the Mac (like `session_test.cpp` / `replay_test.cpp`).

### Types (all in hf::wfc)
```cpp
using Domain = uint64_t;                          // superposition bitmask: bit t set => tile t still allowed (tileCount <= 64)
enum Dir { kRight = 0, kUp = 1, kLeft = 2, kDown = 3 };   // 4 cardinal dirs, FIXED order

struct TileSet {
    uint32_t             tileCount = 0;           // <= 64
    std::vector<int32_t> weight;                  // [tileCount] integer weights (stored now; used by S2)
    std::vector<Domain>  allowed;                 // [tileCount*4]: allowed[t*4 + dir] = mask of tiles permitted on the `dir` side of tile t
};
inline Domain AllowedMask(const TileSet& ts, uint32_t t, int dir) {
    return ts.allowed[static_cast<std::size_t>(t) * 4u + static_cast<std::size_t>(dir)];
}

struct Grid {
    int32_t             w = 0, h = 0;
    std::vector<Domain> cell;                     // [w*h]: cell[z*w + x] = that cell's current domain
    int32_t cellId(int32_t x, int32_t z) const { return z * w + x; }
};
```
**Adjacency symmetry invariant** (the rule-set MUST satisfy this for AC-3 to be sound — document + the
showcase must obey it): if tile `a` is allowed on the `dir` side of tile `b`, then tile `b` must be
allowed on the OPPOSITE side of tile `a`. Opposite(kRight)=kLeft, Opposite(kUp)=kDown, etc. Provide an
`inline int Opposite(int dir)` helper. (S1 does not need to auto-enforce it, but the showcase tileset
must be built symmetric; optionally add a debug `IsSymmetric(ts)` helper used in the test.)

### Core S1 functions (pure integer, FIXED order — the determinism discipline)
1. **`Domain NeighborConstraint(const TileSet& ts, Domain srcDomain, int dir)`** — the union over every
   tile `t` whose bit is set in `srcDomain` of `AllowedMask(ts, t, dir)`. Iterate set bits in ASCENDING
   order (e.g. loop `t` from 0..tileCount-1, `if (srcDomain >> t) & 1` OR in the mask — or a
   lowest-set-bit pop loop; either is deterministic). This is the mask the neighbor on side `dir` must be
   AND-ed with. (Pure bit ops; no popcount needed here.)
2. **`bool Propagate(const TileSet& ts, Grid& g, std::vector<int32_t>& worklist)`** — the AC-3 worklist
   loop. Process cells from `worklist` in a PINNED order; for the cell popped, for each of its 4 in-bounds
   neighbors (fixed order `{kRight, kUp, kLeft, kDown}`, with the standard offset per dir), compute
   `m = NeighborConstraint(ts, g.cell[c], dir)`, `newDom = g.cell[neighborId] & m`; if `newDom !=
   g.cell[neighborId]`, write it and enqueue the neighbor. Returns `false` immediately if any cell's
   domain becomes `0` (a CONTRADICTION — consumed by S3; S1 just needs the bool). **Determinism rules
   (load-bearing — this is where WFC classically goes non-deterministic):**
   - Pop the LOWEST cell id pending each step — NOT a hash-set, NOT insertion-order-only. Use a
     dedup-by-flag scheme: an in-grid `std::vector<uint8_t> queued(w*h, 0)`; to pick the next cell, either
     keep `worklist` sorted, or scan `queued` ascending for the lowest set flag. Whichever — the pop order
     must be a pure function of WHICH cells are pending, never of insertion order or pointer/hash order.
   - The neighbor direction order is fixed `{kRight, kUp, kLeft, kDown}`.
   - No `std::unordered_*`, no `std::hash`, no pointer-keyed containers anywhere (their iteration order is
     implementation-defined = a real cross-platform bit-exactness threat).
   - The "did a domain shrink" comparison is exact integer `!=`.
   The dir→neighbor offset mapping: kRight=(+1,0), kUp=(0,+1), kLeft=(-1,0), kDown=(0,-1) (or your choice,
   but FIX it and keep it consistent with Opposite()).
3. **`uint64_t DigestGrid(const Grid& g)`** → `return hf::net::DigestBytes(g.cell.data(), g.cell.size() *
   sizeof(Domain));` — the pinned-golden currency, reusing `session.h` verbatim.

### Showcase fixture (deterministic, golden-stable — integer literals, no float)
- `TileSet MakeShowcaseTileSet()` — a small fixed, SYMMETRIC rule-set built from integer literals. Use a
  simple readable set, e.g. a 4-tile terrain band: `0=water, 1=sand, 2=grass, 3=rock`, with the rule that
  water only touches water/sand, sand touches water/sand/grass, grass touches sand/grass/rock, rock
  touches grass/rock (a "no water-next-to-grass" gradient) — applied identically in all 4 directions
  (isotropic → trivially symmetric). Give all tiles weight 1 for S1 (weights exercised in S2). Build the
  `allowed[t*4+dir]` masks explicitly. (Any concrete symmetric set is fine — pick one, keep it FIXED
  forever since the golden pins its propagation.)
- `Grid MakeShowcaseGrid(int32_t w, int32_t h)` — a `w*h` grid with every cell's domain = all-tiles-set
  (`(tileCount < 64) ? ((Domain{1} << tileCount) - 1) : ~Domain{0}`).

## The golden (PINNED, cross-platform) — tests/wfc_test.cpp
Self-contained test in the `session_test.cpp` / `replay_test.cpp` shape (copy the `check()` helper +
`HF_TEST_MAIN_INIT()` from `tests/test_main.h`). Register `hf_add_pure_test(wfc_test)` in
`tests/CMakeLists.txt` next to `session_test`/`replay_test`.
- Build `MakeShowcaseTileSet()` + a **16×16** `MakeShowcaseGrid`. Assert `IsSymmetric(ts)` (the rule-set
  is sound). **Pre-collapse one center cell** (e.g. cellId(8,8)) to a single tile (`g.cell[c] = Domain{1}
  << someTile`), seed the `worklist` with that cell, and `Propagate`.
- Print the live digest, then assert:
```
wfc-s1: propagated grid digest = 0x<...>
PASS wfc-s1: showcase tileset is adjacency-symmetric (AC-3 sound)
PASS wfc-s1: Propagate(center-collapsed) digest == pinned uint64 (the cross-platform proof)
PASS wfc-s1: re-running Propagate from the same start is bit-identical (deterministic)
PASS wfc-s1: flipping one adjacency bit in the tileset changes the digest (rules are load-bearing)
PASS wfc-s1: propagation actually shrank >=1 neighbor domain (the constraint did work, not a no-op)
PASS wfc-s1: no cell domain is empty (no contradiction on the showcase) AND Propagate returned true
```
Assertions:
1. **PINNED DIGEST** — `DigestGrid(g)` after propagation == a hard-pinned `uint64_t` literal. Run once,
   read the printed value, pin THAT (the cross-platform make-or-break: identical on MSVC + clang).
2. **SYMMETRY** — `IsSymmetric(MakeShowcaseTileSet())` is true.
3. **REPLAY-STABLE** — a second independent build+propagate yields the identical digest.
4. **RULES LOAD-BEARING** — clone the tileset, flip one bit in one `allowed[...]` mask, re-propagate from
   the same start → a DIFFERENT digest (proves the rules drive the result, not just the grid shape).
5. **DID WORK** — at least one neighbor domain has fewer bits set after propagation than before (the
   constraint propagated, not a no-op). (Compare popcount before/after on an adjacent cell, or assert some
   cell's domain != all-tiles.)
6. **NO CONTRADICTION** — `Propagate` returned `true` and no `g.cell[i] == 0`.

## Cross-platform proof (the cheap loop — NO render-bake)
The CONTROLLER `scp`s `engine/wfc/wfc.h` + `engine/net/session.h` + `tests/wfc_test.cpp` (+ `tests/test_main.h`
+ `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/wfc_test.cpp -o /tmp/wfc && /tmp/wfc`, confirming the test PASSES with the IDENTICAL pinned digest.
NO Metal, NO `tests/golden/*`, NO `--*-shot`.

## Constraints (HARD)
- NEW header `engine/wfc/wfc.h` (new dir `engine/wfc/`); it must compile STANDALONE under clang with just
  `-I engine -I tests` (self-contained: only `<cstdint>/<cstddef>/<vector>` [+`<algorithm>`] + `net/session.h`).
  Do NOT modify `net/session.h` or any existing header (read-only reuse of `DigestBytes`). Do NOT add it
  to any RHI/GPU target.
- Pure-CPU INTEGER: NO float / `<cmath>` / clock / RNG / `<random>` / `<unordered_*>` / `<functional>` /
  `<fstream>`. No `std::hash`, no pointer-keyed or hash-ordered containers in any logic path.
- `tests/wfc_test.cpp` is SELF-CONTAINED (copy the test scaffolding; do NOT include other tests). Register
  `hf_add_pure_test(wfc_test)` in `tests/CMakeLists.txt`. Use `test_main.h` `HF_TEST_MAIN_INIT()`.
- Branch `fix-wfc-s1`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target wfc_test'`
  (you may need to re-run CMake configure first so the new `wfc_test` target is picked up:
  `cmake --build ... ` will fail if the target is unknown — if so, run the configure step the repo uses,
  e.g. `cmake --preset windows-msvc-release` or `cmake -S . -B build/windows-msvc-release`, then build).
  Run the test exe, confirm it PRINTS the digest and PASSES. First run: pick the pinned digest from the
  printed value, pin it, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until: the header compiles, `wfc_test` builds + PASSES on Windows
  with all assertions green, AND (if a local clang exists) you sanity-compiled the test standalone with
  clang. Report: the commit hash, the full test output (printed digest + PASS lines), the exact pinned
  `uint64_t`, confirmation the header is self-contained (list its `#include`s), the showcase tileset you
  chose (so the controller can eyeball it), and the local-clang result (or that none exists — the
  controller runs the Mac clang proof). Commit message via a temp file + `git commit -F` (Bash heredoc).
  Commit to the branch and STOP.
  (The CONTROLLER audits the diff, runs the Mac/clang standalone to confirm the IDENTICAL digest, then
  ff-merges to master + pushes + deletes the branch + advances to S2.)
