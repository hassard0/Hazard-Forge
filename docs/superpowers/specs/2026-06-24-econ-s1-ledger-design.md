# Slice ECON-S1 — Integer ledger + deterministic transactions (Flagship #30 ECON, 1st/6 — beachhead)

The beachhead of flagship #30: **DETERMINISTIC GAMEPLAY-SYSTEMS SIM** — the engine's FIRST gameplay layer.
(All 29 prior flagships are world-simulation, rendering, or proc-gen; none is the deterministic state
machine that makes a *game* out of the physics.) The moat: inventory/crafting/economy/quest state is pure
integer bookkeeping, so it is bit-identical CPU/Vulkan/Metal BY CONSTRUCTION and lockstep/rollback/replay-able
via `net::Session` — exactly what UE5's gameplay glue (Blueprint logic, replicated actor state, float
timers, GAS) structurally cannot do (it's the canonical non-deterministic layer). S1 establishes the core
data model + the atomic transaction primitive (no crafting/economy/quests yet — those are S2–S5).

Pure-CPU INTEGER. The golden is a hard-pinned `net::DigestBytes` (FNV-1a-64) over the ledger, proven
identical on Windows/MSVC + Mac/clang via a standalone clang compile — NO render-bake, NO image, the
cheapest proof loop in the engine.

## NEW file: engine/econ/econ.h (namespace hf::econ)
Header-only and **SELF-CONTAINED**: include ONLY `<cstdint>`, `<cstddef>`, `<vector>`, plus
`#include "net/session.h"` (for `hf::net::DigestBytes`, itself self-contained). NO fpx / RHI / GPU / shader
/ `<cmath>` / float / clock / RNG / `<random>` / `<unordered_*>` / `<map>` / `<functional>` / `std::hash`.
It MUST compile standalone: `clang++ -std=c++20 -I engine -I tests tests/econ_test.cpp` (like
`session_test.cpp` / `wfc_test.cpp`). Determinism = FIXED iteration order (ascending entity id, ascending
item id) — the `wfc.h`/`ai.h` flat-array discipline; NO map/hash-ordered iteration anywhere.

### Types (all in hf::econ)
```cpp
using Qty = int32_t;                 // an integer item quantity (signed for delta math; stock stays >= 0)

// A fixed-size, fixed-order ledger: entity e owns stock[e*itemCount + t] — a dense integer grid
// (the wfc::Grid / ai flat-blackboard discipline; NO map, deterministic iteration by construction).
struct World {
    uint32_t         entityCount = 0;
    uint32_t         itemCount   = 0;
    std::vector<Qty> stock;          // [entityCount * itemCount], row-major: entity major, item minor

    Qty  At(uint32_t e, uint32_t t) const { return stock[(std::size_t)e * itemCount + t]; }
    void Set(uint32_t e, uint32_t t, Qty q) { stock[(std::size_t)e * itemCount + t] = q; }
};

// A transaction Command — ALSO the future net::Session Input (S5). kAdd/kRemove on `dst`/`src`+`item`,
// kTransfer moves `amount` of `item` from `src` to `dst`.
enum Kind : uint32_t { kAdd = 0, kRemove = 1, kTransfer = 2 };
struct Command {
    uint32_t kind;   // Kind
    uint32_t src;    // source entity (kRemove/kTransfer); ignored by kAdd
    uint32_t dst;    // dest entity   (kAdd/kTransfer);   ignored by kRemove
    uint32_t item;   // item id
    Qty      amount; // quantity (must be >= 0; a negative or zero amount is a deterministic no-op)
};
```

### Functions (pure integer, fixed order)
1. **`bool InRange(const World& w, uint32_t e, uint32_t item)`** — `e < entityCount && item < itemCount`
   (bounds guard; an out-of-range command is a deterministic no-op, the `fpx::ApplyCommand` discipline).
2. **`bool Affordable(const World& w, uint32_t e, uint32_t item, Qty amount)`** — `InRange(w,e,item) &&
   w.At(e,item) >= amount` (a Remove/Transfer that can't pay is a deterministic no-op).
3. **`bool ApplyCommand(World& w, const Command& c)`** — the atomic transaction. Returns true iff applied:
   - reject (return false, no mutation) if `c.amount <= 0`, or required ids out of range, or unaffordable.
   - `kAdd`: `w.Set(dst, item, At(dst,item) + amount)` (Add does NOT preserve TotalQuantity — it mints).
   - `kRemove`: requires `Affordable(src,item,amount)`; `w.Set(src,item, At(src,item) - amount)` (burns).
   - `kTransfer`: requires `Affordable(src,item,amount)`; atomically `Remove(src)+Add(dst)` (CONSERVES
     TotalQuantity). If `src==dst` it's a deterministic no-op-but-valid (net zero) — return true, no change.
   (Saturation/overflow: quantities are bounded by the fixtures so int32 never overflows in v1; document
   that v1 assumes bounded quantities — no saturating-add needed, but the Affordable check prevents
   negatives on the Remove side.)
4. **`void RunScript(World& w, const std::vector<Command>& cmds)`** — apply every command in ARRAY ORDER
   (the deterministic input-order contract from `session.h::InputRing` — order is load-bearing).
5. **`uint64_t DigestWorld(const World& w)`** — hash the two counts then the stock array in fixed order via
   `net::DigestBytes`. Suggested: `uint32_t hdr[2] = {entityCount, itemCount}; uint64_t h =
   net::DigestBytes(hdr, sizeof hdr);` then fold the stock — simplest is to digest the contiguous stock
   bytes: combine by hashing stock after the header. Implementer's exact combine is free, but it MUST be a
   pure function of (entityCount, itemCount, stock bytes in order) — pin whatever it computes. (The
   `wfc::DigestGrid` precedent: `net::DigestBytes(stock.data(), stock.size()*sizeof(Qty))` plus the counts.)
6. **`Qty TotalQuantity(const World& w)`** — sum of the entire stock array in fixed order (the conservation
   invariant: kAdd/kRemove change it intentionally, kTransfer preserves it). Use `int64_t` accumulation to
   be safe, return as `Qty` (fixtures keep it in range) OR return `int64_t` — pick and pin.

### Fixtures (deterministic, integer literals — the MakeShowcase* precedent)
- `World MakeShowcaseWorld(uint32_t entityCount, uint32_t itemCount)` — a fixed starting ledger with some
  nonzero stock (integer literals; e.g. a few entities each seeded with varied item counts so transfers/
  removes have something to move). Keep it FIXED forever (the golden pins scripts over it).
- `std::vector<Command> MakeShowcaseScript()` — a fixed deterministic transaction script exercising all
  three kinds (Adds, Removes, Transfers) including at least one unaffordable Remove (to exercise the
  no-op gate) and at least one out-of-range command (to exercise the bounds gate).

## The golden (PINNED, cross-platform) — tests/econ_test.cpp
Self-contained test in the `session_test.cpp`/`wfc_test.cpp` shape (copy the `check()` helper +
`HF_TEST_MAIN_INIT()` from `tests/test_main.h`). Register `hf_add_pure_test(econ_test)` in
`tests/CMakeLists.txt` next to `session_test`/`wfc_test`.
```
econ-s1: ledger digest after showcase script = 0x<...>
PASS econ-s1: DigestWorld after RunScript(showcase) == pinned uint64 (the cross-platform proof)
PASS econ-s1: re-running the same script is bit-identical (deterministic)
PASS econ-s1: a Transfer-only script preserves TotalQuantity (conservation)
PASS econ-s1: flipping one Command.amount changes the digest (order/amounts are load-bearing)
PASS econ-s1: an unaffordable Remove is a no-op (digest unchanged, ApplyCommand returned false)
PASS econ-s1: an out-of-range command is a no-op (digest unchanged, returned false)
PASS econ-s1: no stock slot is negative after the showcase (>= 0 over the fixed scan)
```
Assertions:
1. **PINNED DIGEST** — `DigestWorld(w)` after `RunScript(w, MakeShowcaseScript())` == a hard-pinned
   `uint64_t` (run once, pin the printed value; identical MSVC + clang).
2. **REPLAY-STABLE** — a second independent run from a fresh `MakeShowcaseWorld` + same script → identical
   digest.
3. **CONSERVATION** — build a Transfer-ONLY script over the showcase world; `TotalQuantity` after == before
   (transfers move, never mint/burn).
4. **LOAD-BEARING** — clone the script, change one `Command.amount`, re-run from a fresh world → a DIFFERENT
   digest (amounts drive the result).
5. **AFFORDABILITY GATE** — a `kRemove` exceeding stock: `ApplyCommand` returns false AND the digest is
   unchanged vs before that command.
6. **BOUNDS GATE** — a command with an out-of-range entity/item: `ApplyCommand` returns false AND digest
   unchanged.
7. **NO NEGATIVE STOCK** — scan the whole ledger after the showcase; every slot `>= 0`.

## Cross-platform proof (the cheap loop — NO render-bake)
The CONTROLLER `scp`s `engine/econ/econ.h` + `engine/net/session.h` + `tests/econ_test.cpp`
(+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20
-I engine -I tests tests/econ_test.cpp -o /tmp/econ && /tmp/econ`, confirming the test PASSES with the
IDENTICAL pinned digest. NO Metal, NO `tests/golden/*`, NO `--*-shot`.

## Constraints (HARD)
- NEW header `engine/econ/econ.h` (new dir `engine/econ/`); it must compile STANDALONE under clang with
  just `-I engine -I tests` (self-contained: only `<cstdint>/<cstddef>/<vector>` + `net/session.h`). Do
  NOT modify `net/session.h` or any existing header (read-only reuse of `DigestBytes`). Do NOT add it to
  any RHI/GPU target.
- Pure-CPU INTEGER: NO float / `<cmath>` / clock / RNG / `<random>` / `<unordered_*>` / `<map>` /
  `<functional>` / `std::hash`. No map/hash-ordered containers in any logic path.
- `tests/econ_test.cpp` is SELF-CONTAINED (copy the test scaffolding; do NOT include other tests).
  Register `hf_add_pure_test(econ_test)` in `tests/CMakeLists.txt`. Use `test_main.h` `HF_TEST_MAIN_INIT()`.
- Branch `fix-econ-s1`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target econ_test'`
  (the new target needs a CMake reconfigure first — Ninja usually auto-reconfigures on the first `--build`;
  if the target is unknown, run the repo's configure/preset then build). Run the test exe, confirm it
  PRINTS the digest and PASSES. First run: pick the pinned digest from the printed value, pin it, rebuild,
  confirm green.
- COMPLETION CRITERIA — do NOT commit until: the header compiles, `econ_test` builds + PASSES on Windows
  with all assertions green, AND (if a local clang exists) you sanity-compiled the test standalone with
  clang. Report: the commit hash, the full test output (printed digest + PASS lines), the exact pinned
  `uint64_t`, confirmation the header is self-contained (list its `#include`s), the showcase world/script
  you chose, and the local-clang result (or that none exists — the controller runs the Mac clang proof).
  Commit message via a temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits the diff, runs the Mac/clang standalone to confirm the IDENTICAL digest, then
  ff-merges to master + pushes + deletes the branch + advances to S2 — crafting/recipes.)
