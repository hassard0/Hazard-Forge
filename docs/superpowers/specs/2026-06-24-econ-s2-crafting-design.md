# Slice ECON-S2 — Crafting / recipe transformer + deterministic craft queue (Flagship #30 ECON, 2nd/6)

S1 built the integer ledger + atomic Add/Remove/Transfer transactions. S2 adds the **crafting** layer: a
`RecipeSet` (inputs → outputs, integer costs/yields) + `ApplyRecipe` that atomically consumes inputs and
produces outputs IFF affordable, plus a deterministic **craft queue** drained in fixed order. This is the
transform core of the gameplay-systems sim — ore + fuel → ingot, conserving by recipe. It stays strict
integer (the moat) and reuses the S1 ledger primitives verbatim.

Pure-CPU INTEGER, append-only to `engine/econ/econ.h` (do NOT modify S1's code — ADD below it). NO render.
Golden = pinned `net::DigestBytes` over the ledger after a craft script, identical Windows/MSVC + Mac/clang
via the standalone clang compile.

## Append to engine/econ/econ.h (below S1, in hf::econ)

1. **`Recipe`** — one crafting transform. Inputs are consumed from a crafting entity, outputs produced to
   it (a recipe transforms an entity's own stock). Fixed-size ingredient lists (flat, deterministic — NO
   map):
   ```cpp
   struct Ingredient { uint32_t item; Qty amount; };   // amount > 0
   struct Recipe {
       std::vector<Ingredient> inputs;    // consumed (all must be affordable)
       std::vector<Ingredient> outputs;   // produced
   };
   struct RecipeSet {
       std::vector<Recipe> recipes;       // indexed by recipe id (the fixed iteration order)
   };
   ```
2. **`bool RecipeAffordable(const World& w, const RecipeSet& rs, uint32_t recipeId, uint32_t entity)`** —
   `recipeId < rs.recipes.size()`, `InRange` for the entity, and the entity has `>= amount` of EVERY
   input item (scan inputs in array order). A recipe that can't pay is a deterministic no-op.
3. **`bool ApplyRecipe(World& w, const RecipeSet& rs, uint32_t recipeId, uint32_t entity)`** — the atomic
   craft. Returns false + NO mutation if out of range or not affordable. Otherwise: consume every input
   (`Set(entity, in.item, At - in.amount)`) in array order, THEN produce every output (`Set(entity,
   out.item, At + out.amount)`) in array order. (Consume-before-produce is pinned; it matters when an item
   is both an input and an output — document the chosen order. Reuse the S1 `Affordable`/`At`/`Set`
   primitives; do NOT duplicate the bookkeeping. NOTE: an output item may exceed nothing in v1 — quantities
   stay int32-bounded by the fixtures.)
   - **Conservation-with-transmutation:** a recipe does NOT preserve `TotalQuantity` in general (2 ore + 1
     fuel → 1 ingot reduces total by 2); the conservation property S2 proves is per-recipe BALANCE (the
     ledger change equals exactly `Σoutputs − Σinputs` for the crafting entity), not global invariance.
4. **`CraftOrder`** — a queued craft request (also a future `net::Session` Input alongside `Command`):
   ```cpp
   struct CraftOrder { uint32_t recipeId; uint32_t entity; uint32_t count; };  // craft `count` times
   ```
5. **`void DrainCraftQueue(World& w, const RecipeSet& rs, const std::vector<CraftOrder>& queue)`** — process
   the queue in ARRAY ORDER; for each order, attempt `ApplyRecipe` up to `count` times (stopping early when
   a craft becomes unaffordable — a partial drain is deterministic). Fixed order throughout (the
   `RunScript` precedent). An unaffordable order crafts as many as it can afford then moves on (deterministic).
6. **Fixtures:** `RecipeSet MakeShowcaseRecipes()` — a fixed recipe set with integer literals, e.g. over the
   S1 item ids: `r0 = {2×item0 + 1×item1} → {1×item2}` (ore+fuel→ingot), `r1 = {3×item2} → {1×item3}`
   (ingots→tool), plus a recipe with an item that is both input and output (to pin the consume/produce
   order). `std::vector<CraftOrder> MakeShowcaseCraftQueue()` — a fixed queue exercising affordable crafts,
   a multi-`count` craft, an unaffordable order (partial/zero drain), and an out-of-range recipeId.

## The golden (PINNED, cross-platform) — append to tests/econ_test.cpp
Build `MakeShowcaseWorld` (seed it with enough raw inputs that some crafts succeed) + `MakeShowcaseRecipes`
+ `DrainCraftQueue(MakeShowcaseCraftQueue())`. Print the live digest, then assert:
```
econ-s2: ledger digest after craft queue = 0x<...>
PASS econ-s2: DigestWorld after DrainCraftQueue == pinned uint64 (the cross-platform proof)
PASS econ-s2: re-running the same queue is bit-identical (deterministic)
PASS econ-s2: an affordable recipe consumes inputs + produces outputs (ledger delta == outputs - inputs)
PASS econ-s2: an unaffordable recipe is a deterministic no-op (digest unchanged, ApplyRecipe returned false)
PASS econ-s2: a multi-count order crafts exactly as many times as affordable, then stops (partial drain deterministic)
PASS econ-s2: an out-of-range recipeId is a no-op (returned false, digest unchanged)
PASS econ-s2: no stock slot is negative after the craft queue (>= 0 over the fixed scan)
```
Assertions:
1. **PINNED DIGEST** — `DigestWorld(w)` after `DrainCraftQueue` == a hard-pinned `uint64_t` (run once, pin;
   identical MSVC + clang).
2. **REPLAY-STABLE** — a second run from a fresh world + same recipes + same queue → identical digest.
3. **CRAFT BALANCE** — for a single `ApplyRecipe` on a fresh world, the per-item ledger delta for the
   crafting entity equals exactly `Σoutputs − Σinputs` (compute the expected deltas from the recipe and
   verify each touched item).
4. **AFFORDABILITY GATE** — `ApplyRecipe` for a recipe the entity can't afford returns false AND leaves the
   digest unchanged.
5. **PARTIAL DRAIN** — a `CraftOrder` with `count` larger than affordable crafts exactly `floor(available
   / cost)` times then stops; the result is deterministic (assert the exact post-state or the exact number
   of successful crafts).
6. **BOUNDS GATE** — an out-of-range `recipeId` (or entity) is a no-op returning false, digest unchanged.
7. **NO NEGATIVE STOCK** — scan the whole ledger after the queue; every slot `>= 0`.

Keep S1's assertions green (append-only — S1's pinned digest `0xaa712207f7663e03` unchanged).

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/econ/econ.h` + `engine/net/session.h` + `tests/econ_test.cpp` (+ `tests/test_main.h`
+ `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/econ_test.cpp -o /tmp/econ && /tmp/econ`, confirming ALL assertions PASS with the IDENTICAL pinned
digests. (A local Windows clang at `C:\Program Files\LLVM\bin\clang++.exe` gives a fast pre-check too.)
NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/econ/econ.h` (add `Ingredient`/`Recipe`/`RecipeSet`/`RecipeAffordable`/`ApplyRecipe`/
  `CraftOrder`/`DrainCraftQueue` + fixtures below S1; do NOT modify S1's types/functions — reuse `At`/`Set`/
  `Affordable`/`InRange`/`DigestWorld`/`TotalQuantity`). Header stays SELF-CONTAINED: only `<cstdint>/
  <cstddef>/<vector>` + `net/session.h`. Do NOT include `fpx.h`/`pcg.h`/any other header. Do NOT modify
  `net/session.h` or any existing header.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<random>`/`<unordered_*>`/`<map>`/`<functional>`/`std::hash`.
  No map/hash-ordered containers in any logic path (fixed recipe + queue iteration order).
- `tests/econ_test.cpp` stays self-contained; APPEND the S2 assertions + fixtures. Keep S1's green.
- Branch `fix-econ-s2`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target econ_test'`
  then run `econ_test` and confirm ALL assertions (S1 + S2) PASS, exit 0. First run: pin the digest.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `econ_test` builds + PASSES on Windows with
  every assertion green, and (if the local clang exists) the standalone clang compile passes. Report: commit
  hash, full test output (printed digest + PASS lines), the pinned digest, confirmation S1's digest
  `0xaa712207f7663e03` is unchanged, confirmation the header is still self-contained (list `#include`s), the
  recipe set + craft queue you chose, and the local-clang result. Commit message via temp file + `git commit
  -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digest, ff-merges to
  master + pushes + deletes the branch + advances to S3 — the resource economy tick.)
