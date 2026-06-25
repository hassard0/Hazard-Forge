# Slice ECON-S3 — Resource economy tick: production / consumption flow (Flagship #30 ECON, 3rd/6)

S1 built the ledger + transactions; S2 added crafting. S3 adds the **economy tick** — the per-tick
production/consumption flow that makes the ledger a living economy: producers add stock, consumers drain
it, run in fixed order with zero-floor and integer storage caps. This is the time-stepped core (the
`IntegrateStep` analog for gameplay state) and stays strict integer (the moat) — a determinism property a
float-timer UE5 economy cannot guarantee.

Pure-CPU INTEGER, append-only to `engine/econ/econ.h` (do NOT modify S1/S2 — ADD below). NO render. Golden
= pinned `net::DigestBytes` over the ledger after N ticks, identical Windows/MSVC + Mac/clang via the
standalone clang compile.

## Append to engine/econ/econ.h (below S2, in hf::econ)

1. **Producer / Consumer rules** — per-entity flows, flat + fixed-order (NO map):
   ```cpp
   struct Flow { uint32_t entity; uint32_t item; Qty rate; };   // rate > 0 (units per tick)
   struct EconRules {
       std::vector<Flow> producers;   // each tick: entity gains `rate` of `item` (clamped to cap)
       std::vector<Flow> consumers;   // each tick: entity loses `rate` of `item` (clamped to 0)
       std::vector<Qty>  cap;         // [entityCount*itemCount] per-(entity,item) storage cap; <=0 means "no cap"
   };
   ```
   `cap` is a dense flat array parallel to `World::stock` (same indexing `e*itemCount+t`), OR empty (size 0)
   meaning "no caps anywhere" — handle both. A cap value `<= 0` for a slot means that slot is uncapped.
2. **`Qty CapAt(const EconRules& r, const World& w, uint32_t e, uint32_t t)`** — returns the cap for a slot,
   or a sentinel meaning uncapped (e.g. return `-1` / a `bool` out-param) — pick a clear convention and
   pin it. Out-of-range or empty `cap` → uncapped.
3. **`void EconTick(World& w, const EconRules& r)`** — ONE economy step, in this FIXED order:
   - **Production phase:** for each producer in array order, `newQ = At(e,item) + rate`; if the slot is
     capped, `newQ = min(newQ, cap)`; `Set(e,item, newQ)`. (Clamp UP to the cap — production never exceeds
     storage.) Out-of-range producer entity/item = deterministic skip.
   - **Consumption phase (AFTER all production):** for each consumer in array order, `newQ = max(0, At(e,item)
     − rate)`; `Set(e,item, newQ)`. (Clamp DOWN to zero — never negative.) Out-of-range = skip.
   Production-before-consumption is PINNED (it changes the result when an entity both produces and consumes
   the same item — document the chosen order). Pure integer min/max, no float.
4. **`void RunEconTicks(World& w, const EconRules& r, uint32_t ticks)`** — call `EconTick` `ticks` times
   (fixed loop; deterministic of `(w, r, ticks)` alone — the `RunLockstep`/`StepWorldN` precedent).
5. **Fixtures:** `EconRules MakeShowcaseRules(const World& w)` — fixed producers/consumers/caps over the S1
   showcase world (integer literals): some entities are net producers (rate up to a cap), some net
   consumers (drain to zero), at least one entity that BOTH produces and consumes the same item (to pin the
   phase order), and at least one slot with a real cap (to pin clamping). Make the economy reach a steady
   state within a modest tick count (so the steady-state assertion is meaningful).

## The golden (PINNED, cross-platform) — append to tests/econ_test.cpp
Build `MakeShowcaseWorld` + `MakeShowcaseRules` + `RunEconTicks(w, r, N)` for a fixed N (e.g. 24). Print
the live digest, then assert:
```
econ-s3: ledger digest after N=<N> econ ticks = 0x<...>
PASS econ-s3: DigestWorld after RunEconTicks == pinned uint64 (the cross-platform proof)
PASS econ-s3: re-running N ticks is bit-identical (deterministic)
PASS econ-s3: no stock slot is negative after N ticks (consumers clamp to 0)
PASS econ-s3: no capped slot exceeds its cap after N ticks (producers clamp up)
PASS econ-s3: a pure-producer entity monotonically rises to its cap then holds (deterministic saturation)
PASS econ-s3: a pure-consumer entity monotonically falls to 0 then holds (deterministic depletion)
PASS econ-s3: the economy reaches a STEADY STATE (digest at tick N == digest at tick N+1, i.e. EconTick is idempotent at rest)
```
Assertions:
1. **PINNED DIGEST** — `DigestWorld(w)` after `RunEconTicks(w, r, N)` == a hard-pinned `uint64_t` (run once,
   pin; identical MSVC + clang).
2. **REPLAY-STABLE** — a second run from a fresh world + same rules + N → identical digest.
3. **NON-NEGATIVE** — scan the whole ledger after N ticks; every slot `>= 0` (consumer clamp).
4. **CAP-RESPECTED** — every capped slot is `<= cap` after N ticks (producer clamp).
5. **PRODUCER SATURATION** — a pure-producer slot is non-decreasing over the ticks and ends pinned at its
   cap (or its analytic value if uncapped within N) — assert the monotonic rise + the final value.
6. **CONSUMER DEPLETION** — a pure-consumer slot (with no producer) is non-increasing and reaches 0 — assert
   the monotonic fall + the final 0.
7. **STEADY STATE** — run to tick N (chosen so the economy has settled), then one more `EconTick`; assert
   the digest is UNCHANGED (`EconTick` is idempotent once every flow has hit its cap/floor — the
   steady-state invariant). (If the showcase economy doesn't naturally settle by N, tune rates/caps so it
   does; the steady-state proof is the point.)

Keep S1+S2 assertions green (append-only — S1 `0xaa712207f7663e03`, S2 `0x95147ff9dabbfd13` unchanged).

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/econ/econ.h` + `engine/net/session.h` + `tests/econ_test.cpp` (+ `tests/test_main.h`
+ `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/econ_test.cpp -o /tmp/econ && /tmp/econ`, confirming ALL assertions PASS with the IDENTICAL pinned
digests. (Local Windows clang at `C:\Program Files\LLVM\bin\clang++.exe` is the fast pre-check.) NO Metal,
NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/econ/econ.h` (add `Flow`/`EconRules`/`CapAt`/`EconTick`/`RunEconTicks` + fixtures
  below S2; do NOT modify S1/S2 — reuse `At`/`Set`/`InRange`/`DigestWorld`). Header stays SELF-CONTAINED:
  only `<cstdint>/<cstddef>/<vector>` + `net/session.h`. Do NOT include `fpx.h`/`pcg.h`/any other header. Do
  NOT modify `net/session.h` or any existing header.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/RNG/`<random>`/`<unordered_*>`/`<map>`/`<functional>`/
  `std::hash`. Integer `min`/`max` only (write them inline or use a simple ternary — do NOT pull
  `<algorithm>` if it's avoidable; a ternary is cleanest and clearly deterministic). No map/hash-ordered
  iteration.
- `tests/econ_test.cpp` stays self-contained; APPEND the S3 assertions + fixtures. Keep S1+S2 green.
- Branch `fix-econ-s3`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target econ_test'`
  then run `econ_test` and confirm ALL assertions (S1+S2+S3) PASS, exit 0. First run: pin the digest.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `econ_test` builds + PASSES on Windows with
  every assertion green (esp. steady-state idempotence + the saturation/depletion monotonicity), and the
  local clang standalone compile passes with the identical digest. Report: commit hash, full test output
  (printed digest + PASS lines), the pinned digest + N, confirmation S1/S2 digests unchanged, confirmation
  the header is still self-contained (list `#include`s), the rules fixture you chose, and the local-clang
  result. Commit message via temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digest, ff-merges to
  master + pushes + deletes the branch + advances to S4 — pricing/market + deterministic rolls.)
