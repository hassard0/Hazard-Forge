# Slice ECON-S4 — Pricing / market clearing + deterministic rolls (Flagship #30 ECON, 4th/6 — the fractional slice)

S1–S3 built the ledger, crafting, and the economy tick. S4 adds the **market**: integer supply/demand
pricing, deterministic order clearing (trades at the current price, paid in a currency item), and
**deterministic rolls** (seeded loot/yield via a copied `PcgHash`). This is the arc's one "fractional"
slice — but we keep it STRICT INTEGER (integer-ratio elasticity, NO `fxmul`/`fxdiv`, NO `fpx.h`), so
`econ.h` stays self-contained and the cheap clang-hash proof holds. A deterministic, reproducible market +
loot system is exactly what a float-RNG UE5 economy cannot lockstep/replay.

Pure-CPU INTEGER, append-only to `engine/econ/econ.h` (do NOT modify S1–S3 — ADD below). NO render. Golden
= pinned `net::DigestBytes` over prices + inventory, identical Windows/MSVC + Mac/clang.

## Stay self-contained — copy PcgHash (do NOT include pcg.h)
Like `wfc.h`'s `WfcHash`, copy the pure-`uint32` `pcg::PcgHash` ops VERBATIM into `econ.h` as `EconHash`
(citing `engine/pcg/pcg.h:42-48` as canonical) — pcg.h pulls the fx/particles chain, which would break
self-containment. And do NOT use `fpx.h`: express price elasticity as an **integer ratio**
`(delta * num) / den` with a pinned truncating division, NOT a Q16.16 `fxmul`. The whole slice stays
`int32`/`int64`, no float, no new include.

## Append to engine/econ/econ.h (below S3, in hf::econ)

1. **`EconHash`** — copied verbatim from `pcg.h:42-48` (pure uint32, same constants → same stream):
   ```cpp
   inline uint32_t EconHash(uint32_t seed, uint32_t index) {
       uint32_t h = seed * 2654435761u;
       h ^= (index + 0x9E3779B9u + (h << 6) + (h >> 2));
       h += index * 0x85EBCA6Bu;
       h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
       return h;
   }
   inline constexpr uint32_t kRollSalt = 0x45434F4Eu;  // 'ECON' — the roll stream's salt
   ```
   - **`Qty RollRange(uint32_t seed, uint32_t index, Qty lo, Qty hi)`** — a deterministic integer in
     `[lo, hi]` (inclusive): `lo + (Qty)(EconHash(seed ^ kRollSalt, index) % (uint32_t)(hi - lo + 1))`.
     Requires `hi >= lo`. Pure uint32 → bit-exact cross-platform.

2. **`Market`** — per-item integer prices + bounds + the designated currency item:
   ```cpp
   struct Market {
       std::vector<Qty> price;     // [itemCount] current unit price per item id
       Qty              minPrice = 1;
       Qty              maxPrice = 1000000;
       uint32_t         currency = 0;   // the item id used as money (excluded from its own pricing)
   };
   ```

3. **`Qty UpdatePrice(Qty price, Qty demand, Qty supply, Qty elastNum, Qty elastDen, Qty minP, Qty maxP)`** —
   the integer-ratio elasticity update, MONOTONIC + clamped:
   ```
   delta = demand - supply;                       // >0 excess demand, <0 excess supply
   step  = (int64)delta * elastNum / elastDen;    // integer truncating division (toward zero) — PINNED
   newP  = clamp(price + step, minP, maxP);
   ```
   Use `int64_t` for the multiply to avoid overflow, truncating division toward zero (the C++ `/` default —
   document it). Excess demand raises price, excess supply lowers it, balanced leaves it (within clamp).
   - **`void UpdatePrices(Market& m, const std::vector<Qty>& demand, const std::vector<Qty>& supply,
     Qty elastNum, Qty elastDen)`** — apply `UpdatePrice` to every item in fixed id order (skip `currency`).

4. **`TradeOrder` + market clearing** — a buy/sell at the current price, paid in currency:
   ```cpp
   struct TradeOrder { uint32_t buyer; uint32_t seller; uint32_t item; Qty qty; };  // qty > 0
   // Execute IFF: item != currency, ids in range, seller has qty of item, buyer has qty*price[item] currency.
   // Then: transfer qty of item seller->buyer, transfer qty*price[item] currency buyer->seller. Atomic; an
   // unaffordable/invalid order is a deterministic no-op returning false. (Reuse the S1 Transfer/Affordable
   // primitives; cost = (int64)qty * price clamped to Qty range — fixtures keep it bounded.)
   inline bool ExecuteTrade(World& w, const Market& m, const TradeOrder& o);
   inline void ClearMarket(World& w, const Market& m, const std::vector<TradeOrder>& orders);  // array order
   ```
   (Clearing executes at the price IN `m` at call time — S4 does not auto-reprice mid-batch; `UpdatePrices`
   is a separate explicit step, keeping the two concerns independently pinnable.)

5. **`uint64_t DigestState(const World& w, const Market& m)`** — combine `DigestWorld(w)` with a
   `net::DigestBytes` over `m.price` + the bounds/currency, in fixed order (the combined golden currency).

6. **Fixtures:** `Market MakeShowcaseMarket(const World& w)` (fixed integer prices per item, a currency item
   seeded with enough coin in the world that trades can pay), `std::vector<TradeOrder> MakeShowcaseTrades()`
   (affordable trades, an unaffordable trade [no-op], a currency-as-item trade [rejected], an out-of-range
   order), and fixed demand/supply vectors for the pricing test.

## The goldens (PINNED, cross-platform) — append to tests/econ_test.cpp
```
econ-s4: state digest after trades + repricing = 0x<...>
PASS econ-s4: DigestState after ClearMarket + UpdatePrices == pinned uint64 (the cross-platform proof)
PASS econ-s4: re-running the same trades + repricing is bit-identical (deterministic)
PASS econ-s4: MONOTONIC pricing — excess demand raises price, excess supply lowers it, balanced holds (clamped)
PASS econ-s4: prices clamp to [minPrice, maxPrice] (no runaway)
PASS econ-s4: a trade conserves total currency + total goods (buyer/seller exchange, nothing minted/burned)
PASS econ-s4: an unaffordable / currency-as-item / out-of-range order is a no-op (returns false, digest unchanged)
PASS econ-s4: RollRange is reproducible (same seed+index -> same value; a fixed roll sequence digests to a pinned uint64) and stays in [lo,hi]
```
Assertions:
1. **PINNED DIGEST** — `DigestState(w, m)` after `ClearMarket(MakeShowcaseTrades())` then
   `UpdatePrices(showcase demand/supply)` == a hard-pinned `uint64_t` (run once, pin; identical MSVC+clang).
2. **REPLAY-STABLE** — a second run from fresh world+market+same inputs → identical digest.
3. **MONOTONICITY** — `UpdatePrice` with `demand>supply` returns `>= price` (and strictly `>` when the step
   is nonzero and not clamped); `supply>demand` returns `<= price`; `demand==supply` returns `price`
   (within clamp). Assert all three directions on concrete values.
4. **CLAMP** — an extreme demand drives price to exactly `maxPrice` (not beyond); extreme supply to
   `minPrice`.
5. **TRADE CONSERVATION** — after a successful trade, total currency across (buyer,seller) is unchanged and
   total `item` across them is unchanged (goods + money exchanged, never minted/burned). Use `TotalQuantity`
   restricted to the two entities, or check the four touched slots sum-invariant.
6. **TRADE GATES** — an unaffordable trade, a `currency`-as-`item` trade, and an out-of-range order each
   return false AND leave the digest unchanged.
7. **ROLLS** — `RollRange(seed,i,lo,hi)` == itself on re-call (reproducible) and lies in `[lo,hi]` for a
   range of `i`; a fixed sequence of rolls digests (via `net::DigestBytes` over the rolled values) to a
   pinned `uint64_t`.

Keep S1+S2+S3 assertions green (append-only — S1 `0xaa712207f7663e03`, S2 `0x95147ff9dabbfd13`, S3
`0xca63394d5a6a9a2b` unchanged).

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/econ/econ.h` + `engine/net/session.h` + `tests/econ_test.cpp` (+ `tests/test_main.h`
+ `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine -I tests
tests/econ_test.cpp -o /tmp/econ && /tmp/econ`, confirming ALL assertions PASS with the IDENTICAL pinned
digests. (Local Windows clang is the fast pre-check.) NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- APPEND-ONLY to `engine/econ/econ.h` (add `EconHash`/`kRollSalt`/`RollRange`/`Market`/`UpdatePrice`/
  `UpdatePrices`/`TradeOrder`/`ExecuteTrade`/`ClearMarket`/`DigestState` + fixtures below S3; do NOT modify
  S1–S3 — reuse `At`/`Set`/`Affordable`/`InRange`/`ApplyCommand`(Transfer)/`DigestWorld`/`TotalQuantity`).
  Header stays SELF-CONTAINED: only `<cstdint>/<cstddef>/<vector>` + `net/session.h`. Do NOT include
  `pcg.h`/`fpx.h`/`<algorithm>`/any other header (copy `PcgHash` as `EconHash`; integer-ratio elasticity,
  NO `fxmul`). Do NOT modify `net/session.h` or any existing header.
- Pure-CPU INTEGER: NO float/`<cmath>`/clock/`std::rand`/`<random>`/`<unordered_*>`/`<map>`/`<functional>`/
  `std::hash`. All "randomness" is the deterministic `EconHash`. Integer min/max via ternary.
- `tests/econ_test.cpp` stays self-contained; APPEND the S4 assertions + fixtures. Keep S1–S3 green.
- Branch `fix-econ-s4`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target econ_test'`
  then run `econ_test` and confirm ALL assertions (S1–S4) PASS, exit 0. ALSO compile standalone with the
  local clang (`C:\Program Files\LLVM\bin\clang++.exe`) and confirm the IDENTICAL pinned digests. First run:
  pin the digests.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `econ_test` builds + PASSES on Windows with
  every assertion green (esp. monotonic pricing + trade conservation + roll reproducibility), and the local
  clang standalone passes with identical digests. Report: commit hash, full test output (printed digests +
  PASS lines), the pinned digests, confirmation S1–S3 digests unchanged, confirmation the header is still
  self-contained (list `#include`s — must be the same 4), the market/trades/rolls fixtures, and the
  local-clang result. Commit message via temp file + `git commit -F` (Bash heredoc). Commit to the branch
  and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S5 — the quest state machine + lockstep/rollback/desync
  HEADLINE.)
