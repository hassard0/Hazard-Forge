// Unit test for the deterministic integer LEDGER + atomic transactions (engine/econ/econ.h, Slice
// ECON-S1, flagship #30 ECON beachhead). Pure CPU (hf_core), ASan-eligible like the other pure tests.
//
// SELF-CONTAINED: the test scaffolding (check() + HF_TEST_MAIN_INIT()) is copied from session_test.cpp /
// wfc_test.cpp (NOT included) so this compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/econ_test.cpp` on the Mac — the cheap cross-platform
// proof. Everything is INTEGER bookkeeping over a fixed-order stock grid, so the ledger — and hence
// econ::DigestWorld (FNV-1a-64) over it — is bit-identical run-to-run AND platform-to-platform (MSVC vs
// Apple clang). The golden is a PINNED FNV-1a-64 DigestWorld value IN the test (NO image, NO render-bake).
//
// What this pins (the seven ECON-S1 assertions):
//   (a) DigestWorld after RunScript(showcase) == a hard-pinned uint64 (the cross-platform proof);
//   (b) re-running from a fresh world + the same script is bit-identical (deterministic / replay-stable);
//   (c) a Transfer-only script preserves TotalQuantity (conservation);
//   (d) flipping one Command.amount changes the digest (amounts are load-bearing);
//   (e) an unaffordable Remove is a no-op (ApplyCommand returns false AND the digest is unchanged);
//   (f) an out-of-range command is a no-op (ApplyCommand returns false AND the digest is unchanged);
//   (g) no stock slot is negative after the showcase (>= 0 over the fixed scan).

#include "econ/econ.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::econ;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

int main() {
    HF_TEST_MAIN_INIT();

    // The fixed showcase ledger: 4 entities x 4 items (the showcase script references entity 3 + item 3).
    const uint32_t kEntities = 4;
    const uint32_t kItems    = 4;

    // ---- Run the showcase script over a fresh showcase world. ---------------------------------------
    World w = MakeShowcaseWorld(kEntities, kItems);
    RunScript(w, MakeShowcaseScript());
    const uint64_t digest = DigestWorld(w);

    std::printf("econ-s1: ledger digest after showcase script = 0x%016llx\n",
                static_cast<unsigned long long>(digest));

    // The pinned golden (computed on first run, hardcoded — the regression anchor / cross-platform bar).
    const uint64_t kPinnedDigest = 0xaa712207f7663e03ull;  // PINNED on first run (the cross-platform anchor)

    // ---- (a) PINNED DIGEST — the cross-platform make-or-break (identical on MSVC + clang). ----------
    check(digest == kPinnedDigest,
          "econ-s1: DigestWorld after RunScript(showcase) == pinned uint64 (the cross-platform proof)");

    // ---- (b) REPLAY-STABLE — a fresh world + the same script reproduces the digest. -----------------
    {
        World w2 = MakeShowcaseWorld(kEntities, kItems);
        RunScript(w2, MakeShowcaseScript());
        check(DigestWorld(w2) == digest,
              "econ-s1: re-running the same script is bit-identical (deterministic)");
    }

    // ---- (c) CONSERVATION — a Transfer-ONLY script preserves TotalQuantity. -------------------------
    {
        World wc = MakeShowcaseWorld(kEntities, kItems);
        const int64_t before = TotalQuantity(wc);
        // A fixed Transfer-only script: every command moves stock between entities (mints/burns nothing).
        // Includes a src==dst net-zero transfer + an unaffordable transfer (no-op) — neither changes total.
        const std::vector<Command> transfers{
            { kTransfer, 0, 1, 0, 3 },
            { kTransfer, 1, 2, 1, 5 },
            { kTransfer, 2, 3, 2, 4 },
            { kTransfer, 3, 0, 3, 2 },
            { kTransfer, 1, 1, 0, 7 },     // src==dst: valid net-zero no-op
            { kTransfer, 0, 2, 1, 9999 },  // unaffordable: no-op (still conserves total)
        };
        RunScript(wc, transfers);
        const int64_t after = TotalQuantity(wc);
        check(after == before,
              "econ-s1: a Transfer-only script preserves TotalQuantity (conservation)");
    }

    // ---- (d) LOAD-BEARING — flipping one Command.amount changes the digest. --------------------------
    {
        std::vector<Command> mutated = MakeShowcaseScript();
        // The 0th command is `kAdd dst=0 item=0 amount=5` — a command that DOES apply, so changing its
        // amount changes the resulting ledger (and hence the digest).
        mutated[0].amount += 1;  // 5 -> 6
        World wm = MakeShowcaseWorld(kEntities, kItems);
        RunScript(wm, mutated);
        check(DigestWorld(wm) != digest,
              "econ-s1: flipping one Command.amount changes the digest (order/amounts are load-bearing)");
    }

    // ---- (e) AFFORDABILITY GATE — an unaffordable Remove is a no-op (false + digest unchanged). ------
    {
        World wa = MakeShowcaseWorld(kEntities, kItems);
        RunScript(wa, MakeShowcaseScript());
        const uint64_t beforeCmd = DigestWorld(wa);
        // Entity 0 holds far fewer than 1,000,000 of item 0 -> the Remove cannot pay -> no-op.
        const Command unaffordable{ kRemove, 0, 0, 0, 1000000 };
        const bool applied = ApplyCommand(wa, unaffordable);
        check(!applied && DigestWorld(wa) == beforeCmd,
              "econ-s1: an unaffordable Remove is a no-op (digest unchanged, ApplyCommand returned false)");
    }

    // ---- (f) BOUNDS GATE — an out-of-range command is a no-op (false + digest unchanged). -----------
    {
        World wb = MakeShowcaseWorld(kEntities, kItems);
        RunScript(wb, MakeShowcaseScript());
        const uint64_t beforeCmd = DigestWorld(wb);
        // entity 99 >= entityCount (4) -> out of range -> no-op.
        const Command outOfRange{ kAdd, 0, 99, 0, 5 };
        const bool applied = ApplyCommand(wb, outOfRange);
        check(!applied && DigestWorld(wb) == beforeCmd,
              "econ-s1: an out-of-range command is a no-op (digest unchanged, returned false)");
    }

    // ---- (g) NO NEGATIVE STOCK — scan the whole ledger after the showcase; every slot >= 0. ---------
    {
        bool noNegative = true;
        for (const Qty q : w.stock) if (q < 0) { noNegative = false; break; }
        check(noNegative,
              "econ-s1: no stock slot is negative after the showcase (>= 0 over the fixed scan)");
    }

    // =================================================================================================
    // ECON-S2 — Crafting / recipe transformer + deterministic craft queue (APPEND-ONLY below S1).
    // =================================================================================================
    // The showcase recipes + queue (FIXED forever; the S2 golden pins the post-drain digest):
    //   r0 = {2*item0 + 1*item1} -> {1*item2}   (ore+fuel->ingot)
    //   r1 = {3*item2}           -> {1*item3}   (ingots->tool)
    //   r2 = {1*item2 + 1*item3} -> {2*item2}   (item2 BOTH input+output -> pins consume-before-produce)
    const RecipeSet recipes = MakeShowcaseRecipes();

    // ---- Drain the fixed craft queue over a fresh showcase world. ------------------------------------
    World wq = MakeShowcaseWorld(kEntities, kItems);
    DrainCraftQueue(wq, recipes, MakeShowcaseCraftQueue());
    const uint64_t s2Digest = DigestWorld(wq);

    std::printf("econ-s2: ledger digest after craft queue = 0x%016llx\n",
                static_cast<unsigned long long>(s2Digest));

    // The pinned S2 golden (computed on first run, hardcoded — the cross-platform anchor).
    const uint64_t kPinnedS2Digest = 0x95147ff9dabbfd13ull;  // PINNED on first run (MSVC == clang)

    // ---- (1) PINNED DIGEST — the cross-platform make-or-break (identical on MSVC + clang). -----------
    check(s2Digest == kPinnedS2Digest,
          "econ-s2: DigestWorld after DrainCraftQueue == pinned uint64 (the cross-platform proof)");

    // ---- (2) REPLAY-STABLE — a fresh world + same recipes + same queue reproduces the digest. --------
    {
        World wq2 = MakeShowcaseWorld(kEntities, kItems);
        DrainCraftQueue(wq2, recipes, MakeShowcaseCraftQueue());
        check(DigestWorld(wq2) == s2Digest,
              "econ-s2: re-running the same queue is bit-identical (deterministic)");
    }

    // ---- (3) CRAFT BALANCE — a single ApplyRecipe's per-item ledger delta == Sigma outputs - inputs. -
    {
        World wb = MakeShowcaseWorld(kEntities, kItems);
        const uint32_t entity = 0;
        const uint32_t recipeId = 2;  // r2: {1*item2 + 1*item3} -> {2*item2} (item2 both input + output)
        // Compute expected per-item deltas directly from the recipe (outputs - inputs), over every item.
        const Recipe& r = recipes.recipes[recipeId];
        std::vector<Qty> expected(kItems, 0);
        for (const Ingredient& in  : r.inputs)  expected[in.item]  -= in.amount;
        for (const Ingredient& out : r.outputs) expected[out.item] += out.amount;
        // Snapshot the before-state of the crafting entity's whole row.
        std::vector<Qty> before(kItems);
        for (uint32_t t = 0; t < kItems; ++t) before[t] = wb.At(entity, t);
        const bool applied = ApplyRecipe(wb, recipes, recipeId, entity);
        bool balance = applied;
        for (uint32_t t = 0; t < kItems; ++t)
            if (wb.At(entity, t) - before[t] != expected[t]) balance = false;
        // r2 touches item2 (net +1: consume 1, produce 2) and item3 (net -1) — proves consume-before-produce.
        check(balance,
              "econ-s2: an affordable recipe consumes inputs + produces outputs (ledger delta == outputs - inputs)");
    }

    // ---- (4) AFFORDABILITY GATE — an unaffordable recipe is a no-op (false + digest unchanged). ------
    {
        // Fresh world: entity 0 holds item2 = 10 (10+0+6) and item3 = 19. r1 needs 3*item2, affordable;
        // craft r1 enough to drain item2 below 3, then assert the next r1 is a rejected no-op.
        World wa = MakeShowcaseWorld(kEntities, kItems);
        const uint32_t entity = 0;
        // Drain item2 on entity 0 down to < 3 by repeatedly crafting r1 (3*item2 -> 1*item3).
        while (RecipeAffordable(wa, recipes, /*r1=*/1, entity)) ApplyRecipe(wa, recipes, 1, entity);
        const uint64_t beforeCmd = DigestWorld(wa);
        const bool applied = ApplyRecipe(wa, recipes, /*r1=*/1, entity);  // now unaffordable
        check(!applied && DigestWorld(wa) == beforeCmd,
              "econ-s2: an unaffordable recipe is a deterministic no-op (digest unchanged, ApplyRecipe returned false)");
    }

    // ---- (5) PARTIAL DRAIN — count > affordable crafts exactly floor(available/cost) times, then stops.
    {
        // r0 on entity 1: needs 2*item0 + 1*item1. Fresh entity 1: item0=17 (10+7+0), item1=20 (10+7+3).
        // floor(17/2)=8 limited by item0, floor(20/1)=20 by item1 -> exactly 8 successful crafts.
        World wp = MakeShowcaseWorld(kEntities, kItems);
        const uint32_t entity = 1;
        const Qty item0Before = wp.At(entity, 0);
        const Qty item1Before = wp.At(entity, 1);
        const Qty item2Before = wp.At(entity, 2);
        const int expectedCrafts = item0Before / 2;  // 17/2 = 8 (the binding constraint)
        DrainCraftQueue(wp, recipes, std::vector<CraftOrder>{ { 0, entity, 100 } });  // count >> affordable
        // After N crafts of r0: item0 -= 2N, item1 -= N, item2 += N.
        const bool partial =
            (wp.At(entity, 0) == item0Before - 2 * expectedCrafts) &&
            (wp.At(entity, 1) == item1Before - 1 * expectedCrafts) &&
            (wp.At(entity, 2) == item2Before + 1 * expectedCrafts) &&
            !RecipeAffordable(wp, recipes, 0, entity);  // truly exhausted (can't afford a 9th)
        check(partial && expectedCrafts == 8,
              "econ-s2: a multi-count order crafts exactly as many times as affordable, then stops (partial drain deterministic)");
    }

    // ---- (6) BOUNDS GATE — an out-of-range recipeId is a no-op (false + digest unchanged). -----------
    {
        World wo = MakeShowcaseWorld(kEntities, kItems);
        DrainCraftQueue(wo, recipes, MakeShowcaseCraftQueue());
        const uint64_t beforeCmd = DigestWorld(wo);
        const bool appliedBadRecipe = ApplyRecipe(wo, recipes, /*recipeId=*/99, /*entity=*/0);  // OOR recipe
        const bool appliedBadEntity = ApplyRecipe(wo, recipes, /*recipeId=*/0, /*entity=*/99);  // OOR entity
        check(!appliedBadRecipe && !appliedBadEntity && DigestWorld(wo) == beforeCmd,
              "econ-s2: an out-of-range recipeId is a no-op (returned false, digest unchanged)");
    }

    // ---- (7) NO NEGATIVE STOCK — scan the whole ledger after the queue; every slot >= 0. ------------
    {
        bool noNegative = true;
        for (const Qty q : wq.stock) if (q < 0) { noNegative = false; break; }
        check(noNegative,
              "econ-s2: no stock slot is negative after the craft queue (>= 0 over the fixed scan)");
    }

    // =================================================================================================
    // ECON-S3 — Resource economy TICK: per-tick production / consumption flow (APPEND-ONLY below S2).
    // =================================================================================================
    // The showcase rules over the 4x4 seed world (stock(e,t) = 10 + 7*e + 3*t):
    //   producers: (e0,t0)+5 cap100, (e1,t1)+8 cap60, (e2,t2)+6 cap90 (the both-flows slot)
    //   consumers: (e3,t3)-4 (pure consumer -> 0), (e2,t2)-2 (same slot as the +6 producer)
    // production-before-consumption pins (e2,t2): each tick min(At+6,90)-2 -> settles to 88.
    const uint32_t kTicks = 24;  // N: every flow saturates/floors before this -> the economy has settled

    World we = MakeShowcaseWorld(kEntities, kItems);
    const EconRules rules = MakeShowcaseRules(we);
    RunEconTicks(we, rules, kTicks);
    const uint64_t s3Digest = DigestWorld(we);

    std::printf("econ-s3: ledger digest after N=%u econ ticks = 0x%016llx\n",
                kTicks, static_cast<unsigned long long>(s3Digest));

    // The pinned S3 golden (computed on first run, hardcoded — the cross-platform anchor; MSVC == clang).
    const uint64_t kPinnedS3Digest = 0xca63394d5a6a9a2bull;  // PINNED on first run (MSVC == clang)

    // ---- (1) PINNED DIGEST — the cross-platform make-or-break (identical on MSVC + clang). -----------
    check(s3Digest == kPinnedS3Digest,
          "econ-s3: DigestWorld after RunEconTicks == pinned uint64 (the cross-platform proof)");

    // ---- (2) REPLAY-STABLE — a fresh world + same rules + N ticks reproduces the digest. ------------
    {
        World we2 = MakeShowcaseWorld(kEntities, kItems);
        const EconRules rules2 = MakeShowcaseRules(we2);
        RunEconTicks(we2, rules2, kTicks);
        check(DigestWorld(we2) == s3Digest,
              "econ-s3: re-running N ticks is bit-identical (deterministic)");
    }

    // ---- (3) NON-NEGATIVE — scan the whole ledger after N ticks; every slot >= 0 (consumer clamp). ---
    {
        bool noNegative = true;
        for (const Qty q : we.stock) if (q < 0) { noNegative = false; break; }
        check(noNegative,
              "econ-s3: no stock slot is negative after N ticks (consumers clamp to 0)");
    }

    // ---- (4) CAP-RESPECTED — every capped slot is <= its cap after N ticks (producer clamp). --------
    {
        bool capRespected = true;
        for (uint32_t e = 0; e < kEntities; ++e)
            for (uint32_t t = 0; t < kItems; ++t) {
                const Qty cap = CapAt(rules, we, e, t);
                if (cap != -1 && we.At(e, t) > cap) capRespected = false;
            }
        check(capRespected,
              "econ-s3: no capped slot exceeds its cap after N ticks (producers clamp up)");
    }

    // ---- (5) PRODUCER SATURATION — a pure-producer slot (e0,t0) rises monotonically to its cap, holds.
    {
        // (e0,t0): +5/tick, cap 100, starts at 10. Track tick-by-tick: non-decreasing, ends == cap.
        World ws = MakeShowcaseWorld(kEntities, kItems);
        const EconRules rs = MakeShowcaseRules(ws);
        const Qty cap = CapAt(rs, ws, 0, 0);
        bool monotonicRise = true;
        Qty prev = ws.At(0, 0);
        for (uint32_t i = 0; i < kTicks; ++i) {
            EconTick(ws, rs);
            const Qty now = ws.At(0, 0);
            if (now < prev) monotonicRise = false;  // never decreases (pure producer)
            prev = now;
        }
        const bool endsAtCap = (ws.At(0, 0) == cap) && (cap == 100);
        check(monotonicRise && endsAtCap,
              "econ-s3: a pure-producer entity monotonically rises to its cap then holds (deterministic saturation)");
    }

    // ---- (6) CONSUMER DEPLETION — a pure-consumer slot (e3,t3) falls monotonically to 0, holds. ------
    {
        // (e3,t3): -4/tick, no producer, starts at 10+21+9 = 40. Track tick-by-tick: non-increasing, ends 0.
        World wd = MakeShowcaseWorld(kEntities, kItems);
        const EconRules rd = MakeShowcaseRules(wd);
        bool monotonicFall = true;
        Qty prev = wd.At(3, 3);
        for (uint32_t i = 0; i < kTicks; ++i) {
            EconTick(wd, rd);
            const Qty now = wd.At(3, 3);
            if (now > prev) monotonicFall = false;  // never increases (pure consumer)
            prev = now;
        }
        const bool endsAtZero = (wd.At(3, 3) == 0);
        check(monotonicFall && endsAtZero,
              "econ-s3: a pure-consumer entity monotonically falls to 0 then holds (deterministic depletion)");
    }

    // ---- (7) STEADY STATE — after N ticks one more EconTick leaves the digest UNCHANGED (idempotent). -
    {
        // `we` is already at tick N (settled). Capture its digest, run ONE more tick, assert no change.
        const uint64_t atRest = DigestWorld(we);
        World wss = we;  // copy the settled world so we don't disturb `we`
        EconTick(wss, rules);
        check(DigestWorld(wss) == atRest,
              "econ-s3: the economy reaches a STEADY STATE (digest at tick N == digest at tick N+1, EconTick idempotent at rest)");
    }

    // =================================================================================================
    // ECON-S4 — Pricing / market clearing + deterministic rolls (APPEND-ONLY below S3).
    // =================================================================================================
    // The market: integer supply/demand pricing, deterministic order clearing (trades at the current price
    // paid in a currency item), and deterministic rolls (a copied PcgHash). The 4x4 showcase world is
    // coin-seeded so buyers can pay; currency = item0. We clear the showcase order book at the showcase
    // prices, THEN reprice item1/2/3 from the showcase demand/supply, and pin DigestState over the result.

    // ---- Build the coin-seeded world + the fixed market, clear the order book, then reprice. ---------
    World wm4 = MakeShowcaseWorld(kEntities, kItems);
    SeedShowcaseCoin(wm4);                                  // +500 coin (item0) per entity so trades can pay
    Market market = MakeShowcaseMarket(wm4);
    ClearMarket(wm4, market, MakeShowcaseTrades());          // execute the order book at the showcase prices
    UpdatePrices(market, MakeShowcaseDemand(wm4), MakeShowcaseSupply(wm4), /*elastNum=*/1, /*elastDen=*/4);
    const uint64_t s4Digest = DigestState(wm4, market);

    std::printf("econ-s4: state digest after trades + repricing = 0x%016llx\n",
                static_cast<unsigned long long>(s4Digest));

    // The pinned S4 golden (computed on first run, hardcoded — the cross-platform anchor; MSVC == clang).
    const uint64_t kPinnedS4Digest = 0x087296381931e8b0ull;  // PINNED on first run (MSVC == clang)

    // ---- (1) PINNED DIGEST — the cross-platform make-or-break (identical on MSVC + clang). -----------
    check(s4Digest == kPinnedS4Digest,
          "econ-s4: DigestState after ClearMarket + UpdatePrices == pinned uint64 (the cross-platform proof)");

    // ---- (2) REPLAY-STABLE — fresh world + market + same inputs reproduces the digest. --------------
    {
        World wm2 = MakeShowcaseWorld(kEntities, kItems);
        SeedShowcaseCoin(wm2);
        Market market2 = MakeShowcaseMarket(wm2);
        ClearMarket(wm2, market2, MakeShowcaseTrades());
        UpdatePrices(market2, MakeShowcaseDemand(wm2), MakeShowcaseSupply(wm2), 1, 4);
        check(DigestState(wm2, market2) == s4Digest,
              "econ-s4: re-running the same trades + repricing is bit-identical (deterministic)");
    }

    // ---- (3) MONOTONICITY — excess demand raises, excess supply lowers, balanced holds (clamped). ----
    {
        const Qty p = 100, num = 1, den = 1, lo = 1, hi = 1000000;
        const Qty up   = UpdatePrice(p, /*demand=*/200, /*supply=*/50,  num, den, lo, hi);  // delta +150
        const Qty down = UpdatePrice(p, /*demand=*/50,  /*supply=*/200, num, den, lo, hi);  // delta -150
        const Qty hold = UpdatePrice(p, /*demand=*/77,  /*supply=*/77,  num, den, lo, hi);  // delta 0
        const bool mono =
            (up   >  p) &&   // strictly rises (step +150 nonzero, not clamped)
            (down <  p) &&   // strictly falls (step -150 nonzero, not clamped)
            (hold == p);     // balanced -> holds exactly
        check(mono,
              "econ-s4: MONOTONIC pricing — excess demand raises, excess supply lowers, balanced holds (clamped)");
    }

    // ---- (4) CLAMP — extreme demand -> exactly maxPrice; extreme supply -> exactly minPrice. ---------
    {
        const Qty p = 500, num = 1000, den = 1, lo = 1, hi = 1000000;
        const Qty hi2 = UpdatePrice(p, /*demand=*/2000000, /*supply=*/0,       num, den, lo, hi);  // huge +step
        const Qty lo2 = UpdatePrice(p, /*demand=*/0,       /*supply=*/2000000, num, den, lo, hi);  // huge -step
        check(hi2 == hi && lo2 == lo,
              "econ-s4: prices clamp to [minPrice, maxPrice] (no runaway)");
    }

    // ---- (5) TRADE CONSERVATION — a successful trade conserves total currency + total item across the
    // two parties (goods + money exchanged, nothing minted/burned). -----------------------------------
    {
        World wt = MakeShowcaseWorld(kEntities, kItems);
        SeedShowcaseCoin(wt);
        Market mt = MakeShowcaseMarket(wt);
        const TradeOrder o{ /*buyer=*/0, /*seller=*/1, /*item=*/2, /*qty=*/3 };  // affordable
        const Qty curBefore = wt.At(o.buyer, mt.currency) + wt.At(o.seller, mt.currency);
        const Qty itmBefore = wt.At(o.buyer, o.item)      + wt.At(o.seller, o.item);
        const bool ok = ExecuteTrade(wt, mt, o);
        const Qty curAfter = wt.At(o.buyer, mt.currency) + wt.At(o.seller, mt.currency);
        const Qty itmAfter = wt.At(o.buyer, o.item)      + wt.At(o.seller, o.item);
        check(ok && curBefore == curAfter && itmBefore == itmAfter,
              "econ-s4: a trade conserves total currency + total goods (buyer/seller exchange, nothing minted/burned)");
    }

    // ---- (6) TRADE GATES — unaffordable / currency-as-item / out-of-range each return false + leave the
    // digest unchanged. -------------------------------------------------------------------------------
    {
        World wg = MakeShowcaseWorld(kEntities, kItems);
        SeedShowcaseCoin(wg);
        Market mg = MakeShowcaseMarket(wg);
        const uint64_t before = DigestState(wg, mg);
        const bool a = ExecuteTrade(wg, mg, TradeOrder{ 0, 2, 2, 100000 });  // UNAFFORDABLE (1.2M coin)
        const bool b = ExecuteTrade(wg, mg, TradeOrder{ 0, 1, 0, 5 });       // CURRENCY-AS-ITEM (item0)
        const bool c = ExecuteTrade(wg, mg, TradeOrder{ 0, 99, 1, 1 });      // OUT-OF-RANGE seller
        const uint64_t after = DigestState(wg, mg);
        check(!a && !b && !c && before == after,
              "econ-s4: an unaffordable / currency-as-item / out-of-range order is a no-op (returns false, digest unchanged)");
    }

    // ---- (7) ROLLS — RollRange is reproducible + in [lo,hi]; a fixed roll sequence digests to a pinned
    // uint64. ----------------------------------------------------------------------------------------
    {
        const uint32_t seed = 0x1234ABCDu;
        const Qty lo = 7, hi = 42;
        bool reproducible = true, inRange = true;
        std::vector<Qty> rolls;
        for (uint32_t i = 0; i < 64; ++i) {
            const Qty r1 = RollRange(seed, i, lo, hi);
            const Qty r2 = RollRange(seed, i, lo, hi);  // same args -> same value
            if (r1 != r2) reproducible = false;
            if (r1 < lo || r1 > hi) inRange = false;
            rolls.push_back(r1);
        }
        const uint64_t rollDigest = hf::net::DigestBytes(rolls.data(), rolls.size() * sizeof(Qty));
        std::printf("econ-s4: fixed roll sequence digest = 0x%016llx\n",
                    static_cast<unsigned long long>(rollDigest));
        const uint64_t kPinnedRollDigest = 0x57a1cd074f19817aull;  // PINNED on first run (MSVC == clang)
        check(reproducible && inRange && rollDigest == kPinnedRollDigest,
              "econ-s4: RollRange is reproducible (same seed+index -> same value; a fixed roll sequence digests to a pinned uint64) and stays in [lo,hi]");
    }

    // =================================================================================================
    // ECON-S5 -- Quest state machine + lockstep/rollback/desync capstone (APPEND-ONLY below S4). HEADLINE.
    // =================================================================================================
    // The flagship headline: wrap the ENTIRE economy + a chained integer quest FSM in net::Session and
    // PROVE the moat -- two peers fed only the command stream re-derive a BIT-IDENTICAL economy+quests
    // (RunLockstep + DigestTrace), a mispredicted command ROLLS BACK to the authority state
    // (RollbackSession over a ScriptedTransport), and a divergence is LOCATED at the exact tick
    // (DesyncDetector). Pure-CPU INTEGER, reuses the net::* netcode machinery verbatim.

    using hf::net::InputRing;
    using hf::net::RunLockstep;
    using hf::net::DigestTrace;
    using hf::net::RollbackSession;
    using hf::net::ScriptedTransport;
    using hf::net::RunWithTransport;
    using hf::net::DesyncDetector;
    using hf::net::ChecksumPacket;
    using hf::net::Schedule;
    using hf::net::RecordLocal;
    using hf::net::IngestRemote;

    // The STATIC config captured by the step lambda: rules/recipes/graph (NOT snapshotted state).
    const RecipeSet s5Recipes = MakeShowcaseRecipes();
    const QuestGraph s5Graph  = MakeShowcaseQuests();
    const EconRules  s5Rules  = MakeShowcaseRules(MakeShowcaseWorld(kEntities, kItems));

    // The net::Session step: apply EVERY command this tick in array order, THEN AdvanceQuests ONCE (so the
    // quest FSM stays lockstep-aligned -- one fixed-order pass per tick after the tick's economy commands).
    auto step = [&](EconState& wst, const std::vector<EconCommand>& cmds, uint32_t /*tick*/) {
        for (const EconCommand& c : cmds) ApplyEconCommand(wst, c, s5Rules, s5Recipes);
        AdvanceQuests(wst.ledger, s5Graph, wst.quests);
    };
    auto econDigest = [&](const EconState& wst) { return DigestEconState(wst); };

    // Build the per-tick input ring from the fixed command stream (one command per tick; T = length).
    const std::vector<EconCommand> s5Stream = MakeShowcaseCommandStream();
    const uint32_t T = static_cast<uint32_t>(s5Stream.size());
    InputRing<EconCommand> s5Ring;
    for (uint32_t t = 0; t < T; ++t) s5Ring.AddInput(t, s5Stream[t]);

    // ---- PART A -- lockstep over net::Session (the determinism invariant). ----------------------------
    {
        // (1) LOCKSTEP INVARIANT: two independent DigestTrace runs from the same init+ring are EQUAL at
        //     EVERY tick (two peers fed the same inputs re-derive the bit-identical economy+quests).
        const std::vector<uint64_t> traceA = DigestTrace(MakeShowcaseState(), s5Ring, T, step, econDigest);
        const std::vector<uint64_t> traceB = DigestTrace(MakeShowcaseState(), s5Ring, T, step, econDigest);
        bool everyTickEqual = (traceA.size() == traceB.size());
        for (std::size_t i = 0; i < traceA.size() && everyTickEqual; ++i)
            if (traceA[i] != traceB[i]) everyTickEqual = false;
        check(everyTickEqual,
              "econ-s5: two peers from the same command stream have EQUAL DigestEconState at EVERY tick (lockstep invariant)");

        // (2) PINNED FINAL: RunLockstep final econ-state digest == a hard-pinned uint64.
        const uint64_t s5Lockstep = RunLockstep(MakeShowcaseState(), s5Ring, T, step, econDigest);
        std::printf("econ-s5: lockstep final econ-state digest = 0x%016llx  (T=%u ticks)\n",
                    static_cast<unsigned long long>(s5Lockstep), T);
        const uint64_t kPinnedS5Lockstep = 0xf27b85103a413a43ull;  // PINNED on first run (MSVC == clang)
        check(s5Lockstep == kPinnedS5Lockstep,
              "econ-s5: net::RunLockstep final digest == pinned uint64 (two peers re-derive the bit-identical economy+quests)");

        // (3) QUEST COMPLETION: advance a Session manually to the end + assert every status == kComplete.
        EconState wq5 = MakeShowcaseState();
        for (uint32_t t = 0; t < T; ++t) step(wq5, s5Ring.At(t), t);
        bool allComplete = !wq5.quests.status.empty();
        for (const uint8_t st : wq5.quests.status)
            if (st != static_cast<uint8_t>(kComplete)) allComplete = false;
        check(allComplete && wq5.quests.status.size() == 3,
              "econ-s5: the quest chain COMPLETES deterministically (final QuestState == all kComplete, pinned 3 objectives)");
    }

    // ---- PART B -- rollback correctness (the GGPO proof for gameplay state). --------------------------
    // Model two peers: local[t] is THIS peer's per-tick command, remote[t] is the OTHER peer's. The TRUE
    // per-tick input is {local[t], remote[t]} (the step applies both in array order). The authority is a
    // RunLockstep over the true combined inputs; the rollback session must converge to it under an
    // adversarial transport that DELAYS a remote command past its origin tick (forcing a mispredict).
    {
        // local[t] = the showcase stream (the economy-driving commands). remote[t] = a SECONDARY stream
        // of cheap txns (mint 1 item1 onto entity1 each tick) so the combined authority is well-defined.
        std::vector<EconCommand> local = s5Stream;
        std::vector<EconCommand> remote(T);
        for (uint32_t t = 0; t < T; ++t) {
            EconCommand c{}; c.tag = kTxnCmd; c.txn = Command{ kAdd, 0, 1, 1, 1 };  // mint 1 item1 -> entity1
            remote[t] = c;
        }
        // Make the DELAYED remote command (tick 3) GENUINELY DIFFERENT from the prediction so the mispredict
        // is real: a bigger mint. Prediction = lastConfirmed (remote[2], the +1 mint) -> differs -> rollback.
        remote[3] = EconCommand{}; remote[3].tag = kTxnCmd; remote[3].txn = Command{ kAdd, 0, 1, 1, 7 };

        // AUTHORITY: RunLockstep over the TRUE combined per-tick inputs {local[t], remote[t]}.
        InputRing<EconCommand> authRing;
        for (uint32_t t = 0; t < T; ++t) {
            authRing.AddInput(t, local[t]);   // local first (the step applies array order: local then remote)
            authRing.AddInput(t, remote[t]);
        }
        const uint64_t authority = RunLockstep(MakeShowcaseState(), authRing, T, step, econDigest);
        std::printf("econ-s5: rollback authority digest = 0x%016llx\n",
                    static_cast<unsigned long long>(authority));

        // ROLLBACK SESSION via the scripted transport. Every remote[t] is delivered ON TIME at tick t,
        // EXCEPT remote[3] which is DELAYED to deliver at tick 5 (>3) -- so ticks 3..4 are simulated with a
        // PREDICTION (lastConfirmed = remote[2], the +1 mint), and when remote[3] (the +7 mint) finally
        // arrives at tick 5 it mispredicts tick 3 -> a genuine ROLLBACK fires + re-sims 3..current.
        ScriptedTransport<EconCommand> tx;
        for (uint32_t t = 0; t < T; ++t) {
            if (t == 3) Schedule(tx, /*deliverTick=*/5, /*forTick=*/3, remote[3]);  // DELAYED past its origin
            else        Schedule(tx, /*deliverTick=*/t, /*forTick=*/t, remote[t]);  // on-time
        }
        RollbackSession<EconState, EconCommand> rb;
        rb.world = MakeShowcaseState();
        RunWithTransport(rb, local, tx, T, step);
        const uint64_t rollbackDigest = DigestEconState(rb.world);
        std::printf("econ-s5: rollback final digest = 0x%016llx, didRollback = %d\n",
                    static_cast<unsigned long long>(rollbackDigest), rb.didRollback ? 1 : 0);

        // (4) ROLLBACK == AUTHORITY (== a pinned uint64): the mispredicted command rolled back to the
        //     BIT-IDENTICAL authority economy+quest state.
        const uint64_t kPinnedS5Authority = 0x687feef8556f9949ull;  // PINNED on first run (MSVC == clang)
        check(rollbackDigest == authority && authority == kPinnedS5Authority,
              "econ-s5: a mispredicted command rolls back to the BIT-IDENTICAL authority economy+quest state (== pinned uint64)");

        // (5) DIDROLLBACK: a real misprediction fired.
        check(rb.didRollback,
              "econ-s5: rollback actually fired (didRollback == true)");

        // (6) ADVERSARIAL CONVERGENCE: a HEAVIER delay/reorder schedule still converges to the SAME pinned
        //     authority digest (the NS4 proof applied to gameplay state). Delay remote[3] to tick 6 AND
        //     remote[1] to tick 4 (reorder), confirm they still both mispredict + converge.
        ScriptedTransport<EconCommand> tx2;
        for (uint32_t t = 0; t < T; ++t) {
            if (t == 3)      Schedule(tx2, /*deliverTick=*/6, /*forTick=*/3, remote[3]);  // delayed further
            else if (t == 1) Schedule(tx2, /*deliverTick=*/4, /*forTick=*/1, remote[1]);  // delayed + reordered
            else             Schedule(tx2, /*deliverTick=*/t, /*forTick=*/t, remote[t]);
        }
        RollbackSession<EconState, EconCommand> rb2;
        rb2.world = MakeShowcaseState();
        RunWithTransport(rb2, local, tx2, T, step);
        check(DigestEconState(rb2.world) == authority && rb2.didRollback,
              "econ-s5: the adversarial (delayed/reordered) schedule converges to the SAME pinned authority digest");
    }

    // ---- PART C -- desync localization (the NS5 detector over gameplay state). ------------------------
    // Two command streams identical except tick K's command differs -> traces match for t<K, diverge at K;
    // DesyncDetector latches the exact tick.
    {
        const uint32_t K = 4;  // PINNED: the tick whose command is mutated to inject the desync
        // streamA = the showcase stream; streamB = identical except tick K's command differs (a craft vs
        // its original tick-4 economy-tick -> a real ledger divergence at tick K, none before).
        InputRing<EconCommand> ringA = s5Ring;
        InputRing<EconCommand> ringB;
        std::vector<EconCommand> streamB = s5Stream;
        { EconCommand c{}; c.tag = kCraftCmd; c.craft = CraftOrder{ 0, 0, 1 }; streamB[K] = c; }  // differ @K
        for (uint32_t t = 0; t < T; ++t) ringB.AddInput(t, streamB[t]);

        const std::vector<uint64_t> traceA  = DigestTrace(MakeShowcaseState(), ringA, T, step, econDigest);
        const std::vector<uint64_t> traceA2 = DigestTrace(MakeShowcaseState(), ringA, T, step, econDigest);
        const std::vector<uint64_t> traceB  = DigestTrace(MakeShowcaseState(), ringB, T, step, econDigest);

        // (7) CLEAN: DesyncDetector over (traceA vs traceA) reports NO desync.
        {
            DesyncDetector d;
            for (uint32_t t = 0; t < T; ++t) RecordLocal(d, t, traceA[t]);
            for (uint32_t t = 0; t < T; ++t) IngestRemote(d, ChecksumPacket{ t, traceA2[t] });
            check(!d.desynced,
                  "econ-s5: identical command streams report NO desync (clean)");
        }

        // (8) LOCATED: traceA vs traceB (tick K changed) -> d.desynced && d.desyncTick == K, traces equal
        //     for t < K. Pin K.
        {
            bool equalBeforeK = true;
            for (uint32_t t = 0; t < K; ++t) if (traceA[t] != traceB[t]) equalBeforeK = false;
            DesyncDetector d;
            for (uint32_t t = 0; t < T; ++t) RecordLocal(d, t, traceA[t]);
            for (uint32_t t = 0; t < T; ++t) IngestRemote(d, ChecksumPacket{ t, traceB[t] });
            std::printf("econ-s5: desync injected at tick K=%u, detector latched tick=%u\n",
                        K, d.desyncTick);
            check(d.desynced && d.desyncTick == K && equalBeforeK,
                  "econ-s5: a one-command divergence is LOCATED at the exact tick K (net::DesyncDetector), traces match for t<K");
        }
    }

    if (g_fail == 0) { std::printf("econ_test: ALL PASS\n"); return 0; }
    std::printf("econ_test: %d FAIL\n", g_fail);
    return 1;
}
