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

    if (g_fail == 0) { std::printf("econ_test: ALL PASS\n"); return 0; }
    std::printf("econ_test: %d FAIL\n", g_fail);
    return 1;
}
