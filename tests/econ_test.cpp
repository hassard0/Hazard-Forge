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

    if (g_fail == 0) { std::printf("econ_test: ALL PASS\n"); return 0; }
    std::printf("econ_test: %d FAIL\n", g_fail);
    return 1;
}
