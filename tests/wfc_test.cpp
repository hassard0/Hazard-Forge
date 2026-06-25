// Unit test for the deterministic WFC domain grid + adjacency rule-set + integer AC-3 propagation core
// (engine/wfc/wfc.h, Slice WFC-S1, flagship #29 WFC beachhead). Pure CPU (hf_core), ASan-eligible like
// the other pure tests.
//
// SELF-CONTAINED: the test scaffolding (check() + HF_TEST_MAIN_INIT()) is copied from session_test.cpp
// (NOT included) so this compiles STANDALONE with `clang++ -std=c++20 -I engine -I tests tests/wfc_test.cpp`
// on the Mac — the cheap cross-platform proof (like session_test / replay_test). Everything is INTEGER set
// logic over uint64 domain bitmasks, so the propagated grid — and hence wfc::DigestGrid (FNV-1a-64) over it
// — is bit-identical run-to-run AND platform-to-platform (MSVC vs Apple clang). The golden is a PINNED
// FNV-1a-64 DigestBytes value IN the test (NO image, NO render-bake).
//
// What this pins (the seven WFC-S1 assertions):
//   (a) DigestGrid(g) after center-collapse propagation == a hard-pinned uint64 (the cross-platform proof);
//   (b) the showcase tileset is adjacency-symmetric (AC-3 sound);
//   (c) re-running Propagate from the same start is bit-identical (deterministic);
//   (d) flipping one adjacency bit in a cloned tileset changes the digest (rules are load-bearing);
//   (e) propagation actually shrank >=1 neighbor domain (the constraint did work, not a no-op);
//   (f) Propagate returned true AND no cell domain is empty (no contradiction on the showcase).

#include "wfc/wfc.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::wfc;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

// Pure-integer popcount (NO <bit>, NO intrinsics — keeps the test self-contained + deterministic).
static int popcount64(Domain d) {
    int n = 0;
    while (d) { d &= (d - Domain{1}); ++n; }
    return n;
}

// Build the 16x16 showcase grid, pre-collapse the center cell to a single tile, seed the worklist, and
// propagate. Returns the propagated grid + the Propagate return value. Deterministic of nothing but the
// fixed showcase fixtures, so two calls are bit-identical.
static Grid runShowcase(const TileSet& ts, bool& okOut) {
    Grid g = MakeShowcaseGrid(16, 16);
    const int32_t c = g.cellId(8, 8);
    g.cell[static_cast<std::size_t>(c)] = Domain{1} << 2;  // collapse center to grass (tile 2)
    std::vector<int32_t> worklist{ c };
    okOut = Propagate(ts, g, worklist);
    return g;
}

int main() {
    HF_TEST_MAIN_INIT();

    const TileSet ts = MakeShowcaseTileSet();

    // The all-tiles domain + an adjacent cell id (the center's right neighbor) we measure for "did work".
    const Domain kAll = (Domain{1} << ts.tileCount) - Domain{1};

    // Run the showcase propagation.
    bool ok = false;
    const Grid g = runShowcase(ts, ok);
    const uint64_t digest = DigestGrid(g);

    std::printf("wfc-s1: propagated grid digest = 0x%016llx\n",
                static_cast<unsigned long long>(digest));

    // The pinned golden (computed on first run, hardcoded — the regression anchor / cross-platform bar).
    const uint64_t kPinnedDigest = 0xaaa67b9af6f293c8ull;  // PINNED on first run (the cross-platform anchor)

    // ---- (b) SYMMETRY — the showcase rule-set is adjacency-symmetric (AC-3 sound). -----------------
    check(IsSymmetric(ts), "wfc-s1: showcase tileset is adjacency-symmetric (AC-3 sound)");

    // ---- (a) PINNED DIGEST — the cross-platform make-or-break (identical on MSVC + clang). ---------
    check(digest == kPinnedDigest,
          "wfc-s1: Propagate(center-collapsed) digest == pinned uint64 (the cross-platform proof)");

    // ---- (c) REPLAY-STABLE — re-running from the same start is bit-identical. -----------------------
    {
        bool ok2 = false;
        const Grid g2 = runShowcase(ts, ok2);
        check(DigestGrid(g2) == digest && ok2 == ok,
              "wfc-s1: re-running Propagate from the same start is bit-identical (deterministic)");
    }

    // ---- (d) RULES LOAD-BEARING — flipping one adjacency bit changes the digest. --------------------
    {
        TileSet flipped = ts;
        // Flip a bit in GRASS's right-side rule — grass (tile 2) is the collapsed center cell, so it IS a
        // propagation SOURCE, making the bit load-bearing. allowed[2*4+kRight] == {sand,grass,rock};
        // toggling bit 1 (sand) removes sand from the right neighbor's permitted set -> a different grid.
        flipped.allowed[static_cast<std::size_t>(2) * 4u + static_cast<std::size_t>(kRight)] ^=
            (Domain{1} << 1);
        bool okF = false;
        const Grid gF = runShowcase(flipped, okF);
        check(DigestGrid(gF) != digest,
              "wfc-s1: flipping one adjacency bit in the tileset changes the digest (rules are load-bearing)");
    }

    // ---- (e) DID WORK — at least one neighbor domain shrank (the constraint propagated). ------------
    {
        // The center's 4 orthogonal neighbors started at all-tiles (popcount == tileCount). After
        // propagating the grass-collapse, grass only permits {sand,grass,rock}, so each neighbor must have
        // lost the `water` bit at minimum -> a strict popcount drop. Measure the right neighbor.
        const int32_t nId = g.cellId(9, 8);
        const int before  = popcount64(kAll);
        const int after   = popcount64(g.cell[static_cast<std::size_t>(nId)]);
        const bool shrank = (after < before);
        // Also confirm SOME cell differs from all-tiles (belt-and-suspenders).
        bool anyShrank = false;
        for (const Domain d : g.cell) if (d != kAll) { anyShrank = true; break; }
        check(shrank && anyShrank,
              "wfc-s1: propagation actually shrank >=1 neighbor domain (the constraint did work, not a no-op)");
    }

    // ---- (f) NO CONTRADICTION — Propagate returned true AND no empty domain. ------------------------
    {
        bool noEmpty = ok;
        for (const Domain d : g.cell) if (d == 0) { noEmpty = false; break; }
        check(noEmpty,
              "wfc-s1: no cell domain is empty (no contradiction on the showcase) AND Propagate returned true");
    }

    if (g_fail == 0) { std::printf("wfc_test: ALL PASS\n"); return 0; }
    std::printf("wfc_test: %d FAIL\n", g_fail);
    return 1;
}
