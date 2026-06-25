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

    // ================================================================================================
    // ---- Slice WFC-S2: min-entropy observe + seeded weighted collapse ------------------------------
    // ================================================================================================
    // Run a FIXED number K of ObserveStep on a fresh 16x16 showcase grid from a fixed seed, pin the
    // resulting grid digest, and assert the six S2 properties (no contradiction, pinned/replay-stable/
    // seed-driven digest, decided+locally-consistent collapses, observer termination).

    // The S2 collapse run as a helper so we can replay it bit-for-bit. Returns the grid + how many of the K
    // steps progressed and whether ANY contradiction occurred.
    auto runS2 = [&](uint32_t seed, int K, int& progressedOut, bool& contradictionOut) -> Grid {
        Grid g = MakeShowcaseGrid(16, 16);
        progressedOut = 0;
        contradictionOut = false;
        for (int i = 0; i < K; ++i) {
            const StepResult r = ObserveStep(ts, g, seed);
            if (r == StepResult::kProgressed)      ++progressedOut;
            else if (r == StepResult::kContradiction) { contradictionOut = true; break; }
            else /* kDone */                       break;
        }
        return g;
    };

    {
        const uint32_t kSeed = 0x1234ABCDu;
        const int      K     = 12;  // PINNED: contradiction-free on the permissive gradient tileset

        int  progressed = 0;
        bool contradiction = false;
        const Grid gS2 = runS2(kSeed, K, progressed, contradiction);
        const uint64_t s2Digest = DigestGrid(gS2);

        std::printf("wfc-s2: after K=%d collapses, grid digest = 0x%016llx\n",
                    K, static_cast<unsigned long long>(s2Digest));

        // PINNED on first run (the cross-platform anchor — identical MSVC + clang).
        const uint64_t kS2PinnedDigest = 0x4c9e67d356f4b920ull;

        // (1) NO CONTRADICTION — all K steps progressed.
        check(!contradiction && progressed == K,
              "wfc-s2: K observe steps all progressed (no contradiction on the showcase)");

        // (2) PINNED DIGEST — the cross-platform proof.
        check(s2Digest == kS2PinnedDigest,
              "wfc-s2: collapsed grid digest == pinned uint64 (the cross-platform proof)");

        // (3) REPLAY-STABLE — same seed -> identical digest.
        {
            int  p2 = 0;
            bool c2 = false;
            const Grid gS2b = runS2(kSeed, K, p2, c2);
            check(DigestGrid(gS2b) == s2Digest,
                  "wfc-s2: re-running the same seed is bit-identical (deterministic)");
        }

        // (4) SEED-DRIVEN — a different seed -> a DIFFERENT digest.
        {
            int  p3 = 0;
            bool c3 = false;
            const Grid gS2c = runS2(kSeed ^ 0xFFFFu, K, p3, c3);
            check(DigestGrid(gS2c) != s2Digest,
                  "wfc-s2: a DIFFERENT seed produces a DIFFERENT digest (the seed drives the result)");
        }

        // (5) DECIDED + CONSISTENT — every cell collapsed by the run has PopCount==1 and its tile is in the
        // allowed mask of each in-bounds neighbor's current domain. We check ALL decided cells (PopCount==1):
        // for each, for each in-bounds neighbor, the neighbor's domain must intersect AllowedMask(chosen,dir).
        {
            static const int kDirs[4] = { kRight, kUp, kLeft, kDown };
            static const int kDx[4]   = { +1, 0, -1, 0 };
            static const int kDz[4]   = {  0, +1, 0, -1 };
            bool consistent = true;
            bool anyDecided = false;
            for (int32_t z = 0; z < gS2.h && consistent; ++z) {
                for (int32_t x = 0; x < gS2.w && consistent; ++x) {
                    const int32_t c = gS2.cellId(x, z);
                    const Domain  d = gS2.cell[static_cast<std::size_t>(c)];
                    if (popcount64(d) != 1) continue;  // only decided cells
                    anyDecided = true;
                    // the chosen tile index (the single set bit).
                    uint32_t chosen = 0;
                    for (uint32_t t = 0; t < ts.tileCount; ++t)
                        if ((d >> t) & Domain{1}) { chosen = t; break; }
                    for (int dd = 0; dd < 4; ++dd) {
                        const int32_t nx = x + kDx[dd];
                        const int32_t nz = z + kDz[dd];
                        if (nx < 0 || nx >= gS2.w || nz < 0 || nz >= gS2.h) continue;
                        const Domain nDom = gS2.cell[static_cast<std::size_t>(gS2.cellId(nx, nz))];
                        // The chosen tile must permit SOME tile remaining in the neighbor on side `dir`.
                        const Domain allow = AllowedMask(ts, chosen, kDirs[dd]);
                        if ((nDom & allow) == 0) { consistent = false; break; }
                    }
                }
            }
            check(anyDecided && consistent,
                  "wfc-s2: every collapsed cell has PopCount==1 (decided) and its tile is allowed by its neighbors");
        }

        // (6) OBSERVER TERMINATION — SelectCell returns -1 on a fully-decided grid.
        {
            Grid tiny = MakeShowcaseGrid(2, 2);
            for (Domain& d : tiny.cell) d = Domain{1} << 0;  // collapse every cell to tile 0
            check(SelectCell(ts, tiny) == -1,
                  "wfc-s2: SelectCell returns -1 only when no cell has PopCount>1 (observer terminates correctly)");
        }
    }

    if (g_fail == 0) { std::printf("wfc_test: ALL PASS\n"); return 0; }
    std::printf("wfc_test: %d FAIL\n", g_fail);
    return 1;
}
