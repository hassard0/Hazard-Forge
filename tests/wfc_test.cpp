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

    // ================================================================================================
    // ---- Slice WFC-S3: the full deterministic BACKTRACKING solver ----------------------------------
    // ================================================================================================
    // PART A — a full solve on the permissive showcase tileset. PART B — a constrained scenario that
    // FORCES backtracking + an unsolvable scenario that returns solved==false (no hang). All ordering
    // pinned, so the fully-collapsed grid digest is bit-identical run-to-run AND MSVC-vs-clang.

    // A full-validity sweep: for EVERY cell + EVERY in-bounds neighbor, the neighbor's single tile must be
    // in AllowedMask(this tile, dir). Returns true iff the whole tilemap obeys every adjacency rule. Assumes
    // every cell PopCount==1 (a fully-collapsed grid).
    auto globallyConsistent = [&](const TileSet& tset, const Grid& gg) -> bool {
        static const int kDirs[4] = { kRight, kUp, kLeft, kDown };
        static const int kDx[4]   = { +1, 0, -1, 0 };
        static const int kDz[4]   = {  0, +1, 0, -1 };
        for (int32_t z = 0; z < gg.h; ++z) {
            for (int32_t x = 0; x < gg.w; ++x) {
                const Domain d = gg.cell[static_cast<std::size_t>(gg.cellId(x, z))];
                if (popcount64(d) != 1) return false;  // not fully collapsed
                uint32_t self = 0;
                for (uint32_t t = 0; t < tset.tileCount; ++t)
                    if ((d >> t) & Domain{1}) { self = t; break; }
                for (int dd = 0; dd < 4; ++dd) {
                    const int32_t nx = x + kDx[dd];
                    const int32_t nz = z + kDz[dd];
                    if (nx < 0 || nx >= gg.w || nz < 0 || nz >= gg.h) continue;
                    const Domain nDom = gg.cell[static_cast<std::size_t>(gg.cellId(nx, nz))];
                    uint32_t nTile = 0;
                    for (uint32_t t = 0; t < tset.tileCount; ++t)
                        if ((nDom >> t) & Domain{1}) { nTile = t; break; }
                    const Domain allow = AllowedMask(tset, self, kDirs[dd]);
                    if (((allow >> nTile) & Domain{1}) == 0) return false;  // rule violated
                }
            }
        }
        return true;
    };

    // ---- PART A — full solve on the showcase -------------------------------------------------------
    {
        const uint32_t kSeed    = 0x1234ABCDu;
        const uint32_t kMaxSteps = 100000u;

        Grid ga = MakeShowcaseGrid(16, 16);
        const SolveResult r = Solve(ts, ga, kSeed, kMaxSteps);
        const uint64_t solveDigest = DigestGrid(ga);

        std::printf("wfc-s3: solve -> solved=%d didBacktrack=%d steps=%u backtracks=%u, digest=0x%016llx\n",
                    static_cast<int>(r.solved), static_cast<int>(r.didBacktrack),
                    r.steps, r.backtracks, static_cast<unsigned long long>(solveDigest));

        // (1) SOLVED — fully collapsed, every cell PopCount==1.
        bool allDecided = r.solved;
        for (const Domain d : ga.cell) if (popcount64(d) != 1) { allDecided = false; break; }
        check(allDecided,
              "wfc-s3: Solve fully collapsed the grid (solved==true, every cell PopCount==1)");

        // (2) PINNED DIGEST — the cross-platform proof.
        const uint64_t kS3PinnedDigest = 0x0fffd74f7e8419acull;  // PINNED on first run
        check(solveDigest == kS3PinnedDigest,
              "wfc-s3: fully-collapsed grid digest == pinned uint64 (the cross-platform proof)");

        // (3) GLOBAL CONSISTENCY — every adjacent pair satisfies the rules.
        check(globallyConsistent(ts, ga),
              "wfc-s3: the collapsed assignment is GLOBALLY consistent (every adjacent pair satisfies the rules)");

        // (4) REPLAY-STABLE — same seed -> identical solved/steps/backtracks/digest.
        {
            Grid g2 = MakeShowcaseGrid(16, 16);
            const SolveResult r2 = Solve(ts, g2, kSeed, kMaxSteps);
            check(r2.solved == r.solved && r2.steps == r.steps && r2.backtracks == r.backtracks &&
                  DigestGrid(g2) == solveDigest,
                  "wfc-s3: re-running the same seed is bit-identical (solved/steps/backtracks/digest all identical)");
        }

        // (5) SEED-DRIVEN — a different seed -> a DIFFERENT digest, ALSO solved + globally consistent.
        {
            Grid g3 = MakeShowcaseGrid(16, 16);
            const SolveResult r3 = Solve(ts, g3, kSeed ^ 0xFFFFu, kMaxSteps);
            const uint64_t d3 = DigestGrid(g3);
            check(r3.solved && d3 != solveDigest && globallyConsistent(ts, g3),
                  "wfc-s3: a different seed yields a different (but still valid+consistent) full collapse");
        }
    }

    // ---- PART B — backtracking actually fires + an unsolvable scenario -----------------------------
    // Strategy (a): a deliberately TIGHT tileset — the canonical WFC "pipes/circuit" rule-set with
    // DIRECTIONAL (anisotropic) adjacency. Each tile has a 4-bit connector signature (a pipe stub on each
    // of its R/U/L/D sides); two tiles may share an edge IFF their stubs MATCH across that edge (both
    // present or both absent — a pipe never dead-ends into a wall). With a CLOSED border (no stub may point
    // off-grid) the only valid layouts are closed loops, and the S2 min-entropy greedy paints itself into
    // corners that AC-3 propagation only catches AFTER a doomed collapse — forcing the solver to BACKTRACK.
    // 8 tiles: 0 empty, 1 horiz(R,L), 2 vert(U,D), 3 cross(R,U,L,D), 4 elbow R+U, 5 U+L, 6 L+D, 7 D+R.
    // Weights {1,3,3,5,2,2,2,2} (the cross weighted heavy) bias the greedy into the trap — chosen so the
    // pinned scenario below fires backtracking deterministically. The tileset is adjacency-symmetric by
    // construction (the stub-match relation is symmetric).
    static const int kStub[8] = {
        0,                                  // 0 empty
        (1<<0)|(1<<2),                      // 1 horiz: R,L
        (1<<1)|(1<<3),                      // 2 vert:  U,D
        (1<<0)|(1<<1)|(1<<2)|(1<<3),        // 3 cross
        (1<<0)|(1<<1),                      // 4 elbow R+U
        (1<<1)|(1<<2),                      // 5 elbow U+L
        (1<<2)|(1<<3),                      // 6 elbow L+D
        (1<<3)|(1<<0)                       // 7 elbow D+R
    };
    auto makePipesTileSet = []() -> TileSet {
        TileSet t;
        t.tileCount = 8;
        const int32_t w[8] = { 1, 3, 3, 5, 2, 2, 2, 2 };
        t.weight.assign(w, w + 8);
        t.allowed.assign(8u * 4u, Domain{0});
        // dir order: kRight=0,kUp=1,kLeft=2,kDown=3 (the stub bit index matches the dir index). Across edge
        // `dir`, my stub on side `dir` must equal the neighbor's stub on side Opposite(dir).
        for (uint32_t tt = 0; tt < 8u; ++tt)
            for (int dir = 0; dir < 4; ++dir) {
                const int myStub = (kStub[tt] >> dir) & 1;
                const int opp    = Opposite(dir);
                Domain mask = 0;
                for (uint32_t u = 0; u < 8u; ++u)
                    if (((kStub[u] >> opp) & 1) == myStub) mask |= (Domain{1} << u);
                t.allowed[static_cast<std::size_t>(tt) * 4u + static_cast<std::size_t>(dir)] = mask;
            }
        return t;
    };

    {
        const TileSet sts = makePipesTileSet();
        check(IsSymmetric(sts), "wfc-s3: the constrained (pipes) tileset is adjacency-symmetric (AC-3 sound)");

        const uint32_t kSeed     = 0x20ae7589u;  // PINNED: this seed fires backtracking on the 5x8 closed grid
        const uint32_t kMaxSteps = 100000u;

        // Build a 5x8 all-domain grid, then forbid any tile whose stub points OFF the grid border (a closed
        // boundary). Propagating that boundary constraint leaves a tight loops-only search the greedy must
        // backtrack through. Factor it so we can replay bit-for-bit.
        auto buildConstrained = [&]() -> Grid {
            const int32_t W = 5, H = 8;
            Grid g;
            g.w = W; g.h = H;
            const Domain all = (Domain{1} << 8) - Domain{1};  // all 8 tiles
            g.cell.assign(static_cast<std::size_t>(W) * static_cast<std::size_t>(H), all);
            for (int32_t z = 0; z < H; ++z)
                for (int32_t x = 0; x < W; ++x) {
                    Domain keep = 0;
                    for (uint32_t t = 0; t < 8u; ++t) {
                        bool ok = true;
                        if (x == W - 1 && (kStub[t] & (1 << 0))) ok = false;  // right border: no R stub
                        if (z == H - 1 && (kStub[t] & (1 << 1))) ok = false;  // top border:   no U stub
                        if (x == 0     && (kStub[t] & (1 << 2))) ok = false;  // left border:  no L stub
                        if (z == 0     && (kStub[t] & (1 << 3))) ok = false;  // bottom border:no D stub
                        if (ok) keep |= (Domain{1} << t);
                    }
                    g.cell[static_cast<std::size_t>(g.cellId(x, z))] &= keep;
                }
            return g;
        };
        // Seed-propagate the whole boundary-constrained grid before solving.
        auto seedAndSolve = [&](Grid& g) -> SolveResult {
            std::vector<int32_t> wl;
            for (int32_t i = 0; i < g.w * g.h; ++i) wl.push_back(i);
            Propagate(sts, g, wl);  // make the boundary constraints live
            return Solve(sts, g, kSeed, kMaxSteps);
        };

        Grid gC = buildConstrained();
        const SolveResult rC = seedAndSolve(gC);
        const uint64_t cDigest = DigestGrid(gC);

        std::printf("wfc-s3: constrained solve -> solved=%d didBacktrack=%d steps=%u backtracks=%u, digest=0x%016llx\n",
                    static_cast<int>(rC.solved), static_cast<int>(rC.didBacktrack),
                    rC.steps, rC.backtracks, static_cast<unsigned long long>(cDigest));

        // (6) BACKTRACK FIRED — didBacktrack && backtracks>=1 && solved, globally consistent.
        check(rC.didBacktrack && rC.backtracks >= 1u && rC.solved && globallyConsistent(sts, gC),
              "wfc-s3: backtracking FIRES on the constrained scenario (didBacktrack==true, backtracks>=1) and still solves consistently");

        // (6b) PINNED DIGEST + bit-identical re-run (the deterministic backtracking path).
        const uint64_t kS3ConstrainedDigest = 0x8adb136f5b4c690aull;  // PINNED on first run
        {
            Grid gC2 = buildConstrained();
            const SolveResult rC2 = seedAndSolve(gC2);
            check(cDigest == kS3ConstrainedDigest &&
                  rC2.solved == rC.solved && rC2.didBacktrack == rC.didBacktrack &&
                  rC2.steps == rC.steps && rC2.backtracks == rC.backtracks &&
                  DigestGrid(gC2) == cDigest,
                  "wfc-s3: the constrained solve is also bit-identical on re-run (deterministic backtracking path) + pinned digest");
        }

        // (7) UNSOLVABLE IS DETERMINISTIC — two horizontally-adjacent cells pre-pinned to tiles whose stubs
        // CLASH across their shared edge (cell A = tile 1 horiz, which presents an R stub on its right edge;
        // cell B = tile 0 empty, which presents NO L stub) -> the stub-match rule forbids them adjacent ->
        // propagation empties a domain -> Solve must return solved==false within maxSteps (no hang),
        // reproducibly. (Both cells already single-tile, so there is no alternative to back out to.)
        auto buildUnsolvable = [&]() -> Grid {
            Grid g;
            g.w = 2; g.h = 1;
            const Domain all = (Domain{1} << 8) - Domain{1};
            g.cell.assign(2, all);
            g.cell[static_cast<std::size_t>(g.cellId(0, 0))] = Domain{1} << 1;  // horiz: R stub on its right edge
            g.cell[static_cast<std::size_t>(g.cellId(1, 0))] = Domain{1} << 0;  // empty: no L stub -> clash
            return g;
        };
        {
            Grid gU = buildUnsolvable();
            std::vector<int32_t> wl{ gU.cellId(0,0), gU.cellId(1,0) };
            const bool propOk = Propagate(sts, gU, wl);  // already contradicts (empties a domain)
            const SolveResult rU = Solve(sts, gU, kSeed, kMaxSteps);

            Grid gU2 = buildUnsolvable();
            std::vector<int32_t> wl2{ gU2.cellId(0,0), gU2.cellId(1,0) };
            Propagate(sts, gU2, wl2);
            const SolveResult rU2 = Solve(sts, gU2, kSeed, kMaxSteps);

            std::printf("wfc-s3: unsolvable -> propOk=%d solved=%d steps=%u (reproduce solved=%d)\n",
                        static_cast<int>(propOk), static_cast<int>(rU.solved), rU.steps,
                        static_cast<int>(rU2.solved));
            check(!rU.solved && !rU2.solved && rU.steps <= kMaxSteps,
                  "wfc-s3: an UNSOLVABLE scenario returns solved==false deterministically (no hang)");
        }
    }

    // ================================================================================================
    // ---- Slice WFC-S4: adjacency LEARNED from a sample + region pre-constraints ---------------------
    // ================================================================================================
    // PART A — learn an adjacency rule-set from a fixed sample tilemap (the Wang model) and solve a fresh
    // 16x16 from it. PART B — region pre-constraints (PinCell/ConstrainCell/ApplyConstraints) honored
    // through the solve. All ordering pinned, so the learned-adjacency digest + the solved grid digests are
    // bit-identical run-to-run AND MSVC-vs-clang.

    // ---- PART A — learned rules reproduce + solve --------------------------------------------------
    {
        // A fixed 7x7 gradient-island sample (ids 0=water,1=sand,2=grass,3=rock). Concentric rings:
        // a water border, a sand ring, a grass core, and a single rock at the dead center. Every adjacent
        // pair is a gradient step (water-water, water-sand, sand-sand, sand-grass, grass-grass, grass-rock),
        // so the LEARNED rules match the S1 showcase gradient and a 16x16 is solvable.
        SampleMap sample;
        sample.w = 7; sample.h = 7;
        const int32_t S[49] = {
            //  x: 0  1  2  3  4  5  6
                0, 0, 0, 0, 0, 0, 0,   // z=0
                0, 1, 1, 1, 1, 1, 0,   // z=1
                0, 1, 2, 2, 2, 1, 0,   // z=2
                0, 1, 2, 3, 2, 1, 0,   // z=3  <- rock(3) at center, all neighbors grass(2)
                0, 1, 2, 2, 2, 1, 0,   // z=4
                0, 1, 1, 1, 1, 1, 0,   // z=5
                0, 0, 0, 0, 0, 0, 0    // z=6
        };
        sample.tile.assign(S, S + 49);

        const TileSet learned = LearnTileSet(sample);

        // Expected per-tile occurrence counts (computed independently from the literal sample).
        int32_t expCount[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < 49; ++i) ++expCount[S[i]];  // water=24, sand=16, grass=8, rock=1

        // The learned adjacency-mask digest (over the allowed[] vector, the same currency as DigestGrid).
        const uint64_t learnedAdjDigest =
            hf::net::DigestBytes(learned.allowed.data(), learned.allowed.size() * sizeof(Domain));

        std::printf("wfc-s4: learned tileset: tiles=%u, weights=[%d,%d,%d,%d], adjacency digest=0x%016llx\n",
                    learned.tileCount, learned.weight[0], learned.weight[1], learned.weight[2],
                    learned.weight[3], static_cast<unsigned long long>(learnedAdjDigest));

        // (1) SYMMETRIC — every pair observed from both sides => adjacency-symmetric (AC-3 sound).
        check(IsSymmetric(learned),
              "wfc-s4: LearnTileSet is adjacency-symmetric (learned rules are AC-3 sound)");

        // (2) ADJACENCY DIGEST PINNED — the rules are a deterministic function of the sample.
        const uint64_t kS4AdjDigest = 0xb3b956701ab39c83ull;  // PINNED on first run (cross-platform anchor)
        check(learnedAdjDigest == kS4AdjDigest,
              "wfc-s4: learned adjacency-mask digest == pinned uint64 (rules derived deterministically from the sample)");

        // (3) WEIGHTS = COUNTS — weight[t] equals the literal occurrence count of tile t in the sample.
        check(learned.tileCount == 4u &&
              learned.weight[0] == expCount[0] && learned.weight[1] == expCount[1] &&
              learned.weight[2] == expCount[2] && learned.weight[3] == expCount[3],
              "wfc-s4: learned weights == per-tile occurrence counts in the sample");

        // (4) SOLVE PINNED + CONSISTENT — Solve a fresh 16x16 from the learned rules.
        const uint32_t kSeed     = 0x1234ABCDu;
        const uint32_t kMaxSteps = 100000u;
        // A fresh all-tiles 16x16 grid for the learned 4-tile set (MakeShowcaseGrid also uses 4 tiles).
        Grid gL = MakeShowcaseGrid(16, 16);
        const SolveResult rL = Solve(learned, gL, kSeed, kMaxSteps);
        const uint64_t solveDigestL = DigestGrid(gL);

        std::printf("wfc-s4: learned solve -> solved=%d steps=%u backtracks=%u, digest=0x%016llx\n",
                    static_cast<int>(rL.solved), rL.steps, rL.backtracks,
                    static_cast<unsigned long long>(solveDigestL));

        const uint64_t kS4SolveDigest = 0x2e31ea681c0bd906ull;  // PINNED on first run
        bool allDecidedL = rL.solved;
        for (const Domain d : gL.cell) if (popcount64(d) != 1) { allDecidedL = false; break; }
        check(allDecidedL && globallyConsistent(learned, gL) && solveDigestL == kS4SolveDigest,
              "wfc-s4: Solve(learnedTileSet, 16x16, seed) fully collapses + globally consistent, digest == pinned");

        // (5) NO INVENTED TILES — every tile id in the solved grid also appears in the sample.
        {
            // Set of tile ids present in the sample.
            uint64_t sampleTiles = 0;
            for (int i = 0; i < 49; ++i) sampleTiles |= (Domain{1} << static_cast<uint32_t>(S[i]));
            bool onlyObserved = true;
            for (const Domain d : gL.cell) {
                // d is a single-bit domain (decided); its bit must be a sample tile.
                if ((d & sampleTiles) != d) { onlyObserved = false; break; }
            }
            check(onlyObserved,
                  "wfc-s4: every tile in the solved output also appears in the sample (no tile invented out of nothing)");
        }
    }

    // ---- PART B — region pre-constraints honored ---------------------------------------------------
    {
        const uint32_t kSeed     = 0x1234ABCDu;
        const uint32_t kMaxSteps = 100000u;
        const Domain   kSandMask = Domain{1} << 1;  // sand-only constraint mask

        // The showcase gradient tileset (4 tiles, isotropic gradient) for the constrained solve.
        // Pin water(0) at the top-left corner, rock(3) at the top-right corner (both gradient-consistent,
        // and BOTH off the constrained bottom row so they don't clash with the sand mask), and constrain
        // the entire bottom border row (z=0) to sand-only.
        auto buildConstrained = [&]() -> Grid {
            Grid g = MakeShowcaseGrid(16, 16);
            PinCell(g, 0, 15, 0);             // water at the top-left corner
            PinCell(g, 15, 15, 3);            // rock at the top-right corner
            for (int32_t x = 0; x < 16; ++x)  // the whole bottom border row -> sand only
                ConstrainCell(g, x, 0, kSandMask);
            return g;
        };

        Grid gB = buildConstrained();
        const bool applied = ApplyConstraints(ts, gB);
        const SolveResult rB = Solve(ts, gB, kSeed, kMaxSteps);
        const uint64_t solveDigestB = DigestGrid(gB);

        std::printf("wfc-s4: constrained solve -> solved=%d, applied=%d, digest=0x%016llx\n",
                    static_cast<int>(rB.solved), static_cast<int>(applied),
                    static_cast<unsigned long long>(solveDigestB));

        // (6) PINS PROPAGATE — ApplyConstraints returned true (no immediate contradiction).
        check(applied,
              "wfc-s4: ApplyConstraints propagated the pins without contradiction");

        // (7) SOLVE PINNED + CONSISTENT — Solve fully collapses, globally consistent, pinned digest.
        const uint64_t kS4ConstrainedDigest = 0xdeaac98f4242eecaull;  // PINNED on first run
        bool allDecidedB = rB.solved;
        for (const Domain d : gB.cell) if (popcount64(d) != 1) { allDecidedB = false; break; }
        check(allDecidedB && globallyConsistent(ts, gB) && solveDigestB == kS4ConstrainedDigest,
              "wfc-s4: Solve with pins fully collapses + globally consistent, digest == pinned");

        // (8) PINS HONORED — every pinned cell holds exactly its pinned tile; the border row is all sand.
        {
            const Domain water = Domain{1} << 0;
            const Domain rock  = Domain{1} << 3;
            bool pinsHonored =
                (gB.cell[static_cast<std::size_t>(gB.cellId(0, 15))]  == water) &&
                (gB.cell[static_cast<std::size_t>(gB.cellId(15, 15))] == rock);
            bool borderSand = true;
            for (int32_t x = 0; x < 16; ++x)
                if (gB.cell[static_cast<std::size_t>(gB.cellId(x, 0))] != kSandMask) { borderSand = false; break; }
            check(pinsHonored && borderSand,
                  "wfc-s4: every pinned cell holds EXACTLY its pinned tile + the sand-constrained border row holds only sand");
        }

        // (9) REPLAY-STABLE — re-running the constrained solve from the same seed is bit-identical.
        {
            Grid gB2 = buildConstrained();
            const bool applied2 = ApplyConstraints(ts, gB2);
            const SolveResult rB2 = Solve(ts, gB2, kSeed, kMaxSteps);
            check(applied2 == applied && rB2.solved == rB.solved &&
                  rB2.steps == rB.steps && rB2.backtracks == rB.backtracks &&
                  DigestGrid(gB2) == solveDigestB,
                  "wfc-s4: re-running the constrained solve is bit-identical (deterministic)");
        }

        // (10) CONTRADICTORY PINS DETERMINISTIC — two ADJACENT cells pinned to a forbidden pair (water
        // next to rock — the gradient forbids water-rock adjacency) -> ApplyConstraints returns false
        // (or Solve solved==false), reproducibly, no hang.
        {
            auto buildBadPins = [&]() -> Grid {
                Grid g = MakeShowcaseGrid(16, 16);
                PinCell(g, 5, 5, 0);  // water
                PinCell(g, 6, 5, 3);  // rock, immediately to the right -> water-rock forbidden
                return g;
            };
            Grid gBad = buildBadPins();
            const bool appliedBad = ApplyConstraints(ts, gBad);
            const SolveResult rBad = Solve(ts, gBad, kSeed, kMaxSteps);

            Grid gBad2 = buildBadPins();
            const bool appliedBad2 = ApplyConstraints(ts, gBad2);
            const SolveResult rBad2 = Solve(ts, gBad2, kSeed, kMaxSteps);

            std::printf("wfc-s4: contradictory pins -> applied=%d solved=%d (reproduce applied=%d solved=%d)\n",
                        static_cast<int>(appliedBad), static_cast<int>(rBad.solved),
                        static_cast<int>(appliedBad2), static_cast<int>(rBad2.solved));
            check((!appliedBad || !rBad.solved) && (!appliedBad2 || !rBad2.solved) &&
                  appliedBad == appliedBad2 && rBad.solved == rBad2.solved &&
                  rBad.steps <= kMaxSteps,
                  "wfc-s4: contradictory pins -> ApplyConstraints returns false (or Solve solved==false), deterministically (no hang)");
        }
    }

    if (g_fail == 0) { std::printf("wfc_test: ALL PASS\n"); return 0; }
    std::printf("wfc_test: %d FAIL\n", g_fail);
    return 1;
}
