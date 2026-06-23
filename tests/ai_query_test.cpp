// Slice AI2 — Deterministic AI: ENVIRONMENT QUERIES (integer scoring over the navmesh) — the 2nd slice
// of the DETERMINISTIC AI flagship (GitHub issue #28, hf::ai). ai.h is APPEND-ONLY (AI1 byte-frozen); AI2
// adds the integer candidate generator + the integer scorer chain (Q16.16 distance² + nav-reachability) +
// RunQuery (lowest combined score, tie -> lowest candidate index).
//
// What this test PINS (the determinism contract + the scorer semantics):
//   * RunQuery TWO-RUN determinism: the same query + navmesh yield the same {bestIndex, bestScore} on two
//     independent runs (fixed candidate order + integer scoring + lowest-index tie-break).
//   * The CLOSEST REACHABLE candidate wins (lowest distance² among reachable).
//   * An UNREACHABLE candidate (a different nav component / no path) is penalized (kBigPenalty) and is
//     NEVER chosen even when it is the NEAREST candidate.
//   * A constructed TIE (two candidates with the same combined score) resolves to the LOWEST index.
//   * The generator is deterministic (the same candidates, in the same fixed order, on two calls).
//   * ScoreDistanceSq matches a hand-computed verdict::FxDist2.
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "ai/ai.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace ai      = hf::ai;
namespace nav     = hf::nav;
namespace fpx     = hf::sim::fpx;
namespace verdict = hf::game::verdict;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A Q16.16 helper: integer world units -> fx.
static fpx::fx FI(int v) { return (fpx::fx)((int64_t)v * (int64_t)fpx::kOne); }

// Build a TINY synthetic NavScene by hand: a row of N polys laid out left-to-right, each adjacent to its
// neighbours so they form ONE connected component, PLUS an isolated poly (no neighbours) forming a SECOND
// component. We bypass the full nav pipeline so the reachability cases are exact + cheap. The poly idx/
// region machinery is unused by AI2's scorer (it reads cx/cz/comp/polys.nbr only), so we set centroids
// directly + wire nbr[] by hand.
static ai::NavScene MakeLineScene(int n, int32_t spacing, bool addIsolated) {
    ai::NavScene s;
    s.navScale = 1;   // voxel == world units for the synthetic scene (so WorldToVoxel is identity*scale)
    const int total = n + (addIsolated ? 1 : 0);
    s.polys.assign((size_t)total, nav::Poly{});
    s.cx.assign((size_t)total, 0);
    s.cz.assign((size_t)total, 0);
    for (int i = 0; i < total; ++i) {
        for (int e = 0; e < 3; ++e) s.polys[(size_t)i].nbr[e] = nav::kNoNeighbour;
    }
    // The connected row: poly i centroid at (i*spacing, 0); neighbour links i<->i+1.
    for (int i = 0; i < n; ++i) {
        s.cx[(size_t)i] = (int32_t)i * spacing;
        s.cz[(size_t)i] = 0;
    }
    for (int i = 0; i < n; ++i) {
        if (i > 0)     s.polys[(size_t)i].nbr[0] = (uint32_t)(i - 1);
        if (i < n - 1) s.polys[(size_t)i].nbr[1] = (uint32_t)(i + 1);
    }
    // The isolated poly (its OWN component): far away, no neighbours.
    if (addIsolated) {
        s.cx[(size_t)n] = (int32_t)n * spacing + 10000;   // far from the row
        s.cz[(size_t)n] = 10000;
    }
    s.nComp = nav::ConnectedComponents(s.polys, s.comp);
    return s;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- (1) ScoreDistanceSq == a hand-computed verdict::FxDist2, narrowed >> kFrac (NO sqrt) ----------
    {
        const fpx::FxVec3 b{0, 0, 0};
        const fpx::FxVec3 q{0, 0, 0};
        check(ai::ScoreDistanceSq(b, q) == 0, "ai2: ScoreDistanceSq of identical points is 0");
        // (3,0,4): FxDist2 = (3*kOne)² + (4*kOne)² = 25 * kOne² (Q32.32). ScoreDistanceSq narrows >> kFrac
        // = 25 * kOne (= 25 << 16 = 1638400), which fits int32 and keeps full resolution. Hand-check both.
        const fpx::FxVec3 u{FI(3), 0, FI(4)};
        const int64_t handRaw = (int64_t)FI(3) * (int64_t)FI(3) + (int64_t)FI(4) * (int64_t)FI(4);
        check(verdict::FxDist2(u, b) == handRaw, "ai2: FxDist2 matches the hand product (Q32.32)");
        const int32_t handNarrow = (int32_t)(handRaw >> fpx::kFrac);   // == 25 << 16
        check(handNarrow == (int32_t)(25 << 16), "ai2: the narrowed d² is 25 * kOne (full resolution)");
        check(ai::ScoreDistanceSq(u, b) == handNarrow,
              "ai2: ScoreDistanceSq == FxDist2 >> kFrac (narrowed, NOT saturated)");
    }

    // ---- (2) GenerateRing is deterministic: same candidates, same order, on two calls ----------------
    {
        const fpx::FxVec3 anchor{FI(10), 0, FI(20)};
        const auto c1 = ai::GenerateRing(anchor, fpx::kOne, ai::kRingDirCount);
        const auto c2 = ai::GenerateRing(anchor, fpx::kOne, ai::kRingDirCount);
        check(c1.size() == (size_t)ai::kRingDirCount, "ai2: GenerateRing yields `count` candidates");
        bool same = (c1.size() == c2.size());
        for (size_t i = 0; same && i < c1.size(); ++i)
            same = (c1[i].x == c2[i].x && c1[i].y == c2[i].y && c1[i].z == c2[i].z);
        check(same, "ai2: GenerateRing is deterministic (byte-identical candidates, same order)");
        // The first candidate is anchor + (+radius, 0) in the table's fixed k=0 direction (dx=64/64=1).
        check(c1[0].x == anchor.x + fpx::kOne && c1[0].z == anchor.z,
              "ai2: GenerateRing k=0 is the +X ring point (fixed direction table)");
        // count clamps to kRingDirCount.
        const auto cClamp = ai::GenerateRing(anchor, fpx::kOne, 999);
        check(cClamp.size() == (size_t)ai::kRingDirCount, "ai2: GenerateRing clamps count to the table size");
    }

    // ---- (3) Reachability: same component + finite path -> 0; different component -> kBigPenalty ------
    {
        // A 5-poly connected row (one component) + an isolated poly (a second component).
        ai::NavScene s = MakeLineScene(5, 100, /*addIsolated=*/true);
        check(s.nComp == 2u, "ai2: synthetic scene has 2 connected components");
        // Agent at poly 0 (voxel 0,0). A candidate near poly 3 (voxel ~300,0) is in the SAME component.
        const fpx::FxVec3 candReach{FI(300), 0, 0};
        check(ai::ScoreNavReachable(candReach, 0, 0, s) == 0,
              "ai2: a candidate in the agent's component + finite path -> reachable (0)");
        // A candidate at the isolated poly's voxel (a DIFFERENT component) -> penalized.
        const fpx::FxVec3 candIso{FI(5 * 100 + 10000), 0, FI(10000)};
        check(ai::ScoreNavReachable(candIso, 0, 0, s) == ai::kBigPenalty,
              "ai2: a candidate in a different nav component -> kBigPenalty (unreachable)");
    }

    // ---- (4) The closest REACHABLE candidate wins; an UNREACHABLE nearest candidate is never chosen ---
    {
        // A hand-built 2-poly scene so a ring candidate genuinely lands on a DIFFERENT component than the
        // agent. navScale=1, radius=1 -> ring candidates sit at voxels like (+1,0) [k=0] and (0,+1) [k=4].
        //   poly 0: the AGENT's component, centroid at voxel (1, 0) — the k=0 ring candidate maps here.
        //   poly 1: an ISOLATED component (no neighbours), centroid at voxel (0, 1) — k=4 maps here.
        ai::NavScene s;
        s.navScale = 1;
        s.polys.assign(2, nav::Poly{});
        for (int e = 0; e < 3; ++e) { s.polys[0].nbr[e] = nav::kNoNeighbour; s.polys[1].nbr[e] = nav::kNoNeighbour; }
        s.cx = {1, 0}; s.cz = {0, 1};   // poly 0 at (1,0); poly 1 at (0,1) — disconnected (no nbr links)
        s.nComp = nav::ConnectedComponents(s.polys, s.comp);
        check(s.nComp == 2u, "ai2: the 2-poly scene is two separate components");

        ai::EqsQuery q;
        q.anchor = fpx::FxVec3{0, 0, 0};
        q.agentVx = 1; q.agentVz = 0;   // the agent sits at poly 0's cell (its component)
        q.radius = fpx::kOne;
        q.count = ai::kRingDirCount;
        // Target AT poly 1 (the isolated cell) so the candidate NEAREST the target is the k=4 ring point,
        // which maps to poly 1 — a DIFFERENT component -> UNREACHABLE. The reachable k=0 must win instead.
        q.target = fpx::FxVec3{0, 0, FI(1)};

        const auto cands = ai::GenerateRing(q.anchor, q.radius, q.count);
        // Sanity: the k=4 candidate (0,+1) is the nearest to the target AND is unreachable.
        check(ai::ScoreNavReachable(cands[4], q.agentVx, q.agentVz, s) == ai::kBigPenalty,
              "ai2: the candidate nearest the target lands on the isolated component (unreachable)");

        const ai::EqsResult r = ai::RunQuery(q, s);
        check(r.bestIndex >= 0, "ai2: RunQuery picks a candidate");
        check(r.bestScore < ai::kBigPenalty,
              "ai2: the chosen candidate is reachable (an unreachable nearest candidate is never chosen)");
        check(ai::ScoreNavReachable(cands[(size_t)r.bestIndex], q.agentVx, q.agentVz, s) == 0,
              "ai2: the winner's reachability score is 0");
    }

    // ---- (5) RunQuery TWO-RUN determinism over the canonical navmesh scene ----------------------------
    {
        const ai::NavScene scene = ai::BuildNavScene();
        check(!scene.polys.empty(), "ai2: BuildNavScene produced a non-empty navmesh");
        const ai::EqsQuery q = ai::BuildAi2Scene(scene);
        const ai::EqsResult r1 = ai::RunQuery(q, scene);
        const ai::EqsResult r2 = ai::RunQuery(q, scene);
        check(r1.bestIndex == r2.bestIndex && r1.bestScore == r2.bestScore,
              "ai2: RunQuery two runs BYTE-IDENTICAL (bestIndex + bestScore)");
        check(r1.bestIndex >= 0, "ai2: the canonical query selects a candidate");
    }

    // ---- (6) A constructed TIE resolves to the LOWEST candidate index ---------------------------------
    {
        // Force EVERY candidate to the SAME position (radius 0 -> the integer ring offset is 0 for every
        // direction) so every candidate's distance² AND reachability are identical -> a pure tie. With all
        // scores equal, the LOWEST index (0) must win (the strict-< update keeps the first). An empty
        // navmesh makes reachability uniformly kBigPenalty too — doubly uniform.
        ai::NavScene empty;          // no polys -> ScoreNavReachable returns kBigPenalty for every candidate
        empty.navScale = 1;
        empty.nComp = nav::ConnectedComponents(empty.polys, empty.comp);

        ai::EqsQuery q;
        q.anchor = fpx::FxVec3{FI(50), 0, FI(50)};
        q.target = fpx::FxVec3{FI(70), 0, FI(50)};   // some target; all candidates coincide so d² is uniform
        q.agentVx = 0; q.agentVz = 0;
        q.radius = 0;                // radius 0 -> every candidate == the anchor -> a genuine full tie
        q.count = ai::kRingDirCount;

        const ai::EqsResult r = ai::RunQuery(q, empty);
        check(r.bestIndex == 0, "ai2: a full tie resolves to the LOWEST candidate index (0)");

        // Confirm it really WAS a tie: every candidate's combined score equals the winner's.
        const auto cands = ai::GenerateRing(q.anchor, q.radius, q.count);
        bool allEqual = (cands.size() == (size_t)ai::kRingDirCount);
        for (const auto& c : cands)
            if (ai::ScoreCandidate(c, q, empty) != r.bestScore) allEqual = false;
        check(allEqual, "ai2: the constructed tie really had equal scores across all candidates");
    }

    // ---- (7) An empty query (count 0) yields bestIndex == -1 (deterministic, no UB) -------------------
    {
        const ai::NavScene scene = ai::BuildNavScene();
        ai::EqsQuery q = ai::BuildAi2Scene(scene);
        q.count = 0;
        const ai::EqsResult r = ai::RunQuery(q, scene);
        check(r.bestIndex == -1, "ai2: an empty candidate set yields bestIndex == -1");
    }

    if (g_fail == 0) std::printf("ai_query_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
