// Slice AI3 — Deterministic AI: LINE-OF-SIGHT / PERCEPTION (the feasibility CRUX) — the 3rd slice of the
// DETERMINISTIC AI flagship (GitHub issue #28, hf::ai). ai.h is APPEND-ONLY (AI1/AI2 byte-frozen); AI3 adds
// the PURE-INTEGER segment-vs-AABB line-of-sight (no float, no division — every `t` is a fraction num/den
// compared by cross-multiplication, the navmesh.h PointInTriXZ discipline).
//
// What this test PINS (the determinism contract + the slab-test semantics + the boundary/corner rule):
//   * LineOfSight TWO-RUN determinism (the same query yields the same bool twice — integer by construction).
//   * A ray straight THROUGH a blocker -> false (occluded).
//   * A ray MISSING all blockers -> true (visible).
//   * A ray that ENDS before reaching a blocker -> true (the [0,1] clamp; the box is beyond the segment).
//   * A ray STARTING strictly inside a blocker -> false (the strict-interior rule).
//   * An axis-parallel ray GRAZING a box face -> the documented boundary rule (touching is NOT blocked),
//     consistent on two runs.
//   * A ray to a target exactly at a box CORNER -> the documented corner rule (grazing a corner is NOT
//     blocked).
//   * WritePerception writes canSeeTarget + (when visible) the last-seen cell, leaving it unchanged when
//     occluded.
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "ai/ai.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace ai  = hf::ai;
namespace fpx = hf::sim::fpx;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A Q16.16 helper: integer world units -> fx.
static fpx::fx FI(int v) { return (fpx::fx)((int64_t)v * (int64_t)fpx::kOne); }
// Half-unit helper: v + 0.5 world units in Q16.16.
static fpx::fx FH(int v) { return (fpx::fx)((((int64_t)v << 1) + 1) << (fpx::kFrac - 1)); }

static fpx::FxVec3 V(fpx::fx x, fpx::fx z) { return fpx::FxVec3{x, 0, z}; }

// A single box [x0,z0]-[x1,z1] in WHOLE world units (Q16.16).
static ai::AiBlocker Box(int x0, int z0, int x1, int z1) {
    ai::AiBlocker b;
    b.min = fpx::FxVec3{FI(x0), 0, FI(z0)};
    b.max = fpx::FxVec3{FI(x1), FI(2), FI(z1)};
    return b;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- (1) A ray straight THROUGH a blocker -> false; a ray MISSING all -> true --------------------
    {
        // One box covering [4,4]-[6,6]. A horizontal ray at z=5 from x=0 to x=10 passes straight through it.
        const ai::AiBlocker boxes[] = { Box(4, 4, 6, 6) };
        const bool through = ai::LineOfSight(V(FI(0), FI(5)), V(FI(10), FI(5)), boxes, 1);
        check(through == false, "ai3: a ray straight through a blocker is occluded (false)");
        // A parallel ray at z=9 (above the box) misses entirely.
        const bool miss = ai::LineOfSight(V(FI(0), FI(9)), V(FI(10), FI(9)), boxes, 1);
        check(miss == true, "ai3: a ray missing all blockers is visible (true)");
        // No blockers at all -> always visible.
        const bool none = ai::LineOfSight(V(FI(0), FI(5)), V(FI(10), FI(5)), nullptr, 0);
        check(none == true, "ai3: zero blockers -> always visible");
    }

    // ---- (2) A ray that ENDS before reaching a blocker -> true (the [0,1] clamp) ----------------------
    {
        // Box at [8,4]-[10,6]. A ray from x=0 to x=4 (z=5) is aimed AT the box's row but STOPS at x=4,
        // well before the box at x>=8 -> the box is beyond the segment -> visible.
        const ai::AiBlocker boxes[] = { Box(8, 4, 10, 6) };
        const bool ends = ai::LineOfSight(V(FI(0), FI(5)), V(FI(4), FI(5)), boxes, 1);
        check(ends == true, "ai3: a ray ending before a blocker is visible (the [0,1] clamp)");
        // Extending the same ray PAST the box (x=12) now penetrates it -> occluded (sanity for the clamp).
        const bool past = ai::LineOfSight(V(FI(0), FI(5)), V(FI(12), FI(5)), boxes, 1);
        check(past == false, "ai3: extending the ray through the box occludes it (clamp sanity)");
    }

    // ---- (3) A ray STARTING strictly inside a blocker -> false (the strict-interior rule) -------------
    {
        const ai::AiBlocker boxes[] = { Box(4, 4, 8, 8) };
        // Start at the box CENTER (6,6) -> strictly inside -> occluded regardless of direction.
        const bool inside = ai::LineOfSight(V(FI(6), FI(6)), V(FI(20), FI(20)), boxes, 1);
        check(inside == false, "ai3: a ray starting strictly inside a blocker is occluded");
        // A degenerate (zero-length) ray at the box center is also occluded (the from==to interior rule).
        const bool degIn = ai::LineOfSight(V(FI(6), FI(6)), V(FI(6), FI(6)), boxes, 1);
        check(degIn == false, "ai3: a zero-length ray strictly inside a blocker is occluded");
        // A degenerate ray OUTSIDE every box sees itself.
        const bool degOut = ai::LineOfSight(V(FI(0), FI(0)), V(FI(0), FI(0)), boxes, 1);
        check(degOut == true, "ai3: a zero-length ray outside all blockers is visible");
    }

    // ---- (4) Axis-parallel ray GRAZING a box face -> the documented rule: touching is NOT blocked -----
    {
        // Box [4,4]-[6,6]. A horizontal ray exactly along the box's TOP face (z=6) grazes it.
        const ai::AiBlocker boxes[] = { Box(4, 4, 6, 6) };
        const bool grazeTop = ai::LineOfSight(V(FI(0), FI(6)), V(FI(10), FI(6)), boxes, 1);
        check(grazeTop == true, "ai3: an axis-parallel ray grazing a box face is NOT blocked (boundary rule)");
        // The same along the BOTTOM face (z=4) grazes too.
        const bool grazeBot = ai::LineOfSight(V(FI(0), FI(4)), V(FI(10), FI(4)), boxes, 1);
        check(grazeBot == true, "ai3: grazing the opposite face is also NOT blocked (consistent rule)");
        // A vertical ray exactly along the LEFT face (x=4) grazes too.
        const bool grazeLeft = ai::LineOfSight(V(FI(4), FI(0)), V(FI(4), FI(10)), boxes, 1);
        check(grazeLeft == true, "ai3: a vertical ray grazing the left face is NOT blocked");
        // A ray ONE unit inside the face (z=5) genuinely penetrates -> occluded (the face rule is exact).
        const bool insideFace = ai::LineOfSight(V(FI(0), FI(5)), V(FI(10), FI(5)), boxes, 1);
        check(insideFace == false, "ai3: a ray one unit inside the face penetrates (occluded)");
    }

    // ---- (5) A ray to a target exactly at a box CORNER -> grazing a corner is NOT blocked -------------
    {
        // Box [4,4]-[6,6]. A diagonal ray from (0,0) heading to the box's near corner (4,4) ends exactly
        // at the corner -> a measure-zero touch -> NOT blocked (the corner rule).
        const ai::AiBlocker boxes[] = { Box(4, 4, 6, 6) };
        const bool toCorner = ai::LineOfSight(V(FI(0), FI(0)), V(FI(4), FI(4)), boxes, 1);
        check(toCorner == true, "ai3: a ray ending exactly at a box corner is NOT blocked (corner rule)");
        // A diagonal that clips the corner and continues INTO the box (to its center) IS blocked.
        const bool intoBox = ai::LineOfSight(V(FI(0), FI(0)), V(FI(5), FI(5)), boxes, 1);
        check(intoBox == false, "ai3: a diagonal continuing into the box interior is occluded");
        // A diagonal that just TOUCHES the far corner (6,6) from outside, grazing the corner, is NOT blocked
        // when it ends exactly there.
        const bool toFarCorner = ai::LineOfSight(V(FI(0), FI(12)), V(FI(6), FI(6)), boxes, 1);
        check(toFarCorner == true, "ai3: a ray ending exactly at the far corner is NOT blocked");
    }

    // ---- (6) Two-run determinism over the canonical scene (every cell) --------------------------------
    {
        const ai::Ai3Scene scene = ai::BuildAi3Scene();
        std::vector<uint8_t> r1, r2;
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<uint8_t>& out = (pass == 0) ? r1 : r2;
            out.reserve((size_t)scene.gridW * scene.gridH);
            for (int z = 0; z < scene.gridH; ++z)
                for (int x = 0; x < scene.gridW; ++x) {
                    const fpx::FxVec3 cell = V(ai::CellCenterWorld(x), ai::CellCenterWorld(z));
                    const bool vis = ai::LineOfSight(scene.agent, cell,
                                                     scene.blockers.data(), (int)scene.blockers.size());
                    out.push_back(vis ? 1u : 0u);
                }
        }
        check(r1 == r2, "ai3: two visibility rasters are BYTE-IDENTICAL (deterministic)");
        // The scene must produce BOTH visible and occluded cells (a non-trivial shadow exists).
        int vis = 0, occ = 0;
        for (uint8_t b : r1) { if (b) ++vis; else ++occ; }
        check(vis > 0 && occ > 0, "ai3: the canonical scene yields both visible AND occluded cells");
        // The agent's own cell is visible (it sees itself; it is not inside a blocker).
        // The target is occluded from the agent by the central wall (the headline occlusion).
        const bool seeTarget = ai::LineOfSight(scene.agent, scene.target,
                                               scene.blockers.data(), (int)scene.blockers.size());
        check(seeTarget == false, "ai3: the target is occluded from the agent by the central wall");
    }

    // ---- (7) WritePerception writes canSeeTarget + last-seen cell; unchanged when occluded ------------
    {
        ai::Blackboard bb;
        ai::WritePerception(bb, true, 7, 9);
        check(bb.Get(ai::kBbCanSeeTarget) == 1, "ai3: WritePerception sets canSeeTarget=1 when visible");
        check(bb.Get(ai::kBbLastSeenCellX) == 7 && bb.Get(ai::kBbLastSeenCellZ) == 9,
              "ai3: WritePerception records the last-seen cell when visible");
        // Now occluded: canSeeTarget -> 0 but the last-seen cell is LEFT UNCHANGED (the memory).
        ai::WritePerception(bb, false, 99, 99);
        check(bb.Get(ai::kBbCanSeeTarget) == 0, "ai3: WritePerception sets canSeeTarget=0 when occluded");
        check(bb.Get(ai::kBbLastSeenCellX) == 7 && bb.Get(ai::kBbLastSeenCellZ) == 9,
              "ai3: the last-seen cell is preserved when occluded (last-known-position memory)");
    }

    // ---- (8) Multi-blocker: visible iff NO blocker occludes (a clear lane between two boxes) ----------
    {
        // Two boxes leaving a gap at z in (6,8): box A [4,4]-[6,6], box B [4,8]-[6,10]. A horizontal ray at
        // z=7 threads the gap (visible); at z=5 it hits A (occluded); at z=9 it hits B (occluded).
        const ai::AiBlocker boxes[] = { Box(4, 4, 6, 6), Box(4, 8, 6, 10) };
        check(ai::LineOfSight(V(FI(0), FI(7)), V(FI(10), FI(7)), boxes, 2) == true,
              "ai3: a ray threading the gap between two blockers is visible");
        check(ai::LineOfSight(V(FI(0), FI(5)), V(FI(10), FI(5)), boxes, 2) == false,
              "ai3: a ray hitting the first blocker is occluded");
        check(ai::LineOfSight(V(FI(0), FI(9)), V(FI(10), FI(9)), boxes, 2) == false,
              "ai3: a ray hitting the second blocker is occluded");
    }

    // ---- (9) Half-cell coordinates (the showcase uses cell CENTERS) behave correctly -----------------
    {
        // Box [4,4]-[6,6]. A ray from cell-center (0.5,5.5) to (10.5,5.5) crosses the box interior.
        const ai::AiBlocker boxes[] = { Box(4, 4, 6, 6) };
        check(ai::LineOfSight(V(FH(0), FH(5)), V(FH(10), FH(5)), boxes, 1) == false,
              "ai3: a half-cell-center ray through the box is occluded");
        // A half-cell ray below the box (z=2.5) misses.
        check(ai::LineOfSight(V(FH(0), FH(2)), V(FH(10), FH(2)), boxes, 1) == true,
              "ai3: a half-cell-center ray below the box is visible");
    }

    if (g_fail == 0) std::printf("ai_los_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
