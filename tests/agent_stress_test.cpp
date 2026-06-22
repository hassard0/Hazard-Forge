// Slice DX6 — THE DETERMINISM-STRESS FUZZER — THE CAPSTONE (FLAGSHIP #31 THE AGENT EXPERIENCE,
// hf::game::verdict). DX6 sweeps the rollback/snapshot point over the FIXED canonical command stream
// and asserts that replaying through ANY rollback point recovers the bit-identical authority — a
// bounded, seeded, reproducible fuzzer over the lockstep MOAT (#27), emitting a golden PASS matrix.
//
// What this test PINS (the spec's DX6 proofs):
//   * allCorrected == points — EVERY rollback/snapshot point recovers the authority bit-exactly (the moat).
//   * points == ticks+1 — the inclusive sweep.
//   * DETERMINISM — two stress runs produce byte-identical reports (SerializeStressReport equal).
//   * THE NO-PERTURBATION CONTROL — mispredict==auth -> allDiverged==0 (the diverged flag is meaningful,
//     not always-true) while allCorrected stays == points (the moat holds with no divergence too).
//   * THE REAL PERTURBATION — a flipped command arg -> allDiverged > 0 (the fuzzer fires).
//   * THE LAST POINT — rollbackAt==ticks has diverged==false (zero speculative ticks -> nothing diverges).
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests. verdict.h is APPEND-ONLY (VD1-VD6 + the
// DX5 additions byte-frozen); this exercises only the DX6 append.
#include "game/verdict.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace verdict = hf::game::verdict;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    // Build the FIXED canonical world + the REAL (perturbed) mispredict stream.
    verdict::VerdictWorld world0;
    const verdict::CanonicalReplay cr = verdict::BuildCanonicalReplay(world0);
    const verdict::VerdictSnapshot world0Snap = verdict::SnapshotWorld(world0);
    const std::vector<verdict::Command> mispredict = verdict::PerturbCanonicalStream(cr.stream);
    check(!mispredict.empty(), "perturb precondition: mispredict stream non-empty");
    // The perturbation actually changed the stream (a real divergence-causing flip).
    bool perturbed = (mispredict.size() == cr.stream.size());
    if (perturbed) {
        bool anyDiff = false;
        for (size_t i = 0; i < cr.stream.size(); ++i)
            if (mispredict[i].arg.x != cr.stream[i].arg.x) anyDiff = true;
        perturbed = anyDiff;
    }
    check(perturbed, "PerturbCanonicalStream actually perturbs the stream (a real divergence)");

    // === THE SWEEP: RunDeterminismStress over the canonical scene. ===
    const verdict::StressReport report =
        verdict::RunDeterminismStress(world0Snap, cr.params, cr.stream, mispredict, cr.ticks);

    // points == ticks+1 (inclusive sweep); perPoint has exactly that many entries.
    check(report.points == cr.ticks + 1u, "points == ticks+1 (inclusive sweep)");
    check(report.perPoint.size() == report.points, "perPoint count == points");

    // === THE MOAT: allCorrected == points (every rollback point recovers the authority). ===
    check(report.allCorrected == report.points,
          "allCorrected == points (EVERY snapshot point recovers authority bit-exactly)");
    {
        bool everyCorrected = true;
        for (const verdict::StressPoint& p : report.perPoint)
            if (!p.corrected) everyCorrected = false;
        check(everyCorrected, "every perPoint.corrected == true");
    }
    // Each point's rollbackAt is the swept index (0..ticks in order).
    {
        bool ordered = true;
        for (uint32_t i = 0; i < report.perPoint.size(); ++i)
            if (report.perPoint[i].rollbackAt != i) ordered = false;
        check(ordered, "perPoint[i].rollbackAt == i (the 0..ticks sweep order)");
    }

    // === THE REAL PERTURBATION: the fuzzer fires — allDiverged > 0. ===
    check(report.allDiverged > 0u, "real perturbation -> allDiverged > 0 (the fuzzer fires)");
    check(report.allDiverged < report.points,
          "allDiverged < points (the last point has zero speculative ticks)");

    // === THE LAST POINT: rollbackAt==ticks -> diverged==false (no speculative ticks). ===
    check(!report.perPoint.back().diverged,
          "last point (rollbackAt==ticks) has diverged==false (zero speculative ticks)");
    check(report.perPoint.back().corrected,
          "last point still corrected==true (the moat holds at the trivial point)");

    // === DETERMINISM: two stress runs produce byte-identical reports. ===
    {
        verdict::VerdictWorld w2;
        const verdict::CanonicalReplay cr2 = verdict::BuildCanonicalReplay(w2);
        const verdict::VerdictSnapshot snap2 = verdict::SnapshotWorld(w2);
        const std::vector<verdict::Command> mis2 = verdict::PerturbCanonicalStream(cr2.stream);
        const verdict::StressReport report2 =
            verdict::RunDeterminismStress(snap2, cr2.params, cr2.stream, mis2, cr2.ticks);
        check(verdict::SerializeStressReport(report) == verdict::SerializeStressReport(report2),
              "two stress runs byte-identical (determinism)");
        check(report2.allCorrected == report2.points && report2.allDiverged == report.allDiverged,
              "two stress runs agree on allCorrected/allDiverged");
    }

    // === THE NO-PERTURBATION CONTROL: mispredict==auth -> allDiverged==0 (the flag is meaningful). ===
    {
        const verdict::StressReport control =
            verdict::RunDeterminismStress(world0Snap, cr.params, cr.stream, cr.stream, cr.ticks);
        check(control.allDiverged == 0u,
              "no-perturbation control (mispredict==auth) -> allDiverged == 0 (the flag is meaningful)");
        check(control.allCorrected == control.points,
              "control still allCorrected == points (the moat holds with no divergence)");
        bool noneDiverged = true;
        for (const verdict::StressPoint& p : control.perPoint)
            if (p.diverged) noneDiverged = false;
        check(noneDiverged, "control: every perPoint.diverged == false");
    }

    // === SerializeStressReport is a pure function (re-serializing the SAME report is byte-equal). ===
    check(verdict::SerializeStressReport(report) == verdict::SerializeStressReport(report),
          "SerializeStressReport pure (same report -> byte-equal text)");

    if (g_fail == 0) std::printf("agent_stress_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
