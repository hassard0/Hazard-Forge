// Slice IK2 — Deterministic IK Control-Rig: THE TWO-BONE LAW-OF-COSINES LIMB SOLVE (the 2nd slice of
// FLAGSHIP #32). The pure-CPU contract test for engine/anim/ik.h::TwoBoneSolve / EndEffector (the
// law-of-cosines limb solver the GPU shaders/ik_twobone.comp copies VERBATIM + proves bit-identical).
// Pure C++ (header-only, no device, no backend symbols). Namespace hf::anim::ik.
//
// What this test PINS (the contracts the GPU solve + later IK slices build on):
//   * determinism: TwoBoneSolve is a pure function -> two calls byte-identical.
//   * the elbow FOLDS: a near (but reachable) target -> a bent elbow (aElbow away from 0 AND from pi).
//   * end-effector residual within a documented LSB band over a reachable sweep (forward-kinematics).
//   * over-extended (d >= lenU+lenL): aElbow ~= 0 (straight limb) AND the end-effector lies on the
//     root->target ray (the deterministic clamp).
//   * un-reachable-near (d < |lenU-lenL|): the deterministic reach clamp to dMin (no NaN, no divergence).
//   * the pole controls the bend SIDE: flipping the pole flips the elbow to the other half-plane.
//
// The HONEST within-band caveat: FxAcosLut carries IK1's documented LSB band, so the end-effector reaches
// the target within a BOUNDED, DETERMINISTIC residual (NOT zero). The headline is DETERMINISM + cross-
// platform bit-identity, NOT analytic reach. The band magnitude is reported + checked here.
#include "anim/ik.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace ik  = hf::anim::ik;
namespace fpx = hf::sim::fpx;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static int32_t iabs32(int32_t v) { return v < 0 ? -v : v; }

// Q16.16 helpers (host-side, for the test scene only).
static ik::fx F(double v) { return (ik::fx)std::llround(v * (double)ik::kOne); }
static int32_t Residual(const ik::FxVec3& a, const ik::FxVec3& b) {
    return fpx::FxLength(ik::FxVec3{a.x - b.x, a.y - b.y, a.z - b.z});
}

int main() {
    HF_TEST_MAIN_INIT();
    const ik::fx lenU = F(1.0);   // upper bone 1.0
    const ik::fx lenL = F(0.8);   // lower bone 0.8 (asymmetric -> dMin = 0.2)
    const ik::FxVec3 root{0, 0, 0};
    const ik::FxVec3 pole{0, 0, ik::kOne};   // +Z bend hint (the limb bends in the XY plane)

    // ===== determinism: TwoBoneSolve is a pure function (two calls byte-identical) =====
    {
        const ik::FxVec3 tgt{F(1.2), F(0.3), 0};
        const ik::TwoBoneResult r1 = ik::TwoBoneSolve(root, tgt, pole, lenU, lenL);
        const ik::TwoBoneResult r2 = ik::TwoBoneSolve(root, tgt, pole, lenU, lenL);
        const bool same = r1.qUpper.x == r2.qUpper.x && r1.qUpper.y == r2.qUpper.y &&
                          r1.qUpper.z == r2.qUpper.z && r1.qUpper.w == r2.qUpper.w &&
                          r1.qLower.x == r2.qLower.x && r1.qLower.y == r2.qLower.y &&
                          r1.qLower.z == r2.qLower.z && r1.qLower.w == r2.qLower.w &&
                          r1.aRoot == r2.aRoot && r1.aElbow == r2.aElbow && r1.reach == r2.reach;
        check(same, "TwoBoneSolve is deterministic (two calls byte-identical)");
    }

    // ===== the elbow FOLDS: a near (reachable) target -> a bent elbow (aElbow in (band, pi-band)) =====
    {
        // A target at reach ~1.0 (well inside [0.2, 1.8]) -> the elbow must bend noticeably.
        const ik::FxVec3 tgt{F(0.9), F(0.4), 0};   // |tgt| ~= 0.985
        const ik::TwoBoneResult r = ik::TwoBoneSolve(root, tgt, pole, lenU, lenL);
        // aElbow is the INTERIOR angle: a folded limb has aElbow well below pi and above 0.
        check(r.aElbow > F(0.3), "near target -> elbow folded (aElbow > ~0.3 rad)");
        check(r.aElbow < ik::kPi - F(0.3), "near target -> elbow not straight (aElbow < pi-0.3)");
    }

    // ===== end-effector residual within a documented band over a reachable sweep (forward-kinematics) =====
    {
        // The documented residual band: the FxAcosLut LSB band propagated through axis-angle + FK. A few
        // hundred LSB (~0.01 world units). The test reports the observed max; the band is generous + beaten.
        const int32_t kResidualBand = F(0.02);   // ~1310 LSB; the documented bounded-deterministic reach band
        int32_t maxResidual = 0;
        const int N = 64;
        // Sweep targets across a reachable arc (reach in [0.4, 1.7], angle across the +X half).
        for (int k = 0; k < N; ++k) {
            const double ang = -1.2 + 2.4 * (double)k / (double)(N - 1);   // [-1.2, 1.2] rad about +X
            const double reach = 0.4 + 1.3 * (double)k / (double)(N - 1);  // [0.4, 1.7] (inside [0.2,1.8])
            const ik::FxVec3 tgt{F(reach * std::cos(ang)), F(reach * std::sin(ang)), 0};
            const ik::TwoBoneResult r = ik::TwoBoneSolve(root, tgt, pole, lenU, lenL);
            const ik::FxVec3 ee = ik::EndEffector(root, tgt, pole, lenU, lenL, r.qUpper, r.qLower);
            const int32_t res = Residual(ee, tgt);
            if (res > maxResidual) maxResidual = res;
        }
        std::printf("ik_twobone_test: reachable-sweep maxResidual=%d LSB (band=%d)\n",
                    maxResidual, kResidualBand);
        check(maxResidual <= kResidualBand, "end-effector within the documented LUT-residual band");
    }

    // ===== over-extended (d >= lenU+lenL): straight limb (bend ~= 0) + end-effector on the root->target ray =====
    // NOTE the angle convention: aElbow is the INTERIOR elbow angle (acos result). A STRAIGHT limb has the
    // bones colinear -> interior angle aElbow == pi -> the BEND (pi - aElbow) == 0. The spec's "elbowAngle~0"
    // refers to that bend. Folded (near) -> aElbow == 0. (The lower-bone LOCAL rotation is aElbow - pi, so
    // straight -> 0 rotation -> the limb points straight at the target.)
    {
        const ik::FxVec3 tgt{F(3.0), F(1.0), 0};   // |tgt| ~= 3.16 >> 1.8 -> clamped to dMax
        const ik::TwoBoneResult r = ik::TwoBoneSolve(root, tgt, pole, lenU, lenL);
        const ik::fx bend = (ik::fx)(ik::kPi - r.aElbow);
        check(bend <= F(0.02), "over-extended -> straight limb (bend = pi - aElbow ~= 0)");
        check(r.reach == (ik::fx)(lenU + lenL), "over-extended -> reach clamped to dMax");
        // The end-effector lies on the root->target ray at distance ~lenU+lenL (the straight limb points
        // at the target). Check the end-effector direction matches the axis (cross ~ 0).
        const ik::FxVec3 ee = ik::EndEffector(root, tgt, pole, lenU, lenL, r.qUpper, r.qLower);
        const ik::FxVec3 axis = fpx::FxNormalize(ik::FxVec3{tgt.x, tgt.y, tgt.z});
        const ik::FxVec3 eeDir = fpx::FxNormalize(ee);
        const ik::FxVec3 cr = ik::FxCross(axis, eeDir);
        const int32_t off = fpx::FxLength(cr);
        check(off <= F(0.03), "over-extended -> end-effector on the root->target ray");
    }

    // ===== un-reachable-near (d < |lenU-lenL|): the deterministic reach clamp to dMin =====
    {
        const ik::fx dMin = (ik::fx)(lenU - lenL);   // 0.2
        const ik::FxVec3 tgt{F(0.05), F(0.02), 0};   // |tgt| ~= 0.054 < 0.2 -> clamped UP to dMin
        const ik::TwoBoneResult r = ik::TwoBoneSolve(root, tgt, pole, lenU, lenL);
        check(r.reach == dMin, "un-reachable-near -> reach clamped UP to dMin");
        // aElbow ~= 0 (the bones fold FLAT/antiparallel at the near boundary d=|lenU-lenL|), no NaN/garbage.
        check(r.aElbow <= F(0.05), "un-reachable-near -> elbow fully folded (aElbow ~= 0)");
        // Determinism survives the degenerate near case.
        const ik::TwoBoneResult r2 = ik::TwoBoneSolve(root, tgt, pole, lenU, lenL);
        check(r.qUpper.x == r2.qUpper.x && r.qLower.w == r2.qLower.w,
              "un-reachable-near solve deterministic");
    }

    // ===== the pole controls the bend SIDE: flipping the pole flips the elbow half-plane =====
    {
        const ik::FxVec3 tgt{F(1.0), 0, 0};               // straight ahead on +X
        const ik::FxVec3 poleA{0, 0, ik::kOne};           // +Z hint
        const ik::FxVec3 poleB{0, 0, (ik::fx)(-ik::kOne)};// -Z hint (opposite side)
        const ik::TwoBoneResult ra = ik::TwoBoneSolve(root, tgt, poleA, lenU, lenL);
        const ik::TwoBoneResult rb = ik::TwoBoneSolve(root, tgt, poleB, lenU, lenL);
        const ik::FxVec3 elbowA = ik::ElbowPos(root, tgt, lenU, ra.qUpper);
        const ik::FxVec3 elbowB = ik::ElbowPos(root, tgt, lenU, rb.qUpper);
        // The limb bends in the plane spanned by axis (+X) + pole (+/-Z), so the elbow lifts to opposite
        // Z signs when the pole flips (the bend-plane normal n = axis x pole flips sign).
        check((elbowA.z > 0 && elbowB.z < 0) || (elbowA.z < 0 && elbowB.z > 0),
              "flipping the pole flips the elbow to the opposite half-plane");
        // The interior angle is the SAME (the geometry is mirrored, not changed).
        check(iabs32(ra.aElbow - rb.aElbow) <= 4, "flipping the pole preserves the elbow interior angle");
    }

    if (g_fail == 0) std::printf("ik_twobone_test: ALL PASS\n");
    else std::printf("ik_twobone_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
