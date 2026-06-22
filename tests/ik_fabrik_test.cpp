// Slice IK3 — Deterministic IK Control-Rig: FABRIK N-BONE CHAIN + LOOK-AT (the 3rd slice of FLAGSHIP #32).
// The pure-CPU contract test for engine/anim/ik.h::FabrikSolve / LookAtRotation (the n-bone chain solver
// the GPU shaders/ik_fabrik.comp copies VERBATIM + proves bit-identical, plus the shortest-arc look-at).
// Pure C++ (header-only, no device, no backend symbols). Namespace hf::anim::ik.
//
// What this test PINS (the contracts the GPU solve + later IK slices build on):
//   * determinism: FabrikSolve is a pure function -> two runs byte-identical.
//   * segment-length preservation: max drift <= a documented LSB band over a reachable sweep (the chain
//     stays connected — drift = FxNormalize LSB only).
//   * end-effector residual within a documented band over a reachable sweep (the bounded iterative
//     Gauss-Seidel residual — NOT zero, the honest within-band caveat).
//   * unreachable target (> Sum(len)): a STRAIGHTENED chain pointing at the target (segments colinear).
//   * iters convergence: more iters -> the residual is non-increasing within band (monotone-ish).
//   * LookAtRotation aims: FxRotate(LookAtRotation(fwd,to), fwd) . to -> kOne band; identity for fwd~to;
//     a deterministic pi-flip for fwd~-to.
//
// The HONEST within-band caveat: FABRIK's end-effector reaches the target within a BOUNDED, DETERMINISTIC
// residual (the Gauss-Seidel residual + FxNormalize LSB — NOT zero). The headline is DETERMINISM + cross-
// platform bit-identity, NOT analytic convergence. The band magnitudes are reported + checked here.
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

// Q16.16 helper (host-side, for the test scene only).
static ik::fx F(double v) { return (ik::fx)std::llround(v * (double)ik::kOne); }

// BuildRestChain(root, segLen, K): a K-joint chain laid out straight along +X from root with the fixed
// per-segment length segLen, lengths captured. Matches the showcase rest init.
static ik::IkChainN BuildRestChain(const ik::FxVec3& root, ik::fx segLen, int K) {
    ik::IkChainN c{};
    c.count = K;
    c.pos[0] = root;
    for (int i = 1; i < K; ++i) c.pos[i] = ik::FxVec3{c.pos[i - 1].x + segLen, c.pos[i - 1].y, c.pos[i - 1].z};
    ik::CaptureChainLengths(c);
    return c;
}

static bool ChainEqual(const ik::IkChainN& a, const ik::IkChainN& b) {
    if (a.count != b.count) return false;
    for (int i = 0; i < a.count; ++i)
        if (a.pos[i].x != b.pos[i].x || a.pos[i].y != b.pos[i].y || a.pos[i].z != b.pos[i].z) return false;
    return true;
}

int main() {
    HF_TEST_MAIN_INIT();
    const int K = 6;                 // a 6-bone chain (the showcase shape)
    const ik::fx segLen = F(0.5);    // each segment 0.5 -> total reach 2.5
    const ik::FxVec3 root{0, 0, 0};
    const int iters = 12;

    // ===== determinism: FabrikSolve is a pure function (two runs byte-identical) =====
    {
        const ik::FxVec3 tgt{F(1.6), F(0.8), 0};
        ik::IkChainN c1 = BuildRestChain(root, segLen, K);
        ik::IkChainN c2 = BuildRestChain(root, segLen, K);
        ik::FabrikSolve(c1, root, tgt, iters);
        ik::FabrikSolve(c2, root, tgt, iters);
        check(ChainEqual(c1, c2), "FabrikSolve is deterministic (two runs byte-identical)");
    }

    // ===== segment-length preservation: max drift <= band over a reachable sweep =====
    {
        // The documented length-drift band: the FxNormalize LSB propagated through the position updates.
        // A few LSB per joint per iteration -> a small absolute Q16.16 drift. 0.004 world units is generous.
        const ik::fx kLenDriftBand = F(0.004);   // ~262 LSB
        ik::fx maxDrift = 0;
        const int N = 40;
        for (int t = 0; t < N; ++t) {
            const double frac = (double)t / (double)(N - 1);
            const double ang   = -1.0 + 2.0 * frac;       // [-1, 1] rad about +X
            const double reach = 0.6 + 1.6 * frac;        // [0.6, 2.2] (inside totalLen 2.5)
            const ik::FxVec3 tgt{F(reach * std::cos(ang)), F(reach * std::sin(ang)), 0};
            ik::IkChainN c = BuildRestChain(root, segLen, K);
            ik::FabrikSolve(c, root, tgt, iters);
            const ik::fx drift = ik::ChainMaxLenDrift(c);
            if (drift > maxDrift) maxDrift = drift;
        }
        std::printf("ik_fabrik_test: reachable-sweep maxLenDriftLsb=%d (band=%d)\n", maxDrift, kLenDriftBand);
        check(maxDrift <= kLenDriftBand, "segment lengths preserved within the documented drift band");
    }

    // ===== end-effector residual within a documented band over a reachable sweep =====
    {
        // The documented residual band: the bounded iterative Gauss-Seidel residual + FxNormalize LSB over
        // REACHABLE targets (reach < totalLen). With iters=12 the residual converges well below 0.02.
        const ik::fx kResidualBand = F(0.02);
        ik::fx maxResidual = 0;
        const int N = 40;
        for (int t = 0; t < N; ++t) {
            const double frac = (double)t / (double)(N - 1);
            const double ang   = -1.0 + 2.0 * frac;
            const double reach = 0.6 + 1.6 * frac;        // all <= 2.2 < 2.5 (reachable)
            const ik::FxVec3 tgt{F(reach * std::cos(ang)), F(reach * std::sin(ang)), 0};
            ik::IkChainN c = BuildRestChain(root, segLen, K);
            ik::FabrikSolve(c, root, tgt, iters);
            const ik::fx res = ik::ChainResidual(c, tgt);
            if (res > maxResidual) maxResidual = res;
        }
        std::printf("ik_fabrik_test: reachable-sweep maxResidual=%d LSB (band=%d)\n",
                    maxResidual, kResidualBand);
        check(maxResidual <= kResidualBand, "end-effector within the documented residual band");
    }

    // ===== unreachable target (> Sum(len)) -> a straightened chain pointing at the target =====
    {
        // A target far beyond totalLen=2.5: FABRIK straightens the chain along the root->target ray. Each
        // segment direction should match the axis (cross ~ 0) -> the chain is colinear toward the target.
        const ik::FxVec3 tgt{F(4.0), F(2.0), 0};   // |tgt| ~ 4.47 >> 2.5
        ik::IkChainN c = BuildRestChain(root, segLen, K);
        ik::FabrikSolve(c, root, tgt, iters);
        const ik::FxVec3 axis = fpx::FxNormalize(ik::FxVec3{tgt.x - root.x, tgt.y - root.y, tgt.z - root.z});
        ik::fx maxOff = 0;
        for (int i = 0; i + 1 < c.count; ++i) {
            const ik::FxVec3 seg = fpx::FxNormalize(ik::FxVec3{c.pos[i + 1].x - c.pos[i].x,
                                                               c.pos[i + 1].y - c.pos[i].y,
                                                               c.pos[i + 1].z - c.pos[i].z});
            const ik::fx off = fpx::FxLength(ik::FxCross(axis, seg));   // ~0 when colinear
            if (off > maxOff) maxOff = off;
        }
        std::printf("ik_fabrik_test: unreachable maxSegColinearOff=%d LSB\n", maxOff);
        check(maxOff <= F(0.02), "unreachable target -> straightened chain (segments colinear toward target)");
        // The end-effector lies at ~totalLen along the ray -> residual ~ |tgt| - totalLen (large, by design).
        const ik::fx res = ik::ChainResidual(c, tgt);
        check(res > F(1.0), "unreachable target -> end-effector falls short (residual large by design)");
    }

    // ===== iters convergence: more iters -> residual non-increasing within band (reachable) =====
    {
        const ik::FxVec3 tgt{F(1.8), F(1.0), 0};   // |tgt| ~ 2.06 < 2.5 (reachable)
        ik::fx prevRes = 0;
        bool havePrev = false;
        bool monotone = true;
        for (int it = 1; it <= 16; ++it) {
            ik::IkChainN c = BuildRestChain(root, segLen, K);
            ik::FabrikSolve(c, root, tgt, it);
            const ik::fx res = ik::ChainResidual(c, tgt);
            // Allow a small LSB band of non-monotonicity from the integer FxNormalize rounding: the
            // residual converges toward 0 but can wobble a few hundred LSB between iters as the chain
            // snaps to the target (the Gauss-Seidel iterative residual is monotone-ish, NOT strictly).
            if (havePrev && res > prevRes + F(0.01)) monotone = false;
            prevRes = res;
            havePrev = true;
        }
        std::printf("ik_fabrik_test: iters-convergence finalResidual=%d LSB (monotone=%d)\n",
                    prevRes, (int)monotone);
        check(monotone, "more iters -> residual non-increasing within band");
        check(prevRes <= F(0.02), "converged residual within band at 16 iters");
    }

    // ===== LookAtRotation aims: FxRotate(LookAtRotation(fwd,to), fwd) . to -> kOne band =====
    {
        const ik::fx kAimBand = F(0.01);   // dot within 0.01 of kOne
        // A sweep of unit fwd/to pairs (the moving-target aim).
        ik::fx minDot = (ik::fx)0x7fffffff;
        const int N = 24;
        for (int t = 0; t < N; ++t) {
            const double fa = 2.0 * 3.14159265358979 * (double)t / (double)N;
            const double ta = fa + 0.7;   // a fixed offset (avoid the exact 0/pi degenerate beats here)
            const ik::FxVec3 fwd = fpx::FxNormalize(ik::FxVec3{F(std::cos(fa)), F(std::sin(fa)), F(0.3)});
            const ik::FxVec3 to  = fpx::FxNormalize(ik::FxVec3{F(std::cos(ta)), F(std::sin(ta)), F(-0.2)});
            const ik::FxQuat q = ik::LookAtRotation(fwd, to);
            const ik::FxVec3 aimed = fpx::FxRotate(q, fwd);
            const ik::fx dot = ik::FxDotV(aimed, to);   // -> kOne when fwd rotated onto to
            if (dot < minDot) minDot = dot;
        }
        std::printf("ik_fabrik_test: look-at minAimDot=%d (kOne=%d, band=%d)\n",
                    minDot, ik::kOne, kAimBand);
        check(minDot >= ik::kOne - kAimBand, "LookAtRotation aims fwd onto to (dot -> kOne band)");
    }

    // ===== LookAtRotation identity for fwd~to =====
    {
        const ik::FxVec3 fwd = fpx::FxNormalize(ik::FxVec3{F(0.6), F(0.5), F(0.2)});
        const ik::FxQuat q = ik::LookAtRotation(fwd, fwd);
        check(q.x == 0 && q.y == 0 && q.z == 0 && q.w == ik::kOne,
              "LookAtRotation(fwd, fwd) -> identity");
    }

    // ===== LookAtRotation deterministic pi-flip for fwd~-to =====
    {
        const ik::FxVec3 fwd = fpx::FxNormalize(ik::FxVec3{F(1.0), F(0.2), F(0.1)});
        const ik::FxVec3 to  = ik::FxVec3{(ik::fx)(-fwd.x), (ik::fx)(-fwd.y), (ik::fx)(-fwd.z)};
        const ik::FxQuat q1 = ik::LookAtRotation(fwd, to);
        const ik::FxQuat q2 = ik::LookAtRotation(fwd, to);
        // Deterministic (two calls byte-identical).
        check(q1.x == q2.x && q1.y == q2.y && q1.z == q2.z && q1.w == q2.w,
              "LookAtRotation antiparallel is deterministic (two calls byte-identical)");
        // A pi rotation: w == cos(pi/2) == 0 (within the LUT band) and fwd rotates to ~ -fwd (to).
        const ik::FxVec3 aimed = fpx::FxRotate(q1, fwd);
        const ik::fx dot = ik::FxDotV(aimed, to);
        std::printf("ik_fabrik_test: antiparallel pi-flip aimDot=%d, qw=%d\n", dot, q1.w);
        check(dot >= ik::kOne - F(0.02), "LookAtRotation antiparallel flips fwd onto -fwd (pi rotation)");
    }

    if (g_fail == 0) std::printf("ik_fabrik_test: ALL PASS\n");
    else std::printf("ik_fabrik_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
