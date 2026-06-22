// Slice IK1 — Deterministic IK Control-Rig: THE FIXED-POINT ANGLE LUT (the beachhead of FLAGSHIP #32).
// The pure-CPU contract test for engine/anim/ik.h::FxAcosLut / FxSinLut / FxCosLut / FxAtan2Lut (the
// host-baked Q16.16 angle LUTs the GPU shaders/ik_angle.comp copies VERBATIM + proves bit-identical).
// Pure C++ (header-only, no device, no backend symbols). Namespace hf::anim::ik.
//
// What this test PINS (the contracts the GPU sweep + later IK slices build on):
//   * FxAcosLut within a documented LSB band vs std::acos over a dense cosine sweep [-1, 1].
//   * acos endpoints/midpoint: acos(kOne)==0, acos(-kOne)==pi-snap, acos(0)==pi/2-snap.
//   * acos MONOTONE non-increasing (the table is non-increasing by construction; the lerp preserves it).
//   * FxSinLut/FxCosLut: known angles (sin0=0, cos0=kOne, sin(pi/2)=kOne, cos(pi)=-kOne) + the
//     sin^2+cos^2 ~= kOne round-trip within band, over a dense theta sweep.
//   * FxAtan2Lut quadrant correctness (the four quadrants + the four axes) within band.
//   * determinism: every lookup is a pure function -> two calls byte-identical.
//
// Pure C++ (hf_core), ASan-eligible like the other anim/sim-math tests. ik.h #includes sim/fpx.h +
// sim/joint.h + anim/skeleton.h + anim/animation.h READ-ONLY.
#include "anim/ik.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace ik = hf::anim::ik;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static int32_t iabs32(int32_t v) { return v < 0 ? -v : v; }

int main() {
    HF_TEST_MAIN_INIT();
    const ik::fx kOne = ik::kOne;
    const double kPiD = 3.14159265358979323846;

    // ===== acos: within-band vs std::acos over a dense cosine sweep [-1, 1] =====
    // The documented within-band tolerance: a host-snapped 256-bin table + linear lerp. With 256 bins the
    // worst case is the steep-slope endpoints; we allow a generous LSB band that the implementation beats
    // comfortably (the test reports the observed max). 256 LSB (~0.004 rad) is the documented band.
    // The HONEST acos characterization: acos has a sqrt-singularity slope at x=+/-1 where ANY uniform-x
    // table + linear lerp is fundamentally limited (the lerp chord under-shoots the infinite-slope curve).
    // So the band is split: the INTERIOR (|x| <= 1 - 2/kAcosBins, away from the two endpoint bins) is TIGHT
    // (~tens of LSB); the two ENDPOINT bins carry the singularity error (documented, ~1k LSB at 1024 bins).
    // Both are DETERMINISTIC + bit-identical cross-platform (the headline); the magnitude is the within-band
    // caveat (the fluid.h binned-LUT fidelity-simplification shape).
    {
        const int32_t kInteriorTolLsb = 96;    // tight away from the endpoints
        const int32_t kEndpointTolLsb = 1100;  // the documented sqrt-singularity endpoint band (1024 bins)
        const double endpointX = 1.0 - 2.0 / (double)ik::kAcosBins;  // the last interior bin edge
        int32_t maxInterior = 0, maxEndpoint = 0;
        const int N = 8001;
        for (int k = 0; k < N; ++k) {
            const double xc = -1.0 + 2.0 * (double)k / (double)(N - 1);
            const ik::fx x = (ik::fx)std::llround(xc * (double)kOne);
            const ik::fx got = ik::FxAcosLut(x);
            const double want = std::acos(xc < -1.0 ? -1.0 : (xc > 1.0 ? 1.0 : xc));
            const ik::fx wantFx = (ik::fx)std::llround(want * (double)kOne);
            const int32_t err = iabs32(got - wantFx);
            if (std::fabs(xc) <= endpointX) { if (err > maxInterior) maxInterior = err; }
            else                            { if (err > maxEndpoint) maxEndpoint = err; }
        }
        std::printf("ik_angle_test: acos interiorMaxLsb=%d (tol=%d), endpointMaxLsb=%d (tol=%d)\n",
                    maxInterior, kInteriorTolLsb, maxEndpoint, kEndpointTolLsb);
        check(maxInterior <= kInteriorTolLsb, "FxAcosLut TIGHT band over the interior");
        check(maxEndpoint <= kEndpointTolLsb, "FxAcosLut within documented endpoint band (sqrt singularity)");
    }

    // ===== acos endpoints / midpoint =====
    {
        check(ik::FxAcosLut(kOne) == 0, "acos(kOne) == 0");
        const ik::fx wantPi = (ik::fx)std::llround(kPiD * (double)kOne);
        check(iabs32(ik::FxAcosLut(-kOne) - wantPi) <= 2, "acos(-kOne) == pi-snap");
        const ik::fx wantHalfPi = (ik::fx)std::llround((kPiD / 2.0) * (double)kOne);
        check(iabs32(ik::FxAcosLut(0) - wantHalfPi) <= 256, "acos(0) == pi/2-snap");
    }

    // ===== acos monotone non-increasing over the cosine domain =====
    {
        bool mono = true;
        ik::fx prev = ik::FxAcosLut(-kOne);
        const int N = 2049;
        for (int k = 1; k < N; ++k) {
            const ik::fx x = (ik::fx)(-kOne + (ik::fx)(((int64_t)2 * kOne * k) / (N - 1)));
            const ik::fx cur = ik::FxAcosLut(x);
            if (cur > prev) { mono = false; break; }   // non-increasing
            prev = cur;
        }
        check(mono, "FxAcosLut monotone non-increasing");
    }

    // ===== sin/cos: known angles =====
    {
        check(ik::FxSinLut(0) == 0, "sin(0) == 0");
        check(ik::FxCosLut(0) == kOne, "cos(0) == kOne");
        const ik::fx halfPi = ik::kHalfPi;
        check(iabs32(ik::FxSinLut(halfPi) - kOne) <= 16, "sin(pi/2) == kOne");
        check(iabs32(ik::FxCosLut(halfPi)) <= 64, "cos(pi/2) == 0");
        check(iabs32(ik::FxCosLut(ik::kPi) - (-kOne)) <= 16, "cos(pi) == -kOne");
    }

    // ===== sin/cos: round-trip sin^2 + cos^2 ~= kOne + within-band vs std =====
    {
        const int32_t kRtTolLsb = 700;   // documented round-trip band (Q16.16 of sin^2+cos^2 vs kOne)
        const int32_t kTrigTolLsb = 256; // documented per-value band vs std::sin/std::cos
        int32_t maxRt = 0, maxTrig = 0;
        const int N = 4001;
        for (int k = 0; k < N; ++k) {
            const double th = 2.0 * kPiD * (double)k / (double)(N - 1);
            const ik::fx theta = (ik::fx)std::llround(th * (double)kOne);
            const ik::fx s = ik::FxSinLut(theta);
            const ik::fx c = ik::FxCosLut(theta);
            // sin^2 + cos^2 in Q16.16 (the products are Q32.32 -> >>16 back to Q16.16).
            const int64_t s2 = ((int64_t)s * (int64_t)s) >> 16;
            const int64_t c2 = ((int64_t)c * (int64_t)c) >> 16;
            const int32_t rtErr = iabs32((int32_t)(s2 + c2) - kOne);
            if (rtErr > maxRt) maxRt = rtErr;
            const ik::fx ws = (ik::fx)std::llround(std::sin(th) * (double)kOne);
            const ik::fx wc = (ik::fx)std::llround(std::cos(th) * (double)kOne);
            maxTrig = std::max(maxTrig, std::max(iabs32(s - ws), iabs32(c - wc)));
        }
        std::printf("ik_angle_test: sin/cos roundTripMaxLsb=%d (tol=%d), valMaxLsb=%d (tol=%d)\n",
                    maxRt, kRtTolLsb, maxTrig, kTrigTolLsb);
        check(maxRt <= kRtTolLsb, "FxSinLut/FxCosLut sin^2+cos^2 within band");
        check(maxTrig <= kTrigTolLsb, "FxSinLut/FxCosLut within band vs std");
    }

    // ===== atan2: quadrant correctness (four quadrants + four axes) =====
    {
        const ik::fx kHalfPi = ik::kHalfPi;
        const ik::fx kPi = ik::kPi;
        // Axes.
        check(ik::FxAtan2Lut(0, kOne) == 0, "atan2(0,+x) == 0");
        check(iabs32(ik::FxAtan2Lut(kOne, 0) - kHalfPi) <= 64, "atan2(+y,0) == +pi/2");
        check(iabs32(ik::FxAtan2Lut(-kOne, 0) + kHalfPi) <= 64, "atan2(-y,0) == -pi/2");
        check(iabs32(ik::FxAtan2Lut(0, -kOne) - kPi) <= 64, "atan2(0,-x) == +pi");
        check(ik::FxAtan2Lut(0, 0) == 0, "atan2(0,0) == 0 (degenerate)");
        // Quadrants (y=+/-1, x=+/-1 -> +/- pi/4, +/- 3pi/4). Within band.
        const ik::fx q1 = ik::FxAtan2Lut(kOne, kOne);    // +pi/4
        const ik::fx q2 = ik::FxAtan2Lut(kOne, -kOne);   // +3pi/4
        const ik::fx q3 = ik::FxAtan2Lut(-kOne, -kOne);  // -3pi/4
        const ik::fx q4 = ik::FxAtan2Lut(-kOne, kOne);   // -pi/4
        const ik::fx qpi4 = (ik::fx)std::llround((kPiD / 4.0) * (double)kOne);
        const ik::fx q3pi4 = (ik::fx)std::llround((3.0 * kPiD / 4.0) * (double)kOne);
        const int32_t kQTol = 256;
        check(iabs32(q1 - qpi4) <= kQTol, "atan2(+,+) == +pi/4 (Q I)");
        check(iabs32(q2 - q3pi4) <= kQTol, "atan2(+,-) == +3pi/4 (Q II)");
        check(iabs32(q3 + q3pi4) <= kQTol, "atan2(-,-) == -3pi/4 (Q III)");
        check(iabs32(q4 + qpi4) <= kQTol, "atan2(-,+) == -pi/4 (Q IV)");
        // Within-band vs std::atan2 over a dense ring of directions.
        int32_t maxErr = 0;
        const int N = 720;
        for (int k = 0; k < N; ++k) {
            const double ang = -kPiD + 2.0 * kPiD * (double)k / (double)N;
            const ik::fx y = (ik::fx)std::llround(std::sin(ang) * (double)kOne);
            const ik::fx x = (ik::fx)std::llround(std::cos(ang) * (double)kOne);
            const ik::fx got = ik::FxAtan2Lut(y, x);
            ik::fx want = (ik::fx)std::llround(std::atan2((double)y / kOne, (double)x / kOne) * (double)kOne);
            // Wrap the difference into (-pi, pi] so the +/-pi seam doesn't false-fail.
            int32_t d = got - want;
            const ik::fx twoPi = ik::kTwoPi;
            if (d >  ik::kPi) d -= twoPi;
            if (d < -ik::kPi) d += twoPi;
            if (iabs32(d) > maxErr) maxErr = iabs32(d);
        }
        std::printf("ik_angle_test: atan2 maxErrLsb=%d (tol=512)\n", maxErr);
        check(maxErr <= 512, "FxAtan2Lut within band vs std::atan2");
    }

    // ===== determinism: every lookup is a pure function (two calls byte-identical) =====
    {
        bool det = true;
        for (int k = -2000; k <= 2000 && det; k += 7) {
            const ik::fx x = (ik::fx)k * 32;
            if (ik::FxAcosLut(x) != ik::FxAcosLut(x)) det = false;
            if (ik::FxSinLut(x)  != ik::FxSinLut(x))  det = false;
            if (ik::FxCosLut(x)  != ik::FxCosLut(x))  det = false;
            if (ik::FxAtan2Lut(x, (ik::fx)(kOne - x)) != ik::FxAtan2Lut(x, (ik::fx)(kOne - x))) det = false;
        }
        check(det, "all lookups are pure functions (two calls byte-identical)");
    }

    if (g_fail == 0) std::printf("ik_angle_test: ALL PASS\n");
    else std::printf("ik_angle_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
