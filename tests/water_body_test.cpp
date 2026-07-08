// Slice WV1 — THE WATER GAMEPLAY VOLUME (hf::sim::water, engine/sim/water_body.h): the analytic integer
// Gerstner surface driving rigid-body Archimedes buoyancy + drag (the render/water.h equation made the
// Q16.16 physics ground truth; CP7's SphereCapVolume reused VERBATIM against the analytic surface).
//
// What this test PINS (the spec's physics proofs):
//   (a) FLAT WATER (amps=0), rho_b = 0.5 -> settles HALF-submerged (the CP7 analytic Archimedes pin, now
//       against the ANALYTIC surface — tighter than the SPH pool-quantile version): the settled window-
//       mean depth is pinned within a small band of the analytic d = r; the residual bob band is pinned.
//   (b) WAVES ON: the same sphere BOBS with the wave period — the measured mean peak interval (upward
//       mean-crossings) is pinned within a tick band of the analytic wave period, and the phase-locked
//       bob-trace digest is pinned (drag phase lag is real physics — the PERIOD is pinned, not the phase).
//   (c) a DENSE sphere (rho_b > rho_f) sinks to the AABB floor (pos.y == groundY + r exactly).
//   (d) IDENTITY-AT-ZERO: a zero WaterBody (density==dragK==0) -> SimWaterTick == fpx::SimTick BIT-EXACT
//       (RunWaterLockstep == fpx::RunLockstep over the same init+stream), plus the hand force balance:
//       at exactly half-submersion the one-tick buoyant dv cancels the gravity dv within an LSB band.
//   (e) TWO bodies at different (x,z) ride DIFFERENT wave phases (both traces pinned, proven different).
//   (f) LOCKSTEP: a peer fed ONLY the impulse-command stream re-derives the floating state bit-for-bit;
//       ROLLBACK corrects a genuinely-diverged misprediction — with the ZERO-BYTE WATER SNAPSHOT headline:
//       the snapshot is the fpx body vector ALONE (the water is analytic/stateless; the wave field is
//       re-derived from the tick number, so an infinite ocean adds zero snapshot bytes).
//   (g) THE SHOWCASE SCENARIO: two runs byte-identical + the final-state and trace digests pinned
//       (identical under MSVC and local clang — the cross-compiler anchor).
//
// Pure C++ (hf_core), pure integer sim — the ONE transcendental (sine) is the ik.h host-baked LUT.
#include "sim/water_body.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace water = hf::sim::water;
namespace fpx = hf::sim::fpx;
namespace couple = hf::sim::couple;
using water::fx;
using water::kOne;
using water::Snap;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// The cross-compiler pinned digests ((h): identical under MSVC and local clang — pure integer + the
// host-baked LUT, so any divergence is a real determinism bug).
static constexpr uint64_t kPinBobTrace      = 0xc834cfc97a176d2eull;   // (b) the phase-locked bob trace
static constexpr uint64_t kPinPhaseA        = 0xbb18a82adb674f73ull;   // (e) body A (x=-3) trace
static constexpr uint64_t kPinPhaseB        = 0x35d196b791d7396aull;   // (e) body B (x=+3) trace
static constexpr uint64_t kPinScenario      = 0x8eaa0116a528c87dull;   // (g) the showcase final-state digest
static constexpr uint64_t kPinScenarioTrace = 0xa0baf237aee1e51full;   // (g) the showcase trace digest

static fx absfx(fx v) { return v < 0 ? -v : v; }

// A flat-water volume (amps=0 — the still ocean at y=4 over a floor at y=0), density 1, drag 2/s.
static water::WaterBody FlatWater() {
    water::WaterBody wb;
    wb.bounds.lo = fpx::FxVec3{Snap(-8.0), 0, Snap(-8.0)};
    wb.bounds.hi = fpx::FxVec3{Snap(8.0), Snap(4.0), Snap(8.0)};
    wb.waveCount = 0;
    wb.waterDensity = kOne;
    wb.dragK = Snap(2.0);
    return wb;
}

// The single-wave bob ocean: ONE wave amp 0.25, wavelength 4, speed 2, dir +X. Analytic period:
// k = 2*pi/4, w = 2*k = pi -> T = 2*pi/w = 2 s = 120 ticks at 60 Hz (integer: kTwoPi/wdt ~ 120.05).
static water::WaterBody BobWater() {
    water::WaterBody wb = FlatWater();
    wb.waves[0] = water::WaterWave{Snap(0.25), Snap(4.0), Snap(2.0), kOne, 0};
    wb.waveCount = 1;
    return wb;
}

static fpx::FxWorld OneSphereWorld(fx x, fx y, fx z, double rho) {
    fpx::FxWorld w;
    w.gravity = fpx::FxVec3{0, Snap(-9.8), 0};
    w.groundY = 0;
    w.bodies.push_back(couple::BodyFromDensity(fpx::FxVec3{x, y, z}, Snap(0.5), Snap(rho)));
    return w;
}

// FNV-1a-64 over a Q16.16 trace (the pinned phase-locked digest).
static uint64_t TraceDigest(const std::vector<fx>& trace) {
    uint64_t h = 1469598103934665603ull;
    for (fx v : trace) {
        for (int b = 0; b < 4; ++b) {
            h ^= (uint64_t)(((uint32_t)v >> (b * 8)) & 0xFFu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

int main() {
    HF_TEST_MAIN_INIT();
    const fx dt = kOne / 60;
    const fx r = Snap(0.5);
    const fx stillY = Snap(4.0);

    // ---- (0) The analytic surface primitives -----------------------------------------------------
    {
        const water::WaterBody flat = FlatWater();
        check(water::SurfaceHeight(flat, 0, 0, 0) == stillY,
              "surface: amps=0 -> h == the still level EXACTLY (identity-at-zero)");
        check(water::SurfaceHeight(flat, Snap(3.7), Snap(-2.1), 977u) == stillY,
              "surface: amps=0 -> h == still level at any (x,z,tick)");
        check(water::SurfaceVelY(flat, Snap(1.0), Snap(1.0), 42u) == 0,
              "surface: amps=0 -> the analytic vertical velocity is EXACTLY 0");
        const water::WaterBody bob = BobWater();
        // Determinism + boundedness: the wave sum stays within +-amp of the still level over a sweep.
        bool bounded = true, deterministic = true;
        for (int i = 0; i < 200; ++i) {
            const fx x = Snap(-8.0) + (fx)(((int64_t)i << 16) * 16 / 200);
            const fx h1 = water::SurfaceHeight(bob, x, 0, (uint32_t)(i * 7));
            const fx h2 = water::SurfaceHeight(bob, x, 0, (uint32_t)(i * 7));
            if (h1 != h2) deterministic = false;
            if (absfx(h1 - stillY) > Snap(0.25) + 64) bounded = false;   // amp + a LUT-lerp LSB guard
        }
        check(deterministic, "surface: SurfaceHeight is a pure function (two calls bit-identical)");
        check(bounded, "surface: |h - still| <= amp (+LSB band) across a position/tick sweep");
    }

    // ---- (d1) The hand force balance at exact half-submersion (flat water, drag off) --------------
    {
        water::WaterBody wb = FlatWater();
        wb.dragK = 0;
        fpx::FxWorld w = OneSphereWorld(0, stillY, 0, 0.5);   // center at the surface -> d = r exactly
        fpx::FxBody& b = w.bodies[0];
        const fx dvG = fpx::fxmul(w.gravity.y, dt);           // the one-tick gravity dv (negative)
        water::StepFloatBody(wb, b, w.gravity, dt, 0u);
        const fx dvB = b.vel.y;                                // the one-tick buoyancy dv (positive)
        std::printf("WV1 balance: dvB=%d dvG=%d residual=%d LSB\n", dvB, dvG, dvB + dvG);
        check(absfx(dvB + dvG) <= 64,
              "balance: at d==r the buoyant dv cancels the gravity dv within 64 LSB (~0.001 wu/s)");
    }

    // ---- (a) FLAT water, rho_b = 0.5 -> settles HALF-submerged (the analytic Archimedes pin) ------
    {
        const water::WaterBody wb = FlatWater();
        fpx::FxWorld w = OneSphereWorld(0, Snap(5.0), 0, 0.5);
        fpx::FxWorld w2 = w;
        water::StepWaterSteps(wb, w, dt, 4, 0u, 600);
        water::StepWaterSteps(wb, w2, dt, 4, 0u, 600);
        check(water::WaterWorldsEqual(w, w2), "flat settle: two runs BIT-IDENTICAL");
        // The settled window: 120 more ticks; track the depth band + mean.
        int64_t sumD = 0;
        fx dLo = 0, dHi = 0;
        for (int t = 0; t < 120; ++t) {
            water::StepWaterSteps(wb, w, dt, 4, (uint32_t)(600 + t), 1);
            const fx d = couple::SubmergedDepth(w.bodies[0], stillY);
            if (t == 0) { dLo = dHi = d; }
            else { if (d < dLo) dLo = d; if (d > dHi) dHi = d; }
            sumD += d;
        }
        const fx meanD = (fx)(sumD / 120);
        std::printf("WV1 flat Archimedes rho=0.50: meanD=%d (analytic r=%d), band [%d,%d] (bob %d), "
                    "digest=0x%016llx\n",
                    meanD, r, dLo, dHi, dHi - dLo, (unsigned long long)water::WaterDigest(w));
        check(absfx(meanD - r) <= Snap(0.02),
              "flat Archimedes rho=0.5: settled HALF-submerged (mean d within 0.02 wu of the analytic r)");
        check(dHi - dLo <= Snap(0.02), "flat Archimedes rho=0.5: the residual bob band is < 0.02 wu");
    }

    // ---- (c) A DENSE sphere (rho_b = 1.5 > rho_f) sinks to the AABB floor, waves ON ----------------
    {
        const water::WaterBody wb = BobWater();
        fpx::FxWorld w = OneSphereWorld(0, Snap(5.0), 0, 1.5);
        water::StepWaterSteps(wb, w, dt, 4, 0u, 600);
        std::printf("WV1 dense sink: pos.y=%d (floor+r=%d)\n", w.bodies[0].pos.y, r);
        check(w.bodies[0].pos.y == r, "dense rho=1.5: SINKS to the AABB floor (pos.y == groundY + r)");
    }

    // ---- (b) WAVES ON: the rho=0.5 sphere BOBS with the wave period --------------------------------
    {
        const water::WaterBody wb = BobWater();
        // The analytic integer wave period in ticks: kTwoPi / (w*dt), w = fxmul(speed, fxdiv(kTwoPi, L)).
        const fx k = fpx::fxdiv(water::kTwoPi, Snap(4.0));
        const fx wfreq = fpx::fxmul(Snap(2.0), k);
        const fx wdt = fpx::fxmul(wfreq, dt);
        const int periodTicks = (int)((int64_t)water::kTwoPi / wdt);   // ~120
        fpx::FxWorld w = OneSphereWorld(0, Snap(5.0), 0, 0.5);
        water::StepWaterSteps(wb, w, dt, 4, 0u, 480);                  // transient decay (drag)
        std::vector<fx> trace;
        int64_t sumY = 0;
        for (int t = 0; t < 480; ++t) {                                 // 4 wave periods
            water::StepWaterSteps(wb, w, dt, 4, (uint32_t)(480 + t), 1);
            trace.push_back(w.bodies[0].pos.y);
            sumY += w.bodies[0].pos.y;
        }
        const fx meanY = (fx)(sumY / 480);
        // Peak measurement: upward mean-crossings with a 30-tick debounce (documented guard).
        std::vector<int> crossings;
        for (size_t t = 1; t < trace.size(); ++t) {
            if (trace[t - 1] < meanY && trace[t] >= meanY) {
                if (crossings.empty() || (int)t - crossings.back() >= 30) crossings.push_back((int)t);
            }
        }
        fx bobAmp = 0;
        for (fx y : trace) if (absfx(y - meanY) > bobAmp) bobAmp = absfx(y - meanY);
        int64_t sumIv = 0;
        for (size_t c = 1; c < crossings.size(); ++c) sumIv += crossings[c] - crossings[c - 1];
        const int nIv = (int)crossings.size() - 1;
        const int meanIv = nIv > 0 ? (int)(sumIv / nIv) : 0;
        const uint64_t bobDigest = TraceDigest(trace);
        std::printf("WV1 bob: analytic period=%d ticks, measured mean interval=%d (n=%d), bobAmp=%d, "
                    "trace digest=0x%016llx\n",
                    periodTicks, meanIv, nIv, bobAmp, (unsigned long long)bobDigest);
        check(nIv >= 2, "bob: at least 3 upward mean-crossings over 4 wave periods (a real bob)");
        check(bobAmp > Snap(0.05), "bob: the bob amplitude is non-trivial (> 0.05 wu — the wave drives it)");
        check(meanIv >= periodTicks - 12 && meanIv <= periodTicks + 12,
              "bob: the measured bob period matches the wave period within +-12 ticks (10%)");
        check(bobDigest == kPinBobTrace, "bob: the phase-locked bob-trace digest is pinned");
    }

    // ---- (e) TWO bodies at different (x,z) ride DIFFERENT wave phases ------------------------------
    {
        const water::WaterBody wb = BobWater();   // wavelength 4, dir +X -> x=-3 vs x=+3 is ~3*pi apart
        fpx::FxWorld w;
        w.gravity = fpx::FxVec3{0, Snap(-9.8), 0};
        w.groundY = 0;
        w.bodies.push_back(couple::BodyFromDensity(fpx::FxVec3{Snap(-3.0), Snap(5.0), 0}, r, Snap(0.5)));
        w.bodies.push_back(couple::BodyFromDensity(fpx::FxVec3{Snap(3.0), Snap(5.0), 0}, r, Snap(0.5)));
        water::StepWaterSteps(wb, w, dt, 4, 0u, 480);
        std::vector<fx> traceA, traceB;
        int differ = 0;
        for (int t = 0; t < 240; ++t) {
            water::StepWaterSteps(wb, w, dt, 4, (uint32_t)(480 + t), 1);
            traceA.push_back(w.bodies[0].pos.y);
            traceB.push_back(w.bodies[1].pos.y);
            if (w.bodies[0].pos.y != w.bodies[1].pos.y) ++differ;
        }
        const uint64_t digA = TraceDigest(traceA), digB = TraceDigest(traceB);
        std::printf("WV1 phases: differing ticks=%d/240, traceA=0x%016llx traceB=0x%016llx\n",
                    differ, (unsigned long long)digA, (unsigned long long)digB);
        check(differ > 200, "phases: two bodies at different (x,z) ride DIFFERENT wave phases");
        check(digA == kPinPhaseA, "phases: body-A trace digest pinned");
        check(digB == kPinPhaseB, "phases: body-B trace digest pinned");
        check(digA != digB, "phases: the two pinned traces are distinct");
    }

    // ---- (d2) IDENTITY-AT-ZERO: a zero WaterBody -> the water tick == the plain fpx tick BIT-EXACT --
    {
        water::WaterBody zero{};                   // density == dragK == 0 -> StepFloatBody is an EXACT no-op
        fpx::FxWorld init = OneSphereWorld(0, Snap(5.0), 0, 0.5);
        init.bodies.push_back(couple::BodyFromDensity(fpx::FxVec3{Snap(1.0), Snap(7.0), 0}, r, Snap(0.8)));
        std::vector<fpx::FxCommand> stream;
        stream.push_back(fpx::FxCommand{20u, fpx::kCmdImpulse, 0u, fpx::FxVec3{Snap(0.5), 0, 0}});
        const fpx::FxWorld a = water::RunWaterLockstep(zero, init, stream, 90, dt, 4);
        const fpx::FxWorld b = fpx::RunLockstep(init, stream, 90, dt, 4);
        check(water::WaterWorldsEqual(a, b),
              "identity-at-zero: zero water -> RunWaterLockstep == fpx::RunLockstep BIT-EXACT");
    }

    // ---- (f) LOCKSTEP + ROLLBACK over the showcase ocean (the zero-byte water snapshot headline) ---
    {
        const water::WaterBody wb = water::ShowcaseWater();
        const fpx::FxWorld init = water::MakeShotWorld();
        std::vector<fpx::FxCommand> auth;
        auth.push_back(fpx::FxCommand{30u, fpx::kCmdImpulse, 0u, fpx::FxVec3{Snap(1.5), 0, 0}});
        auth.push_back(fpx::FxCommand{90u, fpx::kCmdImpulse, 1u, fpx::FxVec3{0, Snap(2.0), Snap(-0.5)}});
        const fpx::FxWorld authority = water::RunWaterLockstep(wb, init, auth, 240, dt, 4);
        const fpx::FxWorld replica = water::RunWaterLockstep(wb, init, auth, 240, dt, 4);
        check(water::WaterWorldsEqual(authority, replica),
              "lockstep: a peer fed ONLY the impulse stream re-derives the floating state bit-for-bit");
        std::printf("WV1 lockstep: authority digest=0x%016llx\n",
                    (unsigned long long)water::WaterDigest(authority));
        // Rollback: mispredict the t90 impulse as a different push -> genuinely diverges -> corrected.
        std::vector<fpx::FxCommand> mis = auth;
        mis[1].arg = fpx::FxVec3{Snap(-2.0), 0, 0};
        // The non-vacuous divergence control: the mispredicted full run differs from authority.
        const fpx::FxWorld misFull = water::RunWaterLockstep(wb, init, mis, 240, dt, 4);
        check(!water::WaterWorldsEqual(misFull, authority),
              "rollback control: the mispredicted stream GENUINELY diverges");
        const fpx::FxWorld corrected = water::RunWaterRollback(wb, init, auth, mis, 240, 60, dt, 4);
        check(water::WaterWorldsEqual(corrected, authority),
              "rollback: restore(fpx bodies ONLY — the ZERO-BYTE water snapshot) + resim == authority");
    }

    // ---- (g) The showcase scenario: two-run identical + the pinned digests -------------------------
    {
        const water::WaterShotRun run = water::RunWaterShotScenario();
        const water::WaterShotRun run2 = water::RunWaterShotScenario();
        check(run.digest == run2.digest && run.traceDigest == run2.traceDigest,
              "scenario: two runs byte-identical (digest + trace)");
        std::printf("WV1 scenario: digest=0x%016llx trace=0x%016llx depths={%d,%d,%d} (r=%d, 2r=%d)\n",
                    (unsigned long long)run.digest, (unsigned long long)run2.traceDigest,
                    run.depths[0], run.depths[1], run.depths[2], r, r + r);
        check(run.digest == kPinScenario, "scenario: the final-state digest is pinned (MSVC == clang)");
        check(run.traceDigest == kPinScenarioTrace, "scenario: the trace digest is pinned");
        check(run.depths[0] < run.depths[1],
              "scenario: the LIGHT sphere (rho 0.35) rides HIGHER than the medium (rho 0.60)");
        check(run.depths[2] == r + r, "scenario: the DENSE sphere is fully submerged (d == 2r) at the floor");
        check(run.finalWorld.bodies[2].pos.y == r, "scenario: the dense sphere rests ON the AABB floor");
        // The raster itself is deterministic (the showcase golden's substance).
        std::vector<uint8_t> imgA, imgB;
        uint32_t wA = 0, hA = 0, wB = 0, hB = 0;
        water::RenderWaterShot(run, imgA, wA, hA);
        water::RenderWaterShot(run2, imgB, wB, hB);
        check(wA == wB && hA == hB && imgA == imgB, "scenario: two rasters BYTE-IDENTICAL");
    }

    if (g_fail == 0) std::printf("water_body_test: ALL PASS\n");
    else std::printf("water_body_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
