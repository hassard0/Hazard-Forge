// Slice FF1 — RIGID-BODY FORCE-FIELD VOLUMES (hf::sim::ff, engine/sim/force_field.h): the PT2 particle
// force-field math (radial/vortex/wind) applied to fpx RIGID BODIES with bounded AABB volumes, mutating
// velocities pre-step (the WV1 discipline) — UE5-Physics-Fields-class authored fields, bit-exact and
// lockstep/rollback-replayable.
//
// What this test PINS (the spec's proofs):
//   (t) TWIN FIDELITY: EvalFieldForce with kFalloffLinear == particles::AccumulateForce BIT-EXACT over a
//       position grid, for radial (kFieldPoint), vortex, and wind — the twin can never drift from PT2.
//   (a) RADIAL RING EXACT: four axis-aligned ring bodies blasted outward by a repeller move EXACTLY
//       symmetric, ANALYTICALLY-pinned integer displacements (constants designed so every fxmul is exact:
//       dt = 1/64, strength = -4.0, invMass = 1 -> dv = 4096/step, pos step = 64k, disp = 32*N*(N+1)).
//   (b) VORTEX ORBIT: the exact first-tick tangential dv, the orbit direction (a +Y-axis vortex pushes a
//       +x body toward -z), the outward-spiral radius band (discrete integration spirals out — honest),
//       and the pinned trajectory digest (MSVC == clang).
//   (c) WIND VOLUME BOUNDED: a wind lane pushes a resting sphere across the frictionless fpx floor (fpx
//       ground contact has NO friction — the body slides smoothly, no stick-slip; the exact displacement
//       is pinned), while a body OUTSIDE the AABB is BIT-EXACT untouched vs a no-field control run.
//   (d) IDENTITY-AT-ZERO: no volumes AND all-disabled volumes -> RunFieldLockstep == fpx::RunLockstep
//       BIT-EXACT over the same init + (body-)command stream.
//   (e) FALLOFF: kFalloffInvSq is EXACTLY quarter force at 2x distance (integer-exact), and the near
//       clamp plateaus at full strength for dist <= nearClamp (kFalloffLinear is covered by (t)).
//   (f) LOCKSTEP + ROLLBACK with mid-run TOGGLE commands: a peer re-derives the fielded world bit-for-bit;
//       rollback (bodies + the per-volume enabled BITS) corrects a mispredicted toggle; the restore-
//       without-bits control GENUINELY diverges (the enabled bits ARE state, not redundant).
//   (g) THE SHOWCASE SCENARIO: two runs byte-identical, the final-state + trace digests pinned
//       (MSVC == clang), the raster byte-identical, and the story asserts (burst ring expands, wind
//       boxes cross the lane).
//
// Pure C++ (hf_core), pure integer — NO float in any per-tick path (Snap is scene-build-time only).
#include "sim/force_field.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace ff = hf::sim::ff;
namespace fpx = hf::sim::fpx;
namespace particles = hf::sim::particles;
using ff::fx;
using ff::kOne;
using ff::Snap;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static fx absfx(fx v) { return v < 0 ? -v : v; }

// The cross-compiler pinned digests (identical under MSVC and local clang — pure integer, so any
// divergence is a real determinism bug). Baked from the first verified MSVC run.
static constexpr uint64_t kPinVortexTraj    = 0xc93c28f5fbd234c1ull;  // (b) the vortex trajectory digest
static constexpr fx       kPinWindDispX     = 396240;                 // (c) body A final pos.x (exact)
static constexpr uint64_t kPinScenario      = 0x98e2c0fd3bc370b5ull;  // (g) showcase final-state digest
static constexpr uint64_t kPinScenarioTrace = 0x541cad84d0715015ull;  // (g) showcase trace digest

// FNV-1a-64 over a Q16.16 trace (the WV1 TraceDigest twin).
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

// A dynamic unit-mass sphere.
static fpx::FxBody Body(fx x, fx y, fx z, fx r) {
    fpx::FxBody b;
    b.pos = fpx::FxVec3{x, y, z};
    b.invMass = kOne;
    b.flags = fpx::kFlagDynamic;
    b.radius = r;
    return b;
}

// A huge-bounds volume (the field math unconstrained by the AABB — for the twin/falloff unit proofs).
static ff::FieldVolume HugeVolume(uint32_t kind) {
    ff::FieldVolume v;
    v.kind = kind;
    v.bounds.lo = fpx::FxVec3{Snap(-100.0), Snap(-100.0), Snap(-100.0)};
    v.bounds.hi = fpx::FxVec3{Snap(100.0), Snap(100.0), Snap(100.0)};
    return v;
}

int main() {
    HF_TEST_MAIN_INIT();
    const fx dt60 = kOne / 60;

    // ---- (t) TWIN FIDELITY: EvalFieldForce(kFalloffLinear) == particles::AccumulateForce BIT-EXACT ----
    {
        const fpx::FxVec3 center{Snap(0.2), Snap(0.1), Snap(-0.3)};
        const fpx::FxVec3 axisY{0, kOne, 0};
        const fpx::FxVec3 windDir{Snap(0.6), 0, Snap(0.8)};
        // The PT2 originals (particles.h read-only) and the FF1 twins, same authored numbers.
        const particles::ForceField pfPoint{particles::kFieldPoint, center, axisY, Snap(1.7), Snap(2.5)};
        const particles::ForceField pfVort{particles::kFieldVortex, center, axisY, Snap(-2.3), Snap(3.0)};
        const particles::ForceField pfWind{particles::kFieldWind, center, windDir, Snap(-2.2), 0};
        ff::FieldVolume vPoint = HugeVolume(ff::kFieldRadial);
        vPoint.center = center; vPoint.strength = Snap(1.7);
        vPoint.falloff = ff::kFalloffLinear; vPoint.radius = Snap(2.5);
        ff::FieldVolume vVort = HugeVolume(ff::kFieldVortex);
        vVort.center = center; vVort.axis = axisY; vVort.strength = Snap(-2.3);
        vVort.falloff = ff::kFalloffLinear; vVort.radius = Snap(3.0);
        ff::FieldVolume vWind = HugeVolume(ff::kFieldWind);
        vWind.axis = windDir; vWind.strength = Snap(-2.2);

        bool pointOk = true, vortOk = true, windOk = true;
        int inside = 0;
        for (int i = -3; i <= 3; ++i) {
            for (int j = -3; j <= 3; ++j) {
                const fpx::FxVec3 pos{Snap(i * 0.7 + 0.05), Snap(0.3), Snap(j * 0.55 - 0.1)};
                particles::FxParticle p{};
                p.pos = pos;
                const fpx::FxVec3 rp = particles::AccumulateForce(p, &pfPoint, 1);
                const fpx::FxVec3 rv = particles::AccumulateForce(p, &pfVort, 1);
                const fpx::FxVec3 rw = particles::AccumulateForce(p, &pfWind, 1);
                const fpx::FxVec3 tp = ff::EvalFieldForce(vPoint, pos);
                const fpx::FxVec3 tv = ff::EvalFieldForce(vVort, pos);
                const fpx::FxVec3 tw = ff::EvalFieldForce(vWind, pos);
                if (tp.x != rp.x || tp.y != rp.y || tp.z != rp.z) pointOk = false;
                if (tv.x != rv.x || tv.y != rv.y || tv.z != rv.z) vortOk = false;
                if (tw.x != rw.x || tw.y != rw.y || tw.z != rw.z) windOk = false;
                if (rp.x != 0 || rp.z != 0) ++inside;   // the grid genuinely exercises the falloff region
            }
        }
        check(pointOk, "twin: radial(kFalloffLinear) == PT2 AccumulateForce(kFieldPoint) BIT-EXACT");
        check(vortOk, "twin: vortex(kFalloffLinear) == PT2 AccumulateForce(kFieldVortex) BIT-EXACT");
        check(windOk, "twin: wind == PT2 AccumulateForce(kFieldWind) BIT-EXACT");
        check(inside >= 10, "twin: the grid has non-trivial in-radius coverage (the proof is not vacuous)");
    }

    // ---- (a) RADIAL RING EXACT: analytically-pinned symmetric burst displacements ---------------------
    // Designed-exact constants: dt = kOne/64 (1024), strength = -4.0 (repel), falloff kNone, invMass = 1,
    // axis-aligned ring at d = 4.0 -> per tick dv = fxmul(4*kOne, 1024) = 4096 EXACTLY (outward), pos gain
    // at tick k = fxmul(4096k, 1024) = 64k EXACTLY -> displacement after N ticks = 64*N(N+1)/2 = 32N(N+1).
    {
        const fx dt = kOne / 64;
        const int N = 32;
        const fx d = Snap(4.0);
        fpx::FxWorld w;
        w.gravity = fpx::FxVec3{0, 0, 0};
        w.groundY = Snap(-10.0);
        w.bodies.push_back(Body(d, 0, 0, Snap(0.25)));
        w.bodies.push_back(Body(-d, 0, 0, Snap(0.25)));
        w.bodies.push_back(Body(0, 0, d, Snap(0.25)));
        w.bodies.push_back(Body(0, 0, -d, Snap(0.25)));
        std::vector<ff::FieldVolume> vols(1, HugeVolume(ff::kFieldRadial));
        vols[0].center = fpx::FxVec3{0, 0, 0};
        vols[0].strength = Snap(-4.0);
        vols[0].falloff = ff::kFalloffNone;
        ff::StepFieldSteps(vols, w, dt, 4, 0u, N);
        const fx disp = 32 * N * (N + 1);   // the analytic exact displacement (33792 for N=32)
        check(w.bodies[0].pos.x == d + disp, "radial: +x ring body displaced EXACTLY 32*N*(N+1) outward");
        check(w.bodies[1].pos.x == -d - disp, "radial: -x ring body displaced EXACTLY symmetric");
        check(w.bodies[2].pos.z == d + disp, "radial: +z ring body displaced EXACTLY 32*N*(N+1) outward");
        check(w.bodies[3].pos.z == -d - disp, "radial: -z ring body displaced EXACTLY symmetric");
        check(w.bodies[0].vel.x == 4096 * N && w.bodies[1].vel.x == -4096 * N,
              "radial: the exact velocity ladder (dv = 4096/step, mirrored)");
        check(w.bodies[0].pos.y == 0 && w.bodies[0].pos.z == 0,
              "radial: the axis-aligned body moves ONLY along its spoke (dir is exactly +-kOne)");
        std::printf("FF1 radial ring: N=%d disp=%d (exact 32*N*(N+1)) vel=%d\n", N, disp, 4096 * N);
    }

    // ---- (b) VORTEX ORBIT: exact first-tick tangential dv + orbit direction + pinned trajectory -------
    {
        const int N = 120;
        fpx::FxWorld w;
        w.gravity = fpx::FxVec3{0, 0, 0};
        w.groundY = Snap(-10.0);
        w.bodies.push_back(Body(Snap(3.0), 0, 0, Snap(0.25)));
        std::vector<ff::FieldVolume> vols(1, HugeVolume(ff::kFieldVortex));
        vols[0].center = fpx::FxVec3{0, 0, 0};
        vols[0].axis = fpx::FxVec3{0, kOne, 0};
        vols[0].strength = Snap(2.0);
        vols[0].falloff = ff::kFalloffNone;
        const fx r0 = fpx::FxLength(w.bodies[0].pos);
        std::vector<fx> traj;
        fx minZ = 0;
        static const std::vector<ff::FfCommand> kEmpty;
        for (int t = 0; t < N; ++t) {
            ff::SimFieldTick(vols, w, kEmpty, (uint32_t)t, dt60, 4);
            if (t == 0) {
                // tang = FxCross(+Y, rPerp=(3,0,0)) = (0,0,-3) -> dir (0,0,-kOne); mag = 2.0 ->
                // dv.z = fxmul(fxmul(-2*kOne, kOne), 1092) = -2184 EXACTLY. The orbit direction assert.
                check(w.bodies[0].vel.z == -2184,
                      "vortex: the first-tick tangential dv is EXACTLY -2184 (a +Y vortex pushes +x -> -z)");
                check(w.bodies[0].vel.x == 0 && w.bodies[0].vel.y == 0,
                      "vortex: the first-tick force is purely tangential (x/y untouched)");
            }
            traj.push_back(w.bodies[0].pos.x);
            traj.push_back(w.bodies[0].pos.z);
            if (w.bodies[0].pos.z < minZ) minZ = w.bodies[0].pos.z;
        }
        check(minZ < Snap(-1.0), "vortex: the body genuinely orbits (swings well into the -z half)");
        const fx rEnd = fpx::FxLength(w.bodies[0].pos);
        check(rEnd >= r0 - Snap(0.05),
              "vortex: the discrete orbit never collapses inward (spirals OUTWARD — honest: constant "
              "tangential force with no drag adds energy)");
        const uint64_t digest = TraceDigest(traj);
        std::printf("FF1 vortex: trajDigest=0x%016llx r0=%d rEnd=%d minZ=%d\n",
                    (unsigned long long)digest, r0, rEnd, minZ);
        check(digest == kPinVortexTraj, "vortex: the trajectory digest is pinned (MSVC == clang)");
    }

    // ---- (c) WIND VOLUME BOUNDED: pushes the inside body, BIT-EXACT ignores the outside body ----------
    {
        const int N = 120;
        auto makeWorld = []() {
            fpx::FxWorld w;
            w.gravity = fpx::FxVec3{0, Snap(-9.8), 0};
            w.groundY = 0;
            w.bodies.push_back(Body(0, Snap(0.5), 0, Snap(0.5)));         // A: inside the lane
            w.bodies.push_back(Body(Snap(20.0), Snap(0.5), 0, Snap(0.5)));  // B: OUTSIDE the AABB
            return w;
        };
        std::vector<ff::FieldVolume> vols(1);
        vols[0].kind = ff::kFieldWind;
        vols[0].bounds.lo = fpx::FxVec3{Snap(-2.0), 0, Snap(-2.0)};
        vols[0].bounds.hi = fpx::FxVec3{Snap(12.0), Snap(4.0), Snap(2.0)};
        vols[0].axis = fpx::FxVec3{kOne, 0, 0};
        vols[0].strength = Snap(3.0);
        fpx::FxWorld w = makeWorld();
        ff::StepFieldSteps(vols, w, dt60, 4, 0u, N);
        // The no-field control (empty volume list -> the identical fpx path).
        std::vector<ff::FieldVolume> none;
        fpx::FxWorld wc = makeWorld();
        ff::StepFieldSteps(none, wc, dt60, 4, 0u, N);
        const fpx::FxBody& bIn = w.bodies[0];
        const fpx::FxBody& bOut = w.bodies[1];
        const fpx::FxBody& bOutC = wc.bodies[1];
        check(bOut.pos.x == bOutC.pos.x && bOut.pos.y == bOutC.pos.y && bOut.pos.z == bOutC.pos.z &&
              bOut.vel.x == bOutC.vel.x && bOut.vel.y == bOutC.vel.y && bOut.vel.z == bOutC.vel.z &&
              bOut.orient.x == bOutC.orient.x && bOut.orient.w == bOutC.orient.w &&
              bOut.angVel.x == bOutC.angVel.x,
              "wind: the body OUTSIDE the AABB is BIT-EXACT vs the no-field run (the bounded-volume proof)");
        check(bIn.pos.x != wc.bodies[0].pos.x, "wind: the body INSIDE the lane is genuinely pushed");
        check(bIn.pos.x > Snap(5.0), "wind: 2 s of 3 wu/s^2 wind carries the resting sphere > 5 wu");
        check(bIn.pos.z == 0, "wind: a +x wind adds NO lateral drift");
        // Frictionless fpx floor (documented): the slide is smooth (no stick-slip); the exact final x is
        // pinned. NOTE 6.0 wu is the ideal 0.5*a*t^2; the integer ladder lands within an LSB band of it.
        check(absfx(bIn.pos.x - Snap(6.0)) < Snap(0.1),
              "wind: the displacement is the frictionless ballistic value (within the integer-ladder band)");
        std::printf("FF1 wind: inside disp x=%d (pin %d) outside untouched=1\n", bIn.pos.x, kPinWindDispX);
        check(bIn.pos.x == kPinWindDispX, "wind: the exact displacement is pinned (MSVC == clang)");
    }

    // ---- (d) IDENTITY-AT-ZERO: no volumes / all-disabled == plain fpx BIT-EXACT -----------------------
    {
        fpx::FxWorld init;
        init.gravity = fpx::FxVec3{0, Snap(-9.8), 0};
        init.groundY = 0;
        init.bodies.push_back(Body(0, Snap(3.0), 0, Snap(0.5)));
        init.bodies.push_back(Body(Snap(0.6), Snap(5.0), 0, Snap(0.5)));
        init.bodies.push_back(Body(Snap(10.0), Snap(2.0), 0, Snap(0.5)));
        std::vector<fpx::FxCommand> fxStream;
        fxStream.push_back(fpx::FxCommand{20u, fpx::kCmdImpulse, 0u, fpx::FxVec3{Snap(0.8), 0, 0}});
        fxStream.push_back(fpx::FxCommand{40u, fpx::kCmdImpulse, 2u, fpx::FxVec3{0, Snap(2.0), Snap(0.3)}});
        std::vector<ff::FfCommand> ffStream;
        for (const fpx::FxCommand& c : fxStream)
            ffStream.push_back(ff::FfCommand{c.tick, ff::kFfCmdBody, c, 0u, 0u});
        const fpx::FxWorld plain = fpx::RunLockstep(init, fxStream, 90, dt60, 4);
        const fpx::FxWorld noVols =
            ff::RunFieldLockstep(std::vector<ff::FieldVolume>{}, init, ffStream, 90, dt60, 4);
        check(ff::FfWorldsEqual(noVols, plain),
              "identity-at-zero: NO volumes -> RunFieldLockstep == fpx::RunLockstep BIT-EXACT");
        std::vector<ff::FieldVolume> disabled = ff::ShowcaseVolumes();
        for (ff::FieldVolume& v : disabled) v.enabled = 0u;
        const fpx::FxWorld allOff = ff::RunFieldLockstep(disabled, init, ffStream, 90, dt60, 4);
        check(ff::FfWorldsEqual(allOff, plain),
              "identity-at-zero: ALL-DISABLED volumes -> == fpx::RunLockstep BIT-EXACT");
    }

    // ---- (e) FALLOFF: inverse-square quarter-exact + the near-clamp plateau ---------------------------
    {
        ff::FieldVolume v = HugeVolume(ff::kFieldRadial);
        v.center = fpx::FxVec3{0, 0, 0};
        v.strength = Snap(8.0);                 // attract (the PT2 sign) -> F points toward the center
        v.falloff = ff::kFalloffInvSq;
        v.nearClamp = kOne;
        const fpx::FxVec3 f1 = ff::EvalFieldForce(v, fpx::FxVec3{Snap(2.0), 0, 0});
        const fpx::FxVec3 f2 = ff::EvalFieldForce(v, fpx::FxVec3{Snap(4.0), 0, 0});
        check(f1.x == -131072 && f1.y == 0 && f1.z == 0,
              "falloff: invsq at dist 2*near is EXACTLY strength/4 (8.0 -> 2.0, integer-exact)");
        check(f2.x == -32768, "falloff: invsq at dist 4*near is EXACTLY strength/16 (0.5)");
        check(f1.x == 4 * f2.x, "falloff: doubling the distance quarters the force EXACTLY");
        const fpx::FxVec3 fAt = ff::EvalFieldForce(v, fpx::FxVec3{kOne, 0, 0});
        const fpx::FxVec3 fNear = ff::EvalFieldForce(v, fpx::FxVec3{kOne / 2, 0, 0});
        check(fAt.x == -Snap(8.0), "falloff: at dist == nearClamp the force is EXACTLY full strength");
        check(fNear.x == fAt.x, "falloff: inside the near clamp the force PLATEAUS (pinned, no singularity)");
        std::printf("FF1 falloff: f(2)=%d f(4)=%d plateau=%d\n", f1.x, f2.x, fNear.x);
    }

    // ---- (f) LOCKSTEP + ROLLBACK with mid-run toggles (the bits-are-state proof) -----------------------
    {
        const std::vector<ff::FieldVolume> vols = ff::ShowcaseVolumes();
        const fpx::FxWorld init = ff::MakeFieldShotWorld();
        std::vector<ff::FfCommand> auth;
        auth.push_back(ff::FfCommand{20u, ff::kFfCmdToggle, fpx::FxCommand{}, 2u, 0u});   // wind OFF
        auth.push_back(ff::FfCommand{30u, ff::kFfCmdBody,
                                     fpx::FxCommand{30u, fpx::kCmdImpulse, 9u,
                                                    fpx::FxVec3{0, 0, Snap(0.5)}}, 0u, 0u});
        auth.push_back(ff::FfCommand{51u, ff::kFfCmdToggle, fpx::FxCommand{}, 2u, 1u});   // wind back ON
        const fpx::FxWorld authority = ff::RunFieldLockstep(vols, init, auth, 120, dt60, 4);
        const fpx::FxWorld replica = ff::RunFieldLockstep(vols, init, auth, 120, dt60, 4);
        check(ff::FfWorldsEqual(authority, replica),
              "lockstep: a peer fed ONLY the command stream (toggles included) re-derives bit-for-bit");
        std::printf("FF1 lockstep: authority digest=0x%016llx\n",
                    (unsigned long long)ff::FfDigest(authority));

        // The misprediction: at t51 the client toggles the WRONG volume (vortex OFF instead of wind ON).
        std::vector<ff::FfCommand> mis = auth;
        mis[2] = ff::FfCommand{51u, ff::kFfCmdToggle, fpx::FxCommand{}, 1u, 0u};
        const fpx::FxWorld misFull = ff::RunFieldLockstep(vols, init, mis, 120, dt60, 4);
        check(!ff::FfWorldsEqual(misFull, authority),
              "rollback control: the mispredicted toggle GENUINELY diverges");
        const fpx::FxWorld corrected =
            ff::RunFieldRollback(vols, init, auth, mis, 120, 50, dt60, 4);
        check(ff::FfWorldsEqual(corrected, authority),
              "rollback: restore(bodies + enabled BITS) + resim == authority BIT-EXACT");

        // The bits-are-state control: the same rollback but restoring the BODY snapshot ONLY (the
        // speculation's wrong vortex-OFF bit survives) -> the resim GENUINELY diverges from authority.
        {
            std::vector<ff::FieldVolume> vv = vols;
            fpx::FxWorld w = init;
            for (int t = 0; t < 50; ++t) ff::SimFieldTick(vv, w, auth, (uint32_t)t, dt60, 4);
            const fpx::FxWorld bodySnap = fpx::SnapshotWorld(w);
            for (int s = 0; s < 3; ++s) ff::SimFieldTick(vv, w, mis, (uint32_t)(50 + s), dt60, 4);
            fpx::RestoreWorld(w, bodySnap);   // bodies ONLY — vv keeps the wrong vortex-OFF bit
            for (int t = 50; t < 120; ++t) ff::SimFieldTick(vv, w, auth, (uint32_t)t, dt60, 4);
            check(!ff::FfWorldsEqual(w, authority),
                  "snapshot completeness: restoring bodies WITHOUT the enabled bits diverges (the "
                  "toggled bits ARE state)");
        }
    }

    // ---- (g) The showcase scenario: two-run identical + the pinned digests + the story ----------------
    {
        const ff::FieldShotRun run = ff::RunFieldShotScenario();
        const ff::FieldShotRun run2 = ff::RunFieldShotScenario();
        check(run.digest == run2.digest && run.traceDigest == run2.traceDigest,
              "scenario: two runs byte-identical (digest + trace)");
        std::printf("FF1 scenario: digest=0x%016llx trace=0x%016llx\n",
                    (unsigned long long)run.digest, (unsigned long long)run.traceDigest);
        check(run.digest == kPinScenario, "scenario: the final-state digest is pinned (MSVC == clang)");
        check(run.traceDigest == kPinScenarioTrace, "scenario: the trace digest is pinned");
        const fpx::FxWorld init = ff::MakeFieldShotWorld();
        // The burst ring expands (every ring body ends farther from the burst center than it started).
        const fpx::FxVec3 bc{Snap(-6.0), Snap(0.4), Snap(-6.0)};
        bool expanded = true;
        for (int i = 0; i < 6; ++i) {
            const fx dNow = fpx::FxLength(fpx::FxSub(run.finalWorld.bodies[i].pos, bc));
            const fx dWas = fpx::FxLength(fpx::FxSub(init.bodies[i].pos, bc));
            if (dNow <= dWas + Snap(1.0)) expanded = false;
        }
        check(expanded, "scenario: the radial burst blasts EVERY ring body outward (> 1 wu)");
        // The wind boxes cross the lane (+x); no lateral z teleports out of the lane band.
        bool blown = true;
        for (int i = 9; i < 12; ++i)
            if (run.finalWorld.bodies[i].pos.x <= init.bodies[i].pos.x + Snap(5.0)) blown = false;
        check(blown, "scenario: the wind lane carries all three boxes > 5 wu downwind");
        // The raster is deterministic (the showcase golden's substance).
        std::vector<uint8_t> imgA, imgB;
        uint32_t wA = 0, hA = 0, wB = 0, hB = 0;
        ff::RenderFieldShot(run, imgA, wA, hA);
        ff::RenderFieldShot(run2, imgB, wB, hB);
        check(wA == wB && hA == hB && imgA == imgB, "scenario: two rasters BYTE-IDENTICAL");
        check(wA == 560u && hA == 560u, "scenario: the raster is the documented 560x560");
    }

    if (g_fail == 0) std::printf("force_field_test: ALL PASS\n");
    else std::printf("force_field_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
