#pragma once
// Slice WV1 — THE WATER GAMEPLAY VOLUME (parity++ #2): connect the Gerstner RENDER surface to
// rigid-body ARCHIMEDES buoyancy — a boat bobbing on the rendered ocean. engine/render/water.h ships a
// Gerstner-wave render surface (float, in-shader displacement) and engine/sim/couple.h CP7 ships exact
// integer Archimedes buoyancy (SphereCapVolume) — but they were NOT CONNECTED: CP7's buoyancy world is a
// separate SPH-particle basin, and there was no water VOLUME object a rigid body could float on. WV1
// unifies them: a WaterBody whose ANALYTIC Gerstner surface height drives rigid-body Archimedes buoyancy
// + drag — the rendered ocean and the physics ocean are the same equation. UE5's Water plugin does this
// with float physics; ours is bit-exact + lockstep-replayable. Pure CPU, header-only, NO device, NO
// backend symbols, NO <cmath> (the ONE transcendental — sine — comes from the HOST-BAKED ik.h LUT).
// Namespace hf::sim::water.
//
// ===================================================================================================
// THE DETERMINISM DECISION (made consciously, documented): the render Gerstner (render/water.h) is
// FLOAT; the sim must be INTEGER. WV1 therefore defines the CANONICAL Gerstner in Q16.16 INTEGER (a
// fixed wave set: <=4 waves with Q16.16 amplitude/wavelength/speed/direction; sine via the HOST-BAKED
// sine LUT that already ships in anim/ik.h — FxSinLut/FxCosLut, the seq.h SineEaseTable / ik.h
// host-baked-LUT precedent — ZERO runtime transcendentals). The integer surface height h(x,z,tick) is
// THE ground truth the physics uses. The RENDER then evaluates the same wave set in float in the shader
// (visually identical within the float-class band) — the render is the FLOAT MIRROR of the INTEGER
// truth, the INVERSE of the usual "shader copies the CPU" direction. HONEST NOTE: render-vs-sim height
// can differ by float rounding + the LUT's within-LSB band (invisible on screen, but stated — the two
// are the same EQUATION, not the same BITS).
//
// THE v1 PHYSICS SCOPE (documented gap): the sim surface is HEIGHT-ONLY — a Gerstner wave also displaces
// points HORIZONTALLY (the sharp-crest pinch, Q*A*D*cos(theta)); that horizontal displacement affects
// the VISUALS (the render keeps it) but is OMITTED from the physics surface in v1 (the physics reads the
// vertical sum at the body's UNDISPLACED (x,z) — a standard game-physics simplification; the horizontal
// orbital velocity of the water is likewise omitted from drag, which uses only the analytic VERTICAL
// surface velocity). A body's submerged volume vs a LOCALLY-FLAT surface at h(x,z,tick) is itself an
// approximation of the volume under a curved surface (exact when the wavelength >> the body radius).
//
// THE ELEGANT BIT (the headline): the water is ANALYTIC + STATELESS — h(x,z,tick) is a pure function of
// the wave set and the tick, so an INFINITE deterministic ocean costs a ZERO-BYTE water snapshot. The
// lockstep/rollback snapshot is the fpx body vector ALONE; a peer (or a rollback restore) re-derives the
// exact wave field from the tick number. UE5's water physics cannot make that claim (float + stateful).
//
// REUSE MAP (read-only #includes — fpx.h / couple.h / anim/ik.h are byte-UNTOUCHED):
//   * sim/fpx.h    — fx/FxVec3/FxBody/FxWorld/fxmul/fxdiv/FxLength/FxNormalize/StepWorld/BuildPairs/
//                    IntegrateOrientation + the FPX5 lockstep mold (FxCommand/ApplyCommand/SnapshotWorld/
//                    RestoreWorld) reused VERBATIM.
//   * sim/couple.h — CP7's SphereCapVolume (the EXACT integer spherical-cap volume), SphereVolume,
//                    BodyFromDensity, SubmergedDepth — the Archimedes toolbox, reused VERBATIM against
//                    the ANALYTIC surface instead of the SPH pool-quantile estimator (TIGHTER: no
//                    particle-spacing quantization, no splash sensitivity).
//   * anim/ik.h    — FxSinLut/FxCosLut (256-bin host-baked Q16.16 trig LUT + int32 lerp) + kTwoPi. The
//                    "zero runtime transcendental" doctrine: the LUT is built ONCE from <cmath> at first
//                    use (host bake), the hot path is pure int32.
//
// INTEGER DOMAIN BOUNDS (documented): the spatial phase k*(D.p) is a Q16.16 int32 — it must stay within
// +-32767 RADIANS, i.e. |dirX*x + dirZ*z| <= 32767 / k world units (for the showcase waves k <= ~3.3 ->
// a ~10,000-wu playfield; far beyond the WaterBody AABBs used). The time phase is folded mod 2*pi in
// int64 BEFORE narrowing (exact for any uint32 tick). SphereCapVolume bounds the body radius to r <= 16
// wu (the CP7 int64 triple-product bound).
//
// THE GPU CONVENTION (the CP7/GAS1 precedent): WV1 ships PURE CPU on BOTH backends — NO GPU dispatch,
// NO new shader, NO new RHI; the Vulkan --wv1-float-shot and the Metal --wv1-float run THIS identical
// integer code, so the showcase golden is bit-identical cross-backend BY CONSTRUCTION (strict zero-
// differing-pixel). A LIT hero — the boat on the RENDERED Gerstner ocean — is the natural WV2 capstone
// (the render/water.h float mirror + this sim in one frame); NOT built in WV1.
//
// HONEST CAVEATS: (a) height-only physics surface + vertical-only water velocity (above). (b) The
// bob equilibrium is a damped forced oscillation — drag introduces a real PHASE LAG vs the wave, and the
// discrete integer integrator adds its own band; the tests pin the MEASURED period against the wave
// period within a tick band rather than forcing equality. (c) Buoyancy samples the PRE-STEP body
// position (deterministic, one-frame explicit — the CP2/CP7 convention). (d) The volume AABB gates
// buoyancy by the body CENTER's XZ footprint (a body straddling the volume edge is all-in or all-out).

#include <cstdint>
#include <vector>

#include "sim/fpx.h"     // read-only: the fixed-point substrate + rigid solver + FPX5 lockstep mold
#include "sim/couple.h"  // read-only: CP7 SphereCapVolume / SphereVolume / BodyFromDensity / SubmergedDepth
#include "anim/ik.h"     // read-only: FxSinLut / FxCosLut / kTwoPi (the host-baked trig LUT)

namespace hf::sim {
namespace water {

// Reuse the fpx Q16.16 toolbox verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::FxAdd;
using fpx::FxSub;
using fpx::FxScale;
inline constexpr int kFrac = fpx::kFrac;   // 16
inline constexpr fx  kOne  = fpx::kOne;    // 65536
inline constexpr fx  kTwoPi = hf::anim::ik::kTwoPi;   // 411775 (Q16.16 2*pi, the ik.h host snap)

// ----- The wave set + the water volume ---------------------------------------------------------------
// One integer Gerstner wave (height-only v1). dir(X,Z) is the (assumed-unit, host-snapped Q16.16)
// horizontal propagation direction; amp the wave height; wavelength crest-to-crest; speed the phase
// speed (w = speed * k, k = 2*pi / wavelength — the render/water.h parameterization, in Q16.16).
struct WaterWave {
    fx amp        = 0;      // A (Q16.16 world units); 0 -> the wave contributes EXACTLY nothing
    fx wavelength = kOne;   // L (Q16.16 world units, > 0)
    fx speed      = 0;      // S (Q16.16 wu/s); phase speed, w = S*k
    fx dirX       = kOne;   // D.x (Q16.16, unit with dirZ — host-snapped literals)
    fx dirZ       = 0;      // D.z
};

// THE WATER GAMEPLAY VOLUME: an AABB of water whose surface is the analytic integer Gerstner sum.
// bounds.hi.y is the STILL-WATER level (waves displace around it); bounds.lo.y the floor (a dense body
// sinks to it via the fpx ground clamp — scenes set FxWorld::groundY = bounds.lo.y). The XZ extent gates
// which bodies the volume acts on. STATELESS: stepping the world never mutates a WaterBody.
struct WaterBody {
    fpx::FxAabb bounds{};        // the water volume (lo <= hi); hi.y = still level, lo.y = floor
    WaterWave   waves[4]{};      // the FIXED wave set (golden-stable; <= 4 waves)
    uint32_t    waveCount = 0;   // active waves in waves[]
    fx          waterDensity = 0;   // rho_f (Q16.16); buoyant force = rho_f * |g| * V_submerged. 0 -> OFF
    fx          dragK        = 0;   // linear drag vs the water's local velocity (per second). 0 -> OFF
    fx          tickDt       = kOne / 60;   // the fixed tick dt the tick->phase conversion uses
};

// ----- The canonical INTEGER Gerstner surface (the ground truth the physics uses) ---------------------
// WavePhase: theta = k*(D.x*x + D.z*z) - w*t at tick t. k = fxdiv(2*pi, L); w = fxmul(S, k). The time
// phase w*dt*tick is formed EXACTLY in int64 then folded mod 2*pi BEFORE narrowing to int32 (exact for
// any uint32 tick — no drift, no overflow); the spatial phase is the documented +-32767-radian domain.
inline fx WavePhase(const WaterWave& wv, fx tickDt, fx x, fx z, uint32_t tick) {
    if (wv.wavelength <= 0) return 0;                          // degenerate wave -> deterministic 0
    const fx k   = fpx::fxdiv(kTwoPi, wv.wavelength);          // spatial angular frequency (Q16.16)
    const fx w   = fpx::fxmul(wv.speed, k);                    // temporal angular frequency
    const fx dot = fpx::fxmul(wv.dirX, x) + fpx::fxmul(wv.dirZ, z);
    const fx spatial = fpx::fxmul(k, dot);                     // k*(D.p), the bounded spatial phase
    const fx wdt = fpx::fxmul(w, tickDt);                      // radians per tick (Q16.16)
    const fx timePhase = (fx)(((int64_t)wdt * (int64_t)tick) % (int64_t)kTwoPi);   // exact int64 fold
    return spatial - timePhase;
}

// SurfaceHeight(body, x, z, tick) -> the ABSOLUTE world-space water surface height at (x,z) and tick:
// bounds.hi.y + sum_j amp_j * sin(theta_j). HEIGHT-ONLY v1 (the horizontal Gerstner displacement is
// render-only — documented above). Pure integer + the host-baked sine LUT; amps=0 -> EXACTLY hi.y (the
// identity-at-zero: still water). This is the equation the render mirrors in float.
inline fx SurfaceHeight(const WaterBody& body, fx x, fx z, uint32_t tick) {
    fx h = body.bounds.hi.y;
    for (uint32_t j = 0; j < body.waveCount && j < 4u; ++j) {
        const WaterWave& wv = body.waves[j];
        if (wv.amp == 0) continue;                             // a zero wave contributes exactly nothing
        h += fpx::fxmul(wv.amp, hf::anim::ik::FxSinLut(WavePhase(wv, body.tickDt, x, z, tick)));
    }
    return h;
}

// SurfaceVelY(body, x, z, tick) -> the ANALYTIC vertical velocity of the surface (wu/s): the exact time
// derivative of the height sum, d/dt [A*sin(k*D.p - w*t)] = -A*w*cos(theta) — the integer differentiable
// form (the cosine from the same host-baked LUT; NO finite difference, NO extra state). The drag target.
inline fx SurfaceVelY(const WaterBody& body, fx x, fx z, uint32_t tick) {
    fx vy = 0;
    for (uint32_t j = 0; j < body.waveCount && j < 4u; ++j) {
        const WaterWave& wv = body.waves[j];
        if (wv.amp == 0 || wv.wavelength <= 0) continue;
        const fx k = fpx::fxdiv(kTwoPi, wv.wavelength);
        const fx w = fpx::fxmul(wv.speed, k);
        const fx c = hf::anim::ik::FxCosLut(WavePhase(wv, body.tickDt, x, z, tick));
        vy -= fpx::fxmul(fpx::fxmul(wv.amp, w), c);
    }
    return vy;
}

// InFootprint: the volume gate — the body CENTER's (x,z) inside the AABB's XZ extent (inclusive; the
// documented all-in-or-all-out caveat). Pure integer compares.
inline bool InFootprint(const WaterBody& body, const FxVec3& pos) {
    return pos.x >= body.bounds.lo.x && pos.x <= body.bounds.hi.x &&
           pos.z >= body.bounds.lo.z && pos.z <= body.bounds.hi.z;
}

// ----- StepFloatBody: the per-body Archimedes buoyancy + drag against the ANALYTIC surface ------------
// For ONE dynamic body (the CP7 force model, the SPH pool-quantile estimator replaced by the exact
// analytic local surface — tighter, no particle quantization):
//   * identity-at-zero: waterDensity==0 AND dragK==0 -> return before ANY math (an EXACT no-op).
//   * outside the XZ footprint -> free-fall (no buoyancy, no drag).
//   * h = SurfaceHeight at the body's PRE-STEP (x,z); d = couple::SubmergedDepth(b, h) (clamped [0,2r]);
//     d == 0 (clear of the local surface) -> free-fall.
//   * BUOYANCY: V = couple::SphereCapVolume(radius, d) (the CP7 exact cap); up = -normalize(gravity)
//     (+Y fallback at gravity 0 — the fpx FxNormalize contract); F = rho_f * |g| * V;
//     vel += up * F * invMass * dt (the invMass-scaled force -> the exact Archimedes balance:
//     equilibrium where V(d)/V_total == rho_body/rho_fluid — rho_b = 0.5 -> HALF-submerged, d = r).
//   * DRAG: linear relaxation toward the water's local velocity (0, SurfaceVelY, 0) — the wave's
//     analytic vertical motion drives the body (the bob forcing); horizontal water motion omitted (v1).
//     vel += dragK * (vWater - vel) * dt per axis, ONLY while submerged (d > 0).
// Deterministic: pure integer, fixed op order; samples the PRE-STEP position (the CP2/CP7 convention).
inline void StepFloatBody(const WaterBody& water, fpx::FxBody& b, const FxVec3& gravity, fx dt,
                          uint32_t tick) {
    if (!(b.flags & fpx::kFlagDynamic)) return;                 // static/kinematic -> untouched
    if (water.waterDensity == 0 && water.dragK == 0) return;    // identity-at-zero: an EXACT no-op
    if (!InFootprint(water, b.pos)) return;                     // outside the volume -> free-fall
    const fx h = SurfaceHeight(water, b.pos.x, b.pos.z, tick);  // the local analytic surface
    const fx d = couple::SubmergedDepth(b, h);                  // depth of the sphere bottom below h
    if (d <= 0) return;                                         // clear of the water -> free-fall

    // BUOYANCY: F = rho_f * |g| * V_cap(d) along -normalize(gravity); dv = F * invMass * dt.
    if (water.waterDensity > 0) {
        const FxVec3 up = fpx::FxNormalize(FxVec3{-gravity.x, -gravity.y, -gravity.z});
        const fx gmag = fpx::FxLength(gravity);
        const fx vol = couple::SphereCapVolume(b.radius, d);
        const fx buoyMag = fpx::fxmul(fpx::fxmul(water.waterDensity, gmag), vol);
        b.vel = FxAdd(b.vel, FxScale(FxScale(FxScale(up, buoyMag), b.invMass), dt));
    }

    // DRAG: dv = dragK * (vWater - vel) * dt (linear; vWater = the analytic vertical surface velocity).
    if (water.dragK > 0) {
        const FxVec3 vWater{0, SurfaceVelY(water, b.pos.x, b.pos.z, tick), 0};
        const FxVec3 rel = FxSub(vWater, b.vel);
        b.vel.x += fpx::fxmul(fpx::fxmul(water.dragK, rel.x), dt);
        b.vel.y += fpx::fxmul(fpx::fxmul(water.dragK, rel.y), dt);
        b.vel.z += fpx::fxmul(fpx::fxmul(water.dragK, rel.z), dt);
    }
}

// ApplyWaterForces: StepFloatBody over EVERY body in DETERMINISTIC ASCENDING order (each body is
// independent — the water never reacts back in v1; a one-way analytic field, so the order is moot but
// pinned anyway, the CP2 discipline).
inline void ApplyWaterForces(const WaterBody& water, fpx::FxWorld& w, fx dt, uint32_t tick) {
    const size_t n = w.bodies.size();
    for (size_t i = 0; i < n; ++i)
        StepFloatBody(water, w.bodies[i], w.gravity, dt, tick);
}

// ----- The composed water tick (the FPX5 SimTick mold with the water exchange prepended) ---------------
// SimWaterTick: (1) apply this tick's commands in ARRAY ORDER (fpx::ApplyCommand VERBATIM — external
// impulses, e.g. a push on the boat); (2) ApplyWaterForces (buoyancy + drag from the analytic surface at
// THIS tick); (3) re-broadphase + fpx::StepWorld (integrate + contacts — the boat still collides with
// other bodies / the floor); (4) IntegrateOrientation per body (the FPX5 6-DOF convention). With a
// zero WaterBody (density==dragK==0) step (2) is an EXACT no-op, so SimWaterTick == fpx::SimTick
// BIT-FOR-BIT (the identity-at-zero proof the test pins).
inline void SimWaterTick(const WaterBody& water, fpx::FxWorld& w,
                         const std::vector<fpx::FxCommand>& stream, uint32_t tick, fx dt, int iters) {
    for (const fpx::FxCommand& c : stream)
        if (c.tick == tick) fpx::ApplyCommand(w, c);
    ApplyWaterForces(water, w, dt, tick);
    std::vector<uint32_t> offsets;
    std::vector<fpx::FxPair> pairs;
    fpx::BuildPairs(w, offsets, pairs);
    fpx::StepWorld(w, std::span<const fpx::FxPair>(pairs), dt, iters);
    for (fpx::FxBody& b : w.bodies) fpx::IntegrateOrientation(b, dt);
}

// StepWaterSteps: run K ticks starting at startTick (the test/showcase driver).
inline void StepWaterSteps(const WaterBody& water, fpx::FxWorld& w, fx dt, int iters, uint32_t startTick,
                           int steps) {
    static const std::vector<fpx::FxCommand> kEmpty;
    for (int s = 0; s < steps; ++s)
        SimWaterTick(water, w, kEmpty, startTick + (uint32_t)s, dt, iters);
}

// ----- LOCKSTEP + ROLLBACK (the FPX5 mold, VERBATIM — commands/snapshot reused, step fn swapped) --------
// THE ZERO-BYTE WATER SNAPSHOT (the headline): the snapshot is fpx::SnapshotWorld(w) — the BODY vector
// ALONE. The water is analytic + stateless (h is a pure function of the wave set + tick), so an INFINITE
// deterministic ocean adds ZERO bytes to the rollback state; restoring the body snapshot + re-simulating
// from the tick number re-derives the exact wave field for free. Commands = external impulses on the
// boat (fpx::kCmdImpulse), the wire format a netcode layer sends.

// RunWaterLockstep: the peer entry point — run `ticks` SimWaterTicks from a COPY of init. authority and
// replica fed the SAME init + stream (inputs ONLY) re-derive the floating state BIT-IDENTICALLY.
inline fpx::FxWorld RunWaterLockstep(const WaterBody& water, const fpx::FxWorld& init,
                                     const std::vector<fpx::FxCommand>& stream, int ticks, fx dt,
                                     int iters) {
    fpx::FxWorld w = init;
    for (int t = 0; t < ticks; ++t)
        SimWaterTick(water, w, stream, (uint32_t)t, dt, iters);
    return w;
}

// RunWaterRollback: the FPX5 RunRollback harness with the step fn swapped. (1) advance to mispredictTick
// with the authoritative stream; (2) SNAPSHOT (fpx bodies ONLY — the water snapshots nothing); (2b)
// speculate <=3 ticks with the MISPREDICTED stream (the diverging client prediction); (3) restore +
// re-simulate the correct stream -> the corrected world == RunWaterLockstep(authStream) exactly.
inline fpx::FxWorld RunWaterRollback(const WaterBody& water, const fpx::FxWorld& init,
                                     const std::vector<fpx::FxCommand>& authStream,
                                     const std::vector<fpx::FxCommand>& mispredictStream, int ticks,
                                     int mispredictTick, fx dt, int iters) {
    fpx::FxWorld w = init;
    for (int t = 0; t < mispredictTick; ++t)
        SimWaterTick(water, w, authStream, (uint32_t)t, dt, iters);
    const fpx::FxWorld snap = fpx::SnapshotWorld(w);   // fpx bodies ONLY — the water is a ZERO-BYTE snapshot
    int specTicks = ticks - mispredictTick;
    if (specTicks > 3) specTicks = 3;
    for (int s = 0; s < specTicks; ++s)
        SimWaterTick(water, w, mispredictStream, (uint32_t)(mispredictTick + s), dt, iters);
    fpx::RestoreWorld(w, snap);
    for (int t = mispredictTick; t < ticks; ++t)
        SimWaterTick(water, w, authStream, (uint32_t)t, dt, iters);
    return w;
}

// ----- WaterDigest: the FNV-1a-64 pin over the full body state (the CoupleDigest mold, bodies only —
// the water contributes NO state to hash, BY CONSTRUCTION). Field-wise byte mixing (layout-independent).
inline uint64_t WaterDigest(const fpx::FxWorld& w) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint32_t v) {
        for (int b = 0; b < 4; ++b) {
            h ^= (uint64_t)((v >> (b * 8)) & 0xFFu);
            h *= 1099511628211ull;
        }
    };
    for (const fpx::FxBody& b : w.bodies) {
        mix((uint32_t)b.pos.x);    mix((uint32_t)b.pos.y);    mix((uint32_t)b.pos.z);
        mix((uint32_t)b.vel.x);    mix((uint32_t)b.vel.y);    mix((uint32_t)b.vel.z);
        mix((uint32_t)b.invMass);  mix(b.flags);              mix((uint32_t)b.radius);
        mix((uint32_t)b.orient.x); mix((uint32_t)b.orient.y); mix((uint32_t)b.orient.z);
        mix((uint32_t)b.orient.w);
        mix((uint32_t)b.angVel.x); mix((uint32_t)b.angVel.y); mix((uint32_t)b.angVel.z);
    }
    return h;
}

// WaterWorldsEqual: field-wise bit equality of two body vectors (the lockstep memcmp, layout-independent).
inline bool WaterWorldsEqual(const fpx::FxWorld& a, const fpx::FxWorld& b) {
    if (a.bodies.size() != b.bodies.size()) return false;
    for (size_t i = 0; i < a.bodies.size(); ++i) {
        const fpx::FxBody& x = a.bodies[i];
        const fpx::FxBody& y = b.bodies[i];
        if (x.pos.x != y.pos.x || x.pos.y != y.pos.y || x.pos.z != y.pos.z) return false;
        if (x.vel.x != y.vel.x || x.vel.y != y.vel.y || x.vel.z != y.vel.z) return false;
        if (x.invMass != y.invMass || x.flags != y.flags || x.radius != y.radius) return false;
        if (x.orient.x != y.orient.x || x.orient.y != y.orient.y || x.orient.z != y.orient.z ||
            x.orient.w != y.orient.w) return false;
        if (x.angVel.x != y.angVel.x || x.angVel.y != y.angVel.y || x.angVel.z != y.angVel.z) return false;
    }
    return true;
}

// ===================================================================================================
// THE FIXED SHOWCASE SCENARIO + the pure-integer side-view raster (shared VERBATIM by the Vulkan
// --wv1-float-shot and the Metal --wv1-float so the golden is bit-identical cross-backend BY
// CONSTRUCTION — the GAS1 shared-scenario discipline, with the RASTER also shared to remove even
// copy-drift risk). All host-snapped constants; NO RNG, NO clock.
// ===================================================================================================

// Snap(v): round-to-nearest Q16.16 of a host double (ties away from zero — the cloth.h restLen snap).
// Scene-build-time only (the golden-stable literals below); the per-tick sim never touches double.
inline constexpr fx Snap(double v) {
    const double s = v * (double)kOne;
    return (fx)(s + (s < 0 ? -0.5 : 0.5));
}

// The FIXED showcase ocean: a 16x16-wu volume, still level y=4, floor y=0, three crossing waves (the
// render/water.h ShowcaseWaves flavor, host-snapped to Q16.16 — the unit-direction literals shared).
inline WaterBody ShowcaseWater() {
    WaterBody wb;
    wb.bounds.lo = FxVec3{Snap(-8.0), 0, Snap(-8.0)};
    wb.bounds.hi = FxVec3{Snap(8.0), Snap(4.0), Snap(8.0)};
    wb.waves[0] = WaterWave{Snap(0.28), Snap(5.0), Snap(1.5), kOne, 0};
    wb.waves[1] = WaterWave{Snap(0.14), Snap(3.0), Snap(1.1), Snap(0.8), Snap(0.6)};
    wb.waves[2] = WaterWave{Snap(0.08), Snap(1.9), Snap(1.9), Snap(-0.50702014), Snap(0.86193424)};
    wb.waveCount = 3;
    wb.waterDensity = kOne;      // rho_f = 1.0
    wb.dragK = Snap(2.0);        // linear drag 2.0 / s
    wb.tickDt = kOne / 60;
    return wb;
}

// The showcase run: 3 density-authored spheres (light 0.35 / medium 0.60 / dense 1.50, r=0.5) dropped
// at different (x,z) onto the showcase ocean, stepped kShotSteps ticks. The dense one sinks to the AABB
// floor; the light/medium ones ride DIFFERENT wave phases. Per-tick body-y traces feed the trace digest.
inline constexpr int      kShotSteps  = 480;   // 8 s at 60 Hz — settled + mid-bob
inline constexpr uint32_t kShotBodies = 3;

struct WaterShotRun {
    fpx::FxWorld finalWorld;
    WaterBody    waterUsed;
    uint64_t     digest = 0;         // WaterDigest(finalWorld)
    uint64_t     traceDigest = 0;    // FNV over every tick's 3 body-y values (the phase-locked trace)
    fx           depths[3] = {0, 0, 0};   // final submerged depth per body vs the analytic local surface
};

inline fpx::FxWorld MakeShotWorld() {
    fpx::FxWorld w;
    w.gravity = FxVec3{0, Snap(-9.8), 0};
    w.groundY = 0;   // == the water AABB floor (bounds.lo.y)
    const fx r = Snap(0.5);
    w.bodies.push_back(couple::BodyFromDensity(FxVec3{Snap(-4.0), Snap(5.0), 0}, r, Snap(0.35)));  // light
    w.bodies.push_back(couple::BodyFromDensity(FxVec3{0, Snap(5.0), 0}, r, Snap(0.60)));           // medium
    w.bodies.push_back(couple::BodyFromDensity(FxVec3{Snap(4.0), Snap(5.0), 0}, r, Snap(1.50)));   // dense
    return w;
}

inline WaterShotRun RunWaterShotScenario() {
    WaterShotRun run;
    run.waterUsed = ShowcaseWater();
    run.finalWorld = MakeShotWorld();
    uint64_t th = 1469598103934665603ull;
    auto mix = [&th](uint32_t v) {
        for (int b = 0; b < 4; ++b) {
            th ^= (uint64_t)((v >> (b * 8)) & 0xFFu);
            th *= 1099511628211ull;
        }
    };
    static const std::vector<fpx::FxCommand> kEmpty;
    for (int t = 0; t < kShotSteps; ++t) {
        SimWaterTick(run.waterUsed, run.finalWorld, kEmpty, (uint32_t)t, run.waterUsed.tickDt, 4);
        for (uint32_t i = 0; i < kShotBodies; ++i)
            mix((uint32_t)run.finalWorld.bodies[i].pos.y);
    }
    run.traceDigest = th;
    run.digest = WaterDigest(run.finalWorld);
    for (uint32_t i = 0; i < kShotBodies; ++i) {
        const fpx::FxBody& b = run.finalWorld.bodies[i];
        const fx h = SurfaceHeight(run.waterUsed, b.pos.x, b.pos.z, (uint32_t)kShotSteps);
        run.depths[i] = InFootprint(run.waterUsed, b.pos) ? couple::SubmergedDepth(b, h) : 0;
    }
    return run;
}

// RenderWaterShot: the PURE-INTEGER side-view raster (z=0 slice) — the wave surface polyline at the
// final tick + the water fill + the volume outline + the 3 spheres riding it. All geometry maps world
// Q16.16 -> pixels by integer 40-px/wu shifts; colors are fixed constants -> identical both backends.
// BGRA8, 640x384 (x in [-8,8], y in [-1.0,8.6] wu).
inline void RenderWaterShot(const WaterShotRun& run, std::vector<uint8_t>& bgra, uint32_t& outW,
                            uint32_t& outH) {
    const int W = 640, H = 384, kScale = 40;      // 40 px per world unit (uniform -> round spheres)
    const fx xMin = Snap(-8.0), yMin = Snap(-1.0);
    outW = (uint32_t)W; outH = (uint32_t)H;
    bgra.assign((size_t)W * H * 4, 0);
    for (size_t p = 0; p < (size_t)W * H; ++p) {
        bgra[p * 4 + 0] = 18; bgra[p * 4 + 1] = 12; bgra[p * 4 + 2] = 8; bgra[p * 4 + 3] = 255;
    }
    auto putPx = [&](int ix, int iy, uint8_t r, uint8_t g, uint8_t b) {
        if (ix < 0 || ix >= W || iy < 0 || iy >= H) return;
        uint8_t* dst = &bgra[((size_t)iy * W + ix) * 4];
        dst[0] = b; dst[1] = g; dst[2] = r; dst[3] = 255;
    };
    auto mapX = [&](fx x) { return (int)(((int64_t)(x - xMin) * kScale) >> kFrac); };
    auto mapY = [&](fx y) { return H - 1 - (int)(((int64_t)(y - yMin) * kScale) >> kFrac); };

    const WaterBody& wb = run.waterUsed;
    const uint32_t tick = (uint32_t)kShotSteps;   // the FIXED sample tick (the final state's tick)

    // (1) The water column per pixel-x: sample the integer surface at each column's world x (z=0).
    for (int px = 0; px < W; ++px) {
        const fx xWu = xMin + (fx)(((int64_t)px << kFrac) / kScale);
        if (xWu < wb.bounds.lo.x || xWu > wb.bounds.hi.x) continue;
        const fx h = SurfaceHeight(wb, xWu, 0, tick);
        const int ySurf = mapY(h), yFloor = mapY(wb.bounds.lo.y);
        for (int py = ySurf; py <= yFloor && py < H; ++py) {
            if (py < 0) continue;
            putPx(px, py, 24, 66, 118);                       // the water body: deep blue
        }
        putPx(px, ySurf, 96, 190, 235);                       // the surface polyline (2 px)
        putPx(px, ySurf + 1, 60, 130, 190);
    }
    // (2) The volume AABB outline (side view) + the floor line.
    {
        const int x0 = mapX(wb.bounds.lo.x), x1 = mapX(wb.bounds.hi.x);
        const int yTop = mapY(wb.bounds.hi.y), yBot = mapY(wb.bounds.lo.y);
        for (int px = x0; px <= x1; ++px) { putPx(px, yBot, 120, 96, 60); putPx(px, yTop, 70, 70, 74); }
        for (int py = yTop; py <= yBot; ++py) { putPx(x0, py, 70, 70, 74); putPx(x1, py, 70, 70, 74); }
    }
    // (3) The 3 spheres (light warm-yellow / medium orange / dense slate), filled integer discs.
    const uint8_t colR[3] = {235, 220, 120}, colG[3] = {200, 120, 126}, colB[3] = {90, 50, 134};
    for (uint32_t i = 0; i < kShotBodies && i < (uint32_t)run.finalWorld.bodies.size(); ++i) {
        const fpx::FxBody& b = run.finalWorld.bodies[i];
        const int cx = mapX(b.pos.x), cy = mapY(b.pos.y);
        const int rPx = (int)(((int64_t)b.radius * kScale) >> kFrac);
        for (int dy = -rPx; dy <= rPx; ++dy)
            for (int dx = -rPx; dx <= rPx; ++dx)
                if (dx * dx + dy * dy <= rPx * rPx) putPx(cx + dx, cy + dy, colR[i], colG[i], colB[i]);
        putPx(cx, cy, 20, 20, 20);                            // center mark
    }
}

}  // namespace water
}  // namespace hf::sim
