#pragma once
// Slice FF1 — RIGID-BODY FORCE-FIELD VOLUMES (parity++ honorable mention: "the cheapest of all"): the
// PT2 particle force-field math (point/vortex/wind — engine/sim/particles.h) applied to RIGID BODIES,
// with bounded AABB VOLUMES. UE5 ships Physics Fields (a radial impulse blowing debris outward, a wind
// volume pushing crates across a floor); Hazard Forge had the field math ONLY for particles — fpx rigid
// bodies had no authored field volumes. FF1 closes that: an authored FieldVolume (radial burst / vortex
// swirl / wind lane) mutates body velocities PRE-STEP (the WV1 water_body.h discipline: analytic volume
// -> velocity mutation -> the untouched fpx::StepWorld), bit-exact Q16.16 integer on every platform AND
// lockstep/rollback-replayable — deterministic gameplay force fields UE5's float Physics Fields cannot
// replay bit-for-bit. Pure CPU, header-only, NO device, NO backend symbols, NO new RHI, NO float.
// Namespace hf::sim::ff.
//
// ===================================================================================================
// THE FIELD-MATH REUSE (the PT2 contract): particles.h is #included READ-ONLY and its int64 helpers
// (particles::FxDot / particles::FxCross — the convex.h-verbatim forms the PT2 shader copies) are used
// DIRECTLY. The per-kind force evaluation (EvalFieldForce below) is a documented TWIN of
// particles::AccumulateForce's per-field body — same ops, same order, same gates — rather than a call
// into it, because a FieldVolume needs (a) an AABB bound, (b) a falloff MODE (PT2 hard-codes the linear
// radius falloff), and (c) an enabled bit, none of which ForceField carries. FIDELITY IS PINNED, not
// asserted by hope: with falloff == kFalloffLinear the twin is BIT-EXACT equal to
// particles::AccumulateForce over a position grid (force_field_test's twin-fidelity proof), so the twin
// can never silently drift from the PT2 math.
//
// THE v1 PHYSICS SCOPE (documented gaps):
//   * LINEAR FORCE ONLY, applied at the center of mass — NO torque from field gradients (a real field
//     exerts torque on an extended body; v1 treats the body as a point at pos). A vortex still ORBITS
//     bodies linearly, which is the visible effect; body spin from fields is a future slice.
//   * The volume gates on the body CENTER (all-in-or-all-out at the AABB face — the WV1 footprint
//     caveat). A body straddling the boundary feels full force or none.
//   * Mass scaling: dv = F * invMass * dt (the WV1 buoyancy op order) — the field is a FORCE, so a
//     heavy crate (small invMass) drifts slower than debris in the same wind, as it should.
//   * fpx ground contact is a positional clamp with NO friction (ResolveGround) — a wind-pushed body
//     slides on the floor WITHOUT stick-slip and coasts forever once it exits the volume. The tests pin
//     the actual frictionless displacement.
//
// FALLOFF MODES (per-volume, `falloff`):
//   * kFalloffNone   — full strength everywhere inside the AABB (the PT2 kFieldWind convention).
//   * kFalloffLinear — the PT2 radius falloff VERBATIM: (radius - dist) / radius, 1 at the center -> 0
//     at `radius` (and exactly 0 beyond it); the twin-fidelity anchor mode.
//   * kFalloffInvSq  — near-clamped inverse square: factor = near^2 / max(dist, near)^2 (Q16.16 via
//     fxmul/fxdiv). The near clamp bounds the singularity (factor == kOne for dist <= nearClamp); at
//     dist == 2*near the factor is EXACTLY a quarter of the dist == near factor (integer-exact — the
//     test pins it). nearClamp must be > 0 (<= 0 -> factor 0, a deterministic degenerate).
//   WIND ignores falloff entirely (no center to measure from — the PT2 convention: uniform force).
//
// LOCKSTEP STATE ACCOUNTING (the WV1 zero-byte headline, refined): volume GEOMETRY/config (bounds,
// kind, center/axis, strength, falloff) is AUTHORED — it lives outside the snapshot, like the WV1 wave
// set. But FF1's command stream can TOGGLE a volume's `enabled` flag mid-run, so the enabled bits ARE
// state: the snapshot is the fpx body vector + ONE BYTE PER VOLUME (FfSnapshot). The test proves both
// directions: rollback restoring bodies+bits corrects a mispredicted toggle bit-exactly, and a control
// that restores bodies ONLY genuinely diverges (the bits are not redundant).
//
// INTEGER DOMAIN BOUNDS (documented): all math is the fpx Q16.16 toolbox (int64 inside fxmul/fxdiv/
// FxLength). Distances/extents must respect the fpx +-32768-wu product bound; kFalloffInvSq squares a
// distance (fxmul(d, d)), so field distances there must stay <= ~181 wu (fine for any authored volume).
//
// REUSE MAP (read-only #includes — fpx.h / particles.h are byte-UNTOUCHED):
//   * sim/fpx.h       — fx/FxVec3/FxBody/FxWorld/fxmul/fxdiv/FxLength/FxNormalize/BuildPairs/StepWorld/
//                       IntegrateOrientation + the FPX5 lockstep mold (FxCommand/ApplyCommand/
//                       SnapshotWorld/RestoreWorld) reused VERBATIM.
//   * sim/particles.h — the PT2 field vocabulary: FxDot/FxCross (the int64 convex.h-verbatim helpers)
//                       used DIRECTLY, ForceField/AccumulateForce as the twin-fidelity oracle.

#include <cstdint>
#include <vector>

#include "sim/fpx.h"        // read-only: the fixed-point substrate + rigid solver + FPX5 lockstep mold
#include "sim/particles.h"  // read-only: the PT2 field math (FxDot/FxCross + the AccumulateForce oracle)

namespace hf::sim {
namespace ff {

// Reuse the fpx Q16.16 toolbox verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::FxAdd;
using fpx::FxSub;
using fpx::FxScale;
using fpx::fxmul;
using fpx::fxdiv;
using fpx::FxLength;
using fpx::FxNormalize;
using particles::FxDot;    // the PT2 int64 dot   (convex.h-verbatim) — read-only reuse
using particles::FxCross;  // the PT2 int64 cross (convex.h-verbatim) — read-only reuse
inline constexpr int kFrac = fpx::kFrac;   // 16
inline constexpr fx  kOne  = fpx::kOne;    // 65536

// ----- The field kinds (the PT2 enum, renamed for the rigid-body context) ------------------------------
inline constexpr uint32_t kFieldRadial = 0u;  // attractor (strength>0) / repeller (strength<0) about `center`
inline constexpr uint32_t kFieldVortex = 1u;  // swirl about `axis` through `center`, tangential
inline constexpr uint32_t kFieldWind   = 2u;  // uniform force = axis*strength (no center, falloff ignored)

// ----- The falloff modes ---------------------------------------------------------------------------------
inline constexpr uint32_t kFalloffNone   = 0u;  // full strength inside the AABB
inline constexpr uint32_t kFalloffLinear = 1u;  // the PT2 (radius - dist)/radius falloff, 0 beyond radius
inline constexpr uint32_t kFalloffInvSq  = 2u;  // near-clamped inverse square: near^2 / max(dist,near)^2

// ----- FieldVolume: ONE authored bounded force-field volume ---------------------------------------------
// The PT2 ForceField grown a bound + falloff mode + toggle: `bounds` gates WHICH bodies feel the field
// (body CENTER inside, inclusive — the fpx AabbOverlap <= convention); the field math inside is the PT2
// vocabulary. `enabled` is the ONLY runtime-mutable member (toggle commands) — everything else is
// authored config outside the lockstep snapshot.
struct FieldVolume {
    fpx::FxAabb bounds{};              // the volume (lo <= hi, inclusive); the body-center gate
    uint32_t    kind = kFieldRadial;   // kFieldRadial / kFieldVortex / kFieldWind
    FxVec3      center{};              // Q16.16 field origin (radial/vortex; unused for wind)
    FxVec3      axis{};                // Q16.16 unit-ish: vortex spin axis OR wind direction (host-snapped)
    fx          strength = 0;          // Q16.16 signed magnitude (radial: + attract / - repel — the PT2 sign)
    fx          radius = 0;            // Q16.16 kFalloffLinear cutoff (the PT2 radius; unused otherwise)
    fx          nearClamp = kOne;      // Q16.16 kFalloffInvSq near clamp (must be > 0; unused otherwise)
    uint32_t    falloff = kFalloffNone;
    uint32_t    enabled = 1u;          // the toggle bit — the ONE stateful field (in FfSnapshot)
};

// InVolume: the body-center gate — inclusive compares on all three axes (the WV1 InFootprint sibling,
// extended to Y; the documented all-in-or-all-out caveat). Pure integer compares.
inline bool InVolume(const FieldVolume& v, const FxVec3& pos) {
    return pos.x >= v.bounds.lo.x && pos.x <= v.bounds.hi.x &&
           pos.y >= v.bounds.lo.y && pos.y <= v.bounds.hi.y &&
           pos.z >= v.bounds.lo.z && pos.z <= v.bounds.hi.z;
}

// FalloffFactor(v, dist) -> the Q16.16 scale in [0, kOne]:
//   kFalloffNone:   kOne.
//   kFalloffLinear: the PT2 form VERBATIM — fxdiv(radius - dist, radius) for dist < radius, else 0
//                   (radius <= 0 -> 0, deterministic degenerate).
//   kFalloffInvSq:  dEff = max(dist, nearClamp); fxdiv(near^2, dEff^2) — kOne on the clamp plateau
//                   (dist <= near), EXACTLY a quarter at each distance doubling (integer-exact).
inline fx FalloffFactor(const FieldVolume& v, fx dist) {
    if (v.falloff == kFalloffLinear) {
        if (v.radius <= 0 || dist >= v.radius) return 0;
        return fxdiv(v.radius - dist, v.radius);
    }
    if (v.falloff == kFalloffInvSq) {
        if (v.nearClamp <= 0) return 0;
        const fx dEff = dist < v.nearClamp ? v.nearClamp : dist;
        return fxdiv(fxmul(v.nearClamp, v.nearClamp), fxmul(dEff, dEff));
    }
    return kOne;   // kFalloffNone (and any unknown mode — deterministic full strength)
}

// ----- EvalFieldForce: the field force at a position (the PT2 AccumulateForce per-field body, TWINNED) --
// PURE field math — NO bounds/enabled gate (ApplyFields gates; keeping this pure lets the test pin the
// twin against particles::AccumulateForce directly). Per kind, the SAME ops in the SAME order as PT2:
//   kFieldRadial: d = center - pos; dist = FxLength(d); dist <= 0 -> 0 (dir undefined at the center);
//                 dir = FxNormalize(d); mag = fxmul(strength, FalloffFactor(dist)); F = dir * mag.
//                 (kFalloffLinear reproduces PT2 kFieldPoint BIT-EXACTLY: mag == 0 at/beyond radius.)
//   kFieldVortex: r = pos - center; rPerp = r - axis*FxDot(r,axis); dist = FxLength(rPerp); dist <= 0 ->
//                 0; tang = FxCross(axis, rPerp); |tang| <= 0 -> 0; dir = FxNormalize(tang);
//                 mag = fxmul(strength, FalloffFactor(dist)); F = dir * mag.
//   kFieldWind:   F = axis * strength (componentwise fxmul — the PT2 form; falloff IGNORED, documented).
inline FxVec3 EvalFieldForce(const FieldVolume& v, const FxVec3& pos) {
    FxVec3 force{0, 0, 0};
    if (v.kind == kFieldRadial) {
        const FxVec3 d = FxSub(v.center, pos);
        const fx dist = FxLength(d);
        if (dist <= 0) return force;
        const FxVec3 dir = FxNormalize(d);
        const fx mag = fxmul(v.strength, FalloffFactor(v, dist));
        force.x = fxmul(dir.x, mag);
        force.y = fxmul(dir.y, mag);
        force.z = fxmul(dir.z, mag);
    } else if (v.kind == kFieldVortex) {
        const FxVec3 r = FxSub(pos, v.center);
        const fx along = FxDot(r, v.axis);
        const FxVec3 rPerp = FxSub(r, FxScale(v.axis, along));
        const fx dist = FxLength(rPerp);
        if (dist <= 0) return force;
        const FxVec3 tang = FxCross(v.axis, rPerp);
        if (FxLength(tang) <= 0) return force;
        const FxVec3 dir = FxNormalize(tang);
        const fx mag = fxmul(v.strength, FalloffFactor(v, dist));
        force.x = fxmul(dir.x, mag);
        force.y = fxmul(dir.y, mag);
        force.z = fxmul(dir.z, mag);
    } else if (v.kind == kFieldWind) {
        force.x = fxmul(v.axis.x, v.strength);
        force.y = fxmul(v.axis.y, v.strength);
        force.z = fxmul(v.axis.z, v.strength);
    }
    return force;
}

// ----- ApplyFields: mutate body velocities PRE-STEP (the WV1 StepFloatBody discipline) ------------------
// Per body ASCENDING, per volume ASCENDING (the PT2 fixed-array-order sum — associative-order-pinned):
// skip non-dynamic bodies; skip disabled volumes; skip volumes whose AABB does not contain the body
// CENTER; accumulate F += EvalFieldForce; then dv = F * invMass * dt (the WV1 op order:
// FxScale(FxScale(F, invMass), dt)). The caller runs fpx::StepWorld as usual afterwards. LINEAR force at
// the center of mass ONLY — no field torque (the documented v1 gap). IDENTITY-AT-ZERO: with no volumes
// (or all disabled / all misses) F stays exactly (0,0,0) and vel += 0 is an EXACT no-op (fxmul(0,s)==0),
// so the composed tick equals plain fpx bit-for-bit — the test pins it.
inline void ApplyFields(const std::vector<FieldVolume>& volumes, fpx::FxWorld& w, fx dt) {
    const size_t n = w.bodies.size();
    for (size_t i = 0; i < n; ++i) {
        fpx::FxBody& b = w.bodies[i];
        if (!(b.flags & fpx::kFlagDynamic)) continue;   // static/kinematic -> untouched
        FxVec3 force{0, 0, 0};
        for (size_t f = 0; f < volumes.size(); ++f) {
            const FieldVolume& v = volumes[f];
            if (!v.enabled) continue;
            if (!InVolume(v, b.pos)) continue;
            const FxVec3 fv = EvalFieldForce(v, b.pos);
            force.x += fv.x;
            force.y += fv.y;
            force.z += fv.z;
        }
        b.vel = FxAdd(b.vel, FxScale(FxScale(force, b.invMass), dt));
    }
}

// ----- StepFieldWorld: the composed tick WITHOUT commands (fields -> re-broadphase -> StepWorld -> ------
// orientation; the fpx::SimTick tail with the field exchange prepended — the WV1 SimWaterTick shape).
inline void StepFieldWorld(const std::vector<FieldVolume>& volumes, fpx::FxWorld& w, fx dt, int iters) {
    ApplyFields(volumes, w, dt);
    std::vector<uint32_t> offsets;
    std::vector<fpx::FxPair> pairs;
    fpx::BuildPairs(w, offsets, pairs);
    fpx::StepWorld(w, std::span<const fpx::FxPair>(pairs), dt, iters);
    for (fpx::FxBody& b : w.bodies) fpx::IntegrateOrientation(b, dt);
}

// ----- The FF1 command stream: body inputs + volume TOGGLES ----------------------------------------------
// kFfCmdBody wraps an fpx::FxCommand VERBATIM (impulse / set-angvel — the FPX5 wire format); kFfCmdToggle
// sets volumes[volume].enabled = enable (out-of-range volume -> deterministic no-op, the ApplyCommand
// convention). Streams are processed in ARRAY ORDER per tick on every peer.
inline constexpr uint32_t kFfCmdBody   = 0u;
inline constexpr uint32_t kFfCmdToggle = 1u;

struct FfCommand {
    uint32_t       tick = 0;      // the tick this input applies on
    uint32_t       kind = kFfCmdBody;
    fpx::FxCommand body{};        // kFfCmdBody payload (its own .tick is ignored — FfCommand::tick rules)
    uint32_t       volume = 0;    // kFfCmdToggle payload: the volume index
    uint32_t       enable = 0;    // kFfCmdToggle payload: the target enabled state (0/1)
};

inline void ApplyFieldCommand(std::vector<FieldVolume>& volumes, fpx::FxWorld& w, const FfCommand& c) {
    if (c.kind == kFfCmdBody) {
        fpx::ApplyCommand(w, c.body);
    } else if (c.kind == kFfCmdToggle) {
        if (c.volume < (uint32_t)volumes.size()) volumes[c.volume].enabled = c.enable ? 1u : 0u;
    }
}

// SimFieldTick: the deterministic per-tick step (the FPX5 SimTick mold with the field exchange + toggle
// commands). (1) apply this tick's commands in ARRAY ORDER (toggles mutate `volumes`; body commands hit
// the world); (2) ApplyFields; (3) re-broadphase + StepWorld + IntegrateOrientation (VERBATIM the
// fpx::SimTick tail). With no volumes and body-only commands this equals fpx::SimTick BIT-FOR-BIT.
inline void SimFieldTick(std::vector<FieldVolume>& volumes, fpx::FxWorld& w,
                         const std::vector<FfCommand>& stream, uint32_t tick, fx dt, int iters) {
    for (const FfCommand& c : stream)
        if (c.tick == tick) ApplyFieldCommand(volumes, w, c);
    ApplyFields(volumes, w, dt);
    std::vector<uint32_t> offsets;
    std::vector<fpx::FxPair> pairs;
    fpx::BuildPairs(w, offsets, pairs);
    fpx::StepWorld(w, std::span<const fpx::FxPair>(pairs), dt, iters);
    for (fpx::FxBody& b : w.bodies) fpx::IntegrateOrientation(b, dt);
}

// StepFieldSteps: run K command-free ticks starting at startTick (the test/showcase driver).
inline void StepFieldSteps(std::vector<FieldVolume>& volumes, fpx::FxWorld& w, fx dt, int iters,
                           uint32_t startTick, int steps) {
    static const std::vector<FfCommand> kEmpty;
    for (int s = 0; s < steps; ++s)
        SimFieldTick(volumes, w, kEmpty, startTick + (uint32_t)s, dt, iters);
}

// ----- LOCKSTEP + ROLLBACK (the WV1/FPX5 mold; the snapshot = bodies + ONE BYTE PER VOLUME) -------------
// Volume geometry/config is authored (outside the snapshot, like the WV1 wave set); the toggled `enabled`
// bits ARE state because commands mutate them, so they ride in the snapshot (the test's restore-without-
// bits control proves they are NOT redundant).
struct FfSnapshot {
    fpx::FxWorld         world;     // fpx::SnapshotWorld — the body vector + scalars (deep copy)
    std::vector<uint8_t> enabled;   // one byte per volume — the ONLY field state
};

inline std::vector<uint8_t> EnabledBits(const std::vector<FieldVolume>& volumes) {
    std::vector<uint8_t> bits(volumes.size(), 0u);
    for (size_t i = 0; i < volumes.size(); ++i) bits[i] = volumes[i].enabled ? 1u : 0u;
    return bits;
}

inline FfSnapshot SnapshotFieldState(const std::vector<FieldVolume>& volumes, const fpx::FxWorld& w) {
    return FfSnapshot{fpx::SnapshotWorld(w), EnabledBits(volumes)};
}

inline void RestoreFieldState(std::vector<FieldVolume>& volumes, fpx::FxWorld& w, const FfSnapshot& s) {
    fpx::RestoreWorld(w, s.world);
    for (size_t i = 0; i < volumes.size() && i < s.enabled.size(); ++i)
        volumes[i].enabled = s.enabled[i] ? 1u : 0u;
}

// RunFieldLockstep: the peer entry point — run `ticks` SimFieldTicks from COPIES of the authored volumes
// + init world. Authority and replica fed the SAME init + stream (inputs ONLY) re-derive the fielded
// world BIT-IDENTICALLY (toggles included — both peers mutate their volume copies in the same order).
inline fpx::FxWorld RunFieldLockstep(const std::vector<FieldVolume>& volumesInit,
                                     const fpx::FxWorld& init, const std::vector<FfCommand>& stream,
                                     int ticks, fx dt, int iters) {
    std::vector<FieldVolume> volumes = volumesInit;
    fpx::FxWorld w = init;
    for (int t = 0; t < ticks; ++t)
        SimFieldTick(volumes, w, stream, (uint32_t)t, dt, iters);
    return w;
}

// RunFieldRollback: the FPX5 RunRollback harness over the field snapshot. (1) advance to mispredictTick
// with the authoritative stream; (2) SNAPSHOT (bodies + enabled bits); (2b) speculate <=3 ticks with the
// MISPREDICTED stream (a wrong toggle genuinely flips a volume); (3) restore BOTH bodies and bits +
// re-simulate the correct stream -> the corrected world == RunFieldLockstep(authStream) exactly.
inline fpx::FxWorld RunFieldRollback(const std::vector<FieldVolume>& volumesInit,
                                     const fpx::FxWorld& init,
                                     const std::vector<FfCommand>& authStream,
                                     const std::vector<FfCommand>& mispredictStream, int ticks,
                                     int mispredictTick, fx dt, int iters) {
    std::vector<FieldVolume> volumes = volumesInit;
    fpx::FxWorld w = init;
    for (int t = 0; t < mispredictTick; ++t)
        SimFieldTick(volumes, w, authStream, (uint32_t)t, dt, iters);
    const FfSnapshot snap = SnapshotFieldState(volumes, w);
    int specTicks = ticks - mispredictTick;
    if (specTicks > 3) specTicks = 3;
    for (int s = 0; s < specTicks; ++s)
        SimFieldTick(volumes, w, mispredictStream, (uint32_t)(mispredictTick + s), dt, iters);
    RestoreFieldState(volumes, w, snap);
    for (int t = mispredictTick; t < ticks; ++t)
        SimFieldTick(volumes, w, authStream, (uint32_t)t, dt, iters);
    return w;
}

// ----- FfDigest / FfWorldsEqual: the WV1 WaterDigest/WaterWorldsEqual twins (bodies only — volume ------
// config is authored; the enabled bits are compared via EnabledBits where a test needs them). FNV-1a-64,
// field-wise byte mixing (layout-independent).
inline uint64_t FfDigest(const fpx::FxWorld& w) {
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

inline bool FfWorldsEqual(const fpx::FxWorld& a, const fpx::FxWorld& b) {
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
// THE FIXED SHOWCASE SCENARIO + the pure-integer TOP-DOWN raster (shared VERBATIM by the Vulkan
// --ff1-fields-shot and the Metal --ff1-fields so the golden is bit-identical cross-backend BY
// CONSTRUCTION — the WV1/GAS1 shared-scenario-plus-shared-raster discipline). All host-snapped
// constants; NO RNG, NO clock. Three volumes, three stories in one frame: a RADIAL BURST blasts a ring
// of debris outward, a VORTEX swirls three bodies at different radii, a WIND LANE pushes three boxes
// across the frictionless floor. Body trails show the motion.
// ===================================================================================================

// Snap(v): round-to-nearest Q16.16 of a host double (ties away from zero — the WV1/cloth.h snap).
// Scene-build-time only; the per-tick sim never touches double.
inline constexpr fx Snap(double v) {
    const double s = v * (double)kOne;
    return (fx)(s + (s < 0 ? -0.5 : 0.5));
}

inline constexpr int      kShotSteps  = 240;   // 4 s at 60 Hz
inline constexpr uint32_t kShotBodies = 12;    // 6 burst + 3 vortex + 3 wind
inline constexpr int      kTrailEvery = 6;     // trail sample cadence (ticks)

// The FIXED showcase volumes: [0] radial burst (linear falloff), [1] vortex swirl (no falloff),
// [2] wind lane (uniform). World stage: XZ in [-14, 14], floor y = 0.
inline std::vector<FieldVolume> ShowcaseVolumes() {
    std::vector<FieldVolume> vols(3);
    // [0] RADIAL BURST: a repeller (strength < 0 — the PT2 sign) centered in the SW quadrant.
    vols[0].kind = kFieldRadial;
    vols[0].bounds.lo = FxVec3{Snap(-10.5), 0, Snap(-10.5)};
    vols[0].bounds.hi = FxVec3{Snap(-1.5), Snap(2.0), Snap(-1.5)};
    vols[0].center = FxVec3{Snap(-6.0), Snap(0.4), Snap(-6.0)};
    vols[0].strength = Snap(-2.2);
    vols[0].falloff = kFalloffLinear;
    vols[0].radius = Snap(4.0);
    // [1] VORTEX SWIRL: +Y axis through the SE quadrant, full strength inside the AABB.
    vols[1].kind = kFieldVortex;
    vols[1].bounds.lo = FxVec3{Snap(1.5), 0, Snap(-10.5)};
    vols[1].bounds.hi = FxVec3{Snap(10.5), Snap(2.0), Snap(-1.5)};
    vols[1].center = FxVec3{Snap(6.0), Snap(0.4), Snap(-6.0)};
    vols[1].axis = FxVec3{0, kOne, 0};
    vols[1].strength = Snap(1.4);
    vols[1].falloff = kFalloffNone;
    // [2] WIND LANE: +X wind across the north band, uniform (falloff ignored for wind).
    vols[2].kind = kFieldWind;
    vols[2].bounds.lo = FxVec3{Snap(-12.0), 0, Snap(4.0)};
    vols[2].bounds.hi = FxVec3{Snap(8.0), Snap(2.0), Snap(9.0)};
    vols[2].axis = FxVec3{kOne, 0, 0};
    vols[2].strength = Snap(1.6);
    return vols;
}

// The showcase world: 12 dynamic spheres (r = 0.4, invMass = 1) resting on the y=0 floor — a 6-ring
// around the burst center, 3 vortex riders at radii 1.0/1.8/2.6, 3 wind boxes queued in the lane.
inline fpx::FxWorld MakeFieldShotWorld() {
    fpx::FxWorld w;
    w.gravity = FxVec3{0, Snap(-9.8), 0};
    w.groundY = 0;
    const fx r = Snap(0.4);
    auto add = [&w, r](fx x, fx z) {
        fpx::FxBody b;
        b.pos = FxVec3{x, r, z};
        b.invMass = kOne;
        b.flags = fpx::kFlagDynamic;
        b.radius = r;
        w.bodies.push_back(b);
    };
    // [0..5] the burst ring (radius 1.5 around (-6,-6); 60-degree spokes, host-snapped).
    add(Snap(-6.0 + 1.5), Snap(-6.0));
    add(Snap(-6.0 + 0.75), Snap(-6.0 + 1.29903811));
    add(Snap(-6.0 - 0.75), Snap(-6.0 + 1.29903811));
    add(Snap(-6.0 - 1.5), Snap(-6.0));
    add(Snap(-6.0 - 0.75), Snap(-6.0 - 1.29903811));
    add(Snap(-6.0 + 0.75), Snap(-6.0 - 1.29903811));
    // [6..8] the vortex riders (radii 1.0 / 1.8 / 2.6 from (6,-6), staggered spokes).
    add(Snap(6.0 + 1.0), Snap(-6.0));
    add(Snap(6.0), Snap(-6.0 + 1.8));
    add(Snap(6.0 - 2.6), Snap(-6.0));
    // [9..11] the wind boxes queued at the west end of the lane.
    add(Snap(-10.0), Snap(6.5));
    add(Snap(-8.5), Snap(5.5));
    add(Snap(-7.0), Snap(7.5));
    return w;
}

struct FieldShotRun {
    fpx::FxWorld             finalWorld;
    std::vector<FieldVolume> volumesUsed;
    uint64_t                 digest = 0;       // FfDigest(finalWorld)
    uint64_t                 traceDigest = 0;  // FNV over every tick's per-body (pos.x, pos.z)
    std::vector<FxVec3>      trail;            // positions sampled every kTrailEvery ticks (12 per sample)
};

inline FieldShotRun RunFieldShotScenario() {
    FieldShotRun run;
    run.volumesUsed = ShowcaseVolumes();
    run.finalWorld = MakeFieldShotWorld();
    uint64_t th = 1469598103934665603ull;
    auto mix = [&th](uint32_t v) {
        for (int b = 0; b < 4; ++b) {
            th ^= (uint64_t)((v >> (b * 8)) & 0xFFu);
            th *= 1099511628211ull;
        }
    };
    static const std::vector<FfCommand> kEmpty;
    const fx dt = kOne / 60;
    for (int t = 0; t < kShotSteps; ++t) {
        SimFieldTick(run.volumesUsed, run.finalWorld, kEmpty, (uint32_t)t, dt, 4);
        for (uint32_t i = 0; i < kShotBodies; ++i) {
            mix((uint32_t)run.finalWorld.bodies[i].pos.x);
            mix((uint32_t)run.finalWorld.bodies[i].pos.z);
        }
        if ((t % kTrailEvery) == 0)
            for (uint32_t i = 0; i < kShotBodies; ++i) run.trail.push_back(run.finalWorld.bodies[i].pos);
    }
    run.traceDigest = th;
    run.digest = FfDigest(run.finalWorld);
    return run;
}

// RenderFieldShot: the PURE-INTEGER top-down raster (XZ plane, +x right / +z up) — the three volume
// outlines (burst orange / vortex cyan / wind green, with wind-direction dashes), field-center marks,
// per-body trail dots (dim) and final body discs (bright). World Q16.16 -> pixels by an integer
// 20-px/wu mapping; fixed color constants -> identical both backends. BGRA8, 560x560 (x,z in [-14,14]).
inline void RenderFieldShot(const FieldShotRun& run, std::vector<uint8_t>& bgra, uint32_t& outW,
                            uint32_t& outH) {
    const int W = 560, H = 560, kScale = 20;   // 20 px per world unit
    const fx xMin = Snap(-14.0), zMin = Snap(-14.0);
    outW = (uint32_t)W; outH = (uint32_t)H;
    bgra.assign((size_t)W * H * 4, 0);
    for (size_t p = 0; p < (size_t)W * H; ++p) {
        bgra[p * 4 + 0] = 16; bgra[p * 4 + 1] = 11; bgra[p * 4 + 2] = 8; bgra[p * 4 + 3] = 255;
    }
    auto putPx = [&](int ix, int iy, uint8_t r, uint8_t g, uint8_t b) {
        if (ix < 0 || ix >= W || iy < 0 || iy >= H) return;
        uint8_t* dst = &bgra[((size_t)iy * W + ix) * 4];
        dst[0] = b; dst[1] = g; dst[2] = r; dst[3] = 255;
    };
    auto mapX = [&](fx x) { return (int)(((int64_t)(x - xMin) * kScale) >> kFrac); };
    auto mapZ = [&](fx z) { return H - 1 - (int)(((int64_t)(z - zMin) * kScale) >> kFrac); };

    // (1) The volume outlines + center marks (+ the wind-direction dashes).
    const uint8_t volR[3] = {205, 60, 90}, volG[3] = {120, 165, 170}, volB[3] = {50, 205, 90};
    for (size_t v = 0; v < run.volumesUsed.size() && v < 3; ++v) {
        const FieldVolume& fv = run.volumesUsed[v];
        const int x0 = mapX(fv.bounds.lo.x), x1 = mapX(fv.bounds.hi.x);
        const int zTop = mapZ(fv.bounds.hi.z), zBot = mapZ(fv.bounds.lo.z);
        for (int px = x0; px <= x1; ++px) { putPx(px, zTop, volR[v], volG[v], volB[v]);
                                            putPx(px, zBot, volR[v], volG[v], volB[v]); }
        for (int py = zTop; py <= zBot; ++py) { putPx(x0, py, volR[v], volG[v], volB[v]);
                                                putPx(x1, py, volR[v], volG[v], volB[v]); }
        if (fv.kind != kFieldWind) {   // a 5-px cross at the field center
            const int cx = mapX(fv.center.x), cz = mapZ(fv.center.z);
            for (int o = -2; o <= 2; ++o) { putPx(cx + o, cz, volR[v], volG[v], volB[v]);
                                            putPx(cx, cz + o, volR[v], volG[v], volB[v]); }
        } else {                       // wind-direction dashes: 3 rows of +x dashes inside the lane
            for (int row = 0; row < 3; ++row) {
                const fx zRow = fv.bounds.lo.z +
                                (fx)(((int64_t)(fv.bounds.hi.z - fv.bounds.lo.z) * (row + 1)) / 4);
                const int py = mapZ(zRow);
                for (int px = x0 + 4; px + 8 <= x1; px += 16)
                    for (int o = 0; o < 8; ++o) putPx(px + o, py, 60, 105, 60);
            }
        }
    }

    // (2) The trails (dim per-group color): 1-px dots at every sampled position.
    const uint8_t bodR[3] = {235, 90, 150}, bodG[3] = {150, 205, 225}, bodB[3] = {70, 240, 130};
    auto groupOf = [](uint32_t i) { return i < 6 ? 0 : (i < 9 ? 1 : 2); };
    const uint32_t samples = (uint32_t)(run.trail.size() / kShotBodies);
    for (uint32_t s = 0; s < samples; ++s) {
        for (uint32_t i = 0; i < kShotBodies; ++i) {
            const FxVec3& p = run.trail[(size_t)s * kShotBodies + i];
            const int g = groupOf(i);
            putPx(mapX(p.x), mapZ(p.z), (uint8_t)(bodR[g] / 2), (uint8_t)(bodG[g] / 2),
                  (uint8_t)(bodB[g] / 2));
        }
    }

    // (3) The final bodies: filled integer discs (bright per-group color) + a dark center pip.
    for (uint32_t i = 0; i < kShotBodies && i < (uint32_t)run.finalWorld.bodies.size(); ++i) {
        const fpx::FxBody& b = run.finalWorld.bodies[i];
        const int cx = mapX(b.pos.x), cz = mapZ(b.pos.z);
        const int rPx = (int)(((int64_t)b.radius * kScale) >> kFrac);
        const int g = groupOf(i);
        for (int dz = -rPx; dz <= rPx; ++dz)
            for (int dx = -rPx; dx <= rPx; ++dx)
                if (dx * dx + dz * dz <= rPx * rPx) putPx(cx + dx, cz + dz, bodR[g], bodG[g], bodB[g]);
        putPx(cx, cz, 20, 20, 20);
    }
}

}  // namespace ff
}  // namespace hf::sim
