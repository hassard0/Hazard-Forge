#pragma once
// Slice BD1 — Deterministic GPU Crowds: THE INTEGER STEERING PRIMITIVE (the BEACHHEAD of FLAGSHIP #18:
// DETERMINISTIC GPU CROWDS — boids + steering + path-following, hf::sim::boids). A boid is a POINT with a
// Q16.16 position + velocity (the fpx::FxBody cousin, no orientation/mass). Each tick it computes a Reynolds
// STEERING force — SteerSeek (proportional pull toward a shared target) + SteerSeparation (brute-force
// all-pairs push away from too-close neighbors) — accumulated into the per-agent velocity, then integrates.
// Pure CPU, header-only, NO device, NO backend symbols, NO <cmath> on the bit-exact path. The STRUCTURAL
// TWIN of the FPX1/GR1 integer beachhead: a pure-integer per-agent update proven GPU==CPU BIT-EXACT, with a
// cross-backend BIT-IDENTICAL integer golden.
//
// THE int32 GOAL & THE int64 REALITY (the honest proof-strength call): BD1 keeps the SEPARATION SEARCH +
// the per-axis CLAMPS pure-int32 (no FxNormalize/FxISqrt on the per-agent path — the spec's "accumulate raw
// integer deltas" rule + an axis-box clamp, NOT a radial magnitude clamp). BUT the steer/integrate path
// MUST scale by Q16.16 gains + dt: `force = FxScale(desired, seekGain)`, `vel += force*dt`, `pos += vel*dt`
// — each an fxmul ((int64)a*b >> 16). fxmul uses an int64 INTERMEDIATE (Q16.16 world-scale products overflow
// int32, the FPX1/GR1/FL1 lesson). DXC compiles int64 (the Vulkan path); glslc (the Metal HLSL->SPIR-V->MSL
// frontend) CANNOT parse int64_t in HLSL. So shaders/boids_steer.comp.hlsl is VULKAN-SPIR-V-ONLY (in the
// Vulkan compile list, NOT in the Metal hf_gen_msl list); on Metal the --boids-steer showcase runs the CPU
// StepBoids — byte-identical to the Vulkan GPU result BY CONSTRUCTION (the fpx_integrate.comp / grain_
// integrate.comp convention), while the Vulkan side carries the GPU==CPU memcmp proof. The math here is
// copied VERBATIM by boids_steer.comp so the GPU exercises the EXACT integer ops.
//
// THE CROSS-BACKEND CRUX (the make-or-break for GPU==CPU): the steering is JACOBI — every agent computes its
// force from a FROZEN snapshot of ALL agents' positions, THEN integrates. So the per-agent update is
// INDEPENDENT of update order: the GPU one-thread-per-agent write (reading the previous step's positions
// from a SEPARATE input buffer, writing the next step's into an output buffer — ping-pong) is trivially
// race-free / order-independent, and two runs are byte-identical. The brute-force separation loop iterates
// ALL agents in the SAME fixed index order (skip self by index) on both paths -> bit-exact.
//
// REUSE MAP (engine/sim/fpx.h, read-only): fx/kFrac/kOne, fxmul, FxVec3/FxAdd/FxSub/FxScale. A small
// ClampAxis is re-implemented here (do NOT modify fpx.h / vehicle.h). The integrate shape mirrors
// fpx.h::IntegrateBody (vel += a*dt; pos += vel*dt) without gravity-only forcing.
//
// OUT OF SCOPE (later BD slices): the grid-hash neighbor list (BD2 — BD1 is brute-force all-pairs),
// alignment + cohesion (BD3), path-following (BD4), lockstep/rollback (BD5), the lit 3D render (BD6 — BD1's
// render is the 2D top-down diagnostic). HONEST CAVEATS (the GR4-shape): boids are POINTS with steering
// FORCES (a soft separation push, NOT a hard non-penetration contact — agents can briefly overlap); the
// per-axis clamp is an axis-BOX, not a radial magnitude clamp (deterministic + integer-cheap). Determinism +
// cross-platform bit-identity is the headline.

#include <cstdint>
#include <span>
#include <vector>

#include "sim/fpx.h"   // read-only: fx/kFrac/kOne/fxmul + FxVec3/FxAdd/FxSub/FxScale

namespace hf::sim {
namespace boids {

// Re-export the fpx Q16.16 vocabulary (boids share the fixed-point format exactly).
using fx = fpx::fx;
using FxVec3 = fpx::FxVec3;
inline constexpr int kFrac = fpx::kFrac;
inline constexpr fx  kOne  = fpx::kOne;
using fpx::fxmul;   // (a*b) >> kFrac with an int64 intermediate (the steer/integrate path's scalar mul)

// ----- A boid: a point with a Q16.16 position + velocity (the fpx::FxBody cousin, no orient/mass) ---------
// std430-packable as 6 x int32 (pos.xyz, vel.xyz) — the GPU Agent mirror.
struct Agent {
    FxVec3 pos;   // Q16.16 world position
    FxVec3 vel;   // Q16.16 velocity (world units / second)
};

// ----- The host-fixed Q16.16 steering tuning (golden-stable; NO RNG) -------------------------------------
struct BoidsConfig {
    fx     seekGain  = 0;   // Q16.16 gain on the (target - pos) seek force (proportional / un-normalized)
    fx     sepGain   = 0;   // Q16.16 gain on the accumulated separation push
    fx     sepRadius = 0;   // Q16.16 neighbor radius — only agents within this push each other apart
    fx     maxForce  = 0;   // Q16.16 per-AXIS force clamp (the axis-box magnitude limit, not radial)
    fx     maxSpeed  = 0;   // Q16.16 per-AXIS speed clamp
    FxVec3 target;          // Q16.16 the shared seek target
    FxVec3 gravity;         // Q16.16 constant acceleration (default 0 — BD1 is planar steering)
};

// ----- ClampAxis: the deterministic scalar clamp to [-limit, +limit] (limit >= 0). int32, no sqrt -------
// The per-component magnitude cap (the axis-box clamp the spec locks — deterministic + integer-cheap; a
// radial |v| clamp would need the int64 FxISqrt). The shader copies THIS body VERBATIM.
inline fx ClampAxis(fx v, fx limit) {
    if (v >  limit) return  limit;
    if (v < -limit) return -limit;
    return v;
}

// ClampAxisVec(v, limit): per-axis ClampAxis on each component (a box clamp to the [-limit,limit]^3 cube).
inline FxVec3 ClampAxisVec(const FxVec3& v, fx limit) {
    return FxVec3{ClampAxis(v.x, limit), ClampAxis(v.y, limit), ClampAxis(v.z, limit)};
}

// ----- SteerSeek(a, target, cfg): the proportional (un-normalized) seek force ----------------------------
// desired = target - pos; force = FxScale(desired, seekGain). NO FxNormalize/maxSpeed*unit (which would need
// the int64 FxISqrt) — un-normalized seek is the standard "arrive" behaviour (pulls harder when far, eases
// in when close) and is integer-cheap. The fxmul inside FxScale uses an int64 intermediate (the integrate
// path's int64 already pins the shader Vulkan-only). The shader copies THIS body VERBATIM.
inline FxVec3 SteerSeek(const Agent& a, const FxVec3& target, const BoidsConfig& cfg) {
    const FxVec3 desired = FxSub(target, a.pos);
    return FxScale(desired, cfg.seekGain);
}

// ----- SteerSeparation(a, others, cfg): the brute-force all-pairs separation push ------------------------
// For each OTHER agent o (skip self by index) with squared-distance < sepRadius², accumulate (a.pos - o.pos)
// — the raw away-direction — into a running integer sum (NO per-pair normalize, the scout's "accumulate raw
// integer deltas" rule). Scaled by sepGain. The squared-distance d² = dx²+dy²+dz² and sepRadius² are formed
// in int64 (a dx of a few Q16.16 world units squared exceeds int32), but they are only a COMPARE (no result
// stored) — and since the shader's integrate path is already int64 (Vulkan-only), this int64 compare costs
// nothing in proof-strength. selfIndex < 0 => no self to skip (compare against an external snapshot).
inline FxVec3 SteerSeparation(const Agent& a, std::span<const Agent> others, const BoidsConfig& cfg,
                              int selfIndex) {
    const int64_t r2 = (int64_t)cfg.sepRadius * (int64_t)cfg.sepRadius;
    FxVec3 sum{0, 0, 0};
    const int n = (int)others.size();
    for (int j = 0; j < n; ++j) {
        if (j == selfIndex) continue;                       // skip self by INDEX (fixed order, bit-exact)
        const Agent& o = others[(size_t)j];
        const fx dx = a.pos.x - o.pos.x;
        const fx dy = a.pos.y - o.pos.y;
        const fx dz = a.pos.z - o.pos.z;
        const int64_t d2 = (int64_t)dx * (int64_t)dx + (int64_t)dy * (int64_t)dy + (int64_t)dz * (int64_t)dz;
        if (d2 < r2) {                                      // within the separation radius -> push apart
            sum.x += dx;                                    // raw integer adds (no per-pair normalize)
            sum.y += dy;
            sum.z += dz;
        }
    }
    return FxScale(sum, cfg.sepGain);
}

// ----- StepBoids(agents, cfg, dt): one deterministic Reynolds steering tick (JACOBI) ---------------------
// (1) snapshot the current positions/velocities (the frozen input every agent's force reads — order
//     independence + GPU race-freedom); (2) per agent in FIXED index order: force = seek + separation +
//     gravity; clamp force per-axis to ±maxForce; vel += force*dt; clamp vel per-axis to ±maxSpeed; pos +=
//     vel*dt. Fixed op order + the frozen snapshot -> two runs bit-identical AND bit-exact GPU==CPU (the GPU
//     reads the previous step's agent buffer + writes a fresh one — ping-pong, the same Jacobi semantics).
inline void StepBoids(std::vector<Agent>& agents, const BoidsConfig& cfg, fx dt) {
    const std::vector<Agent> prev = agents;                 // frozen snapshot (the Jacobi input)
    const std::span<const Agent> snap(prev);
    const int n = (int)agents.size();
    for (int i = 0; i < n; ++i) {
        const Agent& a = prev[(size_t)i];                   // read the FROZEN state
        // (a) the steering force: seek + separation + gravity.
        FxVec3 force = SteerSeek(a, cfg.target, cfg);
        const FxVec3 sep = SteerSeparation(a, snap, cfg, i);
        force = FxAdd(force, sep);
        force = FxAdd(force, cfg.gravity);
        // (b) per-axis clamp the force (the axis-box magnitude limit).
        force = ClampAxisVec(force, cfg.maxForce);
        // (c) integrate velocity: vel += force * dt; then per-axis clamp to ±maxSpeed.
        FxVec3 vel = FxAdd(a.vel, FxScale(force, dt));
        vel = ClampAxisVec(vel, cfg.maxSpeed);
        // (d) integrate position: pos += vel * dt.
        const FxVec3 pos = FxAdd(a.pos, FxScale(vel, dt));
        agents[(size_t)i].vel = vel;
        agents[(size_t)i].pos = pos;
    }
}

// StepBoidsSteps(agents, cfg, dt, steps): run `steps` StepBoids ticks (the showcase settle loop).
inline void StepBoidsSteps(std::vector<Agent>& agents, const BoidsConfig& cfg, fx dt, int steps) {
    for (int s = 0; s < steps; ++s) StepBoids(agents, cfg, dt);
}

// ----- MeasureBoids: the deterministic flock statistics (Q16.16) -----------------------------------------
// meanSpeed = mean per-axis-summed |vel| component magnitude (an integer L1 proxy — NO sqrt, deterministic);
// meanToTarget = mean L1 distance |pos - target| (the "they sought it" stat — drops as the flock converges);
// minSep = the minimum L1 pairwise separation (the "they didn't collapse" stat — stays above a floor when
// separation is on, goes to ~0 when sepGain=0). All integer L1 metrics (|dx|+|dy|+|dz|) so NO sqrt / NO
// int64-result — bit-exact + cross-platform identical.
struct BoidsStats {
    fx meanSpeed    = 0;   // mean L1 speed
    fx meanToTarget = 0;   // mean L1 distance to the target
    fx minSep       = 0;   // minimum L1 pairwise separation
};

inline fx FxAbs(fx v) { return v < 0 ? -v : v; }
inline fx L1(const FxVec3& v) { return FxAbs(v.x) + FxAbs(v.y) + FxAbs(v.z); }

inline BoidsStats MeasureBoids(const std::vector<Agent>& agents, const BoidsConfig& cfg) {
    BoidsStats s;
    const int n = (int)agents.size();
    if (n == 0) return s;
    int64_t speedSum = 0, targSum = 0;
    for (int i = 0; i < n; ++i) {
        speedSum += L1(agents[(size_t)i].vel);
        targSum  += L1(FxSub(agents[(size_t)i].pos, cfg.target));
    }
    s.meanSpeed    = (fx)(speedSum / n);
    s.meanToTarget = (fx)(targSum / n);
    // min L1 pairwise separation over j>i (a single canonical orientation; deterministic).
    int64_t minSep = -1;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const int64_t d = L1(FxSub(agents[(size_t)i].pos, agents[(size_t)j].pos));
            if (minSep < 0 || d < minSep) minSep = d;
        }
    s.minSep = (fx)(minSep < 0 ? 0 : minSep);
    return s;
}

}  // namespace boids
}  // namespace hf::sim
