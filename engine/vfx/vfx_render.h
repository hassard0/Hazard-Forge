#pragma once
// Slice VR1 — VFX RENDERER VARIETY (parity++ audit #6): ribbon TRAILS + jittered BEAMS + MESH emitters +
// PARTICLE LIGHTS over the deterministic GPU particle sim (engine/sim/particles.h PT1-PT6) + the PA1 flow-VM
// authoring layer (engine/sim/particle_author.h). The sim + authoring ship bit-exact, but the RENDERER was
// sphere-instanced sprites only — no trails, no beams, no velocity-oriented meshes, no light-emitting
// particles (the Niagara renderer-variety axis). VR1 builds the deterministic GEOMETRY GENERATORS for all
// four: every generator is a PURE Q16.16/integer function of (pool state, params, tick) — digestable,
// two-run byte-identical, cross-compiler identical — and the float lit draw stays the established
// presentation class (the FPX6/PT6 convention: the float bridge READS the bit-exact integer layer, never
// feeds back). PURE CPU, header-only, NO device, NO backend symbols, NO new shader, NO new RHI.
// Namespace hf::vfx::render.
//
// THE FOUR GENERATORS (the deterministic half — all pure integer):
//   1. RIBBON TRAILS: TrailHistory (a per-slot ring buffer of the last K positions — a VR1-LOCAL structure;
//      sim/particles.h ParticlePool is byte-UNTOUCHED) + UpdateTrails (deterministic ascending-slot fold,
//      one push per alive slot per tick, seed-guarded against free-list slot reuse) + BuildRibbons (a
//      degenerate-index-free strip per trail, the SP1 SweepStrip winding discipline, velocity-ALIGNED
//      cross-up side vectors — the deterministic v1 choice; see the KINK caveat below — with a monotonic
//      age taper: the oldest sample is the thin tail, the newest the full-width head).
//   2. BEAMS: BuildBeam(a, b, widthQ, segments, jitterAmpQ, seed, tick) — a strip along the segment-lerped
//      centerline, each INTERIOR point offset by two pcg::PcgHash-derived Q16.16 jitters (side + up) in
//      [-amp, +amp] (the lightning look). Deterministic per (endpoints, seed, tick); jitterAmpQ==0 is the
//      EXACT straight strip (the identity control — the jitter term is hash*0, identically zero).
//      HONESTY: beams are VISUAL-ONLY geometry — there is NO gameplay hit/damage query attached; a hitscan
//      would be a separate deterministic raycast, not this strip.
//   3. MESH EMITTERS: BuildMeshInstances(pool) — one {pos, orient} integer instance per ALIVE particle,
//      orientation = the SP1 integer half-angle YAW (spline::YawFromTangent — +Z onto the velocity's XZ
//      direction) COMPOSED with an integer half-angle PITCH about local X (the same sqrt((1±cos)/2)
//      identity over the vertical component) -> a unit-to-±LSB FxQuat with NO trig/LUT/float. The float
//      bridge MeshInstanceTransform feeds the EXISTING instanced-lit pipeline (the FPX6 FxBodyTransform
//      shape) with ANY mesh (the showcase marks cubes).
//   4. PARTICLE LIGHTS: BuildParticleLights(pool, maxN) — the brightest min(maxN, alive) particles
//      (integer brightness = remaining life = lifetime - age; DETERMINISTIC selection: brightest-then-
//      lowest-slot tie-break, a total order -> pinned) become light entries. The float bridge
//      ParticleLightsToClustered turns them into render::clustered::Light entries for the EXISTING ML1
//      clustered many-light assignment (render/manylight.h + render/clustered.h, both byte-UNTOUCHED) —
//      the composition seam the test runs clustered::BuildClusters over and pins the assignDigest.
//
// THE KINK CAVEAT (honesty, pinned by the test's worst case): velocity-aligned ribbons compute the side
// vector as SideDir(segment direction) = normalize(tangent x up). At a DIRECTION REVERSAL (a trail that
// doubles back, e.g. +X then -X) the side vector FLIPS SIGN between adjacent samples, so the strip
// self-crosses (a visible kink/twist). Camera-facing ribbons would hide this but need the camera in the
// deterministic layer; v1 documents it instead (tests/vfx_render_test.cpp pins the exact worst-case
// digest). A VERTICAL segment (zero XZ tangent) hits FxNormalize's (0,kOne,0) fallback -> side == up —
// deterministic, never degenerate INDICES (SweepStrip's discipline: no repeated index in any triple, by
// construction), though a stationary particle can produce zero-AREA triangles (documented, not degenerate
// indices).
//
// TRAIL HISTORY vs THE PT5 SNAPSHOT (the netcode note): TrailHistory is PRESENTATION state — a pure fold
// over the pool's past K ticks. It is deliberately NOT part of particles.h::ParticleSnapshot (the frozen
// PT5 shape is byte-untouched; lockstep/rollback correctness needs pool+freeList+spawnCursor+tick+cfg
// ONLY). Two consequences: (1) a peer that replays ticks and calls UpdateTrails each tick re-derives the
// IDENTICAL TrailHistory (it is a deterministic fold — DigestTrails pins this); (2) a ROLLBACK that
// restores a pool snapshot and re-simulates WITHOUT re-recording trails shows up to K ticks of stale
// ribbon (visual-only divergence, self-healing in K ticks). A netcode layer that wants rollback-exact
// ribbons snapshots TrailHistory alongside (all four arrays are POD vectors — trivially copyable /
// memcmp-able); VR1 documents the choice rather than widening the frozen PT5 struct.
//
// REUSE MAP: sim/particles.h (ParticlePool/FxParticle/CountAlive + the Q16.16 aliases — READ-ONLY),
// sim/particle_author.h (the PA1 pulsing-fountain asset the showcase drives — READ-ONLY),
// spline/spline.h (YawFromTangent/SideDir/FxCrossQ — the SP1 integer half-angle + strip vocabulary,
// READ-ONLY), sim/fpx.h (FxQuat/FxQuatMul/FxQuatNormalize/FxRotate/FxISqrt — READ-ONLY),
// pcg/pcg.h (PcgHash — the beam jitter avalanche), render/clustered.h + render/manylight.h (Light/
// BuildClusters/DigestAssignment/ComputeAssignStats — the ML1 seam, READ-ONLY), net/session.h
// (DigestBytes — the digest currency), math/math.h (the float bridge ONLY).
//
// THE SHOWCASE (the LA1/GAS1 shared-scenario convention): RunVfxShotScenario + RenderVfxShot live HERE so
// the Vulkan --vr1-vfx-shot and the Metal --vr1-vfx run the IDENTICAL bytes -> the pure-INTEGER raster is
// strict-zero cross-backend BY CONSTRUCTION. The ONE float in the scenario is the ML1 assignment
// (clustered.h uses std::pow/std::log slice math) — its assignDigest is printed on the stat line and
// pinned PER-TOOLCHAIN-FAMILY (the documented ML1 honesty convention: MSVC == clang-on-Windows verified;
// the Mac prints its own value, so a last-ULP libm divergence surfaces explicitly). The assignDigest is
// deliberately NOT folded into the image or the combined integer digest, so the strict-zero golden stays
// pure integer. (A lit HDR hero composing all four generators into the real render pipeline is the
// natural VR2 — noted, not built here.)

#include <algorithm>   // std::sort (the deterministic brightest-N total-order selection)
#include <cstdint>
#include <vector>

#include "sim/particles.h"        // read-only: the bit-exact PT1-PT6 particle system
#include "sim/particle_author.h"  // read-only: the PA1 authored-effect layer (the showcase driver)
#include "spline/spline.h"        // read-only: YawFromTangent / SideDir / FxCrossQ (the SP1 vocabulary)
#include "pcg/pcg.h"              // read-only: PcgHash (the beam-jitter avalanche)
#include "render/clustered.h"     // read-only: Light / Grid / BuildClusters (the ML1 seam)
#include "render/manylight.h"     // read-only: DigestAssignment / ComputeAssignStats / FnvAppend
#include "net/session.h"          // read-only: DigestBytes (the digest currency)
#include "math/math.h"            // float bridge only: Mat4 / Quat / FromTRS / MulPointDivide

namespace hf::vfx {
namespace render {

// Reuse the Q16.16 vocabulary verbatim (NO new fixed-point primitives).
using sim::particles::fx;
using sim::particles::FxVec3;
using sim::particles::fxmul;
using sim::particles::FxAdd;
using sim::particles::FxSub;
using sim::particles::FxScale;
using sim::particles::FxLength;
using sim::particles::FxNormalize;
using sim::particles::ParticlePool;
using sim::particles::FxParticle;
using sim::particles::kFlagAlive;
using sim::fpx::FxQuat;
using sim::fpx::FxQuatMul;
using sim::fpx::FxQuatNormalize;
using sim::fpx::FxRotate;
using sim::fpx::FxISqrt;
inline constexpr int kFrac = sim::particles::kFrac;
inline constexpr fx  kOne  = sim::particles::kOne;

namespace mlns = hf::render::manylight;   // qualified aliases (hf::vfx::render shadows hf::render)
namespace clns = hf::render::clustered;

// ===== The digest currency (the pauthor::DigestAuthored hand-LE discipline: NEVER hash a padded host
// struct — every int32 folded little-endian field by field, byte-stable MSVC == clang == Mac) ==========
inline void PutU32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back((uint8_t)( v        & 0xFFu));
    buf.push_back((uint8_t)((v >> 8)  & 0xFFu));
    buf.push_back((uint8_t)((v >> 16) & 0xFFu));
    buf.push_back((uint8_t)((v >> 24) & 0xFFu));
}
inline void PutVec3(std::vector<uint8_t>& buf, const FxVec3& v) {
    PutU32(buf, (uint32_t)v.x); PutU32(buf, (uint32_t)v.y); PutU32(buf, (uint32_t)v.z);
}

// ===== 1. RIBBON TRAILS ===================================================================================

// TrailHistory: the per-slot ring buffer of the last K positions — a VR1-LOCAL tracked structure (the
// frozen ParticlePool layout is byte-untouched; see the header note on the PT5 snapshot). Per pool slot i:
//   ring [i*K .. i*K+K-1] : the recorded Q16.16 positions (a circular buffer)
//   head [i]              : the NEXT write index (0..K-1)
//   count[i]              : valid entries (0..K); 0 == no trail
//   seed [i]              : the FxParticle::seed the history belongs to — the SLOT-REUSE GUARD: the
//                           free-list recycles slots, and without this a new particle would inherit the
//                           dead one's trail (a teleport streak). seed mismatch -> reset before push.
struct TrailHistory {
    uint32_t              K = 0;      // ring capacity (past positions per tracked slot)
    std::vector<FxVec3>   ring;       // capacity * K entries
    std::vector<uint32_t> head;       // per-slot next write index
    std::vector<uint32_t> count;      // per-slot valid entry count (0..K)
    std::vector<uint32_t> seed;       // per-slot owning particle seed (0 == empty)
};

inline TrailHistory InitTrailHistory(uint32_t capacity, uint32_t K) {
    TrailHistory th;
    th.K = K;
    th.ring.assign((size_t)capacity * K, FxVec3{});
    th.head.assign((size_t)capacity, 0u);
    th.count.assign((size_t)capacity, 0u);
    th.seed.assign((size_t)capacity, 0u);
    return th;
}

// UpdateTrails: ONE deterministic history tick — ascending slot order (the determinism contract; a pure
// single-thread host fold, the Emit/RecycleDead sibling). Per slot:
//   * ALIVE: if the recorded seed differs (a recycled slot now owns a NEW particle) RESET the ring first;
//     then push the particle's current pos (head advances mod K, count saturates at K).
//   * NOT alive (dead or empty): RESET — the dead particle's trail EXPIRES immediately (the pinned
//     behavior: the tick after death the slot draws no ribbon).
// Call AFTER the sim tick (StepParticles/StepAuthoredEffect) so the newest ring entry is this tick's pos.
inline void UpdateTrails(const ParticlePool& pool, TrailHistory& th) {
    const uint32_t n = (uint32_t)pool.particles.size();
    for (uint32_t i = 0; i < n && i < (uint32_t)th.head.size(); ++i) {
        const FxParticle& p = pool.particles[(size_t)i];
        if (!(p.flags & kFlagAlive)) {                       // dead/empty -> the trail expires NOW
            th.head[i] = 0u; th.count[i] = 0u; th.seed[i] = 0u;
            continue;
        }
        if (th.seed[i] != p.seed) {                          // slot reuse -> reset before recording
            th.head[i] = 0u; th.count[i] = 0u; th.seed[i] = p.seed;
        }
        th.ring[(size_t)i * th.K + th.head[i]] = p.pos;      // push this tick's position
        th.head[i] = (th.head[i] + 1u) % th.K;
        if (th.count[i] < th.K) ++th.count[i];
    }
}

// DigestTrails: the hand-LE digest of the ENTIRE history state (K, then per slot head/count/seed + the
// count valid ring entries oldest->newest). Pins the deterministic-fold property (two identical runs ->
// identical digest; a replayed peer re-derives it — the netcode note above).
inline uint64_t DigestTrails(const TrailHistory& th) {
    std::vector<uint8_t> buf;
    buf.reserve(th.ring.size() * 12u + th.head.size() * 12u + 8u);
    PutU32(buf, th.K);
    PutU32(buf, (uint32_t)th.head.size());
    for (size_t i = 0; i < th.head.size(); ++i) {
        PutU32(buf, th.head[i]); PutU32(buf, th.count[i]); PutU32(buf, th.seed[i]);
        for (uint32_t j = 0; j < th.count[i]; ++j) {         // oldest -> newest (the BuildRibbons order)
            const uint32_t idx = (th.head[i] + th.K - th.count[i] + j) % th.K;
            PutVec3(buf, th.ring[i * (size_t)th.K + idx]);
        }
    }
    return hf::net::DigestBytes(buf.data(), buf.size());
}

// StripGeom: the shared integer strip currency (ribbons AND beams) — Q16.16 positions, 2 per sample
// (left, right), and the SP1 SweepStrip PINNED winding: span i emits (a,b,c)(c,b,d) with a=2i b=2i+1
// c=2(i+1) d=2(i+1)+1 — NO repeated index inside any triple, by construction (degenerate-INDEX-free).
struct StripGeom {
    std::vector<FxVec3>   positions;   // 2 per sample: [2k] = left, [2k+1] = right
    std::vector<uint32_t> indices;     // 6 per span (the pinned SweepStrip winding)
};

inline uint64_t DigestStrip(const StripGeom& s) {
    std::vector<uint8_t> buf;
    buf.reserve(s.positions.size() * 12u + s.indices.size() * 4u + 8u);
    PutU32(buf, (uint32_t)s.positions.size());
    for (const FxVec3& p : s.positions) PutVec3(buf, p);
    PutU32(buf, (uint32_t)s.indices.size());
    for (const uint32_t i : s.indices) PutU32(buf, i);
    return hf::net::DigestBytes(buf.data(), buf.size());
}

// RibbonSet: every drawable trail's strip in ONE geometry (a strip per trail, index-disjoint), plus the
// per-trail base offsets so a renderer/test can walk trails individually.
struct RibbonSet {
    StripGeom             geom;
    uint32_t              trailCount = 0;     // trails with >= 2 recorded positions (drawable)
    std::vector<uint32_t> trailFirstVert;     // per trail: first vertex index into geom.positions
    std::vector<uint32_t> trailSampleCount;   // per trail: history sample count (verts = 2 * this)
};

// BuildRibbons: the ribbon GEOMETRY generator — a PURE function of (history, widthQ). Per slot (ascending)
// with count >= 2, walk the ring OLDEST -> NEWEST and emit a two-rail strip:
//   dir_j  = p[j+1] - p[j] (the last sample reuses the previous segment's direction — velocity-aligned),
//   side_j = spline::SideDir(FxNormalize(dir_j))   (tangent x up, renormalized — the SP1 discipline;
//            vertical dir -> the (0,kOne,0) fallback: side == up, documented),
//   halfW_j = widthQ * (j+1) / (2*n)               (the AGE TAPER: oldest thinnest, newest full width —
//            integer division, MONOTONIC non-decreasing in j by construction),
//   left/right = p[j] -/+ side_j * halfW_j; indices per the pinned SweepStrip winding.
// Pure integer; NO degenerate index triples (winding); the KINK caveat at direction reversals is
// documented in the header comment + pinned by the test's worst case.
inline RibbonSet BuildRibbons(const TrailHistory& th, fx widthQ) {
    RibbonSet out;
    const uint32_t slots = (uint32_t)th.head.size();
    for (uint32_t i = 0; i < slots; ++i) {
        const uint32_t n = th.count[i];
        if (n < 2u || widthQ <= 0) continue;
        const uint32_t baseVert = (uint32_t)out.geom.positions.size();
        // Gather the polyline oldest -> newest (the ring unwind).
        FxVec3 pts[64];   // th.K is a small showcase constant; guard anyway
        const uint32_t m = n <= 64u ? n : 64u;
        for (uint32_t j = 0; j < m; ++j) {
            const uint32_t idx = (th.head[i] + th.K - n + j) % th.K;
            pts[j] = th.ring[(size_t)i * th.K + idx];
        }
        for (uint32_t j = 0; j < m; ++j) {
            const FxVec3 dir = (j + 1u < m) ? FxSub(pts[j + 1u], pts[j])
                                            : FxSub(pts[j], pts[j - 1u]);   // m>=2 -> j-1 safe
            const FxVec3 side = spline::SideDir(FxNormalize(dir));
            const fx halfW = (fx)(((int64_t)widthQ * (int64_t)(j + 1u)) / (int64_t)(2u * m));
            out.geom.positions.push_back(FxSub(pts[j], FxScale(side, halfW)));   // left
            out.geom.positions.push_back(FxAdd(pts[j], FxScale(side, halfW)));   // right
        }
        for (uint32_t j = 0; j + 1u < m; ++j) {   // the pinned SweepStrip winding per span
            const uint32_t a = baseVert + 2u * j;
            const uint32_t b = a + 1u;
            const uint32_t c = baseVert + 2u * (j + 1u);
            const uint32_t d = c + 1u;
            out.geom.indices.push_back(a); out.geom.indices.push_back(b); out.geom.indices.push_back(c);
            out.geom.indices.push_back(c); out.geom.indices.push_back(b); out.geom.indices.push_back(d);
        }
        out.trailFirstVert.push_back(baseVert);
        out.trailSampleCount.push_back(m);
        ++out.trailCount;
    }
    return out;
}

// ===== 2. BEAMS ===========================================================================================

// JitterFromHash: map a 32-bit avalanche to a SIGNED Q16.16 offset in [-amp, +amp) — the low 16 hash bits
// centered ([-32768, 32767]) scaled by amp/32768 (ONE int64 multiply + arithmetic shift). amp==0 -> 0
// EXACTLY for every hash (the identity control's algebraic guarantee).
inline fx JitterFromHash(uint32_t h, fx amp) {
    const int32_t centered = (int32_t)(h & 0xFFFFu) - 32768;
    return (fx)(((int64_t)centered * (int64_t)amp) >> 15);
}

// BuildBeam: the beam GEOMETRY generator — a PURE function of (a, b, widthQ, segments, jitterAmpQ, seed,
// tick). Centerline: P_j = a + (b-a)*j/segments (exact integer lerp: (delta * j) / segments in int64,
// truncating — endpoints EXACT: P_0 == a, P_segments == b). INTERIOR points (0 < j < segments) get two
// pcg::PcgHash jitters — stream key seed ^ tick*0x9E3779B9u (golden-ratio tick fold), index 2j / 2j+1 —
// one along the beam's SIDE vector, one along its UP vector (side = SideDir(beam dir), up = normalize(
// side x dir) — computed ONCE from the straight beam, so jitter never re-aims the beam). Strip rails:
// left/right = P_j -/+ side*(widthQ/2); indices = the pinned SweepStrip winding -> positions.size() ==
// (segments+1)*2, indices.size() == segments*6 (the EXACT segment-count contract). Endpoints are NEVER
// jittered (the beam stays anchored). jitterAmpQ==0 -> every offset is identically 0 -> the EXACT straight
// strip, independent of seed/tick (the identity control). segments < 1 or widthQ <= 0 -> empty.
inline StripGeom BuildBeam(const FxVec3& a, const FxVec3& b, fx widthQ, int segments,
                           fx jitterAmpQ, uint32_t seed, uint32_t tick) {
    StripGeom out;
    if (segments < 1 || widthQ <= 0) return out;
    const FxVec3 delta = FxSub(b, a);
    const FxVec3 dir   = FxNormalize(delta);
    const FxVec3 side  = spline::SideDir(dir);                       // tangent x up, renormalized
    const FxVec3 up    = FxNormalize(spline::FxCrossQ(side, dir));   // the second jitter axis
    const fx halfW = widthQ >> 1;
    const uint32_t key = seed ^ (tick * 0x9E3779B9u);                // the per-tick jitter stream
    out.positions.reserve(((size_t)segments + 1u) * 2u);
    for (int j = 0; j <= segments; ++j) {
        FxVec3 p{ a.x + (fx)(((int64_t)delta.x * j) / segments),
                  a.y + (fx)(((int64_t)delta.y * j) / segments),
                  a.z + (fx)(((int64_t)delta.z * j) / segments) };
        if (j > 0 && j < segments) {                                 // interior points only — anchored ends
            const fx js = JitterFromHash(pcg::PcgHash(key, (uint32_t)(2 * j)),      jitterAmpQ);
            const fx ju = JitterFromHash(pcg::PcgHash(key, (uint32_t)(2 * j + 1)),  jitterAmpQ);
            p = FxAdd(p, FxAdd(FxScale(side, js), FxScale(up, ju)));
        }
        out.positions.push_back(FxSub(p, FxScale(side, halfW)));     // left
        out.positions.push_back(FxAdd(p, FxScale(side, halfW)));     // right
    }
    out.indices.reserve((size_t)segments * 6u);
    for (int i = 0; i < segments; ++i) {
        const uint32_t va = (uint32_t)(2 * i);
        const uint32_t vb = va + 1u;
        const uint32_t vc = (uint32_t)(2 * (i + 1));
        const uint32_t vd = vc + 1u;
        out.indices.push_back(va); out.indices.push_back(vb); out.indices.push_back(vc);
        out.indices.push_back(vc); out.indices.push_back(vb); out.indices.push_back(vd);
    }
    return out;
}

// ===== 3. MESH EMITTERS ===================================================================================

// FxInstance: ONE mesh-emitter instance in the INTEGER layer — the bit-exact position + the ±LSB-unit
// orientation quaternion (the digestable half; the float Mat4 is the presentation bridge below).
struct FxInstance {
    FxVec3 pos;
    FxQuat orient;
};

// OrientFromVelocity: the integer orientation that rotates +Z onto the velocity direction — the SP1
// half-angle YAW (spline::YawFromTangent: +Z onto the XZ direction, w/y = sqrt((kOne±dz)/2) via FxISqrt)
// COMPOSED with a half-angle PITCH about local X. Pitch derivation: with the unit dir's horizontal length
// h = |dir.xz| and vertical dy, the pitch phi about X mapping +Z -> (0, dy, h) satisfies cos(phi) = h,
// sin(phi) = -dy (right-hand X rotation: +Z -> (0, -sin, cos)); half-angle: w = sqrt((kOne+h)/2),
// x = sign(-dy) * sqrt((kOne-h)/2) — h >= 0 always, so |phi| <= 90 degrees and the sqrt identity never
// wraps. q = FxQuatNormalize(yaw * pitch) (Hamilton: pitch applied first, then yaw — FxRotate(q, +Z)
// lands on dir to within the integer-sqrt LSB band, the test pins the tolerance). Zero velocity ->
// identity (deterministic). NO trig, NO LUT, NO float.
inline FxQuat OrientFromVelocity(const FxVec3& vel) {
    const fx len = FxLength(vel);
    if (len == 0) return FxQuat{0, 0, 0, kOne};
    const FxVec3 dir = FxNormalize(vel);
    const FxQuat yaw = spline::YawFromTangent(dir);              // flattens to XZ internally
    fx h = FxLength(FxVec3{dir.x, 0, dir.z});                    // horizontal length, >= 0
    if (h > kOne) h = kOne;                                      // integer-normalize 1-LSB overshoot clamp
    const fx wq = (fx)FxISqrt((int64_t)((kOne + h) >> 1) << kFrac);
    fx xq       = (fx)FxISqrt((int64_t)((kOne - h) >> 1) << kFrac);
    if (dir.y > 0) xq = -xq;                                     // sign(sin(phi)) = sign(-dy); dy==0 -> xq==0
    return FxQuatNormalize(FxQuatMul(yaw, FxQuat{xq, 0, 0, wq}));
}

// BuildMeshInstances: the mesh-emitter generator — a PURE function of the pool: one FxInstance per ALIVE
// particle (ascending slot order, the alive-bit guard — count == CountAlive(pool) EXACTLY), pos = the
// bit-exact particle position, orient = OrientFromVelocity(vel). Pure integer.
inline std::vector<FxInstance> BuildMeshInstances(const ParticlePool& pool) {
    std::vector<FxInstance> out;
    out.reserve(pool.particles.size());
    for (const FxParticle& p : pool.particles)
        if (p.flags & kFlagAlive)
            out.push_back(FxInstance{p.pos, OrientFromVelocity(p.vel)});
    return out;
}

inline uint64_t DigestInstances(const std::vector<FxInstance>& v) {
    std::vector<uint8_t> buf;
    buf.reserve(v.size() * 28u + 4u);
    PutU32(buf, (uint32_t)v.size());
    for (const FxInstance& in : v) {
        PutVec3(buf, in.pos);
        PutU32(buf, (uint32_t)in.orient.x); PutU32(buf, (uint32_t)in.orient.y);
        PutU32(buf, (uint32_t)in.orient.z); PutU32(buf, (uint32_t)in.orient.w);
    }
    return hf::net::DigestBytes(buf.data(), buf.size());
}

// MeshInstanceTransform: the FLOAT presentation bridge (the FPX6 FxBodyTransform shape) — translate(pos) *
// rotate(orient) * scale(s), feeding the EXISTING instanced-lit pipeline with ANY mesh (cube/sphere/glTF).
// Render-only: READS the bit-exact integer instance, never feeds back.
inline math::Mat4 MeshInstanceTransform(const FxInstance& in, float s) {
    using sim::fpx::FxToFloat;
    const math::Vec3 t{FxToFloat(in.pos.x), FxToFloat(in.pos.y), FxToFloat(in.pos.z)};
    const math::Quat q = math::Normalize(math::Quat{
        FxToFloat(in.orient.x), FxToFloat(in.orient.y), FxToFloat(in.orient.z), FxToFloat(in.orient.w)});
    return math::FromTRS(t, q, math::Vec3{s, s, s});
}

// ===== 4. PARTICLE LIGHTS =================================================================================

inline constexpr uint32_t kMaxParticleLights = 64;   // the ML1-feed cap (the brightest-N selection bound)

// ParticleLightFx: ONE selected particle light in the INTEGER layer — the bit-exact position, the integer
// brightness key, the owning seed (drives the palette color in the float bridge), and the pool slot.
struct ParticleLightFx {
    FxVec3   pos;
    fx       bright = 0;   // Q16.16 remaining life (lifetime - age) — the deterministic brightness key
    uint32_t seed   = 0;   // the particle's spawn hash (the color-ramp selector)
    uint32_t slot   = 0;   // the pool slot (the pinned tie-break key)
};

// BuildParticleLights: the light generator — a PURE function of (pool, maxN). Brightness = remaining life
// (lifetime - age, Q16.16 — a fresh particle burns brightest and fades to death; pure integer). Selection:
// EXACTLY min(maxN, alive) entries, sorted brightest-first with the LOWEST-SLOT tie-break (a strict total
// order over (bright desc, slot asc) — deterministic on every platform/STL: no two entries compare equal).
inline std::vector<ParticleLightFx> BuildParticleLights(const ParticlePool& pool, uint32_t maxN) {
    std::vector<ParticleLightFx> all;
    all.reserve(pool.particles.size());
    const uint32_t n = (uint32_t)pool.particles.size();
    for (uint32_t i = 0; i < n; ++i) {
        const FxParticle& p = pool.particles[(size_t)i];
        if (!(p.flags & kFlagAlive)) continue;
        all.push_back(ParticleLightFx{p.pos, p.lifetime - p.age, p.seed, i});
    }
    std::sort(all.begin(), all.end(), [](const ParticleLightFx& a, const ParticleLightFx& b) {
        if (a.bright != b.bright) return a.bright > b.bright;   // brightest first
        return a.slot < b.slot;                                 // then the lowest slot (the pinned tie-break)
    });
    if (all.size() > (size_t)maxN) all.resize((size_t)maxN);
    return all;
}

inline uint64_t DigestLights(const std::vector<ParticleLightFx>& v) {
    std::vector<uint8_t> buf;
    buf.reserve(v.size() * 24u + 4u);
    PutU32(buf, (uint32_t)v.size());
    for (const ParticleLightFx& l : v) {
        PutVec3(buf, l.pos);
        PutU32(buf, (uint32_t)l.bright); PutU32(buf, l.seed); PutU32(buf, l.slot);
    }
    return hf::net::DigestBytes(buf.data(), buf.size());
}

// LightColorFromSeed: the FLOAT bridge's fixed 8-entry ember palette — selected by an INTEGER hash fold
// (seed * Knuth >> 29 -> 0..7), so the CHOICE is integer-deterministic; only the palette values are float
// literals (bit-identical everywhere — no arithmetic).
inline math::Vec3 LightColorFromSeed(uint32_t seed) {
    static const float kR[8] = {1.00f, 1.00f, 0.95f, 0.55f, 0.30f, 0.60f, 1.00f, 0.85f};
    static const float kG[8] = {0.45f, 0.75f, 0.90f, 0.85f, 0.65f, 0.40f, 0.30f, 0.90f};
    static const float kB[8] = {0.15f, 0.25f, 0.35f, 1.00f, 1.00f, 1.00f, 0.55f, 0.40f};
    const uint32_t k = (seed * 2654435761u) >> 29;   // 0..7
    return math::Vec3{kR[k], kG[k], kB[k]};
}

// ParticleLightsToClustered: the FLOAT bridge into THE EXISTING ML1 clustered many-light path — one
// render::clustered::Light per selected particle light, position transformed world -> VIEW space by the
// caller's view matrix (the clustered.h contract), radius = radiusQ (Q16.16 -> float), color =
// LightColorFromSeed, intensity scaled by the normalized remaining life (bright/lifetimeQ, render-only
// float). The output feeds clns::BuildClusters VERBATIM — the composition seam the test pins.
inline std::vector<clns::Light> ParticleLightsToClustered(const std::vector<ParticleLightFx>& in,
                                                          fx radiusQ, fx lifetimeQ,
                                                          const math::Mat4& view) {
    using sim::fpx::FxToFloat;
    std::vector<clns::Light> out;
    out.reserve(in.size());
    for (const ParticleLightFx& l : in) {
        clns::Light L{};
        float vw = 0.0f;
        const math::Vec3 world{FxToFloat(l.pos.x), FxToFloat(l.pos.y), FxToFloat(l.pos.z)};
        L.viewPos   = math::MulPointDivide(view, world, vw);   // affine: w stays 1
        L.radius    = FxToFloat(radiusQ);
        L.color     = LightColorFromSeed(l.seed);
        L.intensity = lifetimeQ > 0 ? 2.0f * FxToFloat(l.bright) / FxToFloat(lifetimeQ) : 1.0f;
        out.push_back(L);
    }
    return out;
}

// ===== THE VR1 SHOWCASE — the shared scenario + raster (the LA1/GAS1 convention) ==========================
// Header-local so BOTH backends (Vulkan --vr1-vfx-shot / Metal --vr1-vfx) run the IDENTICAL bytes: the PA1
// pulsing fountain stepped kShotSteps ticks with per-tick trail recording, then all FOUR generators over
// the final pool + ONE pure-INTEGER side-view raster -> strict-zero cross-backend BY CONSTRUCTION. The
// pinned digests + the test's scenario are the same artifact with three witnesses.

inline constexpr uint32_t kShotSteps    = 96;          // PA1 fountain ticks (>1.5 lifetimes of churn)
inline constexpr uint32_t kShotTrailK   = 8;           // ring capacity: 8 past positions per trail
inline constexpr fx       kShotTrailW   = kOne / 6;    // ribbon full width (~0.167 wu at the head)
inline constexpr fx       kShotBeamW    = kOne / 8;    // beam strip width
inline constexpr int      kShotBeamSegs = 24;          // beam segment count (EXACT — the pinned contract)
inline constexpr fx       kShotBeamAmp  = kOne / 3;    // beam jitter amplitude (the lightning wobble)
inline constexpr uint32_t kShotBeamSeed = 0x56523101u; // 'V','R','1',v1
inline constexpr uint32_t kShotLightN   = kMaxParticleLights;   // brightest-64 cap
inline constexpr fx       kShotLightRad = kOne * 3;    // per-light attenuation radius (3 wu)

// The fixed beam endpoints: an arc anchor high on each side of the fountain plume.
inline FxVec3 ShotBeamA() { return FxVec3{-5 * kOne, 6 * kOne, 0}; }
inline FxVec3 ShotBeamB() { return FxVec3{ 5 * kOne, 6 * kOne, 0}; }

struct VfxShotRun {
    // The driven sim (the PA1 pulsing-fountain asset, stepped kShotSteps ticks).
    uint32_t aliveCount = 0;
    std::vector<FxVec3> alivePos;          // the final alive particle positions (the raster dots)
    // The four generators' outputs + their pinned integer digests.
    TrailHistory                 trails;
    RibbonSet                    ribbons;
    StripGeom                    beam;
    std::vector<FxInstance>      instances;
    std::vector<ParticleLightFx> lights;
    uint64_t trailsDigest = 0, ribbonDigest = 0, beamDigest = 0, instDigest = 0, lightDigest = 0;
    uint64_t digest = 0;                   // the combined PURE-INTEGER digest (the stat-line currency)
    // The ML1 composition seam (float clustered assignment — pinned PER-TOOLCHAIN-FAMILY, the ML1
    // convention; NOT folded into `digest` or the raster, so the strict-zero golden stays pure integer).
    mlns::AssignStats assignStats;
    uint64_t          assignDigest = 0;
};

// RunVfxShotScenario: the pure function both backends call. Drives the FIXED PA1 pulsing-fountain asset
// (particle_author.h::MakePulsingFountainEffect — the AUTHORED effect, its graph digest already pinned by
// particle_author_test) kShotSteps ticks with zero player input, recording trails each tick; then runs
// all four VR1 generators over the final pool and the ML1 clustered assignment over the particle lights.
inline VfxShotRun RunVfxShotScenario() {
    namespace pa = sim::pauthor;
    VfxShotRun run;

    // (1) The driven effect: the PA1 asset, the PA1 showcase capacity, dt = 1/60 (the PT scene tick).
    pa::AuthoredEffect effect = pa::MakePulsingFountainEffect();
    ParticlePool pool = sim::particles::InitParticlePool(pa::kShowcaseCapacity);
    run.trails = InitTrailHistory(pa::kShowcaseCapacity, kShotTrailK);
    const fx dt = kOne / 60;
    const std::vector<hf::flow::Reg> noInput((size_t)pa::kShowcaseChannels, 0);
    for (uint32_t t = 0; t < kShotSteps; ++t) {
        pa::StepAuthoredEffect(effect, pool, dt, pool.tick, noInput);   // the bit-exact authored tick
        UpdateTrails(pool, run.trails);                                 // the per-tick trail record
    }
    run.aliveCount = sim::particles::CountAlive(pool);
    for (const FxParticle& p : pool.particles)
        if (p.flags & kFlagAlive) run.alivePos.push_back(p.pos);

    // (2) The four generators (all pure integer functions of the final pool / params / tick).
    run.ribbons   = BuildRibbons(run.trails, kShotTrailW);
    run.beam      = BuildBeam(ShotBeamA(), ShotBeamB(), kShotBeamW, kShotBeamSegs, kShotBeamAmp,
                              kShotBeamSeed, pool.tick);
    run.instances = BuildMeshInstances(pool);
    run.lights    = BuildParticleLights(pool, kShotLightN);

    // (3) The pinned integer digests + the combined stat-line digest (FnvAppend chain — pure integer).
    run.trailsDigest = DigestTrails(run.trails);
    run.ribbonDigest = DigestStrip(run.ribbons.geom);
    run.beamDigest   = DigestStrip(run.beam);
    run.instDigest   = DigestInstances(run.instances);
    run.lightDigest  = DigestLights(run.lights);
    uint64_t h = run.trailsDigest;
    h = mlns::FnvAppend(h, &run.ribbonDigest, sizeof(run.ribbonDigest));
    h = mlns::FnvAppend(h, &run.beamDigest,   sizeof(run.beamDigest));
    h = mlns::FnvAppend(h, &run.instDigest,   sizeof(run.instDigest));
    h = mlns::FnvAppend(h, &run.lightDigest,  sizeof(run.lightDigest));
    run.digest = h;

    // (4) The ML1 composition seam: the particle lights fed through render::clustered::BuildClusters over
    // the fixed showcase camera (eye {0,10,20} -> the fountain at {0,2,0}, 60deg, 1280x720, 16x9x24 —
    // the ML1 grid shape). FLOAT (pow/log slice math) — assignDigest pinned per-toolchain-family.
    {
        const float kNear = 0.5f, kFar = 90.0f, fovY = 1.04719755f, W = 1280.0f, H = 720.0f;
        const math::Mat4 view = math::Mat4::LookAt({0.0f, 10.0f, 20.0f}, {0.0f, 2.0f, 0.0f}, {0, 1, 0});
        const math::Mat4 proj = math::Mat4::Perspective(fovY, W / H, kNear, kFar);
        const clns::Grid grid = clns::MakeGrid(proj, kNear, kFar, W, H, 16, 9, 24);
        const std::vector<clns::Light> viewLights =
            ParticleLightsToClustered(run.lights, kShotLightRad, effect.baseCfg.lifetime, view);
        clns::ClusterBuffers cb = clns::BuildClusters(grid, viewLights);
        if (cb.lightIndices.empty()) cb.lightIndices.push_back(0u);   // the ML1 non-zero-size convention
        run.assignStats  = mlns::ComputeAssignStats(cb);
        run.assignDigest = mlns::DigestAssignment(cb);
    }
    return run;
}

// ===== RenderVfxShot: the PURE-INTEGER side-view raster both backends call (strict-zero BY CONSTRUCTION).
// World window x in [-6.4, +6.4], y in [-2.6, +7.8] -> 512x416 BGRA8 (40 px per world unit; y up on
// screen — the beam anchors at (+-5, 6) and the ground at y=-2 both frame inside).
// Layers (back to front): dark backdrop + the groundY line, alive-particle dots (white), ribbon rails
// (per-trail hashed cool palette, both rails as integer Bresenham lines), the beam rails (hot white-
// orange), mesh-emitter squares (amber, plus a rotated-+Z direction tick — the orientation made visible),
// particle-light diamonds (per-seed palette index -> a fixed BGRA table). Every coordinate is an int64
// Q16.16 -> pixel scale; every color is a literal — NO float anywhere.
inline constexpr int kShotImgW = 512;
inline constexpr int kShotImgH = 416;

inline void VfxShotPixel(std::vector<uint8_t>& bgra, int x, int y, uint8_t b, uint8_t g, uint8_t r) {
    if (x < 0 || x >= kShotImgW || y < 0 || y >= kShotImgH) return;
    const size_t p = ((size_t)y * kShotImgW + (size_t)x) * 4u;
    bgra[p + 0] = b; bgra[p + 1] = g; bgra[p + 2] = r; bgra[p + 3] = 255;
}
inline int VfxShotPx(fx wx) { return 256 + (int)(((int64_t)wx * 40) >> kFrac); }
inline int VfxShotPy(fx wy) { return 312 - (int)(((int64_t)wy * 40) >> kFrac); }

// Integer Bresenham with per-pixel clipping (deterministic, pure integer).
inline void VfxShotLine(std::vector<uint8_t>& bgra, int x0, int y0, int x1, int y1,
                        uint8_t b, uint8_t g, uint8_t r) {
    int dx = x1 > x0 ? x1 - x0 : x0 - x1, sx = x0 < x1 ? 1 : -1;
    int dy = y1 > y0 ? y0 - y1 : y1 - y0, sy = y0 < y1 ? 1 : -1;   // dy <= 0 (the classic form)
    int err = dx + dy;
    // Bound the walk (window diagonal) so a wild segment can never loop long.
    for (int guard = 0; guard < 4096; ++guard) {
        VfxShotPixel(bgra, x0, y0, b, g, r);
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

inline void RenderVfxShot(const VfxShotRun& run, std::vector<uint8_t>& bgra,
                          uint32_t& outW, uint32_t& outH) {
    outW = (uint32_t)kShotImgW; outH = (uint32_t)kShotImgH;
    bgra.assign((size_t)kShotImgW * kShotImgH * 4u, 0);
    // Backdrop: a dark vertical wash (row-indexed integer ramp) + the ground line at y = -2 wu.
    for (int y = 0; y < kShotImgH; ++y) {
        const int base = 14 + (y * 22) / kShotImgH;   // 14 .. 35 top->bottom
        for (int x = 0; x < kShotImgW; ++x) {
            const size_t p = ((size_t)y * kShotImgW + (size_t)x) * 4u;
            bgra[p + 0] = (uint8_t)(base + 12); bgra[p + 1] = (uint8_t)(base + 2);
            bgra[p + 2] = (uint8_t)base;        bgra[p + 3] = 255;
        }
    }
    { const int gy = VfxShotPy(-2 * kOne);
      VfxShotLine(bgra, 0, gy, kShotImgW - 1, gy, 90, 84, 74); }
    // (1) alive particles: single white-blue dots.
    for (const FxVec3& p : run.alivePos)
        VfxShotPixel(bgra, VfxShotPx(p.x), VfxShotPy(p.y), 235, 220, 205);
    // (2) ribbons: both rails per trail, a hashed cool palette (integer slot fold).
    static const uint8_t kTrailB[4] = {250, 230, 205, 255};
    static const uint8_t kTrailG[4] = {170, 120, 190, 210};
    static const uint8_t kTrailR[4] = { 60,  70,  40, 120};
    for (uint32_t t = 0; t < run.ribbons.trailCount; ++t) {
        const uint32_t base = run.ribbons.trailFirstVert[t];
        const uint32_t m    = run.ribbons.trailSampleCount[t];
        const uint32_t c    = t & 3u;
        for (uint32_t j = 0; j + 1u < m; ++j)
            for (uint32_t rail = 0; rail < 2u; ++rail) {
                const FxVec3& p0 = run.ribbons.geom.positions[base + 2u * j + rail];
                const FxVec3& p1 = run.ribbons.geom.positions[base + 2u * (j + 1u) + rail];
                VfxShotLine(bgra, VfxShotPx(p0.x), VfxShotPy(p0.y), VfxShotPx(p1.x), VfxShotPy(p1.y),
                            kTrailB[c], kTrailG[c], kTrailR[c]);
            }
    }
    // (3) the beam: both rails, hot white-orange (the lightning arc over the plume).
    {
        const size_t samples = run.beam.positions.size() / 2u;
        for (size_t j = 0; j + 1u < samples; ++j)
            for (size_t rail = 0; rail < 2u; ++rail) {
                const FxVec3& p0 = run.beam.positions[2u * j + rail];
                const FxVec3& p1 = run.beam.positions[2u * (j + 1u) + rail];
                VfxShotLine(bgra, VfxShotPx(p0.x), VfxShotPy(p0.y), VfxShotPx(p1.x), VfxShotPy(p1.y),
                            170, 235, 255);
            }
    }
    // (4) mesh-emitter instances: 5x5 amber squares + the rotated-+Z direction tick (FxRotate — the
    // orientation made VISIBLE in the integer layer). Draw a sparse sample (every 4th) to keep the plume
    // readable; the count/digest cover the full set.
    for (size_t i = 0; i < run.instances.size(); i += 4) {
        const FxInstance& in = run.instances[i];
        const int cx = VfxShotPx(in.pos.x), cy = VfxShotPy(in.pos.y);
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx)
                VfxShotPixel(bgra, cx + dx, cy + dy, 40, 150, 235);
        const FxVec3 fwd = FxRotate(in.orient, FxVec3{0, 0, kOne});   // where the mesh's +Z points
        const int tx = cx + (int)(((int64_t)fwd.x * 8) >> kFrac);
        const int ty = cy - (int)(((int64_t)fwd.y * 8) >> kFrac);
        VfxShotLine(bgra, cx, cy, tx, ty, 120, 200, 255);
    }
    // (5) particle lights: 7px diamonds in the per-seed palette (the SAME integer palette fold the float
    // bridge uses -> the marker color IS the light color choice, integer-witnessed).
    static const uint8_t kLampB[8] = { 40,  64,  90, 255, 255, 255, 140, 100};
    static const uint8_t kLampG[8] = {115, 190, 230, 215, 165, 100,  75, 230};
    static const uint8_t kLampR[8] = {255, 255, 245, 140,  75, 155, 255, 215};
    for (const ParticleLightFx& l : run.lights) {
        const int cx = VfxShotPx(l.pos.x), cy = VfxShotPy(l.pos.y);
        const uint32_t k = (l.seed * 2654435761u) >> 29;   // the LightColorFromSeed selector, verbatim
        for (int dy = -3; dy <= 3; ++dy) {
            const int span = 3 - (dy < 0 ? -dy : dy);
            for (int dx = -span; dx <= span; ++dx)
                VfxShotPixel(bgra, cx + dx, cy + dy, kLampB[k], kLampG[k], kLampR[k]);
        }
    }
}

}  // namespace render
}  // namespace hf::vfx
