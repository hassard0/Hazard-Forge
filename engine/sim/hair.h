#pragma once
// Slice HR1 — DETERMINISTIC STRAND/HAIR SIMULATION CORE (Track-S slice S2 of
// docs/SUPERIORITY_ROADMAP.md): Q16.16 fixed-point PBD RODS — strands as 1D constraint chains with
// BENDING stiffness, root-pinned, gravity-driven, with strand<->strand collision — BIT-IDENTICAL on
// every platform AND lockstep/rollback-replayable. This is the ONE deterministic-sim material family
// the engine was missing (rigid fpx / cloth / fluid / grain / fracture / particles / vehicles /
// ragdoll all shipped); UE5's Groom is float / non-deterministic, so a deterministic strand sim is a
// new headline. Namespace hf::sim::hair, header-only, pure CPU, NO device / backend symbols, NO float
// on the sim path, NO RNG, NO clock.
//
// THE ~80% MOLD-REUSE (engine/sim/cloth.h #included READ-ONLY, byte-UNTOUCHED — hair is the additive
// sibling; a strand is cloth's 1D degenerate case + bending):
//   * STATE     : HairVert IS cloth::ClothParticle verbatim (pos/prev/vel/invMass/flags, the CL1
//                 std430-packable 44-byte POD) — vert 0 (the ROOT) pinned via kFlagPinned/invMass 0.
//   * INTEGRATE : cloth::IntegrateParticle AS-IS (the CL1 semi-implicit Euler + prev snapshot + ground
//                 floor-clamp; the prev snapshot is the PBD/Verlet anchor the re-encode below reads).
//   * STRETCH   : cloth::Constraint + cloth::SolveDistanceConstraint AS-IS (the CL3 PBD projection,
//                 pinned share 0) over consecutive-vert (i, i+1) chains.
//   * COLLISION : the CL7 pair-grid mold EXACTLY — cloth::BuildClothAdjacency (the strand-neighbor
//                 exclusion set), cloth::BuildSelfCandidates (the FL2-twin bounded dense grid, 27-cell
//                 stencil in the FIXED (dz,dy,dx ascending; ascending j) order, box accept) and
//                 cloth::SolveSelfCollision (the two-pass JACOBI gather/apply, race-free per vert).
//                 Because the exclusion is the constraint 1-ring and the strand graph has stretch
//                 (+/-1) AND bend (+/-2) edges, exactly the SAME-STRAND verts <= 2 apart are excluded
//                 -> verts of DIFFERENT strands plus same-strand verts >= 3 apart closer than
//                 thickness = 2*radius separate via the sphere-sphere projection, as specced.
//   * DIGEST    : cloth::ClothDigest AS-IS (HairVert == ClothParticle) — the golden currency.
//   * LOCKSTEP  : the CL5/FPX5 command+snapshot mold (HairCommand root-drag stream; snapshot = the
//                 deep vector copy; RunHairLockstep / RunHairRollback are the RunClothLockstep /
//                 RunClothRollback twins).
//
// THE NEW PHYSICS vs cloth — BENDING STIFFNESS (the exact integer scheme, documented):
//   A strand's bend constraint is a DISTANCE constraint between vert i and vert i+2 (the standard PBD
//   bending-as-distance trick — NO angles, NO trig, stays integer) with rest2 = 2*restLen (the straight
//   rest shape: |p_{i+2} - p_i| == 2L iff the three verts are collinear at rest spacing). The k_bend
//   stiffness (Q16.16 in [0, kOne]) is applied as a FRACTIONAL PROJECTION — SolveBendConstraint is the
//   CL3 SolveDistanceConstraint body with the penetration PRE-SCALED by k_bend before the inverse-mass
//   split:
//       d = pos[j] - pos[i]; len = FxLength(d); pen = len - rest2; penS = fxmul(pen, k_bend)
//       n = FxNormalize(d); wi = fxdiv(invMass_i, wsum); wj = fxdiv(invMass_j, wsum)
//       pos[i] += n * fxmul(penS, wi);  pos[j] -= n * fxmul(penS, wj)
//   k_bend == kOne -> fxmul(pen, kOne) == pen EXACTLY (the Q16.16 identity) -> the full CL3 distance
//   projection; k_bend == 0 -> the pass is SKIPPED entirely (the identity-at-zero: a zero-stiffness
//   strand is bit-identical to the pure stretch chain). k_bend is PER STRAND (a std::vector<fx>, one
//   entry per strand — "some stiff, some limp" in one solve).
//
//   HONEST CAVEATS (the CL3/FPX3 reality + the known PBD-bending softness): (1) the per-iteration
//   fractional projection SATURATES — effective stiffness ~ 1-(1-k)^iters, so high k_bend does NOT
//   approach a rigid rod, it approaches the plain distance constraint, which itself is soft for long
//   chains under gravity (a "stiff" strand still droops visibly; the tests pin the REAL droop numbers,
//   they do not pretend rigidity); (2) fxmul TRUNCATES toward zero, so a small pen with a small k_bend
//   can round the correction to 0 LSBs — a stiffness dead-zone at sub-LSB errors (deterministic);
//   (3) bending-as-distance cannot anchor the ROOT DIRECTION with a single pinned vert (ANY straight
//   configuration satisfies every bend constraint, so a 1-pin strand pivots freely at the root and
//   hangs vertical regardless of k_bend) — a direction-clamped root pins verts 0 AND 1 (the standard
//   PBD rooting trick; InitStrands takes pinCount for exactly this).
//
// THE STEP (StepHair) — the FL7/couple_cf VELOCITY RE-ENCODE discipline (the newest slice convention),
// NOT the CL3 keep-integrated-velocity one: after the constraint passes the velocity is RE-DERIVED from
// the net position change, v = (pos - prev) / dt (per-axis fxdiv; fxdiv(x, 0) == 0 by the fpx
// contract), then scaled by the Q16.16 damp factor (damp == kOne is the EXACT identity — fxmul(v, kOne)
// == v for every int32). This is what lets a swinging strand SETTLE (the CL3 convention would grow the
// integrated velocity unboundedly on a hanging chain); damp < kOne is the standard PBD strand damping.
//   1. gravity integrate every vert (cloth::IntegrateParticle — pinned untouched)
//   2. if radius > 0: build the collision candidate list ONCE from the post-integrate positions (the
//      FL4/CL7 per-step broadphase discipline; candidates re-tested against CURRENT positions each pass)
//   3. `iters` rounds of (STRETCH Gauss-Seidel pass -> BEND fractional pass -> COLLISION Jacobi pass),
//      each in its FIXED build order
//   4. ground clamp (cloth::CollidePlane)
//   5. velocity re-encode: vel = fxmul((pos - prev) / dt, damp) per axis, non-pinned verts only
// Pure integer, fixed op order -> two runs byte-identical AND bit-identical on every platform.
//
// THE CPU/GPU CHOICE (documented, the CF1/FR8 precedent): HR1 ships PURE CPU on BOTH backends — the
// stretch/bend/collision projections are int64-backed (FxLength/FxNormalize/fxdiv), which under the
// house convention would make any GPU kernel DXC/Vulkan-only with a Metal CPU reference anyway; CF1 and
// FR8 established that a pure-CPU slice run IDENTICALLY on Vulkan-Windows and Metal-Mac is
// cross-backend bit-identical BY CONSTRUCTION and that this is an acceptable proof shape. NO new
// shader, NO new RHI; a GPU kernel is a future refinement, not HR1.
//
// BOUNDS: every command target / strand lookup / adjacency access is range-checked (deterministic
// no-op on out-of-range); per-vert counts are showcase-scale (hundreds) so all int32 sums stay far
// below 2^31 (the CL7 overflow-bound argument applies verbatim to the reused Jacobi gather).

#include <cstdint>
#include <vector>

#include "sim/cloth.h"   // READ-ONLY reuse: ClothParticle/Constraint/IntegrateParticle/
                         // SolveDistanceConstraint/BuildClothAdjacency/BuildSelfCandidates/
                         // SolveSelfCollision/CollidePlane/ClothDigest + the fpx Q16.16 toolbox

namespace hf::sim {
namespace hair {

// The fpx Q16.16 toolbox through the cloth re-exports (NO new fixed-point primitives).
using cloth::fx;
using cloth::FxVec3;
using cloth::fxmul;
using cloth::fxdiv;
using cloth::FxAdd;
using cloth::FxSub;
using cloth::FxScale;
using cloth::FxLength;
using cloth::FxNormalize;
inline constexpr int kFrac = cloth::kFrac;
inline constexpr fx  kOne  = cloth::kOne;

// ----- The strand state: HairVert IS the CL1 particle (the mold reused verbatim) -------------------
// N strands x M verts each, FLAT array: strand s owns verts [s*M, (s+1)*M). Vert 0 of each strand is
// the ROOT (pinned; a direction-clamped root also pins vert 1 — see the bending caveat above).
using HairVert = cloth::ClothParticle;
using cloth::kFlagPinned;
using cloth::Constraint;
using cloth::kConstraintStructural;   // reused as the STRETCH kind (consecutive-vert)
using cloth::kConstraintBend;         // reused as the BEND kind (2-away)
using cloth::ClothAdjacency;          // the collision exclusion set (the CL7 CSR table)

// The strand layout: S strands x M verts, restLen the Q16.16 segment rest length.
struct HairStrands {
    int S = 0;         // strand count
    int M = 0;         // verts per strand (>= 2 for stretch, >= 3 for bend)
    fx  restLen = 0;   // Q16.16 rest length of one segment
};

// VertIndex(hs, s, i): the flat index of strand s's vert i (row-major, strand-major).
inline int VertIndex(const HairStrands& hs, int s, int i) { return s * hs.M + i; }

// StrandOfVert(hs, v): the strand owning flat vert v (integer divide; v assumed in range).
inline int StrandOfVert(const HairStrands& hs, int v) { return hs.M > 0 ? v / hs.M : 0; }

// ----- InitStrands: the deterministic strand lattice (roots on a bar) ------------------------------
// Strand s's vert i sits at origin + s*rootStep + growDir * (i * restLen) (component-wise fxmul; with
// an axis-aligned unit growDir the layout is EXACT integer multiples). The first `pinCount` verts of
// every strand are PINNED (invMass 0, kFlagPinned) — pinCount 1 = the free-pivot root, pinCount 2 = the
// direction-clamped root (required for the bending droop proof, see the header caveat). prev == pos,
// vel == 0 at rest. pinCount is clamped to [1, M]. Returns the populated S*M vert vector.
inline std::vector<HairVert> InitStrands(const HairStrands& hs, const FxVec3& origin,
                                         const FxVec3& rootStep, const FxVec3& growDir,
                                         int pinCount = 1) {
    if (pinCount < 1) pinCount = 1;
    if (pinCount > hs.M) pinCount = hs.M;
    std::vector<HairVert> verts((size_t)(hs.S * hs.M));
    for (int s = 0; s < hs.S; ++s) {
        const FxVec3 root{origin.x + (fx)(s * (int64_t)rootStep.x),
                          origin.y + (fx)(s * (int64_t)rootStep.y),
                          origin.z + (fx)(s * (int64_t)rootStep.z)};
        for (int i = 0; i < hs.M; ++i) {
            HairVert v;
            const fx along = (fx)((int64_t)i * (int64_t)hs.restLen);   // i*restLen (exact int multiple)
            v.pos = FxVec3{root.x + fxmul(growDir.x, along),
                           root.y + fxmul(growDir.y, along),
                           root.z + fxmul(growDir.z, along)};
            v.prev = v.pos;
            v.vel = FxVec3{0, 0, 0};
            if (i < pinCount) { v.invMass = 0;    v.flags = kFlagPinned; }
            else              { v.invMass = kOne; v.flags = 0; }
            verts[(size_t)VertIndex(hs, s, i)] = v;
        }
    }
    return verts;
}

// ----- BuildHairConstraints: the per-strand STRETCH + BEND chains (fixed strand-major order) --------
// Per strand s (ascending), per vert i (ascending): STRETCH (base+i, base+i+1) restLen for i < M-1
// (kConstraintStructural — the CL3 full projection); BEND (base+i, base+i+2) rest 2*restLen for
// i < M-2 (kConstraintBend — the fractional k_bend projection). Two SEPARATE lists so the solver's
// stretch pass and bend pass are cleanly ordered (and the identity-at-zero is trivially exact: k_bend
// 0 skips the whole bend list). Deterministic (pure index arithmetic, host-snapped rest lengths).
struct HairConstraints {
    std::vector<Constraint> stretch;   // (i, i+1) restLen — the CL3 distance chain
    std::vector<Constraint> bend;      // (i, i+2) rest 2*restLen — the PBD bending-as-distance trick
};

inline HairConstraints BuildHairConstraints(const HairStrands& hs) {
    HairConstraints hc;
    if (hs.M >= 2) hc.stretch.reserve((size_t)(hs.S * (hs.M - 1)));
    if (hs.M >= 3) hc.bend.reserve((size_t)(hs.S * (hs.M - 2)));
    const fx rest2 = (fx)(2 * (int64_t)hs.restLen);   // the straight rest shape: |p_{i+2}-p_i| == 2L
    for (int s = 0; s < hs.S; ++s) {
        const int base = s * hs.M;
        for (int i = 0; i + 1 < hs.M; ++i)
            hc.stretch.push_back(Constraint{(uint32_t)(base + i), (uint32_t)(base + i + 1),
                                            hs.restLen, kConstraintStructural});
        for (int i = 0; i + 2 < hs.M; ++i)
            hc.bend.push_back(Constraint{(uint32_t)(base + i), (uint32_t)(base + i + 2),
                                         rest2, kConstraintBend});
    }
    return hc;
}

// BuildHairExclusion(vertCount, hc): the collision EXCLUSION set — the CL7 constraint 1-ring over the
// CONCATENATED stretch+bend graph. Same-strand verts <= 2 apart share a constraint -> excluded; verts
// >= 3 apart on the same strand AND all cross-strand pairs remain collision candidates (the spec).
inline ClothAdjacency BuildHairExclusion(size_t vertCount, const HairConstraints& hc) {
    std::vector<Constraint> all;
    all.reserve(hc.stretch.size() + hc.bend.size());
    all.insert(all.end(), hc.stretch.begin(), hc.stretch.end());
    all.insert(all.end(), hc.bend.begin(), hc.bend.end());
    return cloth::BuildClothAdjacency(vertCount, all);
}

// ----- SolveBendConstraint: the k_bend FRACTIONAL distance projection (the new physics) -------------
// The CL3 SolveDistanceConstraint body with the penetration PRE-SCALED by k_bend (Q16.16 fractional
// projection, the exact integer scheme documented in the header banner). k_bend <= 0 -> exact no-op
// (identity-at-zero); k_bend == kOne -> fxmul(pen, kOne) == pen EXACTLY -> the full CL3 projection.
// Pinned (invMass 0) endpoints take share 0 and never move; both-pinned / coincident pairs skip.
inline void SolveBendConstraint(std::vector<HairVert>& verts, const Constraint& c, fx kBend) {
    if (kBend <= 0) return;                              // identity-at-zero: EXACT no-op
    if (c.i >= verts.size() || c.j >= verts.size()) return;   // bounds-checked skip
    HairVert& pi = verts[(size_t)c.i];
    HairVert& pj = verts[(size_t)c.j];
    const fx wsum = pi.invMass + pj.invMass;
    if (wsum == 0) return;                               // both pinned -> skip
    const FxVec3 d = FxSub(pj.pos, pi.pos);
    const fx len = FxLength(d);
    if (len == 0) return;                                // coincident -> no deterministic normal -> skip
    const fx pen  = len - c.restLen;
    const fx penS = fxmul(pen, kBend);                   // THE fractional projection (truncating fxmul)
    const FxVec3 n = FxNormalize(d);
    const fx wi = fxdiv(pi.invMass, wsum);
    const fx wj = fxdiv(pj.invMass, wsum);
    pi.pos = FxAdd(pi.pos, FxScale(n, fxmul(penS, wi)));
    pj.pos = FxSub(pj.pos, FxScale(n, fxmul(penS, wj)));
}

// ----- HairParams: the per-step knobs (all Q16.16 / int, no float) ----------------------------------
struct HairParams {
    FxVec3 gravity;         // Q16.16 gravity acceleration (world units / s^2)
    fx     dt = 0;          // Q16.16 timestep
    fx     groundY = 0;     // the ground plane (the CL1/CL4 floor clamp)
    int    iters = 4;       // K rounds of (stretch -> bend -> collision) per step
    fx     radius = 0;      // strand vert radius r; collision thickness = 2*r; 0 = collision OFF
    fx     damp = kOne;     // velocity damping at the re-encode (kOne = EXACT identity / no damping)
};

// ----- StepHair: one full PBD strand step (the make-or-break reference) -----------------------------
// integrate -> per-step collision broadphase -> `iters` x (stretch -> bend -> collision) -> ground
// clamp -> velocity re-encode (the FL7 discipline; see the header banner for the exact order + why).
// kBendPerStrand[s] is strand s's bend stiffness (out-of-range strand -> 0, bounds-checked). radius
// == 0 -> the collision broadphase + pass are never touched (identity: the pure stretch+bend chain).
// Pure integer, fixed order -> two-run byte-identical + cross-platform bit-identical.
inline void StepHair(const HairStrands& hs, std::vector<HairVert>& verts,
                     const HairConstraints& hc, const ClothAdjacency& excl,
                     const std::vector<fx>& kBendPerStrand, const HairParams& p) {
    // (1) gravity integrate every vert (CL1 verbatim; pinned untouched; prev snapshotted for the
    //     re-encode below).
    const size_t n = verts.size();
    for (size_t i = 0; i < n; ++i)
        cloth::IntegrateParticle(verts[i], p.gravity, p.groundY, p.dt);

    // (2) the collision broadphase ONCE per step from the post-integrate positions (the FL4/CL7
    //     per-step candidate discipline). thickness = 2*radius (sphere-sphere, verts share radius r).
    const fx thickness = (fx)(2 * (int64_t)p.radius);
    cloth::ClothSelfList candidates;
    if (p.radius > 0)
        candidates = cloth::BuildSelfCandidates(verts, excl, thickness);

    // (3) `iters` rounds of (stretch Gauss-Seidel -> bend fractional -> collision Jacobi), each list
    //     in its FIXED build order (the deterministic-order contract).
    for (int it = 0; it < p.iters; ++it) {
        for (size_t e = 0; e < hc.stretch.size(); ++e)
            cloth::SolveDistanceConstraint(verts, hc.stretch[e]);
        for (size_t e = 0; e < hc.bend.size(); ++e) {
            const int s = StrandOfVert(hs, (int)hc.bend[e].i);
            const fx k = (s >= 0 && (size_t)s < kBendPerStrand.size()) ? kBendPerStrand[(size_t)s] : 0;
            SolveBendConstraint(verts, hc.bend[e], k);
        }
        if (p.radius > 0)
            cloth::SolveSelfCollision(verts, candidates, thickness, p.groundY);
    }

    // (4) ground clamp AFTER the constraint passes (a projection may have pushed a vert below).
    cloth::CollidePlane(verts, p.groundY);

    // (5) velocity RE-ENCODE (the FL7/couple_cf discipline): v = (pos - prev) / dt, damped. Pinned
    //     verts skip (they never moved; vel stays 0). fxdiv(x, 0) == 0 by the fpx contract, and
    //     fxmul(v, kOne) == v EXACTLY, so damp == kOne is the exact identity.
    for (size_t i = 0; i < n; ++i) {
        HairVert& v = verts[i];
        if (v.flags & kFlagPinned) continue;
        v.vel.x = fxmul(fxdiv(v.pos.x - v.prev.x, p.dt), p.damp);
        v.vel.y = fxmul(fxdiv(v.pos.y - v.prev.y, p.dt), p.damp);
        v.vel.z = fxmul(fxdiv(v.pos.z - v.prev.z, p.dt), p.damp);
    }
}

// StepHairSteps: run K full strand steps (the showcase / test K-step driver).
inline void StepHairSteps(const HairStrands& hs, std::vector<HairVert>& verts,
                          const HairConstraints& hc, const ClothAdjacency& excl,
                          const std::vector<fx>& kBendPerStrand, const HairParams& p, int steps) {
    for (int s = 0; s < steps; ++s)
        StepHair(hs, verts, hc, excl, kBendPerStrand, p);
}

// ----- The lockstep command stream (the CL5/FPX5 mold): ROOT DRAG -----------------------------------
// kCmdRootMove drags strand `strand`'s ROOT (vert 0) by `arg` — pos += arg AND prev = pos, so the drag
// is a positional teleport with NO velocity kick (the root is pinned; its vel stays 0 and the strand
// follows through the constraints next steps). Out-of-range strand is a deterministic no-op. Commands
// at the same tick apply in ARRAY ORDER (the deterministic input-order contract).
inline constexpr uint32_t kCmdRootMove = 0u;

struct HairCommand {
    uint32_t tick   = 0;   // the tick this input applies on
    uint32_t kind   = 0;   // kCmdRootMove
    uint32_t strand = 0;   // the target STRAND index (the command moves its root vert)
    FxVec3   arg;          // the Q16.16 root displacement
};

inline void ApplyHairCommand(const HairStrands& hs, std::vector<HairVert>& verts,
                             const HairCommand& c) {
    if (c.strand >= (uint32_t)hs.S) return;                       // bounds-checked no-op
    const size_t root = (size_t)VertIndex(hs, (int)c.strand, 0);
    if (root >= verts.size()) return;                             // bounds-checked no-op
    if (c.kind == kCmdRootMove) {
        HairVert& r = verts[root];
        r.pos = FxAdd(r.pos, c.arg);
        r.prev = r.pos;                                           // no velocity kick (a pinned drag)
    }
}

// SimHairTick: apply this tick's commands (array order) then StepHair once — the fpx SimTick twin.
inline void SimHairTick(const HairStrands& hs, std::vector<HairVert>& verts,
                        const HairConstraints& hc, const ClothAdjacency& excl,
                        const std::vector<fx>& kBendPerStrand, const HairParams& p,
                        const std::vector<HairCommand>& stream, uint32_t tick) {
    for (const HairCommand& c : stream)
        if (c.tick == tick) ApplyHairCommand(hs, verts, c);
    StepHair(hs, verts, hc, excl, kBendPerStrand, p);
}

// RunHairLockstep: THE peer entry point — `ticks` SimHairTicks from a COPY of `init` fed the command
// stream ALONE (inputs, not state). authority == replica BIT-EXACT by determinism (the lockstep proof
// memcmps them). The RunClothLockstep twin.
inline std::vector<HairVert> RunHairLockstep(const HairStrands& hs,
                                             const std::vector<HairVert>& init,
                                             const HairConstraints& hc, const ClothAdjacency& excl,
                                             const std::vector<fx>& kBendPerStrand,
                                             const HairParams& p,
                                             const std::vector<HairCommand>& stream, int ticks) {
    std::vector<HairVert> verts = init;
    for (int t = 0; t < ticks; ++t)
        SimHairTick(hs, verts, hc, excl, kBendPerStrand, p, stream, (uint32_t)t);
    return verts;
}

// RunHairRollback: the rollback harness (the RunClothRollback twin). Advance 0..mispredictTick with
// the authoritative stream, SNAPSHOT (the deep vector copy), speculatively advance a few ticks with
// the MISPREDICTED stream (the diverging client prediction), then RESTORE + re-simulate
// mispredictTick..ticks with the correct stream. The proof asserts the result == RunHairLockstep(init,
// authStream, ticks) AND that the full mispredicted run DIFFERED (a real divergence was corrected).
inline std::vector<HairVert> RunHairRollback(const HairStrands& hs,
                                             const std::vector<HairVert>& init,
                                             const HairConstraints& hc, const ClothAdjacency& excl,
                                             const std::vector<fx>& kBendPerStrand,
                                             const HairParams& p,
                                             const std::vector<HairCommand>& authStream,
                                             const std::vector<HairCommand>& mispredictStream,
                                             int ticks, int mispredictTick) {
    std::vector<HairVert> verts = init;
    for (int t = 0; t < mispredictTick; ++t)
        SimHairTick(hs, verts, hc, excl, kBendPerStrand, p, authStream, (uint32_t)t);
    const std::vector<HairVert> snap = verts;                     // SNAPSHOT (deep copy)
    int specTicks = ticks - mispredictTick;
    if (specTicks > 3) specTicks = 3;                             // bounded speculation (the CL5 shape)
    for (int s = 0; s < specTicks; ++s)
        SimHairTick(hs, verts, hc, excl, kBendPerStrand, p, mispredictStream,
                    (uint32_t)(mispredictTick + s));
    verts = snap;                                                 // ROLLBACK (restore)
    for (int t = mispredictTick; t < ticks; ++t)
        SimHairTick(hs, verts, hc, excl, kBendPerStrand, p, authStream, (uint32_t)t);
    return verts;
}

// ----- Deterministic diagnostics (test/showcase helpers, pure integer) ------------------------------

// MaxStretchError: max |segment length - restLen| over the stretch chain, Q16.16 LSBs — the honest PBD
// stretch residual (iterative Gauss-Seidel leaves a deterministic, NON-zero error; the tests PIN it).
inline fx MaxStretchError(const std::vector<HairVert>& verts, const HairConstraints& hc) {
    fx worst = 0;
    for (const Constraint& c : hc.stretch) {
        if (c.i >= verts.size() || c.j >= verts.size()) continue;
        const fx len = FxLength(FxSub(verts[(size_t)c.j].pos, verts[(size_t)c.i].pos));
        fx err = len - c.restLen;
        if (err < 0) err = -err;
        if (err > worst) worst = err;
    }
    return worst;
}

// MinFreePairDistance: the minimum FxLength over all NON-constraint-connected pairs with AT LEAST ONE
// dynamic vert (both-pinned pairs are excluded — pinned roots may be AUTHORED closer than the collision
// thickness and the solver correctly never moves them, the wsum==0 skip; this metric reports what the
// solver is accountable for). Brute-force O(n^2) diagnostic so it does NOT share the grid broadphase
// with the solver (the CL7 discipline — a broadphase bug shows up here). No such pair -> INT32_MAX.
inline fx MinFreePairDistance(const std::vector<HairVert>& verts, const ClothAdjacency& excl) {
    const uint32_t n = (uint32_t)verts.size();
    fx best = INT32_MAX;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i + 1u; j < n; ++j) {
            if ((verts[i].flags & kFlagPinned) && (verts[j].flags & kFlagPinned)) continue;
            if (cloth::IsConstraintConnected(excl, i, j)) continue;
            const fx len = FxLength(FxSub(verts[(size_t)i].pos, verts[(size_t)j].pos));
            if (len < best) best = len;
        }
    return best;
}

// CountFreePenetrating: the number of NON-connected, not-both-pinned pairs with distance <
// thickness - slack (the CL7 CountSelfPenetrating twin with the both-pinned exclusion; slack is the
// honest PBD residual tolerance — pass 0 for the raw count).
inline int CountFreePenetrating(const std::vector<HairVert>& verts, const ClothAdjacency& excl,
                                fx thickness, fx slack) {
    const uint32_t n = (uint32_t)verts.size();
    const fx bar = thickness - slack;
    int pen = 0;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i + 1u; j < n; ++j) {
            if ((verts[i].flags & kFlagPinned) && (verts[j].flags & kFlagPinned)) continue;
            if (cloth::IsConstraintConnected(excl, i, j)) continue;
            if (FxLength(FxSub(verts[(size_t)i].pos, verts[(size_t)j].pos)) < bar) ++pen;
        }
    return pen;
}

// HairDigest: the FNV-1a-64 digest of the full vert state (HairVert == ClothParticle -> the CL7
// ClothDigest reused verbatim; field-wise, layout/padding-independent — the golden currency).
inline uint64_t HairDigest(const std::vector<HairVert>& verts) {
    return cloth::ClothDigest(verts);
}

}  // namespace hair
}  // namespace hf::sim
