#pragma once
// Slice CL1 — Deterministic GPU Cloth: Q16.16 fixed-point PARTICLE LATTICE INTEGRATOR + grid build
// (the BEACHHEAD of FLAGSHIP #8: DETERMINISTIC GPU CLOTH — a position-based-dynamics cloth solver that
// is BIT-IDENTICAL CPU<->Vulkan<->Metal AND frame/run reproducible, unlike UE5's float Chaos Cloth /
// NvCloth). CL1 is ONLY the integer particle lattice + gravity integrate + pin + ground floor-clamp —
// the integrator beachhead. NO constraints (CL2/CL3), NO collision (CL4), NO render (CL6), NO float on
// the bit-exact path. Pure CPU, header-only, NO device, NO backend symbols, NO <cmath>. Namespace
// hf::sim::cloth. The STRUCTURAL TWIN of the FPX1 integer beachhead (engine/sim/fpx.h): a pure-integer
// per-particle update proven GPU==CPU BIT-EXACT, with a cross-backend BIT-IDENTICAL integer golden.
//
// The integrator shader shaders/cloth_integrate.comp.hlsl copies IntegrateParticles's per-particle math
// VERBATIM (the fxmul + integrate + floor-clamp), so tests/cloth_test.cpp + the GPU pass exercise the
// EXACT math — which is what makes the integrated particle array bit-identical GPU==CPU AND cross-backend.
//
// THE CROSS-BACKEND CRUX (the make-or-break for GPU==CPU, like fpx.h's host-snapped FxBody array): the
// GPU consumes host-snapped Q16.16 INTEGERS (the ClothParticle array) and does ZERO floating point —
// every step is `(int64)a*b >> kFrac` (an ARITHMETIC right shift on int64, deterministic + identical on
// every compiler/vendor) + integer add + integer compare. In CL1 each particle is INDEPENDENT (no
// inter-particle constraints until CL2/CL3), so the GPU per-particle write is trivially order-
// independent / race-free with NO atomics, and two runs are byte-identical.
//
// THE int32-vs-int64 DECISION (the FPX1 lesson, documented): the integrate is `vel += gravity*dt;
// pos += vel*dt`, both componentwise fxmul — the SAME form as fpx_integrate.comp, which needed int64
// because the (int64)a*b product before the >>kFrac shift exceeds int32 for Q16.16 gravity*dt (gravity
// ≈ -9.8*65536 = -642253; products of two Q16.16 world-scale values blow past 2^31). To stay bit-exact
// to this int64-intermediate reference WITHOUT any overflow fragility, shaders/cloth_integrate.comp.hlsl
// uses int64 (like fpx_integrate.comp) and is therefore VULKAN-SPIR-V-ONLY (glslc — the Metal
// HLSL->SPIR-V->MSL frontend — cannot parse int64_t in HLSL), NOT in the Metal hf_gen_msl list; the
// Metal --cloth-integrate showcase runs the CPU IntegrateParticles (the SAME bit-exact reference the
// Vulkan GPU==CPU memcmp compares against) -> byte-identical to the Vulkan GPU result BY CONSTRUCTION.
// Same established convention as fpx_integrate.comp / fpx_solve.comp / swraster.comp.
//
// REUSE MAP (file:line): the Q16.16 toolbox is engine/sim/fpx.h — fx (int32 Q16.16, fpx.h:46), fxmul
// (fpx.h:54, the int64-intermediate multiply), FxVec3 + FxAdd/FxSub/FxScale (fpx.h:59-72), kOne/kFrac
// (fpx.h:47-48). The per-particle integrate mirrors fpx.h::IntegrateBody (fpx.h:149, the semi-implicit
// Euler + ground floor-clamp) — READ, NOT modified (cloth is the additive sibling; fpx.h #included
// read-only + stays byte-unchanged).

#include <cmath>       // CL6 render-only: std::fma / std::sqrt (the float normal math; NOT on the bit-exact sim path)
#include <cstdint>
#include <vector>

#include "sim/fpx.h"   // read-only: fx / fxmul / FxVec3 / FxAdd / FxSub / FxScale / kOne / kFrac
#include "math/math.h" // CL6 render-only: math::Vec3 (ClothVertToWorld output; NOT on the bit-exact sim path)

namespace hf::sim {
namespace cloth {

// Reuse the fpx Q16.16 scalar + vector toolbox verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::fxmul;
using fpx::FxAdd;
using fpx::FxSub;
using fpx::FxScale;
inline constexpr int kFrac = fpx::kFrac;     // Q16.16 fractional bits (== fpx::kFrac, MUST match the shader)
inline constexpr fx  kOne  = fpx::kOne;      // 1.0 in Q16.16 (65536)

// ----- The cloth particle (the std430 GPU mirror; the FxBody packing discipline) ------------------
// A single lattice point. std430-packable as plain int32s (pos.xyz, prev.xyz, vel.xyz, invMass, flags)
// = 11 x 4-byte = 44 bytes, NO padding holes (memcmp-able; the GPU ClothParticle mirror).
//   * pos  : current Q16.16 world position.
//   * prev : the PREVIOUS position (prev = pos BEFORE the position integrate each step) — CL2/CL3's PBD
//     verlet/constraint pass needs it; CL1 maintains it so the buffer layout is final from the beachhead.
//   * vel  : Q16.16 velocity (world units / second).
//   * invMass : Q16.16 inverse mass (0 => infinite mass / PINNED, never integrates). Carried for CL3.
//   * flags : bit0 = PINNED (the particle is fixed — invMass 0, never moves).
struct ClothParticle {
    FxVec3   pos;             // Q16.16 current position
    FxVec3   prev;            // Q16.16 previous position (prev = pos before the position integrate)
    FxVec3   vel;             // Q16.16 velocity (world units / second)
    fx       invMass = 0;     // Q16.16 inverse mass (0 => pinned / infinite mass)
    uint32_t flags   = 0;     // bit0 = PINNED
};

inline constexpr uint32_t kFlagPinned = 1u;   // bit0: the particle is fixed (never integrates)

// ----- The cloth grid: a fixed W x H row-major lattice -------------------------------------------
// A flat W x H sheet of particles, index r*W + c (r in [0,H), c in [0,W)). spacing is the Q16.16
// distance between adjacent particles; origin is the Q16.16 world position of particle (r=0,c=0). The
// float layout constants are host-snapped to Q16.16 once at build (NOT in the per-step integer sim).
struct ClothGrid {
    int    W = 0, H = 0;      // lattice dims (columns x rows)
    fx     spacing = 0;       // Q16.16 spacing between adjacent particles
    FxVec3 origin;            // Q16.16 world position of particle (r=0, c=0)
};

// Index of particle (row r, col c) in the row-major particle array.
inline int ParticleIndex(const ClothGrid& grid, int r, int c) { return r * grid.W + c; }

// ----- InitGrid: the deterministic flat W x H sheet with the TWO top corners pinned --------------
// Builds a flat sheet lying in the XY plane (y increases DOWNWARD per row so the sheet hangs from its
// top edge under gravity): particle (r,c) at origin + (c*spacing, -r*spacing, 0). All particles start
// at rest (vel 0, prev == pos). The two TOP corners (row 0: col 0 and col W-1) are PINNED (invMass 0,
// flags |= kFlagPinned); every other particle is dynamic with invMass = kOne (unit mass). Pure integer
// (the spacing/origin are already host-snapped Q16.16; this only adds integer multiples of spacing).
// Returns the populated particle vector (size W*H).
inline std::vector<ClothParticle> InitGrid(const ClothGrid& grid) {
    std::vector<ClothParticle> particles((size_t)(grid.W * grid.H));
    for (int r = 0; r < grid.H; ++r) {
        for (int c = 0; c < grid.W; ++c) {
            ClothParticle p;
            // Host-snapped layout: integer multiples of the (already Q16.16) spacing -> exact, no float.
            p.pos = FxVec3{grid.origin.x + (fx)(c * (int)grid.spacing),
                           grid.origin.y - (fx)(r * (int)grid.spacing),
                           grid.origin.z};
            p.prev = p.pos;
            p.vel  = FxVec3{0, 0, 0};
            // Pin the two TOP corners (row 0, the first + last column).
            const bool pinned = (r == 0) && (c == 0 || c == grid.W - 1);
            if (pinned) {
                p.invMass = 0;
                p.flags   = kFlagPinned;
            } else {
                p.invMass = kOne;
                p.flags   = 0;
            }
            particles[(size_t)ParticleIndex(grid, r, c)] = p;
        }
    }
    return particles;
}

// ----- IntegrateParticle: the deterministic per-particle semi-implicit-Euler step (the SHADER math) --
// For a single particle, if NOT pinned (!(flags & kFlagPinned)):
//   vel += gravity * dt   (component-wise fxmul)        — the FPX1 velocity integrate
//   prev = pos                                          — snapshot before the position move (for CL2/CL3)
//   pos += vel * dt        (component-wise fxmul)        — the FPX1 position integrate
//   single non-penetration FLOOR clamp: if (pos.y < groundY) { pos.y = groundY; if (vel.y < 0) vel.y = 0; }
// Pinned particles are UNTOUCHED. Pure integer; fixed op order; no RNG, no clock. Each particle is
// INDEPENDENT of every other (no inter-particle coupling in CL1), so the order over particles does NOT
// matter -> two-run bit-identical AND the GPU per-thread write is race-free with NO atomics. The shader
// runs THIS exact per-particle body. Copied verbatim from fpx.h::IntegrateBody, plus the prev=pos snap.
inline void IntegrateParticle(ClothParticle& p, const FxVec3& gravity, fx groundY, fx dt) {
    if (p.flags & kFlagPinned) return;
    // (1) integrate velocity: vel += gravity * dt.
    p.vel.x += fxmul(gravity.x, dt);
    p.vel.y += fxmul(gravity.y, dt);
    p.vel.z += fxmul(gravity.z, dt);
    // (2) snapshot the previous position (the PBD/verlet anchor CL2/CL3 reads).
    p.prev = p.pos;
    // (3) integrate position: pos += vel * dt.
    p.pos.x += fxmul(p.vel.x, dt);
    p.pos.y += fxmul(p.vel.y, dt);
    p.pos.z += fxmul(p.vel.z, dt);
    // (4) ground floor clamp (no restitution — CL1 is free-fall + pin + ground only).
    if (p.pos.y < groundY) {
        p.pos.y = groundY;
        if (p.vel.y < 0) p.vel.y = 0;
    }
}

// ----- IntegrateParticles: one integrate STEP over the whole lattice ------------------------------
// Apply IntegrateParticle to every particle once. The make-or-break reference the GPU memcmp's against.
// Order-independent (particles are independent in CL1) -> bit-identical regardless of GPU scheduling.
inline void IntegrateParticles(const ClothGrid& grid, std::vector<ClothParticle>& particles,
                               const FxVec3& gravity, fx dt, fx groundY) {
    (void)grid;   // grid dims are implicit in particles.size(); kept for a symmetric/extensible signature
    const size_t n = particles.size();
    for (size_t i = 0; i < n; ++i)
        IntegrateParticle(particles[i], gravity, groundY, dt);
}

// ----- IntegrateParticlesSteps: run K integrate steps (the showcase / GPU K-step driver) ----------
// K successive IntegrateParticles steps over the lattice. The GPU cloth_integrate.comp runs THIS exact
// K-step loop per thread (one thread per particle, since particles are independent in CL1).
inline void IntegrateParticlesSteps(const ClothGrid& grid, std::vector<ClothParticle>& particles,
                                    const FxVec3& gravity, fx dt, fx groundY, int steps) {
    for (int s = 0; s < steps; ++s)
        IntegrateParticles(grid, particles, gravity, dt, groundY);
}

// CountPinned(particles): the deterministic count of PINNED particles (a reporting/stat helper).
inline int CountPinned(const std::vector<ClothParticle>& particles) {
    int n = 0;
    for (const ClothParticle& p : particles)
        if (p.flags & kFlagPinned) ++n;
    return n;
}

// ===== Slice CL2 — Deterministic GPU Cloth: the DISTANCE-CONSTRAINT GRAPH BUILD ======================
// Enumerate the cloth's distance constraints over the W x H lattice — the graph the CL3 PBD solver will
// project. Three kinds of edge (the standard mass-spring cloth topology): STRUCTURAL (each particle to
// its RIGHT + DOWN neighbour — the woven warp/weft), SHEAR (the two DIAGONALS of each cell — resist
// in-plane shear), BEND (each particle to its 2-away RIGHT + DOWN neighbour — resist out-of-plane
// folding, modelled here as a plain distance constraint, the documented simplification vs a dihedral
// model). Each unordered edge is OWNED by its LOWER linear index so it is enumerated EXACTLY ONCE.
//
// THE int32-NATIVE DECISION (the FPX2 twin, documented): the count->scan->emit compaction is pure int32
// index/compare arithmetic over the lattice (per-particle owned-edge COUNT -> single-thread exclusive
// prefix-SUM -> per-particle EMIT at offset) — so the cloth_edge_{count,scan,emit} shaders MSL-generate
// NATIVELY on Metal and run as TRUE GPU passes on both backends (unlike CL1's int64 cloth_integrate).
// The ONLY length is restLen = FxLength(pos[j]-pos[i]); FxLength internally is int64 (fpx.h::FxISqrt over
// the int64 sum-of-squares), but the lattice is HOST-SNAPPED and FLAT at build, so restLen is a build-
// time CONSTANT PER KIND (structural = spacing; shear = FxLength(spacing,spacing,0); bend = 2*spacing).
// We compute these three int32 restLens host-side once (the only int64 use, on the CPU) and the SHADER
// reads them from params per edge kind — so the edge buffer + the three shaders are PURE int32 (NO int64,
// NO sqrt in-shader), exactly the NAV1 TriYSpan / FPX2 rest-data precedent.

inline constexpr uint32_t kConstraintStructural = 0u;  // right/down neighbour (warp/weft)
inline constexpr uint32_t kConstraintShear      = 1u;  // cell diagonal (in-plane shear)
inline constexpr uint32_t kConstraintBend       = 2u;  // 2-away right/down (out-of-plane bend)

// A single distance constraint (i<j canonical). std430-packable as 4 x int32 (16 bytes): i, j, restLen,
// kind — NO padding holes (memcmp-able; the GPU Constraint mirror). restLen is the Q16.16 rest distance.
struct Constraint {
    uint32_t i = 0, j = 0;   // the two endpoint particle indices (i<j by construction)
    fx       restLen = 0;    // Q16.16 rest length = FxLength(pos[j]-pos[i]) at build (host-snapped int32)
    uint32_t kind = 0;       // kConstraintStructural / kConstraintShear / kConstraintBend
};

// CountOwnedEdges(grid, r, c): the number of edges OWNED by particle (r,c) — i.e. edges whose LOWER
// linear index is r*W+c. Owner = the neighbour with the LARGER index in each pair, so each particle owns
// exactly the FORWARD edges that stay in-bounds, in the FIXED per-particle order [right, down, diag1
// (down-right), diag2 (down-left), bend-right, bend-down]. diag2 (down-left) is owned by (r,c) because
// (r,c) = r*W+c < (r+1)*W+(c-1) for c>=1 — its lower index. Pure int32 (the count shader copies this).
inline int CountOwnedEdges(const ClothGrid& grid, int r, int c) {
    const int W = grid.W, H = grid.H;
    int n = 0;
    if (c + 1 < W) ++n;                       // STRUCTURAL right  (r,c)-(r,c+1)
    if (r + 1 < H) ++n;                       // STRUCTURAL down   (r,c)-(r+1,c)
    if (r + 1 < H && c + 1 < W) ++n;          // SHEAR diag1 down-right (r,c)-(r+1,c+1)
    if (r + 1 < H && c - 1 >= 0) ++n;         // SHEAR diag2 down-left  (r,c)-(r+1,c-1)
    if (c + 2 < W) ++n;                       // BEND right (r,c)-(r,c+2)
    if (r + 2 < H) ++n;                       // BEND down  (r,c)-(r+2,c)
    return n;
}

// BuildConstraints(grid, particles): the CPU reference constraint graph (the make-or-break the GPU
// memcmp's against). Mirrors the count->scan->emit the GPU runs: (1) per-particle owned-edge count;
// (2) exclusive prefix-sum -> per-particle write offset; (3) per-particle emit at its DISJOINT offset
// in the FIXED order [right, down, diag1, diag2, bend-right, bend-down]. restLen = FxLength(pos[j]-pos[i])
// at build (host-snapped). The result is grouped by OWNER index (ascending) then the fixed per-particle
// order -> fully deterministic, the exact ascending-owner list the GPU emit produces byte-for-byte.
inline std::vector<Constraint> BuildConstraints(const ClothGrid& grid,
                                                const std::vector<ClothParticle>& particles) {
    const int W = grid.W, H = grid.H;
    const int n = W * H;

    // (1) per-particle owned-edge count.
    std::vector<int> counts((size_t)n, 0);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            counts[(size_t)(r * W + c)] = CountOwnedEdges(grid, r, c);

    // (2) exclusive prefix-sum -> per-particle write offset (the single-thread serial scan).
    std::vector<int> offset((size_t)n, 0);
    int running = 0;
    for (int p = 0; p < n; ++p) { offset[(size_t)p] = running; running += counts[(size_t)p]; }
    const int total = running;

    // (3) per-particle emit each owned edge at offset[p] + local++, in the FIXED order. restLen is
    // FxLength of the host-snapped endpoint delta at build.
    std::vector<Constraint> out((size_t)total);
    auto emit = [&](int& base, int& local, int i, int j, uint32_t kind) {
        const FxVec3 d = fpx::FxSub(particles[(size_t)j].pos, particles[(size_t)i].pos);
        out[(size_t)(base + local)] =
            Constraint{(uint32_t)i, (uint32_t)j, fpx::FxLength(d), kind};
        ++local;
    };
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            const int p = r * W + c;
            int base = offset[(size_t)p], local = 0;
            if (c + 1 < W)              emit(base, local, p, r * W + (c + 1),       kConstraintStructural);
            if (r + 1 < H)              emit(base, local, p, (r + 1) * W + c,       kConstraintStructural);
            if (r + 1 < H && c + 1 < W) emit(base, local, p, (r + 1) * W + (c + 1), kConstraintShear);
            if (r + 1 < H && c - 1 >= 0)emit(base, local, p, (r + 1) * W + (c - 1), kConstraintShear);
            if (c + 2 < W)              emit(base, local, p, r * W + (c + 2),       kConstraintBend);
            if (r + 2 < H)              emit(base, local, p, (r + 2) * W + c,       kConstraintBend);
        }
    return out;
}

// CountConstraintsByKind(constraints, outStruct, outShear, outBend): the deterministic per-kind tally
// (a reporting/stat helper for the showcase proof line). Pure integer.
inline void CountConstraintsByKind(const std::vector<Constraint>& constraints,
                                   int& outStruct, int& outShear, int& outBend) {
    outStruct = outShear = outBend = 0;
    for (const Constraint& e : constraints) {
        if (e.kind == kConstraintStructural) ++outStruct;
        else if (e.kind == kConstraintShear) ++outShear;
        else if (e.kind == kConstraintBend)  ++outBend;
    }
}

// ===== Slice CL3 — Deterministic GPU Cloth: the PBD DISTANCE-CONSTRAINT SOLVER (the make-or-break) =====
// Project the CL2 distance constraints with Position-Based Dynamics so the cloth holds its shape + DRAPES
// (the pinned corners hold; the lattice stays cohesive instead of CL1's free-fall scatter). This is the
// VERBATIM generalization of fpx.h::ResolvePair (fpx.h:340 — the sphere-pair positional projection) from a
// sphere PAIR to a constraint EDGE: where ResolvePair has rest distance (a.radius + b.radius) and resolves
// only OVERLAP (pen = restDist - dist > 0 -> push apart), a DISTANCE constraint has rest distance c.restLen
// and resolves BOTH directions (pen = dist - restLen; pen>0 stretched -> pull together, pen<0 compressed ->
// push apart). The inverse-mass split (wi = fxdiv(invMass_i, wsum)), the FxNormalize(d) (via FxISqrt, NO
// std::sqrt), and the FxScale(n, fxmul(pen, w)) correction are otherwise IDENTICAL to ResolvePair — pinned
// particles (invMass==0) take share 0 and never move. Pure integer, int64-backed Q16.16 (fxdiv/FxISqrt) —
// shaders/cloth_solve.comp copies this VERBATIM so the GPU exercises the EXACT ops (Vulkan-only int64; the
// Metal showcase runs THIS CPU StepCloth -> byte-identical by construction, the fpx_solve.comp convention).
//
// HONEST CAVEAT (FPX3-identical): PBD is ITERATIVE — after `iters` Gauss-Seidel passes the residual
// constraint error is DETERMINISTIC but NOT zero (stiffness ∝ iterations); fxdiv/FxISqrt truncation makes
// the solver bit-REPRODUCIBLE, not analytically exact. The headline is DETERMINISM + cross-platform
// bit-identity (the UE5 Chaos differentiator), NOT "more physically correct".

// Bring fpx's PBD primitives into the cloth namespace (read-only): the integer divide, the integer
// normalize (via FxISqrt — NO std::sqrt), and FxLength. NO new fixed-point primitives.
using fpx::fxdiv;
using fpx::FxNormalize;
using fpx::FxLength;

// ----- SolveDistanceConstraint: project ONE distance constraint (the bit-exact core) -----------------
// Given the particle array + a single Constraint c (endpoints c.i, c.j; rest distance c.restLen):
//   d   = pos[j] - pos[i]
//   len = FxLength(d); if len == 0 -> skip (degenerate; no deterministic direction)
//   pen = len - c.restLen      (>0 stretched, <0 compressed — a distance constraint resolves BOTH)
//   n   = FxNormalize(d)
//   wsum = invMass_i + invMass_j; if wsum == 0 -> skip (both pinned)
//   wi = fxdiv(invMass_i, wsum); wj = fxdiv(invMass_j, wsum)
//   pos[i] += n * fxmul(pen, wi)   (the i endpoint moves toward j when stretched)
//   pos[j] -= n * fxmul(pen, wj)
// Pinned (invMass 0) -> its share is 0 -> never moves. This is fpx.h::ResolvePair with rest length
// c.restLen instead of (a.radius+b.radius), resolving BOTH pen signs (NOT only overlap). int64-backed
// (fxdiv/FxISqrt) — the shader copies this body VERBATIM.
inline void SolveDistanceConstraint(std::vector<ClothParticle>& particles, const Constraint& c) {
    ClothParticle& pi = particles[(size_t)c.i];
    ClothParticle& pj = particles[(size_t)c.j];
    const fx wsum = pi.invMass + pj.invMass;
    if (wsum == 0) return;                              // both pinned -> skip
    const FxVec3 d = FxSub(pj.pos, pi.pos);
    const fx len = FxLength(d);
    if (len == 0) return;                               // coincident -> no deterministic normal -> skip
    const fx pen = len - c.restLen;
    const FxVec3 n = FxNormalize(d);
    const fx wi = fxdiv(pi.invMass, wsum);
    const fx wj = fxdiv(pj.invMass, wsum);
    const FxVec3 ci = FxScale(n, fxmul(pen, wi));       // i moves +ci (toward j when stretched)
    const FxVec3 cj = FxScale(n, fxmul(pen, wj));       // j moves -cj
    pi.pos = FxAdd(pi.pos, ci);
    pj.pos = FxSub(pj.pos, cj);
}

// ----- StepCloth: one full PBD step (integrate + K Gauss-Seidel constraint passes + floor clamp) ------
// The make-or-break reference the GPU cloth_solve.comp memcmp's against (copied VERBATIM into the shader):
//   (1) IntegrateParticles(grid, particles, gravity, dt, groundY)  — CL1, one semi-implicit-Euler step.
//   (2) `iters` Gauss-Seidel passes, EACH iterating ALL constraints in the FIXED CL2 emit order applying
//       SolveDistanceConstraint. SEQUENTIAL — each constraint reads the positions already updated by the
//       earlier constraints THIS pass -> order-dependent -> single-thread on the GPU (the fpx SolveContacts
//       discipline). Pinned particles (invMass 0) never move.
//   (3) ground floor-clamp: every particle with pos.y < groundY snaps pos.y = groundY (the non-penetration
//       floor; the constraint passes can push a particle below ground, so clamp AFTER them).
// The pinned corners hold + the constraints keep the lattice cohesive -> the sheet DRAPES (unlike CL1's
// free-fall). Pure integer, fixed op order, no RNG/clock -> two-run bit-identical AND bit-exact GPU==CPU.
inline void StepCloth(const ClothGrid& grid, std::vector<ClothParticle>& particles,
                      const std::vector<Constraint>& constraints,
                      const FxVec3& gravity, fx dt, fx groundY, int iters) {
    // (1) integrate one step (CL1).
    IntegrateParticles(grid, particles, gravity, dt, groundY);
    // (2) `iters` Gauss-Seidel constraint passes in the FIXED CL2 emit order (sequential -> order-dependent).
    for (int it = 0; it < iters; ++it)
        for (size_t e = 0; e < constraints.size(); ++e)
            SolveDistanceConstraint(particles, constraints[e]);
    // (3) ground floor clamp AFTER the constraint passes (a constraint may have pushed a particle below).
    const size_t n = particles.size();
    for (size_t i = 0; i < n; ++i)
        if (particles[i].pos.y < groundY) particles[i].pos.y = groundY;
}

// ----- StepClothSteps: run K full PBD steps (the showcase / GPU K-step driver) -------------------------
// K successive StepCloth steps over the lattice. The GPU cloth_solve.comp runs THIS exact K-step loop on a
// SINGLE thread (the Gauss-Seidel order-dependence makes single-thread necessary for bit-exactness).
inline void StepClothSteps(const ClothGrid& grid, std::vector<ClothParticle>& particles,
                           const std::vector<Constraint>& constraints,
                           const FxVec3& gravity, fx dt, fx groundY, int iters, int steps) {
    for (int s = 0; s < steps; ++s)
        StepCloth(grid, particles, constraints, gravity, dt, groundY, iters);
}

// ----- EdgeResidual: a deterministic integer cohesion metric (summed |edge len - restLen|) -------------
// The summed absolute distance-constraint error over all edges, in Q16.16 (int64 accumulator -> no
// overflow over thousands of edges). DETERMINISTIC + bit-exact CPU<->GPU (pure integer FxLength compares).
// A coherent DRAPE keeps this small (the constraints held the lattice); free-fall scatter would blow it up.
inline int64_t EdgeResidual(const std::vector<ClothParticle>& particles,
                            const std::vector<Constraint>& constraints) {
    int64_t r = 0;
    for (const Constraint& c : constraints) {
        const FxVec3 d = FxSub(particles[(size_t)c.j].pos, particles[(size_t)c.i].pos);
        const fx len = FxLength(d);
        const fx err = len - c.restLen;
        r += (err < 0) ? -(int64_t)err : (int64_t)err;
    }
    return r;
}

// ===== Slice CL4 — Deterministic GPU Cloth: INTEGER COLLISION (cloth-vs-FPX rigid sphere + ground) ====
// Project the CL3-solved cloth particles OUT of a small STATIC set of rigid SPHERE colliders + the ground
// plane, so the cloth DRAPES over a rigid sphere. This is the FIRST deformable-meets-rigid INTEGER
// interaction: the cloth particle and the rigid sphere are the SAME Q16.16 world units (a SphereCollider
// reuses fpx::FxBody's pos + radius), so the projection is exactly fpx.h::ResolvePair's sphere-vs-point
// push generalized to a particle vs a STATIC sphere (the particle takes the FULL correction; the sphere
// never moves). Per particle it is INDEPENDENT (each particle vs the read-only collider set), order-
// independent; but the normalize uses int64 (FxNormalize via FxISqrt) so the GPU shader is Vulkan-only +
// Metal runs THIS CPU reference (the CL3 convention). Applied each step AFTER the CL3 constraint passes
// (so the cloth holds together AND stays out of the sphere). Pure integer, copied VERBATIM into
// shaders/cloth_collide.comp.
//
// HONEST CAVEAT (CL3-identical): a single positional projection per step is NOT a full contact solve —
// transient interpenetration can occur and is resolved over steps (deterministic). The headline is
// DETERMINISM + cross-platform bit-identity + a visible drape, NOT a physically perfect contact model.
// Self-collision (cloth-vs-cloth) + DYNAMIC (FPX-moving) colliders + friction are OUT of scope (later
// CL slices). CL4's colliders are a small STATIC set.

// A static SPHERE collider: a Q16.16 world-space center + radius (the SAME units as ClothParticle.pos).
// std430-packable as 4 x int32 (center.xyz, radius) — the GPU SphereCollider mirror. Reuses fpx::FxBody's
// pos+radius semantics; a free helper FromBody builds one from an fpx::FxBody (the cloth-vs-FPX seam).
struct SphereCollider {
    FxVec3 center;        // Q16.16 world-space sphere center (== fpx::FxBody::pos)
    fx     radius = 0;    // Q16.16 sphere radius (== fpx::FxBody::radius)
};

// SphereFromBody(b): build a cloth SphereCollider from an fpx::FxBody (its pos + radius). The composition
// seam — the cloth drapes over the SAME rigid sphere the FPX sim integrates (same Q16.16 world units).
inline SphereCollider SphereFromBody(const fpx::FxBody& b) {
    return SphereCollider{b.pos, b.radius};
}

// ----- CollideParticleSphere: project ONE particle out of ONE sphere (the bit-exact core) -------------
// If the particle is PINNED -> untouched. Else: an int32 AABB reject first (skip if |p-center| per-axis >
// radius — the common no-overlap case, NO int64); then d = p.pos - s.center; dist = FxLength(d); if
// dist < s.radius the particle is INSIDE -> push it to the surface along the OUTWARD normal:
// n = FxNormalize(d) (dist==0 -> FxNormalize's fixed +Y fallback {0,kOne,0}); p.pos = s.center +
// FxScale(n, s.radius) (snap to the surface). The static sphere takes no correction (the particle gets
// the full push — the ResolvePair "one body static" case). Pure integer, int64-backed (FxLength/
// FxNormalize). Returns true iff this particle was projected (a CONTACT — for the coverage stat).
inline bool CollideParticleSphere(ClothParticle& p, const SphereCollider& s) {
    if (p.flags & kFlagPinned) return false;
    // int32 AABB reject: if the particle is outside the sphere's AABB on ANY axis, it cannot be inside
    // (NO int64 — the cheap common-case skip before the FxLength). |p-center| per-axis compared to radius.
    const fx dx = p.pos.x - s.center.x;
    const fx dy = p.pos.y - s.center.y;
    const fx dz = p.pos.z - s.center.z;
    const fx ax = dx < 0 ? -dx : dx;
    const fx ay = dy < 0 ? -dy : dy;
    const fx az = dz < 0 ? -dz : dz;
    if (ax > s.radius || ay > s.radius || az > s.radius) return false;   // outside the AABB -> no overlap
    const FxVec3 d = FxVec3{dx, dy, dz};
    const fx dist = FxLength(d);
    if (dist >= s.radius) return false;                                  // outside the sphere -> untouched
    const FxVec3 n = FxNormalize(d);                                     // dist==0 -> {0,kOne,0} fallback
    p.pos = FxAdd(s.center, FxScale(n, s.radius));                       // snap to the surface
    return true;
}

// ----- CollideSpheres: project a particle array out of a STATIC sphere set -----------------------------
// For each particle (in index order), for each sphere (in the FIXED collider order), apply
// CollideParticleSphere. Deterministic (fixed sphere order, fixed per-particle order). Returns the total
// number of (particle, sphere) projections this call (the CONTACT count — a coverage stat, deterministic
// + bit-exact CPU<->GPU). Pinned particles never move. The shader copies THIS double loop VERBATIM.
inline int CollideSpheres(std::vector<ClothParticle>& particles,
                          const std::vector<SphereCollider>& spheres) {
    int contacts = 0;
    const size_t n = particles.size();
    for (size_t i = 0; i < n; ++i)
        for (size_t s = 0; s < spheres.size(); ++s)
            if (CollideParticleSphere(particles[i], spheres[s])) ++contacts;
    return contacts;
}

// ----- CollidePlane: clamp every particle to a ground plane (pos.y >= groundY) -------------------------
// The CL3 floor-clamp, factored out as the plane collider (pinned particles ARE clamped too — a pinned
// corner above the floor is unaffected; the clamp only raises a particle that fell below). Pure integer.
inline void CollidePlane(std::vector<ClothParticle>& particles, fx groundY) {
    const size_t n = particles.size();
    for (size_t i = 0; i < n; ++i)
        if (particles[i].pos.y < groundY) particles[i].pos.y = groundY;
}

// ----- StepClothCollide: one full PBD step WITH collision (CL3 StepCloth + CollideSpheres) --------------
// IDENTICAL to CL3's StepCloth (integrate -> `iters` Gauss-Seidel constraint passes -> ground floor-clamp)
// EXCEPT it applies CollideSpheres AFTER the constraint passes + the floor clamp (so the cloth both holds
// together AND stays out of the spheres). With an EMPTY sphere set this is BYTE-IDENTICAL to StepCloth
// (CollideSpheres is a no-op + the floor clamp is the same) — the CL3 zero-collider equivalence. Returns
// the contact count for this step. The GPU cloth_collide.comp runs THIS exact body per step on one thread.
inline int StepClothCollide(const ClothGrid& grid, std::vector<ClothParticle>& particles,
                            const std::vector<Constraint>& constraints,
                            const std::vector<SphereCollider>& spheres,
                            const FxVec3& gravity, fx dt, fx groundY, int iters) {
    // (1)-(3) the CL3 step VERBATIM (integrate + constraint passes + ground floor-clamp).
    StepCloth(grid, particles, constraints, gravity, dt, groundY, iters);
    // (4) collision: clamp to the ground plane (already clamped by StepCloth; CollidePlane is idempotent)
    //     THEN project out of every static sphere. Deterministic, pinned never move.
    CollidePlane(particles, groundY);
    return CollideSpheres(particles, spheres);
}

// ----- StepClothCollideSteps: run K full PBD+collision steps (the showcase / GPU K-step driver) --------
// K successive StepClothCollide steps. The GPU cloth_collide.comp runs THIS exact K-step loop on a SINGLE
// thread (the Gauss-Seidel order-dependence in the inner constraint solve makes single-thread necessary).
// Returns the contact count of the FINAL step (the settled-drape contact stat).
inline int StepClothCollideSteps(const ClothGrid& grid, std::vector<ClothParticle>& particles,
                                 const std::vector<Constraint>& constraints,
                                 const std::vector<SphereCollider>& spheres,
                                 const FxVec3& gravity, fx dt, fx groundY, int iters, int steps) {
    int contacts = 0;
    for (int s = 0; s < steps; ++s)
        contacts = StepClothCollide(grid, particles, constraints, spheres, gravity, dt, groundY, iters);
    return contacts;
}

// kCollideEps: the fixed-point penetration tolerance (Q16.16 LSBs). CollideParticleSphere snaps a particle
// to the surface via FxNormalize + FxScale, both of which TRUNCATE toward zero — so the snapped FxLength
// lands a few LSBs SHORT of radius (the FPX3 "residual is deterministic, not analytically zero" reality).
// A particle within this slack is "on the surface", NOT penetrating. Small (a handful of LSBs); the snap
// error is bounded by the per-axis truncation. Pure integer; bit-exact CPU<->GPU.
inline constexpr fx kCollideEps = 16;   // ~16 Q16.16 LSBs (a tiny fraction of a world unit)

// CountPenetrating(particles, spheres): the deterministic count of particles still INSIDE any sphere
// beyond the snap tolerance (dist < radius - kCollideEps) — a coverage/diagnostic helper (0 after
// StepClothCollide projects them all to the surface). Pure integer FxLength compares -> bit-exact CPU<->GPU.
inline int CountPenetrating(const std::vector<ClothParticle>& particles,
                            const std::vector<SphereCollider>& spheres) {
    int pen = 0;
    for (const ClothParticle& p : particles) {
        for (const SphereCollider& s : spheres) {
            const FxVec3 d = FxSub(p.pos, s.center);
            if (FxLength(d) < s.radius - kCollideEps) { ++pen; break; }
        }
    }
    return pen;
}

// ===== Slice CL5 — Deterministic GPU Cloth: LOCKSTEP + ROLLBACK netcode harness (the HEADLINE) =========
// Prove the bit-exact CL1-CL4 cloth is true cross-platform LOCKSTEP + ROLLBACK: a peer fed the INPUT
// command stream ALONE (NOT full state) re-derives the authority's exact cloth state bit-for-bit, and a
// mispredicted input is corrected by rolling back to a saved snapshot + re-simulating the authoritative
// stream. PURE CPU, 0 backend symbols, NO new shader / RHI: this is a determinism PROPERTY of the existing
// bit-exact StepClothCollide (CL4) — the cross-backend zero-diff golden (the converged state is bit-
// identical on Vulkan-Windows AND Metal-Mac from the same inputs) IS the cross-platform-lockstep evidence.
// This is the DIRECT TWIN of fpx.h's FxCommand/ApplyCommand/SimTick/SnapshotWorld/RestoreWorld/RunLockstep/
// RunRollback (~fpx.h:531-601) — the SAME shape over ClothParticle/StepClothCollide. NO <cmath>, NO RNG,
// NO clock. This extends the FPX5 lockstep/rollback headline from RIGID to DEFORMABLE bodies — a
// deterministic, rollback-able, bit-identical-cross-platform cloth UE5's float Chaos Cloth cannot provide.
//
// A ClothCommand is the deterministic per-tick INPUT a netcode layer would put on the wire (NOT full
// state). kCmdWind adds `arg` (a wind delta-velocity) to a particle's velocity (an integer add, skipping
// pinned particles — they hold); kCmdPin/kCmdUnpin toggle a particle's PINNED flag (pin: invMass->0,
// set kFlagPinned + zero its velocity so the pinned point is at rest; unpin: clear kFlagPinned, invMass->
// kOne so it becomes dynamic). A std::vector<ClothCommand> is the command STREAM, processed in ARRAY ORDER
// for each tick (the deterministic-order contract — the same order on every peer/platform), so authority +
// replica fed the same stream re-derive the same state exactly.

inline constexpr uint32_t kCmdWind  = 0u;   // arg added to target particle's velocity (a wind impulse)
inline constexpr uint32_t kCmdPin   = 1u;   // target particle becomes PINNED (invMass 0, vel zeroed)
inline constexpr uint32_t kCmdUnpin = 2u;   // target particle becomes DYNAMIC (kFlagPinned cleared, invMass kOne)

struct ClothCommand {
    uint32_t tick   = 0;   // the tick this input applies on
    uint32_t kind   = 0;   // kCmdWind / kCmdPin / kCmdUnpin
    uint32_t target = 0;   // the target particle index
    FxVec3   arg;          // the Q16.16 payload (wind delta-velocity for kCmdWind; ignored for pin/unpin)
};

// ApplyClothCommand(particles, c): apply ONE input command to the cloth (pure integer — add to vel / set
// the pin flag + invMass). Out-of-range target is a no-op (deterministic). Unknown kind is a no-op. The
// input event the lockstep/rollback streams are made of. The fpx.h::ApplyCommand twin.
inline void ApplyClothCommand(std::vector<ClothParticle>& particles, const ClothCommand& c) {
    if (c.target >= (uint32_t)particles.size()) return;
    ClothParticle& p = particles[(size_t)c.target];
    if (c.kind == kCmdWind) {
        if (p.flags & kFlagPinned) return;   // pinned points hold — wind does not move them
        p.vel.x += c.arg.x;
        p.vel.y += c.arg.y;
        p.vel.z += c.arg.z;
    } else if (c.kind == kCmdPin) {
        p.flags |= kFlagPinned;
        p.invMass = 0;
        p.vel = FxVec3{0, 0, 0};             // a pinned point is at rest
    } else if (c.kind == kCmdUnpin) {
        p.flags &= ~kFlagPinned;
        p.invMass = kOne;                    // unit mass dynamic again
    }
}

// SimClothTick(grid, particles, constraints, spheres, stream, tick, gravity, dt, groundY, iters): the
// deterministic per-tick step. (1) apply ALL commands in `stream` whose .tick == `tick`, in ARRAY ORDER
// (the deterministic input-order contract); (2) StepClothCollide one step (CL4 — integrate + K constraint
// passes + ground clamp + sphere collision). Pure integer, fixed order -> bit-identical on every peer/
// platform. The fpx.h::SimTick twin (StepClothCollide replaces StepWorld+IntegrateOrientation).
inline void SimClothTick(const ClothGrid& grid, std::vector<ClothParticle>& particles,
                         const std::vector<Constraint>& constraints,
                         const std::vector<SphereCollider>& spheres,
                         const std::vector<ClothCommand>& stream, uint32_t tick,
                         const FxVec3& gravity, fx dt, fx groundY, int iters) {
    for (const ClothCommand& c : stream)
        if (c.tick == tick) ApplyClothCommand(particles, c);
    StepClothCollide(grid, particles, constraints, spheres, gravity, dt, groundY, iters);
}

// SnapshotCloth(particles): a deep copy of the full integer particle array (the rollback primitive — a
// lossless saved tick). (std::vector copy is a deep copy.) The fpx.h::SnapshotWorld twin.
inline std::vector<ClothParticle> SnapshotCloth(const std::vector<ClothParticle>& particles) {
    return particles;   // value copy: deep-copies the particle array
}

// RestoreCloth(particles, snap): restore the cloth to a saved snapshot (the rollback). Bit-exact round-trip
// with SnapshotCloth (RestoreCloth(p, SnapshotCloth(p0)) leaves p == p0 byte-for-byte). The fpx.h::
// RestoreWorld twin.
inline void RestoreCloth(std::vector<ClothParticle>& particles, const std::vector<ClothParticle>& snap) {
    particles = snap;
}

// RunClothLockstep(grid, init, constraints, spheres, stream, ticks, gravity, dt, groundY, iters): THE peer
// entry point. Run `ticks` SimClothTicks from a COPY of `init`, applying the command stream -> the final
// cloth state. authority = RunClothLockstep(...); replica = RunClothLockstep(...) from the SAME init +
// stream (inputs ONLY — no state shared) -> BIT-IDENTICAL by determinism (the lockstep proof memcmps them).
// The fpx.h::RunLockstep twin.
inline std::vector<ClothParticle> RunClothLockstep(const ClothGrid& grid,
                                                   const std::vector<ClothParticle>& init,
                                                   const std::vector<Constraint>& constraints,
                                                   const std::vector<SphereCollider>& spheres,
                                                   const std::vector<ClothCommand>& stream, int ticks,
                                                   const FxVec3& gravity, fx dt, fx groundY, int iters) {
    std::vector<ClothParticle> particles = init;
    for (int t = 0; t < ticks; ++t)
        SimClothTick(grid, particles, constraints, spheres, stream, (uint32_t)t, gravity, dt, groundY, iters);
    return particles;
}

// RunClothRollback(grid, init, constraints, spheres, authStream, mispredictStream, ticks, mispredictTick,
// gravity, dt, groundY, iters): the rollback harness. (1) run ticks 0..mispredictTick from init applying
// authStream, SAVING a snapshot AT mispredictTick (before that tick is simulated); (2) speculatively
// advance a few ticks from the snapshot with the MISPREDICTED stream (the wrong input) — this is the
// client prediction that diverges; (3) "receive" the authoritative input -> RestoreCloth to the snapshot +
// RE-SIMULATE mispredictTick..ticks with the CORRECT authStream -> the final corrected cloth. The proof
// asserts this == RunClothLockstep(init, authStream, ticks) (rollback corrected the misprediction EXACTLY)
// AND that the mispredicted-before-rollback state DIFFERED from the authority (a real divergence was
// fixed). The fpx.h::RunRollback twin.
inline std::vector<ClothParticle> RunClothRollback(const ClothGrid& grid,
                                                   const std::vector<ClothParticle>& init,
                                                   const std::vector<Constraint>& constraints,
                                                   const std::vector<SphereCollider>& spheres,
                                                   const std::vector<ClothCommand>& authStream,
                                                   const std::vector<ClothCommand>& mispredictStream,
                                                   int ticks, int mispredictTick,
                                                   const FxVec3& gravity, fx dt, fx groundY, int iters) {
    std::vector<ClothParticle> particles = init;
    // (1) advance 0..mispredictTick with the authoritative stream.
    for (int t = 0; t < mispredictTick; ++t)
        SimClothTick(grid, particles, constraints, spheres, authStream, (uint32_t)t, gravity, dt, groundY, iters);
    // (2) SAVE the snapshot at mispredictTick (the rollback restore point).
    const std::vector<ClothParticle> snap = SnapshotCloth(particles);
    // (2b) speculatively advance a few ticks with the MISPREDICTED stream (the wrong input) — this is the
    // client prediction that diverges from authority. Bounded to the remaining ticks.
    int specTicks = ticks - mispredictTick;
    if (specTicks > 3) specTicks = 3;
    for (int s = 0; s < specTicks; ++s)
        SimClothTick(grid, particles, constraints, spheres, mispredictStream,
                     (uint32_t)(mispredictTick + s), gravity, dt, groundY, iters);
    // (3) ROLLBACK: restore the snapshot + re-simulate mispredictTick..ticks with the CORRECT authStream.
    RestoreCloth(particles, snap);
    for (int t = mispredictTick; t < ticks; ++t)
        SimClothTick(grid, particles, constraints, spheres, authStream, (uint32_t)t, gravity, dt, groundY, iters);
    return particles;
}

// ===== Slice CL6 — Deterministic GPU Cloth: LIT 3D RENDER helpers (the money-shot, COMPLETES flagship #8) =
// CL1-CL5 above are STRICT integer / bit-exact (the cloth SIM is fixed-point, NO float). CL6 RENDERS the
// bit-exact draped cloth as a lit 3D surface — the ONLY float in the cloth arc, and it is RENDER-ONLY,
// cleanly separated from the sim below this banner. The mesh DERIVES from the bit-exact ClothParticle
// lattice (StepClothCollide, CL4) -> provenance is exact; only the final raster/shade is float (the
// MC6/FPX6/NAV6 "FLOAT visresolve-bar"). NO new RHI, NO new shader — the showcase feeds these float
// vertices into the EXISTING lit-mesh pipeline (lit.vert + lit.frag, scene::MeshVertexLayout).
//
// This is the DIRECT TWIN of render/mc.h's RenderVertex / BuildRenderMesh (the MC5/MC6 render-mesh build):
// ClothVertToWorld = the ONE host float divide (pos / (float)kOne); ClothToRenderMesh = the W x H particle
// lattice -> a lit triangle mesh (two triangles per cell quad, (W-1)(H-1)*2 tris, W*H verts, per-vertex
// normals = the normalized average of the adjacent face normals). These touch ZERO bit-exact sim state
// (they consume a const particle array + grid by value) and add NO backend / RHI symbol (pure host float,
// header-only). The integer sim (CL1-CL5) stays byte-identical.

// ClothVertToWorld(p): the ONE documented host float crossing — a Q16.16 world position -> a float
// math::Vec3 world position (pos / (float)kOne). Render-only; NOT used on the bit-exact sim path.
inline math::Vec3 ClothVertToWorld(const FxVec3& pos) {
    const float inv = 1.0f / (float)kOne;
    return math::Vec3{(float)pos.x * inv, (float)pos.y * inv, (float)pos.z * inv};
}

// A render-ready cloth vertex: world position + smooth per-vertex normal. POD float6 (no padding holes),
// trivially copied into scene::Vertex (pos/color/uv/normal/tangent) by the showcase — the render/mc.h
// RenderVertex twin.
struct ClothRenderVertex {
    float px, py, pz;   // world position = particle pos / kOne (the ONE host float divide)
    float nx, ny, nz;   // smooth per-vertex normal (unit length; averaged from adjacent cell-quad faces)
};

// ClothToRenderMesh(grid, particles, outVerts, outIdx): convert the W x H particle lattice into a lit
// triangle mesh in the engine's vertex format. ONE vertex per particle (world position = ClothVertToWorld,
// per-vertex normal = the normalized average of the adjacent cell-quad FACE normals); TWO triangles per
// cell quad over (W-1)x(H-1) cells, the standard row-major split (a,b,c) + (a,c,d) with
// a=(r,c), b=(r,c+1), c=(r+1,c+1), d=(r+1,c). outVerts.size() == W*H; outIdx.size() == (W-1)(H-1)*2*3.
// A degenerate grid (W<2 || H<2 || empty particles) yields ZERO triangles (the empty no-op). Pure +
// deterministic host float (same particles -> same floats every run); RENDER-ONLY, the sim is untouched.
//
// The face normal of a cell quad is normalize(cross(b-a, d-a)) — the cloth's two top corners are pinned
// and it hangs y-down under gravity, so the lattice winding gives an outward-ish normal; the showcase
// renders the lit-mesh pipeline double-sided-tolerant (the drape's back-faces are simply darker). std::fma
// in the cross/length so a CPU shade mirror could reproduce the exact host floats (the visresolve FP
// discipline, the MC5/MC6 precedent).
inline void ClothToRenderMesh(const ClothGrid& grid, const std::vector<ClothParticle>& particles,
                              std::vector<ClothRenderVertex>& outVerts, std::vector<uint32_t>& outIdx) {
    const int W = grid.W, H = grid.H;
    outVerts.clear();
    outIdx.clear();
    if (W < 1 || H < 1 || (size_t)(W * H) != particles.size() || particles.empty()) return;

    // 1) one vertex per particle at its float world position (the ONE host float divide). Normals start at
    //    zero and accumulate the adjacent face normals below.
    outVerts.assign((size_t)(W * H), ClothRenderVertex{0, 0, 0, 0, 0, 0});
    for (int i = 0; i < W * H; ++i) {
        const math::Vec3 w = ClothVertToWorld(particles[(size_t)i].pos);
        outVerts[(size_t)i].px = w.x; outVerts[(size_t)i].py = w.y; outVerts[(size_t)i].pz = w.z;
    }
    if (W < 2 || H < 2) return;   // a 1-wide/tall lattice has no quads -> no triangles (still valid verts)

    // 2) two triangles per cell quad over (W-1)x(H-1) cells (row-major split), and accumulate each cell's
    //    FACE normal onto its 4 corner vertices (so a shared vertex gets the average of its adjacent faces).
    //    The winding is (a,d,cc) + (a,cc,b) — the FRONT of the rest sheet faces +Z (toward the showcase's
    //    fixed +Z camera), so a single-sided lit pipeline shows the curtain front (NOT its culled back). The
    //    face normal cross(d-a, b-a) therefore points +Z on the rest sheet (and follows the drape elsewhere).
    outIdx.reserve((size_t)((W - 1) * (H - 1) * 6));
    auto addFaceNormal = [&](uint32_t a, uint32_t d, uint32_t b) {
        const ClothRenderVertex& va = outVerts[a];
        const ClothRenderVertex& vd = outVerts[d];
        const ClothRenderVertex& vb = outVerts[b];
        const float e1x = vd.px - va.px, e1y = vd.py - va.py, e1z = vd.pz - va.pz;
        const float e2x = vb.px - va.px, e2y = vb.py - va.py, e2z = vb.pz - va.pz;
        float nx = std::fma(e1y, e2z, -e1z * e2y);
        float ny = std::fma(e1z, e2x, -e1x * e2z);
        float nz = std::fma(e1x, e2y, -e1y * e2x);
        const float len2 = std::fma(nx, nx, std::fma(ny, ny, nz * nz));
        const float invLen = (len2 > 0.0f) ? (1.0f / std::sqrt(len2)) : 0.0f;
        nx *= invLen; ny *= invLen; nz *= invLen;
        return math::Vec3{nx, ny, nz};
    };
    for (int r = 0; r < H - 1; ++r)
        for (int c = 0; c < W - 1; ++c) {
            const uint32_t a = (uint32_t)(r * W + c);
            const uint32_t b = (uint32_t)(r * W + (c + 1));
            const uint32_t cc = (uint32_t)((r + 1) * W + (c + 1));
            const uint32_t d = (uint32_t)((r + 1) * W + c);
            // The cell quad's face normal (the two triangles are coplanar in the rest sheet; for a draped
            // sheet we use the quad's (a->d, a->b) normal — a deterministic per-cell value, +Z on the rest).
            const math::Vec3 fn = addFaceNormal(a, d, b);
            const uint32_t corners[4] = {a, b, cc, d};
            for (int k = 0; k < 4; ++k) {
                outVerts[corners[k]].nx += fn.x;
                outVerts[corners[k]].ny += fn.y;
                outVerts[corners[k]].nz += fn.z;
            }
            // Two triangles: (a,d,cc) + (a,cc,b) — the row-major quad split, wound so the front faces +Z.
            outIdx.push_back(a);  outIdx.push_back(d);  outIdx.push_back(cc);
            outIdx.push_back(a);  outIdx.push_back(cc); outIdx.push_back(b);
        }

    // 3) normalize the accumulated per-vertex normals (the average of the adjacent face normals). A vertex
    //    with no contributing face (impossible for W,H>=2) keeps its zero normal.
    for (size_t i = 0; i < outVerts.size(); ++i) {
        ClothRenderVertex& v = outVerts[i];
        const float len2 = std::fma(v.nx, v.nx, std::fma(v.ny, v.ny, v.nz * v.nz));
        if (len2 > 0.0f) {
            const float invLen = 1.0f / std::sqrt(len2);
            v.nx *= invLen; v.ny *= invLen; v.nz *= invLen;
        }
    }
}

// ===== Slice CL7 — Deterministic GPU Cloth: SELF-COLLISION (the Track-R R3 refinement) ================
// FLAGSHIP #8 shipped with a DOCUMENTED fidelity gap: NO self-collision — a folding cloth passes through
// itself. CL7 closes it with a PBD particle-particle separation pass in pure Q16.16 integer, and the
// headline is real: a DETERMINISTIC SELF-COLLIDING CLOTH, bit-identical CPU<->Vulkan<->Metal AND
// lockstep-replayable, is something no major engine ships (UE5 Chaos Cloth self-collision is float /
// non-deterministic). Three pieces, all APPEND-ONLY below CL6 (CL1-CL6 byte-unchanged):
//   (1) EXCLUSION: verts sharing a CL2 distance constraint (structural/shear/bend) must NOT self-collide
//       (a constrained pair sits at restLen by design — structural neighbours are ALWAYS closer than any
//       useful thickness and colliding them would fight the CL3 solver). A CSR adjacency table built once
//       from the constraint list is the exclusion set. NOTE this "constraint ring" already includes the
//       2-away BEND pairs (bend is a plain distance constraint in CL2), so the exclusion is the full
//       constraint 1-ring which spans lattice distance <= 2 along rows/columns.
//   (2) BROADPHASE: the FL2 grid-hash discipline (engine/sim/fluid.h), cloth-LOCAL twin (cloth.h cannot
//       include fluid.h mid-namespace and FluidParticle != ClothParticle): a bounded dense int32 grid at
//       cell-size == thickness, count->scan->emit cell table, per-vert candidates from the 27-cell stencil
//       in the FIXED (dz,dy,dx ascending, then ascending j) order, box-accepted per axis (|d| < thickness,
//       PURE INT32 — the exact radial cull happens in the solve, the FL2/FL3 split). Candidates are
//       exclusion-filtered AT BUILD so the solve (and the GPU shader) never needs the adjacency table.
//   (3) PROJECTION (the JACOBI discipline, the FL7 twin): pass 1 computes EVERY vert's correction from the
//       OLD positions into a SEPARATE scratch buffer (each vert i gathers over its candidates: pair closer
//       than thickness -> push i away along the pair axis by its inverse-mass share of the penetration —
//       exactly the CL3 SolveDistanceConstraint / fpx ResolvePair split, pinned share 0); pass 2 applies
//       all corrections + the ground clamp. Race-free, MULTI-THREADABLE per vert (one GPU thread per vert,
//       NO single-thread, NO TDR — this pass chips at the documented single-thread TDR ceiling), and the
//       per-vert correction is an INTEGER SUM so it is independent of candidate enumeration order.
//
// COMPOSITION (the FL7 StepFluidVisc mold, documented choice): StepClothSelf = the CL4 StepClothCollide
// VERBATIM (integrate + Gauss-Seidel constraints + ground + spheres) THEN `selfIters` Jacobi self-collision
// iterations against a candidate list built ONCE per step from the post-CL4 positions (the FL4 "neighbour
// list once per step" discipline — candidates are re-tested against CURRENT positions each iteration, the
// broadphase set is per-step). The self pass runs AFTER the constraint projection (the CL4 house order:
// cohesion first, contacts second); the constraints re-absorb the self-push next step.
//
// THE IDENTITY-AT-ZERO DISCIPLINE (every HF lobe's off-switch contract): thickness == 0 (or selfIters == 0)
// -> StepClothSelf early-returns BEFORE any self-collision state is touched -> BIT-IDENTICAL to the
// pre-CL7 StepClothCollide (and RunClothLockstepSelf == RunClothLockstep through the harness).
//
// INTEGER WIDTH + THE METAL SPLIT (the CL3/CL4/FL7 convention, MATCHED): the projection uses FxLength/
// FxNormalize/fxdiv (int64) -> the two GPU passes (cloth_self.comp + cloth_self_apply.comp) are
// VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc — the Metal HLSL frontend — cannot; NOT in the Metal
// hf_gen_msl list); the Metal --cl7-self showcase runs THIS CPU reference (byte-identical to the Vulkan
// GPU result BY CONSTRUCTION — the cloth_solve.comp/cloth_collide.comp convention).
//
// OVERFLOW BOUND (documented like FL7): each candidate contributes < thickness to the per-axis correction
// sum; with showcase-scale vert counts (hundreds) even a fully-degenerate coincident cluster stays
// |corr| < n_candidates * thickness << 2^31 for thickness ~ kOne/2. The accumulation is int32 like the
// FL7 XSPH accumulate.
//
// HONEST CAVEATS: PBD self-collision is ITERATIVE — after `selfIters` Jacobi passes the residual pair
// penetration is DETERMINISTIC but NOT analytically zero (the tests count penetrating pairs with a small
// documented slack and pin the min pair distance honestly); a pair moving more than ~thickness in one step
// can still tunnel (velocities are bounded in the pinned scenes so they don't — a CCD pass is future work);
// candidates are per-STEP (a pair entering thickness mid-selfIters is caught next step, the FL4 discipline).

// Bring the fpx grid primitives into the cloth namespace (read-only, the FL2 usings' twin).
using fpx::FloorDiv;    // deterministic floor-division (correct for negative coords)
using fpx::FxCell;      // the int3 cell coordinate
using fpx::CellId;      // the flat cell linearization

// ----- ClothAdjacency: the CSR constraint-neighbour table (the self-collision EXCLUSION set) ----------
// adj[start[i]..start[i+1]) = every vert sharing a CL2 constraint with i (both directions of every edge),
// grouped by vert in the FIXED constraint-list order (deterministic count->scan->emit, the FL2 shape).
struct ClothAdjacency {
    std::vector<uint32_t> start;   // particleCount+1 exclusive prefix-sum offsets (CSR)
    std::vector<uint32_t> adj;     // constraint-connected neighbour indices grouped by vert
};

// BuildClothAdjacency(particleCount, constraints): count->scan->emit both directions of every constraint
// edge. Out-of-range endpoints are skipped (bounds-checked, deterministic no-op). Built ONCE per cloth
// (the constraint graph is static), like BuildConstraints.
inline ClothAdjacency BuildClothAdjacency(size_t particleCount,
                                          const std::vector<Constraint>& constraints) {
    const uint32_t n = (uint32_t)particleCount;
    ClothAdjacency a;
    // (1) COUNT: each in-range constraint adds one neighbour to BOTH endpoints.
    std::vector<uint32_t> counts((size_t)n, 0u);
    for (const Constraint& c : constraints) {
        if (c.i >= n || c.j >= n) continue;              // bounds-checked skip
        ++counts[c.i];
        ++counts[c.j];
    }
    // (2) SCAN: exclusive prefix-sum -> start (n+1 entries; the last == total).
    a.start.assign((size_t)n + 1u, 0u);
    uint32_t running = 0;
    for (uint32_t i = 0; i < n; ++i) { a.start[i] = running; running += counts[i]; }
    a.start[n] = running;
    // (3) EMIT: scatter each edge's two directions at the per-vert cursors (fixed constraint order).
    a.adj.assign((size_t)running, 0u);
    std::vector<uint32_t> cursor((size_t)n, 0u);
    for (const Constraint& c : constraints) {
        if (c.i >= n || c.j >= n) continue;
        a.adj[a.start[c.i] + cursor[c.i]] = c.j; ++cursor[c.i];
        a.adj[a.start[c.j] + cursor[c.j]] = c.i; ++cursor[c.j];
    }
    return a;
}

// IsConstraintConnected(a, i, j): true iff verts i and j share a CL2 constraint (a linear scan of i's
// CSR slice — the per-vert degree is <= 12 on the lattice, so this is a handful of compares). Bounds-safe.
inline bool IsConstraintConnected(const ClothAdjacency& a, uint32_t i, uint32_t j) {
    if ((size_t)i + 1u >= a.start.size()) return false;
    for (uint32_t s = a.start[i]; s < a.start[i + 1u]; ++s)
        if (a.adj[s] == j) return true;
    return false;
}

// ----- The self-collision broadphase: the FL2 bounded-dense-grid twin over cloth verts ----------------
// Cell size == thickness: any pair with FxLength < thickness is within one cell per axis, so the 27-cell
// stencil is COMPLETE (the FL2 cell-size==h argument). Pure int32 throughout (FloorDiv + index arithmetic).
struct ClothSelfGrid {
    fx     cell = 0;      // Q16.16 cell size (== the self-collision thickness)
    FxCell cellMin;       // the integer cell coord of the grid's (0,0,0) corner
    FxCell gridDim;       // the grid extent in cells per axis
};

// SelfCellOf(p, cell): the integer grid cell of a position (FloorDiv per axis — the fluid::CellOf twin).
inline FxCell SelfCellOf(const FxVec3& p, fx cell) {
    return FxCell{FloorDiv(p.x, cell), FloorDiv(p.y, cell), FloorDiv(p.z, cell)};
}

// kMaxSelfCells: the dense-grid size cap. A pathological extent/thickness ratio would blow the dense cell
// table; past the cap BuildSelfCandidates falls back to the brute-force all-pairs enumeration — the SAME
// accept predicate over the SAME pair set, and the per-vert correction is an integer SUM (order-free), so
// the fallback is BIT-IDENTICAL in effect, just slower (a bounds-safety valve, deterministic either way).
inline constexpr int64_t kMaxSelfCells = (int64_t)1 << 22;   // 4M cells

// ----- ClothSelfList: per-vert self-collision candidates (CSR, exclusion-filtered at build) -----------
// cand[start[i]..start[i+1]) = every vert j != i with |pos_i - pos_j| < thickness PER AXIS (the FL2 box
// accept — over-inclusive; the exact radial cull happens in the solve) and NOT constraint-connected to i.
// BOTH directions are present (j in i's slice AND i in j's slice) — each vert gathers its OWN correction.
struct ClothSelfList {
    std::vector<uint32_t> start;   // particleCount+1 exclusive prefix-sum offsets (CSR)
    std::vector<uint32_t> cand;    // candidate j indices grouped by i (fixed stencil order)
};

// SelfBoxAccept(a, b, t): the PURE INT32 per-axis |d| < t candidate test (the fluid::NeighborAccept twin).
inline bool SelfBoxAccept(const FxVec3& a, const FxVec3& b, fx t) {
    fx dx = a.x - b.x; if (dx < 0) dx = -dx;
    fx dy = a.y - b.y; if (dy < 0) dy = -dy;
    fx dz = a.z - b.z; if (dz < 0) dz = -dz;
    return dx < t && dy < t && dz < t;
}

// BuildSelfCandidates(particles, excl, thickness): the per-step broadphase (the FL2 MakeGrid +
// BuildCellTable + BuildNeighborList discipline, fused). thickness <= 0 or fewer than 2 verts -> an empty
// list (the off-switch). Grid path: bucket verts into the dense grid (count->scan->emit), then per vert
// scan the 27-cell stencil in the FIXED (dz,dy,dx ascending; ascending j within a cell) order, box-accept,
// exclusion-filter. Oversized grid (see kMaxSelfCells) -> brute-force ascending-j fallback (same set).
inline ClothSelfList BuildSelfCandidates(const std::vector<ClothParticle>& particles,
                                         const ClothAdjacency& excl, fx thickness) {
    const uint32_t n = (uint32_t)particles.size();
    ClothSelfList list;
    list.start.assign((size_t)n + 1u, 0u);
    if (thickness <= 0 || n < 2u) return list;           // empty candidate list (self-collision off)

    // Grid bounds over the vert positions (the fluid::MakeGrid twin).
    ClothSelfGrid grid;
    grid.cell = thickness;
    FxCell lo = SelfCellOf(particles[0].pos, thickness), hi = lo;
    for (uint32_t i = 1; i < n; ++i) {
        const FxCell c = SelfCellOf(particles[(size_t)i].pos, thickness);
        if (c.x < lo.x) lo.x = c.x; if (c.x > hi.x) hi.x = c.x;
        if (c.y < lo.y) lo.y = c.y; if (c.y > hi.y) hi.y = c.y;
        if (c.z < lo.z) lo.z = c.z; if (c.z > hi.z) hi.z = c.z;
    }
    grid.cellMin = lo;
    grid.gridDim = FxCell{hi.x - lo.x + 1, hi.y - lo.y + 1, hi.z - lo.z + 1};
    const int64_t cells64 = (int64_t)grid.gridDim.x * grid.gridDim.y * grid.gridDim.z;

    if (cells64 > kMaxSelfCells) {
        // Brute-force fallback (deterministic; the same accept predicate -> the same candidate SET; the
        // solve's per-vert sum is order-independent over the set). O(n^2) — the bounds-safety valve.
        std::vector<uint32_t> counts((size_t)n, 0u);
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < n; ++j) {
                if (j == i) continue;
                if (!SelfBoxAccept(particles[(size_t)i].pos, particles[(size_t)j].pos, thickness)) continue;
                if (IsConstraintConnected(excl, i, j)) continue;
                ++counts[i];
            }
        uint32_t running = 0;
        for (uint32_t i = 0; i < n; ++i) { list.start[i] = running; running += counts[i]; }
        list.start[n] = running;
        list.cand.assign((size_t)running, 0u);
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t local = 0;
            for (uint32_t j = 0; j < n; ++j) {
                if (j == i) continue;
                if (!SelfBoxAccept(particles[(size_t)i].pos, particles[(size_t)j].pos, thickness)) continue;
                if (IsConstraintConnected(excl, i, j)) continue;
                list.cand[list.start[i] + local] = j;
                ++local;
            }
        }
        return list;
    }

    // Dense cell table: count->scan->emit vert indices into cells (the fluid::BuildCellTable twin).
    const uint32_t cells = (uint32_t)cells64;
    auto flatId = [&grid](const FxCell& c) {
        const FxCell local{c.x - grid.cellMin.x, c.y - grid.cellMin.y, c.z - grid.cellMin.z};
        return CellId(local, grid.gridDim);
    };
    std::vector<uint32_t> cellStart((size_t)cells + 1u, 0u);
    {
        std::vector<uint32_t> counts((size_t)cells, 0u);
        for (uint32_t i = 0; i < n; ++i) ++counts[flatId(SelfCellOf(particles[(size_t)i].pos, thickness))];
        uint32_t running = 0;
        for (uint32_t c = 0; c < cells; ++c) { cellStart[c] = running; running += counts[c]; }
        cellStart[cells] = running;
    }
    std::vector<uint32_t> cellVerts((size_t)n, 0u);
    {
        std::vector<uint32_t> cursor((size_t)cells, 0u);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t c = flatId(SelfCellOf(particles[(size_t)i].pos, thickness));
            cellVerts[cellStart[c] + cursor[c]] = i;   // ascending i within a cell (the ascending loop)
            ++cursor[c];
        }
    }

    // Per-vert candidates over the 27-cell stencil (count then emit, the fluid::BuildNeighborList twin).
    auto scanStencil = [&](uint32_t i, uint32_t* emitAt) -> uint32_t {
        const FxCell ci = SelfCellOf(particles[(size_t)i].pos, thickness);
        uint32_t local = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = flatId(nc);
            for (uint32_t s = cellStart[cell]; s < cellStart[cell + 1u]; ++s) {
                const uint32_t j = cellVerts[s];
                if (j == i) continue;                                     // NO self
                if (!SelfBoxAccept(particles[(size_t)i].pos, particles[(size_t)j].pos, thickness)) continue;
                if (IsConstraintConnected(excl, i, j)) continue;          // the exclusion ring
                if (emitAt) emitAt[local] = j;
                ++local;
            }
        }
        return local;
    };
    {
        std::vector<uint32_t> counts((size_t)n, 0u);
        for (uint32_t i = 0; i < n; ++i) counts[i] = scanStencil(i, nullptr);
        uint32_t running = 0;
        for (uint32_t i = 0; i < n; ++i) { list.start[i] = running; running += counts[i]; }
        list.start[n] = running;
        list.cand.assign((size_t)running, 0u);
        for (uint32_t i = 0; i < n; ++i)
            if (list.start[i + 1u] > list.start[i]) scanStencil(i, &list.cand[list.start[i]]);
    }
    return list;
}

// ----- SolveSelfCollision: ONE Jacobi self-collision iteration (the bit-exact core) -------------------
// PASS 1 (gather, from OLD positions): for each vert i (skipping PINNED — share 0, never moves), for each
// candidate j in the FIXED list order: d = pos_i - pos_j; dist = FxLength(d); pair closer than thickness ->
//   pen  = thickness - dist                (> 0, the pair overlap)
//   n    = FxNormalize(d)                  (the push-apart axis, pointing i AWAY from j); dist == 0 ->
//          the deterministic INDEX tie-break (i < j ? +Y : -Y) so a coincident pair separates (the raw
//          FxNormalize fallback would push both the SAME way — documented deviation from CollideParticleSphere)
//   wsum = invMass_i + invMass_j; wsum == 0 -> skip; wi = fxdiv(invMass_i, wsum)   (the CL3/ResolvePair split)
//   corr_i += n * fxmul(pen, wi)           (i accumulates ONLY ITS OWN half; j's half is computed when j
//                                           gathers — the pairwise math is symmetric over the OLD state)
// PASS 2 (apply): every non-pinned vert: pos += corr; then the ground clamp (pos.y >= groundY). Pass 1
// reads positions READ-ONLY and writes only corr[i] -> race-free, MULTI-THREADABLE per vert (the Jacobi
// discipline; the GPU cloth_self.comp / cloth_self_apply.comp copy the two passes VERBATIM). Returns the
// number of (i, candidate) projections gathered (a deterministic coverage stat). int64-backed (FxLength/
// FxNormalize/fxdiv) -> Vulkan-only shaders; Metal runs THIS reference (the CL3/CL4 convention).
inline int SolveSelfCollision(std::vector<ClothParticle>& particles, const ClothSelfList& list,
                              fx thickness, fx groundY) {
    const uint32_t n = (uint32_t)particles.size();
    if (thickness <= 0 || list.start.size() != (size_t)n + 1u) return 0;   // bounds-checked no-op
    std::vector<FxVec3> corr((size_t)n, FxVec3{0, 0, 0});
    int projections = 0;
    // PASS 1: gather every correction from the OLD positions (read-only) into the scratch buffer.
    for (uint32_t i = 0; i < n; ++i) {
        const ClothParticle& pi = particles[(size_t)i];
        if (pi.flags & kFlagPinned) continue;                    // pinned share 0 -> corr stays 0
        FxVec3 acc{0, 0, 0};
        for (uint32_t s = list.start[i]; s < list.start[i + 1u]; ++s) {
            const uint32_t j = list.cand[s];
            if (j >= n) continue;                                // bounds-checked skip
            const ClothParticle& pj = particles[(size_t)j];
            const fx wsum = pi.invMass + pj.invMass;
            if (wsum == 0) continue;                             // both pinned -> skip (mirrors CL3)
            const FxVec3 d = FxSub(pi.pos, pj.pos);
            const fx dist = FxLength(d);
            if (dist >= thickness) continue;                     // the exact radial cull (FL3's role)
            const fx pen = thickness - dist;
            // The push-apart axis: coincident pair -> the deterministic INDEX tie-break (+Y for the lower
            // index, -Y for the higher) so the pair actually separates.
            const FxVec3 axis = (dist == 0)
                ? (i < j ? FxVec3{0, kOne, 0} : FxVec3{0, -kOne, 0})
                : FxNormalize(d);
            const fx wi = fxdiv(pi.invMass, wsum);
            const FxVec3 push = FxScale(axis, fxmul(pen, wi));
            acc = FxAdd(acc, push);
            ++projections;
        }
        corr[(size_t)i] = acc;
    }
    // PASS 2: apply all corrections, then the ground clamp (pinned untouched).
    for (uint32_t i = 0; i < n; ++i) {
        ClothParticle& p = particles[(size_t)i];
        if (p.flags & kFlagPinned) continue;
        p.pos = FxAdd(p.pos, corr[(size_t)i]);
        if (p.pos.y < groundY) p.pos.y = groundY;
    }
    return projections;
}

// ----- StepClothSelf: the composed SELF-COLLIDING step (CL4 step -> per-step candidates -> K Jacobi) ---
// StepClothCollide VERBATIM (the pre-CL7 composed step, UNTOUCHED), then — iff thickness > 0 AND
// selfIters > 0 — build the candidate list ONCE from the post-step positions and run `selfIters`
// SolveSelfCollision Jacobi iterations. thickness == 0 (or selfIters == 0) -> EARLY RETURN before any
// self-collision state is touched -> BIT-IDENTICAL to StepClothCollide (the identity-at-zero). State
// shape unchanged (positions only) -> the CL5 snapshot/lockstep machinery applies VERBATIM. Returns the
// CL4 contact count (the sphere-contact stat, unchanged semantics).
inline int StepClothSelf(const ClothGrid& grid, std::vector<ClothParticle>& particles,
                         const std::vector<Constraint>& constraints, const ClothAdjacency& excl,
                         const std::vector<SphereCollider>& spheres,
                         const FxVec3& gravity, fx dt, fx groundY, int iters,
                         fx thickness, int selfIters) {
    const int contacts = StepClothCollide(grid, particles, constraints, spheres,
                                          gravity, dt, groundY, iters);
    if (thickness <= 0 || selfIters <= 0) return contacts;   // identity-at-zero: EXACT no-op
    const ClothSelfList list = BuildSelfCandidates(particles, excl, thickness);
    for (int k = 0; k < selfIters; ++k)
        SolveSelfCollision(particles, list, thickness, groundY);
    return contacts;
}

// ----- StepClothSelfSteps: run K composed self-colliding steps (the showcase / GPU K-step driver) ------
inline int StepClothSelfSteps(const ClothGrid& grid, std::vector<ClothParticle>& particles,
                              const std::vector<Constraint>& constraints, const ClothAdjacency& excl,
                              const std::vector<SphereCollider>& spheres,
                              const FxVec3& gravity, fx dt, fx groundY, int iters,
                              fx thickness, int selfIters, int steps) {
    int contacts = 0;
    for (int s = 0; s < steps; ++s)
        contacts = StepClothSelf(grid, particles, constraints, excl, spheres,
                                 gravity, dt, groundY, iters, thickness, selfIters);
    return contacts;
}

// ----- SimClothTickSelf / RunClothLockstepSelf: the CL5 harness over the SELF-COLLIDING step -----------
// The CL5 machinery applies VERBATIM (the state shape is unchanged); only the step fn is swapped
// (StepClothCollide -> StepClothSelf), the FL7 SimFluidTickVisc/RunFluidLockstepVisc mold. A peer fed the
// input stream alone re-derives the SELF-COLLIDING cloth bit-for-bit. thickness == 0 -> identical to
// SimClothTick/RunClothLockstep by the identity-at-zero.
inline void SimClothTickSelf(const ClothGrid& grid, std::vector<ClothParticle>& particles,
                             const std::vector<Constraint>& constraints, const ClothAdjacency& excl,
                             const std::vector<SphereCollider>& spheres,
                             const std::vector<ClothCommand>& stream, uint32_t tick,
                             const FxVec3& gravity, fx dt, fx groundY, int iters,
                             fx thickness, int selfIters) {
    for (const ClothCommand& c : stream)
        if (c.tick == tick) ApplyClothCommand(particles, c);
    StepClothSelf(grid, particles, constraints, excl, spheres, gravity, dt, groundY, iters,
                  thickness, selfIters);
}

inline std::vector<ClothParticle> RunClothLockstepSelf(const ClothGrid& grid,
                                                       const std::vector<ClothParticle>& init,
                                                       const std::vector<Constraint>& constraints,
                                                       const ClothAdjacency& excl,
                                                       const std::vector<SphereCollider>& spheres,
                                                       const std::vector<ClothCommand>& stream, int ticks,
                                                       const FxVec3& gravity, fx dt, fx groundY, int iters,
                                                       fx thickness, int selfIters) {
    std::vector<ClothParticle> particles = init;
    for (int t = 0; t < ticks; ++t)
        SimClothTickSelf(grid, particles, constraints, excl, spheres, stream, (uint32_t)t,
                         gravity, dt, groundY, iters, thickness, selfIters);
    return particles;
}

// ----- CountSelfPenetrating: the deterministic self-penetration metric (the CL7 physics proof) --------
// The number of unordered vert pairs (i < j), NOT constraint-connected, with FxLength(pos_i - pos_j) <
// thickness - slack. Deliberately BRUTE-FORCE O(n^2) (a test/diagnostic helper, NOT the sim path) so the
// metric does NOT share the grid broadphase with the solver — a broadphase bug shows up here. `slack` is
// the honest PBD-residual tolerance (the solve is iterative; pairs settle a few LSBs under thickness —
// pass 0 for the raw count, a small slack for the "resolved" count; the tests pin BOTH). Pure integer.
inline int CountSelfPenetrating(const std::vector<ClothParticle>& particles,
                                const ClothAdjacency& excl, fx thickness, fx slack) {
    const uint32_t n = (uint32_t)particles.size();
    const fx bar = thickness - slack;
    int pen = 0;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i + 1u; j < n; ++j) {
            if (IsConstraintConnected(excl, i, j)) continue;
            const FxVec3 d = FxSub(particles[(size_t)i].pos, particles[(size_t)j].pos);
            if (FxLength(d) < bar) ++pen;
        }
    return pen;
}

// MinSelfDistance(particles, excl): the minimum FxLength over all non-constraint-connected unordered
// pairs (the pinned "how separated did CL7 leave the cloth" stat). Brute-force O(n^2) diagnostic like
// CountSelfPenetrating. No such pair (tiny/fully-connected cloth) -> INT32_MAX (deterministic sentinel).
inline fx MinSelfDistance(const std::vector<ClothParticle>& particles, const ClothAdjacency& excl) {
    const uint32_t n = (uint32_t)particles.size();
    fx best = INT32_MAX;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i + 1u; j < n; ++j) {
            if (IsConstraintConnected(excl, i, j)) continue;
            const FxVec3 d = FxSub(particles[(size_t)i].pos, particles[(size_t)j].pos);
            const fx len = FxLength(d);
            if (len < best) best = len;
        }
    return best;
}

// ----- ClothDigest: the deterministic FNV-1a-64 digest of the whole particle state (the pin) -----------
// Hashes every particle's 11 int32 words FIELD-WISE (pos/prev/vel/invMass/flags, little-endian byte order
// by explicit shifts — NO reinterpret_cast, so the digest is layout/padding-independent and identical on
// every compiler/platform). The golden-discipline pin the CL7 tests + showcases print (the fluid.h::
// FluidDigest twin over ClothParticle).
inline uint64_t ClothDigest(const std::vector<ClothParticle>& particles) {
    uint64_t h = 1469598103934665603ull;                    // FNV-1a 64 offset basis
    auto mix = [&h](uint32_t v) {
        for (int b = 0; b < 4; ++b) {
            h ^= (uint64_t)((v >> (b * 8)) & 0xFFu);
            h *= 1099511628211ull;                          // FNV-1a 64 prime
        }
    };
    for (const ClothParticle& p : particles) {
        mix((uint32_t)p.pos.x);  mix((uint32_t)p.pos.y);  mix((uint32_t)p.pos.z);
        mix((uint32_t)p.prev.x); mix((uint32_t)p.prev.y); mix((uint32_t)p.prev.z);
        mix((uint32_t)p.vel.x);  mix((uint32_t)p.vel.y);  mix((uint32_t)p.vel.z);
        mix((uint32_t)p.invMass); mix(p.flags);
    }
    return h;
}

}  // namespace cloth
}  // namespace hf::sim
