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

}  // namespace cloth
}  // namespace hf::sim
