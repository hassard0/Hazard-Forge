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

#include <cstdint>
#include <vector>

#include "sim/fpx.h"   // read-only: fx / fxmul / FxVec3 / FxAdd / FxSub / FxScale / kOne / kFrac

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

}  // namespace cloth
}  // namespace hf::sim
