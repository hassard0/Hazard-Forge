#pragma once
// Slice FL1 — Deterministic GPU Fluid: Q16.16 fixed-point PARTICLE POOL INTEGRATOR + dam-break block
// (the BEACHHEAD of FLAGSHIP #9: DETERMINISTIC GPU FLUID via Position-Based Fluids — a PBF density-
// constraint fluid solver that is BIT-IDENTICAL CPU<->Vulkan<->Metal AND frame/run reproducible, unlike
// UE5's float/non-deterministic Niagara). FL1 is ONLY the integer particle POOL + gravity integrate +
// ground floor-clamp — the integrator beachhead. NO neighbors (FL2), NO density/lambda kernel (FL3),
// NO PBF density-constraint solve (FL4), NO lockstep (FL5), NO float render (FL6), NO float on the
// bit-exact path. Pure CPU, header-only, NO device, NO backend symbols, NO <cmath>. Namespace
// hf::sim::fluid. PBF is the DENSITY-CONSTRAINT TWIN of the shipped cloth PBD solver — the whole flagship
// reuses the proven engine/sim/fpx.h Q16.16 toolbox + the cloth CL1-CL6 mold; this completes the
// deterministic-sim trilogy (rigid FPX -> cloth -> fluid). The STRUCTURAL TWIN of the cloth CL1 integer
// beachhead (engine/sim/cloth.h): a pure-integer per-particle update proven GPU==CPU BIT-EXACT, with a
// cross-backend BIT-IDENTICAL integer golden — the SAME shape over a 3D particle BLOCK instead of a
// flat 2D sheet.
//
// The integrator shader shaders/fluid_integrate.comp.hlsl copies IntegrateFluid's per-particle math
// VERBATIM (the fxmul + integrate + prev-snap + floor-clamp), so tests/fluid_test.cpp + the GPU pass
// exercise the EXACT math — which is what makes the integrated particle array bit-identical GPU==CPU AND
// cross-backend.
//
// THE CROSS-BACKEND CRUX (the make-or-break for GPU==CPU, like cloth.h's host-snapped ClothParticle
// array): the GPU consumes host-snapped Q16.16 INTEGERS (the FluidParticle array) and does ZERO floating
// point — every step is `(int64)a*b >> kFrac` (an ARITHMETIC right shift on int64, deterministic +
// identical on every compiler/vendor) + integer add + integer compare. In FL1 each particle is
// INDEPENDENT (no inter-particle neighbours/density/constraints until FL2-FL4), so the GPU per-particle
// write is trivially order-independent / race-free with NO atomics, and two runs are byte-identical.
//
// THE int32-vs-int64 DECISION (the CL1/FPX1 lesson, documented): the integrate is `vel += gravity*dt;
// pos += vel*dt`, both componentwise fxmul — the SAME form as cloth_integrate.comp / fpx_integrate.comp,
// which needed int64 because the (int64)a*b product before the >>kFrac shift exceeds int32 for Q16.16
// gravity*dt (gravity ≈ -9.8*65536 = -642253; products of two Q16.16 world-scale values blow past 2^31).
// To stay bit-exact to this int64-intermediate reference WITHOUT any overflow fragility,
// shaders/fluid_integrate.comp.hlsl uses int64 (like cloth_integrate.comp) and is therefore VULKAN-SPIR-V-
// ONLY (glslc — the Metal HLSL->SPIR-V->MSL frontend — cannot parse int64_t in HLSL), NOT in the Metal
// hf_gen_msl list; the Metal --fluid-integrate showcase runs the CPU fluid::IntegrateFluid (the SAME
// bit-exact reference the Vulkan GPU==CPU memcmp compares against) -> byte-identical to the Vulkan GPU
// result BY CONSTRUCTION. Same established convention as cloth_integrate.comp / fpx_integrate.comp.
//
// REUSE MAP (file:line): the Q16.16 toolbox is engine/sim/fpx.h — fx (int32 Q16.16, fpx.h:46), fxmul
// (fpx.h:54, the int64-intermediate multiply), FxVec3 + FxAdd/FxSub/FxScale (fpx.h:59-72), kOne/kFrac
// (fpx.h:47-48). The per-particle integrate mirrors fpx.h::IntegrateBody (fpx.h:149, the semi-implicit
// Euler + ground floor-clamp) + cloth.h::IntegrateParticle (cloth.h:134, plus the prev=pos snap) — READ,
// NOT modified (fluid is the additive sibling; fpx.h #included read-only + stays byte-unchanged).

#include <cstdint>
#include <vector>

#include "sim/fpx.h"   // read-only: fx / fxmul / FxVec3 / FxAdd / FxSub / FxScale / kOne / kFrac

namespace hf::sim {
namespace fluid {

// Reuse the fpx Q16.16 scalar + vector toolbox verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::fxmul;
using fpx::FxAdd;
using fpx::FxSub;
using fpx::FxScale;
inline constexpr int kFrac = fpx::kFrac;     // Q16.16 fractional bits (== fpx::kFrac, MUST match the shader)
inline constexpr fx  kOne  = fpx::kOne;      // 1.0 in Q16.16 (65536)

// ----- The fluid particle (the std430 GPU mirror; the ClothParticle packing discipline) -------------
// A single fluid particle. std430-packable as plain int32s (pos.xyz, prev.xyz, vel.xyz, invMass, flags)
// = 11 x 4-byte = 44 bytes, NO padding holes (memcmp-able; the GPU FluidParticle mirror). IDENTICAL
// layout to cloth.h::ClothParticle (the proven beachhead packing).
//   * pos  : current Q16.16 world position.
//   * prev : the PREVIOUS position (prev = pos BEFORE the position integrate each step) — FL3/FL4's PBF
//     density/constraint pass needs it (Verlet/predicted-position anchor); FL1 maintains it so the buffer
//     layout is final from the beachhead.
//   * vel  : Q16.16 velocity (world units / second).
//   * invMass : Q16.16 inverse mass (0 => infinite mass / STATIC, never integrates). Carried for FL4.
//   * flags : bit0 = STATIC (a future boundary/wall particle — invMass 0, never moves). Reserved in FL1.
struct FluidParticle {
    FxVec3   pos;             // Q16.16 current position
    FxVec3   prev;            // Q16.16 previous position (prev = pos before the position integrate)
    FxVec3   vel;             // Q16.16 velocity (world units / second)
    fx       invMass = 0;     // Q16.16 inverse mass (0 => static / infinite mass)
    uint32_t flags   = 0;     // bit0 = STATIC (reserved for FL4 boundary particles)
};

inline constexpr uint32_t kFlagStatic = 1u;   // bit0: the particle is a fixed boundary (never integrates)

// ----- The dam-break block config: a fixed W x H x D lattice of fluid particles in a corner ----------
// The classic dam-break initial condition: a solid W x H x D block of fluid particles at a spacing, with
// its corner at `origin` (above the ground). The float layout constants are host-snapped to Q16.16 once
// at build (NOT in the per-step integer sim). FL1 picks a modest count (10x10x10 = 1000) — well under the
// FL4 TDR budget so the SAME scene survives the later slices.
struct FluidBlock {
    int    W = 0, H = 0, D = 0;   // block dims (x columns, y rows, z layers)
    fx     spacing = 0;           // Q16.16 spacing between adjacent particles
    FxVec3 origin;                // Q16.16 world position of particle (0,0,0) — the block's corner
};

// Index of particle (ix, iy, iz) in the x-major / y-mid / z-minor particle array (the deterministic,
// fixed traversal order the shader's one-thread-per-particle dispatch + the CPU reference share).
inline int ParticleIndex(const FluidBlock& block, int ix, int iy, int iz) {
    return (iz * block.H + iy) * block.W + ix;
}

// ----- InitBlock: the deterministic W x H x D dam-break block (all dynamic, at rest) -----------------
// Builds a solid block of fluid particles: particle (ix,iy,iz) at origin + (ix*spacing, iy*spacing,
// iz*spacing). All particles start at rest (vel 0, prev == pos), DYNAMIC (invMass = kOne, flags 0) — FL1
// has no boundary/static particles yet (that is FL4). The block sits in a corner ABOVE the ground so it
// FALLS and piles at the ground (the dam-break fall). Pure integer (the spacing/origin are already
// host-snapped Q16.16; this only adds integer multiples of spacing). Returns the populated particle
// vector (size W*H*D), in the ParticleIndex traversal order.
inline std::vector<FluidParticle> InitBlock(const FluidBlock& block) {
    std::vector<FluidParticle> particles((size_t)(block.W * block.H * block.D));
    for (int iz = 0; iz < block.D; ++iz) {
        for (int iy = 0; iy < block.H; ++iy) {
            for (int ix = 0; ix < block.W; ++ix) {
                FluidParticle p;
                // Host-snapped layout: integer multiples of the (already Q16.16) spacing -> exact, no float.
                p.pos = FxVec3{block.origin.x + (fx)(ix * (int)block.spacing),
                               block.origin.y + (fx)(iy * (int)block.spacing),
                               block.origin.z + (fx)(iz * (int)block.spacing)};
                p.prev    = p.pos;
                p.vel     = FxVec3{0, 0, 0};
                p.invMass = kOne;   // unit mass dynamic (FL1 has no static/boundary particles)
                p.flags   = 0;
                particles[(size_t)ParticleIndex(block, ix, iy, iz)] = p;
            }
        }
    }
    return particles;
}

// ----- IntegrateFluidParticle: the deterministic per-particle semi-implicit-Euler step (SHADER math) --
// For a single particle, if NOT static (!(flags & kFlagStatic)):
//   vel += gravity * dt   (component-wise fxmul)        — the FPX1/CL1 velocity integrate
//   prev = pos                                          — snapshot before the position move (for FL3/FL4)
//   pos += vel * dt        (component-wise fxmul)        — the FPX1/CL1 position integrate
//   single non-penetration FLOOR clamp: if (pos.y < groundY) { pos.y = groundY; if (vel.y < 0) vel.y = 0; }
// Static particles are UNTOUCHED. Pure integer; fixed op order; no RNG, no clock. Each particle is
// INDEPENDENT of every other (no inter-particle coupling in FL1), so the order over particles does NOT
// matter -> two-run bit-identical AND the GPU per-thread write is race-free with NO atomics. The shader
// runs THIS exact per-particle body. Copied verbatim from cloth.h::IntegrateParticle (= fpx.h::IntegrateBody
// plus the prev=pos snap).
inline void IntegrateFluidParticle(FluidParticle& p, const FxVec3& gravity, fx groundY, fx dt) {
    if (p.flags & kFlagStatic) return;
    // (1) integrate velocity: vel += gravity * dt.
    p.vel.x += fxmul(gravity.x, dt);
    p.vel.y += fxmul(gravity.y, dt);
    p.vel.z += fxmul(gravity.z, dt);
    // (2) snapshot the previous position (the PBF predicted-position anchor FL3/FL4 reads).
    p.prev = p.pos;
    // (3) integrate position: pos += vel * dt.
    p.pos.x += fxmul(p.vel.x, dt);
    p.pos.y += fxmul(p.vel.y, dt);
    p.pos.z += fxmul(p.vel.z, dt);
    // (4) ground floor clamp (no restitution — FL1 is free-fall + ground only).
    if (p.pos.y < groundY) {
        p.pos.y = groundY;
        if (p.vel.y < 0) p.vel.y = 0;
    }
}

// ----- IntegrateFluid: one integrate STEP over the whole particle pool ------------------------------
// Apply IntegrateFluidParticle to every particle once. The make-or-break reference the GPU memcmp's
// against. Order-independent (particles are independent in FL1) -> bit-identical regardless of GPU
// scheduling.
inline void IntegrateFluid(std::vector<FluidParticle>& particles, const FxVec3& gravity, fx dt,
                           fx groundY) {
    const size_t n = particles.size();
    for (size_t i = 0; i < n; ++i)
        IntegrateFluidParticle(particles[i], gravity, groundY, dt);
}

// ----- IntegrateFluidSteps: run K integrate steps (the showcase / GPU K-step driver) ----------------
// K successive IntegrateFluid steps over the pool. The GPU fluid_integrate.comp runs THIS exact K-step
// loop per thread (one thread per particle, since particles are independent in FL1).
inline void IntegrateFluidSteps(std::vector<FluidParticle>& particles, const FxVec3& gravity, fx dt,
                                fx groundY, int steps) {
    for (int s = 0; s < steps; ++s)
        IntegrateFluid(particles, gravity, dt, groundY);
}

// CountAtGround(particles, groundY): the deterministic count of particles resting at the ground floor
// (pos.y == groundY) — a reporting/stat helper for the showcase coverage proof. Pure integer compare ->
// bit-exact CPU<->GPU.
inline int CountAtGround(const std::vector<FluidParticle>& particles, fx groundY) {
    int n = 0;
    for (const FluidParticle& p : particles)
        if (p.pos.y == groundY) ++n;
    return n;
}

}  // namespace fluid
}  // namespace hf::sim
