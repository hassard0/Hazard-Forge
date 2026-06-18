#pragma once
// Slice GR1 — Deterministic GPU Granular/Sand: Q16.16 fixed-point GRAIN POOL INTEGRATOR + dropped block
// (the BEACHHEAD of FLAGSHIP #10: DETERMINISTIC GPU GRANULAR / SAND via Position-Based granular dynamics —
// the 4th member of the deterministic-sim family (rigid FPX -> cloth -> fluid -> GRAIN), adding the one
// physics the trilogy never modeled: dry friction / shear (angle-of-repose). GR1 is ONLY the integer grain
// POOL + gravity integrate + radius-aware ground rest — the integrator beachhead. NO neighbors (GR2), NO
// frictionless contact solve (GR3), NO Coulomb friction (GR4 — the new physics), NO lockstep (GR5), NO float
// render (GR6), NO float on the bit-exact path. Pure CPU, header-only, NO device, NO backend symbols, NO
// <cmath>. Namespace hf::sim::grain. The whole flagship reuses the proven engine/sim/fpx.h Q16.16 toolbox +
// the FL1-FL6 / CL1-CL6 mold. The STRUCTURAL TWIN of the FL1/CL1 integer beachhead (engine/sim/fluid.h /
// engine/sim/cloth.h): a pure-integer per-particle update proven GPU==CPU BIT-EXACT, with a cross-backend
// BIT-IDENTICAL integer golden — over a 3D grain BLOCK, with a first-class `radius` field.
//
// TWO DELTAS vs the FluidParticle twin the GR1 spec locks (everything else is the IntegrateFluidParticle
// body verbatim): (1) a first-class `radius` field -> 48-byte std430 packing (carried for GR3 contact /
// GR4 friction; GR1 uses it ONLY for the ground rest); (2) a RADIUS-AWARE ground rest — the grain's SURFACE
// rests on the floor (pos.y < groundY + radius -> groundY + radius), not its center.
//
// The integrator shader shaders/grain_integrate.comp.hlsl copies IntegrateGrainParticle's per-particle math
// VERBATIM (the fxmul + integrate + prev-snap + radius-aware floor-clamp), so tests/grain_test.cpp + the GPU
// pass exercise the EXACT math — which is what makes the integrated grain array bit-identical GPU==CPU AND
// cross-backend.
//
// THE CROSS-BACKEND CRUX (the make-or-break for GPU==CPU, like fluid.h's host-snapped FluidParticle array):
// the GPU consumes host-snapped Q16.16 INTEGERS (the GrainParticle array) and does ZERO floating point —
// every step is `(int64)a*b >> kFrac` (an ARITHMETIC right shift on int64, deterministic + identical on
// every compiler/vendor) + integer add + integer compare. In GR1 each grain is INDEPENDENT (no inter-grain
// neighbours/contact/friction until GR2-GR4), so the GPU per-thread write is order-independent / race-free
// with NO atomics, and two runs are byte-identical.
//
// THE int32-vs-int64 DECISION (the FL1/CL1 lesson, documented): the integrate is `vel += gravity*dt;
// pos += vel*dt`, both componentwise fxmul — the SAME form as fluid_integrate.comp / cloth_integrate.comp /
// fpx_integrate.comp, which needed int64 because the (int64)a*b product before the >>kFrac shift exceeds
// int32 for Q16.16 gravity*dt (gravity ≈ -9.8*65536 = -642253; products of two Q16.16 world-scale values
// blow past 2^31). To stay bit-exact to this int64-intermediate reference WITHOUT any overflow fragility,
// shaders/grain_integrate.comp.hlsl uses int64 (like fluid_integrate.comp) and is therefore VULKAN-SPIR-V-
// ONLY (glslc — the Metal HLSL->SPIR-V->MSL frontend — cannot parse int64_t in HLSL), NOT in the Metal
// hf_gen_msl list; the Metal --grain-integrate showcase runs the CPU grain::IntegrateGrains (the SAME
// bit-exact reference the Vulkan GPU==CPU memcmp compares against) -> byte-identical to the Vulkan GPU
// result BY CONSTRUCTION. Same established convention as fluid_integrate.comp (fluid.h:28-37).
//
// REUSE MAP (file:line): the Q16.16 toolbox is engine/sim/fpx.h — fx (int32 Q16.16, fpx.h:46), fxmul
// (fpx.h:54, the int64-intermediate multiply), FxVec3 + FxAdd/FxSub/FxScale (fpx.h:59-72), kOne/kFrac
// (fpx.h:47-48). The per-particle integrate mirrors fluid.h::IntegrateFluidParticle (fluid.h:148-165) with
// the radius-aware ground rest — READ, NOT modified (grain is the additive sibling; fpx.h #included
// read-only + stays byte-unchanged, exactly as fluid.h:52 / cloth.h do).

#include <cstdint>
#include <vector>

#include "sim/fpx.h"   // read-only: fx / fxmul / FxVec3 / FxAdd / FxSub / FxScale / kOne / kFrac

namespace hf::sim {
namespace grain {

// Reuse the fpx Q16.16 scalar + vector toolbox verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::fxmul;
using fpx::FxAdd;
using fpx::FxSub;
using fpx::FxScale;
inline constexpr int kFrac = fpx::kFrac;     // Q16.16 fractional bits (== fpx::kFrac, MUST match the shader)
inline constexpr fx  kOne  = fpx::kOne;      // 1.0 in Q16.16 (65536)

// ----- The grain particle (the std430 GPU mirror; the FluidParticle packing discipline + the radius delta) -
// A single grain. std430-packable as plain int32s (pos.xyz, prev.xyz, vel.xyz, invMass, radius, flags) =
// 12 x 4-byte = 48 bytes, NO padding holes (memcmp-able; the GPU GrainParticle mirror). IDENTICAL to
// fluid.h::FluidParticle (44 bytes) PLUS the first-class `radius` field (the ONE packing delta, +4 bytes ->
// 48). Treats FxVec3 as 3 plain int32s (NOT a 16-byte-aligned vec3), so array stride 48 is a multiple of the
// 4-byte scalar alignment.
//   * pos     : current Q16.16 world position.
//   * prev    : the PREVIOUS position (prev = pos BEFORE the position integrate each step) — GR2-GR4's
//     neighbour/contact pass needs it (Verlet/predicted-position anchor); GR1 maintains it so the buffer
//     layout is FINAL from the beachhead (no struct churn in GR2-GR6).
//   * vel     : Q16.16 velocity (world units / second).
//   * invMass : Q16.16 inverse mass (0 => infinite mass / STATIC, never integrates). Carried for GR3/GR4.
//   * radius  : the Q16.16 grain radius (carried for GR3 contact / GR4 friction; GR1 uses it ONLY for the
//     radius-aware ground rest — the grain's SURFACE rests on the floor).
//   * flags   : bit0 = STATIC (a future boundary/wall grain — invMass 0, never moves). Reserved in GR1.
struct GrainParticle {
    FxVec3   pos;             // Q16.16 current position
    FxVec3   prev;            // Q16.16 previous position (prev = pos before the position integrate)
    FxVec3   vel;             // Q16.16 velocity (world units / second)
    fx       invMass = 0;     // Q16.16 inverse mass (0 => static / infinite mass)
    fx       radius  = 0;     // Q16.16 grain radius (the GR1 ground-rest input; GR3/GR4 contact/friction)
    uint32_t flags   = 0;     // bit0 = STATIC (reserved for GR4 boundary grains)
};

inline constexpr uint32_t kFlagStatic = 1u;   // bit0: the grain is a fixed boundary (never integrates)

// ----- The dropped-block config: a fixed W x H x D lattice of grains above the ground --------------------
// A solid W x H x D block of grains at a spacing, with its corner at `origin` (above the ground), uniform
// `radius`. The float layout constants are host-snapped to Q16.16 once at build (NOT in the per-step integer
// sim). GR1 picks a modest count (10x10x10 = 1000) — well under any budget; keep the SAME scene so it
// survives every later GR slice. spacing >= 2*radius so the initial block is non-overlapping (clean hand-off
// to GR3's contact solve).
struct GrainBlock {
    int    W = 0, H = 0, D = 0;   // block dims (x columns, y rows, z layers)
    fx     spacing = 0;           // Q16.16 spacing between adjacent grains (>= 2*radius -> non-overlapping)
    fx     radius  = 0;           // Q16.16 uniform grain radius
    FxVec3 origin;                // Q16.16 world position of grain (0,0,0) — the block's corner
};

// Index of grain (ix, iy, iz) in the x-major / y-mid / z-minor grain array (the deterministic, fixed
// traversal order the shader's one-thread-per-grain dispatch + the CPU reference share).
inline int GrainIndex(const GrainBlock& block, int ix, int iy, int iz) {
    return (iz * block.H + iy) * block.W + ix;
}

// ----- InitGrainBlock: the deterministic W x H x D dropped block (all dynamic, uniform radius, at rest) ---
// Builds a solid block of grains: grain (ix,iy,iz) at origin + (ix*spacing, iy*spacing, iz*spacing). All
// grains start at rest (vel 0, prev == pos), DYNAMIC (invMass = kOne, flags 0), uniform radius — GR1 has no
// boundary/static grains yet (that is GR4). The block sits in a corner ABOVE the ground so it FALLS and piles
// at the ground. Pure integer (the spacing/origin are already host-snapped Q16.16; this only adds integer
// multiples of spacing). Returns the populated grain vector (size W*H*D), in the GrainIndex traversal order.
inline std::vector<GrainParticle> InitGrainBlock(const GrainBlock& block) {
    std::vector<GrainParticle> grains((size_t)(block.W * block.H * block.D));
    for (int iz = 0; iz < block.D; ++iz) {
        for (int iy = 0; iy < block.H; ++iy) {
            for (int ix = 0; ix < block.W; ++ix) {
                GrainParticle p;
                // Host-snapped layout: integer multiples of the (already Q16.16) spacing -> exact, no float.
                p.pos = FxVec3{block.origin.x + (fx)(ix * (int)block.spacing),
                               block.origin.y + (fx)(iy * (int)block.spacing),
                               block.origin.z + (fx)(iz * (int)block.spacing)};
                p.prev    = p.pos;
                p.vel     = FxVec3{0, 0, 0};
                p.invMass = kOne;          // unit mass dynamic (GR1 has no static/boundary grains)
                p.radius  = block.radius;  // the uniform grain radius
                p.flags   = 0;
                grains[(size_t)GrainIndex(block, ix, iy, iz)] = p;
            }
        }
    }
    return grains;
}

// ----- IntegrateGrainParticle: the deterministic per-grain semi-implicit-Euler step (SHADER math) --------
// For a single grain, if NOT static (!(flags & kFlagStatic)):
//   vel += gravity * dt   (component-wise fxmul)            — the FL1/CL1 velocity integrate
//   prev = pos                                              — snapshot before the position move (GR2-GR4)
//   pos += vel * dt        (component-wise fxmul)            — the FL1/CL1 position integrate
//   RADIUS-AWARE non-penetration ground rest: if (pos.y < groundY + radius) { pos.y = groundY + radius;
//     if (vel.y < 0) vel.y = 0; }   — the grain's SURFACE rests on the floor (groundY + radius is a trivial
//     Q16.16 integer add); sets up GR3's collider projection. radius 0 reduces to the FL1 plain ground clamp.
// Static grains are UNTOUCHED. Pure integer; fixed op order; no RNG, no clock. Each grain is INDEPENDENT of
// every other (no inter-grain coupling in GR1), so the order over grains does NOT matter -> two-run
// bit-identical AND the GPU per-thread write is race-free with NO atomics. The shader runs THIS exact
// per-grain body. The IntegrateFluidParticle (fluid.h:148-165) body verbatim, with the radius-aware clamp.
inline void IntegrateGrainParticle(GrainParticle& p, const FxVec3& gravity, fx groundY, fx dt) {
    if (p.flags & kFlagStatic) return;
    // (1) integrate velocity: vel += gravity * dt.
    p.vel.x += fxmul(gravity.x, dt);
    p.vel.y += fxmul(gravity.y, dt);
    p.vel.z += fxmul(gravity.z, dt);
    // (2) snapshot the previous position (the predicted-position anchor GR2-GR4 reads).
    p.prev = p.pos;
    // (3) integrate position: pos += vel * dt.
    p.pos.x += fxmul(p.vel.x, dt);
    p.pos.y += fxmul(p.vel.y, dt);
    p.pos.z += fxmul(p.vel.z, dt);
    // (4) RADIUS-AWARE ground rest (no restitution — GR1 is free-fall + ground only). The grain's surface
    // rests on the floor: clamp the CENTER to groundY + radius.
    const fx restY = groundY + p.radius;
    if (p.pos.y < restY) {
        p.pos.y = restY;
        if (p.vel.y < 0) p.vel.y = 0;
    }
}

// ----- IntegrateGrains: one integrate STEP over the whole grain pool ------------------------------------
// Apply IntegrateGrainParticle to every grain once. The make-or-break reference the GPU memcmp's against.
// Order-independent (grains are independent in GR1) -> bit-identical regardless of GPU scheduling.
inline void IntegrateGrains(std::vector<GrainParticle>& grains, const FxVec3& gravity, fx dt, fx groundY) {
    const size_t n = grains.size();
    for (size_t i = 0; i < n; ++i)
        IntegrateGrainParticle(grains[i], gravity, groundY, dt);
}

// ----- IntegrateGrainSteps: run K integrate steps (the showcase / GPU K-step driver) --------------------
// K successive IntegrateGrains steps over the pool. The GPU grain_integrate.comp runs THIS exact K-step loop
// per thread (one thread per grain, since grains are independent in GR1).
inline void IntegrateGrainSteps(std::vector<GrainParticle>& grains, const FxVec3& gravity, fx dt, fx groundY,
                                int steps) {
    for (int s = 0; s < steps; ++s)
        IntegrateGrains(grains, gravity, dt, groundY);
}

// CountAtGround(grains, groundY): the deterministic count of grains resting at the ground floor (the grain's
// surface on the floor: pos.y == groundY + radius) — a reporting/stat helper for the showcase coverage proof.
// Pure integer compare -> bit-exact CPU<->GPU.
inline int CountAtGround(const std::vector<GrainParticle>& grains, fx groundY) {
    int n = 0;
    for (const GrainParticle& p : grains)
        if (p.pos.y == groundY + p.radius) ++n;
    return n;
}

}  // namespace grain
}  // namespace hf::sim
