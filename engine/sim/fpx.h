#pragma once
// Slice FPX1 — Deterministic Fixed-Point Physics: Q16.16 INTEGRATOR + integer broadphase (the
// BEACHHEAD of FLAGSHIP #6: a DETERMINISTIC FIXED-POINT physics solver — the first physics in the
// engine that is BIT-IDENTICAL CPU<->Vulkan<->Metal AND frame-to-frame / run-to-run reproducible,
// unlike the existing FLOAT engine/physics/ solver held only to "visually identical at rest").
// Pure CPU, header-only, NO device, NO backend symbols, NO <cmath> on the bit-exact path.
// Namespace hf::sim::fpx. The STRUCTURAL TWIN of the VT1/MC1 integer beachhead (render/vt.h,
// render/mc.h): a pure-integer per-item update proven GPU==CPU BIT-EXACT, with a cross-backend
// BIT-IDENTICAL integer golden.
//
// The integrator shader shaders/fpx_integrate.comp.hlsl copies IntegrateStep's per-body math
// VERBATIM (the fxmul + integrate + floor-clamp), so tests/fpx_test.cpp + the GPU pass exercise the
// EXACT math — which is what makes the integrated body array bit-identical GPU==CPU AND cross-backend.
//
// THE CROSS-BACKEND CRUX (the make-or-break for GPU==CPU, like vt.h's host-snap / swraster.h's
// integer ScreenVerts): the GPU consumes host-snapped Q16.16 INTEGERS (the FxBody array) and does
// ZERO floating point — every step is `(int64)a*b >> kFrac` (an ARITHMETIC right shift on int64,
// deterministic + identical on every compiler/vendor) + integer add + integer compare. Each body is
// INDEPENDENT (no inter-body coupling until FPX3's collision response), so the GPU per-body write is
// trivially order-independent / race-free with NO atomics, and two runs are byte-identical.
//
// THE FIXED-POINT FORMAT: fx = int32_t in Q16.16 (kFrac=16, kOne=1<<16). Positions are bounded to
// +-32768 world units (the int32 integer part), all products use an int64 intermediate -> NO overflow
// within that bound (a fxmul of two values in [-32768,32768) Q16.16 is |a*b| <= (32768<<16)^2 which
// is < 2^62, well inside int64). The swraster.h SwEdge / mc.h d2 int64-intermediate discipline.
//
// REUSE MAP (file:line): FxISqrt copies engine/render/mc.h:461-472 (ISqrt, integer binary sqrt) on
// int64; the Q-format mirrors mc.h:457 (kFixed) + swraster.h:90 (kSub); FloorDiv copies the
// swraster.h:185 floorDiv lambda (deterministic floor division for negative numerators). The
// algorithm mirrored in fixed-point is engine/physics/world.cpp:72-209 (Step: integrate velocity ->
// integrate position -> ground clamp) — READ, NOT modified (fpx is additive + parallel; the float
// solver's phys.png golden stays byte-identical).

#include <cstdint>
#include <span>
#include <vector>

namespace hf::sim {
namespace fpx {

// ----- Q16.16 fixed-point scalar -------------------------------------------------------------------
using fx = int32_t;                       // a Q16.16 fixed-point scalar
inline constexpr int kFrac = 16;          // fractional bits
inline constexpr fx  kOne  = 1 << kFrac;  // 1.0 in Q16.16 (65536)

// fxmul(a,b) = (a * b) >> kFrac, with an int64 INTERMEDIATE (the swraster SwEdge / mc d2 pattern) so
// the product never overflows within the documented +-32768 bound, then an ARITHMETIC right shift
// (deterministic, pinned identically CPU<->HLSL/MSL). Truncates toward negative infinity for negative
// products (>> is floor on two's-complement), which is the bit the shader reproduces VERBATIM.
inline fx fxmul(fx a, fx b) {
    return (fx)(((int64_t)a * (int64_t)b) >> kFrac);
}

// ----- The integer 3-vector (the twin of math::Vec3) -----------------------------------------------
struct FxVec3 {
    fx x = 0, y = 0, z = 0;
};

inline FxVec3 FxAdd(const FxVec3& a, const FxVec3& b) {
    return FxVec3{a.x + b.x, a.y + b.y, a.z + b.z};
}
inline FxVec3 FxSub(const FxVec3& a, const FxVec3& b) {
    return FxVec3{a.x - b.x, a.y - b.y, a.z - b.z};
}
// Component-wise fxmul by a scalar s (Q16.16): the per-axis (v.axis * s) >> kFrac.
inline FxVec3 FxScale(const FxVec3& v, fx s) {
    return FxVec3{fxmul(v.x, s), fxmul(v.y, s), fxmul(v.z, s)};
}

// ----- Integer floor square root (copied from render/mc.h::ISqrt, on int64) ------------------------
// floor(sqrt(v)) for a non-negative int64 v, pure integer (binary digit-by-digit) — identical on every
// compiler/vendor, NO <cmath>. Provided for FPX2/FPX3's length-based broadphase/collision; FPX1's
// integrator only needs add + fxmul.
inline int64_t FxISqrt(int64_t v) {
    if (v <= 0) return 0;
    int64_t bit = (int64_t)1 << 62;
    while (bit > v) bit >>= 2;
    int64_t res = 0;
    while (bit != 0) {
        if (v >= res + bit) { v -= res + bit; res = (res >> 1) + bit; }
        else { res >>= 1; }
        bit >>= 2;
    }
    return res;
}

// FxLength(v) = sqrt(x^2 + y^2 + z^2) in Q16.16. The sum of squares is formed in int64 Q-format
// ((a in Q16.16)^2 = (a*a) which is Q32.32-ish; we keep it as the squared Q16.16 magnitude so
// FxISqrt of (sum >> kFrac scaled appropriately) returns the Q16.16 length). Concretely: each
// (axis*axis) is Q32.32 in an int64; their sum is the squared length in Q32.32; floor-sqrt of that
// Q32.32 value yields the length in Q16.16 (since sqrt(Q32.32)=Q16.16). Pure integer, NO float.
inline fx FxLength(const FxVec3& v) {
    int64_t sx = (int64_t)v.x * (int64_t)v.x;
    int64_t sy = (int64_t)v.y * (int64_t)v.y;
    int64_t sz = (int64_t)v.z * (int64_t)v.z;
    return (fx)FxISqrt(sx + sy + sz);
}

// ----- The rigid body + world (the fixed-point twin of physics::RigidBody / physics::World) --------
// flags bit 0 = dynamic (integrated by gravity); invMass==0 / !(flags&1) => static/kinematic (never
// moves). std430-packable as plain int32s (pos.xyz, vel.xyz, invMass, flags) — the GPU FxBody mirror.
struct FxBody {
    FxVec3   pos;            // Q16.16 world position
    FxVec3   vel;            // Q16.16 velocity (world units / second)
    fx       invMass = 0;    // Q16.16 inverse mass (0 => infinite mass / static)
    uint32_t flags   = 0;    // bit0 = dynamic
    // Slice FPX2 broadphase: the body's per-axis half-extent (a Q16.16 sphere/box radius). DEFAULTED to 0
    // so FPX1's bodies/tests/showcase are UNCHANGED in behavior (FPX1's IntegrateStep IGNORES radius — it
    // only reads pos/vel/invMass/flags). The integer AABB BodyAabb(b) = {pos - radius, pos + radius}.
    fx       radius  = 0;    // Q16.16 broadphase half-extent (0 => a point; FPX2's AabbOverlap input)
};

inline constexpr uint32_t kFlagDynamic = 1u;

struct FxWorld {
    FxVec3              gravity;          // Q16.16 acceleration (e.g. (0, -9.8, 0))
    fx                  groundY = 0;      // Q16.16 ground plane height (a single floor clamp)
    std::vector<FxBody> bodies;
};

// ----- IntegrateStep: the deterministic semi-implicit-Euler integrator (the VERBATIM shader math) ---
// For each body, if dynamic (flags & kFlagDynamic):
//   vel += gravity * dt   (component-wise fxmul)
//   pos += vel * dt       (component-wise fxmul)
//   single non-penetration FLOOR clamp: if (pos.y < groundY) { pos.y = groundY; if (vel.y < 0) vel.y = 0; }
// Pure integer; fixed iteration order; no RNG, no clock. Each body is independent of every other body
// (no inter-body coupling), so the order over bodies does NOT matter -> two-run bit-identical AND the
// GPU per-thread write is race-free with NO atomics. The shader runs THIS exact per-body body.
inline void IntegrateBody(FxBody& b, const FxVec3& gravity, fx groundY, fx dt) {
    if (!(b.flags & kFlagDynamic)) return;
    // (1) integrate velocity: vel += gravity * dt.
    b.vel.x += fxmul(gravity.x, dt);
    b.vel.y += fxmul(gravity.y, dt);
    b.vel.z += fxmul(gravity.z, dt);
    // (2) integrate position: pos += vel * dt.
    b.pos.x += fxmul(b.vel.x, dt);
    b.pos.y += fxmul(b.vel.y, dt);
    b.pos.z += fxmul(b.vel.z, dt);
    // (3) ground floor clamp (no restitution yet — FPX3).
    if (b.pos.y < groundY) {
        b.pos.y = groundY;
        if (b.vel.y < 0) b.vel.y = 0;
    }
}

inline void IntegrateStep(FxWorld& w, fx dt) {
    const size_t n = w.bodies.size();
    for (size_t i = 0; i < n; ++i)
        IntegrateBody(w.bodies[i], w.gravity, w.groundY, dt);
}

// ----- Integer broadphase cell quantization --------------------------------------------------------
// FloorDiv(n, d): deterministic FLOOR division for positive divisor d, ANY-sign numerator n (the
// swraster.h:185 floorDiv lambda, copied VERBATIM). Plain C++/HLSL integer `/` truncates TOWARD ZERO,
// so for a negative n it would land in the WRONG cell at the boundary; FloorDiv corrects that so cell
// assignment is monotone across 0. The shader's BroadphaseCell copies THIS body VERBATIM.
inline int32_t FloorDiv(int32_t n, int32_t d) {
    int32_t q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

// An integer cell coordinate (the int3 twin).
struct FxCell {
    int32_t x = 0, y = 0, z = 0;
};

// BroadphaseCell(p, cellSize): the integer grid cell a Q16.16 position falls in, FloorDiv per axis
// (the vt.h VtPageId floor-quantize analog, deterministic for negatives). cellSize is in the SAME
// Q16.16 units as p (e.g. kOne*2 for a 2-world-unit cell).
inline FxCell BroadphaseCell(const FxVec3& p, fx cellSize) {
    return FxCell{FloorDiv(p.x, cellSize), FloorDiv(p.y, cellSize), FloorDiv(p.z, cellSize)};
}

// CellId(cell, gridDim): flat linearization of a cell coord into a grid of gridDim.{x,y,z} cells (the
// mc.h cellId pattern). Caller offsets cell into [0,gridDim) before linearizing.
inline uint32_t CellId(const FxCell& cell, const FxCell& gridDim) {
    return (uint32_t)((cell.z * gridDim.y + cell.y) * gridDim.x + cell.x);
}

// ===== Slice FPX2 — integer AABB broadphase: the DETERMINISTIC candidate-pair generator ==============
// Pure INT32 (no fxmul, no int64, no products — only integer add/sub + compares), so the GPU
// fpx_pair_count/scan/emit shaders copy this VERBATIM and MSL-generate NATIVELY on Metal (unlike FPX1's
// int64 fxmul integrator, which is Vulkan-only). The whole broadphase is bit-identical cross-vendor by
// construction: AabbOverlap is six integer compares, the count/emit are per-body independent over j>i in
// a fixed order, and the offset prefix-sum is a single-thread serial scan. This is the prerequisite for
// FPX3's collision response (narrowphase + impulses run per candidate pair).

// An integer (Q16.16) axis-aligned bounding box. lo/hi are inclusive corners in the SAME Q16.16 units as
// FxBody::pos. Pure integer add/sub to build, six compares to overlap-test.
struct FxAabb {
    FxVec3 lo, hi;
};

// BodyAabb(b) = { pos - radius (per-axis), pos + radius (per-axis) }. Pure integer (no float, no products).
// radius is the body's broadphase half-extent (0 => a point AABB lo==hi==pos). The shader's BodyAabb
// copies THIS body VERBATIM.
inline FxAabb BodyAabb(const FxBody& b) {
    return FxAabb{
        FxVec3{b.pos.x - b.radius, b.pos.y - b.radius, b.pos.z - b.radius},
        FxVec3{b.pos.x + b.radius, b.pos.y + b.radius, b.pos.z + b.radius},
    };
}

// AabbOverlap(a, b): the deterministic broadphase predicate — SIX integer compares, NO products, NO
// int64. Two AABBs overlap iff they overlap on EVERY axis (separating-axis test on the 3 axes). Touching
// (a.lo.x == b.hi.x) counts as overlap (<=), matching the inclusive-corner convention. Bit-identical
// cross-vendor by construction (the strongest form). The shader copies THIS body VERBATIM.
inline bool AabbOverlap(const FxAabb& a, const FxAabb& b) {
    return a.lo.x <= b.hi.x && b.lo.x <= a.hi.x &&
           a.lo.y <= b.hi.y && b.lo.y <= a.hi.y &&
           a.lo.z <= b.hi.z && b.lo.z <= a.hi.z;
}

// CountPairs(world, perBodyCountOut): the CPU reference count. perBodyCountOut[i] = #{ j>i :
// AabbOverlap(BodyAabb(i), BodyAabb(j)) }; returns the total over all i. Ordered j>i so each unordered
// pair is counted ONCE, deterministically, with a canonical i<j orientation (the gpu_culled.h
// source-order discipline). radius is carried on each body now. The shader fpx_pair_count.comp computes
// THIS per-thread (one thread per body i).
inline uint32_t CountPairs(const FxWorld& world, std::span<uint32_t> perBodyCountOut) {
    const uint32_t n = (uint32_t)world.bodies.size();
    uint32_t total = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const FxAabb ai = BodyAabb(world.bodies[i]);
        uint32_t c = 0;
        for (uint32_t j = i + 1; j < n; ++j)
            if (AabbOverlap(ai, BodyAabb(world.bodies[j]))) ++c;
        if (i < perBodyCountOut.size()) perBodyCountOut[i] = c;
        total += c;
    }
    return total;
}

// A candidate collision pair (i<j, canonical). std430-packable as 2 x uint32 (8 bytes) — the GPU FxPair
// mirror.
struct FxPair {
    uint32_t i, j;
};

// BuildPairs(world, perBodyOffset, pairsOut): the full CPU mesher (the MC3 count->scan->emit, but on
// bodies). (1) CountPairs -> per-body counts; (2) exclusive prefix-sum -> perBodyOffset (each body's
// disjoint write base); (3) for each i, emit each overlapping (i, j>i) at pairsOut[perBodyOffset[i] +
// local++]. The pair list is grouped by i (ascending) then j (ascending) -> fully deterministic, the
// exact list the GPU memcmp's against. The GPU does the SAME three passes (count/scan/emit).
inline void BuildPairs(const FxWorld& world, std::vector<uint32_t>& perBodyOffset,
                       std::vector<FxPair>& pairsOut) {
    const uint32_t n = (uint32_t)world.bodies.size();
    std::vector<uint32_t> counts((size_t)n, 0u);
    const uint32_t total = CountPairs(world, std::span<uint32_t>(counts));

    // Exclusive prefix-sum of counts -> perBodyOffset (the single-thread serial scan, == mc_scan).
    perBodyOffset.assign((size_t)n, 0u);
    uint32_t running = 0;
    for (uint32_t i = 0; i < n; ++i) {
        perBodyOffset[i] = running;
        running += counts[i];
    }

    // Emit each overlapping (i, j>i) into i's disjoint range [offset[i], offset[i]+counts[i]).
    pairsOut.assign((size_t)total, FxPair{0u, 0u});
    for (uint32_t i = 0; i < n; ++i) {
        const FxAabb ai = BodyAabb(world.bodies[i]);
        uint32_t local = 0;
        for (uint32_t j = i + 1; j < n; ++j) {
            if (AabbOverlap(ai, BodyAabb(world.bodies[j]))) {
                pairsOut[perBodyOffset[i] + local] = FxPair{i, j};
                ++local;
            }
        }
    }
}

// ===== Slice FPX3 — fixed-point PBD POSITIONAL collision response (the MAKE-OR-BREAK of FLAGSHIP #6) ===
// A POSITION-BASED-DYNAMICS positional solver: resolve each CONTACT (a body-vs-ground or a penetrating
// sphere-sphere pair from FPX2's BuildPairs) by moving the two bodies apart along the contact normal by
// their inverse-mass-weighted share of the penetration depth — purely POSITIONAL (NO velocity impulses,
// NO restitution; FPX4+). Over the FPX2 candidate pairs + the ground, iterate K times in a FIXED order
// (Gauss-Seidel: each contact reads the LATEST positions) -> inherently sequential -> the GPU mirror is
// SINGLE-THREAD ([numthreads(1,1,1)], the mc_scan/vt_alloc pattern) -> bit-exact GPU==CPU + cross-backend.
//
// THE int64/glslc METAL LESSON (FPX1): fxdiv + FxISqrt use int64 -> DXC compiles (Vulkan), glslc (the
// Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64 in HLSL. So shaders/fpx_solve.comp.hlsl is
// VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal hf_gen_msl list); the Metal --fpx-solve
// showcase runs the CPU StepWorld over the same world -> byte-identical to the Vulkan GPU result by
// construction (the Vulkan side carries the GPU==CPU proof). The math here is copied VERBATIM by
// fpx_solve.comp so the GPU exercises the EXACT integer ops -> the GPU==CPU memcmp catches any divergence.

// fxdiv(a,b) = (a << kFrac) / b in Q16.16 — an int64 SHIFT then a TRUNCATING integer divide (truncation
// TOWARD ZERO, identical C++/HLSL/MSL). The most overflow-sensitive op: a is shifted up by 16 bits into an
// int64 (|a|<2^31 -> |a<<16|<2^47, well inside int64) then divided by the non-zero denominator b (a mass
// sum or a distance). Guard b==0 -> return 0 (a contact with a zero denominator is degenerate -> no move).
inline fx fxdiv(fx a, fx b) {
    if (b == 0) return 0;
    return (fx)(((int64_t)a << kFrac) / (int64_t)b);
}

// FxNormalize(v): the unit vector of v in Q16.16 (integer normalize, NO std::sqrt / <cmath>). len =
// FxLength(v) (the FxISqrt of the int64 sum-of-squares); if len==0 return a fixed fallback (0,kOne,0) so a
// coincident pair has a deterministic normal; else { fxdiv(v.x,len), fxdiv(v.y,len), fxdiv(v.z,len) }.
inline FxVec3 FxNormalize(const FxVec3& v) {
    const fx len = FxLength(v);
    if (len == 0) return FxVec3{0, kOne, 0};
    return FxVec3{fxdiv(v.x, len), fxdiv(v.y, len), fxdiv(v.z, len)};
}

// ResolveGround(b, groundY): the ground contact (NO normalize — axis-aligned). A body with
// pos.y - radius < groundY penetrates by pen = groundY + radius - pos.y; resolve pos.y += pen (the ground
// is static/infinite mass -> the body takes the full correction). Static bodies (invMass==0) take nothing.
// Pure integer, no divide, no sqrt — the safe core. The shader copies THIS body VERBATIM.
inline void ResolveGround(FxBody& b, fx groundY) {
    if (b.invMass == 0) return;
    const fx pen = groundY + b.radius - b.pos.y;
    if (pen > 0) b.pos.y += pen;
}

// ResolvePair(a, b): the sphere-sphere contact. d = b.pos - a.pos; dist = FxLength(d); pen =
// (a.radius + b.radius) - dist; if pen > 0: n = FxNormalize(d); the inverse-mass shares
// wi = fxdiv(a.invMass, a.invMass + b.invMass), wj = kOne - wi; a.pos -= n * fxmul(pen, wi);
// b.pos += n * fxmul(pen, wj). Static bodies (invMass==0) take no correction; if BOTH static, skip.
// int64 in dist (FxISqrt) / fxdiv / fxmul. The shader copies THIS body VERBATIM.
inline void ResolvePair(FxBody& a, FxBody& b) {
    const fx invSum = a.invMass + b.invMass;
    if (invSum == 0) return;                          // both static -> skip
    const FxVec3 d = FxSub(b.pos, a.pos);
    const fx dist = FxLength(d);
    const fx pen = (a.radius + b.radius) - dist;
    if (pen <= 0) return;
    const FxVec3 n = FxNormalize(d);
    const fx wi = fxdiv(a.invMass, invSum);
    const fx wj = kOne - wi;
    const FxVec3 ci = FxScale(n, fxmul(pen, wi));     // a moves -ci
    const FxVec3 cj = FxScale(n, fxmul(pen, wj));     // b moves +cj
    a.pos = FxSub(a.pos, ci);
    b.pos = FxAdd(b.pos, cj);
}

// SolveContacts(w, pairs, iterations): the CPU reference. K iterations; each iteration resolves ALL ground
// contacts (in body order) THEN ALL pair contacts in the FIXED FPX2 pair order (ascending) — a
// deterministic Gauss-Seidel sweep. Sequential (each contact reads the latest positions) -> order-dependent
// -> the GPU mirror MUST be single-thread. The shader runs THIS exact double loop.
inline void SolveContacts(FxWorld& w, std::span<const FxPair> pairs, int iterations) {
    const size_t n = w.bodies.size();
    for (int it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n; ++i) ResolveGround(w.bodies[i], w.groundY);
        for (const FxPair& p : pairs) {
            if (p.i < n && p.j < n) ResolvePair(w.bodies[p.i], w.bodies[p.j]);
        }
    }
}

// StepWorld(w, pairs, dt, solveIters) = IntegrateStep(dt) + SolveContacts(pairs, solveIters). The full
// deterministic fixed-point physics step. The showcase runs `steps` of THIS over a fixed pair list.
inline void StepWorld(FxWorld& w, std::span<const FxPair> pairs, fx dt, int solveIters) {
    IntegrateStep(w, dt);
    SolveContacts(w, pairs, solveIters);
}

// CountResidualOverlaps(w, pairs): the deterministic count of pairs still penetrating after the solve (a
// reporting/diagnostic helper — NOT necessarily 0; bit-exact CPU<->GPU because it is pure integer compares
// over FxLength). Pen > 0 means the spheres still overlap.
inline uint32_t CountResidualOverlaps(const FxWorld& w, std::span<const FxPair> pairs) {
    const size_t n = w.bodies.size();
    uint32_t r = 0;
    for (const FxPair& p : pairs) {
        if (p.i >= n || p.j >= n) continue;
        const FxVec3 d = FxSub(w.bodies[p.j].pos, w.bodies[p.i].pos);
        const fx pen = (w.bodies[p.i].radius + w.bodies[p.j].radius) - FxLength(d);
        if (pen > 0) ++r;
    }
    return r;
}

}  // namespace fpx
}  // namespace hf::sim
