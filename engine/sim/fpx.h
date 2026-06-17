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

// ----- Slice FPX4: the Q16.16 quaternion (declared early so FxBody can carry a DEFAULTED orient) -----
// A fixed-point quaternion (the fixed-point twin of math::Quat). identity = {0,0,0,kOne}. std430-packable
// as 4 x int32 (x,y,z,w) — the GPU FxQuat mirror. The FPX4 math (FxQuatMul/FxQuatNormalize/FxRotate/
// IntegrateOrientation/IntegrateBodyFull) lives in the FPX4 section below; only the type + kHalf are
// declared here so FxBody can DEFAULT an `orient`/`angVel` without changing FPX1-FPX3 behavior.
inline constexpr fx kHalf = kOne / 2;   // 0.5 in Q16.16 (the 0.5*dq*dt factor of q' = 0.5*ω⊗q)
struct FxQuat {
    fx x = 0, y = 0, z = 0, w = kOne;
};

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
    // Slice FPX4 orientation: a Q16.16 quaternion + angular velocity. DEFAULTED (identity orient, zero
    // angVel) so FPX1-FPX3's IntegrateStep/StepWorld + their GPU packs (FpxBodyGpu etc., which pack ONLY
    // the FPX1-3 fields above) are byte-identical — FPX1-3 NEVER read these. FPX4's IntegrateBodyFull /
    // fpx_orient.comp read+write them; the FPX4 GPU pack appends orient.xyzw + angVel.xyz.
    FxQuat   orient;         // Q16.16 orientation quaternion (default identity {0,0,0,kOne})
    FxVec3   angVel;         // Q16.16 angular velocity (radians/second, body-axis omega; default 0)
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

// ===== Slice FPX4 — fixed-point integer QUATERNION ORIENTATION (Phase 11 #4) ========================
// Add ORIENTATION to the deterministic fixed-point sim: each body carries a Q16.16 quaternion `orient`
// + an angular velocity `angVel`, integrated per-body from angVel as a fixed-point quaternion,
// normalized via the integer sqrt (FxISqrt) — so the physics state is full 6-DOF (position +
// orientation) and STILL bit-identical CPU<->Vulkan<->Metal AND frame-deterministic. ADDITIVE: the
// orient/angVel fields below are DEFAULTED on FxBody so FPX1-FPX3's IntegrateStep/StepWorld + their GPU
// packs (which pack ONLY the FPX1-3 fields) are byte-identical, and their goldens are unchanged.
//
// THE int64/glslc METAL LESSON (FPX1/FPX3): FxQuatMul's Hamilton products + FxQuatNormalize's FxISqrt
// use the int64 fxmul/fxdiv. DXC compiles int64 (the Vulkan path); glslc (the Metal HLSL->SPIR-V->MSL
// frontend) CANNOT parse int64_t in HLSL. So shaders/fpx_orient.comp.hlsl is VULKAN-SPIR-V-ONLY (in the
// Vulkan compile list, NOT the Metal hf_gen_msl list); the Metal --fpx-orient showcase runs the CPU
// IntegrateBodyFull over the same bodies -> byte-identical to the Vulkan GPU result by construction (the
// Vulkan side carries the GPU==CPU bit-identity proof). The math here is copied VERBATIM by
// fpx_orient.comp so the GPU exercises the EXACT integer ops -> the GPU==CPU memcmp catches divergence.
// (FxQuat + kHalf are declared near FxBody above so the body can carry a DEFAULTED orient/angVel.)

// FxQuatMul(a, b): the Hamilton product a*b, each term an int64 fxmul (the swraster/mc int64 discipline,
// no overflow within the documented bound; for unit quaternions every component is in [-kOne,kOne]). The
// standard quaternion multiplication. The shader copies THIS body VERBATIM.
inline FxQuat FxQuatMul(const FxQuat& a, const FxQuat& b) {
    FxQuat r;
    r.w = fxmul(a.w, b.w) - fxmul(a.x, b.x) - fxmul(a.y, b.y) - fxmul(a.z, b.z);
    r.x = fxmul(a.w, b.x) + fxmul(a.x, b.w) + fxmul(a.y, b.z) - fxmul(a.z, b.y);
    r.y = fxmul(a.w, b.y) - fxmul(a.x, b.z) + fxmul(a.y, b.w) + fxmul(a.z, b.x);
    r.z = fxmul(a.w, b.z) + fxmul(a.x, b.y) - fxmul(a.y, b.x) + fxmul(a.z, b.w);
    return r;
}

// FxQuatNormalize(q): the unit quaternion of q in Q16.16 (integer normalize, NO std::sqrt / <cmath>).
// len = FxISqrt of the int64 Q32.32 sum-of-squares (each q.c*q.c is Q32.32; sqrt(Q32.32)=Q16.16, the
// FxLength discipline on 4 components). len==0 -> return identity (a deterministic fallback). Otherwise
// {fxdiv(x,len), fxdiv(y,len), fxdiv(z,len), fxdiv(w,len)}. The fixed-point normalize is NOT perfect:
// |q| drifts slightly but DETERMINISTICALLY (the |q|≈kOne-within-tol check is the proof). The shader
// copies THIS body VERBATIM.
inline FxQuat FxQuatNormalize(const FxQuat& q) {
    const int64_t sx = (int64_t)q.x * (int64_t)q.x;
    const int64_t sy = (int64_t)q.y * (int64_t)q.y;
    const int64_t sz = (int64_t)q.z * (int64_t)q.z;
    const int64_t sw = (int64_t)q.w * (int64_t)q.w;
    const fx len = (fx)FxISqrt(sx + sy + sz + sw);
    if (len == 0) return FxQuat{0, 0, 0, kOne};
    return FxQuat{fxdiv(q.x, len), fxdiv(q.y, len), fxdiv(q.z, len), fxdiv(q.w, len)};
}

// FxRotate(q, v): rotate the vector v by the (unit) quaternion q, via the optimized form
// v' = v + 2*cross(q.xyz, cross(q.xyz, v) + q.w*v) — all fxmul, NO float. For the orientation-gizmo viz
// (project FxRotate(orient, axis) for the 3 local axes). The shader does NOT need this (host-only viz).
inline FxVec3 FxRotate(const FxQuat& q, const FxVec3& v) {
    // u = q.xyz; t = cross(u, v) + q.w*v ; v' = v + 2*cross(u, t).
    const FxVec3 u{q.x, q.y, q.z};
    // c1 = cross(u, v).
    const FxVec3 c1{
        fxmul(u.y, v.z) - fxmul(u.z, v.y),
        fxmul(u.z, v.x) - fxmul(u.x, v.z),
        fxmul(u.x, v.y) - fxmul(u.y, v.x),
    };
    // t = c1 + q.w*v.
    const FxVec3 t{c1.x + fxmul(q.w, v.x), c1.y + fxmul(q.w, v.y), c1.z + fxmul(q.w, v.z)};
    // c2 = cross(u, t).
    const FxVec3 c2{
        fxmul(u.y, t.z) - fxmul(u.z, t.y),
        fxmul(u.z, t.x) - fxmul(u.x, t.z),
        fxmul(u.x, t.y) - fxmul(u.y, t.x),
    };
    // v' = v + 2*c2.
    return FxVec3{v.x + 2 * c2.x, v.y + 2 * c2.y, v.z + 2 * c2.z};
}

// IntegrateOrientation(b, dt): the deterministic quaternion angular integrator. dq = ω⊗q (the pure-
// quaternion product of ωquat={angVel.xyz,0} with the current orient); orient += 0.5*dq*dt component-
// wise; then renormalize. Fixed op order, per-body independent. The shader copies THIS body VERBATIM.
inline void IntegrateOrientation(FxBody& b, fx dt) {
    const FxQuat omega{b.angVel.x, b.angVel.y, b.angVel.z, 0};
    const FxQuat dq = FxQuatMul(omega, b.orient);
    b.orient.x += fxmul(fxmul(dq.x, kHalf), dt);
    b.orient.y += fxmul(fxmul(dq.y, kHalf), dt);
    b.orient.z += fxmul(fxmul(dq.z, kHalf), dt);
    b.orient.w += fxmul(fxmul(dq.w, kHalf), dt);
    b.orient = FxQuatNormalize(b.orient);
}

// IntegrateBodyFull(b, gravity, dt) = the FPX1 translational IntegrateStep body (vel += gravity*dt; pos
// += vel*dt; NO ground clamp here — FPX4's showcase has no ground) + IntegrateOrientation. Only dynamic
// bodies move (translation gated by kFlagDynamic, matching IntegrateBody); orientation integrates for
// every body (a static body with angVel=0 is unchanged anyway). FPX1-FPX3 keep calling the original
// IntegrateStep — UNCHANGED. The shader runs THIS exact per-body body K times.
inline void IntegrateBodyFull(FxBody& b, const FxVec3& gravity, fx dt) {
    if (b.flags & kFlagDynamic) {
        b.vel.x += fxmul(gravity.x, dt);
        b.vel.y += fxmul(gravity.y, dt);
        b.vel.z += fxmul(gravity.z, dt);
        b.pos.x += fxmul(b.vel.x, dt);
        b.pos.y += fxmul(b.vel.y, dt);
        b.pos.z += fxmul(b.vel.z, dt);
    }
    IntegrateOrientation(b, dt);
}

// ===== Slice FPX5 — LOCKSTEP + ROLLBACK PROOF (FLAGSHIP #6's HEADLINE) ================================
// The netcode primitive UE5's float Chaos cannot provide: prove that the bit-exact fixed-point sim
// (FPX1-FPX4) is true cross-platform LOCKSTEP + ROLLBACK-ready. Two peers fed ONLY a deterministic
// input/command STREAM (not full state) re-simulate to BIT-IDENTICAL physics state; a MISPREDICTED input
// is corrected by ROLLING BACK to a saved snapshot + re-simulating the authoritative stream. PURE CPU,
// 0 backend symbols, NO new shader / RHI: this is a determinism PROPERTY of the existing fpx sim — the
// cross-backend zero-diff golden (the converged state is bit-identical on Vulkan-Windows AND Metal-Mac
// from the same inputs) IS the cross-platform-lockstep evidence. NO <cmath>, NO RNG, NO clock.
//
// A FxCommand is the deterministic per-tick INPUT a netcode layer would put on the wire (NOT full state).
// kind=0 apply-impulse `arg` to bodyId's velocity (an integer add, the deterministic input event);
// kind=1 set bodyId's angular velocity to `arg`. A std::vector<FxCommand> is the command STREAM. The
// stream is processed in ARRAY ORDER for each tick (the deterministic-order contract — the same order on
// every peer/platform), so authority + replica fed the same stream re-derive the same state exactly.

inline constexpr uint32_t kCmdImpulse = 0u;   // arg added to bodyId's velocity (an input impulse)
inline constexpr uint32_t kCmdSetAngVel = 1u; // bodyId's angVel set to arg (an input spin)

struct FxCommand {
    uint32_t tick   = 0;   // the tick this input applies on
    uint32_t kind   = 0;   // kCmdImpulse / kCmdSetAngVel
    uint32_t bodyId = 0;   // the target body index
    FxVec3   arg;          // the Q16.16 payload (impulse delta-velocity / angular velocity)
};

// ApplyCommand(w, c): apply ONE input command to the world (pure integer — add to vel / set angVel).
// Out-of-range bodyId is a no-op (deterministic). Unknown kind is a no-op. The input event the
// lockstep/rollback streams are made of.
inline void ApplyCommand(FxWorld& w, const FxCommand& c) {
    if (c.bodyId >= (uint32_t)w.bodies.size()) return;
    FxBody& b = w.bodies[c.bodyId];
    if (c.kind == kCmdImpulse) {
        b.vel.x += c.arg.x;
        b.vel.y += c.arg.y;
        b.vel.z += c.arg.z;
    } else if (c.kind == kCmdSetAngVel) {
        b.angVel = c.arg;
    }
}

// SimTick(w, stream, tick, dt, solveIters): the deterministic per-tick step. (1) apply ALL commands in
// `stream` whose .tick == `tick`, in ARRAY ORDER (the deterministic input-order contract); (2) rebuild
// the candidate pair list from the CURRENT positions (BuildPairs is deterministic — realistic lockstep,
// re-broadphased each tick); (3) StepWorld (IntegrateStep + SolveContacts) THEN IntegrateBodyFull's
// orientation integrate per body (so the per-tick step is full 6-DOF: translation+collision via
// StepWorld, orientation via IntegrateOrientation — applied AFTER the solve so positions are settled).
// Pure integer, fixed order -> bit-identical on every peer/platform.
inline void SimTick(FxWorld& w, const std::vector<FxCommand>& stream, uint32_t tick, fx dt,
                    int solveIters) {
    for (const FxCommand& c : stream)
        if (c.tick == tick) ApplyCommand(w, c);
    std::vector<uint32_t> offsets;
    std::vector<FxPair> pairs;
    BuildPairs(w, offsets, pairs);
    StepWorld(w, std::span<const FxPair>(pairs), dt, solveIters);
    // Orientation integrate (the FPX4 angular step) for each body — IntegrateOrientation only, since
    // StepWorld already did the translational integrate + the collision solve.
    for (FxBody& b : w.bodies) IntegrateOrientation(b, dt);
}

// SnapshotWorld(w): a deep copy of the full integer world state (the std::vector<FxBody> + the scalar
// gravity/groundY). The ROLLBACK primitive — a lossless saved tick. (std::vector copy is a deep copy.)
inline FxWorld SnapshotWorld(const FxWorld& w) {
    return w;   // value copy: deep-copies the bodies vector + the scalars
}

// RestoreWorld(w, snap): restore the world to a saved snapshot (the rollback). Bit-exact round-trip
// with SnapshotWorld (RestoreWorld(w, SnapshotWorld(w0)) leaves w == w0 byte-for-byte).
inline void RestoreWorld(FxWorld& w, const FxWorld& snap) {
    w = snap;
}

// RunLockstep(init, stream, ticks, dt, iters): THE peer entry point. Run `ticks` SimTicks from a COPY of
// `init`, applying the command stream -> the final world. authority = RunLockstep(init, stream, N);
// replica = RunLockstep(init, stream, N) from the SAME init + stream (inputs ONLY — no state shared) ->
// BIT-IDENTICAL by determinism (the lockstep proof memcmps them).
inline FxWorld RunLockstep(const FxWorld& init, const std::vector<FxCommand>& stream, int ticks, fx dt,
                           int iters) {
    FxWorld w = init;
    for (int t = 0; t < ticks; ++t)
        SimTick(w, stream, (uint32_t)t, dt, iters);
    return w;
}

// RunRollback(init, authStream, mispredictStream, ticks, mispredictTick, dt, iters): the rollback
// harness. (1) run ticks 0..mispredictTick from init applying authStream, SAVING a snapshot AT
// mispredictTick (before that tick is simulated); (2) advance a few ticks from the snapshot with the
// MISPREDICTED stream (the wrong input) — this is the speculative client-prediction that diverges;
// (3) "receive" the authoritative input -> RestoreWorld to the snapshot + RE-SIMULATE
// mispredictTick..ticks with the CORRECT authStream -> the final corrected world. The proof asserts
// this == RunLockstep(init, authStream, ticks) (rollback corrected the misprediction EXACTLY) AND that
// the mispredicted-before-rollback state DIFFERED from the authority (a real divergence was fixed).
inline FxWorld RunRollback(const FxWorld& init, const std::vector<FxCommand>& authStream,
                           const std::vector<FxCommand>& mispredictStream, int ticks, int mispredictTick,
                           fx dt, int iters) {
    FxWorld w = init;
    // (1) advance 0..mispredictTick with the authoritative stream.
    for (int t = 0; t < mispredictTick; ++t)
        SimTick(w, authStream, (uint32_t)t, dt, iters);
    // (2) SAVE the snapshot at mispredictTick (the rollback restore point).
    const FxWorld snap = SnapshotWorld(w);
    // (2b) speculatively advance a few ticks with the MISPREDICTED stream (the wrong input) — this is the
    // client prediction that diverges from authority. Bounded to the remaining ticks.
    int specTicks = ticks - mispredictTick;
    if (specTicks > 3) specTicks = 3;
    for (int s = 0; s < specTicks; ++s)
        SimTick(w, mispredictStream, (uint32_t)(mispredictTick + s), dt, iters);
    // (3) ROLLBACK: restore the snapshot + re-simulate mispredictTick..ticks with the CORRECT authStream.
    RestoreWorld(w, snap);
    for (int t = mispredictTick; t < ticks; ++t)
        SimTick(w, authStream, (uint32_t)t, dt, iters);
    return w;
}

}  // namespace fpx
}  // namespace hf::sim
