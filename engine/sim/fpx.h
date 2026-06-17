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

}  // namespace fpx
}  // namespace hf::sim
