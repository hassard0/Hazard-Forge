// Slice FPX2 — Deterministic Fixed-Point Physics: the integer-AABB BROADPHASE per-body PAIR-COUNT compute
// pass (the 2nd FPX slice, after FPX1's integrator; the MC2 mc_count analog on bodies). ONE thread per
// body i (i < bodyCount). The thread builds body i's integer AABB (BodyAabb, VERBATIM engine/sim/fpx.h),
// scans every j > i, and writes perBodyCount[i] = Σ AabbOverlap(BodyAabb(i), BodyAabb(j)) — the number of
// later bodies whose AABB overlaps i's (each unordered pair counted ONCE, canonical i<j). NO atomics
// (each thread writes its own perBodyCount[i]); the host sums for the total. enabled=0 -> write 0.
//
// WHY BIT-IDENTICAL to the CPU fpx.h::CountPairs (the make-or-break): the whole predicate is PURE INT32 —
// BodyAabb is integer add/sub (pos ± radius) and AabbOverlap is SIX integer compares, NO products, NO
// int64. So this MSL-generates NATIVELY on Metal (no int64, no --msl-version 20200), unlike FPX1's int64
// fxmul integrator which is Vulkan-only. A divergence vs the header is exactly what the host's GPU==CPU
// memcmp (perBodyCount) catches.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gBodies       : the Q16.16 FxBody array (pos.xyz, vel.xyz, invMass, flags, radius — std430 ints), READ.
//   b1 perBodyCount  : one uint per body (the overlap count for j>i), WRITE.
//   b2 gParams       : { bodyCount, enabled, _, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations (same as
// mc_count.comp / fpx_integrate.comp), not backend CODE symbols.

#define HF_FPX_THREADS 64

// std430 FxBody mirror (engine/sim/fpx.h::FxBody): pos.xyz, vel.xyz, invMass (int), flags (uint), radius
// (int). 9 x 4-byte = 36 bytes, no padding holes (memcmp-able). The broadphase reads pos + radius.
struct FpxBody {
    int  px, py, pz;   // Q16.16 position
    int  vx, vy, vz;   // Q16.16 velocity
    int  invMass;      // Q16.16 inverse mass (carried; unused by the broadphase)
    uint flags;        // bit0 = dynamic (carried; unused by the broadphase)
    int  radius;       // Q16.16 broadphase half-extent
};

// Params (std430). Mirrors the C++ upload struct.
//   cfg : x=bodyCount, y=enabled, z=unused, w=unused
struct FpxPairParams {
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FpxBody>       gBodies      : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>          perBodyCount : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<FpxPairParams> gParams      : register(u2);

// An integer AABB (lo/hi inclusive corners) — VERBATIM fpx.h::FxAabb.
struct FxAabb { int3 lo; int3 hi; };

// BodyAabb(b) = { pos - radius, pos + radius } per-axis — VERBATIM fpx.h::BodyAabb. Pure integer add/sub.
FxAabb BodyAabb(FpxBody b) {
    FxAabb a;
    a.lo = int3(b.px - b.radius, b.py - b.radius, b.pz - b.radius);
    a.hi = int3(b.px + b.radius, b.py + b.radius, b.pz + b.radius);
    return a;
}

// AabbOverlap(a, b): SIX integer compares — VERBATIM fpx.h::AabbOverlap. NO products, NO int64.
bool AabbOverlap(FxAabb a, FxAabb b) {
    return a.lo.x <= b.hi.x && b.lo.x <= a.hi.x &&
           a.lo.y <= b.hi.y && b.lo.y <= a.hi.y &&
           a.lo.z <= b.hi.z && b.lo.z <= a.hi.z;
}

[numthreads(HF_FPX_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int bodyCount = gParams[0].cfg.x;
    int enabled   = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= bodyCount) return;

    // Disabled -> write 0 (perBodyCount stays the cleared all-zero upload; the byte-identical no-op).
    if (enabled == 0) { perBodyCount[i] = 0u; return; }

    FxAabb ai = BodyAabb(gBodies[i]);
    uint c = 0u;
    for (uint j = i + 1u; j < (uint)bodyCount; ++j)
        if (AabbOverlap(ai, BodyAabb(gBodies[j]))) c += 1u;

    perBodyCount[i] = c;   // order-independent per-body integer write (NO atomics)
}
