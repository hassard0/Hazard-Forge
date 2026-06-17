// Slice FPX2 — Deterministic Fixed-Point Physics: the integer-AABB BROADPHASE per-body PAIR-EMIT compute
// pass (the MC3 mc_emit analog on bodies). ONE thread per body i (i < bodyCount). The thread rebuilds
// body i's AABB (BodyAabb VERBATIM), reads its write base triOffset = perBodyOffset[i] (the FPX2
// prefix-sum), re-scans every j > i, and EMITS each overlapping pair (i, j) into gPairs at
// perBodyOffset[i] + local++. Each body writes into its OWN DISJOINT [offset[i], offset[i]+count[i])
// range -> race-free, NO atomics. enabled=0 -> emit nothing (gPairs stays the pre-cleared upload).
//
// WHY BIT-IDENTICAL to the CPU fpx.h::BuildPairs (the make-or-break): every value written is PURE INT32 —
// the same six-compare AabbOverlap over the same host-snapped integer AABBs the CPU uses, the j>i scan in
// the same fixed order, and each body's range is disjoint so a thread race CANNOT change any byte. The
// grouped-by-i (ascending) then j-ascending order matches BuildPairs exactly. A divergence vs the header
// is exactly what the host's GPU==CPU memcmp (gPairs) catches. INT32 only -> MSL-gens natively on Metal.
//
// Buffers (storage, bound at compute bindings 0..3; on Metal these land at buffer(0..3)):
//   b0 gBodies       : the Q16.16 FxBody array (pos/vel/invMass/flags/radius — std430 ints), READ.
//   b1 perBodyOffset : one uint per body (the FPX2 prefix-sum write base), READ.
//   b2 gPairs        : the output FxPair array {i, j} per slot, WRITE (pre-cleared).
//   b3 gParams       : { bodyCount, enabled, _, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_FPX_THREADS 64

// std430 FxBody mirror (engine/sim/fpx.h::FxBody) — IDENTICAL to fpx_pair_count.comp's FpxBody (36 bytes).
struct FpxBody {
    int  px, py, pz;   // Q16.16 position
    int  vx, vy, vz;   // Q16.16 velocity
    int  invMass;      // Q16.16 inverse mass (carried; unused)
    uint flags;        // bit0 = dynamic (carried; unused)
    int  radius;       // Q16.16 broadphase half-extent
};

// std430 FxPair mirror (engine/sim/fpx.h::FxPair): 2 x uint = 8 bytes.
struct FxPair { uint i; uint j; };

// Params (std430). Mirrors the C++ upload struct.
//   cfg : x=bodyCount, y=enabled, z=unused, w=unused
struct FpxPairParams { int4 cfg; };

[[vk::binding(0, 0)]] RWStructuredBuffer<FpxBody>       gBodies       : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>          perBodyOffset : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<FxPair>        gPairs        : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<FpxPairParams> gParams       : register(u3);

// An integer AABB — VERBATIM fpx.h::FxAabb.
struct FxAabb { int3 lo; int3 hi; };

// BodyAabb(b) = { pos - radius, pos + radius } per-axis — VERBATIM fpx.h::BodyAabb.
FxAabb BodyAabb(FpxBody b) {
    FxAabb a;
    a.lo = int3(b.px - b.radius, b.py - b.radius, b.pz - b.radius);
    a.hi = int3(b.px + b.radius, b.py + b.radius, b.pz + b.radius);
    return a;
}

// AabbOverlap(a, b): SIX integer compares — VERBATIM fpx.h::AabbOverlap.
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

    // Disabled -> emit nothing (gPairs stays the pre-cleared upload; the byte-identical no-op).
    if (enabled == 0) return;

    FxAabb ai = BodyAabb(gBodies[i]);
    uint base  = perBodyOffset[i];   // this body's disjoint write base (from the prefix-sum)
    uint local = 0u;
    for (uint j = i + 1u; j < (uint)bodyCount; ++j) {
        if (AabbOverlap(ai, BodyAabb(gBodies[j]))) {
            FxPair p; p.i = i; p.j = j;
            gPairs[base + local] = p;
            local += 1u;
        }
    }
}
