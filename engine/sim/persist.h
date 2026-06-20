#pragma once
// Slice PS1 — Deterministic Persistent Contacts: THE CONTACT FEATURE ID (the integer BEACHHEAD of
// FLAGSHIP #21: DETERMINISTIC WARM-STARTED CONTACT CACHING + SLEEPING ISLANDS, hf::sim::persist). Warm-
// starting (re-applying last tick's accumulated contact impulses) and sleeping (resting low-energy bodies)
// are the stabilization machinery every shipping physics engine relies on — but float caches + thresholds
// diverge machine-to-machine, which is why their networked physics is never bit-deterministic. This flagship
// makes that machinery DETERMINISTIC. PS1 builds the primitive every later slice needs: a deterministic
// integer CONTACT FEATURE ID that uniquely + reproducibly names a contact point across ticks, so the cache
// can match this tick's manifold points to last tick's accumulated impulses.
//
// Header-only, namespace hf::sim::persist, #include "sim/fric.h" READ-ONLY (do NOT modify fric.h/convex.h/
// fpx.h — persist.h is a NEW additive sibling). fric.h transitively gives convex (SatResult, ContactManifold,
// FxBox, SatPair) + fpx (FxBody).
//
// THE PURE-INT32 WIN (the STRONGEST proof tier): unlike the int64 contact MATH (SAT/manifold/impulse — those
// shaders are Vulkan-SPIR-V-only because their Q16.16 products need int64), the contact KEY is PURE int32 —
// compares + bit shifts + xors, NO Q16.16 products, NO int64, NO float. So shaders/persist_key.comp.hlsl is
// MSL-NATIVE (it goes IN the metal_headless hf_gen_msl list, NOT the Vulkan-only list) — a TRUE GPU pass on
// BOTH backends, with strict zero-differing-pixel cross-vendor (the FR2/MC1/BD2 tier). The shader copies
// MakeContactKey + ContactKeyHash VERBATIM so the GPU ContactKey[]+hash[] is byte-identical to the CPU
// reference (the Vulkan GPU==CPU memcmp + the Metal GPU strict-zero are the proofs).
//
// THE ORDER-NORMALIZATION (the determinism crux): a contact between bodies (i, j) and (j, i) is the SAME
// contact — so the key STORES bodyA < bodyB ALWAYS (swap if the caller passes bodyA > bodyB). The same pair
// thus yields the same key regardless of the broadphase iteration order. The key is an IDENTITY (which two
// bodies + which SAT axis + which clip corner), NOT a geometric frame: when the bodies are swapped only the
// stored bodyA/bodyB fields swap; the axisIndex is the SatResult's, stored AS-IS (documented). The key
// legitimately CHANGES when the contact feature changes (a face sliding to a new corner set) — the documented
// "warm-start misses at sliding contacts" caveat.

#include <cstdint>
#include <vector>

#include "sim/fric.h"   // read-only: convex::SatResult/ContactManifold/FxBox/SatPair + fpx::FxBody (transitive)

namespace hf::sim {
namespace persist {

// ----- ContactKey: the deterministic integer identity of one contact POINT --------------------------------
// bodyA < bodyB ALWAYS (order-normalized — the same pair yields the same key regardless of iteration order);
// axisIndex = the convex::SatResult::axisIndex (0..14, the SAT min-pen axis that generated the manifold);
// featureIndex = the manifold point's clip-order index (0..3, its slot in the ContactManifold::points[] /
// fric::FrictionManifold::pts[]). Together (bodyA, bodyB, axisIndex, featureIndex) is a deterministic
// per-contact-point identity, stable tick-to-tick while the same SAT axis + clip corner persists.
// std430-packable as 4 x uint32 (16 bytes) — the GPU ContactKey mirror the persist_key.comp memcmp compares.
struct ContactKey {
    uint32_t bodyA        = 0;
    uint32_t bodyB        = 0;
    uint32_t axisIndex    = 0;
    uint32_t featureIndex = 0;
};

// ----- MakeContactKey(bodyAIdx, bodyBIdx, sat, pointIndex) -> ContactKey -----------------------------------
// Order-normalize the body indices (swap so bodyA < bodyB), take sat.axisIndex, featureIndex = pointIndex.
// PURE INTEGER (a compare + a swap + a copy — NO products). The swap is ONLY on the stored bodyA/bodyB fields:
// when the bodies are swapped the SAT axis was computed A->B for the ORIGINAL order, but the key is just a
// stable IDENTITY, so the axisIndex is stored as-is (documented — the key is an identity, not a geometric
// frame). The shader copies THIS body VERBATIM.
inline ContactKey MakeContactKey(uint32_t bodyAIdx, uint32_t bodyBIdx,
                                 const convex::SatResult& sat, uint32_t pointIndex) {
    ContactKey k;
    if (bodyAIdx <= bodyBIdx) { k.bodyA = bodyAIdx; k.bodyB = bodyBIdx; }
    else                      { k.bodyA = bodyBIdx; k.bodyB = bodyAIdx; }   // swap -> bodyA < bodyB
    k.axisIndex    = sat.axisIndex;
    k.featureIndex = pointIndex;
    return k;
}

// ----- ContactKeysEqual(a, b) -> bool: field-by-field equality (the cache match predicate) -----------------
// PURE INTEGER. The persistent-cache (PS2) lookup uses this to match this tick's manifold points to last
// tick's accumulated impulses.
inline bool ContactKeysEqual(const ContactKey& a, const ContactKey& b) {
    return a.bodyA == b.bodyA && a.bodyB == b.bodyB
        && a.axisIndex == b.axisIndex && a.featureIndex == b.featureIndex;
}

// ----- ContactKeyHash(k) -> uint32_t: a deterministic integer hash of the four fields --------------------
// A FIXED bit-mix over the four small uint32 fields (bodyA/bodyB < a few thousand, axisIndex < 16,
// featureIndex < 4). PURE INT32 — only shifts + xors + adds, NO products, NO int64, NO float -> MSL-native.
// Used by PS2's cache bucket lookup. The packed mix is injective over the realistic field ranges (bodyA in
// [0,4096), bodyB in [0,4096), axisIndex in [0,16), featureIndex in [0,4)) so distinct keys hash distinct;
// a final avalanche xor-shift spreads the bits for bucket distribution. The shader copies THIS body VERBATIM.
inline uint32_t ContactKeyHash(const ContactKey& k) {
    // The injective packing of the four small fields (the spec's
    // (bodyA<<20)^(bodyB<<8)^(axisIndex<<4)^featureIndex mix).
    uint32_t h = (k.bodyA << 20) ^ (k.bodyB << 8) ^ (k.axisIndex << 4) ^ k.featureIndex;
    // A fixed avalanche (xorshift-style spread; shifts + xor + add only — no products) so the hash distributes
    // across buckets while staying a deterministic function of the four fields.
    h ^= h >> 15;
    h += (h << 7);
    h ^= h >> 11;
    return h;
}

// ----- KeyMeasure: the deterministic summary over a set of contact keys ------------------------------------
// totalKeys = the total number of keys measured; distinctKeys = the number of DISTINCT keys (duplicates — the
// SAME contact re-derived — collapse); maxHashCollisions = the largest number of DISTINCT keys sharing one
// hash value (0 means every distinct key hashed uniquely — collision-free over the set). Pure integer, fixed
// scan order -> deterministic. The showcase prints + asserts (distinct contacts -> distinct keys; matching
// contacts -> equal keys).
struct KeyMeasure {
    uint32_t totalKeys         = 0;
    uint32_t distinctKeys      = 0;
    uint32_t maxHashCollisions = 0;
};

// MeasureKeys(keys): scan the key array; count total, distinct (by field tuple), and the max number of
// distinct keys colliding on a single hash. Pure integer, FIXED O(n^2) scan order (the contact sets are tiny)
// -> deterministic (two runs byte-identical).
inline KeyMeasure MeasureKeys(const std::vector<ContactKey>& keys) {
    KeyMeasure m;
    m.totalKeys = (uint32_t)keys.size();
    // Distinct keys: a key is distinct if no EARLIER key equals it (fixed ascending scan).
    for (size_t i = 0; i < keys.size(); ++i) {
        bool seen = false;
        for (size_t j = 0; j < i; ++j)
            if (ContactKeysEqual(keys[i], keys[j])) { seen = true; break; }
        if (!seen) ++m.distinctKeys;
    }
    // Max hash collisions: over the DISTINCT keys, the largest group sharing one hash, minus 1 (0 == none).
    for (size_t i = 0; i < keys.size(); ++i) {
        // Only consider the FIRST occurrence of each distinct key (the representative).
        bool firstOcc = true;
        for (size_t j = 0; j < i; ++j)
            if (ContactKeysEqual(keys[i], keys[j])) { firstOcc = false; break; }
        if (!firstOcc) continue;
        const uint32_t hi = ContactKeyHash(keys[i]);
        uint32_t sharing = 0;   // DISTINCT keys (first-occurrences) with the same hash, INCLUDING i
        for (size_t j = 0; j < keys.size(); ++j) {
            bool jFirst = true;
            for (size_t p = 0; p < j; ++p)
                if (ContactKeysEqual(keys[j], keys[p])) { jFirst = false; break; }
            if (!jFirst) continue;
            if (ContactKeyHash(keys[j]) == hi) ++sharing;
        }
        const uint32_t collisions = sharing > 0 ? sharing - 1u : 0u;
        if (collisions > m.maxHashCollisions) m.maxHashCollisions = collisions;
    }
    return m;
}

// =========================================================================================================
// Slice PS2 — THE PERSISTENT MANIFOLD CACHE. APPENDED after MeasureKeys (PS1's lines above are BYTE-FROZEN).
// PS1 built the deterministic contact feature ID (ContactKey). PS2 builds the persistent CACHE: a store that
// matches THIS tick's friction manifold points to LAST tick's accumulated contact impulses by ContactKey, so
// a matched point INHERITS its prior normal+tangent impulses (an unmatched point cold-starts at zero, a stale
// key is evicted). This is the structural store the PS3 warm-started solver reads. INTEGER bit-exact, every
// scan + store order PINNED -> deterministic (two runs byte-identical) AND bit-identical CPU<->Vulkan<->Metal.
//
// THE int64 REALITY (the FC2/CX1 lesson): the manifold it caches is built by the int64 fric::BuildFrictionPoints
// (BoxSatStable/BuildManifold + the tangent basis). DXC compiles int64 (the Vulkan path); glslc (the Metal
// HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL. So shaders/persist_cache.comp is VULKAN-SPIR-V-ONLY
// (NOT in hf_gen_msl — UNLIKE PS1's persist_key.comp which IS, because the KEY alone is pure int32) — it
// recomputes the keyed manifold (int64) + matches the cache (passed as an SSBO) per pair; the Metal
// --persist-cache runs the CPU BuildKeyedManifold + MatchCache below -> byte-identical to the Vulkan GPU result
// BY CONSTRUCTION (the fric_points.comp convention), while the Vulkan side carries the GPU==CPU memcmp proof.
// persist_cache.comp copies BuildKeyedManifold + MatchCache's bodies VERBATIM (the SAME fixed scan/match order,
// the SAME int64 manifold ops) so the GPU inherited-impulse manifold is byte-identical to the CPU reference.

// Pull the fric narrowphase + Q16.16 types PS2 uses (re-use, do NOT redefine; fric.h is read-only/byte-frozen).
using convex::fx;
using convex::FxBox;
using convex::FxBody;
using convex::SatResult;

// ----- CachedContact: one cached contact's key + its three persisted accumulated impulses -----------------
// key = the contact's PS1 ContactKey (the deterministic identity); normalImpulse/tangentImpulse1/tangentImpulse2
// = LAST tick's accumulated impulses (the fric::FrictionPoint accumulator fields). std430-packable as 4 x uint32
// (the key) + 3 x int32 (the impulses) = 7 x int32 (28 bytes) — the GPU CachedContact mirror the persist_cache
// SSBO scan reads.
struct CachedContact {
    ContactKey key;
    fx         normalImpulse   = 0;
    fx         tangentImpulse1 = 0;
    fx         tangentImpulse2 = 0;
};

// ----- PersistentCache: the store — LAST tick's accumulated impulses keyed by ContactKey ------------------
// A flat vector + a FIXED linear scan (the contact sets are tiny; PS1's ContactKeyHash is available for a
// bucket optimization, but the deterministic baseline is the fixed-order scan). UpdateCache rewrites it to
// EXACTLY this tick's contacts (so absent keys are evicted); MatchCache reads it (read-only).
struct PersistentCache {
    std::vector<CachedContact> entries;
};

// ----- KeyedFrictionManifold: a fric::FrictionManifold + the ContactKey for each of its fm.count points ----
// fric::FrictionPoint is byte-frozen and has no key field, so PS2 stores the keys in a PARALLEL array —
// keys[i] <-> fm.pts[i] for i in [0, fm.count). The unused slots stay default (a zeroed key).
struct KeyedFrictionManifold {
    fric::FrictionManifold fm;
    ContactKey             keys[4];
};

// ----- BuildKeyedManifold(bodyAIdx, bodyBIdx, bodyA, boxA, bodyB, boxB) -> KeyedFrictionManifold -----------
// The PINNED steps (the shader copies THIS body VERBATIM):
//   (1) fm = fric::BuildFrictionPoints(bodyA, boxA, bodyB, boxB) (FC2, frozen — BoxSatStable -> BuildManifold
//       -> the A->B tangent basis + zeroed accumulators);
//   (2) sat = convex::BoxSatStable(bodyA, boxA, bodyB, boxB) (the SAME SAT the manifold used — its axisIndex
//       feeds the key) — re-derived so the key carries the min-pen axis;
//   (3) for each of the fm.count points i: keys[i] = MakeContactKey(bodyAIdx, bodyBIdx, sat, i) (PS1).
// The fm's accumulators are ZERO at build (the FC2 contract; MatchCache fills the matched ones). Pure integer,
// FIXED order -> bit-identical CPU<->Vulkan<->Metal.
inline KeyedFrictionManifold BuildKeyedManifold(uint32_t bodyAIdx, uint32_t bodyBIdx,
                                                const FxBody& bodyA, const FxBox& boxA,
                                                const FxBody& bodyB, const FxBox& boxB) {
    KeyedFrictionManifold keyed;
    keyed.fm = fric::BuildFrictionPoints(bodyA, boxA, bodyB, boxB);
    const SatResult sat = convex::BoxSatStable(bodyA, boxA, bodyB, boxB);
    for (uint32_t i = 0; i < keyed.fm.count; ++i)
        keyed.keys[i] = MakeContactKey(bodyAIdx, bodyBIdx, sat, i);
    return keyed;
}

// ----- MatchCache(cache, keyed) -> mutates keyed.fm: inherit prior impulses by key ------------------------
// For each manifold point i in FIXED order [0, keyed.fm.count): scan the cache in FIXED order (entries[0..) for
// the FIRST entry whose key equals keyed.keys[i] (ContactKeysEqual); if found, copy the cached
// normalImpulse/tangentImpulse1/tangentImpulse2 into keyed.fm.pts[i] (the warm-start inheritance); if not found,
// leave them at zero (a fresh contact cold-starts — BuildKeyedManifold already zeroed them). FIXED scan order
// -> deterministic. The shader copies THIS body VERBATIM (the GPU==CPU memcmp make-or-break).
inline void MatchCache(const PersistentCache& cache, KeyedFrictionManifold& keyed) {
    for (uint32_t i = 0; i < keyed.fm.count; ++i) {
        for (size_t e = 0; e < cache.entries.size(); ++e) {
            if (ContactKeysEqual(cache.entries[e].key, keyed.keys[i])) {
                keyed.fm.pts[i].normalImpulse   = cache.entries[e].normalImpulse;
                keyed.fm.pts[i].tangentImpulse1 = cache.entries[e].tangentImpulse1;
                keyed.fm.pts[i].tangentImpulse2 = cache.entries[e].tangentImpulse2;
                break;   // the FIRST matching entry wins (fixed scan order)
            }
        }
    }
}

// ----- UpdateCache(cache, keyed) -> rewrites cache: store THIS tick's contacts, evict absent keys ----------
// After a (future PS3) solve, REPLACE the cache with exactly this tick's contacts: clear it, then for each
// point i in FIXED order append {keyed.keys[i], keyed.fm.pts[i].normalImpulse, ...}. Keys not present this tick
// are thereby EVICTED (the new cache IS exactly this tick's set). FIXED order -> deterministic. (PS2 proves the
// match + the evict; the impulses it stores are whatever the manifold carries — synthesized in PS2's showcase,
// the real solved impulses in PS3.)
inline void UpdateCache(PersistentCache& cache, const KeyedFrictionManifold& keyed) {
    cache.entries.clear();
    for (uint32_t i = 0; i < keyed.fm.count; ++i) {
        cache.entries.push_back(CachedContact{keyed.keys[i],
                                              keyed.fm.pts[i].normalImpulse,
                                              keyed.fm.pts[i].tangentImpulse1,
                                              keyed.fm.pts[i].tangentImpulse2});
    }
}

// ----- CacheMeasure: the deterministic match summary over one keyed manifold against a cache ---------------
// contacts = keyed.fm.count; matched = the number of points whose key was found in the cache (inherited);
// coldStart = the number of points NOT found (cold-started at zero) = contacts - matched. Pure integer, FIXED
// scan order -> deterministic. The showcase prints + asserts (matched + coldStart == contacts).
struct CacheMeasure {
    uint32_t contacts  = 0;
    uint32_t matched   = 0;
    uint32_t coldStart = 0;
};

// MeasureCache(cache, keyed): count, in FIXED order, how many of keyed's points have a matching cache key.
// Read-only (does NOT mutate keyed). Deterministic.
inline CacheMeasure MeasureCache(const PersistentCache& cache, const KeyedFrictionManifold& keyed) {
    CacheMeasure m;
    m.contacts = keyed.fm.count;
    for (uint32_t i = 0; i < keyed.fm.count; ++i) {
        bool found = false;
        for (size_t e = 0; e < cache.entries.size(); ++e)
            if (ContactKeysEqual(cache.entries[e].key, keyed.keys[i])) { found = true; break; }
        if (found) ++m.matched; else ++m.coldStart;
    }
    return m;
}

// =========================================================================================================
// Slice PS3 — THE WARM-STARTED CONE SOLVER. APPENDED after MeasureCache (PS1/PS2's lines above are
// BYTE-FROZEN). PS1 built the contact key, PS2 the persistent cache. PS3 is the WARM-STARTED solver: an
// ACCUMULATED sequential-impulse friction solve that seeds each contact's impulse accumulators from last
// tick's cached values (MatchCache, PS2) + RE-APPLIES them (the prime) before the Gauss-Seidel sweeps, so a
// resting stack converges in far fewer iterations and rests tighter. INTEGER bit-exact, every order PINNED ->
// bit-identical CPU<->Vulkan<->Metal.
//
// THE ACCUMULATED FORM (the proper warm-start, NOT prime + FC3-sweep): FC3's SolveFrictionImpulse is
// NON-accumulated — it computes a FRESH full impulse each sweep and applies it; that cannot be warm-started
// (re-applying last tick's impulse on top of a fresh full impulse DOUBLE-COUNTS). PS3 uses the ACCUMULATED
// form: each FrictionPoint's normalImpulse/tangentImpulse1/tangentImpulse2 are ACCUMULATORS holding the TOTAL
// impulse so far; the cone clamps the TOTAL; each sweep applies only the DELTA (newTotal - oldTotal). The
// effective masses kn/kt, the lever arms, the inertia apply are EXACTLY FC3's (reused in form). The seeded
// accumulators arrive from MatchCache (warm) or zero (cold); SolveFrictionWarm PRIMES once (applies the
// seeded total to the bodies) then runs the accumulated sweeps. After the sweeps the accumulators hold the
// TOTAL impulse — ready for UpdateCache to store for next tick.
//
// THE int64 REALITY (the FC3/FC4 lesson): the whole chain is int64. DXC compiles int64 (Vulkan); glslc
// cannot. So shaders/persist_warm.comp is VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl), single-thread over the
// small world (the fric_step.comp convention); the Metal --persist-warm runs the CPU StepWarmWorldN ->
// byte-identical to the Vulkan GPU result BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU memcmp.
// persist_warm.comp copies StepWarmWorldN VERBATIM (the same fixed orders, the same int64 ops, the same cache
// logic) so the GPU final body world is byte-identical to the CPU reference.

// Pull the fric solve + world types PS3 uses (re-use, do NOT redefine; fric.h is read-only/byte-frozen).
using convex::FxVec3;
using convex::FxMat3;
using convex::FxDot;
using convex::FxCross;
using convex::FxMat3MulVec;
using convex::FxBoxInvInertiaBody;
using convex::WorldInvInertia;
using convex::ConvexWorld;
using convex::IsDynamic;
using convex::BoxSatStable;
using convex::SatPair;
using convex::kOne;
using fric::FrictionPoint;
using fric::FrictionManifold;
using fric::FrictionStepConfig;
using fpx::FxAdd;
using fpx::FxScale;
using fpx::fxmul;
using fpx::fxdiv;

// ----- SolveFrictionWarm: the ACCUMULATED cone solver with the warm-start prime ---------------------------
// Mutates bodyA/bodyB vel+angVel AND the keyed.fm accumulators. The accumulators ARRIVE SEEDED (from MatchCache
// for a matched contact, or zero for a cold one). The PINNED steps (the shader copies THIS body VERBATIM):
//   (1) PRIME ONCE: for each point apply the seeded TOTAL impulse J = n*normalImpulse + t1*tangentImpulse1 +
//       t2*tangentImpulse2 to both bodies (re-inject last tick's converged impulse so velocities start near
//       the solved state). A zero seed primes nothing (a cold contact).
//   (2) `iters` Gauss-Seidel sweeps; per point in order, NORMAL then t1 then t2 (the FC3 order), but ACCUMULATED:
//       - NORMAL: vn = (vpB-vpA).n; kn = invMa+invMb + the FC3 angular terms; djn = fxdiv(-fxmul(kOne+
//         restitution, vn), kn); newTotal = max(0, normalImpulse + djn); applied = newTotal - normalImpulse;
//         apply J = n*applied; normalImpulse = newTotal.
//       - TANGENT t (t1 then t2): recompute vpA/vpB; vt = (vpB-vpA).t; kt = the FC3 form; djt = fxdiv(-vt, kt);
//         newTotal = clamp(tangentImpulse + djt, -fxmul(mu, normalImpulse), +fxmul(mu, normalImpulse)) (the cone
//         on the ACCUMULATED normal); applied = newTotal - tangentImpulse; apply Jt = t*applied;
//         tangentImpulse = newTotal.
// PURE INTEGER, FIXED order -> bit-identical CPU<->Vulkan<->Metal.
inline void SolveFrictionWarm(fpx::FxBody& bodyA, fpx::FxBody& bodyB,
                              const FxMat3& invIaW, const FxMat3& invIbW,
                              FrictionManifold& fm, fx restitution, fx mu, uint32_t iters) {
    if (fm.count == 0) return;

    const fx invMassA = bodyA.invMass;
    const fx invMassB = bodyB.invMass;

    // ---- (1) PRIME ONCE: re-inject the seeded TOTAL impulse at every point (the warm-start). ----
    for (uint32_t pi = 0; pi < fm.count; ++pi) {
        const FrictionPoint& fp = fm.pts[pi];
        const FxVec3 p  = fp.point;
        const FxVec3 rA = fpx::FxSub(p, bodyA.pos);
        const FxVec3 rB = fpx::FxSub(p, bodyB.pos);
        // J = n*jn + t1*jt1 + t2*jt2 (the accumulated totals seeded into the point).
        const FxVec3 J = FxAdd(FxScale(fp.normal, fp.normalImpulse),
                               FxAdd(FxScale(fp.t1, fp.tangentImpulse1),
                                     FxScale(fp.t2, fp.tangentImpulse2)));
        bodyA.vel    = fpx::FxSub(bodyA.vel, FxScale(J, invMassA));
        bodyA.angVel = fpx::FxSub(bodyA.angVel, FxMat3MulVec(invIaW, FxCross(rA, J)));
        bodyB.vel    = FxAdd(bodyB.vel, FxScale(J, invMassB));
        bodyB.angVel = FxAdd(bodyB.angVel, FxMat3MulVec(invIbW, FxCross(rB, J)));
    }

    // ---- (2) The accumulated Gauss-Seidel sweeps (apply only the DELTA each time). ----
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t pi = 0; pi < fm.count; ++pi) {
            FrictionPoint& fp = fm.pts[pi];
            const FxVec3 p  = fp.point;
            const FxVec3 n  = fp.normal;   // already A->B (BuildFrictionPoints sign-corrected it ONCE)
            const FxVec3 rA = fpx::FxSub(p, bodyA.pos);
            const FxVec3 rB = fpx::FxSub(p, bodyB.pos);

            // ---- NORMAL impulse (the FC3 effective-mass form, ACCUMULATED) ----
            {
                const FxVec3 vpA = FxAdd(bodyA.vel, FxCross(bodyA.angVel, rA));
                const FxVec3 vpB = FxAdd(bodyB.vel, FxCross(bodyB.angVel, rB));
                const fx vn = FxDot(fpx::FxSub(vpB, vpA), n);
                const FxVec3 raxn = FxCross(rA, n);
                const FxVec3 rbxn = FxCross(rB, n);
                const fx angA = FxDot(n, FxCross(FxMat3MulVec(invIaW, raxn), rA));
                const fx angB = FxDot(n, FxCross(FxMat3MulVec(invIbW, rbxn), rB));
                const fx kn = invMassA + invMassB + angA + angB;
                if (kn > 0) {
                    const fx djn = fxdiv(-fxmul(kOne + restitution, vn), kn);
                    fx newTotal = fp.normalImpulse + djn;
                    if (newTotal < 0) newTotal = 0;           // clamp the ACCUMULATED total >= 0
                    const fx applied = newTotal - fp.normalImpulse;
                    const FxVec3 J = FxScale(n, applied);      // apply only the DELTA
                    bodyA.vel    = fpx::FxSub(bodyA.vel, FxScale(J, invMassA));
                    bodyA.angVel = fpx::FxSub(bodyA.angVel, FxMat3MulVec(invIaW, FxCross(rA, J)));
                    bodyB.vel    = FxAdd(bodyB.vel, FxScale(J, invMassB));
                    bodyB.angVel = FxAdd(bodyB.angVel, FxMat3MulVec(invIbW, FxCross(rB, J)));
                    fp.normalImpulse = newTotal;
                }
            }

            // ---- TANGENT friction impulses (t1 then t2), the cone on the ACCUMULATED normal ----
            const fx coneLo = -fxmul(mu, fp.normalImpulse);   // -mu * the accumulated jn
            const fx coneHi =  fxmul(mu, fp.normalImpulse);   // +mu * the accumulated jn
            const FxVec3 tangents[2] = {fp.t1, fp.t2};
            fx* tacc[2] = {&fp.tangentImpulse1, &fp.tangentImpulse2};
            for (int ti = 0; ti < 2; ++ti) {
                const FxVec3 t = tangents[ti];
                const FxVec3 vpA = FxAdd(bodyA.vel, FxCross(bodyA.angVel, rA));
                const FxVec3 vpB = FxAdd(bodyB.vel, FxCross(bodyB.angVel, rB));
                const fx vt = FxDot(fpx::FxSub(vpB, vpA), t);
                const FxVec3 raxt = FxCross(rA, t);
                const FxVec3 rbxt = FxCross(rB, t);
                const fx angA = FxDot(t, FxCross(FxMat3MulVec(invIaW, raxt), rA));
                const fx angB = FxDot(t, FxCross(FxMat3MulVec(invIbW, rbxt), rB));
                const fx kt = invMassA + invMassB + angA + angB;
                if (kt <= 0) continue;
                const fx djt = fxdiv(-vt, kt);
                fx newTotal = *tacc[ti] + djt;
                if (newTotal < coneLo) newTotal = coneLo;          // CLAMP the ACCUMULATED tangent to the cone
                else if (newTotal > coneHi) newTotal = coneHi;
                const fx applied = newTotal - *tacc[ti];
                const FxVec3 Jt = FxScale(t, applied);              // apply only the DELTA
                bodyA.vel    = fpx::FxSub(bodyA.vel, FxScale(Jt, invMassA));
                bodyA.angVel = fpx::FxSub(bodyA.angVel, FxMat3MulVec(invIaW, FxCross(rA, Jt)));
                bodyB.vel    = FxAdd(bodyB.vel, FxScale(Jt, invMassB));
                bodyB.angVel = FxAdd(bodyB.angVel, FxMat3MulVec(invIbW, FxCross(rB, Jt)));
                *tacc[ti] = newTotal;
            }
        }
    }
}

// ----- WarmStepConfig: a thin alias of fric::FrictionStepConfig (same knobs; PS3 adds no new ones) ---------
using WarmStepConfig = fric::FrictionStepConfig;

// ----- StepWarmWorld(world, cache, cfg): ONE deterministic warm-started tick ------------------------------
// The FC4 fric::StepFrictionWorld 5-pass shell REPRODUCED with TWO changes: (1) per overlapping pair,
// BuildKeyedManifold -> MatchCache(cache, keyed) (seed the accumulators from last tick) -> SolveFrictionWarm
// (instead of SolveFrictionImpulse); (2) AFTER the solve sweeps + the position de-penetration, REBUILD the
// cache (a fresh PersistentCache accumulating every overlapping pair's UpdateCache-style entries from the
// CONVERGED accumulators) swapped into `cache` for next tick. The cache PERSISTS across ticks (passed in/out).
// ALL orders PINNED. The shader copies THIS body VERBATIM (the GPU final body world == CPU make-or-break).
inline void StepWarmWorld(ConvexWorld& world, PersistentCache& cache, const WarmStepConfig& cfg) {
    const size_t n = world.bodies.size();

    // (1) Predict-integrate every dynamic body + the per-tick velocity retention (== FC4 step 1).
    for (size_t i = 0; i < n; ++i) {
        if (IsDynamic(world.bodies[i])) {
            fpx::IntegrateBodyFull(world.bodies[i], cfg.gravity, cfg.dt);
            if (cfg.linDamp != kOne) world.bodies[i].vel = FxScale(world.bodies[i].vel, cfg.linDamp);
            if (cfg.angDamp != kOne) world.bodies[i].angVel = FxScale(world.bodies[i].angVel, cfg.angDamp);
        }
    }

    // (2) The world inverse inertias, recomputed ONCE per tick from the post-integrate orient (== FC4 step 2).
    std::vector<FxMat3> invIW(n);
    for (size_t i = 0; i < n; ++i) {
        const FxVec3 invIbody = FxBoxInvInertiaBody(world.boxes[i], world.bodies[i].invMass);
        invIW[i] = WorldInvInertia(world.bodies[i], invIbody);
    }

    // The CONVERGED keyed manifold per overlapping pair (in the FIXED i<j order) — captured so step (5) can
    // rebuild the cache from this tick's accumulators. A non-overlapping pair (count==0) is skipped + NOT
    // stored (so its key is evicted).
    struct PairAcc { size_t i, j; KeyedFrictionManifold keyed; };
    std::vector<PairAcc> pairAccs;

    // (3) Impulse solve — over the all-pairs i<j list in the FIXED order: per pair BuildKeyedManifold
    // (re-derived from the CURRENT positions) -> MatchCache(cache) (seed the accumulators from LAST tick — the
    // warm-start) -> SolveFrictionWarm with cfg.solveIters accumulated sweeps (the prime ONCE + the
    // accumulated Gauss-Seidel; the accumulator now carries state across the sweeps, so the iteration lives
    // INSIDE SolveFrictionWarm — the natural per-pair analog of FC4's stateless outer world loop). The
    // mutation is in place (later pairs see earlier updates). Skip static-static pairs.
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;  // static-static
            KeyedFrictionManifold keyed = BuildKeyedManifold((uint32_t)i, (uint32_t)j,
                                                             world.bodies[i], world.boxes[i],
                                                             world.bodies[j], world.boxes[j]);
            if (keyed.fm.count == 0) continue;
            MatchCache(cache, keyed);   // seed the accumulators from last tick's cache (warm-start)
            SolveFrictionWarm(world.bodies[i], world.bodies[j], invIW[i], invIW[j], keyed.fm,
                              cfg.restitution, cfg.mu, cfg.solveIters);   // prime once + solveIters accum sweeps
            pairAccs.push_back(PairAcc{i, j, keyed});
        }
    }

    // (4) Position de-penetration (== FC4 step 4, convex.h:907-932 reproduced VERBATIM). LINEAR only.
    for (uint32_t pit = 0; pit < cfg.posIters; ++pit) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const fx invSum = world.bodies[i].invMass + world.bodies[j].invMass;
                if (invSum == 0) continue;   // both static -> skip
                const SatResult sat = BoxSatStable(world.bodies[i], world.boxes[i],
                                                   world.bodies[j], world.boxes[j]);
                if (!sat.overlap) continue;
                FxVec3 nrm = sat.axis;
                if (FxDot(nrm, fpx::FxSub(world.bodies[j].pos, world.bodies[i].pos)) < 0)
                    nrm = FxVec3{-nrm.x, -nrm.y, -nrm.z};
                fx excess = sat.penetration - cfg.slop;
                if (excess <= 0) continue;
                const fx corrected = fxmul(excess, cfg.beta);
                const fx wi = fxdiv(world.bodies[i].invMass, invSum);
                const fx wj = kOne - wi;
                const FxVec3 ci = FxScale(nrm, fxmul(corrected, wi));
                const FxVec3 cj = FxScale(nrm, fxmul(corrected, wj));
                world.bodies[i].pos = fpx::FxSub(world.bodies[i].pos, ci);
                world.bodies[j].pos = FxAdd(world.bodies[j].pos, cj);
            }
        }
    }

    // (5) Rebuild the cache: a fresh PersistentCache accumulating every active pair's CONVERGED accumulators
    // (UpdateCache-style, in the FIXED i<j pair / point order). Keys absent this tick are thereby EVICTED.
    // Swapped into `cache` for next tick (the persistent store). The accumulators carry the warm-start.
    PersistentCache next;
    for (const PairAcc& pa : pairAccs) {
        for (uint32_t k = 0; k < pa.keyed.fm.count; ++k)
            next.entries.push_back(CachedContact{pa.keyed.keys[k],
                                                 pa.keyed.fm.pts[k].normalImpulse,
                                                 pa.keyed.fm.pts[k].tangentImpulse1,
                                                 pa.keyed.fm.pts[k].tangentImpulse2});
    }
    cache.entries.swap(next.entries);
}

// ----- StepWarmWorldN(world, cache, cfg, ticks): run `ticks` StepWarmWorld steps (the cache persists). -----
inline void StepWarmWorldN(ConvexWorld& world, PersistentCache& cache,
                           const WarmStepConfig& cfg, uint32_t ticks) {
    for (uint32_t t = 0; t < ticks; ++t) StepWarmWorld(world, cache, cfg);
}

// ----- WarmMeasure: the deterministic rest summary (max residual relative velocity over the contacts) ------
// maxResidual = the max |relative contact-point velocity along n/t1/t2| over every contact point of every
// overlapping pair (the convergence test — a tighter-converged warm stack has a SMALLER residual than the
// cold one at the same low iters); maxSpeed = the max FxLength(vel) over the dynamic bodies; contacts = the
// total contact points. Pure integer, FIXED scan order -> deterministic.
struct WarmMeasure {
    fx       maxResidual = 0;   // max |v_rel . n/t1/t2| over all contact points
    fx       maxSpeed    = 0;   // max FxLength(vel) over the dynamic bodies
    uint32_t contacts    = 0;
};

// WarmMeasure(world): scan every overlapping pair's contact points, accumulate the residual relative-velocity
// magnitude along n/t1/t2 + the dynamic body speeds. Read-only (does NOT mutate the world). Deterministic.
inline WarmMeasure MeasureWarm(const ConvexWorld& world) {
    auto absfx = [](fx v) { return v < 0 ? -v : v; };
    WarmMeasure m;
    const size_t n = world.bodies.size();
    for (size_t i = 0; i < n; ++i) {
        if (IsDynamic(world.bodies[i])) {
            const fx sp = fpx::FxLength(world.bodies[i].vel);
            if (sp > m.maxSpeed) m.maxSpeed = sp;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;
            const FrictionManifold fm = fric::BuildFrictionPoints(world.bodies[i], world.boxes[i],
                                                                  world.bodies[j], world.boxes[j]);
            for (uint32_t k = 0; k < fm.count; ++k) {
                const FrictionPoint& fp = fm.pts[k];
                const FxVec3 rA = fpx::FxSub(fp.point, world.bodies[i].pos);
                const FxVec3 rB = fpx::FxSub(fp.point, world.bodies[j].pos);
                const FxVec3 vpA = FxAdd(world.bodies[i].vel, FxCross(world.bodies[i].angVel, rA));
                const FxVec3 vpB = FxAdd(world.bodies[j].vel, FxCross(world.bodies[j].angVel, rB));
                const FxVec3 vrel = fpx::FxSub(vpB, vpA);
                const fx rn  = absfx(FxDot(vrel, fp.normal));
                const fx rt1 = absfx(FxDot(vrel, fp.t1));
                const fx rt2 = absfx(FxDot(vrel, fp.t2));
                if (rn  > m.maxResidual) m.maxResidual = rn;
                if (rt1 > m.maxResidual) m.maxResidual = rt1;
                if (rt2 > m.maxResidual) m.maxResidual = rt2;
                ++m.contacts;
            }
        }
    }
    return m;
}

// =========================================================================================================
// Slice PS4 — DETERMINISTIC SLEEPING ISLANDS (THE NEW PHYSICS). APPENDED after MeasureWarm (PS1/PS2/PS3's
// lines above are BYTE-FROZEN). PS1 built the contact key, PS2 the persistent cache, PS3 the warm-started
// solver. PS4 adds DETERMINISTIC SLEEPING: a per-body INTEGER kinetic-energy accumulator + a fixed wake/sleep
// HYSTERESIS, with contact-graph ISLAND propagation so a resting tower sleeps and wakes ATOMICALLY — the
// warm-started step SKIPS integrate + solve for sleeping islands (asleep bodies stay EXACTLY put → zero
// residual). Sleeping is the scale + stability aid every shipping engine uses, but float energy thresholds
// diverge machine-to-machine; here it is bit-identical CPU/Vulkan/Metal. INTEGER bit-exact, every order
// PINNED → bit-identical CPU↔Vulkan↔Metal AND two-run byte-identical.
//
// THE int64 REALITY (the PS3 lesson): the whole chain is int64 (the warm solve + the FxLength sqrt). DXC
// compiles int64 (the Vulkan path); glslc cannot. So shaders/persist_sleep.comp is VULKAN-SPIR-V-ONLY (NOT
// in hf_gen_msl), single-thread over the small world; the Metal --persist-sleep / --persist-wake runs the
// CPU StepWarmSleepWorldN — byte-identical to the Vulkan GPU result BY CONSTRUCTION, while the Vulkan side
// carries the GPU==CPU memcmp. persist_sleep.comp copies StepWarmSleepWorldN VERBATIM (the same fixed orders,
// the same int64 ops, the same sleep state + island propagation + skip) so the GPU final body world + sleep
// states are byte-identical to the CPU reference (the GPU==CPU memcmp is the make-or-break).

// ----- SleepState: per-body sleep bookkeeping (a PARALLEL array; FxBody is byte-frozen, no sleep field) ----
// energy      = this tick's KineticEnergy (the integer motion magnitude, the LATEST measured value);
// quietTicks  = the number of consecutive ticks the body has been QUIET (energy < sleepThreshold); reset to 0
//               when the body becomes energetic (energy > wakeThreshold). A body is a SLEEP CANDIDATE when
//               quietTicks >= sleepDelay (the hysteresis delay).
// asleep      = the FINAL per-body asleep flag after island propagation (quiet-candidate AND no awake
//               neighbour). A static body is always treated as inert/"asleep" for propagation but never moves.
// std430-packable as energy (int32) + quietTicks (uint32) + asleep (uint32) — the GPU SleepState mirror.
struct SleepState {
    fx       energy     = 0;
    uint32_t quietTicks = 0;
    bool     asleep     = false;
};

// ----- SleepConfig: the PS3 WarmStepConfig knobs + the three FIXED integer sleep-hysteresis constants -------
// warm        = the PS3 warm-step config (gravity/dt/solveIters/restitution/slop/beta/linDamp/angDamp/posIters
//               + mu) — reused VERBATIM for the awake bodies' warm solve.
// sleepThreshold = the integer KE below which a body is "quiet" (a candidate to accumulate quietTicks).
// wakeThreshold  = the integer KE above which a sleeping/quiet body WAKES (>= sleepThreshold — a hysteresis
//                  BAND so a body resting right at the threshold does not flicker awake/asleep each tick).
// sleepDelay     = the number of consecutive quiet ticks before a body becomes a sleep candidate.
// All FIXED integer constants → deterministic, bit-identical cross-backend.
struct SleepConfig {
    WarmStepConfig warm;
    // The thresholds are sized ABOVE the warm solver's documented steady-state contact JITTER (the PS3
    // "accumulated GS residual is not analytic / within-a-band" reality — a resting stack churns a small
    // per-tick velocity, ~0.3–0.7 unit/s for the tower scene, even though it rests POSITIONALLY). So "quiet"
    // means "below the resting jitter band", not "literally zero velocity": a body whose motion stays under
    // sleepThreshold for sleepDelay ticks sleeps (and is then FROZEN → exactly zero residual). A genuinely
    // disturbed body (a thrown box, KE ~6 unit/s) far exceeds wakeThreshold and wakes its island. FIXED
    // integer constants → a deterministic heuristic (the spec's documented honesty), bit-identical CPU/GPU.
    fx             sleepThreshold = kOne;        // ~1.0 unit/s — above the resting-jitter band, "quiet"
    fx             wakeThreshold  = (fx)(2 * (int)kOne);  // ~2.0 unit/s — the wake band top (>= sleepThreshold)
    uint32_t       sleepDelay     = 30;          // ~0.5 s at 60 Hz of sustained quiet before sleeping
};

// ----- KineticEnergy(b): the deterministic integer motion magnitude (the sleep measure) --------------------
// FxLength(vel) + FxLength(angVel) — an L2 motion sum in Q16.16 (the FxLength reused, int64 sqrt). A static /
// non-dynamic body reads its stored vel/angVel (normally zero). PURE INTEGER, FIXED ops → deterministic.
inline fx KineticEnergy(const fpx::FxBody& b) {
    return fpx::FxLength(b.vel) + fpx::FxLength(b.angVel);
}

// ----- UpdateQuietTicks(state, energy, cfg): the per-body hysteresis timer update -------------------------
// Stores energy; then: if energy > wakeThreshold → WAKE (quietTicks = 0, asleep cleared); else if energy <
// sleepThreshold → quiet (quietTicks++ saturating); else (in the hysteresis BAND between the two) → HOLD the
// current quietTicks (no increment, no reset) so a body lingering in the band neither flickers awake nor
// accrues toward sleep. FIXED integer compares → deterministic.
inline void UpdateQuietTicks(SleepState& s, fx energy, const SleepConfig& cfg) {
    s.energy = energy;
    if (energy > cfg.wakeThreshold) {
        s.quietTicks = 0;
        s.asleep     = false;
    } else if (energy < cfg.sleepThreshold) {
        if (s.quietTicks != 0xFFFFFFFFu) ++s.quietTicks;   // saturating increment
    }
    // else: in the [sleepThreshold, wakeThreshold] band → hold quietTicks (the no-flicker band).
}

// ----- PropagateWake: build the contact adjacency + propagate WAKEFULNESS to a fixed point ----------------
// THE ISLAND PROPAGATION (fixed-order, union-find-free): a quiet dynamic body must STAY AWAKE if ANY body it
// contacts is awake (else a tower would half-sleep). The contact adjacency is THIS tick's all-pairs
// BoxSatStable overlaps (the SAME overlap test the solve + de-pen use). A STATIC body is NEVER an island-waker
// (it is inert — the floor must not keep the tower awake) and is reported asleep=true. The per-body seed
// "awake" = (dynamic AND quietTicks < sleepDelay). Then propagate: repeat (a fixed bound of `n` passes — a
// chain of n bodies needs at most n-1 hops; bound by n is safe) in FIXED body order: a DYNAMIC body becomes
// awake if any ADJACENT body is awake. Statics never become awake (they stay inert). Finally asleep =
// (dynamic AND NOT awake) for dynamics, true for statics. Deterministic (fixed adjacency + fixed pass bound +
// fixed body order). Writes the per-body asleep flag into sleep[].
inline void PropagateWake(const ConvexWorld& world, std::vector<SleepState>& sleep, uint32_t sleepDelay) {
    const size_t n = world.bodies.size();
    // (1) The contact adjacency from this tick's overlapping pairs (FIXED i<j scan, the BoxSatStable overlap).
    std::vector<uint8_t> adj(n * n, 0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;  // static-static
            const SatResult sat = BoxSatStable(world.bodies[i], world.boxes[i],
                                               world.bodies[j], world.boxes[j]);
            if (sat.overlap) { adj[i * n + j] = 1; adj[j * n + i] = 1; }
        }
    }
    // (2) Seed: a DYNAMIC body is awake iff its own quietTicks < sleepDelay (not yet a sleep candidate). A
    // static body is inert (never awake → never an island-waker → the floor doesn't keep the tower awake).
    std::vector<uint8_t> awake(n, 0);
    for (size_t i = 0; i < n; ++i)
        if (IsDynamic(world.bodies[i]) && sleep[i].quietTicks < sleepDelay) awake[i] = 1;
    // (3) Propagate wakefulness to a fixed point — a fixed bound of `n` passes in FIXED body order; a DYNAMIC
    // body becomes awake if any adjacent body is awake. (Statics never flip — they stay inert.) Stop early if a
    // pass changes nothing (deterministic either way; the bound guarantees full propagation through a chain).
    for (size_t pass = 0; pass < n; ++pass) {
        bool changed = false;
        for (size_t i = 0; i < n; ++i) {
            if (!IsDynamic(world.bodies[i]) || awake[i]) continue;
            for (size_t j = 0; j < n; ++j) {
                if (adj[i * n + j] && awake[j]) { awake[i] = 1; changed = true; break; }
            }
        }
        if (!changed) break;
    }
    // (4) The final per-body asleep flag: a dynamic body sleeps iff NOT awake; a static body is always inert.
    for (size_t i = 0; i < n; ++i) {
        if (IsDynamic(world.bodies[i])) sleep[i].asleep = (awake[i] == 0);
        else                            sleep[i].asleep = true;   // static → inert (never moves)
    }
}

// ----- StepWarmSleepWorld(world, cache, sleep, cfg): ONE warm-started tick WITH deterministic sleeping ------
// The PS3 StepWarmWorld tick, extended with sleeping (the spec's 5-step recipe). ALL orders PINNED. The shader
// copies THIS body VERBATIM (the GPU final body world + sleep states == CPU make-or-break):
//   (0) sleep[] is sized to the body count (each entry persists across ticks, passed in/out).
//   (1) Compute each body's KineticEnergy from the PRE-integrate state + update its quietTicks (hysteresis).
//   (2) Build the contact adjacency (all-pairs overlap) + PropagateWake → the per-body asleep flag.
//   (3) Predict-integrate ONLY AWAKE dynamic bodies (asleep bodies stay EXACTLY put → zero drift).
//   (4) The warm impulse solve + the position de-penetration run ONLY over pairs with at least one AWAKE
//       dynamic body (a fully-asleep pair is skipped — its cached impulses persist untouched).
//   (5) Rebuild the cache over the SOLVED (active) pairs; the sleeping pairs keep their prior cache entries
//       (so a sleeping island's warm-start survives intact for the moment it wakes).
inline void StepWarmSleepWorld(ConvexWorld& world, PersistentCache& cache,
                               std::vector<SleepState>& sleep, const SleepConfig& cfg) {
    const size_t n = world.bodies.size();
    if (sleep.size() != n) sleep.assign(n, SleepState{});
    const WarmStepConfig& w = cfg.warm;

    // (1) Per-body KineticEnergy (PRE-integrate) + the hysteresis quietTicks update.
    for (size_t i = 0; i < n; ++i)
        UpdateQuietTicks(sleep[i], KineticEnergy(world.bodies[i]), cfg);

    // (2) The contact adjacency + island wakefulness propagation → the per-body asleep flag.
    PropagateWake(world, sleep, cfg.sleepDelay);

    // (2b) FREEZE every asleep dynamic body: zero its vel + angVel so it carries EXACTLY zero residual motion
    // (a real engine parks a sleeping body at zero velocity; this makes "zero residual" literally exact + keeps
    // a slept body quiet → it stays asleep). Asleep bodies are then skipped by integrate (step 3) so position
    // is frozen too. Statics are already zero-velocity. FIXED order, pure copy → deterministic.
    for (size_t i = 0; i < n; ++i) {
        if (IsDynamic(world.bodies[i]) && sleep[i].asleep) {
            world.bodies[i].vel    = FxVec3{0, 0, 0};
            world.bodies[i].angVel = FxVec3{0, 0, 0};
        }
    }

    // (3) Predict-integrate ONLY AWAKE dynamic bodies (== PS3 step 1, gated by !asleep). Asleep bodies keep
    // their EXACT pos/vel(=0)/orient/angVel(=0) (zero drift — the deterministic-rest headline).
    for (size_t i = 0; i < n; ++i) {
        if (IsDynamic(world.bodies[i]) && !sleep[i].asleep) {
            fpx::IntegrateBodyFull(world.bodies[i], w.gravity, w.dt);
            if (w.linDamp != kOne) world.bodies[i].vel = FxScale(world.bodies[i].vel, w.linDamp);
            if (w.angDamp != kOne) world.bodies[i].angVel = FxScale(world.bodies[i].angVel, w.angDamp);
        }
    }

    // (= PS3 step 2) The world inverse inertias, recomputed ONCE per tick from the post-integrate orient.
    std::vector<FxMat3> invIW(n);
    for (size_t i = 0; i < n; ++i) {
        const FxVec3 invIbody = FxBoxInvInertiaBody(world.boxes[i], world.bodies[i].invMass);
        invIW[i] = WorldInvInertia(world.bodies[i], invIbody);
    }

    // A pair is ACTIVE iff at least one of its bodies is an AWAKE dynamic body (a fully-asleep / static-only
    // pair is skipped — its cached impulses persist). FIXED i<j order.
    auto pairActive = [&](size_t i, size_t j) {
        const bool ai = IsDynamic(world.bodies[i]) && !sleep[i].asleep;
        const bool aj = IsDynamic(world.bodies[j]) && !sleep[j].asleep;
        return ai || aj;
    };

    struct PairAcc { size_t i, j; KeyedFrictionManifold keyed; };
    std::vector<PairAcc> pairAccs;

    // (4a = PS3 step 3) Impulse solve over the active pairs only (FIXED i<j order, in place).
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;  // static-static
            if (!pairActive(i, j)) continue;                                             // fully-asleep → skip
            KeyedFrictionManifold keyed = BuildKeyedManifold((uint32_t)i, (uint32_t)j,
                                                             world.bodies[i], world.boxes[i],
                                                             world.bodies[j], world.boxes[j]);
            if (keyed.fm.count == 0) continue;
            MatchCache(cache, keyed);   // seed the accumulators from last tick's cache (warm-start)
            SolveFrictionWarm(world.bodies[i], world.bodies[j], invIW[i], invIW[j], keyed.fm,
                              w.restitution, w.mu, w.solveIters);
            pairAccs.push_back(PairAcc{i, j, keyed});
        }
    }

    // (4b = PS3 step 4) Position de-penetration — over the active pairs only, posIters sweeps, FIXED i<j.
    for (uint32_t pit = 0; pit < w.posIters; ++pit) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const fx invSum = world.bodies[i].invMass + world.bodies[j].invMass;
                if (invSum == 0) continue;          // both static → skip
                if (!pairActive(i, j)) continue;    // fully-asleep → skip (both stay exactly put)
                const SatResult sat = BoxSatStable(world.bodies[i], world.boxes[i],
                                                   world.bodies[j], world.boxes[j]);
                if (!sat.overlap) continue;
                FxVec3 nrm = sat.axis;
                if (FxDot(nrm, fpx::FxSub(world.bodies[j].pos, world.bodies[i].pos)) < 0)
                    nrm = FxVec3{-nrm.x, -nrm.y, -nrm.z};
                fx excess = sat.penetration - w.slop;
                if (excess <= 0) continue;
                const fx corrected = fxmul(excess, w.beta);
                // An ASLEEP body must NOT be pushed (zero drift); split the correction by invMass but ZERO the
                // share of an asleep/static body so only the awake partner moves (the awake body separates).
                const bool moveI = IsDynamic(world.bodies[i]) && !sleep[i].asleep;
                const bool moveJ = IsDynamic(world.bodies[j]) && !sleep[j].asleep;
                fx wi = fxdiv(world.bodies[i].invMass, invSum);
                fx wj = kOne - wi;
                if (!moveI && moveJ) { wi = 0; wj = kOne; }
                else if (moveI && !moveJ) { wi = kOne; wj = 0; }
                else if (!moveI && !moveJ) continue;   // neither moves (both asleep/static) → skip
                const FxVec3 ci = FxScale(nrm, fxmul(corrected, wi));
                const FxVec3 cj = FxScale(nrm, fxmul(corrected, wj));
                if (moveI) world.bodies[i].pos = fpx::FxSub(world.bodies[i].pos, ci);
                if (moveJ) world.bodies[j].pos = FxAdd(world.bodies[j].pos, cj);
            }
        }
    }

    // (5) Rebuild the cache: the SLEEPING pairs keep their prior cache entries (so a sleeping island's
    // warm-start survives), the ACTIVE pairs store their freshly-converged accumulators. Build the new cache
    // as: first carry over every cache entry whose pair was NOT active this tick (its key persists untouched),
    // then append the active pairs' converged accumulators. FIXED order → deterministic.
    PersistentCache next;
    // (5a) Carry over the entries of pairs that were NOT solved this tick (sleeping / inactive). An entry's
    // pair is (key.bodyA, key.bodyB); it was active iff pairActive(bodyA, bodyB) (those are re-stored in 5b).
    for (const CachedContact& e : cache.entries) {
        const size_t a = e.key.bodyA, b = e.key.bodyB;
        bool wasActive = (a < n && b < n) && pairActive(a, b);
        if (!wasActive) next.entries.push_back(e);   // persist the sleeping pair's warm-start untouched
    }
    // (5b) Append the active pairs' converged accumulators (UpdateCache-style, FIXED i<j / point order).
    for (const PairAcc& pa : pairAccs) {
        for (uint32_t k = 0; k < pa.keyed.fm.count; ++k)
            next.entries.push_back(CachedContact{pa.keyed.keys[k],
                                                 pa.keyed.fm.pts[k].normalImpulse,
                                                 pa.keyed.fm.pts[k].tangentImpulse1,
                                                 pa.keyed.fm.pts[k].tangentImpulse2});
    }
    cache.entries.swap(next.entries);
}

// ----- StepWarmSleepWorldN(world, cache, sleep, cfg, ticks): run `ticks` sleeping warm steps. ---------------
inline void StepWarmSleepWorldN(ConvexWorld& world, PersistentCache& cache,
                                std::vector<SleepState>& sleep, const SleepConfig& cfg, uint32_t ticks) {
    for (uint32_t t = 0; t < ticks; ++t) StepWarmSleepWorld(world, cache, sleep, cfg);
}

// ----- SleepMeasure: the deterministic sleep summary -------------------------------------------------------
// asleepCount  = the number of DYNAMIC bodies asleep; awakeCount = the number of dynamic bodies awake;
// maxSpeed     = the max KineticEnergy over the AWAKE dynamic bodies (an all-asleep world reports 0 — the
//                zero-residual headline; asleep bodies don't move, so the awake max is the true residual);
// dynamicCount = the number of dynamic bodies. Pure integer, FIXED scan order → deterministic.
struct SleepMeasure {
    uint32_t asleepCount  = 0;
    uint32_t awakeCount   = 0;
    fx       maxSpeed     = 0;
    uint32_t dynamicCount = 0;
};

// MeasureSleep(world, sleep): scan the dynamic bodies; count asleep/awake + the max awake KineticEnergy.
// Read-only. Deterministic.
inline SleepMeasure MeasureSleep(const ConvexWorld& world, const std::vector<SleepState>& sleep) {
    SleepMeasure m;
    const size_t n = world.bodies.size();
    for (size_t i = 0; i < n; ++i) {
        if (!IsDynamic(world.bodies[i])) continue;
        ++m.dynamicCount;
        const bool asleep = (i < sleep.size()) ? sleep[i].asleep : false;
        if (asleep) {
            ++m.asleepCount;
        } else {
            ++m.awakeCount;
            const fx ke = KineticEnergy(world.bodies[i]);
            if (ke > m.maxSpeed) m.maxSpeed = ke;
        }
    }
    return m;
}

}  // namespace persist
}  // namespace hf::sim
