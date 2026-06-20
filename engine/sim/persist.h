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

}  // namespace persist
}  // namespace hf::sim
