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

}  // namespace persist
}  // namespace hf::sim
