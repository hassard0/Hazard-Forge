// Slice PS1 — Deterministic Persistent Contacts: THE CONTACT FEATURE ID (the integer BEACHHEAD of
// FLAGSHIP #21: DETERMINISTIC WARM-STARTED CONTACT CACHING + SLEEPING ISLANDS, hf::sim::persist). The
// integer core (engine/sim/persist.h) that the GPU shaders/persist_key.comp.hlsl copies VERBATIM + proves
// bit-identical. PURE INT32 (no Q16.16 products — only compares, shifts, xors) -> MSL-native, a TRUE GPU
// pass on BOTH backends. Pure CPU (header-only, hf_core), ASan-eligible. persist.h #includes sim/fric.h
// read-only (transitively convex + fpx).
//
// What this test PINS (the contracts the GPU persist_key.comp + the GPU==CPU proof build on):
//   * MakeContactKey order-normalizes the body indices: MakeContactKey(i, j, sat, p) ==
//     MakeContactKey(j, i, sat, p) field-for-field (the same pair yields the same key regardless of
//     iteration order — bodyA < bodyB ALWAYS).
//   * Distinct contact features (different pair / axis / feature index) get DISTINCT keys.
//   * Identical features get EQUAL keys (ContactKeysEqual) + EQUAL hash (a re-derived "next tick" point that
//     keeps the same pair + SAT axis + clip corner re-derives the SAME key).
//   * ContactKeyHash is deterministic (two calls equal) + collision-light over the showcase contacts.
//   * MeasureKeys: deterministic total/distinct/maxCollision summary; two runs byte-identical.
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests.
#include "sim/persist.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace persist = hf::sim::persist;
namespace convex = hf::sim::convex;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A small SatResult builder (only .axisIndex matters to the key; the rest rides along).
static convex::SatResult MakeSat(uint32_t axisIndex) {
    convex::SatResult s;
    s.overlap = true;
    s.axisIndex = axisIndex;
    s.penetration = 0;
    s.axis = convex::FxVec3{0, 0, 0};
    return s;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= order-normalization: (i,j) == (j,i) =================
    {
        const convex::SatResult sat = MakeSat(2);
        const persist::ContactKey kij = persist::MakeContactKey(3, 7, sat, 1);
        const persist::ContactKey kji = persist::MakeContactKey(7, 3, sat, 1);
        // bodyA < bodyB ALWAYS.
        check(kij.bodyA == 3 && kij.bodyB == 7, "order-normalize: (3,7) -> bodyA=3, bodyB=7");
        check(kji.bodyA == 3 && kji.bodyB == 7, "order-normalize: (7,3) -> bodyA=3, bodyB=7 (swapped)");
        check(kij.axisIndex == 2 && kij.featureIndex == 1, "key carries axisIndex+featureIndex");
        // The two keys are equal field-for-field (the same pair identity regardless of caller order).
        check(persist::ContactKeysEqual(kij, kji), "order-normalize: (i,j) key == (j,i) key");
        check(persist::ContactKeyHash(kij) == persist::ContactKeyHash(kji),
              "order-normalize: equal keys -> equal hash");
    }

    // ================= equal body indices (a self-pair is allowed; no swap) =================
    {
        const convex::SatResult sat = MakeSat(5);
        const persist::ContactKey k = persist::MakeContactKey(4, 4, sat, 0);
        check(k.bodyA == 4 && k.bodyB == 4, "equal indices: no swap, both 4");
    }

    // ================= distinct features -> distinct keys =================
    {
        const persist::ContactKey base = persist::MakeContactKey(1, 2, MakeSat(0), 0);
        const persist::ContactKey diffPair = persist::MakeContactKey(1, 3, MakeSat(0), 0);
        const persist::ContactKey diffAxis = persist::MakeContactKey(1, 2, MakeSat(4), 0);
        const persist::ContactKey diffFeat = persist::MakeContactKey(1, 2, MakeSat(0), 3);
        check(!persist::ContactKeysEqual(base, diffPair), "distinct pair -> distinct key");
        check(!persist::ContactKeysEqual(base, diffAxis), "distinct axis -> distinct key");
        check(!persist::ContactKeysEqual(base, diffFeat), "distinct feature -> distinct key");
    }

    // ================= identical features -> equal keys + equal hash (the warm-start match) =================
    {
        // Tick t: a contact at pair (5,9), SAT axis 1, clip corner 2. Tick t+1: the SAME contact re-derived.
        const persist::ContactKey kT  = persist::MakeContactKey(5, 9, MakeSat(1), 2);
        const persist::ContactKey kT1 = persist::MakeContactKey(5, 9, MakeSat(1), 2);
        check(persist::ContactKeysEqual(kT, kT1), "next-tick same feature -> equal key (cache match)");
        check(persist::ContactKeyHash(kT) == persist::ContactKeyHash(kT1),
              "next-tick same feature -> equal hash");
    }

    // ================= ContactKeyHash determinism =================
    {
        const persist::ContactKey k = persist::MakeContactKey(11, 23, MakeSat(9), 3);
        check(persist::ContactKeyHash(k) == persist::ContactKeyHash(k), "hash deterministic (two calls equal)");
    }

    // ================= ContactKeyHash collision-light over a representative contact set =================
    {
        // The showcase-style contact set: a handful of pairs, each contributing 1..4 contact points across
        // SAT axes 0..14. Distinct keys should map to distinct hashes (no collision in this small set).
        std::vector<persist::ContactKey> keys;
        for (uint32_t pair = 0; pair < 12; ++pair) {
            const uint32_t bodyA = pair * 2u;
            const uint32_t bodyB = pair * 2u + 1u;
            const uint32_t axis = pair % 15u;
            for (uint32_t pt = 0; pt < 4u; ++pt)
                keys.push_back(persist::MakeContactKey(bodyA, bodyB, MakeSat(axis), pt));
        }
        std::set<uint32_t> distinctKeys;   // by (bodyA,bodyB,axis,feature) tuple via a packed compare
        std::set<uint32_t> distinctHashes;
        // Count distinct keys (field tuple) and distinct hashes; for distinct keys the hashes must be distinct.
        for (size_t i = 0; i < keys.size(); ++i) {
            // A unique injective fingerprint of the four small fields (NOT the hash) for the distinct count.
            const uint32_t fp = (keys[i].bodyA << 20) ^ (keys[i].bodyB << 8)
                              ^ (keys[i].axisIndex << 4) ^ keys[i].featureIndex;
            distinctKeys.insert(fp);
            distinctHashes.insert(persist::ContactKeyHash(keys[i]));
        }
        check(distinctHashes.size() == distinctKeys.size(),
              "hash collision-light: distinct keys -> distinct hashes over the showcase set");
    }

    // ================= MeasureKeys: deterministic summary + two runs byte-identical =================
    {
        std::vector<persist::ContactKey> keys;
        // Two pairs with overlapping features: pair (0,1) axis 0 pts 0..3, and a DUPLICATE of one of them.
        for (uint32_t pt = 0; pt < 4u; ++pt)
            keys.push_back(persist::MakeContactKey(0, 1, MakeSat(0), pt));
        // A duplicate of (0,1,axis0,pt2) — same feature, so it should NOT raise distinctKeys.
        keys.push_back(persist::MakeContactKey(1, 0, MakeSat(0), 2));   // swapped caller order, same key
        const persist::KeyMeasure m1 = persist::MeasureKeys(keys);
        const persist::KeyMeasure m2 = persist::MeasureKeys(keys);
        check(m1.totalKeys == 5u, "MeasureKeys: totalKeys counts every key");
        check(m1.distinctKeys == 4u, "MeasureKeys: distinctKeys collapses the duplicate (4 distinct)");
        check(std::memcmp(&m1, &m2, sizeof(persist::KeyMeasure)) == 0,
              "MeasureKeys: two runs byte-identical (deterministic)");
    }

    // =========================================================================================================
    // Slice PS2 — THE PERSISTENT MANIFOLD CACHE. The cache matches THIS tick's keyed friction manifold to LAST
    // tick's accumulated impulses by ContactKey: a matched point INHERITS its prior impulses, an unmatched point
    // cold-starts at zero, a stale key is evicted. Pure CPU, FIXED scan + store order -> deterministic. The
    // shaders/persist_cache.comp copies BuildKeyedManifold + MatchCache VERBATIM (the GPU==CPU memcmp proof).
    namespace fric = hf::sim::fric;
    namespace fpx = hf::sim::fpx;
    {
        // A small deterministic two-box pair (a unit box overlapping a unit box on +X) -> a face manifold.
        const fpx::FxQuat qI{0, 0, 0, convex::kOne};
        auto bodyAt = [&](convex::fx x, convex::fx y, convex::fx z) {
            fpx::FxBody b; b.pos = {x, y, z}; b.orient = qI; return b;
        };
        auto fi = [&](int v) { return (convex::fx)(v * (int)convex::kOne); };
        const convex::FxBox kUnit{convex::FxVec3{convex::kOne, convex::kOne, convex::kOne}};
        // Two unit boxes 1 unit apart on X (half-extent 1 each -> they overlap by 1).
        const fpx::FxBody bA = bodyAt(0, 0, 0);
        const fpx::FxBody bB = bodyAt(fi(1), 0, 0);

        // ================= BuildKeyedManifold: keys parallel to the manifold points =================
        const persist::KeyedFrictionManifold keyed0 = persist::BuildKeyedManifold(0, 1, bA, kUnit, bB, kUnit);
        check(keyed0.fm.count > 0, "BuildKeyedManifold: the overlapping pair yields >=1 contact point");
        // The keys must equal MakeContactKey(0,1,sat,i) for each point i (parallel array).
        const convex::SatResult satRef = convex::BoxSatStable(bA, kUnit, bB, kUnit);
        bool keysParallel = true;
        for (uint32_t i = 0; i < keyed0.fm.count; ++i) {
            const persist::ContactKey expect = persist::MakeContactKey(0, 1, satRef, i);
            if (!persist::ContactKeysEqual(keyed0.keys[i], expect)) keysParallel = false;
        }
        check(keysParallel, "BuildKeyedManifold: keys[i] == MakeContactKey(0,1,sat,i) (parallel to pts)");
        // The fm accumulators are ZERO at build (the FC2 contract).
        bool accZero = true;
        for (uint32_t i = 0; i < keyed0.fm.count; ++i)
            if (keyed0.fm.pts[i].normalImpulse != 0 || keyed0.fm.pts[i].tangentImpulse1 != 0 ||
                keyed0.fm.pts[i].tangentImpulse2 != 0) accZero = false;
        check(accZero, "BuildKeyedManifold: accumulators zeroed at build");

        // ================= MatchCache: a matching key inherits, a non-matching key cold-starts =================
        {
            // Seed a cache with the FIRST point's key carrying synthesized impulses + a BOGUS key.
            persist::PersistentCache cache;
            const convex::fx kN = fi(7), kT1 = fi(3), kT2 = -fi(2);
            cache.entries.push_back({keyed0.keys[0], kN, kT1, kT2});
            cache.entries.push_back({persist::MakeContactKey(99, 100, MakeSat(13), 3), fi(5), fi(5), fi(5)});

            persist::KeyedFrictionManifold keyed = persist::BuildKeyedManifold(0, 1, bA, kUnit, bB, kUnit);
            persist::MatchCache(cache, keyed);
            // Point 0 matched -> inherited the exact cached impulses.
            check(keyed.fm.pts[0].normalImpulse == kN && keyed.fm.pts[0].tangentImpulse1 == kT1 &&
                  keyed.fm.pts[0].tangentImpulse2 == kT2, "MatchCache: matched point inherits cached impulses");
            // Every OTHER point had no cache entry -> cold-started at zero.
            bool othersZero = true;
            for (uint32_t i = 1; i < keyed.fm.count; ++i)
                if (keyed.fm.pts[i].normalImpulse != 0 || keyed.fm.pts[i].tangentImpulse1 != 0 ||
                    keyed.fm.pts[i].tangentImpulse2 != 0) othersZero = false;
            check(othersZero, "MatchCache: a non-matching key cold-starts at zero");
        }

        // ================= store-then-match round-trips =================
        {
            // Build a keyed manifold, synthesize per-point impulses, UpdateCache, then a FRESH MatchCache returns
            // exactly the stored impulses.
            persist::KeyedFrictionManifold stored = persist::BuildKeyedManifold(0, 1, bA, kUnit, bB, kUnit);
            for (uint32_t i = 0; i < stored.fm.count; ++i) {
                stored.fm.pts[i].normalImpulse   = fi((int)i + 1);
                stored.fm.pts[i].tangentImpulse1 = fi((int)i + 10);
                stored.fm.pts[i].tangentImpulse2 = -fi((int)i + 5);
            }
            persist::PersistentCache cache;
            persist::UpdateCache(cache, stored);
            check(cache.entries.size() == stored.fm.count,
                  "UpdateCache: the new cache holds exactly this tick's contacts");
            persist::KeyedFrictionManifold fresh = persist::BuildKeyedManifold(0, 1, bA, kUnit, bB, kUnit);
            persist::MatchCache(cache, fresh);
            bool roundTrip = true;
            for (uint32_t i = 0; i < fresh.fm.count; ++i)
                if (fresh.fm.pts[i].normalImpulse != stored.fm.pts[i].normalImpulse ||
                    fresh.fm.pts[i].tangentImpulse1 != stored.fm.pts[i].tangentImpulse1 ||
                    fresh.fm.pts[i].tangentImpulse2 != stored.fm.pts[i].tangentImpulse2) roundTrip = false;
            check(roundTrip, "store-then-match round-trips: MatchCache after UpdateCache returns the stored impulses");
        }

        // ================= UpdateCache evicts a stale key =================
        {
            // Tick 1: cache holds pair (0,1)'s contacts PLUS a synthesized stale key from a removed pair.
            persist::KeyedFrictionManifold t1 = persist::BuildKeyedManifold(0, 1, bA, kUnit, bB, kUnit);
            for (uint32_t i = 0; i < t1.fm.count; ++i) t1.fm.pts[i].normalImpulse = fi(4);
            persist::PersistentCache cache;
            persist::UpdateCache(cache, t1);
            // Inject a stale entry (a key NOT present in tick 2's manifold).
            const persist::ContactKey staleKey = persist::MakeContactKey(50, 51, MakeSat(9), 2);
            cache.entries.push_back({staleKey, fi(8), fi(8), fi(8)});
            const size_t beforeEvict = cache.entries.size();
            check(beforeEvict == (size_t)t1.fm.count + 1u, "UpdateCache(evict): pre-evict cache has the stale key");
            // Tick 2: the SAME pair -> the SAME keys -> UpdateCache REPLACES the cache with tick-2's set only.
            persist::KeyedFrictionManifold t2 = persist::BuildKeyedManifold(0, 1, bA, kUnit, bB, kUnit);
            persist::UpdateCache(cache, t2);
            check(cache.entries.size() == t2.fm.count,
                  "UpdateCache: evicts the stale key (the new cache is exactly this tick's set)");
            bool staleGone = true;
            for (const persist::CachedContact& c : cache.entries)
                if (persist::ContactKeysEqual(c.key, staleKey)) staleGone = false;
            check(staleGone, "UpdateCache: the stale key is no longer in the cache");
        }

        // ================= MatchCache measure + two runs byte-identical =================
        {
            persist::KeyedFrictionManifold keyed = persist::BuildKeyedManifold(0, 1, bA, kUnit, bB, kUnit);
            persist::PersistentCache cache;
            // Cache: match point 0, miss the rest.
            cache.entries.push_back({keyed.keys[0], fi(1), fi(2), fi(3)});
            const persist::CacheMeasure m1 = persist::MeasureCache(cache, keyed);
            const persist::CacheMeasure m2 = persist::MeasureCache(cache, keyed);
            check(m1.matched == 1u, "MeasureCache: exactly one point matched the cache");
            check(m1.coldStart == keyed.fm.count - 1u, "MeasureCache: the remaining points cold-start");
            check(std::memcmp(&m1, &m2, sizeof(persist::CacheMeasure)) == 0,
                  "MeasureCache: two runs byte-identical (deterministic)");
        }
    }

    if (g_fail == 0) std::printf("persist_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
