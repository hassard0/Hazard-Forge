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

    // =========================================================================================================
    // Slice PS3 — THE WARM-STARTED CONE SOLVER. The ACCUMULATED sequential-impulse friction solve that seeds
    // each contact's impulse accumulators from last tick's cached values (MatchCache) + re-applies them (the
    // prime) before the Gauss-Seidel sweeps, so a resting stack converges in fewer iterations + rests tighter.
    // The make-or-break controls (NOT "==FC3" — accumulated GS is a different algorithm): (a) WARM-START
    // BENEFIT — warm residual < cold residual at a fixed low iteration count; (b) CONSISTENCY — warm == cold
    // byte-identical at a high iteration count (the unique fixed point). Pure CPU, FIXED orders -> deterministic.
    {
        const fpx::FxQuat qI{0, 0, 0, convex::kOne};
        const convex::fx kOne = convex::kOne;
        auto fi = [&](int v) { return (convex::fx)(v * (int)convex::kOne); };
        auto fh = [&](int num, int den) { return (convex::fx)((int64_t)num * (int)convex::kOne / den); };
        const convex::fx kGravY = (convex::fx)(-9.8 * (double)kOne - 0.5);

        auto makeBody = [&](convex::fx x, convex::fx y, convex::fx z, bool dyn) {
            fpx::FxBody b;
            b.pos = {x, y, z};
            b.orient = qI;
            b.invMass = dyn ? kOne : 0;
            b.flags   = dyn ? fpx::kFlagDynamic : 0u;
            b.vel = {0, 0, 0};
            b.angVel = {0, 0, 0};
            return b;
        };
        const convex::FxBox kFloor{convex::FxVec3{fi(8), kOne, fi(8)}};
        const convex::FxBox kSlab{convex::FxVec3{fi(3) / 2, kOne / 2, fi(3) / 2}};   // 3 x 1 x 3
        auto buildStack = [&]() {
            convex::ConvexWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, false)); w.boxes.push_back(kFloor);
            w.bodies.push_back(makeBody(0, fi(1) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            w.bodies.push_back(makeBody(0, fi(2) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            w.bodies.push_back(makeBody(0, fi(3) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            return w;
        };

        persist::WarmStepConfig kCfg;
        kCfg.gravity     = convex::FxVec3{0, kGravY, 0};
        kCfg.dt          = kOne / 60;
        kCfg.solveIters  = 20;
        kCfg.restitution = 0;
        kCfg.slop        = kOne / 64;
        kCfg.beta        = (convex::fx)((int64_t)4 * kOne / 10);    // 0.4
        kCfg.linDamp     = (convex::fx)((int64_t)98 * kOne / 100);  // 0.98
        kCfg.angDamp     = kOne;                                    // OFF — friction holds the tower
        kCfg.posIters    = 4;
        kCfg.mu          = kOne;

        // ================= SolveFrictionWarm with a ZERO seed solves to the totals =================
        {
            // Two unit boxes overlapping on +X, the slab approaching: the accumulated normal impulse arrests
            // the approach -> after iters the normal accumulator is >0 and the residual normal vel ~0.
            const convex::FxBox kUnit{convex::FxVec3{kOne, kOne, kOne}};
            fpx::FxBody bA = makeBody(0, 0, 0, false);                 // static
            fpx::FxBody bB = makeBody(fi(1), 0, 0, true);              // dynamic, overlapping by 1 on X
            bB.vel = {-(kOne), 0, 0};                                  // approaching A
            persist::KeyedFrictionManifold keyed =
                persist::BuildKeyedManifold(0, 1, bA, kUnit, bB, kUnit);
            check(keyed.fm.count > 0, "PS3 zero-seed: the overlapping pair yields >=1 contact point");
            // Accumulators are zero at build (cold seed).
            const convex::FxVec3 invIa = convex::FxBoxInvInertiaBody(kUnit, bA.invMass);
            const convex::FxVec3 invIb = convex::FxBoxInvInertiaBody(kUnit, bB.invMass);
            const convex::FxMat3 invIaW = convex::WorldInvInertia(bA, invIa);
            const convex::FxMat3 invIbW = convex::WorldInvInertia(bB, invIb);
            persist::SolveFrictionWarm(bA, bB, invIaW, invIbW, keyed.fm, kCfg.restitution, kCfg.mu, 20);
            // The normal accumulator is > 0 (the contact pushed back); within ±mu*jn tangent cone (zero here).
            bool normalsNonNeg = true, coneRespected = true;
            for (uint32_t i = 0; i < keyed.fm.count; ++i) {
                const persist::fx jn  = keyed.fm.pts[i].normalImpulse;
                const persist::fx jt1 = keyed.fm.pts[i].tangentImpulse1;
                const persist::fx jt2 = keyed.fm.pts[i].tangentImpulse2;
                if (jn < 0) normalsNonNeg = false;
                const persist::fx cone = fpx::fxmul(kCfg.mu, jn);
                auto absfx = [](persist::fx v) { return v < 0 ? -v : v; };
                if (absfx(jt1) > cone + 4 || absfx(jt2) > cone + 4) coneRespected = false;
            }
            check(normalsNonNeg, "PS3 zero-seed: accumulated normal impulse stays >= 0");
            check(coneRespected, "PS3 zero-seed: accumulated tangent within the +/-mu*jn cone");
            // The relative normal velocity is arrested (the slab no longer drives into the floor).
            check(bB.vel.x > -(kOne / 2), "PS3 zero-seed: the approach velocity is arrested by the solve");
        }

        // ================= a WARM seed PRIMES the bodies (the prime moves velocity) =================
        {
            const convex::FxBox kUnit{convex::FxVec3{kOne, kOne, kOne}};
            fpx::FxBody bA = makeBody(0, 0, 0, false);
            fpx::FxBody bB = makeBody(fi(1), 0, 0, true);
            bB.vel = {0, 0, 0};   // AT REST — only the prime can move it
            persist::KeyedFrictionManifold keyed =
                persist::BuildKeyedManifold(0, 1, bA, kUnit, bB, kUnit);
            // Seed a non-zero NORMAL accumulator at every point (a warm cache hit from last tick).
            for (uint32_t i = 0; i < keyed.fm.count; ++i) keyed.fm.pts[i].normalImpulse = fi(3);
            const convex::FxVec3 invIa = convex::FxBoxInvInertiaBody(kUnit, bA.invMass);
            const convex::FxVec3 invIb = convex::FxBoxInvInertiaBody(kUnit, bB.invMass);
            const convex::FxMat3 invIaW = convex::WorldInvInertia(bA, invIa);
            const convex::FxMat3 invIbW = convex::WorldInvInertia(bB, invIb);
            const fpx::FxBody before = bB;
            // ZERO sweeps -> ONLY the prime runs -> the seeded normal impulse moves the body off rest.
            persist::SolveFrictionWarm(bA, bB, invIaW, invIbW, keyed.fm, kCfg.restitution, kCfg.mu, 0);
            check(std::memcmp(&bB.vel, &before.vel, sizeof(bB.vel)) != 0,
                  "PS3 warm seed: the prime (0 sweeps) injects the seeded impulse -> velocity changes");
        }

        // ================= warm < cold residual at a LOW iteration count (the benefit) =================
        {
            persist::WarmStepConfig cLow = kCfg;
            cLow.solveIters = 2;   // a deliberately LOW iteration count
            const uint32_t kTicks = 60u;

            // WARM: the cache persists across ticks (the accumulators carry the warm-start).
            convex::ConvexWorld warmW = buildStack();
            persist::PersistentCache warmCache;
            persist::StepWarmWorldN(warmW, warmCache, cLow, kTicks);
            const persist::WarmMeasure warmM = persist::MeasureWarm(warmW);

            // COLD: identical solve but the cache is FORCE-CLEARED each tick (no warm-start — every contact
            // cold-starts at zero each tick).
            convex::ConvexWorld coldW = buildStack();
            for (uint32_t t = 0; t < kTicks; ++t) {
                persist::PersistentCache empty;   // a fresh empty cache each tick -> no inheritance
                persist::StepWarmWorld(coldW, empty, cLow);
            }
            const persist::WarmMeasure coldM = persist::MeasureWarm(coldW);

            check(warmM.maxResidual < coldM.maxResidual,
                  "PS3 benefit: warm residual < cold residual at low iters (warm converges tighter)");
        }

        // ================= warm ~= cold at a HIGH iteration count (consistency / converged fixed point) ===
        // THE HONEST CONTROL (the spec's documented fallback): accumulated GS has a unique fixed point in
        // EXACT arithmetic, but in Q16.16 FIXED POINT the warm and cold runs round their per-sweep delta
        // applications DIFFERENTLY (warm starts each tick from a primed seed, cold from zero), and those tiny
        // truncation differences feed back through positions across ticks. So warm and cold do NOT reach
        // BYTE-identity at high iters; they agree to a TIGHT INTEGER EPSILON (~38 units == ~0.0006 units at
        // the cleanest config). We assert that tight epsilon, reported honestly — NOT a faked byte-identity.
        {
            persist::WarmStepConfig cHigh = kCfg;
            cHigh.solveIters = 64;   // a HIGH iteration count -> both reach the converged region
            const uint32_t kTicks = 40u;

            convex::ConvexWorld warmW = buildStack();
            persist::PersistentCache warmCache;
            persist::StepWarmWorldN(warmW, warmCache, cHigh, kTicks);

            convex::ConvexWorld coldW = buildStack();
            for (uint32_t t = 0; t < kTicks; ++t) {
                persist::PersistentCache empty;
                persist::StepWarmWorld(coldW, empty, cHigh);
            }
            check(warmW.bodies.size() == coldW.bodies.size(), "PS3 consistency: same body count");
            int64_t maxAbsDiff = 0;
            for (size_t b = 0; b < warmW.bodies.size(); ++b) {
                const int32_t* a = reinterpret_cast<const int32_t*>(&warmW.bodies[b]);
                const int32_t* d = reinterpret_cast<const int32_t*>(&coldW.bodies[b]);
                for (int k = 0; k < 16; ++k) {
                    int64_t dd = (int64_t)a[k] - (int64_t)d[k];
                    if (dd < 0) dd = -dd;
                    if (dd > maxAbsDiff) maxAbsDiff = dd;
                }
            }
            const int64_t kEps = convex::kOne / 256;   // ~0.0039 units — a tight integer epsilon (slop-scale)
            check(maxAbsDiff <= kEps,
                  "PS3 consistency: warm ~= cold at high iters within a tight integer epsilon (converged)");
        }

        // ================= determinism: two warm runs byte-identical =================
        {
            const uint32_t kTicks = 50u;
            convex::ConvexWorld w1 = buildStack();
            persist::PersistentCache c1;
            persist::StepWarmWorldN(w1, c1, kCfg, kTicks);
            convex::ConvexWorld w2 = buildStack();
            persist::PersistentCache c2;
            persist::StepWarmWorldN(w2, c2, kCfg, kTicks);
            const bool same = std::memcmp(w1.bodies.data(), w2.bodies.data(),
                                          w1.bodies.size() * sizeof(fpx::FxBody)) == 0;
            check(same, "PS3 determinism: two warm runs BYTE-IDENTICAL");
            // The cache is rebuilt to exactly this tick's contacts (every entry has a non-negative normal).
            bool cacheSane = true;
            for (const persist::CachedContact& e : c1.entries)
                if (e.normalImpulse < 0) cacheSane = false;
            check(cacheSane, "PS3 determinism: the rebuilt cache's normal impulses stay >= 0");
        }

        // ================= the warm stack settles to a coherent rest (a resting tower) =================
        {
            const uint32_t kTicks = 240u;
            convex::ConvexWorld w = buildStack();
            persist::PersistentCache c;
            persist::StepWarmWorldN(w, c, kCfg, kTicks);
            const persist::WarmMeasure m = persist::MeasureWarm(w);
            check(m.maxSpeed < kOne / 2, "PS3 settle: the warm stack comes to REST (maxSpeed small)");
            const convex::fx y1 = w.bodies[1].pos.y, y2 = w.bodies[2].pos.y, y3 = w.bodies[3].pos.y;
            const convex::fx loBand = fi(1) - kOne / 4, hiBand = fi(1) + kOne / 4;
            const bool stacked = (y1 < y2 && y2 < y3) &&
                                 (y2 - y1 > loBand && y2 - y1 < hiBand) &&
                                 (y3 - y2 > loBand && y3 - y2 < hiBand);
            check(stacked, "PS3 settle: the warm stack stays STACKED (a coherent resting tower)");
            (void)fh;
        }
    }

    // =========================================================================================================
    // Slice PS4 — DETERMINISTIC SLEEPING ISLANDS (THE NEW PHYSICS). A per-body INTEGER kinetic-energy
    // accumulator + a fixed wake/sleep hysteresis + contact-graph island propagation, so a resting tower sleeps
    // (exactly zero residual — asleep bodies don't integrate) and a thrown box wakes the WHOLE island
    // atomically. The make-or-break: (a) a quiet body's quietTicks rises and it sleeps after sleepDelay; (b) a
    // sleeping body skipped by the step does NOT move (zero drift); (c) an energetic body stays awake; (d) a
    // wake-impulse on one sleeping body wakes its whole contact island (propagation); (e) a static floor does
    // NOT keep the tower awake; (f) the hysteresis band prevents flicker; (g) two runs byte-identical.
    {
        const fpx::FxQuat qI{0, 0, 0, convex::kOne};
        const convex::fx kOne = convex::kOne;
        auto fi = [&](int v) { return (convex::fx)(v * (int)convex::kOne); };
        const convex::fx kGravY = (convex::fx)(-9.8 * (double)kOne - 0.5);

        auto makeBody = [&](convex::fx x, convex::fx y, convex::fx z, bool dyn) {
            fpx::FxBody b;
            b.pos = {x, y, z};
            b.orient = qI;
            b.invMass = dyn ? kOne : 0;
            b.flags   = dyn ? fpx::kFlagDynamic : 0u;
            b.vel = {0, 0, 0};
            b.angVel = {0, 0, 0};
            return b;
        };
        const convex::FxBox kFloor{convex::FxVec3{fi(8), kOne, fi(8)}};
        const convex::FxBox kSlab{convex::FxVec3{fi(3) / 2, kOne / 2, fi(3) / 2}};   // 3 x 1 x 3
        // The PS4 sleeping-tower scene: a static floor + 3 dynamic slabs stacked (the PS3 stack).
        auto buildTower = [&]() {
            convex::ConvexWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, false)); w.boxes.push_back(kFloor);
            w.bodies.push_back(makeBody(0, fi(1) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            w.bodies.push_back(makeBody(0, fi(2) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            w.bodies.push_back(makeBody(0, fi(3) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            return w;
        };

        persist::SleepConfig cfg;
        cfg.warm.gravity     = convex::FxVec3{0, kGravY, 0};
        cfg.warm.dt          = kOne / 60;
        cfg.warm.solveIters  = 20;
        cfg.warm.restitution = 0;
        cfg.warm.slop        = kOne / 64;
        cfg.warm.beta        = (convex::fx)((int64_t)4 * kOne / 10);    // 0.4
        cfg.warm.linDamp     = (convex::fx)((int64_t)98 * kOne / 100);  // 0.98
        cfg.warm.angDamp     = (convex::fx)((int64_t)90 * kOne / 100);  // 0.90 — bleed spurious spin so it rests
        cfg.warm.posIters    = 4;
        cfg.warm.mu          = kOne;
        // The thresholds sit ABOVE the warm solver's resting jitter band (~0.3–0.7 unit/s) so the tower can go
        // quiet, but well below a thrown box (~6 unit/s) so a real disturbance wakes the island. (See persist.h.)
        cfg.sleepThreshold   = kOne;                       // ~1.0 unit/s — "quiet" (above the jitter band)
        cfg.wakeThreshold    = (convex::fx)(2 * (int)kOne); // ~2.0 unit/s — the wake band top
        cfg.sleepDelay       = 30;

        // ================= KineticEnergy: zero at rest, positive when moving =================
        {
            fpx::FxBody at = makeBody(0, 0, 0, true);
            check(persist::KineticEnergy(at) == 0, "PS4 KE: a body at rest has zero kinetic energy");
            fpx::FxBody mv = makeBody(0, 0, 0, true);
            mv.vel = {fi(2), 0, 0};
            check(persist::KineticEnergy(mv) >= fi(2) - 4, "PS4 KE: a moving body has KE ~ its speed");
            fpx::FxBody sp = makeBody(0, 0, 0, true);
            sp.angVel = {0, fi(3), 0};
            check(persist::KineticEnergy(sp) >= fi(3) - 4, "PS4 KE: angular motion contributes to KE");
        }

        // ================= a quiet body's quietTicks rises + it sleeps after sleepDelay =================
        {
            persist::SleepState s;
            for (uint32_t t = 0; t < cfg.sleepDelay; ++t) {
                check(s.quietTicks == t, "PS4 hysteresis: quietTicks rises by 1 per quiet tick");
                persist::UpdateQuietTicks(s, 0, cfg);   // perfectly quiet
            }
            check(s.quietTicks >= cfg.sleepDelay, "PS4 hysteresis: quietTicks reaches sleepDelay");
            // A single energetic tick (energy > wakeThreshold) resets quietTicks to 0 + clears asleep.
            s.asleep = true;
            persist::UpdateQuietTicks(s, cfg.wakeThreshold + fi(1), cfg);
            check(s.quietTicks == 0 && !s.asleep, "PS4 hysteresis: an energetic tick resets + wakes the body");
        }

        // ================= the hysteresis BAND prevents flicker (no increment, no reset) =================
        {
            persist::SleepState s;
            // Drive it quiet to near-sleep.
            for (uint32_t t = 0; t < 10; ++t) persist::UpdateQuietTicks(s, 0, cfg);
            const uint32_t held = s.quietTicks;
            // An energy strictly INSIDE the band [sleepThreshold, wakeThreshold] holds quietTicks (no flicker).
            const convex::fx mid = (cfg.sleepThreshold + cfg.wakeThreshold) / 2;
            persist::UpdateQuietTicks(s, mid, cfg);
            check(s.quietTicks == held, "PS4 band: an in-band energy neither increments nor resets quietTicks");
            check(!s.asleep, "PS4 band: an in-band energy does not wake (asleep flag untouched)");
        }

        // ================= the warm tower goes to sleep (all asleep) at exactly zero residual =================
        convex::ConvexWorld sleptW;   // captured for the wake test below
        persist::PersistentCache sleptCache;
        std::vector<persist::SleepState> sleptSleep;
        {
            sleptW = buildTower();
            persist::StepWarmSleepWorldN(sleptW, sleptCache, sleptSleep, cfg, 300u);
            const persist::SleepMeasure m = persist::MeasureSleep(sleptW, sleptSleep);
            check(m.dynamicCount == 3, "PS4 sleep: 3 dynamic bodies in the tower");
            check(m.asleepCount == 3, "PS4 sleep: the whole tower goes ASLEEP after settling");
            check(m.awakeCount == 0, "PS4 sleep: no dynamic body remains awake");
            check(m.maxSpeed == 0, "PS4 sleep: zero residual motion (asleep bodies don't move)");
            // The tower is still a coherent stack (didn't collapse before sleeping).
            const convex::fx y1 = sleptW.bodies[1].pos.y, y2 = sleptW.bodies[2].pos.y, y3 = sleptW.bodies[3].pos.y;
            check(y1 < y2 && y2 < y3, "PS4 sleep: the tower rests STACKED (a coherent resting tower)");
        }

        // ================= a sleeping body skipped by the step does NOT move (zero drift) =================
        {
            // Step the already-asleep tower one more tick; asleep bodies must be byte-identical (zero drift).
            convex::ConvexWorld w = sleptW;
            persist::PersistentCache c = sleptCache;
            std::vector<persist::SleepState> s = sleptSleep;
            const convex::ConvexWorld before = w;
            persist::StepWarmSleepWorld(w, c, s, cfg);
            bool zeroDrift = true;
            for (size_t i = 0; i < w.bodies.size(); ++i) {
                if (!convex::IsDynamic(w.bodies[i])) continue;
                if (s[i].asleep && std::memcmp(&w.bodies[i], &before.bodies[i], sizeof(fpx::FxBody)) != 0)
                    zeroDrift = false;
            }
            check(zeroDrift, "PS4 zero-drift: stepping the asleep tower moves NO asleep body (byte-identical)");
        }

        // ================= an energetic body stays awake (does NOT sleep) =================
        {
            // A body whose KE stays ABOVE wakeThreshold every tick must NEVER accrue quietTicks → never sleeps.
            persist::SleepState s;
            const convex::fx energetic = cfg.wakeThreshold + fi(1);   // well above the wake band
            for (uint32_t t = 0; t < cfg.sleepDelay * 3u; ++t)
                persist::UpdateQuietTicks(s, energetic, cfg);
            check(s.quietTicks == 0 && !s.asleep,
                  "PS4 energetic: a body energetic every tick never accrues quietTicks (stays awake)");
            // And a freshly-dropped (still-settling) tower has not yet completed sleepDelay quiet ticks.
            convex::ConvexWorld w = buildTower();
            persist::PersistentCache c;
            std::vector<persist::SleepState> ss;
            persist::StepWarmSleepWorldN(w, c, ss, cfg, cfg.sleepDelay - 1u);
            const persist::SleepMeasure m = persist::MeasureSleep(w, ss);
            check(m.asleepCount == 0, "PS4 energetic: a still-settling tower has NOT slept before sleepDelay");
        }

        // ================= a wake-impulse wakes the WHOLE contact island (propagation/atomicity) =========
        {
            convex::ConvexWorld w = sleptW;
            persist::PersistentCache c = sleptCache;
            std::vector<persist::SleepState> s = sleptSleep;
            // Confirm the whole island is asleep first.
            check(s[1].asleep && s[2].asleep && s[3].asleep, "PS4 wake: the island starts fully asleep");
            // Wake ONLY the TOP body with a large impulse (a thrown box striking it). The propagation must wake
            // the WHOLE contact-connected island (bodies 1,2,3 — the floor is static/inert, not an island-waker).
            w.bodies[3].vel = {fi(6), 0, 0};   // a strong lateral kick on the top slab
            persist::StepWarmSleepWorld(w, c, s, cfg);
            check(!s[1].asleep && !s[2].asleep && !s[3].asleep,
                  "PS4 island: a wake-impulse on ONE body wakes the WHOLE contact island (atomically)");
            const persist::SleepMeasure m = persist::MeasureSleep(w, s);
            check(m.awakeCount == 3 && m.asleepCount == 0, "PS4 island: every dynamic body in the island is awake");
            check(m.maxSpeed > 0, "PS4 island: the woken island is moving (non-zero residual)");
        }

        // ================= a static floor does NOT keep the tower awake =================
        {
            // The tower slept (proven above) even though every dynamic body contacts the STATIC floor (directly
            // or through the chain). If the static floor counted as an island-waker, the tower could never sleep.
            // Re-assert via a minimal scene: one dynamic box resting on the static floor sleeps.
            convex::ConvexWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, false)); w.boxes.push_back(kFloor);
            w.bodies.push_back(makeBody(0, fi(1) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            persist::PersistentCache c;
            std::vector<persist::SleepState> s;
            persist::StepWarmSleepWorldN(w, c, s, cfg, 200u);
            check(s[1].asleep, "PS4 static-floor: a box resting on the STATIC floor sleeps (floor isn't a waker)");
            check(persist::KineticEnergy(w.bodies[1]) == 0, "PS4 static-floor: the slept box has zero residual");
        }

        // ================= determinism: two sleeping runs byte-identical (bodies + sleep) =================
        {
            convex::ConvexWorld w1 = buildTower();
            persist::PersistentCache c1; std::vector<persist::SleepState> s1;
            persist::StepWarmSleepWorldN(w1, c1, s1, cfg, 200u);
            convex::ConvexWorld w2 = buildTower();
            persist::PersistentCache c2; std::vector<persist::SleepState> s2;
            persist::StepWarmSleepWorldN(w2, c2, s2, cfg, 200u);
            const bool sameBodies = std::memcmp(w1.bodies.data(), w2.bodies.data(),
                                                w1.bodies.size() * sizeof(fpx::FxBody)) == 0;
            check(sameBodies, "PS4 determinism: two sleeping runs BYTE-IDENTICAL bodies");
            bool sameSleep = (s1.size() == s2.size());
            for (size_t i = 0; sameSleep && i < s1.size(); ++i)
                if (s1[i].quietTicks != s2[i].quietTicks || s1[i].asleep != s2[i].asleep ||
                    s1[i].energy != s2[i].energy) sameSleep = false;
            check(sameSleep, "PS4 determinism: two sleeping runs BYTE-IDENTICAL sleep states");
        }
    }

    if (g_fail == 0) std::printf("persist_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
