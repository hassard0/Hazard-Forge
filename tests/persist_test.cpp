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

    // =========================================================================================================
    // Slice PS5 — LOCKSTEP + ROLLBACK (THE NETCODE HEADLINE). The whole warm+sleeping sim is replayable from
    // inputs alone, with the replayable state = the TRIPLE (bodies + persistent cache + per-body sleep state).
    // The make-or-break: (a) two peers fed only a command stream converge BYTE-IDENTICAL over all three;
    // (b) SnapshotPersist/RestorePersist round-trip the triple exactly; (c) a rollback re-sims from a snapshot
    // bit-for-bit to the authority; (d) the mispredicted intermediate genuinely DIVERGED; (e) a snapshot taken
    // while the tower is ASLEEP restores so the replica stays asleep (the sleep state is part of the state);
    // (f) two RunPersistLockstep runs byte-identical.
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
        // The SAME PS4 warm+sleep tower scene (a static floor + 3 dynamic slabs).
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
        cfg.warm.angDamp     = (convex::fx)((int64_t)90 * kOne / 100);  // 0.90
        cfg.warm.posIters    = 4;
        cfg.warm.mu          = kOne;
        cfg.sleepThreshold   = kOne;
        cfg.wakeThreshold    = (convex::fx)(2 * (int)kOne);
        cfg.sleepDelay       = 30;

        // The deterministic command stream: a couple of early perturbations, then (with the tower settled +
        // ASLEEP) a wake-impulse at a fixed later tick wakes + topples it. A wake-impulse on a sleeping body is
        // the PS4 wake event; ApplyConvexCommands runs BEFORE the step so the KE/sleep evaluation sees it.
        const uint32_t kTicks    = 220;
        const uint32_t kWakeTick = 160;   // > sleepDelay after settling: the tower is asleep by here
        std::vector<convex::ConvexCommand> authStream;
        // Early perturbations (small lateral nudges while still settling — exercise the warm solve).
        authStream.push_back(convex::ConvexCommand{2u,  convex::kConvexCmdAddImpulse, 3u, convex::FxVec3{fi(1) / 2, 0, 0}});
        authStream.push_back(convex::ConvexCommand{5u,  convex::kConvexCmdAddImpulse, 2u, convex::FxVec3{-fi(1) / 4, 0, 0}});
        // The wake-impulse (a strong lateral kick on the top slab — wakes the island + topples it).
        authStream.push_back(convex::ConvexCommand{kWakeTick, convex::kConvexCmdAddImpulse, 3u, convex::FxVec3{fi(6), 0, 0}});

        const convex::ConvexWorld w0 = buildTower();
        const persist::PersistentCache cache0;            // cold start (empty cache)
        const std::vector<persist::SleepState> sleep0;    // sized on first step

        // ================= lockstep: authority == replica byte-for-byte (bodies + cache + sleep) =================
        {
            bool identical = false;
            const persist::PersistState auth = persist::RunPersistLockstep(w0, cache0, sleep0, cfg, authStream,
                                                                           kTicks, &identical);
            check(identical, "PS5 lockstep: authority == replica BIT-IDENTICAL (bodies + cache + sleep)");
            // The tower actually woke + moved by kWakeTick (the scene exercises wake, not a frozen no-op).
            const persist::SleepMeasure m = persist::MeasureSleep(auth.world, auth.sleep);
            check(m.awakeCount > 0 || m.maxSpeed > 0,
                  "PS5 lockstep: the wake-impulse actually woke/moved the tower (the scene is non-trivial)");
            (void)m;
        }

        // ================= determinism: two RunPersistLockstep runs byte-identical (the triple) =================
        {
            const persist::PersistState a = persist::RunPersistLockstep(w0, cache0, sleep0, cfg, authStream, kTicks);
            const persist::PersistState b = persist::RunPersistLockstep(w0, cache0, sleep0, cfg, authStream, kTicks);
            check(persist::PersistStatesEqual(a.world.bodies, a.cache, a.sleep,
                                              b.world.bodies, b.cache, b.sleep),
                  "PS5 determinism: two RunPersistLockstep runs BYTE-IDENTICAL (bodies + cache + sleep)");
        }

        // ================= snapshot/restore round-trips the triple exactly =================
        {
            // Advance partway to a non-trivial state (some cache + some sleep), snapshot, mutate, restore.
            convex::ConvexWorld w = w0;
            persist::PersistentCache c = cache0;
            std::vector<persist::SleepState> s = sleep0;
            for (uint32_t t = 0; t < 50u; ++t) persist::SimPersistTick(w, c, s, cfg, authStream, t);
            const persist::PersistSnapshot snap = persist::SnapshotPersist(w, c, s, 50u);
            // Mutate all three (advance several ticks — bodies + cache + sleep all change).
            for (uint32_t t = 50u; t < 70u; ++t) persist::SimPersistTick(w, c, s, cfg, authStream, t);
            check(!persist::PersistStatesEqual(w.bodies, c, s, snap.bodies, snap.cache, snap.sleep),
                  "PS5 snapshot: advancing past the snapshot genuinely changed the triple (a real round-trip)");
            persist::RestorePersist(w, c, s, snap);
            check(persist::PersistStatesEqual(w.bodies, c, s, snap.bodies, snap.cache, snap.sleep),
                  "PS5 snapshot: RestorePersist restores the (bodies + cache + sleep) triple EXACTLY");
        }

        // ================= a snapshot taken while the tower is ASLEEP restores -> the replica stays asleep ======
        {
            // Settle + sleep WITHOUT the wake-impulse (only the early nudges, which fade before sleepDelay).
            std::vector<convex::ConvexCommand> settleStream;
            settleStream.push_back(authStream[0]);
            settleStream.push_back(authStream[1]);
            convex::ConvexWorld w = w0;
            persist::PersistentCache c = cache0;
            std::vector<persist::SleepState> s = sleep0;
            for (uint32_t t = 0; t < kWakeTick; ++t) persist::SimPersistTick(w, c, s, cfg, settleStream, t);
            const persist::SleepMeasure mAsleep = persist::MeasureSleep(w, s);
            check(mAsleep.asleepCount == 3 && mAsleep.awakeCount == 0,
                  "PS5 sleep-snapshot: the tower is fully ASLEEP at the snapshot point");
            const persist::PersistSnapshot snap = persist::SnapshotPersist(w, c, s, kWakeTick);
            // A replica restores the snapshot + steps quietly (no command) -> must stay asleep (zero residual).
            // The replica is seeded with the static scene GEOMETRY first (boxes are immutable/shared and NOT
            // snapshotted by design — a real peer already knows the collision geometry); RestorePersist then
            // overwrites the dynamic triple (bodies + cache + sleep) from the snapshot.
            convex::ConvexWorld rw = w0; persist::PersistentCache rc; std::vector<persist::SleepState> rs;
            persist::RestorePersist(rw, rc, rs, snap);
            const std::vector<convex::ConvexCommand> none;
            persist::SimPersistTick(rw, rc, rs, cfg, none, kWakeTick);
            const persist::SleepMeasure mRep = persist::MeasureSleep(rw, rs);
            check(mRep.asleepCount == 3 && mRep.maxSpeed == 0,
                  "PS5 sleep-snapshot: the replica restored from the asleep snapshot STAYS ASLEEP (zero residual)");
        }

        // ================= rollback: corrected == authority AND the mispredict genuinely diverged =================
        {
            // The mispredicted stream: the WRONG wake-impulse (different body, different direction, wrong tick) —
            // a client prediction that diverges, corrected by the rollback re-sim of the authStream.
            std::vector<convex::ConvexCommand> mispredictStream;
            mispredictStream.push_back(convex::ConvexCommand{kWakeTick, convex::kConvexCmdAddImpulse, 2u,
                                                             convex::FxVec3{-fi(7), 0, fi(3)}});
            bool correctedEq = false, mispredictDiverged = false;
            persist::RunPersistRollback(w0, cache0, sleep0, cfg, authStream, mispredictStream,
                                        kTicks, kWakeTick, &correctedEq, &mispredictDiverged);
            check(mispredictDiverged,
                  "PS5 rollback: the mispredicted intermediate genuinely DIVERGED (a real divergence corrected)");
            check(correctedEq,
                  "PS5 rollback: the corrected re-sim == authority BIT-EXACT (bodies + cache + sleep)");
        }
    }

    // ============================================================================================
    // Slice PS6 — THE LIT 3D RENDER CAPSTONE (the money-shot): PersistToRenderInstances over the
    // converged warm+sleep world. A render-only FLOAT delegate to the frozen CX6 bridge; the bit-exact
    // PS1-PS5 sim is NOT mutated. Tests (pure CPU): (1) the provenance contract — two calls produce
    // byte-equal math::Mat4 arrays (the render is a pure function of the bit-exact sim); (2) the instance
    // split is correct (floor count == the static body count, boxes count == the dynamic body count);
    // (3) a render of the ASLEEP settled world differs from the WOKEN toppled world (the render reflects
    // the sim state). The SAME PS4 warm+sleep tower scene + PS5 command stream as the block above.
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
        cfg.warm.angDamp     = (convex::fx)((int64_t)90 * kOne / 100);  // 0.90
        cfg.warm.posIters    = 4;
        cfg.warm.mu          = kOne;
        cfg.sleepThreshold   = kOne;
        cfg.wakeThreshold    = (convex::fx)(2 * (int)kOne);
        cfg.sleepDelay       = 30;

        const uint32_t kTicks    = 220;
        const uint32_t kWakeTick = 160;
        std::vector<convex::ConvexCommand> authStream;
        authStream.push_back(convex::ConvexCommand{2u,  convex::kConvexCmdAddImpulse, 3u, convex::FxVec3{fi(1) / 2, 0, 0}});
        authStream.push_back(convex::ConvexCommand{5u,  convex::kConvexCmdAddImpulse, 2u, convex::FxVec3{-fi(1) / 4, 0, 0}});
        authStream.push_back(convex::ConvexCommand{kWakeTick, convex::kConvexCmdAddImpulse, 3u, convex::FxVec3{fi(6), 0, 0}});

        const convex::ConvexWorld w0 = buildTower();
        const persist::PersistentCache cache0;
        const std::vector<persist::SleepState> sleep0;

        // The converged AUTHORITY world (the woken/toppled tower) — the exact state the showcase renders.
        const persist::PersistState authority =
            persist::RunPersistLockstep(w0, cache0, sleep0, cfg, authStream, kTicks);
        const convex::ConvexWorld& cw = authority.world;

        // Count the static + dynamic bodies (the expected split — floor == static, boxes == dynamic).
        uint32_t staticCount = 0, dynamicCount = 0;
        for (const fpx::FxBody& b : cw.bodies) {
            if (convex::IsDynamic(b)) ++dynamicCount; else ++staticCount;
        }

        // ================= (1) provenance: two calls byte-equal (the render is a pure function) =================
        {
            const convex::ConvexRenderInstances a = persist::PersistToRenderInstances(cw);
            const convex::ConvexRenderInstances b = persist::PersistToRenderInstances(cw);
            bool equal = (a.floor.size() == b.floor.size()) && (a.boxes.size() == b.boxes.size());
            for (size_t k = 0; k < a.floor.size() && equal; ++k)
                if (std::memcmp(a.floor[k].m, b.floor[k].m, sizeof(float) * 16) != 0) equal = false;
            for (size_t k = 0; k < a.boxes.size() && equal; ++k)
                if (std::memcmp(a.boxes[k].m, b.boxes[k].m, sizeof(float) * 16) != 0) equal = false;
            check(equal,
                  "PS6 render: two PersistToRenderInstances calls produce BYTE-EQUAL Mat4 arrays (provenance)");
        }

        // ================= (2) the instance split is correct (floor == static, boxes == dynamic) =================
        {
            const convex::ConvexRenderInstances ri = persist::PersistToRenderInstances(cw);
            check(ri.floor.size() == staticCount,
                  "PS6 render: floor instance count == the static body count");
            check(ri.boxes.size() == dynamicCount,
                  "PS6 render: boxes instance count == the dynamic body count");
            check(ri.floor.size() == 1 && ri.boxes.size() == 3,
                  "PS6 render: the scene split is {floor:1, boxes:3} (the warm+sleep tower)");
        }

        // ================= (3) the asleep settled world vs the woken toppled world DIFFER =================
        {
            // Settle + SLEEP the SAME tower WITHOUT the wake-impulse (only the early nudges, which fade).
            std::vector<convex::ConvexCommand> settleStream;
            settleStream.push_back(authStream[0]);
            settleStream.push_back(authStream[1]);
            const persist::PersistState asleep =
                persist::RunPersistLockstep(w0, cache0, sleep0, cfg, settleStream, kWakeTick);

            const convex::ConvexRenderInstances riAsleep  = persist::PersistToRenderInstances(asleep.world);
            const convex::ConvexRenderInstances riToppled = persist::PersistToRenderInstances(cw);
            check(riAsleep.boxes.size() == riToppled.boxes.size(),
                  "PS6 render: both worlds have the same dynamic-box count (only the transforms differ)");
            bool differ = false;
            for (size_t k = 0; k < riAsleep.boxes.size() && !differ; ++k)
                if (std::memcmp(riAsleep.boxes[k].m, riToppled.boxes[k].m, sizeof(float) * 16) != 0) differ = true;
            check(differ,
                  "PS6 render: the ASLEEP settled world's box matrices DIFFER from the WOKEN toppled world's "
                  "(the render reflects the sim state)");

            // The toppled tower is genuinely non-trivial (the wake actually moved/tilted a dynamic body).
            const persist::SleepMeasure m = persist::MeasureSleep(cw, authority.sleep);
            check(m.awakeCount > 0 || m.maxSpeed > 0,
                  "PS6 render: the converged authority world is the non-trivial woken/toppled tower");
        }
    }

    // =========================================================================================================
    // Slice PS7 — SPATIAL ISLAND PARTITIONING + GENERAL-HULL CONTACT PERSISTENCE (Track-R R7). What this block
    // PINS:
    //   (a) the existing PS1-PS6 assertions above are UNTOUCHED (the frozen-behavior regression gate);
    //   (b) ISLAND EQUIVALENCE — the spatial (broadphase-candidates -> narrowphase-confirm -> union-find)
    //       partition == the all-pairs partition, canonical-digest-EQUAL, on a dense scene AND a spread field;
    //       the candidate-pair count is pinned << n(n-1)/2 on the spread scene (the perf-structure proof);
    //   (c) STEP BIT-IDENTITY — the spatial warm+sleep steps (box AND hull) == the frozen all-pairs steps
    //       BYTE-IDENTICAL over full runs incl. sleep + wake (bodies + cache + sleep, the strongest proof);
    //   (d) HULL SLEEP — a mixed tetra/octa/box pile settles + goes FULLY ASLEEP at a pinned step through the
    //       spatial hull path; a struck sleeping hull WAKES (and the untouched islands STAY asleep — the
    //       partial-world wake only an island partition delivers) + re-settles at a pinned step; an all-asleep
    //       step is a body-byte-no-op (the sleeping-path cost floor);
    //   (e) HULL WARM-START METRICS — pinned honestly (on this pile the cache shows NO measurable benefit:
    //       warm == cold bit-for-bit; the structural warm-start benefit proof remains WH3's frozen tower test);
    //   (f) LOCKSTEP + ROLLBACK — the spatial hull-persist world through the WH5 command+snapshot mold;
    //   (g) DIGESTS — pinned FNV-1a values (asserted identical under MSVC and clang builds).
    // All initial-pose constants are EXACT Q16.16 integers (no libm in the scene builders -> the pinned digests
    // are compiler-independent by construction).
    {
        namespace warmhull = hf::sim::warmhull;
        namespace gjk = hf::sim::gjk;
        const convex::fx kOne = convex::kOne;
        auto fi = [&](int v) { return (convex::fx)(v * (int)kOne); };
        const convex::fx kGravY = (convex::fx)(-9.8 * (double)kOne - 0.5);
        auto makeBody = [&](convex::fx x, convex::fx y, convex::fx z, bool dyn) {
            fpx::FxBody b;
            b.pos = {x, y, z};
            b.orient = fpx::FxQuat{0, 0, 0, kOne};
            b.invMass = dyn ? kOne : 0;
            b.flags   = dyn ? fpx::kFlagDynamic : 0u;
            b.vel = {0, 0, 0};
            b.angVel = {0, 0, 0};
            return b;
        };

        // ================= (1) union-find canonical partition: order-independence + labels =================
        {
            // 6 bodies: 0 static, 1-5 dynamic. Edges {1-2, 2-3} and {4-5} -> islands {1,2,3} and {4,5}.
            std::vector<fpx::FxBody> bodies;
            for (int i = 0; i < 6; ++i) bodies.push_back(makeBody(0, 0, 0, i != 0));
            const std::vector<fpx::FxPair> fwd = {{1u,2u},{2u,3u},{4u,5u}};
            const std::vector<fpx::FxPair> rev = {{4u,5u},{2u,3u},{1u,2u}};   // reversed processing order
            const std::vector<fpx::FxPair> shf = {{2u,3u},{4u,5u},{1u,2u}};   // shuffled
            const persist::IslandPartition pf = persist::IslandsFromEdges(bodies, fwd);
            const persist::IslandPartition pr = persist::IslandsFromEdges(bodies, rev);
            const persist::IslandPartition ps = persist::IslandsFromEdges(bodies, shf);
            check(pf.islandCount == 2u, "PS7 union-find: two islands from {1-2-3} + {4-5}");
            check(pf.islandOf[0] == persist::kStaticIslandLabel, "PS7 union-find: the static body carries the inert label");
            check(pf.islandOf[1] == 0u && pf.islandOf[2] == 0u && pf.islandOf[3] == 0u,
                  "PS7 union-find: {1,2,3} share canonical island 0 (min-member order)");
            check(pf.islandOf[4] == 1u && pf.islandOf[5] == 1u, "PS7 union-find: {4,5} share canonical island 1");
            check(persist::IslandPartitionsEqual(pf, pr) && persist::IslandPartitionsEqual(pf, ps),
                  "PS7 union-find: the partition is ORDER-INDEPENDENT (forward == reversed == shuffled edges)");
            check(persist::IslandPartitionDigest(pf) == persist::IslandPartitionDigest(ps),
                  "PS7 union-find: equal partitions -> equal canonical digests");
        }

        // ================= (2) BOX: spatial islands == all-pairs + spatial step bit-identity =================
        const convex::FxBox kFloor{convex::FxVec3{fi(8), kOne, fi(8)}};
        const convex::FxBox kSlab{convex::FxVec3{fi(3) / 2, kOne / 2, fi(3) / 2}};   // 3 x 1 x 3
        persist::SleepConfig bcfg;
        bcfg.warm.gravity     = convex::FxVec3{0, kGravY, 0};
        bcfg.warm.dt          = kOne / 60;
        bcfg.warm.solveIters  = 20;
        bcfg.warm.restitution = 0;
        bcfg.warm.slop        = kOne / 64;
        bcfg.warm.beta        = (convex::fx)((int64_t)4 * kOne / 10);    // 0.4
        bcfg.warm.linDamp     = (convex::fx)((int64_t)98 * kOne / 100);  // 0.98
        bcfg.warm.angDamp     = (convex::fx)((int64_t)90 * kOne / 100);  // 0.90
        bcfg.warm.posIters    = 4;
        bcfg.warm.mu          = kOne;
        bcfg.sleepThreshold   = kOne;
        bcfg.wakeThreshold    = (convex::fx)(2 * (int)kOne);
        bcfg.sleepDelay       = 30;
        const convex::fx kBoxCell = fi(8);   // >= 2*(hx+hy+hz) = 7 for the slab (the BP2 stencil bound)

        // --- (2a) dense scene: TWO 2-slab towers on the floor -> exactly 2 islands, spatial == all-pairs ---
        {
            convex::ConvexWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, false)); w.boxes.push_back(kFloor);
            w.bodies.push_back(makeBody(-fi(4), fi(1) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            w.bodies.push_back(makeBody(-fi(4), fi(2) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            w.bodies.push_back(makeBody( fi(4), fi(1) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            w.bodies.push_back(makeBody( fi(4), fi(2) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            persist::PersistentCache c; std::vector<persist::SleepState> s;
            persist::StepWarmSleepWorldN(w, c, s, bcfg, 60u);   // mid-settle: contacts formed
            uint32_t cand = 0;
            const persist::IslandPartition ap = persist::BuildIslandsAllPairs(w);
            const persist::IslandPartition sp = persist::BuildIslandsSpatial(w, kBoxCell, &cand);
            check(persist::IslandPartitionsEqual(ap, sp),
                  "PS7 box islands (dense): spatial partition == all-pairs partition");
            check(persist::IslandPartitionDigest(ap) == persist::IslandPartitionDigest(sp),
                  "PS7 box islands (dense): canonical digests EQUAL");
            check(persist::IslandPartitionDigest(sp) == 0x354e38ccb032fa56ull,
                  "PS7 box islands (dense): the pinned partition digest");
            check(sp.islandCount == 2u, "PS7 box islands (dense): the two towers are TWO islands");
            check(cand == 6u && cand < 10u,
                  "PS7 box islands (dense): pinned candidate pairs 6 < all-pairs 10");
        }

        // --- (2b) spread field: 24 slabs 8 units apart -> candidates << n^2 (the perf-structure pin) ---
        {
            convex::ConvexWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, false));
            w.boxes.push_back(convex::FxBox{convex::FxVec3{fi(120), kOne, fi(120)}});
            for (int k = 0; k < 24; ++k) {
                w.bodies.push_back(makeBody(fi(-92 + 8 * k), fi(1) + kOne * 5 / 8, 0, true));
                w.boxes.push_back(kSlab);
            }
            persist::PersistentCache c; std::vector<persist::SleepState> s;
            persist::StepWarmSleepWorldN(w, c, s, bcfg, 40u);
            uint32_t cand = 0;
            const persist::IslandPartition ap = persist::BuildIslandsAllPairs(w);
            const persist::IslandPartition sp = persist::BuildIslandsSpatial(w, kBoxCell, &cand);
            const uint32_t n = (uint32_t)w.bodies.size();
            const uint32_t allPairs = n * (n - 1u) / 2u;   // 300
            check(persist::IslandPartitionsEqual(ap, sp),
                  "PS7 box islands (spread): spatial partition == all-pairs partition");
            check(persist::IslandPartitionDigest(sp) == 0x65e62d6300b10260ull,
                  "PS7 box islands (spread): the pinned partition digest");
            check(sp.islandCount == 24u, "PS7 box islands (spread): 24 isolated slabs -> 24 singleton islands");
            check(allPairs == 300u && cand == 24u,
                  "PS7 box islands (spread): pinned 24 candidate pairs << 300 all-pairs (O(n*k) vs O(n^2))");
            std::printf("ps7 spread-field: candidatePairs=%u allPairs=%u (islands=%u)\n",
                        cand, allPairs, sp.islandCount);
            // The spatial STEP is bit-identical on the spread scene too (60 further ticks).
            convex::ConvexWorld wa = w; persist::PersistentCache ca = c; std::vector<persist::SleepState> sa = s;
            convex::ConvexWorld ws = w; persist::PersistentCache cs = c; std::vector<persist::SleepState> ss = s;
            persist::StepWarmSleepWorldN(wa, ca, sa, bcfg, 60u);
            persist::StepWarmSleepWorldSpatialN(ws, cs, ss, bcfg, kBoxCell, 60u);
            check(persist::PersistStatesEqual(wa.bodies, ca, sa, ws.bodies, cs, ss),
                  "PS7 box step (spread): spatial == all-pairs BIT-IDENTICAL (bodies + cache + sleep)");
        }

        // --- (2c) the PS4 tower: spatial step == all-pairs step over settle + SLEEP (300 ticks) ---
        auto buildTower = [&]() {
            convex::ConvexWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, false)); w.boxes.push_back(kFloor);
            w.bodies.push_back(makeBody(0, fi(1) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            w.bodies.push_back(makeBody(0, fi(2) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            w.bodies.push_back(makeBody(0, fi(3) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            return w;
        };
        {
            convex::ConvexWorld wa = buildTower(); persist::PersistentCache ca; std::vector<persist::SleepState> sa;
            persist::StepWarmSleepWorldN(wa, ca, sa, bcfg, 300u);
            convex::ConvexWorld ws = buildTower(); persist::PersistentCache cs; std::vector<persist::SleepState> ss;
            persist::StepWarmSleepWorldSpatialN(ws, cs, ss, bcfg, kBoxCell, 300u);
            check(persist::PersistStatesEqual(wa.bodies, ca, sa, ws.bodies, cs, ss),
                  "PS7 box step (tower settle+sleep): spatial == all-pairs BIT-IDENTICAL over 300 ticks");
            check(persist::DigestBodyWorld(ws.bodies) == 0xa0e687263ec6171cull,
                  "PS7 box step (tower): the pinned settled-asleep body digest (MSVC == clang)");
        }

        // --- (2d) the PS5 command stream (nudges + a wake-impulse at 160): identity holds THROUGH wake ---
        {
            std::vector<convex::ConvexCommand> stream;
            stream.push_back(convex::ConvexCommand{2u,   convex::kConvexCmdAddImpulse, 3u, convex::FxVec3{fi(1) / 2, 0, 0}});
            stream.push_back(convex::ConvexCommand{5u,   convex::kConvexCmdAddImpulse, 2u, convex::FxVec3{-fi(1) / 4, 0, 0}});
            stream.push_back(convex::ConvexCommand{160u, convex::kConvexCmdAddImpulse, 3u, convex::FxVec3{fi(6), 0, 0}});
            convex::ConvexWorld wa = buildTower(); persist::PersistentCache ca; std::vector<persist::SleepState> sa;
            convex::ConvexWorld ws = buildTower(); persist::PersistentCache cs; std::vector<persist::SleepState> ss;
            for (uint32_t t = 0; t < 220u; ++t) {
                persist::SimPersistTick(wa, ca, sa, bcfg, stream, t);            // the frozen PS5 tick (all-pairs)
                convex::ApplyConvexCommands(ws, stream, t);                      // the same inputs
                persist::StepWarmSleepWorldSpatial(ws, cs, ss, bcfg, kBoxCell);  // the spatial tick
            }
            check(persist::PersistStatesEqual(wa.bodies, ca, sa, ws.bodies, cs, ss),
                  "PS7 box step (PS5 command run): spatial == all-pairs BIT-IDENTICAL through sleep + WAKE");
            check(persist::DigestBodyWorld(ws.bodies) == 0xd586b89433fa0c0aull,
                  "PS7 box step (PS5 command run): the pinned final body digest");
        }

        // ================= (3) HULL: spatial islands + the spatial warm+sleep hull path =================
        // Exact Q16.16 orientation constants (host-precomputed; no libm -> compiler-independent digests):
        const fpx::FxQuat kTiltZp05{0, 0, (convex::fx)1638,  (convex::fx)65515};   // tiltZ(+0.05 rad)
        const fpx::FxQuat kTiltZp02{0, 0, (convex::fx)655,   (convex::fx)65532};   // tiltZ(+0.02 rad)
        const fpx::FxQuat kTiltZn02{0, 0, (convex::fx)-655,  (convex::fx)65532};   // tiltZ(-0.02 rad)
        const convex::fx kHullCell = fi(6);   // >= 2*(sqrt(3) + 0.5-margin) AABB diameter of a tumbling unit hull

        // --- (3a) the WH4 box-hull tower: spatial hull step == the frozen WH4 step over 800 ticks ---
        {
            warmhull::HullSleepConfig scfg;
            scfg.warm.gravity = convex::FxVec3{0, kGravY, 0};
            scfg.warm.dt = kOne / 60; scfg.warm.solveIters = 8; scfg.warm.restitution = 0;
            scfg.warm.slop = kOne / 64;
            scfg.warm.beta = (convex::fx)((int64_t)2 * kOne / 10);      // 0.2
            scfg.warm.linDamp = (convex::fx)((int64_t)95 * kOne / 100); // 0.95
            scfg.warm.angDamp = kOne;                                   // OFF — the WH4 headline config
            scfg.warm.posIters = 4;
            scfg.sleepThreshold = kOne; scfg.wakeThreshold = (convex::fx)(2 * (int)kOne); scfg.sleepTicks = 30;
            // The WH4 tower: a unit static support + 4 dynamic unit box hulls, 0.02 above rest, alternating
            // +-0.02 tilts (the exact WH4 scene; Y constants are the exact fd() integers).
            const convex::fx kTowerY[4] = {(convex::fx)132382, (convex::fx)264765,
                                           (convex::fx)397148, (convex::fx)529530};
            auto buildHullTower = [&]() {
                gjk::HullWorld w;
                w.bodies.push_back(makeBody(0, 0, 0, false));
                w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));
                for (int k = 0; k < 4; ++k) {
                    fpx::FxBody b = makeBody(0, kTowerY[k], 0, true);
                    b.orient = (k % 2) ? kTiltZp02 : kTiltZn02;
                    w.bodies.push_back(b);
                    w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));
                }
                return w;
            };
            gjk::HullWorld wa = buildHullTower(); warmhull::HullCache ca; std::vector<warmhull::HullSleepState> sa;
            warmhull::StepWarmSleepHullWorldN(wa, ca, sa, scfg, 800u);
            gjk::HullWorld ws = buildHullTower(); warmhull::HullCache cs; std::vector<warmhull::HullSleepState> ss;
            persist::StepWarmSleepHullWorldSpatialN(ws, cs, ss, scfg, kHullCell, 800u);
            check(warmhull::WarmHullStatesEqual(wa.bodies, ca, sa, ws.bodies, cs, ss),
                  "PS7 hull step (WH4 tower): spatial == frozen all-pairs BIT-IDENTICAL over 800 ticks");
            const warmhull::HullSleepMeasure sm = warmhull::MeasureHullSleep(ws, ss);
            check(sm.asleepCount == 4u && sm.maxSpeed == 0,
                  "PS7 hull step (WH4 tower): the tower is FULLY ASLEEP through the spatial path");
            check(persist::DigestBodyWorld(ws.bodies) == 0x4bda223ac42f8adfull,
                  "PS7 hull step (WH4 tower): the pinned asleep body digest");
            uint32_t cand = 0;
            const persist::IslandPartition ap = persist::BuildHullIslandsAllPairs(ws);
            const persist::IslandPartition sp = persist::BuildHullIslandsSpatial(ws, kHullCell, &cand);
            check(persist::IslandPartitionsEqual(ap, sp),
                  "PS7 hull islands (tower): spatial partition == all-pairs partition");
            check(sp.islandCount == 1u && cand == 4u,
                  "PS7 hull islands (tower): ONE island, pinned 4 candidate pairs < all-pairs 10");
            check(persist::IslandPartitionDigest(sp) == 0x1794dcb0bf7e0c2bull,
                  "PS7 hull islands (tower): the pinned partition digest");
        }

        // --- (3b-3f) THE MIXED-HULL PILE (tetra/octa/box on a floor hull — the GJ6 shapes) ---
        // The PS7 hull-sleep scene: gentle near-rest drops (the WH4 recipe — the frozen accumulated warm
        // solver is NOT validated for hard drops; a large fall pumps energy through GJK near-field phantom
        // contacts, the documented gjk.h iteration-cap band) + angDamp 0.3 (the GJ4 stability knob — the 0.9
        // retain leaves tetra/octa rocking above the sleep threshold forever).
        warmhull::HullSleepConfig pcfg;
        pcfg.warm.gravity = convex::FxVec3{0, kGravY, 0};
        pcfg.warm.dt = kOne / 60; pcfg.warm.solveIters = 8; pcfg.warm.restitution = 0;
        pcfg.warm.slop = kOne / 64;
        pcfg.warm.beta = (convex::fx)((int64_t)2 * kOne / 10);      // 0.2
        pcfg.warm.linDamp = (convex::fx)((int64_t)95 * kOne / 100); // 0.95
        pcfg.warm.angDamp = (convex::fx)((int64_t)30 * kOne / 100); // 0.30 — the GJ4 knob
        pcfg.warm.posIters = 4;
        pcfg.sleepThreshold = kOne; pcfg.wakeThreshold = (convex::fx)(2 * (int)kOne); pcfg.sleepTicks = 30;
        auto buildPile = [&]() {
            gjk::HullWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, false));
            w.hulls.push_back(gjk::MakeBox(fi(4), kOne, fi(4)));                 // 0 floor (half 4 x 1 x 4)
            { fpx::FxBody b = makeBody((convex::fx)-144179, (convex::fx)134348, 0, true);   // (-2.2, 2.05)
              w.bodies.push_back(b); w.hulls.push_back(gjk::MakeTetra(kOne)); }  // 1 tetra
            { fpx::FxBody b = makeBody(0, (convex::fx)134348, 0, true);          // (0, 2.05)
              b.orient = kTiltZp05;
              w.bodies.push_back(b); w.hulls.push_back(gjk::MakeOcta(kOne)); }   // 2 octa
            { fpx::FxBody b = makeBody((convex::fx)144179, (convex::fx)132382, 0, true);    // (2.2, 2.02)
              w.bodies.push_back(b); w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne)); }   // 3 box
            return w;
        };

        // (3b) the pile settles + goes FULLY ASLEEP at the PINNED step through the spatial path.
        gjk::HullWorld pw = buildPile();
        warmhull::HullCache pc; std::vector<warmhull::HullSleepState> psl;
        uint32_t firstAllAsleep = 0;
        for (uint32_t t = 0; t < 400u; ++t) {
            persist::StepWarmSleepHullWorldSpatial(pw, pc, psl, pcfg, kHullCell);
            const warmhull::HullSleepMeasure m = warmhull::MeasureHullSleep(pw, psl);
            if (firstAllAsleep == 0 && m.asleepCount == m.dynamicCount) firstAllAsleep = t + 1;
        }
        check(firstAllAsleep == 38u,
              "PS7 hull sleep: the mixed tetra/octa/box pile goes FULLY ASLEEP at the pinned step 38");
        {
            const warmhull::HullSleepMeasure m = warmhull::MeasureHullSleep(pw, psl);
            check(m.asleepCount == 3u && m.maxSpeed == 0,
                  "PS7 hull sleep: all 3 dynamic hulls asleep at 400 ticks, zero residual");
            check(persist::DigestBodyWorld(pw.bodies) == 0x95549e6550d6aa85ull,
                  "PS7 hull sleep: the pinned settled-asleep pile digest (MSVC == clang)");
            std::printf("ps7 hull pile: firstAllAsleep=%u asleep=%u/%u\n",
                        firstAllAsleep, m.asleepCount, m.dynamicCount);
        }
        // The pile's islands: 3 singletons (the hulls rest apart), spatial == all-pairs, 5 candidates < 6.
        {
            uint32_t cand = 0;
            const persist::IslandPartition ap = persist::BuildHullIslandsAllPairs(pw);
            const persist::IslandPartition sp = persist::BuildHullIslandsSpatial(pw, kHullCell, &cand);
            check(persist::IslandPartitionsEqual(ap, sp),
                  "PS7 hull islands (pile): spatial partition == all-pairs partition");
            check(sp.islandCount == 3u && cand == 5u,
                  "PS7 hull islands (pile): 3 singleton islands, pinned 5 candidate pairs < all-pairs 6");
        }
        // (3c) an all-asleep step is a BODY-BYTE-NO-OP (the sleeping-path cost floor: no integrate, no solve,
        // no de-pen touches any body — the step's remaining work is the O(n*k) island scan alone).
        {
            const gjk::HullWorld before = pw;
            gjk::HullWorld w2 = pw; warmhull::HullCache c2 = pc; std::vector<warmhull::HullSleepState> s2 = psl;
            persist::StepWarmSleepHullWorldSpatial(w2, c2, s2, pcfg, kHullCell);
            check(gjk::HullBodiesEqual(w2.bodies, before.bodies),
                  "PS7 hull sleep: stepping the all-asleep pile is a body-byte-NO-OP (zero drift)");
        }
        // (3d) a struck sleeping hull WAKES; the other islands STAY ASLEEP (the partial-world wake the island
        // partition delivers); the pile RE-SETTLES fully asleep at the pinned step.
        {
            pw.bodies[3].vel = convex::FxVec3{fi(3), 0, 0};   // strike the box (KE 3.0 > wakeThreshold 2.0)
            persist::StepWarmSleepHullWorldSpatial(pw, pc, psl, pcfg, kHullCell);
            check(!psl[3].asleep, "PS7 hull wake: the struck sleeping box WAKES");
            check(psl[1].asleep && psl[2].asleep,
                  "PS7 hull wake: the untouched tetra + octa islands STAY ASLEEP (partial-world wake)");
            uint32_t reAsleep = 0;
            for (uint32_t t = 0; t < 1200u; ++t) {
                persist::StepWarmSleepHullWorldSpatial(pw, pc, psl, pcfg, kHullCell);
                const warmhull::HullSleepMeasure mm = warmhull::MeasureHullSleep(pw, psl);
                if (mm.asleepCount == mm.dynamicCount) { reAsleep = t + 1; break; }
            }
            check(reAsleep == 141u,
                  "PS7 hull wake: the struck box re-settles -> the pile is FULLY ASLEEP again at the pinned step 141");
            check(persist::DigestBodyWorld(pw.bodies) == 0x98d8e3e056a9433eull,
                  "PS7 hull wake: the pinned re-settled pile digest");
        }
        // (3e) HULL WARM-START METRICS (the honest pin): on THIS pile at low iters the persistent cache shows
        // NO measurable benefit — the warm and cold runs are BIT-IDENTICAL (the pile's floor contacts flicker
        // tick-to-tick, so the per-face keys rarely re-match; the structural warm-start benefit proof remains
        // WH3's frozen tower test, where deep persistent contacts DO re-match). Pinned as equality — NOT faked.
        {
            warmhull::HullSleepConfig wcfg = pcfg;
            wcfg.sleepTicks = 0x7FFFFFFFu;   // sleep disabled -> the pure warm-vs-cold lever
            wcfg.warm.solveIters = 2;        // a deliberately LOW iteration count
            const uint32_t K = 240;
            gjk::HullWorld ww = buildPile(); warmhull::HullCache wc; std::vector<warmhull::HullSleepState> wsl;
            persist::StepWarmSleepHullWorldSpatialN(ww, wc, wsl, wcfg, kHullCell, K);
            const gjk::HullStackMeasure wm = gjk::MeasureHullStack(ww);
            gjk::HullWorld cw = buildPile(); std::vector<warmhull::HullSleepState> csl;
            for (uint32_t t = 0; t < K; ++t) {
                warmhull::HullCache fresh;   // force-cleared each tick -> every contact cold-starts
                persist::StepWarmSleepHullWorldSpatial(cw, fresh, csl, wcfg, kHullCell);
            }
            const gjk::HullStackMeasure cm = gjk::MeasureHullStack(cw);
            check(wm.maxSpeed == 127317 && wm.maxPenetration == 0,
                  "PS7 warm-start metrics: pinned WARM pile metrics {maxSpeed 127317, maxPen 0}");
            check(cm.maxSpeed == 127317 && cm.maxPenetration == 0,
                  "PS7 warm-start metrics: pinned COLD pile metrics {maxSpeed 127317, maxPen 0}");
            check(wm.maxSpeed == cm.maxSpeed && wm.maxPenetration == cm.maxPenetration,
                  "PS7 warm-start metrics: warm == cold on this scene (NO measurable benefit — reported honestly)");
            std::printf("ps7 warm-vs-cold (pile, iters=2, K=%u): warm{maxSpeed=%d,maxPen=%d} == "
                        "cold{maxSpeed=%d,maxPen=%d} (no benefit on this scene; WH3's tower carries the benefit proof)\n",
                        K, (int)wm.maxSpeed, (int)wm.maxPenetration, (int)cm.maxSpeed, (int)cm.maxPenetration);
        }
        // (3f) LOCKSTEP + ROLLBACK over the spatial hull-persist world (the WH5 command+snapshot mold):
        // settle -> SLEEP -> a wake-impulse at tick 60 -> re-settle -> fully ASLEEP at the final tick; the
        // replayable state is the TRIPLE (bodies + HullCache + HullSleepState[]).
        {
            const uint32_t kTicks = 260, kWakeTick = 60, kRollbackAt = 60;
            std::vector<convex::ConvexCommand> auth;
            auth.push_back(convex::ConvexCommand{5u, convex::kConvexCmdAddImpulse, 1u,
                                                 convex::FxVec3{(convex::fx)32768, 0, 0}});   // a 0.5 nudge
            auth.push_back(convex::ConvexCommand{kWakeTick, convex::kConvexCmdAddImpulse, 3u,
                                                 convex::FxVec3{fi(3), 0, 0}});               // the wake kick
            std::vector<convex::ConvexCommand> mis = auth;
            mis.push_back(convex::ConvexCommand{kRollbackAt, convex::kConvexCmdAddImpulse, 2u,
                                                convex::FxVec3{-fi(8), 0, fi(2)}});           // the WRONG impulse
            const gjk::HullWorld w0 = buildPile();
            bool identical = false;
            const warmhull::WarmHullState a1 =
                persist::RunPersistHullLockstep(w0, pcfg, kHullCell, auth, kTicks, &identical);
            check(identical,
                  "PS7 lockstep: authority == replica BIT-IDENTICAL (bodies + hull cache + sleep) from inputs alone");
            const warmhull::WarmHullState a2 =
                persist::RunPersistHullLockstep(w0, pcfg, kHullCell, auth, kTicks);
            check(warmhull::WarmHullStatesEqual(a1.world.bodies, a1.cache, a1.sleep,
                                                a2.world.bodies, a2.cache, a2.sleep),
                  "PS7 lockstep: two RunPersistHullLockstep runs BYTE-IDENTICAL (determinism)");
            bool corrected = false, diverged = false;
            persist::RunPersistHullRollback(w0, pcfg, kHullCell, auth, mis, kTicks, kRollbackAt,
                                            &corrected, &diverged);
            check(diverged, "PS7 rollback: the mispredicted triple genuinely DIVERGED (a real divergence)");
            check(corrected, "PS7 rollback: the corrected re-sim == authority BIT-EXACT over the TRIPLE");
            const warmhull::HullSleepMeasure m = warmhull::MeasureHullSleep(a1.world, a1.sleep);
            check(m.asleepCount == 3u && m.maxSpeed == 0,
                  "PS7 lockstep: the converged world is FULLY ASLEEP (settle -> wake -> re-settle replayed)");
            check(persist::DigestBodyWorld(a1.world.bodies) == 0x3a49757d1f7d6750ull,
                  "PS7 lockstep: the pinned converged authority digest");
            uint32_t cand = 0;
            const persist::IslandPartition sp = persist::BuildHullIslandsSpatial(a1.world, kHullCell, &cand);
            check(persist::IslandPartitionDigest(sp) == 0x23e749cf0720569aull && sp.islandCount == 3u,
                  "PS7 lockstep: the pinned final island partition (3 islands)");
            std::printf("ps7 lockstep: {hulls:%zu, ticks:%u, asleep:%u/%u, islands:%u, candidatePairs:%u, "
                        "digest:0x%016llx}\n",
                        a1.world.bodies.size(), kTicks, m.asleepCount, m.dynamicCount, sp.islandCount, cand,
                        (unsigned long long)persist::DigestBodyWorld(a1.world.bodies));
        }
    }

    if (g_fail == 0) std::printf("persist_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
