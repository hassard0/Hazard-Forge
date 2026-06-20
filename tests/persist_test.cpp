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

    if (g_fail == 0) std::printf("persist_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
