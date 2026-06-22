// Slice HF1 — Hull Friction + Joints: THE TAGGED FRICTION MANIFOLD ON THE EPA NORMAL (the friction BEACHHEAD of
// FLAGSHIP #30: HULL FRICTION + HULL JOINTS, hf::sim::hullfric). The integer core (engine/sim/hullfric.h) the GPU
// shaders/hullfric_points.comp.hlsl copies VERBATIM + proves bit-identical. hullfric.h WRAPS the FROZEN warmhull
// keyed manifold + appends the fric::MakeTangentBasis integer tangent basis (on the sign-corrected EPA normal) +
// the per-point impulse accumulators + the basis-axis cache field. #includes warmhull/gjk/fric/persist/manifold
// READ-ONLY (ALL BYTE-FROZEN).
//
// What this test PINS (the contracts the GPU hullfric_points.comp + the GPU==CPU proof build on, the spec proofs):
//   * MakeTangentBasis is ORTHONORMAL on a spread of EPA-style normals (t1.n ~ 0, t2.n ~ 0, t1.t2 ~ 0, |t1|=|t2|=1
//     within the fixed-point band) — the basis the friction beachhead rests on.
//   * BuildHullFrictionManifold's contact points / normal-magnitude / depths == the FROZEN warmhull manifold (HF1
//     is ADDITIVE over the geometry — it appends the basis + accumulators, it does NOT perturb the narrowphase).
//   * basisAxis == fric::LeastAlignedAxis(normal) (the cardinal-axis the Gram-Schmidt picked — the warm-start
//     flip-detect field).
//   * MatchHullFrictionCache WARM-SEEDS the three impulses on a key+basis match AND COLD-STARTS the tangents on a
//     basis-axis FLIP (the crux groundwork — a flipped basis makes last tick's tangents stale).
//   * UpdateHullFrictionCache rewrites the cache to exactly this tick's contacts (absent keys evicted).
//   * Two-run determinism (a fixed pair set yields byte-identical manifolds on two builds).
//
// Pure C++ (hf_core), ASan-eligible like the other sim tests.
#include "sim/hullfric.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace hullfric = hf::sim::hullfric;
namespace warmhull = hf::sim::warmhull;
namespace fric     = hf::sim::fric;
namespace convex   = hf::sim::convex;
namespace gjk      = hf::sim::gjk;
namespace fpx      = hf::sim::fpx;
using gjk::fx;
using gjk::kOne;
using gjk::FxVec3;
using gjk::FxHull;
using fpx::FxBody;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static fx absfx(fx v) { return v < 0 ? -v : v; }

static FxBody BodyAt(fx px, fx py, fx pz) {
    FxBody b; b.pos = {px, py, pz}; b.orient = {0, 0, 0, kOne}; return b;
}

// The deterministic curated hull pair set (== the --hf1-points showcase): a box-on-box flat (4-point face), a box
// on a TILTED static box (a non-cardinal EPA normal), a second box-on-box flat.
static std::vector<hullfric::HullFrictionPair> MakePairs() {
    const fx overlap = kOne / 8;
    std::vector<hullfric::HullFrictionPair> pairs;
    const FxHull boxH = gjk::MakeBox(kOne, kOne, kOne);
    pairs.push_back({0u, BodyAt(gjk::FromInt(-4), 0, 0), boxH,
                     1u, BodyAt(gjk::FromInt(-4), gjk::FromInt(2) - overlap, 0), boxH});
    {
        FxBody tiltedBase = BodyAt(0, 0, 0);
        tiltedBase.orient = {0, 0, (fx)(0.19509032f * 65536.0f), (fx)(0.98078528f * 65536.0f)};  // ~22.5deg/Z
        FxBody topB = BodyAt(0, (fx)((1.0 + 1.41421356 - 0.125) * 65536.0), 0);
        pairs.push_back({2u, tiltedBase, boxH, 3u, topB, boxH});
    }
    pairs.push_back({4u, BodyAt(gjk::FromInt(4), 0, 0), boxH,
                     5u, BodyAt(gjk::FromInt(4), gjk::FromInt(2) - overlap, 0), boxH});
    return pairs;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ===== Test 1: MakeTangentBasis orthonormal on a spread of EPA-style normals =====
    {
        // A spread of unit-ish normals (host-snapped). MakeTangentBasis expects a unit normal; we feed normalized.
        std::vector<FxVec3> normals;
        normals.push_back(FxVec3{0, kOne, 0});                  // +Y (the flat-contact normal)
        normals.push_back(FxVec3{0, -kOne, 0});                 // -Y
        normals.push_back(FxVec3{kOne, 0, 0});                  // +X
        normals.push_back(FxVec3{0, 0, kOne});                  // +Z
        normals.push_back(fpx::FxNormalize(FxVec3{(fx)(0.3826834f * 65536.0f), (fx)(0.9238795f * 65536.0f), 0}));
        normals.push_back(fpx::FxNormalize(FxVec3{(fx)(0.5f * 65536.0f), (fx)(0.5f * 65536.0f),
                                                  (fx)(0.70710678f * 65536.0f)}));
        normals.push_back(fpx::FxNormalize(FxVec3{(fx)(-0.6f * 65536.0f), (fx)(0.8f * 65536.0f), 0}));
        const fx kBand = kOne / 256;   // ~0.004 fixed-point band
        fx maxDotErr = 0, minLen = 0, maxLen = 0;
        bool first = true;
        for (const FxVec3& n : normals) {
            const fric::TangentBasis tb = fric::MakeTangentBasis(n);
            const fx d0 = absfx(convex::FxDot(n, tb.t1));
            const fx d1 = absfx(convex::FxDot(n, tb.t2));
            const fx d2 = absfx(convex::FxDot(tb.t1, tb.t2));
            if (d0 > maxDotErr) maxDotErr = d0;
            if (d1 > maxDotErr) maxDotErr = d1;
            if (d2 > maxDotErr) maxDotErr = d2;
            const fx l1 = fpx::FxLength(tb.t1), l2 = fpx::FxLength(tb.t2);
            if (first) { minLen = l1; maxLen = l1; first = false; }
            if (l1 < minLen) minLen = l1; if (l1 > maxLen) maxLen = l1;
            if (l2 < minLen) minLen = l2; if (l2 > maxLen) maxLen = l2;
        }
        check(maxDotErr <= kBand, "MakeTangentBasis: orthogonality residual within the fixed-point band");
        check(absfx(minLen - kOne) <= kBand && absfx(maxLen - kOne) <= kBand,
              "MakeTangentBasis: both tangents are unit length within the band");
    }

    // ===== Test 2: BuildHullFrictionManifold geometry == the FROZEN warmhull manifold (HF1 is additive) =====
    {
        std::vector<hullfric::HullFrictionPair> pairs = MakePairs();
        bool anyContact = false, geomExact = true;
        for (const hullfric::HullFrictionPair& kp : pairs) {
            const hullfric::HullFrictionManifold hm = hullfric::BuildHullFrictionManifold(
                kp.bodyAIdx, kp.bodyA, kp.hullA, kp.bodyBIdx, kp.bodyB, kp.hullB);
            const warmhull::KeyedHullManifoldWH2 km = warmhull::BuildKeyedHullManifold(
                kp.bodyAIdx, kp.bodyA, kp.hullA, kp.bodyBIdx, kp.bodyB, kp.hullB);
            if (hm.count != km.manifold.count) { geomExact = false; continue; }
            if (hm.count > 0) anyContact = true;
            for (uint32_t i = 0; i < hm.count && i < 4u; ++i) {
                if (hm.points[i].x != km.manifold.points[i].x ||
                    hm.points[i].y != km.manifold.points[i].y ||
                    hm.points[i].z != km.manifold.points[i].z ||
                    hm.depths[i]   != km.manifold.depths[i] ||
                    !warmhull::HullContactKeysEqual(hm.keys[i], km.keys[i]))
                    geomExact = false;
                // the accumulators must be ZERO at build (the cold-start contract).
                if (hm.pts[i].normalImpulse != 0 || hm.pts[i].tangentImpulse1 != 0 ||
                    hm.pts[i].tangentImpulse2 != 0)
                    geomExact = false;
            }
        }
        check(anyContact, "BuildHullFrictionManifold: the curated pair set produces contacts");
        check(geomExact, "BuildHullFrictionManifold: points/depths/keys == warmhull manifold + accumulators zeroed");
    }

    // ===== Test 3: basisAxis == fric::LeastAlignedAxis(normal); the basis is orthonormal on the EPA normal =====
    {
        std::vector<hullfric::HullFrictionManifold> ms =
            hullfric::BuildAllHullFrictionManifoldsPairs(MakePairs());
        bool axisOk = true;
        const hullfric::HullFrictionMeasure meas = hullfric::MeasureHullFriction(ms);
        for (const hullfric::HullFrictionManifold& hm : ms) {
            if (hm.count == 0) continue;
            if (hm.basisAxis != (int32_t)fric::LeastAlignedAxis(hm.normal)) axisOk = false;
            // the basis vectors must be orthogonal to the (sign-corrected) normal.
            if (absfx(convex::FxDot(hm.normal, hm.t1)) > kOne / 256) axisOk = false;
            if (absfx(convex::FxDot(hm.normal, hm.t2)) > kOne / 256) axisOk = false;
        }
        check(axisOk, "BuildHullFrictionManifold: basisAxis == LeastAlignedAxis(normal), basis _|_ normal");
        check(meas.pairsWithContact > 0 && meas.maxDotErr <= kOne / 256,
              "MeasureHullFriction: basis orthonormal across the contact set");
    }

    // ===== Test 4: MatchHullFrictionCache warm-seeds on key+basis match; cold-starts on a basis-axis FLIP =====
    {
        std::vector<hullfric::HullFrictionManifold> ms =
            hullfric::BuildAllHullFrictionManifoldsPairs(MakePairs());
        // pick the first manifold with a contact.
        int picked = -1;
        for (size_t i = 0; i < ms.size(); ++i) if (ms[i].count > 0) { picked = (int)i; break; }
        check(picked >= 0, "MatchHullFrictionCache: a contact manifold exists to test");
        if (picked >= 0) {
            hullfric::HullFrictionManifold base = ms[(size_t)picked];

            // Seed a cache from a synthesized solved state (nonzero impulses) at this manifold's basisAxis.
            hullfric::HullFrictionCache cache;
            for (uint32_t i = 0; i < base.count && i < 4u; ++i) {
                cache.entries.push_back(hullfric::CachedHullFrictionContact{
                    base.keys[i], (fx)(1000 + (int)i), (fx)(-200 + (int)i), (fx)(300 + (int)i), base.basisAxis});
            }

            // (a) a fresh manifold with the SAME keys + SAME basis -> WARM-seed all three impulses.
            hullfric::HullFrictionManifold warm = ms[(size_t)picked];   // zeroed accumulators
            hullfric::MatchHullFrictionCache(cache, warm);
            bool warmSeeded = true;
            for (uint32_t i = 0; i < warm.count && i < 4u; ++i) {
                if (warm.pts[i].normalImpulse   != (fx)(1000 + (int)i) ||
                    warm.pts[i].tangentImpulse1 != (fx)(-200 + (int)i) ||
                    warm.pts[i].tangentImpulse2 != (fx)(300 + (int)i))
                    warmSeeded = false;
            }
            check(warmSeeded, "MatchHullFrictionCache: key+basis match warm-seeds all three impulses");

            // (b) the SAME keys but a FLIPPED basis axis -> the tangents (and normal) COLD-START at zero.
            hullfric::HullFrictionManifold flipped = ms[(size_t)picked];
            flipped.basisAxis = (base.basisAxis + 1) % 3;   // a deliberate basis-axis flip
            hullfric::MatchHullFrictionCache(cache, flipped);
            bool coldStarted = true;
            for (uint32_t i = 0; i < flipped.count && i < 4u; ++i) {
                if (flipped.pts[i].normalImpulse != 0 || flipped.pts[i].tangentImpulse1 != 0 ||
                    flipped.pts[i].tangentImpulse2 != 0)
                    coldStarted = false;
            }
            check(coldStarted, "MatchHullFrictionCache: a basis-axis FLIP cold-starts the impulses (the crux)");

            // (c) UpdateHullFrictionCache rewrites the cache to exactly this manifold's contacts.
            hullfric::HullFrictionCache fresh;
            hullfric::UpdateHullFrictionCache(fresh, warm);
            check(fresh.entries.size() == base.count,
                  "UpdateHullFrictionCache: the cache holds exactly this tick's contacts");
            bool basisStored = true;
            for (const auto& e : fresh.entries) if (e.basisAxis != base.basisAxis) basisStored = false;
            check(basisStored, "UpdateHullFrictionCache: stores the basis axis per entry");
        }
    }

    // ===== Test 5: two-run determinism (a fixed pair set yields byte-identical manifolds) =====
    {
        std::vector<hullfric::HullFrictionManifold> a = hullfric::BuildAllHullFrictionManifoldsPairs(MakePairs());
        std::vector<hullfric::HullFrictionManifold> b = hullfric::BuildAllHullFrictionManifoldsPairs(MakePairs());
        bool same = (a.size() == b.size());
        for (size_t i = 0; same && i < a.size(); ++i)
            if (std::memcmp(&a[i], &b[i], sizeof(hullfric::HullFrictionManifold)) != 0) same = false;
        check(same, "BuildAllHullFrictionManifoldsPairs: two runs byte-identical (determinism)");
    }

    if (g_fail == 0) std::printf("hullfric_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
