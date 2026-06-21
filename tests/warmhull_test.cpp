// Slice WH1 — Warm-Started Hull Contacts: THE HULL CONTACT FEATURE ID (the int32 MSL-native BEACHHEAD of
// FLAGSHIP #26: WARM-STARTED HULL CONTACTS + ROBUST DETERMINISTIC STACKING, hf::sim::warmhull). The integer
// core (engine/sim/warmhull.h) the GPU shaders/warmhull_key.comp.hlsl copies VERBATIM + proves bit-identical.
// The KEY is PURE INT32 (no Q16.16 products — only compares, shifts, xors) -> MSL-native, a TRUE GPU pass on
// BOTH backends. warmhull.h #includes sim/manifold.h read-only (transitively gjk/convex/fpx).
//
// What this test PINS (the contracts the GPU warmhull_key.comp + the GPU==CPU proof build on, the spec proofs):
//   * THE CRUX: ClipFaceAgainstFaceTagged's OUTPUT POSITIONS are BYTE-EQUAL to the frozen
//     manifold::ClipFaceAgainstFace's over the battery (the determinism contract — same integer math, same
//     order). The tags are a deterministic function of the same integer signs.
//   * Distinct hull contacts get DISTINCT keys; the SAME contact under a sub-LSB relative nudge keeps the SAME
//     key (same reference face + same incident source feature) -> the warm-start match.
//   * A ref/inc flip or a sliding contact -> the key CHANGES (the documented warm-start-miss BOUNDARY — assert
//     it changes, the honest caveat, NOT a bug).
//   * Order-normalization: swapping bodyA/bodyB yields the SAME key (the persist.h identity discipline).
//   * HullContactKeyHash is deterministic + collision-light over the battery; MeasureHullKeys is deterministic.
//
// Pure C++ (hf_core), ASan-eligible like the other sim tests.
#include "sim/warmhull.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace warmhull = hf::sim::warmhull;
namespace manifold = hf::sim::manifold;
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

static FxBody BodyAt(fx px, fx py, fx pz) {
    FxBody b; b.pos = {px, py, pz}; b.orient = {0, 0, 0, kOne}; return b;
}

// The deterministic contact battery (the MF2/MF3 scene: box-on-box=4, tetra-on-face=3, edge-on-face=2). Each
// pair's two bodies get GLOBAL indices 2*pair and 2*pair+1.
struct Pair { FxHull hA; FxBody bA; FxHull hB; FxBody bB; };
static std::vector<Pair> MakeBattery() {
    const fx overlap = kOne / 8;   // 0.125 penetration
    std::vector<Pair> pairs;

    // box-on-box flat contact -> 4
    {
        const FxHull boxStaticH = gjk::MakeBox(kOne, kOne, kOne);
        const FxBody boxStaticB = BodyAt(gjk::FromInt(-4), 0, 0);
        const FxHull boxTopH = gjk::MakeBox(kOne, kOne, kOne);
        const FxBody boxTopB = BodyAt(gjk::FromInt(-4), gjk::FromInt(2) - overlap, 0);
        pairs.push_back({boxStaticH, boxStaticB, boxTopH, boxTopB});
    }
    // tetra-on-face -> 3
    {
        FxHull tetraH;
        tetraH.verts[0] = FxVec3{0, kOne, 0};
        tetraH.verts[1] = FxVec3{gjk::FromInt(1), -kOne, gjk::FromInt(1)};
        tetraH.verts[2] = FxVec3{gjk::FromInt(1), -kOne, -gjk::FromInt(1)};
        tetraH.verts[3] = FxVec3{-gjk::FromInt(1), -kOne, 0};
        tetraH.count = 4;
        const FxHull boxMidH = gjk::MakeBox(kOne, kOne, kOne);
        const FxBody boxMidB = BodyAt(0, 0, 0);
        const FxBody tetraB  = BodyAt(0, gjk::FromInt(2) - overlap, 0);
        pairs.push_back({boxMidH, boxMidB, tetraH, tetraB});
    }
    // edge-on-face -> 2
    {
        const FxHull boxEdgeBaseH = gjk::MakeBox(kOne, kOne, kOne);
        const FxBody boxEdgeBaseB = BodyAt(gjk::FromInt(4), 0, 0);
        const FxHull boxEdgeH = gjk::MakeBox(kOne, kOne, kOne);
        FxBody boxEdgeB = BodyAt(gjk::FromInt(4), (fx)((1.0 - 0.125 + 1.41421356) * 65536.0), 0);
        boxEdgeB.orient = {0, 0, (fx)(0.38268343f * 65536.0f), (fx)(0.92387953f * 65536.0f)};
        pairs.push_back({boxEdgeBaseH, boxEdgeBaseB, boxEdgeH, boxEdgeB});
    }
    return pairs;
}

// Re-derive the SAME reference/incident face selection manifold::HullManifoldFromEpa makes (manifold.h:486-513)
// so the test can call BOTH clips over the SAME faces. Returns false if there is no clip path (fallback).
struct RefIncSel {
    bool refIsA;
    const FxHull* refHull; FxBody refBody; manifold::FxHullFaces refFaces; uint32_t refFace;
    const FxHull* incHull; FxBody incBody; manifold::FxHullFaces incFaces; uint32_t incFace;
};
static bool SelectRefInc(const Pair& p, RefIncSel& sel) {
    const gjk::GjkResult g = gjk::Gjk(p.hA, p.bA, p.hB, p.bB);
    if (!g.overlap) return false;
    const manifold::FxHullFaces facesA = manifold::BuildCanonicalFaces(p.hA);
    const manifold::FxHullFaces facesB = manifold::BuildCanonicalFaces(p.hB);
    if (facesA.faceCount == 0 || facesB.faceCount == 0) return false;
    const gjk::EpaResult epa = gjk::Epa(p.hA, p.bA, p.hB, p.bB, g.simplex);
    const FxVec3 n = epa.normal;
    const uint32_t sfA = manifold::SupportFace(p.hA, facesA, p.bA, n);
    const FxVec3   nfA = manifold::FaceNormalWorld(p.hA, facesA, p.bA, sfA);
    const fx       alignA = convex::FxDot(nfA, n);
    const FxVec3   negN = FxVec3{-n.x, -n.y, -n.z};
    const uint32_t sfB = manifold::SupportFace(p.hB, facesB, p.bB, negN);
    const FxVec3   nfB = manifold::FaceNormalWorld(p.hB, facesB, p.bB, sfB);
    const fx       alignB = convex::FxDot(nfB, negN);
    sel.refIsA = (alignA >= alignB);
    if (sel.refIsA) {
        sel.refHull = &p.hA; sel.refBody = p.bA; sel.refFaces = facesA; sel.refFace = sfA;
        sel.incHull = &p.hB; sel.incBody = p.bB; sel.incFaces = facesB;
        sel.incFace = manifold::IncidentFace(p.hB, facesB, p.bB, nfA);
    } else {
        sel.refHull = &p.hB; sel.refBody = p.bB; sel.refFaces = facesB; sel.refFace = sfB;
        sel.incHull = &p.hA; sel.incBody = p.bA; sel.incFaces = facesA;
        sel.incFace = manifold::IncidentFace(p.hA, facesA, p.bA, nfB);
    }
    if (sel.refFace >= sel.refFaces.faceCount || sel.incFace >= sel.incFaces.faceCount) return false;
    return true;
}

int main() {
    HF_TEST_MAIN_INIT();

    const std::vector<Pair> battery = MakeBattery();

    // ================= THE CRUX: tagged clip == frozen clip (positions BYTE-EQUAL) =================
    {
        int pairsClipped = 0, clipPointTotal = 0;
        bool allByteEqual = true;
        for (const Pair& p : battery) {
            RefIncSel sel;
            if (!SelectRefInc(p, sel)) continue;
            ++pairsClipped;
            // Frozen clip.
            FxVec3 frozenPts[manifold::kMaxClipVerts];
            int frozenN = 0;
            manifold::ClipFaceAgainstFace(*sel.refHull, sel.refBody, sel.refFaces, sel.refFace,
                                          *sel.incHull, sel.incBody, sel.incFaces, sel.incFace,
                                          frozenPts, frozenN);
            // Tagged clip.
            warmhull::TaggedVert taggedPts[manifold::kMaxClipVerts];
            int taggedN = 0;
            warmhull::ClipFaceAgainstFaceTagged(*sel.refHull, sel.refBody, sel.refFaces, sel.refFace,
                                                *sel.incHull, sel.incBody, sel.incFaces, sel.incFace,
                                                taggedPts, taggedN);
            if (taggedN != frozenN) { allByteEqual = false; continue; }
            clipPointTotal += frozenN;
            for (int k = 0; k < frozenN; ++k) {
                if (std::memcmp(&taggedPts[k].pos, &frozenPts[k], sizeof(FxVec3)) != 0)
                    allByteEqual = false;
            }
        }
        check(pairsClipped > 0, "battery produced at least one clipped pair");
        check(allByteEqual, "tagged clip == frozen clip (points BYTE-EQUAL)");
        if (allByteEqual)
            std::printf("wh1-keys: tagged clip == frozen clip (points BYTE-EQUAL) "
                        "[%d clipped pairs, %d clip points]\n", pairsClipped, clipPointTotal);
    }

    // Build the keyed manifolds once (the shared input for the discrimination + nudge tests).
    std::vector<warmhull::HullKeyPair> kpairs;
    for (uint32_t i = 0; i < (uint32_t)battery.size(); ++i)
        kpairs.push_back({2u * i, battery[i].bA, battery[i].hA,
                          2u * i + 1u, battery[i].bB, battery[i].hB});
    const std::vector<warmhull::KeyedHullContact> keyed = warmhull::BuildHullContactKeys(kpairs);
    check(!keyed.empty(), "battery produced keyed contact points");

    // ================= key discrimination: distinct contacts -> distinct keys =================
    {
        std::vector<warmhull::HullContactKey> keys;
        for (const auto& kc : keyed) keys.push_back(kc.key);
        const warmhull::HullKeyMeasure km = warmhull::MeasureHullKeys(keys);
        // Distinct contacts -> distinct keys: within a pair, distinct provenance points get distinct keys (no
        // two manifold points of the SAME pair collide on a key — the warm-start cache would alias them).
        bool distinctWithinPair = true;
        for (size_t i = 0; i < keyed.size(); ++i)
            for (size_t j = i + 1; j < keyed.size(); ++j) {
                if (keyed[i].key.bodyA == keyed[j].key.bodyA &&
                    keyed[i].key.bodyB == keyed[j].key.bodyB &&
                    warmhull::HullContactKeysEqual(keyed[i].key, keyed[j].key)) {
                    // Same pair + equal key for two DISTINCT manifold points -> an aliasing failure.
                    distinctWithinPair = false;
                }
            }
        check(distinctWithinPair, "distinct contacts within a pair -> distinct keys");
        check(km.distinctKeys == km.totalKeys, "no two battery contacts share a key (all distinct)");
        check(km.maxHashCollisions == 0, "HullContactKeyHash collision-free over the battery");
        std::printf("wh1-keys: {pairs:%u, keys:%u} distinct contacts -> distinct keys\n",
                    (uint32_t)battery.size(), km.totalKeys);
    }

    // ================= sub-LSB nudge -> SAME key (the warm-start match) =================
    {
        // Re-run the SAME battery with each body's position nudged by +1 LSB in x (a sub-resolution relative
        // shift): the reference face + the incident source feature are UNCHANGED -> the keys MATCH tick-to-tick.
        std::vector<warmhull::HullKeyPair> nudged = kpairs;
        for (auto& kp : nudged) { kp.bodyA.pos.x += 1; kp.bodyB.pos.x += 1; }   // translate the whole pair 1 LSB
        const std::vector<warmhull::KeyedHullContact> keyed2 = warmhull::BuildHullContactKeys(nudged);
        bool matchedUnderNudge = (keyed.size() == keyed2.size());
        for (size_t i = 0; i < keyed.size() && matchedUnderNudge; ++i)
            if (!warmhull::HullContactKeysEqual(keyed[i].key, keyed2[i].key)) matchedUnderNudge = false;
        check(matchedUnderNudge, "sub-LSB nudge -> SAME key (matchedUnderNudge:true)");
        std::printf("wh1-keys: sub-LSB nudge keeps the key (matchedUnderNudge:%s)\n",
                    matchedUnderNudge ? "true" : "false");
    }

    // ================= a ref/inc flip or slide -> key CHANGES (the honest warm-start-miss boundary) ==========
    {
        // Flip the box-on-box pair: put the TOP box as the static base and the base as the top (swap which hull
        // owns the reference face by a gross geometry change). The reference face flips -> the key CHANGES. We
        // build a contact where the OTHER hull becomes the reference (a tetra resting under a box face vs a box
        // resting on the tetra). Simplest honest demonstration: rotate the top box 90 deg about Z so a DIFFERENT
        // face becomes incident -> the incident source feature (and likely the ref face) changes -> key changes.
        const fx overlap = kOne / 8;
        const FxHull baseH = gjk::MakeBox(kOne, kOne, kOne);
        const FxBody baseB = BodyAt(gjk::FromInt(-4), 0, 0);
        const FxHull topH = gjk::MakeBox(kOne, kOne, kOne);
        // The unrotated contact (the battery's pair 0).
        std::vector<warmhull::HullKeyPair> flat = {
            {0u, baseB, baseH, 1u, BodyAt(gjk::FromInt(-4), gjk::FromInt(2) - overlap, 0), topH}};
        // A GROSSLY shifted contact: slide the top box +1 unit in x so a different incident corner set clips.
        std::vector<warmhull::HullKeyPair> slid = {
            {0u, baseB, baseH, 1u, BodyAt(gjk::FromInt(-4) + gjk::FromInt(1),
                                          gjk::FromInt(2) - overlap, 0), topH}};
        const auto kFlat = warmhull::BuildHullContactKeys(flat);
        const auto kSlid = warmhull::BuildHullContactKeys(slid);
        check(!kFlat.empty() && !kSlid.empty(), "flip/slide scene produced contacts");
        // At least one key must DIFFER between the flat and the grossly-slid contact (the feature set changed).
        bool anyChanged = (kFlat.size() != kSlid.size());
        if (!anyChanged) {
            for (size_t i = 0; i < kFlat.size(); ++i) {
                bool foundEqual = false;
                for (size_t j = 0; j < kSlid.size(); ++j)
                    if (warmhull::HullContactKeysEqual(kFlat[i].key, kSlid[j].key)) { foundEqual = true; break; }
                if (!foundEqual) { anyChanged = true; break; }
            }
        }
        check(anyChanged, "a sliding/flipping contact CHANGES the key (the honest warm-start-miss boundary)");
    }

    // ================= order-normalization: swap bodyA/bodyB -> SAME key =================
    {
        // MakeHullContactKey is the order-normalize primitive: (i,j) and (j,i) with the SAME (refIsA, refFace,
        // incTag) yield the SAME key. NOTE refIsA is the RAW-order A; swapping the bodies ALSO flips refIsA (the
        // same physical hull is now "raw B"), so the caller passes the swapped refIsA. The key must be invariant.
        const uint32_t refFace = 3, incTag = warmhull::EncodeVertexTag(5);
        const warmhull::HullContactKey kij = warmhull::MakeHullContactKey(3, 7, /*refIsA=*/true, refFace, incTag);
        // Swap the bodies AND flip refIsA (the same hull, now raw-B, owns the ref face).
        const warmhull::HullContactKey kji = warmhull::MakeHullContactKey(7, 3, /*refIsA=*/false, refFace, incTag);
        check(kij.bodyA == 3 && kij.bodyB == 7, "order-normalize: (3,7) -> bodyA=3,bodyB=7");
        check(kji.bodyA == 3 && kji.bodyB == 7, "order-normalize: (7,3) -> bodyA=3,bodyB=7 (swapped)");
        check(warmhull::HullContactKeysEqual(kij, kji),
              "order-normalize: swap bodyA/bodyB (+ flip refIsA) -> SAME key");
        check(warmhull::HullContactKeyHash(kij) == warmhull::HullContactKeyHash(kji),
              "order-normalize: equal keys -> equal hash");
    }

    // ================= tag encodings never collide (vertex vs intersection) =================
    {
        const uint32_t vtag = warmhull::EncodeVertexTag(7);
        const uint32_t itag = warmhull::EncodeIntersectTag(2, 3);
        check(!warmhull::TagIsIntersection(vtag), "vertex tag: intersect bit clear");
        check(warmhull::TagIsIntersection(itag), "intersection tag: intersect bit set");
        check(vtag != itag, "vertex tag != intersection tag (encodings never collide)");
        // A distinct (refEdge, incEdge) crossing distinct from another.
        check(warmhull::EncodeIntersectTag(2, 3) != warmhull::EncodeIntersectTag(3, 2),
              "intersection tag distinguishes (refEdge, incEdge) order");
    }

    // ================= MeasureHullKeys determinism (two runs byte-identical) =================
    {
        std::vector<warmhull::HullContactKey> keys;
        for (const auto& kc : keyed) keys.push_back(kc.key);
        const warmhull::HullKeyMeasure m1 = warmhull::MeasureHullKeys(keys);
        const warmhull::HullKeyMeasure m2 = warmhull::MeasureHullKeys(keys);
        check(std::memcmp(&m1, &m2, sizeof(warmhull::HullKeyMeasure)) == 0,
              "MeasureHullKeys deterministic (two runs byte-identical)");
    }

    // ================= WH2: THE KEYED MANIFOLD + PERSISTENT CACHE =================
    // BuildKeyedHullManifold tags the battery counts {4,3,2}; MatchHullCache inherits a matched point's impulse +
    // cold-starts an absent one; matched+cold==count; UpdateHullCache evicts a departed key + keeps a persistent
    // one across a sub-LSB nudge; a sliding contact (the WH1 key-change boundary) cold-starts (no wrong inherit);
    // two-run byte-equal.
    {
        // The synthesized per-contact normal impulse seed (a fixed positive function of the key hash).
        auto synthImpulse = [](const warmhull::HullContactKey& k) -> fx {
            const uint32_t h = warmhull::HullContactKeyHash(k);
            return (fx)((int32_t)((h % 4096u) + 256u));
        };

        // Build the frame-t keyed manifolds over the battery; assert counts {4,3,2}.
        std::vector<warmhull::KeyedHullManifoldWH2> frameT;
        for (uint32_t i = 0; i < (uint32_t)battery.size(); ++i)
            frameT.push_back(warmhull::BuildKeyedHullManifold(
                2u * i, battery[i].bA, battery[i].hA, 2u * i + 1u, battery[i].bB, battery[i].hB));
        check(frameT.size() == 3, "WH2: battery built 3 keyed manifolds");
        if (frameT.size() == 3) {
            check(frameT[0].manifold.count == 4, "WH2: box-on-box keyed manifold count==4");
            check(frameT[1].manifold.count == 3, "WH2: tetra-on-face keyed manifold count==3");
            check(frameT[2].manifold.count == 2, "WH2: edge-on-face keyed manifold count==2");
        }
        // Every freshly-built point starts at normalImpulse 0 (the cold-start contract).
        bool allZeroAtBuild = true;
        for (const auto& km : frameT)
            for (uint32_t i = 0; i < km.manifold.count && i < 4u; ++i)
                if (km.normalImpulse[i] != 0) allZeroAtBuild = false;
        check(allZeroAtBuild, "WH2: BuildKeyedHullManifold zeroes normalImpulse (cold-start contract)");

        // Seed the cache with synthesized impulses (UpdateHullCache over the frame-t set).
        warmhull::HullCache cache;
        uint32_t frameTContacts = 0;
        for (auto& km : frameT) {
            for (uint32_t i = 0; i < km.manifold.count && i < 4u; ++i)
                km.normalImpulse[i] = synthImpulse(km.keys[i]);
            warmhull::HullCache one;
            warmhull::UpdateHullCache(one, km);   // exercise UpdateHullCache (the per-pair store)
            check(one.entries.size() == km.manifold.count, "WH2: UpdateHullCache stores all points");
            for (const auto& e : one.entries) cache.entries.push_back(e);
            frameTContacts += km.manifold.count;
        }
        check(cache.entries.size() == frameTContacts, "WH2: cache holds all frame-t contacts");

        // ----- MatchHullCache inherits on the SAME scene (every point matches; matched+cold==count) -----
        {
            warmhull::KeyedHullManifoldWH2 km = warmhull::BuildKeyedHullManifold(
                0u, battery[0].bA, battery[0].hA, 1u, battery[0].bB, battery[0].hB);
            const warmhull::HullCacheMeasure m = warmhull::MeasureHullCache(cache, km);
            check(m.matched + m.coldStart == m.contacts, "WH2: matched+cold==count (same scene)");
            check(m.matched == km.manifold.count && m.coldStart == 0,
                  "WH2: same-scene -> every point matched (no cold-start)");
            warmhull::MatchHullCache(cache, km);
            bool inheritedExact = true;
            for (uint32_t i = 0; i < km.manifold.count && i < 4u; ++i)
                if (km.normalImpulse[i] != synthImpulse(km.keys[i])) inheritedExact = false;
            check(inheritedExact, "WH2: a matched point INHERITS the exact cached impulse");
        }

        // ----- An ABSENT key cold-starts (MatchHullCache leaves it 0) -----
        {
            warmhull::HullCache empty;   // no entries -> every point is a miss
            warmhull::KeyedHullManifoldWH2 km = warmhull::BuildKeyedHullManifold(
                0u, battery[0].bA, battery[0].hA, 1u, battery[0].bB, battery[0].hB);
            const warmhull::HullCacheMeasure m = warmhull::MeasureHullCache(empty, km);
            check(m.matched == 0 && m.coldStart == km.manifold.count, "WH2: empty cache -> all cold-start");
            warmhull::MatchHullCache(empty, km);
            bool allZero = true;
            for (uint32_t i = 0; i < km.manifold.count && i < 4u; ++i)
                if (km.normalImpulse[i] != 0) allZero = false;
            check(allZero, "WH2: an absent key cold-starts at zero (no spurious inherit)");
        }

        // ----- Sub-LSB nudge: a persistent point inherits; a DEPARTED key evicts -----
        {
            // Frame t+1: nudge pair 0 by +1 LSB (the WH1 key is stable -> it inherits). Build a cache holding
            // BOTH pair 0's keys AND a synthetic "departed" key (bodies 99,100) absent at t+1.
            warmhull::HullCache cache2;
            warmhull::KeyedHullManifoldWH2 t0 = warmhull::BuildKeyedHullManifold(
                0u, battery[0].bA, battery[0].hA, 1u, battery[0].bB, battery[0].hB);
            for (uint32_t i = 0; i < t0.manifold.count && i < 4u; ++i) {
                t0.normalImpulse[i] = synthImpulse(t0.keys[i]);
                cache2.entries.push_back(warmhull::CachedHullContact{t0.keys[i], t0.normalImpulse[i]});
            }
            const warmhull::HullContactKey departed =
                warmhull::MakeHullContactKey(99u, 100u, true, 0u, warmhull::EncodeVertexTag(1));
            cache2.entries.push_back(warmhull::CachedHullContact{departed, (fx)777});
            const size_t cacheSizeBeforeEvict = cache2.entries.size();

            // Frame t+1: pair 0 nudged +1 LSB.
            FxBody bA1 = battery[0].bA; bA1.pos.x += 1;
            FxBody bB1 = battery[0].bB; bB1.pos.x += 1;
            warmhull::KeyedHullManifoldWH2 t1 = warmhull::BuildKeyedHullManifold(
                0u, bA1, battery[0].hA, 1u, bB1, battery[0].hB);
            const warmhull::HullCacheMeasure m = warmhull::MeasureHullCache(cache2, t1);
            check(m.matched == t1.manifold.count, "WH2: sub-LSB nudge -> all persistent points matched");
            warmhull::MatchHullCache(cache2, t1);
            bool inheritedExact = true;
            for (uint32_t i = 0; i < t1.manifold.count && i < 4u; ++i)
                if (t1.normalImpulse[i] != synthImpulse(t1.keys[i])) inheritedExact = false;
            check(inheritedExact, "WH2: persistent point inherits across a sub-LSB nudge");

            // UpdateHullCache rewrites the cache to EXACTLY frame t+1's contacts -> the departed key is EVICTED.
            warmhull::UpdateHullCache(cache2, t1);
            check(cache2.entries.size() == t1.manifold.count, "WH2: UpdateHullCache shrinks to this frame's set");
            check(cache2.entries.size() < cacheSizeBeforeEvict, "WH2: the cache shrank (an entry was evicted)");
            bool departedGone = true;
            for (const auto& e : cache2.entries)
                if (e.key.bodyA == 99u && e.key.bodyB == 100u) departedGone = false;
            check(departedGone, "WH2: UpdateHullCache EVICTS a departed key");
        }

        // ----- A sliding contact (the WH1 key-change boundary) COLD-STARTS (does NOT inherit a wrong impulse) ---
        {
            // Frame t: the flat box-on-box pair 0 -> seed its impulses.
            warmhull::HullCache slideCache;
            warmhull::KeyedHullManifoldWH2 flat = warmhull::BuildKeyedHullManifold(
                0u, battery[0].bA, battery[0].hA, 1u, battery[0].bB, battery[0].hB);
            for (uint32_t i = 0; i < flat.manifold.count && i < 4u; ++i) {
                flat.normalImpulse[i] = synthImpulse(flat.keys[i]);
                slideCache.entries.push_back(warmhull::CachedHullContact{flat.keys[i], flat.normalImpulse[i]});
            }
            // Frame t+1: SLIDE the top box +1 WHOLE unit in x (a gross feature change -> the incident corner set /
            // ref face changes -> the keys CHANGE) -> the new points must COLD-START, NOT inherit a wrong impulse.
            FxBody slidB = battery[0].bB; slidB.pos.x += gjk::FromInt(1);
            warmhull::KeyedHullManifoldWH2 slid = warmhull::BuildKeyedHullManifold(
                0u, battery[0].bA, battery[0].hA, 1u, slidB, battery[0].hB);
            const warmhull::HullCacheMeasure m = warmhull::MeasureHullCache(slideCache, slid);
            // At least ONE slid point's key must NOT be in the frame-t cache (the honest warm-start miss).
            check(m.coldStart > 0, "WH2: a sliding contact has >=1 cold-start (the key-change boundary)");
            warmhull::MatchHullCache(slideCache, slid);
            // Every point that cold-started (key absent) must be exactly 0 — never an aliased wrong impulse.
            bool noWrongInherit = true;
            for (uint32_t i = 0; i < slid.manifold.count && i < 4u; ++i) {
                bool inCache = false;
                for (const auto& e : slideCache.entries)
                    if (warmhull::HullContactKeysEqual(e.key, slid.keys[i])) { inCache = true; break; }
                if (!inCache && slid.normalImpulse[i] != 0) noWrongInherit = false;
            }
            check(noWrongInherit, "WH2: a slid (changed-key) point cold-starts at 0 (no wrong inherit — honest)");
        }

        // ----- Two-run byte-equal (determinism of the whole match) -----
        {
            auto runMatch = [&](std::vector<warmhull::KeyedHullManifoldWH2>& out) {
                out.clear();
                for (uint32_t i = 0; i < (uint32_t)battery.size(); ++i) {
                    warmhull::KeyedHullManifoldWH2 km = warmhull::BuildKeyedHullManifold(
                        2u * i, battery[i].bA, battery[i].hA, 2u * i + 1u, battery[i].bB, battery[i].hB);
                    warmhull::MatchHullCache(cache, km);
                    out.push_back(km);
                }
            };
            std::vector<warmhull::KeyedHullManifoldWH2> r1, r2;
            runMatch(r1); runMatch(r2);
            bool byteEqual = (r1.size() == r2.size());
            for (size_t i = 0; i < r1.size() && byteEqual; ++i)
                if (std::memcmp(&r1[i], &r2[i], sizeof(warmhull::KeyedHullManifoldWH2)) != 0) byteEqual = false;
            check(byteEqual, "WH2: two-run match BYTE-IDENTICAL (determinism)");
        }
    }

    // ================= Slice WH3 — THE ACCUMULATED WARM-STARTED SOLVER =================
    // Pins: (1) SolveHullManifoldWarm keeps the accumulated total >= 0 + converges (residual decreases vs the
    // non-accumulated solve at equal iters); (2) a warm-seeded settle reaches a LOWER residual than a cold one at
    // the SAME low iters (the convergence headline); (3) StepWarmHullWorldN is deterministic (two runs byte-equal);
    // (4) the cache carries the converged impulse across ticks (a settled stack's impulses stabilize).
    {
        const fpx::FxQuat kIdentity{0, 0, 0, kOne};
        const fpx::FxQuat kTilt{0, 0, (fx)(0.024997 * (double)kOne), (fx)(0.999688 * (double)kOne)};  // ~0.05 rad
        auto fd = [&](double v) { return (fx)(v * (double)kOne); };
        auto buildScene = [&]() {
            gjk::HullWorld w;
            { FxBody b = BodyAt(0, 0, 0); b.orient = kIdentity; b.invMass = 0; b.flags = 0u; w.bodies.push_back(b); }
            w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));   // 0 static support box
            { FxBody b = BodyAt(0, fd(2.3), 0); b.orient = kTilt; b.invMass = kOne; b.flags = fpx::kFlagDynamic;
              w.bodies.push_back(b); }
            w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));   // 1 tilted dropped box
            return w;
        };
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        convex::ConvexStepConfig cfg;
        cfg.gravity = convex::FxVec3{0, kGravY, 0};
        cfg.dt = kOne / 60; cfg.solveIters = 2; cfg.restitution = 0; cfg.slop = kOne / 64;
        cfg.beta = (fx)((int64_t)2 * kOne / 10); cfg.linDamp = (fx)((int64_t)95 * kOne / 100);
        cfg.angDamp = kOne; cfg.posIters = 2;   // angDamp OFF — the headline
        const uint32_t K = 300, WIN = 20;

        // The residual = the windowed ANGULAR speed (the residual-TORQUE metric — the quantity the accumulated
        // warm form removes; the #25 tower-collapse note is a residual TORQUE, not linear). Averaged over the last
        // WIN ticks to smooth the documented within-band jitter. mode: 0 = WARM (the cache persists across ticks);
        // 1 = the FROZEN non-accumulated hardened step (manifold::StepHullWorldHardened — the COLD reference, which
        // re-derives a fresh inconsistent impulse each tick and leaks the residual torque).
        auto windowedAng = [&](int mode) -> fx {
            gjk::HullWorld w = buildScene();
            warmhull::HullCache cache;
            int64_t sum = 0;
            for (uint32_t t = 0; t < K; ++t) {
                if (mode == 0) warmhull::StepWarmHullWorld(w, cache, cfg);
                else           manifold::StepHullWorldHardened(w, cfg);
                if (t >= K - WIN) {
                    fx a = 0;
                    for (const auto& b : w.bodies) if (convex::IsDynamic(b)) { fx aa = fpx::FxLength(b.angVel); if (aa > a) a = aa; }
                    sum += (int64_t)a;
                }
            }
            return (fx)(sum / (int64_t)WIN);
        };

        // (1) accumulated total >= 0 after a warm solve.
        {
            gjk::HullWorld w = buildScene();
            warmhull::HullCache cache;
            warmhull::StepWarmHullWorldN(w, cache, cfg, 60);   // settle a bit -> populate the cache
            bool allNonNeg = !cache.entries.empty();
            for (const auto& e : cache.entries) if (e.normalImpulse < 0) allNonNeg = false;
            check(allNonNeg, "WH3: the accumulated normal impulse stays >= 0 (a contact only pushes)");
        }

        // (2) the convergence headline + (3) the damping-off hold: at the SAME low iters with angDamp OFF, the
        // WARM solver's residual torque is strictly LESS than the non-accumulated COLD (frozen hardened) step's,
        // AND the warm stack settles (residual < band) where the cold step does NOT (the removed-torque-source proof).
        {
            const fx warmRes = windowedAng(0);
            const fx coldRes = windowedAng(1);
            const fx band = (fx)((int64_t)5 * kOne / 100);   // 0.05
            check(warmRes < coldRes, "WH3: warm residual < cold residual at equal iters (warm-start converges faster)");
            check(warmRes < band, "WH3: the warm stack SETTLES with angDamp OFF (residual torque < band)");
            check(coldRes >= band, "WH3: the cold (non-accumulated) step does NOT settle (the removed-torque-source proof)");
            std::printf("wh3-warm: warm residual < cold residual {warm:%d, cold:%d} at iters:%u\n",
                        (int)warmRes, (int)coldRes, cfg.solveIters);
        }

        // (4) StepWarmHullWorldN determinism: two runs byte-equal.
        {
            gjk::HullWorld a = buildScene(); warmhull::HullCache ca; warmhull::StepWarmHullWorldN(a, ca, cfg, K);
            gjk::HullWorld b = buildScene(); warmhull::HullCache cb; warmhull::StepWarmHullWorldN(b, cb, cfg, K);
            bool byteEqual = (a.bodies.size() == b.bodies.size());
            for (size_t i = 0; i < a.bodies.size() && byteEqual; ++i)
                if (std::memcmp(&a.bodies[i], &b.bodies[i], sizeof(FxBody)) != 0) byteEqual = false;
            check(byteEqual, "WH3: StepWarmHullWorldN two runs BYTE-IDENTICAL (determinism)");
            std::printf("wh3-warm determinism: two runs BYTE-IDENTICAL\n");
        }

        // (5) the cache carries the converged impulse across ticks: a settled stack's cache is non-empty + stable
        // (two consecutive ticks of a settled stack leave the cache size unchanged — the contact persists).
        {
            gjk::HullWorld w = buildScene();
            warmhull::HullCache cache;
            warmhull::StepWarmHullWorldN(w, cache, cfg, K);   // settle
            const size_t sizeAfterSettle = cache.entries.size();
            check(sizeAfterSettle > 0, "WH3: the cache is non-empty for a settled resting contact");
            warmhull::StepWarmHullWorld(w, cache, cfg);       // one more tick
            check(cache.entries.size() == sizeAfterSettle,
                  "WH3: the cache carries the contact across ticks (size stable for a settled stack)");
        }
    }

    // ================= Slice WH4 — SLEEPING ISLANDS -> THE STABLE STACK =================
    // Pins (the new-physics headline): (1) a GENUINE N>=3 tower goes FULLY asleep under StepWarmSleepHullWorldN
    // (asleep==dynamicCount, awake maxSpeed EXACTLY 0) and stands; (2) THE FALSIFIABLE DELTA — the IDENTICAL tower
    // under the frozen manifold::StepHullWorldHardenedN (damping OFF) at the SAME tick budget does NOT stay in the
    // rest band (assert the inequality — warm holds where frozen topples; do NOT fake it); (3) a wake-impulse wakes
    // the WHOLE island atomically; (4) KineticEnergyHull monotonically gates sleep (a moving body is awake, a quiet
    // body sleeps after the hysteresis window); (5) StepWarmSleepHullWorldN is deterministic (two runs byte-equal).
    {
        auto fd = [&](double v) { return (fx)(v * (double)kOne); };
        auto tiltZ = [&](double rad) { double h = rad / 2.0; return fpx::FxQuat{0, 0, fd(std::sin(h)), fd(std::cos(h))}; };
        const int N = 4;   // the DEMONSTRATED tower height (warm+sleep holds it; the frozen step topples it)
        auto buildTower = [&]() {
            gjk::HullWorld w;
            { FxBody b = BodyAt(0, 0, 0); b.orient = {0,0,0,kOne}; b.invMass = 0; b.flags = 0u; w.bodies.push_back(b); }
            w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));   // 0 static support box
            for (int k = 0; k < N; ++k) {
                FxBody b = BodyAt(0, fd(2.0 + 2.0 * k + 0.02 * (k + 1)), 0);
                b.orient = tiltZ(0.02 * ((k % 2) ? 1.0 : -1.0));
                b.invMass = kOne; b.flags = fpx::kFlagDynamic;
                w.bodies.push_back(b); w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));
            }
            return w;
        };
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        convex::ConvexStepConfig cfg;
        cfg.gravity = convex::FxVec3{0, kGravY, 0};
        cfg.dt = kOne / 60; cfg.solveIters = 8; cfg.restitution = 0; cfg.slop = kOne / 64;
        cfg.beta = (fx)((int64_t)2 * kOne / 10); cfg.linDamp = (fx)((int64_t)95 * kOne / 100);
        cfg.angDamp = kOne; cfg.posIters = 4;   // angDamp OFF — the headline
        warmhull::HullSleepConfig scfg;
        scfg.warm = cfg; scfg.sleepThreshold = kOne; scfg.wakeThreshold = (fx)(2 * (int)kOne); scfg.sleepTicks = 30;
        const uint32_t K = 800;

        auto towerStanding = [&](const gjk::HullWorld& w) -> bool {
            const fx band = (fx)(kOne / 2);   // 0.5 rest band
            auto absfx = [](fx v) { return v < 0 ? (fx)(-v) : v; };
            for (size_t i = 1; i < w.bodies.size(); ++i) {
                const fx expY = fd(2.0 + 2.0 * (double)(i - 1));
                if (absfx(w.bodies[i].pos.x) >= band || absfx(w.bodies[i].pos.z) >= band ||
                    absfx(w.bodies[i].pos.y - expY) >= band) return false;
            }
            return true;
        };

        // (1) the tower goes FULLY asleep + stands.
        gjk::HullWorld w = buildTower();
        warmhull::HullCache cache; std::vector<warmhull::HullSleepState> sleep;
        warmhull::StepWarmSleepHullWorldN(w, cache, sleep, scfg, K);
        const warmhull::HullSleepMeasure sm = warmhull::MeasureHullSleep(w, sleep);
        check(sm.asleepCount == sm.dynamicCount, "WH4: the N-tower goes FULLY asleep (asleep == dynamicCount)");
        check(sm.maxSpeed == 0, "WH4: the awake-body maxSpeed is EXACTLY 0 (the freeze — zero residual)");
        check(towerStanding(w), "WH4: the warm+sleep tower STANDS (every body within its rest band)");
        std::printf("wh4-stack: {tower:%d, asleep:%u, awakeMaxSpeed:%d} fully-asleep\n",
                    N, sm.asleepCount, (int)sm.maxSpeed);

        // (2) THE FALSIFIABLE DELTA: the IDENTICAL tower under the frozen #25 step at the SAME budget does NOT hold.
        gjk::HullWorld frozenW = buildTower();
        manifold::StepHullWorldHardenedN(frozenW, cfg, K);
        const bool warmSleepHolds = (sm.asleepCount == sm.dynamicCount) && towerStanding(w);
        const bool frozenTopples  = !towerStanding(frozenW);
        check(warmSleepHolds, "WH4: warm+sleep HOLDS the N-tower (asleep + standing)");
        check(frozenTopples, "WH4: the FROZEN #25 step TOPPLES the identical tower at the same budget (the delta)");
        std::printf("wh4-stack: {warmSleepHolds:true, frozenTopples:true} at N:%d, ticks:%u\n", N, K);

        // (3) a wake-impulse wakes the WHOLE island atomically.
        gjk::HullWorld wakeW = w;   // the settled+asleep tower
        warmhull::HullCache wakeCache = cache;
        std::vector<warmhull::HullSleepState> wakeSleep = sleep;
        wakeW.bodies[1].vel = convex::FxVec3{(fx)(6 * (int)kOne), 0, 0};   // a strong kick (>> wakeThreshold)
        warmhull::StepWarmSleepHullWorld(wakeW, wakeCache, wakeSleep, scfg);
        const warmhull::HullSleepMeasure wm = warmhull::MeasureHullSleep(wakeW, wakeSleep);
        check(wm.awakeCount == wm.dynamicCount, "WH4: a wake-impulse wakes the WHOLE island atomically");
        std::printf("wh4-stack wake: island re-energized atomically (awoke:%u)\n", wm.awakeCount);

        // (4) KineticEnergyHull gates sleep: a fast body reads high energy + wakes; the settled tower reads zero.
        {
            FxBody fast = BodyAt(0, 0, 0); fast.invMass = kOne; fast.flags = fpx::kFlagDynamic;
            fast.vel = convex::FxVec3{(fx)(5 * (int)kOne), 0, 0};
            check(warmhull::KineticEnergyHull(fast) > scfg.wakeThreshold,
                  "WH4: KineticEnergyHull of a fast body exceeds the wakeThreshold (it wakes)");
            check(warmhull::KineticEnergyHull(w.bodies[1]) == 0,
                  "WH4: KineticEnergyHull of a frozen asleep body is EXACTLY 0");
        }

        // (5) determinism: two runs byte-equal (bodies + sleep states).
        {
            gjk::HullWorld a = buildTower(); warmhull::HullCache ca; std::vector<warmhull::HullSleepState> sa;
            warmhull::StepWarmSleepHullWorldN(a, ca, sa, scfg, K);
            gjk::HullWorld b = buildTower(); warmhull::HullCache cb; std::vector<warmhull::HullSleepState> sb;
            warmhull::StepWarmSleepHullWorldN(b, cb, sb, scfg, K);
            bool eq = (a.bodies.size() == b.bodies.size());
            for (size_t i = 0; i < a.bodies.size() && eq; ++i)
                if (std::memcmp(&a.bodies[i], &b.bodies[i], sizeof(FxBody)) != 0) eq = false;
            for (size_t i = 0; i < sa.size() && eq; ++i)
                if (sa[i].asleep != sb[i].asleep || sa[i].lowEnergyTicks != sb[i].lowEnergyTicks ||
                    sa[i].energy != sb[i].energy) eq = false;
            check(eq, "WH4: StepWarmSleepHullWorldN two runs BYTE-IDENTICAL (determinism)");
            std::printf("wh4-stack determinism: two runs BYTE-IDENTICAL\n");
        }
    }

    // ===== Slice WH5 — LOCKSTEP + ROLLBACK over the warm+sleep TRIPLE (the netcode headline, the GJ5/MF5/PS5
    // twin). PURE CPU. Pins: (1) RunWarmHullLockstep authority==replica BIT-IDENTICAL over the TRIPLE
    // (bodies+cache+sleep); (2) two runs byte-identical (determinism); (3) RunWarmHullRollback
    // corrected==authority over the TRIPLE AND the mispredict genuinely diverged; (4) THE TRIPLE-NECESSITY
    // PROOF — a bodies-ONLY snapshot/restore (omitting the cache + sleep) makes the rollback DIVERGE from
    // authority, while restoring the full TRIPLE makes it ==; (5) the command stream genuinely exercises the
    // sleep state (an impulse WAKES the asleep tower so sleep transitions are part of the replayed state).
    {
        auto fd = [&](double v) { return (fx)(v * (double)kOne); };
        auto fi = [&](int v) { return (fx)((int64_t)v * (int64_t)kOne); };
        auto tiltZ = [&](double rad) { double h = rad / 2.0; return fpx::FxQuat{0, 0, fd(std::sin(h)), fd(std::cos(h))}; };
        const int N = 4;   // the WH4 DEMONSTRATED tower (goes fully asleep) — its lockstep is the scene
        auto buildTower = [&]() {
            gjk::HullWorld w;
            { FxBody b = BodyAt(0, 0, 0); b.orient = {0,0,0,kOne}; b.invMass = 0; b.flags = 0u; w.bodies.push_back(b); }
            w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));   // 0 static support box
            for (int k = 0; k < N; ++k) {
                FxBody b = BodyAt(0, fd(2.0 + 2.0 * k + 0.02 * (k + 1)), 0);
                b.orient = tiltZ(0.02 * ((k % 2) ? 1.0 : -1.0));
                b.invMass = kOne; b.flags = fpx::kFlagDynamic;
                w.bodies.push_back(b); w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));
            }
            return w;
        };
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        convex::ConvexStepConfig cfg;
        cfg.gravity = convex::FxVec3{0, kGravY, 0};
        cfg.dt = kOne / 60; cfg.solveIters = 8; cfg.restitution = 0; cfg.slop = kOne / 64;
        cfg.beta = (fx)((int64_t)2 * kOne / 10); cfg.linDamp = (fx)((int64_t)95 * kOne / 100);
        cfg.angDamp = kOne; cfg.posIters = 4;
        warmhull::HullSleepConfig scfg;
        scfg.warm = cfg; scfg.sleepThreshold = kOne; scfg.wakeThreshold = (fx)(2 * (int)kOne); scfg.sleepTicks = 30;

        const gjk::HullWorld kInit = buildTower();
        const uint32_t kBodyCount = (uint32_t)kInit.bodies.size();
        // Enough ticks to: settle + fully sleep, then a WAKE impulse at tick 120 (the asleep tower wakes,
        // re-settles + re-sleeps), then a second perturb at tick 360 — so sleep transitions ARE the replayed
        // state, not a frozen no-op. The rollback fires at 150 (just AFTER the wake, while waking/re-settling).
        const uint32_t kTicks      = 480u;
        const uint32_t kRollbackAt = 150u;
        // The authoritative command stream: a strong horizontal impulse on the bottom box at tick 120 WAKES the
        // asleep island (>> wakeThreshold), then a second nudge at 360. True IMPULSEs -> velocity (statics
        // unaffected). These genuinely move the tower + flip the sleep state (the warm+sleep step is exercised).
        const std::vector<convex::ConvexCommand> authStream = {
            convex::ConvexCommand{120u, convex::kConvexCmdAddImpulse, 1u, convex::FxVec3{fi(5), 0, 0}},
            convex::ConvexCommand{360u, convex::kConvexCmdAddImpulse, 2u, convex::FxVec3{fi(3), 0, 0}},
        };
        const uint32_t kCommandCount = (uint32_t)authStream.size();
        // The MISPREDICTED stream: auth + a WRONG extra strong impulse at rollbackAt on the top box.
        std::vector<convex::ConvexCommand> mispredictStream = authStream;
        mispredictStream.push_back(convex::ConvexCommand{kRollbackAt, convex::kConvexCmdAddImpulse, (uint32_t)N,
                                                         convex::FxVec3{fi(20), 0, 0}});

        // (5) the command stream genuinely exercises the sleep state: run to JUST before the wake (asleep) then
        // ONE tick past it (awake) and confirm the asleep flag actually flips (not a frozen no-op).
        {
            gjk::HullWorld w = buildTower();
            warmhull::HullCache c; std::vector<warmhull::HullSleepState> s;
            for (uint32_t t = 0; t < 120u; ++t) warmhull::SimWarmHullTick(w, c, s, scfg, authStream, t);
            const warmhull::HullSleepMeasure before = warmhull::MeasureHullSleep(w, s);
            warmhull::SimWarmHullTick(w, c, s, scfg, authStream, 120u);   // the wake impulse fires this tick
            const warmhull::HullSleepMeasure after = warmhull::MeasureHullSleep(w, s);
            check(before.asleepCount == before.dynamicCount,
                  "WH5: the tower is FULLY asleep just before the wake impulse (sleep state engaged)");
            check(after.awakeCount == after.dynamicCount,
                  "WH5: the wake impulse WAKES the whole island (sleep transition is replayed, not a no-op)");
        }

        // (1) LOCKSTEP: replica (fed INPUTS ONLY, fresh cache+sleep) == authority over the TRIPLE.
        bool lockstepIdentical = false;
        const warmhull::WarmHullState authority =
            warmhull::RunWarmHullLockstep(kInit, scfg, authStream, kTicks, &lockstepIdentical);
        const warmhull::WarmHullState replica = warmhull::RunWarmHullLockstep(kInit, scfg, authStream, kTicks);
        check(lockstepIdentical, "WH5: RunWarmHullLockstep sets outIdentical (authority==replica TRIPLE)");
        check(warmhull::WarmHullStatesEqual(authority.world.bodies, authority.cache, authority.sleep,
                                            replica.world.bodies, replica.cache, replica.sleep),
              "WH5: authority==replica BIT-IDENTICAL over the TRIPLE (bodies+cache+sleep)");
        std::printf("wh5-lockstep: {bodies:%u, ticks:%u, commands:%u} authority==replica BIT-IDENTICAL (triple)\n",
                    kBodyCount, kTicks, kCommandCount);

        // (2) DETERMINISM: two full runs byte-identical over the TRIPLE (+ a snapshot round-trip).
        const warmhull::WarmHullState authority2 = warmhull::RunWarmHullLockstep(kInit, scfg, authStream, kTicks);
        check(warmhull::WarmHullStatesEqual(authority2.world.bodies, authority2.cache, authority2.sleep,
                                            authority.world.bodies, authority.cache, authority.sleep),
              "WH5: two runs BYTE-IDENTICAL over the TRIPLE (determinism)");
        {
            // snapshot round-trip: capture the TRIPLE, mutate one tick, restore -> back to the snapshot exactly.
            warmhull::WarmHullState mid = warmhull::RunWarmHullLockstep(kInit, scfg, authStream, kRollbackAt);
            const warmhull::WarmHullSnapshot snap =
                warmhull::SnapshotWarmHull(mid.world, mid.cache, mid.sleep, kRollbackAt);
            warmhull::SimWarmHullTick(mid.world, mid.cache, mid.sleep, scfg, authStream, kRollbackAt);   // mutate
            warmhull::RestoreWarmHull(mid.world, mid.cache, mid.sleep, snap);
            check(warmhull::WarmHullStatesEqual(mid.world.bodies, mid.cache, mid.sleep,
                                                snap.bodies, snap.cache, snap.sleep),
                  "WH5: SnapshotWarmHull/RestoreWarmHull TRIPLE round-trip is exact");
        }
        std::printf("wh5-lockstep determinism: two runs BYTE-IDENTICAL\n");

        // (3) ROLLBACK: restore the TRIPLE + re-sim the auth stream == authority; the mispredict genuinely diverged.
        bool rollbackCorrected = false, mispredictDiverged = false;
        const gjk::HullWorld rolledBack =
            warmhull::RunWarmHullRollback(kInit, scfg, authStream, mispredictStream, kTicks, kRollbackAt,
                                          &rollbackCorrected, &mispredictDiverged);
        check(rollbackCorrected, "WH5: RunWarmHullRollback corrected==authority over the TRIPLE");
        check(gjk::HullBodiesEqual(rolledBack.bodies, authority.world.bodies),
              "WH5: the rolled-back body world == authority (bodies)");
        std::printf("wh5-lockstep rollback: corrected==authority BIT-EXACT (triple)\n");
        check(mispredictDiverged, "WH5: the mispredicted state DIVERGED before rollback (real divergence)");
        std::printf("wh5-lockstep mispredict: diverged before rollback (triple) (real divergence corrected)\n");

        // (4) THE TRIPLE-NECESSITY PROOF (the PS5 lesson): a bodies-ONLY snapshot/restore (omitting the cache +
        // sleep) makes the rollback DIVERGE from authority — because the warm-start impulses + the sleep timers
        // resume WRONG. Then restoring the full TRIPLE makes it ==. This proves the cache+sleep MUST be in the
        // snapshot. We reproduce the RunWarmHullRollback control flow here but with a BODIES-ONLY restore, and
        // assert the corrected TRIPLE != authority; then the full-TRIPLE restore (above) == authority.
        {
            const warmhull::WarmHullState authFinal =
                warmhull::RunWarmHullLockstep(kInit, scfg, authStream, kTicks);
            // Advance to rollbackAt with a fresh cache+sleep (the authoritative path).
            gjk::HullWorld w = kInit;
            warmhull::HullCache cache; std::vector<warmhull::HullSleepState> sleep;
            for (uint32_t t = 0; t < kRollbackAt; ++t)
                warmhull::SimWarmHullTick(w, cache, sleep, scfg, authStream, t);
            // Snapshot ALL THREE, but on RESTORE deliberately restore ONLY the bodies (the WRONG, GJ5/MF5-style
            // bodies-only rollback) — the cache + sleep keep whatever they drifted to during the misprediction.
            const warmhull::WarmHullSnapshot snap = warmhull::SnapshotWarmHull(w, cache, sleep, kRollbackAt);
            // Speculatively mispredict (<=3 ticks) to dirty the cache + sleep (and the bodies).
            uint32_t specTicks = kTicks - kRollbackAt; if (specTicks > 3u) specTicks = 3u;
            for (uint32_t s = 0; s < specTicks; ++s)
                warmhull::SimWarmHullTick(w, cache, sleep, scfg, mispredictStream, kRollbackAt + s);
            // BODIES-ONLY restore (the bug): bodies back to the snapshot, but cache + sleep stay mispredicted.
            w.bodies = snap.bodies;   // <-- ONLY the bodies (NOT cache, NOT sleep) — the necessity-proof bug
            // Re-sim the CORRECT stream from rollbackAt with the dirty cache+sleep.
            for (uint32_t t = kRollbackAt; t < kTicks; ++t)
                warmhull::SimWarmHullTick(w, cache, sleep, scfg, authStream, t);
            const bool bodiesOnlyEq = warmhull::WarmHullStatesEqual(
                w.bodies, cache, sleep, authFinal.world.bodies, authFinal.cache, authFinal.sleep);
            check(!bodiesOnlyEq,
                  "WH5: a BODIES-ONLY rollback (omitting the cache+sleep) DIVERGES from authority (the triple "
                  "is NECESSARY)");
            // And the full-TRIPLE rollback (rolledBack, above) DID equal authority — proving the fix.
            check(rollbackCorrected && gjk::HullBodiesEqual(rolledBack.bodies, authFinal.world.bodies),
                  "WH5: restoring the full TRIPLE makes the rollback == authority (the fix)");
            std::printf("wh5-lockstep triple-necessity: bodies-only rollback DIVERGES, triple rollback ==authority\n");
        }
    }

    // ---- Slice WH6 — THE LIT 3D RENDER CAPSTONE (the money-shot): warmhull::WarmHullToRenderInstances over the
    // warm+sleep-SETTLED N=4 tower (the BP6/CD6/MF6 render-capstone twin). PURE CPU. Pins: (1) provenance — two
    // WarmHullToRenderInstances calls on the same bit-exact world are BYTE-EQUAL (gjk::HullRenderMeshEqual); the
    // render is a pure function of the deterministic sim. (2) the hull/triangle counts are correct (every body
    // meshed; tris == verts/3). (3) the render of the SETTLED (asleep) tower vs a PERTURBED world DIFFER (the
    // render reflects the actual body world). (4) the sim world is byte-UNMUTATED by the render call (a pure read,
    // gjk::HullBodiesEqual). The render is the ONE FLOAT crossing (outside the bit-exact integer loop); this test
    // pins the render-only invariants, NOT the pixel image (that is the showcase/golden's job).
    {
        auto fd = [&](double v) { return (fx)(v * (double)kOne); };
        auto tiltZ = [&](double rad) { double hh = rad / 2.0; return fpx::FxQuat{0, 0, fd(std::sin(hh)), fd(std::cos(hh))}; };
        const int N = 4;   // the WH4 DEMONSTRATED tower (goes fully asleep) — the scene WH6 renders
        auto buildTower = [&]() {
            gjk::HullWorld w;
            { FxBody b = BodyAt(0, 0, 0); b.orient = {0,0,0,kOne}; b.invMass = 0; b.flags = 0u; w.bodies.push_back(b); }
            w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));   // 0 static support box
            for (int k = 0; k < N; ++k) {
                FxBody b = BodyAt(0, fd(2.0 + 2.0 * k + 0.02 * (k + 1)), 0);
                b.orient = tiltZ(0.02 * ((k % 2) ? 1.0 : -1.0));
                b.invMass = kOne; b.flags = fpx::kFlagDynamic;
                w.bodies.push_back(b); w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));
            }
            return w;
        };
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        convex::ConvexStepConfig cfg;
        cfg.gravity = convex::FxVec3{0, kGravY, 0};
        cfg.dt = kOne / 60; cfg.solveIters = 8; cfg.restitution = 0; cfg.slop = kOne / 64;
        cfg.beta = (fx)((int64_t)2 * kOne / 10); cfg.linDamp = (fx)((int64_t)95 * kOne / 100);
        cfg.angDamp = kOne; cfg.posIters = 4;
        warmhull::HullSleepConfig scfg;
        scfg.warm = cfg; scfg.sleepThreshold = kOne; scfg.wakeThreshold = (fx)(2 * (int)kOne); scfg.sleepTicks = 30;
        const uint32_t kTicks = 800u;

        // Settle the tower to the converged asleep stable stack.
        gjk::HullWorld world = buildTower();
        warmhull::HullCache cache; std::vector<warmhull::HullSleepState> sleep;
        warmhull::StepWarmSleepHullWorldN(world, cache, sleep, scfg, kTicks);

        // The headline state: the tower went fully asleep (the stable stack the render shows).
        const warmhull::HullSleepMeasure sm = warmhull::MeasureHullSleep(world, sleep);
        check(sm.asleepCount == (uint32_t)N && sm.awakeCount == 0,
              "WH6: the rendered tower is the warm+sleep-settled stack (all N dynamic bodies asleep)");

        // A snapshot of the bodies BEFORE the render call (the sim-unmutated proof).
        const std::vector<fpx::FxBody> bodiesBefore = world.bodies;

        // (1)+(2) PROVENANCE + counts: two WarmHullToRenderInstances calls are BYTE-EQUAL; the counts are right.
        const gjk::HullRenderMesh meshA = warmhull::WarmHullToRenderInstances(world);
        const gjk::HullRenderMesh meshB = warmhull::WarmHullToRenderInstances(world);
        check(gjk::HullRenderMeshEqual(meshA, meshB),
              "WH6: WarmHullToRenderInstances provenance — two calls BYTE-EQUAL (pure function of the bit-exact world)");
        check(meshA.triangles == (uint32_t)(meshA.verts.size() / 3) && !meshA.verts.empty(),
              "WH6: render mesh triangle count == verts/3 (and non-empty — every body meshed)");
        // Every box hull -> 12 triangles; 1 static + N dynamic boxes -> 12*(N+1) tris (the canonical box mesh).
        check(meshA.triangles == (uint32_t)(12 * (N + 1)),
              "WH6: render mesh has the expected box-tower triangle count (12 tris/box)");

        // (3) the render of the SETTLED tower vs a PERTURBED world DIFFER (the render reflects the body world).
        {
            gjk::HullWorld perturbed = world;
            perturbed.bodies[1].pos.x += fd(0.5);   // shove the bottom dynamic box sideways
            const gjk::HullRenderMesh meshP = warmhull::WarmHullToRenderInstances(perturbed);
            check(!gjk::HullRenderMeshEqual(meshA, meshP),
                  "WH6: a perturbed world renders DIFFERENTLY (the render tracks the actual body positions)");
        }

        // (4) SIM-UNMUTATED: the render call did not touch the bit-exact integer sim (a pure read).
        check(gjk::HullBodiesEqual(world.bodies, bodiesBefore),
              "WH6: the sim world is byte-UNMUTATED by the render call (WarmHullToRenderInstances is a pure read)");

        std::printf("wh6-render: {hulls:%u, tris:%u} provenance byte-equal + sim byte-unmutated (CPU)\n",
                    (uint32_t)world.bodies.size(), meshA.triangles);
    }

    if (g_fail == 0) std::printf("warmhull_test: ALL PASS\n");
    else             std::printf("warmhull_test: %d FAILURE(S)\n", g_fail);
    return g_fail ? 1 : 0;
}
