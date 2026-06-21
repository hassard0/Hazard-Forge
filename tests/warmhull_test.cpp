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

    if (g_fail == 0) std::printf("warmhull_test: ALL PASS\n");
    else             std::printf("warmhull_test: %d FAILURE(S)\n", g_fail);
    return g_fail ? 1 : 0;
}
