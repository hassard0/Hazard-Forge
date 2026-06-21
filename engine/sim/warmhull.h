#pragma once
// Slice WH1 — Warm-Started Hull Contacts: THE HULL CONTACT FEATURE ID (the int32 MSL-native BEACHHEAD of
// FLAGSHIP #26: WARM-STARTED HULL CONTACTS + ROBUST DETERMINISTIC STACKING, hf::sim::warmhull). Flagship #25
// hardened the hull narrowphase (multi-point manifolds + full inertia) so a single hull settles on a support,
// but documented that TALL stacks destabilize: the non-accumulated Gauss-Seidel + the fixed-point 4-point
// manifold leave a per-tick residual torque that integrates into a spurious spin and topples the tower. #26
// removes the SOURCE of that torque by generalizing the box-only warm-start + sleeping-islands machinery
// (persist.h, #21) to the hardened hull multi-point manifolds — an accumulated, warm-started solver that
// converges to a consistent island equilibrium instead of re-deriving a slightly-inconsistent impulse each
// tick. WH1 is the BEACHHEAD: the deterministic HULL CONTACT FEATURE ID that names a contact across frames so
// next tick's manifold points can inherit last tick's accumulated impulses.
//
// THE CRUX: a hull manifold has no discrete SAT axis (EPA gives a continuous normal + a Sutherland-Hodgman
// clip), and the clip-ORDER slot renumbers under sub-LSB motion — so the key must be a GEOMETRIC PROVENANCE
// (reference face, incident source feature), NOT an array slot. The provenance is computed inside a tagged
// parallel clip (ClipFaceAgainstFaceTagged) that mirrors the frozen manifold::ClipFaceAgainstFace BYTE-FOR-BYTE
// (same fixed edge order, same strict-integer inside test, same fxdiv crossing in the same pinned order) and
// additionally carries a per-output-vertex provenance tag.
//
// THE PURE-INT32 WIN (the STRONGEST proof tier, persist.h's PS1 split): the KEY itself is PURE int32 (small
// face/vertex/edge indices + shift/xor/avalanche hashing — NO Q16.16 products, NO int64, NO float), so
// shaders/warmhull_key.comp.hlsl is MSL-NATIVE (it goes IN the metal_headless hf_gen_msl list, NOT the
// Vulkan-only list) — a TRUE GPU pass on BOTH backends, strict zero-differing-pixel cross-vendor. The MANIFOLD
// that PRODUCES the indices is int64 (GJK/EPA + the SH clip), but that runs HOST-SIDE; the shader only packs
// the already-computed int32 indices into keys (exactly persist_key.comp's role). The shader copies
// MakeHullContactKey + HullContactKeyHash VERBATIM so the GPU HullContactKey[]+hash[] is byte-identical to the
// CPU reference (the Vulkan GPU==CPU memcmp + the Metal GPU strict-zero are the proofs).
//
// THE ORDER-NORMALIZATION (the persist.h determinism crux): a contact between bodies (i,j) and (j,i) is the
// SAME contact — so the key STORES bodyA < bodyB ALWAYS. Plus a refIsA/role bit folded into refFaceId so the
// key is identity-stable when the bodies swap (the reference face belongs to one specific hull regardless of
// iteration order). The key legitimately CHANGES when the contact feature changes (a face sliding to a new
// reference face, or a ref/incident flip) — the documented "warm-start misses at sliding contacts" caveat
// (in scope: the key changes -> a safe cold-start, NOT a bug).
//
// Header-only, namespace hf::sim::warmhull, #include "sim/manifold.h" READ-ONLY (transitively gjk/broad/ccd/
// convex/fric/persist/fpx — ALL BYTE-FROZEN; warmhull.h is a brand-new additive sibling that NEVER edits a
// frozen header). manifold.h gives HullContactMulti/HullManifoldFromEpa/ClipFaceAgainstFace/SupportFace/
// IncidentFace/BuildCanonicalFaces/FxHullFaces + the Q16.16 toolbox.

#include <cstdint>
#include <cstring>
#include <vector>

#include "sim/manifold.h"   // read-only: manifold::HullContactMulti/HullManifoldFromEpa/ClipFaceAgainstFace/
                            // SupportFace/IncidentFace/BuildCanonicalFaces/FxHullFaces/FaceNormalWorld/
                            // FaceCentroidWorld + gjk/convex/fpx (the frozen Q16.16 toolbox), transitively.

namespace hf::sim {
namespace warmhull {

// Pull the frozen helpers into this namespace (REUSE, do NOT redefine).
using gjk::fx;
using gjk::kOne;
using gjk::FxVec3;
using gjk::FxHull;
using gjk::FxBody;
using gjk::HullWorld;
using convex::FxDot;
using convex::FxCross;
using fpx::FxAdd;
using fpx::FxSub;
using fpx::fxmul;
using fpx::fxdiv;
using manifold::FxHullFaces;
using manifold::kMaxClipVerts;
using manifold::kMaxFaceVerts;
using manifold::kMaxHullFaces;

// ----- The provenance tag bit layout (PURE int32) ----------------------------------------------------------
// A clip output vertex is either an ORIGINAL surviving incident-polygon vertex (it keeps the incident hull's
// LOCAL vertex index — a small uint invariant under rigid motion) or a clip INTERSECTION point (an edge
// crossing of a reference-face edge against an incident-polygon edge — packed (refEdge, incEdge), both
// FIXED-order small integers from the SH loop). A high TAG BIT distinguishes the two so the encodings NEVER
// collide. Pure int32 (shifts + ors, NO products) so the shader can pack the resulting key MSL-natively.
constexpr uint32_t kTagIntersectBit = 0x80000000u;   // high bit: 1 == intersection, 0 == original vertex
constexpr uint32_t kTagRefEdgeShift  = 8u;           // refEdge in bits [8..15]
constexpr uint32_t kTagIncEdgeMask   = 0xFFu;        // incEdge in bits [0..7]

// EncodeVertexTag(incLocalVertIdx): an ORIGINAL incident vertex -> its small local vertex index, high bit 0.
inline uint32_t EncodeVertexTag(uint32_t incLocalVertIdx) {
    return incLocalVertIdx & 0x7FFFFFFFu;            // mask off the tag bit (local idx < kMaxHullVerts << 0x80..)
}
// EncodeIntersectTag(refEdge, incEdge): a clip crossing -> (refEdge, incEdge) packed under the intersect bit.
inline uint32_t EncodeIntersectTag(uint32_t refEdge, uint32_t incEdge) {
    return kTagIntersectBit | ((refEdge & 0xFFu) << kTagRefEdgeShift) | (incEdge & kTagIncEdgeMask);
}
inline bool TagIsIntersection(uint32_t tag) { return (tag & kTagIntersectBit) != 0u; }

// =========================================================================================================
// ClipFaceAgainstFaceTagged — THE DETERMINISM CONTRACT. A NEW provenance-carrying parallel clip that MIRRORS
// the frozen manifold::ClipFaceAgainstFace (manifold.h:372) EXACTLY: the SAME FIXED edge order (e=0..refVc-1,
// k=0..polyN-1, prev=last), the SAME strict-integer inside test (FxDot(sideN, p-a) >= 0, on-plane = inside,
// NO tolerance band), the SAME fxdiv crossing formula in the SAME pinned iteration order. Additionally it
// carries, per polygon vertex, a provenance TAG through the clip:
//   - an ORIGINAL incident vertex keeps its incident hull LOCAL vertex index (EncodeVertexTag);
//   - a clip CROSSING gets (refEdge=e, incEdge=the incident-boundary edge the crossing segment lies on)
//     (EncodeIntersectTag); the segment's incEdge is the incEdge of the SOURCE vertex `prev` (each vertex
//     carries the incident-face edge LEAVING it along the current polygon boundary — original vertex at
//     original slot k leaves along incident edge k; a crossing inherits the segment's edge), which keeps the
//     crossing's incident-edge identity GEOMETRIC + invariant under sub-LSB motion (NOT the renumbering slot).
// THE CONTRACT: the tagged clip's OUTPUT POSITIONS are byte-equal to the frozen clip's (run both, memcmp the
// positions — asserted in the showcase + test). The tags are a deterministic function of the SAME integer
// signs (the 1-LSB-flip discipline applied to the tag). Pure integer (int64 FxDot/FxCross/fxdiv/fxmul).
// =========================================================================================================

// A tagged polygon vertex: its world-space position (the SAME bits the frozen clip computes) + its provenance
// tag + `incEdgeOut` = the incident-face edge index of the segment LEAVING this vertex along the current
// polygon boundary (so a crossing on segment (prev,cur) inherits prev.incEdgeOut as its incident edge).
struct TaggedVert {
    FxVec3   pos;
    uint32_t tag       = 0;   // EncodeVertexTag / EncodeIntersectTag
    uint32_t incEdgeOut = 0;  // the incident-boundary edge index of the outgoing segment (for crossing provenance)
};

inline void ClipFaceAgainstFaceTagged(const FxHull& refHull, const FxBody& refBody, const FxHullFaces& refFaces,
                                      uint32_t refFace,
                                      const FxHull& incHull, const FxBody& incBody, const FxHullFaces& incFaces,
                                      uint32_t incFace,
                                      TaggedVert outVerts[kMaxClipVerts], int& outN) {
    outN = 0;
    if (refFace >= refFaces.faceCount || incFace >= incFaces.faceCount) return;
    const uint32_t refVc = refFaces.vertCount[refFace];
    const uint32_t incVc = incFaces.vertCount[incFace];
    if (refVc < 3 || incVc < 3) return;   // degenerate -> empty (== frozen)

    // The reference face's world-space vertices (FIXED order) + outward normal + centroid (== frozen).
    FxVec3 refV[kMaxFaceVerts];
    for (uint32_t k = 0; k < refVc; ++k) {
        const uint32_t vi = refFaces.vertIdx[refFace][k];
        refV[k] = FxAdd(fpx::FxRotate(refBody.orient, refHull.verts[vi]), refBody.pos);
    }
    const FxVec3 refN = manifold::FaceNormalWorld(refHull, refFaces, refBody, refFace);
    const FxVec3 refC = manifold::FaceCentroidWorld(refHull, refFaces, refBody, refFace);

    // The incident polygon (the thing we clip), world-space, FIXED order. Each vertex carries its LOCAL vertex
    // index as the original-vertex tag, and incEdgeOut = k (the incident edge LEAVING slot k along the boundary).
    TaggedVert poly[kMaxClipVerts];
    int polyN = (int)incVc;
    for (uint32_t k = 0; k < incVc; ++k) {
        const uint32_t vi = incFaces.vertIdx[incFace][k];
        poly[k].pos = FxAdd(fpx::FxRotate(incBody.orient, incHull.verts[vi]), incBody.pos);
        poly[k].tag = EncodeVertexTag(vi);
        poly[k].incEdgeOut = k;   // edge (k -> k+1) of the incident face
    }

    // Sutherland-Hodgman against each ref-face EDGE's inward side plane, FIXED order e=0..refVc-1 (== frozen).
    for (uint32_t e = 0; e < refVc; ++e) {
        const FxVec3 a    = refV[e];
        const FxVec3 b    = refV[(e + 1u) % refVc];
        const FxVec3 edge = FxSub(b, a);
        FxVec3 sideN = FxCross(refN, edge);
        if (FxDot(sideN, FxSub(refC, a)) < 0) sideN = FxVec3{-sideN.x, -sideN.y, -sideN.z};

        auto fside = [&](const FxVec3& p) -> fx { return FxDot(sideN, FxSub(p, a)); };

        TaggedVert out[kMaxClipVerts];
        int outNloc = 0;
        if (polyN == 0) break;
        TaggedVert prev = poly[polyN - 1];
        fx fprev = fside(prev.pos);
        for (int k = 0; k < polyN; ++k) {
            const TaggedVert cur = poly[k];
            const fx fcur = fside(cur.pos);
            const bool curIn  = (fcur >= 0);
            const bool prevIn = (fprev >= 0);
            if (curIn) {
                if (!prevIn && outNloc < (int)kMaxClipVerts) {
                    // entering: emit the crossing point first, then cur (== frozen position math).
                    const fx denom = fprev - fcur;
                    const fx tp = (denom != 0) ? fxdiv(fprev, denom) : 0;
                    TaggedVert xv;
                    xv.pos = FxVec3{prev.pos.x + fxmul(cur.pos.x - prev.pos.x, tp),
                                    prev.pos.y + fxmul(cur.pos.y - prev.pos.y, tp),
                                    prev.pos.z + fxmul(cur.pos.z - prev.pos.z, tp)};
                    // The crossing lies on the segment LEAVING prev -> its incident edge is prev.incEdgeOut;
                    // it is cut by reference edge e. The crossing's outgoing segment continues toward cur, so
                    // its own incEdgeOut stays prev.incEdgeOut (still on the same incident boundary edge).
                    xv.tag = EncodeIntersectTag(e, prev.incEdgeOut);
                    xv.incEdgeOut = prev.incEdgeOut;
                    out[outNloc++] = xv;
                }
                if (outNloc < (int)kMaxClipVerts) out[outNloc++] = cur;
            } else if (prevIn && outNloc < (int)kMaxClipVerts) {
                // leaving: emit the crossing point only (== frozen position math).
                const fx denom = fprev - fcur;
                const fx tp = (denom != 0) ? fxdiv(fprev, denom) : 0;
                TaggedVert xv;
                xv.pos = FxVec3{prev.pos.x + fxmul(cur.pos.x - prev.pos.x, tp),
                                prev.pos.y + fxmul(cur.pos.y - prev.pos.y, tp),
                                prev.pos.z + fxmul(cur.pos.z - prev.pos.z, tp)};
                xv.tag = EncodeIntersectTag(e, prev.incEdgeOut);
                xv.incEdgeOut = prev.incEdgeOut;
                out[outNloc++] = xv;
            }
            prev = cur; fprev = fcur;
        }
        polyN = outNloc;
        for (int k = 0; k < polyN; ++k) poly[k] = out[k];
    }

    outN = polyN;
    for (int k = 0; k < polyN; ++k) outVerts[k] = poly[k];
}

// ----- HullContactKey: the deterministic integer identity of one hull contact POINT -------------------------
// bodyA < bodyB ALWAYS (order-normalized — the same pair yields the same key regardless of iteration order);
// refFaceId = the reference-face provenance: the FIXED face index manifold::SupportFace chose, with the
// refIsA/role bit folded in the high bit (so the key names a SPECIFIC hull's face independent of iteration
// order); incVertId = the clipped point's provenance tag (an original incident LOCAL vertex index, or a packed
// (refEdge, incEdge) crossing under the intersect bit). bodyRole records which stored body owns the reference
// face (0 == bodyA owns it, 1 == bodyB owns it) — folded into refFaceId so a body swap leaves the key bit-equal.
// std430-packable as 4 x uint32 (16 bytes) — the GPU HullContactKey mirror the warmhull_key.comp memcmp compares.
struct HullContactKey {
    uint32_t bodyA     = 0;
    uint32_t bodyB     = 0;
    uint32_t refFaceId = 0;   // (refIsAStored << 31) | refFace  — the reference-face identity (hull + index)
    uint32_t incVertId = 0;   // the incident source-feature provenance tag (vertex idx OR (refEdge,incEdge))
};

constexpr uint32_t kRefFaceRoleBit = 0x80000000u;   // high bit of refFaceId: 1 == the STORED bodyB owns the ref face

// ----- MakeHullContactKey(bodyAIdx, bodyBIdx, refIsA, refFace, incTag) -> HullContactKey --------------------
// Order-normalize the body indices (swap so bodyA < bodyB). refIsA is "the RAW-order body A owns the reference
// face" (manifold.h:502's refIsA). We fold WHICH STORED body owns the reference face into refFaceId's high bit
// so the key is invariant under a body swap: if the raw order is already normalized (bodyAIdx <= bodyBIdx) the
// stored owner == refIsA-A; if swapped, the stored owner flips. PURE INT32 (a compare + a swap + shifts/ors —
// NO products). The shader copies THIS body VERBATIM.
inline HullContactKey MakeHullContactKey(uint32_t bodyAIdx, uint32_t bodyBIdx,
                                         bool refIsA, uint32_t refFace, uint32_t incTag) {
    HullContactKey k;
    bool swapped;
    if (bodyAIdx <= bodyBIdx) { k.bodyA = bodyAIdx; k.bodyB = bodyBIdx; swapped = false; }
    else                      { k.bodyA = bodyBIdx; k.bodyB = bodyAIdx; swapped = true; }
    // Which STORED body owns the reference face? The reference face belongs to RAW body A iff refIsA. After the
    // order-normalize swap, RAW-A is STORED-B and vice-versa — so the stored owner is bodyB iff (refIsA XOR
    // swapped) is FALSE... work it out: storedOwnerIsB == (refIsA && swapped) || (!refIsA && !swapped)
    //   == (refIsA == swapped). Fold that single bit into refFaceId's high bit; the low bits are the face index.
    const bool storedOwnerIsB = (refIsA == swapped);
    k.refFaceId = (storedOwnerIsB ? kRefFaceRoleBit : 0u) | (refFace & 0x7FFFFFFFu);
    k.incVertId = incTag;
    return k;
}

// ----- HullContactKeysEqual(a, b) -> bool: field-by-field equality (the cache match predicate). PURE INT32. ---
inline bool HullContactKeysEqual(const HullContactKey& a, const HullContactKey& b) {
    return a.bodyA == b.bodyA && a.bodyB == b.bodyB
        && a.refFaceId == b.refFaceId && a.incVertId == b.incVertId;
}

// ----- HullContactKeyHash(k) -> uint32_t: a deterministic integer hash of the four fields -------------------
// A FIXED bit-mix over the four small uint32 fields (bodyA/bodyB < a few thousand, refFaceId a small index +
// the high role bit, incVertId a small vertex idx OR a packed (refEdge,incEdge) + the intersect bit). PURE
// INT32 — only shifts + xors + adds, NO products, NO int64, NO float -> MSL-native. This is persist.h:84-94's
// ContactKeyHash avalanche idiom (the same (a<<20)^(b<<8)^(c<<4)^d packing skeleton, adapted to the hull
// fields) so the GPU shader copies it VERBATIM. The final xor-shift avalanche spreads the bits for buckets.
inline uint32_t HullContactKeyHash(const HullContactKey& k) {
    // The packing of the four fields (the persist.h shift/xor mix; refFaceId/incVertId carry their high tag bits).
    uint32_t h = (k.bodyA << 20) ^ (k.bodyB << 8) ^ (k.refFaceId << 4) ^ k.incVertId;
    // The fixed avalanche (xorshift-style; shifts + xor + add only — no products), copied from persist.h.
    h ^= h >> 15;
    h += (h << 7);
    h ^= h >> 11;
    return h;
}

// ----- KeyedHullContact: one tagged manifold contact POINT (its world position + depth + its HullContactKey) --
// The output of BuildHullContactKeys per manifold point. point/depth mirror the convex::ContactManifold WH1
// tags (BYTE-EQUAL to manifold::HullContactMulti by construction); key is the geometric-provenance feature ID.
struct KeyedHullContact {
    FxVec3         point;
    fx             depth = 0;
    HullContactKey key;
    // The RAW provenance (the order-UN-normalized inputs the GPU warmhull_key.comp consumes + order-normalizes;
    // the showcase feeds THESE to the shader + checks MakeHullContactKey(raw) == key). bodyAIdx/bodyBIdx are the
    // caller's GLOBAL body indices in raw order; refIsA/refFace/incTag are the geometric provenance.
    uint32_t bodyAIdx = 0;
    uint32_t bodyBIdx = 0;
    uint32_t refIsA   = 0;   // 0/1 (the RAW-order body A owns the reference face)
    uint32_t refFace  = 0;
    uint32_t incTag   = 0;
};

// ----- KeyedHullManifold: the per-pair tagged manifold (count 0-4 KeyedHullContact + the EPA normal) ---------
struct KeyedHullManifold {
    uint32_t         count = 0;
    KeyedHullContact pts[4];
    FxVec3           normal;
};

// =========================================================================================================
// BuildHullContactKeysForPair — run the FROZEN narrowphase + HullContactMulti for ONE pair, then TAG each
// manifold point with its HullContactKey. It MIRRORS manifold::HullManifoldFromEpa's reference/incident face
// selection + clip + keep + reduce EXACTLY (the SAME refIsA, refFace, incFace, the SAME candidate-keep depth
// test, the SAME deepest-first reduce) but runs ClipFaceAgainstFaceTagged instead of the frozen clip so it can
// carry the provenance tag through the identical control flow. The OUTPUT POSITIONS are byte-equal to
// HullContactMulti's (asserted in the showcase/test); the tags name the (refFace, incident source feature)
// provenance. bodyAIdx/bodyBIdx are the GLOBAL body indices feeding the order-normalized key.
// =========================================================================================================
inline KeyedHullManifold BuildHullContactKeysForPair(uint32_t bodyAIdx, const FxBody& bodyA, const FxHull& hullA,
                                                     uint32_t bodyBIdx, const FxBody& bodyB, const FxHull& hullB) {
    KeyedHullManifold out;   // count 0
    // The FROZEN manifold (the contract reference WH1 tags — positions/depths/normal come from HERE).
    const convex::ContactManifold m = manifold::HullContactMulti(bodyA, hullA, bodyB, hullB);
    if (m.count == 0) return out;
    out.normal = m.normal;

    // Re-derive the SAME reference/incident selection HullManifoldFromEpa made (== manifold.h:486-513), so the
    // tagged clip runs over the SAME faces. We re-run gjk::Gjk -> gjk::Epa (the FROZEN narrowphase) to get the
    // SAME EPA normal n the manifold used.
    const gjk::GjkResult g = gjk::Gjk(hullA, bodyA, hullB, bodyB);
    const FxHullFaces facesA = manifold::BuildCanonicalFaces(hullA);
    const FxHullFaces facesB = manifold::BuildCanonicalFaces(hullB);

    // If the FROZEN manifold fell back to the single-point witness (non-canonical hull / degenerate / empty
    // clip), there is no clip provenance: tag the point with a sentinel refFace/incTag (a deterministic
    // cold-start key). We detect the fallback by: no overlap, or no face tables, or the clip yields nothing.
    bool haveClip = g.overlap && facesA.faceCount != 0 && facesB.faceCount != 0;

    // Per-manifold-point tag (index-aligned with m.points[]); default = the fallback sentinel.
    uint32_t pointTag[4]   = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    bool     refIsA        = true;
    uint32_t refFaceIdx    = 0xFFFFu;   // sentinel (fallback) until the clip path fills it

    if (haveClip) {
        const gjk::EpaResult epa = gjk::Epa(hullA, bodyA, hullB, bodyB, g.simplex);
        const FxVec3 n = epa.normal;   // UNIT, A->B (== HullManifoldFromEpa)

        // Reference-hull selection (== manifold.h:494-513), pure integer, the SAME compares.
        const uint32_t sfA = manifold::SupportFace(hullA, facesA, bodyA, n);
        const FxVec3   nfA = manifold::FaceNormalWorld(hullA, facesA, bodyA, sfA);
        const fx       alignA = FxDot(nfA, n);
        const FxVec3   negN = FxVec3{-n.x, -n.y, -n.z};
        const uint32_t sfB = manifold::SupportFace(hullB, facesB, bodyB, negN);
        const FxVec3   nfB = manifold::FaceNormalWorld(hullB, facesB, bodyB, sfB);
        const fx       alignB = FxDot(nfB, negN);
        refIsA = (alignA >= alignB);   // tie-break A (>=) — IDENTICAL to manifold.h:502

        const FxHull&      refHull  = refIsA ? hullA : hullB;
        const FxBody&      refBody  = refIsA ? bodyA : bodyB;
        const FxHullFaces& refFaces = refIsA ? facesA : facesB;
        const uint32_t     refFace  = refIsA ? sfA : sfB;
        const FxVec3       refN     = refIsA ? nfA : nfB;

        const FxHull&      incHull  = refIsA ? hullB : hullA;
        const FxBody&      incBody  = refIsA ? bodyB : bodyA;
        const FxHullFaces& incFaces = refIsA ? facesB : facesA;
        const uint32_t     incFace  = manifold::IncidentFace(incHull, incFaces, incBody, refN);

        if (refFace < refFaces.faceCount && incFace < incFaces.faceCount) {
            refFaceIdx = refFace;
            const FxVec3 refC = manifold::FaceCentroidWorld(refHull, refFaces, refBody, refFace);

            // The TAGGED clip (positions byte-equal to the frozen clip HullManifoldFromEpa ran).
            TaggedVert clipped[kMaxClipVerts];
            int clipN = 0;
            ClipFaceAgainstFaceTagged(refHull, refBody, refFaces, refFace,
                                      incHull, incBody, incFaces, incFace, clipped, clipN);

            // The candidate-keep (== manifold.h:531-538): keep clipped verts with depth d = FxDot(refN, refC -
            // vertex) >= 0, in clip order, carrying the tag.
            FxVec3   candPts[kMaxClipVerts];
            fx       candDepth[kMaxClipVerts];
            uint32_t candTag[kMaxClipVerts];
            int      candN = 0;
            for (int k = 0; k < clipN; ++k) {
                const FxVec3 rel = FxVec3{refC.x - clipped[k].pos.x, refC.y - clipped[k].pos.y,
                                          refC.z - clipped[k].pos.z};
                const fx d = FxDot(refN, rel);
                if (d >= 0) { candPts[candN] = clipped[k].pos; candDepth[candN] = d;
                              candTag[candN] = clipped[k].tag; ++candN; }
            }
            if (candN > 0) {
                // The deepest-first reduce (== manifold.h:543-554), carrying the tag with each kept point.
                int deepest = 0;
                for (int k = 1; k < candN; ++k) if (candDepth[k] > candDepth[deepest]) deepest = k;
                uint32_t orderTag[4];
                orderTag[0] = candTag[deepest];
                uint32_t cnt = 1;
                for (int k = 0; k < candN && cnt < 4; ++k) {
                    if (k == deepest) continue;
                    orderTag[cnt] = candTag[k];
                    ++cnt;
                }
                // The reduce order here is IDENTICAL to HullManifoldFromEpa's, so orderTag[i] aligns with
                // m.points[i] (the byte-equal positions). Copy the tags into pointTag[].
                for (uint32_t i = 0; i < cnt && i < 4; ++i) pointTag[i] = orderTag[i];
            }
        }
    }

    // Emit the keyed contacts: positions/depths come from the FROZEN manifold m (the contract); the key is the
    // geometric provenance (refIsA + refFaceIdx + pointTag). A fallback point (no clip provenance) carries the
    // sentinel refFace/tag -> a deterministic distinct cold-start key.
    out.count = m.count;
    for (uint32_t i = 0; i < m.count && i < 4; ++i) {
        out.pts[i].point = m.points[i];
        out.pts[i].depth = m.depths[i];
        out.pts[i].key   = MakeHullContactKey(bodyAIdx, bodyBIdx, refIsA, refFaceIdx, pointTag[i]);
        out.pts[i].bodyAIdx = bodyAIdx;
        out.pts[i].bodyBIdx = bodyBIdx;
        out.pts[i].refIsA   = refIsA ? 1u : 0u;
        out.pts[i].refFace  = refFaceIdx;
        out.pts[i].incTag   = pointTag[i];
    }
    return out;
}

// ----- BuildHullContactKeys(pairs): the per-pair battery -> the flat list of KeyedHullContact ----------------
// A pair is (bodyAIdx, bodyA, hullA, bodyBIdx, bodyB, hullB). Runs BuildHullContactKeysForPair per pair in
// FIXED order and flattens every manifold point into one list (the deterministic key set the showcase/shader
// pack + measure). Pure integer (the int64 manifold + the int32 key), fixed scan order -> deterministic.
struct HullKeyPair {
    uint32_t bodyAIdx; FxBody bodyA; FxHull hullA;
    uint32_t bodyBIdx; FxBody bodyB; FxHull hullB;
};

inline std::vector<KeyedHullContact> BuildHullContactKeys(const std::vector<HullKeyPair>& pairs) {
    std::vector<KeyedHullContact> out;
    for (const HullKeyPair& p : pairs) {
        const KeyedHullManifold km = BuildHullContactKeysForPair(p.bodyAIdx, p.bodyA, p.hullA,
                                                                 p.bodyBIdx, p.bodyB, p.hullB);
        for (uint32_t i = 0; i < km.count && i < 4; ++i) out.push_back(km.pts[i]);
    }
    return out;
}

// ----- HullKeyMeasure: the deterministic summary over a set of hull contact keys ----------------------------
// totalKeys = the total number of keys measured; distinctKeys = the number of DISTINCT keys (the SAME contact
// re-derived collapses); maxHashCollisions = the largest number of DISTINCT keys sharing one hash (0 == every
// distinct key hashed uniquely). Pure integer, fixed scan order -> deterministic. The persist.h KeyMeasure twin.
struct HullKeyMeasure {
    uint32_t totalKeys         = 0;
    uint32_t distinctKeys      = 0;
    uint32_t maxHashCollisions = 0;
};

// MeasureHullKeys(keys): scan the key array; count total, distinct (by field tuple), and the max number of
// distinct keys colliding on a single hash. Pure integer, FIXED O(n^2) scan order (the contact sets are tiny)
// -> deterministic (two runs byte-identical). The persist.h MeasureKeys twin.
inline HullKeyMeasure MeasureHullKeys(const std::vector<HullContactKey>& keys) {
    HullKeyMeasure m;
    m.totalKeys = (uint32_t)keys.size();
    for (size_t i = 0; i < keys.size(); ++i) {
        bool seen = false;
        for (size_t j = 0; j < i; ++j)
            if (HullContactKeysEqual(keys[i], keys[j])) { seen = true; break; }
        if (!seen) ++m.distinctKeys;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        bool firstOcc = true;
        for (size_t j = 0; j < i; ++j)
            if (HullContactKeysEqual(keys[i], keys[j])) { firstOcc = false; break; }
        if (!firstOcc) continue;
        const uint32_t hi = HullContactKeyHash(keys[i]);
        uint32_t sharing = 0;
        for (size_t j = 0; j < keys.size(); ++j) {
            bool jFirst = true;
            for (size_t p = 0; p < j; ++p)
                if (HullContactKeysEqual(keys[j], keys[p])) { jFirst = false; break; }
            if (!jFirst) continue;
            if (HullContactKeyHash(keys[j]) == hi) ++sharing;
        }
        const uint32_t collisions = sharing > 0 ? sharing - 1u : 0u;
        if (collisions > m.maxHashCollisions) m.maxHashCollisions = collisions;
    }
    return m;
}

// =========================================================================================================
// Slice WH2 — THE KEYED MANIFOLD + PERSISTENT CACHE (the structural store). APPENDED after MeasureHullKeys
// (WH1's lines above + manifold.h/gjk.h/persist.h/convex.h/fric.h/broad.h/ccd.h/fpx.h BYTE-FROZEN). WH1 built
// the deterministic hull contact feature ID (HullContactKey — the int32 MSL-native key that names a contact
// across frames). WH2 builds the PERSISTENT STORE that key unlocks: a per-frame cache that matches THIS tick's
// manifold points to LAST tick's ACCUMULATED contact impulses by key — so next tick's solver (WH3) can warm-
// start each point from the impulse it carried last frame. This is the structural backbone of warm-starting
// (no new physics yet — the PS2 step of the persist.h template). The control flow MIRRORS persist::MatchCache/
// UpdateCache (persist.h:221-248) over the WH1 hull key + the multi-point hull manifold (persist's
// BuildKeyedManifold is box-only — WH2's hull version is new, reusing manifold::HullContactMulti + WH1's
// BuildHullContactKeysForPair).
//
// THE int64 REALITY (the WH1/MF3 lesson): the manifold WH2 caches is built by the int64 manifold::
// HullContactMulti (GJK/EPA + the Sutherland-Hodgman clip), so shaders/warmhull_cache.comp.hlsl is
// VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT in the metal_headless hf_gen_msl list —
// UNLIKE WH1's warmhull_key.comp (the KEY alone is pure int32 -> MSL-native). The Vulkan --wh2-cache-shot
// dispatches warmhull_cache.comp (one thread per pair: it consumes the host-built keyed manifold points — the
// int64-derived positions/depths/keys — and MATCHES the persistent cache SSBO in FIXED order, copying
// MatchHullCache's body VERBATIM) -> the GPU matched KeyedHullManifoldWH2[] is memcmp'd BIT-EXACT vs the CPU
// reference; the Metal --wh2-cache runs the CPU BuildKeyedHullManifold + MatchHullCache (byte-identical by
// construction, the persist_cache.comp convention).

// ----- KeyedHullManifoldWH2: a convex::ContactManifold + a PARALLEL HullContactKey + per-point normalImpulse ---
// The frozen convex::ContactManifold (positions/depths/normal/count, manifold.h:292 — reused verbatim) +
// keys[i] (the WH1 geometric-provenance key for manifold point i) + normalImpulse[i] (the per-point accumulated
// NORMAL impulse, zeroed on cold-start, seeded from the cache on a warm match). std430-packable (the manifold
// POD + 4 x (4-uint key) + 4 x fx). WH1 already defines a small KeyedHullManifold (count/pts/normal) used by
// BuildHullContactKeysForPair; WH2 uses a DISTINCT name (KeyedHullManifoldWH2) so WH1's type stays byte-frozen.
struct KeyedHullManifoldWH2 {
    convex::ContactManifold manifold;          // positions/depths/normal/count — the frozen POD, reused verbatim
    HullContactKey          keys[4];           // keys[i] <-> manifold.points[i] for i in [0, manifold.count)
    fx                      normalImpulse[4] = {0, 0, 0, 0};   // per-point accumulated normal impulse (0 = cold)
};

// ----- BuildKeyedHullManifold(bodyAIdx, bodyA, hullA, bodyBIdx, bodyB, hullB) -> KeyedHullManifoldWH2 ----------
// Run manifold::HullContactMulti (the FROZEN narrowphase — positions/depths/normal come from HERE) then tag each
// of its 1-4 points with its WH1 key via BuildHullContactKeysForPair (the byte-identical tagged clip). The keys
// are aligned index-for-index with the manifold points (BuildHullContactKeysForPair emits in the SAME deepest-
// first reduce order HullManifoldFromEpa uses). normalImpulse starts 0 (the cold-start contract; MatchHullCache
// fills the matched ones). Pure integer, FIXED order -> bit-identical CPU<->Vulkan<->Metal.
inline KeyedHullManifoldWH2 BuildKeyedHullManifold(uint32_t bodyAIdx, const FxBody& bodyA, const FxHull& hullA,
                                                   uint32_t bodyBIdx, const FxBody& bodyB, const FxHull& hullB) {
    KeyedHullManifoldWH2 out;   // manifold count 0, keys zeroed, impulses 0
    // The FROZEN manifold (the contract — positions/depths/normal).
    out.manifold = manifold::HullContactMulti(bodyA, hullA, bodyB, hullB);
    if (out.manifold.count == 0) return out;
    // The WH1 keyed manifold (the SAME positions, plus the per-point geometric-provenance key).
    const KeyedHullManifold km = BuildHullContactKeysForPair(bodyAIdx, bodyA, hullA, bodyBIdx, bodyB, hullB);
    for (uint32_t i = 0; i < out.manifold.count && i < 4; ++i) {
        out.keys[i] = (i < km.count) ? km.pts[i].key : HullContactKey{};
        out.normalImpulse[i] = 0;   // cold-start
    }
    return out;
}

// ----- CachedHullContact: one cached hull contact's key + its persisted accumulated normal impulse -----------
// key = the WH1 HullContactKey (the deterministic identity); normalImpulse = LAST tick's accumulated normal
// impulse. std430-packable as 4 x uint32 (the key) + 1 x int32 (the impulse) = 5 x int32 (20 bytes) — the GPU
// CachedHullContact mirror the warmhull_cache SSBO scan reads. (WH2 caches the NORMAL impulse only — hull
// friction is a separate future flagship.)
struct CachedHullContact {
    HullContactKey key;
    fx             normalImpulse = 0;
};

// ----- HullCache: the store — LAST tick's accumulated normal impulses keyed by HullContactKey ---------------
// A flat vector + a FIXED linear scan (the contact sets are tiny; HullContactKeyHash is available for a bucket
// optimization, but the deterministic baseline is the fixed-order scan — the persist.h PersistentCache shape).
// UpdateHullCache rewrites it to EXACTLY this tick's contacts (so absent keys are evicted); MatchHullCache reads
// it (read-only). FIXED order -> deterministic.
struct HullCache {
    std::vector<CachedHullContact> entries;
};

// ----- MatchHullCache(cache, keyed) -> mutates keyed.normalImpulse: inherit prior impulses by key ------------
// For each manifold point i in FIXED order [0, keyed.manifold.count): scan the cache in FIXED order (entries[0..)
// for the FIRST entry whose key equals keyed.keys[i] (HullContactKeysEqual); if found, copy the cached
// normalImpulse into keyed.normalImpulse[i] (the WARM inherit); if not found, leave it 0 (a fresh contact cold-
// starts — BuildKeyedHullManifold already zeroed it). FIXED scan order -> deterministic. The shader copies THIS
// body VERBATIM (the GPU==CPU memcmp make-or-break).
inline void MatchHullCache(const HullCache& cache, KeyedHullManifoldWH2& keyed) {
    for (uint32_t i = 0; i < keyed.manifold.count && i < 4; ++i) {
        for (size_t e = 0; e < cache.entries.size(); ++e) {
            if (HullContactKeysEqual(cache.entries[e].key, keyed.keys[i])) {
                keyed.normalImpulse[i] = cache.entries[e].normalImpulse;
                break;   // the FIRST matching entry wins (fixed scan order)
            }
        }
    }
}

// ----- UpdateHullCache(cache, keyed) -> rewrites cache: store THIS tick's contacts, evict absent keys --------
// After a (future WH3) solve, REPLACE the cache with exactly this tick's contacts: clear it, then for each point
// i in FIXED order append {keyed.keys[i], keyed.normalImpulse[i]}. Keys not present this tick are thereby EVICTED
// (the new cache IS exactly this tick's set). FIXED order -> deterministic. (WH2 proves the match + the evict;
// the impulses it stores are whatever the manifold carries — synthesized in WH2's showcase, the real solved
// impulses in WH3.) Mirrors persist::UpdateCache (persist.h:240-248).
inline void UpdateHullCache(HullCache& cache, const KeyedHullManifoldWH2& keyed) {
    cache.entries.clear();
    for (uint32_t i = 0; i < keyed.manifold.count && i < 4; ++i)
        cache.entries.push_back(CachedHullContact{keyed.keys[i], keyed.normalImpulse[i]});
}

// ----- HullCacheMeasure: the deterministic match summary over one keyed manifold against a cache -------------
// contacts = keyed.manifold.count; matched = the number of points whose key was found in the cache (inherited);
// coldStart = the number NOT found (cold-started at zero) = contacts - matched. Pure integer, FIXED scan order
// -> deterministic. The showcase prints + asserts (matched + coldStart == contacts). The persist.h CacheMeasure
// twin.
struct HullCacheMeasure {
    uint32_t contacts  = 0;
    uint32_t matched   = 0;
    uint32_t coldStart = 0;
};

// MeasureHullCache(cache, keyed): count, in FIXED order, how many of keyed's points have a matching cache key.
// Read-only (does NOT mutate keyed). Deterministic.
inline HullCacheMeasure MeasureHullCache(const HullCache& cache, const KeyedHullManifoldWH2& keyed) {
    HullCacheMeasure m;
    m.contacts = keyed.manifold.count;
    for (uint32_t i = 0; i < keyed.manifold.count && i < 4; ++i) {
        bool found = false;
        for (size_t e = 0; e < cache.entries.size(); ++e)
            if (HullContactKeysEqual(cache.entries[e].key, keyed.keys[i])) { found = true; break; }
        if (found) ++m.matched; else ++m.coldStart;
    }
    return m;
}

// =========================================================================================================
// Slice WH3 — THE ACCUMULATED WARM-STARTED SOLVER (the core solve). APPENDED after MeasureHullCache (WH1/WH2's
// lines above + manifold.h/gjk.h/persist.h/convex.h/fric.h/fpx.h BYTE-FROZEN). WH1 built the hull contact
// feature ID; WH2 the persistent cache that matches THIS tick's manifold points to LAST tick's accumulated
// impulses by key. WH3 is the ENGINE of the flagship: an ACCUMULATED, warm-started NORMAL sequential-impulse
// solver that primes each contact point from its cached impulse (WH2) + applies only the per-iteration DELTA
// (clamping the ACCUMULATED total >= 0). This is the direct hull-normal generalization of persist::
// SolveFrictionWarm (persist.h:335) — the accumulated form convex::SolveManifoldImpulse (the NON-accumulated
// per-point kernel, convex.h:651) "cannot be warm-started" (persist.h:284-290) because it keeps no running
// total. The accumulated form converges to a consistent island equilibrium across ticks, removing the residual-
// torque SOURCE the #25 tower-collapse note (convex.h:758-763) damps with the global angular crutch.
//
// THE int64 REALITY (the MF4/WH2 lesson): the manifold it solves is built by the int64 manifold::
// HullContactMulti (GJK/EPA + the SH clip), so shaders/warmhull_warm.comp.hlsl is VULKAN-SPIR-V-ONLY (NOT in
// hf_gen_msl), single-thread over the small world copying StepWarmHullWorldN VERBATIM, chunked 1 tick/dispatch
// (the StepWarmHullWorld step is HEAVIER than the hardened step -> the documented Windows ~2s TDR rule). The
// Metal --wh3-warm runs the CPU StepWarmHullWorldN (byte-identical by construction), while the Vulkan side
// carries the GPU==CPU memcmp.

// Pull the world-step helpers WH3 uses (REUSE — convex.h/manifold.h/fpx.h read-only/byte-frozen).
using convex::FxScale;
using convex::FxMat3;
using convex::FxMat3MulVec;
using convex::IsDynamic;
using convex::ConvexStepConfig;
using fpx::FxLength;
using manifold::FxHullFaces;
using manifold::BuildCanonicalFaces;
using manifold::FxHullInertiaBodyFull;
using manifold::WorldInvInertiaFull;
using manifold::HullContactMulti;
using gjk::MeasureHullStack;
using gjk::HullStackMeasure;

// ----- SolveHullManifoldWarm: the ACCUMULATED normal sequential-impulse solver with the warm-start prime ----
// Mutates bodyA/bodyB vel+angVel AND keyed.normalImpulse[] (the accumulators). The accumulators ARRIVE SEEDED
// (from MatchHullCache for a matched contact, or zero for a cold one). The PINNED steps (the shader copies THIS
// body VERBATIM):
//   (0) SIGN-CORRECT the manifold normal to point A->B ONCE (== convex::SolveManifoldImpulse's flip).
//   (1) PRIME ONCE: for each point apply the seeded TOTAL normal impulse J = n*normalImpulse[p] to both bodies
//       (re-inject last tick's converged impulse so velocities start near the solved state). A zero seed primes
//       nothing (a cold contact).
//   (2) `iters` Gauss-Seidel sweeps; per point in FIXED order, ACCUMULATED (the convex.h:645-651 effective-mass
//       k + the contact-point velocity math, with the full WorldInvInertiaFull tensor):
//       vn = (vpB-vpA).n; k = invMa+invMb + n·((invIaW·(rA×n))×rA) + n·((invIbW·(rB×n))×rB);
//       jnTotal = normalImpulse[p] + fxdiv(-(1+e)·vn, k); CLAMP jnTotal >= 0 (a contact only PUSHES);
//       apply the DELTA dJ = n*(jnTotal - normalImpulse[p]); normalImpulse[p] = jnTotal.
//   (3) the converged accumulators stay in keyed.normalImpulse[] -> UpdateHullCache persists them for next tick.
// PURE INTEGER, FIXED order -> bit-identical CPU<->Vulkan<->Metal. (NOTE: UNLIKE the NON-accumulated kernel,
// the warm form does NOT early-skip on vn>=0 — the accumulated clamp >= 0 already prevents a pull, and skipping
// would discard the persisted total; this is exactly the persist::SolveFrictionWarm normal-branch shape.)
inline void SolveHullManifoldWarm(fpx::FxBody& bodyA, fpx::FxBody& bodyB,
                                  const FxMat3& invIaW, const FxMat3& invIbW,
                                  KeyedHullManifoldWH2& keyed, fx restitution, uint32_t iters) {
    if (keyed.manifold.count == 0) return;

    // (0) sign-correct the normal A->B ONCE (== convex::SolveManifoldImpulse).
    FxVec3 n = keyed.manifold.normal;
    if (FxDot(n, FxSub(bodyB.pos, bodyA.pos)) < 0) n = FxVec3{-n.x, -n.y, -n.z};

    const fx invMassA = bodyA.invMass;
    const fx invMassB = bodyB.invMass;
    const uint32_t cnt = keyed.manifold.count < 4u ? keyed.manifold.count : 4u;

    // (1) PRIME ONCE: re-inject the seeded TOTAL normal impulse at every point (the warm-start kick).
    for (uint32_t pi = 0; pi < cnt; ++pi) {
        const fx seed = keyed.normalImpulse[pi];
        if (seed == 0) continue;   // a cold contact primes nothing
        const FxVec3 p  = keyed.manifold.points[pi];
        const FxVec3 rA = FxSub(p, bodyA.pos);
        const FxVec3 rB = FxSub(p, bodyB.pos);
        const FxVec3 J  = FxScale(n, seed);
        bodyA.vel    = FxSub(bodyA.vel, FxScale(J, invMassA));
        bodyA.angVel = FxSub(bodyA.angVel, FxMat3MulVec(invIaW, FxCross(rA, J)));
        bodyB.vel    = FxAdd(bodyB.vel, FxScale(J, invMassB));
        bodyB.angVel = FxAdd(bodyB.angVel, FxMat3MulVec(invIbW, FxCross(rB, J)));
    }

    // (2) the accumulated Gauss-Seidel sweeps (apply only the DELTA each time; clamp the ACCUMULATED total >= 0).
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t pi = 0; pi < cnt; ++pi) {
            const FxVec3 p  = keyed.manifold.points[pi];
            const FxVec3 rA = FxSub(p, bodyA.pos);
            const FxVec3 rB = FxSub(p, bodyB.pos);
            const FxVec3 vpA = FxAdd(bodyA.vel, FxCross(bodyA.angVel, rA));
            const FxVec3 vpB = FxAdd(bodyB.vel, FxCross(bodyB.angVel, rB));
            const fx vn = FxDot(FxSub(vpB, vpA), n);
            const FxVec3 raxn = FxCross(rA, n);
            const FxVec3 rbxn = FxCross(rB, n);
            const fx angA = FxDot(n, FxCross(FxMat3MulVec(invIaW, raxn), rA));
            const fx angB = FxDot(n, FxCross(FxMat3MulVec(invIbW, rbxn), rB));
            const fx k = invMassA + invMassB + angA + angB;
            if (k <= 0) continue;   // degenerate -> skip (== the kernel)
            fx jnTotal = keyed.normalImpulse[pi] + fxdiv(-fxmul(kOne + restitution, vn), k);
            if (jnTotal < 0) jnTotal = 0;                    // clamp the ACCUMULATED total >= 0
            const fx applied = jnTotal - keyed.normalImpulse[pi];
            const FxVec3 J = FxScale(n, applied);            // apply only the DELTA
            bodyA.vel    = FxSub(bodyA.vel, FxScale(J, invMassA));
            bodyA.angVel = FxSub(bodyA.angVel, FxMat3MulVec(invIaW, FxCross(rA, J)));
            bodyB.vel    = FxAdd(bodyB.vel, FxScale(J, invMassB));
            bodyB.angVel = FxAdd(bodyB.angVel, FxMat3MulVec(invIbW, FxCross(rB, J)));
            keyed.normalImpulse[pi] = jnTotal;               // write back the converged accumulated total
        }
    }
}

// ----- WarmHullStepConfig: a thin alias of convex::ConvexStepConfig (same knobs; WH3 adds none) -------------
using WarmHullStepConfig = convex::ConvexStepConfig;

// ----- StepWarmHullWorld(world, cache, cfg): ONE deterministic WARM-started hardened tick -------------------
// manifold::StepHullWorldHardened (manifold.h:809) with EXACTLY step (3) — the impulse solve — replaced by the
// warm-started accumulated solve over the persistent cache. Steps (1) integrate+damp, (2) FULL inertia, (4)
// position de-penetration are UNCHANGED from the hardened step (copied VERBATIM). Step (3) — per overlapping
// pair in the FIXED i<j order (the persist::StepWarmWorld shape):
//   - BuildKeyedHullManifold (re-derived from the CURRENT positions — WH2);
//   - MatchHullCache to warm-seed each point's normalImpulse from `cache` (last tick's accumulated totals);
//   - SolveHullManifoldWarm with cfg.solveIters ACCUMULATED sweeps (the iteration lives INSIDE the solver —
//     prime ONCE + accumulated Gauss-Seidel; the mutation is IN PLACE so later pairs see earlier updates);
//   - capture the CONVERGED keyed manifolds, then rewrite `cache` to EXACTLY this tick's solved contacts
//     (UpdateHullCache semantics over ALL active pairs) so the accumulated impulses persist for next tick +
//     absent keys are evicted.
// The cache is the per-tick mutable replayable state (carried in/out). Pure integer, FIXED order -> identical
// CPU/GPU. The headline: with cfg.angDamp = kOne (OFF) the warm solve HOLDS the stack (the removed-torque-source
// proof) where the frozen hardened step with damping OFF does NOT.
inline void StepWarmHullWorld(HullWorld& world, HullCache& cache, const convex::ConvexStepConfig& cfg) {
    const size_t n = world.bodies.size();

    // (1) predict-integrate dynamic bodies + per-tick damping (== StepHullWorldHardened step 1, VERBATIM).
    for (size_t i = 0; i < n; ++i) {
        if (IsDynamic(world.bodies[i])) {
            fpx::IntegrateBodyFull(world.bodies[i], cfg.gravity, cfg.dt);
            if (cfg.linDamp != kOne) world.bodies[i].vel = FxScale(world.bodies[i].vel, cfg.linDamp);
            if (cfg.angDamp != kOne) world.bodies[i].angVel = FxScale(world.bodies[i].angVel, cfg.angDamp);
        }
    }

    // (2) world inverse inertias once/tick — the FULL tensor (== StepHullWorldHardened step 2, VERBATIM).
    std::vector<FxMat3> invIW(n);
    for (size_t i = 0; i < n; ++i) {
        const FxHullFaces faces = BuildCanonicalFaces(world.hulls[i]);
        const FxMat3 invIbody = FxHullInertiaBodyFull(world.hulls[i], faces, world.bodies[i].invMass);
        invIW[i] = WorldInvInertiaFull(world.bodies[i], invIbody);
    }

    // (3 — THE SWAP) the WARM-started accumulated solve over the persistent cache, in the FIXED i<j order — per
    // pair: BuildKeyedHullManifold (re-derived from the CURRENT positions) -> MatchHullCache (seed each point's
    // accumulator from LAST tick — the warm-start) -> SolveHullManifoldWarm with cfg.solveIters ACCUMULATED
    // sweeps (prime ONCE + the accumulated Gauss-Seidel — the iteration lives INSIDE the solver, the natural
    // per-pair analog of persist::StepWarmWorld; the mutation is IN PLACE so later pairs see earlier updates).
    // The CONVERGED keyed manifolds are captured to rebuild the cache (step 5 below). Skip static-static.
    struct WarmPair { KeyedHullManifoldWH2 keyed; };
    std::vector<WarmPair> wp;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;  // static-static
            KeyedHullManifoldWH2 keyed = BuildKeyedHullManifold(
                (uint32_t)i, world.bodies[i], world.hulls[i],
                (uint32_t)j, world.bodies[j], world.hulls[j]);
            if (keyed.manifold.count == 0) continue;
            MatchHullCache(cache, keyed);   // seed the accumulators from last tick (the warm-start)
            SolveHullManifoldWarm(world.bodies[i], world.bodies[j], invIW[i], invIW[j],
                                  keyed, cfg.restitution, cfg.solveIters);   // prime once + solveIters accum sweeps
            wp.push_back(WarmPair{keyed});
        }
    }

    // (5) Rewrite the cache to EXACTLY this tick's solved contacts (the converged accumulated impulses) — absent
    // keys are thereby evicted (the new cache IS this tick's set). FIXED i<j pair / point order (== UpdateHullCache
    // over ALL active pairs). The accumulators carry the warm-start to next tick.
    cache.entries.clear();
    for (const WarmPair& w : wp) {
        const uint32_t c = w.keyed.manifold.count < 4u ? w.keyed.manifold.count : 4u;
        for (uint32_t pi = 0; pi < c; ++pi)
            cache.entries.push_back(CachedHullContact{w.keyed.keys[pi], w.keyed.normalImpulse[pi]});
    }

    // (4) position de-penetration (== StepHullWorldHardened step 4, VERBATIM — re-derives HullContactMulti).
    for (uint32_t pit = 0; pit < cfg.posIters; ++pit) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const fx invSum = world.bodies[i].invMass + world.bodies[j].invMass;
                if (invSum == 0) continue;
                const convex::ContactManifold m = HullContactMulti(world.bodies[i], world.hulls[i],
                                                                   world.bodies[j], world.hulls[j]);
                if (m.count == 0) continue;
                FxVec3 nrm = m.normal;
                if (FxDot(nrm, FxSub(world.bodies[j].pos, world.bodies[i].pos)) < 0)
                    nrm = FxVec3{-nrm.x, -nrm.y, -nrm.z};
                const fx excess = m.depths[0] - cfg.slop;
                if (excess <= 0) continue;
                const fx corrected = fxmul(excess, cfg.beta);
                const fx wi = fxdiv(world.bodies[i].invMass, invSum);
                const fx wj = kOne - wi;
                const FxVec3 ci = FxScale(nrm, fxmul(corrected, wi));
                const FxVec3 cj = FxScale(nrm, fxmul(corrected, wj));
                world.bodies[i].pos = FxSub(world.bodies[i].pos, ci);
                world.bodies[j].pos = FxAdd(world.bodies[j].pos, cj);
            }
        }
    }
    // (5) orientation was already integrated in step (1).
}

// ----- StepWarmHullWorldN(world, cache, cfg, ticks): run `ticks` StepWarmHullWorld steps -> the stack settles.
// The cache carries the accumulated impulses ACROSS ticks (the warm-start that converges the island equilibrium).
inline void StepWarmHullWorldN(HullWorld& world, HullCache& cache, const convex::ConvexStepConfig& cfg,
                               uint32_t ticks) {
    for (uint32_t t = 0; t < ticks; ++t) StepWarmHullWorld(world, cache, cfg);
}

// ----- WarmHullMeasure(world): the residual metric — DELEGATES to manifold::MeasureHullStack ----------------
// The deterministic rest (max dynamic speed) + interpenetration (max HullContact depth) summary the convergence
// proof compares (warm residual < cold residual at equal iters). A thin alias so the showcase/test name the WH3
// metric explicitly while reusing the frozen MeasureHullStack VERBATIM.
inline HullStackMeasure WarmHullMeasure(const HullWorld& world) {
    return MeasureHullStack(world);
}

}  // namespace warmhull
}  // namespace hf::sim
