#pragma once
// Slice MF1 — Hull Narrowphase Hardening: HULL FACE TOPOLOGY (the new primitive, the BEACHHEAD of FLAGSHIP
// #25: DETERMINISTIC HULL NARROWPHASE HARDENING, hf::sim::manifold). The frozen GJK/EPA hull narrowphase
// (engine/sim/gjk.h) is VERTICES-ONLY (gjk::FxHull is verts[] + count; gjk.h:53 notes "Faces are NOT needed
// for GJ1 — faces arrive in a later slice"). That later slice is MF1: it promotes the per-hull POLYGON FACE
// topology the whole flagship needs. gjk::HullContact (gjk.h:1155) hardcodes a SINGLE-POINT manifold so a
// face-resting hull TEETERS on one point; gjk::FxHullInvInertiaBody (gjk.h:1127) is an AABB-diagonal inertia
// approximation. The hardened multi-point manifold (MF2/MF3) and the full inertia tick (MF4) all CONSUME MF1's
// faces. MF1 itself is the geometry-table primitive ONLY: each canonical hull's deterministic, outward-wound
// polygon face table + the reference/incident face selectors + the per-face-colored float render.
//
// Header-only, namespace hf::sim::manifold, #include "sim/ccd.h" READ-ONLY (transitively gjk/broad/convex/fpx/
// fric/persist/grain — ALL BYTE-FROZEN; this header REUSES their helpers, it does NOT redefine the fixed-point
// format and it NEVER edits gjk.h or any sibling). The whole flagship's additivity rests on this: manifold.h
// is a BRAND-NEW sibling that only includes the frozen headers read-only.
//
// PURE INTEGER (the GJ1/CX1/CCD discipline): face construction + the outward normal + the support/incident
// selectors are all int64 FxDot/FxCross over Q16.16. The ONLY float is the render (FacesToRenderInstances) —
// OUTSIDE any bit-exact loop, the GJ6/BP6/CD6 render-capstone idiom. NO <cmath> in the geometry math.
//
// NO GPU COMPUTE: MF1 adds NO compute dispatch (so NO TDR/VUID risk); the only GPU work is the normal lit
// RENDER of the showcase (a draw, not a heavy dispatch). The numeric proofs are pure-CPU 0px-by-construction;
// the golden IMAGE is the one float crossing (render-only), in-band visresolve like every render capstone.

#include <cstdint>
#include <cstring>
#include <vector>

#include "sim/ccd.h"   // read-only: transitively gjk::FxHull/HullWorld/HullToRenderInstances/HullRenderMesh +
                        // convex::FxDot/FxCross/FxVec3/FxSub/FxAdd + fpx::FxBody/FxRotate/FxToFloat + the Q16.16
                        // toolbox. ccd.h is the outermost frozen sibling -> includes it (the spec's seam).

namespace hf::sim {
namespace manifold {

// Pull the Q16.16 helpers into this namespace (REUSE, do NOT redefine the fixed-point format).
using gjk::fx;
using gjk::kOne;
using gjk::FxVec3;
using gjk::FxHull;
using gjk::FxBody;
using gjk::HullWorld;
using convex::FxDot;     // the Q16.16 dot (int64 intermediate) — ranks the faces
using convex::FxCross;   // the Q16.16 cross (int64) — the raw face normal
using fpx::FxAdd;        // FxAdd/FxSub live in fpx.h (convex pulls them in transitively)
using fpx::FxSub;

// ----- The fixed canonical ceilings (std430-packable, like gjk::kMaxHullVerts at gjk.h:50). A face table is a
// fixed-size array so MF2/MF3 can hand it to a shader later. kMaxHullFaces=8 is the OCTA face count (the most
// faces of any canonical hull); kMaxFaceVerts=4 is the BOX/WEDGE quad (the widest face of any canonical hull).
// The canonical hulls only — a general quickhull face builder is OUT OF SCOPE (the documented YAGNI,
// gjk.h:1537), mirroring gjk::HullTriIndices.
constexpr uint32_t kMaxHullFaces = 8;
constexpr uint32_t kMaxFaceVerts = 4;

// ----- FxHullFaces: a hull's POLYGON FACE topology as a fixed-size, std430-packable table. faceCount faces;
// face f has vertCount[f] vertices whose indices into hull.verts are vertIdx[f][0..vertCount[f]). The winding
// recorded here is the gjk::HullTriIndices source winding — it is ARBITRARY: FaceNormalWorld canonicalizes the
// normal OUTWARD against the hull centroid (the gjk::EpaAddFace / HullToRenderMesh idiom), so the source
// winding does NOT matter. std430-packable: kMaxHullFaces x (kMaxFaceVerts uint indices + a uint vert count) +
// a uint face count. count==0 -> a non-canonical hull (no face table).
struct FxHullFaces {
    uint32_t vertIdx[kMaxHullFaces][kMaxFaceVerts] = {};   // per-face vertex indices into hull.verts
    uint32_t vertCount[kMaxHullFaces] = {};               // per-face vertex count (3 tri / 4 quad)
    uint32_t faceCount = 0;                               // # of faces (tetra:4 box:6 octa:8 wedge:5)
};

// ----- IsOctaHull(hull): the SAME structural axis-pole test gjk::HullTriIndices (gjk.h:1517-1521) /
// HullTypeColor (gjk.h:1548-1552) use to disambiguate the two 6-vert canonical hulls. An OCTA vertex is an axis
// pole -> exactly TWO zero coordinates; a WEDGE vertex never is. A PURE function of the verts (no float).
inline bool IsOctaHull(const FxHull& hull) {
    if (hull.count != 6) return false;
    for (uint32_t i = 0; i < 6; ++i) {
        const int zeros = (hull.verts[i].x == 0) + (hull.verts[i].y == 0) + (hull.verts[i].z == 0);
        if (zeros < 2) return false;
    }
    return true;
}

// ----- BuildCanonicalFaces(hull): the deterministic POLYGON faces of the four canonical hulls, by PROMOTING
// the canonical triangle groupings gjk::HullTriIndices (gjk.h:1502-1538) already encodes into outward-wound
// polygon faces (the coplanar triangles merged back into the polygon they triangulate). Keyed by vertex count
// (the canonical hulls have distinct counts: tetra 4 / box 8 / 6=octa-or-wedge), octa/wedge disambiguated by
// the SAME structural axis-pole test (IsOctaHull). Any other count -> faceCount=0 (the canonical hulls only —
// the documented YAGNI, gjk.h:1537; a general quickhull face builder is OUT OF SCOPE).
//   tetra (count 4): 4 triangular faces {0,1,2} {0,1,3} {0,2,3} {1,2,3}                       (gjk.h:1507)
//   box   (count 8): 6 QUAD faces -x{0,1,3,2} +x{4,5,7,6} -y{0,1,5,4} +y{2,3,7,6}
//                    -z{0,2,6,4} +z{1,3,7,5}                                            (gjk.h:1510-1512)
//   octa  (count 6): 8 triangular faces (the 8 axis-pole sign combos)                  (gjk.h:1525-1526)
//   wedge (count 6): 5 faces = 2 tri caps {0,2,4} {1,5,3} + 3 quad sides {0,1,3,2}
//                    {0,4,5,1} {2,3,5,4}                                                (gjk.h:1530-1534)
// The winding is the source winding; FaceNormalWorld re-orients each normal OUTWARD. Pure integer (no float).
inline FxHullFaces BuildCanonicalFaces(const FxHull& hull) {
    FxHullFaces faces;
    auto tri = [&](uint32_t a, uint32_t b, uint32_t c) {
        const uint32_t f = faces.faceCount++;
        faces.vertIdx[f][0] = a; faces.vertIdx[f][1] = b; faces.vertIdx[f][2] = c;
        faces.vertCount[f] = 3;
    };
    auto quad = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        const uint32_t f = faces.faceCount++;
        faces.vertIdx[f][0] = a; faces.vertIdx[f][1] = b; faces.vertIdx[f][2] = c; faces.vertIdx[f][3] = d;
        faces.vertCount[f] = 4;
    };

    if (hull.count == 4) {
        // Tetra: 4 triangular faces (every triple of the 4 verts) — promotes gjk.h:1507.
        tri(0, 1, 2); tri(0, 1, 3); tri(0, 2, 3); tri(1, 2, 3);
    } else if (hull.count == 8) {
        // Box: 6 QUAD faces — the six quads gjk.h:1510-1512 triangulates (promoted back to quads).
        quad(0, 1, 3, 2); quad(4, 5, 7, 6); quad(0, 1, 5, 4);
        quad(2, 3, 7, 6); quad(0, 2, 6, 4); quad(1, 3, 7, 5);
    } else if (hull.count == 6) {
        if (IsOctaHull(hull)) {
            // Octa: 8 triangular faces (the 8 axis-pole sign combos) — promotes gjk.h:1525-1526.
            tri(0, 2, 4); tri(0, 4, 3); tri(0, 3, 5); tri(0, 5, 2);
            tri(1, 4, 2); tri(1, 3, 4); tri(1, 5, 3); tri(1, 2, 5);
        } else {
            // Wedge: 2 triangular caps + 3 quad sides = 5 faces — promotes gjk.h:1530-1534 (the 3 quad sides
            // gjk::HullTriIndices triangulates into 6 tris are merged back into the 3 polygon quads).
            tri(0, 2, 4); tri(1, 5, 3);                   // the two triangular caps
            quad(0, 1, 3, 2); quad(0, 4, 5, 1); quad(2, 3, 5, 4);  // the three quad sides
        }
    }
    // Any other count -> faceCount 0 (the canonical hulls only; the documented YAGNI, gjk.h:1537).
    return faces;
}

// ----- HullCentroidWorld(hull, body): the world-space centroid of the hull (the integer mean of the world
// verts). The origin-symmetric canonical hulls have a local centroid AT the origin, so this is essentially
// body.pos — but computed honestly over the verts so it is correct for any hull. The OUTWARD reference for the
// normal orientation. Pure integer (the FxRotate world transform + an integer average).
inline FxVec3 HullCentroidWorld(const FxHull& hull, const FxBody& body) {
    if (hull.count == 0) return body.pos;
    int64_t sx = 0, sy = 0, sz = 0;
    for (uint32_t i = 0; i < hull.count; ++i) {
        const FxVec3 w = FxAdd(fpx::FxRotate(body.orient, hull.verts[i]), body.pos);
        sx += (int64_t)w.x; sy += (int64_t)w.y; sz += (int64_t)w.z;
    }
    return FxVec3{(fx)(sx / (int64_t)hull.count), (fx)(sy / (int64_t)hull.count),
                  (fx)(sz / (int64_t)hull.count)};
}

// ----- FaceCentroidWorld(hull, faces, body, f): the world-space centroid of face f (the integer mean of its
// world vertices). Pure integer. f must be < faces.faceCount.
inline FxVec3 FaceCentroidWorld(const FxHull& hull, const FxHullFaces& faces, const FxBody& body, uint32_t f) {
    const uint32_t vc = faces.vertCount[f];
    if (vc == 0) return body.pos;
    int64_t sx = 0, sy = 0, sz = 0;
    for (uint32_t k = 0; k < vc; ++k) {
        const uint32_t vi = faces.vertIdx[f][k];
        const FxVec3 w = FxAdd(fpx::FxRotate(body.orient, hull.verts[vi]), body.pos);
        sx += (int64_t)w.x; sy += (int64_t)w.y; sz += (int64_t)w.z;
    }
    return FxVec3{(fx)(sx / (int64_t)vc), (fx)(sy / (int64_t)vc), (fx)(sz / (int64_t)vc)};
}

// ----- FaceNormalWorld(hull, faces, body, f): the OUTWARD world-space normal of face f. The FxCross of two of
// the face's edges (edge0 = v1-v0, edge1 = v2-v0), oriented OUTWARD by flipping it to DISAGREE with
// (faceCentroid - hullCentroid) — the integer idiom gjk::EpaAddFace uses (gjk.h:727) and HullToRenderMesh uses
// render-side (gjk.h:1624-1629). So the source winding in the face table does NOT matter — the normal is
// canonicalized outward exactly as the render soup already does. NOT normalized (Q16.16 magnitude is the
// triangle-area scale) — SupportFace/IncidentFace rank by a NORMALIZED direction's dot anyway and only the SIGN
// of the outward test matters; the un-normalized vector keeps the math pure integer (no FxISqrt needed). Pure
// integer (int64 FxCross/FxDot). f must be < faces.faceCount.
inline FxVec3 FaceNormalWorld(const FxHull& hull, const FxHullFaces& faces, const FxBody& body, uint32_t f) {
    const uint32_t vc = faces.vertCount[f];
    if (vc < 3) return FxVec3{0, 0, 0};
    const FxVec3 v0 = FxAdd(fpx::FxRotate(body.orient, hull.verts[faces.vertIdx[f][0]]), body.pos);
    const FxVec3 v1 = FxAdd(fpx::FxRotate(body.orient, hull.verts[faces.vertIdx[f][1]]), body.pos);
    const FxVec3 v2 = FxAdd(fpx::FxRotate(body.orient, hull.verts[faces.vertIdx[f][2]]), body.pos);
    FxVec3 n = FxCross(FxSub(v1, v0), FxSub(v2, v0));
    // Orient OUTWARD against the hull centroid (flip to point AWAY from the interior) — gjk::EpaAddFace idiom.
    const FxVec3 hc = HullCentroidWorld(hull, body);
    const FxVec3 fc = FaceCentroidWorld(hull, faces, body, f);
    if (FxDot(n, FxSub(fc, hc)) < 0) n = FxVec3{-n.x, -n.y, -n.z};
    return n;
}

// ----- SupportFace(hull, faces, body, dir): the REFERENCE-face selector — the face whose OUTWARD world normal
// is MOST PARALLEL to `dir` (the max FxDot(faceNormalWorld, dir)). Scanned in FIXED face-index order with a
// STRICT-GREATER update so ties keep the LOWEST index (the gjk::SupportLocal rule, gjk.h:85). Deterministic,
// pure integer (the int64 FxDot). Returns kMaxHullFaces (an out-of-range sentinel) for an empty face table.
inline uint32_t SupportFace(const FxHull& hull, const FxHullFaces& faces, const FxBody& body,
                            const FxVec3& dir) {
    if (faces.faceCount == 0) return kMaxHullFaces;
    uint32_t best = 0;
    fx bestDot = FxDot(FaceNormalWorld(hull, faces, body, 0), dir);
    for (uint32_t f = 1; f < faces.faceCount; ++f) {
        const fx d = FxDot(FaceNormalWorld(hull, faces, body, f), dir);
        if (d > bestDot) {          // STRICT-greater -> ties keep the LOWEST index (gjk::SupportLocal rule)
            bestDot = d;
            best = f;
        }
    }
    return best;
}

// ----- IncidentFace(hull, faces, body, refNormal): the INCIDENT-face selector — the face whose OUTWARD world
// normal is MOST ANTI-PARALLEL to `refNormal` (the reference face's normal). The most anti-parallel face is the
// MOST PARALLEL to -refNormal -> SupportFace against the negated reference normal (same FIXED order + strict-
// greater lowest-index tie-break). Deterministic, pure integer. This is the standard SAT clip incident-face
// selection MF2 consumes.
inline uint32_t IncidentFace(const FxHull& hull, const FxHullFaces& faces, const FxBody& body,
                             const FxVec3& refNormal) {
    return SupportFace(hull, faces, body, FxVec3{-refNormal.x, -refNormal.y, -refNormal.z});
}

// =========================================================================================================
// Render-only float helper (the GJ6/BP6/CD6 render-bridge shape). OUTSIDE any bit-exact integer loop — the ONE
// float crossing. Tints each FACE a distinct flat color from a fixed palette so the face decomposition is
// VISIBLE (box = 6 quads, octa = 8 tris, etc.) — NOT per-hull-type like gjk::HullTypeColor.
// =========================================================================================================

// ----- kFacePalette: a fixed warm/matte palette indexed by face number (mod its size). KEEP GREEN LOW so the
// lit.frag up-normal blue-sky ambient does not push the read toward green/iridescence (the GF6/FR6/CX6 hard
// lesson, the gjk::HullTypeColor discipline). 8 entries cover the octa's 8 faces (the max); the floor (if any)
// stays cool grey. Render-only float.
inline void FaceColor(uint32_t faceIndex, float out[3]) {
    static const float kFacePalette[8][3] = {
        {0.90f, 0.32f, 0.12f},   // rust orange
        {0.82f, 0.52f, 0.12f},   // amber gold
        {0.78f, 0.24f, 0.18f},   // brick red
        {0.86f, 0.62f, 0.16f},   // ochre
        {0.70f, 0.40f, 0.10f},   // tan
        {0.92f, 0.46f, 0.20f},   // warm coral
        {0.74f, 0.30f, 0.22f},   // terracotta
        {0.80f, 0.56f, 0.24f},   // sand
    };
    const float* c = kFacePalette[faceIndex % 8u];
    out[0] = c[0]; out[1] = c[1]; out[2] = c[2];
}

// ----- FacesToRenderInstances(world): the render payload for the showcase — a FLOAT world-space TRIANGLE SOUP
// (the gjk::HullRenderMesh shape) where each FACE is flat-tinted a distinct color from kFacePalette (the face
// decomposition made VISIBLE). Mirrors gjk::HullToRenderMesh's float transform + outward-normal logic, but
// triangulates each POLYGON face as a triangle FAN from its first vertex and colors by FACE INDEX (not hull
// type). A PURE FUNCTION of `world` (no clock/RNG) — two calls produce byte-equal verts (the provenance
// contract the showcase/test assert via gjk::HullRenderMeshEqual). The integer sim is untouched; the LOCAL
// verts are FxToFloat'd render-only. Non-canonical hulls (no face table) are skipped.
inline gjk::HullRenderMesh FacesToRenderInstances(const HullWorld& world) {
    gjk::HullRenderMesh out;
    const size_t n = world.bodies.size() < world.hulls.size() ? world.bodies.size() : world.hulls.size();

    // Rotate a LOCAL float vector by the body's (normalized) unit quaternion (== gjk::HullToRenderMesh's qrot).
    auto qrot = [](float qx, float qy, float qz, float qw, const float v[3], float out3[3]) {
        const float tx = 2.0f * (qy * v[2] - qz * v[1]);
        const float ty = 2.0f * (qz * v[0] - qx * v[2]);
        const float tz = 2.0f * (qx * v[1] - qy * v[0]);
        out3[0] = v[0] + qw * tx + (qy * tz - qz * ty);
        out3[1] = v[1] + qw * ty + (qz * tx - qx * tz);
        out3[2] = v[2] + qw * tz + (qx * ty - qy * tx);
    };

    for (size_t i = 0; i < n; ++i) {
        const FxBody& b = world.bodies[i];
        const FxHull& hull = world.hulls[i];
        const FxHullFaces faces = BuildCanonicalFaces(hull);
        if (faces.faceCount == 0) continue;   // non-canonical hull -> no faces

        // The body's world transform pieces (render-only float; == gjk::HullToRenderMesh).
        const float px = fpx::FxToFloat(b.pos.x), py = fpx::FxToFloat(b.pos.y), pz = fpx::FxToFloat(b.pos.z);
        float qx = fpx::FxToFloat(b.orient.x), qy = fpx::FxToFloat(b.orient.y),
              qz = fpx::FxToFloat(b.orient.z), qw = fpx::FxToFloat(b.orient.w);
        const float qlen2 = qx * qx + qy * qy + qz * qz + qw * qw;
        if (qlen2 > 0.0f) {
            // Newton-free normalize via the same std::sqrt gjk::HullToRenderMesh uses (render-only float).
            const float ql = std::sqrt(qlen2);
            qx /= ql; qy /= ql; qz /= ql; qw /= ql;
        }
        const float cw[3] = {px, py, pz};   // world centroid (origin-symmetric canonical hulls -> body pos)

        // Transform each local vert to world space once (cache).
        float worldV[gjk::kMaxHullVerts][3];
        for (uint32_t v = 0; v < hull.count; ++v) {
            const float lv[3] = {fpx::FxToFloat(hull.verts[v].x), fpx::FxToFloat(hull.verts[v].y),
                                 fpx::FxToFloat(hull.verts[v].z)};
            float rv[3]; qrot(qx, qy, qz, qw, lv, rv);
            worldV[v][0] = rv[0] + px; worldV[v][1] = rv[1] + py; worldV[v][2] = rv[2] + pz;
        }

        // Emit each FACE as an outward-wound triangle FAN, flat-colored by face index.
        for (uint32_t f = 0; f < faces.faceCount; ++f) {
            const uint32_t vc = faces.vertCount[f];
            if (vc < 3) continue;
            float fcolor[3]; FaceColor(f, fcolor);

            // Flat outward face normal = (v1-v0) x (v2-v0), normalized + flipped OUTWARD against the hull
            // centroid (== gjk::HullToRenderMesh's per-triangle logic, but ONE normal for the whole polygon).
            const float* A = worldV[faces.vertIdx[f][0]];
            const float* B0 = worldV[faces.vertIdx[f][1]];
            const float* C0 = worldV[faces.vertIdx[f][2]];
            float e1[3] = {B0[0] - A[0], B0[1] - A[1], B0[2] - A[2]};
            float e2[3] = {C0[0] - A[0], C0[1] - A[1], C0[2] - A[2]};
            float nrm[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                            e1[2] * e2[0] - e1[0] * e2[2],
                            e1[0] * e2[1] - e1[1] * e2[0]};
            const float nl = std::sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
            if (nl > 0.0f) { nrm[0] /= nl; nrm[1] /= nl; nrm[2] /= nl; }
            // Polygon centroid for the outward test.
            float fcx = 0.0f, fcy = 0.0f, fcz = 0.0f;
            for (uint32_t k = 0; k < vc; ++k) {
                const float* P = worldV[faces.vertIdx[f][k]];
                fcx += P[0]; fcy += P[1]; fcz += P[2];
            }
            fcx /= (float)vc; fcy /= (float)vc; fcz /= (float)vc;
            const float outDot = nrm[0] * (fcx - cw[0]) + nrm[1] * (fcy - cw[1]) + nrm[2] * (fcz - cw[2]);
            const bool flip = outDot < 0.0f;
            if (flip) { nrm[0] = -nrm[0]; nrm[1] = -nrm[1]; nrm[2] = -nrm[2]; }

            // Triangle fan v0,vk,vk+1 (k=1..vc-2). If we flipped the normal, reverse the winding to match.
            for (uint32_t k = 1; k + 1 < vc; ++k) {
                uint32_t ia = faces.vertIdx[f][0];
                uint32_t ib = faces.vertIdx[f][k];
                uint32_t ic = faces.vertIdx[f][k + 1];
                if (flip) { const uint32_t t = ib; ib = ic; ic = t; }
                const uint32_t idx3[3] = {ia, ib, ic};
                for (int j = 0; j < 3; ++j) {
                    const float* P = worldV[idx3[j]];
                    gjk::HullRenderVertex hv;
                    hv.pos[0] = P[0]; hv.pos[1] = P[1]; hv.pos[2] = P[2];
                    hv.normal[0] = nrm[0]; hv.normal[1] = nrm[1]; hv.normal[2] = nrm[2];
                    hv.color[0] = fcolor[0]; hv.color[1] = fcolor[1]; hv.color[2] = fcolor[2];
                    out.verts.push_back(hv);
                }
            }
        }
    }
    out.triangles = (uint32_t)(out.verts.size() / 3);
    return out;
}

// =========================================================================================================
// Slice MF2 — Hull Narrowphase Hardening: SUTHERLAND-HODGMAN FACE CLIP -> THE MULTI-POINT MANIFOLD.
// APPENDED after FacesToRenderInstances (MF1's lines above are BYTE-FROZEN; gjk/broad/ccd/convex/fpx +
// every sibling sim header BYTE-FROZEN). MF1 built the per-hull POLYGON FACE topology; MF2 USES it to
// GENERATE the multi-point contact manifold that fixes gjk::HullContact's documented single-point teeter:
// given two overlapping hulls + the EPA contact normal, pick the REFERENCE face (most parallel to the
// normal) and the INCIDENT face (most anti-parallel on the other hull), CLIP the incident polygon against
// the reference face's side planes by deterministic Sutherland-Hodgman, and keep the clipped points below
// the reference plane as the 1-4 contact points. This is the EXACT idiom convex::BuildManifold
// (convex.h:352-399) runs for BOXES, GENERALIZED from box quads (2 axes -> 4 fixed side planes) to
// arbitrary convex polygon faces (MF1's FxHullFaces -> one inward side-plane per face EDGE). The output is a
// convex::ContactManifold VERBATIM (the frozen solver convex::SolveManifoldImpulse already loops 0..count,
// convex.h:662 -> a count-4 manifold needs ZERO solver change). PURE INTEGER (int64 FxDot/FxCross/fxdiv/
// fxmul); NO float, NO <cmath> in the clip math. THE CRUX (the convex::BuildManifold discipline): the SH
// clip's keep/drop + intersection tie-breaks are bit-deterministic (FIXED edge order, STRICT integer sign,
// NO tolerance band) so two runs are byte-equal — a 1-LSB flip changes count + cascades.
//
// HONEST SIMPLIFICATION (inherited from convex.h:280-284, documented + in scope): the 4-point reduction
// keeps the DEEPEST point + the first 3 in CLIP ORDER (NOT the area-maximizing 4 a production solver keeps);
// the face-clip contact point is the CLIPPED INCIDENT VERTEX position itself (not projected onto the
// reference plane); its depth is the signed distance below the reference face. Canonical hulls only.

using fpx::fxmul;   // the Q16.16 multiply (int64 intermediate) — the side-plane intersection lerp
using fpx::fxdiv;   // the Q16.16 divide  (int64 intermediate) — the crossing-edge t parameter

// kMaxClipVerts: a fixed-size scratch polygon ceiling. The incident face is <= kMaxFaceVerts (4); clipping a
// 4-gon by <=4 side planes stays <= 8 (the convex.h:460 poly[8] bound). 8 covers every canonical face.
constexpr uint32_t kMaxClipVerts = 8;

// ----- ClipFaceAgainstFace: the deterministic Sutherland-Hodgman clip of the INCIDENT face polygon against
// the REFERENCE face's side planes, GENERALIZING convex::BuildManifold's box quad-clip to arbitrary polygon
// faces. Inputs are world-space face tables (MF1). The reference face has one inward SIDE PLANE per EDGE
// (v[k] -> v[k+1]): the side-plane normal sideN = FxCross(refFaceNormal, edgeDir) is oriented INWARD toward
// the face interior (flipped to AGREE with refFaceCentroid - v[k]); a point p is INSIDE iff
// FxDot(sideN, p - v[k]) >= 0 (strict-integer sign; on-plane counts as inside; NO tolerance band). Edges are
// clipped in FIXED order k = 0..refVertCount-1; a crossing edge (prev,cur) emits the fxdiv intersection in a
// PINNED iteration order (== convex.h:480-513). Returns the surviving clipped polygon (<= kMaxClipVerts) in
// outPts; outN is its vertex count (0 if the reference face is degenerate or the clip empties — the caller's
// fallback floor). Pure integer (int64 FxDot/FxCross/fxdiv/fxmul), every order pinned.
inline void ClipFaceAgainstFace(const FxHull& refHull, const FxBody& refBody, const FxHullFaces& refFaces,
                                uint32_t refFace,
                                const FxHull& incHull, const FxBody& incBody, const FxHullFaces& incFaces,
                                uint32_t incFace,
                                FxVec3 outPts[kMaxClipVerts], int& outN) {
    outN = 0;
    if (refFace >= refFaces.faceCount || incFace >= incFaces.faceCount) return;
    const uint32_t refVc = refFaces.vertCount[refFace];
    const uint32_t incVc = incFaces.vertCount[incFace];
    if (refVc < 3 || incVc < 3) return;   // degenerate reference/incident face -> empty (the fallback floor)

    // The reference face's world-space vertices (FIXED order) + its outward normal + its centroid.
    FxVec3 refV[kMaxFaceVerts];
    for (uint32_t k = 0; k < refVc; ++k) {
        const uint32_t vi = refFaces.vertIdx[refFace][k];
        refV[k] = FxAdd(fpx::FxRotate(refBody.orient, refHull.verts[vi]), refBody.pos);
    }
    const FxVec3 refN  = FaceNormalWorld(refHull, refFaces, refBody, refFace);
    const FxVec3 refC  = FaceCentroidWorld(refHull, refFaces, refBody, refFace);

    // The incident polygon (the thing we clip), world-space, FIXED order.
    FxVec3 poly[kMaxClipVerts];
    int polyN = (int)incVc;
    for (uint32_t k = 0; k < incVc; ++k) {
        const uint32_t vi = incFaces.vertIdx[incFace][k];
        poly[k] = FxAdd(fpx::FxRotate(incBody.orient, incHull.verts[vi]), incBody.pos);
    }

    // Sutherland-Hodgman against each ref-face EDGE's inward side plane, in FIXED edge order k=0..refVc-1.
    // For edge (a = refV[k], b = refV[(k+1)%refVc]): edgeDir = b - a; sideN = FxCross(refN, edgeDir) oriented
    // INWARD (flipped to AGREE with refC - a). INSIDE test f(p) = FxDot(sideN, p - a) >= 0 (>= so on-plane
    // counts as inside; NO tolerance band). A crossing edge emits the intersection at t = f(prev)/(f(prev)-
    // f(cur)) (int64 fxdiv) -> prev + t*(cur-prev). Iteration PINNED (edge 0..polyN-1, prev=last).
    for (uint32_t e = 0; e < refVc; ++e) {
        const FxVec3 a    = refV[e];
        const FxVec3 b    = refV[(e + 1u) % refVc];
        const FxVec3 edge = FxSub(b, a);
        FxVec3 sideN = FxCross(refN, edge);
        // Orient INWARD: the face interior (refC) must be on the inside (f(refC) >= 0).
        if (FxDot(sideN, FxSub(refC, a)) < 0) sideN = FxVec3{-sideN.x, -sideN.y, -sideN.z};

        auto fside = [&](const FxVec3& p) -> fx { return FxDot(sideN, FxSub(p, a)); };

        FxVec3 out[kMaxClipVerts];
        int outNloc = 0;
        if (polyN == 0) break;
        FxVec3 prev = poly[polyN - 1];
        fx fprev = fside(prev);
        for (int k = 0; k < polyN; ++k) {
            const FxVec3 cur = poly[k];
            const fx fcur = fside(cur);
            const bool curIn  = (fcur >= 0);
            const bool prevIn = (fprev >= 0);
            if (curIn) {
                if (!prevIn && outNloc < (int)kMaxClipVerts) {
                    // entering: emit the crossing point first, then cur.
                    const fx denom = fprev - fcur;
                    const fx tp = (denom != 0) ? fxdiv(fprev, denom) : 0;
                    out[outNloc++] = FxVec3{prev.x + fxmul(cur.x - prev.x, tp),
                                            prev.y + fxmul(cur.y - prev.y, tp),
                                            prev.z + fxmul(cur.z - prev.z, tp)};
                }
                if (outNloc < (int)kMaxClipVerts) out[outNloc++] = cur;
            } else if (prevIn && outNloc < (int)kMaxClipVerts) {
                // leaving: emit the crossing point only.
                const fx denom = fprev - fcur;
                const fx tp = (denom != 0) ? fxdiv(fprev, denom) : 0;
                out[outNloc++] = FxVec3{prev.x + fxmul(cur.x - prev.x, tp),
                                        prev.y + fxmul(cur.y - prev.y, tp),
                                        prev.z + fxmul(cur.z - prev.z, tp)};
            }
            prev = cur; fprev = fcur;
        }
        polyN = outNloc;
        for (int k = 0; k < polyN; ++k) poly[k] = out[k];
    }

    outN = polyN;
    for (int k = 0; k < polyN; ++k) outPts[k] = poly[k];
}

// ----- HullManifoldFromEpa: the multi-point contact manifold from an EPA result. The caller has already run
// gjk::Gjk -> gjk::Epa; MF2 takes the gjk::EpaResult and does the reference/incident face selection + the
// Sutherland-Hodgman clip, returning a convex::ContactManifold (count 1-4) VERBATIM (no new manifold type).
//
//   n = the EPA unit contact normal, SIGNED from A->B (epa.normal). The "reference" hull is the one whose
//   SupportFace(n) is MOST PARALLEL to n (the better-aligned face owner): compare SupportFace(A,n)'s normal
//   dot n against SupportFace(B,-n)'s normal dot -n (B's reference face points along -n toward A); tie-break
//   A. The reference face = SupportFace(refHull, refDir); the incident face = IncidentFace(incHull, refN)
//   (MF1). Clip the incident polygon against the reference face's side planes (ClipFaceAgainstFace). Keep
//   clipped vertices with depth d = FxDot(n, refFaceCenter - vertex) >= 0 (below/inside the ref face);
//   contact point = the vertex itself (the documented convex.h:282-284 simplification). Reduce to <=4
//   keeping the DEEPEST (tie -> lowest clip-order index) then up to 3 more in clip order. normal = n.
//   FALLBACK: if a selected face is degenerate or the clip yields 0 kept points, emit the single
//   closest-point / EPA-witness midpoint as count==1 (NEVER count 0 for an overlapping pair).
inline convex::ContactManifold HullManifoldFromEpa(const FxHull& hullA, const FxBody& bodyA,
                                                   const FxHull& hullB, const FxBody& bodyB,
                                                   const gjk::EpaResult& epa) {
    convex::ContactManifold m;   // count 0, zeroed by the struct defaults
    const FxVec3 n = epa.normal;   // UNIT, A->B

    // The EPA-witness midpoint — the deterministic single-point floor (the gjk::HullContact behavior).
    const FxVec3 witnessMid = FxVec3{(epa.contactA.x + epa.contactB.x) / 2,
                                     (epa.contactA.y + epa.contactB.y) / 2,
                                     (epa.contactA.z + epa.contactB.z) / 2};
    auto fallbackOnePoint = [&]() -> convex::ContactManifold {
        convex::ContactManifold f;
        f.count = 1u;
        f.points[0] = witnessMid;
        f.depths[0] = epa.depth;
        f.normal = n;
        return f;
    };

    const FxHullFaces facesA = BuildCanonicalFaces(hullA);
    const FxHullFaces facesB = BuildCanonicalFaces(hullB);
    if (facesA.faceCount == 0 || facesB.faceCount == 0) return fallbackOnePoint();

    // Choose the reference hull: the one whose SupportFace(toward the other) is MOST PARALLEL to its search
    // direction. A's reference face is SupportFace(A, n) (points along +n toward B); B's reference face is
    // SupportFace(B, -n) (points along -n toward A). Compare the two alignments; tie -> A (the lowest-index
    // hull, the gjk::SupportLocal tie discipline).
    const uint32_t sfA = SupportFace(hullA, facesA, bodyA, n);
    const FxVec3   nfA = FaceNormalWorld(hullA, facesA, bodyA, sfA);
    const fx       alignA = FxDot(nfA, n);
    const FxVec3   negN = FxVec3{-n.x, -n.y, -n.z};
    const uint32_t sfB = SupportFace(hullB, facesB, bodyB, negN);
    const FxVec3   nfB = FaceNormalWorld(hullB, facesB, bodyB, sfB);
    const fx       alignB = FxDot(nfB, negN);

    const bool refIsA = (alignA >= alignB);   // tie-break A (>=)

    const FxHull&      refHull  = refIsA ? hullA : hullB;
    const FxBody&      refBody  = refIsA ? bodyA : bodyB;
    const FxHullFaces& refFaces = refIsA ? facesA : facesB;
    const uint32_t     refFace  = refIsA ? sfA : sfB;
    const FxVec3       refN     = refIsA ? nfA : nfB;   // the reference face's OUTWARD world normal

    const FxHull&      incHull  = refIsA ? hullB : hullA;
    const FxBody&      incBody  = refIsA ? bodyB : bodyA;
    const FxHullFaces& incFaces = refIsA ? facesB : facesA;
    const uint32_t     incFace  = IncidentFace(incHull, incFaces, incBody, refN);

    if (refFace >= refFaces.faceCount || incFace >= incFaces.faceCount) return fallbackOnePoint();

    // The reference face center (the clip-keep reference plane passes through it, outward normal refN).
    const FxVec3 refC = FaceCentroidWorld(refHull, refFaces, refBody, refFace);

    // Sutherland-Hodgman clip the incident polygon against the reference face's side planes.
    FxVec3 clipped[kMaxClipVerts];
    int clipN = 0;
    ClipFaceAgainstFace(refHull, refBody, refFaces, refFace,
                        incHull, incBody, incFaces, incFace, clipped, clipN);
    if (clipN == 0) return fallbackOnePoint();   // empty clip -> the single-point floor

    // Keep the penetrating clipped vertices (depth d = FxDot(refN, refC - vertex) >= 0 == below/inside the
    // reference face along its OUTWARD normal). The contact point = the vertex itself (documented), depth=d.
    // refN is the reference face's outward normal; the EPA n agrees with it up to sign — d uses refN so the
    // depth is the signed distance below the reference face regardless of which hull is reference.
    FxVec3 candPts[kMaxClipVerts];
    fx     candDepth[kMaxClipVerts];
    int    candN = 0;
    for (int k = 0; k < clipN; ++k) {
        const FxVec3 rel = FxVec3{refC.x - clipped[k].x, refC.y - clipped[k].y, refC.z - clipped[k].z};
        const fx d = FxDot(refN, rel);
        if (d >= 0) { candPts[candN] = clipped[k]; candDepth[candN] = d; ++candN; }
    }
    if (candN == 0) return fallbackOnePoint();   // no clipped vertex penetrates -> the single-point floor

    // Reduce to <=4: ALWAYS keep the DEEPEST (max depth, tie -> lowest clip-order index), then up to 3 MORE
    // in clip order. Deterministic (== convex.h:539-552).
    int deepest = 0;
    for (int k = 1; k < candN; ++k) if (candDepth[k] > candDepth[deepest]) deepest = k;
    m.points[0] = candPts[deepest];
    m.depths[0] = candDepth[deepest];
    uint32_t cnt = 1;
    for (int k = 0; k < candN && cnt < 4; ++k) {
        if (k == deepest) continue;
        m.points[cnt] = candPts[k];
        m.depths[cnt] = candDepth[k];
        ++cnt;
    }
    m.count = cnt;
    m.normal = n;   // the EPA normal, SIGNED A->B (the SolveManifoldImpulse / de-pen convention)
    return m;
}

// ----- HullManifold(bodyA, hullA, bodyB, hullB): the convenience wrapper that runs the FROZEN gjk::Gjk ->
// gjk::Epa narrowphase then HullManifoldFromEpa. Separated -> {count=0}. The MF2 multi-point counterpart of
// gjk::HullContact (which hardcodes the single point). Pure integer.
inline convex::ContactManifold HullManifold(const FxBody& bodyA, const FxHull& hullA,
                                            const FxBody& bodyB, const FxHull& hullB) {
    convex::ContactManifold m;   // count 0
    const gjk::GjkResult g = gjk::Gjk(hullA, bodyA, hullB, bodyB);
    if (!g.overlap) return m;
    const gjk::EpaResult e = gjk::Epa(hullA, bodyA, hullB, bodyB, g.simplex);
    return HullManifoldFromEpa(hullA, bodyA, hullB, bodyB, e);
}

// =========================================================================================================
// Slice MF3 — Hull Narrowphase Hardening: THE MULTI-POINT MANIFOLD GPU SHADER (the int64 GPU==CPU beat).
// APPENDED after HullManifold (MF1/MF2's lines above are BYTE-FROZEN; gjk/broad/ccd/convex/fpx + every sibling
// sim header BYTE-FROZEN). MF2 built the CPU multi-point manifold (HullManifoldFromEpa, the Sutherland-Hodgman
// face clip); MF3 lifts it ONTO THE GPU. shaders/hull_manifold.comp.hlsl is the GPU generator — one thread per
// overlapping pair runs the SAME gjk::Gjk -> gjk::Epa -> HullManifoldFromEpa call chain and writes the SAME
// 1-4 point convex::ContactManifold, and the proof is the GPU manifold array is BYTE-IDENTICAL to the CPU
// HullContactMulti over a fixed pair battery. HullContactMulti is the single named function the shader mirrors
// VERBATIM (the SAME int64 FxDot/FxCross/fxdiv ops, the SAME fixed orders, the SAME strict-integer tie-breaks).
// =========================================================================================================

// ----- HullContactMulti(bodyA, hullA, bodyB, hullB): the HARDENED multi-point drop-in for gjk::HullContact
// (gjk.h:1155, which hardcodes count 1). Runs the FROZEN gjk::Gjk -> gjk::Epa narrowphase, then MF2's
// HullManifoldFromEpa, returning a convex::ContactManifold (count 1-4; separated -> count 0). It is the same
// call chain as HullManifold (it simply IS that function — the MF3 point is that the SHADER copies THIS body
// VERBATIM, so HullContactMulti is the NAMED hardened entry the GPU mirror is checked against). Pure integer,
// deterministic, identical CPU/GPU.
inline convex::ContactManifold HullContactMulti(const FxBody& bodyA, const FxHull& hullA,
                                                const FxBody& bodyB, const FxHull& hullB) {
    convex::ContactManifold m;   // count 0, zeroed by the struct defaults
    const gjk::GjkResult g = gjk::Gjk(hullA, bodyA, hullB, bodyB);
    if (!g.overlap) return m;     // separated -> empty manifold (count 0)
    const gjk::EpaResult e = gjk::Epa(hullA, bodyA, hullB, bodyB, g.simplex);
    return HullManifoldFromEpa(hullA, bodyA, hullB, bodyB, e);
}

// =========================================================================================================
// Slice MF4 — Hull Narrowphase Hardening: FULL INERTIA + THE RESTACKED-STABILITY STEP (the new-physics money
// beat, the HEADLINE of FLAGSHIP #25). APPENDED after HullContactMulti (MF1/MF2/MF3's lines above are
// BYTE-FROZEN; gjk/broad/ccd/convex/fpx + every sibling sim header BYTE-FROZEN). MF1 built the face topology;
// MF2 the CPU multi-point manifold; MF3 lifted it onto the GPU. MF4 is the PAYOFF: it wires the hardened
// manifold + a FULL convex inertia tensor into a stepped world, so a hull dropped FLAT on another SETTLES TO
// REST where the frozen single-point step (gjk::StepHullWorld) leaves it TEETERING. StepHullWorldHardened is
// the gjk::StepHullWorld 5-pass shell (gjk.h:1188) with EXACTLY TWO callee swaps and NOTHING else:
//   (step 2 inertia) FxHullInvInertiaBody+WorldInvInertia -> FxHullInertiaBodyFull+WorldInvInertiaFull;
//   (steps 3+4 contact) gjk::HullContact -> HullContactMulti (MF3).
// The frozen convex::SolveManifoldImpulse (convex.h:651) ALREADY takes convex::FxMat3 invIaW/invIbW + already
// loops 0..count over the manifold points, so it consumes the full tensor + the multi-point manifold with ZERO
// change. The frozen gjk::StepHullWorld + its 224 goldens are UNTOUCHED (additivity by construction).
//
// THE FULL INERTIA (the new physics): FxHullInertiaBodyFull computes the canonical hull's full symmetric
// inertia tensor I by SIGNED-TETRAHEDRON decomposition (a tetra fan from the hull centroid — at the origin for
// the origin-symmetric canonical hulls — over each BuildCanonicalFaces face triangle, accumulating the
// standard covariance integrals), in BODY-LOCAL space (the verts are bounded by the hull's local extents),
// int64-accumulated, deferring the Q16.16 normalization; scaled by mass = 1/invMass; then FxMat3SymInverse.
// THE CRUX (overflow + the cube cross-check): the tetra covariance integrals accumulate triple products of
// vertex coordinates. We keep them BODY-LOCAL (bounded), accumulate in explicit int64, defer the scaling, and
// the cube cross-check asserts FxMat3SymInverse(FxHullInertiaBodyFull(MakeBox)) == gjk::FxHullInvInertiaBody's
// diagonal (the full tensor REDUCES to the AABB answer for a cube). WorldInvInertiaFull rotates the inverse
// body tensor to world (R·M·Rᵀ, R = convex::BoxAxes) — generalizing convex::WorldInvInertia's outer-product.
//
// THE int64 GPU MIRROR (the GJ4/CX4 lesson): the whole step is int64 (the GJK/EPA + the inertia products +
// the SH clip). DXC compiles int64 (Vulkan); glslc cannot. So shaders/hull_step_hardened.comp.hlsl is
// VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl); the Metal --mf4-stack runs THIS CPU StepHullWorldHardenedN (byte-
// identical to the Vulkan GPU result BY CONSTRUCTION), chunked 1 tick/dispatch (TDR-safe — heavier than the
// single-point step). The Vulkan --mf4-stack-shot carries the GPU==CPU memcmp.
// =========================================================================================================

using convex::FxMat3;
using fpx::fxmul;
using fpx::fxdiv;

// ----- FxMat3SymInverse(M): the deterministic integer inverse of a SYMMETRIC 3x3 (adjugate/determinant) -----
// M is the 9-element FxMat3 (Q16.16); for a symmetric tensor only the upper triangle matters but we read all 9.
// The cofactors + the determinant are computed with fxmul (int64 intermediate) in a FIXED order, then each
// adjugate entry is fxdiv'd by the determinant (int64, fixed order). A DEGENERATE/zero determinant -> a
// deterministic ZERO matrix (the caller falls back to the diagonal WorldInvInertia floor — see
// FxHullInertiaBodyFull). Pure integer, deterministic, identical CPU/GPU.
inline FxMat3 FxMat3SymInverse(const FxMat3& M) {
    // Name the entries (row-major). M is symmetric so m01==m10 etc., but we read them generally.
    const fx a = M.m[0], b = M.m[1], c = M.m[2];
    const fx d = M.m[3], e = M.m[4], f = M.m[5];
    const fx g = M.m[6], h = M.m[7], i = M.m[8];
    // Cofactors (Q16.16; each term is fxmul of two Q16.16 -> Q16.16, the difference is Q16.16).
    const fx A =  (fxmul(e, i) - fxmul(f, h));   // cofactor 00
    const fx B = -(fxmul(d, i) - fxmul(f, g));   // cofactor 01
    const fx C =  (fxmul(d, h) - fxmul(e, g));   // cofactor 02
    const fx D = -(fxmul(b, i) - fxmul(c, h));   // cofactor 10
    const fx E =  (fxmul(a, i) - fxmul(c, g));   // cofactor 11
    const fx F = -(fxmul(a, h) - fxmul(b, g));   // cofactor 12
    const fx G =  (fxmul(b, f) - fxmul(c, e));   // cofactor 20
    const fx H = -(fxmul(a, f) - fxmul(c, d));   // cofactor 21
    const fx I =  (fxmul(a, e) - fxmul(b, d));   // cofactor 22
    // det = a*A + b*B + c*C (the first-row expansion; A,B,C are the row-0 cofactors).
    const fx det = fxmul(a, A) + fxmul(b, B) + fxmul(c, C);
    FxMat3 inv;   // zero-initialized -> the degenerate fallback by default
    if (det == 0) return inv;
    // inverse = adjugate / det = transpose(cofactor) / det. (For a symmetric M the inverse is symmetric.)
    inv.m[0] = fxdiv(A, det); inv.m[1] = fxdiv(D, det); inv.m[2] = fxdiv(G, det);
    inv.m[3] = fxdiv(B, det); inv.m[4] = fxdiv(E, det); inv.m[5] = fxdiv(H, det);
    inv.m[6] = fxdiv(C, det); inv.m[7] = fxdiv(F, det); inv.m[8] = fxdiv(I, det);
    return inv;
}

// ----- FxHullInertiaBodyFull(hull, faces, invMass): the FULL symmetric body-space INVERSE inertia FxMat3 -----
// Computes the canonical hull's full inertia tensor I by signed-tetrahedron decomposition: a tetra fan from the
// hull's LOCAL centroid (the origin for the origin-symmetric canonical hulls) over each BuildCanonicalFaces
// face, triangulated as a FAN. For each tetra (origin O, a, b, c) the standard covariance integrals accumulate
// (int64, body-local — the verts are bounded by a few kOne). The diagonal of I is Ixx = ∫(y²+z²)ρ dV etc.; the
// off-diagonals are -∫xy ρ dV etc. We accumulate the UNIT-DENSITY volume V + the six unit-density second
// moments, then density = mass/V (mass = 1/invMass), I = density·(moment tensor), then FxMat3SymInverse(I).
// Static (invMass==0) -> the ZERO FxMat3 (takes no angular impulse). Degenerate (V<=0) -> the diagonal floor
// (FxMat3Diagonal of gjk::FxHullInvInertiaBody — the deterministic fallback). Pure integer, deterministic.
inline FxMat3 FxHullInertiaBodyFull(const FxHull& hull, const FxHullFaces& faces, fx invMass) {
    if (invMass == 0) return FxMat3{};   // static -> zero (no angular response)

    // The diagonal floor (the deterministic fallback for a degenerate/empty face table).
    auto diagFloor = [&]() -> FxMat3 {
        const FxVec3 d = gjk::FxHullInvInertiaBody(hull, invMass);
        return convex::FxMat3Diagonal(d);
    };
    if (faces.faceCount == 0) return diagFloor();

    // ----- Accumulate the unit-density volume + the unit-density second moments in int64 (BODY-LOCAL). -----
    // Each tetra is (O=origin, a, b, c) where a,b,c are LOCAL face-fan verts. det = a·(b×c) = 6·tetraVolume.
    // The canonical covariance for a tetra apex-at-origin: ∫ x_i x_j dV = (det/120)·S_ij with
    //   S_ij = 2(a_i a_j + b_i b_j + c_i c_j) + (a_i b_j + a_j b_i) + (a_i c_j + a_j c_i) + (b_i c_j + b_j c_i).
    // We DEFER the /120 + the density scaling to the very end. All products are Q16.16 (fxmul, int64 inside);
    // the per-tetra contribution det·S_ij is Q16.16·Q16.16 -> we keep it in a Q16.16 fxmul and int64-SUM.
    int64_t accVol = 0;                    // Σ det (Q16.16; == 6·volume)
    int64_t accXX = 0, accYY = 0, accZZ = 0;   // Σ det·S_ii (Q16.16)
    int64_t accXY = 0, accXZ = 0, accYZ = 0;   // Σ det·S_ij (Q16.16)

    auto addTetra = [&](const FxVec3& a, const FxVec3& b, const FxVec3& c) {
        // det = a·(b×c) (Q16.16; FxCross + FxDot are int64 inside).
        const fx det = FxDot(a, FxCross(b, c));
        accVol += (int64_t)det;
        // S_ij (Q16.16). Helper: the symmetric covariance sum for axis pair (ai..ci, aj..cj).
        auto Sij = [&](fx ai, fx bi, fx ci, fx aj, fx bj, fx cj) -> fx {
            return 2 * (fxmul(ai, aj) + fxmul(bi, bj) + fxmul(ci, cj))
                 + (fxmul(ai, bj) + fxmul(aj, bi))
                 + (fxmul(ai, cj) + fxmul(aj, ci))
                 + (fxmul(bi, cj) + fxmul(bj, ci));
        };
        const fx Sxx = Sij(a.x, b.x, c.x, a.x, b.x, c.x);
        const fx Syy = Sij(a.y, b.y, c.y, a.y, b.y, c.y);
        const fx Szz = Sij(a.z, b.z, c.z, a.z, b.z, c.z);
        const fx Sxy = Sij(a.x, b.x, c.x, a.y, b.y, c.y);
        const fx Sxz = Sij(a.x, b.x, c.x, a.z, b.z, c.z);
        const fx Syz = Sij(a.y, b.y, c.y, a.z, b.z, c.z);
        accXX += (int64_t)fxmul(det, Sxx);
        accYY += (int64_t)fxmul(det, Syy);
        accZZ += (int64_t)fxmul(det, Szz);
        accXY += (int64_t)fxmul(det, Sxy);
        accXZ += (int64_t)fxmul(det, Sxz);
        accYZ += (int64_t)fxmul(det, Syz);
    };

    // Fan every polygon face from its first vertex -> the tetra apex at the LOCAL origin (centroid). FIXED order.
    for (uint32_t f = 0; f < faces.faceCount; ++f) {
        const uint32_t vc = faces.vertCount[f];
        if (vc < 3) continue;
        const FxVec3 v0 = hull.verts[faces.vertIdx[f][0]];
        for (uint32_t k = 1; k + 1 < vc; ++k) {
            const FxVec3 v1 = hull.verts[faces.vertIdx[f][k]];
            const FxVec3 v2 = hull.verts[faces.vertIdx[f][k + 1]];
            addTetra(v0, v1, v2);
        }
    }

    if (accVol <= 0) return diagFloor();   // degenerate winding/volume -> the diagonal floor

    // ----- Normalize: density = mass / volume; mass = 1/invMass; volume = accVol/6 (accVol is 6·V in Q16.16).
    // The moment integrals are accVol-scaled by det (== 6V) and carry a deferred /120 (the tetra covariance
    // constant). So ∫ x_i x_j ρ dV  =  density · (accXX_etc / 120). With density = mass / V = mass·6 / accVol
    // (accVol in Q16.16 -> the /6 cancels into the constant): the unit-density second moment per axis pair is
    //   moment_ij = accIJ / 120   (a Q16.16 value, deferred).
    // Then the DENSITY scale: ρ = mass / V, V = accVol / (6·kOne)·kOne ... we do it with explicit fxdiv to stay
    // deterministic. mass = fxdiv(kOne, invMass). VOL (Q16.16) = accVol / 6. moment_ij = accIJ / 120.
    const fx mass = fxdiv(kOne, invMass);
    const fx vol  = (fx)(accVol / 6);                       // Q16.16 volume (accVol == 6V)
    if (vol == 0) return diagFloor();
    const fx density = fxdiv(mass, vol);                    // ρ = mass / V (Q16.16)

    // The unit-density second moments (Q16.16): momentIJ = accIJ / 120.
    auto moment = [&](int64_t acc) -> fx { return (fx)(acc / 120); };
    const fx mXX = moment(accXX), mYY = moment(accYY), mZZ = moment(accZZ);
    const fx mXY = moment(accXY), mXZ = moment(accXZ), mYZ = moment(accYZ);

    // The inertia tensor (Q16.16): Ixx = ρ(mYY+mZZ), Iyy = ρ(mXX+mZZ), Izz = ρ(mXX+mYY);
    // Ixy = -ρ·mXY, Ixz = -ρ·mXZ, Iyz = -ρ·mYZ. (Symmetric.)
    FxMat3 I;
    I.m[0] = fxmul(density, mYY + mZZ);
    I.m[4] = fxmul(density, mXX + mZZ);
    I.m[8] = fxmul(density, mXX + mYY);
    I.m[1] = I.m[3] = -fxmul(density, mXY);
    I.m[2] = I.m[6] = -fxmul(density, mXZ);
    I.m[5] = I.m[7] = -fxmul(density, mYZ);

    const FxMat3 invI = FxMat3SymInverse(I);
    // If the inverse degenerated (zero determinant), fall back to the diagonal floor (deterministic).
    if (invI.m[0] == 0 && invI.m[4] == 0 && invI.m[8] == 0) return diagFloor();
    return invI;
}

// ----- WorldInvInertiaFull(body, invIbodyFull): R · invIbodyFull · Rᵀ (the full-tensor world rotation) -------
// Generalizes convex::WorldInvInertia's diagonal outer-product (convex.h:622) to a FULL symmetric body matrix:
// R's columns are the body's WORLD face axes (convex::BoxAxes); the world inverse inertia is R·M·Rᵀ. We build
// R column-major (axis k is column k) and compute R·M then (R·M)·Rᵀ via FxMat3MulVec row dots. Pure integer
// (the products are fxmul, int64 inside). Static body -> M is the zero matrix -> R·0·Rᵀ = 0 (no angular impulse).
inline FxMat3 WorldInvInertiaFull(const FxBody& body, const FxMat3& M) {
    FxVec3 ax[3];
    convex::BoxAxes(body, ax);   // ax[k] = the world k-th face axis (R's column k)
    // R (column-major: R[r][k] = ax[k][r]). RM = R·M. Compute RM row by row.
    // R = [ax0 ax1 ax2] as columns -> R rows: r0=(ax0.x,ax1.x,ax2.x), r1=(ax0.y,ax1.y,ax2.y), r2=(ax0.z,ax1.z,ax2.z).
    auto Mcol = [&](int c) -> FxVec3 { return FxVec3{M.m[0 + c], M.m[3 + c], M.m[6 + c]}; };
    // RM[r][c] = Rrow_r · Mcol_c.
    const FxVec3 Rrow0{ax[0].x, ax[1].x, ax[2].x};
    const FxVec3 Rrow1{ax[0].y, ax[1].y, ax[2].y};
    const FxVec3 Rrow2{ax[0].z, ax[1].z, ax[2].z};
    FxMat3 RM;
    for (int c = 0; c < 3; ++c) {
        const FxVec3 mc = Mcol(c);
        RM.m[0 + c] = FxDot(Rrow0, mc);
        RM.m[3 + c] = FxDot(Rrow1, mc);
        RM.m[6 + c] = FxDot(Rrow2, mc);
    }
    // Out = RM · Rᵀ. Rᵀ's column c == R's row c == ax[*][c]; so Rᵀcol_c = (ax0[c]? ...) -> Rᵀ[r][c]=R[c][r]=ax[r][c].
    // Rᵀ column c = (ax[c].x, ax[c].y, ax[c].z) == ax[c] (since Rᵀcol_c = R row c = ax[*] x-of... ) -> ax[c].
    auto RTcol = [&](int c) -> FxVec3 { return ax[c]; };
    const FxVec3 RMrow0{RM.m[0], RM.m[1], RM.m[2]};
    const FxVec3 RMrow1{RM.m[3], RM.m[4], RM.m[5]};
    const FxVec3 RMrow2{RM.m[6], RM.m[7], RM.m[8]};
    FxMat3 out;
    for (int c = 0; c < 3; ++c) {
        const FxVec3 tc = RTcol(c);
        out.m[0 + c] = FxDot(RMrow0, tc);
        out.m[3 + c] = FxDot(RMrow1, tc);
        out.m[6 + c] = FxDot(RMrow2, tc);
    }
    return out;
}

// ----- StepHullWorldHardened(world, cfg): ONE deterministic HARDENED tick — the gjk::StepHullWorld 5-pass shell
// (gjk.h:1188) with EXACTLY TWO callee swaps: (step 2 inertia) the diagonal FxHullInvInertiaBody+WorldInvInertia
// -> the full FxHullInertiaBodyFull+WorldInvInertiaFull; (steps 3+4 contact) gjk::HullContact -> HullContactMulti
// (MF3). NOTHING else changes: the SAME integrate+damp, the SAME Gauss-Seidel order, the SAME de-pen using
// m.depths[0]/m.normal. The shader copies THIS body VERBATIM. Pure integer, deterministic, identical CPU/GPU.
inline void StepHullWorldHardened(HullWorld& world, const convex::ConvexStepConfig& cfg) {
    const size_t n = world.bodies.size();

    // (1) predict-integrate dynamic bodies + per-tick damping (== gjk::StepHullWorld step 1, VERBATIM).
    for (size_t i = 0; i < n; ++i) {
        if (convex::IsDynamic(world.bodies[i])) {
            fpx::IntegrateBodyFull(world.bodies[i], cfg.gravity, cfg.dt);
            if (cfg.linDamp != kOne) world.bodies[i].vel = convex::FxScale(world.bodies[i].vel, cfg.linDamp);
            if (cfg.angDamp != kOne) world.bodies[i].angVel = convex::FxScale(world.bodies[i].angVel, cfg.angDamp);
        }
    }

    // (2 — SWAP) world inverse inertias once/tick — the FULL tensor (FxHullInertiaBodyFull + WorldInvInertiaFull),
    // replacing the diagonal FxHullInvInertiaBody + convex::WorldInvInertia. Statics -> the zero matrix.
    std::vector<convex::FxMat3> invIW(n);
    for (size_t i = 0; i < n; ++i) {
        const FxHullFaces faces = BuildCanonicalFaces(world.hulls[i]);
        const convex::FxMat3 invIbody = FxHullInertiaBodyFull(world.hulls[i], faces, world.bodies[i].invMass);
        invIW[i] = WorldInvInertiaFull(world.bodies[i], invIbody);
    }

    // (3 — SWAP) impulse solve — world Gauss-Seidel over the all-pairs i<j list (FIXED order), ONE
    // SolveManifoldImpulse sweep per pair, the manifold = HullContactMulti (MF3 multi-point) instead of
    // gjk::HullContact. (== gjk::StepHullWorld step 3 with the single swap.)
    for (uint32_t sweep = 0; sweep < cfg.solveIters; ++sweep) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;
                const convex::ContactManifold m = HullContactMulti(world.bodies[i], world.hulls[i],
                                                                   world.bodies[j], world.hulls[j]);
                if (m.count == 0) continue;
                convex::SolveManifoldImpulse(world.bodies[i], world.bodies[j], invIW[i], invIW[j], m,
                                             cfg.restitution, 1);
            }
        }
    }

    // (4 — SWAP) position de-penetration — cfg.posIters sweeps, FIXED i<j order, the SAME slop+beta linear push
    // using m.depths[0] (MF2 guarantees points[0]/depths[0] is the DEEPEST) + m.normal, the manifold =
    // HullContactMulti. (== gjk::StepHullWorld step 4 with the single swap.)
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
                const FxVec3 ci = convex::FxScale(nrm, fxmul(corrected, wi));
                const FxVec3 cj = convex::FxScale(nrm, fxmul(corrected, wj));
                world.bodies[i].pos = FxSub(world.bodies[i].pos, ci);
                world.bodies[j].pos = FxAdd(world.bodies[j].pos, cj);
            }
        }
    }
    // (5) orientation was already integrated in step (1).
}

// ----- StepHullWorldHardenedN(world, cfg, ticks): run `ticks` StepHullWorldHardened steps. ------------------
inline void StepHullWorldHardenedN(HullWorld& world, const convex::ConvexStepConfig& cfg, uint32_t ticks) {
    for (uint32_t t = 0; t < ticks; ++t) StepHullWorldHardened(world, cfg);
}

// =========================================================================================================
// Slice MF5 — Hull Narrowphase Hardening: LOCKSTEP + ROLLBACK over the hardened stack (the NETCODE HEADLINE
// of FLAGSHIP #25). APPENDED after StepHullWorldHardenedN (MF1-MF4's lines above are BYTE-FROZEN). PURE CPU —
// NO shader, NO RHI. The direct GJ5/BP5/CD5 twin: the SAME frozen lockstep harness with the step swapped from
// gjk::StepHullWorld to StepHullWorldHardened (MF4). StepHullWorldHardened is a fully deterministic integer
// tick whose ONLY mutable replayable state is the bodies vector — the `hulls` are immutable/shared, and the
// faces, the full inertia tensors, and the per-pair multi-point manifolds are RE-DERIVED each tick from the
// bodies/hulls (NOT state to snapshot — exactly why the lockstep holds THROUGH the hardened multi-point/full-
// inertia solve). So MF5 REUSES the frozen gjk:: command/snapshot/equality machinery VERBATIM (no new snapshot
// type): gjk::ApplyHullCommands, gjk::HullSnapshot/SnapshotHull/RestoreHull/HullBodiesEqual, convex::
// ConvexCommand. Both backends run THIS identical CPU harness -> the golden is bit-identical BY CONSTRUCTION
// (cross-vendor 0 px). gjk.h/convex.h/etc + MF1-MF4 BYTE-FROZEN.
// =========================================================================================================

// ----- SimHullTickHardened(world, cfg, commands, tick): ONE deterministic HARDENED tick with its inputs -----
// (1) gjk::ApplyHullCommands(world, commands, tick) — this tick's perturbations, in array order, BEFORE the
// step so the impulse/spin integrates this tick; (2) StepHullWorldHardened(world, cfg) — the MF4 hardened
// 5-pass tick. The gjk::SimHullTick shape (gjk.h:1335) with StepHullWorld swapped for StepHullWorldHardened,
// NOTHING else. Pure integer, fixed order -> bit-identical on every peer/platform.
inline void SimHullTickHardened(HullWorld& world, const convex::ConvexStepConfig& cfg,
                                const std::vector<convex::ConvexCommand>& commands, uint32_t tick) {
    gjk::ApplyHullCommands(world, commands, tick);
    StepHullWorldHardened(world, cfg);
}

// ----- RunHullLockstepHardened(world0, cfg, commands, ticks): two peers converge from inputs alone ----------
// THE peer entry point (the gjk::RunHullLockstep control flow, gjk.h:1379, over SimHullTickHardened). Two
// independent peers (authority + replica) BOTH start from `world0`, BOTH run SimHullTickHardened for `ticks`
// with the SAME command stream (INPUTS ONLY — no state shared) -> BIT-IDENTICAL by determinism, each
// re-deriving the multi-point manifolds + the full inertia every tick. Sets *outIdentical (if non-null) to
// whether the two final body vectors are byte-identical (the make-or-break lockstep proof) + returns the
// converged AUTHORITY world (for the golden). The peer step order is PINNED.
inline HullWorld RunHullLockstepHardened(const HullWorld& world0, const convex::ConvexStepConfig& cfg,
                                         const std::vector<convex::ConvexCommand>& commands, uint32_t ticks,
                                         bool* outIdentical = nullptr) {
    HullWorld authority = world0;   // a fresh copy
    HullWorld replica   = world0;   // the second peer fed the SAME inputs
    for (uint32_t t = 0; t < ticks; ++t) {
        SimHullTickHardened(authority, cfg, commands, t);
        SimHullTickHardened(replica,   cfg, commands, t);
    }
    if (outIdentical) *outIdentical = gjk::HullBodiesEqual(authority.bodies, replica.bodies);
    return authority;
}

// ----- RunHullRollbackHardened(world0, cfg, authStream, mispredictStream, ticks, rollbackAt, ...) ----------
// The rollback harness (the gjk::RunHullRollback control flow, gjk.h:1402, over SimHullTickHardened).
// (1) advance ticks 0..rollbackAt from `world0` applying authStream; (2) SAVE a gjk::HullSnapshot AT
// rollbackAt (gjk::SnapshotHull — the body world); (2b) speculatively advance a few ticks (<=3) with the
// MISPREDICTED stream (a WRONG/extra impulse — the client prediction that diverges), capturing that diverged
// intermediate; (3) ROLLBACK — gjk::RestoreHull to the snapshot + RE-SIMULATE rollbackAt..ticks with the
// CORRECT authStream -> the corrected final world. Returns the corrected world; sets *outCorrectedEqAuthority
// (if non-null) to whether it == RunHullLockstepHardened(world0, cfg, authStream, ticks) byte-for-byte, and
// *outMispredictDiverged (if non-null) to whether the speculative pre-rollback state DIFFERED from the
// authority at the same tick (proving a REAL divergence was corrected). cfg + the streams are CONSTANT.
inline HullWorld RunHullRollbackHardened(const HullWorld& world0, const convex::ConvexStepConfig& cfg,
                                         const std::vector<convex::ConvexCommand>& authStream,
                                         const std::vector<convex::ConvexCommand>& mispredictStream,
                                         uint32_t ticks, uint32_t rollbackAt,
                                         bool* outCorrectedEqAuthority = nullptr,
                                         bool* outMispredictDiverged = nullptr) {
    HullWorld w = world0;
    // (1) advance 0..rollbackAt with the authoritative stream.
    for (uint32_t t = 0; t < rollbackAt; ++t)
        SimHullTickHardened(w, cfg, authStream, t);
    // (2) SAVE the snapshot at rollbackAt (the rollback restore point — just the body world).
    const gjk::HullSnapshot snap = gjk::SnapshotHull(w, rollbackAt);
    // (2b) speculatively advance a few ticks with the MISPREDICTED stream (the wrong/extra impulse — the
    // client prediction that diverges). Bounded to the remaining ticks (<=3). Capture the diverged state.
    uint32_t specTicks = ticks - rollbackAt;
    if (specTicks > 3u) specTicks = 3u;
    for (uint32_t s = 0; s < specTicks; ++s)
        SimHullTickHardened(w, cfg, mispredictStream, rollbackAt + s);
    HullWorld speculative = w;   // the diverged pre-rollback intermediate (for the "real divergence" proof)
    // (3) ROLLBACK: restore the snapshot (the body world) + re-sim rollbackAt..ticks with the authStream.
    gjk::RestoreHull(w, snap);
    for (uint32_t t = rollbackAt; t < ticks; ++t)
        SimHullTickHardened(w, cfg, authStream, t);

    if (outCorrectedEqAuthority || outMispredictDiverged) {
        // The authority advanced the SAME number of speculative ticks (rollbackAt + specTicks) with the
        // CORRECT stream — the apples-to-apples comparison point for the misprediction-diverged proof.
        HullWorld authAtSpec = world0;
        for (uint32_t t = 0; t < rollbackAt + specTicks; ++t)
            SimHullTickHardened(authAtSpec, cfg, authStream, t);
        if (outMispredictDiverged)
            *outMispredictDiverged = !gjk::HullBodiesEqual(speculative.bodies, authAtSpec.bodies);
        if (outCorrectedEqAuthority) {
            const HullWorld authFinal = RunHullLockstepHardened(world0, cfg, authStream, ticks, nullptr);
            *outCorrectedEqAuthority = gjk::HullBodiesEqual(w.bodies, authFinal.bodies);
        }
    }
    return w;
}

}  // namespace manifold
}  // namespace hf::sim
