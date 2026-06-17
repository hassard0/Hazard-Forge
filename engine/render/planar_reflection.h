#pragma once
// Slice DE — Planar Reflections (flat mirror-plane scene reflection) — pure CPU math (header-only,
// no device, no backend symbols). Namespace hf::render::planar. Mirrors reflection_probe.h /
// cubemap.h / gtao.h: a tiny shared-math header ABOVE the RHI seam (ZERO vk*/MTL*/mtl::/Backend::Metal
// CODE symbols — the only mentions of "vk"/"MTL" anywhere in this slice's above-seam files are
// seam-discipline doc comments + the [[vk::binding]] HLSL decorations). The reflection-render path
// (samples/hello_triangle/main.cpp + metal_headless/visual_test.mm) AND the unit test
// (tests/planar_reflection_test.cpp) consume the SAME ReflectionMatrix / ObliqueNearClip / PlaneToView,
// so the GPU reflected-camera math and the CPU test exercise the IDENTICAL reflection — which is what
// makes the reflectivity=0 render byte-identical to the matte render AND bit-identical cross-backend.
//
// THE TECHNIQUE (planar reflections — a flat MIRROR plane reflecting the scene). A flat reflector (a
// mirror floor / still water plane) reflects the surrounding scene. The classic, exact way (distinct
// from screen-space SSR and the cubemap probes DA/DD): RENDER THE SCENE A SECOND TIME through a camera
// that is the main camera REFLECTED across the mirror plane, into a 2D color reflection render target,
// then the mirror surface samples that texture at its own screen-space position and blends it in by
// `reflectivity`. The reflected camera sees exactly what a viewer would see in the mirror, so the
// reflection lines up perfectly with the real geometry (a true mirror, not an approximation).
//
// THE FULL PIPELINE (what the render does with these functions):
//   1. reflectedViewProj = mainProj * ReflectionMatrix(N, planeD) * mainView.
//      ReflectionMatrix is applied in WORLD space (it reflects every world point across the plane),
//      so reflectedView = mainView * ReflectionMatrix maps a world point to where its mirror image
//      would appear in the main camera — i.e. the camera "behind the mirror". (Equivalently:
//      reflectedView = mainView * R; reflectedViewProj = mainProj * reflectedView.)
//   2. clipPlaneView = PlaneToView(N, planeD, reflectedView): the world mirror plane expressed in the
//      reflected camera's view space. Feed it to ObliqueNearClip so the reflection render's NEAR plane
//      IS the mirror plane — geometry BEHIND the mirror (on the far side from the viewer) is clipped
//      away and cannot leak into the reflection. (Lengyel oblique-frustum near-clip.)
//   3. FRONT-FACE WINDING FLIP: a reflection is an improper (handedness-flipping) transform, so a
//      front-facing (CCW) triangle becomes back-facing (CW) in the reflected render. With back-face
//      culling on, the reflected geometry would be wrongly culled. The render fixes this by disabling
//      back-face culling for the reflection pass (the existing GraphicsPipelineDesc.cullNone knob — NO
//      new RHI seam). The mirror floor itself is EXCLUDED from the reflected scene (no mirror-in-mirror
//      recursion; YAGNI per the spec).
//   4. The mirror surface shader samples the reflection RT at its OWN screen UV (SV_Position.xy /
//      screenSize) and blends lerp(matteColor, reflectionSample, reflectivity).
//
// THE reflectivity=0 NO-OP PROOF (what makes this golden-safe — like CR radius=0==no-AO,
// DA parallaxStrength=0==plain-cubemap): the mirror surface blends lerp(matteColor, reflectionSample,
// reflectivity); with reflectivity == 0 the lerp returns matteColor EXACTLY (the reflection texture is
// never read), so the planar-reflection render equals the matte (non-reflective) render — no blend
// bias, no projection drift. The showcase INTERNALLY renders with reflectivity == 0 and asserts it is
// BYTE-IDENTICAL (SHA) to the engine's matte-surface render of the same scene, then renders the real
// reflectivity > 0 version as the golden. The unit test additionally proves ReflectionMatrix reflects
// a point across the plane + is INVOLUTORY (R·R == I) + the oblique near-clip + PlaneToView.
//
// PLANE CONVENTION (used consistently below): a plane is the unit normal `planeNormal` (N) + the
// scalar `planeD` such that a point p lies ON the plane when dot(N, p) + planeD == 0. The signed
// distance from p to the plane is dot(N, p) + planeD (positive on the side N points toward). For a
// mirror floor at world height y == y0 facing UP, N = (0,1,0) and planeD = -y0 (so dot(N,p)+planeD =
// p.y - y0). N is assumed unit (the caller normalizes); these functions do not renormalize.
//
// Pure, deterministic functions: no RNG, no time.

#include "math/math.h"

#include <cmath>

namespace hf::render::planar {

// --- ReflectionMatrix(planeNormal, planeD) -----------------------------------------------------
// The Householder reflection matrix across the plane dot(N, p) + planeD == 0. For a point p it
// produces the mirror image p' = p - 2*(dot(N, p) + planeD)*N, i.e. p reflected to the other side of
// the plane at equal distance. The 3x3 block is I - 2*outer(N, N) (the standard Householder reflector)
// and the translation column is -2*planeD*N (the plane-offset term). Built column-major
// (element(row, col) == m[col*4 + row]), consistent with math::Mat4.
//
// INVOLUTORY: reflecting twice returns the original, so R·R == I (proven by the unit test). This holds
// only when |N| == 1; the caller passes a unit normal. A point ON the plane (dot(N,p)+planeD == 0) is
// a fixed point (unchanged). Used in the render as reflectedView = mainView * ReflectionMatrix(N, d):
// the main camera looking at the reflected world == the virtual camera behind the mirror.
inline math::Mat4 ReflectionMatrix(const math::Vec3& planeNormal, float planeD) {
    const float nx = planeNormal.x, ny = planeNormal.y, nz = planeNormal.z;
    math::Mat4 r = math::Mat4::Identity();
    // 3x3 Householder block: R[row][col] = delta(row,col) - 2*N[row]*N[col].
    r.m[0]  = 1.0f - 2.0f * nx * nx;  // (0,0)
    r.m[1]  =      - 2.0f * ny * nx;  // (1,0)
    r.m[2]  =      - 2.0f * nz * nx;  // (2,0)
    r.m[4]  =      - 2.0f * nx * ny;  // (0,1)
    r.m[5]  = 1.0f - 2.0f * ny * ny;  // (1,1)
    r.m[6]  =      - 2.0f * nz * ny;  // (2,1)
    r.m[8]  =      - 2.0f * nx * nz;  // (0,2)
    r.m[9]  =      - 2.0f * ny * nz;  // (1,2)
    r.m[10] = 1.0f - 2.0f * nz * nz;  // (2,2)
    // Translation column (col 3): -2*planeD*N.
    r.m[12] = -2.0f * planeD * nx;
    r.m[13] = -2.0f * planeD * ny;
    r.m[14] = -2.0f * planeD * nz;
    r.m[15] = 1.0f;
    return r;
}

// --- PlaneToView(planeNormal, planeD, reflectedView) -------------------------------------------
// Transform the WORLD mirror plane (N, planeD) into the reflected camera's VIEW space, returning the
// view-space plane coefficients (a,b,c,d) for which a view-space point pv satisfies
// dot((a,b,c), pv) + d == 0 on the plane. This is the clip plane ObliqueNearClip consumes.
//
// A plane transforms by the INVERSE-TRANSPOSE of the point transform: if pv = V * pw (view = V*world)
// then planeView = inverse-transpose(V) * planeWorld, applied to the 4-vector (N.x, N.y, N.z, planeD).
// For a rigid view matrix the inverse-transpose is still computed generally here (V may carry the
// reflection), so the result is correct for the reflected view too. Pure / deterministic.
//
// Hand-check (axis-aligned, no reflection): for the identity view and the floor plane N=(0,1,0),
// planeD=0, the view-space plane is exactly (0,1,0,0).
inline math::Vec4 PlaneToView(const math::Vec3& planeNormal, float planeD,
                              const math::Mat4& reflectedView) {
    math::Mat4 invV = reflectedView.Inverse();
    // planeView = transpose(inverse(V)) * planeWorld. With column-major storage, multiplying by the
    // transpose means each output component is a DOT of planeWorld with a COLUMN of inverse(V) — i.e.
    // out[i] = sum_k invV(i,k) * plane[k] over the i-th ROW of invV... but transpose swaps row/col, so
    // out[i] = sum_k invV(k,i) * plane[k] = dot(column i of invV, plane). Column i of invV is
    // invV.m[i*4 + 0..3].
    const float p[4] = {planeNormal.x, planeNormal.y, planeNormal.z, planeD};
    math::Vec4 out;
    out.x = invV.m[0]  * p[0] + invV.m[1]  * p[1] + invV.m[2]  * p[2] + invV.m[3]  * p[3];
    out.y = invV.m[4]  * p[0] + invV.m[5]  * p[1] + invV.m[6]  * p[2] + invV.m[7]  * p[3];
    out.z = invV.m[8]  * p[0] + invV.m[9]  * p[1] + invV.m[10] * p[2] + invV.m[11] * p[3];
    out.w = invV.m[12] * p[0] + invV.m[13] * p[1] + invV.m[14] * p[2] + invV.m[15] * p[3];
    return out;
}

// --- ObliqueNearClip(proj, clipPlaneView) ------------------------------------------------------
// Modify a perspective projection so its NEAR plane coincides with the given view-space clip plane
// (the mirror plane). This is Lengyel's oblique-frustum near-clip ("Oblique View Frustum Depth
// Projection and Clipping", Journal of Game Development 2005). After this transform the near plane of
// the frustum is exactly `clipPlaneView`, so geometry on the far side of the mirror is clipped at the
// hardware near plane and cannot leak into the reflection render. Pure / deterministic.
//
// THE TECHNIQUE: in clip space the near plane is the row (-w <= z meaning row3+row2). Lengyel replaces
// the projection's 3rd row (the z/depth row, here columns of row index 2 in column-major storage) so
// that the clip-space plane equals the chosen oblique plane C, scaled so the FAR corner of the frustum
// stays put. With Q = the projection-space point furthest in the clip plane's opposite corner and the
// scale a = (Q dotted into the projection) the new row2 = (a / dot(C, Q)) * C - (the original row3).
// We use the standard formulation:
//     C' = C * (2 * dot(proj_row3, Q) / dot(C, Q));   row2 = C' - row3
// computed against the projection's existing row3 (the perspective w-row, here proj.m[3,7,11,15]).
//
// CONVENTION NOTE: math::Mat4::Perspective is RH, depth [0,1] (Vulkan), with row3 = (0,0,-1,0)
// (m[11] == -1). Lengyel's original derivation is for the GL [-1,1] depth convention; for the [0,1]
// convention the near plane is z == 0 (not z == -w), so the replaced row is row2 directly (the depth
// row) and the far-corner scale uses row3. The implementation below follows the [0,1]-depth oblique
// near-clip: it sets row2 := scaled clip plane such that the near plane becomes `clipPlaneView`.
//
// DEGENERATE GUARD: if the clip plane is degenerate (≈ zero, or dot(C, Q) ≈ 0 — the plane passes
// through the projection's reference corner so the scale would blow up / the plane is effectively at
// infinity), the projection is returned UNCHANGED (no oblique clip — the standard frustum near plane is
// used). This keeps the function total and means "no mirror plane" falls back to a normal render.
inline math::Mat4 ObliqueNearClip(const math::Mat4& proj, const math::Vec4& clipPlaneView) {
    // Degenerate plane (near-zero normal) -> no oblique clip.
    const float n2 = clipPlaneView.x * clipPlaneView.x + clipPlaneView.y * clipPlaneView.y +
                     clipPlaneView.z * clipPlaneView.z;
    if (n2 < 1e-12f) return proj;

    math::Mat4 r = proj;

    // The projection's perspective row (clip w as a function of view xyzw) is row index 3:
    //   w_clip = proj(3,0)*x + proj(3,1)*y + proj(3,2)*z + proj(3,3)*w
    //          = proj.m[3]*x + proj.m[7]*y + proj.m[11]*z + proj.m[15]*w.
    // For math::Mat4::Perspective this row is (0,0,-1,0).
    const float r3x = proj.m[3], r3y = proj.m[7], r3z = proj.m[11], r3w = proj.m[15];

    // Q = the clip-space corner opposite the clip plane, transformed back into projection space. Using
    // the standard Lengyel construction with the projection inverse to find the far corner the oblique
    // plane must preserve. sgn picks the corner on the clip plane's side.
    const float sgnx = (clipPlaneView.x > 0.0f) ? 1.0f : ((clipPlaneView.x < 0.0f) ? -1.0f : 0.0f);
    const float sgny = (clipPlaneView.y > 0.0f) ? 1.0f : ((clipPlaneView.y < 0.0f) ? -1.0f : 0.0f);

    math::Mat4 invP = proj.Inverse();
    // Qclip = (sgn(C.x), sgn(C.y), 1, 1) in clip space; bring it to view/projection space via invP.
    const float qcx = sgnx, qcy = sgny, qcz = 1.0f, qcw = 1.0f;
    math::Vec4 Q;
    Q.x = invP.m[0]  * qcx + invP.m[4]  * qcy + invP.m[8]  * qcz + invP.m[12] * qcw;
    Q.y = invP.m[1]  * qcx + invP.m[5]  * qcy + invP.m[9]  * qcz + invP.m[13] * qcw;
    Q.z = invP.m[2]  * qcx + invP.m[6]  * qcy + invP.m[10] * qcz + invP.m[14] * qcw;
    Q.w = invP.m[3]  * qcx + invP.m[7]  * qcy + invP.m[11] * qcz + invP.m[15] * qcw;

    const float cDotQ = clipPlaneView.x * Q.x + clipPlaneView.y * Q.y +
                        clipPlaneView.z * Q.z + clipPlaneView.w * Q.w;
    if (cDotQ > -1e-9f && cDotQ < 1e-9f) return proj;  // degenerate scale -> leave proj unchanged.

    // Scale the clip plane so the modified depth row preserves the far corner, then write it as the
    // new depth row (row index 2): m[2], m[6], m[10], m[14]. The factor uses the projection's w-row so
    // the result is consistent with the [0,1]-depth perspective (row3 == (0,0,-1,0) -> dot(r3,Q) == -Q.z).
    const float r3DotQ = r3x * Q.x + r3y * Q.y + r3z * Q.z + r3w * Q.w;
    const float scale = r3DotQ / cDotQ;
    r.m[2]  = clipPlaneView.x * scale;
    r.m[6]  = clipPlaneView.y * scale;
    r.m[10] = clipPlaneView.z * scale;
    r.m[14] = clipPlaneView.w * scale;
    return r;
}

}  // namespace hf::render::planar
