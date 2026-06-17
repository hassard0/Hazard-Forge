// Slice DE — Planar Reflections. Pure CPU math: the householder ReflectionMatrix (reflect a point
// across the mirror plane; involutory R·R == I), the Lengyel ObliqueNearClip (near plane == mirror
// plane: a point behind the mirror is clipped, a point in front is kept), and PlaneToView (world plane
// -> reflected view space). No device, ASan-eligible (links hf_core). Mirrors the math the
// --planar-shot showcase + the planar.frag mirror shader use (engine/render/planar_reflection.h), so
// this test pins the reflection BEFORE any GPU work — which is what makes the reflectivity=0 render
// byte-identical to the matte render AND bit-identical cross-backend.
//
// Properties pinned (per the spec):
//   * Reflection matrix: a point at height +h above a y==0 plane reflects to -h; a point ON the plane
//     is unchanged; the normal direction flips.
//   * Involutory: R·R == I for several planes (axis-aligned + a slanted plane + an offset plane).
//   * Oblique clip: ObliqueNearClip sets the near plane to the mirror plane — a point just IN FRONT of
//     the mirror (viewer side) is kept (clip depth in [0,1]); a point just BEHIND is clipped (depth <
//     0); degenerate clip plane -> proj unchanged.
//   * Plane-to-view: PlaneToView transforms a world plane into reflected view space (hand-checked for
//     an axis-aligned plane with the identity view).
//   * Determinism: same inputs -> identical matrices/vectors.
#include "render/planar_reflection.h"

#include <cmath>
#include <cstdio>

namespace pl = hf::render::planar;
using hf::math::Mat4;
using hf::math::Vec3;
using hf::math::Vec4;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps) { return std::fabs(a - b) <= eps; }
static bool approx3(const Vec3& a, const Vec3& b, float eps) {
    return approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps);
}

// Multiply a column-major Mat4 by a homogeneous Vec4.
static Vec4 mul(const Mat4& m, const Vec4& v) {
    return {
        m.m[0]*v.x + m.m[4]*v.y + m.m[8] *v.z + m.m[12]*v.w,
        m.m[1]*v.x + m.m[5]*v.y + m.m[9] *v.z + m.m[13]*v.w,
        m.m[2]*v.x + m.m[6]*v.y + m.m[10]*v.z + m.m[14]*v.w,
        m.m[3]*v.x + m.m[7]*v.y + m.m[11]*v.z + m.m[15]*v.w,
    };
}

int main() {
    // ---- ReflectionMatrix: reflect a point across the y==0 plane (N=(0,1,0), d=0). ----------------
    {
        const Vec3 N{0, 1, 0};
        const float d = 0.0f;
        Mat4 R = pl::ReflectionMatrix(N, d);
        // A point +h above the floor -> -h below; xz unchanged.
        Vec3 above = hf::math::MulPoint(R, Vec3{2.0f, 3.0f, -1.5f});
        check(approx3(above, Vec3{2.0f, -3.0f, -1.5f}, 1e-5f),
              "ReflectionMatrix: +h above y=0 reflects to -h (xz unchanged)");
        // A point ON the plane is a fixed point.
        Vec3 onPlane = hf::math::MulPoint(R, Vec3{5.0f, 0.0f, 7.0f});
        check(approx3(onPlane, Vec3{5.0f, 0.0f, 7.0f}, 1e-5f),
              "ReflectionMatrix: a point ON the plane is unchanged");
        // The plane normal direction (w=0) flips.
        Vec4 nDir = mul(R, Vec4{0, 1, 0, 0});
        check(approx3(Vec3{nDir.x, nDir.y, nDir.z}, Vec3{0, -1, 0}, 1e-5f),
              "ReflectionMatrix: the normal direction flips");
    }

    // ---- ReflectionMatrix: an OFFSET floor at y == y0 (N=(0,1,0), d = -y0). -----------------------
    {
        const float y0 = 1.5f;
        const Vec3 N{0, 1, 0};
        Mat4 R = pl::ReflectionMatrix(N, -y0);
        // A point at y0 + h reflects to y0 - h (mirror across y == y0).
        Vec3 p = hf::math::MulPoint(R, Vec3{0.0f, y0 + 2.0f, 0.0f});
        check(approx(p.y, y0 - 2.0f, 1e-5f), "ReflectionMatrix: offset plane mirrors across y==y0");
        Vec3 onP = hf::math::MulPoint(R, Vec3{3.0f, y0, -4.0f});
        check(approx3(onP, Vec3{3.0f, y0, -4.0f}, 1e-5f), "ReflectionMatrix: offset plane fixed point");
    }

    // ---- Involutory: R·R == I for several planes. ------------------------------------------------
    {
        struct PlaneCase { Vec3 n; float d; const char* label; };
        PlaneCase cases[] = {
            {{0, 1, 0}, 0.0f, "y=0 floor"},
            {{0, 1, 0}, -2.5f, "y=2.5 offset floor"},
            {{1, 0, 0}, -1.0f, "x=1 wall"},
            {{0, 0, 1}, 3.0f, "z=-3 wall"},
            // A slanted, normalized plane.
            {hf::math::normalize(Vec3{0.4f, 0.8f, -0.447213f}), 0.7f, "slanted"},
        };
        for (const auto& c : cases) {
            Mat4 R = pl::ReflectionMatrix(c.n, c.d);
            Mat4 RR = R * R;
            Mat4 I = Mat4::Identity();
            bool same = true;
            for (int k = 0; k < 16; ++k) same = same && approx(RR.m[k], I.m[k], 1e-4f);
            check(same, "ReflectionMatrix: involutory (R*R == I)");
            (void)c.label;
        }
    }

    // ---- Oblique near-clip: near plane == the clip plane (in front kept, behind clipped). ---------
    // Operate DIRECTLY in view space (which is what ObliqueNearClip transforms): pick a clip plane in
    // view space and verify a view-space point on its POSITIVE (kept) side projects to a visible depth
    // [0,1], a point on its NEGATIVE side projects behind the near plane (depth<0), and a point ON the
    // plane is at depth ~0 (the new near plane). The CONVENTION: clipPlaneView points toward the KEPT
    // half-space; geometry with dot(C.xyz, pv)+C.w > 0 is in front of the oblique near plane.
    {
        Mat4 proj = Mat4::Perspective(1.04719755f /*60deg*/, 16.0f / 9.0f, 0.1f, 100.0f);
        // A view-space clip plane tilted off the z axis (a non-trivial oblique near plane). The normal
        // points toward the KEPT half-space (deeper into the scene, more-negative Z), so geometry on the
        // far side of the plane is kept. C: dot((0, 0.30, -0.95), pv) - 5.0 = 0  -> the near plane is a
        // slanted plane ~5 units deep (it crosses the -Z axis at z ≈ -5.26).
        Vec4 C{0.0f, 0.30f, -0.95f, -5.0f};
        Mat4 oblique = pl::ObliqueNearClip(proj, C);

        auto clip = [&](const Mat4& projM, const Vec3& viewP, float& z, float& w) {
            Vec4 pc = mul(projM, Vec4{viewP.x, viewP.y, viewP.z, 1.0f});
            z = pc.z; w = pc.w;
        };
        auto sdist = [&](const Vec3& pv) { return C.x*pv.x + C.y*pv.y + C.z*pv.z + C.w; };

        // A deep point in front of the camera, on the POSITIVE side of C (kept).
        Vec3 kept{0.0f, 0.0f, -10.0f};
        check(sdist(kept) > 0.0f, "oblique test setup: 'kept' point is on the +side of C");
        float zk, wk; clip(oblique, kept, zk, wk);
        check(wk > 0.0f, "ObliqueNearClip: kept point has w>0 (in front of the camera)");
        check(zk / wk >= -1e-3f && zk / wk <= 1.0f + 1e-3f,
              "ObliqueNearClip: a point in FRONT of the (oblique) near plane is KEPT (depth in [0,1])");

        // A point on the NEGATIVE side of C (between the camera and the oblique plane) is clipped.
        Vec3 behind{0.0f, 0.0f, -2.0f};
        check(sdist(behind) < 0.0f, "oblique test setup: 'behind' point is on the -side of C");
        float zb, wb; clip(oblique, behind, zb, wb);
        if (wb > 1e-5f)
            check(zb / wb < 0.0f, "ObliqueNearClip: a point BEHIND the near plane is CLIPPED (depth<0)");
        else
            check(true, "ObliqueNearClip: behind-plane point clipped (w<=0)");

        // A point exactly ON the clip plane sits at the new near plane: depth ≈ 0. Solve for a point on
        // C along -Z at x=y=0: (-0.95)*z - 5 = 0 -> z = -5/0.95.
        Vec3 onPlane{0.0f, 0.0f, -5.0f / 0.95f};
        check(approx(sdist(onPlane), 0.0f, 1e-4f), "oblique test setup: 'onPlane' point lies on C");
        float zm, wm; clip(oblique, onPlane, zm, wm);
        if (wm > 1e-5f)
            check(approx(zm / wm, 0.0f, 2e-3f), "ObliqueNearClip: a point ON the near plane is at depth ~0");

        // Far geometry stays in front of the camera (w>0). (Oblique near-clip skews the depth mapping
        // away from the near plane — it deliberately distorts depth so the NEAR plane becomes the clip
        // plane; only the near plane is exact, so we don't pin the far depth to exactly 1.)
        Vec3 farPt{0.0f, 0.0f, -100.0f};
        float zf, wf; clip(oblique, farPt, zf, wf);
        check(wf > 0.0f, "ObliqueNearClip: far geometry stays in front of the camera (w>0)");
    }

    // ---- Oblique near-clip: with the REAL reflected-camera pipeline (floor mirror). ---------------
    // The full pipeline: reflView = mainView * ReflectionMatrix(floor); clipPlaneView = PlaneToView(
    // floor, reflView); the geometry standing ON the floor (the objects that appear in the reflection)
    // must survive the oblique clip. We only assert that an on-floor object point projects in front of
    // the camera with a finite depth (the visual golden + the reflectivity=0 proof cover the rest).
    {
        const Vec3 N{0, 1, 0};
        const float planeD = 0.0f;
        Mat4 proj = Mat4::Perspective(1.04719755f, 16.0f / 9.0f, 0.1f, 100.0f);
        Mat4 mainView = Mat4::LookAt(Vec3{0, 3, 6}, Vec3{0, 0.5f, 0}, Vec3{0, 1, 0});
        Mat4 reflView = mainView * pl::ReflectionMatrix(N, planeD);
        Vec4 clipPlaneView = pl::PlaneToView(N, planeD, reflView);
        Mat4 oblique = pl::ObliqueNearClip(proj, clipPlaneView);
        // An object standing on the floor at y=1 (above the mirror): in the reflected view it lands at
        // its mirror image and must project to a finite, in-front-of-camera point.
        Vec4 pv = mul(reflView, Vec4{0.0f, 1.0f, 0.0f, 1.0f});
        Vec4 pc = mul(oblique, pv);
        check(pc.w > 0.0f, "reflected pipeline: an on-floor object projects in front of the camera (w>0)");
    }

    // ---- Oblique near-clip: degenerate clip plane -> proj UNCHANGED. ------------------------------
    {
        Mat4 proj = Mat4::Perspective(1.0f, 1.7f, 0.05f, 50.0f);
        Mat4 same = pl::ObliqueNearClip(proj, Vec4{0, 0, 0, 0});
        bool unchanged = true;
        for (int k = 0; k < 16; ++k) unchanged = unchanged && (same.m[k] == proj.m[k]);
        check(unchanged, "ObliqueNearClip: degenerate (zero) clip plane -> proj unchanged");
    }

    // ---- PlaneToView: hand-check an axis-aligned plane with the identity view. --------------------
    {
        // Identity view: view space == world space, so the plane is unchanged (up to sign/scale).
        Mat4 I = Mat4::Identity();
        Vec4 pv = pl::PlaneToView(Vec3{0, 1, 0}, 0.0f, I);
        check(approx(pv.x, 0.0f, 1e-5f) && approx(pv.y, 1.0f, 1e-5f) &&
              approx(pv.z, 0.0f, 1e-5f) && approx(pv.w, 0.0f, 1e-5f),
              "PlaneToView: identity view leaves an axis-aligned plane unchanged");

        // Offset floor y == 2 (N=(0,1,0), d=-2) under the identity view -> (0,1,0,-2).
        Vec4 pv2 = pl::PlaneToView(Vec3{0, 1, 0}, -2.0f, I);
        check(approx(pv2.y, 1.0f, 1e-5f) && approx(pv2.w, -2.0f, 1e-5f),
              "PlaneToView: identity view preserves the plane offset d");

        // A pure translation view (camera moved up by 2): view = Translate(0,-2,0) (world->view).
        // A world point on y==0 is at view y==-2, so the view-space plane is y == -2 -> (0,1,0,2)?
        // Check: a point ON the world plane must satisfy dot(pvPlane.xyz, pView)+pvPlane.w == 0.
        Mat4 V = Mat4::Translate(Vec3{0, -2, 0});
        Vec4 pvT = pl::PlaneToView(Vec3{0, 1, 0}, 0.0f, V);
        // World point (3,0,-1) on the plane -> view (3,-2,-1).
        float onPlane = pvT.x * 3.0f + pvT.y * (-2.0f) + pvT.z * (-1.0f) + pvT.w;
        check(approx(onPlane, 0.0f, 1e-4f),
              "PlaneToView: a translated view keeps on-plane world points on the view plane");
    }

    // ---- Determinism: same inputs -> identical matrices/vectors. ----------------------------------
    {
        Mat4 a = pl::ReflectionMatrix(Vec3{0, 1, 0}, -1.0f);
        Mat4 b = pl::ReflectionMatrix(Vec3{0, 1, 0}, -1.0f);
        bool same = true;
        for (int k = 0; k < 16; ++k) same = same && (a.m[k] == b.m[k]);
        check(same, "ReflectionMatrix: deterministic (bit-identical re-eval)");

        Mat4 proj = Mat4::Perspective(1.0f, 1.6f, 0.1f, 80.0f);
        Mat4 view = Mat4::LookAt(Vec3{1, 4, 5}, Vec3{0, 0, 0}, Vec3{0, 1, 0});
        Vec4 cp1 = pl::PlaneToView(Vec3{0, 1, 0}, 0.0f, view);
        Vec4 cp2 = pl::PlaneToView(Vec3{0, 1, 0}, 0.0f, view);
        check(cp1.x == cp2.x && cp1.y == cp2.y && cp1.z == cp2.z && cp1.w == cp2.w,
              "PlaneToView: deterministic");
        Mat4 o1 = pl::ObliqueNearClip(proj, cp1);
        Mat4 o2 = pl::ObliqueNearClip(proj, cp2);
        bool osame = true;
        for (int k = 0; k < 16; ++k) osame = osame && (o1.m[k] == o2.m[k]);
        check(osame, "ObliqueNearClip: deterministic");
    }

    if (g_fail == 0) std::printf("planar_reflection_test: all checks passed\n");
    else std::printf("planar_reflection_test: %d checks FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
