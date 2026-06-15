#include "math/math.h"
#include "anim/skeleton.h"
#include "anim/animation.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace hf;
using namespace hf::math;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

int main() {
    // ---- Quat identity / normalize -------------------------------------------------------------
    {
        Quat id = Quat::Identity();
        check(approx(id.x, 0) && approx(id.y, 0) && approx(id.z, 0) && approx(id.w, 1),
              "Quat::Identity == (0,0,0,1)");
        Quat n = Normalize(Quat{0, 0, 0, 2});
        check(approx(n.w, 1.0f), "Normalize scales w back to 1");
        Quat z = Normalize(Quat{0, 0, 0, 0});  // degenerate -> identity, no NaN
        check(approx(z.w, 1.0f), "Normalize(0) is safe -> identity");
    }

    // ---- QuatToMat4: identity quaternion -> identity matrix ------------------------------------
    {
        Mat4 m = QuatToMat4(Quat::Identity());
        Mat4 i = Mat4::Identity();
        for (int k = 0; k < 16; ++k) check(approx(m.m[k], i.m[k]), "QuatToMat4(identity)==I");
    }

    // ---- 90deg rotation about Z as a quaternion: sin(45)=cos(45)=~0.7071, axis z --------------
    // q = (0,0,sin(45),cos(45)). Should map +x -> +y (matches Mat4::RotateZ(90)).
    {
        float s = std::sin(0.7853981634f), c = std::cos(0.7853981634f);
        Quat qz{0, 0, s, c};
        Mat4 m = QuatToMat4(qz);
        // Column-major apply: out.row = sum_k m[k*4+row]*v[k]. +x -> column 0.
        check(approx(m.m[0], 0.0f) && approx(m.m[1], 1.0f) && approx(m.m[2], 0.0f),
              "QuatZ(90) maps +x to +y");
        // Compare against the analytic RotateZ(90).
        Mat4 rz = Mat4::RotateZ(1.5707963f);
        for (int k = 0; k < 16; ++k) check(approx(m.m[k], rz.m[k]), "QuatZ(90)==RotateZ(90)");
    }

    // ---- Slerp endpoints + halfway --------------------------------------------------------------
    {
        Quat a = Quat::Identity();
        float s = std::sin(0.7853981634f), c = std::cos(0.7853981634f);
        Quat b{0, 0, s, c};  // 90deg about z
        Quat r0 = Slerp(a, b, 0.0f);
        check(approx(r0.w, a.w) && approx(r0.z, a.z), "Slerp(t=0) == a");
        Quat r1 = Slerp(a, b, 1.0f);
        check(approx(r1.z, b.z) && approx(r1.w, b.w), "Slerp(t=1) == b");
        // Shortest-arc: slerp with a near-opposite-sign quaternion flips it; result stays unit.
        Quat bn{0, 0, -s, -c};
        Quat rf = Slerp(a, bn, 0.5f);
        check(approx(rf.x*rf.x + rf.y*rf.y + rf.z*rf.z + rf.w*rf.w, 1.0f, 1e-3f),
              "Slerp result is unit-length");
    }

    // ---- FromTRS: pure translation, pure scale, T*R*S compose ---------------------------------
    {
        Mat4 t = FromTRS(Vec3{1, 2, 3}, Quat::Identity(), Vec3{1, 1, 1});
        check(approx(t.m[12], 1) && approx(t.m[13], 2) && approx(t.m[14], 3), "FromTRS translation");
        Mat4 sc = FromTRS(Vec3{0, 0, 0}, Quat::Identity(), Vec3{2, 3, 4});
        check(approx(sc.m[0], 2) && approx(sc.m[5], 3) && approx(sc.m[10], 4), "FromTRS scale diag");
        // Identity TRS == identity matrix.
        Mat4 id = FromTRS(Vec3{0, 0, 0}, Quat::Identity(), Vec3{1, 1, 1});
        Mat4 I = Mat4::Identity();
        for (int k = 0; k < 16; ++k) check(approx(id.m[k], I.m[k]), "FromTRS(identity)==I");
    }

    // ---- SampleAnimation: identity rest pose + no channels -> palette of identities ------------
    {
        anim::Skeleton sk;
        anim::Joint root;  // parent=-1, identity inverseBind, rest TRS identity
        anim::Joint child; child.parent = 0;
        sk.joints = {root, child};
        anim::Animation empty;  // no channels, duration 0
        std::vector<Mat4> pal = anim::SampleAnimation(sk, empty, 0.0f);
        check(pal.size() == 2, "palette has one matrix per joint");
        Mat4 I = Mat4::Identity();
        for (size_t j = 0; j < pal.size(); ++j)
            for (int k = 0; k < 16; ++k)
                check(approx(pal[j].m[k], I.m[k]), "identity rest pose -> identity palette");
    }

    // ---- SampleAnimation: single joint, 90deg Z rotation channel -> palette == RotateZ(90) -----
    // One root joint with identity inverse-bind; a rotation channel keyed (t=0 -> 90deg about z).
    {
        anim::Skeleton sk;
        anim::Joint root;  // identity rest, identity inverseBind
        sk.joints = {root};

        float s = std::sin(0.7853981634f), c = std::cos(0.7853981634f);
        anim::Animation anim;
        anim.duration = 1.0f;
        anim::Channel ch;
        ch.jointIndex = 0;
        ch.path = anim::Channel::Path::Rotation;
        ch.interp = anim::Channel::Interp::Linear;
        ch.times = {0.0f, 1.0f};
        // Both keys = 90deg about z, so sampling anywhere yields exactly RotateZ(90).
        ch.values = {0, 0, s, c,  0, 0, s, c};
        anim.channels.push_back(ch);

        std::vector<Mat4> pal = anim::SampleAnimation(sk, anim, 0.5f);
        check(pal.size() == 1, "single-joint palette size 1");
        Mat4 expect = Mat4::RotateZ(1.5707963f);
        for (int k = 0; k < 16; ++k)
            check(approx(pal[0].m[k], expect.m[k], 1e-3f),
                  "single-joint 90deg-Z rotation -> RotateZ(90) palette");
    }

    // ---- SampleAnimation: hierarchy — child inherits parent's global transform ----------------
    // Parent translates +x by 5; child is local-translated +y by 2 relative to parent. The child's
    // global should be translate(5,2,0). With identity inverse-bind, palette == global.
    {
        anim::Skeleton sk;
        anim::Joint parent;                 // parent of nothing's parent
        anim::Joint child; child.parent = 0;
        child.t = Vec3{0, 2, 0};            // child rest-local offset
        sk.joints = {parent, child};

        anim::Animation anim;
        anim.duration = 1.0f;
        anim::Channel ch;                    // animate the PARENT translation to (5,0,0)
        ch.jointIndex = 0;
        ch.path = anim::Channel::Path::Translation;
        ch.times = {0.0f};
        ch.values = {5, 0, 0};
        anim.channels.push_back(ch);

        std::vector<Mat4> pal = anim::SampleAnimation(sk, anim, 0.0f);
        check(approx(pal[0].m[12], 5) && approx(pal[0].m[13], 0) && approx(pal[0].m[14], 0),
              "parent global translate (5,0,0)");
        check(approx(pal[1].m[12], 5) && approx(pal[1].m[13], 2) && approx(pal[1].m[14], 0),
              "child global = parent * local = translate(5,2,0)");
    }

    // ---- BlendAnimations: weight 0 == pure A, weight 1 == pure B ------------------------------
    // Two single-joint clips that translate the (identity-rest, identity-inverseBind) root to two
    // different positions. The blended palette at w=0 must equal A's palette, at w=1 equal B's.
    {
        anim::Skeleton sk;
        sk.joints = {anim::Joint{}};  // one root, identity rest + inverseBind

        auto makeTransClip = [](float x, float y, float z) {
            anim::Animation a; a.duration = 1.0f;
            anim::Channel ch;
            ch.jointIndex = 0;
            ch.path = anim::Channel::Path::Translation;
            ch.times = {0.0f};
            ch.values = {x, y, z};
            a.channels.push_back(ch);
            return a;
        };
        anim::Animation A = makeTransClip(2, 0, 0);
        anim::Animation B = makeTransClip(0, 6, 0);

        std::vector<Mat4> palA = anim::SampleAnimation(sk, A, 0.0f);
        std::vector<Mat4> palB = anim::SampleAnimation(sk, B, 0.0f);

        std::vector<Mat4> b0 = anim::BlendAnimations(sk, A, 0.0f, B, 0.0f, 0.0f);
        std::vector<Mat4> b1 = anim::BlendAnimations(sk, A, 0.0f, B, 0.0f, 1.0f);
        check(b0.size() == 1 && b1.size() == 1, "blend palette size 1");
        for (int k = 0; k < 16; ++k) check(approx(b0[0].m[k], palA[0].m[k]), "blend w=0 == pure A");
        for (int k = 0; k < 16; ++k) check(approx(b1[0].m[k], palB[0].m[k]), "blend w=1 == pure B");

        // ---- weight 0.5 of a single-joint translation channel -> hand-checked midpoint ---------
        // A translates to (2,0,0), B to (0,6,0); the 0.5 blend translation is (1,3,0).
        std::vector<Mat4> bMid = anim::BlendAnimations(sk, A, 0.0f, B, 0.0f, 0.5f);
        check(approx(bMid[0].m[12], 1.0f) && approx(bMid[0].m[13], 3.0f) && approx(bMid[0].m[14], 0.0f),
              "blend w=0.5 translation midpoint == (1,3,0)");
    }

    // ---- BlendLocalPoses: slerp of two rotations at 0.5 is normalized (unit length) ------------
    {
        anim::JointPose pa; pa.r = Quat::Identity();        // identity
        anim::JointPose pb;
        float s = std::sin(0.7853981634f), c = std::cos(0.7853981634f);
        pb.r = Quat{0, 0, s, c};                            // 90deg about z
        std::vector<anim::JointPose> a = {pa};
        std::vector<anim::JointPose> b = {pb};
        std::vector<anim::JointPose> mid = anim::BlendLocalPoses(a, b, 0.5f);
        check(mid.size() == 1, "BlendLocalPoses size 1");
        const Quat& r = mid[0].r;
        check(approx(r.x*r.x + r.y*r.y + r.z*r.z + r.w*r.w, 1.0f, 1e-3f),
              "BlendLocalPoses rotation is unit-length");
        // t/s default to rest (identity), so the midpoint t is the lerp of (0,0,0) -> (0,0,0).
        check(approx(mid[0].t.x, 0) && approx(mid[0].t.y, 0) && approx(mid[0].t.z, 0),
              "BlendLocalPoses rest translation stays origin");

        // weight 0 returns A's pose exactly, weight 1 returns B's pose exactly (per-component).
        std::vector<anim::JointPose> m0 = anim::BlendLocalPoses(a, b, 0.0f);
        std::vector<anim::JointPose> m1 = anim::BlendLocalPoses(a, b, 1.0f);
        check(approx(m0[0].r.w, pa.r.w) && approx(m0[0].r.z, pa.r.z), "BlendLocalPoses w=0 == A rot");
        check(approx(m1[0].r.z, pb.r.z) && approx(m1[0].r.w, pb.r.w), "BlendLocalPoses w=1 == B rot");
    }

    if (g_fail == 0) { std::printf("anim_test OK\n"); return 0; }
    std::printf("anim_test: %d failures\n", g_fail);
    return 1;
}
