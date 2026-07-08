// Slice SK1 — DETERMINISTIC SKELETAL-ANIMATION ASSET IMPORT (USD / UsdSkel; engine/asset/usd_skel.h,
// hf::asset). The parity++ asset-import gap: gltf_loader.h ALREADY imports skin+skeleton+animation (device-
// coupled, via cgltf), but fbx_loader.h / usd_loader.h are GEOMETRY ONLY. SK1 adds a SECOND, device-free,
// clean-room skeletal importer — USD — matching the usd_loader.h / fbx_loader.h discipline (UsdSkelImport +
// a pinned net::DigestBytes, cross-compiler exact BY CONSTRUCTION). What this test PINS:
//   (a) SKELETON: the imported bone hierarchy (parent indices, USD path names, decomposed rest TRS,
//       inverse-bind translations) vs the authored 3-joint chain — exact.
//   (b) SKIN: per-vertex bone indices + weights (remapped to skeleton order, renormalized to sum 1); and the
//       >4-influence handling (NormalizeInfluences: top-4 by weight + renormalize) pinned on a synthetic 5-
//       influence vertex.
//   (c) ANIMATION: imported clip channels (joint target / path / interp / keyframe times) exact; sampling AT
//       a keyframe reproduces the stored quat; BETWEEN keys reproduces the pinned linear/slerp value (Q16.16).
//   (d) POSE ROUND-TRIP: the imported rig posed at clip times -> the integer model-space palette digest;
//       plus a skinned VERTEX position (the imported skin composed with SampleAnimation) pinned in Q16.16.
//   (e) COMPOSE: the imported skeleton feeds retarget.h (AN2) — a self-retarget of the imported clip
//       reproduces the pose digest (the delta collapses to identity), proving imported rigs feed the stack.
//   (f) DETERMINISM: the import digest + the shot scenario digests PINNED identical MSVC + local clang (the
//       fixture's bind/rest are translation-only -> every folded float is exactly representable); two shot
//       runs identical; RenderSk1Shot two runs byte-identical.
// Pure C++ (hf_core), ASan-eligible. gltf_loader/skeleton.h/animation.h/retarget.h composed read-only
// (byte-untouched). Standalone: clang++ -std=c++20 -I engine -I tests tests/usd_skel_test.cpp
// engine/anim/animation.cpp. The test asset is a hand-authored UsdSkel LITERAL (Sk1UsdaAsset()), imported
// through the real parser (no .usd file on disk).
#include "asset/usd_skel.h"
#include "asset/usd_skel_shot.h"
#include "anim/animation.h"
#include "anim/retarget.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "test_main.h"

using namespace hf;
namespace rt = hf::anim::retarget;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static void checkEq(long long got, long long want, const char* what) {
    if (got != want) { std::printf("FAIL: %s (got %lld want %lld)\n", what, got, want); ++g_fail; }
}
static void checkHex(uint64_t got, uint64_t want, const char* what) {
    if (got != want)
        { std::printf("FAIL: %s (got %016llx want %016llx)\n", what, (unsigned long long)got,
                      (unsigned long long)want); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    const asset::UsdSkelImport s =
        asset::ImportUsdSkel(asset::Sk1UsdaAsset(), std::strlen(asset::Sk1UsdaAsset()));

    // ---- (a) SKELETON ------------------------------------------------------------------------------------
    check(s.ok, "import ok");
    checkEq((long long)s.skeleton.joints.size(), 3, "3 joints imported");
    checkEq((long long)s.timeCodesPerSecond, 24, "timeCodesPerSecond == 24");
    if (s.skeleton.joints.size() == 3) {
        checkEq(s.skeleton.joints[0].parent, -1, "joint0 root");
        checkEq(s.skeleton.joints[1].parent,  0, "joint1 parent==0");
        checkEq(s.skeleton.joints[2].parent,  1, "joint2 parent==1");
        check(s.jointNames[0] == "Root",          "joint0 name Root");
        check(s.jointNames[1] == "Root/Hip",      "joint1 name Root/Hip");
        check(s.jointNames[2] == "Root/Hip/Knee", "joint2 name Root/Hip/Knee");
        // rest LOCAL TRS: Root at origin; Hip/Knee each +1 in Y; identity rot; unit scale (all exact).
        checkEq(rt::QuantizeFx(s.skeleton.joints[0].t.y), 0,     "root rest t.y == 0");
        checkEq(rt::QuantizeFx(s.skeleton.joints[1].t.y), 65536, "hip rest t.y == 1");
        checkEq(rt::QuantizeFx(s.skeleton.joints[2].t.y), 65536, "knee rest t.y == 1");
        for (int j = 0; j < 3; ++j) {
            checkEq(rt::QuantizeFx(s.skeleton.joints[j].r.w), 65536, "rest rot identity w==1");
            checkEq(rt::QuantizeFx(s.skeleton.joints[j].s.x), 65536, "rest scale x==1");
        }
        // inverse-bind WORLD translations: 0, -1, -2 (bind Y of 0/1/2).
        checkEq(rt::QuantizeFx(s.skeleton.joints[0].inverseBind.m[13]),  0,       "ib0.ty == 0");
        checkEq(rt::QuantizeFx(s.skeleton.joints[1].inverseBind.m[13]), -65536,   "ib1.ty == -1");
        checkEq(rt::QuantizeFx(s.skeleton.joints[2].inverseBind.m[13]), -131072,  "ib2.ty == -2");
    }

    // ---- (b) SKIN ----------------------------------------------------------------------------------------
    checkEq((long long)s.vertices.size(), 6, "6 skinned vertices");
    checkEq((long long)s.indices.size(), 12, "2 quads -> 12 triangle indices");
    if (s.vertices.size() == 6) {
        // vertex 0: authored jointIndices [0,1], weights [0.8,0.2] (already skeleton order).
        checkEq((int)s.vertices[0].joints[0], 0, "v0 joint0 == 0");
        checkEq((int)s.vertices[0].joints[1], 1, "v0 joint1 == 1");
        checkEq(rt::QuantizeFx(s.vertices[0].weights[0]), rt::QuantizeFx(0.8f), "v0 w0 == 0.8");
        checkEq(rt::QuantizeFx(s.vertices[0].weights[1]), rt::QuantizeFx(0.2f), "v0 w1 == 0.2");
        // weights normalized to sum 1 (Q16.16 exact for these authored pairs).
        for (int v = 0; v < 6; ++v) {
            const float sum = s.vertices[v].weights[0] + s.vertices[v].weights[1] +
                              s.vertices[v].weights[2] + s.vertices[v].weights[3];
            checkEq(rt::QuantizeFx(sum), 65536, "vertex weights sum to 1");
        }
        // vertex 4: authored [2,1] w [0.9,0.1] -> joints (2,1,0,0).
        checkEq((int)s.vertices[4].joints[0], 2, "v4 joint0 == 2 (knee)");
        checkEq((int)s.vertices[4].joints[1], 1, "v4 joint1 == 1 (hip)");
    }
    // >4-influence truncation + renormalize: 5 influences, weights {0.1,0.4,0.05,0.25,0.2} -> top4 keeps
    // idx {1,2,0,3} (drops the 0.05 at idx 4), renormalized to sum 1.
    {
        int32_t idx[5] = {3, 1, 4, 2, 0};
        float   wt[5]  = {0.1f, 0.4f, 0.05f, 0.25f, 0.2f};
        float oj[4], ow[4];
        asset::NormalizeInfluences(idx, wt, 5, oj, ow);
        checkEq((int)oj[0], 1, "trunc keep0 idx1 (0.4)");
        checkEq((int)oj[1], 2, "trunc keep1 idx2 (0.25)");
        checkEq((int)oj[2], 0, "trunc keep2 idx0 (0.2)");
        checkEq((int)oj[3], 3, "trunc keep3 idx3 (0.1)");
        check(ow[0] > ow[1] && ow[1] > ow[2] && ow[2] > ow[3], "trunc weights descending");
        checkEq(rt::QuantizeFx(ow[0] + ow[1] + ow[2] + ow[3]), 65536, "trunc renormalized sum == 1");
    }

    // ---- (c) ANIMATION -----------------------------------------------------------------------------------
    checkEq((long long)s.animations.size(), 1, "1 clip");
    if (!s.animations.empty()) {
        const anim::Animation& clip = s.animations[0];
        checkEq(rt::QuantizeFx(clip.duration), 65536, "clip duration == 1.0s");
        // 4 channels: rotation on joints 1,2 (3 keys each) + translation on joints 1,2 (2 keys each).
        checkEq((long long)clip.channels.size(), 4, "4 channels (2 rot + 2 trans)");
        int rotKeys = 0, transKeys = 0;
        for (const anim::Channel& c : clip.channels) {
            check(c.interp == anim::Channel::Interp::Linear, "channel interp Linear");
            if (c.path == anim::Channel::Path::Rotation)    rotKeys   += (int)c.times.size();
            if (c.path == anim::Channel::Path::Translation) transKeys += (int)c.times.size();
        }
        checkEq(rotKeys, 6, "6 rotation keyframes");
        checkEq(transKeys, 4, "4 translation keyframes");

        // AT a keyframe (t=0.5s, frame 12): joint 1 rotation == the stored 45deg-about-Z quat (exact).
        {
            const std::vector<anim::JointPose> pose = anim::SampleLocalPose(s.skeleton, clip, 0.5f);
            checkEq(rt::QuantizeFx(pose[1].r.z), 25080, "keyframe rot z (Q16.16)");
            checkEq(rt::QuantizeFx(pose[1].r.w), 60547, "keyframe rot w (Q16.16)");
        }
        // BETWEEN keys (t=0.25s, slerp identity<->45deg): the pinned interpolated value (Q16.16 absorbs the
        // sub-ULP float differences -> cross-compiler exact).
        {
            const std::vector<anim::JointPose> pose = anim::SampleLocalPose(s.skeleton, clip, 0.25f);
            checkEq(rt::QuantizeFx(pose[1].r.z), 12785, "interp rot z (Q16.16)");
            checkEq(rt::QuantizeFx(pose[1].r.w), 64277, "interp rot w (Q16.16)");
        }
    }

    // ---- (d) POSE ROUND-TRIP: skinned vertex position via the imported skin + SampleAnimation ------------
    if (!s.animations.empty() && s.vertices.size() == 6) {
        const std::vector<math::Mat4> palette = anim::SampleAnimation(s.skeleton, s.animations[0], 1.0f);
        const scene::SkinnedVertex& v = s.vertices[4];   // top strip vertex (0.9 knee / 0.1 hip)
        const math::Vec3 p{v.pos[0], v.pos[1], v.pos[2]};
        math::Vec3 acc{0, 0, 0};
        for (int i = 0; i < 4; ++i) {
            if (v.weights[i] <= 0.0f) continue;
            const math::Vec3 tp = math::MulPoint(palette[(int)v.joints[i]], p);
            acc.x += tp.x * v.weights[i]; acc.y += tp.y * v.weights[i]; acc.z += tp.z * v.weights[i];
        }
        checkEq(rt::QuantizeFx(acc.x), -36045, "skinned v4 x @t1.0 (Q16.16)");
        checkEq(rt::QuantizeFx(acc.y),  62259, "skinned v4 y @t1.0 (Q16.16)");
        checkEq(rt::QuantizeFx(acc.z),      0, "skinned v4 z @t1.0 (Q16.16)");
    }

    // ---- (e)+(f) COMPOSE + DETERMINISM: the shot scenario (imported rig -> QuantizePose -> integer FK ----
    // -> model palette; and a self-retarget through AN2). Digests PINNED identical MSVC + clang.
    {
        const asset::sk1::Sk1ShotRun run  = asset::sk1::RunSk1ShotScenario();
        const asset::sk1::Sk1ShotRun run2 = asset::sk1::RunSk1ShotScenario();
        check(run.digest == run2.digest, "shot two-run identical");
        checkEq(run.bones, 3, "shot bones == 3");
        checkEq(run.verts, 6, "shot verts == 6");
        checkEq(run.clips, 1, "shot clips == 1");
        checkEq(run.keyframes, 10, "shot keyframes == 10");
        checkHex(run.importDigest,   0xb8c46aebee4ee32bull, "import digest pinned (MSVC==clang)");
        checkHex(run.poseDigest,     0xb5585d947041fe24ull, "pose digest pinned (MSVC==clang)");
        checkHex(run.retargetDigest, 0xb5585d947041fe24ull, "retarget compose digest pinned");
        checkHex(run.digest,         0xf7a315563d264826ull, "combined shot digest pinned");
        // (e) self-retarget reproduces the pose digest (imported rig drives AN2 bit-exact).
        check(run.poseDigest == run.retargetDigest, "self-retarget reproduces imported pose (AN2 compose)");
        // (f) the shared raster is byte-identical across two runs.
        std::vector<uint8_t> a, b; uint32_t aw = 0, ah = 0, bw = 0, bh = 0;
        asset::sk1::RenderSk1Shot(run, a, aw, ah);
        asset::sk1::RenderSk1Shot(run2, b, bw, bh);
        check(aw == bw && ah == bh && a == b, "RenderSk1Shot two runs byte-identical");
        checkEq((long long)aw, 420, "shot width 420");
        checkEq((long long)ah, 360, "shot height 360");
    }

    // ---- Robustness: malformed / empty input -> ok=false (no crash) --------------------------------------
    {
        const asset::UsdSkelImport bad = asset::ImportUsdSkel("#usda 1.0\ndef Xform \"x\" {}\n", 24);
        check(!bad.ok, "no Skeleton -> ok=false");
        const asset::UsdSkelImport empty = asset::ImportUsdSkel(nullptr, 0);
        check(!empty.ok, "null input -> ok=false");
    }

    if (g_fail == 0) std::printf("usd_skel_test: ALL PASS\n");
    else             std::printf("usd_skel_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
