// Slice AN2 — DETERMINISTIC ANIMATION RETARGETING (engine/anim/retarget.h, hf::anim::retarget): play
// one skeleton's clip on a DIFFERENTLY-PROPORTIONED skeleton — the bind-delta rotation retarget + the
// root-motion height scale, all Q16.16 integer past the ONE QuantizeFx boundary (the MM1/AN1
// discipline). What this test PINS:
//   (a) IDENTITY: self-retarget (source == target, a NONTRIVIAL exact-unit bind on the spine) reproduces
//       the SOURCE local pose FIELD-EXACT every frame of the multi-frame clip (the bind delta collapses
//       to identity: q (x) conj(q) == (0,0,0,kOne), identity (x) p == p). Height ratio == kOne.
//   (b) BIND-DELTA: the pinned composition R = (tbind (x) conj(sbind)) (x) sanim hand-verified on two
//       cases (sbind=id/tbind=qs/sanim=id -> R=qs; sbind=qs/tbind=id/sanim=qs -> R=identity — the delta
//       cancels the source's own rest); and a full-map bind-delta bone's retargeted local rot + the
//       TARGET WORLD pose preserving TARGET proportions (limb-tip model-space Y == the long-leg value,
//       NOT the source's — pinned).
//   (c) ROOT SCALE: source root strides D, the 2x-height target root strides ~2D (ratio pinned kOne*2);
//       non-root translations == the TARGET bind (proportions), NOT the clip.
//   (d) UNMAPPED: an extra target bone with no source-name match HOLDS its target bind (rotation +
//       translation), and BuildRetargetMap records -1 for it. Overrides win over name matching (pinned).
//   (e) FULL CLIP: the source-model + target-model + retargeted-local digests over the 4-frame walk,
//       PINNED identical MSVC + local clang (everything past QuantizeFx is integer; the clip is exact
//       binary-fraction keys sampled AT keyframe times — cross-compiler exact BY CONSTRUCTION).
//   (f) DETERMINISM: RunRetargetShotScenario two runs identical; RenderRetargetShot two runs byte-
//       identical.
// Pure C++ (hf_core), ASan-eligible. animation.h/skeleton.h/motion_match.h composed read-only
// (byte-untouched); standalone: clang++ -std=c++20 -I engine -I tests tests/retarget_test.cpp
// engine/anim/animation.cpp.
#include "anim/retarget.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace rt = hf::anim::retarget;
using rt::fx;
using rt::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static void checkEq(long long got, long long want, const char* what) {
    if (got != want) { std::printf("FAIL: %s (got %lld want %lld)\n", what, got, want); ++g_fail; }
}
static void checkHex(uint64_t got, uint64_t want, const char* what) {
    if (got != want) {
        std::printf("FAIL: %s (got %016llx want %016llx)\n", what, (unsigned long long)got,
                    (unsigned long long)want);
        ++g_fail;
    }
}

int main() {
    HF_TEST_MAIN_INIT();

    const std::vector<std::string> names = rt::ShowcaseBoneNames();
    const anim::Animation clip = rt::MakeRetargetClip();

    // ---- (a) IDENTITY: self-retarget bit-exact every frame (nontrivial exact-unit spine bind) --------
    {
        anim::Skeleton s = rt::MakeSourceSkeleton();
        s.joints[1].r = math::Quat{0.5f, 0.5f, 0.5f, 0.5f};   // qSwing bind on the spine (exact unit)
        const rt::RetargetMap m = rt::BuildRetargetMap(s, names, s, names, "root", "legLowerL");
        checkEq(m.heightRatio, kOne, "identity height ratio == kOne");
        checkEq(m.mappedCount, 7, "identity all 7 bones mapped by name");
        bool exact = true;
        for (int f = 0; f < rt::kShotFrames; ++f) {
            const std::vector<anim::JointPose> sp = anim::SampleLocalPose(s, clip, (float)f);
            const std::vector<rt::FxJointPose> want = rt::QuantizePose(sp);
            const std::vector<rt::FxJointPose> got = rt::Retarget(m, sp);
            if (!rt::FxPoseEqual(want, got)) exact = false;
        }
        check(exact, "IDENTITY: self-retarget == source pose FIELD-EXACT every frame");
    }

    // ---- (b) BIND-DELTA hand cases + full-map delta with target proportions --------------------------
    {
        const rt::FxQuat qs{32768, 32768, 32768, 32768};   // qSwing = (0.5,0.5,0.5,0.5), exact unit
        const rt::FxQuat id = rt::FxQuatIdentity();
        // Case 1: sbind=id, tbind=qs, sanim=id -> R = qs (D = qs, R = qs (x) id).
        const rt::FxQuat R1 = rt::FxQuatMul(rt::FxQuatMul(qs, rt::FxQuatConj(id)), id);
        check(R1.x == 32768 && R1.y == 32768 && R1.z == 32768 && R1.w == 32768,
              "BIND-DELTA case1 R == qSwing");
        // Case 2: sbind=qs, tbind=id, sanim=qs -> R = identity (the delta cancels the source rest).
        const rt::FxQuat R2 = rt::FxQuatMul(rt::FxQuatMul(id, rt::FxQuatConj(qs)), qs);
        check(R2.x == 0 && R2.y == 0 && R2.z == 0 && R2.w == kOne,
              "BIND-DELTA case2 R == identity (conj(q) (x) q collapses)");

        // Full map: a TARGET whose armUpperL carries a DIFFERENT (qSwing) bind rotation than the source
        // (identity). Retarget the rest pose (sanim == source rest == identity) -> R = tbind (x) conj(id)
        // (x) id == tbind == qSwing. And the target WORLD pose keeps TARGET proportions.
        anim::Skeleton src = rt::MakeSourceSkeleton();
        anim::Skeleton tgt = rt::MakeTargetSkeleton();
        tgt.joints[3].r = math::Quat{0.5f, 0.5f, 0.5f, 0.5f};   // different armUpperL bind
        const rt::RetargetMap m = rt::BuildRetargetMap(src, names, tgt, names, "root", "legLowerL");
        // Sample the source at rest (t == 0: root/leg/arm channels all at their identity keys except
        // legU key0 is identity, armU key0 is qSwing — use a pose where armUpperL source is identity by
        // building an explicit rest pose).
        std::vector<anim::JointPose> rest(src.joints.size());
        for (size_t j = 0; j < src.joints.size(); ++j) {
            rest[j].t = src.joints[j].t; rest[j].r = src.joints[j].r; rest[j].s = src.joints[j].s;
        }
        const std::vector<rt::FxJointPose> tp = rt::Retarget(m, rest);
        check(tp[3].r.x == 32768 && tp[3].r.y == 32768 && tp[3].r.z == 32768 && tp[3].r.w == 32768,
              "BIND-DELTA full-map: rest -> armUpperL local == target bind qSwing");
        // TARGET proportions: the target legLowerL world-Y is the LONG-leg value (-4.0), not source's.
        const std::vector<rt::FxJointModel> gT = rt::ForwardKinematics(tgt, rt::BindPose(tgt));
        const std::vector<rt::FxJointModel> gS = rt::ForwardKinematics(src, rt::BindPose(src));
        checkEq(gT[6].pos.y, -262144, "target legLowerL bind world Y == -4.0 (long leg)");
        checkEq(gS[6].pos.y, -131072, "source legLowerL bind world Y == -2.0 (normal leg)");
        check(gT[6].pos.y != gS[6].pos.y, "target proportions differ from source (not the source's)");
    }

    // ---- (c) ROOT SCALE: source root strides D, 2x-height target root strides 2D; non-root == bind ---
    {
        const anim::Skeleton src = rt::MakeSourceSkeleton();
        const anim::Skeleton tgt = rt::MakeTargetSkeleton();
        const rt::RetargetMap m = rt::BuildRetargetMap(src, names, tgt, names, "root", "legLowerL");
        checkEq(m.heightRatio, 2 * kOne, "root-scale height ratio == 2.0 (target 2x taller)");
        const std::vector<anim::JointPose> sp = anim::SampleLocalPose(src, clip, 3.0f);
        const std::vector<rt::FxJointPose> tp = rt::Retarget(m, sp);
        checkEq(rt::QuantizeFx(sp[0].t.z), 98304, "source root stride z == 1.5 (98304)");
        checkEq(tp[0].t.z, 196608, "target root stride z == 3.0 (2x = 196608)");
        // Non-root spine translation == the TARGET bind (0,1,0), NOT the clip / source.
        checkEq(tp[1].t.x, 0, "target spine bind t.x");
        checkEq(tp[1].t.y, kOne, "target spine bind t.y == 1.0 (target proportion)");
        checkEq(tp[1].t.z, 0, "target spine bind t.z");
    }

    // ---- (d) UNMAPPED bone holds bind; overrides win -------------------------------------------------
    {
        const anim::Skeleton src = rt::MakeSourceSkeleton();
        anim::Skeleton tgt = rt::MakeTargetSkeleton();
        anim::Joint tail; tail.parent = 0; tail.t = math::Vec3{0.0f, -0.5f, 0.0f};
        tail.r = math::Quat{0.5f, 0.5f, 0.5f, 0.5f};
        tgt.joints.push_back(tail);                      // index 7, no source name
        std::vector<std::string> tn = names; tn.push_back("tail");
        const rt::RetargetMap m = rt::BuildRetargetMap(src, names, tgt, tn, "root", "legLowerL");
        checkEq(m.targetToSource[7], -1, "UNMAPPED tail source index == -1");
        checkEq(m.mappedCount, 7, "mapped count excludes the unmapped tail");
        const std::vector<anim::JointPose> sp = anim::SampleLocalPose(src, clip, 1.0f);
        const std::vector<rt::FxJointPose> tp = rt::Retarget(m, sp);
        const std::vector<rt::FxJointPose> tb = rt::BindPose(tgt);
        check(rt::FxPoseEqual({tp[7]}, {tb[7]}), "UNMAPPED tail holds target bind (rot + translation)");
        check(tp[7].r.x == 32768 && tp[7].r.w == 32768, "UNMAPPED tail keeps its bind qSwing rotation");

        // Overrides win: force tail (7) to sample source head (2).
        std::vector<rt::RetargetOverride> ov = {rt::RetargetOverride{7, 2}};
        const rt::RetargetMap mo = rt::BuildRetargetMap(src, names, tgt, tn, "root", "legLowerL", ov);
        checkEq(mo.targetToSource[7], 2, "override forces tail -> source head");
        checkEq(mo.mappedCount, 8, "override raises mapped count to 8");
    }

    // ---- (e) FULL CLIP: the pinned cross-compiler digests --------------------------------------------
    {
        const rt::RetargetShotRun run = rt::RunRetargetShotScenario();
        checkEq(run.srcBones, 7, "shot srcBones");
        checkEq(run.tgtBones, 7, "shot tgtBones");
        checkEq(run.mapped, 7, "shot mapped");
        checkEq(run.frames.size(), 4, "shot frames");
        checkEq(run.heightRatio, 2 * kOne, "shot height ratio == 2.0");
        // PINNED (MSVC == local clang; everything integer past QuantizeFx, exact-binary-fraction clip).
        checkHex(run.sourceDigest, 0x4f3157c153553a0eull, "shot sourceDigest");
        checkHex(run.targetDigest, 0x4be39a78b1b64c70ull, "shot targetDigest (full model-space palette)");
        checkHex(run.localDigest, 0xc1bafb0702205a64ull, "shot localDigest (retargeted local poses)");
        checkHex(run.digest, 0xe861d306897f2d75ull, "shot combined digest");
    }

    // ---- (f) DETERMINISM: scenario + raster two-run identical ----------------------------------------
    {
        const rt::RetargetShotRun a = rt::RunRetargetShotScenario();
        const rt::RetargetShotRun b = rt::RunRetargetShotScenario();
        check(a.digest == b.digest && a.sourceDigest == b.sourceDigest &&
                  a.targetDigest == b.targetDigest && a.localDigest == b.localDigest,
              "DETERMINISM: two scenario runs identical");
        std::vector<uint8_t> img1, img2; uint32_t w1 = 0, h1 = 0, w2 = 0, h2 = 0;
        rt::RenderRetargetShot(a, img1, w1, h1);
        rt::RenderRetargetShot(b, img2, w2, h2);
        check(w1 == w2 && h1 == h2 && img1 == img2, "DETERMINISM: two raster runs byte-identical");
        check(w1 == 520 && h1 == 420, "raster size 520x420");
    }

    if (g_fail == 0) std::printf("retarget_test: ALL PASS\n");
    else             std::printf("retarget_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
