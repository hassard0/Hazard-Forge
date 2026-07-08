// Slice AL1 — DETERMINISTIC ANIMATION LAYERING (engine/anim/anim_layer.h, hf::anim::layer): clip
// notifies/events + additive poses + layered slot blending + montages + the notify->GT1/GAS1 tagged-effect
// bridge, all Q16.16 integer past the ONE QuantizeFx boundary (the MM1/AN2 discipline). What this test PINS:
//   (a) NOTIFY: the HALF-OPEN (prevTick, curTick] window (exclusive-low / inclusive-high — the mirror of
//       seq::SampleEvents [tPrev, t)); notifies fire at the exact crossed tick, once each, in ascending
//       (tick, index) order; the boundary convention (a notify AT prevTick does NOT fire, AT curTick DOES);
//       loop wrap fires every occurrence; empty window fires nothing; the pinned fired-trace digest.
//   (b) ADDITIVE: weight-0 == base bit-exact; weight-1 == base (x) delta pinned; a mid-weight nlerp pinned
//       (the identity-slerp integer normalize); an identity-delta additive == base (any weight).
//   (c) LAYERED: a base + an upper-body-masked wave layer — the leg bone follows base (masked out), the arm
//       bone follows the wave (pinned); layer ORDER matters (two orders pinned distinct); a weight-0 layer
//       == base bit-exact.
//   (d) MONTAGE: the 2-section concatenated timeline + the pinned LINEAR blend-in/out curve; the hit notify
//       fires at the exact global tick; JumpToSection lands at the pinned tick; a single no-blend section
//       == the raw clip (weight kOne, field-exact).
//   (e) BRIDGE: the attack montage's "hit" notify routes to a GT1/GAS1 damage ability BEFORE the GAS step;
//       the target's health drops on exactly the hit tick (the pinned health trace + hit tick).
//   (f) DETERMINISM: RunAl1ShotScenario two runs identical + the pinned digests (MSVC == clang);
//       RenderAl1Shot two runs byte-identical.
// Pure C++ (hf_core), ASan-eligible. animation.h/retarget.h/skeleton.h/seq.h/gameplay_tags.h/ability.h
// composed read-only (byte-untouched); standalone: clang++ -std=c++20 -I engine -I tests -I third_party
// tests/anim_layer_test.cpp engine/anim/animation.cpp.
#include "anim/anim_layer.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
using hf::anim::Skeleton;
using hf::anim::Animation;
namespace L = hf::anim::layer;
namespace rt = hf::anim::retarget;
using L::fx;
using L::kOne;
using L::FxQuat;
using L::FxJointPose;

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
static bool quatEq(const FxQuat& q, fx x, fx y, fx z, fx w) {
    return q.x == x && q.y == y && q.z == z && q.w == w;
}
static bool poseEq(const FxJointPose& p, const FxJointPose& b) {
    return quatEq(p.r, b.r.x, b.r.y, b.r.z, b.r.w) && p.t.x == b.t.x && p.t.y == b.t.y && p.t.z == b.t.z;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= (a) NOTIFY — the half-open (prevTick, curTick] window ==========================
    {
        L::NotifyTrack tr;
        tr.notifies = {L::Notify{3, 10, 0}, L::Notify{5, 20, 0}, L::Notify{5, 21, 0}};
        // (2,5]: tick3 crosses in, tick5 crosses in (both ids, index-order tie-break).
        const std::vector<L::FiredNotify> f1 = L::SampleNotifies(tr, 2, 5);
        checkEq((long long)f1.size(), 3, "notify (2,5] fires 3");
        check(f1.size() == 3 && f1[0].tick == 3 && f1[0].notifyId == 10, "notify [0] = tick3 id10");
        check(f1.size() == 3 && f1[1].tick == 5 && f1[1].notifyId == 20, "notify [1] = tick5 id20 (index tie-break)");
        check(f1.size() == 3 && f1[2].tick == 5 && f1[2].notifyId == 21, "notify [2] = tick5 id21");
        // (3,5]: tick3 == prevTick does NOT re-fire (exclusive-low); tick5 DOES (inclusive-high).
        const std::vector<L::FiredNotify> f2 = L::SampleNotifies(tr, 3, 5);
        checkEq((long long)f2.size(), 2, "notify (3,5] excludes prevTick boundary");
        check(f2.size() == 2 && f2[0].notifyId == 20, "notify (3,5] first is tick5");
        // (5,8]: nothing (all notifies at/below 5, exclusive-low at 5).
        checkEq((long long)L::SampleNotifies(tr, 5, 8).size(), 0, "notify (5,8] empty");
        // Empty/negative window fires nothing.
        checkEq((long long)L::SampleNotifies(tr, 5, 5).size(), 0, "notify empty window");
        // LOOP wrap: a notify at local 1 in a loopLen-4 track fires at 1, 5, 9 over (0,9].
        L::NotifyTrack lt;
        lt.loopLen = 4;
        lt.notifies = {L::Notify{1, 10, 0}};
        const std::vector<L::FiredNotify> lw = L::SampleNotifies(lt, 0, 9);
        checkEq((long long)lw.size(), 3, "notify loop (0,9] fires 3 occurrences");
        check(lw.size() == 3 && lw[0].tick == 1 && lw[1].tick == 5 && lw[2].tick == 9,
              "notify loop occurrences {1,5,9}");
        // The pinned fired-trace digest.
        checkHex(L::DigestFired(f1), 0xb691995f2da1d4cdull, "notify fired-trace digest (2,5]");
    }

    // ================= (b) ADDITIVE — identity-slerp compose (base (x) delta) =========================
    {
        FxJointPose base;  base.r = FxQuat{0, 0, 0, kOne};  base.t = L::FxV3{kOne, 0, 0};
        FxJointPose delta; delta.r = L::Al1SwingQuat();     delta.t = L::FxV3{0, 2 * kOne, 0};
        // weight 0 == base bit-exact.
        check(poseEq(L::ApplyAdditiveBone(base, delta, 0), base), "additive w0 == base bit-exact");
        // weight 1 == base (x) delta (base identity -> == qSwing); translation base + delta.
        const FxJointPose a1 = L::ApplyAdditiveBone(base, delta, kOne);
        check(quatEq(a1.r, 32768, 32768, 32768, 32768), "additive w1 rot == qSwing (base identity)");
        check(a1.t.x == 65536 && a1.t.y == 131072 && a1.t.z == 0, "additive w1 t == base + delta");
        // a mid-weight nlerp (the identity-slerp integer normalize) — pinned.
        const FxJointPose am = L::ApplyAdditiveBone(base, delta, kOne / 2);
        check(quatEq(am.r, 18918, 18918, 18918, 56756), "additive wHalf rot pinned (integer nlerp)");
        check(am.t.x == 65536 && am.t.y == 65536 && am.t.z == 0, "additive wHalf t pinned");
        // an identity-delta additive == base (any weight).
        FxJointPose idd;   // identity rotation + zero translation
        check(poseEq(L::ApplyAdditiveBone(base, idd, kOne / 3), base), "additive identity-delta == base");
    }

    // ================= (c) LAYERED — upper-body mask + fold order =====================================
    {
        const Skeleton sk = L::MakeAl1Skeleton();
        const std::vector<FxJointPose> baseP = rt::BindPose(sk);
        std::vector<FxJointPose> wave = rt::BindPose(sk);
        wave[2].t = L::FxV3{0, -kOne, kOne};    // kneeL (leg) — should be MASKED OUT
        wave[6].r = L::Al1SwingQuat();          // shoulderR (arm) — should FOLLOW at full weight
        L::AnimLayerStack st;
        st.base = baseP;
        L::Layer ly;
        ly.pose = wave; ly.weight = kOne; ly.mask = L::MakeUpperBodyMask();
        st.layers = {ly};
        const std::vector<FxJointPose> out = L::EvaluateLayers(st);
        check(poseEq(out[2], baseP[2]), "layered leg(kneeL) follows base (masked out)");
        check(quatEq(out[6].r, 32768, 32768, 32768, 32768), "layered arm(shoulderR) follows the wave");
        // weight-0 layer == base bit-exact.
        L::AnimLayerStack st0 = st;
        st0.layers[0].weight = 0;
        const std::vector<FxJointPose> out0 = L::EvaluateLayers(st0);
        bool eqBase = out0.size() == baseP.size();
        for (size_t j = 0; j < out0.size() && eqBase; ++j) if (!poseEq(out0[j], baseP[j])) eqBase = false;
        check(eqBase, "layered weight-0 layer == base bit-exact");
        // ORDER matters: layer A then B vs B then A on the same bone (spine).
        std::vector<FxJointPose> LA = rt::BindPose(sk); LA[5].r = L::Al1SwingQuat();
        std::vector<FxJointPose> LB = rt::BindPose(sk); LB[5].r = FxQuat{0, 0, kOne, 0};   // 180 about z (exact unit)
        L::Layer la; la.pose = LA; la.weight = kOne / 2;
        L::Layer lb; lb.pose = LB; lb.weight = kOne / 2;
        L::AnimLayerStack ab; ab.base = baseP; ab.layers = {la, lb};
        L::AnimLayerStack ba; ba.base = baseP; ba.layers = {lb, la};
        const std::vector<FxJointPose> oAB = L::EvaluateLayers(ab);
        const std::vector<FxJointPose> oBA = L::EvaluateLayers(ba);
        check(quatEq(oAB[5].r, 11784, 11784, 52606, 35353), "layered order A->B spine pinned");
        check(quatEq(oBA[5].r, 17734, 17734, 42813, 42813), "layered order B->A spine pinned");
        check(!quatEq(oBA[5].r, oAB[5].r.x, oAB[5].r.y, oAB[5].r.z, oAB[5].r.w),
              "layered fold ORDER matters (A->B != B->A)");
    }

    // ================= (d) MONTAGE — sections + blend curve + jump + no-blend identity ================
    {
        const Skeleton sk = L::MakeAl1Skeleton();
        const std::vector<Animation> clips = L::MakeAl1Clips();
        const L::Montage m = L::MakeAl1Montage();
        checkEq(L::MontageTotalTicks(m), 16, "montage total ticks == 16");
        checkEq((long long)m.sections.size(), 2, "montage has 2 sections");
        checkEq(L::SectionStart(m, 1), 8, "montage section 1 starts at tick 8");
        // The LINEAR blend-in/out curve (pinned at a few ticks): section A ramps in over 4 ticks; the
        // strike section B ramps out over the last 4 ticks.
        auto weightAt = [&](int32_t tick) {
            int32_t si = -1, lo = 0;
            L::LocateSection(m, tick, si, lo);
            return (si >= 0) ? L::MontageBlendWeight(m.sections[(size_t)si], lo) : 0;
        };
        checkEq(weightAt(0), 0, "montage blend-in t0 == 0");
        checkEq(weightAt(1), 16384, "montage blend-in t1 == kOne/4");
        checkEq(weightAt(4), kOne, "montage blend-in complete at t4 == kOne");
        checkEq(weightAt(12), 49152, "montage blend-out t12 == 3/4 kOne");
        checkEq(weightAt(15), 0, "montage blend-out ends at 0");
        // The hit notify fires at the exact GLOBAL tick (section A length 8 + local 2 == 10).
        checkEq(L::Al1HitGlobalTick(), 10, "montage hit global tick == 10");
        L::MontagePlayer mp;
        const std::vector<L::FiredNotify> fired = L::StepMontage(m, mp, 20);   // advance through the whole montage
        checkEq((long long)fired.size(), 1, "montage fires exactly the hit notify");
        check(fired.size() == 1 && fired[0].tick == 10 && fired[0].notifyId == L::kNotifyHit,
              "montage hit notify at tick 10");
        // JumpToSection lands at the section's start tick.
        L::MontagePlayer mp2;
        checkEq(L::JumpToSection(m, mp2, 1), 8, "JumpToSection(1) lands at tick 8");
        checkEq((long long)mp2.tick, 8, "player tick == 8 after jump");
        // A single no-blend section == the raw clip (weight kOne, field-exact).
        L::Montage single;
        single.secondsPerTick = 1.0f;
        L::MontageSection s0;
        s0.clip = L::kAl1ClipStrike; s0.lengthTicks = 4;
        single.sections = {s0};
        L::MontagePlayer sp;
        std::vector<FxJointPose> spose; fx sw = 0;
        L::EvaluateMontage(single, sp, clips, sk, spose, sw);
        checkEq(sw, kOne, "montage single no-blend weight == kOne");
        const std::vector<FxJointPose> rawQ =
            rt::QuantizePose(anim::SampleLocalPose(sk, clips[(size_t)L::kAl1ClipStrike], 0.0f));
        check(L::FxPoseEqual(spose, rawQ), "montage single no-blend == raw clip (field-exact)");
    }

    // ================= (e) BRIDGE — notify -> GT1/GAS1 tagged effect, timing exact =====================
    {
        const L::Al1ShotRun run = L::RunAl1ShotScenario();
        // The hit notify fires at shot tick 9 (the montage advances one tick per shot tick, so its global
        // tick is 10 — the hit window); the tagged damage ability is issued BEFORE that tick's GAS step.
        checkEq(run.hitTick, 9, "bridge hit notify fires at shot tick 9 (montage global tick 10)");
        // The target health drops on EXACTLY the hit tick and holds (100 -> 75 = -25 damage).
        bool preFull = true, atDrop = false, postHeld = true;
        for (size_t t = 0; t < run.frames.size(); ++t) {
            const fx hp = run.frames[t].targetHp;
            if ((int)t < run.hitTick && hp != L::kAl1TargetHp) preFull = false;
            if ((int)t == run.hitTick && hp == L::kAl1TargetHp - L::kAl1HitDamage) atDrop = true;
            if ((int)t > run.hitTick && hp != L::kAl1TargetHp - L::kAl1HitDamage) postHeld = false;
        }
        check(preFull, "bridge target at full health before the hit tick");
        check(atDrop, "bridge target health drops by exactly 25 ON the hit tick");
        check(postHeld, "bridge target health holds after the hit");
    }

    // ================= (f) DETERMINISM — shot two-run identity + pinned digests + render ===============
    {
        const L::Al1ShotRun r1 = L::RunAl1ShotScenario();
        const L::Al1ShotRun r2 = L::RunAl1ShotScenario();
        check(r1.digest == r2.digest && r1.poseDigest == r2.poseDigest &&
              r1.healthDigest == r2.healthDigest && r1.notifyDigest == r2.notifyDigest,
              "shot two runs identical");
        checkEq(r1.bones, 9, "shot bones == 9");
        checkEq(r1.layers, 2, "shot layers == 2");
        checkEq(r1.sections, 2, "shot sections == 2");
        checkEq(r1.notifies, 11, "shot total notifies fired == 11");
        // The pinned digests (MSVC == clang — everything past QuantizeFx is integer).
        checkHex(r1.poseDigest, 0x79f5a9b0a781e775ull, "shot pose digest");
        checkHex(r1.healthDigest, 0xf295ee12774c4592ull, "shot health digest");
        checkHex(r1.notifyDigest, 0x953c12b0c6197aa7ull, "shot notify digest");
        checkHex(r1.digest, 0xae29d29015f9dca1ull, "shot combined digest");
        // RenderAl1Shot two runs byte-identical.
        std::vector<uint8_t> i1, i2; uint32_t w1 = 0, h1 = 0, w2 = 0, h2 = 0;
        L::RenderAl1Shot(r1, i1, w1, h1);
        L::RenderAl1Shot(r1, i2, w2, h2);
        check(w1 == 520 && h1 == 420, "shot raster is 520x420");
        check(i1 == i2, "shot raster two runs byte-identical");
    }

    if (g_fail == 0) std::printf("anim_layer_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
