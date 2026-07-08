// Slice AN1 — DETERMINISTIC BLEND SPACES (engine/anim/blend_space.h, hf::anim::bs): 1D + 2D
// parameter-driven animation blending — integer params/weights/point-in-triangle/barycentric +
// tick-based slew over the EXISTING animation.h pose stack (SampleLocalPose + BlendLocalPoses, the
// reused 2-pose seam). What this test PINS:
//   (a) 1D: EvaluateWeights1D conventions (clamp-outside {i,i,0}; exact interior sample -> w == 0;
//       the exact integer lerp factor incl. the truncating nondivisible case 21845); identity-at-
//       sample (param AT a sample == that clip's SampleLocalPose FIELD-EXACT — the direct-sample
//       path, no float blend); the exact 50/50 midway blend (hand-derived quantized components +
//       the pinned pose digest); clamping == the edge clip exactly.
//   (b) PHASE SYNC (the naive-implementation trap, proven): walk (2.0 s loop) + run (1.0 s loop —
//       walk at EXACTLY 2x rate) blended 50/50 at phase 0.25 -> both sampled at the SAME normalized
//       phase: blended left-foot z == +0.375 (24576) and root z == 0.75 (49152), the foot-plant-
//       aligned values. The NAIVE same-absolute-time control (BlendLocalPoses at t == 0.5 s both)
//       gives foot z == 0 (the swing CANCELS — sliding feet) and root z == 1.125 (73728). BOTH
//       pinned; distinct -> the sync rule is load-bearing.
//   (c) 2D: identity-at-vertex EXACT (w == {kOne,0,0}, direct-sample path); the exact centroid
//       thirds {21845, 21845, 21846} (the pinned RESIDUAL convention — the sum is kOne EXACTLY) +
//       the hand-derived 3-way LEFT-FOLD pose values (rootZq == 32768: 0.375 + 0.375*21846/65536 =
//       32768.25/65536, llround -> 32768); boundary-INCLUSIVE edge containment ({32768,32768,0} ON
//       the edge); the shared-edge FIRST-TRIANGLE-WINS tie-break; outside-hull clamp to the nearest
//       edge (projection t == 21845 on the base edge) and to a VERTEX past the endpoints (w0 ==
//       kOne, pose == that clip EXACT) incl. the equidistant scan-order tie-break.
//   (d) SLEW: a step change ramps at EXACTLY rate/tick (pinned per-tick values, arrival tick,
//       negative direction, the box-independent 2D axes); rate == 0 -> instant (identity).
//       AdvancePhase wraps mod kOne (pinned).
//   (e) FSM adapter: TickBlendDriver1D pulls the bound StateMachine float param through the ONE
//       QuantizeFx boundary, slews + advances phase (pinned traces); EvaluateBlendDriver1D ==
//       EvaluatePose1D at the driver cursor.
//   (f) DETERMINISM: RunBlendShotScenario two runs identical; pathDigest/weightsDigest/poseDigest
//       PINNED (must be identical under MSVC and local clang — the cross-compiler proof; the
//       fixtures are exact binary fractions, translation-only rotations, per the MM1 discipline);
//       step[0] hand-derived; the clamp leg really clamps; RenderBlendShot two runs byte-identical.
// Pure C++ (hf_core), ASan-eligible. animation.h/skeleton.h/state_machine.h/motion_match.h composed
// read-only (byte-untouched).
#include "anim/blend_space.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace bs = hf::anim::bs;
using bs::fx;
using bs::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Field-exact pose equality (the identity proofs — NO tolerance; the direct-sample path must return
// the same float bits SampleLocalPose returns).
static bool PoseEqual(const std::vector<anim::JointPose>& a, const std::vector<anim::JointPose>& b) {
    if (a.size() != b.size()) return false;
    for (size_t j = 0; j < a.size(); ++j) {
        if (std::memcmp(&a[j].t, &b[j].t, sizeof(a[j].t)) != 0) return false;
        if (std::memcmp(&a[j].r, &b[j].r, sizeof(a[j].r)) != 0) return false;
        if (std::memcmp(&a[j].s, &b[j].s, sizeof(a[j].s)) != 0) return false;
    }
    return true;
}

// The shared 1D fixture: idle @ 0.0, walk @ 1.0, run @ 2.0 (speed axis) over the MM1 rig + clips
// (+ the AN1 run clip — walk at exactly 2x rate).
static bs::BlendSpace1D MakeSpace1D() {
    bs::BlendSpace1D s;
    s.samples = {bs::BlendSample1D{0, 0}, bs::BlendSample1D{kOne, 1}, bs::BlendSample1D{2 * kOne, 2}};
    return s;
}

// The dedicated 2D test triangle: idle (0,0), walk (3,0), run (0,3) — picked so the CENTROID is the
// exact integer point (kOne, kOne) (exact-thirds barycentrics) and the base-edge projection of
// (1, -1) is the nondivisible t = 21845.
static bs::BlendSpace2D MakeTestTri2D() {
    bs::BlendSpace2D s;
    s.samples = {bs::BlendSample2D{0, 0, 0}, bs::BlendSample2D{3 * kOne, 0, 1},
                 bs::BlendSample2D{0, 3 * kOne, 2}};
    s.tris = {bs::BlendTri{0, 1, 2}};
    return s;
}

int main() {
    HF_TEST_MAIN_INIT();

    const anim::Skeleton sk = anim::mm::MakeMMTestRig();
    const std::vector<anim::Animation> clips = bs::MakeShowcaseClips();   // {idle, walk, run}
    const fx kQuarterPhase = kOne / 4;   // phase 0.25: walk t = 0.5 s, run t = 0.25 s, idle t = 0.5 s

    // ================= (a) 1D: weight conventions + identity-at-sample + midway + clamp =============
    {
        const bs::BlendSpace1D s = MakeSpace1D();

        // Weight conventions (pinned).
        bs::Weights1D w = bs::EvaluateWeights1D(s, -kOne);
        check(w.i0 == 0 && w.i1 == 0 && w.w == 0, "1d: below range clamps {0,0,0}");
        w = bs::EvaluateWeights1D(s, 0);
        check(w.i0 == 0 && w.i1 == 0 && w.w == 0, "1d: at the first sample {0,0,0}");
        w = bs::EvaluateWeights1D(s, kOne / 2);
        check(w.i0 == 0 && w.i1 == 1 && w.w == 32768, "1d: midway idle-walk == exactly 32768");
        w = bs::EvaluateWeights1D(s, kOne);
        check(w.i0 == 1 && w.i1 == 2 && w.w == 0, "1d: AT an interior sample -> w == 0 (identity)");
        w = bs::EvaluateWeights1D(s, kOne + kOne / 2);
        check(w.i0 == 1 && w.i1 == 2 && w.w == 32768, "1d: midway walk-run == exactly 32768");
        w = bs::EvaluateWeights1D(s, 5 * kOne);
        check(w.i0 == 2 && w.i1 == 2 && w.w == 0, "1d: above range clamps {n-1,n-1,0}");
        w = bs::EvaluateWeights1D(s, kOne / 4);
        check(w.i0 == 0 && w.i1 == 1 && w.w == 16384, "1d: quarter == exactly 16384");
        // The truncating nondivisible case: span 3.0, offset 1.0 -> floor(65536/3) == 21845.
        bs::BlendSpace1D s3;
        s3.samples = {bs::BlendSample1D{0, 0}, bs::BlendSample1D{3 * kOne, 1}};
        w = bs::EvaluateWeights1D(s3, kOne);
        check(w.i0 == 0 && w.i1 == 1 && w.w == 21845, "1d: nondivisible weight truncates to 21845");
        check(bs::EvaluateWeights1D(bs::BlendSpace1D{}, 0).i0 == -1, "1d: empty space -> {-1,-1,0}");

        // Identity-at-sample: param AT walk == SampleLocalPose(walk, phase*dur) FIELD-EXACT.
        const std::vector<anim::JointPose> atWalk =
            bs::EvaluatePose1D(sk, clips, s, kOne, kQuarterPhase);
        check(PoseEqual(atWalk, anim::SampleLocalPose(sk, clips[1], 0.5f)),
              "1d: identity-at-sample (walk pose bit-exact, direct-sample path)");

        // Clamping == the edge clip exactly.
        check(PoseEqual(bs::EvaluatePose1D(sk, clips, s, -kOne, kQuarterPhase),
                        anim::SampleLocalPose(sk, clips[0], 0.5f)),
              "1d: below-range clamp == idle exactly");
        check(PoseEqual(bs::EvaluatePose1D(sk, clips, s, 5 * kOne, kQuarterPhase),
                        anim::SampleLocalPose(sk, clips[2], 0.25f)),
              "1d: above-range clamp == run exactly (run sampled at ITS phase time 0.25 s)");

        // The exact 50/50 idle-walk blend at phase 0.25 (hand-derived quantized components):
        //   root.z: (0 + 0.75)/2 == 0.375 -> 24576;  lf: (-0.25, (0.03125+0)/2, (0+0.375)/2)
        //   == (-0.25, 0.015625, 0.1875) -> (-16384, 1024, 12288).
        const std::vector<anim::JointPose> mid =
            bs::EvaluatePose1D(sk, clips, s, kOne / 2, kQuarterPhase);
        check(mid.size() == 3, "1d: midway pose has 3 joints");
        check(bs::QuantizeFx(mid[0].t.z) == 24576, "1d: midway root z == 24576 (0.375)");
        check(bs::QuantizeFx(mid[1].t.x) == -16384 && bs::QuantizeFx(mid[1].t.y) == 1024 &&
                  bs::QuantizeFx(mid[1].t.z) == 12288,
              "1d: midway left foot == (-16384, 1024, 12288)");
        const uint64_t midDigest = bs::DigestPoseQ(mid);
        std::printf("1d midway pose digest: %016llx\n", (unsigned long long)midDigest);
        check(midDigest == 0x2bb0f1f447c966d1ull, "1d: midway pose digest PINNED");
    }

    // ================= (b) PHASE SYNC: the foot-plant alignment proof ===============================
    {
        const bs::BlendSpace1D s = MakeSpace1D();
        // PHASE-SYNCED: walk sampled at 0.25*2.0 == 0.5 s, run at 0.25*1.0 == 0.25 s — the SAME
        // normalized phase. Run is walk at exactly 2x, so both feet are at the SAME stride point:
        // lf.z == +0.375 in both -> the blend keeps +0.375 (24576). Roots have both traveled 25% of
        // the loop distance 3.0 -> 0.75 (49152).
        const std::vector<anim::JointPose> synced =
            bs::EvaluatePose1D(sk, clips, s, kOne + kOne / 2, kQuarterPhase);
        const fx syncedFootZ = bs::QuantizeFx(synced[1].t.z);
        const fx syncedRootZ = bs::QuantizeFx(synced[0].t.z);
        check(syncedFootZ == 24576, "sync: phase-synced blended foot z == 24576 (+0.375, aligned)");
        check(syncedRootZ == 49152, "sync: phase-synced blended root z == 49152 (0.75)");

        // NAIVE same-absolute-time control (what BlendAnimations does when handed equal times): both
        // clips at t == 0.5 s -> run is at ITS phase 0.5 (mid-stride the OTHER way, lf.z == -0.375):
        // the swing CANCELS to 0 — the sliding-feet artifact. Root: (0.75 + 1.5)/2 == 1.125.
        const std::vector<anim::JointPose> naive = anim::BlendLocalPoses(
            anim::SampleLocalPose(sk, clips[1], 0.5f), anim::SampleLocalPose(sk, clips[2], 0.5f),
            0.5f);
        const fx naiveFootZ = bs::QuantizeFx(naive[1].t.z);
        const fx naiveRootZ = bs::QuantizeFx(naive[0].t.z);
        check(naiveFootZ == 0, "sync: NAIVE time-synced foot z == 0 (the swing cancels — sliding)");
        check(naiveRootZ == 73728, "sync: NAIVE root z == 73728 (1.125)");
        check(syncedFootZ != naiveFootZ && syncedRootZ != naiveRootZ,
              "sync: the distinction is real (phase-sync != naive time-sync)");
    }

    // ================= (c) 2D: vertex identity + centroid + edge conventions + clamp ================
    {
        const bs::BlendSpace2D s = MakeTestTri2D();

        // Identity-at-vertex: weights EXACT {kOne,0,0}; pose == that clip FIELD-EXACT (direct path).
        bs::Weights2D w = bs::EvaluateWeights2D(s, 0, 0);
        check(w.tri == 0 && !w.clamped && w.w0 == kOne && w.w1 == 0 && w.w2 == 0,
              "2d: vertex weights == {kOne, 0, 0} exactly");
        check(PoseEqual(bs::EvaluatePose2D(sk, clips, s, 0, 0, kQuarterPhase),
                        anim::SampleLocalPose(sk, clips[0], 0.5f)),
              "2d: identity-at-vertex (idle bit-exact)");
        w = bs::EvaluateWeights2D(s, 0, 3 * kOne);
        check(w.w2 == kOne && w.w0 == 0 && w.w1 == 0, "2d: third vertex == {0, 0, kOne}");
        check(PoseEqual(bs::EvaluatePose2D(sk, clips, s, 0, 3 * kOne, kQuarterPhase),
                        anim::SampleLocalPose(sk, clips[2], 0.25f)),
              "2d: identity-at-vertex (run bit-exact, phase-synced 0.25 s)");

        // The exact centroid thirds — THE RESIDUAL CONVENTION: w0 == w1 == floor(kOne/3) == 21845,
        // w2 == kOne - 43690 == 21846; the sum is kOne EXACTLY.
        w = bs::EvaluateWeights2D(s, kOne, kOne);
        check(w.tri == 0 && !w.clamped && w.w0 == 21845 && w.w1 == 21845 && w.w2 == 21846,
              "2d: centroid == {21845, 21845, 21846} (residual on w2; sum == kOne)");
        check(w.w0 + w.w1 + w.w2 == kOne, "2d: centroid weights sum == kOne EXACTLY");

        // The centroid 3-way LEFT-FOLD pose (hand-derived, the pinned composition order):
        //   inner = blend(idle, walk, (21845<<16)/43690 == 32768 exactly) -> root z = 0.375,
        //   outer = inner + (run - inner) * 21846/65536:
        //   root z = 0.375 + 0.375*21846/65536 = 32768.25/65536 -> llround == 32768;
        //   lf.z   = 0.1875 + 0.1875*21846/65536 = 16384.125/65536 -> llround == 16384.
        const std::vector<anim::JointPose> cent =
            bs::EvaluatePose2D(sk, clips, s, kOne, kOne, kQuarterPhase);
        check(bs::QuantizeFx(cent[0].t.z) == 32768, "2d: centroid root z == 32768 (left-fold pinned)");
        check(bs::QuantizeFx(cent[1].t.z) == 16384, "2d: centroid foot z == 16384 (left-fold pinned)");
        std::printf("2d centroid pose digest: %016llx\n", (unsigned long long)bs::DigestPoseQ(cent));
        check(bs::DigestPoseQ(cent) == 0xc8b1bd2807db102dull, "2d: centroid pose digest PINNED");

        // Boundary-INCLUSIVE containment: a point ON the base edge is inside (not clamped).
        w = bs::EvaluateWeights2D(s, kOne + kOne / 2, 0);
        check(w.tri == 0 && !w.clamped && w.w0 == 32768 && w.w1 == 32768 && w.w2 == 0,
              "2d: ON-edge point is contained (boundary inclusive) == {32768, 32768, 0}");

        // Outside the hull, below the base: nearest-edge clamp — projection of (1, -1) onto the base
        // edge (0,0)-(3,0) at t == (1*3)<<16 / 9 == 21845 -> the 2-sample edge lerp {43691, 21845}.
        w = bs::EvaluateWeights2D(s, kOne, -kOne);
        check(w.clamped && w.tri == 0 && w.i0 == 0 && w.i1 == 1 && w.i2 == -1 && w.w0 == 43691 &&
                  w.w1 == 21845 && w.w2 == 0,
              "2d: outside-hull clamps to the nearest edge at the projected t (21845)");

        // Past BOTH adjacent endpoints: clamps to the VERTEX (t clamps to 0), and (-1,-1) is
        // EQUIDISTANT from edge 0's t==0 and edge 2's t==kOne -> the scan-order tie-break keeps the
        // FIRST (edge 0). Pose == idle EXACT (w0 == kOne -> the direct-sample path).
        w = bs::EvaluateWeights2D(s, -kOne, -kOne);
        check(w.clamped && w.i0 == 0 && w.i1 == 1 && w.w0 == kOne && w.w1 == 0,
              "2d: past-the-vertex clamps to the vertex; equidistant tie keeps the FIRST edge");
        check(PoseEqual(bs::EvaluatePose2D(sk, clips, s, -kOne, -kOne, kQuarterPhase),
                        anim::SampleLocalPose(sk, clips[0], 0.5f)),
              "2d: vertex-clamped pose == idle bit-exact");

        // Shared-edge tie-break: two triangles sharing edge 1-2; a point ON the shared edge is
        // contained by BOTH -> the FIRST in authored order wins.
        bs::BlendSpace2D s2 = MakeTestTri2D();
        s2.samples.push_back(bs::BlendSample2D{3 * kOne, 3 * kOne, 1});
        s2.tris.push_back(bs::BlendTri{1, 3, 2});
        w = bs::EvaluateWeights2D(s2, kOne + kOne / 2, kOne + kOne / 2);
        check(w.tri == 0 && !w.clamped && w.w0 == 0 && w.w1 == 32768 && w.w2 == 32768,
              "2d: shared-edge point -> the FIRST triangle wins (lower index)");

        check(bs::EvaluateWeights2D(bs::BlendSpace2D{}, 0, 0).tri == -1, "2d: empty space -> tri -1");
    }

    // ================= (d) SLEW: exact per-tick ramp + instant identity + phase wrap ================
    {
        // A step 0 -> 2.0 at rate kOne/32 (2048/tick): param(k) == min(2048*k, 131072), arrival at
        // tick 64 EXACTLY.
        fx p = 0;
        const fx rate = kOne / 32;
        p = bs::SlewParam(p, 2 * kOne, rate);
        check(p == 2048, "slew: tick 1 == 2048");
        p = bs::SlewParam(p, 2 * kOne, rate);
        check(p == 4096, "slew: tick 2 == 4096");
        for (int t = 2; t < 63; ++t) p = bs::SlewParam(p, 2 * kOne, rate);
        check(p == 129024, "slew: tick 63 == 129024 (one step short)");
        p = bs::SlewParam(p, 2 * kOne, rate);
        check(p == 131072, "slew: tick 64 arrives EXACTLY at the target");
        p = bs::SlewParam(p, 2 * kOne, rate);
        check(p == 131072, "slew: holding at the target is a no-op");
        // Negative direction + the sub-rate final step.
        p = bs::SlewParam(131072, 130000, rate);
        check(p == 130000, "slew: a sub-rate delta lands exactly on the target");
        p = bs::SlewParam(0, -2 * kOne, rate);
        check(p == -2048, "slew: negative direction ramps at -2048/tick");
        // rate == 0 -> disabled -> instant (the identity with an unslewed set).
        check(bs::SlewParam(0, 2 * kOne, 0) == 131072, "slew: rate 0 == instant");
        // 2D box slew: axes are INDEPENDENT — from (0,0) toward (2,-1) after 40 ticks: x is still
        // ramping (81920), y arrived at tick 32 (-65536).
        bs::BlendParam2 c{0, 0};
        for (int t = 0; t < 40; ++t) c = bs::SlewParam2(c, bs::BlendParam2{2 * kOne, -kOne}, rate);
        check(c.x == 81920 && c.y == -65536, "slew2: box axes independent (x mid-ramp, y arrived)");
        // Phase wrap (mod kOne).
        check(bs::AdvancePhase(65024, 1024) == 512, "phase: wraps mod kOne (65024+1024 -> 512)");
        check(bs::AdvancePhase(0, 0) == 0, "phase: zero-rate no-op");
    }

    // ================= (e) FSM adapter: the bound param through the QuantizeFx boundary =============
    {
        anim::StateMachine fsm;
        fsm.AddState(anim::AnimState{"locomotion", 0, true, 1.0f});
        const int speedIdx = fsm.AddParam("speed", 0.0f);

        bs::BlendDriver1D d;
        d.fsmParam = speedIdx;
        d.slewRate = kOne / 32;
        d.phaseRate = kOne / 64;
        fsm.SetParam("speed", 2.0f);   // QuantizeFx(2.0f) == 131072 exactly
        bs::TickBlendDriver1D(d, fsm);
        check(d.param == 2048 && d.phase == 1024, "driver: tick 1 {param 2048, phase 1024}");
        bs::TickBlendDriver1D(d, fsm);
        bs::TickBlendDriver1D(d, fsm);
        check(d.param == 6144 && d.phase == 3072, "driver: tick 3 {param 6144, phase 3072}");
        fsm.SetParam("speed", 0.0f);   // the FSM param drops -> the driver ramps DOWN
        bs::TickBlendDriver1D(d, fsm);
        check(d.param == 4096 && d.phase == 4096, "driver: ramps down after the param drop");

        // The driver pose == EvaluatePose1D at the driver cursor (the adapter adds no pose math).
        const bs::BlendSpace1D s = MakeSpace1D();
        check(bs::DigestPoseQ(bs::EvaluateBlendDriver1D(d, sk, clips, s)) ==
                  bs::DigestPoseQ(bs::EvaluatePose1D(sk, clips, s, d.param, d.phase)),
              "driver: EvaluateBlendDriver1D == EvaluatePose1D at (param, phase)");

        // Unbound driver holds its param; slewRate 0 snaps instantly.
        bs::BlendDriver1D u;
        u.fsmParam = -1;
        u.param = 777;
        bs::TickBlendDriver1D(u, fsm);
        check(u.param == 777, "driver: unbound param holds");
        bs::BlendDriver1D snap;
        snap.fsmParam = speedIdx;
        snap.slewRate = 0;
        fsm.SetParam("speed", 2.0f);
        bs::TickBlendDriver1D(snap, fsm);
        check(snap.param == 131072, "driver: slewRate 0 snaps to the target instantly");
    }

    // ================= (f) The shot scenario: two-run determinism + PINNED digests ==================
    {
        const bs::BlendShotRun run = bs::RunBlendShotScenario();
        const bs::BlendShotRun run2 = bs::RunBlendShotScenario();
        check(run.steps.size() == (size_t)bs::kShotSteps, "shot: 240 steps");
        check(run.pathDigest == run2.pathDigest && run.weightsDigest == run2.weightsDigest &&
                  run.poseDigest == run2.poseDigest && run.digest == run2.digest,
              "shot: two runs IDENTICAL");

        // step[0] hand-derived: one slew tick toward (0, 2) -> (0, 2048); inside tri 0 (idle/strafeL/
        // walk) ON the idle-walk edge: w == {63488, 0, 2048} (w0 == (kOne-2048), residual w2).
        const bs::BlendShotStep& s0 = run.steps[0];
        check(s0.x == 0 && s0.y == 2048 && s0.tri == 0 && s0.clamped == 0 && s0.w0 == 63488 &&
                  s0.w1 == 0 && s0.w2 == 2048,
              "shot: step[0] hand-derived {(0,2048), tri 0, w {63488,0,2048}}");

        // The last leg (target below the hull) really exercises the clamp.
        int clampedTicks = 0;
        for (const bs::BlendShotStep& st : run.steps) clampedTicks += st.clamped;
        check(clampedTicks > 0, "shot: the below-hull leg clamps (clamped ticks > 0)");
        check(run.steps.back().clamped == 1, "shot: the final tick is clamped");

        // PINNED digests — the cross-compiler currency (MSVC == clang; fixtures are exact binary
        // fractions, translation-only rotations, per the MM1 discipline).
        check(run.pathDigest == 0x956fe8b0abaff807ull, "shot: pathDigest PINNED");
        check(run.weightsDigest == 0x7731e08b97647960ull, "shot: weightsDigest PINNED");
        check(run.poseDigest == 0x538ba1d12d089749ull, "shot: poseDigest PINNED");
        std::printf("shot digests: path %016llx weights %016llx pose %016llx combined %016llx\n",
                    (unsigned long long)run.pathDigest, (unsigned long long)run.weightsDigest,
                    (unsigned long long)run.poseDigest, (unsigned long long)run.digest);

        // The shared raster: two renders byte-identical, the fixed 520x420 frame, path pixels drawn.
        std::vector<uint8_t> img1, img2;
        uint32_t w1 = 0, h1 = 0, w2 = 0, h2 = 0;
        bs::RenderBlendShot(run, img1, w1, h1);
        bs::RenderBlendShot(run, img2, w2, h2);
        check(w1 == 520 && h1 == 420, "render: 520x420");
        check(w1 == w2 && h1 == h2 && img1 == img2, "render: two runs byte-identical");
        size_t nonBg = 0;
        for (size_t p = 0; p < img1.size(); p += 4)
            if (!(img1[p] == 24 && img1[p + 1] == 18 && img1[p + 2] == 14)) ++nonBg;
        check(nonBg > 2000, "render: the diagram + bars actually drew");
    }

    if (g_fail == 0) {
        std::printf("blend_space_test: ALL PASS\n");
        return 0;
    }
    std::printf("blend_space_test: %d FAILURES\n", g_fail);
    return 1;
}
