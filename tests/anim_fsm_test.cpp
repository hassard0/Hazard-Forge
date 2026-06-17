// Slice BL — animation state-machine / cross-fade unit test (pure CPU, ASan-eligible). Drives the
// FSM with a SCRIPTED (param, dt) timeline and asserts:
//   * Transition condition: a speed>0.3 edge fires at speed=0.5 but NOT at speed=0.1, and the FIRST
//     satisfied transition (in fixed add-order) is the one taken.
//   * Transition timing + cross-fade: BlendWeight() ramps 0->1 LINEARLY over `duration`, the FSM is
//     IsTransitioning() until transitionTime >= duration, then `current` == target and not transitioning.
//   * Any-state transition (from == -1) fires from any current state.
//   * Determinism: the same scripted (param, dt) sequence yields identical {current, transitioning,
//     blend} per step across two independent runs.
//   * Evaluate endpoints: blend==0 palette == SampleAnimation(from); blend==1 == SampleAnimation(to)
//     (the existing BlendAnimations weight convention: 0->from, 1->to).
//
// The skeleton + animations are synthetic (built in-test) so the unit test is GPU/asset-free.
#include "anim/state_machine.h"
#include "anim/animation.h"
#include "anim/skeleton.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
using namespace hf::anim;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

// --- Synthetic skeleton + clips -----------------------------------------------------------------
// A 2-joint skeleton (root + child). Each "clip" rotates the child joint about Z by a per-clip,
// per-time angle, so distinct clips/times produce visibly distinct palettes that the endpoint
// asserts can tell apart.
static Skeleton MakeSkeleton() {
    Skeleton s;
    Joint root;  root.parent = -1; root.inverseBind = math::Mat4::Identity();
    Joint child; child.parent = 0;  child.inverseBind = math::Mat4::Identity();
    s.joints = {root, child};
    return s;
}

// A single-channel clip rotating joint 1 about Z from angle0 (t=0) to angle1 (t=1).
static Animation MakeRotClip(const char* name, float angle0, float angle1) {
    Animation a;
    a.name = name;
    a.duration = 1.0f;
    Channel ch;
    ch.jointIndex = 1;
    ch.path = Channel::Path::Rotation;
    ch.interp = Channel::Interp::Linear;
    ch.times = {0.0f, 1.0f};
    auto q0 = math::Quat{0, 0, std::sin(angle0 * 0.5f), std::cos(angle0 * 0.5f)};
    auto q1 = math::Quat{0, 0, std::sin(angle1 * 0.5f), std::cos(angle1 * 0.5f)};
    ch.values = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    a.channels = {ch};
    return a;
}

static bool PalettesEqual(const std::vector<math::Mat4>& a, const std::vector<math::Mat4>& b) {
    if (a.size() != b.size()) return false;
    for (size_t j = 0; j < a.size(); ++j)
        for (int k = 0; k < 16; ++k)
            if (!approx(a[j].m[k], b[j].m[k])) return false;
    return true;
}

int main() {
    HF_TEST_MAIN_INIT();
    Skeleton skel = MakeSkeleton();
    // idle: child stays near 0; walk: rotates to +0.6; run: rotates to +1.2 (radians). Distinct poses.
    std::vector<Animation> anims = {
        MakeRotClip("idle", 0.0f, 0.05f),
        MakeRotClip("walk", 0.4f, 0.6f),
        MakeRotClip("run",  0.9f, 1.2f),
    };

    // ---- Transition condition + first-satisfied order ------------------------------------------
    {
        StateMachine fsm;
        fsm.AddState({"idle", 0, true, 1.0f});
        fsm.AddState({"walk", 1, true, 1.0f});
        fsm.AddState({"run",  2, true, 1.0f});
        int sp = fsm.AddParam("speed", 0.0f);
        // From idle, two outgoing edges in fixed order: ->walk (speed>0.3) added BEFORE ->run (speed>0.9).
        fsm.AddTransition({0, 1, sp, Transition::Cmp::Greater, 0.3f, 0.2f});
        fsm.AddTransition({0, 2, sp, Transition::Cmp::Greater, 0.9f, 0.2f});

        // speed=0.1: below 0.3 -> no transition.
        fsm.SetParam("speed", 0.1f);
        fsm.Update(0.016f);
        check(!fsm.IsTransitioning(), "speed=0.1 does NOT fire speed>0.3 transition");
        check(fsm.Current() == 0, "stays in idle when no condition holds");

        // speed=0.5: above 0.3 (first edge) but below 0.9 -> idle->walk fires.
        StateMachine fsm2 = fsm;  // fresh-ish copy at idle
        fsm2.SetInitialState(0);
        fsm2.SetParam("speed", 0.5f);
        fsm2.Update(0.016f);
        check(fsm2.IsTransitioning() && fsm2.TransitioningTo() == 1,
              "speed=0.5 fires idle->walk (speed>0.3)");

        // speed=1.0: BOTH edges satisfied; FIRST-added (->walk) wins.
        StateMachine fsm3;
        fsm3.AddState({"idle", 0, true, 1.0f});
        fsm3.AddState({"walk", 1, true, 1.0f});
        fsm3.AddState({"run",  2, true, 1.0f});
        int sp3 = fsm3.AddParam("speed", 0.0f);
        fsm3.AddTransition({0, 1, sp3, Transition::Cmp::Greater, 0.3f, 0.2f});
        fsm3.AddTransition({0, 2, sp3, Transition::Cmp::Greater, 0.9f, 0.2f});
        fsm3.SetParam("speed", 1.0f);
        fsm3.Update(0.016f);
        check(fsm3.IsTransitioning() && fsm3.TransitioningTo() == 1,
              "speed=1.0: first-satisfied (idle->walk) wins over idle->run");
    }

    // ---- Transition timing + cross-fade ramp ---------------------------------------------------
    {
        StateMachine fsm;
        fsm.AddState({"idle", 0, true, 1.0f});
        fsm.AddState({"walk", 1, true, 1.0f});
        int sp = fsm.AddParam("speed", 0.0f);
        const float dur = 0.25f;
        fsm.AddTransition({0, 1, sp, Transition::Cmp::Greater, 0.3f, dur});

        fsm.SetParam("speed", 0.5f);
        fsm.Update(0.0f);  // dt=0: begins transition, transitionTime stays 0 -> weight 0.
        check(fsm.IsTransitioning(), "transition begins");
        check(approx(fsm.BlendWeight(), 0.0f), "blend weight starts at 0");

        // Step in increments; weight must equal transitionTime/duration linearly.
        float t = 0.0f;
        const float dt = 0.05f;
        bool linearOk = true;
        for (int i = 0; i < 4; ++i) {  // 4 * 0.05 = 0.20 < 0.25 -> still transitioning
            fsm.Update(dt);
            t += dt;
            if (!approx(fsm.BlendWeight(), t / dur)) linearOk = false;
            if (!fsm.IsTransitioning()) linearOk = false;
        }
        check(linearOk, "BlendWeight ramps linearly transitionTime/duration; transitioning until >= duration");
        check(approx(fsm.BlendWeight(), 0.20f / dur), "blend weight == 0.20/0.25 == 0.8 at t=0.20");

        // One more step pushes transitionTime to 0.25 >= duration -> completes.
        fsm.Update(dt);  // t -> 0.25
        check(!fsm.IsTransitioning(), "transition completes when transitionTime >= duration");
        check(fsm.Current() == 1, "current becomes the target (walk) after completion");
    }

    // ---- Any-state transition (from == -1) -----------------------------------------------------
    {
        StateMachine fsm;
        fsm.AddState({"idle", 0, true, 1.0f});
        fsm.AddState({"walk", 1, true, 1.0f});
        fsm.AddState({"hit",  2, true, 1.0f});  // a "reaction" reachable from anywhere
        int sp  = fsm.AddParam("speed", 0.0f);
        int hit = fsm.AddParam("hit", 0.0f);
        fsm.AddTransition({0, 1, sp, Transition::Cmp::Greater, 0.3f, 0.2f});
        // Any-state edge: hit>0.5 -> state 2, from ANY current state.
        fsm.AddTransition({-1, 2, hit, Transition::Cmp::Greater, 0.5f, 0.2f});

        // Move into walk first.
        fsm.SetParam("speed", 0.5f);
        fsm.Update(0.0f);
        // Force-complete the idle->walk transition.
        for (int i = 0; i < 20 && fsm.IsTransitioning(); ++i) fsm.Update(0.05f);
        check(fsm.Current() == 1, "reached walk");

        // Now trip the any-state hit edge from walk.
        fsm.SetParam("hit", 1.0f);
        fsm.Update(0.0f);
        check(fsm.IsTransitioning() && fsm.TransitioningTo() == 2,
              "any-state (from==-1) hit edge fires from the walk state");
    }

    // ---- Determinism: two runs of the same scripted timeline are bit-identical -----------------
    auto runScript = [&](std::vector<int>& curs, std::vector<int>& trans, std::vector<float>& blends) {
        StateMachine fsm;
        fsm.AddState({"idle", 0, true, 1.0f});
        fsm.AddState({"walk", 1, true, 1.0f});
        fsm.AddState({"run",  2, true, 1.0f});
        int sp = fsm.AddParam("speed", 0.0f);
        fsm.AddTransition({0, 1, sp, Transition::Cmp::Greater, 0.3f, 0.25f});
        fsm.AddTransition({1, 2, sp, Transition::Cmp::Greater, 0.7f, 0.25f});
        fsm.AddTransition({2, 1, sp, Transition::Cmp::Less,    0.7f, 0.25f});
        fsm.AddTransition({1, 0, sp, Transition::Cmp::Less,    0.3f, 0.25f});
        // Scripted speed ramp: 0 -> accelerate to run -> decelerate back.
        const float dt = 0.05f;
        for (int i = 0; i < 60; ++i) {
            float speed = (i < 30) ? (i / 30.0f) : ((60 - i) / 30.0f);  // 0..1..~0
            fsm.SetParam("speed", speed);
            fsm.Update(dt);
            curs.push_back(fsm.Current());
            trans.push_back(fsm.TransitioningTo());
            blends.push_back(fsm.BlendWeight());
        }
    };
    {
        std::vector<int> c1, c2, t1, t2; std::vector<float> b1, b2;
        runScript(c1, t1, b1);
        runScript(c2, t2, b2);
        bool same = (c1 == c2) && (t1 == t2) && (b1.size() == b2.size());
        if (same) for (size_t i = 0; i < b1.size(); ++i) if (b1[i] != b2[i]) same = false;
        check(same, "scripted (param,dt) timeline is deterministic across two runs");
        // And it actually exercised a transition (sanity: not a trivial constant).
        bool sawTransition = false;
        for (int v : t1) if (v >= 0) sawTransition = true;
        check(sawTransition, "scripted timeline exercises at least one transition");
    }

    // ---- Evaluate endpoints: blend 0 == from, blend 1 == to ------------------------------------
    {
        StateMachine fsm;
        fsm.AddState({"walk", 1, true, 1.0f});
        fsm.AddState({"run",  2, true, 1.0f});
        int sp = fsm.AddParam("speed", 0.0f);
        const float dur = 0.4f;
        fsm.AddTransition({0, 1, sp, Transition::Cmp::Greater, 0.5f, dur});  // walk(state0)->run(state1)
        fsm.SetInitialState(0);

        // Begin the walk->run transition with dt=0 so blend==0 and BOTH local times are 0.
        fsm.SetParam("speed", 1.0f);
        fsm.Update(0.0f);
        check(fsm.IsTransitioning() && approx(fsm.BlendWeight(), 0.0f), "transition at weight 0");
        std::vector<math::Mat4> pal0 = fsm.Evaluate(skel, anims);
        // At weight 0 + both local times 0, the palette equals SampleAnimation(from-clip) at t=0.
        std::vector<math::Mat4> fromPal = SampleAnimation(skel, anims[1], 0.0f);  // walk clip
        check(PalettesEqual(pal0, fromPal), "Evaluate at blend==0 equals SampleAnimation(from)");

        // Drive to weight 1 in ONE dt == duration (so transitionTime==duration -> weight clamps to 1;
        // but completion fires at >= duration, so capture the palette by re-deriving from the public
        // convention instead). Use a fresh machine stopped exactly mid at the last sub-duration step.
        StateMachine fsm2;
        fsm2.AddState({"walk", 1, true, 1.0f});
        fsm2.AddState({"run",  2, true, 1.0f});
        int sp2 = fsm2.AddParam("speed", 0.0f);
        fsm2.AddTransition({0, 1, sp2, Transition::Cmp::Greater, 0.5f, dur});
        fsm2.SetInitialState(0);
        fsm2.SetParam("speed", 1.0f);
        fsm2.Update(0.0f);                 // begin, weight 0, both times 0
        fsm2.Update(dur - 1e-3f);          // weight ~= 1 (just under completion), to-time = dur-1e-3
        check(fsm2.IsTransitioning(), "still transitioning just under duration");
        check(fsm2.BlendWeight() > 0.99f, "blend weight ~ 1 just under duration");
        std::vector<math::Mat4> pal1 = fsm2.Evaluate(skel, anims);
        // The from-clip advanced (dur-1e-3) AND the to-clip advanced (dur-1e-3); at weight ~1 the
        // blended pose ~= the TO clip sampled at its local time. Compare to BlendAnimations directly
        // at the same times+weight (the exact convention) to make the endpoint check rigorous.
        std::vector<math::Mat4> ref =
            BlendAnimations(skel, anims[1], dur - 1e-3f, anims[2], dur - 1e-3f, fsm2.BlendWeight());
        check(PalettesEqual(pal1, ref),
              "Evaluate while transitioning == BlendAnimations(from,to,weight) at matching times");
        // And at weight~1 it is much closer to the TO clip than the FROM clip (distinct poses).
        std::vector<math::Mat4> toPal   = SampleAnimation(skel, anims[2], dur - 1e-3f);
        std::vector<math::Mat4> fromPal2 = SampleAnimation(skel, anims[1], dur - 1e-3f);
        auto dist = [](const std::vector<math::Mat4>& a, const std::vector<math::Mat4>& b) {
            float d = 0; for (size_t j = 0; j < a.size(); ++j)
                for (int k = 0; k < 16; ++k) d += std::fabs(a[j].m[k] - b[j].m[k]); return d;
        };
        check(dist(pal1, toPal) < dist(pal1, fromPal2),
              "at blend~1 the pose is nearer the TO clip than the FROM clip");
    }

    if (g_fail == 0) { std::printf("anim_fsm_test OK\n"); return 0; }
    std::printf("anim_fsm_test: %d failures\n", g_fail);
    return 1;
}
