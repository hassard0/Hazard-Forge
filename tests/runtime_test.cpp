// Slice AA unit test: the backend-agnostic interactive-runtime core — Camera (yaw/pitch -> basis +
// view), FlyCameraController (synthetic InputState + dt -> expected position/orientation delta), and
// the FixedTimestep accumulator (synthetic dt sequence -> correct fixed-step counts + remainder).
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "runtime/camera.h"
#include "runtime/clock.h"
#include "runtime/fly_camera_controller.h"
#include "runtime/input_state.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
using runtime::Key;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }
static bool approxV(const math::Vec3& a, const math::Vec3& b, float eps = 1e-4f) {
    return approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps);
}

int main() {
    HF_TEST_MAIN_INIT();
    // ============================================================ Camera basis from yaw/pitch.
    {
        runtime::Camera cam;
        // yaw=0,pitch=0 -> forward (0,0,-1), right (1,0,0), up (0,1,0).
        check(approxV(cam.Forward(), {0, 0, -1}), "cam fwd at (0,0) == -Z");
        check(approxV(cam.Right(),   {1, 0,  0}), "cam right at (0,0) == +X");
        check(approxV(cam.Up(),      {0, 1,  0}), "cam up at (0,0) == +Y");

        // yaw = +90deg -> forward points to +X (rotation about +Y, RH).
        cam.yaw = 1.5707963f;
        check(approxV(cam.Forward(), {1, 0, 0}, 1e-3f), "cam fwd at yaw=+90 == +X");
        check(approxV(cam.Right(),   {0, 0, 1}, 1e-3f), "cam right at yaw=+90 == +Z");

        // pitch = +45deg, yaw=0 -> forward up-and-back: (0, sin45, -cos45).
        cam.yaw = 0.0f; cam.pitch = 0.7853982f;
        float s = 0.70710678f;
        check(approxV(cam.Forward(), {0, s, -s}, 1e-3f), "cam fwd at pitch=+45 == (0,+,-)");
        // Right stays horizontal regardless of pitch.
        check(approx(cam.Right().y, 0.0f), "cam right stays horizontal under pitch");

        // Pitch clamp: AddYawPitch past vertical clamps to < pi/2.
        cam.pitch = 0.0f;
        cam.AddYawPitch(0.0f, 100.0f);
        check(cam.pitch < 1.5707963f && cam.pitch > 1.5f, "pitch clamped just under +90");
        cam.AddYawPitch(0.0f, -200.0f);
        check(cam.pitch > -1.5707963f && cam.pitch < -1.5f, "pitch clamped just above -90");
    }

    // ============================================================ Camera View() matches LookAt.
    {
        runtime::Camera cam;
        cam.position = {2.0f, 3.0f, 5.0f};
        cam.yaw = 0.4f; cam.pitch = -0.2f;
        math::Mat4 v = cam.View();
        math::Mat4 ref = math::Mat4::LookAt(cam.position, cam.position + cam.Forward(), {0, 1, 0});
        bool same = true;
        for (int i = 0; i < 16; ++i) if (!approx(v.m[i], ref.m[i])) same = false;
        check(same, "Camera::View == LookAt(pos, pos+fwd, up)");

        // ViewProj == Proj * View.
        math::Mat4 vp = cam.ViewProj();
        math::Mat4 vpRef = cam.Proj() * cam.View();
        same = true;
        for (int i = 0; i < 16; ++i) if (!approx(vp.m[i], vpRef.m[i])) same = false;
        check(same, "Camera::ViewProj == Proj * View");

        // Basis helper agrees with the accessors.
        runtime::CameraBasis b = cam.Basis();
        check(approxV(b.forward, cam.Forward()), "Basis.forward == Forward()");
        check(approxV(b.right, cam.Right()),     "Basis.right == Right()");
        check(approxV(b.up, cam.Up()),           "Basis.up == Up()");
        check(approxV(b.position, cam.position), "Basis.position == position");
        check(approx(b.tanHalfFovY, std::tan(0.5f * cam.fovY)), "Basis.tanHalfFovY correct");
    }

    // ============================================================ FlyCameraController translation.
    {
        runtime::Camera cam;          // at origin, looking down -Z
        runtime::FlyCameraController fly;
        fly.moveSpeed = 10.0f;
        float dt = 0.1f;              // expect 10 * 0.1 = 1.0 unit of travel

        // W moves forward (down -Z).
        runtime::InputState in;
        in.keyDown[(int)Key::W] = true;
        fly.Update(cam, in, dt);
        check(approxV(cam.position, {0, 0, -1.0f}, 1e-3f), "W moves 1 unit forward (-Z)");

        // D moves right (+X) from a fresh camera.
        cam = runtime::Camera{};
        runtime::FlyCameraController fly2; fly2.moveSpeed = 10.0f;
        runtime::InputState inD; inD.keyDown[(int)Key::D] = true;
        fly2.Update(cam, inD, dt);
        check(approxV(cam.position, {1.0f, 0, 0}, 1e-3f), "D moves 1 unit right (+X)");

        // Space moves up (+Y, world).
        cam = runtime::Camera{};
        runtime::FlyCameraController fly3; fly3.moveSpeed = 10.0f;
        runtime::InputState inUp; inUp.keyDown[(int)Key::Space] = true;
        fly3.Update(cam, inUp, dt);
        check(approxV(cam.position, {0, 1.0f, 0}, 1e-3f), "Space moves 1 unit up (+Y)");

        // Diagonal (W+D) is normalized: total travel == speed*dt, not sqrt(2)*that.
        cam = runtime::Camera{};
        runtime::FlyCameraController fly4; fly4.moveSpeed = 10.0f;
        runtime::InputState inWD; inWD.keyDown[(int)Key::W] = true; inWD.keyDown[(int)Key::D] = true;
        fly4.Update(cam, inWD, dt);
        check(approx(math::length(cam.position), 1.0f, 1e-3f), "diagonal move normalized to speed*dt");

        // Shift sprints (x4).
        cam = runtime::Camera{};
        runtime::FlyCameraController fly5; fly5.moveSpeed = 10.0f; fly5.sprintMultiplier = 4.0f;
        runtime::InputState inSprint;
        inSprint.keyDown[(int)Key::W] = true; inSprint.keyDown[(int)Key::Shift] = true;
        fly5.Update(cam, inSprint, dt);
        check(approxV(cam.position, {0, 0, -4.0f}, 1e-3f), "Shift sprints x4");
    }

    // ============================================================ FlyCameraController mouse-look.
    {
        runtime::Camera cam;
        runtime::FlyCameraController fly;
        // First relative-mode frame swallows the delta (transition jump guard).
        runtime::InputState in; in.relativeMouse = true; in.mouseDx = 100.0f; in.mouseDy = 50.0f;
        fly.Update(cam, in, 0.016f);
        check(approx(cam.yaw, 0.0f) && approx(cam.pitch, 0.0f), "first relative frame swallows delta");
        // Second frame applies it: yaw += dx*sens, pitch -= dy*sens (invert screen-y).
        fly.Update(cam, in, 0.016f);
        check(approx(cam.yaw, 100.0f * fly.lookSensitivity), "mouseDx -> yaw");
        check(approx(cam.pitch, -50.0f * fly.lookSensitivity), "mouseDy -> -pitch (look up)");

        // Wheel adjusts move speed and is clamped.
        runtime::FlyCameraController fly2; fly2.moveSpeed = 10.0f;
        runtime::Camera cam2;
        runtime::InputState wheelUp; wheelUp.wheel = 3.0f;
        fly2.Update(cam2, wheelUp, 0.016f);
        check(fly2.moveSpeed > 10.0f, "wheel up increases move speed");
    }

    // ============================================================ FixedTimestep accumulator.
    {
        runtime::FixedTimestep ts(1.0f / 120.0f, 8);
        // 0.05s at 1/120 -> 6 steps (6/120 = 0.05 exactly), remainder ~0.
        int steps = ts.Tick(0.05f);
        check(steps == 6, "0.05s at 1/120 -> 6 fixed steps");
        check(ts.Accumulator() < 1.0f / 120.0f, "remainder < step after tick");
        check(ts.Accumulator() < 1e-4f, "0.05s leaves ~0 remainder");

        // A dt that doesn't divide evenly leaves a remainder that carries to the next tick.
        runtime::FixedTimestep ts2(1.0f / 100.0f, 100);  // 10ms step
        int s1 = ts2.Tick(0.025f);   // 2 steps (0.02), remainder 0.005
        check(s1 == 2, "0.025s at 1/100 -> 2 steps");
        check(approx(ts2.Accumulator(), 0.005f, 1e-5f), "remainder 0.005 carried");
        int s2 = ts2.Tick(0.006f);   // 0.005 + 0.006 = 0.011 -> 1 step, remainder 0.001
        check(s2 == 1, "carried remainder produces the next step");
        check(approx(ts2.Accumulator(), 0.001f, 1e-5f), "remainder 0.001 after carry");

        // Sub-step dt accumulates without emitting a step.
        runtime::FixedTimestep ts3(1.0f / 60.0f);
        check(ts3.Tick(0.001f) == 0, "tiny dt yields 0 steps");
        check(ts3.Tick(0.001f) == 0, "still accumulating, 0 steps");

        // Spiral-of-death cap: a huge dt is capped to maxStepsPerTick and the backlog dropped.
        runtime::FixedTimestep ts4(1.0f / 120.0f, 4);
        int big = ts4.Tick(10.0f);   // would be 1200 steps; capped at 4
        check(big == 4, "huge dt capped at maxStepsPerTick");
        check(ts4.Accumulator() < 1.0f / 120.0f, "backlog dropped after cap (no spiral)");

        // Negative dt ignored.
        runtime::FixedTimestep ts5;
        check(ts5.Tick(-1.0f) == 0, "negative dt ignored");
    }

    if (g_fail == 0) std::printf("runtime_test: all checks passed\n");
    else std::printf("runtime_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
