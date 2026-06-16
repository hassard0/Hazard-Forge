// Slice CN — per-object + camera motion blur. Pure CPU math: the screen-space velocity (cur-prev
// pixel delta from two view-proj matrices), the length clamp (preserve direction), and the depth-aware
// gather tap weight. No device, ASan-eligible (links hf_core). Mirrors the math the --motionblur-shot
// showcase and motion_blur.frag use (engine/render/motion_blur.h).
//
// Properties pinned (per the spec):
//   * Zero velocity: prevViewProj == curViewProj -> ScreenVelocity == (0,0) for any point; and the
//     gather collapses to the center (TapWeight center==1, every off-center tap==0) so the normalized
//     sum equals the center color (the pass-through equivalence proof, in miniature).
//   * Nonzero velocity DIRECTION matches the motion and MAGNITUDE grows with the motion / dt.
//   * Clamp: a huge motion -> ClampVelocity caps the length at maxBlurPx, preserving direction.
//   * TapWeight depth-aware gather: a tap within the extent contributes; beyond it ~0; a nearer moving
//     tap streaks over a farther center; a farther (background) tap does NOT smear a nearer center.
//   * Determinism: same inputs -> identical velocity + weights (pure functions, no RNG/time).
#include "render/motion_blur.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>

namespace mb = hf::render::motionblur;
using hf::math::Mat4;
using hf::math::Vec2;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool finite2(const Vec2& v) { return std::isfinite(v.x) && std::isfinite(v.y); }
static float len2(const Vec2& v) { return std::sqrt(v.x * v.x + v.y * v.y); }

int main() {
    const int W = 1280, H = 720;
    const float aspect = (float)W / (float)H;
    const float fovY = 1.04719755f;  // 60 deg

    // A camera looking down -Z, and a "previous" camera shifted right (a fixed pan). The world point
    // sits in front of both. prevViewProj = proj * prevView; curViewProj = proj * curView.
    Mat4 proj = Mat4::Perspective(fovY, aspect, 0.1f, 100.0f);
    // Both cameras look straight down -Z (parallel forward), the PREV camera shifted +x by a fixed
    // amount (a fixed lateral pan). A world point in front then appears to slide -x between frames.
    Mat4 curView  = Mat4::LookAt(Vec3{0.0f, 0.0f, 0.0f},  Vec3{0.0f, 0.0f, -6.0f}, Vec3{0, 1, 0});
    Mat4 prevView = Mat4::LookAt(Vec3{0.3f, 0.0f, 0.0f},  Vec3{0.3f, 0.0f, -6.0f}, Vec3{0, 1, 0});
    Mat4 curVP  = proj * curView;
    Mat4 prevVP = proj * prevView;

    const Vec3 worldPt{0.0f, 0.0f, -6.0f};   // in front of the camera, off the prev camera's axis

    // ---- Zero velocity: prevViewProj == curViewProj -> exactly (0,0) for any point. ----
    {
        Vec2 v = mb::ScreenVelocity(worldPt, curVP, curVP, W, H);
        check(v.x == 0.0f && v.y == 0.0f, "zero velocity when prevViewProj == curViewProj");
        // A few scattered points all read exactly zero.
        Vec2 a = mb::ScreenVelocity(Vec3{2.0f, -1.0f, -10.0f}, curVP, curVP, W, H);
        Vec2 b = mb::ScreenVelocity(Vec3{-3.0f, 4.0f, -20.0f}, prevVP, prevVP, W, H);
        check(a.x == 0.0f && a.y == 0.0f && b.x == 0.0f && b.y == 0.0f,
              "zero velocity for arbitrary points under an identical projection pair");
    }

    // ---- Zero-velocity gather collapses to the center (the pass-through proof, in miniature). ----
    {
        const float velLen = 0.0f;
        // Center tap: dist 0, same depth -> weight 1.
        float wc = mb::TapWeight(6.0f, 6.0f, 0.0f, velLen);
        check(wc == 1.0f, "zero-vel: center tap weight is exactly 1");
        // Any off-center tap is outside the (zero) extent -> weight 0.
        float wo = mb::TapWeight(6.0f, 6.0f, 3.0f, velLen);
        check(wo == 0.0f, "zero-vel: off-center tap weight is exactly 0");
        // So a normalized gather (sum w*col / sum w) over {center, others} == the center color.
        float centerCol = 0.7f, otherCol = 0.1f;
        float sum = wc * centerCol + wo * otherCol;
        float wsum = wc + wo;
        check(wsum == 1.0f && sum == centerCol, "zero-vel gather returns the center color exactly");
    }

    // ---- Nonzero velocity: direction matches the motion, magnitude grows with it. ----
    {
        Vec2 v = mb::ScreenVelocity(worldPt, prevVP, curVP, W, H);
        check(finite2(v), "velocity is finite for an in-front point");
        check(len2(v) > 1.0f, "a camera pan produces a non-trivial (>1px) screen velocity");
        // PREV camera sat at +x looking -Z, so the point (at x=0) was LEFT of center in the prev frame
        // and is centered now: it slid RIGHTWARD on screen -> velocity.x is positive and dominates.
        check(v.x > 0.0f, "lateral camera pan -> the matching (positive-x) screen velocity");
        check(std::fabs(v.x) > std::fabs(v.y), "the pan velocity is dominantly horizontal");

        // Magnitude grows with the motion: a larger camera shift -> a larger velocity.
        Mat4 prevView2 = Mat4::LookAt(Vec3{0.6f, 0.0f, 0.0f}, Vec3{0.6f, 0.0f, -6.0f}, Vec3{0, 1, 0});
        Vec2 v2 = mb::ScreenVelocity(worldPt, proj * prevView2, curVP, W, H);
        check(len2(v2) > len2(v), "a larger camera motion -> a larger screen velocity");
    }

    // ---- Clamp: a huge motion -> capped at maxBlurPx, direction preserved. ----
    {
        const float maxBlurPx = 24.0f;
        Vec2 big{200.0f, -150.0f};
        Vec2 c = mb::ClampVelocity(big, maxBlurPx);
        check(std::fabs(len2(c) - maxBlurPx) < 1e-3f, "huge velocity clamps to exactly maxBlurPx");
        // Direction preserved: normalized big == normalized clamped.
        float lb = len2(big), lc = len2(c);
        check(std::fabs(big.x / lb - c.x / lc) < 1e-4f &&
              std::fabs(big.y / lb - c.y / lc) < 1e-4f, "clamp preserves the velocity direction");
        // A within-budget velocity passes through unchanged.
        Vec2 small{5.0f, 3.0f};
        Vec2 cs = mb::ClampVelocity(small, maxBlurPx);
        check(cs.x == small.x && cs.y == small.y, "within-budget velocity is unchanged by the clamp");
        // A zero velocity stays exactly zero (preserves the pass-through proof).
        Vec2 cz = mb::ClampVelocity(Vec2{0.0f, 0.0f}, maxBlurPx);
        check(cz.x == 0.0f && cz.y == 0.0f, "zero velocity stays exactly zero through the clamp");
    }

    // ---- TapWeight depth-aware gather. ----
    {
        const float velLen = 20.0f;
        const float centerDepth = 8.0f;
        // A tap within the extent at the same depth contributes (weight 1).
        check(mb::TapWeight(centerDepth, centerDepth, 5.0f, velLen) == 1.0f,
              "an in-extent co-planar tap contributes (weight 1)");
        // A tap BEYOND the extent contributes ~0.
        check(mb::TapWeight(centerDepth, centerDepth, 50.0f, velLen) == 0.0f,
              "a tap beyond the blur extent contributes 0");
        // A NEARER (foreground) moving tap streaks OVER a farther center -> contributes.
        check(mb::TapWeight(3.0f /*near*/, centerDepth /*far center*/, 6.0f, velLen) == 1.0f,
              "a nearer (foreground) tap streaks over a farther center (contributes)");
        // A FARTHER (background) tap does NOT smear a nearer center -> ~0.
        check(mb::TapWeight(30.0f /*far*/, centerDepth /*near center*/, 6.0f, velLen) == 0.0f,
              "a farther (background) tap does not smear a nearer center (0)");
        // The center tap (dist 0) always counts itself even at the extent boundary.
        check(mb::TapWeight(centerDepth, centerDepth, 0.0f, velLen) == 1.0f,
              "the center tap (dist 0) always contributes");
    }

    // ---- Determinism: same inputs -> identical velocity + weights. ----
    {
        Vec2 a = mb::ScreenVelocity(worldPt, prevVP, curVP, W, H);
        Vec2 b = mb::ScreenVelocity(worldPt, prevVP, curVP, W, H);
        check(a.x == b.x && a.y == b.y, "ScreenVelocity is deterministic (identical bits)");
        float wa = mb::TapWeight(4.0f, 7.0f, 3.5f, 12.0f);
        float wb = mb::TapWeight(4.0f, 7.0f, 3.5f, 12.0f);
        check(wa == wb, "TapWeight is deterministic");
        Vec2 ca = mb::ClampVelocity(Vec2{40.0f, 9.0f}, 15.0f);
        Vec2 cb = mb::ClampVelocity(Vec2{40.0f, 9.0f}, 15.0f);
        check(ca.x == cb.x && ca.y == cb.y, "ClampVelocity is deterministic");
    }

    if (g_fail == 0) { std::printf("motion_blur_test: all checks passed\n"); return 0; }
    std::printf("motion_blur_test: %d FAILURES\n", g_fail);
    return 1;
}
