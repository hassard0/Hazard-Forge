// Slice AM — live-editor logic unit test (hf_core, ASan-eligible).
//
// Covers the HEADLESS, deterministic logic behind the live --fly editor (the live mouse interaction
// itself is manual-only on the Vulkan window):
//   (a) cursor PIXEL -> NDC -> ScreenRayThroughCamera passes through a known world point;
//   (b) a click ray at a known screen point picks the expected entity among a couple of world AABBs;
//   (c) ApplyDrag along an axis (prevRay/curRay) moves the transform by the expected delta;
//   (d) FileWatcher.Poll reports a path as changed after its (injected) mtime increases, not before.
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "editor/gizmo.h"
#include "editor/picking.h"
#include "math/math.h"
#include "runtime/camera.h"
#include "runtime/hot_reload.h"
#include "scene/transform.h"

using namespace hf;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-3f) { return std::fabs(a - b) < eps; }

int main() {
    // ---- (a) cursor PIXEL -> NDC -> ray hits a known world point. -----------------------------
    // Round-trips the px<->NDC mapping the live loop uses: project a world point to NDC, convert that
    // NDC to a PIXEL (inverse of PixelToNdc), feed the pixel back through PixelToNdc + the ray cast,
    // and assert the ray still passes through the world point. Uses a realistic framebuffer size.
    {
        const float W = 1280.0f, H = 720.0f;
        runtime::Camera cam;
        cam.position = {1.0f, 2.0f, 8.0f};
        cam.yaw = 0.1f;
        cam.SetPitch(-0.15f);
        cam.aspect = W / H;

        Vec3 target{0.5f, 1.0f, 0.0f};
        float w = 0.0f;
        Vec3 ndc = math::MulPointDivide(cam.ViewProj(), target, w);
        check(w > 0.0f, "target in front of camera (clip w>0)");
        check(ndc.x > -1.0f && ndc.x < 1.0f && ndc.y > -1.0f && ndc.y < 1.0f, "target inside frustum");

        // Inverse of editor::PixelToNdc (which flips Y): ndc -> pixel.
        float px = (ndc.x * 0.5f + 0.5f) * W;
        float py = (-ndc.y * 0.5f + 0.5f) * H;   // undo the Y flip

        editor::Ndc back = editor::PixelToNdc(px, py, W, H);
        check(approx(back.x, ndc.x, 1e-4f) && approx(back.y, ndc.y, 1e-4f),
              "PixelToNdc round-trips NDC<->pixel");

        math::Ray ray = editor::ScreenRayThroughCamera(cam, back.x, back.y);
        Vec3 oc = target - ray.origin;
        float tAlong = math::dot(oc, ray.dir);
        Vec3 closest = ray.origin + ray.dir * tAlong;
        float dist = math::length(closest - target);
        check(tAlong > 0.0f, "target along +ray");
        check(dist < 0.02f, "cursor px -> NDC -> ray passes through the known world point");

        // A click at the exact framebuffer center maps to the camera forward.
        editor::Ndc center = editor::PixelToNdc(W * 0.5f, H * 0.5f, W, H);
        check(approx(center.x, 0.0f) && approx(center.y, 0.0f), "center pixel -> NDC origin");
        math::Ray centerRay = editor::ScreenRayThroughCamera(cam, center.x, center.y);
        check(math::dot(centerRay.dir, cam.Forward()) > 0.999f, "center cursor ray == camera forward");
    }

    // ---- (b) a click ray picks the expected entity among a couple of world AABBs. --------------
    {
        const float W = 1280.0f, H = 720.0f;
        runtime::Camera cam;
        cam.position = {0.0f, 1.5f, 9.0f};
        cam.yaw = 0.0f;
        cam.SetPitch(-0.1f);
        cam.aspect = W / H;

        // Two well-separated boxes. We aim the cursor at box 1's center.
        std::vector<editor::PickAabb> objs = {
            {{{-3.5f, 0.0f, -0.5f}, {-2.5f, 1.0f, 0.5f}}},   // 0: left cube around (-3,0.5,0)
            {{{ 2.0f, 0.5f, -0.5f}, { 3.0f, 1.5f, 0.5f}}},   // 1: right cube around (2.5,1,0)
        };
        Vec3 c1{2.5f, 1.0f, 0.0f};   // center of box 1
        float wclip = 0.0f;
        Vec3 ndc = math::MulPointDivide(cam.ViewProj(), c1, wclip);
        // NDC -> pixel (inverse map), then the live px->NDC->ray path.
        float px = (ndc.x * 0.5f + 0.5f) * W;
        float py = (-ndc.y * 0.5f + 0.5f) * H;
        editor::Ndc n = editor::PixelToNdc(px, py, W, H);
        math::Ray ray = editor::ScreenRayThroughCamera(cam, n.x, n.y);
        editor::PickResult r = editor::PickNearest(ray, objs);
        check(r.index == 1, "click ray at box-1 center picks entity 1");

        // Aim at box 0 instead.
        Vec3 c0{-3.0f, 0.5f, 0.0f};
        ndc = math::MulPointDivide(cam.ViewProj(), c0, wclip);
        px = (ndc.x * 0.5f + 0.5f) * W;
        py = (-ndc.y * 0.5f + 0.5f) * H;
        n = editor::PixelToNdc(px, py, W, H);
        ray = editor::ScreenRayThroughCamera(cam, n.x, n.y);
        r = editor::PickNearest(ray, objs);
        check(r.index == 0, "click ray at box-0 center picks entity 0");

        // A click far off in the corner misses both.
        editor::Ndc corner = editor::PixelToNdc(2.0f, 2.0f, W, H);   // top-left pixel
        math::Ray missRay = editor::ScreenRayThroughCamera(cam, corner.x, corner.y);
        editor::PickResult m = editor::PickNearest(missRay, objs);
        check(m.index < 0, "click ray into empty corner misses every entity");
    }

    // ---- (c) drag along an axis via ApplyDrag moves the transform by the expected delta. -------
    // Mirrors the per-frame prevRay/curRay the live loop feeds: two parallel downward rays offset
    // along +X by 2 -> the selected transform's X moves +2, other axes unchanged.
    {
        scene::Transform xf;
        xf.position = {0, 0, 0};
        math::Ray prev{{1.0f, 5.0f, 0.0f}, {0, -1, 0}};
        math::Ray cur {{3.0f, 5.0f, 0.0f}, {0, -1, 0}};
        scene::Transform moved =
            editor::ApplyDrag(xf, editor::GizmoMode::Translate, editor::kAxisX, prev, cur);
        check(approx(moved.position.x, 2.0f, 1e-2f), "drag along X moves transform +2");
        check(approx(moved.position.y, 0.0f) && approx(moved.position.z, 0.0f),
              "drag along X leaves Y/Z fixed");

        // First frame of a drag: prevRay == curRay -> no movement (no jump on grab).
        scene::Transform first =
            editor::ApplyDrag(xf, editor::GizmoMode::Translate, editor::kAxisX, cur, cur);
        check(approx(first.position.x, 0.0f, 1e-3f), "drag with prevRay==curRay is a no-op (grab frame)");
    }

    // ---- (d) FileWatcher.Poll reports a change only after the (injected) mtime increases. ------
    {
        // In-memory fake filesystem: path -> mtime (negative = missing).
        std::map<std::string, int64_t> mtimes;
        mtimes["scene.json"] = 100;
        mtimes["shader.spv"] = 200;
        runtime::StatFn fake = [&](const std::string& p) -> int64_t {
            auto it = mtimes.find(p);
            return it == mtimes.end() ? -1 : it->second;
        };

        runtime::FileWatcher fw(fake);
        fw.Watch("scene.json");
        fw.Watch("shader.spv");
        check(fw.Count() == 2, "FileWatcher tracks 2 paths");

        // First poll after watching: nothing changed (baseline captured at Watch time).
        check(fw.Poll().empty(), "first Poll reports no changes");

        // Bump only scene.json -> exactly that path is reported, once.
        mtimes["scene.json"] = 101;
        auto c1 = fw.Poll();
        check(c1.size() == 1 && c1[0] == "scene.json", "Poll reports scene.json after its mtime bump");
        check(fw.Poll().empty(), "the same change does not re-fire on the next Poll");

        // Bump the other path independently.
        mtimes["shader.spv"] = 250;
        auto c2 = fw.Poll();
        check(c2.size() == 1 && c2[0] == "shader.spv", "Poll reports shader.spv independently");

        // Bump BOTH between polls -> both reported.
        mtimes["scene.json"] = 102;
        mtimes["shader.spv"] = 251;
        auto c3 = fw.Poll();
        check(c3.size() == 2, "Poll reports both paths when both change");

        // A newly-APPEARING file (missing at baseline -> present) counts as changed.
        runtime::FileWatcher fw2(fake);
        fw2.Watch("new.json");                 // missing -> baseline negative
        check(fw2.Poll().empty(), "missing watched file: first Poll quiet");
        mtimes["new.json"] = 5;                // it appears
        auto a = fw2.Poll();
        check(a.size() == 1 && a[0] == "new.json", "a newly-appearing watched file counts as changed");
    }

    if (g_fail == 0) std::printf("live_editor_test: ALL PASS\n");
    else std::printf("live_editor_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
