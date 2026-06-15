// Slice AB — editor selection/gizmo/picking + play-state unit test (hf_core, ASan-eligible).
//
// Covers: Mat4::Inverse (M*inv==I, identity + TRS); ScreenRayThroughCamera hits a known world point;
// PickNearest picks the nearer AABB + misses; PickGizmoAxis selects the right axis (translate/scale/
// rotate); ApplyDrag translate/rotate/scale deltas; PlayState paused/step/playing; and a scripted
// select->translate->DumpScene round-trip that asserts the saved JSON reflects the moved transform.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ecs/ecs.h"
#include "editor/gizmo.h"
#include "editor/picking.h"
#include "math/math.h"
#include "runtime/camera.h"
#include "runtime/play_state.h"
#include "scene/components.h"
#include "scene/mesh.h"
#include "scene/scene_io.h"

using namespace hf;
using hf::math::Mat4;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-3f) { return std::fabs(a - b) < eps; }

int main() {
    // ---- Mat4::Inverse: M * M.Inverse() == I. -------------------------------------------------
    {
        Mat4 id = Mat4::Identity();
        Mat4 idInv = id.Inverse();
        for (int k = 0; k < 16; ++k) check(approx(idInv.m[k], id.m[k]), "inv(I)==I");

        // Translation inverse.
        Mat4 t = Mat4::Translate({3.0f, -2.0f, 5.0f});
        Mat4 ti = t.Inverse();
        Mat4 prod = t * ti;
        for (int k = 0; k < 16; ++k) check(approx(prod.m[k], id.m[k]), "T*T^-1==I");

        // A full TRS matrix (translate * rotate * scale).
        Mat4 trs = Mat4::Translate({1.5f, 4.0f, -3.0f}) *
                   Mat4::RotateY(0.7f) * Mat4::RotateX(-0.4f) *
                   Mat4::Scale({2.0f, 0.5f, 1.3f});
        Mat4 trsInv = trs.Inverse();
        Mat4 p2 = trs * trsInv;
        for (int k = 0; k < 16; ++k) check(approx(p2.m[k], id.m[k], 2e-3f), "TRS*TRS^-1==I");
        Mat4 p3 = trsInv * trs;
        for (int k = 0; k < 16; ++k) check(approx(p3.m[k], id.m[k], 2e-3f), "TRS^-1*TRS==I");
    }

    // ---- ScreenRayThroughCamera hits a known world point. -------------------------------------
    {
        runtime::Camera cam;
        cam.position = {0.0f, 2.0f, 8.0f};
        cam.yaw = 0.0f;
        cam.SetPitch(-0.1f);
        cam.aspect = 16.0f / 9.0f;

        // Pick a world point in front of the camera, project it to NDC with VP, then cast a ray back
        // through that NDC and assert the ray passes very close to the world point.
        Vec3 target{0.5f, 1.0f, 0.0f};
        float w = 0.0f;
        Vec3 ndc = math::MulPointDivide(cam.ViewProj(), target, w);
        check(w > 0.0f, "target is in front of camera (clip w>0)");
        check(ndc.x > -1.0f && ndc.x < 1.0f && ndc.y > -1.0f && ndc.y < 1.0f, "target inside frustum");

        math::Ray ray = editor::ScreenRayThroughCamera(cam, ndc.x, ndc.y);
        // Closest approach of the ray to `target` should be ~0.
        Vec3 oc = target - ray.origin;
        float tAlong = math::dot(oc, ray.dir);
        Vec3 closest = ray.origin + ray.dir * tAlong;
        float dist = math::length(closest - target);
        check(tAlong > 0.0f, "target is along +ray");
        check(dist < 0.02f, "screen ray passes through the known world point");

        // Center of screen (0,0) ray direction should roughly match the camera forward.
        math::Ray center = editor::ScreenRayThroughCamera(cam, 0.0f, 0.0f);
        Vec3 fwd = cam.Forward();
        check(math::dot(center.dir, fwd) > 0.999f, "center screen ray == camera forward");
    }

    // ---- PickNearest: nearer of two AABBs, and a miss. ----------------------------------------
    {
        // Ray from origin down -Z. Box A at z=-3, Box B at z=-8 — A is nearer.
        math::Ray ray{{0, 0, 0}, {0, 0, -1}};
        std::vector<editor::PickAabb> objs = {
            {{{-1, -1, -9}, {1, 1, -7}}},   // B (far)
            {{{-1, -1, -4}, {1, 1, -2}}},   // A (near)
        };
        editor::PickResult r = editor::PickNearest(ray, objs);
        check(r.index == 1, "PickNearest selects the nearer AABB (index 1)");
        check(r.t > 1.9f && r.t < 2.1f, "PickNearest reports the near entry t (~2)");

        // A ray that misses both (pointing +X away from the boxes).
        math::Ray miss{{0, 0, 0}, {1, 0, 0}};
        editor::PickResult m = editor::PickNearest(miss, objs);
        check(m.index < 0, "PickNearest misses a non-intersecting ray");
    }

    // ---- PickGizmoAxis: translate/scale/rotate axis selection. --------------------------------
    {
        scene::Transform xf;
        xf.position = {0, 0, 0};
        const float L = 2.0f;

        // Translate: a ray aimed at the middle of the +X handle should pick X. We aim from above
        // (+Y), straight down through the point (L*0.5, 0, 0).
        Vec3 onX{L * 0.5f, 0, 0};
        math::Ray rX{{onX.x, 5.0f, onX.z}, {0, -1, 0}};
        check(editor::PickGizmoAxis(rX, xf, editor::GizmoMode::Translate, L) == editor::kAxisX,
              "PickGizmoAxis Translate selects X");

        Vec3 onZ{0, 0, L * 0.5f};
        math::Ray rZ{{onZ.x, 5.0f, onZ.z}, {0, -1, 0}};
        check(editor::PickGizmoAxis(rZ, xf, editor::GizmoMode::Translate, L) == editor::kAxisZ,
              "PickGizmoAxis Translate selects Z");

        // The Y handle points up; aim a ray along -X passing through (0, L*0.5, 0).
        math::Ray rY{{5.0f, L * 0.5f, 0.0f}, {-1, 0, 0}};
        check(editor::PickGizmoAxis(rY, xf, editor::GizmoMode::Translate, L) == editor::kAxisY,
              "PickGizmoAxis Translate selects Y");

        // Scale uses the same segment test.
        check(editor::PickGizmoAxis(rX, xf, editor::GizmoMode::Scale, L) == editor::kAxisX,
              "PickGizmoAxis Scale selects X");

        // A ray that misses every handle (far off to +X beyond the tips, parallel to Y).
        math::Ray rMiss{{10.0f, 5.0f, 10.0f}, {0, -1, 0}};
        check(editor::PickGizmoAxis(rMiss, xf, editor::GizmoMode::Translate, L) == editor::kAxisNone,
              "PickGizmoAxis misses when no handle is near");

        // Rotate: the X-rotation circle lies in the YZ plane (x=0). Aim a ray along -X from
        // (5, 0, L) so it crosses the YZ plane at (0, 0, L) — radius |hit-o| = L, ON the X circle.
        // (The Y circle is in the ZX plane y=0 and the Z circle in the XY plane z=0; this ray is
        // parallel to both, so only X qualifies.)
        math::Ray rotX{{5.0f, 0.0f, L}, {-1, 0, 0}};
        check(editor::PickGizmoAxis(rotX, xf, editor::GizmoMode::Rotate, L) == editor::kAxisX,
              "PickGizmoAxis Rotate selects X");
    }

    // ---- ApplyDrag: translate / scale / rotate deltas. ----------------------------------------
    {
        scene::Transform xf;
        xf.position = {0, 0, 0};

        // Translate along +X: two parallel downward rays at x=1 then x=3 -> +2 on X.
        math::Ray p{{1.0f, 5.0f, 0.0f}, {0, -1, 0}};
        math::Ray c{{3.0f, 5.0f, 0.0f}, {0, -1, 0}};
        scene::Transform moved = editor::ApplyDrag(xf, editor::GizmoMode::Translate, editor::kAxisX, p, c);
        check(approx(moved.position.x, 2.0f, 1e-2f), "ApplyDrag Translate X moves +2");
        check(approx(moved.position.y, 0.0f) && approx(moved.position.z, 0.0f),
              "ApplyDrag Translate leaves other axes fixed");

        // Scale along +Y: rays crossing the Y axis at y=2 then y=3 -> scale.y += 1.
        scene::Transform sf;
        math::Ray ps{{5.0f, 2.0f, 0.0f}, {-1, 0, 0}};
        math::Ray cs{{5.0f, 3.0f, 0.0f}, {-1, 0, 0}};
        scene::Transform scaled = editor::ApplyDrag(sf, editor::GizmoMode::Scale, editor::kAxisY, ps, cs);
        check(approx(scaled.scale.y, 2.0f, 1e-2f), "ApplyDrag Scale Y grows by +1 (1->2)");
        check(approx(scaled.scale.x, 1.0f) && approx(scaled.scale.z, 1.0f),
              "ApplyDrag Scale leaves other axes at 1");

        // Rotate about +Y by ~90deg: in the ZX plane (axis Y), point starts on +Z, ends on +X.
        // PlaneBasis(Y) = u=+Z, v=+X. atan2 goes from 0 -> +pi/2, so euler.y += ~+pi/2.
        scene::Transform rf;
        math::Ray pr{{0.0f, 5.0f, 1.0f}, {0, -1, 0}};   // hits Y-plane (y=0) at (0,0,1) -> on +Z (u)
        math::Ray cr{{1.0f, 5.0f, 0.0f}, {0, -1, 0}};   // hits at (1,0,0) -> on +X (v)
        scene::Transform rotated = editor::ApplyDrag(rf, editor::GizmoMode::Rotate, editor::kAxisY, pr, cr);
        check(approx(rotated.eulerRadians.y, 1.5707963f, 1e-2f), "ApplyDrag Rotate Y ~ +90deg");

        // kAxisNone leaves the transform unchanged.
        scene::Transform same = editor::ApplyDrag(xf, editor::GizmoMode::Translate, editor::kAxisNone, p, c);
        check(approx(same.position.x, 0.0f), "ApplyDrag kAxisNone is a no-op");
    }

    // ---- PlayState: paused -> 0, step -> 1, playing -> passthrough. ---------------------------
    {
        runtime::PlayState ps;
        check(ps.IsPlaying(), "PlayState starts Playing");
        check(ps.StepsThisTick(3) == 3, "Playing passes fixed steps through");

        ps.Pause();
        check(ps.IsPaused(), "Pause -> Paused");
        check(ps.StepsThisTick(3) == 0, "Paused consumes 0 steps");
        check(ps.StepsThisTick(0) == 0, "Paused with 0 accumulated -> 0");

        ps.RequestStep();
        check(ps.StepsThisTick(0) == 1, "Step request advances exactly 1 (even with 0 accumulated)");
        check(ps.StepsThisTick(5) == 0, "Step request is one-shot (next tick 0)");

        ps.RequestStep();
        ps.Play();
        check(ps.StepsThisTick(2) == 2, "Resuming Play ignores a stale step request");

        ps.Toggle();
        check(ps.IsPaused(), "Toggle from Playing -> Paused");
    }

    // ---- Scripted select -> translate -> DumpScene round-trip. --------------------------------
    {
        ecs::Registry reg;
        // DumpScene only COMPARES the mesh pointer (NameOfMesh) — never dereferences it — so an
        // opaque placeholder address is enough to round-trip the transform through the IO layer.
        scene::Mesh* dummy = reinterpret_cast<scene::Mesh*>(0x1);
        scene::SceneResources res;
        res.AddMesh("cube", dummy);

        // Two objects; we "select" object index 1 and translate it.
        std::vector<ecs::Entity> ents;
        for (int i = 0; i < 2; ++i) {
            ecs::Entity e = reg.create();
            scene::Transform t;
            t.position = {(float)i, 0.0f, 0.0f};
            reg.add(e, scene::TransformC{t});
            reg.add(e, scene::MeshC{dummy});
            reg.add(e, scene::MaterialC{});
            ents.push_back(e);
        }

        // Programmatic selection of view-order index 1.
        editor::Selection sel;
        sel.index = 1;
        sel.mode = editor::GizmoMode::Translate;
        check(sel.Has(), "Selection.Has() after selecting index 1");

        // Apply a +Z translate of 4.25 via ApplyDrag (rays crossing the Z axis at z=0 then z=4.25).
        scene::Transform& live = reg.get<scene::TransformC>(ents[sel.index]).t;
        math::Ray prev{{0.0f, 5.0f, 0.0f}, {0, -1, 0}};
        math::Ray cur{{0.0f, 5.0f, 4.25f}, {0, -1, 0}};
        live = editor::ApplyDrag(live, editor::GizmoMode::Translate, editor::kAxisZ, prev, cur);
        check(approx(live.position.z, 4.25f, 1e-2f), "live transform moved +4.25 on Z");

        std::string json = scene::DumpScene(reg, res);
        // The moved object's z position (formatted by DumpScene's %g) should appear in the JSON.
        bool hasMoved = json.find("4.25") != std::string::npos;
        check(hasMoved, "DumpScene JSON reflects the moved (+Z) transform");
    }

    if (g_fail == 0) std::printf("editor_test: ALL PASS\n");
    else std::printf("editor_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
