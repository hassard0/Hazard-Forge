// Slice AB — transform gizmo: drawing + axis hit-testing + drag math. Pure C++.
#include "editor/gizmo.h"

#include <cmath>

namespace hf::editor {

using math::Mat4;
using math::Ray;
using math::Vec3;

namespace {

// The three world basis axes.
Vec3 AxisDir(int axis) {
    switch (axis) {
        case kAxisX: return {1, 0, 0};
        case kAxisY: return {0, 1, 0};
        default:     return {0, 0, 1};
    }
}

// Per-axis base colors (matching DebugDraw::Axes: X red, Y green, Z blue).
Vec3 AxisColor(int axis, bool active) {
    Vec3 c;
    switch (axis) {
        case kAxisX: c = {1.0f, 0.15f, 0.15f}; break;
        case kAxisY: c = {0.15f, 1.0f, 0.15f}; break;
        default:     c = {0.2f, 0.4f, 1.0f};   break;
    }
    if (active) {
        // Brighten the active axis toward white so the grabbed handle is unmistakable.
        c.x = c.x + (1.0f - c.x) * 0.6f;
        c.y = c.y + (1.0f - c.y) * 0.6f;
        c.z = c.z + (1.0f - c.z) * 0.6f;
    }
    return c;
}

// Parameter `s` along the axis line through `origin` (dir `axis`, unit) where the ray comes closest,
// plus the world distance at that closest approach. The axis is treated as an (effectively infinite)
// segment by sampling a long span centered on the origin, so s can be negative or positive.
float RayAxisProjection(const Ray& ray, const Vec3& origin, const Vec3& axisDir, float& outDist) {
    const float kSpan = 1000.0f;
    Vec3 a = origin - axisDir * kSpan;
    Vec3 b = origin + axisDir * kSpan;
    float rayT = 0.0f, dist = 0.0f;
    float segT = math::RayClosestParamToSegment(ray, a, b, rayT, dist);
    outDist = dist;
    // segT in [0,1] over [a,b]; convert to signed distance from `origin` along axisDir.
    return (segT * 2.0f - 1.0f) * kSpan;
}

// Intersect the ray with the plane through `point` whose normal is `normal`. Returns false if the
// ray is parallel to the plane. On success `out` is the world hit point.
bool RayPlane(const Ray& ray, const Vec3& point, const Vec3& normal, Vec3& out) {
    float denom = math::dot(ray.dir, normal);
    if (denom > -1e-6f && denom < 1e-6f) return false;
    float t = math::dot(point - ray.origin, normal) / denom;
    if (t < 0.0f) return false;
    out = ray.origin + ray.dir * t;
    return true;
}

// Two orthonormal in-plane basis vectors for the plane whose normal is one of the world axes.
void PlaneBasis(int axis, Vec3& u, Vec3& v) {
    switch (axis) {
        case kAxisX: u = {0, 1, 0}; v = {0, 0, 1}; break;  // YZ plane
        case kAxisY: u = {0, 0, 1}; v = {1, 0, 0}; break;  // ZX plane
        default:     u = {1, 0, 0}; v = {0, 1, 0}; break;  // XY plane
    }
}

}  // namespace

void EmitGizmo(debug::DebugDraw& dd, const scene::Transform& xform, GizmoMode mode, float handleLen,
               int activeAxis) {
    const Vec3 o = xform.position;
    if (handleLen <= 0.0f) handleLen = 1.0f;

    if (mode == GizmoMode::Translate) {
        for (int axis = 0; axis < 3; ++axis) {
            Vec3 dir = AxisDir(axis);
            Vec3 tip = o + dir * handleLen;
            Vec3 col = AxisColor(axis, axis == activeAxis);
            dd.Line(o, tip, col);
            // Arrowhead: two short fins angled back from the tip, in the axis's plane.
            Vec3 u, v;
            PlaneBasis(axis, u, v);
            float head = handleLen * 0.18f;
            Vec3 back = tip - dir * head;
            dd.Line(tip, back + u * head, col);
            dd.Line(tip, back - u * head, col);
        }
    } else if (mode == GizmoMode::Rotate) {
        for (int axis = 0; axis < 3; ++axis) {
            Vec3 col = AxisColor(axis, axis == activeAxis);
            Vec3 u, v;
            PlaneBasis(axis, u, v);
            const int kSeg = 32;
            const float kTwoPi = 6.28318530718f;
            for (int s = 0; s < kSeg; ++s) {
                float a0 = kTwoPi * (float)s / (float)kSeg;
                float a1 = kTwoPi * (float)(s + 1) / (float)kSeg;
                Vec3 p0 = o + (u * std::cos(a0) + v * std::sin(a0)) * handleLen;
                Vec3 p1 = o + (u * std::cos(a1) + v * std::sin(a1)) * handleLen;
                dd.Line(p0, p1, col);
            }
        }
    } else {  // Scale
        for (int axis = 0; axis < 3; ++axis) {
            Vec3 dir = AxisDir(axis);
            Vec3 tip = o + dir * handleLen;
            Vec3 col = AxisColor(axis, axis == activeAxis);
            dd.Line(o, tip, col);
            // Small box at the end of each axis line.
            float h = handleLen * 0.08f;
            dd.Box(tip - Vec3{h, h, h}, tip + Vec3{h, h, h}, col);
        }
    }
}

int PickGizmoAxis(const Ray& ray, const scene::Transform& xform, GizmoMode mode, float handleLen) {
    const Vec3 o = xform.position;
    if (handleLen <= 0.0f) handleLen = 1.0f;

    if (mode == GizmoMode::Translate || mode == GizmoMode::Scale) {
        // Hit the axis whose handle SEGMENT the ray passes nearest, within a tolerance scaled to the
        // handle length. Tie-break by smallest distance.
        const float tol = handleLen * 0.18f;
        int best = kAxisNone;
        float bestDist = tol;
        for (int axis = 0; axis < 3; ++axis) {
            Vec3 dir = AxisDir(axis);
            Vec3 tip = o + dir * handleLen;
            float rayT = 0.0f, dist = 0.0f;
            (void)math::RayClosestParamToSegment(ray, o, tip, rayT, dist);
            if (dist < bestDist) { bestDist = dist; best = axis; }
        }
        return best;
    }

    // Rotate: the ray must cross each axis's plane near the circle of radius handleLen. Pick the axis
    // whose plane-crossing radius is closest to handleLen, within tolerance.
    const float tol = handleLen * 0.18f;
    int best = kAxisNone;
    float bestErr = tol;
    for (int axis = 0; axis < 3; ++axis) {
        Vec3 n = AxisDir(axis);
        Vec3 hit;
        if (!RayPlane(ray, o, n, hit)) continue;
        float radius = math::length(hit - o);
        float err = std::fabs(radius - handleLen);
        if (err < bestErr) { bestErr = err; best = axis; }
    }
    return best;
}

scene::Transform ApplyDrag(const scene::Transform& xform, GizmoMode mode, int axis, const Ray& prev,
                           const Ray& cur) {
    scene::Transform out = xform;
    if (axis < 0 || axis > 2) return out;
    const Vec3 o = xform.position;
    Vec3 dir = AxisDir(axis);

    if (mode == GizmoMode::Translate) {
        float dPrev = 0.0f, dCur = 0.0f;
        float sPrev = RayAxisProjection(prev, o, dir, dPrev);
        float sCur  = RayAxisProjection(cur,  o, dir, dCur);
        float delta = sCur - sPrev;
        if (axis == kAxisX) out.position.x += delta;
        else if (axis == kAxisY) out.position.y += delta;
        else out.position.z += delta;
        return out;
    }

    if (mode == GizmoMode::Scale) {
        float dPrev = 0.0f, dCur = 0.0f;
        float sPrev = RayAxisProjection(prev, o, dir, dPrev);
        float sCur  = RayAxisProjection(cur,  o, dir, dCur);
        float delta = sCur - sPrev;
        float* s = (axis == kAxisX) ? &out.scale.x : (axis == kAxisY) ? &out.scale.y : &out.scale.z;
        *s += delta;
        if (*s < 1e-3f) *s = 1e-3f;  // keep scale positive
        return out;
    }

    // Rotate: signed angle swept about the axis between the prev/cur plane-hit directions.
    Vec3 n = dir;
    Vec3 u, v;
    PlaneBasis(axis, u, v);
    Vec3 hPrev, hCur;
    if (!RayPlane(prev, o, n, hPrev) || !RayPlane(cur, o, n, hCur)) return out;
    Vec3 dp = hPrev - o;
    Vec3 dc = hCur - o;
    float aPrev = std::atan2(math::dot(dp, v), math::dot(dp, u));
    float aCur  = std::atan2(math::dot(dc, v), math::dot(dc, u));
    float dAng = aCur - aPrev;
    // Wrap to (-pi, pi] so a drag across the +/-pi seam doesn't spin the wrong way.
    const float kPi = 3.14159265358979f;
    while (dAng > kPi) dAng -= 2.0f * kPi;
    while (dAng < -kPi) dAng += 2.0f * kPi;
    if (axis == kAxisX) out.eulerRadians.x += dAng;
    else if (axis == kAxisY) out.eulerRadians.y += dAng;
    else out.eulerRadians.z += dAng;
    return out;
}

}  // namespace hf::editor
