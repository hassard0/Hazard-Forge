// Slice W — debug-draw convenience emitters (engine-object overloads). Pure C++.
#include "debug/debug_emitters.h"

#include <cmath>
#include <limits>

namespace hf::debug {
namespace {

Vec3 TransformPoint(const Mat4& m, const Vec3& p) {
    return {
        m.m[0] * p.x + m.m[4] * p.y + m.m[8]  * p.z + m.m[12],
        m.m[1] * p.x + m.m[5] * p.y + m.m[9]  * p.z + m.m[13],
        m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14],
    };
}

// Transform a direction (w=0): the upper-left 3x3 only (ignores translation). Not normalized.
Vec3 TransformDir(const Mat4& m, const Vec3& v) {
    return {
        m.m[0] * v.x + m.m[4] * v.y + m.m[8]  * v.z,
        m.m[1] * v.x + m.m[5] * v.y + m.m[9]  * v.z,
        m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z,
    };
}

}  // namespace

void AabbWorld(DebugDraw& dd, Vec3 localMin, Vec3 localMax, const Mat4& model, Vec3 color) {
    const Vec3 lc[8] = {
        {localMin.x, localMin.y, localMin.z}, {localMax.x, localMin.y, localMin.z},
        {localMin.x, localMax.y, localMin.z}, {localMax.x, localMax.y, localMin.z},
        {localMin.x, localMin.y, localMax.z}, {localMax.x, localMin.y, localMax.z},
        {localMin.x, localMax.y, localMax.z}, {localMax.x, localMax.y, localMax.z},
    };
    Vec3 mn{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max()};
    Vec3 mx{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max()};
    for (const auto& corner : lc) {
        Vec3 w = TransformPoint(model, corner);
        mn.x = std::min(mn.x, w.x); mn.y = std::min(mn.y, w.y); mn.z = std::min(mn.z, w.z);
        mx.x = std::max(mx.x, w.x); mx.y = std::max(mx.y, w.y); mx.z = std::max(mx.z, w.z);
    }
    dd.Box(mn, mx, color);
}

void MeshAabb(DebugDraw& dd, std::span<const scene::Vertex> verts, const Mat4& model, Vec3 color) {
    if (verts.empty()) return;
    Vec3 mn{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max()};
    Vec3 mx{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max()};
    for (const auto& v : verts) {
        mn.x = std::min(mn.x, v.pos[0]); mn.y = std::min(mn.y, v.pos[1]); mn.z = std::min(mn.z, v.pos[2]);
        mx.x = std::max(mx.x, v.pos[0]); mx.y = std::max(mx.y, v.pos[1]); mx.z = std::max(mx.z, v.pos[2]);
    }
    AabbWorld(dd, mn, mx, model, color);
}

void MeshNormals(DebugDraw& dd, std::span<const scene::Vertex> verts, const Mat4& model, float len,
                 Vec3 color) {
    for (const auto& v : verts) {
        Vec3 p = TransformPoint(model, {v.pos[0], v.pos[1], v.pos[2]});
        Vec3 n = math::normalize(TransformDir(model, {v.normal[0], v.normal[1], v.normal[2]}));
        dd.Line(p, p + n * len, color);
    }
}

void LightArrow(DebugDraw& dd, Vec3 origin, Vec3 dir, float len, Vec3 color) {
    Vec3 d = math::normalize(dir);
    Vec3 tip = origin + d * len;
    dd.Line(origin, tip, color);
    // Build a basis orthogonal to d for the four head fins.
    Vec3 up = (std::fabs(d.y) < 0.99f) ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    Vec3 right = math::normalize(math::cross(d, up));
    Vec3 realUp = math::cross(right, d);
    float headLen = len * 0.18f;
    float headWidth = len * 0.10f;
    Vec3 base = tip - d * headLen;
    dd.Line(tip, base + right * headWidth, color);
    dd.Line(tip, base - right * headWidth, color);
    dd.Line(tip, base + realUp * headWidth, color);
    dd.Line(tip, base - realUp * headWidth, color);
}

void PhysicsContacts(DebugDraw& dd, const physics::World& world, Vec3 pointColor, Vec3 normalColor) {
    const auto& bodies = world.bodies;
    const int n = (int)bodies.size();
    auto cross = [&](const Vec3& p, float s) {
        dd.Line({p.x - s, p.y, p.z}, {p.x + s, p.y, p.z}, pointColor);
        dd.Line({p.x, p.y - s, p.z}, {p.x, p.y + s, p.z}, pointColor);
        dd.Line({p.x, p.y, p.z - s}, {p.x, p.y, p.z + s}, pointColor);
    };
    const float kCrossSize = 0.08f;
    const float kNormalLen = 0.3f;

    // Sphere/ground (body index ascending) — mirrors world.cpp's fixed detection order.
    for (int i = 0; i < n; ++i) {
        const auto& b = bodies[i];
        if (b.shape != physics::Shape::Sphere) continue;
        float pen = b.radius - (b.position.y - world.groundY);
        if (pen > 0.0f) {
            Vec3 pt{b.position.x, world.groundY, b.position.z};
            cross(pt, kCrossSize);
            dd.Line(pt, pt + Vec3{0, 1, 0} * kNormalLen, normalColor);  // ground normal +Y
        }
    }
    // Sphere/sphere (i ascending, j>i ascending).
    for (int i = 0; i < n; ++i) {
        const auto& A = bodies[i];
        if (A.shape != physics::Shape::Sphere) continue;
        for (int j = i + 1; j < n; ++j) {
            const auto& B = bodies[j];
            if (B.shape != physics::Shape::Sphere) continue;
            if (A.invMass == 0.0f && B.invMass == 0.0f) continue;
            Vec3 dvec = B.position - A.position;
            float dist = math::length(dvec);
            float rsum = A.radius + B.radius;
            if (dist < rsum && dist > 1e-6f) {
                Vec3 nrm = dvec / dist;
                float pen = rsum - dist;
                Vec3 pt = A.position + nrm * (A.radius - 0.5f * pen);
                cross(pt, kCrossSize);
                dd.Line(pt, pt + nrm * kNormalLen, normalColor);
            }
        }
    }
}

}  // namespace hf::debug
