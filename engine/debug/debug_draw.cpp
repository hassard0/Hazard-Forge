// Slice W — immediate-mode debug-draw collector. Pure C++ (engine/math + stdlib only).
#include "debug/debug_draw.h"

#include <cmath>

namespace hf::debug {
namespace {

// Transform a point by a column-major Mat4 (w=1): p' = M * [p,1]. element(row,col) == m[col*4+row].
Vec3 TransformPoint(const Mat4& m, const Vec3& p) {
    return {
        m.m[0] * p.x + m.m[4] * p.y + m.m[8]  * p.z + m.m[12],
        m.m[1] * p.x + m.m[5] * p.y + m.m[9]  * p.z + m.m[13],
        m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14],
    };
}

}  // namespace

void DebugDraw::Line(Vec3 a, Vec3 b, Vec3 color) {
    verts_.push_back({{a.x, a.y, a.z}, {color.x, color.y, color.z}});
    verts_.push_back({{b.x, b.y, b.z}, {color.x, color.y, color.z}});
}

void DebugDraw::Ray(Vec3 origin, Vec3 dir, float len, Vec3 color) {
    Line(origin, origin + math::normalize(dir) * len, color);
}

void DebugDraw::Box(Vec3 mn, Vec3 mx, Vec3 color) {
    // 8 corners; bit 0 = X(min/max), bit 1 = Y, bit 2 = Z.
    const Vec3 c[8] = {
        {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z}, {mn.x, mx.y, mn.z}, {mx.x, mx.y, mn.z},
        {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z}, {mn.x, mx.y, mx.z}, {mx.x, mx.y, mx.z},
    };
    // 12 edges of a cube (deterministic order: 4 bottom-face X/Z, 4 top, 4 verticals).
    static const int e[12][2] = {
        {0, 1}, {1, 3}, {3, 2}, {2, 0},   // -Z / +Z faces' rings via Y... actually 4 edges of z=mn ring
        {4, 5}, {5, 7}, {7, 6}, {6, 4},   // z=mx ring
        {0, 4}, {1, 5}, {2, 6}, {3, 7},   // connecting verticals
    };
    for (auto& edge : e) Line(c[edge[0]], c[edge[1]], color);
}

void DebugDraw::Obb(const Mat4& transform, Vec3 he, Vec3 color) {
    // Local [-he,+he] corners, then transformed to world by the model matrix.
    const Vec3 lc[8] = {
        {-he.x, -he.y, -he.z}, {he.x, -he.y, -he.z}, {-he.x, he.y, -he.z}, {he.x, he.y, -he.z},
        {-he.x, -he.y,  he.z}, {he.x, -he.y,  he.z}, {-he.x, he.y,  he.z}, {he.x, he.y,  he.z},
    };
    Vec3 c[8];
    for (int i = 0; i < 8; ++i) c[i] = TransformPoint(transform, lc[i]);
    static const int e[12][2] = {
        {0, 1}, {1, 3}, {3, 2}, {2, 0},
        {4, 5}, {5, 7}, {7, 6}, {6, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    for (auto& edge : e) Line(c[edge[0]], c[edge[1]], color);
}

void DebugDraw::WireSphere(Vec3 center, float radius, Vec3 color, int segments) {
    if (segments < 3) segments = 3;
    const float kTwoPi = 6.28318530718f;
    // Three orthogonal great circles. For each, walk `segments` chords around the unit circle and
    // place it in the chosen plane, scaled by radius and offset to center.
    for (int plane = 0; plane < 3; ++plane) {
        for (int s = 0; s < segments; ++s) {
            float a0 = kTwoPi * (float)s / (float)segments;
            float a1 = kTwoPi * (float)(s + 1) / (float)segments;
            float c0 = std::cos(a0), s0 = std::sin(a0);
            float c1 = std::cos(a1), s1 = std::sin(a1);
            Vec3 p0, p1;
            if (plane == 0) {        // XY plane (z const)
                p0 = {c0, s0, 0.0f}; p1 = {c1, s1, 0.0f};
            } else if (plane == 1) { // XZ plane (y const)
                p0 = {c0, 0.0f, s0}; p1 = {c1, 0.0f, s1};
            } else {                 // YZ plane (x const)
                p0 = {0.0f, c0, s0}; p1 = {0.0f, c1, s1};
            }
            Line(center + p0 * radius, center + p1 * radius, color);
        }
    }
}

void DebugDraw::Grid(float halfSize, float step, Vec3 color) {
    if (step <= 0.0f) return;
    int n = (int)std::floor(halfSize / step);
    // Lines parallel to Z (varying x), then lines parallel to X (varying z). Deterministic order.
    for (int i = -n; i <= n; ++i) {
        float x = (float)i * step;
        Line({x, 0.0f, -halfSize}, {x, 0.0f, halfSize}, color);
    }
    for (int i = -n; i <= n; ++i) {
        float z = (float)i * step;
        Line({-halfSize, 0.0f, z}, {halfSize, 0.0f, z}, color);
    }
}

void DebugDraw::Axes(const Mat4& transform, float len) {
    Vec3 o = TransformPoint(transform, {0, 0, 0});
    Vec3 px = TransformPoint(transform, {len, 0, 0});
    Vec3 py = TransformPoint(transform, {0, len, 0});
    Vec3 pz = TransformPoint(transform, {0, 0, len});
    Line(o, px, {1.0f, 0.15f, 0.15f});   // X = red
    Line(o, py, {0.15f, 1.0f, 0.15f});   // Y = green
    Line(o, pz, {0.2f, 0.4f, 1.0f});     // Z = blue
}

}  // namespace hf::debug
