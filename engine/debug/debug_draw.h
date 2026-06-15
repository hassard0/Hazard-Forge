#pragma once
// Slice W — immediate-mode debug-draw collector. Pure C++ (engine/math + stdlib only); NO RHI or
// graphics-backend symbols. Compiled into hf_core (ASan-scoped, unit-tested) and hf_engine.
//
// Usage: each frame, Clear(), append shapes (Line/Box/WireSphere/Grid/Axes/...), then upload
// Vertices() into a LINE_LIST vertex buffer (see GraphicsPipelineDesc.lineList) and draw it with a
// pipeline whose vertex shader projects pos by FrameData.viewProj and passes color through.
//
// All shapes decompose to line segments (two LineVertex per segment) appended to one vertex list in
// a DETERMINISTIC order, so the same calls produce byte-identical vertex data — drivable and
// verifiable headlessly (the "agentic visualization" lens).
#include <vector>

#include "math/math.h"

namespace hf::debug {

using math::Mat4;
using math::Vec3;

// One vertex of a debug line. Matches shaders/debug_line.vert.hlsl input (pos@0, color@1).
struct LineVertex {
    float pos[3];
    float color[3];
};

class DebugDraw {
public:
    // Drop all accumulated lines (call once at the start of each frame).
    void Clear() { verts_.clear(); }

    // A single world-space line segment from a to b.
    void Line(Vec3 a, Vec3 b, Vec3 color);

    // A ray drawn as a segment from origin extending len along (normalized) dir.
    void Ray(Vec3 origin, Vec3 dir, float len, Vec3 color);

    // An axis-aligned bounding box (min..max corners) as its 12 edges.
    void Box(Vec3 min, Vec3 max, Vec3 color);

    // An oriented box: the 8 corners of a [-halfExtents, +halfExtents] box transformed by `transform`
    // (model matrix), drawn as 12 edges.
    void Obb(const Mat4& transform, Vec3 halfExtents, Vec3 color);

    // A wireframe sphere: three orthogonal great circles (XY, XZ, YZ planes), each `segments` chords.
    void WireSphere(Vec3 center, float radius, Vec3 color, int segments = 16);

    // A ground grid in the XZ plane (y=0): lines at every `step` from -halfSize..+halfSize on both
    // axes. The center axes (x=0, z=0) are emitted at the SAME color (kept simple/deterministic).
    void Grid(float halfSize, float step, Vec3 color);

    // A transform gizmo: the three basis axes of `transform`, each of world length `len`, colored
    // X=red, Y=green, Z=blue.
    void Axes(const Mat4& transform, float len);

    const std::vector<LineVertex>& Vertices() const { return verts_; }
    bool Empty() const { return verts_.empty(); }
    size_t VertexCount() const { return verts_.size(); }

private:
    std::vector<LineVertex> verts_;
};

}  // namespace hf::debug
