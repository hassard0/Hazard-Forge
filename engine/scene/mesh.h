#pragma once
#include <memory>
#include <cstdint>
#include <vector>
#include "rhi/rhi.h"
#include "math/math.h"
#include "scene/vertex.h"
namespace hf::scene {

// CPU-side geometry (the vertex + index arrays) for a procedural mesh. Exposed so callers that need
// the RAW geometry — e.g. the multi-draw-indirect showcase (Slice BM), which packs the cube + sphere
// into one COMBINED vertex/index buffer and references each with a {firstIndex, vertexOffset} slice —
// can build their own buffers. Mesh::Cube/Sphere upload exactly these arrays, so a buffer built from
// CubeGeometry() is byte-identical to the one inside Mesh::Cube (existing goldens unchanged).
struct MeshGeometry {
    std::vector<Vertex>   verts;
    std::vector<uint32_t> indices;
};
MeshGeometry CubeGeometry();
MeshGeometry SphereGeometry(uint32_t segments = 24, uint32_t rings = 16);

// Object-space axis-aligned bounds, recorded at mesh-build time so the debug-draw overlay (Slice W)
// can fit a wireframe AABB around a posed mesh without retaining the full CPU vertex array.
struct MeshBounds { math::Vec3 min{0, 0, 0}; math::Vec3 max{0, 0, 0}; };
class Mesh {
public:
    Mesh(std::unique_ptr<rhi::IBuffer> v, std::unique_ptr<rhi::IBuffer> i, uint32_t indexCount,
         MeshBounds bounds = {})
        : vertices_(std::move(v)), indices_(std::move(i)), indexCount_(indexCount), bounds_(bounds) {}
    rhi::IBuffer& vertices() const { return *vertices_; }
    rhi::IBuffer& indices() const { return *indices_; }
    uint32_t indexCount() const { return indexCount_; }
    // Object-space AABB (see MeshBounds). Zero-extent for meshes built without bounds.
    const MeshBounds& bounds() const { return bounds_; }
    static Mesh Cube(rhi::IRHIDevice& device);
    static Mesh Plane(rhi::IRHIDevice& device);
    static Mesh Sphere(rhi::IRHIDevice& device, uint32_t segments = 24, uint32_t rings = 16);
private:
    std::unique_ptr<rhi::IBuffer> vertices_, indices_;
    uint32_t indexCount_;
    MeshBounds bounds_;
};
} // namespace
