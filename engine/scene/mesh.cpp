#include "scene/mesh.h"
#include "scene/vertex.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace hf::scene {

namespace {

// Object-space AABB over the vertex positions (recorded so the debug-draw overlay can fit a
// wireframe box around a posed mesh; Slice W).
MeshBounds ComputeBounds(const Vertex* verts, uint64_t vertBytes) {
    const size_t count = (size_t)(vertBytes / sizeof(Vertex));
    if (count == 0) return {};
    float mn[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max()};
    float mx[3] = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                   -std::numeric_limits<float>::max()};
    for (size_t i = 0; i < count; ++i) {
        for (int k = 0; k < 3; ++k) {
            mn[k] = std::min(mn[k], verts[i].pos[k]);
            mx[k] = std::max(mx[k], verts[i].pos[k]);
        }
    }
    return MeshBounds{{mn[0], mn[1], mn[2]}, {mx[0], mx[1], mx[2]}};
}

Mesh BuildMesh(rhi::IRHIDevice& device,
               const Vertex* verts, uint64_t vertBytes,
               const uint32_t* indices, uint64_t indexBytes,
               uint32_t indexCount) {
    rhi::BufferDesc vbdesc;
    vbdesc.size = vertBytes;
    vbdesc.initialData = verts;
    vbdesc.usage = rhi::BufferUsage::Vertex;
    auto vbuffer = device.CreateBuffer(vbdesc);

    rhi::BufferDesc ibdesc;
    ibdesc.size = indexBytes;
    ibdesc.initialData = indices;
    ibdesc.usage = rhi::BufferUsage::Index;
    auto ibuffer = device.CreateBuffer(ibdesc);

    return Mesh{std::move(vbuffer), std::move(ibuffer), indexCount,
                ComputeBounds(verts, vertBytes)};
}

} // namespace

Mesh Mesh::Cube(rhi::IRHIDevice& device) {
    // 24 vertices (4 per face) so each face carries its own UV square + tint.
    // Faces wound CCW outward (RH); UVs per face: (0,0)(1,0)(1,1)(0,1).
    // tangent (location 4) = object-space dP/du, i.e. the direction of increasing U along the
    // face (v0->v1, the first UV edge). Orthogonal to the face normal; handedness +1 so the lit
    // fragment shader's B = cross(N,T) gives the +V direction. (Per-face from the known UV layout.)
    const float n = -0.5f, p = 0.5f;
    const Vertex verts[24] = {
        // -Z (back), tint red, normal (0,0,-1), tangent (-1,0,0)
        {{p, n, n}, {1.0f, 0.4f, 0.4f}, {0, 0}, {0, 0, -1}, {-1, 0, 0}},
        {{n, n, n}, {1.0f, 0.4f, 0.4f}, {1, 0}, {0, 0, -1}, {-1, 0, 0}},
        {{n, p, n}, {1.0f, 0.4f, 0.4f}, {1, 1}, {0, 0, -1}, {-1, 0, 0}},
        {{p, p, n}, {1.0f, 0.4f, 0.4f}, {0, 1}, {0, 0, -1}, {-1, 0, 0}},
        // +Z (front), tint green, normal (0,0,1), tangent (1,0,0)
        {{n, n, p}, {0.4f, 1.0f, 0.4f}, {0, 0}, {0, 0, 1}, {1, 0, 0}},
        {{p, n, p}, {0.4f, 1.0f, 0.4f}, {1, 0}, {0, 0, 1}, {1, 0, 0}},
        {{p, p, p}, {0.4f, 1.0f, 0.4f}, {1, 1}, {0, 0, 1}, {1, 0, 0}},
        {{n, p, p}, {0.4f, 1.0f, 0.4f}, {0, 1}, {0, 0, 1}, {1, 0, 0}},
        // -X (left), tint blue, normal (-1,0,0), tangent (0,0,1)
        {{n, n, n}, {0.4f, 0.4f, 1.0f}, {0, 0}, {-1, 0, 0}, {0, 0, 1}},
        {{n, n, p}, {0.4f, 0.4f, 1.0f}, {1, 0}, {-1, 0, 0}, {0, 0, 1}},
        {{n, p, p}, {0.4f, 0.4f, 1.0f}, {1, 1}, {-1, 0, 0}, {0, 0, 1}},
        {{n, p, n}, {0.4f, 0.4f, 1.0f}, {0, 1}, {-1, 0, 0}, {0, 0, 1}},
        // +X (right), tint yellow, normal (1,0,0), tangent (0,0,-1)
        {{p, n, p}, {1.0f, 1.0f, 0.4f}, {0, 0}, {1, 0, 0}, {0, 0, -1}},
        {{p, n, n}, {1.0f, 1.0f, 0.4f}, {1, 0}, {1, 0, 0}, {0, 0, -1}},
        {{p, p, n}, {1.0f, 1.0f, 0.4f}, {1, 1}, {1, 0, 0}, {0, 0, -1}},
        {{p, p, p}, {1.0f, 1.0f, 0.4f}, {0, 1}, {1, 0, 0}, {0, 0, -1}},
        // -Y (bottom), tint magenta, normal (0,-1,0), tangent (1,0,0)
        {{n, n, n}, {1.0f, 0.4f, 1.0f}, {0, 0}, {0, -1, 0}, {1, 0, 0}},
        {{p, n, n}, {1.0f, 0.4f, 1.0f}, {1, 0}, {0, -1, 0}, {1, 0, 0}},
        {{p, n, p}, {1.0f, 0.4f, 1.0f}, {1, 1}, {0, -1, 0}, {1, 0, 0}},
        {{n, n, p}, {1.0f, 0.4f, 1.0f}, {0, 1}, {0, -1, 0}, {1, 0, 0}},
        // +Y (top), tint cyan, normal (0,1,0), tangent (1,0,0)
        {{n, p, p}, {0.4f, 1.0f, 1.0f}, {0, 0}, {0, 1, 0}, {1, 0, 0}},
        {{p, p, p}, {0.4f, 1.0f, 1.0f}, {1, 0}, {0, 1, 0}, {1, 0, 0}},
        {{p, p, n}, {0.4f, 1.0f, 1.0f}, {1, 1}, {0, 1, 0}, {1, 0, 0}},
        {{n, p, n}, {0.4f, 1.0f, 1.0f}, {0, 1}, {0, 1, 0}, {1, 0, 0}},
    };

    // 36 indices: 2 triangles per face over its 4 vertices, CCW outward.
    uint32_t indices[36];
    for (uint32_t f = 0; f < 6; ++f) {
        uint32_t base = f * 4;
        uint32_t* tri = &indices[f * 6];
        tri[0] = base + 0; tri[1] = base + 1; tri[2] = base + 2;
        tri[3] = base + 0; tri[4] = base + 2; tri[5] = base + 3;
    }

    return BuildMesh(device, verts, sizeof(verts), indices, sizeof(indices), 36);
}

Mesh Mesh::Plane(rhi::IRHIDevice& device) {
    // Large flat quad on the XZ plane at y=0, normal +Y, uv 0..4 for tiling.
    // Wound CCW when viewed from above (+Y looking down -Y) so the top face is front-facing.
    const float h = 0.5f;          // half-extent (scaled up by the Renderable transform)
    const float t = 4.0f;          // uv tiling range
    // Winding matches the cube's +Y (top) face exactly: (-x,+z)(+x,+z)(+x,-z)(-x,-z),
    // which is the engine's outward/up-facing front-face order under back-face culling.
    // tangent = +X (direction of increasing U across the quad, v0->v1); normal +Y, handedness +1.
    const Vertex verts[4] = {
        {{-h, 0.0f,  h}, {1.0f, 1.0f, 1.0f}, {0, t}, {0, 1, 0}, {1, 0, 0}},
        {{ h, 0.0f,  h}, {1.0f, 1.0f, 1.0f}, {t, t}, {0, 1, 0}, {1, 0, 0}},
        {{ h, 0.0f, -h}, {1.0f, 1.0f, 1.0f}, {t, 0}, {0, 1, 0}, {1, 0, 0}},
        {{-h, 0.0f, -h}, {1.0f, 1.0f, 1.0f}, {0, 0}, {0, 1, 0}, {1, 0, 0}},
    };
    const uint32_t indices[6] = {0, 1, 2, 0, 2, 3};

    return BuildMesh(device, verts, sizeof(verts), indices, sizeof(indices), 6);
}

Mesh Mesh::Sphere(rhi::IRHIDevice& device, uint32_t segments, uint32_t rings) {
    // UV sphere of radius 0.5 centered at the origin.
    //   ring  (latitude)  0..rings    : 0 = +Y pole, rings = -Y pole
    //   seg   (longitude) 0..segments : wraps around, with a duplicated seam
    //                                    column (seg == segments) so UVs are seamless.
    // Smooth normals: normal == normalize(position) == position*2 (since |pos|==0.5).
    const float radius = 0.5f;
    const float kPi = 3.14159265358979323846f;

    std::vector<Vertex> verts;
    verts.reserve((rings + 1) * (segments + 1));
    for (uint32_t r = 0; r <= rings; ++r) {
        float v = static_cast<float>(r) / static_cast<float>(rings);
        float phi = v * kPi;                 // 0 at +Y pole -> pi at -Y pole
        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);
        for (uint32_t s = 0; s <= segments; ++s) {
            float u = static_cast<float>(s) / static_cast<float>(segments);
            float theta = u * 2.0f * kPi;
            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);
            // Unit-sphere direction (also the smooth normal).
            float nx = sinPhi * cosTheta;
            float ny = cosPhi;
            float nz = sinPhi * sinTheta;
            Vertex vert{};
            vert.pos[0] = nx * radius; vert.pos[1] = ny * radius; vert.pos[2] = nz * radius;
            vert.color[0] = 0.85f; vert.color[1] = 0.85f; vert.color[2] = 0.9f;
            vert.uv[0] = u; vert.uv[1] = v;
            vert.normal[0] = nx; vert.normal[1] = ny; vert.normal[2] = nz;
            // Tangent = normalize(dP/du). P = r*(sinφcosθ, cosφ, sinφsinθ), u->θ=2πu, so
            // dP/du = r*2π*sinφ*(-sinθ, 0, cosθ); normalized -> (-sinθ, 0, cosθ). At the poles
            // (sinφ==0) the tangent degenerates; fall back to +X so it stays finite.
            if (sinPhi > 1e-6f) {
                vert.tangent[0] = -sinTheta; vert.tangent[1] = 0.0f; vert.tangent[2] = cosTheta;
            } else {
                vert.tangent[0] = 1.0f; vert.tangent[1] = 0.0f; vert.tangent[2] = 0.0f;
            }
            verts.push_back(vert);
        }
    }

    // Two triangles per quad. Wound CCW when viewed from outside so the front
    // (outer) face survives back-face culling, matching the cube's outward winding.
    const uint32_t stride = segments + 1;
    std::vector<uint32_t> indices;
    indices.reserve(rings * segments * 6);
    for (uint32_t r = 0; r < rings; ++r) {
        for (uint32_t s = 0; s < segments; ++s) {
            uint32_t a = r * stride + s;          // (r,   s)
            uint32_t b = (r + 1) * stride + s;    // (r+1, s)
            uint32_t c = (r + 1) * stride + s + 1;// (r+1, s+1)
            uint32_t d = r * stride + s + 1;      // (r,   s+1)
            indices.push_back(a); indices.push_back(b); indices.push_back(c);
            indices.push_back(a); indices.push_back(c); indices.push_back(d);
        }
    }

    return BuildMesh(device,
                     verts.data(), verts.size() * sizeof(Vertex),
                     indices.data(), indices.size() * sizeof(uint32_t),
                     static_cast<uint32_t>(indices.size()));
}

} // namespace hf::scene
