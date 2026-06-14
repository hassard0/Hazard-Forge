#include "scene/mesh.h"
#include "scene/vertex.h"

namespace hf::scene {

namespace {

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

    return Mesh{std::move(vbuffer), std::move(ibuffer), indexCount};
}

} // namespace

Mesh Mesh::Cube(rhi::IRHIDevice& device) {
    // 24 vertices (4 per face) so each face carries its own UV square + tint.
    // Faces wound CCW outward (RH); UVs per face: (0,0)(1,0)(1,1)(0,1).
    const float n = -0.5f, p = 0.5f;
    const Vertex verts[24] = {
        // -Z (back), tint red, normal (0,0,-1)
        {{p, n, n}, {1.0f, 0.4f, 0.4f}, {0, 0}, {0, 0, -1}},
        {{n, n, n}, {1.0f, 0.4f, 0.4f}, {1, 0}, {0, 0, -1}},
        {{n, p, n}, {1.0f, 0.4f, 0.4f}, {1, 1}, {0, 0, -1}},
        {{p, p, n}, {1.0f, 0.4f, 0.4f}, {0, 1}, {0, 0, -1}},
        // +Z (front), tint green, normal (0,0,1)
        {{n, n, p}, {0.4f, 1.0f, 0.4f}, {0, 0}, {0, 0, 1}},
        {{p, n, p}, {0.4f, 1.0f, 0.4f}, {1, 0}, {0, 0, 1}},
        {{p, p, p}, {0.4f, 1.0f, 0.4f}, {1, 1}, {0, 0, 1}},
        {{n, p, p}, {0.4f, 1.0f, 0.4f}, {0, 1}, {0, 0, 1}},
        // -X (left), tint blue, normal (-1,0,0)
        {{n, n, n}, {0.4f, 0.4f, 1.0f}, {0, 0}, {-1, 0, 0}},
        {{n, n, p}, {0.4f, 0.4f, 1.0f}, {1, 0}, {-1, 0, 0}},
        {{n, p, p}, {0.4f, 0.4f, 1.0f}, {1, 1}, {-1, 0, 0}},
        {{n, p, n}, {0.4f, 0.4f, 1.0f}, {0, 1}, {-1, 0, 0}},
        // +X (right), tint yellow, normal (1,0,0)
        {{p, n, p}, {1.0f, 1.0f, 0.4f}, {0, 0}, {1, 0, 0}},
        {{p, n, n}, {1.0f, 1.0f, 0.4f}, {1, 0}, {1, 0, 0}},
        {{p, p, n}, {1.0f, 1.0f, 0.4f}, {1, 1}, {1, 0, 0}},
        {{p, p, p}, {1.0f, 1.0f, 0.4f}, {0, 1}, {1, 0, 0}},
        // -Y (bottom), tint magenta, normal (0,-1,0)
        {{n, n, n}, {1.0f, 0.4f, 1.0f}, {0, 0}, {0, -1, 0}},
        {{p, n, n}, {1.0f, 0.4f, 1.0f}, {1, 0}, {0, -1, 0}},
        {{p, n, p}, {1.0f, 0.4f, 1.0f}, {1, 1}, {0, -1, 0}},
        {{n, n, p}, {1.0f, 0.4f, 1.0f}, {0, 1}, {0, -1, 0}},
        // +Y (top), tint cyan, normal (0,1,0)
        {{n, p, p}, {0.4f, 1.0f, 1.0f}, {0, 0}, {0, 1, 0}},
        {{p, p, p}, {0.4f, 1.0f, 1.0f}, {1, 0}, {0, 1, 0}},
        {{p, p, n}, {0.4f, 1.0f, 1.0f}, {1, 1}, {0, 1, 0}},
        {{n, p, n}, {0.4f, 1.0f, 1.0f}, {0, 1}, {0, 1, 0}},
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
    const Vertex verts[4] = {
        {{-h, 0.0f,  h}, {1.0f, 1.0f, 1.0f}, {0, t}, {0, 1, 0}},
        {{ h, 0.0f,  h}, {1.0f, 1.0f, 1.0f}, {t, t}, {0, 1, 0}},
        {{ h, 0.0f, -h}, {1.0f, 1.0f, 1.0f}, {t, 0}, {0, 1, 0}},
        {{-h, 0.0f, -h}, {1.0f, 1.0f, 1.0f}, {0, 0}, {0, 1, 0}},
    };
    const uint32_t indices[6] = {0, 1, 2, 0, 2, 3};

    return BuildMesh(device, verts, sizeof(verts), indices, sizeof(indices), 6);
}

} // namespace hf::scene
