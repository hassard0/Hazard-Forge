#include "asset/gltf_loader.h"
#include "scene/vertex.h"

#define CGLTF_IMPLEMENTATION
#include "cgltf/cgltf.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace hf::asset {

namespace {

// Find a primitive attribute by semantic; returns nullptr if absent.
const cgltf_accessor* FindAttr(const cgltf_primitive& prim, cgltf_attribute_type type) {
    for (cgltf_size i = 0; i < prim.attributes_count; ++i) {
        if (prim.attributes[i].type == type) return prim.attributes[i].data;
    }
    return nullptr;
}

} // namespace

scene::Mesh LoadGltfMesh(rhi::IRHIDevice& device, const char* path) {
    cgltf_options options{};
    cgltf_data* data = nullptr;

    cgltf_result res = cgltf_parse_file(&options, path, &data);
    if (res != cgltf_result_success)
        throw std::runtime_error(std::string("cgltf_parse_file failed for ") + path);

    // RAII-ish guard: free cgltf_data on any exit.
    struct Guard { cgltf_data* d; ~Guard() { if (d) cgltf_free(d); } } guard{data};

    res = cgltf_load_buffers(&options, data, path);
    if (res != cgltf_result_success)
        throw std::runtime_error(std::string("cgltf_load_buffers failed for ") + path);

    res = cgltf_validate(data);
    if (res != cgltf_result_success)
        throw std::runtime_error(std::string("cgltf_validate failed for ") + path);

    if (data->meshes_count == 0 || data->meshes[0].primitives_count == 0)
        throw std::runtime_error(std::string("glTF has no mesh/primitive: ") + path);

    const cgltf_primitive& prim = data->meshes[0].primitives[0];

    const cgltf_accessor* posAcc = FindAttr(prim, cgltf_attribute_type_position);
    if (!posAcc)
        throw std::runtime_error(std::string("glTF primitive has no POSITION: ") + path);
    const cgltf_accessor* nrmAcc = FindAttr(prim, cgltf_attribute_type_normal);
    const cgltf_accessor* uvAcc  = FindAttr(prim, cgltf_attribute_type_texcoord);

    const cgltf_size vertCount = posAcc->count;

    // --- Read positions, compute bbox so we can recentre on the origin. ---
    std::vector<scene::Vertex> verts(vertCount);
    float bbMin[3] = { 1e30f,  1e30f,  1e30f};
    float bbMax[3] = {-1e30f, -1e30f, -1e30f};
    for (cgltf_size i = 0; i < vertCount; ++i) {
        float p[3] = {0, 0, 0};
        cgltf_accessor_read_float(posAcc, i, p, 3);
        scene::Vertex& v = verts[i];
        v.pos[0] = p[0]; v.pos[1] = p[1]; v.pos[2] = p[2];
        for (int k = 0; k < 3; ++k) {
            if (p[k] < bbMin[k]) bbMin[k] = p[k];
            if (p[k] > bbMax[k]) bbMax[k] = p[k];
        }
        // Neutral tint; metallic PBR ignores base colour but keep it sensible.
        v.color[0] = 0.8f; v.color[1] = 0.8f; v.color[2] = 0.85f;
        // Defaults; overwritten below if the accessors exist.
        v.uv[0] = 0.0f; v.uv[1] = 0.0f;
        v.normal[0] = 0.0f; v.normal[1] = 1.0f; v.normal[2] = 0.0f;
    }

    const float center[3] = {0.5f * (bbMin[0] + bbMax[0]),
                             0.5f * (bbMin[1] + bbMax[1]),
                             0.5f * (bbMin[2] + bbMax[2])};
    for (cgltf_size i = 0; i < vertCount; ++i) {
        verts[i].pos[0] -= center[0];
        verts[i].pos[1] -= center[1];
        verts[i].pos[2] -= center[2];
    }

    if (nrmAcc) {
        for (cgltf_size i = 0; i < vertCount; ++i) {
            float n[3] = {0, 1, 0};
            cgltf_accessor_read_float(nrmAcc, i, n, 3);
            verts[i].normal[0] = n[0]; verts[i].normal[1] = n[1]; verts[i].normal[2] = n[2];
        }
    }
    if (uvAcc) {
        for (cgltf_size i = 0; i < vertCount; ++i) {
            float uv[2] = {0, 0};
            cgltf_accessor_read_float(uvAcc, i, uv, 2);
            verts[i].uv[0] = uv[0]; verts[i].uv[1] = uv[1];
        }
    }

    // --- Indices (u16/u32 -> u32). If the primitive is non-indexed, build a trivial index list. ---
    std::vector<uint32_t> indices;
    if (prim.indices) {
        const cgltf_size n = prim.indices->count;
        indices.resize(n);
        for (cgltf_size i = 0; i < n; ++i)
            indices[i] = static_cast<uint32_t>(cgltf_accessor_read_index(prim.indices, i));
    } else {
        indices.resize(vertCount);
        for (cgltf_size i = 0; i < vertCount; ++i) indices[i] = static_cast<uint32_t>(i);
    }

    // --- Upload to GPU buffers via the RHI. ---
    rhi::BufferDesc vbdesc;
    vbdesc.size = verts.size() * sizeof(scene::Vertex);
    vbdesc.initialData = verts.data();
    vbdesc.usage = rhi::BufferUsage::Vertex;
    auto vbuffer = device.CreateBuffer(vbdesc);

    rhi::BufferDesc ibdesc;
    ibdesc.size = indices.size() * sizeof(uint32_t);
    ibdesc.initialData = indices.data();
    ibdesc.usage = rhi::BufferUsage::Index;
    auto ibuffer = device.CreateBuffer(ibdesc);

    return scene::Mesh{std::move(vbuffer), std::move(ibuffer),
                       static_cast<uint32_t>(indices.size())};
}

} // namespace hf::asset
