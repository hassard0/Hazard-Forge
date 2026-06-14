#include "asset/gltf_loader.h"
#include "scene/vertex.h"

#define CGLTF_IMPLEMENTATION
#include "cgltf/cgltf.h"

// stb_image: decode the embedded base-color image (PNG/JPEG) bytes -> RGBA8.
// STB_IMAGE_IMPLEMENTATION is defined in exactly this one TU.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO   // we only ever decode from memory (embedded glb buffer_view)
#include "stb/stb_image.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
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

// Build a scene::Mesh from the first primitive of the first mesh, recentred on origin.
scene::Mesh BuildMesh(rhi::IRHIDevice& device, const cgltf_data* data, const char* path) {
    const cgltf_primitive& prim = data->meshes[0].primitives[0];

    const cgltf_accessor* posAcc = FindAttr(prim, cgltf_attribute_type_position);
    if (!posAcc)
        throw std::runtime_error(std::string("glTF primitive has no POSITION: ") + path);
    const cgltf_accessor* nrmAcc = FindAttr(prim, cgltf_attribute_type_normal);
    const cgltf_accessor* uvAcc  = FindAttr(prim, cgltf_attribute_type_texcoord);
    const cgltf_accessor* tanAcc = FindAttr(prim, cgltf_attribute_type_tangent);

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
        // Neutral tint; the textured pipeline multiplies base colour by this vertex colour.
        v.color[0] = 1.0f; v.color[1] = 1.0f; v.color[2] = 1.0f;
        // Defaults; overwritten below if the accessors exist.
        v.uv[0] = 0.0f; v.uv[1] = 0.0f;
        v.normal[0] = 0.0f; v.normal[1] = 1.0f; v.normal[2] = 0.0f;
        v.tangent[0] = 1.0f; v.tangent[1] = 0.0f; v.tangent[2] = 0.0f;
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

    // --- Tangents (location 4). Prefer the authored TANGENT accessor (VEC4: xyz + w handedness);
    // otherwise accumulate per-triangle tangents from positions+UVs (Lengyel's method) and
    // Gram-Schmidt orthonormalize against the vertex normal. If neither path yields a usable
    // tangent the default (1,0,0) set above is kept so the lit shader's TBN stays finite. ---
    if (tanAcc) {
        for (cgltf_size i = 0; i < vertCount; ++i) {
            float tg[4] = {1, 0, 0, 1};
            cgltf_accessor_read_float(tanAcc, i, tg, 4);
            // w is +/-1 handedness; the engine's lit shader bakes B = cross(N,T)*handedness, but the
            // scene::Vertex carries only xyz. Fold a -1 handedness into the stored tangent direction
            // so cross(N,T) yields the correct bitangent without a separate handedness channel.
            float s = (tg[3] < 0.0f) ? -1.0f : 1.0f;
            verts[i].tangent[0] = tg[0] * s;
            verts[i].tangent[1] = tg[1] * s;
            verts[i].tangent[2] = tg[2] * s;
        }
    } else if (uvAcc) {
        std::vector<float> tan(vertCount * 3, 0.0f);
        for (size_t t = 0; t + 2 < indices.size(); t += 3) {
            uint32_t i0 = indices[t], i1 = indices[t + 1], i2 = indices[t + 2];
            const float* p0 = verts[i0].pos; const float* p1 = verts[i1].pos; const float* p2 = verts[i2].pos;
            const float* w0 = verts[i0].uv;  const float* w1 = verts[i1].uv;  const float* w2 = verts[i2].uv;
            float e1[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
            float e2[3] = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};
            float du1 = w1[0]-w0[0], dv1 = w1[1]-w0[1];
            float du2 = w2[0]-w0[0], dv2 = w2[1]-w0[1];
            float det = du1*dv2 - du2*dv1;
            float r = (std::fabs(det) > 1e-8f) ? (1.0f / det) : 0.0f;
            float tx = (dv2*e1[0] - dv1*e2[0]) * r;
            float ty = (dv2*e1[1] - dv1*e2[1]) * r;
            float tz = (dv2*e1[2] - dv1*e2[2]) * r;
            for (uint32_t vi : {i0, i1, i2}) {
                tan[vi*3+0] += tx; tan[vi*3+1] += ty; tan[vi*3+2] += tz;
            }
        }
        for (cgltf_size i = 0; i < vertCount; ++i) {
            const float* n = verts[i].normal;
            float t[3] = {tan[i*3+0], tan[i*3+1], tan[i*3+2]};
            // Gram-Schmidt: T = normalize(T - N*dot(N,T)).
            float ndt = n[0]*t[0] + n[1]*t[1] + n[2]*t[2];
            t[0] -= n[0]*ndt; t[1] -= n[1]*ndt; t[2] -= n[2]*ndt;
            float len = std::sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]);
            if (len > 1e-6f) {
                verts[i].tangent[0] = t[0]/len; verts[i].tangent[1] = t[1]/len; verts[i].tangent[2] = t[2]/len;
            }
        }
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

// A flat-white 1x1 RGBA8 texture: used as a fallback when the material has no usable
// base-color texture, so the caller always has something valid to bind.
std::unique_ptr<rhi::ITexture> MakeWhiteTexture(rhi::IRHIDevice& device) {
    const uint8_t whitePx[4] = {255, 255, 255, 255};
    return device.CreateTexture({1, 1, rhi::Format::RGBA8_UNorm, whitePx, sizeof(whitePx)});
}

// Decode the material's base-color image (embedded glb buffer_view, or external/data: URI)
// into an RGBA8 rhi::ITexture. Returns a white fallback on any failure.
std::unique_ptr<rhi::ITexture> LoadBaseColorTexture(rhi::IRHIDevice& device,
                                                    const cgltf_image* image) {
    if (!image) return MakeWhiteTexture(device);

    const stbi_uc* srcBytes = nullptr;
    int srcLen = 0;

    if (image->buffer_view) {
        // .glb: the image bytes live inside a buffer_view of an embedded buffer.
        const cgltf_buffer_view* bv = image->buffer_view;
        if (!bv->buffer || !bv->buffer->data) {
            std::fprintf(stderr,
                "[gltf] base-color image buffer_view has no data; using white fallback\n");
            return MakeWhiteTexture(device);
        }
        srcBytes = static_cast<const stbi_uc*>(bv->buffer->data) + bv->offset;
        srcLen = static_cast<int>(bv->size);
    } else if (image->uri) {
        // External or data: URI. cgltf_load_buffers decodes data: URIs into a synthesized
        // buffer; if the loader didn't materialise it into a buffer_view we can't reach the
        // bytes here, so fall back gracefully (the showcase asset is an embedded glb).
        std::fprintf(stderr,
            "[gltf] base-color image is a URI ('%s') without an embedded buffer_view; "
            "using white fallback\n", image->uri);
        return MakeWhiteTexture(device);
    } else {
        return MakeWhiteTexture(device);
    }

    int w = 0, h = 0, comp = 0;
    // Force 4 channels (RGBA8) to match rhi::Format::RGBA8_UNorm and the tight upload path.
    stbi_uc* pixels = stbi_load_from_memory(srcBytes, srcLen, &w, &h, &comp, 4);
    if (!pixels || w <= 0 || h <= 0) {
        std::fprintf(stderr,
            "[gltf] stbi_load_from_memory failed (%s); using white fallback\n",
            stbi_failure_reason() ? stbi_failure_reason() : "unknown");
        if (pixels) stbi_image_free(pixels);
        return MakeWhiteTexture(device);
    }

    rhi::TextureDesc td;
    td.width = static_cast<uint32_t>(w);
    td.height = static_cast<uint32_t>(h);
    td.format = rhi::Format::RGBA8_UNorm;
    td.data = pixels;
    td.dataSize = static_cast<uint64_t>(w) * h * 4;
    auto tex = device.CreateTexture(td);

    stbi_image_free(pixels);
    return tex;
}

// Open + parse + load buffers + validate a glTF/glb file. Returns the cgltf_data*; the
// caller owns it and must cgltf_free it. Throws on failure.
cgltf_data* OpenGltf(const char* path) {
    cgltf_options options{};
    cgltf_data* data = nullptr;

    cgltf_result res = cgltf_parse_file(&options, path, &data);
    if (res != cgltf_result_success)
        throw std::runtime_error(std::string("cgltf_parse_file failed for ") + path);

    res = cgltf_load_buffers(&options, data, path);
    if (res != cgltf_result_success) {
        cgltf_free(data);
        throw std::runtime_error(std::string("cgltf_load_buffers failed for ") + path);
    }

    res = cgltf_validate(data);
    if (res != cgltf_result_success) {
        cgltf_free(data);
        throw std::runtime_error(std::string("cgltf_validate failed for ") + path);
    }

    if (data->meshes_count == 0 || data->meshes[0].primitives_count == 0) {
        cgltf_free(data);
        throw std::runtime_error(std::string("glTF has no mesh/primitive: ") + path);
    }
    return data;
}

} // namespace

scene::Mesh LoadGltfMesh(rhi::IRHIDevice& device, const char* path) {
    cgltf_data* data = OpenGltf(path);
    struct Guard { cgltf_data* d; ~Guard() { if (d) cgltf_free(d); } } guard{data};
    return BuildMesh(device, data, path);
}

GltfModel LoadGltfModel(rhi::IRHIDevice& device, const char* path) {
    cgltf_data* data = OpenGltf(path);
    struct Guard { cgltf_data* d; ~Guard() { if (d) cgltf_free(d); } } guard{data};

    scene::Mesh mesh = BuildMesh(device, data, path);

    // Material: first primitive's pbr_metallic_roughness factors + base-color texture.
    const cgltf_primitive& prim = data->meshes[0].primitives[0];
    const cgltf_material* mat = prim.material;
    const cgltf_image* baseImg = nullptr;
    float metallic = 1.0f, roughness = 1.0f;
    if (mat && mat->has_pbr_metallic_roughness) {
        const cgltf_pbr_metallic_roughness& pbr = mat->pbr_metallic_roughness;
        metallic = pbr.metallic_factor;
        roughness = pbr.roughness_factor;
        if (pbr.base_color_texture.texture && pbr.base_color_texture.texture->image)
            baseImg = pbr.base_color_texture.texture->image;
    }
    auto baseColor = LoadBaseColorTexture(device, baseImg);
    return GltfModel(std::move(mesh), std::move(baseColor), metallic, roughness);
}

} // namespace hf::asset
