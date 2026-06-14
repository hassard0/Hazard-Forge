#include "asset/gltf_loader.h"
#include "scene/vertex.h"
#include "anim/skeleton.h"
#include "anim/animation.h"
#include "math/math.h"

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

// A 1x1 RGBA8 texture of a single constant color: used as a fallback for any absent PBR map so the
// lit-PBR shader can always bind five textures. (r,g,b,a) each 0..255.
std::unique_ptr<rhi::ITexture> MakeSolidTexture(rhi::IRHIDevice& device,
                                                uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    const uint8_t px[4] = {r, g, b, a};
    return device.CreateTexture({1, 1, rhi::Format::RGBA8_UNorm, px, sizeof(px)});
}

// Decode a glTF image (embedded glb buffer_view, or external/data: URI) into an RGBA8 rhi::ITexture.
// Returns nullptr if the image is absent or cannot be reached/decoded; callers substitute a sensible
// fallback. `label` names the map in diagnostics.
std::unique_ptr<rhi::ITexture> DecodeImage(rhi::IRHIDevice& device, const cgltf_image* image,
                                           const char* label) {
    if (!image) return nullptr;

    const stbi_uc* srcBytes = nullptr;
    int srcLen = 0;

    if (image->buffer_view) {
        // .glb: the image bytes live inside a buffer_view of an embedded buffer.
        const cgltf_buffer_view* bv = image->buffer_view;
        if (!bv->buffer || !bv->buffer->data) {
            std::fprintf(stderr, "[gltf] %s image buffer_view has no data; using fallback\n", label);
            return nullptr;
        }
        srcBytes = static_cast<const stbi_uc*>(bv->buffer->data) + bv->offset;
        srcLen = static_cast<int>(bv->size);
    } else if (image->uri) {
        // External or data: URI. cgltf_load_buffers decodes data: URIs into a synthesized
        // buffer; if the loader didn't materialise it into a buffer_view we can't reach the
        // bytes here, so fall back gracefully (the showcase assets are embedded glb).
        std::fprintf(stderr,
            "[gltf] %s image is a URI ('%s') without an embedded buffer_view; using fallback\n",
            label, image->uri);
        return nullptr;
    } else {
        return nullptr;
    }

    int w = 0, h = 0, comp = 0;
    // Force 4 channels (RGBA8) to match rhi::Format::RGBA8_UNorm and the tight upload path.
    stbi_uc* pixels = stbi_load_from_memory(srcBytes, srcLen, &w, &h, &comp, 4);
    if (!pixels || w <= 0 || h <= 0) {
        std::fprintf(stderr, "[gltf] %s stbi_load_from_memory failed (%s); using fallback\n", label,
                     stbi_failure_reason() ? stbi_failure_reason() : "unknown");
        if (pixels) stbi_image_free(pixels);
        return nullptr;
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

// Decode the material's base-color image into an RGBA8 texture; white fallback on any failure.
std::unique_ptr<rhi::ITexture> LoadBaseColorTexture(rhi::IRHIDevice& device,
                                                    const cgltf_image* image) {
    auto tex = DecodeImage(device, image, "base-color");
    return tex ? std::move(tex) : MakeWhiteTexture(device);
}

// Pull the image pointer out of a glTF texture-view (texture -> image), or null if absent.
const cgltf_image* ImageOf(const cgltf_texture_view& view) {
    return (view.texture && view.texture->image) ? view.texture->image : nullptr;
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

PbrModel LoadPbrGltfModel(rhi::IRHIDevice& device, const char* path) {
    cgltf_data* data = OpenGltf(path);
    struct Guard { cgltf_data* d; ~Guard() { if (d) cgltf_free(d); } } guard{data};

    scene::Mesh mesh = BuildMesh(device, data, path);

    // First primitive's material: metallic-roughness factors + emissive factor + the five textures.
    const cgltf_primitive& prim = data->meshes[0].primitives[0];
    const cgltf_material* mat = prim.material;

    float metallic = 1.0f, roughness = 1.0f;
    float emissiveF[3] = {0.0f, 0.0f, 0.0f};
    const cgltf_image* baseImg = nullptr;
    const cgltf_image* mrImg = nullptr;
    const cgltf_image* normalImg = nullptr;
    const cgltf_image* emissiveImg = nullptr;
    const cgltf_image* occlusionImg = nullptr;

    if (mat) {
        if (mat->has_pbr_metallic_roughness) {
            const cgltf_pbr_metallic_roughness& pbr = mat->pbr_metallic_roughness;
            metallic = pbr.metallic_factor;
            roughness = pbr.roughness_factor;
            baseImg = ImageOf(pbr.base_color_texture);
            mrImg = ImageOf(pbr.metallic_roughness_texture);
        }
        normalImg = ImageOf(mat->normal_texture);
        emissiveImg = ImageOf(mat->emissive_texture);
        occlusionImg = ImageOf(mat->occlusion_texture);
        for (int k = 0; k < 3; ++k) emissiveF[k] = mat->emissive_factor[k];
    }

    // Decode each present texture; substitute a sensible 1x1 fallback for any absent map so the
    // lit-PBR shader always binds five textures with neutral semantics:
    //   base       -> white            (albedo unchanged)
    //   metalRough -> (255,255,0,255)  G=rough=1, B=metallic=0 (then scaled by the factors)
    //   normal     -> (128,128,255)    decodes to (0,0,1): no TBN perturbation
    //   emissive   -> black            (adds nothing)
    //   occlusion  -> white            R=1: no ambient occlusion
    auto baseColor = LoadBaseColorTexture(device, baseImg);
    auto metalRough = DecodeImage(device, mrImg, "metallic-roughness");
    if (!metalRough) metalRough = MakeSolidTexture(device, 255, 255, 0, 255);
    auto normalMap = DecodeImage(device, normalImg, "normal");
    if (!normalMap) normalMap = MakeSolidTexture(device, 128, 128, 255, 255);
    auto emissive = DecodeImage(device, emissiveImg, "emissive");
    if (!emissive) emissive = MakeSolidTexture(device, 0, 0, 0, 255);
    auto occlusion = DecodeImage(device, occlusionImg, "occlusion");
    if (!occlusion) occlusion = MakeWhiteTexture(device);

    return PbrModel(std::move(mesh), std::move(baseColor), std::move(metalRough),
                    std::move(normalMap), std::move(emissive), std::move(occlusion),
                    metallic, roughness, emissiveF);
}

// ============================== Skinned model loading ==========================================

namespace {

// Node index within data->nodes (pointer arithmetic). -1 if null/foreign.
int NodeIndex(const cgltf_data* data, const cgltf_node* node) {
    if (!node) return -1;
    ptrdiff_t idx = node - data->nodes;
    if (idx < 0 || (cgltf_size)idx >= data->nodes_count) return -1;
    return static_cast<int>(idx);
}

// Build the skeleton from a skin: collect the skin's joint nodes, topologically sort them so every
// parent precedes its children, and fill each Joint's parent index, inverse-bind matrix and rest
// TRS. Also produces nodeIndex -> jointIndex (skeleton order) and the glTF-skin-order -> jointIndex
// remap (so JOINTS_0 indices and animation channels can be remapped).
struct SkeletonBuild {
    anim::Skeleton skeleton;
    std::vector<int> nodeToJoint;     // size data->nodes_count; -1 if not a joint
    std::vector<int> skinOrderToJoint; // size skin->joints_count
};

SkeletonBuild BuildSkeleton(const cgltf_data* data, const cgltf_skin* skin) {
    SkeletonBuild out;
    out.nodeToJoint.assign(data->nodes_count, -1);

    const cgltf_size n = skin->joints_count;

    // Map each skin-joint slot to its node index, and mark which nodes are joints.
    std::vector<int> skinNode(n, -1);
    std::vector<bool> isJointNode(data->nodes_count, false);
    for (cgltf_size i = 0; i < n; ++i) {
        int ni = NodeIndex(data, skin->joints[i]);
        skinNode[i] = ni;
        if (ni >= 0) isJointNode[ni] = true;
    }

    // Topological sort over the joint set: a joint's effective parent is its nearest ancestor that
    // is also in the joint set (glTF allows non-joint intermediate nodes, though Fox has none).
    // We emit roots first, then children, via a simple repeated-pass / visited approach.
    std::vector<int> nodeParentJointNode(data->nodes_count, -1);  // nearest joint-ancestor node
    for (cgltf_size i = 0; i < n; ++i) {
        int ni = skinNode[i];
        if (ni < 0) continue;
        const cgltf_node* p = data->nodes[ni].parent;
        int pj = -1;
        while (p) {
            int pi = NodeIndex(data, p);
            if (pi >= 0 && isJointNode[pi]) { pj = pi; break; }
            p = p->parent;
        }
        nodeParentJointNode[ni] = pj;
    }

    // Emit in topological order: a joint can be emitted once its parent-joint-node has been emitted
    // (or it has none). Iterate until all joints are placed.
    std::vector<bool> emitted(data->nodes_count, false);
    out.skeleton.joints.reserve(n);
    bool progress = true;
    cgltf_size placed = 0;
    while (placed < n && progress) {
        progress = false;
        for (cgltf_size i = 0; i < n; ++i) {
            int ni = skinNode[i];
            if (ni < 0 || emitted[ni]) continue;
            int pj = nodeParentJointNode[ni];
            if (pj >= 0 && !emitted[pj]) continue;  // parent not emitted yet
            // Place this joint.
            out.nodeToJoint[ni] = static_cast<int>(out.skeleton.joints.size());
            anim::Joint joint;
            joint.parent = (pj >= 0) ? out.nodeToJoint[pj] : -1;
            const cgltf_node& nd = data->nodes[ni];
            if (nd.has_translation) joint.t = math::Vec3{nd.translation[0], nd.translation[1], nd.translation[2]};
            if (nd.has_rotation)    joint.r = math::Quat{nd.rotation[0], nd.rotation[1], nd.rotation[2], nd.rotation[3]};
            if (nd.has_scale)       joint.s = math::Vec3{nd.scale[0], nd.scale[1], nd.scale[2]};
            out.skeleton.joints.push_back(joint);
            emitted[ni] = true;
            ++placed;
            progress = true;
        }
    }

    // Inverse-bind matrices (column-major float4x4 per skin-joint slot). Map slot -> emitted joint.
    out.skinOrderToJoint.assign(n, -1);
    for (cgltf_size i = 0; i < n; ++i) {
        int ni = skinNode[i];
        int ji = (ni >= 0) ? out.nodeToJoint[ni] : -1;
        out.skinOrderToJoint[i] = ji;
        if (ji < 0) continue;
        math::Mat4 ib = math::Mat4::Identity();
        if (skin->inverse_bind_matrices)
            cgltf_accessor_read_float(skin->inverse_bind_matrices, i, ib.m, 16);
        out.skeleton.joints[ji].inverseBind = ib;
    }

    return out;
}

// Parse all animations, remapping each channel's target node to a skeleton joint index.
std::vector<anim::Animation> BuildAnimations(const cgltf_data* data,
                                             const std::vector<int>& nodeToJoint) {
    std::vector<anim::Animation> anims;
    anims.reserve(data->animations_count);
    for (cgltf_size a = 0; a < data->animations_count; ++a) {
        const cgltf_animation& ga = data->animations[a];
        anim::Animation out;
        if (ga.name) out.name = ga.name;
        out.channels.reserve(ga.channels_count);
        float duration = 0.0f;
        for (cgltf_size c = 0; c < ga.channels_count; ++c) {
            const cgltf_animation_channel& gc = ga.channels[c];
            if (gc.target_path == cgltf_animation_path_type_weights ||
                gc.target_path == cgltf_animation_path_type_invalid) continue;  // YAGNI: morph
            int ni = NodeIndex(data, gc.target_node);
            if (ni < 0 || (size_t)ni >= nodeToJoint.size() || nodeToJoint[ni] < 0) continue;

            anim::Channel ch;
            ch.jointIndex = nodeToJoint[ni];
            switch (gc.target_path) {
                case cgltf_animation_path_type_translation: ch.path = anim::Channel::Path::Translation; break;
                case cgltf_animation_path_type_rotation:    ch.path = anim::Channel::Path::Rotation; break;
                case cgltf_animation_path_type_scale:       ch.path = anim::Channel::Path::Scale; break;
                default: continue;
            }
            // Cubic-spline keyframes carry in/out tangents; we sample only the value (treat as
            // linear). The Fox uses LINEAR/STEP, so this is exact for the showcase.
            ch.interp = (gc.sampler->interpolation == cgltf_interpolation_type_step)
                            ? anim::Channel::Interp::Step : anim::Channel::Interp::Linear;

            const cgltf_accessor* in = gc.sampler->input;
            const cgltf_accessor* outAcc = gc.sampler->output;
            const cgltf_size keys = in->count;
            ch.times.resize(keys);
            for (cgltf_size k = 0; k < keys; ++k) {
                float tval = 0.0f;
                cgltf_accessor_read_float(in, k, &tval, 1);
                ch.times[k] = tval;
                if (tval > duration) duration = tval;
            }
            const int comp = (ch.path == anim::Channel::Path::Rotation) ? 4 : 3;
            // For cubic-spline, output has 3 entries per key (in-tangent, value, out-tangent); read
            // the middle (value) one. Otherwise one entry per key.
            const bool cubic = (gc.sampler->interpolation == cgltf_interpolation_type_cubic_spline);
            ch.values.resize(keys * comp);
            for (cgltf_size k = 0; k < keys; ++k) {
                cgltf_size srcKey = cubic ? (k * 3 + 1) : k;
                cgltf_accessor_read_float(outAcc, srcKey, &ch.values[k * comp], comp);
            }
            out.channels.push_back(std::move(ch));
        }
        out.duration = duration;
        anims.push_back(std::move(out));
    }
    return anims;
}

// Build the skinned mesh from the first primitive: positions, UVs, JOINTS_0 (remapped to skeleton
// joint order), WEIGHTS_0 (renormalized), smooth normals when NORMAL is absent. NOT recentred.
scene::Mesh BuildSkinnedMesh(rhi::IRHIDevice& device, const cgltf_data* data,
                             const std::vector<int>& skinOrderToJoint,
                             float bbMin[3], float bbMax[3], const char* path) {
    const cgltf_primitive& prim = data->meshes[0].primitives[0];

    const cgltf_accessor* posAcc = FindAttr(prim, cgltf_attribute_type_position);
    if (!posAcc)
        throw std::runtime_error(std::string("skinned glTF primitive has no POSITION: ") + path);
    const cgltf_accessor* nrmAcc = FindAttr(prim, cgltf_attribute_type_normal);
    const cgltf_accessor* uvAcc  = FindAttr(prim, cgltf_attribute_type_texcoord);
    const cgltf_accessor* jntAcc = FindAttr(prim, cgltf_attribute_type_joints);
    const cgltf_accessor* wgtAcc = FindAttr(prim, cgltf_attribute_type_weights);

    const cgltf_size vertCount = posAcc->count;
    std::vector<scene::SkinnedVertex> verts(vertCount);
    bbMin[0] = bbMin[1] = bbMin[2] =  1e30f;
    bbMax[0] = bbMax[1] = bbMax[2] = -1e30f;

    for (cgltf_size i = 0; i < vertCount; ++i) {
        float p[3] = {0, 0, 0};
        cgltf_accessor_read_float(posAcc, i, p, 3);
        scene::SkinnedVertex& v = verts[i];
        v.pos[0] = p[0]; v.pos[1] = p[1]; v.pos[2] = p[2];
        for (int k = 0; k < 3; ++k) {
            if (p[k] < bbMin[k]) bbMin[k] = p[k];
            if (p[k] > bbMax[k]) bbMax[k] = p[k];
        }
        v.color[0] = v.color[1] = v.color[2] = 1.0f;
        v.uv[0] = 0.0f; v.uv[1] = 0.0f;
        v.normal[0] = 0.0f; v.normal[1] = 1.0f; v.normal[2] = 0.0f;
        v.tangent[0] = 1.0f; v.tangent[1] = 0.0f; v.tangent[2] = 0.0f;
        v.joints[0] = v.joints[1] = v.joints[2] = v.joints[3] = 0.0f;
        v.weights[0] = 1.0f; v.weights[1] = v.weights[2] = v.weights[3] = 0.0f;
    }

    if (uvAcc) {
        for (cgltf_size i = 0; i < vertCount; ++i) {
            float uv[2] = {0, 0};
            cgltf_accessor_read_float(uvAcc, i, uv, 2);
            verts[i].uv[0] = uv[0]; verts[i].uv[1] = uv[1];
        }
    }

    // JOINTS_0: read as four floats (cgltf widens u8/u16 -> float), then remap each glTF-skin-order
    // joint index to the skeleton's topologically-sorted joint index.
    if (jntAcc) {
        for (cgltf_size i = 0; i < vertCount; ++i) {
            float j[4] = {0, 0, 0, 0};
            cgltf_accessor_read_float(jntAcc, i, j, 4);
            for (int k = 0; k < 4; ++k) {
                int slot = static_cast<int>(j[k] + 0.5f);
                int mapped = (slot >= 0 && (size_t)slot < skinOrderToJoint.size())
                                 ? skinOrderToJoint[slot] : 0;
                verts[i].joints[k] = static_cast<float>(mapped < 0 ? 0 : mapped);
            }
        }
    }

    // WEIGHTS_0: read + renormalize so the four weights sum to 1 (guards against quantization drift).
    if (wgtAcc) {
        for (cgltf_size i = 0; i < vertCount; ++i) {
            float w[4] = {0, 0, 0, 0};
            cgltf_accessor_read_float(wgtAcc, i, w, 4);
            float sum = w[0] + w[1] + w[2] + w[3];
            if (sum > 1e-8f) {
                float inv = 1.0f / sum;
                for (int k = 0; k < 4; ++k) w[k] *= inv;
            } else {
                w[0] = 1.0f; w[1] = w[2] = w[3] = 0.0f;
            }
            for (int k = 0; k < 4; ++k) verts[i].weights[k] = w[k];
        }
    }

    // Indices (u16/u32 -> u32). Non-indexed primitives get a trivial list.
    std::vector<uint32_t> indices;
    if (prim.indices) {
        const cgltf_size nn = prim.indices->count;
        indices.resize(nn);
        for (cgltf_size i = 0; i < nn; ++i)
            indices[i] = static_cast<uint32_t>(cgltf_accessor_read_index(prim.indices, i));
    } else {
        indices.resize(vertCount);
        for (cgltf_size i = 0; i < vertCount; ++i) indices[i] = static_cast<uint32_t>(i);
    }

    // Smooth normals from the indexed positions when NORMAL is absent (the Fox has none). Accumulate
    // per-triangle face normals into each vertex, then normalize.
    if (nrmAcc) {
        for (cgltf_size i = 0; i < vertCount; ++i) {
            float n[3] = {0, 1, 0};
            cgltf_accessor_read_float(nrmAcc, i, n, 3);
            verts[i].normal[0] = n[0]; verts[i].normal[1] = n[1]; verts[i].normal[2] = n[2];
        }
    } else {
        std::vector<float> acc(vertCount * 3, 0.0f);
        for (size_t t = 0; t + 2 < indices.size(); t += 3) {
            uint32_t i0 = indices[t], i1 = indices[t + 1], i2 = indices[t + 2];
            const float* p0 = verts[i0].pos; const float* p1 = verts[i1].pos; const float* p2 = verts[i2].pos;
            float e1[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
            float e2[3] = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};
            float fn[3] = {e1[1]*e2[2] - e1[2]*e2[1],
                           e1[2]*e2[0] - e1[0]*e2[2],
                           e1[0]*e2[1] - e1[1]*e2[0]};
            for (uint32_t vi : {i0, i1, i2}) {
                acc[vi*3+0] += fn[0]; acc[vi*3+1] += fn[1]; acc[vi*3+2] += fn[2];
            }
        }
        for (cgltf_size i = 0; i < vertCount; ++i) {
            float n[3] = {acc[i*3+0], acc[i*3+1], acc[i*3+2]};
            float len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
            if (len > 1e-8f) {
                verts[i].normal[0] = n[0]/len; verts[i].normal[1] = n[1]/len; verts[i].normal[2] = n[2]/len;
            }
        }
    }

    rhi::BufferDesc vbdesc;
    vbdesc.size = verts.size() * sizeof(scene::SkinnedVertex);
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

} // namespace

const anim::Animation* SkinnedModel::FindAnimation(const char* name) const {
    for (const auto& a : animations)
        if (a.name == name) return &a;
    return nullptr;
}

SkinnedModel LoadSkinnedGltfModel(rhi::IRHIDevice& device, const char* path) {
    cgltf_data* data = OpenGltf(path);
    struct Guard { cgltf_data* d; ~Guard() { if (d) cgltf_free(d); } } guard{data};

    if (data->skins_count == 0)
        throw std::runtime_error(std::string("glTF has no skin: ") + path);
    const cgltf_skin* skin = &data->skins[0];

    SkeletonBuild sb = BuildSkeleton(data, skin);
    std::vector<anim::Animation> animations = BuildAnimations(data, sb.nodeToJoint);

    float bbMin[3], bbMax[3];
    scene::Mesh mesh = BuildSkinnedMesh(device, data, sb.skinOrderToJoint, bbMin, bbMax, path);

    // Material: first primitive's PBR factors + base-color texture (same path as GltfModel).
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

    return SkinnedModel(std::move(mesh), std::move(baseColor), metallic, roughness,
                        std::move(sb.skeleton), std::move(animations), bbMin, bbMax);
}

} // namespace hf::asset
