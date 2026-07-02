#pragma once
#include "scene/mesh.h"
#include "rhi/rhi.h"
#include "anim/skeleton.h"
#include "anim/animation.h"
#include "math/math.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

// Forward declaration so the CPU geometry-extraction helper can take a cgltf primitive without pulling
// the whole cgltf.h into this header (the .cpp / the test TU define CGLTF_IMPLEMENTATION).
struct cgltf_primitive;

namespace hf::asset {

// ---------------------------------------------------------------------------------------------------
// DEVICE-FREE CPU geometry extraction for a single glTF primitive (issue #36 — the Draco seam).
//
// Builds the engine's scene::Vertex array + the u32 index list for a primitive, handling BOTH the
// uncompressed path (cgltf accessors) AND the KHR_draco_mesh_compression path (the self-contained
// hf::asset::draco decoder in asset/draco_decode.h). No RHI device is touched, so the Draco decode is
// unit-testable without a GPU. BuildPrimitive() calls this, then uploads the result to GPU buffers.
//
//  * recentre — when true, recentre the geometry so its bbox centre sits at the origin (legacy
//               single-mesh behaviour); when false keep authored model-space positions (scene import).
//  * outBbMin/outBbMax — optional: the primitive's (post-recentre) model-space bbox.
//
// Throws std::runtime_error on missing POSITION or a Draco primitive whose blob fails to decode.
struct CpuPrimitive {
    std::vector<scene::Vertex> verts;
    std::vector<uint32_t>      indices;
};
CpuPrimitive BuildPrimitiveCPU(const cgltf_primitive& prim, const char* path, bool recentre,
                               float* outBbMin = nullptr, float* outBbMax = nullptr);

// ---------------------------------------------------------------------------------------------------
// PURE external-image-URI resolution (Slice SC1 — real-Sponza hero bake). Multi-file .gltf assets
// (e.g. the Khronos PBR Sponza) reference their images as file URIs relative to the .gltf's own
// directory instead of embedding them in a .glb buffer_view. These two helpers are the device-free,
// unit-testable seam the image decode path uses to reach those bytes.

// Percent-decode a URI component: every %XX (two hex digits) becomes the byte 0xXX (glTF permits
// e.g. "my%20texture.png" for "my texture.png"). A malformed escape (truncated or non-hex) is copied
// through verbatim rather than dropped. Pure; no filesystem access.
std::string PercentDecodeUri(const char* uri);

// Resolve a RELATIVE file URI against the directory containing `gltfPath` (the .gltf/.glb file) and
// read the whole file's bytes. Returns an empty vector when the URI is not a plain relative file
// reference (data: URIs — cgltf materialises embedded-buffer data: URIs itself; scheme'd URIs like
// http:// are unsupported) or the file cannot be opened/read; callers fall back gracefully.
std::vector<uint8_t> ReadUriBytesRelativeTo(const char* gltfPath, const char* uri);

// ---------------------------------------------------------------------------------------------------
// PURE alpha-mode resolution (Slice SC1b — alpha-mask cutouts). Maps a glTF material's alphaMode +
// alphaCutoff onto the engine's single alpha-TEST material model:
//   OPAQUE (0) -> {masked=false, cutoff=0}   (no test; cutoff 0 also never discards in the shader)
//   MASK   (1) -> {masked=true,  cutoff=authored alphaCutoff}   (spec default 0.5)
//   BLEND  (2) -> {masked=true,  cutoff=0.5}   DOCUMENTED v1 APPROXIMATION: true alpha blending
//                 needs a sorted transparent pass / OIT (the engine has --oit but wiring it into the
//                 hero path is out of scope); a 0.5 alpha test keeps blend-authored foliage/decals
//                 from rendering as opaque cards, which is the visible SC1 artifact.
// `alphaMode` uses the cgltf_alpha_mode numbering (0=opaque, 1=mask, 2=blend); anything else is
// treated as opaque. Pure; unit-tested in gltf_uri_image_test.
struct AlphaMaskInfo {
    bool  masked = false;
    float cutoff = 0.0f;
};
AlphaMaskInfo ResolveAlphaMode(int alphaMode, float alphaCutoff);

// ---------------------------------------------------------------------------------------------------
// PURE deterministic RGBA8 mip-chain generation (Slice SC1b — texture mipmaps). The full mip count
// for a w x h image (floor(log2(max(w,h,1))) + 1; a 1x1 image has 1 level).
uint32_t FullMipCount(uint32_t width, uint32_t height);

// Build mip levels 1..FullMipCount-1 from the tightly-packed RGBA8 level 0 (level i is
// max(1, width>>i) x max(1, height>>i)). Each destination texel is the integer-rounded average of
// the (clamped) 2x2 source block — pure int arithmetic, so the chain is BIT-IDENTICAL on every
// platform/backend (the CPU chain feeds rhi::TextureDesc::mipData on BOTH Vulkan and Metal, unlike
// a GPU blit whose filtering is vendor-defined). Odd source dimensions clamp the +1 sample to the
// edge (each destination still averages exactly 4 samples). Returns an empty vector for a 1x1
// image (no chain). NOTE: byte-space averaging (no sRGB linearization, no normal renormalization) —
// the standard fast path; documented v1 fidelity tradeoff.
std::vector<std::vector<uint8_t>> BuildRgba8MipChain(const uint8_t* mip0,
                                                     uint32_t width, uint32_t height);

// ---------------------------------------------------------------------------------------------------
// Pure scene-graph hierarchy composition (no device, no cgltf) — factored out so it can be unit
// tested in isolation. A SceneNode is a plain node: a local transform (already resolved from a
// glTF node's `matrix` or its TRS) plus child indices.
struct SceneNode {
    math::Mat4 local = math::Mat4::Identity();
    std::vector<int> children;   // indices into the node array
};

// Compose a child's world transform from its parent's world and its own local: world = parent * local.
// (Column-major RH math::Mat4; the same product order used throughout the engine.)
math::Mat4 ComposeWorld(const math::Mat4& parentWorld, const math::Mat4& local);

// Depth-first walk: for each node reachable from `roots`, compute its world transform (parent*local)
// and invoke `visit(nodeIndex, world)`. Pure: operates on plain SceneNode arrays so it can be tested
// without a glTF file or a device. Cycles are guarded against via a visited set.
void WalkHierarchy(const std::vector<SceneNode>& nodes, const std::vector<int>& roots,
                   const std::function<void(int, const math::Mat4&)>& visit);


// Load the first primitive of the first mesh of a glTF/glb file into a scene::Mesh.
//
// GEOMETRY ONLY (legacy): reads POSITION / NORMAL / TEXCOORD_0 into the engine's
// scene::Vertex layout and the index buffer (u16 or u32, widened to u32). glTF
// materials/textures/animation are ignored — the caller renders the returned mesh
// with a fixed material.
//
// The returned geometry is recentred so its bounding box centre sits at the origin,
// so the caller only needs a uniform scale + translate to place it. Vertex colour is
// set to a neutral tint ({0.8, 0.8, 0.85}); UV defaults to (0,0) if TEXCOORD_0 is absent.
//
// Throws std::runtime_error on parse/load/validation failure or missing POSITION.
scene::Mesh LoadGltfMesh(rhi::IRHIDevice& device, const char* path);

// Geometry + base-color material loaded from a glTF/glb file.
//
//  * mesh      — same geometry as LoadGltfMesh (first primitive, recentred to origin).
//  * baseColor — an RGBA8 texture decoded from the material's base-color texture. For a
//                .glb the image is embedded in a buffer_view; the bytes are decoded with
//                stb_image. If the material/texture is absent or cannot be decoded, this
//                falls back to a flat-white 1x1 texture (so the caller always has a valid
//                texture to bind).
//  * metallic  — pbr_metallic_roughness.metallic_factor (defaults to 1.0 per glTF spec).
//  * roughness — pbr_metallic_roughness.roughness_factor (defaults to 1.0 per glTF spec).
struct GltfModel {
    scene::Mesh mesh;
    std::unique_ptr<rhi::ITexture> baseColor;
    float metallic = 1.0f;
    float roughness = 1.0f;

    GltfModel(scene::Mesh m, std::unique_ptr<rhi::ITexture> tex, float met, float rough)
        : mesh(std::move(m)), baseColor(std::move(tex)), metallic(met), roughness(rough) {}
};

// Load geometry + base-color texture + metallic/roughness factors. See GltfModel.
// Throws std::runtime_error on parse/load/validation failure or missing POSITION.
GltfModel LoadGltfModel(rhi::IRHIDevice& device, const char* path);

// A skinned glTF model: skinned-vertex geometry + base-color material + a skeleton + animations.
//
//  * mesh       — first primitive uploaded with the scene::SkinnedVertex layout (stride 88). Reads
//                 POSITION / TEXCOORD_0 / JOINTS_0 / WEIGHTS_0. JOINTS_0 (u8/u16) is widened to
//                 float and REMAPPED from glTF skin-joint order to the skeleton's topologically
//                 sorted order. WEIGHTS_0 is renormalized so each vertex's four weights sum to 1.
//                 NORMAL is computed as smooth per-vertex normals from the indexed positions when
//                 absent (the Fox has none); tangent defaults to (1,0,0). NOT recentred — the skin's
//                 inverse-bind matrices define the bind-space origin, so the caller places the mesh
//                 with an explicit model matrix (see `bbMin`/`bbMax`).
//  * baseColor  — RGBA8 base-color texture (same decode path as GltfModel; white fallback).
//  * metallic / roughness — material factors.
//  * skeleton   — joints (parent, inverse-bind, rest TRS) in topologically sorted order.
//  * animations — every animation clip in the file, channels remapped to skeleton joint indices.
//  * bbMin / bbMax — model-space bounding box of the (un-recentred) bind-pose positions, so the
//                 caller can ground-align + scale the placement.
struct SkinnedModel {
    scene::Mesh mesh;
    std::unique_ptr<rhi::ITexture> baseColor;
    float metallic = 1.0f;
    float roughness = 1.0f;
    anim::Skeleton skeleton;
    std::vector<anim::Animation> animations;
    float bbMin[3] = {0, 0, 0};
    float bbMax[3] = {0, 0, 0};

    SkinnedModel(scene::Mesh m, std::unique_ptr<rhi::ITexture> tex, float met, float rough,
                 anim::Skeleton skel, std::vector<anim::Animation> anims,
                 const float bbmin[3], const float bbmax[3])
        : mesh(std::move(m)), baseColor(std::move(tex)), metallic(met), roughness(rough),
          skeleton(std::move(skel)), animations(std::move(anims)) {
        for (int k = 0; k < 3; ++k) { bbMin[k] = bbmin[k]; bbMax[k] = bbmax[k]; }
    }

    // Find an animation by name; returns nullptr if not present.
    const anim::Animation* FindAnimation(const char* name) const;
};

// Load a skinned glTF/glb model (skin + first animation set). See SkinnedModel.
// Throws std::runtime_error on parse/load/validation failure, missing POSITION, or missing skin.
SkinnedModel LoadSkinnedGltfModel(rhi::IRHIDevice& device, const char* path);

// Geometry + a FULL glTF metallic-roughness PBR material loaded from a glTF/glb file.
//
//  * mesh       — same geometry as LoadGltfMesh (first primitive, recentred to origin). Tangents are
//                 the authored TANGENT if present, else computed from POSITION/UV/NORMAL.
//  * baseColor  — sRGB albedo (RGBA8). White 1x1 fallback when absent.
//  * metalRough — glTF packing: G channel = roughness, B channel = metallic (linear). A 1x1
//                 (255,255,0,255) neutral fallback (rough=1, metallic=0) when absent.
//  * normalMap  — tangent-space normal (linear). Flat (128,128,255,255) -> (0,0,1) fallback.
//  * emissive   — sRGB emissive. Black 1x1 fallback (adds nothing) when absent.
//  * occlusion  — R channel = ambient occlusion (linear). White 1x1 fallback (no occlusion).
//  * metallicFactor / roughnessFactor — pbr_metallic_roughness factors (default 1.0).
//  * emissiveFactor[3] — material emissive_factor (default 0,0,0).
//
// Every texture is non-null so the lit-PBR shader can always bind five textures.
struct PbrModel {
    scene::Mesh mesh;
    std::unique_ptr<rhi::ITexture> baseColor;
    std::unique_ptr<rhi::ITexture> metalRough;
    std::unique_ptr<rhi::ITexture> normalMap;
    std::unique_ptr<rhi::ITexture> emissive;
    std::unique_ptr<rhi::ITexture> occlusion;
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    float emissiveFactor[3] = {0.0f, 0.0f, 0.0f};

    PbrModel(scene::Mesh m,
             std::unique_ptr<rhi::ITexture> base, std::unique_ptr<rhi::ITexture> mr,
             std::unique_ptr<rhi::ITexture> nrm, std::unique_ptr<rhi::ITexture> emis,
             std::unique_ptr<rhi::ITexture> occ, float met, float rough, const float emisF[3])
        : mesh(std::move(m)), baseColor(std::move(base)), metalRough(std::move(mr)),
          normalMap(std::move(nrm)), emissive(std::move(emis)), occlusion(std::move(occ)),
          metallicFactor(met), roughnessFactor(rough) {
        for (int k = 0; k < 3; ++k) emissiveFactor[k] = emisF[k];
    }
};

// Load geometry + the full glTF metallic-roughness PBR material (5 textures + factors). See PbrModel.
// Throws std::runtime_error on parse/load/validation failure or missing POSITION.
PbrModel LoadPbrGltfModel(rhi::IRHIDevice& device, const char* path);

// ============================== Full scene-graph import (Slice V) ===============================

// A decoded, shareable PBR material: the five RGBA8 textures (every one non-null, with neutral 1x1
// fallbacks) + the metallic/roughness/emissive factors. Owned by the GltfScene material table; many
// SceneInstances may point at the same PbrMaterial (glTF materials are deduped on decode).
struct PbrMaterial {
    std::unique_ptr<rhi::ITexture> baseColor;
    std::unique_ptr<rhi::ITexture> metalRough;
    std::unique_ptr<rhi::ITexture> normalMap;
    std::unique_ptr<rhi::ITexture> emissive;
    std::unique_ptr<rhi::ITexture> occlusion;
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    float emissiveFactor[3] = {0.0f, 0.0f, 0.0f};
    // Slice SC1b: glTF alphaMode resolved via ResolveAlphaMode. When alphaMasked the draw should use
    // an alpha-testing pipeline (e.g. lit_cutout.vert + lit_pbr_cutout.frag) with alphaCutoff packed
    // into the per-draw material push constant .w (the #38 packing convention: x=metallic,
    // y=roughness, z=albedo tint, w=alpha cutoff; 0 = no test). Opaque draws ignore both fields, so
    // every pre-SC1b consumer is unchanged.
    bool  alphaMasked = false;
    float alphaCutoff = 0.0f;
};

// One renderable: a (non-owning) mesh + material + the node's composed world transform. The same
// mesh referenced by multiple nodes yields multiple instances at different worldTransforms (e.g. the
// CesiumMilkTruck's single wheels mesh placed at the front and back via two parent transforms).
struct SceneInstance {
    const scene::Mesh* mesh = nullptr;        // into GltfScene::meshStorage
    const PbrMaterial* material = nullptr;     // into GltfScene::materialStorage
    math::Mat4 worldTransform = math::Mat4::Identity();
};

// A fully imported glTF scene: owning storage for the unique meshes + materials, the flat list of
// renderable instances (one per primitive of every mesh-referencing node, in DFS order), and the
// scene-wide world-space AABB over all instances' transformed geometry bounds.
struct GltfScene {
    std::vector<std::unique_ptr<scene::Mesh>> meshStorage;       // unique (mesh,prim) GPU meshes
    std::vector<std::unique_ptr<PbrMaterial>> materialStorage;   // deduped materials (+ a default)
    std::vector<SceneInstance> instances;
    float bbMin[3] = {0, 0, 0};   // world-space scene AABB (over all instances)
    float bbMax[3] = {0, 0, 0};
    // Number of texture definitions in the source glTF (data->textures_count) — a load-proof stat
    // for hero scenes (Slice SC1). 0 for scenes without textures.
    size_t textureCount = 0;

    // A uniform-scale + translate that fits the whole scene to a cube of side `targetSize` and sets
    // it down so its min-Y rests at `groundY` and its XZ centre is at the origin. Pure: does NOT
    // mutate geometry; the caller pre-multiplies this onto each instance's world transform. Returns
    // identity for a degenerate (empty / zero-extent) scene.
    math::Mat4 FitTransform(float targetSize, float groundY) const;
};

// Optional scene-import behaviours (Slice SC1b). The default-constructed value reproduces the
// pre-SC1b behaviour EXACTLY (mip-0-only texture uploads), so every existing caller/golden is
// byte-identical; hero paths opt in explicitly.
struct GltfLoadOptions {
    // Generate full deterministic CPU RGBA8 mip chains (BuildRgba8MipChain) for every decoded
    // image texture and upload them via the existing rhi::TextureDesc::mipData path (trilinear
    // sampling comes from the default samplers, which already carry full-LOD linear mip filtering
    // in both backends; 1-mip textures keep sampling exactly as before). 1x1 fallback textures are
    // unaffected (no chain exists).
    bool generateMipmaps = false;
};

// Walk the default scene's node hierarchy depth-first, composing world = parent*local for each node,
// and emit one SceneInstance per primitive of every mesh-referencing node. glTF materials are decoded
// once and shared; the same (mesh,primitive) geometry is uploaded once and shared across instances.
// Throws std::runtime_error on parse/load/validation failure or missing POSITION.
GltfScene LoadGltfScene(rhi::IRHIDevice& device, const char* path);
// Same, with explicit options (SC1b). LoadGltfScene(device, path) == LoadGltfScene(device, path, {}).
GltfScene LoadGltfScene(rhi::IRHIDevice& device, const char* path, const GltfLoadOptions& opts);

} // namespace hf::asset
