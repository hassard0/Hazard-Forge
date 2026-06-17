#pragma once
// Shared Metal<->RHI helpers. Objective-C++ only (.mm). NEVER include from engine/rhi/ or
// engine/scene/ — keeps the RHI seam free of Metal/Obj-C types.
#import <Metal/Metal.h>
#include "rhi/rhi.h"
#include <stdexcept>
#include <string>

namespace hf::rhi::mtl {

// Binding-index convention. These flat Metal [[buffer/texture/sampler(n)]] indices are produced by
// `spirv-cross --msl --msl-decoration-binding` from the shared HLSL sources (HLSL -> SPIR-V ->
// MSL); with --msl-decoration-binding each resource's Metal index == its SPIR-V binding number, so
// the HLSL [[vk::binding]] / register() values (and, for the MSL-gen path, the HF_MSL_GEN-guarded
// overrides) are chosen to land here:
//   vertex:   buffer(0) = vertex data (stage_in)
//             buffer(1) = per-frame UBO (FrameData)   <- HF_MSL_GEN Frame at vk::binding(1,0)
//             buffer(2) = push constants (model matrix) <- HF_MSL_GEN PushC at vk::binding(2,0)
//   fragment: buffer(0) = per-frame UBO (FrameData)   <- Frame at vk::binding(0,0)
//             texture(0) = material (gTex, set1 b0);   sampler(1) = material sampler (gSmp, set1 b1)
//             texture(1) = shadow map (gShadow, b1);   sampler(2) = shadow sampler (gShadowSmp, b2)
//             texture(3) = normal map (gNormalMap, set1 b3); sampler(4) = normal sampler (gNormalSmp, set1 b4)
// Textures map 1:1 to their binding; samplers carry the binding number too, so the material sampler
// lands at 1 and the shadow sampler at 2 (one higher than the hand-written MSL used). The engine
// binds at these indices; the generated MSL declares them.
constexpr uint32_t kVbVertex      = 0;
constexpr uint32_t kVbFrameUbo    = 1;
constexpr uint32_t kVbPushConst   = 2;
// Skinning joint-palette UBO (set 2 b0 in HLSL -> [[vk::binding(3,2)]] so spirv-cross
// --msl-decoration-binding lands it at vertex buffer(3), past the vertex/frame/pushconst slots).
constexpr uint32_t kVbJointPalette = 3;
// Per-instance vertex stream (binding 1 in the RHI / Vulkan). Flat Metal vertex-buffer slot 4, past
// vertex(0)/frameUbo(1)/pushConst(2)/jointPalette(3). The instanced lit MSL reads its mat4 transform
// attributes (locations 7-10) from this stage_in buffer with a per-instance step function.
constexpr uint32_t kVbInstance    = 4;
constexpr uint32_t kFbFrameUbo    = 0;
// Bloom per-pass params (Slice U): a fragment-stage push constant. The bloom fullscreen shaders
// declare it as a HF_MSL_GEN cbuffer at [[vk::binding(1,0)]] so spirv-cross --msl-decoration-binding
// lands it on fragment buffer(1); MetalCommandBuffer::PushConstants sets it here for those pipelines.
constexpr uint32_t kFbPushConst   = 1;
constexpr uint32_t kFragTexture   = 0;
constexpr uint32_t kFragSampler   = 1;
constexpr uint32_t kFragShadowTex = 1;
constexpr uint32_t kFragShadowSmp = 2;
constexpr uint32_t kFragNormalTex = 3;  // gNormalMap (set1 b3 -> texture(3))
constexpr uint32_t kFragNormalSmp = 4;  // gNormalSmp (set1 b4 -> sampler(4))
// Full-PBR material set (Slice P): metallic-roughness / emissive / occlusion at the next flat
// fragment indices. The lit_pbr.frag HLSL declares these at set1 b5..b10, so
// spirv-cross --msl-decoration-binding lands them on texture(5/7/9) / sampler(6/8/10).
constexpr uint32_t kFragMetalRoughTex = 5;  // gMetalRough  (set1 b5  -> texture(5))
constexpr uint32_t kFragMetalRoughSmp = 6;  // gMetalRoughSmp (set1 b6 -> sampler(6))
constexpr uint32_t kFragEmissiveTex   = 7;  // gEmissive    (set1 b7  -> texture(7))
constexpr uint32_t kFragEmissiveSmp   = 8;  // gEmissiveSmp (set1 b8  -> sampler(8))
constexpr uint32_t kFragOcclusionTex  = 9;  // gOcclusion   (set1 b9  -> texture(9))
constexpr uint32_t kFragOcclusionSmp  = 10; // gOcclusionSmp (set1 b10 -> sampler(10))
// HDR environment map (Slice R): a dedicated equirect texture + trilinear sampler for image-based
// lighting, past the full-PBR material's 0..10. The sky_hdr.frag / lit_pbr_ibl.frag HLSL declare
// these at [[vk::binding(11,3)]]/[[vk::binding(12,3)]] so spirv-cross --msl-decoration-binding lands
// them on Metal fragment texture(11) / sampler(12).
constexpr uint32_t kFragEnvTex = 11;  // gEnv    (set3 b11 -> texture(11))
constexpr uint32_t kFragEnvSmp = 12;  // gEnvSmp (set3 b12 -> sampler(12))
// Clustered-lighting storage buffers (Slice AG): three fragment-stage STORAGE buffers bound via
// setFragmentBuffer:atIndex:. The lit_clustered.frag HLSL declares them at [[vk::binding(13/14/15,3)]]
// so spirv-cross --msl-decoration-binding lands them on Metal FRAGMENT BUFFER slots 13/14/15 (the
// buffer index space, separate from texture/sampler — never collides with the env's texture(11/12)
// nor the fragment FrameData buffer(0) / bloom push-const buffer(1)).
constexpr uint32_t kFragClusterBuf      = 13;  // gClusters     (set3 b13 -> buffer(13))
constexpr uint32_t kFragLightIndexBuf   = 14;  // gLightIndices (set3 b14 -> buffer(14))
constexpr uint32_t kFragLightBuf        = 15;  // gLights       (set3 b15 -> buffer(15))

// Compute (kernel) binding indices. The particle compute MSL is generated from particles.comp.hlsl
// via spirv-cross --msl-decoration-binding: the storage buffer (binding 0) -> buffer(0), and the
// HF_MSL_GEN params cbuffer (binding 1) -> buffer(1).
constexpr uint32_t kCsStorage   = 0;  // RWStructuredBuffer (particle SSBO)
constexpr uint32_t kCsPushConst = 1;  // params cbuffer (dt/time/count)
// Slice CX: the froxel inject's sun CSM shadow map (Texture2D binding 4) + sampler (binding 5). With
// spirv-cross --msl-decoration-binding the SPIR-V bindings map straight to the compute texture(4) /
// sampler(5) slots (a separate index space from the storage buffers). Bound via BindShadowMapCompute.
constexpr uint32_t kCsShadowTex = 4;  // gSunShadow    (binding 4 -> texture(4))
constexpr uint32_t kCsShadowSmp = 5;  // gSunShadowSmp (binding 5 -> sampler(5))

// Map an RHI vertex-attribute / color / depth format to a Metal pixel/vertex format.
inline MTLPixelFormat ToMetalPixelFormat(Format f) {
    switch (f) {
        case Format::RGBA8_UNorm: return MTLPixelFormatRGBA8Unorm;
        case Format::BGRA8_UNorm: return MTLPixelFormatBGRA8Unorm;
        case Format::RGBA16_Float: return MTLPixelFormatRGBA16Float;
        // RGBA32_Float renderable/blendable target (Slice CO: the WBOIT accum + revealage targets use
        // full fp32 so the order-independent accum SUM is bit-identical across draw orders). Mirrors the
        // Vulkan VK_FORMAT_R32G32B32A32_SFLOAT mapping; backend-dir only, no above-seam symbol.
        case Format::RGBA32_Float: return MTLPixelFormatRGBA32Float;
        // R32_Uint single-channel integer visibility-buffer target (Slice DW). Mirrors the Vulkan
        // VK_FORMAT_R32_UINT mapping; backend-dir only, no above-seam symbol.
        case Format::R32_Uint:    return MTLPixelFormatR32Uint;
        case Format::D32_Float:   return MTLPixelFormatDepth32Float;
        default:                  return MTLPixelFormatInvalid;
    }
}

inline MTLVertexFormat ToMetalVertexFormat(Format f) {
    switch (f) {
        case Format::RG32_Float:   return MTLVertexFormatFloat2;
        case Format::RGB32_Float:  return MTLVertexFormatFloat3;
        case Format::RGBA32_Float: return MTLVertexFormatFloat4;
        case Format::RGBA8_UNorm:  return MTLVertexFormatUChar4Normalized;
        default:                  return MTLVertexFormatInvalid;
    }
}

inline Format FromMetalPixelFormat(MTLPixelFormat f) {
    switch (f) {
        case MTLPixelFormatRGBA8Unorm: return Format::RGBA8_UNorm;
        case MTLPixelFormatBGRA8Unorm: return Format::BGRA8_UNorm;
        default:                       return Format::Undefined;
    }
}

[[noreturn]] inline void Fail(const std::string& what) {
    throw std::runtime_error("Metal error: " + what);
}

} // namespace hf::rhi::mtl
