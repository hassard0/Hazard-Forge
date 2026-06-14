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
// Textures map 1:1 to their binding; samplers carry the binding number too, so the material sampler
// lands at 1 and the shadow sampler at 2 (one higher than the hand-written MSL used). The engine
// binds at these indices; the generated MSL declares them.
constexpr uint32_t kVbVertex      = 0;
constexpr uint32_t kVbFrameUbo    = 1;
constexpr uint32_t kVbPushConst   = 2;
constexpr uint32_t kFbFrameUbo    = 0;
constexpr uint32_t kFragTexture   = 0;
constexpr uint32_t kFragSampler   = 1;
constexpr uint32_t kFragShadowTex = 1;
constexpr uint32_t kFragShadowSmp = 2;

// Map an RHI vertex-attribute / color / depth format to a Metal pixel/vertex format.
inline MTLPixelFormat ToMetalPixelFormat(Format f) {
    switch (f) {
        case Format::RGBA8_UNorm: return MTLPixelFormatRGBA8Unorm;
        case Format::BGRA8_UNorm: return MTLPixelFormatBGRA8Unorm;
        case Format::D32_Float:   return MTLPixelFormatDepth32Float;
        default:                  return MTLPixelFormatInvalid;
    }
}

inline MTLVertexFormat ToMetalVertexFormat(Format f) {
    switch (f) {
        case Format::RG32_Float:  return MTLVertexFormatFloat2;
        case Format::RGB32_Float: return MTLVertexFormatFloat3;
        case Format::RGBA8_UNorm: return MTLVertexFormatUChar4Normalized;
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
