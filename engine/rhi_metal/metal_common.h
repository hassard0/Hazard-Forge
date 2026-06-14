#pragma once
// Shared Metal<->RHI helpers. Objective-C++ only (.mm). NEVER include from engine/rhi/ or
// engine/scene/ — keeps the RHI seam free of Metal/Obj-C types.
#import <Metal/Metal.h>
#include "rhi/rhi.h"
#include <stdexcept>
#include <string>

namespace hf::rhi::mtl {

// Binding-index convention (must match the hand-written MSL in shaders/lit.metal):
//   vertex:   buffer(0) = vertex data
//             buffer(1) = per-frame UBO (FrameData)
//             buffer(2) = push constants (model matrix), via setVertexBytes
//   fragment: buffer(0) = per-frame UBO (FrameData)
//             texture(0)/sampler(0) = material
//             texture(1)/sampler(1) = shadow map (depth) + its clamp-to-edge sampler
constexpr uint32_t kVbVertex      = 0;
constexpr uint32_t kVbFrameUbo    = 1;
constexpr uint32_t kVbPushConst   = 2;
constexpr uint32_t kFbFrameUbo    = 0;
constexpr uint32_t kFragTexture   = 0;
constexpr uint32_t kFragSampler   = 0;
constexpr uint32_t kFragShadowTex = 1;
constexpr uint32_t kFragShadowSmp = 1;

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
