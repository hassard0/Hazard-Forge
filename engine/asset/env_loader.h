#pragma once
#include "rhi/rhi.h"
#include <memory>

namespace hf::asset {

// An HDR equirectangular environment map (Slice R): a single 2D RGBA16F texture with a
// CPU-prefiltered mip chain. Mip 0 is the full-resolution sky; each coarser mip is a 2x2
// box-downsample (progressively blurrier → approximates a roughness-prefiltered specular
// environment). The lit_pbr_ibl shader samples mip = roughness*maxLod for specular and a very high
// mip for diffuse irradiance; the sky_hdr shader samples mip 0 for the background.
struct EnvironmentMap {
    std::unique_ptr<rhi::ITexture> equirect;
    int mipLevels = 0;
    int width = 0;
    int height = 0;
};

// Load an equirectangular .hdr file, decode it to linear float RGB with stb_image (stbi_loadf),
// build the box-downsampled mip chain on the CPU, convert each mip to RGBA16F, and upload it as an
// N-mip sampled texture through device.CreateTexture (desc.environment = true). The returned
// texture's maxLod = mipLevels - 1 (pass roughness*maxLod / maxLod-1 to the IBL shader).
//
// Throws std::runtime_error if the file cannot be opened or decoded.
EnvironmentMap LoadHdrEnvironment(rhi::IRHIDevice& device, const char* path);

} // namespace hf::asset
