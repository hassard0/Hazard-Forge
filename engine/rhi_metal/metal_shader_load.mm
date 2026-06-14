#include "rhi_metal/metal_shader_load.h"
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_common.h"

namespace hf::rhi::mtl {

std::unique_ptr<IShaderModule> MakeShaderModuleFromMSL(IRHIDevice& device,
                                                       const std::string& source,
                                                       const std::string& entryPoint) {
    auto& md = static_cast<MetalDevice&>(device);
    return md.CreateShaderModuleMSL(source, entryPoint);
}

} // namespace hf::rhi::mtl
