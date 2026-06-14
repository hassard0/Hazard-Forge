#include "hal/window.h"
#include "rhi/rhi.h"
#include "rhi/rhi_factory.h"
#include <cstdio>

int main() {
    using namespace hf;
    try {
        hal::Window window({"hf-smoke", 64, 64});
        auto device = rhi::CreateDevice(rhi::Backend::Vulkan, window);
        if (device->Swapchain().ColorFormat() == rhi::Format::Undefined) {
            std::fprintf(stderr, "swapchain has undefined format\n");
            return 1;
        }
        device->WaitIdle();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "smoke FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("smoke OK\n");
    return 0;
}
