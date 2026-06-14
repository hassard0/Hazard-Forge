#include "hal/window.h"
#include "rhi/rhi.h"
#include "rhi/rhi_factory.h"
#include <cstdint>
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

        // Exercise the staging-upload + descriptor alloc/free path headlessly.
        {
            const uint8_t pixels[16] = {
                255, 0, 0, 255,   0, 255, 0, 255,
                0, 0, 255, 255,   255, 255, 0, 255,
            };
            auto tex = device->CreateTexture(
                {2, 2, rhi::Format::RGBA8_UNorm, pixels, 16});
            if (!tex) {
                std::fprintf(stderr, "CreateTexture returned null\n");
                return 1;
            }
        }  // tex destroyed here (frees its descriptor set) before WaitIdle.

        device->WaitIdle();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "smoke FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("smoke OK\n");
    return 0;
}
