#pragma once
#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include "rhi/rhi.h"
#include "rhi_vulkan/vulkan_swapchain.h"
#include "hal/window.h"
#include <memory>

namespace hf::rhi::vk {

// Frames-in-flight; double-buffered.
constexpr uint32_t kFramesInFlight = 2;

class VulkanDevice final : public IRHIDevice {
public:
    explicit VulkanDevice(hf::hal::Window& window);
    ~VulkanDevice() override;

    ISwapchain& Swapchain() override { return *swapchain_; }

    std::unique_ptr<IShaderModule> CreateShaderModule(const ShaderModuleDesc&) override;
    std::unique_ptr<IPipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc&) override;
    std::unique_ptr<IBuffer> CreateBuffer(const BufferDesc&) override;
    std::unique_ptr<ITexture> CreateTexture(const TextureDesc&) override;

    FrameContext BeginFrame() override;
    void EndFrame(const FrameContext&) override;
    void WaitIdle() override;

    // Accessors used by sibling Vulkan objects.
    VkDevice device() const { return device_; }
    VmaAllocator allocator() const { return allocator_; }
    VkSampler defaultSampler() const { return defaultSampler_; }
    VkDescriptorSetLayout texturedSetLayout() const { return texturedSetLayout_; }
    VkDescriptorPool descriptorPool() const { return descriptorPool_; }

    // Stage host pixel data into a device-local image (synchronous; see vulkan_texture).
    void UploadToImage(VkImage image, uint32_t w, uint32_t h,
                       const void* data, uint64_t size);

private:
    void CreateSyncObjects();
    void DestroySyncObjects();
    void CreateTextureResources();
    void DestroyTextureResources();

    hf::hal::Window& window_;

    vkb::Instance vkbInstance_{};
    vkb::Device   vkbDevice_{};
    VkSurfaceKHR  surface_ = VK_NULL_HANDLE;
    VkInstance    instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice      device_ = VK_NULL_HANDLE;
    VkQueue       graphicsQueue_ = VK_NULL_HANDLE;
    uint32_t      graphicsQueueFamily_ = 0;
    VmaAllocator  allocator_ = VK_NULL_HANDLE;

    // Texture support: one default sampler + a shared combined-image-sampler set layout + pool.
    VkSampler             defaultSampler_    = VK_NULL_HANDLE;
    VkDescriptorSetLayout texturedSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool      descriptorPool_    = VK_NULL_HANDLE;

    std::unique_ptr<VulkanSwapchain> swapchain_;

    // Per-frame-in-flight sync + command resources.
    struct FrameSync {
        VkCommandPool   pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd  = VK_NULL_HANDLE;
        VkSemaphore     imageAvailable = VK_NULL_HANDLE;
        VkSemaphore     renderFinished = VK_NULL_HANDLE;
        VkFence         inFlight = VK_NULL_HANDLE;
    };
    FrameSync frames_[kFramesInFlight];
    uint32_t  frameIndex_ = 0;
    uint32_t  acquiredImage_ = 0;

    std::unique_ptr<class VulkanCommandBuffer> recorder_;
};

} // namespace hf::rhi::vk
