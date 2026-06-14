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
    std::unique_ptr<IRenderTarget> CreateRenderTarget(uint32_t width, uint32_t height) override;

    FrameContext BeginRenderTargetFrame(IRenderTarget& rt) override;
    void EndRenderTargetFrame(const FrameContext&) override;

    FrameContext BeginFrame() override;
    void EndFrame(const FrameContext&) override;
    void SetFrameUniforms(const void* data, uint32_t size) override;
    void WaitIdle() override;
    void CaptureNextFrame() override { captureArmed_ = true; }
    bool GetCapturedPixels(std::vector<uint8_t>& outBGRA,
                           uint32_t& width, uint32_t& height) override;

    // Accessors used by sibling Vulkan objects.
    VkDevice device() const { return device_; }
    VmaAllocator allocator() const { return allocator_; }
    VkSampler defaultSampler() const { return defaultSampler_; }
    VkDescriptorSetLayout texturedSetLayout() const { return texturedSetLayout_; }
    // The material set layout logically lives at set 1 (set index decided by the pipeline
    // layout array); the layout object is identical to the Slice C textured set layout.
    VkDescriptorSetLayout materialSetLayout() const { return texturedSetLayout_; }
    VkDescriptorPool descriptorPool() const { return descriptorPool_; }
    // Per-frame UBO set (set 0). currentFrameSet() returns the set for the frame being recorded.
    VkDescriptorSetLayout frameSetLayout() const { return frameSetLayout_; }
    VkDescriptorSet currentFrameSet() const { return frameSet_[frameIndex_]; }
    // Swapchain color format — render targets match it so the lit pipeline renders in unchanged.
    VkFormat swapchainFormat() const { return swapchain_->vkFormat(); }

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

    // Texture support: one default sampler + a shared image+sampler set layout (set 1) + pool.
    VkSampler             defaultSampler_    = VK_NULL_HANDLE;
    VkDescriptorSetLayout texturedSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool      descriptorPool_    = VK_NULL_HANDLE;

    // Per-frame uniform buffers (set 0): one host-visible mapped UBO + descriptor set per
    // frame-in-flight, so the CPU never writes a UBO the GPU is still reading.
    static constexpr uint32_t kFrameUboSize = 256;  // >= sizeof(FrameData) (112B), aligned
    VkDescriptorSetLayout frameSetLayout_ = VK_NULL_HANDLE;
    VkBuffer        uboBuffer_[kFramesInFlight]{};
    VmaAllocation   uboAlloc_[kFramesInFlight]{};
    void*           uboMapped_[kFramesInFlight]{};
    VkDescriptorSet frameSet_[kFramesInFlight]{};

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

    // Headless capture state. When captureArmed_ is set, EndFrame copies the rendered
    // swapchain image back to host memory (capturedBGRA_) instead of presenting.
    bool                 captureArmed_ = false;
    std::vector<uint8_t> capturedBGRA_;
    uint32_t             capW_ = 0, capH_ = 0;

    std::unique_ptr<class VulkanCommandBuffer> recorder_;

    // Dedicated command resources for the offscreen render-target pass. Independent of the
    // per-frame swapchain sync: BeginRenderTargetFrame records here, EndRenderTargetFrame
    // submits + waits on rtFence_ before the swapchain pass samples the RT.
    VkCommandPool   rtPool_  = VK_NULL_HANDLE;
    VkCommandBuffer rtCmd_   = VK_NULL_HANDLE;
    VkFence         rtFence_ = VK_NULL_HANDLE;
    std::unique_ptr<class VulkanCommandBuffer> rtRecorder_;
    class VulkanRenderTarget* rtInFlight_ = nullptr;  // RT being recorded between Begin/End
};

} // namespace hf::rhi::vk
