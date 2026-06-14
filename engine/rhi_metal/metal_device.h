#pragma once
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <dispatch/dispatch.h>  // dispatch_semaphore_t for CPU/GPU frame pacing
#include "rhi/rhi.h"
#include "rhi_metal/metal_swapchain.h"
#include <memory>
#include <string>

// NOTE: this header deliberately does NOT include hal/window.h. The windowed path takes a
// Window& (forward-declared via rhi.h) and only the .mm touches its concrete API; the headless
// path takes width/height and never references Window at all. This keeps the Metal backend
// usable in a window-server-less (SSH) session.
namespace hf::hal { class Window; }

namespace hf::rhi::mtl {

// Frames-in-flight; double-buffered (mirrors the Vulkan backend's kFramesInFlight).
constexpr uint32_t kFramesInFlight = 2;

class MetalCommandBuffer;
class MetalRenderTarget;

class MetalDevice final : public IRHIDevice {
public:
    // Windowed: render into the window's CAMetalLayer drawable and present each frame.
    explicit MetalDevice(hf::hal::Window& window);
    // Headless: render into an offscreen MTLTexture (no window/CAMetalLayer/present). Read the
    // result back via CaptureNextFrame() + GetCapturedPixels(). For SSH / CI golden-image runs.
    MetalDevice(uint32_t width, uint32_t height);
    ~MetalDevice() override;

    ISwapchain& Swapchain() override { return *swapchain_; }

    // SPIR-V is not consumable by Metal. The Metal sample compiles MSL via the metal-only
    // CreateShaderModuleMSL() below; this throws to make the misuse loud.
    std::unique_ptr<IShaderModule> CreateShaderModule(const ShaderModuleDesc&) override;
    std::unique_ptr<IPipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc&) override;
    std::unique_ptr<IBuffer> CreateBuffer(const BufferDesc&) override;
    std::unique_ptr<ITexture> CreateTexture(const TextureDesc&) override;

    // Metal-only shader entry: compile MSL source at runtime. Not on IRHIDevice (keeps the RHI
    // seam SPIR-V-shaped); the sample calls this under __APPLE__.
    std::unique_ptr<IShaderModule> CreateShaderModuleMSL(const std::string& source,
                                                         const std::string& entryPoint);

    // Render-targets + shadows slices (master IRHIDevice). Not yet implemented on Metal: these
    // throw. The headless Slice-F scene path does not use them, so the cube/scene render is
    // unaffected. Implementing them is future Metal work (parity with the Vulkan backend).
    std::unique_ptr<IRenderTarget> CreateRenderTarget(uint32_t width, uint32_t height) override;
    std::unique_ptr<IRenderTarget> CreateShadowMap(uint32_t size) override;
    FrameContext BeginShadowPass(IRenderTarget& shadowMap) override;
    void EndShadowPass(const FrameContext&) override;
    void SetShadowMap(IRenderTarget& shadowMap) override;
    FrameContext BeginRenderTargetFrame(IRenderTarget& rt) override;
    void EndRenderTargetFrame(const FrameContext&) override;

    FrameContext BeginFrame() override;
    void EndFrame(const FrameContext&) override;
    void SetFrameUniforms(const void* data, uint32_t size) override;
    void WaitIdle() override;
    void CaptureNextFrame() override { captureArmed_ = true; }
    bool GetCapturedPixels(std::vector<uint8_t>& outBGRA,
                           uint32_t& width, uint32_t& height) override;

    // Accessors used by sibling Metal objects.
    id<MTLDevice> device() const { return device_; }
    id<MTLBuffer> currentFrameUbo() const { return uboBuffer_[frameIndex_]; }
    // The shadow map most recently passed to SetShadowMap (null until set). The command buffer
    // binds its depth texture + sampler to the lit pass's fragment shadow slots in BindPipeline.
    MetalRenderTarget* currentShadowMap() const { return shadowMap_; }

private:
    void Init();              // shared device/queue/UBO/recorder setup
    void CreateFrameResources();
    // Read the just-rendered offscreen color texture back into capturedBGRA_ (headless capture).
    void CaptureFromTexture(id<MTLTexture> colorTex);

    hf::hal::Window* window_ = nullptr;  // null in headless mode

    id<MTLDevice>       device_ = nil;
    id<MTLCommandQueue> queue_  = nil;

    std::unique_ptr<MetalSwapchain> swapchain_;
    std::unique_ptr<MetalCommandBuffer> recorder_;
    // Separate recorder for offscreen render-target passes (its own MTLCommandBuffer, committed +
    // waited in EndRenderTargetFrame) so it never clobbers the swapchain recorder's frame state.
    std::unique_ptr<MetalCommandBuffer> rtRecorder_;
    id<MTLCommandBuffer> rtCmd_ = nil;  // in-flight render-target command buffer
    id<MTLCommandBuffer> shadowCmd_ = nil;  // in-flight depth-only shadow command buffer

    // Shadow map currently bound for sampling by the lit pass (set via SetShadowMap). Not owned.
    MetalRenderTarget* shadowMap_ = nil;

    // Per-frame uniform buffers (shared storage), one per frame-in-flight.
    // 512B matches the Vulkan backend's kFrameUboSize: >= sizeof(FrameData) (288B with the added
    // lightViewProj) and future-proofs UBO growth.
    static constexpr uint32_t kFrameUboSize = 512;
    id<MTLBuffer> uboBuffer_[kFramesInFlight] = {nil, nil};

    // In-flight CPU/GPU pacing: block BeginFrame until the frame N-kFramesInFlight ago finished,
    // so SetFrameUniforms never overwrites a UBO the GPU is still reading.
    dispatch_semaphore_t inFlight_ = nullptr;

    uint32_t frameIndex_ = 0;

    // Current frame's acquired drawable (windowed) + in-flight command buffer + the color texture
    // being rendered (drawable's texture when windowed, offscreen texture when headless).
    id<CAMetalDrawable>  currentDrawable_ = nil;
    id<MTLCommandBuffer> currentCmd_      = nil;
    id<MTLTexture>       currentColorTex_ = nil;

    // Headless capture state.
    bool                 captureArmed_ = false;
    std::vector<uint8_t> capturedBGRA_;
    uint32_t             capW_ = 0, capH_ = 0;
};

} // namespace hf::rhi::mtl
