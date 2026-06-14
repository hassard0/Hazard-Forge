#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_command_buffer.h"
#include "rhi_metal/metal_shader.h"
#include "rhi_metal/metal_pipeline.h"
#include "rhi_metal/metal_buffer.h"
#include "rhi_metal/metal_texture.h"
#include "rhi_metal/metal_common.h"
#import <QuartzCore/CAMetalLayer.h>
#include <cstring>

// NOTE: the windowed constructor MetalDevice(Window&) and the CreateMetalDevice(Window&) factory
// live in metal_device_windowed.mm — they depend on hal/window.h (SDL) and must NOT be linked into
// the headless (SSH / CI) target, which never references a Window.

namespace hf::rhi::mtl {

MetalDevice::MetalDevice(uint32_t width, uint32_t height) {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) Fail("MTLCreateSystemDefaultDevice returned nil");

    queue_ = [device_ newCommandQueue];
    if (!queue_) Fail("newCommandQueue failed");

    // Headless: offscreen swapchain (owns its own BGRA8 color texture, no CAMetalLayer).
    swapchain_ = std::make_unique<MetalSwapchain>(device_, width, height);

    Init();
}

void MetalDevice::Init() {
    inFlight_ = dispatch_semaphore_create(kFramesInFlight);
    CreateFrameResources();
    recorder_ = std::make_unique<MetalCommandBuffer>(*this);
}

MetalDevice::~MetalDevice() {
    WaitIdle();
    recorder_.reset();
    swapchain_.reset();
    // ARC releases device_/queue_/uboBuffer_. inFlight_ is an ObjC object under ARC too.
}

void MetalDevice::CreateFrameResources() {
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        uboBuffer_[i] = [device_ newBufferWithLength:kFrameUboSize
                                             options:MTLResourceStorageModeShared];
        if (!uboBuffer_[i]) Fail("UBO newBuffer failed");
    }
}

std::unique_ptr<IShaderModule> MetalDevice::CreateShaderModule(const ShaderModuleDesc&) {
    Fail("Metal cannot consume SPIR-V; use CreateShaderModuleMSL (the sample does under __APPLE__)");
}

std::unique_ptr<IShaderModule>
MetalDevice::CreateShaderModuleMSL(const std::string& source, const std::string& entryPoint) {
    return std::make_unique<MetalShaderModule>(device_, source, entryPoint);
}

std::unique_ptr<IPipeline> MetalDevice::CreateGraphicsPipeline(const GraphicsPipelineDesc& d) {
    return std::make_unique<MetalPipeline>(*this, d);
}

std::unique_ptr<IBuffer> MetalDevice::CreateBuffer(const BufferDesc& d) {
    return std::make_unique<MetalBuffer>(device_, d);
}

std::unique_ptr<ITexture> MetalDevice::CreateTexture(const TextureDesc& d) {
    return std::make_unique<MetalTexture>(*this, d);
}

// --- Render-targets + shadows: not yet implemented on Metal (stubs) ----------------------------
// Master's IRHIDevice requires these (render-targets/shadows slices). The headless Slice-F scene
// path does not exercise them, so the existing lit-cube/scene render is unaffected. They throw so
// any future caller that needs them fails loudly rather than silently mis-rendering.

std::unique_ptr<IRenderTarget> MetalDevice::CreateRenderTarget(uint32_t, uint32_t) {
    throw std::runtime_error("CreateRenderTarget not implemented on Metal yet");
}

std::unique_ptr<IRenderTarget> MetalDevice::CreateShadowMap(uint32_t) {
    throw std::runtime_error("CreateShadowMap not implemented on Metal yet");
}

FrameContext MetalDevice::BeginShadowPass(IRenderTarget&) {
    throw std::runtime_error("BeginShadowPass not implemented on Metal yet");
}

void MetalDevice::EndShadowPass(const FrameContext&) {
    throw std::runtime_error("EndShadowPass not implemented on Metal yet");
}

void MetalDevice::SetShadowMap(IRenderTarget&) {
    throw std::runtime_error("SetShadowMap not implemented on Metal yet");
}

FrameContext MetalDevice::BeginRenderTargetFrame(IRenderTarget&) {
    throw std::runtime_error("BeginRenderTargetFrame not implemented on Metal yet");
}

void MetalDevice::EndRenderTargetFrame(const FrameContext&) {
    throw std::runtime_error("EndRenderTargetFrame not implemented on Metal yet");
}

void MetalDevice::SetFrameUniforms(const void* data, uint32_t size) {
    if (size > kFrameUboSize) Fail("SetFrameUniforms: size exceeds UBO");
    // Shared-storage buffer: contents() is host-visible; the in-flight semaphore guarantees the
    // GPU is no longer reading uboBuffer_[frameIndex_] (see BeginFrame).
    std::memcpy([uboBuffer_[frameIndex_] contents], data, size);
}

void MetalDevice::WaitIdle() {
    // Submit an empty command buffer and wait for completion to drain the queue.
    if (!queue_) return;
    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    [cb commit];
    [cb waitUntilCompleted];
}

bool MetalDevice::GetCapturedPixels(std::vector<uint8_t>& out, uint32_t& w, uint32_t& h) {
    if (capturedBGRA_.empty()) return false;
    out = std::move(capturedBGRA_);
    w = capW_;
    h = capH_;
    capturedBGRA_.clear();
    capW_ = capH_ = 0;
    return true;
}

void MetalDevice::CaptureFromTexture(id<MTLTexture> colorTex) {
    capW_ = (uint32_t)colorTex.width;
    capH_ = (uint32_t)colorTex.height;
    capturedBGRA_.resize((size_t)capW_ * capH_ * 4);
    const NSUInteger bytesPerRow = (NSUInteger)capW_ * 4;
    MTLRegion region = MTLRegionMake2D(0, 0, capW_, capH_);
    // BGRA8Unorm color texture -> tightly-packed BGRA8, top row first (matches Vulkan capture).
    [colorTex getBytes:capturedBGRA_.data()
           bytesPerRow:bytesPerRow
            fromRegion:region
           mipmapLevel:0];
}

FrameContext MetalDevice::BeginFrame() {
    // CPU/GPU pacing: don't get more than kFramesInFlight frames ahead.
    dispatch_semaphore_wait(inFlight_, DISPATCH_TIME_FOREVER);

    if (swapchain_->headless()) {
        // Headless: render straight into the offscreen color texture (no drawable to acquire).
        currentDrawable_ = nil;
        currentColorTex_ = swapchain_->offscreenColor();
    } else {
        currentDrawable_ = swapchain_->AcquireNext();
        if (!currentDrawable_) {
            // No drawable (e.g. zero-sized / occluded). Release the in-flight slot and skip.
            dispatch_semaphore_signal(inFlight_);
            return FrameContext{nullptr};
        }
        currentColorTex_ = currentDrawable_.texture;
    }

    currentCmd_ = [queue_ commandBuffer];

    recorder_->Begin(currentCmd_, currentColorTex_, swapchain_->depthTexture(),
                     swapchain_->width(), swapchain_->height());
    return FrameContext{recorder_.get()};
}

void MetalDevice::EndFrame(const FrameContext& frame) {
    if (!frame.cmd) return;  // skipped frame

    // --- Headless / capture branch: commit + wait, read the color texture back, no present. ---
    // Headless always takes this branch (there is nothing to present); a windowed device only
    // takes it when capture was armed.
    if (swapchain_->headless() || captureArmed_) {
        [currentCmd_ commit];
        [currentCmd_ waitUntilCompleted];

        if (captureArmed_) {
            // On Apple Silicon the color texture is shared/managed (framebufferOnly=NO for the
            // CAMetalLayer drawable, MTLStorageModeShared for the offscreen texture), so getBytes
            // reads it back directly. On a discrete-GPU Mac a drawable may be private — then this
            // needs a blit into a shared staging texture first. We target Apple Silicon.
            CaptureFromTexture(currentColorTex_);
        }

        dispatch_semaphore_signal(inFlight_);
        frameIndex_ = (frameIndex_ + 1) % kFramesInFlight;
        currentDrawable_ = nil;
        currentCmd_ = nil;
        currentColorTex_ = nil;
        captureArmed_ = false;
        return;
    }

    // Normal windowed path: present the drawable, signal the in-flight semaphore on GPU completion.
    [currentCmd_ presentDrawable:currentDrawable_];

    dispatch_semaphore_t sem = inFlight_;
    [currentCmd_ addCompletedHandler:^(id<MTLCommandBuffer>) {
        dispatch_semaphore_signal(sem);
    }];
    [currentCmd_ commit];

    frameIndex_ = (frameIndex_ + 1) % kFramesInFlight;
    currentDrawable_ = nil;
    currentCmd_ = nil;
    currentColorTex_ = nil;
}

// Headless factory (no window). Lets plain-C++ callers build an offscreen Metal device without
// touching Metal/Obj-C headers; declared extern in metal_offscreen.h.
std::unique_ptr<IRHIDevice> CreateMetalDeviceHeadless(uint32_t width, uint32_t height) {
    return std::make_unique<MetalDevice>(width, height);
}

} // namespace hf::rhi::mtl
