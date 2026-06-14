#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

// Forward declarations of HAL + handle types — NO vulkan headers in this file.
struct VkInstance_T;  using VkInstance = VkInstance_T*;
struct VkSurfaceKHR_T; using VkSurfaceKHR = VkSurfaceKHR_T*;

namespace hf::hal { class Window; }

namespace hf::rhi {

enum class Backend { Vulkan };

enum class Format {
    Undefined,
    RGBA8_UNorm,
    BGRA8_UNorm,
    RG32_Float,    // vec2 vertex attribute
    RGB32_Float,   // vec3 vertex attribute
    D32_Float,     // depth attachment
};

struct ClearColor { float r = 0, g = 0, b = 0, a = 1; };

// --- Resource descriptors ----------------------------------------------------

struct VertexAttribute {
    uint32_t location;   // shader input location
    Format   format;     // attribute element format
    uint32_t offset;     // byte offset within a vertex
};

struct VertexLayout {
    uint32_t stride = 0;                       // bytes per vertex
    std::vector<VertexAttribute> attributes;   // per-attribute layout
};

struct ShaderModuleDesc {
    std::span<const uint32_t> spirv;  // SPIR-V words
};

struct GraphicsPipelineDesc {
    class IShaderModule* vertex = nullptr;
    class IShaderModule* fragment = nullptr;
    VertexLayout vertexLayout;
    Format colorFormat = Format::BGRA8_UNorm;  // must match swapchain format
    bool depthTest = true;
    Format depthFormat = Format::D32_Float;
    uint32_t pushConstantSize = 0;   // bytes, vertex stage
    bool usesFrameUniforms = false;  // when true, layout includes the per-frame UBO set (set 0)
    bool usesTexture = false;        // when true, layout includes the material set (set 1)
    bool fullscreen = false;         // when true: no vertex input (fullscreen triangle from SV_VertexID)
};

enum class BufferUsage { Vertex, Index, Uniform };

struct TextureDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    Format format = Format::RGBA8_UNorm;
    const void* data = nullptr;   // tightly packed pixels
    uint64_t dataSize = 0;        // bytes
};

struct BufferDesc {
    uint64_t size = 0;          // bytes
    const void* initialData = nullptr;  // optional upload-on-create
    BufferUsage usage = BufferUsage::Vertex;
};

// --- Resource interfaces -----------------------------------------------------

class IShaderModule { public: virtual ~IShaderModule() = default; };
class IPipeline     { public: virtual ~IPipeline() = default; };
class IBuffer       { public: virtual ~IBuffer() = default; };
class ITexture      { public: virtual ~ITexture() = default; };

// A sampleable offscreen color image (+ its own depth) you render into. Inheriting ITexture
// lets the post pass bind it via the existing ICommandBuffer::BindTexture.
class IRenderTarget : public ITexture {
public:
    virtual uint32_t width() const = 0;
    virtual uint32_t height() const = 0;
};

// Records draw commands for one frame's swapchain image.
class ICommandBuffer {
public:
    virtual ~ICommandBuffer() = default;
    virtual void BeginRenderPass(const ClearColor& clear) = 0;
    virtual void BindPipeline(IPipeline& pipeline) = 0;
    virtual void BindVertexBuffer(IBuffer& buffer) = 0;
    virtual void BindIndexBuffer(IBuffer& buffer) = 0;
    virtual void BindTexture(ITexture& texture) = 0;
    virtual void Draw(uint32_t vertexCount, uint32_t firstVertex = 0) = 0;
    virtual void DrawIndexed(uint32_t indexCount, uint32_t firstIndex = 0) = 0;
    virtual void PushConstants(const void* data, uint32_t size) = 0;
    virtual void EndRenderPass() = 0;
};

class ISwapchain {
public:
    virtual ~ISwapchain() = default;
    virtual Format ColorFormat() const = 0;
    // Recreate after a window resize (or out-of-date acquire/present).
    virtual void Recreate(uint32_t width, uint32_t height) = 0;
};

// Per-frame handle returned by BeginFrame; passed back to EndFrame.
struct FrameContext {
    ICommandBuffer* cmd = nullptr;  // null if the frame was skipped (e.g. minimized)
};

class IRHIDevice {
public:
    virtual ~IRHIDevice() = default;

    virtual ISwapchain& Swapchain() = 0;

    virtual std::unique_ptr<IShaderModule> CreateShaderModule(const ShaderModuleDesc&) = 0;
    virtual std::unique_ptr<IPipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc&) = 0;
    virtual std::unique_ptr<IBuffer> CreateBuffer(const BufferDesc&) = 0;
    virtual std::unique_ptr<ITexture> CreateTexture(const TextureDesc&) = 0;

    // Offscreen render target: a sampleable color image (swapchain format) + its own depth.
    virtual std::unique_ptr<IRenderTarget> CreateRenderTarget(uint32_t width, uint32_t height) = 0;

    // Begin recording the scene into the render target's color+depth (dynamic rendering).
    // Returns a FrameContext whose cmd records into the RT; pair with EndRenderTargetFrame.
    virtual FrameContext BeginRenderTargetFrame(IRenderTarget& rt) = 0;
    // End recording, submit, and transition the RT color to SHADER_READ_ONLY so a later pass
    // can sample it. Blocks until the RT pass completes (so the swapchain pass sees the result).
    virtual void EndRenderTargetFrame(const FrameContext&) = 0;

    // Acquire next swapchain image + begin command recording.
    // Returns FrameContext{nullptr} when the frame must be skipped this tick.
    virtual FrameContext BeginFrame() = 0;
    // Submit recorded commands and present.
    virtual void EndFrame(const FrameContext&) = 0;

    // Copy per-frame uniform data into the current frame-in-flight's UBO. Call after
    // BeginFrame (so the current frame index is set) and before recording draws; the
    // matching per-frame descriptor set (set 0) is auto-bound on BindPipeline.
    virtual void SetFrameUniforms(const void* data, uint32_t size) = 0;

    // Block until the GPU is idle (call before destroying GPU resources).
    virtual void WaitIdle() = 0;

    // Headless capture: arm before BeginFrame; after EndFrame, retrieve via GetCapturedPixels.
    virtual void CaptureNextFrame() = 0;
    // Returns the last captured frame as tightly-packed BGRA8 (top row first); false if none.
    virtual bool GetCapturedPixels(std::vector<uint8_t>& outBGRA,
                                   uint32_t& width, uint32_t& height) = 0;
};

} // namespace hf::rhi
