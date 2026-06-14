#pragma once
// Hazard Forge — declarative render graph (frame graph).
//
// A HIGH-LEVEL, backend-agnostic orchestration layer that sits ABOVE the RHI. Each frame is
// expressed as a set of passes; every pass DECLARES the resources it reads and writes. The graph
// topologically orders the passes by their data dependencies (a write-then-read on the same
// resource is an edge) and then drives the matching RHI pass scaffolding for each pass.
//
// This layer does NOT generate barriers or transitions — the RHI's Begin*/End* pass methods still
// own all of that (shadow pass transitions the depth to SHADER_READ_ONLY on End, etc.). The graph's
// job is purely the dependency model + the declarative API: it maps a pass's OUTPUT resource to the
// right RHI Begin/End call and runs the pass's record callback inside.
//
// Resource -> RHI scaffolding mapping (by the output resource's kind):
//   ShadowMap   -> BeginShadowPass / EndShadowPass
//   SceneColor  -> BeginRenderTargetFrame / EndRenderTargetFrame
//   Swapchain   -> BeginFrame / EndFrame
//
// Depends only on rhi/ + std. No vk*/Metal symbols.
#include "rhi/rhi.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace hf::render {

// Opaque handle to a named graph resource (transient or external).
struct RgResource {
    uint32_t id = ~0u;
    bool valid() const { return id != ~0u; }
    bool operator==(const RgResource& o) const { return id == o.id; }
};

// How an EXTERNAL resource is realized at execute time and which RHI scaffolding renders into it.
// Every pass that WRITES an external resource is driven through the matching Begin/End calls.
enum class RgResourceKind {
    Transient,   // a logical resource with no RHI backing object (dependency edges only)
    ShadowMap,   // device->Begin/EndShadowPass(*target)
    SceneColor,  // device->Begin/EndRenderTargetFrame(*target)
    Swapchain,   // device->Begin/EndFrame()
};

// A pass's record callback: records the draws for this pass. Receives the device (so it can call
// SetFrameUniforms exactly where the inline code did) and the command buffer to record into. The
// graph has already issued the matching Begin* call; the callback must NOT call Begin*/End* itself.
using RgRecordFn = std::function<void(rhi::IRHIDevice& device, rhi::ICommandBuffer& cmd)>;

class RenderGraph {
public:
    // --- Resource declaration ------------------------------------------------

    // Import an EXTERNAL resource backed by an RHI render target (shadow map / scene-color RT). The
    // `kind` picks the RHI scaffolding used by the pass that writes it.
    RgResource ImportTarget(const std::string& name, RgResourceKind kind, rhi::IRenderTarget& target);

    // Import the swapchain as an external resource (no IRenderTarget; BeginFrame/EndFrame).
    RgResource ImportSwapchain(const std::string& name);

    // Declare a transient (logical) resource used only for dependency ordering.
    RgResource CreateTransient(const std::string& name);

    // --- Pass declaration ----------------------------------------------------

    // Add a pass that READS `reads` and WRITES `writes`, recording its draws via `record`. The pass
    // is driven through the RHI scaffolding selected by its (single) external write resource. A pass
    // may declare exactly one external write (its render output); transient writes are allowed too.
    void AddPass(const std::string& name,
                 std::vector<RgResource> reads,
                 std::vector<RgResource> writes,
                 RgRecordFn record);

    // --- Execution -----------------------------------------------------------

    // Topologically order the passes by their read/write dependencies, then run each pass by driving
    // the matching RHI Begin/End scaffolding and invoking its record callback inside.
    // Throws std::runtime_error on a dependency cycle or a malformed pass.
    void Execute(rhi::IRHIDevice& device);

    // Returns the pass execution order (names) produced by the last Execute(), for tests/inspection.
    const std::vector<std::string>& LastOrder() const { return lastOrder_; }

    // Optional: a callback the graph invokes to RE-ARM headless capture when the swapchain's first
    // acquire reports out-of-date (a fresh-swapchain quirk). Mirrors the inline --shot retry exactly;
    // leave unset for normal present/interactive frames.
    void SetSwapchainRetryArm(std::function<void()> fn) { swapchainRetryArm_ = std::move(fn); }

private:
    struct Resource {
        std::string name;
        RgResourceKind kind = RgResourceKind::Transient;
        rhi::IRenderTarget* target = nullptr;  // non-null for ImportTarget resources
    };
    struct Pass {
        std::string name;
        std::vector<RgResource> reads;
        std::vector<RgResource> writes;
        RgRecordFn record;
    };

    // The external write that selects this pass's RHI scaffolding (ShadowMap/SceneColor/Swapchain).
    // Throws if the pass has zero or more than one external write.
    const Resource& OutputResource(const Pass& pass) const;

    std::vector<uint32_t> TopoOrder() const;  // returns pass indices in execution order

    std::vector<Resource> resources_;
    std::vector<Pass> passes_;
    std::vector<std::string> lastOrder_;
    std::function<void()> swapchainRetryArm_;
};

}  // namespace hf::render
