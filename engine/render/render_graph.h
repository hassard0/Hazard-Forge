#pragma once
// Hazard Forge — declarative render graph (frame graph).
//
// A HIGH-LEVEL, backend-agnostic orchestration layer that sits ABOVE the RHI. Each frame is
// expressed as a set of passes; every pass DECLARES the resources it reads and writes. The graph
// topologically orders the passes by their data dependencies (a write-then-read on the same
// resource is an edge) and then drives the matching RHI pass scaffolding for each pass.
//
// Slice AS — this layer now OWNS inter-pass GPU synchronization. A per-resource STATE TRACKER plus a
// topo-ordered BARRIER SOLVER computes the minimal set of resource-state transitions and emits them
// through the additive RHI primitive `ICommandBuffer::ResourceBarrier` BEFORE each pass that needs
// the new state (and the swapchain->Present transition AFTER the final pass). The solver is pure
// logic (no backend symbols); on Vulkan the barrier becomes an explicit vkCmdPipelineBarrier2 (proven
// hazard-free by the synchronization-validation layer), while Metal's default tracked-hazard model
// makes it a documented no-op. The RHI's Begin*/End* pass methods retain the few intra-pass /
// layout-acquire transitions that are NOT yet owned by the graph (cutover is incremental; the graph
// owns the inter-pass shadow->ShaderRead / sceneColor->ShaderRead / swapchain->Present transitions).
// The graph's job: the dependency model + the declarative API + the state/barrier solver. It maps a
// pass's OUTPUT resource to the right RHI Begin/End call and runs the pass's record callback inside.
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

// Backend-agnostic resource ACCESS STATE the solver tracks per render-target resource. Mirrors
// rhi::ResourceState (the graph maps each value 1:1 when it emits ResourceBarrier); kept here as the
// graph's own enum so the solver is pure logic with no RHI coupling beyond the emit call. The names
// describe the GPU usage, not a backend layout (Vulkan maps them to VkImageLayout + stage/access in
// the backend; Metal's tracked-hazard model makes them no-ops). NOTE: render-TARGET image states
// only — SSBO/indirect buffers keep their existing explicit barriers (out of scope, see the spec).
enum class RgResourceState {
    Undefined,    // initial / contents-don't-matter (freshly imported or about to be fully written)
    ColorTarget,  // being written as a color attachment (scene RT / swapchain image)
    DepthWrite,   // being written as a depth attachment (shadow map render)
    DepthRead,    // sampled/read as depth (reserved; no current pass needs depth-read-only)
    ShaderRead,   // sampled in a shader (shadow map in the lit pass, scene color in the composite)
    Present,      // handed to the presentation engine (swapchain image, terminal state)
};

// One solved transition: the resource `resource` must move from state `from` to state `to`. The
// solver emits these in execution order; the graph issues each via ICommandBuffer::ResourceBarrier
// at the right point (before the pass that needs `to`; the swapchain->Present one AFTER the pass).
struct RgBarrier {
    RgResource     resource;
    RgResourceState from = RgResourceState::Undefined;
    RgResourceState to   = RgResourceState::Undefined;
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

    // --- Barrier solver (pure logic) -----------------------------------------

    // Topologically order the passes, then walk them computing the per-resource state transitions:
    // for each resource a pass accesses, if its currentState != the state the pass requires, record
    // a RgBarrier{resource, from, to} and advance currentState. Coalesces no-ops (from==to emits
    // nothing). The swapchain's terminal Undefined->...->ColorTarget->Present is included. Populates
    // LastOrder() and LastBarriers() WITHOUT touching any device, so the solver is unit-testable on
    // its own. Throws std::runtime_error on a dependency cycle or a malformed pass.
    void Solve();

    // Returns the solved barrier plan (ordered transitions to emit) from the last Solve()/Execute().
    const std::vector<RgBarrier>& LastBarriers() const { return lastBarriers_; }

    // The current tracked state of a resource (Undefined before any Solve()/Execute()). For tests.
    RgResourceState CurrentState(RgResource r) const { return resources_.at(r.id).currentState; }

    // The registered name of a resource handle (for diagnostics/tests).
    const std::string& ResourceName(RgResource r) const { return resources_.at(r.id).name; }

    // --- Execution -----------------------------------------------------------

    // Solve() the barrier plan, then run each pass by driving the matching RHI Begin/End scaffolding,
    // emitting the solved ResourceBarrier transitions at the right points, and invoking its record
    // callback inside. Throws std::runtime_error on a dependency cycle or a malformed pass.
    void Execute(rhi::IRHIDevice& device);

    // Returns the pass execution order (names) produced by the last Solve()/Execute().
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
        // The state tracker's live value. Every render-target resource starts Undefined; Solve()
        // advances it as the barrier plan is computed (and Execute() leaves it at the solved end
        // state). Transients carry no GPU state — they exist only for dependency edges.
        RgResourceState currentState = RgResourceState::Undefined;
    };
    struct Pass {
        std::string name;
        std::vector<RgResource> reads;
        std::vector<RgResource> writes;
        RgRecordFn record;
        // Solver output: the transitions to issue BEFORE this pass records (the resource must be in
        // its required state for the pass). The swapchain->Present transition is appended AFTER the
        // pass instead (see postBarriers).
        std::vector<RgBarrier> preBarriers;
        std::vector<RgBarrier> postBarriers;
    };

    // The external write that selects this pass's RHI scaffolding (ShadowMap/SceneColor/Swapchain).
    // Throws if the pass has zero or more than one external write.
    const Resource& OutputResource(const Pass& pass) const;

    // The state a pass requires a resource to be in for the given access role. Inferred from the
    // resource KIND + whether the pass READS or WRITES it (the spec's "infer from scaffolding kind +
    // access role" model): a ShadowMap write needs DepthWrite, any read needs ShaderRead; a
    // SceneColor/Swapchain write needs ColorTarget; transients have no GPU state (Undefined).
    static RgResourceState RequiredState(RgResourceKind kind, bool isWrite);

    std::vector<uint32_t> TopoOrder() const;  // returns pass indices in execution order

    std::vector<Resource> resources_;
    std::vector<Pass> passes_;
    std::vector<std::string> lastOrder_;
    std::vector<RgBarrier> lastBarriers_;  // the flat solved plan (in execution order) from Solve()
    std::function<void()> swapchainRetryArm_;
};

}  // namespace hf::render
