#include "render/render_graph.h"

#include <stdexcept>
#include <unordered_map>

namespace hf::render {

RgResource RenderGraph::ImportTarget(const std::string& name, RgResourceKind kind,
                                     rhi::IRenderTarget& target) {
    RgResource h{static_cast<uint32_t>(resources_.size())};
    resources_.push_back(Resource{name, kind, &target});
    return h;
}

RgResource RenderGraph::ImportSwapchain(const std::string& name) {
    RgResource h{static_cast<uint32_t>(resources_.size())};
    resources_.push_back(Resource{name, RgResourceKind::Swapchain, nullptr});
    return h;
}

RgResource RenderGraph::CreateTransient(const std::string& name) {
    RgResource h{static_cast<uint32_t>(resources_.size())};
    resources_.push_back(Resource{name, RgResourceKind::Transient, nullptr});
    return h;
}

void RenderGraph::AddPass(const std::string& name,
                          std::vector<RgResource> reads,
                          std::vector<RgResource> writes,
                          RgRecordFn record) {
    passes_.push_back(Pass{name, std::move(reads), std::move(writes), std::move(record)});
}

// The state a pass requires a resource to be in, inferred from the resource KIND + access role.
//   ShadowMap : write -> DepthWrite   (depth-only shadow render), read -> ShaderRead (lit pass samples)
//   SceneColor: write -> ColorTarget  (RT color render),          read -> ShaderRead (composite samples)
//   Swapchain : write -> ColorTarget  (final composite),          read -> ShaderRead (unused today)
//   Transient : Undefined (no GPU-backed image; dependency edges only)
RgResourceState RenderGraph::RequiredState(RgResourceKind kind, bool isWrite) {
    switch (kind) {
        case RgResourceKind::ShadowMap:  return isWrite ? RgResourceState::DepthWrite
                                                        : RgResourceState::ShaderRead;
        case RgResourceKind::SceneColor: return isWrite ? RgResourceState::ColorTarget
                                                        : RgResourceState::ShaderRead;
        case RgResourceKind::Swapchain:  return isWrite ? RgResourceState::ColorTarget
                                                        : RgResourceState::ShaderRead;
        case RgResourceKind::Transient:
        default:                         return RgResourceState::Undefined;
    }
}

const RenderGraph::Resource& RenderGraph::OutputResource(const Pass& pass) const {
    const Resource* out = nullptr;
    for (const RgResource& w : pass.writes) {
        const Resource& r = resources_.at(w.id);
        if (r.kind == RgResourceKind::Transient) continue;
        if (out)
            throw std::runtime_error("RenderGraph: pass '" + pass.name +
                                     "' writes more than one external resource");
        out = &r;
    }
    if (!out)
        throw std::runtime_error("RenderGraph: pass '" + pass.name +
                                 "' has no external write (render output)");
    return *out;
}

// Kahn's algorithm. An edge A->B exists when pass A WRITES a resource that pass B READS (a
// write-then-read dependency), so producers run before consumers. Ties (passes with no ordering
// constraint between them) keep declaration order, which preserves the existing frame sequence.
std::vector<uint32_t> RenderGraph::TopoOrder() const {
    const size_t n = passes_.size();

    // Map each resource id -> list of pass indices that WRITE it (the producers).
    std::unordered_map<uint32_t, std::vector<uint32_t>> writers;
    for (uint32_t p = 0; p < n; ++p)
        for (const RgResource& w : passes_[p].writes)
            writers[w.id].push_back(p);

    // Build edges producer -> consumer for every read that some other pass wrote.
    std::vector<std::vector<uint32_t>> adj(n);
    std::vector<uint32_t> indeg(n, 0);
    for (uint32_t c = 0; c < n; ++c) {
        for (const RgResource& rd : passes_[c].reads) {
            auto it = writers.find(rd.id);
            if (it == writers.end()) continue;
            for (uint32_t prod : it->second) {
                if (prod == c) continue;
                adj[prod].push_back(c);
                ++indeg[c];
            }
        }
    }

    // Kahn: repeatedly take the lowest-index pass with in-degree 0 (stable -> declaration order).
    std::vector<uint32_t> order;
    order.reserve(n);
    std::vector<bool> done(n, false);
    for (size_t step = 0; step < n; ++step) {
        uint32_t next = ~0u;
        for (uint32_t p = 0; p < n; ++p) {
            if (!done[p] && indeg[p] == 0) { next = p; break; }
        }
        if (next == ~0u)
            throw std::runtime_error("RenderGraph: dependency cycle among passes");
        done[next] = true;
        order.push_back(next);
        for (uint32_t m : adj[next]) --indeg[m];
    }
    return order;
}

// Map the graph's pure-logic state to the backend-agnostic RHI state the command buffer transitions.
static rhi::ResourceState ToRhi(RgResourceState s) {
    switch (s) {
        case RgResourceState::Undefined:   return rhi::ResourceState::Undefined;
        case RgResourceState::ColorTarget: return rhi::ResourceState::ColorTarget;
        case RgResourceState::DepthWrite:  return rhi::ResourceState::DepthWrite;
        case RgResourceState::DepthRead:   return rhi::ResourceState::DepthRead;
        case RgResourceState::ShaderRead:  return rhi::ResourceState::ShaderRead;
        case RgResourceState::Present:     return rhi::ResourceState::Present;
    }
    return rhi::ResourceState::Undefined;
}

void RenderGraph::Solve() {
    std::vector<uint32_t> order = TopoOrder();
    lastOrder_.clear();
    lastOrder_.reserve(order.size());
    lastBarriers_.clear();

    // Reset the state tracker: every render-target resource begins Undefined (its contents are
    // produced fresh each frame). Clear any stale per-pass plan from a prior Solve().
    for (Resource& r : resources_) r.currentState = RgResourceState::Undefined;
    for (Pass& p : passes_) { p.preBarriers.clear(); p.postBarriers.clear(); }

    // Record a transition into `plan`, advancing the resource's tracked state. from==to is coalesced
    // (a no-op needs no barrier). Transients (Undefined required state) never transition.
    auto transition = [&](std::vector<RgBarrier>& plan, RgResource h, RgResourceState to) {
        Resource& r = resources_.at(h.id);
        if (r.kind == RgResourceKind::Transient) return;  // no GPU-backed image to transition
        if (to == RgResourceState::Undefined) return;     // nothing required
        if (r.currentState == to) return;                 // already there -> coalesce away
        RgBarrier b{h, r.currentState, to};
        plan.push_back(b);
        lastBarriers_.push_back(b);
        r.currentState = to;
    };

    for (uint32_t pi : order) {
        Pass& pass = passes_[pi];
        lastOrder_.push_back(pass.name);

        // BEFORE the pass: every read must be in its required state, then every write. Reads first so
        // a resource both produced earlier and read here is sampled in ShaderRead before being
        // re-acquired for writing (no pass currently both reads and writes the same external image).
        for (const RgResource& rd : pass.reads)
            transition(pass.preBarriers, rd, RequiredState(resources_.at(rd.id).kind, /*write*/false));
        for (const RgResource& wr : pass.writes)
            transition(pass.preBarriers, wr, RequiredState(resources_.at(wr.id).kind, /*write*/true));

        // AFTER the pass: the swapchain image, once written, transitions to Present (terminal). This
        // is the only post-pass transition the graph owns.
        for (const RgResource& wr : pass.writes) {
            const Resource& r = resources_.at(wr.id);
            if (r.kind == RgResourceKind::Swapchain)
                transition(pass.postBarriers, wr, RgResourceState::Present);
        }
    }
}

void RenderGraph::Execute(rhi::IRHIDevice& device) {
    Solve();
    std::vector<uint32_t> order = TopoOrder();

    // Emit the solved transitions for a pass through the given command buffer. (Vulkan turns each
    // into a vkCmdPipelineBarrier2; Metal's tracked-hazard model no-ops them.)
    //
    // INCREMENTAL CUTOVER (Slice AS): the graph OWNS the inter-pass CONSUMER transitions — the ones a
    // pass needs to READ a resource a previous pass produced: ->ShaderRead (shadow map / scene color
    // sampled downstream) and ->Present (swapchain handed to the presentation engine). Those are the
    // transitions removed from EndShadowPass / EndRenderTargetFrame and now emitted here, validated
    // hazard-free by the synchronization layer. The WRITE-ACQUIRE transitions (->ColorTarget for an RT
    // color render, ->DepthWrite for the shadow render) are STILL owned by the Begin* scaffolding
    // (BeginRenderTargetFrame/BeginShadowPass already transition the image into its attachment layout),
    // so the graph TRACKS them (to know each barrier's correct `from` state) but does NOT re-emit them
    // — emitting both would double-apply and mismatch oldLayout. Likewise the swapchain image has no
    // backing IRenderTarget here, so its ->Present is still owned by Begin/EndFrame (the graph computes
    // it for inspection/tests; tgt==null skips emission). YAGNI to push the cutover further this slice.
    auto emit = [&](rhi::ICommandBuffer& cmd, const std::vector<RgBarrier>& plan) {
        for (const RgBarrier& b : plan) {
            // Only emit consumer transitions; write-acquires stay owned by the Begin* scaffolding.
            const bool consumer = (b.to == RgResourceState::ShaderRead ||
                                   b.to == RgResourceState::DepthRead ||
                                   b.to == RgResourceState::Present);
            if (!consumer) continue;
            rhi::IRenderTarget* tgt = resources_.at(b.resource.id).target;
            if (tgt) cmd.ResourceBarrier(*tgt, ToRhi(b.from), ToRhi(b.to));
        }
    };

    for (uint32_t pi : order) {
        const Pass& pass = passes_[pi];
        const Resource& out = OutputResource(pass);

        switch (out.kind) {
            case RgResourceKind::ShadowMap: {
                rhi::FrameContext sc = device.BeginShadowPass(*out.target);
                if (sc.cmd) {
                    emit(*sc.cmd, pass.preBarriers);
                    if (pass.record) pass.record(device, *sc.cmd);
                }
                device.EndShadowPass(sc);
                break;
            }
            case RgResourceKind::SceneColor: {
                rhi::FrameContext rtc = device.BeginRenderTargetFrame(*out.target);
                if (rtc.cmd) {
                    emit(*rtc.cmd, pass.preBarriers);
                    if (pass.record) pass.record(device, *rtc.cmd);
                }
                device.EndRenderTargetFrame(rtc);
                break;
            }
            case RgResourceKind::Swapchain: {
                rhi::FrameContext frame = device.BeginFrame();
                if (!frame.cmd && swapchainRetryArm_) {
                    // A fresh swapchain may report out-of-date on the first acquire; re-arm the
                    // capture (matches the inline --shot retry) and acquire once more.
                    swapchainRetryArm_();
                    frame = device.BeginFrame();
                }
                if (frame.cmd) {
                    emit(*frame.cmd, pass.preBarriers);
                    if (pass.record) pass.record(device, *frame.cmd);
                    emit(*frame.cmd, pass.postBarriers);
                }
                device.EndFrame(frame);
                break;
            }
            case RgResourceKind::Transient:
            default:
                throw std::runtime_error("RenderGraph: pass '" + pass.name +
                                         "' output is not a renderable external resource");
        }
    }
}

}  // namespace hf::render
