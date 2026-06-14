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

void RenderGraph::Execute(rhi::IRHIDevice& device) {
    std::vector<uint32_t> order = TopoOrder();
    lastOrder_.clear();
    lastOrder_.reserve(order.size());

    for (uint32_t pi : order) {
        const Pass& pass = passes_[pi];
        lastOrder_.push_back(pass.name);
        const Resource& out = OutputResource(pass);

        switch (out.kind) {
            case RgResourceKind::ShadowMap: {
                rhi::FrameContext sc = device.BeginShadowPass(*out.target);
                if (sc.cmd) pass.record(device, *sc.cmd);
                device.EndShadowPass(sc);
                break;
            }
            case RgResourceKind::SceneColor: {
                rhi::FrameContext rtc = device.BeginRenderTargetFrame(*out.target);
                if (rtc.cmd) pass.record(device, *rtc.cmd);
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
                if (frame.cmd) pass.record(device, *frame.cmd);
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
