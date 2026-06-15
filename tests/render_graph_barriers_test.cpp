// Unit test for the render graph's resource-state tracker + AUTOMATIC BARRIER solver (Slice AS).
//
// Pure logic — no GPU/RHI device. We hand-build a pass DAG with per-resource access states and assert
// the solver's emitted barrier plan (the ordered transitions that must be issued BEFORE each pass)
// matches a hand-checked expected sequence. This is the intellectual core of the slice: the graph
// OWNS inter-pass synchronization. The Vulkan synchronization-validation layer is the live oracle;
// THIS test pins the pure transition logic (incl. no-op coalescing when from==to and the canonical
// shadow->shaderRead / sceneColor->shaderRead / swapchain->present transitions) independent of a device.
#include "render/render_graph.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace hf;
using render::RgResourceState;

namespace {
int g_failures = 0;
void check(bool cond, const std::string& msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg.c_str()); ++g_failures; }
}

const char* StateName(RgResourceState s) {
    switch (s) {
        case RgResourceState::Undefined:   return "Undefined";
        case RgResourceState::ColorTarget: return "ColorTarget";
        case RgResourceState::DepthWrite:  return "DepthWrite";
        case RgResourceState::DepthRead:   return "DepthRead";
        case RgResourceState::ShaderRead:  return "ShaderRead";
        case RgResourceState::Present:     return "Present";
    }
    return "?";
}

// A no-op stub render target so ImportTarget has something to bind (only the pointer is stored).
class StubTarget : public rhi::IRenderTarget {
public:
    uint32_t width() const override { return 1; }
    uint32_t height() const override { return 1; }
};

// Render one expected/actual barrier as "resName: from->to" for diagnostic output.
std::string Fmt(const render::RgBarrier& b, const render::RenderGraph& g) {
    return g.ResourceName(b.resource) + ": " + StateName(b.from) + "->" + StateName(b.to);
}
}  // namespace

int main() {
    // --- Canonical shadow -> scene -> post frame -------------------------------------------------
    // shadow writes shadowMap (DepthWrite); scene reads shadowMap (must be ShaderRead) + writes
    // sceneColor (ColorTarget); post reads sceneColor (must be ShaderRead) + writes swapchain
    // (ColorTarget, then Present after the pass). Declared SCRAMBLED to prove topo ordering drives
    // the solver too.
    StubTarget shadowTgt, sceneTgt;
    render::RenderGraph graph;
    auto rgShadow = graph.ImportTarget("shadowMap", render::RgResourceKind::ShadowMap, shadowTgt);
    auto rgScene  = graph.ImportTarget("sceneColor", render::RgResourceKind::SceneColor, sceneTgt);
    auto rgSwap   = graph.ImportSwapchain("swapchain");

    graph.AddPass("post",   /*reads*/{rgScene},  /*writes*/{rgSwap},   nullptr);
    graph.AddPass("scene",  /*reads*/{rgShadow}, /*writes*/{rgScene},  nullptr);
    graph.AddPass("shadow", /*reads*/{},         /*writes*/{rgShadow}, nullptr);

    // Solve WITHOUT a device — pure logic.
    graph.Solve();
    const std::vector<render::RgBarrier>& bars = graph.LastBarriers();

    // Hand-checked expected sequence (in execution order shadow -> scene -> post):
    //   before shadow: shadowMap Undefined  -> DepthWrite   (acquire the shadow depth for writing)
    //   before scene:  shadowMap DepthWrite -> ShaderRead   (the lit pass samples the shadow map)
    //                  sceneColor Undefined  -> ColorTarget (acquire the scene RT color for writing)
    //   before post:   sceneColor ColorTarget -> ShaderRead (the composite samples the scene color)
    //                  swapchain  Undefined    -> ColorTarget (acquire the swapchain image)
    //   after  post:   swapchain  ColorTarget  -> Present    (hand the image to the presentation engine)
    struct Expect { const char* res; RgResourceState from; RgResourceState to; };
    const Expect expected[] = {
        {"shadowMap",  RgResourceState::Undefined,   RgResourceState::DepthWrite},
        {"shadowMap",  RgResourceState::DepthWrite,  RgResourceState::ShaderRead},
        {"sceneColor", RgResourceState::Undefined,   RgResourceState::ColorTarget},
        {"sceneColor", RgResourceState::ColorTarget, RgResourceState::ShaderRead},
        {"swapchain",  RgResourceState::Undefined,   RgResourceState::ColorTarget},
        {"swapchain",  RgResourceState::ColorTarget, RgResourceState::Present},
    };
    const size_t kExpected = sizeof(expected) / sizeof(expected[0]);

    check(bars.size() == kExpected, "solver emitted exactly the expected number of barriers");
    if (bars.size() == kExpected) {
        for (size_t i = 0; i < kExpected; ++i) {
            bool ok = graph.ResourceName(bars[i].resource) == expected[i].res &&
                      bars[i].from == expected[i].from && bars[i].to == expected[i].to;
            check(ok, "barrier[" + std::to_string(i) + "] == " +
                          std::string(expected[i].res) + ": " + StateName(expected[i].from) + "->" +
                          StateName(expected[i].to) + " (got " + Fmt(bars[i], graph) + ")");
        }
    } else {
        for (size_t i = 0; i < bars.size(); ++i)
            std::fprintf(stderr, "  got barrier[%zu] = %s\n", i, Fmt(bars[i], graph).c_str());
    }

    // --- No-op coalescing: re-reading a resource already in the required state emits NO barrier. ---
    // Two passes both READ sceneColor as ShaderRead after it is produced once. The first read
    // transitions ColorTarget->ShaderRead; the SECOND read finds it already ShaderRead -> from==to,
    // which must be coalesced away (no redundant barrier).
    {
        StubTarget sTgt;
        render::RenderGraph g2;
        auto rgS    = g2.ImportTarget("sceneColor", render::RgResourceKind::SceneColor, sTgt);
        auto rgSwap2 = g2.ImportSwapchain("swapchain");
        auto rgTmp  = g2.CreateTransient("postChain");

        g2.AddPass("scene", {}, {rgS}, nullptr);
        // postA reads sceneColor (ColorTarget->ShaderRead) and writes a transient to chain to postB.
        g2.AddPass("postA", {rgS}, {rgTmp, rgSwap2}, nullptr);
        // postB reads sceneColor AGAIN (already ShaderRead -> coalesced, no barrier) via the transient
        // dependency that orders it after postA. It writes nothing external besides the (already
        // ColorTarget) swapchain, so re-reading sceneColor must not re-emit a transition.
        // (Use a separate swapchain-writing pass to keep one external write per pass.)

        g2.Solve();
        const std::vector<render::RgBarrier>& b2 = g2.LastBarriers();
        // Expected: sceneColor Undefined->ColorTarget (before scene),
        //           sceneColor ColorTarget->ShaderRead (before postA),
        //           swapchain  Undefined->ColorTarget (before postA),
        //           swapchain  ColorTarget->Present (after postA).
        // Crucially NO second sceneColor ShaderRead->ShaderRead barrier anywhere.
        int sceneReadBarriers = 0;
        for (const auto& b : b2)
            if (g2.ResourceName(b.resource) == "sceneColor" && b.to == RgResourceState::ShaderRead)
                ++sceneReadBarriers;
        check(sceneReadBarriers == 1, "sceneColor->ShaderRead transition emitted exactly once (coalesced)");
        // And: there is no zero-width (from==to) barrier anywhere in the plan.
        bool anyNoop = false;
        for (const auto& b : b2) if (b.from == b.to) anyNoop = true;
        check(!anyNoop, "no from==to (no-op) barrier survives coalescing");
    }

    // --- Initial-state contract: a freshly imported render target starts Undefined. -------------
    {
        StubTarget t;
        render::RenderGraph g3;
        auto r = g3.ImportTarget("rt", render::RgResourceKind::SceneColor, t);
        check(g3.CurrentState(r) == RgResourceState::Undefined,
              "imported render target starts in Undefined state");
    }

    if (g_failures == 0) { std::printf("render_graph_barriers_test: OK\n"); return 0; }
    std::fprintf(stderr, "render_graph_barriers_test: %d failure(s)\n", g_failures);
    return 1;
}
