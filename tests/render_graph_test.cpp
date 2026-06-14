// Unit test for the declarative render graph's dependency model + topological ordering.
// Pure logic — no GPU/RHI device is created. We feed passes in a SCRAMBLED declaration order with
// resource read/write dependencies and assert Execute() (via a stub device) runs them producer-
// before-consumer (shadow -> scene -> post), the canonical frame order.
#include "render/render_graph.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hf;

namespace {
int g_failures = 0;
void check(bool cond, const std::string& msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg.c_str()); ++g_failures; }
}

// A no-op stub render target so ImportTarget has something to bind. The graph only stores the
// pointer and hands it to Begin*; our stub device ignores it.
class StubTarget : public rhi::IRenderTarget {
public:
    uint32_t width() const override { return 1; }
    uint32_t height() const override { return 1; }
};

// A stub command buffer that records nothing (the test passes empty record callbacks, but the
// graph still hands a cmd to them).
class StubCmd : public rhi::ICommandBuffer {
public:
    void BeginRenderPass(const rhi::ClearColor&) override {}
    void BindPipeline(rhi::IPipeline&) override {}
    void BindVertexBuffer(rhi::IBuffer&) override {}
    void BindIndexBuffer(rhi::IBuffer&) override {}
    void BindTexture(rhi::ITexture&) override {}
    void Draw(uint32_t, uint32_t) override {}
    void DrawIndexed(uint32_t, uint32_t, int32_t) override {}
    void PushConstants(const void*, uint32_t) override {}
    void SetScissor(int32_t, int32_t, uint32_t, uint32_t) override {}
    void EndRenderPass() override {}
};

// A stub device: records which scaffolding Begin* call each pass was driven through, so we can
// assert the graph mapped each pass output to the correct RHI Begin/End pair.
class StubDevice : public rhi::IRHIDevice {
public:
    std::vector<std::string> beginCalls;
    StubCmd cmd;

    rhi::ISwapchain& Swapchain() override { throw std::runtime_error("unused"); }
    std::unique_ptr<rhi::IShaderModule> CreateShaderModule(const rhi::ShaderModuleDesc&) override { return nullptr; }
    std::unique_ptr<rhi::IPipeline> CreateGraphicsPipeline(const rhi::GraphicsPipelineDesc&) override { return nullptr; }
    std::unique_ptr<rhi::IBuffer> CreateBuffer(const rhi::BufferDesc&) override { return nullptr; }
    std::unique_ptr<rhi::ITexture> CreateTexture(const rhi::TextureDesc&) override { return nullptr; }
    std::unique_ptr<rhi::IRenderTarget> CreateRenderTarget(uint32_t, uint32_t) override { return nullptr; }
    std::unique_ptr<rhi::IRenderTarget> CreateShadowMap(uint32_t) override { return nullptr; }

    rhi::FrameContext BeginShadowPass(rhi::IRenderTarget&) override {
        beginCalls.push_back("ShadowPass"); return {&cmd};
    }
    void EndShadowPass(const rhi::FrameContext&) override {}
    void SetShadowMap(rhi::IRenderTarget&) override {}
    rhi::FrameContext BeginRenderTargetFrame(rhi::IRenderTarget&) override {
        beginCalls.push_back("RenderTargetFrame"); return {&cmd};
    }
    void EndRenderTargetFrame(const rhi::FrameContext&) override {}
    rhi::FrameContext BeginFrame() override {
        beginCalls.push_back("Frame"); return {&cmd};
    }
    void EndFrame(const rhi::FrameContext&) override {}
    void SetFrameUniforms(const void*, uint32_t) override {}
    void WaitIdle() override {}
    void CaptureNextFrame() override {}
    bool GetCapturedPixels(std::vector<uint8_t>&, uint32_t&, uint32_t&) override { return false; }
};
}  // namespace

int main() {
    StubTarget shadowTgt, sceneTgt;
    StubDevice device;

    render::RenderGraph graph;
    auto rgShadow = graph.ImportTarget("shadowMap", render::RgResourceKind::ShadowMap, shadowTgt);
    auto rgScene  = graph.ImportTarget("sceneColor", render::RgResourceKind::SceneColor, sceneTgt);
    auto rgSwap   = graph.ImportSwapchain("swapchain");

    auto noop = [](rhi::IRHIDevice&, rhi::ICommandBuffer&) {};

    // Declare passes in a SCRAMBLED order: post, scene, shadow. The dependency edges
    // (shadow writes shadowMap, scene reads it; scene writes sceneColor, post reads it) must
    // reorder them to shadow -> scene -> post regardless of declaration order.
    graph.AddPass("post",   /*reads*/{rgScene},  /*writes*/{rgSwap},   noop);
    graph.AddPass("scene",  /*reads*/{rgShadow}, /*writes*/{rgScene},  noop);
    graph.AddPass("shadow", /*reads*/{},         /*writes*/{rgShadow}, noop);

    graph.Execute(device);

    const std::vector<std::string>& order = graph.LastOrder();
    check(order.size() == 3, "three passes executed");
    if (order.size() == 3) {
        check(order[0] == "shadow", "shadow runs first (producer of shadowMap)");
        check(order[1] == "scene",  "scene runs second (reads shadowMap, writes sceneColor)");
        check(order[2] == "post",   "post runs last (reads sceneColor)");
    }

    // The graph must have driven each pass through the RHI scaffolding selected by its output kind.
    check(device.beginCalls.size() == 3, "three Begin* scaffolding calls");
    if (device.beginCalls.size() == 3) {
        check(device.beginCalls[0] == "ShadowPass", "shadow -> BeginShadowPass");
        check(device.beginCalls[1] == "RenderTargetFrame", "scene -> BeginRenderTargetFrame");
        check(device.beginCalls[2] == "Frame", "post -> BeginFrame");
    }

    // A pass with no external write must be rejected (malformed output).
    {
        render::RenderGraph bad;
        auto t = bad.CreateTransient("tmp");
        bad.AddPass("noOutput", {}, {t}, noop);
        bool threw = false;
        try { StubDevice d2; bad.Execute(d2); } catch (const std::exception&) { threw = true; }
        check(threw, "pass with no external write is rejected");
    }

    if (g_failures == 0) { std::printf("render_graph_test: OK\n"); return 0; }
    std::fprintf(stderr, "render_graph_test: %d failure(s)\n", g_failures);
    return 1;
}
