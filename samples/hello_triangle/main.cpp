#include "hal/window.h"
#include "rhi/rhi.h"
#include "rhi/rhi_factory.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<uint32_t> LoadSpirv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open shader: " + path);
    std::streamsize size = f.tellg();
    if (size % 4 != 0) throw std::runtime_error("SPIR-V size not multiple of 4: " + path);
    f.seekg(0);
    std::vector<uint32_t> words(size / 4);
    f.read(reinterpret_cast<char*>(words.data()), size);
    return words;
}

struct Vertex { float pos[2]; float color[3]; };

} // namespace

int main() {
    using namespace hf;
    try {
        hal::Window window({"Hazard Forge — Hello Triangle", 1280, 720});
        auto device = rhi::CreateDevice(rhi::Backend::Vulkan, window);

        auto vsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/triangle.vert.hlsl.spv");
        auto fsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/triangle.frag.hlsl.spv");
        auto vs = device->CreateShaderModule({std::span<const uint32_t>(vsWords)});
        auto fs = device->CreateShaderModule({std::span<const uint32_t>(fsWords)});

        rhi::VertexLayout layout;
        layout.stride = sizeof(Vertex);
        layout.attributes = {
            {0, rhi::Format::RG32_Float,  offsetof(Vertex, pos)},
            {1, rhi::Format::RGB32_Float, offsetof(Vertex, color)},
        };

        rhi::GraphicsPipelineDesc pdesc;
        pdesc.vertex = vs.get();
        pdesc.fragment = fs.get();
        pdesc.vertexLayout = layout;
        pdesc.colorFormat = device->Swapchain().ColorFormat();
        auto pipeline = device->CreateGraphicsPipeline(pdesc);

        const Vertex verts[3] = {
            {{ 0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
            {{ 0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},
            {{-0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
        };
        rhi::BufferDesc bdesc;
        bdesc.size = sizeof(verts);
        bdesc.initialData = verts;
        auto vbuffer = device->CreateBuffer(bdesc);

        bool running = true;
        while (running) {
            running = window.PumpEvents();
            if (window.ConsumeResized()) {
                device->WaitIdle();
                device->Swapchain().Recreate(window.FramebufferWidth(),
                                             window.FramebufferHeight());
            }
            auto frame = device->BeginFrame();
            if (frame.cmd) {
                frame.cmd->BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1.0f});
                frame.cmd->BindPipeline(*pipeline);
                frame.cmd->BindVertexBuffer(*vbuffer);
                frame.cmd->Draw(3);
                frame.cmd->EndRenderPass();
            }
            device->EndFrame(frame);
        }

        device->WaitIdle();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
