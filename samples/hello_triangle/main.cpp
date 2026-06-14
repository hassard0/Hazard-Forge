#include "hal/window.h"
#include "rhi/rhi.h"
#include "rhi/rhi_factory.h"
#include "math/math.h"

#include <chrono>
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

struct Vertex { float pos[3]; float color[3]; };

} // namespace

int main() {
    using namespace hf;
    try {
        hal::Window window({"Hazard Forge — Hello Cube", 1280, 720});
        auto device = rhi::CreateDevice(rhi::Backend::Vulkan, window);

        auto vsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/cube.vert.hlsl.spv");
        auto fsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/cube.frag.hlsl.spv");
        auto vs = device->CreateShaderModule({std::span<const uint32_t>(vsWords)});
        auto fs = device->CreateShaderModule({std::span<const uint32_t>(fsWords)});

        rhi::VertexLayout layout;
        layout.stride = sizeof(Vertex);
        layout.attributes = {
            {0, rhi::Format::RGB32_Float, offsetof(Vertex, pos)},
            {1, rhi::Format::RGB32_Float, offsetof(Vertex, color)},
        };

        rhi::GraphicsPipelineDesc pdesc;
        pdesc.vertex = vs.get();
        pdesc.fragment = fs.get();
        pdesc.vertexLayout = layout;
        pdesc.colorFormat = device->Swapchain().ColorFormat();
        pdesc.depthTest = true;
        pdesc.pushConstantSize = sizeof(float) * 16;
        auto pipeline = device->CreateGraphicsPipeline(pdesc);

        // 8 corners at +/-0.5 per axis; color = (pos + 0.5) so each corner is distinct.
        // Index encodes sign bits: bit0=x, bit1=y, bit2=z (0 => -0.5, 1 => +0.5).
        //   0:(-,-,-) 1:(+,-,-) 2:(-,+,-) 3:(+,+,-)
        //   4:(-,-,+) 5:(+,-,+) 6:(-,+,+) 7:(+,+,+)
        Vertex verts[8];
        for (int i = 0; i < 8; ++i) {
            float x = (i & 1) ? 0.5f : -0.5f;
            float y = (i & 2) ? 0.5f : -0.5f;
            float z = (i & 4) ? 0.5f : -0.5f;
            verts[i] = {{x, y, z}, {x + 0.5f, y + 0.5f, z + 0.5f}};
        }

        // 36 indices, 2 triangles per face, wound CCW when viewed from outside (RH).
        const uint32_t indices[36] = {
            // -Z (back), outward normal -Z: CCW seen from -Z
            0, 2, 3,  0, 3, 1,
            // +Z (front), outward normal +Z: CCW seen from +Z
            4, 5, 7,  4, 7, 6,
            // -X (left), outward normal -X
            0, 4, 6,  0, 6, 2,
            // +X (right), outward normal +X
            1, 3, 7,  1, 7, 5,
            // -Y (bottom), outward normal -Y
            0, 1, 5,  0, 5, 4,
            // +Y (top), outward normal +Y
            2, 6, 7,  2, 7, 3,
        };

        rhi::BufferDesc vbdesc;
        vbdesc.size = sizeof(verts);
        vbdesc.initialData = verts;
        vbdesc.usage = rhi::BufferUsage::Vertex;
        auto vbuffer = device->CreateBuffer(vbdesc);

        rhi::BufferDesc ibdesc;
        ibdesc.size = sizeof(indices);
        ibdesc.initialData = indices;
        ibdesc.usage = rhi::BufferUsage::Index;
        auto ibuffer = device->CreateBuffer(ibdesc);

        const auto start = std::chrono::steady_clock::now();

        bool running = true;
        while (running) {
            running = window.PumpEvents();
            if (window.ConsumeResized()) {
                device->WaitIdle();
                device->Swapchain().Recreate(window.FramebufferWidth(),
                                             window.FramebufferHeight());
            }

            float t = std::chrono::duration<float>(
                          std::chrono::steady_clock::now() - start).count();

            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            using math::Mat4;
            using math::Vec3;
            Mat4 model = Mat4::RotateY(t) * Mat4::RotateX(t * 0.5f);
            Mat4 view  = Mat4::LookAt({2, 2, 4}, {0, 0, 0}, {0, 1, 0});
            Mat4 proj  = Mat4::Perspective(1.04719755f /*60deg*/, aspect, 0.1f, 100.0f);
            Mat4 mvp   = proj * view * model;

            auto frame = device->BeginFrame();
            if (frame.cmd) {
                frame.cmd->BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1.0f});
                frame.cmd->BindPipeline(*pipeline);
                frame.cmd->PushConstants(mvp.m, sizeof(float) * 16);
                frame.cmd->BindVertexBuffer(*vbuffer);
                frame.cmd->BindIndexBuffer(*ibuffer);
                frame.cmd->DrawIndexed(36);
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
