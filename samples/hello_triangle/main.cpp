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

struct Vertex { float pos[3]; float color[3]; float uv[2]; };

// Procedural 256x256 RGBA8 checkerboard: 8x8 tiles alternating two colors, with the
// base color also shifted by tile position so the grid is clearly readable.
std::vector<uint8_t> MakeCheckerboard() {
    const uint32_t kSize = 256;
    const uint32_t kTiles = 8;
    const uint32_t kTilePx = kSize / kTiles;  // 32px per tile
    std::vector<uint8_t> pixels(static_cast<size_t>(kSize) * kSize * 4);
    for (uint32_t y = 0; y < kSize; ++y) {
        for (uint32_t x = 0; x < kSize; ++x) {
            uint32_t tx = x / kTilePx;
            uint32_t ty = y / kTilePx;
            bool dark = ((tx + ty) & 1) != 0;
            // Position-varying tint so neighbouring tiles differ even within one parity.
            uint8_t px = static_cast<uint8_t>(40 + tx * (215 / (kTiles - 1)));
            uint8_t py = static_cast<uint8_t>(40 + ty * (215 / (kTiles - 1)));
            size_t idx = (static_cast<size_t>(y) * kSize + x) * 4;
            if (dark) {
                pixels[idx + 0] = static_cast<uint8_t>(px / 4);
                pixels[idx + 1] = static_cast<uint8_t>(py / 4);
                pixels[idx + 2] = 60;
            } else {
                pixels[idx + 0] = px;
                pixels[idx + 1] = py;
                pixels[idx + 2] = 230;
            }
            pixels[idx + 3] = 255;
        }
    }
    return pixels;
}

} // namespace

int main() {
    using namespace hf;
    try {
        hal::Window window({"Hazard Forge — Textured Cube", 1280, 720});
        auto device = rhi::CreateDevice(rhi::Backend::Vulkan, window);

        auto vsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/cube_tex.vert.hlsl.spv");
        auto fsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/cube_tex.frag.hlsl.spv");
        auto vs = device->CreateShaderModule({std::span<const uint32_t>(vsWords)});
        auto fs = device->CreateShaderModule({std::span<const uint32_t>(fsWords)});

        rhi::VertexLayout layout;
        layout.stride = sizeof(Vertex);
        layout.attributes = {
            {0, rhi::Format::RGB32_Float, offsetof(Vertex, pos)},
            {1, rhi::Format::RGB32_Float, offsetof(Vertex, color)},
            {2, rhi::Format::RG32_Float,  offsetof(Vertex, uv)},
        };

        rhi::GraphicsPipelineDesc pdesc;
        pdesc.vertex = vs.get();
        pdesc.fragment = fs.get();
        pdesc.vertexLayout = layout;
        pdesc.colorFormat = device->Swapchain().ColorFormat();
        pdesc.depthTest = true;
        pdesc.usesTexture = true;
        pdesc.pushConstantSize = sizeof(float) * 16;
        auto pipeline = device->CreateGraphicsPipeline(pdesc);

        // 24 vertices (4 per face) so each face carries its own UV square + tint.
        // Faces wound CCW outward (RH); UVs per face: (0,0)(1,0)(1,1)(0,1).
        const float n = -0.5f, p = 0.5f;
        const Vertex verts[24] = {
            // -Z (back), tint red
            {{p, n, n}, {1.0f, 0.4f, 0.4f}, {0, 0}},
            {{n, n, n}, {1.0f, 0.4f, 0.4f}, {1, 0}},
            {{n, p, n}, {1.0f, 0.4f, 0.4f}, {1, 1}},
            {{p, p, n}, {1.0f, 0.4f, 0.4f}, {0, 1}},
            // +Z (front), tint green
            {{n, n, p}, {0.4f, 1.0f, 0.4f}, {0, 0}},
            {{p, n, p}, {0.4f, 1.0f, 0.4f}, {1, 0}},
            {{p, p, p}, {0.4f, 1.0f, 0.4f}, {1, 1}},
            {{n, p, p}, {0.4f, 1.0f, 0.4f}, {0, 1}},
            // -X (left), tint blue
            {{n, n, n}, {0.4f, 0.4f, 1.0f}, {0, 0}},
            {{n, n, p}, {0.4f, 0.4f, 1.0f}, {1, 0}},
            {{n, p, p}, {0.4f, 0.4f, 1.0f}, {1, 1}},
            {{n, p, n}, {0.4f, 0.4f, 1.0f}, {0, 1}},
            // +X (right), tint yellow
            {{p, n, p}, {1.0f, 1.0f, 0.4f}, {0, 0}},
            {{p, n, n}, {1.0f, 1.0f, 0.4f}, {1, 0}},
            {{p, p, n}, {1.0f, 1.0f, 0.4f}, {1, 1}},
            {{p, p, p}, {1.0f, 1.0f, 0.4f}, {0, 1}},
            // -Y (bottom), tint magenta
            {{n, n, n}, {1.0f, 0.4f, 1.0f}, {0, 0}},
            {{p, n, n}, {1.0f, 0.4f, 1.0f}, {1, 0}},
            {{p, n, p}, {1.0f, 0.4f, 1.0f}, {1, 1}},
            {{n, n, p}, {1.0f, 0.4f, 1.0f}, {0, 1}},
            // +Y (top), tint cyan
            {{n, p, p}, {0.4f, 1.0f, 1.0f}, {0, 0}},
            {{p, p, p}, {0.4f, 1.0f, 1.0f}, {1, 0}},
            {{p, p, n}, {0.4f, 1.0f, 1.0f}, {1, 1}},
            {{n, p, n}, {0.4f, 1.0f, 1.0f}, {0, 1}},
        };

        // 36 indices: 2 triangles per face over its 4 vertices, CCW outward.
        uint32_t indices[36];
        for (uint32_t f = 0; f < 6; ++f) {
            uint32_t base = f * 4;
            uint32_t* tri = &indices[f * 6];
            tri[0] = base + 0; tri[1] = base + 1; tri[2] = base + 2;
            tri[3] = base + 0; tri[4] = base + 2; tri[5] = base + 3;
        }

        // Procedural checkerboard texture (256x256 RGBA8).
        std::vector<uint8_t> pixels = MakeCheckerboard();
        auto texture = device->CreateTexture(
            {256, 256, rhi::Format::RGBA8_UNorm, pixels.data(), pixels.size()});

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
                frame.cmd->BindTexture(*texture);
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
