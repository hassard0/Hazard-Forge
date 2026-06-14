#include "hal/window.h"
#include "rhi/rhi.h"
#include "rhi/rhi_factory.h"
#include "math/math.h"
#include "scene/vertex.h"
#include "scene/mesh.h"
#include "scene/transform.h"
#include "scene/renderable.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// Per-frame uniform block — must match shaders/lit.*.hlsl + shadow.vert.hlsl + sky.frag.hlsl
// FrameData (352 bytes; kFrameUboSize is 512 so it fits).
struct FrameData {
    float vp[16];
    float lightDir[4];
    float lightColor[4];
    float viewPos[4];
    float ptCount[4];          // x = number of active point lights
    float ptPos[3][4];         // xyz = world position, w = radius
    float ptColor[3][4];       // rgb = color, w = intensity
    float lightViewProj[16];   // directional light's view*ortho (for shadow mapping)
    float camFwd[4];           // camera basis (world space) for sky view-ray reconstruction
    float camRight[4];
    float camUp[4];
    float skyParams[4];        // x = tan(0.5*fovY), y = aspect
};

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

static bool WriteBMP(const char* path, const std::vector<uint8_t>& bgra,
                     uint32_t w, uint32_t h) {
    // 32bpp BI_RGB, bottom-up. Captured data is top-row-first BGRA; BMP wants bottom-up.
    uint32_t imgSize = w * h * 4;
    uint32_t fileSize = 54 + imgSize;
    uint8_t fh[14] = {0}; uint8_t ih[40] = {0};
    fh[0]='B'; fh[1]='M';
    fh[2]=fileSize; fh[3]=fileSize>>8; fh[4]=fileSize>>16; fh[5]=fileSize>>24;
    fh[10]=54;
    ih[0]=40;
    ih[4]=w; ih[5]=w>>8; ih[6]=w>>16; ih[7]=w>>24;
    ih[8]=h; ih[9]=h>>8; ih[10]=h>>16; ih[11]=h>>24;
    ih[12]=1; ih[14]=32;            // planes=1, bpp=32
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    std::fwrite(fh, 1, 14, f); std::fwrite(ih, 1, 40, f);
    for (int y = (int)h - 1; y >= 0; --y)
        std::fwrite(&bgra[(size_t)y * w * 4], 1, w * 4, f);
    std::fclose(f);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    using namespace hf;

    // --shot <path>: render one frame headless, write a BMP, and exit (no present loop).
    const char* shotPath = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) {
            shotPath = argv[i + 1];
            break;
        }
    }
    try {
        hal::Window window({"Hazard Forge — Shadows", 1280, 720});
        auto device = rhi::CreateDevice(rhi::Backend::Vulkan, window);

        auto vsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
        auto fsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
        auto vs = device->CreateShaderModule({std::span<const uint32_t>(vsWords)});
        auto fs = device->CreateShaderModule({std::span<const uint32_t>(fsWords)});

        rhi::GraphicsPipelineDesc pdesc;
        pdesc.vertex = vs.get();
        pdesc.fragment = fs.get();
        pdesc.vertexLayout = scene::MeshVertexLayout();
        pdesc.colorFormat = device->Swapchain().ColorFormat();
        pdesc.depthTest = true;
        pdesc.usesFrameUniforms = true;
        pdesc.usesTexture = true;
        pdesc.pushConstantSize = sizeof(float) * 20;  // mat4 model + float4 material (metallic,roughness)
        auto pipeline = device->CreateGraphicsPipeline(pdesc);

        // Depth-only shadow pipeline: renders the scene from the light into the shadow map.
        // Needs lightViewProj from the frame UBO (set 0 b0); no texture, no color attachment.
        auto shadowVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
        auto shadowFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
        auto shadowVs = device->CreateShaderModule({std::span<const uint32_t>(shadowVsWords)});
        auto shadowFs = device->CreateShaderModule({std::span<const uint32_t>(shadowFsWords)});

        rhi::GraphicsPipelineDesc shadowDesc;
        shadowDesc.vertex = shadowVs.get();
        shadowDesc.fragment = shadowFs.get();
        shadowDesc.vertexLayout = scene::MeshVertexLayout();
        shadowDesc.depthTest = true;
        shadowDesc.depthOnly = true;
        shadowDesc.usesFrameUniforms = true;   // lightViewProj lives in the frame UBO
        shadowDesc.usesTexture = false;
        shadowDesc.pushConstantSize = sizeof(float) * 16;  // mat4 model
        auto shadowPipeline = device->CreateGraphicsPipeline(shadowDesc);

        // 2048^2 depth-only shadow map; point the per-frame sets at it once.
        auto shadowMap = device->CreateShadowMap(2048);
        device->SetShadowMap(*shadowMap);

        // Fullscreen post pipeline: samples the offscreen RT, tonemaps + vignettes -> swapchain.
        auto postVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.vert.hlsl.spv");
        auto postFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.frag.hlsl.spv");
        auto postVs = device->CreateShaderModule({std::span<const uint32_t>(postVsWords)});
        auto postFs = device->CreateShaderModule({std::span<const uint32_t>(postFsWords)});

        rhi::GraphicsPipelineDesc postDesc;
        postDesc.vertex = postVs.get();
        postDesc.fragment = postFs.get();
        postDesc.colorFormat = device->Swapchain().ColorFormat();
        postDesc.depthTest = false;          // fullscreen pass, no depth
        postDesc.usesFrameUniforms = false;  // no per-frame UBO -> material set is set 0
        postDesc.usesTexture = true;         // samples the RT
        postDesc.fullscreen = true;          // no vertex input; 3 verts from SV_VertexID
        auto postPipeline = device->CreateGraphicsPipeline(postDesc);

        // Procedural sky pipeline: fullscreen triangle drawn FIRST in the scene->RT pass. Reads the
        // camera basis from the per-frame UBO (set 0 b0); writes no depth so geometry draws over it.
        auto skyVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.vert.hlsl.spv");
        auto skyFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.frag.hlsl.spv");
        auto skyVs = device->CreateShaderModule({std::span<const uint32_t>(skyVsWords)});
        auto skyFs = device->CreateShaderModule({std::span<const uint32_t>(skyFsWords)});

        rhi::GraphicsPipelineDesc skyDesc;
        skyDesc.vertex = skyVs.get();
        skyDesc.fragment = skyFs.get();
        skyDesc.colorFormat = device->Swapchain().ColorFormat();
        skyDesc.depthTest = false;           // background fill: no depth test, no depth write
        skyDesc.usesFrameUniforms = true;    // camera basis lives in the frame UBO (set 0 b0)
        skyDesc.usesTexture = false;
        skyDesc.fullscreen = true;           // 3 verts from SV_VertexID; cull NONE
        auto skyPipeline = device->CreateGraphicsPipeline(skyDesc);

        // Offscreen render target sized to the framebuffer; recreated on resize.
        auto rt = device->CreateRenderTarget(window.FramebufferWidth(),
                                             window.FramebufferHeight());

        // Procedural checkerboard texture (256x256 RGBA8), shared by all renderables.
        std::vector<uint8_t> pixels = MakeCheckerboard();
        auto texture = device->CreateTexture(
            {256, 256, rhi::Format::RGBA8_UNorm, pixels.data(), pixels.size()});

        // Two primitive meshes from the scene layer.
        scene::Mesh cube = scene::Mesh::Cube(*device);
        scene::Mesh plane = scene::Mesh::Plane(*device);
        scene::Mesh sphere = scene::Mesh::Sphere(*device);

        // Build the scene: a large ground plane + a 3x3 grid of lit cubes.
        std::vector<scene::Renderable> sceneObjects;
        {
            scene::Transform planeT;
            planeT.position = {0.0f, 0.0f, 0.0f};
            planeT.scale = {6.0f, 1.0f, 6.0f};
            // Ground plane: rough dielectric.
            sceneObjects.push_back({&plane, texture.get(), planeT, /*metallic*/ 0.0f, /*roughness*/ 0.8f});

            for (int gx = -1; gx <= 1; ++gx) {
                for (int gz = -1; gz <= 1; ++gz) {
                    // Replace the three cells on the main diagonal with spheres so
                    // the scene shows smooth curved geometry alongside the cubes.
                    bool useSphere = (gx == gz);
                    scene::Transform t;
                    if (useSphere) {
                        t.position = {gx * 1.8f, 0.55f, gz * 1.8f};
                        t.scale = {0.55f, 0.55f, 0.55f};
                        // Shiny metal spheres.
                        sceneObjects.push_back({&sphere, texture.get(), t, /*metallic*/ 1.0f, /*roughness*/ 0.15f});
                    } else {
                        t.position = {gx * 1.8f, 0.6f, gz * 1.8f};
                        t.eulerRadians = {0.0f, (gx + gz) * 0.5f, 0.0f};
                        t.scale = {0.5f, 0.5f, 0.5f};
                        // Matte dielectric cubes.
                        sceneObjects.push_back({&cube, texture.get(), t, /*metallic*/ 0.0f, /*roughness*/ 0.5f});
                    }
                }
            }
        }

        using math::Mat4;
        using math::Vec3;
        const Vec3 eye{4.5f, 4.0f, 6.5f};
        const Vec3 center{0.0f, 0.5f, 0.0f};

        // Fill FrameData (viewProj + directional light + camera pos) for a given aspect.
        auto makeFrameData = [&](float aspect, float t) {
            Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
            Mat4 proj = Mat4::Perspective(1.04719755f /*60deg*/, aspect, 0.1f, 100.0f);
            Mat4 vp = proj * view;
            FrameData fd{};
            for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
            // Strong key directional (warm white) so cast shadows read clearly.
            fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f; fd.lightDir[3] = 0.0f;
            fd.lightColor[0] = 0.95f; fd.lightColor[1] = 0.93f; fd.lightColor[2] = 0.85f; fd.lightColor[3] = 1.0f;
            fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;

            // Three colored point lights as tight accents (kept modest so shadows stay visible).
            fd.ptCount[0] = 3.0f;
            const float kR = 3.0f, kH = 1.1f, kRadius = 3.2f, kInt = 1.0f;
            const float colors[3][3] = {{1.0f, 0.25f, 0.2f},   // warm red
                                        {0.2f, 1.0f, 0.35f},   // green
                                        {0.3f, 0.45f, 1.0f}};  // blue
            for (int li = 0; li < 3; ++li) {
                float a = t * 0.9f + (float)li * 2.0943951f;  // 120deg apart
                fd.ptPos[li][0] = std::cos(a) * kR;
                fd.ptPos[li][1] = kH;
                fd.ptPos[li][2] = std::sin(a) * kR;
                fd.ptPos[li][3] = kRadius;
                fd.ptColor[li][0] = colors[li][0];
                fd.ptColor[li][1] = colors[li][1];
                fd.ptColor[li][2] = colors[li][2];
                fd.ptColor[li][3] = kInt;
            }

            // Directional light's view*projection for shadow mapping. Light looks from
            // sceneCenter - lightDir*12 toward sceneCenter; an ortho box covers the scene.
            Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
            Vec3 sceneCenter{0.0f, 0.5f, 0.0f};
            Vec3 lightEye = sceneCenter - lightDir * 12.0f;
            Mat4 lightView = Mat4::LookAt(lightEye, sceneCenter, {0, 1, 0});
            Mat4 lightOrtho = Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 25.0f);
            Mat4 lightVP = lightOrtho * lightView;
            for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];

            // Camera basis (world space) so the sky shader reconstructs view rays without a
            // matrix inverse. Mirrors LookAt: fwd toward center, right = fwd x worldUp, up = right x fwd.
            Vec3 fwd = math::normalize(center - eye);
            Vec3 right = math::normalize(math::cross(fwd, Vec3{0.0f, 1.0f, 0.0f}));
            Vec3 up = math::cross(right, fwd);
            fd.camFwd[0] = fwd.x;   fd.camFwd[1] = fwd.y;   fd.camFwd[2] = fwd.z;   fd.camFwd[3] = 0.0f;
            fd.camRight[0] = right.x; fd.camRight[1] = right.y; fd.camRight[2] = right.z; fd.camRight[3] = 0.0f;
            fd.camUp[0] = up.x;     fd.camUp[1] = up.y;     fd.camUp[2] = up.z;     fd.camUp[3] = 0.0f;
            fd.skyParams[0] = std::tan(0.5f * 1.04719755f);  // tan(half of 60deg fovY)
            fd.skyParams[1] = aspect;
            fd.skyParams[2] = 0.0f; fd.skyParams[3] = 0.0f;
            return fd;
        };

        // Record every renderable in the scene into the command buffer.
        auto drawScene = [&](rhi::ICommandBuffer* cmd) {
            cmd->BindPipeline(*pipeline);
            for (scene::Renderable& r : sceneObjects) {
                Mat4 m = r.transform.Matrix();
                // Push { float4x4 model; float4 material(metallic,roughness,0,0) } = 80 bytes.
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                pc[16] = r.metallic; pc[17] = r.roughness; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd->PushConstants(pc, sizeof(pc));
                cmd->BindTexture(*r.texture);
                cmd->BindVertexBuffer(r.mesh->vertices());
                cmd->BindIndexBuffer(r.mesh->indices());
                cmd->DrawIndexed(r.mesh->indexCount());
            }
        };

        // Depth-only shadow draw: geometry only (no texture), light-space via the shadow pipeline.
        auto drawDepthOnly = [&](rhi::ICommandBuffer* cmd) {
            cmd->BindPipeline(*shadowPipeline);
            for (scene::Renderable& r : sceneObjects) {
                Mat4 m = r.transform.Matrix();
                cmd->PushConstants(m.m, sizeof(float) * 16);
                cmd->BindVertexBuffer(r.mesh->vertices());
                cmd->BindIndexBuffer(r.mesh->indices());
                cmd->DrawIndexed(r.mesh->indexCount());
            }
        };

        // --- Headless capture mode: render exactly one frame and write it to a BMP. ---
        if (shotPath) {
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
            FrameData fd = makeFrameData(aspect, 0.6f);  // fixed camera; cubes + lights static

            // Pass 0: depth-only shadow pass from the light into the shadow map.
            {
                auto sc = device->BeginShadowPass(*shadowMap);
                device->SetFrameUniforms(&fd, sizeof(FrameData));  // fd has lightViewProj
                sc.cmd->BeginRenderPass(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f});
                drawDepthOnly(sc.cmd);
                sc.cmd->EndRenderPass();
                device->EndShadowPass(sc);
            }

            // Pass 1: render the scene into the offscreen render target (lit, samples shadow map).
            {
                auto rtc = device->BeginRenderTargetFrame(*rt);
                device->SetFrameUniforms(&fd, sizeof(FrameData));
                rtc.cmd->BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1.0f});
                // Sky first: fullscreen gradient + sun, no depth write, behind the geometry.
                rtc.cmd->BindPipeline(*skyPipeline);
                rtc.cmd->Draw(3);
                drawScene(rtc.cmd);
                rtc.cmd->EndRenderPass();
                device->EndRenderTargetFrame(rtc);
            }

            // Pass 2: fullscreen post pass samples the RT into the swapchain (then captured).
            device->CaptureNextFrame();
            auto frame = device->BeginFrame();
            if (!frame.cmd) {
                // A fresh swapchain may report out-of-date on the first acquire; retry once.
                device->CaptureNextFrame();
                frame = device->BeginFrame();
            }
            if (!frame.cmd) {
                std::fprintf(stderr, "FATAL: could not acquire a frame for capture "
                                     "(swapchain out-of-date)\n");
                return 1;
            }
            frame.cmd->BeginRenderPass(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f});
            frame.cmd->BindPipeline(*postPipeline);
            frame.cmd->BindTexture(*rt);
            frame.cmd->Draw(3);
            frame.cmd->EndRenderPass();
            device->EndFrame(frame);

            std::vector<uint8_t> px;
            uint32_t cw = 0, ch = 0;
            if (device->GetCapturedPixels(px, cw, ch)) {
                if (!WriteBMP(shotPath, px, cw, ch)) {
                    std::fprintf(stderr, "FATAL: could not write BMP to %s\n", shotPath);
                    device->WaitIdle();
                    return 1;
                }
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
                device->WaitIdle();
                return 1;
            }
            device->WaitIdle();
            std::printf("wrote %s (%ux%u)\n", shotPath, cw, ch);
            return 0;
        }

        // Capture each renderable's authored base yaw so the interactive loop can spin
        // the cubes by adding elapsed time on top without drifting.
        std::vector<float> baseYaw(sceneObjects.size());
        for (size_t i = 0; i < sceneObjects.size(); ++i)
            baseYaw[i] = sceneObjects[i].transform.eulerRadians.y;

        const auto start = std::chrono::steady_clock::now();

        bool running = true;
        while (running) {
            running = window.PumpEvents();
            if (window.ConsumeResized()) {
                device->WaitIdle();
                device->Swapchain().Recreate(window.FramebufferWidth(),
                                             window.FramebufferHeight());
                // Recreate the offscreen RT to match the new framebuffer size.
                rt = device->CreateRenderTarget(window.FramebufferWidth(),
                                                window.FramebufferHeight());
            }

            float t = std::chrono::duration<float>(
                          std::chrono::steady_clock::now() - start).count();

            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
            FrameData fd = makeFrameData(aspect, t);

            // Spin each cube about its yaw over time (skip index 0: the ground plane).
            for (size_t i = 1; i < sceneObjects.size(); ++i)
                sceneObjects[i].transform.eulerRadians.y = baseYaw[i] + t;

            // Pass 0: depth-only shadow pass from the light into the shadow map.
            {
                auto sc = device->BeginShadowPass(*shadowMap);
                device->SetFrameUniforms(&fd, sizeof(FrameData));  // fd has lightViewProj
                sc.cmd->BeginRenderPass(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f});
                drawDepthOnly(sc.cmd);
                sc.cmd->EndRenderPass();
                device->EndShadowPass(sc);
            }

            // Pass 1: render the scene into the offscreen render target (lit, samples shadow map).
            {
                auto rtc = device->BeginRenderTargetFrame(*rt);
                device->SetFrameUniforms(&fd, sizeof(FrameData));
                rtc.cmd->BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1.0f});
                // Sky first: fullscreen gradient + sun, no depth write, behind the geometry.
                rtc.cmd->BindPipeline(*skyPipeline);
                rtc.cmd->Draw(3);
                drawScene(rtc.cmd);
                rtc.cmd->EndRenderPass();
                device->EndRenderTargetFrame(rtc);
            }

            // Pass 2: fullscreen post pass samples the RT into the swapchain.
            auto frame = device->BeginFrame();
            if (frame.cmd) {
                frame.cmd->BeginRenderPass(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f});
                frame.cmd->BindPipeline(*postPipeline);
                frame.cmd->BindTexture(*rt);
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
