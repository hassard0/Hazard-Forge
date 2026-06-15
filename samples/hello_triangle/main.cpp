#include "hal/window.h"
#include "rhi/rhi.h"
#include "rhi/rhi_factory.h"
#include "math/math.h"
#include "scene/vertex.h"
#include "scene/mesh.h"
#include "scene/instance_grid.h"
#include "scene/transform.h"
#include "scene/renderable.h"
#include "scene/components.h"
#include "scene/scene_io.h"
#include "scene/commands.h"
#include "ecs/ecs.h"
#include "asset/gltf_loader.h"
#include "asset/env_loader.h"
#include "anim/animation.h"
#include "anim/skeleton.h"
#include "render/render_graph.h"

#ifdef HF_HAS_EDITOR
#include "imgui.h"
#include "editor/imgui_renderer.h"
#include "editor/editor_panels.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
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

// Procedural 256x256 RGBA8 tangent-space normal map: an 8x8 grid of radial domes (one per
// checkerboard tile). Each tile's height is a smooth dome h(r) = cos(pi/2 * r/R)^2 falling to 0 at
// the tile edge; the tangent-space normal is derived from the height gradient and encoded 0..1
// (flat = (128,128,255)). Tiling makes the bumps read clearly across the cube faces / ground plane.
std::vector<uint8_t> MakeBumpyNormalMap() {
    const uint32_t kSize = 256;
    const uint32_t kTiles = 8;
    const float kTilePx = (float)kSize / (float)kTiles;  // 32px per tile
    const float kBump = 4.0f;  // gradient strength -> bump steepness (visibly domed tiles)
    auto height = [&](float x, float y) {
        // Local coords within the tile, centered; radius normalized to the half-tile.
        float lx = std::fmod(x, kTilePx) / kTilePx - 0.5f;
        float ly = std::fmod(y, kTilePx) / kTilePx - 0.5f;
        float r = std::sqrt(lx * lx + ly * ly) / 0.5f;  // 0 at center, ~1 at edge
        if (r >= 1.0f) return 0.0f;
        float c = std::cos(1.5707963f * r);
        return c * c;  // smooth dome
    };
    std::vector<uint8_t> px(static_cast<size_t>(kSize) * kSize * 4);
    for (uint32_t y = 0; y < kSize; ++y) {
        for (uint32_t x = 0; x < kSize; ++x) {
            // Central-difference height gradient (wraps at the texture edge for seamless tiling).
            float xl = (float)((x + kSize - 1) % kSize), xr = (float)((x + 1) % kSize);
            float yd = (float)((y + kSize - 1) % kSize), yu = (float)((y + 1) % kSize);
            float dhdx = (height(xr, (float)y) - height(xl, (float)y)) * 0.5f;
            float dhdy = (height((float)x, yu) - height((float)x, yd)) * 0.5f;
            // Tangent-space normal from the height field: N = normalize(-dh/dx, -dh/dy, 1).
            float nx = -dhdx * kBump, ny = -dhdy * kBump, nz = 1.0f;
            float inv = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
            nx *= inv; ny *= inv; nz *= inv;
            size_t idx = (static_cast<size_t>(y) * kSize + x) * 4;
            px[idx + 0] = (uint8_t)std::lround((nx * 0.5f + 0.5f) * 255.0f);
            px[idx + 1] = (uint8_t)std::lround((ny * 0.5f + 0.5f) * 255.0f);
            px[idx + 2] = (uint8_t)std::lround((nz * 0.5f + 0.5f) * 255.0f);
            px[idx + 3] = 255;
        }
    }
    return px;
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
    // --dump-scene: load the scene from data, print DumpScene(...) JSON to stdout, exit 0 (headless,
    //               no render — lets an agent inspect the machine-readable scene state).
    // --commands <file.json>: build the default scene, then apply the JSON command script (mutate the
    //               live ECS, dump/list to stdout, capture renders via the offscreen --shot path),
    //               then exit 0. This is the agentic command channel — an AI agent/CI drives the
    //               engine headlessly from a script. Separate from --editor / interactive.
    const char* shotPath = nullptr;
    const char* skinningShotPath = nullptr;
    const char* pbrShotPath = nullptr;
    const char* iblShotPath = nullptr;
    const char* instancedShotPath = nullptr;
    const char* commandsPath = nullptr;
    bool dumpScene = false;
    bool editor = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) {
            shotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--pbr-shot") == 0 && i + 1 < argc) {
            // Render one frame of the full-PBR showcase (ground + skybox + DamagedHelmet with the
            // full glTF metallic-roughness material set, lit + shadowed), write a BMP, exit.
            pbrShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--ibl-shot") == 0 && i + 1 < argc) {
            // Render one frame of the HDR-IBL showcase (HDR equirect skybox + DamagedHelmet shaded by
            // lit_pbr_ibl so the metal reflects the real captured sky/sun/terrain), write a BMP, exit.
            iblShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--skinning-shot") == 0 && i + 1 < argc) {
            // Render one frame of the skinned-Fox showcase (ground + skybox + GPU-skinned Fox at
            // animation "Survey" t=0.5s, lit + shadowed), write a BMP, exit.
            skinningShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--instanced-shot") == 0 && i + 1 < argc) {
            // Render one frame of the GPU-instanced showcase (ground + skybox + a 12x12 field of
            // instanced spheres drawn in ONE DrawIndexedInstanced, lit + shadowed), write a BMP, exit.
            instancedShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--commands") == 0 && i + 1 < argc) {
            commandsPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--dump-scene") == 0) {
            dumpScene = true;
        } else if (std::strcmp(argv[i], "--editor") == 0) {
            // Overlay the Dear ImGui editor (hierarchy/inspector/stats) on the viewport, rendered
            // through the engine RHI. Works in interactive mode and in the --shot capture.
            editor = true;
        }
    }
    try {
        hal::Window window({"Hazard Forge — Shadows", 1280, 720});
        auto device = rhi::CreateDevice(rhi::Backend::Vulkan, window);

        // --- GPU-instanced showcase (--instanced-shot): a self-contained capture path that does NOT
        // touch the default scene. Ground plane + procedural sky + a 12x12 = 144 field of spheres
        // drawn in ONE DrawIndexedInstanced call, each placed by its per-instance model matrix from a
        // second per-instance vertex stream (binding 1). Lit + shadowed (the field also casts shadows
        // via a single instanced depth-only draw). One frame -> BMP -> exit. Uses SEPARATE instanced
        // pipelines (lit_instanced.vert + shadow_instanced.vert); golden-locked pipelines untouched. --
        if (instancedShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // Instanced lit pipeline: lit_instanced.vert (model from instance stream) + shared lit.frag.
            // Declares the per-instance binding via instanceLayout; push constant carries float4 material.
            auto instVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_instanced.vert.hlsl.spv");
            auto litFsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto instVs = device->CreateShaderModule({std::span<const uint32_t>(instVsWords)});
            auto litFs  = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            rhi::GraphicsPipelineDesc instDesc;
            instDesc.vertex = instVs.get();
            instDesc.fragment = litFs.get();
            instDesc.vertexLayout = scene::MeshVertexLayout();
            instDesc.instanceLayout = scene::InstanceTransformLayout();  // binding 1, per-instance
            instDesc.colorFormat = device->Swapchain().ColorFormat();
            instDesc.depthTest = true;
            instDesc.usesFrameUniforms = true;
            instDesc.usesTexture = true;
            instDesc.pushConstantSize = sizeof(float) * 4;  // float4 material (metallic,roughness)
            auto instPipeline = device->CreateGraphicsPipeline(instDesc);

            // Static lit pipeline for the ground plane (untouched lit.frag, push-constant model+material).
            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get();
            litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = device->Swapchain().ColorFormat();
            litDesc.depthTest = true;
            litDesc.usesFrameUniforms = true;
            litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            // Instanced depth-only shadow pipeline (field casters) + static shadow pipeline (ground).
            auto instShWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow_instanced.vert.hlsl.spv");
            auto shadowFsW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto instShVs = device->CreateShaderModule({std::span<const uint32_t>(instShWords)});
            auto shadowFs = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});
            rhi::GraphicsPipelineDesc instShDesc;
            instShDesc.vertex = instShVs.get();
            instShDesc.fragment = shadowFs.get();
            instShDesc.vertexLayout = scene::MeshVertexLayout();
            instShDesc.instanceLayout = scene::InstanceTransformLayout();
            instShDesc.depthTest = true;
            instShDesc.depthOnly = true;
            instShDesc.usesFrameUniforms = true;
            instShDesc.pushConstantSize = 0;  // all transforms come from the instance stream
            auto instShadowPipeline = device->CreateGraphicsPipeline(instShDesc);

            auto staticShW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto staticShVs = device->CreateShaderModule({std::span<const uint32_t>(staticShW)});
            rhi::GraphicsPipelineDesc stShDesc;
            stShDesc.vertex = staticShVs.get();
            stShDesc.fragment = shadowFs.get();
            stShDesc.vertexLayout = scene::MeshVertexLayout();
            stShDesc.depthTest = true;
            stShDesc.depthOnly = true;
            stShDesc.usesFrameUniforms = true;
            stShDesc.pushConstantSize = sizeof(float) * 16;
            auto staticShadowPipeline = device->CreateGraphicsPipeline(stShDesc);

            // Sky + post.
            auto skyVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.vert.hlsl.spv");
            auto skyFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.frag.hlsl.spv");
            auto skyVsM = device->CreateShaderModule({std::span<const uint32_t>(skyVsW)});
            auto skyFsM = device->CreateShaderModule({std::span<const uint32_t>(skyFsW)});
            rhi::GraphicsPipelineDesc skyD;
            skyD.vertex = skyVsM.get(); skyD.fragment = skyFsM.get();
            skyD.colorFormat = device->Swapchain().ColorFormat();
            skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
            auto skyPipe = device->CreateGraphicsPipeline(skyD);

            auto postVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.vert.hlsl.spv");
            auto postFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.frag.hlsl.spv");
            auto postVsM = device->CreateShaderModule({std::span<const uint32_t>(postVsW)});
            auto postFsM = device->CreateShaderModule({std::span<const uint32_t>(postFsW)});
            rhi::GraphicsPipelineDesc postD;
            postD.vertex = postVsM.get(); postD.fragment = postFsM.get();
            postD.colorFormat = device->Swapchain().ColorFormat();
            postD.depthTest = false; postD.usesFrameUniforms = false;
            postD.usesTexture = true; postD.fullscreen = true;
            auto postPipe = device->CreateGraphicsPipeline(postD);

            auto rt = device->CreateRenderTarget(w, h);
            auto shadowMap = device->CreateShadowMap(2048);
            device->SetShadowMap(*shadowMap);

            // Ground + sphere meshes; checkerboard + flat normal.
            std::vector<uint8_t> checker = MakeCheckerboard();
            auto groundTex = device->CreateTexture(
                {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
            const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
            auto flatNormal = device->CreateTexture(
                {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
            scene::Mesh plane = scene::Mesh::Plane(*device);
            scene::Mesh sphere = scene::Mesh::Sphere(*device);

            // Deterministic 12x12 = 144 instance grid (golden-stable; no RNG).
            const uint32_t kGrid = 12;
            std::vector<scene::InstanceData> instances =
                scene::BuildInstanceGrid(kGrid, /*spacing=*/1.3f, /*scale=*/0.45f);
            const uint32_t kInstanceCount = (uint32_t)instances.size();
            rhi::BufferDesc instBufDesc;
            instBufDesc.size = (uint64_t)instances.size() * sizeof(scene::InstanceData);
            instBufDesc.initialData = instances.data();
            instBufDesc.usage = rhi::BufferUsage::Vertex;
            auto instanceBuffer = device->CreateBuffer(instBufDesc);

            Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});

            // Camera + light + sky frame data (fixed, deterministic). Elevated camera to read the grid.
            const Vec3 eye{9.5f, 8.5f, 11.0f};
            const Vec3 center{0.0f, 0.6f, 0.0f};
            FrameData fd{};
            {
                Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
                Mat4 proj = Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f);
                Mat4 vp = proj * view;
                for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
                fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
                fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
                fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
                fd.ptCount[0] = 0.0f;
                Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
                Vec3 sc{0.0f, 0.6f, 0.0f};
                Vec3 lightEye = sc - lightDir * 18.0f;
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
                Mat4 lightOrtho = Mat4::Ortho(-11.0f, 11.0f, -11.0f, 11.0f, 1.0f, 40.0f);
                Mat4 lightVP = lightOrtho * lightView;
                for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
                Vec3 fwd = math::normalize(center - eye);
                Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
                Vec3 up = math::cross(right, fwd);
                fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
                fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
                fd.camUp[0]=up.x; fd.camUp[1]=up.y; fd.camUp[2]=up.z;
                fd.skyParams[0] = std::tan(0.5f * 1.04719755f);
                fd.skyParams[1] = aspect;
            }

            render::RenderGraph graph;
            render::RgResource rgShadow = graph.ImportTarget(
                "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
            render::RgResource rgScene = graph.ImportTarget(
                "sceneColor", render::RgResourceKind::SceneColor, *rt);
            render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

            graph.AddPass("shadow", {}, {rgShadow},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    // Ground caster (static).
                    cmd.BindPipeline(*staticShadowPipeline);
                    cmd.PushConstants(groundModel.m, sizeof(float) * 16);
                    cmd.BindVertexBuffer(plane.vertices());
                    cmd.BindIndexBuffer(plane.indices());
                    cmd.DrawIndexed(plane.indexCount());
                    // Instanced field casters: ONE instanced draw.
                    cmd.BindPipeline(*instShadowPipeline);
                    cmd.BindVertexBuffer(sphere.vertices());
                    cmd.BindInstanceBuffer(*instanceBuffer);
                    cmd.BindIndexBuffer(sphere.indices());
                    cmd.DrawIndexedInstanced(sphere.indexCount(), kInstanceCount);
                    cmd.EndRenderPass();
                });

            graph.AddPass("scene", {rgShadow}, {rgScene},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
                    cmd.BindPipeline(*skyPipe);
                    cmd.Draw(3);
                    // Ground plane (lit, dielectric).
                    cmd.BindPipeline(*litPipeline);
                    {
                        float pc[20];
                        for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                        pc[16] = 0.0f; pc[17] = 0.85f; pc[18] = 0.0f; pc[19] = 0.0f;
                        cmd.PushConstants(pc, sizeof(pc));
                        cmd.BindMaterial(*groundTex, *flatNormal);
                        cmd.BindVertexBuffer(plane.vertices());
                        cmd.BindIndexBuffer(plane.indices());
                        cmd.DrawIndexed(plane.indexCount());
                    }
                    // Instanced sphere field: ONE DrawIndexedInstanced.
                    cmd.BindPipeline(*instPipeline);
                    {
                        float material[4] = {0.1f, 0.5f, 0.0f, 0.0f};  // metallic, roughness
                        cmd.PushConstants(material, sizeof(material));
                        cmd.BindMaterial(*groundTex, *flatNormal);
                        cmd.BindVertexBuffer(sphere.vertices());
                        cmd.BindInstanceBuffer(*instanceBuffer);
                        cmd.BindIndexBuffer(sphere.indices());
                        cmd.DrawIndexedInstanced(sphere.indexCount(), kInstanceCount);
                    }
                    cmd.EndRenderPass();
                });

            graph.AddPass("post", {rgScene}, {rgSwap},
                [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*postPipe);
                    cmd.BindTexture(*rt);
                    cmd.Draw(3);
                    cmd.EndRenderPass();
                });

            device->CaptureNextFrame();
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(instancedShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — %u instances\n",
                                    instancedShotPath, cw, ch2, kInstanceCount);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", instancedShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- HDR environment IBL showcase (--ibl-shot, Slice R): a self-contained capture path that
        // does NOT touch the default scene. HDR equirect skybox (sky_hdr) + ground plane + the
        // DamagedHelmet shaded by lit_pbr_ibl so the metal reflects the REAL captured sky/sun/terrain
        // (mip-LOD prefiltered), lit + shadowed. One frame -> BMP -> exit. Uses SEPARATE sky_hdr +
        // lit_pbr_ibl pipelines + an environment set bound via BindEnvironment; the golden-locked
        // lit/sky/lit_pbr pipelines are untouched. ---------------------------------------------------
        if (iblShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // Load the HDR equirect environment (N-mip RGBA16F, CPU box-prefiltered).
            hf::asset::EnvironmentMap env = hf::asset::LoadHdrEnvironment(*device, HF_ENV_PATH);
            const float envMaxLod = (float)(env.mipLevels - 1);

            // Lit-PBR-IBL pipeline: shared lit.vert + lit_pbr_ibl.frag; wider PBR material set (set 1)
            // PLUS the dedicated environment set (set 3) declared via usesEnvironment.
            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto iblFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_pbr_ibl.frag.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto iblFs = device->CreateShaderModule({std::span<const uint32_t>(iblFsWords)});
            rhi::GraphicsPipelineDesc iblDesc;
            iblDesc.vertex = litVs.get();
            iblDesc.fragment = iblFs.get();
            iblDesc.vertexLayout = scene::MeshVertexLayout();
            iblDesc.colorFormat = device->Swapchain().ColorFormat();
            iblDesc.depthTest = true;
            iblDesc.usesFrameUniforms = true;
            iblDesc.usesTexture = true;
            iblDesc.pbrMaterial = true;
            iblDesc.usesEnvironment = true;   // adds the set-3 env binding
            iblDesc.pushConstantSize = sizeof(float) * 20;
            auto iblPipeline = device->CreateGraphicsPipeline(iblDesc);

            // Static lit pipeline for the ground plane (untouched lit.frag, 2-texture material set).
            auto litFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto litFs = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get();
            litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = device->Swapchain().ColorFormat();
            litDesc.depthTest = true;
            litDesc.usesFrameUniforms = true;
            litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            // Static depth-only shadow pipeline (ground + helmet casters).
            auto shadowVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto shadowFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto shadowVs = device->CreateShaderModule({std::span<const uint32_t>(shadowVsW)});
            auto shadowFs = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});
            rhi::GraphicsPipelineDesc shDesc;
            shDesc.vertex = shadowVs.get();
            shDesc.fragment = shadowFs.get();
            shDesc.vertexLayout = scene::MeshVertexLayout();
            shDesc.depthTest = true;
            shDesc.depthOnly = true;
            shDesc.usesFrameUniforms = true;
            shDesc.pushConstantSize = sizeof(float) * 16;
            auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

            // HDR sky pipeline (sky_hdr.frag) + post. The sky pipeline also declares usesEnvironment.
            auto skyVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.vert.hlsl.spv");
            auto skyFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky_hdr.frag.hlsl.spv");
            auto skyVsM = device->CreateShaderModule({std::span<const uint32_t>(skyVsW)});
            auto skyFsM = device->CreateShaderModule({std::span<const uint32_t>(skyFsW)});
            rhi::GraphicsPipelineDesc skyD;
            skyD.vertex = skyVsM.get(); skyD.fragment = skyFsM.get();
            skyD.colorFormat = device->Swapchain().ColorFormat();
            skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
            skyD.usesEnvironment = true;
            auto skyPipe = device->CreateGraphicsPipeline(skyD);

            auto postVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.vert.hlsl.spv");
            auto postFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.frag.hlsl.spv");
            auto postVsM = device->CreateShaderModule({std::span<const uint32_t>(postVsW)});
            auto postFsM = device->CreateShaderModule({std::span<const uint32_t>(postFsW)});
            rhi::GraphicsPipelineDesc postD;
            postD.vertex = postVsM.get(); postD.fragment = postFsM.get();
            postD.colorFormat = device->Swapchain().ColorFormat();
            postD.depthTest = false; postD.usesFrameUniforms = false;
            postD.usesTexture = true; postD.fullscreen = true;
            auto postPipe = device->CreateGraphicsPipeline(postD);

            auto rt = device->CreateRenderTarget(w, h);
            auto shadowMap = device->CreateShadowMap(2048);
            device->SetShadowMap(*shadowMap);

            std::vector<uint8_t> checker = MakeCheckerboard();
            auto groundTex = device->CreateTexture(
                {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
            const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
            auto flatNormal = device->CreateTexture(
                {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
            scene::Mesh plane = scene::Mesh::Plane(*device);

            hf::asset::PbrModel helmet = hf::asset::LoadPbrGltfModel(*device, HF_HELMET_MODEL_PATH);

            const float scaleS = 1.6f;
            Mat4 helmetModel = Mat4::Translate({0.0f, scaleS * 1.0f, 0.0f})
                             * Mat4::RotateX(1.5707963f)
                             * Mat4::Scale({scaleS, scaleS, scaleS});
            Mat4 groundModel = Mat4::Scale({8.0f, 1.0f, 8.0f});

            const Vec3 eye{3.0f, 2.4f, 4.0f};
            const Vec3 center{0.0f, 1.2f, 0.0f};
            FrameData fd{};
            {
                Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
                Mat4 proj = Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f);
                Mat4 vp = proj * view;
                for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
                fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
                fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
                fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
                fd.ptCount[0] = 0.0f;
                Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
                Vec3 sc{0.0f, 1.2f, 0.0f};
                Vec3 lightEye = sc - lightDir * 12.0f;
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
                Mat4 lightOrtho = Mat4::Ortho(-5.0f, 5.0f, -5.0f, 5.0f, 1.0f, 25.0f);
                Mat4 lightVP = lightOrtho * lightView;
                for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
                Vec3 fwd = math::normalize(center - eye);
                Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
                Vec3 up = math::cross(right, fwd);
                fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
                fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
                fd.camUp[0]=up.x; fd.camUp[1]=up.y; fd.camUp[2]=up.z;
                fd.skyParams[0] = std::tan(0.5f * 1.04719755f);
                fd.skyParams[1] = aspect;
                fd.skyParams[2] = envMaxLod;   // env maxLod for the IBL fragment shader (Slice R)
            }

            render::RenderGraph graph;
            render::RgResource rgShadow = graph.ImportTarget(
                "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
            render::RgResource rgScene = graph.ImportTarget(
                "sceneColor", render::RgResourceKind::SceneColor, *rt);
            render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

            graph.AddPass("shadow", {}, {rgShadow},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*shadowPipeline);
                    cmd.PushConstants(groundModel.m, sizeof(float) * 16);
                    cmd.BindVertexBuffer(plane.vertices());
                    cmd.BindIndexBuffer(plane.indices());
                    cmd.DrawIndexed(plane.indexCount());
                    cmd.PushConstants(helmetModel.m, sizeof(float) * 16);
                    cmd.BindVertexBuffer(helmet.mesh.vertices());
                    cmd.BindIndexBuffer(helmet.mesh.indices());
                    cmd.DrawIndexed(helmet.mesh.indexCount());
                    cmd.EndRenderPass();
                });

            graph.AddPass("scene", {rgShadow}, {rgScene},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
                    // HDR equirect skybox.
                    cmd.BindPipeline(*skyPipe);
                    cmd.BindEnvironment(*env.equirect);
                    cmd.Draw(3);
                    // Ground plane (lit, dielectric).
                    cmd.BindPipeline(*litPipeline);
                    {
                        float pc[20];
                        for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                        pc[16] = 0.0f; pc[17] = 0.85f; pc[18] = 0.0f; pc[19] = 0.0f;
                        cmd.PushConstants(pc, sizeof(pc));
                        cmd.BindMaterial(*groundTex, *flatNormal);
                        cmd.BindVertexBuffer(plane.vertices());
                        cmd.BindIndexBuffer(plane.indices());
                        cmd.DrawIndexed(plane.indexCount());
                    }
                    // Helmet (full-PBR with HDR IBL): reflects the real sky.
                    cmd.BindPipeline(*iblPipeline);
                    {
                        float pc[20];
                        for (int k = 0; k < 16; ++k) pc[k] = helmetModel.m[k];
                        pc[16] = helmet.metallicFactor; pc[17] = helmet.roughnessFactor;
                        pc[18] = 0.0f; pc[19] = 0.0f;
                        cmd.PushConstants(pc, sizeof(pc));
                        cmd.BindMaterialPBR(*helmet.baseColor, *helmet.metalRough, *helmet.normalMap,
                                            *helmet.emissive, *helmet.occlusion);
                        cmd.BindEnvironment(*env.equirect);
                        cmd.BindVertexBuffer(helmet.mesh.vertices());
                        cmd.BindIndexBuffer(helmet.mesh.indices());
                        cmd.DrawIndexed(helmet.mesh.indexCount());
                    }
                    cmd.EndRenderPass();
                });

            graph.AddPass("post", {rgScene}, {rgSwap},
                [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*postPipe);
                    cmd.BindTexture(*rt);
                    cmd.Draw(3);
                    cmd.EndRenderPass();
                });

            device->CaptureNextFrame();
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(iblShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — env %dx%d, %d mips\n",
                                    iblShotPath, cw, ch2, env.width, env.height, env.mipLevels);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", iblShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- Full-PBR showcase (--pbr-shot): a self-contained capture path that does NOT touch the
        // default scene. Ground plane + procedural sky + the DamagedHelmet rendered with the full
        // glTF metallic-roughness material set (base/metalRough/normal/emissive/occlusion), lit +
        // shadowed. One frame -> BMP -> exit. Uses a SEPARATE lit-PBR pipeline (lit_pbr.frag) +
        // BindMaterialPBR; the golden-locked lit pipeline is untouched. -----------------------------
        if (pbrShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // Lit-PBR pipeline: shared lit.vert + the full-PBR fragment shader; declares the WIDER
            // material set (pbrMaterial=true) so BindMaterialPBR binds all five textures.
            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto pbrFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_pbr.frag.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto pbrFs = device->CreateShaderModule({std::span<const uint32_t>(pbrFsWords)});
            rhi::GraphicsPipelineDesc pbrDesc;
            pbrDesc.vertex = litVs.get();
            pbrDesc.fragment = pbrFs.get();
            pbrDesc.vertexLayout = scene::MeshVertexLayout();
            pbrDesc.colorFormat = device->Swapchain().ColorFormat();
            pbrDesc.depthTest = true;
            pbrDesc.usesFrameUniforms = true;
            pbrDesc.usesTexture = true;
            pbrDesc.pbrMaterial = true;               // wider 5-texture material set
            pbrDesc.pushConstantSize = sizeof(float) * 20;  // mat4 model + float4 material
            auto pbrPipeline = device->CreateGraphicsPipeline(pbrDesc);

            // Static lit pipeline for the ground plane (2-texture material set, untouched lit.frag).
            auto litFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto litFs = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get();
            litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = device->Swapchain().ColorFormat();
            litDesc.depthTest = true;
            litDesc.usesFrameUniforms = true;
            litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            // Static depth-only shadow pipeline (ground + helmet casters).
            auto shadowVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto shadowFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto shadowVs = device->CreateShaderModule({std::span<const uint32_t>(shadowVsW)});
            auto shadowFs = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});
            rhi::GraphicsPipelineDesc shDesc;
            shDesc.vertex = shadowVs.get();
            shDesc.fragment = shadowFs.get();
            shDesc.vertexLayout = scene::MeshVertexLayout();
            shDesc.depthTest = true;
            shDesc.depthOnly = true;
            shDesc.usesFrameUniforms = true;
            shDesc.pushConstantSize = sizeof(float) * 16;
            auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

            // Sky + post.
            auto skyVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.vert.hlsl.spv");
            auto skyFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.frag.hlsl.spv");
            auto skyVsM = device->CreateShaderModule({std::span<const uint32_t>(skyVsW)});
            auto skyFsM = device->CreateShaderModule({std::span<const uint32_t>(skyFsW)});
            rhi::GraphicsPipelineDesc skyD;
            skyD.vertex = skyVsM.get(); skyD.fragment = skyFsM.get();
            skyD.colorFormat = device->Swapchain().ColorFormat();
            skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
            auto skyPipe = device->CreateGraphicsPipeline(skyD);

            auto postVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.vert.hlsl.spv");
            auto postFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.frag.hlsl.spv");
            auto postVsM = device->CreateShaderModule({std::span<const uint32_t>(postVsW)});
            auto postFsM = device->CreateShaderModule({std::span<const uint32_t>(postFsW)});
            rhi::GraphicsPipelineDesc postD;
            postD.vertex = postVsM.get(); postD.fragment = postFsM.get();
            postD.colorFormat = device->Swapchain().ColorFormat();
            postD.depthTest = false; postD.usesFrameUniforms = false;
            postD.usesTexture = true; postD.fullscreen = true;
            auto postPipe = device->CreateGraphicsPipeline(postD);

            auto rt = device->CreateRenderTarget(w, h);
            auto shadowMap = device->CreateShadowMap(2048);
            device->SetShadowMap(*shadowMap);

            // Ground plane: a flat checkerboard dielectric (lit pipeline; flat normal).
            std::vector<uint8_t> checker = MakeCheckerboard();
            auto groundTex = device->CreateTexture(
                {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
            const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
            auto flatNormal = device->CreateTexture(
                {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
            scene::Mesh plane = scene::Mesh::Plane(*device);

            // Load the helmet + its full PBR material (5 textures + factors).
            hf::asset::PbrModel helmet = hf::asset::LoadPbrGltfModel(*device, HF_HELMET_MODEL_PATH);

            // Place the helmet: it is recentred to origin by the loader and authored Z-up, so rotate
            // -90deg about X to stand it up, scale to a sensible size, then lift so it sits on y=0.
            // The mesh's recentred half-height is ~ (bbox/2)*scale; the helmet is ~roughly a unit
            // sphere of radius ~1 model-space, so after scale the centre sits at y = scaleS*~1.
            const float scaleS = 1.6f;
            Mat4 helmetModel = Mat4::Translate({0.0f, scaleS * 1.0f, 0.0f})
                             * Mat4::RotateX(1.5707963f)
                             * Mat4::Scale({scaleS, scaleS, scaleS});

            // Camera + light + sky frame data (fixed, deterministic).
            const Vec3 eye{3.0f, 2.4f, 4.0f};
            const Vec3 center{0.0f, 1.2f, 0.0f};
            FrameData fd{};
            {
                Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
                Mat4 proj = Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f);
                Mat4 vp = proj * view;
                for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
                fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
                fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
                fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
                fd.ptCount[0] = 0.0f;  // no point lights in the showcase
                Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
                Vec3 sc{0.0f, 1.2f, 0.0f};
                Vec3 lightEye = sc - lightDir * 12.0f;
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
                Mat4 lightOrtho = Mat4::Ortho(-5.0f, 5.0f, -5.0f, 5.0f, 1.0f, 25.0f);
                Mat4 lightVP = lightOrtho * lightView;
                for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
                Vec3 fwd = math::normalize(center - eye);
                Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
                Vec3 up = math::cross(right, fwd);
                fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
                fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
                fd.camUp[0]=up.x; fd.camUp[1]=up.y; fd.camUp[2]=up.z;
                fd.skyParams[0] = std::tan(0.5f * 1.04719755f);
                fd.skyParams[1] = aspect;
            }

            Mat4 groundModel = Mat4::Scale({8.0f, 1.0f, 8.0f});

            render::RenderGraph graph;
            render::RgResource rgShadow = graph.ImportTarget(
                "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
            render::RgResource rgScene = graph.ImportTarget(
                "sceneColor", render::RgResourceKind::SceneColor, *rt);
            render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

            graph.AddPass("shadow", {}, {rgShadow},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*shadowPipeline);
                    cmd.PushConstants(groundModel.m, sizeof(float) * 16);
                    cmd.BindVertexBuffer(plane.vertices());
                    cmd.BindIndexBuffer(plane.indices());
                    cmd.DrawIndexed(plane.indexCount());
                    cmd.PushConstants(helmetModel.m, sizeof(float) * 16);
                    cmd.BindVertexBuffer(helmet.mesh.vertices());
                    cmd.BindIndexBuffer(helmet.mesh.indices());
                    cmd.DrawIndexed(helmet.mesh.indexCount());
                    cmd.EndRenderPass();
                });

            graph.AddPass("scene", {rgShadow}, {rgScene},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
                    cmd.BindPipeline(*skyPipe);
                    cmd.Draw(3);
                    // Ground plane (lit, dielectric).
                    cmd.BindPipeline(*litPipeline);
                    {
                        float pc[20];
                        for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                        pc[16] = 0.0f; pc[17] = 0.85f; pc[18] = 0.0f; pc[19] = 0.0f;
                        cmd.PushConstants(pc, sizeof(pc));
                        cmd.BindMaterial(*groundTex, *flatNormal);
                        cmd.BindVertexBuffer(plane.vertices());
                        cmd.BindIndexBuffer(plane.indices());
                        cmd.DrawIndexed(plane.indexCount());
                    }
                    // Helmet (full-PBR).
                    cmd.BindPipeline(*pbrPipeline);
                    {
                        float pc[20];
                        for (int k = 0; k < 16; ++k) pc[k] = helmetModel.m[k];
                        pc[16] = helmet.metallicFactor; pc[17] = helmet.roughnessFactor;
                        pc[18] = 0.0f; pc[19] = 0.0f;
                        cmd.PushConstants(pc, sizeof(pc));
                        cmd.BindMaterialPBR(*helmet.baseColor, *helmet.metalRough, *helmet.normalMap,
                                            *helmet.emissive, *helmet.occlusion);
                        cmd.BindVertexBuffer(helmet.mesh.vertices());
                        cmd.BindIndexBuffer(helmet.mesh.indices());
                        cmd.DrawIndexed(helmet.mesh.indexCount());
                    }
                    cmd.EndRenderPass();
                });

            graph.AddPass("post", {rgScene}, {rgSwap},
                [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*postPipe);
                    cmd.BindTexture(*rt);
                    cmd.Draw(3);
                    cmd.EndRenderPass();
                });

            device->CaptureNextFrame();
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(pbrShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u)\n", pbrShotPath, cw, ch2);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", pbrShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- Skeletal-animation showcase (--skinning-shot): a self-contained capture path that does
        // NOT touch the default scene. Ground plane + procedural sky + the GPU-skinned Fox sampled at
        // animation "Survey", time 0.5s, lit + shadowed. One frame -> BMP -> exit. ----------------
        if (skinningShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // Skinned lit pipeline (set 2 joint palette; reuses lit.frag).
            auto skVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_skinned.vert.hlsl.spv");
            auto litFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto skVs = device->CreateShaderModule({std::span<const uint32_t>(skVsWords)});
            auto litFs = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            rhi::GraphicsPipelineDesc skDesc;
            skDesc.vertex = skVs.get();
            skDesc.fragment = litFs.get();
            skDesc.vertexLayout = scene::SkinnedMeshVertexLayout();
            skDesc.colorFormat = device->Swapchain().ColorFormat();
            skDesc.depthTest = true;
            skDesc.usesFrameUniforms = true;
            skDesc.usesTexture = true;
            skDesc.usesJointPalette = true;
            skDesc.pushConstantSize = sizeof(float) * 20;  // mat4 model + float4 material
            auto skinnedPipeline = device->CreateGraphicsPipeline(skDesc);

            // Static lit pipeline (for the ground plane).
            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get();
            litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = device->Swapchain().ColorFormat();
            litDesc.depthTest = true;
            litDesc.usesFrameUniforms = true;
            litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            // Skinned + static depth-only shadow pipelines.
            auto skShadowWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow_skinned.vert.hlsl.spv");
            auto shadowFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto skShadowVs = device->CreateShaderModule({std::span<const uint32_t>(skShadowWords)});
            auto shadowFs = device->CreateShaderModule({std::span<const uint32_t>(shadowFsWords)});
            rhi::GraphicsPipelineDesc skShDesc;
            skShDesc.vertex = skShadowVs.get();
            skShDesc.fragment = shadowFs.get();
            skShDesc.vertexLayout = scene::SkinnedMeshVertexLayout();
            skShDesc.depthTest = true;
            skShDesc.depthOnly = true;
            skShDesc.usesFrameUniforms = true;
            skShDesc.usesJointPalette = true;
            skShDesc.pushConstantSize = sizeof(float) * 16;
            auto skinnedShadowPipeline = device->CreateGraphicsPipeline(skShDesc);

            auto staticShadowWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto staticShadowVs = device->CreateShaderModule({std::span<const uint32_t>(staticShadowWords)});
            rhi::GraphicsPipelineDesc stShDesc;
            stShDesc.vertex = staticShadowVs.get();
            stShDesc.fragment = shadowFs.get();
            stShDesc.vertexLayout = scene::MeshVertexLayout();
            stShDesc.depthTest = true;
            stShDesc.depthOnly = true;
            stShDesc.usesFrameUniforms = true;
            stShDesc.pushConstantSize = sizeof(float) * 16;
            auto staticShadowPipeline = device->CreateGraphicsPipeline(stShDesc);

            // Sky + post.
            auto skyVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.vert.hlsl.spv");
            auto skyFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.frag.hlsl.spv");
            auto skyVsM = device->CreateShaderModule({std::span<const uint32_t>(skyVsW)});
            auto skyFsM = device->CreateShaderModule({std::span<const uint32_t>(skyFsW)});
            rhi::GraphicsPipelineDesc skyD;
            skyD.vertex = skyVsM.get(); skyD.fragment = skyFsM.get();
            skyD.colorFormat = device->Swapchain().ColorFormat();
            skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
            auto skyPipe = device->CreateGraphicsPipeline(skyD);

            auto postVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.vert.hlsl.spv");
            auto postFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.frag.hlsl.spv");
            auto postVsM = device->CreateShaderModule({std::span<const uint32_t>(postVsW)});
            auto postFsM = device->CreateShaderModule({std::span<const uint32_t>(postFsW)});
            rhi::GraphicsPipelineDesc postD;
            postD.vertex = postVsM.get(); postD.fragment = postFsM.get();
            postD.colorFormat = device->Swapchain().ColorFormat();
            postD.depthTest = false; postD.usesFrameUniforms = false;
            postD.usesTexture = true; postD.fullscreen = true;
            auto postPipe = device->CreateGraphicsPipeline(postD);

            auto rt = device->CreateRenderTarget(w, h);
            auto shadowMap = device->CreateShadowMap(2048);
            device->SetShadowMap(*shadowMap);

            // Ground plane: a flat checkerboard-textured dielectric. Normal maps default to flat.
            std::vector<uint8_t> checker = MakeCheckerboard();
            auto groundTex = device->CreateTexture(
                {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
            const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
            auto flatNormal = device->CreateTexture(
                {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
            scene::Mesh plane = scene::Mesh::Plane(*device);

            // Load the skinned Fox + sample the Survey animation at t=0.5s -> joint palette.
            hf::asset::SkinnedModel fox = hf::asset::LoadSkinnedGltfModel(*device, HF_FOX_MODEL_PATH);
            const anim::Animation* survey = fox.FindAnimation("Survey");
            if (!survey && !fox.animations.empty()) survey = &fox.animations.front();
            std::vector<Mat4> palette;
            if (survey) palette = anim::SampleAnimation(fox.skeleton, *survey, 0.5f);
            else palette.assign(fox.skeleton.joints.size(), Mat4::Identity());
            // Pad to 64 identity matrices for the fixed-size JointPalette UBO.
            std::vector<float> paletteData(64 * 16);
            for (int j = 0; j < 64; ++j) {
                Mat4 mm = (j < (int)palette.size()) ? palette[j] : Mat4::Identity();
                for (int k = 0; k < 16; ++k) paletteData[j * 16 + k] = mm.m[k];
            }

            // Place the fox: uniform scale so it's a sensible size, ground-aligned via its bbox, and
            // translated to the origin. The Fox is authored ~70 units tall in +Y, facing +Z.
            float foxH = fox.bbMax[1] - fox.bbMin[1];
            float scaleS = (foxH > 1e-4f) ? (2.5f / foxH) : 0.05f;  // ~2.5 world units tall
            float cx = 0.5f * (fox.bbMin[0] + fox.bbMax[0]);
            float cz = 0.5f * (fox.bbMin[2] + fox.bbMax[2]);
            // model = Translate(-cx*s, -bbMin.y*s, -cz*s) * Scale(s): centre x/z, sit feet on y=0.
            Mat4 foxModel = Mat4::Translate({-cx * scaleS, -fox.bbMin[1] * scaleS, -cz * scaleS})
                          * Mat4::Scale({scaleS, scaleS, scaleS});

            // Camera + light + sky frame data (fixed, deterministic).
            const Vec3 eye{3.5f, 2.6f, 4.5f};
            const Vec3 center{0.0f, 1.0f, 0.0f};
            FrameData fd{};
            {
                Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
                Mat4 proj = Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f);
                Mat4 vp = proj * view;
                for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
                fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
                fd.lightColor[0] = 0.98f; fd.lightColor[1] = 0.95f; fd.lightColor[2] = 0.88f; fd.lightColor[3] = 1.0f;
                fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
                fd.ptCount[0] = 0.0f;  // no point lights in the showcase
                Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
                Vec3 sc{0.0f, 1.0f, 0.0f};
                Vec3 lightEye = sc - lightDir * 12.0f;
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
                Mat4 lightOrtho = Mat4::Ortho(-6.0f, 6.0f, -6.0f, 6.0f, 1.0f, 25.0f);
                Mat4 lightVP = lightOrtho * lightView;
                for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
                Vec3 fwd = math::normalize(center - eye);
                Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
                Vec3 up = math::cross(right, fwd);
                fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
                fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
                fd.camUp[0]=up.x; fd.camUp[1]=up.y; fd.camUp[2]=up.z;
                fd.skyParams[0] = std::tan(0.5f * 1.04719755f);
                fd.skyParams[1] = aspect;
            }

            // Ground plane transform: wide flat platform under the fox.
            Mat4 groundModel = Mat4::Scale({8.0f, 1.0f, 8.0f});

            render::RenderGraph graph;
            render::RgResource rgShadow = graph.ImportTarget(
                "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
            render::RgResource rgScene = graph.ImportTarget(
                "sceneColor", render::RgResourceKind::SceneColor, *rt);
            render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

            graph.AddPass("shadow", {}, {rgShadow},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    dev.SetJointPalette(paletteData.data(), paletteData.size() * sizeof(float));
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    // Ground (static) caster.
                    cmd.BindPipeline(*staticShadowPipeline);
                    cmd.PushConstants(groundModel.m, sizeof(float) * 16);
                    cmd.BindVertexBuffer(plane.vertices());
                    cmd.BindIndexBuffer(plane.indices());
                    cmd.DrawIndexed(plane.indexCount());
                    // Fox (skinned) caster.
                    cmd.BindPipeline(*skinnedShadowPipeline);
                    cmd.PushConstants(foxModel.m, sizeof(float) * 16);
                    cmd.BindVertexBuffer(fox.mesh.vertices());
                    cmd.BindIndexBuffer(fox.mesh.indices());
                    cmd.DrawIndexed(fox.mesh.indexCount());
                    cmd.EndRenderPass();
                });

            graph.AddPass("scene", {rgShadow}, {rgScene},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    dev.SetJointPalette(paletteData.data(), paletteData.size() * sizeof(float));
                    cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
                    cmd.BindPipeline(*skyPipe);
                    cmd.Draw(3);
                    // Ground plane (static lit).
                    cmd.BindPipeline(*litPipeline);
                    {
                        float pc[20];
                        for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                        pc[16] = 0.0f; pc[17] = 0.85f; pc[18] = 0.0f; pc[19] = 0.0f;
                        cmd.PushConstants(pc, sizeof(pc));
                        cmd.BindMaterial(*groundTex, *flatNormal);
                        cmd.BindVertexBuffer(plane.vertices());
                        cmd.BindIndexBuffer(plane.indices());
                        cmd.DrawIndexed(plane.indexCount());
                    }
                    // Fox (skinned lit).
                    cmd.BindPipeline(*skinnedPipeline);
                    {
                        float pc[20];
                        for (int k = 0; k < 16; ++k) pc[k] = foxModel.m[k];
                        pc[16] = fox.metallic; pc[17] = fox.roughness; pc[18] = 0.0f; pc[19] = 0.0f;
                        cmd.PushConstants(pc, sizeof(pc));
                        cmd.BindMaterial(*fox.baseColor, *flatNormal);
                        cmd.BindVertexBuffer(fox.mesh.vertices());
                        cmd.BindIndexBuffer(fox.mesh.indices());
                        cmd.DrawIndexed(fox.mesh.indexCount());
                    }
                    cmd.EndRenderPass();
                });

            graph.AddPass("post", {rgScene}, {rgSwap},
                [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*postPipe);
                    cmd.BindTexture(*rt);
                    cmd.Draw(3);
                    cmd.EndRenderPass();
                });

            device->CaptureNextFrame();
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(skinningShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u)\n", skinningShotPath, cw, ch2);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", skinningShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

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

        // --- GPU particle system (compute shader animates a storage buffer; drawn as points). ---
        // Particle layout: { float4 posLife; float4 velSeed; } = 32 bytes, matches
        // particles.comp.hlsl / particle.vert.hlsl.
        constexpr uint32_t kParticleCount = 50000;
        constexpr uint32_t kParticleStride = 32;  // 2x float4
        struct GpuParticle { float posLife[4]; float velSeed[4]; };
        static_assert(sizeof(GpuParticle) == kParticleStride, "particle stride");

        // Seed the buffer deterministically so headless capture is golden-stable: spread initial
        // life across [0,4) so particles emit continuously rather than all at once.
        std::vector<GpuParticle> initParticles(kParticleCount);
        for (uint32_t i = 0; i < kParticleCount; ++i) {
            float s = (float)i / (float)kParticleCount;          // stable per-index seed
            float a = s * 6.2831853f;
            float r = 0.25f * (0.5f + 0.5f * std::sin(s * 12.9898f));
            initParticles[i].posLife[0] = std::cos(a) * r;
            initParticles[i].posLife[1] = 0.05f + 2.0f * s;       // staggered heights
            initParticles[i].posLife[2] = std::sin(a) * r;
            initParticles[i].posLife[3] = 0.5f + 3.5f * s;        // staggered remaining life
            float outR = 0.5f + 2.8f * std::fabs(std::sin(s * 7.13f));
            float aVel = s * 9.41f;
            initParticles[i].velSeed[0] = std::cos(aVel) * outR;
            initParticles[i].velSeed[1] = 3.5f + 3.5f * s;
            initParticles[i].velSeed[2] = std::sin(aVel) * outR;
            initParticles[i].velSeed[3] = s;                      // seed
        }
        rhi::BufferDesc particleBufDesc;
        particleBufDesc.size = (uint64_t)kParticleCount * kParticleStride;
        particleBufDesc.initialData = initParticles.data();
        particleBufDesc.usage = rhi::BufferUsage::Storage;
        auto particleBuffer = device->CreateBuffer(particleBufDesc);

        // Compute pipeline: one storage buffer + a {dt,time,count,_pad} push constant.
        auto partCsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/particles.comp.hlsl.spv");
        auto partCs = device->CreateShaderModule({std::span<const uint32_t>(partCsWords)});
        rhi::ComputePipelineDesc cdesc;
        cdesc.compute = partCs.get();
        cdesc.storageBufferCount = 1;
        cdesc.pushConstantSize = sizeof(float) * 2 + sizeof(uint32_t) * 2;  // dt,time,count,_pad
        auto particleCompute = device->CreateComputePipeline(cdesc);

        // Particle graphics pipeline: point list, additive blend, reads the storage buffer as a
        // vertex stream (pos at offset 0, vel at offset 16). Uses the per-frame UBO for viewProj.
        auto partVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/particle.vert.hlsl.spv");
        auto partFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/particle.frag.hlsl.spv");
        auto partVs = device->CreateShaderModule({std::span<const uint32_t>(partVsWords)});
        auto partFs = device->CreateShaderModule({std::span<const uint32_t>(partFsWords)});
        rhi::GraphicsPipelineDesc partDesc;
        partDesc.vertex = partVs.get();
        partDesc.fragment = partFs.get();
        partDesc.vertexLayout.stride = kParticleStride;
        partDesc.vertexLayout.attributes = {
            {0, rhi::Format::RGB32_Float, 0},   // position (posLife.xyz)
            {1, rhi::Format::RGB32_Float, 16},  // velocity (velSeed.xyz)
        };
        partDesc.colorFormat = device->Swapchain().ColorFormat();
        partDesc.depthTest = false;          // particles are emissive points over the scene
        partDesc.usesFrameUniforms = true;   // viewProj from the frame UBO (set 0 b0)
        partDesc.usesTexture = false;
        partDesc.pointList = true;
        partDesc.additiveBlend = true;
        auto particlePipeline = device->CreateGraphicsPipeline(partDesc);

        // Push-constant block for the compute dispatch.
        struct ParticleParams { float dt; float time; uint32_t count; uint32_t pad; };
        const uint32_t kParticleGroups = (kParticleCount + 63) / 64;  // [numthreads(64,1,1)]

        // Dispatch the compute sim into `cmd`, then barrier compute->vertex. Records BEFORE
        // BeginRenderPass (compute must run outside a render pass).
        auto simulateParticles = [&](rhi::ICommandBuffer* cmd, float dt, float time) {
            ParticleParams pp{dt, time, kParticleCount, 0};
            cmd->BindComputePipeline(*particleCompute);
            cmd->BindStorageBuffer(*particleBuffer, 0);
            cmd->ComputePushConstants(&pp, sizeof(pp));
            cmd->DispatchCompute(kParticleGroups);
            cmd->ComputeToVertexBarrier();
        };

        // Draw the particles as additive points (inside the render pass, after the scene).
        auto drawParticles = [&](rhi::ICommandBuffer* cmd) {
            cmd->BindPipeline(*particlePipeline);
            cmd->BindVertexBuffer(*particleBuffer);
            cmd->Draw(kParticleCount);
        };

        // Offscreen render target sized to the framebuffer; recreated on resize.
        auto rt = device->CreateRenderTarget(window.FramebufferWidth(),
                                             window.FramebufferHeight());

        // Procedural checkerboard texture (256x256 RGBA8), shared by all renderables.
        std::vector<uint8_t> pixels = MakeCheckerboard();
        auto texture = device->CreateTexture(
            {256, 256, rhi::Format::RGBA8_UNorm, pixels.data(), pixels.size()});

        // Procedural tangent-space normal map (256x256 RGBA8, domed tiles) for the dielectric
        // surfaces, plus a 1x1 flat normal (0,0,1) for surfaces that should stay smooth (metals,
        // duck). Every renderable binds a normal map so the lit shader's gNormalMap is uniform.
        std::vector<uint8_t> normalPixels = MakeBumpyNormalMap();
        auto bumpNormal = device->CreateTexture(
            {256, 256, rhi::Format::RGBA8_UNorm, normalPixels.data(), normalPixels.size()});
        const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
        auto flatNormal = device->CreateTexture(
            {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});

        // Two primitive meshes from the scene layer.
        scene::Mesh cube = scene::Mesh::Cube(*device);
        scene::Mesh plane = scene::Mesh::Plane(*device);
        scene::Mesh sphere = scene::Mesh::Sphere(*device);

        // Real 3D model loaded from glTF: geometry + base-color texture + PBR material factors.
        // The Duck.glb carries an embedded base-color image (decoded via stb) and a dielectric
        // metallic-roughness material, so it renders as a proper textured yellow rubber duck.
        hf::asset::GltfModel duckModel = hf::asset::LoadGltfModel(*device, HF_MODEL_PATH);
        scene::Mesh& duck = duckModel.mesh;

        // The scene is now DATA: register the named GPU resources the scene file refers to, then
        // LoadScene parses assets/scenes/default.json and creates one ECS entity per object IN FILE
        // ORDER (TransformC + MeshC + MaterialC resolved by name). The default scene reproduces the
        // old hardcoded scene exactly (plane, the 3x3 grid in gx/gz order, then the duck), and views
        // iterate the pools in dense (creation) order, so the draw order is byte-identical.
        scene::SceneResources resources;
        resources.AddMesh("cube", &cube);
        resources.AddMesh("plane", &plane);
        resources.AddMesh("sphere", &sphere);
        resources.AddMesh("duck", &duck);
        resources.AddTexture("checker", texture.get());
        resources.AddTexture("normalmap", bumpNormal.get());
        resources.AddTexture("duck_basecolor", duckModel.baseColor.get());
        resources.AddTexture("flat_normal", flatNormal.get());

        ecs::Registry registry;
        std::vector<ecs::Entity> sceneEntities =
            scene::LoadScene(registry, resources, HF_SCENE_PATH);

        // --dump-scene: print the machine-readable scene state and exit (no render).
        if (dumpScene) {
            std::string json = scene::DumpScene(registry, resources);
            std::fputs(json.c_str(), stdout);
            device->WaitIdle();
            return 0;
        }

#ifdef HF_HAS_EDITOR
        // --- Editor shell (Dear ImGui through the RHI). Created only when --editor is set so the
        // normal render path is byte-identical when the editor is off. The ImGui context is given a
        // fixed display size so panels build identically in headless capture and interactive mode. ---
        std::unique_ptr<editor::ImGuiRenderer> uiRenderer;
        editor::EditorState editorState;
        bool imguiInited = false;
        auto ensureEditor = [&]() {
            if (!editor) return;
            if (!imguiInited) {
                IMGUI_CHECKVERSION();
                ImGui::CreateContext();
                ImGuiIO& io = ImGui::GetIO();
                io.IniFilename = nullptr;   // headless: no imgui.ini side-effects
                io.LogFilename = nullptr;
                io.DisplaySize = ImVec2((float)window.FramebufferWidth(),
                                        (float)window.FramebufferHeight());
                ImGui::StyleColorsDark();
                imguiInited = true;
            }
            if (!uiRenderer) {
                uiRenderer = std::make_unique<editor::ImGuiRenderer>(
                    *device, device->Swapchain().ColorFormat(), HF_SHADER_DIR);
            }
        };
        // Record the editor UI into the swapchain pass (after the fullscreen post triangle). Builds a
        // fresh ImGui frame from the live ECS scene, then draws it over the swapchain via the RHI.
        auto recordEditorOverlay = [&](rhi::ICommandBuffer& cmd) {
            if (!editor || !uiRenderer) return;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            ImGuiIO& io = ImGui::GetIO();
            io.DisplaySize = ImVec2((float)w, (float)h);
            ImGui::NewFrame();
            editor::BuildEditorUI(registry, resources, editorState, w, h);
            ImGui::Render();
            uiRenderer->RenderDrawData(ImGui::GetDrawData(), cmd, w, h);
        };
        ensureEditor();
#endif

        // Grid + duck entities spin in the interactive loop (the ground plane, file index 0, does
        // not — mirroring the old "skip index 0" behaviour).
        std::vector<ecs::Entity> cubeSpinEntities;
        for (size_t i = 1; i < sceneEntities.size(); ++i)
            cubeSpinEntities.push_back(sceneEntities[i]);

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

        // Record every renderable in the scene into the command buffer. Queries the ECS:
        // view<TransformC, MeshC, MaterialC>() yields each drawable entity in creation order,
        // recording the identical draws the old Renderable loop did.
        auto drawScene = [&](rhi::ICommandBuffer* cmd) {
            cmd->BindPipeline(*pipeline);
            for (auto [e, tc, mc, mat] : registry.view<scene::TransformC, scene::MeshC, scene::MaterialC>()) {
                (void)e;
                Mat4 m = tc.t.Matrix();
                // Push { float4x4 model; float4 material(metallic,roughness,0,0) } = 80 bytes.
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                pc[16] = mat.metallic; pc[17] = mat.roughness; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd->PushConstants(pc, sizeof(pc));
                // Bind base-color + normal map together (every renderable has a normal map: the
                // procedural bump for dielectrics, the flat default for metals/duck).
                cmd->BindMaterial(*mat.base, *mat.normal);
                cmd->BindVertexBuffer(mc.mesh->vertices());
                cmd->BindIndexBuffer(mc.mesh->indices());
                cmd->DrawIndexed(mc.mesh->indexCount());
            }
        };

        // Depth-only shadow draw: geometry only (no texture), light-space via the shadow pipeline.
        auto drawDepthOnly = [&](rhi::ICommandBuffer* cmd) {
            cmd->BindPipeline(*shadowPipeline);
            for (auto [e, tc, mc, mat] : registry.view<scene::TransformC, scene::MeshC, scene::MaterialC>()) {
                (void)e; (void)mat;
                Mat4 m = tc.t.Matrix();
                cmd->PushConstants(m.m, sizeof(float) * 16);
                cmd->BindVertexBuffer(mc.mesh->vertices());
                cmd->BindIndexBuffer(mc.mesh->indices());
                cmd->DrawIndexed(mc.mesh->indexCount());
            }
        };

        // --- Render exactly one frame of the CURRENT scene and write it to a BMP at `path`. This is
        // the shared headless capture: --shot calls it once; the --commands "capture" op calls it per
        // capture command (so a script's captures reflect whatever mutations preceded them). It reads
        // the live ECS via drawScene/drawDepthOnly, so the duck-moved / sphere-added / cube-removed
        // scene is exactly what lands in the file. Returns false on a write/capture failure. ---
        auto captureToFile = [&](const char* path) -> bool {
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
            FrameData fd = makeFrameData(aspect, 0.6f);  // fixed camera; cubes + lights static

            // --- Build the frame as a declarative render graph. The three passes (shadow -> scene
            // -> post) become graph nodes that DECLARE their resource reads/writes; the graph topo-
            // sorts them by dependency and drives the matching RHI Begin/End scaffolding. Same draws,
            // same order, identical output — just expressed declaratively. ---
            render::RenderGraph graph;
            render::RgResource rgShadow = graph.ImportTarget(
                "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
            render::RgResource rgScene = graph.ImportTarget(
                "sceneColor", render::RgResourceKind::SceneColor, *rt);
            render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

            // Shadow pass: WRITES shadowMap (depth-only draws from the light).
            graph.AddPass("shadow", /*reads*/{}, /*writes*/{rgShadow},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));  // fd has lightViewProj
                    cmd.BeginRenderPass(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f});
                    drawDepthOnly(&cmd);
                    cmd.EndRenderPass();
                });

            // Scene pass: READS shadowMap, WRITES sceneColor (lit scene into the offscreen RT).
            graph.AddPass("scene", /*reads*/{rgShadow}, /*writes*/{rgScene},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    // Compute: advance the GPU particle sim several fixed steps so the fountain has
                    // developed by the captured frame (deterministic -> golden-stable).
                    for (int step = 0; step < 100; ++step)
                        simulateParticles(&cmd, 1.0f / 60.0f, step / 60.0f);
                    cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1.0f});
                    // Sky first: fullscreen gradient + sun, no depth write, behind the geometry.
                    cmd.BindPipeline(*skyPipeline);
                    cmd.Draw(3);
                    drawScene(&cmd);
                    drawParticles(&cmd);  // additive GPU particles over the scene
                    cmd.EndRenderPass();
                });

            // Post pass: READS sceneColor, WRITES swapchain (fullscreen post; then captured). When
            // the editor is enabled, the ImGui panels are drawn over the swapchain in this same pass,
            // after the fullscreen post triangle, so the capture includes the editor UI.
            graph.AddPass("post", /*reads*/{rgScene}, /*writes*/{rgSwap},
                [&](rhi::IRHIDevice& /*dev*/, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f});
                    cmd.BindPipeline(*postPipeline);
                    cmd.BindTexture(*rt);
                    cmd.Draw(3);
#ifdef HF_HAS_EDITOR
                    recordEditorOverlay(cmd);
#endif
                    cmd.EndRenderPass();
                });

            // Arm headless capture before the graph runs the swapchain pass; re-arm on a fresh-
            // swapchain out-of-date acquire (matches the previous inline retry exactly).
            device->CaptureNextFrame();
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px;
            uint32_t cw = 0, ch = 0;
            if (device->GetCapturedPixels(px, cw, ch)) {
                if (!WriteBMP(path, px, cw, ch)) {
                    std::fprintf(stderr, "FATAL: could not write BMP to %s\n", path);
                    device->WaitIdle();
                    return false;
                }
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
                device->WaitIdle();
                return false;
            }
            device->WaitIdle();
            std::printf("wrote %s (%ux%u)\n", path, cw, ch);
            return true;
        };

        // Free the editor UI RHI + ImGui context before the device is torn down (the headless
        // capture/command paths exit here; the interactive loop frees these at the bottom).
        auto teardownEditor = [&]() {
#ifdef HF_HAS_EDITOR
            uiRenderer.reset();
            if (imguiInited) ImGui::DestroyContext();
#endif
        };

        // --- Headless capture mode: render one frame of the default scene, write a BMP, exit. ---
        if (shotPath) {
            bool ok = captureToFile(shotPath);
            teardownEditor();
            return ok ? 0 : 1;
        }

        // --- Agentic command mode: apply the JSON command script to the live ECS (mutate / dump /
        // capture), then exit. The "capture" op renders the CURRENT (mutated) scene via the shared
        // captureToFile path, so a script's captures reflect everything that ran before them. ---
        if (commandsPath) {
            scene::CaptureFn capture = [&](const char* path) { return captureToFile(path); };
            bool ok = scene::RunCommands(registry, resources, commandsPath, capture);
            device->WaitIdle();
            teardownEditor();
            if (!ok) std::fprintf(stderr, "one or more commands failed\n");
            return ok ? 0 : 1;
        }

        // Capture each spinning entity's authored base yaw so the interactive loop can spin
        // the cubes/duck by adding elapsed time on top without drifting. (The ground plane is
        // excluded from cubeSpinEntities, mirroring the old "skip index 0" behaviour.)
        std::vector<float> baseYaw(cubeSpinEntities.size());
        for (size_t i = 0; i < cubeSpinEntities.size(); ++i)
            baseYaw[i] = registry.get<scene::TransformC>(cubeSpinEntities[i]).t.eulerRadians.y;

        const auto start = std::chrono::steady_clock::now();
        float lastT = 0.0f;  // previous frame time, for the particle sim dt

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

            // Spin each cube/duck about its yaw over time (the ground plane is not in the list).
            // Query + mutate the TransformC of each spinning entity each frame.
            for (size_t i = 0; i < cubeSpinEntities.size(); ++i)
                registry.get<scene::TransformC>(cubeSpinEntities[i]).t.eulerRadians.y = baseYaw[i] + t;

            // Compute the particle sim dt up front; the scene pass closure consumes it.
            float dt = std::min(t - lastT, 1.0f / 30.0f);
            lastT = t;

            // --- Same declarative render graph as the headless path, rebuilt each frame so the
            // per-frame closures capture this frame's `fd`, transforms and dt. shadow -> scene ->
            // post, ordered by their declared resource dependencies. ---
            render::RenderGraph graph;
            render::RgResource rgShadow = graph.ImportTarget(
                "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
            render::RgResource rgScene = graph.ImportTarget(
                "sceneColor", render::RgResourceKind::SceneColor, *rt);
            render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

            // Shadow pass: WRITES shadowMap.
            graph.AddPass("shadow", /*reads*/{}, /*writes*/{rgShadow},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));  // fd has lightViewProj
                    cmd.BeginRenderPass(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f});
                    drawDepthOnly(&cmd);
                    cmd.EndRenderPass();
                });

            // Scene pass: READS shadowMap, WRITES sceneColor.
            graph.AddPass("scene", /*reads*/{rgShadow}, /*writes*/{rgScene},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    // Compute: advance the GPU particle sim one frame-step (clamped dt for stability).
                    simulateParticles(&cmd, dt, t);
                    cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1.0f});
                    // Sky first: fullscreen gradient + sun, no depth write, behind the geometry.
                    cmd.BindPipeline(*skyPipeline);
                    cmd.Draw(3);
                    drawScene(&cmd);
                    drawParticles(&cmd);  // additive GPU particles over the scene
                    cmd.EndRenderPass();
                });

            // Post pass: READS sceneColor, WRITES swapchain (the post pass tolerates a skipped frame
            // via the graph's null-cmd guard, exactly as the inline `if (frame.cmd)` did).
            graph.AddPass("post", /*reads*/{rgScene}, /*writes*/{rgSwap},
                [&](rhi::IRHIDevice& /*dev*/, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f});
                    cmd.BindPipeline(*postPipeline);
                    cmd.BindTexture(*rt);
                    cmd.Draw(3);
#ifdef HF_HAS_EDITOR
                    recordEditorOverlay(cmd);
#endif
                    cmd.EndRenderPass();
                });

            graph.Execute(*device);
        }

        device->WaitIdle();
#ifdef HF_HAS_EDITOR
        uiRenderer.reset();  // free UI RHI resources before the device is torn down
        if (imguiInited) ImGui::DestroyContext();
#endif
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
