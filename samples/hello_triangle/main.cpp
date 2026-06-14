#include "hal/window.h"
#include "rhi/rhi.h"
#include "rhi/rhi_factory.h"
#include "math/math.h"
#include "scene/vertex.h"
#include "scene/mesh.h"
#include "scene/transform.h"
#include "scene/renderable.h"
#include "asset/gltf_loader.h"
#include "render/render_graph.h"

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

        // Build the scene: a large ground plane + a 3x3 grid of lit cubes.
        std::vector<scene::Renderable> sceneObjects;
        {
            scene::Transform planeT;
            planeT.position = {0.0f, 0.0f, 0.0f};
            planeT.scale = {6.0f, 1.0f, 6.0f};
            // Ground plane: rough dielectric, bumped by the procedural normal map.
            sceneObjects.push_back({&plane, texture.get(), planeT, /*metallic*/ 0.0f, /*roughness*/ 0.8f,
                                    bumpNormal.get()});

            for (int gx = -1; gx <= 1; ++gx) {
                for (int gz = -1; gz <= 1; ++gz) {
                    // Centre cell is reserved for the glTF model (added below); skip it.
                    if (gx == 0 && gz == 0) continue;
                    // Replace the three cells on the main diagonal with spheres so
                    // the scene shows smooth curved geometry alongside the cubes.
                    bool useSphere = (gx == gz);
                    scene::Transform t;
                    if (useSphere) {
                        t.position = {gx * 1.8f, 0.55f, gz * 1.8f};
                        t.scale = {0.55f, 0.55f, 0.55f};
                        // Shiny metal spheres: keep a flat (smooth) normal.
                        sceneObjects.push_back({&sphere, texture.get(), t, /*metallic*/ 1.0f, /*roughness*/ 0.15f,
                                                flatNormal.get()});
                    } else {
                        t.position = {gx * 1.8f, 0.6f, gz * 1.8f};
                        t.eulerRadians = {0.0f, (gx + gz) * 0.5f, 0.0f};
                        t.scale = {0.5f, 0.5f, 0.5f};
                        // Matte dielectric cubes: bumped by the procedural normal map.
                        sceneObjects.push_back({&cube, texture.get(), t, /*metallic*/ 0.0f, /*roughness*/ 0.5f,
                                                bumpNormal.get()});
                    }
                }
            }

            // The glTF model as the centrepiece: a textured rubber Duck. The loader recentred it
            // on the origin (~165 model units across); a modest scale sits it nicely among the
            // cubes. Material comes from the glTF: dielectric (metallic ~0), so it reads as a
            // lit/shadowed textured duck rather than chrome. Base-color texture from the asset.
            scene::Transform duckT;
            duckT.position = {0.0f, 1.35f, 0.0f};
            duckT.eulerRadians = {0.0f, 2.3f, 0.0f};  // turn the bill toward the camera
            duckT.scale = {0.022f, 0.022f, 0.022f};
            float duckRough = duckModel.roughness > 0.0f ? duckModel.roughness : 0.5f;
            sceneObjects.push_back({&duck, duckModel.baseColor.get(), duckT,
                                    duckModel.metallic, duckRough, flatNormal.get()});
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
                // Bind base-color + normal map together (every renderable has a normal map: the
                // procedural bump for dielectrics, the flat default for metals/duck).
                cmd->BindMaterial(*r.texture, *r.normalMap);
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

            // Post pass: READS sceneColor, WRITES swapchain (fullscreen post; then captured).
            graph.AddPass("post", /*reads*/{rgScene}, /*writes*/{rgSwap},
                [&](rhi::IRHIDevice& /*dev*/, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f});
                    cmd.BindPipeline(*postPipeline);
                    cmd.BindTexture(*rt);
                    cmd.Draw(3);
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

            // Spin each cube about its yaw over time (skip index 0: the ground plane).
            for (size_t i = 1; i < sceneObjects.size(); ++i)
                sceneObjects[i].transform.eulerRadians.y = baseYaw[i] + t;

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
                    cmd.EndRenderPass();
                });

            graph.Execute(*device);
        }

        device->WaitIdle();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
