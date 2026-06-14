// Headless Metal RHI bring-up on Apple Silicon over SSH (no window server).
//
// This drives the REAL Metal RHI backend exactly like a sample would: it constructs the real
// MetalDevice in HEADLESS mode (offscreen MTLTexture color + D32 depth, no window/CAMetalLayer),
// builds the full Slice-F scene from the engine's scene layer (Mesh::Cube + Mesh::Plane: a ground
// plane + a 3x3 grid of lit cubes, a procedural checkerboard texture), a lit graphics pipeline
// from GENERATED MSL (the *.gen.metal produced at build time from the shared HLSL sources via
// HLSL -> SPIR-V -> spirv-cross; see the sibling CMakeLists.txt), a per-frame UBO with a
// directional light, and per-object push-constant model matrices. It renders ONE frame, reads the
// offscreen texture back via CaptureNextFrame()/GetCapturedPixels() and writes a PNG via ImageIO.
//
// Unlike the first bring-up cut, this exercises MetalDevice/MetalSwapchain/MetalCommandBuffer/
// MetalPipeline/MetalTexture/MetalBuffer through the IRHIDevice/ICommandBuffer interfaces — no
// inline pipeline/encoder shims.
//
//   Build: configured by the sibling CMakeLists.txt
//   Run:   ./visual_test out.png
//   Exit:  0 = success, non-zero = failure (reason on stderr)

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include "rhi/rhi.h"
#include "rhi_metal/metal_offscreen.h"   // CreateMetalDeviceHeadless (Metal-free)
#include "rhi_metal/metal_shader_load.h" // MakeShaderModuleFromMSL (Metal-free)
#include "math/math.h"
#include "scene/vertex.h"
#include "scene/mesh.h"
#include "scene/transform.h"
#include "scene/renderable.h"
#include "asset/gltf_loader.h"
#include "render/render_graph.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hf;

static int fail(const std::string& msg) {
    std::fprintf(stderr, "ERROR: %s\n", msg.c_str());
    return 1;
}

// ---- Per-frame uniform block — must match shaders/lit.metal + shaders/shadow.metal FrameData
// (288 bytes) byte-for-byte. Layout mirrors the Vulkan sample (samples/hello_triangle/main.cpp). ----
struct FrameData {
    float vp[16];
    float lightDir[4];
    float lightColor[4];
    float viewPos[4];
    float ptCount[4];          // x = number of active point lights (unused here)
    float ptPos[3][4];
    float ptColor[3][4];
    float lightViewProj[16];   // directional light's view*ortho (for shadow mapping)
    float camFwd[4];           // sky/camera-basis fields (unused here; layout parity)
    float camRight[4];
    float camUp[4];
    float skyParams[4];
};

static std::string LoadText(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::streamsize size = f.tellg();
    f.seekg(0);
    std::string s(static_cast<size_t>(size), '\0');
    f.read(s.data(), size);
    return s;
}

// 256x256 RGBA8 checkerboard — same generator as the Vulkan sample, for parity.
static std::vector<uint8_t> MakeCheckerboard() {
    const uint32_t kSize = 256, kTiles = 8, kTilePx = kSize / kTiles;
    std::vector<uint8_t> px(static_cast<size_t>(kSize) * kSize * 4);
    for (uint32_t y = 0; y < kSize; ++y)
        for (uint32_t x = 0; x < kSize; ++x) {
            uint32_t tx = x / kTilePx, ty = y / kTilePx;
            bool dark = ((tx + ty) & 1) != 0;
            uint8_t pxc = static_cast<uint8_t>(40 + tx * (215 / (kTiles - 1)));
            uint8_t pyc = static_cast<uint8_t>(40 + ty * (215 / (kTiles - 1)));
            size_t idx = (static_cast<size_t>(y) * kSize + x) * 4;
            if (dark) { px[idx] = pxc / 4; px[idx+1] = pyc / 4; px[idx+2] = 60; }
            else      { px[idx] = pxc;     px[idx+1] = pyc;     px[idx+2] = 230; }
            px[idx+3] = 255;
        }
    return px;
}

// Procedural 256x256 RGBA8 tangent-space normal map: an 8x8 grid of radial domes (one per
// checkerboard tile), encoded 0..1 (flat = (128,128,255)). Identical generator to the Vulkan
// sample so both backends bump the dielectric surfaces the same way.
static std::vector<uint8_t> MakeBumpyNormalMap() {
    const uint32_t kSize = 256, kTiles = 8;
    const float kTilePx = (float)kSize / (float)kTiles;
    const float kBump = 4.0f;  // visibly domed tiles (matches the Vulkan sample)
    auto height = [&](float x, float y) {
        float lx = std::fmod(x, kTilePx) / kTilePx - 0.5f;
        float ly = std::fmod(y, kTilePx) / kTilePx - 0.5f;
        float r = std::sqrt(lx * lx + ly * ly) / 0.5f;
        if (r >= 1.0f) return 0.0f;
        float c = std::cos(1.5707963f * r);
        return c * c;
    };
    std::vector<uint8_t> px(static_cast<size_t>(kSize) * kSize * 4);
    for (uint32_t y = 0; y < kSize; ++y)
        for (uint32_t x = 0; x < kSize; ++x) {
            float xl = (float)((x + kSize - 1) % kSize), xr = (float)((x + 1) % kSize);
            float yd = (float)((y + kSize - 1) % kSize), yu = (float)((y + 1) % kSize);
            float dhdx = (height(xr, (float)y) - height(xl, (float)y)) * 0.5f;
            float dhdy = (height((float)x, yu) - height((float)x, yd)) * 0.5f;
            float nx = -dhdx * kBump, ny = -dhdy * kBump, nz = 1.0f;
            float inv = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
            nx *= inv; ny *= inv; nz *= inv;
            size_t idx = (static_cast<size_t>(y) * kSize + x) * 4;
            px[idx + 0] = (uint8_t)std::lround((nx * 0.5f + 0.5f) * 255.0f);
            px[idx + 1] = (uint8_t)std::lround((ny * 0.5f + 0.5f) * 255.0f);
            px[idx + 2] = (uint8_t)std::lround((nz * 0.5f + 0.5f) * 255.0f);
            px[idx + 3] = 255;
        }
    return px;
}

static bool WritePNG(const char* outPath, const std::vector<uint8_t>& bgra,
                     uint32_t W, uint32_t H) {
    // Captured pixels are tightly-packed BGRA8, top row first. Swap to RGBA for ImageIO.
    std::vector<uint8_t> rgba = bgra;
    for (size_t i = 0; i + 3 < rgba.size(); i += 4) std::swap(rgba[i], rgba[i + 2]);

    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(rgba.data(), W, H, 8, (size_t)W * 4, cs,
                                             kCGImageAlphaPremultipliedLast);
    CGImageRef img = CGBitmapContextCreateImage(ctx);

    CFStringRef typePNG = (__bridge CFStringRef)UTTypePNG.identifier;
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        nullptr, (const UInt8*)outPath, strlen(outPath), false);
    CGImageDestinationRef dst = CGImageDestinationCreateWithURL(url, typePNG, 1, nullptr);
    bool ok = false;
    if (dst) {
        CGImageDestinationAddImage(dst, img, nullptr);
        ok = CGImageDestinationFinalize(dst);
        CFRelease(dst);
    }
    CFRelease(url); CGImageRelease(img);
    CGContextRelease(ctx); CGColorSpaceRelease(cs);
    return ok;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        const char* outPath = argc > 1 ? argv[1] : "metal_scene.png";
        const uint32_t W = 1280, H = 720;

        try {
            // ---- Real Metal RHI device, headless (offscreen color + depth, no window). ----
            auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

            // ---- Shaders: compile GENERATED MSL at runtime through the RHI. The MSL is produced
            // at build time from the SHARED HLSL sources (HLSL -> SPIR-V via glslc -> MSL via
            // spirv-cross), so there is no hand-written MSL to drift. HF_GEN_SHADER_DIR points at
            // the build dir where the .gen.metal files were emitted; entry points were renamed by
            // spirv-cross to the stable names used here. ----
            std::string msl = LoadText(std::string(HF_GEN_SHADER_DIR) + "/lit.vert.gen.metal");
            std::string mslF = LoadText(std::string(HF_GEN_SHADER_DIR) + "/lit.frag.gen.metal");
            auto vs = rhi::mtl::MakeShaderModuleFromMSL(*device, msl, "vertex_main");
            auto fs = rhi::mtl::MakeShaderModuleFromMSL(*device, mslF, "fragment_main");

            // ---- Lit graphics pipeline (color = swapchain BGRA, depth test on). ----
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

            // ---- Fullscreen post pipeline: samples the offscreen RT, applies ACES tonemap +
            // gamma + grade + vignette, writes the swapchain output (which gets captured). ----
            std::string postVsMsl = LoadText(std::string(HF_GEN_SHADER_DIR) + "/post.vert.gen.metal");
            std::string postFsMsl = LoadText(std::string(HF_GEN_SHADER_DIR) + "/post.frag.gen.metal");
            auto postVs = rhi::mtl::MakeShaderModuleFromMSL(*device, postVsMsl, "post_vertex");
            auto postFs = rhi::mtl::MakeShaderModuleFromMSL(*device, postFsMsl, "post_fragment");

            rhi::GraphicsPipelineDesc postDesc;
            postDesc.vertex = postVs.get();
            postDesc.fragment = postFs.get();
            postDesc.colorFormat = device->Swapchain().ColorFormat();
            postDesc.depthTest = false;          // fullscreen pass, no depth
            postDesc.usesFrameUniforms = false;  // no per-frame UBO
            postDesc.usesTexture = true;         // samples the RT color image
            postDesc.fullscreen = true;          // no vertex input; 3 verts from [[vertex_id]]
            auto postPipeline = device->CreateGraphicsPipeline(postDesc);

            // ---- Depth-only shadow pipeline: transforms geometry into the light's clip space and
            // writes only depth (no fragment stage, no color attachment). Needs lightViewProj from
            // the per-frame UBO; no texture. ----
            std::string shadowMsl = LoadText(std::string(HF_GEN_SHADER_DIR) + "/shadow.vert.gen.metal");
            auto shadowVs = rhi::mtl::MakeShaderModuleFromMSL(*device, shadowMsl, "shadow_vertex");

            rhi::GraphicsPipelineDesc shadowDesc;
            shadowDesc.vertex = shadowVs.get();
            shadowDesc.fragment = nullptr;                // depth-only: no fragment stage
            shadowDesc.vertexLayout = scene::MeshVertexLayout();
            shadowDesc.depthTest = true;
            shadowDesc.depthOnly = true;                  // no color attachment, depth write + bias
            shadowDesc.usesFrameUniforms = true;          // lightViewProj lives in the frame UBO
            shadowDesc.usesTexture = false;
            shadowDesc.pushConstantSize = sizeof(float) * 16;  // mat4 model
            auto shadowPipeline = device->CreateGraphicsPipeline(shadowDesc);

            // ---- Procedural sky pipeline: a fullscreen triangle drawn FIRST in the scene->RT
            // pass as the background. Reads the camera basis (camFwd/Right/Up + skyParams) from the
            // per-frame UBO; writes no depth so the scene geometry draws over it. Built from the
            // GENERATED sky MSL (sky.vert/sky.frag .gen.metal), mirroring the Vulkan sample so the
            // Metal background shows the same gradient sky instead of a flat clear color. ----
            std::string skyVsMsl = LoadText(std::string(HF_GEN_SHADER_DIR) + "/sky.vert.gen.metal");
            std::string skyFsMsl = LoadText(std::string(HF_GEN_SHADER_DIR) + "/sky.frag.gen.metal");
            auto skyVs = rhi::mtl::MakeShaderModuleFromMSL(*device, skyVsMsl, "sky_vertex");
            auto skyFs = rhi::mtl::MakeShaderModuleFromMSL(*device, skyFsMsl, "sky_fragment");

            rhi::GraphicsPipelineDesc skyDesc;
            skyDesc.vertex = skyVs.get();
            skyDesc.fragment = skyFs.get();
            skyDesc.colorFormat = device->Swapchain().ColorFormat();
            skyDesc.depthTest = false;           // background fill: no depth test, no depth write
            skyDesc.usesFrameUniforms = true;    // camera basis lives in the frame UBO
            skyDesc.usesTexture = false;
            skyDesc.fullscreen = true;           // 3 verts from [[vertex_id]]; no vertex input
            auto skyPipeline = device->CreateGraphicsPipeline(skyDesc);

            // ---- GPU particle system: a compute kernel (generated from particles.comp.hlsl)
            // animates a storage buffer of 50k particles each frame; particle.vert/frag draw them
            // as additive points over the scene. Mirrors the Vulkan hello_triangle integration. ----
            constexpr uint32_t kParticleCount = 50000;
            constexpr uint32_t kParticleStride = 32;  // { float4 posLife; float4 velSeed; }
            struct GpuParticle { float posLife[4]; float velSeed[4]; };
            static_assert(sizeof(GpuParticle) == kParticleStride, "particle stride");

            std::vector<GpuParticle> initParticles(kParticleCount);
            for (uint32_t i = 0; i < kParticleCount; ++i) {
                float s = (float)i / (float)kParticleCount;
                float a = s * 6.2831853f;
                float r = 0.25f * (0.5f + 0.5f * std::sin(s * 12.9898f));
                initParticles[i].posLife[0] = std::cos(a) * r;
                initParticles[i].posLife[1] = 0.05f + 2.0f * s;
                initParticles[i].posLife[2] = std::sin(a) * r;
                initParticles[i].posLife[3] = 0.5f + 3.5f * s;
                float outR = 0.5f + 2.8f * std::fabs(std::sin(s * 7.13f));
                float aVel = s * 9.41f;
                initParticles[i].velSeed[0] = std::cos(aVel) * outR;
                initParticles[i].velSeed[1] = 3.5f + 3.5f * s;
                initParticles[i].velSeed[2] = std::sin(aVel) * outR;
                initParticles[i].velSeed[3] = s;
            }
            rhi::BufferDesc particleBufDesc;
            particleBufDesc.size = (uint64_t)kParticleCount * kParticleStride;
            particleBufDesc.initialData = initParticles.data();
            particleBufDesc.usage = rhi::BufferUsage::Storage;
            auto particleBuffer = device->CreateBuffer(particleBufDesc);

            std::string partCsMsl = LoadText(std::string(HF_GEN_SHADER_DIR) + "/particles.comp.gen.metal");
            auto partCs = rhi::mtl::MakeShaderModuleFromMSL(*device, partCsMsl, "particles_main");
            rhi::ComputePipelineDesc cdesc;
            cdesc.compute = partCs.get();
            cdesc.storageBufferCount = 1;
            cdesc.pushConstantSize = sizeof(float) * 2 + sizeof(uint32_t) * 2;
            auto particleCompute = device->CreateComputePipeline(cdesc);

            std::string partVsMsl = LoadText(std::string(HF_GEN_SHADER_DIR) + "/particle.vert.gen.metal");
            std::string partFsMsl = LoadText(std::string(HF_GEN_SHADER_DIR) + "/particle.frag.gen.metal");
            auto partVs = rhi::mtl::MakeShaderModuleFromMSL(*device, partVsMsl, "particle_vertex");
            auto partFs = rhi::mtl::MakeShaderModuleFromMSL(*device, partFsMsl, "particle_fragment");
            rhi::GraphicsPipelineDesc partDesc;
            partDesc.vertex = partVs.get();
            partDesc.fragment = partFs.get();
            partDesc.vertexLayout.stride = kParticleStride;
            partDesc.vertexLayout.attributes = {
                {0, rhi::Format::RGB32_Float, 0},
                {1, rhi::Format::RGB32_Float, 16},
            };
            partDesc.colorFormat = device->Swapchain().ColorFormat();
            partDesc.depthTest = false;
            partDesc.usesFrameUniforms = true;
            partDesc.usesTexture = false;
            partDesc.pointList = true;
            partDesc.additiveBlend = true;
            auto particlePipeline = device->CreateGraphicsPipeline(partDesc);

            struct ParticleParams { float dt; float time; uint32_t count; uint32_t pad; };
            const uint32_t kParticleGroups = (kParticleCount + 63) / 64;

            // ---- Offscreen render target (scene -> RT -> post -> output), at output size. ----
            auto rt = device->CreateRenderTarget(W, H);

            // ---- Shadow map: a 2048x2048 depth-only sampleable target. SetShadowMap points the lit
            // pass's fragment shadow slots (texture/sampler index 1) at it. ----
            auto shadowMap = device->CreateShadowMap(2048);
            device->SetShadowMap(*shadowMap);

            // ---- Procedural checkerboard texture (256x256 RGBA8), shared by all renderables. ----
            std::vector<uint8_t> pixels = MakeCheckerboard();
            auto texture = device->CreateTexture(
                {256, 256, rhi::Format::RGBA8_UNorm, pixels.data(), pixels.size()});

            // Procedural tangent-space normal map (domed tiles) for the dielectric surfaces + a 1x1
            // flat normal (0,0,1) for smooth surfaces. Same generators as the Vulkan sample so both
            // backends bump identically; every renderable binds a normal map (uniform gNormalMap).
            std::vector<uint8_t> normalPixels = MakeBumpyNormalMap();
            auto bumpNormal = device->CreateTexture(
                {256, 256, rhi::Format::RGBA8_UNorm, normalPixels.data(), normalPixels.size()});
            const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
            auto flatNormal = device->CreateTexture(
                {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});

            // ---- Primitive meshes from the scene layer. ----
            scene::Mesh cube = scene::Mesh::Cube(*device);
            scene::Mesh plane = scene::Mesh::Plane(*device);
            scene::Mesh sphere = scene::Mesh::Sphere(*device);

            // ---- Real 3D model loaded from glTF: geometry + base-color texture + PBR factors.
            // Duck.glb embeds a base-color image (decoded via stb) and a dielectric material, so
            // it renders as a textured rubber duck. Same loader/layout/pipeline as Vulkan. ----
            hf::asset::GltfModel duckModel = hf::asset::LoadGltfModel(*device, HF_MODEL_PATH);
            scene::Mesh& duck = duckModel.mesh;

            // ---- Build the scene: a rough-dielectric ground plane + a 3x3 grid mixing shiny metal
            // spheres (on the main diagonal) with matte dielectric cubes — the canonical PBR
            // material showcase, matching the Vulkan hello_triangle scene so both backends render
            // the same material variety (metal spheres next to dielectric cubes). ----
            std::vector<scene::Renderable> sceneObjects;
            {
                scene::Transform planeT;
                planeT.position = {0.0f, 0.0f, 0.0f};
                planeT.scale = {6.0f, 1.0f, 6.0f};
                sceneObjects.push_back({&plane, texture.get(), planeT, /*metallic*/ 0.0f, /*roughness*/ 0.8f,
                                        bumpNormal.get()});

                for (int gx = -1; gx <= 1; ++gx)
                    for (int gz = -1; gz <= 1; ++gz) {
                        // Centre cell is reserved for the glTF model (added below); skip it.
                        if (gx == 0 && gz == 0) continue;
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

                // The glTF model as the centrepiece: a textured rubber Duck. Material from the
                // glTF (dielectric, base-color texture). Placement matches the Vulkan sample so
                // both backends render the same scene.
                scene::Transform duckT;
                duckT.position = {0.0f, 1.35f, 0.0f};
                duckT.eulerRadians = {0.0f, 2.3f, 0.0f};
                duckT.scale = {0.022f, 0.022f, 0.022f};
                float duckRough = duckModel.roughness > 0.0f ? duckModel.roughness : 0.5f;
                sceneObjects.push_back({&duck, duckModel.baseColor.get(), duckT,
                                        duckModel.metallic, duckRough, flatNormal.get()});
            }

            // ---- Frame uniforms: same camera + light as the Vulkan Slice-F sample. ----
            using math::Mat4; using math::Vec3;
            const Vec3 eye{4.5f, 4.0f, 6.5f};
            const Vec3 center{0.0f, 0.5f, 0.0f};
            const float aspect = (float)W / (float)H;
            Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
            // NDC-Y convention, owned by the Metal BACKEND (CPU-side), not the shaders. math::
            // Perspective/Ortho bake the Vulkan clip-space Y flip (+Y down). Metal NDC is +Y up, so
            // we undo the flip here by negating the projection's Y row (column-major rows are m[1],
            // m[5], m[9], m[13]) BEFORE composing the view-proj. This means the SHARED HLSL->MSL
            // shaders need NO Metal-specific clip.y flip, so the generated MSL works unchanged.
            auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                          p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };
            Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f /*60deg*/, aspect, 0.1f, 100.0f));
            Mat4 vp = proj * view;

            FrameData fd{};
            for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
            fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f; fd.lightDir[3] = 0;
            fd.lightColor[0] = fd.lightColor[1] = fd.lightColor[2] = fd.lightColor[3] = 1.0f;
            fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;

            // ---- Directional light's view*projection for shadow mapping. Same construction as the
            // Vulkan Slice-I sample: the light looks from sceneCenter - lightDir*12 toward the
            // scene center; an ortho box covers the 3x3 cube grid + ground plane. ----
            Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
            Vec3 sceneCenter{0.0f, 0.5f, 0.0f};
            Vec3 lightEye = sceneCenter - lightDir * 12.0f;
            Mat4 lightView = Mat4::LookAt(lightEye, sceneCenter, {0, 1, 0});
            // Same CPU-side Y-flip for the shadow light's ortho projection: the shadow-map render
            // (generated shadow.vert MSL, no clip.y flip) and the lit pass's sample formula
            // (smUV = proj.xy*0.5+0.5) stay self-consistent because BOTH derive from this flipped
            // lightViewProj — exactly as before, just with the flip moved CPU-side.
            Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 25.0f));
            Mat4 lightVP = lightOrtho * lightView;
            for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];

            // ---- Camera basis (world space) for the sky shader's view-ray reconstruction (no
            // matrix inverse). Mirrors the Vulkan sample: fwd toward center, right = fwd x worldUp,
            // up = right x fwd. The sky.frag MSL maps NDC -> ray using these + skyParams. Note the
            // Metal projection Y was flipped CPU-side (+Y up); the sky shader already negates
            // ndc.y, and reconstructs the ray from the world-space basis, so it stays consistent. ----
            Vec3 fwd = math::normalize(center - eye);
            Vec3 right = math::normalize(math::cross(fwd, Vec3{0.0f, 1.0f, 0.0f}));
            Vec3 up = math::cross(right, fwd);
            fd.camFwd[0] = fwd.x;     fd.camFwd[1] = fwd.y;     fd.camFwd[2] = fwd.z;     fd.camFwd[3] = 0.0f;
            fd.camRight[0] = right.x; fd.camRight[1] = right.y; fd.camRight[2] = right.z; fd.camRight[3] = 0.0f;
            fd.camUp[0] = up.x;       fd.camUp[1] = up.y;       fd.camUp[2] = up.z;       fd.camUp[3] = 0.0f;
            fd.skyParams[0] = std::tan(0.5f * 1.04719755f);  // tan(half of 60deg fovY)
            fd.skyParams[1] = aspect;
            fd.skyParams[2] = 0.0f; fd.skyParams[3] = 0.0f;

            // ---- Build the frame as a declarative render graph. The three passes (shadow -> scene
            // -> post) become graph nodes that DECLARE their resource reads/writes; the graph topo-
            // sorts them by dependency and drives the matching RHI Begin/End scaffolding. Same draws,
            // same order as before — the rendered output is byte-identical (golden DIFF 0.0000). ----
            render::RenderGraph graph;
            render::RgResource rgShadow = graph.ImportTarget(
                "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
            render::RgResource rgScene = graph.ImportTarget(
                "sceneColor", render::RgResourceKind::SceneColor, *rt);
            render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

            // ---- Pass 0 (shadow): WRITES shadowMap. Depth-only draws from the light; the lit pass
            // samples the resulting depth for PCF shadows. ----
            graph.AddPass("shadow", /*reads*/{}, /*writes*/{rgShadow},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));  // fd has lightViewProj
                    cmd.BeginRenderPass(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f});
                    cmd.BindPipeline(*shadowPipeline);
                    for (scene::Renderable& r : sceneObjects) {
                        Mat4 m = r.transform.Matrix();
                        cmd.PushConstants(m.m, sizeof(float) * 16);
                        cmd.BindVertexBuffer(r.mesh->vertices());
                        cmd.BindIndexBuffer(r.mesh->indices());
                        cmd.DrawIndexed(r.mesh->indexCount());
                    }
                    cmd.EndRenderPass();
                });

            // ---- Pass 1 (scene): READS shadowMap, WRITES sceneColor. Lit, textured scene into the
            // offscreen RT (sky + geometry + additive GPU particles). ----
            graph.AddPass("scene", /*reads*/{rgShadow}, /*writes*/{rgScene},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    // Compute: advance the GPU particle sim a fixed number of deterministic steps so
                    // the fountain has developed by the captured frame (golden-stable).
                    for (int step = 0; step < 100; ++step) {
                        ParticleParams pp{1.0f / 60.0f, step / 60.0f, kParticleCount, 0};
                        cmd.BindComputePipeline(*particleCompute);
                        cmd.BindStorageBuffer(*particleBuffer, 0);
                        cmd.ComputePushConstants(&pp, sizeof(pp));
                        cmd.DispatchCompute(kParticleGroups);
                        cmd.ComputeToVertexBarrier();
                    }
                    cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1.0f});
                    // Sky first: fullscreen gradient + sun, no depth write, behind the geometry.
                    cmd.BindPipeline(*skyPipeline);
                    cmd.Draw(3);
                    cmd.BindPipeline(*pipeline);
                    for (scene::Renderable& r : sceneObjects) {
                        Mat4 m = r.transform.Matrix();
                        // Push { float4x4 model; float4 material(metallic,roughness,0,0) } = 80 bytes.
                        float pc[20];
                        for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                        pc[16] = r.metallic; pc[17] = r.roughness; pc[18] = 0.0f; pc[19] = 0.0f;
                        cmd.PushConstants(pc, sizeof(pc));
                        cmd.BindMaterial(*r.texture, *r.normalMap);
                        cmd.BindVertexBuffer(r.mesh->vertices());
                        cmd.BindIndexBuffer(r.mesh->indices());
                        cmd.DrawIndexed(r.mesh->indexCount());
                    }
                    // Additive GPU particles over the scene.
                    cmd.BindPipeline(*particlePipeline);
                    cmd.BindVertexBuffer(*particleBuffer);
                    cmd.Draw(kParticleCount);
                    cmd.EndRenderPass();
                });

            // ---- Pass 2 (post): READS sceneColor, WRITES swapchain. Fullscreen post samples the RT
            // into the swapchain output, which is then captured. ----
            graph.AddPass("post", /*reads*/{rgScene}, /*writes*/{rgSwap},
                [&](rhi::IRHIDevice& /*dev*/, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f});
                    cmd.BindPipeline(*postPipeline);
                    cmd.BindTexture(*rt);
                    cmd.Draw(3);
                    cmd.EndRenderPass();
                });

            // Arm headless capture before the graph runs the swapchain pass, then execute.
            device->CaptureNextFrame();
            graph.Execute(*device);

            std::vector<uint8_t> bgra;
            uint32_t cw = 0, ch = 0;
            if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");

            if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
            device->WaitIdle();

            std::printf("OK wrote %s (%ux%u)\n", outPath, cw, ch);
            return 0;
        } catch (const std::exception& e) {
            return fail(std::string("exception: ") + e.what());
        }
    }
}
