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

            // ---- Primitive meshes from the scene layer. ----
            scene::Mesh cube = scene::Mesh::Cube(*device);
            scene::Mesh plane = scene::Mesh::Plane(*device);
            scene::Mesh sphere = scene::Mesh::Sphere(*device);

            // ---- Build the scene: a rough-dielectric ground plane + a 3x3 grid mixing shiny metal
            // spheres (on the main diagonal) with matte dielectric cubes — the canonical PBR
            // material showcase, matching the Vulkan hello_triangle scene so both backends render
            // the same material variety (metal spheres next to dielectric cubes). ----
            std::vector<scene::Renderable> sceneObjects;
            {
                scene::Transform planeT;
                planeT.position = {0.0f, 0.0f, 0.0f};
                planeT.scale = {6.0f, 1.0f, 6.0f};
                sceneObjects.push_back({&plane, texture.get(), planeT, /*metallic*/ 0.0f, /*roughness*/ 0.8f});

                for (int gx = -1; gx <= 1; ++gx)
                    for (int gz = -1; gz <= 1; ++gz) {
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

            // ---- Pass 0: depth-only shadow pass from the light into the shadow map. Renders the
            // scene geometry (no textures) through the depth-only pipeline; the lit pass samples the
            // resulting depth for PCF shadows. ----
            {
                auto sc = device->BeginShadowPass(*shadowMap);
                if (!sc.cmd) return fail("BeginShadowPass returned no command buffer");
                device->SetFrameUniforms(&fd, sizeof(FrameData));  // fd has lightViewProj
                sc.cmd->BeginRenderPass(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f});
                sc.cmd->BindPipeline(*shadowPipeline);
                for (scene::Renderable& r : sceneObjects) {
                    Mat4 m = r.transform.Matrix();
                    sc.cmd->PushConstants(m.m, sizeof(float) * 16);
                    sc.cmd->BindVertexBuffer(r.mesh->vertices());
                    sc.cmd->BindIndexBuffer(r.mesh->indices());
                    sc.cmd->DrawIndexed(r.mesh->indexCount());
                }
                sc.cmd->EndRenderPass();
                device->EndShadowPass(sc);
            }

            // ---- Pass 1: render the scene into the offscreen render target (lit, textured). ----
            {
                auto rtc = device->BeginRenderTargetFrame(*rt);
                if (!rtc.cmd) return fail("BeginRenderTargetFrame returned no command buffer");
                device->SetFrameUniforms(&fd, sizeof(FrameData));
                rtc.cmd->BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1.0f});
                rtc.cmd->BindPipeline(*pipeline);
                for (scene::Renderable& r : sceneObjects) {
                    Mat4 m = r.transform.Matrix();
                    // Push { float4x4 model; float4 material(metallic,roughness,0,0) } = 80 bytes.
                    float pc[20];
                    for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                    pc[16] = r.metallic; pc[17] = r.roughness; pc[18] = 0.0f; pc[19] = 0.0f;
                    rtc.cmd->PushConstants(pc, sizeof(pc));
                    rtc.cmd->BindTexture(*r.texture);
                    rtc.cmd->BindVertexBuffer(r.mesh->vertices());
                    rtc.cmd->BindIndexBuffer(r.mesh->indices());
                    rtc.cmd->DrawIndexed(r.mesh->indexCount());
                }
                rtc.cmd->EndRenderPass();
                device->EndRenderTargetFrame(rtc);
            }

            // ---- Pass 2: fullscreen post pass samples the RT into the swapchain output, captured.
            device->CaptureNextFrame();
            auto frame = device->BeginFrame();
            if (!frame.cmd) return fail("BeginFrame returned no command buffer (headless)");

            frame.cmd->BeginRenderPass(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f});
            frame.cmd->BindPipeline(*postPipeline);
            frame.cmd->BindTexture(*rt);
            frame.cmd->Draw(3);
            frame.cmd->EndRenderPass();
            device->EndFrame(frame);

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
