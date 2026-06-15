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
#include "scene/instance_grid.h"
#include "scene/transform.h"
#include "scene/renderable.h"
#include "scene/components.h"
#include "scene/scene_io.h"
#include "ecs/ecs.h"
#include "asset/gltf_loader.h"
#include "asset/env_loader.h"
#include "anim/animation.h"
#include "anim/skeleton.h"
#include "physics/world.h"
#include "physics/body.h"
#include "game/roll_game.h"
#include "ui/text.h"               // Slice BA: baked 8x8 font atlas + screen-space text layout (pure CPU)
#include "render/render_graph.h"
#include "render/csm.h"     // Slice AD: cascaded-shadow split + per-cascade ortho fit (pure math)
#include "render/spot.h"    // Slice AE: spot-light perspective shadow projection + cone (pure math)
#include "render/point_shadow.h" // Slice AF: omnidirectional point-light 6-face cube shadow (pure math)
#include "render/probe.h"        // Slice AK: reflection + irradiance probe atlas math (pure math)
#include "render/clustered.h"     // Slice AG: clustered / Forward+ light culling (pure math)
#include "render/taa.h"           // Slice AP: temporal anti-aliasing jitter + resolve-blend (pure math)
#include "render/frustum.h"        // Slice AQ: Gribb-Hartmann frustum extraction + sphere cull (pure math)
#include "render/gpu_cull.h"        // Slice AR: GPU-cull CPU mirror (ordered compaction + sphere test)
#include "debug/debug_draw.h"
#include "debug/debug_emitters.h"
#include "runtime/camera.h"  // Slice AA: backend-agnostic Camera for the scripted-pose --camera path
#include "runtime/parallel_record.h"  // Slice AU: deterministic parallel command-recording partition
#include "editor/gizmo.h"    // Slice AB: transform gizmo emit (drawn through the debug-line layer)

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
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
    float prevViewProj[16];    // TAA (Slice AP): previous frame's view-proj (layout parity; identity reprojection)
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

// --- Skeletal-animation showcase (Slice O). Mirrors the Vulkan --skinning-shot path: ground plane
// + procedural sky + the GPU-skinned Fox sampled at animation "Survey", time 0.5s, lit + shadowed.
// One offscreen frame -> PNG. The MSL is generated from the shared HLSL (lit_skinned/shadow_skinned)
// by the sibling CMake gen rules; the joint palette binds at vertex buffer(3) (set 2). ----------
// `blend` selects the palette source: false = single-clip sample of "Survey" t=0.5s (Slice O);
// true (Slice X) = 50/50 cross-clip blend of "Walk" t=0.3s and "Run" t=0.2s via BlendAnimations.
// Everything else (scene/camera/light/pipelines) is shared so the two PNGs are directly comparable.
static int RunSkinningShowcase(const char* outPath, bool blend = false) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };

    // Same CPU-side Y-flip convention as the main showcase (Metal NDC +Y up).
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Skinned lit pipeline (reuses lit.frag).
    auto skVs = loadMSL("lit_skinned.vert.gen.metal", "skinned_vertex");
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc skDesc;
    skDesc.vertex = skVs.get(); skDesc.fragment = litFs.get();
    skDesc.vertexLayout = scene::SkinnedMeshVertexLayout();
    skDesc.colorFormat = device->Swapchain().ColorFormat();
    skDesc.depthTest = true; skDesc.usesFrameUniforms = true;
    skDesc.usesTexture = true; skDesc.usesJointPalette = true;
    skDesc.pushConstantSize = sizeof(float) * 20;
    auto skinnedPipeline = device->CreateGraphicsPipeline(skDesc);

    // Static lit pipeline (ground plane).
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Skinned + static depth-only shadow pipelines.
    auto skShadowVs = loadMSL("shadow_skinned.vert.gen.metal", "skinned_shadow_vertex");
    rhi::GraphicsPipelineDesc skShDesc;
    skShDesc.vertex = skShadowVs.get(); skShDesc.fragment = nullptr;
    skShDesc.vertexLayout = scene::SkinnedMeshVertexLayout();
    skShDesc.depthTest = true; skShDesc.depthOnly = true;
    skShDesc.usesFrameUniforms = true; skShDesc.usesJointPalette = true;
    skShDesc.pushConstantSize = sizeof(float) * 16;
    auto skinnedShadowPipeline = device->CreateGraphicsPipeline(skShDesc);

    auto staticShadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc stShDesc;
    stShDesc.vertex = staticShadowVs.get(); stShDesc.fragment = nullptr;
    stShDesc.vertexLayout = scene::MeshVertexLayout();
    stShDesc.depthTest = true; stShDesc.depthOnly = true;
    stShDesc.usesFrameUniforms = true; stShDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(stShDesc);

    // Sky + post.
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh plane = scene::Mesh::Plane(*device);

    hf::asset::SkinnedModel fox = hf::asset::LoadSkinnedGltfModel(*device, HF_FOX_MODEL_PATH);
    std::vector<Mat4> palette;
    if (blend) {
        const anim::Animation* walk = fox.FindAnimation("Walk");
        const anim::Animation* run  = fox.FindAnimation("Run");
        if (!walk && !fox.animations.empty()) walk = &fox.animations.front();
        if (!run) run = walk;
        palette = (walk && run)
            ? anim::BlendAnimations(fox.skeleton, *walk, 0.3f, *run, 0.2f, 0.5f)
            : std::vector<Mat4>(fox.skeleton.joints.size(), Mat4::Identity());
    } else {
        const anim::Animation* survey = fox.FindAnimation("Survey");
        if (!survey && !fox.animations.empty()) survey = &fox.animations.front();
        palette = survey
            ? anim::SampleAnimation(fox.skeleton, *survey, 0.5f)
            : std::vector<Mat4>(fox.skeleton.joints.size(), Mat4::Identity());
    }
    std::vector<float> paletteData(64 * 16);
    for (int j = 0; j < 64; ++j) {
        Mat4 mm = (j < (int)palette.size()) ? palette[j] : Mat4::Identity();
        for (int k = 0; k < 16; ++k) paletteData[j * 16 + k] = mm.m[k];
    }

    float foxH = fox.bbMax[1] - fox.bbMin[1];
    float scaleS = (foxH > 1e-4f) ? (2.5f / foxH) : 0.05f;
    float cx = 0.5f * (fox.bbMin[0] + fox.bbMax[0]);
    float cz = 0.5f * (fox.bbMin[2] + fox.bbMax[2]);
    Mat4 foxModel = Mat4::Translate({-cx * scaleS, -fox.bbMin[1] * scaleS, -cz * scaleS})
                  * Mat4::Scale({scaleS, scaleS, scaleS});
    Mat4 groundModel = Mat4::Scale({8.0f, 1.0f, 8.0f});

    const Vec3 eye{3.5f, 2.6f, 4.5f};
    const Vec3 center{0.0f, 1.0f, 0.0f};
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 0.98f; fd.lightColor[1] = 0.95f; fd.lightColor[2] = 0.88f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 sc{0.0f, 1.0f, 0.0f};
        Vec3 lightEye = sc - lightDir * 12.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-6.0f, 6.0f, -6.0f, 6.0f, 1.0f, 25.0f));
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
            dev.SetJointPalette(paletteData.data(), paletteData.size() * sizeof(float));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*staticShadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(plane.vertices());
            cmd.BindIndexBuffer(plane.indices());
            cmd.DrawIndexed(plane.indexCount());
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u)\n", outPath, cw, ch);
    return 0;
}

// --- Full-PBR showcase (Slice P). Mirrors the Vulkan --pbr-shot path: ground plane + procedural sky
// + the DamagedHelmet rendered with the full glTF metallic-roughness material set
// (base/metalRough/normal/emissive/occlusion), lit + shadowed. One offscreen frame -> PNG. The MSL
// is generated from the shared HLSL (lit.vert + lit_pbr.frag) by the sibling CMake gen rules; the
// full-PBR material textures bind at the flat fragment indices from metal_common.h. ---------------
static int RunPbrShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Lit-PBR pipeline (shared lit.vert + lit_pbr.frag; full 5-texture material set).
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto pbrFs = loadMSL("lit_pbr.frag.gen.metal", "pbr_fragment");
    rhi::GraphicsPipelineDesc pbrDesc;
    pbrDesc.vertex = litVs.get(); pbrDesc.fragment = pbrFs.get();
    pbrDesc.vertexLayout = scene::MeshVertexLayout();
    pbrDesc.colorFormat = device->Swapchain().ColorFormat();
    pbrDesc.depthTest = true; pbrDesc.usesFrameUniforms = true;
    pbrDesc.usesTexture = true; pbrDesc.pbrMaterial = true;
    pbrDesc.pushConstantSize = sizeof(float) * 20;
    auto pbrPipeline = device->CreateGraphicsPipeline(pbrDesc);

    // Static lit pipeline (ground plane).
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Static depth-only shadow pipeline.
    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

    // Sky + post.
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
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
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 1.2f, 0.0f};
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 lightEye = sc - lightDir * 12.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-5.0f, 5.0f, -5.0f, 5.0f, 1.0f, 25.0f));
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u)\n", outPath, cw, ch);
    return 0;
}

// --- Data-driven material-graph showcase (Slice AV). Mirrors the Vulkan --material-shot path: ground
// plane + procedural sky + a sphere shaded by the BUILD-TIME-GENERATED fragment (mat_showcase.frag,
// codegen'd from assets/materials/showcase.mat.json; here its committed HLSL flows through the same
// HLSL->SPIR-V->MSL gen as every other shader). The material pipeline reuses lit.vert + the existing
// PBR material set (no new RHI seam). One offscreen frame -> PNG. ----------------------------------
// `matMslFile` / `matEntry` select WHICH generated material fragment to render: showcase (Slice AV)
// or showcase2 (Slice AW). Metal stays BUILD-TIME for both — the generated MSL is what runs; runtime
// authoring is a Vulkan/Windows feature. Everything else (scene/camera/light) is identical.
static int RunMaterialShowcaseImpl(const char* outPath, const char* matMslFile, const char* matEntry) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Material pipeline (shared lit.vert + the selected generated material fragment; full PBR set).
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto matFs = loadMSL(matMslFile, matEntry);
    rhi::GraphicsPipelineDesc matDesc;
    matDesc.vertex = litVs.get(); matDesc.fragment = matFs.get();
    matDesc.vertexLayout = scene::MeshVertexLayout();
    matDesc.colorFormat = device->Swapchain().ColorFormat();
    matDesc.depthTest = true; matDesc.usesFrameUniforms = true;
    matDesc.usesTexture = true; matDesc.pbrMaterial = true;
    matDesc.pushConstantSize = sizeof(float) * 20;
    auto matPipeline = device->CreateGraphicsPipeline(matDesc);

    // Static lit pipeline (ground plane).
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Static depth-only shadow pipeline.
    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

    // Sky + post.
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto checkerTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    const uint8_t whitePx[4] = {255, 255, 255, 255};
    auto whiteTex = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, whitePx, sizeof(whitePx)});
    const uint8_t blackPx[4] = {0, 0, 0, 255};
    auto blackTex = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, blackPx, sizeof(blackPx)});
    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    const float sphereR = 1.0f;
    Mat4 sphereModel = Mat4::Translate({0.0f, sphereR, 0.0f}) * Mat4::Scale({sphereR, sphereR, sphereR});
    Mat4 groundModel = Mat4::Scale({8.0f, 1.0f, 8.0f});

    const Vec3 eye{2.0f, 1.7f, 2.8f};
    const Vec3 center{0.0f, 0.95f, 0.0f};
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 1.0f, 0.0f};
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 lightEye = sc - lightDir * 12.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-5.0f, 5.0f, -5.0f, 5.0f, 1.0f, 25.0f));
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
            cmd.BindPipeline(*shadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(plane.vertices());
            cmd.BindIndexBuffer(plane.indices());
            cmd.DrawIndexed(plane.indexCount());
            cmd.PushConstants(sphereModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(sphere.vertices());
            cmd.BindIndexBuffer(sphere.indices());
            cmd.DrawIndexed(sphere.indexCount());
            cmd.EndRenderPass();
        });

    graph.AddPass("scene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
            cmd.BindPipeline(*skyPipe);
            cmd.Draw(3);
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
            cmd.BindPipeline(*matPipeline);
            {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = sphereModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.35f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterialPBR(*checkerTex, *whiteTex, *flatNormal, *blackTex, *whiteTex);
                cmd.BindVertexBuffer(sphere.vertices());
                cmd.BindIndexBuffer(sphere.indices());
                cmd.DrawIndexed(sphere.indexCount());
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u)\n", outPath, cw, ch);
    return 0;
}

// Slice AV: the first showcase material. Slice AW: the second (different node mix). Metal stays
// build-time for both; runtime authoring is a Vulkan/Windows feature.
static int RunMaterialShowcase(const char* outPath) {
    return RunMaterialShowcaseImpl(outPath, "mat_showcase.frag.gen.metal", "material_fragment");
}
static int RunMaterialShowcase2(const char* outPath) {
    return RunMaterialShowcaseImpl(outPath, "mat_showcase2.frag.gen.metal", "material2_fragment");
}

// --- Multi-material scene (Slice AZ). Mirrors the Vulkan --material-multi-shot path: three spheres
// in a row, each shaded by a DISTINCT generated graph material (showcase / showcase2 / showcase3 —
// the last exercising the new Swizzle/MakeFloat3/Power/Saturate/OneMinus nodes), + ground + sky,
// lit + shadowed. One draw per material (bind that material's pipeline, draw its sphere). The same
// committed generated HLSL flows through the HLSL->SPIR-V->MSL gen as every other shader. Camera /
// light match the Vulkan path EXACTLY (modulo the Metal NDC +Y flip). One offscreen frame -> PNG.
static int RunMaterialMultiShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Three distinct material pipelines (shared lit.vert + each generated material fragment; PBR set).
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    struct MatSrc { const char* file; const char* entry; };
    const MatSrc matSrc[3] = {
        {"mat_showcase.frag.gen.metal",  "material_fragment"},
        {"mat_showcase2.frag.gen.metal", "material2_fragment"},
        {"mat_showcase3.frag.gen.metal", "material3_fragment"},
    };
    std::vector<std::unique_ptr<rhi::IShaderModule>> matFs;
    std::vector<std::unique_ptr<rhi::IPipeline>> matPipes;
    for (int m = 0; m < 3; ++m) {
        matFs.push_back(loadMSL(matSrc[m].file, matSrc[m].entry));
        rhi::GraphicsPipelineDesc d;
        d.vertex = litVs.get(); d.fragment = matFs.back().get();
        d.vertexLayout = scene::MeshVertexLayout();
        d.colorFormat = device->Swapchain().ColorFormat();
        d.depthTest = true; d.usesFrameUniforms = true;
        d.usesTexture = true; d.pbrMaterial = true;
        d.pushConstantSize = sizeof(float) * 20;
        matPipes.push_back(device->CreateGraphicsPipeline(d));
    }

    // Static lit pipeline (ground plane).
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Static depth-only shadow pipeline.
    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

    // Sky + post.
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto checkerTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    const uint8_t whitePx[4] = {255, 255, 255, 255};
    auto whiteTex = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, whitePx, sizeof(whitePx)});
    const uint8_t blackPx[4] = {0, 0, 0, 255};
    auto blackTex = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, blackPx, sizeof(blackPx)});
    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    const float sphereR = 0.8f;
    const float spacing = 2.0f;
    Mat4 sphereModel[3];
    for (int m = 0; m < 3; ++m) {
        float x = (float)(m - 1) * spacing;
        sphereModel[m] = Mat4::Translate({x, sphereR, 0.0f}) * Mat4::Scale({sphereR, sphereR, sphereR});
    }
    Mat4 groundModel = Mat4::Scale({8.0f, 1.0f, 8.0f});

    const Vec3 eye{0.0f, 1.9f, 4.4f};
    const Vec3 center{0.0f, 0.7f, 0.0f};
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 1.0f, 0.0f};
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 lightEye = sc - lightDir * 12.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-6.0f, 6.0f, -6.0f, 6.0f, 1.0f, 25.0f));
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
            cmd.BindPipeline(*shadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(plane.vertices());
            cmd.BindIndexBuffer(plane.indices());
            cmd.DrawIndexed(plane.indexCount());
            cmd.BindVertexBuffer(sphere.vertices());
            cmd.BindIndexBuffer(sphere.indices());
            for (int m = 0; m < 3; ++m) {
                cmd.PushConstants(sphereModel[m].m, sizeof(float) * 16);
                cmd.DrawIndexed(sphere.indexCount());
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("scene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
            cmd.BindPipeline(*skyPipe);
            cmd.Draw(3);
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
            cmd.BindVertexBuffer(sphere.vertices());
            cmd.BindIndexBuffer(sphere.indices());
            for (int m = 0; m < 3; ++m) {
                cmd.BindPipeline(*matPipes[m]);
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = sphereModel[m].m[k];
                pc[16] = 0.0f; pc[17] = 0.35f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterialPBR(*checkerTex, *whiteTex, *flatNormal, *blackTex, *whiteTex);
                cmd.DrawIndexed(sphere.indexCount());
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u)\n", outPath, cw, ch);
    return 0;
}

// --- Full glTF scene-graph import showcase (Slice V). Mirrors the Vulkan --scene-shot path: ground
// plane + procedural sky + the CesiumMilkTruck imported via LoadGltfScene (node hierarchy walked to
// world transforms, one renderable per primitive of every mesh-referencing node, deduped PBR
// materials), lit + shadowed. The same wheels mesh is drawn at the front and back positions purely
// from the composed node transforms. One offscreen frame -> PNG. Reuses the Slice-P lit-PBR pipeline
// + BindMaterialPBR; no golden-locked pipeline/shader is touched. ----------------------------------
static int RunSceneShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Lit-PBR pipeline (shared lit.vert + lit_pbr.frag; full 5-texture material set).
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto pbrFs = loadMSL("lit_pbr.frag.gen.metal", "pbr_fragment");
    rhi::GraphicsPipelineDesc pbrDesc;
    pbrDesc.vertex = litVs.get(); pbrDesc.fragment = pbrFs.get();
    pbrDesc.vertexLayout = scene::MeshVertexLayout();
    pbrDesc.colorFormat = device->Swapchain().ColorFormat();
    pbrDesc.depthTest = true; pbrDesc.usesFrameUniforms = true;
    pbrDesc.usesTexture = true; pbrDesc.pbrMaterial = true;
    pbrDesc.pushConstantSize = sizeof(float) * 20;
    auto pbrPipeline = device->CreateGraphicsPipeline(pbrDesc);

    // Static lit pipeline (ground plane).
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Static depth-only shadow pipeline.
    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

    // Sky + post.
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh plane = scene::Mesh::Plane(*device);

    // Import the full truck scene (node hierarchy -> instances + deduped materials).
    hf::asset::GltfScene truck = hf::asset::LoadGltfScene(*device, HF_TRUCK_MODEL_PATH);
    std::printf("[scene] imported %zu instances, %zu meshes, %zu materials\n",
                truck.instances.size(), truck.meshStorage.size(), truck.materialStorage.size());

    // The asset's "Yup2Zup" root already lands the truck upright in our Y-up world; rotate about Y
    // to present a 3/4 side view (logo + both wheel sets visible). Then fit the ORIENTED scene to the
    // ground. (Matches the Vulkan --scene-shot path exactly.)
    Mat4 orient = Mat4::RotateY(2.1f);
    float oMin[3] = { 1e30f,  1e30f,  1e30f};
    float oMax[3] = {-1e30f, -1e30f, -1e30f};
    for (int c = 0; c < 8; ++c) {
        float p[3] = {
            (c & 1) ? truck.bbMax[0] : truck.bbMin[0],
            (c & 2) ? truck.bbMax[1] : truck.bbMin[1],
            (c & 4) ? truck.bbMax[2] : truck.bbMin[2],
        };
        float x = orient.m[0]*p[0] + orient.m[4]*p[1] + orient.m[8]*p[2]  + orient.m[12];
        float y = orient.m[1]*p[0] + orient.m[5]*p[1] + orient.m[9]*p[2]  + orient.m[13];
        float z = orient.m[2]*p[0] + orient.m[6]*p[1] + orient.m[10]*p[2] + orient.m[14];
        float wp[3] = {x, y, z};
        for (int k = 0; k < 3; ++k) { if (wp[k] < oMin[k]) oMin[k] = wp[k]; if (wp[k] > oMax[k]) oMax[k] = wp[k]; }
    }
    Mat4 sceneFit;
    {
        float ext[3] = {oMax[0]-oMin[0], oMax[1]-oMin[1], oMax[2]-oMin[2]};
        float maxExt = ext[0]; if (ext[1] > maxExt) maxExt = ext[1]; if (ext[2] > maxExt) maxExt = ext[2];
        float scale = (maxExt > 1e-6f) ? (5.0f / maxExt) : 1.0f;
        float cx = 0.5f * (oMin[0] + oMax[0]);
        float cz = 0.5f * (oMin[2] + oMax[2]);
        sceneFit = Mat4::Translate({-cx * scale, -oMin[1] * scale, -cz * scale})
                 * Mat4::Scale({scale, scale, scale});
    }
    Mat4 placement = sceneFit * orient;
    Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});

    const Vec3 eye{5.0f, 3.2f, 6.0f};
    const Vec3 center{0.0f, 1.0f, 0.0f};
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 1.0f, 0.0f};
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 lightEye = sc - lightDir * 14.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-6.0f, 6.0f, -6.0f, 6.0f, 1.0f, 30.0f));
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
            cmd.BindPipeline(*shadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(plane.vertices());
            cmd.BindIndexBuffer(plane.indices());
            cmd.DrawIndexed(plane.indexCount());
            for (const auto& inst : truck.instances) {
                Mat4 world = placement * inst.worldTransform;
                cmd.PushConstants(world.m, sizeof(float) * 16);
                cmd.BindVertexBuffer(inst.mesh->vertices());
                cmd.BindIndexBuffer(inst.mesh->indices());
                cmd.DrawIndexed(inst.mesh->indexCount());
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("scene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
            cmd.BindPipeline(*skyPipe);
            cmd.Draw(3);
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
            cmd.BindPipeline(*pbrPipeline);
            for (const auto& inst : truck.instances) {
                Mat4 world = placement * inst.worldTransform;
                const hf::asset::PbrMaterial& m = *inst.material;
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = world.m[k];
                pc[16] = m.metallicFactor; pc[17] = m.roughnessFactor;
                pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterialPBR(*m.baseColor, *m.metalRough, *m.normalMap,
                                    *m.emissive, *m.occlusion);
                cmd.BindVertexBuffer(inst.mesh->vertices());
                cmd.BindIndexBuffer(inst.mesh->indices());
                cmd.DrawIndexed(inst.mesh->indexCount());
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u)\n", outPath, cw, ch);
    return 0;
}

// --- HDR environment IBL showcase (Slice R). Mirrors the Vulkan --ibl-shot path: HDR equirect
// skybox (sky_hdr) + ground plane + the DamagedHelmet shaded by lit_pbr_ibl so the metal reflects
// the REAL captured sky/sun/terrain (mip-LOD prefiltered), lit + shadowed. One offscreen frame ->
// PNG. The sky_hdr/lit_pbr_ibl MSL is generated from the shared HLSL by the sibling CMake gen rules;
// the env map binds at flat fragment texture(11)/sampler(12). ----------------------------------------
static int RunIblShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Load the HDR equirect environment (N-mip RGBA16F, CPU box-prefiltered).
    hf::asset::EnvironmentMap env = hf::asset::LoadHdrEnvironment(*device, HF_ENV_PATH);
    const float envMaxLod = (float)(env.mipLevels - 1);

    // Lit-PBR-IBL pipeline (shared lit.vert + lit_pbr_ibl.frag; full 5-texture material set + env).
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto iblFs = loadMSL("lit_pbr_ibl.frag.gen.metal", "pbr_ibl_fragment");
    rhi::GraphicsPipelineDesc iblDesc;
    iblDesc.vertex = litVs.get(); iblDesc.fragment = iblFs.get();
    iblDesc.vertexLayout = scene::MeshVertexLayout();
    iblDesc.colorFormat = device->Swapchain().ColorFormat();
    iblDesc.depthTest = true; iblDesc.usesFrameUniforms = true;
    iblDesc.usesTexture = true; iblDesc.pbrMaterial = true;
    iblDesc.usesEnvironment = true;
    iblDesc.pushConstantSize = sizeof(float) * 20;
    auto iblPipeline = device->CreateGraphicsPipeline(iblDesc);

    // Static lit pipeline (ground plane).
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Static depth-only shadow pipeline.
    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

    // HDR sky (sky_hdr.frag) + post.
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky_hdr.frag.gen.metal", "sky_hdr_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    skyD.usesEnvironment = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
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
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 1.2f, 0.0f};
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 lightEye = sc - lightDir * 12.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-5.0f, 5.0f, -5.0f, 5.0f, 1.0f, 25.0f));
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
            cmd.BindPipeline(*skyPipe);
            cmd.BindEnvironment(*env.equirect);
            cmd.Draw(3);
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — env %dx%d, %d mips\n",
                outPath, cw, ch, env.width, env.height, env.mipLevels);
    return 0;
}

// --- Integrated CAPSTONE showcase (Slice Z). Mirrors the Vulkan --capstone-shot path BYTE-FOR-BYTE
// (same deterministic physics step budget, same Walk+Run fox blend times, same transforms, same
// camera/light). ONE composed frame: HDR equirect environment (sky_hdr) as background + IBL source;
// on the ground sit the imported CesiumMilkTruck (LoadGltfScene, lit_pbr), the GPU-skinned Fox at a
// fixed Walk+Run blend (lit_skinned + BlendAnimations), the full-PBR DamagedHelmet reflecting the
// environment (lit_pbr_ibl) on a pedestal cube, a settled physics sphere pyramid (instanced lit), and
// — in front — sorted translucent glass spheres (transparent pipeline). All opaque geometry casts
// into one directional shadow map; the whole scene renders into an HDR RGBA16F target finished by the
// bloom chain. The integration test: 7 opaque pipelines + a transparent pipeline with distinct
// descriptor sets coexisting in one scene pass. -------------------------------------------------
// Slice AA: optional scripted camera pose. When `active`, the fixed capstone camera is replaced by a
// runtime::Camera at this pose (yaw/pitch radians + world position) — golden-verifying the Camera
// math on Metal. When inactive (the default --capstone path), the byte-for-byte fixed camera is used
// so the capstone golden is unchanged.
struct ScriptedCamera {
    bool active = false;
    float yaw = 0.0f, pitch = 0.0f;
    math::Vec3 position{0, 0, 0};
};

static int RunCapstoneShowcase(const char* outPath, ScriptedCamera scripted = {}) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // === HDR environment (sky + IBL). ===
    hf::asset::EnvironmentMap env = hf::asset::LoadHdrEnvironment(*device, HF_ENV_PATH);
    const float envMaxLod = (float)(env.mipLevels - 1);

    // === Shaders. ===
    auto litVs  = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto litFs  = loadMSL("lit.frag.gen.metal", "fragment_main");
    auto pbrFs  = loadMSL("lit_pbr.frag.gen.metal", "pbr_fragment");
    auto iblFs  = loadMSL("lit_pbr_ibl.frag.gen.metal", "pbr_ibl_fragment");
    auto skVs   = loadMSL("lit_skinned.vert.gen.metal", "skinned_vertex");
    auto instVs = loadMSL("lit_instanced.vert.gen.metal", "instanced_vertex");

    // Static lit (ground + pedestal).
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = kHdr;
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
    litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Full-PBR (truck).
    rhi::GraphicsPipelineDesc pbrDesc;
    pbrDesc.vertex = litVs.get(); pbrDesc.fragment = pbrFs.get();
    pbrDesc.vertexLayout = scene::MeshVertexLayout();
    pbrDesc.colorFormat = kHdr;
    pbrDesc.depthTest = true; pbrDesc.usesFrameUniforms = true; pbrDesc.usesTexture = true;
    pbrDesc.pbrMaterial = true;
    pbrDesc.pushConstantSize = sizeof(float) * 20;
    auto pbrPipeline = device->CreateGraphicsPipeline(pbrDesc);

    // Full-PBR + IBL (helmet).
    rhi::GraphicsPipelineDesc iblDesc;
    iblDesc.vertex = litVs.get(); iblDesc.fragment = iblFs.get();
    iblDesc.vertexLayout = scene::MeshVertexLayout();
    iblDesc.colorFormat = kHdr;
    iblDesc.depthTest = true; iblDesc.usesFrameUniforms = true; iblDesc.usesTexture = true;
    iblDesc.pbrMaterial = true; iblDesc.usesEnvironment = true;
    iblDesc.pushConstantSize = sizeof(float) * 20;
    auto iblPipeline = device->CreateGraphicsPipeline(iblDesc);

    // Skinned lit (fox).
    rhi::GraphicsPipelineDesc skDesc;
    skDesc.vertex = skVs.get(); skDesc.fragment = litFs.get();
    skDesc.vertexLayout = scene::SkinnedMeshVertexLayout();
    skDesc.colorFormat = kHdr;
    skDesc.depthTest = true; skDesc.usesFrameUniforms = true; skDesc.usesTexture = true;
    skDesc.usesJointPalette = true;
    skDesc.pushConstantSize = sizeof(float) * 20;
    auto skinnedPipeline = device->CreateGraphicsPipeline(skDesc);

    // Instanced lit (physics spheres).
    rhi::GraphicsPipelineDesc instDesc;
    instDesc.vertex = instVs.get(); instDesc.fragment = litFs.get();
    instDesc.vertexLayout = scene::MeshVertexLayout();
    instDesc.instanceLayout = scene::InstanceTransformLayout();
    instDesc.colorFormat = kHdr;
    instDesc.depthTest = true; instDesc.usesFrameUniforms = true; instDesc.usesTexture = true;
    instDesc.pushConstantSize = sizeof(float) * 4;
    auto instPipeline = device->CreateGraphicsPipeline(instDesc);

    // Transparent glass.
    auto tVs = loadMSL("transparent.vert.gen.metal", "transparent_vertex");
    auto tFs = loadMSL("transparent.frag.gen.metal", "transparent_fragment");
    rhi::GraphicsPipelineDesc tDesc;
    tDesc.vertex = tVs.get(); tDesc.fragment = tFs.get();
    tDesc.vertexLayout = scene::MeshVertexLayout();
    tDesc.colorFormat = kHdr;
    tDesc.depthTest = true; tDesc.depthWrite = false; tDesc.alphaBlend = true;
    tDesc.cullNone = true; tDesc.usesFrameUniforms = true; tDesc.usesTexture = false;
    tDesc.pushConstantSize = sizeof(float) * 20;
    auto transparentPipeline = device->CreateGraphicsPipeline(tDesc);

    // === Shadow pipelines (static + skinned + instanced; shadow frag is nullptr on Metal). ===
    auto staticShadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    auto skShadowVs     = loadMSL("shadow_skinned.vert.gen.metal", "skinned_shadow_vertex");
    auto instShVs       = loadMSL("shadow_instanced.vert.gen.metal", "instanced_shadow_vertex");

    rhi::GraphicsPipelineDesc stShDesc;
    stShDesc.vertex = staticShadowVs.get(); stShDesc.fragment = nullptr;
    stShDesc.vertexLayout = scene::MeshVertexLayout();
    stShDesc.depthTest = true; stShDesc.depthOnly = true; stShDesc.usesFrameUniforms = true;
    stShDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(stShDesc);

    rhi::GraphicsPipelineDesc skShDesc;
    skShDesc.vertex = skShadowVs.get(); skShDesc.fragment = nullptr;
    skShDesc.vertexLayout = scene::SkinnedMeshVertexLayout();
    skShDesc.depthTest = true; skShDesc.depthOnly = true; skShDesc.usesFrameUniforms = true;
    skShDesc.usesJointPalette = true;
    skShDesc.pushConstantSize = sizeof(float) * 16;
    auto skinnedShadowPipeline = device->CreateGraphicsPipeline(skShDesc);

    rhi::GraphicsPipelineDesc instShDesc;
    instShDesc.vertex = instShVs.get(); instShDesc.fragment = nullptr;
    instShDesc.vertexLayout = scene::MeshVertexLayout();
    instShDesc.instanceLayout = scene::InstanceTransformLayout();
    instShDesc.depthTest = true; instShDesc.depthOnly = true; instShDesc.usesFrameUniforms = true;
    auto instShadowPipeline = device->CreateGraphicsPipeline(instShDesc);

    // === Sky (HDR equirect). ===
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky_hdr.frag.gen.metal", "sky_hdr_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    skyD.usesEnvironment = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    // === Bloom pipelines. ===
    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    struct BloomParams { float texel[2]; float threshold; float knee; float strength; float intensity; };
    const uint32_t kBloomPC = sizeof(BloomParams);
    auto makeBloomPipe = [&](rhi::IShaderModule* fs, rhi::Format colorFmt) {
        rhi::GraphicsPipelineDesc d;
        d.vertex = postVs.get(); d.fragment = fs;
        d.colorFormat = colorFmt;
        d.depthTest = false; d.usesFrameUniforms = false;
        d.usesTexture = true; d.fullscreen = true;
        d.fragmentPushConstants = true; d.pushConstantSize = kBloomPC;
        return device->CreateGraphicsPipeline(d);
    };
    auto prefilterFs  = loadMSL("bloom_prefilter.frag.gen.metal", "bloom_prefilter_fragment");
    auto downsampleFs = loadMSL("bloom_downsample.frag.gen.metal", "bloom_downsample_fragment");
    auto upsampleFs   = loadMSL("bloom_upsample.frag.gen.metal", "bloom_upsample_fragment");
    auto compositeFs  = loadMSL("bloom_composite.frag.gen.metal", "bloom_composite_fragment");
    auto prefilterPipe  = makeBloomPipe(prefilterFs.get(), kHdr);
    auto downsamplePipe = makeBloomPipe(downsampleFs.get(), kHdr);
    auto upsamplePipe   = makeBloomPipe(upsampleFs.get(), kHdr);
    auto compositePipe  = makeBloomPipe(compositeFs.get(), device->Swapchain().ColorFormat());

    // === Render targets. ===
    auto rt = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);
    const int kMips = 5;
    std::vector<std::unique_ptr<rhi::IRenderTarget>> down, up;
    std::vector<uint32_t> mw(kMips), mh(kMips);
    for (int i = 0; i < kMips; ++i) {
        uint32_t dw = std::max(1u, W >> (i + 1));
        uint32_t dh = std::max(1u, H >> (i + 1));
        mw[i] = dw; mh[i] = dh;
        down.push_back(device->CreateRenderTarget(dw, dh, kHdr));
        up.push_back(device->CreateRenderTarget(dw, dh, kHdr));
    }

    // === Meshes + textures. ===
    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh cube = scene::Mesh::Cube(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    // === Imported PBR helmet. ===
    hf::asset::PbrModel helmet = hf::asset::LoadPbrGltfModel(*device, HF_HELMET_MODEL_PATH);

    // === Imported truck scene. ===
    hf::asset::GltfScene truck = hf::asset::LoadGltfScene(*device, HF_TRUCK_MODEL_PATH);

    // === Skinned fox + Walk+Run blended palette (fixed times, deterministic). ===
    hf::asset::SkinnedModel fox = hf::asset::LoadSkinnedGltfModel(*device, HF_FOX_MODEL_PATH);
    std::vector<Mat4> foxPalette;
    {
        const anim::Animation* walk = fox.FindAnimation("Walk");
        const anim::Animation* run  = fox.FindAnimation("Run");
        if (!walk && !fox.animations.empty()) walk = &fox.animations.front();
        if (!run) run = walk;
        if (walk && run)
            foxPalette = anim::BlendAnimations(fox.skeleton, *walk, 0.3f, *run, 0.2f, 0.5f);
        else
            foxPalette.assign(fox.skeleton.joints.size(), Mat4::Identity());
    }
    std::vector<float> paletteData(64 * 16);
    for (int j = 0; j < 64; ++j) {
        Mat4 mm = (j < (int)foxPalette.size()) ? foxPalette[j] : Mat4::Identity();
        for (int k = 0; k < 16; ++k) paletteData[j * 16 + k] = mm.m[k];
    }

    // === Settled physics sphere pyramid (deterministic step budget). ===
    physics::World world;
    {
        const float R = 0.5f;
        const int kLayers = 3;
        const float d = 2.0f * R;
        const float dy = R * 1.41421356f;
        for (int k = 0; k < kLayers; ++k) {
            int m = kLayers - k;
            float off = 0.5f * (float)(m - 1) * d;
            float y = R + (float)k * dy;
            for (int gx = 0; gx < m; ++gx)
                for (int gz = 0; gz < m; ++gz) {
                    float x = (float)gx * d - off;
                    float z = (float)gz * d - off;
                    world.bodies.push_back(physics::MakeDynamicSphere({x, y + 0.01f, z}, R));
                }
        }
    }
    const float dtP = 1.0f / 120.0f;
    for (int s = 0; s < 240; ++s) world.Step(dtP);

    // === Layout transforms (mirror the Vulkan path exactly). ===
    const Mat4 physicsPlace = Mat4::Translate({-5.0f, 0.0f, 2.6f});
    std::vector<scene::InstanceData> instances;
    instances.reserve(world.bodies.size());
    for (const auto& b : world.bodies) {
        Mat4 m = physicsPlace * b.Transform();
        scene::InstanceData di;
        for (int k = 0; k < 16; ++k) di.model[k] = m.m[k];
        instances.push_back(di);
    }
    const uint32_t kInstanceCount = (uint32_t)instances.size();
    rhi::BufferDesc instBufDesc;
    instBufDesc.size = (uint64_t)instances.size() * sizeof(scene::InstanceData);
    instBufDesc.initialData = instances.data();
    instBufDesc.usage = rhi::BufferUsage::Vertex;
    auto instanceBuffer = device->CreateBuffer(instBufDesc);

    Mat4 truckOrient = Mat4::RotateY(2.1f);
    float oMin[3] = { 1e30f,  1e30f,  1e30f};
    float oMax[3] = {-1e30f, -1e30f, -1e30f};
    for (int c = 0; c < 8; ++c) {
        float p[3] = {
            (c & 1) ? truck.bbMax[0] : truck.bbMin[0],
            (c & 2) ? truck.bbMax[1] : truck.bbMin[1],
            (c & 4) ? truck.bbMax[2] : truck.bbMin[2],
        };
        float x = truckOrient.m[0]*p[0] + truckOrient.m[4]*p[1] + truckOrient.m[8]*p[2]  + truckOrient.m[12];
        float y = truckOrient.m[1]*p[0] + truckOrient.m[5]*p[1] + truckOrient.m[9]*p[2]  + truckOrient.m[13];
        float z = truckOrient.m[2]*p[0] + truckOrient.m[6]*p[1] + truckOrient.m[10]*p[2] + truckOrient.m[14];
        float wp[3] = {x, y, z};
        for (int k = 0; k < 3; ++k) {
            if (wp[k] < oMin[k]) oMin[k] = wp[k];
            if (wp[k] > oMax[k]) oMax[k] = wp[k];
        }
    }
    Mat4 truckFit;
    {
        float ext[3] = {oMax[0]-oMin[0], oMax[1]-oMin[1], oMax[2]-oMin[2]};
        float maxExt = ext[0]; if (ext[1] > maxExt) maxExt = ext[1]; if (ext[2] > maxExt) maxExt = ext[2];
        float scale = (maxExt > 1e-6f) ? (5.0f / maxExt) : 1.0f;
        float cx = 0.5f * (oMin[0] + oMax[0]);
        float cz = 0.5f * (oMin[2] + oMax[2]);
        truckFit = Mat4::Translate({-cx * scale, -oMin[1] * scale, -cz * scale})
                 * Mat4::Scale({scale, scale, scale});
    }
    const Mat4 truckPlace = Mat4::Translate({-2.9f, 0.0f, 0.2f}) * truckFit * truckOrient;

    float foxH = fox.bbMax[1] - fox.bbMin[1];
    float foxScale = (foxH > 1e-4f) ? (2.2f / foxH) : 0.05f;
    float fcx = 0.5f * (fox.bbMin[0] + fox.bbMax[0]);
    float fcz = 0.5f * (fox.bbMin[2] + fox.bbMax[2]);
    const Mat4 foxModel = Mat4::Translate({3.4f, 0.0f, 1.6f})
                        * Mat4::RotateY(-0.9f)
                        * Mat4::Translate({-fcx * foxScale, -fox.bbMin[1] * foxScale, -fcz * foxScale})
                        * Mat4::Scale({foxScale, foxScale, foxScale});

    const float pedH = 0.5f;
    const Mat4 pedestalModel = Mat4::Translate({0.0f, pedH, -1.8f}) * Mat4::Scale({1.0f, pedH, 1.0f});
    const float helmetScale = 1.3f;
    const Mat4 helmetModel = Mat4::Translate({0.0f, 2.0f * pedH + helmetScale, -1.8f})
                           * Mat4::RotateX(1.5707963f) * Mat4::Scale({helmetScale, helmetScale, helmetScale});

    const Mat4 groundModel = Mat4::Scale({14.0f, 1.0f, 14.0f});

    struct Glass { Vec3 pos; float scale; float r, g, b, baseAlpha; };
    std::vector<Glass> glass = {
        {{-1.4f, 1.3f, 3.8f}, 1.4f, 0.22f, 0.55f, 0.98f, 0.34f},
        {{ 1.3f, 1.2f, 4.2f}, 1.3f, 0.32f, 0.95f, 0.45f, 0.34f},
    };

    const float aspect = (float)W / (float)H;
    // Default fixed capstone camera. When a scripted pose is supplied (Slice AA --camera), the
    // runtime::Camera below overrides eye/center/fwd/right/up so the SAME scene is framed from an
    // arbitrary pose — golden-verifying the Camera->viewProj path on Metal.
    Vec3 eye{0.0f, 4.0f, 10.0f};
    Vec3 center{0.0f, 1.2f, 0.2f};
    Vec3 camFwd, camRight, camUp;
    {
        Vec3 fwd = math::normalize(center - eye);
        camRight = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
        camUp = math::cross(camRight, fwd);
        camFwd = fwd;
    }
    if (scripted.active) {
        runtime::Camera cam;
        cam.aspect = aspect; cam.yaw = scripted.yaw; cam.SetPitch(scripted.pitch);
        cam.position = scripted.position;
        runtime::CameraBasis b = cam.Basis();
        eye = b.position;
        center = b.position + b.forward;
        camFwd = b.forward; camRight = b.right; camUp = b.up;
    }
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 sc{0.0f, 1.0f, 0.0f};
        Vec3 lightEye = sc - lightDir * 20.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-9.0f, 9.0f, -9.0f, 9.0f, 1.0f, 45.0f));
        Mat4 lightVP = lightOrtho * lightView;
        for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
        fd.camFwd[0]=camFwd.x; fd.camFwd[1]=camFwd.y; fd.camFwd[2]=camFwd.z;
        fd.camRight[0]=camRight.x; fd.camRight[1]=camRight.y; fd.camRight[2]=camRight.z;
        fd.camUp[0]=camUp.x; fd.camUp[1]=camUp.y; fd.camUp[2]=camUp.z;
        fd.skyParams[0] = std::tan(0.5f * 1.04719755f);
        fd.skyParams[1] = aspect;
        fd.skyParams[2] = envMaxLod;
    }

    std::sort(glass.begin(), glass.end(), [&](const Glass& a, const Glass& b) {
        return math::length(a.pos - eye) > math::length(b.pos - eye);
    });

    const float kExposure = 1.7f;
    const float kThreshold = 1.0f;
    const float kKnee = 0.6f;
    const float kUpStrength = 1.0f;
    const float kBloomStrength = 0.14f;
    auto mkPC = [&](uint32_t tw, uint32_t th, float strength) {
        BloomParams p{}; p.texel[0] = 1.0f / (float)tw; p.texel[1] = 1.0f / (float)th;
        p.threshold = kThreshold; p.knee = kKnee; p.strength = strength; p.intensity = kExposure;
        return p;
    };

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    std::vector<render::RgResource> rgDown(kMips), rgUp(kMips);
    for (int i = 0; i < kMips; ++i) {
        rgDown[i] = graph.ImportTarget("down" + std::to_string(i),
                                       render::RgResourceKind::SceneColor, *down[i]);
        rgUp[i]   = graph.ImportTarget("up" + std::to_string(i),
                                       render::RgResourceKind::SceneColor, *up[i]);
    }
    render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

    graph.AddPass("shadow", {}, {rgShadow},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            dev.SetJointPalette(paletteData.data(), paletteData.size() * sizeof(float));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*staticShadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(plane.vertices());
            cmd.BindIndexBuffer(plane.indices());
            cmd.DrawIndexed(plane.indexCount());
            cmd.PushConstants(pedestalModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(cube.vertices());
            cmd.BindIndexBuffer(cube.indices());
            cmd.DrawIndexed(cube.indexCount());
            cmd.PushConstants(helmetModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(helmet.mesh.vertices());
            cmd.BindIndexBuffer(helmet.mesh.indices());
            cmd.DrawIndexed(helmet.mesh.indexCount());
            for (const auto& inst : truck.instances) {
                Mat4 world = truckPlace * inst.worldTransform;
                cmd.PushConstants(world.m, sizeof(float) * 16);
                cmd.BindVertexBuffer(inst.mesh->vertices());
                cmd.BindIndexBuffer(inst.mesh->indices());
                cmd.DrawIndexed(inst.mesh->indexCount());
            }
            cmd.BindPipeline(*skinnedShadowPipeline);
            cmd.PushConstants(foxModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(fox.mesh.vertices());
            cmd.BindIndexBuffer(fox.mesh.indices());
            cmd.DrawIndexed(fox.mesh.indexCount());
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
            dev.SetJointPalette(paletteData.data(), paletteData.size() * sizeof(float));
            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
            cmd.BindPipeline(*skyPipe);
            cmd.BindEnvironment(*env.equirect);
            cmd.Draw(3);
            cmd.BindPipeline(*litPipeline);
            cmd.BindMaterial(*groundTex, *flatNormal);
            {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.85f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = pedestalModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.5f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(cube.vertices());
                cmd.BindIndexBuffer(cube.indices());
                cmd.DrawIndexed(cube.indexCount());
            }
            cmd.BindPipeline(*pbrPipeline);
            for (const auto& inst : truck.instances) {
                Mat4 world = truckPlace * inst.worldTransform;
                const hf::asset::PbrMaterial& m = *inst.material;
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = world.m[k];
                pc[16] = m.metallicFactor; pc[17] = m.roughnessFactor;
                pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterialPBR(*m.baseColor, *m.metalRough, *m.normalMap,
                                    *m.emissive, *m.occlusion);
                cmd.BindVertexBuffer(inst.mesh->vertices());
                cmd.BindIndexBuffer(inst.mesh->indices());
                cmd.DrawIndexed(inst.mesh->indexCount());
            }
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
            cmd.BindPipeline(*instPipeline);
            {
                float material[4] = {0.1f, 0.5f, 0.0f, 0.0f};
                cmd.PushConstants(material, sizeof(material));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(sphere.vertices());
                cmd.BindInstanceBuffer(*instanceBuffer);
                cmd.BindIndexBuffer(sphere.indices());
                cmd.DrawIndexedInstanced(sphere.indexCount(), kInstanceCount);
            }
            cmd.BindPipeline(*transparentPipeline);
            cmd.BindVertexBuffer(sphere.vertices());
            cmd.BindIndexBuffer(sphere.indices());
            for (const auto& g : glass) {
                Mat4 model = Mat4::Translate(g.pos) * Mat4::Scale({g.scale, g.scale, g.scale});
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = model.m[k];
                pc[16] = g.r; pc[17] = g.g; pc[18] = g.b; pc[19] = g.baseAlpha;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.DrawIndexed(sphere.indexCount());
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("prefilter", {rgScene}, {rgDown[0]},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            BloomParams p = mkPC(W, H, kBloomStrength);
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*prefilterPipe);
            cmd.BindTexture(*rt);
            cmd.PushConstants(&p, sizeof(p));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });
    for (int i = 1; i < kMips; ++i) {
        graph.AddPass("down" + std::to_string(i), {rgDown[i - 1]}, {rgDown[i]},
            [&, i](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                BloomParams p = mkPC(mw[i - 1], mh[i - 1], kUpStrength);
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*downsamplePipe);
                cmd.BindTexture(*down[i - 1]);
                cmd.PushConstants(&p, sizeof(p));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });
    }
    graph.AddPass("upTop", {rgDown[kMips - 1]}, {rgUp[kMips - 1]},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            BloomParams p = mkPC(mw[kMips - 1], mh[kMips - 1], 0.0f);
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*upsamplePipe);
            cmd.BindTexturePair(*down[kMips - 1], *down[kMips - 1]);
            cmd.PushConstants(&p, sizeof(p));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });
    for (int i = kMips - 2; i >= 0; --i) {
        graph.AddPass("up" + std::to_string(i), {rgUp[i + 1], rgDown[i]}, {rgUp[i]},
            [&, i](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                BloomParams p = mkPC(mw[i + 1], mh[i + 1], kUpStrength);
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*upsamplePipe);
                cmd.BindTexturePair(*up[i + 1], *down[i]);
                cmd.PushConstants(&p, sizeof(p));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });
    }
    graph.AddPass("composite", {rgScene, rgUp[0]}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            BloomParams p = mkPC(W, H, kBloomStrength);
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*compositePipe);
            cmd.BindTexturePair(*rt, *up[0]);
            cmd.PushConstants(&p, sizeof(p));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — capstone: truck(%zu) + fox + helmet + %u spheres + %zu glass\n",
                outPath, cw, ch, truck.instances.size(), kInstanceCount, glass.size());
    return 0;
}

// --- HDR bloom showcase (Slice U). Mirrors the Vulkan --bloom-shot path: the HDR-IBL helmet scene
// rendered into an HDR (RGBA16F) render target, then a threshold -> 5-mip downsample -> tent
// upsample/combine bloom chain, then composite (add bloom + the post.frag tonemap/grade) to the LDR
// swapchain. The HDR sun + the helmet's emissive cyan gauge bloom; the rest stays sharp. The bloom
// MSL is generated from the shared HLSL by the sibling CMake gen rules; per-pass params bind at the
// fragment push-constant slot. -----------------------------------------------------------------
static int RunBloomShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    hf::asset::EnvironmentMap env = hf::asset::LoadHdrEnvironment(*device, HF_ENV_PATH);
    const float envMaxLod = (float)(env.mipLevels - 1);

    // Scene pipelines, but writing the HDR (RGBA16F) target.
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto iblFs = loadMSL("lit_pbr_ibl.frag.gen.metal", "pbr_ibl_fragment");
    rhi::GraphicsPipelineDesc iblDesc;
    iblDesc.vertex = litVs.get(); iblDesc.fragment = iblFs.get();
    iblDesc.vertexLayout = scene::MeshVertexLayout();
    iblDesc.colorFormat = kHdr;
    iblDesc.depthTest = true; iblDesc.usesFrameUniforms = true;
    iblDesc.usesTexture = true; iblDesc.pbrMaterial = true; iblDesc.usesEnvironment = true;
    iblDesc.pushConstantSize = sizeof(float) * 20;
    auto iblPipeline = device->CreateGraphicsPipeline(iblDesc);

    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = kHdr;
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky_hdr.frag.gen.metal", "sky_hdr_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    skyD.usesEnvironment = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    // Bloom pipelines (fullscreen, fragment push constants).
    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    struct BloomParams { float texel[2]; float threshold; float knee; float strength; float intensity; };
    const uint32_t kBloomPC = sizeof(BloomParams);
    auto makeBloomPipe = [&](rhi::IShaderModule* fs, rhi::Format colorFmt) {
        rhi::GraphicsPipelineDesc d;
        d.vertex = postVs.get(); d.fragment = fs;
        d.colorFormat = colorFmt;
        d.depthTest = false; d.usesFrameUniforms = false;
        d.usesTexture = true; d.fullscreen = true;
        d.fragmentPushConstants = true; d.pushConstantSize = kBloomPC;
        return device->CreateGraphicsPipeline(d);
    };
    auto prefilterFs = loadMSL("bloom_prefilter.frag.gen.metal", "bloom_prefilter_fragment");
    auto downsampleFs = loadMSL("bloom_downsample.frag.gen.metal", "bloom_downsample_fragment");
    auto upsampleFs  = loadMSL("bloom_upsample.frag.gen.metal", "bloom_upsample_fragment");
    auto compositeFs = loadMSL("bloom_composite.frag.gen.metal", "bloom_composite_fragment");
    auto prefilterPipe = makeBloomPipe(prefilterFs.get(), kHdr);
    auto downsamplePipe = makeBloomPipe(downsampleFs.get(), kHdr);
    auto upsamplePipe  = makeBloomPipe(upsampleFs.get(), kHdr);
    auto compositePipe = makeBloomPipe(compositeFs.get(), device->Swapchain().ColorFormat());

    auto rt = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    const int kMips = 5;
    std::vector<std::unique_ptr<rhi::IRenderTarget>> down, up;
    std::vector<uint32_t> mw(kMips), mh(kMips);
    for (int i = 0; i < kMips; ++i) {
        uint32_t dw = std::max(1u, W >> (i + 1));
        uint32_t dh = std::max(1u, H >> (i + 1));
        mw[i] = dw; mh[i] = dh;
        down.push_back(device->CreateRenderTarget(dw, dh, kHdr));
        up.push_back(device->CreateRenderTarget(dw, dh, kHdr));
    }

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
                     * Mat4::RotateX(1.5707963f) * Mat4::Scale({scaleS, scaleS, scaleS});
    Mat4 groundModel = Mat4::Scale({8.0f, 1.0f, 8.0f});

    const Vec3 eye{3.0f, 2.4f, 4.0f};
    const Vec3 center{0.0f, 1.2f, 0.0f};
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 1.2f, 0.0f};
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 lightEye = sc - lightDir * 12.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-5.0f, 5.0f, -5.0f, 5.0f, 1.0f, 25.0f));
        Mat4 lightVP = lightOrtho * lightView;
        for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
        Vec3 fwd = math::normalize(center - eye);
        Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
        Vec3 up3 = math::cross(right, fwd);
        fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
        fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
        fd.camUp[0]=up3.x; fd.camUp[1]=up3.y; fd.camUp[2]=up3.z;
        fd.skyParams[0] = std::tan(0.5f * 1.04719755f);
        fd.skyParams[1] = aspect;
        fd.skyParams[2] = envMaxLod;
    }

    const float kExposure = 1.7f;
    const float kThreshold = 1.0f;
    const float kKnee = 0.6f;
    const float kUpStrength = 1.0f;
    const float kBloomStrength = 0.14f;
    auto mkPC = [&](uint32_t tw, uint32_t th, float strength) {
        BloomParams p{}; p.texel[0] = 1.0f / (float)tw; p.texel[1] = 1.0f / (float)th;
        p.threshold = kThreshold; p.knee = kKnee; p.strength = strength; p.intensity = kExposure;
        return p;
    };

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    std::vector<render::RgResource> rgDown(kMips), rgUp(kMips);
    for (int i = 0; i < kMips; ++i) {
        rgDown[i] = graph.ImportTarget("down" + std::to_string(i),
                                       render::RgResourceKind::SceneColor, *down[i]);
        rgUp[i]   = graph.ImportTarget("up" + std::to_string(i),
                                       render::RgResourceKind::SceneColor, *up[i]);
    }
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
            cmd.BindEnvironment(*env.equirect);
            cmd.Draw(3);
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

    graph.AddPass("prefilter", {rgScene}, {rgDown[0]},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            BloomParams p = mkPC(W, H, kBloomStrength);
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*prefilterPipe);
            cmd.BindTexture(*rt);
            cmd.PushConstants(&p, sizeof(p));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    for (int i = 1; i < kMips; ++i) {
        graph.AddPass("down" + std::to_string(i), {rgDown[i - 1]}, {rgDown[i]},
            [&, i](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                BloomParams p = mkPC(mw[i - 1], mh[i - 1], kUpStrength);
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*downsamplePipe);
                cmd.BindTexture(*down[i - 1]);
                cmd.PushConstants(&p, sizeof(p));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });
    }

    graph.AddPass("upTop", {rgDown[kMips - 1]}, {rgUp[kMips - 1]},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            BloomParams p = mkPC(mw[kMips - 1], mh[kMips - 1], 0.0f);
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*upsamplePipe);
            cmd.BindTexturePair(*down[kMips - 1], *down[kMips - 1]);
            cmd.PushConstants(&p, sizeof(p));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    for (int i = kMips - 2; i >= 0; --i) {
        graph.AddPass("up" + std::to_string(i), {rgUp[i + 1], rgDown[i]}, {rgUp[i]},
            [&, i](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                BloomParams p = mkPC(mw[i + 1], mh[i + 1], kUpStrength);
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*upsamplePipe);
                cmd.BindTexturePair(*up[i + 1], *down[i]);
                cmd.PushConstants(&p, sizeof(p));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });
    }

    graph.AddPass("composite", {rgScene, rgUp[0]}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            BloomParams p = mkPC(W, H, kBloomStrength);
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*compositePipe);
            cmd.BindTexturePair(*rt, *up[0]);
            cmd.PushConstants(&p, sizeof(p));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — HDR bloom, %d mips\n", outPath, cw, ch, kMips);
    return 0;
}

// --- GPU-instanced showcase (Slice Q). Mirrors the Vulkan --instanced-shot path: ground plane +
// procedural sky + a 12x12 = 144 field of spheres drawn in ONE DrawIndexedInstanced, each placed by
// its per-instance model matrix from a second per-instance vertex stream (binding 1), lit + shadowed
// (the field also casts shadows in a single instanced depth-only draw). One offscreen frame -> PNG.
// The instanced MSL is generated from the shared HLSL (lit_instanced.vert / shadow_instanced.vert)
// by the sibling CMake gen rules; the per-instance attributes bind at Metal vertex buffer slot 4. ---
static int RunInstancedShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Instanced lit pipeline (lit_instanced.vert + shared lit.frag; per-instance binding).
    auto instVs = loadMSL("lit_instanced.vert.gen.metal", "instanced_vertex");
    auto litFs  = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc instDesc;
    instDesc.vertex = instVs.get(); instDesc.fragment = litFs.get();
    instDesc.vertexLayout = scene::MeshVertexLayout();
    instDesc.instanceLayout = scene::InstanceTransformLayout();  // binding 1, per-instance
    instDesc.colorFormat = device->Swapchain().ColorFormat();
    instDesc.depthTest = true; instDesc.usesFrameUniforms = true;
    instDesc.usesTexture = true; instDesc.pushConstantSize = sizeof(float) * 4;  // float4 material
    auto instPipeline = device->CreateGraphicsPipeline(instDesc);

    // Static lit pipeline (ground plane).
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Instanced depth-only shadow pipeline (field) + static shadow pipeline (ground).
    auto instShVs = loadMSL("shadow_instanced.vert.gen.metal", "instanced_shadow_vertex");
    rhi::GraphicsPipelineDesc instShDesc;
    instShDesc.vertex = instShVs.get(); instShDesc.fragment = nullptr;
    instShDesc.vertexLayout = scene::MeshVertexLayout();
    instShDesc.instanceLayout = scene::InstanceTransformLayout();
    instShDesc.depthTest = true; instShDesc.depthOnly = true;
    instShDesc.usesFrameUniforms = true; instShDesc.pushConstantSize = 0;
    auto instShadowPipeline = device->CreateGraphicsPipeline(instShDesc);

    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    // Sky + post.
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    // Deterministic 12x12 = 144 instance grid (identical builder to the Vulkan path).
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

    const Vec3 eye{9.5f, 8.5f, 11.0f};
    const Vec3 center{0.0f, 0.6f, 0.0f};
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 0.6f, 0.0f};
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 lightEye = sc - lightDir * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-11.0f, 11.0f, -11.0f, 11.0f, 1.0f, 40.0f));
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
            cmd.BindPipeline(*staticShadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(plane.vertices());
            cmd.BindIndexBuffer(plane.indices());
            cmd.DrawIndexed(plane.indexCount());
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
            cmd.BindPipeline(*instPipeline);
            {
                float material[4] = {0.1f, 0.5f, 0.0f, 0.0f};
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — %u instances\n", outPath, cw, ch, kInstanceCount);
    return 0;
}

// --- Physics showcase (Slice S). Mirrors the Vulkan --physics-shot path: build a physics::World
// (static ground plane y=0 + a 4-layer SQUARE-PYRAMID packing of 30 unit spheres nestled into each
// other's pockets), STEP it a FIXED 240 times @ dt=1/120 so it settles into a stable pile, then
// upload ONE instance transform per resting body and render the pile via the EXISTING instanced lit
// + instanced shadow pipelines over the ground plane + skybox. One offscreen frame -> PNG. The
// physics core is pure C++ (hf_core, shared with the Vulkan build); only the SCENE differs. --------
static int RunPhysicsShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Build + settle the pyramid (identical scenario + step budget to the Vulkan path).
    physics::World world;
    const float R = 0.5f;
    const int kLayers = 4;
    const float d = 2.0f * R;
    const float dy = R * 1.41421356f;
    for (int k = 0; k < kLayers; ++k) {
        int m = kLayers - k;
        float off = 0.5f * (float)(m - 1) * d;
        float y = R + (float)k * dy;
        for (int gx = 0; gx < m; ++gx)
            for (int gz = 0; gz < m; ++gz) {
                float x = (float)gx * d - off;
                float z = (float)gz * d - off;
                world.bodies.push_back(physics::MakeDynamicSphere({x, y + 0.01f, z}, R));
            }
    }
    const float dt = 1.0f / 120.0f;
    for (int s = 0; s < 240; ++s) world.Step(dt);

    std::vector<scene::InstanceData> instances;
    instances.reserve(world.bodies.size());
    for (const auto& b : world.bodies) {
        Mat4 m = b.Transform();
        scene::InstanceData inst;
        for (int k = 0; k < 16; ++k) inst.model[k] = m.m[k];
        instances.push_back(inst);
    }
    const uint32_t kInstanceCount = (uint32_t)instances.size();

    auto instVs = loadMSL("lit_instanced.vert.gen.metal", "instanced_vertex");
    auto litFs  = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc instDesc;
    instDesc.vertex = instVs.get(); instDesc.fragment = litFs.get();
    instDesc.vertexLayout = scene::MeshVertexLayout();
    instDesc.instanceLayout = scene::InstanceTransformLayout();
    instDesc.colorFormat = device->Swapchain().ColorFormat();
    instDesc.depthTest = true; instDesc.usesFrameUniforms = true;
    instDesc.usesTexture = true; instDesc.pushConstantSize = sizeof(float) * 4;
    auto instPipeline = device->CreateGraphicsPipeline(instDesc);

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    auto instShVs = loadMSL("shadow_instanced.vert.gen.metal", "instanced_shadow_vertex");
    rhi::GraphicsPipelineDesc instShDesc;
    instShDesc.vertex = instShVs.get(); instShDesc.fragment = nullptr;
    instShDesc.vertexLayout = scene::MeshVertexLayout();
    instShDesc.instanceLayout = scene::InstanceTransformLayout();
    instShDesc.depthTest = true; instShDesc.depthOnly = true;
    instShDesc.usesFrameUniforms = true; instShDesc.pushConstantSize = 0;
    auto instShadowPipeline = device->CreateGraphicsPipeline(instShDesc);

    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    rhi::BufferDesc instBufDesc;
    instBufDesc.size = (uint64_t)instances.size() * sizeof(scene::InstanceData);
    instBufDesc.initialData = instances.data();
    instBufDesc.usage = rhi::BufferUsage::Vertex;
    auto instanceBuffer = device->CreateBuffer(instBufDesc);

    Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});

    const Vec3 eye{6.5f, 4.5f, 7.0f};
    const Vec3 center{0.0f, 1.0f, 0.0f};
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 1.0f, 0.0f};
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 lightEye = sc - lightDir * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 40.0f));
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
            cmd.BindPipeline(*staticShadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(plane.vertices());
            cmd.BindIndexBuffer(plane.indices());
            cmd.DrawIndexed(plane.indexCount());
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
            cmd.BindPipeline(*instPipeline);
            {
                float material[4] = {0.1f, 0.5f, 0.0f, 0.0f};
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — %u rigid bodies settled\n", outPath, cw, ch, kInstanceCount);
    return 0;
}

// --- Playable game sample (Slice AX). Mirrors the Vulkan --game-shot path EXACTLY: build the
// deterministic roll-a-ball game (game::MakeRollGame: ground + dynamic player sphere + 3 fixed
// pickups), run game::StepGame over the FULL game::ScriptedTrack() at the engine fixed dt (the
// winning playthrough — score 3, won), then render ONE frame at the FIXED mid-track capture step
// (kGameCaptureStep == 250, identical to the Vulkan side) — the ground + the player sphere + the
// remaining (uncollected) pickups, lit + shadowed via the existing static-lit scene path with
// distinct solid-color tints. Identical scene/camera/light/colors to the Vulkan path so the only
// difference vs the BMP-golden is backend-NDC handling. One offscreen frame -> PNG. The gameplay
// (engine/game/roll_game) is pure C++ (hf_core), shared byte-for-byte with the Vulkan build. -------
static int RunGameShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Identical capture step + track to the Vulkan --game-shot.
    const int kGameCaptureStep = 250;
    const float dtG = 1.0f / 120.0f;
    std::vector<game::GameInput> track = game::ScriptedTrack();

    physics::World capWorld;
    game::GameState capState = game::MakeRollGame(capWorld);
    for (int s = 0; s < kGameCaptureStep && s < (int)track.size(); ++s)
        game::StepGame(capWorld, capState, track[s], dtG);
    const physics::RigidBody& capPlayer = capWorld.bodies[(size_t)capState.playerBodyIndex];

    physics::World finWorld;
    game::GameState finState = game::MakeRollGame(finWorld);
    for (const auto& in : track) game::StepGame(finWorld, finState, in, dtG);
    const physics::RigidBody& finPlayer = finWorld.bodies[(size_t)finState.playerBodyIndex];

    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    const uint8_t playerPx[4] = {40, 110, 230, 255};   // blue
    auto playerTex = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, playerPx, sizeof(playerPx)});
    const uint8_t pickupPx[4] = {245, 200, 40, 255};   // gold
    auto pickupTex = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, pickupPx, sizeof(pickupPx)});

    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});
    Mat4 playerModel = capPlayer.Transform();

    std::vector<Mat4> pickupModels;
    for (const auto& p : capState.pickups) {
        if (p.collected) continue;
        pickupModels.push_back(math::FromTRS(p.pos, math::Quat::Identity(),
                                             {2.0f * p.radius, 2.0f * p.radius, 2.0f * p.radius}));
    }

    const Vec3 eye{9.5f, 4.0f, 6.5f};
    const Vec3 center{5.0f, 0.4f, 1.6f};
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{5.0f, 0.4f, 1.6f};
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 lightEye = sc - lightDir * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 40.0f));
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

    auto litPush = [](const Mat4& model, float metallic, float roughness, float* pc) {
        for (int k = 0; k < 16; ++k) pc[k] = model.m[k];
        pc[16] = metallic; pc[17] = roughness; pc[18] = 0.0f; pc[19] = 0.0f;
    };

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
            cmd.BindPipeline(*staticShadowPipeline);
            cmd.BindVertexBuffer(sphere.vertices());
            cmd.BindIndexBuffer(sphere.indices());
            cmd.PushConstants(playerModel.m, sizeof(float) * 16);
            cmd.DrawIndexed(sphere.indexCount());
            for (const Mat4& pm : pickupModels) {
                cmd.PushConstants(pm.m, sizeof(float) * 16);
                cmd.DrawIndexed(sphere.indexCount());
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("scene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
            cmd.BindPipeline(*skyPipe);
            cmd.Draw(3);
            cmd.BindPipeline(*litPipeline);
            {
                float pc[20]; litPush(groundModel, 0.0f, 0.85f, pc);
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            {
                float pc[20]; litPush(playerModel, 0.1f, 0.4f, pc);
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*playerTex, *flatNormal);
                cmd.BindVertexBuffer(sphere.vertices());
                cmd.BindIndexBuffer(sphere.indices());
                cmd.DrawIndexed(sphere.indexCount());
            }
            {
                cmd.BindMaterial(*pickupTex, *flatNormal);
                cmd.BindVertexBuffer(sphere.vertices());
                cmd.BindIndexBuffer(sphere.indices());
                for (const Mat4& pm : pickupModels) {
                    float pc[20]; litPush(pm, 0.3f, 0.3f, pc);
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.DrawIndexed(sphere.indexCount());
                }
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
    graph.Execute(*device);

    std::printf("game: {score:%d, won:%s, steps:%d, player:[%.3f, %.3f, %.3f]}\n",
                finState.score, finState.won ? "true" : "false", finState.step,
                finPlayer.position.x, finPlayer.position.y, finPlayer.position.z);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — capture step %d, %zu pickups remaining\n",
                outPath, cw, ch, kGameCaptureStep, pickupModels.size());
    return 0;
}

// --- Text / HUD showcase (Slice BA). Mirrors the Vulkan --hud-shot / --game-hud-shot paths: the
// deterministic roll-a-ball scene at the fixed mid-track capture step (identical recipe to
// RunGameShowcase), lit + shadowed, PLUS a screen-space HUD text overlay drawn OVER the tonemapped
// scene in the post pass through the new text pipeline (baked 8x8 font atlas + alpha-blended quad
// batch). gameMode=true draws just the live "SCORE: N" (own golden game_hud.png; game.png unchanged);
// gameMode=false draws the full HUD ("HAZARD FORGE" + "SCORE: 0" + a fixed stat line -> hud.png).
// The text layout (engine/ui/text.cpp LayoutText) emits NDC with the +Y-DOWN (Vulkan/clip) convention,
// so on Metal — whose clip-space Y is UP — the quad Y is flipped here so the HUD reads upright, exactly
// as FlipProjY flips the 3D projection for parity. Fixed text/positions/scale/color => deterministic. -
struct MtlHudLine { std::string text; float x, y, scale; float color[4]; };

static int RunHudShowcase(const char* outPath, bool gameMode) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    const int kGameCaptureStep = 250;
    const float dtG = 1.0f / 120.0f;
    std::vector<game::GameInput> track = game::ScriptedTrack();
    physics::World capWorld;
    game::GameState capState = game::MakeRollGame(capWorld);
    for (int s = 0; s < kGameCaptureStep && s < (int)track.size(); ++s)
        game::StepGame(capWorld, capState, track[s], dtG);
    const physics::RigidBody& capPlayer = capWorld.bodies[(size_t)capState.playerBodyIndex];

    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    // --- Text / HUD pipeline (alphaBlend + cullNone, no depth, no frame uniforms, frag push color). ---
    auto textVs = loadMSL("text.vert.gen.metal", "text_vertex");
    auto textFs = loadMSL("text.frag.gen.metal", "text_fragment");
    rhi::GraphicsPipelineDesc textD;
    textD.vertex = textVs.get(); textD.fragment = textFs.get();
    textD.vertexLayout.stride = sizeof(ui::TextVertex);
    textD.vertexLayout.attributes = {
        {0, rhi::Format::RG32_Float, (uint32_t)offsetof(ui::TextVertex, posPx)},
        {1, rhi::Format::RG32_Float, (uint32_t)offsetof(ui::TextVertex, uv)},
    };
    textD.colorFormat = device->Swapchain().ColorFormat();
    textD.depthTest = false; textD.usesFrameUniforms = false; textD.usesTexture = true;
    textD.alphaBlend = true; textD.cullNone = true;
    textD.pushConstantSize = sizeof(float) * 4; textD.fragmentPushConstants = true;
    auto textPipe = device->CreateGraphicsPipeline(textD);

    // Baked-font atlas -> sampled texture.
    std::vector<uint8_t> atlasPx((size_t)ui::kAtlasW * ui::kAtlasH * 4);
    ui::BuildFontAtlas(atlasPx.data(), ui::kAtlasW, ui::kAtlasH);
    auto atlasTex = device->CreateTexture(
        {(uint32_t)ui::kAtlasW, (uint32_t)ui::kAtlasH, rhi::Format::RGBA8_UNorm,
         atlasPx.data(), atlasPx.size()});

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    const uint8_t playerPx[4] = {40, 110, 230, 255};
    auto playerTex = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, playerPx, sizeof(playerPx)});
    const uint8_t pickupPx[4] = {245, 200, 40, 255};
    auto pickupTex = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, pickupPx, sizeof(pickupPx)});

    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});
    Mat4 playerModel = capPlayer.Transform();
    std::vector<Mat4> pickupModels;
    for (const auto& p : capState.pickups) {
        if (p.collected) continue;
        pickupModels.push_back(math::FromTRS(p.pos, math::Quat::Identity(),
                                             {2.0f * p.radius, 2.0f * p.radius, 2.0f * p.radius}));
    }

    const Vec3 eye{9.5f, 4.0f, 6.5f};
    const Vec3 center{5.0f, 0.4f, 1.6f};
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{5.0f, 0.4f, 1.6f};
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 lightEye = sc - lightDir * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 40.0f));
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

    auto litPush = [](const Mat4& model, float metallic, float roughness, float* pc) {
        for (int k = 0; k < 16; ++k) pc[k] = model.m[k];
        pc[16] = metallic; pc[17] = roughness; pc[18] = 0.0f; pc[19] = 0.0f;
    };

    // --- Deterministic HUD lines (same content/positions/scale/color as the Vulkan path). ---
    std::vector<MtlHudLine> hudLines;
    char scoreBuf[32];
    std::snprintf(scoreBuf, sizeof(scoreBuf), "SCORE: %d", capState.score);
    if (gameMode) {
        hudLines.push_back({scoreBuf, 32.0f, 32.0f, 4.0f, {1.0f, 1.0f, 1.0f, 1.0f}});
    } else {
        hudLines.push_back({"HAZARD FORGE", 32.0f, 28.0f, 5.0f, {1.0f, 0.85f, 0.2f, 1.0f}});
        hudLines.push_back({"SCORE: 0",     32.0f, 76.0f, 4.0f, {1.0f, 1.0f, 1.0f, 1.0f}});
        hudLines.push_back({"PICKUPS: 3  LEVEL: 1", 32.0f, 112.0f, 3.0f, {0.6f, 0.9f, 1.0f, 1.0f}});
    }
    // Build each line's quad batch ONCE (Metal NDC Y is up -> flip the layout's +Y-down NDC).
    struct HudBatch { std::vector<ui::TextVertex> verts; const float* color; };
    std::vector<HudBatch> hudBatches;
    for (const auto& ln : hudLines) {
        std::vector<ui::TextVertex> verts;
        int q = ui::LayoutText(ln.text, ln.x, ln.y, ln.scale, (int)W, (int)H, verts);
        if (q == 0) continue;
        for (auto& v : verts) v.posPx[1] = -v.posPx[1];  // Vulkan-clip(+Y down) -> Metal-clip(+Y up)
        hudBatches.push_back({std::move(verts), ln.color});
    }
    std::vector<std::unique_ptr<rhi::IBuffer>> hudVbs;
    for (auto& b : hudBatches) {
        rhi::BufferDesc bd;
        bd.size = b.verts.size() * sizeof(ui::TextVertex);
        bd.initialData = b.verts.data(); bd.usage = rhi::BufferUsage::Vertex;
        hudVbs.push_back(device->CreateBuffer(bd));
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
            cmd.BindPipeline(*staticShadowPipeline);
            cmd.BindVertexBuffer(sphere.vertices());
            cmd.BindIndexBuffer(sphere.indices());
            cmd.PushConstants(playerModel.m, sizeof(float) * 16);
            cmd.DrawIndexed(sphere.indexCount());
            for (const Mat4& pm : pickupModels) {
                cmd.PushConstants(pm.m, sizeof(float) * 16);
                cmd.DrawIndexed(sphere.indexCount());
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("scene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
            cmd.BindPipeline(*skyPipe);
            cmd.Draw(3);
            cmd.BindPipeline(*litPipeline);
            {
                float pc[20]; litPush(groundModel, 0.0f, 0.85f, pc);
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            {
                float pc[20]; litPush(playerModel, 0.1f, 0.4f, pc);
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*playerTex, *flatNormal);
                cmd.BindVertexBuffer(sphere.vertices());
                cmd.BindIndexBuffer(sphere.indices());
                cmd.DrawIndexed(sphere.indexCount());
            }
            {
                cmd.BindMaterial(*pickupTex, *flatNormal);
                cmd.BindVertexBuffer(sphere.vertices());
                cmd.BindIndexBuffer(sphere.indices());
                for (const Mat4& pm : pickupModels) {
                    float pc[20]; litPush(pm, 0.3f, 0.3f, pc);
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.DrawIndexed(sphere.indexCount());
                }
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("post", {rgScene}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*postPipe);
            cmd.BindTexture(*rt);
            cmd.Draw(3);
            // HUD overlay: alpha-blended text quads over the tonemapped scene, same final pass.
            cmd.BindPipeline(*textPipe);
            cmd.BindTexture(*atlasTex);
            for (size_t i = 0; i < hudBatches.size(); ++i) {
                cmd.PushConstants(hudBatches[i].color, sizeof(float) * 4);
                cmd.BindVertexBuffer(*hudVbs[i]);
                cmd.Draw((uint32_t)hudBatches[i].verts.size());
            }
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — %s HUD, score %d\n",
                outPath, cw, ch, gameMode ? "game" : "standard", capState.score);
    return 0;
}

// --- SSAO showcase (Slice Y). Mirrors the Vulkan --ssao-shot path: the SAME settled physics
// sphere-pyramid scene rendered into an HDR (RGBA16F) target, PLUS classic screen-space ambient
// occlusion — a view-space normal+linear-depth g-buffer prepass, a 16-sample baked-hemisphere-kernel
// AO pass, a box blur, and a composite that MULTIPLIES the lit scene by the blurred AO (then the
// usual exposure/ACES/grade/vignette). aoOn=true applies AO; aoOn=false renders the same scene with
// AO forced off through the IDENTICAL composite. SEPARATE gbuffer/ssao/blur/composite pipelines +
// shaders; existing pipelines/shaders/goldens untouched. -----------------------------------------
static int RunSsaoShowcase(const char* outPath, bool aoOn = true) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Build + settle the pyramid (identical scenario + step budget to the Vulkan path).
    physics::World world;
    const float R = 0.5f;
    const int kLayers = 4;
    const float d = 2.0f * R;
    const float dy = R * 1.41421356f;
    for (int k = 0; k < kLayers; ++k) {
        int m = kLayers - k;
        float off = 0.5f * (float)(m - 1) * d;
        float y = R + (float)k * dy;
        for (int gx = 0; gx < m; ++gx)
            for (int gz = 0; gz < m; ++gz) {
                float x = (float)gx * d - off;
                float z = (float)gz * d - off;
                world.bodies.push_back(physics::MakeDynamicSphere({x, y + 0.01f, z}, R));
            }
    }
    const float dt = 1.0f / 120.0f;
    for (int s = 0; s < 240; ++s) world.Step(dt);

    std::vector<scene::InstanceData> instances;
    instances.reserve(world.bodies.size());
    for (const auto& b : world.bodies) {
        Mat4 m = b.Transform();
        scene::InstanceData inst;
        for (int k = 0; k < 16; ++k) inst.model[k] = m.m[k];
        instances.push_back(inst);
    }
    const uint32_t kInstanceCount = (uint32_t)instances.size();

    // Lit scene pipelines (writing the HDR RT) — UNCHANGED lit shaders.
    auto instVs = loadMSL("lit_instanced.vert.gen.metal", "instanced_vertex");
    auto litFs  = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc instDesc;
    instDesc.vertex = instVs.get(); instDesc.fragment = litFs.get();
    instDesc.vertexLayout = scene::MeshVertexLayout();
    instDesc.instanceLayout = scene::InstanceTransformLayout();
    instDesc.colorFormat = kHdr;
    instDesc.depthTest = true; instDesc.usesFrameUniforms = true;
    instDesc.usesTexture = true; instDesc.pushConstantSize = sizeof(float) * 4;
    auto instPipeline = device->CreateGraphicsPipeline(instDesc);

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = kHdr;
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Shadow pipelines (UNCHANGED).
    auto instShVs = loadMSL("shadow_instanced.vert.gen.metal", "instanced_shadow_vertex");
    rhi::GraphicsPipelineDesc instShDesc;
    instShDesc.vertex = instShVs.get(); instShDesc.fragment = nullptr;
    instShDesc.vertexLayout = scene::MeshVertexLayout();
    instShDesc.instanceLayout = scene::InstanceTransformLayout();
    instShDesc.depthTest = true; instShDesc.depthOnly = true;
    instShDesc.usesFrameUniforms = true; instShDesc.pushConstantSize = 0;
    auto instShadowPipeline = device->CreateGraphicsPipeline(instShDesc);

    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    // Sky (writing HDR RT) — UNCHANGED.
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    // NEW: g-buffer prepass pipelines (static + instanced), RGBA16F view-space normal + linear depth.
    auto gbVs   = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbInVs = loadMSL("gbuffer_instanced.vert.gen.metal", "gbuffer_instanced_vertex");
    auto gbFs   = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;   // model(16) + view(16)
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    rhi::GraphicsPipelineDesc gbInDesc;
    gbInDesc.vertex = gbInVs.get(); gbInDesc.fragment = gbFs.get();
    gbInDesc.vertexLayout = scene::MeshVertexLayout();
    gbInDesc.instanceLayout = scene::InstanceTransformLayout();
    gbInDesc.colorFormat = kHdr;
    gbInDesc.depthTest = true; gbInDesc.usesFrameUniforms = true;
    gbInDesc.pushConstantSize = sizeof(float) * 16;   // view(16)
    auto gbInstPipeline = device->CreateGraphicsPipeline(gbInDesc);

    // NEW: SSAO + blur + composite fullscreen pipelines (fragment push constants).
    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    struct SsaoParams { float texel[2]; float radius, bias, intensity, tanHalfFovY, aspect, pad; };
    struct BlurParams { float texel[2]; float pad[2]; };
    struct SsaoCompParams { float texel[2]; float aoStrength, intensity; };

    auto ssaoFs = loadMSL("ssao.frag.gen.metal", "ssao_fragment");
    auto blurFs = loadMSL("ssao_blur.frag.gen.metal", "ssao_blur_fragment");
    auto compFs = loadMSL("ssao_composite.frag.gen.metal", "ssao_composite_fragment");

    rhi::GraphicsPipelineDesc ssaoD;
    ssaoD.vertex = postVs.get(); ssaoD.fragment = ssaoFs.get();
    ssaoD.colorFormat = kHdr;
    ssaoD.depthTest = false; ssaoD.usesTexture = true; ssaoD.fullscreen = true;
    ssaoD.fragmentPushConstants = true; ssaoD.pushConstantSize = sizeof(SsaoParams);
    auto ssaoPipe = device->CreateGraphicsPipeline(ssaoD);

    rhi::GraphicsPipelineDesc blurD;
    blurD.vertex = postVs.get(); blurD.fragment = blurFs.get();
    blurD.colorFormat = kHdr;
    blurD.depthTest = false; blurD.usesTexture = true; blurD.fullscreen = true;
    blurD.fragmentPushConstants = true; blurD.pushConstantSize = sizeof(BlurParams);
    auto blurPipe = device->CreateGraphicsPipeline(blurD);

    rhi::GraphicsPipelineDesc compD;
    compD.vertex = postVs.get(); compD.fragment = compFs.get();
    compD.colorFormat = device->Swapchain().ColorFormat();
    compD.depthTest = false; compD.usesTexture = true; compD.fullscreen = true;
    compD.fragmentPushConstants = true; compD.pushConstantSize = sizeof(SsaoCompParams);
    auto compPipe = device->CreateGraphicsPipeline(compD);

    auto rt       = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf     = device->CreateRenderTarget(W, H, kHdr);
    auto aoRT     = device->CreateRenderTarget(W, H, kHdr);
    auto aoBlurRT = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    rhi::BufferDesc instBufDesc;
    instBufDesc.size = (uint64_t)instances.size() * sizeof(scene::InstanceData);
    instBufDesc.initialData = instances.data();
    instBufDesc.usage = rhi::BufferUsage::Vertex;
    auto instanceBuffer = device->CreateBuffer(instBufDesc);

    Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});

    const Vec3 eye{6.5f, 4.5f, 7.0f};
    const Vec3 center{0.0f, 1.0f, 0.0f};
    const float aspect = (float)W / (float)H;
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    FrameData fd{};
    {
        Mat4 proj = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * viewM;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 sc{0.0f, 1.0f, 0.0f};
        Vec3 lightEye = sc - lightDir * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 40.0f));
        Mat4 lightVP = lightOrtho * lightView;
        for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
        Vec3 fwd = math::normalize(center - eye);
        Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
        Vec3 up = math::cross(right, fwd);
        fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
        fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
        fd.camUp[0]=up.x; fd.camUp[1]=up.y; fd.camUp[2]=up.z;
        fd.skyParams[0] = std::tan(0.5f * kFovY);
        fd.skyParams[1] = aspect;
    }

    SsaoParams sp{};
    sp.texel[0] = 1.0f / (float)W; sp.texel[1] = 1.0f / (float)H;
    sp.radius = 0.30f; sp.bias = 0.025f; sp.intensity = 1.6f;
    sp.tanHalfFovY = std::tan(0.5f * kFovY); sp.aspect = aspect; sp.pad = 0.0f;
    BlurParams blurP{}; blurP.texel[0] = 1.0f / (float)W; blurP.texel[1] = 1.0f / (float)H;
    SsaoCompParams cp{}; cp.texel[0] = 1.0f / (float)W; cp.texel[1] = 1.0f / (float)H;
    cp.aoStrength = aoOn ? 1.0f : 0.0f; cp.intensity = 1.7f;

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgGbuf = graph.ImportTarget(
        "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
    render::RgResource rgAO = graph.ImportTarget(
        "ao", render::RgResourceKind::SceneColor, *aoRT);
    render::RgResource rgAOBlur = graph.ImportTarget(
        "aoBlur", render::RgResourceKind::SceneColor, *aoBlurRT);
    render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

    graph.AddPass("shadow", {}, {rgShadow},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*staticShadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(plane.vertices());
            cmd.BindIndexBuffer(plane.indices());
            cmd.DrawIndexed(plane.indexCount());
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
            cmd.BindPipeline(*instPipeline);
            {
                float material[4] = {0.1f, 0.5f, 0.0f, 0.0f};
                cmd.PushConstants(material, sizeof(material));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(sphere.vertices());
                cmd.BindInstanceBuffer(*instanceBuffer);
                cmd.BindIndexBuffer(sphere.indices());
                cmd.DrawIndexedInstanced(sphere.indexCount(), kInstanceCount);
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("gbuffer", {}, {rgGbuf},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
            cmd.BindPipeline(*gbStaticPipeline);
            {
                float pc[32];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                for (int k = 0; k < 16; ++k) pc[16 + k] = viewM.m[k];
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            cmd.BindPipeline(*gbInstPipeline);
            {
                cmd.PushConstants(viewM.m, sizeof(float) * 16);
                cmd.BindVertexBuffer(sphere.vertices());
                cmd.BindInstanceBuffer(*instanceBuffer);
                cmd.BindIndexBuffer(sphere.indices());
                cmd.DrawIndexedInstanced(sphere.indexCount(), kInstanceCount);
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("ssao", {rgGbuf}, {rgAO},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{1, 1, 1, 1});
            cmd.BindPipeline(*ssaoPipe);
            cmd.BindTexture(*gbuf);
            cmd.PushConstants(&sp, sizeof(sp));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    graph.AddPass("ssaoBlur", {rgAO}, {rgAOBlur},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{1, 1, 1, 1});
            cmd.BindPipeline(*blurPipe);
            cmd.BindTexture(*aoRT);
            cmd.PushConstants(&blurP, sizeof(blurP));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    graph.AddPass("composite", {rgScene, rgAOBlur}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*compPipe);
            cmd.BindTexturePair(*rt, *aoBlurRT);
            cmd.PushConstants(&cp, sizeof(cp));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — SSAO %s, %u bodies\n",
                outPath, cw, ch, aoOn ? "ON" : "OFF", kInstanceCount);
    return 0;
}

// --- Temporal anti-aliasing showcase (Slice AP). Mirrors the Vulkan --taa-shot path byte-for-byte
// in INTENT: the SAME settled sphere-pyramid scene + camera + lights, rendered as a FIXED N=8
// accumulation loop. Each accumulation frame jitters the PROJECTION by the SAME deterministic
// Halton(2,3) sub-pixel offset (render::taa::Jitter) the Vulkan path uses, renders the lit + shadowed
// scene into an HDR RT, then taa_resolve blends it into a neighborhood-clamped history (ping-ponged
// between two RGBA16F textures: first frame unblended, then render::taa::kSteadyAlpha of the new
// frame). The 8th resolved frame is tonemapped through post.frag and captured. Static scene/camera +
// deterministic jitter => two runs DIFF 0.0000. New golden tests/golden/metal/taa.png; existing
// pipelines/shaders/goldens untouched. -----------------------------------------------------------
static int RunTaaShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace taa = render::taa;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Static scene: a settled 4-layer instanced sphere pyramid (identical recipe to the Vulkan path).
    physics::World world;
    {
        const float R = 0.5f;
        const int kLayers = 4;
        const float d = 2.0f * R;
        const float dy = R * 1.41421356f;
        for (int k = 0; k < kLayers; ++k) {
            int m = kLayers - k;
            float off = 0.5f * (float)(m - 1) * d;
            float y = R + (float)k * dy;
            for (int gx = 0; gx < m; ++gx)
                for (int gz = 0; gz < m; ++gz) {
                    float x = (float)gx * d - off;
                    float z = (float)gz * d - off;
                    world.bodies.push_back(physics::MakeDynamicSphere({x, y + 0.01f, z}, R));
                }
        }
    }
    for (int s = 0; s < 240; ++s) world.Step(1.0f / 120.0f);
    std::vector<scene::InstanceData> instances;
    instances.reserve(world.bodies.size());
    for (const auto& b : world.bodies) {
        Mat4 m = b.Transform();
        scene::InstanceData inst;
        for (int k = 0; k < 16; ++k) inst.model[k] = m.m[k];
        instances.push_back(inst);
    }
    const uint32_t kInstanceCount = (uint32_t)instances.size();

    // Lit pipelines (HDR RT) — UNCHANGED lit/instanced shaders.
    auto instVs = loadMSL("lit_instanced.vert.gen.metal", "instanced_vertex");
    auto litFs  = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc instDesc;
    instDesc.vertex = instVs.get(); instDesc.fragment = litFs.get();
    instDesc.vertexLayout = scene::MeshVertexLayout();
    instDesc.instanceLayout = scene::InstanceTransformLayout();
    instDesc.colorFormat = kHdr;
    instDesc.depthTest = true; instDesc.usesFrameUniforms = true;
    instDesc.usesTexture = true; instDesc.pushConstantSize = sizeof(float) * 4;
    auto instPipeline = device->CreateGraphicsPipeline(instDesc);

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = kHdr;
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Shadow pipelines (UNCHANGED).
    auto instShVs = loadMSL("shadow_instanced.vert.gen.metal", "instanced_shadow_vertex");
    rhi::GraphicsPipelineDesc instShDesc;
    instShDesc.vertex = instShVs.get(); instShDesc.fragment = nullptr;
    instShDesc.vertexLayout = scene::MeshVertexLayout();
    instShDesc.instanceLayout = scene::InstanceTransformLayout();
    instShDesc.depthTest = true; instShDesc.depthOnly = true;
    instShDesc.usesFrameUniforms = true; instShDesc.pushConstantSize = 0;
    auto instShadowPipeline = device->CreateGraphicsPipeline(instShDesc);

    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    // Sky (HDR RT) — UNCHANGED procedural sky.
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    // TAA resolve + final post (fullscreen, fragment push constants).
    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto taaFs  = loadMSL("taa_resolve.frag.gen.metal", "taa_resolve_fragment");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    struct TaaParams { float texel[2]; float alpha; float firstFrame; };

    rhi::GraphicsPipelineDesc taaD;
    taaD.vertex = postVs.get(); taaD.fragment = taaFs.get();
    taaD.colorFormat = kHdr;
    taaD.depthTest = false; taaD.usesTexture = true; taaD.fullscreen = true;
    taaD.fragmentPushConstants = true; taaD.pushConstantSize = sizeof(TaaParams);
    auto taaPipe = device->CreateGraphicsPipeline(taaD);

    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    // Render targets: HDR scene + two ping-pong history textures.
    auto rt    = device->CreateRenderTarget(W, H, kHdr);
    auto histA = device->CreateRenderTarget(W, H, kHdr);
    auto histB = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    rhi::BufferDesc instBufDesc;
    instBufDesc.size = (uint64_t)instances.size() * sizeof(scene::InstanceData);
    instBufDesc.initialData = instances.data();
    instBufDesc.usage = rhi::BufferUsage::Vertex;
    auto instanceBuffer = device->CreateBuffer(instBufDesc);

    Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});
    const Vec3 eye{6.5f, 4.5f, 7.0f};
    const Vec3 center{0.0f, 1.0f, 0.0f};
    const float aspect = (float)W / (float)H;
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    Mat4 unjittered = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f)) * viewM;

    FrameData fdBase{};
    {
        fdBase.lightDir[0] = -0.5f; fdBase.lightDir[1] = -1.0f; fdBase.lightDir[2] = -0.3f;
        fdBase.lightColor[0] = 1.0f; fdBase.lightColor[1] = 0.97f; fdBase.lightColor[2] = 0.9f; fdBase.lightColor[3] = 1.0f;
        fdBase.viewPos[0] = eye.x; fdBase.viewPos[1] = eye.y; fdBase.viewPos[2] = eye.z; fdBase.viewPos[3] = 1.0f;
        fdBase.ptCount[0] = 0.0f;
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 sc{0.0f, 1.0f, 0.0f};
        Vec3 lightEye = sc - lightDir * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 40.0f));
        Mat4 lightVP = lightOrtho * lightView;
        for (int k = 0; k < 16; ++k) fdBase.lightViewProj[k] = lightVP.m[k];
        Vec3 fwd = math::normalize(center - eye);
        Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
        Vec3 up = math::cross(right, fwd);
        fdBase.camFwd[0]=fwd.x; fdBase.camFwd[1]=fwd.y; fdBase.camFwd[2]=fwd.z;
        fdBase.camRight[0]=right.x; fdBase.camRight[1]=right.y; fdBase.camRight[2]=right.z;
        fdBase.camUp[0]=up.x; fdBase.camUp[1]=up.y; fdBase.camUp[2]=up.z;
        fdBase.skyParams[0] = std::tan(0.5f * kFovY);
        fdBase.skyParams[1] = aspect;
        for (int k = 0; k < 16; ++k) fdBase.prevViewProj[k] = unjittered.m[k];
    }

    auto recordScene = [&](rhi::ICommandBuffer& cmd) {
        cmd.BindPipeline(*skyPipe);
        cmd.Draw(3);
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
        cmd.BindPipeline(*instPipeline);
        {
            float material[4] = {0.1f, 0.5f, 0.0f, 0.0f};
            cmd.PushConstants(material, sizeof(material));
            cmd.BindMaterial(*groundTex, *flatNormal);
            cmd.BindVertexBuffer(sphere.vertices());
            cmd.BindInstanceBuffer(*instanceBuffer);
            cmd.BindIndexBuffer(sphere.indices());
            cmd.DrawIndexedInstanced(sphere.indexCount(), kInstanceCount);
        }
    };
    auto recordShadow = [&](rhi::ICommandBuffer& cmd) {
        cmd.BindPipeline(*staticShadowPipeline);
        cmd.PushConstants(groundModel.m, sizeof(float) * 16);
        cmd.BindVertexBuffer(plane.vertices());
        cmd.BindIndexBuffer(plane.indices());
        cmd.DrawIndexed(plane.indexCount());
        cmd.BindPipeline(*instShadowPipeline);
        cmd.BindVertexBuffer(sphere.vertices());
        cmd.BindInstanceBuffer(*instanceBuffer);
        cmd.BindIndexBuffer(sphere.indices());
        cmd.DrawIndexedInstanced(sphere.indexCount(), kInstanceCount);
    };

    // N=8 accumulation loop (identical jitter sequence to the Vulkan path).
    rhi::IRenderTarget* prevHist = histA.get();
    rhi::IRenderTarget* curHist  = histB.get();
    for (int frame = 0; frame < taa::kAccumFrames; ++frame) {
        taa::Vec2 j = taa::Jitter(frame, (int)W, (int)H);
        // Add the sub-pixel NDC jitter into the base projection's clip-space XY translation per unit W
        // BEFORE FlipProjY (so the Y-flip carries through consistently), then compose view-proj.
        Mat4 jProj = Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f);
        jProj.m[2 * 4 + 0] += j.x;
        jProj.m[2 * 4 + 1] += j.y;
        Mat4 jvp = FlipProjY(jProj) * viewM;
        FrameData fd = fdBase;
        for (int k = 0; k < 16; ++k) fd.vp[k] = jvp.m[k];

        const bool first = (frame == 0);
        TaaParams tp{};
        tp.texel[0] = 1.0f / (float)W; tp.texel[1] = 1.0f / (float)H;
        tp.alpha = first ? 1.0f : taa::kSteadyAlpha;
        tp.firstFrame = first ? 1.0f : 0.0f;

        render::RenderGraph graph;
        render::RgResource rgShadow = graph.ImportTarget(
            "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
        render::RgResource rgScene = graph.ImportTarget(
            "sceneColor", render::RgResourceKind::SceneColor, *rt);
        render::RgResource rgPrev = graph.ImportTarget(
            "history", render::RgResourceKind::SceneColor, *prevHist);
        render::RgResource rgCur = graph.ImportTarget(
            "resolved", render::RgResourceKind::SceneColor, *curHist);

        graph.AddPass("shadow", {}, {rgShadow},
            [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                dev.SetFrameUniforms(&fd, sizeof(FrameData));
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                recordShadow(cmd);
                cmd.EndRenderPass();
            });
        graph.AddPass("scene", {rgShadow}, {rgScene},
            [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                dev.SetFrameUniforms(&fd, sizeof(FrameData));
                cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
                recordScene(cmd);
                cmd.EndRenderPass();
            });
        graph.AddPass("taaResolve", {rgScene, rgPrev}, {rgCur},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*taaPipe);
                cmd.BindTexturePair(*rt, *prevHist);
                cmd.PushConstants(&tp, sizeof(tp));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });
        graph.Execute(*device);
        device->WaitIdle();
        std::swap(prevHist, curHist);
    }

    // Final: tonemap the last resolved image (now in prevHist after the swap) -> swapchain.
    {
        render::RenderGraph graph;
        render::RgResource rgResolved = graph.ImportTarget(
            "resolved", render::RgResourceKind::SceneColor, *prevHist);
        render::RgResource rgSwap = graph.ImportSwapchain("swapchain");
        graph.AddPass("post", {rgResolved}, {rgSwap},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*postPipe);
                cmd.BindTexture(*prevHist);
                cmd.Draw(3);
                cmd.EndRenderPass();
            });
        device->CaptureNextFrame();
        graph.Execute(*device);
    }

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — TAA %d-frame accumulation, %u bodies\n",
                outPath, cw, ch, taa::kAccumFrames, kInstanceCount);
    return 0;
}

// --- Screen-space reflections showcase (Slice AH). Mirrors the Vulkan --ssr-shot path: a DARK
// reflective checkerboard floor with several distinct colored objects (cubes + spheres) sitting on it,
// lit + shadowed, rendered into an HDR (RGBA16F) target PLUS a view-space normal+linear-depth g-buffer
// (reusing the SSAO gbuffer shaders). An SSR pass reconstructs each pixel's view-space P + N, computes
// R = reflect(normalize(P), N), and RAY-MARCHES R in view space — projecting each step to a screen UV
// and depth-comparing against the g-buffer — to produce mirror reflections of the objects on the
// floor (masked by a floor reflectivity from dot(N,viewUp), Fresnel, screen-edge fade). A composite
// blends lerp(scene, ssr.rgb, ssr.a) + the usual exposure/ACES/grade/vignette. The screen-space march
// runs entirely through ProjectToUV/ReconstructViewPos, which carry the HF_YS Metal Y-flip sign so the
// marched UVs sample the right texel under Metal's NDC/texture-origin convention (same as SSAO).
// SEPARATE ssr/ssr_composite pipelines + shaders; existing pipelines/shaders/goldens untouched. -----
static int RunSsrShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Scene objects: distinct colored cubes + spheres at known positions (matches the Vulkan path).
    struct Obj { Vec3 pos; float scale; bool cube; float col[3]; };
    const Obj objs[] = {
        {{-2.2f, 0.7f, -0.5f}, 0.7f, true,  {0.90f, 0.20f, 0.20f}},
        {{ 0.0f, 0.9f, -1.2f}, 0.9f, false, {0.20f, 0.85f, 0.30f}},
        {{ 2.3f, 0.6f,  0.2f}, 0.6f, true,  {0.25f, 0.45f, 0.95f}},
        {{-0.9f, 0.5f,  1.4f}, 0.5f, false, {0.95f, 0.80f, 0.20f}},
        {{ 1.4f, 0.75f, 1.6f}, 0.75f,true,  {0.85f, 0.35f, 0.90f}},
    };
    const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));

    // Lit pipeline (static, writing HDR RT) — UNCHANGED lit shaders.
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = kHdr;
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Shadow pipeline (static) — UNCHANGED.
    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    // Sky (writing HDR RT) — UNCHANGED.
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    // G-buffer prepass pipeline (static), view-space normal + linear depth -> RGBA16F.
    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;   // model(16) + view(16)
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    // SSR + composite fullscreen pipelines (fragment push constants).
    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    struct SsrParams {
        float texel[2]; float tanHalfFovY; float aspect;
        float maxDist; float thickness; float reflMin; float reflMax;
        float viewUp[4];
    };
    struct SsrCompParams { float texel[2]; float intensity; float pad; };

    auto ssrFs  = loadMSL("ssr.frag.gen.metal", "ssr_fragment");
    auto compFs = loadMSL("ssr_composite.frag.gen.metal", "ssr_composite_fragment");

    rhi::GraphicsPipelineDesc ssrD;
    ssrD.vertex = postVs.get(); ssrD.fragment = ssrFs.get();
    ssrD.colorFormat = kHdr;
    ssrD.depthTest = false; ssrD.usesTexture = true; ssrD.fullscreen = true;
    ssrD.fragmentPushConstants = true; ssrD.pushConstantSize = sizeof(SsrParams);
    auto ssrPipe = device->CreateGraphicsPipeline(ssrD);

    rhi::GraphicsPipelineDesc compD;
    compD.vertex = postVs.get(); compD.fragment = compFs.get();
    compD.colorFormat = device->Swapchain().ColorFormat();
    compD.depthTest = false; compD.usesTexture = true; compD.fullscreen = true;
    compD.fragmentPushConstants = true; compD.pushConstantSize = sizeof(SsrCompParams);
    auto compPipe = device->CreateGraphicsPipeline(compD);

    auto rt    = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf  = device->CreateRenderTarget(W, H, kHdr);
    auto ssrRT = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    // Dark low-contrast checker floor so the SSR mirror reflections read clearly (matches Vulkan path).
    std::vector<uint8_t> darkFloor(256 * 256 * 4);
    for (uint32_t y = 0; y < 256; ++y)
        for (uint32_t x = 0; x < 256; ++x) {
            bool dark = (((x / 32) + (y / 32)) & 1) != 0;
            uint8_t v = dark ? 18 : 32;
            size_t idx = (static_cast<size_t>(y) * 256 + x) * 4;
            darkFloor[idx + 0] = v; darkFloor[idx + 1] = v;
            darkFloor[idx + 2] = (uint8_t)(v + 6); darkFloor[idx + 3] = 255;
        }
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, darkFloor.data(), darkFloor.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    std::vector<std::unique_ptr<rhi::ITexture>> objTex;
    for (int o = 0; o < kNumObjs; ++o) {
        uint8_t px[4] = {(uint8_t)std::lround(objs[o].col[0] * 255.0f),
                         (uint8_t)std::lround(objs[o].col[1] * 255.0f),
                         (uint8_t)std::lround(objs[o].col[2] * 255.0f), 255};
        objTex.push_back(device->CreateTexture(
            {1, 1, rhi::Format::RGBA8_UNorm, px, sizeof(px)}));
    }

    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);
    scene::Mesh cube = scene::Mesh::Cube(*device);

    Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});
    std::vector<Mat4> objModel(kNumObjs);
    for (int o = 0; o < kNumObjs; ++o)
        objModel[o] = Mat4::Translate(objs[o].pos) * Mat4::Scale(
            {objs[o].scale, objs[o].scale, objs[o].scale});

    const Vec3 eye{0.0f, 2.6f, 7.2f};
    const Vec3 center{0.0f, 0.7f, 0.0f};
    const float aspect = (float)W / (float)H;
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    FrameData fd{};
    {
        Mat4 proj = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * viewM;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.4f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.35f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 lightDir = math::normalize(Vec3{-0.4f, -1.0f, -0.35f});
        Vec3 sc{0.0f, 0.7f, 0.0f};
        Vec3 lightEye = sc - lightDir * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-7.0f, 7.0f, -7.0f, 7.0f, 1.0f, 40.0f));
        Mat4 lightVP = lightOrtho * lightView;
        for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
        Vec3 fwd = math::normalize(center - eye);
        Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
        Vec3 up = math::cross(right, fwd);
        fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
        fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
        fd.camUp[0]=up.x; fd.camUp[1]=up.y; fd.camUp[2]=up.z;
        fd.skyParams[0] = std::tan(0.5f * kFovY);
        fd.skyParams[1] = aspect;
    }

    SsrParams sp{};
    sp.texel[0] = 1.0f / (float)W; sp.texel[1] = 1.0f / (float)H;
    sp.tanHalfFovY = std::tan(0.5f * kFovY); sp.aspect = aspect;
    sp.maxDist = 8.0f; sp.thickness = 0.35f; sp.reflMin = 0.75f; sp.reflMax = 0.92f;
    {
        // viewUp = world up (0,1,0) transformed by the view's 3x3 rotation (column-major rows).
        Vec3 wup{0.0f, 1.0f, 0.0f};
        Vec3 vUp{
            viewM.m[0]*wup.x + viewM.m[4]*wup.y + viewM.m[8]*wup.z,
            viewM.m[1]*wup.x + viewM.m[5]*wup.y + viewM.m[9]*wup.z,
            viewM.m[2]*wup.x + viewM.m[6]*wup.y + viewM.m[10]*wup.z};
        vUp = math::normalize(vUp);
        sp.viewUp[0] = vUp.x; sp.viewUp[1] = vUp.y; sp.viewUp[2] = vUp.z; sp.viewUp[3] = 0.0f;
    }
    SsrCompParams cp{}; cp.texel[0] = 1.0f / (float)W; cp.texel[1] = 1.0f / (float)H;
    cp.intensity = 1.7f; cp.pad = 0.0f;

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgGbuf = graph.ImportTarget(
        "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
    render::RgResource rgSsr = graph.ImportTarget(
        "ssr", render::RgResourceKind::SceneColor, *ssrRT);
    render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

    auto drawObj = [&](rhi::ICommandBuffer& cmd, int o) {
        const scene::Mesh& m = objs[o].cube ? cube : sphere;
        cmd.BindVertexBuffer(m.vertices());
        cmd.BindIndexBuffer(m.indices());
        cmd.DrawIndexed(m.indexCount());
    };

    graph.AddPass("shadow", {}, {rgShadow},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*staticShadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(plane.vertices());
            cmd.BindIndexBuffer(plane.indices());
            cmd.DrawIndexed(plane.indexCount());
            for (int o = 0; o < kNumObjs; ++o) {
                cmd.PushConstants(objModel[o].m, sizeof(float) * 16);
                drawObj(cmd, o);
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("scene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
            cmd.BindPipeline(*skyPipe);
            cmd.Draw(3);
            cmd.BindPipeline(*litPipeline);
            {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.15f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            for (int o = 0; o < kNumObjs; ++o) {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = objModel[o].m[k];
                pc[16] = 0.0f; pc[17] = 0.6f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*objTex[o], *flatNormal);
                drawObj(cmd, o);
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("gbuffer", {}, {rgGbuf},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
            cmd.BindPipeline(*gbStaticPipeline);
            {
                float pc[32];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                for (int k = 0; k < 16; ++k) pc[16 + k] = viewM.m[k];
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            for (int o = 0; o < kNumObjs; ++o) {
                float pc[32];
                for (int k = 0; k < 16; ++k) pc[k] = objModel[o].m[k];
                for (int k = 0; k < 16; ++k) pc[16 + k] = viewM.m[k];
                cmd.PushConstants(pc, sizeof(pc));
                drawObj(cmd, o);
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("ssr", {rgScene, rgGbuf}, {rgSsr},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
            cmd.BindPipeline(*ssrPipe);
            cmd.BindTexturePair(*rt, *gbuf);
            cmd.PushConstants(&sp, sizeof(sp));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    graph.AddPass("composite", {rgScene, rgSsr}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*compPipe);
            cmd.BindTexturePair(*rt, *ssrRT);
            cmd.PushConstants(&cp, sizeof(cp));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — SSR, %d objects\n", outPath, cw, ch, kNumObjs);
    return 0;
}

// --- Volumetric fog / light shafts showcase (Slice AJ). Mirrors the Vulkan --volumetric-shot path:
// an OVERHEAD slatted canopy (a pergola of beams with gaps) + a near-overhead directional light, so
// the light streams DOWN through the gaps and carves the foggy air into vertical light SHAFTS (god
// rays) separated by dark shadow volumes under the slats. The scene renders to an HDR RT + the SSAO
// view-space normal+linear-depth g-buffer; a fullscreen volumetric pass reconstructs each pixel's
// world-space view ray (from the camera basis in the frame UBO), clamps the march end to the scene
// depth (fog stops at solids), and RAY-MARCHES 64 steps sampling the directional shadow map per step
// — lit air adds Henyey-Greenstein in-scattering with Beer-Lambert extinction; a composite ADDS it
// over the scene + tonemaps. The per-step shadow sample carries the SAME HF_MSL_GEN V-flip the lit
// pass uses (lightViewProj has its NDC Y-flip baked in via FlipProjY, kept self-consistent with the
// shadow render). SEPARATE volumetric/volumetric_composite pipelines + shaders; existing pipelines/
// shaders/goldens untouched. -----
static int RunVolumetricShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Occluders: an overhead slatted canopy (slats run deep in Z, spaced along X with gaps) + a back
    // wall. Matches the Vulkan path.
    struct Occ { Vec3 pos; Vec3 scale; float col[3]; };
    std::vector<Occ> occs;
    const float kCanopyY = 5.4f;
    const int kSlats = 7;
    for (int p = 0; p < kSlats; ++p) {
        float x = -5.4f + (float)p * 1.8f;
        occs.push_back({{x, kCanopyY, -1.0f}, {0.45f, 0.30f, 7.0f}, {0.40f, 0.41f, 0.47f}});
    }
    occs.push_back({{0.0f, 3.0f, -9.0f}, {12.0f, 3.0f, 0.4f}, {0.30f, 0.31f, 0.36f}});
    const int kNumOcc = (int)occs.size();

    // Lit pipeline (static, writing HDR RT) — UNCHANGED lit shaders.
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = kHdr;
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Shadow pipeline (static) — UNCHANGED.
    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    // G-buffer prepass pipeline (static), view-space normal + linear depth -> RGBA16F.
    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    // Volumetric + composite fullscreen pipelines. The volumetric pass needs BOTH frame uniforms
    // (camera basis + lightViewProj + shadow map in set 0 t1/s1) AND a texture (the g-buffer in set 1
    // t0/s0), plus a fragment push constant for the fog params.
    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    struct VolParams {
        float texel[2]; float density; float g;
        float extinction; float marchDist; float steps; float pad;
    };
    struct VolCompParams { float texel[2]; float intensity; float pad; };

    auto volFs  = loadMSL("volumetric.frag.gen.metal", "volumetric_fragment");
    auto compFs = loadMSL("volumetric_composite.frag.gen.metal", "volumetric_composite_fragment");

    rhi::GraphicsPipelineDesc volD;
    volD.vertex = postVs.get(); volD.fragment = volFs.get();
    volD.colorFormat = kHdr;
    volD.depthTest = false; volD.fullscreen = true;
    volD.usesFrameUniforms = true; volD.usesTexture = true;
    volD.fragmentPushConstants = true; volD.pushConstantSize = sizeof(VolParams);
    auto volPipe = device->CreateGraphicsPipeline(volD);

    rhi::GraphicsPipelineDesc compD;
    compD.vertex = postVs.get(); compD.fragment = compFs.get();
    compD.colorFormat = device->Swapchain().ColorFormat();
    compD.depthTest = false; compD.usesTexture = true; compD.fullscreen = true;
    compD.fragmentPushConstants = true; compD.pushConstantSize = sizeof(VolCompParams);
    auto compPipe = device->CreateGraphicsPipeline(compD);

    auto rt    = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf  = device->CreateRenderTarget(W, H, kHdr);
    auto volRT = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    // Near-black floor so the god rays glow against a dark base (matches Vulkan path).
    std::vector<uint8_t> floorPx(256 * 256 * 4);
    for (uint32_t y = 0; y < 256; ++y)
        for (uint32_t x = 0; x < 256; ++x) {
            bool dark = (((x / 32) + (y / 32)) & 1) != 0;
            uint8_t v = dark ? 12 : 20;
            size_t idx = (static_cast<size_t>(y) * 256 + x) * 4;
            floorPx[idx + 0] = v; floorPx[idx + 1] = v;
            floorPx[idx + 2] = (uint8_t)(v + 3); floorPx[idx + 3] = 255;
        }
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, floorPx.data(), floorPx.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    std::vector<std::unique_ptr<rhi::ITexture>> occTex;
    for (int o = 0; o < kNumOcc; ++o) {
        uint8_t px[4] = {(uint8_t)std::lround(occs[o].col[0] * 255.0f),
                         (uint8_t)std::lround(occs[o].col[1] * 255.0f),
                         (uint8_t)std::lround(occs[o].col[2] * 255.0f), 255};
        occTex.push_back(device->CreateTexture(
            {1, 1, rhi::Format::RGBA8_UNorm, px, sizeof(px)}));
    }

    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh cube = scene::Mesh::Cube(*device);

    Mat4 groundModel = Mat4::Scale({20.0f, 1.0f, 20.0f});
    std::vector<Mat4> occModel(kNumOcc);
    for (int o = 0; o < kNumOcc; ++o)
        occModel[o] = Mat4::Translate(occs[o].pos) * Mat4::Scale(occs[o].scale);

    const Vec3 eye{0.0f, 2.7f, 10.5f};
    const Vec3 center{0.0f, 2.4f, -2.0f};
    const float aspect = (float)W / (float)H;
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    Vec3 lightDir = math::normalize(Vec3{0.05f, -0.74f, 0.67f});
    FrameData fd{};
    {
        Mat4 proj = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * viewM;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = lightDir.x; fd.lightDir[1] = lightDir.y; fd.lightDir[2] = lightDir.z;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.95f; fd.lightColor[2] = 0.82f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 1.5f, -1.0f};
        Vec3 lightEye = sc - lightDir * 26.0f;
        // Near-vertical light: world-Z up reference for the light's view basis (world-Y degenerate).
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 0, -1});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-12.0f, 12.0f, -12.0f, 12.0f, 1.0f, 52.0f));
        Mat4 lightVP = lightOrtho * lightView;
        for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
        Vec3 fwd = math::normalize(center - eye);
        Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
        Vec3 up = math::cross(right, fwd);
        fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
        fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
        fd.camUp[0]=up.x; fd.camUp[1]=up.y; fd.camUp[2]=up.z;
        fd.skyParams[0] = std::tan(0.5f * kFovY);
        fd.skyParams[1] = aspect;
    }

    VolParams vparm{};
    vparm.texel[0] = 1.0f / (float)W; vparm.texel[1] = 1.0f / (float)H;
    vparm.density = 0.8f; vparm.g = 0.4f; vparm.extinction = 0.06f;
    vparm.marchDist = 26.0f; vparm.steps = 64.0f; vparm.pad = 0.0f;
    VolCompParams cp{}; cp.texel[0] = 1.0f / (float)W; cp.texel[1] = 1.0f / (float)H;
    cp.intensity = 0.72f; cp.pad = 0.0f;

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgGbuf = graph.ImportTarget(
        "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
    render::RgResource rgVol = graph.ImportTarget(
        "volumetric", render::RgResourceKind::SceneColor, *volRT);
    render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

    graph.AddPass("shadow", {}, {rgShadow},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*staticShadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(plane.vertices());
            cmd.BindIndexBuffer(plane.indices());
            cmd.DrawIndexed(plane.indexCount());
            for (int o = 0; o < kNumOcc; ++o) {
                cmd.PushConstants(occModel[o].m, sizeof(float) * 16);
                cmd.BindVertexBuffer(cube.vertices());
                cmd.BindIndexBuffer(cube.indices());
                cmd.DrawIndexed(cube.indexCount());
            }
            cmd.EndRenderPass();
        });

    // No sky pass: dark dusk backdrop so the god rays are the bright feature.
    graph.AddPass("scene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.015f, 0.02f, 0.035f, 1});
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
            for (int o = 0; o < kNumOcc; ++o) {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = occModel[o].m[k];
                pc[16] = 0.0f; pc[17] = 0.7f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*occTex[o], *flatNormal);
                cmd.BindVertexBuffer(cube.vertices());
                cmd.BindIndexBuffer(cube.indices());
                cmd.DrawIndexed(cube.indexCount());
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("gbuffer", {}, {rgGbuf},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
            cmd.BindPipeline(*gbStaticPipeline);
            {
                float pc[32];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                for (int k = 0; k < 16; ++k) pc[16 + k] = viewM.m[k];
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            for (int o = 0; o < kNumOcc; ++o) {
                float pc[32];
                for (int k = 0; k < 16; ++k) pc[k] = occModel[o].m[k];
                for (int k = 0; k < 16; ++k) pc[16 + k] = viewM.m[k];
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(cube.vertices());
                cmd.BindIndexBuffer(cube.indices());
                cmd.DrawIndexed(cube.indexCount());
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("volumetric", {rgShadow, rgGbuf}, {rgVol},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
            cmd.BindPipeline(*volPipe);
            cmd.BindTexture(*gbuf);
            cmd.PushConstants(&vparm, sizeof(vparm));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    graph.AddPass("composite", {rgScene, rgVol}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*compPipe);
            cmd.BindTexturePair(*rt, *volRT);
            cmd.PushConstants(&cp, sizeof(cp));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — volumetric, %d occluders\n", outPath, cw, ch, kNumOcc);
    return 0;
}

// --- Debug-visualization showcase (Slice W). Mirrors the Vulkan --debug-shot path: the SAME settled
// physics sphere-pyramid scene (ground + sky + lit/shadowed resting bodies), then an immediate-mode
// DebugDraw overlay (ground grid + per-body wireframe AABB + per-body wire sphere + light-direction
// arrow + physics contact markers) rendered as ONE LINE_LIST draw via the new debug-line pipeline
// (lineList=true, usesFrameUniforms=true, depthTest=true, depthWrite=false), drawn AFTER opaque
// geometry so the lines are occluded where they pass behind the shaded spheres. One PNG -> exit. ----
static int RunDebugShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    physics::World world;
    {
        const float R = 0.5f;
        const int kLayers = 4;
        const float d = 2.0f * R;
        const float dy = R * 1.41421356f;
        for (int k = 0; k < kLayers; ++k) {
            int m = kLayers - k;
            float off = 0.5f * (float)(m - 1) * d;
            float y = R + (float)k * dy;
            for (int gx = 0; gx < m; ++gx)
                for (int gz = 0; gz < m; ++gz) {
                    float x = (float)gx * d - off;
                    float z = (float)gz * d - off;
                    world.bodies.push_back(physics::MakeDynamicSphere({x, y + 0.01f, z}, R));
                }
        }
    }
    const float dtP = 1.0f / 120.0f;
    for (int s = 0; s < 240; ++s) world.Step(dtP);

    std::vector<scene::InstanceData> instances;
    instances.reserve(world.bodies.size());
    for (const auto& b : world.bodies) {
        Mat4 m = b.Transform();
        scene::InstanceData inst;
        for (int k = 0; k < 16; ++k) inst.model[k] = m.m[k];
        instances.push_back(inst);
    }
    const uint32_t kInstanceCount = (uint32_t)instances.size();

    auto instVs = loadMSL("lit_instanced.vert.gen.metal", "instanced_vertex");
    auto litFs  = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc instDesc;
    instDesc.vertex = instVs.get(); instDesc.fragment = litFs.get();
    instDesc.vertexLayout = scene::MeshVertexLayout();
    instDesc.instanceLayout = scene::InstanceTransformLayout();
    instDesc.colorFormat = device->Swapchain().ColorFormat();
    instDesc.depthTest = true; instDesc.usesFrameUniforms = true;
    instDesc.usesTexture = true; instDesc.pushConstantSize = sizeof(float) * 4;
    auto instPipeline = device->CreateGraphicsPipeline(instDesc);

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    auto instShVs = loadMSL("shadow_instanced.vert.gen.metal", "instanced_shadow_vertex");
    rhi::GraphicsPipelineDesc instShDesc;
    instShDesc.vertex = instShVs.get(); instShDesc.fragment = nullptr;
    instShDesc.vertexLayout = scene::MeshVertexLayout();
    instShDesc.instanceLayout = scene::InstanceTransformLayout();
    instShDesc.depthTest = true; instShDesc.depthOnly = true;
    instShDesc.usesFrameUniforms = true; instShDesc.pushConstantSize = 0;
    auto instShadowPipeline = device->CreateGraphicsPipeline(instShDesc);

    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    // NEW debug-line pipeline (LINE_LIST topology; frame uniforms; depth-test on / write off).
    auto dbgVs = loadMSL("debug_line.vert.gen.metal", "debug_line_vertex");
    auto dbgFs = loadMSL("debug_line.frag.gen.metal", "debug_line_fragment");
    rhi::GraphicsPipelineDesc dbgD;
    dbgD.vertex = dbgVs.get(); dbgD.fragment = dbgFs.get();
    dbgD.vertexLayout.stride = sizeof(debug::LineVertex);
    dbgD.vertexLayout.attributes = {
        {0, rhi::Format::RGB32_Float, 0},
        {1, rhi::Format::RGB32_Float, 12},
    };
    dbgD.colorFormat = device->Swapchain().ColorFormat();
    dbgD.lineList = true; dbgD.depthTest = true; dbgD.depthWrite = false;
    dbgD.usesFrameUniforms = true;
    auto debugPipeline = device->CreateGraphicsPipeline(dbgD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    rhi::BufferDesc instBufDesc;
    instBufDesc.size = (uint64_t)instances.size() * sizeof(scene::InstanceData);
    instBufDesc.initialData = instances.data();
    instBufDesc.usage = rhi::BufferUsage::Vertex;
    auto instanceBuffer = device->CreateBuffer(instBufDesc);

    Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});

    // Build the debug overlay (identical construction to the Vulkan --debug-shot path).
    debug::DebugDraw dd;
    const Vec3 lightDirVec = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
    dd.Grid(8.0f, 1.0f, {0.30f, 0.32f, 0.38f});
    const scene::MeshBounds& sb = sphere.bounds();
    for (const auto& b : world.bodies) {
        Mat4 m = b.Transform();
        debug::AabbWorld(dd, sb.min, sb.max, m, {1.0f, 0.85f, 0.1f});
        dd.WireSphere(b.position, b.radius, {0.1f, 0.9f, 0.95f}, 16);
    }
    debug::LightArrow(dd, {3.5f, 4.5f, 3.5f}, lightDirVec, 2.5f, {1.0f, 0.55f, 0.1f});
    debug::PhysicsContacts(dd, world, {1.0f, 0.2f, 0.8f}, {0.2f, 1.0f, 0.3f});

    const uint32_t kLineVertCount = (uint32_t)dd.VertexCount();
    rhi::BufferDesc lineBufDesc;
    lineBufDesc.size = (uint64_t)dd.Vertices().size() * sizeof(debug::LineVertex);
    lineBufDesc.initialData = dd.Vertices().data();
    lineBufDesc.usage = rhi::BufferUsage::Vertex;
    auto lineBuffer = device->CreateBuffer(lineBufDesc);

    const Vec3 eye{6.5f, 4.5f, 7.0f};
    const Vec3 center{0.0f, 1.0f, 0.0f};
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 1.0f, 0.0f};
        Vec3 lightEye = sc - lightDirVec * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 40.0f));
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
            cmd.BindPipeline(*staticShadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(plane.vertices());
            cmd.BindIndexBuffer(plane.indices());
            cmd.DrawIndexed(plane.indexCount());
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
            cmd.BindPipeline(*instPipeline);
            {
                float material[4] = {0.1f, 0.5f, 0.0f, 0.0f};
                cmd.PushConstants(material, sizeof(material));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(sphere.vertices());
                cmd.BindInstanceBuffer(*instanceBuffer);
                cmd.BindIndexBuffer(sphere.indices());
                cmd.DrawIndexedInstanced(sphere.indexCount(), kInstanceCount);
            }
            // Debug overlay: one LINE_LIST draw, after opaque geometry (depth-tested, no depth write).
            if (kLineVertCount > 0) {
                cmd.BindPipeline(*debugPipeline);
                cmd.BindVertexBuffer(*lineBuffer);
                cmd.Draw(kLineVertCount);
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — %u bodies, %u debug-line vertices\n",
                outPath, cw, ch, kInstanceCount, kLineVertCount);
    return 0;
}

// --- Frustum-culling visualization showcase (Slice AQ). Mirrors the Vulkan --cull-shot path: a
// deterministic ground plane + a wide row of cubes/spheres across X, rendered from a pulled-back
// OVERVIEW camera, with the actual (narrower) RENDER camera's frustum drawn as wireframe LINES and
// each object's bounding SPHERE colored GREEN (render camera keeps) / RED (render camera culls). The
// cull partition is the conservative bounding-sphere test in engine/render/frustum.h. The render
// camera's cull frustum is built from the SAME Metal-clip matrix (FlipProjY*View) the renderer would
// use, so the partition + corners match the captured orientation. One PNG -> exit (two-run DIFF
// 0.0000). New golden tests/golden/metal/cull.png. ------------------------------------------------
static int RunCullShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace fr = render::frustum;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    scene::Mesh planeMesh  = scene::Mesh::Plane(*device);
    scene::Mesh cubeMesh   = scene::Mesh::Cube(*device);
    scene::Mesh sphereMesh = scene::Mesh::Sphere(*device);

    struct CullObj { const scene::Mesh* mesh; scene::Transform xform; };
    std::vector<CullObj> objs;
    for (int k = -6; k <= 6; ++k) {
        scene::Transform t;
        t.position = {(float)k * 2.2f, 0.8f, -6.0f};
        t.scale = {0.7f, 0.7f, 0.7f};
        objs.push_back({(k & 1) ? &sphereMesh : &cubeMesh, t});
    }

    // The actual RENDER camera (narrow). Its cull frustum uses the SAME Metal-clip matrix the renderer
    // would build (FlipProjY(Proj)*View), so FromViewProj/Corners match the captured orientation.
    runtime::Camera renderCam;
    renderCam.position = {0.0f, 0.8f, 2.0f};
    renderCam.yaw = 0.0f; renderCam.SetPitch(0.0f);
    renderCam.fovY = 0.8726646f;  // 50 degrees
    renderCam.aspect = (float)W / (float)H;
    renderCam.znear = 0.5f; renderCam.zfar = 12.0f;
    Mat4 renderVP = FlipProjY(renderCam.Proj()) * renderCam.View();
    fr::Frustum cullFrustum = fr::FromViewProj(renderVP);

    struct ObjBound { Vec3 center; float radius; bool kept; };
    std::vector<ObjBound> bounds;
    int drawn = 0, culled = 0;
    for (const auto& ob : objs) {
        const scene::MeshBounds& mb = ob.mesh->bounds();
        Vec3 localCenter{(mb.min.x + mb.max.x) * 0.5f, (mb.min.y + mb.max.y) * 0.5f,
                         (mb.min.z + mb.max.z) * 0.5f};
        Vec3 ext{(mb.max.x - mb.min.x) * 0.5f, (mb.max.y - mb.min.y) * 0.5f,
                 (mb.max.z - mb.min.z) * 0.5f};
        float localRadius = math::length(ext);
        Mat4 model = ob.xform.Matrix();
        Vec3 worldCenter = math::MulPoint(model, localCenter);
        const Vec3& s = ob.xform.scale;
        float maxAbsScale = std::max({std::fabs(s.x), std::fabs(s.y), std::fabs(s.z)});
        float worldRadius = localRadius * maxAbsScale;
        bool cull = fr::SphereOutside(cullFrustum, worldCenter, worldRadius);
        bounds.push_back({worldCenter, worldRadius, !cull});
        if (cull) ++culled; else ++drawn;
    }
    const int total = (int)objs.size();

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto dbgVs = loadMSL("debug_line.vert.gen.metal", "debug_line_vertex");
    auto dbgFs = loadMSL("debug_line.frag.gen.metal", "debug_line_fragment");
    rhi::GraphicsPipelineDesc dbgD;
    dbgD.vertex = dbgVs.get(); dbgD.fragment = dbgFs.get();
    dbgD.vertexLayout.stride = sizeof(debug::LineVertex);
    dbgD.vertexLayout.attributes = {
        {0, rhi::Format::RGB32_Float, 0},
        {1, rhi::Format::RGB32_Float, 12},
    };
    dbgD.colorFormat = device->Swapchain().ColorFormat();
    dbgD.lineList = true; dbgD.depthTest = true; dbgD.depthWrite = false;
    dbgD.usesFrameUniforms = true;
    auto debugPipeline = device->CreateGraphicsPipeline(dbgD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});

    scene::Transform groundXform; groundXform.scale = {16.0f, 1.0f, 16.0f};
    Mat4 groundModel = groundXform.Matrix();

    // Debug overlay: render frustum (white wireframe) + per-object bound spheres green/red.
    debug::DebugDraw dd;
    {
        const Vec3 kFrustumColor{0.95f, 0.95f, 0.98f};
        Vec3 corners[8];
        fr::Corners(renderVP, corners);
        auto edge = [&](int a, int b) { dd.Line(corners[a], corners[b], kFrustumColor); };
        edge(0, 1); edge(1, 3); edge(3, 2); edge(2, 0);
        edge(4, 5); edge(5, 7); edge(7, 6); edge(6, 4);
        edge(0, 4); edge(1, 5); edge(2, 6); edge(3, 7);
        const Vec3 kKeep{0.15f, 0.95f, 0.25f};
        const Vec3 kCull{0.95f, 0.18f, 0.15f};
        for (const auto& bnd : bounds)
            dd.WireSphere(bnd.center, bnd.radius, bnd.kept ? kKeep : kCull, 16);
    }
    const uint32_t kLineVertCount = (uint32_t)dd.VertexCount();
    rhi::BufferDesc lineBufDesc;
    lineBufDesc.size = (uint64_t)dd.Vertices().size() * sizeof(debug::LineVertex);
    lineBufDesc.initialData = dd.Vertices().data();
    lineBufDesc.usage = rhi::BufferUsage::Vertex;
    auto lineBuffer = device->CreateBuffer(lineBufDesc);

    // The OVERVIEW camera (this is what we actually render). Identical pose to the Vulkan --cull-shot.
    runtime::Camera overviewCam;
    overviewCam.position = {6.0f, 11.0f, 16.0f};
    overviewCam.yaw = -0.32f; overviewCam.SetPitch(-0.52f);
    overviewCam.fovY = 1.04719755f;  // 60 degrees
    overviewCam.aspect = (float)W / (float)H;
    overviewCam.znear = 0.1f; overviewCam.zfar = 100.0f;

    const Vec3 lightDirVec = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
    FrameData fd{};
    {
        Mat4 proj = FlipProjY(overviewCam.Proj());
        Mat4 vp = proj * overviewCam.View();
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = overviewCam.position.x; fd.viewPos[1] = overviewCam.position.y;
        fd.viewPos[2] = overviewCam.position.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 1.0f, -4.0f};
        Vec3 lightEye = sc - lightDirVec * 24.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-16.0f, 16.0f, -16.0f, 16.0f, 1.0f, 60.0f));
        Mat4 lightVP = lightOrtho * lightView;
        for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
        runtime::CameraBasis cb = overviewCam.Basis();
        fd.camFwd[0]=cb.forward.x; fd.camFwd[1]=cb.forward.y; fd.camFwd[2]=cb.forward.z;
        fd.camRight[0]=cb.right.x; fd.camRight[1]=cb.right.y; fd.camRight[2]=cb.right.z;
        fd.camUp[0]=cb.up.x; fd.camUp[1]=cb.up.y; fd.camUp[2]=cb.up.z;
        fd.skyParams[0] = cb.tanHalfFovY; fd.skyParams[1] = overviewCam.aspect;
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
            cmd.BindPipeline(*staticShadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(planeMesh.vertices());
            cmd.BindIndexBuffer(planeMesh.indices());
            cmd.DrawIndexed(planeMesh.indexCount());
            for (const auto& ob : objs) {
                Mat4 m = ob.xform.Matrix();
                cmd.PushConstants(m.m, sizeof(float) * 16);
                cmd.BindVertexBuffer(ob.mesh->vertices());
                cmd.BindIndexBuffer(ob.mesh->indices());
                cmd.DrawIndexed(ob.mesh->indexCount());
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("scene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
            cmd.BindPipeline(*skyPipe);
            cmd.Draw(3);
            cmd.BindPipeline(*litPipeline);
            cmd.BindMaterial(*groundTex, *flatNormal);
            {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.8f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(planeMesh.vertices());
                cmd.BindIndexBuffer(planeMesh.indices());
                cmd.DrawIndexed(planeMesh.indexCount());
            }
            for (const auto& ob : objs) {
                Mat4 m = ob.xform.Matrix();
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                pc[16] = 0.0f; pc[17] = 0.6f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(ob.mesh->vertices());
                cmd.BindIndexBuffer(ob.mesh->indices());
                cmd.DrawIndexed(ob.mesh->indexCount());
            }
            if (kLineVertCount > 0) {
                cmd.BindPipeline(*debugPipeline);
                cmd.BindVertexBuffer(*lineBuffer);
                cmd.Draw(kLineVertCount);
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
    graph.Execute(*device);

    std::printf("cull: {drawn: %d, culled: %d, total: %d}\n", drawn, culled, total);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — %d drawn / %d culled of %d, %u debug-line vertices\n",
                outPath, cw, ch, drawn, culled, total, kLineVertCount);
    return 0;
}

// --- Multithreaded command recording showcase (Slice AU). Mirrors the Vulkan --mt-shot path: a
// draw-heavy 12x12 grid of NON-INSTANCED DISTINCT draws whose scene-pass draw list is partitioned
// into `workers` contiguous index ranges; each worker records its range into a sub-encoder vended by
// an MTLParallelRenderCommandEncoder (creation order == worker index == commit order). The draw
// order therefore equals single-threaded recording, so 1-worker and N-worker renders are identical.
// New golden tests/golden/metal/mt.png.
static int RunMtShowcase(const char* outPath, uint32_t workers) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    const uint32_t N = workers < 1 ? 1 : workers;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    scene::Mesh planeMesh  = scene::Mesh::Plane(*device);
    scene::Mesh cubeMesh   = scene::Mesh::Cube(*device);
    scene::Mesh sphereMesh = scene::Mesh::Sphere(*device);

    struct MtObj { const scene::Mesh* mesh; scene::Transform xform; float tint; };
    std::vector<MtObj> objs;
    const int kGrid = 12;
    for (int gz = 0; gz < kGrid; ++gz) {
        for (int gx = 0; gx < kGrid; ++gx) {
            int idx = gz * kGrid + gx;
            scene::Transform t;
            float fx = (float)(gx - kGrid / 2) * 1.5f + 0.75f;
            float fz = (float)(gz - kGrid / 2) * 1.5f + 0.75f;
            float bob = 0.35f * std::sin((float)idx * 0.7f);
            t.position = {fx, 0.7f + bob, fz};
            t.scale = {0.45f, 0.45f, 0.45f};
            objs.push_back({(idx & 1) ? &sphereMesh : &cubeMesh, t,
                            0.35f + 0.5f * (float)((idx * 37) % 100) / 100.0f});
        }
    }
    const uint32_t kDraws = (uint32_t)objs.size();

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});

    scene::Transform groundXform; groundXform.scale = {14.0f, 1.0f, 14.0f};
    Mat4 groundModel = groundXform.Matrix();

    const Vec3 eye{0.0f, 13.0f, 16.0f};
    const Vec3 center{0.0f, 0.5f, 0.0f};
    const Vec3 lightDirVec = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
    FrameData fd{};
    {
        runtime::Camera cam;
        cam.position = eye;
        Vec3 dir = math::normalize(center - eye);
        cam.yaw = std::atan2(dir.x, -dir.z);
        cam.SetPitch(std::asin(dir.y));
        cam.fovY = 0.9599311f;  // 55 degrees
        cam.aspect = (float)W / (float)H;
        cam.znear = 0.5f; cam.zfar = 80.0f;
        Mat4 vp = FlipProjY(cam.Proj()) * cam.View();
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 1.0f, 0.0f};
        Vec3 lightEye = sc - lightDirVec * 26.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-15.0f, 15.0f, -15.0f, 15.0f, 1.0f, 60.0f));
        Mat4 lightVP = lightOrtho * lightView;
        for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
        runtime::CameraBasis cb = cam.Basis();
        fd.camFwd[0]=cb.forward.x; fd.camFwd[1]=cb.forward.y; fd.camFwd[2]=cb.forward.z;
        fd.camRight[0]=cb.right.x; fd.camRight[1]=cb.right.y; fd.camRight[2]=cb.right.z;
        fd.camUp[0]=cb.up.x; fd.camUp[1]=cb.up.y; fd.camUp[2]=cb.up.z;
        fd.skyParams[0] = cb.tanHalfFovY; fd.skyParams[1] = cam.aspect;
    }

    auto recordObject = [&](rhi::ICommandBuffer& c, const MtObj& ob) {
        Mat4 m = ob.xform.Matrix();
        float pc[20];
        for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
        pc[16] = 0.0f; pc[17] = ob.tint; pc[18] = 0.0f; pc[19] = 0.0f;
        c.PushConstants(pc, sizeof(pc));
        c.BindVertexBuffer(ob.mesh->vertices());
        c.BindIndexBuffer(ob.mesh->indices());
        c.DrawIndexed(ob.mesh->indexCount());
    };

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
            cmd.BindPipeline(*staticShadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(planeMesh.vertices());
            cmd.BindIndexBuffer(planeMesh.indices());
            cmd.DrawIndexed(planeMesh.indexCount());
            for (const auto& ob : objs) {
                Mat4 m = ob.xform.Matrix();
                cmd.PushConstants(m.m, sizeof(float) * 16);
                cmd.BindVertexBuffer(ob.mesh->vertices());
                cmd.BindIndexBuffer(ob.mesh->indices());
                cmd.DrawIndexed(ob.mesh->indexCount());
            }
            cmd.EndRenderPass();
        });

    // Scene pass — MULTITHREADED via the parallel render command encoder.
    graph.AddPass("scene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1},
                                /*expectsSecondaries=*/true);

            std::vector<rhi::ICommandBuffer*> secs(N, nullptr);
            for (uint32_t k = 0; k < N; ++k)
                secs[k] = dev.CreateSecondaryCommandBuffer(k);

            // Worker 0's sub-encoder gets the sky + ground first (single-threaded), matching the
            // single-threaded draw order; then the parallel object recording fills each sub-encoder.
            secs[0]->BindPipeline(*skyPipe);
            secs[0]->Draw(3);
            secs[0]->BindPipeline(*litPipeline);
            secs[0]->BindMaterial(*groundTex, *flatNormal);
            {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.7f; pc[18] = 0.0f; pc[19] = 0.0f;
                secs[0]->PushConstants(pc, sizeof(pc));
                secs[0]->BindVertexBuffer(planeMesh.vertices());
                secs[0]->BindIndexBuffer(planeMesh.indices());
                secs[0]->DrawIndexed(planeMesh.indexCount());
            }

            runtime::RecordParallel(kDraws, N,
                [&](uint32_t t, uint32_t b, uint32_t e) {
                    rhi::ICommandBuffer& c = *secs[t];
                    c.BindPipeline(*litPipeline);
                    c.BindMaterial(*groundTex, *flatNormal);
                    for (uint32_t i = b; i < e; ++i) recordObject(c, objs[i]);
                });

            cmd.ExecuteSecondaries(std::span<rhi::ICommandBuffer* const>(secs.data(), secs.size()));
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
    graph.Execute(*device);

    std::printf("mt: {workers:%u, draws:%u, secondaries:%u}\n", N, kDraws, N);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — %u draws across %u workers\n", outPath, cw, ch, kDraws, N);
    return 0;
}

// --- GPU-driven culling + indirect draw showcase (Slice AR). Mirrors the Vulkan --gpu-cull-shot
// path: a compute kernel (cull.comp.gen.metal) frustum-culls a deterministic 1024-instance cube grid
// and ORDER-compacts the survivors into a second instance buffer (single-workgroup prefix sum) +
// writes the indirect draw-args (instanceCount = survivor count). ONE DrawIndexedIndirect renders
// exactly the survivors — the count decided on the GPU. We read the GPU count back and ASSERT it
// equals the CPU reference (engine/render/gpu_cull.h) over the SAME instances + frustum: the
// exact-count proof. One offscreen frame -> PNG (two-run DIFF 0.0000). New golden gpu_cull.png.
//
// METAL CLIP: the camera composes FlipProjY into the projection (the Metal clip convention), and the
// cull frustum is extracted from that SAME composed matrix, so the GPU planes + the CPU reference +
// the rendered orientation all agree. The compute kernel + the indirect indexed draw both run
// headlessly on the M4. ------------------------------------------------------------------------------
static int RunGpuCullShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace fr = render::frustum;
    namespace gc = render::gpu_cull;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // The 32x32 = 1024 instance cube grid (one workgroup). Identical builder + local bound to Vulkan.
    const uint32_t kGridN = 32;
    std::vector<scene::InstanceData> grid =
        scene::BuildInstanceGrid(kGridN, /*spacing=*/1.3f, /*scale=*/0.45f);
    const uint32_t kInstanceCount = (uint32_t)grid.size();
    const Vec3 localCenter{0.0f, 0.0f, 0.0f};
    const float localRadius = std::sqrt(0.75f);

    std::vector<float> models;
    models.reserve((size_t)kInstanceCount * 16);
    for (const auto& inst : grid)
        for (int k = 0; k < 16; ++k) models.push_back(inst.model[k]);

    scene::Mesh planeMesh = scene::Mesh::Plane(*device);
    scene::Mesh cubeMesh  = scene::Mesh::Cube(*device);

    // Camera (identical pose/FOV to the Vulkan path); the cull frustum is built from the SAME
    // FlipProjY-composed view-proj the renderer uses, so the planes match the captured orientation.
    const Vec3 eye{0.0f, 11.0f, 17.0f};
    const Vec3 center{0.0f, 0.0f, 0.0f};
    const float aspect = (float)W / (float)H;
    Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
    Mat4 proj = FlipProjY(Mat4::Perspective(0.6108652f /*35deg*/, aspect, 0.5f, 80.0f));
    Mat4 vp = proj * view;
    fr::Frustum cullFrustum = fr::FromViewProj(vp);

    const uint32_t cpuRef =
        gc::SurvivorCount(models, kInstanceCount, cullFrustum, localCenter, localRadius);

    // GPU buffers: source instances / compacted survivors / indirect args / cull params.
    rhi::BufferDesc srcDesc;
    srcDesc.size = (uint64_t)kInstanceCount * sizeof(scene::InstanceData);
    srcDesc.initialData = grid.data();
    srcDesc.usage = rhi::BufferUsage::Storage;
    auto srcInstances = device->CreateBuffer(srcDesc);

    rhi::BufferDesc survDesc;
    survDesc.size = (uint64_t)kInstanceCount * sizeof(scene::InstanceData);
    survDesc.usage = rhi::BufferUsage::Storage;
    auto survInstances = device->CreateBuffer(survDesc);

    uint32_t argsInit[5] = {cubeMesh.indexCount(), 0u, 0u, 0u, 0u};
    rhi::BufferDesc argsDesc;
    argsDesc.size = sizeof(argsInit);
    argsDesc.initialData = argsInit;
    argsDesc.usage = rhi::BufferUsage::Indirect;
    auto argsBuffer = device->CreateBuffer(argsDesc);

    struct CullParams { float planes[6][4]; float localCenter[4]; uint32_t counts[4]; };
    static_assert(sizeof(CullParams) == 128, "CullParams layout");
    CullParams params{};
    for (int p = 0; p < 6; ++p) {
        params.planes[p][0] = cullFrustum.planes[p].n.x;
        params.planes[p][1] = cullFrustum.planes[p].n.y;
        params.planes[p][2] = cullFrustum.planes[p].n.z;
        params.planes[p][3] = cullFrustum.planes[p].d;
    }
    params.localCenter[0] = localCenter.x; params.localCenter[1] = localCenter.y;
    params.localCenter[2] = localCenter.z; params.localCenter[3] = localRadius;
    params.counts[0] = kInstanceCount;
    params.counts[1] = cubeMesh.indexCount();
    rhi::BufferDesc paramDesc;
    paramDesc.size = sizeof(params);
    paramDesc.initialData = &params;
    paramDesc.usage = rhi::BufferUsage::Storage;
    auto paramBuffer = device->CreateBuffer(paramDesc);

    // Compute cull pipeline: 4 storage buffers, one workgroup of 1024 threads.
    auto cullCs = loadMSL("cull.comp.gen.metal", "cull_main");
    rhi::ComputePipelineDesc cullCdesc;
    cullCdesc.compute = cullCs.get();
    cullCdesc.storageBufferCount = 4;
    cullCdesc.pushConstantSize = 0;
    cullCdesc.threadsPerGroupX = 1024;
    auto cullCompute = device->CreateComputePipeline(cullCdesc);

    // Instanced lit pipeline (draws the survivor stream) + static lit (ground) + sky + post.
    auto instVs = loadMSL("lit_instanced.vert.gen.metal", "instanced_vertex");
    auto litFs  = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc instDesc;
    instDesc.vertex = instVs.get(); instDesc.fragment = litFs.get();
    instDesc.vertexLayout = scene::MeshVertexLayout();
    instDesc.instanceLayout = scene::InstanceTransformLayout();
    instDesc.colorFormat = device->Swapchain().ColorFormat();
    instDesc.depthTest = true; instDesc.usesFrameUniforms = true;
    instDesc.usesTexture = true; instDesc.pushConstantSize = sizeof(float) * 4;
    auto instPipeline = device->CreateGraphicsPipeline(instDesc);

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    Mat4 groundModel = Mat4::Scale({26.0f, 1.0f, 26.0f});

    FrameData fd{};
    {
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 sc{0.0f, 0.0f, 0.0f};
        Vec3 lightEye = sc - lightDir * 30.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-24.0f, 24.0f, -24.0f, 24.0f, 1.0f, 70.0f));
        Mat4 lightVP = lightOrtho * lightView;
        for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
        Vec3 fwd = math::normalize(center - eye);
        Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
        Vec3 up = math::cross(right, fwd);
        fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
        fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
        fd.camUp[0]=up.x; fd.camUp[1]=up.y; fd.camUp[2]=up.z;
        fd.skyParams[0] = std::tan(0.5f * 0.6108652f);
        fd.skyParams[1] = aspect;
    }

    render::RenderGraph graph;
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

    graph.AddPass("scene", {}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            // GPU cull dispatch (outside the render pass): compute compacts survivors + writes args.
            cmd.BindComputePipeline(*cullCompute);
            cmd.BindStorageBuffer(*srcInstances, 0);
            cmd.BindStorageBuffer(*survInstances, 1);
            cmd.BindStorageBuffer(*argsBuffer, 2);
            cmd.BindStorageBuffer(*paramBuffer, 3);
            cmd.DispatchCompute(1);   // ONE workgroup of 1024 threads
            cmd.ComputeToVertexBarrier();

            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
            cmd.BindPipeline(*skyPipe);
            cmd.Draw(3);
            cmd.BindPipeline(*litPipeline);
            {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.85f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(planeMesh.vertices());
                cmd.BindIndexBuffer(planeMesh.indices());
                cmd.DrawIndexed(planeMesh.indexCount());
            }
            // GPU-CULLED instanced cubes: ONE DrawIndexedIndirect (args + survivor stream from compute).
            cmd.BindPipeline(*instPipeline);
            {
                float material[4] = {0.1f, 0.5f, 0.0f, 0.0f};
                cmd.PushConstants(material, sizeof(material));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(cubeMesh.vertices());
                cmd.BindInstanceBuffer(*survInstances);
                cmd.BindIndexBuffer(cubeMesh.indices());
                cmd.DrawIndexedIndirect(*argsBuffer, 0);
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
    graph.Execute(*device);
    device->WaitIdle();

    // Exact-count proof: read the GPU-written instanceCount back and assert == CPU reference.
    uint32_t gpuArgs[5] = {0, 0, 0, 0, 0};
    device->ReadBuffer(*argsBuffer, gpuArgs, sizeof(gpuArgs), 0);
    const uint32_t gpuDrawn = gpuArgs[1];
    std::printf("gpu-cull: {drawn: %u, cpuRef: %u, total: %u}\n", gpuDrawn, cpuRef, kInstanceCount);
    if (gpuDrawn != cpuRef)
        return fail("GPU survivor count != CPU reference (cull-logic mismatch)");

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — GPU-culled %u/%u instances (indirect draw)\n",
                outPath, cw, ch, gpuDrawn, kInstanceCount);
    return 0;
}

// --- Editor gizmo showcase (Slice AB). Mirrors the Vulkan --gizmo-shot path: a small deterministic
// multi-object scene (ground plane + cube + sphere + tall box), SELECT object <objIndex>, render the
// scene plus that object's translate gizmo (3 colored axis arrows) through the debug-line layer from
// a fixed runtime::Camera. One PNG -> exit (two-run DIFF 0.0000). New golden. -----------------------
static int RunGizmoShowcase(int objIndex, const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    scene::Mesh planeMesh = scene::Mesh::Plane(*device);
    scene::Mesh cubeMesh  = scene::Mesh::Cube(*device);
    scene::Mesh sphereMesh = scene::Mesh::Sphere(*device);

    struct GizmoObj { const scene::Mesh* mesh; scene::Transform xform; };
    std::vector<GizmoObj> objs;
    { scene::Transform t; t.scale = {7.0f, 1.0f, 7.0f}; objs.push_back({&planeMesh, t}); }
    { scene::Transform t; t.position = {-2.0f, 0.5f, 0.0f}; objs.push_back({&cubeMesh, t}); }
    { scene::Transform t; t.position = {1.5f, 1.0f, 1.5f}; t.scale = {1.6f, 1.6f, 1.6f};
      objs.push_back({&sphereMesh, t}); }
    { scene::Transform t; t.position = {3.0f, 1.5f, -1.0f}; t.scale = {0.6f, 3.0f, 0.6f};
      objs.push_back({&cubeMesh, t}); }

    int selIndex = objIndex;
    if (selIndex < 0) selIndex = 0;
    if (selIndex >= (int)objs.size()) selIndex = (int)objs.size() - 1;

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto dbgVs = loadMSL("debug_line.vert.gen.metal", "debug_line_vertex");
    auto dbgFs = loadMSL("debug_line.frag.gen.metal", "debug_line_fragment");
    rhi::GraphicsPipelineDesc dbgD;
    dbgD.vertex = dbgVs.get(); dbgD.fragment = dbgFs.get();
    dbgD.vertexLayout.stride = sizeof(debug::LineVertex);
    dbgD.vertexLayout.attributes = {
        {0, rhi::Format::RGB32_Float, 0},
        {1, rhi::Format::RGB32_Float, 12},
    };
    dbgD.colorFormat = device->Swapchain().ColorFormat();
    dbgD.lineList = true; dbgD.depthTest = true; dbgD.depthWrite = false;
    dbgD.usesFrameUniforms = true;
    auto debugPipeline = device->CreateGraphicsPipeline(dbgD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});

    // Build the selected object's translate gizmo through the debug-line layer.
    debug::DebugDraw dd;
    {
        const scene::Transform& xf = objs[selIndex].xform;
        float reach = std::max({xf.scale.x, xf.scale.y, xf.scale.z});
        float handleLen = 1.2f + 0.8f * reach;
        editor::EmitGizmo(dd, xf, editor::GizmoMode::Translate, handleLen, editor::kAxisNone);
    }
    const uint32_t kLineVertCount = (uint32_t)dd.VertexCount();
    rhi::BufferDesc lineBufDesc;
    lineBufDesc.size = (uint64_t)dd.Vertices().size() * sizeof(debug::LineVertex);
    lineBufDesc.initialData = dd.Vertices().data();
    lineBufDesc.usage = rhi::BufferUsage::Vertex;
    auto lineBuffer = device->CreateBuffer(lineBufDesc);

    runtime::Camera cam;
    cam.position = {2.0f, 4.5f, 9.0f};
    cam.yaw = 0.15f; cam.SetPitch(-0.32f);
    cam.aspect = (float)W / (float)H;

    const Vec3 lightDirVec = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
    FrameData fd{};
    {
        // Camera Proj() bakes the Vulkan Y-flip; Metal applies its own flip on top (as the other
        // Metal showcases do) so the captured frame matches the Vulkan orientation.
        Mat4 proj = FlipProjY(cam.Proj());
        Mat4 vp = proj * cam.View();
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = cam.position.x; fd.viewPos[1] = cam.position.y;
        fd.viewPos[2] = cam.position.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 1.0f, 0.0f};
        Vec3 lightEye = sc - lightDirVec * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 40.0f));
        Mat4 lightVP = lightOrtho * lightView;
        for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
        runtime::CameraBasis cb = cam.Basis();
        fd.camFwd[0]=cb.forward.x; fd.camFwd[1]=cb.forward.y; fd.camFwd[2]=cb.forward.z;
        fd.camRight[0]=cb.right.x; fd.camRight[1]=cb.right.y; fd.camRight[2]=cb.right.z;
        fd.camUp[0]=cb.up.x; fd.camUp[1]=cb.up.y; fd.camUp[2]=cb.up.z;
        fd.skyParams[0] = cb.tanHalfFovY; fd.skyParams[1] = cam.aspect;
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
            cmd.BindPipeline(*staticShadowPipeline);
            for (const auto& ob : objs) {
                Mat4 m = ob.xform.Matrix();
                cmd.PushConstants(m.m, sizeof(float) * 16);
                cmd.BindVertexBuffer(ob.mesh->vertices());
                cmd.BindIndexBuffer(ob.mesh->indices());
                cmd.DrawIndexed(ob.mesh->indexCount());
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("scene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
            cmd.BindPipeline(*skyPipe);
            cmd.Draw(3);
            cmd.BindPipeline(*litPipeline);
            cmd.BindMaterial(*groundTex, *flatNormal);
            for (const auto& ob : objs) {
                Mat4 m = ob.xform.Matrix();
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                pc[16] = 0.0f; pc[17] = 0.6f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(ob.mesh->vertices());
                cmd.BindIndexBuffer(ob.mesh->indices());
                cmd.DrawIndexed(ob.mesh->indexCount());
            }
            if (kLineVertCount > 0) {
                cmd.BindPipeline(*debugPipeline);
                cmd.BindVertexBuffer(*lineBuffer);
                cmd.Draw(kLineVertCount);
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — selected object %d, %u gizmo-line vertices\n",
                outPath, cw, ch, selIndex, kLineVertCount);
    return 0;
}

// --- Alpha-blended transparency showcase (Slice T). Mirrors the Vulkan --transparency-shot path:
// checkerboard ground + procedural sky + a few OPAQUE lit cubes (two behind the glass, one in front
// as an occluder), then 4 overlapping tinted GLASS spheres at different depths rendered in a SORTED
// (back-to-front) alpha-blended pass that depth-TESTS the opaque scene but does NOT write depth (the
// new depthWrite=false). One offscreen frame -> PNG. The transparent MSL is generated from the shared
// HLSL (transparent.vert / transparent.frag) by the sibling CMake gen rules. --------
static int RunTransparencyShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Opaque lit pipeline (ground + opaque cubes): shared lit.vert + lit.frag.
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Transparent ("glass") pipeline: transparent.vert + transparent.frag. alphaBlend ON, depthTest
    // ON, depthWrite OFF (reads opaque depth, never writes), double-sided (cullNone). Push = 80 bytes.
    auto tVs = loadMSL("transparent.vert.gen.metal", "transparent_vertex");
    auto tFs = loadMSL("transparent.frag.gen.metal", "transparent_fragment");
    rhi::GraphicsPipelineDesc tDesc;
    tDesc.vertex = tVs.get(); tDesc.fragment = tFs.get();
    tDesc.vertexLayout = scene::MeshVertexLayout();
    tDesc.colorFormat = device->Swapchain().ColorFormat();
    tDesc.depthTest = true; tDesc.depthWrite = false; tDesc.alphaBlend = true; tDesc.cullNone = true;
    tDesc.usesFrameUniforms = true; tDesc.usesTexture = false;
    tDesc.pushConstantSize = sizeof(float) * 20;  // model + float4 tintAlpha
    auto transparentPipeline = device->CreateGraphicsPipeline(tDesc);

    // Static depth-only shadow pipeline (ground + opaque cubes cast shadows).
    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

    // Sky + post.
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh cube = scene::Mesh::Cube(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});

    // Two opaque cubes BEHIND the glass + one IN FRONT (occluder). Identical to the Vulkan path.
    struct Opaque { Mat4 model; };
    std::vector<Opaque> opaques = {
        {Mat4::Translate({-2.2f, 0.6f, -1.0f}) * Mat4::RotateY(0.4f) * Mat4::Scale({0.6f,0.6f,0.6f})},
        {Mat4::Translate({ 2.2f, 0.6f, -1.4f}) * Mat4::RotateY(-0.5f) * Mat4::Scale({0.6f,0.6f,0.6f})},
        {Mat4::Translate({-0.9f, 0.7f, 2.8f}) * Mat4::RotateY(0.2f) * Mat4::Scale({0.7f,0.7f,0.7f})},
    };

    struct Glass { Vec3 pos; float scale; float r, g, b, baseAlpha; };
    std::vector<Glass> glass = {
        {{-1.3f, 1.0f,  1.4f}, 1.1f, 0.85f, 0.20f, 0.20f, 0.28f},  // red
        {{ 0.0f, 1.1f,  0.4f}, 1.2f, 0.20f, 0.55f, 0.95f, 0.26f},  // blue
        {{ 1.4f, 1.0f,  1.2f}, 1.1f, 0.30f, 0.90f, 0.45f, 0.30f},  // green
        {{ 0.6f, 1.8f,  2.0f}, 0.9f, 0.95f, 0.85f, 0.25f, 0.24f},  // amber
    };

    const Vec3 eye{0.0f, 2.6f, 7.5f};
    const Vec3 center{0.0f, 1.0f, 0.0f};
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 0.6f, 0.0f};
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 lightEye = sc - lightDir * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-11.0f, 11.0f, -11.0f, 11.0f, 1.0f, 40.0f));
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

    // Sort glass back-to-front by distance from the camera eye (deterministic for the fixed camera).
    std::sort(glass.begin(), glass.end(), [&](const Glass& a, const Glass& b) {
        return math::length(a.pos - eye) > math::length(b.pos - eye);  // farthest first
    });

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
            cmd.BindVertexBuffer(cube.vertices());
            cmd.BindIndexBuffer(cube.indices());
            for (const auto& o : opaques) {
                cmd.PushConstants(o.model.m, sizeof(float) * 16);
                cmd.DrawIndexed(cube.indexCount());
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("scene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
            cmd.BindPipeline(*skyPipe);
            cmd.Draw(3);
            // Opaque pass (writes depth): ground + cubes.
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
            cmd.BindMaterial(*groundTex, *flatNormal);
            cmd.BindVertexBuffer(cube.vertices());
            cmd.BindIndexBuffer(cube.indices());
            for (const auto& o : opaques) {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = o.model.m[k];
                pc[16] = 0.1f; pc[17] = 0.6f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.DrawIndexed(cube.indexCount());
            }
            // Sorted translucent pass (alpha blend, depth-test, NO depth write).
            cmd.BindPipeline(*transparentPipeline);
            cmd.BindVertexBuffer(sphere.vertices());
            cmd.BindIndexBuffer(sphere.indices());
            for (const auto& g : glass) {
                Mat4 model = Mat4::Translate(g.pos) * Mat4::Scale({g.scale, g.scale, g.scale});
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = model.m[k];
                pc[16] = g.r; pc[17] = g.g; pc[18] = g.b; pc[19] = g.baseAlpha;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.DrawIndexed(sphere.indexCount());
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — %zu glass + %zu opaque\n",
                outPath, cw, ch, glass.size(), opaques.size());
    return 0;
}

// --- Cascaded shadow maps showcase (Slice AD). Mirrors the Vulkan --csm-shot path: a long ground
// plane receding from the camera with cubes + spheres at near/mid/far, a grazing directional light,
// 4 cascades fitted to the camera frustum's depth slices rendered into a single 4096 shadow ATLAS
// (2x2 tiles via SetViewport), and the scene shaded by lit_csm. Renders into an offscreen RT then
// blits to the swapchain (post) so the Metal capture matches the structure of the other goldens. ---
static int RunCsmShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace csm = hf::render::csm;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    // Metal NDC +Y up: flip the clip-space Y row of any projection built by the (Vulkan) math lib.
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // CSM FrameData layout (matches shaders/lit_csm.frag + shadow_csm.vert).
    struct CsmFrameData {
        float viewProj[16];
        float lightDir[4];
        float lightColor[4];
        float viewPos[4];
        float csmSplits[4];
        float cascadeVP[4][16];
        float camFwd[4];
        float camRight[4];
        float camUp[4];
        float skyParams[4];
        float csmAtlas[4];
    };
    static_assert(sizeof(CsmFrameData) == 464, "CSM FrameData layout");

    const int      kCascades    = 4;
    const uint32_t kAtlas       = 4096;
    const int      kTilesPerRow = 2;
    const uint32_t kTile        = kAtlas / kTilesPerRow;
    const float    kSplitLambda = 0.5f;
    const float    kShadowNear  = 0.5f, kShadowFar = 60.0f, kZPadNear = 12.0f;

    auto litVs   = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto csmFs   = loadMSL("lit_csm.frag.gen.metal", "csm_fragment");
    auto csmShVs = loadMSL("shadow_csm.vert.gen.metal", "csm_shadow_vertex");

    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = csmFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
    litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = csmShVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true; shDesc.usesFrameUniforms = true;
    shDesc.pushConstantSize = sizeof(float) * 32;
    auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false; postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowAtlas = device->CreateShadowMap(kAtlas);
    device->SetShadowMap(*shadowAtlas);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh plane  = scene::Mesh::Plane(*device);
    scene::Mesh cube   = scene::Mesh::Cube(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    const Mat4 groundModel = Mat4::Scale({60.0f, 1.0f, 80.0f});
    struct Caster { Mat4 model; const scene::Mesh* mesh; float metallic; float rough; };
    std::vector<Caster> casters;
    auto box = [&](float x, float z, float s, float rot) {
        casters.push_back({Mat4::Translate({x, 0.5f * s, z}) * Mat4::RotateY(rot) * Mat4::Scale({s,s,s}),
                           &cube, 0.0f, 0.8f});
    };
    auto ball = [&](float x, float z, float s) {
        casters.push_back({Mat4::Translate({x, 0.5f * s, z}) * Mat4::Scale({s,s,s}), &sphere, 0.1f, 0.4f});
    };
    box(-3.5f,  9.0f, 2.0f,  0.4f);
    ball( 3.5f,  8.0f, 2.4f);
    box( 0.0f, -2.0f, 2.6f, -0.6f);
    box(-5.5f, -4.0f, 2.0f,  0.9f);
    ball( 5.5f, -5.0f, 2.4f);
    box( 4.0f, -15.0f, 4.0f, 0.2f);
    box(-4.5f, -18.0f, 4.6f, 0.5f);
    box( 0.5f, -28.0f, 5.0f, 0.5f);

    const Vec3 eye{0.0f, 5.0f, 17.0f};
    const Vec3 center{0.0f, 1.2f, -10.0f};
    const float fovY = 1.04719755f;
    const float aspect = (float)W / (float)H;
    Vec3 fwd   = math::normalize(center - eye);
    Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
    Vec3 up3   = math::cross(right, fwd);
    const float tanHalf = std::tan(0.5f * fovY);
    Vec3 lightDir = math::normalize(Vec3{-0.65f, -0.5f, -0.35f});

    CsmFrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(fovY, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.viewProj[k] = vp.m[k];
        fd.lightDir[0]=lightDir.x; fd.lightDir[1]=lightDir.y; fd.lightDir[2]=lightDir.z;
        fd.lightColor[0]=1.5f; fd.lightColor[1]=1.45f; fd.lightColor[2]=1.35f; fd.lightColor[3]=1.0f;
        fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
        fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
        fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
        fd.camUp[0]=up3.x; fd.camUp[1]=up3.y; fd.camUp[2]=up3.z;
        fd.skyParams[0]=tanHalf; fd.skyParams[1]=aspect;
        fd.csmAtlas[0]=(float)kTilesPerRow; fd.csmAtlas[1]=1.0f/(float)kTilesPerRow;
        fd.csmAtlas[2]=(float)kCascades; fd.csmAtlas[3]=0.0f;
    }

    auto splits = csm::CsmSplits(kShadowNear, kShadowFar, kCascades, kSplitLambda);
    for (int c = 0; c < kCascades; ++c) fd.csmSplits[c] = splits[c];
    Mat4 cascadeVP[4];
    {
        float sliceNear = kShadowNear;
        for (int c = 0; c < kCascades; ++c) {
            float sliceFar = splits[c];
            auto corners = csm::FrustumSliceCornersWorld(eye, fwd, right, up3, tanHalf, aspect,
                                                         sliceNear, sliceFar);
            auto fit = csm::FitCascadeLightMatrix(corners, lightDir, kZPadNear);
            // Metal NDC +Y up: flip the cascade's clip-space Y. The lit_csm.frag samples with the
            // matching V-flip under HF_MSL_GEN, so RENDER and SAMPLE stay self-consistent.
            cascadeVP[c] = FlipProjY(fit.lightViewProj);
            for (int k = 0; k < 16; ++k) fd.cascadeVP[c][k] = cascadeVP[c].m[k];
            sliceNear = sliceFar;
        }
    }

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "csmAtlas", render::RgResourceKind::ShadowMap, *shadowAtlas);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

    graph.AddPass("csmShadow", {}, {rgShadow},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(CsmFrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*shadowPipeline);
            for (int c = 0; c < kCascades; ++c) {
                int col = c % kTilesPerRow, row = c / kTilesPerRow;
                cmd.SetViewport((int32_t)(col * kTile), (int32_t)(row * kTile), kTile, kTile);
                {
                    float pc[32];
                    for (int k=0;k<16;++k) pc[k] = cascadeVP[c].m[k];
                    for (int k=0;k<16;++k) pc[16+k] = groundModel.m[k];
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindVertexBuffer(plane.vertices());
                    cmd.BindIndexBuffer(plane.indices());
                    cmd.DrawIndexed(plane.indexCount());
                }
                for (const auto& ca : casters) {
                    float pc[32];
                    for (int k=0;k<16;++k) pc[k] = cascadeVP[c].m[k];
                    for (int k=0;k<16;++k) pc[16+k] = ca.model.m[k];
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindVertexBuffer(ca.mesh->vertices());
                    cmd.BindIndexBuffer(ca.mesh->indices());
                    cmd.DrawIndexed(ca.mesh->indexCount());
                }
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("csmScene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(CsmFrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.55f, 0.68f, 0.82f, 1});
            cmd.BindPipeline(*litPipeline);
            cmd.BindMaterial(*groundTex, *flatNormal);
            {
                float pc[20];
                for (int k=0;k<16;++k) pc[k] = groundModel.m[k];
                pc[16]=0.0f; pc[17]=0.9f; pc[18]=0.0f; pc[19]=0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            for (const auto& ca : casters) {
                float pc[20];
                for (int k=0;k<16;++k) pc[k] = ca.model.m[k];
                pc[16]=ca.metallic; pc[17]=ca.rough; pc[18]=0.0f; pc[19]=0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(ca.mesh->vertices());
                cmd.BindIndexBuffer(ca.mesh->indices());
                cmd.DrawIndexed(ca.mesh->indexCount());
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — CSM: %d cascades, %u atlas, %zu casters\n",
                outPath, cw, ch, kCascades, kAtlas, casters.size());
    return 0;
}

// --- Spot-light shadow showcase (Slice AE). Mirrors the Vulkan --spot-shot path: a ground plane +
// a small cluster of cubes/spheres lit primarily by ONE spot light angled down so it casts a clear
// cone of light with SHARP shadows of the objects within the cone, darkness outside. The spot's
// shadow is a single 2048 PERSPECTIVE map (FOV = 2*outerCone), rendered via the reused CSM depth-only
// caster (spotViewProj in the push constant), then the scene shaded by lit_spot (cone smoothstep +
// distance falloff + 3x3 PCF). Renders into an offscreen RT then blits (post) like the others. ----
static int RunSpotShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace spotns = hf::render::spot;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    // Metal NDC +Y up: flip the clip-space Y row of any projection built by the (Vulkan) math lib.
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Spot FrameData layout (matches shaders/lit_spot.frag). 304 bytes.
    struct SpotFrameData {
        float viewProj[16];
        float lightDir[4];
        float lightColor[4];
        float viewPos[4];
        float spotViewProj[16];
        float spotPos[4];
        float spotDir[4];
        float spotColor[4];
        float spotParams[4];
        float camFwd[4];
        float camRight[4];
        float camUp[4];
        float skyParams[4];
    };
    static_assert(sizeof(SpotFrameData) == 304, "Spot FrameData layout");

    const uint32_t kShadowSize = 2048;
    const float kInnerCone = 0.28f, kOuterCone = 0.40f;
    const float kSpotNear  = 0.5f,  kSpotRange = 34.0f;
    const Vec3  kSpotPos{0.0f, 13.0f, 5.0f};
    const Vec3  kSpotTarget{0.0f, 0.0f, -2.0f};
    const Vec3  kSpotDir = math::normalize(kSpotTarget - kSpotPos);

    auto litVs   = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto spotFs  = loadMSL("lit_spot.frag.gen.metal", "spot_fragment");
    auto shVs    = loadMSL("shadow_csm.vert.gen.metal", "csm_shadow_vertex");

    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = spotFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
    litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true; shDesc.usesFrameUniforms = true;
    shDesc.pushConstantSize = sizeof(float) * 32;
    auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false; postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(kShadowSize);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh plane  = scene::Mesh::Plane(*device);
    scene::Mesh cube   = scene::Mesh::Cube(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    const Mat4 groundModel = Mat4::Scale({40.0f, 1.0f, 40.0f});
    struct Caster { Mat4 model; const scene::Mesh* mesh; float metallic; float rough; };
    std::vector<Caster> casters;
    auto box = [&](float x, float z, float s, float rot) {
        casters.push_back({Mat4::Translate({x, 0.5f * s, z}) * Mat4::RotateY(rot) * Mat4::Scale({s,s,s}),
                           &cube, 0.0f, 0.8f});
    };
    auto ball = [&](float x, float z, float s) {
        casters.push_back({Mat4::Translate({x, 0.5f * s, z}) * Mat4::Scale({s,s,s}), &sphere, 0.05f, 0.45f});
    };
    box(-2.6f,  0.0f, 2.0f,  0.5f);
    ball( 2.4f, -1.0f, 2.2f);
    box( 0.2f, -4.0f, 2.4f, -0.3f);
    ball(-1.4f,  2.6f, 1.6f);

    const Vec3 eye{0.0f, 7.0f, 16.0f};
    const Vec3 center{0.0f, 0.8f, -2.0f};
    const float fovY = 1.04719755f;
    const float aspect = (float)W / (float)H;
    Vec3 fwd   = math::normalize(center - eye);
    Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
    Vec3 up3   = math::cross(right, fwd);
    const float tanHalf = std::tan(0.5f * fovY);

    // Spot perspective light matrix; Metal flips clip-space Y so the lit_spot V-flip matches.
    Mat4 spotVP = FlipProjY(spotns::SpotViewProj(kSpotPos, kSpotDir, kOuterCone, kSpotNear, kSpotRange));

    SpotFrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(fovY, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.viewProj[k] = vp.m[k];
        for (int k = 0; k < 16; ++k) fd.spotViewProj[k] = spotVP.m[k];
        Vec3 ld = math::normalize(Vec3{-0.4f, -0.85f, -0.3f});
        fd.lightDir[0]=ld.x; fd.lightDir[1]=ld.y; fd.lightDir[2]=ld.z;
        fd.lightColor[0]=0.20f; fd.lightColor[1]=0.21f; fd.lightColor[2]=0.24f; fd.lightColor[3]=1.0f;
        fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
        fd.spotPos[0]=kSpotPos.x; fd.spotPos[1]=kSpotPos.y; fd.spotPos[2]=kSpotPos.z; fd.spotPos[3]=1.0f;
        fd.spotDir[0]=kSpotDir.x; fd.spotDir[1]=kSpotDir.y; fd.spotDir[2]=kSpotDir.z;
        fd.spotColor[0]=1.0f; fd.spotColor[1]=0.96f; fd.spotColor[2]=0.85f; fd.spotColor[3]=1.0f;
        fd.spotParams[0]=std::cos(kInnerCone); fd.spotParams[1]=std::cos(kOuterCone);
        fd.spotParams[2]=kSpotRange; fd.spotParams[3]=9.0f;
        fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
        fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
        fd.camUp[0]=up3.x; fd.camUp[1]=up3.y; fd.camUp[2]=up3.z;
        fd.skyParams[0]=tanHalf; fd.skyParams[1]=aspect;
    }

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "spotShadow", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

    graph.AddPass("spotShadow", {}, {rgShadow},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(SpotFrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*shadowPipeline);
            {
                float pc[32];
                for (int k=0;k<16;++k) pc[k] = spotVP.m[k];
                for (int k=0;k<16;++k) pc[16+k] = groundModel.m[k];
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            for (const auto& ca : casters) {
                float pc[32];
                for (int k=0;k<16;++k) pc[k] = spotVP.m[k];
                for (int k=0;k<16;++k) pc[16+k] = ca.model.m[k];
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(ca.mesh->vertices());
                cmd.BindIndexBuffer(ca.mesh->indices());
                cmd.DrawIndexed(ca.mesh->indexCount());
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("spotScene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(SpotFrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.04f, 0.05f, 0.07f, 1});
            cmd.BindPipeline(*litPipeline);
            cmd.BindMaterial(*groundTex, *flatNormal);
            {
                float pc[20];
                for (int k=0;k<16;++k) pc[k] = groundModel.m[k];
                pc[16]=0.0f; pc[17]=0.9f; pc[18]=0.0f; pc[19]=0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            for (const auto& ca : casters) {
                float pc[20];
                for (int k=0;k<16;++k) pc[k] = ca.model.m[k];
                pc[16]=ca.metallic; pc[17]=ca.rough; pc[18]=0.0f; pc[19]=0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(ca.mesh->vertices());
                cmd.BindIndexBuffer(ca.mesh->indices());
                cmd.DrawIndexed(ca.mesh->indexCount());
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — spot light: 1 perspective %u shadow map, %zu casters\n",
                outPath, cw, ch, kShadowSize, casters.size());
    return 0;
}

// --- Omnidirectional point-light shadow showcase (Slice AF). Mirrors the Vulkan --point-shadow-shot
// path: a point light hovers among a RING of cubes/spheres on a ground plane with a back wall,
// casting shadows RADIALLY OUTWARD in every direction. The scene is rendered from the light through
// 6 cube faces (Perspective 90deg, aspect 1) into ONE 3072 shadow ATLAS (3x2 grid of 1024 tiles via
// SetViewport per face), then shaded by lit_point (dominant-axis face select + per-face atlas PCF +
// distance falloff). Renders into an offscreen RT then blits (post) like the others. -------------
static int RunPointShadowShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace ptns = hf::render::point_shadow;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Point FrameData layout (matches shaders/lit_point.frag). 608 bytes.
    struct PointFrameData {
        float viewProj[16];
        float lightDir[4];
        float lightColor[4];
        float viewPos[4];
        float faceVP[6][16];
        float ptPos[4];
        float ptColor[4];
        float atlasParams[4];
        float camFwd[4];
        float camRight[4];
        float camUp[4];
        float skyParams[4];
    };
    static_assert(sizeof(PointFrameData) == 608, "Point FrameData layout");

    const uint32_t kAtlas    = 3072;
    const uint32_t kTile     = 1024;
    const int   kTilesPerRow = 3;
    const int   kTilesPerCol = 2;
    const float kPtNear  = 0.1f,  kPtRange = 30.0f;
    const Vec3  kPtPos{0.0f, 3.0f, 0.0f};

    auto litVs   = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto ptFs    = loadMSL("lit_point.frag.gen.metal", "point_fragment");
    auto shVs    = loadMSL("shadow_csm.vert.gen.metal", "csm_shadow_vertex");

    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = ptFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
    litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true; shDesc.usesFrameUniforms = true;
    shDesc.pushConstantSize = sizeof(float) * 32;
    auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false; postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowAtlas = device->CreateShadowMap(kAtlas);
    device->SetShadowMap(*shadowAtlas);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh plane  = scene::Mesh::Plane(*device);
    scene::Mesh cube   = scene::Mesh::Cube(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    const Mat4 groundModel = Mat4::Scale({40.0f, 1.0f, 40.0f});
    const Mat4 wallModel = Mat4::Translate({0.0f, 4.0f, -9.0f}) * Mat4::Scale({12.0f, 8.0f, 0.5f});
    struct Caster { Mat4 model; const scene::Mesh* mesh; float metallic; float rough; };
    std::vector<Caster> casters;
    auto box = [&](float x, float z, float s, float rot) {
        casters.push_back({Mat4::Translate({x, 0.5f * s, z}) * Mat4::RotateY(rot) * Mat4::Scale({s,s,s}),
                           &cube, 0.0f, 0.8f});
    };
    auto ball = [&](float x, float z, float s) {
        casters.push_back({Mat4::Translate({x, 0.5f * s, z}) * Mat4::Scale({s,s,s}), &sphere, 0.05f, 0.45f});
    };
    box( 4.5f,  0.0f, 1.2f,  0.4f);
    box(-4.5f,  0.0f, 1.2f, -0.4f);
    ball( 0.0f,  4.5f, 1.3f);
    ball( 0.0f, -4.5f, 1.3f);
    box( 3.2f,  3.2f, 1.1f,  0.8f);
    box(-3.2f,  3.2f, 1.1f, -0.8f);
    ball( 3.2f, -3.2f, 1.1f);
    ball(-3.2f, -3.2f, 1.1f);

    const Vec3 eye{0.0f, 9.0f, 13.0f};
    const Vec3 center{0.0f, 1.0f, 0.0f};
    const float fovY = 1.04719755f;
    const float aspect = (float)W / (float)H;
    Vec3 fwd   = math::normalize(center - eye);
    Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
    Vec3 up3   = math::cross(right, fwd);
    const float tanHalf = std::tan(0.5f * fovY);

    // 6 cube-face view-projs; Metal flips clip-space Y so the lit_point V-flip matches.
    std::array<Mat4, ptns::kFaces> faceVPs;
    for (int fi = 0; fi < ptns::kFaces; ++fi)
        faceVPs[fi] = FlipProjY(ptns::FaceViewProj(kPtPos, fi, kPtNear, kPtRange));

    PointFrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(fovY, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.viewProj[k] = vp.m[k];
        for (int fi = 0; fi < ptns::kFaces; ++fi)
            for (int k = 0; k < 16; ++k) fd.faceVP[fi][k] = faceVPs[fi].m[k];
        Vec3 ld = math::normalize(Vec3{-0.3f, -0.9f, -0.25f});
        fd.lightDir[0]=ld.x; fd.lightDir[1]=ld.y; fd.lightDir[2]=ld.z;
        fd.lightColor[0]=0.13f; fd.lightColor[1]=0.14f; fd.lightColor[2]=0.17f; fd.lightColor[3]=1.0f;
        fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
        fd.ptPos[0]=kPtPos.x; fd.ptPos[1]=kPtPos.y; fd.ptPos[2]=kPtPos.z; fd.ptPos[3]=kPtRange;
        fd.ptColor[0]=1.0f; fd.ptColor[1]=0.93f; fd.ptColor[2]=0.82f; fd.ptColor[3]=14.0f;
        fd.atlasParams[0]=(float)kTilesPerRow; fd.atlasParams[1]=(float)kTilesPerCol;
        fd.atlasParams[2]=1.0f/(float)kAtlas;   fd.atlasParams[3]=kPtNear;
        fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
        fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
        fd.camUp[0]=up3.x; fd.camUp[1]=up3.y; fd.camUp[2]=up3.z;
        fd.skyParams[0]=tanHalf; fd.skyParams[1]=aspect;
    }

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "pointAtlas", render::RgResourceKind::ShadowMap, *shadowAtlas);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

    graph.AddPass("pointShadow", {}, {rgShadow},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(PointFrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*shadowPipeline);
            for (int face = 0; face < ptns::kFaces; ++face) {
                auto tile = ptns::FaceTile(face);
                cmd.SetViewport((int32_t)(tile.col * kTile), (int32_t)(tile.row * kTile), kTile, kTile);
                auto drawOne = [&](const Mat4& model, const scene::Mesh& mesh) {
                    float pc[32];
                    for (int k=0;k<16;++k) pc[k] = faceVPs[face].m[k];
                    for (int k=0;k<16;++k) pc[16+k] = model.m[k];
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindVertexBuffer(mesh.vertices());
                    cmd.BindIndexBuffer(mesh.indices());
                    cmd.DrawIndexed(mesh.indexCount());
                };
                drawOne(groundModel, plane);
                drawOne(wallModel, cube);
                for (const auto& ca : casters) drawOne(ca.model, *ca.mesh);
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("pointScene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(PointFrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.03f, 0.04f, 0.06f, 1});
            cmd.BindPipeline(*litPipeline);
            auto drawLit = [&](const Mat4& model, const scene::Mesh& mesh, float metallic, float rough) {
                float pc[20];
                for (int k=0;k<16;++k) pc[k] = model.m[k];
                pc[16]=metallic; pc[17]=rough; pc[18]=0.0f; pc[19]=0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(mesh.vertices());
                cmd.BindIndexBuffer(mesh.indices());
                cmd.DrawIndexed(mesh.indexCount());
            };
            cmd.BindMaterial(*groundTex, *flatNormal);
            drawLit(groundModel, plane, 0.0f, 0.9f);
            drawLit(wallModel, cube, 0.0f, 0.9f);
            for (const auto& ca : casters) drawLit(ca.model, *ca.mesh, ca.metallic, ca.rough);
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — point light: 6-face cube, %u atlas (%dx%d tiles), %zu casters\n",
                outPath, cw, ch, kAtlas, kTilesPerRow, kTilesPerCol, casters.size());
    return 0;
}

// --clustered <out.png>: clustered / Forward+ lighting showcase (Slice AG). A ground plane + a few
// raised objects lit by 192 deterministic point lights culled CPU-side into a 16x9x24 cluster grid
// (render::clustered) -> three storage buffers; the lit_clustered fragment iterates each fragment's
// cluster's lights. Mirrors the Vulkan --clustered-shot exactly (same lights/camera/grid). Renders to
// an offscreen RT then posts to the swapchain (matching the other Metal showcases).
static int RunClusteredShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace cl = hf::render::clustered;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Clustered FrameData layout (matches shaders/lit_clustered.frag). 224 bytes.
    struct ClusteredFrameData {
        float viewProj[16];
        float view[16];
        float lightDir[4];
        float lightColor[4];
        float viewPos[4];
        float clusterParams[4];
        float clusterParams2[4];
        float clusterParams3[4];
    };
    static_assert(sizeof(ClusteredFrameData) == 224, "Clustered FrameData layout");

    // HF_CLUSTERED_BRUTEFORCE collapses the grid to 1x1x1 -> the SAME shader loops ALL lights, a
    // brute-force reference for confirming the cluster culling is correct (must match the 16x9x24).
    const bool bruteForce = (std::getenv("HF_CLUSTERED_BRUTEFORCE") != nullptr);
    const int   CX = bruteForce ? 1 : 16;
    const int   CY = bruteForce ? 1 : 9;
    const int   CZ = bruteForce ? 1 : 24;
    const float kNear = 0.5f, kFar = 90.0f;
    const float fovY = 1.04719755f;
    const float aspect = (float)W / (float)H;

    const Vec3 eye{0.0f, 16.0f, 26.0f};
    const Vec3 center{0.0f, 0.0f, -2.0f};
    Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
    // The cluster math uses VIEW space (unaffected by the clip-space Y flip); only the rendered
    // viewProj is flipped for Metal. tanX/tanY read off the (unflipped) proj are identical to the
    // flipped one (m[0] unchanged; |m[5]| unchanged).
    Mat4 proj = Mat4::Perspective(fovY, aspect, kNear, kFar);
    Mat4 vp = FlipProjY(proj) * view;
    cl::Grid grid = cl::MakeGrid(proj, kNear, kFar, (float)W, (float)H, CX, CY, CZ);

    // 192 deterministic point lights (16x12 grid) — IDENTICAL to the Vulkan --clustered-shot.
    const int LX = 16, LZ = 12;
    const int kNumLights = LX * LZ;
    const float spanX = 34.0f, spanZ = 26.0f;
    const float lightY = 1.4f;
    std::vector<cl::Light> viewLights;
    viewLights.reserve(kNumLights);
    for (int iz = 0; iz < LZ; ++iz) {
        for (int ix = 0; ix < LX; ++ix) {
            int idx = iz * LX + ix;
            float fx = ((float)ix / (float)(LX - 1) - 0.5f) * spanX;
            float fz = ((float)iz / (float)(LZ - 1) - 0.5f) * spanZ - 2.0f;
            static const float palette[6][3] = {
                {1.00f, 0.18f, 0.20f}, {0.20f, 1.00f, 0.30f}, {0.25f, 0.40f, 1.00f},
                {1.00f, 0.80f, 0.15f}, {0.90f, 0.20f, 1.00f}, {0.15f, 0.95f, 0.95f},
            };
            const float* c = palette[(ix * 2 + iz * 3) % 6];
            float radius = 4.0f + ((idx * 7) % 6) * 0.5f;
            cl::Light L{};
            float vw = 0.0f;
            L.viewPos = math::MulPointDivide(view, Vec3{fx, lightY, fz}, vw);  // world -> view
            L.radius = radius;
            L.color = {c[0], c[1], c[2]};
            L.intensity = 2.6f;
            viewLights.push_back(L);
        }
    }
    cl::ClusterBuffers cb = cl::BuildClusters(grid, viewLights);
    if (cb.lightIndices.empty()) cb.lightIndices.push_back(0u);

    rhi::BufferDesc clusterDesc{cb.clusters.size() * sizeof(cl::GpuCluster), cb.clusters.data(),
                                rhi::BufferUsage::Storage};
    auto clusterBuf = device->CreateBuffer(clusterDesc);
    rhi::BufferDesc indexDesc{cb.lightIndices.size() * sizeof(uint32_t), cb.lightIndices.data(),
                              rhi::BufferUsage::Storage};
    auto indexBuf = device->CreateBuffer(indexDesc);
    rhi::BufferDesc lightDesc{cb.lights.size() * sizeof(cl::GpuLight), cb.lights.data(),
                              rhi::BufferUsage::Storage};
    auto lightBuf = device->CreateBuffer(lightDesc);

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto cluFs = loadMSL("lit_clustered.frag.gen.metal", "clustered_fragment");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = cluFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
    litDesc.usesLightClusters = true;
    litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false; postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);

    std::vector<uint8_t> checker = MakeCheckerboard();
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh plane  = scene::Mesh::Plane(*device);
    scene::Mesh cube   = scene::Mesh::Cube(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    const Mat4 groundModel = Mat4::Scale({26.0f, 1.0f, 20.0f});
    struct Obj { Mat4 model; const scene::Mesh* mesh; float metallic; float rough; };
    std::vector<Obj> objs;
    for (int k = 0; k < 7; ++k) {
        float ox = ((k % 4) - 1.5f) * 7.0f;
        float oz = ((k / 4) - 0.5f) * 8.0f - 2.0f;
        float s = 1.2f + (k % 3) * 0.4f;
        if (k % 2 == 0)
            objs.push_back({Mat4::Translate({ox, 0.5f * s, oz}) * Mat4::RotateY(0.3f * k)
                            * Mat4::Scale({s, s, s}), &cube, 0.0f, 0.55f});
        else
            objs.push_back({Mat4::Translate({ox, 0.5f * s, oz}) * Mat4::Scale({s, s, s}),
                            &sphere, 0.05f, 0.4f});
    }

    ClusteredFrameData fd{};
    {
        for (int k = 0; k < 16; ++k) fd.viewProj[k] = vp.m[k];
        for (int k = 0; k < 16; ++k) fd.view[k] = view.m[k];
        Vec3 ld = math::normalize(Vec3{-0.3f, -0.9f, -0.25f});
        fd.lightDir[0]=ld.x; fd.lightDir[1]=ld.y; fd.lightDir[2]=ld.z;
        fd.lightColor[0]=0.05f; fd.lightColor[1]=0.05f; fd.lightColor[2]=0.06f; fd.lightColor[3]=1.0f;
        fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
        fd.clusterParams[0]=(float)CX; fd.clusterParams[1]=(float)CY;
        fd.clusterParams[2]=(float)CZ; fd.clusterParams[3]=kNear;
        fd.clusterParams2[0]=kFar; fd.clusterParams2[1]=(float)W; fd.clusterParams2[2]=(float)H;
        fd.clusterParams2[3]=grid.tanX;
        fd.clusterParams3[0]=grid.tanY;
    }

    render::RenderGraph graph;
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

    graph.AddPass("clusteredScene", {}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(ClusteredFrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.01f, 0.01f, 0.02f, 1});
            cmd.BindPipeline(*litPipeline);
            cmd.BindLightClusters(*clusterBuf, *indexBuf, *lightBuf);
            auto drawLit = [&](const Mat4& model, const scene::Mesh& mesh, float metallic, float rough) {
                float pc[20];
                for (int k=0;k<16;++k) pc[k] = model.m[k];
                pc[16]=metallic; pc[17]=rough; pc[18]=0.0f; pc[19]=0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(mesh.vertices());
                cmd.BindIndexBuffer(mesh.indices());
                cmd.DrawIndexed(mesh.indexCount());
            };
            drawLit(groundModel, plane, 0.0f, 0.7f);
            for (const auto& o : objs) drawLit(o.model, *o.mesh, o.metallic, o.rough);
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
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — clustered Forward+: %d point lights, %dx%dx%d grid, "
                "%zu light-index entries\n", outPath, cw, ch, kNumLights, CX, CY, CZ,
                cb.lightIndices.size());
    return 0;
}

// --probe <out.png>: reflection + irradiance PROBE showcase (Slice AK). Mirrors the Vulkan
// --probe-shot exactly: a Cornell-style box room (red -X / green +X / neutral) is baked from a fixed
// probe at the room center into ONE RGBA16F cube atlas (reflection block + cosine-convolved
// irradiance block), then a metallic sphere + a matte box are shaded by lit_probe sampling that atlas
// as LOCAL GI. The 6 face VPs are FlipProjY'd for Metal (the shaders apply the matching HF_MSL_GEN
// V-flip); the irradiance reconstruction inverts the FLIPPED face VP so it lines up with the sample.
static int RunProbeShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace prb = hf::render::probe;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    struct ProbeFrameData {
        float viewProj[16]; float lightDir[4]; float lightColor[4]; float viewPos[4];
        float faceVP[6][16]; float probePos[4]; float atlasParams[4]; float atlasParams2[4];
        float camFwd[4]; float camRight[4]; float camUp[4]; float skyParams[4]; float pad0[4];
    };
    static_assert(sizeof(ProbeFrameData) == 624, "Probe FrameData layout");

    const Vec3 kProbePos{0.0f, 2.0f, 0.0f};
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float aspect = (float)W / (float)H;

    auto litVs   = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto bakeVs  = loadMSL("probe_bake.vert.gen.metal", "probe_bake_vertex");
    auto bakeFs  = loadMSL("probe_bake.frag.gen.metal", "probe_bake_fragment");
    auto probeFs = loadMSL("lit_probe.frag.gen.metal", "probe_fragment");
    auto postVs  = loadMSL("post.vert.gen.metal", "post_vertex");
    auto blitFs  = loadMSL("probe_blit.frag.gen.metal", "probe_blit_fragment");
    auto irrFs   = loadMSL("probe_irradiance.frag.gen.metal", "probe_irradiance_fragment");
    auto postFs  = loadMSL("post.frag.gen.metal", "post_fragment");

    rhi::GraphicsPipelineDesc bakeDesc;
    bakeDesc.vertex = bakeVs.get(); bakeDesc.fragment = bakeFs.get();
    bakeDesc.vertexLayout = scene::MeshVertexLayout();
    bakeDesc.colorFormat = kHdr;
    bakeDesc.depthTest = true; bakeDesc.usesFrameUniforms = false; bakeDesc.usesTexture = true;
    bakeDesc.pushConstantSize = sizeof(float) * 32;
    auto bakePipeline = device->CreateGraphicsPipeline(bakeDesc);

    rhi::GraphicsPipelineDesc blitDesc;
    blitDesc.vertex = postVs.get(); blitDesc.fragment = blitFs.get();
    blitDesc.colorFormat = kHdr;
    blitDesc.depthTest = false; blitDesc.usesTexture = true; blitDesc.fullscreen = true;
    blitDesc.fragmentPushConstants = true; blitDesc.pushConstantSize = sizeof(float) * 4;
    auto blitPipeline = device->CreateGraphicsPipeline(blitDesc);

    rhi::GraphicsPipelineDesc irrDesc;
    irrDesc.vertex = postVs.get(); irrDesc.fragment = irrFs.get();
    irrDesc.colorFormat = kHdr;
    irrDesc.depthTest = false; irrDesc.usesFrameUniforms = true; irrDesc.usesTexture = true;
    irrDesc.fullscreen = true; irrDesc.fragmentPushConstants = true;
    irrDesc.pushConstantSize = sizeof(float) * 20;
    auto irrPipeline = device->CreateGraphicsPipeline(irrDesc);

    rhi::GraphicsPipelineDesc probeDesc;
    probeDesc.vertex = litVs.get(); probeDesc.fragment = probeFs.get();
    probeDesc.vertexLayout = scene::MeshVertexLayout();
    probeDesc.colorFormat = kHdr;
    probeDesc.depthTest = true; probeDesc.usesFrameUniforms = true; probeDesc.usesTexture = true;
    probeDesc.usesEnvironment = true;
    probeDesc.pushConstantSize = sizeof(float) * 20;
    auto probePipeline = device->CreateGraphicsPipeline(probeDesc);

    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false; postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto reflRT     = device->CreateRenderTarget(prb::kAtlasW, prb::kReflBlockH, kHdr);
    auto probeAtlas = device->CreateRenderTarget(prb::kAtlasW, prb::kAtlasH, kHdr);
    auto sceneRT    = device->CreateRenderTarget(W, H, kHdr);
    auto dummyShadow = device->CreateShadowMap(64);
    device->SetShadowMap(*dummyShadow);

    const uint8_t whitePx[4] = {255, 255, 255, 255};
    auto whiteTex = device->CreateTexture({1, 1, rhi::Format::RGBA8_UNorm, whitePx, sizeof(whitePx)});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture({1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    scene::Mesh cube   = scene::Mesh::Cube(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    auto colorTex = [&](float r, float g, float b) {
        uint8_t px[4] = {(uint8_t)(r * 255), (uint8_t)(g * 255), (uint8_t)(b * 255), 255};
        return device->CreateTexture({1, 1, rhi::Format::RGBA8_UNorm, px, sizeof(px)});
    };
    auto redTex     = colorTex(0.85f, 0.07f, 0.07f);
    auto greenTex   = colorTex(0.10f, 0.75f, 0.12f);
    auto neutralTex = colorTex(0.78f, 0.78f, 0.78f);

    const float Rh = 6.0f, Tk = 0.2f;
    struct Wall { Mat4 model; rhi::ITexture* tex; };
    std::vector<Wall> walls = {
        {Mat4::Translate({-Rh, 2.0f, 0.0f}) * Mat4::Scale({Tk, 2*Rh, 2*Rh}), redTex.get()},
        {Mat4::Translate({ Rh, 2.0f, 0.0f}) * Mat4::Scale({Tk, 2*Rh, 2*Rh}), greenTex.get()},
        {Mat4::Translate({0.0f, 2.0f - Rh, 0.0f}) * Mat4::Scale({2*Rh, Tk, 2*Rh}), neutralTex.get()},
        {Mat4::Translate({0.0f, 2.0f + Rh, 0.0f}) * Mat4::Scale({2*Rh, Tk, 2*Rh}), neutralTex.get()},
        {Mat4::Translate({0.0f, 2.0f, -Rh}) * Mat4::Scale({2*Rh, 2*Rh, Tk}), neutralTex.get()},
        {Mat4::Translate({0.0f, 2.0f,  Rh}) * Mat4::Scale({2*Rh, 2*Rh, Tk}), neutralTex.get()},
    };
    Mat4 sphereModel = Mat4::Translate({-1.9f, 1.4f, 0.8f}) * Mat4::Scale({1.7f, 1.7f, 1.7f});
    Mat4 boxModel    = Mat4::Translate({ 2.0f, 1.2f, 0.6f}) * Mat4::RotateY(0.785f)
                         * Mat4::Scale({1.5f, 2.4f, 1.5f});

    const Vec3 eye{0.0f, 2.4f, 9.0f};
    const Vec3 center{0.0f, 1.6f, 0.0f};
    const float fovY = 1.04719755f;
    Vec3 fwd   = math::normalize(center - eye);
    Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
    Vec3 up3   = math::cross(right, fwd);

    // 6 face VPs, FlipProjY'd for Metal (the lit_probe / irradiance / bake shaders V-flip on sample).
    std::array<Mat4, prb::kFaces> faceVPs;
    for (int fi = 0; fi < prb::kFaces; ++fi)
        faceVPs[fi] = FlipProjY(prb::FaceViewProj(kProbePos, fi));

    ProbeFrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(fovY, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.viewProj[k] = vp.m[k];
        for (int fi = 0; fi < prb::kFaces; ++fi)
            for (int k = 0; k < 16; ++k) fd.faceVP[fi][k] = faceVPs[fi].m[k];
        Vec3 ld = math::normalize(Vec3{-0.3f, -0.85f, -0.35f});
        fd.lightDir[0]=ld.x; fd.lightDir[1]=ld.y; fd.lightDir[2]=ld.z;
        fd.lightColor[0]=0.55f; fd.lightColor[1]=0.55f; fd.lightColor[2]=0.58f; fd.lightColor[3]=1.0f;
        fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
        fd.probePos[0]=kProbePos.x; fd.probePos[1]=kProbePos.y; fd.probePos[2]=kProbePos.z; fd.probePos[3]=1.0f;
        fd.atlasParams[0]=(float)prb::kReflTile/(float)prb::kAtlasW;
        fd.atlasParams[1]=(float)prb::kReflTile/(float)prb::kAtlasH;
        fd.atlasParams[2]=(float)prb::kIrrTile/(float)prb::kAtlasW;
        fd.atlasParams[3]=(float)prb::kIrrTile/(float)prb::kAtlasH;
        fd.atlasParams2[0]=(float)prb::kReflBlockH/(float)prb::kAtlasH;
        fd.atlasParams2[1]=1.0f/(float)prb::kAtlasW;
        fd.atlasParams2[2]=1.0f/(float)prb::kAtlasH;
        fd.atlasParams2[3]=(float)prb::kTilesPerRow;
        fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
        fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
        fd.camUp[0]=up3.x; fd.camUp[1]=up3.y; fd.camUp[2]=up3.z;
    }

    render::RenderGraph graph;
    render::RgResource rgRefl  = graph.ImportTarget("reflRT", render::RgResourceKind::SceneColor, *reflRT);
    render::RgResource rgAtlas = graph.ImportTarget("probeAtlas", render::RgResourceKind::SceneColor, *probeAtlas);
    render::RgResource rgScene = graph.ImportTarget("sceneColor", render::RgResourceKind::SceneColor, *sceneRT);
    render::RgResource rgSwap  = graph.ImportSwapchain("swapchain");

    graph.AddPass("probeBake", {}, {rgRefl},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.03f, 1});
            cmd.BindPipeline(*bakePipeline);
            for (int face = 0; face < prb::kFaces; ++face) {
                auto tile = prb::FaceTile(face);
                cmd.SetViewport((int32_t)(tile.col * prb::kReflTile),
                                (int32_t)(tile.row * prb::kReflTile),
                                prb::kReflTile, prb::kReflTile);
                for (const auto& wl : walls) {
                    float pc[32];
                    for (int k = 0; k < 16; ++k) pc[k]      = faceVPs[face].m[k];
                    for (int k = 0; k < 16; ++k) pc[16 + k] = wl.model.m[k];
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindTexture(*wl.tex);
                    cmd.BindVertexBuffer(cube.vertices());
                    cmd.BindIndexBuffer(cube.indices());
                    cmd.DrawIndexed(cube.indexCount());
                }
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("probeCompose", {rgRefl}, {rgAtlas},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*blitPipeline);
            cmd.BindTexture(*reflRT);
            for (int face = 0; face < prb::kFaces; ++face) {
                auto tile = prb::FaceTile(face);
                cmd.SetViewport((int32_t)(tile.col * prb::kReflTile),
                                (int32_t)(tile.row * prb::kReflTile),
                                prb::kReflTile, prb::kReflTile);
                float srcRect[4] = {
                    (float)tile.col / (float)prb::kTilesPerRow,
                    (float)tile.row / (float)prb::kTilesPerCol,
                    1.0f / (float)prb::kTilesPerRow,
                    1.0f / (float)prb::kTilesPerCol,
                };
                cmd.PushConstants(srcRect, sizeof(srcRect));
                cmd.Draw(3);
            }
            cmd.BindPipeline(*irrPipeline);
            dev.SetFrameUniforms(&fd, sizeof(ProbeFrameData));
            cmd.BindTexture(*reflRT);
            for (int face = 0; face < prb::kFaces; ++face) {
                auto tile = prb::FaceTile(face);
                cmd.SetViewport((int32_t)(tile.col * prb::kIrrTile),
                                (int32_t)(prb::kReflBlockH + tile.row * prb::kIrrTile),
                                prb::kIrrTile, prb::kIrrTile);
                Mat4 invVP = faceVPs[face].Inverse();
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = invVP.m[k];
                pc[16]=kProbePos.x; pc[17]=kProbePos.y; pc[18]=kProbePos.z; pc[19]=1.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.Draw(3);
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("probeScene", {rgAtlas}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(ProbeFrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.03f, 1});
            cmd.BindPipeline(*probePipeline);
            cmd.BindReflectionProbe(*probeAtlas);
            auto drawProbe = [&](const Mat4& model, const scene::Mesh& mesh, rhi::ITexture& tex,
                                 float metallic, float rough) {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = model.m[k];
                pc[16]=metallic; pc[17]=rough; pc[18]=0.0f; pc[19]=0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(tex, *flatNormal);
                cmd.BindReflectionProbe(*probeAtlas);
                cmd.BindVertexBuffer(mesh.vertices());
                cmd.BindIndexBuffer(mesh.indices());
                cmd.DrawIndexed(mesh.indexCount());
            };
            for (size_t wi = 0; wi < walls.size(); ++wi)
                if (wi != 5) drawProbe(walls[wi].model, cube, *walls[wi].tex, 0.0f, 0.95f);
            drawProbe(sphereModel, sphere, *whiteTex, 1.0f, 0.08f);
            drawProbe(boxModel,    cube,   *whiteTex, 0.0f, 0.9f);
            cmd.EndRenderPass();
        });

    graph.AddPass("post", {rgScene}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*postPipe);
            cmd.BindTexture(*sceneRT);
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — reflection+irradiance probe, %dx%d atlas\n",
                outPath, cw, ch, prb::kAtlasW, prb::kAtlasH);
    return 0;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        // --clustered <out.png>: clustered / Forward+ lighting showcase (Slice AG).
        if (argc > 1 && std::strcmp(argv[1], "--clustered") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_clustered.png";
            try { return RunClusteredShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --point-shadow <out.png>: omnidirectional point-light 6-face cube shadow showcase (Slice AF).
        if (argc > 1 && std::strcmp(argv[1], "--point-shadow") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_point_shadow.png";
            try { return RunPointShadowShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --spot <out.png>: spot-light perspective shadow showcase (Slice AE).
        if (argc > 1 && std::strcmp(argv[1], "--spot") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_spot.png";
            try { return RunSpotShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --csm <out.png>: cascaded shadow maps showcase (Slice AD).
        if (argc > 1 && std::strcmp(argv[1], "--csm") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_csm.png";
            try { return RunCsmShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --instanced <out.png>: render the GPU-instanced showcase (Slice Q).
        if (argc > 1 && std::strcmp(argv[1], "--instanced") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_instanced.png";
            try { return RunInstancedShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --physics <out.png>: render the rigid-body physics showcase (Slice S) — a settled
        // square-pyramid pile of spheres stepped to rest by the pure-C++ physics::World.
        if (argc > 1 && std::strcmp(argv[1], "--physics") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_physics.png";
            try { return RunPhysicsShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --game <out.png>: render the playable game sample (Slice AX) — the deterministic roll-a-ball
        // game run over the scripted track, captured at a fixed mid-track step (player + remaining
        // pickup, lit + shadowed). Mirrors the Vulkan --game-shot path; new golden
        // tests/golden/metal/game.png.
        if (argc > 1 && std::strcmp(argv[1], "--game") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_game.png";
            try { return RunGameShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --hud <out.png>: render the text / HUD showcase (Slice BA) — the lit + shadowed scene plus a
        // deterministic screen-space HUD text overlay ("HAZARD FORGE" + "SCORE: 0" + a stat line) drawn
        // alpha-blended over the scene. Mirrors the Vulkan --hud-shot; new golden tests/golden/metal/hud.png.
        if (argc > 1 && std::strcmp(argv[1], "--hud") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_hud.png";
            try { return RunHudShowcase(out, /*gameMode=*/false); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --game-hud <out.png>: the AX game scene plus a live "SCORE: N" HUD overlay (Slice BA). Its own
        // golden tests/golden/metal/game_hud.png; the existing game.png stays byte-identical.
        if (argc > 1 && std::strcmp(argv[1], "--game-hud") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_game_hud.png";
            try { return RunHudShowcase(out, /*gameMode=*/true); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --debug <out.png>: render the debug-visualization showcase (Slice W) — the settled physics
        // pyramid plus an immediate-mode DebugDraw line overlay (grid + AABBs + wire spheres + light
        // arrow + contact markers) through the new LINE_LIST debug pipeline.
        if (argc > 1 && std::strcmp(argv[1], "--debug") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_debug_viz.png";
            try { return RunDebugShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --cull <out.png>: render the frustum-culling visualization showcase (Slice AQ) — a wide row
        // of cubes/spheres from an overview camera with the render camera's frustum as lines and each
        // object's bound sphere green(kept)/red(culled). Mirrors the Vulkan --cull-shot path; new
        // golden tests/golden/metal/cull.png.
        if (argc > 1 && std::strcmp(argv[1], "--cull") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_cull.png";
            try { return RunCullShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --gpu-cull <out.png>: render the GPU-driven culling + indirect-draw showcase (Slice AR) — a
        // compute kernel frustum-culls a 1024-instance cube grid + ORDER-compacts the survivors +
        // writes the indirect draw-args; ONE DrawIndexedIndirect renders exactly the survivors, the
        // count decided on the GPU. Asserts the GPU survivor count == the CPU reference. Mirrors the
        // Vulkan --gpu-cull-shot path; new golden tests/golden/metal/gpu_cull.png.
        if (argc > 1 && std::strcmp(argv[1], "--gpu-cull") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_gpu_cull.png";
            try { return RunGpuCullShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --mt <out.png> [workers]: render the multithreaded-recording showcase (Slice AU) — a
        // draw-heavy 12x12 grid recorded across N worker threads via the parallel render command
        // encoder. Default 4 workers; an optional numeric 3rd arg overrides N (for the 1-vs-N
        // determinism check). Mirrors the Vulkan --mt-shot path; new golden tests/golden/metal/mt.png.
        if (argc > 1 && std::strcmp(argv[1], "--mt") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_mt.png";
            uint32_t workers = 4;
            if (argc > 3) { int n = std::atoi(argv[3]); if (n >= 1) workers = (uint32_t)n; }
            try { return RunMtShowcase(out, workers); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --gizmo <objIndex> <out.png>: render the editor gizmo showcase (Slice AB) — a small
        // multi-object scene with the selected object's translate gizmo drawn through the debug-line
        // layer. Mirrors the Vulkan --gizmo-shot path; new golden tests/golden/metal/gizmo.png.
        if (argc > 1 && std::strcmp(argv[1], "--gizmo") == 0) {
            int idx = argc > 2 ? std::atoi(argv[2]) : 2;
            const char* out = argc > 3 ? argv[3] : "metal_gizmo.png";
            try { return RunGizmoShowcase(idx, out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --skinning <out.png>: render the skeletal-animation showcase (Slice O) instead of the
        // default Slice-F scene.
        if (argc > 1 && std::strcmp(argv[1], "--skinning") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_skinning.png";
            try { return RunSkinningShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --blend <out.png>: render the animation-BLENDING showcase (Slice X) — same scene as
        // --skinning but the joint palette is a 50/50 cross-clip blend of "Walk" and "Run".
        if (argc > 1 && std::strcmp(argv[1], "--blend") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_anim_blend.png";
            try { return RunSkinningShowcase(out, /*blend=*/true); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --pbr <out.png>: render the full-PBR DamagedHelmet showcase (Slice P).
        if (argc > 1 && std::strcmp(argv[1], "--pbr") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_pbr.png";
            try { return RunPbrShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --material <out.png>: render the data-driven material-graph showcase (Slice AV) — a sphere
        // shaded by the build-time-generated fragment from showcase.mat.json.
        if (argc > 1 && std::strcmp(argv[1], "--material") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_mat_graph.png";
            try { return RunMaterialShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --material2 <out.png>: render the SECOND data-driven material (Slice AW) — a sphere shaded
        // by the build-time-generated fragment from showcase2.mat.json (different node mix). Metal
        // stays build-time; the new mat_graph2.png golden is rendered here.
        if (argc > 1 && std::strcmp(argv[1], "--material2") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_mat_graph2.png";
            try { return RunMaterialShowcase2(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --material-multi <out.png>: render the MULTI-material scene (Slice AZ) — three spheres in a
        // row, each a distinct generated graph material (showcase / showcase2 / showcase3, the last
        // exercising the new node types). The new mat_multi.png golden is rendered here.
        if (argc > 1 && std::strcmp(argv[1], "--material-multi") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_mat_multi.png";
            try { return RunMaterialMultiShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --scene <out.png>: render the full glTF scene-graph import showcase (Slice V) — the
        // CesiumMilkTruck imported as a node hierarchy with per-primitive PBR materials, the wheels
        // mesh placed at the front and back via composed node transforms.
        if (argc > 1 && std::strcmp(argv[1], "--scene") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_scene_import.png";
            try { return RunSceneShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --ibl <out.png>: render the HDR-environment-IBL showcase (Slice R).
        if (argc > 1 && std::strcmp(argv[1], "--ibl") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_ibl.png";
            try { return RunIblShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --bloom <out.png>: render the HDR bloom showcase (Slice U) — the HDR-IBL helmet scene into
        // an HDR RT, then the threshold/downsample/upsample bloom chain + composite/tonemap.
        if (argc > 1 && std::strcmp(argv[1], "--bloom") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_bloom.png";
            try { return RunBloomShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --capstone <out.png>: render the integrated capstone showcase (Slice Z) — ONE composed
        // frame combining the HDR-IBL environment, the imported truck, the skinned blended fox, the
        // PBR helmet on a pedestal, a settled physics sphere stack, and sorted glass, all lit +
        // shadowed into an HDR RT finished with the bloom chain.
        if (argc > 1 && std::strcmp(argv[1], "--capstone") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_capstone.png";
            try { return RunCapstoneShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --camera <yaw,pitch,x,y,z> <out.png>: Slice AA scripted-pose capture. Renders the SAME
        // capstone scene but framed by a runtime::Camera at the given pose, golden-verifying the
        // backend-agnostic Camera->viewProj math on Metal (matches the Vulkan --camera-shot pose).
        if (argc > 1 && std::strcmp(argv[1], "--camera") == 0) {
            ScriptedCamera sc; sc.active = true;
            const char* out = "metal_camera_pose.png";
            if (argc > 2) {
                float v[5] = {0, 0, 0, 4, 10};
                std::sscanf(argv[2], "%f,%f,%f,%f,%f", &v[0], &v[1], &v[2], &v[3], &v[4]);
                sc.yaw = v[0]; sc.pitch = v[1]; sc.position = {v[2], v[3], v[4]};
            }
            if (argc > 3) out = argv[3];
            try { return RunCapstoneShowcase(out, sc); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --ssao <out.png>: render the SSAO showcase (Slice Y) — the settled physics sphere-pyramid
        // into an HDR RT, a view-space normal+depth g-buffer prepass, a hemisphere-kernel AO pass, a
        // box blur, and a composite that multiplies the lit scene by the blurred AO (contact AO).
        if (argc > 1 && std::strcmp(argv[1], "--ssao") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_ssao.png";
            try { return RunSsaoShowcase(out, /*aoOn=*/true); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --ssao-off <out.png>: the IDENTICAL SSAO scene with AO forced off, for an on/off comparison.
        if (argc > 1 && std::strcmp(argv[1], "--ssao-off") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_ssao_off.png";
            try { return RunSsaoShowcase(out, /*aoOn=*/false); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --transparency <out.png>: render the alpha-blended transparency showcase (Slice T) —
        // opaque scene + a sorted, depth-tested-but-not-written translucent glass pass.
        if (argc > 1 && std::strcmp(argv[1], "--transparency") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_transparency.png";
            try { return RunTransparencyShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --ssr <out.png>: screen-space reflections showcase (Slice AH) — a dark reflective floor with
        // distinct colored objects, a view-space normal+linear-depth g-buffer, an SSR depth-march, and
        // a composite blending the on-screen reflections + tonemap.
        if (argc > 1 && std::strcmp(argv[1], "--ssr") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_ssr.png";
            try { return RunSsrShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --volumetric <out.png>: volumetric fog / light shafts showcase (Slice AJ) — an overhead
        // slatted canopy + near-overhead light streaming through the gaps, ray-marched against the
        // directional shadow map for Henyey-Greenstein in-scattering (god rays), composited + tonemapped.
        if (argc > 1 && std::strcmp(argv[1], "--volumetric") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_volumetric.png";
            try { return RunVolumetricShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --probe <out.png>: reflection + irradiance PROBE showcase (Slice AK). A Cornell-style box
        // (red left / green right / neutral) is baked from a fixed probe into a cube atlas (reflection
        // block + cosine-convolved irradiance block); a metallic sphere reflects the colored walls and
        // a matte box picks up red/green color bleed. Mirrors the Vulkan --probe-shot exactly.
        if (argc > 1 && std::strcmp(argv[1], "--probe") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_probe.png";
            try { return RunProbeShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --taa <out.png>: temporal anti-aliasing showcase (Slice AP). The settled sphere-pyramid scene
        // rendered as a FIXED 8-frame Halton(2,3)-jittered accumulation, neighborhood-clamped history
        // blend (taa_resolve), tonemapped + captured. Mirrors the Vulkan --taa-shot exactly; two runs
        // DIFF 0.0000.
        if (argc > 1 && std::strcmp(argv[1], "--taa") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_taa.png";
            try { return RunTaaShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
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

            // ---- The scene is now DATA: register the named GPU resources the scene file refers to,
            // then LoadScene parses assets/scenes/default.json into the ECS. The default scene
            // reproduces the old hardcoded scene exactly — a rough-dielectric ground plane + a 3x3
            // grid mixing shiny metal spheres (main diagonal) with matte dielectric cubes + the
            // glTF duck — created IN FILE ORDER (plane, the grid in gx/gz order, then the duck).
            // The render-graph passes query view<TransformC, MeshC, MaterialC>(), which iterates the
            // pools in dense (creation) order, so the draws are byte-identical (golden DIFF 0.0000)
            // and match the Vulkan sample, which loads the SAME scene file. ----
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
            scene::LoadScene(registry, resources, HF_SCENE_PATH);

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
                    for (auto [e, tc, mc, mat] :
                         registry.view<scene::TransformC, scene::MeshC, scene::MaterialC>()) {
                        (void)e; (void)mat;
                        Mat4 m = tc.t.Matrix();
                        cmd.PushConstants(m.m, sizeof(float) * 16);
                        cmd.BindVertexBuffer(mc.mesh->vertices());
                        cmd.BindIndexBuffer(mc.mesh->indices());
                        cmd.DrawIndexed(mc.mesh->indexCount());
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
                    for (auto [e, tc, mc, mat] :
                         registry.view<scene::TransformC, scene::MeshC, scene::MaterialC>()) {
                        (void)e;
                        Mat4 m = tc.t.Matrix();
                        // Push { float4x4 model; float4 material(metallic,roughness,0,0) } = 80 bytes.
                        float pc[20];
                        for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                        pc[16] = mat.metallic; pc[17] = mat.roughness; pc[18] = 0.0f; pc[19] = 0.0f;
                        cmd.PushConstants(pc, sizeof(pc));
                        cmd.BindMaterial(*mat.base, *mat.normal);
                        cmd.BindVertexBuffer(mc.mesh->vertices());
                        cmd.BindIndexBuffer(mc.mesh->indices());
                        cmd.DrawIndexed(mc.mesh->indexCount());
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
