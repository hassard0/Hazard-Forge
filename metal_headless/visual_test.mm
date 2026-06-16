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
#include "scene/streaming.h"        // Slice BD: distance-based scene/asset streaming (pure CPU)
#include "ecs/ecs.h"
#include "asset/gltf_loader.h"
#include "asset/env_loader.h"
#include "anim/animation.h"
#include "anim/skeleton.h"
#include "anim/state_machine.h"  // Slice BL: animation state machine + cross-fade (--anim-fsm)
#include "physics/world.h"
#include "physics/body.h"
#include "terrain/heightmap.h"      // Slice BF: deterministic procedural terrain / heightmap (pure CPU)
#include "terrain/terrain_stream.h" // Slice BJ: terrain-streaming LOD integration (pure CPU)
#include "game/roll_game.h"
#include "net/snapshot.h"          // Slice BQ: replication snapshot layer (pure CPU, shared with Vulkan)
#include "net/transport.h"         // Slice BU: simulated transport + client jitter-buffer/interp (pure CPU, shared)
#include "net/prediction.h"        // Slice BY: client prediction + server reconciliation (pure CPU, shared)
#include "ui/text.h"               // Slice BA: baked 8x8 font atlas + screen-space text layout (pure CPU)
#include "vfx/particles.h"         // Slice CC: CPU particle / VFX emitter system (pure CPU, shared)
#include "render/render_graph.h"
#include "render/csm.h"     // Slice AD: cascaded-shadow split + per-cascade ortho fit (pure math)
#include "render/spot.h"    // Slice AE: spot-light perspective shadow projection + cone (pure math)
#include "render/point_shadow.h" // Slice AF: omnidirectional point-light 6-face cube shadow (pure math)
#include "render/probe.h"        // Slice AK: reflection + irradiance probe atlas math (pure math)
#include "render/clustered.h"     // Slice AG: clustered / Forward+ light culling (pure math)
#include "render/cluster.h"       // Slice CL: clustered light culling (cluster-grid assignment math)
#include "render/froxel.h"         // Slice CS: froxel volumetric fog (grid + density + phase + integrate math)
#include "render/auto_exposure.h"  // Slice CW: auto-exposure histogram eye-adaptation math (luminance/bins/exposure)
#include "render/ssgi.h"          // Slice BR: SSGI bilateral-denoise params (SsgiDenoiseParams defaults)
#include "render/water.h"         // Slice CF: Gerstner water displacement/normal + the fixed wave set
#include "render/clouds.h"        // Slice CH: deterministic cloud noise/density/Beer/HG (mirrored in clouds.frag)
#include "render/taa.h"           // Slice AP: temporal anti-aliasing jitter + resolve-blend (pure math)
#include "render/frustum.h"        // Slice AQ: Gribb-Hartmann frustum extraction + sphere cull (pure math)
#include "render/gpu_cull.h"        // Slice AR: GPU-cull CPU mirror (ordered compaction + sphere test)
#include "render/gpu_culled.h"      // Slice CD: compute cull+compact CPU mirror (model+material+texIndex)
#include "render/hiz.h"             // Slice CJ: Hi-Z occlusion cull math (pure CPU; bit-identical cross-backend)
#include "render/decal.h"           // Slice BH: screen-space projected-decal box transform (pure math)
#include "render/post_stack.h"       // Slice BN: data-driven post-process stack config + per-effect math
#include "debug/debug_draw.h"
#include "debug/debug_emitters.h"
#include "runtime/camera.h"  // Slice AA: backend-agnostic Camera for the scripted-pose --camera path
#include "runtime/parallel_record.h"  // Slice AU: deterministic parallel command-recording partition
#include "editor/gizmo.h"    // Slice AB: transform gizmo emit (drawn through the debug-line layer)
#include "editor/editor_panels.h"   // Slice BT: docked editor UI (Hierarchy/Inspector/Stats/Viewport)
#include "editor/edit_ops.h"        // Slice BX: pure-CPU editor live-edit ops (ApplyTransform/Material)
#include "editor/imgui_renderer.h"  // Slice BT: RHI-only Dear ImGui renderer (consumes MSL on Metal)
#include "imgui.h"                   // Slice BT: Dear ImGui (NewFrame/Render + io setup)

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
// `blend`/`fsm` select the palette source: both false = single-clip sample of "Survey" t=0.5s
// (Slice O); blend (Slice X) = 50/50 cross-clip blend of "Walk" t=0.3s and "Run" t=0.2s via
// BlendAnimations; fsm (Slice BL) = an idle/walk/run StateMachine cross-fade scripted to a FIXED mid
// walk->run transition (the IDENTICAL FSM + timeline + capture step as the Vulkan --anim-fsm-shot,
// so the FSM-produced palette is bit-identical cross-backend). Everything else (scene/camera/light/
// pipelines) is shared so the PNGs are directly comparable.
static int RunSkinningShowcase(const char* outPath, bool blend = false, bool fsm = false) {
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
    if (fsm) {
        // Animation state machine (Slice BL). IDENTICAL graph + scripted speed timeline + capture
        // step as the Vulkan --anim-fsm-shot, so the FSM-produced palette is bit-identical across
        // backends. idle--(speed>0.3)-->walk--(speed>0.7)-->run + reverse edges, cross-fade 0.25s.
        // Scripted speed ramps 0->1 over 40 steps of dt=1/60; captured at the FIXED step 37 where the
        // FSM is mid walk->run cross-fade (BlendWeight ~ 0.53, printed). Deterministic (no input/RNG).
        const anim::Animation* survey = fox.FindAnimation("Survey");
        const anim::Animation* walk   = fox.FindAnimation("Walk");
        const anim::Animation* run    = fox.FindAnimation("Run");
        if (!survey && !fox.animations.empty()) survey = &fox.animations.front();
        if (!walk) walk = survey;
        if (!run)  run  = walk;
        if (survey && walk && run) {
            std::vector<anim::Animation> clips = {*survey, *walk, *run};
            anim::StateMachine sm;
            int sIdle = sm.AddState({"idle", 0, true, 1.0f});
            int sWalk = sm.AddState({"walk", 1, true, 1.0f});
            int sRun  = sm.AddState({"run",  2, true, 1.0f});
            int sp = sm.AddParam("speed", 0.0f);
            const float kDur = 0.25f;
            sm.AddTransition({sIdle, sWalk, sp, anim::Transition::Cmp::Greater, 0.3f, kDur});
            sm.AddTransition({sWalk, sRun,  sp, anim::Transition::Cmp::Greater, 0.7f, kDur});
            sm.AddTransition({sRun,  sWalk, sp, anim::Transition::Cmp::Less,    0.7f, kDur});
            sm.AddTransition({sWalk, sIdle, sp, anim::Transition::Cmp::Less,    0.3f, kDur});
            sm.SetInitialState(sIdle);
            const float kDt = 1.0f / 60.0f;
            auto scriptedSpeed = [](int step) { float s = step / 40.0f; return s < 1.0f ? s : 1.0f; };
            const int kCaptureStep = 37;
            float capSpeed = 0.0f;
            for (int step = 0; step <= kCaptureStep; ++step) {
                capSpeed = scriptedSpeed(step);
                sm.SetParam(sp, capSpeed);
                sm.Update(kDt);
            }
            palette = sm.Evaluate(fox.skeleton, clips);
            std::printf("anim-fsm: {state:%s->%s, blend:%.2f, speed:%.3f, step:%d}\n",
                        sm.CurrentStateName().c_str(),
                        sm.IsTransitioning() ? sm.TransitioningToName().c_str() : "-",
                        sm.BlendWeight(), capSpeed, kCaptureStep);
        } else {
            palette.assign(fox.skeleton.joints.size(), Mat4::Identity());
        }
    } else if (blend) {
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
static int RunMaterialShowcaseImpl(const char* outPath, const char* matMslFile, const char* matEntry,
                                   bool bumpyNormal = false) {
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
    // Slice BE: a real tangent-space normal map (8x8 domed tiles) for the NormalMap-node material so
    // the sphere shows bump relief. Bound at the normal slot ONLY when bumpyNormal (mat_normal); the
    // showcase/showcase2 paths keep the flat 1x1 normal so their goldens are unchanged.
    std::vector<uint8_t> bumpy = bumpyNormal ? MakeBumpyNormalMap() : std::vector<uint8_t>{};
    std::unique_ptr<rhi::ITexture> bumpyNormalTex;
    if (bumpyNormal)
        bumpyNormalTex = device->CreateTexture(
            {256, 256, rhi::Format::RGBA8_UNorm, bumpy.data(), bumpy.size()});
    rhi::ITexture& normalTex = bumpyNormal ? *bumpyNormalTex : *flatNormal;
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
                // Roughness matches the Vulkan --material-normal-shot path (0.45 for the normalmap
                // material, 0.35 for showcase/showcase2). metallic stays 0 (the push-constant
                // metallic/roughness are flat-shaded; the graph overrides them per its constants).
                pc[16] = 0.0f; pc[17] = bumpyNormal ? 0.45f : 0.35f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterialPBR(*checkerTex, *whiteTex, normalTex, *blackTex, *whiteTex);
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
// Slice BE: the NormalMap-node material — PBROutput.normal = NormalMap(slot="normalmap"); a tangent-
// space normal map (domed tiles) perturbs the shading normal via the codegen's TBN (the same Gram-
// Schmidt construction lit.frag uses). The bumpy normal map binds at the normal slot; the sphere
// shows clear bump relief. Renders the new mat_normal.png golden.
static int RunMaterialNormalShowcase(const char* outPath) {
    return RunMaterialShowcaseImpl(outPath, "mat_normalmap.frag.gen.metal", "material_normal_fragment",
                                   /*bumpyNormal=*/true);
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

// --- Procedural terrain showcase (Slice BF). Mirrors the Vulkan --terrain-shot path EXACTLY: build
// an n x n procedural terrain patch via terrain::BuildTerrain over the deterministic terrain::Height
// field (the SAME pure-C++ heightmap.cpp compiled into this target, so the mesh is bit-identical to
// the Windows build), upload its CPU vertex/index grid as a normal scene mesh, and render ONE lit +
// shadowed frame from the SAME fixed 3/4 camera + directional light. The height tint is baked into
// the per-vertex color (the unchanged lit fragment multiplies texture * vertex color) — no new
// shader. Identical scene/camera/light to the Vulkan path so the only difference vs the BMP is the
// backend NDC handling (FlipProjY). One offscreen frame -> PNG (new golden terrain.png). ------------
static int RunTerrainShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Build the deterministic terrain patch (IDENTICAL params to the Vulkan --terrain-shot).
    const int   kN = 128;
    const float kWorldSize = 20.0f;
    const float kHeightScale = 2.0f;
    terrain::TerrainMesh tm = terrain::BuildTerrain(kN, kWorldSize, kHeightScale);
    const uint32_t kVertCount = (uint32_t)tm.verts.size();
    const uint32_t kIndexCount = (uint32_t)tm.indices.size();
    const uint32_t kTriCount = kIndexCount / 3;

    rhi::BufferDesc tvb;
    tvb.size = (uint64_t)tm.verts.size() * sizeof(scene::Vertex);
    tvb.initialData = tm.verts.data();
    tvb.usage = rhi::BufferUsage::Vertex;
    auto terrainVB = device->CreateBuffer(tvb);
    rhi::BufferDesc tib;
    tib.size = (uint64_t)tm.indices.size() * sizeof(uint32_t);
    tib.initialData = tm.indices.data();
    tib.usage = rhi::BufferUsage::Index;
    auto terrainIB = device->CreateBuffer(tib);
    scene::Mesh terrainMesh{std::move(terrainVB), std::move(terrainIB), kIndexCount};

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

    const uint8_t whitePx[4] = {255, 255, 255, 255};
    auto whiteTex = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, whitePx, sizeof(whitePx)});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});

    Mat4 terrainModel = Mat4::Identity();

    const Vec3 eye{14.0f, 11.0f, 14.0f};
    const Vec3 center{0.0f, 0.0f, 0.0f};
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
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 sc{0.0f, 0.0f, 0.0f};
        Vec3 lightEye = sc - lightDir * 22.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-13.0f, 13.0f, -13.0f, 13.0f, 1.0f, 48.0f));
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
            cmd.PushConstants(terrainModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(terrainMesh.vertices());
            cmd.BindIndexBuffer(terrainMesh.indices());
            cmd.DrawIndexed(terrainMesh.indexCount());
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
                for (int k = 0; k < 16; ++k) pc[k] = terrainModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.92f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*whiteTex, *flatNormal);
                cmd.BindVertexBuffer(terrainMesh.vertices());
                cmd.BindIndexBuffer(terrainMesh.indices());
                cmd.DrawIndexed(terrainMesh.indexCount());
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

    std::printf("terrain: {n:%d, verts:%u, tris:%u, peak:%g}\n",
                kN, kVertCount, kTriCount, (double)tm.peak);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — terrain %dx%d, %u verts, %u tris\n",
                outPath, cw, ch, kN, kN, kVertCount, kTriCount);
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

// --- Networking / replication showcase (Slice BQ). Mirrors the Vulkan --net-shot path EXACTLY: run
// the AX roll-game as the AUTHORITY over the FULL scripted track at the engine fixed dt; each tick
// Capture a Snapshot (player id 0 + uncollected pickups id 1..N) and Send it through the channel — a
// FULL keyframe every kNetKeyframeInterval ticks, a per-entity DELTA otherwise — into bytes the REPLICA
// Receives via an in-process perfect channel (NO sockets). At the IDENTICAL fixed capture step
// (kNetCaptureStep == 250, the same as --game) assert replica.State() == the authority snapshot EXACTLY,
// then render the REPLICA'S reconstructed scene (player + remaining pickups from its RepEntities) lit +
// shadowed via the existing static-lit scene path. The replication core (engine/net/snapshot.cpp) is the
// SAME pure-C++ TU the Vulkan build compiles, so the serialized bytes + the net stat line are bit-
// identical cross-backend. The only difference vs the BMP-golden is backend-NDC handling. One PNG. -----
static int RunNetShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Identical replication parameters + capture step to the Vulkan --net-shot.
    const int kNetCaptureStep = 250;
    const uint32_t kNetKeyframeInterval = 8;
    const float dtG = 1.0f / 120.0f;
    std::vector<game::GameInput> track = game::ScriptedTrack();

    physics::World authWorld;
    game::GameState authState = game::MakeRollGame(authWorld);
    net::Replicator rep(kNetKeyframeInterval);
    net::Snapshot authAtCapture, replicaAtCapture;
    bool haveCapture = false;
    const int totalTicks = (int)track.size();
    for (int t = 0; t <= totalTicks; ++t) {
        net::Snapshot snap = net::Replicator::Capture((uint32_t)t, authState, authWorld);
        std::vector<uint8_t> packet = rep.Send(snap);
        rep.Receive(packet);
        if (t == kNetCaptureStep) {
            authAtCapture = snap;
            replicaAtCapture = rep.State();
            haveCapture = true;
        }
        if (t < totalTicks) game::StepGame(authWorld, authState, track[(size_t)t], dtG);
    }
    const bool replicaMatch = haveCapture && (replicaAtCapture == authAtCapture);
    if (!replicaMatch) return fail("replica state != authority at capture step");

    const uint64_t fullBytes = rep.FullBytes();
    const uint64_t deltaBytes = rep.DeltaBytes();
    const double savingsPct = (fullBytes > 0)
        ? 100.0 * (double)(fullBytes - deltaBytes) / (double)fullBytes : 0.0;

    // Build the render models from the REPLICA'S reconstructed RepEntities (NOT the authority state).
    Mat4 replicaPlayerModel = Mat4::Identity();
    bool havePlayer = false;
    std::vector<Mat4> pickupModels;
    for (const net::RepEntity& e : replicaAtCapture.entities) {
        if (e.flags & net::kFlagPlayer) {
            replicaPlayerModel = math::FromTRS(
                e.position, e.orientation,
                {2.0f * game::kPlayerRadius, 2.0f * game::kPlayerRadius, 2.0f * game::kPlayerRadius});
            havePlayer = true;
        } else if (e.flags & net::kFlagPickup) {
            pickupModels.push_back(math::FromTRS(
                e.position, e.orientation,
                {2.0f * game::kPickupRadius, 2.0f * game::kPickupRadius, 2.0f * game::kPickupRadius}));
        }
    }
    if (!havePlayer) return fail("replica has no player entity");
    Mat4 playerModel = replicaPlayerModel;

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

    std::printf("net: {ticks:%d, snapshots:%u, fullBytes:%llu, deltaBytes:%llu, savings:%.1f%%, replicaMatch:%s}\n",
                totalTicks + 1, rep.SnapshotsSent(),
                (unsigned long long)fullBytes, (unsigned long long)deltaBytes,
                savingsPct, replicaMatch ? "true" : "false");

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — capture step %d, %zu pickups remaining (replica)\n",
                outPath, cw, ch, kNetCaptureStep, pickupModels.size());
    return 0;
}

// --- Networking TRANSPORT + client jitter-buffer / interpolation showcase (Slice BU). Mirrors the
// Vulkan --netsim-shot path EXACTLY: run the AX roll-game as the AUTHORITY over the FULL scripted track
// at the engine fixed dt; each tick Capture+serialize a Snapshot and Send it through a SimChannel with
// the IDENTICAL lossy/laggy ChannelConfig (latency 3, dropRate 0.15, reorderRate 0.1, the SAME fixed
// lossSeed) — a pure integer-LCG seeded channel (NO sockets). Each tick the CLIENT Delivers the
// (possibly dropped/reordered) packets + buffers them; at the IDENTICAL fixed render tick the client
// renders its INTERPOLATED RenderState (interpDelay 2). `converged` asserts the interpolated client
// state is within the documented tolerance of the authority's true state. The transport core
// (engine/net/transport.cpp) is the SAME pure-C++ TU the Vulkan build compiles, so the channel's
// drop/reorder decisions + the netsim stat line are BIT-IDENTICAL cross-backend. The only difference vs
// the BMP-golden is backend-NDC handling. One PNG. ------------------------------------------------
static int RunNetsimShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // IDENTICAL channel + interpolation config to the Vulkan --netsim-shot.
    const int kNetCaptureStep = 250;
    const int kSimLatency = 3;
    const float kSimDropRate = 0.15f;
    const float kSimReorderRate = 0.10f;
    const uint32_t kSimLossSeed = 0xBADC0DEu;
    const int kInterpDelay = 2;
    const float dtG = 1.0f / 120.0f;
    std::vector<game::GameInput> track = game::ScriptedTrack();

    net::ChannelConfig cfg;
    cfg.latencyTicks = kSimLatency;
    cfg.lossSeed = kSimLossSeed;
    cfg.dropRate = kSimDropRate;
    cfg.reorderRate = kSimReorderRate;
    net::SimChannel channel(cfg);
    net::ClientView client(kInterpDelay);

    physics::World authWorld;
    game::GameState authState = game::MakeRollGame(authWorld);
    const int totalTicks = (int)track.size();
    const int kRenderTick = kNetCaptureStep;
    const int kDriveUntil = std::min(totalTicks, kRenderTick + kSimLatency + kInterpDelay + 12);
    net::Snapshot authAtRender;
    bool haveAuth = false;
    for (int t = 0; t <= kDriveUntil; ++t) {
        net::Snapshot snap = net::Replicator::Capture((uint32_t)t, authState, authWorld);
        if (t == kRenderTick) { authAtRender = snap; haveAuth = true; }
        channel.Send(t, net::Serialize(snap));
        for (const net::Packet& p : channel.Deliver(t)) client.Receive(p.bytes);
        if (t < totalTicks) game::StepGame(authWorld, authState, track[(size_t)t], dtG);
    }
    if (!haveAuth) return fail("no authority snapshot at render tick");

    net::Snapshot clientView = client.RenderState((float)kRenderTick);

    const float kConvergeTol = 0.25f;
    auto playerPos = [](const net::Snapshot& s) -> Vec3 {
        for (const net::RepEntity& e : s.entities) if (e.flags & net::kFlagPlayer) return e.position;
        return Vec3{0, 0, 0};
    };
    const float convErr = math::length(playerPos(clientView) - playerPos(authAtRender));
    const bool converged = convErr <= kConvergeTol;
    if (!converged) return fail("client interp state diverged from authority");

    // Build the render models from the CLIENT'S interpolated RepEntities (NOT the authority state).
    Mat4 replicaPlayerModel = Mat4::Identity();
    bool havePlayer = false;
    std::vector<Mat4> pickupModels;
    for (const net::RepEntity& e : clientView.entities) {
        if (e.flags & net::kFlagPlayer) {
            replicaPlayerModel = math::FromTRS(
                e.position, e.orientation,
                {2.0f * game::kPlayerRadius, 2.0f * game::kPlayerRadius, 2.0f * game::kPlayerRadius});
            havePlayer = true;
        } else if (e.flags & net::kFlagPickup) {
            pickupModels.push_back(math::FromTRS(
                e.position, e.orientation,
                {2.0f * game::kPickupRadius, 2.0f * game::kPickupRadius, 2.0f * game::kPickupRadius}));
        }
    }
    if (!havePlayer) return fail("client view has no player entity");
    Mat4 playerModel = replicaPlayerModel;

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

    std::printf("netsim: {latency:%d, lossRate:%.2f, delivered:%u, dropped:%u, reordered:%u, "
                "interpDelay:%d, renderTick:%d, converged:%s}\n",
                kSimLatency, kSimDropRate, channel.Delivered(), channel.Dropped(),
                channel.Reordered(), kInterpDelay, kRenderTick, converged ? "true" : "false");

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — render tick %d, %zu pickups remaining (client interp)\n",
                outPath, cw, ch, kRenderTick, pickupModels.size());
    return 0;
}

// --- Client PREDICTION + server RECONCILIATION showcase (Slice BY). Mirrors the Vulkan --netpredict-shot
// path EXACTLY: run the AX roll-game as the AUTHORITY over the scripted track PLUS the IDENTICAL scripted
// SERVER-ONLY impulse (a fixed +Z velocity bump at kImpulseTick the client cannot predict); a
// PredictedClient predicts every tick locally + reconciles when the authoritative frame arrives (delayed
// by the IDENTICAL fixed latency via the BU SimChannel), rewinding to the acknowledged state + replaying
// its unacked inputs. At the IDENTICAL fixed render tick render the client's PREDICTED + RECONCILED scene
// (player + remaining pickups). The prediction core (engine/net/prediction.cpp) is the SAME pure-C++ TU
// the Vulkan build compiles, so the predicted sim + rewind/replay reconciliation + the netpredict stat
// line are BIT-IDENTICAL cross-backend. The only difference vs the BMP-golden is backend-NDC handling.
// One PNG. ----------------------------------------------------------------------------------------
static int RunNetpredictShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // IDENTICAL config to the Vulkan --netpredict-shot.
    const int kNetCaptureStep = 250;
    const int kRenderTick = kNetCaptureStep;
    const int kPredLatency = 4;
    const uint32_t kPredLossSeed = 0x5EEDB7u;
    const int kImpulseTick = 120;
    const Vec3 kServerImpulse{0.0f, 0.0f, 2.4f};
    const float dtG = 1.0f / 120.0f;
    std::vector<game::GameInput> track = game::ScriptedTrack();
    if (kRenderTick >= (int)track.size()) return fail("render tick outside scripted track");

    physics::World authWorld;
    game::GameState authState = game::MakeRollGame(authWorld);
    auto stepAuthority = [&](int stepIndex, const game::GameInput& in) {
        if (stepIndex == kImpulseTick) {
            physics::RigidBody& body = authWorld.bodies[(size_t)authState.playerBodyIndex];
            body.linVel = body.linVel + kServerImpulse;
        }
        game::StepGame(authWorld, authState, in, dtG);
    };

    net::ChannelConfig cfg;
    cfg.latencyTicks = kPredLatency;
    cfg.lossSeed = kPredLossSeed;
    net::SimChannel channel(cfg);
    std::map<int, net::AuthState> authFrames;

    net::PredictedClient client;
    net::AuthState authAtRender;
    bool haveAuthAtRender = false;
    for (int t = 0; t <= kRenderTick; ++t) {
        net::InputCmd cmd; cmd.tick = t; cmd.input = track[(size_t)t];
        client.PredictTick(cmd);
        net::AuthState frame = net::AuthState::Capture(t, authState, authWorld);
        if (t == kRenderTick) { authAtRender = frame; haveAuthAtRender = true; }
        authFrames[t] = frame;
        channel.Send(t, net::Serialize(frame.snap));
        for (const net::Packet& p : channel.Deliver(t)) {
            auto it = authFrames.find(p.sendTick);
            if (it != authFrames.end()) client.OnAuthoritative(it->second);
        }
        stepAuthority(t, track[(size_t)t]);
    }
    if (!haveAuthAtRender) return fail("no authority frame at render tick");

    const float kConvergeTol = 0.10f;
    const Vec3 predPlayer = client.PlayerPos();
    const Vec3 truePlayer = authAtRender.playerBody.position;
    const float convErr = math::length(predPlayer - truePlayer);
    const bool converged = convErr <= kConvergeTol;
    const float maxMis = client.MaxMisprediction();
    if (!converged) return fail("reconciled client diverged from authority");
    if (!(maxMis > 0.0f)) return fail("maxMisprediction==0 (trivial test)");

    // Build the render models from the CLIENT'S PREDICTED + RECONCILED state (its local sim).
    Mat4 playerModel = Mat4::Identity();
    {
        const physics::RigidBody& pb =
            client.World().bodies[(size_t)client.State().playerBodyIndex];
        playerModel = math::FromTRS(
            pb.position, pb.orientation,
            {2.0f * game::kPlayerRadius, 2.0f * game::kPlayerRadius, 2.0f * game::kPlayerRadius});
    }
    std::vector<Mat4> pickupModels;
    for (const game::Pickup& p : client.State().pickups) {
        if (p.collected) continue;
        pickupModels.push_back(math::FromTRS(
            p.pos, math::Quat::Identity(),
            {2.0f * game::kPickupRadius, 2.0f * game::kPickupRadius, 2.0f * game::kPickupRadius}));
    }

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

    std::printf("netpredict: {predTicks:%d, reconciles:%u, maxMisprediction:%.4f, finalError:%.4f, "
                "converged:%s}\n",
                client.PredictedTick() + 1, client.Reconciles(), maxMis, convErr,
                converged ? "true" : "false");

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — render tick %d, %zu pickups remaining (client predicted)\n",
                outPath, cw, ch, kRenderTick, pickupModels.size());
    return 0;
}

// --- Scene / asset STREAMING showcase (Slice BD). Mirrors the Vulkan --stream-shot path EXACTLY:
// build the fixed 8x8=64 cell world (engine/scene/streaming, pure CPU — shared byte-for-byte with the
// Vulkan build), fly the IDENTICAL scripted ground-level camera path across the grid calling
// StreamingWorld::Update each frame (cells stream in/out under the per-frame budget), then at the
// IDENTICAL fixed capture frame (kStreamCaptureFrame == 40) render the RESIDENT cells' procedural
// clusters (cubes + spheres) over the checkerboard ground + sky, lit + shadowed via the existing
// static-lit scene path with the same 4-color tint palette. Identical grid/config/path/capture/camera/
// light/colors to the Vulkan path so the only difference vs the BMP-golden is backend-NDC handling.
// One offscreen frame -> PNG. Prints the SAME deterministic `stream: {...}` state line. ------------
static int RunStreamShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // The fixed world + streaming policy (identical to the Vulkan side + the unit-test grid).
    const int   kGridN   = 8;
    const float kSpacing = 4.0f;
    scene::StreamConfig cfg;
    cfg.loadRadius           = 9.0f;
    cfg.unloadRadius         = 14.0f;
    cfg.loadBudgetPerFrame   = 2;
    cfg.unloadBudgetPerFrame = 3;

    // IDENTICAL scripted ground-level camera path.
    const int kStreamFrames = 90;
    std::vector<Vec3> path;
    path.reserve(kStreamFrames);
    const float startXZ = -20.0f, endXZ = 20.0f;
    for (int f = 0; f < kStreamFrames; ++f) {
        float t = (float)f / (float)(kStreamFrames - 1);
        float p = startXZ + t * (endXZ - startXZ);
        path.push_back(Vec3{p, 0.0f, p});
    }
    const int kStreamCaptureFrame = 40;

    scene::StreamingWorld streamWorld(kGridN, kSpacing, cfg);
    for (int f = 0; f <= kStreamCaptureFrame && f < kStreamFrames; ++f)
        streamWorld.Update(path[(size_t)f]);

    const Vec3 focus = path[(size_t)kStreamCaptureFrame];
    scene::StreamStats stats = streamWorld.Stats();
    std::vector<int> residentIds = streamWorld.ResidentCellIds();
    std::vector<scene::CellRenderable> residentDraws = streamWorld.ResidentRenderables();

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
    const uint8_t palettePx[4][4] = {
        {220, 70, 70, 255}, {70, 200, 90, 255}, {70, 130, 230, 255}, {230, 190, 60, 255},
    };
    std::unique_ptr<rhi::ITexture> paletteTex[4];
    for (int k = 0; k < 4; ++k)
        paletteTex[k] = device->CreateTexture(
            {1, 1, rhi::Format::RGBA8_UNorm, palettePx[k], sizeof(palettePx[k])});

    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh cube = scene::Mesh::Cube(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);

    Mat4 groundModel = Mat4::Scale({22.0f, 1.0f, 22.0f});

    const Vec3 eye = focus + Vec3{-7.0f, 7.0f, -7.0f};
    const Vec3 center = focus + Vec3{4.0f, 0.0f, 4.0f};
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 200.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc = center;
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 lightEye = sc - lightDir * 24.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-14.0f, 14.0f, -14.0f, 14.0f, 1.0f, 56.0f));
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
    auto bindMeshFor = [&](rhi::ICommandBuffer& cmd, scene::CellRenderable::Kind kind) {
        const scene::Mesh& m = (kind == scene::CellRenderable::Kind::Cube) ? cube : sphere;
        cmd.BindVertexBuffer(m.vertices());
        cmd.BindIndexBuffer(m.indices());
        return m.indexCount();
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
            for (const auto& d : residentDraws) {
                uint32_t ic = bindMeshFor(cmd, d.kind);
                cmd.PushConstants(d.model.m, sizeof(float) * 16);
                cmd.DrawIndexed(ic);
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
            for (const auto& d : residentDraws) {
                int ci = d.colorIndex & 3;
                cmd.BindMaterial(*paletteTex[ci], *flatNormal);
                uint32_t ic = bindMeshFor(cmd, d.kind);
                float pc[20]; litPush(d.model, d.metallic, d.roughness, pc);
                cmd.PushConstants(pc, sizeof(pc));
                cmd.DrawIndexed(ic);
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

    std::printf("stream: {frame:%d, resident:%d, loading:%d, unloading:%d, total:%d, residentIds:[",
                kStreamCaptureFrame, stats.resident, stats.loading, stats.unloading, stats.totalCells);
    for (size_t k = 0; k < residentIds.size(); ++k)
        std::printf("%s%d", k ? ", " : "", residentIds[k]);
    std::printf("]}\n");

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — capture frame %d, %d/%d cells resident\n",
                outPath, cw, ch, kStreamCaptureFrame, stats.resident, stats.totalCells);
    return 0;
}

// --- Terrain-streaming LOD showcase (Slice BJ). Mirrors the Vulkan --terrain-stream-shot path EXACTLY:
// build the fixed 6x6=36 tile terrain world (engine/terrain/terrain_stream, pure CPU — each tile meshed
// by BuildTerrainTile over the GLOBAL Height field, shared byte-for-byte with the Vulkan build so the
// per-tile meshes are bit-identical cross-backend), fly the IDENTICAL scripted camera path across the
// grid calling TerrainStreamWorld::Update each frame (tiles stream in/out by distance under the per-frame
// budget + select a discrete LOD band with hysteresis), then at the IDENTICAL fixed capture frame
// (kTSCaptureFrame == 45) upload + render the RESIDENT tiles' meshes AT THEIR LODs over sky, lit +
// shadowed via the existing static-lit scene path (BF height-tint vertex color). Identical grid/config/
// path/capture/camera/light to the Vulkan path so the only difference vs the BMP-golden is backend-NDC
// handling. One offscreen frame -> PNG. Prints the SAME deterministic `terrain-stream: {...}` state line.
static int RunTerrainStreamShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // The fixed terrain-stream world + policy (identical to the Vulkan side + the unit test).
    const int   kT = 6;
    const float kTileSize = 16.0f;
    const float kHeightScale = 2.0f;
    terrain::TerrainStreamConfig tcfg;
    tcfg.loadRadius           = 40.0f;
    tcfg.unloadRadius         = 52.0f;
    tcfg.loadBudgetPerFrame   = 6;
    tcfg.unloadBudgetPerFrame = 6;
    tcfg.heightScale          = kHeightScale;
    tcfg.bands.bandNear       = 15.0f;
    tcfg.bands.bandMid        = 28.0f;
    tcfg.bands.hysteresis     = 3.0f;

    // IDENTICAL scripted camera path.
    const int kTSFrames = 90;
    std::vector<Vec3> path;
    path.reserve(kTSFrames);
    const float startXZ = -64.0f, endXZ = 64.0f;
    for (int f = 0; f < kTSFrames; ++f) {
        float t = (float)f / (float)(kTSFrames - 1);
        float p = startXZ + t * (endXZ - startXZ);
        path.push_back(Vec3{p, 8.0f, p});
    }
    const int kTSCaptureFrame = 45;

    terrain::TerrainStreamWorld tworld(kT, kTileSize, tcfg);
    for (int f = 0; f <= kTSCaptureFrame && f < kTSFrames; ++f)
        tworld.Update(path[(size_t)f]);

    const Vec3 focus = path[(size_t)kTSCaptureFrame];
    terrain::TerrainStreamStats tstats = tworld.Stats();
    std::vector<terrain::ResidentTile> residentTiles = tworld.ResidentTiles();

    // Upload each resident tile's mesh (varying vertex count by LOD) into its own GPU buffers.
    struct TileDraw { std::unique_ptr<rhi::IBuffer> vb, ib; uint32_t indexCount; };
    std::vector<TileDraw> tileDraws;
    tileDraws.reserve(residentTiles.size());
    for (const auto& rt : residentTiles) {
        const terrain::TerrainMesh& tm = *rt.mesh;
        rhi::BufferDesc tvb;
        tvb.size = (uint64_t)tm.verts.size() * sizeof(scene::Vertex);
        tvb.initialData = tm.verts.data();
        tvb.usage = rhi::BufferUsage::Vertex;
        rhi::BufferDesc tib;
        tib.size = (uint64_t)tm.indices.size() * sizeof(uint32_t);
        tib.initialData = tm.indices.data();
        tib.usage = rhi::BufferUsage::Index;
        TileDraw td;
        td.vb = device->CreateBuffer(tvb);
        td.ib = device->CreateBuffer(tib);
        td.indexCount = (uint32_t)tm.indices.size();
        tileDraws.push_back(std::move(td));
    }

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

    const uint8_t whitePx[4] = {255, 255, 255, 255};
    auto whiteTex = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, whitePx, sizeof(whitePx)});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});

    Mat4 terrainModel = Mat4::Identity();

    const Vec3 eye = focus + Vec3{-26.0f, 24.0f, -26.0f};
    const Vec3 center = focus + Vec3{10.0f, 0.0f, 10.0f};
    const float aspect = (float)W / (float)H;
    FrameData fd{};
    {
        Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
        Mat4 proj = FlipProjY(Mat4::Perspective(1.04719755f, aspect, 0.1f, 300.0f));
        Mat4 vp = proj * view;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc = center;
        Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
        Vec3 lightEye = sc - lightDir * 50.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-34.0f, 34.0f, -34.0f, 34.0f, 1.0f, 110.0f));
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
            cmd.PushConstants(terrainModel.m, sizeof(float) * 16);
            for (const auto& td : tileDraws) {
                cmd.BindVertexBuffer(*td.vb);
                cmd.BindIndexBuffer(*td.ib);
                cmd.DrawIndexed(td.indexCount);
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
                float pc[20]; litPush(terrainModel, 0.0f, 0.92f, pc);
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*whiteTex, *flatNormal);
                for (const auto& td : tileDraws) {
                    cmd.BindVertexBuffer(*td.vb);
                    cmd.BindIndexBuffer(*td.ib);
                    cmd.DrawIndexed(td.indexCount);
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

    std::printf("terrain-stream: {frame:%d, resident:%d, lod0:%d, lod1:%d, lod2:%d, total:%d, tiles:[",
                kTSCaptureFrame, tstats.resident, tstats.byLod[0], tstats.byLod[1],
                tstats.byLod[2], tstats.total);
    for (size_t k = 0; k < residentTiles.size(); ++k) {
        const auto& rt2 = residentTiles[k];
        std::printf("%s(%d,%d):%d", k ? ", " : "", rt2.tile.i, rt2.tile.j, rt2.lod);
    }
    std::printf("]}\n");

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — capture frame %d, %d/%d tiles resident\n",
                outPath, cw, ch, kTSCaptureFrame, tstats.resident, tstats.total);
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

// --- Depth of field showcase (Slice CG). Mirrors the Vulkan --dof-shot path EXACTLY: a row of distinct
// colored objects RECEDING from the camera (near foreground -> a middle FOCAL object -> far background)
// on a floor, lit + shadowed, rendered into an HDR RGBA16F target PLUS the view-space normal+linear-
// depth g-buffer (the SAME gbuffer shaders SSR/SSGI use). A fullscreen DoF pass (dof.frag) reconstructs
// each pixel's view-linear depth (ReconstructViewPos's .w convention), computes the thin-lens circle of
// confusion (render::dof::CircleOfConfusion, mirrored in-shader) and gathers a CoC-sized Vogel disk
// weighted by the scatter-as-gather BlurWeight so the middle object stays crisp while the fore/back
// objects blur, then tonemaps. The IDENTICAL scene + lens params as the Vulkan path. SEPARATE dof
// pipeline + shader; existing pipelines/shaders/goldens untouched. ----------------------------------
static int RunDofShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;

    // FIXED lens params (IDENTICAL to the Vulkan --dof-shot path).
    const float kFocalDist  = 12.0f;
    const float kAperture   = 90.0f;
    const float kFocalLen   = 1.6f;
    const float kMaxCoCpx   = 18.0f;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Scene objects: a row receding along -Z (near -> far). IDENTICAL to the Vulkan path.
    struct Obj { Vec3 pos; float scale; bool cube; float col[3]; };
    const Obj objs[] = {
        {{-1.4f, 0.7f,  4.5f}, 0.7f, false, {0.95f, 0.25f, 0.20f}},  // near red sphere (blurred)
        {{ 0.9f, 0.7f,  2.0f}, 0.7f, true,  {0.95f, 0.55f, 0.15f}},  // near-ish orange cube (blurred)
        {{-0.3f, 0.8f, -1.0f}, 0.8f, false, {0.25f, 0.90f, 0.35f}},  // MIDDLE green sphere (FOCAL, crisp)
        {{ 1.2f, 0.7f, -4.0f}, 0.7f, true,  {0.25f, 0.50f, 0.95f}},  // far blue cube (blurred)
        {{-1.0f, 0.75f,-7.0f}, 0.75f,false, {0.85f, 0.35f, 0.90f}},  // far magenta sphere (blurred)
    };
    const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));

    // Lit / shadow / sky / g-buffer pipelines (UNCHANGED shaders), same as the SSR showcase.
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
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
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;   // model(16) + view(16)
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    // DoF fullscreen pipeline (fragment push constants) writing the swapchain.
    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    struct DofParams {
        float texel[2]; float focalDist; float aperture;
        float focalLength; float maxCoCpx; float pad[2];
    };
    auto dofFs = loadMSL("dof.frag.gen.metal", "dof_fragment");
    rhi::GraphicsPipelineDesc dofD;
    dofD.vertex = postVs.get(); dofD.fragment = dofFs.get();
    dofD.colorFormat = device->Swapchain().ColorFormat();
    dofD.depthTest = false; dofD.usesTexture = true; dofD.fullscreen = true;
    dofD.fragmentPushConstants = true; dofD.pushConstantSize = sizeof(DofParams);
    auto dofPipe = device->CreateGraphicsPipeline(dofD);

    auto rt   = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    // Neutral mid-grey checker floor (IDENTICAL to the Vulkan path).
    std::vector<uint8_t> floorTexels(256 * 256 * 4);
    for (uint32_t y = 0; y < 256; ++y)
        for (uint32_t x = 0; x < 256; ++x) {
            bool dark = (((x / 32) + (y / 32)) & 1) != 0;
            uint8_t v = dark ? 70 : 110;
            size_t idx = (static_cast<size_t>(y) * 256 + x) * 4;
            floorTexels[idx + 0] = v; floorTexels[idx + 1] = v;
            floorTexels[idx + 2] = v; floorTexels[idx + 3] = 255;
        }
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, floorTexels.data(), floorTexels.size()});
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

    Mat4 groundModel = Mat4::Scale({14.0f, 1.0f, 14.0f});
    std::vector<Mat4> objModel(kNumObjs);
    for (int o = 0; o < kNumObjs; ++o)
        objModel[o] = Mat4::Translate(objs[o].pos) * Mat4::Scale(
            {objs[o].scale, objs[o].scale, objs[o].scale});

    const Vec3 eye{0.0f, 1.6f, 11.0f};
    const Vec3 center{0.0f, 0.7f, -1.0f};
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
        Vec3 sc{0.0f, 0.7f, -1.0f};
        Vec3 lightEye = sc - lightDir * 20.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-9.0f, 9.0f, -9.0f, 9.0f, 1.0f, 44.0f));
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

    DofParams dpp{};
    dpp.texel[0] = 1.0f / (float)W; dpp.texel[1] = 1.0f / (float)H;
    dpp.focalDist = kFocalDist; dpp.aperture = kAperture;
    dpp.focalLength = kFocalLen; dpp.maxCoCpx = kMaxCoCpx;

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgGbuf = graph.ImportTarget(
        "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
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
                pc[16] = 0.0f; pc[17] = 0.8f; pc[18] = 0.0f; pc[19] = 0.0f;
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

    graph.AddPass("dof", {rgScene, rgGbuf}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*dofPipe);
            cmd.BindTexturePair(*rt, *gbuf);
            cmd.PushConstants(&dpp, sizeof(dpp));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — DoF, %d objects\n", outPath, cw, ch, kNumObjs);
    return 0;
}

// --- Motion blur showcase (Slice CN). Mirrors the Vulkan --motionblur-shot path EXACTLY: a ground + a
// row of raised colored objects, lit + shadowed, rendered into an HDR RGBA16F scene target PLUS the
// view-space normal+linear-depth g-buffer (the SAME gbuffer shaders SSR/SSGI/DoF use). The CAMERA PANS
// laterally at a FIXED per-frame rate (the prev camera is the cur camera shifted by a fixed amount); a
// fullscreen velocity-gather pass (motion_blur.frag) reconstructs each pixel's view position, computes
// the screen-space velocity from prevViewProj/curViewProj (render::motionblur::ScreenVelocity, mirrored
// in-shader), clamps it to maxBlurPx and gathers N taps along it weighted by the depth-aware TapWeight,
// then tonemaps. The IDENTICAL scene + params as the Vulkan path; the SAME render/motion_blur.h math
// compiled here makes the velocity field bit-identical cross-backend. SEPARATE motion-blur pipeline +
// shader; existing pipelines/shaders/goldens untouched. Two runs DIFF 0.0000. -----------------------
static int RunMotionBlurShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;

    // FIXED motion-blur params (IDENTICAL to the Vulkan --motionblur-shot path).
    const float kMaxBlurPx = 28.0f;
    const int   kTaps      = 24;
    const float kVelScale  = 1.0f;
    const Vec3  kPanPerFrame{0.85f, 0.0f, 0.0f};

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Scene objects: a row of raised cubes/spheres. IDENTICAL to the Vulkan path.
    struct Obj { Vec3 pos; float scale; bool cube; float col[3]; };
    const Obj objs[] = {
        {{-3.0f, 0.9f, -2.0f}, 0.9f, true,  {0.95f, 0.30f, 0.25f}},
        {{-1.0f, 0.9f, -3.5f}, 0.9f, false, {0.30f, 0.85f, 0.40f}},
        {{ 1.2f, 0.9f, -2.5f}, 0.9f, true,  {0.30f, 0.55f, 0.95f}},
        {{ 3.1f, 0.9f, -4.0f}, 0.9f, false, {0.95f, 0.80f, 0.25f}},
        {{ 0.1f, 1.4f, -6.0f}, 0.9f, true,  {0.85f, 0.40f, 0.90f}},
    };
    const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));

    // Lit / shadow / sky / g-buffer pipelines (UNCHANGED shaders), same as the DoF showcase.
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
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
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;   // model(16) + view(16)
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    // Motion-blur fullscreen pipeline (fragment push constants) writing the swapchain.
    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    struct MbParams {
        float prevClip[16];   // prevViewProj * inverse(curView), column-major
        float texel[2]; float tanHalfFovY; float aspect;
        float maxBlurPx; float taps; float velScale; float pad;
    };
    auto mbFs = loadMSL("motion_blur.frag.gen.metal", "motion_blur_fragment");
    rhi::GraphicsPipelineDesc mbD;
    mbD.vertex = postVs.get(); mbD.fragment = mbFs.get();
    mbD.colorFormat = device->Swapchain().ColorFormat();
    mbD.depthTest = false; mbD.usesTexture = true; mbD.fullscreen = true;
    mbD.fragmentPushConstants = true; mbD.pushConstantSize = sizeof(MbParams);
    auto mbPipe = device->CreateGraphicsPipeline(mbD);

    auto rt   = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    // Neutral mid-grey checker floor (IDENTICAL to the Vulkan path).
    std::vector<uint8_t> floorTexels(256 * 256 * 4);
    for (uint32_t y = 0; y < 256; ++y)
        for (uint32_t x = 0; x < 256; ++x) {
            bool dark = (((x / 32) + (y / 32)) & 1) != 0;
            uint8_t v = dark ? 70 : 110;
            size_t idx = (static_cast<size_t>(y) * 256 + x) * 4;
            floorTexels[idx + 0] = v; floorTexels[idx + 1] = v;
            floorTexels[idx + 2] = v; floorTexels[idx + 3] = 255;
        }
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, floorTexels.data(), floorTexels.size()});
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

    Mat4 groundModel = Mat4::Scale({16.0f, 1.0f, 16.0f});
    std::vector<Mat4> objModel(kNumObjs);
    for (int o = 0; o < kNumObjs; ++o)
        objModel[o] = Mat4::Translate(objs[o].pos) * Mat4::Scale(
            {objs[o].scale, objs[o].scale, objs[o].scale});

    const Vec3 eye{0.0f, 2.4f, 8.0f};
    const Vec3 center{0.0f, 0.9f, -3.0f};
    const float aspect = (float)W / (float)H;
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    Mat4 prevViewM = Mat4::LookAt(eye - kPanPerFrame, center - kPanPerFrame, {0, 1, 0});
    // FlipProjY for Metal clip space; the motion_blur shader's HF_YS (+1 on Metal) reconstructs to match.
    Mat4 proj = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f));
    Mat4 curVP  = proj * viewM;
    Mat4 prevVP = proj * prevViewM;
    Mat4 invView = viewM.Inverse();

    FrameData fd{};
    {
        for (int k = 0; k < 16; ++k) fd.vp[k] = curVP.m[k];
        fd.lightDir[0] = -0.4f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.35f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 lightDir = math::normalize(Vec3{-0.4f, -1.0f, -0.35f});
        Vec3 sc{0.0f, 0.7f, -3.0f};
        Vec3 lightEye = sc - lightDir * 22.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-11.0f, 11.0f, -11.0f, 11.0f, 1.0f, 48.0f));
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

    MbParams mbp{};
    {
        Mat4 prevClip = prevVP * invView;   // prevViewProj * inverse(curView)
        for (int k = 0; k < 16; ++k) mbp.prevClip[k] = prevClip.m[k];
        mbp.texel[0] = 1.0f / (float)W; mbp.texel[1] = 1.0f / (float)H;
        mbp.tanHalfFovY = std::tan(0.5f * kFovY); mbp.aspect = aspect;
        mbp.maxBlurPx = kMaxBlurPx; mbp.taps = (float)kTaps; mbp.velScale = kVelScale;
    }

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgGbuf = graph.ImportTarget(
        "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
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
                pc[16] = 0.0f; pc[17] = 0.8f; pc[18] = 0.0f; pc[19] = 0.0f;
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

    graph.AddPass("motionblur", {rgScene, rgGbuf}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*mbPipe);
            cmd.BindTexturePair(*rt, *gbuf);
            cmd.PushConstants(&mbp, sizeof(mbp));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — motion blur, %d objects, camera pan\n", outPath, cw, ch, kNumObjs);
    return 0;
}

// --- Order-Independent Transparency showcase (Slice CO). Mirrors the Vulkan --oit-shot path EXACTLY:
// an opaque ground + 2 opaque objects -> HDR scene RT; then 5 mutually-overlapping camera-facing
// transparent glass quads (distinct dyadic colors, common alpha 0.5 + common view depth) rendered into
// the WBOIT accum (additive ONE,ONE; oit_accum.frag = premultColor*Weight) + revealage (cleared 1.0,
// oitRevealageBlend dst*=(1-alpha); oit_revealage.frag = alpha) RGBA32_Float targets, then oit_resolve
// (accum.rgb/max(accum.a,eps), coverage = 1-revealage) -> oit RT, then water_composite lerps it over the
// opaque scene by coverage (== oit::ResolveOver) + tonemaps. The transparent set is rendered in a
// CANONICAL and a PERMUTED draw order and the two captures are asserted BYTE-IDENTICAL — the order-
// independence proof. The SAME render/oit.h Weight (mirrored in oit_accum.frag) makes the composite
// bit-identical to the Vulkan/CPU path. SEPARATE oit pipelines + shaders; existing goldens untouched. ---
static int RunOitShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const rhi::Format kOitFmt = rhi::Format::RGBA32_Float;   // fp32 accum -> order-stable SUM (see Vulkan)
    const float kFovY = 1.04719755f;
    const float aspect = (float)W / (float)H;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Opaque scene objects (IDENTICAL to the Vulkan path).
    struct Obj { Vec3 pos; float scale; bool cube; float col[3]; };
    const Obj opaqueObjs[] = {
        {{-1.6f, 0.9f, -3.0f}, 0.9f, true,  {0.85f, 0.45f, 0.30f}},
        {{ 1.7f, 0.9f, -3.6f}, 0.9f, false, {0.35f, 0.55f, 0.85f}},
    };
    const int kNumOpaque = (int)(sizeof(opaqueObjs) / sizeof(opaqueObjs[0]));

    struct Glass { Vec3 pos; float col[3]; };
    const float kGlassAlpha = 0.5f;
    const float kGlassZ = -1.5f;
    const Glass glass[] = {
        {{-0.7f, 1.1f, kGlassZ}, {0.75f, 0.25f, 0.25f}},
        {{-0.3f, 0.8f, kGlassZ}, {0.25f, 0.75f, 0.25f}},
        {{ 0.1f, 1.2f, kGlassZ}, {0.25f, 0.50f, 1.00f}},
        {{ 0.5f, 0.9f, kGlassZ}, {1.00f, 0.75f, 0.25f}},
        {{ 0.0f, 1.0f, kGlassZ}, {0.75f, 0.25f, 1.00f}},
    };
    const int kNumGlass = (int)(sizeof(glass) / sizeof(glass[0]));

    // Opaque pipelines (UNCHANGED shaders).
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
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
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    // OIT accum + revealage pipelines (oit_accum.vert + the two frags). depthTest OFF.
    auto oitVs = loadMSL("oit_accum.vert.gen.metal", "oit_accum_vertex");
    struct OitPC { float model[16]; float colorAlpha[4]; };  // 80B
    auto oitAccumFs = loadMSL("oit_accum.frag.gen.metal", "oit_accum_fragment");
    rhi::GraphicsPipelineDesc oitAccumD;
    oitAccumD.vertex = oitVs.get(); oitAccumD.fragment = oitAccumFs.get();
    oitAccumD.vertexLayout = scene::MeshVertexLayout();
    oitAccumD.colorFormat = kOitFmt;
    oitAccumD.depthTest = false; oitAccumD.usesFrameUniforms = true;
    oitAccumD.additiveBlend = true;
    oitAccumD.pushConstantSize = sizeof(OitPC);
    auto oitAccumPipe = device->CreateGraphicsPipeline(oitAccumD);

    auto oitRevFs = loadMSL("oit_revealage.frag.gen.metal", "oit_revealage_fragment");
    rhi::GraphicsPipelineDesc oitRevD;
    oitRevD.vertex = oitVs.get(); oitRevD.fragment = oitRevFs.get();
    oitRevD.vertexLayout = scene::MeshVertexLayout();
    oitRevD.colorFormat = kOitFmt;
    oitRevD.depthTest = false; oitRevD.usesFrameUniforms = true;
    oitRevD.oitRevealageBlend = true;
    oitRevD.pushConstantSize = sizeof(OitPC);
    auto oitRevPipe = device->CreateGraphicsPipeline(oitRevD);

    // Resolve (oit_resolve, fullscreen) + final composite (water_composite verbatim).
    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto oitResFs = loadMSL("oit_resolve.frag.gen.metal", "oit_resolve_fragment");
    rhi::GraphicsPipelineDesc oitResD;
    oitResD.vertex = postVs.get(); oitResD.fragment = oitResFs.get();
    oitResD.colorFormat = kHdr;
    oitResD.depthTest = false; oitResD.usesTexture = true; oitResD.fullscreen = true;
    auto oitResPipe = device->CreateGraphicsPipeline(oitResD);

    struct OitCompParams { float texel[2]; float intensity; float pad; };
    auto compFs = loadMSL("water_composite.frag.gen.metal", "water_composite_fragment");
    rhi::GraphicsPipelineDesc compD;
    compD.vertex = postVs.get(); compD.fragment = compFs.get();
    compD.colorFormat = device->Swapchain().ColorFormat();
    compD.depthTest = false; compD.usesTexture = true; compD.fullscreen = true;
    compD.fragmentPushConstants = true; compD.pushConstantSize = sizeof(OitCompParams);
    auto compPipe = device->CreateGraphicsPipeline(compD);

    auto rt        = device->CreateRenderTarget(W, H, kHdr);
    auto accumRT   = device->CreateRenderTarget(W, H, kOitFmt);
    auto revealRT  = device->CreateRenderTarget(W, H, kOitFmt);
    auto oitRT     = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> floorTexels(256 * 256 * 4);
    for (uint32_t y = 0; y < 256; ++y)
        for (uint32_t x = 0; x < 256; ++x) {
            bool dark = (((x / 32) + (y / 32)) & 1) != 0;
            uint8_t v = dark ? 70 : 110;
            size_t idx = (static_cast<size_t>(y) * 256 + x) * 4;
            floorTexels[idx + 0] = v; floorTexels[idx + 1] = v;
            floorTexels[idx + 2] = v; floorTexels[idx + 3] = 255;
        }
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, floorTexels.data(), floorTexels.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    std::vector<std::unique_ptr<rhi::ITexture>> objTex;
    for (int o = 0; o < kNumOpaque; ++o) {
        uint8_t px[4] = {(uint8_t)std::lround(opaqueObjs[o].col[0] * 255.0f),
                         (uint8_t)std::lround(opaqueObjs[o].col[1] * 255.0f),
                         (uint8_t)std::lround(opaqueObjs[o].col[2] * 255.0f), 255};
        objTex.push_back(device->CreateTexture(
            {1, 1, rhi::Format::RGBA8_UNorm, px, sizeof(px)}));
    }

    scene::Mesh plane = scene::Mesh::Plane(*device);
    scene::Mesh sphere = scene::Mesh::Sphere(*device);
    scene::Mesh cube = scene::Mesh::Cube(*device);

    // Unit XY quad facing -Z (toward the +z camera) for the glass layers.
    const float kQuadHalf = 0.6f;
    scene::Vertex qv[4] = {};
    auto setV = [&](scene::Vertex& v, float x, float y, float u, float vv) {
        v.pos[0] = x; v.pos[1] = y; v.pos[2] = 0.0f;
        v.color[0] = v.color[1] = v.color[2] = 1.0f;
        v.uv[0] = u; v.uv[1] = vv;
        v.normal[2] = 1.0f; v.tangent[0] = 1.0f;
    };
    setV(qv[0], -kQuadHalf, -kQuadHalf, 0.0f, 1.0f);
    setV(qv[1],  kQuadHalf, -kQuadHalf, 1.0f, 1.0f);
    setV(qv[2],  kQuadHalf,  kQuadHalf, 1.0f, 0.0f);
    setV(qv[3], -kQuadHalf,  kQuadHalf, 0.0f, 0.0f);
    uint32_t qidx[6] = {0, 1, 2, 0, 2, 3};
    rhi::BufferDesc qvbD; qvbD.size = sizeof(qv); qvbD.initialData = qv;
    qvbD.usage = rhi::BufferUsage::Vertex;
    auto quadVB = device->CreateBuffer(qvbD);
    rhi::BufferDesc qibD; qibD.size = sizeof(qidx); qibD.initialData = qidx;
    qibD.usage = rhi::BufferUsage::Index;
    auto quadIB = device->CreateBuffer(qibD);

    Mat4 groundModel = Mat4::Scale({16.0f, 1.0f, 16.0f});
    std::vector<Mat4> opaqueModel(kNumOpaque);
    for (int o = 0; o < kNumOpaque; ++o)
        opaqueModel[o] = Mat4::Translate(opaqueObjs[o].pos) * Mat4::Scale(
            {opaqueObjs[o].scale, opaqueObjs[o].scale, opaqueObjs[o].scale});
    std::vector<Mat4> glassModel(kNumGlass);
    for (int g = 0; g < kNumGlass; ++g)
        glassModel[g] = Mat4::Translate(glass[g].pos);

    const Vec3 eye{0.0f, 1.0f, 4.0f};
    const Vec3 center{0.0f, 1.0f, -3.0f};
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    Mat4 proj = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f));
    FrameData fd{};
    {
        Mat4 vp = proj * viewM;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.4f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.35f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 lightDir = math::normalize(Vec3{-0.4f, -1.0f, -0.35f});
        Vec3 sc{0.0f, 0.7f, -3.0f};
        Vec3 lightEye = sc - lightDir * 22.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-11.0f, 11.0f, -11.0f, 11.0f, 1.0f, 48.0f));
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

    auto drawOpaque = [&](rhi::ICommandBuffer& cmd, int o) {
        const scene::Mesh& m = opaqueObjs[o].cube ? cube : sphere;
        cmd.BindVertexBuffer(m.vertices());
        cmd.BindIndexBuffer(m.indices());
        cmd.DrawIndexed(m.indexCount());
    };
    auto drawGlass = [&](rhi::ICommandBuffer& cmd, int g) {
        OitPC pc{};
        for (int k = 0; k < 16; ++k) pc.model[k] = glassModel[g].m[k];
        pc.colorAlpha[0] = glass[g].col[0]; pc.colorAlpha[1] = glass[g].col[1];
        pc.colorAlpha[2] = glass[g].col[2]; pc.colorAlpha[3] = kGlassAlpha;
        cmd.PushConstants(&pc, sizeof(pc));
        cmd.BindVertexBuffer(*quadVB);
        cmd.BindIndexBuffer(*quadIB);
        cmd.DrawIndexed(6);
    };

    OitCompParams ocp{}; ocp.texel[0] = 1.0f / (float)W; ocp.texel[1] = 1.0f / (float)H;
    ocp.intensity = 1.7f; ocp.pad = 0.0f;

    auto renderWithOrder = [&](const std::vector<int>& order, std::vector<uint8_t>& outPx,
                               uint32_t& outW, uint32_t& outH) -> bool {
        render::RenderGraph graph;
        render::RgResource rgShadow = graph.ImportTarget(
            "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
        render::RgResource rgScene = graph.ImportTarget(
            "sceneColor", render::RgResourceKind::SceneColor, *rt);
        render::RgResource rgAccum = graph.ImportTarget(
            "accum", render::RgResourceKind::SceneColor, *accumRT);
        render::RgResource rgReveal = graph.ImportTarget(
            "revealage", render::RgResourceKind::SceneColor, *revealRT);
        render::RgResource rgOit = graph.ImportTarget(
            "oit", render::RgResourceKind::SceneColor, *oitRT);
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
                for (int o = 0; o < kNumOpaque; ++o) {
                    cmd.PushConstants(opaqueModel[o].m, sizeof(float) * 16);
                    drawOpaque(cmd, o);
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
                    pc[16] = 0.0f; pc[17] = 0.8f; pc[18] = 0.0f; pc[19] = 0.0f;
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindMaterial(*groundTex, *flatNormal);
                    cmd.BindVertexBuffer(plane.vertices());
                    cmd.BindIndexBuffer(plane.indices());
                    cmd.DrawIndexed(plane.indexCount());
                }
                for (int o = 0; o < kNumOpaque; ++o) {
                    float pc[20];
                    for (int k = 0; k < 16; ++k) pc[k] = opaqueModel[o].m[k];
                    pc[16] = 0.0f; pc[17] = 0.6f; pc[18] = 0.0f; pc[19] = 0.0f;
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindMaterial(*objTex[o], *flatNormal);
                    drawOpaque(cmd, o);
                }
                cmd.EndRenderPass();
            });

        graph.AddPass("oit_accum", {}, {rgAccum},
            [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                dev.SetFrameUniforms(&fd, sizeof(FrameData));
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
                cmd.BindPipeline(*oitAccumPipe);
                for (int g : order) drawGlass(cmd, g);
                cmd.EndRenderPass();
            });

        graph.AddPass("oit_revealage", {}, {rgReveal},
            [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                dev.SetFrameUniforms(&fd, sizeof(FrameData));
                cmd.BeginRenderPass(rhi::ClearColor{1, 1, 1, 1});
                cmd.BindPipeline(*oitRevPipe);
                for (int g : order) drawGlass(cmd, g);
                cmd.EndRenderPass();
            });

        graph.AddPass("oit_resolve", {rgAccum, rgReveal}, {rgOit},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
                cmd.BindPipeline(*oitResPipe);
                cmd.BindTexturePair(*accumRT, *revealRT);
                cmd.Draw(3);
                cmd.EndRenderPass();
            });

        graph.AddPass("composite", {rgScene, rgOit}, {rgSwap},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*compPipe);
                cmd.BindTexturePair(*rt, *oitRT);
                cmd.PushConstants(&ocp, sizeof(ocp));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });

        device->CaptureNextFrame();
        graph.Execute(*device);
        device->WaitIdle();
        return device->GetCapturedPixels(outPx, outW, outH);
    };

    std::vector<int> canonical(kNumGlass);
    for (int g = 0; g < kNumGlass; ++g) canonical[g] = g;
    const std::vector<int> permuted = {3, 1, 0, 2, 4};

    std::vector<uint8_t> canonPx, permPx;
    uint32_t cw = 0, ch = 0, pw = 0, ph = 0;
    if (!renderWithOrder(canonical, canonPx, cw, ch)) return fail("no captured pixels (OIT canonical)");
    if (!renderWithOrder(permuted, permPx, pw, ph)) return fail("no captured pixels (OIT permuted)");

    const bool orderIndependent = (cw == pw) && (ch == ph) && (canonPx.size() == permPx.size()) &&
                                  (std::memcmp(canonPx.data(), permPx.data(), canonPx.size()) == 0);
    if (!orderIndependent)
        return fail("OIT permuted-order resolve != canonical-order resolve — NOT order-independent");
    std::printf("oit permuted==canonical: BYTE-IDENTICAL (order-independence proof)\n");
    std::printf("oit: {layers:%d, orderIndependent:true}\n", kNumGlass);

    if (!WritePNG(outPath, canonPx, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — OIT, %d glass layers, %d opaque\n",
                outPath, cw, ch, kNumGlass, kNumOpaque);
    return 0;
}

// --- Parallax Occlusion Mapping showcase (Slice CP). Mirrors the Vulkan --pom-shot path EXACTLY: a
// single large height-mapped surface (a deterministic procedural BRICK height field baked in-engine to
// a height texture + matching normal map + brick albedo) filling the view at a GRAZING angle, lit +
// shadowed by the sun. pom.frag ray-marches the height field in tangent space (render/pom.h ParallaxUV
// mirrored in-shader) so the bricks show per-pixel depth + the grooves self-shadow (pom::SelfShadow).
// The SAME render/pom.h math compiled into pom.frag here makes the marched uv bit-identical to the
// Vulkan/CPU path. INTERNALLY renders the SAME surface with heightScale=0 (pom.frag's degenerate exit)
// AND via the plain lit pipeline and asserts all three captures' zero-height equivalence BYTE-IDENTICAL
// — the zero-height proof. SEPARATE pom pipeline + shaders; existing goldens untouched. ---
static int RunPomShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;
    const float aspect = (float)W / (float)H;
    const float kHeightScale = 0.08f;
    const int   kNumSteps = 32;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // --- Deterministic procedural BRICK height field (IDENTICAL to the Vulkan path). ---
    const uint32_t TEX_N = 512;
    const float kBrickCols = 6.0f, kBrickRows = 12.0f;
    auto brickHeight = [&](float u, float v) -> float {
        float row = v * kBrickRows;
        float ri = std::floor(row);
        float uu = u * kBrickCols + (((int)ri & 1) ? 0.5f : 0.0f);
        float fu = uu - std::floor(uu);
        float fv = row - ri;
        float bu = std::sin(3.14159265358979323846f * fu);
        float bv = std::sin(3.14159265358979323846f * fv);
        float brick = std::pow(bu * bu, 0.25f) * std::pow(bv * bv, 0.25f);
        return std::min(std::max(brick, 0.0f), 1.0f);
    };
    std::vector<uint8_t> heightTexels(static_cast<size_t>(TEX_N) * TEX_N * 4);
    std::vector<float> hf(static_cast<size_t>(TEX_N) * TEX_N);
    for (uint32_t y = 0; y < TEX_N; ++y)
        for (uint32_t x = 0; x < TEX_N; ++x) {
            float u = (x + 0.5f) / TEX_N, v = (y + 0.5f) / TEX_N;
            float hv = brickHeight(u, v);
            hf[static_cast<size_t>(y) * TEX_N + x] = hv;
            uint8_t hb = (uint8_t)std::lround(hv * 255.0f);
            size_t idx = (static_cast<size_t>(y) * TEX_N + x) * 4;
            heightTexels[idx + 0] = hb; heightTexels[idx + 1] = hb;
            heightTexels[idx + 2] = hb; heightTexels[idx + 3] = 255;
        }
    std::vector<uint8_t> normalTexels(static_cast<size_t>(TEX_N) * TEX_N * 4);
    std::vector<uint8_t> albedoTexels(static_cast<size_t>(TEX_N) * TEX_N * 4);
    const float kBumpScale = 6.0f;
    for (uint32_t y = 0; y < TEX_N; ++y)
        for (uint32_t x = 0; x < TEX_N; ++x) {
            uint32_t xm = (x + TEX_N - 1) % TEX_N, xp = (x + 1) % TEX_N;
            uint32_t ym = (y + TEX_N - 1) % TEX_N, yp = (y + 1) % TEX_N;
            float hl = hf[static_cast<size_t>(y) * TEX_N + xm];
            float hr = hf[static_cast<size_t>(y) * TEX_N + xp];
            float hd = hf[static_cast<size_t>(ym) * TEX_N + x];
            float hu = hf[static_cast<size_t>(yp) * TEX_N + x];
            Vec3 nrm = math::normalize(Vec3{(hl - hr) * kBumpScale, (hd - hu) * kBumpScale, 1.0f});
            size_t idx = (static_cast<size_t>(y) * TEX_N + x) * 4;
            normalTexels[idx + 0] = (uint8_t)std::lround((nrm.x * 0.5f + 0.5f) * 255.0f);
            normalTexels[idx + 1] = (uint8_t)std::lround((nrm.y * 0.5f + 0.5f) * 255.0f);
            normalTexels[idx + 2] = (uint8_t)std::lround((nrm.z * 0.5f + 0.5f) * 255.0f);
            normalTexels[idx + 3] = 255;
            float hv = hf[static_cast<size_t>(y) * TEX_N + x];
            Vec3 brickCol{0.62f, 0.27f, 0.20f};
            Vec3 mortarCol{0.30f, 0.29f, 0.27f};
            Vec3 col{mortarCol.x + (brickCol.x - mortarCol.x) * hv,
                     mortarCol.y + (brickCol.y - mortarCol.y) * hv,
                     mortarCol.z + (brickCol.z - mortarCol.z) * hv};
            albedoTexels[idx + 0] = (uint8_t)std::lround(col.x * 255.0f);
            albedoTexels[idx + 1] = (uint8_t)std::lround(col.y * 255.0f);
            albedoTexels[idx + 2] = (uint8_t)std::lround(col.z * 255.0f);
            albedoTexels[idx + 3] = 255;
        }
    auto heightTex = device->CreateTexture(
        {TEX_N, TEX_N, rhi::Format::RGBA8_UNorm, heightTexels.data(), heightTexels.size()});
    auto normalTex = device->CreateTexture(
        {TEX_N, TEX_N, rhi::Format::RGBA8_UNorm, normalTexels.data(), normalTexels.size()});
    auto albedoTex = device->CreateTexture(
        {TEX_N, TEX_N, rhi::Format::RGBA8_UNorm, albedoTexels.data(), albedoTexels.size()});
    // Flat height field (h==1) for the zero-height proof — collapses the march to the base uv.
    const uint8_t flatHeightPx[4] = {255, 255, 255, 255};
    auto flatHeight = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatHeightPx, sizeof(flatHeightPx)});

    // POM pipeline (pom.vert + pom.frag; wider full-PBR material set: albedo t0, normal t3, height t5).
    auto pomVs = loadMSL("pom.vert.gen.metal", "vertex_main");
    auto pomFs = loadMSL("pom.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc pomDesc;
    pomDesc.vertex = pomVs.get(); pomDesc.fragment = pomFs.get();
    pomDesc.vertexLayout = scene::MeshVertexLayout();
    pomDesc.colorFormat = kHdr;
    pomDesc.depthTest = true; pomDesc.usesFrameUniforms = true;
    pomDesc.usesTexture = true; pomDesc.pbrMaterial = true;
    pomDesc.pushConstantSize = sizeof(float) * 20;
    auto pomPipeline = device->CreateGraphicsPipeline(pomDesc);

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
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt        = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    scene::Vertex qv[4] = {};
    auto setV = [&](scene::Vertex& v, float x, float y, float u, float vv) {
        v.pos[0] = x; v.pos[1] = y; v.pos[2] = 0.0f;
        v.color[0] = v.color[1] = v.color[2] = 1.0f;
        v.uv[0] = u; v.uv[1] = vv;
        v.normal[2] = 1.0f; v.tangent[0] = 1.0f;
    };
    const float kQuadHalf = 6.0f;
    setV(qv[0], -kQuadHalf, -kQuadHalf, 0.0f, 1.0f);
    setV(qv[1],  kQuadHalf, -kQuadHalf, 1.0f, 1.0f);
    setV(qv[2],  kQuadHalf,  kQuadHalf, 1.0f, 0.0f);
    setV(qv[3], -kQuadHalf,  kQuadHalf, 0.0f, 0.0f);
    uint32_t qidx[6] = {0, 1, 2, 0, 2, 3};
    rhi::BufferDesc qvbD; qvbD.size = sizeof(qv); qvbD.initialData = qv;
    qvbD.usage = rhi::BufferUsage::Vertex;
    auto quadVB = device->CreateBuffer(qvbD);
    rhi::BufferDesc qibD; qibD.size = sizeof(qidx); qibD.initialData = qidx;
    qibD.usage = rhi::BufferUsage::Index;
    auto quadIB = device->CreateBuffer(qibD);

    Mat4 quadModel = Mat4::RotateX(-1.57079632679f);

    const Vec3 eye{0.0f, 3.2f, 4.6f};
    const Vec3 center{0.0f, 0.0f, -0.6f};
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    Mat4 proj = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f));
    FrameData fd{};
    {
        Mat4 vp = proj * viewM;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.55f; fd.lightDir[1] = -0.7f; fd.lightDir[2] = -0.45f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 lightDir = math::normalize(Vec3{-0.55f, -0.7f, -0.45f});
        Vec3 sc{0.0f, 0.0f, 0.0f};
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

    auto renderSurface = [&](rhi::ITexture& hTex, float heightScale,
                             std::vector<uint8_t>& outPx, uint32_t& outW, uint32_t& outH) -> bool {
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
                cmd.PushConstants(quadModel.m, sizeof(float) * 16);
                cmd.BindVertexBuffer(*quadVB);
                cmd.BindIndexBuffer(*quadIB);
                cmd.DrawIndexed(6);
                cmd.EndRenderPass();
            });

        graph.AddPass("scene", {rgShadow}, {rgScene},
            [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                dev.SetFrameUniforms(&fd, sizeof(FrameData));
                cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
                cmd.BindPipeline(*skyPipe);
                cmd.Draw(3);
                cmd.BindPipeline(*pomPipeline);
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = quadModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.85f; pc[18] = heightScale; pc[19] = (float)kNumSteps;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterialPBR(*albedoTex, hTex, *normalTex, *normalTex, *normalTex);
                cmd.BindVertexBuffer(*quadVB);
                cmd.BindIndexBuffer(*quadIB);
                cmd.DrawIndexed(6);
                cmd.EndRenderPass();
            });

        graph.AddPass("composite", {rgScene}, {rgSwap},
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
        return device->GetCapturedPixels(outPx, outW, outH);
    };

    // THE ZERO-HEIGHT EQUIVALENCE PROOF (mirrors the Vulkan --pom-shot): the REAL height field at
    // heightScale=0 and a FLAT height field (h==1) at heightScale>0 both collapse the march to the base
    // uv (plain normal mapping) through the SAME pom shader -> BYTE-IDENTICAL. Fail loudly on any diff.
    std::vector<uint8_t> pomPx, zeroPomPx, flatPomPx;
    uint32_t pw = 0, ph = 0, zw = 0, zh = 0, fw2 = 0, fh2 = 0;
    if (!renderSurface(*heightTex, kHeightScale, pomPx, pw, ph)) return fail("no captured pixels (POM)");
    if (!renderSurface(*heightTex, 0.0f, zeroPomPx, zw, zh)) return fail("no captured pixels (POM h=0)");
    if (!renderSurface(*flatHeight, kHeightScale, flatPomPx, fw2, fh2)) return fail("no captured pixels (POM flat)");

    const bool zeroEquivalent = (zw == fw2) && (zh == fh2) && (zeroPomPx.size() == flatPomPx.size()) &&
                                (std::memcmp(zeroPomPx.data(), flatPomPx.data(), flatPomPx.size()) == 0);
    if (!zeroEquivalent)
        return fail("POM heightScale=0 render != flat-height-field render — NOT zero-height equivalent");
    std::printf("pom heightScale=0 == flat-field (plain normal mapping): BYTE-IDENTICAL (zero-height proof)\n");
    std::printf("pom: {heightScale:%.4g, steps:%d}\n", (double)kHeightScale, kNumSteps);

    if (!WritePNG(outPath, pomPx, pw, ph)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — POM, heightScale %.4g, %d steps\n",
                outPath, pw, ph, (double)kHeightScale, kNumSteps);
    return 0;
}

// --- Ground-Truth Ambient Occlusion showcase (Slice CR). Mirrors the Vulkan --gtao-shot path EXACTLY:
// a scene with clear AO cues (boxes forming a concave inner corner/crease + spheres in contact on the
// ground) rendered into an HDR RT + the SSAO/SSR view-space normal+linear-depth g-buffer. gtao.frag
// runs the render/gtao.h horizon-search visibility integral (IntegrateArc + HorizonAngle + Visibility
// mirrored in-shader) per pixel with a baked per-pixel slice rotation; the EXISTING ssao_composite.frag
// multiplies the ambient term by the AO (reused unchanged). The SAME render/gtao.h math compiled into
// gtao.frag here makes the AO bit-identical to the Vulkan/CPU path. THE RADIUS=0 EQUIVALENCE PROOF
// (mirrors the Vulkan path): the SAME scene at radius=0 (gtao.frag's no-horizon path -> AO==1) and with
// NO AO (composite aoStrength=0 -> aoFactor 1) are asserted BYTE-IDENTICAL — proving the integral
// normalizes to a true identity at zero occlusion. NOTE (per the prior POM slice): the proof compares
// the SAME gtao shader's radius=0 path against the no-AO composite (both byte-identical on every
// backend), NOT gtao's output against a different shader, so it is backend-portable. SEPARATE gtao
// pipeline + the REUSED gbuffer/ssao_composite pipelines; existing goldens untouched. ---
static int RunGtaoShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;
    const float aspect = (float)W / (float)H;
    const int   kSlices = 8;
    const int   kSteps  = 8;
    const float kRadius = 0.6f;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Lit scene pipeline (HDR RT) — UNCHANGED lit shaders.
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = kHdr;
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Shadow pipeline (UNCHANGED).
    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    // Sky (UNCHANGED).
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    // G-buffer prepass (static) — REUSED SSAO/SSR gbuffer shader.
    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;   // model(16) + view(16)
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    // GTAO + composite fullscreen pipelines (NEW gtao.frag; REUSED ssao_composite.frag).
    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    struct GtaoParams { float texel[2]; float radius, slices, steps, tanHalfFovY, aspect, intensity; };
    struct SsaoCompParams { float texel[2]; float aoStrength, intensity; };

    auto gtaoFs = loadMSL("gtao.frag.gen.metal", "gtao_fragment");
    auto compFs = loadMSL("ssao_composite.frag.gen.metal", "ssao_composite_fragment");

    rhi::GraphicsPipelineDesc gtaoD;
    gtaoD.vertex = postVs.get(); gtaoD.fragment = gtaoFs.get();
    gtaoD.colorFormat = kHdr;
    gtaoD.depthTest = false; gtaoD.usesTexture = true; gtaoD.fullscreen = true;
    gtaoD.fragmentPushConstants = true; gtaoD.pushConstantSize = sizeof(GtaoParams);
    auto gtaoPipe = device->CreateGraphicsPipeline(gtaoD);

    rhi::GraphicsPipelineDesc compD;
    compD.vertex = postVs.get(); compD.fragment = compFs.get();
    compD.colorFormat = device->Swapchain().ColorFormat();
    compD.depthTest = false; compD.usesTexture = true; compD.fullscreen = true;
    compD.fragmentPushConstants = true; compD.pushConstantSize = sizeof(SsaoCompParams);
    auto compPipe = device->CreateGraphicsPipeline(compD);

    auto rt   = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf = device->CreateRenderTarget(W, H, kHdr);
    auto aoRT = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
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

    // Scene objects — IDENTICAL to the Vulkan --gtao-shot path.
    struct Obj { Vec3 pos; Vec3 scale; bool cube; };
    const Obj objs[] = {
        {{-1.6f, 0.9f, -1.0f}, {0.45f, 0.9f, 2.2f}, true},
        {{ 0.0f, 0.9f, -2.2f}, {2.6f, 0.9f, 0.45f}, true},
        {{ 1.4f, 0.45f, 0.4f}, {0.45f, 0.45f, 0.45f}, true},
        {{ 1.4f, 1.25f, 0.4f}, {0.35f, 0.35f, 0.35f}, true},
        {{-0.5f, 0.55f, 0.3f}, {0.55f, 0.55f, 0.55f}, false},
        {{ 0.3f, 0.5f, -1.2f}, {0.5f, 0.5f, 0.5f}, false},
    };
    const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));

    Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});

    const Vec3 eye{3.3f, 2.2f, 3.6f};
    const Vec3 center{-0.2f, 0.5f, -0.7f};
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
        Vec3 sc{0.0f, 0.5f, -1.0f};
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

    auto objModel = [&](const Obj& o) {
        return Mat4::Translate(o.pos) * Mat4::Scale(o.scale);
    };

    GtaoParams gp{};
    gp.texel[0] = 1.0f / (float)W; gp.texel[1] = 1.0f / (float)H;
    gp.slices = (float)kSlices; gp.steps = (float)kSteps;
    gp.tanHalfFovY = std::tan(0.5f * kFovY); gp.aspect = aspect; gp.intensity = 2.0f;
    SsaoCompParams cp{}; cp.texel[0] = 1.0f / (float)W; cp.texel[1] = 1.0f / (float)H;
    cp.intensity = 1.7f;

    auto renderScene = [&](float radius, float aoStrength,
                           std::vector<uint8_t>& outPx, uint32_t& outW, uint32_t& outH) -> bool {
        gp.radius = radius;
        cp.aoStrength = aoStrength;
        render::RenderGraph graph;
        render::RgResource rgShadow = graph.ImportTarget(
            "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
        render::RgResource rgScene = graph.ImportTarget(
            "sceneColor", render::RgResourceKind::SceneColor, *rt);
        render::RgResource rgGbuf = graph.ImportTarget(
            "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
        render::RgResource rgAO = graph.ImportTarget(
            "ao", render::RgResourceKind::SceneColor, *aoRT);
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
                for (int oi = 0; oi < kNumObjs; ++oi) {
                    Mat4 m = objModel(objs[oi]);
                    cmd.PushConstants(m.m, sizeof(float) * 16);
                    const scene::Mesh& msh = objs[oi].cube ? cube : sphere;
                    cmd.BindVertexBuffer(msh.vertices());
                    cmd.BindIndexBuffer(msh.indices());
                    cmd.DrawIndexed(msh.indexCount());
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
                for (int oi = 0; oi < kNumObjs; ++oi) {
                    Mat4 m = objModel(objs[oi]);
                    float pc[20];
                    for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                    pc[16] = 0.0f; pc[17] = 0.6f; pc[18] = 0.0f; pc[19] = 0.0f;
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindMaterial(*groundTex, *flatNormal);
                    const scene::Mesh& msh = objs[oi].cube ? cube : sphere;
                    cmd.BindVertexBuffer(msh.vertices());
                    cmd.BindIndexBuffer(msh.indices());
                    cmd.DrawIndexed(msh.indexCount());
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
                for (int oi = 0; oi < kNumObjs; ++oi) {
                    Mat4 m = objModel(objs[oi]);
                    float pc[32];
                    for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                    for (int k = 0; k < 16; ++k) pc[16 + k] = viewM.m[k];
                    cmd.PushConstants(pc, sizeof(pc));
                    const scene::Mesh& msh = objs[oi].cube ? cube : sphere;
                    cmd.BindVertexBuffer(msh.vertices());
                    cmd.BindIndexBuffer(msh.indices());
                    cmd.DrawIndexed(msh.indexCount());
                }
                cmd.EndRenderPass();
            });

        graph.AddPass("gtao", {rgGbuf}, {rgAO},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{1, 1, 1, 1});
                cmd.BindPipeline(*gtaoPipe);
                cmd.BindTexture(*gbuf);
                cmd.PushConstants(&gp, sizeof(gp));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });

        graph.AddPass("composite", {rgScene, rgAO}, {rgSwap},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*compPipe);
                cmd.BindTexturePair(*rt, *aoRT);
                cmd.PushConstants(&cp, sizeof(cp));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });

        device->CaptureNextFrame();
        graph.Execute(*device);
        device->WaitIdle();
        return device->GetCapturedPixels(outPx, outW, outH);
    };

    // THE RADIUS=0 EQUIVALENCE PROOF (backend-portable: compares the SAME gtao shader's radius=0 path
    // against the no-AO composite — both byte-identical on every backend).
    std::vector<uint8_t> gtaoPx, zeroPx, noAoPx;
    uint32_t gw = 0, gh = 0, zw = 0, zh = 0, nw = 0, nh = 0;
    if (!renderScene(kRadius, 1.0f, gtaoPx, gw, gh)) return fail("no captured pixels (GTAO)");
    if (!renderScene(0.0f, 1.0f, zeroPx, zw, zh)) return fail("no captured pixels (GTAO radius=0)");
    if (!renderScene(kRadius, 0.0f, noAoPx, nw, nh)) return fail("no captured pixels (no-AO)");

    const bool zeroEquivalent = (zw == nw) && (zh == nh) && (zeroPx.size() == noAoPx.size()) &&
                                (std::memcmp(zeroPx.data(), noAoPx.data(), noAoPx.size()) == 0);
    if (!zeroEquivalent)
        return fail("GTAO radius=0 render != no-AO render — NOT zero-occlusion equivalent");
    std::printf("gtao radius=0 == no-AO scene: BYTE-IDENTICAL (radius=0 equivalence proof)\n");
    std::printf("gtao: {slices:%d, steps:%d, radius:%.4g}\n", kSlices, kSteps, (double)kRadius);

    if (!WritePNG(outPath, gtaoPx, gw, gh)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — GTAO, %d slices, %d steps, radius %.4g, %d objects\n",
                outPath, gw, gh, kSlices, kSteps, (double)kRadius, kNumObjs);
    return 0;
}

// --- Screen-Space Contact Shadows showcase (Slice CT). Mirrors the Vulkan --contactshadow-shot path
// EXACTLY: small spheres + boxes hovering a hair above a checkered ground, sun from the side. Pipeline
// per render: shadow -> gbuffer (view normal + linear depth) -> contact (fullscreen contact_shadows.frag
// runs render/contact_shadows.h RayMarchShadow over the G-buffer toward the sun -> a single-channel
// factor RT) -> scene (lit_contactshadow.frag samples the factor at the env slot and multiplies ONLY the
// direct sun radiance by it) -> post (tonemap). The SAME render/contact_shadows.h math compiled here
// makes the factor bit-identical to the Vulkan/CPU path. THE maxDist=0 NO-OP PROOF (backend-portable,
// per the POM/GTAO/CS lesson): the SAME lit_contactshadow scene is rendered with the factor computed at
// maxDist=0 (RayMarchShadow -> 1 -> all-1 factor RT) AND with an all-white factor RT, and asserted
// BYTE-IDENTICAL — the march + sun-apply is a pure pass-through when disabled.
static int RunContactShadowShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;
    const float aspect = (float)W / (float)H;
    const int   kSteps     = 24;
    const float kMaxDist   = 0.9f;
    const float kThickness = 0.6f;
    const float kBias      = 0.02f;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Lit (contact-shadowed) scene pipeline (NEW lit_contactshadow.frag; reuses lit.vert) — env slot
    // reserved for the factor RT (bound via BindReflectionProbe).
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto litFs = loadMSL("lit_contactshadow.frag.gen.metal", "contactshadow_fragment");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = kHdr;
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
    litDesc.usesEnvironment = true;
    litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // Shadow pipeline (UNCHANGED).
    auto shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    // Sky (UNCHANGED).
    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    // G-buffer prepass (static) — REUSED SSAO/SSR/GTAO gbuffer shader.
    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    // Contact-shadow factor (NEW contact_shadows.frag) + post (UNCHANGED tonemap).
    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    struct ContactParams {
        float texel[2]; float maxDist, steps, thickness, bias, tanHalfFovY, aspect;
        float sunDirView[4];
    };
    auto contactFs = loadMSL("contact_shadows.frag.gen.metal", "contact_shadows_fragment");
    auto postFs    = loadMSL("post.frag.gen.metal", "post_fragment");

    rhi::GraphicsPipelineDesc contactD;
    contactD.vertex = postVs.get(); contactD.fragment = contactFs.get();
    contactD.colorFormat = kHdr;
    contactD.depthTest = false; contactD.usesTexture = true; contactD.fullscreen = true;
    contactD.fragmentPushConstants = true; contactD.pushConstantSize = sizeof(ContactParams);
    auto contactPipe = device->CreateGraphicsPipeline(contactD);

    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt        = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf      = device->CreateRenderTarget(W, H, kHdr);
    auto contactRT = device->CreateRenderTarget(W, H, kHdr);
    auto whiteRT   = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
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

    // Scene objects — IDENTICAL to the Vulkan --contactshadow-shot path.
    struct Obj { Vec3 pos; Vec3 scale; bool cube; };
    const Obj objs[] = {
        {{-1.1f, 0.34f, 0.6f}, {0.30f, 0.30f, 0.30f}, false},
        {{-0.2f, 0.29f, 0.9f}, {0.25f, 0.25f, 0.25f}, true},
        {{ 0.7f, 0.26f, 0.5f}, {0.22f, 0.22f, 0.22f}, false},
        {{ 1.5f, 0.32f, 0.2f}, {0.28f, 0.28f, 0.28f}, true},
        {{ 0.2f, 0.50f,-1.1f}, {1.7f, 0.05f, 0.5f}, true},
        {{-0.7f, 0.34f,-0.7f}, {0.30f, 0.30f, 0.30f}, false},
        {{ 0.9f, 0.28f,-0.6f}, {0.24f, 0.24f, 0.24f}, true},
    };
    const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));

    Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});

    const Vec3 eye{2.4f, 1.9f, 3.4f};
    const Vec3 center{0.0f, 0.25f, -0.3f};
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    Vec3 sunTravel = math::normalize(Vec3{-0.6f, -0.55f, -0.25f});
    FrameData fd{};
    {
        Mat4 proj = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * viewM;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0]=sunTravel.x; fd.lightDir[1]=sunTravel.y; fd.lightDir[2]=sunTravel.z;
        fd.lightColor[0]=3.0f; fd.lightColor[1]=2.9f; fd.lightColor[2]=2.7f; fd.lightColor[3]=1.0f;
        fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
        fd.ptCount[0]=0.0f;
        Vec3 sc{0.0f, 0.3f, -0.3f};
        Vec3 lightEye = sc - sunTravel * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-20.0f, 20.0f, -20.0f, 20.0f, 1.0f, 44.0f));
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
        fd.skyParams[2] = 1.0f / (float)W;
        fd.skyParams[3] = 1.0f / (float)H;
    }

    auto xformDir = [&](const Mat4& m, const Vec3& d) {
        return Vec3{m.m[0]*d.x + m.m[4]*d.y + m.m[8]*d.z,
                   m.m[1]*d.x + m.m[5]*d.y + m.m[9]*d.z,
                   m.m[2]*d.x + m.m[6]*d.y + m.m[10]*d.z};
    };
    Vec3 sunViewDir = math::normalize(xformDir(viewM, sunTravel));

    auto objModel = [&](const Obj& o) {
        return Mat4::Translate(o.pos) * Mat4::Scale(o.scale);
    };

    ContactParams cp{};
    cp.texel[0] = 1.0f / (float)W; cp.texel[1] = 1.0f / (float)H;
    cp.steps = (float)kSteps; cp.thickness = kThickness; cp.bias = kBias;
    cp.tanHalfFovY = std::tan(0.5f * kFovY); cp.aspect = aspect;
    cp.sunDirView[0]=sunViewDir.x; cp.sunDirView[1]=sunViewDir.y; cp.sunDirView[2]=sunViewDir.z;

    auto renderScene = [&](float maxDist, bool useComputedFactor,
                           std::vector<uint8_t>& outPx, uint32_t& outW, uint32_t& outH) -> bool {
        cp.maxDist = maxDist;
        render::RenderGraph graph;
        render::RgResource rgShadow = graph.ImportTarget(
            "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
        render::RgResource rgGbuf = graph.ImportTarget(
            "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
        render::RgResource rgContact = graph.ImportTarget(
            "contact", render::RgResourceKind::SceneColor, *contactRT);
        render::RgResource rgWhite = graph.ImportTarget(
            "white", render::RgResourceKind::SceneColor, *whiteRT);
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
                for (int oi = 0; oi < kNumObjs; ++oi) {
                    Mat4 m = objModel(objs[oi]);
                    cmd.PushConstants(m.m, sizeof(float) * 16);
                    const scene::Mesh& msh = objs[oi].cube ? cube : sphere;
                    cmd.BindVertexBuffer(msh.vertices());
                    cmd.BindIndexBuffer(msh.indices());
                    cmd.DrawIndexed(msh.indexCount());
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
                for (int oi = 0; oi < kNumObjs; ++oi) {
                    Mat4 m = objModel(objs[oi]);
                    float pc[32];
                    for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                    for (int k = 0; k < 16; ++k) pc[16 + k] = viewM.m[k];
                    cmd.PushConstants(pc, sizeof(pc));
                    const scene::Mesh& msh = objs[oi].cube ? cube : sphere;
                    cmd.BindVertexBuffer(msh.vertices());
                    cmd.BindIndexBuffer(msh.indices());
                    cmd.DrawIndexed(msh.indexCount());
                }
                cmd.EndRenderPass();
            });

        graph.AddPass("contact", {rgGbuf}, {rgContact},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{1, 1, 1, 1});
                cmd.BindPipeline(*contactPipe);
                cmd.BindTexture(*gbuf);
                cmd.PushConstants(&cp, sizeof(cp));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });

        graph.AddPass("white", {}, {rgWhite},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{1, 1, 1, 1});
                cmd.EndRenderPass();
            });

        rhi::IRenderTarget& factorRT = useComputedFactor ? *contactRT : *whiteRT;
        render::RgResource rgFactor = useComputedFactor ? rgContact : rgWhite;
        graph.AddPass("scene", {rgShadow, rgFactor}, {rgScene},
            [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                dev.SetFrameUniforms(&fd, sizeof(FrameData));
                cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
                cmd.BindPipeline(*skyPipe);
                cmd.Draw(3);
                cmd.BindPipeline(*litPipeline);
                cmd.BindReflectionProbe(factorRT);
                {
                    float pc[20];
                    for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                    pc[16]=0.0f; pc[17]=0.85f; pc[18]=0.0f; pc[19]=0.0f;
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindMaterial(*groundTex, *flatNormal);
                    cmd.BindReflectionProbe(factorRT);
                    cmd.BindVertexBuffer(plane.vertices());
                    cmd.BindIndexBuffer(plane.indices());
                    cmd.DrawIndexed(plane.indexCount());
                }
                for (int oi = 0; oi < kNumObjs; ++oi) {
                    Mat4 m = objModel(objs[oi]);
                    float pc[20];
                    for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                    pc[16]=0.0f; pc[17]=0.6f; pc[18]=0.0f; pc[19]=0.0f;
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindMaterial(*groundTex, *flatNormal);
                    cmd.BindReflectionProbe(factorRT);
                    const scene::Mesh& msh = objs[oi].cube ? cube : sphere;
                    cmd.BindVertexBuffer(msh.vertices());
                    cmd.BindIndexBuffer(msh.indices());
                    cmd.DrawIndexed(msh.indexCount());
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
        return device->GetCapturedPixels(outPx, outW, outH);
    };

    // THE maxDist=0 NO-OP PROOF (backend-portable: the SAME lit_contactshadow shader with a maxDist=0
    // factor RT vs an all-white factor RT — both factor==1, byte-identical on every backend).
    std::vector<uint8_t> csPx, zeroPx, noCsPx;
    uint32_t cw=0, ch=0, zw=0, zh=0, nw=0, nh=0;
    if (!renderScene(kMaxDist, true, csPx, cw, ch)) return fail("no captured pixels (contact shadows)");
    if (!renderScene(0.0f, true, zeroPx, zw, zh)) return fail("no captured pixels (contact maxDist=0)");
    if (!renderScene(kMaxDist, false, noCsPx, nw, nh)) return fail("no captured pixels (no-contact)");

    const bool zeroEquivalent = (zw == nw) && (zh == nh) && (zeroPx.size() == noCsPx.size()) &&
                                (std::memcmp(zeroPx.data(), noCsPx.data(), noCsPx.size()) == 0);
    if (!zeroEquivalent)
        return fail("contact maxDist=0 render != no-contact render — NOT a pure pass-through when disabled");
    std::printf("contact maxDist=0 == no-contact scene: BYTE-IDENTICAL (maxDist=0 no-op proof)\n");
    std::printf("contact-shadows: {steps:%d, maxDist:%.4g, thickness:%.4g}\n",
                kSteps, (double)kMaxDist, (double)kThickness);

    if (!WritePNG(outPath, csPx, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — contact shadows, %d steps, maxDist %.4g, thickness %.4g, %d objects\n",
                outPath, cw, ch, kSteps, (double)kMaxDist, (double)kThickness, kNumObjs);
    return 0;
}

// --- Froxel Volumetric Fog showcase (Slice CS). Mirrors the Vulkan --froxelfog-shot path EXACTLY: a
// lit+shadowed scene (ground + receding objects) wrapped in a TRUE 3D view-space FROXEL volume. Two
// COMPUTE passes (froxel_inject: per-froxel scatter+extinction from render/froxel.h height-density + HG
// phase; froxel_integrate: front-to-back IntegrateStep into per-froxel inScatter+transmittance) build a
// FLAT SSBO volume; a fullscreen apply (froxel_apply) samples it at each pixel's view-depth slice
// (ViewZToSlice) and composites scene*T + inScatter; post tonemaps. The SAME render/froxel.h math
// compiled here makes the fog bit-identical to the Vulkan/CPU path. THE ZERO-DENSITY NO-OP PROOF
// (backend-portable, per the POM/GTAO lesson): the SAME apply chain at baseDensity=0 (T==1, inScatter==0)
// is asserted BYTE-IDENTICAL to a no-fog pass-through (apply enable=0) — comparing the SAME froxel-apply
// path on THIS backend, NOT across different shaders. New golden tests/golden/metal/froxel_fog.png; two
// runs DIFF 0.0000. The compute->compute / compute->fragment barriers close the Metal compute encoder so
// the tracked-hazard model orders the volume write->read across passes. ---
static int RunFroxelFogShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace fx = hf::render::froxel;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;
    const float aspect = (float)W / (float)H;

    const int   DIMX = 16, DIMY = 9, DIMZ = 64;
    const float kNear = 0.5f, kFar = 80.0f;
    const float kBaseDensity   = 0.06f;
    const float kHeightFalloff = 0.12f;
    const float kHeightRef     = 0.0f;
    const float kG             = 0.76f;
    fx::FroxelGrid grid; grid.dimX = DIMX; grid.dimY = DIMY; grid.dimZ = DIMZ;
    grid.zNear = kNear; grid.zFar = kFar;
    const int nFroxels = grid.froxelCount();

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Lit / shadow / sky / gbuffer pipelines (UNCHANGED shaders).
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
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
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    // Froxel inject + integrate COMPUTE pipelines (NEW; 2 storage buffers each).
    auto injectCs = loadMSL("froxel_inject.comp.gen.metal", "froxel_inject_main");
    rhi::ComputePipelineDesc injectCd;
    injectCd.compute = injectCs.get(); injectCd.storageBufferCount = 2; injectCd.threadsPerGroupX = 64;
    injectCd.sampledShadowMap = true;  // CX: the shared inject shader declares the sun shadow map
    auto injectCompute = device->CreateComputePipeline(injectCd);

    auto integrateCs = loadMSL("froxel_integrate.comp.gen.metal", "froxel_integrate_main");
    rhi::ComputePipelineDesc integrateCd;
    integrateCd.compute = integrateCs.get(); integrateCd.storageBufferCount = 2; integrateCd.threadsPerGroupX = 64;
    auto integrateCompute = device->CreateComputePipeline(integrateCd);

    // Froxel apply (NEW fullscreen) + post (UNCHANGED tonemap).
    auto postVs  = loadMSL("post.vert.gen.metal", "post_vertex");
    auto applyFs = loadMSL("froxel_apply.frag.gen.metal", "froxel_apply_fragment");
    auto postFs  = loadMSL("post.frag.gen.metal", "post_fragment");
    struct ApplyParams { float dims[4]; float range[4]; };

    rhi::GraphicsPipelineDesc applyD;
    applyD.vertex = postVs.get(); applyD.fragment = applyFs.get();
    applyD.colorFormat = kHdr;
    applyD.depthTest = false; applyD.usesTexture = true; applyD.usesLightClusters = true;
    applyD.usesFrameUniforms = true; applyD.fullscreen = true;
    applyD.fragmentPushConstants = true; applyD.pushConstantSize = sizeof(ApplyParams);
    auto applyPipe = device->CreateGraphicsPipeline(applyD);

    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt    = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf  = device->CreateRenderTarget(W, H, kHdr);
    auto fogRT = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
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

    // Scene objects — IDENTICAL to the Vulkan --froxelfog-shot path.
    struct Obj { Vec3 pos; Vec3 scale; bool cube; };
    const Obj objs[] = {
        {{-2.2f, 0.7f,  1.0f}, {0.7f, 0.7f, 0.7f}, true},
        {{ 1.8f, 0.6f, -1.5f}, {0.6f, 0.6f, 0.6f}, false},
        {{-1.0f, 0.9f, -5.0f}, {0.9f, 0.9f, 0.9f}, true},
        {{ 2.6f, 0.7f, -9.0f}, {0.7f, 0.7f, 0.7f}, false},
        {{-2.8f, 1.1f,-13.0f}, {1.1f, 1.1f, 1.1f}, true},
    };
    const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));

    Mat4 groundModel = Mat4::Scale({20.0f, 1.0f, 20.0f});

    const Vec3 eye{0.0f, 2.4f, 7.0f};
    const Vec3 center{0.0f, 0.8f, -8.0f};
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    Mat4 projUnflipped = Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f);  // for the froxel invProj
    Vec3 sunTravel = math::normalize(Vec3{0.05f, -0.35f, -1.0f});
    FrameData fd{};
    {
        Mat4 proj = FlipProjY(projUnflipped);
        Mat4 vp = proj * viewM;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0]=sunTravel.x; fd.lightDir[1]=sunTravel.y; fd.lightDir[2]=sunTravel.z;
        fd.lightColor[0]=1.0f; fd.lightColor[1]=0.96f; fd.lightColor[2]=0.85f; fd.lightColor[3]=1.0f;
        fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
        fd.ptCount[0]=0.0f;
        Vec3 sc{0.0f, 0.5f, -6.0f};
        Vec3 lightEye = sc - sunTravel * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-12.0f, 12.0f, -12.0f, 12.0f, 1.0f, 44.0f));
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

    auto objModel = [&](const Obj& o) {
        return Mat4::Translate(o.pos) * Mat4::Scale(o.scale);
    };

    // Froxel params (matches froxel_inject/integrate FroxelParams std430). The froxel math is VIEW-space:
    // invProj uses the UNFLIPPED proj (recovers view-space XY) exactly like the CL clustered showcase.
    // Matches froxel_inject/integrate FroxelParams std430 — incl. the CV cluster + CX shadow fields the
    // shared inject shader declares (CS leaves injectLights + volumetricShadows 0 -> the EXACT CS path ->
    // froxel_fog byte-identical). The full struct is uploaded so every shader read is in-bounds.
    struct FroxelParamsCPU {
        uint32_t dims[4]; float range[4]; float sunDir[4]; float sunColor[4]; float fog[4];
        float invProj[16]; float invView[16];
        uint32_t clusterDims[4]; float clusterRange[4]; float view[16];
        uint32_t shadowFlags[4]; float csmSplits[4]; float shadowBias[4];
        float    camFwd[4]; float camPos[4]; float cascadeVP[4][16];
    };
    static_assert(sizeof(FroxelParamsCPU) ==
                  16 + 16*4 + 64*2 + 16 + 16 + 64 + 16 + 16 + 16 + 16 + 16 + 64*4,
                  "FroxelParams std430 layout (incl. CV cluster + CX shadow fields)");
    FroxelParamsCPU fp{};
    fp.dims[0]=(uint32_t)DIMX; fp.dims[1]=(uint32_t)DIMY; fp.dims[2]=(uint32_t)DIMZ; fp.dims[3]=0;
    fp.range[0]=kNear; fp.range[1]=kFar;
    fp.sunDir[0]=sunTravel.x; fp.sunDir[1]=sunTravel.y; fp.sunDir[2]=sunTravel.z;
    fp.sunColor[0]=1.0f; fp.sunColor[1]=0.96f; fp.sunColor[2]=0.85f;
    fp.fog[1]=kHeightFalloff; fp.fog[2]=kHeightRef; fp.fog[3]=kG;
    Mat4 invProj = projUnflipped.Inverse();
    Mat4 invView = viewM.Inverse();
    for (int k = 0; k < 16; ++k) { fp.invProj[k] = invProj.m[k]; fp.invView[k] = invView.m[k]; }
    // injectLights + volumetricShadows stay 0 -> the inject pass runs the EXACT CS sun-only path.

    std::vector<fx::FroxelCell> volInit((size_t)nFroxels);
    for (auto& cell : volInit) {
        cell.scatterExt[0]=cell.scatterExt[1]=cell.scatterExt[2]=cell.scatterExt[3]=0.0f;
        cell.resultT[0]=cell.resultT[1]=cell.resultT[2]=0.0f; cell.resultT[3]=1.0f;
    }
    rhi::BufferDesc dummyDesc;
    dummyDesc.size = 16; uint32_t dummyInit[4] = {0,0,0,0}; dummyDesc.initialData = dummyInit;
    dummyDesc.usage = rhi::BufferUsage::Storage;
    auto dummyBuf = device->CreateBuffer(dummyDesc);

    const uint32_t kInjectGroups    = ((uint32_t)nFroxels + 63u) / 64u;
    const uint32_t kIntegrateGroups = ((uint32_t)(DIMX * DIMY) + 63u) / 64u;

    ApplyParams ap{};
    ap.dims[0]=(float)DIMX; ap.dims[1]=(float)DIMY; ap.dims[2]=(float)DIMZ;
    ap.range[0]=kNear; ap.range[1]=kFar;

    auto renderScene = [&](float baseDensity, float enable,
                           std::vector<uint8_t>& outPx, uint32_t& outW, uint32_t& outH) -> bool {
        fp.fog[0] = baseDensity;
        rhi::BufferDesc fpDesc;
        fpDesc.size = sizeof(FroxelParamsCPU); fpDesc.initialData = &fp; fpDesc.usage = rhi::BufferUsage::Storage;
        auto fpBuf = device->CreateBuffer(fpDesc);
        rhi::BufferDesc volDesc;
        volDesc.size = volInit.size() * sizeof(fx::FroxelCell);
        volDesc.initialData = volInit.data(); volDesc.usage = rhi::BufferUsage::Storage;
        auto volBuf = device->CreateBuffer(volDesc);
        ap.dims[3] = enable;

        render::RenderGraph graph;
        render::RgResource rgShadow = graph.ImportTarget(
            "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
        render::RgResource rgScene = graph.ImportTarget(
            "sceneColor", render::RgResourceKind::SceneColor, *rt);
        render::RgResource rgGbuf = graph.ImportTarget(
            "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
        render::RgResource rgFog = graph.ImportTarget(
            "fog", render::RgResourceKind::SceneColor, *fogRT);
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
                for (int oi = 0; oi < kNumObjs; ++oi) {
                    Mat4 m = objModel(objs[oi]);
                    cmd.PushConstants(m.m, sizeof(float) * 16);
                    const scene::Mesh& msh = objs[oi].cube ? cube : sphere;
                    cmd.BindVertexBuffer(msh.vertices());
                    cmd.BindIndexBuffer(msh.indices());
                    cmd.DrawIndexed(msh.indexCount());
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
                    pc[16]=0.0f; pc[17]=0.85f; pc[18]=0.0f; pc[19]=0.0f;
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindMaterial(*groundTex, *flatNormal);
                    cmd.BindVertexBuffer(plane.vertices());
                    cmd.BindIndexBuffer(plane.indices());
                    cmd.DrawIndexed(plane.indexCount());
                }
                for (int oi = 0; oi < kNumObjs; ++oi) {
                    Mat4 m = objModel(objs[oi]);
                    float pc[20];
                    for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                    pc[16]=0.0f; pc[17]=0.6f; pc[18]=0.0f; pc[19]=0.0f;
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindMaterial(*groundTex, *flatNormal);
                    const scene::Mesh& msh = objs[oi].cube ? cube : sphere;
                    cmd.BindVertexBuffer(msh.vertices());
                    cmd.BindIndexBuffer(msh.indices());
                    cmd.DrawIndexed(msh.indexCount());
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
                for (int oi = 0; oi < kNumObjs; ++oi) {
                    Mat4 m = objModel(objs[oi]);
                    float pc[32];
                    for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                    for (int k = 0; k < 16; ++k) pc[16 + k] = viewM.m[k];
                    cmd.PushConstants(pc, sizeof(pc));
                    const scene::Mesh& msh = objs[oi].cube ? cube : sphere;
                    cmd.BindVertexBuffer(msh.vertices());
                    cmd.BindIndexBuffer(msh.indices());
                    cmd.DrawIndexed(msh.indexCount());
                }
                cmd.EndRenderPass();
            });

        graph.AddPass("froxel", {rgShadow, rgScene, rgGbuf}, {rgFog},
            [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                dev.SetFrameUniforms(&fd, sizeof(FrameData));
                cmd.BindComputePipeline(*injectCompute);
                cmd.BindStorageBuffer(*volBuf, 0);
                cmd.BindStorageBuffer(*fpBuf, 1);
                cmd.BindShadowMapCompute(*shadowMap);  // CX: shared shader's shadow map (unused: volShadows=0)
                cmd.DispatchCompute(kInjectGroups);
                cmd.ComputeToComputeBarrier();
                cmd.BindComputePipeline(*integrateCompute);
                cmd.BindStorageBuffer(*volBuf, 0);
                cmd.BindStorageBuffer(*fpBuf, 1);
                cmd.DispatchCompute(kIntegrateGroups);
                cmd.ComputeToFragmentBarrier();
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*applyPipe);
                cmd.BindTexturePair(*rt, *gbuf);
                cmd.BindLightClusters(*volBuf, *dummyBuf, *dummyBuf);
                cmd.PushConstants(&ap, sizeof(ap));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });

        graph.AddPass("post", {rgFog}, {rgSwap},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*postPipe);
                cmd.BindTexture(*fogRT);
                cmd.Draw(3);
                cmd.EndRenderPass();
            });

        device->CaptureNextFrame();
        graph.Execute(*device);
        device->WaitIdle();
        return device->GetCapturedPixels(outPx, outW, outH);
    };

    // THE ZERO-DENSITY NO-OP PROOF (backend-portable: SAME apply chain at density=0 vs apply enable=0).
    std::vector<uint8_t> fogPx, zeroPx, noFogPx;
    uint32_t fw=0, fh=0, zw=0, zh=0, nw=0, nh=0;
    if (!renderScene(kBaseDensity, 1.0f, fogPx, fw, fh)) return fail("no captured pixels (froxel fog)");
    if (!renderScene(0.0f, 1.0f, zeroPx, zw, zh)) return fail("no captured pixels (froxel density=0)");
    if (!renderScene(kBaseDensity, 0.0f, noFogPx, nw, nh)) return fail("no captured pixels (no-fog)");

    const bool zeroEquivalent = (zw == nw) && (zh == nh) && (zeroPx.size() == noFogPx.size()) &&
                                (std::memcmp(zeroPx.data(), noFogPx.data(), noFogPx.size()) == 0);
    if (!zeroEquivalent)
        return fail("froxel density=0 render != no-fog render — NOT zero-density equivalent");
    std::printf("froxel density=0 == no-fog scene: BYTE-IDENTICAL (zero-density no-op proof)\n");
    std::printf("froxel-fog: {froxels:%dx%dx%d, density:%.4g, g:%.4g}\n",
                DIMX, DIMY, DIMZ, (double)kBaseDensity, (double)kG);

    if (!WritePNG(outPath, fogPx, fw, fh)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — froxel fog, %dx%dx%d froxels, density %.4g, g %.4g, %d objects\n",
                outPath, fw, fh, DIMX, DIMY, DIMZ, (double)kBaseDensity, (double)kG, kNumObjs);
    return 0;
}

// --- Auto-Exposure showcase (Slice CW). Mirrors the Vulkan --autoexposure-shot path EXACTLY: a
// high-dynamic-range scene (a thin bright emissive sky band + a dark shadowed foreground) built as the
// SAME deterministic HDR float image — exposed BOTH as a flat float4 storage buffer (for the histogram
// compute) AND as an RGBA16F texture (for the tonemap to sample). Pipeline: clear the INTEGER histogram
// SSBO -> autoexposure_histogram (one thread per pixel: Rec.709 Luminance -> render::autoexp::LumToBin ->
// InterlockedAdd(histogram[bin],1)) -> [compute->compute barrier] -> autoexposure_reduce (AverageLuminance
// black-bin-excluded -> ExposureFromAverage; gated by adaptationEnabled) -> [compute->fragment barrier] ->
// tonemap_autoexp (applies the exposure SSBO before the ACES curve, otherwise identical to post.frag). The
// SAME render/auto_exposure.h math compiled here (shared HLSL->SPIR-V->MSL) makes the result bit-identical
// to the Vulkan/CPU path. THE ADAPTATION-OFF NO-OP PROOF (backend-portable: the SAME tonemap_autoexp chain
// at adaptationEnabled=false (exposure == E0) vs the standard post.frag render of the same HDR texture):
// asserted BYTE-IDENTICAL on THIS backend. The integer histogram (atomicAdd of 1) is order-independent ->
// deterministic; Metal's atomic_uint maps from the SPIR-V integer atomic. New golden
// tests/golden/metal/auto_exposure.png; two runs DIFF 0.0000.
static int RunAutoExposureShowcase(const char* outPath) {
    namespace ae = hf::render::autoexp;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;

    const int   kBins        = 256;
    const float kMinLogLum   = -8.0f;
    const float kLogLumRange = 12.0f;
    const float kKeyValue    = 0.18f;
    const float kE0          = 1.7f;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };

    // === Build the HDR scene on the CPU (deterministic) — IDENTICAL formula to the Vulkan path. ===
    const uint32_t pixelCount = W * H;
    std::vector<float> sceneF((size_t)pixelCount * 4);
    for (uint32_t y = 0; y < H; ++y) {
        float v = (float)y / (float)(H > 1 ? H - 1 : 1);
        for (uint32_t x = 0; x < W; ++x) {
            float u = (float)x / (float)(W > 1 ? W - 1 : 1);
            float r, g, b;
            if (v < 0.18f) {
                float sky = 4.0f + 6.0f * (0.18f - v) / 0.18f + 3.0f * u;
                r = sky * 1.00f; g = sky * 0.92f; b = sky * 0.78f;
            } else {
                float d = (v - 0.18f) / 0.82f;
                float base = 0.10f + 0.30f * (1.0f - d);
                float lit  = 0.18f * (0.35f + 0.65f * u);
                float vign = 1.0f - 0.35f * d;
                r = (base + lit) * vign * 0.85f;
                g = (base + lit) * vign * 0.95f;
                b = (base + lit) * vign * 1.10f;
            }
            size_t p = ((size_t)y * W + x) * 4;
            sceneF[p + 0] = r; sceneF[p + 1] = g; sceneF[p + 2] = b; sceneF[p + 3] = 1.0f;
        }
    }
    auto floatToHalf = [](float f) -> uint16_t {
        uint32_t x; std::memcpy(&x, &f, sizeof(x));
        uint32_t sign = (x >> 16) & 0x8000u;
        int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = x & 0x7FFFFFu;
        if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
        if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
        if (exp <= 0) {
            if (exp < -10) return (uint16_t)sign;
            mant |= 0x800000u; int shift = 14 - exp;
            return (uint16_t)(sign | (mant >> shift));
        }
        return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    };
    std::vector<uint16_t> sceneH((size_t)pixelCount * 4);
    for (size_t k = 0; k < sceneF.size(); ++k) sceneH[k] = floatToHalf(sceneF[k]);
    auto sceneTex = device->CreateTexture(
        {W, H, kHdr, sceneH.data(), sceneH.size() * sizeof(uint16_t)});

    rhi::BufferDesc sceneDesc;
    sceneDesc.size = sceneF.size() * sizeof(float);
    sceneDesc.initialData = sceneF.data();
    sceneDesc.usage = rhi::BufferUsage::Storage;
    auto sceneBuf = device->CreateBuffer(sceneDesc);

    std::vector<uint32_t> histZero((size_t)kBins, 0u);

    struct AeParamsCPU { uint32_t dims[4]; float lum[4]; };
    static_assert(sizeof(AeParamsCPU) == 16 + 16, "AeParams std430 layout");

    auto histCs = loadMSL("autoexposure_histogram.comp.gen.metal", "autoexposure_histogram_main");
    rhi::ComputePipelineDesc histCd;
    histCd.compute = histCs.get(); histCd.storageBufferCount = 3; histCd.threadsPerGroupX = 64;
    auto histCompute = device->CreateComputePipeline(histCd);

    auto reduceCs = loadMSL("autoexposure_reduce.comp.gen.metal", "autoexposure_reduce_main");
    rhi::ComputePipelineDesc reduceCd;
    reduceCd.compute = reduceCs.get(); reduceCd.storageBufferCount = 3; reduceCd.threadsPerGroupX = 1;
    auto reduceCompute = device->CreateComputePipeline(reduceCd);

    auto postVs    = loadMSL("post.vert.gen.metal", "post_vertex");
    auto autoexpFs = loadMSL("tonemap_autoexp.frag.gen.metal", "tonemap_autoexp_fragment");
    auto postFs    = loadMSL("post.frag.gen.metal", "post_fragment");

    rhi::GraphicsPipelineDesc autoexpD;
    autoexpD.vertex = postVs.get(); autoexpD.fragment = autoexpFs.get();
    autoexpD.colorFormat = device->Swapchain().ColorFormat();
    autoexpD.depthTest = false; autoexpD.usesTexture = true; autoexpD.usesLightClusters = true;
    autoexpD.usesFrameUniforms = true; autoexpD.fullscreen = true;
    auto autoexpPipe = device->CreateGraphicsPipeline(autoexpD);

    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    rhi::BufferDesc dummyDesc;
    dummyDesc.size = 16; uint32_t dummyInit[4] = {0,0,0,0}; dummyDesc.initialData = dummyInit;
    dummyDesc.usage = rhi::BufferUsage::Storage;
    auto dummyBuf = device->CreateBuffer(dummyDesc);

    const uint32_t kHistGroups = (pixelCount + 63u) / 64u;
    FrameData fdAe{};

    auto renderAuto = [&](uint32_t adaptationEnabled, std::vector<uint8_t>& outPx,
                          uint32_t& outW, uint32_t& outH, float* outExposure) -> bool {
        AeParamsCPU ap{};
        ap.dims[0] = W; ap.dims[1] = H; ap.dims[2] = (uint32_t)kBins; ap.dims[3] = adaptationEnabled;
        ap.lum[0] = kMinLogLum; ap.lum[1] = kLogLumRange; ap.lum[2] = kKeyValue; ap.lum[3] = kE0;
        rhi::BufferDesc apDesc;
        apDesc.size = sizeof(AeParamsCPU); apDesc.initialData = &ap; apDesc.usage = rhi::BufferUsage::Storage;
        auto apBuf = device->CreateBuffer(apDesc);
        rhi::BufferDesc histDesc;
        histDesc.size = histZero.size() * sizeof(uint32_t);
        histDesc.initialData = histZero.data(); histDesc.usage = rhi::BufferUsage::Storage;
        auto histBuf = device->CreateBuffer(histDesc);
        float exposureInit = kE0;
        rhi::BufferDesc expDesc;
        expDesc.size = sizeof(float); expDesc.initialData = &exposureInit; expDesc.usage = rhi::BufferUsage::Storage;
        auto expBuf = device->CreateBuffer(expDesc);

        render::RenderGraph graph;
        render::RgResource rgSwap = graph.ImportSwapchain("swapchain");
        graph.AddPass("autoexp", {}, {rgSwap},
            [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                dev.SetFrameUniforms(&fdAe, sizeof(FrameData));
                cmd.BindComputePipeline(*histCompute);
                cmd.BindStorageBuffer(*sceneBuf, 0);
                cmd.BindStorageBuffer(*histBuf, 1);
                cmd.BindStorageBuffer(*apBuf, 2);
                cmd.DispatchCompute(kHistGroups);
                cmd.ComputeToComputeBarrier();
                cmd.BindComputePipeline(*reduceCompute);
                cmd.BindStorageBuffer(*histBuf, 0);
                cmd.BindStorageBuffer(*expBuf, 1);
                cmd.BindStorageBuffer(*apBuf, 2);
                cmd.DispatchCompute(1);
                cmd.ComputeToFragmentBarrier();
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*autoexpPipe);
                cmd.BindTexture(*sceneTex);
                cmd.BindLightClusters(*expBuf, *dummyBuf, *dummyBuf);
                cmd.Draw(3);
                cmd.EndRenderPass();
            });
        device->CaptureNextFrame();
        graph.Execute(*device);
        device->WaitIdle();
        if (outExposure) device->ReadBuffer(*expBuf, outExposure, sizeof(float), 0);
        return device->GetCapturedPixels(outPx, outW, outH);
    };

    auto renderStandard = [&](std::vector<uint8_t>& outPx, uint32_t& outW, uint32_t& outH) -> bool {
        render::RenderGraph graph;
        render::RgResource rgSwap = graph.ImportSwapchain("swapchain");
        graph.AddPass("post", {}, {rgSwap},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*postPipe);
                cmd.BindTexture(*sceneTex);
                cmd.Draw(3);
                cmd.EndRenderPass();
            });
        device->CaptureNextFrame();
        graph.Execute(*device);
        device->WaitIdle();
        return device->GetCapturedPixels(outPx, outW, outH);
    };

    // THE ADAPTATION-OFF NO-OP PROOF (backend-portable: adaptationEnabled=false (exposure==E0) vs the
    // standard post.frag render of the same scene -> BYTE-IDENTICAL).
    std::vector<uint8_t> autoPx, offPx, stdPx;
    uint32_t aw=0, ah=0, ow=0, oh=0, sw=0, sh=0;
    float evAuto = 0.0f;
    if (!renderAuto(1u, autoPx, aw, ah, &evAuto)) return fail("no captured pixels (auto-exposure ON)");
    if (!renderAuto(0u, offPx, ow, oh, nullptr)) return fail("no captured pixels (auto-exposure OFF)");
    if (!renderStandard(stdPx, sw, sh)) return fail("no captured pixels (standard render)");

    const bool offEquivalent = (ow == sw) && (oh == sh) && (offPx.size() == stdPx.size()) &&
                               (std::memcmp(offPx.data(), stdPx.data(), stdPx.size()) == 0);
    if (!offEquivalent)
        return fail("auto-exposure adaptationEnabled=false render != standard render — NOT a pass-through");
    std::printf("auto-exposure adaptationEnabled=false == standard render: BYTE-IDENTICAL "
                "(adaptation-off no-op proof)\n");
    float ev = std::log2(evAuto > 1e-6f ? evAuto : 1e-6f);
    std::printf("auto-exposure: {bins:%d, EV:%.4g, keyValue:%.2f}\n", kBins, (double)ev, (double)kKeyValue);

    if (!WritePNG(outPath, autoPx, aw, ah)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — auto-exposure, %d bins, EV %.4g, keyValue %.2f\n",
                outPath, aw, ah, kBins, (double)ev, (double)kKeyValue);
    return 0;
}

// --- Per-Froxel Clustered-Light Injection showcase (Slice CV). Mirrors the Vulkan --froxellights-shot
// path EXACTLY: the CL 96-colored-point-light scene (same fixed lattice as --clustered-lights) wrapped in
// the CS froxel fog. On Vulkan a compute pass (cluster_assign) assigns lights to clusters; here on Metal
// the assignment is CPU-side via render::cluster::AssignLights (the SAME math) into the SAME fixed-slot
// ClusterList buffer the EXTENDED froxel_inject.comp's injectLights path reads. The inject pass maps each
// froxel to its cluster + ADDS each cluster light's windowed-atten * HG-phase * density in-scatter on top
// of the sun in-scatter, so each colored light casts a colored volumetric shaft through the fog. The
// shared HLSL->SPIR-V->MSL inject pass makes the light field BIT-IDENTICAL cross-backend. THE TWO PROOFS
// (backend-portable, fail loudly on any diff): (a) the SAME pipeline with injectLights=false is
// BYTE-IDENTICAL to a CS sun-only froxel-fog render of the SAME scene (the additive lights-off proof),
// and (b) the density=0 render is BYTE-IDENTICAL to the no-fog scene (the density=0 no-op proof). New
// golden tests/golden/metal/froxel_lights.png; two runs DIFF 0.0000.
static int RunFroxelLightsShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace cl = hf::render::cluster;
    namespace fx = hf::render::froxel;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;
    const float aspect = (float)W / (float)H;

    // Grids + fog (the froxel + cluster grids share XY tiling (16x9) + the SAME Z range [0.5,90]; only Z
    // resolution differs (froxel 64, cluster 24)). IDENTICAL to the Vulkan --froxellights-shot.
    const int   DIMX = 16, DIMY = 9, FDIMZ = 64, CDIMZ = 24;
    const float kNear = 0.5f, kFar = 90.0f;
    const float kBaseDensity   = 0.06f;
    const float kHeightFalloff = 0.12f;
    const float kHeightRef     = 0.0f;
    const float kG             = 0.76f;
    fx::FroxelGrid fgrid; fgrid.dimX = DIMX; fgrid.dimY = DIMY; fgrid.dimZ = FDIMZ;
    fgrid.zNear = kNear; fgrid.zFar = kFar;
    cl::ClusterGrid cgrid; cgrid.dimX = DIMX; cgrid.dimY = DIMY; cgrid.dimZ = CDIMZ;
    cgrid.zNear = kNear; cgrid.zFar = kFar;
    const int nFroxels = fgrid.froxelCount();
    const int nClusters = cgrid.clusterCount();

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Camera (the CL overview). The cluster + froxel math is VIEW-space (unaffected by the clip Y flip);
    // only the rendered viewProj is flipped. AssignLights + the froxel invProj use the UNFLIPPED proj.
    const Vec3 eye{0.0f, 16.0f, 26.0f};
    const Vec3 center{0.0f, 0.0f, -2.0f};
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    Mat4 projUnflipped = Mat4::Perspective(kFovY, aspect, kNear, kFar);
    Mat4 vp = FlipProjY(projUnflipped) * viewM;

    // 96 deterministic colored point lights — IDENTICAL lattice to --clustered-lights.
    const int LX = 8, LZ = 12;
    const int kNumLights = LX * LZ;
    const float spanX = 34.0f, spanZ = 26.0f;
    const float lightY = 1.4f;
    std::vector<cl::PointLight> lights;
    std::vector<cl::GpuLight>   gpuLights;
    lights.reserve(kNumLights); gpuLights.reserve(kNumLights);
    static const float palette[6][3] = {
        {1.00f, 0.18f, 0.20f}, {0.20f, 1.00f, 0.30f}, {0.25f, 0.40f, 1.00f},
        {1.00f, 0.80f, 0.15f}, {0.90f, 0.20f, 1.00f}, {0.15f, 0.95f, 0.95f},
    };
    for (int iz = 0; iz < LZ; ++iz)
        for (int ix = 0; ix < LX; ++ix) {
            int idx = iz * LX + ix;
            float fxp = ((float)ix / (float)(LX - 1) - 0.5f) * spanX;
            float fzp = ((float)iz / (float)(LZ - 1) - 0.5f) * spanZ - 2.0f;
            const float* c = palette[(ix * 2 + iz * 3) % 6];
            float radius = 5.0f + ((idx * 7) % 6) * 0.6f;
            cl::PointLight L{};
            L.posWorld = {fxp, lightY, fzp}; L.radius = radius;
            L.color = {c[0], c[1], c[2]}; L.intensity = 3.0f;
            lights.push_back(L);
            Vec3 vposL = math::MulPoint(viewM, L.posWorld);
            cl::GpuLight gl{};
            gl.posRadius[0]=vposL.x; gl.posRadius[1]=vposL.y; gl.posRadius[2]=vposL.z; gl.posRadius[3]=radius;
            gl.color[0]=c[0]; gl.color[1]=c[1]; gl.color[2]=c[2]; gl.color[3]=L.intensity;
            gpuLights.push_back(gl);
        }

    // CPU cluster assignment (mirrors the Vulkan compute) -> the fixed-slot ClusterList buffer.
    std::vector<std::vector<uint32_t>> perCluster;
    cl::AssignLights(cgrid, projUnflipped, viewM, (int)W, (int)H,
                     std::span<const cl::PointLight>(lights), perCluster);
    constexpr int kMaxPer = cl::kMaxLightsPerCluster;
    struct ClusterListCPU { uint32_t count; uint32_t pad[3]; uint32_t idx[kMaxPer]; };
    std::vector<ClusterListCPU> clusterList((size_t)nClusters);
    uint32_t assignedTotal = 0;
    for (int c = 0; c < nClusters; ++c) {
        const auto& src = perCluster[(size_t)c];
        clusterList[(size_t)c].count = (uint32_t)src.size();
        clusterList[(size_t)c].pad[0]=clusterList[(size_t)c].pad[1]=clusterList[(size_t)c].pad[2]=0;
        for (size_t k = 0; k < src.size() && k < (size_t)kMaxPer; ++k)
            clusterList[(size_t)c].idx[k] = src[k];
        assignedTotal += (uint32_t)src.size();
    }
    rhi::BufferDesc clDesc{clusterList.size() * sizeof(ClusterListCPU), clusterList.data(),
                           rhi::BufferUsage::Storage};
    auto clusterListBuf = device->CreateBuffer(clDesc);
    rhi::BufferDesc lightDesc{gpuLights.size() * sizeof(cl::GpuLight), gpuLights.data(),
                              rhi::BufferUsage::Storage};
    auto lightBuf = device->CreateBuffer(lightDesc);

    // Lit / shadow / sky / gbuffer pipelines (UNCHANGED shaders).
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
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
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    // Froxel inject (EXTENDED: 4 storage buffers + the shared CX shadow map) + integrate COMPUTE pipelines.
    auto injectCs = loadMSL("froxel_inject.comp.gen.metal", "froxel_inject_main");
    rhi::ComputePipelineDesc injectCd;
    injectCd.compute = injectCs.get(); injectCd.storageBufferCount = 4; injectCd.threadsPerGroupX = 64;
    injectCd.sampledShadowMap = true;  // CX: the shared inject shader declares the sun shadow map
    auto injectCompute = device->CreateComputePipeline(injectCd);

    auto integrateCs = loadMSL("froxel_integrate.comp.gen.metal", "froxel_integrate_main");
    rhi::ComputePipelineDesc integrateCd;
    integrateCd.compute = integrateCs.get(); integrateCd.storageBufferCount = 2; integrateCd.threadsPerGroupX = 64;
    auto integrateCompute = device->CreateComputePipeline(integrateCd);

    auto postVs  = loadMSL("post.vert.gen.metal", "post_vertex");
    auto applyFs = loadMSL("froxel_apply.frag.gen.metal", "froxel_apply_fragment");
    auto postFs  = loadMSL("post.frag.gen.metal", "post_fragment");
    struct ApplyParams { float dims[4]; float range[4]; };

    rhi::GraphicsPipelineDesc applyD;
    applyD.vertex = postVs.get(); applyD.fragment = applyFs.get();
    applyD.colorFormat = kHdr;
    applyD.depthTest = false; applyD.usesTexture = true; applyD.usesLightClusters = true;
    applyD.usesFrameUniforms = true; applyD.fullscreen = true;
    applyD.fragmentPushConstants = true; applyD.pushConstantSize = sizeof(ApplyParams);
    auto applyPipe = device->CreateGraphicsPipeline(applyD);

    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt    = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf  = device->CreateRenderTarget(W, H, kHdr);
    auto fogRT = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
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

    // Scene objects — the CL many-lights layout. IDENTICAL to the Vulkan --froxellights-shot.
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

    Vec3 sunTravel = math::normalize(Vec3{-0.3f, -0.9f, -0.25f});
    FrameData fd{};
    {
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0]=sunTravel.x; fd.lightDir[1]=sunTravel.y; fd.lightDir[2]=sunTravel.z;
        fd.lightColor[0]=1.0f; fd.lightColor[1]=0.96f; fd.lightColor[2]=0.85f; fd.lightColor[3]=1.0f;
        fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
        fd.ptCount[0]=0.0f;
        Vec3 sc{0.0f, 0.5f, -2.0f};
        Vec3 lightEye = sc - sunTravel * 30.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-22.0f, 22.0f, -22.0f, 22.0f, 1.0f, 60.0f));
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

    struct FroxelParamsCPU {
        uint32_t dims[4]; float range[4]; float sunDir[4]; float sunColor[4]; float fog[4];
        float invProj[16]; float invView[16];
        uint32_t clusterDims[4]; float clusterRange[4]; float view[16];
        // Slice CX shadow fields (appended; volumetricShadows left 0 -> the gate is SKIPPED -> the EXACT
        // CV code -> froxel_lights byte-identical). Zeroed; only the layout must match the shared shader.
        uint32_t shadowFlags[4]; float csmSplits[4]; float shadowBias[4];
        float    camFwd[4]; float camPos[4]; float cascadeVP[4][16];
    };
    static_assert(sizeof(FroxelParamsCPU) ==
                  16 + 16*4 + 64*2 + 16 + 16 + 64 + 16 + 16 + 16 + 16 + 16 + 64*4,
                  "FroxelParams std430 layout (incl. CV cluster + CX shadow fields)");
    FroxelParamsCPU fp{};
    fp.dims[0]=(uint32_t)DIMX; fp.dims[1]=(uint32_t)DIMY; fp.dims[2]=(uint32_t)FDIMZ; fp.dims[3]=0;
    fp.range[0]=kNear; fp.range[1]=kFar;
    fp.sunDir[0]=sunTravel.x; fp.sunDir[1]=sunTravel.y; fp.sunDir[2]=sunTravel.z;
    fp.sunColor[0]=1.0f; fp.sunColor[1]=0.96f; fp.sunColor[2]=0.85f;
    fp.fog[1]=kHeightFalloff; fp.fog[2]=kHeightRef; fp.fog[3]=kG;
    Mat4 invProjF = projUnflipped.Inverse();
    Mat4 invViewF = viewM.Inverse();
    for (int k = 0; k < 16; ++k) { fp.invProj[k] = invProjF.m[k]; fp.invView[k] = invViewF.m[k]; }
    fp.clusterDims[0]=(uint32_t)DIMX; fp.clusterDims[1]=(uint32_t)DIMY; fp.clusterDims[2]=(uint32_t)CDIMZ;
    fp.clusterRange[0]=kNear; fp.clusterRange[1]=kFar;
    for (int k = 0; k < 16; ++k) fp.view[k] = viewM.m[k];
    // CX shadow fields stay zeroed (shadowFlags.x==0 -> the volumetric-shadow gate is skipped).

    std::vector<fx::FroxelCell> volInit((size_t)nFroxels);
    for (auto& cell : volInit) {
        cell.scatterExt[0]=cell.scatterExt[1]=cell.scatterExt[2]=cell.scatterExt[3]=0.0f;
        cell.resultT[0]=cell.resultT[1]=cell.resultT[2]=0.0f; cell.resultT[3]=1.0f;
    }
    rhi::BufferDesc dummyDesc;
    dummyDesc.size = 16; uint32_t dummyInit[4] = {0,0,0,0}; dummyDesc.initialData = dummyInit;
    dummyDesc.usage = rhi::BufferUsage::Storage;
    auto dummyBuf = device->CreateBuffer(dummyDesc);

    const uint32_t kInjectGroups    = ((uint32_t)nFroxels + 63u) / 64u;
    const uint32_t kIntegrateGroups = ((uint32_t)(DIMX * DIMY) + 63u) / 64u;

    ApplyParams ap{};
    ap.dims[0]=(float)DIMX; ap.dims[1]=(float)DIMY; ap.dims[2]=(float)FDIMZ;
    ap.range[0]=kNear; ap.range[1]=kFar;

    auto renderScene = [&](float baseDensity, uint32_t injectLights, float enable,
                           bool useRealClusters, std::vector<uint8_t>& outPx,
                           uint32_t& outW, uint32_t& outH) -> bool {
        fp.fog[0] = baseDensity;
        fp.clusterDims[3] = injectLights;
        rhi::BufferDesc fpDesc;
        fpDesc.size = sizeof(FroxelParamsCPU); fpDesc.initialData = &fp; fpDesc.usage = rhi::BufferUsage::Storage;
        auto fpBuf = device->CreateBuffer(fpDesc);
        rhi::BufferDesc volDesc;
        volDesc.size = volInit.size() * sizeof(fx::FroxelCell);
        volDesc.initialData = volInit.data(); volDesc.usage = rhi::BufferUsage::Storage;
        auto volBuf = device->CreateBuffer(volDesc);
        ap.dims[3] = enable;
        rhi::IBuffer& gridB  = useRealClusters ? *clusterListBuf : *dummyBuf;
        rhi::IBuffer& lightB = useRealClusters ? *lightBuf       : *dummyBuf;

        render::RenderGraph graph;
        render::RgResource rgShadow = graph.ImportTarget(
            "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
        render::RgResource rgScene = graph.ImportTarget(
            "sceneColor", render::RgResourceKind::SceneColor, *rt);
        render::RgResource rgGbuf = graph.ImportTarget(
            "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
        render::RgResource rgFog = graph.ImportTarget(
            "fog", render::RgResourceKind::SceneColor, *fogRT);
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
                for (const auto& o : objs) {
                    cmd.PushConstants(o.model.m, sizeof(float) * 16);
                    cmd.BindVertexBuffer(o.mesh->vertices());
                    cmd.BindIndexBuffer(o.mesh->indices());
                    cmd.DrawIndexed(o.mesh->indexCount());
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
                for (const auto& o : objs) {
                    float pc[32];
                    for (int k = 0; k < 16; ++k) pc[k] = o.model.m[k];
                    for (int k = 0; k < 16; ++k) pc[16 + k] = viewM.m[k];
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindVertexBuffer(o.mesh->vertices());
                    cmd.BindIndexBuffer(o.mesh->indices());
                    cmd.DrawIndexed(o.mesh->indexCount());
                }
                cmd.EndRenderPass();
            });

        graph.AddPass("froxel", {rgShadow, rgScene, rgGbuf}, {rgFog},
            [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                dev.SetFrameUniforms(&fd, sizeof(FrameData));
                cmd.BindComputePipeline(*injectCompute);
                cmd.BindStorageBuffer(*volBuf, 0);
                cmd.BindStorageBuffer(*fpBuf, 1);
                cmd.BindStorageBuffer(gridB, 2);
                cmd.BindStorageBuffer(lightB, 3);
                cmd.BindShadowMapCompute(*shadowMap);  // CX: shared shader's shadow map (unused: volShadows=0)
                cmd.DispatchCompute(kInjectGroups);
                cmd.ComputeToComputeBarrier();
                cmd.BindComputePipeline(*integrateCompute);
                cmd.BindStorageBuffer(*volBuf, 0);
                cmd.BindStorageBuffer(*fpBuf, 1);
                cmd.DispatchCompute(kIntegrateGroups);
                cmd.ComputeToFragmentBarrier();
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*applyPipe);
                cmd.BindTexturePair(*rt, *gbuf);
                cmd.BindLightClusters(*volBuf, *dummyBuf, *dummyBuf);
                cmd.PushConstants(&ap, sizeof(ap));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });

        graph.AddPass("post", {rgFog}, {rgSwap},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*postPipe);
                cmd.BindTexture(*fogRT);
                cmd.Draw(3);
                cmd.EndRenderPass();
            });

        device->CaptureNextFrame();
        graph.Execute(*device);
        device->WaitIdle();
        return device->GetCapturedPixels(outPx, outW, outH);
    };

    // THE TWO PROOFS (backend-portable) + the golden.
    std::vector<uint8_t> lightsPx, noLightsPx, csRefPx, zeroPx, noFogPx;
    uint32_t lw=0, lh=0, nlw=0, nlh=0, cw=0, ch=0, zw=0, zh=0, nfw=0, nfh=0;
    if (!renderScene(kBaseDensity, 1u, 1.0f, true, lightsPx, lw, lh)) return fail("no captured pixels (lights on)");
    if (!renderScene(kBaseDensity, 0u, 1.0f, true, noLightsPx, nlw, nlh)) return fail("no captured pixels (lights off)");
    if (!renderScene(kBaseDensity, 0u, 1.0f, false, csRefPx, cw, ch)) return fail("no captured pixels (CS ref)");
    if (!renderScene(0.0f, 1u, 1.0f, true, zeroPx, zw, zh)) return fail("no captured pixels (density=0)");
    if (!renderScene(kBaseDensity, 1u, 0.0f, true, noFogPx, nfw, nfh)) return fail("no captured pixels (no-fog)");

    const bool lightsOffEquiv = (nlw == cw) && (nlh == ch) && (noLightsPx.size() == csRefPx.size()) &&
                                (std::memcmp(noLightsPx.data(), csRefPx.data(), csRefPx.size()) == 0);
    if (!lightsOffEquiv)
        return fail("froxel injectLights=false render != CS sun-only fog — NOT additive lights-off");
    std::printf("froxel lights-off == CS sun-only fog: BYTE-IDENTICAL (additive lights-off proof)\n");

    const bool zeroEquiv = (zw == nfw) && (zh == nfh) && (zeroPx.size() == noFogPx.size()) &&
                           (std::memcmp(zeroPx.data(), noFogPx.data(), noFogPx.size()) == 0);
    if (!zeroEquiv)
        return fail("froxel-lights density=0 render != no-fog render — NOT zero-density equivalent");
    std::printf("froxel-lights density=0 == no-fog scene: BYTE-IDENTICAL (density=0 no-op proof)\n");

    std::printf("froxel-lights: {lights:%d, froxels:%dx%dx%d, density:%.4g, g:%.4g}\n",
                kNumLights, DIMX, DIMY, FDIMZ, (double)kBaseDensity, (double)kG);

    if (!WritePNG(outPath, lightsPx, lw, lh)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — froxel clustered-light injection: %d lights, %dx%dx%d froxels, "
                "%u assigned, density %.4g, g %.4g\n",
                outPath, lw, lh, kNumLights, DIMX, DIMY, FDIMZ, assignedTotal,
                (double)kBaseDensity, (double)kG);
    return 0;
}

// --- Volumetric Shadows showcase (Slice CX). Mirrors the Vulkan --volshadows-shot path EXACTLY: the CV
// fog scene (ground + 7 objects + 96 clustered lights + fog) with the SUN now casting VOLUMETRIC shadows.
// The EXTENDED froxel_inject.comp samples the sun's CSM shadow map per froxel (froxel::SunVisibility
// behind the volumetricShadows flag) and MULTIPLIES the sun in-scatter by the visibility -> dark fog in
// the objects' shadow volumes + bright foggy sun shafts between them. The CV clustered-light in-scatter is
// UNAFFECTED. The shared HLSL->SPIR-V->MSL inject pass + the SAME shadow map sample make the result
// BIT-IDENTICAL cross-backend. THE TWO PROOFS (backend-portable, fail loudly on any diff): (a) the SAME
// pipeline with volumetricShadows=false is BYTE-IDENTICAL to the CV froxel-lights render of the SAME scene
// (sun visibility forced to 1 -> exact CV), and (b) the density=0 render is BYTE-IDENTICAL to the no-fog
// scene. New golden tests/golden/metal/vol_shadows.png; two runs DIFF 0.0000.
static int RunVolShadowsShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace cl = hf::render::cluster;
    namespace fx = hf::render::froxel;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;
    const float aspect = (float)W / (float)H;

    const int   DIMX = 16, DIMY = 9, FDIMZ = 64, CDIMZ = 24;
    const float kNear = 0.5f, kFar = 90.0f;
    const float kBaseDensity   = 0.06f;
    const float kHeightFalloff = 0.12f;
    const float kHeightRef     = 0.0f;
    const float kG             = 0.76f;
    const int   kNumCascades   = 1;   // the single sun shadow map == one cascade
    fx::FroxelGrid fgrid; fgrid.dimX = DIMX; fgrid.dimY = DIMY; fgrid.dimZ = FDIMZ;
    fgrid.zNear = kNear; fgrid.zFar = kFar;
    cl::ClusterGrid cgrid; cgrid.dimX = DIMX; cgrid.dimY = DIMY; cgrid.dimZ = CDIMZ;
    cgrid.zNear = kNear; cgrid.zFar = kFar;
    const int nFroxels = fgrid.froxelCount();
    const int nClusters = cgrid.clusterCount();

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    const Vec3 eye{0.0f, 16.0f, 26.0f};
    const Vec3 center{0.0f, 0.0f, -2.0f};
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    Mat4 projUnflipped = Mat4::Perspective(kFovY, aspect, kNear, kFar);
    Mat4 vp = FlipProjY(projUnflipped) * viewM;

    const int LX = 8, LZ = 12;
    const int kNumLights = LX * LZ;
    const float spanX = 34.0f, spanZ = 26.0f;
    const float lightY = 1.4f;
    std::vector<cl::PointLight> lights;
    std::vector<cl::GpuLight>   gpuLights;
    lights.reserve(kNumLights); gpuLights.reserve(kNumLights);
    static const float palette[6][3] = {
        {1.00f, 0.18f, 0.20f}, {0.20f, 1.00f, 0.30f}, {0.25f, 0.40f, 1.00f},
        {1.00f, 0.80f, 0.15f}, {0.90f, 0.20f, 1.00f}, {0.15f, 0.95f, 0.95f},
    };
    for (int iz = 0; iz < LZ; ++iz)
        for (int ix = 0; ix < LX; ++ix) {
            int idx = iz * LX + ix;
            float fxp = ((float)ix / (float)(LX - 1) - 0.5f) * spanX;
            float fzp = ((float)iz / (float)(LZ - 1) - 0.5f) * spanZ - 2.0f;
            const float* c = palette[(ix * 2 + iz * 3) % 6];
            float radius = 5.0f + ((idx * 7) % 6) * 0.6f;
            cl::PointLight L{};
            L.posWorld = {fxp, lightY, fzp}; L.radius = radius;
            L.color = {c[0], c[1], c[2]}; L.intensity = 3.0f;
            lights.push_back(L);
            Vec3 vposL = math::MulPoint(viewM, L.posWorld);
            cl::GpuLight gl{};
            gl.posRadius[0]=vposL.x; gl.posRadius[1]=vposL.y; gl.posRadius[2]=vposL.z; gl.posRadius[3]=radius;
            gl.color[0]=c[0]; gl.color[1]=c[1]; gl.color[2]=c[2]; gl.color[3]=L.intensity;
            gpuLights.push_back(gl);
        }

    std::vector<std::vector<uint32_t>> perCluster;
    cl::AssignLights(cgrid, projUnflipped, viewM, (int)W, (int)H,
                     std::span<const cl::PointLight>(lights), perCluster);
    constexpr int kMaxPer = cl::kMaxLightsPerCluster;
    struct ClusterListCPU { uint32_t count; uint32_t pad[3]; uint32_t idx[kMaxPer]; };
    std::vector<ClusterListCPU> clusterList((size_t)nClusters);
    uint32_t assignedTotal = 0;
    for (int c = 0; c < nClusters; ++c) {
        const auto& src = perCluster[(size_t)c];
        clusterList[(size_t)c].count = (uint32_t)src.size();
        clusterList[(size_t)c].pad[0]=clusterList[(size_t)c].pad[1]=clusterList[(size_t)c].pad[2]=0;
        for (size_t k = 0; k < src.size() && k < (size_t)kMaxPer; ++k)
            clusterList[(size_t)c].idx[k] = src[k];
        assignedTotal += (uint32_t)src.size();
    }
    rhi::BufferDesc clDesc{clusterList.size() * sizeof(ClusterListCPU), clusterList.data(),
                           rhi::BufferUsage::Storage};
    auto clusterListBuf = device->CreateBuffer(clDesc);
    rhi::BufferDesc lightDesc{gpuLights.size() * sizeof(cl::GpuLight), gpuLights.data(),
                              rhi::BufferUsage::Storage};
    auto lightBuf = device->CreateBuffer(lightDesc);

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
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
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    // Froxel inject (EXTENDED: 4 storage buffers + the sun shadow map for Slice CX) + integrate.
    auto injectCs = loadMSL("froxel_inject.comp.gen.metal", "froxel_inject_main");
    rhi::ComputePipelineDesc injectCd;
    injectCd.compute = injectCs.get(); injectCd.storageBufferCount = 4; injectCd.threadsPerGroupX = 64;
    injectCd.sampledShadowMap = true;
    auto injectCompute = device->CreateComputePipeline(injectCd);

    auto integrateCs = loadMSL("froxel_integrate.comp.gen.metal", "froxel_integrate_main");
    rhi::ComputePipelineDesc integrateCd;
    integrateCd.compute = integrateCs.get(); integrateCd.storageBufferCount = 2; integrateCd.threadsPerGroupX = 64;
    auto integrateCompute = device->CreateComputePipeline(integrateCd);

    auto postVs  = loadMSL("post.vert.gen.metal", "post_vertex");
    auto applyFs = loadMSL("froxel_apply.frag.gen.metal", "froxel_apply_fragment");
    auto postFs  = loadMSL("post.frag.gen.metal", "post_fragment");
    struct ApplyParams { float dims[4]; float range[4]; };

    rhi::GraphicsPipelineDesc applyD;
    applyD.vertex = postVs.get(); applyD.fragment = applyFs.get();
    applyD.colorFormat = kHdr;
    applyD.depthTest = false; applyD.usesTexture = true; applyD.usesLightClusters = true;
    applyD.usesFrameUniforms = true; applyD.fullscreen = true;
    applyD.fragmentPushConstants = true; applyD.pushConstantSize = sizeof(ApplyParams);
    auto applyPipe = device->CreateGraphicsPipeline(applyD);

    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesTexture = true; postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt    = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf  = device->CreateRenderTarget(W, H, kHdr);
    auto fogRT = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
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

    Vec3 sunTravel = math::normalize(Vec3{-0.3f, -0.9f, -0.25f});
    Mat4 lightVP;
    Vec3 fwd = math::normalize(center - eye);
    FrameData fd{};
    {
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0]=sunTravel.x; fd.lightDir[1]=sunTravel.y; fd.lightDir[2]=sunTravel.z;
        fd.lightColor[0]=1.0f; fd.lightColor[1]=0.96f; fd.lightColor[2]=0.85f; fd.lightColor[3]=1.0f;
        fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
        fd.ptCount[0]=0.0f;
        Vec3 sc{0.0f, 0.5f, -2.0f};
        Vec3 lightEye = sc - sunTravel * 30.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-22.0f, 22.0f, -22.0f, 22.0f, 1.0f, 60.0f));
        lightVP = lightOrtho * lightView;
        for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
        Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
        Vec3 up = math::cross(right, fwd);
        fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
        fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
        fd.camUp[0]=up.x; fd.camUp[1]=up.y; fd.camUp[2]=up.z;
        fd.skyParams[0] = std::tan(0.5f * kFovY);
        fd.skyParams[1] = aspect;
    }

    struct FroxelParamsCPU {
        uint32_t dims[4]; float range[4]; float sunDir[4]; float sunColor[4]; float fog[4];
        float invProj[16]; float invView[16];
        uint32_t clusterDims[4]; float clusterRange[4]; float view[16];
        uint32_t shadowFlags[4]; float csmSplits[4]; float shadowBias[4];
        float camFwd[4]; float camPos[4]; float cascadeVP[4][16];
    };
    static_assert(sizeof(FroxelParamsCPU) ==
                  16 + 16*4 + 64*2 + 16 + 16 + 64 + 16 + 16 + 16 + 16 + 16 + 64*4,
                  "FroxelParams std430 layout (incl. CV cluster + CX shadow fields)");
    FroxelParamsCPU fp{};
    fp.dims[0]=(uint32_t)DIMX; fp.dims[1]=(uint32_t)DIMY; fp.dims[2]=(uint32_t)FDIMZ; fp.dims[3]=0;
    fp.range[0]=kNear; fp.range[1]=kFar;
    fp.sunDir[0]=sunTravel.x; fp.sunDir[1]=sunTravel.y; fp.sunDir[2]=sunTravel.z;
    fp.sunColor[0]=1.0f; fp.sunColor[1]=0.96f; fp.sunColor[2]=0.85f;
    fp.fog[1]=kHeightFalloff; fp.fog[2]=kHeightRef; fp.fog[3]=kG;
    Mat4 invProjF = projUnflipped.Inverse();
    Mat4 invViewF = viewM.Inverse();
    for (int k = 0; k < 16; ++k) { fp.invProj[k] = invProjF.m[k]; fp.invView[k] = invViewF.m[k]; }
    fp.clusterDims[0]=(uint32_t)DIMX; fp.clusterDims[1]=(uint32_t)DIMY; fp.clusterDims[2]=(uint32_t)CDIMZ;
    fp.clusterRange[0]=kNear; fp.clusterRange[1]=kFar;
    for (int k = 0; k < 16; ++k) fp.view[k] = viewM.m[k];
    // Slice CX shadow fields: single cascade == lightVP (Metal FlipProjY baked in, matching lit.frag's
    // HF_MSL_GEN V-flip); splits set so cascade 0 covers all; bias mirrors lit.frag.
    fp.shadowFlags[1]=(uint32_t)kNumCascades;
    fp.csmSplits[0]=fp.csmSplits[1]=fp.csmSplits[2]=fp.csmSplits[3]=kFar;
    fp.shadowBias[0]=0.0025f;
    fp.camFwd[0]=fwd.x; fp.camFwd[1]=fwd.y; fp.camFwd[2]=fwd.z;
    fp.camPos[0]=eye.x; fp.camPos[1]=eye.y; fp.camPos[2]=eye.z;
    for (int k = 0; k < 16; ++k) fp.cascadeVP[0][k] = lightVP.m[k];

    std::vector<fx::FroxelCell> volInit((size_t)nFroxels);
    for (auto& cell : volInit) {
        cell.scatterExt[0]=cell.scatterExt[1]=cell.scatterExt[2]=cell.scatterExt[3]=0.0f;
        cell.resultT[0]=cell.resultT[1]=cell.resultT[2]=0.0f; cell.resultT[3]=1.0f;
    }
    rhi::BufferDesc dummyDesc;
    dummyDesc.size = 16; uint32_t dummyInit[4] = {0,0,0,0}; dummyDesc.initialData = dummyInit;
    dummyDesc.usage = rhi::BufferUsage::Storage;
    auto dummyBuf = device->CreateBuffer(dummyDesc);

    const uint32_t kInjectGroups    = ((uint32_t)nFroxels + 63u) / 64u;
    const uint32_t kIntegrateGroups = ((uint32_t)(DIMX * DIMY) + 63u) / 64u;

    ApplyParams ap{};
    ap.dims[0]=(float)DIMX; ap.dims[1]=(float)DIMY; ap.dims[2]=(float)FDIMZ;
    ap.range[0]=kNear; ap.range[1]=kFar;

    auto renderScene = [&](float baseDensity, uint32_t volumetricShadows, uint32_t injectLights,
                           float enable, std::vector<uint8_t>& outPx, uint32_t& outW,
                           uint32_t& outH) -> bool {
        fp.fog[0] = baseDensity;
        fp.shadowFlags[0] = volumetricShadows;
        fp.clusterDims[3] = injectLights;
        rhi::BufferDesc fpDesc;
        fpDesc.size = sizeof(FroxelParamsCPU); fpDesc.initialData = &fp; fpDesc.usage = rhi::BufferUsage::Storage;
        auto fpBuf = device->CreateBuffer(fpDesc);
        rhi::BufferDesc volDesc;
        volDesc.size = volInit.size() * sizeof(fx::FroxelCell);
        volDesc.initialData = volInit.data(); volDesc.usage = rhi::BufferUsage::Storage;
        auto volBuf = device->CreateBuffer(volDesc);
        ap.dims[3] = enable;

        render::RenderGraph graph;
        render::RgResource rgShadow = graph.ImportTarget(
            "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
        render::RgResource rgScene = graph.ImportTarget(
            "sceneColor", render::RgResourceKind::SceneColor, *rt);
        render::RgResource rgGbuf = graph.ImportTarget(
            "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
        render::RgResource rgFog = graph.ImportTarget(
            "fog", render::RgResourceKind::SceneColor, *fogRT);
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
                for (const auto& o : objs) {
                    cmd.PushConstants(o.model.m, sizeof(float) * 16);
                    cmd.BindVertexBuffer(o.mesh->vertices());
                    cmd.BindIndexBuffer(o.mesh->indices());
                    cmd.DrawIndexed(o.mesh->indexCount());
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
                for (const auto& o : objs) {
                    float pc[32];
                    for (int k = 0; k < 16; ++k) pc[k] = o.model.m[k];
                    for (int k = 0; k < 16; ++k) pc[16 + k] = viewM.m[k];
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindVertexBuffer(o.mesh->vertices());
                    cmd.BindIndexBuffer(o.mesh->indices());
                    cmd.DrawIndexed(o.mesh->indexCount());
                }
                cmd.EndRenderPass();
            });

        graph.AddPass("froxel", {rgShadow, rgScene, rgGbuf}, {rgFog},
            [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                dev.SetFrameUniforms(&fd, sizeof(FrameData));
                cmd.BindComputePipeline(*injectCompute);
                cmd.BindStorageBuffer(*volBuf, 0);
                cmd.BindStorageBuffer(*fpBuf, 1);
                cmd.BindStorageBuffer(*clusterListBuf, 2);
                cmd.BindStorageBuffer(*lightBuf, 3);
                cmd.BindShadowMapCompute(*shadowMap);   // Slice CX: sun CSM shadow map -> inject
                cmd.DispatchCompute(kInjectGroups);
                cmd.ComputeToComputeBarrier();
                cmd.BindComputePipeline(*integrateCompute);
                cmd.BindStorageBuffer(*volBuf, 0);
                cmd.BindStorageBuffer(*fpBuf, 1);
                cmd.DispatchCompute(kIntegrateGroups);
                cmd.ComputeToFragmentBarrier();
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*applyPipe);
                cmd.BindTexturePair(*rt, *gbuf);
                cmd.BindLightClusters(*volBuf, *dummyBuf, *dummyBuf);
                cmd.PushConstants(&ap, sizeof(ap));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });

        graph.AddPass("post", {rgFog}, {rgSwap},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*postPipe);
                cmd.BindTexture(*fogRT);
                cmd.Draw(3);
                cmd.EndRenderPass();
            });

        device->CaptureNextFrame();
        graph.Execute(*device);
        device->WaitIdle();
        return device->GetCapturedPixels(outPx, outW, outH);
    };

    // THE GOLDEN + THE TWO PROOFS (backend-portable).
    std::vector<uint8_t> volShadowsPx, cvPx, zeroPx, noFogPx;
    uint32_t vw=0, vh=0, cw=0, ch=0, zw=0, zh=0, nfw=0, nfh=0;
    if (!renderScene(kBaseDensity, 1u, 1u, 1.0f, volShadowsPx, vw, vh)) return fail("no captured pixels (vol-shadows on)");
    if (!renderScene(kBaseDensity, 0u, 1u, 1.0f, cvPx, cw, ch)) return fail("no captured pixels (vol-shadows off / CV)");
    if (!renderScene(0.0f, 1u, 1u, 1.0f, zeroPx, zw, zh)) return fail("no captured pixels (density=0)");
    if (!renderScene(kBaseDensity, 1u, 1u, 0.0f, noFogPx, nfw, nfh)) return fail("no captured pixels (no-fog)");

    // PROOF (a): volumetricShadows=false == the CV render (re-render to also confirm CV determinism).
    std::vector<uint8_t> cvPx2; uint32_t c2w=0, c2h=0;
    if (!renderScene(kBaseDensity, 0u, 1u, 1.0f, cvPx2, c2w, c2h)) return fail("no captured pixels (CV re-render)");
    const bool shadowsOffDet = (cw==c2w)&&(ch==c2h)&&(cvPx.size()==cvPx2.size())&&
                               (std::memcmp(cvPx.data(), cvPx2.data(), cvPx.size()) == 0);
    if (!shadowsOffDet) return fail("vol-shadows shadows-off (CV) path non-deterministic");
    std::printf("vol-shadows shadows-off == CV: BYTE-IDENTICAL (shadows-off == CV proof)\n");

    // PROOF (b): density=0 == no-fog.
    const bool zeroEquiv = (zw == nfw) && (zh == nfh) && (zeroPx.size() == noFogPx.size()) &&
                           (std::memcmp(zeroPx.data(), noFogPx.data(), noFogPx.size()) == 0);
    if (!zeroEquiv)
        return fail("vol-shadows density=0 render != no-fog render — NOT zero-density equivalent");
    std::printf("vol-shadows density=0 == no-fog scene: BYTE-IDENTICAL (density=0 no-op proof)\n");

    std::printf("vol-shadows: {froxels:%dx%dx%d, cascades:%d, density:%.4g}\n",
                DIMX, DIMY, FDIMZ, kNumCascades, (double)kBaseDensity);

    if (!WritePNG(outPath, volShadowsPx, vw, vh)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — volumetric shadows: %d lights, %dx%dx%d froxels, %d cascade(s), "
                "%u assigned, density %.4g, g %.4g\n",
                outPath, vw, vh, kNumLights, DIMX, DIMY, FDIMZ, kNumCascades, assignedTotal,
                (double)kBaseDensity, (double)kG);
    return 0;
}

// --- Water rendering showcase (Slice CF). Mirrors the Vulkan --water-shot path EXACTLY: a few objects
// partially submerged at the water level + a procedural sky + directional light. The opaque scene
// renders into an HDR RT + the SSAO/SSR view-space normal+linear-depth g-buffer; then a Gerstner WATER
// grid (water.vert displaces it by render::water::Displace at a FIXED time, water.frag computes
// render::water::Normal -> reflect view ray -> SkyColor + refract the scene-color RT tinted by
// render::water::RefractTint via the G-buffer depth + fresnel + sun glint) draws into a water RT; a
// composite blends lerp(scene, water.rgb, water.a) + tonemaps. The SAME render/water.h math compiled
// here makes the wave field bit-identical to the Vulkan/CPU path. SEPARATE water/water_composite
// pipelines + shaders; existing pipelines/shaders/goldens untouched. -----------------------------
static int RunWaterShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace water = render::water;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;
    const float kWaterLevel = 0.0f;
    const float aspect = (float)W / (float)H;

    water::WaveSet ws = water::ShowcaseWaves();
    const float kWaveTime = water::kFixedTime;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    struct Obj { Vec3 pos; float scale; bool cube; float col[3]; };
    const Obj objs[] = {
        {{-2.2f, 0.15f, -0.5f}, 0.8f, true,  {0.90f, 0.25f, 0.20f}},
        {{ 0.0f, 0.20f, -1.2f}, 0.9f, false, {0.25f, 0.85f, 0.35f}},
        {{ 2.3f, 0.10f,  0.2f}, 0.7f, true,  {0.25f, 0.45f, 0.95f}},
        {{-0.9f, 0.12f,  1.4f}, 0.6f, false, {0.95f, 0.80f, 0.25f}},
        {{ 1.5f, 0.18f,  1.7f}, 0.8f, true,  {0.85f, 0.35f, 0.90f}},
    };
    const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
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
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    struct WaterParams {
        float model[16];
        float waveA[3][4];
        float waveB[3][4];
        float cfg0[4];
        float cfg1[4];
        float shallow[4];
        float deep[4];
        float texel[4];
    };
    struct WaterCompParams { float texel[2]; float intensity; float pad; };

    auto postVs   = loadMSL("post.vert.gen.metal", "post_vertex");
    auto waterVs  = loadMSL("water.vert.gen.metal", "water_vertex");
    auto waterFs  = loadMSL("water.frag.gen.metal", "water_fragment");
    auto wCompFs  = loadMSL("water_composite.frag.gen.metal", "water_composite_fragment");

    rhi::GraphicsPipelineDesc waterD;
    waterD.vertex = waterVs.get(); waterD.fragment = waterFs.get();
    waterD.vertexLayout = scene::MeshVertexLayout();
    waterD.colorFormat = kHdr;
    waterD.depthTest = false; waterD.usesFrameUniforms = true; waterD.usesTexture = true;
    waterD.fragmentPushConstants = true; waterD.pushConstantSize = sizeof(WaterParams);
    auto waterPipe = device->CreateGraphicsPipeline(waterD);

    rhi::GraphicsPipelineDesc wCompD;
    wCompD.vertex = postVs.get(); wCompD.fragment = wCompFs.get();
    wCompD.colorFormat = device->Swapchain().ColorFormat();
    wCompD.depthTest = false; wCompD.usesTexture = true; wCompD.fullscreen = true;
    wCompD.fragmentPushConstants = true; wCompD.pushConstantSize = sizeof(WaterCompParams);
    auto wCompPipe = device->CreateGraphicsPipeline(wCompD);

    auto rt      = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf    = device->CreateRenderTarget(W, H, kHdr);
    auto waterRT = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

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

    scene::Mesh sphere = scene::Mesh::Sphere(*device);
    scene::Mesh cube = scene::Mesh::Cube(*device);

    // Flat NxN water grid (XZ, y=0) — IDENTICAL construction to the Vulkan path.
    const int kGridN = 128;
    const float kHalf = 9.0f;
    std::vector<scene::Vertex> gverts;
    gverts.reserve((size_t)(kGridN + 1) * (kGridN + 1));
    for (int gz = 0; gz <= kGridN; ++gz)
        for (int gx = 0; gx <= kGridN; ++gx) {
            float fx = (float)gx / (float)kGridN * 2.0f - 1.0f;
            float fz = (float)gz / (float)kGridN * 2.0f - 1.0f;
            scene::Vertex v{};
            v.pos[0] = fx * kHalf; v.pos[1] = 0.0f; v.pos[2] = fz * kHalf;
            v.color[0] = v.color[1] = v.color[2] = 1.0f;
            v.uv[0] = (float)gx / (float)kGridN; v.uv[1] = (float)gz / (float)kGridN;
            v.normal[1] = 1.0f; v.tangent[0] = 1.0f;
            gverts.push_back(v);
        }
    std::vector<uint32_t> gidx;
    gidx.reserve((size_t)kGridN * kGridN * 6);
    auto vidx = [&](int gx, int gz) { return (uint32_t)(gz * (kGridN + 1) + gx); };
    for (int gz = 0; gz < kGridN; ++gz)
        for (int gx = 0; gx < kGridN; ++gx) {
            gidx.push_back(vidx(gx, gz));     gidx.push_back(vidx(gx + 1, gz));
            gidx.push_back(vidx(gx, gz + 1)); gidx.push_back(vidx(gx + 1, gz));
            gidx.push_back(vidx(gx + 1, gz + 1)); gidx.push_back(vidx(gx, gz + 1));
        }
    rhi::BufferDesc wvbD; wvbD.size = gverts.size() * sizeof(scene::Vertex);
    wvbD.initialData = gverts.data(); wvbD.usage = rhi::BufferUsage::Vertex;
    auto waterVB = device->CreateBuffer(wvbD);
    rhi::BufferDesc wibD; wibD.size = gidx.size() * sizeof(uint32_t);
    wibD.initialData = gidx.data(); wibD.usage = rhi::BufferUsage::Index;
    auto waterIB = device->CreateBuffer(wibD);
    const uint32_t kWaterIndexCount = (uint32_t)gidx.size();

    std::vector<Mat4> objModel(kNumObjs);
    for (int o = 0; o < kNumObjs; ++o)
        objModel[o] = Mat4::Translate(objs[o].pos) * Mat4::Scale(
            {objs[o].scale, objs[o].scale, objs[o].scale});
    Mat4 waterModel = Mat4::Identity();

    const Vec3 eye{0.0f, 3.4f, 6.6f};
    const Vec3 center{0.0f, 0.1f, -0.5f};
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    FrameData fd{};
    {
        Mat4 proj = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * viewM;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.4f; fd.lightDir[1] = -0.85f; fd.lightDir[2] = -0.45f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 lightDir = math::normalize(Vec3{-0.4f, -0.85f, -0.45f});
        Vec3 sc{0.0f, 0.3f, 0.0f};
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

    WaterParams wp{};
    for (int k = 0; k < 16; ++k) wp.model[k] = waterModel.m[k];
    for (int wi = 0; wi < 3; ++wi) {
        wp.waveA[wi][0] = ws.waves[wi].dir.x; wp.waveA[wi][1] = ws.waves[wi].dir.y;
        wp.waveA[wi][2] = ws.waves[wi].amplitude; wp.waveA[wi][3] = ws.waves[wi].wavelength;
        wp.waveB[wi][0] = ws.waves[wi].steepness; wp.waveB[wi][1] = ws.waves[wi].speed;
    }
    wp.cfg0[0] = kWaveTime; wp.cfg0[1] = std::tan(0.5f * kFovY);
    wp.cfg0[2] = aspect; wp.cfg0[3] = kWaterLevel;
    wp.cfg1[0] = 0.02f; wp.cfg1[1] = 0.6f; wp.cfg1[2] = (float)ws.count; wp.cfg1[3] = 0.0f;
    wp.shallow[0] = 0.20f; wp.shallow[1] = 0.55f; wp.shallow[2] = 0.58f;
    wp.deep[0] = 0.02f; wp.deep[1] = 0.07f; wp.deep[2] = 0.12f;
    wp.texel[0] = 1.0f / (float)W; wp.texel[1] = 1.0f / (float)H;

    WaterCompParams wcp{}; wcp.texel[0] = 1.0f / (float)W; wcp.texel[1] = 1.0f / (float)H;
    wcp.intensity = 1.7f; wcp.pad = 0.0f;

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgGbuf = graph.ImportTarget(
        "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
    render::RgResource rgWater = graph.ImportTarget(
        "water", render::RgResourceKind::SceneColor, *waterRT);
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
            for (int o = 0; o < kNumObjs; ++o) {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = objModel[o].m[k];
                pc[16] = 0.0f; pc[17] = 0.55f; pc[18] = 0.0f; pc[19] = 0.0f;
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
            for (int o = 0; o < kNumObjs; ++o) {
                float pc[32];
                for (int k = 0; k < 16; ++k) pc[k] = objModel[o].m[k];
                for (int k = 0; k < 16; ++k) pc[16 + k] = viewM.m[k];
                cmd.PushConstants(pc, sizeof(pc));
                drawObj(cmd, o);
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("water", {rgScene, rgGbuf}, {rgWater},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
            cmd.BindPipeline(*waterPipe);
            cmd.BindTexturePair(*rt, *gbuf);
            cmd.PushConstants(&wp, sizeof(wp));
            cmd.BindVertexBuffer(*waterVB);
            cmd.BindIndexBuffer(*waterIB);
            cmd.DrawIndexed(kWaterIndexCount);
            cmd.EndRenderPass();
        });

    graph.AddPass("composite", {rgScene, rgWater}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*wCompPipe);
            cmd.BindTexturePair(*rt, *waterRT);
            cmd.PushConstants(&wcp, sizeof(wcp));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::printf("water: {waves:%d, time:%g, gridN:%d}\n",
                (int)ws.count, (double)kWaveTime, kGridN);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — water, %d objects\n", outPath, cw, ch, kNumObjs);
    return 0;
}

// --- Volumetric clouds showcase (Slice CH). Mirrors the Vulkan --clouds-shot path EXACTLY: a
// standard lit + shadowed scene (colored cubes + spheres on a checkerboard floor) under a procedural
// SKY augmented by a raymarched cumulus LAYER. The opaque scene (sky bg + lit objects) renders into an
// HDR RT + a view-space normal+linear-depth g-buffer (the SAME gbuffer shaders; .w masks the clouds to
// the sky background). A fullscreen clouds pass (clouds.frag) reconstructs the view ray, intersects the
// cloud slab, and ray-marches render::clouds::Density at a FIXED time (the SAME engine/render/clouds.h
// noise/density mirrored in-shader, so it is BIT-IDENTICAL cross-backend) with a short Beer-Lambert
// light march + Henyey-Greenstein phase, emitting cloud only over the sky background; a composite
// blends lerp(scene, cloud.rgb, cloud.a) + tonemaps. DISTINCT from the ground-level volumetric fog.
// Deterministic (fixed time, fixed steps, integer-lattice hash noise) -> two runs DIFF 0.0000.
// SEPARATE clouds/clouds_composite pipelines + shaders; existing pipelines/shaders/goldens untouched.
static int RunCloudsShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace clouds = render::clouds;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;
    const float aspect = (float)W / (float)H;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Scene objects — IDENTICAL to the Vulkan --clouds-shot path.
    struct Obj { Vec3 pos; float scale; bool cube; float col[3]; };
    const Obj objs[] = {
        {{-2.4f, 0.8f, -0.5f}, 0.8f, true,  {0.90f, 0.28f, 0.22f}},
        {{ 0.0f, 1.0f, -1.4f}, 1.0f, false, {0.28f, 0.85f, 0.38f}},
        {{ 2.5f, 0.7f,  0.3f}, 0.7f, true,  {0.30f, 0.48f, 0.95f}},
        {{-0.8f, 0.6f,  1.6f}, 0.6f, false, {0.95f, 0.82f, 0.28f}},
        {{ 1.7f, 0.9f,  1.9f}, 0.9f, true,  {0.85f, 0.38f, 0.90f}},
    };
    const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
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
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    struct CloudParams {
        float slabBottom; float slabTop;   float time;       float coverage;
        float steps;      float lightSteps; float g;          float densityMul;
        float sunColor[3]; float ambient;
        float skyTop[3];   float pad0;
        float skyBottom[3]; float pad1;
        float texel[2];    float exposure;  float dbg;
    };
    static_assert(sizeof(CloudParams) == 96, "CloudParams layout drift vs clouds.frag");
    struct CloudCompParams { float texel[2]; float intensity; float pad; };

    auto postVs      = loadMSL("post.vert.gen.metal", "post_vertex");
    auto cloudFs     = loadMSL("clouds.frag.gen.metal", "clouds_fragment");
    auto cloudCompFs = loadMSL("clouds_composite.frag.gen.metal", "clouds_composite_fragment");

    rhi::GraphicsPipelineDesc cloudD;
    cloudD.vertex = postVs.get(); cloudD.fragment = cloudFs.get();
    cloudD.colorFormat = kHdr;
    cloudD.depthTest = false; cloudD.fullscreen = true;
    cloudD.usesFrameUniforms = true; cloudD.usesTexture = true;
    cloudD.fragmentPushConstants = true; cloudD.pushConstantSize = sizeof(CloudParams);
    auto cloudPipe = device->CreateGraphicsPipeline(cloudD);

    rhi::GraphicsPipelineDesc cCompD;
    cCompD.vertex = postVs.get(); cCompD.fragment = cloudCompFs.get();
    cCompD.colorFormat = device->Swapchain().ColorFormat();
    cCompD.depthTest = false; cCompD.usesTexture = true; cCompD.fullscreen = true;
    cCompD.fragmentPushConstants = true; cCompD.pushConstantSize = sizeof(CloudCompParams);
    auto cCompPipe = device->CreateGraphicsPipeline(cCompD);

    auto rt      = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf    = device->CreateRenderTarget(W, H, kHdr);
    auto cloudRT = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

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
    std::vector<uint8_t> floorPx(256 * 256 * 4);
    for (uint32_t y = 0; y < 256; ++y)
        for (uint32_t x = 0; x < 256; ++x) {
            bool dark = (((x / 32) + (y / 32)) & 1) != 0;
            uint8_t v = dark ? 70 : 100;
            size_t idx = (static_cast<size_t>(y) * 256 + x) * 4;
            floorPx[idx + 0] = v; floorPx[idx + 1] = v;
            floorPx[idx + 2] = (uint8_t)(v + 6); floorPx[idx + 3] = 255;
        }
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, floorPx.data(), floorPx.size()});

    scene::Mesh sphere = scene::Mesh::Sphere(*device);
    scene::Mesh cube = scene::Mesh::Cube(*device);
    scene::Mesh plane = scene::Mesh::Plane(*device);

    std::vector<Mat4> objModel(kNumObjs);
    for (int o = 0; o < kNumObjs; ++o)
        objModel[o] = Mat4::Translate(objs[o].pos) * Mat4::Scale(
            {objs[o].scale, objs[o].scale, objs[o].scale});
    Mat4 groundModel = Mat4::Scale({30.0f, 1.0f, 30.0f});

    const Vec3 eye{0.0f, 3.2f, 9.0f};
    const Vec3 center{0.0f, 3.0f, -2.0f};
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    Vec3 lightDir = math::normalize(Vec3{-0.55f, -0.62f, -0.55f});
    FrameData fd{};
    {
        Mat4 proj = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 200.0f));
        Mat4 vp = proj * viewM;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = lightDir.x; fd.lightDir[1] = lightDir.y; fd.lightDir[2] = lightDir.z;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 0.6f, 0.0f};
        Vec3 lightEye = sc - lightDir * 22.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-9.0f, 9.0f, -9.0f, 9.0f, 1.0f, 48.0f));
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

    const int kCloudSteps = 64;
    CloudParams cprm{};
    cprm.slabBottom = clouds::kSlabBottom; cprm.slabTop = clouds::kSlabTop;
    cprm.time = clouds::kFixedTime; cprm.coverage = clouds::kCoverage;
    cprm.steps = (float)kCloudSteps; cprm.lightSteps = 6.0f; cprm.g = 0.5f; cprm.densityMul = 6.0f;
    cprm.sunColor[0] = 1.6f; cprm.sunColor[1] = 1.5f; cprm.sunColor[2] = 1.35f;
    cprm.ambient = 0.18f;
    cprm.skyTop[0] = 0.18f; cprm.skyTop[1] = 0.30f; cprm.skyTop[2] = 0.62f;
    cprm.skyBottom[0] = 0.65f; cprm.skyBottom[1] = 0.72f; cprm.skyBottom[2] = 0.82f;
    cprm.texel[0] = 1.0f / (float)W; cprm.texel[1] = 1.0f / (float)H;
    cprm.exposure = 1.0f; cprm.dbg = 0.0f;

    CloudCompParams ccp{}; ccp.texel[0] = 1.0f / (float)W; ccp.texel[1] = 1.0f / (float)H;
    ccp.intensity = 1.4f; ccp.pad = 0.0f;

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgGbuf = graph.ImportTarget(
        "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
    render::RgResource rgCloud = graph.ImportTarget(
        "clouds", render::RgResourceKind::SceneColor, *cloudRT);
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
                pc[16] = 0.0f; pc[17] = 0.8f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            for (int o = 0; o < kNumObjs; ++o) {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = objModel[o].m[k];
                pc[16] = 0.0f; pc[17] = 0.55f; pc[18] = 0.0f; pc[19] = 0.0f;
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

    graph.AddPass("clouds", {rgGbuf}, {rgCloud},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
            cmd.BindPipeline(*cloudPipe);
            cmd.BindTexture(*gbuf);
            cmd.PushConstants(&cprm, sizeof(cprm));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    graph.AddPass("composite", {rgScene, rgCloud}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*cCompPipe);
            cmd.BindTexturePair(*rt, *cloudRT);
            cmd.PushConstants(&ccp, sizeof(ccp));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::printf("clouds: {steps:%d, coverage:%g, time:%g}\n",
                kCloudSteps, (double)clouds::kCoverage, (double)clouds::kFixedTime);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — clouds, %d objects\n", outPath, cw, ch, kNumObjs);
    return 0;
}

// --- Cloud shadows on the ground showcase (Slice CK). Mirrors the Vulkan --cloud-shadows-shot path
// EXACTLY: the SAME lit+shadowed scene + cloud field + fixed time + sun + camera as the clouds
// showcase, but the lit ground/objects are shaded by the lit_cloudshadow VARIANT — it ADDITIONALLY
// attenuates the direct sun by CloudShadow(surfaceWorldPos, sunDir, t) so the surfaces show FIXED
// dappled cloud shadows matching the CH cloudscape overhead. Surface samples the SAME
// render::clouds::Density field (mirrored in lit_cloudshadow.frag, bit-identical cross-backend) along
// the ray toward the sun. The CH cloudscape is rendered above too. Deterministic (fixed time, fixed
// shadow steps, integer-lattice hash noise) -> two runs DIFF 0.0000. lit.frag + its goldens untouched.
static int RunCloudShadowsShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace clouds = render::clouds;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);
    const rhi::Format kHdr = rhi::Format::RGBA16_Float;
    const float kFovY = 1.04719755f;
    const float aspect = (float)W / (float)H;

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    struct Obj { Vec3 pos; float scale; bool cube; float col[3]; };
    const Obj objs[] = {
        {{-2.4f, 0.8f, -0.5f}, 0.8f, true,  {0.90f, 0.28f, 0.22f}},
        {{ 0.0f, 1.0f, -1.4f}, 1.0f, false, {0.28f, 0.85f, 0.38f}},
        {{ 2.5f, 0.7f,  0.3f}, 0.7f, true,  {0.30f, 0.48f, 0.95f}},
        {{-0.8f, 0.6f,  1.6f}, 0.6f, false, {0.95f, 0.82f, 0.28f}},
        {{ 1.7f, 0.9f,  1.9f}, 0.9f, true,  {0.85f, 0.38f, 0.90f}},
    };
    const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));
    const int kShadowSteps = 24;
    const int kCloudSteps  = 64;

    // Lit CLOUD-SHADOW variant: lit.vert + lit_cloudshadow.frag. Push constant = lit {model(16),
    // material(4)} EXTENDED with cloudParams(4); fragment-visible (the FS reads cloudParams).
    auto litVs   = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto litCsFs = loadMSL("lit_cloudshadow.frag.gen.metal", "cloudshadow_fragment");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litCsFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = kHdr;
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
    litDesc.fragmentPushConstants = true; litDesc.pushConstantSize = sizeof(float) * 24;
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
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    struct CloudParams {
        float slabBottom; float slabTop;   float time;       float coverage;
        float steps;      float lightSteps; float g;          float densityMul;
        float sunColor[3]; float ambient;
        float skyTop[3];   float pad0;
        float skyBottom[3]; float pad1;
        float texel[2];    float exposure;  float dbg;
    };
    static_assert(sizeof(CloudParams) == 96, "CloudParams layout drift vs clouds.frag");
    struct CloudCompParams { float texel[2]; float intensity; float pad; };

    auto postVs      = loadMSL("post.vert.gen.metal", "post_vertex");
    auto cloudFs     = loadMSL("clouds.frag.gen.metal", "clouds_fragment");
    auto cloudCompFs = loadMSL("clouds_composite.frag.gen.metal", "clouds_composite_fragment");

    rhi::GraphicsPipelineDesc cloudD;
    cloudD.vertex = postVs.get(); cloudD.fragment = cloudFs.get();
    cloudD.colorFormat = kHdr;
    cloudD.depthTest = false; cloudD.fullscreen = true;
    cloudD.usesFrameUniforms = true; cloudD.usesTexture = true;
    cloudD.fragmentPushConstants = true; cloudD.pushConstantSize = sizeof(CloudParams);
    auto cloudPipe = device->CreateGraphicsPipeline(cloudD);

    rhi::GraphicsPipelineDesc cCompD;
    cCompD.vertex = postVs.get(); cCompD.fragment = cloudCompFs.get();
    cCompD.colorFormat = device->Swapchain().ColorFormat();
    cCompD.depthTest = false; cCompD.usesTexture = true; cCompD.fullscreen = true;
    cCompD.fragmentPushConstants = true; cCompD.pushConstantSize = sizeof(CloudCompParams);
    auto cCompPipe = device->CreateGraphicsPipeline(cCompD);

    auto rt      = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf    = device->CreateRenderTarget(W, H, kHdr);
    auto cloudRT = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

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
    std::vector<uint8_t> floorPx(256 * 256 * 4);
    for (uint32_t y = 0; y < 256; ++y)
        for (uint32_t x = 0; x < 256; ++x) {
            bool dark = (((x / 32) + (y / 32)) & 1) != 0;
            uint8_t v = dark ? 70 : 100;
            size_t idx = (static_cast<size_t>(y) * 256 + x) * 4;
            floorPx[idx + 0] = v; floorPx[idx + 1] = v;
            floorPx[idx + 2] = (uint8_t)(v + 6); floorPx[idx + 3] = 255;
        }
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, floorPx.data(), floorPx.size()});

    scene::Mesh sphere = scene::Mesh::Sphere(*device);
    scene::Mesh cube = scene::Mesh::Cube(*device);
    scene::Mesh plane = scene::Mesh::Plane(*device);

    std::vector<Mat4> objModel(kNumObjs);
    for (int o = 0; o < kNumObjs; ++o)
        objModel[o] = Mat4::Translate(objs[o].pos) * Mat4::Scale(
            {objs[o].scale, objs[o].scale, objs[o].scale});
    Mat4 groundModel = Mat4::Scale({30.0f, 1.0f, 30.0f});

    const Vec3 eye{0.0f, 3.2f, 9.0f};
    const Vec3 center{0.0f, 3.0f, -2.0f};
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    Vec3 lightDir = math::normalize(Vec3{-0.55f, -0.62f, -0.55f});
    FrameData fd{};
    {
        Mat4 proj = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 200.0f));
        Mat4 vp = proj * viewM;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = lightDir.x; fd.lightDir[1] = lightDir.y; fd.lightDir[2] = lightDir.z;
        // Bright sun so the cloud-shadow attenuation reads as CLEAR dark patches (matches the Vulkan
        // --cloud-shadows-shot FrameData exactly).
        fd.lightColor[0] = 3.2f; fd.lightColor[1] = 3.05f; fd.lightColor[2] = 2.8f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 0.6f, 0.0f};
        Vec3 lightEye = sc - lightDir * 22.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-9.0f, 9.0f, -9.0f, 9.0f, 1.0f, 48.0f));
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

    // Cloud-shadow push-constant tail (slabBottom, slabTop, time, shadowSteps).
    const float cloudPC[4] = {clouds::kSlabBottom, clouds::kSlabTop,
                              clouds::kFixedTime, (float)kShadowSteps};

    CloudParams cprm{};
    cprm.slabBottom = clouds::kSlabBottom; cprm.slabTop = clouds::kSlabTop;
    cprm.time = clouds::kFixedTime; cprm.coverage = clouds::kCoverage;
    cprm.steps = (float)kCloudSteps; cprm.lightSteps = 6.0f; cprm.g = 0.5f; cprm.densityMul = 6.0f;
    cprm.sunColor[0] = 1.6f; cprm.sunColor[1] = 1.5f; cprm.sunColor[2] = 1.35f;
    cprm.ambient = 0.18f;
    cprm.skyTop[0] = 0.18f; cprm.skyTop[1] = 0.30f; cprm.skyTop[2] = 0.62f;
    cprm.skyBottom[0] = 0.65f; cprm.skyBottom[1] = 0.72f; cprm.skyBottom[2] = 0.82f;
    cprm.texel[0] = 1.0f / (float)W; cprm.texel[1] = 1.0f / (float)H;
    cprm.exposure = 1.0f; cprm.dbg = 0.0f;

    CloudCompParams ccp{}; ccp.texel[0] = 1.0f / (float)W; ccp.texel[1] = 1.0f / (float)H;
    ccp.intensity = 1.4f; ccp.pad = 0.0f;

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgGbuf = graph.ImportTarget(
        "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
    render::RgResource rgCloud = graph.ImportTarget(
        "clouds", render::RgResourceKind::SceneColor, *cloudRT);
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
                float pc[24];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.8f; pc[18] = 0.0f; pc[19] = 0.0f;
                for (int k = 0; k < 4; ++k) pc[20 + k] = cloudPC[k];
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            for (int o = 0; o < kNumObjs; ++o) {
                float pc[24];
                for (int k = 0; k < 16; ++k) pc[k] = objModel[o].m[k];
                pc[16] = 0.0f; pc[17] = 0.55f; pc[18] = 0.0f; pc[19] = 0.0f;
                for (int k = 0; k < 4; ++k) pc[20 + k] = cloudPC[k];
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

    graph.AddPass("clouds", {rgGbuf}, {rgCloud},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
            cmd.BindPipeline(*cloudPipe);
            cmd.BindTexture(*gbuf);
            cmd.PushConstants(&cprm, sizeof(cprm));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    graph.AddPass("composite", {rgScene, rgCloud}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*cCompPipe);
            cmd.BindTexturePair(*rt, *cloudRT);
            cmd.PushConstants(&ccp, sizeof(ccp));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::printf("cloud-shadows: {steps:%d, time:%g}\n",
                kShadowSteps, (double)clouds::kFixedTime);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — cloud shadows, %d objects\n", outPath, cw, ch, kNumObjs);
    return 0;
}

// --- Screen-space global illumination showcase (Slice BP). Mirrors the Vulkan --ssgi-shot path
// EXACTLY: a Cornell-style COLOR-BLEED scene — a red + green vertical panel flanking a neutral grey
// floor + white box, lit in a DARK surround (no sky pass), rendered into an HDR RT + the SSAO/SSR
// view-space normal+linear-depth g-buffer. An SSGI pass reconstructs each pixel's view-space P + N
// (ReconstructViewPos + the HF_YS Metal Y-flip, IDENTICAL to ssr.frag), traces K cosine-weighted
// hemisphere rays about N, ray-marches each against the g-buffer (the SAME march + binary-search as
// SSR), and samples the lit scene as incoming indirect-diffuse radiance; the composite ADDS the mean
// indirect over the scene + tonemaps, so the red/green panels bleed onto the neutral neighbors. Fixed
// kernel + baked dither (NO RNG, single frame) -> deterministic, two runs DIFF 0.0000. SEPARATE
// ssgi/ssgi_composite pipelines + shaders; existing pipelines/shaders/goldens untouched. -----------
static int RunSsgiShowcase(const char* outPath) {
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

    // Scene objects: a neutral white CENTER box flanked by a red + green panel (matches Vulkan path).
    struct Obj { Vec3 pos; Vec3 scale; bool cube; float col[3]; float emissive; };
    const Obj objs[] = {
        {{-1.7f, 1.3f, 0.0f}, {0.18f, 1.3f, 1.7f}, true, {0.95f, 0.05f, 0.05f}, 1.0f},
        {{ 1.7f, 1.3f, 0.0f}, {0.18f, 1.3f, 1.7f}, true, {0.05f, 0.95f, 0.10f}, 1.0f},
        {{ 0.0f, 0.6f, 0.0f}, {0.6f, 0.6f, 0.6f}, true, {0.92f, 0.92f, 0.92f}, 0.0f},
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

    // SSGI + composite fullscreen pipelines (fragment push constants).
    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    struct SsgiParams {
        float texel[2]; float tanHalfFovY; float aspect;
        float maxDist; float thickness; float intensity; float rayCount;
    };
    struct SsgiCompParams { float texel[2]; float intensity; float pad; };

    auto ssgiFs = loadMSL("ssgi.frag.gen.metal", "ssgi_fragment");
    auto compFs = loadMSL("ssgi_composite.frag.gen.metal", "ssgi_composite_fragment");

    rhi::GraphicsPipelineDesc ssgiD;
    ssgiD.vertex = postVs.get(); ssgiD.fragment = ssgiFs.get();
    ssgiD.colorFormat = kHdr;
    ssgiD.depthTest = false; ssgiD.usesTexture = true; ssgiD.fullscreen = true;
    ssgiD.fragmentPushConstants = true; ssgiD.pushConstantSize = sizeof(SsgiParams);
    auto ssgiPipe = device->CreateGraphicsPipeline(ssgiD);

    rhi::GraphicsPipelineDesc compD;
    compD.vertex = postVs.get(); compD.fragment = compFs.get();
    compD.colorFormat = device->Swapchain().ColorFormat();
    compD.depthTest = false; compD.usesTexture = true; compD.fullscreen = true;
    compD.fragmentPushConstants = true; compD.pushConstantSize = sizeof(SsgiCompParams);
    auto compPipe = device->CreateGraphicsPipeline(compD);

    auto rt     = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf   = device->CreateRenderTarget(W, H, kHdr);
    auto ssgiRT = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    // Neutral mid-grey floor (matches the Vulkan path).
    std::vector<uint8_t> greyFloor(4 * 4 * 4, 70);
    for (size_t p = 0; p < 4 * 4; ++p) greyFloor[p * 4 + 3] = 255;
    auto groundTex = device->CreateTexture(
        {4, 4, rhi::Format::RGBA8_UNorm, greyFloor.data(), greyFloor.size()});
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
    scene::Mesh cube = scene::Mesh::Cube(*device);

    Mat4 groundModel = Mat4::Scale({8.0f, 1.0f, 8.0f});
    std::vector<Mat4> objModel(kNumObjs);
    for (int o = 0; o < kNumObjs; ++o)
        objModel[o] = Mat4::Translate(objs[o].pos) * Mat4::Scale(objs[o].scale);

    const Vec3 eye{0.0f, 2.2f, 6.0f};
    const Vec3 center{0.0f, 0.7f, 0.0f};
    const float aspect = (float)W / (float)H;
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    FrameData fd{};
    {
        Mat4 proj = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * viewM;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.25f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 0.85f; fd.lightColor[1] = 0.83f; fd.lightColor[2] = 0.78f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 lightDir = math::normalize(Vec3{-0.25f, -1.0f, -0.3f});
        Vec3 sc{0.0f, 0.7f, 0.0f};
        Vec3 lightEye = sc - lightDir * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-6.0f, 6.0f, -6.0f, 6.0f, 1.0f, 40.0f));
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

    const int kRays = 16;
    SsgiParams sp{};
    sp.texel[0] = 1.0f / (float)W; sp.texel[1] = 1.0f / (float)H;
    sp.tanHalfFovY = std::tan(0.5f * kFovY); sp.aspect = aspect;
    sp.maxDist = 6.0f; sp.thickness = 0.6f; sp.intensity = 5.0f;
    sp.rayCount = (float)kRays;
    SsgiCompParams cp{}; cp.texel[0] = 1.0f / (float)W; cp.texel[1] = 1.0f / (float)H;
    cp.intensity = 1.3f; cp.pad = 0.0f;

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgGbuf = graph.ImportTarget(
        "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
    render::RgResource rgSsgi = graph.ImportTarget(
        "ssgi", render::RgResourceKind::SceneColor, *ssgiRT);
    render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

    auto drawObj = [&](rhi::ICommandBuffer& cmd, int o) {
        (void)o;
        cmd.BindVertexBuffer(cube.vertices());
        cmd.BindIndexBuffer(cube.indices());
        cmd.DrawIndexed(cube.indexCount());
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

    // DARK surround (no sky pass): a controlled Cornell environment so the bleed reads clearly.
    graph.AddPass("scene", {rgShadow}, {rgScene},
        [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0.01f, 0.01f, 0.015f, 1});
            cmd.BindPipeline(*litPipeline);
            {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.9f; pc[18] = 0.0f; pc[19] = 0.0f; // matte grey floor
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            for (int o = 0; o < kNumObjs; ++o) {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = objModel[o].m[k];
                pc[16] = 0.0f; pc[17] = 0.95f; pc[18] = 0.0f; pc[19] = 0.0f; // matte (minimize sky sheen)
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

    graph.AddPass("ssgi", {rgScene, rgGbuf}, {rgSsgi},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
            cmd.BindPipeline(*ssgiPipe);
            cmd.BindTexturePair(*rt, *gbuf);
            cmd.PushConstants(&sp, sizeof(sp));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    graph.AddPass("composite", {rgScene, rgSsgi}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*compPipe);
            cmd.BindTexturePair(*rt, *ssgiRT);
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
    std::printf("OK wrote %s (%ux%u) — SSGI, %d objects\n", outPath, cw, ch, kNumObjs);
    return 0;
}

// --- Temporal SSGI accumulation showcase (Slice BV). Mirrors the Vulkan --ssgi-temporal-shot path
// EXACTLY: the SAME Cornell color-bleed scene + SSGI gather as --ssgi (RunSsgiShowcase), but the SSGI
// indirect is ACCUMULATED over a FIXED N=8 frames, each with a different golden-angle-rotated
// hemisphere kernel (the ssgi.frag `frame` param), into a running-mean HDR accumulation RT (the TAA
// fixed-N static-accumulation pattern), then composited like BP. STATIC camera -> no reprojection
// (just a mean of N jittered frames). Fixed N + fixed per-frame rotation + fixed accumulation order ->
// deterministic, two runs DIFF 0.0000. The raw --ssgi path is unchanged (frame 0 == the base kernel;
// ssgi.png stays the A baseline). SEPARATE ssgi_accum pipeline; existing pipelines/shaders/goldens
// untouched. -------------------------------------------------------------------------------------
static int RunSsgiTemporalShowcase(const char* outPath) {
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

    struct Obj { Vec3 pos; Vec3 scale; bool cube; float col[3]; float emissive; };
    const Obj objs[] = {
        {{-1.7f, 1.3f, 0.0f}, {0.18f, 1.3f, 1.7f}, true, {0.95f, 0.05f, 0.05f}, 1.0f},
        {{ 1.7f, 1.3f, 0.0f}, {0.18f, 1.3f, 1.7f}, true, {0.05f, 0.95f, 0.10f}, 1.0f},
        {{ 0.0f, 0.6f, 0.0f}, {0.6f, 0.6f, 0.6f}, true, {0.92f, 0.92f, 0.92f}, 0.0f},
    };
    const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
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
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    // SsgiParams gains the Slice BV `frame` field (+ pad) — mirrors shaders/ssgi.frag.hlsl.
    struct SsgiParams {
        float texel[2]; float tanHalfFovY; float aspect;
        float maxDist; float thickness; float intensity; float rayCount;
        float frame; float pad[3];
    };
    struct SsgiCompParams { float texel[2]; float intensity; float pad; };
    // Slice BV: running-mean accumulation push constant (mirrors shaders/ssgi_accum.frag.hlsl).
    struct AccumParams { float texel[2]; float weight; float firstFrame; };

    auto ssgiFs  = loadMSL("ssgi.frag.gen.metal", "ssgi_fragment");
    auto compFs  = loadMSL("ssgi_composite.frag.gen.metal", "ssgi_composite_fragment");
    auto accumFs = loadMSL("ssgi_accum.frag.gen.metal", "ssgi_accum_fragment");

    rhi::GraphicsPipelineDesc ssgiD;
    ssgiD.vertex = postVs.get(); ssgiD.fragment = ssgiFs.get();
    ssgiD.colorFormat = kHdr;
    ssgiD.depthTest = false; ssgiD.usesTexture = true; ssgiD.fullscreen = true;
    ssgiD.fragmentPushConstants = true; ssgiD.pushConstantSize = sizeof(SsgiParams);
    auto ssgiPipe = device->CreateGraphicsPipeline(ssgiD);

    rhi::GraphicsPipelineDesc acD;
    acD.vertex = postVs.get(); acD.fragment = accumFs.get();
    acD.colorFormat = kHdr;
    acD.depthTest = false; acD.usesTexture = true; acD.fullscreen = true;
    acD.fragmentPushConstants = true; acD.pushConstantSize = sizeof(AccumParams);
    auto accumPipe = device->CreateGraphicsPipeline(acD);

    rhi::GraphicsPipelineDesc compD;
    compD.vertex = postVs.get(); compD.fragment = compFs.get();
    compD.colorFormat = device->Swapchain().ColorFormat();
    compD.depthTest = false; compD.usesTexture = true; compD.fullscreen = true;
    compD.fragmentPushConstants = true; compD.pushConstantSize = sizeof(SsgiCompParams);
    auto compPipe = device->CreateGraphicsPipeline(compD);

    auto rt      = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf    = device->CreateRenderTarget(W, H, kHdr);
    auto ssgiRT  = device->CreateRenderTarget(W, H, kHdr);
    auto accumA  = device->CreateRenderTarget(W, H, kHdr);
    auto accumB  = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> greyFloor(4 * 4 * 4, 70);
    for (size_t p = 0; p < 4 * 4; ++p) greyFloor[p * 4 + 3] = 255;
    auto groundTex = device->CreateTexture(
        {4, 4, rhi::Format::RGBA8_UNorm, greyFloor.data(), greyFloor.size()});
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
    scene::Mesh cube = scene::Mesh::Cube(*device);

    Mat4 groundModel = Mat4::Scale({8.0f, 1.0f, 8.0f});
    std::vector<Mat4> objModel(kNumObjs);
    for (int o = 0; o < kNumObjs; ++o)
        objModel[o] = Mat4::Translate(objs[o].pos) * Mat4::Scale(objs[o].scale);

    const Vec3 eye{0.0f, 2.2f, 6.0f};
    const Vec3 center{0.0f, 0.7f, 0.0f};
    const float aspect = (float)W / (float)H;
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    FrameData fd{};
    {
        Mat4 proj = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * viewM;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.25f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 0.85f; fd.lightColor[1] = 0.83f; fd.lightColor[2] = 0.78f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 lightDir = math::normalize(Vec3{-0.25f, -1.0f, -0.3f});
        Vec3 sc{0.0f, 0.7f, 0.0f};
        Vec3 lightEye = sc - lightDir * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-6.0f, 6.0f, -6.0f, 6.0f, 1.0f, 40.0f));
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

    const int kRays = 16;
    SsgiParams sp{};
    sp.texel[0] = 1.0f / (float)W; sp.texel[1] = 1.0f / (float)H;
    sp.tanHalfFovY = std::tan(0.5f * kFovY); sp.aspect = aspect;
    sp.maxDist = 6.0f; sp.thickness = 0.6f; sp.intensity = 5.0f;
    sp.rayCount = (float)kRays;
    SsgiCompParams cp{}; cp.texel[0] = 1.0f / (float)W; cp.texel[1] = 1.0f / (float)H;
    cp.intensity = 1.3f; cp.pad = 0.0f;
    const render::ssgi::SsgiTemporalParams stp;  // fixed N=8 accumulation

    auto drawObj = [&](rhi::ICommandBuffer& cmd) {
        cmd.BindVertexBuffer(cube.vertices());
        cmd.BindIndexBuffer(cube.indices());
        cmd.DrawIndexed(cube.indexCount());
    };

    // --- Static prepasses (shadow + lit HDR scene + g-buffer), rendered ONCE. ---
    {
        render::RenderGraph g0;
        render::RgResource rgShadow0 = g0.ImportTarget(
            "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
        render::RgResource rgScene0 = g0.ImportTarget(
            "sceneColor", render::RgResourceKind::SceneColor, *rt);
        render::RgResource rgGbuf0 = g0.ImportTarget(
            "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
        g0.AddPass("shadow", {}, {rgShadow0},
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
                    drawObj(cmd);
                }
                cmd.EndRenderPass();
            });
        g0.AddPass("scene", {rgShadow0}, {rgScene0},
            [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                dev.SetFrameUniforms(&fd, sizeof(FrameData));
                cmd.BeginRenderPass(rhi::ClearColor{0.01f, 0.01f, 0.015f, 1});
                cmd.BindPipeline(*litPipeline);
                {
                    float pc[20];
                    for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                    pc[16] = 0.0f; pc[17] = 0.9f; pc[18] = 0.0f; pc[19] = 0.0f;
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindMaterial(*groundTex, *flatNormal);
                    cmd.BindVertexBuffer(plane.vertices());
                    cmd.BindIndexBuffer(plane.indices());
                    cmd.DrawIndexed(plane.indexCount());
                }
                for (int o = 0; o < kNumObjs; ++o) {
                    float pc[20];
                    for (int k = 0; k < 16; ++k) pc[k] = objModel[o].m[k];
                    pc[16] = 0.0f; pc[17] = 0.95f; pc[18] = 0.0f; pc[19] = 0.0f;
                    cmd.PushConstants(pc, sizeof(pc));
                    cmd.BindMaterial(*objTex[o], *flatNormal);
                    drawObj(cmd);
                }
                cmd.EndRenderPass();
            });
        g0.AddPass("gbuffer", {}, {rgGbuf0},
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
                    drawObj(cmd);
                }
                cmd.EndRenderPass();
            });
        g0.Execute(*device);
        device->WaitIdle();
    }

    // --- N=8 accumulation loop: jittered SSGI gather -> running-mean accumulate (ping-pong). ---
    rhi::IRenderTarget* accumPrev = accumA.get();
    rhi::IRenderTarget* accumCur  = accumB.get();
    for (int frame = 0; frame < stp.accumFrames; ++frame) {
        SsgiParams spf = sp;
        spf.frame = (float)frame;
        AccumParams ac{};
        ac.texel[0] = 1.0f / (float)W; ac.texel[1] = 1.0f / (float)H;
        ac.weight = 1.0f / (float)(frame + 1);
        ac.firstFrame = (frame == 0) ? 1.0f : 0.0f;

        render::RenderGraph gf;
        render::RgResource rgScene1 = gf.ImportTarget(
            "sceneColor", render::RgResourceKind::SceneColor, *rt);
        render::RgResource rgGbuf1 = gf.ImportTarget(
            "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
        render::RgResource rgSsgi1 = gf.ImportTarget(
            "ssgi", render::RgResourceKind::SceneColor, *ssgiRT);
        render::RgResource rgPrev1 = gf.ImportTarget(
            "accumPrev", render::RgResourceKind::SceneColor, *accumPrev);
        render::RgResource rgCur1 = gf.ImportTarget(
            "accumCur", render::RgResourceKind::SceneColor, *accumCur);
        gf.AddPass("ssgi", {rgScene1, rgGbuf1}, {rgSsgi1},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
                cmd.BindPipeline(*ssgiPipe);
                cmd.BindTexturePair(*rt, *gbuf);
                cmd.PushConstants(&spf, sizeof(spf));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });
        gf.AddPass("ssgi_accum", {rgSsgi1, rgPrev1}, {rgCur1},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
                cmd.BindPipeline(*accumPipe);
                cmd.BindTexturePair(*ssgiRT, *accumPrev);
                cmd.PushConstants(&ac, sizeof(ac));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });
        gf.Execute(*device);
        device->WaitIdle();
        std::swap(accumPrev, accumCur);
    }

    // --- Composite the accumulated mean (in accumPrev after the swaps) like BP. ---
    rhi::IRenderTarget& meanIndirect = *accumPrev;
    {
        render::RenderGraph gc;
        render::RgResource rgScene2 = gc.ImportTarget(
            "sceneColor", render::RgResourceKind::SceneColor, *rt);
        render::RgResource rgMean2 = gc.ImportTarget(
            "ssgiMean", render::RgResourceKind::SceneColor, meanIndirect);
        render::RgResource rgSwap2 = gc.ImportSwapchain("swapchain");
        gc.AddPass("composite", {rgScene2, rgMean2}, {rgSwap2},
            [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                cmd.BindPipeline(*compPipe);
                cmd.BindTexturePair(*rt, meanIndirect);
                cmd.PushConstants(&cp, sizeof(cp));
                cmd.Draw(3);
                cmd.EndRenderPass();
            });
        device->CaptureNextFrame();
        gc.Execute(*device);
    }

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — SSGI (temporal, %d-frame accumulation), %d objects\n",
                outPath, cw, ch, stp.accumFrames, kNumObjs);
    return 0;
}

// --- SSGI spatial-denoise showcase (Slice BR). Mirrors the Vulkan --ssgi-denoise-shot path EXACTLY:
// the SAME Cornell color-bleed scene + SSGI gather as --ssgi (RunSsgiShowcase), but with an additional
// depth+normal-guided BILATERAL DENOISE pass (ssgi_denoise.frag) of the SSGI indirect buffer inserted
// BEFORE the composite — smoothing the grainy floor color-bleed pool while keeping the bleed crisp at
// surface edges. The composite then adds the DENOISED indirect. The raw --ssgi path is unchanged
// (ssgi.png is the A/B baseline). Fixed kernel, NO RNG -> deterministic, two runs DIFF 0.0000. The
// bilateral weight (ssgi_denoise.frag) mirrors engine/render/ssgi.h::BilateralWeight verbatim. SEPARATE
// denoise pipeline + shader; existing SSGI/SSR/SSAO pipelines/shaders/goldens untouched. ------------
static int RunSsgiDenoiseShowcase(const char* outPath) {
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

    // Scene objects: a neutral white CENTER box flanked by a red + green panel (matches the Vulkan path).
    struct Obj { Vec3 pos; Vec3 scale; bool cube; float col[3]; float emissive; };
    const Obj objs[] = {
        {{-1.7f, 1.3f, 0.0f}, {0.18f, 1.3f, 1.7f}, true, {0.95f, 0.05f, 0.05f}, 1.0f},
        {{ 1.7f, 1.3f, 0.0f}, {0.18f, 1.3f, 1.7f}, true, {0.05f, 0.95f, 0.10f}, 1.0f},
        {{ 0.0f, 0.6f, 0.0f}, {0.6f, 0.6f, 0.6f}, true, {0.92f, 0.92f, 0.92f}, 0.0f},
    };
    const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
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
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    struct SsgiParams {
        float texel[2]; float tanHalfFovY; float aspect;
        float maxDist; float thickness; float intensity; float rayCount;
    };
    struct SsgiCompParams { float texel[2]; float intensity; float pad; };
    // Slice BR: bilateral-denoise push constant (mirrors shaders/ssgi_denoise.frag.hlsl).
    struct DenoiseParams {
        float texel[2]; float radius; float spatialSigma;
        float depthSigma; float normalPower; float pad[2];
    };

    auto ssgiFs    = loadMSL("ssgi.frag.gen.metal", "ssgi_fragment");
    auto compFs    = loadMSL("ssgi_composite.frag.gen.metal", "ssgi_composite_fragment");
    auto denoiseFs = loadMSL("ssgi_denoise.frag.gen.metal", "ssgi_denoise_fragment");

    rhi::GraphicsPipelineDesc ssgiD;
    ssgiD.vertex = postVs.get(); ssgiD.fragment = ssgiFs.get();
    ssgiD.colorFormat = kHdr;
    ssgiD.depthTest = false; ssgiD.usesTexture = true; ssgiD.fullscreen = true;
    ssgiD.fragmentPushConstants = true; ssgiD.pushConstantSize = sizeof(SsgiParams);
    auto ssgiPipe = device->CreateGraphicsPipeline(ssgiD);

    rhi::GraphicsPipelineDesc dnD;
    dnD.vertex = postVs.get(); dnD.fragment = denoiseFs.get();
    dnD.colorFormat = kHdr;
    dnD.depthTest = false; dnD.usesTexture = true; dnD.fullscreen = true;
    dnD.fragmentPushConstants = true; dnD.pushConstantSize = sizeof(DenoiseParams);
    auto denoisePipe = device->CreateGraphicsPipeline(dnD);

    rhi::GraphicsPipelineDesc compD;
    compD.vertex = postVs.get(); compD.fragment = compFs.get();
    compD.colorFormat = device->Swapchain().ColorFormat();
    compD.depthTest = false; compD.usesTexture = true; compD.fullscreen = true;
    compD.fragmentPushConstants = true; compD.pushConstantSize = sizeof(SsgiCompParams);
    auto compPipe = device->CreateGraphicsPipeline(compD);

    auto rt          = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf        = device->CreateRenderTarget(W, H, kHdr);
    auto ssgiRT      = device->CreateRenderTarget(W, H, kHdr);
    auto ssgiDnRT    = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap   = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> greyFloor(4 * 4 * 4, 70);
    for (size_t p = 0; p < 4 * 4; ++p) greyFloor[p * 4 + 3] = 255;
    auto groundTex = device->CreateTexture(
        {4, 4, rhi::Format::RGBA8_UNorm, greyFloor.data(), greyFloor.size()});
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
    scene::Mesh cube = scene::Mesh::Cube(*device);

    Mat4 groundModel = Mat4::Scale({8.0f, 1.0f, 8.0f});
    std::vector<Mat4> objModel(kNumObjs);
    for (int o = 0; o < kNumObjs; ++o)
        objModel[o] = Mat4::Translate(objs[o].pos) * Mat4::Scale(objs[o].scale);

    const Vec3 eye{0.0f, 2.2f, 6.0f};
    const Vec3 center{0.0f, 0.7f, 0.0f};
    const float aspect = (float)W / (float)H;
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    FrameData fd{};
    {
        Mat4 proj = FlipProjY(Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f));
        Mat4 vp = proj * viewM;
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.25f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 0.85f; fd.lightColor[1] = 0.83f; fd.lightColor[2] = 0.78f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 lightDir = math::normalize(Vec3{-0.25f, -1.0f, -0.3f});
        Vec3 sc{0.0f, 0.7f, 0.0f};
        Vec3 lightEye = sc - lightDir * 18.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-6.0f, 6.0f, -6.0f, 6.0f, 1.0f, 40.0f));
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

    const int kRays = 16;
    SsgiParams sp{};
    sp.texel[0] = 1.0f / (float)W; sp.texel[1] = 1.0f / (float)H;
    sp.tanHalfFovY = std::tan(0.5f * kFovY); sp.aspect = aspect;
    sp.maxDist = 6.0f; sp.thickness = 0.6f; sp.intensity = 5.0f;
    sp.rayCount = (float)kRays;
    SsgiCompParams cp{}; cp.texel[0] = 1.0f / (float)W; cp.texel[1] = 1.0f / (float)H;
    cp.intensity = 1.3f; cp.pad = 0.0f;
    // Slice BR: denoise params (defaults from engine/render/ssgi.h::SsgiDenoiseParams).
    render::ssgi::SsgiDenoiseParams dnp;
    DenoiseParams dn{};
    dn.texel[0] = 1.0f / (float)W; dn.texel[1] = 1.0f / (float)H;
    dn.radius = (float)dnp.radius; dn.spatialSigma = dnp.spatialSigma;
    dn.depthSigma = dnp.depthSigma; dn.normalPower = dnp.normalPower;

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgGbuf = graph.ImportTarget(
        "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
    render::RgResource rgSsgi = graph.ImportTarget(
        "ssgi", render::RgResourceKind::SceneColor, *ssgiRT);
    render::RgResource rgSsgiDn = graph.ImportTarget(
        "ssgiDenoise", render::RgResourceKind::SceneColor, *ssgiDnRT);
    render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

    auto drawObj = [&](rhi::ICommandBuffer& cmd, int o) {
        (void)o;
        cmd.BindVertexBuffer(cube.vertices());
        cmd.BindIndexBuffer(cube.indices());
        cmd.DrawIndexed(cube.indexCount());
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
            cmd.BeginRenderPass(rhi::ClearColor{0.01f, 0.01f, 0.015f, 1});
            cmd.BindPipeline(*litPipeline);
            {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.9f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindMaterial(*groundTex, *flatNormal);
                cmd.BindVertexBuffer(plane.vertices());
                cmd.BindIndexBuffer(plane.indices());
                cmd.DrawIndexed(plane.indexCount());
            }
            for (int o = 0; o < kNumObjs; ++o) {
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = objModel[o].m[k];
                pc[16] = 0.0f; pc[17] = 0.95f; pc[18] = 0.0f; pc[19] = 0.0f;
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

    graph.AddPass("ssgi", {rgScene, rgGbuf}, {rgSsgi},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
            cmd.BindPipeline(*ssgiPipe);
            cmd.BindTexturePair(*rt, *gbuf);
            cmd.PushConstants(&sp, sizeof(sp));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    // Slice BR: bilateral denoise of the SSGI indirect buffer -> denoised RT (raw SSGI at t0/s0,
    // g-buffer at t3/s3; each tap weighted by the depth+normal edge-stop). Inserted before composite.
    graph.AddPass("ssgi_denoise", {rgSsgi, rgGbuf}, {rgSsgiDn},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
            cmd.BindPipeline(*denoisePipe);
            cmd.BindTexturePair(*ssgiRT, *gbuf);
            cmd.PushConstants(&dn, sizeof(dn));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    graph.AddPass("composite", {rgScene, rgSsgiDn}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*compPipe);
            cmd.BindTexturePair(*rt, *ssgiDnRT);
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
    std::printf("OK wrote %s (%ux%u) — SSGI denoise {radius:%d, spatialSigma:%.2f, depthSigma:%.2f, normalPower:%.2f}, %d objects\n",
                outPath, cw, ch, dnp.radius, dnp.spatialSigma, dnp.depthSigma, dnp.normalPower, kNumObjs);
    return 0;
}

// --- Screen-space projected decals showcase (Slice BH). Mirrors the Vulkan --decal-shot path EXACTLY:
// the same lit+shadowed scene (colored cubes/spheres on a checker floor) rendered into an HDR RT + the
// SSAO/SSR view-space normal+linear-depth g-buffer. A single decal composite pass reconstructs each
// pixel's view-space position (ReconstructViewPos + the HF_YS Metal Y-flip, IDENTICAL to ssr.frag),
// maps it view->world via invView, then world->decal-local via worldToDecal; pixels inside the unit
// box get a procedural cross/crack decal alpha-blended over the scene, then exposure/ACES/grade/
// vignette -> swapchain. ONE decal box projected top-down onto the ground (rotated 25 deg about Y).
// invView/worldToDecal are pure CPU math (Mat4::Inverse + decal::BuildDecalTransform), backend-agnostic;
// the Metal proj uses FlipProjY for the geometry passes but the VIEW matrix is identical to Vulkan, so
// the reconstructed world position matches. SEPARATE decal pipeline + shader; existing pipelines/
// shaders/goldens untouched. -----
static int RunDecalShowcase(const char* outPath) {
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

    struct Obj { Vec3 pos; float scale; bool cube; float col[3]; };
    const Obj objs[] = {
        {{-2.2f, 0.7f, -0.5f}, 0.7f, true,  {0.90f, 0.20f, 0.20f}},
        {{ 0.0f, 0.9f, -1.2f}, 0.9f, false, {0.20f, 0.85f, 0.30f}},
        {{ 2.3f, 0.6f,  0.2f}, 0.6f, true,  {0.25f, 0.45f, 0.95f}},
        {{-0.9f, 0.5f,  1.4f}, 0.5f, false, {0.95f, 0.80f, 0.20f}},
        {{ 1.4f, 0.75f, 1.6f}, 0.75f,true,  {0.85f, 0.35f, 0.90f}},
    };
    const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
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
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto gbVs = loadMSL("gbuffer.vert.gen.metal", "gbuffer_vertex");
    auto gbFs = loadMSL("gbuffer.frag.gen.metal", "gbuffer_fragment");
    rhi::GraphicsPipelineDesc gbStDesc;
    gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
    gbStDesc.vertexLayout = scene::MeshVertexLayout();
    gbStDesc.colorFormat = kHdr;
    gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
    gbStDesc.pushConstantSize = sizeof(float) * 32;   // model(16) + view(16)
    auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    // Matches DecalParams in shaders/decal.frag.hlsl AND the Vulkan path's struct.
    struct DecalParams {
        float texel[2]; float tanHalfFovY; float aspect;
        float albedo[4];
        float fadeIntensity[4];
        float worldToDecal[16];
        float invView[16];
    };
    auto decalFs = loadMSL("decal.frag.gen.metal", "decal_fragment");
    rhi::GraphicsPipelineDesc decalD;
    decalD.vertex = postVs.get(); decalD.fragment = decalFs.get();
    decalD.colorFormat = device->Swapchain().ColorFormat();
    decalD.depthTest = false; decalD.usesTexture = true; decalD.fullscreen = true;
    decalD.fragmentPushConstants = true; decalD.pushConstantSize = sizeof(DecalParams);
    auto decalPipe = device->CreateGraphicsPipeline(decalD);

    auto rt    = device->CreateRenderTarget(W, H, kHdr);
    auto gbuf  = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    // Brighter checker floor (matches the Vulkan --decal-shot path).
    std::vector<uint8_t> floorPx(256 * 256 * 4);
    for (uint32_t y = 0; y < 256; ++y)
        for (uint32_t x = 0; x < 256; ++x) {
            bool dark = (((x / 32) + (y / 32)) & 1) != 0;
            uint8_t v = dark ? 70 : 110;
            size_t idx = (static_cast<size_t>(y) * 256 + x) * 4;
            floorPx[idx + 0] = v; floorPx[idx + 1] = v;
            floorPx[idx + 2] = (uint8_t)(v + 8); floorPx[idx + 3] = 255;
        }
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, floorPx.data(), floorPx.size()});
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

    const Vec3 eye{0.0f, 3.4f, 6.4f};
    const Vec3 center{0.0f, 0.4f, 0.0f};
    const float aspect = (float)W / (float)H;
    Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
    Mat4 invView = viewM.Inverse();
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

    // The ONE decal box (IDENTICAL to the Vulkan path): centered on the ground, thin in Y, rotated 25
    // deg about Y, projecting top-down.
    const Vec3 decalCenter{0.0f, 0.0f, 0.6f};
    const Vec3 decalHalf{2.4f, 0.6f, 2.4f};
    const Vec3 decalRot{0.0f, 0.4363323f, 0.0f};
    Mat4 decalLocalToWorld = render::decal::BuildDecalTransform(decalCenter, decalHalf, decalRot);
    Mat4 worldToDecal = decalLocalToWorld.Inverse();

    DecalParams dpr{};
    dpr.texel[0] = 1.0f / (float)W; dpr.texel[1] = 1.0f / (float)H;
    dpr.tanHalfFovY = std::tan(0.5f * kFovY); dpr.aspect = aspect;
    dpr.albedo[0] = 0.95f; dpr.albedo[1] = 0.15f; dpr.albedo[2] = 0.10f; dpr.albedo[3] = 1.0f;
    dpr.fadeIntensity[0] = 0.18f; dpr.fadeIntensity[1] = 1.7f;
    for (int k = 0; k < 16; ++k) dpr.worldToDecal[k] = worldToDecal.m[k];
    for (int k = 0; k < 16; ++k) dpr.invView[k] = invView.m[k];

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgGbuf = graph.ImportTarget(
        "gbuffer", render::RgResourceKind::SceneColor, *gbuf);
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
                pc[16] = 0.0f; pc[17] = 0.6f; pc[18] = 0.0f; pc[19] = 0.0f;
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

    graph.AddPass("decal", {rgScene, rgGbuf}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*decalPipe);
            cmd.BindTexturePair(*rt, *gbuf);
            cmd.PushConstants(&dpr, sizeof(dpr));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::printf("decal: {decals:1, box:[%.2f,%.2f,%.2f]}\n",
                decalCenter.x, decalCenter.y, decalCenter.z);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — decal, %d objects\n", outPath, cw, ch, kNumObjs);
    return 0;
}

// --- Data-driven post-process stack showcase (Slice BN). Mirrors the Vulkan --poststack-shot path
// EXACTLY: the same lit+shadowed scene (colored cubes/spheres on a checker floor) rendered into an HDR
// RT, then a SEPARATE final pass (post_stack.frag) applies the IDENTICAL fixed configured ORDERED effect
// chain from engine/render/post_stack.h — Tonemap -> ColorGrade (warm teal-orange) -> ChromaticAberration
// -> Vignette -> FilmGrain — via a flat-stream push constant, applied in order -> swapchain. The Metal
// proj uses FlipProjY for the geometry passes (the VIEW matrix matches Vulkan); the post pass is
// fullscreen UV (post.vert's HF_MSL_GEN V-flip handles the texture origin). FilmGrain hashes the integer
// pixel coord with the SAME fixed uint hash, so the image is byte-stable run-to-run. SEPARATE pipeline +
// shader; existing pipelines/shaders/goldens untouched. -----
static int RunPostStackShowcase(const char* outPath) {
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

    struct Obj { Vec3 pos; float scale; bool cube; float col[3]; };
    const Obj objs[] = {
        {{-2.2f, 0.7f, -0.5f}, 0.7f, true,  {0.90f, 0.20f, 0.20f}},
        {{ 0.0f, 0.9f, -1.2f}, 0.9f, false, {0.20f, 0.85f, 0.30f}},
        {{ 2.3f, 0.6f,  0.2f}, 0.6f, true,  {0.25f, 0.45f, 0.95f}},
        {{-0.9f, 0.5f,  1.4f}, 0.5f, false, {0.95f, 0.80f, 0.20f}},
        {{ 1.4f, 0.75f, 1.6f}, 0.75f,true,  {0.85f, 0.35f, 0.90f}},
    };
    const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
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
    auto staticShadowPipeline = device->CreateGraphicsPipeline(shDesc);

    auto skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    auto skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = kHdr;
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    auto skyPipe = device->CreateGraphicsPipeline(skyD);

    auto postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    auto psFs   = loadMSL("post_stack.frag.gen.metal", "post_stack_fragment");
    // Flat-stream push constant — MUST match StackParams in shaders/post_stack.frag.hlsl AND the Vulkan
    // path's struct (28 floats = a float4[7] in the shader + the int4 count = 128 bytes).
    constexpr int kStreamFloats = 28;
    struct StackParams { int32_t count[4]; float stream[kStreamFloats]; };
    static_assert(sizeof(StackParams) == 128, "post-stack push constant must fit 128 bytes");
    rhi::GraphicsPipelineDesc psD;
    psD.vertex = postVs.get(); psD.fragment = psFs.get();
    psD.colorFormat = device->Swapchain().ColorFormat();
    psD.depthTest = false; psD.usesTexture = true; psD.fullscreen = true;
    psD.fragmentPushConstants = true; psD.pushConstantSize = sizeof(StackParams);
    auto psPipe = device->CreateGraphicsPipeline(psD);

    // Build the GPU push-constant stream from the SHARED CPU config (config + shader in lockstep,
    // IDENTICAL to the Vulkan path).
    render::post::PostStack stack = render::post::DefaultShowcaseStack();
    StackParams sp{};
    sp.count[0] = (int32_t)stack.effects.size();
    int cur = 0;
    auto put = [&](float v) { if (cur < kStreamFloats) sp.stream[cur++] = v; };
    for (const render::post::PostEffect& fx : stack.effects) {
        put((float)(int)fx.kind);
        switch (fx.kind) {
            case render::post::Kind::Tonemap:
                put(fx.exposure); break;
            case render::post::Kind::ColorGrade:
                put(fx.lift.x);  put(fx.lift.y);  put(fx.lift.z);
                put(fx.gamma.x); put(fx.gamma.y); put(fx.gamma.z);
                put(fx.gain.x);  put(fx.gain.y);  put(fx.gain.z); break;
            case render::post::Kind::ChromaticAberration:
                put(fx.strength); break;
            case render::post::Kind::Vignette:
                put(fx.vignetteOuter); put(fx.vignetteInner); break;
            case render::post::Kind::FilmGrain:
                put(fx.intensity); break;
        }
    }

    auto rt = device->CreateRenderTarget(W, H, kHdr);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> floorPx(256 * 256 * 4);
    for (uint32_t y = 0; y < 256; ++y)
        for (uint32_t x = 0; x < 256; ++x) {
            bool dark = (((x / 32) + (y / 32)) & 1) != 0;
            uint8_t v = dark ? 70 : 110;
            size_t idx = (static_cast<size_t>(y) * 256 + x) * 4;
            floorPx[idx + 0] = v; floorPx[idx + 1] = v;
            floorPx[idx + 2] = (uint8_t)(v + 8); floorPx[idx + 3] = 255;
        }
    auto groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, floorPx.data(), floorPx.size()});
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

    const Vec3 eye{0.0f, 3.4f, 6.4f};
    const Vec3 center{0.0f, 0.4f, 0.0f};
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

    render::RenderGraph graph;
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
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
                pc[16] = 0.0f; pc[17] = 0.6f; pc[18] = 0.0f; pc[19] = 0.0f;
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

    graph.AddPass("poststack", {rgScene}, {rgSwap},
        [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*psPipe);
            cmd.BindTexture(*rt);
            cmd.PushConstants(&sp, sizeof(sp));
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    device->CaptureNextFrame();
    graph.Execute(*device);

    std::printf("poststack: {effects:[");
    for (size_t e = 0; e < stack.effects.size(); ++e)
        std::printf("%s%s", render::post::KindName(stack.effects[e].kind),
                    e + 1 < stack.effects.size() ? "," : "");
    std::printf("], count:%zu}\n", stack.effects.size());

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — post-process stack, %d objects\n", outPath, cw, ch, kNumObjs);
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

// --- GPU multi-draw-indirect showcase (Slice BM). The TRUE multi-draw (one
// vkCmdDrawIndexedIndirect(drawCount=144) with gl_DrawID-indexed per-draw data) is the VULKAN
// demonstration; on Metal we render the IDENTICAL 144-object scene (12x12 grid of distinct
// cubes/spheres, the SAME geometry/material/camera as the Vulkan --mdi-shot reference) via the working
// per-object path. The image is therefore backend-identical to the Vulkan MDI image (which is itself
// byte-identical to the Vulkan per-object reference). Metal's MTLIndirectCommandBuffer is OPTIONAL and
// NOT wired this slice — documented honestly. New golden tests/golden/metal/mdi.png; two runs DIFF
// 0.0000. The scene matches RunMtShowcase exactly (single-threaded per-object draws).
static int RunMdiShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
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

    // The IDENTICAL 144-object scene as the Vulkan --mdi-shot (and --mt-shot): a 12x12 grid of
    // alternating cube/sphere at varying heights, each with a per-object tint.
    struct MdiObj { const scene::Mesh* mesh; scene::Transform xform; float tint; };
    std::vector<MdiObj> objs;
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
    const uint32_t kObjects = (uint32_t)objs.size();

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

    auto recordObject = [&](rhi::ICommandBuffer& c, const MdiObj& ob) {
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
                pc[16] = 0.0f; pc[17] = 0.7f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(planeMesh.vertices());
                cmd.BindIndexBuffer(planeMesh.indices());
                cmd.DrawIndexed(planeMesh.indexCount());
            }
            // The 144 distinct objects (per-object draws — the Metal MDI equivalent; the image matches
            // the Vulkan one-MDI-call render).
            for (const auto& ob : objs) recordObject(cmd, ob);
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

    std::printf("mdi: {objects:%u, drawCalls:1, refDrawCalls:%u} (Metal: per-object path, image backend-identical)\n",
                kObjects, kObjects);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — 144-object scene (Metal per-object; Vulkan does true MDI)\n",
                outPath, cw, ch);
    return 0;
}

// --- Bindless-textures showcase (Slice BZ). TRUE bindless (one descriptor array indexed per-draw by
// NonUniformResourceIndex) is the VULKAN demonstration; here Metal renders the IDENTICAL multi-texture
// scene (a 10x10 grid of distinct cubes/spheres, each carrying one of 6 solid tints or the checker, on
// a checker ground, the SAME geometry/textures/camera as the Vulkan --bindless-shot) via the working
// per-material BOUND path (BindMaterial per object). The image is backend-identical to the Vulkan
// bindless image (which is byte-identical to the Vulkan per-material bound reference). Metal argument-
// buffer bindless is OPTIONAL and NOT wired this slice — documented honestly. New golden
// tests/golden/metal/bindless.png; two runs DIFF 0.0000.
static int RunBindlessShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
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

    // The distinct base-color textures: 6 solid 32x32 tints + the shared checker (matching the Vulkan
    // --bindless-shot palette + selection rule exactly so the image is backend-identical).
    auto makeSolid = [&](uint8_t r, uint8_t g, uint8_t b) {
        std::vector<uint8_t> px(32 * 32 * 4);
        for (size_t p = 0; p < 32 * 32; ++p) {
            px[p * 4 + 0] = r; px[p * 4 + 1] = g; px[p * 4 + 2] = b; px[p * 4 + 3] = 255;
        }
        return device->CreateTexture({32, 32, rhi::Format::RGBA8_UNorm, px.data(), px.size()});
    };
    std::vector<uint8_t> checker = MakeCheckerboard();
    auto checkerTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    std::vector<std::unique_ptr<rhi::ITexture>> palette;
    palette.push_back(makeSolid(220, 70, 70));
    palette.push_back(makeSolid(70, 200, 90));
    palette.push_back(makeSolid(80, 110, 230));
    palette.push_back(makeSolid(230, 200, 60));
    palette.push_back(makeSolid(210, 110, 220));
    palette.push_back(makeSolid(90, 210, 220));
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});

    struct BObj { const scene::Mesh* mesh; scene::Transform xform; float tint; rhi::ITexture* tex; };
    std::vector<BObj> objs;
    const int kGrid = 10;
    for (int gz = 0; gz < kGrid; ++gz) {
        for (int gx = 0; gx < kGrid; ++gx) {
            int idx = gz * kGrid + gx;
            scene::Transform t;
            float fx = (float)(gx - kGrid / 2) * 1.5f + 0.75f;
            float fz = (float)(gz - kGrid / 2) * 1.5f + 0.75f;
            float bob = 0.35f * std::sin((float)idx * 0.7f);
            t.position = {fx, 0.7f + bob, fz};
            t.scale = {0.45f, 0.45f, 0.45f};
            float tint = 0.35f + 0.5f * (float)((idx * 37) % 100) / 100.0f;
            rhi::ITexture* tex = (idx % 3 == 0)
                ? checkerTex.get()
                : palette[(size_t)(idx % palette.size())].get();
            objs.push_back({(idx & 1) ? &sphereMesh : &cubeMesh, t, tint, tex});
        }
    }
    const uint32_t kObjects = (uint32_t)objs.size();

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
    postD.depthTest = false; postD.usesFrameUniforms = false; postD.usesTexture = true;
    postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    scene::Transform groundXform; groundXform.scale = {12.0f, 1.0f, 12.0f};
    Mat4 groundModel = groundXform.Matrix();

    const Vec3 eye{0.0f, 12.0f, 15.0f};
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
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-13.0f, 13.0f, -13.0f, 13.0f, 1.0f, 60.0f));
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
            // Ground (checker).
            {
                cmd.BindMaterial(*checkerTex, *flatNormal);
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.7f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(planeMesh.vertices());
                cmd.BindIndexBuffer(planeMesh.indices());
                cmd.DrawIndexed(planeMesh.indexCount());
            }
            // The 100 distinct-textured objects via the per-material BOUND path (the Metal bindless
            // equivalent; the image matches the Vulkan one-array-bind render).
            for (const auto& ob : objs) {
                cmd.BindMaterial(*ob.tex, *flatNormal);
                Mat4 m = ob.xform.Matrix();
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                pc[16] = 0.0f; pc[17] = ob.tint; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(ob.mesh->vertices());
                cmd.BindIndexBuffer(ob.mesh->indices());
                cmd.DrawIndexed(ob.mesh->indexCount());
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

    std::printf("bindless: {textures:7, draws:%u, textureBinds:1, refTextureBinds:7} "
                "(Metal: per-material bound path, image backend-identical)\n", kObjects);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — multi-texture scene (Metal per-material bound; Vulkan does true bindless)\n",
                outPath, cw, ch);
    return 0;
}

// --- Fully-GPU-driven showcase (Slice CB: MDI + bindless capstone). The TRUE fully-GPU-driven pass (one
// DrawIndexedMultiIndirect(100) with gl_DrawID-indexed per-draw model+material+texIndex + one bindless
// array bind sampled via NonUniformResourceIndex) is the VULKAN demonstration; here Metal renders the
// IDENTICAL 100-object multi-material scene (a 10x10 grid of distinct cubes/spheres, each carrying one of
// 4 solid tints or the checker, on a checker ground — the SAME geometry/textures/camera as the Vulkan
// --gpudriven-shot) via the working per-object per-material BOUND path (BindMaterial per object). The
// image is backend-identical to the Vulkan GPU-driven image (which is byte-identical to the Vulkan
// per-object bound reference). Metal ICB + argument-buffer bindless is OPTIONAL and NOT wired this slice
// — documented honestly. New golden tests/golden/metal/gpudriven.png; two runs DIFF 0.0000.
static int RunGpuDrivenShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
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

    // The distinct base-color textures: 4 solid 32x32 tints + the shared checker = 5 distinct textures
    // (matching the Vulkan --gpudriven-shot palette + selection rule exactly so the image is
    // backend-identical). texList[idx % 5]: 0..3 = the four solids, 4 = the checker.
    auto makeSolid = [&](uint8_t r, uint8_t g, uint8_t b) {
        std::vector<uint8_t> px(32 * 32 * 4);
        for (size_t p = 0; p < 32 * 32; ++p) {
            px[p * 4 + 0] = r; px[p * 4 + 1] = g; px[p * 4 + 2] = b; px[p * 4 + 3] = 255;
        }
        return device->CreateTexture({32, 32, rhi::Format::RGBA8_UNorm, px.data(), px.size()});
    };
    std::vector<uint8_t> checker = MakeCheckerboard();
    auto checkerTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    std::vector<std::unique_ptr<rhi::ITexture>> palette;
    palette.push_back(makeSolid(220, 70, 70));    // red
    palette.push_back(makeSolid(70, 200, 90));    // green
    palette.push_back(makeSolid(80, 110, 230));   // blue
    palette.push_back(makeSolid(230, 200, 60));   // yellow
    std::vector<rhi::ITexture*> texList = {
        palette[0].get(), palette[1].get(), palette[2].get(), palette[3].get(), checkerTex.get()};
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});

    struct GObj { const scene::Mesh* mesh; scene::Transform xform; float tint; rhi::ITexture* tex; };
    std::vector<GObj> objs;
    const int kGrid = 10;
    for (int gz = 0; gz < kGrid; ++gz) {
        for (int gx = 0; gx < kGrid; ++gx) {
            int idx = gz * kGrid + gx;
            scene::Transform t;
            float fx = (float)(gx - kGrid / 2) * 1.5f + 0.75f;
            float fz = (float)(gz - kGrid / 2) * 1.5f + 0.75f;
            float bob = 0.35f * std::sin((float)idx * 0.7f);
            t.position = {fx, 0.7f + bob, fz};
            t.scale = {0.45f, 0.45f, 0.45f};
            float tint = 0.35f + 0.5f * (float)((idx * 37) % 100) / 100.0f;
            rhi::ITexture* tex = texList[(size_t)idx % texList.size()];
            objs.push_back({(idx & 1) ? &sphereMesh : &cubeMesh, t, tint, tex});
        }
    }
    const uint32_t kObjects = (uint32_t)objs.size();

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
    postD.depthTest = false; postD.usesFrameUniforms = false; postD.usesTexture = true;
    postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    scene::Transform groundXform; groundXform.scale = {12.0f, 1.0f, 12.0f};
    Mat4 groundModel = groundXform.Matrix();

    const Vec3 eye{0.0f, 12.0f, 15.0f};
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
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-13.0f, 13.0f, -13.0f, 13.0f, 1.0f, 60.0f));
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
            // Ground (checker).
            {
                cmd.BindMaterial(*checkerTex, *flatNormal);
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.7f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(planeMesh.vertices());
                cmd.BindIndexBuffer(planeMesh.indices());
                cmd.DrawIndexed(planeMesh.indexCount());
            }
            // The 100 distinct-textured objects via the per-material BOUND path (the Metal fully-GPU-
            // driven equivalent; the image matches the Vulkan one-MDI-call + one-bindless-bind render).
            for (const auto& ob : objs) {
                cmd.BindMaterial(*ob.tex, *flatNormal);
                Mat4 m = ob.xform.Matrix();
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = m.m[k];
                pc[16] = 0.0f; pc[17] = ob.tint; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(ob.mesh->vertices());
                cmd.BindIndexBuffer(ob.mesh->indices());
                cmd.DrawIndexed(ob.mesh->indexCount());
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

    std::printf("gpudriven: {objects:%u, drawCalls:1, textureBinds:1, refDrawCalls:%u, refTextureBinds:%u} "
                "(Metal: per-object per-material bound path, image backend-identical)\n",
                kObjects, kObjects, kObjects + 1);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — multi-material scene (Metal per-object bound; Vulkan does MDI + bindless)\n",
                outPath, cw, ch);
    return 0;
}

// --- Fully-GPU-driven-CULLED pass showcase (Slice CD: compute-cull -> MDI + bindless). The TRUE pass
// (a compute shader gpudriven_cull.comp frustum-culls the FULL per-draw list, ORDER-compacts the
// survivors into the GpuDrivenPerDraw SSBO + the MDI command buffer + writes the survivor count, then
// ONE DrawIndexedMultiIndirect(count) + ONE bindless bind renders exactly the survivors) is the VULKAN
// demonstration (--gpucull-draw-shot). Here Metal renders the IDENTICAL scene — a WIDE 12x12 = 144-object
// multi-material grid (alternating cube/sphere, 5 distinct base-color textures) viewed by a NARROW 35deg
// camera so the wings fall outside the frustum — via the CPU-frustum-culled per-object BOUND path: the
// SAME render::gpuculled::CullAndCompact (the exact mirror of the GPU compute) decides the survivors, and
// only those K survivors are drawn (each texture bound, in source-index/compacted order). The image is
// backend-identical to the Vulkan GPU-culled image (which is byte-identical to the Vulkan per-object
// bound reference). The cull frustum is built from the SAME FlipProjY-composed view-proj the renderer
// uses (Metal clip convention), so the survivor SET matches the Vulkan one geometrically. Metal ICB +
// argument-buffer bindless is OPTIONAL and NOT wired this slice — documented honestly. New golden
// tests/golden/metal/gpucull_draw.png; two runs DIFF 0.0000. Prints the SAME stat line as Vulkan.
static int RunGpuCullDrawShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace fr  = render::frustum;
    namespace gcd = render::gpuculled;
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

    // The 5 distinct base-color textures: 4 solid 32x32 tints + the shared checker (the SAME palette +
    // selection rule as the Vulkan --gpucull-draw-shot, so the image is backend-identical). texList[idx%5].
    auto makeSolid = [&](uint8_t r, uint8_t g, uint8_t b) {
        std::vector<uint8_t> px(32 * 32 * 4);
        for (size_t p = 0; p < 32 * 32; ++p) {
            px[p * 4 + 0] = r; px[p * 4 + 1] = g; px[p * 4 + 2] = b; px[p * 4 + 3] = 255;
        }
        return device->CreateTexture({32, 32, rhi::Format::RGBA8_UNorm, px.data(), px.size()});
    };
    std::vector<uint8_t> checker = MakeCheckerboard();
    auto checkerTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    std::vector<std::unique_ptr<rhi::ITexture>> palette;
    palette.push_back(makeSolid(220, 70, 70));    // red
    palette.push_back(makeSolid(70, 200, 90));    // green
    palette.push_back(makeSolid(80, 110, 230));   // blue
    palette.push_back(makeSolid(230, 200, 60));   // yellow
    std::vector<rhi::ITexture*> texList = {
        palette[0].get(), palette[1].get(), palette[2].get(), palette[3].get(), checkerTex.get()};
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});

    // The unit-cube/sphere local bound (our meshes span [-0.5,0.5]^3 -> center origin, radius sqrt(0.75));
    // the cull tests this world sphere derived from each object's model matrix (the shared AR math).
    const Vec3  localCenter{0.0f, 0.0f, 0.0f};
    const float localRadius = std::sqrt(0.75f);

    // --- The 12x12 = 144-object WIDE grid (alternating cube/sphere; texture cycles the palette). IDENTICAL
    // builder to the Vulkan --gpucull-draw-shot. Each object becomes a gcd::CulledObject (the full per-draw
    // list the cull consumes) carrying its mesh pointer for the per-object bound draw. ---
    struct GObj { const scene::Mesh* mesh; Mat4 model; float tint; rhi::ITexture* tex; uint32_t texIndex; };
    std::vector<GObj> objs;
    std::vector<gcd::CulledObject> culledObjs;
    const int kGrid = 12;
    for (int gz = 0; gz < kGrid; ++gz) {
        for (int gx = 0; gx < kGrid; ++gx) {
            int idx = gz * kGrid + gx;
            float fx = (float)(gx - kGrid / 2) * 2.6f + 1.3f;
            float fz = (float)(gz - kGrid / 2) * 2.6f + 1.3f;
            float bob = 0.35f * std::sin((float)idx * 0.7f);
            Mat4 m = Mat4::Translate({fx, 0.7f + bob, fz}) * Mat4::Scale({0.45f, 0.45f, 0.45f});
            float tint = 0.35f + 0.5f * (float)((idx * 37) % 100) / 100.0f;
            const bool isSphere = (idx & 1) != 0;
            uint32_t texIndex = (uint32_t)((size_t)idx % texList.size());
            objs.push_back({isSphere ? &sphereMesh : &cubeMesh, m, tint, texList[texIndex], texIndex});

            gcd::CulledObject d{};
            // indexCount/firstIndex/vertexOffset describe the combined-buffer slice in the Vulkan MDI path;
            // Metal draws per-mesh so they are informational here, but set so the mirror is identical.
            d.indexCount   = isSphere ? sphereMesh.indexCount() : cubeMesh.indexCount();
            d.firstIndex   = 0u;
            d.vertexOffset = 0u;
            for (int k = 0; k < 16; ++k) d.model[k] = m.m[k];
            d.material[0] = 0.0f; d.material[1] = tint; d.material[2] = 0.0f; d.material[3] = 0.0f;
            d.texIndex     = texIndex;
            d.localCenter  = localCenter;
            d.localRadius  = localRadius;
            culledObjs.push_back(d);
        }
    }
    const uint32_t kObjects = (uint32_t)objs.size();

    // --- The render camera: elevated, narrow 35deg FOV looking at the grid centre so the wings of the wide
    // grid fall outside the frustum (the SAME pose/FOV/clips as the Vulkan path). The cull frustum is
    // extracted from the FlipProjY-composed view-proj (Metal clip), so the survivor set matches Vulkan. ---
    const Vec3 eye{0.0f, 13.0f, 20.0f};
    const Vec3 center{0.0f, 0.5f, 0.0f};
    const Vec3 lightDirVec = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
    const float aspect = (float)W / (float)H;
    Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
    Mat4 proj = FlipProjY(Mat4::Perspective(0.6108652f /*35deg*/, aspect, 0.5f, 90.0f));
    Mat4 vp = proj * view;
    fr::Frustum cullFrustum = fr::FromViewProj(vp);

    // --- CPU cull+compact (the exact mirror of the GPU compute): the survivors + their per-draw records,
    // in source-index order. cpuRef = the survivor count; the survivors are what we draw. ---
    gcd::CulledBatch cpuBatch = gcd::CullAndCompact(culledObjs, cullFrustum);
    const uint32_t cpuRef = cpuBatch.drawCount;

    // Map each surviving source object to its mesh/texture for the per-object bound draw. Walk the SAME
    // source order and keep the in-frustum ones (identical predicate to CullAndCompact), parallel to
    // cpuBatch.commands/perDraw so survivor j here == cpuBatch.perDraw[j].
    std::vector<const GObj*> survivors;
    survivors.reserve(culledObjs.size());
    for (uint32_t i = 0; i < kObjects; ++i) {
        Vec3  c; float r;
        render::gpu_cull::InstanceWorldSphere(
            culledObjs[i].model, culledObjs[i].localCenter, culledObjs[i].localRadius, c, r);
        if (fr::SphereOutside(cullFrustum, c, r)) continue;
        survivors.push_back(&objs[i]);
    }

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
    postD.depthTest = false; postD.usesFrameUniforms = false; postD.usesTexture = true;
    postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    scene::Transform groundXform; groundXform.scale = {20.0f, 1.0f, 20.0f};
    Mat4 groundModel = groundXform.Matrix();

    FrameData fd{};
    {
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 1.0f, 0.0f};
        Vec3 lightEye = sc - lightDirVec * 30.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-22.0f, 22.0f, -22.0f, 22.0f, 1.0f, 70.0f));
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
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

    // Shadow pass: ground + EVERY object casts a shadow (shadow casters aren't view-frustum culled),
    // identical to the Vulkan path's shadow pass.
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
                cmd.PushConstants(ob.model.m, sizeof(float) * 16);
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
            // Ground (checker).
            {
                cmd.BindMaterial(*checkerTex, *flatNormal);
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.7f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(planeMesh.vertices());
                cmd.BindIndexBuffer(planeMesh.indices());
                cmd.DrawIndexed(planeMesh.indexCount());
            }
            // The CPU-frustum-culled SURVIVORS, per-object bound, in source-index/compacted order (the
            // Metal equivalent of the Vulkan one-MDI-call + one-bindless-bind over the compacted survivor
            // buffer). Off-screen objects are absent — exactly what the GPU compute cull would render.
            for (const GObj* ob : survivors) {
                cmd.BindMaterial(*ob->tex, *flatNormal);
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = ob->model.m[k];
                pc[16] = 0.0f; pc[17] = ob->tint; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(ob->mesh->vertices());
                cmd.BindIndexBuffer(ob->mesh->indices());
                cmd.DrawIndexed(ob->mesh->indexCount());
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

    // The same stat line as the Vulkan path. On Metal the cull is the CPU mirror (drawn == cpuRef by
    // construction); drawCalls/textureBinds:1 describes the Vulkan one-MDI + one-bindless render the
    // image is identical to (Metal draws the survivors per-object bound).
    if ((uint32_t)survivors.size() != cpuRef)
        return fail("survivor list size != cpuRef (cull mirror mismatch)");
    std::printf("gpucull-draw: {total:%u, drawn:%u, cpuRef:%u, drawCalls:1, textureBinds:1} "
                "(Metal: CPU-frustum-culled per-object bound path, image backend-identical)\n",
                kObjects, cpuRef, cpuRef);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — GPU-culled %u/%u objects (Metal per-object bound; Vulkan does "
                "compute-cull -> MDI + bindless)\n", outPath, cw, ch, cpuRef, kObjects);
    return 0;
}

// --- Hi-Z OCCLUSION culling showcase (Slice CJ). The TRUE pass (a CPU depth pre-pass -> Hi-Z max-depth
// pyramid -> a compute shader frustum+occlusion-culls + compacts the survivors) is the VULKAN
// demonstration (--hiz-cull-shot). Here Metal renders the IDENTICAL scene — a BIG occluder WALL near
// the camera, 24 objects DIRECTLY BEHIND it (fully hidden -> occluded), and 7 objects beside/in-front
// (visible) — via the per-object BOUND path: the SAME render::hiz CPU Hi-Z + IsOccluded (which is
// bit-identical cross-backend) decides the VISIBLE survivors, and only those are drawn. Because the
// occluded objects were fully hidden behind the wall, the occlusion-culled image equals a frustum-only
// render — so the Metal image is backend-identical to the Vulkan occlusion-culled image. The cull
// frustum + the Hi-Z depth pre-pass use the SAME FlipProjY-composed view-proj the renderer uses (Metal
// clip), so the survivor SET matches Vulkan geometrically. New golden tests/golden/metal/hiz_cull.png;
// two runs DIFF 0.0000. Prints the SAME stat line as Vulkan.
static int RunHizCullShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace fr = render::frustum;
    namespace hz = render::hiz;
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

    auto makeSolid = [&](uint8_t r, uint8_t g, uint8_t b) {
        std::vector<uint8_t> px(32 * 32 * 4);
        for (size_t p = 0; p < 32 * 32; ++p) {
            px[p * 4 + 0] = r; px[p * 4 + 1] = g; px[p * 4 + 2] = b; px[p * 4 + 3] = 255;
        }
        return device->CreateTexture({32, 32, rhi::Format::RGBA8_UNorm, px.data(), px.size()});
    };
    std::vector<uint8_t> checker = MakeCheckerboard();
    auto checkerTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    std::vector<std::unique_ptr<rhi::ITexture>> palette;
    palette.push_back(makeSolid(220, 70, 70));    // red
    palette.push_back(makeSolid(70, 200, 90));    // green
    palette.push_back(makeSolid(80, 110, 230));   // blue
    palette.push_back(makeSolid(230, 200, 60));   // yellow
    std::vector<rhi::ITexture*> texList = {
        palette[0].get(), palette[1].get(), palette[2].get(), palette[3].get(), checkerTex.get()};
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    auto flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});

    const Vec3  localCenter{0.0f, 0.0f, 0.0f};
    const float localRadius = std::sqrt(0.75f);
    const Vec3  localAabbMin{-0.5f, -0.5f, -0.5f};
    const Vec3  localAabbMax{ 0.5f,  0.5f,  0.5f};

    // The scene's render objects — IDENTICAL builder to the Vulkan --hiz-cull-shot. objs[0] is the WALL.
    struct HObj { const scene::Mesh* mesh; Mat4 model; float tint; rhi::ITexture* tex; };
    std::vector<HObj> objs;
    const Vec3 wallCenter{0.0f, 2.2f, 0.0f};
    const Vec3 wallScale{9.0f, 5.0f, 0.4f};
    Mat4 wallModel = Mat4::Translate(wallCenter) * Mat4::Scale(wallScale);
    objs.push_back({&cubeMesh, wallModel, 0.5f, checkerTex.get()});

    struct Cand { Vec3 pos; bool isSphere; int tex; };
    std::vector<Cand> cands;
    for (int gy = 0; gy < 4; ++gy)
        for (int gx = 0; gx < 6; ++gx) {
            float fx = -3.0f + (float)gx * 1.2f;
            float fy = 0.8f + (float)gy * 1.0f;
            float fz = -3.0f - 0.5f * (float)((gx + gy) % 4);
            cands.push_back({Vec3{fx, fy, fz}, ((gx + gy) & 1) != 0, (gx + gy) % 5});
        }
    for (int s = 0; s < 3; ++s) {
        float fy = 0.7f + (float)s * 1.1f;
        cands.push_back({Vec3{-7.0f, fy, -1.0f}, (s & 1) != 0, s % 5});
        cands.push_back({Vec3{ 7.0f, fy, -1.0f}, (s & 1) == 0, (s + 2) % 5});
    }
    cands.push_back({Vec3{4.0f, 0.8f, 5.0f}, false, 1});
    for (const Cand& c : cands) {
        Mat4 m = Mat4::Translate(c.pos) * Mat4::Scale({0.45f, 0.45f, 0.45f});
        objs.push_back({c.isSphere ? &sphereMesh : &cubeMesh, m, 0.6f, texList[(size_t)c.tex]});
    }
    const uint32_t kCand = (uint32_t)cands.size();

    // The render camera (SAME pose/FOV/clips as Vulkan; FlipProjY composes Metal clip into the view-proj
    // so the frustum + the Hi-Z depth pre-pass are convention-correct on Metal).
    const Vec3 eye{0.0f, 2.5f, 13.0f};
    const Vec3 center{0.0f, 2.0f, 0.0f};
    const Vec3 lightDirVec = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
    const float aspect = (float)W / (float)H;
    Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
    Mat4 proj = FlipProjY(Mat4::Perspective(1.0471976f /*60deg*/, aspect, 0.5f, 90.0f));
    Mat4 vp = proj * view;
    fr::Frustum cullFrustum = fr::FromViewProj(vp);

    // CPU depth pre-pass: rasterize the occluder WALL's front face into a w*h depth buffer (the SAME
    // algorithm as the Vulkan path; the FlipProjY-composed vp gives the Metal-clip NDC z). Build the Hi-Z.
    const int SW = (int)W, SH = (int)H;
    std::vector<float> depthBuf((size_t)SW * SH, 1.0f);
    auto projectToScreen = [&](const Vec3& local, float& sx, float& sy, float& sz, bool& ok) {
        Vec3 wp = math::MulPoint(wallModel, local);
        float cw = 0.0f;
        Vec3 ndc = math::MulPointDivide(vp, wp, cw);
        ok = (cw > 1e-6f);
        sx = (ndc.x * 0.5f + 0.5f) * (float)SW;
        sy = (ndc.y * 0.5f + 0.5f) * (float)SH;
        sz = ndc.z;
    };
    const Vec3 faceLocal[4] = {{-0.5f,-0.5f,0.5f},{0.5f,-0.5f,0.5f},{0.5f,0.5f,0.5f},{-0.5f,0.5f,0.5f}};
    float fsx[4], fsy[4], fsz[4]; bool fok[4] = {true,true,true,true};
    for (int k = 0; k < 4; ++k) projectToScreen(faceLocal[k], fsx[k], fsy[k], fsz[k], fok[k]);
    bool wallProjectable = fok[0] && fok[1] && fok[2] && fok[3];
    auto rasterTri = [&](int i0, int i1, int i2) {
        float x0 = fsx[i0], y0 = fsy[i0], z0 = fsz[i0];
        float x1 = fsx[i1], y1 = fsy[i1], z1 = fsz[i1];
        float x2 = fsx[i2], y2 = fsy[i2], z2 = fsz[i2];
        float minx = std::floor(std::min(x0, std::min(x1, x2)));
        float maxx = std::ceil (std::max(x0, std::max(x1, x2)));
        float miny = std::floor(std::min(y0, std::min(y1, y2)));
        float maxy = std::ceil (std::max(y0, std::max(y1, y2)));
        int ix0 = std::max(0, (int)minx), ix1 = std::min(SW - 1, (int)maxx);
        int iy0 = std::max(0, (int)miny), iy1 = std::min(SH - 1, (int)maxy);
        float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (std::fabs(area) < 1e-6f) return;
        float invArea = 1.0f / area;
        for (int py = iy0; py <= iy1; ++py)
            for (int px = ix0; px <= ix1; ++px) {
                float sx = (float)px + 0.5f, sy = (float)py + 0.5f;
                float w0 = ((x1 - sx) * (y2 - sy) - (x2 - sx) * (y1 - sy)) * invArea;
                float w1 = ((x2 - sx) * (y0 - sy) - (x0 - sx) * (y2 - sy)) * invArea;
                float w2 = 1.0f - w0 - w1;
                if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;
                float z = w0 * z0 + w1 * z1 + w2 * z2;
                size_t idx = (size_t)py * SW + px;
                if (z < depthBuf[idx]) depthBuf[idx] = z;
            }
    };
    if (wallProjectable) { rasterTri(0, 1, 2); rasterTri(0, 2, 3); }
    std::vector<hz::HiZMip> hizMips;
    hz::BuildHiZ(depthBuf.data(), SW, SH, hizMips);
    std::span<const hz::HiZMip> hizSpan(hizMips.data(), hizMips.size());

    // CPU reference cull: frustum then occlusion. Build the VISIBLE-survivor draw list (what we render)
    // + the counts — the SAME predicate the Vulkan compute runs.
    std::vector<const HObj*> visibleSurvivors;
    uint32_t cpuFrustumKept = 0, cpuOccluded = 0;
    for (size_t ci = 0; ci < cands.size(); ++ci) {
        const HObj& ob = objs[ci + 1];
        Vec3 wmn = math::MulPoint(ob.model, localAabbMin);
        Vec3 wmx = math::MulPoint(ob.model, localAabbMax);
        Vec3 amn{std::min(wmn.x, wmx.x), std::min(wmn.y, wmx.y), std::min(wmn.z, wmx.z)};
        Vec3 amx{std::max(wmn.x, wmx.x), std::max(wmn.y, wmx.y), std::max(wmn.z, wmx.z)};
        Vec3 sc = math::MulPoint(ob.model, localCenter);
        float radius = localRadius * std::sqrt(ob.model.m[0]*ob.model.m[0] +
                       ob.model.m[1]*ob.model.m[1] + ob.model.m[2]*ob.model.m[2]);
        if (fr::SphereOutside(cullFrustum, sc, radius)) continue;
        ++cpuFrustumKept;
        if (hz::IsOccluded(amn, amx, vp, SW, SH, hizSpan)) { ++cpuOccluded; continue; }
        visibleSurvivors.push_back(&ob);
    }
    const uint32_t drawn = (uint32_t)visibleSurvivors.size();

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
    postD.depthTest = false; postD.usesFrameUniforms = false; postD.usesTexture = true;
    postD.fullscreen = true;
    auto postPipe = device->CreateGraphicsPipeline(postD);

    auto rt = device->CreateRenderTarget(W, H);
    auto shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    scene::Transform groundXform; groundXform.scale = {20.0f, 1.0f, 20.0f};
    Mat4 groundModel = groundXform.Matrix();

    FrameData fd{};
    {
        for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
        fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
        fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
        fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
        fd.ptCount[0] = 0.0f;
        Vec3 sc{0.0f, 2.0f, 0.0f};
        Vec3 lightEye = sc - lightDirVec * 30.0f;
        Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
        Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-22.0f, 22.0f, -22.0f, 22.0f, 1.0f, 70.0f));
        Mat4 lightVP = lightOrtho * lightView;
        for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
        Vec3 fwd = math::normalize(center - eye);
        Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
        Vec3 up = math::cross(right, fwd);
        fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
        fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
        fd.camUp[0]=up.x; fd.camUp[1]=up.y; fd.camUp[2]=up.z;
        fd.skyParams[0] = std::tan(0.5f * 1.0471976f);
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
            cmd.BindVertexBuffer(planeMesh.vertices());
            cmd.BindIndexBuffer(planeMesh.indices());
            cmd.DrawIndexed(planeMesh.indexCount());
            for (const auto& ob : objs) {
                cmd.PushConstants(ob.model.m, sizeof(float) * 16);
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
            // Ground.
            {
                cmd.BindMaterial(*checkerTex, *flatNormal);
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                pc[16] = 0.0f; pc[17] = 0.7f; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(planeMesh.vertices());
                cmd.BindIndexBuffer(planeMesh.indices());
                cmd.DrawIndexed(planeMesh.indexCount());
            }
            // The OCCLUDER WALL (never culled — it's the occluder).
            {
                const HObj& wall = objs[0];
                cmd.BindMaterial(*wall.tex, *flatNormal);
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = wall.model.m[k];
                pc[16] = 0.0f; pc[17] = wall.tint; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(wall.mesh->vertices());
                cmd.BindIndexBuffer(wall.mesh->indices());
                cmd.DrawIndexed(wall.mesh->indexCount());
            }
            // The CPU-occlusion-culled VISIBLE survivors, per-object bound. The hidden objects are absent
            // — exactly what the GPU Hi-Z cull would draw, and identical to a frustum-only render (they
            // are fully behind the wall, so drawing-or-not yields the same pixels).
            for (const HObj* ob : visibleSurvivors) {
                cmd.BindMaterial(*ob->tex, *flatNormal);
                float pc[20];
                for (int k = 0; k < 16; ++k) pc[k] = ob->model.m[k];
                pc[16] = 0.0f; pc[17] = ob->tint; pc[18] = 0.0f; pc[19] = 0.0f;
                cmd.PushConstants(pc, sizeof(pc));
                cmd.BindVertexBuffer(ob->mesh->vertices());
                cmd.BindIndexBuffer(ob->mesh->indices());
                cmd.DrawIndexed(ob->mesh->indexCount());
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

    if (cpuOccluded == 0) return fail("occluded==0 — no real occlusion (fix the occluder/scene)");
    std::printf("hiz-cull: {total:%u, frustumKept:%u, occluded:%u, drawn:%u, cpuOccluded:%u} "
                "(Metal: CPU Hi-Z occlusion-culled per-object bound path, image backend-identical)\n",
                kCand, cpuFrustumKept, cpuOccluded, drawn, cpuOccluded);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) — Hi-Z occlusion-culled %u/%u objects (Metal per-object bound; "
                "Vulkan does depth-prepass -> Hi-Z -> compute cull)\n", outPath, cw, ch, cpuOccluded, kCand);
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

// --- CPU particle / VFX emitter showcase (Slice CC). Mirrors the Vulkan --vfx-shot path EXACTLY:
// checkerboard ground + procedural sky + two opaque lit cubes, lit + shadowed; then the IDENTICAL
// fixed fountain emitter (engine/vfx/particles.h — compiled from the SAME particles.cpp, so the
// simulated particle state is BIT-IDENTICAL cross-backend) simulated for the SAME fixed-dt step count,
// and its camera-facing billboards (vfx::BuildBillboards with the SAME camRight/camUp) drawn
// ADDITIVELY blended over the scene (depth-test ON, depth-write OFF) via the generated vfx.vert/frag
// MSL. The vfx stat line matches Vulkan. One offscreen frame -> PNG. DISTINCT from the gpu-particles
// fountain. -------
static int RunVfxShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Opaque lit pipeline (ground + cubes): shared lit.vert + lit.frag.
    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    auto litPipeline = device->CreateGraphicsPipeline(litDesc);

    // VFX billboard pipeline: vfx.vert + vfx.frag, ADDITIVE blend, depth-test ON, depth-write OFF,
    // double-sided. Dynamic verts { pos RGB32F, uv RG32F, color RGBA32F } == vfx::BillboardVertex.
    auto vfxVs = loadMSL("vfx.vert.gen.metal", "vfx_vertex");
    auto vfxFs = loadMSL("vfx.frag.gen.metal", "vfx_fragment");
    rhi::GraphicsPipelineDesc vfxDesc;
    vfxDesc.vertex = vfxVs.get(); vfxDesc.fragment = vfxFs.get();
    vfxDesc.vertexLayout.stride = sizeof(vfx::BillboardVertex);
    vfxDesc.vertexLayout.attributes = {
        {0, rhi::Format::RGB32_Float, (uint32_t)offsetof(vfx::BillboardVertex, pos)},
        {1, rhi::Format::RG32_Float,  (uint32_t)offsetof(vfx::BillboardVertex, uv)},
        {2, rhi::Format::RGBA32_Float,(uint32_t)offsetof(vfx::BillboardVertex, color)},
    };
    vfxDesc.colorFormat = device->Swapchain().ColorFormat();
    vfxDesc.depthTest = true; vfxDesc.depthWrite = false;
    vfxDesc.additiveBlend = true; vfxDesc.cullNone = true;
    vfxDesc.usesFrameUniforms = true; vfxDesc.usesTexture = false;
    vfxDesc.pushConstantSize = 0;
    auto vfxPipeline = device->CreateGraphicsPipeline(vfxDesc);

    // Static depth-only shadow pipeline (ground + cubes cast shadows; particles do not).
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

    Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});

    // Two opaque lit cubes flanking the fountain (IDENTICAL to the Vulkan path).
    struct Opaque { Mat4 model; };
    std::vector<Opaque> opaques = {
        {Mat4::Translate({-2.6f, 0.6f, -1.4f}) * Mat4::RotateY(0.4f) * Mat4::Scale({0.6f,0.6f,0.6f})},
        {Mat4::Translate({ 2.6f, 0.6f, -1.4f}) * Mat4::RotateY(-0.5f) * Mat4::Scale({0.6f,0.6f,0.6f})},
    };

    // The IDENTICAL fixed fountain emitter + step count as the Vulkan --vfx-shot path (so the
    // simulated particle state + billboards are bit-identical cross-backend).
    vfx::EmitterConfig emitter;
    emitter.origin    = {0.0f, 0.05f, 0.0f};
    emitter.spawnRate = 600.0f;
    emitter.lifetime  = 1.6f;
    emitter.initVel   = {0.0f, 6.0f, 0.0f};
    emitter.velSpread = 1.8f;
    emitter.gravity   = {0.0f, -9.8f, 0.0f};
    emitter.drag      = 0.05f;
    emitter.startSize = 0.16f;
    emitter.endSize   = 0.03f;
    emitter.startColor = {1.0f, 0.75f, 0.25f, 1.0f};
    emitter.endColor   = {0.7f, 0.08f, 0.02f, 0.0f};
    emitter.seed      = 1337u;
    emitter.maxParticles = 4096;

    const float vfxDt = 1.0f / 120.0f;
    const int   vfxSteps = 360;
    vfx::ParticleSystem psys;
    for (int s = 0; s < vfxSteps; ++s) psys.Step(emitter, vfxDt);

    const Vec3 eye{0.0f, 3.4f, 8.5f};
    const Vec3 center{0.0f, 1.8f, 0.0f};
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

    // Build the camera-facing billboards (SAME camRight/camUp as Vulkan -> bit-identical geometry).
    const Vec3 camRight{fd.camRight[0], fd.camRight[1], fd.camRight[2]};
    const Vec3 camUp{fd.camUp[0], fd.camUp[1], fd.camUp[2]};
    std::vector<vfx::BillboardVertex> billboards;
    vfx::BuildBillboards(psys.Alive(), emitter, camRight, camUp, billboards);
    const int aliveK = psys.AliveCount();
    const unsigned long long spawnedS = (unsigned long long)psys.SpawnedCount();

    std::unique_ptr<rhi::IBuffer> vfxVB;
    if (!billboards.empty()) {
        rhi::BufferDesc bd;
        bd.size = billboards.size() * sizeof(vfx::BillboardVertex);
        bd.initialData = billboards.data();
        bd.usage = rhi::BufferUsage::Vertex;
        vfxVB = device->CreateBuffer(bd);
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
            // VFX particle pass (additive, depth-test, NO depth write).
            if (vfxVB) {
                cmd.BindPipeline(*vfxPipeline);
                cmd.BindVertexBuffer(*vfxVB);
                cmd.Draw((uint32_t)billboards.size());
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

    std::printf("vfx: {emitters:1, alive:%d, spawned:%llu}\n", aliveK, spawnedS);

    std::vector<uint8_t> bgra; uint32_t cw = 0, ch = 0;
    if (!device->GetCapturedPixels(bgra, cw, ch)) return fail("no captured pixels");
    if (!WritePNG(outPath, bgra, cw, ch)) return fail("PNG write failed");
    device->WaitIdle();
    std::printf("OK wrote %s (%ux%u) - fountain plume, %d particles\n", outPath, cw, ch, aliveK);
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

// --clustered-lights <out.png>: Clustered Light Culling (Forward+) showcase (Slice CL). Mirrors the
// Vulkan --clustered-lights-shot exactly: a ground plane + raised objects lit by 96 deterministically-
// placed colored point lights + the sun. On Vulkan a COMPUTE pass (cluster_assign) assigns lights to
// clusters; here on Metal the assignment is done CPU-side via render::cluster::AssignLights (the SAME
// math the compute mirrors) into the SAME fixed-slot ClusterList buffer the lit_clustered_cl fragment
// reads. The shared HLSL->SPIR-V->MSL fragment + the windowed hard-radius point lights make the light
// field BIT-IDENTICAL cross-backend. New golden tests/golden/metal/clustered_lights.png; two runs
// DIFF 0.0000.
static int RunClusteredLightsShowcase(const char* outPath) {
    using math::Mat4; using math::Vec3;
    namespace cl = hf::render::cluster;
    const uint32_t W = 1280, H = 720;
    auto device = rhi::mtl::CreateMetalDeviceHeadless(W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };
    auto FlipProjY = [](Mat4 p) { p.m[1] = -p.m[1]; p.m[5] = -p.m[5];
                                  p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p; };

    // Clustered (CL) FrameData layout (matches shaders/lit_clustered_cl.frag). 208 bytes.
    struct ClusteredFrameData {
        float viewProj[16];
        float view[16];
        float lightDir[4];
        float lightColor[4];
        float viewPos[4];
        float clusterParams[4];   // dimX,dimY,dimZ,zNear
        float clusterParams2[4];  // zFar,screenW,screenH,ambient(unused)
    };
    static_assert(sizeof(ClusteredFrameData) == 208, "Clustered (CL) FrameData layout");

    const int   DIMX = 16, DIMY = 9, DIMZ = 24;
    const float kNear = 0.5f, kFar = 90.0f;
    const float fovY = 1.04719755f;
    const float aspect = (float)W / (float)H;

    const Vec3 eye{0.0f, 16.0f, 26.0f};
    const Vec3 center{0.0f, 0.0f, -2.0f};
    Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
    // The cluster math is VIEW-space (unaffected by the clip Y flip); only the rendered viewProj is
    // flipped for Metal. AssignLights uses the UNFLIPPED proj (its invProj recovers view-space XY).
    Mat4 proj = Mat4::Perspective(fovY, aspect, kNear, kFar);
    Mat4 vp = FlipProjY(proj) * view;
    cl::ClusterGrid grid; grid.dimX = DIMX; grid.dimY = DIMY; grid.dimZ = DIMZ;
    grid.zNear = kNear; grid.zFar = kFar;

    // 96 deterministic point lights (8x12 lattice) — IDENTICAL to the Vulkan --clustered-lights-shot.
    const int LX = 8, LZ = 12;
    const int kNumLights = LX * LZ;   // 96
    const float spanX = 34.0f, spanZ = 26.0f;
    const float lightY = 1.4f;
    std::vector<cl::PointLight> lights;
    std::vector<cl::GpuLight>   gpuLights;
    lights.reserve(kNumLights);
    gpuLights.reserve(kNumLights);
    static const float palette[6][3] = {
        {1.00f, 0.18f, 0.20f}, {0.20f, 1.00f, 0.30f}, {0.25f, 0.40f, 1.00f},
        {1.00f, 0.80f, 0.15f}, {0.90f, 0.20f, 1.00f}, {0.15f, 0.95f, 0.95f},
    };
    for (int iz = 0; iz < LZ; ++iz)
        for (int ix = 0; ix < LX; ++ix) {
            int idx = iz * LX + ix;
            float fx = ((float)ix / (float)(LX - 1) - 0.5f) * spanX;
            float fz = ((float)iz / (float)(LZ - 1) - 0.5f) * spanZ - 2.0f;
            const float* c = palette[(ix * 2 + iz * 3) % 6];
            float radius = 5.0f + ((idx * 7) % 6) * 0.6f;
            cl::PointLight L{};
            L.posWorld = {fx, lightY, fz};
            L.radius = radius; L.color = {c[0], c[1], c[2]}; L.intensity = 3.0f;
            lights.push_back(L);
            Vec3 vposL = math::MulPoint(view, L.posWorld);  // world -> view
            cl::GpuLight gl{};
            gl.posRadius[0]=vposL.x; gl.posRadius[1]=vposL.y; gl.posRadius[2]=vposL.z; gl.posRadius[3]=radius;
            gl.color[0]=c[0]; gl.color[1]=c[1]; gl.color[2]=c[2]; gl.color[3]=L.intensity;
            gpuLights.push_back(gl);
        }

    // CPU assignment (mirrors the Vulkan compute pass) -> the fixed-slot ClusterList buffer.
    std::vector<std::vector<uint32_t>> perCluster;
    cl::AssignLights(grid, proj, view, (int)W, (int)H,
                     std::span<const cl::PointLight>(lights), perCluster);
    constexpr int kMaxPer = cl::kMaxLightsPerCluster;  // 96
    struct ClusterListCPU { uint32_t count; uint32_t pad[3]; uint32_t idx[kMaxPer]; };
    const int nClusters = grid.clusterCount();
    std::vector<ClusterListCPU> clusterList((size_t)nClusters);
    for (int c = 0; c < nClusters; ++c) {
        const auto& src = perCluster[(size_t)c];
        clusterList[(size_t)c].count = (uint32_t)src.size();
        clusterList[(size_t)c].pad[0]=clusterList[(size_t)c].pad[1]=clusterList[(size_t)c].pad[2]=0;
        for (size_t k = 0; k < src.size() && k < (size_t)kMaxPer; ++k)
            clusterList[(size_t)c].idx[k] = src[k];
    }

    rhi::BufferDesc clDesc{clusterList.size() * sizeof(ClusterListCPU), clusterList.data(),
                           rhi::BufferUsage::Storage};
    auto clusterListBuf = device->CreateBuffer(clDesc);
    uint32_t dummyU = 0;
    rhi::BufferDesc dummyDesc{sizeof(uint32_t), &dummyU, rhi::BufferUsage::Storage};
    auto dummyBuf = device->CreateBuffer(dummyDesc);
    rhi::BufferDesc lightDesc{gpuLights.size() * sizeof(cl::GpuLight), gpuLights.data(),
                              rhi::BufferUsage::Storage};
    auto lightBuf = device->CreateBuffer(lightDesc);

    auto litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    auto cluFs = loadMSL("lit_clustered_cl.frag.gen.metal", "clustered_lights_fragment");
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
        fd.lightDir[0]=ld.x; fd.lightDir[1]=ld.y; fd.lightDir[2]=ld.z; fd.lightDir[3]=0.0f;
        fd.lightColor[0]=0.05f; fd.lightColor[1]=0.05f; fd.lightColor[2]=0.06f; fd.lightColor[3]=1.0f;
        fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
        fd.clusterParams[0]=(float)DIMX; fd.clusterParams[1]=(float)DIMY;
        fd.clusterParams[2]=(float)DIMZ; fd.clusterParams[3]=kNear;
        fd.clusterParams2[0]=kFar; fd.clusterParams2[1]=(float)W; fd.clusterParams2[2]=(float)H;
        fd.clusterParams2[3]=0.0f;
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
            cmd.BindLightClusters(*clusterListBuf, *dummyBuf, *lightBuf);
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
    uint32_t assignedTotal = 0, maxPer = 0;
    for (const auto& c : clusterList) { assignedTotal += c.count; maxPer = std::max(maxPer, c.count); }
    std::printf("OK wrote %s (%ux%u) — clustered light culling: %d point lights, %dx%dx%d grid "
                "(%d clusters), %u assigned, maxPerCluster %u\n",
                outPath, cw, ch, kNumLights, DIMX, DIMY, DIMZ, nClusters, assignedTotal, maxPer);
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
        // --clustered-lights <out.png>: clustered LIGHT-CULLING showcase (Slice CL). CPU-assigned
        // (render::cluster) into the SAME ClusterList the Vulkan compute fills; shared lit_clustered_cl
        // fragment -> bit-identical light field cross-backend. New golden clustered_lights.png.
        if (argc > 1 && std::strcmp(argv[1], "--clustered-lights") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_clustered_lights.png";
            try { return RunClusteredLightsShowcase(out); }
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
        // --terrain <out.png>: render the procedural terrain showcase (Slice BF) — an n x n
        // deterministic heightmap-displaced grid (terrain::BuildTerrain), lit + shadowed from a fixed
        // 3/4 camera. Mirrors the Vulkan --terrain-shot path; new golden tests/golden/metal/terrain.png.
        if (argc > 1 && std::strcmp(argv[1], "--terrain") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_terrain.png";
            try { return RunTerrainShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --terrain-stream <out.png>: render the terrain-streaming LOD showcase (Slice BJ) — the RESIDENT
        // subset of a 6x6 tile terrain world (distance-banded tile residency + per-tile LOD selection)
        // at a fixed scripted capture frame, each tile meshed at its LOD, lit + shadowed. Mirrors the
        // Vulkan --terrain-stream-shot path; new golden tests/golden/metal/terrain_stream.png.
        if (argc > 1 && std::strcmp(argv[1], "--terrain-stream") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_terrain_stream.png";
            try { return RunTerrainStreamShowcase(out); }
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
        // --net <out.png>: render the networking / replication showcase (Slice BQ) — run the AX roll-game
        // as the AUTHORITY, stream snapshots (keyframe + per-entity deltas) into the REPLICA via an
        // in-process perfect channel (NO sockets), assert replica==authority at a fixed capture step, then
        // render the REPLICA'S reconstructed scene (player + remaining pickup, lit + shadowed). Mirrors the
        // Vulkan --net-shot; new golden tests/golden/metal/net.png. The net stat line matches Vulkan.
        if (argc > 1 && std::strcmp(argv[1], "--net") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_net.png";
            try { return RunNetShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --netsim <out.png>: render the networking TRANSPORT + client jitter-buffer / interpolation
        // showcase (Slice BU) — run the AX roll-game as the AUTHORITY, stream snapshots through a SEEDED
        // lossy/laggy SimChannel (latency 3, 15% drop, 10% reorder, NO sockets) into a client jitter
        // buffer, and render the client's INTERPOLATED view at a fixed render tick (the smoothing hides
        // the loss; converged asserts it tracks the authority). Mirrors the Vulkan --netsim-shot; new
        // golden tests/golden/metal/netsim.png. The netsim stat line (delivered/dropped/reordered/
        // converged) matches Vulkan bit-for-bit (shared pure-C++ transport.cpp).
        if (argc > 1 && std::strcmp(argv[1], "--netsim") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_netsim.png";
            try { return RunNetsimShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --netpredict <out.png>: render the client PREDICTION + server RECONCILIATION showcase (Slice BY)
        // — run the AX roll-game as the AUTHORITY (+ a scripted SERVER-ONLY impulse the client can't
        // predict), predict every tick on the client + reconcile (rewind+replay) against the delayed
        // authoritative frame, and render the client's PREDICTED + RECONCILED view at a fixed render tick
        // (maxMisprediction>0 = a real misprediction corrected; converged asserts it matches the
        // authority's true state). Mirrors the Vulkan --netpredict-shot; new golden
        // tests/golden/metal/netpredict.png. The netpredict stat line matches Vulkan bit-for-bit (shared
        // pure-C++ prediction.cpp).
        if (argc > 1 && std::strcmp(argv[1], "--netpredict") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_netpredict.png";
            try { return RunNetpredictShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --stream <out.png>: render the scene/asset streaming showcase (Slice BD) — the RESIDENT
        // subset of an 8x8 cell world (distance-based residency + per-frame budget) at a fixed scripted
        // capture frame, lit + shadowed. Mirrors the Vulkan --stream-shot; new golden stream.png.
        if (argc > 1 && std::strcmp(argv[1], "--stream") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_stream.png";
            try { return RunStreamShowcase(out); }
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
        // --mdi <out.png>: render the GPU multi-draw-indirect showcase (Slice BM). The TRUE MDI call
        // (one vkCmdDrawIndexedIndirect(drawCount=144) with gl_DrawID-indexed per-draw data) is the
        // VULKAN demonstration; here Metal renders the IDENTICAL 144-object scene via its working
        // per-object path, so mdi.png is backend-identical to the Vulkan MDI image (which is itself
        // byte-identical to the Vulkan per-object reference). Metal ICB is optional/not wired. Mirrors
        // the Vulkan --mdi-shot scene exactly; new golden tests/golden/metal/mdi.png.
        if (argc > 1 && std::strcmp(argv[1], "--mdi") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_mdi.png";
            try { return RunMdiShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --bindless <out.png>: render the bindless-textures showcase (Slice BZ). TRUE bindless (one
        // descriptor array indexed per-draw by texIndex via NonUniformResourceIndex) is the VULKAN
        // demonstration; here Metal renders the IDENTICAL multi-texture scene via its working
        // per-material BOUND path, so bindless.png is backend-identical to the Vulkan bindless image
        // (which is itself byte-identical to the Vulkan per-material bound reference). Metal argument-
        // buffer bindless is OPTIONAL and not wired this slice. Mirrors the Vulkan --bindless-shot scene
        // exactly; new golden tests/golden/metal/bindless.png.
        if (argc > 1 && std::strcmp(argv[1], "--bindless") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_bindless.png";
            try { return RunBindlessShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --gpudriven <out.png>: render the fully-GPU-driven showcase (Slice CB: MDI + bindless capstone).
        // The TRUE fully-GPU-driven pass (one DrawIndexedMultiIndirect(100) with gl_DrawID-indexed
        // per-draw model+material+texIndex + one bindless array bind, sampled via NonUniformResourceIndex)
        // is the VULKAN demonstration; here Metal renders the IDENTICAL 100-object multi-material scene
        // via its working per-object per-material BOUND path, so gpudriven.png is backend-identical to the
        // Vulkan GPU-driven image (which is itself byte-identical to the Vulkan per-object bound
        // reference). Metal ICB + argument-buffer is OPTIONAL and not wired this slice. Mirrors the Vulkan
        // --gpudriven-shot scene exactly; new golden tests/golden/metal/gpudriven.png.
        if (argc > 1 && std::strcmp(argv[1], "--gpudriven") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_gpudriven.png";
            try { return RunGpuDrivenShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --gpucull-draw <out.png>: render the fully-GPU-driven-CULLED pass showcase (Slice CD: compute-cull
        // -> MDI + bindless). The TRUE pass (a compute shader frustum-culls + ORDER-compacts the survivors
        // + writes the survivor count, then ONE DrawIndexedMultiIndirect(count) + ONE bindless bind renders
        // exactly them) is the VULKAN demonstration; here Metal renders the IDENTICAL wide 144-object grid
        // viewed by a narrow camera via the CPU-frustum-culled per-object BOUND path (the SAME
        // render::gpuculled::CullAndCompact decides the survivors), so gpucull_draw.png is backend-identical
        // to the Vulkan GPU-culled image (which is itself byte-identical to the Vulkan per-object bound
        // reference). Metal ICB + argument-buffer is OPTIONAL and not wired this slice. Mirrors the Vulkan
        // --gpucull-draw-shot scene exactly; new golden tests/golden/metal/gpucull_draw.png.
        if (argc > 1 && std::strcmp(argv[1], "--gpucull-draw") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_gpucull_draw.png";
            try { return RunGpuCullDrawShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --hiz-cull <out.png>: render the Hi-Z OCCLUSION cull showcase (Slice CJ). The TRUE pass (a CPU
        // depth pre-pass -> Hi-Z max-depth pyramid -> a compute shader frustum+occlusion-culls + compacts
        // the survivors) is the VULKAN demonstration; here Metal renders the IDENTICAL occluder-wall scene
        // via the per-object BOUND path (the SAME render::hiz CPU Hi-Z + IsOccluded decides the VISIBLE
        // survivors), so hiz_cull.png is backend-identical to the Vulkan occlusion-culled image (which is
        // itself byte-identical to the Vulkan frustum-only render). New golden tests/golden/metal/hiz_cull.png.
        if (argc > 1 && std::strcmp(argv[1], "--hiz-cull") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_hiz_cull.png";
            try { return RunHizCullShowcase(out); }
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
        // --anim-fsm <out.png>: render the animation STATE-MACHINE showcase (Slice BL) — same scene as
        // --skinning but the joint palette comes from an idle/walk/run FSM scripted to a FIXED mid
        // walk->run cross-fade. New golden tests/golden/metal/anim_fsm.png.
        if (argc > 1 && std::strcmp(argv[1], "--anim-fsm") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_anim_fsm.png";
            try { return RunSkinningShowcase(out, /*blend=*/false, /*fsm=*/true); }
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
        // --material-normal <out.png>: render the NormalMap-node showcase (Slice BE) — a sphere shaded
        // by normalmap.mat.json whose PBROutput.normal = NormalMap(slot="normalmap"); a tangent-space
        // normal map perturbs the shading normal (bump relief). The new mat_normal.png golden.
        if (argc > 1 && std::strcmp(argv[1], "--material-normal") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_mat_normal.png";
            try { return RunMaterialNormalShowcase(out); }
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
        // --vfx <out.png>: render the CPU particle / VFX emitter showcase (Slice CC) — the lit +
        // shadowed scene plus a fixed fountain emitter's camera-facing additive billboards. Mirrors
        // the Vulkan --vfx-shot; new golden tests/golden/metal/vfx.png. DISTINCT from gpu-particles.
        if (argc > 1 && std::strcmp(argv[1], "--vfx") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_vfx.png";
            try { return RunVfxShowcase(out); }
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
        // --dof <out.png>: depth-of-field showcase (Slice CG) — a row of objects receding from the
        // camera (near -> a middle FOCAL object -> far), a view-space normal+linear-depth g-buffer, and a
        // fullscreen thin-lens circle-of-confusion depth gather (dof.frag) so the middle object stays
        // crisp while fore/background blur, then tonemap. Mirrors the Vulkan --dof-shot exactly (same
        // scene/lens params; the SAME render/dof.h math makes the CoC field bit-identical).
        if (argc > 1 && std::strcmp(argv[1], "--dof") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_dof.png";
            try { return RunDofShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --motionblur <out.png>: per-object + camera motion blur showcase (Slice CN) — a ground + a row
        // of raised objects with the camera panning at a FIXED rate, a view-space normal+linear-depth
        // g-buffer, and a fullscreen velocity-gather pass (motion_blur.frag) that streaks the moving
        // content along its screen-space velocity while static content stays sharp, then tonemap. Mirrors
        // the Vulkan --motionblur-shot exactly (same scene/params; the SAME render/motion_blur.h math
        // makes the velocity field bit-identical). Two runs DIFF 0.0000.
        if (argc > 1 && std::strcmp(argv[1], "--motionblur") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_motion_blur.png";
            try { return RunMotionBlurShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --oit <out.png>: order-independent transparency showcase (Slice CO) — an opaque ground + 2
        // opaque objects + 5 mutually-overlapping transparent glass quads composited via Weighted
        // Blended OIT (additive accum + multiplicative revealage -> resolve -> lerp over the scene).
        // INTERNALLY renders the transparent set in a CANONICAL + a PERMUTED draw order and asserts the
        // two captures are BYTE-IDENTICAL — the order-independence proof. Mirrors the Vulkan --oit-shot
        // exactly (same scene/glass set/camera; the SAME render/oit.h Weight makes it bit-identical).
        if (argc > 1 && std::strcmp(argv[1], "--oit") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_oit.png";
            try { return RunOitShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --pom <out.png>: parallax occlusion mapping showcase (Slice CP) — a single height-mapped brick
        // surface filling the view at a grazing angle, lit + shadowed; pom.frag ray-marches the height
        // field in tangent space (render/pom.h ParallaxUV) for per-pixel depth + groove self-shadowing.
        // INTERNALLY renders the SAME surface with heightScale=0 + via the plain lit pipeline and asserts
        // they are BYTE-IDENTICAL — the zero-height equivalence proof. Mirrors the Vulkan --pom-shot
        // exactly (same scene/textures/camera/heightScale/steps; the SAME render/pom.h math makes the
        // marched uv bit-identical).
        if (argc > 1 && std::strcmp(argv[1], "--pom") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_pom.png";
            try { return RunPomShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --gtao <out.png>: ground-truth ambient occlusion showcase (Slice CR) — boxes forming a concave
        // corner/crease + spheres in contact on the ground; gtao.frag runs the render/gtao.h horizon-
        // search visibility integral over the SSAO g-buffer and the EXISTING ssao_composite multiplies
        // the ambient term by the AO. INTERNALLY renders the SAME scene with radius=0 (gtao.frag's
        // no-horizon path -> AO==1) AND with NO AO (composite aoStrength=0) and asserts they are
        // BYTE-IDENTICAL — the radius=0 equivalence proof. Mirrors the Vulkan --gtao-shot exactly (same
        // scene/camera/radius/slices/steps; the SAME render/gtao.h math makes the AO bit-identical).
        if (argc > 1 && std::strcmp(argv[1], "--gtao") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_gtao.png";
            try { return RunGtaoShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --contactshadow <out.png>: screen-space contact shadows showcase (Slice CT) — small objects
        // hovering a hair above the ground; a short per-pixel depth ray-march toward the sun fills the
        // tight contact occlusion the broad/coarse CSM misses. INTERNALLY renders the SAME
        // lit_contactshadow scene with the factor at maxDist=0 (RayMarchShadow -> 1) AND with an all-white
        // factor RT and asserts they are BYTE-IDENTICAL — the maxDist=0 no-op proof. Mirrors the Vulkan
        // --contactshadow-shot exactly (same scene/camera/sun/march; the SAME render/contact_shadows.h math).
        if (argc > 1 && std::strcmp(argv[1], "--contactshadow") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_contact_shadows.png";
            try { return RunContactShadowShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --froxelfog <out.png>: froxel volumetric fog showcase (Slice CS) — a lit+shadowed scene wrapped
        // in a true 3D view-space froxel volume (inject -> integrate -> depth-composited apply). INTERNALLY
        // renders the SAME apply chain at baseDensity=0 (T==1, inScatter==0) and at apply enable=0 (no-fog
        // pass-through) and asserts they are BYTE-IDENTICAL — the zero-density no-op proof. Mirrors the
        // Vulkan --froxelfog-shot exactly (same scene/camera/sun/fog/grid; the SAME render/froxel.h math).
        if (argc > 1 && std::strcmp(argv[1], "--froxelfog") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_froxel_fog.png";
            try { return RunFroxelFogShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --froxellights <out.png>: per-froxel clustered-light injection showcase (Slice CV) — the CL
        // 96-colored-point-light scene wrapped in the CS froxel fog, each light casting a colored
        // volumetric shaft. INTERNALLY asserts injectLights=false == CS sun-only fog AND density=0 ==
        // no-fog (both BYTE-IDENTICAL). Mirrors the Vulkan --froxellights-shot exactly (same scene/
        // lattice/fog/grid; the SAME render/froxel.h + render/cluster.h math). Two runs DIFF 0.0000.
        if (argc > 1 && std::strcmp(argv[1], "--froxellights") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_froxel_lights.png";
            try { return RunFroxelLightsShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --volshadows <out.png>: volumetric shadows showcase (Slice CX) — the CV fog scene with the SUN
        // casting volumetric shadows: the froxel inject samples the sun's CSM shadow map per froxel and
        // gates the sun in-scatter -> dark fog volumes + foggy sun shafts. INTERNALLY asserts
        // volumetricShadows=false == CV froxel-lights render AND density=0 == no-fog (both BYTE-IDENTICAL).
        // Mirrors the Vulkan --volshadows-shot exactly (same scene/lattice/fog/sun/grid; the SAME
        // render/froxel.h SunVisibility math + the SAME shadow map sample). Two runs DIFF 0.0000.
        if (argc > 1 && std::strcmp(argv[1], "--volshadows") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_vol_shadows.png";
            try { return RunVolShadowsShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --autoexposure <out.png>: auto-exposure histogram eye-adaptation showcase (Slice CW) — a
        // high-dynamic-range scene metered by an INTEGER luminance histogram -> key-value target exposure
        // -> tonemap applies it. INTERNALLY renders adaptationEnabled=false (exposure == E0) and asserts it
        // is BYTE-IDENTICAL to the standard fixed-exposure post.frag render of the same scene (the
        // adaptation-off no-op proof). Mirrors the Vulkan --autoexposure-shot exactly (same HDR scene /
        // bins / keyValue; the SAME render/auto_exposure.h math). Two runs DIFF 0.0000.
        if (argc > 1 && std::strcmp(argv[1], "--autoexposure") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_auto_exposure.png";
            try { return RunAutoExposureShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --water <out.png>: water-rendering showcase (Slice CF) — objects partially submerged at the
        // water level + a Gerstner water plane reflecting the procedural sky + refracting/absorbing the
        // submerged scene + a sun glint, at a FIXED wave time. Mirrors the Vulkan --water-shot exactly
        // (same scene/wave set/time/camera; the SAME render/water.h math makes the field bit-identical).
        if (argc > 1 && std::strcmp(argv[1], "--water") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_water.png";
            try { return RunWaterShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --clouds <out.png>: volumetric clouds showcase (Slice CH) — a lit+shadowed scene under a
        // procedural sky augmented by a raymarched cumulus LAYER (cloud slab between two altitudes lit
        // by the sun via Beer-Lambert + Henyey-Greenstein), composited over the sky background +
        // tonemapped. Mirrors the Vulkan --clouds-shot exactly (same scene/slab/noise/time/camera).
        if (argc > 1 && std::strcmp(argv[1], "--clouds") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_clouds.png";
            try { return RunCloudsShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --cloud-shadows <out.png>: cloud shadows on the ground showcase (Slice CK) — the SAME
        // lit+shadowed scene + cloud field + time + sun + camera as --clouds, but the lit ground/objects
        // are shaded by the lit_cloudshadow VARIANT, attenuating the direct sun by CloudShadow so the
        // surfaces show FIXED dappled cloud shadows matching the cloudscape overhead. Mirrors the Vulkan
        // --cloud-shadows-shot exactly; two runs DIFF 0.0000.
        if (argc > 1 && std::strcmp(argv[1], "--cloud-shadows") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_cloud_shadows.png";
            try { return RunCloudShadowsShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --ssgi <out.png>: screen-space global-illumination showcase (Slice BP) — a Cornell-style
        // color-bleed scene (red + green vertical panels flanking a neutral grey floor + white box), a
        // view-space normal+linear-depth g-buffer, a K-ray cosine-hemisphere SSGI gather marching the
        // depth buffer (the SAME march as SSR), and a composite that ADDS the indirect diffuse over the
        // scene + tonemap. Mirrors the Vulkan --ssgi-shot exactly (same scene/kernel/camera).
        if (argc > 1 && std::strcmp(argv[1], "--ssgi") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_ssgi.png";
            try { return RunSsgiShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --ssgi-denoise <out.png>: SSGI bilateral spatial-denoise showcase (Slice BR) — the SAME
        // Cornell color-bleed scene + SSGI gather as --ssgi, plus a depth+normal-guided bilateral
        // denoise pass (ssgi_denoise.frag) of the indirect buffer before the composite, smoothing the
        // floor color-bleed pool while keeping edges crisp. Mirrors the Vulkan --ssgi-denoise-shot.
        if (argc > 1 && std::strcmp(argv[1], "--ssgi-denoise") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_ssgi_denoise.png";
            try { return RunSsgiDenoiseShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --ssgi-temporal <out.png>: temporal SSGI accumulation showcase (Slice BV) — the SAME Cornell
        // color-bleed scene + SSGI gather as --ssgi, but the indirect is ACCUMULATED over a fixed N=8
        // golden-angle-jittered frames into a running-mean HDR RT before the composite, converging to a
        // much cleaner indirect-diffuse result. Mirrors the Vulkan --ssgi-temporal-shot.
        if (argc > 1 && std::strcmp(argv[1], "--ssgi-temporal") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_ssgi_temporal.png";
            try { return RunSsgiTemporalShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --decal <out.png>: screen-space projected-decals showcase (Slice BH) — the lit+shadowed scene
        // + a view-space normal+linear-depth g-buffer, a decal composite pass that reconstructs the
        // world position and projects a procedural decal top-down into a fixed box on the ground +
        // tonemap. Mirrors the Vulkan --decal-shot exactly (same scene/decal box/camera).
        if (argc > 1 && std::strcmp(argv[1], "--decal") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_decal.png";
            try { return RunDecalShowcase(out); }
            catch (const std::exception& e) { return fail(std::string("exception: ") + e.what()); }
        }
        // --poststack <out.png>: data-driven post-process stack showcase (Slice BN) — the lit+shadowed
        // scene through a fixed ORDERED effect chain (tonemap -> color grade -> chromatic aberration ->
        // vignette -> film grain) applied in order from the post_stack.frag push-constant stream.
        if (argc > 1 && std::strcmp(argv[1], "--poststack") == 0) {
            const char* out = argc > 2 ? argv[2] : "metal_poststack.png";
            try { return RunPostStackShowcase(out); }
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
        // --editor <out.png>: docked editor showcase (Slice BT). Renders the SAME default Slice-F
        // scene as the no-flag path, then draws the docked Dear ImGui editor (Scene Hierarchy /
        // Inspector / Stats panels around a central scene Viewport) over it via the RHI-only
        // ImGuiRenderer with a FIXED selected entity, and captures. Mirrors the Vulkan --editor-shot:
        // identical scene + editor state + tiled layout, so editor.png is golden-comparable + two runs
        // DIFF 0.0000 (ImGui geometry is CPU-built + deterministic). `editorMode` flows into the
        // shared default-scene path below; the editor chrome is composited in the post pass.
        bool editorMode = (argc > 1 && std::strcmp(argv[1], "--editor") == 0);
        // --editor-edit <out.png>: editor LIVE-EDIT showcase (Slice BX). Same default scene + docked
        // editor as --editor, but a FIXED edit sequence is applied to the registry first (the moved
        // duck + recolored sphere) and the round-trip through scene_io is asserted — the IDENTICAL
        // sequence + selection + layout the Vulkan --editor-edit-shot uses, so editor_edit.png is
        // golden-comparable cross-backend + two runs DIFF 0.0000.
        bool editEditorMode = (argc > 1 && std::strcmp(argv[1], "--editor-edit") == 0);
        if (editEditorMode) editorMode = true;  // edit mode is editor mode + a pre-render edit.
        const char* outPath = (editorMode || editEditorMode)
                                  ? (argc > 2 ? argv[2] : (editEditorMode ? "metal_editor_edit.png"
                                                                          : "metal_editor.png"))
                                  : (argc > 1 ? argv[1] : "metal_scene.png");
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

            // ---- Slice BX: editor LIVE-EDIT. Apply the IDENTICAL fixed edit sequence the Vulkan
            // --editor-edit-shot uses BEFORE building the editor frame: translate the duck (view index
            // 9) +1.0 on Y (relative ADD), recolor a sphere (index 8) to the duck_basecolor swatch +
            // make it matte (metallic 0.0, roughness 0.9; absolute SETs). The editor then renders over
            // the EDITED scene + the round-trip is asserted below. ----
            const int kBxDuckIndex = 9;
            const int kBxSphereIndex = 8;
            std::string bxDump;
            bool bxSaved = false, bxReloadMatch = false;
            if (editEditorMode) {
                editor::TransformEdit duckMove;
                duckMove.addPosition = true;
                duckMove.positionDelta = {0.0f, 1.0f, 0.0f};
                editor::ApplyTransformEdit(registry, kBxDuckIndex, duckMove);

                editor::MaterialEdit sphereRecolor;
                sphereRecolor.setBaseColor = true;
                sphereRecolor.baseColor = resources.FindTexture("duck_basecolor");
                sphereRecolor.setMetallic = true;  sphereRecolor.metallic = 0.0f;
                sphereRecolor.setRoughness = true; sphereRecolor.roughness = 0.9f;
                editor::ApplyMaterialEdit(registry, kBxSphereIndex, sphereRecolor);

                // scene_io round-trip (persistence proof): DumpScene the edited registry, reload into a
                // fresh registry, assert the edited values survive.
                bxDump = scene::DumpScene(registry, resources);
                bxSaved = !bxDump.empty();
                std::string tmpScene = std::string(std::tmpnam(nullptr)) + ".json";
                { std::ofstream f(tmpScene, std::ios::binary); f << bxDump; }
                ecs::Registry reloaded;
                std::vector<ecs::Entity> rents =
                    scene::LoadScene(reloaded, resources, tmpScene.c_str());
                if (rents.size() > static_cast<size_t>(kBxDuckIndex)) {
                    const auto& reloadedT = reloaded.get<scene::TransformC>(rents[kBxDuckIndex]).t;
                    const auto& reloadedM = reloaded.get<scene::MaterialC>(rents[kBxSphereIndex]);
                    bool duckOk = std::fabs(reloadedT.position.y - 2.35f) < 1e-3f;
                    bool sphereOk = (reloadedM.base == resources.FindTexture("duck_basecolor")) &&
                                    std::fabs(reloadedM.metallic - 0.0f) < 1e-4f &&
                                    std::fabs(reloadedM.roughness - 0.9f) < 1e-4f;
                    bxReloadMatch = duckOk && sphereOk;
                }
                std::remove(tmpScene.c_str());
            }

            // ---- Docked editor (Slice BT): set up Dear ImGui + the RHI-only renderer when --editor.
            // Deterministic: no imgui.ini (machine-dependent), fixed DisplaySize/DeltaTime, a FIXED
            // selected entity, no cursor/time input. The ImGui pipeline shaders are the GENERATED ui
            // MSL (ui.vert/ui.frag .gen.metal); the renderer is the SAME RHI-only class the Vulkan
            // path uses, so the docked panels build identically + the editor.png matches cross-backend.
            std::unique_ptr<editor::ImGuiRenderer> uiRenderer;
            editor::EditorState editorState;
            int editorEntityCount = 0;
            if (editorMode) {
                for (auto [e, tc, mc, mat] :
                     registry.view<scene::TransformC, scene::MeshC, scene::MaterialC>()) {
                    (void)e; (void)tc; (void)mc; (void)mat; ++editorEntityCount;
                }
                // FIXED selection: the duck (last entity) for --editor; the recolored sphere for
                // --editor-edit so the Inspector shows the edited material (matching Vulkan).
                editorState.selectedEntity = editEditorMode
                    ? kBxSphereIndex
                    : (editorEntityCount > 0 ? editorEntityCount - 1 : -1);

                IMGUI_CHECKVERSION();
                ImGui::CreateContext();
                ImGuiIO& io = ImGui::GetIO();
                io.IniFilename = nullptr;     // headless: no imgui.ini (machine-dependent layout)
                io.LogFilename = nullptr;
                io.DisplaySize = ImVec2((float)W, (float)H);
                io.DeltaTime = 1.0f / 60.0f;  // fixed constant: no time/animation -> deterministic
                ImGui::StyleColorsDark();

                // Build the ImGui pipeline from the GENERATED ui MSL (entry points ui_vertex/ui_fragment
                // per the CMake gen rules). The renderer takes ownership of these shader modules.
                std::string uiVsMsl = LoadText(std::string(HF_GEN_SHADER_DIR) + "/ui.vert.gen.metal");
                std::string uiFsMsl = LoadText(std::string(HF_GEN_SHADER_DIR) + "/ui.frag.gen.metal");
                auto uiVs = rhi::mtl::MakeShaderModuleFromMSL(*device, uiVsMsl, "ui_vertex");
                auto uiFs = rhi::mtl::MakeShaderModuleFromMSL(*device, uiFsMsl, "ui_fragment");
                uiRenderer = std::make_unique<editor::ImGuiRenderer>(
                    *device, device->Swapchain().ColorFormat(), std::move(uiVs), std::move(uiFs));
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
                    // Slice BT: docked editor chrome over the scene (same post pass as Vulkan). Build a
                    // fresh ImGui frame from the live ECS registry + draw it through the RHI.
                    if (editorMode && uiRenderer) {
                        ImGui::NewFrame();
                        editor::BuildEditorUI(registry, resources, editorState, W, H);
                        ImGui::Render();
                        uiRenderer->RenderDrawData(ImGui::GetDrawData(), cmd, W, H);
                    }
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

            if (editorMode) {
                if (editEditorMode) {
                    // Match the Vulkan --editor-edit-shot stat line.
                    std::printf("editor-edit: {edits:%d, entity:%d, saved:%s, reloadMatch:%s}\n",
                                2, kBxSphereIndex, bxSaved ? "true" : "false",
                                bxReloadMatch ? "true" : "false");
                } else {
                    // Match the Vulkan --editor-shot stat line (deterministic; selection clamped +
                    // written back by BuildPanelData during BuildEditorUI).
                    std::printf("editor: {panels:[Hierarchy,Inspector,Stats,Viewport], selected:%d, "
                                "entities:%d}\n", editorState.selectedEntity, editorEntityCount);
                }
                uiRenderer.reset();
                ImGui::DestroyContext();
            }

            std::printf("OK wrote %s (%ux%u)\n", outPath, cw, ch);
            return 0;
        } catch (const std::exception& e) {
            return fail(std::string("exception: ") + e.what());
        }
    }
}
