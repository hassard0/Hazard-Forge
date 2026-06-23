// Slice AN — native Cocoa/AppKit windowed Metal interactive viewport (the `--fly` loop on macOS).
//
// WHY native Cocoa and not the SDL `hello_triangle` sample: the Mac Mini dev rig has Command Line
// Tools only (no Xcode), and crucially NO SDL3 (no libSDL3, no pkg-config, no conan) — the whole
// SDL windowed path (engine/hal/window.cpp + the MetalDevice(Window&) ctor) cannot build there. So,
// exactly like the sibling `metal_headless/` target, this is a STANDALONE conan-free / SDL-free
// CMake target that drives the REAL Metal RHI classes — but WINDOWED: it opens an NSWindow whose
// content view is layer-backed by a CAMetalLayer, builds the same Slice-F scene + render graph as
// `metal_headless`'s RunSceneShowcase, and runs the SAME backend-agnostic runtime loop the Windows
// Vulkan `--fly` viewport runs: pump input -> advance the camera -> render -> present the drawable.
//
//   Build: configured by the sibling CMakeLists.txt (cmake -S mac_window -B build-mac-window -G Ninja)
//   Run:   ./mac_window          (opens a 1280x720 window; WASD move, RIGHT-drag look, ESC quits)
//
// HONESTY: a live on-screen window CANNOT be verified over SSH (no display session). The deliverable
// verified over SSH is that this COMPILES + LINKS. The user confirms the window opens / renders /
// flies by running the binary on the Mac Mini's physical display.

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

#include "platform/crash_dialogs.h"     // hf::platform::DisableCrashDialogs() -- no-op on Apple/clang
#include "rhi/rhi.h"
#include "rhi_metal/metal_windowed.h"   // CreateMetalDeviceWindowedLayer (Metal-free)
#include "rhi_metal/metal_shader_load.h" // MakeShaderModuleFromMSL (Metal-free)
#include "math/math.h"
#include "scene/mesh.h"
#include "scene/vertex.h"   // scene::MeshVertexLayout (the lit/pbr/shadow pipeline vertex layout)
#include "asset/gltf_loader.h"
#include "render/render_graph.h"
#include "runtime/camera.h"
#include "runtime/clock.h"
#include "runtime/fly_camera_controller.h"
#include "runtime/input_state.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hf;

// ---- Per-frame uniform block — byte-for-byte the FrameData the generated MSL reads (same struct
// the headless visual_test uses; keep in lockstep). ------------------------------------------------
struct FrameData {
    float vp[16];
    float lightDir[4];
    float lightColor[4];
    float viewPos[4];
    float ptCount[4];
    // Point-light arrays — HF_MAX_POINT_LIGHTS=8 (issue #3, matches shaders/frame_data.hlsli).
    float ptPos[8][4];
    float ptColor[8][4];
    float lightViewProj[16];
    float camFwd[4];
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

// 256x256 RGBA8 checkerboard — same generator as the headless scene showcase (ground texture).
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

// ---------------------------------------------------------------------------------------------------
// The interactive viewport. Holds the windowed Metal device, the Slice-F scene + render graph, and
// the runtime camera/controller; AppKit feeds it a runtime::InputState each frame. Construction
// mirrors metal_headless RunSceneShowcase byte-for-byte (same scene, same shaders, same graph) so the
// rendered frame matches the committed scene golden — only the camera is live and the present target
// is the window's CAMetalLayer instead of an offscreen texture.
// ---------------------------------------------------------------------------------------------------
struct Viewport {
    uint32_t W = 1280, H = 720;
    std::unique_ptr<rhi::IRHIDevice> device;

    // Pipelines / resources (held for the lifetime of the loop).
    std::unique_ptr<rhi::IShaderModule> litVs, pbrFs, litFs, shadowVs, skyVs, skyFs, postVs, postFs;
    std::unique_ptr<rhi::IPipeline> pbrPipeline, litPipeline, shadowPipeline, skyPipe, postPipe;
    std::unique_ptr<rhi::IRenderTarget> rt, shadowMap;
    std::unique_ptr<rhi::ITexture> groundTex, flatNormal;
    std::optional<scene::Mesh> plane;  // scene::Mesh has no default ctor; constructed in Init().
    hf::asset::GltfScene truck;
    math::Mat4 placement, groundModel;

    render::RenderGraph graph;

    // Live camera + controller + fixed-timestep accumulator.
    runtime::Camera cam;
    runtime::FlyCameraController controller;
    runtime::FixedTimestep clock{1.0f / 120.0f};

    FrameData fd{};

    void Init(void* caMetalLayer);
    void RenderFrame(const runtime::InputState& in, float dt);
    void Resize(uint32_t w, uint32_t h);

private:
    static math::Mat4 FlipProjY(math::Mat4 p) {
        p.m[1] = -p.m[1]; p.m[5] = -p.m[5]; p.m[9] = -p.m[9]; p.m[13] = -p.m[13]; return p;
    }
    void UpdateFrameData();
};

void Viewport::Init(void* caMetalLayer) {
    using math::Mat4; using math::Vec3;
    device = rhi::mtl::CreateMetalDeviceWindowedLayer(caMetalLayer, W, H);

    auto loadMSL = [&](const char* file, const char* entry) {
        std::string src = LoadText(std::string(HF_GEN_SHADER_DIR) + "/" + file);
        return rhi::mtl::MakeShaderModuleFromMSL(*device, src, entry);
    };

    // Lit-PBR pipeline (truck), static lit (ground), depth-only shadow, sky, post — identical to the
    // headless RunSceneShowcase.
    litVs = loadMSL("lit.vert.gen.metal", "vertex_main");
    pbrFs = loadMSL("lit_pbr.frag.gen.metal", "pbr_fragment");
    rhi::GraphicsPipelineDesc pbrDesc;
    pbrDesc.vertex = litVs.get(); pbrDesc.fragment = pbrFs.get();
    pbrDesc.vertexLayout = scene::MeshVertexLayout();
    pbrDesc.colorFormat = device->Swapchain().ColorFormat();
    pbrDesc.depthTest = true; pbrDesc.usesFrameUniforms = true;
    pbrDesc.usesTexture = true; pbrDesc.pbrMaterial = true;
    pbrDesc.pushConstantSize = sizeof(float) * 20;
    pbrPipeline = device->CreateGraphicsPipeline(pbrDesc);

    litFs = loadMSL("lit.frag.gen.metal", "fragment_main");
    rhi::GraphicsPipelineDesc litDesc;
    litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
    litDesc.vertexLayout = scene::MeshVertexLayout();
    litDesc.colorFormat = device->Swapchain().ColorFormat();
    litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
    litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
    litPipeline = device->CreateGraphicsPipeline(litDesc);

    shadowVs = loadMSL("shadow.vert.gen.metal", "shadow_vertex");
    rhi::GraphicsPipelineDesc shDesc;
    shDesc.vertex = shadowVs.get(); shDesc.fragment = nullptr;
    shDesc.vertexLayout = scene::MeshVertexLayout();
    shDesc.depthTest = true; shDesc.depthOnly = true;
    shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
    shadowPipeline = device->CreateGraphicsPipeline(shDesc);

    skyVs = loadMSL("sky.vert.gen.metal", "sky_vertex");
    skyFs = loadMSL("sky.frag.gen.metal", "sky_fragment");
    rhi::GraphicsPipelineDesc skyD;
    skyD.vertex = skyVs.get(); skyD.fragment = skyFs.get();
    skyD.colorFormat = device->Swapchain().ColorFormat();
    skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
    skyPipe = device->CreateGraphicsPipeline(skyD);

    postVs = loadMSL("post.vert.gen.metal", "post_vertex");
    postFs = loadMSL("post.frag.gen.metal", "post_fragment");
    rhi::GraphicsPipelineDesc postD;
    postD.vertex = postVs.get(); postD.fragment = postFs.get();
    postD.colorFormat = device->Swapchain().ColorFormat();
    postD.depthTest = false; postD.usesFrameUniforms = false;
    postD.usesTexture = true; postD.fullscreen = true;
    postPipe = device->CreateGraphicsPipeline(postD);

    rt = device->CreateRenderTarget(W, H);
    shadowMap = device->CreateShadowMap(2048);
    device->SetShadowMap(*shadowMap);

    std::vector<uint8_t> checker = MakeCheckerboard();
    groundTex = device->CreateTexture(
        {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
    const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
    flatNormal = device->CreateTexture(
        {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
    plane.emplace(scene::Mesh::Plane(*device));

    truck = hf::asset::LoadGltfScene(*device, HF_TRUCK_MODEL_PATH);
    std::printf("[scene] imported %zu instances, %zu meshes, %zu materials\n",
                truck.instances.size(), truck.meshStorage.size(), truck.materialStorage.size());

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
    placement = sceneFit * orient;
    groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});

    // Live camera: start at the headless scene's fixed eye, looking at the truck. yaw/pitch derived
    // from the look direction so the fly controller picks up smoothly from this pose.
    const Vec3 eye{5.0f, 3.2f, 6.0f};
    const Vec3 center{0.0f, 1.0f, 0.0f};
    Vec3 dir = math::normalize(center - eye);
    cam.position = eye;
    cam.yaw = std::atan2(dir.x, -dir.z);
    cam.SetPitch(std::asin(dir.y));
    cam.fovY = 1.04719755f;
    cam.aspect = (float)W / (float)H;
    cam.znear = 0.1f; cam.zfar = 100.0f;

    // Build the render graph ONCE; the pass lambdas read the live `fd`/scene each Execute().
    render::RgResource rgShadow = graph.ImportTarget(
        "shadowMap", render::RgResourceKind::ShadowMap, *shadowMap);
    render::RgResource rgScene = graph.ImportTarget(
        "sceneColor", render::RgResourceKind::SceneColor, *rt);
    render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

    graph.AddPass("shadow", {}, {rgShadow},
        [this](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
            dev.SetFrameUniforms(&fd, sizeof(FrameData));
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*shadowPipeline);
            cmd.PushConstants(groundModel.m, sizeof(float) * 16);
            cmd.BindVertexBuffer(plane->vertices());
            cmd.BindIndexBuffer(plane->indices());
            cmd.DrawIndexed(plane->indexCount());
            for (const auto& inst : truck.instances) {
                math::Mat4 world = placement * inst.worldTransform;
                cmd.PushConstants(world.m, sizeof(float) * 16);
                cmd.BindVertexBuffer(inst.mesh->vertices());
                cmd.BindIndexBuffer(inst.mesh->indices());
                cmd.DrawIndexed(inst.mesh->indexCount());
            }
            cmd.EndRenderPass();
        });

    graph.AddPass("scene", {rgShadow}, {rgScene},
        [this](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
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
                cmd.BindVertexBuffer(plane->vertices());
                cmd.BindIndexBuffer(plane->indices());
                cmd.DrawIndexed(plane->indexCount());
            }
            cmd.BindPipeline(*pbrPipeline);
            for (const auto& inst : truck.instances) {
                math::Mat4 world = placement * inst.worldTransform;
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
        [this](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
            cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
            cmd.BindPipeline(*postPipe);
            cmd.BindTexture(*rt);
            cmd.Draw(3);
            cmd.EndRenderPass();
        });

    UpdateFrameData();
}

// Recompute the camera-dependent FrameData (view-proj + camera basis). The light's view-proj +
// color and sky params are constant for this scene; computed once here too.
void Viewport::UpdateFrameData() {
    using math::Mat4; using math::Vec3;
    Mat4 view = cam.View();
    Mat4 proj = FlipProjY(cam.Proj());  // FlipProjY matches the headless MSL Y convention.
    Mat4 vp = proj * view;
    for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];

    fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
    fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
    fd.viewPos[0] = cam.position.x; fd.viewPos[1] = cam.position.y;
    fd.viewPos[2] = cam.position.z; fd.viewPos[3] = 1.0f;
    fd.ptCount[0] = 0.0f;

    Vec3 sc{0.0f, 1.0f, 0.0f};
    Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
    Vec3 lightEye = sc - lightDir * 14.0f;
    Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
    Mat4 lightOrtho = FlipProjY(Mat4::Ortho(-6.0f, 6.0f, -6.0f, 6.0f, 1.0f, 30.0f));
    Mat4 lightVP = lightOrtho * lightView;
    for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];

    runtime::CameraBasis b = cam.Basis();
    fd.camFwd[0]=b.forward.x; fd.camFwd[1]=b.forward.y; fd.camFwd[2]=b.forward.z;
    fd.camRight[0]=b.right.x; fd.camRight[1]=b.right.y; fd.camRight[2]=b.right.z;
    fd.camUp[0]=b.up.x; fd.camUp[1]=b.up.y; fd.camUp[2]=b.up.z;
    fd.skyParams[0] = b.tanHalfFovY;
    fd.skyParams[1] = b.aspect;
}

void Viewport::RenderFrame(const runtime::InputState& in, float dt) {
    // Advance the camera from input (same FlyCameraController the Windows --fly viewport uses). The
    // FixedTimestep accumulator is ticked for parity with --fly's loop; camera motion uses real dt.
    clock.Tick(dt);
    controller.Update(cam, in, dt);
    UpdateFrameData();
    graph.Execute(*device);
}

void Viewport::Resize(uint32_t w, uint32_t h) {
    if (w == 0 || h == 0) return;
    W = w; H = h;
    cam.aspect = (float)W / (float)H;
    device->Swapchain().Recreate(W, H);
    // The offscreen scene RT keeps its initial size (the post pass blits it to the resized drawable);
    // a full RT-resize is a follow-up. The window still presents correctly.
}

// ===================================================================================================
// AppKit glue: an NSWindow with a layer-backed CAMetalLayer content view, an NSEvent->InputState
// pump, and a CVDisplayLink-free tight runloop driving Viewport::RenderFrame each tick.
// ===================================================================================================

// Map AppKit virtual key codes (kVK_*) to our backend-agnostic runtime::Key. ANSI layout codes.
static runtime::Key MapKeyCode(unsigned short kc) {
    using K = runtime::Key;
    switch (kc) {
        case 0x0D: return K::W;
        case 0x00: return K::A;
        case 0x01: return K::S;
        case 0x02: return K::D;
        case 0x0C: return K::Q;
        case 0x0E: return K::E;
        case 0x31: return K::Space;  // spacebar
        case 0x23: return K::P;
        case 0x1F: return K::O;
        case 0x05: return K::G;
        case 0x0F: return K::R;
        case 0x11: return K::T;
        case 0x35: return K::Esc;
        default:   return K::Count;
    }
}

@interface HFMetalView : NSView
@end
@implementation HFMetalView
- (BOOL)wantsUpdateLayer { return YES; }
- (CALayer*)makeBackingLayer { return [CAMetalLayer layer]; }
- (BOOL)acceptsFirstResponder { return YES; }
@end

@interface HFAppDelegate : NSObject <NSWindowDelegate> {
@public
    Viewport* viewport;
    HFMetalView* view;
    runtime::InputState input;
    bool rightDragging;
    bool quit;
    double lastTime;
}
@end

@implementation HFAppDelegate

- (void)applyEvent:(NSEvent*)e {
    switch ([e type]) {
        case NSEventTypeKeyDown: {
            runtime::Key k = MapKeyCode([e keyCode]);
            if (k == runtime::Key::Esc) { quit = true; break; }
            if (k != runtime::Key::Count) input.keyDown[(int)k] = true;
            break;
        }
        case NSEventTypeKeyUp: {
            runtime::Key k = MapKeyCode([e keyCode]);
            if (k != runtime::Key::Count) input.keyDown[(int)k] = false;
            break;
        }
        case NSEventTypeFlagsChanged: {
            NSEventModifierFlags f = [e modifierFlags];
            input.keyDown[(int)runtime::Key::Shift] = (f & NSEventModifierFlagShift) != 0;
            input.keyDown[(int)runtime::Key::Ctrl]  = (f & NSEventModifierFlagControl) != 0;
            break;
        }
        case NSEventTypeRightMouseDown: rightDragging = true;  input.relativeMouse = true;  break;
        case NSEventTypeRightMouseUp:   rightDragging = false; input.relativeMouse = false; break;
        case NSEventTypeRightMouseDragged:
            // Mouse-look deltas (accumulated for this frame). AppKit deltaY is +down already-ish; the
            // fly controller treats +dy as look-down, matching the SDL HAL.
            input.mouseDx += (float)[e deltaX];
            input.mouseDy += (float)[e deltaY];
            break;
        case NSEventTypeLeftMouseDown:  input.mouseButtons[0] = true;  break;
        case NSEventTypeLeftMouseUp:    input.mouseButtons[0] = false; break;
        case NSEventTypeScrollWheel:    input.wheel += (float)[e scrollingDeltaY] * 0.1f; break;
        default: break;
    }
}

- (void)windowWillClose:(NSNotification*)n { quit = true; [NSApp terminate:nil]; }

- (void)windowDidResize:(NSNotification*)n {
    NSRect b = [view bounds];
    CGFloat s = [[view window] backingScaleFactor];
    uint32_t w = (uint32_t)(b.size.width * s), h = (uint32_t)(b.size.height * s);
    CAMetalLayer* layer = (CAMetalLayer*)[view layer];
    layer.drawableSize = CGSizeMake(w, h);
    if (viewport) viewport->Resize(w, h);
}

// One render tick: drain pending events into `input`, compute dt, render+present, reset per-frame
// deltas. Scheduled repeatedly off the main runloop.
- (void)tick {
    if (quit) return;
    NSEvent* e;
    while ((e = [NSApp nextEventMatchingMask:NSEventMaskAny
                                   untilDate:[NSDate distantPast]
                                      inMode:NSDefaultRunLoopMode
                                     dequeue:YES])) {
        [self applyEvent:e];
        [NSApp sendEvent:e];
    }

    double now = CACurrentMediaTime();
    float dt = (lastTime > 0.0) ? (float)(now - lastTime) : (1.0f / 60.0f);
    lastTime = now;
    if (dt > 0.1f) dt = 0.1f;  // clamp huge first-frame / stall dt

    @try {
        viewport->RenderFrame(input, dt);
    } @catch (...) {
        std::fprintf(stderr, "render exception\n"); quit = true;
    }

    // Per-frame deltas reset (level key/button state persists).
    input.mouseDx = 0.0f; input.mouseDy = 0.0f; input.wheel = 0.0f;

    if (quit) { [NSApp terminate:nil]; return; }
    [self performSelector:@selector(tick) withObject:nil afterDelay:0.0];
}

@end

int main(int /*argc*/, const char** /*argv*/) {
    // Headless operability (no-op on Apple/clang; present for call-site uniformity across exes).
    hf::platform::DisableCrashDialogs();
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        const uint32_t W = 1280, H = 720;
        NSRect frame = NSMakeRect(0, 0, W, H);
        NSWindow* window = [[NSWindow alloc]
            initWithContentRect:frame
                      styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                 NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable)
                        backing:NSBackingStoreBuffered
                          defer:NO];
        [window setTitle:@"Hazard Forge — Metal (--fly)"];

        HFMetalView* view = [[HFMetalView alloc] initWithFrame:frame];
        [view setWantsLayer:YES];
        [window setContentView:view];
        [window makeFirstResponder:view];

        CAMetalLayer* layer = (CAMetalLayer*)[view layer];
        CGFloat scale = [window backingScaleFactor];
        uint32_t fbW = (uint32_t)(W * scale), fbH = (uint32_t)(H * scale);
        layer.contentsScale = scale;
        layer.drawableSize = CGSizeMake(fbW, fbH);

        Viewport viewport;
        viewport.W = fbW; viewport.H = fbH;
        try {
            viewport.Init((__bridge void*)layer);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "init failed: %s\n", ex.what());
            return 1;
        }

        HFAppDelegate* del = [[HFAppDelegate alloc] init];
        del->viewport = &viewport;
        del->view = view;
        del->rightDragging = false;
        del->quit = false;
        del->lastTime = 0.0;
        [window setDelegate:del];

        [window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

        std::printf("--fly (Metal/Cocoa): WASD move, RIGHT-drag look, Space/E up, Ctrl/Q down, "
                    "Shift sprint, wheel speed, ESC quit\n");

        [del performSelector:@selector(tick) withObject:nil afterDelay:0.0];
        [NSApp run];
    }
    return 0;
}
