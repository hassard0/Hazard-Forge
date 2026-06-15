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
#include "physics/world.h"
#include "physics/body.h"
#include "render/render_graph.h"
#include "render/csm.h"
#include "debug/debug_draw.h"
#include "debug/debug_emitters.h"
#include "runtime/camera.h"
#include "runtime/clock.h"
#include "runtime/fly_camera_controller.h"
#include "runtime/input_state.h"
#include "runtime/play_state.h"
#include "editor/picking.h"
#include "editor/gizmo.h"

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
#include <cstdlib>
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
    const char* blendShotPath = nullptr;
    const char* pbrShotPath = nullptr;
    const char* sceneShotPath = nullptr;
    const char* iblShotPath = nullptr;
    const char* bloomShotPath = nullptr;
    const char* instancedShotPath = nullptr;
    const char* physicsShotPath = nullptr;
    const char* debugShotPath = nullptr;
    const char* transparencyShotPath = nullptr;
    const char* ssaoShotPath = nullptr;
    const char* ssaoShotOffPath = nullptr;
    const char* ssaoDebugPath = nullptr;
    const char* capstoneShotPath = nullptr;
    const char* csmShotPath = nullptr;       // --csm-shot <out.bmp> (Slice AD: cascaded shadows)
    const char* commandsPath = nullptr;
    // Slice AA (interactive runtime): scripted-pose headless capture + live fly viewport.
    const char* cameraShotPath = nullptr;   // --camera-shot <yaw,pitch,x,y,z> <out.bmp>
    const char* cameraShotPose = nullptr;   // the "<yaw,pitch,x,y,z>" arg
    bool fly = false;                       // --fly: open the window and run the live fly loop
    bool flyDryRun = false;                 // --fly-dry-run: exercise the loop headlessly, then exit
    bool dumpScene = false;
    bool editor = false;
    // Slice AB (editor interaction): gizmo capture + headless pick test.
    const char* gizmoShotPath = nullptr;    // --gizmo-shot <objIndex> <out.bmp>
    int gizmoShotIndex = 0;                  // selected object index for --gizmo-shot
    bool pickTest = false;                   // --pick-test: headless pick demo, prints picked index
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) {
            shotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--pbr-shot") == 0 && i + 1 < argc) {
            // Render one frame of the full-PBR showcase (ground + skybox + DamagedHelmet with the
            // full glTF metallic-roughness material set, lit + shadowed), write a BMP, exit.
            pbrShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--scene-shot") == 0 && i + 1 < argc) {
            // Slice V: render one frame of the full glTF scene-graph import showcase (ground + skybox
            // + the CesiumMilkTruck imported as a node hierarchy: body + two wheel sets, each
            // primitive with its own deduped PBR material), lit + shadowed, write a BMP, exit.
            sceneShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--ibl-shot") == 0 && i + 1 < argc) {
            // Render one frame of the HDR-IBL showcase (HDR equirect skybox + DamagedHelmet shaded by
            // lit_pbr_ibl so the metal reflects the real captured sky/sun/terrain), write a BMP, exit.
            iblShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--bloom-shot") == 0 && i + 1 < argc) {
            // Slice U: render the HDR-IBL helmet showcase into an HDR (RGBA16F) render target, run a
            // threshold->downsample->upsample bloom chain on half-res HDR mips, then composite the
            // bloom + tonemap to the swapchain. The HDR sun and the helmet's emissive gauge bloom.
            bloomShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--skinning-shot") == 0 && i + 1 < argc) {
            // Render one frame of the skinned-Fox showcase (ground + skybox + GPU-skinned Fox at
            // animation "Survey" t=0.5s, lit + shadowed), write a BMP, exit.
            skinningShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--blend-shot") == 0 && i + 1 < argc) {
            // Slice X: render one frame of the animation-BLENDING showcase (same scene/camera/light
            // as --skinning-shot, but the joint palette is a 50/50 cross-clip blend of "Walk" t=0.3s
            // and "Run" t=0.2s via anim::BlendAnimations), lit + shadowed, write a BMP, exit.
            blendShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--instanced-shot") == 0 && i + 1 < argc) {
            // Render one frame of the GPU-instanced showcase (ground + skybox + a 12x12 field of
            // instanced spheres drawn in ONE DrawIndexedInstanced, lit + shadowed), write a BMP, exit.
            instancedShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--physics-shot") == 0 && i + 1 < argc) {
            // Slice S: build a physics::World (ground plane + a dropped pyramid of spheres), step it a
            // fixed number of times until it settles, then render the RESTING bodies via the existing
            // instanced pipeline (one instance transform per body), lit + shadowed. One BMP -> exit.
            physicsShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--debug-shot") == 0 && i + 1 < argc) {
            // Slice W: the SAME settled physics sphere-pyramid scene (lit + shadowed over the
            // checkerboard ground + sky) PLUS an immediate-mode DEBUG OVERLAY drawn via DebugDraw —
            // a ground grid, a colored wireframe AABB hugging each settled body, per-body wire
            // spheres, a light-direction arrow, and physics contact markers — rendered through the
            // new LINE_LIST debug pipeline, depth-tested (occluded by opaque geometry) but not
            // depth-writing. One BMP -> exit. New golden; existing pipelines/shaders untouched.
            debugShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--transparency-shot") == 0 && i + 1 < argc) {
            // Slice T: opaque scene (checkerboard ground + skybox + a couple of opaque lit objects)
            // plus a handful of overlapping tinted GLASS objects rendered in a sorted, alpha-blended
            // translucent pass (depth-test, depth-WRITE off), back-to-front. One BMP -> exit.
            transparencyShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--ssao-shot") == 0 && i + 1 < argc) {
            // Slice Y: the settled physics sphere-pyramid scene (lit + shadowed over the checkerboard
            // ground + sky) WITH classic SSAO applied — a view-space normal+linear-depth g-buffer
            // prepass, a 16-sample hemisphere-kernel AO pass, a box blur, and a composite that darkens
            // the lit scene by the blurred AO so the sphere-sphere and sphere-ground contact crevices
            // darken. One BMP -> exit. New golden; existing pipelines/shaders untouched.
            ssaoShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--ssao-shot-off") == 0 && i + 1 < argc) {
            // Slice Y comparison: render the IDENTICAL SSAO scene but with the AO term forced off
            // (aoStrength = 0) through the very same composite pipeline, so the only difference from
            // --ssao-shot is the AO multiply. Lets a reviewer compare contact darkening on vs off.
            ssaoShotOffPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--ssao-debug") == 0 && i + 1 < argc) {
            ssaoDebugPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--capstone-shot") == 0 && i + 1 < argc) {
            // Slice Z: the integrated capstone showcase. ONE composed, deterministic frame that
            // combines the HDR-IBL environment (sky_hdr + lit_pbr_ibl), the imported CesiumMilkTruck
            // (LoadGltfScene), the GPU-skinned Fox at a fixed Walk+Run blend (BlendAnimations), the
            // full-PBR DamagedHelmet on a pedestal, a settled physics sphere pyramid (instanced
            // pipeline), and a translucent glass panel (transparent pipeline, sorted) — all lit +
            // directionally shadowed, rendered into an HDR RGBA16F target and finished with the bloom
            // chain. One BMP -> exit. New golden; existing pipelines/shaders/showcases untouched.
            capstoneShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--csm-shot") == 0 && i + 1 < argc) {
            // Slice AD: cascaded-shadow-map showcase. A long ground plane receding from the camera
            // with shadow-casters (cubes/spheres/truck) at near/mid/far and a grazing directional
            // light (long shadows). Renders a 4-cascade shadow ATLAS (2x2 tiles in one 4096 map),
            // then the scene with lit_csm (per-fragment cascade selection + atlas PCF). One BMP ->
            // exit. New golden; the single-shadow path/shaders/goldens are untouched.
            csmShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--commands") == 0 && i + 1 < argc) {
            commandsPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--camera-shot") == 0 && i + 2 < argc) {
            // Slice AA: HEADLESS scripted-camera capture. Build the interactive (capstone-equivalent)
            // scene, set a runtime::Camera to the pose "<yaw,pitch,x,y,z>", render ONE frame from
            // that pose, capture to BMP, exit. Verifies the Camera->viewProj->render path by golden.
            cameraShotPose = argv[i + 1];
            cameraShotPath = argv[i + 2];
        } else if (std::strcmp(argv[i], "--fly") == 0) {
            // Slice AA: open the window and run the LIVE fly viewport — the same capstone-equivalent
            // scene, navigated with a runtime::Camera driven by WASD + mouse-look, fixed-timestep
            // animation/physics advance, rendered straight to the swapchain. ESC quits.
            fly = true;
        } else if (std::strcmp(argv[i], "--fly-dry-run") == 0) {
            // Slice AA: headless CI exercise of the fly loop — feed a few synthetic InputState frames
            // through one loop iteration each (no window present), then exit 0. Confirms the loop
            // logic + camera/clock integration without a GUI.
            fly = true;
            flyDryRun = true;
        } else if (std::strcmp(argv[i], "--dump-scene") == 0) {
            dumpScene = true;
        } else if (std::strcmp(argv[i], "--editor") == 0) {
            // Overlay the Dear ImGui editor (hierarchy/inspector/stats) on the viewport, rendered
            // through the engine RHI. Works in interactive mode and in the --shot capture.
            editor = true;
        } else if (std::strcmp(argv[i], "--gizmo-shot") == 0 && i + 2 < argc) {
            // Slice AB: select object <objIndex> in a small deterministic scene, render it + the
            // selected object's translate gizmo (axis arrows) through the debug-line layer, capture.
            gizmoShotIndex = std::atoi(argv[i + 1]);
            gizmoShotPath = argv[i + 2];
        } else if (std::strcmp(argv[i], "--pick-test") == 0) {
            // Slice AB: HEADLESS pick demo — cast a scripted screen ray and print the picked index.
            pickTest = true;
        }
    }

    // --pick-test: fully headless (no window/GPU). Build the same deterministic multi-object scene
    // the gizmo-shot uses, cast a scripted screen-center ray through a fixed camera, and print which
    // object index is selected (demonstrating scriptable selection for agents; also unit-tested).
    if (pickTest) {
        using math::Vec3;
        // World AABBs for the gizmo-shot scene's objects (see BuildGizmoScene below for the matching
        // placement): 0 = ground plane, 1 = cube at (-2,0.5,0), 2 = sphere at (1.5,1,1.5),
        // 3 = tall box at (3,1.5,-1).
        std::vector<editor::PickAabb> objs = {
            {{{-7.0f, -0.05f, -7.0f}, {7.0f, 0.05f, 7.0f}}},   // 0 ground
            {{{-2.5f, 0.0f, -0.5f},  {-1.5f, 1.0f, 0.5f}}},     // 1 cube
            {{{0.7f, 0.2f, 0.7f},    {2.3f, 1.8f, 2.3f}}},      // 2 sphere
            {{{2.7f, 0.0f, -1.3f},   {3.3f, 3.0f, -0.7f}}},     // 3 tall box
        };
        // Fixed camera looking at the cube/sphere cluster.
        runtime::Camera cam;
        cam.position = {2.0f, 4.5f, 9.0f};
        cam.yaw = 0.15f;
        cam.SetPitch(-0.32f);
        cam.aspect = 1280.0f / 720.0f;
        // Aim the ray at the sphere (object 2): project its center to NDC, then cast back through it.
        Vec3 sphereCenter{1.5f, 1.0f, 1.5f};
        float wclip = 0.0f;
        Vec3 ndc = math::MulPointDivide(cam.ViewProj(), sphereCenter, wclip);
        math::Ray ray = editor::ScreenRayThroughCamera(cam, ndc.x, ndc.y);
        editor::PickResult r = editor::PickNearest(ray, objs);
        std::printf("pick-test: screen ndc=(%.3f,%.3f) -> picked object index %d (t=%.3f)\n",
                    ndc.x, ndc.y, r.index, r.t);
        const char* names[] = {"ground", "cube", "sphere", "tallbox"};
        if (r.index >= 0 && r.index < 4)
            std::printf("pick-test: that is the '%s'\n", names[r.index]);
        // The scripted ray aims at the sphere (object 2); assert that is what we pick.
        bool ok = (r.index == 2);
        std::printf("pick-test: %s (expected object 2 = sphere)\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    try {
        hal::Window window({"Hazard Forge — Shadows", 1280, 720});
        auto device = rhi::CreateDevice(rhi::Backend::Vulkan, window);

        // --- Cascaded shadow maps showcase (--csm-shot, Slice AD). A long ground plane receding from
        // the camera with shadow-casting objects (cubes + a sphere + the truck) at NEAR / MID / FAR
        // distances, lit by a grazing directional light so the shadows are long. Four cascades fit the
        // camera frustum's depth slices and render into a single 4096x4096 shadow ATLAS (2x2 tiles of
        // 2048). The scene is shaded by lit_csm, which picks a cascade per fragment from its view-space
        // depth, samples that cascade's atlas tile with 3x3 PCF, so shadows stay crisp from near to far.
        // One BMP -> exit. The single-shadow path/shaders/goldens are entirely untouched. ------------
        if (csmShotPath) {
            using math::Mat4; using math::Vec3;
            namespace csm = hf::render::csm;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // --- CSM FrameData layout (matches shaders/lit_csm.frag + shadow_csm.vert). ---
            struct CsmFrameData {
                float viewProj[16];     //   0
                float lightDir[4];      //  64
                float lightColor[4];    //  80
                float viewPos[4];       //  96
                float csmSplits[4];     // 112
                float cascadeVP[4][16]; // 128  (256B) -> ends 384
                float camFwd[4];        // 384
                float camRight[4];      // 400
                float camUp[4];         // 416
                float skyParams[4];     // 432
                float csmAtlas[4];      // 448  x=tilesPerRow, y=tileUVScale, z=numCascades
            };
            static_assert(sizeof(CsmFrameData) == 464, "CSM FrameData layout");

            // === Cascade config. ===
            const int   kCascades   = 4;
            const uint32_t kAtlas   = 4096;
            const int   kTilesPerRow = 2;                 // 2x2 grid
            const uint32_t kTile    = kAtlas / kTilesPerRow;  // 2048 px per cascade
            const float kSplitLambda = 0.5f;
            const float kShadowNear = 0.5f;               // CSM range near (matches camera near)
            const float kShadowFar  = 60.0f;              // CSM range far (covers the receding ground)
            const float kZPadNear   = 12.0f;              // pull each cascade's near plane back

            // === Shaders. ===
            auto litVsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto csmFsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_csm.frag.hlsl.spv");
            auto csmShWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow_csm.vert.hlsl.spv");
            auto shadowFsW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto litVs   = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto csmFs   = device->CreateShaderModule({std::span<const uint32_t>(csmFsWords)});
            auto csmShVs = device->CreateShaderModule({std::span<const uint32_t>(csmShWords)});
            auto shadowFs= device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});

            const rhi::Format kSwap = device->Swapchain().ColorFormat();

            // Lit-CSM scene pipeline (lit.vert + lit_csm.frag), straight to the swapchain (LDR).
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = csmFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = kSwap;
            litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;  // model mat4 + material(metallic,rough)
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            // CSM shadow caster: depth-only, push constant = cascadeVP mat4 + model mat4 (128B).
            rhi::GraphicsPipelineDesc shDesc;
            shDesc.vertex = csmShVs.get(); shDesc.fragment = shadowFs.get();
            shDesc.vertexLayout = scene::MeshVertexLayout();
            shDesc.depthTest = true; shDesc.depthOnly = true; shDesc.usesFrameUniforms = true;
            shDesc.pushConstantSize = sizeof(float) * 32;   // cascadeVP(16) + model(16)
            auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

            // === Shadow atlas + meshes + textures. ===
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

            // === Scene layout. Camera looks down the -Z corridor; the ground recedes into -Z. ===
            // Casters at NEAR (+Z, close), MID, FAR (-Z, distant) so each cascade has work to shadow.
            const Mat4 groundModel = Mat4::Scale({60.0f, 1.0f, 80.0f});  // big receding floor
            struct Caster { Mat4 model; const scene::Mesh* mesh; float metallic; float rough; float r,g,b; };
            std::vector<Caster> casters;
            // Cube + sphere meshes have half-extent 0.5, so a primitive scaled by s has half-height
            // 0.5*s; placing the center at y = 0.5*s seats it exactly ON the ground (no peter-panning).
            auto box = [&](float x, float z, float s, float rot, float r, float g, float b) {
                Caster c; c.model = Mat4::Translate({x, 0.5f * s, z}) * Mat4::RotateY(rot)
                                  * Mat4::Scale({s, s, s});
                c.mesh = &cube; c.metallic = 0.0f; c.rough = 0.8f; c.r=r; c.g=g; c.b=b;
                casters.push_back(c);
            };
            auto ball = [&](float x, float z, float s, float r, float g, float b) {
                Caster c; c.model = Mat4::Translate({x, 0.5f * s, z}) * Mat4::Scale({s, s, s});
                c.mesh = &sphere; c.metallic = 0.1f; c.rough = 0.4f; c.r=r; c.g=g; c.b=b;
                casters.push_back(c);
            };
            // NEAR band (close to camera, +Z).
            box(-3.5f,  9.0f, 2.0f,  0.4f, 0.85f, 0.35f, 0.30f);
            ball( 3.5f,  8.0f, 2.4f,        0.30f, 0.55f, 0.85f);
            // MID band.
            box( 0.0f, -2.0f, 2.6f, -0.6f, 0.30f, 0.80f, 0.45f);
            box(-5.5f, -4.0f, 2.0f,  0.9f, 0.80f, 0.75f, 0.30f);
            ball( 5.5f, -5.0f, 2.4f,        0.85f, 0.45f, 0.45f);
            // FAR band — taller so the long shadow reads clearly in the far cascade.
            box( 4.0f, -15.0f, 4.0f, 0.2f, 0.78f, 0.40f, 0.78f);
            box(-4.5f, -18.0f, 4.6f, 0.5f, 0.55f, 0.62f, 0.85f);
            box( 0.5f, -28.0f, 5.0f, 0.5f, 0.65f, 0.65f, 0.40f);

            // === Fixed deterministic camera + grazing directional light. ===
            const Vec3 eye{0.0f, 5.0f, 17.0f};
            const Vec3 center{0.0f, 1.2f, -10.0f};
            const float fovY = 1.04719755f;  // 60deg
            Vec3 fwd   = math::normalize(center - eye);
            Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
            Vec3 up3   = math::cross(right, fwd);
            const float tanHalf = std::tan(0.5f * fovY);
            // Grazing light from the upper-left -> long shadows raking across the corridor.
            Vec3 lightDir = math::normalize(Vec3{-0.65f, -0.5f, -0.35f});

            CsmFrameData fd{};
            {
                Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
                Mat4 proj = Mat4::Perspective(fovY, aspect, 0.1f, 100.0f);
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

            // === Build the N cascades: practical-split distances + per-cascade ortho fit. ===
            auto splits = csm::CsmSplits(kShadowNear, kShadowFar, kCascades, kSplitLambda);
            for (int c = 0; c < kCascades; ++c) fd.csmSplits[c] = splits[c];
            Mat4 cascadeVP[4];
            {
                float sliceNear = kShadowNear;
                for (int c = 0; c < kCascades; ++c) {
                    float sliceFar = splits[c];
                    auto corners = csm::FrustumSliceCornersWorld(eye, fwd, right, up3, tanHalf,
                                                                 aspect, sliceNear, sliceFar);
                    auto fit = csm::FitCascadeLightMatrix(corners, lightDir, kZPadNear);
                    cascadeVP[c] = fit.lightViewProj;
                    for (int k = 0; k < 16; ++k) fd.cascadeVP[c][k] = fit.lightViewProj.m[k];
                    sliceNear = sliceFar;
                }
            }

            render::RenderGraph graph;
            render::RgResource rgShadow = graph.ImportTarget(
                "csmAtlas", render::RgResourceKind::ShadowMap, *shadowAtlas);
            render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

            // --- Shadow pass: clear the whole atlas once, then for each cascade SetViewport(tile) and
            // draw every caster with that cascade's lightViewProj (pushed as a constant). ---
            graph.AddPass("csmShadow", {}, {rgShadow},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(CsmFrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*shadowPipeline);
                    for (int c = 0; c < kCascades; ++c) {
                        int col = c % kTilesPerRow, row = c / kTilesPerRow;
                        cmd.SetViewport((int32_t)(col * kTile), (int32_t)(row * kTile), kTile, kTile);
                        // Ground.
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

            // --- Scene pass straight to the swapchain: ground + casters, shaded by lit_csm. ---
            graph.AddPass("csmScene", {rgShadow}, {rgSwap},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(CsmFrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0.55f, 0.68f, 0.82f, 1});  // sky-ish
                    cmd.BindPipeline(*litPipeline);
                    // Ground.
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
                    // Casters (flat-colored via vertex color * white texture; use groundTex tinted? -
                    // simpler: bind a 1x1 white tex would be cleaner, but groundTex*color reads fine).
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

            device->CaptureNextFrame();
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(csmShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — CSM: %d cascades, %ux%u atlas (%dx%d tiles), "
                                    "%zu casters\n", csmShotPath, cw, ch2, kCascades, kAtlas, kAtlas,
                                    kTilesPerRow, kTilesPerRow, casters.size());
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", csmShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels (CSM)\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- Integrated CAPSTONE showcase (--capstone-shot, Slice Z): ONE composed deterministic
        // frame that orchestrates every per-feature showcase into a single coherent scene. The HDR
        // equirect environment (sky_hdr) is the background AND the IBL source; on the ground under it
        // sit the imported CesiumMilkTruck (LoadGltfScene, lit_pbr), the GPU-skinned Fox at a fixed
        // Walk+Run blend (lit_skinned + BlendAnimations), the full-PBR DamagedHelmet reflecting the
        // environment (lit_pbr_ibl) on a pedestal cube, a settled physics sphere pyramid (instanced
        // lit pipeline), and — IN FRONT — a couple of sorted translucent glass spheres (transparent
        // pipeline, depthWrite off). All opaque geometry casts into a single directional shadow map
        // (static + skinned + instanced shadow pipelines). The whole scene renders into an HDR
        // RGBA16F target which is finished by the existing bloom chain (prefilter -> down -> up ->
        // composite/tonemap). This is the INTEGRATION TEST: 7 distinct opaque pipelines + a
        // transparent pipeline with different descriptor sets (frame UBO, 2-tex material, 5-tex PBR
        // set, environment, joint palette, instance buffer) must coexist in ONE scene render pass
        // without state leaking between consecutive draws. One BMP -> exit. Reuses ALL existing
        // pipelines/shaders/loaders; nothing existing is touched. -------------------------------------
        if (capstoneShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
            const rhi::Format kHdr = rhi::Format::RGBA16_Float;

            // === HDR environment (sky + IBL). ===
            hf::asset::EnvironmentMap env = hf::asset::LoadHdrEnvironment(*device, HF_ENV_PATH);
            const float envMaxLod = (float)(env.mipLevels - 1);

            // === Shaders shared across pipelines. ===
            auto litVsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto litFsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto pbrFsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_pbr.frag.hlsl.spv");
            auto iblFsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_pbr_ibl.frag.hlsl.spv");
            auto skVsWords   = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_skinned.vert.hlsl.spv");
            auto instVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_instanced.vert.hlsl.spv");
            auto litVs  = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto litFs  = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            auto pbrFs  = device->CreateShaderModule({std::span<const uint32_t>(pbrFsWords)});
            auto iblFs  = device->CreateShaderModule({std::span<const uint32_t>(iblFsWords)});
            auto skVs   = device->CreateShaderModule({std::span<const uint32_t>(skVsWords)});
            auto instVs = device->CreateShaderModule({std::span<const uint32_t>(instVsWords)});

            // Static lit (ground + pedestal): writes the HDR target.
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = kHdr;
            litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            // Full-PBR (truck): 5-texture material set.
            rhi::GraphicsPipelineDesc pbrDesc;
            pbrDesc.vertex = litVs.get(); pbrDesc.fragment = pbrFs.get();
            pbrDesc.vertexLayout = scene::MeshVertexLayout();
            pbrDesc.colorFormat = kHdr;
            pbrDesc.depthTest = true; pbrDesc.usesFrameUniforms = true; pbrDesc.usesTexture = true;
            pbrDesc.pbrMaterial = true;
            pbrDesc.pushConstantSize = sizeof(float) * 20;
            auto pbrPipeline = device->CreateGraphicsPipeline(pbrDesc);

            // Full-PBR + IBL (helmet): 5-texture material set + environment.
            rhi::GraphicsPipelineDesc iblDesc;
            iblDesc.vertex = litVs.get(); iblDesc.fragment = iblFs.get();
            iblDesc.vertexLayout = scene::MeshVertexLayout();
            iblDesc.colorFormat = kHdr;
            iblDesc.depthTest = true; iblDesc.usesFrameUniforms = true; iblDesc.usesTexture = true;
            iblDesc.pbrMaterial = true; iblDesc.usesEnvironment = true;
            iblDesc.pushConstantSize = sizeof(float) * 20;
            auto iblPipeline = device->CreateGraphicsPipeline(iblDesc);

            // Skinned lit (fox): joint palette (set 2).
            rhi::GraphicsPipelineDesc skDesc;
            skDesc.vertex = skVs.get(); skDesc.fragment = litFs.get();
            skDesc.vertexLayout = scene::SkinnedMeshVertexLayout();
            skDesc.colorFormat = kHdr;
            skDesc.depthTest = true; skDesc.usesFrameUniforms = true; skDesc.usesTexture = true;
            skDesc.usesJointPalette = true;
            skDesc.pushConstantSize = sizeof(float) * 20;
            auto skinnedPipeline = device->CreateGraphicsPipeline(skDesc);

            // Instanced lit (physics spheres): per-instance transform stream.
            rhi::GraphicsPipelineDesc instDesc;
            instDesc.vertex = instVs.get(); instDesc.fragment = litFs.get();
            instDesc.vertexLayout = scene::MeshVertexLayout();
            instDesc.instanceLayout = scene::InstanceTransformLayout();
            instDesc.colorFormat = kHdr;
            instDesc.depthTest = true; instDesc.usesFrameUniforms = true; instDesc.usesTexture = true;
            instDesc.pushConstantSize = sizeof(float) * 4;
            auto instPipeline = device->CreateGraphicsPipeline(instDesc);

            // Transparent glass: depthTest on, depthWrite off, alpha blend, double-sided.
            auto tVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/transparent.vert.hlsl.spv");
            auto tFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/transparent.frag.hlsl.spv");
            auto tVs = device->CreateShaderModule({std::span<const uint32_t>(tVsWords)});
            auto tFs = device->CreateShaderModule({std::span<const uint32_t>(tFsWords)});
            rhi::GraphicsPipelineDesc tDesc;
            tDesc.vertex = tVs.get(); tDesc.fragment = tFs.get();
            tDesc.vertexLayout = scene::MeshVertexLayout();
            tDesc.colorFormat = kHdr;
            tDesc.depthTest = true; tDesc.depthWrite = false; tDesc.alphaBlend = true;
            tDesc.cullNone = true; tDesc.usesFrameUniforms = true; tDesc.usesTexture = false;
            tDesc.pushConstantSize = sizeof(float) * 20;
            auto transparentPipeline = device->CreateGraphicsPipeline(tDesc);

            // === Shadow pipelines: static + skinned + instanced casters share one shadow map. ===
            auto shadowFsW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto staticShW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto skShadowW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow_skinned.vert.hlsl.spv");
            auto instShW     = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow_instanced.vert.hlsl.spv");
            auto shadowFs   = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});
            auto staticShVs = device->CreateShaderModule({std::span<const uint32_t>(staticShW)});
            auto skShadowVs = device->CreateShaderModule({std::span<const uint32_t>(skShadowW)});
            auto instShVs   = device->CreateShaderModule({std::span<const uint32_t>(instShW)});

            rhi::GraphicsPipelineDesc stShDesc;
            stShDesc.vertex = staticShVs.get(); stShDesc.fragment = shadowFs.get();
            stShDesc.vertexLayout = scene::MeshVertexLayout();
            stShDesc.depthTest = true; stShDesc.depthOnly = true; stShDesc.usesFrameUniforms = true;
            stShDesc.pushConstantSize = sizeof(float) * 16;
            auto staticShadowPipeline = device->CreateGraphicsPipeline(stShDesc);

            rhi::GraphicsPipelineDesc skShDesc;
            skShDesc.vertex = skShadowVs.get(); skShDesc.fragment = shadowFs.get();
            skShDesc.vertexLayout = scene::SkinnedMeshVertexLayout();
            skShDesc.depthTest = true; skShDesc.depthOnly = true; skShDesc.usesFrameUniforms = true;
            skShDesc.usesJointPalette = true;
            skShDesc.pushConstantSize = sizeof(float) * 16;
            auto skinnedShadowPipeline = device->CreateGraphicsPipeline(skShDesc);

            rhi::GraphicsPipelineDesc instShDesc;
            instShDesc.vertex = instShVs.get(); instShDesc.fragment = shadowFs.get();
            instShDesc.vertexLayout = scene::MeshVertexLayout();
            instShDesc.instanceLayout = scene::InstanceTransformLayout();
            instShDesc.depthTest = true; instShDesc.depthOnly = true; instShDesc.usesFrameUniforms = true;
            auto instShadowPipeline = device->CreateGraphicsPipeline(instShDesc);

            // === Sky (HDR equirect). ===
            auto skyVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.vert.hlsl.spv");
            auto skyFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky_hdr.frag.hlsl.spv");
            auto skyVsM = device->CreateShaderModule({std::span<const uint32_t>(skyVsW)});
            auto skyFsM = device->CreateShaderModule({std::span<const uint32_t>(skyFsW)});
            rhi::GraphicsPipelineDesc skyD;
            skyD.vertex = skyVsM.get(); skyD.fragment = skyFsM.get();
            skyD.colorFormat = kHdr;
            skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
            skyD.usesEnvironment = true;
            auto skyPipe = device->CreateGraphicsPipeline(skyD);

            // === Bloom pipelines (fullscreen, fragment push constants). ===
            auto postVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.vert.hlsl.spv");
            auto postVsM = device->CreateShaderModule({std::span<const uint32_t>(postVsW)});
            auto loadFs = [&](const char* name) {
                auto words = LoadSpirv(std::string(HF_SHADER_DIR) + "/" + name + ".spv");
                return device->CreateShaderModule({std::span<const uint32_t>(words)});
            };
            struct BloomParams { float texel[2]; float threshold; float knee; float strength; float intensity; };
            const uint32_t kBloomPC = sizeof(BloomParams);
            auto makeBloomPipe = [&](rhi::IShaderModule* fs, rhi::Format colorFmt) {
                rhi::GraphicsPipelineDesc d;
                d.vertex = postVsM.get(); d.fragment = fs;
                d.colorFormat = colorFmt;
                d.depthTest = false; d.usesFrameUniforms = false;
                d.usesTexture = true; d.fullscreen = true;
                d.fragmentPushConstants = true; d.pushConstantSize = kBloomPC;
                return device->CreateGraphicsPipeline(d);
            };
            auto prefilterFs  = loadFs("bloom_prefilter.frag.hlsl");
            auto downsampleFs = loadFs("bloom_downsample.frag.hlsl");
            auto upsampleFs   = loadFs("bloom_upsample.frag.hlsl");
            auto compositeFs  = loadFs("bloom_composite.frag.hlsl");
            auto prefilterPipe  = makeBloomPipe(prefilterFs.get(), kHdr);
            auto downsamplePipe = makeBloomPipe(downsampleFs.get(), kHdr);
            auto upsamplePipe   = makeBloomPipe(upsampleFs.get(), kHdr);
            auto compositePipe  = makeBloomPipe(compositeFs.get(), device->Swapchain().ColorFormat());

            // === Render targets: HDR scene + 5-level half-res mip chain. ===
            auto rt = device->CreateRenderTarget(w, h, kHdr);
            auto shadowMap = device->CreateShadowMap(2048);
            device->SetShadowMap(*shadowMap);
            const int kMips = 5;
            std::vector<std::unique_ptr<rhi::IRenderTarget>> down, up;
            std::vector<uint32_t> mw(kMips), mh(kMips);
            for (int i = 0; i < kMips; ++i) {
                uint32_t dw = std::max(1u, w >> (i + 1));
                uint32_t dh = std::max(1u, h >> (i + 1));
                mw[i] = dw; mh[i] = dh;
                down.push_back(device->CreateRenderTarget(dw, dh, kHdr));
                up.push_back(device->CreateRenderTarget(dw, dh, kHdr));
            }

            // === Shared meshes + textures. ===
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

            // === Imported truck scene (node hierarchy -> instances + deduped materials). ===
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

            // === Layout transforms (place everything on the ground plane y=0). ===
            // Physics pyramid: drop it onto the ground to the LEFT-FRONT.
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

            // Truck: stand-upright fit (same recipe as --scene-shot) then place to the LEFT-BACK.
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

            // Fox: ground-aligned uniform scale (same recipe as --skinning-shot), placed RIGHT-FRONT.
            float foxH = fox.bbMax[1] - fox.bbMin[1];
            float foxScale = (foxH > 1e-4f) ? (2.2f / foxH) : 0.05f;
            float fcx = 0.5f * (fox.bbMin[0] + fox.bbMax[0]);
            float fcz = 0.5f * (fox.bbMin[2] + fox.bbMax[2]);
            const Mat4 foxModel = Mat4::Translate({3.4f, 0.0f, 1.6f})
                                * Mat4::RotateY(-0.9f)
                                * Mat4::Translate({-fcx * foxScale, -fox.bbMin[1] * foxScale, -fcz * foxScale})
                                * Mat4::Scale({foxScale, foxScale, foxScale});

            // Helmet on a pedestal cube at CENTER-BACK so its metal reflects the environment.
            const float pedH = 0.5f;                       // pedestal half-height (cube is unit)
            const Mat4 pedestalModel = Mat4::Translate({0.0f, pedH, -1.8f}) * Mat4::Scale({1.0f, pedH, 1.0f});
            const float helmetScale = 1.3f;
            const Mat4 helmetModel = Mat4::Translate({0.0f, 2.0f * pedH + helmetScale, -1.8f})
                                   * Mat4::RotateX(1.5707963f) * Mat4::Scale({helmetScale, helmetScale, helmetScale});

            // Ground plane.
            const Mat4 groundModel = Mat4::Scale({14.0f, 1.0f, 14.0f});

            // Translucent glass spheres IN FRONT (sorted back-to-front), so the scene reads through them.
            struct Glass { Vec3 pos; float scale; float r, g, b, baseAlpha; };
            std::vector<Glass> glass = {
                {{-1.4f, 1.3f, 3.8f}, 1.4f, 0.22f, 0.55f, 0.98f, 0.34f},  // blue
                {{ 1.3f, 1.2f, 4.2f}, 1.3f, 0.32f, 0.95f, 0.45f, 0.34f},  // green
            };

            // === Camera + light + sky frame data (fixed, deterministic). Wide framing so EVERY
            // element is visible together. ===
            const Vec3 eye{0.0f, 4.0f, 10.0f};
            const Vec3 center{0.0f, 1.2f, 0.2f};
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
                Vec3 sc{0.0f, 1.0f, 0.0f};
                Vec3 lightEye = sc - lightDir * 20.0f;
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
                Mat4 lightOrtho = Mat4::Ortho(-9.0f, 9.0f, -9.0f, 9.0f, 1.0f, 45.0f);
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

            // Sort glass back-to-front from the eye (deterministic for the fixed camera).
            std::sort(glass.begin(), glass.end(), [&](const Glass& a, const Glass& b) {
                return math::length(a.pos - eye) > math::length(b.pos - eye);
            });

            // === Bloom tuning (matches --bloom-shot). ===
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

            // --- Shadow pass: ALL opaque casters into one shadow map (static + skinned + instanced). ---
            graph.AddPass("shadow", {}, {rgShadow},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    dev.SetJointPalette(paletteData.data(), paletteData.size() * sizeof(float));
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    // Static casters: ground, pedestal, truck primitives, helmet.
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
                    // Skinned caster: fox.
                    cmd.BindPipeline(*skinnedShadowPipeline);
                    cmd.PushConstants(foxModel.m, sizeof(float) * 16);
                    cmd.BindVertexBuffer(fox.mesh.vertices());
                    cmd.BindIndexBuffer(fox.mesh.indices());
                    cmd.DrawIndexed(fox.mesh.indexCount());
                    // Instanced casters: physics spheres.
                    cmd.BindPipeline(*instShadowPipeline);
                    cmd.BindVertexBuffer(sphere.vertices());
                    cmd.BindInstanceBuffer(*instanceBuffer);
                    cmd.BindIndexBuffer(sphere.indices());
                    cmd.DrawIndexedInstanced(sphere.indexCount(), kInstanceCount);
                    cmd.EndRenderPass();
                });

            // --- Scene pass into the HDR target: sky -> opaque (lit/PBR/IBL/skinned/instanced) ->
            // sorted transparent glass. Re-bind every required descriptor before each pipeline's draws
            // so no descriptor-set state leaks across the 7 consecutive pipeline switches. ---
            graph.AddPass("scene", {rgShadow}, {rgScene},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    dev.SetJointPalette(paletteData.data(), paletteData.size() * sizeof(float));
                    cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1});
                    // HDR sky background.
                    cmd.BindPipeline(*skyPipe);
                    cmd.BindEnvironment(*env.equirect);
                    cmd.Draw(3);
                    // Ground + pedestal (static lit, dielectric).
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
                    // Truck (full-PBR): every imported instance with its own world + material set.
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
                    // Helmet (full-PBR + IBL): reflects the bound environment.
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
                    // Fox (skinned lit): re-bind the joint palette + 2-tex material.
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
                    // Physics spheres (instanced lit).
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
                    // Sorted translucent glass (depth-test, no depth write) over the opaque scene.
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

            // --- Bloom chain (identical to --bloom-shot). ---
            graph.AddPass("prefilter", {rgScene}, {rgDown[0]},
                [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                    BloomParams p = mkPC(w, h, kBloomStrength);
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
                    BloomParams p = mkPC(w, h, kBloomStrength);
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*compositePipe);
                    cmd.BindTexturePair(*rt, *up[0]);
                    cmd.PushConstants(&p, sizeof(p));
                    cmd.Draw(3);
                    cmd.EndRenderPass();
                });

            device->CaptureNextFrame();
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(capstoneShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — capstone: truck(%zu inst) + fox + helmet + "
                                    "%u physics spheres + %zu glass, IBL+shadows+bloom\n",
                                    capstoneShotPath, cw, ch2, truck.instances.size(),
                                    kInstanceCount, glass.size());
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", capstoneShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- INTERACTIVE RUNTIME (Slice AA): scripted-pose capture (--camera-shot) + live fly
        // viewport (--fly / --fly-dry-run). Builds a scene IDENTICAL in construction to the capstone
        // showcase (same meshes/models/physics-step-budget/fox Walk+Run blend/transforms/light/bloom),
        // but the camera is a runtime::Camera (yaw/pitch/pos) instead of the fixed hand-built capstone
        // camera. The capstone block above is left byte-for-byte untouched so its golden is unaffected;
        // this is a PARALLEL path with its own (scripted) camera and its own new golden. --camera-shot
        // sets the camera to the CLI pose and captures one frame; --fly runs the real-time loop
        // (pump input -> fixed-timestep advance -> FlyCameraController -> render -> present), ESC
        // quits; --fly-dry-run feeds synthetic input through one loop iteration headlessly. ----------
        if (cameraShotPath || fly) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
            const rhi::Format kHdr = rhi::Format::RGBA16_Float;

            // === HDR environment (sky + IBL). ===
            hf::asset::EnvironmentMap env = hf::asset::LoadHdrEnvironment(*device, HF_ENV_PATH);
            const float envMaxLod = (float)(env.mipLevels - 1);

            // === Shaders (same set as the capstone). ===
            auto litVsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto litFsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto pbrFsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_pbr.frag.hlsl.spv");
            auto iblFsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_pbr_ibl.frag.hlsl.spv");
            auto skVsWords   = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_skinned.vert.hlsl.spv");
            auto instVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_instanced.vert.hlsl.spv");
            auto litVs  = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto litFs  = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            auto pbrFs  = device->CreateShaderModule({std::span<const uint32_t>(pbrFsWords)});
            auto iblFs  = device->CreateShaderModule({std::span<const uint32_t>(iblFsWords)});
            auto skVs   = device->CreateShaderModule({std::span<const uint32_t>(skVsWords)});
            auto instVs = device->CreateShaderModule({std::span<const uint32_t>(instVsWords)});

            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = kHdr;
            litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            rhi::GraphicsPipelineDesc pbrDesc;
            pbrDesc.vertex = litVs.get(); pbrDesc.fragment = pbrFs.get();
            pbrDesc.vertexLayout = scene::MeshVertexLayout();
            pbrDesc.colorFormat = kHdr;
            pbrDesc.depthTest = true; pbrDesc.usesFrameUniforms = true; pbrDesc.usesTexture = true;
            pbrDesc.pbrMaterial = true;
            pbrDesc.pushConstantSize = sizeof(float) * 20;
            auto pbrPipeline = device->CreateGraphicsPipeline(pbrDesc);

            rhi::GraphicsPipelineDesc iblDesc;
            iblDesc.vertex = litVs.get(); iblDesc.fragment = iblFs.get();
            iblDesc.vertexLayout = scene::MeshVertexLayout();
            iblDesc.colorFormat = kHdr;
            iblDesc.depthTest = true; iblDesc.usesFrameUniforms = true; iblDesc.usesTexture = true;
            iblDesc.pbrMaterial = true; iblDesc.usesEnvironment = true;
            iblDesc.pushConstantSize = sizeof(float) * 20;
            auto iblPipeline = device->CreateGraphicsPipeline(iblDesc);

            rhi::GraphicsPipelineDesc skDesc;
            skDesc.vertex = skVs.get(); skDesc.fragment = litFs.get();
            skDesc.vertexLayout = scene::SkinnedMeshVertexLayout();
            skDesc.colorFormat = kHdr;
            skDesc.depthTest = true; skDesc.usesFrameUniforms = true; skDesc.usesTexture = true;
            skDesc.usesJointPalette = true;
            skDesc.pushConstantSize = sizeof(float) * 20;
            auto skinnedPipeline = device->CreateGraphicsPipeline(skDesc);

            rhi::GraphicsPipelineDesc instDesc;
            instDesc.vertex = instVs.get(); instDesc.fragment = litFs.get();
            instDesc.vertexLayout = scene::MeshVertexLayout();
            instDesc.instanceLayout = scene::InstanceTransformLayout();
            instDesc.colorFormat = kHdr;
            instDesc.depthTest = true; instDesc.usesFrameUniforms = true; instDesc.usesTexture = true;
            instDesc.pushConstantSize = sizeof(float) * 4;
            auto instPipeline = device->CreateGraphicsPipeline(instDesc);

            auto tVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/transparent.vert.hlsl.spv");
            auto tFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/transparent.frag.hlsl.spv");
            auto tVs = device->CreateShaderModule({std::span<const uint32_t>(tVsWords)});
            auto tFs = device->CreateShaderModule({std::span<const uint32_t>(tFsWords)});
            rhi::GraphicsPipelineDesc tDesc;
            tDesc.vertex = tVs.get(); tDesc.fragment = tFs.get();
            tDesc.vertexLayout = scene::MeshVertexLayout();
            tDesc.colorFormat = kHdr;
            tDesc.depthTest = true; tDesc.depthWrite = false; tDesc.alphaBlend = true;
            tDesc.cullNone = true; tDesc.usesFrameUniforms = true; tDesc.usesTexture = false;
            tDesc.pushConstantSize = sizeof(float) * 20;
            auto transparentPipeline = device->CreateGraphicsPipeline(tDesc);

            auto shadowFsW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto staticShW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto skShadowW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow_skinned.vert.hlsl.spv");
            auto instShW     = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow_instanced.vert.hlsl.spv");
            auto shadowFs   = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});
            auto staticShVs = device->CreateShaderModule({std::span<const uint32_t>(staticShW)});
            auto skShadowVs = device->CreateShaderModule({std::span<const uint32_t>(skShadowW)});
            auto instShVs   = device->CreateShaderModule({std::span<const uint32_t>(instShW)});

            rhi::GraphicsPipelineDesc stShDesc;
            stShDesc.vertex = staticShVs.get(); stShDesc.fragment = shadowFs.get();
            stShDesc.vertexLayout = scene::MeshVertexLayout();
            stShDesc.depthTest = true; stShDesc.depthOnly = true; stShDesc.usesFrameUniforms = true;
            stShDesc.pushConstantSize = sizeof(float) * 16;
            auto staticShadowPipeline = device->CreateGraphicsPipeline(stShDesc);

            rhi::GraphicsPipelineDesc skShDesc;
            skShDesc.vertex = skShadowVs.get(); skShDesc.fragment = shadowFs.get();
            skShDesc.vertexLayout = scene::SkinnedMeshVertexLayout();
            skShDesc.depthTest = true; skShDesc.depthOnly = true; skShDesc.usesFrameUniforms = true;
            skShDesc.usesJointPalette = true;
            skShDesc.pushConstantSize = sizeof(float) * 16;
            auto skinnedShadowPipeline = device->CreateGraphicsPipeline(skShDesc);

            rhi::GraphicsPipelineDesc instShDesc;
            instShDesc.vertex = instShVs.get(); instShDesc.fragment = shadowFs.get();
            instShDesc.vertexLayout = scene::MeshVertexLayout();
            instShDesc.instanceLayout = scene::InstanceTransformLayout();
            instShDesc.depthTest = true; instShDesc.depthOnly = true; instShDesc.usesFrameUniforms = true;
            auto instShadowPipeline = device->CreateGraphicsPipeline(instShDesc);

            auto skyVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.vert.hlsl.spv");
            auto skyFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky_hdr.frag.hlsl.spv");
            auto skyVsM = device->CreateShaderModule({std::span<const uint32_t>(skyVsW)});
            auto skyFsM = device->CreateShaderModule({std::span<const uint32_t>(skyFsW)});
            rhi::GraphicsPipelineDesc skyD;
            skyD.vertex = skyVsM.get(); skyD.fragment = skyFsM.get();
            skyD.colorFormat = kHdr;
            skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
            skyD.usesEnvironment = true;
            auto skyPipe = device->CreateGraphicsPipeline(skyD);

            auto postVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.vert.hlsl.spv");
            auto postVsM = device->CreateShaderModule({std::span<const uint32_t>(postVsW)});
            auto loadFs = [&](const char* name) {
                auto words = LoadSpirv(std::string(HF_SHADER_DIR) + "/" + name + ".spv");
                return device->CreateShaderModule({std::span<const uint32_t>(words)});
            };
            struct BloomParams { float texel[2]; float threshold; float knee; float strength; float intensity; };
            const uint32_t kBloomPC = sizeof(BloomParams);
            auto makeBloomPipe = [&](rhi::IShaderModule* fs, rhi::Format colorFmt) {
                rhi::GraphicsPipelineDesc d;
                d.vertex = postVsM.get(); d.fragment = fs;
                d.colorFormat = colorFmt;
                d.depthTest = false; d.usesFrameUniforms = false;
                d.usesTexture = true; d.fullscreen = true;
                d.fragmentPushConstants = true; d.pushConstantSize = kBloomPC;
                return device->CreateGraphicsPipeline(d);
            };
            auto prefilterFs  = loadFs("bloom_prefilter.frag.hlsl");
            auto downsampleFs = loadFs("bloom_downsample.frag.hlsl");
            auto upsampleFs   = loadFs("bloom_upsample.frag.hlsl");
            auto compositeFs  = loadFs("bloom_composite.frag.hlsl");
            auto prefilterPipe  = makeBloomPipe(prefilterFs.get(), kHdr);
            auto downsamplePipe = makeBloomPipe(downsampleFs.get(), kHdr);
            auto upsamplePipe   = makeBloomPipe(upsampleFs.get(), kHdr);
            auto compositePipe  = makeBloomPipe(compositeFs.get(), device->Swapchain().ColorFormat());

            // === Render targets (recreated on resize in the fly loop). ===
            auto rt = device->CreateRenderTarget(w, h, kHdr);
            auto shadowMap = device->CreateShadowMap(2048);
            device->SetShadowMap(*shadowMap);
            const int kMips = 5;
            std::vector<std::unique_ptr<rhi::IRenderTarget>> down, up;
            std::vector<uint32_t> mw(kMips), mh(kMips);
            auto buildMips = [&](uint32_t fw, uint32_t fh) {
                down.clear(); up.clear();
                for (int i = 0; i < kMips; ++i) {
                    uint32_t dw = std::max(1u, fw >> (i + 1));
                    uint32_t dh = std::max(1u, fh >> (i + 1));
                    mw[i] = dw; mh[i] = dh;
                    down.push_back(device->CreateRenderTarget(dw, dh, kHdr));
                    up.push_back(device->CreateRenderTarget(dw, dh, kHdr));
                }
            };
            buildMips(w, h);

            // === Shared meshes + textures (same as capstone). ===
            std::vector<uint8_t> checker = MakeCheckerboard();
            auto groundTex = device->CreateTexture(
                {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
            const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
            auto flatNormal = device->CreateTexture(
                {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
            scene::Mesh plane = scene::Mesh::Plane(*device);
            scene::Mesh cube = scene::Mesh::Cube(*device);
            scene::Mesh sphere = scene::Mesh::Sphere(*device);

            hf::asset::PbrModel helmet = hf::asset::LoadPbrGltfModel(*device, HF_HELMET_MODEL_PATH);
            hf::asset::GltfScene truck = hf::asset::LoadGltfScene(*device, HF_TRUCK_MODEL_PATH);
            hf::asset::SkinnedModel fox = hf::asset::LoadSkinnedGltfModel(*device, HF_FOX_MODEL_PATH);

            // Fox Walk+Run blended palette (same fixed times as capstone -> deterministic capture).
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

            // Settled physics sphere pyramid (same step budget as capstone).
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
            for (int s = 0; s < 240; ++s) world.Step(1.0f / 120.0f);

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

            // Truck placement (same recipe as capstone).
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

            // Bloom tuning (matches capstone / --bloom-shot).
            const float kExposure = 1.7f, kThreshold = 1.0f, kKnee = 0.6f;
            const float kUpStrength = 1.0f, kBloomStrength = 0.14f;
            auto mkPC = [&](uint32_t tw, uint32_t th, float strength) {
                BloomParams p{}; p.texel[0] = 1.0f / (float)tw; p.texel[1] = 1.0f / (float)th;
                p.threshold = kThreshold; p.knee = kKnee; p.strength = strength; p.intensity = kExposure;
                return p;
            };

            // === Fixed directional light (same as capstone). The CAMERA is the only thing the new
            // runtime drives; the light/shadow framing stays put so the scene is consistently lit. ===
            const Vec3 lightDirRaw{-0.5f, -1.0f, -0.3f};
            FrameData lightTemplate{};
            {
                Vec3 lightDir = math::normalize(lightDirRaw);
                Vec3 sc{0.0f, 1.0f, 0.0f};
                Vec3 lightEye = sc - lightDir * 20.0f;
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
                Mat4 lightOrtho = Mat4::Ortho(-9.0f, 9.0f, -9.0f, 9.0f, 1.0f, 45.0f);
                Mat4 lightVP = lightOrtho * lightView;
                for (int k = 0; k < 16; ++k) lightTemplate.lightViewProj[k] = lightVP.m[k];
                lightTemplate.lightDir[0] = lightDirRaw.x; lightTemplate.lightDir[1] = lightDirRaw.y;
                lightTemplate.lightDir[2] = lightDirRaw.z;
                lightTemplate.lightColor[0] = 1.0f; lightTemplate.lightColor[1] = 0.97f;
                lightTemplate.lightColor[2] = 0.9f; lightTemplate.lightColor[3] = 1.0f;
                lightTemplate.ptCount[0] = 0.0f;
            }

            // Fill a FrameData from the runtime::Camera (viewProj + basis + sky params), keeping the
            // fixed light fields. This is the centralized Camera->FrameData path the showcases used to
            // duplicate by hand.
            auto makeFrameData = [&](const runtime::Camera& cam) {
                FrameData fd = lightTemplate;
                Mat4 vp = cam.ViewProj();
                for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
                runtime::CameraBasis b = cam.Basis();
                fd.viewPos[0] = b.position.x; fd.viewPos[1] = b.position.y;
                fd.viewPos[2] = b.position.z; fd.viewPos[3] = 1.0f;
                fd.camFwd[0]=b.forward.x; fd.camFwd[1]=b.forward.y; fd.camFwd[2]=b.forward.z;
                fd.camRight[0]=b.right.x; fd.camRight[1]=b.right.y; fd.camRight[2]=b.right.z;
                fd.camUp[0]=b.up.x; fd.camUp[1]=b.up.y; fd.camUp[2]=b.up.z;
                fd.skyParams[0] = b.tanHalfFovY; fd.skyParams[1] = b.aspect; fd.skyParams[2] = envMaxLod;
                return fd;
            };

            // Record the whole frame (shadow -> scene -> bloom) into a render graph for camera `cam`.
            // `toSwapchain`: when true the bloom composite writes + presents the swapchain (fly loop);
            // when false the same composite still targets the swapchain but the frame is captured.
            auto recordFrame = [&](render::RenderGraph& graph, const FrameData& fd,
                                   const std::vector<Glass>& sortedGlass) {
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
                    [&, fd](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
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
                            Mat4 wmat = truckPlace * inst.worldTransform;
                            cmd.PushConstants(wmat.m, sizeof(float) * 16);
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
                    [&, fd, sortedGlass](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
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
                            Mat4 wmat = truckPlace * inst.worldTransform;
                            const hf::asset::PbrMaterial& m = *inst.material;
                            float pc[20];
                            for (int k = 0; k < 16; ++k) pc[k] = wmat.m[k];
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
                        for (const auto& g : sortedGlass) {
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
                        BloomParams p = mkPC(w, h, kBloomStrength);
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
                        BloomParams p = mkPC(w, h, kBloomStrength);
                        cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                        cmd.BindPipeline(*compositePipe);
                        cmd.BindTexturePair(*rt, *up[0]);
                        cmd.PushConstants(&p, sizeof(p));
                        cmd.Draw(3);
                        cmd.EndRenderPass();
                    });
            };

            auto sortGlass = [&](const Vec3& eye) {
                std::vector<Glass> g = glass;
                std::sort(g.begin(), g.end(), [&](const Glass& a, const Glass& b) {
                    return math::length(a.pos - eye) > math::length(b.pos - eye);
                });
                return g;
            };

            // === --camera-shot: scripted pose -> ONE captured frame. ===
            if (cameraShotPath) {
                runtime::Camera cam;
                cam.aspect = aspect;
                // Parse "<yaw,pitch,x,y,z>" (radians, world units). Missing fields keep defaults.
                if (cameraShotPose) {
                    float v[5] = {0, 0, 0, 4, 10};
                    std::sscanf(cameraShotPose, "%f,%f,%f,%f,%f", &v[0], &v[1], &v[2], &v[3], &v[4]);
                    cam.yaw = v[0]; cam.SetPitch(v[1]); cam.position = {v[2], v[3], v[4]};
                }
                FrameData fd = makeFrameData(cam);
                render::RenderGraph graph;
                recordFrame(graph, fd, sortGlass(cam.position));
                device->CaptureNextFrame();
                graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
                graph.Execute(*device);
                std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
                bool ok = false;
                if (device->GetCapturedPixels(px, cw, ch2)) {
                    ok = WriteBMP(cameraShotPath, px, cw, ch2);
                    if (ok) std::printf("wrote %s (%ux%u) — camera-shot pose yaw=%.3f pitch=%.3f "
                                        "pos=(%.2f,%.2f,%.2f)\n", cameraShotPath, cw, ch2,
                                        cam.yaw, cam.pitch, cam.position.x, cam.position.y, cam.position.z);
                    else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", cameraShotPath);
                } else {
                    std::fprintf(stderr, "FATAL: no captured pixels\n");
                }
                device->WaitIdle();
                return ok ? 0 : 1;
            }

            // === --fly / --fly-dry-run: live navigable viewport. ===
            runtime::Camera cam;
            cam.aspect = aspect;
            // A sensible starting pose: the capstone overview (eye ~ (0,4,10) looking at the scene).
            cam.position = {0.0f, 4.0f, 10.0f};
            cam.yaw = 0.0f; cam.SetPitch(-0.25f);
            runtime::FlyCameraController controller;
            runtime::FixedTimestep clock(1.0f / 120.0f);

            if (flyDryRun) {
                // Headless CI exercise: feed synthetic input frames through the loop logic (camera +
                // clock update + frame record) WITHOUT presenting, then exit. Proves the loop drives
                // the camera and records a valid graph without a GUI window.
                runtime::InputState in;
                in.relativeMouse = true;
                int totalSteps = 0;
                math::Vec3 startPos = cam.position;
                for (int frame = 0; frame < 8; ++frame) {
                    in.keyDown[(int)runtime::Key::W] = true;       // drive forward
                    in.mouseDx = 6.0f; in.mouseDy = -3.0f;         // look around
                    float dt = 1.0f / 60.0f;
                    totalSteps += clock.Tick(dt);
                    controller.Update(cam, in, dt);
                    FrameData fd = makeFrameData(cam);
                    render::RenderGraph graph;
                    recordFrame(graph, fd, sortGlass(cam.position));
                    // Topo-sort/validate the graph (no Execute -> no swapchain present needed).
                    (void)graph;
                }
                bool moved = math::length(cam.position - startPos) > 1e-3f;
                std::printf("fly-dry-run: %d frames, %d fixed steps, camera moved %s "
                            "(yaw=%.3f pitch=%.3f pos=(%.2f,%.2f,%.2f))\n",
                            8, totalSteps, moved ? "yes" : "NO",
                            cam.yaw, cam.pitch, cam.position.x, cam.position.y, cam.position.z);
                device->WaitIdle();
                return moved ? 0 : 1;
            }

            // Live windowed loop.
            window.SetRelativeMouse(true);  // mouse-look engaged; ESC quits, frees the cursor on exit
            std::printf("--fly: WASD move, mouse look, Space/E up, Ctrl/Q down, Shift sprint, "
                        "wheel speed, ESC quit | P play/pause, O step, G/R/T gizmo mode, Ctrl+S save\n");

            // Slice AB: editor run-state + selection wired into the live loop. The simulation is static
            // this scene, so play/pause/step is demonstrated by the accumulated fixed-step count the
            // PlayState gates; Ctrl+S serializes a representative ECS (the fly scene's placed objects)
            // back to disk via scene::DumpScene; G/R/T pick the gizmo manipulation mode. Live click-to-
            // pick + axis drag is manual (the picking/gizmo math itself is unit-tested + golden-verified
            // via --pick-test / --gizmo-shot); this loop drives the same headless-proven logic. -------
            runtime::PlayState play;
            editor::Selection sel;
            sel.index = 0; sel.mode = editor::GizmoMode::Translate;
            // A representative editable ECS mirroring a couple of the fly scene's static objects, so
            // Ctrl+S has real transforms to serialize through the IO layer.
            ecs::Registry editReg;
            scene::SceneResources editRes;
            editRes.AddMesh("cube", &cube);
            editRes.AddMesh("sphere", &sphere);
            {
                ecs::Entity ped = editReg.create();
                scene::Transform pt; pt.position = {0.0f, pedH, -1.8f}; pt.scale = {1.0f, pedH, 1.0f};
                editReg.add(ped, scene::TransformC{pt});
                editReg.add(ped, scene::MeshC{&cube});
                editReg.add(ped, scene::MaterialC{});
            }
            bool prevP = false, prevO = false, prevCtrlS = false;
            int savedCount = 0;
            int liveSteps = 0;

            auto last = std::chrono::steady_clock::now();
            bool running = true;
            while (running) {
                running = window.PumpEvents();
                const runtime::InputState& in = window.Input();
                if (in.Down(runtime::Key::Esc)) running = false;
                if (window.ConsumeResized()) {
                    device->WaitIdle();
                    w = window.FramebufferWidth();
                    h = window.FramebufferHeight();
                    device->Swapchain().Recreate(w, h);
                    rt = device->CreateRenderTarget(w, h, kHdr);
                    buildMips(w, h);
                    cam.aspect = (h > 0) ? (float)w / (float)h : 1.0f;
                }
                // --- Editor controls (edge-triggered). ---
                bool nowP = in.Down(runtime::Key::P);
                if (nowP && !prevP) { play.Toggle();
                    std::printf("[editor] %s\n", play.IsPlaying() ? "PLAY" : "PAUSE"); }
                prevP = nowP;
                bool nowO = in.Down(runtime::Key::O);
                if (nowO && !prevO) { play.RequestStep(); }
                prevO = nowO;
                if (in.Down(runtime::Key::G)) sel.mode = editor::GizmoMode::Translate;
                if (in.Down(runtime::Key::R)) sel.mode = editor::GizmoMode::Rotate;
                if (in.Down(runtime::Key::T)) sel.mode = editor::GizmoMode::Scale;
                bool nowCtrlS = in.Down(runtime::Key::Ctrl) && in.Down(runtime::Key::S);
                if (nowCtrlS && !prevCtrlS) {
                    std::string json = scene::DumpScene(editReg, editRes);
                    std::ofstream out("fly_scene_edit.json", std::ios::binary);
                    if (out) { out << json; ++savedCount;
                        std::printf("[editor] saved fly_scene_edit.json (%zu bytes)\n", json.size()); }
                }
                prevCtrlS = nowCtrlS;

                auto now = std::chrono::steady_clock::now();
                float dt = std::min(std::chrono::duration<float>(now - last).count(), 0.1f);
                last = now;

                // Fixed-timestep advance, GATED by the editor run-state: paused freezes simulation
                // time (0 steps) unless a single step was requested; playing passes all steps through.
                int fixedSteps = clock.Tick(dt);
                liveSteps += play.StepsThisTick(fixedSteps);
                // The camera still flies while paused (paused freezes SIM, not the viewport).
                controller.Update(cam, in, dt);

                FrameData fd = makeFrameData(cam);
                render::RenderGraph graph;
                recordFrame(graph, fd, sortGlass(cam.position));
                graph.Execute(*device);  // composite -> swapchain -> present
            }
            window.SetRelativeMouse(false);
            std::printf("--fly: exited after %d simulated fixed steps, %d save(s)\n",
                        liveSteps, savedCount);
            (void)sel;
            device->WaitIdle();
            return 0;
        }

        // --- Physics showcase (--physics-shot, Slice S): a self-contained capture path that does NOT
        // touch the default scene. Build a physics::World with a static ground plane (y=0) and a
        // deterministic arrangement of dynamic spheres dropped from a modest height; STEP the world a
        // FIXED number of times (240 @ dt=1/120) so it fully settles into a stable pile; then upload
        // ONE instance transform per body (each body's Mat4) and render the resting bodies with the
        // EXISTING instanced lit + instanced shadow pipelines over the ground plane + skybox. One
        // frame -> BMP -> exit. Reuses the Slice-Q instanced pipeline; golden-locked pipelines and the
        // physics core (pure C++, hf_core) are untouched. ----------------------------------------------
        // --- Debug-visualization showcase (--debug-shot, Slice W): build the SAME settled physics
        // sphere-pyramid scene as --physics-shot (ground + sky + lit/shadowed resting bodies), then
        // overlay an immediate-mode DebugDraw layer: a ground grid, a wireframe AABB hugging each
        // body, per-body wire spheres, a directional-light arrow, and physics contact markers. The
        // overlay is one LINE_LIST draw (a per-frame CPU-built line vertex buffer, the ImGui dynamic-
        // geometry pattern) using a NEW debug-line pipeline (lineList=true, usesFrameUniforms=true,
        // depthTest=true, depthWrite=false), drawn in the scene RT pass AFTER opaque geometry so the
        // lines are correctly occluded where they pass behind the shaded spheres. One BMP -> exit. ---
        if (debugShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // Settled physics pyramid (identical construction to --physics-shot).
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
                scene::InstanceData di;
                for (int k = 0; k < 16; ++k) di.model[k] = m.m[k];
                instances.push_back(di);
            }
            const uint32_t kInstanceCount = (uint32_t)instances.size();

            // --- Instanced lit + static lit + shadow + sky + post pipelines (same as physics shot). ---
            auto instVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_instanced.vert.hlsl.spv");
            auto litFsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto instVs = device->CreateShaderModule({std::span<const uint32_t>(instVsWords)});
            auto litFs  = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            rhi::GraphicsPipelineDesc instDesc;
            instDesc.vertex = instVs.get(); instDesc.fragment = litFs.get();
            instDesc.vertexLayout = scene::MeshVertexLayout();
            instDesc.instanceLayout = scene::InstanceTransformLayout();
            instDesc.colorFormat = device->Swapchain().ColorFormat();
            instDesc.depthTest = true; instDesc.usesFrameUniforms = true; instDesc.usesTexture = true;
            instDesc.pushConstantSize = sizeof(float) * 4;
            auto instPipeline = device->CreateGraphicsPipeline(instDesc);

            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = device->Swapchain().ColorFormat();
            litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            auto instShWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow_instanced.vert.hlsl.spv");
            auto shadowFsW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto instShVs = device->CreateShaderModule({std::span<const uint32_t>(instShWords)});
            auto shadowFs = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});
            rhi::GraphicsPipelineDesc instShDesc;
            instShDesc.vertex = instShVs.get(); instShDesc.fragment = shadowFs.get();
            instShDesc.vertexLayout = scene::MeshVertexLayout();
            instShDesc.instanceLayout = scene::InstanceTransformLayout();
            instShDesc.depthTest = true; instShDesc.depthOnly = true; instShDesc.usesFrameUniforms = true;
            auto instShadowPipeline = device->CreateGraphicsPipeline(instShDesc);

            auto staticShW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto staticShVs = device->CreateShaderModule({std::span<const uint32_t>(staticShW)});
            rhi::GraphicsPipelineDesc stShDesc;
            stShDesc.vertex = staticShVs.get(); stShDesc.fragment = shadowFs.get();
            stShDesc.vertexLayout = scene::MeshVertexLayout();
            stShDesc.depthTest = true; stShDesc.depthOnly = true; stShDesc.usesFrameUniforms = true;
            stShDesc.pushConstantSize = sizeof(float) * 16;
            auto staticShadowPipeline = device->CreateGraphicsPipeline(stShDesc);

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

            // --- NEW: debug-line pipeline (LINE_LIST, frame uniforms, depth-test on / write off). ---
            auto dbgVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/debug_line.vert.hlsl.spv");
            auto dbgFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/debug_line.frag.hlsl.spv");
            auto dbgVs = device->CreateShaderModule({std::span<const uint32_t>(dbgVsW)});
            auto dbgFs = device->CreateShaderModule({std::span<const uint32_t>(dbgFsW)});
            rhi::GraphicsPipelineDesc dbgD;
            dbgD.vertex = dbgVs.get(); dbgD.fragment = dbgFs.get();
            dbgD.vertexLayout.stride = sizeof(debug::LineVertex);
            dbgD.vertexLayout.attributes = {
                {0, rhi::Format::RGB32_Float, 0},
                {1, rhi::Format::RGB32_Float, 12},
            };
            dbgD.colorFormat = device->Swapchain().ColorFormat();
            dbgD.lineList = true;
            dbgD.depthTest = true;
            dbgD.depthWrite = false;
            dbgD.usesFrameUniforms = true;
            auto debugPipeline = device->CreateGraphicsPipeline(dbgD);

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
            scene::Mesh sphere = scene::Mesh::Sphere(*device);

            rhi::BufferDesc instBufDesc;
            instBufDesc.size = (uint64_t)instances.size() * sizeof(scene::InstanceData);
            instBufDesc.initialData = instances.data();
            instBufDesc.usage = rhi::BufferUsage::Vertex;
            auto instanceBuffer = device->CreateBuffer(instBufDesc);

            Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});

            // --- Build the debug overlay (CPU-side, then upload once). ---
            debug::DebugDraw dd;
            const Vec3 lightDirVec = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
            dd.Grid(8.0f, 1.0f, {0.30f, 0.32f, 0.38f});           // ground grid (XZ, y=0)
            // Per-body wireframe AABB (yellow) + wire sphere (cyan) hugging each settled sphere.
            const scene::MeshBounds& sb = sphere.bounds();         // unit sphere ±0.5
            for (const auto& b : world.bodies) {
                Mat4 m = b.Transform();
                debug::AabbWorld(dd, sb.min, sb.max, m, {1.0f, 0.85f, 0.1f});
                dd.WireSphere(b.position, b.radius, {0.1f, 0.9f, 0.95f}, 16);
            }
            // Directional-light arrow (orange), anchored above the pile pointing along the light.
            debug::LightArrow(dd, {3.5f, 4.5f, 3.5f}, lightDirVec, 2.5f, {1.0f, 0.55f, 0.1f});
            // Physics contact markers (magenta crosses + green normals) at the settled contacts.
            debug::PhysicsContacts(dd, world, {1.0f, 0.2f, 0.8f}, {0.2f, 1.0f, 0.3f});

            const uint32_t kLineVertCount = (uint32_t)dd.VertexCount();
            rhi::BufferDesc lineBufDesc;
            lineBufDesc.size = (uint64_t)dd.Vertices().size() * sizeof(debug::LineVertex);
            lineBufDesc.initialData = dd.Vertices().data();
            lineBufDesc.usage = rhi::BufferUsage::Vertex;
            auto lineBuffer = device->CreateBuffer(lineBufDesc);

            // Camera (same framing as the physics shot).
            const Vec3 eye{6.5f, 4.5f, 7.0f};
            const Vec3 center{0.0f, 1.0f, 0.0f};
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
                Vec3 sc{0.0f, 1.0f, 0.0f};
                Vec3 lightEye = sc - lightDirVec * 18.0f;
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
                Mat4 lightOrtho = Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 40.0f);
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
                    // --- Debug overlay: one LINE_LIST draw, AFTER opaque geometry (depth-tested
                    // so lines behind the spheres are occluded; depth-write off so they never fight). ---
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
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(debugShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — %u bodies, %u debug-line vertices\n",
                                    debugShotPath, cw, ch2, kInstanceCount, kLineVertCount);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", debugShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- Editor gizmo showcase (--gizmo-shot <objIndex> <out.bmp>, Slice AB): build a small
        // deterministic multi-object scene (ground plane + cube + sphere + tall box), programmatically
        // SELECT object <objIndex>, and render the scene PLUS the selected object's TRANSLATE gizmo
        // (3 colored axis arrows) drawn through the Slice-W debug-line layer (depthTest on / write off,
        // AFTER opaque geometry) from a fixed camera. The gizmo is emitted by editor::EmitGizmo at the
        // selected object's Transform via debug::DebugDraw, then uploaded as one LINE_LIST draw — the
        // SAME path --debug-shot uses. One BMP -> exit. New golden; existing goldens untouched. ----
        if (gizmoShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // --- Scene objects: transforms + (matching) world AABBs. These MUST match the --pick-test
            // AABBs above so headless picking and the rendered gizmo agree on object placement. ---
            scene::Mesh planeMesh = scene::Mesh::Plane(*device);
            scene::Mesh cubeMesh  = scene::Mesh::Cube(*device);
            scene::Mesh sphereMesh = scene::Mesh::Sphere(*device);

            struct GizmoObj { const scene::Mesh* mesh; scene::Transform xform; Vec3 color; };
            std::vector<GizmoObj> objs;
            // 0: ground plane (Plane is the XZ unit quad; scale to a 14x14 ground).
            { scene::Transform t; t.scale = {7.0f, 1.0f, 7.0f};
              objs.push_back({&planeMesh, t, {0.55f, 0.55f, 0.6f}}); }
            // 1: cube at (-2, 0.5, 0) (Cube is a unit [-0.5,0.5] cube -> half-extent 0.5).
            { scene::Transform t; t.position = {-2.0f, 0.5f, 0.0f};
              objs.push_back({&cubeMesh, t, {0.8f, 0.35f, 0.25f}}); }
            // 2: sphere at (1.5, 1.0, 1.5) (unit sphere radius 0.5 -> scale 1.6 => radius 0.8).
            { scene::Transform t; t.position = {1.5f, 1.0f, 1.5f}; t.scale = {1.6f, 1.6f, 1.6f};
              objs.push_back({&sphereMesh, t, {0.3f, 0.6f, 0.85f}}); }
            // 3: tall box at (3, 1.5, -1), 0.6 wide / 3 tall.
            { scene::Transform t; t.position = {3.0f, 1.5f, -1.0f}; t.scale = {0.6f, 3.0f, 0.6f};
              objs.push_back({&cubeMesh, t, {0.45f, 0.75f, 0.4f}}); }

            int selIndex = gizmoShotIndex;
            if (selIndex < 0) selIndex = 0;
            if (selIndex >= (int)objs.size()) selIndex = (int)objs.size() - 1;
            editor::Selection sel;
            sel.index = selIndex;
            sel.mode = editor::GizmoMode::Translate;

            // --- Pipelines: static lit + static shadow + sky + post + debug-line (LDR). ---
            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto litFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto litFs = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = device->Swapchain().ColorFormat();
            litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            auto staticShW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto shadowFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto staticShVs = device->CreateShaderModule({std::span<const uint32_t>(staticShW)});
            auto shadowFs = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});
            rhi::GraphicsPipelineDesc stShDesc;
            stShDesc.vertex = staticShVs.get(); stShDesc.fragment = shadowFs.get();
            stShDesc.vertexLayout = scene::MeshVertexLayout();
            stShDesc.depthTest = true; stShDesc.depthOnly = true; stShDesc.usesFrameUniforms = true;
            stShDesc.pushConstantSize = sizeof(float) * 16;
            auto staticShadowPipeline = device->CreateGraphicsPipeline(stShDesc);

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

            auto dbgVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/debug_line.vert.hlsl.spv");
            auto dbgFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/debug_line.frag.hlsl.spv");
            auto dbgVs = device->CreateShaderModule({std::span<const uint32_t>(dbgVsW)});
            auto dbgFs = device->CreateShaderModule({std::span<const uint32_t>(dbgFsW)});
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

            auto rt = device->CreateRenderTarget(w, h);
            auto shadowMap = device->CreateShadowMap(2048);
            device->SetShadowMap(*shadowMap);

            std::vector<uint8_t> checker = MakeCheckerboard();
            auto groundTex = device->CreateTexture(
                {256, 256, rhi::Format::RGBA8_UNorm, checker.data(), checker.size()});
            const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
            auto flatNormal = device->CreateTexture(
                {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});

            // --- Build the gizmo overlay: the selected object's translate gizmo, sized to the object,
            // emitted through the debug-line layer at the object's transform. ---
            debug::DebugDraw dd;
            {
                const scene::Transform& xf = objs[sel.index].xform;
                // Handle length scales with the object so it reads at any size (sphere bigger, etc.).
                float reach = std::max({xf.scale.x, xf.scale.y, xf.scale.z});
                float handleLen = 1.2f + 0.8f * reach;
                editor::EmitGizmo(dd, xf, sel.mode, handleLen, editor::kAxisNone);
            }
            const uint32_t kLineVertCount = (uint32_t)dd.VertexCount();
            rhi::BufferDesc lineBufDesc;
            lineBufDesc.size = (uint64_t)dd.Vertices().size() * sizeof(debug::LineVertex);
            lineBufDesc.initialData = dd.Vertices().data();
            lineBufDesc.usage = rhi::BufferUsage::Vertex;
            auto lineBuffer = device->CreateBuffer(lineBufDesc);

            // --- Fixed camera (same pose --pick-test uses). ---
            runtime::Camera cam;
            cam.position = {2.0f, 4.5f, 9.0f};
            cam.yaw = 0.15f; cam.SetPitch(-0.32f);
            cam.aspect = aspect;

            const Vec3 lightDirVec = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
            FrameData fd{};
            {
                Mat4 vp = cam.ViewProj();
                for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
                fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
                fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
                fd.viewPos[0] = cam.position.x; fd.viewPos[1] = cam.position.y;
                fd.viewPos[2] = cam.position.z; fd.viewPos[3] = 1.0f;
                fd.ptCount[0] = 0.0f;
                Vec3 sc{0.0f, 1.0f, 0.0f};
                Vec3 lightEye = sc - lightDirVec * 18.0f;
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
                Mat4 lightOrtho = Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 40.0f);
                Mat4 lightVP = lightOrtho * lightView;
                for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
                runtime::CameraBasis cb = cam.Basis();
                fd.camFwd[0]=cb.forward.x; fd.camFwd[1]=cb.forward.y; fd.camFwd[2]=cb.forward.z;
                fd.camRight[0]=cb.right.x; fd.camRight[1]=cb.right.y; fd.camRight[2]=cb.right.z;
                fd.camUp[0]=cb.up.x; fd.camUp[1]=cb.up.y; fd.camUp[2]=cb.up.z;
                fd.skyParams[0] = cb.tanHalfFovY; fd.skyParams[1] = aspect;
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
                    // Gizmo overlay: one LINE_LIST draw, after opaque geometry (depth-tested).
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
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(gizmoShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — selected object %d, %u gizmo-line vertices\n",
                                    gizmoShotPath, cw, ch2, sel.index, kLineVertCount);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", gizmoShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- SSAO showcase (--ssao-shot / --ssao-shot-off, Slice Y): the SAME settled physics
        // sphere-pyramid scene as --physics-shot (ground + sky + lit/shadowed resting bodies), rendered
        // into an HDR (RGBA16F) scene target, PLUS classic screen-space ambient occlusion: a separate
        // view-space normal+linear-depth g-buffer prepass, a 16-sample baked-hemisphere-kernel AO pass
        // (reconstructing view-space position from linear depth + projection params, oriented by a TBN
        // from a tiled rotation noise), a 4x4 box blur, and a final composite that MULTIPLIES the lit
        // scene by the blurred AO (then the usual exposure/ACES/grade/vignette). --ssao-shot applies AO
        // (aoStrength=1); --ssao-shot-off renders the identical scene with AO forced off (aoStrength=0)
        // through the IDENTICAL composite pipeline for a clean on/off comparison. SEPARATE
        // gbuffer/ssao/blur/composite pipelines + shaders; existing pipelines/shaders/goldens untouched.
        if (ssaoShotPath || ssaoShotOffPath || ssaoDebugPath) {
            using math::Mat4; using math::Vec3;
            const bool aoOn = (ssaoShotPath != nullptr) || (ssaoDebugPath != nullptr);
            const bool aoDebug = (ssaoDebugPath != nullptr);
            const char* outPath = ssaoDebugPath ? ssaoDebugPath
                                : (aoOn ? ssaoShotPath : ssaoShotOffPath);
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
            const rhi::Format kHdr = rhi::Format::RGBA16_Float;
            const float kFovY = 1.04719755f;

            // Build + settle the pyramid (identical scenario + step budget to --physics-shot).
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
                scene::InstanceData di;
                for (int k = 0; k < 16; ++k) di.model[k] = m.m[k];
                instances.push_back(di);
            }
            const uint32_t kInstanceCount = (uint32_t)instances.size();

            // --- Lit scene pipelines (writing the HDR RT) — UNCHANGED lit shaders. ---
            auto instVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_instanced.vert.hlsl.spv");
            auto litFsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto instVs = device->CreateShaderModule({std::span<const uint32_t>(instVsWords)});
            auto litFs  = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            rhi::GraphicsPipelineDesc instDesc;
            instDesc.vertex = instVs.get(); instDesc.fragment = litFs.get();
            instDesc.vertexLayout = scene::MeshVertexLayout();
            instDesc.instanceLayout = scene::InstanceTransformLayout();
            instDesc.colorFormat = kHdr;
            instDesc.depthTest = true; instDesc.usesFrameUniforms = true; instDesc.usesTexture = true;
            instDesc.pushConstantSize = sizeof(float) * 4;
            auto instPipeline = device->CreateGraphicsPipeline(instDesc);

            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = kHdr;
            litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            // --- Shadow pipelines (UNCHANGED). ---
            auto instShWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow_instanced.vert.hlsl.spv");
            auto shadowFsW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto instShVs = device->CreateShaderModule({std::span<const uint32_t>(instShWords)});
            auto shadowFs = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});
            rhi::GraphicsPipelineDesc instShDesc;
            instShDesc.vertex = instShVs.get(); instShDesc.fragment = shadowFs.get();
            instShDesc.vertexLayout = scene::MeshVertexLayout();
            instShDesc.instanceLayout = scene::InstanceTransformLayout();
            instShDesc.depthTest = true; instShDesc.depthOnly = true; instShDesc.usesFrameUniforms = true;
            auto instShadowPipeline = device->CreateGraphicsPipeline(instShDesc);

            auto staticShW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto staticShVs = device->CreateShaderModule({std::span<const uint32_t>(staticShW)});
            rhi::GraphicsPipelineDesc stShDesc;
            stShDesc.vertex = staticShVs.get(); stShDesc.fragment = shadowFs.get();
            stShDesc.vertexLayout = scene::MeshVertexLayout();
            stShDesc.depthTest = true; stShDesc.depthOnly = true; stShDesc.usesFrameUniforms = true;
            stShDesc.pushConstantSize = sizeof(float) * 16;
            auto staticShadowPipeline = device->CreateGraphicsPipeline(stShDesc);

            // --- Sky (writing HDR RT) — UNCHANGED sky shaders. ---
            auto skyVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.vert.hlsl.spv");
            auto skyFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.frag.hlsl.spv");
            auto skyVsM = device->CreateShaderModule({std::span<const uint32_t>(skyVsW)});
            auto skyFsM = device->CreateShaderModule({std::span<const uint32_t>(skyFsW)});
            rhi::GraphicsPipelineDesc skyD;
            skyD.vertex = skyVsM.get(); skyD.fragment = skyFsM.get();
            skyD.colorFormat = kHdr;
            skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
            auto skyPipe = device->CreateGraphicsPipeline(skyD);

            // --- NEW: g-buffer prepass pipelines (static + instanced), writing view-space normal +
            // linear depth into an RGBA16F target. Push constant = { model, view } (static) /
            // { view } (instanced). usesFrameUniforms for the shared viewProj. ---
            auto gbVsW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/gbuffer.vert.hlsl.spv");
            auto gbInVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/gbuffer_instanced.vert.hlsl.spv");
            auto gbFsW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/gbuffer.frag.hlsl.spv");
            auto gbVs   = device->CreateShaderModule({std::span<const uint32_t>(gbVsW)});
            auto gbInVs = device->CreateShaderModule({std::span<const uint32_t>(gbInVsW)});
            auto gbFs   = device->CreateShaderModule({std::span<const uint32_t>(gbFsW)});
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

            // --- NEW: SSAO + blur + composite fullscreen pipelines (fragment push constants). ---
            auto postVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.vert.hlsl.spv");
            auto postVsM = device->CreateShaderModule({std::span<const uint32_t>(postVsW)});
            auto loadFs = [&](const char* name) {
                auto words = LoadSpirv(std::string(HF_SHADER_DIR) + "/" + name + ".spv");
                return device->CreateShaderModule({std::span<const uint32_t>(words)});
            };
            struct SsaoParams { float texel[2]; float radius, bias, intensity, tanHalfFovY, aspect, pad; };
            struct BlurParams { float texel[2]; float pad[2]; };
            struct SsaoCompParams { float texel[2]; float aoStrength, intensity; };

            auto ssaoFs = loadFs("ssao.frag.hlsl");
            auto blurFs = loadFs("ssao_blur.frag.hlsl");
            auto compFs = loadFs("ssao_composite.frag.hlsl");

            rhi::GraphicsPipelineDesc ssaoD;
            ssaoD.vertex = postVsM.get(); ssaoD.fragment = ssaoFs.get();
            ssaoD.colorFormat = kHdr;
            ssaoD.depthTest = false; ssaoD.usesTexture = true; ssaoD.fullscreen = true;
            ssaoD.fragmentPushConstants = true; ssaoD.pushConstantSize = sizeof(SsaoParams);
            auto ssaoPipe = device->CreateGraphicsPipeline(ssaoD);

            rhi::GraphicsPipelineDesc blurD;
            blurD.vertex = postVsM.get(); blurD.fragment = blurFs.get();
            blurD.colorFormat = kHdr;
            blurD.depthTest = false; blurD.usesTexture = true; blurD.fullscreen = true;
            blurD.fragmentPushConstants = true; blurD.pushConstantSize = sizeof(BlurParams);
            auto blurPipe = device->CreateGraphicsPipeline(blurD);

            rhi::GraphicsPipelineDesc compD;
            compD.vertex = postVsM.get(); compD.fragment = compFs.get();
            compD.colorFormat = device->Swapchain().ColorFormat();
            compD.depthTest = false; compD.usesTexture = true; compD.fullscreen = true;
            compD.fragmentPushConstants = true; compD.pushConstantSize = sizeof(SsaoCompParams);
            auto compPipe = device->CreateGraphicsPipeline(compD);

            // --- Render targets: HDR lit scene + RGBA16F g-buffer (full res) + AO + blurred AO. ---
            auto rt    = device->CreateRenderTarget(w, h, kHdr);
            auto gbuf  = device->CreateRenderTarget(w, h, kHdr);
            auto aoRT  = device->CreateRenderTarget(w, h, kHdr);
            auto aoBlurRT = device->CreateRenderTarget(w, h, kHdr);
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
            Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
            FrameData fd{};
            {
                Mat4 proj = Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f);
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
                Mat4 lightOrtho = Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 40.0f);
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

            // SSAO tunables. Strong intensity so the contact AO is clearly visible.
            SsaoParams sp{};
            sp.texel[0] = 1.0f / (float)w; sp.texel[1] = 1.0f / (float)h;
            sp.radius = 0.30f; sp.bias = 0.025f; sp.intensity = 1.6f;
            sp.tanHalfFovY = std::tan(0.5f * kFovY); sp.aspect = aspect; sp.pad = 0.0f;
            BlurParams blurP{}; blurP.texel[0] = 1.0f / (float)w; blurP.texel[1] = 1.0f / (float)h;
            SsaoCompParams cp{}; cp.texel[0] = 1.0f / (float)w; cp.texel[1] = 1.0f / (float)h;
            cp.aoStrength = aoDebug ? -1.0f : (aoOn ? 1.0f : 0.0f); cp.intensity = 1.7f;

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

            // Lit scene -> HDR RT.
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

            // G-buffer prepass -> RGBA16F (view-space normal + linear depth). Clear w=0 = background.
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

            // SSAO pass -> AO RT.
            graph.AddPass("ssao", {rgGbuf}, {rgAO},
                [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{1, 1, 1, 1});
                    cmd.BindPipeline(*ssaoPipe);
                    cmd.BindTexture(*gbuf);
                    cmd.PushConstants(&sp, sizeof(sp));
                    cmd.Draw(3);
                    cmd.EndRenderPass();
                });

            // Blur pass -> blurred AO RT.
            graph.AddPass("ssaoBlur", {rgAO}, {rgAOBlur},
                [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{1, 1, 1, 1});
                    cmd.BindPipeline(*blurPipe);
                    cmd.BindTexture(*aoRT);
                    cmd.PushConstants(&blurP, sizeof(blurP));
                    cmd.Draw(3);
                    cmd.EndRenderPass();
                });

            // Composite: lit HDR scene * AO -> tonemap -> swapchain.
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
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(outPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — SSAO %s, %u bodies\n",
                                    outPath, cw, ch2, aoOn ? "ON" : "OFF", kInstanceCount);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", outPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        if (physicsShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // Build the world: a static ground plane (y=0) + a deterministic 4-layer SQUARE-PYRAMID
            // sphere packing (30 unit spheres: 16+9+4+1). Each higher layer is an (N-k)x(N-k) grid
            // nestled into the POCKETS of the layer below — offset half a diameter in x/z and raised
            // by R*sqrt(2) (the rest height of a sphere sitting in a 4-sphere square pocket) — and
            // dropped a hair (0.01 m) above its rest height so the solver seats it. No RNG: every
            // position is a pure function of (layer, gx, gz), so the settled pile is golden-stable.
            physics::World world;
            const float R = 0.5f;
            const int kLayers = 4;
            const float d = 2.0f * R;                 // in-layer contact spacing (one diameter)
            const float dy = R * 1.41421356f;         // vertical rise per pocket-nested layer
            for (int k = 0; k < kLayers; ++k) {
                int m = kLayers - k;                  // this layer is m x m
                float off = 0.5f * (float)(m - 1) * d;
                float y = R + (float)k * dy;
                for (int gx = 0; gx < m; ++gx) {
                    for (int gz = 0; gz < m; ++gz) {
                        float x = (float)gx * d - off;
                        float z = (float)gz * d - off;
                        world.bodies.push_back(
                            physics::MakeDynamicSphere({x, y + 0.01f, z}, R));
                    }
                }
            }
            // Step to rest: 240 fixed steps @ dt=1/120 (2 s of sim). This pyramid is essentially at
            // rest (|vel| < 0.05 m/s, KE ~ 1e-3) by ~step 60; 240 leaves a wide margin.
            const float dt = 1.0f / 120.0f;
            for (int s = 0; s < 240; ++s) world.Step(dt);

            // One instance transform per resting body.
            std::vector<scene::InstanceData> instances;
            instances.reserve(world.bodies.size());
            for (const auto& b : world.bodies) {
                Mat4 m = b.Transform();
                scene::InstanceData d;
                for (int k = 0; k < 16; ++k) d.model[k] = m.m[k];
                instances.push_back(d);
            }
            const uint32_t kInstanceCount = (uint32_t)instances.size();

            // Instanced lit pipeline (lit_instanced.vert + shared lit.frag), per-instance binding 1.
            auto instVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_instanced.vert.hlsl.spv");
            auto litFsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto instVs = device->CreateShaderModule({std::span<const uint32_t>(instVsWords)});
            auto litFs  = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            rhi::GraphicsPipelineDesc instDesc;
            instDesc.vertex = instVs.get();
            instDesc.fragment = litFs.get();
            instDesc.vertexLayout = scene::MeshVertexLayout();
            instDesc.instanceLayout = scene::InstanceTransformLayout();
            instDesc.colorFormat = device->Swapchain().ColorFormat();
            instDesc.depthTest = true;
            instDesc.usesFrameUniforms = true;
            instDesc.usesTexture = true;
            instDesc.pushConstantSize = sizeof(float) * 4;
            auto instPipeline = device->CreateGraphicsPipeline(instDesc);

            // Static lit pipeline for the ground plane.
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

            // Instanced depth-only shadow pipeline (body casters) + static shadow pipeline (ground).
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
            instShDesc.pushConstantSize = 0;
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

            // Camera framed on the settled pyramid (centered near origin, ~2.6 m tall).
            const Vec3 eye{6.5f, 4.5f, 7.0f};
            const Vec3 center{0.0f, 1.0f, 0.0f};
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
                Vec3 sc{0.0f, 1.0f, 0.0f};
                Vec3 lightEye = sc - lightDir * 18.0f;
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
                Mat4 lightOrtho = Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 40.0f);
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
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(physicsShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — %u rigid bodies settled\n",
                                    physicsShotPath, cw, ch2, kInstanceCount);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", physicsShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

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

        // --- Alpha-blended transparency showcase (--transparency-shot, Slice T): a self-contained
        // capture path that does NOT touch the default scene. Checkerboard ground plane + procedural
        // sky + a few OPAQUE lit objects (cubes) to show THROUGH, then a handful of overlapping tinted
        // GLASS spheres at different depths rendered in a SORTED (back-to-front) alpha-blended pass that
        // depth-TESTS the opaque scene but does NOT write depth (the new depthWrite=false). One frame ->
        // BMP -> exit. Uses a SEPARATE transparent pipeline (transparent.vert + transparent.frag);
        // golden-locked pipelines/shaders untouched. --
        if (transparencyShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // Opaque lit pipeline (ground + opaque cubes): shared lit.vert + lit.frag, 80-byte push.
            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto litFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto litFs = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get();
            litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = device->Swapchain().ColorFormat();
            litDesc.depthTest = true;
            litDesc.usesFrameUniforms = true;
            litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;  // model + float4 material
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            // Transparent ("glass") pipeline: transparent.vert + transparent.frag. alphaBlend ON,
            // depthTest ON, depthWrite OFF (reads opaque depth, never writes), double-sided (cullNone).
            // Push constant = { model(64), tintAlpha(16) } = 80 bytes (separate range from lit).
            auto tVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/transparent.vert.hlsl.spv");
            auto tFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/transparent.frag.hlsl.spv");
            auto tVs = device->CreateShaderModule({std::span<const uint32_t>(tVsWords)});
            auto tFs = device->CreateShaderModule({std::span<const uint32_t>(tFsWords)});
            rhi::GraphicsPipelineDesc tDesc;
            tDesc.vertex = tVs.get();
            tDesc.fragment = tFs.get();
            tDesc.vertexLayout = scene::MeshVertexLayout();
            tDesc.colorFormat = device->Swapchain().ColorFormat();
            tDesc.depthTest = true;
            tDesc.depthWrite = false;   // Slice T: depth-test against opaque, do NOT write depth
            tDesc.alphaBlend = true;    // src_alpha / one_minus_src_alpha over the opaque scene
            tDesc.cullNone = true;      // double-sided glass
            tDesc.usesFrameUniforms = true;
            tDesc.usesTexture = false;  // self-contained shader; binds no textures
            tDesc.pushConstantSize = sizeof(float) * 20;  // model + float4 tintAlpha
            auto transparentPipeline = device->CreateGraphicsPipeline(tDesc);

            // Static depth-only shadow pipeline (ground + opaque cubes cast shadows).
            auto shVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto shFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto shVs = device->CreateShaderModule({std::span<const uint32_t>(shVsWords)});
            auto shFs = device->CreateShaderModule({std::span<const uint32_t>(shFsWords)});
            rhi::GraphicsPipelineDesc shDesc;
            shDesc.vertex = shVs.get();
            shDesc.fragment = shFs.get();
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

            // Opaque cubes (behind/among the glass) so there is something to see THROUGH. Spread in X
            // and depth, sitting on the ground.
            // Two opaque cubes sit BEHIND the glass (so they read clearly THROUGH it), and one opaque
            // cube sits clearly IN FRONT of the rearmost glass sphere (so it must OCCLUDE the part of
            // that sphere behind it — the depth-test/depthWrite=false correctness check).
            struct Opaque { Mat4 model; float r, g, b; };
            std::vector<Opaque> opaques = {
                {Mat4::Translate({-2.2f, 0.6f, -1.0f}) * Mat4::RotateY(0.4f) * Mat4::Scale({0.6f,0.6f,0.6f}), 0.90f, 0.30f, 0.20f},
                {Mat4::Translate({ 2.2f, 0.6f, -1.4f}) * Mat4::RotateY(-0.5f) * Mat4::Scale({0.6f,0.6f,0.6f}), 0.95f, 0.85f, 0.25f},
                // In-front occluder: at z=+2.6 (closer to the camera than the glass spheres), overlapping
                // them in screen space, so it covers the glass behind it.
                {Mat4::Translate({-0.9f, 0.7f, 2.8f}) * Mat4::RotateY(0.2f) * Mat4::Scale({0.7f,0.7f,0.7f}), 0.25f, 0.85f, 0.40f},
            };

            // Translucent glass spheres: overlapping, different colors + depths + base alphas. Each is
            // a position + uniform scale + RGB tint + base alpha. Sorted back-to-front each frame.
            struct Glass { Vec3 pos; float scale; float r, g, b, baseAlpha; };
            std::vector<Glass> glass = {
                {{-1.3f, 1.0f,  1.4f}, 1.1f, 0.85f, 0.20f, 0.20f, 0.28f},  // red
                {{ 0.0f, 1.1f,  0.4f}, 1.2f, 0.20f, 0.55f, 0.95f, 0.26f},  // blue
                {{ 1.4f, 1.0f,  1.2f}, 1.1f, 0.30f, 0.90f, 0.45f, 0.30f},  // green
                {{ 0.6f, 1.8f,  2.0f}, 0.9f, 0.95f, 0.85f, 0.25f, 0.24f},  // amber, in front
            };

            // Camera + light + sky frame data (fixed, deterministic). Lower, head-on camera to read
            // the see-through layering clearly.
            const Vec3 eye{0.0f, 2.6f, 7.5f};
            const Vec3 center{0.0f, 1.0f, 0.0f};
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

            // Sort the glass back-to-front by distance from the camera eye (farthest first) so the
            // alpha over-blend is correct. Deterministic for the fixed camera.
            std::sort(glass.begin(), glass.end(), [&](const Glass& a, const Glass& b) {
                float da = math::length(a.pos - eye);
                float db = math::length(b.pos - eye);
                return da > db;  // farthest first
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
                    // Ground caster.
                    cmd.PushConstants(groundModel.m, sizeof(float) * 16);
                    cmd.BindVertexBuffer(plane.vertices());
                    cmd.BindIndexBuffer(plane.indices());
                    cmd.DrawIndexed(plane.indexCount());
                    // Opaque cubes cast shadows (glass does not, by design).
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
                    // --- Opaque pass (writes depth). Ground plane + opaque cubes. ---
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
                        // metallic, roughness; (tint is baked via vertex color? use material only) —
                        // the lit shader multiplies gTex (checker) by vertex color; we want a solid
                        // tint, so push a dielectric material and rely on the bound checker for texture.
                        pc[16] = 0.1f; pc[17] = 0.6f; pc[18] = 0.0f; pc[19] = 0.0f;
                        cmd.PushConstants(pc, sizeof(pc));
                        cmd.DrawIndexed(cube.indexCount());
                    }
                    // --- Sorted translucent pass (alpha-blended, depth-test, NO depth write). The
                    // glass blends over the opaque scene and reads (but does not write) depth, so
                    // opaque geometry in FRONT correctly occludes it and overlapping glass blends. ---
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
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(transparencyShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — %zu glass + %zu opaque\n",
                                    transparencyShotPath, cw, ch2, glass.size(), opaques.size());
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", transparencyShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- HDR bloom showcase (--bloom-shot, Slice U): the SAME HDR-IBL helmet scene as --ibl-shot,
        // but rendered into an HDR (RGBA16F) render target so highlights keep values >1, then run
        // through a true bloom chain — threshold bright-pass -> 5 progressively half-res HDR mips
        // (13-tap downsample) -> tent-filter upsample/combine back up -> composite that adds the
        // bloom and applies the same exposure/ACES/grade/grain/vignette as post.frag, writing the LDR
        // swapchain. The HDR sun and the helmet's emissive cyan gauge bloom (soft halo); the rest of
        // the frame stays sharp. SEPARATE HDR pipelines + bloom shaders; nothing existing is touched.
        if (bloomShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
            const rhi::Format kHdr = rhi::Format::RGBA16_Float;

            hf::asset::EnvironmentMap env = hf::asset::LoadHdrEnvironment(*device, HF_ENV_PATH);
            const float envMaxLod = (float)(env.mipLevels - 1);

            // --- Scene pipelines, but writing the HDR (RGBA16F) target. ---
            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto iblFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_pbr_ibl.frag.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto iblFs = device->CreateShaderModule({std::span<const uint32_t>(iblFsWords)});
            rhi::GraphicsPipelineDesc iblDesc;
            iblDesc.vertex = litVs.get(); iblDesc.fragment = iblFs.get();
            iblDesc.vertexLayout = scene::MeshVertexLayout();
            iblDesc.colorFormat = kHdr;            // HDR scene target
            iblDesc.depthTest = true; iblDesc.usesFrameUniforms = true;
            iblDesc.usesTexture = true; iblDesc.pbrMaterial = true; iblDesc.usesEnvironment = true;
            iblDesc.pushConstantSize = sizeof(float) * 20;
            auto iblPipeline = device->CreateGraphicsPipeline(iblDesc);

            auto litFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto litFs = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = kHdr;            // HDR scene target
            litDesc.depthTest = true; litDesc.usesFrameUniforms = true;
            litDesc.usesTexture = true; litDesc.pushConstantSize = sizeof(float) * 20;
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            auto shadowVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto shadowFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto shadowVs = device->CreateShaderModule({std::span<const uint32_t>(shadowVsW)});
            auto shadowFs = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});
            rhi::GraphicsPipelineDesc shDesc;
            shDesc.vertex = shadowVs.get(); shDesc.fragment = shadowFs.get();
            shDesc.vertexLayout = scene::MeshVertexLayout();
            shDesc.depthTest = true; shDesc.depthOnly = true; shDesc.usesFrameUniforms = true;
            shDesc.pushConstantSize = sizeof(float) * 16;
            auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

            auto skyVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.vert.hlsl.spv");
            auto skyFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky_hdr.frag.hlsl.spv");
            auto skyVsM = device->CreateShaderModule({std::span<const uint32_t>(skyVsW)});
            auto skyFsM = device->CreateShaderModule({std::span<const uint32_t>(skyFsW)});
            rhi::GraphicsPipelineDesc skyD;
            skyD.vertex = skyVsM.get(); skyD.fragment = skyFsM.get();
            skyD.colorFormat = kHdr;               // HDR scene target
            skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
            skyD.usesEnvironment = true;
            auto skyPipe = device->CreateGraphicsPipeline(skyD);

            // --- Bloom pipelines (all fullscreen, fragment push constants). ---
            auto postVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.vert.hlsl.spv");
            auto postVsM = device->CreateShaderModule({std::span<const uint32_t>(postVsW)});
            auto loadFs = [&](const char* name) {
                auto words = LoadSpirv(std::string(HF_SHADER_DIR) + "/" + name + ".spv");
                return device->CreateShaderModule({std::span<const uint32_t>(words)});
            };
            struct BloomParams { float texel[2]; float threshold; float knee; float strength; float intensity; };
            const uint32_t kBloomPC = sizeof(BloomParams);

            auto makeBloomPipe = [&](rhi::IShaderModule* fs, rhi::Format colorFmt) {
                rhi::GraphicsPipelineDesc d;
                d.vertex = postVsM.get(); d.fragment = fs;
                d.colorFormat = colorFmt;
                d.depthTest = false; d.usesFrameUniforms = false;
                d.usesTexture = true; d.fullscreen = true;
                d.fragmentPushConstants = true; d.pushConstantSize = kBloomPC;
                return device->CreateGraphicsPipeline(d);
            };
            auto prefilterFs = loadFs("bloom_prefilter.frag.hlsl");
            auto downsampleFs = loadFs("bloom_downsample.frag.hlsl");
            auto upsampleFs  = loadFs("bloom_upsample.frag.hlsl");
            auto compositeFs = loadFs("bloom_composite.frag.hlsl");
            auto prefilterPipe = makeBloomPipe(prefilterFs.get(), kHdr);
            auto downsamplePipe = makeBloomPipe(downsampleFs.get(), kHdr);
            auto upsamplePipe  = makeBloomPipe(upsampleFs.get(), kHdr);
            // Composite writes the LDR swapchain.
            auto compositePipe = makeBloomPipe(compositeFs.get(), device->Swapchain().ColorFormat());

            // --- Render targets: HDR scene + a 5-level half-res HDR mip chain (down + up). ---
            auto rt = device->CreateRenderTarget(w, h, kHdr);
            auto shadowMap = device->CreateShadowMap(2048);
            device->SetShadowMap(*shadowMap);

            const int kMips = 5;
            std::vector<std::unique_ptr<rhi::IRenderTarget>> down, up;
            std::vector<uint32_t> mw(kMips), mh(kMips);
            for (int i = 0; i < kMips; ++i) {
                uint32_t dw = std::max(1u, w >> (i + 1));   // start at half res
                uint32_t dh = std::max(1u, h >> (i + 1));
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
                Vec3 up3 = math::cross(right, fwd);
                fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
                fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
                fd.camUp[0]=up3.x; fd.camUp[1]=up3.y; fd.camUp[2]=up3.z;
                fd.skyParams[0] = std::tan(0.5f * 1.04719755f);
                fd.skyParams[1] = aspect;
                fd.skyParams[2] = envMaxLod;
            }

            // Bloom tuning. threshold/knee in exposure-applied domain (intensity = exposure = 1.7).
            const float kExposure = 1.7f;
            const float kThreshold = 1.0f;
            const float kKnee = 0.6f;
            const float kUpStrength = 1.0f;     // coarse->fine accumulation gain inside the chain
            const float kBloomStrength = 0.14f; // bloom presence in the final composite
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

            // Prefilter: bright-pass the full-res scene into down[0] (half res). `texel` is the SCENE
            // (source) texel so the bilinear box samples the full-res footprint.
            graph.AddPass("prefilter", {rgScene}, {rgDown[0]},
                [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                    BloomParams p = mkPC(w, h, kBloomStrength);
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*prefilterPipe);
                    cmd.BindTexture(*rt);
                    cmd.PushConstants(&p, sizeof(p));
                    cmd.Draw(3);
                    cmd.EndRenderPass();
                });

            // Downsample chain: down[i] = downsample(down[i-1]). `texel` is the SOURCE (down[i-1]) size.
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

            // Coarsest "up" mip is just the coarsest downsample (no coarser level to combine).
            graph.AddPass("upTop", {rgDown[kMips - 1]}, {rgUp[kMips - 1]},
                [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                    BloomParams p = mkPC(mw[kMips - 1], mh[kMips - 1], 0.0f);
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*upsamplePipe);
                    // Seed up[top] = down[top]: bind down[top] as BOTH inputs; coarse strength 0 so
                    // the upsampled "coarse" term drops out and only the fine down[top] remains.
                    cmd.BindTexturePair(*down[kMips - 1], *down[kMips - 1]);
                    cmd.PushConstants(&p, sizeof(p));
                    cmd.Draw(3);
                    cmd.EndRenderPass();
                });

            // Upsample/combine: up[i] = down[i] + tent_upsample(up[i+1]) * strength. `texel` = COARSE
            // (up[i+1]) source size. Goes from second-coarsest down to the finest (down[0]).
            for (int i = kMips - 2; i >= 0; --i) {
                graph.AddPass("up" + std::to_string(i), {rgUp[i + 1], rgDown[i]}, {rgUp[i]},
                    [&, i](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                        BloomParams p = mkPC(mw[i + 1], mh[i + 1], kUpStrength);
                        cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                        cmd.BindPipeline(*upsamplePipe);
                        // primary = coarser accumulated (up[i+1]); secondary = this level's down[i].
                        cmd.BindTexturePair(*up[i + 1], *down[i]);
                        cmd.PushConstants(&p, sizeof(p));
                        cmd.Draw(3);
                        cmd.EndRenderPass();
                    });
            }

            // Composite: HDR scene + bloom (up[0]) -> tonemap -> swapchain.
            graph.AddPass("composite", {rgScene, rgUp[0]}, {rgSwap},
                [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                    BloomParams p = mkPC(w, h, kBloomStrength);
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*compositePipe);
                    cmd.BindTexturePair(*rt, *up[0]);
                    cmd.PushConstants(&p, sizeof(p));
                    cmd.Draw(3);
                    cmd.EndRenderPass();
                });

            device->CaptureNextFrame();
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(bloomShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — HDR bloom, %d mips\n",
                                    bloomShotPath, cw, ch2, kMips);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", bloomShotPath);
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

        // --- Full glTF scene-graph import showcase (--scene-shot): a self-contained capture path
        // that does NOT touch the default scene. Ground plane + procedural sky + the CesiumMilkTruck
        // imported via LoadGltfScene (node hierarchy walked to world transforms, one renderable per
        // primitive of every mesh-referencing node, deduped PBR materials), lit + shadowed. The same
        // wheels mesh is drawn at the FRONT and BACK positions purely from the composed node
        // transforms. One frame -> BMP -> exit. Reuses the Slice-P lit-PBR pipeline + BindMaterialPBR;
        // no golden-locked pipeline/shader is touched. -----------------------------------------------
        if (sceneShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // Lit-PBR pipeline (shared lit.vert + full-PBR fragment; 5-texture material set).
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
            pbrDesc.pbrMaterial = true;
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

            // Depth-only shadow pipeline (ground + every truck primitive casts).
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

            // Ground plane: flat checkerboard dielectric.
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
                        truck.instances.size(), truck.meshStorage.size(),
                        truck.materialStorage.size());

            // Orientation + placement. The asset's root "Yup2Zup" node rotates the Y-up authored
            // truck into Z-up, which lays it on its side in our Y-up engine; rotate -90deg about X to
            // stand it back upright on its wheels. Then fit the WHOLE scene (post-orientation AABB is
            // handled by composing: fit is computed on the imported world AABB, the orientation is
            // applied on top, so we recompute placement by wrapping the fit in the orientation). We
            // apply: world' = ground-fit * orient * instanceWorld, where ground-fit re-grounds after
            // the orientation. To keep it simple and robust we: (1) orient each instance, (2) fit the
            // oriented scene by recomputing its world AABB here.
            // The asset's "Yup2Zup" root node already lands the truck upright in our Y-up world
            // (imported world AABB: height/Y is the smallest extent, length along Z, width along X),
            // so no extra orientation fix is needed — the node-hierarchy transforms stand it on its
            // wheels directly. We still rotate about Y so the long side faces the camera nicely.
            Mat4 orient = Mat4::RotateY(2.1f);

            // Recompute the oriented scene AABB so the fit grounds the truck correctly after orient.
            float oMin[3] = { 1e30f,  1e30f,  1e30f};
            float oMax[3] = {-1e30f, -1e30f, -1e30f};
            {
                // Transform the original world AABB's 8 corners by orient.
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
                    for (int k = 0; k < 3; ++k) {
                        if (wp[k] < oMin[k]) oMin[k] = wp[k];
                        if (wp[k] > oMax[k]) oMax[k] = wp[k];
                    }
                }
            }
            // Build a fit (uniform scale + ground/centre) from the ORIENTED AABB.
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
            // Final per-instance placement = sceneFit * orient * instanceWorld.
            Mat4 placement = sceneFit * orient;

            // Camera + light + sky frame data (fixed, deterministic).
            const Vec3 eye{5.0f, 3.2f, 6.0f};
            const Vec3 center{0.0f, 1.0f, 0.0f};
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
                Vec3 sc{0.0f, 1.0f, 0.0f};
                Vec3 lightEye = sc - lightDir * 14.0f;
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
                Mat4 lightOrtho = Mat4::Ortho(-6.0f, 6.0f, -6.0f, 6.0f, 1.0f, 30.0f);
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

            Mat4 groundModel = Mat4::Scale({10.0f, 1.0f, 10.0f});

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
                    // Truck (full-PBR): iterate every imported instance with its own world + material.
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
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(sceneShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u)\n", sceneShotPath, cw, ch2);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", sceneShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- Skeletal-animation showcase (--skinning-shot): a self-contained capture path that does
        // NOT touch the default scene. Ground plane + procedural sky + the GPU-skinned Fox sampled at
        // animation "Survey", time 0.5s, lit + shadowed. One frame -> BMP -> exit. ----------------
        // Both --skinning-shot and --blend-shot (Slice X) drive this single capture path; they
        // differ ONLY in how the joint palette is computed (single-clip sample vs. cross-clip blend)
        // and which file they write to. Everything else (scene/camera/light/pipelines) is shared so
        // the two BMPs are directly comparable.
        const char* skinOrBlendPath = skinningShotPath ? skinningShotPath : blendShotPath;
        if (skinOrBlendPath) {
            const bool isBlend = (blendShotPath != nullptr);
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

            // Load the skinned Fox + build the joint palette.
            //   --skinning-shot: single-clip sample of "Survey" at t=0.5s.
            //   --blend-shot   : 50/50 cross-clip blend of "Walk" (t=0.3s) and "Run" (t=0.2s) via
            //                    anim::BlendAnimations. Walk+Run are both locomotion gaits, so their
            //                    midpoint is a clearly intermediate four-legged stance distinct from
            //                    either pure clip (Survey is near-static and would dominate a blend).
            hf::asset::SkinnedModel fox = hf::asset::LoadSkinnedGltfModel(*device, HF_FOX_MODEL_PATH);
            std::vector<Mat4> palette;
            if (isBlend) {
                const anim::Animation* walk = fox.FindAnimation("Walk");
                const anim::Animation* run  = fox.FindAnimation("Run");
                if (!walk && !fox.animations.empty()) walk = &fox.animations.front();
                if (!run) run = walk;
                if (walk && run)
                    palette = anim::BlendAnimations(fox.skeleton, *walk, 0.3f, *run, 0.2f, 0.5f);
                else
                    palette.assign(fox.skeleton.joints.size(), Mat4::Identity());
            } else {
                const anim::Animation* survey = fox.FindAnimation("Survey");
                if (!survey && !fox.animations.empty()) survey = &fox.animations.front();
                if (survey) palette = anim::SampleAnimation(fox.skeleton, *survey, 0.5f);
                else palette.assign(fox.skeleton.joints.size(), Mat4::Identity());
            }
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
                ok = WriteBMP(skinOrBlendPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u)\n", skinOrBlendPath, cw, ch2);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", skinOrBlendPath);
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
