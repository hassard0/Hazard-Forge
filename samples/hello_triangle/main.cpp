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
#include "game/roll_game.h"
#include "ui/text.h"
#include "render/render_graph.h"
#include "render/csm.h"
#include "render/spot.h"
#include "render/point_shadow.h"
#include "render/probe.h"
#include "render/clustered.h"
#include "render/taa.h"
#include "render/frustum.h"
#include "render/gpu_cull.h"
#include "debug/debug_draw.h"
#include "debug/debug_emitters.h"
#include "runtime/camera.h"
#include "runtime/clock.h"
#include "runtime/fly_camera_controller.h"
#include "runtime/input_state.h"
#include "runtime/play_state.h"
#include "runtime/hot_reload.h"
#include "runtime/parallel_record.h"
#include "editor/picking.h"
#include "editor/gizmo.h"
#include "editor/introspect.h"
// Slice AW (live runtime material authoring): in-process graph load + runtime dxc-subprocess compile
// + the live-swap controller. Pure host logic (no backend symbols) above the RHI seam.
#include "material/material_loader.h"
#include "material/runtime_compile.h"
#include "material/live_material.h"

#ifdef HF_HAS_EDITOR
#include "imgui.h"
#include "editor/imgui_renderer.h"
#include "editor/editor_panels.h"
#endif

#include <algorithm>
#include <array>
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
// FrameData (416 bytes; kFrameUboSize is 1024 so it fits — verified below). The trailing
// prevViewProj (Slice AP) is purely additive: it carries the PREVIOUS frame's (unjittered) view-proj
// so the TAA resolve can reproject history (identity for the static --taa-shot; the lit/shadow/sky
// shaders ignore it). 352 + 64 = 416 B << 1024.
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
    float prevViewProj[16];    // TAA (Slice AP): previous frame's unjittered view-proj for reprojection
};
static_assert(sizeof(FrameData) == 416, "FrameData layout drift");
static_assert(sizeof(FrameData) <= 1024, "FrameData must fit kFrameUboSize (1024)");

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

// --- HUD text overlay (Slice BA) -----------------------------------------------------------------
// A small reusable screen-space text overlay: builds the baked-font atlas as a sampled texture + a
// text pipeline (alphaBlend + cullNone, no depth, no frame uniforms, fragment push-const text color)
// ONCE, then draws one or more HUD lines as a single alpha-blended quad batch over the final target.
// Reuses the EXISTING alpha-blend + sampled-texture paths — no new RHI. The fragment push constant
// is the text color (float4 rgb + opacity).
using namespace hf;  // bring rhi::/ui:: into scope for the HUD helper (anon namespace, file scope)

struct HudLine { std::string text; float x; float y; float scale; float color[4]; };

struct HudOverlay {
    std::unique_ptr<rhi::IShaderModule> vs, fs;
    std::unique_ptr<rhi::IPipeline> pipeline;
    std::unique_ptr<rhi::ITexture> atlas;

    HudOverlay(rhi::IRHIDevice& device, rhi::Format colorFormat) {
        auto vsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/text.vert.hlsl.spv");
        auto fsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/text.frag.hlsl.spv");
        vs = device.CreateShaderModule({std::span<const uint32_t>(vsW)});
        fs = device.CreateShaderModule({std::span<const uint32_t>(fsW)});

        std::vector<uint8_t> px((size_t)ui::kAtlasW * ui::kAtlasH * 4);
        ui::BuildFontAtlas(px.data(), ui::kAtlasW, ui::kAtlasH);
        rhi::TextureDesc td;
        td.width = ui::kAtlasW; td.height = ui::kAtlasH; td.format = rhi::Format::RGBA8_UNorm;
        td.data = px.data(); td.dataSize = px.size();
        atlas = device.CreateTexture(td);

        rhi::GraphicsPipelineDesc pd;
        pd.vertex = vs.get(); pd.fragment = fs.get();
        pd.vertexLayout.stride = sizeof(ui::TextVertex);
        pd.vertexLayout.attributes = {
            {0, rhi::Format::RG32_Float, (uint32_t)offsetof(ui::TextVertex, posPx)},
            {1, rhi::Format::RG32_Float, (uint32_t)offsetof(ui::TextVertex, uv)},
        };
        pd.colorFormat = colorFormat;
        pd.depthTest = false;             // overlay: no depth test/write
        pd.usesFrameUniforms = false;     // no per-frame set -> atlas binds at set 0
        pd.usesTexture = true;            // atlas at the material set (set 0)
        pd.alphaBlend = true;             // src_alpha / one_minus_src_alpha
        pd.cullNone = true;               // screen-space quads — don't back-face cull
        pd.pushConstantSize = sizeof(float) * 4;   // text color (rgb + opacity)
        pd.fragmentPushConstants = true;  // color is read in the fragment stage
        pipeline = device.CreateGraphicsPipeline(pd);
    }

    // Draw the given lines into the CURRENT (already-open) render pass. Builds one combined vertex
    // buffer per group of same-colored quads; each distinct color is one draw (one push constant).
    // The vertex buffers are returned so the caller keeps them alive until the frame submits.
    void Draw(rhi::IRHIDevice& device, rhi::ICommandBuffer& cmd, const std::vector<HudLine>& lines,
              int screenW, int screenH, std::vector<std::unique_ptr<rhi::IBuffer>>& keepAlive) {
        cmd.BindPipeline(*pipeline);
        cmd.BindTexture(*atlas);
        for (const HudLine& ln : lines) {
            std::vector<ui::TextVertex> verts;
            int quads = ui::LayoutText(ln.text, ln.x, ln.y, ln.scale, screenW, screenH, verts);
            if (quads == 0) continue;
            rhi::BufferDesc bd;
            bd.size = verts.size() * sizeof(ui::TextVertex);
            bd.initialData = verts.data();
            bd.usage = rhi::BufferUsage::Vertex;
            auto vb = device.CreateBuffer(bd);
            cmd.PushConstants(ln.color, sizeof(float) * 4);
            cmd.BindVertexBuffer(*vb);
            cmd.Draw((uint32_t)verts.size());
            keepAlive.push_back(std::move(vb));
        }
    }
};

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
    const char* materialShotPath = nullptr;
    // Slice AW (live runtime material authoring): render the material showcase via the RUNTIME path
    // (in-process codegen -> dxc subprocess -> SPIR-V -> pipeline). For showcase.mat.json this is
    // byte-identical to --material-shot (same dxc -> same SPIR-V -> same image). An optional second
    // arg selects a different .mat.json (e.g. showcase2.mat.json).
    const char* materialLiveShotPath = nullptr;
    const char* materialLiveShotMat = nullptr;
    bool materialHotswapDryRun = false;  // --material-hotswap-dry-run: headless A->B swap proof.
    // Slice AZ (multi-material scene): three spheres, each a DISTINCT graph material (showcase /
    // showcase2 / showcase3), drawn one-per-material via the existing material pipeline.
    const char* materialMultiShotPath = nullptr;
    const char* sceneShotPath = nullptr;
    const char* iblShotPath = nullptr;
    const char* bloomShotPath = nullptr;
    const char* instancedShotPath = nullptr;
    const char* physicsShotPath = nullptr;
    const char* gameShotPath = nullptr;
    const char* hudShotPath = nullptr;       // --hud-shot <out.bmp> (Slice BA: text/HUD overlay)
    const char* gameHudShotPath = nullptr;   // --game-hud-shot <out.bmp> (Slice BA: game + score HUD)
    const char* debugShotPath = nullptr;
    const char* transparencyShotPath = nullptr;
    const char* ssaoShotPath = nullptr;
    const char* ssaoShotOffPath = nullptr;
    const char* ssaoDebugPath = nullptr;
    const char* capstoneShotPath = nullptr;
    const char* csmShotPath = nullptr;       // --csm-shot <out.bmp> (Slice AD: cascaded shadows)
    const char* spotShotPath = nullptr;      // --spot-shot <out.bmp> (Slice AE: spot-light shadows)
    const char* pointShotPath = nullptr;     // --point-shadow-shot <out.bmp> (Slice AF: omni point)
    const char* clusteredShotPath = nullptr; // --clustered-shot <out.bmp> (Slice AG: clustered lights)
    const char* ssrShotPath = nullptr;       // --ssr-shot <out.bmp> (Slice AH: screen-space reflections)
    const char* volumetricShotPath = nullptr; // --volumetric-shot <out.bmp> (Slice AJ: light shafts)
    const char* probeShotPath = nullptr;     // --probe-shot <out.bmp> (Slice AK: reflection/irradiance probes)
    const char* taaShotPath = nullptr;       // --taa-shot <out.bmp> (Slice AP: temporal anti-aliasing)
    const char* cullShotPath = nullptr;      // --cull-shot <out.bmp> (Slice AQ: frustum-culling viz)
    const char* gpuCullShotPath = nullptr;   // --gpu-cull-shot <out.bmp> (Slice AR: GPU cull + indirect)
    const char* mtShotPath = nullptr;        // --mt-shot <out.bmp> (Slice AU: multithreaded recording)
    int mtWorkers = 4;                       // --mt-shot ... --workers N (default 4)
    const char* commandsPath = nullptr;
    // Slice AA (interactive runtime): scripted-pose headless capture + live fly viewport.
    const char* cameraShotPath = nullptr;   // --camera-shot <yaw,pitch,x,y,z> <out.bmp>
    const char* cameraShotPose = nullptr;   // the "<yaw,pitch,x,y,z>" arg
    bool fly = false;                       // --fly: open the window and run the live fly loop
    bool flyDryRun = false;                 // --fly-dry-run: exercise the loop headlessly, then exit
    bool dumpScene = false;
    // --introspect [outpath]: build the default scene + a representative EngineState (camera +
    // lights) and write the full machine-readable engine-state JSON (editor::DescribeEngine) to
    // outpath, or stdout if omitted. Headless, no GPU, deterministic. The agent-facing OBSERVE side.
    bool introspect = false;
    const char* introspectPath = nullptr;
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
        } else if (std::strcmp(argv[i], "--material-shot") == 0 && i + 1 < argc) {
            // Slice AV: render one frame of the data-driven material-graph showcase (ground + skybox
            // + a sphere shaded by the build-time-generated fragment from showcase.mat.json: a
            // textured base with a fresnel rim), lit + shadowed, write a BMP, exit.
            materialShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--material-live-shot") == 0 && i + 1 < argc) {
            // Slice AW: render the material showcase via the RUNTIME compile path. <out> is required;
            // an OPTIONAL trailing <mat.json> (when the next argv is not another flag) selects the
            // material (default: assets/materials/showcase.mat.json).
            materialLiveShotPath = argv[i + 1];
            if (i + 2 < argc && argv[i + 2][0] != '-') { materialLiveShotMat = argv[i + 2]; ++i; }
            ++i;
        } else if (std::strcmp(argv[i], "--material-hotswap-dry-run") == 0) {
            // Slice AW: headless live-swap proof (no GUI): load showcase -> render hash, load
            // showcase2 -> render hash, assert each matches its golden + the swap happened cleanly.
            materialHotswapDryRun = true;
        } else if (std::strcmp(argv[i], "--material-multi-shot") == 0 && i + 1 < argc) {
            // Slice AZ: render one frame of the MULTI-material scene (three spheres in a row, each a
            // distinct graph material: showcase / showcase2 / showcase3) + ground + sky, lit +
            // shadowed, write a BMP, exit. One draw per material (bind that material's pipeline).
            materialMultiShotPath = argv[i + 1];
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
        } else if (std::strcmp(argv[i], "--game-shot") == 0 && i + 1 < argc) {
            // Slice AX: run the deterministic roll-a-ball game (game::MakeRollGame/StepGame over the
            // full game::ScriptedTrack at the engine fixed dt), then render ONE frame at a fixed
            // mid-track capture step — the ground + the player sphere + the remaining (uncollected)
            // pickups, lit + shadowed via the existing static-lit scene path. Prints the deterministic
            // winning game-state line. One BMP -> exit.
            gameShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--hud-shot") == 0 && i + 1 < argc) {
            // Slice BA: the standard lit + shadowed scene PLUS a deterministic screen-space HUD text
            // overlay ("HAZARD FORGE" + "SCORE: 0" + a fixed stat line — NO clock), drawn as an
            // alpha-blended quad batch over the final target through the new text pipeline. One BMP.
            hudShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--game-hud-shot") == 0 && i + 1 < argc) {
            // Slice BA: the AX game scene (identical to --game-shot) PLUS a "SCORE: N" HUD overlay
            // (N from the deterministic GameState at the capture step). Its OWN golden; --game-shot's
            // game.png stays byte-identical. One BMP -> exit.
            gameHudShotPath = argv[i + 1];
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
        } else if (std::strcmp(argv[i], "--spot-shot") == 0 && i + 1 < argc) {
            // Slice AE: spot-light shadow showcase. A ground plane + a few cubes/spheres lit
            // primarily by ONE spot light angled down so it casts a clear cone of light with SHARP
            // shadows of the objects within the cone (darkness outside the cone). A single 2048
            // PERSPECTIVE shadow map (FOV = 2*outerCone) is rendered, then the scene shaded by
            // lit_spot (cone smoothstep + distance falloff + 3x3 PCF spot shadow). One BMP -> exit.
            // New golden; the single-shadow/CSM paths/shaders/goldens are untouched.
            spotShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--point-shadow-shot") == 0 && i + 1 < argc) {
            // Slice AF: omnidirectional point-light shadow showcase. A point light hovers among a
            // RING of cubes/spheres on a ground plane with a back wall, casting shadows RADIALLY
            // OUTWARD in every direction. The scene is rendered from the light through 6 cube faces
            // (FOV=90, aspect=1) into a 3x2 atlas (1024 tiles in a 3072 map), then shaded by
            // lit_point (dominant-axis face select + per-face atlas PCF + distance falloff). One BMP
            // -> exit. New golden; the single-shadow/CSM/spot paths/shaders/goldens are untouched.
            pointShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--clustered-shot") == 0 && i + 1 < argc) {
            // Slice AG: clustered / Forward+ lighting showcase. A ground plane + a few objects are lit
            // by HUNDREDS of deterministic point lights (a fixed grid of varied colors/radii). The
            // lights are culled CPU-side into a 16x9x24 cluster grid (render::clustered) producing
            // three storage buffers (clusters / lightIndices / lights); the lit_clustered fragment
            // computes each fragment's cluster and iterates ONLY that cluster's lights. One BMP ->
            // exit. New golden; all existing lit/shadow paths/shaders/goldens are untouched.
            clusteredShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--ssr-shot") == 0 && i + 1 < argc) {
            // Slice AH: screen-space reflections showcase. A flat reflective checkerboard FLOOR with
            // several distinct colored objects (cubes + spheres) sitting ON it, lit + shadowed, rendered
            // into an HDR RGBA16F target PLUS a view-space normal+linear-depth g-buffer (reusing the
            // SSAO gbuffer shaders). An SSR pass ray-marches the depth buffer to produce mirror-like
            // reflections of the objects on the floor, then a composite blends + tonemaps. One BMP ->
            // exit. New golden; existing lit/ssao/bloom paths/shaders/goldens are untouched.
            ssrShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--volumetric-shot") == 0 && i + 1 < argc) {
            // Slice AJ: volumetric fog / light shafts. A directional light at a grazing angle streams
            // BETWEEN shadow-casting occluders (a slotted wall + pillars), fog filling the air. The
            // scene renders to an HDR RT + the SSAO view-space depth g-buffer; a fullscreen volumetric
            // pass ray-marches the view ray sampling the directional shadow map per step, accumulating
            // Henyey-Greenstein-weighted in-scattering (god rays) with Beer-Lambert extinction; a
            // composite adds it over the scene + tonemaps. One BMP -> exit. New golden; existing
            // lit/ssao/ssr paths/shaders/goldens untouched.
            volumetricShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--probe-shot") == 0 && i + 1 < argc) {
            // Slice AK: reflection + irradiance PROBE showcase. A Cornell-style box room (red left
            // wall, green right wall, neutral floor/ceiling/back) is baked from a fixed probe at the
            // room center into a single RGBA16F cube atlas (reflection block + cosine-convolved
            // irradiance block). A metallic sphere + a matte box inside are shaded by lit_probe so the
            // sphere REFLECTS the colored walls (local, not the sky) and the matte box picks up red/
            // green color BLEED from the irradiance. One BMP -> exit. New golden; existing paths/
            // shaders/goldens untouched.
            probeShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--taa-shot") == 0 && i + 1 < argc) {
            // Slice AP: temporal anti-aliasing showcase. Renders a FIXED N=8 accumulation frames over
            // a static lit + shadowed scene, advancing only the deterministic Halton(2,3) sub-pixel
            // jitter index 0..7; each frame's jittered HDR scene is blended into a neighborhood-clamped
            // history (taa_resolve.frag) and the 8th resolved frame is tonemapped + captured. The hard
            // silhouette edges of a single aliased frame resolve into smooth anti-aliased edges. Static
            // scene/camera/jitter => two runs are bit-identical (golden). One BMP -> exit. New golden;
            // existing paths/shaders/image goldens untouched.
            taaShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--cull-shot") == 0 && i + 1 < argc) {
            // Slice AQ: frustum-culling visualization. Renders a deterministic multi-object scene from
            // a pulled-back OVERVIEW camera, draws the actual (narrower) RENDER camera's frustum as
            // debug lines, and draws each object's bounding sphere GREEN (the render camera keeps it)
            // or RED (the render camera culls it). Prints a {drawn, culled, total} stat. The cull
            // partition is the conservative bounding-sphere test (engine/render/frustum.h) the render
            // submission path uses — render-invariant. One BMP -> exit. New golden (cull.png).
            cullShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--gpu-cull-shot") == 0 && i + 1 < argc) {
            // Slice AR: GPU-driven culling + indirect draw. A compute shader frustum-culls a
            // 1024-instance cube grid, ORDER-compacts the survivors into a second instance buffer, and
            // writes the indirect draw-args (instanceCount = survivor count) — then ONE
            // DrawIndexedIndirect renders exactly the survivors, the count decided on the GPU. Prints
            // `gpu-cull: {drawn:<gpu>, cpuRef:<cpu>, total:1024}` and ASSERTS gpu==cpuRef (the
            // exact-count proof against engine/render/gpu_cull.h). One BMP -> exit. New golden.
            gpuCullShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--mt-shot") == 0 && i + 1 < argc) {
            // Slice AU: multithreaded command recording. A draw-heavy grid of NON-instanced DISTINCT
            // draws (each a separate recorded command) is recorded across N worker threads — each
            // worker records a SECONDARY command buffer over its CONTIGUOUS draw sub-range; the primary
            // executes them in worker-index order. Determinism oracle: --workers 1 and --workers 4
            // produce BYTE-IDENTICAL captures. Prints `mt: {workers:N, draws:D, secondaries:N}`. One
            // BMP -> exit. New golden mt.png.
            mtShotPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            mtWorkers = std::atoi(argv[i + 1]);
            if (mtWorkers < 1) mtWorkers = 1;
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
        } else if (std::strcmp(argv[i], "--introspect") == 0) {
            introspect = true;
            // Optional output path: consume the next arg only if it isn't another flag.
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                introspectPath = argv[i + 1];
                ++i;
            }
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

        // --- Spot-light shadow showcase (--spot-shot, Slice AE). A ground plane + a few cubes and a
        // sphere lit primarily by ONE spot light mounted above and angled down at the cluster, so it
        // casts a clear cone of light with SHARP shadows of the objects within the cone, and darkness
        // outside the cone. The spot's shadow is a single 2048 PERSPECTIVE map (FOV = 2*outerCone),
        // rendered via the reused CSM depth-only caster (spotViewProj in the push constant), then the
        // scene is shaded by lit_spot: cone smoothstep + distance falloff + 3x3 PCF spot shadow. The
        // directional light is DIM so the spot dominates. One BMP -> exit. The single-shadow/CSM
        // paths/shaders/goldens are entirely untouched. ----------------------------------------------
        if (spotShotPath) {
            using math::Mat4; using math::Vec3;
            namespace spotns = hf::render::spot;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // --- Spot FrameData layout (matches shaders/lit_spot.frag). 304 bytes < kFrameUboSize. ---
            struct SpotFrameData {
                float viewProj[16];     //   0
                float lightDir[4];      //  64
                float lightColor[4];    //  80
                float viewPos[4];       //  96
                float spotViewProj[16]; // 112 -> ends 176
                float spotPos[4];       // 176
                float spotDir[4];       // 192
                float spotColor[4];     // 208
                float spotParams[4];    // 224  x=cosInner,y=cosOuter,z=range,w=intensity
                float camFwd[4];        // 240
                float camRight[4];      // 256
                float camUp[4];         // 272
                float skyParams[4];     // 288
            };
            static_assert(sizeof(SpotFrameData) == 304, "Spot FrameData layout");

            // === Spot config. ===
            const uint32_t kShadowSize = 2048;
            const float kInnerCone = 0.28f;     // ~16deg half-angle (full brightness)
            const float kOuterCone = 0.40f;     // ~23deg half-angle (cone edge)
            const float kSpotNear  = 0.5f;
            const float kSpotRange  = 34.0f;
            const Vec3  kSpotPos{0.0f, 13.0f, 5.0f};
            const Vec3  kSpotTarget{0.0f, 0.0f, -2.0f};
            const Vec3  kSpotDir = math::normalize(kSpotTarget - kSpotPos);

            // === Shaders. Reuse lit.vert + the CSM depth-only caster (spotViewProj in push const). ===
            auto litVsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto spotFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_spot.frag.hlsl.spv");
            auto shVsWords   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow_csm.vert.hlsl.spv");
            auto shadowFsW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto litVs    = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto spotFs   = device->CreateShaderModule({std::span<const uint32_t>(spotFsWords)});
            auto shVs     = device->CreateShaderModule({std::span<const uint32_t>(shVsWords)});
            auto shadowFs = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});

            const rhi::Format kSwap = device->Swapchain().ColorFormat();

            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = spotFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = kSwap;
            litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;  // model mat4 + material(metallic,rough)
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            rhi::GraphicsPipelineDesc shDesc;
            shDesc.vertex = shVs.get(); shDesc.fragment = shadowFs.get();
            shDesc.vertexLayout = scene::MeshVertexLayout();
            shDesc.depthTest = true; shDesc.depthOnly = true; shDesc.usesFrameUniforms = true;
            shDesc.pushConstantSize = sizeof(float) * 32;   // spotViewProj(16) + model(16)
            auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

            // === Shadow map + meshes + textures. ===
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

            // === Scene layout: ground + a small cluster of casters under the cone. ===
            const Mat4 groundModel = Mat4::Scale({40.0f, 1.0f, 40.0f});
            struct Caster { Mat4 model; const scene::Mesh* mesh; float metallic; float rough; };
            std::vector<Caster> casters;
            auto box = [&](float x, float z, float s, float rot) {
                casters.push_back({Mat4::Translate({x, 0.5f * s, z}) * Mat4::RotateY(rot)
                                   * Mat4::Scale({s, s, s}), &cube, 0.0f, 0.8f});
            };
            auto ball = [&](float x, float z, float s) {
                casters.push_back({Mat4::Translate({x, 0.5f * s, z}) * Mat4::Scale({s, s, s}),
                                   &sphere, 0.05f, 0.45f});
            };
            box(-2.6f,  0.0f, 2.0f,  0.5f);
            ball( 2.4f, -1.0f, 2.2f);
            box( 0.2f, -4.0f, 2.4f, -0.3f);
            ball(-1.4f,  2.6f, 1.6f);

            // === Fixed deterministic camera. ===
            const Vec3 eye{0.0f, 7.0f, 16.0f};
            const Vec3 center{0.0f, 0.8f, -2.0f};
            const float fovY = 1.04719755f;  // 60deg
            Vec3 fwd   = math::normalize(center - eye);
            Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
            Vec3 up3   = math::cross(right, fwd);
            const float tanHalf = std::tan(0.5f * fovY);

            // Spot perspective light matrix (Vulkan clip space; the shader's V-flip is MSL-only).
            Mat4 spotVP = spotns::SpotViewProj(kSpotPos, kSpotDir, kOuterCone, kSpotNear, kSpotRange);

            SpotFrameData fd{};
            {
                Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
                Mat4 proj = Mat4::Perspective(fovY, aspect, 0.1f, 100.0f);
                Mat4 vp = proj * view;
                for (int k = 0; k < 16; ++k) fd.viewProj[k] = vp.m[k];
                for (int k = 0; k < 16; ++k) fd.spotViewProj[k] = spotVP.m[k];
                // DIM directional fill (the spot is the dominant light).
                Vec3 ld = math::normalize(Vec3{-0.4f, -0.85f, -0.3f});
                fd.lightDir[0]=ld.x; fd.lightDir[1]=ld.y; fd.lightDir[2]=ld.z;
                fd.lightColor[0]=0.20f; fd.lightColor[1]=0.21f; fd.lightColor[2]=0.24f; fd.lightColor[3]=1.0f;
                fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
                fd.spotPos[0]=kSpotPos.x; fd.spotPos[1]=kSpotPos.y; fd.spotPos[2]=kSpotPos.z; fd.spotPos[3]=1.0f;
                fd.spotDir[0]=kSpotDir.x; fd.spotDir[1]=kSpotDir.y; fd.spotDir[2]=kSpotDir.z;
                fd.spotColor[0]=1.0f; fd.spotColor[1]=0.96f; fd.spotColor[2]=0.85f; fd.spotColor[3]=1.0f;
                fd.spotParams[0]=std::cos(kInnerCone); fd.spotParams[1]=std::cos(kOuterCone);
                fd.spotParams[2]=kSpotRange; fd.spotParams[3]=9.0f;   // intensity
                fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
                fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
                fd.camUp[0]=up3.x; fd.camUp[1]=up3.y; fd.camUp[2]=up3.z;
                fd.skyParams[0]=tanHalf; fd.skyParams[1]=aspect;
            }

            render::RenderGraph graph;
            render::RgResource rgShadow = graph.ImportTarget(
                "spotShadow", render::RgResourceKind::ShadowMap, *shadowMap);
            render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

            // --- Shadow pass: render every caster + the ground into the perspective spot map. ---
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

            // --- Scene pass straight to the swapchain: ground + casters, shaded by lit_spot. ---
            graph.AddPass("spotScene", {rgShadow}, {rgSwap},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(SpotFrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0.04f, 0.05f, 0.07f, 1});  // dim night sky
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

            device->CaptureNextFrame();
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(spotShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — spot light: 1 perspective %ux%u shadow map, "
                                    "%zu casters\n", spotShotPath, cw, ch2, kShadowSize, kShadowSize,
                                    casters.size());
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", spotShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels (spot)\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- Omnidirectional point-light shadow showcase (--point-shadow-shot, Slice AF). A point
        // light hovers a few units above a ground plane, surrounded by a RING of cubes and spheres at
        // different azimuths plus a back wall, so it casts shadows RADIALLY OUTWARD in every direction
        // (objects on the left shadow leftward, right ones rightward, front/back too). The scene is
        // rendered from the light through 6 cube faces (Perspective 90deg, aspect 1) into ONE 3072
        // shadow ATLAS — a 3x2 grid of 1024 tiles — via SetViewport per face (exactly like CSM's N
        // cascades). The scene is then shaded by lit_point: per-fragment dominant-axis face selection
        // + that face's atlas-tile PCF + distance falloff. A DIM directional fill keeps shadows from
        // pure black. One BMP -> exit. The single-shadow/CSM/spot paths/shaders/goldens are untouched.
        if (pointShotPath) {
            using math::Mat4; using math::Vec3;
            namespace ptns = hf::render::point_shadow;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // --- Point FrameData layout (matches shaders/lit_point.frag). 608 bytes < kFrameUboSize. ---
            struct PointFrameData {
                float viewProj[16];     //   0
                float lightDir[4];      //  64  dim directional fill
                float lightColor[4];    //  80
                float viewPos[4];       //  96
                float faceVP[6][16];    // 112  6 face view-projs (384B) -> ends 496
                float ptPos[4];         // 496  xyz pos, w=range
                float ptColor[4];       // 512  rgb color, w=intensity
                float atlasParams[4];   // 528  x=tilesPerRow, y=tilesPerCol, z=1/atlasSize, w=near
                float camFwd[4];        // 544
                float camRight[4];      // 560
                float camUp[4];         // 576
                float skyParams[4];     // 592
            };
            static_assert(sizeof(PointFrameData) == 608, "Point FrameData layout");

            // === Atlas config: 3x2 grid of 1024 tiles in a 3072 square map. ===
            const uint32_t kAtlas    = 3072;
            const uint32_t kTile     = 1024;
            const int   kTilesPerRow = 3;       // 3x2 grid -> 6 faces
            const int   kTilesPerCol = 2;
            const float kPtNear  = 0.1f;
            const float kPtRange = 30.0f;
            const Vec3  kPtPos{0.0f, 3.0f, 0.0f};   // light hovers above the ring center

            // === Shaders. Reuse lit.vert + the CSM depth-only caster (faceVP in push const). ===
            auto litVsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto ptFsWords   = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_point.frag.hlsl.spv");
            auto shVsWords   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow_csm.vert.hlsl.spv");
            auto shadowFsW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto litVs    = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto ptFs     = device->CreateShaderModule({std::span<const uint32_t>(ptFsWords)});
            auto shVs     = device->CreateShaderModule({std::span<const uint32_t>(shVsWords)});
            auto shadowFs = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});

            const rhi::Format kSwap = device->Swapchain().ColorFormat();

            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = ptFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = kSwap;
            litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;  // model mat4 + material(metallic,rough)
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            rhi::GraphicsPipelineDesc shDesc;
            shDesc.vertex = shVs.get(); shDesc.fragment = shadowFs.get();
            shDesc.vertexLayout = scene::MeshVertexLayout();
            shDesc.depthTest = true; shDesc.depthOnly = true; shDesc.usesFrameUniforms = true;
            shDesc.pushConstantSize = sizeof(float) * 32;   // faceVP(16) + model(16)
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

            // === Scene: ground + a RING of casters around the light + a back wall. The light sits at
            // the ring center, so each object's shadow points radially AWAY from the center — proving
            // the 6 cube faces all contribute (left objs shadow left, right right, front/back too). ===
            const Mat4 groundModel = Mat4::Scale({40.0f, 1.0f, 40.0f});
            // Back wall behind the ring (-Z), faces +Z so the -Z cube face shadows onto it.
            const Mat4 wallModel = Mat4::Translate({0.0f, 4.0f, -9.0f}) * Mat4::Scale({12.0f, 8.0f, 0.5f});
            struct Caster { Mat4 model; const scene::Mesh* mesh; float metallic; float rough; };
            std::vector<Caster> casters;
            auto box = [&](float x, float z, float s, float rot) {
                casters.push_back({Mat4::Translate({x, 0.5f * s, z}) * Mat4::RotateY(rot)
                                   * Mat4::Scale({s, s, s}), &cube, 0.0f, 0.8f});
            };
            auto ball = [&](float x, float z, float s) {
                casters.push_back({Mat4::Translate({x, 0.5f * s, z}) * Mat4::Scale({s, s, s}),
                                   &sphere, 0.05f, 0.45f});
            };
            // Ring at radius ~4.5 around the light's ground projection (0,0). Each object casts its
            // shadow outward along its own azimuth -> different cube faces.
            box( 4.5f,  0.0f, 1.2f,  0.4f);   // +X  -> +X face
            box(-4.5f,  0.0f, 1.2f, -0.4f);   // -X  -> -X face
            ball( 0.0f,  4.5f, 1.3f);          // +Z
            ball( 0.0f, -4.5f, 1.3f);          // -Z
            box( 3.2f,  3.2f, 1.1f,  0.8f);   // +X+Z diagonal
            box(-3.2f,  3.2f, 1.1f, -0.8f);   // -X+Z
            ball( 3.2f, -3.2f, 1.1f);          // +X-Z
            ball(-3.2f, -3.2f, 1.1f);          // -X-Z

            // === Fixed deterministic camera looking down at the ring from the +Z+Y front. ===
            const Vec3 eye{0.0f, 9.0f, 13.0f};
            const Vec3 center{0.0f, 1.0f, 0.0f};
            const float fovY = 1.04719755f;  // 60deg
            Vec3 fwd   = math::normalize(center - eye);
            Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
            Vec3 up3   = math::cross(right, fwd);
            const float tanHalf = std::tan(0.5f * fovY);

            // === 6 cube-face view-projs (Vulkan clip space; the shader's V-flip is MSL-only). ===
            auto faceVPs = ptns::FaceViewProjs(kPtPos, kPtNear, kPtRange);

            PointFrameData fd{};
            {
                Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
                Mat4 proj = Mat4::Perspective(fovY, aspect, 0.1f, 100.0f);
                Mat4 vp = proj * view;
                for (int k = 0; k < 16; ++k) fd.viewProj[k] = vp.m[k];
                for (int fi = 0; fi < ptns::kFaces; ++fi)
                    for (int k = 0; k < 16; ++k) fd.faceVP[fi][k] = faceVPs[fi].m[k];
                // DIM directional fill (the point light dominates).
                Vec3 ld = math::normalize(Vec3{-0.3f, -0.9f, -0.25f});
                fd.lightDir[0]=ld.x; fd.lightDir[1]=ld.y; fd.lightDir[2]=ld.z;
                fd.lightColor[0]=0.13f; fd.lightColor[1]=0.14f; fd.lightColor[2]=0.17f; fd.lightColor[3]=1.0f;
                fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
                fd.ptPos[0]=kPtPos.x; fd.ptPos[1]=kPtPos.y; fd.ptPos[2]=kPtPos.z; fd.ptPos[3]=kPtRange;
                fd.ptColor[0]=1.0f; fd.ptColor[1]=0.93f; fd.ptColor[2]=0.82f; fd.ptColor[3]=14.0f; // intensity
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
            render::RgResource rgSwap = graph.ImportSwapchain("swapchain");

            // --- Shadow pass: clear the atlas once, then for each cube face SetViewport(tile) and draw
            // every caster (+ ground + wall) with that face's view-proj pushed as a constant. ---
            graph.AddPass("pointShadow", {}, {rgShadow},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(PointFrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*shadowPipeline);
                    for (int face = 0; face < ptns::kFaces; ++face) {
                        auto tile = ptns::FaceTile(face);
                        cmd.SetViewport((int32_t)(tile.col * kTile), (int32_t)(tile.row * kTile),
                                        kTile, kTile);
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

            // --- Scene pass straight to the swapchain: ground + wall + casters, shaded by lit_point. ---
            graph.AddPass("pointScene", {rgShadow}, {rgSwap},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(PointFrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0.03f, 0.04f, 0.06f, 1});  // dim night sky
                    cmd.BindPipeline(*litPipeline);
                    auto drawLit = [&](const Mat4& model, const scene::Mesh& mesh, float metallic,
                                       float rough) {
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

            device->CaptureNextFrame();
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(pointShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — point light: 6-face cube, %ux%u atlas "
                                    "(%dx%d tiles), %zu casters\n", pointShotPath, cw, ch2, kAtlas,
                                    kAtlas, kTilesPerRow, kTilesPerCol, casters.size());
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", pointShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels (point)\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- Reflection + irradiance PROBE showcase (--probe-shot, Slice AK). A Cornell-style box
        // room (red LEFT wall -X, green RIGHT wall +X, neutral floor/ceiling/back/front) is baked from
        // a fixed probe at the room center into ONE RGBA16F cube atlas:
        //   * REFLECTION block: the room rendered through the 6 cube faces (FOV 90, aspect 1) into a
        //     3x2 grid of 512 tiles (1536x1024), reusing the point-shadow cube-atlas machinery but
        //     rendering COLOR. (The dynamic hero objects are EXCLUDED to avoid recursion.)
        //   * IRRADIANCE block: a 3x2 grid of 64 tiles (192x128) below it, each a cosine-hemisphere
        //     convolution of the reflection block -> diffuse color bleed.
        // The atlas binds at the env slot (BindReflectionProbe). Hero objects inside — a METALLIC
        // SPHERE (reflects the red/green walls on its sides: LOCAL reflection, not the global sky) and
        // a MATTE box (picks up red bleed on its left, green on its right from the irradiance) — are
        // shaded by lit_probe. One BMP -> exit. New golden; existing paths/shaders/goldens untouched.
        if (probeShotPath) {
            using math::Mat4; using math::Vec3;
            namespace prb = hf::render::probe;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // --- lit_probe FrameData (matches shaders/lit_probe.frag). 624 bytes < kFrameUboSize. ---
            struct ProbeFrameData {
                float viewProj[16];    //   0
                float lightDir[4];     //  64
                float lightColor[4];   //  80
                float viewPos[4];      //  96
                float faceVP[6][16];   // 112  6 probe-face view-projs -> ends 496
                float probePos[4];     // 496
                float atlasParams[4];  // 512  x=reflTileU y=reflTileV z=irrTileU w=irrTileV
                float atlasParams2[4]; // 528  x=irrBlockV0 y=texelU z=texelV w=tilesPerRow
                float camFwd[4];       // 544
                float camRight[4];     // 560
                float camUp[4];        // 576
                float skyParams[4];    // 592
                float pad0[4];         // 608
            };
            static_assert(sizeof(ProbeFrameData) == 624, "Probe FrameData layout");

            const Vec3 kProbePos{0.0f, 2.0f, 0.0f};   // room center

            // === Shaders. lit.vert (scene) + lit_probe (probe-lit). CSM depth-less COLOR bake reuses
            //     lit.vert + lit.frag-equivalent: we bake with a dedicated cube-color pipeline that is
            //     just lit.vert + a flat-tint fragment. Reuse lit_probe.frag? No — recursion. We bake
            //     with lit.frag (procedural sky IBL) — but the room walls dominate via their tint, and
            //     the captured COLOR is what matters. To keep tints, the bake uses lit_probe.frag with
            //     NO probe bound is wrong. Simplest: bake with the SAME lit.vert + a tiny flat shader.
            //     We reuse lit.frag (Slice F) which honours the per-draw tint via vertex color * tex. ===
            auto litVsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto bakeVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/probe_bake.vert.hlsl.spv");
            auto bakeFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/probe_bake.frag.hlsl.spv");
            auto probeFsW    = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_probe.frag.hlsl.spv");
            auto postVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.vert.hlsl.spv");
            auto blitFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/probe_blit.frag.hlsl.spv");
            auto irrFsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/probe_irradiance.frag.hlsl.spv");
            auto postFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.frag.hlsl.spv");
            auto litVs   = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto bakeVs  = device->CreateShaderModule({std::span<const uint32_t>(bakeVsWords)});
            auto bakeFs  = device->CreateShaderModule({std::span<const uint32_t>(bakeFsWords)});
            auto probeFs = device->CreateShaderModule({std::span<const uint32_t>(probeFsW)});
            auto postVs  = device->CreateShaderModule({std::span<const uint32_t>(postVsWords)});
            auto blitFs  = device->CreateShaderModule({std::span<const uint32_t>(blitFsWords)});
            auto irrFs   = device->CreateShaderModule({std::span<const uint32_t>(irrFsWords)});
            auto postFs  = device->CreateShaderModule({std::span<const uint32_t>(postFsWords)});

            const rhi::Format kSwap = device->Swapchain().ColorFormat();
            const rhi::Format kHdr  = rhi::Format::RGBA16_Float;

            // Bake pipeline: probe_bake.vert (faceVP + model in the PUSH CONSTANT, like the CSM
            // caster) + probe_bake.frag (flat wall color) into an RGBA16F target. The per-face VP via
            // push constant lets all 6 faces render into their atlas tiles in ONE render pass via
            // SetViewport — a single shared per-frame UBO would otherwise make every face see the LAST
            // face's VP. No frame uniforms needed.
            rhi::GraphicsPipelineDesc bakeDesc;
            bakeDesc.vertex = bakeVs.get(); bakeDesc.fragment = bakeFs.get();
            bakeDesc.vertexLayout = scene::MeshVertexLayout();
            bakeDesc.colorFormat = kHdr;
            bakeDesc.depthTest = true; bakeDesc.usesFrameUniforms = false; bakeDesc.usesTexture = true;
            bakeDesc.pushConstantSize = sizeof(float) * 32;  // faceVP(16) + model(16)
            auto bakePipeline = device->CreateGraphicsPipeline(bakeDesc);

            // Blit pipeline (fullscreen): copy reflection tiles into the final atlas. Reads the source
            // RT via BindTexture (material set 0); per-tile srcRect via a fragment push constant.
            rhi::GraphicsPipelineDesc blitDesc;
            blitDesc.vertex = postVs.get(); blitDesc.fragment = blitFs.get();
            blitDesc.colorFormat = kHdr;
            blitDesc.depthTest = false; blitDesc.usesTexture = true; blitDesc.fullscreen = true;
            blitDesc.fragmentPushConstants = true;
            blitDesc.pushConstantSize = sizeof(float) * 4;   // srcRect (xy origin, zw size)
            auto blitPipeline = device->CreateGraphicsPipeline(blitDesc);

            // Irradiance convolution pipeline (fullscreen): reads the reflection RT (material set 1)
            // + the frame UBO (set 0, faceVP/atlas params) + a per-tile push constant (invFaceVP+pos).
            rhi::GraphicsPipelineDesc irrDesc;
            irrDesc.vertex = postVs.get(); irrDesc.fragment = irrFs.get();
            irrDesc.colorFormat = kHdr;
            irrDesc.depthTest = false; irrDesc.usesFrameUniforms = true; irrDesc.usesTexture = true;
            irrDesc.fullscreen = true; irrDesc.fragmentPushConstants = true;
            irrDesc.pushConstantSize = sizeof(float) * 20;   // invFaceVP(16) + probePos(4)
            auto irrPipeline = device->CreateGraphicsPipeline(irrDesc);

            // lit_probe pipeline: base lit PBR + probe IBL (set 3 env slot = probe atlas).
            rhi::GraphicsPipelineDesc probeDesc;
            probeDesc.vertex = litVs.get(); probeDesc.fragment = probeFs.get();
            probeDesc.vertexLayout = scene::MeshVertexLayout();
            probeDesc.colorFormat = kHdr;
            probeDesc.depthTest = true; probeDesc.usesFrameUniforms = true; probeDesc.usesTexture = true;
            probeDesc.usesEnvironment = true;   // reserve set 3 (the probe atlas)
            probeDesc.pushConstantSize = sizeof(float) * 20;  // model + material
            auto probePipeline = device->CreateGraphicsPipeline(probeDesc);

            // Post pipeline -> swapchain.
            rhi::GraphicsPipelineDesc postDesc;
            postDesc.vertex = postVs.get(); postDesc.fragment = postFs.get();
            postDesc.colorFormat = kSwap;
            postDesc.depthTest = false; postDesc.usesTexture = true; postDesc.fullscreen = true;
            auto postPipeline = device->CreateGraphicsPipeline(postDesc);

            // === Render targets. ===
            auto reflRT    = device->CreateRenderTarget(prb::kAtlasW, prb::kReflBlockH, kHdr); // 1536x1024
            auto probeAtlas = device->CreateRenderTarget(prb::kAtlasW, prb::kAtlasH, kHdr);    // 1536x1152
            auto sceneRT   = device->CreateRenderTarget(w, h, kHdr);
            // The frame set (set 0) has a shadow-map slot the lit pipelines declare; point it at a
            // dummy depth map so the set is fully valid (lit_probe never samples it — probe IBL only).
            auto dummyShadow = device->CreateShadowMap(64);
            device->SetShadowMap(*dummyShadow);

            // === Textures + meshes. ===
            const uint8_t whitePx[4] = {255, 255, 255, 255};
            auto whiteTex = device->CreateTexture({1, 1, rhi::Format::RGBA8_UNorm, whitePx, sizeof(whitePx)});
            const uint8_t flatNormalPx[4] = {128, 128, 255, 255};
            auto flatNormal = device->CreateTexture(
                {1, 1, rhi::Format::RGBA8_UNorm, flatNormalPx, sizeof(flatNormalPx)});
            scene::Mesh cube   = scene::Mesh::Cube(*device);
            scene::Mesh sphere = scene::Mesh::Sphere(*device);

            // === The room: 6 inward-facing walls (big thin cubes) forming a box around the probe. The
            // shared cube mesh has fixed per-face vertex colors, so each wall is TINTED via the push
            // constant (material.z/w unused -> we pass tint in a SEPARATE 4-float block; lit.frag uses
            // vertex color * tex, so to color a wall we scale by binding a 1x1 colored texture). To get
            // per-wall color without per-wall textures we instead create small 1x1 colored textures. ===
            auto colorTex = [&](float r, float g, float b) {
                uint8_t px[4] = {(uint8_t)(r * 255), (uint8_t)(g * 255), (uint8_t)(b * 255), 255};
                return device->CreateTexture({1, 1, rhi::Format::RGBA8_UNorm, px, sizeof(px)});
            };
            auto redTex     = colorTex(0.85f, 0.07f, 0.07f);   // left wall
            auto greenTex   = colorTex(0.10f, 0.75f, 0.12f);   // right wall
            auto neutralTex = colorTex(0.78f, 0.78f, 0.78f);   // floor/ceiling/back/front

            const float R = 6.0f;   // room half-extent
            const float T = 0.2f;   // wall thickness
            struct Wall { Mat4 model; rhi::ITexture* tex; };
            std::vector<Wall> walls = {
                {Mat4::Translate({-R, 2.0f, 0.0f}) * Mat4::Scale({T, 2*R, 2*R}), redTex.get()},     // -X red
                {Mat4::Translate({ R, 2.0f, 0.0f}) * Mat4::Scale({T, 2*R, 2*R}), greenTex.get()},   // +X green
                {Mat4::Translate({0.0f, 2.0f - R, 0.0f}) * Mat4::Scale({2*R, T, 2*R}), neutralTex.get()}, // floor
                {Mat4::Translate({0.0f, 2.0f + R, 0.0f}) * Mat4::Scale({2*R, T, 2*R}), neutralTex.get()}, // ceiling
                {Mat4::Translate({0.0f, 2.0f, -R}) * Mat4::Scale({2*R, 2*R, T}), neutralTex.get()},       // back
                {Mat4::Translate({0.0f, 2.0f,  R}) * Mat4::Scale({2*R, 2*R, T}), neutralTex.get()},       // front
            };

            // === Hero objects (NOT baked into the probe). ===
            Mat4 sphereModel = Mat4::Translate({-1.9f, 1.4f, 0.8f}) * Mat4::Scale({1.7f, 1.7f, 1.7f});
            // A tall WHITE matte box centered between the walls, rotated 45deg so the camera sees TWO
            // side faces at once: the left-facing face (toward the red -X wall) reads warm/red and the
            // right-facing face (toward the green +X wall) reads green — clear diffuse color bleed on a
            // near-white albedo.
            Mat4 boxModel    = Mat4::Translate({ 2.0f, 1.2f, 0.6f}) * Mat4::RotateY(0.785f)
                                 * Mat4::Scale({1.5f, 2.4f, 1.5f});

            // === Fixed deterministic camera, looking into the room from the front (+Z), slightly high. ===
            const Vec3 eye{0.0f, 2.4f, 9.0f};
            const Vec3 center{0.0f, 1.6f, 0.0f};
            const float fovY = 1.04719755f;  // 60deg
            Vec3 fwd   = math::normalize(center - eye);
            Vec3 right = math::normalize(math::cross(fwd, Vec3{0, 1, 0}));
            Vec3 up3   = math::cross(right, fwd);

            auto faceVPs = prb::FaceViewProjs(kProbePos);

            // === Fill the shared ProbeFrameData (used by lit_probe + irradiance). ===
            ProbeFrameData fd{};
            {
                Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
                Mat4 proj = Mat4::Perspective(fovY, aspect, 0.1f, 100.0f);
                Mat4 vp = proj * view;
                for (int k = 0; k < 16; ++k) fd.viewProj[k] = vp.m[k];
                for (int fi = 0; fi < prb::kFaces; ++fi)
                    for (int k = 0; k < 16; ++k) fd.faceVP[fi][k] = faceVPs[fi].m[k];
                Vec3 ld = math::normalize(Vec3{-0.3f, -0.85f, -0.35f});
                fd.lightDir[0]=ld.x; fd.lightDir[1]=ld.y; fd.lightDir[2]=ld.z;
                fd.lightColor[0]=0.55f; fd.lightColor[1]=0.55f; fd.lightColor[2]=0.58f; fd.lightColor[3]=1.0f;
                fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
                fd.probePos[0]=kProbePos.x; fd.probePos[1]=kProbePos.y; fd.probePos[2]=kProbePos.z; fd.probePos[3]=1.0f;
                fd.atlasParams[0]=(float)prb::kReflTile/(float)prb::kAtlasW; // reflTileU
                fd.atlasParams[1]=(float)prb::kReflTile/(float)prb::kAtlasH; // reflTileV
                fd.atlasParams[2]=(float)prb::kIrrTile/(float)prb::kAtlasW;  // irrTileU
                fd.atlasParams[3]=(float)prb::kIrrTile/(float)prb::kAtlasH;  // irrTileV
                fd.atlasParams2[0]=(float)prb::kReflBlockH/(float)prb::kAtlasH; // irrBlockV0
                fd.atlasParams2[1]=1.0f/(float)prb::kAtlasW;                    // texelU
                fd.atlasParams2[2]=1.0f/(float)prb::kAtlasH;                    // texelV
                fd.atlasParams2[3]=(float)prb::kTilesPerRow;                    // 3
                fd.camFwd[0]=fwd.x; fd.camFwd[1]=fwd.y; fd.camFwd[2]=fwd.z;
                fd.camRight[0]=right.x; fd.camRight[1]=right.y; fd.camRight[2]=right.z;
                fd.camUp[0]=up3.x; fd.camUp[1]=up3.y; fd.camUp[2]=up3.z;
            }

            const float kReflTileU = (float)prb::kReflTile / (float)prb::kAtlasW;
            const float kReflTileV_atlas = (float)prb::kReflTile / (float)prb::kAtlasH;

            render::RenderGraph graph;
            render::RgResource rgRefl  = graph.ImportTarget("reflRT", render::RgResourceKind::SceneColor, *reflRT);
            render::RgResource rgAtlas = graph.ImportTarget("probeAtlas", render::RgResourceKind::SceneColor, *probeAtlas);
            render::RgResource rgScene = graph.ImportTarget("sceneColor", render::RgResourceKind::SceneColor, *sceneRT);
            render::RgResource rgSwap  = graph.ImportSwapchain("swapchain");

            // --- Pass 1: bake the ROOM into the reflection RT, one wall set per cube face tile. The
            //     face VP rides in the push constant so all 6 faces share one render pass. ---
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

            // --- Pass 2: compose the FINAL atlas. Reflection tiles = passthrough copy of reflRT;
            //     irradiance tiles = cosine convolution of reflRT. Both read reflRT (not the atlas
            //     being written), so no read-after-write hazard. One pass, cleared once. ---
            graph.AddPass("probeCompose", {rgRefl}, {rgAtlas},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    // Reflection block: copy each face tile 1:1 into the top of the atlas.
                    cmd.BindPipeline(*blitPipeline);
                    cmd.BindTexture(*reflRT);
                    for (int face = 0; face < prb::kFaces; ++face) {
                        auto tile = prb::FaceTile(face);
                        cmd.SetViewport((int32_t)(tile.col * prb::kReflTile),
                                        (int32_t)(tile.row * prb::kReflTile),
                                        prb::kReflTile, prb::kReflTile);
                        // Source rect in reflRT (its own UV space; reflRT is exactly the refl block).
                        float srcRect[4] = {
                            (float)tile.col / (float)prb::kTilesPerRow,
                            (float)tile.row / (float)prb::kTilesPerCol,
                            1.0f / (float)prb::kTilesPerRow,
                            1.0f / (float)prb::kTilesPerCol,
                        };
                        cmd.PushConstants(srcRect, sizeof(srcRect));
                        cmd.Draw(3);
                    }
                    // Irradiance block: convolve reflRT into each small tile below the reflection block.
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

            // --- Pass 3: shade the room + hero objects with lit_probe, sampling the probe atlas. ---
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
                    // Room walls (matte dielectric) — they also pick up the probe, but read as walls.
                    // Skip the FRONT (+Z) wall (index 5) in the scene so the camera sees INTO the open
                    // box (Cornell-box style); it stays in the BAKE so reflections are complete.
                    for (size_t wi = 0; wi < walls.size(); ++wi)
                        if (wi != 5) drawProbe(walls[wi].model, cube, *walls[wi].tex, 0.0f, 0.95f);
                    // Hero: metallic sphere (reflects the colored walls) + matte box (color bleed).
                    drawProbe(sphereModel, sphere, *whiteTex, 1.0f, 0.08f);
                    drawProbe(boxModel,    cube,   *whiteTex, 0.0f, 0.9f);
                    cmd.EndRenderPass();
                });

            // --- Pass 4: post -> swapchain. ---
            graph.AddPass("post", {rgScene}, {rgSwap},
                [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*postPipeline);
                    cmd.BindTexture(*sceneRT);
                    cmd.Draw(3);
                    cmd.EndRenderPass();
                });

            device->CaptureNextFrame();
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);
            (void)kReflTileU; (void)kReflTileV_atlas;

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(probeShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — reflection+irradiance probe baked at "
                                    "(%.1f,%.1f,%.1f), %dx%d atlas\n", probeShotPath, cw, ch2,
                                    kProbePos.x, kProbePos.y, kProbePos.z, prb::kAtlasW, prb::kAtlasH);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", probeShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels (probe)\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- Clustered / Forward+ lighting showcase (--clustered-shot, Slice AG): a ground plane +
        // a few raised objects lit by HUNDREDS of deterministic point lights. The lights are culled
        // CPU-side (render::clustered) into a 16x9x24 cluster grid -> three storage buffers (clusters
        // / lightIndices / lights); the lit_clustered fragment computes each fragment's cluster and
        // iterates ONLY that cluster's lights. A correct clustered result is indistinguishable from
        // brute-force: a rich quilt of overlapping colored light pools with smooth falloff and NO
        // tile banding. One BMP -> exit. New golden; all existing paths/shaders/goldens untouched.
        if (clusteredShotPath) {
            using math::Mat4; using math::Vec3;
            namespace cl = hf::render::clustered;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // --- Clustered FrameData layout (matches shaders/lit_clustered.frag). 224 bytes < 1024. ---
            struct ClusteredFrameData {
                float viewProj[16];      //   0
                float view[16];          //  64  world -> view
                float lightDir[4];       // 128  dim directional fill
                float lightColor[4];     // 144
                float viewPos[4];        // 160  world-space camera
                float clusterParams[4];  // 176  x=CX y=CY z=CZ w=znear
                float clusterParams2[4]; // 192  x=zfar y=screenW z=screenH w=tanX
                float clusterParams3[4]; // 208  x=tanY (yzw unused)
            };
            static_assert(sizeof(ClusteredFrameData) == 224, "Clustered FrameData layout");

            // === Cluster grid + camera. Exponential z slices between znear..zfar over the frustum. ===
            // A 1x1x1 grid (HF_CLUSTERED_BRUTEFORCE env var) collapses every fragment to ONE cluster
            // holding ALL lights -> the SAME shader becomes a brute-force loop over all 192 lights.
            // Rendering both and confirming they match visually is the strongest correctness check.
            const bool bruteForce = (std::getenv("HF_CLUSTERED_BRUTEFORCE") != nullptr);
            const int   CX = bruteForce ? 1 : 16;
            const int   CY = bruteForce ? 1 : 9;
            const int   CZ = bruteForce ? 1 : 24;
            const float kNear = 0.5f, kFar = 90.0f;
            const float fovY = 1.04719755f;  // 60deg

            // Fixed deterministic camera: looking down the field of lights from a front-high vantage.
            const Vec3 eye{0.0f, 16.0f, 26.0f};
            const Vec3 center{0.0f, 0.0f, -2.0f};
            Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
            Mat4 proj = Mat4::Perspective(fovY, aspect, kNear, kFar);
            Mat4 vp = proj * view;
            cl::Grid grid = cl::MakeGrid(proj, kNear, kFar, (float)w, (float)h, CX, CY, CZ);

            // === MANY deterministic point lights: a 16x12 grid (192) hovering just above a wide floor.
            // Positions/colors/radii are derived purely from the index (NO rng), so the result is bit-
            // stable. Colors cycle through a vivid palette; radii vary so pools overlap richly. ===
            const int   LX = 16, LZ = 12;
            const int   kNumLights = LX * LZ;   // 192
            const float spanX = 34.0f, spanZ = 26.0f;
            const float lightY = 1.4f;          // just above the floor, so pools spread on the ground
            std::vector<cl::Light> worldLights;  // store world-space too (for the brute-force ref)
            std::vector<cl::Light> viewLights;   // view-space (what the culler + GPU buffer consume)
            worldLights.reserve(kNumLights);
            viewLights.reserve(kNumLights);
            for (int iz = 0; iz < LZ; ++iz) {
                for (int ix = 0; ix < LX; ++ix) {
                    int idx = iz * LX + ix;
                    float fx = ((float)ix / (float)(LX - 1) - 0.5f) * spanX;
                    float fz = ((float)iz / (float)(LZ - 1) - 0.5f) * spanZ - 2.0f;
                    // A vivid 6-hue palette cycled by index; some pairs mix to secondary colors.
                    static const float palette[6][3] = {
                        {1.00f, 0.18f, 0.20f}, // red
                        {0.20f, 1.00f, 0.30f}, // green
                        {0.25f, 0.40f, 1.00f}, // blue
                        {1.00f, 0.80f, 0.15f}, // amber
                        {0.90f, 0.20f, 1.00f}, // magenta
                        {0.15f, 0.95f, 0.95f}, // cyan
                    };
                    const float* c = palette[(ix * 2 + iz * 3) % 6];
                    // Radius varies 4.0..6.5 in a fixed pattern so neighbouring pools overlap.
                    float radius = 4.0f + ((idx * 7) % 6) * 0.5f;
                    float intensity = 2.6f;

                    cl::Light L{};
                    L.viewPos = {fx, lightY, fz};   // (world position; transformed below)
                    L.radius = radius;
                    L.color = {c[0], c[1], c[2]};
                    L.intensity = intensity;
                    worldLights.push_back(L);

                    // Transform the world position into VIEW space for culling + the GPU buffer.
                    float vw = 0.0f;
                    Vec3 vpos = math::MulPointDivide(view, L.viewPos, vw);  // affine: w stays 1
                    cl::Light Lv = L;
                    Lv.viewPos = vpos;
                    viewLights.push_back(Lv);
                }
            }

            // === CPU cull -> three GPU buffers (clusters / lightIndices / lights). ===
            cl::ClusterBuffers cb = cl::BuildClusters(grid, viewLights);
            // lightIndices may be empty in a degenerate case; pad to 1 so the buffer is non-zero size.
            if (cb.lightIndices.empty()) cb.lightIndices.push_back(0u);

            rhi::BufferDesc clusterDesc;
            clusterDesc.size = cb.clusters.size() * sizeof(cl::GpuCluster);
            clusterDesc.initialData = cb.clusters.data();
            clusterDesc.usage = rhi::BufferUsage::Storage;
            auto clusterBuf = device->CreateBuffer(clusterDesc);

            rhi::BufferDesc indexDesc;
            indexDesc.size = cb.lightIndices.size() * sizeof(uint32_t);
            indexDesc.initialData = cb.lightIndices.data();
            indexDesc.usage = rhi::BufferUsage::Storage;
            auto indexBuf = device->CreateBuffer(indexDesc);

            rhi::BufferDesc lightDesc;
            lightDesc.size = cb.lights.size() * sizeof(cl::GpuLight);
            lightDesc.initialData = cb.lights.data();
            lightDesc.usage = rhi::BufferUsage::Storage;
            auto lightBuf = device->CreateBuffer(lightDesc);

            // === Shaders. Reuse lit.vert; clustered fragment reads the three storage buffers. ===
            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto cluFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_clustered.frag.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto cluFs = device->CreateShaderModule({std::span<const uint32_t>(cluFsWords)});
            const rhi::Format kSwap = device->Swapchain().ColorFormat();

            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = cluFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = kSwap;
            litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
            litDesc.usesLightClusters = true;               // declares the set-3 cluster buffers
            litDesc.pushConstantSize = sizeof(float) * 20;  // model mat4 + material(metallic,rough)
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            // === Floor + a sparse set of raised objects (so pools wrap over geometry, not just flat). ===
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
            // A few low boxes + balls scattered deterministically, low enough to catch the light pools.
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
                // VERY dim directional fill so the colored point-light pools dominate the image.
                fd.lightColor[0]=0.05f; fd.lightColor[1]=0.05f; fd.lightColor[2]=0.06f; fd.lightColor[3]=1.0f;
                fd.viewPos[0]=eye.x; fd.viewPos[1]=eye.y; fd.viewPos[2]=eye.z; fd.viewPos[3]=1.0f;
                fd.clusterParams[0]=(float)CX; fd.clusterParams[1]=(float)CY;
                fd.clusterParams[2]=(float)CZ; fd.clusterParams[3]=kNear;
                fd.clusterParams2[0]=kFar; fd.clusterParams2[1]=(float)w; fd.clusterParams2[2]=(float)h;
                fd.clusterParams2[3]=grid.tanX;
                fd.clusterParams3[0]=grid.tanY;
            }

            render::RenderGraph graph;
            render::RgResource rgSwap = graph.ImportSwapchain("swapchain");
            graph.AddPass("clusteredScene", {}, {rgSwap},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(ClusteredFrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0.01f, 0.01f, 0.02f, 1});  // near-black night
                    cmd.BindPipeline(*litPipeline);
                    cmd.BindLightClusters(*clusterBuf, *indexBuf, *lightBuf);
                    auto drawLit = [&](const Mat4& model, const scene::Mesh& mesh, float metallic,
                                       float rough) {
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

            device->CaptureNextFrame();
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(clusteredShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — clustered Forward+: %d point lights, "
                                    "%dx%dx%d grid (%d clusters), %zu light-index entries\n",
                                    clusteredShotPath, cw, ch2, kNumLights, CX, CY, CZ,
                                    grid.clusterCount(), cb.lightIndices.size());
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", clusteredShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels (clustered)\n");
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

            // === Slice AM: debug-line pipeline for the LIVE editor gizmo overlay. Draws the selected
            // object's gizmo (emitted by editor::EmitGizmo into a DebugDraw line list) into the HDR
            // scene target after opaque geometry — the SAME path --gizmo-shot uses, here per-frame. The
            // line vertex buffer is rebuilt each frame from the current selection (see the fly loop).
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
            dbgD.colorFormat = kHdr;
            dbgD.lineList = true; dbgD.depthTest = true; dbgD.depthWrite = false;
            dbgD.usesFrameUniforms = true;
            auto debugLinePipeline = device->CreateGraphicsPipeline(dbgD);
            // Per-frame gizmo line buffer + vertex count, referenced inside recordFrame's scene pass.
            std::unique_ptr<rhi::IBuffer> gizmoLineBuffer;
            uint32_t gizmoVertCount = 0;

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
                        // Slice AM: live editor gizmo overlay — one LINE_LIST draw after opaque
                        // geometry (depth-tested, no depth write). Rebuilt each frame for the current
                        // selection; empty (gizmoVertCount==0) when nothing is selected.
                        if (gizmoVertCount > 0 && gizmoLineBuffer) {
                            cmd.BindPipeline(*debugLinePipeline);
                            cmd.BindVertexBuffer(*gizmoLineBuffer);
                            cmd.Draw(gizmoVertCount);
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

            // === Slice AM: LIVE editor state shared by --fly and --fly-dry-run. ===
            runtime::PlayState play;
            editor::Selection sel;            // starts unselected; a left-click picks.
            sel.index = -1; sel.mode = editor::GizmoMode::Translate;

            // The LIVE editable scene: real meshes (with bounds) so picking + gizmo drag act on actual
            // geometry. A couple of cubes + a sphere at distinct spots; Ctrl+S serializes THIS registry
            // via scene::DumpScene, and scene hot-reload re-LoadScenes into it.
            ecs::Registry editReg;
            scene::SceneResources editRes;
            editRes.AddMesh("cube", &cube);
            editRes.AddMesh("sphere", &sphere);
            editRes.AddMesh("plane", &plane);
            // Register the textures the scene schema can name so a hot-reloaded scene JSON resolves
            // them. A scene referencing a resource NOT registered here is rejected by LoadScene and the
            // current scene is kept (logged) — see the design spec's hot-reload scope note.
            editRes.AddTexture("checker", groundTex.get());
            editRes.AddTexture("flat_normal", flatNormal.get());
            auto buildDefaultEditScene = [&](ecs::Registry& reg) {
                auto addObj = [&](scene::Mesh* mesh, math::Vec3 pos, math::Vec3 scl) {
                    ecs::Entity e = reg.create();
                    scene::Transform t; t.position = pos; t.scale = scl;
                    reg.add(e, scene::TransformC{t});
                    reg.add(e, scene::MeshC{mesh});
                    reg.add(e, scene::MaterialC{});
                };
                addObj(&cube,   {-2.5f, 0.5f,  0.0f}, {1.0f, 1.0f, 1.0f});
                addObj(&sphere, { 1.5f, 1.0f,  1.5f}, {1.6f, 1.6f, 1.6f});
                addObj(&cube,   { 3.0f, 1.5f, -1.0f}, {0.6f, 3.0f, 0.6f});
            };
            buildDefaultEditScene(editReg);

            // World AABB of an entity: its mesh object-space bounds transformed by its Transform and
            // re-fit to an axis-aligned box (8-corner transform -> min/max). Used for ray picking.
            auto worldAabb = [](const scene::Transform& xf, const scene::MeshBounds& b) {
                math::Mat4 m = xf.Matrix();
                math::Vec3 mn{1e30f, 1e30f, 1e30f}, mx{-1e30f, -1e30f, -1e30f};
                for (int c = 0; c < 8; ++c) {
                    math::Vec3 corner{ (c & 1) ? b.max.x : b.min.x,
                                       (c & 2) ? b.max.y : b.min.y,
                                       (c & 4) ? b.max.z : b.min.z };
                    float wclip = 1.0f;
                    math::Vec3 wp = math::MulPointDivide(m, corner, wclip);
                    mn.x = std::min(mn.x, wp.x); mn.y = std::min(mn.y, wp.y); mn.z = std::min(mn.z, wp.z);
                    mx.x = std::max(mx.x, wp.x); mx.y = std::max(mx.y, wp.y); mx.z = std::max(mx.z, wp.z);
                }
                return math::Aabb{mn, mx};
            };
            // Snapshot the live entities (in view order) + their world AABBs for one pick query.
            auto snapshotPickList = [&](std::vector<ecs::Entity>& ents,
                                        std::vector<editor::PickAabb>& boxes) {
                ents.clear(); boxes.clear();
                for (auto [e, tc, mc] : editReg.view<scene::TransformC, scene::MeshC>()) {
                    ents.push_back(e);
                    scene::MeshBounds mb = mc.mesh ? mc.mesh->bounds() : scene::MeshBounds{};
                    boxes.push_back({worldAabb(tc.t, mb)});
                }
            };
            // Gizmo handle length for the selected entity, sized to its world AABB so it always reads.
            auto handleLenFor = [&](const editor::PickAabb& box) {
                math::Vec3 ext{box.box.max.x - box.box.min.x, box.box.max.y - box.box.min.y,
                               box.box.max.z - box.box.min.z};
                float reach = std::max({ext.x, ext.y, ext.z});
                return 1.2f + 0.6f * reach;
            };
            // Rebuild the gizmo line vertex buffer for the current selection (empty if nothing picked).
            auto rebuildGizmoLines = [&](int activeAxis) {
                gizmoVertCount = 0; gizmoLineBuffer.reset();
                if (!sel.Has()) return;
                std::vector<ecs::Entity> ents; std::vector<editor::PickAabb> boxes;
                snapshotPickList(ents, boxes);
                if (sel.index < 0 || sel.index >= (int)ents.size()) { sel.index = -1; return; }
                const scene::Transform& xf = editReg.get<scene::TransformC>(ents[sel.index]).t;
                float handleLen = handleLenFor(boxes[sel.index]);
                debug::DebugDraw dd;
                editor::EmitGizmo(dd, xf, sel.mode, handleLen, activeAxis);
                if (dd.VertexCount() == 0) return;
                rhi::BufferDesc bd;
                bd.size = (uint64_t)dd.Vertices().size() * sizeof(debug::LineVertex);
                bd.initialData = dd.Vertices().data();
                bd.usage = rhi::BufferUsage::Vertex;
                gizmoLineBuffer = device->CreateBuffer(bd);
                gizmoVertCount = (uint32_t)dd.VertexCount();
            };

            // --- Drag state for the gizmo (the per-frame prevRay/curRay manipulation). ---
            int   dragAxis = editor::kAxisNone;   // grabbed gizmo axis while left-dragging (-1 = none)
            math::Ray dragPrevRay{};              // last frame's cursor ray (seeded on grab -> no jump)

            // Cursor ray for an input snapshot: framebuffer px -> NDC -> world ray. (The px->NDC map is
            // editor::PixelToNdc, shared with the unit test.) Only meaningful while NOT mouse-looking.
            auto cursorRay = [&](const runtime::InputState& in) {
                editor::Ndc n = editor::PixelToNdc(in.mouseX, in.mouseY, (float)w, (float)h);
                return editor::ScreenRayThroughCamera(cam, n.x, n.y);
            };

            // One editor interaction step for an input snapshot: left-click picks (or grabs a gizmo
            // axis), left-drag manipulates the selected transform, left-release ends the drag. Returns
            // the active axis to brighten (for the gizmo render). Pure logic over picking/gizmo math —
            // identical in --fly (real cursor) and --fly-dry-run (synthetic InputState).
            auto editorInteract = [&](const runtime::InputState& in, bool leftEdge,
                                      bool leftDown) -> int {
                math::Ray ray = cursorRay(in);
                if (leftEdge) {
                    // Grab a gizmo axis if one is hovered for the current selection; else pick an entity.
                    if (sel.Has()) {
                        std::vector<ecs::Entity> ents; std::vector<editor::PickAabb> boxes;
                        snapshotPickList(ents, boxes);
                        if (sel.index >= 0 && sel.index < (int)ents.size()) {
                            const scene::Transform& xf = editReg.get<scene::TransformC>(ents[sel.index]).t;
                            float hl = handleLenFor(boxes[sel.index]);
                            int axis = editor::PickGizmoAxis(ray, xf, sel.mode, hl);
                            if (axis != editor::kAxisNone) {
                                dragAxis = axis;
                                dragPrevRay = ray;   // seed: first drag frame is a no-op (no jump)
                                return dragAxis;
                            }
                        }
                    }
                    // No axis grabbed -> pick the entity under the cursor (or clear on a miss).
                    std::vector<ecs::Entity> ents; std::vector<editor::PickAabb> boxes;
                    snapshotPickList(ents, boxes);
                    editor::PickResult r = editor::PickNearest(ray, boxes);
                    sel.index = r.index;   // -1 on a miss = deselect
                    dragAxis = editor::kAxisNone;
                    return editor::kAxisNone;
                }
                if (leftDown && dragAxis != editor::kAxisNone && sel.Has()) {
                    // Apply the per-frame drag: prevRay (last frame) -> curRay (this frame).
                    std::vector<ecs::Entity> ents; std::vector<editor::PickAabb> boxes;
                    snapshotPickList(ents, boxes);
                    if (sel.index >= 0 && sel.index < (int)ents.size()) {
                        scene::Transform& live = editReg.get<scene::TransformC>(ents[sel.index]).t;
                        live = editor::ApplyDrag(live, sel.mode, dragAxis, dragPrevRay, ray);
                    }
                    dragPrevRay = ray;
                    return dragAxis;
                }
                if (!leftDown) dragAxis = editor::kAxisNone;  // release ends the drag
                return dragAxis;
            };

            // --- Hot-reload: watch the active scene JSON + a representative shader .spv. ---
            runtime::FileWatcher watcher;
            const std::string scenePath  = HF_SCENE_PATH;
            const std::string shaderPath = std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv";
            watcher.Watch(scenePath);
            watcher.Watch(shaderPath);
            // Apply detected file changes. Scene reload is robust end-to-end; shader reload recreates
            // the lit fragment module (best-effort) — see the design spec for scope. Returns a summary.
            auto applyHotReload = [&](const std::vector<std::string>& changed) {
                for (const std::string& p : changed) {
                    if (p == scenePath) {
                        try {
                            ecs::Registry fresh;
                            scene::LoadScene(fresh, editRes, scenePath.c_str());
                            editReg = std::move(fresh);
                            sel.index = -1; dragAxis = editor::kAxisNone;
                            std::printf("[hot-reload] scene reloaded from %s\n", scenePath.c_str());
                        } catch (const std::exception& ex) {
                            std::printf("[hot-reload] scene reload FAILED (%s) — keeping current\n",
                                        ex.what());
                        }
                    } else if (p == shaderPath) {
                        try {
                            auto words = LoadSpirv(shaderPath);
                            litFs = device->CreateShaderModule({std::span<const uint32_t>(words)});
                            std::printf("[hot-reload] lit.frag.hlsl.spv shader module recreated "
                                        "(pipeline hot-swap deferred — see spec)\n");
                        } catch (const std::exception& ex) {
                            std::printf("[hot-reload] shader reload FAILED (%s)\n", ex.what());
                        }
                    }
                }
            };

            if (flyDryRun) {
                // Headless CI exercise (Slice AA + AM): feed synthetic input frames through the FULL
                // loop logic (camera + clock + ONE pick + ONE drag + a scene hot-reload) WITHOUT
                // presenting, then exit. Proves the live loop's editor logic path runs end-to-end with
                // no GUI window. The real mouse interaction in the window is manual-only.
                runtime::InputState in;
                int totalSteps = 0;
                math::Vec3 startPos = cam.position;

                // Frame 0: a synthetic left-click aimed at the sphere (entity 1). Project its center to
                // NDC, convert to a cursor pixel, and drive a left-press edge through editorInteract.
                {
                    math::Vec3 sphereCenter{1.5f, 1.0f, 1.5f};
                    float wclip = 0.0f;
                    math::Vec3 ndc = math::MulPointDivide(cam.ViewProj(), sphereCenter, wclip);
                    in.mouseX = (ndc.x * 0.5f + 0.5f) * (float)w;
                    in.mouseY = (-ndc.y * 0.5f + 0.5f) * (float)h;
                    editorInteract(in, /*leftEdge=*/true, /*leftDown=*/true);
                }
                bool picked = sel.Has();
                int pickedIndex = sel.index;

                // Frames: a synthetic translate-drag along +X of the selected entity (grab the X handle,
                // then drag the cursor to move it). Capture the before/after X to prove the drag moved it.
                math::Vec3 beforePos{};
                math::Vec3 afterPos{};
                if (picked) {
                    std::vector<ecs::Entity> ents; std::vector<editor::PickAabb> boxes;
                    snapshotPickList(ents, boxes);
                    beforePos = editReg.get<scene::TransformC>(ents[pickedIndex]).t.position;
                    // Grab the +X translate handle: aim a downward ray through (origin + X*handleLen/2).
                    sel.mode = editor::GizmoMode::Translate;
                    const scene::Transform& xf = editReg.get<scene::TransformC>(ents[pickedIndex]).t;
                    float hl = handleLenFor(boxes[pickedIndex]);
                    // Build a fake cursor by projecting a point ON the +X handle to a pixel.
                    math::Vec3 onHandle = xf.position + math::Vec3{hl * 0.5f, 0, 0};
                    float wclip = 0.0f;
                    math::Vec3 ndc = math::MulPointDivide(cam.ViewProj(), onHandle, wclip);
                    in.mouseX = (ndc.x * 0.5f + 0.5f) * (float)w;
                    in.mouseY = (-ndc.y * 0.5f + 0.5f) * (float)h;
                    editorInteract(in, /*leftEdge=*/true, /*leftDown=*/true);  // grab the X axis
                    // Drag: move the cursor to a point further along +X each frame.
                    for (int f = 0; f < 4; ++f) {
                        math::Vec3 farther = xf.position + math::Vec3{hl * 0.5f + (f + 1) * 0.4f, 0, 0};
                        ndc = math::MulPointDivide(cam.ViewProj(), farther, wclip);
                        in.mouseX = (ndc.x * 0.5f + 0.5f) * (float)w;
                        in.mouseY = (-ndc.y * 0.5f + 0.5f) * (float)h;
                        editorInteract(in, /*leftEdge=*/false, /*leftDown=*/true);
                    }
                    editorInteract(in, /*leftEdge=*/false, /*leftDown=*/false);  // release
                    afterPos = editReg.get<scene::TransformC>(ents[pickedIndex]).t.position;
                }
                bool dragged = (afterPos.x - beforePos.x) > 1e-3f;

                // Exercise the camera + clock for a few frames (as before).
                runtime::InputState flyIn;
                flyIn.relativeMouse = true;
                for (int frame = 0; frame < 8; ++frame) {
                    flyIn.keyDown[(int)runtime::Key::W] = true;
                    flyIn.mouseDx = 6.0f; flyIn.mouseDy = -3.0f;
                    float dt = 1.0f / 60.0f;
                    totalSteps += clock.Tick(dt);
                    controller.Update(cam, flyIn, dt);
                    FrameData fd = makeFrameData(cam);
                    render::RenderGraph graph;
                    recordFrame(graph, fd, sortGlass(cam.position));
                    (void)graph;
                }
                bool moved = math::length(cam.position - startPos) > 1e-3f;

                // Exercise the hot-reload path end-to-end: DUMP the current edited registry to a temp
                // JSON (it references only registered resources: cube/sphere/plane), then re-LOAD it
                // through the SAME applyHotReload path the live FileWatcher drives, swapping the live
                // registry. Proves LoadScene->swap works on real data (the stock default.json names a
                // 'duck' mesh not registered in the fly editRes, so we round-trip our own scene here).
                size_t before = editReg.aliveCount();
                bool reloaded = false;
                {
                    std::string tmpScene = "fly_dryrun_scene.json";
                    std::string json = scene::DumpScene(editReg, editRes);
                    std::ofstream out(tmpScene, std::ios::binary);
                    out << json; out.close();
                    runtime::FileWatcher dryWatch;
                    dryWatch.Watch(tmpScene);              // baseline
                    // Re-LoadScene via the same logic the live watcher uses (inline, temp path).
                    try {
                        ecs::Registry fresh;
                        scene::LoadScene(fresh, editRes, tmpScene.c_str());
                        size_t loaded = fresh.aliveCount();
                        editReg = std::move(fresh);
                        sel.index = -1;
                        reloaded = (loaded == before && loaded > 0);
                    } catch (const std::exception& ex) {
                        std::printf("[hot-reload] dry-run scene reload FAILED (%s)\n", ex.what());
                    }
                }
                // Also confirm the live-path applyHotReload is robust on the stock scene (caught + kept).
                applyHotReload({scenePath});

                std::printf("fly-dry-run: %d frames, %d fixed steps, camera moved %s | "
                            "pick=%s(idx %d) drag+X=%s reload=%s (entities %zu->%zu)\n",
                            8, totalSteps, moved ? "yes" : "NO",
                            picked ? "yes" : "NO", pickedIndex, dragged ? "yes" : "NO",
                            reloaded ? "yes" : "NO", before, editReg.aliveCount());
                device->WaitIdle();
                bool ok = moved && picked && dragged && reloaded;
                return ok ? 0 : 1;
            }

            // Live windowed loop. Mouse-look is engaged only while the RIGHT button is held, so the
            // cursor stays visible for LEFT-click picking + gizmo drag (Slice AM). ESC quits.
            std::printf("--fly: WASD move, RIGHT-drag look, Space/E up, Ctrl/Q down, Shift sprint, "
                        "wheel speed, ESC quit | LEFT-click pick, LEFT-drag gizmo, P play/pause, O "
                        "step, G/R/T gizmo mode, Ctrl+S save\n");

            bool prevP = false, prevO = false, prevCtrlS = false, prevLeft = false, prevRight = false;
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

                // --- Mouse-look on RIGHT button (engages/releases relative mouse). ---
                bool nowRight = in.mouseButtons[1];
                if (nowRight && !prevRight) window.SetRelativeMouse(true);
                if (!nowRight && prevRight) window.SetRelativeMouse(false);
                prevRight = nowRight;

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

                // --- LEFT-click pick + LEFT-drag gizmo (only while NOT mouse-looking on the right). ---
                int activeAxis = editor::kAxisNone;
                if (!nowRight) {
                    bool nowLeft = in.mouseButtons[0];
                    bool leftEdge = nowLeft && !prevLeft;
                    activeAxis = editorInteract(in, leftEdge, nowLeft);
                    prevLeft = nowLeft;
                } else {
                    prevLeft = false;  // ignore left while flying so a release after fly doesn't pick
                }
                rebuildGizmoLines(activeAxis);

                // --- Hot-reload poll (scene JSON + shader .spv). ---
                applyHotReload(watcher.Poll());

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

        // --- Frustum-culling visualization showcase (--cull-shot, Slice AQ): a deterministic scene of
        // a ground plane + a wide row of cubes/spheres spread across X. A pulled-back OVERVIEW camera
        // renders everything; we then draw (a) the actual, NARROWER RENDER camera's view frustum as
        // wireframe LINES (its 8 corners un-projected through inverse(viewProj)) and (b) each object's
        // world bounding SPHERE colored GREEN if the render camera KEEPS it / RED if it CULLS it. The
        // cull decision is the conservative bounding-sphere test in engine/render/frustum.h — exactly
        // the test the render submission path uses — so this is the render-invariant proof made
        // visible. A {drawn, culled, total} stat line is printed. One BMP -> exit. New golden cull.png;
        // existing image goldens untouched (culling removes nothing on-screen in the normal scenes). --
        if (cullShotPath) {
            using math::Mat4; using math::Vec3;
            namespace fr = render::frustum;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            scene::Mesh planeMesh  = scene::Mesh::Plane(*device);
            scene::Mesh cubeMesh   = scene::Mesh::Cube(*device);
            scene::Mesh sphereMesh = scene::Mesh::Sphere(*device);

            // --- The objects the RENDER camera will cull against: a wide row across X at z=-6, far
            // enough left/right to fall outside the narrow render frustum. Alternating cube/sphere. --
            struct CullObj { const scene::Mesh* mesh; scene::Transform xform; };
            std::vector<CullObj> objs;
            for (int k = -6; k <= 6; ++k) {
                scene::Transform t;
                t.position = {(float)k * 2.2f, 0.8f, -6.0f};
                t.scale = {0.7f, 0.7f, 0.7f};
                objs.push_back({(k & 1) ? &sphereMesh : &cubeMesh, t});
            }

            // --- The actual RENDER camera (narrow): at the origin-ish, looking down -Z, 50deg FOV. It
            // sees only the central few cubes; the wings fall outside its frustum. ---
            runtime::Camera renderCam;
            renderCam.position = {0.0f, 0.8f, 2.0f};
            renderCam.yaw = 0.0f; renderCam.SetPitch(0.0f);
            renderCam.fovY = 0.8726646f;  // 50 degrees
            renderCam.aspect = aspect;
            renderCam.znear = 0.5f; renderCam.zfar = 12.0f;
            Mat4 renderVP = renderCam.ViewProj();   // UNJITTERED (no TAA here) — the cull frustum.
            fr::Frustum cullFrustum = fr::FromViewProj(renderVP);

            // --- World bounding sphere per object + the keep/cull partition. worldCenter = model*
            // localCenter; worldRadius = localRadius * maxAbsScale(model) (uniform-scale-safe; our
            // scenes are uniform). Conservative: a straddling sphere is KEPT. ---
            struct ObjBound { Vec3 center; float radius; bool kept; };
            std::vector<ObjBound> bounds;
            int drawn = 0, culled = 0;
            for (const auto& ob : objs) {
                const scene::MeshBounds& mb = ob.mesh->bounds();
                Vec3 localCenter{(mb.min.x + mb.max.x) * 0.5f,
                                 (mb.min.y + mb.max.y) * 0.5f,
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

            // --- Pipelines: static lit + static shadow + sky + post + debug-line. ---
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

            scene::Transform groundXform; groundXform.scale = {16.0f, 1.0f, 16.0f};
            Mat4 groundModel = groundXform.Matrix();

            // --- Build the debug overlay: the render frustum (white wireframe) + each object's bound
            // sphere green(kept)/red(culled). Deterministic order -> byte-identical vertex stream. ---
            debug::DebugDraw dd;
            {
                const Vec3 kFrustumColor{0.95f, 0.95f, 0.98f};
                Vec3 corners[8];
                fr::Corners(renderVP, corners);
                // corner index: bit0=x(-/+), bit1=y(-/+), bit2=z(near/far). Near face = 0..3, far 4..7.
                auto edge = [&](int a, int b) { dd.Line(corners[a], corners[b], kFrustumColor); };
                // near quad (z=near): 0-1,1-3,3-2,2-0
                edge(0, 1); edge(1, 3); edge(3, 2); edge(2, 0);
                // far quad (z=far): 4-5,5-7,7-6,6-4
                edge(4, 5); edge(5, 7); edge(7, 6); edge(6, 4);
                // connecting edges near->far
                edge(0, 4); edge(1, 5); edge(2, 6); edge(3, 7);
                // Per-object bound spheres, colored by keep/cull.
                const Vec3 kKeep{0.15f, 0.95f, 0.25f};   // green
                const Vec3 kCull{0.95f, 0.18f, 0.15f};   // red
                for (const auto& bnd : bounds)
                    dd.WireSphere(bnd.center, bnd.radius, bnd.kept ? kKeep : kCull, 16);
            }
            const uint32_t kLineVertCount = (uint32_t)dd.VertexCount();
            rhi::BufferDesc lineBufDesc;
            lineBufDesc.size = (uint64_t)dd.Vertices().size() * sizeof(debug::LineVertex);
            lineBufDesc.initialData = dd.Vertices().data();
            lineBufDesc.usage = rhi::BufferUsage::Vertex;
            auto lineBuffer = device->CreateBuffer(lineBufDesc);

            // --- The OVERVIEW camera: pulled back + up + to the side so the whole row + the render
            // frustum are in view. Distinct from the render camera (this is what we actually render). --
            runtime::Camera overviewCam;
            overviewCam.position = {6.0f, 11.0f, 16.0f};
            overviewCam.yaw = -0.32f; overviewCam.SetPitch(-0.52f);
            overviewCam.fovY = 1.04719755f;  // 60 degrees
            overviewCam.aspect = aspect;
            overviewCam.znear = 0.1f; overviewCam.zfar = 100.0f;

            const Vec3 lightDirVec = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
            FrameData fd{};
            {
                Mat4 vp = overviewCam.ViewProj();
                for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
                fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
                fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f; fd.lightColor[3] = 1.0f;
                fd.viewPos[0] = overviewCam.position.x; fd.viewPos[1] = overviewCam.position.y;
                fd.viewPos[2] = overviewCam.position.z; fd.viewPos[3] = 1.0f;
                fd.ptCount[0] = 0.0f;
                Vec3 sc{0.0f, 1.0f, -4.0f};
                Vec3 lightEye = sc - lightDirVec * 24.0f;
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
                Mat4 lightOrtho = Mat4::Ortho(-16.0f, 16.0f, -16.0f, 16.0f, 1.0f, 60.0f);
                Mat4 lightVP = lightOrtho * lightView;
                for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
                runtime::CameraBasis cb = overviewCam.Basis();
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
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::printf("cull-shot: {drawn: %d, culled: %d, total: %d}\n", drawn, culled, total);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(cullShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — %d drawn / %d culled of %d, %u debug-line vertices\n",
                                    cullShotPath, cw, ch2, drawn, culled, total, kLineVertCount);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", cullShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- GPU-driven culling + indirect draw (--gpu-cull-shot, Slice AR): a compute shader
        // (shaders/cull.comp.hlsl) frustum-culls a deterministic 1024-instance cube grid, ORDER-
        // compacts the survivors into a second instance buffer via a single-workgroup prefix sum, and
        // writes the indirect draw-args buffer (instanceCount = GPU survivor count). ONE
        // DrawIndexedIndirect then renders exactly the survivors — the draw count is decided on the
        // GPU, never round-tripped. We read the GPU count back ONCE and ASSERT it equals the CPU
        // reference over the SAME instances (engine/render/gpu_cull.h, mirroring frustum.h): the
        // exact-count proof. One BMP -> exit. New golden gpu_cull.png; existing goldens untouched. ----
        if (gpuCullShotPath) {
            using math::Mat4; using math::Vec3;
            namespace fr = render::frustum;
            namespace gc = render::gpu_cull;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            scene::Mesh planeMesh = scene::Mesh::Plane(*device);
            scene::Mesh cubeMesh  = scene::Mesh::Cube(*device);

            // --- The 32x32 = 1024 instance grid (one workgroup). Deterministic (no RNG); a WIDE grid
            // so a narrow camera sees only a central subset (a real cull). Same builder the
            // --instanced-shot uses; the unit cube's local bound is center=origin, radius=sqrt(0.75). -
            const uint32_t kGridN = 32;
            std::vector<scene::InstanceData> grid =
                scene::BuildInstanceGrid(kGridN, /*spacing=*/1.3f, /*scale=*/0.45f);
            const uint32_t kInstanceCount = (uint32_t)grid.size();
            const Vec3 localCenter{0.0f, 0.0f, 0.0f};
            const float localRadius = std::sqrt(0.75f);

            // Flatten to the 16-floats-per-instance model stream the shader/mirror consume.
            std::vector<float> models;
            models.reserve((size_t)kInstanceCount * 16);
            for (const auto& inst : grid)
                for (int k = 0; k < 16; ++k) models.push_back(inst.model[k]);

            // --- The render camera: elevated, looking at the grid centre, narrow 35deg FOV so the
            // wings of the wide grid fall outside the frustum. This view-proj (Vulkan clip) feeds BOTH
            // the GPU cull (its six planes) and the CPU reference. ---
            const Vec3 eye{0.0f, 11.0f, 17.0f};
            const Vec3 center{0.0f, 0.0f, 0.0f};
            Mat4 view = Mat4::LookAt(eye, center, {0, 1, 0});
            Mat4 proj = Mat4::Perspective(0.6108652f /*35deg*/, aspect, 0.5f, 80.0f);
            Mat4 vp = proj * view;
            fr::Frustum cullFrustum = fr::FromViewProj(vp);

            // --- CPU REFERENCE survivor count (the exact-count proof's right-hand side). ---
            const uint32_t cpuRef =
                gc::SurvivorCount(models, kInstanceCount, cullFrustum, localCenter, localRadius);

            // --- GPU buffers. Source instances (Storage, also a vertex stream); compacted survivors
            // (Storage, consumed as the per-instance stream by the indirect draw); indirect args
            // (Indirect); cull params (Storage: 6 planes + localCenter/radius + counts). ---
            rhi::BufferDesc srcDesc;
            srcDesc.size = (uint64_t)kInstanceCount * sizeof(scene::InstanceData);
            srcDesc.initialData = grid.data();
            srcDesc.usage = rhi::BufferUsage::Storage;
            auto srcInstances = device->CreateBuffer(srcDesc);

            rhi::BufferDesc survDesc;
            survDesc.size = (uint64_t)kInstanceCount * sizeof(scene::InstanceData);
            survDesc.usage = rhi::BufferUsage::Storage;  // written by compute, read as instance stream
            auto survInstances = device->CreateBuffer(survDesc);

            // Indirect args: {indexCount, instanceCount, firstIndex, vertexOffset, firstInstance}.
            // Seed indexCount so even a no-dispatch path is well-formed; the compute overwrites all 5.
            uint32_t argsInit[5] = {cubeMesh.indexCount(), 0u, 0u, 0u, 0u};
            rhi::BufferDesc argsDesc;
            argsDesc.size = sizeof(argsInit);
            argsDesc.initialData = argsInit;
            argsDesc.usage = rhi::BufferUsage::Indirect;
            auto argsBuffer = device->CreateBuffer(argsDesc);

            // Cull params (matches shaders/cull.comp.hlsl Params, std430): 6 planes (n.xyz,d) +
            // localCenter(xyz)+radius(w) + counts(instanceCount,indexCount,_,_).
            struct CullParams {
                float planes[6][4];
                float localCenter[4];
                uint32_t counts[4];
            };
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

            // --- Compute cull pipeline: 4 storage buffers (instances/survivors/args/params), one
            // workgroup of 1024 threads (ordered prefix-sum compaction over <=1024 instances). ---
            auto cullCsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/cull.comp.hlsl.spv");
            auto cullCs = device->CreateShaderModule({std::span<const uint32_t>(cullCsWords)});
            rhi::ComputePipelineDesc cullCdesc;
            cullCdesc.compute = cullCs.get();
            cullCdesc.storageBufferCount = 4;
            cullCdesc.pushConstantSize = 0;       // all params come from the storage buffer
            cullCdesc.threadsPerGroupX = 1024;    // [numthreads(1024,1,1)] — one workgroup
            auto cullCompute = device->CreateComputePipeline(cullCdesc);

            // --- Instanced lit pipeline (lit_instanced.vert + shared lit.frag): draws the survivor
            // stream. Same pipeline the --instanced-shot uses (per-instance mat4 at locations 7-10). --
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
            instDesc.pushConstantSize = sizeof(float) * 4;  // float4 material
            auto instPipeline = device->CreateGraphicsPipeline(instDesc);

            // Static lit pipeline (ground) + sky + post.
            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = device->Swapchain().ColorFormat();
            litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

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
            Mat4 groundModel = Mat4::Scale({26.0f, 1.0f, 26.0f});

            // --- Frame uniforms (fixed, deterministic). ---
            FrameData fd{};
            {
                for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
                fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
                fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f;
                fd.lightColor[3] = 1.0f;
                fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1;
                fd.ptCount[0] = 0.0f;
                Vec3 lightDir = math::normalize(Vec3{-0.5f, -1.0f, -0.3f});
                Vec3 sc{0.0f, 0.0f, 0.0f};
                Vec3 lightEye = sc - lightDir * 30.0f;
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
                Mat4 lightOrtho = Mat4::Ortho(-24.0f, 24.0f, -24.0f, 24.0f, 1.0f, 70.0f);
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
                    // GPU cull dispatch (OUTSIDE the render pass): compute compacts survivors + writes
                    // the indirect args, then barrier compute->(vertex+indirect).
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
                    // Ground plane.
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
                    // GPU-CULLED instanced cubes: ONE DrawIndexedIndirect. The survivor stream + the
                    // indirect args were produced by the compute dispatch.
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
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);
            device->WaitIdle();

            // --- Exact-count proof: read the GPU-written instanceCount back and assert == CPU ref. ---
            uint32_t gpuArgs[5] = {0, 0, 0, 0, 0};
            device->ReadBuffer(*argsBuffer, gpuArgs, sizeof(gpuArgs), 0);
            const uint32_t gpuDrawn = gpuArgs[1];
            std::printf("gpu-cull: {drawn: %u, cpuRef: %u, total: %u}\n",
                        gpuDrawn, cpuRef, kInstanceCount);
            if (gpuDrawn != cpuRef) {
                std::fprintf(stderr,
                    "FATAL: GPU survivor count %u != CPU reference %u (cull-logic mismatch)\n",
                    gpuDrawn, cpuRef);
                device->WaitIdle();
                return 1;
            }

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(gpuCullShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — GPU-culled %u/%u instances (indirect draw)\n",
                                    gpuCullShotPath, cw, ch2, gpuDrawn, kInstanceCount);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", gpuCullShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- Multithreaded command recording (--mt-shot <out.bmp> [--workers N], Slice AU): a
        // draw-heavy grid of NON-INSTANCED DISTINCT draws (each object is a SEPARATE recorded
        // DrawIndexed — real per-draw recording, not one instanced call). The scene pass's object
        // draw list is partitioned into N CONTIGUOUS index ranges (worker k -> [k*span,(k+1)*span));
        // each worker records its range into its OWN SECONDARY command buffer (Vulkan: one
        // VkCommandPool per worker thread). The primary executes the secondaries in WORKER-INDEX
        // ORDER, so the draw order == single-threaded order. Determinism oracle: --workers 1 and
        // --workers 4 produce BYTE-IDENTICAL captures (the printed pixel hash matches). Prints
        // `mt: {workers:N, draws:D, secondaries:N}`. One BMP -> exit. New golden mt.png. ----
        if (mtShotPath) {
            using math::Mat4; using math::Vec3;
            const uint32_t N = (uint32_t)mtWorkers;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            scene::Mesh planeMesh  = scene::Mesh::Plane(*device);
            scene::Mesh cubeMesh   = scene::Mesh::Cube(*device);
            scene::Mesh sphereMesh = scene::Mesh::Sphere(*device);

            // --- The draw-heavy scene: a 12x12 grid of 144 DISTINCT objects (alternating cube/sphere)
            // at varying heights, each recorded as its own draw. Deterministic placement (no RNG) so the
            // render is reproducible. This is the list the N workers split into contiguous ranges. ---
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

            // --- Pipelines: static lit + static shadow + sky + post. ---
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

            auto rt = device->CreateRenderTarget(w, h);
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

            // --- Camera: elevated three-quarter view so the whole grid + its shadows are visible. ---
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
                cam.aspect = aspect;
                cam.znear = 0.5f; cam.zfar = 80.0f;
                Mat4 vp = cam.ViewProj();
                for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
                fd.lightDir[0] = -0.5f; fd.lightDir[1] = -1.0f; fd.lightDir[2] = -0.3f;
                fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.97f; fd.lightColor[2] = 0.9f;
                fd.lightColor[3] = 1.0f;
                fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
                fd.ptCount[0] = 0.0f;
                Vec3 sc{0.0f, 1.0f, 0.0f};
                Vec3 lightEye = sc - lightDirVec * 26.0f;
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 1, 0});
                Mat4 lightOrtho = Mat4::Ortho(-15.0f, 15.0f, -15.0f, 15.0f, 1.0f, 60.0f);
                Mat4 lightVP = lightOrtho * lightView;
                for (int k = 0; k < 16; ++k) fd.lightViewProj[k] = lightVP.m[k];
                runtime::CameraBasis cb = cam.Basis();
                fd.camFwd[0]=cb.forward.x; fd.camFwd[1]=cb.forward.y; fd.camFwd[2]=cb.forward.z;
                fd.camRight[0]=cb.right.x; fd.camRight[1]=cb.right.y; fd.camRight[2]=cb.right.z;
                fd.camUp[0]=cb.up.x; fd.camUp[1]=cb.up.y; fd.camUp[2]=cb.up.z;
                fd.skyParams[0] = cb.tanHalfFovY; fd.skyParams[1] = aspect;
            }

            // Helper: record one lit object draw into the given command buffer.
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

            // Shadow pass — single-threaded (only the main scene pass is parallel this slice).
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

            // Scene pass — MULTITHREADED. The primary opens the pass expecting secondaries; N worker
            // threads each record a contiguous object sub-range into their own secondary; the primary
            // executes them in worker-index order.
            uint32_t recordedSecondaries = 0;
            graph.AddPass("scene", {rgShadow}, {rgScene},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1},
                                        /*expectsSecondaries=*/true);

                    // Create one secondary per worker thread (main thread; pools not thread-safe).
                    std::vector<rhi::ICommandBuffer*> secs(N, nullptr);
                    for (uint32_t k = 0; k < N; ++k)
                        secs[k] = dev.CreateSecondaryCommandBuffer(k);

                    // Pre-warm worker 0's secondary on the MAIN thread with the sky + ground (sky once;
                    // ground draws the lit material). This also forces the per-pair material-set cache
                    // insertion to happen single-threaded, so the parallel object recording below never
                    // mutates the cache (workers share NO mutable state — only their disjoint
                    // secondaries + index ranges).
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

                    // Parallel: worker k records objs[begin,end) into secs[k]. Each worker re-binds the
                    // lit pipeline + the (already-cached) shared material into its own secondary, then
                    // records its distinct object draws. The material set is the SAME immutable set for
                    // all (warmed above), so no worker touches shared mutable state.
                    runtime::RecordParallel(kDraws, N,
                        [&](uint32_t t, uint32_t b, uint32_t e) {
                            rhi::ICommandBuffer& c = *secs[t];
                            c.BindPipeline(*litPipeline);
                            c.BindMaterial(*groundTex, *flatNormal);
                            for (uint32_t i = b; i < e; ++i) recordObject(c, objs[i]);
                        });

                    cmd.ExecuteSecondaries(std::span<rhi::ICommandBuffer* const>(secs.data(),
                                                                                 secs.size()));
                    recordedSecondaries = N;
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

            std::printf("mt: {workers:%u, draws:%u, secondaries:%u}\n", N, kDraws,
                        recordedSecondaries);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                // FNV-1a 64-bit hash of the captured pixels — the determinism oracle's fingerprint.
                // --workers 1 and --workers 4 MUST print the same value (byte-identical render).
                uint64_t hash = 1469598103934665603ull;
                for (uint8_t byte : px) { hash ^= byte; hash *= 1099511628211ull; }
                std::printf("mt-hash: %016llx (%ux%u)\n", (unsigned long long)hash, cw, ch2);
                ok = WriteBMP(mtShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — %u draws across %u workers\n",
                                    mtShotPath, cw, ch2, kDraws, N);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", mtShotPath);
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
        // --- Screen-space reflections showcase (--ssr-shot, Slice AH): a flat reflective checkerboard
        // FLOOR with several DISTINCT colored objects (cubes + spheres) sitting ON it, lit + shadowed,
        // rendered into an HDR (RGBA16F) scene target PLUS a view-space normal+linear-depth g-buffer
        // (reusing the SSAO gbuffer shaders). An SSR pass reconstructs each pixel's view-space P and N,
        // computes R = reflect(normalize(P), N), and RAY-MARCHES R in view space, projecting each step
        // to screen UV and comparing the ray's depth to the g-buffer depth there; on a hit it samples
        // the HDR scene color -> mirror reflection (masked by a floor reflectivity from dot(N,viewUp),
        // Fresnel, screen-edge fade, binary-search-refined). A composite blends lerp(scene, ssr.rgb,
        // ssr.a) + the usual exposure/ACES/grade/vignette. SEPARATE ssr/ssr_composite pipelines +
        // shaders; existing gbuffer/ssao/bloom pipelines/shaders/goldens untouched.
        if (ssrShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
            const rhi::Format kHdr = rhi::Format::RGBA16_Float;
            const float kFovY = 1.04719755f;

            // --- Scene objects: distinct colored cubes + spheres on the floor at known positions. ---
            struct Obj { Vec3 pos; float scale; bool cube; float col[3]; };
            const Obj objs[] = {
                {{-2.2f, 0.7f, -0.5f}, 0.7f, true,  {0.90f, 0.20f, 0.20f}},  // red cube
                {{ 0.0f, 0.9f, -1.2f}, 0.9f, false, {0.20f, 0.85f, 0.30f}},  // green sphere
                {{ 2.3f, 0.6f,  0.2f}, 0.6f, true,  {0.25f, 0.45f, 0.95f}},  // blue cube
                {{-0.9f, 0.5f,  1.4f}, 0.5f, false, {0.95f, 0.80f, 0.20f}},  // yellow sphere
                {{ 1.4f, 0.75f, 1.6f}, 0.75f,true,  {0.85f, 0.35f, 0.90f}},  // magenta cube
            };
            const int kNumObjs = (int)(sizeof(objs) / sizeof(objs[0]));

            // --- Lit pipeline (static, writing HDR RT) — UNCHANGED lit shaders. ---
            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto litFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto litFs = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = kHdr;
            litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            // --- Shadow pipeline (static) — UNCHANGED. ---
            auto staticShW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto shadowFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto staticShVs = device->CreateShaderModule({std::span<const uint32_t>(staticShW)});
            auto shadowFs   = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});
            rhi::GraphicsPipelineDesc stShDesc;
            stShDesc.vertex = staticShVs.get(); stShDesc.fragment = shadowFs.get();
            stShDesc.vertexLayout = scene::MeshVertexLayout();
            stShDesc.depthTest = true; stShDesc.depthOnly = true; stShDesc.usesFrameUniforms = true;
            stShDesc.pushConstantSize = sizeof(float) * 16;
            auto staticShadowPipeline = device->CreateGraphicsPipeline(stShDesc);

            // --- Sky (writing HDR RT) — UNCHANGED. ---
            auto skyVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.vert.hlsl.spv");
            auto skyFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.frag.hlsl.spv");
            auto skyVsM = device->CreateShaderModule({std::span<const uint32_t>(skyVsW)});
            auto skyFsM = device->CreateShaderModule({std::span<const uint32_t>(skyFsW)});
            rhi::GraphicsPipelineDesc skyD;
            skyD.vertex = skyVsM.get(); skyD.fragment = skyFsM.get();
            skyD.colorFormat = kHdr;
            skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
            auto skyPipe = device->CreateGraphicsPipeline(skyD);

            // --- G-buffer prepass pipeline (static), view-space normal + linear depth -> RGBA16F. ---
            auto gbVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/gbuffer.vert.hlsl.spv");
            auto gbFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/gbuffer.frag.hlsl.spv");
            auto gbVs  = device->CreateShaderModule({std::span<const uint32_t>(gbVsW)});
            auto gbFs  = device->CreateShaderModule({std::span<const uint32_t>(gbFsW)});
            rhi::GraphicsPipelineDesc gbStDesc;
            gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
            gbStDesc.vertexLayout = scene::MeshVertexLayout();
            gbStDesc.colorFormat = kHdr;
            gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
            gbStDesc.pushConstantSize = sizeof(float) * 32;   // model(16) + view(16)
            auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

            // --- SSR + composite fullscreen pipelines (fragment push constants). ---
            auto postVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.vert.hlsl.spv");
            auto postVsM = device->CreateShaderModule({std::span<const uint32_t>(postVsW)});
            auto loadFs = [&](const char* name) {
                auto words = LoadSpirv(std::string(HF_SHADER_DIR) + "/" + name + ".spv");
                return device->CreateShaderModule({std::span<const uint32_t>(words)});
            };
            struct SsrParams {
                float texel[2]; float tanHalfFovY; float aspect;
                float maxDist; float thickness; float reflMin; float reflMax;
                float viewUp[4];
            };
            struct SsrCompParams { float texel[2]; float intensity; float pad; };

            auto ssrFs  = loadFs("ssr.frag.hlsl");
            auto compFs = loadFs("ssr_composite.frag.hlsl");

            rhi::GraphicsPipelineDesc ssrD;
            ssrD.vertex = postVsM.get(); ssrD.fragment = ssrFs.get();
            ssrD.colorFormat = kHdr;
            ssrD.depthTest = false; ssrD.usesTexture = true; ssrD.fullscreen = true;
            ssrD.fragmentPushConstants = true; ssrD.pushConstantSize = sizeof(SsrParams);
            auto ssrPipe = device->CreateGraphicsPipeline(ssrD);

            rhi::GraphicsPipelineDesc compD;
            compD.vertex = postVsM.get(); compD.fragment = compFs.get();
            compD.colorFormat = device->Swapchain().ColorFormat();
            compD.depthTest = false; compD.usesTexture = true; compD.fullscreen = true;
            compD.fragmentPushConstants = true; compD.pushConstantSize = sizeof(SsrCompParams);
            auto compPipe = device->CreateGraphicsPipeline(compD);

            // --- Render targets: HDR lit scene + RGBA16F g-buffer + SSR reflection. ---
            auto rt    = device->CreateRenderTarget(w, h, kHdr);
            auto gbuf  = device->CreateRenderTarget(w, h, kHdr);
            auto ssrRT = device->CreateRenderTarget(w, h, kHdr);
            auto shadowMap = device->CreateShadowMap(2048);
            device->SetShadowMap(*shadowMap);

            // DARK, low-contrast checker floor: a near-black polished surface so the SSR mirror
            // reflections of the objects read clearly on top of it (the bright default checkerboard
            // would wash the reflections out). 8x8 tiles alternating two dark greys.
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
            // Per-object solid-color 1x1 textures (lit.frag multiplies gTex.Sample * vertex color;
            // the mesh's baked vertex color is near-white, so this drives the object hue).
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
            // Per-object model matrices (translate * scale).
            std::vector<Mat4> objModel(kNumObjs);
            for (int o = 0; o < kNumObjs; ++o) {
                objModel[o] = Mat4::Translate(objs[o].pos) * Mat4::Scale(
                    {objs[o].scale, objs[o].scale, objs[o].scale});
            }

            // Camera: a grazing look down at the floor so the lower screen is filled by the reflective
            // floor and the objects' inverted reflections sit directly below them.
            const Vec3 eye{0.0f, 2.6f, 7.2f};
            const Vec3 center{0.0f, 0.7f, 0.0f};
            Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
            FrameData fd{};
            {
                Mat4 proj = Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f);
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
                Mat4 lightOrtho = Mat4::Ortho(-7.0f, 7.0f, -7.0f, 7.0f, 1.0f, 40.0f);
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

            // SSR params. viewUp = world up (0,1,0) expressed in VIEW space = (view rotation) * up.
            // For a LookAt view, world-up transformed by the view's 3x3 rotation gives its view-space
            // direction; the floor's view normal aligns with this, objects don't.
            SsrParams sp{};
            sp.texel[0] = 1.0f / (float)w; sp.texel[1] = 1.0f / (float)h;
            sp.tanHalfFovY = std::tan(0.5f * kFovY); sp.aspect = aspect;
            sp.maxDist = 8.0f; sp.thickness = 0.35f; sp.reflMin = 0.75f; sp.reflMax = 0.92f;
            const bool ssrDbg = (std::getenv("HF_SSR_DBG") != nullptr);
            if (ssrDbg) sp.reflMax = -1.0f;   // debug: SSR pass outputs hit weight as grayscale
            {
                // viewUp = upper-left 3x3 of viewM applied to world up (0,1,0) = column 1 of the
                // rotation (rows of the 3x3 stored column-major in m). Mat4::LookAt stores the basis
                // so that m[1],m[5],m[9] = the y-row; transforming (0,1,0) picks that row.
                Vec3 wup{0.0f, 1.0f, 0.0f};
                Vec3 vUp{
                    viewM.m[0]*wup.x + viewM.m[4]*wup.y + viewM.m[8]*wup.z,
                    viewM.m[1]*wup.x + viewM.m[5]*wup.y + viewM.m[9]*wup.z,
                    viewM.m[2]*wup.x + viewM.m[6]*wup.y + viewM.m[10]*wup.z};
                vUp = math::normalize(vUp);
                sp.viewUp[0] = vUp.x; sp.viewUp[1] = vUp.y; sp.viewUp[2] = vUp.z; sp.viewUp[3] = 0.0f;
            }
            SsrCompParams cp{}; cp.texel[0] = 1.0f / (float)w; cp.texel[1] = 1.0f / (float)h;
            cp.intensity = 1.7f; cp.pad = ssrDbg ? -1.0f : 0.0f;

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

            // Lit scene -> HDR RT. Floor is SMOOTH (low roughness) so its in-shader IBL is mirror-ish;
            // objects are matte. The floor's SSR reflection is added on top in the composite.
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
                        pc[16] = 0.0f; pc[17] = 0.15f; pc[18] = 0.0f; pc[19] = 0.0f; // metallic 0, smooth
                        cmd.PushConstants(pc, sizeof(pc));
                        cmd.BindMaterial(*groundTex, *flatNormal);
                        cmd.BindVertexBuffer(plane.vertices());
                        cmd.BindIndexBuffer(plane.indices());
                        cmd.DrawIndexed(plane.indexCount());
                    }
                    for (int o = 0; o < kNumObjs; ++o) {
                        float pc[20];
                        for (int k = 0; k < 16; ++k) pc[k] = objModel[o].m[k];
                        pc[16] = 0.0f; pc[17] = 0.6f; pc[18] = 0.0f; pc[19] = 0.0f; // matte objects
                        cmd.PushConstants(pc, sizeof(pc));
                        cmd.BindMaterial(*objTex[o], *flatNormal);
                        drawObj(cmd, o);
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
                    for (int o = 0; o < kNumObjs; ++o) {
                        float pc[32];
                        for (int k = 0; k < 16; ++k) pc[k] = objModel[o].m[k];
                        for (int k = 0; k < 16; ++k) pc[16 + k] = viewM.m[k];
                        cmd.PushConstants(pc, sizeof(pc));
                        drawObj(cmd, o);
                    }
                    cmd.EndRenderPass();
                });

            // SSR ray-march -> SSR RT (rgb = reflection, a = weight). Scene at t0/s0, gbuffer at t3/s3.
            graph.AddPass("ssr", {rgScene, rgGbuf}, {rgSsr},
                [&](rhi::IRHIDevice&, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 0});
                    cmd.BindPipeline(*ssrPipe);
                    cmd.BindTexturePair(*rt, *gbuf);
                    cmd.PushConstants(&sp, sizeof(sp));
                    cmd.Draw(3);
                    cmd.EndRenderPass();
                });

            // Composite: lerp(scene, ssr.rgb, ssr.a) -> tonemap -> swapchain.
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
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(ssrShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — SSR, %d objects\n",
                                    ssrShotPath, cw, ch2, kNumObjs);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", ssrShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- Volumetric fog / light shafts showcase (--volumetric-shot, Slice AJ): a row of
        // shadow-casting PILLARS (a slotted wall) with a grazing directional light streaming BETWEEN
        // them, fog filling the air. The scene renders to an HDR (RGBA16F) RT + a view-space
        // normal+linear-depth g-buffer (reusing the SSAO gbuffer shaders, only .w used here). A
        // fullscreen volumetric pass reconstructs each pixel's world-space view ray, clamps the march
        // end to the scene depth (fog stops at solids), and RAY-MARCHES 64 steps sampling the
        // directional shadow map per step — lit air adds Henyey-Greenstein in-scattering with
        // Beer-Lambert extinction (god rays); shadowed air behind a pillar stays dark. A composite adds
        // the in-scatter over the scene + tonemaps. SEPARATE volumetric/volumetric_composite pipelines
        // + shaders; existing lit/ssao/ssr pipelines/shaders/goldens untouched.
        if (volumetricShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
            const rhi::Format kHdr = rhi::Format::RGBA16_Float;
            const float kFovY = 1.04719755f;

            // --- Occluders: an OVERHEAD slatted canopy (a row of beams running across the view, like a
            // pergola / venetian blind) with gaps between the slats. The near-overhead directional light
            // streams DOWN through the gaps, so the air below is carved into vertical light SHAFTS
            // (under the gaps) separated by dark shadow volumes (under the slats). The camera looks
            // roughly HORIZONTALLY across these shafts, so the bright beams and dark gaps read clearly as
            // distinct, parallel god rays raking down into the scene. A back wall + floor give the
            // shafts a surface to land on. ---
            struct Occ { Vec3 pos; Vec3 scale; float col[3]; };
            std::vector<Occ> occs;
            const float kCanopyY = 5.4f;          // height of the overhead slats
            const int kSlats = 7;
            // Slats run along Z (deep), spaced along X with gaps; each slat is thin in X, long in Z.
            for (int p = 0; p < kSlats; ++p) {
                float x = -5.4f + (float)p * 1.8f;   // slats at x = -5.4,-3.6,...,5.4 (gap ~1.0)
                occs.push_back({{x, kCanopyY, -1.0f}, {0.45f, 0.30f, 7.0f}, {0.40f, 0.41f, 0.47f}});
            }
            // A back wall behind the scene so the shafts terminate on a lit surface (depth reference).
            occs.push_back({{0.0f, 3.0f, -9.0f}, {12.0f, 3.0f, 0.4f}, {0.30f, 0.31f, 0.36f}});
            const int kNumOcc = (int)occs.size();

            // --- Lit pipeline (static, writing HDR RT) — UNCHANGED lit shaders. ---
            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto litFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto litFs = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = kHdr;
            litDesc.depthTest = true; litDesc.usesFrameUniforms = true; litDesc.usesTexture = true;
            litDesc.pushConstantSize = sizeof(float) * 20;
            auto litPipeline = device->CreateGraphicsPipeline(litDesc);

            // --- Shadow pipeline (static) — UNCHANGED. ---
            auto staticShW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto shadowFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.frag.hlsl.spv");
            auto staticShVs = device->CreateShaderModule({std::span<const uint32_t>(staticShW)});
            auto shadowFs   = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});
            rhi::GraphicsPipelineDesc stShDesc;
            stShDesc.vertex = staticShVs.get(); stShDesc.fragment = shadowFs.get();
            stShDesc.vertexLayout = scene::MeshVertexLayout();
            stShDesc.depthTest = true; stShDesc.depthOnly = true; stShDesc.usesFrameUniforms = true;
            stShDesc.pushConstantSize = sizeof(float) * 16;
            auto staticShadowPipeline = device->CreateGraphicsPipeline(stShDesc);

            // --- Sky (writing HDR RT) — UNCHANGED. ---
            auto skyVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.vert.hlsl.spv");
            auto skyFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.frag.hlsl.spv");
            auto skyVsM = device->CreateShaderModule({std::span<const uint32_t>(skyVsW)});
            auto skyFsM = device->CreateShaderModule({std::span<const uint32_t>(skyFsW)});
            rhi::GraphicsPipelineDesc skyD;
            skyD.vertex = skyVsM.get(); skyD.fragment = skyFsM.get();
            skyD.colorFormat = kHdr;
            skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
            auto skyPipe = device->CreateGraphicsPipeline(skyD);

            // --- G-buffer prepass pipeline (static), view-space normal + linear depth -> RGBA16F. ---
            auto gbVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/gbuffer.vert.hlsl.spv");
            auto gbFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/gbuffer.frag.hlsl.spv");
            auto gbVs  = device->CreateShaderModule({std::span<const uint32_t>(gbVsW)});
            auto gbFs  = device->CreateShaderModule({std::span<const uint32_t>(gbFsW)});
            rhi::GraphicsPipelineDesc gbStDesc;
            gbStDesc.vertex = gbVs.get(); gbStDesc.fragment = gbFs.get();
            gbStDesc.vertexLayout = scene::MeshVertexLayout();
            gbStDesc.colorFormat = kHdr;
            gbStDesc.depthTest = true; gbStDesc.usesFrameUniforms = true;
            gbStDesc.pushConstantSize = sizeof(float) * 32;   // model(16) + view(16)
            auto gbStaticPipeline = device->CreateGraphicsPipeline(gbStDesc);

            // --- Volumetric + composite fullscreen pipelines. The volumetric pass needs BOTH frame
            // uniforms (camera basis + lightViewProj + the shadow map in set 0 t1/s1) AND a texture
            // (the g-buffer in set 1 t0/s0), plus a fragment push constant for the fog params. ---
            auto postVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.vert.hlsl.spv");
            auto postVsM = device->CreateShaderModule({std::span<const uint32_t>(postVsW)});
            auto loadFs = [&](const char* name) {
                auto words = LoadSpirv(std::string(HF_SHADER_DIR) + "/" + name + ".spv");
                return device->CreateShaderModule({std::span<const uint32_t>(words)});
            };
            struct VolParams {
                float texel[2]; float density; float g;
                float extinction; float marchDist; float steps; float pad;
            };
            struct VolCompParams { float texel[2]; float intensity; float pad; };

            auto volFs  = loadFs("volumetric.frag.hlsl");
            auto compFs = loadFs("volumetric_composite.frag.hlsl");

            rhi::GraphicsPipelineDesc volD;
            volD.vertex = postVsM.get(); volD.fragment = volFs.get();
            volD.colorFormat = kHdr;
            volD.depthTest = false; volD.fullscreen = true;
            volD.usesFrameUniforms = true; volD.usesTexture = true;
            volD.fragmentPushConstants = true; volD.pushConstantSize = sizeof(VolParams);
            auto volPipe = device->CreateGraphicsPipeline(volD);

            rhi::GraphicsPipelineDesc compD;
            compD.vertex = postVsM.get(); compD.fragment = compFs.get();
            compD.colorFormat = device->Swapchain().ColorFormat();
            compD.depthTest = false; compD.usesTexture = true; compD.fullscreen = true;
            compD.fragmentPushConstants = true; compD.pushConstantSize = sizeof(VolCompParams);
            auto compPipe = device->CreateGraphicsPipeline(compD);

            // --- Render targets: HDR lit scene + RGBA16F g-buffer + volumetric in-scatter. ---
            auto rt    = device->CreateRenderTarget(w, h, kHdr);
            auto gbuf  = device->CreateRenderTarget(w, h, kHdr);
            auto volRT = device->CreateRenderTarget(w, h, kHdr);
            auto shadowMap = device->CreateShadowMap(2048);
            device->SetShadowMap(*shadowMap);

            // Ground + textures.
            std::vector<uint8_t> floorPx(256 * 256 * 4);
            for (uint32_t y = 0; y < 256; ++y)
                for (uint32_t x = 0; x < 256; ++x) {
                    bool dark = (((x / 32) + (y / 32)) & 1) != 0;
                    uint8_t v = dark ? 12 : 20;          // near-black floor: dark base so beams glow
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

            // Camera looks generally TOWARD the light (so HG forward-scatter makes the beams streaming
            // through the slots bright). It sits back and a little high so the tall slotted wall and the
            // beams above the floor are in frame.
            const Vec3 eye{0.0f, 2.7f, 10.5f};
            const Vec3 center{0.0f, 2.4f, -2.0f};
            Mat4 viewM = Mat4::LookAt(eye, center, {0, 1, 0});
            // Photon travel direction: the sun is high overhead, tilted slightly toward the camera so
            // the shafts streaming down through the canopy gaps SLOPE forward (toward +z) into the
            // scene rather than falling perfectly vertical — more legible god rays. Mostly -Y (down).
            Vec3 lightDir = math::normalize(Vec3{0.05f, -0.74f, 0.67f});
            FrameData fd{};
            {
                Mat4 proj = Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f);
                Mat4 vp = proj * viewM;
                for (int k = 0; k < 16; ++k) fd.vp[k] = vp.m[k];
                fd.lightDir[0] = lightDir.x; fd.lightDir[1] = lightDir.y; fd.lightDir[2] = lightDir.z;
                fd.lightColor[0] = 1.0f; fd.lightColor[1] = 0.95f; fd.lightColor[2] = 0.82f; fd.lightColor[3] = 1.0f;
                fd.viewPos[0] = eye.x; fd.viewPos[1] = eye.y; fd.viewPos[2] = eye.z; fd.viewPos[3] = 1.0f;
                fd.ptCount[0] = 0.0f;
                // Directional shadow: an ortho frustum centered on the scene, looking along lightDir.
                Vec3 sc{0.0f, 1.5f, -1.0f};
                Vec3 lightEye = sc - lightDir * 26.0f;
                // Light is near-vertical, so use a world-Z up reference for the light's view basis
                // (world-Y would be near-parallel to the look direction and degenerate).
                Mat4 lightView = Mat4::LookAt(lightEye, sc, {0, 0, -1});
                Mat4 lightOrtho = Mat4::Ortho(-12.0f, 12.0f, -12.0f, 12.0f, 1.0f, 52.0f);
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

            // March params: 64 steps, forward-scatter g=0.6, density/extinction tuned for clear but
            // smooth shafts. marchDist covers the scene + open air behind it.
            VolParams vparm{};
            vparm.texel[0] = 1.0f / (float)w; vparm.texel[1] = 1.0f / (float)h;
            vparm.density = 0.8f; vparm.g = 0.4f; vparm.extinction = 0.06f;
            vparm.marchDist = 26.0f; vparm.steps = 64.0f; vparm.pad = 0.0f;

            const bool volDbg = (std::getenv("HF_VOL_DBG") != nullptr);
            VolCompParams cp{}; cp.texel[0] = 1.0f / (float)w; cp.texel[1] = 1.0f / (float)h;
            // Lower exposure than the other showcases: the god rays are the bright feature, so the base
            // scene is kept dim/dusk and not pushed to clip — that preserves the shaft contrast.
            cp.intensity = 0.72f; cp.pad = volDbg ? -1.0f : 0.0f;

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

            // NOTE: no sky pass here. The volumetric god rays are the bright feature, so the scene is
            // kept on a DARK clear (dusk) backdrop — bright lit fog beams glow against the dark air,
            // and the pillars' shadow volumes carve dark channels through them. (The sky pipeline is
            // still built above for parity but intentionally not drawn.)
            graph.AddPass("scene", {rgShadow}, {rgScene},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    dev.SetFrameUniforms(&fd, sizeof(FrameData));
                    cmd.BeginRenderPass(rhi::ClearColor{0.015f, 0.02f, 0.035f, 1});
                    cmd.BindPipeline(*litPipeline);
                    {
                        float pc[20];
                        for (int k = 0; k < 16; ++k) pc[k] = groundModel.m[k];
                        pc[16] = 0.0f; pc[17] = 0.85f; pc[18] = 0.0f; pc[19] = 0.0f; // matte floor
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

            // Volumetric ray-march -> volumetric RT. Frame uniforms (set 0: camera basis +
            // lightViewProj + shadow map t1/s1) + g-buffer (set 1 t0/s0 via BindTexture).
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

            // Composite: scene + volumetric -> tonemap -> swapchain.
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
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(volumetricShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — volumetric, %d pillars\n",
                                    volumetricShotPath, cw, ch2, kNumOcc);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", volumetricShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

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

        // === Temporal anti-aliasing (Slice AP). ============================================
        // Renders a FIXED N=8 accumulation frames over a STATIC lit + shadowed scene (a settled
        // instanced sphere pyramid on a ground plane under a sky — clean curved silhouettes that
        // alias hard in a single frame). Each accumulation frame jitters the PROJECTION by a
        // deterministic Halton(2,3) sub-pixel offset (render::taa::Jitter), renders the scene into an
        // HDR RT, then taa_resolve blends it into a neighborhood-clamped history (ping-ponged between
        // two RGBA16F textures). The first frame outputs the current frame unblended (empty history);
        // subsequent frames fold in render::taa::kSteadyAlpha of the new jittered frame. The 8th
        // resolved frame is tonemapped through the existing post.frag chain and captured. Static
        // scene/camera + deterministic jitter => two runs are bit-identical (golden). No new RHI: the
        // jitter rides SetFrameUniforms (jittered vp + additive prevViewProj), history uses the
        // existing RGBA16F RT + BindTexturePair path, the resolve reuses post.vert.
        if (taaShotPath) {
            using math::Mat4; using math::Vec3;
            namespace taa = render::taa;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
            const rhi::Format kHdr = rhi::Format::RGBA16_Float;
            const float kFovY = 1.04719755f;

            // --- Static scene geometry: a settled 4-layer instanced sphere pyramid (same recipe as
            // --physics-shot / --ssao-shot) — deterministic, with hard curved silhouettes. ---
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
                scene::InstanceData di;
                for (int k = 0; k < 16; ++k) di.model[k] = m.m[k];
                instances.push_back(di);
            }
            const uint32_t kInstanceCount = (uint32_t)instances.size();

            // --- Lit pipelines (HDR RT) — UNCHANGED lit/instanced shaders. ---
            auto instVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit_instanced.vert.hlsl.spv");
            auto litFsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto litVsWords  = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto instVs = device->CreateShaderModule({std::span<const uint32_t>(instVsWords)});
            auto litFs  = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            auto litVs  = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            rhi::GraphicsPipelineDesc instDesc;
            instDesc.vertex = instVs.get(); instDesc.fragment = litFs.get();
            instDesc.vertexLayout = scene::MeshVertexLayout();
            instDesc.instanceLayout = scene::InstanceTransformLayout();
            instDesc.colorFormat = kHdr;
            instDesc.depthTest = true; instDesc.usesFrameUniforms = true; instDesc.usesTexture = true;
            instDesc.pushConstantSize = sizeof(float) * 4;
            auto instPipeline = device->CreateGraphicsPipeline(instDesc);

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
            auto staticShW   = LoadSpirv(std::string(HF_SHADER_DIR) + "/shadow.vert.hlsl.spv");
            auto instShVs = device->CreateShaderModule({std::span<const uint32_t>(instShWords)});
            auto shadowFs = device->CreateShaderModule({std::span<const uint32_t>(shadowFsW)});
            auto staticShVs = device->CreateShaderModule({std::span<const uint32_t>(staticShW)});
            rhi::GraphicsPipelineDesc instShDesc;
            instShDesc.vertex = instShVs.get(); instShDesc.fragment = shadowFs.get();
            instShDesc.vertexLayout = scene::MeshVertexLayout();
            instShDesc.instanceLayout = scene::InstanceTransformLayout();
            instShDesc.depthTest = true; instShDesc.depthOnly = true; instShDesc.usesFrameUniforms = true;
            auto instShadowPipeline = device->CreateGraphicsPipeline(instShDesc);
            rhi::GraphicsPipelineDesc stShDesc;
            stShDesc.vertex = staticShVs.get(); stShDesc.fragment = shadowFs.get();
            stShDesc.vertexLayout = scene::MeshVertexLayout();
            stShDesc.depthTest = true; stShDesc.depthOnly = true; stShDesc.usesFrameUniforms = true;
            stShDesc.pushConstantSize = sizeof(float) * 16;
            auto staticShadowPipeline = device->CreateGraphicsPipeline(stShDesc);

            // --- Sky (HDR RT) — UNCHANGED procedural sky. ---
            auto skyVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.vert.hlsl.spv");
            auto skyFsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/sky.frag.hlsl.spv");
            auto skyVsM = device->CreateShaderModule({std::span<const uint32_t>(skyVsW)});
            auto skyFsM = device->CreateShaderModule({std::span<const uint32_t>(skyFsW)});
            rhi::GraphicsPipelineDesc skyD;
            skyD.vertex = skyVsM.get(); skyD.fragment = skyFsM.get();
            skyD.colorFormat = kHdr;
            skyD.depthTest = false; skyD.usesFrameUniforms = true; skyD.fullscreen = true;
            auto skyPipe = device->CreateGraphicsPipeline(skyD);

            // --- TAA resolve + final post (fullscreen, fragment push constants). ---
            auto postVsW = LoadSpirv(std::string(HF_SHADER_DIR) + "/post.vert.hlsl.spv");
            auto postVsM = device->CreateShaderModule({std::span<const uint32_t>(postVsW)});
            auto loadFs = [&](const char* name) {
                auto words = LoadSpirv(std::string(HF_SHADER_DIR) + "/" + name + ".spv");
                return device->CreateShaderModule({std::span<const uint32_t>(words)});
            };
            struct TaaParams { float texel[2]; float alpha; float firstFrame; };
            auto taaFs  = loadFs("taa_resolve.frag.hlsl");
            auto postFs = loadFs("post.frag.hlsl");

            rhi::GraphicsPipelineDesc taaD;
            taaD.vertex = postVsM.get(); taaD.fragment = taaFs.get();
            taaD.colorFormat = kHdr;
            taaD.depthTest = false; taaD.usesTexture = true; taaD.fullscreen = true;
            taaD.fragmentPushConstants = true; taaD.pushConstantSize = sizeof(TaaParams);
            auto taaPipe = device->CreateGraphicsPipeline(taaD);

            rhi::GraphicsPipelineDesc postD;
            postD.vertex = postVsM.get(); postD.fragment = postFs.get();
            postD.colorFormat = device->Swapchain().ColorFormat();
            postD.depthTest = false; postD.usesTexture = true; postD.fullscreen = true;
            auto postPipe = device->CreateGraphicsPipeline(postD);

            // --- Render targets: HDR scene + two ping-pong history textures. ---
            auto rt    = device->CreateRenderTarget(w, h, kHdr);
            auto histA = device->CreateRenderTarget(w, h, kHdr);
            auto histB = device->CreateRenderTarget(w, h, kHdr);
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
            Mat4 baseProj = Mat4::Perspective(kFovY, aspect, 0.1f, 100.0f);
            Mat4 unjittered = baseProj * viewM;  // the unjittered view-proj (prevViewProj source)

            // Static FrameData (lights/camera/sky); the per-frame jittered vp is written each frame.
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
                Mat4 lightOrtho = Mat4::Ortho(-8.0f, 8.0f, -8.0f, 8.0f, 1.0f, 40.0f);
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
                // prevViewProj = the unjittered view-proj. Static camera => identity reprojection; the
                // field is real (written + uploaded) so a moving-camera shot would exercise it.
                for (int k = 0; k < 16; ++k) fdBase.prevViewProj[k] = unjittered.m[k];
            }

            // Records the lit + shadowed scene (the geometry is identical every frame; only the
            // jittered viewProj in `fd` differs). Used as both the shadow and scene pass bodies.
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

            // === N=8 accumulation loop. Each frame: render the jittered scene -> rt, then resolve
            // (rt + prevHistory) -> curHistory with the neighborhood-clamped exponential blend. ===
            rhi::IRenderTarget* prevHist = histA.get();   // history read this frame
            rhi::IRenderTarget* curHist  = histB.get();   // resolved written this frame
            for (int frame = 0; frame < taa::kAccumFrames; ++frame) {
                // Build the jittered projection: add the sub-pixel NDC offset into the clip-space XY
                // translation per unit W (column 2, rows 0/1) AFTER composing the base projection, so
                // depth is untouched. (Mat4 is column-major: m[col*4+row].)
                taa::Vec2 j = taa::Jitter(frame, (int)w, (int)h);
                Mat4 jProj = baseProj;
                jProj.m[2 * 4 + 0] += j.x;   // m[2][0]: clip x += jitter.x * w
                jProj.m[2 * 4 + 1] += j.y;   // m[2][1]: clip y += jitter.y * w
                Mat4 jvp = jProj * viewM;
                FrameData fd = fdBase;
                for (int k = 0; k < 16; ++k) fd.vp[k] = jvp.m[k];

                const bool first = (frame == 0);
                TaaParams tp{};
                tp.texel[0] = 1.0f / (float)w; tp.texel[1] = 1.0f / (float)h;
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
                // Resolve: (current rt, prev history) -> current history (neighborhood-clamped blend).
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
                std::swap(prevHist, curHist);  // this frame's resolved becomes next frame's history
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
                graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
                graph.Execute(*device);
            }

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(taaShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — TAA %d-frame accumulation, %u bodies\n",
                                    taaShotPath, cw, ch2, taa::kAccumFrames, kInstanceCount);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", taaShotPath);
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

        // --- Playable game sample (--game-shot, Slice AX): a self-contained capture path that does
        // NOT touch the default scene. Build the deterministic roll-a-ball game (game::MakeRollGame:
        // ground + a dynamic player sphere + 3 fixed pickups), run game::StepGame over the FULL
        // game::ScriptedTrack() at the engine fixed dt (the full winning playthrough — score 3, won),
        // then render ONE frame at a FIXED MID-TRACK capture step (kGameCaptureStep) where the player
        // and at least one uncollected pickup are both on-screen, so the golden is legible. The player
        // sphere + each remaining pickup + the ground go through the EXISTING static lit + shadow
        // scene path (one per-draw model+material push constant each; distinct solid-color textures
        // tint them) — NO new RHI. Prints the deterministic winning state line. One BMP -> exit. The
        // pure-CPU gameplay (engine/game/roll_game) is golden-stable; pipelines/shaders are untouched.
        if (gameShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // Fixed capture step: a winning run is 380 steps long and collects pickup2 (the last) near
            // step ~300, so by the END all pickups are gone. We capture at step 250 — pickups 0 and 1
            // are already collected, pickup2 (x=5, z=2.5) is still present, and the player sits near
            // (5.4, 0.5, 0.8) rolling toward it: player + one remaining pickup both legible. The STATE
            // line below reports the FULL winning run (score 3, won, 380 steps), per the spec.
            const int kGameCaptureStep = 250;

            const float dtG = 1.0f / 120.0f;
            std::vector<game::GameInput> track = game::ScriptedTrack();

            // (1) The captured frame's world/state: step the game up to kGameCaptureStep.
            physics::World capWorld;
            game::GameState capState = game::MakeRollGame(capWorld);
            for (int s = 0; s < kGameCaptureStep && s < (int)track.size(); ++s)
                game::StepGame(capWorld, capState, track[s], dtG);
            const physics::RigidBody& capPlayer =
                capWorld.bodies[(size_t)capState.playerBodyIndex];

            // (2) The reported FINAL state: the full winning playthrough (independent world, so the
            // capture-step state is unaffected). Deterministic.
            physics::World finWorld;
            game::GameState finState = game::MakeRollGame(finWorld);
            for (const auto& in : track) game::StepGame(finWorld, finState, in, dtG);
            const physics::RigidBody& finPlayer =
                finWorld.bodies[(size_t)finState.playerBodyIndex];

            // --- Static lit pipeline (player + pickups + ground) + static shadow + sky + post. ------
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

            auto rt = device->CreateRenderTarget(w, h);
            auto shadowMap = device->CreateShadowMap(2048);
            device->SetShadowMap(*shadowMap);

            // Ground (checkerboard) + a flat normal map + distinct 1x1 solid-color tints: blue player,
            // gold pickups. These tint `albedo` via gTex in lit.frag (vertex color is white).
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
            // Player model: unit sphere (r=0.5) scaled to the collider diameter.
            Mat4 playerModel = capPlayer.Transform();

            // The remaining (uncollected) pickups at the capture step -> their model matrices.
            std::vector<Mat4> pickupModels;
            for (const auto& p : capState.pickups) {
                if (p.collected) continue;
                Mat4 m = math::FromTRS(p.pos, math::Quat::Identity(),
                                       {2.0f * p.radius, 2.0f * p.radius, 2.0f * p.radius});
                pickupModels.push_back(m);
            }

            // Camera framing the action at the capture step: the player (~5.4, 0.5, 0.8) and the
            // remaining pickup (5, 0.3, 2.5). A raised 3/4 view centered on their midpoint.
            const Vec3 eye{9.5f, 4.0f, 6.5f};
            const Vec3 center{5.0f, 0.4f, 1.6f};
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
                Vec3 sc{5.0f, 0.4f, 1.6f};
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

            // Helper: a 20-float push constant = { model(16); metallic, roughness, 0, 0 }.
            auto litPush = [](const Mat4& model, float metallic, float roughness) {
                std::array<float, 20> pc{};
                for (int k = 0; k < 16; ++k) pc[k] = model.m[k];
                pc[16] = metallic; pc[17] = roughness; pc[18] = 0.0f; pc[19] = 0.0f;
                return pc;
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
                    // Ground.
                    {
                        auto pc = litPush(groundModel, 0.0f, 0.85f);
                        cmd.PushConstants(pc.data(), sizeof(float) * 20);
                        cmd.BindMaterial(*groundTex, *flatNormal);
                        cmd.BindVertexBuffer(plane.vertices());
                        cmd.BindIndexBuffer(plane.indices());
                        cmd.DrawIndexed(plane.indexCount());
                    }
                    // Player sphere (blue).
                    {
                        auto pc = litPush(playerModel, 0.1f, 0.4f);
                        cmd.PushConstants(pc.data(), sizeof(float) * 20);
                        cmd.BindMaterial(*playerTex, *flatNormal);
                        cmd.BindVertexBuffer(sphere.vertices());
                        cmd.BindIndexBuffer(sphere.indices());
                        cmd.DrawIndexed(sphere.indexCount());
                    }
                    // Remaining pickups (gold).
                    {
                        cmd.BindMaterial(*pickupTex, *flatNormal);
                        cmd.BindVertexBuffer(sphere.vertices());
                        cmd.BindIndexBuffer(sphere.indices());
                        for (const Mat4& pm : pickupModels) {
                            auto pc = litPush(pm, 0.3f, 0.3f);
                            cmd.PushConstants(pc.data(), sizeof(float) * 20);
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
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            // Deterministic game-state line (the FULL winning playthrough).
            std::printf("game: {score:%d, won:%s, steps:%d, player:[%.3f, %.3f, %.3f]}\n",
                        finState.score, finState.won ? "true" : "false", finState.step,
                        finPlayer.position.x, finPlayer.position.y, finPlayer.position.z);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(gameShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — capture step %d, %zu pickups remaining\n",
                                    gameShotPath, cw, ch2, kGameCaptureStep, pickupModels.size());
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", gameShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- Text / HUD showcases (--hud-shot + --game-hud-shot, Slice BA): a self-contained capture
        // path that renders a lit + shadowed scene + a deterministic screen-space HUD text overlay
        // (the baked 8x8 font atlas + alpha-blended quad batch through the new text pipeline, drawn
        // OVER the final target in the post pass). --hud-shot draws the standard scene + a full HUD
        // ("HAZARD FORGE" + "SCORE: 0" + a fixed stat line, NO clock). --game-hud-shot draws the AX
        // game scene (identical to --game-shot) + a "SCORE: N" overlay (N from the deterministic
        // GameState at the capture step), with its OWN golden so game.png stays byte-identical. Fixed
        // text/positions/scale/color -> deterministic. One BMP -> exit. ------------------------------
        if (hudShotPath || gameHudShotPath) {
            using math::Mat4; using math::Vec3;
            const bool gameMode = (gameHudShotPath != nullptr);
            const char* outPath = gameMode ? gameHudShotPath : hudShotPath;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // Deterministic roll-a-ball scene at the fixed mid-track capture step (identical recipe
            // to --game-shot, so the underlying scene matches the existing game golden). The HUD score
            // is read from capState.score at this step.
            const int kGameCaptureStep = 250;
            const float dtG = 1.0f / 120.0f;
            std::vector<game::GameInput> track = game::ScriptedTrack();
            physics::World capWorld;
            game::GameState capState = game::MakeRollGame(capWorld);
            for (int s = 0; s < kGameCaptureStep && s < (int)track.size(); ++s)
                game::StepGame(capWorld, capState, track[s], dtG);
            const physics::RigidBody& capPlayer = capWorld.bodies[(size_t)capState.playerBodyIndex];

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

            // The reusable HUD text overlay (baked-font atlas + alphaBlend text pipeline).
            HudOverlay hud(*device, device->Swapchain().ColorFormat());

            auto rt = device->CreateRenderTarget(w, h);
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
                Vec3 sc{5.0f, 0.4f, 1.6f};
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

            auto litPush = [](const Mat4& model, float metallic, float roughness) {
                std::array<float, 20> pc{};
                for (int k = 0; k < 16; ++k) pc[k] = model.m[k];
                pc[16] = metallic; pc[17] = roughness; pc[18] = 0.0f; pc[19] = 0.0f;
                return pc;
            };

            // --- Deterministic HUD content (fixed text/positions/scale/color — NO clock). ---
            std::vector<HudLine> hudLines;
            char scoreBuf[32];
            std::snprintf(scoreBuf, sizeof(scoreBuf), "SCORE: %d", capState.score);
            if (gameMode) {
                // Game HUD: just the live score, top-left, bright white.
                hudLines.push_back({scoreBuf, 32.0f, 32.0f, 4.0f, {1.0f, 1.0f, 1.0f, 1.0f}});
            } else {
                // Full HUD: title + score + a fixed stat line.
                hudLines.push_back({"HAZARD FORGE", 32.0f, 28.0f, 5.0f, {1.0f, 0.85f, 0.2f, 1.0f}});
                hudLines.push_back({"SCORE: 0",     32.0f, 76.0f, 4.0f, {1.0f, 1.0f, 1.0f, 1.0f}});
                hudLines.push_back({"PICKUPS: 3  LEVEL: 1", 32.0f, 112.0f, 3.0f, {0.6f, 0.9f, 1.0f, 1.0f}});
            }
            std::vector<std::unique_ptr<rhi::IBuffer>> hudKeepAlive;

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
                        auto pc = litPush(groundModel, 0.0f, 0.85f);
                        cmd.PushConstants(pc.data(), sizeof(float) * 20);
                        cmd.BindMaterial(*groundTex, *flatNormal);
                        cmd.BindVertexBuffer(plane.vertices());
                        cmd.BindIndexBuffer(plane.indices());
                        cmd.DrawIndexed(plane.indexCount());
                    }
                    {
                        auto pc = litPush(playerModel, 0.1f, 0.4f);
                        cmd.PushConstants(pc.data(), sizeof(float) * 20);
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
                            auto pc = litPush(pm, 0.3f, 0.3f);
                            cmd.PushConstants(pc.data(), sizeof(float) * 20);
                            cmd.DrawIndexed(sphere.indexCount());
                        }
                    }
                    cmd.EndRenderPass();
                });

            graph.AddPass("post", {rgScene}, {rgSwap},
                [&](rhi::IRHIDevice& dev, rhi::ICommandBuffer& cmd) {
                    cmd.BeginRenderPass(rhi::ClearColor{0, 0, 0, 1});
                    cmd.BindPipeline(*postPipe);
                    cmd.BindTexture(*rt);
                    cmd.Draw(3);
                    // HUD overlay: alpha-blended text quads OVER the tonemapped scene, in the SAME
                    // final-target pass (after post). Reuses the existing alpha-blend + texture paths.
                    hud.Draw(dev, cmd, hudLines, (int)w, (int)h, hudKeepAlive);
                    cmd.EndRenderPass();
                });

            device->CaptureNextFrame();
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(outPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u) — %s HUD, score %d\n",
                                    outPath, cw, ch2, gameMode ? "game" : "standard", capState.score);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", outPath);
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

        // --- Data-driven material-graph showcase (--material-shot, Slice AV): a self-contained
        // capture path that does NOT touch the default scene. Ground plane + procedural sky + a sphere
        // shaded by the BUILD-TIME-GENERATED fragment shader (shaders/generated/mat_showcase.frag.hlsl,
        // codegen'd from assets/materials/showcase.mat.json). The material pipeline REUSES the existing
        // lit.vert + the wider PBR material set/descriptor layout (pbrMaterial=true, BindMaterialPBR) —
        // NO new RHI seam. The graph computes baseColor = Lerp(TextureSample(checker), tint,
        // Fresnel(power=3)) so the sphere shows a textured base with a warm fresnel rim. One frame ->
        // BMP -> exit. -------------------------------------------------------------------------------
        // Slice AW: --material-live-shot routes through the SAME render block but sources the material
        // fragment SPIR-V from the RUNTIME compile path (in-process codegen -> dxc subprocess ->
        // SPIR-V) instead of the build-time .spv. For showcase.mat.json the SAME dxc + flags yield
        // BYTE-IDENTICAL SPIR-V -> a byte-identical image to --material-shot (the runtime==build-time
        // proof). `outPath` is whichever flag was given.
        const char* matOutPath = materialShotPath ? materialShotPath : materialLiveShotPath;
        if (matOutPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // Material pipeline: shared lit.vert + the material fragment; wider PBR material set
            // (pbrMaterial=true) so BindMaterialPBR binds the textures (the showcase graphs read the
            // base-color slot; the layout is the existing PBR one).
            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            std::vector<uint32_t> matFsWords;
            if (materialLiveShotPath) {
                // RUNTIME path: load the selected .mat.json -> validate -> dxc-subprocess compile.
                const char* matJson = materialLiveShotMat ? materialLiveShotMat : HF_MAT_JSON;
                material::LoadResult lr = material::LoadGraphFromFile(matJson);
                if (!lr.ok) {
                    std::fprintf(stderr, "FATAL: --material-live-shot load failed: %s\n", lr.error.c_str());
                    return 1;
                }
                std::string cerr;
                auto spv = material::CompileGraphToSpirv(lr.graph, &cerr);
                if (!spv) {
                    std::fprintf(stderr, "FATAL: --material-live-shot runtime compile failed: %s\n",
                                 cerr.c_str());
                    return 1;
                }
                matFsWords = std::move(*spv);
                std::printf("material-live-shot: runtime-compiled %s (%zu SPIR-V words)\n",
                            matJson, matFsWords.size());
            } else {
                // BUILD-TIME path (--material-shot, Slice AV): the committed generated .spv.
                matFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/mat_showcase.frag.hlsl.spv");
            }
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto matFs = device->CreateShaderModule({std::span<const uint32_t>(matFsWords)});
            rhi::GraphicsPipelineDesc matDesc;
            matDesc.vertex = litVs.get();
            matDesc.fragment = matFs.get();
            matDesc.vertexLayout = scene::MeshVertexLayout();
            matDesc.colorFormat = device->Swapchain().ColorFormat();
            matDesc.depthTest = true;
            matDesc.usesFrameUniforms = true;
            matDesc.usesTexture = true;
            matDesc.pbrMaterial = true;               // reuse the existing 5-texture material set
            matDesc.pushConstantSize = sizeof(float) * 20;  // mat4 model + float4 material
            auto matPipeline = device->CreateGraphicsPipeline(matDesc);

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

            // Static depth-only shadow pipeline (ground + sphere casters).
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

            // Checkerboard base-color texture (read by the graph's TextureSample(checker) node) + the
            // flat dummies for the rest of the PBR material slots (unused by the showcase graph).
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

            // Sphere placement: sit a unit sphere on the ground (radius 1, scaled to ~1.0 tall).
            const float sphereR = 1.0f;
            Mat4 sphereModel = Mat4::Translate({0.0f, sphereR, 0.0f}) * Mat4::Scale({sphereR, sphereR, sphereR});

            const Vec3 eye{2.0f, 1.7f, 2.8f};
            const Vec3 center{0.0f, 0.95f, 0.0f};
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
                    // Sphere (data-driven material graph). The checker is bound at the base-color slot
                    // (the graph's TextureSample(checker) reads it); the other slots get neutral dummies.
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
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(matOutPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u)\n", matOutPath, cw, ch2);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", matOutPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- Multi-material scene (--material-multi-shot, Slice AZ): three spheres in a row, each
        // shaded by a DISTINCT graph material (showcase / showcase2 / showcase3) via the EXISTING
        // material pipeline — one draw per material (bind that material's pipeline + its push
        // constant, draw its sphere). + ground plane + procedural sky + the standard light, lit +
        // shadowed. Reuses lit.vert + the PBR descriptor layout (pbrMaterial=true); NO new RHI seam.
        // showcase3 exercises the Slice-AZ node expansion (Swizzle/MakeFloat3/Power/Saturate/OneMinus
        // — incl. the gap-closing scalar-from-float4 Swizzle). Deterministic fixed camera/light. One
        // frame -> BMP -> exit. ----------------------------------------------------------------------
        if (materialMultiShotPath) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});

            // Three distinct material pipelines from the committed build-time generated .spv (each
            // codegen'd from its own .mat.json). Same descriptor layout / push-constant size; only the
            // fragment differs.
            const char* matSpv[3] = {"/mat_showcase.frag.hlsl.spv",
                                     "/mat_showcase2.frag.hlsl.spv",
                                     "/mat_showcase3.frag.hlsl.spv"};
            std::vector<std::unique_ptr<rhi::IShaderModule>> matFs;
            std::vector<std::unique_ptr<rhi::IPipeline>> matPipes;
            for (int m = 0; m < 3; ++m) {
                auto words = LoadSpirv(std::string(HF_SHADER_DIR) + matSpv[m]);
                matFs.push_back(device->CreateShaderModule({std::span<const uint32_t>(words)}));
                rhi::GraphicsPipelineDesc d;
                d.vertex = litVs.get();
                d.fragment = matFs.back().get();
                d.vertexLayout = scene::MeshVertexLayout();
                d.colorFormat = device->Swapchain().ColorFormat();
                d.depthTest = true;
                d.usesFrameUniforms = true;
                d.usesTexture = true;
                d.pbrMaterial = true;
                d.pushConstantSize = sizeof(float) * 20;
                matPipes.push_back(device->CreateGraphicsPipeline(d));
            }

            // Static lit pipeline for the ground plane.
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

            // Static depth-only shadow pipeline (ground + 3 spheres as casters).
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

            // Three spheres in a row along X (radius 0.8, spacing 2.0), sitting on the ground.
            const float sphereR = 0.8f;
            const float spacing = 2.0f;
            Mat4 sphereModel[3];
            for (int m = 0; m < 3; ++m) {
                float x = (float)(m - 1) * spacing;
                sphereModel[m] = Mat4::Translate({x, sphereR, 0.0f}) *
                                 Mat4::Scale({sphereR, sphereR, sphereR});
            }

            // Deterministic fixed camera framing all three spheres + ground + the standard light.
            const Vec3 eye{0.0f, 1.9f, 4.4f};
            const Vec3 center{0.0f, 0.7f, 0.0f};
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
                    // Three spheres, one draw per material: bind that material's pipeline, push its
                    // model transform, bind the (shared) PBR textures, draw the sphere.
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
            graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
            graph.Execute(*device);

            std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
            bool ok = false;
            if (device->GetCapturedPixels(px, cw, ch2)) {
                ok = WriteBMP(materialMultiShotPath, px, cw, ch2);
                if (ok) std::printf("wrote %s (%ux%u)\n", materialMultiShotPath, cw, ch2);
                else std::fprintf(stderr, "FATAL: could not write BMP to %s\n", materialMultiShotPath);
            } else {
                std::fprintf(stderr, "FATAL: no captured pixels\n");
            }
            device->WaitIdle();
            return ok ? 0 : 1;
        }

        // --- Live material hot-swap dry-run (--material-hotswap-dry-run, Slice AW): exercise the
        // live edit->recompile->swap loop DETERMINISTICALLY + headlessly (no GUI). Drive the
        // material::LiveMaterial controller with the REAL dxc-subprocess compiler: load showcase ->
        // render the material sphere scene -> FNV-hash the pixels; then SwitchTo showcase2 (a runtime
        // recompile + pipeline rebuild) -> render -> hash. Asserts: BOTH compiles succeeded (the swap
        // happened, swapCount==2), the two materials produce DIFFERENT images (the swap is visible),
        // and re-rendering showcase reproduces its first hash (determinism). Proves the live swap path
        // works end-to-end without a window and without leaking/crashing. -----------------------------
        if (materialHotswapDryRun) {
            using math::Mat4; using math::Vec3;
            uint32_t w = window.FramebufferWidth();
            uint32_t h = window.FramebufferHeight();
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            // Static pipelines + resources shared across both renders (built ONCE; only the material
            // pipeline is rebuilt per swap, exactly as the live loop does).
            auto litVsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.vert.hlsl.spv");
            auto litVs = device->CreateShaderModule({std::span<const uint32_t>(litVsWords)});
            auto litFsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/lit.frag.hlsl.spv");
            auto litFs = device->CreateShaderModule({std::span<const uint32_t>(litFsWords)});
            rhi::GraphicsPipelineDesc litDesc;
            litDesc.vertex = litVs.get(); litDesc.fragment = litFs.get();
            litDesc.vertexLayout = scene::MeshVertexLayout();
            litDesc.colorFormat = device->Swapchain().ColorFormat();
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
            shDesc.depthTest = true; shDesc.depthOnly = true;
            shDesc.usesFrameUniforms = true; shDesc.pushConstantSize = sizeof(float) * 16;
            auto shadowPipeline = device->CreateGraphicsPipeline(shDesc);

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

            // Render the material scene with the given material-fragment SPIR-V; return the FNV-1a
            // hash of the captured pixels (0 on capture failure). The material pipeline is built FRESH
            // each call (the live-swap rebuild); everything else is reused.
            auto renderWithMaterial = [&](const std::vector<uint32_t>& matFsWords) -> uint64_t {
                auto matFs = device->CreateShaderModule({std::span<const uint32_t>(matFsWords)});
                rhi::GraphicsPipelineDesc matDesc;
                matDesc.vertex = litVs.get(); matDesc.fragment = matFs.get();
                matDesc.vertexLayout = scene::MeshVertexLayout();
                matDesc.colorFormat = device->Swapchain().ColorFormat();
                matDesc.depthTest = true; matDesc.usesFrameUniforms = true;
                matDesc.usesTexture = true; matDesc.pbrMaterial = true;
                matDesc.pushConstantSize = sizeof(float) * 20;
                auto matPipeline = device->CreateGraphicsPipeline(matDesc);

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
                graph.SetSwapchainRetryArm([&] { device->CaptureNextFrame(); });
                graph.Execute(*device);

                std::vector<uint8_t> px; uint32_t cw = 0, ch2 = 0;
                if (!device->GetCapturedPixels(px, cw, ch2)) return 0;
                uint64_t hash = 1469598103934665603ull;
                for (uint8_t byte : px) { hash ^= byte; hash *= 1099511628211ull; }
                device->WaitIdle();
                return hash;
            };

            // Drive the live-swap controller with the REAL dxc-subprocess compiler.
            material::LiveMaterial live(HF_MAT_JSON, &material::CompileGraphToSpirv);
            if (!live.LoadInitial()) {
                std::fprintf(stderr, "FATAL: hotswap dry-run: initial compile of %s failed: %s\n",
                             HF_MAT_JSON, live.lastError().c_str());
                return 1;
            }
            uint64_t hashA = renderWithMaterial(live.activeSpirv());
            uint64_t hashA2 = renderWithMaterial(live.activeSpirv());  // determinism re-render.

            material::ReloadStatus st = live.SwitchTo(HF_MAT2_JSON);
            if (st != material::ReloadStatus::Swapped) {
                std::fprintf(stderr, "FATAL: hotswap dry-run: swap to %s did not happen: %s\n",
                             HF_MAT2_JSON, live.lastError().c_str());
                return 1;
            }
            uint64_t hashB = renderWithMaterial(live.activeSpirv());

            std::printf("hotswap-dry-run: showcase hash  = %016llx\n", (unsigned long long)hashA);
            std::printf("hotswap-dry-run: showcase2 hash = %016llx\n", (unsigned long long)hashB);
            std::printf("hotswap-dry-run: swapCount = %d\n", live.swapCount());

            bool ok = true;
            if (hashA == 0 || hashB == 0) { std::fprintf(stderr, "FATAL: dry-run: capture failed\n"); ok = false; }
            if (hashA != hashA2) { std::fprintf(stderr, "FATAL: dry-run: showcase render not deterministic\n"); ok = false; }
            if (hashA == hashB) { std::fprintf(stderr, "FATAL: dry-run: swap produced no visible change (A==B)\n"); ok = false; }
            if (live.swapCount() != 2) { std::fprintf(stderr, "FATAL: dry-run: expected 2 swaps, got %d\n", live.swapCount()); ok = false; }
            device->WaitIdle();
            if (ok) std::printf("hotswap-dry-run: PASS (A->B swap deterministic, visible, no crash)\n");
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

        // --introspect [outpath]: write the FULL machine-readable engine-state JSON (the agent-facing
        // OBSERVE call) and exit. We fill a representative EngineState from the SAME values the
        // interactive showcase uses below (eye/center camera, the warm key directional, and the three
        // colored point lights at their t=0 phase), so the dump describes the live default scene.
        // Deterministic + backend-agnostic (DescribeEngine is pure hf_core).
        if (introspect) {
            using math::Vec3;
            editor::EngineState state;
            state.backend = "vulkan";

            // Camera: matches makeFrameData's eye/center + 60deg vertical FOV. yaw/pitch are derived
            // from the look direction (eye -> center) so an agent sees the actual framing.
            const Vec3 eye{4.5f, 4.0f, 6.5f};
            const Vec3 center{0.0f, 0.5f, 0.0f};
            Vec3 fwd = math::normalize(center - eye);
            state.hasCamera = true;
            state.camera.position = eye;
            state.camera.yaw = std::atan2(fwd.x, -fwd.z);
            state.camera.pitch = std::asin(fwd.y);
            state.camera.fovDeg = 60.0f;

            // Directional key light (matches makeFrameData's lightDir/lightColor).
            state.hasDirectional = true;
            state.directional.dir = {-0.5f, -1.0f, -0.3f};
            state.directional.color = {0.95f, 0.93f, 0.85f};

            // Three colored point lights at their t=0 phase (matches makeFrameData's accent lights).
            const float kR = 3.0f, kH = 1.1f, kRadius = 3.2f, kInt = 1.0f;
            const float ptColors[3][3] = {{1.0f, 0.25f, 0.2f},   // warm red
                                          {0.2f, 1.0f, 0.35f},   // green
                                          {0.3f, 0.45f, 1.0f}};  // blue
            for (int li = 0; li < 3; ++li) {
                float a = (float)li * 2.0943951f;  // t=0; 120deg apart
                editor::LightPoint p;
                p.pos = {std::cos(a) * kR, kH, std::sin(a) * kR};
                p.color = {ptColors[li][0], ptColors[li][1], ptColors[li][2]};
                p.radius = kRadius;
                p.intensity = kInt;
                state.points.push_back(p);
            }

            std::string json = editor::DescribeEngine(registry, resources, state);
            if (introspectPath) {
                std::ofstream f(introspectPath, std::ios::binary);
                if (!f) {
                    std::fprintf(stderr, "FATAL: cannot write introspect output '%s'\n", introspectPath);
                    device->WaitIdle();
                    return 1;
                }
                f << json;
                std::printf("introspect: wrote %s\n", introspectPath);
            } else {
                std::fputs(json.c_str(), stdout);
            }
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
