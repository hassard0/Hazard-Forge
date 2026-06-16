#pragma once
// Hazard Forge — Dear ImGui renderer THROUGH the engine RHI.
//
// This draws ImGui's draw data (vertex/index buffers + per-cmd clip rects + font atlas) using ONLY
// the rhi:: seam — no vk*/Metal symbols. It owns:
//   - a UI graphics pipeline (ImDrawVert layout, alpha blend, no depth, ortho push constant),
//   - the font atlas uploaded as an rhi::ITexture (RGBA8 from io.Fonts->GetTexDataAsRGBA32),
//   - per-frame vertex/index buffers (grown as needed) it uploads ImDrawData into.
//
// Usage per frame (inside an open render pass that targets the swapchain):
//   ImGui::NewFrame(); ... build panels ...; ImGui::Render();
//   uiRenderer.RenderDrawData(ImGui::GetDrawData(), cmd, fbWidth, fbHeight);
//
// Editor-only; this module is NOT part of the metal_headless target, so the Metal path is untouched.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rhi/rhi.h"

struct ImDrawData;  // forward decl — imgui.h is only needed in the .cpp

namespace hf::editor {

class ImGuiRenderer {
public:
    // Builds the UI pipeline + uploads the ImGui font atlas. `swapchainColorFormat` must match the
    // pass the UI draws into (the post/swapchain pass). `shaderDir` is where ui.vert/frag .spv live.
    // (Vulkan path: loads SPIR-V ui.vert.hlsl.spv / ui.frag.hlsl.spv from disk.)
    ImGuiRenderer(rhi::IRHIDevice& device, rhi::Format swapchainColorFormat,
                  const std::string& shaderDir);

    // Builds the UI pipeline from CALLER-SUPPLIED shader modules + uploads the font atlas. This keeps
    // the renderer RHI-only (no SPIR-V vs MSL load policy baked in): the Vulkan caller passes
    // CreateShaderModule(SPIR-V) modules, the Metal caller passes CreateShaderModuleMSL modules. The
    // pipeline state + draw path are identical. The renderer takes ownership of the modules.
    ImGuiRenderer(rhi::IRHIDevice& device, rhi::Format swapchainColorFormat,
                  std::unique_ptr<rhi::IShaderModule> vs, std::unique_ptr<rhi::IShaderModule> fs);
    ~ImGuiRenderer();

    ImGuiRenderer(const ImGuiRenderer&) = delete;
    ImGuiRenderer& operator=(const ImGuiRenderer&) = delete;

    // Record the draws for `drawData` into `cmd` (a command buffer with an OPEN render pass whose
    // color target is `fbWidth`x`fbHeight` swapchain). Uploads vertices/indices, sets per-cmd
    // scissor, binds the font atlas, and issues DrawIndexed with the right offsets.
    void RenderDrawData(const ImDrawData* drawData, rhi::ICommandBuffer& cmd,
                        uint32_t fbWidth, uint32_t fbHeight);

private:
    rhi::IRHIDevice& device_;

    std::unique_ptr<rhi::IShaderModule> vs_, fs_;
    std::unique_ptr<rhi::IPipeline> pipeline_;
    std::unique_ptr<rhi::ITexture> fontAtlas_;

    // Per-frame upload buffers, grown when a frame needs more than the current capacity. They are
    // host-visible (Vertex/Index usage) so the upload is a recreate-with-initialData each frame —
    // simple and correct for the editor's modest vertex counts.
    std::unique_ptr<rhi::IBuffer> vertexBuffer_;
    std::unique_ptr<rhi::IBuffer> indexBuffer_;
    size_t vertexCapacity_ = 0;  // in vertices
    size_t indexCapacity_ = 0;   // in indices
};

}  // namespace hf::editor
