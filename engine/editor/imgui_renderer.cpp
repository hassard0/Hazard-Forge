#include "editor/imgui_renderer.h"

#include "imgui.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace hf::editor {

namespace {

std::vector<uint32_t> LoadSpirv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("ImGuiRenderer: cannot open shader: " + path);
    std::streamsize size = f.tellg();
    if (size % 4 != 0) throw std::runtime_error("ImGuiRenderer: SPIR-V not multiple of 4: " + path);
    f.seekg(0);
    std::vector<uint32_t> words(static_cast<size_t>(size) / 4);
    f.read(reinterpret_cast<char*>(words.data()), size);
    return words;
}

// The vertex push constant: screen-space ortho as {scale.xy, translate.xy}. Matches ui.vert.hlsl.
struct UiPush { float scale[2]; float translate[2]; };

}  // namespace

ImGuiRenderer::ImGuiRenderer(rhi::IRHIDevice& device, rhi::Format swapchainColorFormat,
                             const std::string& shaderDir)
    : device_(device) {
    auto vsWords = LoadSpirv(shaderDir + "/ui.vert.hlsl.spv");
    auto fsWords = LoadSpirv(shaderDir + "/ui.frag.hlsl.spv");
    vs_ = device_.CreateShaderModule({std::span<const uint32_t>(vsWords)});
    fs_ = device_.CreateShaderModule({std::span<const uint32_t>(fsWords)});

    // --- Font atlas -> rhi::ITexture (RGBA8). ---
    ImGuiIO& io = ImGui::GetIO();
    unsigned char* pixels = nullptr;
    int texW = 0, texH = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &texW, &texH);
    rhi::TextureDesc td;
    td.width = static_cast<uint32_t>(texW);
    td.height = static_cast<uint32_t>(texH);
    td.format = rhi::Format::RGBA8_UNorm;
    td.data = pixels;
    td.dataSize = static_cast<uint64_t>(texW) * texH * 4;
    fontAtlas_ = device_.CreateTexture(td);
    // ImGui wants a texture id so it can reference the font in draw cmds; we always bind the font
    // atlas (single texture editor), so the id is just a non-null marker.
    io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(fontAtlas_.get()));

    // --- UI graphics pipeline: ImDrawVert layout, alpha blend, no depth, ortho push constant. ---
    rhi::GraphicsPipelineDesc pd;
    pd.vertex = vs_.get();
    pd.fragment = fs_.get();
    pd.vertexLayout.stride = sizeof(ImDrawVert);
    pd.vertexLayout.attributes = {
        {0, rhi::Format::RG32_Float, (uint32_t)offsetof(ImDrawVert, pos)},
        {1, rhi::Format::RG32_Float, (uint32_t)offsetof(ImDrawVert, uv)},
        {2, rhi::Format::RGBA8_UNorm, (uint32_t)offsetof(ImDrawVert, col)},
    };
    pd.colorFormat = swapchainColorFormat;
    pd.depthTest = false;            // UI overlay: no depth test/write
    pd.usesFrameUniforms = false;    // no per-frame UBO -> material/texture set is set 0
    pd.usesTexture = true;           // font atlas at the material set (set 0)
    pd.alphaBlend = true;            // src_alpha / one_minus_src_alpha
    pd.cullNone = true;              // ImGui quads are clockwise-wound — must NOT be back-face culled
    pd.pushConstantSize = sizeof(UiPush);
    pipeline_ = device_.CreateGraphicsPipeline(pd);
}

ImGuiRenderer::~ImGuiRenderer() = default;

void ImGuiRenderer::RenderDrawData(const ImDrawData* drawData, rhi::ICommandBuffer& cmd,
                                   uint32_t fbWidth, uint32_t fbHeight) {
    if (!drawData || drawData->CmdListsCount == 0) return;
    if (drawData->TotalVtxCount == 0) return;

    // --- Upload all cmd-list vertices/indices into one combined vertex + index buffer. Each draw
    // cmd then offsets into them via firstIndex + vertexOffset (so we keep one bind for the frame). ---
    const size_t totalVtx = static_cast<size_t>(drawData->TotalVtxCount);
    const size_t totalIdx = static_cast<size_t>(drawData->TotalIdxCount);

    std::vector<ImDrawVert> verts(totalVtx);
    std::vector<ImDrawIdx> indices(totalIdx);
    size_t vtxOff = 0, idxOff = 0;
    for (int n = 0; n < drawData->CmdListsCount; ++n) {
        const ImDrawList* cl = drawData->CmdLists[n];
        std::copy(cl->VtxBuffer.Data, cl->VtxBuffer.Data + cl->VtxBuffer.Size, verts.begin() + vtxOff);
        std::copy(cl->IdxBuffer.Data, cl->IdxBuffer.Data + cl->IdxBuffer.Size, indices.begin() + idxOff);
        vtxOff += cl->VtxBuffer.Size;
        idxOff += cl->IdxBuffer.Size;
    }

    // Recreate the GPU buffers when capacity is insufficient (host-visible, upload on create).
    if (totalVtx > vertexCapacity_) {
        rhi::BufferDesc bd;
        bd.size = totalVtx * sizeof(ImDrawVert);
        bd.initialData = verts.data();
        bd.usage = rhi::BufferUsage::Vertex;
        vertexBuffer_ = device_.CreateBuffer(bd);
        vertexCapacity_ = totalVtx;
    } else {
        // Same-size path: recreate anyway (simplest correct upload; editor vertex counts are small).
        rhi::BufferDesc bd;
        bd.size = vertexCapacity_ * sizeof(ImDrawVert);
        bd.initialData = verts.data();
        bd.usage = rhi::BufferUsage::Vertex;
        vertexBuffer_ = device_.CreateBuffer(bd);
    }
    if (totalIdx > indexCapacity_) {
        rhi::BufferDesc bd;
        bd.size = totalIdx * sizeof(ImDrawIdx);
        bd.initialData = indices.data();
        bd.usage = rhi::BufferUsage::Index;
        indexBuffer_ = device_.CreateBuffer(bd);
        indexCapacity_ = totalIdx;
    } else {
        rhi::BufferDesc bd;
        bd.size = indexCapacity_ * sizeof(ImDrawIdx);
        bd.initialData = indices.data();
        bd.usage = rhi::BufferUsage::Index;
        indexBuffer_ = device_.CreateBuffer(bd);
    }

    // --- Record the draws. ---
    cmd.BindPipeline(*pipeline_);

    // Screen-space ortho: maps pixel coords [0,fb] -> clip [-1,1]. Vulkan clip-space Y is down so the
    // +Y-down ImGui coords map directly with a positive Y scale (matches the lit pass conventions).
    UiPush push{};
    push.scale[0] = 2.0f / drawData->DisplaySize.x;
    push.scale[1] = 2.0f / drawData->DisplaySize.y;
    push.translate[0] = -1.0f - drawData->DisplayPos.x * push.scale[0];
    push.translate[1] = -1.0f - drawData->DisplayPos.y * push.scale[1];
    cmd.PushConstants(&push, sizeof(push));

    cmd.BindVertexBuffer(*vertexBuffer_);
    cmd.BindIndexBuffer(*indexBuffer_);
    cmd.BindTexture(*fontAtlas_);

    const ImVec2 clipOff = drawData->DisplayPos;       // (0,0) unless multi-viewport
    const ImVec2 clipScale = drawData->FramebufferScale;  // (1,1) for our fixed display size

    uint32_t globalVtxOffset = 0;
    uint32_t globalIdxOffset = 0;
    for (int n = 0; n < drawData->CmdListsCount; ++n) {
        const ImDrawList* cl = drawData->CmdLists[n];
        for (int c = 0; c < cl->CmdBuffer.Size; ++c) {
            const ImDrawCmd& pcmd = cl->CmdBuffer[c];
            if (pcmd.UserCallback != nullptr) {
                // We do not use user callbacks; skip safely.
                continue;
            }
            // Clip rect in framebuffer pixels.
            float clipMinX = (pcmd.ClipRect.x - clipOff.x) * clipScale.x;
            float clipMinY = (pcmd.ClipRect.y - clipOff.y) * clipScale.y;
            float clipMaxX = (pcmd.ClipRect.z - clipOff.x) * clipScale.x;
            float clipMaxY = (pcmd.ClipRect.w - clipOff.y) * clipScale.y;
            if (clipMinX < 0.0f) clipMinX = 0.0f;
            if (clipMinY < 0.0f) clipMinY = 0.0f;
            if (clipMaxX > (float)fbWidth)  clipMaxX = (float)fbWidth;
            if (clipMaxY > (float)fbHeight) clipMaxY = (float)fbHeight;
            if (clipMaxX <= clipMinX || clipMaxY <= clipMinY) continue;

            cmd.SetScissor((int32_t)clipMinX, (int32_t)clipMinY,
                           (uint32_t)(clipMaxX - clipMinX), (uint32_t)(clipMaxY - clipMinY));

            // The font atlas is the only texture; it is already bound. Issue the indexed draw with
            // the per-cmd-list base vertex + first index offsets.
            cmd.DrawIndexed(pcmd.ElemCount,
                            pcmd.IdxOffset + globalIdxOffset,
                            (int32_t)(pcmd.VtxOffset + globalVtxOffset));
        }
        globalIdxOffset += (uint32_t)cl->IdxBuffer.Size;
        globalVtxOffset += (uint32_t)cl->VtxBuffer.Size;
    }
}

}  // namespace hf::editor
