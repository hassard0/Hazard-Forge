#pragma once
// Slice BZ — Bindless textures. Pure-CPU TEXTURE-INDEX TABLE (header-only, no device, no backend
// symbols). Assigns each UNIQUE scene texture a stable index in a single large descriptor ARRAY and
// maps each material to its texture index, so a draw can sample its base-color texture by INDEX
// (textures[NonUniformResourceIndex(idx)]) instead of rebinding a per-material descriptor set.
//
// The Vulkan bindless path fills ONE runtime descriptor array with `table.textures` IN ORDER (index i
// -> textures[i]) and binds it ONCE; each draw pushes its material's `Intern`ed index. The bound
// REFERENCE path binds the SAME `ITexture*` as a per-material material set. Because both paths sample
// the SAME texel of the SAME texture, the bindless render is BYTE-IDENTICAL to the bound render — that
// equality (plus "one texture binding for the whole scene") is the proof (asserted in --bindless-shot).
//
// PURE CPU: `ITexture*` is treated as an OPAQUE handle — never dereferenced — so this builder is a
// plain pointer-interning data structure, unit-testable with no GPU (tests/bindless_test.cpp).
//
// DETERMINISM: indices are assigned in FIRST-INSERTION order (the order Intern first sees a texture),
// so two builds over the same material list produce identical index assignments (pinned by the test).

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace hf::rhi { class ITexture; }

namespace hf::render::bindless {

// The texture-index table: `textures[i]` is the i-th unique texture (the array the Vulkan descriptor
// array is filled with, in order); `indexOf[t]` is texture t's stable index. Built by Intern-ing each
// material's textures.
struct BindlessTable {
    std::vector<hf::rhi::ITexture*> textures;                  // index -> texture (insertion order)
    std::unordered_map<hf::rhi::ITexture*, uint32_t> indexOf;  // texture -> stable index

    uint32_t size() const { return (uint32_t)textures.size(); }
};

// Return the stable index for `tex`, adding it to the table on first sight (deterministic
// first-insertion order). Re-interning the SAME pointer returns the SAME index (dedup); distinct
// pointers get distinct, increasing indices. nullptr is a valid opaque handle (interned like any
// other), but scenes never pass it.
inline uint32_t Intern(BindlessTable& table, hf::rhi::ITexture* tex) {
    auto it = table.indexOf.find(tex);
    if (it != table.indexOf.end()) return it->second;
    uint32_t idx = (uint32_t)table.textures.size();
    table.textures.push_back(tex);
    table.indexOf.emplace(tex, idx);
    return idx;
}

// A scene material referencing ONE base-color texture (the texture the lit fragment samples). Opaque
// handle; the index is resolved via Intern. (Bindless BUFFERS / multi-texture-per-material is YAGNI
// this slice — one sampled-image array indexed by the base-color texture, byte-identical to the bound
// path.)
struct Material {
    hf::rhi::ITexture* baseColor = nullptr;
};

// Build the table over `materials` IN ORDER: Intern each material's base-color texture, recording the
// resolved per-material index into `outIndices` (parallel to `materials`). Identical base-color
// textures share one index (dedup); the table's `textures` array is the deduplicated set in
// first-insertion order. Pure function — deterministic, no GPU. The showcase fills the Vulkan
// descriptor array with `table.textures` and pushes `outIndices[i]` as draw i's texIndex.
inline BindlessTable BuildTable(const std::vector<Material>& materials,
                                std::vector<uint32_t>& outIndices) {
    BindlessTable table;
    outIndices.clear();
    outIndices.reserve(materials.size());
    for (const Material& m : materials) outIndices.push_back(Intern(table, m.baseColor));
    return table;
}

}  // namespace hf::render::bindless
