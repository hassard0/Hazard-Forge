// Slice BZ — Bindless textures. Pure-CPU test of the texture-index TABLE / interner
// (engine/render/bindless.h, namespace hf::render::bindless). No GPU, no backend symbols — this pins
// the index-assignment contract the Vulkan bindless path (one runtime descriptor array filled with
// table.textures IN ORDER, each draw pushing its material's Intern'ed index) consumes:
//
//   * Interning: Intern returns a STABLE index per texture, DEDUPLICATES identical textures (same
//     pointer -> same index), assigns DISTINCT, increasing indices to distinct textures, in
//     first-insertion order.
//   * Material coverage: BuildTable over a known material set covers EVERY referenced texture exactly
//     once (the textures array is the deduplicated set), and each material resolves to the right index
//     (outIndices[i] == the index of materials[i].baseColor).
//   * Determinism: two independent builds of the same material list produce IDENTICAL index
//     assignments (same textures array, same per-material indices).
//
// `ITexture*` is an OPAQUE handle here (never dereferenced), so we use fabricated distinct pointers as
// stand-in texture handles — exactly how the builder treats them. The render-invariance proof (bindless
// image == bound image) lives in --bindless-shot; this test pins the CPU index table the GPU path reads.
//
// Pure C++ (hf_core), ASan-eligible like the other render-data tests.
#include "render/bindless.h"

#include <cstdio>
#include <vector>

namespace bl = hf::render::bindless;
namespace rhi = hf::rhi;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Fabricate N distinct opaque ITexture* handles (never dereferenced — the builder treats them as
// opaque keys). A byte array gives N guaranteed-distinct, stable addresses.
static rhi::ITexture* FakeTex(size_t i) {
    static unsigned char storage[64];
    return reinterpret_cast<rhi::ITexture*>(&storage[i]);
}

int main() {
    rhi::ITexture* tA = FakeTex(0);
    rhi::ITexture* tB = FakeTex(1);
    rhi::ITexture* tC = FakeTex(2);
    rhi::ITexture* tD = FakeTex(3);

    // ---- Interning: stable indices, dedup, distinct->distinct, first-insertion order. ----
    {
        bl::BindlessTable t;
        uint32_t iA0 = bl::Intern(t, tA);
        uint32_t iB0 = bl::Intern(t, tB);
        uint32_t iA1 = bl::Intern(t, tA);   // re-intern A
        uint32_t iC0 = bl::Intern(t, tC);
        uint32_t iB1 = bl::Intern(t, tB);   // re-intern B

        check(iA0 == 0, "first interned texture gets index 0");
        check(iB0 == 1, "second distinct texture gets index 1");
        check(iC0 == 2, "third distinct texture gets index 2");
        // Dedup: re-interning a known texture returns its ORIGINAL index (stable), no new slot.
        check(iA1 == iA0, "re-interning A returns the same (stable) index");
        check(iB1 == iB0, "re-interning B returns the same (stable) index");
        // Distinct textures -> distinct indices.
        check(iA0 != iB0 && iB0 != iC0 && iA0 != iC0, "distinct textures -> distinct indices");
        // The table holds exactly the 3 unique textures, in first-insertion order.
        check(t.size() == 3, "table has exactly 3 unique textures after dedup");
        check(t.textures.size() == 3, "textures array length == unique count");
        check(t.textures[0] == tA && t.textures[1] == tB && t.textures[2] == tC,
              "textures array is in first-insertion order (A,B,C)");
        // indexOf round-trips: textures[indexOf[t]] == t.
        check(t.indexOf.at(tA) == 0 && t.indexOf.at(tB) == 1 && t.indexOf.at(tC) == 2,
              "indexOf maps each texture to its array slot");
    }

    // ---- Material coverage: BuildTable over a known material set covers every texture; each
    //      material resolves to the right index. The set shares textures across materials (A used by
    //      materials 0 and 3; B by 1 and 4) so dedup is exercised. ----
    std::vector<bl::Material> materials = {
        {tA},  // 0
        {tB},  // 1
        {tC},  // 2
        {tA},  // 3 (shares A with 0)
        {tB},  // 4 (shares B with 1)
        {tD},  // 5
    };
    std::vector<uint32_t> indices;
    bl::BindlessTable table = bl::BuildTable(materials, indices);

    check(indices.size() == materials.size(), "one resolved index per material");
    // 4 unique textures (A,B,C,D) across 6 materials.
    check(table.size() == 4, "table covers exactly the 4 unique textures");
    // Coverage: every material's base texture is in the table, at the resolved index.
    bool coverageOk = true;
    for (size_t i = 0; i < materials.size(); ++i) {
        if (table.indexOf.find(materials[i].baseColor) == table.indexOf.end()) coverageOk = false;
        if (indices[i] != table.indexOf.at(materials[i].baseColor)) coverageOk = false;
        if (table.textures[indices[i]] != materials[i].baseColor) coverageOk = false;
    }
    check(coverageOk, "every material's base texture is covered + resolves to the right index");
    // The shared textures collapse to one index each (dedup across materials).
    check(indices[0] == indices[3], "materials 0 and 3 (both A) resolve to the same index");
    check(indices[1] == indices[4], "materials 1 and 4 (both B) resolve to the same index");
    // First-insertion order: A=0, B=1, C=2, D=3.
    check(indices[0] == 0 && indices[1] == 1 && indices[2] == 2 && indices[5] == 3,
          "indices follow first-insertion order (A0,B1,C2,D3)");

    // ---- Determinism: a second independent build is IDENTICAL (same textures array + per-material
    //      indices). ----
    std::vector<uint32_t> indices2;
    bl::BindlessTable table2 = bl::BuildTable(materials, indices2);
    bool sameTextures = (table.textures.size() == table2.textures.size());
    for (size_t i = 0; sameTextures && i < table.textures.size(); ++i)
        if (table.textures[i] != table2.textures[i]) sameTextures = false;
    bool sameIndices = (indices.size() == indices2.size());
    for (size_t i = 0; sameIndices && i < indices.size(); ++i)
        if (indices[i] != indices2[i]) sameIndices = false;
    check(sameTextures, "two builds -> identical textures array");
    check(sameIndices, "two builds -> identical per-material indices");

    std::printf("bindless table: %u materials -> %u unique textures (1 array bind), %zu indices\n",
                (uint32_t)materials.size(), table.size(), indices.size());

    if (g_fail == 0) { std::printf("bindless_test: all checks passed\n"); return 0; }
    std::printf("bindless_test: %d FAILURES\n", g_fail);
    return 1;
}
