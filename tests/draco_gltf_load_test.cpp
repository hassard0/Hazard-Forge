// Acceptance test for issue #36 (DRACO-DR5): a REAL KHR_draco_mesh_compression glTF that was silently
// DROPPED before now LOADS correctly through the glTF loader's device-free CPU seam.
//
// It parses the committed assets/models/BoxDraco/Box.gltf (cgltf_parse_file + cgltf_load_buffers, the
// SAME path the engine loader uses), finds the Draco primitive, and runs BuildPrimitiveCPU — the
// device-free geometry extraction that wires in the self-contained hf::asset::draco decoder. It asserts
// the decoded Box is a valid cube: 12 faces / 36 indices, 8 distinct corner positions at {-0.5,+0.5}^3,
// and pins a deterministic digest over the decoded geometry (the cross-platform anchor).
//
// cgltf's IMPLEMENTATION lives in engine/asset/gltf_loader.cpp (compiled into hf_core, which this test
// links) — so we include ONLY the cgltf declarations here (NOT CGLTF_IMPLEMENTATION) to avoid duplicate
// symbols. Pure CPU (hf_core), no RHI/device, ASan-eligible.

#include "asset/gltf_loader.h"
#include "asset/draco_decode.h"
#include "scene/vertex.h"
#include "cgltf/cgltf.h"
#include "net/session.h"  // hf::net::DigestBytes — the FNV-1a-64 golden currency

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"

using namespace hf::asset;

#ifndef HF_BOX_DRACO_GLTF
#error "HF_BOX_DRACO_GLTF (path to assets/models/BoxDraco/Box.gltf) must be defined by the build"
#endif

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS: %s\n", what); }
}

// Find the first primitive carrying a Draco blob.
static const cgltf_primitive* FindDracoPrim(const cgltf_data* data) {
    for (cgltf_size m = 0; m < data->meshes_count; ++m)
        for (cgltf_size p = 0; p < data->meshes[m].primitives_count; ++p)
            if (data->meshes[m].primitives[p].has_draco_mesh_compression)
                return &data->meshes[m].primitives[p];
    return nullptr;
}

int main() {
    HF_TEST_MAIN_INIT();

    const char* path = HF_BOX_DRACO_GLTF;

    // --- Parse + load the buffers (the real loader path; cgltf reads Box.bin via the URI). ---
    cgltf_options options{};
    cgltf_data* data = nullptr;
    cgltf_result res = cgltf_parse_file(&options, path, &data);
    check(res == cgltf_result_success && data != nullptr, "cgltf_parse_file(Box.gltf) succeeds");
    if (res != cgltf_result_success || !data) {
        std::printf("draco_gltf_load_test: %d FAILED (could not parse %s)\n", g_fail + 1, path);
        return 1;
    }
    res = cgltf_load_buffers(&options, data, path);
    check(res == cgltf_result_success, "cgltf_load_buffers(Box.bin) succeeds");

    // --- The primitive must be Draco-flagged (the whole point of the asset). ---
    const cgltf_primitive* prim = FindDracoPrim(data);
    check(prim != nullptr, "Box.gltf has a KHR_draco_mesh_compression primitive");

    if (res == cgltf_result_success && prim) {
        // --- Decode the Draco blob directly (the substrate the loader calls). ---
        const cgltf_buffer_view* bv = prim->draco_mesh_compression.buffer_view;
        const uint8_t* bytes = bv ? cgltf_buffer_view_data(bv) : nullptr;
        check(bytes != nullptr && bv->size > 0, "Draco bufferView resolves to non-empty bytes");

        draco::DecodedMesh dm = draco::DecodeDracoMesh(bytes, static_cast<std::size_t>(bv->size));
        check(dm.ok, "DecodeDracoMesh: the Box Draco blob decodes (ok)");
        check(dm.num_faces == 12, "DecodeDracoMesh: 12 faces (a cube)");
        check(dm.indices.size() == 36, "DecodeDracoMesh: 36 indices (12 tris)");
        check(dm.num_points == 8, "DecodeDracoMesh: 8 corner points");

        // --- The 8 distinct positions are the cube corners at {-0.5,+0.5}^3 (within quantization eps). ---
        bool allCorners = (dm.positions.size() == dm.num_points * 3u);
        const float kHalf = 0.5f;
        const float kEps  = 1e-2f;   // 11-bit quantization of a ~1.0 range -> ~5e-4 step; 1e-2 is ample
        for (uint32_t i = 0; i < dm.num_points && allCorners; ++i) {
            for (int k = 0; k < 3; ++k) {
                float c = dm.positions[i * 3 + k];
                if (std::fabs(std::fabs(c) - kHalf) > kEps) { allCorners = false; break; }
            }
        }
        check(allCorners, "DecodeDracoMesh: every position is a cube corner at +/-0.5");

        // Every index addresses a valid point.
        bool idxOk = true;
        for (uint32_t idx : dm.indices)
            if (idx >= dm.num_points) { idxOk = false; break; }
        check(idxOk, "DecodeDracoMesh: all indices in range");

        // --- The full loader seam: BuildPrimitiveCPU (device-free) yields the SAME cube. ---
        CpuPrimitive cpu = BuildPrimitiveCPU(*prim, path, /*recentre=*/false);
        check(cpu.verts.size() == 8, "BuildPrimitiveCPU: 8 vertices (the cube)");
        check(cpu.indices.size() == 36, "BuildPrimitiveCPU: 36 indices (the cube)");
        // The 8 BuildPrimitiveCPU vertices are the same +/-0.5 corners.
        bool cpuCorners = (cpu.verts.size() == 8);
        for (const hf::scene::Vertex& v : cpu.verts) {
            for (int k = 0; k < 3; ++k)
                if (std::fabs(std::fabs(v.pos[k]) - kHalf) > kEps) { cpuCorners = false; break; }
        }
        check(cpuCorners, "BuildPrimitiveCPU: every vertex is a cube corner at +/-0.5");

        // --- Pinned digest over the decoded geometry (deterministic cross-platform anchor). ---
        // Quantize each position to the nearest 11-bit step before hashing so the digest is exact
        // across compilers (the dequantized float is bit-identical, but rounding to an int makes the
        // golden robust to any future toolchain float-format drift). Encode faces as raw u32 indices.
        std::vector<uint8_t> blob;
        auto putU32 = [&](uint32_t x) {
            blob.push_back(static_cast<uint8_t>(x & 0xFF));
            blob.push_back(static_cast<uint8_t>((x >> 8) & 0xFF));
            blob.push_back(static_cast<uint8_t>((x >> 16) & 0xFF));
            blob.push_back(static_cast<uint8_t>((x >> 24) & 0xFF));
        };
        putU32(dm.num_faces);
        putU32(dm.num_points);
        for (float c : dm.positions) {
            // round-to-nearest signed int of (c * 1000): -500 / +500 for the corners.
            int32_t q = static_cast<int32_t>(std::lround(c * 1000.0f));
            putU32(static_cast<uint32_t>(q));
        }
        for (uint32_t idx : dm.indices) putU32(idx);

        const uint64_t digest = hf::net::DigestBytes(blob.data(), blob.size());
        std::printf("draco_gltf_load_test: decoded-Box digest = 0x%016llx\n",
                    static_cast<unsigned long long>(digest));
        // PINNED golden: the Box Draco -> cube decode is a codec (same bytes in -> same bytes out).
        const uint64_t kBoxDigest = 0xc8c260f93eaffb7eULL;
        check(digest == kBoxDigest, "DecodeDracoMesh: decoded-Box digest matches the pinned golden");
    }

    cgltf_free(data);

    if (g_fail == 0) std::printf("draco_gltf_load_test: ALL PASS\n");
    else             std::printf("draco_gltf_load_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
