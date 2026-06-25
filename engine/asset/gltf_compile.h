#pragma once
// engine/asset/gltf_compile.h — namespace hf::asset (issue #16, glTF FORMAT-COVERAGE gap closer).
//
// Closes the deterministic content-addressed asset pipeline's FORMAT-COVERAGE gap: the pipeline
// (asset_compiler.h) compiled ONLY Wavefront OBJ. This header adds CompileGltf — parse a glTF / glb
// from MEMORY via cgltf, extract the first mesh's first primitive's POSITION vertices + indices, and
// emit the EXACT SAME canonical Q16.16 CompiledMesh blob CompileObj produces (same magic / version /
// vertexCount / indexCount / params header / per-vertex 8 Q16.16 components / indices). So a glTF and
// an OBJ of equivalent geometry compile to the byte-identical artifact shape, and the moat invariant
// holds: same bytes in -> byte-identical blob out, identical MSVC / Windows-clang / Mac-clang.
//
// WHY A SEPARATE HEADER (the BOLD call): asset_compiler.h is included WIDELY and is intentionally
// self-contained (no cgltf, no float-render). cgltf is a heavyweight ~7k-line single-file C library
// whose CGLTF_IMPLEMENTATION must live in EXACTLY ONE translation unit (gltf_loader.cpp owns it in
// hf_core / hf_engine). Pulling cgltf into asset_compiler.h would force it on every includer. So the
// glTF path is quarantined here: this header includes asset_compiler.h (for the blob encoders +
// CompileParams) + cgltf's DECLARATIONS only. The cgltf IMPLEMENTATION symbols come from the linked
// library (hf_core's gltf_loader.cpp) at link time — so a TU that includes THIS header MUST NOT define
// CGLTF_IMPLEMENTATION (it already exists in the lib). A pure standalone clang compile (no hf_core)
// defines CGLTF_IMPLEMENTATION itself, in its own TU, BEFORE including this header.
//
// DETERMINISM: the ONLY float op is the same FxQuantize(f) = (int32_t)(f * 65536.0f) the OBJ path
// uses (an exact power-of-two scale + identical truncation on every compiler). cgltf_accessor_read_float
// returns the authored buffer floats verbatim (no matrix/normalization math for the position read), so
// the quantized integers are bit-stable cross-compiler. cgltf's parse is a pure byte->struct decode with
// no clock / RNG / pointer-identity leaking into the geometry. The blob is pure integer once emitted.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "asset/asset_compiler.h"   // hf::asset — CompileParams, FxQuantize/FxMul, PutU32, the blob format
#include "cgltf/cgltf.h"            // DECLARATIONS only — implementation comes from the linked lib (or a
                                    // standalone TU that defines CGLTF_IMPLEMENTATION before this include)

namespace hf::asset {

// --- The raw geometry CompileGltf extracts (mirrors the ObjMesh fields CompileObj consumes) --------------
// Only POSITION + indices are read from the glTF (parity with what the OBJ blob carries that is geometry-
// load-bearing under scale); uv / normal are emitted as zero Q16.16 — IDENTICAL to how CompileObj treats an
// OBJ corner that omits vt / vn (ObjVertex defaults uv/normal to 0). v1-deferred: TEXCOORD_0 / NORMAL read.
struct GltfGeom {
    std::vector<float>    posX, posY, posZ;   // per-vertex POSITION (authored model-space floats)
    std::vector<uint32_t> indices;            // triangle-list indices (or implicit 0..n-1 if non-indexed)
};

// Parse a glTF / glb blob from memory and pull the first mesh's first primitive POSITION + indices. Returns
// false on a parse / buffer-load failure or an empty / position-less primitive (so the caller emits an empty
// — but still well-formed — CompiledMesh, never garbage). Pure of the parse: same bytes -> same GltfGeom.
inline bool ParseGltfGeom(const uint8_t* bytes, std::size_t n, GltfGeom& out) {
    out.posX.clear(); out.posY.clear(); out.posZ.clear(); out.indices.clear();
    if (!bytes || n == 0) return false;

    cgltf_options options{};                 // type = invalid -> auto-detect (glTF JSON vs glb binary)
    cgltf_data* data = nullptr;
    if (cgltf_parse(&options, bytes, (cgltf_size)n, &data) != cgltf_result_success) return false;

    // Resolve buffers. For an EMBEDDED base64 data: URI (the deterministic self-contained case) the gltf
    // path may be null. For a glb the buffer is the binary chunk. External file URIs are NOT resolvable
    // from a pure memory blob (no path) — those primitives read zero, a documented v1 limitation.
    bool ok = false;
    if (cgltf_load_buffers(&options, data, nullptr) == cgltf_result_success &&
        data->meshes_count > 0 && data->meshes[0].primitives_count > 0) {
        const cgltf_primitive& prim = data->meshes[0].primitives[0];
        const cgltf_accessor* posAcc = nullptr;
        for (cgltf_size i = 0; i < prim.attributes_count; ++i) {
            if (prim.attributes[i].type == cgltf_attribute_type_position) { posAcc = prim.attributes[i].data; break; }
        }
        if (posAcc && posAcc->count > 0) {
            const cgltf_size vc = posAcc->count;
            out.posX.reserve(vc); out.posY.reserve(vc); out.posZ.reserve(vc);
            for (cgltf_size i = 0; i < vc; ++i) {
                float p[3] = {0, 0, 0};
                cgltf_accessor_read_float(posAcc, i, p, 3);   // authored floats, verbatim
                out.posX.push_back(p[0]); out.posY.push_back(p[1]); out.posZ.push_back(p[2]);
            }
            if (prim.indices && prim.indices->count > 0) {
                const cgltf_size ic = prim.indices->count;
                out.indices.reserve(ic);
                for (cgltf_size i = 0; i < ic; ++i)
                    out.indices.push_back((uint32_t)cgltf_accessor_read_index(prim.indices, i));
            } else {
                // Non-indexed primitive -> the implicit identity index list 0..vc-1 (matches the gltf_loader
                // convention). Keeps the blob's indexCount field meaningful + geometry round-trippable.
                out.indices.reserve(vc);
                for (cgltf_size i = 0; i < vc; ++i) out.indices.push_back((uint32_t)i);
            }
            ok = true;
        }
    }
    cgltf_free(data);
    return ok;
}

// --- CompileGltf — raw glTF/glb bytes -> the canonical Q16.16 blob (the SAME format CompileObj emits) -----
// Byte-for-byte the CompileObj encoding: magic, version, vertexCount, indexCount, the 4-word params header,
// then per vertex 8 Q16.16 components (pos.xyz scaled by p.scale via FxMul(FxQuantize, scale), uv.xy = 0,
// normal.xyz = 0 — exactly CompileObj's treatment of a vt/vn-less OBJ corner), then the indices. A glTF and
// an OBJ that decode to the SAME positions + indices therefore compile to the IDENTICAL blob bytes.
// On a parse failure the result is a well-formed EMPTY mesh (vertexCount = indexCount = 0) — never garbage.
inline std::vector<uint8_t> CompileGltf(const uint8_t* bytes, std::size_t n, const CompileParams& p) {
    GltfGeom g;
    ParseGltfGeom(bytes, n, g);   // on failure g stays empty -> an empty-but-valid CompiledMesh

    std::vector<uint8_t> b;
    PutU32(b, kCompiledMeshMagic);
    PutU32(b, kCompiledMeshVersion);
    PutU32(b, (uint32_t)g.posX.size());     // vertexCount
    PutU32(b, (uint32_t)g.indices.size());  // indexCount
    // params header (any param change re-digests the artifact) — identical layout to CompileObj.
    PutU32(b, p.recomputeNormals);
    PutU32(b, (uint32_t)p.scale);
    PutU32(b, p.tangentMode);
    PutU32(b, p.flags);
    // per vertex: 8 Q16.16 components in fixed order (pos.xyz scaled, uv.xy = 0, normal.xyz = 0).
    for (std::size_t i = 0; i < g.posX.size(); ++i) {
        PutU32(b, (uint32_t)FxMul(FxQuantize(g.posX[i]), p.scale));
        PutU32(b, (uint32_t)FxMul(FxQuantize(g.posY[i]), p.scale));
        PutU32(b, (uint32_t)FxMul(FxQuantize(g.posZ[i]), p.scale));
        for (int k = 0; k < 2; ++k) PutU32(b, (uint32_t)FxQuantize(0.0f));   // uv.xy (absent -> 0)
        for (int k = 0; k < 3; ++k) PutU32(b, (uint32_t)FxQuantize(0.0f));   // normal.xyz (absent -> 0)
    }
    for (uint32_t idx : g.indices) PutU32(b, idx);
    return b;
}

// --- GetOrCompileGltf — the glTF analog of S3's GetOrCompile (cache-on-miss via CompileGltf) --------------
// Wires the glTF path into the SAME content-addressed cache the OBJ path uses. The key is the SAME triple
// MakeKey(Mesh, HashRawAsset(bytes,n), HashParams(p)) — so a glTF and an OBJ that happen to share raw bytes
// would collide (they never do: distinct file formats), and the cache stays content-addressed. On a MISS,
// CompileGltf -> Insert; on a HIT, the stored blob (byte-identical to a cold CompileGltf). `wasHit` is a
// RETURN flag only, NEVER serialized — exactly the S3 GetOrCompile contract.
inline CompileResult GetOrCompileGltf(AssetCache& c, const uint8_t* bytes, std::size_t n, const CompileParams& p) {
    const CacheKey key = MakeKey((uint32_t)AssetKind::Mesh, HashRawAsset(bytes, n), HashParams(p));
    if (const std::vector<uint8_t>* hit = Lookup(c, key)) return CompileResult{ *hit, true };
    std::vector<uint8_t> blob = CompileGltf(bytes, n, p);
    Insert(c, key, blob);
    return CompileResult{ blob, false };
}

}  // namespace hf::asset
