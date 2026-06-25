#pragma once
// engine/asset/asset_compiler.h — namespace hf::asset (Slice ASSET-S1, issue #16 beachhead).
//
// The beachhead of the DETERMINISTIC CONTENT-ADDRESSED ASSET PIPELINE: the layer above the existing
// loaders (obj_loader.h / gltf_loader) that turns raw assets into byte-identical compiled artifacts via a
// content-hash cache + incremental rebuild. S1 establishes the cache KEY — a stable CacheKey that is a pure
// function of (raw content bytes, integer compile params, asset kind), identical across MSVC / Windows-clang
// / Mac-clang, independent of any pointer, clock, or file mtime.
//
// THE LOAD-BEARING INVARIANT: a cache key is CONTENT + PARAMS ONLY — last_write_time / mtime NEVER enters a
// key. mtime is the TRIGGER for a recompile (slice S6's watch), it is NEVER the IDENTITY of an artifact. If
// two machines (or two builds) see the same bytes + the same params, they MUST derive the same key,
// regardless of when the file was touched. This is the invariant the whole flagship rests on.
//
// SELF-CONTAINED ON PURPOSE: this header includes ONLY <cstdint>/<cstddef>/<vector> + net/session.h (for
// hf::net::DigestBytes, the engine-wide FNV-1a-64). NO replay.h, NO <cmath>/float/clock/mtime/RNG/<random>/
// <unordered_*>/<map>/<functional>/std::hash/<algorithm>/<string> — pure-CPU integer; `scale` is Q16.16
// int32 (65536 == 1.0). It MUST compile standalone:
//   clang++ -std=c++20 -I engine -I tests tests/asset_compiler_test.cpp
// This is ONE growing header — every later slice (S2–S5) APPENDS a section BELOW S1; do NOT modify S1's
// symbols once pinned.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "net/session.h"   // hf::net::DigestBytes (FNV-1a-64)

namespace hf::asset {

// --- Inline little-endian appenders (self-contained — mirror replay.h's LE appenders, local to here) ------
// Byte-by-byte LE, NEVER a struct memcpy — this is the cross-platform-stable serialization. S2+ will reuse
// these and add PutBytes/GetU32/GetU64 as needed; S1 needs only PutU32/PutU64.
inline void PutU32(std::vector<uint8_t>& b, uint32_t v) {            // 4 bytes LE
    b.push_back((uint8_t)(v & 0xFF)); b.push_back((uint8_t)((v >> 8) & 0xFF));
    b.push_back((uint8_t)((v >> 16) & 0xFF)); b.push_back((uint8_t)((v >> 24) & 0xFF));
}
inline void PutU64(std::vector<uint8_t>& b, uint64_t v) {            // 8 bytes LE
    for (int i = 0; i < 8; ++i) b.push_back((uint8_t)((v >> (8 * i)) & 0xFF));
}

// --- Types (all in hf::asset) ----------------------------------------------------------------------------
using CacheKey = uint64_t;                       // an FNV-1a-64 content-address

enum class AssetKind : uint32_t {                // S2+ append kinds; do NOT renumber
    Mesh = 0, Texture = 1, Audio = 2, Scene = 3,
};

// Integer/enum/Q16.16 compile options ONLY — NO float (float in the key would break cross-compiler
// determinism). `scale` is Q16.16 fixed-point (int32, the engine's fpx convention: 65536 == 1.0).
struct CompileParams {
    uint32_t recomputeNormals = 0;       // bool as 0/1
    int32_t  scale            = 65536;   // Q16.16 (kOne) — uniform import scale, integer-exact
    uint32_t tangentMode      = 0;       // enum: 0 none / 1 mikktspace-style / ... (S2 may use)
    uint32_t flags            = 0;       // reserved bitfield (future options; serialized so it's load-bearing)
};

struct AssetId {                         // the content-address of one asset
    uint32_t kind        = 0;            // an AssetKind value
    CacheKey contentHash = 0;            // HashRawAsset over the raw bytes
    CacheKey paramHash   = 0;            // HashParams over the compile options
};

// --- Functions (pure, deterministic) ---------------------------------------------------------------------
// The content hash — a pure FNV over the raw input (empty input -> the FNV offset basis, a fixed value).
inline CacheKey HashRawAsset(const void* bytes, std::size_t n) {
    return net::DigestBytes(bytes, n);
}

// Hand-LE-serialize EVERY field in a FIXED order, then DigestBytes. Serialize `scale` as (uint32_t)scale —
// the int32 bit pattern, LE-stable. NEVER memcpy the struct (padding).
inline CacheKey HashParams(const CompileParams& p) {
    std::vector<uint8_t> buf;
    PutU32(buf, p.recomputeNormals);
    PutU32(buf, (uint32_t)p.scale);
    PutU32(buf, p.tangentMode);
    PutU32(buf, p.flags);
    return net::DigestBytes(buf.data(), buf.size());
}

// Hand-LE-serialize the triple in a FIXED order then DigestBytes — the final content-address:
// key = f(kind, content, params), order-fixed.
inline CacheKey MakeKey(uint32_t kind, CacheKey contentHash, CacheKey paramHash) {
    std::vector<uint8_t> buf;
    PutU32(buf, kind);
    PutU64(buf, contentHash);
    PutU64(buf, paramHash);
    return net::DigestBytes(buf.data(), buf.size());
}

// The convenience composer.
inline AssetId MakeAssetId(uint32_t kind, const void* bytes, std::size_t n, const CompileParams& p) {
    return AssetId{ kind, HashRawAsset(bytes, n), HashParams(p) };
}

// --- Fixture (deterministic, FIXED forever — the golden pins its key) ------------------------------------
inline const char* ShowcaseRawBytes() {
    // A fixed small raw blob (OBJ-ish text). Keep FIXED forever.
    return "hf-asset-fixture-v1\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
}
inline std::size_t ShowcaseRawLen() {
    const char* s = ShowcaseRawBytes();
    std::size_t n = 0;
    while (s[n] != '\0') ++n;        // no <cstring>; count bytes (excludes the NUL terminator)
    return n;
}
inline CompileParams ShowcaseParams() {
    // A fixed non-default params. Keep FIXED.
    CompileParams p;
    p.recomputeNormals = 1;
    p.scale            = 2 * 65536;  // 2.0 in Q16.16
    p.tangentMode      = 1;
    p.flags            = 0;
    return p;
}

}  // namespace hf::asset
