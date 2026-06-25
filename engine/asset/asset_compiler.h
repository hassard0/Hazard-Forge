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

#include "net/session.h"        // hf::net::DigestBytes (FNV-1a-64)
#include "asset/obj_loader.h"   // hf::asset::ParseObj / ObjMesh / ObjVertex (S2 — pure header-only loader)

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

// =========================================================================================================
// Slice ASSET-S2 — Deterministic compiled-artifact format (issue #16). APPEND-ONLY below S1; S1's symbols
// (incl. the key 0x7fb6a48b4b99f1b7) are UNCHANGED. Turns a raw OBJ -> a canonical, versioned, hand-LE
// binary blob whose geometry is quantized to Q16.16 INTEGERS (NOT raw float bits) so the artifact is
// bit-identical across MSVC / Windows-clang / Mac-clang. The golden is the artifact's pinned DigestBytes.
// =========================================================================================================

// --- The Q16.16 quantizer + integer multiply (inline, self-contained) ------------------------------------
// Exact float->Q16.16: 65536 is a power of two, so f*65536.0f shifts the exponent with NO rounding (exact
// for |f| < 32768), and the truncation is identical on every compiler. The ONE float op in the compile step;
// the artifact it produces is pure integer. (No <cmath>: a bare multiply + cast.)
inline int32_t FxQuantize(float f) { return (int32_t)(f * 65536.0f); }
// Q16.16 fixed-point multiply (integer): (a*b) >> 16 via int64 intermediate (the fpx convention).
inline int32_t FxMul(int32_t a, int32_t b) { return (int32_t)(((int64_t)a * (int64_t)b) >> 16); }

// --- Additional inline LE appenders/readers (self-contained; mirror replay.h, still NOT including it) -----
inline void PutBytes(std::vector<uint8_t>& b, const uint8_t* p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) b.push_back(p[i]);
}
inline uint32_t GetU32(const std::vector<uint8_t>& b, std::size_t& off) {   // 4 bytes LE
    uint32_t v = (uint32_t)b[off] | ((uint32_t)b[off + 1] << 8) |
                 ((uint32_t)b[off + 2] << 16) | ((uint32_t)b[off + 3] << 24);
    off += 4;
    return v;
}
inline uint64_t GetU64(const std::vector<uint8_t>& b, std::size_t& off) {   // 8 bytes LE
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t)b[off + (std::size_t)i] << (8 * i);
    off += 8;
    return v;
}

// --- The compiled-mesh artifact (canonical, versioned, hand-LE) ------------------------------------------
constexpr uint32_t kCompiledMeshMagic   = 0x484D4D31;  // 'HMM1' (Hazard Mesh, v-tagged by `version`)
constexpr uint32_t kCompiledMeshVersion = 1;           // bump if the Q16.16 grid / layout changes

struct CompiledMesh {                     // the DECODED view (S2 round-trip target)
    uint32_t magic = 0, version = 0;
    uint32_t vertexCount = 0, indexCount = 0;
    CompileParams params;                 // the options baked in (echoed in the header so they're load-bearing)
    std::vector<int32_t>  verts;          // vertexCount * 8 Q16.16 ints: pos.xyz, uv.xy, normal.xyz (fixed stride)
    std::vector<uint32_t> indices;        // indexCount triangle-list indices
};

// --- CompileObj — raw OBJ text -> the canonical Q16.16 blob ----------------------------------------------
// ParseObj -> hand-LE blob in FIXED field order. v1 applies `scale` geometrically (an exact integer Q16.16
// multiply on each position component); `recomputeNormals`/`tangentMode`/`flags` are recorded in the header
// (load-bearing on the digest) but their geometry effect is a documented v1-deferred refinement.
inline std::vector<uint8_t> CompileObj(const char* text, std::size_t n, const CompileParams& p) {
    ObjMesh m = ParseObj(text, n);
    std::vector<uint8_t> b;
    PutU32(b, kCompiledMeshMagic);
    PutU32(b, kCompiledMeshVersion);
    PutU32(b, (uint32_t)m.vertices.size());   // vertexCount
    PutU32(b, (uint32_t)m.indices.size());    // indexCount
    // params header (any param change re-digests the artifact)
    PutU32(b, p.recomputeNormals);
    PutU32(b, (uint32_t)p.scale);
    PutU32(b, p.tangentMode);
    PutU32(b, p.flags);
    // per vertex: 8 Q16.16 components in fixed order (pos.xyz scaled, uv.xy, normal.xyz)
    for (const ObjVertex& v : m.vertices) {
        for (int i = 0; i < 3; ++i) PutU32(b, (uint32_t)FxMul(FxQuantize(v.pos[i]), p.scale));
        for (int i = 0; i < 2; ++i) PutU32(b, (uint32_t)FxQuantize(v.uv[i]));
        for (int i = 0; i < 3; ++i) PutU32(b, (uint32_t)FxQuantize(v.normal[i]));
    }
    for (uint32_t idx : m.indices) PutU32(b, idx);
    return b;
}

// --- DecodeCompiledMesh — the round-trip (false on truncation / bad magic) --------------------------------
// Reads the fixed-order fields back hand-LE (NEVER a struct memcpy), validates magic + the exact byte length
// (header + 8*vertexCount*4 + indexCount*4).
inline bool DecodeCompiledMesh(const std::vector<uint8_t>& blob, CompiledMesh& out) {
    const std::size_t kHeaderBytes = 8 * 4;   // magic,version,vertexCount,indexCount + 4 params words
    if (blob.size() < kHeaderBytes) return false;
    std::size_t off = 0;
    uint32_t magic       = GetU32(blob, off);
    uint32_t version     = GetU32(blob, off);
    uint32_t vertexCount = GetU32(blob, off);
    uint32_t indexCount  = GetU32(blob, off);
    if (magic != kCompiledMeshMagic) return false;
    const std::size_t expect = kHeaderBytes +
                               (std::size_t)vertexCount * 8 * 4 +
                               (std::size_t)indexCount * 4;
    if (blob.size() != expect) return false;
    CompileParams params;
    params.recomputeNormals = GetU32(blob, off);
    params.scale            = (int32_t)GetU32(blob, off);
    params.tangentMode      = GetU32(blob, off);
    params.flags            = GetU32(blob, off);
    out.magic = magic; out.version = version;
    out.vertexCount = vertexCount; out.indexCount = indexCount;
    out.params = params;
    out.verts.clear();   out.verts.reserve((std::size_t)vertexCount * 8);
    out.indices.clear(); out.indices.reserve((std::size_t)indexCount);
    for (uint32_t i = 0; i < vertexCount * 8; ++i) out.verts.push_back((int32_t)GetU32(blob, off));
    for (uint32_t i = 0; i < indexCount; ++i)      out.indices.push_back(GetU32(blob, off));
    return true;
}

// The artifact's content-address — a pure FNV over the canonical Q16.16 blob (pure integer, mtime-free).
inline CacheKey DigestArtifact(const std::vector<uint8_t>& blob) {
    return net::DigestBytes(blob.data(), blob.size());
}

// =========================================================================================================
// Slice ASSET-S3 — The content-addressed cache (issue #16). APPEND-ONLY below S2; S1's key
// 0x7fb6a48b4b99f1b7 and S2's artifact digest 0xf7ee13c169dc0464 are UNCHANGED. A key->artifact store
// (sorted-unique vector + hand binary search — the chunk_diff.h ChunkDiffStore mold), GetOrCompile
// (compile-on-miss), hand-LE serialize/deserialize round-trip, and a store digest INDEPENDENT of insertion
// order (the content-addressed property: two machines that compiled the same assets in any order have the
// byte-identical cache). Pure integer; only GetOrCompile's miss path reaches S2's CompileObj.
//
// LE helpers PutBytes/GetU32/GetU64 were already added by S2 above — REUSED here, not redefined.
// =========================================================================================================

// --- The cache (sorted-unique vector, the ChunkDiffStore mold) -------------------------------------------
struct CacheEntry { CacheKey key = 0; std::vector<uint8_t> blob; };

struct AssetCache {
    std::vector<CacheEntry> entries;   // SORTED-UNIQUE by key (binary-search insert/lookup) — order-stable
};

// Binary-search for `key`; return its blob or nullptr. Hand-written (mirrors chunk_diff.h's Find/
// InsertSortedUnique loop) — NO std::lower_bound, NO <algorithm>. A pure const lookup, no mutation.
inline const std::vector<uint8_t>* Lookup(const AssetCache& c, CacheKey key) {
    std::size_t lo = 0, hi = c.entries.size();
    while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (c.entries[mid].key < key) lo = mid + 1; else hi = mid; }
    return (lo < c.entries.size() && c.entries[lo].key == key) ? &c.entries[lo].blob : nullptr;
}

// Insert (or overwrite) the blob for `key`, keeping `entries` sorted-unique by key (the InsertSortedUnique
// binary-search pattern from chunk_diff.h:67). A re-insert of an existing key overwrites its blob.
inline void Insert(AssetCache& c, CacheKey key, const std::vector<uint8_t>& blob) {
    std::size_t lo = 0, hi = c.entries.size();
    while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (c.entries[mid].key < key) lo = mid + 1; else hi = mid; }
    if (lo < c.entries.size() && c.entries[lo].key == key) { c.entries[lo].blob = blob; return; }  // overwrite
    c.entries.insert(c.entries.begin() + (std::ptrdiff_t)lo, CacheEntry{ key, blob });
}

// --- GetOrCompile — the cache's reason to exist ----------------------------------------------------------
struct CompileResult { std::vector<uint8_t> blob; bool wasHit = false; };

// key = MakeKey(Mesh, HashRawAsset(bytes,n), HashParams(p)); on a HIT return the stored blob + wasHit=true;
// on a MISS, CompileObj -> Insert -> return blob + wasHit=false. The hit blob is byte-identical to a cold
// CompileObj of the same inputs. `wasHit` is a RETURN flag ONLY — NEVER serialized, NEVER in the blob bytes.
inline CompileResult GetOrCompile(AssetCache& c, const void* bytes, std::size_t n, const CompileParams& p) {
    const CacheKey key = MakeKey((uint32_t)AssetKind::Mesh, HashRawAsset(bytes, n), HashParams(p));
    if (const std::vector<uint8_t>* hit = Lookup(c, key)) return CompileResult{ *hit, true };
    std::vector<uint8_t> blob = CompileObj((const char*)bytes, n, p);
    Insert(c, key, blob);
    return CompileResult{ blob, false };
}

// --- Serialize / Deserialize / Digest (hand-LE, order-stable) --------------------------------------------
// Layout: PutU32(entryCount), then per entry IN SORTED-KEY ORDER: PutU64(key), PutU32(blobLen), PutBytes(blob).
// (entries is kept sorted by key, so the byte stream is identical regardless of insertion order — the
// content-addressed property.) wasHit / mtime NEVER appear.
inline std::vector<uint8_t> SerializeCache(const AssetCache& c) {
    std::vector<uint8_t> b;
    PutU32(b, (uint32_t)c.entries.size());
    for (const CacheEntry& e : c.entries) {
        PutU64(b, e.key);
        PutU32(b, (uint32_t)e.blob.size());
        PutBytes(b, e.blob.data(), e.blob.size());
    }
    return b;
}

// Reads the fixed-order stream back hand-LE (NEVER a struct memcpy). Returns false on ANY truncation
// (a short header, a missing key/len, or a blob that runs past the buffer). Entries are read in stored
// (sorted-key) order, so `out.entries` stays sorted-unique.
inline bool DeserializeCache(const std::vector<uint8_t>& bytes, AssetCache& out) {
    out.entries.clear();
    if (bytes.size() < 4) return false;
    std::size_t off = 0;
    const uint32_t count = GetU32(bytes, off);
    out.entries.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (off + 8 + 4 > bytes.size()) return false;          // key + blobLen header
        const CacheKey key  = GetU64(bytes, off);
        const uint32_t blen = GetU32(bytes, off);
        if (off + (std::size_t)blen > bytes.size()) return false;   // blob runs past the buffer
        CacheEntry e;
        e.key = key;
        e.blob.assign(bytes.begin() + (std::ptrdiff_t)off, bytes.begin() + (std::ptrdiff_t)(off + blen));
        off += blen;
        out.entries.push_back(std::move(e));
    }
    return true;
}

inline CacheKey DigestCache(const AssetCache& c) {
    std::vector<uint8_t> b = SerializeCache(c);
    return net::DigestBytes(b.data(), b.size());
}

// --- Fixtures (FIXED forever — the golden pins a cache built from all three) ------------------------------
// Two MORE distinct tiny OBJ texts beyond S1's ShowcaseRawBytes() (a quad and a different triangle), each
// with its own byte length. Keep FIXED.
inline const char* ShowcaseRawBytesB() {
    // A unit quad (4 verts, 2 tris) — distinct from S1's fixture.
    return "hf-asset-fixture-B\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n";
}
inline std::size_t ShowcaseRawLenB() {
    const char* s = ShowcaseRawBytesB();
    std::size_t n = 0;
    while (s[n] != '\0') ++n;
    return n;
}
inline const char* ShowcaseRawBytesC() {
    // A different triangle (a shifted apex) — distinct from S1 + B.
    return "hf-asset-fixture-C\nv 0 0 0\nv 2 0 0\nv 1 3 0\nf 1 2 3\n";
}
inline std::size_t ShowcaseRawLenC() {
    const char* s = ShowcaseRawBytesC();
    std::size_t n = 0;
    while (s[n] != '\0') ++n;
    return n;
}

// Build a fresh cache by GetOrCompile-ing all three fixtures with ShowcaseParams(). FIXED — the golden pins
// its DigestCache (order-independent, so insertion order here does not change the digest).
inline AssetCache MakeShowcaseCache() {
    AssetCache c;
    GetOrCompile(c, ShowcaseRawBytes(),  ShowcaseRawLen(),  ShowcaseParams());
    GetOrCompile(c, ShowcaseRawBytesB(), ShowcaseRawLenB(), ShowcaseParams());
    GetOrCompile(c, ShowcaseRawBytesC(), ShowcaseRawLenC(), ShowcaseParams());
    return c;
}

}  // namespace hf::asset
