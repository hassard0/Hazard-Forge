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

// =========================================================================================================
// Slice ASSET-S4 — Dependency graph + deterministic incremental rebuild (issue #16). APPEND-ONLY below S3;
// S1's key 0x7fb6a48b4b99f1b7, S2's artifact 0xf7ee13c169dc0464, and S3's cache 0x029174f13e64c9f1 are
// UNCHANGED. Adds the incremental-rebuild plan: a dependency graph (sorted-unique adjacency, the
// ChunkDiffStore discipline), a deterministic InvalidationSet (reverse reachability — changed + transitive
// dependents), a hand Kahn topological RebuildOrder (the flow.h lowest-id ascending-scan, NO std::sort /
// lower_bound / <algorithm>), and a Rebuild plan whose digest pins the recompile order.
//
// NODE IDENTITY: `NodeId` is a STABLE LOGICAL asset id (the graph node) — it PERSISTS across edits, distinct
// from S1's content-addressed AssetId (which changes when bytes change). The graph keys on NodeId; the cache
// (S3) keys on the content-address. mtime NEVER enters the graph or any digest.
// =========================================================================================================

using NodeId = uint32_t;                          // a STABLE logical asset id (persists across edits)

struct DepGraph {
    // nodes[i] = (node, its sorted-unique list of DEPENDENCIES — the nodes it needs). Sorted by node.
    // Edge semantics: `node` DEPENDS ON each id in its dependency list (so a change to a dependency must
    // recompile `node`). Reverse edges (dependents) are derived on demand in InvalidationSet.
    std::vector<std::pair<NodeId, std::vector<NodeId>>> nodes;
};

// Binary-search for `n` in g.nodes (sorted by .first); return its index or g.nodes.size() if absent.
// Hand-written (the ChunkDiffStore Find loop) — NO std::lower_bound, NO <algorithm>.
inline std::size_t FindNodeIndex(const DepGraph& g, NodeId n) {
    std::size_t lo = 0, hi = g.nodes.size();
    while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (g.nodes[mid].first < n) lo = mid + 1; else hi = mid; }
    return (lo < g.nodes.size() && g.nodes[lo].first == n) ? lo : g.nodes.size();
}

// Sorted-unique insert of node `n` (no deps yet). A no-op if `n` already present.
inline void AddNode(DepGraph& g, NodeId n) {
    std::size_t lo = 0, hi = g.nodes.size();
    while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (g.nodes[mid].first < n) lo = mid + 1; else hi = mid; }
    if (lo < g.nodes.size() && g.nodes[lo].first == n) return;   // already present
    g.nodes.insert(g.nodes.begin() + (std::ptrdiff_t)lo,
                   std::pair<NodeId, std::vector<NodeId>>{ n, std::vector<NodeId>{} });
}

// Sorted-unique insert of `to` into a dependency vector (kept sorted-unique). Hand binary search.
inline void InsertSortedUniqueDep(std::vector<NodeId>& v, NodeId to) {
    std::size_t lo = 0, hi = v.size();
    while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (v[mid] < to) lo = mid + 1; else hi = mid; }
    if (lo < v.size() && v[lo] == to) return;   // already present
    v.insert(v.begin() + (std::ptrdiff_t)lo, to);
}

// `from` DEPENDS ON `to`. Ensures BOTH nodes exist; adds `to` to `from`'s sorted-unique dependency list.
inline void AddDep(DepGraph& g, NodeId from, NodeId to) {
    AddNode(g, from);
    AddNode(g, to);
    const std::size_t fi = FindNodeIndex(g, from);   // re-find: AddNode may have shifted indices
    InsertSortedUniqueDep(g.nodes[fi].second, to);
}

// The sorted-unique DEPENDENCY list of `n` (the nodes it needs); nullptr if `n` is absent. Hand binary search.
inline const std::vector<NodeId>* Dependencies(const DepGraph& g, NodeId n) {
    const std::size_t i = FindNodeIndex(g, n);
    return (i < g.nodes.size()) ? &g.nodes[i].second : nullptr;
}

// --- InvalidationSet — who must recompile when `changed` changes (reverse reachability) -------------------
// The set = `changed` itself + every node that transitively DEPENDS ON it. A node is in the set if it equals
// `changed` OR it depends on any node already in the set. Computed by a bounded fixed-point iteration over
// nodes.size() passes (membership only grows -> cycles handled, bounded -> no hang). Returned SORTED
// ASCENDING by NodeId; empty if `changed` is not in the graph.
inline std::vector<NodeId> InvalidationSet(const DepGraph& g, NodeId changed) {
    std::vector<NodeId> out;
    if (FindNodeIndex(g, changed) == g.nodes.size()) return out;   // absent -> empty
    // `in[i]` parallels g.nodes[i]: 1 if that node is in the set. Seed `changed`.
    std::vector<uint8_t> in(g.nodes.size(), 0);
    in[FindNodeIndex(g, changed)] = 1;
    // Bounded fixed-point: at most nodes.size() passes (each pass can add at least one if it grows).
    for (std::size_t pass = 0; pass < g.nodes.size(); ++pass) {
        bool grew = false;
        for (std::size_t i = 0; i < g.nodes.size(); ++i) {
            if (in[i]) continue;
            // node i is in the set if any of its dependencies is already in the set.
            const std::vector<NodeId>& deps = g.nodes[i].second;
            for (std::size_t d = 0; d < deps.size(); ++d) {
                const std::size_t di = FindNodeIndex(g, deps[d]);
                if (di < g.nodes.size() && in[di]) { in[i] = 1; grew = true; break; }
            }
        }
        if (!grew) break;
    }
    // Emit in node order (g.nodes is sorted by .first -> ascending NodeId).
    for (std::size_t i = 0; i < g.nodes.size(); ++i) if (in[i]) out.push_back(g.nodes[i].first);
    return out;
}

// --- RebuildOrder — the deterministic topological recompile sequence (hand Kahn lowest-id scan) -----------
struct OrderResult { std::vector<NodeId> order; bool ok = true; };  // ok=false iff a cycle blocks a total order

// Kahn topological sort over the SUBGRAPH induced by `subset` (only edges between subset members count):
// repeatedly emit the LOWEST-id not-yet-emitted subset node whose in-subset dependencies are ALL already
// emitted (dependencies BEFORE dependents). The flow.h TopoOrder lowest-id ascending-scan — NO std::sort,
// NO lower_bound. If no node is emittable but unemitted subset nodes remain -> a cycle -> ok=false (no hang).
inline OrderResult RebuildOrder(const DepGraph& g, const std::vector<NodeId>& subset) {
    OrderResult r;
    // `inSubset[i]` / `emitted[i]` parallel a private sorted-unique copy of `subset` (so binary-membership
    // works and emission is ascending-deterministic).
    std::vector<NodeId> sub;          // sorted-unique copy of subset
    for (std::size_t i = 0; i < subset.size(); ++i) InsertSortedUniqueDep(sub, subset[i]);
    const std::size_t m = sub.size();
    std::vector<uint8_t> emitted(m, 0);
    // membership test in `sub` (hand binary search).
    auto subIndex = [&sub, m](NodeId n) -> std::size_t {
        std::size_t lo = 0, hi = m;
        while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (sub[mid] < n) lo = mid + 1; else hi = mid; }
        return (lo < m && sub[lo] == n) ? lo : m;
    };
    std::size_t emittedCount = 0;
    while (emittedCount < m) {
        // find the LOWEST-id unemitted subset node whose in-subset deps are all emitted (sub is ascending).
        std::size_t pick = m;
        for (std::size_t i = 0; i < m; ++i) {
            if (emitted[i]) continue;
            const std::vector<NodeId>* deps = Dependencies(g, sub[i]);
            bool ready = true;
            if (deps) {
                for (std::size_t d = 0; d < deps->size(); ++d) {
                    const std::size_t si = subIndex((*deps)[d]);
                    if (si != m && !emitted[si]) { ready = false; break; }   // an in-subset dep not yet emitted
                }
            }
            if (ready) { pick = i; break; }     // ascending scan -> first ready is the lowest id
        }
        if (pick == m) { r.ok = false; return r; }   // nothing emittable but nodes remain -> cycle
        emitted[pick] = 1;
        r.order.push_back(sub[pick]);
        ++emittedCount;
    }
    return r;
}

// --- Rebuild — the incremental result --------------------------------------------------------------------
struct RebuildResult { std::vector<NodeId> recompiled; CacheKey digest = 0; bool ok = true; };

// Union the InvalidationSet of every changed node (sorted-unique), topo-order it (RebuildOrder), and return
// the ordered recompile list + a digest over it (PutU32 each NodeId in order -> DigestBytes). ok=false on a
// cycle. The COUNT proves incrementality: only changed + dependents are in `recompiled`, NOT the whole graph.
inline RebuildResult Rebuild(const DepGraph& g, const std::vector<NodeId>& changedNodes) {
    RebuildResult r;
    std::vector<NodeId> uni;   // sorted-unique union of all invalidation sets
    for (std::size_t c = 0; c < changedNodes.size(); ++c) {
        const std::vector<NodeId> inv = InvalidationSet(g, changedNodes[c]);
        for (std::size_t i = 0; i < inv.size(); ++i) InsertSortedUniqueDep(uni, inv[i]);
    }
    const OrderResult ord = RebuildOrder(g, uni);
    if (!ord.ok) { r.ok = false; return r; }
    r.recompiled = ord.order;
    std::vector<uint8_t> b;
    for (std::size_t i = 0; i < r.recompiled.size(); ++i) PutU32(b, r.recompiled[i]);
    r.digest = net::DigestBytes(b.data(), b.size());
    return r;
}

// --- Fixture (FIXED forever — the golden pins its invalidation sets + rebuild) ----------------------------
// A fixed 6-node graph: 0 mesh, 1 mesh, 2 scene deps{0,1}; 3 texture, 4 material deps{3}, 5 scene2 deps{4,1}.
// (node 1 is shared by scene 2 and scene2 5; node 3 only affects 4 and 5.) Keep FIXED.
inline DepGraph MakeShowcaseGraph() {
    DepGraph g;
    AddNode(g, 0);   // mesh
    AddNode(g, 1);   // mesh
    AddDep(g, 2, 0); // scene depends on mesh 0
    AddDep(g, 2, 1); // scene depends on mesh 1
    AddNode(g, 3);   // texture
    AddDep(g, 4, 3); // material depends on texture 3
    AddDep(g, 5, 4); // scene2 depends on material 4
    AddDep(g, 5, 1); // scene2 depends on mesh 1
    return g;
}

// =========================================================================================================
// Slice ASSET-S5 — Manifest / batch compile, the build AS a net::Session lockstep replay (issue #16).
// APPEND-ONLY below S4; S1's key 0x7fb6a48b4b99f1b7, S2's artifact 0xf7ee13c169dc0464, S3's cache
// 0x029174f13e64c9f1, and S4's rebuild 0x0808b56e0322d8c1 are UNCHANGED. Ties S1–S4 into a BATCH COMPILE
// that produces a deterministic MANIFEST (every asset's NodeId -> its artifact digest) AND makes the build
// LITERALLY a net::Session lockstep replay: the same job stream yields the same per-tick DigestTrace and the
// same final manifest, on any machine, in any submit order. NO new include (net/session.h already present —
// S5 USES more of it: InputRing / RunLockstep / DigestTrace). mtime NEVER enters the manifest or any digest.
// =========================================================================================================

// --- The manifest (the build output — sorted-unique by node, order-stable) -------------------------------
struct ManifestEntry { NodeId node = 0; CacheKey artifactDigest = 0; };
struct Manifest { std::vector<ManifestEntry> entries; };   // SORTED-UNIQUE by node (the order-stable store)

// Sorted-unique insert (or overwrite) of `node` -> `artifactDigest` (the InsertSortedUnique binary-search
// pattern; mirrors S3's Insert). A re-upsert of an existing node overwrites its digest. NO <algorithm>.
inline void UpsertManifest(Manifest& m, NodeId node, CacheKey artifactDigest) {
    std::size_t lo = 0, hi = m.entries.size();
    while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (m.entries[mid].node < node) lo = mid + 1; else hi = mid; }
    if (lo < m.entries.size() && m.entries[lo].node == node) { m.entries[lo].artifactDigest = artifactDigest; return; }
    m.entries.insert(m.entries.begin() + (std::ptrdiff_t)lo, ManifestEntry{ node, artifactDigest });
}

// Hand-LE: PutU32(count), then per entry IN SORTED-NODE ORDER: PutU32(node), PutU64(artifactDigest). The
// entries are kept sorted by node, so the byte stream is identical regardless of compile order — the
// content-addressed property AT THE BUILD LEVEL. mtime NEVER appears.
inline std::vector<uint8_t> SerializeManifest(const Manifest& m) {
    std::vector<uint8_t> b;
    PutU32(b, (uint32_t)m.entries.size());
    for (const ManifestEntry& e : m.entries) {
        PutU32(b, e.node);
        PutU64(b, e.artifactDigest);
    }
    return b;
}

inline CacheKey ManifestDigest(const Manifest& m) {
    std::vector<uint8_t> b = SerializeManifest(m);
    return net::DigestBytes(b.data(), b.size());
}

// --- The compile job (a value-copyable net::Session Input) -----------------------------------------------
// One asset to compile: a stable NodeId + a pointer to its raw bytes (static fixtures in the test) + params.
// Value-copyable (shallow ptr) so net::Session can ring/snapshot it; the bytes outlive the run.
struct CompileJob { NodeId node = 0; const char* bytes = nullptr; std::size_t n = 0; CompileParams params; };

// --- CompileSet — the plain batch (order-independent) ----------------------------------------------------
// Compile every job into `cache` (GetOrCompile) and record node -> DigestArtifact(blob) in a manifest.
// Order-independent: the manifest is sorted by node, so ANY job ordering yields the same ManifestDigest.
inline Manifest CompileSet(const std::vector<CompileJob>& jobs, AssetCache& cache) {
    Manifest m;
    for (const CompileJob& j : jobs) {
        CompileResult r = GetOrCompile(cache, j.bytes, j.n, j.params);
        UpsertManifest(m, j.node, DigestArtifact(r.blob));
    }
    return m;
}

// --- StepBuild — the build AS a net::Session lockstep replay (THE HEADLINE) -------------------------------
// The deterministic build transition: compile each job this tick into `cache`, upsert its artifact digest
// into the manifest. A free function the test wraps in a lambda capturing the cache (like seq's
// StepPlayhead). Signature matches net::Session: step(World&, const std::vector<Input>&, uint32_t tick).
inline void StepBuild(AssetCache& cache, Manifest& w, const std::vector<CompileJob>& jobs, uint32_t /*tick*/) {
    for (const CompileJob& j : jobs) {
        CompileResult r = GetOrCompile(cache, j.bytes, j.n, j.params);
        UpsertManifest(w, j.node, DigestArtifact(r.blob));
    }
}

// --- Fixtures (FIXED forever — the golden pins the manifest + trace) --------------------------------------
// 3 jobs: node 0 = ShowcaseRawBytes (S1), node 1 = ShowcaseRawBytesB, node 2 = ShowcaseRawBytesC (S3), all
// with ShowcaseParams(). Keep FIXED.
inline std::vector<CompileJob> MakeShowcaseJobs() {
    std::vector<CompileJob> jobs;
    jobs.push_back(CompileJob{ 0, ShowcaseRawBytes(),  ShowcaseRawLen(),  ShowcaseParams() });
    jobs.push_back(CompileJob{ 1, ShowcaseRawBytesB(), ShowcaseRawLenB(), ShowcaseParams() });
    jobs.push_back(CompileJob{ 2, ShowcaseRawBytesC(), ShowcaseRawLenC(), ShowcaseParams() });
    return jobs;
}

// The SAME three jobs in a DIFFERENT order (for the order-independence golden). FIXED.
inline std::vector<CompileJob> MakeShowcaseJobsReordered() {
    std::vector<CompileJob> jobs;
    jobs.push_back(CompileJob{ 2, ShowcaseRawBytesC(), ShowcaseRawLenC(), ShowcaseParams() });
    jobs.push_back(CompileJob{ 0, ShowcaseRawBytes(),  ShowcaseRawLen(),  ShowcaseParams() });
    jobs.push_back(CompileJob{ 1, ShowcaseRawBytesB(), ShowcaseRawLenB(), ShowcaseParams() });
    return jobs;
}

}  // namespace hf::asset
