#pragma once

// Deterministic chunk-diff layer (issue #41) — the engine path for RUNTIME MUTATIONS that must survive
// chunk streaming regeneration.
//
// Chunk streaming generates a chunk as a pure function of (worldSeed, cx, cz): re-entering a chunk yields
// byte-identical content — great for the determinism brand, until the player changes something (destroys a
// building, parks a car, paints graffiti, moves a prop). Today a sample author hand-maintains a parallel
// `std::unordered_set<uint64_t> destroyed` checked in render + collision loops; every new mutation kind
// needs another parallel structure. This layer is the engine dispatch: a per-(cx,cz) DIFF of removed /
// added / modified entities, content-addressed by a stable entity hash, applied to the freshly generated
// entity list. The brand extends from "same world from seed" to "same world from (seed + diff log)" — which
// IS a save-game (a save is just `(seed, ChunkDiffStore)`), a multiplayer world-sync delta, and an undo log.
//
// DETERMINISM IS LOAD-BEARING and is WHY this exists, so — unlike the issue's `std::unordered_map` sketch —
// every container here is an ORDER-STABLE sorted vector (removed sorted-unique by hash; modified sorted-
// unique by hash; chunks sorted by packed key). `unordered_*` iteration order is implementation-defined and
// would make the diff hash + serialization differ across runs/compilers — the exact property we must keep.
// So a ChunkDiffStore hashes + serializes byte-identically regardless of the order the marks were applied
// (content-addressed), which is what makes it a deterministic save/sync artifact.
//
// Self-contained: only <cstdint>/<cstddef>/<vector> + net/session.h (DigestBytes + the LE byte appenders'
// discipline). NO RHI / GPU / float / clock / RNG / <map> / <unordered_*> / std::hash.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "net/session.h"   // hf::net::DigestBytes (FNV-1a-64) — self-contained, read-only reuse

namespace hf::world {

using EntityHash = uint64_t;   // a stable per-entity key (positional or seed-derived — the caller's choice)
using ChunkKey   = uint64_t;   // packed (cx, cz)

// Pack a signed chunk coordinate into a stable key: high 32 = cx, low 32 = cz (two's-complement bits).
inline ChunkKey PackChunk(int32_t cx, int32_t cz) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(cx)) << 32) |
            static_cast<uint64_t>(static_cast<uint32_t>(cz));
}

// A generic streamed entity: a stable hash + a kind tag + an integer transform (pos + yaw). Integer so it is
// bit-exact + hashable + serializable; `kind` distinguishes building / car / graffiti / npc / prop; richer
// per-entity payloads (v1 scope note) are carried in a caller side-table keyed by `hash`.
struct EntityRef {
    EntityHash hash = 0;
    uint32_t   kind = 0;
    int32_t    x = 0, y = 0, z = 0;
    int32_t    yaw = 0;
};

// A modification delta applied to a generated entity (move/rotate + small state flags, e.g. a graffiti id).
struct EntityOverride {
    int32_t  dx = 0, dy = 0, dz = 0, dyaw = 0;
    uint32_t flags = 0;
};

// One chunk's diff. All three lists are order-stable (removed/modified sorted by hash; added in insertion
// order — the caller's deterministic add order).
struct ChunkDiff {
    std::vector<EntityHash>                          removed;    // sorted-unique
    std::vector<std::pair<EntityHash, EntityOverride>> modified; // sorted-unique by .first
    std::vector<EntityRef>                           added;      // appended runtime spawns
};

// --- ordered insert helpers (binary-search, keep sorted-unique) -------------------------------------
inline bool InsertSortedUnique(std::vector<EntityHash>& v, EntityHash h) {
    std::size_t lo = 0, hi = v.size();
    while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (v[mid] < h) lo = mid + 1; else hi = mid; }
    if (lo < v.size() && v[lo] == h) return false;      // already present
    v.insert(v.begin() + static_cast<std::ptrdiff_t>(lo), h);
    return true;
}
inline void UpsertSorted(std::vector<std::pair<EntityHash, EntityOverride>>& v, EntityHash h,
                         const EntityOverride& o) {
    std::size_t lo = 0, hi = v.size();
    while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (v[mid].first < h) lo = mid + 1; else hi = mid; }
    if (lo < v.size() && v[lo].first == h) { v[lo].second = o; return; }   // overwrite (last-write-wins)
    v.insert(v.begin() + static_cast<std::ptrdiff_t>(lo), {h, o});
}

// --- ChunkDiffStore: the per-(cx,cz) diff log -------------------------------------------------------
// Chunks are held in a sorted vector<(ChunkKey, ChunkDiff)> so iteration / hashing / serialization are
// order-stable. (A real open world has thousands of chunks but only the mutated ones get an entry — the
// store is sparse over the diffed chunks, never the whole world.)
class ChunkDiffStore {
public:
    void MarkRemoved(int32_t cx, int32_t cz, EntityHash h) { InsertSortedUnique(At(cx, cz).removed, h); }
    void MarkModified(int32_t cx, int32_t cz, EntityHash h, const EntityOverride& o) {
        UpsertSorted(At(cx, cz).modified, h, o);
    }
    void MarkAdded(int32_t cx, int32_t cz, const EntityRef& e) { At(cx, cz).added.push_back(e); }

    // The diff for a chunk, or nullptr if the chunk has never been mutated (a pristine generated chunk).
    const ChunkDiff* Find(int32_t cx, int32_t cz) const {
        const ChunkKey k = PackChunk(cx, cz);
        std::size_t lo = 0, hi = chunks_.size();
        while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (chunks_[mid].first < k) lo = mid + 1; else hi = mid; }
        return (lo < chunks_.size() && chunks_[lo].first == k) ? &chunks_[lo].second : nullptr;
    }

    const std::vector<std::pair<ChunkKey, ChunkDiff>>& Chunks() const { return chunks_; }

private:
    ChunkDiff& At(int32_t cx, int32_t cz) {
        const ChunkKey k = PackChunk(cx, cz);
        std::size_t lo = 0, hi = chunks_.size();
        while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (chunks_[mid].first < k) lo = mid + 1; else hi = mid; }
        if (lo < chunks_.size() && chunks_[lo].first == k) return chunks_[lo].second;
        chunks_.insert(chunks_.begin() + static_cast<std::ptrdiff_t>(lo), {k, ChunkDiff{}});
        return chunks_[lo].second;
    }
    std::vector<std::pair<ChunkKey, ChunkDiff>> chunks_;   // sorted by ChunkKey
};

// --- ApplyChunk: the dispatch — fold a chunk's diff into its freshly generated entity list -----------
// Given the entities a generator produced for (cx,cz) IN GENERATION ORDER, return the runtime entity list:
//   1. drop any entity whose hash is in `removed`,
//   2. apply the matching `EntityOverride` (add the integer delta) to surviving entities,
//   3. append the `added` runtime spawns (in insertion order).
// Deterministic of (generated, diff) alone — re-entering the chunk reproduces the byte-identical result.
inline std::vector<EntityRef> ApplyChunk(const std::vector<EntityRef>& generated, const ChunkDiff* diff) {
    if (!diff) return generated;   // pristine chunk — no mutations
    std::vector<EntityRef> out;
    out.reserve(generated.size() + diff->added.size());
    for (const EntityRef& e : generated) {
        // removed? (binary search the sorted list)
        {
            std::size_t lo = 0, hi = diff->removed.size();
            while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (diff->removed[mid] < e.hash) lo = mid + 1; else hi = mid; }
            if (lo < diff->removed.size() && diff->removed[lo] == e.hash) continue;   // dropped
        }
        EntityRef r = e;
        // modified? (binary search) — apply the integer delta
        {
            std::size_t lo = 0, hi = diff->modified.size();
            while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (diff->modified[mid].first < e.hash) lo = mid + 1; else hi = mid; }
            if (lo < diff->modified.size() && diff->modified[lo].first == e.hash) {
                const EntityOverride& o = diff->modified[lo].second;
                r.x += o.dx; r.y += o.dy; r.z += o.dz; r.yaw += o.dyaw;
            }
        }
        out.push_back(r);
    }
    for (const EntityRef& a : diff->added) out.push_back(a);   // runtime spawns, insertion order
    return out;
}

inline std::vector<EntityRef> ApplyChunk(const ChunkDiffStore& store, int32_t cx, int32_t cz,
                                         const std::vector<EntityRef>& generated) {
    return ApplyChunk(generated, store.Find(cx, cz));
}

// --- Serialization (hand little-endian, the wav.cpp/replay discipline — NO host-struct memcpy) -------
// A save game IS `(seed, ChunkDiffStore bytes)`. The byte layout is order-stable (sorted store) so the same
// logical diff serializes byte-identically across runs/compilers — savable, diffable, ship-over-the-wire.
inline void PutU32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(uint8_t(v)); b.push_back(uint8_t(v >> 8)); b.push_back(uint8_t(v >> 16)); b.push_back(uint8_t(v >> 24));
}
inline void PutU64(std::vector<uint8_t>& b, uint64_t v) { PutU32(b, uint32_t(v)); PutU32(b, uint32_t(v >> 32)); }
inline void PutI32(std::vector<uint8_t>& b, int32_t v) { PutU32(b, static_cast<uint32_t>(v)); }

inline void PutEntity(std::vector<uint8_t>& b, const EntityRef& e) {
    PutU64(b, e.hash); PutU32(b, e.kind); PutI32(b, e.x); PutI32(b, e.y); PutI32(b, e.z); PutI32(b, e.yaw);
}
inline void PutOverride(std::vector<uint8_t>& b, const EntityOverride& o) {
    PutI32(b, o.dx); PutI32(b, o.dy); PutI32(b, o.dz); PutI32(b, o.dyaw); PutU32(b, o.flags);
}

inline std::vector<uint8_t> SerializeStore(const ChunkDiffStore& store) {
    std::vector<uint8_t> b;
    PutU32(b, static_cast<uint32_t>(store.Chunks().size()));
    for (const auto& [key, d] : store.Chunks()) {
        PutU64(b, key);
        PutU32(b, static_cast<uint32_t>(d.removed.size()));
        for (EntityHash h : d.removed) PutU64(b, h);
        PutU32(b, static_cast<uint32_t>(d.modified.size()));
        for (const auto& [h, o] : d.modified) { PutU64(b, h); PutOverride(b, o); }
        PutU32(b, static_cast<uint32_t>(d.added.size()));
        for (const EntityRef& e : d.added) PutEntity(b, e);
    }
    return b;
}

// --- Deserialization: load a store back from save bytes (the inverse of SerializeStore) -------------
// Reads via a cursor; the decoded store is byte-identical-on-re-serialize to the original (a real round-
// trip, so a save game genuinely restores). Returns true on success; false on a truncated/short buffer.
inline uint32_t GetU32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
inline uint64_t GetU64(const uint8_t* p) { return uint64_t(GetU32(p)) | (uint64_t(GetU32(p + 4)) << 32); }
inline int32_t  GetI32(const uint8_t* p) { return static_cast<int32_t>(GetU32(p)); }

inline bool DeserializeStore(const std::vector<uint8_t>& b, ChunkDiffStore& out) {
    std::size_t off = 0;
    auto need = [&](std::size_t n) { return off + n <= b.size(); };
    if (!need(4)) return false;
    const uint32_t chunkCount = GetU32(&b[off]); off += 4;
    for (uint32_t c = 0; c < chunkCount; ++c) {
        if (!need(8 + 4)) return false;
        const ChunkKey key = GetU64(&b[off]); off += 8;
        const int32_t cx = static_cast<int32_t>(uint32_t(key >> 32));
        const int32_t cz = static_cast<int32_t>(uint32_t(key & 0xFFFFFFFFu));
        const uint32_t nRem = GetU32(&b[off]); off += 4;
        for (uint32_t i = 0; i < nRem; ++i) { if (!need(8)) return false; out.MarkRemoved(cx, cz, GetU64(&b[off])); off += 8; }
        if (!need(4)) return false;
        const uint32_t nMod = GetU32(&b[off]); off += 4;
        for (uint32_t i = 0; i < nMod; ++i) {
            if (!need(8 + 20)) return false;
            const EntityHash h = GetU64(&b[off]); off += 8;
            EntityOverride o; o.dx = GetI32(&b[off]); o.dy = GetI32(&b[off + 4]); o.dz = GetI32(&b[off + 8]);
            o.dyaw = GetI32(&b[off + 12]); o.flags = GetU32(&b[off + 16]); off += 20;
            out.MarkModified(cx, cz, h, o);
        }
        if (!need(4)) return false;
        const uint32_t nAdd = GetU32(&b[off]); off += 4;
        for (uint32_t i = 0; i < nAdd; ++i) {
            if (!need(28)) return false;   // EntityRef = hash(8)+kind(4)+x/y/z/yaw(16) = 28 bytes
            EntityRef e; e.hash = GetU64(&b[off]); e.kind = GetU32(&b[off + 8]); e.x = GetI32(&b[off + 12]);
            e.y = GetI32(&b[off + 16]); e.z = GetI32(&b[off + 20]); e.yaw = GetI32(&b[off + 24]); off += 28;
            out.MarkAdded(cx, cz, e);
        }
    }
    return true;
}

// A pinned-golden currency for a whole store / an applied entity list.
inline uint64_t DigestStore(const ChunkDiffStore& store) {
    const std::vector<uint8_t> b = SerializeStore(store);
    return net::DigestBytes(b.data(), b.size());
}
inline uint64_t DigestEntities(const std::vector<EntityRef>& es) {
    std::vector<uint8_t> b;
    for (const EntityRef& e : es) PutEntity(b, e);
    return net::DigestBytes(b.data(), b.size());
}

}  // namespace hf::world
