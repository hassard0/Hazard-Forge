#pragma once
// engine/asset/asset_hotreload_live.h — namespace hf::asset (issue #16, the LIVE bridge).
//
// Makes the deterministic hot-reload core (asset_compiler.h, Slice ASSET-S6) GENUINELY LIVE: it wires the
// NodeId-keyed AssetWatcher + HotReload() to the REAL filesystem — real file mtimes (std::filesystem::
// last_write_time, the runtime::FileWatcher DefaultStat precedent) and re-reading the changed file BYTES off
// disk. The bit-exact tests in asset_compiler.h INJECT `currentMtimes` (an in-memory "filesystem"); THIS layer
// stats real paths and re-reads real bytes, so a genuine `:w` on disk -> next PollAndReload() detects it ->
// re-reads -> recompiles ONLY the dirty set. The "watch for changes, no rebuild required" capability, real.
//
// DELIBERATELY THE NON-PURE SIBLING: asset_compiler.h stays the one standalone-clang-compilable, bit-exact,
// NO-<string>/NO-<filesystem> header whose S1-S6 goldens are pinned. This file is APPEND-ONLY beside it and
// MAY use <string>/<filesystem>/<fstream>/<vector> — it is the live wiring, NOT a golden-pinned artifact. It
// adds NO new symbol to asset_compiler.h and changes NONE of its bytes; the LiveAssetSource just holds the
// real path + the current on-disk bytes, and feeds them into the EXISTING HotReload() unchanged.
//
// THE S1 INVARIANT IS PRESERVED EXACTLY: mtime is the TRIGGER only — RealMtime decides WHAT to re-read &
// recompile; it NEVER enters a CacheKey, an artifact, or the manifest digest. The manifestDigest HotReload
// returns is over {NodeId, artifactDigest} (content+params only), identical whether the bytes arrived via an
// injected fixture or a real fopen. Two machines that edit a file to the same bytes get the same digest.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "asset/asset_compiler.h"   // AssetWatcher / WatchAsset / HotReload / CompileJob / DepGraph / cache

namespace hf::asset {

// =========================================================================================================
// The live source — a stable NodeId + a REAL file path + its current on-disk bytes (re-read on edit).
// CompileJob (S5) holds a `const char* bytes` (a non-owning view); LiveAssetSource OWNS the bytes vector so
// the view it hands to HotReload stays valid for the whole reload pass. Bytes are re-read by PollAndReload
// only for nodes whose real mtime increased — unchanged files keep their already-read bytes (and stay S3
// cache hits).
// =========================================================================================================
struct LiveAssetSource {
    NodeId               node = 0;     // the stable logical asset id (the DepGraph / cache key node)
    std::string          path;         // the REAL file path on disk
    CompileParams        params;       // integer compile options (Q16.16 scale etc.) — content+params only
    std::vector<uint8_t> bytes;        // the current on-disk bytes (OWNED; re-read on a real mtime bump)
    int64_t              lastMtime = 0;// the last real mtime we re-read at (diagnostic; watcher is the truth)
};

// The live reloader: the EXISTING deterministic core (watcher + cache + graph) plus the live source list
// (paths + owned bytes). A single PollAndReload() over this is the live loop.
struct LiveHotReloader {
    AssetWatcher                 watcher;   // NodeId-keyed real-mtime baselines (S6)
    AssetCache                   cache;     // content-addressed artifact store (S3)
    DepGraph                     graph;     // dependency graph for incremental invalidation (S4)
    std::vector<LiveAssetSource> sources;   // the live sources (path + owned current bytes)
};

// --- RealMtime — std::filesystem::last_write_time -> a monotonic int64 (the FileWatcher DefaultStat rule) --
// Returns the file_time_type clock's tick count (only ORDERING matters, not the unit — exactly the
// runtime::FileWatcher convention). A missing/unreadable path -> the sentinel -1 (negative == absent), so a
// file that later APPEARS counts as a change (a baseline-(-1) -> a positive mtime is an increase). NEVER an
// artifact key — mtime is the recompile trigger only (the S1 invariant).
inline int64_t RealMtime(const std::string& path) {
    std::error_code ec;
    auto t = std::filesystem::last_write_time(path, ec);
    if (ec) return -1;   // missing / unreadable -> sentinel
    return static_cast<int64_t>(t.time_since_epoch().count());
}

// --- ReadFileBytes — slurp a file's raw bytes (binary). Empty vector if missing/unreadable. ---------------
// Binary mode (no newline translation) so OBJ/FBX/USD bytes are the EXACT on-disk bytes the content hash
// addresses — the same bytes any other machine reads. A missing file yields empty bytes (CompileObj of empty
// -> a valid header-only artifact; the digest still reflects "no geometry").
inline std::vector<uint8_t> ReadFileBytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// --- RegisterLiveAsset — register a real file as a watched, live-reloadable source -------------------------
// Reads the file's current bytes off disk, records the LiveAssetSource (node + path + params + owned bytes),
// adds the node to the dependency graph, and WatchAssets the baseline REAL mtime (so the just-registered
// file is NOT reported as changed on the first PollAndReload — only a SUBSEQUENT edit is). A re-register of an
// existing node updates its path/params/bytes and re-baselines its mtime.
inline void RegisterLiveAsset(LiveHotReloader& r, NodeId node, const std::string& path, const CompileParams& p) {
    const int64_t mtime = RealMtime(path);
    std::vector<uint8_t> bytes = ReadFileBytes(path);
    // upsert the source (linear scan — the source list is tiny, like SourceForNode).
    LiveAssetSource* slot = nullptr;
    for (std::size_t i = 0; i < r.sources.size(); ++i) if (r.sources[i].node == node) { slot = &r.sources[i]; break; }
    if (slot) {
        slot->path = path; slot->params = p; slot->bytes = std::move(bytes); slot->lastMtime = mtime;
    } else {
        r.sources.push_back(LiveAssetSource{ node, path, p, std::move(bytes), mtime });
    }
    AddNode(r.graph, node);              // ensure the node exists in the graph
    WatchAsset(r.watcher, node, mtime);  // baseline the real mtime (S6 watcher) — next poll only sees an INCREASE
}

// --- RegisterDependency — `from` DEPENDS ON `to` (a scene->mesh edge) so dependents recompile -------------
// Thin wrapper over S4's AddDep: when `to`'s file is edited, `from` is in the InvalidationSet and recompiles
// too. (Both nodes should also be RegisterLiveAsset'd to have real bytes; AddDep ensures graph presence.)
inline void RegisterDependency(LiveHotReloader& r, NodeId fromNode, NodeId toNode) {
    AddDep(r.graph, fromNode, toNode);
}

// --- PollAndReload — THE LIVE LOOP: stat real paths, re-read edited bytes, recompile the dirty set ---------
// Stats EVERY source's real path (RealMtime) to build `currentMtimes`; for any node whose real mtime INCREASED
// over the watcher baseline, RE-READS its file bytes off disk into the LiveAssetSource (the live "no rebuild
// required" step). Then builds the std::vector<CompileJob> from the CURRENT sources (node, bytes.data(),
// bytes.size(), params) and calls the EXISTING, unchanged HotReload(watcher, cache, graph, jobs,
// currentMtimes). HotReload itself does the PollChanged -> InvalidationSet -> RebuildOrder -> recompile-dirty
// -> full-manifest pass; it advances the watcher baselines, so an immediate second PollAndReload with no
// further edit reports NO recompile. Returns the ReloadBatch (recompiled set in topo order + new
// manifestDigest + ok).
//
// THE RE-READ-VS-WATCHER ORDER: we compute currentMtimes ONCE, pre-read the increased nodes' bytes against
// THOSE same mtimes (a snapshot), then hand the SAME currentMtimes to HotReload — so the bytes a node
// recompiles from are exactly the bytes that produced the mtime HotReload trips on. (A file edited AGAIN
// between our stat and HotReload's PollChanged is simply caught on the NEXT poll, the standard watcher
// guarantee — no torn read of the digest, mtime stays the trigger only.)
inline ReloadBatch PollAndReload(LiveHotReloader& r) {
    // 1) stat real paths -> currentMtimes (a snapshot the rest of this pass is consistent with).
    std::vector<std::pair<NodeId, int64_t>> currentMtimes;
    currentMtimes.reserve(r.sources.size());
    for (std::size_t i = 0; i < r.sources.size(); ++i) {
        currentMtimes.push_back({ r.sources[i].node, RealMtime(r.sources[i].path) });
    }
    // 2) re-read bytes for any node whose real mtime INCREASED over the watcher baseline (the live re-read).
    //    We compare against the watcher's last-seen mtime WITHOUT mutating it (HotReload's PollChanged does
    //    the authoritative update); this just decides which files to slurp fresh.
    for (std::size_t i = 0; i < r.sources.size(); ++i) {
        const NodeId node = currentMtimes[i].first;
        const int64_t cur = currentMtimes[i].second;
        // binary-search the watcher baseline for this node (mirrors PollChanged's lookup).
        std::size_t lo = 0, hi = r.watcher.seen.size();
        while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (r.watcher.seen[mid].first < node) lo = mid + 1; else hi = mid; }
        const bool present = (lo < r.watcher.seen.size() && r.watcher.seen[lo].first == node);
        if (!present || cur > r.watcher.seen[lo].second) {
            r.sources[i].bytes = ReadFileBytes(r.sources[i].path);   // RE-READ the edited file off disk
            r.sources[i].lastMtime = cur;
        }
    }
    // 3) build CompileJobs from the CURRENT (possibly re-read) source bytes — views into the owned vectors.
    std::vector<CompileJob> jobs;
    jobs.reserve(r.sources.size());
    for (std::size_t i = 0; i < r.sources.size(); ++i) {
        const LiveAssetSource& s = r.sources[i];
        jobs.push_back(CompileJob{ s.node, reinterpret_cast<const char*>(s.bytes.data()), s.bytes.size(), s.params });
    }
    // 4) the EXISTING deterministic reload pass (watch -> invalidate -> recompile dirty -> full manifest).
    return HotReload(r.watcher, r.cache, r.graph, jobs, currentMtimes);
}

}  // namespace hf::asset
