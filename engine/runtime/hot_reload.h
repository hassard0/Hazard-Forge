#pragma once
// Slice AM — file hot-reload watcher. PURE C++ (hf_core): <filesystem> + <functional> + std only;
// NO rhi/SDL/backend symbols, so it lives in hf_core (ASan-scoped, unit-tested).
//
// FileWatcher polls a set of {path, lastMtime} entries and reports which paths changed since the
// previous Poll(). The live editor watches the active scene JSON (reload via LoadScene) and the
// compiled shader .spv outputs (recreate shader module, best-effort).
//
// The mtime source is INJECTABLE so the watcher is testable without touching the real filesystem:
// a StatFn returns a monotonic mtime for a path (or a negative value if the path is missing). The
// default ctor uses a std::filesystem::last_write_time-based stat for the live loop; tests inject a
// fake stat backed by an in-memory map and drive the clock by hand.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace hf::ecs { class Registry; }
namespace hf::scene { struct SceneResources; }

namespace hf::runtime {

// Returns a monotonic modification time for `path` (e.g. nanoseconds since epoch). A NEGATIVE return
// means the path is currently missing/unreadable. Only the ordering of returned values matters, not
// the unit — Poll() reports a change when the value INCREASES (or transitions from missing->present).
using StatFn = std::function<int64_t(const std::string&)>;

// A std::filesystem-backed stat for live use. Missing/unreadable -> -1.
StatFn DefaultStat();

class FileWatcher {
public:
    // Default: watch real files via DefaultStat(). Tests pass a fake stat.
    explicit FileWatcher(StatFn stat = DefaultStat()) : stat_(std::move(stat)) {}

    // Start tracking `path`. Records its CURRENT mtime as the baseline, so the next Poll() does NOT
    // report it as changed (only subsequent increases do). Re-watching a path re-baselines it.
    void Watch(std::string path);

    // Paths whose mtime increased since the previous Poll() (or since Watch()). A path that newly
    // APPEARS (was missing at baseline, now present) counts as changed; a path that DISAPPEARS does
    // not (its baseline is kept negative until it returns). Order matches Watch() order.
    std::vector<std::string> Poll();

    std::size_t Count() const { return entries_.size(); }

private:
    struct Entry { std::string path; int64_t mtime; };
    StatFn stat_;
    std::vector<Entry> entries_;
};

// --- Slice DX4 — the deterministic headless scene hot-reload (factored from the live editor's
// FileWatcher-driven LoadScene swap, main.cpp:10972-10995). ApplyReload is the reusable, golden-able
// reload primitive: a reload is exactly a clear + LoadScene into the SAME registry, and its VALUE is
// the guarantee that this equals a COLD load of the same path (no stale entities/components survive).
// scene_io + ecs are hf_core/backend-agnostic — the header's "NO rhi/SDL/backend symbols" contract is
// preserved (these are intra-hf_core deps; FileWatcher above is byte-unchanged).

struct ReloadResult {
    bool loaded;          // LoadScene(path) succeeded (no throw)
    int  entityCount;     // entities live in `reg` after the reload
    bool equalToColdLoad; // DumpScene(reg after reload) == DumpScene(a fresh cold-load of path)
};

// CLEAR `reg`, LoadScene(`path`) into it, and report whether the reloaded world's DumpScene is
// byte-identical to a fresh registry cold-loaded from the same path (the load-equivalence guarantee:
// a reload == a cold load, no residue). On a LoadScene throw, `reg` is left CLEARED (empty) and
// {loaded:false, entityCount:0, equalToColdLoad:false} is returned — never a half-built world.
ReloadResult ApplyReload(ecs::Registry& reg, const scene::SceneResources& res,
                         const std::string& path);

}  // namespace hf::runtime
