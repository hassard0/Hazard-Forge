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

}  // namespace hf::runtime
