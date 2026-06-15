// Slice AM — file hot-reload watcher implementation. Pure C++ (std::filesystem).
#include "runtime/hot_reload.h"

#include <filesystem>
#include <system_error>

namespace hf::runtime {

StatFn DefaultStat() {
    return [](const std::string& path) -> int64_t {
        std::error_code ec;
        auto t = std::filesystem::last_write_time(path, ec);
        if (ec) return -1;  // missing / unreadable
        // file_time_type's clock duration -> a monotonic integer count. Only ordering matters.
        return static_cast<int64_t>(t.time_since_epoch().count());
    };
}

void FileWatcher::Watch(std::string path) {
    int64_t now = stat_ ? stat_(path) : -1;
    for (auto& e : entries_) {
        if (e.path == path) { e.mtime = now; return; }  // re-baseline an existing watch
    }
    entries_.push_back(Entry{std::move(path), now});
}

std::vector<std::string> FileWatcher::Poll() {
    std::vector<std::string> changed;
    for (auto& e : entries_) {
        int64_t cur = stat_ ? stat_(e.path) : -1;
        // Changed when the mtime increased, OR the file transitioned from missing (negative
        // baseline) to present. A present->missing transition is NOT a change (keeps old baseline
        // intact until the file returns, so a transient delete doesn't fire spuriously).
        bool appeared = (e.mtime < 0 && cur >= 0);
        bool advanced = (cur >= 0 && e.mtime >= 0 && cur > e.mtime);
        if (appeared || advanced) {
            changed.push_back(e.path);
            e.mtime = cur;
        } else if (cur >= 0) {
            e.mtime = cur;  // keep baseline current (handles equal mtimes without re-firing)
        }
        // cur < 0 (missing): leave baseline as-is so a later reappearance counts as "appeared".
    }
    return changed;
}

}  // namespace hf::runtime
