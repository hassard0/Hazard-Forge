// Real-filesystem integration test for the LIVE asset hot-reload bridge
// (engine/asset/asset_hotreload_live.h, issue #16).
//
// Unlike asset_compiler_test (which INJECTS currentMtimes — an in-memory "filesystem"), THIS test does REAL
// file IO: it writes a temp OBJ to std::filesystem::temp_directory_path(), registers it, then OVERWRITES it
// with different geometry on disk and proves the next PollAndReload() detects the edit (real mtime bump),
// re-reads the bytes, and recompiles EXACTLY the dirty node — the new manifestDigest reflecting the new
// content. It asserts the STRUCTURE (edited -> recompiled + digest changes; unedited -> not recompiled),
// which is deterministic even though wall-clock mtimes vary machine-to-machine.
//
// NOT a pure golden test (real IO, no pinned hash) — it links hf_core for net::DigestBytes and is registered
// as an ordinary test. asset_compiler.h is UNTOUCHED; this exercises only the additive live sibling.

#include "asset/asset_hotreload_live.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include "test_main.h"

using namespace hf::asset;

static int g_fail = 0;
static void check(bool c, const char* what) {
    if (!c) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else    { std::printf("PASS: %s\n", what); }
}

// Write `text` to `path` as raw bytes (binary, truncating) and flush+close so the OS commits the mtime.
static void WriteFile(const std::filesystem::path& path, const char* text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    out.close();
}

// Did `set` (a recompiled NodeId list) contain `node`?
static bool Contains(const std::vector<NodeId>& set, NodeId node) {
    for (NodeId n : set) if (n == node) return true;
    return false;
}

// Force `path`'s mtime strictly forward so coarse-resolution filesystems (or two writes in the same tick)
// still register an INCREASE — keeps the assertions deterministic regardless of clock granularity.
static void BumpMtimeForward(const std::filesystem::path& path) {
    std::error_code ec;
    auto t = std::filesystem::last_write_time(path, ec);
    if (!ec) std::filesystem::last_write_time(path, t + std::chrono::seconds(2), ec);
}

int main() {
    HF_TEST_MAIN_INIT();

    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path();
    // A unique-ish temp path (clock-tagged) so concurrent test runs don't collide.
    const unsigned long long tag =
        (unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path objPath = dir / ("hf_live_hotreload_" + std::to_string(tag) + ".obj");
    const std::string path = objPath.string();

    // --- The two real geometries (a triangle, then a DIFFERENT triangle — a moved apex) ------------------
    const char* triA = "# live-hotreload triangle A\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
    const char* triB = "# live-hotreload triangle B (moved apex)\nv 0 0 0\nv 1 0 0\nv 0 5 0\nf 1 2 3\n";

    const NodeId kMesh = 0;
    CompileParams params;   // defaults (recomputeNormals=0, scale=kOne, etc.)

    // ===== (1) Write the temp OBJ + register it. The just-registered baseline -> first poll, no edit. =====
    WriteFile(objPath, triA);
    check(fs::exists(objPath), "live: temp OBJ written to disk");

    LiveHotReloader r;
    RegisterLiveAsset(r, kMesh, path, params);
    check(r.sources.size() == 1, "live: RegisterLiveAsset recorded the source");
    check(!r.sources[0].bytes.empty(), "live: RegisterLiveAsset read the real file bytes off disk");

    // First poll with NO edit since the register baseline -> NO recompile (the watch baseline is current).
    ReloadBatch b0 = PollAndReload(r);
    check(b0.ok, "live: first PollAndReload ok");
    check(!Contains(b0.recompiled, kMesh),
          "live: just-registered baseline, no edit -> node NOT recompiled (watch is current)");
    const CacheKey digestA = b0.manifestDigest;
    std::printf("live: manifest digest after baseline (triangle A) = 0x%016llx\n",
                (unsigned long long)digestA);

    // ===== (2) OVERWRITE the file with DIFFERENT geometry on disk -> next poll recompiles the node. =====
    std::this_thread::sleep_for(std::chrono::milliseconds(50));  // help fine-res clocks separate the writes
    WriteFile(objPath, triB);
    BumpMtimeForward(objPath);   // guarantee a strict mtime INCREASE even on coarse-resolution filesystems

    ReloadBatch b1 = PollAndReload(r);
    check(b1.ok, "live: edit-poll ok");
    check(Contains(b1.recompiled, kMesh),
          "live: a REAL file edit on disk -> the node IS recompiled (the dirty set contains it)");
    // The re-read bytes are the EDITED geometry (triB) — prove the live re-read actually happened.
    check(std::string(r.sources[0].bytes.begin(), r.sources[0].bytes.end()) == std::string(triB),
          "live: PollAndReload RE-READ the edited bytes off disk (source bytes == triangle B)");
    const CacheKey digestB = b1.manifestDigest;
    std::printf("live: manifest digest after edit (triangle B) = 0x%016llx\n",
                (unsigned long long)digestB);
    check(digestB != digestA,
          "live: the edited geometry produced a NEW artifact -> manifestDigest CHANGED (content-addressed)");

    // ===== (3) A second poll with NO further edit -> NO recompile (the watch baseline updated). =====
    ReloadBatch b2 = PollAndReload(r);
    check(b2.ok, "live: settle-poll ok");
    check(!Contains(b2.recompiled, kMesh),
          "live: an unedited poll after the edit -> node NOT recompiled (baseline advanced)");
    check(b2.manifestDigest == digestB,
          "live: settled manifest digest stable + equals the post-edit digest (no spurious rebuild)");

    // ===== (4) Determinism: a fresh reloader over the SAME on-disk bytes -> the SAME digest. ==========
    {
        LiveHotReloader r2;
        RegisterLiveAsset(r2, kMesh, path, params);
        ReloadBatch fresh = PollAndReload(r2);
        check(fresh.manifestDigest == digestB,
              "live: a fresh reloader over the same on-disk bytes -> identical manifestDigest (mtime-free, "
              "content-addressed)");
    }

    // --- Clean up the temp file. --------------------------------------------------------------------------
    std::error_code ec;
    fs::remove(objPath, ec);
    check(!fs::exists(objPath), "live: temp OBJ cleaned up");

    if (g_fail == 0) std::printf("\nasset_hotreload_live_test: ALL CHECKS PASSED\n");
    else             std::printf("\nasset_hotreload_live_test: %d CHECK(S) FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
