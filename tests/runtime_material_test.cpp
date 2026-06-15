// Unit test for the live runtime material-authoring controller (engine/material/live_material.*,
// Slice AW). TDD: written BEFORE the implementation. This pins the RELOAD ORCHESTRATION — the host
// logic that drives the live edit->recompile->swap loop — independent of the GPU and of dxc:
//
//   1. SWAP ON SUCCESS: a stub compiler that returns SPIR-V words causes the controller to promote
//      the new bytes to "active", bump swapCount, and set changed().
//   2. KEEP-OLD + RECORD-ERROR ON FAILURE: a stub compiler that returns nullopt leaves the PREVIOUS
//      active SPIR-V in place, does NOT bump swapCount, and surfaces lastError() (fail-safe).
//   3. FILEWATCHER FIRES EXACTLY ON MTIME CHANGE: Poll() reloads only after the watched file's
//      (injected) mtime increases — not before, and not twice for one change.
//   4. MALFORMED / INVALID GRAPH DOES NOT SWAP: a .mat.json that fails to parse, or a graph that
//      fails validation, keeps the previous material and surfaces an error (the compiler is never
//      even reached for an invalid graph).
//
// The compiler is STUBBED so this test is GPU/dxc-free + deterministic; the REAL dxc-subprocess
// compile is integration-tested via the --material-live-shot byte-equality proof on Windows.
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests. Uses temp files on disk for the
// loader + a fake-stat FileWatcher for the mtime poll (no real filesystem mtime dependence).
#include "material/live_material.h"
#include "material/shader_graph.h"
#include "runtime/hot_reload.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A minimal valid showcase-shaped material document (parses + validates + codegens).
static const char* kValidMat = R"JSON({
  "name": "unit_mat",
  "nodes": [
    { "id": 1, "type": "UV" },
    { "id": 2, "type": "TextureSample", "texture": "checker" },
    { "id": 3, "type": "Constant", "value": [0.9, 0.2, 0.1, 1.0], "outType": "float4" },
    { "id": 4, "type": "Fresnel", "power": 3.0 },
    { "id": 5, "type": "Lerp" },
    { "id": 99, "type": "PBROutput" }
  ],
  "edges": [
    { "from": 1, "to": 2, "port": "uv" },
    { "from": 2, "to": 5, "port": "a" },
    { "from": 3, "to": 5, "port": "b" },
    { "from": 4, "to": 5, "port": "t" },
    { "from": 5, "to": 99, "port": "baseColor" }
  ]
})JSON";

// A DIFFERENT but still valid material (different baseColor constant) so a swap is observable.
static const char* kValidMat2 = R"JSON({
  "name": "unit_mat2",
  "nodes": [
    { "id": 1, "type": "Constant", "value": [0.1, 0.8, 0.3, 1.0], "outType": "float3" },
    { "id": 99, "type": "PBROutput" }
  ],
  "edges": [
    { "from": 1, "to": 99, "port": "baseColor" }
  ]
})JSON";

// A graph that PARSES as JSON but FAILS validation (no PBROutput sink).
static const char* kInvalidGraph = R"JSON({
  "name": "broken",
  "nodes": [ { "id": 1, "type": "UV" } ],
  "edges": []
})JSON";

// Not JSON at all.
static const char* kMalformed = "{ this is not json";

static std::string TempPath(const char* leaf) {
    const char* tmp = std::getenv("TEMP");
    if (!tmp) tmp = std::getenv("TMPDIR");
    if (!tmp) tmp = ".";
    return std::string(tmp) + "/hf_runtime_mat_" + leaf;
}

static void WriteFile(const std::string& path, const char* contents) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(contents, (std::streamsize)std::char_traits<char>::length(contents));
}

int main() {
    using namespace material;

    const std::string matPath = TempPath("a.mat.json");
    const std::string matPath2 = TempPath("b.mat.json");

    // A stub compiler whose success/failure + payload we control. It NEVER touches dxc/GPU. It
    // records how many times it was invoked so we can assert the controller does NOT call it for an
    // invalid graph. The success payload encodes a tiny "tag" word so distinct graphs yield distinct
    // SPIR-V (lets us see a swap happened).
    int compileCalls = 0;
    bool compileShouldFail = false;
    auto stubCompiler = [&](const Graph& g, std::string* err) -> std::optional<std::vector<uint32_t>> {
        ++compileCalls;
        if (compileShouldFail) { if (err) *err = "stub: forced compile failure"; return std::nullopt; }
        // Derive a deterministic word from the graph so different graphs -> different "SPIR-V".
        uint32_t tag = (uint32_t)g.nodes.size() * 1000u + (uint32_t)g.edges.size();
        return std::vector<uint32_t>{0x07230203u, tag, 0xDEADBEEFu};
    };

    // ============================ 1. SWAP ON SUCCESS ==========================================
    {
        WriteFile(matPath, kValidMat);
        compileCalls = 0; compileShouldFail = false;
        LiveMaterial lm(matPath, stubCompiler);
        bool ok = lm.LoadInitial();
        check(ok, "LoadInitial succeeds on a valid material");
        check(!lm.activeSpirv().empty(), "active SPIR-V is set after a successful initial load");
        check(lm.swapCount() == 1, "initial load counts as one swap");
        check(lm.changed(), "changed() true after the initial successful load");
        check(lm.lastError().empty(), "no error after a successful load");
        check(compileCalls == 1, "compiler invoked exactly once for the initial load");

        // A forced Reload of the same path recompiles + swaps.
        lm.ClearChanged();
        ReloadStatus st = lm.Reload();
        check(st == ReloadStatus::Swapped, "Reload of a valid material swaps");
        check(lm.swapCount() == 2, "Reload bumps swapCount");
        check(lm.changed(), "changed() true after Reload swap");
    }

    // ============================ 2. KEEP-OLD + RECORD-ERROR ON COMPILE FAILURE ================
    {
        WriteFile(matPath, kValidMat);
        compileCalls = 0; compileShouldFail = false;
        LiveMaterial lm(matPath, stubCompiler);
        check(lm.LoadInitial(), "initial load (pre-failure) ok");
        std::vector<uint32_t> before = lm.activeSpirv();
        int swapsBefore = lm.swapCount();

        // Now force the compiler to fail and reload: the controller must KEEP the old SPIR-V.
        compileShouldFail = true;
        lm.ClearChanged();
        ReloadStatus st = lm.Reload();
        check(st == ReloadStatus::Kept, "Reload with a failing compile is Kept (no swap)");
        check(lm.activeSpirv() == before, "previous SPIR-V KEPT after a compile failure");
        check(lm.swapCount() == swapsBefore, "swapCount unchanged after a compile failure");
        check(!lm.lastError().empty(), "lastError recorded after a compile failure");
        check(!lm.changed(), "changed() false after a kept (failed) reload");
    }

    // ============================ 3. FILEWATCHER FIRES EXACTLY ON MTIME CHANGE ==================
    {
        WriteFile(matPath, kValidMat);
        compileShouldFail = false;

        // Injected fake mtime: the controller's watcher polls THIS, not the real filesystem.
        std::map<std::string, int64_t> mtimes;
        mtimes[matPath] = 100;
        runtime::StatFn fake = [&](const std::string& p) -> int64_t {
            auto it = mtimes.find(p);
            return it == mtimes.end() ? -1 : it->second;
        };
        LiveMaterial lm(matPath, stubCompiler, runtime::FileWatcher{fake});
        check(lm.LoadInitial(), "initial load ok (watcher path)");
        int swapsAfterLoad = lm.swapCount();

        // No mtime change yet -> Poll reports NoChange and does NOT reload.
        ReloadStatus s0 = lm.Poll();
        check(s0 == ReloadStatus::NoChange, "Poll with no mtime change -> NoChange");
        check(lm.swapCount() == swapsAfterLoad, "no reload when the file did not change");

        // Bump the mtime + change the file contents -> the next Poll reloads + swaps exactly once.
        WriteFile(matPath, kValidMat2);
        mtimes[matPath] = 200;
        ReloadStatus s1 = lm.Poll();
        check(s1 == ReloadStatus::Swapped, "Poll after an mtime increase reloads + swaps");
        check(lm.swapCount() == swapsAfterLoad + 1, "exactly one reload for one mtime change");

        // A SECOND Poll with no further change does nothing (the change is consumed once).
        ReloadStatus s2 = lm.Poll();
        check(s2 == ReloadStatus::NoChange, "second Poll (no new change) -> NoChange");
        check(lm.swapCount() == swapsAfterLoad + 1, "no double-fire for a single mtime change");
    }

    // ============================ 4. MALFORMED / INVALID GRAPH DOES NOT SWAP ====================
    {
        // (a) An invalid graph (parses, fails validation) keeps the previous material; the compiler
        //     is NOT reached.
        WriteFile(matPath, kValidMat);
        compileCalls = 0; compileShouldFail = false;
        LiveMaterial lm(matPath, stubCompiler);
        check(lm.LoadInitial(), "initial load ok (invalid-graph case)");
        std::vector<uint32_t> before = lm.activeSpirv();
        int swapsBefore = lm.swapCount();
        int callsBefore = compileCalls;

        WriteFile(matPath, kInvalidGraph);
        ReloadStatus st = lm.Reload();
        check(st == ReloadStatus::Kept, "reload of an INVALID graph is Kept (no swap)");
        check(lm.activeSpirv() == before, "invalid graph: previous SPIR-V kept");
        check(lm.swapCount() == swapsBefore, "invalid graph: swapCount unchanged");
        check(!lm.lastError().empty(), "invalid graph: error surfaced");
        check(compileCalls == callsBefore, "invalid graph: compiler NOT invoked (validation gate)");

        // (b) Malformed JSON: same fail-safe behavior.
        WriteFile(matPath, kMalformed);
        ReloadStatus st2 = lm.Reload();
        check(st2 == ReloadStatus::Kept, "reload of MALFORMED json is Kept (no swap)");
        check(lm.activeSpirv() == before, "malformed json: previous SPIR-V kept");
        check(!lm.lastError().empty(), "malformed json: error surfaced");
    }

    // ============================ 5. INITIAL LOAD FAILURE IS REPORTED ===========================
    {
        // A controller pointed at a missing file: LoadInitial returns false + records an error,
        // active SPIR-V stays empty (the caller treats a FIRST-load failure as fatal).
        compileShouldFail = false;
        LiveMaterial lm(TempPath("does_not_exist.mat.json"), stubCompiler);
        check(!lm.LoadInitial(), "LoadInitial on a missing file returns false");
        check(lm.activeSpirv().empty(), "no active SPIR-V after a failed initial load");
        check(!lm.lastError().empty(), "failed initial load records an error");
    }

    // ============================ 6. SwitchTo a DIFFERENT material ==============================
    {
        WriteFile(matPath, kValidMat);
        WriteFile(matPath2, kValidMat2);
        compileShouldFail = false;
        LiveMaterial lm(matPath, stubCompiler);
        check(lm.LoadInitial(), "initial load (switch case) ok");
        std::vector<uint32_t> first = lm.activeSpirv();
        ReloadStatus st = lm.SwitchTo(matPath2);
        check(st == ReloadStatus::Swapped, "SwitchTo a valid material swaps");
        check(lm.path() == matPath2, "SwitchTo updates the active path");
        check(lm.activeSpirv() != first, "SwitchTo replaced the active SPIR-V (different graph)");
    }

    if (g_fail == 0) { std::printf("runtime_material_test OK\n"); return 0; }
    std::printf("runtime_material_test: %d failures\n", g_fail);
    return 1;
}
