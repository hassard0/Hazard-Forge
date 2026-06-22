// Slice DX4 — unit test for engine/runtime/hot_reload.{h,cpp}: the FileWatcher (injected-StatFn
// change detection) + ApplyReload (the deterministic headless scene reload, reload == cold-load,
// no residue). Pure C++ (hf_core, ASan-eligible) — the watcher is driven by an in-memory fake stat
// (a clock-by-hand map), and the reload uses opaque sentinel resource pointers (scene_io never
// dereferences them; it maps pointer<->name only).
#include "runtime/hot_reload.h"
#include "scene/scene_io.h"
#include "ecs/ecs.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Write `text` to `path` (binary => LF-clean), returning the path for chaining.
static std::string writeTmp(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary);
    f << text;
    return path;
}

int main() {
    HF_TEST_MAIN_INIT();

    // --- Part 1: FileWatcher driven by an in-memory fake StatFn (no real filesystem). ---------------
    {
        // An in-memory mtime map; the StatFn returns the mapped value or -1 (missing) for unknowns.
        std::map<std::string, int64_t> mtimes;
        runtime::StatFn fake = [&mtimes](const std::string& p) -> int64_t {
            auto it = mtimes.find(p);
            return it == mtimes.end() ? -1 : it->second;
        };

        const std::string pA = "watchA";
        const std::string pB = "watchB";
        mtimes[pA] = 100;  // pA present at baseline
        // pB intentionally absent at baseline (missing) -> tests the missing->present "appeared" case.

        runtime::FileWatcher w(fake);
        w.Watch(pA);
        w.Watch(pB);
        check(w.Count() == 2, "watcher tracks 2 paths");

        // First Poll baselines: NO change reported yet (Watch recorded the current mtimes).
        check(w.Poll().empty(), "first Poll() after Watch is empty (baseline)");

        // Bump pA's mtime -> Poll reports exactly pA.
        mtimes[pA] = 200;
        {
            auto changed = w.Poll();
            check(changed.size() == 1 && changed[0] == pA, "bumped mtime -> Poll reports the path");
        }
        // Polling again with no further change is empty (baseline re-current at 200).
        check(w.Poll().empty(), "Poll() with no change is empty");

        // pB transitions missing->present: counts as "appeared" (a change).
        mtimes[pB] = 50;
        {
            auto changed = w.Poll();
            check(changed.size() == 1 && changed[0] == pB, "missing->present counts as changed");
        }

        // A disappearance (present->missing) does NOT count as a change.
        mtimes.erase(pA);
        check(w.Poll().empty(), "present->missing (disappearance) is NOT a change");
        // ...and the later reappearance counts as "appeared" again.
        mtimes[pA] = 300;
        {
            auto changed = w.Poll();
            check(changed.size() == 1 && changed[0] == pA, "reappearance after a disappearance is changed");
        }

        // Re-watching a path re-baselines it (no spurious fire on the next Poll).
        mtimes[pA] = 999;
        w.Watch(pA);  // re-baseline at 999
        check(w.Poll().empty(), "re-Watch re-baselines (no re-fire)");
    }

    // --- Part 2: ApplyReload — reload == cold-load, no residue. -------------------------------------
    {
        // Sentinel named resources (opaque, never dereferenced). A uses 'duck' (A-only); B does not.
        scene::SceneResources res;
        res.AddMesh("cube",   reinterpret_cast<scene::Mesh*>(0x1001));
        res.AddMesh("plane",  reinterpret_cast<scene::Mesh*>(0x1002));
        res.AddMesh("sphere", reinterpret_cast<scene::Mesh*>(0x1003));
        res.AddMesh("duck",   reinterpret_cast<scene::Mesh*>(0x1004));
        res.AddTexture("checker",     reinterpret_cast<rhi::ITexture*>(0x2001));
        res.AddTexture("flat_normal", reinterpret_cast<rhi::ITexture*>(0x2002));

        // A: 3 entities, one of which uses the A-only 'duck' mesh.
        const std::string sceneA =
            "[\n"
            "  {\"mesh\": \"plane\", \"position\": [0, 0, 0]},\n"
            "  {\"mesh\": \"sphere\", \"position\": [1, 0, 0]},\n"
            "  {\"mesh\": \"duck\", \"position\": [-1, 0, 0]}\n"
            "]\n";
        // B: 2 entities, NONE using 'duck' (uses 'cube' instead).
        const std::string sceneB =
            "[\n"
            "  {\"mesh\": \"plane\", \"position\": [0, 0, 0]},\n"
            "  {\"mesh\": \"cube\", \"position\": [2, 0, 0]}\n"
            "]\n";

        std::string pathA = writeTmp("hot_reload_test_A.json", sceneA);
        std::string pathB = writeTmp("hot_reload_test_B.json", sceneB);

        // Load A.
        ecs::Registry reg;
        scene::LoadScene(reg, res, pathA.c_str());
        check(reg.aliveCount() == 3, "A loads 3 entities");
        std::string dumpA = scene::DumpScene(reg, res);
        check(dumpA.find("\"mesh\": \"duck\"") != std::string::npos, "A's dump contains the duck mesh");

        // ApplyReload(B) into the SAME registry.
        runtime::ReloadResult rr = runtime::ApplyReload(reg, res, pathB);
        check(rr.loaded, "ApplyReload(B) loaded");
        check(rr.entityCount == 2, "after reload entityCount == B's count (2)");
        check(reg.aliveCount() == 2, "registry holds exactly B's entities after reload");
        check(rr.equalToColdLoad, "reload == cold-load (DumpScene byte-identical)");

        // No residue: A's A-only 'duck' is GONE; the post-reload dump equals a cold load of B.
        std::string dumpAfter = scene::DumpScene(reg, res);
        check(dumpAfter.find("\"mesh\": \"duck\"") == std::string::npos,
              "no residue: A-only duck mesh absent after reload");

        ecs::Registry coldB;
        scene::LoadScene(coldB, res, pathB.c_str());
        std::string dumpColdB = scene::DumpScene(coldB, res);
        check(dumpAfter == dumpColdB, "post-reload dump == fresh cold-load of B byte-identical");

        // Two ApplyReloads are byte-identical (determinism).
        ecs::Registry reg2;
        scene::LoadScene(reg2, res, pathA.c_str());
        runtime::ReloadResult rr2 = runtime::ApplyReload(reg2, res, pathB);
        check(rr2.entityCount == rr.entityCount && rr2.equalToColdLoad == rr.equalToColdLoad,
              "two ApplyReloads agree on result");
        check(scene::DumpScene(reg2, res) == dumpAfter, "two ApplyReloads byte-identical");

        // A failed reload (unknown mesh) leaves the registry CLEARED (no half-built / no stale world).
        std::string badPath = writeTmp("hot_reload_test_bad.json",
                                       "[{\"mesh\": \"nope\", \"position\": [0,0,0]}]\n");
        ecs::Registry reg3;
        scene::LoadScene(reg3, res, pathA.c_str());  // 3 entities
        runtime::ReloadResult rrBad = runtime::ApplyReload(reg3, res, badPath);
        check(!rrBad.loaded, "ApplyReload of an unknown-mesh scene reports !loaded");
        check(reg3.aliveCount() == 0, "a failed reload leaves the registry CLEARED (no stale world)");

        std::remove(pathA.c_str());
        std::remove(pathB.c_str());
        std::remove(badPath.c_str());
    }

    if (g_fail == 0) { std::printf("hot_reload_test OK\n"); return 0; }
    std::printf("hot_reload_test: %d failures\n", g_fail);
    return 1;
}
