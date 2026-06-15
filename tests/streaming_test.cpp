// Slice BD unit test: distance-based scene/asset streaming (engine/scene/streaming). Pure C++ (no
// GPU): validates radius residency + hysteresis band, the per-frame load BUDGET throttle with
// nearest-first ordering, that a camera oscillating in the hysteresis band does NOT thrash, and that
// a scripted camera path replays bit-identically (deterministic resident set every frame). Clean
// under MSVC /fsanitize=address.
#include "scene/streaming.h"
#include "math/math.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace hf;
using hf::scene::CellState;
using hf::scene::StreamConfig;
using hf::scene::StreamingWorld;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A budget that never throttles (for the pure-residency tests): process every queued cell each frame.
static StreamConfig UnthrottledConfig(float loadR, float unloadR) {
    StreamConfig c;
    c.loadRadius = loadR;
    c.unloadRadius = unloadR;
    c.loadBudgetPerFrame = 1000000;     // effectively infinite
    c.unloadBudgetPerFrame = 1000000;
    return c;
}

int main() {
    using math::Vec3;

    // A fixed 8x8 grid centered on the origin, cells spaced 4 units apart, on the y=0 plane. This is
    // the deterministic world all tests share (the grid is a pure function of N + spacing).
    const int kGridN = 8;
    const float kSpacing = 4.0f;

    // ---------------------------------------------------------------------------------------------
    // 1. Radius residency + hysteresis band (budget = infinity).
    //    Inside loadRadius -> Resident; beyond unloadRadius -> Unloaded; a band cell KEEPS its prior
    //    state (tested both ways: was-resident stays resident, was-unloaded stays unloaded).
    // ---------------------------------------------------------------------------------------------
    {
        // loadRadius 5, unloadRadius 9: a cell at distance ~6.5 sits in the band.
        StreamingWorld world(kGridN, kSpacing, UnthrottledConfig(5.0f, 9.0f));

        // The cell nearest a chosen camera position, and helpers to find cells by distance.
        const Vec3 cam{0.0f, 0.0f, 0.0f};

        // One Update with an infinite budget: every desired transition completes this frame.
        world.Update(cam);

        // Find a cell clearly inside loadRadius, one clearly beyond unloadRadius, and one in the band.
        int insideId = -1, outsideId = -1, bandId = -1;
        for (const auto& c : world.Cells()) {
            float d = math::length(c.center - cam);
            if (d <= 4.0f && insideId < 0) insideId = c.id;            // well inside loadRadius(5)
            if (d > 9.5f && outsideId < 0) outsideId = c.id;           // well beyond unloadRadius(9)
            if (d > 6.0f && d < 8.5f && bandId < 0) bandId = c.id;     // strictly inside the band
        }
        check(insideId >= 0 && outsideId >= 0 && bandId >= 0, "found inside/outside/band cells");

        check(world.StateOf(insideId) == CellState::Resident, "cell inside loadRadius is Resident");
        check(world.StateOf(outsideId) == CellState::Unloaded, "cell beyond unloadRadius is Unloaded");

        // The band cell: after the first Update from cam@origin it was Unloaded (dist > loadRadius),
        // so it should still be Unloaded (the band keeps the prior Unloaded state).
        check(world.StateOf(bandId) == CellState::Unloaded,
              "band cell that was Unloaded stays Unloaded (hysteresis)");

        // Now drive the camera CLOSE to the band cell so it becomes resident, then pull back so the
        // cell falls into the band again: it must STAY resident (the band keeps the prior Resident).
        Vec3 bandCenter{0, 0, 0};
        for (const auto& c : world.Cells()) if (c.id == bandId) bandCenter = c.center;

        world.Update(bandCenter);                       // cam ON the band cell -> Resident
        check(world.StateOf(bandId) == CellState::Resident, "band cell becomes Resident up close");

        // Pull the camera back to a position where the band cell is again in the (load, unload] band.
        Vec3 pullBack = bandCenter - math::normalize(bandCenter - cam) * 7.0f;
        float dband = math::length(bandCenter - pullBack);
        check(dband > 5.0f && dband < 9.0f, "pull-back leaves the band cell in the hysteresis band");
        world.Update(pullBack);
        check(world.StateOf(bandId) == CellState::Resident,
              "band cell that was Resident stays Resident (hysteresis)");
    }

    // ---------------------------------------------------------------------------------------------
    // 2. Budget throttle + nearest-first order.
    //    With loadBudgetPerFrame = K and M > K cells newly in range, exactly K load per frame; after
    //    ceil(M/K) frames all are resident; loads happen NEAREST-first (assert the order).
    // ---------------------------------------------------------------------------------------------
    {
        const int K = 2;
        StreamConfig cfg;
        cfg.loadRadius = 7.0f;
        cfg.unloadRadius = 12.0f;
        cfg.loadBudgetPerFrame = K;
        cfg.unloadBudgetPerFrame = K;

        StreamingWorld world(kGridN, kSpacing, cfg);
        const Vec3 cam{0.0f, 0.0f, 0.0f};

        // M = how many cells are within loadRadius of the camera (the desired-resident set).
        std::vector<int> desired;
        for (const auto& c : world.Cells())
            if (math::length(c.center - cam) <= cfg.loadRadius) desired.push_back(c.id);
        const int M = (int)desired.size();
        check(M > K, "more cells in range than the per-frame budget (a real throttle)");

        // Expected nearest-first load order: desired cells sorted by ascending distance (ties by id).
        std::vector<int> expectedOrder = desired;
        std::sort(expectedOrder.begin(), expectedOrder.end(), [&](int a, int b) {
            Vec3 ca{}, cb{};
            for (const auto& c : world.Cells()) { if (c.id == a) ca = c.center; if (c.id == b) cb = c.center; }
            float da = math::length(ca - cam), db = math::length(cb - cam);
            if (da != db) return da < db;
            return a < b;
        });

        // Step frames, recording the order cells first reach Resident, and the per-frame load count.
        std::vector<int> actualOrder;
        std::vector<bool> seenResident(world.Cells().size(), false);
        const int frames = (M + K - 1) / K;             // ceil(M/K)
        for (int f = 0; f < frames; ++f) {
            int residentBefore = world.Stats().resident;
            world.Update(cam);
            int loadedThisFrame = world.Stats().resident - residentBefore;
            // Each frame loads at most K (the last frame may load fewer if M isn't a multiple of K).
            check(loadedThisFrame <= K, "no more than K cells loaded per frame");
            if (f < frames - 1)
                check(loadedThisFrame == K, "exactly K cells loaded on a non-final frame");
            // Record newly-resident cells (in ascending-id order is fine; the ORDER assertion uses the
            // resident-id set growth, which we capture via ResidentCellIds each frame below).
            for (int id : world.ResidentCellIds()) {
                if (!seenResident[(size_t)id]) { seenResident[(size_t)id] = true; actualOrder.push_back(id); }
            }
        }

        // All M are resident after ceil(M/K) frames.
        check(world.Stats().resident == M, "all M cells resident after ceil(M/K) frames");

        // Nearest-first: within each budget batch the nearest cells load first. Because ResidentCellIds
        // is sorted by id (not distance), we instead assert the per-frame BATCHES are the nearest ones.
        // Re-run with an explicit per-frame capture to assert ordering by distance batch.
        StreamingWorld world2(kGridN, kSpacing, cfg);
        std::vector<int> batchOrder;                    // ids in the order they first became resident
        std::vector<bool> seen2(world2.Cells().size(), false);
        for (int f = 0; f < frames; ++f) {
            world2.Update(cam);
            // Capture the cells that became resident THIS frame, nearest-first within the batch.
            std::vector<int> newlyResident;
            for (int id : world2.ResidentCellIds())
                if (!seen2[(size_t)id]) { seen2[(size_t)id] = true; newlyResident.push_back(id); }
            std::sort(newlyResident.begin(), newlyResident.end(), [&](int a, int b) {
                Vec3 ca{}, cb{};
                for (const auto& c : world2.Cells()) { if (c.id == a) ca = c.center; if (c.id == b) cb = c.center; }
                float da = math::length(ca - cam), db = math::length(cb - cam);
                if (da != db) return da < db;
                return a < b;
            });
            for (int id : newlyResident) batchOrder.push_back(id);
        }
        check(batchOrder == expectedOrder, "loads complete NEAREST-first (ascending distance order)");
    }

    // ---------------------------------------------------------------------------------------------
    // 3. Hysteresis no-thrash: a camera oscillating WITHIN the band does not repeatedly load/unload a
    //    band cell. Pick a cell, move the camera so the cell oscillates between (loadRadius, unloadR],
    //    and assert the cell's load/unload transition count stays bounded (one initial load, then
    //    stable — never a load->unload->load oscillation).
    // ---------------------------------------------------------------------------------------------
    {
        StreamConfig cfg = UnthrottledConfig(5.0f, 9.0f);
        StreamingWorld world(kGridN, kSpacing, cfg);

        // Choose a real grid cell off the origin (the grid x/z centers are multiples of the spacing
        // offset from -half; for N=8, spacing=4 the nearest off-origin cell is at (2,0,2)).
        int cellId = -1; Vec3 cellCenter{};
        for (const auto& c : world.Cells()) {
            if (std::fabs(c.center.x - 2.0f) < 0.1f && std::fabs(c.center.z - 2.0f) < 0.1f) {
                cellId = c.id; cellCenter = c.center;
            }
        }
        check(cellId >= 0, "found the oscillation test cell");
        if (cellId < 0) { std::printf("streaming_test: setup failure\n"); return 1; }

        Vec3 toCell = math::normalize(cellCenter);      // direction origin->cell
        // Two camera positions: "near" leaves the cell at dist ~4 (< loadRadius -> resident); "far"
        // leaves the cell at dist ~7 (in the band -> keep state). Oscillating between them must NOT
        // thrash, because once resident the band keeps it resident.
        Vec3 camNear = cellCenter - toCell * 4.0f;       // dist(cell) = 4  (inside loadRadius)
        Vec3 camFar  = cellCenter - toCell * 7.0f;       // dist(cell) = 7  (in the band)
        check(math::length(cellCenter - camNear) < 5.0f, "near anchor keeps cell inside loadRadius");
        float dfar = math::length(cellCenter - camFar);
        check(dfar > 5.0f && dfar < 9.0f, "far anchor keeps cell in the band");

        world.Update(camNear);                           // load it once
        check(world.StateOf(cellId) == CellState::Resident, "oscillation cell starts Resident");
        int loads = world.LoadEventCount(cellId);
        int unloads = world.UnloadEventCount(cellId);

        for (int i = 0; i < 20; ++i) world.Update((i & 1) ? camFar : camNear);

        check(world.LoadEventCount(cellId) == loads, "no extra loads while oscillating in the band");
        check(world.UnloadEventCount(cellId) == unloads, "no unloads while oscillating in the band");
        check(world.StateOf(cellId) == CellState::Resident, "cell stays Resident through oscillation");
    }

    // ---------------------------------------------------------------------------------------------
    // 4. Determinism: running the same SCRIPTED camera path twice yields identical per-frame
    //    ResidentCellIds (the contract the --stream-shot golden + state assertion rest on).
    // ---------------------------------------------------------------------------------------------
    {
        StreamConfig cfg;
        cfg.loadRadius = 6.0f;
        cfg.unloadRadius = 10.0f;
        cfg.loadBudgetPerFrame = 3;
        cfg.unloadBudgetPerFrame = 3;

        // A scripted path flying diagonally across the grid (NO live input/RNG/clock).
        std::vector<Vec3> path;
        const float half = (kGridN - 1) * 0.5f * kSpacing;
        for (int f = 0; f < 60; ++f) {
            float t = (float)f / 59.0f;
            float x = -half + t * (2.0f * half);
            float z = -half + t * (2.0f * half);
            path.push_back(Vec3{x, 0.0f, z});
        }

        auto runPath = [&]() {
            StreamingWorld w(kGridN, kSpacing, cfg);
            std::vector<std::vector<int>> perFrame;
            for (const Vec3& p : path) { w.Update(p); perFrame.push_back(w.ResidentCellIds()); }
            return perFrame;
        };

        auto run1 = runPath();
        auto run2 = runPath();
        check(run1 == run2, "scripted path replays bit-identical per-frame ResidentCellIds");

        // And the resident set is non-trivial somewhere along the path (the test actually exercised it).
        bool anyResident = false;
        for (const auto& f : run1) if (!f.empty()) anyResident = true;
        check(anyResident, "the scripted path streamed some cells resident");
    }

    if (g_fail == 0) std::printf("streaming_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
