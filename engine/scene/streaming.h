#pragma once
// Slice BD — distance-based scene / asset STREAMING (Phase 4 #7).
//
// PURE CPU above the scene: this module has ZERO RHI / graphics-backend symbols (no vk*/MTL*/mtl::/
// Backend::Metal). It depends ONLY on engine/math + the C++ stdlib, and is compiled into BOTH hf_core
// (ASan-scoped, unit-tested) and hf_engine (the live --stream-shot showcase). A large world is divided
// into a fixed deterministic NxN grid of CELLS; `StreamingWorld::Update(cameraPos)` keeps cells inside
// `loadRadius` Resident, unloads cells beyond the larger `unloadRadius` (HYSTERESIS band in between),
// and processes loads/unloads under a per-frame BUDGET so content streams in OVER frames — NEAREST
// first, deterministically.
//
// Determinism: the resident set is a pure function of (camera position, radii, budget, prior state).
// Driven by a fixed SCRIPTED camera path (a std::vector<Vec3>, no live input/RNG/clock) + a fixed
// budget, the load/unload event sequence and the resident set at every frame are bit-stable: same
// path + same build => identical. The "load" here is the SYNCHRONOUS construction of a cell's
// procedural renderables — the MVP models the budget/over-frames behavior, NOT async disk I/O. Async
// file streaming is a future slice.
#include <vector>

#include "math/math.h"

namespace hf::scene {

// The lifecycle of a single cell. Unloaded <-> (Loading -> Resident) <-> (Unloading -> Unloaded).
// Loading/Unloading are the in-flight states a cell sits in while it waits its turn under the budget.
enum class CellState { Unloaded, Loading, Resident, Unloading };

// One procedural renderable inside a cell's content. PURE DATA — no mesh/texture POINTERS, no RHI
// types: the streaming layer describes WHAT to draw (a primitive kind, a model matrix, a material),
// and the renderer (the --stream-shot showcase) maps `kind`/`colorIndex` to real GPU meshes/textures.
// This keeps the streaming module backend-free and the unit test GPU-free.
struct CellRenderable {
    enum class Kind { Cube, Sphere };
    Kind kind = Kind::Cube;
    math::Mat4 model = math::Mat4::Identity();
    float metallic = 0.0f;
    float roughness = 0.5f;
    int colorIndex = 0;            // selects a tint in the showcase's small fixed palette
};

// A single world cell: a stable id, a fixed center, the current streaming state, and (once Resident)
// its built procedural renderables.
struct Cell {
    int id = -1;
    math::Vec3 center{0, 0, 0};
    CellState state = CellState::Unloaded;
    std::vector<CellRenderable> renderables;   // populated on load (Loading->Resident), freed on unload
    int loadEvents = 0;                         // # of Loading->Resident completions (no-thrash test)
    int unloadEvents = 0;                       // # of Unloading->Unloaded completions
};

// Streaming policy knobs. unloadRadius MUST be > loadRadius (the difference is the hysteresis band
// that prevents thrashing). The budgets cap how many cells finish loading / unloading per frame.
struct StreamConfig {
    float loadRadius = 6.0f;
    float unloadRadius = 10.0f;     // > loadRadius (hysteresis)
    int loadBudgetPerFrame = 4;
    int unloadBudgetPerFrame = 4;
};

// Aggregate per-frame counts (for the showcase state line + the budget-throttle test).
struct StreamStats {
    int resident = 0;
    int loading = 0;
    int unloading = 0;
    int totalCells = 0;
};

// Owns the full NxN cell grid + each cell's state. The grid is a pure function of (gridN, spacing):
// cells are laid out on the y=0 plane, centered on the origin, ids assigned row-major (z outer, x
// inner) so the layout is identical run-to-run and across backends.
class StreamingWorld {
public:
    StreamingWorld(int gridN, float spacing, const StreamConfig& cfg);

    // Advance one streaming frame for the given camera position:
    //   1. compute desired residency (dist<=loadRadius -> resident; dist>unloadRadius -> unloaded;
    //      band cells keep their current state — hysteresis),
    //   2. enqueue to-load (Unloaded->Loading) + to-unload (Resident->Unloading) by ASCENDING
    //      distance (nearest first),
    //   3. process up to loadBudgetPerFrame loads (Loading->Resident, building renderables) and
    //      unloadBudgetPerFrame unloads (Unloading->Unloaded, freeing them) this frame.
    void Update(const math::Vec3& cameraPos);

    // --- Accessors -------------------------------------------------------------------------------
    const std::vector<Cell>& Cells() const { return cells_; }
    CellState StateOf(int id) const { return cells_[(size_t)id].state; }

    // Ids of all Resident cells, ascending (the assertable streaming state).
    std::vector<int> ResidentCellIds() const;
    // Every Resident cell's renderables flattened (the renderer's draw list).
    std::vector<CellRenderable> ResidentRenderables() const;

    StreamStats Stats() const;

    // Per-cell transition counters (the no-thrash hysteresis test reads these): a load event is each
    // Loading->Resident completion; an unload event is each Unloading->Unloaded completion.
    int LoadEventCount(int id) const { return cells_[(size_t)id].loadEvents; }
    int UnloadEventCount(int id) const { return cells_[(size_t)id].unloadEvents; }

private:
    // Build a cell's procedural renderables deterministically from its id + center (a small fixed
    // cluster of primitives — the synchronous "load").
    static std::vector<CellRenderable> BuildCellContent(const Cell& cell);

    int gridN_ = 0;
    float spacing_ = 0.0f;
    StreamConfig cfg_;
    std::vector<Cell> cells_;
};

}  // namespace hf::scene
