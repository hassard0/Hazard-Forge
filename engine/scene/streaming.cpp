// Slice BD — distance-based scene/asset streaming (see streaming.h). Pure C++ (engine/math + stdlib
// only); NO RHI or graphics-backend symbols. Deterministic: no RNG, no clock, no live input.
#include "scene/streaming.h"

#include <algorithm>
#include <cmath>

namespace hf::scene {

StreamingWorld::StreamingWorld(int gridN, float spacing, const StreamConfig& cfg)
    : gridN_(gridN), spacing_(spacing), cfg_(cfg) {
    // Build the fixed NxN grid on the y=0 plane, centered on the origin. Ids are row-major (z outer,
    // x inner) so the layout is identical run-to-run and across backends.
    const float half = (gridN_ - 1) * 0.5f * spacing_;
    cells_.reserve((size_t)gridN_ * (size_t)gridN_);
    int id = 0;
    for (int gz = 0; gz < gridN_; ++gz) {
        for (int gx = 0; gx < gridN_; ++gx) {
            Cell c;
            c.id = id++;
            c.center = math::Vec3{-half + gx * spacing_, 0.0f, -half + gz * spacing_};
            c.state = CellState::Unloaded;
            cells_.push_back(c);
        }
    }
}

void StreamingWorld::Update(const math::Vec3& cameraPos) {
    // (1) Compute the desired residency and resolve in-flight states against it. A cell's DESIRE is a
    //     pure function of its distance to the camera + the prior state (hysteresis band keeps state):
    //       dist <= loadRadius            -> wants Resident
    //       dist >  unloadRadius          -> wants Unloaded
    //       loadRadius < dist <= unloadR  -> band: keep the current state.
    //     We collect the cells that need to be ENQUEUED to-load / to-unload (with their distance for
    //     the nearest-first ordering). A cell already in the in-flight state it wants stays queued; a
    //     cell whose desire reversed mid-flight is cancelled back to a stable state (deterministic).
    struct Pending { int id; float dist; };
    std::vector<Pending> toLoad;     // cells that should end up Resident but aren't yet
    std::vector<Pending> toUnload;   // cells that should end up Unloaded but aren't yet

    for (auto& c : cells_) {
        const float d = math::length(c.center - cameraPos);
        const bool inLoad   = d <= cfg_.loadRadius;
        const bool outUnload = d > cfg_.unloadRadius;

        if (inLoad) {
            // Wants Resident. Cancel any in-flight unload; queue a load if not already resident.
            if (c.state == CellState::Unloading) c.state = CellState::Resident;  // cancel unload
            if (c.state == CellState::Unloaded) c.state = CellState::Loading;
            if (c.state == CellState::Loading) toLoad.push_back({c.id, d});
        } else if (outUnload) {
            // Wants Unloaded. Cancel any in-flight load; queue an unload if currently resident.
            if (c.state == CellState::Loading) c.state = CellState::Unloaded;    // cancel load
            if (c.state == CellState::Resident) c.state = CellState::Unloading;
            if (c.state == CellState::Unloading) toUnload.push_back({c.id, d});
        } else {
            // Hysteresis band: keep the current state, but a cell that was mid-flight keeps streaming
            // to its in-flight target (so a load that started before entering the band still finishes,
            // and likewise an unload) — this is the prior state being honored, not a new transition.
            if (c.state == CellState::Loading) toLoad.push_back({c.id, d});
            if (c.state == CellState::Unloading) toUnload.push_back({c.id, d});
        }
    }

    // (2) Order both queues NEAREST-first (ascending distance; ties by ascending id for stability).
    auto byDistThenId = [](const Pending& a, const Pending& b) {
        if (a.dist != b.dist) return a.dist < b.dist;
        return a.id < b.id;
    };
    std::sort(toLoad.begin(), toLoad.end(), byDistThenId);
    std::sort(toUnload.begin(), toUnload.end(), byDistThenId);

    // (3) Process up to the per-frame budget. A load COMPLETES (Loading->Resident) by synchronously
    //     building the cell's procedural renderables; an unload COMPLETES (Unloading->Unloaded) by
    //     freeing them. Cells beyond the budget stay in their in-flight state for a later frame.
    int loadBudget = cfg_.loadBudgetPerFrame;
    for (const Pending& p : toLoad) {
        if (loadBudget <= 0) break;
        Cell& c = cells_[(size_t)p.id];
        c.renderables = BuildCellContent(c);
        c.state = CellState::Resident;
        ++c.loadEvents;
        --loadBudget;
    }

    int unloadBudget = cfg_.unloadBudgetPerFrame;
    for (const Pending& p : toUnload) {
        if (unloadBudget <= 0) break;
        Cell& c = cells_[(size_t)p.id];
        c.renderables.clear();
        c.renderables.shrink_to_fit();
        c.state = CellState::Unloaded;
        ++c.unloadEvents;
        --unloadBudget;
    }
}

std::vector<int> StreamingWorld::ResidentCellIds() const {
    std::vector<int> ids;
    for (const auto& c : cells_)
        if (c.state == CellState::Resident) ids.push_back(c.id);
    // cells_ is already in ascending-id order, so ids is sorted; sort defensively anyway.
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::vector<CellRenderable> StreamingWorld::ResidentRenderables() const {
    std::vector<CellRenderable> out;
    for (const auto& c : cells_)
        if (c.state == CellState::Resident)
            out.insert(out.end(), c.renderables.begin(), c.renderables.end());
    return out;
}

StreamStats StreamingWorld::Stats() const {
    StreamStats s;
    s.totalCells = (int)cells_.size();
    for (const auto& c : cells_) {
        switch (c.state) {
            case CellState::Resident:  ++s.resident;  break;
            case CellState::Loading:   ++s.loading;   break;
            case CellState::Unloading: ++s.unloading; break;
            case CellState::Unloaded:  break;
        }
    }
    return s;
}

std::vector<CellRenderable> StreamingWorld::BuildCellContent(const Cell& cell) {
    // A small deterministic procedural cluster, generated purely from the cell's id + center: a center
    // cube plus four corner spheres at fixed offsets. Material params + tint are derived from the id so
    // neighboring cells read distinctly, with NO RNG/clock — same id => same content.
    std::vector<CellRenderable> out;

    const math::Vec3 c = cell.center;
    const int idx = cell.id;

    // Center tower cube, resting on the ground (a taller landmark so a resident cell reads clearly
    // against the bare ground of unloaded cells).
    {
        CellRenderable r;
        r.kind = CellRenderable::Kind::Cube;
        r.model = math::Mat4::Translate(math::Vec3{c.x, 0.9f, c.z}) *
                  math::Mat4::Scale(math::Vec3{0.9f, 0.9f, 0.9f});
        r.metallic = 0.1f;
        r.roughness = 0.6f;
        r.colorIndex = idx % 4;          // a tint in the showcase's 4-color palette
        out.push_back(r);
    }

    // Four corner spheres at fixed offsets (resting on the ground).
    const float off = 1.3f;
    const float dx[4] = {-off, off, -off, off};
    const float dz[4] = {-off, -off, off, off};
    for (int k = 0; k < 4; ++k) {
        CellRenderable r;
        r.kind = CellRenderable::Kind::Sphere;
        r.model = math::Mat4::Translate(math::Vec3{c.x + dx[k], 0.5f, c.z + dz[k]}) *
                  math::Mat4::Scale(math::Vec3{0.5f, 0.5f, 0.5f});
        r.metallic = 0.2f;
        r.roughness = 0.4f;
        r.colorIndex = (idx + k + 1) % 4;
        out.push_back(r);
    }

    return out;
}

}  // namespace hf::scene
