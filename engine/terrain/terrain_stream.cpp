// Slice BJ — terrain-streaming LOD integration (see terrain_stream.h). Pure C++ (engine/math +
// engine/terrain/heightmap + stdlib only); NO RHI or graphics-backend symbols. Deterministic: no RNG,
// no clock, no live input. Reuses the BD residency policy shape + the BF heightmap mesh generator.
#include "terrain/terrain_stream.h"

#include <algorithm>
#include <cmath>

namespace hf::terrain {

int LodFor(float dist, const LodBands& bands) {
    if (dist <= bands.bandNear) return 0;
    if (dist <= bands.bandMid) return 1;
    return 2;
}

int LodNext(int priorLod, float dist, const LodBands& bands) {
    // The band the distance falls in right now (ignoring hysteresis).
    const int want = LodFor(dist, bands);
    // Refining (want is sharper = smaller index) happens immediately.
    if (want <= priorLod) return want;
    // Coarsening (want is a larger index): only step once we are PAST the relevant band edge by the
    // hysteresis margin, so a tile hovering on a band boundary does not flicker. We coarsen at most to
    // `want`, one consistent target; the relevant edge is the band edge BELOW `want`.
    //   want==1 means we left the near band -> edge is bandNear; coarsen only if dist > bandNear + h.
    //   want==2 means we left the mid band  -> edge is bandMid;  coarsen only if dist > bandMid  + h.
    const float edge = (want == 1) ? bands.bandNear : bands.bandMid;
    if (dist > edge + bands.hysteresis) return want;
    return priorLod;  // stay (within the hysteresis margin past the edge)
}

TerrainStreamWorld::TerrainStreamWorld(int T, float tileSize, const TerrainStreamConfig& cfg)
    : grid_(T, tileSize), cfg_(cfg) {
    tiles_.reserve((size_t)grid_.Count());
    for (int j = 0; j < T; ++j) {
        for (int i = 0; i < T; ++i) {
            Tile tl;
            tl.tile = grid_.TileAt(i, j);
            tl.center = grid_.TileCenter(i, j);
            tl.state = TileState::Unloaded;
            tl.lod = kLodCount - 1;   // coarsest until the first resident LOD update refines it
            tiles_.push_back(std::move(tl));
        }
    }
}

void TerrainStreamWorld::BuildTileMesh(Tile& tl) {
    tl.mesh = grid_.BuildTileMesh(tl.tile.i, tl.tile.j, tl.lod, cfg_.heightScale);
    ++tl.buildCount;
}

void TerrainStreamWorld::Update(const math::Vec3& cameraPos) {
    // (1) Desired residency + enqueue (nearest-first), mirroring the BD StreamingWorld policy.
    struct Pending { int id; float dist; };
    std::vector<Pending> toLoad, toUnload;

    for (auto& tl : tiles_) {
        const float d = math::length(tl.center - cameraPos);
        const bool inLoad = d <= cfg_.loadRadius;
        const bool outUnload = d > cfg_.unloadRadius;

        if (inLoad) {
            if (tl.state == TileState::Unloading) tl.state = TileState::Resident;  // cancel unload
            if (tl.state == TileState::Unloaded) tl.state = TileState::Loading;
            if (tl.state == TileState::Loading) toLoad.push_back({tl.tile.id, d});
        } else if (outUnload) {
            if (tl.state == TileState::Loading) tl.state = TileState::Unloaded;    // cancel load
            if (tl.state == TileState::Resident) tl.state = TileState::Unloading;
            if (tl.state == TileState::Unloading) toUnload.push_back({tl.tile.id, d});
        } else {
            // Hysteresis band: keep prior state, but in-flight tiles keep streaming to their target.
            if (tl.state == TileState::Loading) toLoad.push_back({tl.tile.id, d});
            if (tl.state == TileState::Unloading) toUnload.push_back({tl.tile.id, d});
        }
    }

    auto byDistThenId = [](const Pending& a, const Pending& b) {
        if (a.dist != b.dist) return a.dist < b.dist;
        return a.id < b.id;
    };
    std::sort(toLoad.begin(), toLoad.end(), byDistThenId);
    std::sort(toUnload.begin(), toUnload.end(), byDistThenId);

    // (2) Process under the per-frame budget. A load COMPLETES by selecting the tile's LOD from its
    //     center distance (fresh — prior lod is the coarsest default or its last resident lod) and
    //     building the mesh at that LOD. An unload frees the mesh.
    int loadBudget = cfg_.loadBudgetPerFrame;
    for (const Pending& p : toLoad) {
        if (loadBudget <= 0) break;
        Tile& tl = tiles_[(size_t)p.id];
        tl.lod = LodNext(tl.lod, p.dist, cfg_.bands);
        BuildTileMesh(tl);
        tl.state = TileState::Resident;
        --loadBudget;
    }

    int unloadBudget = cfg_.unloadBudgetPerFrame;
    for (const Pending& p : toUnload) {
        if (unloadBudget <= 0) break;
        Tile& tl = tiles_[(size_t)p.id];
        tl.mesh = TerrainMesh{};
        tl.state = TileState::Unloaded;
        --unloadBudget;
    }

    // (3) LOD selection for every RESIDENT tile (hysteresis). A LOD change rebuilds the tile mesh.
    for (auto& tl : tiles_) {
        if (tl.state != TileState::Resident) continue;
        const float d = math::length(tl.center - cameraPos);
        const int nextLod = LodNext(tl.lod, d, cfg_.bands);
        if (nextLod != tl.lod) {
            tl.lod = nextLod;
            BuildTileMesh(tl);
        }
    }
}

std::vector<ResidentTile> TerrainStreamWorld::ResidentTiles() const {
    std::vector<ResidentTile> out;
    // tiles_ is already in ascending-id (row-major) order.
    for (const auto& tl : tiles_) {
        if (tl.state != TileState::Resident) continue;
        ResidentTile rt;
        rt.tile = tl.tile;
        rt.lod = tl.lod;
        rt.mesh = &tl.mesh;
        out.push_back(rt);
    }
    return out;
}

TerrainStreamStats TerrainStreamWorld::Stats() const {
    TerrainStreamStats s;
    s.total = (int)tiles_.size();
    for (const auto& tl : tiles_) {
        switch (tl.state) {
            case TileState::Resident:
                ++s.resident;
                if (tl.lod >= 0 && tl.lod < kLodCount) ++s.byLod[tl.lod];
                break;
            case TileState::Loading:   ++s.loading; break;
            case TileState::Unloading: break;
            case TileState::Unloaded:  break;
        }
    }
    return s;
}

}  // namespace hf::terrain
