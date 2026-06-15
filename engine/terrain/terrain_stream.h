#pragma once
// Slice BJ — Terrain-streaming LOD integration (Phase 4 #12).
//
// The open-world capstone: ties Slice BD (distance-based streaming residency) + Slice BF (procedural
// heightmap terrain) into a streamed, LOD-selected terrain field. A fixed T x T grid of square terrain
// TILES covers the world; each tile (i,j) covers a `tileSize x tileSize` XZ region at a fixed world
// offset and is meshed by terrain::BuildTerrainTile over THAT tile's region using the GLOBAL Height(x,z)
// — so adjacent tiles meet seamlessly at equal LOD (shared-edge vertices are bit-identical). Each
// resident tile's mesh resolution is chosen by a discrete distance band (LOD0 near / LOD1 mid / LOD2
// far) with hysteresis, and the resident set is governed by the BD load/unload-radius + hysteresis +
// per-frame-budget policy.
//
// PURE CPU above the scene: ZERO RHI / graphics-backend symbols (no vk*/MTL*/mtl::/Backend::Metal). It
// depends ONLY on engine/math + engine/terrain/heightmap + the C++ stdlib, and is compiled into BOTH
// hf_core (ASan-scoped, unit-tested) and hf_engine (the live --terrain-stream-shot showcase) AND the
// standalone metal_headless target (so the tile meshes are bit-identical cross-backend).
//
// Determinism: the resident tile set AND each tile's LOD are pure functions of (camera position, radii,
// budget, distance->LOD bands, prior state). Driven by a fixed SCRIPTED camera path (a std::vector<Vec3>,
// no live input/RNG/clock), the per-frame {resident tiles + their LODs} are bit-stable run-to-run and
// across backends. LOD is a discrete <= band function with hysteresis (don't coarsen until past the band
// edge by a margin) so there is no float-threshold flicker. Tiles share the global Height => no per-tile
// randomness.
#include <vector>

#include "math/math.h"
#include "terrain/heightmap.h"

namespace hf::terrain {

// --- LOD model ------------------------------------------------------------------------------------
// Three discrete levels of detail. The per-LOD tile mesh resolution `n` (vertices per side) is fixed:
//
//   LOD0 (near):  n = 96   -> 96*96   =  9216 verts, (96-1)^2*6 = 54150 indices = 18050 tris
//   LOD1 (mid):   n = 48   -> 48*48   =  2304 verts, (48-1)^2*6 = 13254 indices =  4418 tris
//   LOD2 (far):   n = 24   -> 24*24   =   576 verts, (24-1)^2*6 =  3174 indices =  1058 tris
//
// (verts == n*n, indices == (n-1)*(n-1)*6, tris == indices/3.) The counts are pinned by the unit test.
constexpr int kLodCount = 3;
constexpr int kLodN[kLodCount] = {96, 48, 24};

inline int LodVertexCount(int lod) { int n = kLodN[lod]; return n * n; }
inline int LodIndexCount(int lod)  { int n = kLodN[lod]; return (n - 1) * (n - 1) * 6; }

// --- LOD distance bands (with hysteresis) ---------------------------------------------------------
// A tile's LOD is selected from the distance between its CENTER and the camera. The bands are <=
// (LodFor): dist <= bandNear -> LOD0; dist <= bandMid -> LOD1; else LOD2. Hysteresis prevents a tile
// from COARSENING (moving to a higher LOD index) the instant it crosses a band edge: it only coarsens
// once it is past the edge by `hysteresis` world units. Refining (toward LOD0) happens immediately at
// the <= band (a tile getting closer should sharpen without lag). This keeps the scripted path stable
// and flicker-free.
struct LodBands {
    float bandNear = 0.0f;     // dist <= bandNear -> LOD0
    float bandMid = 0.0f;      // dist <= bandMid  -> LOD1, else LOD2
    float hysteresis = 0.0f;   // coarsening margin past the band edge
};

// The discrete band LOD for a distance, IGNORING hysteresis (the pure band function: <= bands).
int LodFor(float dist, const LodBands& bands);

// The hysteresis-aware LOD update: given the tile's PRIOR lod + its current center distance, return its
// next lod. Refines immediately when the band wants a sharper (lower-index) LOD; only coarsens once the
// distance is past the relevant band edge by bands.hysteresis. Deterministic, pure.
int LodNext(int priorLod, float dist, const LodBands& bands);

// --- Tile grid ------------------------------------------------------------------------------------
// A fixed T x T grid of square tiles, the whole grid CENTERED on the origin. Tile (i,j) — i is the X
// column, j the Z row — covers [worldMinX, worldMinX+tileSize] x [worldMinZ, worldMinZ+tileSize] where
//   worldMinX = -T*tileSize/2 + i*tileSize,   worldMinZ = -T*tileSize/2 + j*tileSize.
// Ids are row-major (z outer, x inner): id = j*T + i. The layout is a pure function of (T, tileSize),
// so it is identical run-to-run and across backends.
struct TileId {
    int i = 0;   // X column
    int j = 0;   // Z row
    int id = 0;  // j*T + i
};

class TileGrid {
public:
    TileGrid(int T, float tileSize) : t_(T), tileSize_(tileSize) {}

    int Dim() const { return t_; }
    float TileSize() const { return tileSize_; }
    int Count() const { return t_ * t_; }

    int IdOf(int i, int j) const { return j * t_ + i; }
    TileId TileAt(int i, int j) const { return TileId{i, j, IdOf(i, j)}; }

    // The world-space MIN corner (smallest X,Z) of tile (i,j).
    math::Vec3 TileWorldMin(int i, int j) const {
        const float origin = -t_ * tileSize_ * 0.5f;
        return math::Vec3{origin + i * tileSize_, 0.0f, origin + j * tileSize_};
    }
    // The world-space CENTER of tile (i,j) (XZ; y=0).
    math::Vec3 TileCenter(int i, int j) const {
        math::Vec3 m = TileWorldMin(i, j);
        return math::Vec3{m.x + tileSize_ * 0.5f, 0.0f, m.z + tileSize_ * 0.5f};
    }

    // Build tile (i,j)'s mesh at the given LOD over its world region (samples GLOBAL Height).
    TerrainMesh BuildTileMesh(int i, int j, int lod, float heightScale) const {
        math::Vec3 m = TileWorldMin(i, j);
        return BuildTerrainTile(m.x, m.z, tileSize_, kLodN[lod], heightScale);
    }

private:
    int t_ = 0;
    float tileSize_ = 0.0f;
};

// --- Residency policy knobs (BD-style) ------------------------------------------------------------
// unloadRadius MUST be > loadRadius (the difference is the residency hysteresis band). The budgets cap
// how many tiles finish loading / unloading per frame (so the field streams in over frames).
struct TerrainStreamConfig {
    float loadRadius = 0.0f;
    float unloadRadius = 0.0f;       // > loadRadius (residency hysteresis)
    int loadBudgetPerFrame = 0;
    int unloadBudgetPerFrame = 0;
    float heightScale = 1.0f;
    LodBands bands;
};

// One tile's streaming lifecycle state (mirrors BD's CellState).
enum class TileState { Unloaded, Loading, Resident, Unloading };

// A resident tile exposed to the renderer + the state assertion: its grid id, its current LOD, and
// (built on load / when its LOD changes) its mesh.
struct ResidentTile {
    TileId tile;
    int lod = 0;
    const TerrainMesh* mesh = nullptr;  // points into the owning world's tile store (stable while resident)
};

// Aggregate per-frame counts (for the showcase state line + tests).
struct TerrainStreamStats {
    int resident = 0;
    int loading = 0;
    int byLod[kLodCount] = {0, 0, 0};
    int total = 0;
};

// Owns the T x T tile grid + each tile's residency state + LOD + built mesh. Update(cameraPos) runs the
// BD residency policy (load/unload by radius + per-frame budget, nearest-first) and the LOD selection
// (each resident tile's LOD from its center distance, with hysteresis). Pure CPU.
class TerrainStreamWorld {
public:
    TerrainStreamWorld(int T, float tileSize, const TerrainStreamConfig& cfg);

    // Advance one streaming frame for the camera position:
    //   1. residency: tiles inside loadRadius want Resident; beyond unloadRadius want Unloaded; the
    //      band keeps prior state (hysteresis). Enqueue to-load / to-unload nearest-first.
    //   2. budget: complete up to loadBudgetPerFrame loads (build the tile mesh at its selected LOD)
    //      and unloadBudgetPerFrame unloads (free the mesh) this frame.
    //   3. LOD: for every RESIDENT tile, update its LOD from its center distance (hysteresis). If a
    //      resident tile's LOD changed, REBUILD its mesh at the new LOD.
    void Update(const math::Vec3& cameraPos);

    const TileGrid& Grid() const { return grid_; }

    // Resident tiles (id-ascending) with their LOD + mesh pointer — the deterministic assertable state.
    std::vector<ResidentTile> ResidentTiles() const;

    TileState StateOf(int id) const { return tiles_[(size_t)id].state; }
    int LodOf(int id) const { return tiles_[(size_t)id].lod; }
    // Per-tile (re)build counters (a tile rebuilds on initial load + on each LOD change).
    int BuildCount(int id) const { return tiles_[(size_t)id].buildCount; }

    TerrainStreamStats Stats() const;

private:
    struct Tile {
        TileId tile;
        math::Vec3 center{0, 0, 0};
        TileState state = TileState::Unloaded;
        int lod = kLodCount - 1;   // start coarsest; LodNext refines on the first resident update
        TerrainMesh mesh;          // populated on load / LOD change, cleared on unload
        int buildCount = 0;        // # of mesh (re)builds (initial load + LOD changes)
    };

    void BuildTileMesh(Tile& tl);  // builds tl.mesh at tl.lod (++buildCount)

    TileGrid grid_;
    TerrainStreamConfig cfg_;
    std::vector<Tile> tiles_;
};

}  // namespace hf::terrain
