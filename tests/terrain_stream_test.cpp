// Slice BJ unit test: terrain-streaming LOD integration (engine/terrain/terrain_stream). Pure C++ (no
// GPU). Validates:
//   * Tile coverage: tile (i,j) covers the expected world region; a vertex on the SHARED EDGE of two
//     adjacent tiles has the SAME world position + height (seamlessness at equal LOD).
//   * LOD selection: LodFor returns the right band for sample distances; LodNext hysteresis prevents a
//     tile from coarsening within the band after it refined.
//   * Residency + LOD determinism: running a scripted camera path TWICE yields identical per-frame
//     {resident tile ids + their LODs}.
//   * Per-LOD mesh counts: each LOD's tile mesh has the documented vertex/index counts; the BuildTerrainTile
//     mesh structure (index range, count) is well-formed.
// Clean under MSVC /fsanitize=address.
#include "terrain/terrain_stream.h"
#include "terrain/heightmap.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
using hf::terrain::LodBands;
using hf::terrain::TerrainStreamConfig;
using hf::terrain::TerrainStreamWorld;
using hf::terrain::TileGrid;
using hf::terrain::TileState;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool feq(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

int main() {
    HF_TEST_MAIN_INIT();
    using math::Vec3;

    // The fixed grid all tests share: 6x6 = 36 square tiles, each 16x16, centered on the origin.
    const int   kT = 6;
    const float kTileSize = 16.0f;
    const float kHeightScale = 2.0f;
    // The whole grid spans [-48, 48] on X and Z (= -T*tileSize/2 .. +T*tileSize/2).

    // -------------------------------------------------------------------------------------------
    // 1. Tile coverage + grid layout. Tile (i,j) covers [worldMinX, worldMinX+tileSize] x
    //    [worldMinZ, worldMinZ+tileSize] with worldMinX = -T*tileSize/2 + i*tileSize, etc. Ids are
    //    row-major (id = j*T + i). Centers are worldMin + tileSize/2.
    // -------------------------------------------------------------------------------------------
    {
        TileGrid grid(kT, kTileSize);
        check(grid.Dim() == kT, "grid dim == T");
        check(grid.Count() == kT * kT, "grid count == T*T (36)");
        const float origin = -kT * kTileSize * 0.5f;  // -48
        check(feq(origin, -48.0f), "grid origin (min corner) == -48");

        // Corner tile (0,0): min corner at (-48,-48); center at (-40,-40).
        Vec3 m00 = grid.TileWorldMin(0, 0);
        check(feq(m00.x, -48.0f) && feq(m00.z, -48.0f), "tile (0,0) world-min == (-48,-48)");
        Vec3 c00 = grid.TileCenter(0, 0);
        check(feq(c00.x, -40.0f) && feq(c00.z, -40.0f), "tile (0,0) center == (-40,-40)");
        check(grid.IdOf(0, 0) == 0, "tile (0,0) id == 0");

        // Tile (3,2): i=3 col, j=2 row. min = (-48+48, -48+32) = (0,-16); center (8,-8); id = 2*6+3=15.
        Vec3 m32 = grid.TileWorldMin(3, 2);
        check(feq(m32.x, 0.0f) && feq(m32.z, -16.0f), "tile (3,2) world-min == (0,-16)");
        Vec3 c32 = grid.TileCenter(3, 2);
        check(feq(c32.x, 8.0f) && feq(c32.z, -8.0f), "tile (3,2) center == (8,-8)");
        check(grid.IdOf(3, 2) == 15, "tile (3,2) id == 15 (row-major j*T+i)");

        // Coverage: every world point inside tile (i,j) maps back to (i,j). Sample tile (3,2)'s center.
        int ci = (int)std::floor((c32.x - origin) / kTileSize);
        int cj = (int)std::floor((c32.z - origin) / kTileSize);
        check(ci == 3 && cj == 2, "tile (3,2) center falls inside the (3,2) cell");
    }

    // -------------------------------------------------------------------------------------------
    // 2. Seamlessness: a vertex on the SHARED EDGE of two horizontally-adjacent tiles, at the SAME
    //    LOD, has the SAME world position + height. Tile (i,j) right edge (ix=n-1) coincides with tile
    //    (i+1,j) left edge (ix=0). Because BuildTerrainTile samples the GLOBAL Height(x,z) at absolute
    //    world coords, those vertices are bit-identical -> the tiles meet with no crack at equal LOD.
    // -------------------------------------------------------------------------------------------
    {
        TileGrid grid(kT, kTileSize);
        const int lod = 0;                 // equal LOD => seamless
        const int n = terrain::kLodN[lod]; // 96
        // Two adjacent tiles in the same row: (2,3) and (3,3).
        terrain::TerrainMesh left  = grid.BuildTileMesh(2, 3, lod, kHeightScale);
        terrain::TerrainMesh right = grid.BuildTileMesh(3, 3, lod, kHeightScale);
        check((int)left.verts.size() == n * n, "tile mesh has n*n verts");

        // For each row iz, left's right-edge vertex (ix=n-1) == right's left-edge vertex (ix=0).
        bool allMatch = true;
        for (int iz = 0; iz < n; ++iz) {
            const scene::Vertex& lv = left.verts[(size_t)(iz * n + (n - 1))];
            const scene::Vertex& rv = right.verts[(size_t)(iz * n + 0)];
            if (!(feq(lv.pos[0], rv.pos[0], 1e-5f) && feq(lv.pos[1], rv.pos[1], 1e-5f) &&
                  feq(lv.pos[2], rv.pos[2], 1e-5f))) {
                allMatch = false;
                break;
            }
        }
        check(allMatch, "shared edge verts of adjacent equal-LOD tiles are bit-identical (seamless)");

        // The shared edge X is exactly the boundary world coordinate (tile (2,3) max-X = tile (3,3) min-X).
        float sharedX = grid.TileWorldMin(2, 3).x + kTileSize;
        check(feq(left.verts[(size_t)(0 * n + (n - 1))].pos[0], sharedX, 1e-4f),
              "shared edge X == the tile boundary world X");
        check(feq(right.verts[(size_t)0].pos[0], sharedX, 1e-4f),
              "right tile left edge X == the same boundary world X");

        // And the height there equals the GLOBAL Height(x,z) at that world coordinate.
        const scene::Vertex& probe = left.verts[(size_t)(5 * n + (n - 1))];
        float wantY = terrain::Height(probe.pos[0], probe.pos[2]) * kHeightScale;
        check(feq(probe.pos[1], wantY, 1e-4f), "edge vertex Y == global Height(x,z)*scale");
    }

    // -------------------------------------------------------------------------------------------
    // 3. LOD selection bands (LodFor) + hysteresis (LodNext).
    // -------------------------------------------------------------------------------------------
    {
        LodBands bands; bands.bandNear = 18.0f; bands.bandMid = 40.0f; bands.hysteresis = 4.0f;

        // Pure band function (<= bands).
        check(terrain::LodFor(0.0f, bands) == 0, "dist 0 -> LOD0");
        check(terrain::LodFor(18.0f, bands) == 0, "dist == bandNear -> LOD0 (<= band)");
        check(terrain::LodFor(18.01f, bands) == 1, "dist just past bandNear -> LOD1");
        check(terrain::LodFor(40.0f, bands) == 1, "dist == bandMid -> LOD1 (<= band)");
        check(terrain::LodFor(40.01f, bands) == 2, "dist just past bandMid -> LOD2");
        check(terrain::LodFor(100.0f, bands) == 2, "far dist -> LOD2");

        // Refining (toward LOD0) is immediate.
        check(terrain::LodNext(2, 10.0f, bands) == 0, "LodNext refines coarse->LOD0 immediately when near");
        check(terrain::LodNext(1, 5.0f, bands) == 0, "LodNext refines LOD1->LOD0 immediately when near");

        // Hysteresis prevents coarsening within the margin past the band edge AFTER an upgrade. A tile at
        // LOD0 that drifts just past bandNear (but within bandNear+hysteresis) stays LOD0.
        check(terrain::LodNext(0, 19.0f, bands) == 0, "LOD0 stays LOD0 within hysteresis past bandNear");
        check(terrain::LodNext(0, 18.5f, bands) == 0, "LOD0 stays LOD0 just past bandNear (no flicker)");
        // Only once it is past bandNear + hysteresis does it coarsen to LOD1.
        check(terrain::LodNext(0, 23.0f, bands) == 1, "LOD0 coarsens to LOD1 past bandNear+hysteresis");
        // Same on the mid edge for LOD1 -> LOD2.
        check(terrain::LodNext(1, 41.0f, bands) == 1, "LOD1 stays LOD1 within hysteresis past bandMid");
        check(terrain::LodNext(1, 45.0f, bands) == 2, "LOD1 coarsens to LOD2 past bandMid+hysteresis");
    }

    // -------------------------------------------------------------------------------------------
    // 4. Per-LOD mesh vertex / index counts (the documented LOD table).
    // -------------------------------------------------------------------------------------------
    {
        TileGrid grid(kT, kTileSize);
        const int expectN[3] = {96, 48, 24};
        for (int lod = 0; lod < terrain::kLodCount; ++lod) {
            int n = expectN[lod];
            check(terrain::kLodN[lod] == n, "kLodN matches the documented per-LOD resolution");
            terrain::TerrainMesh m = grid.BuildTileMesh(1, 1, lod, kHeightScale);
            check((int)m.verts.size() == n * n, "LOD mesh vertex count == n*n");
            check((int)m.indices.size() == (n - 1) * (n - 1) * 6, "LOD mesh index count == (n-1)^2*6");
            check(terrain::LodVertexCount(lod) == n * n, "LodVertexCount helper == n*n");
            check(terrain::LodIndexCount(lod) == (n - 1) * (n - 1) * 6, "LodIndexCount helper == (n-1)^2*6");
            // Every index is in range.
            bool inRange = true;
            for (uint32_t idx : m.indices) if (idx >= (uint32_t)(n * n)) { inRange = false; break; }
            check(inRange, "all LOD mesh indices are in [0, n*n)");
        }
    }

    // -------------------------------------------------------------------------------------------
    // 5. Residency + LOD determinism: a scripted camera path replayed TWICE yields identical per-frame
    //    {resident tile ids + their LODs}. This is the contract the --terrain-stream-shot golden +
    //    state assertion rest on.
    // -------------------------------------------------------------------------------------------
    {
        TerrainStreamConfig cfg;
        cfg.loadRadius = 34.0f;
        cfg.unloadRadius = 46.0f;
        cfg.loadBudgetPerFrame = 4;
        cfg.unloadBudgetPerFrame = 4;
        cfg.heightScale = kHeightScale;
        cfg.bands.bandNear = 18.0f;
        cfg.bands.bandMid = 40.0f;
        cfg.bands.hysteresis = 4.0f;

        // Scripted path flying diagonally across the grid (NO live input/RNG/clock).
        std::vector<Vec3> path;
        const float startXZ = -48.0f, endXZ = 48.0f;
        const int frames = 60;
        for (int f = 0; f < frames; ++f) {
            float t = (float)f / (float)(frames - 1);
            float p = startXZ + t * (endXZ - startXZ);
            path.push_back(Vec3{p, 6.0f, p});
        }

        // Per-frame state = the sorted list of (id, lod) pairs for resident tiles.
        auto runPath = [&]() {
            TerrainStreamWorld w(kT, kTileSize, cfg);
            std::vector<std::vector<std::pair<int, int>>> perFrame;
            for (const Vec3& p : path) {
                w.Update(p);
                std::vector<std::pair<int, int>> state;
                for (const auto& rt : w.ResidentTiles())
                    state.emplace_back(rt.tile.id, rt.lod);
                perFrame.push_back(std::move(state));
            }
            return perFrame;
        };

        auto run1 = runPath();
        auto run2 = runPath();
        check(run1 == run2, "scripted path replays bit-identical per-frame {resident ids + LODs}");

        // The path actually streamed tiles in AND exercised >1 LOD somewhere along it.
        bool anyResident = false;
        bool sawLod0 = false, sawLod2 = false;
        for (const auto& f : run1) {
            if (!f.empty()) anyResident = true;
            for (const auto& [id, lod] : f) { (void)id; if (lod == 0) sawLod0 = true; if (lod == 2) sawLod2 = true; }
        }
        check(anyResident, "scripted path streamed some tiles resident");
        check(sawLod0 && sawLod2, "scripted path exercised a mix of LODs (near LOD0 + far LOD2)");

        // Every resident tile's mesh pointer is non-null and sized to its LOD (the renderer contract).
        TerrainStreamWorld w(kT, kTileSize, cfg);
        for (const Vec3& p : path) w.Update(p);
        for (const auto& rt : w.ResidentTiles()) {
            check(rt.mesh != nullptr, "resident tile has a non-null mesh");
            if (rt.mesh) {
                int n = terrain::kLodN[rt.lod];
                check((int)rt.mesh->verts.size() == n * n, "resident tile mesh sized to its LOD");
            }
        }
        // Stats consistency: resident == sum(byLod), total == 36.
        terrain::TerrainStreamStats s = w.Stats();
        check(s.total == 36, "stats.total == 36 tiles");
        check(s.resident == s.byLod[0] + s.byLod[1] + s.byLod[2], "stats resident == sum(byLod)");
    }

    // -------------------------------------------------------------------------------------------
    // 6. A resident tile that gets CLOSER refines its LOD (and rebuilds its mesh); the mesh follows the
    //    new LOD. Drive the camera onto a tile, then assert its LOD is the finest + the mesh resized.
    // -------------------------------------------------------------------------------------------
    {
        TerrainStreamConfig cfg;
        cfg.loadRadius = 34.0f;
        cfg.unloadRadius = 46.0f;
        cfg.loadBudgetPerFrame = 64;       // no throttle for this targeted test
        cfg.unloadBudgetPerFrame = 64;
        cfg.heightScale = kHeightScale;
        cfg.bands.bandNear = 18.0f;
        cfg.bands.bandMid = 40.0f;
        cfg.bands.hysteresis = 4.0f;

        TerrainStreamWorld w(kT, kTileSize, cfg);
        TileGrid grid(kT, kTileSize);
        Vec3 c = grid.TileCenter(3, 3);

        // Mid-overhead first: tile (3,3) loads resident (dist 25 <= loadRadius 34) but at a coarse LOD
        // (dist 25 > bandNear 18 + hysteresis -> LOD1).
        w.Update(Vec3{c.x, 25.0f, c.z});
        int idTile = grid.IdOf(3, 3);
        check(w.StateOf(idTile) == TileState::Resident, "target tile resident from overhead");
        int coarseLod = w.LodOf(idTile);
        check(coarseLod >= 1, "target tile starts at a coarse LOD when far");

        // Now move the camera right onto the tile center: it should refine to LOD0 + rebuild.
        int buildsBefore = w.BuildCount(idTile);
        w.Update(c);
        check(w.LodOf(idTile) == 0, "target tile refines to LOD0 when the camera is on it");
        check(w.BuildCount(idTile) > buildsBefore, "refining the LOD rebuilt the tile mesh");
        for (const auto& rt : w.ResidentTiles())
            if (rt.tile.id == idTile)
                check(rt.mesh && (int)rt.mesh->verts.size() == terrain::kLodN[0] * terrain::kLodN[0],
                      "refined tile mesh is now LOD0-sized (96x96)");
    }

    if (g_fail == 0) std::printf("terrain_stream_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
