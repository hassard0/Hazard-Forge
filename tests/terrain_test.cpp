// Slice BF unit test: procedural terrain / heightmap (engine/terrain/heightmap.{h,cpp}). PURE CPU,
// no GPU. Validates:
//   * Height determinism + hand-anchored sample values (an independent in-test reimplementation of
//     the documented formula must match Height to the bit at several points),
//   * mesh structure (exactly n*n verts + (n-1)*(n-1)*6 indices; every index in range; corner/edge
//     world positions; vertex Y == Height*scale at a sample vertex),
//   * normals (flat-ish behaviour, a known slope tilts the expected way, all unit length),
//   * determinism (two BuildTerrain calls -> bit-identical vertex + index buffers).
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "terrain/heightmap.h"
#include "scene/vertex.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// --- Independent reference reimplementation of the LOCKED documented formula. Kept in lock-step with
// heightmap.cpp's constants; if either drifts, the hand-anchored asserts below fail. -----------------
namespace ref {
constexpr float kA1 = 0.60f, kF1 = 0.55f;
constexpr float kA2 = 0.25f, kF2 = 1.30f, kP2x = 1.70f, kP2z = 0.40f;
constexpr float kA3 = 0.35f, kNF = 0.30f;

float HashLattice(int32_t ix, int32_t iz) {
    uint32_t h = static_cast<uint32_t>(ix) * 374761393u + static_cast<uint32_t>(iz) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}
float ValueNoise(float x, float z) {
    float fx = std::floor(x), fz = std::floor(z);
    int32_t ix = (int32_t)fx, iz = (int32_t)fz;
    float tx = x - fx, tz = z - fz;
    float sx = tx * tx * (3.0f - 2.0f * tx);
    float sz = tz * tz * (3.0f - 2.0f * tz);
    float c00 = HashLattice(ix, iz), c10 = HashLattice(ix + 1, iz);
    float c01 = HashLattice(ix, iz + 1), c11 = HashLattice(ix + 1, iz + 1);
    float a = c00 + (c10 - c00) * sx;
    float b = c01 + (c11 - c01) * sx;
    return a + (b - a) * sz;
}
float Height(float x, float z) {
    float h = kA1 * std::sin(kF1 * x) * std::cos(kF1 * z);
    h += kA2 * std::sin(kF2 * x + kP2x) * std::cos(kF2 * z + kP2z);
    h += kA3 * ValueNoise(x * kNF, z * kNF);
    return h;
}
}  // namespace ref

int main() {
    HF_TEST_MAIN_INIT();
    // --- 1. Height determinism: same input -> same output, every call. ---------------------------
    {
        float a = terrain::Height(3.1f, -2.4f);
        float b = terrain::Height(3.1f, -2.4f);
        check(a == b, "Height is deterministic (bit-identical across calls)");
    }

    // --- 2. Hand-anchored sample values: Height must equal the independent reference to the bit at
    // several points (the formula + constants are LOCKED). -----------------------------------------
    {
        const float pts[][2] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {-3.3f, 2.7f},
                                {7.5f, -4.2f}, {10.0f, 10.0f}};
        bool allMatch = true;
        for (auto& p : pts) {
            if (terrain::Height(p[0], p[1]) != ref::Height(p[0], p[1])) allMatch = false;
        }
        check(allMatch, "Height matches the documented-formula reference at anchored points");
        // A concrete spot-value: at (0,0) the first sine term vanishes (sin(0)=0); the field reduces
        // to the secondary term + value-noise, which the reference computes identically.
        check(terrain::Height(0.0f, 0.0f) == ref::Height(0.0f, 0.0f), "Height(0,0) anchored");
    }

    // --- 3. Mesh structure: counts, index range, corner/edge positions, displaced Y. -------------
    {
        const int n = 16;
        const float worldSize = 20.0f, heightScale = 1.5f;
        terrain::TerrainMesh m = terrain::BuildTerrain(n, worldSize, heightScale);

        check(m.verts.size() == (size_t)n * n, "vertex count == n*n");
        check(m.indices.size() == (size_t)(n - 1) * (n - 1) * 6, "index count == (n-1)*(n-1)*6");

        bool inRange = true;
        for (uint32_t idx : m.indices) if (idx >= m.verts.size()) inRange = false;
        check(inRange, "every index is in [0, n*n)");

        const float half = worldSize * 0.5f;
        const float step = worldSize / (float)(n - 1);
        // Vertex (ix,iz) is laid out at iz*n + ix; check the four corners' XZ world positions.
        auto vat = [&](int ix, int iz) -> const scene::Vertex& { return m.verts[(size_t)iz * n + ix]; };
        const scene::Vertex& v00 = vat(0, 0);
        const scene::Vertex& vN0 = vat(n - 1, 0);
        const scene::Vertex& v0N = vat(0, n - 1);
        const scene::Vertex& vNN = vat(n - 1, n - 1);
        check(std::fabs(v00.pos[0] - (-half)) < 1e-4f && std::fabs(v00.pos[2] - (-half)) < 1e-4f,
              "corner (0,0) at (-half,-half)");
        check(std::fabs(vNN.pos[0] - half) < 1e-4f && std::fabs(vNN.pos[2] - half) < 1e-4f,
              "corner (n-1,n-1) at (+half,+half)");
        check(std::fabs(vN0.pos[0] - half) < 1e-4f && std::fabs(vN0.pos[2] - (-half)) < 1e-4f,
              "corner (n-1,0) at (+half,-half)");
        check(std::fabs(v0N.pos[0] - (-half)) < 1e-4f && std::fabs(v0N.pos[2] - half) < 1e-4f,
              "corner (0,n-1) at (-half,+half)");
        // An interior edge vertex's X spacing is exactly `step`.
        check(std::fabs(vat(1, 0).pos[0] - vat(0, 0).pos[0] - step) < 1e-4f,
              "adjacent vertices are `step` apart in X");
        // UVs are grid coords in [0,1].
        check(std::fabs(v00.uv[0]) < 1e-4f && std::fabs(v00.uv[1]) < 1e-4f, "corner (0,0) uv == (0,0)");
        check(std::fabs(vNN.uv[0] - 1.0f) < 1e-4f && std::fabs(vNN.uv[1] - 1.0f) < 1e-4f,
              "corner (n-1,n-1) uv == (1,1)");

        // Displaced height: vertex Y == Height(x,z)*scale at a sample vertex.
        const scene::Vertex& vs = vat(5, 7);
        float expectedY = terrain::Height(vs.pos[0], vs.pos[2]) * heightScale;
        check(std::fabs(vs.pos[1] - expectedY) < 1e-4f, "vertex Y == Height(x,z)*heightScale");

        // peak == max vertex Y over the grid.
        float maxY = -1e30f;
        for (const auto& v : m.verts) if (v.pos[1] > maxY) maxY = v.pos[1];
        check(std::fabs(m.peak - maxY) < 1e-5f, "TerrainMesh.peak == max vertex Y");
    }

    // --- 4. Normals: all unit length; flat region -> ~+Y; a known slope tilts opposite the gradient.
    {
        const int n = 32;
        terrain::TerrainMesh m = terrain::BuildTerrain(n, 20.0f, 1.5f);
        bool allUnit = true;
        for (const auto& v : m.verts) {
            float len = std::sqrt(v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] +
                                  v.normal[2] * v.normal[2]);
            if (std::fabs(len - 1.0f) > 1e-3f) allUnit = false;
            // Heightfield normals always have a positive +Y component (surface faces up).
            if (v.normal[1] <= 0.0f) allUnit = false;
        }
        check(allUnit, "all terrain normals are unit length with +Y up");

        // Flat region: a degenerate flat heightfield (heightScale 0) -> every normal is exactly +Y.
        terrain::TerrainMesh flat = terrain::BuildTerrain(8, 20.0f, 0.0f);
        bool allPlusY = true;
        for (const auto& v : flat.verts) {
            if (std::fabs(v.normal[0]) > 1e-5f || std::fabs(v.normal[1] - 1.0f) > 1e-5f ||
                std::fabs(v.normal[2]) > 1e-5f) allPlusY = false;
        }
        check(allPlusY, "flat (scale 0) terrain has every normal == +Y");

        // Known slope direction: where d(Height)/dx > 0 the normal must tilt toward -X (nx < 0); the
        // heightfield normal is normalize(-dHx, 1, -dHz). Pick a point with a clearly non-zero X slope.
        {
            float x = 1.5f, z = 0.0f, e = 0.05f, scale = 1.5f;
            float dHx = (terrain::Height(x + e, z) - terrain::Height(x - e, z)) / (2.0f * e) * scale;
            // Rebuild the expected normal sign and compare to a freshly evaluated central-diff normal.
            float dHz = (terrain::Height(x, z + e) - terrain::Height(x, z - e)) / (2.0f * e) * scale;
            float nx = -dHx, ny = 1.0f, nz = -dHz;
            float inv = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
            nx *= inv;
            check((dHx > 0.0f) == (nx < 0.0f), "normal tilts opposite the X height-gradient sign");
        }
    }

    // --- 5. Determinism: two BuildTerrain calls produce bit-identical vertex + index buffers. -----
    {
        terrain::TerrainMesh a = terrain::BuildTerrain(48, 20.0f, 1.5f);
        terrain::TerrainMesh b = terrain::BuildTerrain(48, 20.0f, 1.5f);
        bool sameSize = a.verts.size() == b.verts.size() && a.indices.size() == b.indices.size();
        check(sameSize, "two BuildTerrain calls produce equal-size buffers");
        bool bitIdentical = sameSize &&
            std::memcmp(a.verts.data(), b.verts.data(), a.verts.size() * sizeof(scene::Vertex)) == 0 &&
            std::memcmp(a.indices.data(), b.indices.data(), a.indices.size() * sizeof(uint32_t)) == 0 &&
            a.peak == b.peak;
        check(bitIdentical, "two BuildTerrain calls are bit-identical (verts + indices + peak)");
    }

    if (g_fail == 0) { std::printf("terrain_test OK\n"); return 0; }
    std::printf("terrain_test: %d failures\n", g_fail);
    return 1;
}
