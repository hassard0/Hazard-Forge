#include "terrain/heightmap.h"

#include <cmath>
#include <cstdint>

namespace hf::terrain {

namespace {

// --- Deterministic constants (LOCKED). All amplitudes/frequencies/phases are fixed compile-time
// literals; the whole field is a pure function of (x,z). Documented here so a reader can reproduce
// any sample by hand. -------------------------------------------------------------------------------
constexpr float kA1 = 0.60f;   // primary rolling-hill amplitude
constexpr float kF1 = 0.55f;   // primary frequency
constexpr float kA2 = 0.25f;   // secondary ripple amplitude
constexpr float kF2 = 1.30f;   // secondary frequency
constexpr float kP2x = 1.70f;  // secondary X phase
constexpr float kP2z = 0.40f;  // secondary Z phase
constexpr float kA3 = 0.35f;   // value-noise amplitude
constexpr float kNF = 0.30f;   // value-noise lattice frequency (world units -> lattice coords)

// 2D integer hash -> a float in [0,1). A small, fully deterministic integer mix (Wang-style avalanche
// on a single combined 32-bit key). No RNG, no float-time, no platform-dependent behaviour: pure
// uint32 arithmetic (wraps mod 2^32 identically on every target), so the lattice is bit-identical on
// Windows/Vulkan and Apple/Metal.
float HashLattice(int32_t ix, int32_t iz) {
    uint32_t h = static_cast<uint32_t>(ix) * 374761393u + static_cast<uint32_t>(iz) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    // Map the top 24 bits to [0,1) so the result is exactly representable as a float (no rounding).
    return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}

// Smoothstep-faded bilinear value noise over the integer lattice. Smoothstep (3t^2-2t^3) on the
// fractional coords gives C1-continuous noise (no creased seams) while staying a pure function.
float ValueNoise(float x, float z) {
    float fx = std::floor(x);
    float fz = std::floor(z);
    int32_t ix = static_cast<int32_t>(fx);
    int32_t iz = static_cast<int32_t>(fz);
    float tx = x - fx;
    float tz = z - fz;
    float sx = tx * tx * (3.0f - 2.0f * tx);
    float sz = tz * tz * (3.0f - 2.0f * tz);
    float c00 = HashLattice(ix,     iz);
    float c10 = HashLattice(ix + 1, iz);
    float c01 = HashLattice(ix,     iz + 1);
    float c11 = HashLattice(ix + 1, iz + 1);
    float a = c00 + (c10 - c00) * sx;
    float b = c01 + (c11 - c01) * sx;
    return a + (b - a) * sz;
}

}  // namespace

float Height(float x, float z) {
    float h = kA1 * std::sin(kF1 * x) * std::cos(kF1 * z);
    h += kA2 * std::sin(kF2 * x + kP2x) * std::cos(kF2 * z + kP2z);
    h += kA3 * ValueNoise(x * kNF, z * kNF);
    return h;
}

TerrainMesh BuildTerrain(int n, float worldSize, float heightScale) {
    TerrainMesh out;
    if (n < 2) return out;

    const float half = worldSize * 0.5f;
    const float step = worldSize / static_cast<float>(n - 1);
    // Finite-difference epsilon for the central-difference normal. A small fixed world-space offset
    // (independent of n) so the normal samples the analytic slope of Height, not the discretized grid.
    const float eps = step * 0.5f;

    out.verts.reserve(static_cast<size_t>(n) * static_cast<size_t>(n));
    float peak = -1e30f;
    for (int iz = 0; iz < n; ++iz) {
        for (int ix = 0; ix < n; ++ix) {
            float x = -half + step * static_cast<float>(ix);
            float z = -half + step * static_cast<float>(iz);
            float y = Height(x, z) * heightScale;
            if (y > peak) peak = y;

            // Central finite-difference normal: dy/dx and dy/dz of the displaced surface, scaled by
            // heightScale. The surface tangents are (1,dHx,0) along +X and (0,dHz,1) along +Z, where
            // dHx = scale * d(Height)/dx and dHz = scale * d(Height)/dz. N = normalize(cross(dz, dx))
            // with dx = (2*eps, 2*scale*dHx, 0), dz = (0, 2*scale*dHz, 2*eps); cross(dz,dx) yields an
            // up-facing (+Y) normal for a flat region.
            float hx0 = Height(x - eps, z);
            float hx1 = Height(x + eps, z);
            float hz0 = Height(x, z - eps);
            float hz1 = Height(x, z + eps);
            float dHx = (hx1 - hx0) / (2.0f * eps) * heightScale;
            float dHz = (hz1 - hz0) / (2.0f * eps) * heightScale;
            // N = normalize(-dHx, 1, -dHz) — the standard heightfield normal (== cross(dz,dx) up to
            // a positive scale, which normalize removes).
            float nx = -dHx, ny = 1.0f, nz = -dHz;
            float inv = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
            nx *= inv; ny *= inv; nz *= inv;

            float u = static_cast<float>(ix) / static_cast<float>(n - 1);
            float v = static_cast<float>(iz) / static_cast<float>(n - 1);

            // Deterministic height-based color ramp (legibility tint baked into the vertex color, which
            // the unchanged lit shader multiplies into the surface — no new shader). Normalize the
            // height into [0,1] over the field's amplitude band, then ramp low->grass, mid->rock,
            // high->snow. Pure function of y.
            const float kBand = (kA1 + kA2 + kA3) * heightScale;  // ~max |height*scale|
            float t = 0.5f + 0.5f * (kBand > 0.0f ? (y / kBand) : 0.0f);
            t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
            float r, g, b;
            if (t < 0.5f) {
                float k = t * 2.0f;                       // 0..1 across the low band
                r = 0.20f + (0.45f - 0.20f) * k;
                g = 0.45f + (0.40f - 0.45f) * k;
                b = 0.18f + (0.28f - 0.18f) * k;
            } else {
                float k = (t - 0.5f) * 2.0f;              // 0..1 across the high band
                r = 0.45f + (0.95f - 0.45f) * k;
                g = 0.40f + (0.95f - 0.40f) * k;
                b = 0.28f + (1.00f - 0.28f) * k;
            }

            scene::Vertex vert{};
            vert.pos[0] = x;  vert.pos[1] = y;  vert.pos[2] = z;
            vert.color[0] = r; vert.color[1] = g; vert.color[2] = b;
            vert.uv[0] = u; vert.uv[1] = v;
            vert.normal[0] = nx; vert.normal[1] = ny; vert.normal[2] = nz;
            vert.tangent[0] = 1.0f; vert.tangent[1] = 0.0f; vert.tangent[2] = 0.0f;
            out.verts.push_back(vert);
        }
    }
    out.peak = peak;

    // Two triangles per quad. Wound CCW when viewed from above (+Y looking down), matching the engine's
    // Plane/Cube top-face winding so the terrain survives back-face culling. Grid index (ix,iz) ->
    // iz*n + ix.
    out.indices.reserve(static_cast<size_t>(n - 1) * static_cast<size_t>(n - 1) * 6);
    for (int iz = 0; iz < n - 1; ++iz) {
        for (int ix = 0; ix < n - 1; ++ix) {
            uint32_t a = static_cast<uint32_t>(iz * n + ix);          // (ix,   iz)
            uint32_t b = static_cast<uint32_t>(iz * n + ix + 1);      // (ix+1, iz)
            uint32_t c = static_cast<uint32_t>((iz + 1) * n + ix);    // (ix,   iz+1)
            uint32_t d = static_cast<uint32_t>((iz + 1) * n + ix + 1);// (ix+1, iz+1)
            // CCW-from-above winding (matches Mesh::Plane's (-x,+z)(+x,+z)(+x,-z) order).
            out.indices.push_back(c); out.indices.push_back(d); out.indices.push_back(b);
            out.indices.push_back(c); out.indices.push_back(b); out.indices.push_back(a);
        }
    }

    return out;
}

TerrainMesh BuildTerrainTile(float worldMinX, float worldMinZ, float tileSize, int n,
                             float heightScale) {
    TerrainMesh out;
    if (n < 2) return out;

    // World-space step across the tile (the tile spans exactly tileSize, so the last column/row lands
    // on worldMinX/Z + tileSize — the shared edge with the neighbouring tile, sampled at the SAME
    // global coordinate => bit-identical there at equal LOD).
    const float step = tileSize / static_cast<float>(n - 1);
    // Same finite-difference epsilon convention as BuildTerrain (a small fixed world-space offset,
    // independent of n, so the normal samples the analytic slope of Height).
    const float eps = step * 0.5f;

    out.verts.reserve(static_cast<size_t>(n) * static_cast<size_t>(n));
    float peak = -1e30f;
    for (int iz = 0; iz < n; ++iz) {
        for (int ix = 0; ix < n; ++ix) {
            float x = worldMinX + step * static_cast<float>(ix);
            float z = worldMinZ + step * static_cast<float>(iz);
            float y = Height(x, z) * heightScale;
            if (y > peak) peak = y;

            // Central finite-difference heightfield normal (identical formula to BuildTerrain).
            float hx0 = Height(x - eps, z);
            float hx1 = Height(x + eps, z);
            float hz0 = Height(x, z - eps);
            float hz1 = Height(x, z + eps);
            float dHx = (hx1 - hx0) / (2.0f * eps) * heightScale;
            float dHz = (hz1 - hz0) / (2.0f * eps) * heightScale;
            float nx = -dHx, ny = 1.0f, nz = -dHz;
            float inv = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
            nx *= inv; ny *= inv; nz *= inv;

            // UVs LOCAL to the tile (grid coords in [0,1]^2).
            float u = static_cast<float>(ix) / static_cast<float>(n - 1);
            float v = static_cast<float>(iz) / static_cast<float>(n - 1);

            // Deterministic height-based color ramp (identical to BuildTerrain: pure function of y).
            const float kBand = (kA1 + kA2 + kA3) * heightScale;  // ~max |height*scale|
            float t = 0.5f + 0.5f * (kBand > 0.0f ? (y / kBand) : 0.0f);
            t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
            float r, g, b;
            if (t < 0.5f) {
                float k = t * 2.0f;
                r = 0.20f + (0.45f - 0.20f) * k;
                g = 0.45f + (0.40f - 0.45f) * k;
                b = 0.18f + (0.28f - 0.18f) * k;
            } else {
                float k = (t - 0.5f) * 2.0f;
                r = 0.45f + (0.95f - 0.45f) * k;
                g = 0.40f + (0.95f - 0.40f) * k;
                b = 0.28f + (1.00f - 0.28f) * k;
            }

            scene::Vertex vert{};
            vert.pos[0] = x;  vert.pos[1] = y;  vert.pos[2] = z;
            vert.color[0] = r; vert.color[1] = g; vert.color[2] = b;
            vert.uv[0] = u; vert.uv[1] = v;
            vert.normal[0] = nx; vert.normal[1] = ny; vert.normal[2] = nz;
            vert.tangent[0] = 1.0f; vert.tangent[1] = 0.0f; vert.tangent[2] = 0.0f;
            out.verts.push_back(vert);
        }
    }
    out.peak = peak;

    // Two CCW-from-above triangles per quad (identical winding to BuildTerrain).
    out.indices.reserve(static_cast<size_t>(n - 1) * static_cast<size_t>(n - 1) * 6);
    for (int iz = 0; iz < n - 1; ++iz) {
        for (int ix = 0; ix < n - 1; ++ix) {
            uint32_t a = static_cast<uint32_t>(iz * n + ix);
            uint32_t b = static_cast<uint32_t>(iz * n + ix + 1);
            uint32_t c = static_cast<uint32_t>((iz + 1) * n + ix);
            uint32_t d = static_cast<uint32_t>((iz + 1) * n + ix + 1);
            out.indices.push_back(c); out.indices.push_back(d); out.indices.push_back(b);
            out.indices.push_back(c); out.indices.push_back(b); out.indices.push_back(a);
        }
    }

    return out;
}

}  // namespace hf::terrain
