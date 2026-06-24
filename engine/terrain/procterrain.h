#pragma once
// Slice PT1 — Integer fBm heightfield generation (the BEACHHEAD of FLAGSHIP #26: DETERMINISTIC
// PROCEDURAL TERRAIN, hf::terrain). Pure CPU, header-only. NO device, NO backend symbols, NO new RHI,
// NO new shader. Namespace hf::terrain.
//
// The irreducible primitive: a DETERMINISTIC INTEGER HEIGHTFIELD. IntHeight(x,z) returns a Q16.16
// height from a fractal sum (fBm) of integer value-noise octaves, bit-identical CPU<->Vulkan<->Metal
// BY CONSTRUCTION — NO runtime sin/sqrt/floor: the strict-integer twin of the existing FLOAT
// terrain::Height (engine/terrain/heightmap.cpp:26-61, which is FROZEN by the `terrain`/`terrain_stream`
// goldens + terrain_test.cpp and is NOT touched here). This is the additive sibling of the float
// heightmap.h — the strict-integer heightfield the erosion (PT2/PT3) and render (PT4-PT6) slices build on.
//
// WHY IT IS BIT-EXACT (the cross-backend crux): every operation is a pure uint32/int32/int64 integer
// op that wraps + shifts identically on every compiler/vendor. IntHashLattice is a Wang-style avalanche
// on a combined 32-bit key (== heightmap.cpp's HashLattice integer mix, mapped to a Q16.16 fraction).
// IntValueNoise is a smoothstep-faded bilinear blend over that lattice in Q16.16 (the t^2(3-2t) fade
// done in fixed point). IntHeight sums octaves with halving amplitude / doubling frequency. The whole
// field is a pure function of (x,z,octaves,seed) — NO <cmath>, NO float on the bit-exact path.
//
// REUSE MAP: fx/kOne/kFrac/fxmul come from engine/sim/fpx.h (read-only — the Q16.16 toolbox). The
// integer hash shape mirrors engine/terrain/heightmap.cpp:26-32 HashLattice; the smoothstep-bilinear
// shape mirrors heightmap.cpp:36-52 ValueNoise; the fBm octave loop mirrors the standard fractal sum.

#include <cstdint>
#include <cmath>      // PT4 render bridge ONLY (std::sqrt for the host-float normal). NOT on the PT1-3 integer path.
#include <vector>

#include "sim/fpx.h"  // fx / kOne / kFrac / fxmul + FxToFloat (Q16.16 toolbox), read-only.
#include "terrain/heightmap.h"  // PT4 render bridge ONLY: scene::Vertex + TerrainMesh (read-only). BuildTerrain FROZEN.

namespace hf::terrain {

// Pull the Q16.16 primitives from the fixed-point toolbox (read-only reuse).
using hf::sim::fpx::fx;
using hf::sim::fpx::kOne;
using hf::sim::fpx::kFrac;
using hf::sim::fpx::fxmul;
using hf::sim::fpx::FxToFloat;  // PT4 render bridge ONLY: the single Q16.16 -> float crossing.

// IntHashLattice(ix, iz, seed): a deterministic Q16.16 corner value in [0, kOne). The SAME integer hash
// shape as heightmap.cpp's float HashLattice (a Wang-style avalanche on a combined 32-bit key) with the
// seed folded in before mixing; the top 16 bits of the avalanched hash become the Q16.16 fraction (so
// the result is strictly < kOne == 1.0). Pure uint32 arithmetic — wraps mod 2^32 identically on every
// target, so the lattice is bit-identical Windows/Vulkan <-> Apple/Metal. NO float.
inline fx IntHashLattice(int32_t ix, int32_t iz, uint32_t seed) {
    uint32_t h = static_cast<uint32_t>(ix) * 374761393u + static_cast<uint32_t>(iz) * 668265263u;
    h += seed * 2246822519u;                 // fold the seed in (a distinct prime) before the avalanche
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return static_cast<fx>(h >> 16);         // top 16 bits => Q16.16 fraction in [0, kOne)
}

// IntValueNoise(x, z, seed): smoothstep-faded bilinear value noise over the integer lattice, x/z in
// Q16.16. ix = (int32_t)(x >> kFrac) is an ARITHMETIC right shift == floor (C++20 makes >> on signed
// arithmetic), tx = x - (ix << kFrac) the fractional part in [0, kOne). The smoothstep fade
// sx = t^2(3-2t) is done in Q16.16; the 4 lattice corners are blended bilinearly. Pure integer — the
// strict-integer twin of heightmap.cpp ValueNoise. Returns a value in [0, kOne).
inline fx IntValueNoise(fx x, fx z, uint32_t seed) {
    const int32_t ix = static_cast<int32_t>(x >> kFrac);   // floor(x) (arithmetic shift, C++20)
    const int32_t iz = static_cast<int32_t>(z >> kFrac);   // floor(z)
    const fx tx = x - (static_cast<fx>(ix) << kFrac);      // fractional part in [0, kOne)
    const fx tz = z - (static_cast<fx>(iz) << kFrac);
    // smoothstep s = t*t*(3 - 2t) in Q16.16 (the C1-continuous fade => no creased seams).
    const fx sx = fxmul(fxmul(tx, tx), 3 * kOne - 2 * tx);
    const fx sz = fxmul(fxmul(tz, tz), 3 * kOne - 2 * tz);
    const fx c00 = IntHashLattice(ix,     iz,     seed);
    const fx c10 = IntHashLattice(ix + 1, iz,     seed);
    const fx c01 = IntHashLattice(ix,     iz + 1, seed);
    const fx c11 = IntHashLattice(ix + 1, iz + 1, seed);
    const fx a = c00 + fxmul(c10 - c00, sx);
    const fx b = c01 + fxmul(c11 - c01, sx);
    return a + fxmul(b - a, sz);
}

// IntHeight(x, z, octaves, seed): fractional Brownian motion (fBm) — a fractal sum of value-noise
// octaves with halving amplitude / doubling frequency, x/z in Q16.16. Each octave is offset in the
// noise hash (seed + o) so the layers are decorrelated. octaves <= 0 => 0 (a flat no-op). Pure integer;
// NO <cmath>, NO float. The base amplitude (kOne/2) keeps the summed height well inside the +-32768
// Q16.16 world bound (sum of halving amps < kOne, value-noise < kOne => |h| < kOne).
inline fx IntHeight(fx x, fx z, int octaves, uint32_t seed) {
    fx h = 0;
    fx amp = kOne / 2;            // base amplitude (sum over octaves converges below kOne)
    fx freq = kOne / 4;           // base frequency (1/4 lattice cell per world unit => smooth rolling)
    for (int o = 0; o < octaves; ++o) {
        h += fxmul(amp, IntValueNoise(fxmul(x, freq), fxmul(z, freq), seed + static_cast<uint32_t>(o)));
        amp >>= 1;                // halve amplitude
        freq <<= 1;               // double frequency
    }
    return h;
}

// GenHeightField(seed, n, worldSize, octaves): sample IntHeight at an n x n grid over [0, worldSize)^2
// in Q16.16 (cell (gx,gz) -> x = gx*worldSize/n, z = gz*worldSize/n via an int64 intermediate =>
// deterministic floor division, NO float). Returns the n*n field row-major (index gz*n + gx). Pure
// integer; NO <cmath>, NO float. octaves <= 0 => every cell 0 (the flat no-op).
inline std::vector<fx> GenHeightField(uint32_t seed, int n, fx worldSize, int octaves) {
    std::vector<fx> field;
    if (n <= 0) return field;
    field.resize(static_cast<size_t>(n) * static_cast<size_t>(n));
    for (int gz = 0; gz < n; ++gz) {
        const fx z = static_cast<fx>((static_cast<int64_t>(gz) * static_cast<int64_t>(worldSize)) /
                                     static_cast<int64_t>(n));
        for (int gx = 0; gx < n; ++gx) {
            const fx x = static_cast<fx>((static_cast<int64_t>(gx) * static_cast<int64_t>(worldSize)) /
                                         static_cast<int64_t>(n));
            field[static_cast<size_t>(gz) * static_cast<size_t>(n) + static_cast<size_t>(gx)] =
                IntHeight(x, z, octaves, seed);
        }
    }
    return field;
}

// ============================ SLICE PT4 — eroded terrain MESH (the RENDER BRIDGE) =========================
// (APPEND-ONLY after PT1; PT1's IntHashLattice/IntValueNoise/IntHeight/GenHeightField are NOT modified.)
//
// THE ONE FLOAT CROSSING OF THE WHOLE FLAGSHIP. BuildIntTerrainMesh turns the bit-exact INTEGER eroded
// heightfield (PT1 GenHeightField + PT2 ErodeHydraulic + PT3 ErodeThermal — all strict Q16.16) into a lit
// 3D terrain MESH (scene::Vertex / uint32 indices) for the existing lit/shadow pipeline. It MIRRORS the
// FROZEN float BuildTerrain (heightmap.cpp:66-130) — same world layout, same UVs, same height-tint color
// ramp, same (n-1)*(n-1)*6 two-CCW-triangles-per-quad winding — but drives Y from the INTEGER grid via the
// SINGLE FxToFloat crossing (render-only). This is the FO5/PCG6 render-bridge precedent: the DATA (the
// grid) is bit-exact + byte-identical cross-backend; only this float mesh build + the GPU raster diverge
// (the visresolve bar — deterministic per-run, NOT a strict zero-diff cross-vendor pixel compare). This is
// the ONLY float code in procterrain.h; PT1-PT3 stay pure integer. NO RHI, NO backend symbols.
//
// grid is the n*n row-major eroded field (index gz*n + gx). Vertex (ix,iz): world x = -half + step*ix,
// z = -half + step*iz (half = worldSize/2, step = worldSize/(n-1)); y = FxToFloat(grid[iz*n+ix]) *
// heightScale. The central-difference normal samples the INTEGER grid neighbours (clamped at edges),
// FxToFloat'd, finite-differenced -> N = normalize(-dHx, 1, -dHz) in host float (deterministic, NOT
// bit-exact — the float side of the bar). An empty/degenerate grid (n < 2 or size mismatch) -> a flat
// plane (all y=0, normals up — the no-op).
inline TerrainMesh BuildIntTerrainMesh(const std::vector<fx>& grid, int n, float worldSize,
                                       float heightScale) {
    TerrainMesh out;
    if (n < 2) return out;
    const bool haveGrid = (grid.size() == static_cast<size_t>(n) * static_cast<size_t>(n));

    const float half = worldSize * 0.5f;
    const float step = worldSize / static_cast<float>(n - 1);

    // Read a grid cell as a float height (render-only). Out-of-grid / no-grid -> 0 (flat). Clamp indices to
    // the grid edge so the boundary central-difference uses the nearest in-grid sample (mirrors the analytic
    // edge behaviour of BuildTerrain — no neighbour wrap).
    auto gridH = [&](int ix, int iz) -> float {
        if (!haveGrid) return 0.0f;
        if (ix < 0) ix = 0; else if (ix >= n) ix = n - 1;
        if (iz < 0) iz = 0; else if (iz >= n) iz = n - 1;
        return FxToFloat(grid[static_cast<size_t>(iz) * static_cast<size_t>(n) + static_cast<size_t>(ix)]);
    };

    // Height band over the actual integer grid (for the color ramp normalization). For a flat/no grid the
    // band is 0 -> the ramp falls back to the neutral mid color (matches the BuildTerrain kBand>0 guard).
    fx minFx = 0, maxFx = 0;
    if (haveGrid) {
        minFx = grid[0]; maxFx = grid[0];
        for (fx h : grid) { if (h < minFx) minFx = h; if (h > maxFx) maxFx = h; }
    }
    const float bandLo = FxToFloat(minFx) * heightScale;
    const float bandHi = FxToFloat(maxFx) * heightScale;
    const float bandSpan = bandHi - bandLo;  // 0 for a flat grid

    out.verts.reserve(static_cast<size_t>(n) * static_cast<size_t>(n));
    float peak = -1e30f;
    for (int iz = 0; iz < n; ++iz) {
        for (int ix = 0; ix < n; ++ix) {
            const float x = -half + step * static_cast<float>(ix);
            const float z = -half + step * static_cast<float>(iz);
            const float y = gridH(ix, iz) * heightScale;   // THE SINGLE Q16.16 -> float crossing (render-only).
            if (y > peak) peak = y;

            // Central finite-difference normal from the INTEGER grid neighbours (one grid step apart),
            // FxToFloat'd then finite-differenced. N = normalize(-dHx, 1, -dHz) — the standard heightfield
            // normal (up-facing for a flat region). Host float -> deterministic, NOT bit-exact.
            const float hx0 = gridH(ix - 1, iz) * heightScale;
            const float hx1 = gridH(ix + 1, iz) * heightScale;
            const float hz0 = gridH(ix, iz - 1) * heightScale;
            const float hz1 = gridH(ix, iz + 1) * heightScale;
            const float dHx = (hx1 - hx0) / (2.0f * step);
            const float dHz = (hz1 - hz0) / (2.0f * step);
            float nx = -dHx, ny = 1.0f, nz = -dHz;
            const float inv = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
            nx *= inv; ny *= inv; nz *= inv;

            const float u = static_cast<float>(ix) / static_cast<float>(n - 1);
            const float v = static_cast<float>(iz) / static_cast<float>(n - 1);

            // Height-tint vertex color ramp (mirrors heightmap.cpp:102-120: low->grass, mid->rock, high->snow),
            // normalized against THIS grid's height band (bandLo..bandHi). bandSpan==0 (flat) -> t=0.5 (neutral).
            float t = (bandSpan > 0.0f) ? (y - bandLo) / bandSpan : 0.5f;
            t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
            float r, g, b;
            if (t < 0.5f) {
                const float k = t * 2.0f;                     // 0..1 across the low band
                r = 0.20f + (0.45f - 0.20f) * k;
                g = 0.45f + (0.40f - 0.45f) * k;
                b = 0.18f + (0.28f - 0.18f) * k;
            } else {
                const float k = (t - 0.5f) * 2.0f;            // 0..1 across the high band
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

    // Two CCW-from-above triangles per quad — IDENTICAL winding to BuildTerrain (heightmap.cpp:137-147).
    out.indices.reserve(static_cast<size_t>(n - 1) * static_cast<size_t>(n - 1) * 6);
    for (int iz = 0; iz < n - 1; ++iz) {
        for (int ix = 0; ix < n - 1; ++ix) {
            const uint32_t a = static_cast<uint32_t>(iz * n + ix);
            const uint32_t b = static_cast<uint32_t>(iz * n + ix + 1);
            const uint32_t c = static_cast<uint32_t>((iz + 1) * n + ix);
            const uint32_t d = static_cast<uint32_t>((iz + 1) * n + ix + 1);
            out.indices.push_back(c); out.indices.push_back(d); out.indices.push_back(b);
            out.indices.push_back(c); out.indices.push_back(b); out.indices.push_back(a);
        }
    }

    return out;
}

// ============================ SLICE PT5 — bilinear integer height SAMPLE (the SEATING) ====================
// (APPEND-ONLY after PT4; PT1-PT4 logic is NOT modified.) The pure-integer primitive that SEATS the foliage
// meadow (FLAGSHIP #25) onto the eroded terrain surface: SampleHeight returns the bilinearly-interpolated
// Q16.16 height of the eroded grid at an arbitrary world XZ (in the SAME [0, worldSize) Q16.16 units the
// grid was generated over by GenHeightField). A plant scattered at world XZ is lifted to the terrain by
// plant.y = SampleHeight(grid, n, worldSize, plant.x, plant.z) — a PURE FUNCTION of the grid, so the seating
// is deterministic + provenance-checkable (recompute and compare). NO float: the four-corner bilinear blend
// is done with fxmul (Q16.16), exactly like the PT1 IntValueNoise corner blend (linear is fine for height).
//
// Map x/z to fractional grid coords gfx = (fx)((int64)x*(n-1)/worldSize) (Q16.16; the (n-1)/worldSize scale
// makes a world X of ix*worldSize/(n-1) land EXACTLY on integer grid index ix — i.e. it aligns with PT4's
// BuildIntTerrainMesh vertex spacing step = worldSize/(n-1), so SampleHeight at a mesh vertex's world XZ
// returns that vertex's grid height EXACTLY: the plant sits ON the mesh surface). The integer cell gx =
// gfx>>kFrac (clamped to [0, n-2] so the +1 neighbour stays in-grid), the fractional tx = gfx - (gx<<kFrac)
// in [0, kOne); the 4 grid corners grid[gz*n+gx{,+1}] / grid[(gz+1)*n+gx{,+1}] are blended bilinearly with
// fxmul. A degenerate grid (n<2 or size mismatch) -> 0 (flat). Pure integer; NO <cmath>, NO float, NO RNG.
inline fx SampleHeight(const std::vector<fx>& grid, int n, fx worldSize, fx x, fx z) {
    if (n < 2 || worldSize <= 0) return 0;
    if (grid.size() != static_cast<size_t>(n) * static_cast<size_t>(n)) return 0;

    // Fractional grid coordinates in Q16.16 (the (n-1)/worldSize scale aligns with the PT4 mesh spacing).
    fx gfx = static_cast<fx>((static_cast<int64_t>(x) * static_cast<int64_t>(n - 1)) /
                             static_cast<int64_t>(worldSize));
    fx gfz = static_cast<fx>((static_cast<int64_t>(z) * static_cast<int64_t>(n - 1)) /
                             static_cast<int64_t>(worldSize));
    // Clamp the fractional coords into the valid blend domain [0, (n-1)*kOne] so the integer cell stays in
    // [0, n-2] (the +1 neighbour in-grid) and the fraction stays in [0, kOne].
    const fx loFx = 0;
    const fx hiFx = static_cast<fx>(n - 1) << kFrac;   // (n-1).0 in Q16.16
    if (gfx < loFx) gfx = loFx; else if (gfx > hiFx) gfx = hiFx;
    if (gfz < loFx) gfz = loFx; else if (gfz > hiFx) gfz = hiFx;

    int gx = static_cast<int>(gfx >> kFrac);
    int gz = static_cast<int>(gfz >> kFrac);
    if (gx > n - 2) gx = n - 2;                         // keep gx+1 in-grid (gx already >= 0 by the clamp)
    if (gz > n - 2) gz = n - 2;
    const fx tx = gfx - (static_cast<fx>(gx) << kFrac); // fractional part in [0, kOne]
    const fx tz = gfz - (static_cast<fx>(gz) << kFrac);

    const size_t row0 = static_cast<size_t>(gz) * static_cast<size_t>(n);
    const size_t row1 = static_cast<size_t>(gz + 1) * static_cast<size_t>(n);
    const fx h00 = grid[row0 + static_cast<size_t>(gx)];
    const fx h10 = grid[row0 + static_cast<size_t>(gx + 1)];
    const fx h01 = grid[row1 + static_cast<size_t>(gx)];
    const fx h11 = grid[row1 + static_cast<size_t>(gx + 1)];
    // Bilinear blend (the PT1 IntValueNoise corner-blend shape; linear in fixed point).
    const fx a = h00 + fxmul(h10 - h00, tx);
    const fx b = h01 + fxmul(h11 - h01, tx);
    return a + fxmul(b - a, tz);
}

}  // namespace hf::terrain
