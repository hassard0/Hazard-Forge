#pragma once
// Slice PT2 — Integer hydraulic erosion (engine/terrain/erosion.h), THE HEADLINE + CRUX of FLAGSHIP #26
// (DETERMINISTIC PROCEDURAL TERRAIN, hf::terrain). Pure CPU, header-only. NO device, NO backend symbols,
// NO new RHI, NO new shader. Namespace hf::terrain.
//
// The moat: a DETERMINISTIC, bit-identical RUNTIME integer hydraulic erosion. A sediment-transport flux
// that moves material downhill, iterated, carving drainage valleys into the PT1 GenHeightField output.
// A seed + iteration count -> the EXACT same eroded terrain on every machine (UE5 landscape erosion is an
// offline non-deterministic bake). Erodes the PT1 heightfield in place; PT1 (procterrain.h) is read-only.
//
// THE CRUX — fixed-point stability. Per iteration, in PINNED row-major (Gauss-Seidel) cell order (the
// fpx::SolveContacts order-determinism discipline), for each cell find its LOWEST in-grid 4-neighbour,
// dh = h[cell] - h[low]; if dh > 0, move erode = fxmul(dh, kErodeRate) of height FROM the cell TO that
// neighbour: h[cell] -= erode; h[low] += erode. Two guarantees:
//   * EXACT mass conservation — the SAME erode value is subtracted from one cell and added to another, so
//     the grid sum is preserved bit-for-bit (assert sum(eroded) == sum(base) EXACTLY, no tolerance fudge).
//   * No oscillation / no inversion — with kErodeRate <= kOne/4 the post-move gap is dh - 2*erode =
//     dh*(1 - 2*rate); at rate = kOne/8 the gap is dh*(1 - 1/4) = 3*dh/4 > 0, so the cell stays ABOVE its
//     neighbour — the gradient strictly shrinks, never flips. Provably stable.
//
// WHY IT IS BIT-EXACT (the cross-backend crux): every op is a pure int32/int64 fixed-point op. fxmul's
// truncating >> is deterministic on every backend (>> truncates identically) -> bit-identical
// CPU<->Vulkan<->Metal. CPU-host (the int64 fxmul is Vulkan-DXC-only in shaders — keep erosion host-
// evaluated, the FO1-FO4 / FPX1 lesson). NO <cmath>, NO float, NO clock/RNG.
//
// REUSE MAP: fx/kOne/kFrac/fxmul come from engine/sim/fpx.h (read-only — the Q16.16 toolbox), via
// terrain/procterrain.h (read-only — PT1's IntHeight/GenHeightField). procterrain.h PT1 logic is FROZEN.

#include <cstdint>
#include <vector>

#include "sim/fpx.h"            // fx / kOne / kFrac / fxmul (Q16.16 toolbox), read-only.
#include "terrain/procterrain.h"  // PT1 GenHeightField/IntHeight (read-only); pulls fx/kOne/fxmul too.

namespace hf::terrain {

// The provably-stable erosion rate (kOne/8 <= kOne/4 -> the post-move gap 3*dh/4 > 0, no inversion).
inline constexpr fx kErodeRate = kOne / 8;

// ErodeHydraulic(grid, n, iterations): the mass-conserving contractive integer flux above. `iterations`
// passes, each a PINNED row-major (Gauss-Seidel) sweep over the n x n grid — IN PLACE so later cells in
// the sweep see the updated values. For each cell (gx,gz) scan its 4 in-grid neighbours (up/down/left/
// right) for the LOWEST; dh = h[cell] - h[low]; if dh > 0 move erode = fxmul(dh, kErodeRate) from the cell
// to that neighbour (h[cell] -= erode; h[low] += erode — the SAME erode -> exact mass conservation).
// iterations <= 0 -> no change (the no-op). Pure integer; NO <cmath>, NO float, NO clock/RNG.
inline void ErodeHydraulic(std::vector<fx>& grid, int n, int iterations) {
    if (n <= 0 || iterations <= 0) return;                       // the no-op (iterations<=0 -> unchanged)
    if (grid.size() != static_cast<size_t>(n) * static_cast<size_t>(n)) return;
    for (int it = 0; it < iterations; ++it) {
        for (int gz = 0; gz < n; ++gz) {
            for (int gx = 0; gx < n; ++gx) {
                const size_t ci = static_cast<size_t>(gz) * static_cast<size_t>(n) + static_cast<size_t>(gx);
                const fx hc = grid[ci];
                // Find the LOWEST in-grid 4-neighbour (pinned scan order: up, down, left, right).
                fx     lowH = hc;
                size_t lowI = ci;
                if (gz > 0) {                                    // up
                    const size_t ni = ci - static_cast<size_t>(n);
                    if (grid[ni] < lowH) { lowH = grid[ni]; lowI = ni; }
                }
                if (gz + 1 < n) {                                // down
                    const size_t ni = ci + static_cast<size_t>(n);
                    if (grid[ni] < lowH) { lowH = grid[ni]; lowI = ni; }
                }
                if (gx > 0) {                                    // left
                    const size_t ni = ci - 1;
                    if (grid[ni] < lowH) { lowH = grid[ni]; lowI = ni; }
                }
                if (gx + 1 < n) {                                // right
                    const size_t ni = ci + 1;
                    if (grid[ni] < lowH) { lowH = grid[ni]; lowI = ni; }
                }
                const fx dh = hc - lowH;                         // >= 0 (lowH is the min including hc)
                if (dh > 0) {
                    const fx erode = fxmul(dh, kErodeRate);      // SAME value moved -> exact conservation
                    grid[ci]   -= erode;
                    grid[lowI] += erode;
                }
            }
        }
    }
}

// GridSum(grid): an int64 accumulate of the whole field (for the mass-conservation proof — the per-cell
// fx values stay in the Q16.16 world band, but the sum over n*n cells needs int64 headroom). Pure integer.
inline int64_t GridSum(const std::vector<fx>& grid) {
    int64_t s = 0;
    for (fx h : grid) s += static_cast<int64_t>(h);
    return s;
}

// CountChanged(a, b): the number of cells that differ between two equal-length fields (the changed-cell
// counter for the carving proof). Pure integer.
inline int CountChanged(const std::vector<fx>& a, const std::vector<fx>& b) {
    int c = 0;
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) ++c;
    return c;
}

}  // namespace hf::terrain
