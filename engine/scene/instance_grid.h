#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include "math/math.h"
#include "scene/vertex.h"

namespace hf::scene {

// Build a deterministic GRID of per-instance model matrices for the Slice-Q instanced-rendering
// showcase. NO randomness — every transform is a pure function of the instance index, so the
// resulting frame is golden-stable across runs and backends. `n` x `n` instances are laid out on a
// regular grid centered at the origin on the XZ plane with `spacing` world units between cells; each
// instance gets a small per-index height bob and a per-index Y rotation derived from sin/cos of the
// index, plus a uniform `scale`. Returns n*n InstanceData records (column-major mat4 each), ordered
// row-major by (gx, gz) so the draw order is stable.
inline std::vector<InstanceData> BuildInstanceGrid(uint32_t n, float spacing, float scale) {
    using math::Mat4; using math::Vec3;
    std::vector<InstanceData> out;
    out.reserve((size_t)n * n);
    const float half = 0.5f * (float)(n - 1) * spacing;
    for (uint32_t gx = 0; gx < n; ++gx) {
        for (uint32_t gz = 0; gz < n; ++gz) {
            uint32_t idx = gx * n + gz;
            float x = (float)gx * spacing - half;
            float z = (float)gz * spacing - half;
            // Deterministic per-instance variation (no RNG): a height bob and a Y spin keyed off the
            // grid coords so neighbouring instances visibly differ.
            float phase = (float)gx * 0.6f + (float)gz * 0.9f;
            float y = scale + 0.35f * (0.5f + 0.5f * std::sin(phase));  // sit above ground, bobbing
            float yaw = 0.5f * std::sin((float)idx * 0.7f);
            Mat4 m = Mat4::Translate({x, y, z})
                   * Mat4::RotateY(yaw)
                   * Mat4::Scale({scale, scale, scale});
            InstanceData d;
            for (int k = 0; k < 16; ++k) d.model[k] = m.m[k];
            out.push_back(d);
        }
    }
    return out;
}

} // namespace hf::scene
