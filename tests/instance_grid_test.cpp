// Slice Q unit test: the per-instance transform layout (binding 1) + the deterministic instance-grid
// builder. Pure C++ (no GPU): validates the RHI VertexLayout the instanced pipeline declares and the
// golden-stability + placement of the CPU-built instance matrices.
#include "scene/vertex.h"
#include "scene/instance_grid.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

int main() {
    HF_TEST_MAIN_INIT();
    // --- Instance transform layout: mat4 as 4x RGBA32_Float at locations 7-10, stride 64. ---
    rhi::VertexLayout il = scene::InstanceTransformLayout();
    check(il.stride == 64, "instance stride == 64");
    check(il.attributes.size() == 4, "4 instance attributes");
    for (uint32_t k = 0; k < 4; ++k) {
        check(il.attributes[k].location == 7 + k, "instance locations 7..10");
        check(il.attributes[k].format == rhi::Format::RGBA32_Float, "instance attr is RGBA32_Float");
        check(il.attributes[k].offset == k * 16, "instance attr offset = k*16");
    }
    check(sizeof(scene::InstanceData) == 64, "InstanceData is 64 bytes");

    // --- Grid builder: 12x12 = 144 instances, deterministic. ---
    auto a = scene::BuildInstanceGrid(12, 1.3f, 0.45f);
    auto b = scene::BuildInstanceGrid(12, 1.3f, 0.45f);
    check(a.size() == 144, "12x12 grid -> 144 instances");
    // Determinism: two builds are byte-identical (golden-stable, no RNG).
    bool identical = (a.size() == b.size());
    for (size_t i = 0; i < a.size() && identical; ++i)
        for (int k = 0; k < 16; ++k)
            if (a[i].model[k] != b[i].model[k]) identical = false;
    check(identical, "grid build is deterministic across calls");

    // The grid is centered: for n=12, spacing 1.3, the first instance (gx=0,gz=0) sits at
    // x=z=-half = -0.5*(11)*1.3 = -7.15; the last (gx=11,gz=11) at +7.15. Translation lives in
    // column 3 (m[12]=x, m[14]=z) for the column-major model.
    const float half = 0.5f * 11.0f * 1.3f;
    check(approx(a.front().model[12], -half), "first instance x = -half");
    check(approx(a.front().model[14], -half), "first instance z = -half");
    check(approx(a.back().model[12], +half), "last instance x = +half");
    check(approx(a.back().model[14], +half), "last instance z = +half");
    // Every instance is lifted above the ground (y > 0) so spheres sit on the plane.
    bool allAbove = true;
    for (auto& d : a) if (d.model[13] <= 0.0f) allAbove = false;
    check(allAbove, "all instances lifted above ground (y>0)");
    // Distinct placement: first and last instance differ.
    check(!approx(a.front().model[12], a.back().model[12]), "instances occupy distinct positions");

    if (g_fail == 0) std::printf("instance_grid_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
