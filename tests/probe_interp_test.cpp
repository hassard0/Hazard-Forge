// Slice DL — DDGI Slice 4: Probe-Grid Update / Trilinear SH Interpolation. Pure CPU math: the FLOOR-cell
// 8-corner trilinear lookup (probegi::NearestProbes), the fma-blended SH at a world position
// (probesh::InterpolateSH), the cosine-lobe irradiance (probesh::InterpolateIrradiance), and the
// interp dispatch sizing (probegi::InterpDispatchGroups) — the EXACT trilinear blend the
// --probeinterp-shot showcase + the Metal --probeinterp showcase run and the probe_interp.comp shader
// copies VERBATIM, so the GPU blended-SH SSBO is BIT-EXACT to this CPU reference. No device, ASan-eligible
// (links hf_core).
//
// Properties pinned (per the spec §6):
//   * Partition of unity: Σ NearestProbes(p).w == 1 for random interior points.
//   * Lattice-point identity: NearestProbes(probePos(px,py,pz)) -> that corner has weight 1, the rest 0;
//     InterpolateSH(probePos(...)) == that probe's SH byte-identical.
//   * Boundary clamp: a far-outside point -> valid corner indices (in range), weights still sum 1.
//   * dim==1 axis degeneracy: a 1-thick axis -> frac 0 on it, no out-of-range index, weights sum 1.
//   * Disabled guard: spacing<=0 / dim<=0 -> valid=false -> InterpolateSH returns the zero SH;
//     InterpDispatchGroups 0 at any zero dim / probeCount 0.
//   * Blend linearity: 8 identical SH -> that SH; a 2-probe-bright case shifts the blend toward it.
//   * Determinism: same inputs -> bit-identical blend.
#include "render/probe_gi.h"
#include "render/probe_sh.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace probegi = hf::render::probegi;
namespace probesh = hf::render::probesh;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }

// Build a deterministic ProbeSH[probeCount] for a grid: each probe gets a distinct, reproducible set of
// 27 coeffs derived from its flat index (no RNG) so a byte-identical blend is meaningful.
static std::vector<probesh::ProbeSH> MakeProbes(const probegi::ProbeGrid& grid) {
    int n = grid.probeCount();
    std::vector<probesh::ProbeSH> probes((size_t)(n > 0 ? n : 0));
    for (int p = 0; p < n; ++p) {
        for (int i = 0; i < 9; ++i)
            for (int c = 0; c < 3; ++c)
                probes[p].coeff[i][c] = 0.01f * (float)(p + 1) + 0.1f * (float)i + 0.3f * (float)c;
    }
    return probes;
}

int main() {
    probegi::ProbeGrid grid;
    grid.origin = Vec3{-2.0f, 0.0f, -2.0f};
    grid.dimX = 4; grid.dimY = 3; grid.dimZ = 4; grid.spacing = 2.0f;
    std::vector<probesh::ProbeSH> probes = MakeProbes(grid);
    const int probeCount = grid.probeCount();

    // ====================================================================================
    // Partition of unity: Σ NearestProbes(p).w == 1 for interior points; indices in range.
    // ====================================================================================
    {
        // A deterministic spread of interior query points (no RNG): step fractionally across cells.
        bool allSumOne = true, allInRange = true;
        for (int qz = 0; qz < 7; ++qz)
            for (int qy = 0; qy < 5; ++qy)
                for (int qx = 0; qx < 7; ++qx) {
                    Vec3 q{grid.origin.x + 0.37f * (float)qx * grid.spacing,
                           grid.origin.y + 0.41f * (float)qy * grid.spacing,
                           grid.origin.z + 0.53f * (float)qz * grid.spacing};
                    probegi::ProbeTrilinear t = probegi::NearestProbes(q, grid);
                    check(t.valid, "interior NearestProbes valid");
                    float s = 0.0f;
                    for (int c = 0; c < 8; ++c) {
                        s += t.w[c];
                        if (t.idx[c] < 0 || t.idx[c] >= probeCount) allInRange = false;
                        if (t.w[c] < -1e-6f || t.w[c] > 1.0f + 1e-6f) allSumOne = false;
                    }
                    if (!approx(s, 1.0f, 1e-5f)) allSumOne = false;
                }
        check(allSumOne, "partition of unity: Σw == 1 over interior points (weights in [0,1])");
        check(allInRange, "interior corner indices all in [0, probeCount)");
    }

    // ====================================================================================
    // Lattice-point identity: at a probe position, that corner weight == 1, rest 0; the blend == that
    // probe's SH byte-identical.
    // ====================================================================================
    {
        bool allIdentity = true;
        for (int pz = 0; pz < grid.dimZ; ++pz)
            for (int py = 0; py < grid.dimY; ++py)
                for (int px = 0; px < grid.dimX; ++px) {
                    Vec3 pp = grid.probePos(px, py, pz);
                    int flat = grid.flatIndex(px, py, pz);
                    probegi::ProbeTrilinear t = probegi::NearestProbes(pp, grid);
                    // The corner whose flat index == this probe must have weight ~1; all others ~0.
                    float wThis = 0.0f, wOther = 0.0f;
                    for (int c = 0; c < 8; ++c) {
                        if (t.idx[c] == flat) wThis += t.w[c];
                        else wOther += t.w[c];
                    }
                    if (!approx(wThis, 1.0f, 1e-5f) || !approx(wOther, 0.0f, 1e-5f)) allIdentity = false;

                    // InterpolateSH at the probe position == that probe's SH byte-identical.
                    probesh::ProbeSH blend = probesh::InterpolateSH(pp, grid, probes.data(), probeCount);
                    if (std::memcmp(&blend, &probes[flat], sizeof(probesh::ProbeSH)) != 0) allIdentity = false;
                }
        check(allIdentity, "lattice-point identity: NearestProbes weight 1 + InterpolateSH == probe SH (byte-identical)");
    }

    // ====================================================================================
    // Boundary clamp: a far-outside point -> valid indices, weights sum 1, clamps to a boundary corner.
    // ====================================================================================
    {
        Vec3 farPos{grid.origin.x + 1000.0f, grid.origin.y - 1000.0f, grid.origin.z + 1000.0f};
        probegi::ProbeTrilinear t = probegi::NearestProbes(farPos, grid);
        check(t.valid, "far-outside NearestProbes valid");
        float s = 0.0f; bool inRange = true;
        for (int c = 0; c < 8; ++c) {
            s += t.w[c];
            if (t.idx[c] < 0 || t.idx[c] >= probeCount) inRange = false;
        }
        check(inRange, "far-outside corner indices all in range");
        check(approx(s, 1.0f, 1e-5f), "far-outside weights still sum to 1 (frac saturates)");
        // The blend equals SOME single boundary probe (saturated frac -> one corner weight 1).
        probesh::ProbeSH blend = probesh::InterpolateSH(farPos, grid, probes.data(), probeCount);
        bool matchesAProbe = false;
        for (int p = 0; p < probeCount; ++p)
            if (std::memcmp(&blend, &probes[p], sizeof(probesh::ProbeSH)) == 0) matchesAProbe = true;
        check(matchesAProbe, "far-outside blend clamps to a single boundary probe's SH");
    }

    // ====================================================================================
    // dim==1 axis degeneracy: a 1-thick axis -> frac 0 on it, no out-of-range index, weights sum 1.
    // ====================================================================================
    {
        probegi::ProbeGrid flat = grid;
        flat.dimY = 1;   // collapse Y
        std::vector<probesh::ProbeSH> fprobes = MakeProbes(flat);
        int fcount = flat.probeCount();
        Vec3 q{flat.origin.x + 0.6f * flat.spacing, flat.origin.y + 5.0f, flat.origin.z + 0.6f * flat.spacing};
        probegi::ProbeTrilinear t = probegi::NearestProbes(q, flat);
        check(t.valid, "dim==1 NearestProbes valid");
        float s = 0.0f; bool inRange = true;
        // The +y corners (c with bit1 set) must share the SAME flat index as their -y partner (no Y blend).
        bool noYBlend = true;
        for (int c = 0; c < 8; ++c) {
            s += t.w[c];
            if (t.idx[c] < 0 || t.idx[c] >= fcount) inRange = false;
            int partner = c ^ 2;   // toggle the y bit
            if (t.idx[c] != t.idx[partner]) noYBlend = false;
        }
        check(inRange, "dim==1 corner indices all in range");
        check(approx(s, 1.0f, 1e-5f), "dim==1 weights sum to 1");
        check(noYBlend, "dim==1: the degenerate axis contributes no second corner (frac 0)");
    }

    // ====================================================================================
    // Disabled guard: spacing<=0 / dim<=0 -> valid=false, idx all 0, w[0]=1; InterpolateSH -> zero SH.
    // ====================================================================================
    {
        probesh::ProbeSH zeroRef{}; std::memset(&zeroRef, 0, sizeof(zeroRef));

        probegi::ProbeGrid badSpacing = grid; badSpacing.spacing = 0.0f;
        probegi::ProbeTrilinear t1 = probegi::NearestProbes(Vec3{0, 0, 0}, badSpacing);
        check(!t1.valid && t1.idx[0] == 0 && approx(t1.w[0], 1.0f),
              "spacing<=0 -> valid=false, idx[0]=0, w[0]=1");
        probesh::ProbeSH b1 = probesh::InterpolateSH(Vec3{0, 0, 0}, badSpacing, probes.data(), probeCount);
        check(std::memcmp(&b1, &zeroRef, sizeof(probesh::ProbeSH)) == 0,
              "spacing<=0 -> InterpolateSH zero SH");

        probegi::ProbeGrid badDim = grid; badDim.dimX = 0;
        probegi::ProbeTrilinear t2 = probegi::NearestProbes(Vec3{0, 0, 0}, badDim);
        check(!t2.valid, "dimX==0 -> valid=false");
        probesh::ProbeSH b2 = probesh::InterpolateSH(Vec3{0, 0, 0}, badDim, probes.data(), probeCount);
        check(std::memcmp(&b2, &zeroRef, sizeof(probesh::ProbeSH)) == 0,
              "dimX==0 -> InterpolateSH zero SH");

        // probeCount<=0 guard on the SH path -> zero SH even with a valid grid.
        probesh::ProbeSH b3 = probesh::InterpolateSH(grid.probePos(0, 0, 0), grid, probes.data(), 0);
        check(std::memcmp(&b3, &zeroRef, sizeof(probesh::ProbeSH)) == 0,
              "probeCount<=0 -> InterpolateSH zero SH");

        // InterpolateIrradiance on the disabled path -> {0,0,0}.
        Vec3 irr = probesh::InterpolateIrradiance(Vec3{0, 0, 0}, badDim, probes.data(), probeCount, Vec3{0, 1, 0});
        check(approx(irr.x, 0.0f) && approx(irr.y, 0.0f) && approx(irr.z, 0.0f),
              "disabled InterpolateIrradiance -> {0,0,0}");

        // InterpDispatchGroups: 0 at any zero dim / probeCount 0 / nQueries<=0; ceil(n/64) otherwise.
        check(probegi::InterpDispatchGroups(grid, 100) == 2, "InterpDispatchGroups(100) == ceil(100/64) == 2");
        check(probegi::InterpDispatchGroups(grid, 64) == 1, "InterpDispatchGroups(64) == 1");
        check(probegi::InterpDispatchGroups(grid, 0) == 0, "InterpDispatchGroups(nQueries=0) == 0");
        check(probegi::InterpDispatchGroups(grid, -5) == 0, "InterpDispatchGroups(nQueries<0) == 0");
        check(probegi::InterpDispatchGroups(badDim, 100) == 0, "InterpDispatchGroups(probeCount=0) == 0");
    }

    // ====================================================================================
    // Blend linearity: 8 identical SH -> that SH; a 2-probe-bright case shifts the blend toward it.
    // ====================================================================================
    {
        // All probes identical -> blend at any interior point == that SH exactly.
        std::vector<probesh::ProbeSH> uniform((size_t)probeCount);
        probesh::ProbeSH one{};
        for (int i = 0; i < 9; ++i) for (int c = 0; c < 3; ++c) one.coeff[i][c] = 0.5f + 0.01f * (float)i;
        for (int p = 0; p < probeCount; ++p) uniform[p] = one;
        Vec3 mid{grid.origin.x + 0.5f * grid.spacing, grid.origin.y + 0.5f * grid.spacing,
                 grid.origin.z + 0.5f * grid.spacing};
        probesh::ProbeSH blendU = probesh::InterpolateSH(mid, grid, uniform.data(), probeCount);
        check(std::memcmp(&blendU, &one, sizeof(probesh::ProbeSH)) == 0,
              "blend linearity: 8 identical SH -> that SH exactly");

        // A bright probe: move the query closer to a bright corner -> the blend's coeff[0][0] rises
        // monotonically toward the bright value.
        std::vector<probesh::ProbeSH> mixed((size_t)probeCount);
        std::memset(mixed.data(), 0, mixed.size() * sizeof(probesh::ProbeSH));
        int brightFlat = grid.flatIndex(1, 1, 1);
        mixed[brightFlat].coeff[0][0] = 1.0f;   // a single bright DC coeff
        Vec3 near = grid.probePos(1, 1, 1);
        Vec3 mid2 = grid.probePos(1, 1, 1) + (grid.probePos(2, 1, 1) - grid.probePos(1, 1, 1)) * 0.25f;
        Vec3 far = grid.probePos(1, 1, 1) + (grid.probePos(2, 1, 1) - grid.probePos(1, 1, 1)) * 0.75f;
        float bNear = probesh::InterpolateSH(near, grid, mixed.data(), probeCount).coeff[0][0];
        float bMid = probesh::InterpolateSH(mid2, grid, mixed.data(), probeCount).coeff[0][0];
        float bFar = probesh::InterpolateSH(far, grid, mixed.data(), probeCount).coeff[0][0];
        check(approx(bNear, 1.0f, 1e-5f), "blend at the bright probe == its value");
        check(bMid > bFar + 1e-4f && bNear > bMid + 1e-4f,
              "blend falls monotonically moving away from the bright probe");
    }

    // ====================================================================================
    // Determinism: same inputs -> bit-identical blend (no RNG/time).
    // ====================================================================================
    {
        Vec3 q{grid.origin.x + 1.3f, grid.origin.y + 0.7f, grid.origin.z + 2.1f};
        probesh::ProbeSH a = probesh::InterpolateSH(q, grid, probes.data(), probeCount);
        probesh::ProbeSH b = probesh::InterpolateSH(q, grid, probes.data(), probeCount);
        check(std::memcmp(&a, &b, sizeof(probesh::ProbeSH)) == 0, "InterpolateSH deterministic (bit-identical)");
    }

    if (g_fail == 0) std::printf("probe_interp_test: ALL PASS\n");
    else std::printf("probe_interp_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
