// Slice SC5 — FOLIAGE SCATTER AT SCALE: the pure-CPU composition header engine/render/sc5_foliage.h
// (PCG scatter -> deterministic wind bend -> integer distance-LOD -> the per-LOD instance transform
// buffers) pinned at 10k+ instances. THE SCALE PROOFS:
//   1. COUNT PINS: 12,123 plants survive the overlap prune; per-LOD 1312/5181/4610/1020 -> 11,103
//      DRAWN (>= 10,000 through the instanced path).
//   2. DIGEST PINS (cross-compiler): the INTEGER plant digest (bit-exact FO2-FO4 data — the
//      cross-platform pin) and the FLOAT transform-buffer digest (the exact bytes the instanced
//      draws consume — the x64 MSVC+clang pin; the LUT-based lean keeps libm out of the transform
//      build, so the float bytes ARE cross-compiler pinnable — the FO-A gap fix). This same test
//      compiles standalone under LLVM clang++ and must print/assert the SAME pins (verified at
//      slice time; the pins in sc5_foliage.h were produced by both compilers).
//   3. WIND IS LIVE: two different fixed frames (120 vs 121) produce DIFFERENT pinned digests for
//      BOTH artifacts — the wind bend is load-bearing in the transforms, not baked decor.
//   4. DETERMINISM: two full pipeline runs are byte-identical (vector-level memcmp).
//   5. SHUFFLE-INVARIANCE AT 10k (the PCG5 property re-asserted at scale): PruneOverlaps of the
//      REVERSED pre-prune instance list returns the identical survivor sequence (the canonical
//      z/x-sort makes the greedy prune input-order-independent). The unit-scale property is
//      pcg_test's; this re-asserts it at the 18,496-candidate scale SC5 actually runs.
//   6. LOD SANITY: every bucket populated, buckets partition the plant set, culled plants are
//      exactly those beyond farR (spot-checked via FoliageLod monotonicity on the pinned counts).
//   7. EMPTY NO-OP: cellsX=0 -> zero plants -> zero transforms -> the empty digest.
// Pure C++ (hf_core), no RHI/device, ASan-eligible.
#include "render/sc5_foliage.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace sc5 = hf::render::sc5;
namespace fol = hf::foliage;
namespace pcg = hf::pcg;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();
    const sc5::Sc5Config cfg = sc5::Sc5DefaultConfig();

    // --- The full 10k pipeline at THE showcase frame. -----------------------------------------------
    std::vector<fol::FoliageInstance> plants = sc5::Sc5BuildPlants(cfg, sc5::kSc5Frame);
    const sc5::Sc5RenderSet rs = sc5::Sc5BuildRenderSet(plants, cfg);

    // (1) count pins: survivors + per-LOD buckets + the >= 10k drawn headline.
    check((uint32_t)plants.size() == sc5::kSc5ExpectedPlants, "plant count == 12123 pin");
    check(rs.counts[0] == sc5::kSc5ExpectedNear,   "LOD0 (near) count == 1312 pin");
    check(rs.counts[1] == sc5::kSc5ExpectedMid,    "LOD1 (mid) count == 5181 pin");
    check(rs.counts[2] == sc5::kSc5ExpectedFar,    "LOD2 (far) count == 4610 pin");
    check(rs.counts[3] == sc5::kSc5ExpectedCulled, "LOD3 (culled) count == 1020 pin");
    check(rs.Drawn() == sc5::kSc5ExpectedDrawn,    "drawn == 11103 pin");
    check(rs.Drawn() >= 10000u,                    "drawn >= 10,000 (the scale headline)");
    check(rs.counts[0] + rs.counts[1] + rs.counts[2] + rs.counts[3] == (uint32_t)plants.size(),
          "LOD buckets partition the plant set");
    check(rs.lod[0].size() == rs.counts[0] && rs.lod[1].size() == rs.counts[1] &&
          rs.lod[2].size() == rs.counts[2], "per-bucket transform lists match the counts");

    // (2) digest pins: the integer plant digest (cross-platform) + the float transform-buffer digest
    //     (x64 cross-compiler — the FO-A LUT lean keeps libm out, see sc5_foliage.h).
    const uint64_t dPlant = sc5::Sc5PlantDigest(plants);
    const uint64_t dXform = sc5::Sc5TransformDigest(rs);
    std::printf("sc5: frame %u plantDigest 0x%016llx xformDigest 0x%016llx\n",
                sc5::kSc5Frame, (unsigned long long)dPlant, (unsigned long long)dXform);
    check(dPlant == sc5::kSc5PlantDigestF120, "plant digest (frame 120) == pinned");
    check(dXform == sc5::kSc5XformDigestF120, "transform-buffer digest (frame 120) == pinned");

    // (3) the wind is LIVE: frame 121 produces DIFFERENT pinned digests (wind is load-bearing).
    {
        std::vector<fol::FoliageInstance> plantsB = plants;   // same placement; re-wind + re-LOD
        fol::ApplyWind(plantsB, cfg.wind, sc5::kSc5FrameB);
        fol::AssignLods(plantsB, cfg.lodCam, cfg.nearR, cfg.farR);
        const sc5::Sc5RenderSet rsB = sc5::Sc5BuildRenderSet(plantsB, cfg);
        const uint64_t dPlantB = sc5::Sc5PlantDigest(plantsB);
        const uint64_t dXformB = sc5::Sc5TransformDigest(rsB);
        std::printf("sc5: frame %u plantDigest 0x%016llx xformDigest 0x%016llx\n",
                    sc5::kSc5FrameB, (unsigned long long)dPlantB, (unsigned long long)dXformB);
        check(dPlantB == sc5::kSc5PlantDigestF121, "plant digest (frame 121) == pinned");
        check(dXformB == sc5::kSc5XformDigestF121, "transform-buffer digest (frame 121) == pinned");
        check(dPlantB != dPlant, "two wind frames -> different plant digests (wind is live)");
        check(dXformB != dXform, "two wind frames -> different transform digests (wind is live)");
        // Placement itself must NOT move under wind: only bend (and possibly nothing else) changes.
        bool placementStable = plantsB.size() == plants.size();
        for (size_t i = 0; placementStable && i < plants.size(); ++i)
            placementStable = std::memcmp(&plants[i].base, &plantsB[i].base,
                                          sizeof(pcg::PcgInstance)) == 0;
        check(placementStable, "wind only re-annotates bend — placement bytes unchanged");
    }

    // (4) determinism: a second full pipeline run is byte-identical.
    {
        std::vector<fol::FoliageInstance> again = sc5::Sc5BuildPlants(cfg, sc5::kSc5Frame);
        check(again.size() == plants.size() &&
              std::memcmp(again.data(), plants.data(),
                          plants.size() * sizeof(fol::FoliageInstance)) == 0,
              "two pipeline runs byte-identical");
    }

    // (5) shuffle-invariance at 10k: PruneOverlaps(REVERSED pre-prune list) == the same survivors.
    //     (The PCG5 unit property re-asserted at the 18,496-candidate scale SC5 runs.)
    {
        const std::vector<pcg::FxVec3> pts = pcg::ScatterMasked(
            cfg.stream, cfg.field.graph.area, cfg.field.graph.cellsX, cfg.field.graph.cellsZ,
            cfg.field.graph.mask, cfg.field.graph.density);
        const std::vector<pcg::PcgInstance> pre =
            pcg::BuildInstances(pts, cfg.stream, cfg.field.graph.transform);
        std::vector<pcg::PcgInstance> rev(pre.rbegin(), pre.rend());
        const std::vector<pcg::PcgInstance> a =
            pcg::PruneOverlaps(pre, cfg.field.graph.pruneRadius);
        const std::vector<pcg::PcgInstance> b =
            pcg::PruneOverlaps(rev, cfg.field.graph.pruneRadius);
        check(a.size() == b.size() && a.size() == plants.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(pcg::PcgInstance)) == 0,
              "PruneOverlaps shuffle-invariant at 10k (reversed input -> identical survivors)");
    }

    // (6) LOD sanity: culled == beyond farR, and the pick is the FoliageLod of each plant.
    {
        bool lodConsistent = true;
        for (const fol::FoliageInstance& p : plants)
            if (p.lod != fol::FoliageLod(p.base.pos, cfg.lodCam, cfg.nearR, cfg.farR))
                { lodConsistent = false; break; }
        check(lodConsistent, "every plant.lod == FoliageLod(pos, cam, nearR, farR)");
        check(rs.counts[0] > 0 && rs.counts[1] > 0 && rs.counts[2] > 0 && rs.counts[3] > 0,
              "all four LOD buckets populated (near/mid/far/culled)");
    }

    // (7) empty no-op: cellsX=0 -> zero plants -> zero transforms -> the empty digest.
    {
        sc5::Sc5Config empty = cfg;
        empty.field.graph.cellsX = 0;
        const std::vector<fol::FoliageInstance> none = sc5::Sc5BuildPlants(empty, sc5::kSc5Frame);
        const sc5::Sc5RenderSet rsNone = sc5::Sc5BuildRenderSet(none, empty);
        check(none.empty() && rsNone.Drawn() == 0 && rsNone.counts[3] == 0,
              "empty graph -> zero plants, zero transforms");
        check(sc5::Sc5TransformDigest(rsNone) == net::DigestBytes(nullptr, 0),
              "empty render set -> the empty digest");
    }

    if (g_fail == 0)
        std::printf("sc5_foliage_test PASSED: {plants:%u, perLod:%u/%u/%u/%u, drawn:%u, "
                    "plantDigest:0x%016llx, xformDigest:0x%016llx}\n",
                    (uint32_t)plants.size(), rs.counts[0], rs.counts[1], rs.counts[2], rs.counts[3],
                    rs.Drawn(), (unsigned long long)dPlant, (unsigned long long)dXform);
    else
        std::printf("sc5_foliage_test FAILED (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
