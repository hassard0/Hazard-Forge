// Slice ML1 — MANY-LIGHT RENDERING (Track-S S7). Pure CPU math over engine/render/manylight.h:
// the 128-hashed-light field + its render::clustered assignment (the SAME
// BuildShowcaseAssignment the --manylight-shot showcases call on Vulkan AND Metal), pinning the
// FNV-1a-64 assignment digest + the cluster stats. No device, ASan-eligible (links hf_core).
// Existing tests/clustered_test.cpp is UNTOUCHED — this file only ADDS coverage.
//
// PINNED VALUES: kPinnedDigest / kPinnedMaxPerCluster / kPinnedTotalAssignments were captured from
// the MSVC x64 release build and verified IDENTICAL under clang-on-Windows (same UCRT libm, no FMA
// contraction at the x86-64 baseline). The Metal showcase prints the Mac's value in its stat line,
// so any cross-platform libm last-ULP divergence in the pow/log slice math would surface there
// explicitly rather than pass silently.
#include "render/manylight.h"
#include "render/clustered.h"
#include "math/math.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace cl = hf::render::clustered;
namespace ml = hf::render::manylight;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// The pinned deterministic artifacts of the 128-light showcase assignment (1280x720 camera).
static const uint64_t kPinnedDigest           = 0x42d5535632f152eaull;
static const uint32_t kPinnedMaxPerCluster    = 40u;
static const uint64_t kPinnedTotalAssignments = 4056ull;

int main() {
    HF_TEST_MAIN_INIT();

    // ---- The hashed light field is deterministic + in-range. ----
    {
        std::vector<cl::Light> lights = ml::MakeManyLights(ml::kShowcaseSeed, ml::kShowcaseLights);
        check((int)lights.size() == 128, "MakeManyLights returns exactly 128 lights");
        bool inRange = true;
        for (const cl::Light& L : lights) {
            if (L.viewPos.x < -17.0f || L.viewPos.x > 17.0f) inRange = false;   // spanX/2
            if (L.viewPos.z < -15.0f || L.viewPos.z > 11.0f) inRange = false;   // spanZ/2 - 2
            if (L.viewPos.y < 0.8f || L.viewPos.y >= 2.4f) inRange = false;
            if (L.radius < 3.0f || L.radius >= 6.0f) inRange = false;
            if (L.color.x < 0.15f || L.color.x > 1.0f) inRange = false;
            if (L.color.y < 0.15f || L.color.y > 1.0f) inRange = false;
            if (L.color.z < 0.15f || L.color.z > 1.0f) inRange = false;
            if (L.intensity != 2.6f) inRange = false;
        }
        check(inRange, "every hashed light is inside the authored field/radius/color bands");

        // Two generations are byte-identical (pure hash — no hidden state).
        std::vector<cl::Light> again = ml::MakeManyLights(ml::kShowcaseSeed, ml::kShowcaseLights);
        check(again.size() == lights.size() &&
                  std::memcmp(lights.data(), again.data(),
                              lights.size() * sizeof(cl::Light)) == 0,
              "MakeManyLights twice -> byte-identical light array");

        // A different seed produces a different field (the hash actually keys on the seed).
        std::vector<cl::Light> other = ml::MakeManyLights(ml::kShowcaseSeed + 1u, ml::kShowcaseLights);
        check(std::memcmp(lights.data(), other.data(),
                          lights.size() * sizeof(cl::Light)) != 0,
              "different seed -> different light field");
    }

    // ---- The SSBO capacity cap: 1024 lights build a valid deterministic assignment too. ----
    {
        std::vector<cl::Light> big = ml::MakeManyLights(ml::kShowcaseSeed, ml::kMaxManyLights);
        check((int)big.size() == 1024, "kMaxManyLights field builds (N=1024)");
        ml::ShowcaseAssignment a = ml::BuildShowcaseAssignment(ml::kShowcaseSeed,
                                                               ml::kMaxManyLights, 1280.0f, 720.0f);
        ml::ShowcaseAssignment b = ml::BuildShowcaseAssignment(ml::kShowcaseSeed,
                                                               ml::kMaxManyLights, 1280.0f, 720.0f);
        check(a.digest == b.digest, "1024-light assignment digest is run-stable");
        check(a.stats.totalAssignments > 0, "1024-light assignment is non-empty");
    }

    // ---- The showcase assignment (128 lights, 1280x720): pinned digest + pinned cluster stats,
    //      two runs byte-identical, and structural consistency against clustered.h. ----
    {
        ml::ShowcaseAssignment sa = ml::BuildShowcaseAssignment(
            ml::kShowcaseSeed, ml::kShowcaseLights, 1280.0f, 720.0f);
        check((int)sa.buffers.clusters.size() == 16 * 9 * 24, "clusters array sized 16x9x24");
        check(sa.buffers.lights.size() == 128, "one GpuLight per input light");

        // Determinism: a SECOND full assignment is byte-identical in every buffer.
        ml::ShowcaseAssignment sb = ml::BuildShowcaseAssignment(
            ml::kShowcaseSeed, ml::kShowcaseLights, 1280.0f, 720.0f);
        check(sa.buffers.clusters.size() == sb.buffers.clusters.size() &&
                  std::memcmp(sa.buffers.clusters.data(), sb.buffers.clusters.data(),
                              sa.buffers.clusters.size() * sizeof(cl::GpuCluster)) == 0,
              "two runs -> byte-identical cluster array");
        check(sa.buffers.lightIndices.size() == sb.buffers.lightIndices.size() &&
                  std::memcmp(sa.buffers.lightIndices.data(), sb.buffers.lightIndices.data(),
                              sa.buffers.lightIndices.size() * sizeof(uint32_t)) == 0,
              "two runs -> byte-identical light-index array");
        check(sa.digest == sb.digest, "two runs -> identical digest");

        // Structural: offsets are the prefix sum; every listed light overlaps its cluster's AABB.
        uint32_t expectOffset = 0;
        uint64_t sumCount = 0;
        bool offsetsOk = true, overlapsOk = true;
        for (size_t c = 0; c < sa.buffers.clusters.size(); ++c) {
            if (sa.buffers.clusters[c].offset != expectOffset) offsetsOk = false;
            expectOffset += sa.buffers.clusters[c].count;
            sumCount += sa.buffers.clusters[c].count;
            int cz = (int)(c / (sa.grid.cx * sa.grid.cy));
            int rem = (int)(c % (sa.grid.cx * sa.grid.cy));
            int cy = rem / sa.grid.cx;
            int cx = rem % sa.grid.cx;
            for (uint32_t k = 0; k < sa.buffers.clusters[c].count; ++k) {
                uint32_t li = sa.buffers.lightIndices[sa.buffers.clusters[c].offset + k];
                if (li >= sa.viewLights.size() ||
                    !cl::LightOverlapsCluster(sa.grid, sa.viewLights[li], cx, cy, cz))
                    overlapsOk = false;
            }
        }
        check(offsetsOk, "offsets == running prefix sum of counts");
        check(overlapsOk, "every listed light overlaps its cluster's AABB");
        check(sumCount == sa.stats.totalAssignments, "stats.totalAssignments == sum(count)");
        check(sa.stats.maxPerCluster > 0 && sa.stats.maxPerCluster <= 128,
              "maxPerCluster in (0, 128]");
        // 128 lights over a 16x9x24 grid MUST light many clusters (the whole point of the slice).
        check(sa.stats.totalAssignments > 128, "lights land in many clusters (assignments >> N)");

        // The digest is DigestBytes over clusters||lightIndices (the FnvAppend identity).
        {
            std::vector<uint8_t> cat;
            cat.resize(sa.buffers.clusters.size() * sizeof(cl::GpuCluster) +
                       sa.buffers.lightIndices.size() * sizeof(uint32_t));
            std::memcpy(cat.data(), sa.buffers.clusters.data(),
                        sa.buffers.clusters.size() * sizeof(cl::GpuCluster));
            std::memcpy(cat.data() + sa.buffers.clusters.size() * sizeof(cl::GpuCluster),
                        sa.buffers.lightIndices.data(),
                        sa.buffers.lightIndices.size() * sizeof(uint32_t));
            check(hf::net::DigestBytes(cat.data(), cat.size()) == sa.digest,
                  "DigestAssignment == DigestBytes(clusters||lightIndices)");
        }

        // ---- THE PINS (captured MSVC x64, verified identical under local clang). ----
        std::printf("manylight showcase assignment: lights:%d maxPerCluster:%u "
                    "totalAssignments:%llu digest:0x%016llx\n",
                    ml::kShowcaseLights, sa.stats.maxPerCluster,
                    (unsigned long long)sa.stats.totalAssignments,
                    (unsigned long long)sa.digest);
        check(sa.digest == kPinnedDigest, "assignment digest matches the pinned value");
        check(sa.stats.maxPerCluster == kPinnedMaxPerCluster, "maxPerCluster matches the pin");
        check(sa.stats.totalAssignments == kPinnedTotalAssignments,
              "totalAssignments matches the pin");
    }

    if (g_fail == 0) std::printf("manylight_test: all checks passed\n");
    else std::printf("manylight_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
