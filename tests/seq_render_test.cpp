// Slice SEQ-S6 — Deterministic Cinematic Sequencer LIT 3D RENDER CAPSTONE: the pure-CPU contract test for the
// engine/seq/seq_render.h render bridge (the money-shot, the 6th and FINAL slice of FLAGSHIP #25 SEQ). The
// cheap pre-bake proof — does NOT need Metal: it pins that the FLOAT render bridge faithfully + deterministically
// carries the bit-exact Q16.16 transform timeline into render instances (the provenance the image golden then
// visually confirms). Pure C++ (header-only, no device, no backend symbols). Namespace hf::seq.
//
// SELF-CONTAINED: the test scaffolding (check() + HF_TEST_MAIN_INIT()) is copied from seq_test.cpp (NOT
// included) so this compiles STANDALONE with `clang++ -std=c++20 -I engine -I tests tests/seq_render_test.cpp`.
//
// What this PINS (the four SEQ-S6 assertions):
//   (1) provenance count   — SeqToRenderInstances(MakeCutsceneTrail(24)) has exactly 24 instances;
//   (2) deterministic float — two SeqToRenderInstances calls are BYTE-IDENTICAL (the bridge has no RNG/clock);
//   (3) the empty no-op    — empty samples -> empty instances;
//   (4) provenance fidelity — each instance's translation column == FxToFloat of the sample's bit-exact
//                             translation (the float bridge carries the integer state faithfully).
#include "seq/seq_render.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::seq;
namespace seq  = hf::seq;
namespace fpx  = hf::sim::fpx;
namespace math = hf::math;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

int main() {
    HF_TEST_MAIN_INIT();

    // (1) PROVENANCE COUNT: 24 samples -> 24 render instances (one per sampled transform).
    {
        const std::vector<FxTransform> trail = seq::MakeCutsceneTrail(24);
        const std::vector<math::Mat4>  inst  = seq::SeqToRenderInstances(trail);
        check(trail.size() == 24 && inst.size() == 24,
              "seq-render: SeqToRenderInstances(MakeCutsceneTrail(24)) has 24 instances (provenance count)");
    }

    // (2) DETERMINISTIC FLOAT BRIDGE: two SeqToRenderInstances calls over the same trail are byte-identical
    // (the bridge is a pure function — no RNG, no clock).
    {
        const std::vector<FxTransform> trail = seq::MakeCutsceneTrail(24);
        const std::vector<math::Mat4>  a = seq::SeqToRenderInstances(trail);
        const std::vector<math::Mat4>  b = seq::SeqToRenderInstances(trail);
        bool identical = (a.size() == b.size());
        for (std::size_t k = 0; k < a.size() && identical; ++k)
            if (std::memcmp(a[k].m, b[k].m, sizeof(float) * 16) != 0) identical = false;
        check(identical,
              "seq-render: two SeqToRenderInstances calls are byte-identical (deterministic float bridge)");
    }

    // (3) THE EMPTY NO-OP: empty samples -> empty instances (the cleared base scene).
    {
        const std::vector<FxTransform> empty;
        check(seq::SeqToRenderInstances(empty).empty(),
              "seq-render: empty samples -> empty instances (the no-op)");
    }

    // (4) PROVENANCE FIDELITY: each instance's translation column (math::Mat4 is column-major, translation in
    // m[12]/m[13]/m[14]) == FxToFloat of the sample's bit-exact integer translation. This proves the float
    // bridge faithfully carries the bit-exact integer state.
    {
        const std::vector<FxTransform> trail = seq::MakeCutsceneTrail(24);
        const std::vector<math::Mat4>  inst  = seq::SeqToRenderInstances(trail);
        bool faithful = (trail.size() == inst.size());
        for (std::size_t k = 0; k < trail.size() && faithful; ++k) {
            const float tx = fpx::FxToFloat(trail[k].t.x);
            const float ty = fpx::FxToFloat(trail[k].t.y);
            const float tz = fpx::FxToFloat(trail[k].t.z);
            if (inst[k].m[12] != tx || inst[k].m[13] != ty || inst[k].m[14] != tz) faithful = false;
        }
        check(faithful,
              "seq-render: each instance translation == FxToFloat of the sample's bit-exact translation (provenance)");
    }

    if (g_fail == 0) std::printf("seq_render_test: ALL PASS\n");
    else             std::printf("seq_render_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
