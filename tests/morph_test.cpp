// Slice MP1 — DETERMINISTIC MORPH TARGETS / BLEND SHAPES (engine/anim/morph.h, hf::anim::morph). The
// next-tier parity gap #5: morph targets / blend shapes were explicitly SKIPPED (gltf_loader.cpp "YAGNI:
// morph" drops the glTF `weights` channel; CAPABILITIES.md confirms none built). UE5 ships morph targets
// (facial animation). MP1 adds them deterministically: per-vertex morph deltas, a weighted blend, an animated
// weight track, and composition with skinning (morph BEFORE skin). What this test PINS:
//   (a) BLEND: weights 0 == base (field-exact); single weight-1 == base+delta (exact Q16.16); a multi-target
//       blend (smile+blink) + a fractional blend (smile 0.75 + blink 1.0) pinned exact; the ascending
//       accumulation order pinned via the blend digest.
//   (b) IMPORT: BuildMorphSet/BuildWeightTrack reshape the glTF-style authored accessors — vert/target/weight
//       counts, target names, a specific delta, the default weights, and the weight-track keyframes — vs the
//       authored asset; import digest pinned (MSVC==clang).
//   (c) WEIGHT ANIM: sampling the weight track AT a keyframe == that key; BETWEEN keys == the pinned linear
//       value; the "smile" animates 0->1 with a pinned midpoint (0.5 at t=0.25).
//   (d) COMPOSE: the morph->skin order — a rigged vertex with a morph delta AND a bone transform -> the pinned
//       model-space position; reversing the order (skin then add delta) gives a DIFFERENT (wrong) result
//       (proves the order matters when the bone rotates).
//   (e) DETERMINISM: the import + blend + compose + combined shot digests PINNED identical MSVC + local clang
//       (every authored delta/weight is a multiple of 1/8 -> exactly representable -> cross-compiler exact);
//       two shot runs identical; RenderMp1Shot two runs byte-identical.
// Pure C++ (hf_core), ASan-eligible. gltf_loader.cpp is BYTE-UNTOUCHED (its "YAGNI: morph" stays) — MP1 is a
// NEW header composing anim/animation.h + motion_match.h + retarget.h read-only. Standalone: clang++ -std=c++20
// -I engine -I tests tests/morph_test.cpp engine/anim/animation.cpp. The asset is a hand-authored glTF-style
// accessor LITERAL (Mp1MorphAsset, the SK1 authored-literal discipline — no .glb on disk).
#include "anim/morph.h"
#include "anim/morph_shot.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "test_main.h"

using namespace hf;
namespace mo = hf::anim::morph;
namespace mp1 = hf::anim::morph::mp1;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static void checkEq(long long got, long long want, const char* what) {
    if (got != want) { std::printf("FAIL: %s (got %lld want %lld)\n", what, got, want); ++g_fail; }
}
static void checkHex(uint64_t got, uint64_t want, const char* what) {
    if (got != want)
        std::printf("FAIL: %s (got %016llx want %016llx)\n", what, (unsigned long long)got,
                    (unsigned long long)want), ++g_fail;
}

int main() {
    HF_TEST_MAIN_INIT();

    const mo::GltfMorphAsset asset = mp1::Mp1MorphAsset();
    const mo::MorphSet   ms = mo::BuildMorphSet(asset);
    const mo::WeightTrack wt = mo::BuildWeightTrack(asset);
    const int G = mp1::kGrid;

    // ---- (b) IMPORT: the reshape vs the authored asset --------------------------------------------------
    checkEq((long long)ms.basePositions.size(), (long long)G * G, "49 base vertices imported");
    checkEq((long long)ms.targets.size(), 2, "2 morph targets");
    check(ms.targets[0].name == "smile", "target0 name smile");
    check(ms.targets[1].name == "blink", "target1 name blink");
    checkEq((long long)ms.weights.size(), 2, "2 default weights");
    checkEq(mo::QuantizeFx(ms.weights[0]), 0, "default weight0 == 0");
    checkEq(mo::QuantizeFx(ms.weights[1]), 0, "default weight1 == 0");
    // smile corner delta: row1,col0 (v=7) -> |0-3|*0.125 = 0.375 in Y (24576 in Q16.16).
    checkEq(mo::QuantizeFx(ms.targets[0].positionDeltas[7].y), 24576, "smile corner delta.y == 0.375");
    checkEq(mo::QuantizeFx(ms.targets[0].positionDeltas[7].x), 0,     "smile corner delta.x == 0");
    // blink eye delta: row4,col1 (v=29) -> -0.25 in Y (-16384).
    checkEq(mo::QuantizeFx(ms.targets[1].positionDeltas[29].y), -16384, "blink eye delta.y == -0.25");
    // WeightTrack: 3 keys, 2 targets, key1 (t=0.5) == [1,0], key2 (t=1.0) == [1,1].
    checkEq(wt.targetCount, 2, "track targetCount == 2");
    checkEq((long long)wt.times.size(), 3, "track 3 keyframes");
    checkEq(mo::QuantizeFx(wt.times[1]), 32768, "track key1 time == 0.5");
    checkEq(mo::QuantizeFx(wt.values[1 * 2 + 0]), 65536, "track key1 smile == 1.0");
    checkEq(mo::QuantizeFx(wt.values[1 * 2 + 1]), 0,     "track key1 blink == 0.0");
    checkEq(mo::QuantizeFx(wt.values[2 * 2 + 1]), 65536, "track key2 blink == 1.0");

    // ---- (a) BLEND ----------------------------------------------------------------------------------------
    const std::vector<mo::FxV3> baseQ = mo::QuantizeVerts(ms.basePositions);
    // weights 0 == base (field-exact).
    {
        const std::vector<mo::fx> w0 = {0, 0};
        const std::vector<mo::FxV3> d = mo::ApplyMorphFx(baseQ, ms.targets, w0);
        bool same = d.size() == baseQ.size();
        for (size_t v = 0; v < d.size() && same; ++v)
            same = (d[v].x == baseQ[v].x && d[v].y == baseQ[v].y && d[v].z == baseQ[v].z);
        check(same, "weights 0 -> base bit-exact (identity)");
    }
    // single weight-1 (smile only): v=7 base.y = (1-3)*0.5 = -1.0 (-65536) + delta 24576 = -40960.
    {
        const std::vector<mo::fx> w = {65536, 0};
        const std::vector<mo::FxV3> d = mo::ApplyMorphFx(baseQ, ms.targets, w);
        checkEq(d[7].y, -40960, "single weight-1 smile v7.y == base+delta");
        checkEq(d[7].x, baseQ[7].x, "single weight-1 smile v7.x unchanged");
    }
    // multi-target blend (smile 1 + blink 1): v=29 base.y=(4-3)*0.5=0.5(32768) + blink -16384 = 16384.
    {
        const std::vector<mo::fx> w = {65536, 65536};
        const std::vector<mo::FxV3> d = mo::ApplyMorphFx(baseQ, ms.targets, w);
        checkEq(d[29].y, 16384, "smile+blink v29.y == base+blinkDelta");
    }
    // fractional blend (smile 0.75 + blink 1.0): v=7 base.y=-65536 + 0.75*24576(18432) = -47104.
    {
        const std::vector<mo::fx> w = {49152, 65536};   // 0.75, 1.0
        const std::vector<mo::FxV3> d = mo::ApplyMorphFx(baseQ, ms.targets, w);
        checkEq(d[7].y, -47104, "smile 0.75 v7.y == base + 0.75*delta");
    }
    // float and integer blend paths AGREE (exact binary-fraction authored data).
    {
        const std::vector<float> wf = {1.0f, 1.0f};
        const std::vector<math::Vec3> df = mo::ApplyMorph(ms.basePositions, ms.targets, wf);
        const std::vector<mo::fx> wq = {65536, 65536};
        const std::vector<mo::FxV3> dq = mo::ApplyMorphFx(baseQ, ms.targets, wq);
        bool agree = df.size() == dq.size();
        for (size_t v = 0; v < df.size() && agree; ++v)
            agree = (mo::QuantizeFx(df[v].x) == dq[v].x && mo::QuantizeFx(df[v].y) == dq[v].y);
        check(agree, "float ApplyMorph == integer ApplyMorphFx (exact authored data)");
    }

    // ---- (c) WEIGHT ANIM ----------------------------------------------------------------------------------
    // AT a keyframe (t=0.5) -> [1.0, 0.0].
    {
        const std::vector<mo::fx> w = mo::SampleMorphWeightsFx(wt, 0.5f);
        checkEq((long long)w.size(), 2, "sampled 2 weights");
        checkEq(w[0], 65536, "t=0.5 smile == 1.0 (at keyframe)");
        checkEq(w[1], 0,     "t=0.5 blink == 0.0");
    }
    // BETWEEN keys (t=0.25, key0 [0,0] -> key1 [1,0], frac 0.5) -> smile 0.5.
    {
        const std::vector<mo::fx> w = mo::SampleMorphWeightsFx(wt, 0.25f);
        checkEq(w[0], 32768, "t=0.25 smile == 0.5 (linear midpoint)");
        checkEq(w[1], 0,     "t=0.25 blink == 0.0");
    }
    // BETWEEN keys (t=0.75, key1 [1,0] -> key2 [1,1], frac 0.5) -> smile 1.0, blink 0.5.
    {
        const std::vector<mo::fx> w = mo::SampleMorphWeightsFx(wt, 0.75f);
        checkEq(w[0], 65536, "t=0.75 smile == 1.0");
        checkEq(w[1], 32768, "t=0.75 blink == 0.5 (linear midpoint)");
    }
    // clamp before/after the clip.
    {
        const std::vector<mo::fx> lo = mo::SampleMorphWeightsFx(wt, -1.0f);
        const std::vector<mo::fx> hi = mo::SampleMorphWeightsFx(wt, 99.0f);
        checkEq(lo[0], 0,     "t<0 clamps to first key (smile 0)");
        checkEq(hi[0], 65536, "t>dur clamps to last key (smile 1)");
        checkEq(hi[1], 65536, "t>dur clamps to last key (blink 1)");
    }

    // ---- (d) COMPOSE: morph->skin order ------------------------------------------------------------------
    // Witness: base (1,0,0), delta (0,0.5,0) at weight 1; joint = 180deg about Z + translate (+2,0,0).
    //   morph-THEN-skin (correct): skin(base+delta) = skin(1,0.5,0) = (-1,-0.5,0)+(2,0,0) = (1.0,-0.5,0).
    //   skin-THEN-morph (wrong):   skin(base)+delta = (1,0,0)+(0,0.5,0)          = (1.0, 0.5,0).
    {
        const mp1::Mp1ShotRun run = mp1::RunMp1ShotScenario();
        checkEq(run.morphThenSkin.x, 65536,  "morph->skin x == 1.0");
        checkEq(run.morphThenSkin.y, -32768, "morph->skin y == -0.5 (delta rotated with vertex)");
        checkEq(run.skinThenMorph.x, 65536,  "skin->morph x == 1.0");
        checkEq(run.skinThenMorph.y, 32768,  "skin->morph y == +0.5 (delta NOT rotated — wrong)");
        check(run.morphThenSkin.y != run.skinThenMorph.y, "compose order MATTERS (results differ)");
    }

    // ---- (e) DETERMINISM: the shot scenario digests (PINNED identical MSVC + clang) ----------------------
    {
        const mp1::Mp1ShotRun run  = mp1::RunMp1ShotScenario();
        const mp1::Mp1ShotRun run2 = mp1::RunMp1ShotScenario();
        check(run.digest == run2.digest, "shot two-run identical");
        checkEq(run.verts, (long long)G * G, "shot verts == 49");
        checkEq(run.targets, 2, "shot targets == 2");
        checkEq(run.weightKeys, 3, "shot weightKeys == 3");
        checkEq((long long)run.frames.size(), mp1::kShotFrames, "shot frames == 5");
        checkHex(run.importDigest,  0x7051bf4d0e7081f0ull, "import digest pinned (MSVC==clang)");
        checkHex(run.blendDigest,   0x594ea182fb2e8a96ull, "blend digest pinned (MSVC==clang)");
        checkHex(run.composeDigest, 0x0f4745fddc1c8784ull, "compose digest pinned (MSVC==clang)");
        checkHex(run.digest,        0x83670e4484312df3ull, "combined shot digest pinned");
        // shared raster byte-identical across two runs.
        std::vector<uint8_t> a, b; uint32_t aw = 0, ah = 0, bw = 0, bh = 0;
        mp1::RenderMp1Shot(run, a, aw, ah);
        mp1::RenderMp1Shot(run2, b, bw, bh);
        check(aw == bw && ah == bh && a == b, "RenderMp1Shot two runs byte-identical");
        checkEq((long long)aw, 700, "shot width 700");
        checkEq((long long)ah, 420, "shot height 420");
        std::printf("MP1 DIGESTS import=%016llx blend=%016llx compose=%016llx combined=%016llx\n",
                    (unsigned long long)run.importDigest, (unsigned long long)run.blendDigest,
                    (unsigned long long)run.composeDigest, (unsigned long long)run.digest);
    }

    if (g_fail == 0) std::printf("morph_test: ALL PASS\n");
    else             std::printf("morph_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
