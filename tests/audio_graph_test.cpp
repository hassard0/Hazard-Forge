// Unit test for Slice AU1 — the DETERMINISTIC PROCEDURAL AUDIO GRAPH + 3D SPATIALIZATION
// (engine/audio/audio_graph.h, Track-S S10). Pure CPU (hf_core), ASan-eligible like the other pure
// tests. The graph is INTEGER end to end (Q15 PCM, Q16.16 positions): no float anywhere in the sample
// path, so every buffer here is bit-identical run-to-run AND platform-to-platform (MSVC vs clang) —
// every pinned value below is a hard integer golden.
//
// What this pins:
//   (a) DETERMINISM — two renders byte-identical; the showcase chain (osc->adsr->gain->delay->pan->out)
//       digest hard-pinned; one big render == N block renders (state carries across block boundaries).
//   (b) GRAPH SEMANTICS — a diamond evaluates each node ONCE (mix == sum of both arms over the SAME osc
//       block; the osc phase advanced exactly frames*inc); a cycle -> ok=false deterministically
//       (RenderBlock emits silence + false); a disconnected input reads SILENCE (the flow.h convention).
//   (c) THE DSP PROOF — kDelay: an impulse produces echoes at exactly D, 2D, 3D... with the EXACT
//       truncated Q15-decayed integer amplitudes (pinned), decaying to EXACT silence at echo 14 (the
//       honest integer-truncation character); kMix hard-clamps hot signals (pinned, incl. the -32768
//       negative clamp asymmetry); kGain at kQ15One (32768) is bit-exact identity.
//   (d) SPATIALIZATION — emitter at the listener's RIGHT pans hard right (pinned coeffs: panL==0,
//       panR==32767, L channel EXACTLY silent); directly ahead == dead-center (panL==panR==16384);
//       far quieter than near (pinned integer gains 2048 vs 32768); the spatialized render digest pinned.
//   (e) COMPOSITION — the SAME voice rendered by the EXISTING mixer.h path (audio::Render) and by the
//       graph (osc->adsr->gain->pan->out) is BYTE-IDENTICAL (the layer composes the mixer's exact
//       integer ops, it does not fork them); the graph block feeds wav.h's EncodeWav byte-stably.
//   (f) the AU1 showcase scene (RenderAu1Showcase) digest + stats pinned (the WAV-golden anchor).

#include "audio/audio_graph.h"
#include "audio/mixer.h"
#include "audio/wav.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::audio;
namespace ag = hf::audio::graph;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();
    const int kSR = 48000;

    // =================================================================================================
    // (a) DETERMINISM: two renders byte-identical; pinned showcase digest; one-buffer == N-blocks.
    // =================================================================================================
    {
        const ag::Graph g = ag::MakeShowcaseGraph();
        const int kTotal = 16 * 256;   // 4096 frames

        const std::vector<int16_t> run1 = ag::RenderGraph(g, kSR, kTotal);
        const std::vector<int16_t> run2 = ag::RenderGraph(g, kSR, kTotal);
        check(run1.size() == static_cast<size_t>(kTotal) * 2, "showcase render fills frames*2 samples");
        check(run1 == run2, "determinism: two renders byte-identical");

        // Block-boundary continuity: 16 x 256-frame blocks on ONE persistent state == one big render.
        std::vector<int16_t> blocks;
        {
            ag::GraphState st = ag::MakeGraphState(g);
            check(st.ok, "showcase graph topo-orders (ok=true)");
            for (int b = 0; b < 16; ++b) ag::RenderBlock(g, st, kSR, 256, blocks);
        }
        check(run1 == blocks, "block-boundary: one big render == 16 block renders (byte-identical)");

        const uint64_t h = ag::DigestAudioBlock(run1);
        std::printf("PIN showcase-graph digest: 0x%016llx\n", static_cast<unsigned long long>(h));
        const uint64_t kPinnedShowcase = 0xbdb925d50344266full;
        check(h == kPinnedShowcase, "pinned: showcase (osc->adsr->gain->delay->pan->out) digest");

        // The two digest currencies agree (net::DigestBytes over LE samples == dsp::DigestBuffer).
        check(ag::DigestAudioBlock(run1) == dsp::DigestBuffer(run1),
              "DigestAudioBlock == dsp::DigestBuffer (no digest fork)");
    }

    // =================================================================================================
    // (b) GRAPH SEMANTICS: diamond once-only, cycle rejection, disconnected input reads silence.
    // =================================================================================================
    {
        // Diamond: n0 osc -> {n1 gain 0.5, n2 gain 0.25} -> n3 mix -> n4 out.
        ag::Graph g;
        g.nodes.resize(5);
        auto self = [&](ag::NodeId i) {
            g.nodes[i].in[0] = i; g.nodes[i].in[1] = i; g.nodes[i].in[2] = i; g.nodes[i].in[3] = i;
        };
        self(0); g.nodes[0].kind = ag::kOsc;  g.nodes[0].wave = dsp::Wave::Sine; g.nodes[0].freqHz = 440;
        self(1); g.nodes[1].kind = ag::kGain; g.nodes[1].in[0] = 0; g.nodes[1].gainQ15 = 16384;
        self(2); g.nodes[2].kind = ag::kGain; g.nodes[2].in[0] = 0; g.nodes[2].gainQ15 = 8192;
        self(3); g.nodes[3].kind = ag::kMix;  g.nodes[3].in[0] = 1; g.nodes[3].in[1] = 2;
        self(4); g.nodes[4].kind = ag::kOut;  g.nodes[4].in[0] = 3;

        const int kN = 512;
        ag::GraphState st = ag::MakeGraphState(g);
        check(st.ok, "diamond graph topo-orders");
        std::vector<int16_t> out;
        check(ag::RenderBlock(g, st, kSR, kN, out), "diamond RenderBlock returns true");

        // Each frame == sum of both arms over the SAME oscillator block (evaluated once).
        const std::vector<int16_t> osc = dsp::RenderOsc(dsp::Wave::Sine, 440, kSR, kN);
        bool sumOk = true;
        for (int f = 0; f < kN; ++f) {
            const int32_t expect = dsp::ClampI16(dsp::MulQ15(osc[static_cast<size_t>(f)], 16384) +
                                                 dsp::MulQ15(osc[static_cast<size_t>(f)], 8192));
            if (out[static_cast<size_t>(f) * 2] != expect ||
                out[static_cast<size_t>(f) * 2 + 1] != expect) { sumOk = false; break; }
        }
        check(sumOk, "diamond: mix == gain(0.5)+gain(0.25) of the SAME osc block, both channels");

        // The shared osc advanced its phase EXACTLY kN * inc — evaluated once, not once per arm.
        const uint32_t inc = static_cast<uint32_t>(
            (static_cast<uint64_t>(440) << 32) / static_cast<uint32_t>(kSR));
        check(st.st[0].phase == static_cast<uint32_t>(inc * static_cast<uint32_t>(kN)),
              "diamond: osc phase advanced exactly frames*inc (each node evaluated ONCE)");
        check(ag::CountEdges(g) == 5u, "diamond: CountEdges == 5 (osc->g1, osc->g2, g1->mix, g2->mix, mix->out)");

        // Cycle rejection: deterministic ok=false; RenderBlock emits silence + returns false.
        const ag::Graph cyc = ag::MakeCyclicGraph();
        ag::GraphState cs1 = ag::MakeGraphState(cyc);
        ag::GraphState cs2 = ag::MakeGraphState(cyc);
        check(!cs1.ok && !cs2.ok, "cycle: MakeGraphState ok=false, deterministically");
        std::vector<int16_t> cout1;
        check(!ag::RenderBlock(cyc, cs1, kSR, 64, cout1), "cycle: RenderBlock returns false");
        check(cout1 == std::vector<int16_t>(128, 0), "cycle: RenderBlock emits exact silence");

        // Disconnected input reads SILENCE: a gain with no input -> all-zero output.
        ag::Graph sg;
        sg.nodes.resize(2);
        sg.nodes[0].kind = ag::kGain;
        sg.nodes[0].in[0] = 0; sg.nodes[0].in[1] = 0; sg.nodes[0].in[2] = 0; sg.nodes[0].in[3] = 0;
        sg.nodes[0].gainQ15 = ag::kQ15One;
        sg.nodes[1].kind = ag::kOut;
        sg.nodes[1].in[0] = 0; sg.nodes[1].in[1] = 1; sg.nodes[1].in[2] = 1; sg.nodes[1].in[3] = 1;
        const std::vector<int16_t> sil = ag::RenderGraph(sg, kSR, 64);
        check(sil == std::vector<int16_t>(128, 0), "disconnected input reads silence (the flow.h convention)");
    }

    // =================================================================================================
    // (c) THE DSP PROOF: delay echoes (exact pinned integers), mix clamp, gain identity-at-one.
    // =================================================================================================
    {
        // (c.i) kDelay: an impulse through delay(D=100, feedback=16384 == 0.5 Q15). The impulse is a
        // one-frame 16383 spike built from square->adsr(dur=0)->gain(0.5):
        //   frame 0: square(phase 0) = +32767; env(t=0,dur=0) = 32767 -> MulQ15 = 32766;
        //            gain 16384 -> MulQ15(32766,16384) = 16383. Every later frame: env = 0.
        ag::Graph g;
        g.nodes.resize(5);
        auto self = [&](ag::NodeId i) {
            g.nodes[i].in[0] = i; g.nodes[i].in[1] = i; g.nodes[i].in[2] = i; g.nodes[i].in[3] = i;
        };
        self(0); g.nodes[0].kind = ag::kOsc;   g.nodes[0].wave = dsp::Wave::Square; g.nodes[0].freqHz = 440;
        self(1); g.nodes[1].kind = ag::kAdsr;  g.nodes[1].in[0] = 0;
                 g.nodes[1].env = dsp::Adsr{0, 0, 32767, 0}; g.nodes[1].durSample = 0;
        self(2); g.nodes[2].kind = ag::kGain;  g.nodes[2].in[0] = 1; g.nodes[2].gainQ15 = 16384;
        self(3); g.nodes[3].kind = ag::kDelay; g.nodes[3].in[0] = 2;
                 g.nodes[3].delaySamples = 100; g.nodes[3].feedbackQ15 = 16384;
        self(4); g.nodes[4].kind = ag::kOut;   g.nodes[4].in[0] = 3;

        const int kN = 1600;
        const std::vector<int16_t> out = ag::RenderGraph(g, kSR, kN);

        // The EXACT truncated Q15 halving cascade (MulQ15 floors): echo k at frame 100*k.
        const int16_t kEcho[16] = {16383, 8191, 4095, 2047, 1023, 511, 255, 127,
                                   63, 31, 15, 7, 3, 1, 0, 0};
        bool echoesOk = true;
        for (int k = 0; k < 16; ++k) {
            const int16_t l = out[static_cast<size_t>(k) * 100 * 2];
            if (l != kEcho[k]) {
                std::printf("FAIL detail: echo %d = %d, expected %d\n", k, l, kEcho[k]);
                echoesOk = false;
            }
        }
        check(echoesOk, "delay: pinned echo cascade 16383,8191,4095,...,1 then EXACT silence at echo 14");
        std::printf("PIN delay echoes (D=100, fb=16384): 16383 8191 4095 2047 1023 511 255 127 "
                    "63 31 15 7 3 1 0 (echo 14 == first exact-zero echo)\n");

        // Everything OFF the echo lattice is exactly zero (impulse in, comb out).
        bool zerosOk = true;
        for (int f = 0; f < kN; ++f) {
            if (f % 100 == 0) continue;
            if (out[static_cast<size_t>(f) * 2] != 0) { zerosOk = false; break; }
        }
        check(zerosOk, "delay: all non-echo frames exactly zero");

        // (c.ii) kMix clamps hot signals: two full-scale 440 Hz squares sum to +-65534 -> hard clamp.
        ag::Graph mg;
        mg.nodes.resize(4);
        auto mself = [&](ag::NodeId i) {
            mg.nodes[i].in[0] = i; mg.nodes[i].in[1] = i; mg.nodes[i].in[2] = i; mg.nodes[i].in[3] = i;
        };
        mself(0); mg.nodes[0].kind = ag::kOsc; mg.nodes[0].wave = dsp::Wave::Square; mg.nodes[0].freqHz = 440;
        mself(1); mg.nodes[1].kind = ag::kOsc; mg.nodes[1].wave = dsp::Wave::Square; mg.nodes[1].freqHz = 440;
        mself(2); mg.nodes[2].kind = ag::kMix; mg.nodes[2].in[0] = 0; mg.nodes[2].in[1] = 1;
        mself(3); mg.nodes[3].kind = ag::kOut; mg.nodes[3].in[0] = 2;
        const std::vector<int16_t> mout = ag::RenderGraph(mg, kSR, 128);
        check(mout[0] == 32767, "mix clamp: +32767 + +32767 -> pinned 32767 (hard clamp)");
        // First negative-half frame (phase crosses 2^31 at frame 55 for 440 Hz @ 48 kHz): the sum
        // -65534 clamps to -32768 (the pinned asymmetric int16 clamp floor).
        check(mout[55 * 2] == -32768, "mix clamp: -32767 + -32767 -> pinned -32768 at frame 55");
        std::printf("PIN mix clamp: frame 0 = 32767 (from +65534), frame 55 = -32768 (from -65534)\n");

        // (c.iii) kGain at kQ15One (32768) is BIT-EXACT identity: osc->gain(1.0)->out == osc->out.
        ag::Graph ga, gb;
        ga.nodes.resize(3);
        auto aself = [&](ag::NodeId i) {
            ga.nodes[i].in[0] = i; ga.nodes[i].in[1] = i; ga.nodes[i].in[2] = i; ga.nodes[i].in[3] = i;
        };
        aself(0); ga.nodes[0].kind = ag::kOsc;  ga.nodes[0].wave = dsp::Wave::Sine; ga.nodes[0].freqHz = 440;
        aself(1); ga.nodes[1].kind = ag::kGain; ga.nodes[1].in[0] = 0; ga.nodes[1].gainQ15 = ag::kQ15One;
        aself(2); ga.nodes[2].kind = ag::kOut;  ga.nodes[2].in[0] = 1;
        gb.nodes.resize(2);
        auto bself = [&](ag::NodeId i) {
            gb.nodes[i].in[0] = i; gb.nodes[i].in[1] = i; gb.nodes[i].in[2] = i; gb.nodes[i].in[3] = i;
        };
        bself(0); gb.nodes[0].kind = ag::kOsc; gb.nodes[0].wave = dsp::Wave::Sine; gb.nodes[0].freqHz = 440;
        bself(1); gb.nodes[1].kind = ag::kOut; gb.nodes[1].in[0] = 0;
        check(ag::RenderGraph(ga, kSR, 1024) == ag::RenderGraph(gb, kSR, 1024),
              "gain at kQ15One (32768): bit-exact identity (identity-at-one)");
    }

    // =================================================================================================
    // (d) SPATIALIZATION: pinned integer coefficients + the spatialized render digest.
    // =================================================================================================
    {
        // Emitter at the listener's RIGHT (2, 0, 0), refDist 2.0: unity distance gain, hard-right pan.
        ag::SpatializeParams right;
        right.emitterPos[0] = 131072;   // +2.0
        right.refDist = 131072;         // 2.0
        const ag::SpatialCoeffs cr = ag::ComputeSpatialCoeffs(right);
        check(cr.distQ16 == 131072, "spatial right: dist == 131072 (exactly 2.0)");
        check(cr.gainDistQ15 == 32768, "spatial right: distance gain == kQ15One (exact unity at refDist)");
        check(cr.panQ15 == 32767, "spatial right: pan == +32767 (hard right)");
        check(cr.panL == 0 && cr.panR == 32767, "spatial right: panL == 0, panR == 32767 (pinned)");
        std::printf("PIN spatial right: dist=131072 gain=32768 pan=32767 panL=0 panR=32767\n");

        // Directly AHEAD (0, 0, 2): dead-center — panL == panR EXACTLY.
        ag::SpatializeParams ahead;
        ahead.emitterPos[2] = 131072;
        ahead.refDist = 131072;
        const ag::SpatialCoeffs ca = ag::ComputeSpatialCoeffs(ahead);
        check(ca.panQ15 == 0, "spatial ahead: pan == 0");
        check(ca.panL == 16384 && ca.panR == 16384, "spatial ahead: panL == panR == 16384 (pinned equal)");
        check(ca.gainDistQ15 == 32768, "spatial ahead: unity gain at refDist");
        std::printf("PIN spatial ahead: pan=0 panL=16384 panR=16384 gain=32768\n");

        // FAR (8, 0, 0) vs near, refDist 2.0: pinned inverse-square gain 32768/16 == 2048.
        ag::SpatializeParams far1;
        far1.emitterPos[0] = 524288;    // +8.0
        far1.refDist = 131072;          // 2.0
        const ag::SpatialCoeffs cf = ag::ComputeSpatialCoeffs(far1);
        check(cf.gainDistQ15 == 2048, "spatial far: gain == 2048 (refDist^2/dist^2 = 1/16, exact)");
        check(cf.gainDistQ15 < cr.gainDistQ15, "spatial: far quieter than near");
        std::printf("PIN spatial far(8.0, ref 2.0): gain=2048 (near gain=32768)\n");

        // A spatialized render: sine -> kSpatial(right) -> out. LEFT channel EXACTLY silent,
        // RIGHT channel == MulQ15(sine, 32767) frame-for-frame; whole-render digest pinned.
        ag::Graph g;
        g.nodes.resize(3);
        auto self = [&](ag::NodeId i) {
            g.nodes[i].in[0] = i; g.nodes[i].in[1] = i; g.nodes[i].in[2] = i; g.nodes[i].in[3] = i;
        };
        self(0); g.nodes[0].kind = ag::kOsc; g.nodes[0].wave = dsp::Wave::Sine; g.nodes[0].freqHz = 330;
        self(1); g.nodes[1].kind = ag::kSpatial; g.nodes[1].in[0] = 0; g.nodes[1].spatial = right;
        self(2); g.nodes[2].kind = ag::kOut; g.nodes[2].in[0] = 1;
        const int kN = 1024;
        const std::vector<int16_t> out = ag::RenderGraph(g, kSR, kN);
        const std::vector<int16_t> osc = dsp::RenderOsc(dsp::Wave::Sine, 330, kSR, kN);
        bool chOk = true;
        int64_t sumL = 0, sumR = 0;
        for (int f = 0; f < kN; ++f) {
            const int16_t l = out[static_cast<size_t>(f) * 2];
            const int16_t r = out[static_cast<size_t>(f) * 2 + 1];
            const int32_t base = dsp::MulQ15(osc[static_cast<size_t>(f)], 32768);   // == osc (identity)
            if (l != 0 || r != dsp::ClampI16(dsp::MulQ15(base, 32767))) { chOk = false; break; }
            sumL += (l < 0 ? -l : l);
            sumR += (r < 0 ? -r : r);
        }
        check(chOk, "spatial render: L exactly 0, R == MulQ15(sine, 32767) frame-for-frame");
        check(sumR > sumL, "spatial render: right-of-listener pans right (|R| > |L|)");
        const uint64_t hs = ag::DigestAudioBlock(out);
        std::printf("PIN spatialized render digest: 0x%016llx\n", static_cast<unsigned long long>(hs));
        const uint64_t kPinnedSpatial = 0x976fc317e8ce5b7dull;
        check(hs == kPinnedSpatial, "pinned: spatialized (sine->kSpatial(right)->out) render digest");
    }

    // =================================================================================================
    // (e) COMPOSITION with the EXISTING mixer: the same voice through audio::Render (mixer.h) and
    //     through the graph (osc->adsr->gain->pan->out) is BYTE-IDENTICAL — the layer composes the
    //     mixer's exact integer ops (same table, same MulQ15 order, same pan law), it does not fork.
    // =================================================================================================
    {
        const int kMixSR = 44100;
        const int kN = 4096;

        Voice v;
        v.wave = Wave::Sine;
        v.freqHz = 440;
        v.startSample = 0;
        v.durSample = kN;
        v.gain = 24000;
        v.pan = 6000;
        v.env = Adsr{400, 800, 20000, 1200};
        MixConfig cfg{kMixSR, 2, kN};
        std::vector<int16_t> mixerOut;
        Render(cfg, {v}, mixerOut);

        ag::Graph g;
        g.nodes.resize(5);
        auto self = [&](ag::NodeId i) {
            g.nodes[i].in[0] = i; g.nodes[i].in[1] = i; g.nodes[i].in[2] = i; g.nodes[i].in[3] = i;
        };
        self(0); g.nodes[0].kind = ag::kOsc;  g.nodes[0].wave = dsp::Wave::Sine; g.nodes[0].freqHz = 440;
        self(1); g.nodes[1].kind = ag::kAdsr; g.nodes[1].in[0] = 0;
                 g.nodes[1].env = dsp::Adsr{400, 800, 20000, 1200}; g.nodes[1].durSample = kN;
        self(2); g.nodes[2].kind = ag::kGain; g.nodes[2].in[0] = 1; g.nodes[2].gainQ15 = 24000;
        self(3); g.nodes[3].kind = ag::kPan;  g.nodes[3].in[0] = 2; g.nodes[3].panQ15 = 6000;
        self(4); g.nodes[4].kind = ag::kOut;  g.nodes[4].in[0] = 3;
        const std::vector<int16_t> graphOut = ag::RenderGraph(g, kMixSR, kN);

        check(graphOut.size() == mixerOut.size(), "composition: same buffer size");
        check(graphOut == mixerOut,
              "composition: graph (osc->adsr->gain->pan->out) == mixer.h audio::Render, BYTE-IDENTICAL");

        // And the graph block flows into the EXISTING wav.h encoder byte-stably (the WAV path).
        const std::vector<uint8_t> wavA = EncodeWav(kMixSR, 2, graphOut);
        const std::vector<uint8_t> wavB = EncodeWav(kMixSR, 2, mixerOut);
        check(wavA == wavB, "composition: EncodeWav(graph block) == EncodeWav(mixer block)");
    }

    // =================================================================================================
    // (f) The AU1 showcase SCENE (the --au1-graph-shot / --au1-graph WAV source): stats + digest pinned,
    //     two runs byte-identical.
    // =================================================================================================
    {
        std::vector<int16_t> buf1, buf2;
        const ag::Au1ShowcaseStats s1 = ag::RenderAu1Showcase(buf1);
        const ag::Au1ShowcaseStats s2 = ag::RenderAu1Showcase(buf2);
        check(buf1 == buf2, "showcase scene: two renders byte-identical");
        check(s1.nodes == 11u, "showcase scene: 11 nodes");
        check(s1.edges == 10u, "showcase scene: 10 edges");
        check(s1.frames == 96000, "showcase scene: 96000 frames (2 s @ 48 kHz)");
        check(buf1.size() == 192000u, "showcase scene: 192000 interleaved samples");
        std::printf("PIN au1 showcase scene digest: 0x%016llx (nodes:%u edges:%u frames:%d)\n",
                    static_cast<unsigned long long>(s1.digest), s1.nodes, s1.edges, s1.frames);
        const uint64_t kPinnedScene = 0x9ab527c7a057c06bull;
        check(s1.digest == kPinnedScene, "pinned: AU1 showcase scene digest");
        check(s1.digest == s2.digest, "showcase scene: stat digests agree");
    }

    if (g_fail == 0) {
        std::printf("audio_graph_test: ALL PASS\n");
        return 0;
    }
    std::printf("audio_graph_test: %d FAILURES\n", g_fail);
    return 1;
}
