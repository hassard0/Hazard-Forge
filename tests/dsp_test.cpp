// Unit test for the DSP1 wavetable oscillator NODE (engine/audio/dsp.h, flagship #23 DETERMINISTIC AUDIO,
// beachhead). Pure CPU (hf_core), ASan-eligible like the other pure tests.
//
// The oscillator is INTEGER / FIXED-POINT (Q15 int16 PCM, uint32 phase): no float anywhere, so the
// rendered buffer is bit-identical run-to-run AND platform-to-platform (MSVC vs Apple clang). The golden
// is a PINNED FNV-1a-64 DigestBuffer value (NO image, NO render-bake).
//
// What this pins:
//   * Block-boundary determinism (MAKE-OR-BREAK) — a 440Hz sine of N*256 frames rendered as ONE buffer
//     (RenderOsc) is BYTE-IDENTICAL to N separate 256-frame RenderBlock calls on ONE persistent OscNode
//     (the 32-bit phase accumulator carries across block boundaries).
//   * Pinned hash — DigestBuffer(one-buffer) == a hard-pinned uint64_t (the regression anchor / golden).
//   * Replay-stable — two RenderOsc calls byte-identical.
//   * Per-wave — Sine/Saw/Square each a deterministic pinned hash; a 440Hz Square is hard +-32767 only.
//   * Freq-sensitive — 220Hz vs 440Hz differ.

#include "audio/dsp.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::audio;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();
    const int    kSR        = 48000;
    const int    kBlock     = 256;
    const int    kN         = 16;
    const int    kTotal     = kN * kBlock;   // 4096 frames
    const uint32_t kFreq    = 440;

    // ---- (1) BLOCK-BOUNDARY DETERMINISM (make-or-break): one big buffer == N persistent blocks. -----
    const std::vector<int16_t> oneBuf = dsp::RenderOsc(dsp::Wave::Sine, kFreq, kSR, kTotal);
    std::vector<int16_t> nBlocks;
    {
        dsp::OscNode osc;
        osc.wave = dsp::Wave::Sine;
        osc.freqHz = kFreq;
        osc.phase = 0;
        for (int b = 0; b < kN; ++b) dsp::RenderBlock(osc, kSR, kBlock, nBlocks);
    }
    const uint64_t hOne    = dsp::DigestBuffer(oneBuf);
    const uint64_t hBlocks = dsp::DigestBuffer(nBlocks);
    check(oneBuf.size() == static_cast<size_t>(kTotal), "RenderOsc fills totalFrames");
    check(nBlocks.size() == static_cast<size_t>(kTotal), "N RenderBlock calls fill N*block frames");
    check(oneBuf == nBlocks, "block-boundary: one-buffer == N-blocks (byte-identical)");
    check(hOne == hBlocks, "block-boundary: DigestBuffer(one) == DigestBuffer(N-blocks)");

    // ---- (2) PINNED HASH — the regression anchor / golden. ------------------------------------------
    const uint64_t kPinnedSine440 = 0x61f818d005aa9762ull;
    check(hOne == kPinnedSine440, "pinned hash: DigestBuffer(440Hz sine, 4096f) matches golden");

    // ---- (3) REPLAY-STABLE — two RenderOsc calls byte-identical. ------------------------------------
    {
        const std::vector<int16_t> again = dsp::RenderOsc(dsp::Wave::Sine, kFreq, kSR, kTotal);
        check(dsp::DigestBuffer(again) == hOne, "replay-stable: two RenderOsc calls byte-identical");
    }

    // ---- (4) PER-WAVE — Sine/Saw/Square each a deterministic pinned hash. ---------------------------
    const std::vector<int16_t> sineBuf   = oneBuf;
    const std::vector<int16_t> sawBuf    = dsp::RenderOsc(dsp::Wave::Saw,    kFreq, kSR, kTotal);
    const std::vector<int16_t> squareBuf = dsp::RenderOsc(dsp::Wave::Square, kFreq, kSR, kTotal);
    const uint64_t hSine   = dsp::DigestBuffer(sineBuf);
    const uint64_t hSaw    = dsp::DigestBuffer(sawBuf);
    const uint64_t hSquare = dsp::DigestBuffer(squareBuf);
    const uint64_t kPinnedSaw440    = 0xa26107d87d53ef03ull;
    const uint64_t kPinnedSquare440 = 0xcedfc6d4113edf24ull;
    check(hSine   == kPinnedSine440,    "per-wave: sine hash pinned");
    check(hSaw    == kPinnedSaw440,     "per-wave: saw hash pinned");
    check(hSquare == kPinnedSquare440,  "per-wave: square hash pinned");
    check(hSine != hSaw && hSaw != hSquare && hSine != hSquare, "per-wave: three waves all differ");
    {
        bool onlyPm = true;
        for (const int16_t s : squareBuf) if (s != 32767 && s != -32767) { onlyPm = false; break; }
        check(onlyPm, "per-wave: 440Hz Square samples are only +-32767");
    }

    // ---- (5) FREQ-SENSITIVE — 220Hz vs 440Hz differ. ------------------------------------------------
    {
        const std::vector<int16_t> s220 = dsp::RenderOsc(dsp::Wave::Sine, 220, kSR, kTotal);
        check(dsp::DigestBuffer(s220) != hSine, "freq-sensitive: 220Hz != 440Hz");
    }

    // ======================================================================================
    // DSP2 — Integer ADSR envelope NODE (engine/audio/dsp.h, flagship #23, 2nd slice).
    // ======================================================================================
    // The same INTEGER buffer-hash bar as DSP1: the osc->env chain is bit-identical run-to-run AND
    // platform-to-platform, and the envelope's elapsed-sample state (`env.t`) carries across blocks.
    const int kAtk = 1000, kDec = 2000, kSus = 24000, kRel = 1500;
    dsp::Adsr kAdsr;
    kAdsr.attack = kAtk; kAdsr.decay = kDec; kAdsr.sustainLevel = kSus; kAdsr.release = kRel;

    // ---- (D2-1) BLOCK-BOUNDARY DETERMINISM (make-or-break): one ApplyEnvBlock over the whole 440Hz
    //      sine source == N separate 256-frame ApplyEnvBlock calls on ONE persistent EnvNode. ---------
    std::vector<int16_t> envOne;
    {
        dsp::EnvNode node;
        node.env = kAdsr;
        node.durSample = kTotal;
        dsp::ApplyEnvBlock(node, oneBuf, envOne);
    }
    std::vector<int16_t> envN;
    {
        dsp::EnvNode node;
        node.env = kAdsr;
        node.durSample = kTotal;
        for (int b = 0; b < kN; ++b) {
            std::vector<int16_t> chunk(oneBuf.begin() + b * kBlock, oneBuf.begin() + (b + 1) * kBlock);
            dsp::ApplyEnvBlock(node, chunk, envN);
        }
    }
    const uint64_t hEnvOne = dsp::DigestBuffer(envOne);
    const uint64_t hEnvN   = dsp::DigestBuffer(envN);
    check(envOne.size() == static_cast<size_t>(kTotal), "dsp2: ApplyEnvBlock fills totalFrames");
    check(envN.size()   == static_cast<size_t>(kTotal), "dsp2: N ApplyEnvBlock calls fill N*block");
    check(envOne == envN, "dsp2: block-boundary: one-buffer == N-blocks (byte-identical)");
    check(hEnvOne == hEnvN, "dsp2: block-boundary: DigestBuffer(one) == DigestBuffer(N-blocks)");

    // ---- (D2-2) GATE-OFF TAIL is exact silence: render a few blocks PAST durSample; EnvelopeAt
    //      returns 0 for t > durSample, so the tail samples (the source kept going) are all exactly 0. -
    {
        const int kExtra = 3 * kBlock;   // 768 frames past the end
        std::vector<int16_t> tailSrc = dsp::RenderOsc(dsp::Wave::Sine, kFreq, kSR, kTotal + kExtra);
        std::vector<int16_t> tailOut;
        dsp::EnvNode node;
        node.env = kAdsr;
        node.durSample = kTotal;
        dsp::ApplyEnvBlock(node, tailSrc, tailOut);
        bool tailZeros = true;
        for (size_t i = static_cast<size_t>(kTotal); i < tailOut.size(); ++i)
            if (tailOut[i] != 0) { tailZeros = false; break; }
        check(tailZeros, "dsp2: gate-off tail (t > durSample) is exact silence");
    }

    // ---- (D2-3) BYPASS NO-OP: enabled=false => output == source BYTE-IDENTICAL. ---------------------
    {
        std::vector<int16_t> bypass;
        dsp::EnvNode node;
        node.env = kAdsr;
        node.durSample = kTotal;
        node.enabled = false;
        dsp::ApplyEnvBlock(node, oneBuf, bypass);
        check(bypass == oneBuf, "dsp2: bypass no-op (enabled=false) == source byte-identical");
    }

    // ---- (D2-4) ATTACK ramps from 0: sample 0 of an attack>0 env is 0 (silent onset) and the level is
    //      monotone non-decreasing across the attack region; hand-check a couple points. --------------
    {
        check(dsp::EnvelopeAt(kAdsr, 0, kTotal) == 0, "dsp2: attack sample 0 is silent (level 0)");
        bool mono = true;
        int prev = -1;
        for (int t = 0; t < kAtk; ++t) {
            const int lv = dsp::EnvelopeAt(kAdsr, t, kTotal);
            if (lv < prev) { mono = false; break; }
            prev = lv;
        }
        check(mono, "dsp2: attack level monotone non-decreasing across [0, attack)");
        // Hand-checked points: linear 0->32767 over [0,1000). t=500 -> 32767*500/1000 = 16383.
        check(dsp::EnvelopeAt(kAdsr, 500, kTotal) == 16383, "dsp2: attack midpoint == 16383");
        check(dsp::EnvelopeAt(kAdsr, 250, kTotal) == 8191,  "dsp2: attack quarter  == 8191");
    }

    // ---- (D2-5) PINNED HASH — the osc->env chain digest == a hard-pinned uint64_t. ------------------
    const uint64_t kPinnedEnv440 = 0x6ce8566f08ab8e17ull;
    check(hEnvOne == kPinnedEnv440, "dsp2: pinned hash: DigestBuffer(osc->env, 4096f) matches golden");

    // ---- Showcase / --dsp1-osc numeric proof (printed in the test; no image golden). ---------------
    std::printf("dsp1-osc: wavetable oscillator (440Hz sine, sampleRate 48000, frames 4096)\n");
    std::printf("dsp1-osc: block-boundary determinism: one-buffer == N-blocks {hash:0x%016llx}\n",
                static_cast<unsigned long long>(hOne));
    std::printf("dsp1-osc: per-wave hashes {sine:0x%016llx, saw:0x%016llx, square:0x%016llx}\n",
                static_cast<unsigned long long>(hSine),
                static_cast<unsigned long long>(hSaw),
                static_cast<unsigned long long>(hSquare));
    std::printf("dsp1-osc: provenance {frames:4096, sampleRate:48000, blockSize:256}\n");

    std::printf("dsp2-env: ADSR envelope node (a/d/s/r = %d/%d/%d/%d, durSample %d)\n",
                kAtk, kDec, kSus, kRel, kTotal);
    std::printf("dsp2-env: block-boundary determinism: one-buffer == N-blocks {hash:0x%016llx}\n",
                static_cast<unsigned long long>(hEnvOne));
    std::printf("dsp2-env: gate-off tail is exact silence {tailZeros:true}\n");
    std::printf("dsp2-env: bypass no-op (enabled=false) == source BYTE-IDENTICAL\n");
    std::printf("dsp2-env: osc->env pinned {hash:0x%016llx}\n",
                static_cast<unsigned long long>(hEnvOne));

    // ======================================================================================
    // DSP3 — Fixed-point biquad FILTER NODE (engine/audio/dsp.h, flagship #23, 3rd slice, THE CRUX).
    // ======================================================================================
    // Q14 Direct-Form-I biquad with committed integer presets. Same INTEGER buffer-hash bar: bit-identical
    // run-to-run AND platform-to-platform, and the delay line (x1/x2/y1/y2) carries across blocks.

    // A saw source spanning N*256 frames is the filter test input (the carried delay line is the crux).
    const std::vector<int16_t> sawSrc = dsp::RenderOsc(dsp::Wave::Saw, kFreq, kSR, kTotal);

    // ---- (D3-1) PASSTHROUGH NO-OP (make-or-break): kBiquadPassthrough over a source == source. --------
    {
        std::vector<int16_t> pass;
        dsp::BiquadNode bq{dsp::kBiquadPassthrough};
        dsp::FilterBlock(bq, sawSrc, pass);
        check(pass == sawSrc, "dsp3: passthrough preset == source byte-identical (additive-identity no-op)");
        check(dsp::DigestBuffer(pass) == dsp::DigestBuffer(sawSrc), "dsp3: passthrough digest == source digest");
    }

    // ---- (D3-2) BLOCK-BOUNDARY DETERMINISM (make-or-break): one FilterBlock over the whole saw source
    //      == N separate 256-frame FilterBlock calls on ONE persistent BiquadNode (delay line carries). --
    std::vector<int16_t> lpOne;
    {
        dsp::BiquadNode bq{dsp::kBiquadLowpass2k};
        dsp::FilterBlock(bq, sawSrc, lpOne);
    }
    std::vector<int16_t> lpN;
    {
        dsp::BiquadNode bq{dsp::kBiquadLowpass2k};
        for (int b = 0; b < kN; ++b) {
            std::vector<int16_t> chunk(sawSrc.begin() + b * kBlock, sawSrc.begin() + (b + 1) * kBlock);
            dsp::FilterBlock(bq, chunk, lpN);
        }
    }
    const uint64_t hLpOne = dsp::DigestBuffer(lpOne);
    const uint64_t hLpN   = dsp::DigestBuffer(lpN);
    check(lpOne.size() == static_cast<size_t>(kTotal), "dsp3: FilterBlock fills totalFrames");
    check(lpN.size()   == static_cast<size_t>(kTotal), "dsp3: N FilterBlock calls fill N*block");
    check(lpOne == lpN, "dsp3: block-boundary: one-buffer == N-blocks (byte-identical)");
    check(hLpOne == hLpN, "dsp3: block-boundary: DigestBuffer(one) == DigestBuffer(N-blocks)");

    // ---- (D3-3) STABILITY / BOUNDED OUTPUT: a full-scale +32767 step through the lowpass for many blocks
    //      must stay in int16 AND settle near the input level (not oscillate to the rails). --------------
    int   stepMaxAbs   = 0;
    int   stepSettled  = 0;
    bool  stepRailed   = false;
    {
        const int kSteps   = 64;                 // 64 blocks of 256 = 16384 frames of full-scale DC
        const int kStepLen = kSteps * kBlock;
        std::vector<int16_t> stepSrc(static_cast<size_t>(kStepLen), 32767);
        std::vector<int16_t> stepOut;
        dsp::BiquadNode bq{dsp::kBiquadLowpass2k};
        for (int b = 0; b < kSteps; ++b) {
            std::vector<int16_t> chunk(stepSrc.begin() + b * kBlock, stepSrc.begin() + (b + 1) * kBlock);
            dsp::FilterBlock(bq, chunk, stepOut);
        }
        for (const int16_t s : stepOut) {
            const int a = s < 0 ? -static_cast<int>(s) : static_cast<int>(s);
            if (a > stepMaxAbs) stepMaxAbs = a;
        }
        // Steady state: average of the last block. A unity-DC-gain lowpass settles to ~the input (32767).
        long acc = 0;
        for (int i = kStepLen - kBlock; i < kStepLen; ++i) acc += stepOut[static_cast<size_t>(i)];
        stepSettled = static_cast<int>(acc / kBlock);
        // Railed-indefinitely check: the last block must NOT be all-rails (a stable filter settles).
        bool allRail = true;
        for (int i = kStepLen - kBlock; i < kStepLen; ++i)
            if (stepOut[static_cast<size_t>(i)] >= 32767 || stepOut[static_cast<size_t>(i)] <= -32768) { allRail = false; break; }
        stepRailed = !allRail ? false : true;
        check(stepMaxAbs <= 32767, "dsp3: stability: output stays within int16 (no overflow)");
        check(!stepRailed, "dsp3: stability: last block not pinned to the rails (filter settles)");
        // Unity-DC-gain lowpass settles near the input (32767). Bounded window proves no runaway/oscillation.
        check(stepSettled > 30000 && stepSettled <= 32767, "dsp3: stability: step settles near input level (DC gain ~1)");
    }

    // ---- (D3-4) LOWPASS ATTENUATES HIGHS: a 12kHz tone and a 500Hz tone at equal amplitude through the
    //      lowpass separately -> the high-freq output energy is materially LOWER (coarse integer compare).
    uint64_t lowEnergy = 0, highEnergy = 0;
    {
        const std::vector<int16_t> loSrc = dsp::RenderOsc(dsp::Wave::Sine,   500, kSR, kTotal);
        const std::vector<int16_t> hiSrc = dsp::RenderOsc(dsp::Wave::Sine, 12000, kSR, kTotal);
        std::vector<int16_t> loOut, hiOut;
        { dsp::BiquadNode bq{dsp::kBiquadLowpass2k}; dsp::FilterBlock(bq, loSrc, loOut); }
        { dsp::BiquadNode bq{dsp::kBiquadLowpass2k}; dsp::FilterBlock(bq, hiSrc, hiOut); }
        // Skip the first block to let the filter settle past the transient, then sum |sample|^2.
        for (size_t i = static_cast<size_t>(kBlock); i < loOut.size(); ++i)
            lowEnergy  += static_cast<uint64_t>(static_cast<int64_t>(loOut[i]) * loOut[i]);
        for (size_t i = static_cast<size_t>(kBlock); i < hiOut.size(); ++i)
            highEnergy += static_cast<uint64_t>(static_cast<int64_t>(hiOut[i]) * hiOut[i]);
        check(highEnergy < lowEnergy, "dsp3: lowpass attenuates highs (12kHz energy < 500Hz energy)");
    }

    // ---- (D3-5) PINNED HASH — the saw->lowpass digest == a hard-pinned uint64_t (the golden anchor). ---
    const uint64_t kPinnedSawLowpass = 0x0dd89c23d9b537d2ull;
    check(hLpOne == kPinnedSawLowpass, "dsp3: pinned hash: DigestBuffer(saw->lowpass, 4096f) matches golden");

    // ---- DSP3 showcase / numeric proof (printed; no image golden). ----------------------------------
    std::printf("dsp3-filter: biquad filter node (Q14 Direct-Form-I, presets: passthrough/lowpass2k/highpass2k)\n");
    std::printf("dsp3-filter: passthrough preset == source BYTE-IDENTICAL (additive-identity no-op)\n");
    std::printf("dsp3-filter: block-boundary determinism: one-buffer == N-blocks {hash:0x%016llx}\n",
                static_cast<unsigned long long>(hLpOne));
    std::printf("dsp3-filter: stable bounded output over %d blocks {maxAbs:%d, settled:%d, railed:%s}\n",
                64, stepMaxAbs, stepSettled, stepRailed ? "true" : "false");
    std::printf("dsp3-filter: lowpass attenuates highs {lowEnergy:%llu, highEnergy:%llu} Hh < L\n",
                static_cast<unsigned long long>(lowEnergy), static_cast<unsigned long long>(highEnergy));
    std::printf("dsp3-filter: saw->lowpass pinned {hash:0x%016llx}\n",
                static_cast<unsigned long long>(hLpOne));

    // ======================================================================================
    // DSP4 — Declarative node-graph PATCH (engine/audio/dsp.h, flagship #23, 4th slice, THE HEADLINE).
    // ======================================================================================
    // A flat node array with integer-index wiring evaluated per block in fixed topo order. Same INTEGER
    // buffer-hash bar: bit-identical run-to-run AND platform-to-platform, and every node state carries
    // across EvaluateBlock calls (so one-buffer == N-blocks).

    // ---- (D4-1) SINGLE-OSC PATCH == BARE OSC (make-or-break no-op): a one-Osc patch (outNode 0) renders
    //      byte-identical to RenderOsc with the same params (the graph adds zero overhead). --------------
    {
        dsp::Patch p;
        dsp::DspNode oscN;
        oscN.type = dsp::NodeType::Osc;
        oscN.osc.wave = dsp::Wave::Sine;
        oscN.osc.freqHz = kFreq;
        oscN.osc.phase = 0;
        p.nodes.push_back(oscN);
        p.outNode = 0;
        const std::vector<int16_t> patchOut = dsp::RenderPatch(p, kSR, kTotal);
        check(patchOut == oneBuf, "dsp4: single-osc patch == bare RenderOsc (byte-identical, graph transparent)");
        check(dsp::DigestBuffer(patchOut) == hOne, "dsp4: single-osc patch digest == bare osc digest");
    }

    // ---- (D4-2) CHAIN EQUIVALENCE: a 3-node patch Osc->Env->Filter renders byte-identical to the
    //      hand-chained DSP1->DSP2->DSP3 on fresh nodes with the same params. ---------------------------
    std::vector<int16_t> handChain;
    {
        // Hand-chained reference: RenderOsc -> ApplyEnvBlock -> FilterBlock.
        std::vector<int16_t> oscBuf = dsp::RenderOsc(dsp::Wave::Sine, kFreq, kSR, kTotal);
        std::vector<int16_t> envBuf;
        { dsp::EnvNode e; e.env = kAdsr; e.durSample = kTotal; dsp::ApplyEnvBlock(e, oscBuf, envBuf); }
        { dsp::BiquadNode bq{dsp::kBiquadLowpass2k}; dsp::FilterBlock(bq, envBuf, handChain); }
    }
    auto makeChainPatch = []() -> dsp::Patch {
        dsp::Patch p;
        dsp::DspNode o; o.type = dsp::NodeType::Osc; o.osc.wave = dsp::Wave::Sine; o.osc.freqHz = 440; o.osc.phase = 0;
        dsp::DspNode e; e.type = dsp::NodeType::Env; e.in0 = 0;
        dsp::DspNode f; f.type = dsp::NodeType::Filter; f.in0 = 1; f.filt.c = dsp::kBiquadLowpass2k;
        p.nodes.push_back(o); p.nodes.push_back(e); p.nodes.push_back(f);
        p.outNode = 2;
        return p;
    };
    {
        dsp::Patch p = makeChainPatch();
        p.nodes[1].env.env = kAdsr; p.nodes[1].env.durSample = kTotal;
        const std::vector<int16_t> patchChain = dsp::RenderPatch(p, kSR, kTotal);
        check(patchChain == handChain, "dsp4: chain Osc->Env->Filter == hand-chained DSP1->DSP2->DSP3 (byte-identical)");
    }

    // ---- (D4-3) BLOCK-BOUNDARY DETERMINISM: that 3-node chain patch as ONE N*256 buffer (RenderPatch)
    //      vs N separate 256-frame EvaluateBlock calls on ONE persistent Patch -> byte-identical. --------
    uint64_t hChainOne = 0;
    {
        dsp::Patch pOne = makeChainPatch();
        pOne.nodes[1].env.env = kAdsr; pOne.nodes[1].env.durSample = kTotal;
        const std::vector<int16_t> chainOne = dsp::RenderPatch(pOne, kSR, kTotal);

        dsp::Patch pN = makeChainPatch();
        pN.nodes[1].env.env = kAdsr; pN.nodes[1].env.durSample = kTotal;
        std::vector<int16_t> chainN;
        for (int b = 0; b < kN; ++b) dsp::EvaluateBlock(pN, kSR, kBlock, chainN);

        hChainOne = dsp::DigestBuffer(chainOne);
        check(chainOne.size() == static_cast<size_t>(kTotal), "dsp4: RenderPatch chain fills totalFrames");
        check(chainN.size()   == static_cast<size_t>(kTotal), "dsp4: N EvaluateBlock chain calls fill N*block");
        check(chainOne == chainN, "dsp4: block-boundary: one-buffer == N-blocks (byte-identical, all states carry)");
        check(hChainOne == dsp::DigestBuffer(chainN), "dsp4: block-boundary: DigestBuffer(one) == DigestBuffer(N-blocks)");
    }

    // ---- (D4-4) MIX NODE: a Mix of two oscillators (different freqs) == hand-mixed per frame; and
    //      gain0=gain1=0 -> exact silence. -------------------------------------------------------------
    uint64_t hMix = 0;
    {
        const uint32_t kFreqA = 440, kFreqB = 660;
        const int32_t  kG0 = 24000, kG1 = 16000;
        // Hand-mixed reference.
        std::vector<int16_t> a = dsp::RenderOsc(dsp::Wave::Sine, kFreqA, kSR, kTotal);
        std::vector<int16_t> b = dsp::RenderOsc(dsp::Wave::Sine, kFreqB, kSR, kTotal);
        std::vector<int16_t> handMix;
        handMix.reserve(static_cast<size_t>(kTotal));
        for (int f = 0; f < kTotal; ++f)
            handMix.push_back(dsp::ClampI16(dsp::MulQ15(a[f], kG0) + dsp::MulQ15(b[f], kG1)));

        // Patch: osc0, osc1, mix(in0=0,in1=1).
        auto makeMixPatch = [&](int32_t g0, int32_t g1) -> dsp::Patch {
            dsp::Patch p;
            dsp::DspNode o0; o0.type = dsp::NodeType::Osc; o0.osc.wave = dsp::Wave::Sine; o0.osc.freqHz = kFreqA;
            dsp::DspNode o1; o1.type = dsp::NodeType::Osc; o1.osc.wave = dsp::Wave::Sine; o1.osc.freqHz = kFreqB;
            dsp::DspNode m;  m.type = dsp::NodeType::Mix; m.in0 = 0; m.in1 = 1; m.gain0 = g0; m.gain1 = g1;
            p.nodes.push_back(o0); p.nodes.push_back(o1); p.nodes.push_back(m);
            p.outNode = 2;
            return p;
        };
        dsp::Patch pMix = makeMixPatch(kG0, kG1);
        const std::vector<int16_t> patchMix = dsp::RenderPatch(pMix, kSR, kTotal);
        check(patchMix == handMix, "dsp4: mix node == hand-mixed ClampI16(MulQ15(a,g0)+MulQ15(b,g1)) per frame");
        hMix = dsp::DigestBuffer(patchMix);

        dsp::Patch pSilent = makeMixPatch(0, 0);
        const std::vector<int16_t> silent = dsp::RenderPatch(pSilent, kSR, kTotal);
        bool allZero = true;
        for (const int16_t s : silent) if (s != 0) { allZero = false; break; }
        check(allZero, "dsp4: mix node gain0=gain1=0 -> exact silence");
    }

    // ---- (D4-5) PINNED HASH — a representative patch (two-osc mix -> lowpass filter) digest == a hard-
    //      pinned uint64_t; a different wiring -> a different hash. ------------------------------------
    auto makeMixFilterPatch = [](int32_t g0, int32_t g1) -> dsp::Patch {
        dsp::Patch p;
        dsp::DspNode o0; o0.type = dsp::NodeType::Osc; o0.osc.wave = dsp::Wave::Sine; o0.osc.freqHz = 440;
        dsp::DspNode o1; o1.type = dsp::NodeType::Osc; o1.osc.wave = dsp::Wave::Saw;  o1.osc.freqHz = 220;
        dsp::DspNode m;  m.type = dsp::NodeType::Mix; m.in0 = 0; m.in1 = 1; m.gain0 = g0; m.gain1 = g1;
        dsp::DspNode flt; flt.type = dsp::NodeType::Filter; flt.in0 = 2; flt.filt.c = dsp::kBiquadLowpass2k;
        p.nodes.push_back(o0); p.nodes.push_back(o1); p.nodes.push_back(m); p.nodes.push_back(flt);
        p.outNode = 3;
        return p;
    };
    uint64_t hRep = 0;
    {
        dsp::Patch p = makeMixFilterPatch(20000, 12000);
        const std::vector<int16_t> rep = dsp::RenderPatch(p, kSR, kTotal);
        hRep = dsp::DigestBuffer(rep);
        const uint64_t kPinnedMixFilter = 0x849b4f1b42d8e92eull;
        check(hRep == kPinnedMixFilter, "dsp4: pinned hash: DigestBuffer(two-osc mix -> lowpass, 4096f) matches golden");

        // A different wiring (different gains) -> a different hash.
        dsp::Patch p2 = makeMixFilterPatch(10000, 30000);
        const uint64_t hRep2 = dsp::DigestBuffer(dsp::RenderPatch(p2, kSR, kTotal));
        check(hRep2 != hRep, "dsp4: different patch wiring -> different hash");
    }

    // ---- DSP4 showcase / numeric proof (printed; no image golden). ----------------------------------
    std::printf("dsp4-patch: declarative node graph (Osc/Env/Filter/Mix, flat integer-index wiring)\n");
    std::printf("dsp4-patch: single-osc patch == bare RenderOsc BYTE-IDENTICAL (graph overhead transparent)\n");
    std::printf("dsp4-patch: chain Osc->Env->Filter == hand-chained DSP1->DSP2->DSP3 BYTE-IDENTICAL\n");
    std::printf("dsp4-patch: block-boundary determinism: one-buffer == N-blocks {hash:0x%016llx}\n",
                static_cast<unsigned long long>(hChainOne));
    std::printf("dsp4-patch: mix of two oscillators pinned {hash:0x%016llx}\n",
                static_cast<unsigned long long>(hMix));
    std::printf("dsp4-patch: representative patch (mix->lowpass) pinned {hash:0x%016llx}\n",
                static_cast<unsigned long long>(hRep));

    // ======================================================================================
    // DSP5 — Deterministic 3D spatialization NODE (engine/audio/dsp.h, flagship #23, 5th slice).
    // ======================================================================================
    // mono -> stereo: integer distance attenuation + constant-power pan (kPanLut) + integer ITD, all from
    // the lateral component normX (NO atan2/trig/float). Same INTEGER buffer-hash bar: bit-identical
    // run-to-run AND platform-to-platform, and the ITD ring carries across SpatializeBlock calls.
    const int32_t kRefDist = 256;

    // The baked kPanLut endpoints + centre (the constant-power properties the proofs rely on).
    check(dsp::kPanLut[0].gainL == 32767 && dsp::kPanLut[0].gainR == 0,  "dsp5: kPanLut[0]  == {32767,0} (full-left -> R silent)");
    check(dsp::kPanLut[64].gainL == 0 && dsp::kPanLut[64].gainR == 32767, "dsp5: kPanLut[64] == {0,32767} (full-right -> L silent)");
    check(dsp::kPanLut[32].gainL == dsp::kPanLut[32].gainR,               "dsp5: kPanLut[32] centre gainL == gainR EXACTLY");
    check(dsp::kPanLut[32].gainL == 23170,                               "dsp5: kPanLut[32] centre == 23170");

    // A mono tone source (a sine) is the spatializer input across the DSP5 checks.
    const std::vector<int16_t> mono = dsp::RenderOsc(dsp::Wave::Sine, kFreq, kSR, kTotal);

    // ---- (D5-1) DEAD-CENTRE L==R (make-or-break): source directly in front (dx=0, dz=refDist) -> normX=0,
    //      itd=0, pg.gainL==pg.gainR -> the deinterleaved L and R channels are BYTE-IDENTICAL. -----------
    bool centerLR = false;
    {
        dsp::SpatialNode sp;
        sp.listener = {0, 0, 0};
        sp.source   = {0, 0, kRefDist};   // straight ahead
        sp.refDist  = kRefDist;
        std::vector<int16_t> st;
        dsp::SpatializeBlock(sp, mono, st);
        check(st.size() == static_cast<size_t>(kTotal) * 2, "dsp5: stereo buffer is 2*frames interleaved");
        std::vector<int16_t> chL, chR;
        for (size_t i = 0; i + 1 < st.size(); i += 2) { chL.push_back(st[i]); chR.push_back(st[i + 1]); }
        centerLR = (chL == chR);
        check(centerLR, "dsp5: dead-centre L channel == R channel BYTE-IDENTICAL");
    }

    // ---- (D5-2) PAN EDGES: full-left source -> every R sample 0; symmetric full-right -> every L 0. -----
    bool panEdges = false;
    {
        // Full-left: dx very negative, dz 0 -> normX = -32768 -> panIdx 0 -> pg = {32767,0} -> R silent.
        dsp::SpatialNode spL;
        spL.listener = {0, 0, 0};
        spL.source   = {-1000000, 0, 0};
        spL.refDist  = kRefDist;
        std::vector<int16_t> stL;
        dsp::SpatializeBlock(spL, mono, stL);
        bool rSilent = true;
        for (size_t i = 1; i < stL.size(); i += 2) if (stL[i] != 0) { rSilent = false; break; }

        // Full-right: dx very positive -> normX = +32768 -> panIdx 64 -> pg = {0,32767} -> L silent.
        dsp::SpatialNode spR;
        spR.listener = {0, 0, 0};
        spR.source   = {1000000, 0, 0};
        spR.refDist  = kRefDist;
        std::vector<int16_t> stR;
        dsp::SpatializeBlock(spR, mono, stR);
        bool lSilent = true;
        for (size_t i = 0; i < stR.size(); i += 2) if (stR[i] != 0) { lSilent = false; break; }

        panEdges = rSilent && lSilent;
        check(rSilent, "dsp5: full-left -> every R sample is 0");
        check(lSilent, "dsp5: full-right -> every L sample is 0");
    }

    // ---- (D5-3) DISTANCE ATTENUATION MONOTONE: a source at 2x refDist has lower peak energy than at
    //      refDist (deterministic monotone falloff). ----------------------------------------------------
    uint64_t eNear = 0, eFar = 0;
    {
        auto energyAt = [&](int32_t dz) -> uint64_t {
            dsp::SpatialNode sp;
            sp.listener = {0, 0, 0};
            sp.source   = {0, 0, dz};   // straight ahead, varying distance
            sp.refDist  = kRefDist;
            std::vector<int16_t> st;
            dsp::SpatializeBlock(sp, mono, st);
            uint64_t e = 0;
            for (const int16_t s : st) e += static_cast<uint64_t>(static_cast<int64_t>(s) * s);
            return e;
        };
        eNear = energyAt(kRefDist);        // unity gain (within refDist)
        eFar  = energyAt(kRefDist * 2);    // attenuated
        check(eFar < eNear, "dsp5: distance attenuation monotone (2x refDist energy < refDist energy)");
    }

    // ---- (D5-4) BLOCK-BOUNDARY DETERMINISM (make-or-break): a fixed OFF-CENTRE source spatialized as ONE
    //      N*256 mono buffer vs N separate 256-frame SpatializeBlock calls on ONE persistent SpatialNode
    //      -> byte-identical (the ITD ring carries across block boundaries). ------------------------------
    uint64_t hBlk = 0;
    {
        auto offCentre = []() -> dsp::SpatialNode {
            dsp::SpatialNode sp;
            sp.listener = {0, 0, 0};
            sp.source   = {180, 0, 240};   // off to the right + ahead (non-zero normX -> non-zero ITD)
            sp.refDist  = 256;
            return sp;
        };
        dsp::SpatialNode spOne = offCentre();
        std::vector<int16_t> stOne;
        dsp::SpatializeBlock(spOne, mono, stOne);

        dsp::SpatialNode spN = offCentre();
        std::vector<int16_t> stN;
        for (int b = 0; b < kN; ++b) {
            std::vector<int16_t> chunk(mono.begin() + b * kBlock, mono.begin() + (b + 1) * kBlock);
            dsp::SpatializeBlock(spN, chunk, stN);
        }
        hBlk = dsp::DigestBuffer(stOne);
        check(stOne.size() == static_cast<size_t>(kTotal) * 2, "dsp5: one-buffer stereo fills 2*frames");
        check(stN.size()   == static_cast<size_t>(kTotal) * 2, "dsp5: N-block stereo fills 2*frames");
        check(stOne == stN, "dsp5: block-boundary: one-buffer == N-blocks BYTE-IDENTICAL (ITD ring carries)");
        check(hBlk == dsp::DigestBuffer(stN), "dsp5: block-boundary: DigestBuffer(one) == DigestBuffer(N-blocks)");
    }

    // ---- (D5-5) PINNED HASH — osc -> spatial(off-centre) stereo digest == a hard-pinned uint64_t; moving
    //      the source changes the hash. ------------------------------------------------------------------
    uint64_t hSpat = 0;
    {
        dsp::SpatialNode sp;
        sp.listener = {0, 0, 0};
        sp.source   = {120, 0, 300};   // a fixed off-centre placement
        sp.refDist  = 256;
        std::vector<int16_t> st;
        dsp::SpatializeBlock(sp, mono, st);
        hSpat = dsp::DigestBuffer(st);
        const uint64_t kPinnedSpatial = 0x3517ac94697d87caull;
        check(hSpat == kPinnedSpatial, "dsp5: pinned hash: DigestBuffer(osc->spatial off-centre) matches golden");

        // Moving the source changes the hash.
        dsp::SpatialNode sp2;
        sp2.listener = {0, 0, 0};
        sp2.source   = {-200, 0, 300};   // mirrored to the left
        sp2.refDist  = 256;
        std::vector<int16_t> st2;
        dsp::SpatializeBlock(sp2, mono, st2);
        check(dsp::DigestBuffer(st2) != hSpat, "dsp5: moving the source -> different hash");
    }

    // ---- DSP5 showcase / numeric proof (printed; no image golden). ----------------------------------
    std::printf("dsp5-spatial: spatialization node (distance atten + constant-power pan + integer ITD, kMaxItd %d)\n",
                dsp::kMaxItd);
    std::printf("dsp5-spatial: dead-centre L==R BYTE-IDENTICAL {center:%s}\n", centerLR ? "true" : "false");
    std::printf("dsp5-spatial: full-left -> R silent / full-right -> L silent {panEdges:%s}\n",
                panEdges ? "ok" : "FAIL");
    std::printf("dsp5-spatial: distance attenuation monotone {near:%llu, far:%llu} vF < vN\n",
                static_cast<unsigned long long>(eNear), static_cast<unsigned long long>(eFar));
    std::printf("dsp5-spatial: block-boundary determinism: one-buffer == N-blocks {hash:0x%016llx}\n",
                static_cast<unsigned long long>(hBlk));
    std::printf("dsp5-spatial: osc->spatial pinned {hash:0x%016llx}\n",
                static_cast<unsigned long long>(hSpat));

    if (g_fail == 0) { std::printf("dsp_test: ALL CHECKS PASSED\n"); return 0; }
    std::printf("dsp_test: %d failures\n", g_fail);
    return 1;
}
