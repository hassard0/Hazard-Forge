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

    if (g_fail == 0) { std::printf("dsp_test: ALL CHECKS PASSED\n"); return 0; }
    std::printf("dsp_test: %d failures\n", g_fail);
    return 1;
}
