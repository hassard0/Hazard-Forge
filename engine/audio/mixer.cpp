// Slice BB — deterministic software audio mixer (see mixer.h). Pure C++; NO RHI/backend symbols.
//
// Everything in Render() is integer / fixed-point so the rendered buffer is bit-identical run-to-run
// AND across compilers (MSVC vs Apple clang). The only floating-point in this file is the COMMENT
// describing how the constant sine table was generated; the table itself is committed integer data.
#include "audio/mixer.h"

#include <cstdint>

namespace hf::audio {
namespace {

// --- Sine wavetable ---------------------------------------------------------------------------
// A FIXED, committed full-wave table of kSineN int16 samples: kSineTable[i] == round(32767 *
// sin(2*pi*i / kSineN)), i in [0, kSineN). 256 entries is plenty for the procedural showcase voices
// and keeps the table compact. Lookup is NEAREST (truncating) — no interpolation — which keeps the
// oscillator purely integer (a linear interp would still be integer, but nearest is simpler and the
// determinism contract only needs *a* fixed deterministic mapping). The phase is a 32-bit accumulator
// spanning one full cycle as 2^32; the table index is the top 8 bits (phase >> 24).
constexpr int kSineN = 256;
constexpr int kSineShift = 24;   // 32-bit phase -> top 8 bits index a 256-entry table
const int16_t kSineTable[kSineN] = {
         0,    804,   1608,   2410,   3212,   4011,   4808,   5602,
      6393,   7179,   7962,   8739,   9512,  10278,  11039,  11793,
     12539,  13279,  14010,  14732,  15446,  16151,  16846,  17530,
     18204,  18868,  19519,  20159,  20787,  21403,  22005,  22594,
     23170,  23731,  24279,  24811,  25329,  25832,  26319,  26790,
     27245,  27683,  28105,  28510,  28898,  29268,  29621,  29956,
     30273,  30571,  30852,  31113,  31356,  31580,  31785,  31971,
     32137,  32285,  32412,  32521,  32609,  32678,  32728,  32757,
     32767,  32757,  32728,  32678,  32609,  32521,  32412,  32285,
     32137,  31971,  31785,  31580,  31356,  31113,  30852,  30571,
     30273,  29956,  29621,  29268,  28898,  28510,  28105,  27683,
     27245,  26790,  26319,  25832,  25329,  24811,  24279,  23731,
     23170,  22594,  22005,  21403,  20787,  20159,  19519,  18868,
     18204,  17530,  16846,  16151,  15446,  14732,  14010,  13279,
     12539,  11793,  11039,  10278,   9512,   8739,   7962,   7179,
      6393,   5602,   4808,   4011,   3212,   2410,   1608,    804,
         0,   -804,  -1608,  -2410,  -3212,  -4011,  -4808,  -5602,
     -6393,  -7179,  -7962,  -8739,  -9512, -10278, -11039, -11793,
    -12539, -13279, -14010, -14732, -15446, -16151, -16846, -17530,
    -18204, -18868, -19519, -20159, -20787, -21403, -22005, -22594,
    -23170, -23731, -24279, -24811, -25329, -25832, -26319, -26790,
    -27245, -27683, -28105, -28510, -28898, -29268, -29621, -29956,
    -30273, -30571, -30852, -31113, -31356, -31580, -31785, -31971,
    -32137, -32285, -32412, -32521, -32609, -32678, -32728, -32757,
    -32767, -32757, -32728, -32678, -32609, -32521, -32412, -32285,
    -32137, -31971, -31785, -31580, -31356, -31113, -30852, -30571,
    -30273, -29956, -29621, -29268, -28898, -28510, -28105, -27683,
    -27245, -26790, -26319, -25832, -25329, -24811, -24279, -23731,
    -23170, -22594, -22005, -21403, -20787, -20159, -19519, -18868,
    -18204, -17530, -16846, -16151, -15446, -14732, -14010, -13279,
    -12539, -11793, -11039, -10278,  -9512,  -8739,  -7962,  -7179,
     -6393,  -5602,  -4808,  -4011,  -3212,  -2410,  -1608,   -804,
};

constexpr int16_t kSquareAmp = 32767;   // full-scale square amplitude

// Q15 fixed-point multiply: (a * b) >> 15. Inputs are Q15 (or an int16 sample * Q15 gain); the int64
// intermediate avoids overflow before the shift. Result is the Q15 product (still in int16 range when
// both inputs are).
inline int32_t MulQ15(int32_t a, int32_t b) {
    return static_cast<int32_t>((static_cast<int64_t>(a) * b) >> 15);
}

inline int16_t ClampI16(int32_t v) {
    if (v >  32767) return  32767;
    if (v < -32768) return -32768;
    return static_cast<int16_t>(v);
}

// Sample an oscillator. `phase` is the voice's 32-bit phase accumulator for this sample; `sampleIdx`
// is the offset within the voice (used to advance the per-voice noise LCG deterministically). Returns
// an int16 source value in roughly [-32767, 32767].
inline int32_t Oscillate(const Voice& v, uint32_t phase, int sampleIdx) {
    switch (v.wave) {
        case Wave::Sine:
            return kSineTable[(phase >> kSineShift) & (kSineN - 1)];
        case Wave::Square:
            // Top bit of the phase splits the cycle: first half (+), second half (-).
            return (phase & 0x80000000u) ? -kSquareAmp : kSquareAmp;
        case Wave::Noise: {
            // Per-voice LCG, advanced (sampleIdx+1) times from the seed so each sample offset is a
            // fixed, deterministic value independent of how the loop is structured.
            uint32_t s = v.noiseSeed;
            for (int i = 0; i <= sampleIdx; ++i) s = s * 1664525u + 1013904223u;
            return static_cast<int16_t>(s >> 16);   // high 16 bits as a signed int16
        }
    }
    return 0;
}

}  // namespace

int EnvelopeAt(const Adsr& env, int t, int durSample) {
    if (t < 0 || t > durSample) return 0;

    const int a = env.attack;
    const int d = env.decay;
    const int s = env.sustainLevel;
    const int r = env.release;
    const int releaseStart = durSample - r;   // sample offset where the release ramp begins

    // Release region takes precedence (it runs against the END of the voice).
    if (r > 0 && t >= releaseStart) {
        // Level at the start of release is the sustain level; ramp sustain -> 0 over r samples.
        const int into = t - releaseStart;             // 0..r
        // s * (r - into) / r, integer.
        return static_cast<int>((static_cast<int64_t>(s) * (r - into)) / r);
    }
    // Attack: 0 -> 32767 over [0, a).
    if (a > 0 && t < a) {
        return static_cast<int>((static_cast<int64_t>(32767) * t) / a);
    }
    // Decay: 32767 -> s over [a, a+d).
    if (d > 0 && t < a + d) {
        const int into = t - a;                         // 0..d
        return static_cast<int>(32767 + (static_cast<int64_t>(s - 32767) * into) / d);
    }
    // Sustain: hold s.
    return s;
}

void Render(const MixConfig& cfg, const std::vector<Voice>& voices, std::vector<int16_t>& outInterleaved) {
    const int frames = cfg.totalSamples;
    const int ch = cfg.channels;
    outInterleaved.assign(static_cast<size_t>(frames) * ch, 0);
    if (frames <= 0 || ch != 2 || cfg.sampleRate <= 0) return;

    // Per-frame int32 accumulators (one pair per output frame). All voices sum into these WIDE
    // accumulators; we clamp to int16 ONCE per frame at the end. This is the spec's "accumulate in
    // int32, clamp int16" path — overflow is handled by a single hard clamp, not pairwise.
    std::vector<int32_t> accL(static_cast<size_t>(frames), 0);
    std::vector<int32_t> accR(static_cast<size_t>(frames), 0);

    for (const Voice& v : voices) {
        // Phase increment per sample: one full cycle == 2^32. inc = freqHz * 2^32 / sampleRate.
        const uint32_t inc = static_cast<uint32_t>(
            (static_cast<uint64_t>(v.freqHz) << 32) / static_cast<uint32_t>(cfg.sampleRate));

        // Linear (constant-sum) pan law, integer + endpoint-exact:
        //   panL = (32768 - pan) / 2,  panR = (32768 + pan) / 2  (both Q15).
        // pan = -32768 -> L=32768(clamped 32767),0 ; pan=0 -> 16384,16384 ; pan=+32767 -> 0,32767.
        // We clamp the Q15 pan weight to the max so a full-left input does not exceed 1.0.
        int32_t panL = (32768 - static_cast<int>(v.pan)) >> 1;
        int32_t panR = (32768 + static_cast<int>(v.pan)) >> 1;
        if (panL > 32767) panL = 32767;
        if (panR > 32767) panR = 32767;

        const int begin = v.startSample;
        const int end = v.startSample + v.durSample;
        uint32_t phase = 0;   // each voice's phase starts at 0 at its own startSample

        for (int n = begin; n < end && n < frames; ++n) {
            if (n < 0) { phase += inc; continue; }
            const int local = n - v.startSample;                 // offset within the voice
            const int32_t osc = Oscillate(v, phase, local);
            const int32_t env = EnvelopeAt(v.env, local, v.durSample);   // Q15
            int32_t s = MulQ15(osc, env);                        // source * envelope (Q15 scale)
            s = MulQ15(s, v.gain);                                // * gain (Q15)

            accL[static_cast<size_t>(n)] += MulQ15(s, panL);
            accR[static_cast<size_t>(n)] += MulQ15(s, panR);
            phase += inc;
        }
    }

    for (int n = 0; n < frames; ++n) {
        const size_t li = static_cast<size_t>(n) * 2;
        outInterleaved[li]     = ClampI16(accL[static_cast<size_t>(n)]);
        outInterleaved[li + 1] = ClampI16(accR[static_cast<size_t>(n)]);
    }
}

}  // namespace hf::audio
