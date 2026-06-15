#pragma once
// Slice BB — deterministic software audio mixer. Pure C++ (stdlib only); NO RHI or graphics-backend
// symbols (zero vk*/MTL*/mtl::/Backend::Metal). Compiled into hf_core (ASan-scoped, unit-tested) and
// hf_engine, exactly like engine/physics and engine/math.
//
// The mix path is INTEGER / FIXED-POINT end to end: Q15 gains, an int32 accumulator, and a hard clamp
// to int16. There is NO float/double anywhere in Render()'s sample loop. Floating-point mixing can
// differ in the last bit across compilers (MSVC vs Apple clang), which would break a byte-exact
// cross-platform WAV golden; the integer path is bit-identical on every platform and every run. (A
// float path is a possible future enhancement; intentionally not here.)
//
// "Q15" means a signed 1.15 fixed-point fraction: an int in [-32768, 32767] interpreted as value/32768
// (so 32767 ~= 1.0). Multiplying two Q15 values and shifting right by 15 yields the Q15 product.
#include <cstdint>
#include <vector>

namespace hf::audio {

// Oscillator/source kind for a voice.
enum class Wave {
    Sine,    // integer phase accumulator -> fixed int16 wavetable (nearest lookup); see kSineTable.
    Square,  // sign of the phase half: +amplitude on the first half of the cycle, -amplitude on the second.
    Noise,   // per-voice integer LCG (seed = seed*1664525 + 1013904223); the high 16 bits are the sample.
};

// Piecewise-linear ADSR envelope, all times in SAMPLES, level in Q15 (0..32767). Evaluated from the
// sample offset WITHIN the voice (0 == voice start, durSample == voice end):
//   attack:  0          -> 32767        over [0, attack)
//   decay:   32767      -> sustainLevel over [attack, attack+decay)
//   sustain: sustainLevel (held)        over [attack+decay, durSample-release)
//   release: sustainLevel -> 0          over [durSample-release, durSample]
//   else (offset < 0 or > durSample):   0
// Degenerate stages (count 0) are skipped (no divide-by-zero); a release longer than the remaining
// time simply starts the ramp earlier. The whole evaluation is integer-only.
struct Adsr {
    int attack = 0;            // samples
    int decay = 0;             // samples
    int sustainLevel = 32767;  // Q15
    int release = 0;           // samples
};

// One procedural voice. gain/pan are Q15. pan maps -32768..+32767 to full-left..full-right with a
// LINEAR (constant-sum) pan law (documented in mixer.cpp); the endpoints are integer-exact.
struct Voice {
    Wave wave = Wave::Sine;
    int freqHz = 440;       // oscillator frequency in Hz (ignored by Noise)
    int startSample = 0;    // first output sample the voice is audible
    int durSample = 0;      // number of samples the voice lasts (envelope spans this)
    int16_t gain = 32767;   // Q15 amplitude
    int16_t pan = 0;        // Q15 pan: -full-left .. 0 center .. +full-right
    Adsr env{};             // amplitude envelope (samples / Q15)
    uint32_t noiseSeed = 0; // per-voice LCG seed (Wave::Noise only)
};

// Output configuration. channels is fixed at 2 (stereo) for this slice.
struct MixConfig {
    int sampleRate = 44100;
    int channels = 2;
    int totalSamples = 0;   // number of FRAMES (per-channel samples) to render
};

// Evaluate the ADSR envelope at sample offset `t` within a voice of length `durSample`. Returns Q15
// in [0, 32767]. Pure integer; exposed for unit testing.
int EnvelopeAt(const Adsr& env, int t, int durSample);

// Render `cfg.totalSamples` stereo frames of the mixed voices into `outInterleaved` (L,R,L,R,...),
// sized to totalSamples*channels. Integer-only sample loop: for each frame, sum each active voice's
// (oscillator * envelope * gain) into an int32 per-channel accumulator (after the integer pan split),
// then hard-clamp to int16. Deterministic: two calls with identical inputs produce bit-identical
// output, on every platform.
void Render(const MixConfig& cfg, const std::vector<Voice>& voices, std::vector<int16_t>& outInterleaved);

}  // namespace hf::audio
