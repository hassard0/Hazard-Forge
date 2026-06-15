// Slice BB — deterministic 16-bit PCM WAV writer (see wav.h). Pure C++; NO RHI/backend symbols.
#include "audio/wav.h"

#include <fstream>

namespace hf::audio {
namespace {

// Little-endian byte appenders — we serialize EVERY multi-byte field by hand so the output is
// byte-identical regardless of host endianness (no struct memcpy, no timestamps).
void PutU16(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back(static_cast<uint8_t>(v & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}
void PutU32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(static_cast<uint8_t>(v & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}
void PutTag(std::vector<uint8_t>& b, const char* tag) {
    b.push_back(static_cast<uint8_t>(tag[0]));
    b.push_back(static_cast<uint8_t>(tag[1]));
    b.push_back(static_cast<uint8_t>(tag[2]));
    b.push_back(static_cast<uint8_t>(tag[3]));
}

}  // namespace

std::vector<uint8_t> EncodeWav(int sampleRate, int channels, const std::vector<int16_t>& interleaved) {
    const uint16_t kBits = 16;
    const uint32_t dataBytes = static_cast<uint32_t>(interleaved.size()) * (kBits / 8);
    const uint32_t byteRate  = static_cast<uint32_t>(sampleRate) * static_cast<uint32_t>(channels) * (kBits / 8);
    const uint16_t blockAlign = static_cast<uint16_t>(channels * (kBits / 8));

    std::vector<uint8_t> b;
    b.reserve(44 + dataBytes);

    // --- RIFF header ---
    PutTag(b, "RIFF");
    PutU32(b, 36u + dataBytes);   // chunk size = 4 ("WAVE") + (8+16 fmt) + (8 + dataBytes)
    PutTag(b, "WAVE");

    // --- fmt subchunk (PCM) ---
    PutTag(b, "fmt ");
    PutU32(b, 16u);                                   // subchunk size (PCM)
    PutU16(b, 1u);                                    // audio format = 1 (PCM)
    PutU16(b, static_cast<uint16_t>(channels));
    PutU32(b, static_cast<uint32_t>(sampleRate));
    PutU32(b, byteRate);
    PutU16(b, blockAlign);
    PutU16(b, kBits);

    // --- data subchunk ---
    PutTag(b, "data");
    PutU32(b, dataBytes);
    for (int16_t s : interleaved) PutU16(b, static_cast<uint16_t>(s));   // little-endian sample bytes

    return b;
}

bool WriteWav(const std::string& path, int sampleRate, int channels,
              const std::vector<int16_t>& interleaved) {
    std::vector<uint8_t> bytes = EncodeWav(sampleRate, channels, interleaved);
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(f);
}

}  // namespace hf::audio
