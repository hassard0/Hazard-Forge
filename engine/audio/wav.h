#pragma once
// Slice BB — deterministic 16-bit PCM WAV writer. Pure C++ (stdlib only); NO RHI/backend symbols.
// Part of hf_core (ASan-scoped, unit-tested). The header is the standard 44-byte canonical PCM WAV
// (RIFF/WAVE/fmt /data, format=1, 16-bit, little-endian) with NO timestamps or other non-deterministic
// fields, so the same buffer always encodes to byte-identical bytes.
#include <cstdint>
#include <string>
#include <vector>

namespace hf::audio {

// Build the full WAV byte stream (44-byte header + interleaved sample bytes) for a known buffer.
// `interleaved` is L,R,L,R,... int16 PCM. Little-endian on every platform (we serialize bytes by
// hand, never memcpy a host-endian struct). Exposed so tests can assert header fields without IO.
std::vector<uint8_t> EncodeWav(int sampleRate, int channels, const std::vector<int16_t>& interleaved);

// Encode (EncodeWav) and write the bytes to `path`. Returns true on success.
bool WriteWav(const std::string& path, int sampleRate, int channels,
              const std::vector<int16_t>& interleaved);

}  // namespace hf::audio
