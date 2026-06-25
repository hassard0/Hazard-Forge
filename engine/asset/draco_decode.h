#pragma once
// engine/asset/draco_decode.h -- SELF-CONTAINED, CLEAN-ROOM DRACO mesh decoder (issue #36).
//
// A spec-driven, deterministic decoder for KHR_draco_mesh_compression glTF meshes (>50% of the
// real-world glTF library). Implemented FROM the published Draco bitstream specification
// (github.com/google/draco/docs/spec -- Apache-2.0); we implement from the SPEC, we do NOT vendor
// or copy any draco source. The decode is a codec (same bytes in -> same bytes out), so every slice
// is golden-verified with a PINNED net::DigestBytes value, bit-identical cross-platform
// (MSVC / Windows-clang / Mac-clang).
//
// SELF-CONTAINED: this header includes ONLY <cstddef>/<cstdint>/<vector> + net/session.h (for
// net::DigestBytes). NO <cmath>/float/clock/RNG/<random>/<unordered_*>/<map>/<functional>/std::hash/
// <algorithm>/<string>/<cstring>, NO draco or cgltf. It compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/draco_test.cpp`. Pure-CPU INTEGER, every read is
// bounds-checked (out-of-range -> error=true, returns 0, never UB).
//
// This is ONE growing header: every later slice (DR2-DR5) APPENDS below DR1; DR1's pinned symbols
// do NOT change once committed.
//
// DR1 (the beachhead): the little-endian byte reader (mem_get_le16/24/32), the LEB128 varint decode,
// and the Draco header parser -- the substrate every later slice builds on.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "net/session.h"  // hf::net::DigestBytes -- the FNV-1a-64 golden currency

namespace hf::asset::draco {

// ================================ DR1 -- constants (the wire contract) ===========================
// From the Draco spec (variable.descriptions.md) -- FROZEN once pinned.
constexpr uint16_t kMetadataFlagMask = 32768;  // flags bit => per-mesh metadata is present

enum EncoderType   : uint8_t { kPointCloud = 0, kTriangularMesh = 1 };
enum MeshEncMethod : uint8_t { kMeshSequential = 0, kMeshEdgebreaker = 1 };

// The parsed Draco stream header (draco.decoder.md ParseHeader output).
struct DracoHeader {
    uint8_t  major = 0, minor = 0;   // bitstream version (Box = 2.2)
    uint8_t  encoderType = 0;        // EncoderType (Box = kTriangularMesh)
    uint8_t  encoderMethod = 0;      // MeshEncMethod (Box = kMeshEdgebreaker)
    uint16_t flags = 0;              // bit kMetadataFlagMask => metadata present
    bool     valid = false;          // magic == "DRACO" && no read error
};

// ================================ DR1 -- the little-endian byte reader ===========================
// A bounds-checked cursor over a const byte span. mem_get_le* match core.functions.md exactly:
// byte 0 is the LOW byte. Any out-of-bounds read sets error=true and returns 0 (deterministic, never
// UB) so downstream parsing degrades to valid=false rather than reading past the buffer.
struct ByteReader {
    const uint8_t* p = nullptr;
    std::size_t    n = 0;
    std::size_t    pos = 0;
    bool           error = false;

    // n - pos, clamped to 0 if the cursor is somehow past the end.
    std::size_t Remaining() const {
        return (pos <= n) ? (n - pos) : 0;
    }

    // One byte, advance the cursor. Out-of-range -> error, return 0.
    uint8_t U8() {
        if (pos >= n) { error = true; return 0; }
        return p[pos++];
    }

    // mem_get_le16: mem[0] | mem[1]<<8 ; pos += 2.
    uint32_t LE16() {
        if (Remaining() < 2) { error = true; pos = n; return 0; }
        const uint32_t b0 = p[pos + 0];
        const uint32_t b1 = p[pos + 1];
        pos += 2;
        return b0 | (b1 << 8);
    }

    // mem_get_le24: mem[0] | mem[1]<<8 | mem[2]<<16 ; pos += 3.
    uint32_t LE24() {
        if (Remaining() < 3) { error = true; pos = n; return 0; }
        const uint32_t b0 = p[pos + 0];
        const uint32_t b1 = p[pos + 1];
        const uint32_t b2 = p[pos + 2];
        pos += 3;
        return b0 | (b1 << 8) | (b2 << 16);
    }

    // mem_get_le32: mem[0] | mem[1]<<8 | mem[2]<<16 | mem[3]<<24 ; pos += 4.
    uint32_t LE32() {
        if (Remaining() < 4) { error = true; pos = n; return 0; }
        const uint32_t b0 = p[pos + 0];
        const uint32_t b1 = p[pos + 1];
        const uint32_t b2 = p[pos + 2];
        const uint32_t b3 = p[pos + 3];
        pos += 4;
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }

    // Advance k bytes (clamped). Skipping past the end sets error.
    void Skip(std::size_t k) {
        if (k > Remaining()) { error = true; pos = n; return; }
        pos += k;
    }

    // LEB128 (core.functions.md DecodeVarint): result |= (in & 0x7F) << shift; continue while
    // (in & 0x80); shift += 7. Truncated to 32 bits. A continuation byte past the end -> error.
    uint32_t VarU32() {
        uint32_t result = 0;
        uint32_t shift = 0;
        while (true) {
            if (pos >= n) { error = true; return 0; }
            const uint32_t in = p[pos++];
            result |= static_cast<uint32_t>(in & 0x7Fu) << shift;
            if ((in & 0x80u) == 0u) break;
            shift += 7;
        }
        return result;
    }

    // LEB128, full 64 bits.
    uint64_t VarU64() {
        uint64_t result = 0;
        uint32_t shift = 0;
        while (true) {
            if (pos >= n) { error = true; return 0; }
            const uint64_t in = p[pos++];
            result |= static_cast<uint64_t>(in & 0x7Full) << shift;
            if ((in & 0x80ull) == 0ull) break;
            shift += 7;
        }
        return result;
    }
};

// ================================ DR1 -- ParseHeader =============================================
// draco.decoder.md ParseHeader: UI8[5] magic "DRACO", UI8 major, UI8 minor, UI8 encoderType, UI8
// encoderMethod, UI16 flags (LE16). Returns valid=false (NOT an exception) on a bad magic or a
// truncated buffer -- the bounds-checked reader guarantees no UB on short input.
inline DracoHeader ParseHeader(ByteReader& r) {
    DracoHeader h;

    // Read the 5 magic bytes and compare to {'D','R','A','C','O'} by bytes (no <cstring>).
    static const uint8_t kMagic[5] = { 0x44, 0x52, 0x41, 0x43, 0x4F };  // "DRACO"
    bool magicOk = true;
    for (int i = 0; i < 5; ++i) {
        const uint8_t b = r.U8();
        if (b != kMagic[i]) magicOk = false;
    }

    h.major         = r.U8();
    h.minor         = r.U8();
    h.encoderType   = r.U8();
    h.encoderMethod = r.U8();
    h.flags         = static_cast<uint16_t>(r.LE16());

    h.valid = magicOk && !r.error;
    return h;
}

}  // namespace hf::asset::draco
