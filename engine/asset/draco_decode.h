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

// ================================ DR2 -- rANS entropy decoder ====================================
// The shared symbol/binary range-ANS coder. BOTH the edgebreaker connectivity (DR3) and the attribute
// residuals (DR4) decode through this. Implemented spec-driven from rans.decoding.md (clean-room, NOT
// copied from draco source). DR2 ships the RAW symbol path + the binary rABS path -- both fully
// self-contained (no bit reader). The TAGGED symbol path + bit reader are deferred to DR3.
//
// APPEND-ONLY: nothing above this line is touched; DR1's pinned varint digest 0x2d4aaca6fd14312a
// stays frozen.

// ---- DR2 constants (variable.descriptions.md -- FROZEN) -----------------------------------------
constexpr uint32_t kIoBase          = 256;
constexpr uint32_t kLRansBase       = 4096;   // L_RANS_BASE (raw default base before precision scaling)
constexpr uint32_t kTaggedRansBase  = 16384;  // TAGGED_RANS_BASE  (DR3)
constexpr uint32_t kTaggedRansPrec  = 4096;   // TAGGED_RANS_PRECISION (DR3)
constexpr uint8_t  kTaggedSymbols   = 0;      // TAGGED_SYMBOLS scheme
constexpr uint8_t  kRawSymbols      = 1;      // RAW_SYMBOLS scheme
constexpr uint32_t kRabsP8Precision = 256;    // rabs_ans_p8_precision
constexpr uint32_t kRabsLBase       = 4096;   // rabs_l_base

// ---- The decoder state + a single decoded-symbol record (spec ans_decoder_ / rans_sym) ----------
struct AnsDecoder { const uint8_t* buf = nullptr; int buf_offset = 0; uint32_t state = 0; };
struct RansSym    { uint32_t val = 0, prob = 0, cum_prob = 0; };

// ---- Inline little-endian reads matching DR1 mem_get_le16/24/32 (low byte first). ---------------
// (Used by RansInitDecoder, which reads from an arbitrary buf+offset -- not a ByteReader cursor.)
inline uint32_t MemGetLe16(const uint8_t* m) {
    return static_cast<uint32_t>(m[0]) | (static_cast<uint32_t>(m[1]) << 8);
}
inline uint32_t MemGetLe24(const uint8_t* m) {
    return static_cast<uint32_t>(m[0]) | (static_cast<uint32_t>(m[1]) << 8)
         | (static_cast<uint32_t>(m[2]) << 16);
}
inline uint32_t MemGetLe32(const uint8_t* m) {
    return static_cast<uint32_t>(m[0]) | (static_cast<uint32_t>(m[1]) << 8)
         | (static_cast<uint32_t>(m[2]) << 16) | (static_cast<uint32_t>(m[3]) << 24);
}

// ---- RansInitDecoder (spec verbatim) ------------------------------------------------------------
// x = buf[offset-1] >> 6 selects 1..4 state bytes (LE, masked 0x3F / 0x3FFF / 0x3FFFFF / 0x3FFFFFFF);
// ans.state += l_rans_base. `offset` is the rANS buffer length; the header bytes live at its END and
// are read BACKWARD. Caller guarantees offset>=1 and buf has >= offset bytes (DecodeRawSymbols checks).
inline void RansInitDecoder(AnsDecoder& ans, const uint8_t* buf, int offset, uint32_t l_rans_base) {
    ans.buf = buf;
    const uint32_t x = static_cast<uint32_t>(buf[offset - 1]) >> 6;
    if (x == 0) {
        ans.buf_offset = offset - 1;
        ans.state = static_cast<uint32_t>(buf[offset - 1]) & 0x3Fu;
    } else if (x == 1) {
        ans.buf_offset = offset - 2;
        ans.state = MemGetLe16(buf + offset - 2) & 0x3FFFu;
    } else if (x == 2) {
        ans.buf_offset = offset - 3;
        ans.state = MemGetLe24(buf + offset - 3) & 0x3FFFFFu;
    } else {  // x == 3
        ans.buf_offset = offset - 4;
        ans.state = MemGetLe32(buf + offset - 4) & 0x3FFFFFFFu;
    }
    ans.state += l_rans_base;
}

// ---- RansRead (spec verbatim: renorm-in, fetch_sym inline, advance state) ------------------------
// Returns the decoded symbol value. lut_table maps rem->symbol; prob_table holds {prob, cum_prob}.
inline uint32_t RansRead(AnsDecoder& ans, uint32_t l_rans_base, uint32_t rans_precision,
                         const std::vector<uint32_t>& lut_table,
                         const std::vector<RansSym>& prob_table) {
    while (ans.state < l_rans_base && ans.buf_offset > 0) {
        ans.state = ans.state * kIoBase + ans.buf[--ans.buf_offset];
    }
    const uint32_t quo = ans.state / rans_precision;
    const uint32_t rem = ans.state % rans_precision;
    // fetch_sym: symbol = lut_table[rem]; sym = {symbol, prob[symbol], cum_prob[symbol]}.
    const uint32_t symbol = (rem < lut_table.size()) ? lut_table[rem] : 0u;
    RansSym sym;
    if (symbol < prob_table.size()) {
        sym.val = symbol;
        sym.prob = prob_table[symbol].prob;
        sym.cum_prob = prob_table[symbol].cum_prob;
    }
    ans.state = quo * sym.prob + rem - sym.cum_prob;
    return sym.val;
}

// ---- rans_build_look_up_table (spec verbatim) ---------------------------------------------------
// cum_prob accumulates; lut_table[act_prob..cum_prob) = i; prob_table[i] = {prob, cum_prob}.
// lut_table must be sized to rans_precision; prob_table sized to num_symbols.
inline void RansBuildLookUpTable(const std::vector<uint32_t>& token_probs, uint32_t num_symbols,
                                 std::vector<uint32_t>& lut_table, std::vector<RansSym>& prob_table) {
    prob_table.assign(num_symbols, RansSym{});
    uint32_t cum_prob = 0;
    uint32_t act_prob = 0;
    for (uint32_t i = 0; i < num_symbols; ++i) {
        prob_table[i].val = i;
        prob_table[i].prob = token_probs[i];
        prob_table[i].cum_prob = cum_prob;
        cum_prob += token_probs[i];
        for (uint32_t j = act_prob; j < cum_prob && j < lut_table.size(); ++j) {
            lut_table[j] = i;
        }
        act_prob = cum_prob;
    }
}

// ---- BuildSymbolTables (spec verbatim) ----------------------------------------------------------
// Read num_symbols per-symbol prob tokens from `r` (a DR1 ByteReader positioned at the prob stream),
// then rans_build_look_up_table. token = prob_data & 3; token==3 -> zero (offset+1) probs, i += offset;
// else prob = prob_data>>2, read `token` extra bytes eb with prob |= eb << (8*(j+1)-2).
inline void BuildSymbolTables(ByteReader& r, uint32_t num_symbols, uint32_t rans_precision,
                              std::vector<uint32_t>& lut_table, std::vector<RansSym>& prob_table) {
    std::vector<uint32_t> token_probs(num_symbols, 0u);
    for (uint32_t i = 0; i < num_symbols; ++i) {
        const uint32_t prob_data = r.U8();
        const uint32_t token = prob_data & 3u;
        if (token == 3u) {
            const uint32_t offset = prob_data >> 2;
            for (uint32_t j = 0; j < offset + 1u && (i + j) < num_symbols; ++j) {
                token_probs[i + j] = 0u;
            }
            i += offset;  // outer ++i then advances past the run
        } else {
            uint32_t prob = prob_data >> 2;
            for (uint32_t j = 0; j < token; ++j) {
                const uint32_t eb = r.U8();
                prob |= eb << (8u * (j + 1u) - 2u);
            }
            if (i < num_symbols) token_probs[i] = prob;
        }
    }
    lut_table.assign(rans_precision, 0u);
    RansBuildLookUpTable(token_probs, num_symbols, lut_table, prob_table);
}

// ---- DecodeRawSymbols (spec verbatim, RAW path) -------------------------------------------------
// `r` is the ByteReader over the whole symbol blob. Reads the header (max_bit_length, num_symbols_),
// the prob tables, then the rANS buffer (size bytes); RansInitDecoder reads them BACKWARD from
// buffer[size-1]. Advances `r` past the consumed buffer. Returns false on any read error.
inline bool DecodeRawSymbols(ByteReader& r, uint32_t num_values, std::vector<uint32_t>& out) {
    const uint32_t max_bit_length = r.U8();
    const uint32_t num_symbols = r.VarU32();
    if (r.error) return false;

    uint32_t rans_precision_bits = (3u * max_bit_length) / 2u;
    if (rans_precision_bits > 20u) rans_precision_bits = 20u;
    if (rans_precision_bits < 12u) rans_precision_bits = 12u;
    const uint32_t rans_precision = 1u << rans_precision_bits;
    const uint32_t l_rans_base = rans_precision * 4u;

    std::vector<uint32_t> lut_table;
    std::vector<RansSym>  prob_table;
    BuildSymbolTables(r, num_symbols, rans_precision, lut_table, prob_table);
    if (r.error) return false;

    const uint64_t size = r.VarU64();
    if (r.error) return false;
    // The rANS buffer is the next `size` bytes; it is read BACKWARD from buffer[size-1].
    if (size == 0u || size > r.Remaining()) return false;
    const uint8_t* buffer = r.p + r.pos;
    const int offset = static_cast<int>(size);

    AnsDecoder ans;
    RansInitDecoder(ans, buffer, offset, l_rans_base);

    out.clear();
    out.reserve(num_values);
    for (uint32_t i = 0; i < num_values; ++i) {
        out.push_back(RansRead(ans, l_rans_base, rans_precision, lut_table, prob_table));
    }

    r.Skip(static_cast<std::size_t>(size));  // advance past the consumed rANS buffer
    return !r.error;
}

// ---- DecodeSymbols (spec verbatim) --------------------------------------------------------------
// scheme UI8; kRawSymbols -> DecodeRawSymbols. kTaggedSymbols is a STUB returning false (DR3
// implements the tagged path + the bit reader). num_components only matters for tagged.
inline bool DecodeSymbols(ByteReader& r, uint32_t num_values, uint32_t /*num_components*/,
                          std::vector<uint32_t>& out) {
    const uint8_t scheme = r.U8();
    if (r.error) return false;
    if (scheme == kTaggedSymbols) {
        return false;  // DR3: DecodeTaggedSymbols + the bit reader
    }
    if (scheme == kRawSymbols) {
        return DecodeRawSymbols(r, num_values, out);
    }
    return false;  // unknown scheme
}

// ---- The binary rABS decoder (spec: RabsDescRead) -----------------------------------------------
// An init mirroring RansInitDecoder with l_rans_base = kRabsLBase.
inline void RabsInitDecoder(AnsDecoder& ans, const uint8_t* buf, int offset) {
    RansInitDecoder(ans, buf, offset, kRabsLBase);
}

// RabsDescRead (spec verbatim): p = kRabsP8Precision - p0; renorm-in one byte if state<kRabsLBase;
// quot=x/P; rem=x%P; xn=quot*p; val = rem<p; state = val ? xn+rem : x-xn-p; return val.
inline uint8_t RabsDescRead(AnsDecoder& ans, uint32_t p0) {
    const uint32_t p = kRabsP8Precision - p0;
    if (ans.state < kRabsLBase && ans.buf_offset > 0) {
        ans.state = ans.state * kIoBase + ans.buf[--ans.buf_offset];
    }
    const uint32_t x = ans.state;
    const uint32_t quot = x / kRabsP8Precision;
    const uint32_t rem = x % kRabsP8Precision;
    const uint32_t xn = quot * p;
    const uint8_t val = (rem < p) ? 1u : 0u;
    if (val) {
        ans.state = xn + rem;
    } else {
        ans.state = x - xn - p;
    }
    return val;
}

}  // namespace hf::asset::draco
