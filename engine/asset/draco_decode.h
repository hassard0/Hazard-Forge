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

// Forward declaration: the TAGGED-symbol path (implemented in DR3 below, an allowed completion of the
// DR2 stub -- the DR2 RAW/rABS digests are unaffected).
inline bool DecodeTaggedSymbols(ByteReader& r, uint32_t num_values, uint32_t num_components,
                                std::vector<uint32_t>& out);

// ---- DecodeSymbols (spec verbatim) --------------------------------------------------------------
// scheme UI8; kRawSymbols -> DecodeRawSymbols; kTaggedSymbols -> DecodeTaggedSymbols (DR3).
// num_components only matters for the tagged path.
inline bool DecodeSymbols(ByteReader& r, uint32_t num_values, uint32_t num_components,
                          std::vector<uint32_t>& out) {
    const uint8_t scheme = r.U8();
    if (r.error) return false;
    if (scheme == kTaggedSymbols) {
        return DecodeTaggedSymbols(r, num_values, num_components, out);
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

// ================================ DR3 -- the bit reader + tagged-symbol completion ===============
// APPEND-ONLY: nothing above this line changes. DR1's varint digest 0x2d4aaca6fd14312a and DR2's rANS
// (0xbc91b8ba74fbf8b1) / rABS (0xb6efc1e48524ecd8) stay frozen. DR3 adds (a) a Draco bit reader
// (LSB-first within each byte), (b) the deferred DecodeTaggedSymbols (completing the DR2 stub -- an
// allowed completion, NOT a DR2 modification), and (c) the full edgebreaker connectivity decode.
//
// Implemented spec-driven, clean-room, from the published Draco spec (corner.md, connectivity.decoder.md,
// edgebreaker.decoder.md, edgebreaker.traversal[.valence].md, boundary.decoder.md, sequential.decoder.md,
// variable.descriptions.md). We implement from the SPEC pseudocode; we do NOT copy any draco source.

// ---- DR3 constants (variable.descriptions.md -- FROZEN) -----------------------------------------
constexpr uint8_t  kStandardEdgebreaker = 0;   // STANDARD_EDGEBREAKER
constexpr uint8_t  kValenceEdgebreaker  = 2;   // VALENCE_EDGEBREAKER
constexpr uint8_t  kMeshSeqEncoding     = 0;   // MESH_SEQUENTIAL_ENCODING
constexpr uint8_t  kMeshEbEncoding      = 1;   // MESH_EDGEBREAKER_ENCODING

constexpr int kTopologyC = 0;   // TOPOLOGY_C
constexpr int kTopologyS = 1;   // TOPOLOGY_S
constexpr int kTopologyL = 3;   // TOPOLOGY_L
constexpr int kTopologyR = 5;   // TOPOLOGY_R
constexpr int kTopologyE = 7;   // TOPOLOGY_E

constexpr int kMinValence        = 2;   // MIN_VALENCE
constexpr int kMaxValence        = 7;   // MAX_VALENCE
constexpr int kNumUniqueValences = 6;   // NUM_UNIQUE_VALENCES

constexpr int kLeftFaceEdge   = 0;   // LEFT_FACE_EDGE
constexpr int kRightFaceEdge  = 1;   // RIGHT_FACE_EDGE
constexpr int kInvalidCornerIndex = -1;

// edge_breaker_symbol_to_topology_id (variable.descriptions.md): the per-context valence symbol index
// (0..4) maps to a CLERS topology constant. 0:C(0) 1:S(1) 2:L(3) 3:R(5) 4:E(7).
constexpr int kEbSymbolToTopologyId[5] = { kTopologyC, kTopologyS, kTopologyL, kTopologyR, kTopologyE };

// ---- The Draco bit reader (LSB-first within each byte) ------------------------------------------
// Draco's symbol/topology-split bits are packed LSB-first inside each byte and bytes are consumed
// low-to-high. ReadBits(nbits) returns the next nbits as an unsigned value (bit 0 == the next bit).
// Bounds-checked: reading past the end sets error=true and yields 0 (deterministic, never UB).
struct BitReader {
    const uint8_t* p = nullptr;
    std::size_t    n = 0;        // byte count of the bit-coded span
    std::size_t    byte_pos = 0; // current byte
    uint32_t       bit_pos = 0;  // bit within the current byte (0..7)
    bool           error = false;

    void Reset(const uint8_t* data, std::size_t bytes) {
        p = data; n = bytes; byte_pos = 0; bit_pos = 0; error = false;
    }
    // ResetBitReader(): align to the next whole byte (the spec calls this after the f[1] split bits).
    void ResetBitReader() {
        if (bit_pos != 0) { ++byte_pos; bit_pos = 0; }
    }
    uint32_t ReadBits(uint32_t nbits) {
        uint32_t value = 0;
        for (uint32_t i = 0; i < nbits; ++i) {
            if (byte_pos >= n) { error = true; return value; }
            const uint32_t bit = (static_cast<uint32_t>(p[byte_pos]) >> bit_pos) & 1u;
            value |= (bit << i);
            ++bit_pos;
            if (bit_pos == 8u) { bit_pos = 0; ++byte_pos; }
        }
        return value;
    }
    // Bytes consumed so far (rounded up if mid-byte) -- to advance an outer ByteReader past the span.
    std::size_t BytesConsumed() const {
        return byte_pos + (bit_pos != 0 ? 1u : 0u);
    }
};

// ---- DecodeTaggedSymbols (completing the DR2 stub) ----------------------------------------------
// The TAGGED scheme (rans.decoding.md): a per-group tag rANS over TAGGED_RANS_BASE/TAGGED_RANS_PRECISION
// selects the bit-length of each component group, then the raw component values are read bit-packed
// (size bits each) via the bit reader. `r` is the ByteReader positioned at the tagged blob. Reads the
// prob tables for the tag symbols, the rANS buffer, then the bit-coded raw values; advances `r` past
// the consumed buffer. Returns false on any read error.
inline bool DecodeTaggedSymbols(ByteReader& r, uint32_t num_values, uint32_t num_components,
                                std::vector<uint32_t>& out) {
    // The tag-symbol prob table (num_symbols = number of distinct bit-lengths) at fixed precision.
    std::vector<uint32_t> lut_table;
    std::vector<RansSym>  prob_table;
    const uint32_t num_symbols = r.VarU32();
    if (r.error) return false;
    lut_table.assign(kTaggedRansPrec, 0u);
    {
        std::vector<uint32_t> token_probs(num_symbols, 0u);
        for (uint32_t i = 0; i < num_symbols; ++i) {
            const uint32_t prob_data = r.U8();
            const uint32_t token = prob_data & 3u;
            if (token == 3u) {
                const uint32_t offset = prob_data >> 2;
                for (uint32_t j = 0; j < offset + 1u && (i + j) < num_symbols; ++j)
                    token_probs[i + j] = 0u;
                i += offset;
            } else {
                uint32_t prob = prob_data >> 2;
                for (uint32_t j = 0; j < token; ++j) {
                    const uint32_t eb = r.U8();
                    prob |= eb << (8u * (j + 1u) - 2u);
                }
                if (i < num_symbols) token_probs[i] = prob;
            }
        }
        RansBuildLookUpTable(token_probs, num_symbols, lut_table, prob_table);
    }
    if (r.error) return false;

    // The rANS buffer holding the tags is read BACKWARD from its end (TAGGED_RANS_BASE).
    const uint64_t size = r.VarU64();
    if (r.error) return false;
    if (size == 0u || size > r.Remaining()) return false;
    const uint8_t* rans_buf = r.p + r.pos;
    AnsDecoder ans;
    RansInitDecoder(ans, rans_buf, static_cast<int>(size), kTaggedRansBase);
    r.Skip(static_cast<std::size_t>(size));
    if (r.error) return false;

    // The remaining bytes of `r` are the bit-packed raw component values. Read them LSB-first.
    BitReader bits;
    bits.Reset(r.p + r.pos, r.Remaining());

    out.clear();
    out.reserve(num_values);
    const uint32_t nc = (num_components == 0u) ? 1u : num_components;
    uint32_t produced = 0;
    while (produced < num_values) {
        // One tag per component group: the symbol value IS the bit-length of this group's values.
        const uint32_t bit_length = RansRead(ans, kTaggedRansBase, kTaggedRansPrec, lut_table, prob_table);
        for (uint32_t c = 0; c < nc && produced < num_values; ++c) {
            const uint32_t v = (bit_length == 0u) ? 0u : bits.ReadBits(bit_length);
            out.push_back(v);
            ++produced;
        }
        if (bits.error) return false;
    }
    // Advance r past the consumed bit bytes.
    r.Skip(bits.BytesConsumed());
    return !r.error;
}

// ================================ DR3 -- edgebreaker connectivity decode =========================
// The decode result: the cube's triangle topology as three corner-vertex lists. Face f's three corner
// vertices are (face_to_vertex[0][f], face_to_vertex[1][f], face_to_vertex[2][f]).
struct Connectivity {
    uint32_t num_faces = 0, num_vertices = 0;
    std::vector<uint32_t> face_to_vertex[3];
    bool ok = false;
};

// The whole edgebreaker decoder state in one struct (mirrors the spec's globals). All vectors grow as
// needed; every index is bounds-guarded so a malformed stream sets ok=false and never reads out of
// range. att_dec=0 (MESH_VERTEX_ATTRIBUTE) throughout DR3, so Opposite()==PosOpposite() and the
// attribute-seam paths are stubbed.
struct EdgebreakerDecoder {
    // ---- parsed header fields ----
    uint8_t  edgebreaker_traversal_type = 0;
    uint32_t num_encoded_vertices = 0;
    uint32_t num_faces = 0;
    uint8_t  num_attribute_data = 0;
    uint32_t num_encoded_symbols = 0;
    uint32_t num_encoded_split_symbols = 0;

    // ---- topology split events ----
    std::vector<uint32_t> source_symbol_id;   // decoder-side queue (popped back-to-front)
    std::vector<uint32_t> split_symbol_id;
    std::vector<uint8_t>  source_edge_bit;

    // ---- the mesh corner table being built ----
    std::vector<int> face_to_vertex[3];       // signed: vertices indexed >=0, may be relabeled
    std::vector<int> opposite_corners_;       // default -1
    std::vector<int> corner_to_vertex_map_;   // corner -> vertex (default -1)
    std::vector<int> vertex_corners_;         // vertex -> a representative corner (default -1)
    std::vector<uint8_t> is_vert_hole_;
    std::vector<int> vertex_valences_;        // VALENCE only

    // ---- traversal state ----
    int last_symbol_ = -1;
    int active_context_ = -1;
    int last_vert_added = -1;
    std::vector<int> active_corner_stack;
    std::vector<int> topology_split_id;
    std::vector<int> split_active_corners;

    // ---- the bit-coded standard symbol stream + the start-face rABS buffer ----
    BitReader symbol_bits;                     // STANDARD: eb_symbol_buffer
    const uint8_t* eb_start_face_buffer = nullptr;
    std::size_t    eb_start_face_buffer_size = 0;
    uint8_t        eb_start_face_buffer_prob_zero = 0;

    // ---- VALENCE state ----
    std::vector<std::vector<uint32_t>> ebv_context_symbols;  // [NUM_UNIQUE_VALENCES][...]
    std::vector<uint32_t>              ebv_context_counters;

    bool ok = true;

    void Fail() { ok = false; }

    // ---- corner.md primitives (exact) ----
    static int Next(int c)     { return c < 0 ? c : ((c % 3) == 2 ? c - 2 : c + 1); }
    static int Previous(int c) { return c < 0 ? c : ((c % 3) == 0 ? c + 2 : c - 1); }

    int PosOpposite(int c) const {
        if (c < 0 || static_cast<std::size_t>(c) >= opposite_corners_.size()) return -1;
        return opposite_corners_[c];
    }
    // att_dec=0 throughout DR3 => Opposite == PosOpposite (attribute-seam paths are DR4+; stubbed).
    int Opposite(int /*att_dec*/, int c) const { return PosOpposite(c); }
    int SwingLeft(int att_dec, int c)  { return Next(Opposite(att_dec, Next(c))); }
    int SwingRight(int att_dec, int c) { return Previous(Opposite(att_dec, Previous(c))); }

    void EnsureCorner(int c) {
        if (c < 0) return;
        const std::size_t need = static_cast<std::size_t>(c) + 1u;
        if (opposite_corners_.size() < need) opposite_corners_.resize(need, -1);
        if (corner_to_vertex_map_.size() < need) corner_to_vertex_map_.resize(need, -1);
    }
    void EnsureVertex(int v) {
        if (v < 0) return;
        const std::size_t need = static_cast<std::size_t>(v) + 1u;
        if (vertex_corners_.size() < need) vertex_corners_.resize(need, -1);
        if (is_vert_hole_.size() < need)   is_vert_hole_.resize(need, 1u);
        if (edgebreaker_traversal_type == kValenceEdgebreaker &&
            vertex_valences_.size() < need) vertex_valences_.resize(need, 0);
    }

    void SetOppositeCorners(int a, int b) {
        EnsureCorner(a); EnsureCorner(b);
        if (a < 0 || b < 0) { Fail(); return; }
        opposite_corners_[a] = b;
        opposite_corners_[b] = a;
    }

    void MapCornerToVertex(int corner_id, int vert_id) {
        EnsureCorner(corner_id);
        if (corner_id < 0) { Fail(); return; }
        corner_to_vertex_map_[corner_id] = vert_id;
        if (vert_id >= 0) { EnsureVertex(vert_id); vertex_corners_[vert_id] = corner_id; }
    }

    int CornerToVert(int /*att_dec*/, int corner_id) const {
        if (corner_id < 0) return -1;
        const int local = corner_id % 3;
        const std::size_t face = static_cast<std::size_t>(corner_id) / 3u;
        if (face >= face_to_vertex[0].size()) return -1;
        if (local == 0) return face_to_vertex[0][face];
        if (local == 1) return face_to_vertex[1][face];
        return face_to_vertex[2][face];
    }
    // CornerToVerts -> (v,n,p) for att_dec=0 (position attribute).
    void CornerToVerts(int /*att_dec*/, int corner_id, int& v, int& nx, int& pv) const {
        v = nx = pv = -1;
        if (corner_id < 0) return;
        const int local = corner_id % 3;
        const std::size_t face = static_cast<std::size_t>(corner_id) / 3u;
        if (face >= face_to_vertex[0].size()) return;
        const int a = face_to_vertex[0][face], b = face_to_vertex[1][face], c = face_to_vertex[2][face];
        if (local == 0) { v = a; nx = b; pv = c; }
        else if (local == 1) { v = b; nx = c; pv = a; }
        else { v = c; nx = a; pv = b; }
    }

    // ReplaceVerts(from,to): relabel every occurrence in the three lists.
    void ReplaceVerts(int from, int to) {
        for (int k = 0; k < 3; ++k) {
            for (std::size_t i = 0; i < face_to_vertex[k].size(); ++i)
                if (face_to_vertex[k][i] == from) face_to_vertex[k][i] = to;
        }
    }

    // UpdateCornersAfterMerge(c,v): walk SwingLeft from Next(opp) remapping corners to v.
    void UpdateCornersAfterMerge(int c, int v) {
        const int opp_corner = PosOpposite(c);
        if (opp_corner >= 0) {
            int corner_n = Next(opp_corner);
            int guard = 0;
            while (corner_n >= 0) {
                MapCornerToVertex(corner_n, v);
                corner_n = SwingLeft(0, corner_n);
                if (++guard > 1000000) { Fail(); break; }
            }
        }
    }

    // ---- topology split helpers (edgebreaker.decoder.md) ----
    bool IsTopologySplit(int encoder_symbol_id, int& out_face_edge, int& out_split_id) {
        if (source_symbol_id.empty()) return false;
        if (static_cast<int>(source_symbol_id.back()) != encoder_symbol_id) return false;
        out_face_edge = static_cast<int>(source_edge_bit.back());
        out_split_id  = static_cast<int>(split_symbol_id.back());
        source_edge_bit.pop_back();
        split_symbol_id.pop_back();
        source_symbol_id.pop_back();
        return true;
    }

    // ---- the C/S/L/R/E switch -- the heart (edgebreaker.decoder.md NewActiveCornerReached) ----
    void NewActiveCornerReached(int new_corner, int symbol_id) {
        bool check_topology_split = false;
        EnsureCorner(new_corner + 2);
        int vert = -1, next = -1, prev = -1;

        switch (last_symbol_) {
        case kTopologyC: {
            if (active_corner_stack.empty()) { Fail(); return; }
            int corner_a = active_corner_stack.back();
            int corner_b = Previous(corner_a);
            int guard = 0;
            while (PosOpposite(corner_b) >= 0) {
                corner_b = Previous(PosOpposite(corner_b));
                if (++guard > 1000000) { Fail(); return; }
            }
            SetOppositeCorners(corner_a, new_corner + 1);
            SetOppositeCorners(corner_b, new_corner + 2);
            active_corner_stack.back() = new_corner;

            vert = CornerToVert(0, Next(corner_a));
            next = CornerToVert(0, Next(corner_b));
            prev = CornerToVert(0, Previous(corner_a));
            if (edgebreaker_traversal_type == kValenceEdgebreaker) {
                EnsureVertex(next); EnsureVertex(prev);
                if (next >= 0) vertex_valences_[next] += 1;
                if (prev >= 0) vertex_valences_[prev] += 1;
            }
            face_to_vertex[0].push_back(vert);
            face_to_vertex[1].push_back(next);
            face_to_vertex[2].push_back(prev);
            EnsureVertex(vert);
            if (vert >= 0) is_vert_hole_[vert] = 0u;
            MapCornerToVertex(new_corner, vert);
            MapCornerToVertex(new_corner + 1, next);
            MapCornerToVertex(new_corner + 2, prev);
            break;
        }
        case kTopologyS: {
            if (active_corner_stack.empty()) { Fail(); return; }
            int corner_b = active_corner_stack.back(); active_corner_stack.pop_back();
            for (std::size_t i = 0; i < topology_split_id.size(); ++i) {
                if (topology_split_id[i] == symbol_id)
                    active_corner_stack.push_back(split_active_corners[i]);
            }
            if (active_corner_stack.empty()) { Fail(); return; }
            int corner_a = active_corner_stack.back();
            SetOppositeCorners(corner_a, new_corner + 2);
            SetOppositeCorners(corner_b, new_corner + 1);
            active_corner_stack.back() = new_corner;

            vert = CornerToVert(0, Previous(corner_a));
            next = CornerToVert(0, Next(corner_a));
            prev = CornerToVert(0, Previous(corner_b));
            MapCornerToVertex(new_corner, vert);
            MapCornerToVertex(new_corner + 1, next);
            MapCornerToVertex(new_corner + 2, prev);
            int corner_n = Next(corner_b);
            int vertex_n = CornerToVert(0, corner_n);
            if (edgebreaker_traversal_type == kValenceEdgebreaker) {
                EnsureVertex(vert); EnsureVertex(vertex_n);
                if (vert >= 0 && vertex_n >= 0) vertex_valences_[vert] += vertex_valences_[vertex_n];
            }
            ReplaceVerts(vertex_n, vert);
            if (edgebreaker_traversal_type == kValenceEdgebreaker) {
                EnsureVertex(next); EnsureVertex(prev);
                if (next >= 0) vertex_valences_[next] += 1;
                if (prev >= 0) vertex_valences_[prev] += 1;
            }
            face_to_vertex[0].push_back(vert);
            face_to_vertex[1].push_back(next);
            face_to_vertex[2].push_back(prev);
            UpdateCornersAfterMerge(new_corner + 1, vert);
            if (vertex_n >= 0) { EnsureVertex(vertex_n); vertex_corners_[vertex_n] = kInvalidCornerIndex; }
            break;
        }
        case kTopologyR: {
            if (active_corner_stack.empty()) { Fail(); return; }
            int corner_a = active_corner_stack.back();
            int opp_corner = new_corner + 2;
            SetOppositeCorners(opp_corner, corner_a);
            active_corner_stack.back() = new_corner;
            check_topology_split = true;

            vert = CornerToVert(0, Previous(corner_a));
            next = CornerToVert(0, Next(corner_a));
            prev = ++last_vert_added;
            if (edgebreaker_traversal_type == kValenceEdgebreaker) {
                EnsureVertex(vert); EnsureVertex(next); EnsureVertex(prev);
                if (vert >= 0) vertex_valences_[vert] += 1;
                if (next >= 0) vertex_valences_[next] += 1;
                if (prev >= 0) vertex_valences_[prev] += 2;
            }
            face_to_vertex[0].push_back(vert);
            face_to_vertex[1].push_back(next);
            face_to_vertex[2].push_back(prev);
            MapCornerToVertex(new_corner + 2, prev);
            MapCornerToVertex(new_corner, vert);
            MapCornerToVertex(new_corner + 1, next);
            break;
        }
        case kTopologyL: {
            if (active_corner_stack.empty()) { Fail(); return; }
            int corner_a = active_corner_stack.back();
            int opp_corner = new_corner + 1;
            SetOppositeCorners(opp_corner, corner_a);
            active_corner_stack.back() = new_corner;
            check_topology_split = true;

            vert = CornerToVert(0, Next(corner_a));
            next = ++last_vert_added;
            prev = CornerToVert(0, Previous(corner_a));
            if (edgebreaker_traversal_type == kValenceEdgebreaker) {
                EnsureVertex(vert); EnsureVertex(next); EnsureVertex(prev);
                if (vert >= 0) vertex_valences_[vert] += 1;
                if (next >= 0) vertex_valences_[next] += 2;
                if (prev >= 0) vertex_valences_[prev] += 1;
            }
            face_to_vertex[0].push_back(vert);
            face_to_vertex[1].push_back(next);
            face_to_vertex[2].push_back(prev);
            MapCornerToVertex(new_corner + 2, prev);
            MapCornerToVertex(new_corner, vert);
            MapCornerToVertex(new_corner + 1, next);
            break;
        }
        case kTopologyE: {
            active_corner_stack.push_back(new_corner);
            check_topology_split = true;
            vert = last_vert_added + 1;
            next = vert + 1;
            prev = next + 1;
            if (edgebreaker_traversal_type == kValenceEdgebreaker) {
                EnsureVertex(vert); EnsureVertex(next); EnsureVertex(prev);
                vertex_valences_[vert] += 2;
                vertex_valences_[next] += 2;
                vertex_valences_[prev] += 2;
            }
            face_to_vertex[0].push_back(vert);
            face_to_vertex[1].push_back(next);
            face_to_vertex[2].push_back(prev);
            last_vert_added = prev;
            MapCornerToVertex(new_corner, vert);
            MapCornerToVertex(new_corner + 1, next);
            MapCornerToVertex(new_corner + 2, prev);
            break;
        }
        default:
            Fail();
            return;
        }

        if (edgebreaker_traversal_type == kValenceEdgebreaker) {
            EnsureVertex(next);
            int active_valence = (next >= 0) ? vertex_valences_[next] : 0;
            int clamped_valence = active_valence;
            if (active_valence < kMinValence) clamped_valence = kMinValence;
            else if (active_valence > kMaxValence) clamped_valence = kMaxValence;
            active_context_ = clamped_valence - kMinValence;
        }

        if (check_topology_split) {
            int encoder_symbol_id = static_cast<int>(num_encoded_symbols) - symbol_id - 1;
            int split_edge = 0, enc_split_id = 0;
            while (IsTopologySplit(encoder_symbol_id, split_edge, enc_split_id)) {
                if (active_corner_stack.empty()) { Fail(); return; }
                int act_top_corner = active_corner_stack.back();
                int new_active_corner =
                    (split_edge == kRightFaceEdge) ? Next(act_top_corner) : Previous(act_top_corner);
                int dec_split_id = static_cast<int>(num_encoded_symbols) - enc_split_id - 1;
                topology_split_id.push_back(dec_split_id);
                split_active_corners.push_back(new_active_corner);
            }
        }
    }

    // ---- ParseEdgebreakerStandardSymbol (edgebreaker.decoder.md) ----
    void ParseEdgebreakerStandardSymbol() {
        uint32_t symbol = symbol_bits.ReadBits(1);
        if (static_cast<int>(symbol) != kTopologyC) {
            uint32_t suffix = symbol_bits.ReadBits(2);
            symbol |= (suffix << 1);
        }
        last_symbol_ = static_cast<int>(symbol);
        if (symbol_bits.error) Fail();
    }

    // ---- EdgebreakerValenceDecodeSymbol (edgebreaker.traversal.valence.md) ----
    void EdgebreakerValenceDecodeSymbol() {
        if (active_context_ != -1) {
            const std::size_t ctx = static_cast<std::size_t>(active_context_);
            if (ctx >= ebv_context_symbols.size() || ctx >= ebv_context_counters.size()) { Fail(); return; }
            if (ebv_context_counters[ctx] == 0u) { Fail(); return; }
            const uint32_t idx = --ebv_context_counters[ctx];
            if (idx >= ebv_context_symbols[ctx].size()) { Fail(); return; }
            const uint32_t symbol_id = ebv_context_symbols[ctx][idx];
            if (symbol_id >= 5u) { Fail(); return; }
            last_symbol_ = kEbSymbolToTopologyId[symbol_id];
        } else {
            last_symbol_ = kTopologyE;
        }
    }

    void EdgebreakerDecodeSymbol() {
        if (edgebreaker_traversal_type == kValenceEdgebreaker) EdgebreakerValenceDecodeSymbol();
        else if (edgebreaker_traversal_type == kStandardEdgebreaker) ParseEdgebreakerStandardSymbol();
        else Fail();
    }

    // ---- ProcessInteriorEdges (edgebreaker.decoder.md) ----
    void ProcessInteriorEdges() {
        if (eb_start_face_buffer == nullptr || eb_start_face_buffer_size == 0) {
            // No interior-edge buffer: nothing to weld (a closed manifold may still have an empty buffer).
            // The spec always inits the rANS decoder; if there is no buffer we simply skip the welds.
            return;
        }
        AnsDecoder ans;
        RansInitDecoder(ans, eb_start_face_buffer, static_cast<int>(eb_start_face_buffer_size), kLRansBase);

        int guard = 0;
        while (!active_corner_stack.empty()) {
            int corner_a = active_corner_stack.back(); active_corner_stack.pop_back();
            const uint8_t interior_face = RabsDescRead(ans, eb_start_face_buffer_prob_zero);
            if (interior_face) {
                int corner_b = Previous(corner_a);
                int g2 = 0;
                while (PosOpposite(corner_b) >= 0) {
                    corner_b = Previous(PosOpposite(corner_b));
                    if (++g2 > 1000000) { Fail(); return; }
                }
                int corner_c = Next(corner_a);
                g2 = 0;
                while (PosOpposite(corner_c) >= 0) {
                    corner_c = Next(PosOpposite(corner_c));
                    if (++g2 > 1000000) { Fail(); return; }
                }
                int new_corner = static_cast<int>(face_to_vertex[0].size()) * 3;
                EnsureCorner(new_corner + 2);
                SetOppositeCorners(new_corner, corner_a);
                SetOppositeCorners(new_corner + 1, corner_b);
                SetOppositeCorners(new_corner + 2, corner_c);

                int tv, next_a, tp, next_b, next_c;
                CornerToVerts(0, corner_a, tv, next_a, tp);
                CornerToVerts(0, corner_b, tv, next_b, tp);
                CornerToVerts(0, corner_c, tv, next_c, tp);
                MapCornerToVertex(new_corner, next_b);
                MapCornerToVertex(new_corner + 1, next_c);
                MapCornerToVertex(new_corner + 2, next_a);
                face_to_vertex[0].push_back(next_b);
                face_to_vertex[1].push_back(next_c);
                face_to_vertex[2].push_back(next_a);
                EnsureVertex(next_a); EnsureVertex(next_b); EnsureVertex(next_c);
                if (next_b >= 0) is_vert_hole_[next_b] = 0u;
                if (next_c >= 0) is_vert_hole_[next_c] = 0u;
                if (next_a >= 0) is_vert_hole_[next_a] = 0u;
            }
            if (++guard > 100000000) { Fail(); return; }
        }
    }

    // ---- DecodeEdgeBreakerConnectivity (edgebreaker.decoder.md) ----
    void DecodeEdgeBreakerConnectivity() {
        is_vert_hole_.assign(num_encoded_vertices + num_encoded_split_symbols, 1u);
        last_vert_added = -1;
        for (uint32_t i = 0; i < num_encoded_symbols; ++i) {
            EdgebreakerDecodeSymbol();
            if (!ok) return;
            int corner = static_cast<int>(3u * i);
            NewActiveCornerReached(corner, static_cast<int>(i));
            if (!ok) return;
        }
        ProcessInteriorEdges();
    }
};

// ---- ParseEdgebreakerConnectivityData (edgebreaker.decoder.md) -----------------------------------
inline void ParseEdgebreakerConnectivityData(ByteReader& r, EdgebreakerDecoder& d) {
    d.edgebreaker_traversal_type   = r.U8();
    d.num_encoded_vertices         = r.VarU32();
    d.num_faces                    = r.VarU32();
    d.num_attribute_data           = r.U8();
    d.num_encoded_symbols          = r.VarU32();
    d.num_encoded_split_symbols    = r.VarU32();
    if (r.error) d.Fail();
}

// ---- DecodeTopologySplitEvents (ParseTopologySplitEvents + ProcessSplitData) ---------------------
// The split source/split-symbol ids + the f[1] source_edge_bit run, then ResetBitReader (byte-align).
inline void DecodeTopologySplitEvents(ByteReader& r, EdgebreakerDecoder& d) {
    const uint32_t num_topology_splits = r.VarU32();
    if (r.error) { d.Fail(); return; }
    std::vector<uint32_t> source_id_delta(num_topology_splits, 0u);
    std::vector<uint32_t> split_id_delta(num_topology_splits, 0u);
    for (uint32_t i = 0; i < num_topology_splits; ++i) {
        source_id_delta[i] = r.VarU32();
        split_id_delta[i]  = r.VarU32();
    }
    // The source_edge_bit f[1] values are bit-packed (LSB-first) over the remaining bytes; after the
    // run the reader is byte-realigned (ResetBitReader). We read them from the current cursor.
    BitReader bits;
    bits.Reset(r.p + r.pos, r.Remaining());
    std::vector<uint8_t> source_edge_bit(num_topology_splits, 0u);
    for (uint32_t i = 0; i < num_topology_splits; ++i)
        source_edge_bit[i] = static_cast<uint8_t>(bits.ReadBits(1));
    bits.ResetBitReader();
    r.Skip(bits.BytesConsumed());
    if (r.error || bits.error) { d.Fail(); return; }

    // ProcessSplitData: source_symbol_id[i] = source_id_delta[i] + last_id;
    // split_symbol_id[i] = source_symbol_id[i] - split_id_delta[i]; last_id = source_symbol_id[i].
    d.source_symbol_id.clear();
    d.split_symbol_id.clear();
    d.source_edge_bit = source_edge_bit;
    uint32_t last_id = 0;
    for (uint32_t i = 0; i < num_topology_splits; ++i) {
        const uint32_t src = source_id_delta[i] + last_id;
        d.source_symbol_id.push_back(src);
        d.split_symbol_id.push_back(src - split_id_delta[i]);
        last_id = src;
    }
}

// ---- EdgebreakerTraversalStart (edgebreaker.traversal[.valence].md) ------------------------------
// STANDARD: parse the bit-coded symbol buffer + the start-face rABS buffer + the attribute conn data.
// VALENCE : parse the start-face + attribute conn data, init valences, read per-context symbol streams.
inline void EdgebreakerTraversalStart(ByteReader& r, EdgebreakerDecoder& d) {
    d.last_symbol_ = -1;
    d.active_context_ = -1;

    if (d.edgebreaker_traversal_type == kStandardEdgebreaker) {
        // ParseEdgebreakerTraversalStandardSymbolData: sz varUI64, then eb_symbol_buffer[sz].
        const uint64_t sym_sz = r.VarU64();
        if (r.error || sym_sz > r.Remaining()) { d.Fail(); return; }
        d.symbol_bits.Reset(r.p + r.pos, static_cast<std::size_t>(sym_sz));
        r.Skip(static_cast<std::size_t>(sym_sz));

        // ParseEdgebreakerTraversalStandardFaceData: prob_zero UI8, sz varUI32, buffer[sz].
        d.eb_start_face_buffer_prob_zero = r.U8();
        const uint32_t face_sz = r.VarU32();
        if (r.error || face_sz > r.Remaining()) { d.Fail(); return; }
        d.eb_start_face_buffer = r.p + r.pos;
        d.eb_start_face_buffer_size = face_sz;
        r.Skip(face_sz);

        // ParseEdgebreakerTraversalStandardAttributeConnectivityData (consumed, used by DR4 boundary).
        for (uint32_t i = 0; i < d.num_attribute_data; ++i) {
            (void)r.U8();                       // attribute_connectivity_decoders_prob_zero[i]
            const uint32_t sz = r.VarU32();     // attribute_connectivity_decoders_size[i]
            if (r.error || sz > r.Remaining()) { d.Fail(); return; }
            r.Skip(sz);                         // attribute_connectivity_decoders_buffer[i]
        }
        if (r.error) d.Fail();
    } else if (d.edgebreaker_traversal_type == kValenceEdgebreaker) {
        // EdgeBreakerTraversalValenceStart: face data + attribute conn data first.
        d.eb_start_face_buffer_prob_zero = r.U8();
        const uint32_t face_sz = r.VarU32();
        if (r.error || face_sz > r.Remaining()) { d.Fail(); return; }
        d.eb_start_face_buffer = r.p + r.pos;
        d.eb_start_face_buffer_size = face_sz;
        r.Skip(face_sz);

        for (uint32_t i = 0; i < d.num_attribute_data; ++i) {
            (void)r.U8();
            const uint32_t sz = r.VarU32();
            if (r.error || sz > r.Remaining()) { d.Fail(); return; }
            r.Skip(sz);
        }

        d.vertex_valences_.assign(d.num_encoded_vertices + d.num_encoded_split_symbols, 0);
        d.ebv_context_symbols.assign(kNumUniqueValences, std::vector<uint32_t>());
        d.ebv_context_counters.assign(kNumUniqueValences, 0u);
        for (int i = 0; i < kNumUniqueValences; ++i) {
            const uint32_t cnt = r.VarU32();        // ParseValenceContextCounters(i)
            if (r.error) { d.Fail(); return; }
            d.ebv_context_counters[i] = cnt;
            if (cnt > 0u) {
                std::vector<uint32_t> syms;
                if (!DecodeSymbols(r, cnt, 1u, syms)) { d.Fail(); return; }
                d.ebv_context_symbols[i] = syms;
            }
        }
    } else {
        d.Fail();
    }
}

// ---- DecodeEdgebreakerConnectivityData (edgebreaker.decoder.md) ----------------------------------
inline Connectivity DecodeEdgebreakerConnectivityData(ByteReader& r) {
    EdgebreakerDecoder d;
    ParseEdgebreakerConnectivityData(r, d);
    if (!d.ok) return Connectivity{};
    DecodeTopologySplitEvents(r, d);
    if (!d.ok) return Connectivity{};
    EdgebreakerTraversalStart(r, d);
    if (!d.ok) return Connectivity{};
    d.DecodeEdgeBreakerConnectivity();

    Connectivity out;
    out.ok = d.ok && (static_cast<uint32_t>(d.face_to_vertex[0].size()) == d.num_faces);
    out.num_faces = static_cast<uint32_t>(d.face_to_vertex[0].size());
    out.num_vertices = d.num_encoded_vertices + d.num_encoded_split_symbols;
    if (out.ok) {
        for (int k = 0; k < 3; ++k) {
            out.face_to_vertex[k].reserve(d.face_to_vertex[k].size());
            for (int v : d.face_to_vertex[k])
                out.face_to_vertex[k].push_back(static_cast<uint32_t>(v < 0 ? 0 : v));
        }
    }
    return out;
}

// ---- DecodeConnectivity (connectivity.decoder.md DecodeConnectivityData dispatch) ----------------
// The decode entry: after the header (and optional metadata), dispatch on encoderMethod. For DR3 only
// MESH_EDGEBREAKER is implemented; MESH_SEQUENTIAL returns ok=false (a DR-later stub).
inline Connectivity DecodeConnectivity(ByteReader& r, const DracoHeader& h) {
    if (!h.valid) return Connectivity{};
    if (h.encoderMethod == kMeshEbEncoding) {
        return DecodeEdgebreakerConnectivityData(r);
    }
    // MESH_SEQUENTIAL_ENCODING (and anything else) -- not decoded in DR3.
    return Connectivity{};
}

// ================================ DR4 -- attribute decode (positions) ============================
// APPEND-ONLY: nothing above this line is touched. DR1's varint digest 0x2d4aaca6fd14312a, DR2's rANS
// (0xbc91b8ba74fbf8b1) / rABS (0xb6efc1e48524ecd8), and DR3's connectivity (0x1f478b2e11afa703) stay
// frozen. DR4 turns the DR3 cube TOPOLOGY into the cube's actual VERTEX POSITIONS: it parses the
// attribute-decoder metadata, walks the corner table to order the encoded attribute values
// (EdgeBreakerTraverser_ProcessCorner), decodes the quantized integer POSITION residuals through the DR2
// rANS, runs the PARALLELOGRAM prediction + WRAP transform to reconstruct the integer values, and
// dequantizes them (the ONE float step). Implemented spec-driven, clean-room, from the published Draco
// spec (attributes.decoder.md, sequential.decoder.md, sequential.integer.attribute.decoder.md,
// sequential.quantization.attribute.decoder.md, prediction.decoder.md, prediction.parallelogram.decoder.md,
// prediction.wrap.transform.md, edgebreaker.traversal.md, boundary.decoder.md, corner.md,
// variable.descriptions.md). We implement from the SPEC pseudocode; we do NOT copy any draco source.
//
// SCOPE: the Box (and the overwhelmingly common glTF case) has ONE vertex POSITION attribute decoder
// (decoder 0: MESH_VERTEX_ATTRIBUTE, SEQUENTIAL_ATTRIBUTE_ENCODER_QUANTIZATION, MESH_PREDICTION_PARALLELOGRAM
// + PREDICTION_TRANSFORM_WRAP). POSITION is the headline and is fully decoded. Additional decoders
// (e.g. a MESH_CORNER_ATTRIBUTE NORMAL with attribute seams) are parsed but their values are decoded
// best-effort; the position decode never depends on them. Every read is bounds-checked: a malformed or
// unsupported stream sets ok=false and never reads out of range / never crashes.

// ---- DR4 constants (variable.descriptions.md -- FROZEN) -----------------------------------------
constexpr uint8_t  kSeqAttEncGeneric      = 0;   // SEQUENTIAL_ATTRIBUTE_ENCODER_GENERIC
constexpr uint8_t  kSeqAttEncInteger      = 1;   // SEQUENTIAL_ATTRIBUTE_ENCODER_INTEGER
constexpr uint8_t  kSeqAttEncQuantization = 2;   // SEQUENTIAL_ATTRIBUTE_ENCODER_QUANTIZATION
constexpr uint8_t  kSeqAttEncNormals      = 3;   // SEQUENTIAL_ATTRIBUTE_ENCODER_NORMALS

constexpr uint8_t  kMeshVertexAttribute = 0;     // att_dec_decoder_type MESH_VERTEX_ATTRIBUTE
constexpr uint8_t  kMeshCornerAttribute = 1;     // att_dec_decoder_type MESH_CORNER_ATTRIBUTE

constexpr uint8_t  kMeshTraversalDepthFirst       = 0;  // MESH_TRAVERSAL_DEPTH_FIRST
constexpr uint8_t  kMeshTraversalPredictionDegree = 1;  // MESH_TRAVERSAL_PREDICTION_DEGREE

constexpr int8_t   kPredictionNone           = -2;  // PREDICTION_NONE
constexpr int8_t   kPredictionDifference     = 0;   // PREDICTION_DIFFERENCE
constexpr int8_t   kMeshPredParallelogram    = 1;   // MESH_PREDICTION_PARALLELOGRAM
constexpr int8_t   kMeshPredConstrainedMulti = 4;   // MESH_PREDICTION_CONSTRAINED_MULTI_PARALLELOGRAM

constexpr int8_t   kPredTransformWrap   = 1;   // PREDICTION_TRANSFORM_WRAP
constexpr int8_t   kPredTransformNormal = 3;   // PREDICTION_TRANSFORM_NORMAL_OCTAHEDRON_CANONICALIZED

constexpr uint8_t  kAttPosition = 0;   // att_dec_att_type GeometryAttribute::POSITION
constexpr uint8_t  kAttNormal   = 1;   // att_dec_att_type GeometryAttribute::NORMAL

// ---- Per-attribute parsed metadata (attributes.decoder.md ParseAttributeDecodersData) -----------
struct DracoAttribute {
    uint8_t  att_type = 0;        // POSITION / NORMAL / ...
    uint8_t  data_type = 0;       // DT_* (9 == DT_FLOAT32 for the Box)
    uint8_t  num_components = 0;  // 3 for a position
    uint8_t  normalized = 0;
    uint32_t unique_id = 0;
    uint8_t  seq_att_dec_decoder_type = 0;  // SEQUENTIAL_ATTRIBUTE_ENCODER_*
};

struct DracoAttributeDecoder {
    uint8_t data_id = 0;
    uint8_t decoder_type = 0;     // MESH_VERTEX_ATTRIBUTE / MESH_CORNER_ATTRIBUTE
    uint8_t traversal_method = 0; // MESH_TRAVERSAL_DEPTH_FIRST / ...
    std::vector<DracoAttribute> attributes;
};

// ---- ParseAttributeDecodersData (attributes.decoder.md, exact) ----------------------------------
// `r` is positioned just past the connectivity data. For MESH_EDGEBREAKER each decoder carries
// data_id/decoder_type/traversal_method; then per decoder the attribute count + per-attribute
// type/data_type/num_components/normalized/unique_id; then the per-attribute seq decoder types.
inline bool ParseAttributeDecodersData(ByteReader& r, uint8_t encoder_method,
                                       std::vector<DracoAttributeDecoder>& out) {
    const uint32_t num_decoders = r.U8();
    if (r.error) return false;
    out.assign(num_decoders, DracoAttributeDecoder{});
    if (encoder_method == kMeshEdgebreaker) {
        for (uint32_t i = 0; i < num_decoders; ++i) {
            out[i].data_id          = r.U8();
            out[i].decoder_type     = r.U8();
            out[i].traversal_method = r.U8();
        }
    }
    for (uint32_t i = 0; i < num_decoders; ++i) {
        const uint32_t na = r.VarU32();
        if (r.error) return false;
        out[i].attributes.assign(na, DracoAttribute{});
        for (uint32_t j = 0; j < na; ++j) {
            out[i].attributes[j].att_type       = r.U8();
            out[i].attributes[j].data_type      = r.U8();
            out[i].attributes[j].num_components  = r.U8();
            out[i].attributes[j].normalized      = r.U8();
            out[i].attributes[j].unique_id       = r.VarU32();
        }
        for (uint32_t j = 0; j < na; ++j) {
            out[i].attributes[j].seq_att_dec_decoder_type = r.U8();
        }
    }
    return !r.error;
}

// ---- ConvertSymbolToSignedInt (sequential.integer.attribute.decoder.md, exact) ------------------
// is_positive = !(val & 1); val >>= 1; positive -> val ; else -val - 1.
inline int32_t ConvertSymbolToSignedInt(uint32_t val) {
    const bool is_positive = (val & 1u) == 0u;
    val >>= 1;
    if (is_positive) return static_cast<int32_t>(val);
    return -static_cast<int32_t>(val) - 1;
}

// ---- The position-attribute traversal + integer reconstruction (the DR4 core) -------------------
// All maps live in this transient struct, built ONLY for curr_att_dec=0 (the vertex POSITION). The
// corner-table data (face_to_vertex, opposite_corners_, vertex_corners_) comes from the DR3
// EdgebreakerDecoder, which we re-run with its state retained (see DecodeDracoMesh).
struct PositionDecodeState {
    const EdgebreakerDecoder* d = nullptr;
    uint32_t num_vertices = 0;   // num_encoded_vertices + num_encoded_split_symbols
    uint32_t num_faces = 0;

    // GenerateSequence outputs (attributes.decoder.md / edgebreaker.traversal.md).
    std::vector<int> encoded_value_index_to_corner_map;       // push order during the walk
    std::vector<int> vertex_to_encoded_attribute_value_index; // [num_vertices] -> encoded index
    std::vector<uint8_t> is_face_visited;
    std::vector<uint8_t> is_vertex_visited;
    int vertex_visited_point_ids = 0;
    std::vector<int> corner_traversal_stack;
    bool ok = true;

    // ---- corner helpers (corner.md, position attribute att_dec=0) ----
    bool FaceVisited(int face) const {
        if (face < 0) return true;
        if (static_cast<std::size_t>(face) >= is_face_visited.size()) return true;
        return is_face_visited[face] != 0u;
    }
    int PosOpp(int c) const { return d->PosOpposite(c); }
    int GetLeftCorner(int c) const {
        if (c < 0) return kInvalidCornerIndex;
        return PosOpp(EdgebreakerDecoder::Previous(c));
    }
    int GetRightCorner(int c) const {
        if (c < 0) return kInvalidCornerIndex;
        return PosOpp(EdgebreakerDecoder::Next(c));
    }
    int CornerToVert0(int c) const { return d->CornerToVert(0, c); }

    // IsOnPositionBoundary(vert) (boundary.decoder.md): a hole vertex (no representative corner) is a
    // boundary; otherwise (no decoder-0 attribute seams in the supported case) it is interior.
    bool IsOnPositionBoundary(int v) const {
        if (v < 0) return true;
        if (static_cast<std::size_t>(v) >= d->vertex_corners_.size()) return true;
        return d->vertex_corners_[v] < 0;
    }

    // OnNewVertexVisited (edgebreaker.traversal.md).
    void OnNewVertexVisited(int vert, int corner) {
        if (vert < 0 || static_cast<std::size_t>(vert) >= vertex_to_encoded_attribute_value_index.size()) {
            ok = false; return;
        }
        encoded_value_index_to_corner_map.push_back(corner);
        vertex_to_encoded_attribute_value_index[vert] = vertex_visited_point_ids;
        ++vertex_visited_point_ids;
    }

    // EdgeBreakerTraverser_ProcessCorner (edgebreaker.traversal.md, DEPTH_FIRST, exact).
    void ProcessCorner(int corner_id) {
        int face = corner_id / 3;
        if (FaceVisited(face)) return;
        corner_traversal_stack.push_back(corner_id);
        if (static_cast<std::size_t>(face) >= d->face_to_vertex[0].size()) { ok = false; return; }
        const int next_vert = d->face_to_vertex[1][face];
        const int prev_vert = d->face_to_vertex[2][face];
        if (next_vert >= 0 && static_cast<std::size_t>(next_vert) < is_vertex_visited.size()
            && !is_vertex_visited[next_vert]) {
            is_vertex_visited[next_vert] = 1u;
            OnNewVertexVisited(next_vert, EdgebreakerDecoder::Next(corner_id));
        }
        if (prev_vert >= 0 && static_cast<std::size_t>(prev_vert) < is_vertex_visited.size()
            && !is_vertex_visited[prev_vert]) {
            is_vertex_visited[prev_vert] = 1u;
            OnNewVertexVisited(prev_vert, EdgebreakerDecoder::Previous(corner_id));
        }
        int guard = 0;
        while (!corner_traversal_stack.empty()) {
            corner_id = corner_traversal_stack.back();
            int face_id = corner_id / 3;
            if (corner_id < 0 || FaceVisited(face_id)) {
                corner_traversal_stack.pop_back();
                continue;
            }
            while (true) {
                face_id = corner_id / 3;
                if (face_id < 0 || static_cast<std::size_t>(face_id) >= is_face_visited.size()) { ok = false; return; }
                is_face_visited[face_id] = 1u;
                const int vert_id = CornerToVert0(corner_id);
                if (vert_id >= 0 && static_cast<std::size_t>(vert_id) < is_vertex_visited.size()
                    && !is_vertex_visited[vert_id]) {
                    const bool on_boundary = IsOnPositionBoundary(vert_id);
                    is_vertex_visited[vert_id] = 1u;
                    OnNewVertexVisited(vert_id, corner_id);
                    if (!ok) return;
                    if (!on_boundary) {
                        corner_id = GetRightCorner(corner_id);
                        if (++guard > 100000000) { ok = false; return; }
                        continue;
                    }
                }
                const int right_corner_id = GetRightCorner(corner_id);
                const int left_corner_id  = GetLeftCorner(corner_id);
                const int right_face_id = right_corner_id < 0 ? -1 : right_corner_id / 3;
                const int left_face_id  = left_corner_id  < 0 ? -1 : left_corner_id  / 3;
                if (FaceVisited(right_face_id)) {
                    if (FaceVisited(left_face_id)) {
                        corner_traversal_stack.pop_back();
                        break;
                    } else {
                        corner_id = left_corner_id;
                    }
                } else {
                    if (FaceVisited(left_face_id)) {
                        corner_id = right_corner_id;
                    } else {
                        corner_traversal_stack.back() = left_corner_id;
                        corner_traversal_stack.push_back(right_corner_id);
                        break;
                    }
                }
                if (++guard > 100000000) { ok = false; return; }
            }
        }
    }

    // GenerateSequence (DEPTH_FIRST): EdgeBreakerTraverser_ProcessCorner over every 3*i.
    void GenerateSequence() {
        is_face_visited.assign(num_faces, 0u);
        is_vertex_visited.assign(num_vertices, 0u);
        vertex_to_encoded_attribute_value_index.assign(num_vertices, -1);
        encoded_value_index_to_corner_map.clear();
        vertex_visited_point_ids = 0;
        corner_traversal_stack.clear();
        for (uint32_t i = 0; i < num_faces; ++i) {
            ProcessCorner(3 * static_cast<int>(i));
            if (!ok) return;
        }
    }
};

// ---- The decoded mesh + the full decode entry ---------------------------------------------------
struct DecodedMesh {
    uint32_t num_faces = 0, num_points = 0;
    std::vector<float>    positions;   // num_points * 3 (the dequantized cube corners)
    std::vector<float>    normals;     // num_points * 3 (if present; best-effort)
    std::vector<uint32_t> indices;     // num_faces * 3 (from DR3 face_to_vertex, point-mapped)
    bool ok = false;
};

// Re-run the DR3 edgebreaker decode but RETAIN the full corner-table state (DR4 needs vertex_corners_,
// opposite_corners_, face_to_vertex). This duplicates DecodeEdgebreakerConnectivityData's body but does
// NOT modify it (append-only). On success the returned decoder's face_to_vertex[k] hold the topology.
inline bool DecodeEdgebreakerFull(ByteReader& r, EdgebreakerDecoder& d) {
    ParseEdgebreakerConnectivityData(r, d);
    if (!d.ok) return false;
    DecodeTopologySplitEvents(r, d);
    if (!d.ok) return false;
    EdgebreakerTraversalStart(r, d);
    if (!d.ok) return false;
    d.DecodeEdgeBreakerConnectivity();
    if (!d.ok) return false;
    return static_cast<uint32_t>(d.face_to_vertex[0].size()) == d.num_faces;
}

// DecodeDracoMesh: ParseHeader (DR1) -> DecodeEdgebreaker connectivity with retained state (DR3) ->
// DecodeAttributeData for the POSITION (DR4) -> assemble positions + indices. Returns ok=false (never
// crashes) on any malformed / unsupported input. Only MESH_EDGEBREAKER with a vertex POSITION attribute
// using PARALLELOGRAM + WRAP + QUANTIZATION is fully supported (the Box / common glTF case).
inline DecodedMesh DecodeDracoMesh(const uint8_t* bytes, std::size_t n) {
    DecodedMesh mesh;
    ByteReader r{ bytes, n, 0, false };
    const DracoHeader h = ParseHeader(r);
    if (!h.valid || h.encoderMethod != kMeshEbEncoding) return mesh;

    EdgebreakerDecoder d;
    if (!DecodeEdgebreakerFull(r, d)) return mesh;

    const uint32_t num_vertices = d.num_encoded_vertices + d.num_encoded_split_symbols;
    const uint32_t num_faces = static_cast<uint32_t>(d.face_to_vertex[0].size());

    // ---- ParseAttributeDecodersData ----
    std::vector<DracoAttributeDecoder> decoders;
    if (!ParseAttributeDecodersData(r, h.encoderMethod, decoders)) return mesh;
    if (decoders.empty() || decoders[0].attributes.empty()) return mesh;

    // We support a vertex POSITION decoder as decoder 0 (the Box layout). Anything else -> ok=false.
    const DracoAttributeDecoder& dec0 = decoders[0];
    const DracoAttribute& att0 = dec0.attributes[0];
    if (dec0.decoder_type != kMeshVertexAttribute) return mesh;
    if (att0.att_type != kAttPosition) return mesh;
    if (att0.num_components == 0) return mesh;
    if (dec0.traversal_method != kMeshTraversalDepthFirst) return mesh;
    if (att0.seq_att_dec_decoder_type != kSeqAttEncQuantization) return mesh;
    const uint32_t num_components = att0.num_components;

    // ---- GenerateSequence for the POSITION (curr_att_dec = 0) ----
    // (DecodeAttributeSeams / RecomputeVerticesInternal apply to non-position decoders only; for the
    //  vertex POSITION the per-vertex map below is the full mapping.)
    PositionDecodeState ps;
    ps.d = &d;
    ps.num_vertices = num_vertices;
    ps.num_faces = num_faces;
    ps.GenerateSequence();
    if (!ps.ok) return mesh;
    const uint32_t num_values = static_cast<uint32_t>(ps.encoded_value_index_to_corner_map.size());
    if (num_values != num_vertices) return mesh;  // a closed vertex attribute: one value per vertex

    // ---- DecodePortableAttributes -> SequentialIntegerAttributeDecoder for the POSITION ----
    // ParsePredictionData: prediction_scheme I8; if != NONE: transform_type I8, compressed UI8.
    const int8_t prediction_scheme = static_cast<int8_t>(r.U8());
    if (r.error) return mesh;
    if (prediction_scheme == kPredictionNone) return mesh;  // the supported Box path always predicts
    const int8_t transform_type = static_cast<int8_t>(r.U8());
    const uint8_t compressed = r.U8();
    if (r.error) return mesh;
    if (transform_type != kPredTransformWrap) return mesh;  // POSITION uses the WRAP transform

    // Decode the integer residual symbols (num_values * num_components) via the DR2 rANS.
    std::vector<uint32_t> decoded_symbols;
    const uint32_t total_values = num_values * num_components;
    if (compressed > 0) {
        if (!DecodeSymbols(r, total_values, num_components, decoded_symbols)) return mesh;
    }
    if (decoded_symbols.size() != total_values) return mesh;

    // ConvertSymbolsToSignedInts.
    std::vector<int32_t> signed_values(total_values, 0);
    for (uint32_t i = 0; i < total_values; ++i)
        signed_values[i] = ConvertSymbolToSignedInt(decoded_symbols[i]);

    // DecodePredictionData(MESH_PREDICTION_PARALLELOGRAM): only DecodeTransformData runs ->
    // ParseWrapTransformData: wrap_min I32, wrap_max I32 (read BEFORE ComputeOriginalValues).
    const int32_t wrap_min = static_cast<int32_t>(r.LE32());
    const int32_t wrap_max = static_cast<int32_t>(r.LE32());
    if (r.error) return mesh;
    const int32_t max_dif = 1 + wrap_max - wrap_min;
    if (max_dif <= 0) return mesh;

    // PredictionSchemeWrapDecodingTransform_ComputeOriginalValue (prediction.wrap.transform.md).
    auto wrap_transform = [&](const int32_t* pred, const int32_t* corr, int32_t* out) {
        for (uint32_t c = 0; c < num_components; ++c) {
            int32_t clamped = pred[c];
            if (clamped > wrap_max) clamped = wrap_max;
            else if (clamped < wrap_min) clamped = wrap_min;
            int32_t o = clamped + corr[c];
            if (o > wrap_max) o -= max_dif;
            else if (o < wrap_min) o += max_dif;
            out[c] = o;
        }
    };

    // MeshPredictionSchemeParallelogramDecoder_ComputeOriginalValues (parallelogram + wrap).
    std::vector<int32_t> original_values(total_values, 0);
    {
        std::vector<int32_t> zero_pred(num_components, 0);
        wrap_transform(zero_pred.data(), &signed_values[0], &original_values[0]);  // p = 0
        std::vector<int32_t> pred_vals(num_components, 0);
        for (uint32_t p = 1; p < num_values; ++p) {
            const int corner_id = ps.encoded_value_index_to_corner_map[p];
            const uint32_t dst = p * num_components;
            // ComputeParallelogramPrediction (prediction.parallelogram.decoder.md).
            bool used = false;
            const int oci = d.Opposite(0, corner_id);
            if (oci >= 0) {
                int v = -1, nx = -1, pv = -1;
                d.CornerToVerts(0, oci, v, nx, pv);
                if (v >= 0 && nx >= 0 && pv >= 0
                    && static_cast<uint32_t>(v) < num_vertices
                    && static_cast<uint32_t>(nx) < num_vertices
                    && static_cast<uint32_t>(pv) < num_vertices) {
                    const int vert_opp  = ps.vertex_to_encoded_attribute_value_index[v];
                    const int vert_next = ps.vertex_to_encoded_attribute_value_index[nx];
                    const int vert_prev = ps.vertex_to_encoded_attribute_value_index[pv];
                    if (vert_opp >= 0 && vert_next >= 0 && vert_prev >= 0
                        && static_cast<uint32_t>(vert_opp)  < p
                        && static_cast<uint32_t>(vert_next) < p
                        && static_cast<uint32_t>(vert_prev) < p) {
                        const uint32_t off_opp  = static_cast<uint32_t>(vert_opp)  * num_components;
                        const uint32_t off_next = static_cast<uint32_t>(vert_next) * num_components;
                        const uint32_t off_prev = static_cast<uint32_t>(vert_prev) * num_components;
                        for (uint32_t c = 0; c < num_components; ++c)
                            pred_vals[c] = (original_values[off_next + c] + original_values[off_prev + c])
                                         - original_values[off_opp + c];
                        used = true;
                    }
                }
            }
            if (!used) {
                const uint32_t src = (p - 1) * num_components;
                wrap_transform(&original_values[src], &signed_values[dst], &original_values[dst]);
            } else {
                wrap_transform(pred_vals.data(), &signed_values[dst], &original_values[dst]);
            }
        }
    }

    // ---- DecodeDataNeededByPortableTransforms -> ParseQuantizationData (the quantization attribute) --
    // num_components floats min_values, one float range, one byte quantization_bits.
    std::vector<float> min_values(num_components, 0.0f);
    for (uint32_t c = 0; c < num_components; ++c) {
        const uint32_t bits = r.LE32();
        if (r.error) return mesh;
        float f = 0.0f;
        // bit-cast the IEEE-754 little-endian word to float WITHOUT <cstring>/<bit> (a tiny union copy).
        union { uint32_t u; float f; } cvt; cvt.u = bits; f = cvt.f;
        min_values[c] = f;
    }
    const uint32_t range_bits = r.LE32();
    if (r.error) return mesh;
    float range = 0.0f; { union { uint32_t u; float f; } cvt; cvt.u = range_bits; range = cvt.f; }
    const uint8_t quantization_bits = r.U8();
    if (r.error) return mesh;
    if (quantization_bits == 0u || quantization_bits > 31u) return mesh;

    // ---- TransformAttributesToOriginalFormat -> DequantizeValues (the ONE float step) --------------
    const int32_t max_quantized_value = (1 << quantization_bits) - 1;
    if (max_quantized_value <= 0) return mesh;
    const float max_quantized_value_factor = 1.0f / static_cast<float>(max_quantized_value);

    std::vector<float> dequantized(total_values, 0.0f);
    for (uint32_t i = 0; i < num_values; ++i) {
        for (uint32_t c = 0; c < num_components; ++c) {
            int32_t q = original_values[i * num_components + c];
            const bool neg = q < 0;
            if (neg) q = -q;
            float norm_value = static_cast<float>(q) * max_quantized_value_factor;
            if (neg) norm_value = -norm_value;
            float value = norm_value * range + min_values[c];
            dequantized[i * num_components + c] = value;
        }
    }

    // ---- Assemble: positions indexed by encoded-attribute-value index; indices from face_to_vertex. --
    mesh.num_faces = num_faces;
    mesh.num_points = num_values;
    mesh.positions = dequantized;  // num_values * num_components (== 3 for a position)
    mesh.indices.assign(static_cast<std::size_t>(num_faces) * 3u, 0u);
    bool index_ok = true;
    for (uint32_t f = 0; f < num_faces; ++f) {
        for (int k = 0; k < 3; ++k) {
            const int vert = d.face_to_vertex[k][f];
            if (vert < 0 || static_cast<uint32_t>(vert) >= num_vertices) { index_ok = false; break; }
            const int enc = ps.vertex_to_encoded_attribute_value_index[vert];
            if (enc < 0 || static_cast<uint32_t>(enc) >= num_values) { index_ok = false; break; }
            mesh.indices[static_cast<std::size_t>(f) * 3u + static_cast<std::size_t>(k)] =
                static_cast<uint32_t>(enc);
        }
        if (!index_ok) break;
    }
    if (!index_ok) { mesh.ok = false; return mesh; }

    mesh.ok = true;
    return mesh;
}

}  // namespace hf::asset::draco
