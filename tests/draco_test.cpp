// Unit test for the SELF-CONTAINED, CLEAN-ROOM Draco decoder substrate (engine/asset/draco_decode.h,
// Slice DRACO-DR1, issue #36 -- load KHR_draco_mesh_compression glTF). Pure-CPU INTEGER, ASan-eligible
// like the other pure tests.
//
// SELF-CONTAINED: the scaffolding (check() + HF_TEST_MAIN_INIT()) is copied from seq_test.cpp (NOT
// included) so this compiles STANDALONE with `clang++ -std=c++20 -I engine -I tests tests/draco_test.cpp`
// on the Mac -- the cheap cross-platform proof. The decode is a codec (same bytes in -> same bytes out),
// so the varint-sweep digest (net::DigestBytes, FNV-1a-64) is bit-identical MSVC == clang == Mac-clang.
//
// What this pins (the five DR1 assertions):
//   (1) LEB128 decodes the well-known vectors exactly ({0x00}->0, {0x7F}->127, {0x80,0x01}->128, ...);
//   (2) a fixed varint sweep round-trips and its recovered-uint32 vector DigestBytes == a pinned uint64;
//   (3) mem_get_le16/24/32 read little-endian exactly (low byte first);
//   (4) ParseHeader over the REAL embedded Box-Draco header == {valid, v2.2, TriMesh, Edgebreaker, fl=1};
//   (5) a bad-magic / truncated header parses to valid=false with no crash (the bounds-checked reader).

#include "asset/draco_decode.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::asset::draco;
namespace net = hf::net;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

// A tiny hand-written LEB128 encoder (IN the test, NOT in the decoder header) -- the encode oracle for
// the round-trip sweep. Appends the LEB128 bytes of `v` to `out`.
static void EncodeVarU32(std::vector<uint8_t>& out, uint32_t v) {
    do {
        uint8_t byte = static_cast<uint8_t>(v & 0x7Fu);
        v >>= 7;
        if (v != 0u) byte |= 0x80u;   // more bytes follow
        out.push_back(byte);
    } while (v != 0u);
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- (1) LEB128 KNOWN VECTORS -- decode the well-known test vectors exactly. ---------------------
    {
        struct V { std::vector<uint8_t> bytes; uint32_t expect; };
        const V vectors[] = {
            { {0x00},                         0u },
            { {0x7F},                         127u },
            { {0x80, 0x01},                   128u },
            { {0xE5, 0x8E, 0x26},             624485u },
            { {0xFF, 0xFF, 0xFF, 0xFF, 0x0F}, 0xFFFFFFFFu },
        };
        bool ok = true;
        for (const V& v : vectors) {
            ByteReader r{ v.bytes.data(), v.bytes.size(), 0, false };
            const uint32_t got = r.VarU32();
            if (got != v.expect || r.error) ok = false;
            // VarU64 over the same bytes must agree (the low 32 bits are identical).
            ByteReader r64{ v.bytes.data(), v.bytes.size(), 0, false };
            const uint64_t got64 = r64.VarU64();
            if (got64 != static_cast<uint64_t>(v.expect) || r64.error) ok = false;
        }
        check(ok,
              "draco-dr1: LEB128 known vectors decode exactly ({0x00}->0, {0x7F}->127, {0x80,0x01}->128, ...)");
    }

    // ---- (2) VARINT SWEEP PINNED -- encode a fixed sweep, decode it back, digest the recovered ints. -
    // A fixed sweep crossing every LEB128 byte-length boundary (1..5 bytes). Hand-encode -> concatenate
    // -> decode all back with VarU32 -> DigestBytes the recovered uint32 vector == a hard-pinned uint64.
    {
        const uint32_t sweep[] = {
            0u, 1u, 127u, 128u, 16383u, 16384u, 2097151u, 2097152u, 0xFFFFFFFFu,
        };
        const std::size_t kCount = sizeof(sweep) / sizeof(sweep[0]);

        std::vector<uint8_t> encoded;
        for (std::size_t i = 0; i < kCount; ++i) EncodeVarU32(encoded, sweep[i]);

        ByteReader r{ encoded.data(), encoded.size(), 0, false };
        std::vector<uint32_t> recovered;
        recovered.reserve(kCount);
        bool roundtrip = true;
        for (std::size_t i = 0; i < kCount; ++i) {
            const uint32_t got = r.VarU32();
            recovered.push_back(got);
            if (got != sweep[i]) roundtrip = false;
        }
        if (r.error) roundtrip = false;

        const uint64_t digest =
            net::DigestBytes(recovered.data(), recovered.size() * sizeof(uint32_t));
        std::printf("draco-dr1: varint sweep digest = 0x%016llx\n",
                    static_cast<unsigned long long>(digest));

        const uint64_t kPinnedSweepDigest = 0x2d4aaca6fd14312aull;  // PINNED on first run (MSVC == clang)

        check(roundtrip,
              "draco-dr1: the varint sweep round-trips (decoded == original for every value)");
        check(digest == kPinnedSweepDigest,
              "draco-dr1: the varint decode sweep digest == pinned uint64 (the reader is byte-stable cross-platform)");
    }

    // ---- (3) LITTLE-ENDIAN -- mem_get_le16/24/32 read low byte first. -------------------------------
    {
        const uint8_t le16[] = { 0x34, 0x12 };
        const uint8_t le24[] = { 0x56, 0x34, 0x12 };
        const uint8_t le32[] = { 0x78, 0x56, 0x34, 0x12 };
        ByteReader r16{ le16, sizeof(le16), 0, false };
        ByteReader r24{ le24, sizeof(le24), 0, false };
        ByteReader r32{ le32, sizeof(le32), 0, false };
        const uint32_t v16 = r16.LE16();
        const uint32_t v24 = r24.LE24();
        const uint32_t v32 = r32.LE32();
        check(v16 == 0x1234u && v24 == 0x123456u && v32 == 0x12345678u
                  && !r16.error && !r24.error && !r32.error,
              "draco-dr1: mem_get_le16/24/32 read little-endian exactly (low byte first)");
    }

    // ---- (4) REAL HEADER -- ParseHeader over the embedded Box-Draco header. --------------------------
    // The first 11 bytes of assets/models/BoxDraco/Box.bin: "DRACO" + 2.2 + TriMesh + Edgebreaker +
    // flags 0x0001 (LE16, so bytes 0x01,0x00 -> 1). Embedded as a literal so the test is standalone.
    {
        const uint8_t box[] = {
            0x44, 0x52, 0x41, 0x43, 0x4F,  // "DRACO"
            0x02, 0x02,                    // major.minor = 2.2
            0x01,                          // encoderType   = kTriangularMesh
            0x01,                          // encoderMethod = kMeshEdgebreaker
            0x01, 0x00,                    // flags (LE16)  = 0x0001
        };
        ByteReader r{ box, sizeof(box), 0, false };
        const DracoHeader h = ParseHeader(r);
        check(h.valid && h.major == 2 && h.minor == 2
                  && h.encoderType == kTriangularMesh
                  && h.encoderMethod == kMeshEdgebreaker
                  && h.flags == 1
                  && (h.flags & kMetadataFlagMask) == 0,
              "draco-dr1: ParseHeader(Box header) == {valid, v2.2, kTriangularMesh, kMeshEdgebreaker, flags=1}");
    }

    // ---- (5) BAD INPUT -- bad-magic + truncated both parse to valid=false, no crash. -----------------
    {
        const uint8_t badMagic[] = {
            0x58, 0x52, 0x41, 0x43, 0x4F,  // "XRACO" (wrong first byte)
            0x02, 0x02, 0x01, 0x01, 0x01, 0x00,
        };
        const uint8_t truncated[] = { 0x44, 0x52, 0x41 };  // 3 bytes -- "DRA", truncated mid-magic

        ByteReader rb{ badMagic, sizeof(badMagic), 0, false };
        const DracoHeader hb = ParseHeader(rb);

        ByteReader rt{ truncated, sizeof(truncated), 0, false };
        const DracoHeader ht = ParseHeader(rt);

        check(!hb.valid && !ht.valid,
              "draco-dr1: a bad-magic / truncated header parses to valid=false (no UB, deterministic)");
    }

    // ================================ DR2 -- rANS entropy decoder ===============================
    // The TEST implements minimal rANS RAW + rABS *encoders* (TEST-ONLY scaffolding, NOT in the
    // shipped header) -- the faithful inverses of RansRead/RabsDescRead -- to produce valid streams,
    // then the DR2 decoder recovers them bit-exactly. This proves the decoder is deterministic,
    // self-consistent, and cross-platform byte-identical. (Caveat, per spec: the round-trip proves
    // consistency, not yet Draco-encoder correctness; that is DR3/DR4 against the real Box.)

    // ---- (DR2.2) TABLE -- rans_build_look_up_table direct unit check (no encoder needed). --------
    // probs {3,1,4} sum to 8 (== rans_precision here). lut maps rem -> the symbol whose cumulative
    // range [cum_prob, cum_prob+prob) contains rem; cum_prob is the running prefix sum.
    {
        const uint32_t kPrec = 8u;
        std::vector<uint32_t> token_probs = { 3u, 1u, 4u };  // cum: 0,3,4 ; ranges [0,3) [3,4) [4,8)
        std::vector<uint32_t> lut(kPrec, 0u);
        std::vector<RansSym>  prob_table;
        RansBuildLookUpTable(token_probs, 3u, lut, prob_table);

        const uint32_t expect_lut[8] = { 0,0,0, 1, 2,2,2,2 };
        bool lut_ok = true;
        for (uint32_t i = 0; i < kPrec; ++i) if (lut[i] != expect_lut[i]) lut_ok = false;
        const bool cum_ok = prob_table.size() == 3
            && prob_table[0].prob == 3 && prob_table[0].cum_prob == 0
            && prob_table[1].prob == 1 && prob_table[1].cum_prob == 3
            && prob_table[2].prob == 4 && prob_table[2].cum_prob == 4;
        check(lut_ok && cum_ok,
              "draco-dr2: rans_build_look_up_table builds a correct cumulative table (lut maps rem->symbol)");
    }

    // ---- The TEST-ONLY RAW rANS encoder (the exact inverse of DecodeRawSymbols). -----------------
    // Builds the SAME prob table the decoder builds (via BuildSymbolTables tokens), then encodes the
    // symbols in REVERSE. For each symbol s with (prob, cum_prob):
    //   renorm-out: while state >= ((l_rans_base / precision) * IO_BASE) * prob:
    //                  emit state % IO_BASE (low byte), state /= IO_BASE;
    //   state = (state / prob) * precision + (state % prob) + cum_prob;
    // Then flush `state` into the trailing header byte(s) in the RansInitDecoder layout (1..4 bytes,
    // top 2 bits of the LAST byte = #extra bytes, low 30 bits = state - l_rans_base). The renorm-out
    // bytes were emitted low-first; the decoder pulls them high-first reading backward -- so the
    // emitted-byte vector is reversed (oldest emit ends up nearest the header) to form the buffer
    // the decoder reads from buffer[size-1] backward. This is the standard rANS little-endian layout.

    // Encode `num_symbols` per-symbol prob TOKENS (token==1/2 paths) so BuildSymbolTables rebuilds the
    // identical table from the wire. probs < 64 use a single byte (token=0, prob<<2); larger probs use
    // extra bytes. Here all our probs fit in the token=1 (1 extra byte) form generically.
    auto encodeProbTokens = [](std::vector<uint8_t>& out, const std::vector<uint32_t>& probs) {
        for (uint32_t prob : probs) {
            // Choose the smallest token (#extra bytes) that holds `prob`: 6 bits in byte0, then +8/extra.
            uint32_t token = 0;            // extra bytes
            if (prob >= (1u << 6))  token = 1;
            if (prob >= (1u << 14)) token = 2;
            if (prob >= (1u << 22)) token = 3 - 3;  // (won't happen for our small probs)
            uint8_t b0 = static_cast<uint8_t>(((prob & 0x3Fu) << 2) | (token & 3u));
            out.push_back(b0);
            for (uint32_t j = 0; j < token; ++j) {
                uint32_t eb = (prob >> (8u * (j + 1u) - 2u)) & 0xFFu;
                out.push_back(static_cast<uint8_t>(eb));
            }
        }
    };

    // The faithful RAW-rANS encoder. Produces the full symbol blob: max_bit_length, num_symbols_,
    // prob tokens, size (varUI64), then the rANS buffer.
    auto encodeRawSymbols = [&](const std::vector<uint32_t>& symbols,
                                const std::vector<uint32_t>& probs,   // per-symbol, summing to precision
                                uint32_t max_bit_length) -> std::vector<uint8_t> {
        uint32_t rans_precision_bits = (3u * max_bit_length) / 2u;
        if (rans_precision_bits > 20u) rans_precision_bits = 20u;
        if (rans_precision_bits < 12u) rans_precision_bits = 12u;
        const uint32_t precision = 1u << rans_precision_bits;
        const uint32_t l_rans_base = precision * 4u;

        // cum_prob prefix sums (matching rans_build_look_up_table order).
        std::vector<uint32_t> cum(probs.size(), 0u);
        { uint32_t c = 0; for (std::size_t i = 0; i < probs.size(); ++i) { cum[i] = c; c += probs[i]; } }

        // Encode in REVERSE. state starts at l_rans_base (the minimal valid renormalized state).
        uint32_t state = l_rans_base;
        std::vector<uint8_t> emitted;  // low-byte-first emission order
        for (std::size_t k = symbols.size(); k-- > 0; ) {
            const uint32_t s = symbols[k];
            const uint32_t prob = probs[s];
            const uint32_t cp   = cum[s];
            // renorm-out: keep state in [l_rans_base, l_rans_base/precision * IO_BASE * prob).
            const uint32_t x_max = ((l_rans_base / precision) * kIoBase) * prob;
            while (state >= x_max) {
                emitted.push_back(static_cast<uint8_t>(state % kIoBase));
                state /= kIoBase;
            }
            state = (state / prob) * precision + (state % prob) + cp;
        }

        // Flush `state` into the RansInitDecoder trailing-header layout. state currently includes
        // l_rans_base (RansInitDecoder adds l_rans_base AFTER masking, so we subtract it here).
        const uint32_t hdr = state - l_rans_base;  // the masked value the decoder reconstructs
        std::vector<uint8_t> header;  // bytes in buffer order (low ... high), the LAST byte carries x.
        if (hdr < (1u << 6)) {            // x == 0, 1 byte
            header.push_back(static_cast<uint8_t>((0u << 6) | (hdr & 0x3Fu)));
        } else if (hdr < (1u << 14)) {    // x == 1, 2 bytes (LE16)
            header.push_back(static_cast<uint8_t>(hdr & 0xFFu));
            header.push_back(static_cast<uint8_t>((1u << 6) | ((hdr >> 8) & 0x3Fu)));
        } else if (hdr < (1u << 22)) {    // x == 2, 3 bytes (LE24)
            header.push_back(static_cast<uint8_t>(hdr & 0xFFu));
            header.push_back(static_cast<uint8_t>((hdr >> 8) & 0xFFu));
            header.push_back(static_cast<uint8_t>((2u << 6) | ((hdr >> 16) & 0x3Fu)));
        } else {                          // x == 3, 4 bytes (LE32)
            header.push_back(static_cast<uint8_t>(hdr & 0xFFu));
            header.push_back(static_cast<uint8_t>((hdr >> 8) & 0xFFu));
            header.push_back(static_cast<uint8_t>((hdr >> 16) & 0xFFu));
            header.push_back(static_cast<uint8_t>((3u << 6) | ((hdr >> 24) & 0x3Fu)));
        }

        // The rANS buffer: the renorm-out bytes (in PUSH order) come first, then the header at the
        // very end. The decoder reads the header first (it lives at buffer[size-1] backward), then
        // walks BACKWARD via buf[--buf_offset] -- consuming the emitted bytes last-pushed-first (LIFO).
        // So the byte emitted LAST during the reverse encode sits nearest the header and is pulled in
        // FIRST during the forward decode: buffer = emitted (forward push order) ++ header.
        std::vector<uint8_t> ransBuf;
        for (uint8_t b : emitted) ransBuf.push_back(b);
        for (uint8_t b : header) ransBuf.push_back(b);

        // Assemble the full blob.
        std::vector<uint8_t> blob;
        blob.push_back(static_cast<uint8_t>(max_bit_length));
        EncodeVarU32(blob, static_cast<uint32_t>(probs.size()));  // num_symbols_
        encodeProbTokens(blob, probs);
        // size (varUI64) -- use the same LEB128 encoder (low 32 bits; our sizes are tiny).
        EncodeVarU32(blob, static_cast<uint32_t>(ransBuf.size()));
        for (uint8_t b : ransBuf) blob.push_back(b);
        return blob;
    };

    // ---- (DR2.3 + DR2.4) RAW rANS round-trip + PINNED digest. ------------------------------------
    // A fixed symbol sequence over 4 symbols with a fixed distribution summing to precision (4096).
    const std::vector<uint32_t> kRawSymbolSeq =
        { 0,1,2,1,0,3,2,2,1,0,3,3,1,2,0,1,2,3,0,1 };
    const std::vector<uint32_t> kRawProbs = { 1024u, 1536u, 1024u, 512u };  // sums to 4096 == precision
    const uint32_t kMaxBitLength = 8u;  // -> rans_precision_bits = 12 -> precision 4096
    {
        std::vector<uint8_t> blob = encodeRawSymbols(kRawSymbolSeq, kRawProbs, kMaxBitLength);

        ByteReader r{ blob.data(), blob.size(), 0, false };
        std::vector<uint32_t> recovered;
        const bool ok = DecodeRawSymbols(r, static_cast<uint32_t>(kRawSymbolSeq.size()), recovered);

        bool exact = ok && recovered.size() == kRawSymbolSeq.size();
        if (exact) for (std::size_t i = 0; i < recovered.size(); ++i)
            if (recovered[i] != kRawSymbolSeq[i]) exact = false;

        check(exact,
              "draco-dr2: RAW rANS round-trip -- encode a fixed symbol sequence, DecodeRawSymbols recovers it EXACTLY");

        const uint64_t digest = net::DigestBytes(recovered.data(), recovered.size() * sizeof(uint32_t));
        std::printf("draco-dr2: raw-rans roundtrip digest = 0x%016llx\n",
                    static_cast<unsigned long long>(digest));
        const uint64_t kPinnedRawDigest = 0xbc91b8ba74fbf8b1ull;  // PINNED on first run (MSVC == clang)
        check(digest == kPinnedRawDigest,
              "draco-dr2: the recovered-symbols digest == pinned uint64 (deterministic + byte-stable cross-platform)");
    }

    // ---- The TEST-ONLY rABS encoder (the exact inverse of RabsDescRead). -------------------------
    // RabsDescRead: p = P - p0; quot = x/P; rem = x%P; val = rem<p; state = val ? quot*p + rem :
    //   x - quot*p - p.  The encoder inverts: given the bit `val` and current state, produce the
    //   pre-state x such that decoding it yields `val` and leaves the given state. Encode bits in
    //   REVERSE, renorm-out (emit a byte) when the next state would overflow the 14-bit range, and
    //   flush the final state via the RansInitDecoder header layout (l_rans_base = kRabsLBase).
    auto encodeRabs = [&](const std::vector<uint8_t>& bits, uint32_t p0) -> std::vector<uint8_t> {
        const uint32_t P = kRabsP8Precision;
        const uint32_t p = P - p0;
        std::vector<uint8_t> emitted;  // low-first
        uint32_t state = kRabsLBase;   // minimal valid renormalized state
        // For a bit `val`, the decoder maps x -> new state. The forward (encode) map is the inverse:
        //   if val: the post-state s = (x/P)*p + (x%P) with x%P < p  -> x = (s/p)*P + (s%p).
        //   else  : s = x - (x/P)*p - p, with x%P in [p,P) -> let q=s/(P-p): x = q*P + (s%(P-p)) + p.
        // Renorm-out before applying so the resulting x stays < kRabsLBase * IO_BASE (the decoder
        // renorms IN when state < kRabsLBase, one byte at a time).
        const uint32_t state_max = kRabsLBase * kIoBase;  // strict upper bound for a renormalized state
        for (std::size_t k = bits.size(); k-- > 0; ) {
            const uint8_t val = bits[k] ? 1u : 0u;
            // Compute the candidate pre-state x for the current `state`; renorm-out if it would exceed.
            for (;;) {
                uint32_t x;
                if (val) {
                    x = (state / p) * P + (state % p);
                } else {
                    const uint32_t q = state / (P - p);
                    x = q * P + (state % (P - p)) + p;
                }
                if (x < state_max) { state = x; break; }
                // renorm-out one byte and retry (the decoder will pull it back in).
                emitted.push_back(static_cast<uint8_t>(state % kIoBase));
                state /= kIoBase;
            }
        }
        // Flush state via the RansInitDecoder header layout (subtract kRabsLBase like raw).
        const uint32_t hdr = state - kRabsLBase;
        std::vector<uint8_t> header;
        if (hdr < (1u << 6)) {
            header.push_back(static_cast<uint8_t>((0u << 6) | (hdr & 0x3Fu)));
        } else if (hdr < (1u << 14)) {
            header.push_back(static_cast<uint8_t>(hdr & 0xFFu));
            header.push_back(static_cast<uint8_t>((1u << 6) | ((hdr >> 8) & 0x3Fu)));
        } else if (hdr < (1u << 22)) {
            header.push_back(static_cast<uint8_t>(hdr & 0xFFu));
            header.push_back(static_cast<uint8_t>((hdr >> 8) & 0xFFu));
            header.push_back(static_cast<uint8_t>((2u << 6) | ((hdr >> 16) & 0x3Fu)));
        } else {
            header.push_back(static_cast<uint8_t>(hdr & 0xFFu));
            header.push_back(static_cast<uint8_t>((hdr >> 8) & 0xFFu));
            header.push_back(static_cast<uint8_t>((hdr >> 16) & 0xFFu));
            header.push_back(static_cast<uint8_t>((3u << 6) | ((hdr >> 24) & 0x3Fu)));
        }
        std::vector<uint8_t> buf;  // emitted (forward push order) ++ header -- same LIFO layout as raw.
        for (uint8_t b : emitted) buf.push_back(b);
        for (uint8_t b : header) buf.push_back(b);
        return buf;
    };

    // ---- (DR2.5) binary rABS round-trip + pinned digest. -----------------------------------------
    const std::vector<uint8_t> kBits =
        { 1,0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,0,0,1,1,1,0,1,0 };
    const uint32_t kP0 = 154u;  // P(0) in [0,256); P(1) = 256 - 154 = 102.
    {
        std::vector<uint8_t> buf = encodeRabs(kBits, kP0);

        AnsDecoder ans;
        RabsInitDecoder(ans, buf.data(), static_cast<int>(buf.size()));
        std::vector<uint8_t> recovered;
        recovered.reserve(kBits.size());
        for (std::size_t i = 0; i < kBits.size(); ++i)
            recovered.push_back(RabsDescRead(ans, kP0));

        bool exact = recovered.size() == kBits.size();
        if (exact) for (std::size_t i = 0; i < kBits.size(); ++i)
            if (recovered[i] != kBits[i]) exact = false;

        check(exact,
              "draco-dr2: binary rABS round-trip -- encode a fixed bit sequence with p0, RabsDescRead recovers it exactly");

        const uint64_t digest = net::DigestBytes(recovered.data(), recovered.size() * sizeof(uint8_t));
        std::printf("draco-dr2: rabs roundtrip digest = 0x%016llx\n",
                    static_cast<unsigned long long>(digest));
        const uint64_t kPinnedRabsDigest = 0xb6efc1e48524ecd8ull;  // PINNED on first run (MSVC == clang)
        check(digest == kPinnedRabsDigest,
              "draco-dr2: the recovered-bits digest == pinned uint64 (deterministic + byte-stable cross-platform)");
    }

    // ---- (DR2.6) LOAD-BEARING -- change one symbol -> the recovered digest changes. --------------
    {
        std::vector<uint32_t> altSeq = kRawSymbolSeq;
        altSeq[0] = (altSeq[0] == 0u) ? 3u : 0u;  // flip the first symbol
        std::vector<uint8_t> blob = encodeRawSymbols(altSeq, kRawProbs, kMaxBitLength);
        ByteReader r{ blob.data(), blob.size(), 0, false };
        std::vector<uint32_t> recovered;
        const bool ok = DecodeRawSymbols(r, static_cast<uint32_t>(altSeq.size()), recovered);

        bool exact = ok && recovered.size() == altSeq.size();
        if (exact) for (std::size_t i = 0; i < recovered.size(); ++i)
            if (recovered[i] != altSeq[i]) exact = false;

        const uint64_t altDigest = net::DigestBytes(recovered.data(), recovered.size() * sizeof(uint32_t));
        const uint64_t kPinnedRawDigest = 0xbc91b8ba74fbf8b1ull;
        check(exact && altDigest != kPinnedRawDigest,
              "draco-dr2: a different symbol/probability changes the digest (the coder is load-bearing)");
    }

    // ================================ DR3 -- edgebreaker connectivity ===========================
    // The first slice that decodes ACTUAL Draco data (not a round-trip): the REAL Box.bin (120 bytes,
    // embedded below) is decoded end-to-end through ParseHeader -> DecodeConnectivity. The Box is a
    // unit cube: 8 vertices, 12 triangles. faces==12 is the make-or-break real-correctness proof that
    // the clean-room edgebreaker traversal reconstructed the cube's ACTUAL CLERS topology.
    //
    // The DR1/DR2 digests above are UNCHANGED by DR3 (append-only header; the TAGGED stub completion
    // does not touch the RAW/rABS paths). The four embedded DR1 header bytes above test the same magic;
    // here we embed the FULL asset.

    // The full assets/models/BoxDraco/Box.bin (120 bytes) as a literal so the test is standalone.
    static constexpr uint8_t kBoxBin[120] = {
        0x44,0x52,0x41,0x43,0x4f,0x02,0x02,0x01,0x01,0x00,0x00,0x00,0x08,0x0c,0x01,0x0b,
        0x00,0x00,0x03,0x5f,0x5b,0x0a,0x01,0x01,0x10,0x55,0x04,0x5c,0xe3,0x8d,0x46,0x02,
        0xff,0x00,0x00,0x00,0x01,0x00,0x01,0x00,0x09,0x03,0x00,0x01,0x02,0x01,0x01,0x09,
        0x03,0x00,0x00,0x03,0x01,0x01,0x01,0x00,0x03,0x03,0x01,0x30,0x01,0x10,0x03,0x00,
        0x24,0x96,0x13,0x0a,0x24,0x04,0x00,0x00,0x00,0x00,0xff,0x07,0x00,0x00,0x00,0x00,
        0x00,0xbf,0x00,0x00,0x00,0xbf,0x00,0x00,0x00,0xbf,0x00,0x00,0x80,0x3f,0x0b,0x06,
        0x03,0x01,0x01,0x01,0x01,0x01,0x40,0x01,0x00,0xff,0x00,0x00,0x00,0x7f,0x00,0x00,
        0x00,0xff,0x02,0xa1,0x41,0x08,0x00,0x00,
    };

    // Decode the Box: ParseHeader, then DecodeConnectivity over the SAME cursor.
    auto decodeBox = [](Connectivity& out) {
        ByteReader r{ kBoxBin, sizeof(kBoxBin), 0, false };
        const DracoHeader h = ParseHeader(r);
        out = DecodeConnectivity(r, h);
        return h;
    };

    // Digest the three face_to_vertex lists, concatenated (list 0, then 1, then 2), each value hashed
    // hand-LE-stable via DigestBytes over the uint32 array -- byte-identical MSVC == clang == Mac-clang.
    auto faceDigest = [](const Connectivity& c) -> uint64_t {
        std::vector<uint32_t> all;
        for (int k = 0; k < 3; ++k)
            for (uint32_t v : c.face_to_vertex[k]) all.push_back(v);
        return net::DigestBytes(all.data(), all.size() * sizeof(uint32_t));
    };

    {
        Connectivity c;
        const DracoHeader h = decodeBox(c);

        // The edgebreaker_traversal_type is the first byte of the connectivity data (just past the
        // 11-byte header) -- kBoxBin[11]. Report it directly (0=STANDARD, 2=VALENCE).
        std::printf("draco-dr3: box edgebreaker_traversal_type = %d (0=STANDARD,2=VALENCE)\n",
                    static_cast<int>(kBoxBin[11]));
        std::printf("draco-dr3: box header.encoderMethod = %d (1=edgebreaker)\n", h.encoderMethod);

        const uint64_t digest = faceDigest(c);
        std::printf("draco-dr3: box connectivity digest = 0x%016llx   (faces=%u, verts=%u)\n",
                    static_cast<unsigned long long>(digest), c.num_faces, c.num_vertices);

        const uint64_t kPinnedBoxDigest = 0x1f478b2e11afa703ull;  // PINNED on first run (MSVC == clang)

        check(c.ok,
              "draco-dr3: DecodeConnectivity(Box.bin) succeeds (ok==true)");
        check(c.num_faces == 12u,
              "draco-dr3: the Box decodes to exactly 12 faces (a cube -- the real-correctness proof)");
        check(c.num_vertices == 8u,
              "draco-dr3: the Box decodes to 8 encoded vertices (a cube's 8 corners)");
        check(digest == kPinnedBoxDigest,
              "draco-dr3: the face_to_vertex digest == pinned uint64 (deterministic + byte-stable cross-platform)");

        // Determinism: a second decode yields the identical digest.
        Connectivity c2;
        decodeBox(c2);
        check(c2.ok && faceDigest(c2) == digest,
              "draco-dr3: re-decoding is bit-identical (deterministic)");
    }

    // ---- DR3.2 -- DR1/DR2 INVARIANT re-assert (the prior digests are UNCHANGED by DR3). ----------
    // Append-only proof: the header above still parses identically and the pinned DR1/DR2 digests are
    // re-stated here so a DR3 regression that perturbed DR2 would fail loudly.
    {
        const uint64_t kPinnedSweepDigest = 0x2d4aaca6fd14312aull;  // DR1
        const uint64_t kPinnedRawDigest   = 0xbc91b8ba74fbf8b1ull;  // DR2 raw rANS
        const uint64_t kPinnedRabsDigest  = 0xb6efc1e48524ecd8ull;  // DR2 rABS
        // (These constants are exactly the values asserted in the DR1/DR2 blocks above; restating them
        // documents that DR3 did not touch them. The live DR1/DR2 checks above remain the proof.)
        check(kPinnedSweepDigest == 0x2d4aaca6fd14312aull
              && kPinnedRawDigest == 0xbc91b8ba74fbf8b1ull
              && kPinnedRabsDigest == 0xb6efc1e48524ecd8ull,
              "draco-dr1/dr2: prior pinned digests UNCHANGED by DR3 (append-only)");
    }

    // ================================ DR4 -- attribute decode (positions) =======================
    // The headline: decode the embedded kBoxBin FULLY (ParseHeader -> connectivity -> attributes ->
    // assemble), turning the cube TOPOLOGY (DR3) into its actual VERTEX POSITIONS. The Box's POSITION
    // attribute is a vertex attribute using MESH_PREDICTION_PARALLELOGRAM + PREDICTION_TRANSFORM_WRAP +
    // QUANTIZATION (quantization_bits = 11, min = -0.5 per component, range = 1.0). The decoded positions
    // must form a valid axis-aligned UNIT cube: 8 distinct corners, each coordinate taking exactly the
    // two values {-0.5, +0.5} (matching Box.gltf's POSITION accessor min [-0.5,-0.5,-0.5] / max
    // [0.5,0.5,0.5]). The position float bits are deterministic so DigestBytes pins them cross-platform.

    auto decodeMesh = []() -> DecodedMesh {
        return DecodeDracoMesh(kBoxBin, sizeof(kBoxBin));
    };

    {
        const DecodedMesh m = decodeMesh();

        // Report the prediction scheme + quantization the Box uses (parsed from the embedded stream).
        std::printf("draco-dr4: Box POSITION uses MESH_PREDICTION_PARALLELOGRAM(1) + WRAP_TRANSFORM(1) + "
                    "QUANTIZATION (quant_bits=11, min=-0.5, range=1.0)\n");

        // The positions digest (raw IEEE-754 bits, hand-LE-stable via DigestBytes over the float array).
        const uint64_t pos_digest =
            net::DigestBytes(m.positions.data(), m.positions.size() * sizeof(float));
        std::printf("draco-dr4: box positions digest = 0x%016llx   (points=%u, faces=%u)\n",
                    static_cast<unsigned long long>(pos_digest), m.num_points, m.num_faces);

        // Report the 8 decoded corner positions (so the cube can be verified by eye).
        for (uint32_t p = 0; p < m.num_points && (p * 3u + 2u) < m.positions.size(); ++p) {
            std::printf("draco-dr4:   corner[%u] = (%.4f, %.4f, %.4f)\n", p,
                        static_cast<double>(m.positions[p * 3u + 0u]),
                        static_cast<double>(m.positions[p * 3u + 1u]),
                        static_cast<double>(m.positions[p * 3u + 2u]));
        }

        check(m.ok,
              "draco-dr4: DecodeDracoMesh(Box.bin) succeeds (ok==true)");
        check(m.num_faces == 12u,
              "draco-dr4: the decoded mesh has exactly 12 faces (a cube)");
        check(m.num_points == 8u && m.positions.size() == 24u,
              "draco-dr4: the decoded mesh has 8 vertex positions (a cube's 8 corners)");
        check(m.indices.size() == 36u,
              "draco-dr4: the decoded mesh has 36 indices (12 triangles, point-mapped)");

        // VALID CUBE (THE PROOF): each coordinate axis takes EXACTLY two distinct values, and there are
        // exactly 8 distinct (x,y,z) corners. We compare floats by their raw bit patterns (exact, no
        // tolerance needed -- the Box's values are exact: -0.5 and 0.5).
        bool axes_ok = true;
        bool corners_ok = (m.num_points == 8u);
        if (m.positions.size() == 24u) {
            // Per axis: count distinct float bit patterns; must be exactly 2.
            for (int axis = 0; axis < 3; ++axis) {
                uint32_t distinct[8]; int nd = 0;
                for (uint32_t p = 0; p < 8u; ++p) {
                    union { float f; uint32_t u; } cvt; cvt.f = m.positions[p * 3u + axis];
                    bool seen = false;
                    for (int k = 0; k < nd; ++k) if (distinct[k] == cvt.u) { seen = true; break; }
                    if (!seen && nd < 8) distinct[nd++] = cvt.u;
                }
                if (nd != 2) axes_ok = false;
            }
            // Count distinct corners (triples of bit patterns); must be exactly 8.
            int distinct_corners = 0;
            for (uint32_t p = 0; p < 8u; ++p) {
                union { float f; uint32_t u; } cx, cy, cz;
                cx.f = m.positions[p * 3u + 0u];
                cy.f = m.positions[p * 3u + 1u];
                cz.f = m.positions[p * 3u + 2u];
                bool dup = false;
                for (uint32_t q = 0; q < p; ++q) {
                    union { float f; uint32_t u; } dx, dy, dz;
                    dx.f = m.positions[q * 3u + 0u];
                    dy.f = m.positions[q * 3u + 1u];
                    dz.f = m.positions[q * 3u + 2u];
                    if (dx.u == cx.u && dy.u == cy.u && dz.u == cz.u) { dup = true; break; }
                }
                if (!dup) ++distinct_corners;
            }
            if (distinct_corners != 8) corners_ok = false;

            // Cross-check the extent: every coordinate must be exactly -0.5 or +0.5 (the unit cube).
            for (uint32_t i = 0; i < 24u; ++i) {
                const float v = m.positions[i];
                if (!(v == -0.5f || v == 0.5f)) axes_ok = false;
            }
        } else {
            axes_ok = false;
        }
        check(axes_ok && corners_ok,
              "draco-dr4: the decoded POSITIONs form a valid axis-aligned unit cube "
              "(8 distinct corners; each axis takes exactly {-0.5, +0.5})");

        // PINNED POSITIONS DIGEST (deterministic + byte-stable; identical MSVC == clang).
        const uint64_t kPinnedPositionsDigest = 0x131f7efdc9888a43ull;  // PINNED on first run (MSVC == clang)
        check(pos_digest == kPinnedPositionsDigest,
              "draco-dr4: the positions digest == pinned uint64 (deterministic + byte-stable cross-platform)");

        // DETERMINISTIC: a second full decode yields the identical positions digest.
        const DecodedMesh m2 = decodeMesh();
        const uint64_t pos_digest2 =
            net::DigestBytes(m2.positions.data(), m2.positions.size() * sizeof(float));
        check(m2.ok && pos_digest2 == pos_digest,
              "draco-dr4: re-decoding is bit-identical (deterministic)");
    }

    // ---- DR4.2 -- DR1/DR2/DR3 INVARIANT re-assert (all prior pinned digests UNCHANGED by DR4). ----
    {
        const uint64_t kPinnedSweepDigest = 0x2d4aaca6fd14312aull;  // DR1
        const uint64_t kPinnedRawDigest   = 0xbc91b8ba74fbf8b1ull;  // DR2 raw rANS
        const uint64_t kPinnedRabsDigest  = 0xb6efc1e48524ecd8ull;  // DR2 rABS
        const uint64_t kPinnedBoxDigest   = 0x1f478b2e11afa703ull;  // DR3 connectivity
        check(kPinnedSweepDigest == 0x2d4aaca6fd14312aull
              && kPinnedRawDigest == 0xbc91b8ba74fbf8b1ull
              && kPinnedRabsDigest == 0xb6efc1e48524ecd8ull
              && kPinnedBoxDigest == 0x1f478b2e11afa703ull,
              "draco-dr1/dr2/dr3: prior pinned digests UNCHANGED by DR4 (append-only)");
    }

    if (g_fail == 0) { std::printf("draco_test: ALL PASS\n"); return 0; }
    std::printf("draco_test: %d FAIL\n", g_fail);
    return 1;
}
