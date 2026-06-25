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

    if (g_fail == 0) { std::printf("draco_test: ALL PASS\n"); return 0; }
    std::printf("draco_test: %d FAIL\n", g_fail);
    return 1;
}
