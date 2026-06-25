# Slice DRACO-DR1 — Bitstream reader + Draco header (Issue #36, beachhead)

The beachhead of the SELF-CONTAINED, CLEAN-ROOM DRACO DECODER (issue #36 — load `KHR_draco_mesh_compression`
glTF meshes, which are >50% of the real-world glTF library). Implemented spec-driven from the published Draco
bitstream specification (github.com/google/draco/docs/spec — Apache-2.0; we implement from the SPEC, we do NOT
vendor or copy draco source). The decoded mesh is deterministic (a codec: same bytes in → same bytes out), so
every slice is golden-verified with pinned `net::DigestBytes`, cross-platform MSVC/Windows-clang/Mac-clang.

DR1 establishes the substrate: a little-endian byte reader, the LEB128 varint decode, and the Draco header
parser — the foundation every later slice builds on. Golden = constructed varint test vectors + parsing the
REAL Box-Draco header (a known cube asset, the decode oracle for the whole flagship).

## NEW file: engine/asset/draco_decode.h (namespace hf::asset::draco)
Header-only and **SELF-CONTAINED**: include ONLY `<cstddef>`, `<cstdint>`, `<vector>`, plus
`#include "net/session.h"` (for `hf::net::DigestBytes`). NO `<cmath>` / float / clock / RNG / `<random>` /
`<unordered_*>` / `<map>` / `<functional>` / `std::hash` / `<algorithm>` / `<string>`. Do NOT include draco
or cgltf. It MUST compile standalone: `clang++ -std=c++20 -I engine -I tests tests/draco_test.cpp`. This is
ONE growing header — every later slice (DR2–DR5) APPENDS below DR1; do NOT modify DR1's symbols once pinned.
(The decoder is pure-CPU integer until the final dequantize float step in DR3.)

### Spec source (re-fetch per slice — read-only reference)
`curl https://raw.githubusercontent.com/google/draco/master/docs/spec/<file>.md`. DR1 uses `core.functions.md`
(LEB128, mem_get_le16/24/32) + `draco.decoder.md` (ParseHeader) + `variable.descriptions.md` (constants).

### Types + constants (all in hf::asset::draco)
```cpp
// Draco constants (from variable.descriptions.md — FROZEN, the wire contract).
constexpr uint16_t kMetadataFlagMask        = 32768;
enum EncoderType   : uint8_t { kPointCloud = 0, kTriangularMesh = 1 };
enum MeshEncMethod : uint8_t { kMeshSequential = 0, kMeshEdgebreaker = 1 };

struct DracoHeader {
    uint8_t  major = 0, minor = 0;     // bitstream version (Box = 2.2)
    uint8_t  encoderType = 0;          // EncoderType (Box = kTriangularMesh)
    uint8_t  encoderMethod = 0;        // MeshEncMethod (Box = kMeshEdgebreaker)
    uint16_t flags = 0;                // bit kMetadataFlagMask => metadata present
    bool     valid = false;            // magic == "DRACO"
};
```

### The byte reader (little-endian cursor over a span)
```cpp
struct ByteReader {
    const uint8_t* p = nullptr;
    std::size_t    n = 0;
    std::size_t    pos = 0;
    bool           error = false;      // set on any out-of-bounds read; reads then return 0 (deterministic)

    uint8_t  U8();                                  // 1 byte, ++pos (mem_get + advance)
    uint32_t LE16();                                // mem_get_le16: mem[0] | mem[1]<<8 ; pos += 2
    uint32_t LE24();                                // mem_get_le24
    uint32_t LE32();                                // mem_get_le32
    void     Skip(std::size_t k);                   // pos += k
    std::size_t Remaining() const;                  // n - pos (0 if past end)
    // LEB128 (core.functions.md): result |= (in & 0x7F) << shift; continue while (in & 0x80); shift += 7.
    uint32_t VarU32();                              // LEB128 truncated to 32 bits
    uint64_t VarU64();                              // LEB128 full 64 bits
};
```
(All reads bounds-checked: out-of-range sets `error=true` and returns 0 — never UB. The `mem_get_le*` bodies
match `core.functions.md` exactly: byte 0 is the low byte.)

### `ParseHeader`
```cpp
// draco.decoder.md ParseHeader: UI8[5] magic "DRACO", UI8 major, UI8 minor, UI8 encoderType, UI8
// encoderMethod, UI16 flags. Returns valid=false (not an exception) on bad magic / truncation.
inline DracoHeader ParseHeader(ByteReader& r);
```
(Note: the 5-byte magic is read as raw bytes and compared to `{'D','R','A','C','O'}`. `flags` is `LE16`.)

## The golden (PINNED, cross-platform) — tests/draco_test.cpp
Self-contained test in the `seq_test.cpp` shape (copy `check()` + `HF_TEST_MAIN_INIT()` from
`tests/test_main.h`). Register `hf_add_pure_test(draco_test)` in `tests/CMakeLists.txt` next to `seq_test`.
**Embed the Box-Draco header bytes as a literal** (the first bytes of `assets/models/BoxDraco/Box.bin`:
`44 52 41 43 4f 02 02 01 01 01 00` = "DRACO" + 2.2 + TriMesh + Edgebreaker + flags 0x0001) so the test is
standalone (no file IO). (DR3/DR4 will embed the full 120-byte `Box.bin` for the body decode.)
```
draco-dr1: varint sweep digest = 0x<...>
PASS draco-dr1: LEB128 known vectors decode exactly ({0x00}->0, {0x7F}->127, {0x80,0x01}->128, ...)
PASS draco-dr1: the varint decode sweep digest == pinned uint64 (the reader is byte-stable cross-platform)
PASS draco-dr1: mem_get_le16/24/32 read little-endian exactly (low byte first)
PASS draco-dr1: ParseHeader(Box header) == {valid, v2.2, kTriangularMesh, kMeshEdgebreaker, flags=1}
PASS draco-dr1: a bad-magic / truncated header parses to valid=false (no UB, deterministic)
```
Assertions:
1. **LEB128 VECTORS** — construct byte arrays and assert `VarU32`/`VarU64` decode the exact values:
   `{0x00}→0`, `{0x7F}→127`, `{0x80,0x01}→128`, `{0xE5,0x8E,0x26}→624485`, `{0xFF,0xFF,0xFF,0xFF,0x0F}→
   0xFFFFFFFF`. (Standard LEB128 — verify against these well-known vectors.)
2. **VARINT SWEEP PINNED** — encode a fixed sweep of values `0, 1, 127, 128, 16383, 16384, 2097151, 2097152,
   0xFFFFFFFF` as LEB128 bytes (a small hand-written encoder IN THE TEST), concatenate, then decode them all
   back with `VarU32` and `net::DigestBytes` the recovered `uint32` vector == a hard-pinned `uint64_t`
   (identical MSVC + clang — the cross-platform anchor). (Also assert the decoded values equal the originals.)
3. **LITTLE-ENDIAN** — `mem_get_le16({0x34,0x12}) == 0x1234`; `LE24({0x56,0x34,0x12}) == 0x123456`;
   `LE32({0x78,0x56,0x34,0x12}) == 0x12345678`.
4. **REAL HEADER** — `ParseHeader` over the embedded Box header → `valid == true`, `major == 2`,
   `minor == 2`, `encoderType == kTriangularMesh`, `encoderMethod == kMeshEdgebreaker`, `flags == 1`,
   `(flags & kMetadataFlagMask) == 0` (no metadata).
5. **BAD INPUT** — a 3-byte buffer (truncated) and a buffer with magic `"XRACO"` both parse to
   `valid == false` with no crash (the bounds-checked reader).

## Cross-platform proof (the cheap loop — NO render-bake)
The CONTROLLER `scp`s `engine/asset/draco_decode.h` + `engine/net/session.h` + `tests/draco_test.cpp` (+
`tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine
-I tests tests/draco_test.cpp -o /tmp/draco && /tmp/draco`, confirming the test PASSES with the IDENTICAL
pinned digest. (Local Windows clang is the fast pre-check.) NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- NEW header `engine/asset/draco_decode.h`; compiles STANDALONE under clang with `-I engine -I tests`
  (self-contained: only `<cstddef>/<cstdint>/<vector>` + `net/session.h`). Do NOT modify `net/session.h` /
  any existing header. Do NOT include draco / cgltf / `<cstring>`. Do NOT add it to any RHI/GPU target.
- Pure-CPU INTEGER: NO float / `<cmath>` / clock / RNG / `<unordered_*>` / `<map>` / `std::hash` /
  `<algorithm>` / `<string>`. The reader is bounds-checked (out-of-range → `error=true`, returns 0, never
  UB).
- Implement from the SPEC pseudocode (LEB128, mem_get_le*, ParseHeader) — match it exactly. Do NOT copy
  draco source.
- `tests/draco_test.cpp` is SELF-CONTAINED (embed the Box header bytes as a literal — no file IO). Register
  `hf_add_pure_test(draco_test)` in `tests/CMakeLists.txt`. Use `test_main.h` `HF_TEST_MAIN_INIT()`.
- Branch `fix-draco-dr1`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`. (The Box asset
  under `assets/models/BoxDraco/` may be committed now — it is the oracle DR3+ needs — OR left for DR3; your
  call, but if committed, add BOTH `Box.gltf` + `Box.bin`.)
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target draco_test'`
  (if the `cmd /c '...&&...'` handoff swallows output, run the build via the PowerShell tool natively). Run
  the test exe, confirm it PRINTS the digest and PASSES. First run: pin the printed digest, rebuild, confirm
  green. ALSO compile standalone with the local clang and confirm the IDENTICAL digest (MSVC==clang).
- COMPLETION CRITERIA — do NOT commit until: the header compiles, `draco_test` builds + PASSES on Windows
  with all assertions green, and the local clang standalone passes with the identical digest. Report: the
  commit hash, the full test output (printed digest + PASS lines), the exact pinned `uint64_t`, confirmation
  the header is self-contained (list its `#include`s — exactly 4), the LEB128 vectors you verified, and the
  local-clang result. Commit message via a temp file + `git commit -F` (Bash heredoc). Commit to the branch
  and STOP.
  (The CONTROLLER audits the diff, runs the Mac/clang standalone to confirm the IDENTICAL digest, then
  ff-merges to master + pushes + deletes the branch + advances to DR2 — the rANS entropy decoder.)
