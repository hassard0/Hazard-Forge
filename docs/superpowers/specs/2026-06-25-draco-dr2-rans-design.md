# Slice DRACO-DR2 — rANS entropy decoder (Issue #36)

DR1 gave the bitstream reader + header (varint sweep `0x2d4aaca6fd14312a`). DR2 adds the **rANS entropy
decoder** — the shared symbol/binary range-ANS coder that BOTH the edgebreaker connectivity (DR3) and the
attribute residuals (DR4) decode through. This is the entropy foundation. Implemented spec-driven from
`rans.decoding.md` + the constants in `variable.descriptions.md` (clean-room, NOT copied from draco source).

**Golden strategy (honest):** rANS is a leaf primitive only exercised in context (real validation happens at
DR3/DR4 when the Box's actual symbols decode to the known cube). DR2's cross-platform golden is a
**self-contained round-trip**: a minimal rANS *encoder* lives IN THE TEST (test scaffolding, NOT shipped),
encodes a fixed symbol sequence, and the DR2 *decoder* recovers it bit-exactly — proving the decoder is
deterministic + self-consistent + cross-platform-identical. (Caveat documented: the round-trip proves
consistency, not yet Draco-encoder correctness; that is DR3/DR4's job against the real Box.)

Scope DR2 to the **RAW symbol path + the binary rABS path** (both fully self-contained — no bit reader). The
**TAGGED symbol path + the bit reader** are deferred to DR3 (where edgebreaker/tagged first need them).

Pure-integer, append-only to `engine/asset/draco_decode.h` (below DR1; do NOT modify DR1 —
`0x2d4aaca6fd14312a` stays pinned). NO new include.

## Append to engine/asset/draco_decode.h (below DR1, in hf::asset::draco)

### 1. Constants (from variable.descriptions.md — FROZEN)
```cpp
constexpr uint32_t kIoBase            = 256;
constexpr uint32_t kLRansBase         = 4096;     // L_RANS_BASE (raw default base before precision scaling)
constexpr uint32_t kTaggedRansBase    = 16384;
constexpr uint32_t kTaggedRansPrec    = 4096;
constexpr uint8_t  kTaggedSymbols     = 0;
constexpr uint8_t  kRawSymbols        = 1;
constexpr uint32_t kRabsP8Precision   = 256;      // rabs_ans_p8_precision
constexpr uint32_t kRabsLBase         = 4096;     // rabs_l_base
```

### 2. The rANS symbol decoder (spec: RansInitDecoder / RansRead / fetch_sym)
```cpp
struct AnsDecoder { const uint8_t* buf = nullptr; int buf_offset = 0; uint32_t state = 0; };
struct RansSym    { uint32_t val = 0, prob = 0, cum_prob = 0; };

// RansInitDecoder: x = buf[offset-1] >> 6 selects 1..4 state bytes (LE, masked 0x3F/0x3FFF/0x3FFFFF/
// 0x3FFFFFFF); ans.state += l_rans_base. (Reuse DR1 mem_get_le16/24/32 — they read low byte first.)
inline void RansInitDecoder(AnsDecoder& ans, const uint8_t* buf, int offset, uint32_t l_rans_base);

// RansRead: renorm-in (while state < l_rans_base && buf_offset>0: state = state*kIoBase + buf[--buf_offset]);
// quo=state/precision, rem=state%precision; fetch_sym via lut_table[rem]; state = quo*sym.prob + rem -
// sym.cum_prob; return sym.val. lut_table_ + probability_table_ passed in.
inline uint32_t RansRead(AnsDecoder& ans, uint32_t l_rans_base, uint32_t rans_precision,
                         const std::vector<uint32_t>& lut_table, const std::vector<RansSym>& prob_table);
```

### 3. Probability tables (spec: BuildSymbolTables / rans_build_look_up_table)
```cpp
// BuildSymbolTables: read num_symbols_ per-symbol prob tokens. For each: prob_data UI8; token = prob_data&3;
// if token==3: offset = prob_data>>2; zero (offset+1) probs; i += offset. else: prob = prob_data>>2; read
// `token` extra bytes eb, prob |= eb << (8*(j+1)-2); token_probs[i]=prob. Then rans_build_look_up_table:
// cum_prob accumulates; lut_table_[act_prob..cum_prob) = i; probability_table_[i] = {prob, cum_prob}.
// Returns the built lut_table_ + probability_table_ (sized to rans_precision for the lut). `r` is the
// DR1 ByteReader positioned at the prob-token stream.
inline void BuildSymbolTables(ByteReader& r, uint32_t num_symbols, uint32_t rans_precision,
                              std::vector<uint32_t>& lut_table, std::vector<RansSym>& prob_table);
```

### 4. `DecodeRawSymbols` + `DecodeSymbols` (RAW path; TAGGED deferred to DR3)
```cpp
// DecodeRawSymbols (rans.decoding.md): max_bit_length UI8; num_symbols_ varUI32; rans_precision_bits =
// clamp((3*max_bit_length)/2, 12, 20); rans_precision = 1<<bits; l_rans_base = rans_precision*4;
// BuildSymbolTables; size varUI64; buffer UI8[size]; RansInitDecoder(buffer, size, l_rans_base); loop
// num_values RansRead -> out. (`r` is the ByteReader over the whole symbol blob; the rANS buffer is the
// next `size` bytes — RansInitDecoder reads them BACKWARD from buffer[size-1].)
inline bool DecodeRawSymbols(ByteReader& r, uint32_t num_values, std::vector<uint32_t>& out);

// DecodeSymbols: scheme UI8; if kTaggedSymbols -> DecodeTaggedSymbols (DR3, return false/unsupported for
// now); if kRawSymbols -> DecodeRawSymbols. num_components only matters for tagged.
inline bool DecodeSymbols(ByteReader& r, uint32_t num_values, uint32_t num_components,
                          std::vector<uint32_t>& out);
```

### 5. The binary rABS decoder (spec: RabsDescRead)
```cpp
// RabsDescRead: p = kRabsP8Precision - p0; if state<kRabsLBase: state = state*kIoBase + buf[--buf_offset];
// x=state; quot=x/kRabsP8Precision; rem=x%kRabsP8Precision; xn=quot*p; val = rem<p; state = val ? xn+rem :
// x-xn-p; return val. (An init mirrors RansInitDecoder with l_rans_base = kRabsLBase.) Used for boolean
// streams (e.g. edgebreaker start-face config, attribute flags).
inline void RabsInitDecoder(AnsDecoder& ans, const uint8_t* buf, int offset);  // l_rans_base = kRabsLBase
inline uint8_t RabsDescRead(AnsDecoder& ans, uint32_t p0);
```

## The golden (PINNED, cross-platform) — append to tests/draco_test.cpp
The TEST implements a minimal RAW-rANS *encoder* (the inverse of RansRead/RansInitDecoder) + a rABS encoder —
TEST-ONLY scaffolding — to produce valid streams, then the DR2 decoder recovers them. (Standard rANS encode:
build the same prob table; for symbols in REVERSE: renorm-out while state >= ((l_rans_base/precision)*kIoBase)
*prob: emit state%kIoBase, state/=kIoBase; then state = (state/prob)*precision + (state%prob) + cum_prob;
finally flush state via the RansInitDecoder header byte(s). rABS encode is the inverse of RabsDescRead.)
```
draco-dr2: raw-rans roundtrip digest = 0x<...>   rabs roundtrip digest = 0x<...>
PASS draco-dr1: ... (DR1 varint digest 0x2d4aaca6fd14312a UNCHANGED)
PASS draco-dr2: rans_build_look_up_table builds a correct cumulative table (lut maps rem->symbol)
PASS draco-dr2: RAW rANS round-trip — encode a fixed symbol sequence, DecodeRawSymbols recovers it EXACTLY
PASS draco-dr2: the recovered-symbols digest == pinned uint64 (deterministic + byte-stable cross-platform)
PASS draco-dr2: binary rABS round-trip — encode a fixed bit sequence with p0, RabsDescRead recovers it exactly
PASS draco-dr2: a different symbol/probability changes the digest (the coder is load-bearing)
```
Assertions:
1. **DR1 INVARIANT** — re-assert the DR1 varint sweep digest `0x2d4aaca6fd14312a` UNCHANGED.
2. **TABLE** — build a probability table for a small symbol set (e.g. probs summing to `rans_precision`) and
   assert `rans_build_look_up_table` fills `lut_table[rem]` = the symbol whose cumulative range contains
   `rem`, and `prob_table[s].cum_prob` is the running sum (a direct unit check, no encoder needed).
3. **RAW ROUND-TRIP** — a fixed symbol sequence (e.g. `{0,1,2,1,0,3,2,2,1,0,...}` with a fixed probability
   distribution), encode with the test encoder, `DecodeRawSymbols` → assert the recovered vector EQUALS the
   original.
4. **PINNED RAW DIGEST** — `net::DigestBytes` of the recovered symbol vector == a hard-pinned `uint64_t`
   (identical MSVC + clang). (Also pin the encoded-bytes digest as a second anchor if convenient.)
5. **rABS ROUND-TRIP** — a fixed bit sequence with a fixed `p0`, encode, `RabsDescRead` × N → assert the
   recovered bits equal the original; pin the recovered-bits digest.
6. **LOAD-BEARING** — change one symbol (or the probability table) → the recovered/encoded digest changes
   (the coder actually depends on its input).

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/asset/draco_decode.h` + `engine/net/session.h` + `tests/draco_test.cpp` (+
`tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine
-I tests tests/draco_test.cpp -o /tmp/draco && /tmp/draco`, confirming ALL assertions PASS with the IDENTICAL
pinned digests. Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/asset/draco_decode.h` (add the constants, `AnsDecoder`/`RansSym`, `RansInitDecoder`/
  `RansRead`/`fetch_sym` (inline helper)/`BuildSymbolTables`/`rans_build_look_up_table`/`DecodeRawSymbols`/
  `DecodeSymbols`/`RabsInitDecoder`/`RabsDescRead` below DR1). Do NOT modify DR1 — `0x2d4aaca6fd14312a`
  stays pinned. The TAGGED path may be a stub that returns false (DR3 implements it).
- NO new include. Self-contained (4 includes). STILL NO `<cmath>`/float/clock/RNG/`<unordered_*>`/`<map>`/
  `std::hash`/`<algorithm>`/`<string>`/`<cstring>`. All integer; reads bounds-checked (the DR1 reader
  discipline). Implement from the SPEC pseudocode — match it exactly; do NOT copy draco source.
- `tests/draco_test.cpp` stays self-contained; APPEND the DR2 assertions + the test-only rANS/rABS encoder.
  Keep ALL DR1 assertions green.
- Branch `fix-draco-dr2`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target draco_test'`
  (if the `cmd /c '...&&...'` handoff swallows output, run the build via the PowerShell tool natively). Run
  `draco_test`, confirm ALL assertions (DR1 + DR2) PASS, exit 0. ALSO compile standalone with the local clang
  `C:\Program Files\LLVM\bin\clang++.exe` and confirm the IDENTICAL digests. First run: pin the raw + rABS
  round-trip digests, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `draco_test` builds + PASSES on Windows
  with every assertion green (esp. DR1 digest unchanged + the RAW round-trip + the rABS round-trip + the
  table check), and the local clang standalone passes with identical digests. Report: commit hash, full test
  output (printed digests + PASS lines), the pinned raw + rABS `uint64`s, confirmation the DR1 digest is
  unchanged, confirmation the header is self-contained (4 includes), the symbol sequence + probabilities you
  round-tripped, how the test-only encoder works (so I can audit it's a faithful inverse), and the local-clang
  result. Flag any deviation. Commit message via a temp file + `git commit -F` (Bash heredoc). Commit to the
  branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to DR3 — the edgebreaker connectivity decoder, which uses
  this rANS coder to decode the Box's actual CLERS symbols into the cube topology.)
