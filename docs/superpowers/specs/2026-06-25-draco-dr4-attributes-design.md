# Slice DRACO-DR4 — Attribute decode (Issue #36)

DR3 decoded the Box connectivity (12 faces / 8 verts, digest `0x1f478b2e11afa703`). DR4 decodes the
**attributes** — the quantized POSITION (and NORMAL) residuals + the prediction scheme + dequantization —
turning the connectivity into the cube's actual **vertex positions** (8 corner coordinates forming a unit
cube). This is the second real-correctness proof: the decoded positions must be a valid cube. Spec-driven
clean-room (NOT copied from draco source).

**Read the spec (re-fetch read-only):** `attributes.decoder`, `sequential.decoder`,
`sequential.integer.attribute.decoder`, `sequential.quantization.attribute.decoder`,
`sequential.normal.attribute.decoder`, `prediction.decoder`, `prediction.parallelogram.decoder`,
`prediction.multi.parallelogram.decoder`, `prediction.wrap.transform`, `prediction.difference` (if present),
plus `variable.descriptions` (constants). Follow the pseudocode EXACTLY.

Pure-integer until the final dequantize (the ONE float step — positions become floats). Append-only to
`engine/asset/draco_decode.h` (below DR3; do NOT modify DR1–DR3 — varint `0x2d4aaca6fd14312a`, rANS
`0xbc91b8ba74fbf8b1`, rABS `0xb6efc1e48524ecd8`, connectivity `0x1f478b2e11afa703` stay pinned). NO new
include (`<cmath>` is NOT needed — dequant is integer→float via multiply/add).

## What DR4 must add (in hf::asset::draco)

### The attribute-decode pipeline (attributes.decoder.md — DecodeAttributeData phases)
For the Box (1 attribute decoder, POSITION+NORMAL, MESH_EDGEBREAKER), implement the phases that matter:
- **`ParseAttributeDecodersData`** — `num_attributes_decoders` UI8; per decoder the data-id/decoder-type/
  traversal-method + per attribute the att_type/data_type/num_components/normalized/unique_id +
  seq_att_dec_decoder_type.
- **The corner→value mapping** — `DecodeAttributeSeams`, `UpdateVertexToCornerMap`, `GenerateSequence`
  (the traversal that orders encoded attribute values), `encoded_attribute_value_index_to_corner_map`,
  `UpdatePointToAttributeIndexMapping`. For a pure vertex attribute (POSITION) this maps each encoded value
  to a vertex/corner. (Implement the general edgebreaker path per the spec; the Box has no attribute seams
  for POSITION, so the simple per-vertex mapping applies.)
- **`DecodePortableAttributes`** → per attribute, the **sequential integer attribute decode**
  (`sequential.integer.attribute.decoder.md`): read the prediction scheme + transform, decode the integer
  residuals via the DR2 `DecodeSymbols` (the rANS), then run the **prediction scheme** in the corner-table
  traversal order to reconstruct the integer attribute values. The Box's POSITION likely uses
  **MESH_PREDICTION_PARALLELOGRAM** (or DIFFERENCE) with the **WRAP transform**
  (`prediction.parallelogram.decoder.md` + `prediction.wrap.transform.md`): each value = prediction(from the
  parallelogram of decoded neighbors via the corner table) + residual, wrapped into the quantized range.
- **`DecodeDataNeededByPortableTransforms` + `TransformAttributesToOriginalFormat`** → the **quantization
  transform** (`sequential.quantization.attribute.decoder.md`): read min_values[] (f32 per component) +
  range (f32) + quantization_bits; dequantize each integer value: `f = min + (q / max_quantized_value) *
  range` (the ONE float step). For NORMAL (octahedral), the normal transform — implement if the Box's NORMAL
  needs it, else decode POSITION first (the headline) and NORMAL best-effort.

### The decode entry + result
```cpp
struct DecodedMesh {
    uint32_t num_faces = 0, num_points = 0;
    std::vector<float>    positions;   // num_points * 3 (the dequantized cube corners)
    std::vector<float>    normals;     // num_points * 3 (if present)
    std::vector<uint32_t> indices;     // num_faces * 3 (from DR3 face_to_vertex, point-mapped)
    bool ok = false;
};
// Full decode: ParseHeader (DR1) -> DecodeConnectivity (DR3) -> DecodeAttributeData (DR4) -> assemble.
inline DecodedMesh DecodeDracoMesh(const uint8_t* bytes, std::size_t n);
```

## The golden (PINNED, cross-platform) — append to tests/draco_test.cpp
Decode the embedded `kBoxBin` (DR3's literal) fully.
```
draco-dr4: box positions digest = 0x<...>   (points=<N>)
PASS draco-dr1..dr3: ... (all prior digests UNCHANGED)
PASS draco-dr4: DecodeDracoMesh(Box) succeeds + has 12 faces
PASS draco-dr4: the decoded POSITIONs form a valid unit cube (8 distinct corners, axis-aligned)
PASS draco-dr4: the positions digest == pinned uint64 (deterministic + byte-stable cross-platform)
PASS draco-dr4: re-decoding is bit-identical (deterministic)
```
Assertions:
1. **PRIOR INVARIANT** — DR1/DR2/DR3 digests UNCHANGED.
2. **DECODE** — `DecodeDracoMesh(Box)` → `ok`, `num_faces == 12`.
3. **VALID CUBE (THE PROOF)** — the decoded positions are 8 distinct corners of an axis-aligned cube. Assert
   the structure: exactly 8 (or 24) distinct positions; each coordinate takes exactly 2 distinct values (a
   box); report the 8 corner coords so I can verify (e.g. a unit cube `{-0.5,0.5}^3` or `{0,1}^3` or similar
   — the Box.gltf's min/max accessor bounds confirm the expected extent).
4. **PINNED POSITIONS DIGEST** — `net::DigestBytes` over the position floats (their raw IEEE-754 bits, hand-LE
   — positions are deterministic so this is byte-stable) == a hard-pinned `uint64_t` (identical MSVC + clang;
   the dequant float math is exact for the Box's values).
5. **DETERMINISTIC** — re-decode → identical.

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/asset/draco_decode.h` + `engine/net/session.h` + `tests/draco_test.cpp` (+ the test
scaffold) to the Mac and runs the standalone clang compile, confirming faces==12 + the cube + the IDENTICAL
positions digest. (If float dequant differs MSVC vs clang in the low bits, the positions are still a valid
cube — but pin the digest only if MSVC==clang; if they differ, quantize the reported positions to a tolerance
and assert the cube structure + a fixed-point digest instead. Report which.)

## Constraints (HARD)
- APPEND-ONLY to `engine/asset/draco_decode.h`. Do NOT modify DR1–DR3 — their digests stay pinned. NO new
  include. Self-contained (4 includes). The dequant is the ONLY float (`(float)` casts + multiply/add — NO
  `<cmath>`). Bounds-checked; malformed → ok=false. Implement from the SPEC; do NOT copy draco source.
- `tests/draco_test.cpp` stays self-contained. Keep ALL DR1–DR3 assertions green.
- Branch `fix-draco-dr4`, commit there, do NOT merge. Do NOT commit `tests/golden/*`.
- Build Windows (PowerShell tool, single-quoted cmd for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target draco_test'`
  (native PowerShell if the handoff swallows output). Run `draco_test`, ALL (DR1–DR4) PASS, exit 0. ALSO
  local clang standalone, identical digests. First run: pin the positions digest.
- **Hard slice — report honestly.** If the POSITIONs don't form a cube, DEBUG against the spec (the
  prediction-scheme parallelogram corner walk, the wrap transform, the quantization min/range parse, the
  traversal order are the culprits). Do NOT fake the golden — report the decoded positions you DO get + your
  diagnosis. An honest partial ("residuals decode, prediction runs, positions are X not a cube because Y") is
  more valuable than a fake pass.
- COMPLETION CRITERIA — commit when: the header compiles, `draco_test` PASSES with the cube + pinned digest +
  DR1–DR3 unchanged, local clang matches. Report: commit hash (or honest blocker), full output, the 8 decoded
  corner positions, the prediction scheme + quantization-bits the Box uses, confirmation DR1–DR3 unchanged +
  self-contained, the local-clang result, any deviation. Commit via temp file + `git commit -F`, STOP.
  (The CONTROLLER audits, Mac/clang-proves the cube + digest, ff-merges + advances to DR5 — wiring
  DecodeDracoMesh into the glTF loader so real Draco glTF files load, closing #36.)
