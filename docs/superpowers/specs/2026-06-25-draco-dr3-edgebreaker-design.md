# Slice DRACO-DR3 — Edgebreaker connectivity decoder (Issue #36, THE CRUX)

DR1 gave the bitstream+header, DR2 the rANS entropy core. DR3 decodes the **edgebreaker connectivity** — the
hardest part of Draco: a corner-table mesh topology reconstructed from a CLERS symbol traversal, producing
the triangle `face_to_vertex` lists. Validated end-to-end against the REAL Box oracle (`Box.bin` → exactly 12
faces / a cube). This is the first slice that decodes ACTUAL Draco data (not a round-trip), so it is the first
real-correctness proof. Spec-driven clean-room from the published spec (NOT copied from draco source).

**Read the spec files (re-fetch read-only):** `curl https://raw.githubusercontent.com/google/draco/master/docs/spec/<f>.md`
for: `connectivity.decoder`, `edgebreaker.decoder`, `edgebreaker.traversal`, `edgebreaker.traversal.valence`,
`corner`, `boundary.decoder`, and `sequential.decoder` (for the symbol/face-data parse helpers
`ParseEdgebreakerTraversalStandardFaceData`). Follow their pseudocode EXACTLY.

Pure-integer, append-only to `engine/asset/draco_decode.h` (below DR2; do NOT modify DR1/DR2 — varint
`0x2d4aaca6fd14312a`, rANS `0xbc91b8ba74fbf8b1`, rABS `0xb6efc1e48524ecd8` stay pinned). NO new include.

## What DR3 must add (in hf::asset::draco)

### 0. Deferred-from-DR2 prerequisites (the bit reader + tagged symbols)
- A **bit reader** over a byte span: `ReadBits(nbits)` (Draco reads bits LSB-first within each byte; see how
  `eb_symbol_buffer.ReadBits` + `ResetBitReader` are used) + `ResetBitReader()`. The edgebreaker symbol
  buffer + the topology-split `f[1]` bits use it.
- `DecodeTaggedSymbols` (the DR2 stub): now implement it (uses `BuildSymbolTables` + `RansInitDecoder` +
  `RansRead` at `TAGGED_RANS_BASE`/`TAGGED_RANS_PRECISION` per group + the bit reader for the `size`-bit
  raw values + `ResetBitReader`). `DecodeSymbols` already branches to it.

### 1. Corner-table primitives (corner.md — exact)
```cpp
inline int Next(int c)     { return c < 0 ? c : ((c % 3) == 2 ? c - 2 : c + 1); }
inline int Previous(int c) { return c < 0 ? c : ((c % 3) == 0 ? c + 2 : c - 1); }
// opposite_corners_ (a growable int vector, default -1); PosOpposite(c) = c<size? opposite_corners_[c] : -1;
// SetOppositeCorners(a,b) sets both; SwingLeft(c)=Next(Opposite(Next(c))); SwingRight similar.
// CornerToVert via corner_to_vertex_map_ (MapCornerToVertex(c,v) sets it). CornerToVerts(c)->(v,n,p) =
// (map[c], map[Next(c)], map[Previous(c)]).
```
(For DR3, att_dec=0 / MESH_VERTEX_ATTRIBUTE, so `Opposite` == `PosOpposite`. The attribute-seam paths are
DR4+; stub them to PosOpposite.)

### 2. Parse + decode (edgebreaker.decoder.md + .traversal[.valence].md)
- `ParseEdgebreakerConnectivityData`: `edgebreaker_traversal_type` UI8 (0=STANDARD, 2=VALENCE),
  `num_encoded_vertices` varUI32, `num_faces` varUI32, `num_attribute_data` UI8, `num_encoded_symbols`
  varUI32, `num_encoded_split_symbols` varUI32.
- `DecodeTopologySplitEvents` (ParseTopologySplitEvents + ProcessSplitData) — the split source/split symbol
  ids + the `source_edge_bit` f[1] bits (via the bit reader). For the Box (a closed cube) there are likely 0
  splits, but implement it per spec.
- `EdgebreakerTraversalStart` → for VALENCE: `EdgeBreakerTraversalValenceStart`
  (`ParseEdgebreakerTraversalStandardFaceData` + `ParseEdgebreakerTraversalStandardAttributeConnectivityData`
  + init `vertex_valences_` + per-context `ParseValenceContextCounters` + `DecodeSymbols` into
  `ebv_context_symbols[i]`). For STANDARD: read the `eb_symbol_buffer` (the bit-coded CLERS stream). Implement
  BOTH (the Box's `edgebreaker_traversal_type` selects; report which it is).
- `EdgebreakerDecodeSymbol`: STANDARD = `ParseEdgebreakerStandardSymbol` (ReadBits(1); if !=TOPOLOGY_C read 2
  more bits, symbol |= suffix<<1); VALENCE = `EdgebreakerValenceDecodeSymbol` (pull from
  `ebv_context_symbols[active_context_]` by `--ebv_context_counters`, mapping the symbol index to a CLERS
  topology via the spec's table). TOPOLOGY constants: C=0,S=1,L=3,R=5,E=7.
- `DecodeEdgeBreakerConnectivity`: `is_vert_hole_.assign(num_encoded_vertices+num_encoded_split_symbols,
  true); last_vert_added=-1; for i in num_encoded_symbols: EdgebreakerDecodeSymbol(); NewActiveCornerReached(
  3*i, i); then ProcessInteriorEdges()`.
- `NewActiveCornerReached(new_corner, symbol_id)`: the big switch on `last_symbol_` (C/S/L/R/E) — implement
  EXACTLY per `edgebreaker.decoder.md` (the spec body is in the design references; corner_a/corner_b
  manipulation, SetOppositeCorners, face_to_vertex push, MapCornerToVertex, vertex_valences_ updates for
  VALENCE, the active_corner_stack, the check_topology_split / IsTopologySplit loop, ReplaceVerts/
  UpdateCornersAfterMerge for S). This is the heart — match the pseudocode line-for-line.
- `ProcessInteriorEdges`: RansInitDecoder over `eb_start_face_buffer` (L_RANS_BASE) + per active corner
  RabsDescRead(eb_start_face_buffer_prob_zero) → if interior, weld the corner (the spec body).
- Constants needed (fetch from variable.descriptions.md): TOPOLOGY_C/S/L/R/E, STANDARD_EDGEBREAKER=0,
  VALENCE_EDGEBREAKER=2, MIN_VALENCE/MAX_VALENCE/NUM_UNIQUE_VALENCES, LEFT_FACE_EDGE/RIGHT_FACE_EDGE,
  kInvalidCornerIndex=-1.

### 3. The decode entry + result
```cpp
struct Connectivity {
    uint32_t num_faces = 0, num_vertices = 0;
    std::vector<uint32_t> face_to_vertex[3];   // the 3 corner-vertex lists; face f = (fv[0][f],fv[1][f],fv[2][f])
    bool ok = false;
};
// Parse the Box.bin body after the header: DecodeConnectivityData() per connectivity.decoder.md.
inline Connectivity DecodeConnectivity(ByteReader& r, const DracoHeader& h);
```

## The golden (PINNED, cross-platform) — append to tests/draco_test.cpp
**Embed the full 120-byte `Box.bin` as a literal `constexpr uint8_t kBoxBin[120]`** (read it from
`assets/models/BoxDraco/Box.bin`) so the test decodes the REAL asset standalone.
```
draco-dr3: box connectivity digest = 0x<...>   (faces=<N>)
PASS draco-dr1/dr2: ... (DR1/DR2 digests UNCHANGED)
PASS draco-dr3: DecodeConnectivity(Box.bin) succeeds (ok==true)
PASS draco-dr3: the Box decodes to exactly 12 faces (a cube — the real-correctness proof)
PASS draco-dr3: the face_to_vertex digest == pinned uint64 (deterministic + byte-stable cross-platform)
PASS draco-dr3: re-decoding is bit-identical (deterministic)
```
Assertions:
1. **PRIOR INVARIANT** — DR1 `0x2d4aaca6fd14312a`, DR2 `0xbc91b8ba74fbf8b1` + `0xb6efc1e48524ecd8` UNCHANGED.
2. **DECODE OK** — `DecodeConnectivity(Box)` returns `ok == true`.
3. **12 FACES (THE REAL PROOF)** — `num_faces == 12` (a cube has 12 triangles). This is the make-or-break:
   the clean-room edgebreaker decoded the Box's ACTUAL CLERS topology correctly.
4. **PINNED DIGEST** — `net::DigestBytes` over the three `face_to_vertex` lists (concatenated, hand-LE) == a
   hard-pinned `uint64_t` (identical MSVC + clang). (Report the actual decoded faces so I can sanity-check
   they form a valid cube — 8 distinct vertices, each used by multiple tris.)
5. **DETERMINISTIC** — a second decode → identical digest.

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/asset/draco_decode.h` + `engine/net/session.h` + `tests/draco_test.cpp` (+
`tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine
-I tests tests/draco_test.cpp -o /tmp/draco && /tmp/draco`, confirming ALL assertions PASS with the IDENTICAL
pinned digests (esp. faces==12 + the face digest). Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/asset/draco_decode.h`. Do NOT modify DR1/DR2 — their digests stay pinned. The
  TAGGED-symbol stub from DR2 is now implemented (that is an allowed completion, not a DR2 modification).
- NO new include. Self-contained (4 includes). STILL NO `<cmath>`/float/clock/RNG/`<unordered_*>`/`<map>`/
  `std::hash`/`<algorithm>`/`<string>`/`<cstring>`. All integer, bounds-checked (the DR1 reader discipline —
  corner/vertex vectors grow as needed; a malformed stream sets ok=false, never crashes). Implement from the
  SPEC pseudocode — match it exactly; do NOT copy draco source.
- `tests/draco_test.cpp` stays self-contained (embed the 120-byte Box.bin literal). Keep ALL DR1/DR2
  assertions green.
- Branch `fix-draco-dr3`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target draco_test'`
  (if it swallows output, run via the PowerShell tool natively). Run `draco_test`, confirm ALL (DR1–DR3)
  PASS, exit 0. ALSO local clang standalone, identical digests. First run: pin the connectivity digest.
- **This is the hardest slice.** If the Box does NOT decode to 12 faces, DEBUG carefully against the spec
  (the corner-table Next/Previous, the C/S/L/R/E switch, the valence symbol mapping, the bit-reader
  endianness are the usual culprits). Report HONESTLY if you cannot get faces==12 — do NOT fake the golden;
  report exactly where the decode diverges (e.g. "decodes N faces, the symbol stream is X, the issue is Y").
  A partial honest result ("connectivity parses, traversal runs, but produces N!=12 faces because Z") is more
  valuable than a faked pass.
- COMPLETION CRITERIA — commit when: the header compiles, `draco_test` builds + PASSES on Windows with
  faces==12 + the pinned digest + DR1/DR2 unchanged, and local clang matches. Report: commit hash, full test
  output, the pinned digest + the decoded faces (so I can verify a valid cube), `edgebreaker_traversal_type`
  the Box uses, confirmation DR1/DR2 unchanged + self-contained, the local-clang result, and any deviation /
  honest blocker. Commit message via temp file + `git commit -F`. Commit to the branch and STOP.
  (The CONTROLLER audits, Mac/clang-proves faces==12 + the digest, ff-merges + advances to DR4 — the
  attribute decode that turns the connectivity into the cube's 24 positions.)
