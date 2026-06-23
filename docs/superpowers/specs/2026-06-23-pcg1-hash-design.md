# Slice PCG1 — Seeded hash-PRNG primitive (Issue #22, flagship #22 DETERMINISTIC PCG, 1st/beachhead slice)

The irreducible primitive every later PCG slice consumes: a **deterministic seeded integer hash-PRNG** in Q16.16,
pure int32 (MSL-native), provable with a **STRICT zero-differing-pixel cross-backend golden** (NOT a float
visresolve-bar — this is an integer slice like the sim flagships' integer slices). Establishes the new
`engine/pcg/pcg.h` header + `hf::pcg` namespace. NO GPU compute shader (PCG is a CPU-side host generator that
will later feed the existing instanced render — by design).

## The header — `engine/pcg/pcg.h` (NEW, header-only, namespace `hf::pcg`)
Reuse `engine/sim/fpx.h` read-only for `fx`/`kOne`/`kFrac`/`fxmul` and `FxVec3`. Copy the integer avalanche hash
SHAPE **verbatim** from `engine/sim/particles.h` `ParticleHash` (~line 144) and the no-trig direction table from
`particles.h` `EmitDir` (~line 158) — read those two functions and mirror their exact integer ops (do NOT
improvise a new hash; reuse the proven one so the determinism story is identical). Provide:
- `uint32_t PcgHash(uint32_t seed, uint32_t index)` — the ParticleHash avalanche over `(seed, index)`.
- `fx PcgRand01(uint32_t seed, uint32_t index)` — a Q16.16 value in `[0, kOne)`, taken as the **top 16 bits** of
  the hash shifted into the fractional range (a pure shift/mask — NO division, NO float). Must be `< kOne`.
- `fx PcgRandRange(uint32_t seed, uint32_t index, fx lo, fx hi)` — `lo + fxmul(PcgRand01(...), hi - lo)`.
- `FxVec3 PcgUnitDir(uint32_t seed, uint32_t index)` — index the `EmitDir` integer direction table by the hash
  (a deterministic ~unit direction; reuse EmitDir's table/logic verbatim — no trig).
- `struct PcgStream { uint32_t seed; uint32_t salt; }` + helpers that fold `salt` into the index so distinct
  layers (the future scatter/mask/transform stages) hash to distinct streams (the `emitterId` salt pattern from
  particles). e.g. `PcgRand01(stream, index)` mixes `salt` into the hash input.
Keep it ALL int32 — this slice must be MSL-native (no int64 fxmul). NO new RHI, NO shader.

## CPU test — `tests/pcg_test.cpp` (register in tests/CMakeLists.txt next to particles_test/grain_test)
Assertions: (1) **replay-stable** — `PcgHash`/`PcgRand01`/`PcgUnitDir` return identical values for the same
`(seed,index)` across calls; (2) **seed-sensitive** — a different seed yields a different stream (sample many
indices, assert the sequences differ); (3) **range bounds** — `PcgRand01 ∈ [0, kOne)` over a swept index range
(0..N), `PcgRandRange ∈ [lo, hi]`; (4) **salt separation** — two `PcgStream`s with different salt give different
sequences for the same index; (5) **uniformity sanity** — bucket N=4096 `PcgRand01` samples into K bins and
assert every bin is within a loose band of N/K (deterministic uniformity — a fixed integer assertion, no float
tolerance flakiness). Follow particles_test's structure.

## Showcase — `--pcg1-hash-shot` (Vulkan, main.cpp) + `--pcg1-hash` (Metal, visual_test.mm)
A **2D top-down strict-integer point plot**: take the first N (e.g. 4096) samples `p_i = (PcgRand01(stream, 2i),
PcgRand01(stream, 2i+1))` as a point in the unit square, map each to an INTEGER pixel coord in the output image
(`px = (rand >> someShift)` scaled to image WxH by pure integer math — NO float rounding that could differ
cross-vendor), and write a marker pixel (a small integer plus/dot) into an RGBA8 buffer, then save. The result is
a deterministic speckle field. **This is a CPU-rendered 2D image (no GPU pass needed) — write the pixels directly
like other strict-integer debug-viz showcases; OR if you route it through the blit/present path, keep the buffer
fill pure-integer.** SAME seed/N/stream/image-size IN BOTH renderers (main.cpp and visual_test.mm) — the image
must be **byte-identical cross-backend by construction** (pure integer math, no GPU float raster).

## Proof (STRICT integer — zero-diff cross-backend, the strongest bar)
```
pcg1-hash: seeded hash-PRNG (N=4096 samples, seed=<S>)
pcg1-hash: two-run BYTE-IDENTICAL
pcg1-hash: different seed -> different field {seedA_hash:<hA>, seedB_hash:<hB>} hA != hB (same N, both valid)
pcg1-hash: provenance {samples:4096, range:[0,kOne), uniform:true}
```
Assertions: (1) two runs byte-identical; (2) a different seed produces a DIFFERENT speckle (hash/checksum of the
two images differs) but the same sample count + all-in-range (valid); (3) provenance + coherent (non-empty, the
field has the expected sample count). Register `pcg1_hash` in verify.ps1 $Goldens (Flag `--pcg1-hash`) +
`--pcg1-hash-shot` in $vkShots, mirroring how `pt6_render`/`grain_render` are registered.

## Constraints (HARD)
- NEW engine/pcg/pcg.h + tests/pcg_test.cpp ONLY (plus the showcase blocks in main.cpp/visual_test.mm + the
  verify.ps1 registration). Reuse fpx.h/particles.h READ-ONLY — do NOT modify them or any existing header/shader/
  golden. NO new RHI, NO new shader (pure CPU integer generation + a CPU-filled 2D image). All int32 (MSL-native).
- This is a STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (the controller bakes
  Metal and confirms the Metal image == the Vulkan image byte-for-byte / zero-diff, NOT a float visresolve mean).
  So the 2D plot MUST be pure integer (no float pixel math that diverges cross-vendor).
- Branch `fix-issue-22-pcg1`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows (`cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle'`
  via the PowerShell tool, single-quoting the cmd arg so PowerShell doesn't choke on the (x86) parens; under Git
  Bash the vcvars silently no-ops). Also build + run the test target if there's a tests build preset.
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--pcg1-hash-shot` exits 0, the proof lines
  print, two-run byte-identical, different-seed-different verified, AND pcg_test passes. Commit to the branch and
  STOP. Report: the commit hash, the proof output, the image path, confirmation the showcase uses identical
  seed/N/stream/size in both renderers, and that the plot is pure-integer (so it'll be zero-diff cross-backend).
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop (do not extend the giant else-if chain).
