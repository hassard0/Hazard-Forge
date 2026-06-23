# Slice PCG3 — Density mask + importance rejection (Issue #22, flagship #22 DETERMINISTIC PCG, 3rd slice)

The control layer: an analytic **density mask** (a scalar field in Q16.16) that the scatter follows — a point
survives in proportion to the mask value at its position (importance rejection), so density varies across the
area (a falloff disc, a half-plane, etc.) instead of a uniform grid. Builds on PCG2's `ScatterGrid`. This slice
introduces the first **int64** path (the radial mask's `FxLength`), but because PCG is a CPU-side host generator
(NO GPU shader), it runs CPU-on-both-backends → still **byte-identical cross-backend** (strict zero-diff golden),
the documented `fpx-solve` boundary but simpler (no shader at all). NOT a float visresolve-bar slice.

## The addition — `engine/pcg/pcg.h` (APPEND-ONLY after the PCG2 block)
Add to `hf::pcg` (do NOT modify PCG1/PCG2 — append after `ScatterGrid`):
- An analytic mask. A small enum + struct:
  ```
  enum class PcgMaskType { Uniform, Radial, HalfPlane };
  struct PcgMask {
      PcgMaskType type = PcgMaskType::Uniform;
      FxVec3 center;     // Radial: disc centre (XZ); HalfPlane: a point on the plane
      fx     radius = 0; // Radial: falloff radius
      FxVec3 axis;       // HalfPlane: the inward normal (XZ)
      fx Eval(const FxVec3& p) const;   // returns a Q16.16 weight in [0, kOne]
  };
  ```
  `Eval` semantics (all return a value clamped to `[0, kOne]`):
  - **Uniform** → always `kOne` (the no-op control — every point passes).
  - **Radial** → `clamp(kOne - fxdiv(dist, radius), 0, kOne)` where `dist = FxLength((p - center) with Y zeroed)`
    (XZ distance; **`FxLength` is the int64 path** — reuse `fpx.h`'s `FxLength`/`FxISqrt` verbatim). 1 at the
    centre, falling linearly to 0 at `radius`, 0 beyond. `radius <= 0` → 0 everywhere (a degenerate/zero mask).
  - **HalfPlane** → `kOne` on the inward side of the plane through `center` with normal `axis`, `0` on the other
    (a hard 0/kOne step via the sign of the integer dot product `FxDot(p - center, axis)`). (Keep this simple; it
    is a secondary mask — the radial one drives the showcase.)
- `std::vector<FxVec3> ScatterMasked(const PcgStream& stream, const PcgArea& area, int cellsX, int cellsZ, const PcgMask& mask, fx density)`:
  generate the PCG2 grid (call `ScatterGrid` internally so the candidate positions are IDENTICAL to PCG2), then
  KEEP each candidate (in the same fixed cell order) iff `fxmul(mask.Eval(p), density) > PcgRand01(keepStream, idx)`,
  where `keepStream` is `stream` with a **distinct salt** (e.g. `stream.salt ^ 0xA11C0DE`) so the keep-decision
  draw is independent of the jitter draw, and `idx` is the linear cell index. Return the surviving subset
  (variable count, fixed order). `density` is the global density knob: `density = kOne` + a Uniform mask keeps
  ALL points (since `fxmul(kOne,kOne) = kOne > PcgRand01 ∈ [0,kOne)` always → the no-op == `ScatterGrid`);
  `density = 0` keeps NONE. Keep the int boundary clean: only the radial `Eval` touches int64 (`FxLength`); the
  rest is int32.

## CPU test — extend `tests/pcg_test.cpp` (add a PCG3 section, keep PCG1/PCG2 checks)
Assertions: (1) **no-op == PCG2** — `ScatterMasked` with a `Uniform` mask and `density == kOne` returns a vector
**byte-equal to `ScatterGrid`** (same positions, same order — the make-or-break no-op proof); (2) **zero mask** —
a `Radial` mask with `radius <= 0` (or `density == 0`) keeps **0** points; (3) **radial monotone** — the kept
count is **non-decreasing in `density`** (sweep density 0 → kOne and assert the survivor count is monotone) AND
the radial disc keeps fewer than the full grid (a genuine subset for a radius smaller than the area); (4)
**replay-stable** — two calls with the same args are byte-equal; (5) **containment** — every surviving point is
still one of the `ScatterGrid` candidates (the mask only REMOVES, never moves). Print `pcg_test: ALL CHECKS PASSED`.

## Showcase — `--pcg3-mask-shot` (Vulkan, main.cpp) + `--pcg3-mask` (Metal, visual_test.mm)
A **2D top-down strict-integer plot of a radial-density disc of points**: the same fixed area / 48×48 cells /
seed/salt as PCG2, a `Radial` mask centred in the area with a radius ~⅓ of the area extent, `density = kOne`. The
result is a **disc of points dense at the centre, thinning to nothing at the rim** (vs PCG2's full square) — the
visible payoff. Map each surviving point's (x,z) to an INTEGER pixel coord by pure integer math (reuse PCG2's
pure-integer pixel/marker/checksum code VERBATIM — NO float pixel math). SAME area/cells/seed/salt/mask/density/
image-size IN BOTH renderers so the image is byte-identical cross-backend by construction.

## Proof (STRICT integer — zero-diff cross-backend)
```
pcg3-mask: radial density mask (cells=2304, radius=<R>, density=kOne, seed=<S>)
pcg3-mask: two-run BYTE-IDENTICAL
pcg3-mask: uniform mask + density=kOne == ScatterGrid BYTE-IDENTICAL (no-op control) {full:2304}
pcg3-mask: radial keeps a deterministic subset {kept:<K>, of:2304} K < 2304 AND monotone in density
pcg3-mask: provenance {maskType:radial, kept:<K>, zero-mask:0}
```
Assertions: (1) two runs byte-identical; (2) THE NO-OP — uniform mask + density=kOne is byte-identical to
`ScatterGrid` (PCG2); (3) the radial mask keeps a deterministic subset `K < 2304`, survivor count monotone
non-decreasing in density, zero-mask → 0; provenance coherent. Register `pcg3_mask` in verify.ps1 $Goldens (Flag
`--pcg3-mask`) + `--pcg3-mask-shot` in $vkShots, mirroring `pcg2_scatter`.

## Constraints (HARD)
- APPEND-ONLY to engine/pcg/pcg.h (do NOT modify PCG1/PCG2) + extend tests/pcg_test.cpp + the showcase blocks +
  verify.ps1 registration. Reuse fpx.h (incl. `FxLength`/`FxISqrt`/`FxDot`/`fxdiv`)/particles.h/PCG1/PCG2
  READ-ONLY. Do NOT modify any other header/shader/golden. NO new RHI, NO new shader. Only the radial `Eval`
  uses int64 (`FxLength`) — CPU-side, so still byte-identical cross-backend; everything else int32.
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (controller bakes Metal, requires
  Metal image == Vulkan image byte-for-byte). The 2D plot MUST be pure integer (no float pixel math). Do NOT route
  through a GPU float raster.
- Branch `fix-issue-22-pcg3`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows + pcg_test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle pcg_test'`
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--pcg3-mask-shot` exits 0, the proof lines print,
  two-run byte-identical, the uniform+density=kOne == ScatterGrid no-op holds, the radial subset + monotonicity +
  zero-mask verified, AND pcg_test passes. Commit message via a temp file + `git commit -F` (use the Bash tool
  heredoc; PowerShell here-strings break on `--flag` dashes). Commit to the branch and STOP. Report: commit hash,
  proof output, image path, confirmation both renderers use identical params, that pcg_test passes, and that the
  plot is pure-integer.
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
