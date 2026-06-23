# Slice PCG4 — Per-instance transform rules (Issue #22, flagship #22 DETERMINISTIC PCG, 4th slice)

The "rules" layer: turn the scattered POINTS into full INSTANCES — each gets a seed-stable random yaw + a random
uniform scale within designer ranges. This is what makes a procedural field look natural (rocks at varied angles
and sizes) instead of identical clones. Builds on PCG2/PCG3. Strict int32 (the yaw comes from a BAKED integer
quaternion table — NO runtime transcendentals, NO <cmath>), provable with a **STRICT zero-differing-pixel
cross-backend golden**. NOT a float visresolve-bar slice.

## Why a baked yaw table (NOT QFromAxisAngleSnapped)
`vehicle.h::QFromAxisAngleSnapped` is a small-angle Taylor series — it diverges for the full 0..2π yaw range PCG
needs. Instead, mirror `particles.h::EmitDir`'s pattern: a **baked table of host-constant Q16.16 quaternions**
(bit-identical cross-vendor by construction — they are stored integer literals, no runtime trig). Use a 16-entry
yaw table (a full turn in 22.5° steps). A yaw quaternion about +Y is `{x=0, y=sin(θ/2), z=0, w=cos(θ/2)}`.
**Use these EXACT literals (do NOT recompute them — they are pre-snapped Q16.16, each ≈unit):**
```
// kPcgYaw16[k] = yaw quaternion about +Y for θ = k * 22.5°, as fpx::FxQuat{x,y,z,w} (x=z=0).
static const fpx::FxQuat kPcgYaw16[16] = {
    {0,     0, 0, 65536},  {0, 12785, 0, 64277},  {0, 25080, 0, 60547},  {0, 36410, 0, 54491},
    {0, 46341, 0, 46341},  {0, 54491, 0, 36410},  {0, 60547, 0, 25080},  {0, 64277, 0, 12785},
    {0, 65536, 0,     0},  {0, 64277, 0,-12785},  {0, 60547, 0,-25080},  {0, 54491, 0,-36410},
    {0, 46341, 0,-46341},  {0, 36410, 0,-54491},  {0, 25080, 0,-60547},  {0, 12785, 0,-64277},
};
```
(Each entry's `y²+w² ≈ kOne²` to within ~0.0002% — unit to tolerance. Index it with the hash; `& 15` selects an
entry. `kPcgYaw16[0]` is identity.)

## The addition — `engine/pcg/pcg.h` (APPEND-ONLY after the PCG3 block)
Add to `hf::pcg` (do NOT modify PCG1/PCG2/PCG3):
- The `kPcgYaw16` table above (an anonymous-namespace or `inline` static const at file scope inside `hf::pcg`).
- `struct PcgInstance { FxVec3 pos; fpx::FxQuat orient; fx scale; };`
- `struct PcgTransform { bool randomYaw = true; fx scaleLo = kOne; fx scaleHi = kOne; };`
  (`randomYaw == false` → identity orientation; `scaleLo == scaleHi` → fixed scale — together the no-op control.)
- `std::vector<PcgInstance> BuildInstances(const std::vector<FxVec3>& points, const PcgStream& stream, const PcgTransform& rule)`:
  for each point `i` (fixed order): `orient = rule.randomYaw ? kPcgYaw16[PcgHash(yawStream, i) & 15] : fpx::FxQuat{0,0,0,kOne}`
  (identity), and `scale = PcgRandRange(scaleStream, i, rule.scaleLo, rule.scaleHi)`, where `yawStream`/`scaleStream`
  are `stream` with DISTINCT salts (e.g. `stream.salt ^ 0x4A09`, `stream.salt ^ 0x5CA1E`) so yaw and scale draw
  independent sequences (and independent of the PCG2 jitter / PCG3 keep draws). Emit `PcgInstance{point, orient,
  scale}`. Pure int32 (the table is integer literals; PcgRandRange's only mul is fxmul). NO float, NO trig.

## CPU test — extend `tests/pcg_test.cpp` (add a PCG4 section, keep PCG1–3 checks)
Assertions: (1) **count** — `BuildInstances` returns exactly `points.size()` instances; (2) **unit orient** —
every `orient` satisfies `|y² + w² - kOne²|` within a loose integer tolerance (the table is ≈unit — use a
tolerance like `kOne*kOne/1000`; x==z==0 exactly); (3) **scale in range** — every `scale ∈ [scaleLo, scaleHi]`;
(4) **no-op** — `rule{randomYaw=false, scaleLo=kOne, scaleHi=kOne}` → every instance has identity orient
(`{0,0,0,kOne}`) AND `scale == kOne` AND `pos` equal to the input points (the transform only annotates, never
moves); (5) **replay-stable** — two calls byte-equal; (6) **seed-sensitive** — a different `stream.seed` gives a
different yaw/scale sequence (same count). Print `pcg_test: ALL CHECKS PASSED`.

## Showcase — `--pcg4-rules-shot` (Vulkan, main.cpp) + `--pcg4-rules` (Metal, visual_test.mm)
A **2D top-down strict-integer plot of oriented, scaled markers**: take the PCG3 radial-disc scatter (same fixed
area / 48×48 cells / seed / radius / density), `BuildInstances` with `randomYaw=true`, `scaleLo=kOne/2`,
`scaleHi=kOne*3/2` (sizes 0.5×–1.5×). Draw each instance as a small **oriented cross/arrow**: define a few
integer arm-endpoint offsets (e.g. `(±armLen, 0, 0)` scaled by `instance.scale`), **rotate each by `instance.orient`
via `fpx::FxRotate` (pure-integer quaternion rotation)**, add to the instance's pixel center, floor to integer
pixels, and draw the arm lines/dots — so the marker visibly shows BOTH the yaw (arm direction) and the scale (arm
length). All pixel math pure-integer (reuse PCG3's pixel-center map + integer line/dot draw; NO float pixel math).
SAME area/cells/seed/rule/image-size IN BOTH renderers → byte-identical cross-backend by construction.

## Proof (STRICT integer — zero-diff cross-backend)
```
pcg4-rules: per-instance transform rules (instances=<K>, yaw:16-table, scale:[0.5,1.5], seed=<S>)
pcg4-rules: two-run BYTE-IDENTICAL
pcg4-rules: every orient unit AND every scale in range {instances:<K>, unit:<K>, inRange:<K>}
pcg4-rules: no-op rule (identity yaw + fixed scale) == input points unchanged {posMatch:<K>}
pcg4-rules: provenance {yawTable:16, scaleLo:32768, scaleHi:98304, instances:<K>}
```
Assertions: (1) two runs byte-identical; (2) all K orients unit-within-tol AND all K scales in range; (3) the
no-op rule yields identity orient + fixed scale with positions == the input points; (4) provenance coherent.
Register `pcg4_rules` in verify.ps1 $Goldens (Flag `--pcg4-rules`) + `--pcg4-rules-shot` in $vkShots, mirroring
`pcg3_mask`.

## Constraints (HARD)
- APPEND-ONLY to engine/pcg/pcg.h (do NOT modify PCG1/2/3) + extend tests/pcg_test.cpp + the showcase blocks +
  verify.ps1 registration. Reuse fpx.h (incl. `FxQuat`/`FxRotate`/`FxQuatNormalize`)/particles.h/PCG1–3
  READ-ONLY. Do NOT modify any other header/shader/golden. NO new RHI, NO new shader. Strict int32 — use the
  BAKED `kPcgYaw16` literals (do NOT call <cmath> or QFromAxisAngleSnapped; do NOT recompute the table).
- STRICT-INTEGER slice: the cross-backend golden must be **zero-differing-pixel** (controller bakes Metal, requires
  Metal image == Vulkan image byte-for-byte). The 2D plot MUST be pure integer — `FxRotate` is integer; floor to
  pixels with integer math. Do NOT route through a GPU float raster.
- Branch `fix-issue-22-pcg4`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows + pcg_test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target hello_triangle pcg_test'`
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, `--pcg4-rules-shot` exits 0, the proof lines
  print, two-run byte-identical, all orients unit + scales in range, the no-op rule holds, AND pcg_test passes.
  Commit message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit to the branch and STOP.
  Report: commit hash, proof output, image path, confirmation both renderers use identical params, that pcg_test
  passes, and that the plot is pure-integer (FxRotate + integer floor, no float pixel math).
- If main.cpp's arg-parse hits MSVC C1061, give the flag its OWN parse loop.
