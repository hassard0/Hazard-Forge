# Slice METAL-RT S2 — native Metal hardware RT hard shadows (issue #42/#35, RT3 parity)

S1 landed the RHI acceleration-structure seam on Metal (`CreateBlas`/`CreateTlas`/`BindAccelStructure`/
`SupportsHardwareRayQuery` → true on the M4) and `--rt2-query-rhi` proved a native Metal HW ray query is
byte-equal to the CPU integer reference. S2 delivers the first real CAPABILITY gain that closes the #42
overclaim: **native Metal HW hard shadows.** Today `--rt3-shadow` on Metal runs the CPU
`rtrace::RenderSceneShadowed` reference (the Vulkan `rt_shadow.comp` HLSL+int64 RayQuery can't lower to
MSL — `visual_test.mm:25112-25114`). S2 hand-authors a native MSL shadow kernel (the `rt_query.metal`
pattern, int64 via MSL `long`) and wires the Metal showcase to run it on the HW accel-structure through
the S1 seam, byte-equal to the CPU reference.

Determinism is preserved exactly as S1/RT2: the float HW BVH is a candidate generator (margin-inflated
AABBs), the kernel never commits/narrows, and ALL correctness (primary closest-hit, shadow occlusion,
Lambert shade) is the integer fx math — so the golden stays **strict memcmp byte-equal** to the CPU
`rtrace::RenderSceneShadowed`.

## Execution model (Metal compiles ONLY on the M4)
Same as S1: the implementer WRITES the MSL kernel + the showcase wiring on Windows (uncompilable here),
commits the draft, STOPS. The CONTROLLER drives the Mac compile/test round-trips + byte-equal validation.

## Sources to lift from (READ these)
- `shaders/rt_query.metal` (239 lines) — the native MSL base: the `intersection_query<>` primary closest-hit
  that DRAINs every candidate AABB (`:220-236`) and does the integer `(t, primIndex)` total-order reduction.
  rt_shadow.metal REUSES this primary-ray body verbatim, then ADDS the shadow ray + shade.
- `shaders/rt_shadow.comp.hlsl` (303 lines) — the PROVEN Vulkan shadow kernel (HW==CPU on Vulkan): primary
  RayQuery closest-hit → build a shadow ray toward the light → secondary RayQuery (any-hit occlusion) →
  integer Lambert diffuse with the shadow factor. **Translate its integer fx math to MSL `long` verbatim**
  (the same HLSL-int64 → MSL-`long` translation `rt_query.metal` already did for the primary ray). This is
  the authority for the shadow-ray construction + the exact shade arithmetic.
- `engine/render/rtrace.h:454` `RenderSceneShadowed` — the CPU reference the HW must memcmp-equal (frozen;
  the byte-exact oracle). The shade math in rt_shadow.comp already mirrors this; copy it faithfully.
- `metal_headless/visual_test.mm:25120` `RunRt3ShadowShowcase` — the existing Metal showcase that currently
  computes ONLY the CPU `rtrace::RenderSceneShadowed` (`:25165`). S2 ADDS the HW path through the S1 seam +
  the memcmp.
- S1's proven `--rt2-query-rhi` showcase in `visual_test.mm` (the new `RunRt2QueryRhiShowcase`) — the
  template for building accel via `device->CreateBlas`/`CreateTlas`, binding via `cmd->BindAccelStructure`,
  dispatching, and reading back. RT3 clones this accel/dispatch scaffold and swaps the kernel + adds the
  light/shadow params + the CPU-shadowed oracle.

## What to build
1. **NEW `shaders/rt_shadow.metal`** — a native MSL `kernel` mirroring `rt_query.metal`'s structure +
   `rt_shadow.comp`'s shadow+shade:
   - Same buffers as `rt_query.metal` (the `primitive_acceleration_structure accel [[buffer(3)]]`, the
     `gPrims[]` info, the params/image buffers) PLUS the light/scene params `rt_shadow.comp` reads (light
     position/dir, the surface/material constants — match its cbuffer layout, packed into a params buffer).
   - Per pixel: (a) the `rt_query.metal` primary closest-hit (integer `(t, primIndex)` reduction); (b) if a
     primary hit, reconstruct the integer hit point + normal (as `rt_shadow.comp`/`rtrace.h` do), build the
     shadow ray toward the light, run a SECOND `intersection_query` draining candidates as an **any-hit
     boolean OR** (the scout: shadow occlusion is order-independent — no commit, OR the candidates that the
     fx test confirms occlude); (c) the integer Lambert-diffuse shade × the shadow factor → RGBA8, written
     EXACTLY as `rt_shadow.comp`/`RenderSceneShadowed` (byte-equal is the bar). Use `long` for all int64 fx
     (fxmul/fxdiv/fxisqrt — the `rt_query.metal:53-66` helpers; reuse them).
   - Header comment like `rt_query.metal`: native MSL, int64 via `long`, MTLLanguageVersion2_4, NO HLSL/SPIR-V.
2. **`RunRt3ShadowShowcase` (visual_test.mm) — wire the HW path:** keep computing the CPU
   `rtrace::RenderSceneShadowed` as the oracle, but when `device->SupportsHardwareRayQuery()` (true on M4):
   build the accel via `device->CreateBlas`/`CreateTlas` (the SAME RT2 scene's primitives), bind via
   `cmd->BindAccelStructure`, load + dispatch `rt_shadow.metal`, read back the HW image, and
   `memcmp(hwImage, cpuShadowed) == 0` + two-run byte-identical. Print the RT2-RHI-style proof lines
   (`rt3-shadow: ... Metal-HW(via RHI seam)==CPU rtrace::RenderSceneShadowed BYTE-EQUAL`). On a non-RT
   device, keep the existing CPU-only behavior (graceful fallback). The OUTPUT image is the (now HW, byte-
   equal) shadowed render → still matches the committed `rt3_shadow` golden.
3. **CMake / shader wiring** — ensure `rt_shadow.metal` is found by the showcase's `loadMSL` (mirror how
   `rt_query.metal` is located/loaded; metal_headless copies/compiles the `.metal` files — match that).

## Proof / golden
- `--rt3-shadow` on the M4: `SupportsHardwareRayQuery()` true → HW path; `memcmp(HW, rtrace::RenderSceneShadowed)
  == 0` byte-equal; two runs byte-identical; the rendered image matches the committed
  `tests/golden/metal/rt3_shadow.png` DIFF 0.0000 (same scene — the HW now produces what the CPU ref did,
  byte-for-byte). Strict (an integer-reconciled slice, NOT float visresolve).
- No NEW golden (reuses `rt3_shadow.png`); if the scene/name differs, the controller bakes/registers.

## Constraints (HARD)
- NEW `shaders/rt_shadow.metal` + an ADDITIVE HW branch in `RunRt3ShadowShowcase`. Reuse the S1 seam
  (`CreateBlas`/`CreateTlas`/`BindAccelStructure`) — do NOT add new RHI. Do NOT touch `rtrace.h` (frozen,
  the oracle), the Vulkan backend, `rt_query.metal` (leave verbatim — copy its helpers into rt_shadow.metal
  or share), or any existing golden. Existing Metal goldens stay byte-identical (the change only adds the HW
  branch + a new shader).
- The integer shade MUST be byte-equal to `rtrace::RenderSceneShadowed` — copy the fx arithmetic from
  `rt_shadow.comp` verbatim (it already proves HW==CPU on Vulkan). Any float is ONLY the candidate-widening
  traversal ray (never read for correctness).
- Branch `fix-metalrt-s2`. Commit the draft (can't compile on Windows — say so). Do NOT merge. Commit via
  temp file + `git commit -F`.
- COMPLETION (implementer): rt_shadow.metal + the showcase HW branch written + committed, accurately lifted
  from rt_query.metal + rt_shadow.comp + rtrace.h. REPORT: commit hash, files changed, the exact lift points
  (the shadow-ray construction + the shade arithmetic line refs from rt_shadow.comp), how you structured the
  shadow any-hit drain-and-OR, the params-buffer layout you chose (must match what the showcase uploads),
  and every spot you're unsure compiles on Metal (for the controller's Mac pass). Do NOT claim it builds.
  (The CONTROLLER then: push → Mac build → iterate MSL fixes → run `--rt3-shadow` → confirm
  `memcmp(HW, CPU)==0` + two-run 0.0000 + matches `rt3_shadow.png` → confirm existing Metal goldens
  unaffected → ff-merge → advance to RT4 reflections.)
