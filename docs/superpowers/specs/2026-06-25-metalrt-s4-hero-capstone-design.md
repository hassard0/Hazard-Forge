# Slice METAL-RT S4 — RT6 lit hero capstone + cross-vendor closure (issues #42/#35, CLOSE)

The capstone of real Metal hardware ray tracing in the RHI. S1 landed the Metal accel seam; S2/S3 made
`--rt3-shadow` and `--rt4-reflect` real HW (byte-equal to the CPU reference). S4 delivers the money-shot —
the curated lit hero scene — as native Metal HW RT, then CLOSES the flagship: every Metal HW RT golden is
byte-identical to the CPU integer reference (and therefore to the Vulkan HW image), and CAPABILITIES.md is
updated to retire the "Metal-only standalone, NOT wired into engine/rhi_metal/" caveat — resolving the #42
overclaim and #35.

The hero kernel is the RT4 reflection kernel with ONE change: the miss color (the primary miss AND every
reflected/secondary miss) uses a **sky gradient** instead of a flat background — per `rtrace.h`:
"RenderSceneHero — IDENTICAL to RenderSceneReflected EXCEPT every miss color uses the sky gradient." Same
determinism: integer fx owns all correctness, the golden stays strict memcmp byte-equal.

## Execution model — same as S1/S2/S3
Implementer WRITES `shaders/rt_hero.metal` + the showcase HW branch on Windows (uncompilable here), commits
the draft, STOPS. Controller drives the Mac round-trip (runtime-compiled `.metal` → fast scp+rerun iterate).
This pattern has compiled+run FIRST TRY for S2 and S3 — RT6 is the same shape.

## Sources to lift from (READ these)
- `shaders/rt_reflect.metal` (S3, just merged, 383 lines) — the native MSL base: primary + shadow +
  reflection + blend, with the `TraceClosestHw`/`TraceAnyHitHw` helpers (function-local
  `intersection_query<>`). `rt_hero.metal` is this kernel with the sky-gradient miss substituted.
- `shaders/rt_hero.comp.hlsl` — the PROVEN Vulkan hero kernel (HW==CPU on Vulkan). The AUTHORITY for the
  **sky-gradient miss color function** (a function of the ray direction) — translate it to MSL `long`
  verbatim. Everything else equals `rt_reflect.comp`.
- `engine/render/rtrace.h:626` `RenderSceneHero` — the CPU reference the HW must memcmp-equal (FROZEN
  oracle). The sky-gradient miss math in `rt_hero.comp` mirrors it; copy faithfully.
- `metal_headless/visual_test.mm:25676` `RunRt6HeroShowcase` — the existing showcase (CPU
  `rtrace::RenderSceneHero` @ ~`:25727`). ADD the HW path through the S1 seam + memcmp, exactly as S2/S3
  added it (clone the S3 `RunRt4ReflectShowcase` HW branch; `HwPrim`/`HwParams` are shared file-scope structs
  ahead of the RT showcases).

## What to build
1. **NEW `shaders/rt_hero.metal`** — native MSL `kernel` = `rt_reflect.metal` VERBATIM, with the miss color
   changed from "read the flat background param" to the **sky-gradient function** from `rt_hero.comp.hlsl`
   (a deterministic integer function of the ray direction y / screen position — copy its exact fx
   arithmetic). The sky gradient applies to BOTH the primary miss AND any reflected-ray miss (per
   `RenderSceneHero`). All int64 fx via `long`. Same buffer bindings as `rt_reflect.metal`. Header comment
   mirroring it.
2. **`RunRt6HeroShowcase` (visual_test.mm) — ADD the HW branch** (clone the S3 HW branch): keep the CPU
   `rtrace::RenderSceneHero` as the oracle; when `device->SupportsHardwareRayQuery()`: build the hero scene's
   accel via the seam, bind, load+dispatch `rt_hero.metal`, readback, `memcmp(hw, cpuHero) == 0` + two-run
   byte-identical; print `rt6-hero: ... Metal-HW(via RHI seam)==CPU rtrace::RenderSceneHero BYTE-EQUAL`.
   Non-RT device → existing CPU-only behavior.

## Proof / golden
- `--rt6-hero` on the M4: `SupportsHardwareRayQuery()` true → HW; `memcmp(HW, rtrace::RenderSceneHero) == 0`
  byte-equal; two runs byte-identical; matches committed `tests/golden/metal/rt6_hero.png` DIFF 0.0000.
  Strict. No NEW golden (reuses `rt6_hero.png`).

## Constraints (HARD)
- NEW `shaders/rt_hero.metal` + ADDITIVE HW branch in `RunRt6HeroShowcase` only. Reuse the S1 seam + S3
  reflection logic. Do NOT touch `rtrace.h` (frozen), the Vulkan backend, the other `rt_*.metal` (leave
  verbatim), `rhi.h`, or any golden. Existing Metal goldens stay byte-identical.
- The integer sky-gradient miss MUST be byte-equal to `RenderSceneHero` — copy `rt_hero.comp`'s fx arithmetic
  verbatim.
- Branch `fix-metalrt-s4`. Commit the draft (can't compile on Windows). Do NOT commit the `mat_*.frag.hlsl`
  CRLF churn. Do NOT merge.
- COMPLETION (implementer): rt_hero.metal + the showcase HW branch committed, lifted accurately. REPORT:
  commit hash, files, the exact sky-gradient-miss line refs from `rt_hero.comp`, and any Metal-compile
  uncertainties. Do NOT claim it builds.

## CONTROLLER close-out (after RT6 merges) — this is what CLOSES #42/#35
1. Mac: run `--rt6-hero` → confirm `memcmp(HW,CPU)==0` + two-run 0.0000 + matches `rt6_hero.png`. ff-merge.
2. Re-run the full Metal HW RT set on the M4 through the seam (`--rt2-query-rhi`, `--rt3-shadow`,
   `--rt4-reflect`, `--rt6-hero`) and confirm each is byte-equal to its CPU reference + matches its committed
   golden (the cross-vendor closure: Metal HW == CPU == Vulkan HW, by construction). Spot-confirm a couple
   existing NON-RT Metal goldens are unaffected by the cumulative changes.
3. **Update `docs/CAPABILITIES.md`** — find the RT entries that say the Metal `MTLAccelerationStructure` +
   `intersection_query` path is "proven standalone... NOT wired into engine/rhi_metal/" and that Metal RT
   shadows/reflections run the CPU reference. Replace with the new TRUTH: `engine/rhi_metal/` now implements
   the `IAccelStructure` seam (`CreateBlas`/`CreateTlas`/`BindAccelStructure`/`SupportsHardwareRayQuery`→true
   on Apple-silicon HW); `--rt2-query-rhi`/`--rt3-shadow`/`--rt4-reflect`/`--rt6-hero` run REAL Metal HW ray
   tracing through the RHI seam, byte-equal to the CPU integer reference (determinism preserved: float BVH =
   candidate generator, integer fx owns correctness). Keep an honest caveat: the true two-level instance-AS
   is a degenerate single-instance TLAS in v1 (multi-instance transformed scenes are future), and
   fragment-stage RT (the graphics-pipeline accel binding on Metal) is not yet wired.
4. **Update ARCHITECTURE.md** if it has a Metal-RT caveat mirroring the above.
5. **Close #42 and #35** with the evidence: `engine/rhi_metal/metal_accel.*` + the seam overrides;
   `SupportsHardwareRayQuery()` true on the M4; `--rt2-query-rhi`/`--rt3-shadow`/`--rt4-reflect`/`--rt6-hero`
   all byte-equal HW; the int64→MSL blocker sidestepped by native `intersection_query` + MSL `long`. Note the
   honest v1 scope (degenerate TLAS, no fragment-stage RT yet).
