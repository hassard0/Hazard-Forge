# Slice METAL-RT S3 — native Metal HW RT mirror reflections (issue #42/#35, RT4 parity)

S1 landed the Metal RHI accel seam; S2 delivered native Metal HW hard shadows (`--rt3-shadow` byte-equal
to the CPU reference). S3 delivers the next capability gain that closes the #42 overclaim: **native Metal
HW mirror reflections.** Today `--rt4-reflect` on Metal runs the CPU `rtrace::RenderSceneReflected`
reference (the Vulkan `rt_reflect.comp` HLSL+int64 RayQuery can't lower to MSL — `visual_test.mm:25352`).
S3 hand-authors `shaders/rt_reflect.metal` (the proven RT3 shadow kernel + a one-bounce mirror reflection
ray + the integer channel blend) and runs it on the HW accel through the S1 seam, byte-equal to the CPU
reference.

Determinism is preserved exactly as S1/S2: float HW BVH = candidate generator (margin-inflated AABBs),
kernel never commits/narrows, ALL correctness is the integer fx math (primary closest-hit, shadow
occlusion, the reflected-ray closest-hit + its shadowed shade, the channel blend) — so the golden stays
**strict memcmp byte-equal** to the CPU `rtrace::RenderSceneReflected`.

## Execution model — same as S1/S2
Implementer WRITES the MSL kernel + the showcase HW branch on Windows (uncompilable here), commits the
draft, STOPS. Controller drives the Mac round-trips. NOTE (confirmed in S2): `rt_*.metal` is RUNTIME-compiled
(`LoadText` from `HF_SHADER_DIR`, no CMake staging), so MSL errors surface when `--rt4-reflect` RUNS, and
MSL fixes iterate by scp-the-`.metal` + rerun (only the C++ `visual_test.mm` needs a rebuild).

## Sources to lift from (READ these)
- `shaders/rt_shadow.metal` (S2, just merged, 307 lines) — the native MSL base with the primary closest-hit
  + the shadow-ray any-hit drain-and-OR + the integer Lambert×shadow shade + the fx helpers (`long`).
  `rt_reflect.metal` REUSES all of this (the "shadowed primary color" is RT3's exact output), then ADDS the
  reflection bounce.
- `shaders/rt_reflect.comp.hlsl` (365 lines) — the PROVEN Vulkan reflection kernel (HW==CPU on Vulkan):
  primary closest-hit → the RT3 shadowed primary color `cS` → if the surface is REFLECTIVE, build the mirror
  reflection ray (reflect the view dir about the normal), trace it (closest-hit), compute the reflected
  hit's shadowed shade, and BLEND it into `cS` (the integer per-channel blend). **Translate its integer fx
  math to MSL `long` verbatim** (the same translation S2 did from rt_shadow.comp). This is the authority for
  the reflect-ray construction, the reflectivity test, and the exact blend arithmetic.
- `engine/render/rtrace.h:523` `RenderSceneReflected` — the CPU reference the HW must memcmp-equal (FROZEN
  oracle). The blend math in `rt_reflect.comp` already mirrors it; copy faithfully.
- `metal_headless/visual_test.mm:25360` `RunRt4ReflectShowcase` — the existing showcase (CPU
  `rtrace::RenderSceneReflected` @ `:25405`). S3 ADDS the HW path through the S1 seam + the memcmp, exactly
  as S2 added it to `RunRt3ShadowShowcase` (use the S2 HW branch as the template — same `device->CreateBlas`/
  `CreateTlas` + `cmd->BindAccelStructure` + dispatch + readback + memcmp scaffold; `HwPrim`/`HwParams` are
  now shared file-scope structs ahead of the RT showcases).

## What to build
1. **NEW `shaders/rt_reflect.metal`** — native MSL `kernel` = rt_shadow.metal's primary+shadow+shade, PLUS:
   for a primary hit whose surface is REFLECTIVE (the `rt_reflect.comp` reflectivity test), build the mirror
   reflection ray (`reflect(dir, normal)` in integer fx), run the primary closest-hit AGAIN for the reflected
   ray (reuse the same `intersection_query` drain + `(t, primIndex)` reduction), compute the reflected hit's
   shadowed shade (reuse the shadow path), and blend per `rt_reflect.comp`'s integer channel blend into the
   primary shaded color → RGBA8 written EXACTLY as `RenderSceneReflected`. All int64 fx via `long`. Same
   buffer bindings as rt_shadow.metal (accel @ buffer(3), gPrims, params/image). Header comment mirroring
   rt_shadow.metal.
   - Note: a third `intersection_query<>` instance for the reflected primary (after the primary `q` + the
     shadow `sq`) — S2 proved two compile; if three is a problem on Metal, reuse/reset one (the controller
     resolves on the Mac). Flag it.
2. **`RunRt4ReflectShowcase` (visual_test.mm) — ADD the HW branch** (clone the S2 `RunRt3ShadowShowcase` HW
   branch): keep the CPU `rtrace::RenderSceneReflected` as the oracle; when `device->SupportsHardwareRayQuery()`:
   build accel via the seam, bind, load+dispatch `rt_reflect.metal`, readback, `memcmp(hw, cpuReflected) == 0`
   + two-run byte-identical; print `rt4-reflect: ... Metal-HW(via RHI seam)==CPU rtrace::RenderSceneReflected
   BYTE-EQUAL`. Non-RT device → existing CPU-only behavior.

## Proof / golden
- `--rt4-reflect` on the M4: `SupportsHardwareRayQuery()` true → HW; `memcmp(HW, rtrace::RenderSceneReflected)
  == 0` byte-equal; two runs byte-identical; matches committed `tests/golden/metal/rt4_reflect.png` DIFF
  0.0000. Strict (integer-reconciled). No NEW golden (reuses `rt4_reflect.png`).

## Constraints (HARD)
- NEW `shaders/rt_reflect.metal` + ADDITIVE HW branch in `RunRt4ReflectShowcase` only. Reuse the S1 seam +
  the S2 shadow logic. Do NOT touch `rtrace.h` (frozen oracle), the Vulkan backend, `rt_query.metal`/
  `rt_shadow.metal` (leave verbatim — copy what you need into rt_reflect.metal), `rhi.h`, or any existing
  golden. Existing Metal goldens stay byte-identical (only a new shader + a HW branch).
- The integer reflect + blend MUST be byte-equal to `RenderSceneReflected` — copy `rt_reflect.comp`'s fx
  arithmetic verbatim. Float is ONLY the candidate-widening traversal rays (never read for correctness).
- Branch `fix-metalrt-s3`. Commit the draft (can't compile on Windows — say so). Do NOT merge.
- COMPLETION (implementer): rt_reflect.metal + the showcase HW branch committed, accurately lifted. REPORT:
  commit hash, files changed, the exact lift points (reflect-ray construction + reflectivity test + blend
  arithmetic line refs from rt_reflect.comp), how you structured the reflected-ray trace (3rd query or reuse),
  and every spot you're unsure compiles on Metal. Do NOT claim it builds.
  (CONTROLLER then: push → Mac build → run `--rt4-reflect` → iterate MSL fixes (scp .metal + rerun) →
  confirm memcmp(HW,CPU)==0 + two-run 0.0000 + matches `rt4_reflect.png` → confirm existing Metal goldens
  unaffected → ff-merge → advance to RT6 hero capstone + cross-vendor closure to CLOSE #42/#35.)
