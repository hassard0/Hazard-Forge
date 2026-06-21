# Slice MF3 — Hull Narrowphase Hardening: THE MULTI-POINT MANIFOLD GPU SHADER (the int64 GPU==CPU beat) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of
> FLAGSHIP #25 (DETERMINISTIC HULL NARROWPHASE HARDENING, `hf::sim::manifold`). MF1 built the polygon FACE
> topology; MF2 built the CPU multi-point manifold (`HullManifoldFromEpa` — the Sutherland–Hodgman face clip,
> proven a pure function, two runs byte-equal). MF3 lifts that manifold ONTO THE GPU: a compute shader generates
> the same 1-4 point `convex::ContactManifold` per overlapping pair, and the proof is that the GPU manifold is
> BYTE-IDENTICAL to the CPU `HullContactMulti` over a fixed pair battery. THE HEADLINE: the multi-point
> manifold's variable-length point set + the SH clip's integer tie-breaks reproduce EXACTLY on the device — the
> flagship's hardest determinism surface, proven on real GPU hardware. Because the manifold math is int64
> (`FxDot`/`FxCross`/`fxdiv` over Q16.16), the shader is **int64 → Vulkan-SPIR-V-ONLY** (DXC compiles int64;
> glslc/`hf_gen_msl` cannot, so it is NOT MSL-generated; the Metal `--mf3-manifold` runs the CPU
> `HullContactMulti` VERBATIM, byte-identical by construction — the GJ2/GJ3/CD1/CD3 hull-shader split). The
> DIRECT precedent is `shaders/convex_manifold.comp.hlsl` (the box `ContactManifold` GPU shader, the GPU twin of
> `convex::BuildManifold`) — MF3 is its hull generalization, exactly as MF2 generalized `convex::BuildManifold`
> in C++. APPEND to `engine/sim/manifold.h` (MF1/MF2 + gjk/broad/ccd/convex/fpx/etc BYTE-FROZEN). Branch:
> `slice-mf3`. See [[hazard-forge-manifold-roadmap]], [[hazard-forge-gjk-roadmap]],
> [[hazard-forge-gpu-tdr-chunking]], [[hazard-forge-metal-showcase-gate]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/manifold.h` (additive — MF1/MF2 byte-unchanged) with `HullContactMulti(bodyA, hullA,
bodyB, hullB)` — the HARDENED multi-point drop-in for `gjk::HullContact`: run the frozen `gjk::Gjk`→`gjk::Epa`
narrowphase, then `HullManifoldFromEpa` (MF2), returning the full `convex::ContactManifold` (count 1-4; separated
→ count 0). It is identical in spirit to MF2's `HullManifold` (it may simply BE that function, or a thin alias) —
the point of MF3 is the SHADER copies this body VERBATIM. Add `shaders/hull_manifold.comp.hlsl` (int64, the GPU
generator — one thread per pair, copies `HullContactMulti` verbatim). Add the showcase `--mf3-manifold-shot
<out>` (Vulkan: dispatch the shader over a pair battery, read back the manifolds, `memcmp` vs the CPU
`HullContactMulti`) / `--mf3-manifold` (Metal: run the CPU `HullContactMulti` — byte-identical by construction).
Bake the integer-render golden `mf3_manifold`. **NO new RHI** (reuse the existing compute dispatch + the
instanced-lit render pipeline); ONE new compute shader (int64 Vulkan-only).

## Design call: the shader copies `HullContactMulti` verbatim; the proof is GPU==CPU bit-identity

`convex_manifold.comp.hlsl` already proves the BOX manifold (`convex::BuildManifold`) bit-identical GPU==CPU; MF3
is its hull twin. The discipline (the GJ3/CD1/CD3 reality):
- **`HullContactMulti(bodyA, hullA, bodyB, hullB)` is the single function the shader mirrors.** It runs the FROZEN
  `gjk::Gjk` (gjk.h:457) → `gjk::Epa` (gjk.h:743) → MF2's `HullManifoldFromEpa` (manifold.h). Returns a
  `convex::ContactManifold`. The GPU shader (`hull_manifold.comp.hlsl`) copies this call chain VERBATIM (the GJK +
  EPA + clip already have proven HLSL forms — `gjk_distance.comp.hlsl`/`gjk_epa.comp.hlsl` for the narrowphase,
  `convex_manifold.comp.hlsl` for the clip idiom; MF3 assembles the hull-face-clip variant). Every op is the same
  int64 `FxDot`/`FxCross`/`fxdiv`/`fxmul`, the same FIXED orders, the same strict-integer tie-breaks → the
  manifold POD is byte-identical.
- **int64 → Vulkan-SPIR-V-only.** Register `shaders/hull_manifold.comp.hlsl:cs` in
  `samples/hello_triangle/CMakeLists.txt` (DXC → SPIR-V, `vulkan1.3`, int64-capable). Do NOT add it to
  `hf_gen_msl` in `metal_headless/CMakeLists.txt` (glslc/SPIRV-Cross can't lower int64 — the precedent:
  `hull_step.comp.hlsl`/`gjk_epa.comp.hlsl`/`ccd_step.comp.hlsl` are all Vulkan-only). The Metal `--mf3-manifold`
  runs the CPU `HullContactMulti` (the SAME C++ → byte-identical by construction; cross-vendor strict zero on the
  integer proof).
- **The GPU dispatch (Vulkan):** upload the pair battery (the body+hull pairs) to an SSBO, dispatch
  `hull_manifold.comp` (one thread per pair, the `convex_manifold.comp.hlsl` dispatch shape), read back the
  `ContactManifold` array, and `memcmp` it against the CPU `HullContactMulti` over the SAME battery. ANY 1-LSB
  divergence in the clip → a memcmp mismatch (the make-or-break proof). **TDR: the manifold dispatch is per-pair
  bounded work (GJK ≤ `kGjkMaxIter`, EPA ≤ `kEpaMaxIter`, the clip ≤ `kMaxClipVerts` against ≤ `kMaxFaceVerts`
  planes) — NO unbounded loop, a small battery → no TDR concern (unlike a heavy whole-world step); still, keep
  the battery modest and verify ~3 runs (NOT a 30x loop), the efficient-verification discipline.**

> The std430 `convex::ContactManifold` layout (convex.h:290-291: `count` uint32 + 4×`FxVec3` points + 4×`fx`
> depths + `FxVec3` normal) is ALREADY the GPU mirror `convex_manifold.comp.hlsl` uses — reuse it verbatim; do
> NOT invent a new GPU manifold struct.

## Reuse map (file:line)
- **MF1/MF2 `engine/sim/manifold.h` (APPEND after `HullManifold`):** `HullManifoldFromEpa`, `HullManifold`,
  `ClipFaceAgainstFace`, the MF1 face primitives. MF1/MF2 byte-frozen. `HullContactMulti` is the named hardened
  entry (the `gjk::HullContact` counterpart).
- **shaders (read-only precedents — REUSE the pattern):** `shaders/convex_manifold.comp.hlsl` (THE template — the
  box `ContactManifold` GPU generator + its SSBO layout + dispatch), `shaders/gjk_epa.comp.hlsl` /
  `gjk_distance.comp.hlsl` (the int64 GJK/EPA narrowphase in HLSL), `shaders/hull_step.comp.hlsl` (the int64
  Vulkan-only hull-shader registration precedent).
- **convex.h (read-only):** `convex::ContactManifold` (convex.h:292 — the std430 POD, GPU-mirrored verbatim),
  `convex::BuildManifold` (convex.h:352 — the box-clip the HLSL already mirrors).
- **gjk.h (read-only):** `gjk::Gjk`/`gjk::Epa`/`gjk::EpaResult`/`gjk::HullContact` (the single-point baseline the
  GPU twin replaces), `gjk::HullToRenderInstances` (the render base).
- **The GPU==CPU memcmp showcase precedent:** `--ccd-step-shot` / hull-step / `--gjk-*` (dispatch the int64
  compute, read back, `memcmp` the POD vs the CPU; the integer golden is identical both backends). Mirror for
  `--mf3-manifold-shot`.
- **Registration:** `samples/hello_triangle/CMakeLists.txt` (`shaders/hull_manifold.comp.hlsl:cs` — Vulkan),
  `scripts/verify.ps1` (append `mf3_manifold` to the Mac loop + `--mf3-manifold-shot` to `$vkShots`),
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), append to
  `tests/manifold_test.cpp`. **NO `hf_gen_msl` entry** (int64 → Metal CPU-ref).

## Design decisions (locked)
1. **APPEND to `engine/sim/manifold.h`** (MF1/MF2 byte-frozen): `HullContactMulti` (the hardened entry). **ONE new
   shader `shaders/hull_manifold.comp.hlsl`** (int64, Vulkan-only — DXC SPIR-V, NOT MSL). **NO new RHI** (reuse
   the existing compute-dispatch + readback seam + the instanced-lit render). gjk.h/convex.h/ALL other sim headers
   + ALL OTHER shaders BYTE-UNCHANGED.
2. **Showcase `--mf3-manifold-shot <out>` (Vulkan) AND `--mf3-manifold` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--mf3-manifold` BEFORE reporting DONE — the recurring omitted-Metal-showcase failure).**
   Vulkan: build a deterministic pair battery (box-on-box / tetra-on-face / edge-on-face — the MF2 scene),
   DISPATCH `hull_manifold.comp`, read back, `memcmp` vs CPU `HullContactMulti`; render the contact scene LIT 3D
   with the manifold points marked. Metal: run the CPU `HullContactMulti` (byte-identical), render the same.
   Golden = `tests/golden/metal/mf3_manifold.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1) GPU==CPU:** `mf3-manifold: {pairs:<P>, counts:[4,3,2]} GPU==CPU BIT-EXACT` — the GPU `ContactManifold`
     array `memcmp`-equals the CPU `HullContactMulti` array over the battery, byte-for-byte; assert.
   - **(2) determinism:** `mf3-manifold determinism: two runs BYTE-IDENTICAL` (the GPU readback is stable across
     ~3 runs — the efficient determinism check, NOT 30x).
   - **(3) counts + validity:** `mf3-manifold: {boxOnBox:4, tetraOnFace:3, edgeOnFace:2} depthsOK` — the GPU
     manifold reproduces MF2's counts {4,3,2} and every depth ≥ 0.
   - **(4) consistency:** `mf3-manifold normal: GPU == EPA normal` — the GPU manifold normal matches the EPA seed.
   - **Golden discipline: ONLY `tests/golden/metal/mf3_manifold.png`; do NOT commit it.** Existing 223 goldens
     UNTOUCHED.
4. **Cross-backend bar (int64 → strict on the integer proof).** Vulkan GPU==CPU bit-exact (~3× clean — bounded
   per-pair dispatch, TDR-safe). Metal CPU-ref byte-identical by construction. The integer manifold is strict-zero
   cross-vendor. The golden IMAGE is the render: committed = Mac-Metal bake; verify.ps1 re-renders + compares at
   `0.0000` (same-backend); cross-vendor visresolve is the in-band float DIAGNOSTIC (~20-55).
5. **Tests — APPEND to `tests/manifold_test.cpp` (CPU):** `HullContactMulti` over the battery returns counts
   {4,3,2}; matches `HullManifold`/`HullManifoldFromEpa` (the same result); separated pair → count 0; two-run
   byte-equal. (The GPU==CPU memcmp is the showcase's job — the test covers the CPU entry.) Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `manifold-gpu` (features) + `--mf3-manifold-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **ONE new compute shader, NO new RHI.** `shaders/hull_manifold.comp.hlsl` (int64, Vulkan-only — DXC SPIR-V;
  NOT in `hf_gen_msl`). It reuses the EXISTING compute-dispatch + SSBO-readback seam (the `convex_manifold.comp` /
  `hull_step.comp` path — no new pipeline type, no new descriptor layout beyond the standard compute SSBO set the
  existing manifold/step shaders already use). The render reuses the instanced-lit pipeline. `engine/sim/manifold.h`
  APPEND-only (MF1/MF2 frozen); gjk.h/convex.h/broad.h/ccd.h/fpx.h + ALL other sim headers + ALL OTHER shaders
  UNCHANGED. Report the seam: manifold.h APPEND-only + ONE new Vulkan-only shader; NO rhi.h change, NO MSL entry,
  NO frozen-file edit.

## Out of scope (YAGNI — later slices)
The full inertia tensor + the hardened restacked STEP (MF4 — MF3 produces the manifold but does NOT yet drive a
stepped world; the frozen `gjk::StepHullWorld` still uses the single-point `HullContact`). Lockstep (MF5), the
settled-stack capstone (MF6). True int32-MSL-native GPU-on-both (impossible for int64 — the documented glslc
limit; Metal runs the CPU-ref, the same proof strength every gjk/ccd hull beat carries). The area-maximizing
4-point reduction (inherited MF2 deferral). MF3 claims ONLY: the multi-point manifold is generated ON THE GPU
byte-identical to the CPU (the variable-length point set + the integer clip reproduce on-device), with the integer
golden + the four proofs.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "manifold|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--mf3-manifold-shot` on Vulkan: the 4 proof lines (GPU==CPU memcmp) + exit 0 under the
   conan validation layer → ZERO VUID. **~3 runs all GPU==CPU (bounded per-pair dispatch — TDR-safe). VERIFY the
   image shows the contact scene with the multi-point manifolds marked**, no garbage/NaN/iridescence.
3. Metal: `visual_test --mf3-manifold` → `tests/golden/metal/mf3_manifold.png`; two runs DIFF 0.0000. **Confirm
   `--mf3-manifold` is wired in `visual_test.mm` (grep it) BEFORE the Mac bake; confirm NO `hf_gen_msl` entry was
   added (int64 → Metal CPU-ref).** Cross-vendor = FLOAT visresolve in-band on the render; STRICT ZERO on the
   integer manifold.
4. **Render-invariance:** ONLY `mf3_manifold.png` added; the other 223 byte-identical (+ controller introspect
   rebake).
5. Introspect: exactly `+manifold-gpu` + `--mf3-manifold-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + MF1/MF2 manifold.h code + gjk.h/convex.h/broad.h/ccd.h/fpx.h + ALL other sim headers
   + ALL OTHER shaders byte-unchanged; manifold.h APPEND-only; exactly ONE new shader `hull_manifold.comp.hlsl`,
   Vulkan-only). `mf3_manifold` in the Mac loop + `--mf3-manifold-shot` in `$vkShots`; `hull_manifold.comp.hlsl:cs`
   in the hello_triangle shader list; NO `hf_gen_msl(hull_manifold ...)`.
