# Slice AW — Live Runtime Material Authoring (Phase 4 #2) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan (runtime compile + hot-swap) AND Apple M4+Metal (golden parity).
> Builds on AV (material graph) + AM (hot_reload FileWatcher). Closes the agentic author->see loop.

**Goal:** Edit a material `.mat.json` and see the surface re-shade WITHOUT rebuilding the project. The
engine codegens the graph to HLSL in-process, compiles it to SPIR-V at RUNTIME, and hot-swaps the
material pipeline. A FileWatcher on `assets/materials/*.mat.json` drives the live loop. Correctness is
guaranteed by construction: runtime compilation invokes the SAME `dxc` the build uses, so the runtime
SPIR-V is byte-identical to the build-time SPIR-V — the live path provably matches the golden path.

## Why dxc-as-subprocess (the de-risked choice)

The build already compiles HLSL->SPIR-V with a specific standalone `dxc.exe` (`cmake/CompileShaders.cmake`
hints its path). For runtime compilation, **invoke that same dxc as a subprocess** on the in-process-
generated HLSL -> a temp `.spv`, then load it through the EXISTING `.spv` pipeline-creation path. Same
compiler + same flags => byte-identical SPIR-V => the runtime-compiled material is provably the same as
the build-time golden. This avoids linking the DXC COM library (IDxcCompiler3) and its versioning risk.
(Linking dxcompiler is a possible future enhancement, not needed here.) The "runtime" property is real:
no project rebuild — a `.mat.json` change triggers codegen + a dxc invocation + a pipeline swap.

Metal scope: the metal_headless standalone has no runtime HLSL compiler (CLT-only, no conan/spirv-cross at
runtime), so Metal keeps the BUILD-TIME codegen path. Live runtime authoring is a Vulkan/Windows
dev-loop feature; Metal validates codegen correctness via the build-time golden. This asymmetry is honest
and documented.

## Design decisions (locked)

1. **Runtime compile utility (above the seam, no backend symbols).** `engine/material/runtime_compile.{h,cpp}`:
   `std::optional<std::vector<uint32_t>> CompileGraphToSpirv(const Graph&, std::string* errorOut)` =
   `GenerateHlsl(graph)` (reuse AV codegen) -> write a temp `.hlsl` -> invoke `dxc` (path discovered the
   same way the build does: a configured `HF_DXC_PATH` define pointing at the build's dxc, or PATH lookup
   with the WinGet/Vulkan-SDK hint) with the SAME profile/flags the build uses for fragment shaders ->
   read back the `.spv` bytes. Pure host/tooling (subprocess + file IO); NO `vk*`/`MTL*` symbols. Returns
   nullopt + error text on failure (the live loop must fail SAFE — keep the previous material, log the
   error, never crash).

2. **Live material + hot-swap.** Reuse the Slice-AM `FileWatcher` (mtime poll) on the active
   `.mat.json`. On change: reload graph -> `CompileGraphToSpirv` -> if success, create a new material
   pipeline from the SPIR-V and swap it in (destroy the old one safely after the frame); if failure, keep
   the current pipeline + log. Add an RHI path to build a graphics pipeline from RAW SPIR-V bytes at
   runtime IF one doesn't already exist — inspect: the Vulkan backend already creates pipelines from
   `.spv`; expose/route a "create material pipeline from in-memory SPIR-V" that reuses the existing PBR
   pipeline layout/descriptor sets. Keep it additive; no `vk*` above the seam.

3. **Verification handles.**
   - `--material-live-shot <out> [mat.json]` (Vulkan): renders the given material via the RUNTIME path
     (in-process codegen -> dxc subprocess -> SPIR-V -> pipeline -> render). For `showcase.mat.json` the
     output MUST be BYTE-IDENTICAL to AV's `--material-shot` (which used the build-time committed HLSL) —
     this is the runtime==build-time proof (Vulkan SHA match).
   - A SECOND example material `assets/materials/showcase2.mat.json` exercising a different node mix (e.g.
     metallic driven by a TextureSample, roughness by Fresnel, baseColor by Multiply of two Constants) so
     the slice demonstrates the authoring range. New golden `tests/golden/metal/mat_graph2.png` (Metal via
     the build-time codegen of showcase2 — commit its generated HLSL; two M4 runs DIFF 0.0000). On Vulkan
     render showcase2 via the runtime path -> must match the Metal golden's content (same graph -> same
     image).
   - `--material-hotswap-dry-run` (headless, like AM's --fly-dry-run): programmatically load showcase ->
     render hash, then load showcase2 -> render hash, asserting each equals its golden and that the swap
     happened without leaking/crashing (exercises the live swap path deterministically without a GUI).

4. **Tests `tests/runtime_material_test.cpp`** (host logic, no GPU): the reload orchestration — given a
   stub compiler that returns success/failure, the controller swaps on success and KEEPS the old material
   + records the error on failure; FileWatcher fires exactly on mtime change; a malformed `.mat.json`
   (or a graph that fails validation) does NOT swap and surfaces an error. Clean under `windows-msvc-asan`.
   (The actual dxc-subprocess compile is integration-tested via `--material-live-shot` byte-equality on
   Windows; document that the unit test stubs the compiler so it stays GPU/dxc-free and deterministic.)

5. **Introspect.** Add exactly `live-material-authoring` (features) + `--material-live-shot` (showcases).
   The second material does NOT get its own showcase flag (it's rendered via `--material-live-shot
   showcase2.mat.json` / the dry-run). One-pattern rebake, no other drift.

## RHI seam additions (summary)
- At most an additive "create material graphics pipeline from in-memory SPIR-V" entry IF the existing
  `.spv` pipeline path can't be reused directly — pure-interface, backend impl inside `rhi_vulkan/`
  (Metal not required for the runtime path). Runtime compile + watcher + swap controller are pure host
  logic above the seam. No new `vk*`/`MTL*` code symbols above the seam (audit any tick-up = comments).

## Out of scope (YAGNI)
Linking the DXC COM library, runtime material compilation ON Metal, a visual node editor, material
parameter sliders / live scalar tweaking without recompile, shader caching/incremental compile, compile
on a background thread (synchronous recompile is fine for the dev loop), more node types than AV's set,
hot-reloading the vertex stage. One runtime compile via dxc-subprocess, FileWatcher swap, two example
materials, byte-equality + golden proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 28) + new `runtime_material_test` (reload
   orchestration with a stubbed compiler + watcher + fail-safe). Clean under `windows-msvc-asan`.
2. **Runtime==build-time proof:** `--material-live-shot showcase.mat.json` byte-identical (SHA) to AV's
   `--material-shot` on Vulkan (same dxc => same SPIR-V => same image).
3. `--material-live-shot` + `--material-hotswap-dry-run` run under the AT Vulkan-validation gate -> ZERO
   errors. The dry-run asserts the A->B swap matches each golden without crash/leak.
4. Metal: new golden `tests/golden/metal/mat_graph2.png` for showcase2 (build-time codegen path); two
   runs DIFF 0.0000. `mat_graph.png` (AV) unchanged.
5. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `mat_graph2.png`
   added; the other 27 byte-identical.
6. Introspect JSON rebaked exactly `+live-material-authoring` + `--material-live-shot`; introspect test
   updated; no other drift.
7. Seam grep clean (no new code symbols above the seam). `scripts/verify.ps1` updated to include the new
   `mat_graph2` image golden AND (ideally) the Windows `--material-live-shot`==`--material-shot`
   byte-equality assertion.
