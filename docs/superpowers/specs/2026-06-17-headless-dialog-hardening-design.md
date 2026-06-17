# Slice — Headless Crash-Dialog Suppression (agentic-operability hardening) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> A small AGENTIC-OPERABILITY hardening (the mission's core pillar: headless / machine-observable / no modal
> interrupts). On Windows, the test + showcase exes currently pop MODAL C++ runtime / Debug-Assertion / WER
> dialogs on a tripped assert / abort / hard fault — a modal dialog BLOCKS the process indefinitely, which
> HANGS headless automation (background build/test agents). This slice routes those to stderr + a clean
> failure-exit so a headless exe NEVER throws a GUI dialog. No golden, no RHI, no behavior change on success.

**Goal:** Add a tiny `_WIN32`/`_MSC_VER`-guarded `hf::platform::DisableCrashDialogs()` called once at startup
in every engine executable's `main` (the unit-test harness, `samples/hello_triangle`, and `metal_headless`)
so that a CRT assert / `abort()` / invalid-parameter / hard fault writes a diagnostic to **stderr** and exits
with a **non-zero code** instead of popping a modal dialog. On non-Windows (Apple/Metal) it's a no-op. This
makes a tripped assert a clean, machine-observable failure (which ctest + the build agents already handle)
rather than a hang. It changes NOTHING on the success path (no assert tripped → nothing happens).

## Why (the operability gap)
During the DDGI arc the modal CRT/WER dialogs popped during agents' intermediate (broken-build) iteration and
BLOCKED the process — a headless engine must never do this. A controller-side watchdog currently kills the
dialogs externally; this slice is the permanent in-engine fix so the watchdog becomes unnecessary.

## Design decisions (locked)

1. **New header `engine/platform/crash_dialogs.h` (or extend an existing platform header).** Namespace
   `hf::platform`. A single function:
   ```cpp
   // Route CRT asserts / abort() / invalid-parameter / hard faults to stderr + a non-zero exit, suppressing
   // the modal Windows dialog so headless test/showcase exes never block on a GUI prompt. _WIN32+_MSC_VER
   // only; a no-op elsewhere. Call once at the top of main(). This is platform/runtime code (NOT a vk*/MTL*
   // backend symbol) — it is allowed above the RHI seam (like a logging/assert helper); document that in a
   // seam-discipline comment.
   void DisableCrashDialogs();
   ```
   On `_WIN32 && _MSC_VER`, the implementation (header-inline or a small .cpp) does:
   - `#include <crtdbg.h>` then for `_CRT_WARN`, `_CRT_ERROR`, `_CRT_ASSERT`:
     `_CrtSetReportMode(mode, _CRTDBG_MODE_FILE); _CrtSetReportFile(mode, _CRTDBG_FILE_STDERR);` — asserts +
     CRT errors write to stderr and the report returns (no dialog). (In a release build `assert` is compiled
     out; this affects the debug + ASan builds where the dialogs actually appear.)
   - `_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);` — `abort()` doesn't pop the abort/WER
     dialog (still writes its message to stderr, still terminates).
   - `SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);` (`#include
     <windows.h>`) — suppress the WER "stopped working" / critical-error message boxes on a hard fault.
   - Optionally `_set_invalid_parameter_handler(...)` routing a CRT invalid-parameter to `fprintf(stderr,...)`
     + `_exit(3)` instead of the default modal report (document; keep minimal). Optionally `_CrtSetReportHook`
     is NOT needed — the mode/file settings cover it.
   - On non-`_WIN32` (or non-MSVC): the function body is empty (`#else` → `{}`).
   - Keep it dependency-light (only `<crtdbg.h>`/`<windows.h>`/`<cstdlib>`/`<cstdio>`), header-only-inline is
     fine, ZERO RHI/vk/MTL symbols (it's win32 CRT + kernel32 — platform, not backend; document the seam
     exception in a comment, exactly like the existing platform/HAL code that already uses win32 APIs).

2. **Call it at the top of each exe `main`.** Find the actual entry points and add `hf::platform::DisableCrash
   Dialogs();` as the FIRST line of `main` (before any other init): the unit-test harness main (the shared
   test `main` that the `hf_add_pure_test` exes link — find it; if each test has its own main via a shared
   helper, add it there once), `samples/hello_triangle/main.cpp` (`int main`/`WinMain`), and
   `metal_headless`'s entry (on Windows it isn't built, but adding the include is harmless / guarded — only
   add where it compiles; the Mac build is unaffected since the function is a no-op there). Document which
   mains were touched.

3. **No behavior change on success.** When no assert trips / no abort / no fault, `DisableCrashDialogs()` only
   configures the CRT/WER reporting mode — the program runs identically. ALL existing goldens + the existing
   77 image goldens + every proof MUST stay byte-identical (the function does not touch any rendering or
   determinism). This is the make-safe constraint: verify a broad set of existing proofs + the goldens are
   unchanged.

## RHI seam additions (summary)
- **None.** This is platform/runtime CRT + kernel32 code (win32 `SetErrorMode` / CRT `_CrtSetReportMode` /
  `_set_abort_behavior`) — it lives in `engine/platform/` (NOT a backend dir), the same as the existing
  win32/HAL platform code; it is NOT a `vk*`/`MTL*`/`Backend::Metal`/`mtl::` backend symbol (those tokens
  don't appear). The RHI seam (rhi.h, rhi_factory, the backend dirs) is UNCHANGED. rhi_factory dispatch stays
  at baseline 2. Document the platform-code-above-seam rationale in a comment.

## Verification gate
1. `ctest --preset windows-msvc-debug` → still 77/77 (the function is a no-op on the success path; no test
   behavior changes). Clean under `windows-msvc-asan` (77/77; importantly, the ASan exes now route any
   ASan/CRT report to stderr without a dialog — verify ASan still detects + reports errors to stderr, just no
   modal box).
2. **Dialog-suppression behavior proof (manual/scripted, no golden):** demonstrate that a deliberately-tripped
   assert in a throwaway run writes to STDERR and exits NON-ZERO with NO modal dialog. E.g. a tiny temporary
   `--selftest-assert` path (or a disposable test) that calls `assert(false)` after `DisableCrashDialogs()`
   and confirms (a) the process exits with a non-zero code, (b) the assert text appears on stderr, (c) NO GUI
   dialog appears (the process returns immediately rather than blocking). Document the result. (Do NOT commit
   a permanent failing test — use a throwaway/guarded path or just describe the verification you ran.)
3. **Render-invariance:** NO golden changes — `git diff master --stat -- tests/golden` EMPTY. The 77 image
   goldens + the proofs are untouched (spot-check a broad set: `--ddgi`... no, GI composite not yet;
   `--probeinterp-shot` GPU==CPU, `--probesh-shot` GPU==CPU, `--cas-shot` sharpness=0, `--captureprobe-shot`
   face-0, `--clustered-lights-shot` clustered==brute-force, `--oit-shot` permuted==canonical all still
   byte-identical = the startup call is inert for rendering).
4. **Vulkan validation-clean** on a couple showcases (unchanged). Metal build still compiles (the function is a
   no-op there). Gate any Metal re-render on compare.sh EXIT CODE (but no new golden, so no Mac round-trip is
   strictly required — confirm the Mac `metal_headless` still BUILDS with the included header).
5. **Seam grep** clean (no new backend code symbols; rhi.h unchanged). Introspect: OPTIONAL — this is a
   hardening with no showcase; do NOT add a feature/showcase to introspect (it's not a user-facing engine
   feature). So introspect golden UNCHANGED too.

## Out of scope (YAGNI)
Crash dumps / minidump writing, a custom unhandled-exception filter beyond suppressing the box, signal
handling on POSIX, logging-framework integration. One small `DisableCrashDialogs()` startup call that makes the
headless exes route asserts/abort/faults to stderr + exit instead of a modal dialog.

## Verification gate summary
ctest 77/77 debug+asan unchanged; the deliberate-assert run → stderr + non-zero exit + NO dialog; ZERO golden
changes (git diff master -- tests/golden EMPTY); a broad proof spot-check still byte-identical; seam baseline 2
+ rhi.h unchanged; Mac metal_headless still builds. NO golden, NO introspect change, NO RHI.
