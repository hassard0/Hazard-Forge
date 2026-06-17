#pragma once
// ===================================================================================================
// engine/platform/crash_dialogs.h  --  Headless crash-dialog suppression (agentic-operability)
// ---------------------------------------------------------------------------------------------------
// hf::platform::DisableCrashDialogs() routes a tripped CRT assert / abort() / invalid-parameter /
// hard fault to STDERR + a non-zero process exit, INSTEAD of popping a MODAL Windows dialog
// ("Microsoft Visual C++ Runtime Library" / "Debug Assertion Failed" / the WER "stopped working"
// box). A modal dialog BLOCKS the process indefinitely until a human dismisses it, which HANGS a
// headless build/test agent. Routing the report to stderr makes a tripped assert a clean,
// machine-observable failure (ctest + the build agents already handle a non-zero exit) rather than a
// hang. Call it ONCE as the first line of main() in every engine executable.
//
// SUCCESS-PATH INERT: when nothing asserts/aborts/faults, this only configures the CRT/WER reporting
// MODE -- the program runs byte-identically (no rendering / determinism / golden impact).
//
// SEAM DISCIPLINE: this is platform/runtime code -- it uses win32 CRT (<crtdbg.h>, _set_abort_behavior)
// + kernel32 (SetErrorMode, forward-declared to avoid leaking <windows.h> macros), exactly like the
// existing engine/hal win32/SDL platform code that already
// lives above the RHI seam. It contains ZERO vk*/MTL*/Backend::Metal/mtl:: backend symbols, so (like a
// logging/assert helper) it is allowed above the seam and is NOT a backend dependency. The RHI seam
// (rhi.h, rhi_factory, the backend dirs) is untouched.
//
// PLATFORM: active only on _WIN32 && _MSC_VER (the MSVC debug + ASan builds, where the dialogs appear;
// in a release build `assert` is compiled out). On every other target -- including the Apple/clang
// Metal build (metal_headless / mac_window) -- it is a true no-op (empty body), so it compiles and
// links everywhere and changes nothing on non-Windows.
// ===================================================================================================

#if defined(_WIN32) && defined(_MSC_VER)

// All win32/CRT includes are pulled in at FILE SCOPE (outside any namespace) so the standard headers
// they transitively include are not nested inside hf::platform.
#include <crtdbg.h>   // _CrtSetReportMode / _CrtSetReportFile / _CRTDBG_*
#include <cstdio>     // fprintf, stderr
#include <cstdlib>    // _set_abort_behavior, _exit
#include <cstdint>    // uintptr_t

// NOTE: we deliberately do NOT #include <windows.h>. Pulling the full win32 header into this tiny
// startup helper would leak its legacy macros (min/max/near/far/...) into every translation unit that
// includes us -- the unit-test exes use identifiers like `far` and `min`, so windows.h would break
// their compiles. Instead we forward-declare the ONE kernel32 entry point we need (SetErrorMode) with
// its documented stdcall signature + the SEM_* flag values, which is dependency-light and macro-clean.
extern "C" __declspec(dllimport) unsigned int __stdcall SetErrorMode(unsigned int uMode);
#ifndef SEM_FAILCRITICALERRORS
#  define SEM_FAILCRITICALERRORS   0x0001u
#endif
#ifndef SEM_NOGPFAULTERRORBOX
#  define SEM_NOGPFAULTERRORBOX    0x0002u
#endif
#ifndef SEM_NOOPENFILEERRORBOX
#  define SEM_NOOPENFILEERRORBOX   0x8000u
#endif

namespace hf {
namespace platform {

// Invalid-parameter handler: the CRT calls this when a Secure-CRT check trips (e.g. a bad iterator /
// bad printf format in a debug build). The default handler pops a modal report; we route a one-line
// diagnostic to stderr and exit non-zero so the failure is machine-observable, never a GUI prompt.
inline void HfInvalidParameterHandler(const wchar_t* /*expr*/, const wchar_t* /*func*/,
                                      const wchar_t* /*file*/, unsigned int /*line*/,
                                      uintptr_t /*reserved*/) {
    std::fprintf(stderr, "[hf] CRT invalid-parameter detected; exiting non-zero (headless).\n");
    std::fflush(stderr);
    _exit(3);
}

inline void DisableCrashDialogs() {
    // 1. CRT asserts + CRT _RPT/_ASSERT reports -> stderr (mode FILE/stderr), so the report WRITES and
    //    RETURNS instead of opening the modal "Debug Assertion Failed" dialog. Covers WARN/ERROR/ASSERT.
    for (int mode = 0; mode < _CRT_ERRCNT; ++mode) {  // _CRT_WARN=0, _CRT_ERROR=1, _CRT_ASSERT=2
        _CrtSetReportMode(mode, _CRTDBG_MODE_FILE);
        _CrtSetReportFile(mode, _CRTDBG_FILE_STDERR);
    }

    // 2. abort() -> write its message to stderr, raise SIGABRT/report-fault for normal termination, but
    //    DO NOT pop the abort()/WER "This application has requested the Runtime to terminate" dialog.
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);

    // 3. Hard fault (AV/GP fault) -> suppress the WER "stopped working" / critical-error message boxes
    //    so the process terminates to the OS (non-zero exit) without a modal box.
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);

    // 4. CRT invalid-parameter -> our stderr+_exit handler instead of the default modal report.
    _set_invalid_parameter_handler(&HfInvalidParameterHandler);
}

} // namespace platform
} // namespace hf

#else  // ---- non-Windows or non-MSVC (incl. the Apple/clang Metal build): a true no-op ----

namespace hf {
namespace platform {

// No modal-dialog problem off Windows/MSVC; keep the call site uniform with an empty inline no-op.
inline void DisableCrashDialogs() {}

} // namespace platform
} // namespace hf

#endif
