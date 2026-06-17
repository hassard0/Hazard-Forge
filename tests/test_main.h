#pragma once
// tests/test_main.h -- shared startup helper for the unit-test exes.
//
// Each pure test (hf_add_pure_test) and rhi_smoke has its own int main(); they all #include this and
// call HF_TEST_MAIN_INIT() as the first statement of main() so a tripped assert / abort() / hard fault
// routes to STDERR + a non-zero exit instead of a MODAL Windows dialog that would HANG a headless
// build/test agent. The macro is a thin, single-line wrapper over hf::platform::DisableCrashDialogs()
// (a no-op on the success path and off Windows/MSVC). Keeping it in one header means the 77 test mains
// share one hook point rather than each open-coding the call.

#include "platform/crash_dialogs.h"

#define HF_TEST_MAIN_INIT() ::hf::platform::DisableCrashDialogs()
