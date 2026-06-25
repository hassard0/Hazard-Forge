// Unit test for the glTF unsupported-extension diagnostics (engine/asset/gltf_ext.h, issue #36).
//
// Issue #36 is now FIXED: the engine ships a self-contained KHR_draco_mesh_compression decoder, so a
// Draco glb LOADS (the compressed geometry is decoded) — Draco is NO LONGER fatal/unsupported. What
// remains genuinely unsupported is EXT_meshopt_compression (a different codec): a file that REQUIRES it
// must still FAIL LOUDLY rather than render "0 instances" silently. DiagnoseExtensions() is the pure
// decision the loader uses; this test exercises it directly with synthetic inputs — no device, no cgltf,
// no glb file — so it is a fast standalone pure test:
//   clang++ -std=c++20 -I engine -I tests tests/gltf_ext_test.cpp

#include "asset/gltf_ext.h"

#include <cstdio>
#include <string>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::asset;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS: %s\n", what); }
}
static bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

int main() {
    HF_TEST_MAIN_INIT();

    // (1) The Ferrari case: REQUIRES Draco -> now LOADS (issue #36 fixed). Draco is no longer fatal or
    // a warning — the loader decodes it directly, so DiagnoseExtensions stays silent for it.
    {
        ExtDiagnostic d = DiagnoseExtensions({"KHR_draco_mesh_compression"}, 51, 51, "ferrari.glb");
        check(d.fatal.empty(), "gltf-ext: required Draco is NO LONGER fatal (it now decodes + loads)");
        check(d.warning.empty(), "gltf-ext: required Draco produces no warning either (loads cleanly)");
    }

    // (2) A normal Khronos asset (no required compression, no Draco prims) -> both empty -> loads silently.
    {
        ExtDiagnostic d = DiagnoseExtensions({}, 0, 12, "Fox.glb");
        check(d.fatal.empty() && d.warning.empty(), "gltf-ext: a normal file produces NO diagnostic (loads silently)");
    }

    // (3) A harmless REQUIRED material extension must NOT false-alarm.
    {
        ExtDiagnostic d = DiagnoseExtensions({"KHR_materials_unlit"}, 0, 3, "unlit.glb");
        check(d.fatal.empty() && d.warning.empty(), "gltf-ext: a required MATERIAL extension is not a false alarm");
    }

    // (4) Draco USED but not REQUIRED -> loads silently (issue #36: Draco decodes, no warning needed).
    {
        ExtDiagnostic d = DiagnoseExtensions({}, 3, 10, "partial.glb");
        check(d.fatal.empty(), "gltf-ext: Draco-used is NOT fatal (it decodes)");
        check(d.warning.empty(), "gltf-ext: Draco-used emits no warning (the loader decodes it)");
    }

    // (5) EXT_meshopt_compression required -> also FATAL (the modern compression alternative).
    {
        ExtDiagnostic d = DiagnoseExtensions({"EXT_meshopt_compression"}, 0, 8, "meshopt.glb");
        check(!d.fatal.empty() && contains(d.fatal, "EXT_meshopt_compression"), "gltf-ext: required meshopt is FATAL");
    }

    // (6) The predicate is precise: meshopt is still unsupported; Draco is NOT (it decodes now, #36).
    {
        check(IsUnsupportedGeometryExt("EXT_meshopt_compression"), "gltf-ext: meshopt is flagged unsupported");
        check(!IsUnsupportedGeometryExt("KHR_draco_mesh_compression"), "gltf-ext: Draco is NO LONGER flagged unsupported (it loads)");
        check(!IsUnsupportedGeometryExt("KHR_texture_transform"), "gltf-ext: a texture extension is NOT flagged");
    }

    if (g_fail == 0) std::printf("gltf_ext_test: ALL PASS\n");
    else             std::printf("gltf_ext_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
