#pragma once

// glTF extension diagnostics — the PURE decision logic for "can this build load this file?", split out
// of gltf_loader.cpp so it is unit-testable WITHOUT a device, cgltf, or a real glb (issue #36).
//
// ISSUE #36 IS FIXED (DRACO LOADS): the engine now ships a self-contained, clean-room
// KHR_draco_mesh_compression decoder (engine/asset/draco_decode.h), wired into the glTF loader, so a glb
// that declares `extensionsRequired: ["KHR_draco_mesh_compression"]` (the three.js Ferrari and >50% of
// real-world CC0 glbs) LOADS — the compressed geometry is decoded to a real mesh instead of silently
// dropping it. Draco is therefore NO LONGER an unsupported geometry extension.
//
// What remains genuinely unsupported is EXT_meshopt_compression (a different codec, no decoder shipped):
// a file that REQUIRES it still cannot be loaded — its mesh geometry lives entirely in a compressed buffer
// the engine can't read — so we keep that failure LOUD and ACTIONABLE rather than rendering "0 instances"
// with no error. (Material/texture/light extensions are never here: cgltf reads through them transparently,
// so they never block geometry.)
//
// Self-contained: only <string>/<vector>. No engine/RHI/cgltf includes, so it compiles standalone.

#include <string>
#include <vector>

namespace hf::asset {

// The geometry-compression extensions this build CANNOT decode. KHR_draco_mesh_compression is NO LONGER
// here (issue #36: it now decodes + loads); only EXT_meshopt_compression remains genuinely unsupported (a
// different codec, no decoder shipped). A file that REQUIRES one of these still cannot be loaded — its mesh
// geometry lives entirely in the compressed buffer the engine can't read.
inline bool IsUnsupportedGeometryExt(const std::string& ext) {
    return ext == "EXT_meshopt_compression";
}

struct ExtDiagnostic {
    std::string fatal;     // non-empty => the file requires an extension we can't honor; the loader MUST throw this
    std::string warning;   // non-empty => a non-fatal note to log (e.g. Draco used-but-not-required)
};

// Decide the diagnostic from the file's REQUIRED extensions + how many primitives are Draco-compressed.
//   * requiredExts  — the glb's `extensionsRequired` list (cgltf data->extensions_required).
//   * dracoPrims    — count of primitives flagged has_draco_mesh_compression.
//   * totalPrims    — total primitive count (for an actionable "<N>/<M>" message).
//   * path          — the file path, echoed into the message so the author knows WHICH file.
// FATAL when any required extension is an unsupported geometry compression (an EXT_meshopt_compression-
// required file): it genuinely cannot be loaded, so refuse it loudly. Otherwise both empty — a normal file
// (including a Draco-compressed one, now that the engine decodes Draco) loads silently.
//
// NOTE: Draco is no longer surfaced here at all (neither fatal nor warning) — the loader decodes it
// directly (issue #36). The `dracoPrims`/`totalPrims` parameters are retained for ABI/message stability
// but a Draco primitive count alone never produces a diagnostic now.
inline ExtDiagnostic DiagnoseExtensions(const std::vector<std::string>& requiredExts,
                                        int dracoPrims, int totalPrims, const std::string& path) {
    ExtDiagnostic d;

    // Collect the unsupported REQUIRED extensions, preserving order.
    std::vector<std::string> blocking;
    for (const std::string& ext : requiredExts)
        if (IsUnsupportedGeometryExt(ext)) blocking.push_back(ext);

    if (!blocking.empty()) {
        std::string list;
        for (std::size_t i = 0; i < blocking.size(); ++i) {
            if (i) list += ", ";
            list += blocking[i];
        }
        d.fatal = "[gltf] " + path + ": file REQUIRES extension(s) [" + list +
                  "] that this build cannot decode (no Draco/meshopt decompressor). " +
                  std::to_string(dracoPrims) + "/" + std::to_string(totalPrims) +
                  " primitives are compressed and would not render. Integrate a Draco decoder "
                  "(see issue #36) or re-export the asset uncompressed.";
        return d;   // fatal dominates — no need for the softer warning
    }

    // Draco primitives are decoded by the loader now (issue #36), so a Draco count alone is NOT a
    // diagnostic. (dracoPrims/totalPrims are kept in the signature only for message stability.)
    (void)dracoPrims;
    (void)totalPrims;
    return d;
}

}  // namespace hf::asset
