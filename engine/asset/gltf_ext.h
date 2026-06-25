#pragma once

// glTF extension diagnostics — the PURE decision logic for "can this build load this file?", split out
// of gltf_loader.cpp so it is unit-testable WITHOUT a device, cgltf, or a real glb (issue #36).
//
// The bug it fixes: when a glb declares `extensionsRequired: ["KHR_draco_mesh_compression"]` (true of the
// three.js Ferrari and >50% of real-world CC0 glbs), the engine has no Draco decompressor, so the loader
// produced empty/garbage geometry and rendered "0 instances" with NO error — leaving sample authors
// guessing for 45 minutes. Per the glTF spec a loader that cannot honor a REQUIRED extension must not
// silently pretend success; this makes the failure LOUD and ACTIONABLE. Full Draco/meshopt decode (a new
// third-party dependency) is deferred; this is the documented "honest fallback".
//
// Self-contained: only <string>/<vector>. No engine/RHI/cgltf includes, so it compiles standalone.

#include <string>
#include <vector>

namespace hf::asset {

// The geometry-compression extensions this build CANNOT decode (no Draco / no meshopt decompressor). A
// file that REQUIRES one of these cannot be loaded — its mesh geometry lives entirely in the compressed
// buffer the engine can't read. (Material/texture/light extensions are NOT here: cgltf reads through them
// transparently, so they never block geometry.)
inline bool IsUnsupportedGeometryExt(const std::string& ext) {
    return ext == "KHR_draco_mesh_compression" || ext == "EXT_meshopt_compression";
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
// FATAL when any required extension is an unsupported geometry compression (the Ferrari case): the file
// genuinely cannot be loaded, so refuse it loudly. WARNING when Draco primitives exist but the extension
// is only `used`, not `required` (a spec-compliant uncompressed fallback may still render): load proceeds,
// but the author is told some primitives may be empty. Otherwise both empty (a normal file — load silently).
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

    if (dracoPrims > 0) {
        d.warning = "[gltf] " + path + ": " + std::to_string(dracoPrims) + "/" +
                    std::to_string(totalPrims) +
                    " primitives use KHR_draco_mesh_compression (not required); this build has no Draco "
                    "decoder, so those primitives render only from an uncompressed fallback if present, "
                    "else empty. See issue #36.";
    }
    return d;
}

}  // namespace hf::asset
