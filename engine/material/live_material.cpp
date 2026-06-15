// Hazard Forge — live runtime material-authoring controller (Slice AW). See live_material.h.
//
// Pure host logic: reload .mat.json -> validate -> compile (injected) -> swap-or-keep. Never throws,
// never crashes on a bad edit; a failure keeps the previously active SPIR-V and records the error.
#include "material/live_material.h"

#include "material/codegen.h"
#include "material/material_loader.h"

namespace hf::material {

bool LiveMaterial::LoadInitial() {
    ReloadStatus st = DoReload();
    if (!watching_) { watcher_.Watch(path_); watching_ = true; }
    return st == ReloadStatus::Swapped;
}

ReloadStatus LiveMaterial::Reload() { return DoReload(); }

ReloadStatus LiveMaterial::SwitchTo(std::string path) {
    path_ = std::move(path);
    ReloadStatus st = DoReload();
    // Re-baseline the watcher onto the new path so subsequent Poll()s track it.
    watcher_ = runtime::FileWatcher{};
    watcher_.Watch(path_);
    watching_ = true;
    return st;
}

ReloadStatus LiveMaterial::Poll() {
    std::vector<std::string> changed = watcher_.Poll();
    bool hit = false;
    for (const std::string& p : changed)
        if (p == path_) { hit = true; break; }
    if (!hit) return ReloadStatus::NoChange;
    return DoReload();
}

ReloadStatus LiveMaterial::DoReload() {
    // 1. Load + parse the .mat.json. A malformed document fails here (fail-safe: keep old).
    LoadResult lr = LoadGraphFromFile(path_);
    if (!lr.ok) {
        lastError_ = "load failed: " + lr.error;
        changed_ = false;
        return activeSpirv_.empty() ? ReloadStatus::Kept : ReloadStatus::Kept;
    }

    // 2. Validate the graph BEFORE compiling. An invalid graph never reaches the compiler.
    ValidationResult vr = Validate(lr.graph);
    if (!vr) {
        lastError_ = "invalid graph: " + vr.error;
        changed_ = false;
        return ReloadStatus::Kept;
    }

    // 3. Compile to SPIR-V (injected: dxc subprocess in production, a stub in tests). A compile
    //    failure keeps the previous material.
    std::string err;
    std::optional<std::vector<uint32_t>> spirv = compile_(lr.graph, &err);
    if (!spirv || spirv->empty()) {
        lastError_ = err.empty() ? "compile failed" : ("compile failed: " + err);
        changed_ = false;
        return ReloadStatus::Kept;
    }

    // 4. Success: promote the new SPIR-V to active.
    activeSpirv_ = std::move(*spirv);
    lastError_.clear();
    changed_ = true;
    ++swapCount_;
    return ReloadStatus::Swapped;
}

}  // namespace hf::material
