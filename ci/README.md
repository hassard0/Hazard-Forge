# CI

This directory holds the GitHub Actions workflow as a **staged** file:

```
ci/github-actions-ci.yml
```

It is intentionally NOT under `.github/workflows/`.

## Why it lives here (and not under .github/workflows/)

The credential currently used to push this repo **lacks the `workflow` OAuth scope**. GitHub
rejects any push that adds or modifies a file under `.github/workflows/` unless the token carries
that scope, so committing the workflow there would make `git push` fail outright. Keeping the YAML
under `ci/` lets the branch push succeed; the file is ready to be activated whenever a suitably
scoped token is available.

## How to enable CI

1. Obtain a token (PAT or GitHub App installation token) that includes the **`workflow`** scope.
   - Classic PAT: tick the `workflow` scope.
   - Fine-grained PAT / App: grant **Actions: read & write** + **Contents: read & write**.
2. Move the file into place and commit with that token:

   ```sh
   mkdir -p .github/workflows
   git mv ci/github-actions-ci.yml .github/workflows/ci.yml
   git commit -m "ci: enable GitHub Actions workflow"
   git push    # now succeeds because the token has the workflow scope
   ```

3. Push. The `windows-vulkan` and `windows-asan` jobs run on GitHub-hosted `windows-2022` runners.

## Jobs

| Job              | Runner                       | What it does                                                              |
| ---------------- | ---------------------------- | ------------------------------------------------------------------------- |
| `windows-vulkan` | `windows-2022` (hosted)      | conan install (cppstd=17 + Ninja) -> configure -> build -> ctest (22 tests) + the introspection JSON-golden byte match |
| `windows-asan`   | `windows-2022` (hosted)      | `HF_SANITIZE=address` build of the pure-C++ core + tests -> ctest under ASan (22 tests; the 21 pure ones instrumented, `rhi_smoke` not) |
| `macos-metal`    | `[self-hosted, macos, metal]`| headless Metal build -> render + golden-compare **all 22 goldens** (DIFF 0.0) |

### The Metal job

GitHub-hosted macOS runners cannot reach the LAN bench Mac and do not provide a Metal GPU whose
offscreen output matches the baked goldens, so `macos-metal` is gated to a **self-hosted** runner
labelled `metal` (set one up on the bench Mac and register it with the labels `self-hosted, macos,
metal`). On hosted infrastructure the job is skipped.

The job builds `metal_headless` **once**, then for each of the **22** committed goldens runs
`visual_test <flag> /tmp/hf_<name>.png` and compares it to `tests/golden/metal/<name>.png` at
threshold `0.0`. Every pair must report `DIFF 0.0000`; the job fails if any golden drifts. The
22 (golden -> flag) pairs are:

| golden              | flag                          | golden              | flag                  |
| ------------------- | ----------------------------- | ------------------- | --------------------- |
| `scene_shadow`      | *(default)*                   | `gizmo`             | `--gizmo 2`           |
| `skinning`          | `--skinning`                  | `csm`               | `--csm`               |
| `pbr_helmet`        | `--pbr`                       | `spot`              | `--spot`              |
| `instanced`         | `--instanced`                 | `point_shadow`      | `--point-shadow`      |
| `ibl_helmet`        | `--ibl`                       | `clustered`         | `--clustered`         |
| `physics`           | `--physics`                   | `ssr`               | `--ssr`               |
| `transparency`      | `--transparency`              | `scene_import`      | `--scene`             |
| `bloom`             | `--bloom`                     | `debug_viz`         | `--debug`             |
| `anim_blend`        | `--blend`                     | `ssao`              | `--ssao`              |
| `capstone`          | `--capstone`                  | `camera_pose`       | `--camera 0.2,-0.1,0,3,10` |
| `volumetric`        | `--volumetric`                | `probe`             | `--probe`             |

For routine local verification of **both** platforms in one command, use:

```powershell
scripts\verify.ps1
```

That script runs the Windows/Vulkan ctest locally (plus the introspection JSON-golden byte match)
and drives the bench Mac over SSH to build the headless Metal target once and run the **same
22-golden loop**, each compared at threshold `0.0` (every one must report `DIFF 0.0000`). It prints
a per-golden table and an overall `VERIFY: PASS/FAIL`.

## Local equivalents

| Pipeline           | Local command                                                            |
| ------------------ | ------------------------------------------------------------------------ |
| Windows build+test | `cmake --preset windows-msvc-debug` + build + `ctest --preset windows-msvc-debug` |
| ASan build+test    | `cmake --preset windows-msvc-asan` + build + `ctest --preset windows-msvc-asan`   |
| Full cross-platform| `scripts\verify.ps1`                                                      |
