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

| Job              | Runner                       | What it does                                                        |
| ---------------- | ---------------------------- | ------------------------------------------------------------------- |
| `windows-vulkan` | `windows-2022` (hosted)      | conan install (cppstd=17 + Ninja) -> configure -> build -> ctest    |
| `windows-asan`   | `windows-2022` (hosted)      | `HF_SANITIZE=address` build of the pure-C++ tests -> ctest (ASan)   |
| `macos-metal`    | `[self-hosted, macos, metal]`| headless Metal build -> `visual_test` -> golden compare (DIFF 0.0)  |

### The Metal job

GitHub-hosted macOS runners cannot reach the LAN bench Mac and do not provide a Metal GPU whose
offscreen output matches the baked golden, so `macos-metal` is gated to a **self-hosted** runner
labelled `metal` (set one up on the bench Mac and register it with the labels `self-hosted, macos,
metal`). On hosted infrastructure the job is skipped.

For routine local verification of **both** platforms in one command, use:

```powershell
scripts\verify.ps1
```

That script runs the Windows/Vulkan ctest locally and drives the bench Mac over SSH to build the
headless Metal target, render, and compare against `tests/golden/metal/scene_shadow.png` at
threshold `0.0` (must report `DIFF 0.0000`).

## Local equivalents

| Pipeline           | Local command                                                            |
| ------------------ | ------------------------------------------------------------------------ |
| Windows build+test | `cmake --preset windows-msvc-debug` + build + `ctest --preset windows-msvc-debug` |
| ASan build+test    | `cmake --preset windows-msvc-asan` + build + `ctest --preset windows-msvc-asan`   |
| Full cross-platform| `scripts\verify.ps1`                                                      |
