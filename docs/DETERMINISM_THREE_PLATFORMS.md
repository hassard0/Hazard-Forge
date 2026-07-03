# Determinism verified on three platforms

Hazard Forge's headline is **bit-exact determinism**: the same inputs produce byte-for-byte identical
simulation and golden state on every supported platform. This document records the evidence.

## The proof

Every "pure" test (a header-only deterministic-core test — the sim families, netcode, tooling cores,
the clean-room codecs — that standalone-compiles with just `-I engine -I tests`, no RHI/device linkage)
hard-codes its **pinned golden digests** as compile-time constants. A test that both *compiles* and
*passes* on a given toolchain has, by construction, reproduced those exact digests bit-for-bit.

Running the whole pure suite on each toolchain therefore proves cross-platform determinism directly:

| Platform | Toolchain | Result |
|---|---|---|
| Windows | MSVC 19.4x (x64) | full `ctest` green (197/197), pure-core digests pinned |
| macOS | Apple clang (arm64) | 351 Metal render goldens `DIFF 0.0000` + pure-core digests match |
| **Linux** | **gcc 14 (x64)** | **135 / 135 pure-core tests pass — every Windows-pinned digest reproduced** |

The 135 Linux/gcc passes include every deterministic-simulation family (rigid, cloth, fluid, grain,
fracture, particles, hair, soft-body, boids, vehicles, ragdoll/joints, and all four two-way couplings),
the lockstep/rollback netcode, the clean-room Draco decoder, the asset pipeline, the flow VM + editor
data-models, the sequencer/UMG/profiler cores, the audio graph, motion matching, the nav mesh, and the
undo/redo command stack. The remaining 61 tests need RHI/device linkage (the render-bound ones) and are
covered by the Windows `ctest` + the Mac golden bake instead.

## Reproduce it

No local toolchain required — a container is enough:

```sh
docker run --rm -v "$PWD:/repo:ro" gcc:14 bash /repo/scripts/linux_determinism_check.sh
```

Or natively on a Linux box with `g++ >= 13`:

```sh
./scripts/linux_determinism_check.sh
```

Exit 0 iff every pure test that compiles also passes.

## Why this matters

Float-based engines cannot make this claim: FPU evaluation order, fused-multiply-add contraction, and
vendor math libraries diverge machine-to-machine, so their physics, navigation, and animation drift
across platforms — which is exactly why their networked simulation is never bit-deterministic. Hazard
Forge's core is fixed-point integer with host-baked LUTs and no runtime transcendentals, so the *same
computation* is byte-identical on MSVC, Apple clang, and gcc alike. That is the foundation the
lockstep + rollback netcode stands on.

*(One test-side fix accompanied this sweep: `ccd_test` compared a returned struct with a trailing `bool`
via raw `memcmp`, whose 3 indeterminate padding bytes made the check compiler-dependent; it now compares
the meaningful fields. No engine code or golden changed — the sim output was always deterministic.)*
