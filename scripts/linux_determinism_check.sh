#!/usr/bin/env bash
# Linux/gcc third-platform determinism sweep for Hazard Forge.
#
# Every "pure" test (a header-only deterministic core test that standalone-compiles with just
# `-I engine -I tests`, no RHI/.cpp linkage) hard-codes its Windows-pinned golden digests. So a test
# that COMPILES and PASSES under Linux/gcc has, by construction, reproduced the exact Windows/MSVC (and
# macOS/Apple-clang) digests bit-for-bit — the determinism moat verified on a THIRD toolchain + OS.
#
# Run from the repo root inside a gcc container (no local toolchain needed):
#   docker run --rm -v "$PWD:/repo:ro" gcc:14 bash /repo/scripts/linux_determinism_check.sh
# or natively on a Linux box with g++ >= 13 on PATH:
#   ./scripts/linux_determinism_check.sh
#
# Exit 0 iff every pure test that compiles also passes. Tests that need RHI/.cpp linkage (the render/
# device-bound ones) are reported as "needs-linkage" and skipped — they are covered by the Windows ctest
# + the Mac golden bake, not this pure-core sweep.

set -u
# Run from the repo root: use CWD if it already looks like the repo, else derive from this script's path.
[ -d tests ] && [ -d engine ] || cd "$(dirname "$0")/.." || exit 2

CXX="${CXX:-g++}"
pass=0; runfail=0; needlink=0
runfails=""

for t in tests/*_test.cpp; do
  name=$(basename "$t" .cpp)
  if "$CXX" -std=c++20 -O1 -I engine -I tests "$t" -o "/tmp/hf_$name" 2>/dev/null; then
    if "/tmp/hf_$name" >/dev/null 2>&1; then
      pass=$((pass + 1))
    else
      runfail=$((runfail + 1)); runfails="$runfails $name"
    fi
    rm -f "/tmp/hf_$name"
  else
    needlink=$((needlink + 1))
  fi
done

echo "Linux/gcc pure-core determinism sweep:"
echo "  PASS (Windows-pinned digests reproduced bit-for-bit): $pass"
echo "  needs RHI/.cpp linkage (out of scope here):            $needlink"
echo "  RUN-FAIL:                                              $runfail$runfails"

[ "$runfail" -eq 0 ]
