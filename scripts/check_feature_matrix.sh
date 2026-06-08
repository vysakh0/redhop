#!/usr/bin/env bash
# Feature-matrix verification: every combination of the redhop core's
# optional features (--no-default, files, semantic, files+semantic) must
# build cleanly, both for the core crate and for the Python binding.
#
# Why: the published Python wheel ships with `files+semantic`, but a user
# who builds redhop in their own Rust project may pick any combination.
# A bug that only surfaces under e.g. `--no-default-features` (or under
# `files` but not `semantic`) would never be caught by the default
# `cargo test --workspace` run that the existing CI does. This script
# closes that gap.
#
# The Node binding ships a single all-features-on configuration by
# design (see nodejs/Cargo.toml: `redhop = { ..., features = ["files",
# "semantic"] }`), so there's no Node matrix to sweep here.
#
# Run locally:  bash scripts/check_feature_matrix.sh
# In CI:        invoked by the `feature-matrix` job in .github/workflows/ci.yml

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FEATURES=( "" "files" "semantic" "files,semantic" )

fail=0
for ff in "${FEATURES[@]}"; do
    label="${ff:-<none>}"

    echo
    echo "=== [redhop core] --no-default-features --features '$label' ==="
    if [ -z "$ff" ]; then
        cargo check -p redhop --no-default-features --all-targets
    else
        cargo check -p redhop --no-default-features --features "$ff" --all-targets
    fi || fail=1

    echo
    echo "=== [python binding] --no-default-features --features '$label' ==="
    pushd python >/dev/null
    if [ -z "$ff" ]; then
        cargo check --no-default-features --all-targets
    else
        cargo check --no-default-features --features "$ff" --all-targets
    fi || fail=1
    popd >/dev/null
done

if [ "$fail" -ne 0 ]; then
    echo
    echo "FEATURE MATRIX: at least one combination failed to compile." >&2
    exit 1
fi

echo
echo "FEATURE MATRIX: all 8 combinations (4 features × 2 crates) compile cleanly."
