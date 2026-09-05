#!/usr/bin/env bash
# Dependency-ordered publish for every solow crate.
#
# Usage:
#     bash scripts/publish.sh               # dry-run every crate (no network)
#     bash scripts/publish.sh --publish     # actually publish
#     bash scripts/publish.sh --publish --resume-from solow-glm
#         (skip ahead — useful if a run failed mid-way and earlier crates
#         already made it to crates.io)
#
# Requirements:
#   * `cargo login <token>` was already run (crates.io API token).
#   * The workspace version at `Cargo.toml [workspace.package] version`
#     is not already published — crates.io refuses re-publishing the
#     same (name, version) pair.
#
# Between crates the script waits ~30 s to let crates.io's index catch
# up before the next crate's `cargo publish` resolves the just-published
# dependency. Longer waits are almost never needed.

set -euo pipefail

# Leaf → root topological order across the workspace, computed by Kahn's
# algorithm over the actual Cargo.toml `[dependencies]` graph (see
# scripts/verify_publish_order.py). Any change to the workspace's crate
# dependencies must be reflected here — a wrong order surfaces on publish
# as `failed to select a version for solow-X = "^0.2.0"` because a
# dependency was not yet uploaded.
# `publish = false` crates (solow-bench, solow-gallery, solow-polars,
# solow-py) are intentionally omitted.
ORDER=(
    solow-core
    solow-calibration
    solow-cluster
    solow-covariance
    solow-cross-decomposition
    solow-cv
    solow-datasets
    solow-discriminant
    solow-distributions
    solow-feature-selection
    solow-formula
    solow-gp
    solow-kernel-approx
    solow-linalg
    solow-metrics
    solow-multi
    solow-naive-bayes
    solow-neighbors
    solow-neural
    solow-preprocessing
    solow-semi-supervised
    solow-svm
    solow-text
    solow-tree
    solow-viz

    solow-copula
    solow-ensemble
    solow-glm
    solow-manifold
    solow-multivariate
    solow-nonparametric
    solow-optimize
    solow-pipeline
    solow-regression
    solow-summary

    solow-bayes
    solow-decomposition
    solow-discrete
    solow-duration
    solow-emplike
    solow-gam
    solow-gee
    solow-graphics
    solow-impute
    solow-mixed
    solow-othermod
    solow-regime
    solow-robust
    solow-statespace
    solow-stats
    solow-tsa
    solow-var

    solow-fit

    solow
)

MODE="dry-run"
RESUME_FROM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --publish) MODE="publish"; shift ;;
        --resume-from) RESUME_FROM="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

SKIPPING=""
if [[ -n "$RESUME_FROM" ]]; then
    SKIPPING="yes"
fi

publish_one_with_retry() {
    local crate="$1"
    local attempts=0
    while true; do
        local out
        out=$(cargo publish -p "$crate" 2>&1)
        local status=$?
        printf '%s\n' "$out"
        if [[ $status -eq 0 ]]; then
            return 0
        fi
        # Detect crates.io "new crate" rate-limit response.
        if echo "$out" | grep -q "Too Many Requests"; then
            local retry_at
            retry_at=$(echo "$out" | grep -oE "try again after [^ ]+ [^ ]+ [^ ]+ [^ ]+" | head -1)
            attempts=$((attempts + 1))
            if [[ $attempts -gt 8 ]]; then
                echo "     rate-limit persistent after 8 tries; giving up"
                return 1
            fi
            local wait_secs=630  # 10 min + 30 s slop
            echo "     hit crates.io rate-limit ($retry_at); sleeping ${wait_secs}s then retrying $crate"
            sleep "$wait_secs"
            continue
        fi
        # Any other error: fail fast.
        return $status
    done
}

for c in "${ORDER[@]}"; do
    if [[ -n "$SKIPPING" ]]; then
        if [[ "$c" == "$RESUME_FROM" ]]; then
            SKIPPING=""
        else
            echo "skip  $c (resume-from = $RESUME_FROM)"
            continue
        fi
    fi

    echo ""
    echo "==> $c"
    if [[ "$MODE" == "publish" ]]; then
        publish_one_with_retry "$c" || exit $?
        echo "     sleeping 30 s to let the index catch up..."
        sleep 30
    else
        cargo publish -p "$c" --dry-run
    fi
done

echo ""
if [[ "$MODE" == "publish" ]]; then
    echo "All crates published. docs.rs will build within ~10 minutes;"
    echo "lib.rs mirrors crates.io within a few hours."
else
    echo "Dry-run finished. Re-run with --publish to actually upload."
fi
