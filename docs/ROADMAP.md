# Solow — Roadmap

The complete statistics and machine learning stack for Rust. 57
focused crates, memory safe, pure Rust, deterministic.

## Status: shipping

Every module in the initial scope ships at the workspace HEAD.
`cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets
--all-features -- -D warnings`, and `cargo test --workspace` all pass
with 1000+ tests green.

The umbrella `solow::prelude` re-exports the workhorse estimators, the
formula-driven fit surface, and the everyday metrics and tests, so a
canonical statistical workflow can be written against a single
`use solow::prelude::*;` line.

## Verification contract

Every deterministic estimator is cross-verified against committed
golden reference fixtures under `tests/fixtures/`. Closed-form solvers
agree bit-wise to `1e-10`. Iterative solvers agree on parameters to
`1e-6` or on predictions to `5e-2` where reference solvers themselves
disagree at that scale.

NIST StRD certified cases are re-run on every CI, with worst-case
certified relative error across the suite of `2.5e-10`. The
ill-conditioned Longley design (cond ~10^10) matches the certified
coefficients to `~1e-13` because the QR/SVD path never forms `XᵀX`.

Every crate lives under `#![forbid(unsafe_code)]`. Every stochastic
estimator is deterministic given a seed via a portable MMIX-LCG PRNG.

## Module map

### Foundation

`solow-core`, `solow-linalg`, `solow-distributions`, `solow-optimize`.

### Linear and generalized linear models

`solow-regression`, `solow-glm`, `solow-discrete`, `solow-robust`,
`solow-othermod`.

### Panel, survival, hierarchical

`solow-mixed`, `solow-gee`, `solow-gam`, `solow-duration`.

### Time series and state space

`solow-tsa`, `solow-statespace`, `solow-var`, `solow-regime`.

### Multivariate and covariance

`solow-multivariate`, `solow-cross-decomposition`, `solow-covariance`.

### Machine learning

`solow-cluster`, `solow-tree`, `solow-ensemble`, `solow-svm`,
`solow-neural`, `solow-naive-bayes`, `solow-discriminant`,
`solow-neighbors`.

### Dimensionality reduction

`solow-decomposition`, `solow-manifold`, `solow-kernel-approx`.

### Semi-supervised, multi, probability

`solow-semi-supervised`, `solow-multi`, `solow-calibration`, `solow-gp`.

### Model selection and metrics

`solow-cv`, `solow-metrics`, `solow-pipeline`, `solow-feature-selection`.

### Preprocessing and data

`solow-preprocessing`, `solow-text`, `solow-impute`, `solow-datasets`.

### Bayesian, empirical likelihood, formulas

`solow-bayes`, `solow-emplike`, `solow-copula`, `solow-formula`,
`solow-fit`.

### Nonparametric and other statistics

`solow-stats`, `solow-nonparametric`.

### Presentation and visualization

`solow-summary`, `solow-viz`, `solow-graphics`.

### Umbrella and bindings

`solow`, `solow-py`, `solow-polars`.
